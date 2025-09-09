import os
from typing import Any, Dict, List, Optional

import bionty as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn.functional as F
from anndata import AnnData, concat
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from scdataloader.utils import get_descendants, random_str
from scib_metrics.benchmark import Benchmarker
from scipy.stats import spearmanr
from simpler_flash import FlashTransformer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint.model import loss

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class Finetune:
    def __init__(
        self,
        batch_key: str = "batch",
        max_len: int = 5000,
        predict_key: str = "cell_type_ontology_term_id",
        predictor: Optional[torch.nn.Module] = None,
        emb_to_predict_on: str = "cell_type_ontology_term_id",
        num_workers: int = 8,
        batch_size: int = 16,
        embedding_to_use: List[str] = [
            "all",
        ],
        doplot: bool = True,
        num_epochs: int = 10,
        lr: float = 0.0001,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            embedding_to_use (List[str], optional): The list of embeddings to be used for generating expression. Defaults to [ "all" ].
            doplot (bool, optional): Whether to generate plots. Defaults to True.
            genelist (List[str]): The list of genes for which to generate expression data
            save_every (int, optional): The number of cells to save at a time. Defaults to 40_000.
                This is important to avoid memory issues.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embedding_to_use = embedding_to_use
        self.doplot = doplot
        self.predictor = predictor
        self.max_len = max_len
        self.predict_key = predict_key
        self.lr = lr
        self.num_epochs = num_epochs

    def __call__(self, model: torch.nn.Module, adata: AnnData):
        """
        __call__ function to call the embedding

        Args:
            model (torch.nn.Module): The scPRINT model to be used for embedding and annotation.
            adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.

        Raises:
            ValueError: If the model does not have a logger attribute.
            ValueError: If the model does not have a global_step attribute.

        Returns:
            AnnData: The annotated data matrix with embedded cell representations.
            List[str]: List of gene names used in the embedding.
            np.ndarray: The predicted expression values if sample"none".
            dict: Additional metrics and information from the embedding process.
        """
        # one of "all" "sample" "none"
        model.predict_mode = "none"

        # PREPARING THE DATA
        n_train = int(0.8 * len(da))
        train_idx = np.random.choice(len(da), n_train, replace=False)
        val_idx = np.setdiff1d(np.arange(len(da)), train_idx)

        train_data = da[train_idx].copy()
        val_data = da[val_idx].copy()

        print(f"Training data: {train_data.shape}")
        print(f"Validation data: {val_data.shape}")

        mencoders = {}
        for k, v in model.label_decoders.items():
            mencoders[k] = {va: ke for ke, va in v.items()}
        # this needs to remain its original name as it is expect like that by collator, otherwise need to send org_to_id as params
        mencoders.pop("organism_ontology_term_id")

        # Create datasets
        train_dataset = SimpleAnnDataset(
            train_data,
            obs_to_output=[
                "cell_type_ontology_term_id",
                "batch",
                "organism_ontology_term_id",
            ],
            get_knn_cells=model.expr_emb_style == "metacell",
            encoder=mencoders,
        )

        val_dataset = SimpleAnnDataset(
            val_data,
            obs_to_output=[
                "cell_type_ontology_term_id",
                "batch",
                "organism_ontology_term_id",
            ],
            get_knn_cells=model.expr_emb_style == "metacell",
            encoder=mencoders,
        )

        # Create collator
        collator = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            class_names=["cell_type_ontology_term_id", "batch"],
            how="random expr",  # or "all expr" for full expression
            max_len=3000,
            add_zero_genes=0,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            collate_fn=collator,
            batch_size=32,  # Adjust based on GPU memory
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            collate_fn=collator,
            batch_size=32,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

        ## PREPARING THE OPTIM

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        # Setup automatic mixed precision
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        _ = model.train()

        for k, i in model.mat_labels_hierarchy.items():
            model.mat_labels_hierarchy[k] = i.to(model.device)

        ##
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            train_loss = 0.0
            train_steps = 0
            avg_adv = 0
            avg_expr = 0
            avg_cls = 0

            pbar = tqdm(train_loader, desc="Training")
            for batch_idx, batch in enumerate(pbar):
                # if epoch == 0:
                #    break
                # Move batch to device
                optimizer.zero_grad()
                total_loss, cls_loss, current_adv_loss, loss_expr = batch_corr_pass(
                    batch, model, self.predictor
                )
                # Backward pass
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += total_loss.item() if not torch.isnan(total_loss) else 0
                train_steps += 1
                avg_cls += cls_loss.item() if not torch.isnan(cls_loss) else 0
                avg_expr += loss_expr.item() if not torch.isnan(loss_expr) else 0
                avg_adv += (
                    current_adv_loss.item() if not torch.isnan(current_adv_loss) else 0
                )
                # Update progress bar
                # if batch_idx % 35 == 0:
                # print(
                #    f"avg_loss {train_loss / train_steps:.4f}, avg_cls {avg_cls / train_steps:.4f}, avg_expr {avg_expr / train_steps:.4f}, avg_adv {avg_adv / train_steps:.4f}"
                # )
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "avg_loss": f"{train_loss / train_steps:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "cls_loss": f"{cls_loss.item():.4f}",
                        "adv_loss": f"{current_adv_loss.item():.4f}",
                        "expr_loss": f"{loss_expr.item():.4f}",
                    }
                )

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_steps = 0
            val_loss_to_prt = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    loss_val, cls_loss, current_adv_loss, loss_expr = fullpass(batch)
                    val_loss_to_prt += loss_val.item() - (2 * current_adv_loss.item())
                    val_loss += loss_val.item()
                    val_steps += 1
            try:
                avg_val_loss = val_loss_to_prt / val_steps
                avg_train_loss = train_loss / train_steps
            except ZeroDivisionError:
                print(
                    "Error: Division by zero occurred while calculating average losses."
                )
                avg_train_loss = 0
            print(
                "cls_loss: {:.4f}, adv_loss: {:.4f}, expr_loss: {:.4f}".format(
                    cls_loss.item(), current_adv_loss.item(), loss_expr.item()
                )
            )
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Early stopping check (simple implementation)
            if epoch > 5 and val_loss / val_steps > 1.3 * avg_train_loss:
                print("Early stopping due to overfitting")
                break

        print("Manual fine-tuning completed!")
        return model


def batch_corr_pass(batch, model, batch_cls):
    gene_pos = batch["genes"].to(model.device)
    expression = batch["x"].to(model.device)
    depth = batch["depth"].to(model.device)

    # Forward pass with automatic mixed precisio^n
    with torch.cuda.amp.autocast():
        # Forward pass
        output = model.forward(
            gene_pos,
            expression,
            req_depth=depth,
            depth_mult=expression.sum(1),
            do_class=True,
            metacell_token=torch.zeros_like(depth),
        )

        # output["output_cell_embs"][
        #    :, model.classes.index("organism_ontology_term_id") + 1, :
        # ] = batch_vector.unsqueeze(0).repeat(batch["depth"].shape[0], 1)
        output_gen = model._generate(
            cell_embs=output["output_cell_embs"],
            gene_pos=gene_pos,
            depth_mult=expression.sum(1),
            req_depth=depth,
        )
        # model.qkv # use it to fine tune on the gene interactions
        # predict something like known PPI matrices, cell specific GRNs from atac-seq data

        # model.gene_output_embeddings
        # use it to train a classifier on top to predict other modalities from gene embeddings given an additional anndata
        # could be protein expression, ATAC-seq gene activity, transcript dynamics

        # model.gen_output["expression"] # modify the loss so that the model learns to predict KO given additional gene + with learnt KO representation token
        # or expression temporal change given learnt temporal token

        # for batch correction and classification
        # Compute losses
        total_loss = 0

        if "zero_logits" in output_gen:
            loss_expr = loss.zinb(
                theta=output_gen["disp"],
                pi=output_gen["zero_logits"],
                mu=output_gen["mean"],
                target=expression,
            )
            if model.zinb_and_mse:
                loss_expr += (
                    loss.mse(
                        input=torch.log(output_gen["mean"] + 1)
                        * (1 - torch.sigmoid(output_gen["zero_logits"])),
                        target=torch.log(expression + 1),
                    )
                    / 10  # scale to make it more similar to the zinb
                )
        else:
            loss_expr = loss.mse(
                input=torch.log(output_gen["mean"] + 1),
                target=torch.log(expression + 1),
            )

        # Add expression loss to total
        total_loss += loss_expr

        # Classification loss
        cls_output = output.get("cls_output_cell_type_ontology_term_id")
        if cls_output is not None:
            cls_loss = loss.hierarchical_classification(
                pred=cls_output,
                cl=batch["class"][:, 0].to(model.device),
                labels_hierarchy=model.mat_labels_hierarchy.get(
                    "cell_type_ontology_term_id"
                ).to("cuda"),
            )
            total_loss += cls_loss

        pos = model.classes.index("cell_type_ontology_term_id") + 1
        # Apply gradient reversal to the input embedding
        selected_emb = (
            output["compressed_cell_embs"][pos]
            if model.compressor is not None
            else output["input_cell_embs"][:, pos, :]
        )
        adv_input_emb = loss.grad_reverse(selected_emb.clone(), lambd=1.0)
        # Get predictions from the adversarial decoder
        adv_pred = batch_cls(adv_input_emb)
        # do dissim

        # Compute the adversarial loss - Fix: Convert target to long type
        current_adv_loss = torch.nn.functional.cross_entropy(
            input=adv_pred,
            target=batch["class"][:, 1].to(model.device).long(),  # Convert to long type
        )

        # Add adversarial loss to total loss
        total_loss += current_adv_loss * 1
    return total_loss, cls_loss, current_adv_loss, loss_expr
