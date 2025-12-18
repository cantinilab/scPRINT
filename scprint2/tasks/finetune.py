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

from scprint2.model import loss

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class FinetuneBatchClass:
    def __init__(
        self,
        batch_key: str = "batch",
        predict_keys: List[str] = ["cell_type_ontology_term_id"],
        max_len: int = 5000,
        learn_batches_on: Optional[str] = None,
        num_workers: int = 8,
        batch_size: int = 16,
        num_epochs: int = 8,
        do_mmd_on: Optional[str] = None,
        lr: float = 0.0002,
        ft_mode: str = "xpressor",
        frac_train: float = 0.8,
        loss_scalers: dict = {},
        use_knn: bool = True,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            batch_key (str, optional): The key in adata.obs that indicates the batch information. Defaults to "batch".
            learn_batches_on (str, optional): The key in adata.obs to learn batch embeddings on. Defaults to None.
                if none, will not learn the batch embeddings.
                the goal is e.g. when having a new species, to learn an embedding for it during finetuning and replace
                the "learn_batches_on" embedding in the model with it, in this case it should be "organism_ontology_term_id".
                batch correction might indeed be better learnt with this additional argument in some cases.
            do_mmd_on (str, optional):The key in adata.obs to learn batch embeddings on. Defaults to None.
                this embedding should have less batch information in it, after finetuning.
            predict_keys (List[str], optional): List of keys in adata.obs to predict during fine-tuning. Defaults to ["cell_type_ontology_term_id"].
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            max_len (int, optional): The maximum length of the sequences to be processed. Defaults to 5000.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.0002.
            num_epochs (int, optional): The number of epochs to train the model. Defaults to 8.
            ft_mode (str, optional): The fine-tuning mode, either "xpressor" or "full". Defaults to "xpressor".
            frac_train (float, optional): The fraction of data to be used for training. Defaults to 0.8.
            loss_scalers (dict, optional): A dictionary specifying the scaling factors for different loss components. Defaults to {}.
                expr, class, mmd, kl, and any of the predict_keys can be specified.
            use_knn (bool, optional): Whether to use k-nearest neighbors information. Defaults to True.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_key = batch_key
        self.learn_batches_on = learn_batches_on
        self.predict_keys = predict_keys
        self.max_len = max_len
        self.lr = lr
        self.num_epochs = num_epochs
        self.ft_mode = ft_mode
        self.frac_train = frac_train
        self.batch_emb = None
        self.batch_encoder = {}
        self.do_mmd_on = do_mmd_on
        self.loss_scalers = loss_scalers
        self.use_knn = use_knn

    def __call__(
        self,
        model: torch.nn.Module,
        adata: AnnData = None,
        train_data: AnnData = None,
        val_data: AnnData = None,
    ) -> torch.nn.Module:
        """
        __call__ function to call the embedding

        Args:
            model (torch.nn.Module): The scPRINT model to be used for embedding and annotation.
            adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
                Defaults to None.
                if provided, it will be split into training and validation sets.
            train_data (AnnData, optional): The training data. Defaults to None.
                if adata is provided, this will be ignored.
            val_data (AnnData, optional): The validation data. Defaults to None.
                if adata is provided, this will be ignored.

        Raises:
            ValueError: If the model does not have a logger attribute.
            ValueError: If the model does not have a global_step attribute.

        Returns:
            torch.nn.Module: the fine-tuned model
        """
        # one of "all" "sample" "none"
        model.predict_mode = "none"
        if self.ft_mode == "xpressor":
            for val in model.parameters():
                val.requires_grad = False
                # setting all to TRUE

            for val in model.cell_transformer.parameters():
                val.requires_grad = True
            for val in model.transformer.blocks[-1].parameters():
                val.requires_grad = True
            for i in model.transformer.blocks:
                i.cross_attn.requires_grad = True
            for val in model.compressor.parameters():
                val.requires_grad = True
            for val in self.predict_keys:
                for val in model.cls_decoders[val].parameters():
                    val.requires_grad = True
        elif self.ft_mode == "full":
            for val in model.parameters():
                val.requires_grad = True
        else:
            raise ValueError("ft_mode must be one of 'xpressor' or 'full'")

        # PREPARING THE DATA
        if adata is not None:
            n_train = int(self.frac_train * len(adata))
            train_idx = np.random.choice(len(adata), n_train, replace=False)
            val_idx = np.setdiff1d(np.arange(len(adata)), train_idx)

            train_data = adata[train_idx].copy()
            val_data = adata[val_idx].copy()

            print(f"Training data: {train_data.shape}")
            print(f"Validation data: {val_data.shape}")

        mencoders = {}
        for k, v in model.label_decoders.items():
            mencoders[k] = {va: ke for ke, va in v.items()}
        # this needs to remain its original name as it is expect like that by collator, otherwise need to send org_to_id as params

        for i in self.predict_keys:
            if len(set(train_data.obs[i]) - set(mencoders[i].keys())) > 0:
                print("missing labels for ", i)
                train_data.obs[i] = train_data.obs[i].apply(
                    lambda x: x if x in mencoders[i] else "unknown"
                )
        if "organism_ontology_term_id" not in self.predict_keys:
            self.predict_keys.append("organism_ontology_term_id")
        # create datasets
        self.batch_encoder = {
            i: n
            for n, i in enumerate(
                train_data.obs[self.batch_key].astype("category").cat.categories
            )
        }
        mencoders[self.batch_key] = self.batch_encoder
        train_dataset = SimpleAnnDataset(
            train_data,
            obs_to_output=self.predict_keys + [self.batch_key],
            get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
            encoder=mencoders,
        )
        if val_data is not None:
            for i in self.predict_keys:
                if i != "organism_ontology_term_id":
                    if len(set(val_data.obs[i]) - set(mencoders[i].keys())) > 0:
                        val_data.obs[i] = val_data.obs[i].apply(
                            lambda x: x if x in mencoders[i] else "unknown"
                        )
            self.batch_encoder.update(
                {
                    i: n + len(self.batch_encoder)
                    for n, i in enumerate(
                        val_data.obs[self.batch_key].astype("category").cat.categories
                    )
                    if i not in self.batch_encoder
                }
            )
            mencoders[self.batch_key] = self.batch_encoder
            val_dataset = SimpleAnnDataset(
                val_data,
                obs_to_output=self.predict_keys + [self.batch_key],
                get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
                encoder=mencoders,
            )

        # Create collator
        collator = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            class_names=self.predict_keys + [self.batch_key],
            how="random expr",  # or "all expr" for full expression
            max_len=self.max_len,
            org_to_id=mencoders.get("organism_ontology_term_id", {}),
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            collate_fn=collator,
            batch_size=self.batch_size,  # Adjust based on GPU memory
            num_workers=self.num_workers,
            shuffle=True,
        )
        if val_data is not None:
            val_loader = DataLoader(
                val_dataset,
                collate_fn=collator,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

        if self.learn_batches_on is not None:
            if val_data is not None:
                print(
                    "all batch key values in val_data should also be present in train_adata!!!"
                )
            self.batch_emb = torch.nn.Embedding(
                num_embeddings=train_data.obs[self.batch_key].nunique(),
                embedding_dim=(
                    model.compressor[self.learn_batches_on].fc_mu.weight.shape[0]
                    if hasattr(model, "compressor")
                    else model.d_model
                ),
            )

        ## PREPARING THE OPTIM
        all_params = (
            list(model.parameters())
            # + list(batch_cls.parameters())
            + (
                list(self.batch_emb.parameters())
                if self.learn_batches_on is not None
                else []
            )
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            all_params,
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

        for k, i in model.mat_labels_hierarchy.items():
            model.mat_labels_hierarchy[k] = i.to(model.device)

        ## train
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            # Training phase
            train_loss = 0.0
            train_steps = 0
            avg_expr = 0
            avg_cls = 0
            avg_mmd = 0

            pbar = tqdm(train_loader, desc="Training")
            model.train()
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                total_loss, cls_loss, mmd, loss_expr = self.batch_corr_pass(
                    batch, model
                )
                # Backward pass
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += total_loss.item()
                train_steps += 1
                avg_cls += cls_loss.item()
                avg_expr += loss_expr.item()
                avg_mmd += mmd
                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "avg_loss": f"{train_loss / train_steps:.4f}",
                        "cls_loss": f"{cls_loss.item():.4f}",
                        "mmd_loss": f"{mmd:.4f}",
                        "expr_loss": f"{loss_expr.item():.4f}",
                    }
                )

            # Validation phase
            if val_data is not None:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                val_loss_expr = 0.0
                val_mmd = 0.0
                val_cls = 0.0
                val_loss_to_prt = 0.0

                with torch.no_grad():
                    for batch in val_loader:  # tqdm(val_loader, desc="Validation"):
                        loss_val, cls_loss, mmd, loss_expr = self.batch_corr_pass(
                            batch, model
                        )
                        val_loss_to_prt += loss_val.item()
                        val_loss += loss_val.item()
                        val_steps += 1
                        val_loss_expr += loss_expr.item()
                        val_mmd += mmd
                        val_cls += cls_loss.item()
                try:
                    avg_val_loss = val_loss_to_prt / val_steps
                    avg_train_loss = train_loss / train_steps
                except ZeroDivisionError:
                    print(
                        "Error: Division by zero occurred while calculating average losses."
                    )
                    avg_train_loss = 0
                print(
                    "cls_loss: {:.4f}, mmd_loss: {:.4f}, expr_loss: {:.4f}".format(
                        val_cls / val_steps,
                        val_mmd / val_steps,
                        val_loss_expr / val_steps,
                    )
                )
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Store LR before scheduler step for comparison
                lr_before = optimizer.param_groups[0]["lr"]

                # Update learning rate
                scheduler.step(avg_val_loss)

                # Check if LR was reduced
                lr_after = optimizer.param_groups[0]["lr"]
                if lr_after < lr_before:
                    print(
                        f"🔻 Learning rate reduced from {lr_before:.2e} to {lr_after:.2e} (factor: {lr_after / lr_before:.3f})"
                    )
                else:
                    print(f"✅ Learning rate unchanged: {lr_after:.2e}")

                # Early stopping check (simple implementation)
                if epoch > 3 and val_loss / val_steps > 1.3 * avg_train_loss:
                    print("Early stopping due to overfitting")
                    break

        print("Manual fine-tuning completed!")
        model.eval()
        return model

    def batch_corr_pass(self, batch, model):
        gene_pos = batch["genes"].to(model.device)
        expression = batch["x"].to(model.device)
        depth = batch["depth"].to(model.device)
        class_elem = batch["class"].long().to(model.device)
        total_loss = 0
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
            ## adaptor on ct_emb
            # ctpos = model.classes.index("cell_type_ontology_term_id") + 1
            # emb = output["output_cell_embs"][:, ctpos, :]
            #
            # output["output_cell_embs"][:, ctpos, :] = adaptor_layer(
            #    torch.cat([emb, class_elem[:, 1].unsqueeze(1).float()], dim=1)
            # )
            if self.learn_batches_on is not None:
                batch_pos = model.classes.index(self.learn_batches_on) + 1
                output["output_cell_embs"][:, batch_pos, :] = self.batch_emb(
                    class_elem[:, -1]
                )

            ## generate expr loss
            output_gen = model._generate(
                cell_embs=output["output_cell_embs"],
                gene_pos=gene_pos,
                depth_mult=expression.sum(1),
                req_depth=depth,
            )
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
            total_loss += loss_expr * self.loss_scalers.get("expr", 0.5)

            # ct
            cls_loss = 0
            for clas in self.predict_keys:
                cls_output = output.get("cls_output_" + clas)
                # ct_output = output["output_cell_embs"][:, ctpos, :]
                # cls_output = model.cls_decoders["cell_type_ontology_term_id"](ct_output)
                cls_loss += loss.hierarchical_classification(
                    pred=cls_output,
                    cl=class_elem[:, self.predict_keys.index(clas)],
                    labels_hierarchy=(
                        model.mat_labels_hierarchy.get(clas).to("cuda")
                        if clas in model.mat_labels_hierarchy
                        else None
                    ),
                ) * self.loss_scalers.get(clas, 1)

            # organ class
            # org_emb = output["compressed_cell_embs"][
            #    model.classes.index("organism_ontology_term_id") + 1
            # ]
            # cls_loss += F.cross_entropy(
            #    input=batch_cls(org_emb),
            #    target=class_elem[:, 1],
            # )
            total_loss += cls_loss * self.loss_scalers.get("class", 1)
            tot_mmd = 0
            if self.do_mmd_on is not None:
                pos = model.classes.index(self.do_mmd_on) + 1
                # Apply gradient reversal to the input embedding
                selected_emb = (
                    output["compressed_cell_embs"][pos]
                    if model.compressor is not None
                    else output["input_cell_embs"][:, pos, :]
                )
                for i in set(class_elem[:, -1].cpu().numpy()):
                    if (class_elem[:, -1] == i).sum() < 2:
                        # need at least 2 samples to compute mmd
                        class_elem[class_elem[:, -1] == i, 1] = (
                            -1
                        )  # assign to dummy class
                    # compare each batch to all other batches
                for i in set(class_elem[:, -1].cpu().numpy()):
                    if i == -1:
                        continue
                    X, Y = (
                        selected_emb[class_elem[:, -1] == i],
                        selected_emb[class_elem[:, -1] != i],
                    )
                    mmd = mmd_loss(X, Y)
                    if torch.isnan(mmd):
                        print("mmd nan")
                    tot_mmd += mmd.item() if not torch.isnan(mmd) else 0
                # Add adversarial loss to total loss
                total_loss += tot_mmd * self.loss_scalers.get("mmd", 3)
            if "vae_kl_loss" in output:
                total_loss += output["vae_kl_loss"] * self.loss_scalers.get("kl", 0.5)
        return total_loss, cls_loss, tot_mmd, loss_expr


def mmd_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) loss between two 2D embedding matrices.

    Args:
        X (torch.Tensor): Tensor of shape (n1, emb_dim) - first set of embeddings
        Y (torch.Tensor): Tensor of shape (n2, emb_dim) - second set of embeddings

    Returns:
        torch.Tensor: MMD loss value (negative to encourage dissimilarity)
    """

    def rbf_kernel(x, y, sigma):
        """Compute RBF kernel between two sets of vectors"""
        distance = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-distance / (2 * sigma**2))

    def energy_kernel(x, y):
        """Compute Energy kernel between two sets of vectors"""
        distance = torch.cdist(x, y, p=2)
        return -distance

    # Use multiple kernel bandwidths for better performance
    sigmas = [0]  # [0.1, 1.0, 10.0]
    mmd_loss = 0.0

    for sigma in sigmas:
        # K(X, X) - kernel matrix within first group (n1 x n1)
        # k_xx = rbf_kernel(X, X, sigma)
        k_xx = energy_kernel(X, X)
        # K(Y, Y) - kernel matrix within second group (n2 x n2)
        # k_yy = rbf_kernel(Y, Y, sigma)
        k_yy = energy_kernel(Y, Y)
        # K(X, Y) - kernel matrix between groups (n1 x n2)
        # k_xy = rbf_kernel(X, Y, sigma)
        k_xy = energy_kernel(X, Y)

        # Unbiased MMD estimation
        n1 = X.shape[0]
        n2 = Y.shape[0]

        # Remove diagonal elements for unbiased estimation of K(X,X) and K(Y,Y)
        # For K(X,X): exclude diagonal
        if n1 > 1:
            mask_xx = 1 - torch.eye(n1, device=X.device)
            k_xx_term = (k_xx * mask_xx).sum() / (n1 * (n1 - 1))
        else:
            k_xx_term = 0.0

        # For K(Y,Y): exclude diagonal
        if n2 > 1:
            mask_yy = 1 - torch.eye(n2, device=Y.device)
            k_yy_term = (k_yy * mask_yy).sum() / (n2 * (n2 - 1))
        else:
            k_yy_term = 0.0

        # For K(X,Y): use all elements (no diagonal to exclude)
        k_xy_term = k_xy.mean()

        # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        mmd_squared = k_xx_term + k_yy_term - 2 * k_xy_term
        mmd_loss += mmd_squared

    # Return negative MMD to encourage dissimilarity (higher MMD = more different)
    return mmd_loss / len(sigmas)


class FinetuneGRN:
    pass


class FinetuneGeneEmb:
    pass


class FinetuneNewClass:
    pass


class FinetuneUpdateClass:
    pass
