import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scdataloader import Collator
from scdataloader.data import SimpleAnnDataset
from simpler_flash import FlashTransformer
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


class GeneEmbeddingExtractor:
    def __init__(
        self,
        genelist,
        batch_size: int = 64,
        num_workers: int = 8,
        save_every: int = 4_000,
        average: bool = False,
        save_dir: str = None,
        use_knn: bool = False,
    ):
        """
        Args:
            genelist (list): List of genes to restrict to.
            batch_size (int): Batch size for the DataLoader. Defaults to 64.
            num_workers (int): Number of workers for DataLoader. Defaults to 8.
            save_every (int): Save embeddings every `save_every` batches. Defaults to 4000.
            average (bool): Whether to average embeddings across all cells. Defaults to False.
            save_dir (str): Directory to save embeddings. If None, embeddings are not saved. Defaults to None.
            use_knn (bool): Whether to use k-nearest neighbors information. Defaults to False.

        """
        self.genelist = genelist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_every = save_every
        self.average = average
        self.save_dir = save_dir
        self.use_knn = use_knn

    def __call__(self, model, adata):
        model.eval()
        model.pred_log_adata = False
        model.predict_mode = "none"

        # Determine which genes to use
        prevl = len(self.genelist)
        gene_list = [g for g in self.genelist if g in model.genes]
        print(
            "Using {:.2f}% of the genes in the gene list".format(
                len(self.genelist) * 100 / prevl
            )
        )
        if len(gene_list) == 0:
            raise ValueError("No overlap between provided gene_list and model.genes")

        # Set up dataset and dataloader
        # If needed, ensure adata.obs contains 'organism_ontology_term_id' or adapt Collator arguments
        if "organism_ontology_term_id" not in adata.obs:
            # Assign a default organism if needed
            adata.obs["organism_ontology_term_id"] = (
                "NCBITaxon:9606"  # or your relevant organism ID
            )

        adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            max_len=0,
            how="some",
            genelist=gene_list,
            n_bins=model.n_input_bins if model.expr_emb_style == "binned" else 0,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        all_embeddings = []
        expr = []
        model.on_predict_epoch_start()
        count = 0
        full_embeddings = None
        dtype = (
            torch.float16
            if isinstance(model.transformer, FlashTransformer)
            else torch.float32
        )
        # Use autocast to ensure half precision if required by the model
        with (
            torch.no_grad(),
            torch.autocast(device_type=model.device.type, dtype=dtype),
        ):
            for batch in dataloader:
                gene_pos, expression, depth = (
                    batch["genes"].to(model.device),
                    batch["x"].to(model.device),
                    batch["depth"].to(model.device),
                )

                # Run encode_only to get transformer outputs
                output = model(
                    gene_pos=gene_pos,
                    expression=expression,
                    req_depth=depth,
                    depth_mult=expression.sum(1),
                    get_gene_emb=True,
                )
                # transformer_output shape: (B, cell_embs_count + num_genes, d_model)
                # Extract gene embeddings:
                gene_embeddings = output["gene_embedding"]
                # shape: (B, num_genes, d_model)
                if self.average:
                    if full_embeddings is None:
                        # Average the gene embeddings across the batch
                        full_embeddings = gene_embeddings.mean(dim=0, keepdim=True)
                    else:
                        full_embeddings += gene_embeddings.mean(dim=0, keepdim=True)
                    count += 1
                else:
                    all_embeddings.append(gene_embeddings.cpu().numpy())
                expr.append(batch["x"].numpy())
                if (
                    len(all_embeddings) * self.batch_size
                ) >= self.save_every and not self.average:
                    # Save embeddings to disk or process them as needed
                    ad = AnnData(
                        X=np.concatenate(expr, axis=0),
                        var=pd.DataFrame(index=gene_list),
                        varp=np.concatenate(all_embeddings, axis=0),
                    )
                    if self.save_dir is None:
                        print(
                            "reached max len, need to save embeddings but did not specify save_dir"
                        )
                        print("using default save_dir: ./data/")
                        self.save_dir = "./data/"
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    ad.write_h5ad(
                        os.path.join(self.save_dir, f"embeddings_{count}.h5ad")
                    )
                    count += 1
                    del ad
                    all_embeddings = []
                    expr = []
                del output
                torch.cuda.empty_cache()
        ad = AnnData(
            X=np.concatenate(expr, axis=0),
            var=pd.DataFrame(index=gene_list),
            uns=(
                {"all_embeddings": np.concatenate(all_embeddings, axis=0)}
                if not self.average
                else None
            ),
            varm=full_embeddings.cpu().numpy() / count if self.average else None,
        )
        if self.save_dir is not None:
            ad.write_h5ad(os.path.join(self.save_dir, f"embeddings_{count}.h5ad"))
            print("save {} embeddings files under {}".format(count, self.save_dir))
        return ad
