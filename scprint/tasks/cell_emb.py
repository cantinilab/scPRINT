import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import torch

from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from scprint.model import utils
import bionty as bt
from torch.utils.data import DataLoader

import pandas as pd

from lightning.pytorch import Trainer

from scipy.stats import spearmanr

from typing import List
from anndata import AnnData


class Embedder:
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 64,
        num_workers: int = 8,
        how: str = "random expr",
        max_len: int = 2000,
        add_zero_genes: int = 0,
        precision: str = "16-mixed",
        organisms: List[str] = [
            "NCBITaxon:9606",
        ],
        pred_embedding: List[str] = [
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        model_name: str = "scprint",
        output_expression: str = "sample",  # one of "all" "sample" "none"
        plot_corr_size: int = 64,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            model (torch.nn.Module): The model to be used for embedding and annotating cells.
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            how (str, optional): The method to be used for selecting valid genes. Defaults to "most expr".
            max_len (int, optional): The maximum length of the gene sequence. Defaults to 1000.
            add_zero_genes (int, optional): The number of zero genes to add to the gene sequence. Defaults to 100.
            precision (str, optional): The precision to be used in the Trainer. Defaults to "16-mixed".
            organisms (List[str], optional): The list of organisms to be considered. Defaults to [ "NCBITaxon:9606", ].
            pred_embedding (List[str], optional): The list of labels to be used for plotting embeddings. Defaults to [ "cell_type_ontology_term_id", "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id", "sex_ontology_term_id", ].
            model_name (str, optional): The name of the model to be used. Defaults to "scprint".
            output_expression (str, optional): The type of output expression to be used. Can be one of "all", "sample", "none". Defaults to "sample".
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.add_zero_genes = add_zero_genes
        self.organisms = organisms
        self.model.pred_embedding = pred_embedding
        self.model_name = model_name
        self.output_expression = output_expression
        self.plot_corr_size = plot_corr_size
        self.precision = precision
        self.trainer = Trainer(precision=precision)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, adata: AnnData):
        # Add at least the organism you are working with
        adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
        col = Collator(
            organisms=self.organisms,
            valid_genes=self.model.genes,
            how=self.how,
            max_len=self.max_len,
            add_zero_genes=self.add_zero_genes,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.trainer.predict(self.model, dataloader)
        try:
            mdir = (
                self.model.logger.save_dir
                if self.model.logger.save_dir is not None
                else "/tmp"
            )
        except:
            mdir = "/tmp"

        pred_adata = sc.read_h5ad(
            mdir + "/step_" + str(self.model.global_step) + "_" + ".h5ad"
        )
        if self.output_expression == "all":
            adata.obsm["scprint_mu"] = self.model.expr_pred[0]
            adata.obsm["scprint_theta"] = self.model.expr_pred[1]
            adata.obsm["scprint_pi"] = self.model.expr_pred[2]
        elif self.output_expression == "sample":
            adata.obsm["scprint_expr"] = utils.zinb_sample(
                self.model.expr_pred[0],
                self.model.expr_pred[1],
                self.model.expr_pred[2],
            )
        elif self.output_expression == "old":
            expr = np.array(self.model.expr_pred[0])
            expr[
                np.random.binomial(
                    1,
                    p=np.array(
                        torch.nn.functional.sigmoid(
                            self.model.expr_pred[2].to(torch.float32)
                        )
                    ),
                ).astype(bool)
            ] = 0
            expr[expr <= 0.3] = 0
            expr[(expr >= 0.3) & (expr <= 1)] = 1
            adata.obsm["scprint_expr"] = expr.astype(int)
        adata.obsm["scprint_pos"] = self.model.pos
        pred_adata.obs.index = adata.obs.index
        adata.obsm["scprint_umap"] = pred_adata.obsm["X_umap"]
        adata.obsm[self.model_name] = pred_adata.X
        pred_adata.obs.index = adata.obs.index
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)
        print("metrics for this dataset of size:", adata.shape[0])
        for label in self.model.labels:
            res = []
            if label not in adata.obs.columns:
                continue
            class_topred = self.model.label_decoders[label].values()
            if label in self.model.cls_hierarchy:
                class_groupings = {
                    k: [
                        i.ontology_id
                        for i in bt.CellType.filter(k).first().children.all()
                    ]
                    for k in set(adata.obs[label].unique()) - set(class_topred)
                }
            for pred, true in adata.obs[["pred_" + label, label]].values:
                if pred == true:
                    res.append(True)
                    continue

                if label in self.model.cls_hierarchy:
                    if true in class_groupings:
                        res.append(pred in class_groupings[true])
                        continue
                    elif true not in class_topred:
                        raise ValueError(f"true label {true} not in available classes")
                elif true not in class_topred:
                    raise ValueError(f"true label {true} not in available classes")
                res.append(False)
            print("    ", label)
            print("     accuracy:", sum(res) / len(res))
            print(" ")

        # Compute correlation coefficient
        if self.plot_corr_size > 0:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.max_len * 2, flavor="seurat_v3"
            )
            highly_variable = adata.var.index[adata.var.highly_variable].tolist()
            random_indices = np.random.randint(
                low=0, high=adata.shape[0], size=self.plot_corr_size
            )
            adataset = SimpleAnnDataset(
                adata[random_indices], obs_to_output=["organism_ontology_term_id"]
            )
            col = Collator(
                organisms=self.organisms,
                valid_genes=self.model.genes,
                how="some",
                genelist=highly_variable,
            )
            dataloader = DataLoader(
                adataset,
                collate_fn=col,
                batch_size=self.plot_corr_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            # self.trainer.num_predict_batches = 1
            self.trainer.predict(self.model, dataloader)

            res = self.model.expr_pred
            # pos = adata.obsm["scprint_pos"][random_indices]
            out = utils.zinb_sample(
                res[0],
                res[1],
                res[2],
            ).numpy()
            for i in dataloader:
                break
            try:
                mean_expr = pd.read_parquet("../../data/avg_expr.parquet")
                genes_used = [self.model.genes[int(i)] for i in self.model.pos[0]]
                mean_expr = mean_expr[mean_expr.index.isin(genes_used)][
                    ["avg_expr", "avg_expr_wexpr"]
                ].values
                out = np.hstack([out.T, mean_expr])
            except:
                print(
                    "cannot read the mean expr file under scprint/data/avg_expr.parquet"
                )
                out = out.T
                mean_expr = None

            corr_coef, p_value = spearmanr(
                out,
                i["x"].T,
            )
            corr_coef[p_value > 0.05] = 0
            # corr_coef[]
            # only on non zero values,
            # compare a1-b1 corr with a1-b(n) corr. should be higher

            # Plot correlation coefficient
            plt.figure(figsize=(10, 5))
            plt.imshow(
                corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
            )
            plt.colorbar()
            plt.title('Correlation Coefficient of expr and i["x"]')
            plt.show()
            expr = np.array(res[0])
            expr[
                np.random.binomial(
                    1,
                    p=np.array(torch.nn.functional.sigmoid(res[2].to(torch.float32))),
                ).astype(bool)
            ] = 0
            corr_coef, p_value = spearmanr(
                np.hstack([expr.T, mean_expr]) if mean_expr is not None else expr.T,
                i["x"].T,
            )
            corr_coef[p_value > 0.05] = 0
            # corr_coef[]
            # only on non zero values,
            # compare a1-b1 corr with a1-b(n) corr. should be higher

            # Plot correlation coefficient
            plt.figure(figsize=(10, 5))
            plt.imshow(
                corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
            )
            plt.colorbar()
            plt.title('Correlation Coefficient of expr and i["x"]')
            plt.show()
        return adata