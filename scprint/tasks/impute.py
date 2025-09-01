import os
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import sklearn.metrics
import torch
from anndata import AnnData, concat
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from scdataloader.utils import get_descendants, random_str
from scipy.stats import spearmanr
from simpler_flash import FlashTransformer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint.model import utils
from scprint.tasks.denoise import plot_cell_depth_wise_corr_improvement

from . import knn_smooth

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Imputer:
    def __init__(
        self,
        batch_size: int = 10,
        num_workers: int = 1,
        max_cells: int = 500_000,
        doplot: bool = False,
        predict_depth_mult: int = 4,
        genes_to_use: Optional[List[str]] = None,
        genes_to_impute: Optional[List[str]] = None,
        save_every: int = 100_000,
        additional_info: bool = False,
    ):
        """
        Imputer class for imputing missing values in scRNA-seq data using a scPRINT model

        Args:
            batch_size (int, optional): Batch size for processing. Defaults to 10.
            num_workers (int, optional): Number of workers for data loading. Defaults to 1.
            max_cells (int, optional): Number of cells to use for plotting correlation. Defaults to 10000.
            doplot (bool, optional): Whether to generate plots of the similarity between the denoised and true expression data. Defaults to False.
                Only works when downsample_expr is not None and max_cells < 100.
            predict_depth_mult (int, optional): Multiplier for prediction depth. Defaults to 4.
                This will artificially increase the sequencing depth (or number of counts) to 4 times the original depth.
            genes_to_use (Optional[List[str]], optional): List of genes to use for imputation. Defaults to None.
            genes_to_impute (Optional[List[str]], optional): List of genes to impute. Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
                This is important to avoid memory issues.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_cells = max_cells
        self.doplot = doplot
        self.predict_depth_mult = predict_depth_mult
        self.save_every = save_every
        self.additional_info = additional_info
        self.genes_to_use = genes_to_use
        self.genes_to_impute = genes_to_impute

    def __call__(self, model: torch.nn.Module, adata: AnnData):
        """
        __call__ calling the function

        Args:
            model (torch.nn.Module): The scPRINT model to be used for denoising.
            adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.

        Returns:
            AnnData: The denoised annotated data matrix.
        """
        # Select random number
        random_indices = None
        if self.max_cells < adata.shape[0]:
            random_indices = np.random.randint(
                low=0, high=adata.shape[0], size=self.max_cells
            )
            adataset = SimpleAnnDataset(
                adata[random_indices],
                obs_to_output=["organism_ontology_term_id"],
                get_knn_cells=model.expr_emb_style == "metacell",
            )
        else:
            adataset = SimpleAnnDataset(
                adata,
                obs_to_output=["organism_ontology_term_id"],
                get_knn_cells=model.expr_emb_style == "metacell",
            )
        l = len(self.genes_to_use)
        self.genes_to_use = [i for i in model.genes if i in self.genes_to_use]
        print(f"{}% of genes to use are available in the model".format(
            100 * len(self.genes_to_use) / l
        ))
        l = len(self.genes_to_impute)
        self.genes_to_impute = [i for i in model.genes if i in self.genes_to_impute]
        print(f"{}% of genes to impute are available in the model".format(
            100 * len(self.genes_to_impute) / l
        ))

        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            how="some",
            genelist=self.genes_to_use+self.genes_to_impute
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        if method=="masking":
            mask = torch.Tensor([i in self.genes_to_impute for i in model.genes], dtype(bool), device=model.device)
        else:
            mask = None

        prevplot = model.doplot
        model.doplot = self.doplot
        model.on_predict_epoch_start()
        model.eval()
        device = model.device.type
        stored_noisy = None
        rand = random_str()
        dtype = (
            torch.float16
            if type(model.transformer) is FlashTransformer
            else model.dtype
        )
        torch.cuda.empty_cache()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            for batch in tqdm(dataloader):
                gene_pos, expression, depth = (
                    batch["genes"].to(device),
                    batch["x"].to(device),
                    batch["depth"].to(device),
                )
                model._predict(
                    gene_pos,
                    expression,
                    depth,
                    knn_cells=batch["knn_cells"].to(device)
                    if model.expr_emb_style == "metacell"
                    else None,
                    do_generate=False,
                    depth_mult=self.predict_depth_mult,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=self.save_every,
                    name="impute" + rand + "_",
                    mask=mask
                )
        torch.cuda.empty_cache()
        model.log_adata(name="impute" + rand + "_" + str(model.counter))
        try:
            mdir = (
                model.logger.save_dir if model.logger.save_dir is not None else "data"
            )
        except:
            mdir = "data"
        pred_adata = []
        for i in range(model.counter + 1):
            file = (
                mdir
                + "/step_"
                + str(model.global_step)
                + "_"
                + model.name
                + "_impute"
                + rand
                + "_"
                + str(i)
                + "_"
                + str(model.global_rank)
                + ".h5ad"
            )
            pred_adata.append(sc.read_h5ad(file))
            os.remove(file)
        pred_adata = concat(pred_adata)

        model.doplot = prevplot

        pred_adata.X = adata.X if random_indices is None else adata.X[random_indices]
        pred_imp = pred_adata.layers['scprint_mu'][:, pred_adata.var.index.isin(self.genes_to_impute)].toarray()
        true_imp = pred_adata.X[
            :,pred_adata.var.index.isin(self.genes_to_impute)
        ].toarray()
        
        if true_imp.sum() > 0:
            # we had some gt
        
            pred_known = pred_adata.layers['scprint_mu'][:, pred_adata.var.index.isin(self.genes_to_use)].toarray()
            true_known = pred_adata.X[
                :,pred_adata.var.index.isin(self.genes_to_use)
            ].toarray()
            
            if self.apply_zero_pred:
                pred_imp = pred_imp * F.sigmoid(
                        torch.Tensor(
                            pred_adata.layers["scprint_pi"][:, pred_adata.var.index.isin(self.genes_to_impute)].toarray()
                        < 0.5
                    ).numpy()
                )
                pred_known = pred_known * F.sigmoid(
                        torch.Tensor(
                            pred_adata.layers["scprint_pi"][:, pred_adata.var.index.isin(self.genes_to_use)].toarray()
                        < 0.5
                    ).numpy()
                )
            cell_wise_pred = np.array(
                [spearmanr(pred_imp[i], true_imp[i])[0] for i in range(pred_imp.shape[0])]
            )
            cell_wise_known = np.array(
                [spearmanr(pred_known[i], true_known[i])[0] for i in range(pred_known.shape[0])]
            )
            print(
                {
                    "cell_wise_known": np.mean(cell_wise_known),
                    "cell_wise_pred": np.mean(cell_wise_pred),
                }
            )
            if self.doplot:
                print("depth-wise plot")
                plot_cell_depth_wise_corr_improvement(cell_wise_known, cell_wise_pred)

        return random_indices, pred_adata
