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

from scprint2.model import utils
from scprint2.tasks.denoise import plot_cell_depth_wise_corr_improvement

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Imputer:
    def __init__(
        self,
        batch_size: int = 10,
        num_workers: int = 1,
        max_cells: int = 500_000,
        doplot: bool = False,
        method: str = "generative",
        predict_depth_mult: int = 4,
        genes_to_use: Optional[List[str]] = None,
        genes_to_impute: Optional[List[str]] = None,
        save_every: int = 100_000,
        apply_zero_pred: bool = True,
        use_knn: bool = True,
    ):
        """
        Imputer class for imputing missing values in scRNA-seq data using a scPRINT model

        Args:
            batch_size (int, optional): Batch size for processing. Defaults to 10.
            num_workers (int, optional): Number of workers for data loading. Defaults to 1.
            max_cells (int, optional): Number of cells to use for plotting correlation. Defaults to 10000.
            doplot (bool, optional): Whether to generate plots of the similarity between the denoised and true expression data. Defaults to False.
                Only works when downsample_expr is not None and max_cells < 100.
            method (str, optional): Imputation method, either 'masking' or 'generative'. Defaults to 'generative'.
            predict_depth_mult (int, optional): Multiplier for prediction depth. Defaults to 4.
                This will artificially increase the sequencing depth (or number of counts) to 4 times the original depth.
            genes_to_use (Optional[List[str]], optional): List of genes to use for imputation. Defaults to None.
            genes_to_impute (Optional[List[str]], optional): List of genes to impute. Defaults to None.
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
                This is important to avoid memory issues.
            apply_zero_pred (bool, optional): Whether to apply zero prediction adjustment. Defaults to True.
                applying zero inflation might give results closer to the specific biases of sequencing technologies but less biological truthful.
            use_knn (bool, optional): Whether to use k-nearest neighbors information. Defaults to True.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_cells = max_cells
        self.doplot = doplot
        self.predict_depth_mult = predict_depth_mult
        self.save_every = save_every
        self.genes_to_use = genes_to_use
        self.genes_to_impute = genes_to_impute
        self.method = method
        self.apply_zero_pred = apply_zero_pred
        self.use_knn = use_knn

    def __call__(self, model: torch.nn.Module, adata: AnnData) -> tuple[Optional[np.ndarray], AnnData]:
        """
        __call__ calling the function

        Args:
            model (torch.nn.Module): The scPRINT model to be used for denoising.
            adata (AnnData): The anndata of shape n_obs x n_vars. Rows correspond to cells and columns to genes.

        Returns:
            Optional[np.ndarray]: The random indices of the cells used when max_cells < adata.shape[0].
            AnnData: The imputed anndata.
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
                get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
            )
        else:
            adataset = SimpleAnnDataset(
                adata,
                obs_to_output=["organism_ontology_term_id"],
                get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
            )
        genes_to_use = set(model.genes) & set(self.genes_to_use)
        print(
            f"{100 * len(genes_to_use) / len(self.genes_to_use)}% of genes to use are available in the model"
        )
        genes_to_impute = set(model.genes) & set(self.genes_to_impute)
        print(
            f"{100 * len(genes_to_impute) / len(self.genes_to_impute)}% of genes to impute are available in the model"
        )
        tot = genes_to_use | genes_to_impute
        tot = sorted(tot)
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            how="some",
            genelist=list(genes_to_use)
            + (list(genes_to_impute) if self.method == "masking" else []),
            n_bins=model.n_input_bins if model.expr_emb_style == "binned" else 0,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        mask = None
        generate_on = None
        if self.method == "masking":
            mask = torch.Tensor(
                [i in genes_to_use for i in tot],
            ).to(device=model.device, dtype=torch.bool)
        elif self.method == "generative":
            generate_on = (
                torch.Tensor([model.genes.index(i) for i in genes_to_impute])
                .to(device=model.device)
                .long()
                .unsqueeze(0)
                .repeat(self.batch_size, 1)
            )
        else:
            raise ValueError("need to be one of generative or masking")

        prevplot = model.doplot
        model.doplot = self.doplot
        model.on_predict_epoch_start()
        model.eval()
        device = model.device.type
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
                    knn_cells=(
                        batch["knn_cells"].to(device)
                        if model.expr_emb_style == "metacell" and self.use_knn
                        else None
                    ),
                    do_generate=self.method == "generative",
                    depth_mult=self.predict_depth_mult,
                    max_size_in_mem=self.save_every,
                    name="impute" + rand + "_",
                    mask=mask,
                    generate_on=generate_on,
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

        # pred_adata.X = adata.X if random_indices is None else adata.X[random_indices]
        true_imp = pred_adata.X[:, pred_adata.var.index.isin(genes_to_impute)].toarray()

        if true_imp.sum() > 0:
            # we had some gt
            pred_imp = pred_adata.layers["scprint_mu"][
                :, pred_adata.var.index.isin(genes_to_impute)
            ].toarray()
            pred_known = pred_adata.layers["scprint_mu"][
                :, pred_adata.var.index.isin(genes_to_use)
            ].toarray()
            true_known = pred_adata.X[
                :, pred_adata.var.index.isin(genes_to_use)
            ].toarray()

            if self.apply_zero_pred:
                pred_imp = (
                    pred_imp
                    * (
                        1
                        - F.sigmoid(
                            torch.Tensor(
                                pred_adata.layers["scprint_pi"][
                                    :, pred_adata.var.index.isin(genes_to_impute)
                                ].toarray()
                            )
                        )
                    ).numpy()
                )
                pred_known = (
                    pred_known
                    * (
                        1
                        - F.sigmoid(
                            torch.Tensor(
                                pred_adata.layers["scprint_pi"][
                                    :, pred_adata.var.index.isin(genes_to_use)
                                ].toarray()
                            )
                        )
                    ).numpy()
                )
            cell_wise_pred = np.array(
                [
                    spearmanr(pred_imp[i], true_imp[i])[0]
                    for i in range(pred_imp.shape[0])
                ]
            )
            cell_wise_known = np.array(
                [
                    spearmanr(pred_known[i], true_known[i])[0]
                    for i in range(pred_known.shape[0])
                ]
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
