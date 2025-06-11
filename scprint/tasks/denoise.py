import os
from typing import Any, List, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import sklearn.metrics
import torch
from anndata import AnnData, concat
from scdataloader import Collator
from scdataloader.data import SimpleAnnDataset
from scipy.sparse import issparse
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint.model import utils

from . import knn_smooth

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Denoiser:
    def __init__(
        self,
        batch_size: int = 10,
        num_workers: int = 1,
        max_len: int = 5_000,
        how: str = "most var",
        max_cells: int = 500_000,
        doplot: bool = False,
        predict_depth_mult: int = 4,
        downsample_expr: Optional[float] = None,
        dtype: torch.dtype = torch.float16,
        genelist: Optional[List[str]] = None,
        save_every: int = 100_000,
    ):
        """
        Denoiser class for denoising scRNA-seq data using a scPRINT model

        Args:
            batch_size (int, optional): Batch size for processing. Defaults to 10.
            num_workers (int, optional): Number of workers for data loading. Defaults to 1.
            max_len (int, optional): Maximum number of genes to consider. Defaults to 5000.
            how (str, optional): Method to select genes. Options are "most var", "random expr", "some". Defaults to "most var".
                - "most var": select the most variable genes
                - "random expr": select random expressed genes
                - "some": select a subset of genes defined in genelist
            max_cells (int, optional): Number of cells to use for plotting correlation. Defaults to 10000.
            doplot (bool, optional): Whether to generate plots of the similarity between the denoised and true expression data. Defaults to False.
                Only works when downsample_expr is not None and max_cells < 100.
            predict_depth_mult (int, optional): Multiplier for prediction depth. Defaults to 4.
                This will artificially increase the sequencing depth (or number of counts) to 4 times the original depth.
            downsample_expr (Optional[float], optional): Fraction of expression data to downsample. Defaults to None.
                This is usefull to test the ability of the model to denoise the dataset.
                When this option is on, the class output is a tuple of (metrics, random_indices, pred_adata).
                Where the metric is the result of the denoising from the downsampled expression to true expression data.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
            genelist (List[str], optional): The list of genes to be used for embedding. Defaults to []: In this case, "how" needs to be "most var" or "random expr".
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
                This is important to avoid memory issues.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.max_cells = max_cells
        self.doplot = doplot
        self.predict_depth_mult = predict_depth_mult
        self.how = how
        self.downsample_expr = downsample_expr
        self.dtype = dtype
        self.genelist = genelist
        self.save_every = save_every

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
        if self.how == "most var":
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=self.max_len, span=0.99
            )
            self.genelist = adata.var.index[adata.var.highly_variable]
        else:
            self.genelist = adata.var.index
        self.genelist = [i for i in model.genes if i in self.genelist]
        print(f"working on {len(self.genelist)} accepted genes")

        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            max_len=self.max_len,
            how="some" if self.how == "most var" else self.how,
            genelist=self.genelist if self.how != "random expr" else [],
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        prevplot = model.doplot
        model.doplot = self.doplot
        model.on_predict_epoch_start()
        model.eval()
        device = model.device.type
        stored_noisy = None
        with torch.no_grad(), torch.autocast(device_type=device, dtype=self.dtype):
            for batch in tqdm(dataloader):
                gene_pos, expression, depth = (
                    batch["genes"].to(device),
                    batch["x"].to(device),
                    batch["depth"].to(device),
                )
                if self.downsample_expr is not None:
                    expression = utils.downsample_profile(
                        expression, self.downsample_expr
                    )
                if stored_noisy is None:
                    stored_noisy = expression.cpu().numpy()
                else:
                    stored_noisy = np.concatenate(
                        [stored_noisy, expression.cpu().numpy()], axis=0
                    )

                model._predict(
                    gene_pos,
                    expression,
                    depth,
                    knn_cells=batch["knn_cells"].to(device)
                    if model.expr_emb_style == "metacell"
                    else None,
                    predict_mode="denoise",
                    depth_mult=self.predict_depth_mult,
                    max_size_in_mem=self.save_every,
                )
        torch.cuda.empty_cache()
        model.log_adata(name="predict_part_" + str(model.counter))
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
                + "_predict_part_"
                + str(i)
                + "_"
                + str(model.global_rank)
                + ".h5ad"
            )
            pred_adata.append(sc.read_h5ad(file))
        pred_adata = concat(pred_adata)
        metrics = None
        model.doplot = prevplot
        if self.downsample_expr is not None:
            pred_adata.layers["scprint_mu"]
            if model.transformer.attn_type == "hyper":
                # seq len must be a multiple of 128
                num = model.cell_embs_count if not model.cell_transformer else 0
                if (stored_noisy.shape[1] + num) % 128 != 0:
                    stored_noisy = stored_noisy[
                        :, : ((stored_noisy.shape[1]) // 128 * 128) - num
                    ]
            reco = pred_adata.layers["scprint_mu"].data.reshape(pred_adata.shape[0], -1)
            adata = (
                adata[random_indices, adata.var.index.isin(pred_adata.var.index)]
                if random_indices is not None
                else adata[:, adata.var.index.isin(pred_adata.var.index)]
            )
            true = adata.X[
                :,
                pred_adata.layers["scprint_mu"][
                    :, pred_adata.var.index.isin(adata.var.index)
                ]
                .toarray()
                .any(axis=0),
            ].toarray()

            corr_coef, p_value = spearmanr(
                np.vstack([reco[true != 0], stored_noisy[true != 0], true[true != 0]]).T
            )
            metrics = {
                "reco2noisy": corr_coef[0, 1],
                "reco2full": corr_coef[0, 2],
                "noisy2full": corr_coef[1, 2],
            }
            if self.doplot and self.max_cells < 100:
                corr_coef[p_value > 0.05] = 0
                plt.figure(figsize=(10, 5))
                plt.imshow(
                    corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
                )
                plt.colorbar()
                plt.title("Expression Correlation Coefficient")
                plt.show()
        return metrics, random_indices, pred_adata


# testdatasets=['/R4ZHoQegxXdSFNFY5LGe.h5ad', '/SHV11AEetZOms4Wh7Ehb.h5ad',
# '/V6DPJx8rP3wWRQ43LMHb.h5ad', '/Gz5G2ETTEuuRDgwm7brA.h5ad', '/YyBdEsN89p2aF4xJY1CW.h5ad',
# '/SO5yBTUDBgkAmz0QbG8K.h5ad', '/r4iCehg3Tw5IbCLiCIbl.h5ad', '/SqvXr3i3PGXM8toXzUf9.h5ad',
# '/REIyQZE6OMZm1S3W2Dxi.h5ad', '/rYZ7gs0E0cqPOLONC8ia.h5ad', '/FcwMDDbAQPNYIjcYNxoc.h5ad',
# '/fvU5BAMJrm7vrgDmZM0z.h5ad', '/gNNpgpo6gATjuxTE7CCp.h5ad'],


def default_benchmark(
    model: Any,
    default_dataset: str = FILE_DIR
    + "/../../data/gNNpgpo6gATjuxTE7CCp.h5ad",  # r4iCehg3Tw5IbCLiCIbl
    max_len: int = 5000,
):
    """
    default_benchmark function used to run the default denoising benchmark of scPRINT

    Args:
        model (Any): The scPRINT model to be used for the benchmark.
        default_dataset (str, optional): Path to the default dataset to use for benchmarking. Defaults to FILE_DIR + "/../../data/r4iCehg3Tw5IbCLiCIbl.h5ad".
        max_len (int, optional): Maximum number of genes to consider. Defaults to 5000.

    Returns:
        dict: A dictionary containing the benchmark metrics.
    """
    adata = sc.read_h5ad(default_dataset)
    if model.expr_emb_style == "metacell":
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, use_rep="X_pca")
    denoise = Denoiser(
        batch_size=40 if model.expr_emb_style != "metacell" else 20,
        max_len=max_len,
        max_cells=10_000,
        doplot=False,
        num_workers=8,
        predict_depth_mult=5,
        downsample_expr=0.7,
    )
    return denoise(model, adata)[0]


def open_benchmark(model):
    adata = sc.read(
        FILE_DIR + "/../../data/pancreas_atlas.h5ad",
        backup_url="https://figshare.com/ndownloader/files/24539828",
    )
    from scdataloader import Preprocessor

    adata = adata[adata.obs.tech == "inDrop1"]

    train, test = split_molecules(adata.layers["counts"].round().astype(int), 0.9)
    is_missing = np.array(train.sum(axis=0) == 0)
    true = adata.copy()
    true.X = test
    adata.layers["counts"] = train
    test = test[:, ~is_missing.flatten()]
    adata = adata[:, ~is_missing.flatten()]

    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    preprocessor = Preprocessor(
        subset_hvg=3000,
        use_layer="counts",
        is_symbol=True,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
    )
    nadata = preprocessor(adata.copy())

    denoise = Denoiser(
        batch_size=32,
        max_len=15_800,
        max_cells=10_000,
        doplot=False,
        predict_depth_mult=1.2,
        downsample_expr=None,
    )
    expr = denoise(model, nadata)
    denoised = ad.AnnData(
        expr.cpu().numpy(),
        var=nadata.var.loc[
            np.array(denoise.model.genes)[
                denoise.model.pos[0].cpu().numpy().astype(int)
            ]
        ],
    )
    denoised = denoised[:, denoised.var.symbol.isin(true.var.index)]
    loc = true.var.index.isin(denoised.var.symbol)
    true = true[:, loc]
    train = train[:, loc]

    # Ensure expr and adata are aligned by reordering expr to match adata's .var order
    denoised = denoised[
        :, denoised.var.set_index("symbol").index.get_indexer(true.var.index)
    ]
    denoised.X = np.maximum(denoised.X - train.astype(float), 0)
    # scaling and transformation
    target_sum = 1e4

    sc.pp.normalize_total(true, target_sum)
    sc.pp.log1p(true)

    sc.pp.normalize_total(denoised, target_sum)
    sc.pp.log1p(denoised)

    error_mse = sklearn.metrics.mean_squared_error(true.X, denoised.X)
    # scaling
    initial_sum = train.sum()
    target_sum = true.X.sum()
    denoised.X = denoised.X * target_sum / initial_sum
    error_poisson = poisson_nll_loss(true.X, denoised.X)
    return {"error_poisson": error_poisson, "error_mse": error_mse}


def mse(test_data, denoised_data, target_sum=1e4):
    sc.pp.normalize_total(test_data, target_sum)
    sc.pp.log1p(test_data)

    sc.pp.normalize_total(denoised_data, target_sum)
    sc.pp.log1p(denoised_data)

    print("Compute mse value", flush=True)
    return sklearn.metrics.mean_squared_error(test_data.X.todense(), denoised_data.X)


def withknn(adata, k=10, **kwargs):
    adata.layers["denoised"] = knn_smooth.knn_smoothing(
        adata.X.transpose(), k=k, **kwargs
    ).transpose()
    return adata


# from molecular_cross_validation.mcv_sweep import poisson_nll_loss
# copied from: https://github.com/czbiohub/molecular-cross-validation/blob/master/src/molecular_cross_validation/mcv_sweep.py
def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


def split_molecules(
    umis: np.ndarray,
    data_split: float,
    overlap_factor: float = 0.0,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits molecules into two (potentially overlapping) groups.
    :param umis: Array of molecules to split
    :param data_split: Proportion of molecules to assign to the first group
    :param overlap_factor: Overlap correction factor, if desired
    :param random_state: For reproducible sampling
    :return: umis_X and umis_Y, representing ``split`` and ``~(1 - split)`` counts
             sampled from the input array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - data_split) / (1 - data_split + overlap_factor)
    )
    overlap_factor = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap_factor
    umis_Y = umis_Y_disjoint + overlap_factor

    return umis_X, umis_Y
