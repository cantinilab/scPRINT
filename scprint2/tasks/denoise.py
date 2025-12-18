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
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from simpler_flash import FlashTransformer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint2.model import utils

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
        genelist: Optional[List[str]] = None,
        save_every: int = 100_000,
        pred_embedding: List[str] = ["cell_type_ontology_term_id"],
        additional_info: bool = False,
        apply_zero_pred: bool = False,
        use_knn: bool = True,
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
                This is usefull to test the ability of the model to denoise the dataset. only to use the input data as a benchmark dataset.
                When this option is on, the denoiser will output benchmark metrics
            genelist (List[str], optional): The list of genes to be used for embedding. Defaults to []: In this case, "how" needs to be "most var" or "random expr".
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
                This is important to avoid memory issues.
            pred_embedding (List[str], optional): The embedding type to be used as the denoising will also predict the cell embeddings.
            additional_info (bool, optional): Whether to print additional benchmark information during denoising. Defaults to False.
                only useful when downsampling is used.
            apply_zero_pred (bool, optional): Whether to apply zero inflation to the output value during denoising, else uses only the predicted mean.
                applying zero inflation might give results closer to the specific biases of sequencing technologies but less biological truthful.
            use_knn (bool, optional): Whether to use knn cells for denoising when the model uses metacell expression embedding. Defaults to True.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.max_cells = max_cells
        self.doplot = doplot
        self.predict_depth_mult = predict_depth_mult
        self.how = how
        self.downsample_expr = downsample_expr
        self.genelist = genelist
        self.save_every = save_every
        self.pred_embedding = pred_embedding
        self.additional_info = additional_info
        self.apply_zero_pred = apply_zero_pred
        self.use_knn = use_knn

    def __call__(self, model: torch.nn.Module, adata: AnnData) -> tuple[dict, Optional[np.ndarray], AnnData]:
        """
        __call__ calling the function

        Args:
            model (torch.nn.Module): The scPRINT model to be used for denoising.
            adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.

        Returns:
            dict: The benchmark metrics if downsampling is used.
            Optional[np.ndarray]: The random set of cells used if max_cells < adata.shape[0].
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
                get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
            )
        else:
            adataset = SimpleAnnDataset(
                adata,
                obs_to_output=["organism_ontology_term_id"],
                get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
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
            n_bins=model.n_input_bins if model.expr_emb_style == "binned" else 0,
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
        model.pred_log_adata = True
        stored_noisy = None
        rand = random_str()
        dtype = (
            torch.float16
            if type(model.transformer) is FlashTransformer
            else model.dtype
        )
        torch.cuda.empty_cache()
        save_expr = model.save_expr
        model.save_expr = True
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            for batch in tqdm(dataloader):
                gene_pos, expression, depth = (
                    batch["genes"].to(device),
                    batch["x"].to(device),
                    batch["depth"].to(device),
                )
                knn_cells = (
                    batch["knn_cells"].to(device)
                    if model.expr_emb_style == "metacell" and self.use_knn
                    else None
                )
                if self.downsample_expr is not None:
                    expression = utils.downsample_profile(
                        expression, self.downsample_expr
                    )
                    if knn_cells is not None:
                        for i in range(knn_cells.shape[1]):
                            knn_cells[:, i] = utils.downsample_profile(
                                knn_cells[:, i], self.downsample_expr
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
                    knn_cells=(
                        batch["knn_cells"].to(device)
                        if model.expr_emb_style == "metacell" and self.use_knn
                        else None
                    ),
                    knn_cells_info=(
                        batch["knn_cells_info"].to(device)
                        if model.expr_emb_style == "metacell" and self.use_knn
                        else None
                    ),
                    do_generate=False,
                    depth_mult=self.predict_depth_mult,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=self.save_every,
                    name="denoise_" + rand + "_",
                )
        torch.cuda.empty_cache()
        model.log_adata(name="denoise_" + rand + "_" + str(model.counter))
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
                + "_denoise_"
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

        if model.transformer.attn_type == "hyper":
            # seq len must be a multiple of 128
            num = (1 if model.use_metacell_token else 0) + (
                (len(model.classes) + 1) if not model.cell_transformer else 0
            )
            if (stored_noisy.shape[1] + num) % 128 != 0:
                stored_noisy = stored_noisy[
                    :, : ((stored_noisy.shape[1]) // 128 * 128) - num
                ]
        pred_adata.X = stored_noisy

        metrics = None
        model.doplot = prevplot
        model.save_expr = save_expr
        if self.downsample_expr is not None:
            reco = np.array(pred_adata.layers["scprint_mu"].data).reshape(
                pred_adata.shape[0], -1
            )
            # reco = reco * F.sigmoid(
            #    torch.Tensor(np.array(pred_adata.layers["scprint_pi"].data).reshape(pred_adata.shape[0], -1)) < 0.5
            # ).numpy()

            adata = (
                adata[random_indices, adata.var.index.isin(pred_adata.var.index)]
                if random_indices is not None
                else adata[:, adata.var.index.isin(pred_adata.var.index)]
            )
            true = adata[
                :,
                pred_adata.var.index[
                    pred_adata.var.index.isin(adata.var.index)
                ].to_list(),
            ].X.toarray()
            if self.apply_zero_pred:
                reco = (
                    reco
                    * (
                        1
                        - F.sigmoid(
                            torch.Tensor(
                                np.array(pred_adata.layers["scprint_pi"].data).reshape(
                                    pred_adata.shape[0], -1
                                )
                            )
                        )
                    ).numpy()
                )

            corr_coef, p_value = spearmanr(
                np.vstack([reco[true != 0], stored_noisy[true != 0], true[true != 0]]).T
            )
            metrics = {
                "reco2noisy": corr_coef[0, 1],
                "reco2full": corr_coef[0, 2],
                "noisy2full": corr_coef[1, 2],
            }
            if self.additional_info:
                # Sample only 3000 elements for correlation calculation
                if reco.shape[0] > 3000:
                    indices = np.random.choice(reco.shape[0], 3000, replace=False)
                    reco = reco[indices]
                    stored_noisy = stored_noisy[indices]
                    true = true[indices]
                corr, p_value = spearmanr(
                    np.vstack(
                        [
                            reco.flatten(),
                            stored_noisy.flatten(),
                            true.flatten(),
                        ]
                    ).T
                )
                m = {
                    "reco2full": corr[0, 2],
                    "noisy2full": corr[1, 2],
                }
                print("corr with zeros: ")
                print(m)
                cell_wise = np.array(
                    [
                        spearmanr(reco[i][true[i] != 0], true[i][true[i] != 0])[0]
                        for i in range(reco.shape[0])
                    ]
                )
                torm = np.array(
                    [
                        spearmanr(stored_noisy[i][true[i] != 0], true[i][true[i] != 0])[
                            0
                        ]
                        for i in range(reco.shape[0])
                    ]
                )
                cell_wise -= torm
                cell_wise_zero = np.mean(
                    [spearmanr(reco[i], true[i])[0] for i in range(reco.shape[0])]
                )
                print("cell_wise self corr (reco, noisy, true)")
                print(
                    {
                        "cell_wise_w_zero": cell_wise_zero,
                        "cell_wise_to_noisy": np.mean(cell_wise),
                    }
                )
                print("depth-wise plot")
                plot_cell_depth_wise_corr_improvement(cell_wise, (true > 0).sum(1))

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
    folder_dir: str = FILE_DIR + "/../../data/",
    dataset: str = FILE_DIR
    + "/../../data/gNNpgpo6gATjuxTE7CCp.h5ad",  # r4iCehg3Tw5IbCLiCIbl
) -> dict:
    """
    default_benchmark function used to run the default denoising benchmark of scPRINT

    Args:
        model (Any): The scPRINT model to be used for the benchmark.
        folder_dir (str, optional): Directory containing data files.
        dataset (str, optional): Path to the dataset to use for benchmarking.

    Returns:
        dict: A dictionary containing the benchmark metrics.
    """
    if dataset.startswith("https://"):
        adata = sc.read(
            folder_dir + dataset.split("/")[-1],
            backup_url=dataset,
        )
    else:
        adata = sc.read_h5ad(dataset)
    if dataset.split("/")[-1] == "gNNpgpo6gATjuxTE7CCp.h5ad":
        use_layer = "counts"
        is_symbol = True
    else:
        use_layer = None
        is_symbol = False
    max_len = 4000 if adata.X.sum(1).mean() < 150_000 else 8000
    preprocessor = Preprocessor(
        use_layer=use_layer,
        is_symbol=is_symbol,
        force_preprocess=True,
        skip_validate=True,
        do_postp=model.expr_emb_style == "metacell",
        drop_non_primary=False,
    )
    adata = preprocessor(adata.copy())
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
        pred_embedding=model.pred_embedding,
    )
    return denoise(model, adata)[0]


def mse(test_data, denoised_data, target_sum=1e4):
    sc.pp.normalize_total(test_data, target_sum=target_sum)
    sc.pp.log1p(test_data)

    sc.pp.normalize_total(denoised_data, target_sum=target_sum)
    sc.pp.log1p(denoised_data)

    print("Compute mse value", flush=True)
    return sklearn.metrics.mean_squared_error(test_data.X.todense(), denoised_data.X)


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


def plot_cell_depth_wise_corr_improvement(corr_coef, y):
    def linear_func(x, a, b):
        return a * np.log(x) + b

    # Fit the linear curve
    ppot, _ = curve_fit(linear_func, y, corr_coef)

    # Plot the data points
    plt.scatter(
        y, corr_coef, label="denoising increase as depth increase", color="blue"
    )

    # Plot the fitted linear curve
    x_values = np.linspace(min(y), max(y), 100)
    plt.plot(
        x_values,
        linear_func(x_values, *ppot),
        label="Linear Fit",
        color="red",
        linestyle="--",
    )

    plt.xlabel("True sum (depth)")
    plt.ylabel("Denoising improvement")
    plt.ylim(-0.5, 1)
    plt.legend()
    plt.show()
