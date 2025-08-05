import os
from typing import Any, Dict, List

import bionty as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData, concat
from lightning.pytorch import Trainer
from networkx import average_node_connectivity
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from scdataloader.utils import get_descendants
from scib_metrics.benchmark import Benchmarker
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint.model import utils

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class Embedder:
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        how: str = "random expr",
        max_len: int = 2000,
        doclass: bool = True,
        add_zero_genes: int = 0,
        precision: str = "16-mixed",
        pred_embedding: List[str] = [
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        doplot: bool = True,
        keep_all_cls_pred: bool = False,
        dtype: torch.dtype = torch.float16,
        output_expression: str = "none",
        genelist: List[str] = [],
        get_gene_emb: bool = False,
        save_every: int = 40_000,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            how (str, optional): The method to be used for selecting valid genes. Defaults to "random expr".
            max_len (int, optional): The maximum length of the gene sequence. Defaults to 1000.
            add_zero_genes (int, optional): The number of zero genes to add to the gene sequence. Defaults to 100.
            precision (str, optional): The precision to be used in the Trainer. Defaults to "16-mixed".
            pred_embedding (List[str], optional): The list of labels to be used for plotting embeddings. Defaults to [ "cell_type_ontology_term_id", "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id", "sex_ontology_term_id", ].
            doclass (bool, optional): Whether to perform classification. Defaults to True.
            doplot (bool, optional): Whether to generate plots. Defaults to True.
            keep_all_cls_pred (bool, optional): Whether to keep all class predictions. Defaults to False.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
            output_expression (str, optional): The method to output expression data. Options are "none", "all", "sample". Defaults to "none".
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.add_zero_genes = add_zero_genes
        self.pred_embedding = pred_embedding
        self.keep_all_cls_pred = keep_all_cls_pred
        self.precision = precision
        self.doplot = doplot
        self.dtype = dtype
        self.doclass = doclass
        self.output_expression = output_expression
        self.genelist = genelist
        self.get_gene_emb = get_gene_emb
        self.save_every = save_every

    def __call__(self, model: torch.nn.Module, adata: AnnData, cache=False):
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
            np.ndarray: The predicted expression values if output_expression is not "none".
            dict: Additional metrics and information from the embedding process.
        """
        # one of "all" "sample" "none"
        model.predict_mode = "none"
        model.keep_all_cls_pred = self.keep_all_cls_pred
        # Add at least the organism you are working with
        if self.how == "most var":
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=self.max_len
            )
            self.genelist = adata.var.index[adata.var.highly_variable]
        adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            how=self.how if self.how != "most var" else "some",
            max_len=self.max_len,
            add_zero_genes=self.add_zero_genes,
            genelist=self.genelist if self.how in ["most var", "some"] else [],
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        model.eval()
        model.on_predict_epoch_start()
        device = model.device.type
        model.doplot = self.doplot
        with (
            torch.no_grad(),
            torch.autocast(device_type=device, dtype=self.dtype),
        ):
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
                    predict_mode="none",
                    pred_embedding=self.pred_embedding,
                    get_gene_emb=self.get_gene_emb,
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
        if self.output_expression == "sample":
            adata.layers["sampled"] = (
                utils.zinb_sample(
                    torch.from_numpy(pred_adata.layers["scprint_mu"]),
                    torch.from_numpy(pred_adata.layers["scprint_theta"]),
                    torch.from_numpy(pred_adata.layers["scprint_pi"]),
                )
                .cpu()
                .numpy()
            )
        else:
            pass
        pred_adata.obs.index = adata.obs.index
        try:
            adata.obsm["X_scprint_umap"] = pred_adata.obsm["X_umap"]
        except:
            print("too few cells to embed into a umap")
        try:
            adata.obsm["scprint_leiden"] = pred_adata.obsm["leiden"]
        except:
            print("too few cells to compute a clustering")
        adata.obsm["scprint_emb"] = pred_adata.obsm["scprint_emb"]
        for key, value in pred_adata.uns.items():
            adata.uns[key] = value

        pred_adata.obs.index = adata.obs.index
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)
        if self.keep_all_cls_pred:
            allclspred = model.pred
            columns = []
            for cl in model.classes:
                n = model.label_counts[cl]
                columns += [model.label_decoders[cl][i] for i in range(n)]
            allclspred = pd.DataFrame(
                allclspred, columns=columns, index=adata.obs.index
            )
            adata.obs = pd.concat(adata.obs, allclspred)

        metrics = {}
        if self.doclass and not self.keep_all_cls_pred:
            for cl in model.classes:
                res = []
                if cl not in adata.obs.columns:
                    continue
                class_topred = model.label_decoders[cl].values()

                if cl in model.labels_hierarchy:
                    # class_groupings = {
                    #    k: [
                    #        i.ontology_id
                    #        for i in bt.CellType.filter(k).first().children.all()
                    #    ]
                    #    for k in set(adata.obs[cl].unique()) - set(class_topred)
                    # }
                    cur_labels_hierarchy = {
                        model.label_decoders[cl][k]: [
                            model.label_decoders[cl][i] for i in v
                        ]
                        for k, v in model.labels_hierarchy[cl].items()
                    }
                else:
                    cur_labels_hierarchy = {}

                for pred, true in adata.obs[["pred_" + cl, cl]].values:
                    if pred == true:
                        res.append(True)
                        continue
                    if len(cur_labels_hierarchy) > 0:
                        if true in cur_labels_hierarchy:
                            res.append(pred in cur_labels_hierarchy[true])
                            continue
                        elif true not in class_topred:
                            raise ValueError(
                                f"true label {true} not in available classes"
                            )
                        elif true != "unknown":
                            res.append(False)
                    elif true not in class_topred:
                        raise ValueError(f"true label {true} not in available classes")
                    elif true != "unknown":
                        res.append(False)
                    # else true is unknown
                    # else we pass
                if len(res) == 0:
                    # true was always unknown
                    res = [1]
                if self.doplot:
                    print("    ", cl)
                    print("     accuracy:", sum(res) / len(res))
                    print(" ")
                metrics.update({cl + "_accuracy": sum(res) / len(res)})
        return adata, metrics


def compute_corr(
    out: np.ndarray,
    to: np.ndarray,
    doplot: bool = True,
    compute_mean_regress: bool = False,
    plot_corr_size: int = 64,
) -> dict:
    """
    Compute the correlation between the output and target matrices.

    Args:
        out (np.ndarray): The output matrix.
        to (np.ndarray): The target matrix.
        doplot (bool, optional): Whether to generate a plot of the correlation coefficients. Defaults to True.
        compute_mean_regress (bool, optional): Whether to compute mean regression. Defaults to False.
        plot_corr_size (int, optional): The size of the plot for correlation. Defaults to 64.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    metrics = {}
    corr_coef, p_value = spearmanr(
        out,
        to.T,
    )
    corr_coef[p_value > 0.05] = 0
    # corr_coef[]
    # only on non zero values,
    # compare a1-b1 corr with a1-b(n) corr. should be higher

    # Plot correlation coefficient
    val = plot_corr_size + 2 if compute_mean_regress else plot_corr_size
    metrics.update(
        {"recons_corr": np.mean(corr_coef[val:, :plot_corr_size].diagonal())}
    )
    if compute_mean_regress:
        metrics.update(
            {
                "mean_regress": np.mean(
                    corr_coef[
                        plot_corr_size : plot_corr_size + 2,
                        :plot_corr_size,
                    ].flatten()
                )
            }
        )
    if doplot:
        plt.figure(figsize=(10, 5))
        plt.imshow(corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Correlation Coefficient of expr and i["x"]')
        plt.show()
    return metrics


def default_benchmark(
    model: torch.nn.Module,
    default_dataset: str = "pancreas",
    do_class: bool = True,
    coarse: bool = False,
) -> dict:
    """
    Run the default benchmark for embedding and annotation using the scPRINT model.

    Args:
        model (torch.nn.Module): The scPRINT model to be used for embedding and annotation.
        default_dataset (str, optional): The default dataset to use for benchmarking. Options are "pancreas", "lung", or a path to a dataset. Defaults to "pancreas".
        do_class (bool, optional): Whether to perform classification. Defaults to True.
        coarse (bool, optional): Whether to use coarse cell type annotations. Defaults to False.

    Returns:
        dict: A dictionary containing the benchmark metrics.
    """
    if default_dataset == "pancreas":
        adata = sc.read(
            FILE_LOC + "/../../data/pancreas_atlas.h5ad",
            backup_url="https://figshare.com/ndownloader/files/24539828",
        )
        adata.obs["cell_type_ontology_term_id"] = adata.obs["celltype"].replace(
            COARSE if coarse else FINE
        )
        adata.obs["assay_ontology_term_id"] = adata.obs["tech"].replace(
            COARSE if coarse else FINE
        )
    elif default_dataset == "lung":
        adata = sc.read(
            FILE_LOC + "/../../data/lung_atlas.h5ad",
            backup_url="https://figshare.com/ndownloader/files/24539942",
        )
        adata.obs["cell_type_ontology_term_id"] = adata.obs["cell_type"].replace(
            COARSE if coarse else FINE
        )
    else:
        adata = sc.read_h5ad(default_dataset)
        adata.obs["batch"] = adata.obs["assay_ontology_term_id"]
        adata.obs["cell_type"] = adata.obs["cell_type_ontology_term_id"]
    preprocessor = Preprocessor(
        use_layer="counts",
        is_symbol=True,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
    )
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = preprocessor(adata.copy())
    embedder = Embedder(
        pred_embedding=["cell_type_ontology_term_id"] if do_class else [],
        doclass=(default_dataset not in ["pancreas", "lung"]) and do_class,
        max_len=4000,
        keep_all_cls_pred=False,
        output_expression="none",
    )
    embed_adata, metrics = embedder(model, adata.copy())

    bm = Benchmarker(
        embed_adata,
        batch_key="tech" if default_dataset == "pancreas" else "batch",
        label_key="celltype" if default_dataset == "pancreas" else "cell_type",
        embedding_obsm_keys=["scprint"],
        n_jobs=6,
    )
    bm.benchmark()
    metrics.update({"scib": bm.get_results(min_max_scale=False).T.to_dict()["scprint"]})
    metrics["classif"] = compute_classification(
        embed_adata, model.classes, model.label_decoders, model.labels_hierarchy
    )
    return metrics


def compute_classification(
    adata: AnnData,
    classes: List[str],
    label_decoders: Dict[str, Any],
    labels_hierarchy: Dict[str, Any],
    metric_type: List[str] = ["macro", "micro", "weighted"],
) -> Dict[str, Dict[str, float]]:
    """
    Compute classification metrics for the given annotated data.

    Args:
        adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
        classes (List[str]): List of class labels to be used for classification.
        label_decoders (Dict[str, Any]): Dictionary of label decoders for each class.
        labels_hierarchy (Dict[str, Any]): Dictionary representing the hierarchy of labels.
        metric_type (List[str], optional): List of metric types to compute. Defaults to ["macro", "micro", "weighted"].

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing classification metrics for each class.
    """
    metrics = {}
    for label in classes:
        res = []
        if label not in adata.obs.columns:
            continue
        labels_topred = label_decoders[label].values()
        if label in labels_hierarchy:
            parentdf = (
                bt.CellType.filter()
                .df(include=["parents__ontology_id"])
                .set_index("ontology_id")[["parents__ontology_id"]]
            )
            parentdf.parents__ontology_id = parentdf.parents__ontology_id.astype(str)
            class_groupings = {
                k: get_descendants(k, parentdf) for k in set(adata.obs[label].unique())
            }
        for pred, true in adata.obs[["pred_" + label, label]].values:
            if pred == true:
                res.append(true)
                continue
            if label in labels_hierarchy:
                if true in class_groupings:
                    res.append(true if pred in class_groupings[true] else "")
                    continue
                elif true not in labels_topred:
                    raise ValueError(f"true label {true} not in available classes")
            elif true not in labels_topred:
                raise ValueError(f"true label {true} not in available classes")
            res.append("")
        metrics[label] = {}
        metrics[label]["accuracy"] = np.mean(np.array(res) == adata.obs[label].values)
        for x in metric_type:
            metrics[label][x] = f1_score(
                np.array(res), adata.obs[label].values, average=x
            )
    return metrics


FINE = {
    "gamma": "CL:0002275",
    "beta": "CL:0000169",  # "CL:0008024"
    "epsilon": "CL:0005019",  # "CL:0008024"
    "acinar": "CL:0000622",
    "delta": "CL:0000173",  # "CL:0008024"
    "schwann": "CL:0002573",  # "CL:0000125"
    "activated_stellate": "CL:0000057",
    "alpha": "CL:0000171",  # "CL:0008024"
    "mast": "CL:0000097",
    "Mast cell": "CL:0000097",
    "quiescent_stellate": "CL:0000057",
    "t_cell": "CL:0000084",
    "endothelial": "CL:0000115",
    "Endothelium": "CL:0000115",
    "ductal": "CL:0002079",  # CL:0000068
    "macrophage": "CL:0000235",
    "Macrophage": "CL:0000235",
    "B cell": "CL:0000236",
    "Type 2": "CL:0002063",
    "Type 1": "CL:0002062",
    "Ciliated": "CL:4030034",  # respiratory ciliated
    "Dendritic cell": "CL:0000451",  # leukocyte
    "Ionocytes": "CL:0005006",
    "Basal 1": "CL:0000646",  # epithelial
    "Basal 2": "CL:0000646",
    "Secretory": "CL:0000151",
    "Neutrophil_CD14_high": "CL:0000775",
    "Neutrophils_IL1R2": "CL:0000775",
    "Lymphatic": "CL:0002138",
    "Fibroblast": "CL:0000057",
    "T/NK cell": "CL:0000814",
    "inDrop1": "EFO:0008780",
    "inDrop3": "EFO:0008780",
    "inDrop4": "EFO:0008780",
    "inDrop2": "EFO:0008780",
    "fluidigmc1": "EFO:0010058",  # fluidigm c1
    "smarter": "EFO:0010058",  # fluidigm c1
    "celseq2": "EFO:0010010",
    "smartseq2": "EFO:0008931",
    "celseq": "EFO:0008679",
}
COARSE = {
    "beta": "CL:0008024",  # endocrine
    "epsilon": "CL:0008024",
    "delta": "CL:0008024",
    "alpha": "CL:0008024",
    "gamma": "CL:0008024",
    "acinar": "CL:0000150",  # epithelial (gland)
    "ductal": "CL:0000068",  # epithelial (duct)
    "schwann": "CL:0000125",  # glial
    "endothelial": "CL:0000115",
    "Endothelium": "CL:0000115",
    "Lymphatic": "CL:0000115",
    "macrophage": "CL:0000235",  # myeloid leukocyte (not)
    "Macrophage": "CL:0000235",  # myeloid leukocyte
    "mast": "CL:0000097",  # myeloid leukocyte (not)
    "Mast cell": "CL:0000097",  # myeloid leukocyte
    "Neutrophil_CD14_high": "CL:0000775",  # myeloid leukocyte
    "Neutrophils_IL1R2": "CL:0000775",  # myeloid leukocyte
    "t_cell": "CL:0000084",  # leukocyte, lymphocyte (not)
    "T/NK cell": "CL:0000084",  # leukocyte, lymphocyte (not)
    "B cell": "CL:0000236",  # leukocyte, lymphocyte (not)
    "Dendritic cell": "CL:0000451",  # leukocyte, lymphocyte
    "activated_stellate": "CL:0000057",  # fibroblast (not)
    "quiescent_stellate": "CL:0000057",  # fibroblast (not)
    "Fibroblast": "CL:0000057",
    "Type 2": "CL:0000066",  # epithelial
    "Type 1": "CL:0000066",
    "Ionocytes": "CL:0000066",  # epithelial
    "Basal 1": "CL:0000066",  # epithelial
    "Basal 2": "CL:0000066",
    "Ciliated": "CL:0000064",  # ciliated
    "Secretory": "CL:0000151",
    "inDrop1": "EFO:0008780",
    "inDrop3": "EFO:0008780",
    "inDrop4": "EFO:0008780",
    "inDrop2": "EFO:0008780",
    "fluidigmc1": "EFO:0010058",  # fluidigm c1
    "smarter": "EFO:0010058",  # fluidigm c1
    "celseq2": "EFO:0010010",
    "smartseq2": "EFO:0008931",
    "celseq": "EFO:0008679",
}
