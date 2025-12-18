import os
from typing import Any, Dict, List, Optional

import bionty as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
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

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class Embedder:
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        how: str = "random expr",
        max_len: int = 2000,
        doclass: bool = True,
        pred_embedding: List[str] = [
            "all",
        ],
        doplot: bool = True,
        keep_all_labels_pred: bool = False,
        genelist: Optional[List[str]] = None,
        save_every: int = 40_000,
        unknown_label: str = "unknown",
        use_knn: bool = True,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            how (str, optional): The method to be used for selecting valid genes. Defaults to "random expr".
                - "random expr": random expression
                - "most var": highly variable genes in the dataset
                - "some": specific genes (from genelist)
                - "most expr": most expressed genes in the cell
            max_len (int, optional): The maximum length of the gene sequence given to the model. Defaults to 1000.
            doclass (bool, optional): Whether to perform classification. Defaults to True.
            pred_embedding (List[str], optional): The list of labels to be used for plotting embeddings. Defaults to [ "cell_type_ontology_term_id", "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id", "sex_ontology_term_id", ].
            doplot (bool, optional): Whether to generate plots. Defaults to True.
            keep_all_labels_pred (bool, optional): Whether to keep all class predictions. Defaults to False, will only keep the most likely class.
            genelist (List[str], optional): The list of genes to be used for embedding. Defaults to []: In this case, "how" needs to be "most var" or "random expr".
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
                This is important to avoid memory issues.
            unknown_label (str, optional): The label to be used for unknown cell types. Defaults to "unknown".
            use_knn (bool, optional): Whether to use k-nearest neighbors information. Defaults to True.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.pred_embedding = pred_embedding
        self.keep_all_labels_pred = keep_all_labels_pred
        self.doplot = doplot
        self.doclass = doclass
        self.genelist = genelist if genelist is not None else []
        self.save_every = save_every
        self.pred = None
        self.unknown_label = unknown_label
        self.use_knn = use_knn

    def __call__(self, model: torch.nn.Module, adata: AnnData) -> tuple[AnnData, dict]:
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
            dict: classification metrics results when some ground truth information was available in the anndata.
        """
        # one of "all" "sample" "none"
        model.predict_mode = "none"
        self.pred = None
        prevkeep = model.keep_all_labels_pred
        model.keep_all_labels_pred = self.keep_all_labels_pred
        # Add at least the organism you are working with
        if self.how == "most var":
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=self.max_len
            )
            self.genelist = adata.var.index[adata.var.highly_variable]
        adataset = SimpleAnnDataset(
            adata,
            obs_to_output=["organism_ontology_term_id"],
            get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
        )
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            how=self.how if self.how != "most var" else "some",
            max_len=self.max_len,
            add_zero_genes=0,
            genelist=self.genelist if self.how in ["most var", "some"] else [],
            n_bins=model.n_input_bins if model.expr_emb_style == "binned" else 0,
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
        prevplot = model.doplot
        model.pred_log_adata = True
        model.doplot = self.doplot and not self.keep_all_labels_pred
        model.save_expr = False
        rand = random_str()
        dtype = (
            torch.float16
            if isinstance(model.transformer, FlashTransformer)
            else model.dtype
        )
        with (
            torch.no_grad(),
            torch.autocast(device_type=device, dtype=dtype),
        ):
            for batch in tqdm(dataloader):
                gene_pos, expression, depth = (
                    batch["genes"].to(device),
                    batch["x"].to(device),
                    batch["depth"].to(device),
                )
                pred = model._predict(
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
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=self.save_every,
                    name="embed_" + rand + "_",
                )
                torch.cuda.empty_cache()
                if self.keep_all_labels_pred:
                    if pred is not None:
                        self.pred = (
                            pred if self.pred is None else torch.cat([self.pred, pred])
                        )
        model.log_adata(name="embed_" + rand + "_" + str(model.counter))

        model.pos = None
        model.expr_pred = None
        model.embs = None
        if self.keep_all_labels_pred:
            self.pred = (
                model.pred if self.pred is None else torch.cat([self.pred, model.pred])
            )
        model.pred = None
        model.save_expr = True
        try:
            mdir = (
                model.logger.save_dir if model.logger.save_dir is not None else "data"
            )
        except:
            mdir = "data"
        pred_adata = []
        del adataset, dataloader
        for i in range(model.counter + 1):
            file = (
                mdir
                + "/step_"
                + str(model.global_step)
                + "_"
                + model.name
                + "_embed_"
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
        pred_adata.obs.index = adata.obs.index

        try:
            adata.obsm["X_scprint_umap"] = pred_adata.obsm["X_umap"]
        except:
            print("too few cells to embed into a umap")
        try:
            adata.obs["scprint_leiden"] = pred_adata.obs["scprint_leiden"]
        except:
            print("too few cells to compute a clustering")

        if self.pred_embedding == ["all"]:
            pred_embedding = ["other"] + model.classes
        else:
            pred_embedding = self.pred_embedding
        if len(pred_embedding) == 1:
            adata.obsm["scprint_emb"] = pred_adata.obsm[
                "scprint_emb_" + pred_embedding[0]
            ].astype(np.float32)

        else:
            adata.obsm["scprint_emb"] = np.zeros(
                pred_adata.obsm["scprint_emb_" + pred_embedding[0]].shape,
                dtype=np.float32,
            )
            i = 0
            for k, v in pred_adata.obsm.items():
                adata.obsm[k] = v.astype(np.float32)
                if model.compressor is not None:
                    if i == 0:
                        adata.obsm["scprint_emb"] = v.astype(np.float32)
                    else:
                        adata.obsm["scprint_emb"] = np.hstack(
                            [adata.obsm["scprint_emb"], v.astype(np.float32)]
                        )
                else:
                    adata.obsm["scprint_emb"] += v.astype(np.float32)
                i += 1
            if model.compressor is None:
                adata.obsm["scprint_emb"] = adata.obsm["scprint_emb"] / i

        for key, value in pred_adata.uns.items():
            adata.uns[key] = value

        pred_adata.obs.index = adata.obs.index
        model.keep_all_labels_pred = prevkeep
        model.doplot = prevplot
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)
        del pred_adata
        if self.keep_all_labels_pred:
            allclspred = self.pred.to(device="cpu").numpy()
            columns = []
            for cl in model.classes:
                n = model.label_counts[cl]
                columns += [model.label_decoders[cl][i] for i in range(n)]
            allclspred = pd.DataFrame(
                allclspred, columns=columns, index=adata.obs.index
            )
            adata.obs = pd.concat([adata.obs, allclspred], axis=1)

        metrics = {}
        if self.doclass and not self.keep_all_labels_pred:
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
                        elif true != self.unknown_label:
                            res.append(False)
                        elif true not in class_topred:
                            print(f"true label {true} not in available classes")
                            return adata, metrics
                    elif true not in class_topred:
                        print(f"true label {true} not in available classes")
                        return adata, metrics
                    elif true != self.unknown_label:
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
        self.pred = None
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
    folder_dir: str = FILE_LOC + "/../../data/",
    dataset: str = FILE_LOC + "/../../data/gNNpgpo6gATjuxTE7CCp.h5ad",
    do_class: bool = True,
    coarse: bool = False,
) -> dict:
    """
    Run the default benchmark for embedding and annotation using the scPRINT model.

    Args:
        model (torch.nn.Module): The scPRINT model to be used for embedding and annotation.
        folder_dir (str, optional): The directory containing data files.
        dataset (str, optional): The dataset to use for benchmarking. Can be a path or URL.
        do_class (bool, optional): Whether to perform classification. Defaults to True.
        coarse (bool, optional): Whether to use coarse cell type annotations. Defaults to False.
    
    Returns:
        dict: A dictionary containing the benchmark metrics.
    """
    if dataset.startswith("https://"):
        adata = sc.read(
            folder_dir
            + dataset.split("/")[-1]
            + (".h5ad" if not dataset.endswith(".h5ad") else ""),
            backup_url=dataset,
        )
    else:
        adata = sc.read_h5ad(dataset)
    if adata.shape[0] > 100_000:
        adata = adata[
            adata.obs_names[np.random.choice(adata.shape[0], 100_000, replace=False)]
        ]
    max_len = 4000 if adata.X.sum(1).mean() < 50_000 else 8000
    batch_size = 64 if adata.X.sum(1).mean() < 50_000 else 32
    log_every = 10_000
    if dataset.split("/")[-1] in ["24539942", "24539828"]:  # lung and pancreas
        adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
        use_layer = "counts"
        is_symbol = True
        batch_key = "tech" if dataset.split("/")[-1] == "24539828" else "batch"
        label_key = "celltype" if dataset.split("/")[-1] == "24539828" else "cell_type"
        adata.obs["cell_type_ontology_term_id"] = adata.obs[label_key].replace(
            COARSE if coarse else FINE
        )
        adata.obs["assay_ontology_term_id"] = adata.obs[batch_key].replace(
            COARSE if coarse else FINE
        )
    else:
        use_layer = None
        is_symbol = False
        batch_key = (
            "batch"
            if dataset.split("/")[-1] == "661d5ec2-ca57-413c-8374-f49b0054ddba.h5ad"
            else "assay_ontology_term_id"
        )
        label_key = "cell_type_ontology_term_id"
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
        sc.pp.neighbors(adata, use_rep="X_pca")
    embedder = Embedder(
        pred_embedding=(
            model.pred_embedding if model.pred_embedding is not None else ["all"]
        ),
        doclass=do_class,
        max_len=max_len,
        doplot=False,
        keep_all_labels_pred=False,
        save_every=log_every,
        batch_size=batch_size,
        how="random expr",
    )
    adata, metrics = embedder(model, adata)

    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=["scprint_emb"],
    )
    bm.benchmark()
    metrics.update(
        {"scib": bm.get_results(min_max_scale=False).T.to_dict()["scprint_emb"]}
    )
    if model.class_scale > 0:
        metrics["classif"] = compute_classification(
            adata, model.classes, model.label_decoders, model.labels_hierarchy
        )
    return metrics


def compute_classification(
    adata: AnnData,
    classes: List[str],
    label_decoders: Dict[str, Any],
    labels_hierarchy: Dict[str, Any],
    metric_type: List[str] = ["macro", "micro", "weighted"],
    use_unknown: bool = False,
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
    for clss in classes:
        res = []
        if clss not in adata.obs.columns:
            print("not in columns")
            continue
        labels_topred = label_decoders[clss].values()
        if clss in labels_hierarchy:
            parentdf = (
                bt.CellType.filter()
                .df(include=["parents__ontology_id", "ontology_id"])
                .set_index("ontology_id")[["parents__ontology_id"]]
            )
            parentdf.parents__ontology_id = parentdf.parents__ontology_id.astype(str)
            class_groupings = {
                k: get_descendants(k, parentdf) for k in set(adata.obs[clss].unique())
            }
        tokeep = np.array([True] * adata.shape[0])
        for i, (pred, true) in enumerate(adata.obs[["pred_" + clss, clss]].values):
            if pred == true:
                res.append(true)
                continue
            if true == "unknown":
                tokeep[i] = False
            if clss in labels_hierarchy:
                if true in class_groupings:
                    if pred == "unknown" and not use_unknown:
                        tokeep[i] = False
                    res.append(true if pred in class_groupings[true] else "")
                    continue
                elif true not in labels_topred:
                    raise ValueError(f"true label {true} not in available classes")
            elif true not in labels_topred:
                raise ValueError(f"true label {true} not in available classes")
            res.append("")
        metrics[clss] = {}
        metrics[clss]["accuracy"] = np.mean(
            np.array(res)[tokeep] == adata.obs[clss].values[tokeep]
        )
        for x in metric_type:
            metrics[clss][x] = f1_score(
                np.array(res)[tokeep], adata.obs[clss].values[tokeep], average=x
            )
    return metrics


def get_top_labels(model, adata, k=3):
    s = list(model.label_counts.values())
    out = []
    count = 0
    for l, label in enumerate(model.label_counts.keys()):
        if -sum(s) + count + s[l] == 0:
            ct = adata.obs.iloc[:, -sum(s) + count :]
        else:
            ct = adata.obs.iloc[:, -sum(s) + count : -sum(s) + s[l] + count]
        if s[l] < 3:
            # append only the label corresponding to the maximum score (arg-max)
            best_labels = [
                model.label_decoders[label][idx]
                for idx in np.argmax(ct.to_numpy(), axis=1)
            ]
            out.append(
                pd.DataFrame(
                    {label: best_labels}, index=adata.obs_names, columns=[label]
                )
            )
        else:
            res = []
            for j, m in enumerate(np.argsort(ct)[:, ::-1]):
                certainty = ct.iloc[j, m[0]]
                best = model.label_decoders[label][m[0]]
                other = []
                for i in range(1, k):
                    if ct.iloc[j, m[i]] > certainty - 0.15:
                        other.append(model.label_decoders[label][m[i]])
                    else:
                        other.append(None)
                res.append([best] + [certainty] + other)
            out.append(
                pd.DataFrame(
                    res,
                    columns=[label, label + "_certainty"]
                    + [label + "_choice" + str(i) for i in range(1, k)],
                    index=adata.obs_names,
                )
            )
        count += s[l]  # keep the offset consistent with the else branch
    out = pd.concat(out, axis=1)
    return out


def find_coarser_labels(out, model):
    relabel = {}
    for label in model.classes:
        if label in model.labels_hierarchy.keys():
            decoder = {k: v for k, v in model.label_decoders[label].items()}
            ct_hier = {
                decoder[k]: [decoder[u] for u in v]
                for k, v in model.labels_hierarchy[label].items()
            }
            relabel[label] = {}
            for i, r in enumerate(
                out[~out[label + "_choice1"].isna()][
                    [label, label + "_choice1", label + "_choice2"]
                ].values
            ):
                prev_v = 100_000
                res = None
                if r[1] is not None:
                    for k, v in ct_hier.items():
                        if r[2] is not None:
                            if (
                                r[0] in v
                                and r[1] in v
                                and r[2] in v
                                and k != "CL:0000000"
                                and k != "None"
                            ):
                                if len(v) < prev_v:
                                    res = k
                                    prev_v = len(v)
                        else:
                            if (
                                r[0] in v
                                and r[1] in v
                                and k != "CL:0000000"
                                and k != "None"
                            ):
                                if len(v) < prev_v:
                                    res = k
                                    prev_v = len(v)
                    if res is not None:
                        relabel[label].update({i: res})
    return relabel


def display_confusion_matrix(
    nadata, pred="conv_pred_cell_type_ontology_term_id", true="cell_type"
):
    """
    Display the confusion matrix for true vs predicted cell types.

    Args:
        nadata (AnnData): Annotated data object containing predictions and ground truth.
        pred (str): Column name for predictions. Defaults to "conv_pred_cell_type_ontology_term_id".
        true (str): Column name for ground truth. Defaults to "cell_type".
    """
    counts = None
    for k, v in nadata.obs[true].value_counts().items():
        name = k + " - " + str(v)
        if counts is None:
            counts = pd.DataFrame(
                nadata.obs.loc[
                    nadata.obs[true] == k,
                    pred,
                ].value_counts()
            ).rename(columns={"count": name})
        else:
            counts = pd.concat(
                [
                    counts,
                    pd.DataFrame(
                        nadata.obs.loc[
                            nadata.obs[true] == k,
                            pred,
                        ].value_counts(),
                    ).rename(columns={"count": name}),
                ],
                axis=1,
            )
    counts = counts.T
    # Fill NaN values with 0 for visualization
    counts_filled = counts.fillna(0)

    # Create the heatmap
    plt.figure(figsize=(12, 10))

    # Convert to percentages (row-wise normalization)
    counts_percentage = counts_filled.div(counts_filled.sum(axis=1), axis=0) * 100
    counts_percentage = counts_percentage.iloc[:, counts_percentage.values.max(0) > 5]

    ax = sns.heatmap(
        counts_percentage,
        cmap="Blues",
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.5,
        square=True,
    )
    # place the x-label on top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    plt.title(
        "Confusion Matrix: " + true + " vs " + pred + " (Percentage)",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel(pred, fontsize=12)
    ax.set_ylabel(true + " (with counts)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    plt.tight_layout()
    plt.show()


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
