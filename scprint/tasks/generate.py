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
            pred_embedding (List[str], optional): The list of labels to be used for plotting embeddings. Defaults to [ "cell_type_ontology_term_id", "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id", "sex_ontology_term_id", ].
            doclass (bool, optional): Whether to perform classification. Defaults to True.
            doplot (bool, optional): Whether to generate plots. Defaults to True.
            keep_all_labels_pred (bool, optional): Whether to keep all class predictions. Defaults to False, will only keep the most likely class.
            genelist (List[str], optional): The list of genes to be used for embedding. Defaults to []: In this case, "how" needs to be "most var" or "random expr".
            save_every (int, optional): The number of cells to save at a time. Defaults to 100_000.
                This is important to avoid memory issues.
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
            get_knn_cells=model.expr_emb_style == "metacell",
        )
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            how=self.how if self.how != "most var" else "some",
            max_len=self.max_len,
            add_zero_genes=0,
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
        prevplot = model.doplot
        model.doplot = self.doplot and not self.keep_all_labels_pred
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
                model._predict(
                    gene_pos,
                    expression,
                    depth,
                    knn_cells=batch["knn_cells"].to(device)
                    if model.expr_emb_style == "metacell"
                    else None,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=self.save_every,
                    name="embed_" + rand + "_",
                )
                torch.cuda.empty_cache()
        model.log_adata(name="embed_" + rand + "_" + str(model.counter))
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
            allclspred = model.pred.to(device="cpu").numpy()
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
                        elif true != "unknown":
                            res.append(False)
                        elif true not in class_topred:
                            print(f"true label {true} not in available classes")
                            return adata, metrics
                    elif true not in class_topred:
                        print(f"true label {true} not in available classes")
                        return adata, metrics
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
