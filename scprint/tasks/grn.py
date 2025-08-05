import gc
import os.path
from typing import Any, List, Optional

import hdbscan
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import seaborn as sns
import sparse
import torch
import umap
from anndata import AnnData
from anndata.utils import make_index_unique
from bengrn import BenGRN, get_perturb_gt, get_sroy_gt
from bengrn.base import train_classifier
from grnndata import GRNAnnData, from_anndata, read_h5ad
from grnndata import utils as grnutils
from matplotlib import pyplot as plt
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from scdataloader.utils import load_genes
from scprint.utils.sinkhorn import SinkhornDistance

from .tmfg import tmfg

FILEDIR = os.path.dirname(os.path.realpath(__file__))


class GNInfer:
    def __init__(
        self,
        layer: Optional[List[int]] = None,
        batch_size: int = 64,
        num_workers: int = 8,
        drop_unexpressed: bool = False,
        num_genes: int = 3000,
        precision: str = "16-mixed",
        cell_type_col: str = "cell_type",
        how: str = "random expr",  # random expr, most var within, most var across, given
        max_len: int = 3000,
        preprocess: str = "softmax",  # sinkhorn, softmax, none
        head_agg: str = "mean",  # mean, sum, none
        filtration: str = "thresh",  # thresh, top-k, mst, known, none
        k: int = 10,
        apc: bool = False,
        known_grn: Optional[any] = None,
        symmetrize: bool = False,
        doplot: bool = True,
        comp_attn: bool = True,
        max_cells: int = 0,
        forward_mode: str = "none",
        genes: List[str] = [],
        loc: str = "./",
        dtype: torch.dtype = torch.float16,
        locname: str = "",
        add_emb_in_model: bool = False,
    ):
        """
        GNInfer a class to infer gene regulatory networks from a dataset using a scPRINT model.

        Args:
            layer (Optional[list[int]], optional): List of layers to use for the inference. Defaults to None.
            batch_size (int, optional): Batch size for processing. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 8.
            drop_unexpressed (bool, optional): Whether to drop unexpressed genes. Defaults to False.
            num_genes (int, optional): Number of genes to consider. Defaults to 3000.
            precision (str, optional): Precision type for computations. Defaults to "16-mixed".
            cell_type_col (str, optional): Column name for cell type information. Defaults to "cell_type".
            how (str, optional): Method to select genes. Options are "random expr", "most var within", "most var across", "given". Defaults to "random expr".
            preprocess (str, optional): Preprocessing method. Options are "softmax", "sinkhorn", "none". Defaults to "softmax".
            head_agg (str, optional): Aggregation method for heads. Options are "mean", "sum", "none". Defaults to "mean".
            filtration (str, optional): Filtration method for the adjacency matrix. Options are "thresh", "top-k", "mst", "known", "none". Defaults to "thresh".
            k (int, optional): Number of top connections to keep if filtration is "top-k". Defaults to 10.
            apc (bool, optional): Whether to apply Average Product Correction. Defaults to False.
            known_grn (optional): Known gene regulatory network to use as a reference. Defaults to None.
            symmetrize (bool, optional): Whether to symmetrize the adjacency matrix. Defaults to False.
            doplot (bool, optional): Whether to generate plots. Defaults to True.
            max_cells (int, optional): Maximum number of cells to consider. Defaults to 0.
            forward_mode (str, optional): Mode for forward pass. Defaults to "none".
            genes (list, optional): List of genes to consider. Defaults to an empty list.
            loc (str, optional): Location to save results. Defaults to "./".
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
            locname (str, optional): Name for the location. Defaults to an empty string.
            add_emb_in_model (bool, optional): Whether to add cell embeddings in the grn. Defaults to False.

        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.layer = layer
        self.locname = locname
        self.how = how
        assert (
            self.how
            in [
                "most var within",
                "most var across",
                "random expr",
                "given",
                "most expr",
            ]
        ), "how must be one of 'most var within', 'most var across', 'random expr', 'given', 'most expr'"
        self.num_genes = num_genes
        self.preprocess = preprocess
        self.cell_type_col = cell_type_col
        self.filtration = filtration
        self.doplot = doplot
        self.genes = genes
        self.apc = apc
        self.dtype = dtype
        self.forward_mode = forward_mode
        self.k = k
        self.max_len = max_len
        self.symmetrize = symmetrize
        self.known_grn = known_grn
        self.head_agg = head_agg
        self.max_cells = max_cells
        self.curr_genes = None
        self.drop_unexpressed = drop_unexpressed
        self.precision = precision
        self.comp_attn = comp_attn
        self.add_emb_in_model = add_emb_in_model
        if self.filtration != "none" and self.head_agg == "none":
            raise ValueError("filtration must be 'none' when head_agg is 'none'")

    def __call__(self, model: torch.nn.Module, adata: AnnData, cell_type=None):
        """
        __call__ runs the method

        Args:
            model (torch.nn.Module): The model to be used for generating the network
            adata (AnnData): Annotated data matrix of shape `n_obs` × `n_vars`. `n_obs` is the number of cells and `n_vars` is the number of genes.
            cell_type (str, optional): Specific cell type to filter the data. Defaults to None.

        Returns:
            AnnData: Annotated data matrix with predictions and annotations.
            np.ndarray: Filtered adjacency matrix.
        """
        # Add at least the organism you are working with
        if self.layer is None:
            self.layer = list(range(model.nlayers))
        self.n_cell_embs = model.attn.additional_tokens
        subadata = self.predict(model, adata, self.layer, cell_type)
        adjacencies = self.aggregate(model.attn.get(), model.genes)
        if self.head_agg == "none":
            return self.save(
                adjacencies[self.n_cell_embs :, self.n_cell_embs :, :],
                subadata,
            )
        else:
            return self.save(
                self.filter(adjacencies)[self.n_cell_embs :, self.n_cell_embs :],
                subadata,
            )

    def predict(self, model, adata, layer, cell_type=None):
        self.curr_genes = None
        model.pred_log_adata = False
        if cell_type is not None:
            subadata = adata[adata.obs[self.cell_type_col] == cell_type].copy()
        else:
            subadata = adata.copy()
        if self.how == "most var within":
            sc.pp.highly_variable_genes(
                subadata, flavor="seurat_v3", n_top_genes=self.num_genes
            )
            self.curr_genes = (
                subadata.var.index[subadata.var.highly_variable].tolist() + self.genes
            )
            print(
                "number of expressed genes in this cell type: "
                + str((subadata.X.sum(0) > 1).sum())
            )
        elif self.how == "most var across" and cell_type is not None:
            sc.tl.rank_genes_groups(
                adata,
                mask_var=adata.var.index.isin(model.genes),
                groupby=self.cell_type_col,
                groups=[cell_type],
            )
            diff_expr_genes = adata.uns["rank_genes_groups"]["names"][cell_type]
            diff_expr_genes = [gene for gene in diff_expr_genes if gene in model.genes]
            self.curr_genes = diff_expr_genes[: self.num_genes] + self.genes
            self.curr_genes.sort()
        elif self.how == "random expr":
            self.curr_genes = model.genes
            # raise ValueError("cannot do it yet")
            pass
        elif self.how == "given" and len(self.genes) > 0:
            self.curr_genes = self.genes
        elif self.how == "most expr":
            self.curr_genes = adata.var.index[
                adata.X.sum(0).A1.argsort()[::-1]
            ].tolist()[: self.num_genes]
        else:
            raise ValueError(
                "how must be one of 'most var', 'random expr', 'most expr'"
            )
        if self.drop_unexpressed:
            expr = subadata.var[(subadata.X.sum(0) > 0).tolist()[0]].index.tolist()
            self.curr_genes = [i for i in self.curr_genes if i in expr]
        # Order cells by total count
        cell_sums = subadata.X.sum(axis=1)
        order = np.argsort(
            -cell_sums.A1 if scipy.sparse.issparse(subadata.X) else -cell_sums
        )
        subadata = subadata[order].copy()
        subadata = subadata[: self.max_cells] if self.max_cells else subadata
        if len(subadata) == 0:
            raise ValueError("no cells in the dataset")
        adataset = SimpleAnnDataset(
            subadata, obs_to_output=["organism_ontology_term_id"]
        )
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            max_len=self.max_len if self.how == "random expr" else 0,
            how="some" if self.how != "random expr" else "random expr",
            genelist=self.curr_genes if self.how != "random expr" else [],
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        model.attn.comp_attn = self.head_agg == "mean_full"
        model.doplot = self.doplot
        model.on_predict_epoch_start()
        model.eval()
        device = model.device.type

        # reparametrize the attn process
        model.attn.comp_attn = self.comp_attn
        if self.how != "random expr":
            if self.num_genes > 10_000 and not self.comp_attn:
                raise ValueError("need less genes for a non-shared-qk version")
            if not self.comp_attn:
                model.attn.gene_dim = (
                    len(set(self.curr_genes) & set(model.genes))
                    + model.attn.additional_tokens
                )
                model.attn.apply_softmax = self.preprocess == "softmax"
        elif not self.comp_attn:
            raise ValueError(
                "full attention (i.e. comp_attn=Fale) is not supported for random expr"
            )
        with torch.no_grad(), torch.autocast(device_type=device, dtype=self.dtype):
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
                    predict_mode=self.forward_mode,
                    keep_output=False,
                    get_attention_layer=layer if type(layer) is list else [layer],
                )
            torch.cuda.empty_cache()
                
        return subadata

    def aggregate(self, attn, genes):
        if self.head_agg == "mean_full" or not self.comp_attn:
            self.curr_genes = [i for i in genes if i in self.curr_genes]
            return attn.detach().cpu().numpy()
        badloc = torch.isnan(attn.sum((0, 2, 3, 4)))
        attn = attn[:, ~badloc, :, :, :]
        badloc = badloc.detach().cpu().numpy()
        self.curr_genes = (
            np.array(self.curr_genes)[~badloc[self.n_cell_embs :]]
            if self.how == "random expr"
            else [i for i in genes if i in self.curr_genes]
        )
        if self.doplot:
            sns.set_theme(
                style="white", context="poster", rc={"figure.figsize": (14, 10)}
            )
            fit = umap.UMAP()
            mm = fit.fit_transform(attn[0, :, 0, 0, :].detach().cpu().numpy())
            labels = hdbscan.HDBSCAN(
                min_samples=10,
                min_cluster_size=100,
            ).fit_predict(mm)
            plt.scatter(mm[:, 0], mm[:, 1], c=labels)
            plt.title(f"Qs @H{0}")
            plt.show()
            mm = fit.fit_transform(attn[0, :, 1, 0, :].detach().cpu().numpy())
            labels = hdbscan.HDBSCAN(
                min_samples=10,
                min_cluster_size=100,
            ).fit_predict(mm)
            plt.scatter(mm[:, 0], mm[:, 1], c=labels)
            plt.title(f"Ks @H{0}")
            plt.show()
        # attn = attn[:, :, 0, :, :].permute(0, 2, 1, 3) @ attn[:, :, 1, :, :].permute(
        #    0, 2, 3, 1
        # )
        attns = None
        Qs = (
            attn[:, :, 0, :, :]
            .permute(0, 2, 1, 3)
            .reshape(-1, attn.shape[1], attn.shape[-1])
        )
        Ks = (
            attn[:, :, 1, :, :]
            .permute(0, 2, 1, 3)
            .reshape(-1, attn.shape[1], attn.shape[-1])
        )
        for i in range(Qs.shape[0]):
            attn = Qs[i] @ Ks[i].T
            # return attn
            scale = Qs.shape[-1] ** -0.5
            attn = attn * scale
            if self.preprocess == "sinkhorn":
                if attn.numel() > 100_000_000:
                    raise ValueError("you can't sinkhorn such a large matrix")
                sink = SinkhornDistance(0.1, max_iter=200)
                attn = sink(attn)[0]
                attn = attn * Qs.shape[-1]
            elif self.preprocess == "softmax":
                attn = torch.nn.functional.softmax(attn, dim=-1)
            elif self.preprocess == "none":
                pass
            else:
                raise ValueError(
                    "preprocess must be one of 'sinkhorn', 'softmax', 'none'"
                )

            if self.symmetrize:
                attn = (attn + attn.T) / 2
            if self.apc:
                pass
                # attn = attn - (
                #    (attn.sum(-1).unsqueeze(-1) * attn.sum(-2).unsqueeze(-2))
                #    / attn.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
                # )  # .view()
            if self.head_agg == "mean":
                attns = attn + (attns if attns is not None else 0)
            elif self.head_agg == "max":
                attns = torch.max(attn, attns) if attns is not None else attn
            elif self.head_agg == "none":
                attn = attn.reshape(attn.shape[0], attn.shape[1], 1)
                if attns is not None:
                    attns = torch.cat((attns, attn.detach().cpu()), dim=2)
                else:
                    attns = attn.detach().cpu()
            else:
                raise ValueError("head_agg must be one of 'mean', 'max' or 'None'")
        if self.head_agg == "mean":
            attns = attns / Qs.shape[0]
        return (
            attns.detach().cpu().numpy() if self.head_agg != "none" else attns.numpy()
        )

    def filter(self, adj, gt=None):
        if self.filtration == "thresh":
            adj[adj < (1 / adj.shape[-1])] = 0
            res = (adj != 0).sum()
            if res / adj.shape[0] ** 2 < 0.01:
                adj = scipy.sparse.csr_matrix(adj)
        elif self.filtration == "none":
            pass
        elif self.filtration == "top-k":
            args = np.argsort(adj)
            adj[np.arange(adj.shape[0])[:, None], args[:, : -self.k]] = 0
            adj = scipy.sparse.csr_matrix(adj)
        elif self.filtration == "known" and gt is not None:
            gt = gt.reindex(sorted(gt.columns), axis=1)
            gt = gt.reindex(sorted(gt.columns), axis=0)
            gt = gt[gt.index.isin(self.curr_genes)].iloc[
                :, gt.columns.isin(self.curr_genes)
            ]

            loc = np.isin(self.curr_genes, gt.index)
            self.curr_genes = np.array(self.curr_genes)[loc]
            adj = adj[self.n_cell_embs :, self.n_cell_embs :][loc][:, loc]
            adj[gt.values != 1] = 0
            adj = scipy.sparse.csr_matrix(adj)
        elif self.filtration == "tmfg":
            adj = nx.to_scipy_sparse_array(tmfg(adj))
        elif self.filtration == "mst":
            pass
        else:
            raise ValueError("filtration must be one of 'thresh', 'none' or 'top-k'")
        res = (adj != 0).sum() if self.filtration != "none" else adj.shape[0] ** 2
        print(f"avg link count: {res}, sparsity: {res / adj.shape[0] ** 2}")
        return adj

    def save(self, grn, subadata, loc=""):
        grn = GRNAnnData(
            subadata[:, subadata.var.index.isin(self.curr_genes)].copy(), grn=grn
        )
        # grn = grn[:, (grn.X != 0).sum(0) > (self.max_cells / 32)]
        grn.var["TFs"] = [
            True if i in grnutils.TF else False for i in grn.var["symbol"]
        ]
        grn.uns["grn_scprint_params"] = {
            "filtration": self.filtration,
            "how": self.how,
            "preprocess": self.preprocess,
            "head_agg": self.head_agg,
        }
        if loc != "":
            grn.write_h5ad(loc + "grn_fromscprint.h5ad")
            return from_anndata(grn)
        else:
            return grn


def default_benchmark(
    model: Any,
    default_dataset: str = "sroy",
    cell_types: List[str] = [],
    maxlayers: int = 16,
    maxgenes: int = 5000,
    batch_size: int = 32,
    maxcells: int = 1024,
):
    """
    default_benchmark function to run the default scPRINT GRN benchmark

    Args:
        model (Any): The scPRINT model to be used for the benchmark.
        default_dataset (str, optional): The default dataset to use for benchmarking. Defaults to "sroy".
        cell_types (List[str], optional): List of cell types to include in the benchmark. Defaults to [
        ].
        maxlayers (int, optional): Maximum number of layers to use from the model. Defaults to 16.
        maxgenes (int, optional): Maximum number of genes to consider. Defaults to 5000.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        maxcells (int, optional): Maximum number of cells to consider. Defaults to 1024.

    Returns:
        dict: A dictionary containing the benchmark metrics.
    """
    metrics = {}
    layers = list(range(model.nlayers))[max(0, model.nlayers - maxlayers) :]
    clf_omni = None
    if default_dataset == "sroy":
        preprocessor = Preprocessor(
            is_symbol=True,
            force_preprocess=True,
            skip_validate=True,
            do_postp=False,
            min_valid_genes_id=5000,
            min_dataset_size=64,
        )
        clf_self = None
        todo = [
            ("han", "human", "full"),
            ("mine", "human", "full"),
            ("han", "human", "chip"),
            ("han", "human", "ko"),
            ("tran", "mouse", "full"),
            ("zhao", "mouse", "full"),
            ("tran", "mouse", "chip"),
            ("tran", "mouse", "ko"),
        ]
        for da, spe, gt in todo:
            if gt != "full":
                continue
            if "NCBITaxon:10090" not in model.organisms and spe == "mouse":
                continue
            print(da + "_" + gt)
            preadata = get_sroy_gt(get=da, species=spe, gt=gt)
            adata = preprocessor(preadata.copy())
            grn_inferer = GNInfer(
                layer=layers,
                how="most var within",
                preprocess="softmax",
                head_agg="none",
                filtration="none",
                forward_mode="none",
                num_genes=maxgenes,
                num_workers=8,
                max_cells=maxcells,
                doplot=False,
                batch_size=batch_size,
            )
            grn = grn_inferer(model, adata)
            grn.varp["all"] = grn.varp["GRN"]
            grn.var["ensembl_id"] = grn.var.index
            grn.var["symbol"] = make_index_unique(grn.var["symbol"].astype(str))
            grn.var.index = grn.var["symbol"]
            grn.varp["GRN"] = grn.varp["all"].mean(-1).T
            metrics["mean_" + da + "_" + gt] = BenGRN(
                grn, do_auc=True, doplot=False
            ).compare_to(other=preadata)
            grn.varp["GRN"] = grn.varp["GRN"].T
            if spe == "human":
                metrics["mean_" + da + "_" + gt + "_base"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).scprint_benchmark()

            ## OMNI
            if clf_omni is None:
                grn.varp["GRN"] = grn.varp["all"]
                _, m, clf_omni = train_classifier(
                    grn,
                    C=1,
                    train_size=0.9,
                    class_weight={1: 800, 0: 1},
                    shuffle=True,
                    return_full=False,
                )
                joblib.dump(clf_omni, "clf_omni.pkl")
                metrics["omni_classifier"] = m
            coef = clf_omni.coef_[0] if clf_omni.coef_.shape[0] == 1 else clf_omni.coef_
            grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1)
            if spe == "human":
                metrics["omni_" + da + "_" + gt + "_base"] = BenGRN(
                    grn, do_auc=True, doplot=True
                ).scprint_benchmark()
            grn.varp["GRN"] = grn.varp["GRN"].T
            metrics["omni_" + da + "_" + gt] = BenGRN(
                grn, do_auc=True, doplot=False
            ).compare_to(other=preadata)

            ## SELF
            if clf_self is None:
                grn.varp["GRN"] = np.transpose(grn.varp["all"], (1, 0, 2))
                _, m, clf_self = train_classifier(
                    grn,
                    other=preadata,
                    C=1,
                    train_size=0.5,
                    class_weight={1: 40, 0: 1},
                    shuffle=False,
                    return_full=False,
                )
                metrics["self_classifier"] = m
            coef = clf_self.coef_[0] if clf_self.coef_.shape[0] == 1 else clf_self.coef_
            grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
            metrics["self_" + da + "_" + gt] = BenGRN(
                grn, do_auc=True, doplot=False
            ).compare_to(other=preadata)
            if spe == "human":
                grn.varp["GRN"] = grn.varp["GRN"].T
                metrics["self_" + da + "_" + gt + "_base"] = BenGRN(
                    grn, do_auc=True, doplot=True
                ).scprint_benchmark()

            ## chip / ko
            if (da, spe, "chip") in todo:
                preadata = get_sroy_gt(get=da, species=spe, gt="chip")
                grn.varp["GRN"] = grn.varp["all"].mean(-1).T
                metrics["mean_" + da + "_" + "chip"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
                metrics["omni_" + da + "_" + "chip"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)

                grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
                metrics["self_" + da + "_" + "chip"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
            if (da, spe, "ko") in todo:
                preadata = get_sroy_gt(get=da, species=spe, gt="ko")
                grn.varp["GRN"] = grn.varp["all"].mean(-1).T
                metrics["mean_" + da + "_" + "ko"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
                metrics["omni_" + da + "_" + "ko"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
                metrics["self_" + da + "_" + "ko"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
            del grn
    elif default_dataset == "gwps":
        if not os.path.exists(FILEDIR + "/../../data/perturb_gt.h5ad"):
            adata = get_perturb_gt()
            adata.write_h5ad(FILEDIR + "/../../data/perturb_gt.h5ad")
        else:
            adata = read_h5ad(FILEDIR + "/../../data/perturb_gt.h5ad")
        preprocessor = Preprocessor(
            force_preprocess=True,
            skip_validate=True,
            do_postp=False,
            min_valid_genes_id=maxgenes,
            min_dataset_size=64,
        )
        nadata = preprocessor(adata.copy())
        adata.var["isTF"] = False
        adata.var.loc[adata.var.gene_name.isin(grnutils.TF), "isTF"] = True
        adata.var["isTF"].sum()
        grn_inferer = GNInfer(
            layer=layers,
            how="most var within",
            preprocess="softmax",
            head_agg="none",
            filtration="none",
            forward_mode="none",
            num_genes=maxgenes,
            max_cells=maxcells,
            doplot=False,
            num_workers=8,
            batch_size=batch_size,
        )
        grn = grn_inferer(model, nadata)
        grn.varp["all"] = grn.varp["GRN"]

        grn.varp["GRN"] = grn.varp["all"].mean(-1).T
        metrics["mean"] = BenGRN(grn, do_auc=True, doplot=False).compare_to(other=adata)
        grn.var["ensembl_id"] = grn.var.index
        grn.var.index = grn.var["symbol"]
        grn.varp["GRN"] = grn.varp["all"].mean(-1)
        metrics["mean_base"] = BenGRN(
            grn, do_auc=True, doplot=False
        ).scprint_benchmark()

        grn.varp["GRN"] = grn.varp["all"]
        grn.var.index = grn.var["ensembl_id"]
        _, m, clf_omni = train_classifier(
            grn,
            C=1,
            train_size=0.9,
            class_weight={1: 800, 0: 1},
            shuffle=True,
            doplot=False,
            return_full=False,
            use_col="gene_name",
        )
        coef = clf_omni.coef_[0] if clf_omni.coef_.shape[0] == 1 else clf_omni.coef_
        grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
        metrics["omni"] = BenGRN(grn, do_auc=True, doplot=False).compare_to(other=adata)
        metrics["omni_classifier"] = m
        grn.var.index = grn.var["symbol"]
        grn.varp["GRN"] = grn.varp["GRN"].T
        metrics["omni_base"] = BenGRN(
            grn, do_auc=True, doplot=False
        ).scprint_benchmark()
        grn.varp["GRN"] = np.transpose(grn.varp["all"], (1, 0, 2))
        grn.var.index = grn.var["ensembl_id"]
        _, m, clf_self = train_classifier(
            grn,
            other=adata,
            C=1,
            train_size=0.5,
            class_weight={1: 40, 0: 1},
            doplot=False,
            shuffle=False,
            return_full=False,
            use_col="ensembl_id",
        )
        coef = clf_self.coef_[0] if clf_self.coef_.shape[0] == 1 else clf_self.coef_
        grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1).T
        metrics["self"] = BenGRN(grn, do_auc=True, doplot=False).compare_to(other=adata)
        metrics["self_classifier"] = m
        grn.var.index = grn.var["symbol"]
        grn.varp["GRN"] = grn.varp["GRN"].T
        metrics["self_base"] = BenGRN(
            grn, do_auc=True, doplot=False
        ).scprint_benchmark()
    else:
        # max_genes=4000
        adata = sc.read_h5ad(default_dataset)
        adata.var["isTF"] = False
        adata.var.loc[adata.var.symbol.isin(grnutils.TF), "isTF"] = True
        for celltype in cell_types:
            # print(celltype)
            # grn_inferer = GNInfer(
            #    layer=layers,
            #    how="random expr",
            #    preprocess="softmax",
            #    head_agg="max",
            #    filtration="none",
            #    forward_mode="none",
            #    num_workers=8,
            #    num_genes=2200,
            #    max_cells=maxcells,
            #    doplot=False,
            #    batch_size=batch_size,
            # )
            #
            # grn = grn_inferer(model, adata[adata.X.sum(1) > 500], cell_type=celltype)
            # grn.var.index = make_index_unique(grn.var["symbol"].astype(str))
            # metrics[celltype + "_scprint"] = BenGRN(
            #    grn, doplot=False
            # ).scprint_benchmark()
            # del grn
            # gc.collect()
            grn_inferer = GNInfer(
                layer=layers,
                how="most var across",
                preprocess="softmax",
                head_agg="none",
                filtration="none",
                forward_mode="none",
                num_workers=8,
                num_genes=maxgenes,
                max_cells=maxcells,
                doplot=False,
                batch_size=batch_size,
            )
            grn = grn_inferer(model, adata[adata.X.sum(1) > 500], cell_type=celltype)
            grn.var.index = make_index_unique(grn.var["symbol"].astype(str))
            grn.varp["all"] = grn.varp["GRN"]
            grn.varp["GRN"] = grn.varp["GRN"].mean(-1)
            metrics[celltype + "_scprint_mean"] = BenGRN(
                grn, doplot=False
            ).scprint_benchmark()
            if clf_omni is None:
                grn.varp["GRN"] = grn.varp["all"]
                _, m, clf_omni = train_classifier(
                    grn,
                    C=1,
                    train_size=0.6,
                    max_iter=300,
                    class_weight={1: 800, 0: 1},
                    return_full=False,
                    shuffle=True,
                    doplot=False,
                )
                joblib.dump(clf_omni, "clf_omni.pkl")
                metrics["classifier"] = m
            coef = clf_omni.coef_[0] if clf_omni.coef_.shape[0] == 1 else clf_omni.coef_
            grn.varp["GRN"] = grn.varp["all"][:, :, coef > 0].mean(-1)
            metrics[celltype + "_scprint_class"] = BenGRN(
                grn, doplot=False
            ).scprint_benchmark()
            del grn
            gc.collect()
    return metrics
