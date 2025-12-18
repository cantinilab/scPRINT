import gc
import os.path
from typing import Any, List, Optional

import joblib
import networkx as nx
import numpy as np
import scanpy as sc
import scipy.sparse
import seaborn as sns
import torch
import umap
from anndata import AnnData
from anndata.utils import make_index_unique
from bengrn import BenGRN, get_perturb_gt, get_sroy_gt
from bengrn.base import train_classifier

# from bengrn.GeneRNIB_reg2 import run_gene_rnib, NORMAN, OP, ADAMSON
from grnndata import GRNAnnData, from_anndata
from grnndata import utils as grnutils
from matplotlib import pyplot as plt
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from scdataloader.utils import load_genes
from simpler_flash import FlashTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint2.utils.sinkhorn import SinkhornDistance

from .tmfg import tmfg

FILEDIR = os.path.dirname(os.path.realpath(__file__))


class GNInfer:
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        drop_unexpressed: bool = True,
        num_genes: int = 3000,
        max_cells: int = 0,
        cell_type_col: str = "cell_type",
        how: str = "random expr",  # random expr, most var within, most var across, some
        genelist: Optional[List[str]] = None,
        layer: Optional[List[int]] = None,
        preprocess: str = "softmax",  # sinkhorn, softmax, none
        head_agg: str = "mean",  # mean, sum, none, mean_full
        filtration: str = "thresh",  # thresh, top-k, mst, known, none
        k: int = 10,
        known_grn: Optional[Any] = None,
        precomp_attn: bool = False,
        symmetrize: bool = False,
        loc: str = "./",
        use_knn: bool = True,
    ):
        """
        GNInfer a class to infer gene regulatory networks from a dataset using a scPRINT model.

        Args:
            batch_size (int, optional): Batch size for processing. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 8.
            drop_unexpressed (bool, optional): Whether to drop unexpressed genes. Defaults to True.
                In this context, genes that have no expression in the dataset are dropped.
            num_genes (int, optional): Number of genes to consider. Defaults to 3000.
            max_cells (int, optional): Maximum number of cells to consider. Defaults to 0.
                if less than total number of cells, only the top `max_cells` cells with the most counts will be considered.
            cell_type_col (str, optional): Column name for cell type information. Defaults to "cell_type".
            how (str, optional): Method to select genes. Options are "most var", "random expr", "some". Defaults to "most var".
                - "most var across": select the most variable genes across all cell types
                - "most var within": select the most variable genes within a cell type
                - "random expr": select random expressed genes
                - "some": select a subset of genes defined in genelist
                - "most expr": select the most expressed genes in the cell type
            genelist (list, optional): List of genes to consider. Defaults to an empty list.
            layer (Optional[List[int]], optional): List of layers to use for the inference. Defaults to None.
            preprocess (str, optional): Preprocessing method. Options are "softmax", "sinkhorn", "none". Defaults to "softmax".
            head_agg (str, optional): Aggregation method for heads. Options are "mean_full", "mean", "sum", "none". Defaults to "mean".
            filtration (str, optional): Filtration method for the adjacency matrix. Options are "thresh", "top-k", "mst", "known", "none". Defaults to "thresh".
            k (int, optional): Number of top connections to keep if filtration is "top-k". Defaults to 10.
            known_grn (optional): Known gene regulatory network to use as a reference. Defaults to None.
                - We will only keep the genes that are present in the known GRN.
            precomp_attn (bool, optional): Whether to let the model precompute attn or do it at the end.
                This takes more memory but the model can compute mean over the attention matrices instead
                of over the qs and ks then taking the product.
                It is required for mean_full head_agg. Defaults to False.
            symmetrize (bool, optional): Whether to GRN. Defaults to False.
            loc (str, optional): Location to save results. Defaults to "./".
            use_knn (bool, optional): Whether to use k-nearest neighbors information. Defaults to True.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.layer = layer
        self.loc = loc
        self.how = how
        assert self.how in [
            "most var within",
            "most var across",
            "random expr",
            "some",
            "most expr",
        ], "how must be one of 'most var within', 'most var across', 'random expr', 'some', 'most expr'"
        self.num_genes = num_genes if self.how != "some" else len(self.genelist)
        self.preprocess = preprocess
        self.cell_type_col = cell_type_col
        self.filtration = filtration
        self.genelist = genelist if genelist is not None else []
        self.k = k
        self.symmetrize = symmetrize
        self.known_grn = known_grn
        self.head_agg = head_agg
        self.max_cells = max_cells
        self.curr_genes = None
        self.drop_unexpressed = drop_unexpressed
        self.use_knn = use_knn
        if self.filtration != "none" and self.head_agg == "none":
            raise ValueError("filtration must be 'none' when head_agg is 'none'")

    def __call__(self, model: torch.nn.Module, adata: AnnData, cell_type=None) -> tuple[AnnData, np.ndarray]:
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
        adjacencies = self.aggregate(model)
        model.attn.data = None
        if self.head_agg == "none":
            return self.save(
                adjacencies[self.n_cell_embs :, self.n_cell_embs :, :],
                subadata,
            )
        else:
            return self.save(
                self.filter(adjacencies)[self.n_cell_embs :, self.n_cell_embs :],
                subadata,
                loc=self.loc,
            )

    def predict(self, model, adata, layer, cell_type=None):
        """
        part to predict the qks or attns matrices from the adata with the model
        """
        self.curr_genes = None
        model.pred_log_adata = False
        if cell_type is not None:
            subadata = adata[adata.obs[self.cell_type_col] == cell_type].copy()
        else:
            subadata = adata.copy()
        if self.how == "most var within":
            try:
                sc.pp.highly_variable_genes(
                    subadata, flavor="seurat_v3", n_top_genes=self.num_genes
                )
            except ValueError:
                sc.pp.highly_variable_genes(
                    subadata,
                    flavor="seurat_v3",
                    n_top_genes=self.num_genes,
                    span=0.6,
                )
            self.curr_genes = (
                subadata.var.index[subadata.var.highly_variable].tolist()
                + self.genelist
            )
            print(
                "number of expressed genes in this cell type: "
                + str((subadata.X.sum(0) > 1).sum())
            )
        elif self.how == "most var across" and cell_type is not None:
            adata.raw = adata
            sc.tl.rank_genes_groups(
                adata,
                mask_var=adata.var.index.isin(model.genes),
                groupby=self.cell_type_col,
                groups=[cell_type],
            )
            diff_expr_genes = adata.uns["rank_genes_groups"]["names"][cell_type]
            diff_expr_genes = [gene for gene in diff_expr_genes if gene in model.genes]
            self.curr_genes = diff_expr_genes[: self.num_genes] + self.genelist
            self.curr_genes.sort()
        elif self.how == "random expr":
            self.curr_genes = model.genes
            # raise ValueError("cannot do it yet")
            pass
        elif self.how == "some" and len(self.genelist) > 0:
            self.curr_genes = self.genelist
        elif self.how == "most expr":
            self.curr_genes = adata.var.index[
                adata.X.sum(0).A1.argsort()[::-1]
            ].tolist()[: self.num_genes]
        else:
            raise ValueError("something wrong with your inputs")
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
            subadata,
            obs_to_output=["organism_ontology_term_id"],
            get_knn_cells=model.expr_emb_style == "metacell" and self.use_knn,
        )
        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            max_len=self.num_genes if self.how == "random expr" else 0,
            how="some" if self.how != "random expr" else "random expr",
            genelist=self.curr_genes if self.how != "random expr" else [],
            n_bins=model.n_input_bins if model.expr_emb_style == "binned" else 0,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        model.attn.precomp_attn = self.head_agg == "mean_full"
        if self.num_genes > 10_000 and model.attn.precomp_attn:
            raise ValueError("need less genes for a non-shared-qk version")
        prevplot = model.doplot

        model.doplot = False
        model.on_predict_epoch_start()
        model.eval()
        model.attn.data = None
        # reparametrize the attn process

        if model.transformer.attn_type == "hyper":
            self.curr_genes = [i for i in model.genes if i in self.curr_genes]
            num = (1 if model.use_metacell_token else 0) + (
                (len(model.classes) + 1) if not model.cell_transformer else 0
            )
            if (len(self.curr_genes) + num) % 128 != 0:
                self.curr_genes = self.curr_genes[
                    : (len(self.curr_genes) // 128 * 128) - num
                ]
        if self.how != "random expr":
            if model.attn.precomp_attn:
                model.attn.gene_dim = len(set(self.curr_genes) & set(model.genes))
                model.attn.apply_softmax = self.preprocess == "softmax"
            else:
                if subadata.obs["organism_ontology_term_id"].unique().shape[0] > 1:
                    raise ValueError(
                        "only one organism at a time is supported for precomp_attn"
                    )
                n = False
                for i, k in col.start_idx.items():
                    if n:
                        model.attn.gene_dim = k - model.attn.speciesloc
                        break
                    if i == subadata.obs["organism_ontology_term_id"].unique()[0]:
                        model.attn.speciesloc = k
                        n = True
        elif not model.attn.precomp_attn:
            raise ValueError(
                "full attention (i.e. precomp_attn=True) is not supported for random expr"
            )
        device = model.device.type
        # this is a debugger line
        dtype = (
            torch.float16
            if isinstance(model.transformer, FlashTransformer)
            else model.dtype
        )
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
                    knn_cells_info=(
                        batch["knn_cells_info"].to(device)
                        if model.expr_emb_style == "metacell" and self.use_knn
                        else None
                    ),
                    keep_output=False,
                    get_attention_layer=layer if type(layer) is list else [layer],
                )
                torch.cuda.empty_cache()
        model.doplot = prevplot
        return subadata

    def aggregate(self, model):
        """
        part to aggregate the qks and compute the attns
        or to aggregate the attns
        or do nothing if already done
        """
        attn, genes = model.attn.get(), model.genes
        if model.attn.precomp_attn:
            self.curr_genes = [i for i in genes if i in self.curr_genes]
            return attn.detach().cpu().numpy()
        if self.how == "random expr" and self.drop_unexpressed:
            keep = np.array(
                [1] * self.n_cell_embs + [i in self.curr_genes for i in genes],
                dtype=bool,
            )
            attn = attn[:, keep, :, :, :]
        badloc = torch.isnan(attn.sum((0, 2, 3, 4)))
        attn = attn[:, ~badloc, :, :, :]
        badloc = badloc.detach().cpu().numpy()
        self.curr_genes = (
            np.array(self.curr_genes)[~badloc[self.n_cell_embs :]]
            if self.how == "random expr"
            else [i for i in genes if i in self.curr_genes]
        )
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

            if self.preprocess == "sinkhorn":
                scale = Qs.shape[-1] ** -0.5
                attn = attn * scale
                if attn.numel() > 100_000_000:
                    raise ValueError("you can't sinkhorn such a large matrix")
                sink = SinkhornDistance(0.1, max_iter=200)
                attn = sink(attn)[0]
                attn = attn * Qs.shape[-1]
            elif self.preprocess == "softmax":
                scale = Qs.shape[-1] ** -0.5
                attn = attn * scale
                attn = torch.nn.functional.softmax(attn, dim=-1)
            elif self.preprocess == "softpick":
                attn = softpick(attn)
            elif self.preprocess == "none":
                pass
            else:
                raise ValueError(
                    "preprocess must be one of 'sinkhorn', 'softmax', 'none'"
                )
            if self.symmetrize:
                attn = (attn + attn.T) / 2
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
                raise ValueError(
                    "head_agg must be one of 'mean', 'mean_full', 'max' or 'none'"
                )
        if self.head_agg == "mean":
            attns = attns / Qs.shape[0]
        return (
            attns.detach().cpu().numpy() if self.head_agg != "none" else attns.numpy()
        )

    def filter(self, adj, gt=None):
        """
        part to filter the attn matrix given user inputs
        """
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
) -> dict:
    """
    default_benchmark function to run the default scPRINT GRN benchmark

    Args:
        model (Any): The scPRINT model to be used for the benchmark.
        default_dataset (str, optional): The default dataset to use for benchmarking. Defaults to "sroy".
        cell_types (List[str], optional): List of cell types to include in the benchmark. Defaults to [].
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
            do_postp=model.expr_emb_style == "metacell",
            min_valid_genes_id=5000,
            min_dataset_size=64,
            keepdata=True,
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
            if model.expr_emb_style == "metacell":
                sc.pp.neighbors(adata, use_rep="X_pca")
            grn_inferer = GNInfer(
                layer=layers,
                how="most var within",
                preprocess=(
                    "softpick"
                    if model.attention in ["softpick", "softpick-flash"]
                    else "softmax"
                ),
                head_agg="none",
                filtration="none",
                num_genes=maxgenes,
                num_workers=8,
                max_cells=maxcells,
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
        adata = get_perturb_gt()
        preprocessor = Preprocessor(
            force_preprocess=True,
            keepdata=True,
            skip_validate=True,
            do_postp=model.expr_emb_style == "metacell",
            min_valid_genes_id=maxgenes,
            min_dataset_size=64,
        )
        nadata = preprocessor(adata.copy())
        if model.expr_emb_style == "metacell":
            sc.pp.neighbors(nadata, use_rep="X_pca")
        nadata.var["isTF"] = False
        nadata.var.loc[nadata.var.gene_name.isin(grnutils.TF), "isTF"] = True
        nadata.var["isTF"].sum()
        grn_inferer = GNInfer(
            layer=layers,
            how="most var within",
            preprocess=(
                "softpick"
                if model.attention in ["softpick", "softpick-flash"]
                else "softmax"
            ),
            head_agg="none",
            filtration="none",
            num_genes=maxgenes,
            max_cells=maxcells,
            num_workers=8,
            batch_size=batch_size,
        )
        grn = grn_inferer(model, nadata)
        del nadata
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
    elif default_dataset == "genernib":
        raise ValueError("Not implemented")
        # for adata in [NORMAN, OP, ADAMSON]:
        #   adata = sc.read_h5ad(adata)
        #   adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
        #   preprocessor = Preprocessor(
        #       force_preprocess=False,
        #       skip_validate=True,
        #       drop_non_primary=False,
        #       do_postp=False,
        #       min_valid_genes_id=1000,
        #       min_dataset_size=64,
        #       keepdata=True,
        #       is_symbol=True,
        #       use_raw=False,
        #   )
        #   adata = preprocessor(adata.copy())
        #   run_gene_rnib(
        #      adata=adata,
        #      model=model,
        #      layer=layers,
        #      how="most var within",
        #      preprocess="softmax",
        #   )
        #   grn_inferer = GNInfer(
        #      how="most var across",
        #      preprocess="softmax",
        #      head_agg="mean",
        #      filtration="none",
        #      forward_mode="none",
        #      num_genes=3_000,
        #      max_cells=3000,
        #      batch_size=10,
        #      cell_type_col="perturbation",
        #      layer=list(range(model.nlayers))[:],
        #   )
        # grn = grn_inferer(model, adata, cell_type="ctrl")
        # grn.var.index = make_index_unique(grn.var["symbol"].astype(str))

    else:
        # max_genes=4000
        if default_dataset.startswith("https://"):
            adata = sc.read(
                FILEDIR + "/../../data/" + default_dataset.split("/")[-1],
                backup_url=default_dataset,
            )
        else:
            adata = sc.read_h5ad(default_dataset)
        if default_dataset.split("/")[-1] in ["yBCKp6HmXuHa0cZptMo7.h5ad"]:
            use_layer = "counts"
            is_symbol = True
        else:
            use_layer = None
            is_symbol = False

        preprocessor = Preprocessor(
            use_layer=use_layer,
            is_symbol=is_symbol,
            force_preprocess=True,
            skip_validate=True,
            do_postp=model.expr_emb_style == "metacell",
            drop_non_primary=False,
        )
        adata = preprocessor(adata.copy())

        adata.var["isTF"] = False
        adata.var.loc[adata.var.symbol.isin(grnutils.TF), "isTF"] = True
        if model.expr_emb_style == "metacell":
            if "X_pca" not in adata.obsm:
                sc.pp.pca(adata, n_comps=50)
            sc.pp.neighbors(adata, use_rep="X_pca")
        for celltype in list(adata.obs["cell_type"].unique())[:14]:
            # print(celltype)
            # grn_inferer = GNInfer(
            #    layer=layers,
            #    how="random expr",
            #    preprocess="softmax",
            #    head_agg="max",
            #    filtration="none",
            #    num_workers=8,
            #    num_genes=2200,
            #    max_cells=maxcells,
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
                preprocess=(
                    "softpick"
                    if model.attention in ["softpick", "softpick-flash"]
                    else "softmax"
                ),
                head_agg="none",
                filtration="none",
                num_workers=8,
                num_genes=maxgenes,
                max_cells=maxcells,
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


def softpick(x, dim=-1, eps=1e-5):
    # softpick function: relu(exp(x)-1) / sum(abs(exp(x)-1))
    # numerically stable version
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = torch.nn.functional.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (
        torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps
    )  # epsilon is only useful if all inputs are EXACTLY 0. we might not even need it
