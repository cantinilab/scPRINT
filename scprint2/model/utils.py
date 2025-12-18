import gc
import math
import os.path
from collections import Counter
from typing import Dict, List, Optional, Union

import bionty as bt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from matplotlib import pyplot as plt
from scdataloader.utils import translate
from scipy.sparse import csr_matrix
from torch import Tensor
from torch.distributions import Gamma, Poisson

from .. import utils
from ..datasets import DENOISE_DATASETS, EMBEDDING_DATASETS
from ..tasks import cell_emb as embbed_task
from ..tasks import denoise as denoise_task
from ..tasks import grn as grn_task

FILEDIR = os.path.dirname(os.path.realpath(__file__))


def make_adata(
    genes: List[str],
    embs: Union[Tensor, Dict[str, Tensor]],
    pos: Tensor = None,
    expr_pred: List[Tensor] = None,
    classes: List[str] = None,
    pred: Tensor = None,
    label_decoders: Optional[Dict] = None,
    labels_hierarchy: Optional[Dict] = None,
    gtclass: Optional[Tensor] = None,
    doplot: bool = True,
) -> AnnData:
    """
    This function creates an AnnData object from the given input parameters.

    Args:
        genes (list): List of genes that will be used as variable names.
        embs (torch.Tensor|Dict): Embeddings of the cells. The shape of the tensor is (n_cells, n_features).
            if multiple, it is a dict of name -> tensor
        pos (torch.Tensor): Positions of the cells. The shape of the tensor is (n_cells,).
        expr_pred (List[torch.Tensor]): Predicted expression. The shape of the tensors are (n_cells, n_genes).
            the first is mu, the second theta, the third pi if present
        classes (list): List of classes, the order should be the same as in the pred and gtclass tensors.
        pred (torch.Tensor, optional): Predicted labels. The shape of the tensor is (n_cells, n_classes). Default is None.
        label_decoders (dict, optional): Dictionary to map class codes to class names. Default is None.
        labels_hierarchy (dict, optional): Dictionary representing the hierarchy of labels. Default is {}. see the model for defintion.
        gtclass (torch.Tensor, optional): Ground truth class values. Default is None.
        doplot (bool, optional): Whether to generate plots. Default is True.

    Returns:
        anndata.AnnData: The created AnnData object.
    """
    print("logging the anndata")
    colname = ["pred_" + i for i in classes]
    if pred is not None:
        obs = np.array(pred.to(device="cpu", dtype=torch.int32))
        # label decoders is not cls_decoders. one is a dict to map class codes (ints)
        # to class names the other is the module the predict the class
        if label_decoders is not None:
            obs = np.array(
                [
                    [label_decoders[classes[i]][n] for n in name]
                    for i, name in enumerate(obs.T)
                ]
            ).T
        if gtclass is not None:
            colname += classes
            nobs = np.array(gtclass.to(device="cpu", dtype=torch.int32))
            if label_decoders is not None:
                nobs = np.array(
                    [
                        [label_decoders[classes[i]][n] for n in name]
                        for i, name in enumerate(nobs.T)
                    ]
                ).T
            obs = np.hstack([obs, nobs])

    n_cells = embs[list(embs.keys())[0]].shape[0]
    layers = None
    size = len(genes)
    if pos is not None:
        minval = pos.min()
        maxval = pos.max()
        genes = genes[minval : maxval + 1]
        size = len(genes)
        pos = pos - minval
        mu_array = np.zeros((n_cells, size), dtype=np.float32)
        pos = pos.cpu().numpy()
        # Create empty array with same shape as expr_pred[0]
        # Fill array with values from expr_pred[0]
        for idx in range(n_cells):
            mu_array[idx, pos[idx]] = expr_pred[0][idx].cpu().numpy() + 1
        exist = mu_array.sum(0) != 0
        mu_array = mu_array[:, exist]
        mu_array[mu_array == 1] = 0
        layers = {
            "scprint_mu": mu_array,
            #  "used_scprint": csr_matrix(pos),
        }
        if len(expr_pred) > 1:
            theta_array = np.zeros((n_cells, size), dtype=np.float32)
            # Fill array with values from expr_pred[0]
            for idx in range(n_cells):
                theta_array[idx, pos[idx]] = expr_pred[1][idx].cpu().numpy()
            layers["scprint_theta"] = theta_array[:, exist]

            pi_array = np.zeros((n_cells, size), dtype=np.float32)
            # Fill array with values from expr_pred[0]
            for idx in range(n_cells):
                pi_array[idx, pos[idx]] = expr_pred[2][idx].cpu().numpy()
            layers["scprint_pi"] = pi_array[:, exist]
        genes = [n for i, n in enumerate(genes) if exist[i] > 0]
    else:
        genes = []
    adata = AnnData(
        X=csr_matrix((n_cells, len(genes))),
        layers=layers,
        obs=(
            pd.DataFrame(
                obs,
                columns=colname,
            )
            if pred is not None
            else None
        ),
        var=pd.DataFrame(index=genes),
    )

    for k, v in embs.items():
        adata.obsm["scprint_emb_" + k] = v.cpu().numpy()
        rep = "scprint_emb_" + k
    del embs
    accuracy = {}
    if labels_hierarchy is None:
        labels_hierarchy = {}
    if pred is not None:
        for clss in classes:
            if gtclass is not None:
                tr = translate(set(adata.obs[clss]), clss)
                if tr is not None:
                    adata.obs["conv_" + clss] = adata.obs[clss].replace(tr)
            tr = translate(set(adata.obs["pred_" + clss]), clss)
            if tr is not None:
                adata.obs["conv_pred_" + clss] = adata.obs["pred_" + clss].replace(tr)
            res = []
            if label_decoders is not None and gtclass is not None:
                class_topred = label_decoders[clss].values()
                if clss in labels_hierarchy:
                    cur_labels_hierarchy = {
                        label_decoders[clss][k]: [label_decoders[clss][i] for i in v]
                        for k, v in labels_hierarchy[clss].items()
                    }
                else:
                    cur_labels_hierarchy = {}
                for pred, true in adata.obs[["pred_" + clss, clss]].values:
                    if pred == true:
                        res.append(True)
                        continue
                    if len(labels_hierarchy) > 0:
                        if true in cur_labels_hierarchy:
                            res.append(pred in cur_labels_hierarchy[true])
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
                    else:
                        pass
                accuracy["pred_" + clss] = sum(res) / len(res) if len(res) > 0 else 0
        adata.obs = adata.obs.astype("category")
    print(adata)
    if doplot and adata.shape[0] > 100:
        sc.pp.neighbors(adata, use_rep=rep)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added="sprint_leiden")
        if gtclass is not None:
            color = [
                i
                for pair in zip(
                    [
                        "conv_" + i if "conv_" + i in adata.obs.columns else i
                        for i in classes
                    ],
                    [
                        (
                            "conv_pred_" + i
                            if "conv_pred_" + i in adata.obs.columns
                            else "pred_" + i
                        )
                        for i in classes
                    ],
                )
                for i in pair
            ]
            fig, axs = plt.subplots(
                int(len(color) / 2), 2, figsize=(24, len(color) * 4)
            )
            plt.subplots_adjust(wspace=1)
            if len(color) > 2:
                for i, col in enumerate(color):
                    sc.pl.umap(
                        adata,
                        color=col,
                        ax=axs[i // 2, i % 2],
                        show=False,
                    )
                    acc = ""
                    if "pred_" in col and col.split("conv_")[-1] in accuracy:
                        acc = " (accuracy: {:.2f})".format(
                            accuracy[col.split("conv_")[-1]]
                        )
                    axs[i // 2, i % 2].set_title(col + " UMAP" + acc)
                    if "cell_type" in col:
                        axs[i // 2, i % 2].legend(fontsize="x-small")
                    axs[i // 2, i % 2].set_xlabel("UMAP1")
                    axs[i // 2, i % 2].set_ylabel("UMAP2")
            else:
                for i, col in enumerate(color):
                    sc.pl.umap(
                        adata,
                        color=col,
                        ax=axs[i % 2],
                        show=False,
                    )
                    acc = ""
                    if "pred_" in col and col.split("conv_")[-1] in accuracy:
                        acc = " (accuracy: {:.2f})".format(
                            accuracy[col.split("conv_")[-1]]
                        )
                    axs[i % 2].set_title(col + " UMAP" + acc)
                    if "cell_type" in col:
                        axs[i % 2].legend(fontsize="x-small")
                    axs[i % 2].set_xlabel("UMAP1")
                    axs[i % 2].set_ylabel("UMAP2")
        else:
            color = [
                (
                    "conv_pred_" + i
                    if "conv_pred_" + i in adata.obs.columns
                    else "pred_" + i
                )
                for i in classes
            ]
            if len(color) > 1:
                fig, axs = plt.subplots(len(color), 1, figsize=(16, len(color) * 8))
                for i, col in enumerate(color):
                    sc.pl.umap(
                        adata,
                        color=col,
                        ax=axs[i],
                        show=False,
                    )
                    acc = ""
                    if "pred_" in col and col.split("conv_")[-1] in accuracy:
                        acc = " (accuracy: {:.2f})".format(
                            accuracy[col.split("conv_")[-1]]
                        )
                    axs[i].set_title(col + " UMAP of " + rep + " embedding " + acc)
                    axs[i].set_xlabel("UMAP1")
                    axs[i].set_ylabel("UMAP2")
            else:
                fig = sc.pl.umap(
                    adata,
                    color=color,
                    show=False,
                    return_fig=True,
                )
        plt.show()
    else:
        fig = None
    return adata, fig


def _init_weights(
    module: torch.nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    mup_width_scale: float = 1.0,
    rescale_prenorm_residual: bool = True,
):
    """
    This function initializes the weights of the given module. The initialization is done based on the type of the module.

    If the module is a Linear layer, the weights are initialized with a normal distribution with a standard deviation
    that is a product of the initializer range and the mup_init_scale. The learning rate multiplier is also set for the
    weights of the Linear layer. If the module has a bias, it is initialized with zeros.
    If the module is an Embedding layer, no initialization is performed.
    If the rescale_prenorm_residual flag is set to True, the weights of the residual layers are reinitialized according
    to the scheme proposed in the OpenAI GPT-2 Paper. The weights are scaled by a factor of 1/sqrt(N), where N is the
    number of residual layers.

    Args:
        module (torch.nn.Module): The module whose weights are to be initialized.
        n_layer (int): The number of layers in the module.
        initializer_range (float, optional): The range of the initializer. Defaults to 0.02.
        mup_width_scale (float, optional): The scale for the mup initialization. Defaults to 1.0.
        rescale_prenorm_residual (bool, optional): Flag to indicate whether to rescale the prenorm residual. Defaults to True.
    """
    mup_init_scale = math.sqrt(mup_width_scale)
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, std=initializer_range * mup_init_scale)
        optim_cfg = getattr(module.weight, "_optim", {})
        optim_cfg.update({"lr_multiplier": mup_width_scale})
        setattr(module.weight, "_optim", optim_cfg)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        pass

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=initializer_range * mup_init_scale / math.sqrt(2 * n_layer),
                )


def downsample_profile(mat: Tensor, dropout: float, method="new", randsamp=False) -> Tensor:
    """
    This function downsamples the expression profile of a given single cell RNA matrix.

    The noise is applied based on the renoise parameter,
    the total counts of the matrix, and the number of genes. The function first calculates the noise
    threshold (scaler) based on the renoise parameter. It then generates an initial matrix count by
    applying a Poisson distribution to a random tensor scaled by the total counts and the number of genes.
    The function then models the sampling zeros by applying a Poisson distribution to a random tensor
    scaled by the noise threshold, the total counts, and the number of genes. The function also models
    the technical zeros by generating a random tensor and comparing it to the noise threshold. The final
    matrix count is calculated by subtracting the sampling zeros from the initial matrix count and
    multiplying by the technical zeros. The function ensures that the final matrix count is not less
    than zero by taking the maximum of the final matrix count and a tensor of zeros. The function
    returns the final matrix count.

    Args:
        mat (torch.Tensor): The input matrix.
        dropout (float): The renoise parameter.

    Returns:
        torch.Tensor: The matrix count after applying noise.
    """
    # Randomly drop on average N counts to each element of expression using a heavy tail Gaussian distribution
    # here we try to get the scale of the distribution so as to remove the right number of counts from each gene
    # https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02601-5#:~:text=Zero%20measurements%20in%20scRNA%2Dseq,generation%20of%20scRNA%2Dseq%20data.
    if randsamp:
        dropout = torch.rand(mat.shape[0], device=mat.device) * dropout
        dropout = (
            dropout.unsqueeze(1)
            if len(mat.shape) == 2
            else dropout.unsqueeze(1).unsqueeze(1)
        )
    if method == "old":
        totcounts = mat.sum(-1)
        ngenes = mat.shape[-1]
        tnoise = 1 - (1 - dropout) ** (1 / 2)
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson(
            torch.rand(mat.shape, device=mat.device)
            * ((tnoise * totcounts.unsqueeze(-1)) / (0.5 * ngenes))
        ).int()
        # we model the technical zeros (dropping 50% of the genes)
        drop = (torch.rand(mat.shape, device=mat.device) > tnoise).int()

        mat = (mat - res) * drop
        return torch.maximum(
            mat,
            torch.zeros(
                (1, 1) if len(mat.shape) == 2 else (1, 1, 1),
                device=mat.device,
                dtype=torch.int,
            ),
        )
    elif method == "jules":
        scaler = (1 - dropout) ** (1 / 2)
        notdrop = (
            torch.rand(
                mat.shape,
                device=mat.device,
            )
            < scaler
        ).int()
        notdrop[mat == 0] = 0
        # apply the dropout after the poisson, right?
        return notdrop * torch.poisson(mat * scaler)
    elif method == "new":
        dropout = dropout * 1.1
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson((mat * (dropout / 2))).int()
        # we model the technical zeros (dropping 50% of the genes)
        notdrop = (torch.rand(mat.shape, device=mat.device) >= (dropout / 2)).int()
        mat = (mat - res) * notdrop
        return torch.maximum(
            mat,
            torch.zeros(
                (1, 1) if len(mat.shape) == 2 else (1, 1, 1),
                device=mat.device,
                dtype=torch.int,
            ),
        )
    else:
        raise ValueError(f"method {method} not recognized")


def simple_masker(
    shape: List[int],
    mask_ratio: float = 0.15,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        shape (List[int]): The shape of the data.
        mask_ratio (float): The ratio of genes to mask, default to 0.15.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    return torch.rand(shape) < mask_ratio


class WeightedMasker:
    def __init__(
        self,
        genes: List[str],
        TFs: List[str] = utils.fileToList(FILEDIR + "/../../data/main/TFs.txt"),
        tf_weight: float = 10,
    ):
        """
        Randomly mask a batch of data.

        Args:
            genes (List[str]): The list of genes the model might see.
            TFs (List[str]): The list of TFs the model can drop.
            tf_weight (float): How likely it is to drop a non TF compared to a TF.
        """
        TFs = set(TFs)
        self.weights = torch.tensor([tf_weight if gene in TFs else 1 for gene in genes])
        self.max_to_drop = (self.weights == tf_weight).sum()
        self.tf_weight = tf_weight

    def __call__(self, ids: torch.Tensor, mask_ratio: float = 1.0) -> torch.Tensor:
        if self.tf_weight == 0:
            if mask_ratio * ids.shape[1] > self.max_to_drop:
                raise ValueError("Cannot drop more than max_to_drop")
        # Create a tensor of probabilities for each position
        probs = self.weights.expand(ids.shape[0], -1).to(ids.device)
        ids = ids.to(torch.int64)
        probs = torch.gather(
            probs, 1, ids
        )  # Get probabilities only for the indices in ids
        probs = probs / probs.sum(1, keepdim=True)

        # Sample from multinomial for each item in batch
        num_samples = int(ids.shape[1] * mask_ratio)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        sampled = torch.multinomial(probs, num_samples, replacement=False)
        return mask.scatter_(1, sampled, True)


def zinb_sample(
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_probs: torch.Tensor,
    sample_shape: torch.Size = torch.Size([]),
) -> torch.Tensor:
    """
    zinb_sample This function generates a sample from a Zero-Inflated Negative Binomial (ZINB) distribution.

    Args:
        mu (torch.Tensor): The mean of the Negative Binomial (NB) distribution.
        theta (torch.Tensor): The dispersion parameter of the NB distribution.
        zi_probs (torch.Tensor): The zero-inflation probabilities.
        sample_shape (torch.Size, optional): The output shape. Defaults to torch.Size([]).

    Returns:
        torch.Tensor: A sample from the ZINB distribution.
    """
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    p_means = gamma_d.sample(sample_shape)

    # Clamping as distributions objects can have buggy behaviors when
    # their parameters are too high
    l_train = torch.clamp(p_means, max=1e8)
    samp = Poisson(l_train).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
    is_zero = torch.rand_like(samp) <= zi_probs
    samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
    return samp_


class Attention:
    def __init__(
        self,
        gene_dim: int,
        precomp_attn: bool = False,
        apply_softmax: bool = True,
        sum_heads: bool = True,
        additional_tokens: int = 0,
    ):
        """
        Initialize the Attention class.

        Args:
            gene_dim (int): The dimension of the gene.
            additional_tokens (int): The number of additional tokens to add.
            precomp_attn (bool): Whether to compute attention or it is precomputed
            apply_softmax (bool): Whether to apply softmax to the attention.
            sum_heads (bool): Whether to sum the heads.
        """
        self.data: Optional[Tensor] = None
        self.gene_dim: int = gene_dim
        self.additional_tokens: int = additional_tokens
        self.div: Optional[Tensor] = None
        self.apply_softmax: bool = apply_softmax
        self.sum_heads: bool = sum_heads
        self.precomp_attn: bool = precomp_attn
        self.speciesloc: int = 0

    def add(self, *args, **kwargs) -> None:
        if not self.precomp_attn:
            self.add_qk(*args, **kwargs)
        else:
            self.add_attn(*args, **kwargs)

    def add_attn(
        self, x: List[Tensor], pos: Tensor, expr: Optional[Tensor] = None
    ) -> None:
        """
        Aggregate the attention or data based on the precomp_attn flag.

        Args:
            x (List[Tensor]): List of tensors to aggregate. Tensor of size (batch, seq_len, 2, heads, emb)
            pos (Tensor): Position tensor.
        """
        if self.data is None:
            self.data = torch.zeros(
                [
                    self.gene_dim + self.additional_tokens,
                    self.gene_dim + self.additional_tokens,
                ],
                device=pos.device,
                dtype=torch.float32,
            )
            self.div = torch.zeros(1, device=pos.device, dtype=torch.float32)
        for i, elem in enumerate(x):
            if self.apply_softmax:
                attn = torch.nn.functional.softmax(
                    elem[:, :, 0, :, :].permute(0, 2, 1, 3)
                    @ elem[:, :, 1, :, :].permute(0, 2, 3, 1),
                    dim=-1,
                )
                if expr is not None:
                    attn[:, :, self.additional_tokens :, self.additional_tokens :] = (
                        attn[:, :, self.additional_tokens :, self.additional_tokens :]
                        * (expr > 0).float().unsqueeze(1).unsqueeze(-1)
                        * (expr > 0).float().unsqueeze(1).unsqueeze(2)
                    )
                self.data += attn.sum(0).mean(0)
            else:
                self.data[:, :] += (
                    (
                        elem[:, :, 0, :, :].permute(0, 2, 1, 3)
                        @ elem[:, :, 1, :, :].permute(0, 2, 3, 1)
                    )
                    .sum(0)
                    .mean(0)
                )
            self.div += 1

    def add_qk(
        self, x: List[Tensor], pos: Tensor, expr: Optional[Tensor] = None
    ) -> None:
        """
        Add data to the internal storage.

        Args:
            x (List[Tensor]): List of tensors to add.
            pos (Tensor): Position tensor.
        """
        # this is a debugger line
        if self.data is None:
            self.data = torch.zeros(
                [len(x), self.gene_dim + self.additional_tokens] + list(x[0].shape[2:]),
                device=pos.device,
            )
            self.div = torch.zeros(
                self.gene_dim + self.additional_tokens, device=pos.device
            )
        for i in range(x[0].shape[0]):  # batch size
            loc = torch.cat(
                [
                    torch.arange(self.additional_tokens, device=pos.device),
                    pos[i] + self.additional_tokens - self.speciesloc,
                ]
            ).int()
            for j in range(len(x)):  # number of layers * heads
                self.data[j, loc, :, :, :] += x[j][i]
            self.div[loc] += 1

    def get(self) -> Optional[np.ndarray]:
        """
        Get the aggregated attention or data.

        Returns:
            Optional[np.ndarray]: The aggregated attention or data.
        """
        if not self.precomp_attn:
            if self.data is None:
                return None
            # shape is (layers, genes, qkv, heads, emb)
            return self.data / self.div.view(1, self.div.shape[0], 1, 1, 1)
        else:
            if self.data is None:
                return None
            self.data.div_(self.div)
            return self.data


def test(
    model: torch.nn.Module,
    filedir: str,
    do_class: bool = True,
    maxcells_grn: int = 1024,
) -> None:
    """
    Test the given model on the full set of benchmarks and save the results to JSON files.

    Args:
        model (torch.nn.Module): The model to be tested.
        filedir (str): The directory where the data files are located.
        do_class (bool): Whether to perform classification. Defaults to True.
        maxcells_grn (int): Maximum cells for GRN analysis. Defaults to 1024.

    Returns:
        None
    """
    metrics = {}
    tot = {}
    for dataset, path in EMBEDDING_DATASETS.items():
        res = embbed_task.default_benchmark(
            model,
            dataset=path,
            do_class=do_class,
            coarse=False,
        )
        tot["embed_" + dataset] = res
        metrics.update(
            {
                "emb_" + dataset + "/scib": float(res["scib"]["Total"]),
                "emb_" + dataset + "/scib_bio": float(res["scib"]["Bio conservation"]),
                "emb_"
                + dataset
                + "/scib_batch": float(res["scib"]["Batch correction"]),
                "emb_"
                + dataset
                + "/ct_class": float(
                    res["classif"].get("cell_type_ontology_term_id", {}).get("macro", 0)
                    if do_class
                    else 0
                ),
                "emb_"
                + dataset
                + "/ct_class_macro": float(
                    res["classif"].get("cell_type_ontology_term_id", {}).get("macro", 0)
                    if do_class
                    else 0
                ),
            }
        )
        print(metrics)
        gc.collect()
    for dataset, filepath in DENOISE_DATASETS.items():
        res = denoise_task.default_benchmark(model, dataset=filepath)
        tot["denoise_" + dataset] = res
        metrics.update(
            {
                "denoise_"
                + dataset
                + "/reco2full_vs_noisy2full": float(
                    res["reco2full"] - res["noisy2full"]
                ),
            }
        )
        print(metrics)
        gc.collect()
    res = grn_task.default_benchmark(
        model,
        "gwps",
        batch_size=32 if model.d_model <= 512 else 8,
        maxcells=maxcells_grn,
    )
    tot["grn_gwps"] = res
    metrics.update(
        {
            "grn_gwps/auprc_self": float(res["self"]["auprc"]),
            "grn_gwps/epr_self": float(res["self"]["epr"]),
            "grn_gwps/auprc_omni": float(res["omni"]["auprc"]),
            "grn_gwps/epr_omni": float(res["omni"]["epr"]),
            "grn_gwps/auprc": float(res["mean"]["auprc"]),
            "grn_gwps/epr": float(res["mean"]["epr"]),
        }
    )
    print(metrics)
    gc.collect()
    for dataset, filepath in {
        "old_kidney": "https://datasets.cellxgene.cziscience.com/ede85b09-454b-4374-bf60-5f675e989b64.h5ad",
        # "kidney": "https://datasets.cellxgene.cziscience.com/01bc7039-961f-4c24-b407-d535a2a7ba2c.h5ad",
        "lung_smart": "https://datasets.cellxgene.cziscience.com/6ebba0e0-a159-406f-8095-451115673a2c.h5ad",
        # filedir + "/../../data/yBCKp6HmXuHa0cZptMo7.h5ad",
    }.items():
        res = grn_task.default_benchmark(
            model,
            filepath,
            # kidney dataset (2.87, 1.27) (0.00147, 0.00133)
            batch_size=32 if model.d_model <= 512 else 8,
            maxcells=maxcells_grn,
            maxgenes=4000,
        )
        tot["grn_omni_" + dataset] = res
        metrics.update(
            {
                "grn_omni_"
                + dataset
                + "/auprc_class": float(
                    np.mean([i["auprc"] for k, i in res.items() if "_class" in k])
                ),
                "grn_omni_"
                + dataset
                + "/or_class": float(
                    np.mean([i["odd_ratio"] for k, i in res.items() if "_class" in k])
                ),
                "grn_omni_"
                + dataset
                + "/tf_enr_class": float(
                    np.sum(
                        [
                            i.get("TF_enr", False)
                            for k, i in res.items()
                            if "_class" in k
                        ]
                    )
                ),
                "grn_omni_"
                + dataset
                + "/tf_targ_enr_class": float(
                    np.mean(
                        [
                            i["significant_enriched_TFtargets"]
                            for k, i in res.items()
                            if "_class" in k
                        ]
                    )
                ),
                "grn_omni_"
                + dataset
                + "/auprc": float(
                    np.mean([i["auprc"] for k, i in res.items() if "_mean" in k])
                ),
                "grn_omni_"
                + dataset
                + "/epr": float(
                    np.mean([i["epr"] for k, i in res.items() if "_mean" in k])
                ),
                "grn_omni_"
                + dataset
                + "/or": float(
                    np.mean([i["odd_ratio"] for k, i in res.items() if "_mean" in k])
                ),
                "grn_omni_"
                + dataset
                + "/tf_enr": float(
                    np.sum(
                        [i.get("TF_enr", False) for k, i in res.items() if "_mean" in k]
                    )
                ),
                "grn_omni_"
                + dataset
                + "/tf_targ_enr": float(
                    np.mean(
                        [
                            i["significant_enriched_TFtargets"]
                            for k, i in res.items()
                            if "_mean" in k
                        ]
                    )
                ),
                # 'grn_omni/ct': res['classif']['cell_type_ontology_term_id']['accuracy'],
            }
        )
        print(metrics)
        gc.collect()
    return metrics, tot


def relabel_assay_for_adv(label_decoders, labels_hierarchy):
    topred = [
        "EFO:0700011",
        "EFO:0008953",
        "EFO:0030003",
        "EFO:0700016",
        "EFO:0700010",
        "EFO:0022490",
        "EFO:0008722",
        "EFO:0010961",
        "EFO:0008720",
        "EFO:0010550",
        "EFO:0030004",
        "EFO:0008931",
        "EFO:0010010",
        "EFO:0008780",
        "EFO:0009919",
        "EFO:0010713",
        "EFO:0030059",
        "EFO:0008796",
        "EFO:0030002",
        "EFO:0030062",
        "EFO:0008919",
        "EFO:0008930",
        "EFO:0010184",
        "EFO:0008995",
        "EFO:0030080",
        "EFO:0700003",
    ]
    """
    relabel_assay_for_adv a method that groups assays so that the adv classification is easier and doesn't overfit

    Returns:
        dict: a remapping dictionary for grouping the labels to their coarser names when appropriate
    """
    relab = {}
    enc = {v: k for k, v in label_decoders["assay_ontology_term_id"].items()}
    for i in enc.keys():
        if i in topred:
            relab[enc[i]] = topred.index(i)
        else:
            relab[enc[i]] = -1
            prevlen = 10000
            for val in topred:
                li = labels_hierarchy["assay_ontology_term_id"].get(
                    enc.get(val, ""), []
                )
                if enc[i] in li and prevlen > len(li):
                    relab[enc[i]] = topred.index(val)
                    prevlen = len(li)
    return relab
