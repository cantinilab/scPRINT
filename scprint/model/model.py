# from scprint.base.base_model import BaseModel
import copy
import datetime
import os
from functools import partial

# from galore_torch import GaLoreAdamW
from math import factorial
from pathlib import Path
from typing import Dict, Optional

import lightning as L
import pandas as pd
import torch
import numpy as np
import torch.distributed
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.tuner.lr_finder import _LRCallback
from performer_pytorch import Performer
from scipy.sparse import load_npz
from simpler_flash import FlashTransformer
from torch import Tensor, nn, optim

# from .linear_transformer import FastTransformerEncoderWrapper as FastTransformer
from . import decoders, encoders, fsq, loss, utils
from .loss import grad_reverse
from .utils import WeightedMasker, simple_masker

FILEDIR = os.path.dirname(os.path.realpath(__file__))


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


class scPrint(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        genes: list,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 8,
        precpt_gene_emb: Optional[str] = None,
        memmap_gene_emb: bool = False,
        finetune_gene_emb: bool = False,
        freeze_embeddings: bool = True,
        gene_pos_enc: Optional[list] = None,
        normalization: str = "sum",
        attn_bias: str = "none",
        expr_encoder_layers: int = 3,
        transformer: str = "flash",  # "performer", "flash", "normal", "crisscross"
        expr_emb_style: str = "continuous",  # "binned_pos", "cont_pos", "metacell", "full_pos"
        n_input_bins: int = 0,
        mvc_decoder: str = "None",
        pred_embedding: list[str] = [],
        label_counts: Dict[str, int] = {},
        organisms: list[str] = [],
        layers_cls: list[int] = [],
        classes: Dict[str, int] = {},
        labels_hierarchy: Dict[str, Dict[int, list[int]]] = {},
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        class_compression: str = "none",  # "none", "fsq", "vae"
        compress_class_dim: Optional[Dict[str, int]] = None,
        cell_specific_blocks: bool = False,
        depth_atinput: bool = True,
        zinb: bool = True,
        splicing_head: bool = False,
        do_adv_cls: bool = False,
        dropout: float = 0.1,
        use_metacell_token: bool = False,
        lr: float = 0.0001,
        nb_features: Optional[int] = None,
        feature_redraw_interval: Optional[int] = None,
        num_heads_kv: int = 4,
        d_model_cell: int = 128,
        nhead_cell: int = 4,
        nlayers_cell: int = 6,
        num_heads_kv_cell: int = 4,
        **attention_kwargs,
    ):
        """
        scPRINT transformer for single cell biology and the inference of Gene Regulatory networks

        Args:
            genes (list): List of gene names the model will work with.
            precpt_gene_emb (np.array, optional): Gene embeddings of size (len(genes), d_model). Should be in the same order as the genes. Defaults to None.
            gene_pos_enc (list, optional): Gene position encoding of the same size as genes. Provides a location value for each gene in genes. Defaults to None.
            d_model (int, optional): Dimension of the model. Defaults to 512.
            nhead (int, optional): Number of heads in the multihead attention models. Defaults to 8.
            nlayers (int, optional): Number of layers in the transformer model. Defaults to 6.
            expr_encoder_layers (int, optional): Number of layers in the expression encoder. Defaults to 2.
            layers_cls (list[int], optional): List specifying the number of layers in the classifier. Defaults to [].
            classes (Dict[str, int], optional): Classes to predict with the number of classes for each. Defaults to {}.
            organisms (list[str], optional): List of organisms to use for plotting embeddings. Defaults to [].
            labels_hierarchy (Dict[str, Dict[int, list[int]]], optional): Class hierarchy for classes with hierarchical classes. Defaults to {}.
            dropout (float, optional): Dropout value. Defaults to 0.2.
            transformer (str, optional): Transformer type to use. One of "linear", "flash", "flashsparse", "scprint". Defaults to "fast".
            expr_emb_style (str, optional): Style of input embedding. One of "continuous", "binned_pos", "cont_pos", "metacell", "full_pos". Defaults to "continuous".
                "metacell" uses a DeepSet multi gene encoder across the KNN cells
                "full_pos" uses a positional encoding for each gene
                "binned_pos" uses a binned expr embedding for each gene
                "continuous" uses a continuous embedding for each gene with an MLP
            mvc_decoder (str, optional): Style of MVC decoder. One of "None", "inner product", "concat query", "sum query". Defaults to "None".
            pred_embedding (list[str], optional): List of classes to use for plotting embeddings. Defaults to [].
            freeze_embeddings (bool, optional): Whether to freeze the embeddings during training. Defaults to True.
            label_decoders (Optional[Dict[str, Dict[int, str]]], optional): Label decoders to use for plotting the UMAP during validations. Defaults to None.
            zinb (bool, optional): Whet her to use Zero-Inflated Negative Binomial distribution. Defaults to True.
            cell_transformer_layers (int, optional): Number of layers in the cell transformer. Defaults to 6.
            use_metacell_token (bool, optional): Whether to use a metacell token. Defaults to False.
            **attention_kwargs (dict): Additional keyword arguments for the model. see @flashformer.py

        Notes:
            for other parameters of the model that are not part of its class definition, see @trainer.trainer.py

        Raises:
            ValueError: If the expr_emb_style is not one of "continuous", "binned_pos", "metacell", "full_pos".
        """
        super().__init__()
        self.save_hyperparameters()
        # training flags
        self.do_denoise = True
        self.noise = [0.6]
        self.do_cce = False
        self.cce_temp = 0.3
        self.lr = 0.0001
        self.cce_scale = 0.2
        self.do_ecs = False
        self.ecs_threshold = 0.4
        self.ecs_scale = 0.2
        self.do_mvc = False
        self.mvc_scale = 1.0
        self.class_embd_diss_scale = 0.3
        self.do_adv_cls = do_adv_cls
        self.adv_class_scale = 1.0
        self.do_cls = False
        self.run_full_forward = True
        self.class_scale = 1
        self.zinb_and_mse = False
        self.do_next_tp = False
        self.do_generate = False
        self.var_context_length = False
        self.mask_ratio = []
        self.warmup_duration = 500
        self.weight_decay = 0.01
        self.optim = "adamW"
        self.fused_adam = False
        self.lr_reduce_patience = 2
        self.lr_reduce_factor = 0.6
        self.test_every = 20
        self.randsamp = True
        self.lr_reduce_monitor = "val_loss"
        self.name = ""
        self.set_step = None
        self.lrfinder_steps = 0
        self.doplot = True
        self.get_attention_layer = []
        self.embs = None
        self.pred_log_adata = True
        self.predict_depth_mult = 3
        self.predict_mode = "none"
        self.keep_all_labels_pred = False
        self.mask_zeros = False
        self.vae_kl_scale = 0.05
        self.vae_kl_warmup_steps = 40_000  # Default value, can be adjusted

        self.depth_atinput = depth_atinput
        self.tf_masker = WeightedMasker(genes, inv_weight=0.05)
        # should be stored somehow
        self.d_model = d_model
        self.normalization = normalization
        self.attn_bias = attn_bias
        self.organisms = organisms
        self.nlayers = nlayers
        self.gene_pos_enc = gene_pos_enc
        self.use_metacell_token = use_metacell_token
        self.mvc_decoder = mvc_decoder
        # need to store
        self.n_input_bins = n_input_bins
        self.transformer = transformer
        self.label_counts = classes
        self.classes = list(classes.keys())

        self.label_decoders = label_decoders
        self.pred_embedding = pred_embedding
        self.genes = genes
        self.vocab = {i: n for i, n in enumerate(genes)}
        self.expr_emb_style = expr_emb_style
        if self.expr_emb_style not in [
            "category",
            "continuous",
            "metacell",
            "full_pos",
        ]:
            raise ValueError(
                f"expr_emb_style should be one of category, continuous, scaling, "
                f"got {expr_emb_style}"
            )
        self.labels_hierarchy = labels_hierarchy
        self.hparams["labels_hierarchy"] = labels_hierarchy
        self.hparams["classes"] = classes
        self.hparams["label_decoders"] = label_decoders
        self.hparams["gene_pos_enc"] = gene_pos_enc
        self.hparams["genes"] = genes
        self.attn = utils.Attention(
            len(genes),
            additional_tokens=(
                len(classes)
                + (2 if self.depth_atinput else 1)
                + (1 if self.use_metacell_token else 0)
                if not cell_specific_blocks
                else 0
            ),
        )

        self.mat_labels_hierarchy = {}
        for k, v in labels_hierarchy.items():
            tens = torch.zeros((len(v), classes[k]))
            for k2, v2 in v.items():
                tens[k2 - classes[k], v2] = 1
            self.mat_labels_hierarchy[k] = tens.to(bool)

        # encoder
        # gene encoder
        if precpt_gene_emb is not None:
            embeddings = pd.read_parquet(precpt_gene_emb).loc[self.genes]
            if len(embeddings) == 0:
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(embeddings) < len(self.genes):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
            print("number of genes: ", len(embeddings))
            sembeddings = torch.nn.AdaptiveAvgPool1d(d_model)(
                torch.tensor(embeddings.values, dtype=torch.float32)
            )

            gene_encoder = encoders.GeneEncoder(
                len(self.vocab),
                d_model,
                # weights_file=precpt_gene_emb,
                weights=sembeddings,
                freeze=freeze_embeddings,
            )
        else:
            gene_encoder = encoders.GeneEncoder(
                len(self.vocab), d_model, freeze=freeze_embeddings
            )
        if finetune_gene_emb:
            if not freeze_embeddings:
                raise ValueError(
                    "finetune_gene_emb is True but freeze_embeddings is False"
                )
            # Create adapter layers after the frozen base encoder
            self.gene_encoder = torch.nn.Sequential(
                gene_encoder,
                torch.nn.Linear(d_model, d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(d_model, d_model),
            )
        else:
            self.gene_encoder = gene_encoder
        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if expr_emb_style in ["continuous", "full_pos"]:
            self.expr_encoder = encoders.ContinuousValueEncoder(
                d_model, dropout, layers=expr_encoder_layers
            )
        elif expr_emb_style == "binned_pos":
            assert n_input_bins > 0
            self.expr_encoder = encoders.CategoryValueEncoder(n_input_bins, d_model)
        elif expr_emb_style == "metacell":
            self.expr_encoder = encoders.GNN(
                1, d_model // 2, d_model, expr_encoder_layers, dropout, "deepset"
            )

        # Positional Encoding
        if self.gene_pos_enc is not None:
            max_len = max(gene_pos_enc)
            token_to_pos = {token: pos for token, pos in enumerate(self.gene_pos_enc)}
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, max_len=max_len, token_to_pos=token_to_pos
            )
        # Class Encoder
        # always have [base_cell_emb, time_embedding, depth_embedding] + any other class info
        # base cell embedding will store other cell specific information
        self.class_encoder = encoders.CategoryValueEncoder(
            len(self.classes) + 1,
            d_model if not cell_specific_blocks else d_model_cell,
        )
        # self.time_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        if self.depth_atinput:
            self.depth_encoder = encoders.ContinuousValueEncoder(
                d_model, dropout, layers=expr_encoder_layers
            )

        if self.use_metacell_token:
            self.metacell_encoder = encoders.CategoryValueEncoder(2, d_model)
        # compute tensor for mat_labels_hierarchy
        for i in [
            "strict_loading",
            "optim",
            "weight_decay",
            "d_hid",
            "edge_dim",
            "prenorm",
            "domain_spec_batchnorm",
            "use_flash_attn",
            "cell_emb_style",
            "num_batch_labels",
        ]:
            if i in attention_kwargs:
                attention_kwargs.pop(i)
        # Transformer
        # Linear
        if transformer == "linear":
            # linear transformer using the fast transformer package
            # self.transformer = FastTransformerEncoder(
            #    d_model, nhead, d_hid, nlayers, dropout, "linear"
            # )
            raise NotImplementedError("Linear transformer is not implemented")
        elif transformer == "performer":
            self.transformer = Performer(
                dim=d_model,
                depth=nlayers,
                heads=nhead,
                dim_head=d_model // nhead,
                causal=False,
                attn_dropout=dropout,
                ff_dropout=dropout,
                qkv_bias=True,
                nb_features=nb_features,
                feature_redraw_interval=feature_redraw_interval,
                **attention_kwargs,
            )
        else:
            self.transformer = FlashTransformer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                nlayers=nlayers,
                cross_attn=cell_specific_blocks,
                cross_dim=d_model_cell,
                attn_type=transformer,
                num_heads_kv=num_heads_kv,
                **attention_kwargs,
            )
        if cell_specific_blocks:
            attention_kwargs.pop("num_heads_kv", None)
            self.cell_transformer = FlashTransformer(
                d_model=d_model_cell,
                nhead=nhead_cell,
                num_heads_kv=num_heads_kv_cell,
                nlayers=nlayers_cell,
                dropout=dropout,
                cross_attn=True,
                cross_dim=d_model,
                attn_type=transformer,
                **attention_kwargs,
            )
        else:
            self.cell_transformer = None

        # decoders
        # expression
        self.expr_decoder = decoders.ExprDecoder(
            d_model,
            dropout=dropout,
            zinb=zinb,
            use_depth=not self.depth_atinput,
        )
        if splicing_head:
            self.splicing_head = decoders.ExprDecoder(
                d_model,
                dropout=dropout,
                zinb=zinb,
                use_depth=not self.depth_atinput,
            )
        else:
            self.splicing_head = None
        # cls decoder
        self.cls_decoders = torch.nn.ModuleDict()
        # should be a very simple classifier for most things
        # (maybe scale with the number of classes) should be 1 layer...
        for clss, n_cls in classes.items():
            self.cls_decoders[clss] = decoders.ClsDecoder(
                d_model if not cell_specific_blocks else d_model_cell,
                n_cls,
                layers=layers_cls,
                dropout=dropout,
            )
            if clss == "assay_ontology_term_id" and self.do_adv_cls:
                self.adv_cls_decoder = decoders.ClsDecoder(
                    d_model if not cell_specific_blocks else d_model_cell,
                    n_cls,
                    layers=layers_cls,
                    dropout=dropout,
                )
        # expression decoder from batch embbedding
        if mvc_decoder != "None":
            if cell_specific_blocks:
                raise ValueError(
                    "MVC decoder is not supported for cell specific blocks"
                )
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder,
                zinb=zinb,
                # use_depth=not self.depth_atinput,
            )
        else:
            self.mvc_decoder = None

        self.apply(
            partial(
                utils._init_weights,
                n_layer=nlayers,
            )
        )
        for i, dec in self.cls_decoders.items():
            torch.nn.init.constant_(dec.out_layer.bias, -0.13)

        if compress_class_dim is not None:
            self.compressor = torch.nn.ModuleDict()
            dim = d_model_cell if cell_specific_blocks else self.d_model
            for k, v in compress_class_dim.items():
                if v >= 8:
                    self.compressor[k] = decoders.VAEDecoder(
                        dim,
                        layers=[
                            128,
                            compress_class_dim[k],
                        ],
                        dropout=dropout,
                        return_latent=True,
                    )
                else:
                    self.compressor[k] = fsq.FSQ(levels=[2] * v, dim=dim)
        else:
            self.compressor = None

    def on_load_checkpoint(self, checkpoints):
        # if not the same number of labels (due to diff datasets)
        for name, clss in self.cls_decoders.items():
            size = checkpoints["state_dict"][
                "cls_decoders." + name + ".out_layer.bias"
            ].shape[0]
            if size != clss.out_layer.bias.shape[0]:
                self.cls_decoders[name].out_layer = torch.nn.Linear(
                    clss.out_layer.weight.shape[1], size
                )

        # from older model versions
        self.normalization = checkpoints["hyper_parameters"].get("normalization", "sum")
        if (
            checkpoints["state_dict"].get("gene_encoder.0.embedding.weight", None)
            is not None
        ):
            # replace it with the new one gene_encoder.0.embeddings.weight in the state_dict
            checkpoints["state_dict"]["gene_encoder.0.embeddings.weight"] = checkpoints[
                "state_dict"
            ]["gene_encoder.0.embedding.weight"]
            del checkpoints["state_dict"]["gene_encoder.0.embedding.weight"]
        # same
        # when doing batch effect correction and input dataset is not the same
        if (
            "grad_reverse_discriminator_loss.out_layer.bias"
            in checkpoints["state_dict"]
        ):
            for k in list(checkpoints["state_dict"].keys()):
                if "grad_reverse_discriminator_loss" in k:
                    del checkpoints["state_dict"][k]
            print(
                "the discriminator for batch effect correction has been removed. "
                "dropping the legacy key."
            )
        # same
        if (
            checkpoints["state_dict"].get("gene_encoder.embedding.weight", None)
            is not None
        ):
            # replace it with the new one gene_encoder.embeddings.weight in the state_dict
            checkpoints["state_dict"]["gene_encoder.embeddings.weight"] = checkpoints[
                "state_dict"
            ]["gene_encoder.embedding.weight"]
            del checkpoints["state_dict"]["gene_encoder.embedding.weight"]

        if "classes" in checkpoints["hyper_parameters"]:
            if self.label_counts != checkpoints["hyper_parameters"]["classes"]:
                if "label_counts" in checkpoints["hyper_parameters"] and set(
                    checkpoints["hyper_parameters"]["label_counts"].keys()
                ) == set(checkpoints["hyper_parameters"]["classes"]):
                    if self.classes != checkpoints["hyper_parameters"]["classes"]:
                        print("classes have changed, be careful")
                    self.classes = checkpoints["hyper_parameters"]["classes"]
                    self.label_counts = checkpoints["hyper_parameters"]["label_counts"]
                    if self.classes == self.label_counts:
                        raise ValueError(
                            "classes and label_counts are the same, this is not allowed, please use another checkpoint"
                        )
                else:
                    self.label_counts = checkpoints["hyper_parameters"]["classes"]
                    if self.classes != list(
                        checkpoints["hyper_parameters"]["classes"].keys()
                    ):
                        print("classes have changed, be careful")
                        self.classes = list(
                            checkpoints["hyper_parameters"]["classes"].keys()
                        )
            # else it is all good as expected

        else:
            print("no classes in the checkpoint, be careful")

        if (
            self.label_decoders != checkpoints["hyper_parameters"]["label_decoders"]
            or self.labels_hierarchy
            != checkpoints["hyper_parameters"]["labels_hierarchy"]
        ):
            print("label decoders have changed, be careful")
            self.label_decoders = checkpoints["hyper_parameters"]["label_decoders"]
            self.labels_hierarchy = checkpoints["hyper_parameters"]["labels_hierarchy"]
            for k, v in self.labels_hierarchy.items():
                tens = torch.zeros((len(v), self.label_counts[k]))
                for k2, v2 in v.items():
                    tens[k2 - self.label_counts[k], v2] = 1
                self.mat_labels_hierarchy[k] = tens.to(bool)

        if "gene_pos_enc" in checkpoints["hyper_parameters"]:
            if self.gene_pos_enc != checkpoints["hyper_parameters"]["gene_pos_enc"]:
                print(
                    "Gene position encoding has changed in the dataloader compared to last time, trying to revert"
                )
                self.gene_pos_enc = checkpoints["hyper_parameters"]["gene_pos_enc"]
                max_len = max(self.gene_pos_enc)
                token_to_pos = {
                    token: pos for token, pos in enumerate(self.gene_pos_enc)
                }
                self.pos_encoder = encoders.PositionalEncoding(
                    self.d_model, max_len=max_len, token_to_pos=token_to_pos
                )
        mencoders = {}
        if self.label_decoders != checkpoints["hyper_parameters"]["label_decoders"]:
            raise ValueError("label decoders have changed")
        try:
            if self.trainer.datamodule.decoders != self.label_decoders:
                print("label decoders have changed, be careful")
                # if we don't have the same decoders, we need to update the one on the datamodule side
                for k, v in checkpoints["hyper_parameters"]["label_decoders"].items():
                    mencoders[k] = {va: ke for ke, va in v.items()}
                self.trainer.datamodule.encoders = mencoders
        except RuntimeError as e:
            if "scPrint is not attached to a `Trainer`." in str(e):
                print("FYI: scPrint is not attached to a `Trainer`.")
            else:
                raise e
            if self.genes != checkpoints["hyper_parameters"]["genes"]:
                self.genes = checkpoints["hyper_parameters"]["genes"]
                try:
                    self.trainer.datamodule.genes = self.genes
                except RuntimeError as e:
                    if "scPrint is not attached to a `Trainer`." not in str(e):
                        raise e
            if self.organisms != checkpoints["hyper_parameters"]["organisms"]:
                self.organisms = checkpoints["hyper_parameters"]["organisms"]
                try:
                    self.trainer.datamodule.organisms = self.organisms
                except RuntimeError as e:
                    if "scPrint is not attached to a `Trainer`." not in str(e):
                        raise e

        if not is_interactive():
            self.save_hyperparameters()

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        neighbors: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        req_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,  # (minibatch, n_labels, embsize)
        metacell_token: Optional[Tensor] = None,  # (minibatch, 1)
    ):
        """
        _encode given inputs to the model encode into embeddings.

        Args:
            @see self.forward()

        Returns:
            Tensor: the encoded data
        """
        enc = self.gene_encoder(gene_pos)  # (minibatch, seq_len, embsize)
        self.cur_gene_token_embs = enc.clone()
        if expression is not None:
            if self.normalization == "sum":
                expression = expression / expression.sum(1).unsqueeze(1)
                if neighbors is not None:
                    neighbors = neighbors / neighbors.sum(2).unsqueeze(1)
            elif self.normalization == "log":
                expression = torch.log2(1 + expression)
                if neighbors is not None:
                    neighbors = torch.log2(1 + neighbors)
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
            if neighbors is not None:
                expr_emb = self.expr_encoder(expression, mask=mask, neighbors=neighbors)
            else:
                expr_emb = self.expr_encoder(expression, mask=mask)
            enc.add_(expr_emb)
        if self.gene_pos_enc:
            enc.add_(self.pos_encoder(gene_pos))
        if cell_embs is None:
            cell_embs = self.class_encoder(
                torch.arange(
                    len(self.classes) + 1,
                    device=gene_pos.device,
                ).repeat(gene_pos.shape[0], 1)
            )
            if timepoint is not None:
                pass
                # cell_embs[:, 2, :] = self.time_encoder(timepoint)
        if self.use_metacell_token:
            metacell_token = (
                metacell_token
                if metacell_token is not None
                else torch.zeros(gene_pos.shape[0], device=gene_pos.device)
            )
            enc = torch.cat(
                (self.metacell_encoder(metacell_token).unsqueeze(1), enc),
                dim=1,
            )
        if req_depth is not None:
            depth_encoded = self.depth_encoder(torch.log2(1 + req_depth)).unsqueeze(1)
            enc = torch.cat((depth_encoded, enc), dim=1)
        return cell_embs, enc  # self.norm_and_dropout(enc)
        # we already apply prenorm & dropout  # (minibatch, seq_len, embsize)

    def _decoder(
        self,
        transformer_output,
        cell_embs,
        depth_mult,
        get_gene_emb=False,
        do_sample=False,
        do_mvc=False,
        do_class=False,
        req_depth: Optional[Tensor] = None,
        splicing_mult: Optional[Tensor] = None,
    ):
        """
        _decoder given the transformer output, decode into the final output.

        Args:
            @see self.forward()

        Returns:
            dict: the output of the model
        """
        to_skip = (1 if self.use_metacell_token else 0) + (
            1 if self.depth_atinput else 0
        )
        if req_depth is not None:
            req_depth = torch.log2(1 + req_depth)
        output = self.expr_decoder(transformer_output[:, to_skip:, :], req_depth)

        output["mean"] = depth_mult.unsqueeze(1) * output["mean"]
        if do_sample:
            pass
        if self.splicing_head is not None:
            splicing_output = self.splicing_head(
                transformer_output[:, to_skip:, :], req_depth
            )
            output.update({"spl_" + k: v for k, v in splicing_output.items()})
            output["spl_mean"] = splicing_mult.unsqueeze(1) * output["spl_mean"]

        if self.compressor is not None and do_class:
            # Apply VAE to cell embeddings
            output["vae_kl_loss"] = 0
            res = []
            if "default" in self.compressor:
                out = self.compressor["default"](cell_embs[:, 0, :])
                res.append(out[0].unsqueeze(1))
                if len(out) == 4:
                    output["vae_kl_loss"] += out[3]
            else:
                res.append(cell_embs[:, 0, :].unsqueeze(1))
            for i, clsname in enumerate(self.classes):
                out = self.compressor[clsname](cell_embs[:, i + 1, :])
                res.append(out[0].unsqueeze(1))
                if len(out) == 4:
                    output["vae_kl_loss"] += out[3]
            output["cell_embs"] = torch.cat(res, dim=1)
        else:
            output["cell_embs"] = cell_embs
        output["cell_emb"] = torch.mean(output["cell_embs"], dim=1)
        if len(self.classes) > 0 and do_class:
            for i, clsname in enumerate(self.classes):
                output.update(
                    {
                        "cls_output_" + clsname: self.cls_decoders[clsname](
                            cell_embs[:, i + 1, :]
                        )
                    }
                )
        if do_mvc:
            output.update(
                self.mvc_decoder(
                    output["cell_emb"],
                    self.cur_gene_token_embs,
                    req_depth=req_depth if not self.depth_atinput else None,
                )
            )
            output["mvc_mean"] = (
                depth_mult.unsqueeze(1) * output["mvc_mean"]
            )  # (minibatch, seq_len)

        if get_gene_emb:
            output["gene_embedding"] = transformer_output[
                :, to_skip:, :
            ]  # (minibatch, seq_len, embsize)
        return output

    def forward(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        neighbors: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        req_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,  # (new_minibatch_of_nxt_cells,)
        get_gene_emb: bool = False,
        metacell_token: Optional[Tensor] = None,  # (minibatch, 1)
        depth_mult: Optional[Tensor] = None,
        do_sample: bool = False,
        do_mvc: bool = False,
        do_class: bool = False,
        get_attention_layer: list = [],
    ):
        """
        forward also called on self(), a full forward pass on the model

        Args:
            gene_pos (Tensor): A tensor of shape (minibatch, seq_len)
                representing the genes used for each cell in the minibatch.
            expression (Tensor, optional): A tensor of shape (minibatch, seq_len)
                representing the expression levels of genes in the minibatch. Defaults to None.
            neighbors (Tensor, optional): A tensor of shape (minibatch, seq_len, n_neighbors)
                representing the neighbors of each gene in the minibatch. Defaults to None.
            mask (Tensor, optional): A tensor of shape (minibatch, seq_len)
                used to mask certain elements in the sequence during the forward pass. Defaults to None.
            req_depth (Tensor, optional): A tensor of shape (minibatch,)
                representing the full depth of each sequence in the minibatch. Defaults to None.
            depth_mult (Tensor, optional): A tensor of shape (minibatch,)
                representing the depth multiplier for each sequence in the minibatch. Defaults to None.
            timepoint (Tensor, optional): A tensor of shape (minibatch,)
                representing the timepoint associated with each sequence in the minibatch. Defaults to None.
            get_gene_emb (bool, optional): A flag indicating whether to return the gene embeddings.
                If True, the gene embeddings are included in the output. Defaults to False.
            do_sample (bool, optional): A flag indicating whether to sample the expression levels.
                If True, the expression levels are sampled during the forward pass. Defaults to False.
            do_mvc (bool, optional): A flag indicating whether to perform multi-view coding.
                If True, the multi-view coding is performed during the forward pass. Defaults to False.
            do_class (bool, optional): A flag indicating whether to perform classification.
                If True, the classification is performed during the forward pass. Defaults to False.
            get_attention_layer (list, optional): A list indicating which attention layers to return.
                If not empty, the specified attention layers are included in the output. Defaults to [].

        Returns:
            dict of output Tensors: A dictionary containing the output tensors from the forward pass.
                The keys of the dictionary depend on the input flags (get_gene_emb, do_sample, get_attention_layer).
                at minima, the dictionary codntains the following:
                - "mean": the mean expression levels
                - "zero_logits": the logits for zero-inflated expression levels
                - "disp": the dispersion parameter
                - "cell_embs": the cell embeddings per class
                - "cell_emb": the main cell embedding
                - "cls_output": the output of the classifier
        """
        cell_embs, encoding = self._encoder(
            gene_pos,
            expression,
            neighbors,
            mask,
            req_depth=req_depth if self.depth_atinput else None,
            timepoint=timepoint,
            metacell_token=metacell_token,
        )
        num = (
            (1 if self.use_metacell_token else 0)
            + (1 if self.depth_atinput else 0)
            + (len(self.classes) + 1 if not self.cell_transformer else 0)
        )
        if self.attn_bias != "none":
            if not hasattr(self, "nbias"):
                bias_path = os.path.join(
                    Path(FILEDIR).parent.parent, "data", "bias_sparse.npz"
                )
                self.nbias = torch.Tensor(load_npz(bias_path).todense()).to(
                    device=gene_pos.device, dtype=torch.float16
                )
            bias = torch.zeros(
                (
                    gene_pos.shape[0],
                    gene_pos.shape[1] + num,
                    gene_pos.shape[1] + num,
                ),
                device=gene_pos.device,
                dtype=torch.float16,
            )
            # fade slowly through the iterations
            fade_factor = 40000 / (400 + self.trainer.global_step * 2)
            # bias[:, num:, :num] = -10_000  # do not pay attention to the cls embeddings
            bias[:, num:, num:] = (
                self.nbias[gene_pos[:, :, None], gene_pos[:, None, :]] * fade_factor
            )
        if not self.cell_transformer:
            encoding = torch.cat([cell_embs, encoding], dim=1)
        if type(self.transformer) is FlashTransformer:
            if self.mask_zeros:
                mask_zeros = torch.cat(
                    [
                        torch.ones(
                            expression.shape[0],
                            num,
                            dtype=torch.bool,
                            device=expression.device,
                        ),
                        expression != 0,
                    ],
                    dim=1,
                )

            transformer_output = self.transformer(
                encoding,
                return_qkv=get_attention_layer,
                bias=bias if self.attn_bias != "none" else None,
                bias_layer=list(range(self.nlayers - 1)),
                mask_zeros=mask_zeros if self.mask_zeros else None,
            )
        elif type(self.transformer) is Performer:
            transformer_output = self.transformer(encoding)
        else:
            raise ValueError(f"Unknown transformer: {type(self.transformer)}")
        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output
        if self.cell_transformer:
            cell_embs = self.cell_transformer(cell_embs, x_kv=transformer_output)
        else:
            cell_embs, transformer_output = transformer_output.split(
                [
                    len(self.classes) + 1,
                    transformer_output.shape[1] - (len(self.classes) + 1),
                ],
                dim=1,
            )
        # if not provided we will mult by the current expression sum
        depth_mult = expression.sum(1) if depth_mult is None else depth_mult
        res = self._decoder(
            transformer_output,
            cell_embs,
            depth_mult,
            get_gene_emb,
            do_sample,
            do_mvc,
            do_class,
            req_depth=req_depth if not self.depth_atinput else None,
        )
        return (res, qkvs) if len(get_attention_layer) > 0 else res

    def _generate(
        self,
        cell_embs: Tensor,
        gene_pos: Tensor,
        depth_mult: Tensor,
        req_depth: Optional[Tensor] = None,
        metacell_token: Optional[Tensor] = None,
        **decoder_kwargs,
    ):
        """
        _generate given cell_embeddings, generate an expression profile

        the goal was to iterate multiple times,
        to create a trajectory and reach a certain state
        should call forward multiple times

        Args:
            cell_emb(:obj:`Tensor`): A tensor representing cell embeddings. It has a shape of (minibatch, embsize).
            src(:obj:`Tensor`): A tensor representing the source data. It has a shape of (minibatch, seq_len).
            values(:obj:`Tensor`): An optional tensor representing the values. It has a shape of (minibatch, seq_len).
            gen_iters(:obj:`int`): An integer representing the number of generation iterations.
            classes(:obj:`Tensor`): An optional tensor representing the classes. It has a shape of (batch,).
        """
        _, encoding = self._encoder(
            cell_embs=cell_embs,
            gene_pos=gene_pos,
            req_depth=req_depth if self.depth_atinput else None,
            metacell_token=metacell_token,
        )
        if self.cell_transformer:
            transformer_output = self.transformer(encoding, x_kv=cell_embs)
        else:
            encoding = torch.cat([cell_embs, encoding], dim=1)
            transformer_output = self.transformer(encoding)
            cell_embs, transformer_output = transformer_output.split(
                [
                    len(self.classes) + 1,
                    transformer_output.shape[1] - (len(self.classes) + 1),
                ],
                dim=1,
            )
        output = self._decoder(
            transformer_output,
            cell_embs=cell_embs,
            depth_mult=depth_mult,
            req_depth=req_depth if not self.depth_atinput else None,
            **decoder_kwargs,
        )
        output.pop("cell_embs")
        output.pop("cell_emb")
        return output  # (minibatch, seq_len)

    def configure_optimizers(self):
        """@see pl.LightningModule"""
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        # not working because of poor weight decay implem
        if self.optim == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "adamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "galore":
            raise NotImplementedError("Galore optimizer not implemented")
            # param_groups = [
            #    {
            #        "params": [
            #            v for k, v in self.named_parameters() if "transformer" not in k
            #        ]
            #    },
            #    {
            #        "params": [
            #            v for k, v in self.named_parameters() if "transformer" in k
            #        ],
            #        "rank": 128,
            #        "update_proj_gap": 200,
            #        "scale": 0.25,
            #        "proj_type": "std",
            #    },
            # ]
            # optimizer = GaLoreAdamW(param_groups, lr=self.hparams.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")
        if self.lr_reduce_monitor is None:
            print("no lr reduce factor")
            return [optimizer]
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_reduce_patience,
            factor=self.lr_reduce_factor,
            verbose=True,
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.lr_reduce_monitor,
        }
        self.lrfinder_steps = 0
        for val in self.trainer.callbacks:
            if type(val) is _LRCallback:
                self.lrfinder_steps = val.num_training
            if type(val) is LearningRateFinder:
                self.lrfinder_steps = val._num_training_steps
        return [optimizer], [lr_dict]

    def on_fit_start(self):
        """@see pl.LightningModule"""
        if type(self.transformer) is FlashTransformer:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(True)
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx,
    ):
        """
        training_step defines the train loop. It is independent of forward

        @see pl.LightningModule

        Returns:
            _type_: _description_
        """
        total_loss, losses = self._full_training(
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_temp=self.cce_temp,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        try:
            self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
            self.log_dict(losses, prog_bar=True, sync_dist=True)
        except Exception as e:
            print(e)
            print(losses)
        return total_loss

    def _full_training(
        self,
        batch: Dict[str, Tensor],
        do_denoise: bool = False,
        noise: list[float] = [],
        do_next_tp: bool = False,
        do_cce: bool = False,
        cce_temp: float = 0.5,
        do_ecs: bool = False,
        do_mvc: bool = False,
        do_adv_cls: bool = False,
        do_cls: bool = False,
        do_generate: bool = False,
        run_full_forward: bool = True,
        mask_ratio: list[float] = [0.15],
        do_vae_kl: bool = True,
    ):
        """
        _full_training implement the trainng steps: forward (multiple sometimes), loss

        Args:
            batch (dict[Tensors]): A dictionary containing tensors for the training batch:
                - "x": the expression levels of genes in the minibatch
                - "genes": the genes used for each cell in the minibatch
                - "class": the class to predict for each cell
                - "depth": the full depth of each cell in the minibatch
            do_denoise (bool, optional): A flag to indicate whether to perform denoising. Defaults to False.
            noise (list[float], optional): A list of noise levels to be used in denoising. Defaults to [].
            do_next_tp (bool, optional): A flag to indicate whether to perform next time point prediction. Defaults to False.
            do_cce (bool, optional): A flag to indicate whether to perform cross-categorical entropy. Defaults to False.
            cce_temp (float, optional): The similarity threshold for cross-categorical entropy. Defaults to 0.5.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity. Defaults to False.
            do_mvc (bool, optional): A flag to indicate whether to perform multi-view coding. Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification. Defaults to False.
            do_generate (bool, optional): A flag to indicate whether to perform data generation. Defaults to False.
            run_full_forward (bool, optional): A flag to indicate whether to perform a full forward pass. Defaults to True.
            mask_ratio (list, optional): A list of mask ratios to be used in the training. Defaults to [0.15].
            do_vae_kl (bool, optional): A flag to indicate whether to perform VAE KL loss. Defaults to True.

        Returns:
            loss, losses: the total loss as float and the individual losses as dict
        """
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio]
        # dynamically change the context length every 5 steps
        if self.var_context_length and self.trainer.global_step % 5 == 0:
            context_length = torch.randint(400, batch["x"].shape[1], (1,)).item()
        else:
            context_length = batch["x"].shape[1]
        expression = batch["x"][:, :context_length]
        gene_pos = batch["genes"][:, :context_length]
        total_count = batch["depth"]
        clss = batch.get("class", None)
        batch_idx = batch.get("dataset", None)
        metacell_token = batch.get("is_meta", None)
        if metacell_token is None:
            if self.use_metacell_token:
                raise ValueError(
                    "metacell_token is not provided but use_metacell_token is True"
                )

        knn_cells = batch.get("knn_cells", None)
        if knn_cells is not None:
            knn_cells = knn_cells[:, :, :context_length]
        if self.mask_zeros and knn_cells is None:
            keep = expression.sum(0) != 0
            # we can work on smaller datasets
            if keep.sum() != keep.shape[0]:
                expression = expression[:, keep]
                gene_pos = gene_pos[:, keep]
        if self.transformer.attn_type == "hyper":
            # seq len must be a multiple of 128
            num = (
                (1 if self.use_metacell_token else 0)
                + (1 if self.depth_atinput else 0)
                + (len(self.classes) + 1 if not self.cell_transformer else 0)
            )
            if (expression.shape[1] + num) % 128 != 0:
                expression = expression[:, : ((expression.shape[1]) // 128 * 128) - num]
                gene_pos = gene_pos[:, : ((gene_pos.shape[1]) // 128 * 128) - num]
                if knn_cells is not None:
                    knn_cells = knn_cells[
                        :, :, : ((knn_cells.shape[2]) // 128 * 128) - num
                    ]
        total_loss = 0
        losses = {}
        cell_embs = []
        if run_full_forward:
            output = self.forward(
                gene_pos,
                expression,
                neighbors=knn_cells,
                mask=None,
                req_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
                metacell_token=metacell_token,
            )
            if "disp" in output:
                output.pop("disp")
            if "zero_logits" in output:
                output.pop("zero_logits")
            if "mean" in output:
                output.pop("mean")
            l, tot = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_vae_kl=do_vae_kl,
            )
            cell_embs.append(output["cell_emb"].clone())
            full_cell_embs = output["cell_embs"].clone()
            total_loss += tot
            losses.update({"full_forward_" + k: v for k, v in l.items()})
            do_mvc = False
            do_cls = False

        for i in mask_ratio:
            # do noise and mask
            if do_denoise and False:
                if knn_cells is not None:
                    knn_cells = utils.downsample_profile(
                        knn_cells, dropout=0.5, randsamp=self.randsamp
                    )
                    expr = expression
                else:
                    expr = utils.downsample_profile(
                        expression, dropout=0.5, randsamp=self.randsamp
                    )
            else:
                expr = expression
            if i == "TF":
                mask = self.tf_masker(
                    ids=gene_pos,
                    mask_ratio=0.4,
                ).to(gene_pos.device)
            else:
                mask = simple_masker(
                    shape=gene_pos.shape,
                    mask_ratio=i,
                ).to(gene_pos.device)
            output = self.forward(
                gene_pos,
                expression=expr,
                neighbors=knn_cells,
                mask=mask,
                req_depth=expr.sum(1),
                do_mvc=do_mvc,
                do_class=do_cls,
                metacell_token=metacell_token,
            )
            l, tot = self._compute_loss(
                output,
                expr,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_mse=self.zinb_and_mse,
                do_vae_kl=do_vae_kl,
            )
            # we only want to do them once
            do_mvc = False
            do_cls = False

            cell_embs.append(output["cell_emb"].clone())
            total_loss += tot
            pct = str(int(i * 100)) + "%_" if i != "TF" else "TF_"
            losses.update({"mask_" + pct + k: v for k, v in l.items()})
        # TASK 3. denoising
        if do_denoise:
            for i in noise:
                if i == 1.0:
                    expr = torch.zeros_like(expression)
                else:
                    expr = utils.downsample_profile(
                        expression, dropout=i, randsamp=self.randsamp
                    )
                if knn_cells is not None:
                    # knn_cells = utils.downsample_profile(
                    #    knn_cells, dropout=i, randsamp=self.randsamp
                    # )
                    pass
                output = self.forward(
                    gene_pos,
                    expression=expr,
                    neighbors=knn_cells,
                    mask=None,
                    depth_mult=expression.sum(1),
                    req_depth=total_count,
                    do_mvc=do_mvc,
                    do_class=do_cls,
                    metacell_token=metacell_token,
                )
                l, tot = self._compute_loss(
                    output,
                    expression,
                    clss,
                    batch_idx,
                    do_ecs,
                    do_adv_cls & do_cls,
                    do_mse=self.zinb_and_mse,
                    do_vae_kl=do_vae_kl,
                )
                do_mvc = False
                do_cls = False

                cell_embs.append(output["cell_emb"].clone())
                total_loss += tot
                losses.update(
                    {"denoise_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
                )
                # make sure that the cell embedding stay the same even if the expression is decreased

        # TASK 6. expression generation
        if do_generate:
            output = self._generate(
                cell_embs=output["cell_embs"]
                if not run_full_forward
                else full_cell_embs,
                gene_pos=gene_pos,
                depth_mult=expression.sum(1),
                req_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            if "cell_emb" in output:
                cell_embs.append(output["cell_emb"].clone())
            l, tloss = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                ("cell_emb" in output) and do_ecs,
                do_adv_cls & do_cls,
                do_mse=self.zinb_and_mse,
                do_vae_kl=do_vae_kl,
            )
            losses.update({"gen_" + k: v for k, v in l.items()})
            total_loss += tloss

        # TASK 7. next time point prediction
        if do_next_tp:
            pass
        # TASK 4. contrastive cell embedding
        if do_cce:
            loss_cce = 0
            n_pairs = 0
            for i, cell_emb1 in enumerate(cell_embs[:-1]):
                for cell_emb2 in cell_embs[(i + 1) :]:
                    loss_cce += loss.contrastive_loss(
                        cell_emb1, cell_emb2, cce_temp
                    )  # (nlabels, minibatch, minibatch)
                    n_pairs += 1
            avg_loss_cce = loss_cce / max(n_pairs, 1)
            total_loss += avg_loss_cce * self.cce_scale
            # TASK 3b. contrastive graph embedding
            losses.update({"cce": avg_loss_cce})

        # TASK 8. KO profile prediction
        # if we have that information
        # TASK 9. PDgrapher-drug-like perturbation prediction (L1000?)
        return total_loss, losses

    def _compute_loss(
        self,
        output,
        expression,
        clss,
        batch_idx,
        do_ecs=False,
        do_adv_cls=False,
        do_mse=0,
        do_vae_kl=False,
        spl_expression=None,
    ):
        """
        _compute_loss compute the loss of the model given output from the forward pass

        Args:
            output (dict): A dictionary containing the output of the forward pass.
            expression (Tensor): A tensor containing the expression levels of genes.
            mask (Tensor): A tensor indicating the masked positions in the input data.
            clss (Tensor): A tensor containing the class classes for each cell.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity.
                Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification.
                Defaults to False.
            do_mse (float, optional): A scaling factor to indicate whether and how much to weight mean
            squared error loss in addition to zinb loss.
                Defaults to 0.
            do_vae_kl (bool, optional): A flag to indicate whether to perform VAE KL loss.
                Defaults to False.

        Raises:
            ValueError: Raised when an invalid operation or input is encountered.

        Returns:
            tuple: A tuple containing the total loss as a float and the individual losses as a dictionary.
        """
        total_loss = 0
        losses = {}
        # TASK 1. reconstruct masked expression
        if "zero_logits" in output:
            loss_expr = loss.zinb(
                theta=output["disp"],
                pi=output["zero_logits"],
                mu=output["mean"],
                target=expression,
            )
            if do_mse:
                loss_expr += (
                    loss.mse(
                        input=torch.log(output["mean"] + 1)
                        * (1 - torch.sigmoid(output["zero_logits"])),
                        target=torch.log(expression + 1),
                    )
                    / 10  # scale to make it more similar to the zinb
                )
            if self.splicing_head is not None:
                loss_nov_expr = loss.zinb(
                    theta=output["spl_disp"],
                    pi=output["spl_zero_logits"],
                    mu=output["spl_mean"],
                    target=spl_expression,
                )
        elif "disp" in output:
            loss_expr = loss.nb(
                theta=output["disp"],
                mu=output["mean"],
                target=expression,
            )
            if self.splicing_head is not None:
                loss_nov_expr = loss.nb(
                    theta=output["spl_disp"],
                    mu=output["spl_mean"],
                    target=spl_expression,
                )
        elif "mean" in output:
            loss_expr = loss.mse(
                input=output["mean"],
                target=expression,
            )
            if self.splicing_head is not None:
                loss_nov_expr = loss.mse(
                    input=output["spl_mean"],
                    target=spl_expression,
                )
        else:
            loss_expr = 0
        total_loss += loss_expr
        losses.update({"expr": loss_expr})
        if self.splicing_head is not None:
            losses.update({"spl_expr": loss_nov_expr})
            total_loss += loss_nov_expr

        # TASK 2. predict classes
        if len(self.classes) > 0 and "cell_embs" in output:
            ## Calculate pairwise cosine similarity for the embeddings
            # Calculate pairwise cosine similarity more efficiently
            if do_ecs:
                loss_emb_indep = loss.within_sample(output["cell_embs"])
                losses.update({"emb_independence": loss_emb_indep})
                total_loss += self.class_embd_diss_scale * loss_emb_indep
            ## compute class loss
            loss_cls = 0
            loss_adv_cls = 0
            for j, clsname in enumerate(self.classes):
                if "cls_output_" + clsname not in output:
                    continue
                # setting the classes from index to one hot
                try:
                    loss_cls += loss.hierarchical_classification(
                        pred=output["cls_output_" + clsname],
                        cl=clss[:, j],
                        labels_hierarchy=self.mat_labels_hierarchy[clsname]
                        if clsname in self.mat_labels_hierarchy.keys()
                        else None,
                    )
                except:
                    print(clsname)
                    print(self.mat_labels_hierarchy)
                    raise

                # Adversarial part for 'assay_ontology_term_id'
                if do_adv_cls and clsname == "assay_ontology_term_id":
                    pos = self.classes.index("cell_type_ontology_term_id")
                    loc = (
                        pos  # Assuming 'j' correctly corresponds to 'assay_ontology_term_id' index
                        + (2 if self.depth_atinput else 1)
                        + (1 if self.use_metacell_token else 0)
                    )
                    # Apply gradient reversal to the input embedding
                    adv_input_emb = loss.grad_reverse(
                        output["cell_embs"][:, loc, :].clone(), lambd=1.0
                    )
                    # Get predictions from the adversarial decoder
                    adv_pred = self.adv_cls_decoder(adv_input_emb)

                    # Compute the adversarial loss
                    current_adv_loss = loss.hierarchical_classification(
                        pred=adv_pred,
                        cl=clss[
                            :, j
                        ],  # Use the true label for the adversarial target class
                        labels_hierarchy=self.mat_labels_hierarchy[clsname]
                        if clsname in self.mat_labels_hierarchy.keys()
                        else None,
                    )
                    # Add the adversarial loss to the total loss (gradient reversal handles the maximization objective for the generator)
                    total_loss += self.adv_class_scale * current_adv_loss
                    losses.update({"adv_cls": current_adv_loss})

                # This was the old (likely incorrect) way for reference, now handled above
                # if do_adv_cls and clsname == "assay_ontology_term_id":
                #     loss_adv_cls = loss.hierarchical_classification(
                #         pred=output["adv_cls_output"],
                #         cl=clss[:, j],
                #         labels_hierarchy=self.mat_labels_hierarchy[clsname]
                #         if clsname in self.mat_labels_hierarchy.keys()
                #         else None,
                #     )
                #     total_loss -= self.adv_class_scale * loss_adv_cls # Incorrect subtraction
                #     losses.update({"adv_cls": loss_adv_cls})
            total_loss += self.class_scale * loss_cls
            if loss_cls != 0:
                losses.update({"cls": loss_cls})
            # TASK 2bis. adversarial label prediction
            # if do_adv_cls:
            #    embs = output["cell_embs"][
            #        :,
            #        (2 if self.depth_atinput else 1)
            #        + (1 if self.use_metacell_token else 0) :,
            #        :,
            #    ].clone()
            #    for j, adv_cls in enumerate(self.classes):
            #        ind = torch.arange(len(self.classes))
            #        mean_embs = torch.mean(embs[:, ind != j, :], dim=1)
            #        mean_embs = grad_reverse(mean_embs, lambd=1.0)
            #        adv_pred = self.cls_decoders[adv_cls](mean_embs)
            #        loss_adv_cls += loss.hierarchical_classification(
            #            pred=adv_pred,
            #            cl=clss[:, j],
            #            labels_hierarchy=self.mat_labels_hierarchy[adv_cls]
            #            if adv_cls in self.mat_labels_hierarchy.keys()
            #            else None,
            #        )
            #
            #    total_loss += self.adv_class_scale * loss_adv_cls
            #    losses.update({"adv_cls": loss_adv_cls})
        # TASK 2ter. cell KO effect prediction
        # (just use a novel class, cell state and predict if cell death or not from it)
        # add large timepoint and set the KO gene to a KO embedding instead of expression embedding
        # TODO: try to require the gene id to still be predictable (with weight tying)
        if "mvc_zero_logits" in output:
            loss_expr_mvc = loss.zinb(
                theta=output["mvc_disp"],
                pi=output["mvc_zero_logits"],
                mu=output["mvc_mean"],
                target=expression,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        elif "mvc_mean" in output:
            loss_expr_mvc = loss.mse(
                input=output["mvc_mean"],
                target=expression,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        # TASK 5. elastic cell similarity
        if do_ecs and "cell_emb" in output:
            loss_ecs = loss.ecs(output["cell_emb"], ecs_threshold=self.ecs_threshold)
            total_loss += self.ecs_scale * loss_ecs
            losses.update({"ecs": loss_ecs})

        # Add VAE KL loss if present
        if do_vae_kl and "vae_kl_loss" in output:
            vae_kl_loss = output["vae_kl_loss"]
            # Calculate current VAE KL scale based on global step
            if self.trainer.global_step < self.vae_kl_warmup_steps:
                current_vae_kl_scale = (
                    self.vae_kl_scale
                    * float(self.trainer.global_step + 1)
                    / self.vae_kl_warmup_steps
                )
            else:
                current_vae_kl_scale = self.vae_kl_scale

            total_loss += current_vae_kl_scale * vae_kl_loss
            losses.update({"vae_kl": vae_kl_loss, "vae_kl_scale": current_vae_kl_scale})

        return losses, total_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """@see pl.LightningModule"""
        # update params
        # manually warm up lr without a scheduler
        # making sure that we don't do this during lrfinder
        lr_scale = None
        prev_lr = None
        if (
            self.trainer.global_step < self.warmup_duration + self.lrfinder_steps
        ) and self.lrfinder_steps <= self.trainer.global_step:
            for i, pg in enumerate(optimizer.param_groups):
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.warmup_duration
                )
                prev_lr = pg["lr"]
                pg["lr"] = lr_scale * self.hparams.lr
        for i, pg in enumerate(optimizer.param_groups):
            # if pg["lr"] < 2e-5:
            #    pg["lr"] = 2e-5
            self.log("lr_" + str(i), pg["lr"])
        if optimizer.param_groups[0]["lr"] > self.hparams.lr:
            if prev_lr is not None:
                pg["lr"] = prev_lr
            else:
                raise ValueError("OPTIMIZER HAS INCREASED LR. WHYY?")

        optimizer.step(closure=optimizer_closure)

    def on_validation_start(self):
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            try:
                self.name = self.trainer._loggers[0].version
            except:
                print("not on wandb, could not set name")
        self.embs = None
        self.counter = 0

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        """
        validation_step defines the validation loop. It is independent of forward
        @see pl.LightningModule

        Args:
            batch (list[Tensor]): @see training_step
        """
        val_loss, losses = self._full_training(
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=False,
            cce_temp=self.cce_temp,
            do_ecs=False,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_vae_kl=False,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        metacell_token = batch.get("is_meta", None)
        knn_cells = batch.get("knn_cells", None)

        # TODO: make this faster by only calling val loss
        if self.embs is not None:
            if self.embs.shape[0] < 100_000 / self.trainer.world_size:
                self.info = torch.cat([self.info, batch["class"]])
                self._predict(
                    gene_pos,
                    expression,
                    depth,
                    knn_cells=knn_cells,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=120_000,
                    metacell_token=metacell_token,
                )
        else:
            self.info = batch["class"]
            self._predict(
                gene_pos,
                expression,
                depth,
                knn_cells=knn_cells,
                pred_embedding=self.pred_embedding,
                max_size_in_mem=120_000,
                metacell_token=metacell_token,
            )
        self.log("val_loss", val_loss, sync_dist=True)
        expr_loss = mean(
            [
                v.cpu().item() if type(v) == Tensor else v
                for k, v in losses.items()
                if "expr" in k
            ]
        )
        self.log("val_loss_expr", expr_loss, sync_dist=True)
        cls_loss = mean(
            [
                v.cpu().item() if type(v) == Tensor else v
                for k, v in losses.items()
                if "cls" in k
            ]
        )
        self.log("val_loss_cls", cls_loss, sync_dist=True)
        # self.log_dict(losses, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        """@see pl.LightningModule"""
        self.pos = None
        self.expr_pred = None
        self.embs = self.all_gather(self.embs).view(-1, self.embs.shape[-1])
        self.info = self.all_gather(self.info).view(-1, self.info.shape[-1])
        self.pred = (
            self.all_gather(self.pred).view(-1, self.pred.shape[-1])
            if self.pred is not None
            else None
        )
        # self.pos = self.all_gather(self.pos).view(-1, self.pos.shape[-1])
        # self.expr_pred[0] = self.all_gather(self.expr_pred[0]).view(
        #     -1, self.expr_pred[0].shape[-1]
        # )
        # if len(self.expr_pred) > 1:
        #     self.expr_pred[1] = self.all_gather(self.expr_pred[1]).view(
        #         -1, self.expr_pred[1].shape[-1]
        #     )
        # self.expr_pred[2] = self.all_gather(self.expr_pred[2]).view(
        #     -1, self.expr_pred[2].shape[-1]
        # )

        if self.trainer.state.stage != "sanity_check":
            if self.trainer.is_global_zero:
                print("logging anndata")
                sch = self.lr_schedulers()
                if sch is not None:
                    sch.step(self.trainer.callback_metrics["val_loss"])
                # run the test function on specific dataset
                if self.embs is not None:
                    self.log_adata(
                        gtclass=self.info, name="validation_part_" + str(self.counter)
                    )
                if (self.current_epoch + 1) % self.test_every == 0:
                    self.on_test_epoch_end()
                # Synchronize all processes with a timeout
            if torch.distributed.is_initialized():
                # Set a timeout that's longer than your test typically takes
                # Write rank to file for debugging
                self.trainer.strategy.barrier()

    def test_step(self, *args, **kwargs):
        pass

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            try:
                self.name = self.trainer._loggers[0].version
            except:
                print("not on wandb, could not set name")
        # Run the test only on global rank 0
        name = str(self.name) + "_step" + str(self.global_step) + "_test_metrics"
        import json

        try:
            metrics, tot = utils.test(
                self,
                filedir=str(FILEDIR),
                do_class=self.do_cls,
            )
            print(metrics)
            print("done test")
            f = open("metrics_" + name + ".json", "a")
            f.write(
                json.dumps(
                    tot,
                    indent=4,
                    default=lambda x: int(x) if isinstance(x, np.int64) else x,
                )
            )
            f.close()
            if self.set_step is not None:
                print("this part only works in some cases and for wandb")
                self.trainer._loggers[0].log_metrics(metrics, self.set_step)
            else:
                self.log_dict(metrics, sync_dist=False, rank_zero_only=True)
        except Exception as e:
            import traceback

            print(f"Error during test: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            print("Skipping test metrics logging")

    def on_predict_epoch_start(self):
        """@see pl.LightningModule"""
        if self.trainer.is_global_zero:
            try:
                self.name = self.trainer._loggers[0].version
            except:
                print("not on wandb, could not set name")
        self.embs = None
        self.attn.data = None
        self.attn.attn = None
        self.counter = 0
        if type(self.transformer) is FlashTransformer:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)

    def predict_step(self, batch, batch_idx):
        """
        embed given gene expression, encode the gene embedding and cell embedding.

        Args:
            batch @see training_step

        Returns:
            Tensor: _description_
        """
        return self._predict(
            batch["genes"],
            batch["x"],
            batch["depth"],
            batch.get("knn_cells", None),
            self.predict_mode,
            self.pred_embedding,
            self.get_attention_layer,
            self.predict_depth_mult,
        )

    def _predict(
        self,
        gene_pos,
        expression,
        depth,
        knn_cells=None,
        predict_mode="none",
        pred_embedding=[],
        get_attention_layer=[],
        depth_mult=6,
        keep_output=True,
        max_size_in_mem=100_000,
        get_gene_emb=False,
        metacell_token=None,
    ):
        """
        @see predict_step will save output of predict in multiple self variables

        - embs: the cell embeddings (means from label specific embeddings given by self.pred_embedding)
        - pred: the predicted cell classes
        - pos: the genes used
        - expr_pred: the expression prediction. [mean, disp, zero_logits]
        - mean_attn: the mean attention across cells for the given layer (in self.get_attention_layer)

        these will be finalized in self.on_predict_epoch_end()

        Args:
            @see training_step
            other important arguments:
            keep_output (bool, optional): whether to keep the output in memory. Defaults to True.
            self.get_attention_layer (list, optional): the layers to get the attention from. Defaults to [].
            self.pred_embedding (list, optional): the classes to predict. Defaults to [].

        """
        if self.mask_zeros and knn_cells is None:
            keep = expression.sum(0) != 0
            if keep.sum() != keep.shape[0]:
                expression = expression[:, keep]
                gene_pos = gene_pos[:, keep]
        if self.transformer.attn_type == "hyper":
            # seq len must be a multiple of 128
            num = (
                (1 if self.use_metacell_token else 0)
                + (1 if self.depth_atinput else 0)
                + (len(self.classes) + 1 if not self.cell_transformer else 0)
            )
            if (expression.shape[1] + num) % 128 != 0:
                expression = expression[:, : ((expression.shape[1]) // 128 * 128) - num]
                gene_pos = gene_pos[:, : ((gene_pos.shape[1]) // 128 * 128) - num]
                if knn_cells is not None:
                    knn_cells = knn_cells[
                        :, :, : ((knn_cells.shape[2]) // 128 * 128) - num
                    ]
        if predict_mode == "none":
            output = self.forward(
                gene_pos,
                expression,
                depth_mult=expression.sum(1),
                neighbors=knn_cells,
                req_depth=depth,
                get_attention_layer=get_attention_layer,
                do_class=True,
                get_gene_emb=get_gene_emb,
                metacell_token=metacell_token,
            )
            if len(get_attention_layer) > 0:
                # only first 2 (QK)
                self.attn.add(
                    [i[:, :, :2, :] for i in output[1]],
                    gene_pos,
                    expression if self.mask_zeros else None,
                )
                output = output[0]
            cell_embs = output["cell_embs"]
        elif predict_mode == "denoise":
            output = self.forward(
                gene_pos,
                expression,
                depth_mult=expression.sum(1) * depth_mult,
                neighbors=knn_cells,
                req_depth=depth * depth_mult,
                get_attention_layer=get_attention_layer,
                do_class=True,
                get_gene_emb=get_gene_emb,
                metacell_token=metacell_token,
            )
            if len(get_attention_layer) > 0:
                # only first 2 (QK)
                self.attn.add(
                    [i[:, :, :2, :] for i in output[1]],
                    gene_pos,
                    expression if self.mask_zeros else None,
                )
                output = output[0]
            cell_embs = output["cell_embs"]
        elif predict_mode == "generate":
            output = self.forward(
                gene_pos,
                expression,
                neighbors=knn_cells,
                req_depth=depth,
                do_mvc=False,
                do_class=False,
                get_gene_emb=get_gene_emb,
                metacell_token=metacell_token,
            )
            cell_embs = output["cell_embs"]
            output = self._generate(
                output["cell_embs"],
                gene_pos,
                req_depth=None,  # otherwise we have 2 depths passed
                depth_mult=expression.sum(1),
                do_class=self.do_cls,
                do_mvc=False,
            )
        else:
            raise ValueError(
                "predict_mode needs to be one of ['none', 'denoise', 'generate']"
            )

        if len(pred_embedding) == 0:
            pred_embedding = self.classes
        ind = [self.classes.index(i) + 1 for i in pred_embedding]
        if not keep_output:
            return {
                "embs": torch.mean(cell_embs[:, ind, :], dim=1),
                "class": (
                    torch.stack(
                        [
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            for clsname in self.classes
                        ]
                    ).transpose(0, 1)
                    if len(self.classes) > 0
                    else None
                ),
                "pos": gene_pos,
                "expr": (
                    [output["mean"], output["disp"], output["zero_logits"]]
                    if "disp" in output
                    else [output["mean"]]
                ),
            }
        if self.embs is None:
            self.embs = torch.mean(cell_embs[:, ind, :], dim=1)
            # self.embs = output["cls_output_" + "cell_type_ontology_term_id"]
            self.pred = (
                torch.stack(
                    [
                        (
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            if not self.keep_all_labels_pred
                            else output["cls_output_" + clsname]
                        )
                        for clsname in self.classes
                    ]
                ).transpose(0, 1)
                if len(self.classes) > 0
                else None
            )
            self.pos = gene_pos
            self.expr_pred = (
                [output["mean"], output["disp"], output["zero_logits"]]
                if "disp" in output
                else [output["mean"]]
            )
        else:
            self.embs = torch.cat(
                # [self.embs, output["cls_output_" + "cell_type_ontology_term_id"]]
                [self.embs, torch.mean(cell_embs[:, ind, :], dim=1)]
            )
            self.pred = torch.cat(
                [
                    self.pred,
                    (
                        torch.stack(
                            [
                                (
                                    torch.argmax(output["cls_output_" + clsname], dim=1)
                                    if not self.keep_all_labels_pred
                                    else output["cls_output_" + clsname]
                                )
                                for clsname in self.classes
                            ]
                        ).transpose(0, 1)
                        if len(self.classes) > 0
                        else None
                    ),
                ],
            )
            self.pos = torch.cat([self.pos, gene_pos])
            self.expr_pred = (
                [
                    torch.cat([self.expr_pred[0], output["mean"]]),
                    torch.cat([self.expr_pred[1], output["disp"]]),
                    torch.cat([self.expr_pred[2], output["zero_logits"]]),
                ]
                if "disp" in output
                else [torch.cat([self.expr_pred[0], output["mean"]])]
            )
        if self.embs is not None:
            if self.embs.shape[0] > max_size_in_mem:
                if self.pred_log_adata:
                    print("logging")
                    self.log_adata(name="predict_part_" + str(self.counter))
                    self.counter += 1
                else:
                    print(
                        "WARNING, reached max size in memory, deleting the adata, \
                        need to set pred_log_adata to True to log the adata"
                    )
                self.pos = None
                self.expr_pred = None
                self.pred = None
                self.embs = None

    def on_predict_epoch_end(self):
        """@see pl.LightningModule will"""
        if self.pos.shape[0] < 100:
            return
        if self.pred_log_adata:
            print("adding on disk")
            return self.log_adata(name="predict_part_" + str(self.counter))

    def log_adata(self, gtclass=None, name=""):
        """
        log_adata will log an adata from predictions.
        It will log to tensorboard and wandb if available

        see @utils.log_adata
        """
        try:
            mdir = self.logger.save_dir if self.logger.save_dir is not None else "/tmp"
        except:
            mdir = "data/"
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        adata, fig = utils.make_adata(
            genes=self.genes,
            embs=self.embs,
            pos=self.pos,
            expr_pred=self.expr_pred,
            classes=self.classes,
            pred=self.pred,
            attention=self.attn.get(),
            label_decoders=self.label_decoders,
            labels_hierarchy=self.labels_hierarchy,
            gtclass=gtclass,
            doplot=self.doplot,
        )
        adata.write(
            str(mdir)
            + "/step_"
            + str(self.global_step)
            + "_"
            + str(self.name)
            + "_"
            + str(name)
            + "_"
            + str(self.global_rank)
            + ".h5ad"
        )
        if self.doplot:
            try:
                self.logger.experiment.add_figure(fig)
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig])
            except:
                print("couldn't log to wandb")

        return adata
