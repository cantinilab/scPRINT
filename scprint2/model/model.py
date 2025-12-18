import os
from functools import partial

# from galore_torch import GaLoreAdamW
from typing import Dict, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.tuner.lr_finder import _LRCallback
from numpy import mean
from performer_pytorch import Performer
from scdataloader.utils import load_genes
from scipy.sparse import load_npz
from simpler_flash import FlashTransformer
from torch import Tensor, nn, optim

from . import decoders, encoders, fsq, loss, utils
from .utils import WeightedMasker, simple_masker

FILEDIR = os.path.dirname(os.path.realpath(__file__))


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


class scPRINT2(L.LightningModule, PyTorchModelHubMixin):

    def __init__(
        self,
        genes,
        organisms: list[str],
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 8,
        precpt_gene_emb: Optional[str] = None,
        memmap_gene_emb: bool = False,
        finetune_gene_emb: bool = False,
        freeze_embeddings: bool = True,
        gene_pos_file: Optional[str] = None,
        normalization: str = "sum",  # log, sum, both, raw
        attn_bias: Optional[str] = None,
        expr_encoder_layers: int = 3,
        attention: str = "normal",  # "performer", "legacy-flash", "normal", "criss-cross", "hyper", "adasplash", "softpick", "softpick-flash"
        expr_emb_style: str = "continuous",  # "binned", "continuous", "metacell"
        n_input_bins: int = 0,
        mvc_decoder: Optional[
            str
        ] = None,  # "inner product", "concat query", "sum query"
        pred_embedding: Optional[list[str]] = None,
        layers_cls: list[int] = [256, 128],
        classes: Optional[Dict[str, int]] = None,
        labels_hierarchy: Dict[str, Dict[int, list[int]]] = {},
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        compress_class_dim: Optional[Dict[str, int]] = None,
        cell_specific_blocks: bool = False,
        zinb: bool = True,
        splicing_head: bool = False,
        do_adv_cls: bool = False,
        dropout: float = 0.1,
        use_metacell_token: bool = False,
        lr: float = 0.0001,
        nb_features: Optional[int] = None,
        sketcher_size: int = 200,
        feature_redraw_interval: Optional[int] = None,
        num_heads_kv: int = 4,
        d_model_cell: int = 128,
        nhead_cell: int = 4,
        nlayers_cell: int = 6,
        num_heads_kv_cell: int = 4,
        transformer=None,
        drop_path_rate: float = 0.0,
        # unused args from older versions kept for loading old models
        gene_pos_enc=None,
        max_cont_len=None,
        residual_in_fp32=None,
        checkpointing=None,
        fused_dropout_add_ln=None,
        strict_loading=None,
        optim=None,
        weight_decay=None,
        prenorm=None,
        domain_spec_batchnorm=None,
        use_flash_attn=None,
        cell_emb_style=None,
        num_batch_labels=None,
        fused_mlp=None,
        fused_bias_fc=None,
        **attention_kwargs: dict,
    ):
        """
        scPRINT-2: Single-Cell Pretrained Regulatory Inference Network Transformer.

        A foundation model for single-cell biology that learns cell and gene representations
        through self-supervised learning on large-scale single-cell RNA-seq data. The model
        can be used for:
        - Cell type classification and annotation
        - Gene expression denoising and imputation
        - Cell embedding generation for downstream analysis
        - Gene regulatory network inference via attention patterns
        - Multi-species gene expression modeling

        Architecture Overview:
            1. Gene Encoder: Embeds gene identities (optionally with pretrained embeddings)
            2. Expression Encoder: Encodes expression values (continuous, binned, or metacell)
            3. Position Encoder: Optional genomic position encoding
            4. Transformer: Main attention-based encoder (various attention mechanisms)
            5. Cell Transformer: Optional separate transformer for cell embeddings
            6. Decoders: Expression reconstruction, classification, and MVC decoders

        The model supports multiple training objectives:
            - Masked expression prediction (like BERT)
            - Denoising autoencoding
            - Cell embedding contrastive learning (ECS and CCE losses)
            - Multi-class cell type classification with hierarchical labels
            - Multi-view coding (MVC) for robust representations

        Args:
            genes (list | dict): Gene vocabulary. Either a list of gene names or a dict
                mapping organism names to lists of genes for multi-species models.
            organisms (list[str]): List of organism ontology term IDs the model supports.
            d_model (int, optional): Hidden dimension of the transformer. Defaults to 256.
            nhead (int, optional): Number of attention heads. Defaults to 4.
            nlayers (int, optional): Number of transformer layers. Defaults to 8.
            precpt_gene_emb (str, optional): Path to parquet file with pretrained gene
                embeddings. Index should match gene names. Defaults to None.
            memmap_gene_emb (bool, optional): Memory-map gene embeddings for large files.
                Defaults to False.
            finetune_gene_emb (bool, optional): Add trainable adapter layers on top of
                frozen pretrained embeddings. Defaults to False.
            freeze_embeddings (bool, optional): Freeze gene embeddings during training.
                Defaults to True.
            gene_pos_file (str, optional): Path to parquet file with genomic positions.
                Must have 'pos' column with integer positions. Defaults to None.
            normalization (str, optional): Expression normalization method. One of:
                - "sum": Divide by total counts (TPM-like)
                - "log": Log2(1 + x) transform
                - "both": Sum normalization then log transform
                - "raw": No normalization
                Defaults to "sum".
            attn_bias (str, optional): Path to sparse matrix (.npz) with attention biases
                (e.g., gene-gene regulatory priors). Defaults to None.
            expr_encoder_layers (int, optional): Number of layers in expression encoder MLP.
                Defaults to 3.
            attention (str, optional): Attention mechanism type. One of:
                - "normal": Standard PyTorch attention
                - "legacy-flash": Flash attention via simpler-flash
                - "performer": Performer linear attention
                - "hyper": Compressed hyperbolic attention
                - "criss-cross": Criss-cross attention
                Defaults to "normal".
            expr_emb_style (str, optional): Expression embedding approach. One of:
                - "continuous": MLP on continuous expression values
                - "binned": Learned embeddings for discretized expression bins
                - "metacell": DeepSet encoder aggregating KNN neighbors
                Defaults to "continuous".
            n_input_bins (int, optional): Number of expression bins when using binned
                embedding. Required if expr_emb_style="binned". Defaults to 0.
            mvc_decoder (str, optional): Multi-view coding decoder architecture. One of:
                - None: No MVC decoder
                - "inner product": Dot product between cell and gene embeddings
                - "concat query": Concatenate cell embedding with gene queries
                - "sum query": Add cell embedding to gene queries
                Defaults to None.
            pred_embedding (list[str], optional): Class names to use for cell embeddings
                during prediction/logging. Defaults to None (use all).
            layers_cls (list[int], optional): Hidden layer sizes for classification heads.
                Defaults to [256, 128].
            classes (dict[str, int], optional): Classification targets mapping class names
                to number of categories. E.g., {"cell_type_ontology_term_id": 100}.
                Defaults to None.
            labels_hierarchy (dict[str, dict[int, list[int]]], optional): Hierarchical
                label structure for ontology-based classes. Maps parent indices to lists
                of children indices. Defaults to {}.
            label_decoders (dict[str, dict[int, str]], optional): Mapping from encoded
                integers back to label strings for each class. Used for logging/plotting.
                Defaults to None.
            compress_class_dim (dict[str, int], optional): Compressed embedding dimension
                for each class. Uses VAE or FSQ compression. Defaults to None.
            cell_specific_blocks (bool, optional): Use separate transformer for cell
                embeddings with cross-attention to gene transformer. Defaults to False.
            zinb (bool, optional): Use Zero-Inflated Negative Binomial distribution for
                expression reconstruction. If False, uses MSE loss. Defaults to True.
            splicing_head (bool, optional): Add separate decoder for spliced/unspliced
                expression. Defaults to False.
            do_adv_cls (bool, optional): Use adversarial classification to remove batch
                effects from cell type embeddings. Defaults to False.
            dropout (float, optional): Dropout rate throughout the model. Defaults to 0.1.
            use_metacell_token (bool, optional): Add learnable metacell token to distinguish
                single cells from metacells. Defaults to False.
            lr (float, optional): Base learning rate. Defaults to 0.0001.
            nb_features (int, optional): Number of random features for Performer attention.
                Defaults to None.
            sketcher_size (int, optional): Sketch size for sparse attention methods.
                Defaults to 200.
            feature_redraw_interval (int, optional): Steps between random feature redraws
                for Performer. Defaults to None.
            num_heads_kv (int, optional): Number of key-value heads (for MQA/GQA).
                Defaults to 4.
            d_model_cell (int, optional): Hidden dim for cell transformer when using
                cell_specific_blocks. Defaults to 128.
            nhead_cell (int, optional): Attention heads for cell transformer. Defaults to 4.
            nlayers_cell (int, optional): Layers in cell transformer. Defaults to 6.
            num_heads_kv_cell (int, optional): KV heads for cell transformer. Defaults to 4.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
            **attention_kwargs (dict): Additional arguments passed to FlashTransformer.

        Attributes:
            Training Configuration (set these before training):
                noise (list[float]): Dropout rates for denoising task. E.g., [0.6].
                mask_ratio (list[float]): Mask ratios for masked prediction. E.g., [0.15].
                cce_temp (float): Temperature for contrastive loss.
                cce_scale (float): Weight for contrastive cell embedding loss.
                ecs_scale (float): Weight for elastic cell similarity loss.
                ecs_threshold (float): Similarity threshold for ECS loss.
                mvc_scale (float): Weight for MVC reconstruction loss.
                class_scale (float): Weight for classification loss.
                lr_reduce_patience (int): Epochs before reducing learning rate.
                lr_reduce_factor (float): Factor to reduce learning rate by.
                warmup_duration (int): Steps for learning rate warmup.

            Prediction Configuration (set before predict):
                predict_mode (str): "none" or "generate" for expression generation.
                pred_embedding (list[str]): Classes to include in cell embeddings.
                get_attention_layer (list[int]): Layers to extract attention from.
                predict_depth_mult (float): Multiplier for depth in generation.
                pred_log_adata (bool): Whether to log predictions as AnnData.

        Example:
            >>> # Initialize model
            >>> model = scPrint2(
            ...     genes=gene_list,
            ...     organisms=["NCBITaxon:9606"],
            ...     d_model=512,
            ...     nlayers=12,
            ...     classes={"cell_type_ontology_term_id": 100},
            ... )
            >>>
            >>> # Configure training
            >>> model.noise = [0.4, 0.6]
            >>> model.mask_ratio = [0.15, 0.3]
            >>>
            >>> # Train with PyTorch Lightning
            >>> trainer = L.Trainer(max_epochs=100)
            >>> trainer.fit(model, datamodule)
            >>>
            >>> # Generate embeddings
            >>> model.pred_embedding = ["cell_type_ontology_term_id"]
            >>> predictions = trainer.predict(model, datamodule)

        Note:
            The model is designed to work with scDataLoader's DataModule and Collator.
            Gene order must match between model initialization and data loading.
        """
        super().__init__()
        self.save_hyperparameters()
        # training flags
        self.noise = [0.6]
        self.cce_temp = 0.3
        self.lr = lr
        self.cce_scale = 0.2
        self.ecs_threshold = 0.4
        self.ecs_scale = 0.2
        self.mvc_scale = 1.0
        self.class_embd_diss_scale = 0.3
        self.adv_class_scale = 1.0
        self.do_adv_cls = do_adv_cls
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
        self.doplot = False
        self.get_attention_layer = None
        self.embs = None
        self.pred_log_adata = True
        self.predict_depth_mult = 3
        self.predict_mode = "none"
        self.keep_all_labels_pred = False
        self.mask_zeros = False
        self.vae_kl_scale = 0.05
        self.vae_kl_warmup_steps = 40_000  # Default value, can be adjusted
        self.save_expr = False
        self.counter = 0

        # should be stored somehow
        self.d_model = d_model
        self.normalization = normalization
        self.attn_bias = attn_bias if attn_bias != "none" else None
        self.organisms = organisms
        self.nlayers = nlayers
        self.use_metacell_token = use_metacell_token
        self.mvc_decoder = mvc_decoder
        # need to store
        self.n_input_bins = n_input_bins
        self.attention = attention

        if classes is None:
            classes = {}
        self.label_counts = classes
        self.classes = list(classes.keys())

        self.label_decoders = label_decoders
        self.pred_embedding = pred_embedding
        self._genes = genes
        self.expr_emb_style = expr_emb_style
        if labels_hierarchy is None:
            labels_hierarchy = {}
        self.labels_hierarchy = labels_hierarchy
        self.hparams["classes"] = classes
        self.hparams["label_decoders"] = label_decoders
        self.hparams["organisms"] = organisms
        self.hparams["use_metacell_token"] = use_metacell_token
        # 20x more likely to drop a non TF compared to a TF
        self.tf_masker = WeightedMasker(self.genes, tf_weight=0.05)
        self.attn = utils.Attention(
            len(self.genes),
            additional_tokens=(
                (1 if self.use_metacell_token else 0)
                + ((len(classes) + 1) if not cell_specific_blocks else 0)
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
        if gene_pos_file is not None:
            gene_pos_enc = pd.read_parquet(gene_pos_file)
            if len(gene_pos_enc) < len(self.genes):
                print("Warning: only a subset of the genes available in the loc file.")
            for k, v in self._genes.items():
                tokeep = set(gene_pos_enc.index.tolist())
                self._genes[k] = [u for u in v if u in tokeep]
                if len(self._genes[k]) < 100:
                    raise ValueError(
                        f"the gene pos file {gene_pos_file} does not match most of the genes given to the model for species {k}"
                    )
            gene_pos_enc = gene_pos_enc.loc[self.genes, ["pos"]]

        if precpt_gene_emb is not None:
            embeddings = pd.read_parquet(precpt_gene_emb)
            if len(embeddings) < len(self.genes):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
            for k, v in self._genes.items():
                tokeep = set(embeddings.index.tolist())
                self._genes[k] = [u for u in v if u in tokeep]
                if len(self._genes[k]) < 100:
                    raise ValueError(
                        f"the gene embeddings file {precpt_gene_emb} does not match most of the genes given to the model for species {k}"
                    )
            embeddings = embeddings.loc[self.genes]
            print("number of genes: ", len(embeddings))
            if not memmap_gene_emb:
                sembeddings = torch.nn.AdaptiveAvgPool1d(d_model)(
                    torch.tensor(embeddings.values, dtype=torch.float32)
                )
            else:
                embeddings = None
            gene_encoder = encoders.GeneEncoder(
                len(self.genes),
                d_model,
                weights_file=precpt_gene_emb if memmap_gene_emb else None,
                weights=sembeddings if not memmap_gene_emb else None,
                freeze=freeze_embeddings,
            )
        else:
            gene_encoder = encoders.GeneEncoder(
                len(self.genes), d_model, freeze=freeze_embeddings
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
        # Positional Encoding
        if gene_pos_file is not None:
            # redoing it just in case some were dropped with embbeding file step
            gene_pos_enc = gene_pos_enc.loc[self.genes, "pos"].astype(int).tolist()
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, gene_pos_enc=gene_pos_enc
            )
        else:
            self.pos_encoder = None
        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        expr_d_model = d_model  # // 8 if finetune_gene_emb else d_model
        if expr_emb_style in "continuous":
            expr_encoder = encoders.ContinuousValueEncoder(
                expr_d_model, dropout, layers=expr_encoder_layers
            )
        elif expr_emb_style == "binned":
            assert n_input_bins > 0
            assert normalization == "raw", "shouldn't use normalization"
            expr_encoder = encoders.CategoryValueEncoder(n_input_bins, expr_d_model)
        elif expr_emb_style == "metacell":
            expr_encoder = encoders.EasyExprGNN(
                self_dim=expr_d_model * 2,
                output_dim=expr_d_model,
                shared_layers=expr_encoder_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"expr_emb_style should be one of binned, continuous, metacell, "
                f"got {expr_emb_style}"
            )
        if finetune_gene_emb and False:
            self.expr_encoder = encoders.ExprBasedFT(
                d_model,
                gene_encoder,
                expr_encoder,
                dropout,
                layers=expr_encoder_layers,
                intermediary_d=int(d_model * 1.5),
            )
        else:
            self.expr_encoder = expr_encoder

        # Class Encoder
        # always have [base_cell_emb, time_embedding, depth_embedding] + any other class info
        # base cell embedding will store other cell specific information
        self.class_encoder = encoders.CategoryValueEncoder(
            len(self.classes) + 1,
            d_model if not cell_specific_blocks else d_model_cell,
        )

        if self.use_metacell_token:
            self.metacell_encoder = encoders.CategoryValueEncoder(2, d_model)
        # compute tensor for mat_labels_hierarchy
        # old parameters that can still be passed when loading older models (managed in the _on_load_ckpt function)
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
            "transformer",
            "residual_in_fp32",
            "max_cont_len",
        ]:
            if i in attention_kwargs:
                attention_kwargs.pop(i)
        # attention
        # Linear
        if attention == "linear":
            # linear attention using the fast attention package
            # self.attention = FastattentionEncoder(
            #    d_model, nhead, d_hid, nlayers, dropout, "linear"
            # )
            raise NotImplementedError("Linear attention is not implemented")
        elif attention == "performer":
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
            )
        else:
            self.transformer = FlashTransformer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                attn_dropout=dropout,
                nlayers=nlayers,
                cross_attn=cell_specific_blocks,
                cross_dim=d_model_cell,
                attn_type="flash" if attention == "legacy-flash" else attention,
                num_heads_kv=num_heads_kv,
                sketcher_size=sketcher_size,
                drop_path_rate=drop_path_rate,
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
                attn_type="flash" if attention == "legacy-flash" else "normal",
                **attention_kwargs,
            )
        else:
            self.cell_transformer = None

        # decoders
        # expression
        self.splicing_head = None
        if expr_emb_style == "binned":
            self.expr_decoder = decoders.ClsDecoder(
                d_model,
                n_input_bins,
                layers=[d_model // 2, d_model // 4],
                dropout=dropout,
            )
        else:
            self.expr_decoder = decoders.ExprDecoder(
                d_model,
                dropout=dropout,
                zinb=zinb,
                use_depth=True,
            )
            if splicing_head:
                self.splicing_head = decoders.ExprDecoder(
                    d_model,
                    dropout=dropout,
                    zinb=zinb,
                    use_depth=True,
                )
        # cls decoder
        self.cls_decoders = torch.nn.ModuleDict()
        # should be a very simple classifier for most things
        # (maybe scale with the number of classes) should be 1 layer...
        for clss, n_cls in classes.items():
            mdim = d_model_cell if cell_specific_blocks else self.d_model
            dim = compress_class_dim[clss] if compress_class_dim is not None else mdim
            self.cls_decoders[clss] = decoders.ClsDecoder(
                dim,
                n_cls,
                layers=layers_cls,
                dropout=dropout,
            )
        if "cell_type_ontology_term_id" in classes and self.do_adv_cls:
            mdim = d_model_cell if cell_specific_blocks else self.d_model
            dim = (
                compress_class_dim["cell_type_ontology_term_id"]
                if compress_class_dim is not None
                else mdim
            )
            if "assay_ontology_term_id" in classes:
                self.assay_relab = utils.relabel_assay_for_adv(
                    self.label_decoders, self.labels_hierarchy
                )
                self.adv_assay_decoder = decoders.ClsDecoder(
                    dim,
                    len(set(self.assay_relab.values())),
                    layers=layers_cls,
                    dropout=dropout,
                )
            if len(self.organisms) > 1:
                self.adv_organism_decoder = decoders.ClsDecoder(
                    dim,
                    len(self.organisms),
                    layers=layers_cls,
                    dropout=dropout,
                )
        # expression decoder from batch embbedding
        if mvc_decoder is not None:
            if cell_specific_blocks:
                raise ValueError(
                    "MVC decoder is not supported for cell specific blocks"
                )
            self.mvc_decoder = decoders.MVCDecoder(
                d_model, arch_style=mvc_decoder, zinb=zinb, use_depth=True
            )
        else:
            self.mvc_decoder = None

        if compress_class_dim is not None:
            self.compressor = torch.nn.ModuleDict()
            dim = d_model_cell if cell_specific_blocks else self.d_model
            for k, v in compress_class_dim.items():
                if v >= 8:
                    self.compressor[k] = decoders.VAEDecoder(
                        dim,
                        layers=[
                            128,
                            v,
                        ],
                        dropout=dropout,
                        return_latent=True,
                    )
                else:
                    self.compressor[k] = fsq.FSQ(levels=[2] * v, dim=dim)
        else:
            self.compressor = None

        self.apply(
            partial(
                utils._init_weights,
                n_layer=nlayers,
            )
        )
        for i, dec in self.cls_decoders.items():
            torch.nn.init.constant_(dec.out_layer.bias, -0.13)
        self.expr_encoder._init_weights()

        # self.hparams.drop(gene_pos_file)
        # self.hparams.drop(precpt_gene_emb)

    def add_organism(
        self, organism: str, genes: pd.Index, emb: pd.DataFrame, locs=None
    ):
        """
        Add a new organism to an existing model for transfer learning.

        Extends the gene vocabulary and embeddings to include genes from a new
        organism. Useful for applying a pretrained model to a new species.

        Args:
            organism (str): Organism ontology term ID (e.g., "NCBITaxon:10090" for mouse).
            genes (pd.Index): Gene names/IDs for the new organism.
            emb (pd.DataFrame): Gene embeddings DataFrame with genes as index.
                Will be resized to match model's d_model.
            locs (pd.DataFrame, optional): Genomic positions with 'pos' column.
                Required if model uses positional encoding. Defaults to None.

        Raises:
            ValueError: If model requires gene locations but none provided.
            ValueError: If gene positions exceed model's maximum position encoding.

        Note:
            Only genes present in both `genes` and `emb` (and `locs` if provided)
            will be added. The model's gene encoder is expanded in-place.
        """
        if self.pos_encoder is not None and locs is None:
            raise ValueError("this model needs gene locations to add a new organism")

        self.organisms.append(organism)
        if locs is not None:
            overlap = set(locs.index) & set(emb.index) & set(genes.index)
            genes = genes[genes.index.isin(overlap)]
            locs = locs.loc[genes.index]
            pos = locs["pos"]
            token_to_pos = {token: pos for token, pos in enumerate(pos)}
            if self.pos_encoder.pe.shape[0] < max(pos):
                raise ValueError(
                    f"the number of gene locs in the added organism needs to be less than {self.pos_encoder.pe.shape[0]}"
                )

            token_to_pos = {token: pos for token, pos in enumerate(pos)}
            arr = []
            for _, v in token_to_pos.items():
                arr.append(self.pos_encoder.pe[v - 1].to("cpu").numpy())
            pe = torch.Tensor(np.array(arr)).to(self.pos_encoder.pe.device)
            self.pos_encoder.pe = torch.cat([self.pos_encoder.pe, pe], dim=0)
        else:
            overlap = set(emb.index) & set(genes.index)
            genes = genes[genes.index.isin(overlap)]

        emb = emb.loc[genes.index]
        self._genes[organism] = genes.index.tolist()
        if self.gene_encoder is None:
            genc = self.expr_encoder.gene_encoder
        else:
            genc = self.gene_encoder
        if type(genc) is torch.nn.Sequential:
            enc = genc[0]
        else:
            enc = genc
        semb = torch.nn.AdaptiveAvgPool1d(self.d_model)(
            torch.tensor(emb.values, dtype=torch.float32)
        ).to(enc.embeddings.weight.data.device)
        if enc.memmap:
            print("todev.. will fail for now")

        embs = torch.cat([enc.embeddings.weight.data, semb], dim=0)
        enc.embeddings = nn.Embedding(
            embs.shape[0],
            embs.shape[1],
            padding_idx=None,
            _freeze=enc.embeddings.weight.requires_grad,
        )
        enc.embeddings.weight.data.copy_(embs)
        enc.embeddings.weight.data = enc.embeddings.weight.data.to(self.device)
        if type(genc) is torch.nn.Sequential:
            genc[0] = enc
        else:
            genc = enc
        if self.gene_encoder is None:
            self.expr_encoder.gene_encoder = genc
        else:
            self.gene_encoder = genc

    def on_load_checkpoint(self, checkpoints):
        """
        Handle checkpoint loading with backward compatibility.

        Automatically handles:
        - Different class configurations between checkpoint and current model
        - Legacy parameter names and structures
        - Encoder/decoder mismatches with datamodule
        - Gene vocabulary differences
        - Early stopping callback state

        Called automatically by PyTorch Lightning during checkpoint loading.

        Args:
            checkpoints (dict): Checkpoint dictionary from torch.load().

        Note:
            Prints warnings when configurations differ between checkpoint and
            current model. These should be reviewed to ensure expected behavior.
        """
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

        if checkpoints["state_dict"].get("pos_encoder.pe") is not None:
            if self.pos_encoder is None:
                self.pos_encoder = encoders.PositionalEncoding(
                    self.d_model, gene_pos_enc=[0, 1, 2]
                )
            self.pos_encoder.pe = checkpoints["state_dict"]["pos_encoder.pe"]

        if self.label_decoders != checkpoints["hyper_parameters"][
            "label_decoders"
        ] or self.labels_hierarchy != checkpoints["hyper_parameters"].get(
            "labels_hierarchy", {}
        ):
            print("label decoders have changed, be careful")
            self.label_decoders = checkpoints["hyper_parameters"]["label_decoders"]
            self.labels_hierarchy = checkpoints["hyper_parameters"].get(
                "labels_hierarchy", {}
            )
            for k, v in self.labels_hierarchy.items():
                tens = torch.zeros((len(v), self.label_counts[k]))
                for k2, v2 in v.items():
                    tens[k2 - self.label_counts[k], v2] = 1
                self.mat_labels_hierarchy[k] = tens.to(bool)

        if (
            "gene_pos_enc" in checkpoints["hyper_parameters"]
            and checkpoints["hyper_parameters"]["gene_pos_enc"] is not None
        ):
            if (
                self.pos_encoder is None
                or self.pos_encoder.gene_pos_enc
                != checkpoints["hyper_parameters"]["gene_pos_enc"]
            ):
                print(
                    "Gene position encoding has changed in the dataloader compared to last time, trying to revert"
                )
                self.pos_encoder = encoders.PositionalEncoding(
                    self.d_model,
                    gene_pos_enc=checkpoints["hyper_parameters"]["gene_pos_enc"],
                )
                checkpoints["hyper_parameters"].pop("gene_pos_enc")
        mencoders = {}
        if type(checkpoints["hyper_parameters"]["genes"]) is list:
            print("converting a gene list-based model")
            org = checkpoints["hyper_parameters"].get("organisms", self.organisms)
            genedf = load_genes(org)
            checkpoints["hyper_parameters"]["genes"] = {
                i: genedf.index[
                    (genedf.organism == i)
                    & genedf.index.isin(checkpoints["hyper_parameters"]["genes"])
                ].tolist()
                for i in org
            }
        if "precpt_gene_emb" in checkpoints["hyper_parameters"]:
            checkpoints["hyper_parameters"].pop("precpt_gene_emb")

        if "gene_pos_file" in checkpoints["hyper_parameters"]:
            checkpoints["hyper_parameters"].pop("gene_pos_file")

        if "transformer" in checkpoints["hyper_parameters"]:
            checkpoints["hyper_parameters"]["attention"] = checkpoints[
                "hyper_parameters"
            ].pop("transformer")
        try:
            if self.trainer.datamodule.decoders != self.label_decoders:
                print("label decoders have changed, be careful")
                # if we don't have the same decoders, we need to update the one on the datamodule side
                for k, v in self.label_decoders.items():
                    mencoders[k] = {va: ke for ke, va in v.items()}
                self.trainer.datamodule.encoders = mencoders

            es = None
            for k in self.trainer.callbacks:
                if isinstance(k, EarlyStopping):
                    es = k
            if es is not None:
                prev = checkpoints["callbacks"].get(
                    "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"
                )
                if prev is not None:
                    prev = prev["patience"]
                if prev != es.patience:
                    print(
                        "updating the early stopping parameter to {}".format(
                            es.patience
                        )
                    )
                    checkpoints["callbacks"][
                        "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"
                    ]["patience"] = es.patience
                    if prev < es.patience:
                        checkpoints["callbacks"][
                            "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"
                        ]["stopped_epoch"] = 0

        except RuntimeError as e:
            if "scPRINT2 is not attached to a `Trainer`." in str(e):
                print("FYI: scPRINT2 is not attached to a `Trainer`.")
            else:
                raise e
        if (
            self.mvc_decoder is None
            and checkpoints["state_dict"].get("mvc_decoder.gene2query.weight")
            is not None
        ):
            for i in [
                "mvc_decoder.gene2query.weight",
                "mvc_decoder.gene2query.bias",
                "mvc_decoder.norm.weight",
                "mvc_decoder.norm.bias",
                "mvc_decoder.pred_var_zero.weight",
            ]:
                if i in checkpoints["state_dict"]:
                    del checkpoints["state_dict"][i]
        org = checkpoints["hyper_parameters"].get("organisms")
        if self.organisms != org and org is not None:
            self.organisms = org
            try:
                self.trainer.datamodule.organisms = self.organisms
            except RuntimeError as e:
                if "scPRINT2 is not attached to a `Trainer`." not in str(e):
                    raise e
        if self._genes != checkpoints["hyper_parameters"]["genes"]:
            self._genes = checkpoints["hyper_parameters"]["genes"]
        try:
            self.trainer.datamodule.set_valid_genes_collator(self.genes)
        except RuntimeError as e:
            if "scPRINT2 is not attached to a `Trainer`." not in str(e):
                raise e

        if not is_interactive():
            self.save_hyperparameters()

    def _rm_genes(self, names):
        tokeep = ~np.array([g in names for g in self.genes])
        # Keep only embeddings for genes that are NOT being deleted
        kept_embeddings = self.gene_encoder.embeddings.weight.data[tokeep]

        # Create new embeddings layer with reduced vocabulary size
        new_vocab_size = tokeep.sum()
        new_gene_encoder = encoders.GeneEncoder(new_vocab_size, self.d_model)
        # Copy the kept embeddingss to the new encoder
        new_gene_encoder.embeddings.weight.data = kept_embeddings
        # Replace the old encoder with the new one
        if type(self.expr_encoder) is encoders.ExprBasedFT:
            self.expr_encoder.gene_encoder = new_gene_encoder
        self.gene_encoder = new_gene_encoder
        # Update vocabulary
        for k, v in self._genes.items():
            if len(set(v) & set(names)) > 0:
                self._genes[k] = [g for g in v if g not in names]
        self.attn.gene_dim = len(self.genes)
        if self.pos_encoder is not None:
            # Update gene position encoding
            self.pos_encoder.pe = self.pos_encoder.pe[tokeep]

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        neighbors: Optional[Tensor] = None,
        neighbors_info: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,  # (minibatch, n_labels, embsize)
        metacell_token: Optional[Tensor] = None,  # (minibatch, 1)
    ):
        """
        Encode input data into transformer-ready representations.

        Processes gene positions and expression values through the gene encoder,
        expression encoder, and optional positional encoder to create input
        embeddings for the transformer.

        Args:
            gene_pos (Tensor): Gene indices of shape (batch, seq_len).
            expression (Tensor, optional): Expression values of shape (batch, seq_len).
                If None, only gene embeddings are used. Defaults to None.
            neighbors (Tensor, optional): KNN neighbor expression of shape
                (batch, n_neighbors, seq_len). Used with metacell encoder.
                Defaults to None.
            neighbors_info (Tensor, optional): Neighbor weights/distances of shape
                (batch, n_neighbors). Defaults to None.
            mask (Tensor, optional): Boolean mask of shape (batch, seq_len) where
                True indicates positions to mask during encoding. Defaults to None.
            cell_embs (Tensor, optional): Pre-computed cell embeddings of shape
                (batch, n_classes+1, d_model). If None, initialized from class
                encoder. Defaults to None.
            metacell_token (Tensor, optional): Binary indicators of shape (batch,)
                where 1 indicates metacell samples. Defaults to None.

        Returns:
            tuple[Tensor, Tensor]:
                - cell_embs: Cell embeddings of shape (batch, n_classes+1, d_model)
                - encoding: Gene encodings of shape (batch, seq_len, d_model)
        """
        if expression is not None or neighbors is not None:
            if self.normalization in ["sum", "both"]:
                expression = expression / expression.sum(1).unsqueeze(1)
                if neighbors is not None:
                    neighbors = neighbors / neighbors.sum(2).unsqueeze(1)
            elif self.normalization == "raw":
                pass
            elif self.normalization in ["log", "both"]:
                expression = torch.log2(1 + expression)
                if neighbors is not None:
                    neighbors = torch.log2(1 + neighbors)
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
            if neighbors_info is not None:
                # go from size batch, n_neighbor to batch, len, n_neighbor
                neighbors_info = neighbors_info.unsqueeze(1).repeat(
                    1, gene_pos.shape[1], 1
                )
            if type(self.expr_encoder) is encoders.ExprBasedFT:
                enc = self.expr_encoder(
                    gene_pos=gene_pos,
                    expr=expression,
                    mask=mask,
                    neighbors=neighbors,
                    neighbors_info=neighbors_info,
                )
            elif type(self.expr_encoder) is encoders.EasyExprGNN:
                enc = self.gene_encoder(gene_pos)
                expr_emb = self.expr_encoder(
                    expression,
                    mask=mask,
                    neighbors=neighbors,
                    edge_info=neighbors_info,
                )
                enc.add_(expr_emb)
            else:
                expr_emb = self.expr_encoder(expression, mask=mask)
                enc = self.gene_encoder(gene_pos)
                enc.add_(expr_emb)
        elif type(self.expr_encoder) is encoders.ExprBasedFT:
            # Set all mask values to True (mask everything)
            mask = torch.ones_like(gene_pos, dtype=torch.bool)
            # Set expression to zeros
            expression = torch.zeros_like(gene_pos, dtype=torch.float32)
            enc = self.expr_encoder(
                gene_pos=gene_pos,
                expr=expression,
                mask=mask,
                neighbors=neighbors,
                neighbors_info=neighbors_info,
            )
        else:
            enc = self.gene_encoder(gene_pos)  # (minibatch, seq_len, embsize)
        if self.pos_encoder is not None:
            enc.add_(self.pos_encoder(gene_pos))
        if cell_embs is None:
            cell_embs = self.class_encoder(
                torch.arange(
                    len(self.classes) + 1,
                    device=gene_pos.device,
                ).repeat(gene_pos.shape[0], 1)
            )
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
        return cell_embs, enc
        # we already apply prenorm & dropout  # (minibatch, seq_len, embsize)

    def _expr_decoder(
        self,
        transformer_output,
        depth_mult,
        req_depth,
        get_gene_emb=False,
        splicing_mult: Optional[Tensor] = None,
    ):
        """
        Decode transformer output into expression predictions.

        Args:
            transformer_output (Tensor): Transformer output of shape
                (batch, seq_len, d_model).
            depth_mult (Tensor): Depth multiplier of shape (batch,) used to
                scale predicted expression means.
            req_depth (Tensor): Requested/target depth of shape (batch,) passed
                to the decoder for depth-aware prediction.
            get_gene_emb (bool, optional): If True, include raw transformer output
                as "gene_embedding" in results. Defaults to False.
            splicing_mult (Tensor, optional): Multiplier for splicing predictions
                if splicing_head is enabled. Defaults to None.

        Returns:
            dict[str, Tensor]: Expression predictions containing:
                - "mean": Predicted expression means (batch, seq_len)
                - "disp": Dispersion parameters (batch, seq_len) [if ZINB]
                - "zero_logits": Zero-inflation logits (batch, seq_len) [if ZINB]
                - "gene_embedding": Raw embeddings (batch, seq_len, embsize) [if get_gene_emb]
                - "spl_*": Splicing predictions [if splicing_head]
        """
        if self.expr_emb_style != "binned":
            output = self.expr_decoder(transformer_output, req_depth)
            output["mean"] = depth_mult.unsqueeze(1) * output["mean"]
        else:
            # binned case
            output = {"mean": self.expr_decoder(transformer_output)}
        if self.splicing_head is not None:
            splicing_output = self.splicing_head(transformer_output, req_depth)
            output.update({"spl_" + k: v for k, v in splicing_output.items()})
            output["spl_mean"] = splicing_mult.unsqueeze(1) * output["spl_mean"]

        if get_gene_emb:
            output["gene_embedding"] = (
                transformer_output  # (minibatch, seq_len, embsize)
            )
        return output

    def _cell_decoder(
        self,
        cell_embs,
        do_mvc,
        do_class,
        depth_mult,
        req_depth,
        gene_pos=None,
    ):
        """
        Decode cell embeddings into classifications and MVC predictions.

        Optionally applies compression (VAE or FSQ) to cell embeddings and
        generates class predictions and multi-view coded expression.

        Args:
            cell_embs (Tensor): Cell embeddings of shape (batch, n_classes+1, d_model).
            do_mvc (bool): Whether to compute MVC expression predictions.
            do_class (bool): Whether to compute class predictions.
            depth_mult (Tensor): Depth multiplier of shape (batch,).
            req_depth (Tensor): Log-transformed requested depth of shape (batch,).
            gene_pos (Tensor, optional): Gene positions for MVC decoder.
                Required if do_mvc=True. Defaults to None.

        Returns:
            dict[str, Tensor]: Cell decoder outputs containing:
                - "input_cell_embs": Original cell embeddings (batch, n_classes+1, d_model)
                - "input_cell_emb": Mean pooled embedding (batch, d_model)
                - "output_cell_embs": Compressed/decompressed embeddings
                - "compressed_cell_embs": List of compressed embeddings per class
                - "output_cell_emb": Concatenated compressed embedding
                - "cls_output_{class}": Class logits for each class
                - "mvc_mean", "mvc_disp", "mvc_zero_logits": MVC predictions [if do_mvc]
                - "vae_kl_loss": VAE KL divergence [if using VAE compression]
        """
        output = {}
        output["input_cell_embs"] = cell_embs
        output["input_cell_emb"] = torch.mean(output["input_cell_embs"], dim=1)

        if self.compressor is not None:
            # Apply VAE to cell embeddings
            output["vae_kl_loss"] = 0
            res = []
            zs = []
            if "default" in self.compressor:
                out = self.compressor["default"](cell_embs[:, 0, :])
                res.append(out[0].unsqueeze(1))
                if len(out) == 5:
                    output["vae_kl_loss"] += out[4]
                    zs.append(out[3])
                else:
                    zs.append(out[0])
            else:
                res.append(cell_embs[:, 0, :].unsqueeze(1))
                zs.append(cell_embs[:, 0, :])
            for i, clsname in enumerate(self.classes):
                out = self.compressor[clsname](cell_embs[:, i + 1, :])
                res.append(out[0].unsqueeze(1))
                if len(out) == 5:  # VAE case
                    output["vae_kl_loss"] += out[4]
                    zs.append(out[1])
                else:  # FSQ case
                    zs.append(out[2])
            # shape (minibatch, n_classes + 1, embsize)
            output["output_cell_embs"] = torch.cat(res, dim=1)
            # shape [n_classes + 1](minibatch, compressed_embsizes[i])
            output["compressed_cell_embs"] = zs
            # shape (minibatch, sum(compressed_embsizes))
            output["output_cell_emb"] = torch.cat(zs, dim=1)
        else:
            # shape (minibatch, n_classes + 1, embsize)
            output["output_cell_embs"] = cell_embs
            # shape (minibatch, embsize)
            output["output_cell_emb"] = torch.mean(output["output_cell_embs"], dim=1)
        if len(self.classes) > 0 and do_class:
            for i, clsname in enumerate(self.classes):
                output.update(
                    {
                        "cls_output_"
                        + clsname: self.cls_decoders[clsname](
                            output["compressed_cell_embs"][i + 1]
                            if self.compressor is not None
                            else output["input_cell_embs"][:, i + 1, :]
                        )
                    }
                )
        if do_mvc:
            if self.expr_emb_style == "binned":
                raise ValueError("MVC decoding not supported with binned expression")
            output.update(
                self.mvc_decoder(
                    output["input_cell_emb"],
                    # TODO: recomp
                    self.gene_encoder(gene_pos),
                    req_depth=req_depth,
                )
            )
            output["mvc_mean"] = (
                depth_mult.unsqueeze(1) * output["mvc_mean"]
            )  # (minibatch, seq_len)
        return output

    def forward(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        neighbors: Optional[Tensor] = None,
        neighbors_info: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        req_depth: Optional[Tensor] = None,
        get_gene_emb: bool = False,
        metacell_token: Optional[Tensor] = None,  # (minibatch, 1)
        depth_mult: Optional[Tensor] = None,
        do_sample: bool = False,
        do_mvc: bool = False,
        do_class: bool = False,
        get_attention_layer: Optional[list] = None,
        mask_zeros: Optional[Tensor] = None,
    ) -> Dict[str, Tensor] | tuple[Dict[str, Tensor], list]:
        """
        Complete forward pass through the scPRINT-2 model.

        Encodes input expression data, processes through transformer(s), and
        decodes into expression predictions and cell classifications.

        Args:
            gene_pos (Tensor): Gene indices of shape (batch, seq_len) mapping to
                positions in the model's gene vocabulary.
            expression (Tensor, optional): Expression values of shape (batch, seq_len).
                Can be raw counts or normalized depending on model config.
                Defaults to None.
            neighbors (Tensor, optional): KNN neighbor expressions of shape
                (batch, n_neighbors, seq_len) for metacell-style encoding.
                Defaults to None.
            neighbors_info (Tensor, optional): Neighbor weights of shape
                (batch, n_neighbors). Defaults to None.
            mask (Tensor, optional): Boolean mask of shape (batch, seq_len) where
                True indicates positions to mask (set to zero). Defaults to None.
            req_depth (Tensor, optional): Target sequencing depth of shape (batch,)
                for depth-conditional generation. Defaults to None.
            get_gene_emb (bool, optional): Return gene embeddings from transformer.
                Defaults to False.
            metacell_token (Tensor, optional): Binary metacell indicators of shape
                (batch,). Defaults to None.
            depth_mult (Tensor, optional): Expression depth multiplier. If None,
                uses sum of expression values. Defaults to None.
            do_sample (bool, optional): Sample from predicted distribution.
                Currently unused. Defaults to False.
            do_mvc (bool, optional): Compute multi-view coding predictions.
                Defaults to False.
            do_class (bool, optional): Compute classification predictions.
                Defaults to False.
            get_attention_layer (list[int], optional): Layer indices to extract
                attention weights from. Defaults to None.
            mask_zeros (Tensor, optional): Boolean mask for zero-expression genes
                of shape (batch, seq_len + num_special_tokens). Defaults to None.

        Returns:
            dict[str, Tensor] | tuple[dict, list]: Model outputs containing:
                - "mean": Predicted expression (batch, seq_len)
                - "disp": Dispersion parameters (batch, seq_len) [if ZINB]
                - "zero_logits": Zero-inflation logits (batch, seq_len) [if ZINB]
                - "input_cell_embs": Cell embeddings (batch, n_classes+1, d_model)
                - "input_cell_emb": Mean cell embedding (batch, d_model)
                - "output_cell_embs": Processed cell embeddings
                - "output_cell_emb": Final cell embedding
                - "cls_output_{class}": Classification logits for each class
                - "gene_embedding": Gene embeddings [if get_gene_emb]
                - "mvc_*": MVC predictions [if do_mvc]

                If get_attention_layer is not None, returns (outputs_dict, attention_list)
                where attention_list contains QKV tensors from specified layers.

        Example:
            >>> output = model(
            ...     gene_pos=batch["genes"],
            ...     expression=batch["x"],
            ...     req_depth=batch["depth"],
            ...     do_class=True,
            ... )
            >>> predictions = output["mean"]
            >>> cell_types = output["cls_output_cell_type_ontology_term_id"].argmax(-1)
        """
        cell_embs, encoding = self._encoder(
            gene_pos,
            expression,
            neighbors,
            neighbors_info,
            mask,
            metacell_token=metacell_token,
        )

        # attention_bias
        num = (1 if self.use_metacell_token else 0) + (
            (len(self.classes) + 1) if not self.cell_transformer else 0
        )
        if self.attn_bias is not None:
            if not hasattr(self, "nbias_sparse"):
                bias_path = os.path.join(self.attn_bias)
                # Keep as sparse matrix - much more memory efficient
                self.nbias_sparse = load_npz(bias_path)

            bias = torch.zeros(
                (
                    gene_pos.shape[0],
                    gene_pos.shape[1] + num,
                    gene_pos.shape[1] + num,
                ),
                device=gene_pos.device,
                dtype=torch.float16,
            )

            fade_factor = 100

            # Extract only the needed values from sparse matrix
            batch_size = gene_pos.shape[0]

            # Vectorized extraction from sparse matrix
            for b in range(batch_size):
                indices = gene_pos[b].cpu().numpy()
                # Get submatrix for this batch's genes
                submatrix = self.nbias_sparse[np.ix_(indices, indices)]
                bias[b, num:, num:] = (
                    torch.tensor(
                        submatrix.toarray(), device=gene_pos.device, dtype=torch.float16
                    )
                    * fade_factor
                )

            bias[:, num:, :num] = -10_000
        if not self.cell_transformer:
            encoding = torch.cat([cell_embs, encoding], dim=1)
        if type(self.transformer) is FlashTransformer:
            transformer_output = self.transformer(
                encoding,
                return_qkv=get_attention_layer,
                bias=bias if self.attn_bias is not None else None,
                bias_layer=list(range(self.nlayers - 1)),
                mask_zeros=mask_zeros,
            )
        elif type(self.transformer) is Performer:
            transformer_output = self.transformer(encoding)
        else:
            raise ValueError(f"Unknown transformer: {type(self.transformer)}")
        if get_attention_layer is not None:
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
        req_depth = torch.log2(1 + req_depth)
        res = self._expr_decoder(
            transformer_output[:, (1 if self.use_metacell_token else 0) :, :],
            depth_mult,
            req_depth,
            get_gene_emb,
        )
        res.update(
            self._cell_decoder(
                cell_embs,
                do_mvc,
                do_class,
                depth_mult,
                req_depth,
                gene_pos if do_mvc else None,
            )
        )
        return (res, qkvs) if get_attention_layer is not None else res

    def _generate(
        self,
        cell_embs: Tensor,
        gene_pos: Tensor,
        depth_mult: Tensor,
        req_depth: Optional[Tensor] = None,
        metacell_token: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        mask_zeros: Optional[Tensor] = None,
        **decoder_kwargs,
    ):
        """
        Generate expression profiles from cell embeddings.

        Given cell embeddings (e.g., from classification heads), generates
        predicted expression profiles. Useful for understanding what expression
        patterns the model associates with particular cell states.

        Args:
            cell_embs (Tensor): Cell embeddings of shape (batch, n_classes+1, d_model)
                typically from the output of a forward pass.
            gene_pos (Tensor): Gene indices of shape (batch, seq_len) specifying
                which genes to generate expression for.
            depth_mult (Tensor): Depth multiplier of shape (batch,) for scaling
                generated expression values.
            req_depth (Tensor, optional): Target sequencing depth of shape (batch,).
                Defaults to None.
            metacell_token (Tensor, optional): Metacell indicators. Defaults to None.
            bias (Tensor, optional): Attention bias matrix. Defaults to None.
            mask_zeros (Tensor, optional): Zero-expression mask. Defaults to None.

        Returns:
            dict[str, Tensor]: Generated expression containing:
                - "mean": Generated expression means (batch, seq_len)
                - "disp": Dispersion parameters [if ZINB]
                - "zero_logits": Zero-inflation logits [if ZINB]
        """
        _, encoding = self._encoder(
            cell_embs=cell_embs,
            gene_pos=gene_pos,
            metacell_token=metacell_token,
        )
        if type(self.transformer) is FlashTransformer:
            if self.cell_transformer:
                transformer_output = self.transformer(
                    encoding,
                    x_kv=cell_embs,
                    mask_zeros=mask_zeros,
                )
            else:
                encoding = torch.cat([cell_embs, encoding], dim=1)
                transformer_output = self.transformer(
                    encoding,
                    mask_zeros=mask_zeros,
                )
                cell_embs, transformer_output = transformer_output.split(
                    [
                        len(self.classes) + 1,
                        transformer_output.shape[1] - (len(self.classes) + 1),
                    ],
                    dim=1,
                )

        elif type(self.transformer) is Performer:
            encoding = torch.cat([cell_embs, encoding], dim=1)
            transformer_output = self.transformer(encoding)
            cell_embs, transformer_output = transformer_output.split(
                [
                    len(self.classes) + 1,
                    transformer_output.shape[1] - (len(self.classes) + 1),
                ],
                dim=1,
            )
        req_depth = torch.log2(1 + req_depth)
        output = self._expr_decoder(
            transformer_output[:, (1 if self.use_metacell_token else 0) :, :],
            req_depth=req_depth,
            depth_mult=depth_mult,
        )
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
                eps=1e-7,  # 1e-5 to 1e-8
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "adamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-7,  # 1e-5 to 1e-8
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
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    optimizer,
        #    T_0=20000,
        #    T_mult=2,
        #    eta_min=1e-8,
        # )
        # interval = "step"
        # frequency = 10
        # lr_scheduler = optim.lr_scheduler.ExponentialLR(
        #    optimizer,
        #    gamma=0.85,
        # )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_reduce_patience,
            factor=self.lr_reduce_factor,
        )
        interval = "epoch"
        frequency = 1
        # lr_scheduler = StepwiseCAWRWithWD(
        #     optimizer,
        #     T_0=20_000,
        #     T_mult=2,
        #     eta_min=1e-8,
        #     wd_decay=0.9
        # )
        lr_dict = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": interval,
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": frequency,
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
        batch_idx: int,
    ) -> Tensor:
        """
        training_step defines the train loop. It is independent of forward

        @see pl.LightningModule

        Returns:
            Tensor: Total loss value for the training step.
        """
        total_loss, losses = self._full_training(
            batch=batch,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            cce_temp=self.cce_temp,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        if total_loss is None or torch.isnan(total_loss):
            raise ValueError("Loss is NaN")
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
        noise: list[float] = [0.4],
        do_next_tp: bool = False,
        cce_temp: float = 0.5,
        do_generate: bool = False,
        run_full_forward: bool = True,
        mask_ratio: list[float] = [0.15],
        do_vae_kl: bool = True,
    ):
        """
        Execute complete training logic with multiple objectives.

        Performs forward passes with different augmentations (masking, denoising)
        and computes combined loss from multiple training objectives.

        Training Tasks:
            1. Full forward pass (unmasked reconstruction)
            2. Masked expression prediction (for each ratio in mask_ratio)
            3. Denoising (for each dropout in noise)
            4. Expression generation from embeddings (if do_generate)
            5. Contrastive cell embedding (ECS loss)
            6. Cross-view contrastive (CCE loss between augmentations)

        Args:
            batch (dict[str, Tensor]): Training batch (see training_step).
            noise (list[float], optional): Dropout rates for denoising.
                1.0 means full dropout (generation from embeddings only).
                Defaults to [0.4].
            do_next_tp (bool, optional): Reserved for temporal prediction.
                Not implemented. Defaults to False.
            cce_temp (float, optional): Temperature for contrastive loss.
                Defaults to 0.5.
            do_generate (bool, optional): Include expression generation task.
                Defaults to False.
            run_full_forward (bool, optional): Include unmasked forward pass.
                Defaults to True.
            mask_ratio (list[float | str], optional): Mask ratios for masked
                prediction. Can include "TF" for transcription-factor-weighted
                masking. Defaults to [0.15].
            do_vae_kl (bool, optional): Include VAE KL divergence loss.
                Defaults to True.

        Returns:
            tuple[Tensor, dict[str, Tensor]]: (total_loss, individual_losses_dict)
                Individual losses are prefixed with task name (e.g., "mask_15%_expr").
        """
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio]

        # dynamically change the context length every 5 steps
        if self.var_context_length and torch.rand(1).item() < 0.2:
            context_length = torch.randint(800, batch["x"].shape[1], (1,)).item()
        else:
            context_length = batch["x"].shape[1]
        expression = batch["x"]
        gene_pos = batch["genes"]
        # if multiple cells
        knn_cells = batch.get("knn_cells", None)
        knn_cells_info = batch.get("knn_cells_info", None)
        if knn_cells is not None:
            nn = min(6, int(7 * np.random.random()))
        else:
            nn = 0

        total_count = batch["depth"]
        clss = batch.get("class", None)
        # print(clss)
        batch_idx = batch.get("dataset", None)
        metacell_token = batch.get("is_meta", None)
        if metacell_token is None:
            if self.use_metacell_token:
                raise ValueError(
                    "metacell_token is not provided but use_metacell_token is True"
                )

        mask_zeros = None
        if self.mask_zeros:
            num = (1 if self.use_metacell_token else 0) + (
                (len(self.classes) + 1) if not self.cell_transformer else 0
            )
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
            if knn_cells is None:
                keep = expression.sum(0) != 0
                # we can work on smaller datasets
                if keep.sum() != keep.shape[0]:
                    expression = expression[:, keep]
                    gene_pos = gene_pos[:, keep]

        if self.transformer.attn_type == "hyper":
            # seq len must be a multiple of 128
            num = (1 if self.use_metacell_token else 0) + (
                (len(self.classes) + 1) if not self.cell_transformer else 0
            )
            if (expression.shape[1] + num) % 128 != 0:
                context_length = (context_length // 128) * 128 - num
                expression = expression[:, : ((expression.shape[1]) // 128 * 128) - num]
                gene_pos = gene_pos[:, : ((gene_pos.shape[1]) // 128 * 128) - num]
                if knn_cells is not None:
                    knn_cells = knn_cells[
                        :, :, : ((knn_cells.shape[2]) // 128 * 128) - num
                    ]
        total_loss = 0
        losses = {}
        cell_embs = []
        do_cls = self.class_scale > 0
        do_mvc = self.mvc_decoder is not None
        do_knn = knn_cells_info is not None and nn > 0
        if run_full_forward:
            output = self.forward(
                gene_pos,
                expression,
                neighbors=knn_cells[:, :nn] if do_knn else None,
                neighbors_info=knn_cells_info[:, :nn] if do_knn else None,
                mask=None,
                req_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
                metacell_token=metacell_token,
                mask_zeros=mask_zeros,
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
                do_cls,
                do_vae_kl=do_vae_kl,
            )
            cell_embs.append(output["input_cell_emb"].clone())
            full_cell_embs = output["output_cell_embs"].clone()
            total_loss += tot
            losses.update({"full_forward_" + k: v for k, v in l.items()})
            do_mvc = False
            do_cls = False

        for i in mask_ratio:
            # do noise and mask
            if False:
                if knn_cells is not None:
                    knn_cells_sub = utils.downsample_profile(
                        knn_cells[:, :nn], dropout=0.5, randsamp=self.randsamp
                    )
                    expr = expression
                else:
                    expr = utils.downsample_profile(
                        expression, dropout=0.5, randsamp=self.randsamp
                    )
            else:
                expr = expression
                knn_cells_sub = knn_cells[:, :nn] if do_knn else None
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
                neighbors=knn_cells_sub,
                neighbors_info=knn_cells_info[:, :nn] if do_knn else None,
                mask=mask,
                req_depth=expr.sum(1),
                do_mvc=do_mvc,
                do_class=do_cls,
                metacell_token=metacell_token,
                mask_zeros=mask_zeros,
            )
            l, tot = self._compute_loss(
                output,
                expr,
                clss,
                batch_idx,
                do_cls,
                do_mse=self.zinb_and_mse,
                do_vae_kl=do_vae_kl,
            )
            # we only want to do them once
            do_mvc = False
            do_cls = False

            cell_embs.append(output["input_cell_emb"].clone())
            total_loss += tot
            pct = str(int(i * 100)) + "%_" if i != "TF" else "TF_"
            losses.update({"mask_" + pct + k: v for k, v in l.items()})
        # TASK 3. denoising
        for i in noise:
            if i == 1.0:
                expr = torch.zeros_like(expression)
                dnn = 6
            else:
                expr = utils.downsample_profile(
                    expression, dropout=i, randsamp=self.randsamp
                )
                dnn = nn
            do_knn = knn_cells_info is not None and dnn > 0
            output = self.forward(
                gene_pos[:, :context_length],
                expression=expr[:, :context_length],
                neighbors=knn_cells[:, :dnn, :context_length] if do_knn else None,
                neighbors_info=knn_cells_info[:, :dnn] if do_knn else None,
                mask=None,
                depth_mult=expression[:, :context_length].sum(1),
                req_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
                metacell_token=metacell_token,
                mask_zeros=mask_zeros,
            )
            l, tot = self._compute_loss(
                output,
                expression[:, :context_length],
                clss,
                batch_idx,
                do_cls,
                do_mse=self.zinb_and_mse,
                do_vae_kl=do_vae_kl,
            )
            do_mvc = False
            do_cls = False

            cell_embs.append(output["input_cell_emb"].clone())
            total_loss += tot
            losses.update(
                {"denoise_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
            )
            # make sure that the cell embedding stay the same even if the expression is decreased

        # TASK 6. expression generation
        if do_generate:
            output = self._generate(
                cell_embs=(
                    output["output_cell_embs"]
                    if not run_full_forward
                    else full_cell_embs
                ),
                gene_pos=gene_pos,
                depth_mult=(expression.sum(1)),
                req_depth=total_count,
                mask_zeros=mask_zeros,
            )
            l, tloss = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                False,
                do_mse=self.zinb_and_mse,
                do_vae_kl=do_vae_kl,
            )
            losses.update({"gen_" + k: v for k, v in l.items()})
            total_loss += tloss  # * 0.5

        # TASK 7. next time point prediction
        if do_next_tp:
            pass
        # Gather cell embeddings from all devices
        if self.trainer.world_size > 1:
            gathered_cell_embs_list = []
            for cell_emb in cell_embs:
                gathered_emb = self.all_gather(cell_emb)
                # Reshape to combine all devices
                gathered_emb = gathered_emb.view(-1, gathered_emb.shape[-1])
                gathered_cell_embs_list.append(gathered_emb)
        else:
            gathered_cell_embs_list = cell_embs
        # TASK 4. contrastive cell embedding
        if self.ecs_scale > 0:
            loss_ecs = loss.ecs(
                gathered_cell_embs_list[0], ecs_threshold=self.ecs_threshold
            )
            total_loss += self.ecs_scale * loss_ecs
            losses.update({"ecs": loss_ecs})
        # TASK 5. elastic cell similarity
        if self.cce_scale > 0 and len(gathered_cell_embs_list) > 1:
            loss_cce = 0
            n_pairs = 0
            for i, cell_emb1 in enumerate(gathered_cell_embs_list[:-1]):
                for cell_emb2 in gathered_cell_embs_list[(i + 1) :]:
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
        do_cls=False,
        do_mse=0,
        do_vae_kl=False,
        spl_expression=None,
    ):
        """
        Compute losses from model outputs.

        Calculates expression reconstruction loss (ZINB, NB, or MSE), classification
        losses with optional hierarchy, adversarial losses, MVC loss, and VAE KL.

        Args:
            output (dict[str, Tensor]): Model forward pass outputs.
            expression (Tensor): Ground truth expression (batch, seq_len).
            clss (Tensor): Ground truth class labels (batch, n_classes).
            batch_idx (Tensor): Dataset indices for batch effect handling.
            do_cls (bool, optional): Compute classification losses. Defaults to False.
            do_mse (float, optional): Weight for additional MSE loss on expression.
                Defaults to 0.
            do_vae_kl (bool, optional): Include VAE KL loss with warmup.
                Defaults to False.
            spl_expression (Tensor, optional): Spliced expression for splicing head.
                Defaults to None.

        Returns:
            tuple[dict[str, Tensor], Tensor]: (individual_losses, total_loss)
                Individual losses include:
                - "expr": Expression reconstruction loss
                - "cls": Classification loss (if do_cls)
                - "emb_independence": Embedding dissimilarity loss
                - "adv_cls_*": Adversarial classification losses
                - "expr_mvc": MVC reconstruction loss
                - "vae_kl": VAE KL divergence
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
                mask=self.mask_zeros,
            )
            if do_mse:
                loss_expr += (
                    loss.mse(
                        input=output["mean"]
                        * (1 - torch.sigmoid(output["zero_logits"])),
                        target=expression,
                        mask=self.mask_zeros,
                    )
                    / 2  # scale to make it more similar to the zinb
                    # indeed it gets to ~3 at conv whereas zinb gets to ~ 1.1
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
            if self.expr_emb_style == "binned":
                # we are in binned case
                loss_expr = torch.nn.functional.cross_entropy(
                    input=output["mean"].flatten(0, 1),
                    target=expression.long().flatten(),
                )
            else:
                loss_expr = loss.mse(
                    # log1p is done in the function
                    input=output["mean"],
                    target=expression,
                    mask=self.mask_zeros,
                )
                if self.splicing_head is not None:
                    loss_nov_expr = loss.mse(
                        input=torch.log(output["spl_mean"] + 1),
                        target=torch.log(spl_expression + 1),
                    )
        else:
            loss_expr = 0
        total_loss += loss_expr
        losses.update({"expr": loss_expr})
        if self.splicing_head is not None:
            losses.update({"spl_expr": loss_nov_expr})
            total_loss += loss_nov_expr

        # TASK 2. predict classes
        if len(self.classes) > 0 and "input_cell_embs" in output and do_cls:
            # Calculate pairwise cosine similarity for the embeddings
            if self.class_embd_diss_scale > 0:
                loss_emb_indep = loss.within_sample(output["input_cell_embs"])
                losses.update({"emb_independence": loss_emb_indep})
                total_loss += self.class_embd_diss_scale * loss_emb_indep
            # compute class loss
            loss_cls = 0
            for j, clsname in enumerate(self.classes):
                if "cls_output_" + clsname not in output:
                    continue
                # setting the classes from index to one hot
                loss_cls += loss.hierarchical_classification(
                    pred=output["cls_output_" + clsname],
                    cl=clss[:, j],
                    labels_hierarchy=(
                        self.mat_labels_hierarchy[clsname]
                        if clsname in self.mat_labels_hierarchy.keys()
                        else None
                    ),
                )

                # Adversarial part for 'assay_ontology_term_id'
                if self.do_adv_cls and clsname in [
                    "assay_ontology_term_id",
                    "organism_ontology_term_id",
                ]:
                    loc = self.classes.index("cell_type_ontology_term_id") + 1
                    # Apply gradient reversal to the input embedding

                    adv_input_emb = loss.grad_reverse(
                        (
                            output["compressed_cell_embs"][loc].clone()
                            if self.compressor is not None
                            else output["input_cell_embs"][:, loc, :].clone()
                        ),
                        lambd=1.0,
                    )
                    # Get predictions from the adversarial decoder
                    if "assay" in clsname:
                        adv_pred = self.adv_assay_decoder(adv_input_emb)
                        # Replace elements in clss[:, j] using self.assay_relab mapping
                        cl = torch.zeros_like(clss[:, j])
                        for i in range(cl.shape[0]):
                            cl[i] = self.assay_relab.get(clss[i, j].item(), -1)
                    else:
                        adv_pred = self.adv_organism_decoder(adv_input_emb)
                        cl = clss[:, j]

                    # Compute the adversarial loss
                    adv_loss = torch.nn.functional.cross_entropy(
                        input=adv_pred,
                        target=cl.long(),
                    )
                    # Add the adversarial loss to the total loss (gradient reversal handles the maximization objective for the generator)
                    total_loss += self.adv_class_scale * adv_loss
                    losses.update({"adv_cls_" + clsname: adv_loss})

            total_loss += self.class_scale * loss_cls
            if loss_cls != 0:
                losses.update({"cls": loss_cls})

        if "mvc_zero_logits" in output:
            loss_expr_mvc = loss.zinb(
                theta=output["mvc_disp"],
                pi=output["mvc_zero_logits"],
                mu=output["mvc_mean"],
                target=expression,
                mask=self.mask_zeros,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        elif "mvc_mean" in output:
            loss_expr_mvc = loss.mse(
                input=output["mvc_mean"], target=expression, mask=self.mask_zeros
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})

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
                print("OPTIMIZER HAS INCREASED LR. WHYY?")
                print(optimizer.param_groups[0]["lr"], self.hparams.lr)
                optimizer.param_groups[0]["lr"] = self.hparams.lr

        optimizer.step(closure=optimizer_closure)

    def on_validation_start(self):
        print("val start")
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):
        self.embs = None
        self.counter = 0
        self._store_adv_cls = self.do_adv_cls
        self.do_adv_cls = False

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
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            cce_temp=self.cce_temp,
            do_vae_kl=False,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        metacell_token = batch.get("is_meta", None)
        knn_cells = batch.get("knn_cells", None)
        knn_cells_info = batch.get("knn_cells_info", None)

        # TODO: make this faster by only calling val loss
        if self.embs is not None:
            if self.pos.shape[0] < 100_000 / self.trainer.world_size:
                self.info = torch.cat([self.info, batch["class"]])
                self._predict(
                    gene_pos,
                    expression,
                    depth,
                    knn_cells=knn_cells,
                    knn_cells_info=knn_cells_info,
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
                knn_cells_info=knn_cells_info,
                pred_embedding=self.pred_embedding,
                max_size_in_mem=120_000,
                metacell_token=metacell_token,
            )
        self.log("val_loss", val_loss, sync_dist=True)
        expr_loss = mean(
            [
                v.cpu().item() if type(v) is Tensor else v
                for k, v in losses.items()
                if "expr" in k
            ]
        )
        self.log("val_loss_expr", expr_loss, sync_dist=True)
        cls_loss = mean(
            [
                v.cpu().item() if type(v) is Tensor else v
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
        self.do_adv_cls = self._store_adv_cls
        gathered_embs = self.all_gather(self.embs)
        # Merge the dictionaries from all processes
        for key in self.embs.keys():
            self.embs[key] = gathered_embs[key].view(-1, gathered_embs[key].shape[-1])
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
        self.pred = None

    def on_test_start(self):
        """@see pl.LightningModule"""
        print("test start")
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def on_test_epoch_end(self):
        # Run the test only on global rank 0
        name = str(self.name) + "_step" + str(self.global_step) + "_test_metrics"
        import json

        try:
            metrics, tot = utils.test(
                self,
                filedir=str(FILEDIR),
                do_class=self.class_scale > 0,
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
        print("predict epoch start")
        self.embs = None
        self.attn.data = None
        self.attn.attn = None
        self.counter = 0
        if type(self.transformer) is FlashTransformer:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """
        embed given gene expression, encode the gene embedding and cell embedding.

        Args:
            batch (Dict[str, Tensor]): Dictionary containing 'genes', 'x', 'depth', and optionally 'knn_cells'.
            batch_idx: Index of the batch.

        Returns:
            Dict[str, Tensor]: Dictionary containing model predictions.
        """
        return self._predict(
            batch["genes"],
            batch["x"],
            batch["depth"],
            batch.get("knn_cells", None),
            batch.get("knn_cells_info", None),
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
        knn_cells_info=None,
        do_generate=False,
        pred_embedding=None,
        get_attention_layer=None,
        depth_mult=1,
        keep_output=True,
        max_size_in_mem=100_000,
        get_gene_emb=False,
        mask=None,
        metacell_token=None,
        generate_on=None,
        name="predict_part_",
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
        # self.keep_all_labels_pred = True
        # self.mask_zeros = True
        if self.transformer.attn_type == "hyper":
            # seq len must be a multiple of 128
            num = (1 if self.use_metacell_token else 0) + (
                (len(self.classes) + 1) if not self.cell_transformer else 0
            )
            if (expression.shape[1] + num) % 128 != 0:
                expression = expression[:, : ((expression.shape[1]) // 128 * 128) - num]
                gene_pos = gene_pos[:, : ((gene_pos.shape[1]) // 128 * 128) - num]
                if knn_cells is not None:
                    knn_cells = knn_cells[
                        :, :, : ((knn_cells.shape[2]) // 128 * 128) - num
                    ]
        output = self.forward(
            gene_pos,
            expression,
            depth_mult=expression.sum(1) * depth_mult,
            neighbors=knn_cells,
            neighbors_info=knn_cells_info,
            req_depth=depth * depth_mult,
            get_attention_layer=get_attention_layer,
            do_class=True,
            get_gene_emb=get_gene_emb,
            metacell_token=metacell_token,
            mask=mask,
        )
        if get_attention_layer is not None:
            # only first 2 (QK)
            self.attn.add(
                [i[:, :, :2, :] for i in output[1]],
                gene_pos,
                expression if self.mask_zeros else None,
            )
            output = output[0]
        if do_generate:
            output.update(
                self._generate(
                    output["output_cell_embs"],
                    gene_pos if generate_on is None else generate_on,
                    req_depth=depth * depth_mult,  # otherwise we have 2 depths passed
                    depth_mult=expression.sum(1) * depth_mult,
                )
            )
        ind = {}
        if pred_embedding is None:
            pred_embedding = ["all"]
        if "other" in pred_embedding or ["all"] == pred_embedding:
            ind = {"other": 0}
        if ["all"] == pred_embedding:
            pred_embedding = self.classes
        ind.update({i: self.classes.index(i) + 1 for i in pred_embedding})
        if not keep_output:
            return {
                "embs": {
                    n: (
                        output["compressed_cell_embs"][loc]
                        if self.compressor is not None
                        else output["output_cell_embs"][:, loc, :]
                    )
                    for n, loc in ind.items()
                },
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
                "pos": gene_pos if generate_on is None else generate_on,
                "expr": (
                    [output["mean"], output["disp"], output["zero_logits"]]
                    if "disp" in output
                    else [
                        (
                            output["mean"]
                            if self.expr_emb_style != "binned"
                            else output["mean"].argmax(-1)
                        )
                    ]
                ),
            }
        if self.embs is None:
            self.embs = {
                n: (
                    output["compressed_cell_embs"][loc]
                    if self.compressor is not None
                    else output["output_cell_embs"][:, loc, :]
                )
                for n, loc in ind.items()
            }
            self.pred = (
                torch.cat(
                    [
                        (
                            torch.argmax(
                                output["cls_output_" + clsname], dim=1
                            ).unsqueeze(1)
                            if not self.keep_all_labels_pred
                            else output["cls_output_" + clsname]
                        )
                        for clsname in self.classes
                    ],
                    dim=1,
                )
                if len(self.classes) > 0
                else None
            )
            self.pos = gene_pos if generate_on is None else generate_on
            self.expr_pred = (
                [output["mean"], output["disp"], output["zero_logits"]]
                if "disp" in output
                else [
                    (
                        output["mean"]
                        if self.expr_emb_style != "binned"
                        else output["mean"].argmax(-1)
                    )
                ]
            )
        else:
            self.embs = {
                n: (
                    torch.cat([self.embs[n], output["compressed_cell_embs"][loc]])
                    if self.compressor is not None
                    else torch.cat(
                        [self.embs[n], output["output_cell_embs"][:, loc, :]]
                    )
                )
                for n, loc in ind.items()
            }
            self.pred = (
                torch.cat(
                    [
                        self.pred,
                        torch.cat(
                            [
                                (
                                    torch.argmax(
                                        output["cls_output_" + clsname], dim=1
                                    ).unsqueeze(1)
                                    if not self.keep_all_labels_pred
                                    else output["cls_output_" + clsname]
                                )
                                for clsname in self.classes
                            ],
                            dim=1,
                        ),
                    ],
                )
                if len(self.classes) > 0
                else None
            )
            self.pos = torch.cat(
                [self.pos, gene_pos if generate_on is None else generate_on]
            )
            self.expr_pred = (
                [
                    torch.cat([self.expr_pred[0], output["mean"]]),
                    torch.cat([self.expr_pred[1], output["disp"]]),
                    torch.cat([self.expr_pred[2], output["zero_logits"]]),
                ]
                if "disp" in output
                else [
                    torch.cat(
                        [
                            self.expr_pred[0],
                            (
                                output["mean"]
                                if self.expr_emb_style != "binned"
                                else output["mean"].argmax(-1)
                            ),
                        ]
                    )
                ]
            )
        if self.embs is not None:
            if self.pos.shape[0] > max_size_in_mem:
                if self.pred_log_adata:
                    print("logging")
                    self.log_adata(name=name + str(self.counter))
                    self.counter += 1
                else:
                    print(
                        "WARNING, reached max size in memory, deleting the adata, \
                        need to set pred_log_adata to True to log the adata"
                    )
                self.pos = None
                self.expr_pred = None
                self.embs = None
                return self.pred

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
            pos=self.pos if self.save_expr else None,
            expr_pred=self.expr_pred if self.save_expr else None,
            classes=self.classes,
            pred=self.pred if not self.keep_all_labels_pred else None,
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
        if self.doplot and fig is not None:
            logged = False
            try:
                self.logger.experiment.add_figure(fig)
                logged = True
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig], step=self.global_step)
                logged = True
            except:
                print("couldn't log to wandb")
            if not logged:
                fig.savefig(mdir + "/umap_" + self.name + "_" + name + ".png")

        return adata

    @property
    def genes(self) -> list[str]:
        """
        Get flattened list of all genes in the model's vocabulary.

        For multi-organism models, concatenates genes from all organisms
        in consistent order.

        Returns:
            list[str]: Gene names in model vocabulary order.
        """
        if type(self._genes) is list:
            return self._genes
        else:
            genes = []
            for names in self.organisms:
                genes.extend(self._genes[names])
            return genes
