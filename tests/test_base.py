import os
import urllib.request

import lamindb as ln
import numpy as np
import pytest
import scanpy as sc
import torch
from lightning.pytorch import Trainer
from scdataloader import DataModule, Preprocessor
from scdataloader.utils import populate_my_ontology

from scprint import scPrint
from scprint.base import NAME
from scprint.tasks import Denoiser, Embedder, GNInfer
from scprint.trainer import TrainingMode


def test_base():
    assert NAME == "scprint"
    populate_my_ontology(
        organisms_clade=["vertebrates"],
        sex=["PATO:0000384", "PATO:0000383"],
        # celltypes=None,
        # ethnicities=None,
        # assays=None,
        # tissues=None,
        # diseases=None,
        # dev_stages=None,
    )
    filepath = os.path.join(os.path.dirname(__file__), "test.h5ad")
    ckpt_path = os.path.join(os.path.dirname(__file__), "small.ckpt")
    if not os.path.exists(ckpt_path):
        url = "https://huggingface.co/jkobject/scPRINT/resolve/main/small.ckpt"
        urllib.request.urlretrieve(url, ckpt_path)

    adata = sc.read_h5ad(filepath)
    adata.obs.drop(columns="is_primary_data", inplace=True, errors="ignore")
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    preprocessor = Preprocessor(
        do_postp=False,
        force_preprocess=True,
    )
    adata = preprocessor(adata)
    # conf = dict(self.config_init[subcommand])
    model = scPrint.load_from_checkpoint(
        ckpt_path,
        precpt_gene_emb=None,
        # triton gets installed so it must think it has cuda enabled
        transformer="normal",
    )
    dn = Denoiser(
        max_cells=10,
        batch_size=2,
        num_workers=1,
        max_len=300,
        downsample=0.3,
        predict_depth_mult=3,
        dtype=torch.float32,
    )
    metrics, random_indices, adata_denoised = dn(
        model=model,
        adata=adata,
    )
    assert metrics["reco2full"] - metrics["noisy2full"] > 0, "Model is not denoising"
    # emb, class, grn inf and fit function for scPRINT
    # Cell embedding
    cell_embedder = Embedder(
        batch_size=2,
        num_workers=1,
        how="random expr",
        max_len=300,
        doclass=True,
        pred_embedding=[
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        plot_corr_size=8,
        doplot=True,
        keep_all_cls_pred=False,
        dtype=torch.float32,
    )
    adata_emb, metrics = cell_embedder(model, adata[:10, :])
    assert "scprint_emb" in adata_emb.obsm, "Cell embedding failed"
    assert np.isnan(adata_emb.obsm["scprint_emb"]).sum() == 0, (
        "Cell embedding contains NaNs"
    )
    assert any(col.startswith("pred_") for col in adata_emb.obs.columns), (
        "Classification failed"
    )

    # GRN inference
    grn_inferer = GNInfer(
        layer=[0, 1],
        batch_size=2,
        how="random expr",
        preprocess="softmax",
        head_agg="mean",
        filtration="none",
        forward_mode="none",
        num_genes=100,
        max_cells=10,
        doplot=False,
        dtype=torch.float32,
    )
    grn_adata = grn_inferer(model, adata)
    assert "GRN" in grn_adata.varp, "GRN inference failed"
    # make a collection
    file = ln.Artifact(adata, description="test file")
    file.save()
    col = ln.Collection(file, name="test dataset")
    col.save()
    datamodule = DataModule(
        collection_name="test dataset",
        gene_embeddings=os.path.join(os.path.dirname(__file__), "test_emb.parquet"),
        hierarchical_clss=[],
        organisms=["NCBITaxon:9606"],  # , "NCBITaxon:10090"],
        how="most expr",
        max_len=200,
        add_zero_genes=0,
        # how much more you will see the most present vs less present category
        weight_scaler=10,
        clss_to_weight=["sex_ontology_term_id"],
        clss_to_predict=[
            "sex_ontology_term_id",
            "organism_ontology_term_id",
        ],
        batch_size=1,
        num_workers=1,
        # train_oversampling=2,
        validation_split=0.1,
        do_gene_pos=False,
        test_split=0.1,
    )
    _ = datamodule.setup()
    model = scPrint(
        genes=datamodule.genes,
        d_model=64,
        nhead=1,
        nlayers=1,
        # layers_cls = [d_model],
        # labels = datamodule.labels,
        # cls_hierarchy = datamodule.cls_hierarchy,
        dropout=0,
        transformer="normal",
        precpt_gene_emb=os.path.join(os.path.dirname(__file__), "test_emb.parquet"),
        mvc_decoder="inner product",
        fused_dropout_add_ln=False,
        checkpointing=False,
    )
    trainingmode = TrainingMode(
        do_denoise=True,
        noise=[0.1],
        do_cce=False,
        do_ecs=False,
        do_cls=True,
        do_mvc=True,
        mask_ratio=[],
        warmup_duration=10,
        lr_reduce_patience=10,
        test_every=10_000,
    )
    trainer = Trainer(
        gradient_clip_val=500,
        max_time={"minutes": 4},
        limit_val_batches=1,
        callbacks=[trainingmode],
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        overfit_batches=1,
        max_epochs=20,
        reload_dataloaders_every_n_epochs=100_000,
        logger=None,
        num_sanity_val_steps=0,
        max_steps=100,
    )
    initial_loss = None
    for i in range(2):
        trainer.fit(model, datamodule=datamodule)
        trainer.fit_loop.max_epochs = 20 * (
            i + 2
        )  # Reset max_epochs for next iteration
        current_loss = trainer.callback_metrics.get("train_loss")
        if initial_loss is None:
            initial_loss = current_loss
        else:
            assert current_loss < initial_loss, (
                f"Loss not decreasing: initial {initial_loss}, current {current_loss}"
            )
            initial_loss = current_loss
    # cli
    # get_Seq
    # sinkhorn
    # knn_smooth
    # layernorm
    # tmfg
    # utils
    # layer_norm
    # flashattention
    # encoders
