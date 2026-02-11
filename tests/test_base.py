import os
import urllib.request

import lamindb as ln
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from django.db.utils import OperationalError
from lightning.pytorch import Trainer
from scdataloader import DataModule, Preprocessor
from scdataloader.utils import (
    _adding_scbasecamp_genes,
    load_genes,
    populate_my_ontology,
)

from scprint2 import scPRINT2
from scprint2.base import NAME
from scprint2.tasks import Denoiser, Embedder, GNInfer
from scprint2.trainer import TrainingMode


def test_base():
    assert NAME == "scprint2"
    populate_my_ontology(
        organisms_clade=["vertebrates"],
        sex=["PATO:0000384", "PATO:0000383"],
        organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
        # celltypes=None,
        # ethnicities=None,
        # assays=None,
        # tissues=None,
        # diseases=None,
        # dev_stages=None,
    )
    try:
        _adding_scbasecamp_genes()
    except OperationalError as err:
        # External Lamin instance schemas can lag behind current bionty models.
        # Keep the test functional when that remote dependency is inconsistent.
        if "bionty_organism.abbr" not in str(err):
            raise
    filepath = os.path.join(os.path.dirname(__file__), "test.h5ad")
    ckpt_path = os.path.join(os.path.dirname(__file__), "small-v2.ckpt")
    if not os.path.exists(ckpt_path):
        url = "https://huggingface.co/jkobject/scPRINT/resolve/main/small-v2.ckpt"
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
    model = scPRINT2.load_from_checkpoint(
        ckpt_path,
        precpt_gene_emb=None,
        gene_pos_file=None,
    )
    missing = set(model.genes) - set(load_genes(model.organisms).index)
    if len(missing) > 0:
        print(
            "Warning: some genes missmatch exist between model and ontology: solving...",
        )
        model._rm_genes(missing)
    dn = Denoiser(
        max_cells=10,
        batch_size=2,
        num_workers=1,
        max_len=300,
        downsample_expr=0.7,
        predict_depth_mult=10,
        use_knn=False,
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
        doplot=True,
        keep_all_labels_pred=False,
        use_knn=False,
    )
    adata_emb, metrics = cell_embedder(model, adata[:10, :])
    assert "scprint_emb" in adata_emb.obsm, "Cell embedding failed"
    assert (
        np.isnan(adata_emb.obsm["scprint_emb"]).sum() == 0
    ), "Cell embedding contains NaNs"
    assert any(
        col.startswith("pred_") for col in adata_emb.obs.columns
    ), "Classification failed"

    # GRN inference
    grn_inferer = GNInfer(
        layer=[0, 1],
        batch_size=2,
        how="most var within",
        preprocess="softmax",
        head_agg="mean_full",
        filtration="none",
        num_genes=100,
        max_cells=10,
        use_knn=False,
    )
    grn_adata = grn_inferer(model, adata)
    assert "GRN" in grn_adata.varp, "GRN inference failed"
    # make a collection
    file = ln.Artifact(adata, description="test file")
    file.save()
    col = ln.Collection(file, key="test dataset")
    col.save()
    datamodule = DataModule(
        collection_name="test dataset",
        gene_subset=pd.read_parquet(
            os.path.join(os.path.dirname(__file__), "test_emb.parquet")
        ).index.tolist(),
        hierarchical_clss=[],
        how="most expr",
        max_len=200,
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
        test_split=0.1,
    )
    _ = datamodule.setup()
    model = scPRINT2(
        organisms=datamodule.organisms,
        genes=datamodule.genes_dict,
        d_model=64,
        nhead=1,
        num_heads_kv=1,
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
        noise=[0.1],
        mask_ratio=[],
        warmup_duration=10,
        lr_reduce_patience=10,
        test_every=10_000,
        lr_reduce_monitor="train_loss",
    )
    trainer = Trainer(
        gradient_clip_val=500,
        max_time={"minutes": 4},
        limit_val_batches=1,
        callbacks=[trainingmode],
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        overfit_batches=1,
        max_epochs=10,
        reload_dataloaders_every_n_epochs=100_000,
        logger=None,
        num_sanity_val_steps=1,
        max_steps=100,
    )
    initial_loss = None
    for i in range(2):
        trainer.fit(model, datamodule=datamodule)
        trainer.fit_loop.max_epochs = 10 * (
            i + 2
        )  # Reset max_epochs for next iteration
        current_loss = trainer.callback_metrics.get("train_loss")
        if initial_loss is None:
            initial_loss = current_loss
        else:
            assert (
                current_loss < initial_loss
            ), f"Loss not decreasing: initial {initial_loss}, current {current_loss}"
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
