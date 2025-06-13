import scanpy as sc
from scprint import scPrint
from scprint.tasks import Embedder
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import torch

torch.set_float32_matmul_precision("medium")

import lamindb as ln

# m = torch.load("../xzp23r4p.ckpt", map_location=torch.device("cpu"))
# model = scPrint.load_from_checkpoint(
#    "../xzp23r4p.ckpt",
#    precpt_gene_emb=None,
#    classes=m["hyper_parameters"]["label_counts"],
# )
adata = (
    ln.Artifact.filter(description__contains="mouse pancreatic")
    .one()
    .versions[3]
    .load()
)
 embed = Embedder(
    how="random expr",
    max_len=4000,
    add_zero_genes=0,
    num_workers=24,
    pred_embedding=["cell_type_ontology_term_id"],
    keep_all_cls_pred=False,
    output_expression="none",
    batch_size=32,
 )

 adata, metrics = embed(model, adata, cache=False)
# from anndata import concat
# 
# pred_adata = []
# 
# a = [
#     "data/step_0__predict_part_0_0.h5ad",
#     "data/step_0__predict_part_1_0.h5ad",
#     "data/step_0__predict_part_2_0.h5ad",
#     "data/step_0__predict_part_3_0.h5ad",
# ]
# for i in a:
#     pred_adata.append(sc.read_h5ad(i))
# pred_adata = concat(pred_adata)
# adata.obsm["scprint_umap"] = pred_adata.obsm["X_umap"]
# adata.obsm["scprint"] = pred_adata.X

bm = Benchmarker(
    adata,
    batch_key="assay_ontology_term_id",  # batch, tech
    label_key="cell_type_ontology_term_id",  # celltype
    embedding_obsm_keys=["X_pca", "scprint"],
    bio_conservation_metrics=BioConservation(isolated_labels=False),
    batch_correction_metrics=BatchCorrection(),
    n_jobs=40,
)
bm.benchmark()
res = bm.get_results(min_max_scale=False)
print(res)
res.to_csv("results.csv")

1 986 424 314