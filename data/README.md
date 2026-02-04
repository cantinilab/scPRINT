# README

from mpmath.matrices.matrices import rowsep

This dataset contains the largest single cell RNA sequencing (scRNA-seq)
embedding atlas to recorded to date: ~338M cells from ~22,000 datasets, 16
species, 320 tissues.

It has bee generated through 4 distinct runs of multiple days and over 4 GPUs.

## The format

`step_[RUN_COUNT]__predict_part_[SUBSET_COUNT]_[GPU_COUNT]_*`

For each subset of data there are 3 files:

- `*_embs.npz`: The full scprint2 cell embeddings in npz format
- `*_top3.npz`: The top 3 predicted labels for each class and each cell in npz
  format
- `*_scores.npz`: The scores for the top 3 predicted labels for each class and
  each cell in npz format

for each files, rows are cells.

within the `*_embs.npz` files columns are the embedding dimesion which is a
concatenation of the following embeddings, in order:

- the "remaining" embedding (other): 256 dimensions
- cell_type: 64 dimensions
- tissue: 32 dimensions
- disease: 32 dimensions
- age: 8 dimensions
- assay: 12
- self_reported_ethnicity: 8 dimensions
- sex: 2 dimensions
- organism: 8 dimensions
- cell_culture: 2 dimensions

within the `scores.npz` files, columns are the scores for the top N predicted
labels for each class, in order:

- cell_type_ontology_term_id: 3
- tissue_ontology_term_id: 3
- disease_ontology_term_id: 3
- age_group: 3
- assay_ontology_term_id: 3
- self_reported_ethnicity_ontology_term_id: 3
- sex_ontology_term_id: 1
- organism_ontology_term_id: 3
- cell_culture: 1

within the `top3.npz` files, columns are the same top N predicted labels for
each class in the same order as above.

## What can you do with this dataset?

due to the preprocessing dropping out some low quality cells and some species
and modalities and the subsetting to different GPUs. I have not been able to map
them to their original dataset indices yet. However you can still use this for
many purposes:

- look at the distribution of cell types across tissues, diseases, ages, assays
  to define what is missing, what exists, what doesn't
- use the embeddings to train some downstream models, e.g. to predict cell types
  or tissues from embeddings
- assess the similarity in predictions between the top1 and top2 and top3 labels
  according to scprint
