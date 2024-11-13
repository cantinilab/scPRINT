# scPRINT usage

scPRINT can be used to denoise / embed (& predict labels) / infer gene networks on single-cell data.

Example of doing these tasks on a dataset is given in our manuscript and the [example notebooks](notebooks/cancer_usecase.ipynb).

But in a nutshell, here is the most minimal example of how scPRINT works: 

1. once you have loaded an anndata object, make sure you preprocess it so everything is checked out. (number of genes mentioned, gene format, raw counts used, etc...) more information on the Preprocessor is available in the scDataLoader package ()
2. Then you can denoise / embed / infer gn on this anndata using scPRINT and its helper classes. These classes follow a similar pattern to the trainer class in pytorch-lightning.

Here is an example for denoising:

```py
from scprint import scPrint
from scdataloader import Preprocessor
import scanpy as sc
from scprint.tasks import Denoiser

#better to do it in lower precision
import torch
torch.set_float32_matmul_precision('medium')

adata = sc.read_h5ad("../data/temp.h5ad")
#make sure we have this metadata
adata.obs['organism_ontology_term_id'] = "NCBITaxon:9606"
# load the model
model = scPrint.load_from_checkpoint('../data/temp/last.ckpt', precpt_gene_emb=None)

# preprocess to make sure it looks good
preprocessor = Preprocessor(do_postp=False)
adata = preprocessor(adata)

#instanciate the denoiser with params (see the class in tasks/denoiser.py)
how="most var"
denoiser = Denoiser(
    model,
    batch_size=20,
    max_len=2000,
    max_cells=100_000,
    doplot=False,
    num_workers=1,
    predict_depth_mult=10,
    how=how,
    dtype=torch.bfloat16,
)
#denoise
metrics, idxs, genes, expr = denoise(model, adata)
print(metrics)

# apply the denoising to the anndata
adata.layers['before'] = adata.X.copy()
adata.X = adata.X.tolil()
idxs = idxs if idxs is not None else range(adata.shape[0])
for i, idx in enumerate(idxs):
    adata.X[
        idx,
        adata.var.index.get_indexer(
            np.array(model.genes)[genes[i]]
            if how != "most var"
            else genes
        ),
    ] = expr_pred[i]
adata.X = adata.X.tocsr()
```

But you can do the same thing with a bash command line:

```bash
$ scprint denoise --ckpt_path ../data/temp/last.ckpt --adata ../data/temp.h5ad --how "most var" --dtype "torch.bfloat16" --batch_size 20 --max_len 2000 --max_cells 100000 --num_workers 1 --predict_depth_mult 10 --doplot false --species "NCBITaxon:9606"
```

However in this context you might have somewhat less options for the preprocessing of the anndataset. However, all parameters of the denoiser, embedder and gninfer classes are available in the cli interface as well!