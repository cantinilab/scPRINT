# scPRINT-2: 🏃🏃Your next-gen single cell foundation model

[![codecov](https://codecov.io/github/cantinilab/scPRINT-2/graph/badge.svg?token=UQEA9DN2MX)](https://codecov.io/github/cantinilab/scPRINT-2)
[![CI](https://github.com/cantinilab/scPRINT-2/actions/workflows/main.yml/badge.svg)](https://github.com/cantinilab/scPRINT-2/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/scprint-2.svg)](https://badge.fury.io/py/scprint2)
[![Downloads](https://pepy.tech/badge/scprint2)](https://pepy.tech/project/scprint2)
[![Downloads](https://pepy.tech/badge/scprint2/month)](https://pepy.tech/project/scprint2)
[![Downloads](https://pepy.tech/badge/scprint2/week)](https://pepy.tech/project/scprint2)
[![GitHub issues](https://img.shields.io/github/issues/cantinilab/scPRINT-2)](https://img.shields.io/github/issues/cantinilab/scPRINT-2)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14749466.svg)](https://doi.org/10.5281/zenodo.14749466)
[![hugging face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/jkobject/scPRINT)

<img src="docs/fig.png" alt="logo" width="600" />

**scPRINT-2** is a single-cell RNA-seq foundation model built by Jérémie Kalfon
in the Cantini Lab. It uses novel architecture, encoding, decoding, training
paradigms and losses.

**scPRINT-2** has been pretrained on more than 350 million cells from more than
22,000 datasets and 16 species.

**scPRINT-2** can be used to perform the following analyses in a zero-shot mode:

- **expression denoising & imputation**: increase the resolution of your
  scRNAseq data and discover un-measured genes' expression
- **cell embedding and batch correction**: generate a low-dimensional
  representation of your dataset at syntax level (organism, disease, cell type,
  sequencer, ...)
- **label prediction**: predict the cell type, disease, sequencer, sex, age,
  tissue of origin and ethnicity of your cells.
- **gene network inference**: generate a gene network from any cell or cell
  cluster in your scRNAseq dataset
- **cross species integration**: scPRINT-2 has been trained on 16 species and
  can be used to integrate data from different species.

Example of **scPRINT-2** finetuning exist for:

- **new species**: finetune scPRINT-2 on a new organism
- **classification**: finetune scPRINT-2 on your own cell type /disease / age
  labels / more...
- **batch correction of your datasets / atlas**: finetune scPRINT-2 to integrate
  data across species, technologies, and labs.

**scPRINT-2** is a foundation model and can be fine-tuned to perform many other
analysis

[Read the manuscript!](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1)
if you would like to know more about **scPRINT-2**. Or have a look at some of my
[X-plainers](https://twitter.com/jkobject).

🎊 test scPRINT and scDataloader on this simple example
[google collab](https://colab.research.google.com/drive/1CacoQDAwJn86tq2sBhUoZ6M-xAqsYFDI#scrollTo=Lb4E9IhQ7NK8)

## Table of Contents

- [scPRINT-2: 🏃🏃Your next-gen single cell foundation model](#scprint-2-your-next-gen-single-cell-foundation-model)
  - [Table of Contents](#table-of-contents)
  - [Use `scPRINT-2`](#use-scprint-2)
    - [try scPRINT-1 in superbio.ai!](#try-scprint-1-in-superbioai)
    - [Try scPRINT-1 on a Google Colab notebook!](#try-scprint-1-on-a-google-colab-notebook)
    - [To know about: lamin.ai](#to-know-about-laminai)
    - [install](#install)
    - [pytorch and GPUs](#pytorch-and-gpus)
  - [Usage](#usage)
    - [scPRINT-2's basic commands](#scprint-2s-basic-commands)
    - [scPRINT-2's basic command line](#scprint-2s-basic-command-line)
    - [Example notebooks](#example-notebooks)
  - [Documentation](#documentation)
  - [Docker](#docker)
    - [Simple tests:](#simple-tests)
  - [FAQ](#faq)
    - [I have a dataset and want a quick analysis:](#i-have-a-dataset-and-want-a-quick-analysis)
    - [I have a dataset and want some more control over what is going on and which model to use:](#i-have-a-dataset-and-want-some-more-control-over-what-is-going-on-and-which-model-to-use)
    - [What does my anndata need to contain to be run with scPRINT-2](#what-does-my-anndata-need-to-contain-to-be-run-with-scprint-2)
    - [I want to generate an atlas-level embedding](#i-want-to-generate-an-atlas-level-embedding)
    - [I need to generate gene tokens using pLLMs](#i-need-to-generate-gene-tokens-using-pllms)
    - [I want to re-train scPRINT-2 from scratch on my own data](#i-want-to-re-train-scprint-2-from-scratch-on-my-own-data)
    - [I want to regenerate the scPRINT-2 training corpus](#i-want-to-regenerate-the-scprint-2-training-corpus)
    - [I want to fine-tune scPRINT-2 on my own data](#i-want-to-fine-tune-scprint-2-on-my-own-data)
    - [how can I find if scPRINT-2 was trained on my data?](#how-can-i-find-if-scprint-2-was-trained-on-my-data)
    - [can I use scPRINT-2 on other organisms rather than humans?](#can-i-use-scprint-2-on-other-organisms-rather-than-humans)
    - [How long does scPRINT-2 take? What kind of resources do I need? (or in alternative: can I run scPRINT-2 locally?)](#how-long-does-scprint-2-take-what-kind-of-resources-do-i-need-or-in-alternative-can-i-run-scprint-2-locally)
    - [I have different scRNASeq batches. Should I integrate my data before running scPRINT-2?](#i-have-different-scrnaseq-batches-should-i-integrate-my-data-before-running-scprint-2)
    - [I have new labels for my data that scPRINT-2 doesn't predict, how can I fine-tune it to predict them?](#i-have-new-labels-for-my-data-that-scprint-2-doesnt-predict-how-can-i-fine-tune-it-to-predict-them)
    - [where to find the input gene embeddings?](#where-to-find-the-input-gene-embeddings)
  - [Development](#development)
    - [dev install](#dev-install)
    - [Reproducibility](#reproducibility)
    - [Building the Docker Image](#building-the-docker-image)
    - [Pulling the Docker Image from Docker Hub](#pulling-the-docker-image-from-docker-hub)
    - [Running the Docker Container](#running-the-docker-container)
    - [Participate](#participate)

## Use `scPRINT-2`

For the moment **scPRINT-2** has been tested on MacOS and Linux (Ubuntu 20.04)
with Python 3.10+. Its instalation takes on average 2 minutes in `uv` but much
longer on `conda`. We highly recommend using `uv` to manage your python virtual
environments!!

Here is a link to our --still maintained-- previous generation model which
contains larger size models: [scPRINT-1](https://github.com/cantinilab/scPRINT)
(don't forget to star it as well!):

### try scPRINT-1 in superbio.ai!

[HERE](https://app.superbio.ai/apps/67333115ed44f27eb717cf84)

### Try scPRINT-1 on a Google Colab notebook!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CacoQDAwJn86tq2sBhUoZ6M-xAqsYFDI#scrollTo=Vj73HINSzKHL)

### To know about: lamin.ai

To use **scPRINT-2**, you will need to use [lamin.ai](https://lamin.ai/). This
is required to load biological information like genes, cell types, organisms..
(but also to manage the pre-training datasets if this is something you want to
set up)

### install

[Here, is how to install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv <env-name> --python 3.11
source <env-name>/bin/activate
#one of
uv pip install scprint2
# OR uv pip install scprint2[dev] # for the dev dependencies (building etc..) OR
# OR uv pip install scprint2[flash] # to use flashattention2 with triton: only if you have a compatible gpu (e.g. not available for apple GPUs for now, see https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility)
#OR pip install scprint2[dev,flash]

lamin init --storage ./testdb --name test --modules bionty
lamin connect anonymous/testdb
```

⚠️ `./testdb` is set in this example, but be mindful about where you want to
store your data, this might get quite big as you use i,t and if you are on
specific partition you want to consider this.

If you start with lamin and have to do a `lamin init`, you will also need to
populate your ontologies. This is because scPRINT-2 is using ontologies to
define its cell types, diseases, sexes, ethnicities, etc.
([link to view ontologies](https://www.ebi.ac.uk/ols4/ontologies/cl/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCL_0000057))

You can do it via the command:

`scdataloader populate all`

⚠️ It is ok to get warnings with this function

or with this function:

```python
from scdataloader.utils import populate_my_ontology

populate_my_ontology() #to populate everything (can take 2-10mns)

populate_my_ontology( #the minimum for scPRINT-1 to run some inferences (denoising, grn inference)
    organisms: List[str] = ["NCBITaxon:10090", "NCBITaxon:9606"],
    sex: List[str] = ["PATO:0000384", "PATO:0000383"],
    celltypes = None,
    ethnicities = None,
    assays = None,
    tissues = None,
    diseases = None,
    dev_stages = None,
)
_adding_scbasecamp_genes()  #to add when using scPRINT-2
```

A notebook for setting-up scPRINT-2 and lamin is also available
[here](./notebooks/prepare_scprint2.ipynb)

We make use of some additional packages we developed alongside scPRINT-2 (they
are also shipped with scprint-2 already).

Please refer to their documentation for more information:

- [scDataLoader](https://github.com/jkobject/scDataLoader): a dataloader for
  training large cell models.
- [GRnnData](https://github.com/cantinilab/GRnnData): a package to work with
  gene networks from single cell data.
- [benGRN](https://github.com/jkobject/benGRN): a package to benchmark gene
  network inference methods from single cell data.
- [simpler-flash](https://github.com/jkobject/simpler-flash): a package to
  easily use different versions of flash attention in pytorch models.
- [hierarchical-classifier](https://gist.github.com/jkobject/5b36bc4807edb440b86644952a49781e):
  a package to do hierarchical classification with pytorch when your labels can
  be mapped to a graph.

### pytorch and GPUs

scPRINT-2 can run on machines without GPUs, but it will be slow. It is highly
recommended to use a GPU for inference.

Most of the time, everything works out of the box; otherwise, please send an
issue

```python
model = scPRINT2.load_from_checkpoint(
    '../data/temp/last.ckpt', precpt_gene_emb=None, )
```

You will know more by following the
[get-started](https://cantinilab.github.io/scPRINT-2/notebooks/cancer_usecase/)
notebook.

## Usage

To get a sense of how scPRINT-2 works, have a look at our
[get-started](https://cantinilab.github.io/scPRINT-2/notebooks/cancer_usecase/)
notebook.

To start you will also need to download a checkpoint of a pretrain model like
v2-medium or some others from
[hugging face](https://huggingface.co/jkobject/scPRINT-2/)

```bash
$ hf download jkobject/scPRINT v2-medium.ckpt --local-dir .
```

### scPRINT-2's basic commands

This is a template of how you would go and use scPRINT most of the time:

```py
# import stuff
from lightning.pytorch import Trainer
from scprint2 import scPRINT2
from scdataloader import DataModule

# setup a datamodule to train scprint2 from scratch
datamodule = DataModule(...)
# setup a model parameter
model = scPRINT2(...)
# to train / fit / test the model setup a trainer
trainer = Trainer(...)
# call the fit function
trainer.fit(model, datamodule=datamodule)
# to do predictions Denoiser, Embedder, GNInfer
denoiser = Denoiser(...)
adata = sc.read_h5ad(...)
denoiser(model, adata=adata)
...
```

### scPRINT-2's basic command line

Then fine-tune or analyse on your data

```bash
$ scprint2 fit/train/predict/test/denoise/embed/gninfer/impute/gene_emb/generate/finetune --config config/[medium|large|vlarge] ...
```

To denoise a dataset:

```bash
$ scprint2 denoise --adata my_human_anndata.h5ad --ckpt_path v2-medium.ckpt --species "NCBITaxon:9606" --output_filename denoised.h5ad
```

to do embedding and classification on a dataset: (the current version implies
doing a PCA and Umap so it might need a lot of RAM if run as is)

```bash
$ scprint2 embed --adata my_human_anndata.h5ad --ckpt_path v2-medium.ckpt --species "NCBITaxon:9606" --output_filename embedded.h5ad
```

To do gene network inference on a dataset:

```bash
$ scprint2 gninfer --adata my_human_anndata.h5ad --ckpt_path v2-medium.ckpt --species "NCBITaxon:9606" --cell_type 'cell_type_name_from-cell_type-obs_col' --output_filename grn.h5ad
```

To re-train scPRINT-2 from scratch or from a checkpoint:

```bash
$ scprint2 fit --config config/base_v2.yml --config config/pretrain_large.yml --ckpt_path large.ckpt
```

find out more about the commands by running `scprint2 --help` or
`scprint2 [command] --help`.

more examples of using the command line are available in the
[docs](<[https://](https://cantinilab.github.io/scPRINT-2)>).

### Example notebooks

1. [get-started](./notebooks/prepare_scprint2.ipynb): how to set things up
2. [run scPRINT-2 on a new species](/notebooks/scPRINT-2-repro-notebooks/unknown_species_classification.ipynb):
   how to fine-tune scPRINT-2 on a new organism. you will also need to generate
   embeddings and gene locations for your organism, see the FAQ below.
3. [do gene-network inference with scPRINT-2](/notebooks/scPRINT-2-repro-notebooks/gene_networks.ipynb):
   how to use scPRINT-2 to infer gene regulatory networks from your scRNAseq
   data (the first part is about getting ground truth data with benGRN)
4. [generate cell embeddings and cell label predictions from my data](/notebooks/scprint_inference/):
   how to use scPRINT-2 to generate cell embeddings and predict cell type
5. [generate gene output embeddings from my gene expressiond data](/notebooks/gene_embeddings/):
   how to use scPRINT-2 to generate gene embeddings from your scRNAseq data
6. [do counterfactual gene expression prediction with scPRINT-2](/notebooks/counterfactual_imputation/):
   how to use scPRINT-2 to impute gene expression under different conditions
7. [do denoising with scPRINT-2](/notebooks/denoising/): how to use scPRINT-2 to
   denoise your scRNAseq data
8. [do imputation with scPRINT-2 (e.g. on Xenium Panel data)](/notebooks/imputation/):
   how to use scPRINT-2 to impute missing genes in your scRNAseq data
9. [run scPRINT-2 on some Xenium spatial transcriptomics data](/notebooks/xenium_spatial_transcriptomics/):
   how to use scPRINT-2 to analyse spatial transcriptomics data
10. [fine-tune scPRINT-2 for cell type classification and/or batch correction](/notebooks/finetune_celltype_classification/):
    how to fine-tune scPRINT-2 on your own cell type labels

## Documentation

For more information on usage, please see the documentation in
[https://www.jkobject.com/scPRINT-2/](https://cantinilab.github.io/scPRINT-2)

## Docker

By using the `scPRINT-2 Docker image`, you can bypass the complexities of manual
package installation, ensuring a consistent deployment environment. Included in
this repository is a Dockerfile that lets you craft a container for the project;
you have the choice to either build this image on your own or conveniently pull
it from Docker Hub.

Make sure that you have the `docker` command line interface installed on your
system.

A recommended way to install Docker with the correct NVIDIA drivers on Linux is
to use this
[script](https://gist.github.com/xueerchen1990/baad7baa545cb547e8633bc9e5b84786)

/!\ A MORE UP TO DATE DOCKER IMAGE is made as part of the open-problems
benchmark and available on their GitHub for all tasks where scPRINT-2 is
benchmarked

### Simple tests:

An installation of scPRINT-2 and a simple test of the denoiser is performed
during each commit to the main branch with a
[Github action](https://github.com/cantinilab/scPRINT-2/actions) and
[pytest workflow](.github/workflows/main.yml). It also provides an expected
runtime for the installation and run of scPRINT-2. We now explore the different
usages of scPRINT-2:

## FAQ

### I have a dataset and want a quick analysis:

-> use [superbio](#try-scprint-1-in-superbioai)

### I have a dataset and want some more control over what is going on and which model to use:

You will need to understand a few things, like lamindb, scdataloader, and
scprint-2's inference tool.

-> start with a quick intro using the
[google collab notebook](#try-scprint-1-on-a-google-colab-notebook)

-> look at the other FAQ element based on your desired use-case

### What does my anndata need to contain to be run with scPRINT-2

-> your anndata only needs to contain the species ontology id in its
obs['organism_ontology_term_id'] (e.g. "NCBITaxon:9606"). It also needs to
contain .var_names or .var.index with gene ids defined as ENSEMBL_IDs or
HUGO_SYMBOL.

-> That's it. You can then follow the preprocessing steps from various example
notebooks to align your anndata to our gene set, make sure that it fits our
requirements and then send it to the model!

### I want to generate an atlas-level embedding

-> Refer to the notebook
[nice_umap_explain.ipynb](./figures/nice_umap_explain.ipynb).

### I need to generate gene tokens using pLLMs

To run scPRINT-2, you can use the option to define the gene tokens using protein
language model embeddings of genes. This is done by providing the path to a
parquet file of the precomputed set of embeddings for each gene name to
scPRINT-2 via "precpt_gene_emb"

-> To generate this file please refer to the notebook
[generate_gene_embeddings](notebooks/generate_gene_embeddings.ipynb).

### I want to re-train scPRINT-2 from scratch on my own data

-> Refer to the documentation page [pretrain scprint-2](docs/pretrain.md)

### I want to regenerate the scPRINT-2 training corpus

-> Have a look at the
[scDataLoader](https://github.com/cantinilab/scDataLoader)'s README to
understand how to do this.

### I want to fine-tune scPRINT-2 on my own data

-> make sure that you a run of scPRINT-2's inference e.g.
[this one](#example-notebooks)

-> then please refine your question: do you want finetuning to predict labels?
do batch correction? or make scprint work on your species? Have a look at the
[usage](#usage) section and the rest of the [FAQ](#faq) to find the relevant
information.

### how can I find if scPRINT-2 was trained on my data?

If your data is available in [cellxgene](https://cellxgene.cziscience.com/), or
is listed in
[Arc's scBaseCount](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/scBaseCount),
scPRINT-2 was likely trained on it. However, some cells, and datasets were
dropped due to low-quality data, and some were randomly removed to be part of
the validation/test sets.

### can I use scPRINT-2 on other organisms rather than humans?

scPRINT-2 has been pretrained on 16 organisms, check in the model.organisms or
in our manuscript that yours isn't one of them, or highly related first. If so
uses these and make sure that the gene names can be easily mapped by
scdataloader's preprocess function.

If not, scPRINT-2 can be used on other organisms that are not part of its
training set, for this have a look at
[this notebook](notebooks/scPRINT-2-repro-notebooks/unknown_species_classification.ipynb).
You will also need to compute
[gene embeddings](notebooks/generate_gene_embeddings.ipynb) and
[gene locations](notebooks/genelocs.ipynb) for your organism's genetic data.
Have a look at both notebooks.

If you want to use scPRINT-2 on very different organisms than what it was
trained on, you might need to then apply some finetuning, have a look at the
[finetuning notebook](notebooks/scPRINT-2-repro-notebooks/fine_tuning_cross_species_emb_mmd.ipynb)
too.

### How long does scPRINT-2 take? What kind of resources do I need? (or in alternative: can I run scPRINT-2 locally?)

Please look at our
[manuscript table 1 and supplementary Table 1 - 2](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1)
to know more about computational ressources. But know that you will likely need
at least one high performance GPU.

### I have different scRNASeq batches. Should I integrate my data before running scPRINT-2?

scPRINT-2 takes raw count as inputs, so please don't use integrated data. Just
give the raw counts to scPRINT-2 and it will take care of the rest. For better
results you can apply some finetuning of scPRINT-2 on your batches to better
integrate them. See the
[finetuning notebook](notebooks/scPRINT-2-repro-notebooks/fine_tuning_cross_species_emb_mmd.ipynb.ipynb).
You can replace the cross-species MMD loss with a cross-batch MMD loss.

### I have new labels for my data that scPRINT-2 doesn't predict, how can I fine-tune it to predict them?

First have a look at scPRINT-2's inference capabilities and checkout the
finetuning notebooks.

In your case, what you will need to do is to reuse the finetuning notebook but
also update the output layers of the classifier to predict your new labels. You
can do this by changing the number of output classes in the classifier head to
match the number of new labels you have. You will also need to update the
scPRINT-2's `mat_labels_hierarchy` attribute to include your new labels and
their relationships if they are hierarchical and you want this to happen,
otherwise update it with an empty vector.

Make sure also to update the `label_decoders` attribute in the model to include
at the right index your new label decoder / classifier, the name of your new
labels.

Then you can proceed with the finetuning as usual, using your dataset with the
new labels in the `obs` of your anndata. (I am sure chatgpt can help you with it
too)

### where to find the input gene embeddings?

If you think you need the gene embeddings file for loading the model from a
checkpoint, you don't need to recompute them, as the embeddings are also stored
in the model weights. You just need to load the weights like this:

```python
model = scPRINT2.load_from_checkpoint(
    '../../data/temp/last.ckpt',
    precpt_gene_emb=None,
)
```

But if you want to, you can also recreate the gene embedding file through
[this notebook](notebooks/generate_gene_embeddings.ipynb). Just call the
functions, and it should recreate the file itself.

The file itself is also available on
[hugging face](https://huggingface.co/jkobject/scPRINT-2/tree/main)

/!\ Please understand that what I mean by gene embedding is the immutable input
gene embeddings encoding the gene name. scPRINT-2 directly takes raw counts as
input and takes care of doing the embedding on the fly. (it does similarly for a
gene's location in the genome).

## Development

### dev install

If you want to use the latest version of scPRINT-2 and work on the code yourself
use `git clone` and `pip -e` instead of `pip install`.

```bash
git clone https://github.com/cantinilab/scPRINT-2
git clone https://github.com/jkobject/scDataLoader
git clone https://github.com/cantinilab/GRnnData
git clone https://github.com/jkobject/benGRN
pip install -e scprint2[dev]
pip install -e scDataLoader[dev]
pip install -e GRnnData[dev]
pip install -e benGRN[dev]
```

### Reproducibility

**To reproduce the paper please use the version / tag `1.6.4` and you will have
to git clone the repo to have access to all the pre-training functionalities!**

⚠️ When re-training scPRINT-2 from scratch, by default, every N epoch, the
`test()` function will be called `. It is using a predownloadedtest datasets
paths (see https://github.com/cantinilab/scPRINT-2/issues/12). Replace them with
your own paths you want to use these test functions. They are also made
available on hf.co: https://huggingface.co/jkobject/scPRINT-2/tree/main

### Building the Docker Image

To build the Docker image from the provided `Dockerfile`, run the following
command from the root directory of this repository:

```bash
docker build -t scprint2:latest -f Dockerfile .
```

### Pulling the Docker Image from Docker Hub

If you don't want to build the image yourself, you can pull it directly from
Docker Hub:

```bash
docker pull jkobject/scprint2:1.0.0
docker tag jkobject/scprint2:1.0.0 scprint2:latest
```

### Running the Docker Container

Once you have the image (either by building it or pulling it), you can start a
container with:

```bash
docker run --gpus all --rm -it scprint2:latest bash
```

Please note: When running the Docker container, ensure you mount any necessary
folders using the -v option to access them inside the container.

### Participate

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Read the
[training runs](https://wandb.ai/ml4ig/scprint_ablation/reports/scPRINT-2-additive-benchmark--VmlldzoxNTIyOTYwNA?accessToken=0mzwwu64py309mds6zzbgcxllrgcdnd10laivhs3ykh9pqmbs0wxutcu60py2bld)
document to know more about how pre-training was performed and the its behavior.

code coverage is not right as I am using the command line interface for
now. >50% of the code is covered by my current unit test.

Acknowledgement:
[python template](https://github.com/rochacbruno/python-project-template)
[laminDB](https://lamin.ai/) [lightning](https://lightning.ai/)

Created by Jérémie Kalfon
