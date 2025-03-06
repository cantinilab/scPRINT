> ℹ️ main place where scprint is built and maintained

# scPRINT: Large Cell Model for scRNAseq data

[![codecov](https://codecov.io/gh/cantinilab/scPRINT/branch/main/graph/badge.svg?token=GRnnData_token_here)](https://codecov.io/gh/cantinilab/scPRINT)
[![CI](https://github.com/cantinilab/scPRINT/actions/workflows/main.yml/badge.svg)](https://github.com/cantinilab/scPRINT/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/scprint.svg)](https://badge.fury.io/py/scprint)
[![Downloads](https://pepy.tech/badge/scprint)](https://pepy.tech/project/scprint)
[![Downloads](https://pepy.tech/badge/scprint/month)](https://pepy.tech/project/scprint)
[![Downloads](https://pepy.tech/badge/scprint/week)](https://pepy.tech/project/scprint)
[![GitHub issues](https://img.shields.io/github/issues/cantinilab/scPRINT)](https://img.shields.io/github/issues/cantinilab/scPRINT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14749466.svg)](https://doi.org/10.5281/zenodo.14749466)
[![hugging face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/jkobject/scPRINT)

![logo](docs/logo.png)

scPRINT is a large transformer model built for the inference of gene networks (connections between genes explaining the cell's expression profile) from scRNAseq data.

It uses novel encoding and decoding of the cell expression profile and new pre-training methodologies to learn a cell model.

scPRINT can be used to perform the following analyses:

- __expression denoising__: increase the resolution of your scRNAseq data
- __cell embedding__: generate a low-dimensional representation of your dataset
- __label prediction__: predict the cell type, disease, sequencer, sex, and ethnicity of your cells
- __gene network inference__: generate a gene network from any cell or cell cluster in your scRNAseq dataset

[Read the manuscript!](https://www.biorxiv.org/content/10.1101/2024.07.29.605556v1) if you would like to know more about scPRINT. Have a look at some of my [X-plainers](https://twitter.com/jkobject). 

![figure1](docs/figure1.png)

🎊 test scPRINT and scDataloader on this simple [google collab](https://colab.research.google.com/drive/1CacoQDAwJn86tq2sBhUoZ6M-xAqsYFDI#scrollTo=Lb4E9IhQ7NK8)

## Table of Contents

- [scPRINT: Large Cell Model for scRNAseq data](#scprint-large-cell-model-for-scrnaseq-data)
  - [Table of Contents](#table-of-contents)
  - [Install `scPRINT`](#install-scprint)
    - [Test scPRINT on google colab!](#test-scprint-on-google-colab)
    - [Use scPRINT in superbio.ai!](#use-scprint-in-superbioai)
    - [lamin.ai](#laminai)
    - [install](#install)
    - [pytorch and GPUs](#pytorch-and-gpus)
    - [dev install](#dev-install)
  - [Reproducibility](#reproducibility)
  - [Usage](#usage)
    - [scPRINT's basic commands](#scprints-basic-commands)
    - [Notes on GPU/CPU usage with triton](#notes-on-gpucpu-usage-with-triton)
    - [Simple tests:](#simple-tests)
  - [FAQ](#faq)
    - [I want to generate gene networks from scRNAseq data:](#i-want-to-generate-gene-networks-from-scrnaseq-data)
    - [I want to generate cell embeddings and cell label predictions from scRNAseq data:](#i-want-to-generate-cell-embeddings-and-cell-label-predictions-from-scrnaseq-data)
    - [I want to denoise my scRNAseq dataset:](#i-want-to-denoise-my-scrnaseq-dataset)
    - [I want to generate an atlas-level embedding](#i-want-to-generate-an-atlas-level-embedding)
    - [I need to generate gene tokens using pLLMs](#i-need-to-generate-gene-tokens-using-pllms)
    - [I want to pre-train scPRINT from scratch on my own data](#i-want-to-pre-train-scprint-from-scratch-on-my-own-data)
    - [how can I find if scPRINT was trained on my data?](#how-can-i-find-if-scprint-was-trained-on-my-data)
    - [can I use scPRINT on other organisms rather than human?](#can-i-use-scprint-on-other-organisms-rather-than-human)
    - [how long does scPRINT takes? what kind of resources do I need? (or in alternative: can i run scPRINT locally?)](#how-long-does-scprint-takes-what-kind-of-resources-do-i-need-or-in-alternative-can-i-run-scprint-locally)
    - [I have different scRNASeq batches. Should I integrate my data before running scPRINT?](#i-have-different-scrnaseq-batches-should-i-integrate-my-data-before-running-scprint)
    - [where to find the gene embeddings?](#where-to-find-the-gene-embeddings)
  - [Documentation](#documentation)
  - [Model Weights](#model-weights)
  - [Docker](#docker)
    - [Building the Docker Image](#building-the-docker-image)
    - [Pulling the Docker Image from Docker Hub](#pulling-the-docker-image-from-docker-hub)
    - [Running the Docker Container](#running-the-docker-container)
  - [Development](#development)
  - [Work in progress (PR welcomed):](#work-in-progress-pr-welcomed)


## Install `scPRINT`

For the moment scPRINT has been tested on MacOS and Linux (Ubuntu 20.04) with Python 3.10. Its instalation takes on average 10 minutes.

If you want to be using flashattention2, know that it only supports triton 2.0 MLIR's version and torch==2.0.0 for now.

### Test scPRINT on google colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CacoQDAwJn86tq2sBhUoZ6M-xAqsYFDI#scrollTo=Vj73HINSzKHL)

### Use scPRINT in superbio.ai!

[HERE](https://app.superbio.ai/apps/67333115ed44f27eb717cf84)

### lamin.ai

To use scPRINT, you will need to use [lamin.ai](https://lamin.ai/). This is needed to load biological informations like genes, cell types, organisms etc...

### install

To start you will need to do:

```bash
uv venv -n <env-name> python==3.10 #scprint might work with python >3.10, but it is not tested
#one of
uv pip install scprint # OR
uv pip install scprint[dev] # for the dev dependencies (building etc..) OR
uv pip install scprint[flash] # to use flashattention2 with triton: only if you have a compatible gpu (e.g. not available for apple GPUs for now, see https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility)
#OR pip install scPRINT[dev,flash]

lamin init --storage ./testdb --name test --schema bionty
```

if you start with lamin and had to do a `lamin init`, you will also need to populate your ontologies. This is because scPRINT is using ontologies to define its cell types, diseases, sexes, ethnicities, etc.

you can do it manually or with our function:

```python
from scdataloader.utils import populate_my_ontology

populate_my_ontology() #to populate everything (recommended) (can take 2-10mns)

populate_my_ontology( #the minimum for scprint to run some inferences (denoising, grn inference)
organisms: List[str] = ["NCBITaxon:10090", "NCBITaxon:9606"],
    sex: List[str] = ["PATO:0000384", "PATO:0000383"],
    celltypes = None,
    ethnicities = None,
    assays = None,
    tissues = None,
    diseases = None,
    dev_stages = None,
)
```

We make use of some additional packages we developed alongside scPRint.

Please refer to their documentation for more information:

- [scDataLoader](https://github.com/jkobject/scDataLoader): a dataloader for training large cell models.
- [GRnnData](https://github.com/cantinilab/GRnnData): a package to work with gene networks from single cell data.
- [benGRN](https://github.com/jkobject/benGRN): a package to benchmark gene network inference methods from single cell data.

### pytorch and GPUs

scPRINT can run on machines without GPUs, but it will be slow. It is highly recommended to use a GPU for inference.

Once you have a GPU, and installed the required drivers, you might need to install a specific version of pytorch that is compatible with your drivers (e.g. nvidia 550 drivers will lead to a nvidia toolkit 11.7 or 11.8 which might mean you need to re-install a different flavor of pytorch for things to work. e.g. using the command:
`pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118` on my case on linux
 ).

I was able to test it with nvidia 11.7, 11.8, 12.2.

### dev install

If you want to use the latest version of scPRINT and work on the code yourself use `git clone` and `pip -e` instead of `pip install`.

```bash
git clone https://github.com/cantinilab/scPRINT
git clone https://github.com/jkobject/scDataLoader
git clone https://github.com/cantinilab/GRnnData
git clone https://github.com/jkobject/benGRN
pip install -e scPRINT[dev]
pip install -e scDataLoader[dev]
pip install -e GRnnData[dev]
pip install -e benGRN[dev]
```

## Reproducibility 

__To reproduce the paper please use the version / tag `1.6.4` and you will have to git clone the repo to have access to all the pre-training functionalities!__

⚠️ the `test()` function of the pretraining runs (by default run every N epochs) is using a 2 hardcoded test datasets paths (see https://github.com/cantinilab/scPRINT/issues/12). Replace them with your own if you want to use the test functions. They are also made available on hf.co: https://huggingface.co/jkobject/scPRINT/tree/main

## Usage

### scPRINT's basic commands

This is the most minimal example of how scPRINT works:

```py
from lightning.pytorch import Trainer
from scprint import scPrint
from scdataloader import DataModule

datamodule = DataModule(...)
model = scPrint(...)
# to train / fit / test the model
trainer = Trainer(...)
trainer.fit(model, datamodule=datamodule)
# to do predictions Denoiser, Embedder, GNInfer
denoiser = Denoiser(...)
adata = sc.read_h5ad(...)
denoiser(model, adata=adata)
...
```

or, from a bash command line

```bash
$ scprint fit/train/predict/test/denoise/embed/gninfer --config config/[medium|large|vlarge] ...
```

find out more about the commands by running `scprint --help` or `scprint [command] --help`.

more examples of using the command line are available in the [docs](./docs/usage.md).

### Notes on GPU/CPU usage with triton

If you do not have [triton](https://triton-lang.org/main/python-api/triton.html) installed you will not be able to take advantage of GPU acceleration, but you can still use the model on the CPU.

In that case, if loading from a checkpoint that was trained with flashattention, you will need to specify `transformer="normal"` in the `load_from_checkpoint` function like so:

```python
model = scPrint.load_from_checkpoint(
    '../data/temp/last.ckpt', precpt_gene_emb=None,
    transformer="normal")
```

### Simple tests:

An instalation of scPRINT and a simple test of the denoiser is performed during each commit to the main branch with a [Github action](https://github.com/cantinilab/scPRINT/actions) and [pytest workflow](.github/workflows/main.yml). It also provides an expected runtime for the installation and run of scPRINT.

We now explore the different usages of scPRINT:

## FAQ

### I want to generate gene networks from scRNAseq data:

-> Refer to the section . gene network inference in [this notebook](./docs/notebooks/cancer_usecase.ipynb#).

-> More examples in this notebook [./notebooks/assessments/bench_omni.ipynb](./notebooks/bench_omni.ipynb).

### I want to generate cell embeddings and cell label predictions from scRNAseq data:

-> Refer to the embeddings and cell annotations section in [this notebook](./docs/notebooks/cancer_usecase.ipynb#).

### I want to denoise my scRNAseq dataset:

-> Refer to the Denoising of B-cell section in [this notebook](./docs/notebooks/cancer_usecase.ipynb).

-> More example in our benchmark notebook [./notebooks/assessments/bench_denoising.ipynb](./notebooks/bench_denoising.ipynb).

### I want to generate an atlas-level embedding

-> Refer to the notebook [nice_umap.ipynb](./figures/nice_umap.ipynb).

### I need to generate gene tokens using pLLMs

To run scPRINT, you can use the option to define the gene tokens using protein language model embeddings of genes. This is done by providing the path to a parquet file of the precomputed set of embeddings for each gene name to scPRINT via "precpt_gene_emb"

-> To generate this file please refer to the notebook [generate_gene_embeddings](notebooks/generate_gene_embeddings.ipynb).

### I want to pre-train scPRINT from scratch on my own data

-> Refer to the documentation page [pretrain scprint](docs/pretrain.md)

### how can I find if scPRINT was trained on my data?

If your data is available in cellxgene, scPRINT was likely trained on it. However some cells, datasets were dropped due to low quality data and some were randomly removed to be part of the validation / test sets.

### can I use scPRINT on other organisms rather than human?

scPRINT has been pretrained on both humans and mouse, and can be used on any organism with a similar gene set. If you want to use scPRINT on very different organisms, you will need to generate gene embeddings for that organism and re-train scPRINT

### how long does scPRINT takes? what kind of resources do I need? (or in alternative: can i run scPRINT locally?)

please look at our supplementary tables in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.07.29.605556v1)

### I have different scRNASeq batches. Should I integrate my data before running scPRINT?

scPRINT takes raw count as inputs, so please don't use integrated data. Just give the raw counts to scPRINT and it will take care of the rest.

### where to find the gene embeddings?

If you think you need the gene embeddings file for loading the model from a checkpoint, you don't, as the embeddings are also stored in the model weights. You just need to load the weights like this:

```python
model = scPrint.load_from_checkpoint(
    '../../data/temp/last.ckpt',
    precpt_gene_emb=None,
)
```

You can also recreate the gene embedding file through [this notebook](notebooks/generate_gene_embeddings.ipynb). Just call the functions, and it should recreate the file itself.

the file itself is also available on [hugging face](https://huggingface.co/jkobject/scPRINT/tree/main)

## Documentation

For more information on usage please see the documentation in [https://www.jkobject.com/scPRINT/](https://www.jkobject.com/scPRINT/)

## Model Weights

Model weights are available on [hugging face](https://huggingface.co/jkobject/scPRINT/).

## Docker

By using the `scPRINT Docker image`, you can bypass the complexities of manual package installation, ensuring a consistent deployment environment. Included in this repository is a Dockerfile that lets you craft a container for the project; you have the choice to either build this image on your own or conveniently pull it from Docker Hub.

Make sure that you have the `docker` command line interface installed on your system.

A recommended way to install docker with the correct nvidia drivers on linux is to use this [script](https://gist.github.com/xueerchen1990/baad7baa545cb547e8633bc9e5b84786)

### Building the Docker Image

To build the Docker image from the provided `Dockerfile`, run the following command from the root directory of this repository:

```bash
docker build -t scprint:latest -f Dockerfile .
```

### Pulling the Docker Image from Docker Hub

If you don't want to build the image yourself, you can pull it directly from Docker Hub:

```bash
docker pull jkobject/scprint:1.1.3
docker tag jkobject/scprint:1.1.3 scprint:latest
```

### Running the Docker Container

Once you have the image (either by building it or pulling it), you can start a container with:

```bash
docker run --gpus all --rm -it scprint:latest bash
```

Please note: When running the Docker container, ensure you mount any necessary folders using the -v option to access them inside the container.
`
## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Read the [training runs](https://wandb.ai/ml4ig/scprint_scale/reports/scPRINT-trainings--Vmlldzo4ODIxMjgx?accessToken=80metwx7b08hhourotpskdyaxiflq700xzmzymr6scvkp69agybt79l341tv68hp) document to know more about how pre-training was performed and the its behavior.

code coverage is not right as I am using the command line interface for now. >50% of the code is covered by my current unit test.

Acknowledgement:
[python template](https://github.com/rochacbruno/python-project-template)
[laminDB](https://lamin.ai/)
[lightning](https://lightning.ai/)

## Work in progress (PR welcomed):

1. remove the triton dependencies
2. add version with additional labels (tissues, age) and organisms (mouse, zebrafish) and more datasets from cellxgene
3. version with separate transformer blocks for the encoding part of the bottleneck learning and for the cell embeddings
4. improve classifier to output uncertainties and topK predictions when unsure
5. setup latest lamindb version

Awesome Large Cell Model created by Jeremie Kalfon.
