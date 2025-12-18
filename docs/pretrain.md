# Pre-training scPRINT-2

scPRINT-2 is a large model that can be pre-trained on a large dataset of single
cell data.

This pre-training is quite efficient for scPRINT-2 and smaller models can be
pretrained on any hardware with a 20GB NVIDIA GPU.

## Setup of the database

To perform pretraining you will need a large dataset. We recommend using the
[laminDB](https://lamin.ai) to assemble such a large database of dataset and to
use our [scdataloader](https://github.com/jkobject/scDataLoader) package to
perform the data loading to the model.

In addition, you will need to preprocess your datasets. To make sure that the
fields are all here, the genes are in the right format, the raw counts are used,
etc... We recommend using the Preprocessor class of the `scdataloader` package.

Moreover scdataloader works with a set of ontologies. To install these, use the
function `populate_my_ontologies` from the `scdataloader` package.

If you do not have your own database of anndatas, we recommend the
[cellxgene database](https://cellxgene.cziscience.com/) and our associated
helper function to download and preprocess all of cellxgene in a single command
with `scdataloader`.

Finally you might want to generate gene embeddings to use with scPRINT-2 instead
of learning these tokens from scratch. For this you can use the
[gene_embedders](tokenizers.md) module of scPRINT-2, which usage is detailed in
the `notebooks/generate_gene_embeddings.ipynb` notebook and also gene locations
using the additional notebook `notebooks/genelocs.ipynb`

## Pre-training

to pretrain scPRINT-2 we strongly recommend using command line as it can take
multiple days (and using some HPC plateform like slurm or others). If on your
own machine, use something like `screen` at least 😉.

Most of the pre-training usage follows from
[pytorch lightning](https://lightning.ai/docs/pytorch/stable/levels/intermediate.html)
with `scprint-2 fit` you will launch a training run. It will populate both the
datamodule (see `scdataloader`), the model (see `model.py`), the trainer (see
`pytorch lightning`) and the various callbacks.

But you might want to use additional parameters. For this, you can use the
`config` folder and the `yaml` files in it. These files are used to store the
main hyperparameters of the model and the training scheme.

More hyperparameters are given to the scPRINT-2 model via a Trainer callback I
created (see `trainer/trainer.py`). This is used to specify parameters to
scPRINT-2 that are used solely during training and are not part of the model
definition itself, like lr, schedulers, optimizers, etc.. I use a callback as it
is how pytorch lightning requires us to send training parameters to the model.

Thus a full command line to train scPRINT-2 on a slurm cluster might look like
this:

```bash
conda activate scprint2
### slurm level stuff
module load cuda/12.2
sbatch
  -p gpu #gpu partition
  -q gpu #gpu queue
  --gres=gpu:A40:4,gmem:40G #gpu type (4 A40 with 40GB of GPU mem)
  --cpus-per-task 16
  --mem-per-gpu 90G #RAM per GPU
  --ntasks-per-node=1
####
  # actuall scprint-2 command
  slurm/submit.sh 'fit
    --config config/base_v1.yml #base config file (see below)
    --config config/pretrain_medium.yml #the differences when training a large model
    --model.nhead 8 # changing this parameter from the large model directly in command line (cannot do 4 heads of 128dim with A40 GPUs...)
    --scprint_training.name o2uniqsx #an id for the model (not needed but useful)
    '
```

with the base yaml file containing:

```yaml
# general params
project: scprint_scale #project name for saving data and wandb
seed_everything: 42
ckpt_path: null #we don't have a checkpoint weights as we train from scratch
set_float32_matmul_precision: True
wandblog: all #we use wandb here
log_freq: 200
log_graph: True
trainer: #training level params
  precision: 16-mixed #we use mixed precision 16bit for training
  gradient_clip_val: 100 #needed
  log_every_n_steps: 100
  ....
  logger: #we can add multiple loggers (see below)
    - class_path: lightning.pytorch.loggers.WandbLogger
  callbacks: #you can create your own callback and add it here or use lightning's callbacks
    - class_path: lightning.pytorch.callbacks.StochasticWeightAveraging
      init_args:
        swa_lrs: 0.03
    ...
model: # model params
  dropout: 0.1
  transformer: normal #flashattention is used
  ...
data: #datamodule params
  organisms: #we will use these 2 species
    - NCBITaxon:9606
    - NCBITaxon:10090
  gene_position_tolerance: 10_000 #gene location: if genes are closer than 10kb, they are considered as the same location
  gene_embeddings: ./data/main/gene_embeddings/ #the embeddings of genes (see above  )
  collection_name: all no zhang13M # the name of the laminDB collection we will use
  how: random expr # how we collate the expression data (here random expressed genes)
  max_len: 2200 #how many genes we use in the model context during training
  weight_scaler: 50 #how do we scale the weighted random sampling procedure (see our manuscript)
  ...
```

We use wanDB in our case, however scPRINT-2 and pytorch lightning support a
breadth of logging tools:
[see loggers](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers).

We use slurm in our usecase here but scPRINT-2 and pytorch lightning has been
made to work in a breadth of environments
[e.g.](https://lightning.ai/docs/pytorch/stable/levels/intermediate_level_7.html).
