#!/bin/bash
#SBATCH --job-name=ntv3_h100
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/slurm/slurm-%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/slurm/slurm-%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH -A wbg@h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --array=0-13%8
set -e
ulimit -c 0
module load cuda/12.2.0 || true
export TRITON_CACHE_DIR=$TMPDIR/triton_cache
mkdir -p $TRITON_CACHE_DIR
export HF_HOME=/lustre/fswork/projects/rech/xeg/uat95fg/.hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TORCH_HOME=/lustre/fswork/projects/rech/xeg/uat95fg/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/lustre/fswork/projects/rech/xeg/uat95fg/.hf/hub/models--InstaDeepAI--ntv3_base_model/snapshots/0ecff3637f0d3ba5b686d1095083218157c2ca34:$PYTHONPATH
export NTV3_BATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 14 species (homo_sapiens already done by test job 1079026, arabidopsis done by 1075804_2)
SPECIES=(
  mus_musculus bos_taurus caenorhabditis_elegans callithrix_jacchus
  danio_rerio drosophila_melanogaster gallus_gallus heterocephalus_glaber_male
  macaca_mulatta oryctolagus_cuniculus ovis_aries pan_troglodytes
  sus_scrofa zea_mays
)
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID : $SP ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | head
cd /lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
srun .venv/bin/python3 scripts/generate_ntv3_embeddings.py --species $SP
echo "=== done $SP ==="
ls -lh data/main/gene_embs_ntv3/
