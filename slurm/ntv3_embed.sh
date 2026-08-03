#!/bin/bash
#SBATCH --job-name=ntv3_emb
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/slurm/slurm-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/slurm/slurm-%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_p2l
#SBATCH -A wbg@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --signal=SIGUSR1@180
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
cd /lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
nvidia-smi | head -20
srun .venv/bin/python3 scripts/generate_ntv3_embeddings.py
echo "=== outputs ==="
ls -lh data/main/gene_embs_ntv3/
