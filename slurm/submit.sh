#!/bin/bash

# SBATCH --cpus-per-task=24
# SBATCH --hint=nomultithread
# SBATCH --signal=SIGUSR1@180
# SBATCH --requeue

# run script from above
eval "srun scprint fit $1" --trainer.default_root_dir ./$SLURM_JOB_ID