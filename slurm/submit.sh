#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --signal=SIGUSR1@180
#SBATCH --requeue

# run script from above
echo "Running scprint fit $1"
module load cuda/12.2
export TRITON_CACHE_DIR=$TMPDIR/triton_cache
mkdir -p $TRITON_CACHE_DIR
eval "srun scprint fit $1" --trainer.default_root_dir ./$SLURM_JOB_ID
if [ $? -eq 0 ]; then
    # Run completed successfully
    echo "Run completed successfully"
    exit 0
elif [ $? -eq 99 ]; then
    # Run was requeued
    echo "Run was requeued"
    exit 99
else
    # Run failed
    echo "Run failed with exit code $?"
    exit $?
fi