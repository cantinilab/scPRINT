#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --signal=SIGUSR1@180
#SBATCH --requeue

# run script from above
ulimit -c 0     # no core files
echo "Running scprint2 $1"

# If a second parameter is provided, consider it a git commit hash and checkout
if [ -n "$2" ]; then
    echo "Checking out git commit $2"
    git checkout "$2" || { echo "Failed to checkout commit $2"; exit 1; }
fi

module load cuda/12.2
export TRITON_CACHE_DIR=$TMPDIR/triton_cache
mkdir -p $TRITON_CACHE_DIR
eval "srun scprint2 $1" --trainer.default_root_dir ./$SLURM_JOB_ID
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
