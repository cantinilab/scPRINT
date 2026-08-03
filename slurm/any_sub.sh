#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --signal=SIGUSR1@180
#SBATCH --requeue
ulimit -c 0
echo "Running $1"
module load cuda/12.2.0
export TRITON_CACHE_DIR=$TMPDIR/triton_cache
mkdir -p $TRITON_CACHE_DIR
eval "srun $1"
if [ $? -eq 0 ]; then echo "Run completed successfully"; exit 0
elif [ $? -eq 99 ]; then echo "Run was requeued"; exit 99
else echo "Run failed with exit code $?"; exit $?
fi
