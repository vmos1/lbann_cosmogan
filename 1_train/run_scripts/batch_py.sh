#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=gpu
#SBATCH --account=nstaff
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -c 10
#SBATCH --time=01:59:00

#SBATCH --job-name=exagan

### Initial setup
module load esslurm
module use /global/homes/v/vanessen/ws/lbann_install/etc/modulefiles
module load lbann/0.99.0
export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/spack.git
$SPACK_ROOT/share/spack/setup-env.sh
export MV2_ENABLE_AFFINITY=0
export MV2_USE_CUDA=1

### Run the main code
python ../main_code/train_exagan.py --nodes 1 --procs 1 --epochs 50

