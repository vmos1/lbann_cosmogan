#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=gpu
#SBATCH --account=m3363
#SBATCH --ntasks-per-node=8 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-task=1
#SBATCH --time=01:59:00
#SBATCH --job-name=exagan

### Initial setup
module load esslurm
module use /global/cfs/cdirs/m3363/lbann/tom_lbann_install/etc/modulefiles
module load gcc/8.2.0
module load lbann

#unset MKL_THREADING_LAYER
export MKL_THREADING_LAYER=GNU

#export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/spack.git
export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/tom_spack
$SPACK_ROOT/share/spack/setup-env.sh
export MV2_ENABLE_AFFINITY=0
export MV2_USE_CUDA=1
export IBV_FORK_SAFE=1

### Run the main code
code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/2_create_images/'

python $code_dir\run_GAN.py --nodes 1 --procs 1 --epochs 1 

