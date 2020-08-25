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
module purge
module load modules/3.2.11.4 gcc/8.3cuda/10.2.89 mvapich2/2.3.2 cmake/3.14.4 python3/3.7-anaconda-2019.10

module load esslurm
#module load python3/3.7-anaconda-2019.10
module use /global/cfs/cdirs/m3363/lbann/tom_lbann_install/etc/modulefiles
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
code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/1_train/512_sqr/'

##code_dir=/global/cfs/cdirs/m3363/vayyar/cosmogan_data/copy_of_code/main_code/         

##python $code_dir\train_exagan.py --seed 3772 --nodes 2 --procs 4 --suffix $1 --epochs 40
python $code_dir\train_exagan.py --seed 36723723 --nodes 1 --procs 8 --suffix $1 --epochs 20 

echo "--end date" `date` `date +%s`
