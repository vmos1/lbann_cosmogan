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
#SBATCH --time=00:30:00
#SBATCH --job-name=exagan

### Initial setup
module purge
module load modules/3.2.11.4 gcc/8.2.0 cuda/10.2.89 mvapich2/2.3.2 cmake/3.14.4 python3/3.7-anaconda-2019.10

module load esslurm
#module load python3/3.7-anaconda-2019.10
##module use /global/cfs/cdirs/m3363/lbann/tom_lbann_install/etc/modulefiles
##module load gcc/8.2.0
##module load lbann

###unset MKL_THREADING_LAYER
export MKL_THREADING_LAYER=GNU

#export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/spack.git
#export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/tom_spack
#$SPACK_ROOT/share/spack/setup-env.sh


### Brian's installation
export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/spack.git; . $SPACK_ROOT/share/spack/setup-env.sh
spack env activate -p lbann-dev-skylake_avx512

module use /global/cfs/cdirs/m3363/lbann/lbann.git/build/gnu.Release.cgpu.nersc.gov/install/etc/modulefiles/
module load lbann


export MV2_ENABLE_AFFINITY=0
export MV2_USE_CUDA=1
export IBV_FORK_SAFE=1

### Run the main code
code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/1_train/main_code/'

##code_dir=/global/cfs/cdirs/m3363/vayyar/cosmogan_data/copy_of_code/main_code/            

python $code_dir\test_exagan.py --seed 5872 --nodes 1 --procs 8 --epochs 1 -dr /global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_data/20200629_090438_batchsize_512/chkpt/trainer0/sgd.shared.training.epoch.43.step.17160/model0/
###python $code_dir\test_exagan.py --seed 58772 --nodes 1 --procs 8 --epochs 1 -dr /global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_data/20200617_062906_batchsize_1028_exagan/models/trainer0/model0/

