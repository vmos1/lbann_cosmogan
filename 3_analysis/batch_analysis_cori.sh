#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --output=slurm-%x-%j.out
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=00:15:00
#SBATCH --job-name=exagan_run_anlysis

echo "--start date" `date` `date +%s`
conda activate v_py3
code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/3_analysis/'
fldr=$1
echo $fldr
val_file='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_2_smoothing_200k/norm_1_train_val.npy'

python $code_dir\4a_analysis_pandas.py -f $fldr  -v $val_file
conda deactivate
echo "--end date" `date` `date +%s`

