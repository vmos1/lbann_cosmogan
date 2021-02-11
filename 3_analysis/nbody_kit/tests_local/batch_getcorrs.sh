#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --output=slurm-%x-%j.out
#SBATCH --image=nugent68/bccp:1.2
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=00:30:00
#SBATCH --job-name=3ptfnc_analysis

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME

code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/3_analysis/nbody_kit/tests_local'

shifter python $code_dir/parallel_get3pct.py --n 6 --slice 32 --start_i 0 --end_i 1000

echo "--end date" `date` `date +%s`
