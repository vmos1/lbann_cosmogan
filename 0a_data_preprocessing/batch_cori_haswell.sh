#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=regular
#SBATCH --job-name=lbann_data_extraction_haswell
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=haswell
#SBATCH --account=m3363
#################

conda activate v_py3
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "--start date" `date` `date +%s`
srun -n 1 -c 30 python 1_slice_parallel.py  --cores 30 -p full_1_ --mode full
###srun -n 1 -c 30 python 1_slice_parallel.py  --cores 30 -p full_with_smoothing_1 --smoothing --mode full
conda deactivate
echo "--end date" `date` `date +%s`
