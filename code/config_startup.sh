module use /global/homes/v/vanessen/ws/lbann_install/etc/modulefiles
module load lbann/0.99.0
export SPACK_ROOT=/global/cfs/cdirs/m3363/lbann/spack.git
source $SPACK_ROOT/share/spack/setup-env.sh
export MV2_ENABLE_AFFINITY=0
export MV2_USE_CUDA=1

