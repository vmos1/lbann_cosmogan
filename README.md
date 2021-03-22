# lbann_cosmogan

**Project aim**: \
Build Generative Adversarial Neural networks to produce images of matter distribution in the universe for different sets of cosmological parameters.
Dataset consisits of N-body cosmology simulations.
The code is built using the LBANN framework.

This has been developed using codes in [keras](https://github.com/pzharrington/ExaGAN) and [tensorflow](https://github.com/pzharrington/ExaGANv2).
An earlier version of the code in lbann can be found [here](https://github.com/LLNL/lbann/tree/develop/applications/physics/cosmology/ExaGAN).

Data \
The original data can be extracted from [here](https://portal.nersc.gov/project/m3363/).
A subset can be accessed [here](https://portal.nersc.gov/project/m3363/cosmoUniverse_2019_05_4parE/22309462/).

Description of different folders in repo
- **0a_data_preprocessing**:
Codes for obtaining slices from the original hdf5 files.
- **0b_view_data**:
Codes to view sample images and explore normalization functions of input images.
- **1_train**: 
Contains the scripts to train the model and batch scripts to run it.
- **3_analysis**:
Contains scripts that analyze the produced images.


Each folder contains a jupyter notebook to quickly test the code, a folder with the full code, a launch script to run the code on cori GPUs at NERSC, a script to perform post-run computation of metrics for different stored images and a folder with analysis codes to inspect the performance of the code. Below is an example for 2D GAN
| Name | Description |
| --- | ---|
| [1_train/main_code/](https://github.com/vmos1/lbann_cosmogan/tree/master/1_train/main_code) | Folder containing main training and inference code |
| [1_train/run_scripts/launch_lbann_train.ipynb](https://github.com/vmos1/lbann_cosmogan/blob/master/1_train/run_scripts/launch_lbann_train.ipynb) |  Notebook that launches script to run training |
| [1_train/run_scripts/launch_lbann_compute_chisqr.ipynb](https://github.com/vmos1/lbann_cosmogan/blob/master/1_train/run_scripts/launch_lbann_compute_chisqr.ipynb)| Notebook that launches script to run post-run metric computation |
|[3_analysis/3a_analysis_pandas.py ](https://github.com/vmos1/lbann_cosmogan/blob/master/3_analysis/3a_analysis_pandas.py) | Notebook to analyze GAN results and view best epoch-steps |
