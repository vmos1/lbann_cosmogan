import numpy as np
from os.path import abspath, dirname, join
import yaml
import os

### Get input file name and Number of images from config file

config_file=os.environ['config_file']
with open(config_file) as f:
    config_dict= yaml.load(f, Loader=yaml.SafeLoader)
data_file=config_dict['parameters']['ip_data_fname']
num_imgs=config_dict['parameters']['num_imgs']

### Load data
samples = np.load(data_file, allow_pickle=True)[:num_imgs]

print("Data file name: ",data_file)
print("Sample shape",samples.shape)

dims = 128*128*1

# Sample access functions
def f_get_sample(index):
    '''
    Used by the main code to get samples.
    '''
    sample = samples[index].flatten()
    #normalization here if unnormalized
    return sample

def f_num_samples():
    return samples.shape[0]

def f_sample_dims():
    return [dims]
