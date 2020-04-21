import numpy as np
from os.path import abspath, dirname, join
#import google.protobuf.text_format as txtf 

# Data paths
#data_dir = '/global/project/projectdirs/dasrepo/vpa/cosmogan/data/raw_data/raw_train.npy'
# data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/raw_train.npy'
data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/large_dataset_train.npy'
#data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/very_large_dataset_train.npy'

samples = np.load(data_dir, allow_pickle=True)
samples = samples.transpose(0,3,1,2)

### Normalization
###### Transformation functions
def f_transform(x,scale=4.0):
    return np.divide(2.*x, x + scale) - 1.
def f_invtransform(s,scale=4.0):
    return scale*np.divide(1. + s, 1. - s)

## Transform the images 
samples=f_transform(samples,scale=4.0)
#print("Sample shape",sample.shape)
## Check that the transformation function is working correctly
#hist1, bin_edges1 = np.histogram(sample.flatten(), bins=25,density=False)
#print(hist1,bin_edges1)
print(samples.shape)

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
