import numpy as np
from os.path import abspath, dirname, join

# Data paths

data_file='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/512_square/dataset1_smoothing_single_universe/norm_1_train_val.npy'
samples = np.load(data_file, allow_pickle=True)[:25000]

# samples = samples.transpose(0,3,1,2)
# ### Normalization
# ###### Transformation functions
# def f_transform(x,scale=4.0):
#     return np.divide(2.*x, x + scale) - 1.
# def f_invtransform(s,scale=4.0):
#     return scale*np.divide(1. + s, 1. - s)
# ## Transform the images 
# samples=f_transform(samples,scale=4.0)

print("Data file name: ",data_file)
print("Sample shape",samples.shape)

dims = 512*512*1

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
