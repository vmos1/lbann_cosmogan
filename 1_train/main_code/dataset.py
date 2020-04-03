import numpy as np
from os.path import abspath, dirname, join
#import google.protobuf.text_format as txtf ## Edit vpa: Feb 20, 2020

# Data paths
data_dir = '/global/project/projectdirs/dasrepo/vpa/cosmogan/data/raw_data/raw_train.npy'

samples = np.load(data_dir, allow_pickle=True)
samples = samples.transpose(0,3,1,2)

### Augmenting data to make training time larger
# print(samples.shape)
# samples=np.vstack([samples,samples])

print(samples.shape)
dims = 128*128*1

# Sample access functions
def f_get_sample(index):
    '''
    Used by the main code to get samples.
    The normalization has to be included within
    '''
    
    sample = samples[index].flatten()
    
    #normalization here if unnormalized
    ### Normalization
    ###### Transformation functions
    def f_transform(x,scale=4):
        return np.divide(2.*x, x + scale) - 1.
    def f_invtransform(s,scale=4):
        return scale*np.divide(1. + s, 1. - s)
    
    ### Transform the images 
    sample=f_transform(sample,scale=4.0)
    #print("Sample shape",sample.shape)
    
    ### Check that the transformation function is working correctly
    #hist1, bin_edges1 = np.histogram(sample.flatten(), bins=25,density=False)
    #print(hist1,bin_edges1)
    
    return sample

def f_num_samples():
    return samples.shape[0]

def f_sample_dims():
    return [dims]

