import numpy as np
from os.path import abspath, dirname, join
#import google.protobuf.text_format as txtf ## Edit vpa: Feb 20, 2020

# Data paths
#data_dir = '/p/lustre2/brainusr/datasets/cosmoflow/norm_train200K.npy'
data_dir = '/global/project/projectdirs/dasrepo/vpa/cosmogan/data/raw_data/raw_train.npy'

samples = np.load(data_dir, allow_pickle=True)
samples = samples.transpose(0,3,1,2)


dims = 128*128*1

# Sample access functions
def get_sample(index):
    sample = samples[index].flatten()
    #normalization here if unnormalized
    return sample

def num_samples():
    return samples.shape[0]

def sample_dims():
    return [dims]

