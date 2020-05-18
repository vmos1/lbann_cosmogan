## Code to perform the transformation before training ## 
### Done separately to avoid memory overload issues
## May 6, 2020. Author:Venkitesh

import numpy as np

# ip_fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/very_large_dataset_train.npy'
# op_fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/pre_norm_train.npy'

ip_fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/peter_dataset/raw_train.npy'
op_fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/peter_dataset/pre_norm_train.npy'

samples = np.load(ip_fname, allow_pickle=True)
samples = samples.transpose(0,3,1,2)

### Normalization
###### Transformation functions
# def f_transform(x,scale=4.0):
#     return np.divide(2.*x, x + scale) - 1.
# def f_invtransform(s,scale=4.0):
#     return scale*np.divide(1. + s, 1. - s)

### New log-linear transformation
def f_transform(x):
    if x<=50:
        a=0.03; b=-1.0
        return a*x+b
    elif x>50: 
        a=0.5/(np.log(15000)-np.log(50))
        b=0.5-a*np.log(50)
        return a*np.log(x)+b

def f_invtransform(y):
    if y<=0.5:
        a=0.03;b=-1.0
        return (y-b)/a
    elif y>0.5: 
        a=0.5/(np.log(15000)-np.log(50))
        b=0.5-a*np.log(50)
        return np.exp((y-b)/a)

## Transform the images 
# samples_scaled=f_transform(samples,scale=4.0)
samples_scaled=np.vectorize(f_transform)(samples)

### Save to output file
print("Writing from {0} to {1}".format(ip_fname,op_fname))
np.save(op_fname,samples_scaled)