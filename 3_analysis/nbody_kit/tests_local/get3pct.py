### Code to compute and store the 3 point function for a given image
#### Dec 3, 2020

import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import InterpolatedUnivariateSpline

from nbodykit.lab import *
from nbodykit import setup_logging, style

import time
import argparse
import os


def f_make_catalog_2d(img):
    ''' Make catalog for 2d images'''
    x=np.arange(img.shape[0]) 
    y=np.arange(img.shape[1])

    coord=np.array([(i,j,0) for i in x for j in y]) ## Form is (x,y,0)

    ip_dict={}
    ip_dict['Position'] = coord
    ip_dict['Mass'] = img.flatten()

    catalog=ArrayCatalog(ip_dict)
    
    return catalog

def f_make_catalog_3d(img):
    
    x=np.arange(img.shape[0]) 
    y=np.arange(img.shape[1])
    z=np.arange(img.shape[2])

    coord=np.array([(i,j,k) for i in x for j in y for k in z]) ## Form is (x,y,z)

    ip_dict={}
    ip_dict['Position'] = coord
    ip_dict['Mass'] = img.flatten()

    catalog=ArrayCatalog(ip_dict)
    
    return catalog


def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--slice','-s', type=int, default=128, help='Size of image to slice')
    add_arg('--img_index','-idx', type=int, default=0, help='Index of image array to use')

    add_arg('--fname','-f',  type=str,default='/mnt/laptop/data/2d_images_50.npy', help='File name with images')

    
if __name__=="__main__":
#     fname='/mnt/laptop/data/2d_images_50.npy'
    fname='data/2d_images_50.npy'
    slice_idx=16
    img_index=5 # Pick specific image from index
    num_corrs=4
    
    args=f_parse_args()
    print(args)
#     slice_idx=args.slice
#     img_index=args.img_index
#     fname=args.fname
    
    a1=np.load(fname)[:,:,:,0]

    if len(a1.shape)-1==2: 
        print("Image is 2d")
        img=a1[img_index,:slice_idx,:slice_idx]
        cat1=f_make_catalog_2d(img)
    elif len(a1.shape)-1==3:
        print("Image is 3d")
        img=a1[img_index,:slice_idx,:slice_idx,:slice_idx]
        cat1=f_make_catalog_3d(img)

    print(img.shape)

    ### Create catalog
    cat1=f_make_catalog_2d(img)

    ## compute 3 ptfnc
    t1=time.time()
    obj1=SimulationBox3PCF(cat1,list(range(num_corrs)),edges=np.arange(1,10,1),BoxSize=20,weight='Mass')
    t2=time.time()
    op1=obj1.run()
    t3=time.time()
    print(t3-t2,t2-t1)

    ## Save correlators
    data_dir='data_stored_results/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in op1.variables:
        arr1=op1[i]
        fname='img'+str(img_index)+i+'.npy'
        np.savetxt(data_dir+fname,arr1)
    
    


