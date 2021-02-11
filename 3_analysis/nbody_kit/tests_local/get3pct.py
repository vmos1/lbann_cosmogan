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
    add_arg('--ncorrs','-n', type=int, default=4, help='Number of correlators to use')
    add_arg('--fname','-f',  type=str,default='/mnt/laptop/data/2d_images_50.npy', help='File name with images')
#    add_arg('--idx_lst','-i', nargs='*', type = int, default=[0], help='List of slices of input file to use.')
    add_arg('--start_i','-si', type = int, default=0, help='Start index of image input file')
    add_arg('--end_i','-ei', type = int, default=1, help='End index of image input file')
    return parser.parse_args()

if __name__=="__main__":
    
    args=f_parse_args()
    print(args)
    slice_idx=args.slice
    num_corrs=args.ncorrs
    fname=args.fname
    #     fname='/mnt/laptop/data/2d_images_50.npy'
    fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/test_data_for_3ptfunc/2d_images_3000.npy'
    
    ## Load image file
    print("Reading file from ",fname)
    a1=np.load(fname)[:,:,:,0]
    
    idx_lst=np.arange(args.start_i,args.end_i)
    print(idx_lst)
    for img_index in idx_lst:
        if len(a1.shape)-1==2: 
    #         print("Image is 2d")
            img=a1[img_index,:slice_idx,:slice_idx]
            cat1=f_make_catalog_2d(img)
        elif len(a1.shape)-1==3:
    #         print("Image is 3d")
            img=a1[img_index,:slice_idx,:slice_idx,:slice_idx]
            cat1=f_make_catalog_3d(img)

        print(img.shape)

        ## compute 3 ptfnc
        t1=time.time()
        obj1=SimulationBox3PCF(cat1,list(range(num_corrs)),edges=np.arange(1,20,1),BoxSize=40,weight='Mass')
        t2=time.time()
        op1=obj1.run()
        t3=time.time()
        print("Times for index {0}: {1},{2}".format(img_index,t3-t2,t2-t1))

        ## Save correlators
        data_dir='data_stored_results/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        ### Extract and Save correlators as 3D array to file
        corr_list=[]
        for i in op1.variables:  
            corr_list.append(op1[i]) 
        
        arr=np.array(corr_list)
        print(arr.shape)
        fname='img_'+str(img_index)+'-corr'+'.npy'
        np.save(data_dir+fname,arr)
    
    


