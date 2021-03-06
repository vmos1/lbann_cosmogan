#!/usr/bin/env python
# coding: utf-8

# # Extract data from output files
# ### Analyze the output from a single LBANN run
### August 3, 2020

import numpy as np
import pandas as pd
import argparse

import subprocess as sp
import os
import glob
import sys

import time

from pandarallel import pandarallel

sys.path.append('/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/3_analysis')
from modules_image_analysis import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze output data from LBANN run")
    add_arg = parser.add_argument
    
    add_arg('--val_data','-v', type=str, default='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_2_smoothing_200k/norm_1_train_val.npy',help='The .npy file with input data to compare with')
    add_arg('--folder','-f', type=str,help='The full path of the folder containing the data to analyze.')
    add_arg('--cores','-c', type=int, default=64,help='Number of cores to use for parallelization')
    add_arg('--bins_type','-bin', type=str, default='uneven',help='Number of cores to use for parallelization')
   
    return parser.parse_args()

### Transformation functions for image pixel values
def f_transform(x):
    return 2.*x/(x + 4. + 1e-8) - 1.

def f_invtransform(s):
    return 4.*(1. + s)/(1. - s + 1e-8)


# ### Modules for Extraction
def f_get_sorted_df(main_dir):
    
    '''
    Module to create Dataframe with filenames for each epoch and step
    Sorts by step and epoch
    '''
    def f_get_info_from_fname(fname):
        ''' Read file and return dictionary with epoch, step'''
        dict1={}
        dict1['epoch']=np.int32(fname.split('epoch')[-1].split('.')[1])
        dict1['step']=np.int64(fname.split('step')[-1].split('.')[1].split('_')[0])
        return dict1
    
    t1=time.time()
    ### get list of file names
    fldr_loc=main_dir+'/dump_outs/trainer0/model0/'
#        keys=['train_gen','train_input','val_gen','val_input']
#     file_strg_dict={'train_gen': sgd.training*_gen_img*_output0.npy','train_input':sgd.training*_inp_img*_output0.npy','val_gen':sgd.validation*_gen_img*_output0.npy','val_input':sgd.validation*_inp_img*_output0.npy'}
    file_strg_dict={'train_gen':'sgd.training*_gen_img*_output0.npy'}
    keys=['train_gen']

    files_arr,img_arr=np.array([]),np.array([])
    for key in keys:
        print(key)
        files=glob.glob(fldr_loc+file_strg_dict[key])
        files_arr=np.append(files_arr,files)
        img_arr=np.append(img_arr,[key] *len(files))

    print('Number of files',len(files_arr))
    if len(files_arr)<1: print('No files'); raise SystemExit

    ### Create dataframe
    df_files=pd.DataFrame()
    df_files['img_type']=np.array(img_arr)
    df_files['fname']=np.array(files_arr).astype(str)

    # Create list of dictionaries
    dict1=df_files.apply(lambda row : f_get_info_from_fname(row.fname),axis=1)
    keys=dict1[0].keys() # Extract keys of dictionary
    # print(keys)
    # ### Convert list of dicts to dict of lists
    dict_list={key:[k[key] for k in dict1] for key in keys}
    # ### Add columns to Dataframe
    for key in dict_list.keys():
        df_files[key]=dict_list[key]

    df_files=df_files.sort_values(by=['img_type','epoch','step']).reset_index(drop=True) ### sort df by epoch and step
    
    t2=time.time()
    print("time for sorting",t2-t1)

    return df_files[['epoch','step','img_type','fname']]


def f_compute_hist_spect(sample,bins):
    ''' Compute pixel intensity histograms and radial spectrum for 2D arrays
    Input : Image arrays and bins
    Output: dictionary with 5 arrays : Histogram values, errors and bin centers, Spectrum values and errors.
    '''
    ### Compute pixel histogram for row
    gen_hist,gen_err,hist_bins=f_batch_histogram(sample,bins=bins,norm=True,hist_range=None)
    ### Compute spectrum for row
    spec,spec_sdev=f_compute_spectrum(sample,plot=False)

    dict1={'hist_val':gen_hist,'hist_err':gen_err,'hist_bin_centers':hist_bins,'spec_val':spec,'spec_sdev':spec_sdev }
    return dict1

def f_get_images(fname,img_type):
    '''
    Extract image using file name
    '''
    fname,key=fname,img_type
    a1=np.load(fname)
    if key.endswith('input'): 
        size=np.int(np.sqrt(a1.shape[-1])) ### Extract size of images (=128)
        batch_size=a1.shape[0] ### Number of batches
        samples=a1.reshape(batch_size,size,size)
    elif key.endswith('gen') : samples=a1[:,0,:,:]
    else : raise SystemError
    
    return samples
    

def f_high_pixel(images,cutoff=0.9966):
    '''
    Get number of images with a pixel about max cut-off value
    '''
    max_arr=np.amax(images,axis=(1,2))
    num_large=max_arr[max_arr>cutoff].shape[0]

    return num_large


def f_compute_chisqr(dict_val,dict_sample,img_size):
    '''
    Compute chi-square values for sample w.r.t input images
    Input: 2 dictionaries with 4 keys for histogram and spectrum values and errors
    '''
    ### !!Both pixel histograms MUST have same bins and normalization!
    ### Compute chi-sqr
    ### Used in keras code : np.sum(np.divide(np.power(valhist - samphist, 2.0), valhist))
    ###  chi_sqr :: sum((Obs-Val)^2/(Val))
    
    chisqr_dict={}
    
    try: 
        val_dr=dict_val['hist_val'].copy()
        val_dr[val_dr<=0.]=1.0    ### Avoiding division by zero for zero bins

        sq_diff=(dict_val['hist_val']-dict_sample['hist_val'])**2

        size=len(dict_val['hist_val'])
        l1,l2=int(size*0.3),int(size*0.7)
        keys=['chi_1a','chi_1b','chi_1c','chi_1']
        
        for (key,start,end) in zip(keys,[0,l1,l2,0],[l1,l2,None,None]):  # 4 lists : small, medium, large pixel values and full 
            chisqr_dict.update({key:np.sum(np.divide(sq_diff[start:end],val_dr[start:end]))})

        idx=None  # Choosing the number of histograms to use. Eg : -5 to skip last 5 bins
    #     chisqr_dict.update({'chi_sqr1':})

        chisqr_dict.update({'chi_2':np.sum(np.divide(sq_diff[:idx],1.0))}) ## chi-sqr without denominator division
        chisqr_dict.update({'chi_imgvar':np.sum(dict_sample['hist_err'][:idx])/np.sum(dict_val['hist_err'][:idx])}) ## measures total spread in histograms wrt to input data

        idx=int(img_size/2)
        spec_diff=(dict_val['spec_val']-dict_sample['spec_val'])**2
        ### computing the spectral loss chi-square
        chisqr_dict.update({'chi_spec1':np.sum(spec_diff[:idx]/dict_sample['spec_val'][:idx]**2)})

        ### computing the spectral loss chi-square
        chisqr_dict.update({'chi_spec2':np.sum(spec_diff[:idx]/dict_sample['spec_sdev'][:idx]**2)})
        
        spec_loss=1.0*np.log(np.mean((dict_val['spec_val'][:idx]-dict_sample['spec_val'][:idx])**2))+1.0*np.log(np.mean((dict_val['spec_sdev'][:idx]-dict_sample['spec_sdev'][:idx])**2))
#         print(spec_loss)
        chisqr_dict.update({'chi_spec3':spec_loss})
    
    except Exception as e: 
        print(e)
        
        keys=['chi_1a','chi_1b','chi_1c','chi_1','chi_2','chi_imgvar','chi_spec1','chi_spec2']
        chisqr_dict=dict.fromkeys(keys,np.nan)
        pass
    
    return chisqr_dict
    

def f_get_computed_dict(fname,img_type,bins,dict_val):
    '''
    '''
    
    ### Get images from file
    images=f_get_images(fname,img_type)    
    img_size=images.shape[1]
    ### Compute number of images with high pixel values
    high_pixel=f_high_pixel(images,cutoff=0.9898) # pixels over 780
    very_high_pixel=f_high_pixel(images,cutoff=0.9973) # pixels over 3000
    ### Compute spectrum and histograms
    dict_sample=f_compute_hist_spect(images,bins) ## list of 5 numpy arrays 
    ### Compute chi squares
    dict_chisqrs=f_compute_chisqr(dict_val,dict_sample,img_size)
    
    dict1={}
    dict1.update(dict_chisqrs)
    dict1.update({'num_imgs':images.shape[0],'num_large':high_pixel,'num_vlarge':very_high_pixel})
    dict1.update(dict_sample)
    
    return dict1


if __name__=="__main__":
    
    ## Extract image data
    args=parse_args()
    fldr_name=args.folder
    main_dir=fldr_name
    if main_dir.endswith('/'): main_dir=main_dir[:-1]
    
    assert os.path.exists(main_dir), "Directory doesn't exist"
    print("Analyzing data in",main_dir)
    num_cores=args.cores
    
    ### Extract validation data
    fname=args.val_data
    print("Using validation data from ",fname)
    s_val=np.load(fname,mmap_mode='r')[:8000][:,0,:,:]
    print(s_val.shape)

    ### Get dataframe with file names, sorted by epoch and step
    df_files=f_get_sorted_df(main_dir)

    ### Compute 
    t1=time.time()
    transform=False ## Images are in transformed space (-1,1), convert bins to the same space
    if args.bins_type=='uneven':
        bins=np.concatenate([np.array([-0.5]),np.arange(0.5,100.5,5),np.arange(100.5,300.5,20),np.arange(300.5,1000.5,50),np.array([2000])]) #bin edges to use
    else : 
        bins=np.arange(0,1510,10)
    print("Bins",bins)
    if not transform: bins=f_transform(bins)   ### scale to (-1,1) 
    ### Compute histogram and spectrum of raw data 
    dict_val=f_compute_hist_spect(s_val,bins)

    ### Parallel CPU test
#   ##Using pandarallel : https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply

    df=df_files.copy()
    pandarallel.initialize(progress_bar=True)
    # pandarallel.initialize(nb_workers=num_cores,progress_bar=True)

    t2=time.time()
    dict1=df.parallel_apply(lambda row: f_get_computed_dict(fname=row.fname,img_type='train_gen',bins=bins,dict_val=dict_val),axis=1)
    keys=dict1[0].keys()
    ### Convert list of dicts to dict of lists
    dict_list={key:[k[key] for k in dict1] for key in keys}
    ### Add columns to Dataframe
    for key in dict_list.keys():
        df[key]=dict_list[key]

    t3=time.time()
    print("Time ",t3-t2)
    df.head(5)

    ### Save to file
    df.to_pickle(main_dir+'/df_processed.pkle')
    print("Saved file at ",main_dir+'/df_processed.pkle')
