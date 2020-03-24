import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./networks/')
sys.path.append('./utils/')
import GANbuild
import plots




config = 'base'
runId = '0'
run = './expts/'+config+'/run'+runId+'/'

modelpath = run+'models/g_cosmo_best.h5'

if not os.path.isfile(modelpath):
    print("Error: File %s with pre-trained weights could not be found")
    sys.exit()

GAN = GANbuild.DCGAN(config, run)
GAN.genrtor.load_weights(modelpath)

# Plot generated images
plots.save_img_grid(GAN.genrtor, GAN.noise_vect_len, GAN.invtransform, GAN.C_axis, Xterm=True, 
                    scale=GAN.cscale, multichannel=GAN.multichannel)

# Plot pixel intensity histogram and calculate chi-square score
chi = plots.pix_intensity_hist(GAN.val_imgs, GAN.genrtor, GAN.noise_vect_len, 
                               GAN.invtransform, GAN.C_axis, multichannel=GAN.multichannel, Xterm=True)

# Plot power spectrum and calculate chi-square score
pschi = plots.pspect(GAN.val_imgs, GAN.genrtor, GAN.invtransform, GAN.noise_vect_len, 
                    GAN.C_axis, Xterm=True, multichannel=GAN.multichannel)

plt.show()





