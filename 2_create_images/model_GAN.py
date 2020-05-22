import lbann
import lbann.modules.base
import lbann.models.resnet
import math

class CosmoGAN(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, mcr, name=None):
        
        self.instance = 0
        self.name = (name if name else 'ExaGAN{0}'.format(CosmoGAN.global_count))
        
        ## Gathering the CNN modules into variables
        convbnrelu = lbann.models.resnet.ConvBNRelu
        fc = lbann.modules.FullyConnectedModule
        conv = lbann.modules.Convolution2dModule
        
        #bn_stats_grp_sz = 0 #0 global, 1 local
        bn_stats_grp_sz = -1 #0 global, 1 local
        self.datascale = 4.0
        self.linear_scaler=1000.0
        self.inits = {'dense': lbann.NormalInitializer(mean=0,standard_deviation=0.02),
                      'conv': lbann.NormalInitializer(mean=0,standard_deviation=0.02), #should be truncated Normal
                      'convT':lbann.NormalInitializer(mean=0,standard_deviation=0.02)}
        
        #########################
        ##### Generator
        g_neurons = [256,128,64]
        g_kernel_size,g_stride,g_padding=5,2,2

        ### Transpose convolution
        ##(self, num_dims,out_channels,kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,weights=[],activation=None,name=None,transpose=False,parallel_strategy={})
        self.g_convT = [conv(layer, g_kernel_size, stride=g_stride, padding=g_padding, transpose=True, weights=[lbann.Weights(initializer=self.inits['convT'])]) for i,layer in enumerate(g_neurons)] 
        
        ### Fully connected
        fc_size=32768 ### (8 * 8 * 2 * 256)
        self.g_fc1 = fc(fc_size,name=self.name+'_gen_fc1', weights=[lbann.Weights(initializer=self.inits['dense'])])
        
        ### Final conv transpose
        self.g_convT3 = conv(1, g_kernel_size, stride=g_stride, padding=g_padding, activation=lbann.Tanh,name='gen_img',transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
    

    def forward(self, img, z,mcr):
        '''   G + noise -> gen_imgs   '''
        
        print('MCR in forward',mcr)
        gen_img = self.forward_generator(z,mcr=mcr)
        
        return gen_img
        
    def forward_generator(self,z,mcr):
        '''
        Build the Generator
        '''
        x = lbann.Relu(lbann.BatchNormalization(self.g_fc1(z),decay=0.9,scale_init=1.0,epsilon=1e-5))
        dims='512 8 8'
        x = lbann.Reshape(x, dims=dims) #channel first
        
        for count,lyr in enumerate(self.g_convT):
            x = lbann.Relu(lbann.BatchNormalization(lyr(x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        
        img = self.g_convT3(x)
        
        if mcr: ### For multi-channel rescaling, add extra channel to output image
            linear_scale=1/self.linear_scaler
            ch2 = lbann.Tanh(lbann.WeightedSum(self.inv_transform(img),scaling_factors=str(linear_scale)))
            y = lbann.Concatenation(img,ch2,axis=0)
            img = lbann.Reshape(y, dims='2 128 128')
        else:
            img=lbann.Reshape(img,dims='1 128 128')
        
        return img
    
    def inv_transform(self,y): ### Original transformation
        '''
        The inverse of the transformation function that scales the data before training
        '''
        inv_transform = lbann.WeightedSum(
                                      lbann.SafeDivide(
                                      lbann.Add(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y)),
                                      lbann.Subtract(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y))),
                                      scaling_factors=str(self.datascale))

        return inv_transform
   

