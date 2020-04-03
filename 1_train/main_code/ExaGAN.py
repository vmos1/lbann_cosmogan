import lbann
import lbann.modules.base
import lbann.models.resnet


class CosmoGAN(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, name=None,mcr=True):
        
        self.instance = 0
        self.name = (name if name else 'ExaGAN{0}'.format(CosmoGAN.global_count))
        
        print('MCR',mcr)
        
        convbnrelu = lbann.models.resnet.ConvBNRelu
        fc = lbann.modules.FullyConnectedModule
        conv = lbann.modules.Convolution2dModule
        #bn_stats_grp_sz = 0 #0 global, 1 local
        bn_stats_grp_sz = -1 #0 global, 1 local
        
        ##MCR properties #@todo: make multichannel optional
        self.datascale = 4 
        self.linear_scaler=1000.
        
        self.inits = {'dense': lbann.NormalInitializer(mean=0,standard_deviation=0.02),
                      'conv': lbann.NormalInitializer(mean=0,standard_deviation=0.02), #should be truncated Normal
                      'convT':lbann.NormalInitializer(mean=0,standard_deviation=0.02)}
        
        d_neurons = [64,128,256,512]
        ##self, out_channels, kernel_size, stride, padding, bn_zero_init, bn_statistics_group_size, relu, name
        self.d1_conv = [convbnrelu(d_neurons[i], 4, 2, 1, False, bn_stats_grp_sz, False,name=self.name+'_disc1_conv'+str(i))
                   for i in range(len(d_neurons))] 
        #self,size,bias=True,transpose=False,weights=[],activation=None,name=None,data_layout='data_parallel',parallel_strategy={}): 
        self.d1_fc = fc(1,name=self.name+'_disc1_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])
        
        #stacked_discriminator, this will be frozen, no optimizer, 
        #layer has to be named for callback
        self.d2_conv = [convbnrelu(d_neurons[i], 4, 2, 1, False, bn_stats_grp_sz, False,name=self.name+'_disc2_conv'+str(i))
                   for i in range(len(d_neurons))] 
        self.d2_fc = fc(1,name=self.name+'_disc2_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])
        #generator
        g_neurons = [256,128,64]
        
        self.g_convT = [conv(g_neurons[i], 5, stride=2, padding=2, transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
                       for i in range(len(g_neurons))] 
        ### When using mcr, the size is double of 128 * 128
        
        fc_size=32768 if mcr else 16384
        self.g_fc1 = fc(fc_size,name=self.name+'_gen_fc1',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])
        self.g_convT3 = conv(1, 5, stride=2, padding=2, activation=lbann.Tanh,name='gen_img',transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
    
    def forward(self, img, z,mcr):
        '''
        Steps: 
        - Modify image if using mcr
        - D1 + imgs -> d1_real
        - G + noise -> gen_imgs
        - D1 + gen_imgs -> d1_fake
        - Adv (D2) + gen_imgs
        Return D outputs and gen_imgs
        '''
        
        if mcr: ### Multi-channel rescaling
            ch2 = self.inv_transform(lbann.Identity(img))
            y = lbann.Concatenation(lbann.Identity(img),ch2,axis=0)
            img = lbann.Reshape(y, dims='2 128 128')
        
        d1_real = self.forward_discriminator1(img,mcr=mcr)  #instance1
        gen_img = self.forward_generator(z,mcr=mcr)
        d1_fake = self.forward_discriminator1(lbann.StopGradient(gen_img),mcr=mcr) #instance2
        d_adv = self.forward_discriminator2(gen_img,mcr=mcr) #instance 3 //need to freeze
        #d1s share weights, d1_w is copied to d_adv (through replace weight callback) and freeze
        
        return d1_real, d1_fake, d_adv,gen_img
    
    def forward_discriminator1(self,img,mcr=False):
        '''
        Discriminator 1
        '''
        x = lbann.LeakyRelu(self.d1_conv[0](img), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv[1](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv[2](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv[3](x), negative_slope=0.2)
        dims=32768 if mcr else 16384
        y= self.d1_fc(lbann.Reshape(x,dims=str(dims))) 
        
        return y
    
    def forward_discriminator2(self,img,mcr=False):
        '''
        Discriminator 2. Weights are frozen as part of Adversarial network = Stacked G + D
        '''
        x = lbann.LeakyRelu(self.d2_conv[0](img), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv[1](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv[2](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv[3](x), negative_slope=0.2)
        dims=32768 if mcr else 16384
        y= self.d2_fc(lbann.Reshape(x,dims=str(dims))) 
        
        return y
    
    def forward_generator(self,z,mcr=False):
        '''
        Build the Generator
        '''
        x = lbann.Relu(lbann.BatchNormalization(self.g_fc1(z),decay=0.9,scale_init=1.0,epsilon=1e-5))
        dims='512 8 8' if mcr else '256 8 8'
        x = lbann.Reshape(x, dims=dims) #channel first
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[0](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[1](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[2](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        img = self.g_convT3(x)
        
        if mcr: ### For multi-channel rescaling, add extra channel to output image
            linear_scale=1/1000.0
            #ch2 = lbann.Tanh(self.inv_transform(img)/linear_scalar)
            ch2 = lbann.Tanh(lbann.WeightedSum(self.inv_transform(img),scaling_factors=str(linear_scale)))
            y = lbann.Concatenation(img,ch2,axis=0)
            img = lbann.Reshape(y, dims='2 128 128')
        else:
            img=lbann.Reshape(img,dims='1 128 128')
        
        return img
        
    def inv_transform(self,y): 
        '''
        The inverse of the transformation function
        '''
        inv_transform = lbann.WeightedSum(
                                      lbann.SafeDivide(
                                      lbann.Add(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y)),
                                      lbann.Subtract(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y))),
                                      scaling_factors=str(self.datascale))
        #linear_scale = 1/self.linear_scaler
        #CH2 = lbann.Tanh(lbann.WeightedSum(inv_transform,scaling_factors=str(linear_scale)))
        #return CH2  
        return inv_transform
