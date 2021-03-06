import lbann
import lbann.modules.base
import lbann.models.resnet
import math

class ConvLNRelu(lbann.modules.Module):
    """Convolution -> Batch normalization -> ReLU
    Basic unit for ResNets. Assumes image data in NCHW format.
    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 bn_zero_init, bn_statistics_group_size,
                 relu, name):
        """Initialize ConvBNRelu module.
        Args:
            out_channels (int): Number of output channels, i.e. number
                of convolution filters.
            kernel_size (int): Size of convolution kernel.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            bn_zero_init (bool): Zero-initialize batch normalization
                scale.
            bn_statistics_group_size (int): Group size for aggregating
                batch normalization statistics.
            relu (bool): Apply ReLU activation.
            name (str): Module name.
        """
        super().__init__()
        self.name = name
        self.instance = 0

        # Initialize convolution
        self.conv = lbann.modules.Convolution2dModule(
            out_channels, kernel_size,
            stride=stride, padding=padding,
            bias=False,
            name=self.name + '_conv')
            

        # Initialize batch normalization
        bn_scale_init = 0.0 if bn_zero_init else 1.0
        bn_scale = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=bn_scale_init),
            name=self.name + '_bn_scale')
        bn_bias = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0.0),
            name=self.name + '_bn_bias')
        self.bn_weights = [bn_scale, bn_bias]
        self.bn_statistics_group_size = bn_statistics_group_size

        # Initialize ReLU
        self.relu = relu

    def forward(self, x):
        self.instance += 1
        conv = self.conv(x)
#         bn = lbann.BatchNormalization(
#         bn = lbann.LayerNorm(
#             conv, weights=self.bn_weights,
#             statistics_group_size=(-1 if self.bn_statistics_group_size == 0
#                                    else self.bn_statistics_group_size),
#             name='{0}_bn_instance{1}'.format(self.name,self.instance))
        bn=lbann.InstanceNorm(conv, data_layout='data_parallel')
        if self.relu:
            return lbann.Relu(
                bn, name='{0}_relu_instance{1}'.format(self.name,self.instance))
        else:
            return bn


class CosmoGAN(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, mcr, name=None):
        
        self.instance = 0
        self.name = (name if name else 'ExaGAN{0}'.format(CosmoGAN.global_count))
        
        ## Gathering the CNN modules into variables
#         convbnrelu = lbann.models.resnet.ConvBNRelu
        convbnrelu = ConvLNRelu
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
        ##### Discriminator
        d_neurons = [64,128,256,512]
        d_kernel_size,d_stride,d_padding=5,2,2
        
        ### Implementing convolution, bnorm using convbrelu
        ##self, out_channels, kernel_size, stride, padding, bn_zero_init, bn_statistics_group_size, relu, name
        self.d1_conv = [convbnrelu(layer, kernel_size=d_kernel_size, stride=d_stride, padding=d_padding, bn_zero_init=False, bn_statistics_group_size=bn_stats_grp_sz, relu=False, name=self.name+'_disc1_conv'+str(i)) for i,layer in enumerate(d_neurons)]
        
        ## Trying without convbrelu
#         self.d1_conv = [conv(layer,d_kernel_size, stride=d_stride, padding=d_padding, transpose=False, bias= False, weights=[lbann.Weights(initializer=self.inits['conv'])], name=self.name+'_disc1_conv'+str(i)) for i,layer in enumerate(d_neurons)]
        
        ### Fully connected layer
        ##self,size,bias=True,transpose=False,weights=[],activation=None,name=None,data_layout='data_parallel',parallel_strategy={}): 
        self.d1_fc = fc(1,name=self.name+'_disc1_fc', weights=[lbann.Weights(initializer=self.inits['dense'])])
        
        #stacked_discriminator, this will be frozen, no optimizer, 
        #layer has to be named for callback
        self.d2_conv = [convbnrelu(layer, d_kernel_size, d_stride, d_padding, False, bn_stats_grp_sz, False,name=self.name+'_disc2_conv'+str(i)) for i,layer in enumerate(d_neurons)] 
        
#         self.d2_conv = [conv(layer,d_kernel_size, stride=d_stride, padding=d_padding, transpose=False, bias=False, weights=[lbann.Weights(initializer=self.inits['conv'])], name=self.name+'_disc2_conv'+str(i)) for i,layer in enumerate(d_neurons)]

        self.d2_fc = fc(1,name=self.name+'_disc2_fc', weights=[lbann.Weights(initializer=self.inits['dense'])])
        
        #########################
        ##### Generator
        g_neurons = [256,128,64]
        g_kernel_size,g_stride,g_padding=5,2,2

        ### Transpose convolution
        ##(self, num_dims,out_channels,kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,weights=[],activation=None,name=None,transpose=False,parallel_strategy={})
#         self.g_convT = [conv(layer, g_kernel_size, stride=g_stride, padding=g_padding, transpose=True, weights=[lbann.Weights(initializer=self.inits['convT'])]) for i,layer in enumerate(g_neurons)] 
        self.g_convT = [conv(layer, g_kernel_size, stride=g_stride, padding=g_padding, transpose=True,weights=[lbann.Weights(initializer=self.inits['convT'])],name=self.name+'_gen_convt'+str(i)) for i,layer in enumerate(g_neurons)] 

        
        ### Fully connected
        fc_size=32768 ### (8 * 8 * 2 * 256)
        self.g_fc1 = fc(fc_size,name=self.name+'_gen_fc1', weights=[lbann.Weights(initializer=self.inits['dense'])])
        
        ### Final conv transpose
        self.g_convT3 = conv(1, g_kernel_size, stride=g_stride, padding=g_padding, activation=lbann.Tanh,name='gen_img',transpose=True,
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
        
        print('MCR in forward',mcr)
        if mcr: ### Multi-channel rescaling. Add extra channel for real images. Generated images are rescaled inside generator
            linear_scale=1/self.linear_scaler
            ch2=lbann.Tanh(lbann.WeightedSum(self.inv_transform(lbann.Identity(img)),scaling_factors=str(linear_scale)))
            y = lbann.Concatenation(lbann.Identity(img),ch2,axis=0)
            img = lbann.Reshape(y, dims='2 128 128')
        else: 
            img=lbann.Reshape(img,dims='1 128 128')
        
        d1_real = self.forward_discriminator1(img)  #instance1
        gen_img = self.forward_generator(z,mcr=mcr)
        
        d1_fake = self.forward_discriminator1(lbann.StopGradient(gen_img)) #instance2
        d_adv = self.forward_discriminator2(gen_img) #instance 3 //need to freeze
        #d1s share weights, d1_w is copied to d_adv (through replace weight callback) and freeze
        
        return d1_real, d1_fake, d_adv, gen_img, img
    
    def forward_discriminator1(self,img):
        '''
        Discriminator 1
        '''
        
        bn_wts=[lbann.Weights(initializer=lbann.ConstantInitializer(value=1.0)),
                lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0))]
            
        for count,lyr in enumerate(self.d1_conv):
            if count==0: x=lbann.LeakyRelu(lyr(img), negative_slope=0.2)
            else : x = lbann.LeakyRelu(lyr(x), negative_slope=0.2)
            #### without convbrlu        
#             if count==0: x = lbann.LeakyRelu(lbann.BatchNormalization(lyr(img),weights=bn_wts,statistics_group_size=-1),negative_slope=0.2)
#             else: x = lbann.LeakyRelu(lbann.BatchNormalization(lyr(x),weights=bn_wts,statistics_group_size=-1),negative_slope=0.2)

        dims=32768
        #dims=25088 ## for padding=1
        y= self.d1_fc(lbann.Reshape(x,dims=str(dims))) 
        
        return y
        
    def forward_discriminator2(self,img):
        '''
        Discriminator 2. Weights are frozen as part of Adversarial network = Stacked G + D
        '''
        bn_wts=[lbann.Weights(initializer=lbann.ConstantInitializer(value=1.0)),
                    lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0))]
            
        for count,lyr in enumerate(self.d2_conv):
            if count==0: x=lbann.LeakyRelu(lyr(img), negative_slope=0.2)
            else : x = lbann.LeakyRelu(lyr(x), negative_slope=0.2)
            #### without convbrlu
#             if count==0: x = lbann.LeakyRelu(lbann.BatchNormalization(lyr(img),weights=bn_wts,statistics_group_size=-1),negative_slope=0.2)
#             else: x = lbann.LeakyRelu(lbann.BatchNormalization(lyr(x),weights=bn_wts,statistics_group_size=-1),negative_slope=0.2)

        dims=32768
        #dims=25088 ## for padding=1
        y= self.d2_fc(lbann.Reshape(x,dims=str(dims))) 
        
        return y
        
    def forward_generator(self,z,mcr):
        '''
        Build the Generator
        '''
        x = self.g_fc1(z)
#         x = lbann.EntrywiseBatchNormalization(x, decay=0.9, epsilon=1e-5)
        x = lbann.LayerNorm(x)

        x = lbann.EntrywiseScaleBias(x)
        x = lbann.Relu(x)
        dims='512 8 8'
        x = lbann.Reshape(x, dims=dims) #channel first
        
        for count,lyr in enumerate(self.g_convT):
#             x = lbann.Relu(lbann.BatchNormalization(lyr(x),decay=0.9,scale_init=1.0,epsilon=1e-5))
            x = lbann.Relu(lbann.InstanceNorm(lyr(x)))

        
        img = self.g_convT3(x)
        
        if mcr: ### For multi-channel rescaling, add extra channel to output image
            linear_scale=1/self.linear_scaler
#             linear_scale=lbann.Constant(value=0.001)
            ch2 = lbann.Tanh(lbann.WeightedSum(self.inv_transform(img),scaling_factors=str(linear_scale)))
            y = lbann.Concatenation(img,ch2,axis=0)
            img = lbann.Reshape(y, dims='2 128 128')
        else:
            img=lbann.Reshape(img,dims='1 128 128')
        
        return img
    
    def inv_transform(self,y): ### Transform to original space
        '''
        The inverse of the transformation function that scales the data before training
        '''
        inv_transform = lbann.WeightedSum(
                                      lbann.SafeDivide(
                                      lbann.Add(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y)),
                                      lbann.Subtract(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y))),
                                      scaling_factors=str(self.datascale))
        
        return inv_transform
   
 #      def inv_transform(self, y):### New tranformation : log-linear
#         threshold = lbann.Constant(value=0.5, hint_layer=y)
#         is_above_threshold = lbann.Greater(y, threshold)
#         is_below_threshold = lbann.LogicalNot(is_above_threshold)
        
#         below = lbann.SafeDivide(
#             lbann.Subtract(y, lbann.Constant(value=1, hint_layer=y)),
#             lbann.Constant(value=0.03, hint_layer=y),
#         )
#         above = lbann.Exp(lbann.SafeDivide(
#             lbann.Subtract(
#                 y,
#                 lbann.Constant(value=0.5-0.5/math.log(300)*math.log(50), hint_layer=y)),
#             lbann.Constant(value=0.5/math.log(300), hint_layer=y),
#         ))
#         return lbann.Add(
#             lbann.Multiply(is_above_threshold, above),
#             lbann.Multiply(is_below_threshold, below),
#         )


    
# def f_invtransform_new(y):
#     if y<=0.5:
#         a=0.03;b=-1.0
#         return (y-b)/a
#     elif y>0.5: 
#         a=0.5/np.log(300)
#         b=0.5-a*np.log(50)
#         return np.exp((y-b)/a)

