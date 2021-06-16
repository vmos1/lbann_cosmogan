import ExaGAN
import argparse
#import dataset
import lbann
import os
from datetime import datetime
from os.path import abspath, dirname, join

from lbann.contrib.modules.fftshift import FFTShift
from lbann.contrib.modules.radial_profile import RadialProfile
from lbann.util import str_list
import yaml
import shutil

# ==============================================
# Setup and launch experiment
# ==============================================

def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
#     add_arg('--epochs','-e', type=int, default=10, help='The number of epochs')
    add_arg('--config','-cfile',  type=str, default='config_128_cori.yaml', help='Name of config file')

    return parser.parse_args()


def f_invtransform(y,scale=4.0): ### Transform to original space
    '''
    The inverse of the transformation function that scales the data before training
    '''
    inv_transform = lbann.WeightedSum(
                                  lbann.SafeDivide(
                                  lbann.Add(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y)),
                                  lbann.Subtract(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y))),
                                  scaling_factors=str(scale))

    return inv_transform


def list2str(l):
    return ' '.join(l)

def construct_model(num_epochs,mcr,spectral_loss,save_batch_interval):
    """Construct LBANN model.
    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A',name='inp_img')
    
    ### Create expected labels for real and fake data (with label flipping = 0.01)
    prob_flip=0.01
    label_flip_rand = lbann.Uniform(min=0,max=1, neuron_dims='1')
    label_flip_prob = lbann.Constant(value=prob_flip, num_neurons='1')
    ones = lbann.GreaterEqual(label_flip_rand,label_flip_prob, name='is_real')
    zeros = lbann.LogicalNot(ones,name='is_fake')
    gen_ones=lbann.Constant(value=1.0,num_neurons='1')## All ones: no flip. Input for training Generator.
    
    #==============================================
    ### Implement GAN
    ##Create the noise vector
    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="64", name='noise_vec'),dims='1 64')
    ## Creating the GAN object and implementing forward pass for both networks ###
    d1_real, d1_fake, d_adv, gen_img, img  = ExaGAN.CosmoGAN(mcr)(input,z,mcr) 
    
    #==============================================
    ### Compute quantities for adding to Loss and Metrics
    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,ones],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zeros],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,gen_ones],name='d_adv_bce')
    
    #img_loss = lbann.MeanSquaredError([gen_img,img])
    #l1_loss = lbann.L1Norm(lbann.WeightedSum([gen_img,img], scaling_factors="1 -1")) 
    
    #==============================================
    ### Set up source and destination layers
    layers = list(lbann.traverse_layer_graph(input))
    weights = set()
    src_layers,dst_layers = [],[]
    for l in layers:
        if(l.weights and "disc1" in l.name and "instance1" in l.name):
            src_layers.append(l.name)
        #freeze weights in disc2, analogous to discrim.trainable=False in Keras
        if(l.weights and "disc2" in l.name):
            dst_layers.append(l.name)
            for idx in range(len(l.weights)):
                l.weights[idx].optimizer = lbann.NoOptimizer()
        weights.update(l.weights)
    
    #l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)
    
    #==============================================
    ### Define Loss and Metrics
    #Define loss (Objective function)
    loss_list=[d1_real_bce,d1_fake_bce,d_adv_bce] ## Usual GAN loss function
#     loss_list=[d1_real_bce,d1_fake_bce] ## skipping adversarial loss for G for testing spectral loss
    
    if spectral_loss:
        dft_gen_img = lbann.DFTAbs(f_invtransform(gen_img))
        dft_img = lbann.StopGradient(lbann.DFTAbs(f_invtransform(img)))
#         spec_loss = lbann.Log(lbann.MeanSquaredError(dft_gen_img, dft_img))
         
        ## Adding full spectral loss
        gen_fft=FFTShift()(dft_gen_img,[1, 128, 128])
        gen_spec_prof=RadialProfile()(gen_fft,[1, 128, 128],63)
        
        img_fft=FFTShift()(dft_img,[1, 128, 128])
        img_spec_prof=RadialProfile()(img_fft,[1, 128, 128],63)
        spec_loss = lbann.Log(lbann.MeanSquaredError(gen_spec_prof, img_spec_prof))
        loss_list.append(lbann.LayerTerm(spec_loss, scale=spectral_loss))
        
    loss = lbann.ObjectiveFunction(loss_list)
    
    mse=lbann.MeanSquaredError(gen_img,lbann.StopGradient(img))
    #Define metrics
    metrics = [lbann.Metric(d1_real_bce,name='d_real'),lbann.Metric(d1_fake_bce, name='d_fake'), lbann.Metric(d_adv_bce,name='gen_adv'),lbann.Metric(mse,name='image_MSE')]
    if spectral_loss: metrics.append(lbann.Metric(spec_loss,name='spec_loss'))
    
    #==============================================
    ### Define callbacks list
    callbacks_list=[]
    dump_outputs=True
    save_model=False
    print_model=False
    
    callbacks_list.append(lbann.CallbackPrint())
    callbacks_list.append(lbann.CallbackTimer())
    callbacks_list.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers), destination_layers=list2str(dst_layers),batch_interval=1))
    if dump_outputs:
        #callbacks_list.append(lbann.CallbackDumpOutputs(layers='inp_img gen_img_instance1_activation', execution_modes='train validation', directory='dump_outs',batch_interval=save_batch_interval,format='npy')) 
        callbacks_list.append(lbann.CallbackDumpOutputs(layers='gen_img_instance1_activation', execution_modes='train validation test', directory='dump_outs',batch_interval=save_batch_interval,format='npy')) 
    
    if save_model : callbacks_list.append(lbann.CallbackSaveModel(dir='models'))
    if print_model: callbacks_list.append(lbann.CallbackPrintModelDescription())
    
    ### Construct model
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=loss,
                       callbacks=callbacks_list)


def construct_data_reader(data_pct,val_ratio):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.
    """
    import os.path
    import lbann
    
    print('Data and validation pct',data_pct,val_ratio)
    module_file = os.path.abspath(__file__)
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    module_dir = os.path.dirname(module_file)

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.shuffle = True
    data_reader.percent_of_data_to_use = data_pct
    data_reader.validation_percent = val_ratio
    data_reader.python.module = 'dataset'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'f_get_sample'
    data_reader.python.num_samples_function = 'f_num_samples'
    data_reader.python.sample_dims_function = 'f_sample_dims'
    
    # Test set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'test'
    data_reader.shuffle = True
    data_reader.percent_of_data_to_use = 1.0
#     data_reader.validation_percent = val_ratio
    data_reader.python.module = 'dataset'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'f_get_sample'
    data_reader.python.num_samples_function = 'f_num_samples'
    data_reader.python.sample_dims_function = 'f_sample_dims'
    

    return message

if __name__ == '__main__':
#     import lbann
    # export config_file='config_cori_128.yaml'
#     args=f_parse_args()
#     print('Args',args)
    
    ## Add to gdict from config file 
    config_file=os.environ['config_file']
    with open(config_file) as f:
        config_dict= yaml.load(f, Loader=yaml.SafeLoader)
    gdict=config_dict['parameters']

    num_epochs,num_nodes,num_procs,random_seed=gdict['epochs'],gdict['nodes'],gdict['procs'],gdict['seed']
    print("Random seed",random_seed)
    if gdict['mcr']: print("Using Multi-channel rescaling")
    ### Create prefix for foldername
    now=datetime.now()
    fldr_name=now.strftime('%Y%m%d_%H%M%S') ## time format
    
    data_pct,val_ratio=1.0,0.1 # Percentage of data to use, % of data for validation
    batchsize=gdict['batchsize']
    
    work_dir=gdict['op_loc']+"{0}_bsize{1}_{2}".format(fldr_name,batchsize,gdict['run_suffix'])
    print(work_dir)
    print(gdict)
    
    #####################
    ### Run lbann
    trainer = lbann.Trainer(mini_batch_size=batchsize,random_seed=random_seed,callbacks=lbann.CallbackCheckpoint(checkpoint_dir='chkpt', 
#   checkpoint_epochs=10))  
    checkpoint_steps=gdict['checkpoint_size']))
    
    spectral_loss=gdict['lambda_spec']
    if spectral_loss: print("Using Spectral loss with coupling",spectral_loss)
        
    model = construct_model(num_epochs,gdict['mcr'],spectral_loss=spectral_loss,save_batch_interval=int(gdict['checkpoint_size']))
    # Setup optimizer
    opt = lbann.Adam(learn_rate=gdict['learn_rate'],beta1=gdict['beta1'],beta2=gdict['beta2'],eps=float(gdict['eps']))
    # Load data reader from prototext
    data_reader = construct_data_reader(data_pct,val_ratio)
    
    status = lbann.run(trainer,model, data_reader, opt,
                       nodes=num_nodes, procs_per_node=num_procs, 
                       work_dir=work_dir,
                       scheduler='slurm', time_limit=1440, setup_only=False)
    
    ## Copy config file to folder
    shutil.copy(config_file,work_dir)


    print(status)
