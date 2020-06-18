import ExaGAN
import argparse
#import dataset
#import lbann.contrib.lc.launcher
from os.path import abspath, dirname, join
import lbann

# ==============================================
# Setup and launch experiment
# ==============================================

def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--epochs','-e', type=int, default=10, help='The number of epochs')
    add_arg('--procs','-p',  type=int, default=1, help='The number of processes per node')
    add_arg('--nodes','-n',  type=int, default=1, help='The number of GPU nodes requested')
    add_arg('--seed','-s',  type=int, default=232, help='Seed for random number sequence')
    add_arg('--mcr','-m',  action='store_true', default=True, help='Multi-channel rescaling')
    add_arg('--pretrained_dir','-dr', default='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_data/20200617_062906_exagan/chkpt/trainer0/sgd.shared.validation.epoch.9.step.450', help='directory storing model')
    
    return parser.parse_args()

def list2str(l):
    return ' '.join(l)

def construct_model(num_epochs,mcr,save_batch_interval=82):
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
#     loss_list.append(l2_reg)
    loss = lbann.ObjectiveFunction(loss_list)
    
    #Define metrics
    metrics = [lbann.Metric(d1_real_bce,name='d_real'),lbann.Metric(d1_fake_bce, name='d_fake'), lbann.Metric(d_adv_bce,name='gen'),
               #lbann.Metric(img_loss, name='msq_error'), lbann.Metric(l1_loss, name='l1norm_error') 
#                ,lbann.Metric(l2_reg)
              ]
    
    #==============================================
    ### Define callbacks list
    callbacks_list=[]
    dump_outputs=True
    save_model=True
    print_model=False
    check_point=True
    
    callbacks_list.append(lbann.CallbackPrint())
    callbacks_list.append(lbann.CallbackTimer())
    callbacks_list.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers), destination_layers=list2str(dst_layers),batch_interval=1))
    if dump_outputs:
        callbacks_list.append(lbann.CallbackDumpOutputs(layers='gen_img_instance1_activation', execution_modes='test', directory='dump_outs',batch_interval=1,format='npy')) 
    
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
#    data_reader.role = 'train'
    data_reader.role = 'test'
    data_reader.shuffle = True
    data_reader.percent_of_data_to_use = data_pct
    data_reader.validation_percent = val_ratio
    data_reader.python.module = 'dataset'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'f_get_sample'
    data_reader.python.num_samples_function = 'f_num_samples'
    data_reader.python.sample_dims_function = 'f_sample_dims'
    
    return message

if __name__ == '__main__':
    import lbann
    
    ## Read arguments
    args=f_parse_args()
    print('Args',args)
    num_epochs,num_nodes,num_procs,mcr,random_seed=args.epochs,args.nodes,args.procs,args.mcr,args.seed
    print("Random seed",random_seed)
    
#    mcr=False
    size=105060  # Esimated number of *total* samples. Used to estimate save_interval
    data_pct,val_ratio=1.0,0.2 # Percentage of data to use, % of data for validation
    
    batchsize=3000
    ## Determining the batch interval to save generated images for validation. Factor of 2 for 2 images per epoch 
    save_interval=int(size*val_ratio/(2.0*batchsize))
    print('Save interval',save_interval)
    
    #####################
    ### Run lbann
    trainer = lbann.Trainer(mini_batch_size=batchsize,random_seed=random_seed)
    model = construct_model(num_epochs,mcr,save_batch_interval=save_interval)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = construct_data_reader(data_pct,val_ratio)
    
    ### Initialize LBANN inf executable
    lbann_exe = abspath(lbann.lbann_exe())
    lbann_exe = join(dirname(lbann_exe), 'lbann_inf')
    
    print('Loading model from :',args.pretrained_dir)
    status = lbann.run(trainer,model, data_reader, opt,lbann_exe=lbann_exe,
                       nodes=num_nodes, procs_per_node=num_procs,
                       scheduler='slurm', time_limit=1440, setup_only=False,
                       job_name='gen_img_exagan',
                      lbann_args=[f'--load_model_weights_dir={args.pretrained_dir}', 
                                  '--load_model_weights_dir_is_complete',
                                  ]
                      )
    
    print(status)

