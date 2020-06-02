import model_GAN
import argparse
#import dataset
#import lbann.contrib.lc.launcher
import lbann
from os.path import abspath, dirname, join
import lbann.contrib.args


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
    add_arg('--pretrained_dir','-dr', default='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_data/20200529_111342_seed3273_80epochs/models/trainer0/model0', help='directory storing model')
    
    return parser.parse_args()

def list2str(l):
    return ' '.join(l)

def construct_model(num_epochs,mcr,save_batch_interval=82):
    """Construct LBANN model.
    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A',name='inp_img')
    
    #==============================================
    ##Create the noise vector
    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="64", name='noise_vec'),dims='1 64')
    ## Creating the GAN object and implementing forward pass for both networks ###
    gen_img = model_GAN.CosmoGAN(mcr)(input,z,mcr)
    
#     #==============================================
#     ### Set up source and destination layers
#     layers = list(lbann.traverse_layer_graph(input))
#     weights = set()
#     src_layers,dst_layers = [],[]
#     for l in layers:
#         if(l.weights and "disc1" in l.name and "instance1" in l.name):
#             src_layers.append(l.name)
#         #freeze weights in disc2, analogous to discrim.trainable=False in Keras
#         if(l.weights and "disc2" in l.name):
#             dst_layers.append(l.name)
#             for idx in range(len(l.weights)):
#                 l.weights[idx].optimizer = lbann.NoOptimizer()
#         weights.update(l.weights)
    
#     #==============================================
#     ### Define Loss and Metrics
#     #Define loss (Objective function)
#     loss_list=[d1_real_bce,d1_fake_bce,d_adv_bce] ## Usual GAN loss function
# #     loss_list.append(l2_reg)
#     loss = lbann.ObjectiveFunction(loss_list)
    
#     #Define metrics
#     metrics = [lbann.Metric(d1_real_bce,name='d_real'),lbann.Metric(d1_fake_bce, name='d_fake'), lbann.Metric(d_adv_bce,name='gen'),
#                #lbann.Metric(img_loss, name='msq_error') ,lbann.Metric(l1_loss, name='l1norm_error') 
# #                ,lbann.Metric(l2_reg)
#               ]
    
    #==============================================
    ### Define callbacks list
    callbacks_list=[]
    print_model=False
    fname=''
    callbacks_list.append(lbann.CallbackPrint())
    callbacks_list.append(lbann.CallbackTimer())
    callbacks_list.append(lbann.CallbackLoadModel(dirs=str(fname)))
    if print_model: callbacks_list.append(lbann.CallbackPrintModelDescription())
    
    ### Construct model
    return lbann.Model(num_epochs,
#                        weights=weights,
#                        layers=layers,
#                        metrics=metrics,
#                        objective_function=loss,
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
    
    return message

if __name__ == '__main__':
    import lbann
    
    args=f_parse_args()
    print(args)
    num_epochs,num_nodes,num_procs,mcr,random_seed=args.epochs,args.nodes,args.procs,args.mcr,args.seed
    
#    mcr=False
    size=105060  ### Esimated number of *total* samples
    data_pct,val_ratio=1.0,0.2 ## Percentage of data to use, % of data for validation
    batchsize=128
    
    ## Determining the batch interval to save generated images for validation. Factor of 2 for 2 images per epoch 
    save_interval=int(size*val_ratio/(2.0*batchsize))
    print('Save interval',save_interval)
    
    trainer = lbann.Trainer(mini_batch_size=batchsize,random_seed=random_seed)
    model = construct_model(num_epochs,mcr,save_batch_interval=save_interval)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = construct_data_reader(data_pct,val_ratio)
    
#     status = lbann.run(trainer,model, data_reader, opt,
#                        scheduler='slurm',
#                        nodes=num_nodes,
#                        procs_per_node=num_procs,
#                        time_limit=1440,
#                        setup_only=False,
#                        job_name='exagan')
    
    ### Initialize LBANN inf executable
    lbann_exe = abspath(lbann.lbann_exe())
    lbann_exe = join(dirname(lbann_exe), 'lbann_inf')
    
    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    
    status = lbann.run(trainer,model, data_reader_proto, opt,
                       lbann_exe,
                       scheduler='slurm',
                       nodes=1,
                       procs_per_node=1,
                       time_limit=30,
                       setup_only=False,
                       batch_job=False,
                       job_name='gen_images',
                       lbann_args=['--preload_data_store --use_data_store --load_model_weights_dir_is_complete',
                                   f'--metadata={metadata_prototext}',
                                   f'--load_model_weights_dir={args.pretrained_dir}',
                                   f'--index_list_test={args.index_list_test}',
                                   f'--data_filedir_test={args.data_filedir_test}'],
                                   **kwargs)
    
    print(status)