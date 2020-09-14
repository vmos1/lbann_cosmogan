import ExaGAN
import argparse
#import dataset
import lbann
import os
from datetime import datetime

# ==============================================
# Setup and launch experiment
# ==============================================

def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--epochs','-e', type=int, default=10, help='The number of epochs')
    add_arg('--procs','-p',  type=int, default=1, help='The number of processes per node')
    add_arg('--suffix','-sfx',  type=str, default='128', help='The tail end of the name of the folder')
    add_arg('--nodes','-n',  type=int, default=1, help='The number of GPU nodes requested')
    add_arg('--seed','-s',  type=int, default=232, help='Seed for random number sequence')
    add_arg('--batchsize','-b',  type=int, default=100, help='batchsize')
    add_arg('--step_interval','-stp',  type=int, default=1, help='Interval at which checkpointing is done.')
    add_arg('--mcr','-m',  action='store_true', default=True, help='Multi-channel rescaling')
    
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
    save_model=False
    print_model=False
    
    callbacks_list.append(lbann.CallbackPrint())
    callbacks_list.append(lbann.CallbackTimer())
    callbacks_list.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers), destination_layers=list2str(dst_layers),batch_interval=1))
    if dump_outputs:
        #callbacks_list.append(lbann.CallbackDumpOutputs(layers='inp_img gen_img_instance1_activation', execution_modes='train validation', directory='dump_outs',batch_interval=save_batch_interval,format='npy')) 
        callbacks_list.append(lbann.CallbackDumpOutputs(layers='gen_img_instance1_activation', execution_modes='train validation', directory='dump_outs',batch_interval=save_batch_interval,format='npy')) 
    
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
    
    return message

if __name__ == '__main__':
    import lbann
    
    ## Read arguments
    args=f_parse_args()
    print('Args',args)
    num_epochs,num_nodes,num_procs,mcr,random_seed=args.epochs,args.nodes,args.procs,args.mcr,args.seed
    print("Random seed",random_seed)
    ### Create prefix for foldername 
    now=datetime.now()
    fldr_name=now.strftime('%Y%m%d_%H%M%S') ## time format

#    mcr=False
    data_pct,val_ratio=1.0,0.1 # Percentage of data to use, % of data for validation
    batchsize=args.batchsize
    step_interval=args.step_interval # 80 gives you 10 steps per epoch for batchsize 256
    print('Step interval',step_interval)
    
    work_dir="/global/cscratch1/sd/vpa/proj/cosmogan/results_dir/512square/{0}_bsize{1}_{2}".format(fldr_name,batchsize,args.suffix)
    
    #####################
    ### Run lbann
    trainer = lbann.Trainer(mini_batch_size=batchsize,random_seed=random_seed,callbacks=lbann.CallbackCheckpoint(checkpoint_dir='chkpt', 
  checkpoint_epochs=1))  
#    checkpoint_steps=step_interval))
    
    model = construct_model(num_epochs,mcr,save_batch_interval=int(step_interval)) #'step_interval*val_ratio' is the step interval for validation set.
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = construct_data_reader(data_pct,val_ratio)
    
    status = lbann.run(trainer,model, data_reader, opt,
                       nodes=num_nodes, procs_per_node=num_procs, 
                       work_dir=work_dir,
                       scheduler='slurm', time_limit=1440, setup_only=False)
    
    print(status)

