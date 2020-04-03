import ExaGAN
import argparse
#import dataset
#import lbann.contrib.lc.launcher
import lbann
# ==============================================
# Setup and launch experiment
# ==============================================



def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--epochs','-e', type=int, default=10,help='The number of epochs')
    add_arg('--procs','-p',  type=int, default=1,help='The number of processes per node')
    add_arg('--nodes','-n',  type=int, default=1,help='The number of GPU nodes requested')

    return parser.parse_args()

def list2str(l):
    return ' '.join(l)

def construct_model(epochs):
    """Construct LBANN model.

    ExaGAN  model

    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A',name='inp_img')
    
    ### Create expected labels (with label flipping = 0.01)
    prob_flip=0.01
    label_flip_rand = lbann.Uniform(min=0,max=1, neuron_dims='1')
    label_flip_prob = lbann.Constant(value=prob_flip, num_neurons='1')
    ones = lbann.GreaterEqual(label_flip_rand,label_flip_prob, name='is_real')
    zeros = lbann.LogicalNot(ones,name='is_fake')
    
    
    ## Create the noise vector
    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="64", name='noise_vec'),dims='1 64')
    
    ### Creating the GAN object and implementing forward pass for both networks ###
    d1_real, d1_fake, d_adv, gen_img  = ExaGAN.CosmoGAN()(input,z,mcr=True) 
    
    ### Compute Loss
    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,ones],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zeros],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,ones],name='d_adv_bce')
    
    #print('d_adv_bce',d_adv_bce.__dict__)
    
    ### Define Loss (Objective function)
    loss = lbann.ObjectiveFunction([d1_real_bce,d1_fake_bce,d_adv_bce])
    
    ### Define metrics
    # Initialize check metric callback
    metrics = [lbann.Metric(d1_real_bce,name='d_real'),
               lbann.Metric(d1_fake_bce, name='d_fake'),
               lbann.Metric(d_adv_bce,name='gen')]
    
    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
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
    
    ### Define callbacks list
    callbacks=[]
    dump_outputs=True

    callbacks.append(lbann.CallbackPrint())
    callbacks.append(lbann.CallbackTimer())
    callbacks.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers), destination_layers=list2str(dst_layers),batch_interval=1))
    if dump_outputs:
        callbacks.append(lbann.CallbackDumpOutputs(layers='inp_img gen_img_instance1_activation',execution_modes='train validation',\
                                           directory='dump_outs',batch_interval=50,format='npy'))                     
    # Construct model
    mini_batch_size = 64
    num_epochs = epochs
    
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=loss,
                       callbacks=callbacks)

def construct_data_reader():
    """Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.

    """
    import os.path
    import lbann
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
    data_reader.percent_of_data_to_use = 1.0
    data_reader.validation_percent = 0.1
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
    num_epochs,num_nodes,num_procs=args.epochs,args.nodes,args.procs
    
    trainer = lbann.Trainer()
    model = construct_model(num_epochs)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = construct_data_reader()

    status = lbann.run(trainer,model, data_reader, opt,
                       scheduler='slurm',
                       #account='lbpm',
                       nodes=num_nodes,
                       procs_per_node=num_procs,
                       time_limit=1440,
                       setup_only=False,
                       job_name='exagan')
    print(status)
