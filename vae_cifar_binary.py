from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/', 'utils/'])
from utils import load_data, save_params, load_params, init_variables
from visualisation import plot_images
from vae_new import construct_optimizer

dimZ = 128#32
dimH = 1000
n_iter = 200
batch_size = 50
lr = 5e-5
K = 1
checkpoint = -1
data_path = 'cifar_data/'

def main(data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint, data_path):
    # load data
    from import_data_cifar10 import load_data_cifar10
    if data_name == 'plane_frog':
        labels = [0, 6]
    X_train, X_test, Y_train, Y_test = load_data_cifar10(data_path, labels=labels, conv=True)
    dimY = Y_train.shape[1]

    if vae_type == 'A': 
        from conv_generator_cifar10_A import generator
    if vae_type == 'B': 
        from conv_generator_cifar10_B import generator
    if vae_type == 'C': 
        from conv_generator_cifar10_C import generator
    if vae_type == 'D': 
        from conv_generator_cifar10_D import generator
    if vae_type == 'E': 
        from conv_generator_cifar10_E import generator
    if vae_type == 'F': 
        from conv_generator_cifar10_F import generator
    if vae_type == 'G': 
        from conv_generator_cifar10_G import generator
    from conv_encoder_cifar10 import encoder_gaussian as encoder
    shape_high = (32, 32)
    input_shape = (32, 32, 3)
    n_channel = 64

    # then define model
    dec = generator(input_shape, dimH, dimZ, dimY, n_channel, 'sigmoid', 'gen')
    enc, enc_conv, enc_mlp = encoder(input_shape, dimH, dimZ, dimY, n_channel, 'enc')
    
    # define optimisers
    X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)
    Y_ph = tf.placeholder(tf.float32, shape=(batch_size, dimY))
    ll = 'l2'
    fit, eval_acc = construct_optimizer(X_ph, Y_ph, [enc_conv, enc_mlp], dec, ll, K, vae_type)

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    path_name = data_name + '_conv_vae_%s/' % (vae_type + '_' + str(dimZ))
    if not os.path.isdir('save/'+path_name):
        os.mkdir('save/'+path_name)
        print('create path save/' + path_name)
    filename = 'save/' + path_name + 'checkpoint'
    if checkpoint < 0:
        print('training from scratch')
        init_variables(sess)
    else:
        load_params(sess, filename, checkpoint)
    checkpoint += 1
  
    # now start fitting
    beta = 1.0 
    n_iter_ = 10 
    for i in range(int(n_iter/n_iter_)):
        fit(sess, X_train, Y_train, n_iter_, lr, beta)
        # print training and test accuracy
        eval_acc(sess, X_test, Y_test, 'test', beta)

    # save param values
    save_params(sess, filename, checkpoint) 
    checkpoint += 1

if __name__ == '__main__':
    data_name = 'plane_frog'
    vae_type = str(sys.argv[1]) 
    main(data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint, data_path)
    
