from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/', 'utils/'])
from utils import load_data, save_params, load_params, init_variables
from visualisation import plot_images
from vae_new import construct_optimizer

dimZ = 64
dimH = 500
n_iter = 100
batch_size = 50
lr = 1e-4
K = 1
checkpoint = -1

def main(data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint):
    dimY = 10

    if vae_type == 'A': 
        from conv_generator_mnist_A import generator
    if vae_type == 'B': 
        from conv_generator_mnist_B import generator
    if vae_type == 'C': 
        from conv_generator_mnist_C import generator
    if vae_type == 'D': 
        from conv_generator_mnist_D import generator
    if vae_type == 'E': 
        from conv_generator_mnist_E import generator
    if vae_type == 'F': 
        from conv_generator_mnist_F import generator
    if vae_type == 'G': 
        from conv_generator_mnist_G import generator
    from conv_encoder_mnist import encoder_gaussian as encoder
    shape_high = (28, 28)
    input_shape = (28, 28, 1)
    n_channel = 64

    # then define model
    dec = generator(input_shape, dimH, dimZ, dimY, n_channel, 'sigmoid', 'gen')
    enc, enc_conv, enc_mlp = encoder(input_shape, dimH, dimZ, dimY, n_channel, 'enc')
    
    # define optimisers
    X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)
    Y_ph = tf.placeholder(tf.float32, shape=(batch_size, dimY))
    ll = 'l2'
    fit, eval_acc = construct_optimizer(X_ph, Y_ph, [enc_conv, enc_mlp], dec, ll, K, vae_type)
    
    # load data
    from utils_mnist import data_mnist
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0, train_end=60000,
                                                  test_start=0, test_end=10000)
 
    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if not os.path.isdir('save/'):
        os.mkdir('save/')
        print('create path save/')
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
    n_iter_ = min(n_iter,20)
    beta = 1.0
    for i in range(int(n_iter/n_iter_)):
        fit(sess, X_train, Y_train, n_iter_, lr, beta)
        # print training and test accuracy
        eval_acc(sess, X_test, Y_test, 'test')

    # save param values
    save_params(sess, filename, checkpoint) 
    checkpoint += 1

if __name__ == '__main__':
    data_name = 'mnist'
    vae_type = str(sys.argv[1]) 
    main(data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint)
    
