from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/', 'utils/'])
from utils import load_data, save_params, load_params, init_variables
from visualisation import plot_images
from vae_new import construct_optimizer

n_iter = 100
batch_size = 50
lr = 1e-4#5e-5
K = 1
checkpoint = -1
data_path = 'cifar_data/'

# build generative classifiers on features extracted by a good CIFAR10 classifier

def main(data_name, vae_type, fea_layer, n_iter, batch_size, K, checkpoint, data_path):
    # load data
    from import_data_cifar10 import load_data_cifar10
    X_train, X_test, Y_train, Y_test = load_data_cifar10(data_path, conv=True)
    dimY = Y_train.shape[1]

    if vae_type == 'E': 
        from mlp_generator_cifar10_E import generator
    if vae_type == 'F': 
        from mlp_generator_cifar10_F import generator
    if vae_type == 'G': 
        from mlp_generator_cifar10_G import generator
    from mlp_encoder_cifar10 import encoder_gaussian as encoder

    #first build the feature extractor
    input_shape = X_train[0].shape
    sys.path.append('test_attacks/load/')
    from vgg_cifar10 import cifar10vgg
    cnn = cifar10vgg(path='test_attacks/load/vgg_model/', train=False)

    if fea_layer == 'low':
        N_layer = 16
    if fea_layer == 'mid':
        N_layer = 36
    if fea_layer == 'high':
        N_layer = len(cnn.model.layers) - 5
    for layer in cnn.model.layers:
        print(layer.__class__.__name__)
    def feature_extractor(x):
        out = cnn.normalize_production(x * 255.0)
        for i in range(N_layer):
            out = cnn.model.layers[i](out)
        return out
    print(fea_layer, N_layer, cnn.model.layers[N_layer-1].__class__.__name__, \
          cnn.model.layers[N_layer-1].get_config())

    # then define model
    X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)
    Y_ph = tf.placeholder(tf.float32, shape=(batch_size, dimY))
    dimZ = 128#32
    dimH = 1000
    fea_op = feature_extractor(X_ph)
    if len(fea_op.get_shape().as_list()) == 4:
        fea_op = tf.reshape(fea_op, [batch_size, -1])
    dimF = fea_op.get_shape().as_list()[-1]
    dec = generator(dimF, dimH, dimZ, dimY, 'linear', 'gen')
    n_layers_enc = 2
    enc = encoder(dimF, dimH, dimZ, dimY, n_layers_enc, 'enc')

    ll = 'l2'
    identity = lambda x: x
    fea_ph = tf.placeholder(tf.float32, shape=(batch_size, dimF))
    fit, eval_acc = construct_optimizer(fea_ph, Y_ph, [identity, enc], dec, ll, K, vae_type)

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    path_name = data_name + '_conv_vae_fea_%s_%s/' % (vae_type, fea_layer)
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

    # set test phase
    import keras.backend
    keras.backend.set_session(sess)
    cnnfile = 'test_attacks/load/vgg_model/cifar10vgg.h5'
    cnn.model.load_weights(cnnfile)
    print('load weight from', cnnfile)
    keras.backend.set_learning_phase(0)

    # extract features
    def gen_feature(X):
        F = []
        for i in range(int(X.shape[0] / batch_size)):
            batch = X[i*batch_size:(i+1)*batch_size] 
            F.append(sess.run(fea_op, feed_dict={X_ph: batch}))
        return np.concatenate(F, axis=0)
    F_train = gen_feature(X_train)
    F_test = gen_feature(X_test)

    # now start fitting
    beta = 1.0 
    n_iter_ = 10
    for i in range(int(n_iter/n_iter_)):
        fit(sess, F_train, Y_train, n_iter_, lr, beta)
        # print training and test accuracy
        eval_acc(sess, F_test, Y_test, 'test', beta)

    # save param values
    save_params(sess, filename, checkpoint, scope = 'vae') 
    checkpoint += 1

if __name__ == '__main__':
    data_name = 'cifar10'
    vae_type = str(sys.argv[1])
    fea_layer = str(sys.argv[2]) 
    main(data_name, vae_type, fea_layer, n_iter, batch_size, K, checkpoint, data_path)
    
