from __future__ import print_function

import numpy as np
import tensorflow as tf
from mlp import mlp_layer
from convnet import ConvNet, construct_filter_shapes

"""
generator p(x)p(z|x)p(y|z), DBX
note here this is actually a discriminative model: we assume p(x) = p_D(x)
"""
   
def deconv_layer(output_shape, filter_shape, activation, strides, name):
    scale = 1.0 / np.prod(filter_shape[:3])
    seed = int(np.random.randint(0, 1000))#123
    W = tf.Variable(tf.random_uniform(filter_shape, 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed), name = name+'_W')
    
    def apply(x):
        output_shape_x = (x.get_shape().as_list()[0],)+output_shape
        a = tf.nn.conv2d_transpose(x, W, output_shape_x, strides, 'SAME')
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a
        if activation == 'split':
            x1, x2 = tf.split(a, 2, 3)	# a is a 4-D tensor
            return tf.nn.sigmoid(x1), x2
            
    return apply

def generator(input_shape, dimH, dimZ, dimY, n_channel, last_activation, name,
              decoder_input_shape = None, layer_channels = None):

    # first construct p(y|z)
    fc_layers = [dimZ, dimH, dimY]
    pyz_mlp_layers = []
    N_layers = len(fc_layers) - 1
    l = 0
    for i in range(N_layers):
        name_layer = name + '_pyz_mlp_l%d' % l
        if i+1 == N_layers:
            activation = 'linear'
        else:
            activation = 'relu'
        pyz_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))

    def pyz_params(z):
        out = z
        for layer in pyz_mlp_layers:
            out = layer(out)
        return out
    
    # now construct p(z|x)
    # encoder for z (low res)
    if layer_channels is None:
        layer_channels = [n_channel, n_channel*2, n_channel*4]
    filter_width = 5
    filter_shapes = construct_filter_shapes(layer_channels, filter_width)
    fc_layer_sizes = [dimH]
    gen_conv2, conv_output_shape = ConvNet(name+'_pzx_conv', input_shape, filter_shapes, \
                                     fc_layer_sizes, 'relu',
                                     last_activation = 'relu')
    print('generator shared Conv net ' + ' network architecture:', \
            conv_output_shape, fc_layer_sizes)

    fc_layers = [dimH, dimH, dimZ*2]
    pzx_mlp_layers = []
    N_layers = len(fc_layers) - 1
    l = 0
    for i in range(N_layers):
        name_layer = name + '_pzx_mlp_l%d' % l
        if i+1 == N_layers:
            activation = 'linear'
        else:
            activation = 'relu'
        pzx_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))

    def pzx_params(x):
        out = gen_conv2(x)
        for layer in pzx_mlp_layers:
            out = layer(out)
        mu, log_sig = tf.split(out, 2, axis=1)
        return mu, log_sig

    return pyz_params, pzx_params

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())
    
