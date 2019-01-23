from __future__ import print_function

import numpy as np
import tensorflow as tf
from mlp import mlp_layer
from convnet import ConvNet, construct_filter_shapes

"""
generator p(x)p(z|x)p(y|x, z), DFX
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

def generator(input_shape, dimH, dimZ, dimY, n_channel, last_activation, name):

    # first construct p(y|z, x)
    # encoder for z (low res)
    layer_channels = [n_channel for i in range(3)]
    filter_width = 5
    filter_shapes = construct_filter_shapes(layer_channels, filter_width)
    fc_layer_sizes = [dimH]
    gen_conv, conv_output_shape = ConvNet(name+'_pyzx_conv', input_shape, filter_shapes, \
                                     fc_layer_sizes, 'relu',
                                     last_activation = 'relu')
    print('generator shared Conv net ' + ' network architecture:', \
            conv_output_shape, fc_layer_sizes)

    fc_layers = [dimZ+dimH, dimH, dimY]
    pyzx_mlp_layers = []
    N_layers = len(fc_layers) - 1
    l = 0
    for i in range(N_layers):
        name_layer = name + '_pyzx_mlp_l%d' % l
        if i+1 == N_layers:
            activation = 'linear'
        else:
            activation = 'relu'
        pyzx_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))

    def pyzx_params(z, x):
        fea = gen_conv(x)
        out = tf.concat([fea, z], axis=1)
        for layer in pyzx_mlp_layers:
            out = layer(out)
        return out
    
    # now construct p(z|x)
    # encoder for z (low res)
    layer_channels = [n_channel for i in range(3)]
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

    return pyzx_params, pzx_params

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())
    
