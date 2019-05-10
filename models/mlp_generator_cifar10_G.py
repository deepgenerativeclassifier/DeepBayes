import numpy as np
import tensorflow as tf
from mlp import mlp_layer

"""
generator p(y)p(z|y)p(x|z, y)
"""
   
def generator(dimX, dimH, dimZ, dimY, last_activation, name):
    # first construct p(z|y)
    fc_layers = [dimY, dimH, dimZ*2]
    pzy_mlp_layers = []
    N_layers = len(fc_layers) - 1
    l = 0
    for i in range(N_layers):
        name_layer = name + '_pzy_l%d' % l
        if i+1 == N_layers:
            activation = 'linear'
        else:
            activation = 'relu'
        with tf.variable_scope('vae'):
            pzy_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))

    def pzy_params(y):
        out = y
        for layer in pzy_mlp_layers:
            out = layer(out)
        mu, log_sig = tf.split(out, 2, axis=1)
        return mu, log_sig

    # now construct p(x|z)
    fc_layers = [dimZ, dimH, dimH, dimX]
    l = 0
    pxz_mlp_layers = []
    N_layers = len(fc_layers) - 1
    for i in range(N_layers):
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = last_activation
        name_layer = name + '_pxz_mlp_l%d' % l
        with tf.variable_scope('vae'):
            pxz_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))
        l += 1
    
    def pxz_params(z):
        out = z 
        for layer in pxz_mlp_layers:
            out = layer(out)
        return out
    
    return pzy_params, pxz_params

