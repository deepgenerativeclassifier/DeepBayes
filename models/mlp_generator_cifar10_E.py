import numpy as np
import tensorflow as tf
from mlp import mlp_layer

"""
generator p_D(x)p(z|x)p(y|z), note here this is actually a discriminative model
"""
   
def generator(dimX, dimH, dimZ, dimY, last_activation, name):

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
        with tf.variable_scope('vae'):
            pyz_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))

    def pyz_params(z):
        out = z
        for layer in pyz_mlp_layers:
            out = layer(out)
        return out
    
    # now construct p(z|x)
    fc_layers = [dimX, dimH, dimH, dimZ*2]
    l = 0
    pzx_mlp_layers = []
    N_layers = len(fc_layers) - 1
    for i in range(N_layers):
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = 'linear'
        name_layer = name + '_pzx_mlp_l%d' % l
        with tf.variable_scope('vae'):
            pzx_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))
        l += 1

    def pzx_params(x):
        out = x
        for layer in pzx_mlp_layers:
            out = layer(out)
        mu, log_sig = tf.split(out, 2, axis=1)
        return mu, log_sig

    return pyz_params, pzx_params

