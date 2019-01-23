from __future__ import print_function

import os
import numpy as np
import time
import pickle
import tensorflow as tf

def load_data(data_name, path, labels = None, conv = False, seed = 0):
    if data_name == 'mnist':
        from import_data_mnist import load_data_mnist
        data_train, data_test, labels_train, labels_test = \
            load_data_mnist(path, labels, conv, seed)
            
    if data_name == 'omni':
        from import_data_omni import load_data_omni
        data_train, data_test, labels_train, labels_test = \
            load_data_omni(path, labels, conv, seed)
            
    if data_name == 'cifar10':
        from import_data_cifar10 import load_data_cifar10
        data_train, data_test, labels_train, labels_test = \
            load_data_cifar10(path, labels, conv, seed)
        
    return data_train, data_test, labels_train, labels_test

def init_variables(sess, old_var_list = set([])):
    all_var_list = set(tf.all_variables())
    init = tf.initialize_variables(var_list = all_var_list - old_var_list)
    sess.run(init)
    return all_var_list
    
def save_params(sess, filename, checkpoint, scope=None):
    params = tf.trainable_variables()
    if scope is not None:
        params = [v for v in params if scope in v.name]
    param_dict = dict()
    for v in params:
        param_dict[v.name] = sess.run(v)
    filename = filename + '_' + str(checkpoint)
    f = open(filename + '.pkl', 'wb')
    pickle.dump(param_dict, f)
    print('parameters saved at ' + filename + '.pkl')    
    f.close()

def load_params(sess, filename, checkpoint):
    params = tf.trainable_variables()
    filename = filename + '_' + str(checkpoint)
    f = open(filename + '.pkl', 'rb')
    param_dict = pickle.load(f)
    print('param loaded', len(param_dict))
    f.close()
    ops = []
    var_to_init = []
    for v in params:
        if v.name in param_dict.keys():
            ops.append(tf.assign(v, param_dict[v.name]))
        else:
            var_to_init.append(v)
    print('assign to %d tensors..' % len(ops))
    sess.run(ops)
    # init uninitialised params
    all_var = tf.global_variables()
    var = [v for v in all_var if v not in params]
    var_to_init = var_to_init + var
    print('no. of uninitialised variables', len(var_to_init), len(params))
    sess.run(tf.initialize_variables(var_to_init))
    print('loaded parameters from ' + filename + '.pkl')
