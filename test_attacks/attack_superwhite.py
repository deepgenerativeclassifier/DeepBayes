from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import logging, os, sys, pickle, argparse
sys.path.append('../utils/')
sys.path.append('../cleverhans/')
from cleverhans.utils import set_log_level
from model_eval import model_eval

import keras.backend
sys.path.append('load/')
from load_classifier import load_classifier
import pickle

from detect_attacks_logp import search_alpha, comp_logp, logsumexp

from superwhite import SuperWhite

FLAGS = flags.FLAGS
    
def combine(logits, combine_logits):
    # combine logits of shape (K, N, dimY) to shape (N, dimY)
    print('combine the logits from random network snapshots (%s)...' % combine_logits)
    if combine_logits == 'ensemble':
        results = tf.reduce_mean(tf.nn.softmax(logits), 0)	# (N, dimY)
        results = tf.log(tf.clip_by_value(results, 1e-20, np.inf))
    if combine_logits == 'bayes': 
        logits_max = tf.reduce_max(logits, 0)
        logits_ = logits - logits_max	# (dimY, N)
        results = tf.log(tf.clip_by_value(tf.reduce_mean(tf.exp(logits_), 0), 1e-20, np.inf))
        results += logits_max

    return results

def test_attacks(data_name, model_name, attack_method, eps, lbd, batch_size=100, 
                 targeted=False, attack_snapshot=False, save=False):

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")
    set_log_level(logging.DEBUG)

    if data_name == 'mnist':
        from cleverhans.utils_mnist import data_mnist
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=0, train_end=60000,
                                                      test_start=0, test_end=10000)
    if data_name in ['cifar10', 'plane_frog']:
        from import_data_cifar10 import load_data_cifar10
        if data_name == 'plane_frog':
            labels = [0, 6]
        else:
            labels = None
        data_path = '../cifar_data/'
        X_train, X_test, Y_train, Y_test = load_data_cifar10(data_path, labels=labels, conv=True)
    
    source_samples, img_rows, img_cols, channels = X_test.shape
    nb_classes = Y_test.shape[1]

    # Define input TF placeholder
    batch_size = min(batch_size, X_test.shape[0])
    print('use batch_size = %d' % batch_size)
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes))

    # Define TF model graph
    model = load_classifier(sess, model_name, data_name, attack_snapshot=attack_snapshot)
    if 'bayes' in model_name and 'distill' not in model_name and 'query' not in model_name:
        model_name = model_name + '_cnn'

    # Craft adversarial examples
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    if 'bnn' not in model_name:
        keras.backend.set_learning_phase(0)
    else:
        # need to set keras learning phase to training in order to run test-time dropout
        keras.backend.set_learning_phase(1)

    # make adv inputs and labels for the attack if targeted
    if targeted:
        adv_inputs = np.array(
                [[instance] * nb_classes for
                 instance in X_test[:source_samples]], dtype=np.float32)
        one_hot = np.zeros((nb_classes, nb_classes))
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1
        adv_inputs = adv_inputs.reshape(
            (source_samples * nb_classes, img_rows, img_cols, 1))
        adv_ys = np.array([one_hot] * source_samples,
                          dtype=np.float32).reshape((source_samples *
                                                     nb_classes, nb_classes))
    else:
        adv_inputs = X_test[:source_samples]
        adv_ys = Y_test[:source_samples]

    model_logit = lambda x: model.predict(x, softmax=False)
    attack = SuperWhite(model_logit, sess=sess)
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    if 'bnn' in model_name:
        combine_logits = 'ensemble'
    if 'bayes' in model_name:
        combine_logits = 'bayes'
    attack_params = {yname: adv_ys,
                    'eps': eps,
                    'eps_iter': 0.01,
                    'nb_iter': 40,
                    'clip_min': 0., 
                    'clip_max': 1.,
                    'rand_init': True,
                    'delta_marginal': 0,
                    'delta_logit': 0,
                    'delta_kl': 0,
                    'kl_prob_vec': None,
                    'detection_lambda': lbd,
                    'combine_logits': combine_logits,
                    'batch_size': batch_size} 

    # compute statistics on data   
    y_logit_train = []
    print('-------------------------------------')
    print('compute statistics on data') 
    y_logit_op = model.predict(x, softmax=False)
    if attack_snapshot:
        y_logit_op = combine(y_logit_op, combine_logits)
    for i in xrange(int(X_train.shape[0] / batch_size)):
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_logit_train.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
    y_logit_train = np.concatenate(y_logit_train)
    y_train = Y_train[:y_logit_train.shape[0]]
    acc_train = np.mean(np.argmax(y_logit_train, 1) == np.argmax(y_train, 1))
    print('training set accuracy:', 100 * acc_train)
    results_train = comp_logp(y_logit_train, y_train, 'train', comp_logit_dist = True)

    # marginal detection
    alpha, _ = search_alpha(results_train[0], results_train[1], results_train[2], plus=False)
    delta_marginal = -(results_train[1] - alpha * results_train[2])
    print('delta_marginal:', delta_marginal)

    # logit detection
    delta_logit = []
    for i in xrange(nb_classes):
        ind = np.where(y_train[:, i] == 1)[0]
        alpha, _ = search_alpha(results_train[3][ind], results_train[4][i], results_train[5][i], plus=False)
        delta_logit.append(-(results_train[4][i] - alpha * results_train[5][i]))
    delta_logit = np.asarray(delta_logit, dtype='f') 
    print('delta_logit:', delta_logit)

    # kl detection
    logit_mean, _, kl_mean, kl_std, softmax_mean = results_train[-5:]
    delta_kl = []
    for i in xrange(nb_classes):
        ind = np.where(y_train[:, i] == 1)[0]
        logit_tmp = y_logit_train[ind] - logsumexp(y_logit_train[ind], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        alpha, _ = search_alpha(kl, kl_mean[i], kl_std[i], plus=True)
        delta_kl.append(kl_mean[i] + alpha * kl_std[i])
    delta_kl = np.asarray(delta_kl, dtype='f')
    print('delta_kl:', delta_kl)

    # add in params. to attack_params
    attack_params['delta_marginal'] = delta_marginal
    attack_params['delta_logit'] = delta_logit
    attack_params['delta_kl'] = delta_kl
    attack_params['kl_prob_vec'] = np.array(softmax_mean)

    # perform the attack!
    adv = []
    n_batch = int(adv_inputs.shape[0] / batch_size)
    for i in xrange(n_batch):
        adv_batch = adv_inputs[i*batch_size:(i+1)*batch_size]
        attack_params[yname] = adv_ys[i*batch_size:(i+1)*batch_size]	# only for untargeted
        adv.append(attack.generate_np(adv_batch, **attack_params))
        if (i+1) % 10 == 0:
            print('finished %d/%d mini-batch' % (i+1, n_batch))
    adv = np.concatenate(adv, axis=0)

    print('--------------------------------------')
   
    # evaluations
    preds = model.predict(x, softmax=False)	# output logits
    if attack_snapshot:
        preds = combine(preds, combine_logits)
    eval_params = {'batch_size': batch_size}
    accuracy, adv_logits = model_eval(sess, x, y, preds, adv, adv_ys, 
                                      args=eval_params, return_pred=True)
    if targeted:
        success_rate = accuracy * 100
        print('untargeted attack success rate: %.4f' % success_rate)
    else:
        success_rate = (1 - accuracy) * 100
        print('untargeted attack success rate: %.4f' % success_rate, adv.shape)
 
    # Close TF session
    sess.close()
    
    # save results
    if save:    
        if not os.path.isdir('raw_attack_results_superwhite/'):
            os.mkdir('raw_attack_results_superwhite/')
            print('create path raw_attack_results_superwhite/')
        path = 'raw_attack_results_superwhite/' + model_name + '/'
        attack_method = attack_method + '_' + 'eps%.2f' % attack_params['eps']
        attack_method = attack_method + '_lambda%.1f' % attack_params['detection_lambda']
        if not os.path.isdir(path):
            os.mkdir(path); print('create path ' + path)
        filename = data_name + '_' + attack_method
        if targeted:
            filename = filename + '_targeted'
        else:
            filename = filename + '_untargeted'
        true_ys = Y_test[:source_samples]
        results = [adv, true_ys, adv_ys, adv_logits]
        pickle.dump(results, open(path+filename+'.pkl', 'w'))
        print("results saved at %s.pkl" % (path+filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='plane_frog')
    parser.add_argument('--targeted', '-T', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='superwhite')
    parser.add_argument('--eps', '-e', type=float, default=0.1)
    parser.add_argument('--lbd', '-l', type=float, default=0.1)
    parser.add_argument('--victim', '-V', type=str, default='bnn_K10')
    parser.add_argument('--save', '-S', action='store_true', default=False)
    parser.add_argument('--snapshot', '-R', action='store_true', default=False)
    
    args = parser.parse_args()
    test_attacks(data_name=args.data,
                 model_name=args.victim,
                 attack_method=args.attack,
                 eps=args.eps,
                 lbd=args.lbd,
                 batch_size=args.batch_size,
                 targeted=args.targeted,
                 attack_snapshot=args.snapshot,
                 save=args.save)

