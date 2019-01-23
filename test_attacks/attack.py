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

FLAGS = flags.FLAGS

def test_attacks(data_name, model_name, attack_method, eps, batch_size=100, 
                 targeted=False, save=False):

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
    
    source_samples, img_rows, img_cols, channels = X_test.shape
    nb_classes = Y_test.shape[1]
    # test cw
    #source_samples = 100

    # Define input TF placeholder
    batch_size = min(batch_size, X_test.shape[0])
    print('use batch_size = %d' % batch_size)
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes))

    # Define TF model graph  
    model = load_classifier(sess, model_name, data_name)
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

    # Instantiate an attack object
    from attack_config import load_attack
    attack, attack_params, yname = load_attack(sess, attack_method, model, targeted, 
                                               adv_ys, eps, batch_size)
    print(attack_params)
   
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
    eval_params = {'batch_size': batch_size}
    # test clean acc
    #accuracy, _ = model_eval(sess, x, y, preds, X_test, Y_test, 
    #                                  args=eval_params, return_pred=True)
    #print('clean test acc: %.4f' % (accuracy * 100))
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
        if not os.path.isdir('raw_attack_results/'):
            os.mkdir('raw_attack_results/')
            print('create path raw_attack_results/')
        path = 'raw_attack_results/' + model_name + '/'
        if attack_method in ['fgsm', 'pgd', 'mim']:
            attack_method = attack_method + '_' + 'eps%.2f' % attack_params['eps']
        if attack_method == 'cw':    
            c = attack_params['initial_const']
            lr = attack_params['learning_rate']
            attack_method = attack_method + '_c%.2f_lr%.3f' % (c, lr)
        if not os.path.isdir(path):
            os.mkdir(path); print('create path ' + path)
        filename = data_name + '_' + attack_method
        if targeted:
            filename = filename + '_targeted'
        else:
            filename = filename + '_untargeted'
        true_ys = Y_test[:source_samples]
        results = [adv, true_ys, adv_ys, adv_logits]
        pickle.dump(results, open(path+filename+'.pkl', 'wb'))
        print("results saved at %s.pkl" % (path+filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='plane_frog')
    parser.add_argument('--targeted', '-T', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='pgd')
    parser.add_argument('--eps', '-e', type=float, default=0.1)
    parser.add_argument('--victim', '-V', type=str, default='bnn_K10')
    parser.add_argument('--save', '-S', action='store_true', default=False)
    
    args = parser.parse_args()
    test_attacks(data_name=args.data,
                 model_name=args.victim,
                 attack_method=args.attack,
                 eps=args.eps,
                 batch_size=args.batch_size,
                 targeted=args.targeted,
                 save=args.save)

