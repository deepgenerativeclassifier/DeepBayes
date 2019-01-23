from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os, sys
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, tf_model_load
from model_eval import model_eval

FLAGS = flags.FLAGS

def test_attacks(batch_size=128, source_samples=10,
                 model_path=os.path.join("models", "mnist"), targeted=True):
    """
    Test many attacks on MNIST with deep Bayes classifier.
    :param batch_size: size of training batches
    :param source_samples: number of test inputs to attack
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    from cleverhans.utils_mnist import data_mnist
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    img_rows, img_cols, channels = X_train[0].shape
    nb_classes = Y_train.shape[1]

    # Define input TF placeholder
    batch_size = min(batch_size, source_samples)
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes))

    # Define TF model graph  
    model_name = str(sys.argv[1])  
    if model_name == 'bayes':
        from load_bayes_classifier import BayesModel
        conv = True
        checkpoint = 0#int(sys.argv[1])
        K = int(sys.argv[3])
        use_mean = True
        model = BayesModel(sess, 'mnist', conv, K, checkpoint=checkpoint, 
                           attack_snapshot=False, use_mean=use_mean)   
        if use_mean:
            model_name = 'bayes_mean_mlp'
        else: 
            model_name = 'bayes_K%d' % K
    if model_name == 'cnn':
        from load_cnn_classifier import CNNModel
        model = CNNModel(sess, 'mnist')
    if model_name == 'wgan':
        from load_wgan_classifier import WGANModel
        conv = True
        checkpoint = 0#int(sys.argv[1])
        K = int(sys.argv[3])
        T = int(sys.argv[4])
        model = WGANModel(sess, 'mnist', conv, K, T, checkpoint=checkpoint)
        model_name = 'wgan_K%d_T%d' % (K, T)
                       
    preds = model.predict(x, softmax=True)	# output probabilities
    print("Defined TensorFlow model graph.")

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    # Craft adversarial examples
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

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
    attack_method = str(sys.argv[2])
    if attack_method == 'fgsm':
        from cleverhans.attacks import FastGradientMethod 
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = FastGradientMethod(model_prob, sess=sess)
        from attack_config import config_fgsm
        attack_params = config_fgsm(targeted, adv_ys)
    if attack_method == 'bim':
        from cleverhans.attacks import BasicIterativeMethod
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = BasicIterativeMethod(model_prob, sess=sess)
        from attack_config import config_bim
        attack_params = config_bim(targeted, adv_ys)
    if attack_method == 'mim':
        from cleverhans.attacks import MomentumIterativeMethod
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = MomentumIterativeMethod(model_prob, sess=sess)
        from attack_config import config_mim
        attack_params = config_mim(targeted, adv_ys)
    if attack_method == 'jsma':
        from cleverhans.attacks import SaliencyMapMethod 
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = SaliencyMapMethod(model_prob, sess=sess)
        from attack_config import config_jsma
        attack_params = config_jsma(targeted, adv_ys)
    if attack_method == 'vat':
        from cleverhans.attacks import VirtualAdversarialMethod 
        model_logit = lambda x: model.predict(x, softmax=False)
        attack = VirtualAdversarialMethod(model_logit, sess=sess)
        from attack_config import config_vat
        attack_params = config_vat(targeted, adv_ys)
    if attack_method == 'cw':
        from cleverhans.attacks import CarliniWagnerL2
        model_logit = lambda x: model.predict(x, softmax=False)
        attack = CarliniWagnerL2(model_logit, sess=sess)
        from attack_config import config_cw
        attack_params = config_cw(targeted, adv_ys)
    if attack_method == 'elastic':
        from cleverhans.attacks import ElasticNetMethod
        model_logit = lambda x: model.predict(x, softmax=False)
        attack = ElasticNetMethod(model_logit, sess=sess)
        from attack_config import config_elastic
        attack_params = config_elastic(targeted, adv_ys)
    if attack_method == 'deepfool':
        from cleverhans.attacks import DeepFool
        model_logit = lambda x: model.predict(x, softmax=False)
        attack = DeepFool(model_logit, sess=sess)
        from attack_config import config_deepfool
        attack_params = config_deepfool(targeted, adv_ys)
    if attack_method == 'madry':
        from cleverhans.attacks import MadryEtAl
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = MadryEtAl(model_prob, sess=sess)
        from attack_config import config_madry
        attack_params = config_madry(targeted, adv_ys)
    
    attack_params['batch_size'] = batch_size
    print('batchsize', batch_size)
   
    # perform the attack!
    adv = []
    n_batch = int(adv_inputs.shape[0] / batch_size)
    for i in xrange(n_batch):
        adv_batch = adv_inputs[i*batch_size:(i+1)*batch_size]
        adv.append(attack.generate_np(adv_batch, **attack_params))
    adv = np.concatenate(adv, axis=0)
    
    for _ in xrange(5):
        y_adv = []
        for i in xrange(n_batch):    
            adv_batch = adv[i*batch_size:(i+1)*batch_size]
            y_adv.append(sess.run(preds, {x: adv_batch}))
        y_adv = np.concatenate(y_adv, axis=0)
    
        print('--------------------------------------')
        for i in xrange(10):
            print(np.argmax(y_adv[i*10:(i+1)*10], 1))
    
    correct_pred = np.asarray(np.argmax(y_adv, 1) == np.argmax(adv_ys, 1), dtype='f')
    adv_accuracy = np.mean(correct_pred)

    if not targeted:
#        adv_accuracy, y_adv = model_eval(sess, x, y, preds, adv, 
#                                         adv_ys, args=eval_params,
#                                         return_pred=True)
#    else:
#        adv_accuracy, y_adv = model_eval(sess, x, y, preds, adv, 
#                                         Y_test[:source_samples], args=eval_params,
#                                         return_pred=True)
        adv_accuracy = 1. - adv_accuracy

    print('--------------------------------------')
    
    print(np.argmax(adv_ys[:10], 1))
    print(np.argmax(y_adv[:10], 1))
    for i in xrange(5):
        tmp = sess.run(preds, {x: adv[:100]})
        print(np.argmax(tmp[:10], 1))

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()
    
    # visualisation
    vis_adv = True
    if vis_adv:
        N_vis = 100
        sys.path.append('../../utils')
        from visualisation import plot_images
        if channels == 1:
            shape = (img_rows, img_cols)
        else:
            shape = (img_rows, img_cols, channels)
        path = 'figs/'
        filename = model_name + '_' + attack_method
        if targeted:
            filename = filename + '_targeted'
        else:
            filename = filename + '_untargeted'
        plot_images(adv_inputs[:N_vis], shape, path, filename+'_data')
        plot_images(adv[:N_vis], shape, path, filename+'_adv')
        
    save_result = True
    if save_result:
        path = 'results/'
        filename = model_name + '_' + attack_method
        if targeted:
            filename = filename + '_targeted'
            y_input = adv_ys
        else:
            filename = filename + '_untargeted'
            y_input = Y_test[:source_samples]
        results = [adv_inputs, y_input, adv, y_adv]
        import pickle
        pickle.dump(results, open(path+filename+'.pkl', 'w'))
        print("results saved at %s.pkl" % filename)

    return report


def main(argv=None):
    test_attacks(batch_size=FLAGS.batch_size,
                 source_samples=FLAGS.source_samples,
                 model_path=FLAGS.model_path,
                 targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
    flags.DEFINE_integer('source_samples', 10000, 'Nb of test inputs to attack')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
    
