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
sys.path.append('../utils/')
sys.path.append('../cleverhans/')
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, tf_model_load
from model_eval import model_eval

import keras.backend
sys.path.append('load/')
from load_classifier import load_classifier

FLAGS = flags.FLAGS

def test_attacks(data_name, batch_size=128, source_samples=10,
                 model_path=os.path.join("models", "mnist"), targeted=True):
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

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
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                      train_end=60000,
                                                      test_start=0,
                                                      test_end=10000)
    if data_name in ['cifar10', 'plane_frog']:
        from import_data_cifar10 import load_data_cifar10
        labels = None
        if data_name == 'plane_frog':
            labels = [0, 6]
        datapath = '../cifar_data/'
        X_train, X_test, Y_train, Y_test = load_data_cifar10(datapath, labels=labels)

    img_rows, img_cols, channels = X_test[0].shape
    nb_classes = Y_test.shape[1]

    # Define input TF placeholder
    batch_size = min(batch_size, source_samples)
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes))

    # Define TF model graph
    model_name = str(sys.argv[1])
    model = load_classifier(sess, model_name, data_name)
    if 'bayes' in model_name:
        model_name = model_name + '_cnn'

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    if 'bnn' not in model_name:
        keras.backend.set_learning_phase(0)
    else:
        keras.backend.set_learning_phase(1)

    preds = model.predict(x, softmax=False)     # output logits
    eval_params = {'batch_size': batch_size}
    accuracy, y_pred_clean = model_eval(sess, x, y, preds, X_test, Y_test,
                                        args=eval_params, return_pred=True)
    print('Test accuracy on legitimate test examples: %.2f' % (accuracy*100))
    report.clean_train_clean_eval = accuracy
    y_pred_clean = y_pred_clean[:Y_test.shape[0]]
    correct_prediction = (np.argmax(Y_test, 1) == np.argmax(y_pred_clean, 1))
    ind = np.where(correct_prediction==1)[0]
    print('crafting adversarial examples only on correctly prediced images...')
    print('%d / %d in total' % (len(ind), X_test.shape[0]))

    path = 'data_ind/'
    if not os.path.isdir(path):
        os.mkdir(path)
        print('create path ' + path)
    filename = data_name + '_' + model_name
    import pickle
    pickle.dump(ind, open(path+filename+'.pkl', 'wb'))
    print("results saved at %s.pkl" % (path+filename))

    return report


def main(argv=None):
    test_attacks(data_name=str(sys.argv[-1]),
                 batch_size=FLAGS.batch_size,
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

