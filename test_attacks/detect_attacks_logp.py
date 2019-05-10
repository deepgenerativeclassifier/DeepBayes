from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf

import os, sys, pickle, argparse
sys.path.append('../utils/')
from model_eval import model_eval
from scipy.misc import logsumexp
import keras.backend
sys.path.append('load/')
from load_classifier import load_classifier

def comp_logp(logit, y, text, comp_logit_dist = False):
    logpx = logsumexp(logit, axis=1)
    logpx_mean = np.mean(logpx)
    logpx_std = np.sqrt(np.var(logpx))
    logpxy = np.sum(y * logit, axis=1)
    logpxy_mean = []; logpxy_std = []
    for i in xrange(y.shape[1]):
        ind = np.where(y[:, i] == 1)[0]
        logpxy_mean.append(np.mean(logpxy[ind]))
        logpxy_std.append(np.sqrt(np.var(logpxy[ind])))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' \
          % (text, logpx_mean, logpx_std, np.mean(logpxy_mean), np.mean(logpxy_std)))
    
    results = [logpx, logpx_mean, logpx_std, logpxy, logpxy_mean, logpxy_std]
    # compute distribution of the logits
    if comp_logit_dist:
        logit_mean = []
        logit_std = []
        logit_kl_mean = []
        logit_kl_std = []
        softmax_mean = []
        for i in xrange(y.shape[1]):
            ind = np.where(y[:, i] == 1)[0]
            logit_mean.append(np.mean(logit[ind], 0))
            logit_std.append(np.sqrt(np.var(logit[ind], 0)))

            logit_tmp = logit[ind] - logsumexp(logit[ind], axis=1)[:, np.newaxis]
            softmax_mean.append(np.mean(np.exp(logit_tmp), 0))
            logit_kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
            
            logit_kl_mean.append(np.mean(logit_kl))
            logit_kl_std.append(np.sqrt(np.var(logit_kl)))
        
        results.extend([logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean]) 

    return results

def comp_detect(x, x_mean, x_std, alpha, plus):
    if plus:
        detect_rate = np.mean(x > x_mean + alpha * x_std)
    else:
        detect_rate = np.mean(x < x_mean - alpha * x_std)
    return detect_rate * 100
 
def search_alpha(x, x_mean, x_std, target_rate = 5.0, plus = False):
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
    T = 0
    while np.abs(detect_rate - target_rate) > 0.01 and T < 20:
        if detect_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
        T += 1
    return alpha_now, detect_rate

def test_attacks(batch_size, conv, guard_name, targeted, attack_method, victim_name, data_name, save):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    # Get MNIST test data
    use_data = True
    if use_data:
        if data_name == 'mnist':
            img_rows, img_cols, channels = 28, 28, 1
            from cleverhans.utils_mnist import data_mnist
            x_train, y_train, x_clean, y_clean = data_mnist(train_start=0,
                                                      train_end=60000,
                                                      test_start=0,
                                                      test_end=10000)
        if data_name in ['cifar10', 'plane_frog']:
            img_rows, img_cols, channels = 32, 32, 3
            from import_data_cifar10 import load_data_cifar10
            labels = None
            if data_name == 'plane_frog':
                labels = [0, 6]
            datapath = '../cifar_data/'
            x_train, x_clean, y_train, y_clean = load_data_cifar10(datapath, labels=labels)                        
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes))

    # Define TF model graph
    gen = load_classifier(sess, guard_name, data_name)
    if 'bayes' in guard_name and 'distill' not in guard_name and 'query' not in guard_name:
        vae_type = guard_name[-1]
        guard_name += '_cnn'

    # now perform detection
    path = 'raw_attack_results/' + victim_name + '/'
    print(path)
    assert os.path.isdir(path)
    filename = data_name + '_' + attack_method  
    if targeted:
        filename = filename + '_targeted'
    else:
        filename = filename + '_untargeted'
    filename = path + filename + '.pkl'
    x_adv, _, y_clean, adv_logits = pickle.load(open(filename, 'rb'))

    # for cifar-binary, need to extract test data that all the classifiers agree on
    if data_name == 'plane_frog':
        load_path = 'data_ind/'
        ind = range(x_clean.shape[0])
        classifiers = ['bayes_K10_A_cnn', 'bayes_K10_B_cnn', 'bayes_K10_C_cnn',
                      'bayes_K10_D_cnn', 'bayes_K10_E_cnn', 'bayes_K10_F_cnn', 
                      'bayes_K10_G_cnn']#, 'bnn_K10']
        for c in classifiers:
            filename = load_path + data_name + '_' + c + '.pkl'
            tmp = pickle.load(open(filename, 'rb'))
            ind = list(set(ind) & set(tmp))
        print('crafting adversarial examples only on correctly prediced images...')
        print('%d / %d in total' % (len(ind), x_clean.shape[0]))
        x_clean = x_clean[ind]; y_clean = y_clean[ind]
        print(len(ind), x_adv.shape, adv_logits.shape)
        x_adv = x_adv[ind]; adv_logits = adv_logits[ind]
    print("data loaded from %s, %d samples in total" % (filename, x_adv.shape[0]))
    print(x_clean.shape, x_adv.shape)

    if 'bnn' not in guard_name:
        keras.backend.set_learning_phase(0)
    else:
        keras.backend.set_learning_phase(1)
 
    y_logit_op = gen.predict(x, softmax=False)
    # compute classification
    y_logit_adv = []
    for i in xrange(int(x_adv.shape[0] / batch_size)):
        X_batch = x_adv[i*batch_size:(i+1)*batch_size]
        y_logit_adv.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
    y_logit_adv = np.concatenate(y_logit_adv, 0)
    N_adv_total = y_logit_adv.shape[0]
    x_clean = x_clean[:N_adv_total]; y_clean = y_clean[:N_adv_total]
    x_adv = x_adv[:N_adv_total]; adv_logits = adv_logits[:N_adv_total]

    test_attack = False
    if guard_name != victim_name:
        if guard_name + '_cnn' != victim_name:
            print('test transfer attack: attack crafted on victim model')
            test_attack = True
        if 'distill' in victim_name:
            print('test gray-box attack: attack crafted on a distilled model')
            test_attack = True
    if test_attack:
        # test adversarial example transfer, compute the classification again
        print('test adversarial example transfer from %s to %s' % (victim_name, guard_name))
        y_adv = np.zeros((y_logit_adv.shape[0], nb_classes), dtype=np.float32)
        y_adv[np.arange(y_logit_adv.shape[0]), np.argmax(y_logit_adv, 1)] = 1
        # get index of victim sucessful attacks
        y_adv_victim = np.zeros((adv_logits.shape[0], nb_classes), dtype=np.float32)
        y_adv_victim[np.arange(adv_logits.shape[0]), np.argmax(adv_logits, 1)] = 1  
        correct_prediction = (np.argmax(y_adv_victim, 1) == np.argmax(y_clean, 1))
        ind_success_victim = np.where(correct_prediction==0)[0]
    else:
        y_adv = np.zeros((adv_logits.shape[0], nb_classes), dtype=np.float32)
        y_adv[np.arange(adv_logits.shape[0]), np.argmax(adv_logits, 1)] = 1  

    correct_prediction = (np.argmax(y_adv, 1) == np.argmax(y_clean, 1))
    accuracy = np.mean(correct_prediction)
    success_rate = 100.0 * (1 - accuracy)
    ind_success = np.where(correct_prediction==0)[0]
    if not test_attack:
        ind_success_victim = ind_success
    # compute success rate on successful victim attack
    success_rate_victim = 100.0 * (1 - np.mean( ( np.argmax(y_adv[ind_success_victim], 1) \
                                               == np.argmax(y_clean[ind_success_victim], 1) ) ))
    print("attack success rate (all/victim) = %.4f / %.4f" % (success_rate, success_rate_victim))

    # compute the perturbation on successful attacks
    if len(ind_success) > 0:
        diff = x_adv[ind_success] - x_clean[ind_success]
        l2_diff = np.sqrt(np.sum(diff**2, axis=(1, 2, 3)))
        li_diff = np.max(np.abs(diff), axis=(1, 2, 3))
        l0_diff = np.sum((diff != 0), axis=(1, 2, 3))
        print('preturb for successful attack: L_2 = %.3f +- %.3f' % (np.mean(l2_diff), np.sqrt(np.var(l2_diff))))
        print('preturb for successful attack: L_inf = %.3f +- %.3f' % (np.mean(li_diff), np.sqrt(np.var(li_diff))))
        print('preturb for successful attack: L_0 = %.3f +- %.3f' % (np.mean(l0_diff), np.sqrt(np.var(l0_diff))))
        # confidence of the attack (using entropy)
        tmp_logp = adv_logits - logsumexp(adv_logits, 1)[:, np.newaxis]
        tmp_p = np.exp(tmp_logp)
        print(tmp_logp.mean(), tmp_p.mean())
        entropy = -np.sum(tmp_p * tmp_logp, 1)
        print('entropy successful attack: %.3f +- %.3f' % (np.mean(entropy), np.sqrt(np.var(entropy))))
    else:
        print('no successful attack, abort...')
        return 0
       
    # then compute logit on both clean and adv samples
    y_logit_train = []
    print('-------------------------------------')
    print('compute statistics on data') 
    for i in xrange(int(x_train.shape[0] / batch_size)):
        X_batch = x_train[i*batch_size:(i+1)*batch_size]
        y_logit_train.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
    y_logit_train = np.concatenate(y_logit_train)
    y_train = y_train[:y_logit_train.shape[0]]
    results_train = comp_logp(y_logit_train, y_train, 'train', comp_logit_dist = True)

    y_logit_clean = []
    for i in xrange(int(x_clean.shape[0] / batch_size)):
        X_batch = x_clean[i*batch_size:(i+1)*batch_size]
        y_logit_clean.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
    y_logit_clean = np.concatenate(y_logit_clean, 0)

    # now produce the logits!
    results_clean = comp_logp(y_logit_clean, y_clean, 'clean')
    results_adv = comp_logp(y_logit_adv[ind_success], y_adv[ind_success], 'adv (wrong)')
    tmp_logp = y_logit_adv[ind_success] - logsumexp(y_logit_adv[ind_success], 1)[:, np.newaxis]
    tmp_p = np.exp(tmp_logp)
    entropy = -np.sum(tmp_p * tmp_logp, 1)
    print('entropy on ind_success: %.3f +- %.3f' % (np.mean(entropy), np.sqrt(np.var(entropy))))

    # use mean as rejection
    print("-------------------------------------")
    results = {}
    results['success_rate'] = success_rate
    results['success_rate_victim'] = success_rate_victim
    results['mean_dist_l2'] = np.mean(l2_diff)
    results['std_dist_l2'] = np.sqrt(np.var(l2_diff))
    results['mean_dist_l0'] = np.mean(l0_diff)
    results['std_dist_l0'] = np.sqrt(np.var(l0_diff))
    results['mean_dist_li'] = np.mean(li_diff)
    results['std_dist_li'] = np.sqrt(np.var(li_diff))
    if guard_name in ['mlp', 'cnn']:
        plus = True
    else:
        plus = False

    alpha, detect_rate = search_alpha(results_train[0], results_train[1], results_train[2], plus=plus)
    detect_rate = comp_detect(results_train[0], results_train[1], results_train[2], alpha, plus=plus)
    delta_marginal = -(results_train[1] - alpha * results_train[2])
    print('delta_marginal:', delta_marginal)
    print('false alarm rate (reject < mean of logp(x) - %.2f * std): %.4f' % (alpha, detect_rate))
    results['FP_logpx'] = detect_rate
    detect_rate = comp_detect(results_adv[0], results_train[1], results_train[2], alpha, plus=plus)
    print('detection rate (reject < mean of logp(x) - %.2f * std): %.4f' % (alpha, detect_rate))
    results['TP_logpx'] = detect_rate

    fp_rate = []
    tp_rate = []
    delta_logit = []
    for i in xrange(nb_classes):
        ind = np.where(y_train[:, i] == 1)[0]
        alpha, detect_rate = search_alpha(results_train[3][ind], results_train[4][i], results_train[5][i], plus=plus)
        detect_rate = comp_detect(results_train[3][ind], results_train[4][i], results_train[5][i], alpha, plus=plus)
        fp_rate.append(detect_rate)
        delta_logit.append(-(results_train[4][i] - alpha * results_train[5][i]))

        ind = np.where(y_adv[ind_success][:, i] == 1)[0]
        if len(ind) == 0:	# no success attack, skip
            continue
        detect_rate = comp_detect(results_adv[3][ind], results_train[4][i], results_train[5][i], alpha, plus=plus)
        tp_rate.append(detect_rate)
    delta_logit = np.asarray(delta_logit, dtype='f') 
    print('delta_logit:', delta_logit)
    tp_rate = np.mean(tp_rate)
    fp_rate = np.mean(fp_rate)
    print('false alarm rate (reject < mean of logp(x|y) - %.2f * std): %.4f' % (alpha, fp_rate))
    results['FP_logpxy'] = fp_rate
    print('detection rate (reject < mean of logp(x|y) - %.2f * std): %.4f' % (alpha, tp_rate))
    results['TP_logpxy'] = tp_rate
   
    # now test the kl rejection scheme
    logit_mean, _, kl_mean, kl_std, softmax_mean = results_train[-5:]
    fp_rate = []
    tp_rate = []
    delta_kl = []
    for i in xrange(nb_classes):
        ind = np.where(y_train[:, i] == 1)[0]
        logit_tmp = y_logit_train[ind] - logsumexp(y_logit_train[ind], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        alpha, detect_rate = search_alpha(kl, kl_mean[i], kl_std[i], plus=True)
        detect_rate = comp_detect(kl, kl_mean[i], kl_std[i], alpha, plus=True)
        fp_rate.append(detect_rate)
        delta_kl.append(kl_mean[i] + alpha * kl_std[i])

        ind = np.where(y_adv[ind_success][:, i] == 1)[0]
        if len(ind) == 0:	# no success attack, skip
            continue
        logit_tmp = y_logit_adv[ind] - logsumexp(y_logit_adv[ind], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        detect_rate = comp_detect(kl, kl_mean[i], kl_std[i], alpha, plus=True)
        tp_rate.append(detect_rate)
    delta_kl = np.asarray(delta_kl, dtype='f')
    print('delta_kl:', delta_kl)
    tp_rate = np.mean(tp_rate)
    fp_rate = np.mean(fp_rate)
    print('false alarm rate (reject > mean of conditional KL + %.2f * std): %.4f' % (alpha, fp_rate))
    results['FP_kl'] = fp_rate
    print('detection rate (reject > mean of conditional KL + %.2f * std): %.4f' % (alpha, tp_rate))
    results['TP_kl'] = tp_rate

    # save results
    if save:
        if not os.path.isdir('detection_results/'):
            os.mkdir('detection_results/')
            print('create path detection_results/')
        path = 'detection_results/' + guard_name + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
            print('create path ' + path)
        filename = data_name + '_' + victim_name + '_' + attack_method
        if targeted:
            filename = filename + '_targeted'
        else:
            filename = filename + '_untargeted'
        pickle.dump(results, open(path+filename+'.pkl', 'wb'))
        print("results saved at %s.pkl" % (path+filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='mnist')
    parser.add_argument('--conv', '-C', action='store_true', default=False)
    parser.add_argument('--guard', '-G', type=str, default='bayes_K10')
    parser.add_argument('--targeted', '-T', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='fgsm_eps0.10')
    parser.add_argument('--victim', '-V', type=str, default='mlp')
    parser.add_argument('--save', '-S', action='store_true', default=False)

    args = parser.parse_args()
    test_attacks(args.batch_size, args.conv, args.guard, args.targeted, 
                 args.attack, args.victim, args.data, args.save)

