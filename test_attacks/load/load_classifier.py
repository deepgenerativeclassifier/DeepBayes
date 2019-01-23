import tensorflow as tf
import numpy as np

def load_classifier(sess, model_name, data_name, path=None,
                    attack_snapshot=False):

    if 'bayes' in model_name:
        from load_bayes_classifier import BayesModel
        # use for example model_name = 'bayes_K10_A'
        # if with dimZ != 64, then model_name should be 'bayes_K10_Z32_A'
        conv = True
        checkpoint = 0
        vae_type = model_name[-1]
        use_mean = False
        fix_samples = False
        K = int(model_name.split('_')[1][1:])
        if conv:
            model_name += '_cnn'
        else:
            model_name += '_mlp'
        if 'Z' in model_name:
            dimZ = int(model_name.split('_')[2][1:])
        else:
            if data_name == 'mnist':
                dimZ = 64
            else:
                dimZ = 128
        model = BayesModel(sess, data_name, vae_type, conv, K, checkpoint=checkpoint, 
                           attack_snapshot=attack_snapshot, use_mean=use_mean, fix_samples=fix_samples,
                           dimZ=dimZ)   

    return model

