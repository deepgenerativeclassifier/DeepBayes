from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings
import collections

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper

_logger = utils.create_logger("cleverhans.attacks")

from cleverhans.attacks import Attack
import tensorflow as tf

"""
The "superwhite" attack that also considers randomness & detection.
The skeleton of the class is based on cleverhans 2.0 PGD attack class.
"""

class SuperWhite(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    To counter for randomness & detection, loss function
    is edited in this attack
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a MadryEtAl instance.
        """
        super(SuperWhite, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'batch_size',
                                  'delta_marginal', 'delta_logit', 'delta_kl', 
                                  'detection_lambda', 'kl_prob_vec',
                                  'combine_logits']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, rand_init=True, detection_lambda=1.0, 
                     delta_marginal=-50, delta_logit=-50, delta_kl=0.0, 
                     kl_prob_vec = None, combine_logits='ensemble', **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init

        # parameters in YL's 3 detection methods
        self.delta_marginal = delta_marginal	# a scalar
        self.delta_logit = delta_logit	# a vector of shape (n_class,)
        self.kl_prob_vec = kl_prob_vec	# shape (n_class, n_class)
        assert self.kl_prob_vec is not None
        self.delta_kl = delta_kl	# a vector of shape (n_class,)
        self.detection_lambda = detection_lambda
        self.combine_logits = combine_logits
        assert self.combine_logits in ['ensemble', 'bayes']
        print(self.detection_lambda, 'detection_lambda')

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def model_loss(self, y, logits):
        # compute cross entropy loss for y of shape (N, dimY)
        # and logits of shape (K, N, dimY)

        # first normalise logits
        if len(logits.get_shape().as_list()) == 2:
            ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            return tf.reduce_mean(ce_loss)
        else:
            print('attack snapshots...')
            # first normalise logits
            logits -= tf.reduce_max(logits, -1, keep_dims=True)
            logits -= tf.log(tf.reduce_sum(tf.exp(logits), -1, keep_dims=True) + 1e-9)

            # then compute cross entropy
            ce_loss = -tf.reduce_sum(logits * y, -1)
            ce_loss = tf.reduce_mean(ce_loss, 0)
            return tf.reduce_mean(ce_loss) 

    def combine(self, logits):
        # combine logits of shape (K, N, dimY) to shape (N, dimY)
        print('combine the logits from random network snapshots (%s)...' % self.combine_logits)
        if self.combine_logits == 'ensemble':
            results = tf.reduce_mean(tf.nn.softmax(logits), 0)	# (N, dimY)
            results = tf.log(tf.clip_by_value(results, 1e-20, np.inf))
        if self.combine_logits == 'bayes': 
            logits_max = tf.reduce_max(logits, 0)
            logits_ = logits - logits_max	# (dimY, N)
            results = tf.log(tf.clip_by_value(tf.reduce_mean(tf.exp(logits_), 0), 1e-20, np.inf))
            results += logits_max

        return results

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        adv_x = x + eta
        preds = self.model.get_logits(adv_x) # shape (K, N, dimY)
        loss = self.model_loss(y, preds) # see Carlini's recipe
        if self.targeted:
            loss = -loss

        # now forms the predicted output
        if len(preds.get_shape().as_list()) == 2:
            logits = preds
        else:
            logits = self.combine(preds)

        # loss to evade marginal detection
        def logsumexp(x):
            x_max = tf.expand_dims(tf.reduce_max(x, 1), 1)
            res = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x - x_max), 1) , 1e-10, np.inf))
            return res + x_max[:, 0]

        logpx = logsumexp(logits)
        loss_detect_marginal = -tf.reduce_mean(tf.nn.relu(-logpx - self.delta_marginal))

        # loss to evade logit detection
        y_pred = tf.argmax(logits, 1)
        loss_detect_logit = tf.nn.relu(-logits - self.delta_logit)
        loss_detect_logit = -tf.reduce_mean(tf.gather(loss_detect_logit, y_pred, axis=1))

        # loss to evade kl detection
        N = logits.get_shape().as_list()[0]
        logits_normalised = logits - tf.expand_dims(logsumexp(logits), 1)
        kl = tf.reduce_sum(self.kl_prob_vec * (tf.log(self.kl_prob_vec) - tf.expand_dims(logits_normalised, 1)), 2)
        loss_detect_kl = tf.nn.relu(kl - self.delta_kl)
        loss_detect_kl = -tf.reduce_mean(tf.gather(loss_detect_kl, y_pred, axis=1))

        #loss_detect = loss_detect_marginal
        loss_detect = loss_detect_logit
        #loss_detect = loss_detect_kl

        # combine
        print('using lambda_detect = %.2f' % self.detection_lambda)
        loss += self.detection_lambda * loss_detect

        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.

        :param x: A tensor with the input image.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x
