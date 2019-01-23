    
def config_fgsm(targeted, adv_ys, eps, batch_size):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    fgsm_params = {yname: adv_ys,
                   'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 1.,
                   'batch_size': batch_size}
    return fgsm_params, yname
    
def config_mim(targeted, adv_ys, eps, batch_size):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    mim_params = {yname: adv_ys,
                  'eps': eps,
                  'eps_iter': 0.01,
                  'nb_iter': 100,
                  'decay_factor': 1.0,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  'rand_init': True,
                  'batch_size': batch_size}
    return mim_params, yname

def config_cw(targeted, adv_ys, eps, batch_size):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    cw_params = {yname: adv_ys,
                 'max_iterations': 1000,
                 'binary_search_steps': 1,
                 'abort_early': False,
                 'confidence': 0.,
                 'learning_rate': 1e-2,
                 'initial_const': eps,
                 'clip_min': 0., 
                 'clip_max': 1.,
                 'batch_size': batch_size}
    return cw_params, yname

def config_madry(targeted, adv_ys, eps, batch_size):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    madry_params = {yname: adv_ys,
                    'eps': eps,
                    'eps_iter': 0.01,
                    'nb_iter': 40,
                    'clip_min': 0., 
                    'clip_max': 1.,
                    'rand_init': True,
                    'batch_size': batch_size} 
                              
    return madry_params, yname 

def load_attack(sess, attack_method, model, targeted, adv_ys, eps, batch_size):

    if attack_method == 'fgsm':
        from cleverhans.attacks import FastGradientMethod 
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = FastGradientMethod(model_prob, sess=sess)
        attack_params, yname = config_fgsm(targeted, adv_ys, eps, batch_size)

    if attack_method == 'pgd':
        from cleverhans.attacks import MadryEtAl
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = MadryEtAl(model_prob, sess=sess)
        attack_params, yname = config_madry(targeted, adv_ys, eps, batch_size)

    if attack_method == 'mim':
        from cleverhans.attacks import MomentumIterativeMethod
        model_prob = lambda x: model.predict(x, softmax=True)
        attack = MomentumIterativeMethod(model_prob, sess=sess)
        attack_params, yname = config_mim(targeted, adv_ys, eps, batch_size)

    if attack_method == 'cw':
        from cleverhans.attacks import CarliniWagnerL2
        model_logit = lambda x: model.predict(x, softmax=False)
        attack = CarliniWagnerL2(model_logit, sess=sess)
        attack_params, yname = config_cw(targeted, adv_ys, eps, batch_size)

    return attack, attack_params, yname

