    
def config_fgsm(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    fgsm_params = {yname: adv_ys,
                   'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    return fgsm_params
    
def config_bim(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    bim_params = {yname: adv_ys,
                  'eps': 0.3,
                  'eps_iter': 0.01,
                  'nb_iter': 100,
                  'clip_min': 0.,
                  'clip_max': 1.}
    return bim_params
    
def config_mim(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    mim_params = {yname: adv_ys,
                  'eps': 0.1,
                  'eps_iter': 0.01,
                  'nb_iter': 100,
                  'decay_factor': 0.7,
                  'clip_min': 0.,
                  'clip_max': 1.}
    return mim_params
    
def config_jsma(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    jsma_params = {yname: adv_ys,
                   'theta': 1., 
                   'gamma': 0.1,
                   'clip_min': 0., 
                   'clip_max': 1.}
    return jsma_params
    
def config_vat(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    vat_params = {yname: adv_ys,
                  'eps': 2.0, 
                  'xi': 1e-6,
                  'num_iterations': 10,
                  'clip_min': 0., 
                  'clip_max': 1.}
    return vat_params

def config_cw(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    cw_params = {yname: adv_ys,
                 'max_iterations': 10000,
                 'binary_search_steps': 9,
                 'abort_early': True,
                 'confidence': 0.,
                 'learning_rate': 1e-2,
                 'initial_const': 1e-3,
                 'clip_min': 0., 
                 'clip_max': 1.}
    return cw_params
    
def config_elastic(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    elastic_params = {yname: adv_ys,
                      'beta': 1e-3,
                      'confidence': 0.,
                      'learning_rate': 1e-2,
                      'binary_search_steps': 9,
                      'max_iterations': 1000,
                      'abort_early': False,
                      'initial_const': 1e-3,
                      'clip_min': 0., 
                      'clip_max': 1.}
              
    return elastic_params
    
def config_deepfool(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    deepfool_params = {yname: adv_ys,
                      'nb_candidate': 10,
                      'overshoot': 0.02,
                      'max_iter': 50,
                      'clip_min': 0., 
                      'clip_max': 1.}
              
    return deepfool_params    

def config_madry(targeted, adv_ys):
    if targeted:
        yname = 'y_target'
    else:
        yname = 'y'
    madry_params = {yname: adv_ys,
                    'eps': 0.3,
                    'eps_iter': 0.01,
                    'nb_iter': 40,
                    'clip_min': 0., 
                    'clip_max': 1.,
                    'rand_init': False} 
                              
    return madry_params 

    
