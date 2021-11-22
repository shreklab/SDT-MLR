'''

    Runs Separation Experiments

    Compares multiple submodels on the same state division


'''

import numpy as np
import os

from . import core_utils
from ..models import separator
from ..models import core_models



# get state predictions
# submit each block as a minibatch
def get_state_preds(S, Xf_gate): 
    sw = []
    for i in range(np.shape(Xf_gate)[0]): 
        st_weights = S.F.eval(Xf_gate[i].astype(np.float32))
        sw.append(st_weights)
    return np.vstack(sw)
        


## Out-of-bootstrap cross-validation
# fit full model to hyper set --> fit log reg models for each bootstrap
# Assumes: data is already in block format = num_block x T per block x ...
# NOTE: uses schedule to control type of run
# rc = run_config dictionary
# optionally use pre-run separator S (do not retrain that part...jump straight to downstream cross)
def boot_cross(rc, S=[]): 

    # if directory exists --> stop
    # else --> create and populate it
    if(not os.path.isdir(rc['dir_str'])):
        os.mkdir(rc['dir_str'])
    # save metadata:  
    targ_strs = ['dir_str', 'comparison_filter', 'tree_depth', 'tree_width', 'l1_tree_mask', \
            'l1_mlr_mask', 'l2_state_mask', 'num_model', 'even_reg', 'num_epochs', 'schedule']
    core_utils.save_metadata(rc, targ_strs)

    ## generate architecture:
    if(S == []): 
        S = separator.arch_gen(rc['olabs'], rc['Xf_gate'], rc['Xf_drive'], rc['comparison_filter'], rc['tree_depth'], rc['tree_width'], rc['num_model'], even_reg=rc['even_reg'], l1_tree_mask=rc['l1_tree_mask'], l1_mlr_mask=rc['l1_mlr_mask'], l2_state_mask=rc['l2_state_mask'])

        ## run whole model on hyperparameter set:
        # --> train_rll = num_models x num_state x models_per_state
        tr, null, train_rll, null = separator.train_epochs_wrapper(S, rc['olabs'], rc['Xf_gate'], rc['Xf_drive'], rc['schedule'], rc['num_epochs'], rc['hyper_inds'], rc['hyper_inds'])

        ## select best model
        # best model?
        print('post hyper train')
        print(train_rll)
        print(np.shape(train_rll))
        # average train_rll across states
        train_rll = np.mean(train_rll, axis=1) 
        train_posll = np.sum(train_rll * rc['comparison_filter'][None,:], axis=1)
        print(train_posll)
        max_model_ind = np.argmax(train_posll)
        use_trees = np.zeros((S.F.num_model))
        use_trees[max_model_ind] = 1
        use_trees = use_trees > 0.5
        # build new forest with just best tree:
        F2 = core_models.Forest(S.F.tree_depth, S.F.tree_width, 1, S.F.xdim, S.F.even_reg, S.F.l1_mask, S.F.tree_struct, use_trees)
        # build new separator with new forest
        S = separator.arch_gen(rc['olabs'], rc['Xf_gate'], rc['Xf_drive'], rc['comparison_filter'], rc['tree_depth'], rc['tree_width'], 1, even_reg=rc['even_reg'], l1_tree_mask=rc['l1_tree_mask'], l1_mlr_mask=rc['l1_mlr_mask'], l2_state_mask=rc['l2_state_mask'], Fin=F2)

    ## Save important data that does NOT vary across boots:
    # 1. weight preds, 2. Xf_gate, 3. Xf_drive, 4. train_sets, 5. test_sets
    weight_preds = get_state_preds(S, rc['Xf_gate']) 
    np.save(os.path.join(rc['dir_str'], 'weight_preds'), weight_preds)
    np.save(os.path.join(rc['dir_str'], 'Xf_gate'), rc['Xf_gate'])
    np.save(os.path.join(rc['dir_str'], 'Xf_drive'), rc['Xf_drive'])
    np.save(os.path.join(rc['dir_str'], 'train_sets'), rc['train_sets'])
    np.save(os.path.join(rc['dir_str'], 'test_sets'), rc['test_sets'])

    ## fit mlr/drives to each bootstrap
    # --> save forest errors
    save_test_ll = []
    save_train_ll = []
    save_gate_vars = []
    save_drive_vars = []
    # schedule = [0, N] = runs no gate epochs
    new_sched = [0, rc['schedule'][1]]
    for i in range(np.shape(rc['train_sets'])[0]):
        train_inds = rc['train_sets'][i]
        test_inds = rc['test_sets'][i]

        tr, te, train_rll, test_rll = separator.train_epochs_wrapper(S, rc['olabs'], rc['Xf_gate'], rc['Xf_drive'], new_sched, rc['num_epochs'], train_inds, test_inds)

        save_train_ll.append(train_rll)
        save_test_ll.append(test_rll)

        # training vars?
        save_gate_vars.append(S.F.get_analysis_vars())
        save_drive_vars.append(S.MLR.get_analysis_vars())

        # save everything:
        np.save(os.path.join(rc['dir_str'], 'train_ll'), np.array(save_train_ll))
        np.save(os.path.join(rc['dir_str'], 'test_ll'), np.array(save_test_ll))
        np.save(os.path.join(rc['dir_str'], 'gate_vars'), np.array(save_gate_vars))
        np.save(os.path.join(rc['dir_str'], 'drive_vars'), np.array(save_drive_vars))

    return S


# get basic run configuration
# returns python dict with default model parameters
# mode 0 default = no stimulus, 4 state, no lr
# mode 2 = stimulus boosting, 4 states for each, no lr
def get_run_config(run_id):
    d = {}
    # fn string from run id:
    d['dir_str'] = run_id
    # empty data ~ need to fill this in after call:
    d['hyper_inds'] = []
    d['train_sets'] = []
    d['test_sets'] = []
    d['Xf_gate'] = []
    d['Xf_drive'] = []
    d['olabs'] = []
    d['comparison_filter'] = []
    # shared tree info:
    d['tree_depth'] = 2
    d['tree_width'] = 2
    d['l1_tree_mask'] = .01
    d['l1_mlr_mask'] = .01
    d['l2_state_mask'] = 0.0
    d['num_model'] = 100
    d['even_reg'] = 0.1 # entropy regularization --> ensures similar amounts of data to each leaf
    # run stuff:
    d['num_epochs'] = 40
    d['schedule'] = [1,1]
    return d


