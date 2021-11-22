import numpy as np
import numpy.random as npr
import os
import pylab as plt
from multiprocessing import Pool
#from multiprocessing import shared_memory
import copy

from ..model_wrappers import master_timing as mt

# enforces run on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if(__name__ == '__main__'):
    
    ## Sample Run Script
    # walk-through of module usage
    # run as hyperparameter set

    # sample data provided in sample_data directory
    # sample data = 4 cell cluster ca activity across 2 worms 
    Y = np.load('StMLR/sample_data/sample_dat.npz')

    # convert npz dictionary to list of numpy arrays:
    Yl = []
    for i in range(len(Y.keys())):
        Yl.append(Y['arr_'+str(i)])

    # Forecasting specification
    # target_cells = which cells to predict future activity of:
    targ_cells = np.array([0,1])

    ## in_cells ~ which cells should be used as inputs to forecasting model
    # timing setup: t_break = time breakpoint
    # prediction window: t_break...t_break+T2
    # pre-window: t_break-T1...t_break
    # offset window: t_break-T1+T2...t_break+T2
    #
    # cells in pre-window
    in_cells = np.array([0,1,2,3])
    # cells in offset window:
    in_cells_offset = []

    ## Specfifying conditions:
    # expected data format = list of lists of numpy arrays
    # outer list = conditions; inner list = animals within condition
    #
    # all animals in same condition = random intercept-style model
    #Y2 = [Yl]
    # animals in differnt condition = random slope-style model
    # = each animal will get its own copy of all coefficients
    Y2 = [[Yl[0]], [Yl[1]]]

    # number of bootstraps for out-of-bootstrap cross-validation 
    num_boot = 5

    ## Build important tensors
    # Xfs_l = list of input tensors
    # olabs_l = list of 1-hot-labels (discretized changes in target cells) 
    # worm_ids_l = ids for each worm 
    # fb = filterbank = set of gaussian filters used to reduce input dimensionality
    Xfs_l, olabs_l, worm_ids_l, fb = [], [], [], []
    # iter thru conditions:
    for i, Yc in enumerate(Y2): 
        # dt = T2 (length of prediction window)
        # hist = T1 (length of input window)
        basemus, X, worm_ids, t0s = mt.build_tensors(Yc, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)

        # get labels and filtered X
        olab = mt.label_basemus(basemus, thrs=[-.06, -.02, .02, .06])
        Xf,fb = mt.filterX(X)

        # save:
        Xfs_l.append(Xf)
        olabs_l.append(olab)
        worm_ids_l.append(worm_ids)

    # draw train/test sets (here: everything is referred to as hyper set) 
    tot_size = sum([np.shape(xfi)[0] for xfi in Xfs_l])
    hyper_inds = np.ones((tot_size)) > 0.5
    # hyperparameter train/test set generation:
    train_sets_hyper, test_sets_hyper = mt.generate_traintest(tot_size, num_boot, hyper_inds, hyper_inds, train_perc=0.9)
    train_sets_hyper = train_sets_hyper > 0.5
    test_sets_hyper = test_sets_hyper > 0.5
 
    ## Model Configuration
    # mode 0 is the default model 
    # each model is composed of a single Soft Tree and a single set of MLR submodels
    # base run_config:
    mode = 0 
    run_id = 'sample_run' 
    # load standard run configuration ~ stored as dictionary
    rc = mt.get_run_config(mode, run_id)
    # adjusting tree depth (or tree width) alters the number of states (MLR submodels)
    # number of submodels = tree_width ** tree_depth
    rc['tree_depth'] = [2]
    # add required data to run_configuration
    mt.add_dat_rc(rc, hyper_inds, [], [], Xfs_l, Xfs_l, worm_ids_l, olabs_l, train_sets_hyper,
            test_sets_hyper) 

    # run out-of-bootstrap cross-validation on set
    # makes a directory and saves important info there
    mt.boot_cross_hyper(rc)


