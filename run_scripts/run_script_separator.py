import numpy as np
import numpy.random as npr
import os
import pylab as plt
from multiprocessing import Pool
#from multiprocessing import shared_memory
import copy

from ..model_wrappers import master_separator as ms
from ..model_wrappers import core_utils 

# enforces run on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if(__name__ == '__main__'):
    
    ## Sample Run Script
    # walk-through of module usage
    # run as hyperparameter set

    ## load dat: 
    import sys
    #sys.path.append('/home/ztcecere/CodeRepository/PD/')
    sys.path.append('/snl/scratch/ztcecere/PD')
    import data_loader
    #rdir = '/data/ProcAiryData'
    rdir = '/home/ztcecere/ProcAiryData'
    inp, Y, inp_zim, Y_zim = data_loader.load_data(rdir)
    # load ztc buffer-to-buffer
    inp_ztcbuf, Y_ztcbuf = data_loader.load_data_ztcNoStim(rdir)
    # load jh buffer-to-buffer
    inp_jhbuf, Y_jhbuf = data_loader.load_data_jh_buf2buf_trial2(rdir) 


    # Test 1: zim + RA
    Y2 = Y_zim
    fn_cond = 'zim'

    # Forecasting specification
    # target_cells = which cells to predict future activity of:
    targ_cells = np.array([0,1])
    fn_pred = 'RA'

    ## in_cells ~ which cells should be used as inputs to forecasting model
    # timing setup: t_break = time breakpoint
    # prediction window: t_break...t_break+T2
    # pre-window: t_break-T1...t_break
    # offset window: t_break-T1+T2...t_break+T2
    #
    # cells in pre-window
    in_cells = np.array([0,1,2,3])
    # cells in offset window:
    in_cells_offset = np.array([4,5])

    # number of bootstraps for out-of-bootstrap cross-validation 
    num_boot = 1000

    ## Build important tensors for each animal

    # dt = T2 (length of prediction window)
    # hist = T1 (length of input window)
    basemus, X, worm_ids, t0s = core_utils.build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)

    # get labels and filtered X
    olab = core_utils.label_basemus(basemus, thrs=[-.06, -.01, .01, .06])
    Xf,fb = core_utils.filterX(X)

    ## 2 Condition experiment
    # with and without stim
    Xf_drive_full = copy.deepcopy(Xf)
    Xf_drive_ko = copy.deepcopy(Xf)
    Xf_drive_ko[:,:,4:,:] = 0.0

    # make Xf gate and Xf drive:
    # pre xdim... still separates cells and filters
    Xf_gate = copy.deepcopy(Xf)
    Xf_drive = np.concatenate((Xf_drive_full[:,:,None,:,:], Xf_drive_ko[:,:,None,:,:]), axis=2)

    # convert to xdim (flatten across cells and filters)
    [num_block, Tper, num_cell, subx] = np.shape(Xf_gate)
    Xf_gate = np.reshape(Xf_gate, (num_block, Tper, num_cell*subx))
    Xf_drive = np.reshape(Xf_drive, (num_block, Tper, 2, num_cell*subx)) # 2 refers to 2 conditions

    # combine Xf_drive with worm_ids
    # both submodels should get access:
    worm_ids2 = worm_ids[:,:,None,:]
    worm_ids2 = np.tile(worm_ids2, (1,1,2,1)) # 2 refers to 2 conditions
    Xf_drive = np.concatenate((Xf_drive, worm_ids2), axis=3)

    ## make l1 masks for Xf_drive = l1_mlr_mask
    # l1 mask for different cells:
    l1_base_scale = .01
    l1_stim_scale = .05
    l1_cells = np.ones((num_cell, subx))
    l1_cells[:4,:] = l1_base_scale
    l1_cells[4:,:] = l1_stim_scale
    # wid mask?
    l1_wid_scale = .05
    l1_wid = np.ones((np.shape(worm_ids)[-1]))
    l1_wid[:] = l1_wid_scale

    # combine into single mask --> l1_mlr_mask
    l1_mlr_mask = np.hstack((np.reshape(l1_cells,-1), l1_wid))

    ## cross experiment
    # ~50% goes into hyperparam set
    hyper_inds = npr.random(np.shape(Xf_gate)[0]) < 0.45
    # cross-validation train/test set generation:
    # everything trainable
    trainable_inds = np.ones((np.shape(Xf_gate)[0])) > 0.5
    # testable = not hyper set
    testable_inds = np.logical_not(hyper_inds)
    train_sets, test_sets = core_utils.generate_traintest(np.shape(Xf_gate)[0], num_boot, trainable_inds, testable_inds, test_perc=0.5)
    train_sets = train_sets > 0.5
    test_sets = test_sets > 0.5

    # get run configs:
    rc = ms.get_run_config('sep_' + fn_cond + fn_pred + str(l1_stim_scale))
    
    # fill out run config
    rc['hyper_inds'] = hyper_inds
    rc['train_sets'] = train_sets
    rc['test_sets'] = test_sets
    rc['Xf_gate'] = Xf_gate
    rc['Xf_drive'] = Xf_drive
    rc['olabs'] = olab
    rc['l1_mlr_mask'] = l1_mlr_mask

    # basic comparison filter ~ fit for 0th submodel
    rc['comparison_filter'] = np.array([1,0]).astype(np.float32)

    # run out-of-bootstrap cross-validation on set
    # makes a directory and saves important info there
    ms.boot_cross(rc)


