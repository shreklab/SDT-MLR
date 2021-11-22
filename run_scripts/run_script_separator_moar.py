'''

	Run Script for Separator Experiments
	-> Compare models = On/Off cells vs. no stim vs. external stim
	-> Additional runs for worm subsets... compare high variance across conds
        -> replace missing on/off cells with matching averages
        TODO: is it worth it to use inferred calcium?
        ... ideas I like: 1. inferred calcium, 2. thresholded inferred calcium

'''

import numpy as np
import numpy.random as npr
import os
import pylab as plt
import copy

from ..model_wrappers import master_separator as ms
from ..model_wrappers import core_utils 
from ..models import separator

# enforces run on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#NOTE:
# MARKER:VARIABLE (s) mark experimental variables

# remake separator
# keep gating tree --> make new MLR object
# why? if sets have different numbers of worms --> wid_struct will be different shape
# --> different number of xdims in drive models
def remake_separator(S, rc):
    num_models = S.MLR.num_models
    num_state = S.MLR.num_state
    submodels_per_state = S.MLR.models_per_state
    output_cells = S.MLR.output_cells
    output_classes = S.MLR.output_classes   
    l1_mask = rc['l1_mlr_mask']
    l2_state_mask = rc['l2_state_mask']

    # xdim:
    xdim = np.shape(rc['Xf_drive'])[-1]

    print('xdim = ' + str(xdim))

    MLRC = separator.MultiLogRegComp(num_models, num_state, submodels_per_state, xdim, output_cells, output_classes, \
        l1_mask, l2_state_mask)    

    # make new separator:
    Snew = separator.Separator(S.F, MLRC, rc['comparison_filter'])

    return Snew


# run condition function:
# NOTE: this is super specific
# 5 conditions:
# 1. use On/Off cells
# 2. No stim
# 3. raw input ~ split into On and Off channels
# 4. alternate nan replacement system for On/Off cells
# NOTE: root_set = draw hyper_inds...else --> everything is trainable
# Assumes: Y2 = [AVA, RME, SMDV, SMDD, On_cell, Off_cell, On raw, Off raw]
def run_5cond(Y2, targ_cells, in_cells, in_cells_offset, fn_cond, fn_pred, S=[], num_boot=1000, root_set=True):

    # replace nans with 0s in Y2:
    for Yc in Y2:
        naninds = np.isnan(Yc)
        Yc[naninds] = 0.0

    ## Build important tensors for each animal

    # dt = T2 (length of prediction window)
    # hist = T1 (length of input window)
    basemus, X, worm_ids, t0s = core_utils.build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)

    # MARKER:VARIABLE
    # get labels and filtered X
    olab = core_utils.label_basemus(basemus, thrs=[-.06, -.01, .01, .06])
    # simplified:
    #olab = core_utils.label_basemus(basemus, thrs=[0.0])
    Xf,fb = core_utils.filterX(X)

    ## 4 Condition experiment
    # with and without stim
    # 1. cells
    Xf_drive_full = copy.deepcopy(Xf)[:,:,:6,:]
    # 2. no stim
    Xf_drive_ko = copy.deepcopy(Xf)[:,:,:6,:]
    Xf_drive_ko[:,:,4:,:] = 0.0
    # 3. raw input:
    use_inds = np.array([0,1,2,3,6,7])
    Xf_drive_raw = copy.deepcopy(Xf)[:,:,use_inds,:]
    # 4. copied and averaged cells
    use_inds = np.array([0,1,2,3,8,9])
    Xf_drive_copy = copy.deepcopy(Xf)[:,:,use_inds,:]

    # make Xf gate and Xf drive:
    # pre xdim... still separates cells and filters
    # MARKER:VARIABLE
    Xf_gate = copy.deepcopy(Xf)[:,:,:4,:] # only use base cells as gates
    # Xf_gate = copy.deepcopy(Xf)[:,:,:6,:] # use On/Off in gate

    Xf_drive = np.concatenate((Xf_drive_full[:,:,None,:,:], Xf_drive_ko[:,:,None,:,:], \
            Xf_drive_raw[:,:,None], Xf_drive_copy[:,:,None]), axis=2)

    # convert to xdim (flatten across cells and filters)
    [num_block, Tper, num_cell, subx] = np.shape(Xf_gate)
    [num_block, Tper, null, num_cell_drive, subx_drive] = np.shape(Xf_drive)
    Xf_gate = np.reshape(Xf_gate, (num_block, Tper, num_cell*subx))
    Xf_drive = np.reshape(Xf_drive, (num_block, Tper, 4, num_cell_drive*subx_drive)) # 4 refers to 4 conditions

    # combine Xf_drive with worm_ids
    # both submodels should get access:
    worm_ids2 = worm_ids[:,:,None,:]
    worm_ids2 = np.tile(worm_ids2, (1,1,4,1)) # 4 refers to 4 conditions
    Xf_drive = np.concatenate((Xf_drive, worm_ids2), axis=3)

    ## make l1 masks for Xf_drive = l1_mlr_mask
    # l1 mask for different cells:
    l1_base_scale = .01
    # MARKER:VARIABLE
    l1_stim_scale = .3
    l1_cells = np.ones((num_cell_drive, subx_drive))
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
    if(root_set):
        testable_inds = np.logical_not(hyper_inds)
        test_perc = 0.5
    else: # non-root-set --> everything testable
        testable_inds = np.ones((np.shape(Xf_gate)[0])) > 0.5
        test_perc = 0.15
    train_sets, test_sets = core_utils.generate_traintest(np.shape(Xf_gate)[0], num_boot, trainable_inds, testable_inds, test_perc=test_perc)
    train_sets = train_sets > 0.5
    test_sets = test_sets > 0.5

    print('training vs. test')
    print(np.sum(train_sets))
    print(np.sum(test_sets))
    print(np.sum(train_sets == test_sets))

    # get run configs:
    # MARKER:VARIABLE ~ naming
    rc = ms.get_run_config('sep_' + fn_cond + fn_pred + str(l1_stim_scale))
    
    # fill out run config
    rc['hyper_inds'] = hyper_inds
    rc['train_sets'] = train_sets
    rc['test_sets'] = test_sets
    rc['Xf_gate'] = Xf_gate
    rc['Xf_drive'] = Xf_drive
    rc['olabs'] = olab
    rc['l1_mlr_mask'] = l1_mlr_mask

    # MARKER:VARIABLE
    # basic comparison filter ~ fit for all submodels
    #rc['comparison_filter'] = np.array([.25 for z in range(4)]).astype(np.float32)
    # On/Off only comparison filter:
    rc['comparison_filter'] = np.array([1,0,0,0]).astype(np.float32)

    # if using an existing separator
    # --> remake the drive to account for different set sizes:
    if(isinstance(S, separator.Separator)):
        S = remake_separator(S, rc)

    # run out-of-bootstrap cross-validation on set
    # makes a directory and saves important info there
    # boot_cross optionally takes in a pre-trained separator class
    S = ms.boot_cross(rc, S=S)
    return S



if(__name__ == '__main__'):
    
    ## Multi Separator run
    # simple version:
    # fit whole model base dataset --> fit MLR submodels only for each other dataset

    ## load dat: 
    import sys
    #sys.path.append('/home/ztcecere/CodeRepository/PD/')
    sys.path.append('/snl/scratch/ztcecere/PD')
    import data_loader
    #rdir = '/data/ProcAiryData'
    rdir = '/home/ztcecere/ProcAiryData'
    inp, Y, inp_zim, Y_zim, zim_cond_inds = data_loader.load_data(rdir,get_cond_inds=True)

    print('zim conds')
    print(zim_cond_inds)
    input('cont?')

    # load ztc buffer-to-buffer
    inp_ztcbuf, Y_ztcbuf = data_loader.load_data_ztcNoStim(rdir)
    # load jh buffer-to-buffer
    inp_jhbuf, Y_jhbuf = data_loader.load_data_jh_buf2buf_trial2(rdir) 

    # TODO: make long condition
    # = zim_sf runs with minpulse > 15
    Y_zim_long, inp_zim_long = [], []
    for i in range(len(Y_zim)): 
        if(zim_cond_inds[i] == 'SF'): 
            # find transitions:
            trans_inds = np.where(inp_zim[i][1:] != inp_zim[i][:-1])[0]
            # min tspace?
            mintsp = np.amin(np.diff(trans_inds))
            if(mintsp > 15):
                Y_zim_long.append(copy.deepcopy(Y_zim[i]))
                inp_zim_long.append(copy.deepcopy(inp_zim[i]))

    # TODO: make mm condition
    Y_zim_mm, inp_zim_mm = [], []
    for i in range(len(Y_zim)):
        if(zim_cond_inds[i] == 'MM'):
            Y_zim_mm.append(copy.deepcopy(Y_zim[i]))
            inp_zim_mm.append(copy.deepcopy(inp_zim[i]))

    print('len(zim_long) = ' + str(len(Y_zim_long)))
    print('len(zim_mm) = ' + str(len(Y_zim_mm)))

    # order matters
    Y_l = [Y_zim, Y_zim_mm, Y_ztcbuf, Y, Y_zim_long, Y_jhbuf]
    inp_l = [inp_zim, inp_zim_mm, inp_ztcbuf, inp, inp_zim_long, inp_jhbuf]
    fn_conds = ['zim', 'zimMM', 'zimbuf', 'oh', 'zimlong', 'javbuf']

    # Forecasting specification
    # target_cells = which cells to predict future activity of:
    # MARKER:VARIABLE
    targ_cells = np.array([0,1])
    fn_pred = 'RA'

    #targ_cells = np.array([2,3])
    #fn_pred = 'DV'

    # First: combine cell data, input, and On/Off copies into 1 tensor
    # concatenate input (2 channels) into Y:
    # ordering: [AVA, RME, SMDV, SMDD, On, Off, On_external, Off_external, On_copy, Off_copy]
    for i, Y_cond in enumerate(Y_l):
        for j, Yc in enumerate(Y_cond):
            # reshape on channel:
            on_ch = np.reshape(inp_l[i][j], (-1,1))
            # invert to get most of off channel:
            off_ch = 1.0 - on_ch
            # block out beginning of off channel because not yet exposed:
            st_ind = np.where(on_ch[:,0] > 0)[0][0]
            off_ch[:st_ind] = 0.0
            # concatenate all data together:
            # --> T x 10
            Y_l[i][j] = np.concatenate((Yc, on_ch, off_ch, copy.deepcopy(Yc[:,4:6])), axis=1)

    
    # TODO: Second: for specified conditions
    # replace nans with average reps within condition
    # this is mainly for handling missing on/off cells in buffer trials
    nan_conds = ['zimbuf', 'javbuf']
    for i in range(len(Y_l)):
        if(fn_conds[i] in nan_conds):
            print('nan writing: ' + str(i))
            # find the longest trial_length:
            trial_len = -1
            for j in range(len(Y_l[i])):
                if(np.shape(Y_l[i][j])[0] > trial_len):
                    trial_len = np.shape(Y_l[i][j])[0]
            # make sum and count arrays:
            sum_ar = np.zeros((trial_len,2)) # NOTE: 2 cuz only doing for last 2 cols
            count_ar = np.zeros((trial_len,2))
            for j in range(len(Y_l[i])):
                csize = np.shape(Y_l[i][j])[0]
                ginds = np.logical_not(np.isnan(Y_l[i][j][:,-2:]))
                sum_ar[:csize][ginds] = sum_ar[:csize][ginds] + Y_l[i][j][:,-2:][ginds]
                count_ar[:csize][ginds] = count_ar[:csize][ginds] + 1.0
            gcount = count_ar > 0
            sum_ar[gcount] = sum_ar[gcount] / count_ar[gcount]
            # run back thru trials --> overwrite nans:
            for j in range(len(Y_l[i])):
                addz = Y_l[i][j][:,-2:].copy()
                csize = np.shape(addz)[0]
                naninds = np.isnan(addz)
                addz[naninds] = sum_ar[:csize][naninds]
                Y_l[i][j][:,-2:] = addz[:]


    ## in_cells ~ which cells should be used as inputs to forecasting model
    # timing setup: t_break = time breakpoint
    # prediction window: t_break...t_break+T2
    # pre-window: t_break-T1...t_break
    # offset window: t_break-T1+T2...t_break+T2
    #
    # cells in pre-window
    in_cells = np.array([0,1,2,3])
    # cells in offset window:
    in_cells_offset = np.array([4,5,6,7,8,9])

    S = []
    root_set = True
    for i in range(len(Y_l)):
        S = run_5cond(Y_l[i], targ_cells, in_cells, in_cells_offset, fn_conds[i], fn_pred, S=S, num_boot=1000, root_set=root_set)
        root_set = False


