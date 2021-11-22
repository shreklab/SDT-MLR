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


"""
#### Shared Memory Solution
# requires python3.8

# shared memory creation:
# returns: [name, shape, dtype]
def create_shm_obj(np_obj):
    # create new shm
    shm = shared_memory.SharedMemory(create=True, size=np_obj.nbytes)
    # create numpy array backed by shared memory:
    npb = np.ndarray(np_obj.shape, dtype=np_obj.dtype, buffer=shm.buf)
    # copy og data into shared memory:
    npb[:] = np_obj[:]
    return [shm.name, np.shape(npb), npb.dtype]


# convert numpy arrays to shared memory arrays
# writes shared memory info into run_config
# each data entry = [shared memory name, shape, dtype]
# NOTE: writes sharedmem info over nump array in run_config
def create_shm(rc):
    # numpy targets
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper'] 
    for tk in target_keys:
        if(tk in rc):
            rc[tk] = create_shm_obj(rc[tk])
    # list of numpy targets:
    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            rc_new = []
            for ar in rc[tk]:
                rc_new.append(create_shm_obj(ar))
            rc[tk] = rc_new
   

# convert shared memory name to numpy array
# 1. attached to shared memory object (via name)
# 2. attach numpy array buffer to shared memory object
# 3. copy reference for numpy array into run_config dictionary (replaces name)
# NOTE stored in run_config = [name, shape, dtype]
# NOTE: this should be called after mapping 
def convert_shmname_to_np(rc):
    # numpy targets:
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper'] 
    for tk in target_keys:
        if(tk in rc):
            # get info
            [name, shape, dtype] = rc[tk]
            # attach to shared memory:
            cur_shm = shared_memory.SharedMemory(name=name)
            # attach numpy array buffer to shared memory:
            ar = np.ndarray(shape, dtype=dtype, buffer=cur_shm.buf)
            # replace in run_config:
            rc[tk] = ar
    # targets: list of numpy arrays:
    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            cur_v = []
            for l in rc[tk]:
                # get info
                [name, shape, dtype] = l
                # attach to shared memory:
                cur_shm = shared_memory.SharedMemory(name=name)
                # attach numpy array buffer to shared memory:
                ar = np.ndarray(shape, dtype=dtype, buffer=cur_shm.buf)
                # add to current list:
                cur_v.append(ar)
            # replace in run_config:
            rc[tk] = cur_v

"""


#### FileSystem Soln
# rc holds filenames --> each process loads and replaces
# NOTE: save complete path


# make directories here
def convert_np_to_fn(rc):
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper']
    root_dir = rc['dir_str']

    # if directory exists --> stop
    # else --> create and populate it
    if(not os.path.isdir(root_dir)):
        os.mkdir(rc['dir_str'])

    for tk in target_keys:
        if(tk in rc):
            full_path = os.path.join(root_dir, tk+'.npy')
            np.save(full_path, rc[tk]*1)
            rc[tk] = full_path

    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            full_path = os.path.join(root_dir, tk+'.npz')
            np.savez(full_path, *rc[tk])
            rc[tk] = full_path

# convert npz to list
# takes in the npz dictionary
def conv_npz_l(d):
    l = []
    for i in range(len(d.keys())):
        l.append(d['arr_' + str(i)])
    return l


# each process should call this to load the required data
def convert_fn_to_np(rc):
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper']
    for tk in target_keys:
        if(tk in rc):
            rc[tk] = np.load(rc[tk]) > 0.5

    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            rc[tk] = conv_npz_l(np.load(rc[tk]))


if(__name__ == '__main__'):


    VOLS_PER_SECOND = 1.5
    CELL_MANIFEST = ['AVA', 'RME', 'SMDV', 'SMDD', 'ON', 'OFF']

    ## New approach:
    # 1. use On/Off cells (when available)
    # 2. every set gets its own condition
    # 3. use previous 24 timepoints 

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

    # collect all data as different stims
    Yf0 = [Y,Y_zim,Y_ztcbuf,Y_jhbuf]
    inp0 = [inp,inp_zim,inp_ztcbuf,inp_jhbuf]

    # replace nans with 0s
    for i, Yc in enumerate(Yf0):
        for j, Ysub in enumerate(Yc):
            naninds = np.isnan(Yf0[i][j])
            Yf0[i][j][naninds] = 0.0


    ## Tree Boosting Analysis

    Y2 = Yf0

    # prediction type:
    fn_set = 'full'
    fn_pred = 'RA'
    targ_cells = np.array([0,1])

    # general params: 
    in_cells = np.array([0,1,2,3])
    num_tree_cell = len(in_cells)
    in_cells_offset = np.array([4,5])
    num_boot_hyper = 5
    num_boot = 1000

    # build tensors for each subset:
    Xfs_l, olabs_l, worm_ids_l, fb = [], [], [], []
    # iter thru conditions:
    for i, Yc in enumerate(Y2): 
        basemus, X, worm_ids, t0s = mt.build_tensors(Yc, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)

        # get labels and filtered X
        olab = mt.label_basemus(basemus, thrs=[-.06, -.01, .01, .06])
        Xf,fb = mt.filterX(X)

        # save:
        Xfs_l.append(Xf)
        olabs_l.append(olab)
        worm_ids_l.append(worm_ids)

    np.save('timing_fb.npy', fb)

    # hyper set handling:
    try:
        hyper_inds = np.load(fn_set + fn_pred + 'hyper_inds') > 0.5
        train_sets_hyper = np.load(fn_set + fn_pred + 'train_sets_hyper') > 0.5
        test_sets_hyper = np.load(fn_set + fn_pred + 'test_sets_hyper') > 0.5
    except:
        tot_size = sum([np.shape(xfi)[0] for xfi in Xfs_l])
        hyper_inds = npr.rand(tot_size) < 0.3
        np.save(fn_set + fn_pred + 'hyper_inds', hyper_inds*1)
        # hyperparameter train/test set generation:
        train_sets_hyper, test_sets_hyper = mt.generate_traintest(tot_size, num_boot_hyper, hyper_inds, hyper_inds, train_perc=0.95)
        np.save(fn_set + fn_pred + 'train_sets_hyper.npy', train_sets_hyper*1)
        np.save(fn_set + fn_pred + 'test_sets_hyper.npy', test_sets_hyper*1)
        train_sets_hyper = train_sets_hyper > 0.5
        test_sets_hyper = test_sets_hyper > 0.5
       

    # try loading train/test sets:
    try:
        print('loading sets')
        train_sets = np.load(fn_set + fn_pred + '_trainsets.npy') > 0.5
        test_sets = np.load(fn_set + fn_pred + '_testsets.npy') > 0.5
    except:
        tot_size = sum([np.shape(xfi)[0] for xfi in Xfs_l])
        trainable_inds = np.ones(tot_size) > 0.5
        testable_inds = np.logical_not(hyper_inds)
        train_sets, test_sets = mt.generate_traintest(tot_size, num_boot, trainable_inds, testable_inds)
        np.save(fn_set + fn_pred + '_trainsets.npy', train_sets*1)
        np.save(fn_set + fn_pred + '_testsets.npy', test_sets*1)
        train_sets = train_sets > 0.5
        test_sets = test_sets > 0.5


    # Xf network
    Xf_net = [Xf[:,:,:4,:] for Xf in Xfs_l]
    # Xf stim ~ On/Off cells
    Xf_stim = [Xf[:,:,4:6,:] for Xf in Xfs_l]
    # no stim:
    Xf_nos = [0.0*Xf[:,:,4:6,:] for Xf in Xfs_l]

    # load numpy data structures into shared memory arrays
    #sh_hyper_inds = shared_memory.SharedMemory(create=True, size=hyper_inds.nbytes)


    """

    ## HYPER

    ## experiment: with stimulus context vs. without
 
    # base run_config:
    mode = 3
    run_id = fn_set + fn_pred + 'mode' + str(mode) + 'STIM'
    rc = mt.get_run_config(mode, run_id)
    rc['tree_depth'] = [2,2]
    mt.add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_nos, worm_ids_l, olabs_l, train_sets_hyper,
            test_sets_hyper) 
    rc['l1_tree'] = [.01, 2.0]
    rc['l1_mlr_xf1'] = [.01, 1.0]

    # NOTE: converting here --> all data stored in first directory
    # save big datastructures to lists... pass around filenames:
    convert_np_to_fn(rc)

    # dstruct contains all the different rcs:
    dstruct = [rc]

    # get hyperparameter combos:
    
    # Xf_net l1 
    #s = 'l1_mlr_xf1' 
    #vals = [[.01, 1.0], [.02, 1.0], [.04,1.0], [.02, 2.0], [.01, 2.0], [.04, 2.0]]
    #dstruct = mt.new_run_config_axis(dstruct, s, vals)
    # Xf_stim l1
    s = 'l1_mlr_xf2' 
    vals = [[.05, 1.0], [.1, 1.0], [.15, 1.0], [.05, 2.0], [.1, 2.0], [.15, 2.0], [.1, 0.5], [.2, 0.5], [.2,0.25]]
    dstruct = mt.new_run_config_axis(dstruct, s, vals)
    # boost depth:
    #s = 'tree_depth'
    #vals = [[2,2],[2,1]]
    #dstruct = mt.new_run_config_axis(dstruct, s, vals)

    # hyperparameter testing
    def bch(rc): 
        convert_fn_to_np(rc)
        return mt.boot_cross_hyper(rc)

    # limit number of processes in pool
    MAXPROC = 4
    with Pool(MAXPROC) as p:
        p.map(bch, dstruct)

    """
    
    ## CROSS

    """
    # stim config
    mode = 3 # ~ random slope
    run_id = fn_set + fn_pred + 'mode' + str(mode) + 'STIMCELLTEST'
    rc = mt.get_run_config(mode, run_id)
    mt.add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids_l, olabs_l, train_sets_hyper,
            test_sets_hyper) 
    rc['tree_depth'] = [2,2]
    rc['l1_tree'] = [.01, 2.0]
    rc['l1_mlr_xf1'] = [.01, 1.0]
    rc['l1_mlr_xf2'] = [0.15, 1.0]
    """

    """
    # stim config variable
    mode = 3 # ~ random slope
    run_id = fn_set + fn_pred + 'mode' + str(mode) + 'STIMCELLTESTVAR'
    rc = mt.get_run_config(mode, run_id)
    mt.add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids_l, olabs_l, train_sets_hyper,
            test_sets_hyper) 
    rc['tree_depth'] = [2,2]
    rc['l1_tree'] = [.01, 2.0]
    rc['l1_mlr_xf1'] = [.01, 1.0]
    rc['l1_mlr_xf2'] = [0.2, 0.5]

    """

    # nostim config 
    mode = 3 # ~ random slope
    run_id = fn_set + fn_pred + 'mode' + str(mode) + 'NOSTEST'
    rc = mt.get_run_config(mode, run_id)
    mt.add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_nos, worm_ids_l, olabs_l, train_sets_hyper,
            test_sets_hyper) 
    rc['tree_depth'] = [2,2]
    rc['l1_tree'] = [.01, 2.0]
    rc['l1_mlr_xf1'] = [.01, 1.0]
    rc['l1_mlr_xf2'] = [0.1, 1.0]

    # run:
    mt.boot_cross_boosted(rc)

