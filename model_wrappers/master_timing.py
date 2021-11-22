'''

    Master Timing
    Runs all timing experiments
    Uses st_mlr


'''

import tensorflow as tf
import numpy as np
import numpy.random as npr
import pylab as plt
import os
import copy

from ..models import st_mlr 


#### Wavelets
# run cell histories thru wavelets for downsampling
# cell history shapes = T x num_cell x tau
# currently: use gaussian wavelets

# make wavelets
# --> num_wave x tau
def make_wave(mu, var, hist_len):
    t = np.arange(hist_len)
    rawwave = np.exp(-(t - mu)**2.0 / var)
    return rawwave / np.sum(rawwave)


# filterbank = composed of waves
def make_fb(hist_len, muvars):
    fb = []
    for i in range(len(muvars)):
        fb.append(make_wave(muvars[i][0], muvars[i][1], hist_len))
    fb = np.array(fb)
    return fb


# filter X
# expected shape for X = num_block x T per block x in_cell x hist_len (tau)
def filterX(X):
    hl = np.shape(X)[-1]
    # some reasonable muvars:
    muvars = [[hl, 6.0], [hl, 12.0], [.75*hl, 6.0], [.75*hl, 12.0], [.5*hl, 8.0], [.5*hl, 16.0], [.25*hl, 24.0], [.6*hl,24.0]]
    fb = make_fb(hl, muvars)
    # get filtered X:
    fX = np.tensordot(X, fb, axes=(3, 1))
    return fX, fb


#### Build Masks
# hist_mask = 1 x num_tree x 1 x in_cell x hist_len
# wf_mask = 1 x num_tree x 1 x num_worm+1
# mask_cells = list of numpy arrays
# l1s = list of scalars == matches mask_cells
def build_hist_mask(num_tree, num_in_cell, hist_len, mask_cells=[], l1s=[]):
    assert(len(mask_cells) == len(l1s)), 'hist mask: mask_cells l1s mismatch'
    hist_mask = np.zeros((1, num_tree, 1, num_in_cell, hist_len))
    for i in range(len(mask_cells)):
        hist_mask[:,:,:,mask_cells[i],:] = l1s[i]
    return hist_mask


def build_wf_mask(num_tree, num_worm, l1=1.0):
    wf_mask = np.zeros((1, num_tree, 1, num_worm+1))
    wf_mask[:,:,:,1:] = l1
    return wf_mask


#### Build the tensors
# need to know: in_cells, targ_cells, dt, hist_len, block_size
# builds: basemus (targ_cells), X (in_cells), worm_ids
# operates on list of arrays
# output shapes = num_blocks x T per block x ...

def build_tensors(Y, targ_cells, in_cells, in_cells_offset, dt=8, hist_len=24, block_size=16):
    assert(block_size > dt), 'block size - dt mismatch'
    basemus = []
    X = []
    worm_ids = []
    t0s = []
    # iter thru worms:
    for i in range(len(Y)):
        basemus_worm = []
        X_worm = []
        worm_ids_worm = []
        t0s_worm = []
        # iter thru blocks:
        for j in range(hist_len,np.shape(Y[i])[0]-block_size,block_size):
            basemus_block = []
            X_block = []
            worm_ids_block = []
            t0s_block = []

            # iter thru windows within block:
            for k in range(block_size - dt):
                t0 = j + k
                # save current t0:
                t0s_block.append(t0)
                # basemus:
                bm = Y[i][t0-1:t0+dt,targ_cells]
                basemus_block.append(np.mean(bm[1:,:] - bm[:1,:], axis=0))
                # X ~ incells:
                Xnon = Y[i][t0-hist_len:t0,in_cells].T
                # X ~ incells + offset:
                Xoff = Y[i][t0-hist_len+dt:t0+dt,in_cells_offset].T
                # save X:
                X_block.append(np.vstack((Xnon, Xoff)))
                # worm_ids:
                wid = np.zeros((len(Y)+1))
                wid[0] = 1
                wid[i+1] = 1
                worm_ids_block.append(wid)

            # save all data for given worm:
            basemus_worm.append(np.array(basemus_block))
            X_worm.append(np.array(X_block))
            worm_ids_worm.append(np.array(worm_ids_block))
            t0s_worm.append(np.array(t0s_block))

        basemus.append(np.array(basemus_worm))
        X.append(np.array(X_worm))
        worm_ids.append(np.array(worm_ids_worm))
        t0s.append(np.array(t0s_worm))

    return np.vstack(basemus), np.vstack(X), np.vstack(worm_ids), np.vstack(t0s)


# label basemus (build olab)
# basemus = num_blocks x T per block x targ/out_cells
def label_basemus(basemus, thrs=[-.1,-.05, -.02, 0.0, .02, .05, .1]):
    # add end thresholds:
    thrs = [np.amin(basemus)-1] + thrs + [np.amax(basemus)+1]
    olab = np.zeros((np.shape(basemus)[0], np.shape(basemus)[1], np.shape(basemus)[2], len(thrs)-1))
    # iter thru blocks:
    for i in range(np.shape(basemus)[0]):
        # iter thru cells:
        for j in range(np.shape(basemus)[2]):
            # iter thru thresholds:
            for k in range(1,len(thrs)):
                inds = np.logical_and(basemus[i,:,j] >= thrs[k-1], basemus[i,:,j] < thrs[k])
                olab[i,inds,j,k-1] = 1
    return olab


#### Pulse Expansion
# takes in input-cell pair
# --> selects cell trace where pulse duration is in specified range
# operates on a single trace
# only consider 1s region --> to get offs, pass in 1 - inp_tr
def pulse_expand(cell_tr, inp_tr, low_thr, hi_thr):
    # append 0 to beginning and end --> guarantees beginning and completion
    inp_tr2 = np.hstack(([0],inp_tr,[0]))
    # onsets and offsets ~ guaranteed to be same length due to prev line:
    onsets = np.where(inp_tr2[1:] > inp_tr2[:-1])[0]
    offsets = np.where(inp_tr2[1:] < inp_tr2[:-1])[0]
    durs = offsets - onsets
    cell_out = 0.0 * cell_tr
    for i in range(len(onsets)):
        if(durs[i] >= low_thr and durs[i] <= hi_thr):
            cell_out[onsets[i]:offsets[i]] = cell_tr[onsets[i]:offsets[i]]
    return cell_out




## Cross-validation Set Generation


# generate arrays of training / test inds
# takes in 1. number of blocks, 2. trainable inds, 3. testable inds
# ... last 2 should match number of blocks and should be booleans
# --> sample training inds --> make new testable set --> sample testable inds
def generate_traintest(num_blocks, num_boot, trainable_inds, testable_inds, train_perc=0.5): 
    train_sets = np.zeros((num_boot, num_blocks))
    test_sets = np.zeros((num_boot, num_blocks))
    # iter thru boots:
    for i in range(num_boot):
        # sample train inds:
        train_inds = np.logical_and(trainable_inds, npr.rand(num_blocks) < train_perc)
        train_sets[i,:] = train_inds
        # test inds = everything in testable and not in train inds:
        test_inds = np.logical_and(np.logical_not(train_inds), testable_inds)
        test_sets[i,:] = test_inds
    return train_sets, test_sets



#### Boosted Tree Analyses

# random slope xf
def generate_rand_slope_xf(Xf_l):
    # shared term:
    Xf_shared = np.concatenate(Xf_l, axis=0)
    xf_l = [Xf_shared]
    count = 0
    for i, xf in enumerate(Xf_l):
        npc = 0*Xf_shared
        # current xf entry
        npc[count:count+np.shape(xf)[0]] = xf
        # add to list:
        xf_l.append(npc)
        # update count
        count += np.shape(xf)[0]
    # combine across cells (could combine aross last dim as well... doesn't matter)
    return np.concatenate(xf_l, axis=2)


# random slope + l1 mask
def generate_rand_slope_l1(Xf_l, l1_base, l1_variance): 
    l1_base_np = np.ones((np.shape(Xf_l[0])[-1]*np.shape(Xf_l[0])[-2])) * l1_base
    l1_l = [l1_base_np]
    for i in range(len(Xf_l)):
        l1_l.append(l1_base_np * l1_variance)
    return np.hstack((l1_l)), l1_base_np


# modes: boosting experiments
# 0(or other): does not use Xf2
# 1: stack Xf1 and Xf2 in drive/mlr
# 2: separate model set for Xf2
# 3: tree only recieves fixed components (same as 2 otherwise) 
# 4: same as 2 + stack Xf1 and Xf2 in second gate network ~ test stimulus context

# boost lists for random slope models
# ... make 'common data' = shared across conditions
# ... stack condition-speicific data copies on common data
# ... use l1 factors to further penalize additional terms
# ... all l1s are now lists/tuples = (base l1, l1_factor for group terms) 
# Xf1 shape = num_block x t_per_block x in_cell x xdim
# NOTE: Xf1 should be same shape across conditions
def boost_lists_randslope(Xf1, Xf2, worm_ids, l1_tree, l1_mlr_xf1, l1_mlr_xf2, l1_mlr_wid, wid0_factor=0.5, mode=0):
    # Xf1
    Xf1_stack = generate_rand_slope_xf(Xf1) 
    # Xf2:
    Xf2_stack = generate_rand_slope_xf(Xf2)
    # l1 tree:
    l1_tree_mask, l1_tree_mask_base = generate_rand_slope_l1(Xf1, l1_tree[0], l1_tree[1])
    # l1 mlr xf1:
    l1_xf1mlr_mask, l1_xf1mlr_mask_base = generate_rand_slope_l1(Xf1, l1_mlr_xf1[0], l1_mlr_xf1[1])
    # l1 mlr xf2:
    l1_xf2mlr_mask, l1_xf2mlr_mask_base = generate_rand_slope_l1(Xf2, l1_mlr_xf2[0], l1_mlr_xf2[1])
    # l1 mlr wid: intercepts...doesn't get tiled 
    wid_mask = np.ones((np.shape(worm_ids)[-1])) * l1_mlr_wid
    wid_mask[0] = wid0_factor

    if(mode==1):
        Xf_list = [[[Xf1_stack],[Xf1_stack, Xf2_stack, worm_ids]]]
        model_masks = [[[l1_tree_mask],[l1_xf1mlr_mask, l1_xf2mlr_mask, wid_mask]]]
    elif(mode==2):
        Xf_list = [[[Xf1_stack],[Xf1_stack,worm_ids]], [[Xf1_stack],[Xf2_stack]]]
        model_masks = [[[l1_tree_mask],[l1_xf1mlr_mask, wid_mask]], [[l1_tree_mask], [l1_xf2mlr_mask]]]
    elif(mode==3):
        Xf1_small_stack = np.vstack(Xf1)
        l1_tree_small_stack = np.vstack(l1_tree_mask_base)
        Xf_list = [[[Xf1_small_stack],[Xf1_stack,worm_ids]], [[Xf1_small_stack],[Xf2_stack]]]
        model_masks = [[[l1_tree_small_stack],[l1_xf1mlr_mask, wid_mask]], [[l1_tree_small_stack], [l1_xf2mlr_mask]]]
    elif(mode==4):
        Xf_list = [[[Xf1_stack],[Xf1_stack,worm_ids]], [[Xf1_stack,Xf2_stack],[Xf2_stack]]]
        model_masks = [[[l1_tree_mask],[l1_xf1mlr_mask, wid_mask]], [[l1_tree_mask,l1_xf2mlr_mask], [l1_xf2mlr_mask]]]
    else: # mode 0 / default
        Xf_list = [[[Xf1_stack],[Xf1_stack, worm_ids]]]
        model_masks = [[[l1_tree_mask],[l1_xf1mlr_mask, wid_mask]]]

    return Xf_list, model_masks


# join list of worm_ids (across conditions) into a single worm_id
# EX: if 2 conditions and 17 worms each --> 35 dims output
# each worm_id tensor = num_batch x t_per_batch x num_worms + 1
def join_worm_ids_l(worm_ids_l):
    # new number of worms:
    num_worms = [np.shape(wid)[-1] for wid in worm_ids_l]
    num_worms = sum(num_worms) - len(worm_ids_l) # remove redundent intercepts
    # new larger list of worm ids:
    new_wids_l = []
    counter = 0
    for wid in worm_ids_l:
        sh_cur = list(np.shape(wid))
        sub_worms = sh_cur[-1]-1
        sh_cur[-1] = num_worms
        new_counter = counter + sub_worms
        add_wid = np.zeros(sh_cur)
        add_wid[:,:,counter:new_counter] = wid[:,:,1:]
        counter = new_counter
        new_wids_l.append(add_wid)
    out_wids = np.vstack(new_wids_l)
    # add true intercept back in:
    add_cept = np.ones((np.shape(out_wids)[0],np.shape(out_wids)[1],1))
    out_wids = np.concatenate((add_cept, out_wids), axis=-1)
    return out_wids


# save metadata:
def save_metadata(rc):
    meta_strs = ['l1_tree', 'l1_mlr_xf1', 'l1_mlr_xf2', 'num_model', 'num_epoch', 'mode', 'lr', 'even_reg', 'tree_depth', 'tree_width']
    bstr = ''
    for ms in meta_strs:
        bstr = bstr + ms + ': ' + str(rc[ms]) + '\n'
    text_file = open(os.path.join(rc['dir_str'], 'metadata.txt'),'w')
    text_file.write(bstr)
    text_file.close()


# greedy forest mask
# 1. find best performing model
# 2. iter thru other models --> best model to add to first for forest err?
# 3. repeat 2
def make_greedy_forest_mask(rc, B, dat_train, N=3): 
    num_model = rc['num_model']
    f_mask = np.zeros((num_model))
    for i in range(N): 
        print('cur fmask')
        print(f_mask)
        errs = []
        for j in range(num_model):
            if(f_mask[j] > 0.5):
                errs.append(np.nan)
                continue
            f_mask_copy = copy.deepcopy(f_mask)
            f_mask_copy[j] = 1
            f_mask_copy = f_mask_copy / np.sum(f_mask_copy)
            # evaluate train error:
            f_loss = B.forest_loss(dat_train[0], dat_train[1], f_mask_copy.astype(np.float32)).numpy()
            errs.append(f_loss)
        print(errs)
        # save best:
        f_mask[np.nanargmin(np.array(errs))] = 1
    # normalize and set to float:
    f_mask = f_mask / np.sum(f_mask)
    print('final f_mask')
    print(f_mask)
    return f_mask.astype(np.float32)


## Out-of-bootstrap cross-validation on hyper set
# difference between hyper and test set?
# fit whole model to hyper set... test set strategy fits to hyper then to test
# hyperparameters specified in different run config directories
def boot_cross_hyper(rc): 

    # if directory exists --> stop
    # else --> create and populate it
    if(not os.path.isdir(rc['dir_str'])):
        os.mkdir(rc['dir_str'])
    # save metadata:
    save_metadata(rc)

    # build Xf_list and model masks:
    worm_ids = join_worm_ids_l(rc['worm_ids'])
    Xf_list, model_masks = boost_lists_randslope(rc['Xf_net'], rc['Xf_stim'], worm_ids, rc['l1_tree'], rc['l1_mlr_xf1'], rc['l1_mlr_xf2'], rc['l1_mlr_wid'], mode=rc['mode'])
    # stack olab:
    rc['olab'] = np.vstack(rc['olab']) 

    # glean info from data structs:
    output_cells = np.shape(rc['olab'])[-2]
    output_classes = np.shape(rc['olab'])[-1]

    # hyper train and test errors:
    htre_errs = []

    # iter thru hyper train sets:
    for i in range(np.shape(rc['train_sets_hyper'])[0]): 

        train_inds = rc['train_sets_hyper'][i]
        test_inds = rc['test_sets_hyper'][i]

        # build data structure for current train/test sets:
        dat_hyper_train, dat_hyper_test = st_mlr.dat_gen(rc['olab'], Xf_list, train_inds, test_inds)

        xdims = st_mlr.get_xdims(dat_hyper_train)

        # build architecture ~ used for all boots:
        B = st_mlr.arch_gen(output_cells, output_classes, xdims, model_masks,
                rc['tree_depth'], rc['tree_width'], rc['num_model'], rc['lr'], even_reg=rc['even_reg'])

        ## initial fit to hyper set:
        # mode == '' in this case --> train full model
        tr_errs, te_errs = st_mlr.train_epochs_wrapper(B, dat_hyper_train, dat_hyper_test, num_epochs=rc['num_epoch'], mode='')
        # 3 best trees form the mask:
        #sinds = np.argsort(tr_errs[-1])
        #f_mask = np.zeros((rc['num_model']))
        #f_mask[sinds[:3]] = (1.0/3.0)
        #f_mask = f_mask.astype(np.float32)

        # greedily form mask:
        f_mask = make_greedy_forest_mask(rc, B, dat_hyper_train, N=rc['num_model_test'])

        # forest train error:
        f_loss_train = B.forest_loss(dat_hyper_train[0], dat_hyper_train[1], f_mask).numpy()
        f_loss_test = B.forest_loss(dat_hyper_test[0], dat_hyper_test[1], f_mask).numpy()
        htre_errs.append([f_loss_train, f_loss_test])
        np.save(os.path.join(rc['dir_str'], 'hyper_err'), np.array(htre_errs))



## Out-of-bootstrap cross-validation
# goal: correct number of network states?
# works for single number of states
# fit full model to hyper set --> fit log reg models for each bootstrap
# Assumes: tree_depths / tree_widths = lists
# modes:
# 0: does not use Xf2
# 1: stack Xf1 and Xf2 in drive/mlr
# 2: separate model set for Xf2 (boosting)
# rc = run_config dictionary
def boot_cross_boosted(rc): 

    # if directory exists --> stop
    # else --> create and populate it
    if(not os.path.isdir(rc['dir_str'])):
        os.mkdir(rc['dir_str'])
    # save metadata:
    save_metadata(rc)

    # build Xf_list and model masks:
    worm_ids = join_worm_ids_l(rc['worm_ids'])
    Xf_list, model_masks = boost_lists_randslope(rc['Xf_net'], rc['Xf_stim'], worm_ids, rc['l1_tree'], rc['l1_mlr_xf1'], rc['l1_mlr_xf2'], rc['l1_mlr_wid'], mode=rc['mode'])
    # stack olab:
    rc['olab'] = np.vstack(rc['olab']) 

    ## build data structures for hyper set:
    dat_hyper, null = st_mlr.dat_gen(rc['olab'], Xf_list, rc['hyper_inds'], rc['hyper_inds'])

    # glean info from data structs:
    output_cells = np.shape(rc['olab'])[-2]
    output_classes = np.shape(rc['olab'])[-1]
    xdims = st_mlr.get_xdims(dat_hyper)

    # build architecture ~ used for all boots:
    B = st_mlr.arch_gen(output_cells, output_classes, xdims, model_masks,
            rc['tree_depth'], rc['tree_width'], rc['num_model'], rc['lr'], even_reg=rc['even_reg'])

    ## initial fit to hyper set:
    # mode == '' in this case --> train full model
    tr_errs, te_errs = st_mlr.train_epochs_wrapper(B, dat_hyper, dat_hyper, num_epochs=rc['num_epoch'], mode='')
    # 3 best trees form the mask:
    #sinds = np.argsort(tr_errs[-1])
    #f_mask = np.zeros((rc['num_model']))
    #f_mask[sinds[:3]] = (1.0/3.0)
    #f_mask = f_mask.astype(np.float32)

    # greedily form mask:
    f_mask = make_greedy_forest_mask(rc, B, dat_hyper, N=rc['num_model_test'])
    sinds = np.where(f_mask > 0)[0]
    print('sinds')
    print(sinds)

    ## fit mlr/drives to each bootstrap
    # --> save forest errors
    save_loss = []
    save_train_loss = []
    save_train_vars = []
    for i in range(np.shape(rc['train_sets'])[0]):
        train_inds = rc['train_sets'][i]
        test_inds = rc['test_sets'][i]

        # regenerate datasets:
        dat_train, dat_test = st_mlr.dat_gen(rc['olab'], Xf_list, train_inds, test_inds)

        # mode=mlr --> trains only driver models = convex
        tr_errs, te_errs = st_mlr.train_epochs_wrapper(B, dat_train, dat_test, num_epochs=rc['num_epoch'], mode='mlr')

        # get forest error on test set:
        f_loss = B.forest_loss(dat_test[0], dat_test[1], f_mask).numpy()
        save_loss.append(f_loss)
        np.save(os.path.join(rc['dir_str'], 'boosted_test_err'), np.array(save_loss))

        # get forest error on train set:
        f_loss_tr = B.forest_loss(dat_train[0], dat_train[1], f_mask).numpy()
        save_train_loss.append(f_loss_tr)
        np.save(os.path.join(rc['dir_str'], 'boosted_train_err'), np.array(save_train_loss))


        # save training vars:
        if(len(save_train_vars) == 0):
            # initialize:
            for i in range(len(B.model_pairs)):
                for j in range(len(B.model_pairs[i])):
                    save_train_vars.append([B.model_pairs[i][j].get_analysis_vars(sinds)])
        else:
            # zip:
            count = 0
            for i in range(len(B.model_pairs)):
                for j in range(len(B.model_pairs[i])):
                    save_train_vars[count].append(B.model_pairs[i][j].get_analysis_vars(sinds))
                    count += 1
        np.savez(os.path.join(rc['dir_str'], 'boosted_tv'), *save_train_vars)
    return B, f_mask, sinds



# get basic run configuration
# returns python dict with default model parameters
# mode 0 default = no stimulus, 4 state, no lr
# mode 2 = stimulus boosting, 4 states for each, no lr
def get_run_config(mode, run_id):
    d = {}
    # fn string from run id:
    d['dir_str'] = run_id
    # empty data ~ need to fill this in after call:
    d['hyper_inds'] = []
    d['train_sets'] = []
    d['test_sets'] = []
    d['Xf_net'] = []
    d['Xf_stim'] = []
    d['worm_ids'] = []
    d['olab'] = []
    # shared tree info:
    d['l1_tree'] = [.01,1.0] # l1 regularization term for tree features
    d['l1_mlr_xf1'] = [.01,1.0] # l1 regularization term for baseline St-MLR model (and factor for variance terms)
    d['l1_mlr_xf2'] = [0.1,1.0] # l1 regularization term for secondary (typically stimulus) St-MLR model
    d['l1_mlr_wid'] = 0.1 # l1 regularization term for worm identity terms
    d['num_model'] = 25
    d['num_model_test'] = 3 
    d['num_epoch'] = 40
    d['mode'] = mode
    d['lr'] = [] # ranks for MLR models... if empty --> full rank
    d['even_reg'] = 0.1 # entropy regularization --> ensures similar amounts of data to each leaf
    if(mode in [2,3,4]): 
        d['tree_width'] = [2,2]
        d['tree_depth'] = [2,2]
    else:
        d['tree_width'] = [2]
        d['tree_depth'] = [2]
    return d



# add axis of variation to run config list
# each extant run_config gets duplicated for str, value pairs
# skey = string key into rc (run_config)
def new_run_config_axis(rc_list, s, vals):
    new_rc_list = []
    for rc in rc_list:
        for count, v in enumerate(vals):
            rc2 = copy.deepcopy(rc)
            rc2[s] = v
            rc2['dir_str'] = rc['dir_str'] + '_' + str(count)
            new_rc_list.append(rc2)
    return new_rc_list



# add data to run_config
def add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids, olab, train_sets_hyper=[], \
        test_sets_hyper=[]):  
    rc['hyper_inds'] = hyper_inds
    rc['train_sets'] = train_sets
    rc['test_sets'] = test_sets
    rc['Xf_net'] = Xf_net
    rc['Xf_stim'] = Xf_stim
    rc['worm_ids'] = worm_ids
    rc['olab'] = olab
    rc['train_sets_hyper'] = train_sets_hyper
    rc['test_sets_hyper'] = test_sets_hyper



