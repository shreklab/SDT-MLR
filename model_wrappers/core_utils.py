'''

    Core Utilities used by All Wrappers

    Ex: filtering data, chunking data, building masks


'''

import numpy as np
import numpy.random as npr
import os

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



## Cross-validation Set Generation

# generate arrays of training / test inds
# takes in 1. number of blocks, 2. trainable inds, 3. testable inds
# ... last 2 should match number of blocks and should be booleans
# --> sample training inds --> make new testable set --> sample testable inds
# returns: num_boot x num_blocks integer arrays...1s = true, 0s = false
# sample from test first --> everything else goes in train... will work best if testable is subset of trainable
def generate_traintest(num_blocks, num_boot, trainable_inds, testable_inds, test_perc=0.5): 
    assert(test_perc > 0.0 and test_perc < 1.0), 'illegal test set percentage'
    train_sets = np.zeros((num_boot, num_blocks))
    test_sets = np.zeros((num_boot, num_blocks))
    # make index sets:
    test_index_set = np.where(testable_inds)[0]

    # test set size: 
    num_use_test = int(test_perc * len(test_index_set))

    # iter thru boots:
    for i in range(num_boot):

        # shuffle test_index_set --> sample test inds:
        npr.shuffle(test_index_set)
        test_sample = test_index_set[:num_use_test]

        # save current test sample:
        test_sets[i,test_sample] = 1

        ## everything else goes in training:
        # 1. assign all trainable to train --> 2. 0 out test_sample from this
        train_sets[i,trainable_inds] = 1
        train_sets[i,test_sample] = 0

    return train_sets, test_sets



## IO

# save metadata:
# save following types;
# 1. int, 2. float, 3. list, 4. tuple
# format: each line is a different hyperparam
# ... param_name: vals
def save_metadata(rc, targ_strs=['dir_str']):
    bstr = ''
    for ts in targ_strs:
        if(ts in rc):
            bstr = bstr + ts + ': ' + str(rc[ts]) + '\n'
    text_file = open(os.path.join(rc['dir_str'], 'metadata.txt'),'w')
    text_file.write(bstr)
    text_file.close()



# scalar convert:
def scalar_conv(str_v):
    try:
        v = int(str_v)
    except:
        v = float(str_v)
    return v


# list convert:
def list_conv(str_l):
    # strip list stuff:
    str_l = str_l.strip('[]()')
    # split based on spaces:
    lst = str_l.split(' ')
    v = [scalar_conv(i) for i in lst]
    return v


# load metadata:
def load_metadata(rc, fn):
    text_file = open(fn, 'r')
    L = text_file.readlines()
    for line in L:
        # split on : to isolate indicators:
        lspl = line.split(': ')
        indic = lspl[0]
        # check if list type:
        if('[' in lspl[1] or '(' in lspl[1]): 
            rc[indic] = list_conv(lspl[1])
        else: # scalar type:
            rc[indic] = scalar_conv(lspl[1])


