'''

    Separator ~ Essentially state-based Granger Causality

    Goal: compare N submodels across different states
    --> Soft Tree gating model to specify states
    --> Compare MLR submodels on each of the states

'''

import tensorflow as tf
import numpy as np
import numpy.random as npr
import os
import copy

from . import core_models


#### MLR comparator class
# Goal: compare multiple MLR submodels for each state
# NOTE: this class is 100% different from other MLR classes
class MultiLogRegComp(core_models.Driver):

    # num_models = number of overall models fit in parallel
    # num_state = perform a different comparison for each state
    # models_per_state = number of models we're comparing for each state
    # l1_state = constrains model components to be similar across states
    # Assumes: l1/l2 masks = scalars or (len = xdim) or (submodels per state x xdim)
    # TODO: fixed filter for overall calc?
    def __init__(self, num_models, num_state, submodels_per_state, xdim, output_cells, output_classes, l1_mask=0.0, l2_state_mask=0.0): 
        self.num_models = num_models
        self.num_state = num_state
        self.models_per_state = submodels_per_state
        self.xdim = xdim
        self.output_cells = output_cells
        self.output_classes = output_classes
        
        ## l-norm regularizers:
        # l1:
        self.l1_mask = self.build_mask(l1_mask)
        # l2 state:
        self.l2_state_mask = self.build_mask(l2_state_mask)

        self.build_structs()
        self.train_vars = [self.X_struct]



    # build mask
    # 3 options: 
    # 1. l_mask = scalar assigned to all values
    # 2. shape(l_mask) = xdim... all submodels_per_state get same l_mask
    # 3. shape(l_mask) = submodels_per_state x xdim
    def build_mask(self, l_mask):
        lm = tf.constant(0.0)
        if(isinstance(l_mask, float) or isinstance(l_mask,int)):
            lm = tf.constant(l_mask)
        elif(len(np.shape(l_mask)) == 1): # same mask across submodels
            if(len(l_mask) == self.xdim):
                lm = tf.constant(np.reshape(l_mask, (1,1,1,1,1,-1)).astype(np.float32))
        else:
            if(np.shape(l_mask) == (self.models_per_state, self.xdim)):
                lm = tf.constant(np.reshape(l_mask, (1, 1, self.models_per_state, 1, 1, self.xdim)).astype(np.float32))
        
        # TESTING:
        print('lm = ')
        print(lm)
        return lm



    # get the analysis vars
    def get_analysis_vars(self, spec_mods=[]):
        st = self.X_struct.numpy()
        if(len(spec_mods) > 0):
            return st[spec_mods]
        return st


    # 1. gate vars (none here), 2. drive vars 
    def get_train_vars(self):
        return [], self.train_vars


    #### Build

    # build structures 
    def build_structs(self):
        # X structure
        # num_models x num_state x models_per_state x out_cells x out_classes x xdim
        self.X_struct = core_models.var_construct((self.num_models, self.num_state, self.models_per_state, self.output_cells, self.output_classes, self.xdim))

    
    # reinit structure: 
    def reinit_structs(self):
        newv = core_models.np_construct((self.num_models, self.num_state, self.models_per_state, self.output_cells, self.output_classes, self.xdim))
        self.X_struct.assign(newv)
   


    #### Evaluation

    # evaluation
    # returns: batch_size x num_models x num_state x models_per_state x out_cells x out_classes
    # Xf = batch_size x models_per_state x xdim
    # NOTE: softmaxed across out_classes 
    def eval(self, Xf):
        # reshape Xf --> batch_size x 1 (num_model) x 1 (num_state) x models_per_state x 1 (out_cells) x 1 (out_classes) x xdim
        Xf = tf.reshape(Xf, [-1, 1, 1, self.models_per_state, 1, 1, self.xdim])
        # run thru filter:
        # --> reduce across xdim (last dim)
        v = tf.reduce_sum(Xf * tf.expand_dims(self.X_struct, 0), axis=-1)

        # clip v to stop prevent overflow:
        v = tf.clip_by_value(v, -30.0, 30.0)

        # softmax across output classes
        return tf.nn.softmax(v, axis=-1)


    #### Regularizers

    # l1 loss (from 0)
    # X structure
    # num_models x num_state x models_per_state x out_cells x out_classes x xdim
    # average across num models and num_states
    def loss_l1(self):
        mst = self.l1_mask * tf.math.abs(self.X_struct)
        # reduce sum across out_cells/out_classes/xdim
        mst = tf.reduce_sum(mst, axis=[3,4,5])
        # average across num_models, num_state, models_per_state
        return tf.reduce_mean(mst)

   
    # l2 state loss ~ constrain models to be similar across states
    # 1. compute mean across states --> 2. subtract all
    # X structure
    # num_models x num_state x models_per_state x out_cells x out_classes x xdim
    # average across num models and num_states
    def state_loss_l2(self):
        mu = tf.reduce_mean(self.X_struct, axis=1, keepdims=True)
        di = self.X_struct - mu
        # square and scale
        mst = self.l2_state_mask * tf.math.pow(di, 2.0)
        # reduce sum across out_cells/out_classes/xdim
        mst = tf.reduce_sum(mst, axis=[3,4,5])
        # average across num_models, num_state, models_per_state
        return tf.reduce_mean(mst)


    def reg(self, Xf):
        return self.loss_l1() + self.state_loss_l2()



#### Separator
# TODO: Error funcs?
# 1. train all of the submodels separately
# 2. overall error = weighted sum by experiment_filter... tells us how to compare models_per_state

class Separator:

    # requires:
    # 1. Soft Forest object
    # 2. MultiLogRegComp object
    # 3. comparison filter: len = models_per_state... specifies relationship between submodels in overal objective
    def __init__(self, SoftForest, MultiLRC, comp_filter):
        assert(SoftForest.num_state == MultiLRC.num_state), 'Soft Trees and MLRC must have same number of states'
        assert(MultiLRC.models_per_state == len(comp_filter)), 'models_per_state must match comp_filter length'
        self.F = SoftForest
        self.MLR = MultiLRC
        # reshape comp_filter = 1 (batch_size) x 1 (num_models) x 1 (num_state) x models_per_state
        self.comp_filter = tf.constant(np.reshape(comp_filter, (1,1,1,-1)))

        # get train vars:
        tv_gate = self.F.get_train_vars()
        self.gate_train_vars = tv_gate[0] + tv_gate[1]
        tv_drive = self.MLR.get_train_vars()
        self.drive_train_vars = tv_drive[0] + tv_drive[1]

        # train operations
        self.train_op_drive = tf.keras.optimizers.Adam(.01)
        self.train_op_gate = tf.keras.optimizers.Adam(.01)


    # reinit MLR
    def reinit(self):
        self.MLR.reinit_structs()

    #### Error Functions

    # drive likelihood
    # log-likelihood for the drive models
    # assumed input shapes:
    # Xf_drive = batch_size x models_per_state x xdim
    # truths = batch_size x out_cells x out_classes
    # returns batch_size x num_models x num_state x models_per_state
    def drive_ll(self, truths, Xf_drive):
        # MLR eval
        # returns: batch_size x num_models x num_state x models_per_state x out_cells x out_classes
        # softmax'd across out_classes
        drive_pred = self.MLR.eval(Xf_drive)

        # expand truths
        # --> batch_size x 1 (num_models) x 1 (num_state) x 1 (models_per_state) x out_cells x out_classes
        truths = tf.expand_dims(truths, 1)
        truths = tf.expand_dims(truths, 1)
        truths = tf.expand_dims(truths, 1)

        # ll ~ reduce across out_cells, out_classes
        # --> batch_size x num_models x num_state x models_per_state
        ll = tf.reduce_sum(truths * tf.math.log(drive_pred), axis=[4,5])
        return ll


    # drive error
    # optimize all drive models
    # reduce drive_ll (do NOT apply comp_filter)
    def drive_err(self, truths, Xf_gate, Xf_drive):
        # get Soft Tree weights
        # --> batch_size x num_tree/num_models x num_leaf/num_state
        st_weights = self.F.eval(Xf_gate)
        
        # get drive log-likelihood:
        # --> batch_size x num_models x num_state x models_per_state
        dll = self.drive_ll(truths, Xf_drive)

        # scale dll by st_weights
        # reduce sum across batch_size, num_state
        # --> num_models x models_per_state
        ll1 = tf.reduce_sum(tf.expand_dims(st_weights,-1) * dll, axis=[0,2])

        # reduce average across models and models_per state
        ll2 = tf.reduce_mean(ll1)

        # negate to make into error:
        return -1.0 * ll2

    
    # overall error:
    # 1. get drive errors
    # 2. use comp_filter to compare models_per_state --> overall ll
    # 3. use Soft Forest to weight samples towards a specific state
    def overall_err(self, truths, Xf_gate, Xf_drive):
        # get Soft Tree weights
        # --> batch_size x num_tree/num_models x num_leaf/num_state
        st_weights = self.F.eval(Xf_gate)
        
        # get drive log-likelihood:
        # --> batch_size x num_models x num_state x models_per_state
        dll = self.drive_ll(truths, Xf_drive)

        # apply comp_filters to dll
        # --> batch_size x num_models x num_state
        cdll = tf.reduce_sum(dll * self.comp_filter, axis=-1)

        # scale cdll by st_weights
        # reduce sum across batch_size, num_state
        oll = tf.reduce_sum(st_weights * cdll, axis=[0,2])

        # reduce average across models (fit in parallel)
        oll = tf.reduce_mean(oll)

        # negate to make into error:
        return -1.0 * oll


    # reporting log-likelihood
    # goal: error reporting for each model within models_per_state
    # ... and for each overall model separately
    # returns: num_models x num_state x models_per_state (ll... DOES NOT return err)
    def reporting_ll(self, truths, Xf_gate, Xf_drive):
        # get Soft Tree weights
        # --> batch_size x num_tree/num_models x num_leaf/num_state
        st_weights = self.F.eval(Xf_gate)
        
        # get drive log-likelihood:
        # --> batch_size x num_models x num_state x models_per_state
        dll = self.drive_ll(truths, Xf_drive)

        # reduce across batch_size, num_state
        # --> num_models x num_state x models_per_state
        return tf.reduce_sum(tf.expand_dims(st_weights, -1) * dll, axis=0)



    #### Regularization

    # call regularization functions of gate/drive components
    def reg(self, Xf_gate, Xf_drive):
        r_gate = self.F.reg(Xf_gate)
        r_drive = self.MLR.reg(Xf_drive)
        return r_gate + r_drive


    #### Training Functions

    # drive training
    # use overall err... only optimize with drive_train_vars
    @tf.function
    def train_drive(self, dataset):
        for truths, Xf_gate, Xf_drive in dataset:
            with tf.GradientTape() as tape:
                L = self.drive_err(truths, Xf_gate, Xf_drive)
                L1 = self.reg(Xf_gate, Xf_drive)
                g = tape.gradient(L+L1, self.drive_train_vars)
                self.train_op_drive.apply_gradients(zip(g, self.drive_train_vars))

    # gate training
    # use overall err... only optimize with gate_train_vars
    @tf.function
    def train_gate(self, dataset):
        for truths, Xf_gate, Xf_drive in dataset:
            with tf.GradientTape() as tape:
                L = self.overall_err(truths, Xf_gate, Xf_drive)
                L1 = self.reg(Xf_gate, Xf_drive)
                g = tape.gradient(L+L1, self.gate_train_vars)
                self.train_op_gate.apply_gradients(zip(g, self.gate_train_vars))


    #### Testing Functions
    
    def test_overall_err(self, dataset):
        sum_err = []
        for truths, Xf_gate, Xf_drive in dataset:
            sum_err.append(self.overall_err(truths, Xf_gate, Xf_drive).numpy())
        return sum(sum_err)

    
    # returns num_models x num_state x models_per_state
    # summed over all batches
    def test_reporting_ll(self, dataset): 
        rll = []
        for truths, Xf_gate, Xf_drive in dataset:
            rll.append(self.reporting_ll(truths, Xf_gate, Xf_drive))
        return np.sum(np.array(rll),axis=0)

    
    # test state predictions
    # concatenate across batches --> T x num_tree x num_state
    def test_state_weights(self, dataset):
        sw = []
        for truths, Xf_gate, Xf_drive in dataset:
            # --> batch_size x num_tree/num_models x num_leaf/num_state
            st_weights = self.F.eval(Xf_gate)
            sw.append(st_weights.numpy())
        return np.vstack(sw)



#### Train Epochs
# schedule = [num_subepoch_gate, num_subepoch_drive]

def train_epochs(Model, train_dataset, test_dataset, schedule, num_epochs=10):
    # save overall errors as go
    tr_errs, te_errs = [],[]
    for i in range(num_epochs):
        # shuffle train dataset:
        train_dataset = train_dataset.shuffle(buffer_size=1024)
        dbatch = train_dataset.batch(128)

        # run scedule:
        for z in range(schedule[0]):
            Model.train_gate(dbatch)

        for z in range(schedule[1]):
            Model.train_drive(dbatch)
        
        # overall train error:
        tr_errs.append(Model.test_overall_err(dbatch))
        # overall test error:
        te_errs.append(Model.test_overall_err(test_dataset))

    # get reporting log-likelihood for the different models/state
    train_rll = Model.test_reporting_ll(dbatch)
    test_rll = Model.test_reporting_ll(test_dataset)

    return tr_errs, te_errs, train_rll, test_rll


# convert from block format to stacked format
# block format = num_blocks x Tsub x ...
# stacked format = T x ...
def convert_block_to_stacked(block_dat):
    block_sh = list(np.shape(block_dat))
    stack_sh = [block_sh[0] * block_sh[1]] + block_sh[2:]
    return np.reshape(block_dat, stack_sh)


# makes training dataset --> calls epoch run
# Assumes: data structures are in block format = num_block x Tsub x ...
def train_epochs_wrapper(Model, truths, Xf_gate, Xf_drive, schedule, num_epochs, train_inds, test_inds):
    tr = train_inds
    te = test_inds

    # convert from block to state
    truths_tr = convert_block_to_stacked(truths[tr])
    truths_te = convert_block_to_stacked(truths[te])

    Xf_gate_tr = convert_block_to_stacked(Xf_gate[tr])
    Xf_gate_te = convert_block_to_stacked(Xf_gate[te])

    Xf_drive_tr = convert_block_to_stacked(Xf_drive[tr])
    Xf_drive_te = convert_block_to_stacked(Xf_drive[te])

    # break up training / test sets:
    train_dataset = tf.data.Dataset.from_tensor_slices((truths_tr.astype(np.float32), Xf_gate_tr.astype(np.float32), \
        Xf_drive_tr.astype(np.float32)))
    test_dataset = tf.data.Dataset.from_tensor_slices((truths_te.astype(np.float32), Xf_gate_te.astype(np.float32), \
        Xf_drive_te.astype(np.float32))).batch(128)

    # reinit:
    Model.reinit()

    # run training:
    tr_errs, te_errs, train_rll, test_rll = train_epochs(Model, train_dataset, test_dataset, schedule, num_epochs=num_epochs)
    return tr_errs, te_errs, train_rll, test_rll



#### Data / Architecture Generation
# architecture generated using information from data structures
# Data structures:
# 1. truths = num_block x Tsub x output_cells x output_classes
# 2. Xf_gate = num_block x Tsub x xdim
# 3. Xf_drive = num_block x Tsub x models_per_state x xdim
# 4. comparison_filter = models_per_state
def arch_gen(truths, Xf_gate, Xf_drive, comparison_filter, tree_depth, tree_width, num_model, even_reg=0.1, l1_tree_mask=0.0, l1_mlr_mask=0.0, l2_state_mask=0.0, Fin=-1, Min=-1): 
    [num_block, Tsub, output_cells, output_classes] = np.shape(truths)
    [null, null, xdim_gate] = np.shape(Xf_gate)
    [null, null, models_per_state, xdim_drive] = np.shape(Xf_drive)

    # gating forest:
    if(Fin == -1):
        F = core_models.Forest(tree_depth, tree_width, num_model, xdim_gate, even_reg=even_reg, l1_mask=l1_tree_mask)
    else:
        F = Fin

    # comparative multilogreg
    if(Min == -1):
        MLRC = MultiLogRegComp(num_model, F.num_state, models_per_state, xdim_drive, output_cells, output_classes, l1_mask=l1_mlr_mask, l2_state_mask=l2_state_mask)
    else:
        MLRC = Min
    # Separator:
    S = Separator(F, MLRC, comparison_filter)
    
    return S



if(__name__ == '__main__'):


    tree_depth = 2
    tree_width = 2
    num_tree = 7
    xdim = 11
    output_cells = 2
    output_classes = 5

    Xf_gate = npr.rand(3, 32, xdim).astype(np.float32)

    models_per_state = 2

    # Xf_drive = batch_size x models_per_state x xdim
    Xf_drive = npr.rand(3, 32, models_per_state, xdim).astype(np.float32)

    comparison_filter = np.array([1,-1]).astype(np.float32)

    truths = np.zeros((3, 32, output_cells, output_classes))
    truths[:,:,0] = 1

    S = arch_gen(truths, Xf_gate, Xf_drive, comparison_filter, tree_depth, tree_width, num_tree, even_reg=0.1, l1_tree_mask=0.0)

    print(S)

    schedule = [1,1]
    num_epochs = 3
    train_inds = np.array([1,1,1]) > 0.5
    test_inds = train_inds
    tr, te, train_rll, test_rll = train_epochs_wrapper(S, truths, Xf_gate, Xf_drive, schedule, num_epochs, train_inds, test_inds)

    print(tr)
    print(te)
    print(train_rll)

    # TESTING: rebuilding forest:
    use_trees = np.zeros((num_tree))
    use_trees[np.array([2,4])] = 1
    use_trees = use_trees > 0.5

    F2 = core_models.Forest(S.F.tree_depth, S.F.tree_width, 2, S.F.xdim, S.F.even_reg, S.F.l1_mask, S.F.tree_struct, use_trees)


