'''

    Modular Forest
    --> Combine Soft Forest and Logistic Regression Objects


'''

import tensorflow as tf
import numpy as np
import numpy.random as npr
import os
import copy
import abc

dr = npr.default_rng(42)


#### Abstract Classes
# 1. gating models, 2. driver models


class Gate(abc.ABC):

    # get analysis vars
    # spec_mods specifies which models to use = 1D array
    def get_analysis_vars(self, spec_mods=[]):
        pass

    # get training variables
    # required to return 1. list of gate vars, 2. list of
    # drive vars (probs empty)
    def get_train_vars(self):
        pass

    # eval(evaluation): taken in Xf --> output probabilities
    # Xf = batch_size x xdims
    # returns batch_size x num_model x num_state 
    def eval(self,Xf):
        pass

    # reg(regularization): take in Xf --> output scalar regularization loss
    def reg(self,Xf):
        pass


class Driver(abc.ABC):

    # get analysis vars ~ used for post-fitting analysis
    # spec_mods specifies which models to use = 1D array
    def get_analysis_vars(self, spec_mods=[]):
        pass

    # get training variables
    # required to return 1. list of gate vars (probs empty here), 2. list of
    # drive vars
    def get_train_vars(self):
        pass

    # eval(evaluation): taken in Xf --> output probabilities
    # Xf = batch_size x xdims
    # returns batch_size x num_model x num_state 
    def eval(self,Xf):
        pass

    # reg(regularization): take in Xf --> output scalar regularization loss
    def reg(self,Xf):
        pass


#### Helper Functions

# helper variable constructor
def var_construct(v_shape):
    v = dr.random(v_shape) - 0.5
    return tf.Variable(v.astype(np.float32))


###### Gates


#### Null Class
# null gating class
# returns 1s for eval
# use this for basic logistic regression
class Null(Gate):

    # only 1 state
    def __init__(self, num_model):
        # constant eval structure:
        self.num_model = num_model
        self.num_state = 1
        npk = np.ones((1,self.num_model,1)).astype(np.float32)
        self.k = tf.constant(npk)
        self.train_vars = []

    # get the analysis vars
    def get_analysis_vars(self, spec_mods=[]):
        return -1

    def get_train_vars(self):
        return [], []

    # tile by batch_size
    # returns batch_size x num_model x num_state (1)
    def eval(self,Xf):
        Xfs = Xf[:,0]
        Xfs = tf.reshape(Xfs, [-1,1,1]) * 0.0
        v = Xfs + self.k
        return tf.stop_gradient(v)

    def reg(self,Xf):
        return 0.0



#### Forest Class

class Forest(Gate): 

    # input shapes
    # Xf = batch_size x xdim
    # NOTE: tree outputs leaf probabilities ~ does not need to know output shapes

    # primary constructor
    def __init__(self, tree_depth, tree_width, num_tree, xdim, even_reg=0.0, l1_mask=0.0): 
      
        # save fields:
        self.tree_depth = tree_depth
        self.tree_width = tree_width
        self.num_tree = num_tree
        self.xdim = xdim
        self.even_reg = even_reg
        self.num_model = num_tree
        self.num_state = int(tree_width**tree_depth)

        # l1 regularizer:
        if(isinstance(l1_mask, float) or isinstance(l1_mask,int)):
            self.l1_mask = tf.constant(l1_mask)
        else:
            self.l1_mask = tf.constant(np.reshape(l1_mask, (1,1,1,-1)).astype(np.float32))

        # train vars:
        self.train_vars = []

        # build the tree structure:
        self.build_forest()



    # get the analysis vars
    # spec_mods specifies which models to use
    def get_analysis_vars(self, spec_mods=[]):
        # --> N x num_tree x tree_width x xdim
        st = tf.stack(self.train_vars)
        st = st.numpy()
        if(len(spec_mods) > 0):
            return st[:,spec_mods]
        return st

    # 1. gate vars, 2. drive vars (no drive in this class)
    def get_train_vars(self):
        return self.train_vars, []


    #### Forest Construction
    
    ## Tree Structure ~ branches

    # build branch
    # Xf = batch_size x xdim
    # -->
    # hf = num_tree x tree_width x xdim
    def build_branch(self):
        hf = var_construct((self.num_tree, self.tree_width-1, self.xdim))
        self.train_vars.append(hf)
        return [hf]


    def build_forest_helper(self, tree_struct, cdepth):
        # build another branch:
        v = self.build_branch()
        tree_struct.append(v)

        # if reached correct cdepth --> return
        if(cdepth == 0):
            return 

        # descend into branches:
        for i in range(self.tree_width):
            tree_struct.append([])
            self.build_forest_helper(tree_struct[-1], cdepth-1)


    def build_forest(self):
        self.tree_struct = []
        self.build_forest_helper(self.tree_struct, self.tree_depth-1)



    #### Evaluation

    # Xf = batch_size x xdim


    ## forest evaluation

    # return shape: batch_size x num_tree x tree_width
    def eval_branch(self, hf, Xf): 
        # Xf eval:
        # Xf = batch_size x xdim
        # hf = num_tree x tree_width-1 x xdim
        Xf2 = tf.reshape(Xf, [-1,1,1,self.xdim]) # --> batch_size x 1 x 1 x xdim
        hf2 = tf.reshape(hf, [1,self.num_tree,self.tree_width-1,self.xdim]) 
        v = tf.reduce_sum(Xf2 * hf2, axis=3) # --> batch_size x num_tree x tree_width-1

        # add offset --> batch_size x num_tree x tree_width
        v = tf.concat([v, 0*v[:,:,:1]], axis=2)

        # softmax across treewidth
        return tf.nn.softmax(v, axis=2)


    # assumes: pr_up = batch_size x num_tree x 1
    def eval_forest_helper(self, tree_struct, Xf, pr_up):

        # get params for current branch
        hf = tree_struct[0][0]

        # evaluate branch:
        # --> batch_size x num_tree x tree_width
        ev = self.eval_branch(hf, Xf)

        # scale by pr_up
        pr = pr_up * ev

        # if last branch --> return pr
        if(len(tree_struct) == 1):
            return [pr]

        # descend into sub-branches:
        pr_ret = []
        for i in range(1,len(tree_struct)):
            pr_ret = pr_ret + self.eval_forest_helper(tree_struct[i], Xf, pr[:,:,i-1:i])
        return pr_ret


    # returns: batch_size x num_tree x num_leaf
    def eval(self, Xf):
        pr_up = tf.constant(np.ones((1,1,1)).astype(np.float32))
        leaf_prs = self.eval_forest_helper(self.tree_struct, Xf, pr_up)
        return tf.concat(leaf_prs, axis=-1) 



    #### Regularizers

    def loss_l1(self):
        # --> N x num_tree x tree_width x xdim
        st = tf.stack(self.train_vars)
        # scale by mask
        mst = self.l1_mask * tf.math.abs(st)
        # reduce sum across everything but num_tree
        mst = tf.reduce_sum(mst, axis=[0,2,3])
        # reduce mean across trees:
        return tf.reduce_mean(mst)
    

    # even regularization:
    # maximize entropy --> minimize negative entropy
    def loss_even_reg(self, Xf):
        # batch_tree x num_tree x num_leaf
        v = self.eval(Xf)
        # average over batch --> num_tree x num_leaf
        # NOTE: will still be probabilities across num_leaf
        vav = tf.reduce_mean(v, axis=0)
        # negative entropy
        ent = tf.reduce_sum(vav * tf.math.log(vav), axis=1) # --> num_tree
        # get batch_size scaling factor:
        bsc_factor = tf.reduce_sum(1.0 + 0.0*v[:,0,0])
        # average across tree and rescale by batch_size
        return tf.reduce_mean(ent) * self.even_reg * bsc_factor
    

    # full regularizer:
    def reg(self, Xf):
        return self.loss_l1() + self.loss_even_reg(Xf)



###### Drivers


#### Multi Logistic Regression Class

class MultiLogReg(Driver):

    # num models = the number of different logistic regression models
    def __init__(self, num_models, num_state, xdim, output_cells, output_classes, l1_mask=.05):
        self.num_models = num_models
        self.num_state = num_state
        self.xdim = xdim
        self.output_cells = output_cells
        self.output_classes = output_classes
        
        # l1 regularizer:
        if(isinstance(l1_mask, float) or isinstance(l1_mask,int)):
            self.l1_mask = tf.constant(l1_mask)
        else: 
            self.l1_mask = tf.constant(np.reshape(l1_mask, (1,1,1,1,-1)).astype(np.float32))

        self.build_structs()
        self.train_vars = [self.X_struct]


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
        # num_models, num_state x out_cells x out_classes x xdim
        self.X_struct = var_construct((self.num_models, self.num_state, self.output_cells, self.output_classes, self.xdim))

    
    #### Evaluation

    # evaluation
    # returns: batch_size x num_models x num_state x out_cells x out_classes
    # Xf = batch_size x xdim
    # NOTE: NOT softmax'd --> for boosting purposes
    def eval(self, Xf):
        # reshape Xf --> batch_size x 1 x 1 x 1 x 1 x xdim
        Xf = tf.reshape(Xf, [-1, 1, 1, 1, 1, self.xdim])
        # run thru filter:
        # --> batch_size num_models x num_state x out_cells x out_classes
        v = tf.reduce_sum(Xf * tf.expand_dims(self.X_struct, 0), axis=5)
        # softmax across output classes
        #return tf.nn.softmax(v, axis=-1)
        return v


    #### Regularizers

    # X structure
    # num_models x num_state x out_cells x out_classes x xdim
    # average across num models and num_states
    def loss_l1(self):
        mst = self.l1_mask * tf.math.abs(self.X_struct)
        # sum across everything but num_state
        mst = tf.reduce_sum(mst, axis=[2,3,4])
        # average across num models
        return tf.reduce_mean(mst)


    def reg(self, Xf):
        return self.loss_l1()


#### Low-Rank Multi Logistic Regression Class
# splits multinomial X_struct into two smaller structs with rank bound by lr
# parameter

class MultiLogRegLR(Driver):

    # num models = the number of different logistic regression models
    # NOTE: l1_mask should be same shape as in basic MLR
    def __init__(self, num_models, num_state, xdim, output_cells,
            output_classes, l1_mask=.05, lr=1): 
        self.num_models = num_models
        self.num_state = num_state
        self.xdim = xdim
        self.output_cells = output_cells
        self.output_classes = output_classes
        self.lr = lr
        
        # l1 regularizer: 
        if(isinstance(l1_mask, float) or isinstance(l1_mask,int)):
            self.l1_mask = tf.constant(l1_mask)
        else: 
            self.l1_mask = tf.constant(np.reshape(l1_mask, (1,1,1,1,-1)).astype(np.float32))

        self.build_structs()
        self.train_vars = [self.X_struct1, self.X_struct2]


    # get the analysis vars
    def get_analysis_vars(self, spec_mods=[]):
        # get full xstruct:
        xstruct = self.mult_xstructs()
        st = xstruct.numpy()
        if(len(spec_mods) > 0):
            return st[spec_mods]
        return st

    # 1. gate vars (X_struct1), 2. drive vars (X_struct2)
    def get_train_vars(self):
        return [self.X_struct1], [self.X_struct2]


    #### Build

    # build structures
    def build_structs(self):
        # X_struct1 = num_models x num_state x xdim x lr
        self.X_struct1 = var_construct((self.num_models, self.num_state,
            self.xdim, self.lr))
        # X_struct2 = num_models x num_state x out_cells x out_classes x lr
        self.X_struct2 = var_construct((self.num_models, self.num_state,
            self.output_cells, self.output_classes, self.lr))


    #### Mult X_structs
    # make them match shape of large X_struct in original MLR
    # = num_models x num_state x out_cells x out_classes x xdim
    def mult_xstructs(self):
        # reduce across lr
        x1 = tf.reshape(self.X_struct1, [self.num_models, self.num_state, 1, 1,
            self.xdim, self.lr])
        x2 = tf.reshape(self.X_struct2, [self.num_models, self.num_state,
            self.output_cells, self.output_classes, 1, self.lr])
        x = tf.reduce_sum(x1 * x2, axis=-1)
        return x

    
    #### Evaluation
    # low-rank version: X_struct1 = num_models x num_state x xdim x lr
    # X_struct2 = num_models x num_state x out_cells x out_classes x lr
    # where lr << xdim and lr << out_cells * out_classes
    # mult Xf thru both of these

    # evaluation
    # returns: batch_size x num_models x num_state x out_cells x out_classes
    # Xf = batch_size x xdim
    # NOTE: NOT softmax'd --> for boosting purposes
    def eval(self, Xf):
        # get full xstruct
        xstruct = self.mult_xstructs()
        # reshape Xf --> batch_size x 1 x 1 x 1 x 1 x xdim
        Xf = tf.reshape(Xf, [-1, 1, 1, 1, 1, self.xdim])
        # run thru filter:
        # --> batch_size num_models x num_state x out_cells x out_classes
        v = tf.reduce_sum(Xf * tf.expand_dims(xstruct, 0), axis=5)
        # softmax across output classes
        #return tf.nn.softmax(v, axis=-1)
        return v


    #### Regularizers

    # X structure
    # num_models x num_state x out_cells x out_classes x xdim
    # average across num models and num_states
    def loss_l1(self):
        # get full xstruct:
        xstruct = self.mult_xstructs()

        # l1 norm in typical way
        mst = self.l1_mask * tf.math.abs(xstruct)
        # sum across everything but num_state
        mst = tf.reduce_sum(mst, axis=[2,3,4])
        # average across num models
        return tf.reduce_mean(mst)


    def reg(self, Xf):
        return self.loss_l1()


#### Booster
# continually boost with gated logistic regression models

class Booster:

    # takes in [gating, predicting] model pairs = model_pairs
    # NOTE: don't necessarily need gating object
    def __init__(self, model_pairs):
        self.model_pairs = model_pairs
        self.num_model = model_pairs[0][0].num_model
        self.output_cells = model_pairs[0][1].output_cells
        self.output_classes = model_pairs[0][1].output_classes

        # get all train_vars = gate AND drive vars
        self.train_vars = []
        for i in range(len(self.model_pairs)):
            for j in range(len(self.model_pairs[i])):
                g, d = self.model_pairs[i][j].get_train_vars()
                self.train_vars = self.train_vars + g + d

        # get the drive/mlr train vars only:
        self.train_vars_mlr = []
        for i in range(len(self.model_pairs)):
            for j in range(len(self.model_pairs[i])):
                g, d = self.model_pairs[i][j].get_train_vars()
                self.train_vars_mlr = self.train_vars_mlr + d

        self.train_op = tf.keras.optimizers.Adam(.01)


    # eval 
    # each pair:
    # gating --> T x num_model(tree) x num_state
    # predictor --> T x num_model(tree) x num_state x out_cells x out_classes
    # expand/multiply gates; expand/add predictors
    # returns:
    # gate eval = T x num_model x num_state1 x ....
    # pred eval = T x num_model x num_state1 x ... x out_cells x out_classes
    def eval(self, Xf_pairs):
        # repeat num_model for each gating model
        # ex: 2 gates --> T x num_model x num_state1 x ...
        gate_shape = [-1, self.num_model] + len(self.model_pairs) * [1]
        # repeat num_model for each predictor
        # = T x num_model x num_state1 x ... x output_cells x output_classes
        pred_shape = [-1, self.num_model] + len(self.model_pairs) * [1] + [self.output_cells, self.output_classes]
        gate_evz = 1.0
        pred_evz = 0.0
        for i in range(len(self.model_pairs)):
            # gate evaluation
            c_gate_shape = copy.deepcopy(gate_shape)
            c_gate_shape[i+2] = self.model_pairs[i][0].num_state
            gate_ev = self.model_pairs[i][0].eval(Xf_pairs[i][0])
            gate_ev = tf.reshape(gate_ev, c_gate_shape)
            gate_evz = gate_evz * gate_ev
            # predictor evaluation:
            c_pred_shape = copy.deepcopy(pred_shape)
            c_pred_shape[i+2] = self.model_pairs[i][1].num_state
            pred_ev = self.model_pairs[i][1].eval(Xf_pairs[i][1])
            pred_ev = tf.reshape(pred_ev, c_pred_shape)
            pred_evz = pred_evz + pred_ev
        return gate_evz, pred_evz


    # Loss
    # truths = T x out_cells x out_classes
    # eval returns:
    # gate eval = T x num_model x num_state1 x ....
    # pred eval = T x num_model x num_state1 x ... x out_cells x out_classes
    def loss(self, truths, Xf_pairs, separate_mode=False):
        # reshape truths:
        tr_shape = [-1, 1] + len(self.model_pairs) * [1] + [self.output_cells, self.output_classes]
        truths = tf.reshape(truths, tr_shape)
        # eval:
        gate_ev, pred_ev = self.eval(Xf_pairs)
        # softmax pred across last dim (out_classess)
        pred_ev = tf.nn.softmax(pred_ev, axis=-1)
        # multinomial log-likelihood
        # --> T x num_model x ... x num_state x out_cells x out_classes
        mll = tf.reduce_sum(truths * tf.math.log(pred_ev), axis=-1)
        # reduce across cells:
        mll = tf.reduce_sum(mll, axis=-1)
        # scale by gate:
        scale_ll = gate_ev * mll
        # reduce across states:
        scale_ll = tf.reduce_sum(scale_ll, axis=[z for z in range(2,2+len(self.model_pairs))])
        # reduce across T:
        scale_ll = tf.reduce_sum(scale_ll, axis=0)
        if(separate_mode):
            return -1.0 * scale_ll
        # average across models
        return -1.0 * tf.reduce_mean(scale_ll)



    # forest loss
    # average predictions across models --> compute error
    def forest_loss(self, truths, Xf_pairs, f_mask):
        # eval:
        gate_ev, pred_ev = self.eval(Xf_pairs)
        # softmax pred across last dim (out_classess)
        pred_ev = tf.nn.softmax(pred_ev, axis=-1)
        # --> T x num_model x ... x num_state x out_cells x out_classes

        # use gate to average prediction
        gate_ev = tf.expand_dims(gate_ev,-1)
        gate_ev = tf.expand_dims(gate_ev,-1)
        mu_pred = tf.reduce_sum(gate_ev * pred_ev, axis=[z for z in range(2,2+len(self.model_pairs))])
        # --> T x num_model x out_cell x out_dim
        # use mask to average mu_pred across models (mask selects models)
        f_mask = tf.reshape(f_mask, [1,-1,1,1])
        ave_mu_pred = tf.reduce_sum(f_mask * mu_pred, axis=1)
        # --> T x out_cell x out_dim

        # loglikelihood ~ avemupred and truths
        mll = tf.reduce_sum(truths * tf.math.log(ave_mu_pred))
        return -1.0 * mll


    # call regularization functions of components
    def reg(self, Xf_pairs):
        rs = []
        for i in range(len(self.model_pairs)):
            for j in range(2):
                rs.append(self.model_pairs[i][j].reg(Xf_pairs[i][j]))
        return tf.add_n(rs)


    # training 
    # expects to be fed data in tuple of tuple structure
    @tf.function
    def train(self, dataset):
        for truths, Xf_pairs in dataset:
            with tf.GradientTape() as tape:
                L = self.loss(truths, Xf_pairs)
                L1 = self.reg(Xf_pairs)
                g = tape.gradient(L+L1, self.train_vars)
                self.train_op.apply_gradients(zip(g, self.train_vars))


    # training mlr ~ only the drive models
    @tf.function
    def train_mlr(self, dataset):
        for truths, Xf_pairs in dataset:
            with tf.GradientTape() as tape:
                L = self.loss(truths, Xf_pairs)
                L1 = self.reg(Xf_pairs)
                g = tape.gradient(L+L1, self.train_vars_mlr)
                self.train_op.apply_gradients(zip(g, self.train_vars_mlr))



#### Training

# convert to tensorflow dataset:
# olab = T x num_cells x num_classes
# Xf = tuple of tuples
# ... Xf[i][j] should be T x xdim 
def make_dataset(olab, Xf):
    dataset = tf.data.Dataset.from_tensor_slices((olab, Xf))
    return dataset


# train for a number of epochs:
# modes:
# mlr = only train mlr / drive models (convex)
# [else] = train everything
def train_epochs(Model, train_dataset, train_Xf, train_truths, test_Xf, test_truths, num_epochs=10, mode=''):
    tr_errs, te_errs = [],[]
    for i in range(num_epochs):
        # shuffle train dataset:
        train_dataset = train_dataset.shuffle(np.shape(train_truths)[0])
        dbatch = train_dataset.batch(128)
        if(mode == 'mlr'):
            Model.train_mlr(dbatch)
        else:
            Model.train(dbatch) 
        
        # train error:
        tr_errs.append(Model.loss(train_truths, train_Xf, separate_mode=True).numpy())
        te_errs.append(Model.loss(test_truths, test_Xf, separate_mode=True).numpy())
    return tr_errs, te_errs


# training wrapper for typical data format
# data format = [truths, [Model pair feed...
def train_epochs_wrapper(Model, dat_train, dat_test, num_epochs=10, mode=''):

    # convert to dataset:
    train_dataset = tf.data.Dataset.from_tensor_slices(dat_train)

    tr_errs, te_errs = train_epochs(Model, train_dataset, dat_train[1], dat_train[0], dat_test[1], dat_test[0], num_epochs, mode)
    return tr_errs, te_errs


#### Architecture Generation
# generate architectures and reshape data structures

# restructure list of Xfs into single array
# Assumes: Tblock x Tsub x ...
def restructure_Xf(Xf_list):
    Tblock = np.shape(Xf_list[0])[0]
    Tsub = np.shape(Xf_list[0])[1]
    Xf2 = []
    for Xf in Xf_list:
        Xf2.append(np.reshape(Xf, (Tblock * Tsub, -1)))
    # concatenate:
    Xf2 = np.concatenate(Xf2, axis=1)
    return Xf2.astype(np.float32)



# generate gate-drive pair
# input = Xf_list_sub = [[Xf gate arrays...], [Xf drive arrays...]]
# model masks matches shape
# lr = low-rank... if > 0 --> run low-rank mode
def gen_gate_drive(output_cells, output_classes, xdims_sub, model_mask_sub,
        tree_depth, tree_width, num_model, lr=-1, even_reg=0.1): 
    ## gate:
    if(tree_depth <= 0): # logistic regression
        G = Null(num_model)
        num_state = 1
    else:
        G = Forest(tree_depth, tree_width, num_model, xdims_sub[0], even_reg=even_reg, l1_mask=np.hstack(model_mask_sub[0]))
        num_state = int(tree_width**tree_depth)

    ## drive:
    if(lr <= 0):
        MLR = MultiLogReg(num_model, num_state, xdims_sub[1], output_cells, output_classes, l1_mask=np.hstack(model_mask_sub[1]))
    else:
        MLR = MultiLogRegLR(num_model, num_state, xdims_sub[1], output_cells,
                output_classes, l1_mask=np.hstack(model_mask_sub[1]), lr=lr)

    ## returns: data structures and fitting objects
    return [G, MLR]


# block conversion on train/test inds
def block_convert(t_inds, Tsub):
    if(len(np.shape(t_inds)) > 1):
        return np.reshape(t_inds, (-1))
    t_inds = np.reshape(t_inds, (-1,1))
    t_inds = np.tile(t_inds, (1,Tsub))
    return np.reshape(t_inds, (-1))



# Inputs:
# 1. truths = Tblock x Tsub x output_cells x output_classes
# 2. Xf_list = [[[Xf gate arrays], [Xf drive arrays], ...[
# 3. model_masks = matches shape of Xf_list
# all Xfs should have shape Tblock x Tsub x .... other dims (will be flattened)
# 3. tree depths = list matching Xf list
# 4. tree widths = list matching Xf list
# NOTE: depth = 0 --> logistic regression
# TODO: missing training stuff
def arch_gen(output_cells, output_classes, xdims, model_masks, tree_depths,
        tree_widths, num_model, lrs=[], even_reg=0.1): 

    if(len(lrs) == 0):
        lrs = [-1 for k in range(len(tree_depths))]

    # build architecture
    arch = []
    for i in range(len(tree_depths)):
        archc = gen_gate_drive(output_cells, output_classes, xdims[i],
                model_masks[i], tree_depths[i], tree_widths[i], num_model,
                lrs[i], even_reg=even_reg)

        arch.append(archc)

    # make boost structure: 
    B = Booster(arch)
    return B



# data generation helper
def gen_dat_sub(truths, Xf_list_sub, train_inds, test_inds):
    ## gate:
    Xf_gate = restructure_Xf(Xf_list_sub[0])

    ## drive:
    Xf_drive = restructure_Xf(Xf_list_sub[1])

    ## returns: data structures and fitting objects
    return (Xf_gate[train_inds], Xf_drive[train_inds]), (Xf_gate[test_inds], Xf_drive[test_inds])



# Data shaping
# Inputs:
# 1. truths = Tblock x Tsub x output_cells x output_classes
# 2. Xf_list = [[[Xf gate arrays], [Xf drive arrays], ...[
# 3. model_masks = matches shape of Xf_list
# all Xfs should have shape Tblock x Tsub x .... other dims (will be flattened)
def dat_gen(truths, Xf_list, train_inds, test_inds):
    Tblock = np.shape(truths)[0]
    Tsub = np.shape(truths)[1]
    # flatten truths and convert to floats:
    truths = np.reshape(truths, (Tblock*Tsub,np.shape(truths)[2],np.shape(truths)[3])).astype(np.float32)
    # block convert train and test inds:
    train_inds = block_convert(train_inds, Tsub)
    test_inds = block_convert(test_inds, Tsub)

    # build data structures and arch simultaneously
    dat_train = []
    dat_test = []
    arch = []
    for i in range(len(Xf_list)):
        Xfc_train, Xfc_test = gen_dat_sub(truths, Xf_list[i], train_inds, test_inds)
        dat_train.append(Xfc_train)
        dat_test.append(Xfc_test)

    dat_train = [truths[train_inds], tuple(dat_train)]
    dat_test = [truths[test_inds], tuple(dat_test)]

    return tuple(dat_train), tuple(dat_test)



# helper function
# get xdims from dat_train (training data) ~ includes truths
def get_xdims(dat_train):
    # generate xdims data structure:
    xdims = []
    for i in range(len(dat_train[1])):
        xdims_sub = []
        for j in range(len(dat_train[1][i])):
            xdims_sub.append(np.shape(dat_train[1][i][j])[-1])
        xdims.append(xdims_sub)
    return xdims






