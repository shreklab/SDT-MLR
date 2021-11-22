'''

    Core Models

    Divided into gating models and driver models

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
# TODO: gate and driver should probably contain set_variable functions

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

def np_construct(v_shape):
    v = dr.random(v_shape) - 0.5
    return v.astype(np.float32)

# helper variable constructor
def var_construct(v_shape):
    return tf.Variable(np_construct(v_shape))


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
    # NOTE: tree outputs leaf probabilities ~ does not need to know final prediction shapes (ex: num_classes)
    # optionally constructs from tree_struct_template (another tree)... use_trees specifies which sub_trees to construct from
    # TODO: optionally construct from numpy array (also uses use_trees)
    # constructor preference: tree_struct_template > tree_struct_array > new init

    # primary constructor
    def __init__(self, tree_depth, tree_width, num_tree, xdim, even_reg=0.0, l1_mask=0.0, tree_struct_template=[], use_trees=[], tree_struct_ar=[]): 
      
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
        # order of precedence: tree_struct_template > tree_struct_array > new_init
        if(len(tree_struct_template) > 0):
            self.build_forest_template(tree_struct_template, use_trees)
        elif(len(tree_struct_ar) > 0):
            self.build_forest_ar(tree_struct_ar, use_trees)
        else:
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

    ## build

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


    ## construct from template

    def build_branch_template(self, branch, use_trees):
        hf = tf.Variable(copy.deepcopy(branch[0].numpy()[use_trees]))
        self.train_vars.append(hf)
        return [hf]


    def build_forest_helper_template(self, tree_struct, cdepth, tree_struct_template, use_trees): 
        # build another branch:
        v = self.build_branch_template(tree_struct_template[0], use_trees)
        tree_struct.append(v)

        # if reached correct cdepth --> return
        if(cdepth == 0):
            return 

        # descend into branches:
        for i in range(self.tree_width):
            tree_struct.append([])
            self.build_forest_helper_template(tree_struct[-1], cdepth-1, tree_struct_template[i+1], use_trees)


    # tree_struct_template = another tree_struct
    # use_trees = boolean array indicating which trees to use
    def build_forest_template(self, tree_struct_template, use_trees): 
        self.tree_struct = []
        self.build_forest_helper_template(self.tree_struct, self.tree_depth-1, tree_struct_template, use_trees)


    ## construct from numpy array
    # assumed gate shape = num_branches x num_tree x tree_width x xdim

    def build_branch_ar(self, branch_ind, template_ar, use_trees): 
        hf = tf.Variable(template_ar[branch_ind,use_trees])
        self.train_vars.append(hf)
        return [hf]


    def build_forest_helper_ar(self, tree_struct, cdepth, tree_struct_ar, use_trees, branch_ind): 
        # build another branch:
        v = self.build_branch_ar(branch_ind, tree_struct_ar, use_trees)
        tree_struct.append(v)
        # increment branch_ind:
        branch_ind += 1 

        # if reached correct cdepth --> return
        if(cdepth == 0):
            return branch_ind

        # descend into branches:
        for i in range(self.tree_width):
            tree_struct.append([])
            branch_ind = self.build_forest_helper_ar(tree_struct[-1], cdepth-1, tree_struct_ar, use_trees, branch_ind)

        # return updated branch index:
        return branch_ind


    def build_forest_ar(self, tree_struct_ar, use_trees):
        self.tree_struct = []
        self.build_forest_helper_ar(self.tree_struct, self.tree_depth-1, tree_struct_ar, use_trees, 0)


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
        # sum across everything but num_state and num_model
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



