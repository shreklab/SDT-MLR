'''

    Visualize Separator Results

    Data?
    Performance:
    -> train_ll = train set log-likelihoods = num_boot x num_model x num_state x models_per_state
    -> test_ll = test set log-likelihoods = num_boot x num_model x num_state x models_per_state
    -> weight_preds = weights output by gate model (in order) = num_block x num_model x num_state
        ... NOTE: weight preds is for all data... train and test
        ... TODO: what shape???
    Gate Variables:
    -> gate_vars = learned variables for gate model = num_boot x num_branch x num_model x tree_width x xdim
    Drive Variables:
    -> drive_vars = learned variables for drive model = num_boot x num_model x num_state x models_per_state 
                                                        x output_cell x output_classes x xdim
    Raw:
    -> Xf_gate = input data to gating model = num_block x T_per_block x xdim
    -> Xf_drive = input data to drive model(s) = num_block x T_per_block x models_per_state x xdim

    NOTE: Gate xdim is NOT necessarily the same as drive xdim

'''

import numpy as np
import pylab as plt
import os
import abc
import copy
from matplotlib.colors import ColorConverter


#### Xdim conversion
# converting from og space to xdim:
# ... cells x sub_dim, num_worms (flatten) --> xdim

# general xdim converter
# assumed input shape = ... x xdim (xdim is last dim)
# convert to list of numpy arrays
# where each element = dims for particular data type
# ex: guide_shapes = [[4, 8], [18]]
# --> converts to [... x 4 x 8, ... x 18]
def convert_xdim_to_og(dat, guide_shapes):
    dat_l = []
    dat_sh = np.shape(dat)
    dat_sh_pre = list(dat_sh[:-1])

    st = 0
    for i in range(len(guide_shapes)):
        cur_len = int(np.prod(guide_shapes[i]))
        ed = st + cur_len
        dat_sub = dat[...,st:ed]
        dat_l.append(np.reshape(dat_sub, dat_sh_pre + list(guide_shapes[i])))
        st = ed
    return dat_l


#### Data Wrappers
# each datatype (ex: log-likelihood performance data) has a separate object
# TODO: required methods?
# viz: matplotlib inferface == image and keybindings to look around at data


# abstract wrapper class
# requires the above set of methods
class DataWrapper(abc.ABC):

    def viz(self, Operator, save_str=''): 
        pass


# performance data wrapper
# log-likelihoods = num_boot x num_model x num_state x models_per_state
# comparison filter = applied to models_per_state to get overall likelihood difference
class PerfWrapper(DataWrapper):

    # lls = log-likelihoods (shape above)
    def __init__(self, lls, comp_filter=np.array([1,-1]), label_str='', bins=np.arange(-70,70,10)):
        self.label_str = label_str
        self.comp_filter = comp_filter
        self.bins = bins
        # apply comparison filter to lls --> comparison data:
        self.cdat = np.sum(lls * np.reshape(self.comp_filter, (1,1,1,-1)), axis=-1)


    # makes plots but does not call show --> allows for cross-type comparison
    # idea 1: takes in reference to operator
    # --> gets state / model id from operator
    # --> histogram for the different models_per_state
    def viz(self, Operator, save_str=''):
        dat_sub = self.cdat[:,Operator.model_id,Operator.state_id]
        title_str = 'perf viz: model' + str(Operator.model_id) + ', state: ' + str(Operator.state_id)
        plt.figure()
        plt.hist(dat_sub, bins=self.bins)
        plt.title(title_str)

        if(len(save_str) > 0):
            plt.savefig(save_str + title_str + '.svg')




# gating filters
# gate_vars = learned variables for gate model = 
# num_boot x gate_filters_per_overall_model x num_tree x tree_width x in_cells x T (converted xdim)
# -->
# pull from operator: model_id=tree_id
# Static = Assumes: model is static across boots
class GateWrapperStatic(DataWrapper):

    # xax = x-axis to plot = num_cell x num_t
    def __init__(self, gate_vars, in_cells_viz, cell_colours, xaxs, tree_depth=2, ylim=[]):
        # static assumption --> take 0th boot -->
        # gate_filters_per_overall_model x num_tree(num_model) x tree_width x in_cells x T
        self.gate_vars = gate_vars[0]
        self.cell_colours = cell_colours
        self.in_cells_viz = in_cells_viz
        self.xaxs = xaxs
        self.tree_depth = tree_depth
        self.ylim = ylim


    # separate figure for each level of the tree / tree width
    # subplot for each input cell

    # plot given level
    # title tells you place within tree
    def plot_level(self, ind, level_str, model_id, save_str): 
        # iter thru tree_width
        for i in range(np.shape(self.gate_vars)[2]):
            # make figure for current tree:
            fig, axs = plt.subplots(len(self.in_cells_viz),1)
            # iter thru in_cells/axes
            for j in range(len(axs)):
                axs[j].plot(self.xaxs[j], self.gate_vars[ind, model_id, i, self.in_cells_viz[j], :], c=self.cell_colours[j])
                if(self.ylim):
                    axs[j].set_ylim(self.ylim)
            # add indicator as title:
            title_str = level_str + str(i)
            axs[0].set_title(title_str)

            if(len(save_str) > 0):
                plt.savefig(save_str + title_str + '.svg') 


    # recursively descend into tree --> plot each level
    # NOTE: soft trees are depth-first in this project
    # NOTE: also assumes that actual tree_width = reported tree_width + 1

    def plot_level_rec_helper(self, ind, level_str, model_id, cdepth, save_str):
        # check if we've over-run depth:
        if(cdepth <= 0):
            return ind-1 # subtraction undoes the increment at above level

        # plot current level:
        self.plot_level(ind, level_str, model_id, save_str)

        # descend into sub-levels, depth-first
        for i in range(np.shape(self.gate_vars)[2] + 1):
            ind = self.plot_level_rec_helper(ind+1, level_str + str(i), model_id, cdepth-1, save_str)
        return ind


    def viz(self, Operator, save_str=''):
        self.plot_level_rec_helper(0, '', Operator.model_id, self.tree_depth, save_str)




# drive filters
# drive_vars = learned variables for drive model = num_boot x num_model x num_state x models_per_state x output_cell
#           x output_cells x output_classes x in_cells x T
# pull from operator: model_id, state_id, output_cell, output_classes
# Assumes: filterbank has already been applied
class DriveWrapper(DataWrapper):

    # xax = x-axis to plot = num_cell x num_t
    def __init__(self, drive_vars, model_per_state_id, in_cells_viz, cell_colours, xaxs, percs=[50, 90], ylim=[]):
        # only look at one submodel (specified by model_per_state_id)
        # --> drive_vars = num_boot x num_model x num_state x output_cells x output_classes x in_cells x T
        self.drive_vars = drive_vars[:,:,:,model_per_state_id]
        self.cell_colours = cell_colours
        self.percs = percs
        self.in_cells_viz = in_cells_viz
        self.xaxs = xaxs
        self.ylim = ylim

    
    # generate graded colours from base colour:
    def generate_graded_colours(self, base_colour):
        cc = ColorConverter()
        rc = list(cc.to_rgba(base_colour))
        rc[-1] = 1.0
        # get alphas from strongest to weakest:
        alphaz = 1.0 - np.linspace(0,1,len(self.percs)+2)[:len(self.percs)+1] 
        colour_list = []
        for i in range(1+len(self.percs)):
            rcc = copy.deepcopy(rc)
            rcc[-1] = alphaz[i]
            colour_list.append(rcc)
        return colour_list

 
    # missing titles
    # plot percentiles for single [output_cell, output_class] combo
    def plot_percentiles(self, output_cell, output_class, model_id, state_id): 
        # get data subset:
        # --> num_boot x in_cells x T
        dat = self.drive_vars[:, model_id, state_id, output_cell, output_class, :, :]

        # subplot for each in_cell:
        fig, axs = plt.subplots(len(self.in_cells_viz),1)

        title_str = 'outcell' + str(output_cell) + '_outclass' + str(output_class) \
                + '_modelid' + str(model_id) + '_stateid' + str(state_id)
        axs[0].set_title(title_str)

        for i, ax in enumerate(axs):
            # get colours:
            c_list = self.generate_graded_colours(self.cell_colours[i])

            # get data subset:
            dstub = dat[:,self.in_cells_viz[i]]
            # plot median:
            ax.plot(self.xaxs[i], np.median(dstub, axis=0), color=tuple(c_list[0]))
            # plot percentages:
            for j, perc in enumerate(self.percs):
                hi = np.percentile(dstub, 50+(.5*perc), axis=0)
                lo = np.percentile(dstub, 50-(.5*perc), axis=0)
                ax.plot(self.xaxs[i], hi, color=tuple(c_list[j+1]), linewidth=2)
                ax.plot(self.xaxs[i], lo, color=tuple(c_list[j+1]), linewidth=2)
            if(self.ylim):
                axs[i].set_ylim(self.ylim)

        return title_str


    # visualize drive filters for set of [output_cell, output_class] ids
    def viz(self, Operator, save_str=''):
        for i in range(len(Operator.output_cells)):
            title_str = self.plot_percentiles(Operator.output_cells[i], Operator.output_classes[i], \
                    Operator.model_id, Operator.state_id)

            if(len(save_str) > 0):
                plt.savefig(save_str + title_str + '.svg') 




#### Operator
# --> holds on to data wrapper objects
# --> keeps track of model / state
class Operator:

    # dat_wrappers = list of Data Wrapper objects
    def __init__(self, dat_wrappers):
        self.dat_wrappers = dat_wrappers
        self.model_id = 0
        self.state_id = 0
        self.output_cells = []
        self.output_classes = []


    # call viz functions of all data wrappers:
    def viz(self):
        for dw in self.dat_wrappers:
            dw.viz(self)
        plt.pause(1)


    # control loop
    # 1. model_id,[model id]
    # 2. state_id,[state id]
    # 3. drive,[out_cell0],[out_cell1],...
    # ... out_cell0 = cell_id,class
    def cntrl_loop(self):
        while(True):
            s = input('change?')
            if('model_id' in s):
                sspl = s.split(',')
                self.model_id = int(sspl[1])
            elif('state_id' in s):
                sspl = s.split(',')
                self.state_id = int(sspl[1])
            elif('drive' in s):
                sspl = s.split(',')
                self.output_cells = []
                self.output_classes = []
                for i in range(1,len(sspl),2):
                    self.output_cells.append(int(sspl[i]))
                    self.output_classes.append(int(sspl[i+1]))
            # redraw:
            plt.close('all')
            self.viz()


    # save to disk
    # specify model_id and cell-class pairs
    def save_to_disk(self, save_str, model_id, state_ids, out_cells, out_classes):
        self.model_id = model_id
        self.output_cells = out_cells
        self.output_classes = out_classes
        # iter thru states:
        for i, state in enumerate(state_ids):
            # reset state:
            self.state_id = state
            # call all viz with save_str:
            for dw in self.dat_wrappers:
                dw.viz(self, save_str=save_str)




if(__name__ == '__main__'):

    import os
    #rdir = '/data/SampleSeparate/sep_zimRA0.15'

    CELL_COLOURS = ['#66a61e', '#e6ab02','#7570b3', '#e7298a',  '#1b9e77', '#d95f02']

    ## MAC

    # zim + mixed-motif
    #rdir = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimMMRA0.3'
    #save_str = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimMMRA0.3/figs/zimMM'
    #guide_shape_wid = 6
    #guide_shape_gate_cells = 4
    #gate_ylim = []
    #drive_ylim_cells = [-4.1, 4.1]
    #drive_ylim_stim = [-1., 1.]
 
    # zimbuf + mixed-motif
    #rdir = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimbufRA0.3'
    #save_str = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimbufRA0.3/figs/zimbuf'
    #guide_shape_wid = 3
    #guide_shape_gate_cells = 4
    #gate_ylim = []
    #drive_ylim_cells = [-4.1, 4.1]
    #drive_ylim_stim = [-1., 1.]

    # zimRA
    #rdir = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimRA0.3'
    #save_str = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimRA0.3/figs_cells/zim'
    #guide_shape_wid = 17
    #guide_shape_gate_cells = 4

    # ohRA
    #rdir = '/Users/ztcecere/Data/SepBalanced/sepBalanced_ohRA0.3'
    #save_str = '/Users/ztcecere/Data/SepBalanced/sepBalanced_ohRA0.3/figs_cells/oh'
    #guide_shape_wid = 17
    #guide_shape_gate_cells = 4

    # zimRA - rerun
    #rdir = '/Users/ztcecere/Data/SepBalanced/sepRERUN_zimRA0.3'
    #save_str = '/Users/ztcecere/Data/SepBalanced/sepRERUN_zimRA0.3/figs_cells/zim'
    #guide_shape_wid = 17
    #guide_shape_gate_cells = 4
    #gate_ylim = []
    #drive_ylim_cells = [-4.1, 4.1]
    #drive_ylim_stim = [-1., 1.]


    # zim + Long = SFSTnull
    rdir = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimlongRA0.3'
    save_str = '/Users/ztcecere/Data/SepBalanced/sepBalanced_zimlongRA0.3/figs'
    guide_shape_wid = 5
    guide_shape_gate_cells = 4
    gate_ylim = []
    drive_ylim_cells = [-4.1, 4.1]
    drive_ylim_stim = [-1., 1.]


    # standard xaxis:
    xax_cell = (np.arange(24) - 24) / 1.5
    xax_stim = (np.arange(24) - 24 + 6) / 1.5
    xax = [xax_cell for i in range(4)] + [xax_stim for i in range(2)]
    xax = np.vstack(xax)

    test_ll = np.load(os.path.join(rdir,'test_ll.npy'))
    drive_vars = np.load(os.path.join(rdir,'drive_vars.npy'))
    gate_vars = np.load(os.path.join(rdir,'gate_vars.npy'))
    fb = np.load(os.path.join(rdir,'timing_fb.npy'))
    
    # convert drive vars:
    guide_shapes = [[6,8],[guide_shape_wid]]
    drive_vars = convert_xdim_to_og(drive_vars, guide_shapes)
    drive_vars = drive_vars[0] # lop off worm_ids
    print(np.shape(drive_vars))

    # convert gate vars:
    guide_shapes_gate = [[guide_shape_gate_cells,8]]
    gate_vars = convert_xdim_to_og(gate_vars, guide_shapes_gate)
    gate_vars = gate_vars[0] # only 1 block
    print(np.shape(gate_vars))

    # run thru filterbank:
    drive_vars = np.tensordot(drive_vars, fb, axes=(-1,0))
    gate_vars = np.tensordot(gate_vars, fb, axes=(-1,0))
    print(np.shape(drive_vars))
    print(np.shape(gate_vars))

    # NOTE: comp_filter here just specifies how to make distro plots
    # ... does not need to match run comp_filter
    #PW = PerfWrapper(test_ll, comp_filter=np.array([1,-1,0,0,0]))
    # External
    #PW = PerfWrapper(test_ll, comp_filter=np.array([0,-1,1,0,0]))

    # 4-state:
    PW = PerfWrapper(test_ll, comp_filter=np.array([1,-1,0,0]))

    # gate wrappers ~ show all together
    GW = GateWrapperStatic(gate_vars, np.array([z for z in range(guide_shape_gate_cells)]), \
            CELL_COLOURS[:guide_shape_gate_cells], \
            xax[:guide_shape_gate_cells], tree_depth=2, ylim=gate_ylim)


    # drive wrapper args
    # drive_vars, model_per_state_id, in_cells_viz, cell_colours, xax, percs=[50, 90]
    
    # full
    #save_str += 'full'
    #DW = DriveWrapper(drive_vars, 0, np.array([0,1,2,3,4,5]), CELL_COLOURS, xax)

    # cells
    #save_str += 'cells'
    #DW = DriveWrapper(drive_vars, 0, np.array([0,1,2,3]), CELL_COLOURS[:4], xax[:4], ylim=drive_ylim_cells)

    # prime sense only
    save_str += 'stim'
    DW = DriveWrapper(drive_vars, 0, np.array([4,5]), CELL_COLOURS[4:], xax[4:], ylim=drive_ylim_stim)

    OP = Operator([PW, DW, GW])
    # control loop --> interactive
    #OP.cntrl_loop()
    # save to disk:
    # save_to_disk(self, save_str, model_id, state_ids, out_cells, out_classes)
    # AVA
    OP.save_to_disk(save_str+'AVA', 0, [0,1,2,3], [0,0,0,0], [0,1,3,4])
    # RME
    OP.save_to_disk(save_str+'RME', 0, [0,1,2,3], [1,1,1,1], [0,1,3,4])



