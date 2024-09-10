import os
import json
import logging
import warnings
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torch.utils import data as torch_data
from cg_steps import cg_protein

logger = logging.getLogger(__name__)


class PPIDataset:

    def split(self):
        offset = 0
        splits = []
        # the original data is stored following the order of 'train', 'val', 'test' sequentially
        # self.num_samples = [splits.count("train"), splits.count("val"), splits.count("test")]
        # which stores sample occurrence times for training, validation, and test sets
        for num_sample in self.num_samples:
            # * Subset should just restrict the sample index to 'get_item' function for each subset *
            # * e.g., protein = self.data[index].clone() in 'get_item' *
            # * every time to iterate the generated subset, 'get_item' is called to create the contained samples *
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


# Note: input a two-instance interacting complex and output its pkd score (regression task, loss: MSE, metrics: MSE, MAE, Pearsonr)
# the contact residue identification could be used
@R.register("datasets.PDBBINDDataset")
# ** need to consider extra (1) protein cropping function (transform function), (2) label handling, (3) dataset splitting **
# ** currently we do not need to inherent the functions provided in siamdiff.dataset.Atom3DDataset **
class PDBBINDDataset(data.ProteinDataset, PPIDataset):
    # data_path: source data path, output_path: output pickle path, index_path: data splitting path
    def __init__(self, data_path, output_path, index_path, label_path, pickle_name='cg_pdbbind.pkl.gz', transform=None, AA_num_threshold=5000, raw_label_col='neg_log',
                 label_upper=None, label_lower=None, cropping_threshold=None, contact_threshold=8, ancestor_tag=False,
                 task_tag='binding_affinity', use_extra_label=False, verbose=1, **kwargs):

        self.task_tag = task_tag  # for determining the label name of current task
        self.use_extra_label = use_extra_label
        if not ancestor_tag:
            # data_path should be path storing the original CG files
            data_path = os.path.expanduser(data_path)
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            self.data_path = data_path
            self.output_path = output_path
            self.label_path = label_path
            self.pickle_name = pickle_name
            self.cropping_threshold = cropping_threshold # for cropping the protein based on the specified threshold
            self.contact_threshold = contact_threshold # for determining contact residues based on inter-BB distance (if cropping_threshold != None)
            print('current input path for reading the original CG files:', self.data_path)

            pkl_file = os.path.join(self.output_path, self.pickle_name)
            print('current output path for outputting processed pickle data file:', pkl_file)

            self.label_dict, self.extra_label_dict = self.create_labels(label_path, raw_label_col, label_upper, label_lower) # label_type: 'abs' or 'neg_log' for kd
            with open(index_path) as f:
                self.split_list = json.load(f)

            # consider the case that the CG files are already processed and stored into a pickle file for subsequent reads
            if os.path.exists(pkl_file):
                self.load_pickle(pkl_file, transform=transform, verbose=verbose, **kwargs)
            else:
                # ** all protein sub-folder names in specified path (after the sorting function to determine the sample order) **
                proteins = sorted(os.listdir(self.data_path))
                print('protein CG folder number contained in the specified folder:', len(proteins))
                cg_files = [os.path.join(self.data_path, i) for i in proteins]

                # currently AA_num_threshold is imposed to load_cgs so that the generated pickle read by load_pickle also satisfies the AA_num_threshold restriction
                self.load_cgs(cg_files, transform=transform, AA_num_threshold=AA_num_threshold, verbose=verbose, **kwargs)
                # saving: sample number, storage path of original cg samples, protein sequences, cg protein classes
                self.save_pickle(pkl_file, verbose=verbose)

            # start to generate data splitting indicator based on the passed index file
            # can use 'zip' function to iterate self.data, self.pdb_files, and self.sequences, for splitting dataset based on self.split_list
            # re-arrange the dataset following the order of 'train', 'val', 'test' index for easy splitting based on Pytorch Subset function
            # the sample order of each splitting Subset will be shuffled by the 'shuffle' function in Pytorch DataLoader during model training
            # the original sequences, pdb_files, and data are extracted based on all cg pdbs contained in specified folder (in a sorted order)
            # split_list example: {'train': ['1fc2', '2ptc'], 'val': ['2tgp'], 'test': ['3sgb']}
            split_list_size, pdb_files_size = [len(self.split_list[i]) for i in self.split_list.keys()], len(self.pdb_files)
            # the second checking case is for the 10-fold cross validation
            assert (sum(split_list_size) == pdb_files_size) or ((split_list_size[0] + split_list_size[-1]) == pdb_files_size) \
                or (split_list_size[0] == split_list_size[1] == split_list_size[2] == pdb_files_size), \
                "the sample number in split list and loaded CG files should be the same: {}, {}".format(split_list_size, pdb_files_size)

            self.data_split()

    # * need to loosen the residue number restriction here as we need to adapt to the downstream samples as many as possible *
    # * the graph-level label can be registered via 'with protein.graph()'
    def load_cgs(self, cg_files, transform=None, AA_num_threshold=3000, verbose=0, **kwargs): # kwarg: {}
        num_sample = len(cg_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time.")

        self.transform = transform
        self.kwargs = kwargs
        self.sequences = []
        self.pdb_files = []
        self.data = []
        self.must_incomplete = []
        self.maybe_incomplete = []

        if verbose:
            # generating progress bar when iterating it with specified info
            cg_files = tqdm(cg_files, 'constructing proteins from CG files')
        # read and process each cg protein one by one
        for i, cg_file in enumerate(cg_files):
            # for processing specific sample via its name
            # pdb_name = os.path.basename(cg_file)
            # if pdb_name != '69':
            #     continue

            complete_check, protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, AA_num_threshold=AA_num_threshold)
            if not complete_check: # not passing the complete check (currently mainly for over-large protein check)
                if isinstance(protein, str):
                    logger.debug("Can't construct protein from the CG file `%s`. Ignore this sample." % cg_file)
                    self.must_incomplete.append(cg_file)
                    continue
                else: # for the case that the protein class is created successfully but the corresponding CG info may be incomplete
                    self.maybe_incomplete.append(cg_file)

            if hasattr(protein, "residue_feature"): # default: False
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()

            # * adding labels to current generated protein object *
            # * one label (kd) is from original pdbbind file, another (dG) is from complete dG file (with loose dimer structures) *
            # * while there is only one type of label in MANYDC dataset (binary category) *
            with protein.graph():
                # * under current setting, label_dict stores all samples with kd labels, extra_label_dict saves samples with dG labels and loose dimer structures *
                # * in this case, label_dict cannot fully cover samples in extra_label_dict, as there are some samples with ic50 or ki labels in label_dict *
                # * thus, when creating a pickle file for complete dG dataset (specified in yaml file), protein.label i.e., kd label of a protein could be empty *
                # * in order to solve this, currently these samples will be assigned an -inf kd label instead **
                cg_file_key = os.path.basename(cg_file)
                if cg_file_key in self.label_dict.keys():
                    protein.label = torch.tensor(self.label_dict[cg_file_key])
                else:
                    protein.label = torch.tensor(float('-inf'))
                    print('protein {} is assigned an -inf kd label because it does not exist in original PDBBind file'.format(cg_file_key))
                # ** in current load_cgs, every sample in the dataset will be iterated to make sure that it has a dG label in extra_label_dict here **
                protein.extra_label = torch.tensor(self.extra_label_dict[cg_file_key])
            # print(protein, protein.label, protein.extra_label)
            # CG22_Protein(num_atom=870, num_bond=2028, num_residue=372), tensor(3.4685), tensor(-12.9080)
            # print(hasattr(protein, 'extra_label'), hasattr(protein, 'extra_label')) # True, False

            self.data.append(protein) # storing Protein class
            self.pdb_files.append(cg_file) # original cg file local locations: /cg_demo_martini22/1brs
            self.sequences.append(protein.aa_sequence if protein else None) # storing str protein sequences

            if i % 1000 == 0:
                print('{} coarse-grained proteins have been parsed'.format(i))

        # save the maybe_incomplete cg protein list into original source data folder
        if len(self.maybe_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "maybe_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.maybe_incomplete])

        if len(self.must_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "must_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.must_incomplete])

    # ** need to support the protein cropping for further decreasing computational cost and extracting core regions **
    def get_item(self, index):
        # ** original 'clone' function for a protein is located in the basic 'graph' class, we re-write it in 'cg_protein' class **
        # ** clone a protein object, transform function will be performed on it later (change view and crop graph for this clone) **
        protein = self.data[index]
        protein_name = os.path.basename(self.pdb_files[index])
        # ** label needs to be cloned from original protein independently **
        # ** no extra label for MANYDC dataset **
        if self.use_extra_label and hasattr(protein, 'extra_label'):
            protein_label = protein.extra_label.clone()
        else:
            protein_label = protein.label.clone()
        protein = protein.clone()

        # ** earlier than 'transform' functions, as after one time of protein 'subgraph' function, the required attribute 'aa_sequence' is not retained **
        if self.cropping_threshold:
            protein, closest_contact_distance = protein.protein_cropping(self.cropping_threshold, contact_threshold=self.contact_threshold)
            current_res_num = protein.residue_type.size(0)
            # print('retained residue and bead numbers of protein {} after protein cropping: {} and {}'.
            #       format(current_protein_name, current_res_num, protein.num_node)) # from 195 to 81

            # we need to ensure that after the cropping, every downstream sample is valid
            assert current_res_num > 0, "protein {} is empty after the cropping with contact_threshold = {}, closest contact distance = {}".\
                format(protein_name, self.contact_threshold, closest_contact_distance)

        if hasattr(protein, 'residue_feature'): # default: False
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        # the way of generating a batch of data to be fed into the model in every epoch
        item = {'graph': protein}
        item[self.task_tag] = protein_label
        if self.transform: # loaded in self.load_pickle or self.load_cgs via 'kwargs'
            item = self.transform(item)

        # ** send item into the model after transform functions **
        # ** the items will be sent to torchdrug.data.dataloader to be packed together as the batch data **
        return item

    def create_labels(self, label_path, label_type='neg_log', label_upper=None, label_lower=None, extra_filename='PDBBINDdimer_strict_index.csv'):
        # (1) for creating label dict based on original pdbbind label file
        raw_labels = pd.read_csv(label_path)
        # ** currently only kd-based labels are considered **
        raw_labels = raw_labels[raw_labels['label_type'] == 'Kd']
        if label_type == 'neg_log':
            label_type = 'nelog10_abs_label_value'
        elif label_type == 'abs':
            label_type = 'abs_label_value'
        else:
            raise NameError

        pdb_names, resolutions, refined_labels = raw_labels['pdb_code'], raw_labels['resolution'], raw_labels[label_type].to_numpy()
        # label restriction check
        if label_upper:
            label_upper = float(label_upper)
        if label_lower:
            label_lower = float(label_lower)
        if label_upper and label_lower:
            assert label_upper > label_lower, "label_upper should be larger than label_lower: {} and {}".format(label_upper, label_lower)
        # impose restrictions to specified labels
        if label_upper:
            refined_labels[refined_labels > label_upper] = label_upper
        if label_lower:
            refined_labels[refined_labels < label_lower] = label_lower

        label_dict = dict()
        for pdb, label in zip(pdb_names, refined_labels):
            label_dict[pdb] = label

        # (2) for creating extra label dict based on dG label file (with loose dimer structures) which stays in the same path as original pdbbind file
        extra_labels = pd.read_csv(os.path.join(os.path.dirname(label_path), extra_filename))
        # get the delta G value label
        extra_pdb_names, extra_dg_labels = extra_labels['pdb_code'], extra_labels['binding_free_energy']

        extra_label_dict = dict()
        for pdb, label in zip(extra_pdb_names, extra_dg_labels):
            extra_label_dict[pdb] = label

        return label_dict, extra_label_dict

    def data_split(self):
        train_pdb_files, train_data, train_sequences = [], [], []
        val_pdb_files, val_data, val_sequences = [], [], []
        test_pdb_files, test_data, test_sequences = [], [], []

        # ** the order of the following three lists have already been sorted before pickle storage (i.e., before 'load_cgs' function) **
        # ** this order will be kept within each subset after the data splitting **
        for pdb_file, structure, sequence in zip(self.pdb_files, self.data, self.sequences):
            # pdb_file_ = os.path.basename(pdb_file) # used in local
            pdb_file_ = os.path.basename(pdb_file).split('\\')[-1] # used in server

            # all logics are 'if' in case that some samples belong to multiple sets
            if pdb_file_ in self.split_list['train']:
                train_pdb_files.append(pdb_file)
                train_data.append(structure)
                train_sequences.append(sequence)
            if pdb_file_ in self.split_list['val']:
                val_pdb_files.append(pdb_file)
                val_data.append(structure)
                val_sequences.append(sequence)
            if pdb_file_ in self.split_list['test']:
                test_pdb_files.append(pdb_file)
                test_data.append(structure)
                test_sequences.append(sequence)
            if (pdb_file_ not in self.split_list['train']) and (pdb_file_ not in self.split_list['val'])\
                    and (pdb_file_ not in self.split_list['test']):
                print('current sample is not in any subset of data splitting: {}'.format(pdb_file_))

        # arrange the data following the order of training, validation, and test
        self.pdb_files = train_pdb_files + val_pdb_files + test_pdb_files
        self.data = train_data + val_data + test_data
        self.sequences = train_sequences + val_sequences + test_sequences
        self.num_samples = [len(train_pdb_files), len(val_pdb_files), len(test_pdb_files)]
        print('number of samples in training, validation, and test sets: {}, {}, {}'.format(
            self.num_samples[0], self.num_samples[1], self.num_samples[2]))

    @property
    def tasks(self):
        """List of tasks."""
        return ["binding_affinity"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: binding_affinity",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


# in current plan, other pose scoring tasks that take a protein complex as the input and output a score for it will inherit the basic function of PDBBINDDataset
# Note: input a dimer and output its category: biological or crystal (balanced binary classification task, loss: BCE, metrics: AUROC+AUPR)
# the contact residue identification could be used (cropping_threshold could range from 6A to 14A)
@R.register("datasets.MANYDCDataset")
class MANYDCDataset(PDBBINDDataset):
    # when the inherited class contains initialization function, the 'super' needs to be used to initialize the ancestors
    def __init__(self, data_path, output_path, index_path, label_path, pickle_name='cg_manydc.pkl.gz', transform=None, AA_num_threshold=5000, raw_label_col='label',
                 cropping_threshold=None, contact_threshold=8, verbose=1, **kwargs):
        super(MANYDCDataset, self).__init__(data_path=data_path, output_path=output_path, index_path=index_path, label_path=label_path,
                                            ancestor_tag=True, task_tag='interface_class', use_extra_label=False)
        # data_path should be path storing the original CG files
        data_path = os.path.expanduser(data_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.data_path = data_path
        self.output_path = output_path
        self.label_path = label_path
        self.pickle_name = pickle_name
        self.cropping_threshold = cropping_threshold # for cropping the protein based on the specified threshold
        self.contact_threshold = contact_threshold # for determining contact residues based on inter-BB distance (if cropping_threshold != None)
        print('current input path for reading the original CG files:', self.data_path)

        pkl_file = os.path.join(self.output_path, self.pickle_name)
        print('current output path for outputting processed pickle data file:', pkl_file)

        self.label_dict = self.create_labels(label_path=label_path, label_type=raw_label_col)
        # generate the json file storing the data splitting information
        with open(index_path) as f:
            self.split_list = json.load(f)

        # consider the case that the CG files are already processed and stored into a pickle file for subsequent reads
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, verbose=verbose, **kwargs)
        else:
            proteins = sorted(os.listdir(self.data_path))  # all protein sub-folder names in specified path
            print('protein CG folder number contained in the specified folder:', len(proteins))
            cg_files = [os.path.join(self.data_path, i) for i in proteins]

            # currently AA_num_threshold is imposed to load_cgs so that the generated pickle read by load_pickle also satisfies the AA_num_threshold restriction
            self.load_cgs(cg_files, transform=transform, AA_num_threshold=AA_num_threshold, verbose=verbose, **kwargs)
            # saving: sample number, storage path of original cg samples, protein sequences, cg protein classes
            self.save_pickle(pkl_file, verbose=verbose)

        split_list_size, pdb_files_size = sum([len(self.split_list[i]) for i in self.split_list.keys()]), len(self.pdb_files)
        assert split_list_size == pdb_files_size, \
            "the sample number in split list and loaded CG files should be the same: {}, {}".format(split_list_size, pdb_files_size)
        print('the sample number in each of current set:', [len(self.split_list[i]) for i in self.split_list.keys()])

        self.data_split()

    # * need to loosen the residue number restriction here as we need to adapt to the downstream samples as many as possible *
    # * the graph-level label can be registered via 'with protein.graph()'
    def load_cgs(self, cg_files, transform=None, AA_num_threshold=3000, verbose=0, **kwargs): # kwarg: {}
        num_sample = len(cg_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time.")

        self.transform = transform
        self.kwargs = kwargs
        self.sequences = []
        self.pdb_files = []
        self.data = []
        self.must_incomplete = []
        self.maybe_incomplete = []

        if verbose:
            # generating progress bar when iterating it with specified info
            cg_files = tqdm(cg_files, 'constructing proteins from CG files')
        # read and process each cg protein one by one
        for i, cg_file in enumerate(cg_files):
            # for processing specific sample via its name
            # pdb_name = os.path.basename(cg_file)
            # if pdb_name != '69':
            #     continue

            complete_check, protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, AA_num_threshold=AA_num_threshold)
            if not complete_check: # not passing the complete check (currently mainly for over-large protein check)
                if isinstance(protein, str):
                    logger.debug("Can't construct protein from the CG file `%s`. Ignore this sample." % cg_file)
                    self.must_incomplete.append(cg_file)
                    continue
                else: # for the case that the protein class is created successfully but the corresponding CG info may be incomplete
                    self.maybe_incomplete.append(cg_file)

            if hasattr(protein, "residue_feature"): # default: False
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()

            # adding label to current generated protein object
            with protein.graph():
                protein.label = torch.tensor(self.label_dict[os.path.basename(cg_file)])

            self.data.append(protein) # storing Protein class
            self.pdb_files.append(cg_file) # original cg file local locations: /cg_demo_martini22/1brs
            self.sequences.append(protein.aa_sequence if protein else None) # storing str protein sequences

            if i % 1000 == 0:
                print('{} coarse-grained proteins have been parsed'.format(i))

        # save the maybe_incomplete cg protein list into original source data folder
        if len(self.maybe_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "maybe_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.maybe_incomplete])

        if len(self.must_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "must_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.must_incomplete])

    # rewrite the create_labels function
    def create_labels(self, label_path, label_type='label', label_upper=None, label_lower=None):
        raw_labels = pd.read_csv(label_path)
        pdb_names, data_sources, refined_labels = raw_labels['pdb_code'], raw_labels['dataset'], raw_labels[label_type].to_numpy()

        label_dict = dict()
        for pdb, label in zip(pdb_names, refined_labels):
            label_dict[pdb] = label

        return label_dict

    @property
    def tasks(self):
        """List of tasks."""
        return ["interface_class"]

    def __repr__(self):
        # what len(self) returns: https://blog.csdn.net/weixin_45580017/article/details/124553090
        lines = [
            "#sample: %d" % len(self), # currently based on len(self.data)
            "#task: interface_class",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


# in current plan, other pose scoring tasks that take a protein complex as the input and output a score for it will inherit the basic function of PDBBINDDataset
# Note: input a dimer and output its binding affinity: (regression task, loss: MSE, metrics: RMSE+MAE+PEARSONR)
# the contact residue identification could be used (cropping_threshold could use 10A)
@R.register("datasets.ATLASDataset")
class ATLASDataset(PDBBINDDataset):
    # when the inherited class contains initialization function, the 'super' needs to be used to initialize the ancestors
    def __init__(self, data_path, output_path, index_path, label_path, pickle_name='cg_atlas.pkl.gz', transform=None, AA_num_threshold=5000, raw_label_col='dG_update',
                 cropping_threshold=None, contact_threshold=8, use_extra_label=False, verbose=1, **kwargs):
        super(ATLASDataset, self).__init__(data_path=data_path, output_path=output_path, index_path=index_path, label_path=label_path,
                                            ancestor_tag=True, task_tag='binding_affinity', use_extra_label=use_extra_label)
        # data_path should be path storing the original CG files
        data_path = os.path.expanduser(data_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.data_path = data_path
        self.output_path = output_path
        self.label_path = label_path
        self.pickle_name = pickle_name
        self.cropping_threshold = cropping_threshold # for cropping the protein based on the specified threshold
        self.contact_threshold = contact_threshold # for determining contact residues based on inter-BB distance (if cropping_threshold != None)
        print('current input path for reading the original CG files:', self.data_path)

        pkl_file = os.path.join(self.output_path, self.pickle_name)
        print('current output path for outputting processed pickle data file:', pkl_file)

        # ** for the label type to be predicted, only dG is selected for cross-validation, while ddG would be only used for case study due to its incompleteness **
        # ** for the used dG label, the missing dG label can be filled using the corresponding kd label (if it exists) based on -1.3363 * -np.log10(kd * (1e-6)) **
        # * the label file can be complete (rather than only including labels of subset of original dataset) *
        self.label_dict, self.extra_label_dict = self.create_labels(label_path=label_path, label_type=raw_label_col)

        # generate the json file storing the data splitting information
        with open(index_path) as f:
            self.split_list = json.load(f)

        # consider the case that the CG files are already processed and stored into a pickle file for subsequent reads
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, verbose=verbose, **kwargs)
        else:
            proteins = sorted(os.listdir(self.data_path))  # all protein sub-folder names in specified path
            print('protein CG folder number contained in the specified folder:', len(proteins))
            cg_files = [os.path.join(self.data_path, i) for i in proteins]

            # currently AA_num_threshold is imposed to load_cgs so that the generated pickle read by load_pickle also satisfies the AA_num_threshold restriction
            self.load_cgs(cg_files, transform=transform, AA_num_threshold=AA_num_threshold, verbose=verbose, **kwargs)
            # saving: sample number, storage path of original cg samples, protein sequences, cg protein classes
            self.save_pickle(pkl_file, verbose=verbose)

        split_list_size, pdb_files_size = [len(self.split_list[i]) for i in self.split_list.keys()], len(self.pdb_files)
        # the second checking case is for the 10-fold cross validation
        assert (sum(split_list_size) == pdb_files_size) or ((split_list_size[0] + split_list_size[-1]) == pdb_files_size), \
            "the sample number in split list and loaded CG files should be the same: {}, {}".format(split_list_size, pdb_files_size)

        self.data_split()

    # * need to loosen the residue number restriction here as we need to adapt to the downstream samples as many as possible *
    # * the graph-level label can be registered via 'with protein.graph()'
    def load_cgs(self, cg_files, transform=None, AA_num_threshold=3000, verbose=0, **kwargs): # kwarg: {}
        num_sample = len(cg_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time.")

        self.transform = transform
        self.kwargs = kwargs
        self.sequences = []
        self.pdb_files = []
        self.data = []
        self.must_incomplete = []
        self.maybe_incomplete = []

        if verbose:
            # generating progress bar when iterating it with specified info
            cg_files = tqdm(cg_files, 'constructing proteins from CG files')
        # read and process each cg protein one by one
        for i, cg_file in enumerate(cg_files):
            # for processing specific sample via its name
            # pdb_name = os.path.basename(cg_file)
            # if pdb_name != '69':
            #     continue

            complete_check, protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, AA_num_threshold=AA_num_threshold)
            if not complete_check: # not passing the complete check (currently mainly for over-large protein check)
                if isinstance(protein, str):
                    logger.debug("Can't construct protein from the CG file `%s`. Ignore this sample." % cg_file)
                    self.must_incomplete.append(cg_file)
                    continue
                else: # for the case that the protein class is created successfully but the corresponding CG info may be incomplete
                    self.maybe_incomplete.append(cg_file)

            if hasattr(protein, "residue_feature"): # default: False
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()

            # adding label to current generated protein object
            with protein.graph():
                protein.label = torch.tensor(float(self.label_dict[os.path.basename(cg_file)])) # dG label, could have '\N'
                protein.extra_label = torch.tensor(float(self.extra_label_dict[os.path.basename(cg_file)])) # kd label, could have 'n.d.'

            self.data.append(protein) # storing Protein class
            self.pdb_files.append(cg_file) # original cg file local locations: /cg_demo_martini22/1brs
            self.sequences.append(protein.aa_sequence if protein else None) # storing str protein sequences

            if i % 1000 == 0:
                print('{} coarse-grained proteins have been parsed'.format(i))

        # save the maybe_incomplete cg protein list into original source data folder
        if len(self.maybe_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "maybe_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.maybe_incomplete])

        if len(self.must_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "must_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.must_incomplete])

    # rewrite the create_labels function
    def create_labels(self, label_path, label_type='dG_update', extra_label_type='kd_update'):
        raw_labels = pd.read_csv(label_path)
        # ** screen out samples without updated dG labels (i.e., samples which dG labels cannot be filled by corresponding kd labels) **
        # ** before reading the ATLAS csv file, the duplicate WT rows have already been removed manually (the first WT row is selected) **
        # ** for the updated kd and dG labels, currently the updated dG labels are filled by the updated kd labels, **
        # ** (but the filling formula coefficient can be changed, current scheme: for existing dG: keep (alt: -1.3363), missing: -1.3363) **
        # ** since after the below screening, all kd_update values of retained samples must be a real value instead of 'n.d.' **
        raw_labels = raw_labels[raw_labels['dG_update'] != '\\N'] # (534, 22)
        pdb_names, labels, extra_labels = raw_labels['pdb_code'], raw_labels[label_type].to_numpy(), raw_labels[extra_label_type].to_numpy()
        # ** another point is keeping the consistency between labels and pdb structures (i.e., removing structures without updated dG labels) **

        label_dict, extra_label_dict = dict(), dict()
        for pdb, label, extra_label in zip(pdb_names, labels, extra_labels):
            label_dict[pdb] = label
            extra_label_dict[pdb] = extra_label

        return label_dict, extra_label_dict

    @property
    def tasks(self):
        """List of tasks."""
        return ["binding_affinity"]

    def __repr__(self):
        # what len(self) returns: https://blog.csdn.net/weixin_45580017/article/details/124553090
        lines = [
            "#sample: %d" % len(self), # currently based on len(self.data)
            "#task: binding_affinity",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


if __name__ == "__main__":
    # Dataset function test
    # the path for storing the downstream source data
    # test_data_path = 'D:/PROJECT B2_5/note/CG_summary/cg_martini22_downstream_toy/'
    test_data_path = 'D:/PROJECT B2_5/note/CG_summary/cg_martini22_manydc_toy/'

    # the path for storing the output processed pickle file
    # test_output_path = 'D:/PROJECT B2_5/note/CG_summary/cg_demo_martini22_output/'
    test_output_path = 'D:/PROJECT B2_5/note/CG_summary/cg_demo_martini22_output/'

    # the path for storing the index of dataset splitting
    # test_index_path = 'D:/PROJECT B2_5/note/CG_summary/cg_demo_martini22_index/martini22_index.json'
    test_index_path = 'D:/PROJECT B2_5/note/CG_summary/cg_demo_martini22_index/martini22_manydc_index.json'

    # the path for storing all label informatipn of current dataset
    # test_label_path = '../downstream_files/PDBBIND/PDBBIND.csv'
    test_label_path = '../downstream_files/MANYDC/MANYDC.csv'

    # dataset = PDBBINDDataset(test_data_path, test_output_path, test_index_path, test_label_path, label_type='neg_log', label_upper=6, label_lower=-2)
    dataset = MANYDCDataset(test_data_path, test_output_path, test_index_path, test_label_path)

    print('dataset:', dataset)