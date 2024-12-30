import json
import os
import torch
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils import data as torch_data

from torchdrug import data
from torchdrug.core import Registry as R

from cg_maml_steps import cg_protein

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


# for MAML-based binding affinity regression prediction tasks
@R.register("datasets.MAMLDataset")
class MAMLDataset(data.ProteinDataset, PPIDataset):
    def __init__(self, data_path, output_path, index_path, label_path, label_col_name, pickle_name, cropping_threshold, contact_threshold,
                 test_set_id, transform=None, AA_num_threshold=5000, task_tag="binding_affinity", ancestor_tag=False, verbose=1, **kwargs):

        self.task_tag = task_tag  # for determining the label name of current task
        # the following initialization will be executed when MAMLDataset is not a father class to be inherited by other Dataset class
        if not ancestor_tag:
            # data_path is the path storing the original CG files
            data_path = os.path.expanduser(data_path)
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            self.data_path = data_path
            self.output_path = output_path
            self.index_path = index_path
            self.label_path = label_path
            self.pickle_name = pickle_name
            self.test_set_id = test_set_id
            self.label_col_name = label_col_name
            self.cropping_threshold = cropping_threshold
            self.contact_threshold = contact_threshold
            print("current input path for reading the original CG files:", self.data_path)

            pkl_file = os.path.join(self.output_path, self.pickle_name)
            if not os.path.exists(self.output_path): # output pickle storage path
                os.makedirs(self.output_path)
            print("current output path for outputting processed CG pickle data file:", pkl_file)

            # load labels
            self.label_dict = self.create_labels(label_path, label_col_name=self.label_col_name)
            # load data splitting
            with open(self.index_path, "rb") as fin:
                self.split_list = pickle.load(fin)
            # {'name': array(['1a22', '1acb', '1ak4', ..., '6s29', '6saz', '6umt']), 'cluster': array([1, 3, 3, ..., 1, 1, 3])}

            # load the structure info for each sample involved
            if os.path.exists(pkl_file):
                self.load_pickle(pkl_file, transform=transform, verbose=verbose, **kwargs)
            else:
                # * all protein sub-folder names in specified path (after the sorting function to determine the sample order) *
                proteins = sorted(os.listdir(self.data_path))
                print("protein CG folder number contained in the specified folder:", len(proteins))
                cg_files = [os.path.join(self.data_path, i) for i in proteins]

                # AA_num_threshold is imposed to load_cgs so that the generated pickle read by load_pickle also satisfies the defined AA_num_threshold restriction
                self.load_cgs(cg_files, transform=transform, AA_num_threshold=AA_num_threshold, verbose=verbose, **kwargs)
                # main saving content: sample number, storage path of original cg samples, protein sequences, cg protein classes
                self.save_pickle(pkl_file, verbose=verbose)

            name4loaded_pdbs, name4split_pdbs = set([i.split("\\")[-1] for i in self.pdb_files]), set(self.split_list["name"])
            assert name4split_pdbs.issubset(name4loaded_pdbs), \
                "The loaded protein samples should fully cover the structures given by the splitting file: {} VS {}.".\
                format(len(name4loaded_pdbs), len(name4split_pdbs))

            # the validation set is not necessarily needed under current MAML setting
            self.data_split()

    # * the graph-level label can be registered via 'with protein.graph()' *
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
            cg_files = tqdm(cg_files, "constructing proteins from CG files")
        # read and process each cg protein one by one
        for i, cg_file in enumerate(cg_files):
            # for processing specific sample via its name
            # pdb_name = os.path.basename(cg_file)
            # if pdb_name != '69':
            #     continue

            complete_check, protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, AA_num_threshold=AA_num_threshold)
            if not complete_check: # not passing the complete check (currently mainly for over-large protein check)
                if isinstance(protein, str):
                    logger.debug("Cannot construct protein from the CG file `%s`. Ignore this sample." % cg_file)
                    self.must_incomplete.append(cg_file)
                    continue
                else: # for the case that the protein class is created successfully but the corresponding CG info may be incomplete
                    self.maybe_incomplete.append(cg_file)

            if hasattr(protein, "residue_feature"): # default: False
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()

            # * adding labels to current generated protein object *
            with protein.graph():
                cg_file_key = os.path.basename(cg_file) # e.g., 1a22
                # giving sample which label is not in the loaded label dictionary an -inf label
                if cg_file_key in self.label_dict.keys():
                    protein.label = torch.tensor(self.label_dict[cg_file_key])
                else:
                    protein.label = torch.tensor(float('-inf'))
                    print("protein {} is assigned an -inf label because its label is not in the loaded label dictionary".format(cg_file_key))

            self.data.append(protein) # storing CGProtein class
            self.pdb_files.append(cg_file) # original cg file local locations, e.g., /cg_demo_martini22/1brs
            self.sequences.append(protein.aa_sequence if protein else None) # storing str protein sequences

            if i % 1000 == 0:
                print("{} coarse-grained proteins have been parsed".format(i))

        # save the maybe_incomplete cg protein list into original source data folder
        if len(self.maybe_incomplete) > 0:
            # put the check result into the original protein sample storage folder, F:/UOB/External Dataset/PDBbind_source/pdbbind_dimer_strict
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "maybe_incomplete_itp.txt"), 'w') as f:
                f.writelines([line + '\n' for line in self.maybe_incomplete])

        if len(self.must_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "must_incomplete_itp.txt"), 'w') as f:
                f.writelines([line + '\n' for line in self.must_incomplete])

    def get_item(self, index):
        # * original 'clone' function for a protein is located in the basic 'graph' class, we re-write it in 'cg_protein' class *
        # * clone a protein object, transform function will be performed on it later (change view and crop graph for this clone) *
        protein = self.data[index]
        protein_name = os.path.basename(self.pdb_files[index])
        # * label needs to be cloned from original protein independently *
        protein_label = protein.label.clone()
        protein = protein.clone()

        # * earlier than 'transform' functions, as after one time of protein 'subgraph' function, the required attribute 'aa_sequence' will not be retained *
        if self.cropping_threshold:
            protein, closest_contact_distance = protein.protein_cropping(self.cropping_threshold, contact_threshold=self.contact_threshold)
            current_res_num = protein.residue_type.size(0)
            # print('retained residue and bead numbers of protein {} after protein cropping: {} and {}'.
            #       format(current_protein_name, current_res_num, protein.num_node))
            # we need to ensure that after the cropping, every downstream sample is valid
            assert current_res_num > 0, "protein {} is empty after the cropping with contact_threshold = {}, closest contact distance = {}".\
                format(protein_name, self.contact_threshold, closest_contact_distance)

        if hasattr(protein, "residue_feature"): # default: False
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        # the way of generating a batch of data to be fed into the model in every epoch
        item = {"graph": protein}
        item[self.task_tag] = protein_label
        # adding the support of providing protein name along with the CG graph
        item["name"] = protein_name
        if self.transform: # loaded in self.load_pickle or self.load_cgs via 'kwargs'
            item = self.transform(item)

        # * send item into the model after the transform functions *
        # * the items will be sent to Dataloader to be packed together as the batch data *
        return item

    def create_labels(self, label_path, label_col_name="binding_free_energy", label_upper=None, label_lower=None):
        raw_labels = pd.read_csv(label_path)
        pdb_names, selected_labels = raw_labels["pdb_code"], raw_labels[label_col_name]

        # label restriction check
        if label_upper:
            label_upper = float(label_upper)
        if label_lower:
            label_lower = float(label_lower)
        if label_upper and label_lower:
            assert label_upper > label_lower, "label_upper should be larger than label_lower: {} and {}".format(label_upper, label_lower)
        # impose restrictions to specified labels
        if label_upper:
            selected_labels[selected_labels > label_upper] = label_upper
        if label_lower:
            selected_labels[selected_labels < label_lower] = label_lower

        label_dict = dict()
        for pdb, label in zip(pdb_names, selected_labels):
            label_dict[pdb] = label

        return label_dict

    # * the validation set is not necessarily needed under current MAML setting *
    def data_split(self):
        train_pdb_files, train_data, train_sequences, train_task_assignment, train_task2sampid = [], [], [], [], {}
        test_pdb_files, test_data, test_sequences, test_task_assignment, test_task2sampid = [], [], [], [], {}
        # * the order of self.pdb_files has already been sorted based on *protein names* before the output pickle storage (i.e., before 'load_cgs' function) *
        # * i.e., via proteins = sorted(os.listdir(self.data_path)), this order will be kept within each subset after the data splitting *

        # * generate absolute sample ids for each set *
        # print(self.split_list) # {'name': array(['1a22', '1acb', '1ak4', ..., '6s29', '6saz', '6umt']), 'cluster': array([1, 3, 3, ..., 1, 1, 3])}
        protein_name, cluster_id = self.split_list["name"], self.split_list["cluster"] # obtain the task assignment for all involved numbers
        # * make protein_name and cluster_id also strictly follow the sorted order of protein names *
        protein_name_order = np.argsort(protein_name)
        protein_name = protein_name[protein_name_order]
        cluster_id = cluster_id[protein_name_order]

        self.test_set_id = sorted(list(set(
            [max(cluster_id) if i == -1 else i for i in self.test_set_id]))) # obtain ordered test task ids
        test_id = np.isin(cluster_id, self.test_set_id)
        train_id = ~test_id # cluster id contains the task assignment of all samples in whole collected dataset

        train_protein_name, test_protein_name, train_cluster_id, test_cluster_id = \
            protein_name[train_id], protein_name[test_id], cluster_id[train_id], cluster_id[test_id]

        # * this can be understood by that, the allocated sample absolute ids for each set are based on the order of train_protein_name and test_protein_name *
        train_abs_sampid, test_abs_sampid = np.arange(len(train_protein_name)), np.arange(len(test_protein_name))

        # * need to save: 1) protein names, 2) protein graphs, 3) protein sequences, 4) abs ids contained in each cluster in one test, 5) cluster assignment for each sample in one set *
        # * currently self.pdb_files, self.data, self.sequences, train_protein_name, test_protein_name, train_cluster_id, test_cluster_id should all follow the order of protein names *
        train_counter, test_counter = 0, 0
        for pdb_file, structure, sequence in zip(self.pdb_files, self.data, self.sequences):
            # pdb_file_ = os.path.basename(pdb_file) # used in local
            pdb_file_ = os.path.basename(pdb_file).split('\\')[-1] # used in server

            # all logics use 'if', in case that some samples belong to multiple sets (in certain settings from retrieved splitting files)
            if pdb_file_ in train_protein_name:
                # examine the order consistency in self.pdb_files and train_protein_name (based on the same protein name order)
                assert pdb_file_ == train_protein_name[train_counter], "There exists protein name order inconsistency between pdb_files and train_protein_name."
                train_pdb_files.append(pdb_file)
                train_data.append(structure)
                train_sequences.append(sequence)
                current_cluster_id = train_cluster_id[train_counter]
                # also follow the order of protein names to generate the sample order of each cluster/task list in this dict
                if current_cluster_id not in train_task2sampid.keys():
                    train_task2sampid[current_cluster_id] = []
                    train_task2sampid[current_cluster_id].append(train_abs_sampid[train_counter])
                else:
                    train_task2sampid[current_cluster_id].append(train_abs_sampid[train_counter])
                train_task_assignment.append(current_cluster_id)
                train_counter += 1

            if pdb_file_ in test_protein_name:
                assert pdb_file_ == test_protein_name[test_counter], "There exists protein name order inconsistency between pdb_files and test_protein_name."
                test_pdb_files.append(pdb_file)
                test_data.append(structure)
                test_sequences.append(sequence)
                current_cluster_id = test_cluster_id[test_counter]
                if current_cluster_id not in test_task2sampid.keys():
                    test_task2sampid[current_cluster_id] = []
                    test_task2sampid[current_cluster_id].append(test_abs_sampid[test_counter])
                else:
                    test_task2sampid[current_cluster_id].append(test_abs_sampid[test_counter])
                test_task_assignment.append(current_cluster_id)
                test_counter += 1

            # comment this out because meta-training + meta-test or meta-training + independent test will not take up all samples
            # if (pdb_file_ not in train_protein_name) and (pdb_file_ not in test_protein_name):
            #     raise Exception('current sample is not in any subset of data splitting: {}'.format(pdb_file_))

        # if the protein order in complete dataset and in training/test set is consistent, the following assert will also pass
        assert train_task_assignment == train_cluster_id.tolist() and test_task_assignment == test_cluster_id.tolist(), \
            "There still exists inconsistency in protein name order of each allocated set."

        # arrange the data following the order of training, validation, and test
        self.pdb_files = train_pdb_files + test_pdb_files
        self.data = train_data + test_data
        self.sequences = train_sequences + test_sequences
        self.task_assignment = train_task_assignment + test_task_assignment
        self.num_samples = [len(train_pdb_files), len(test_pdb_files)]
        self.train_task2sampid = train_task2sampid
        self.test_task2sampid = test_task2sampid
        print("number of samples in the training and test sets: {}, {}".format(self.num_samples[0], self.num_samples[-1]))

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


@R.register("datasets.STANDARDDataset")
class STANDARDDataset(MAMLDataset):
    def __init__(self, data_path, output_path, index_path, label_path, label_col_name, pickle_name, cropping_threshold, contact_threshold,
                 transform=None, AA_num_threshold=5000, task_tag="binding_affinity", verbose=1, **kwargs):
        super(STANDARDDataset, self).__init__(data_path=data_path, output_path=output_path, index_path=index_path, label_path=label_path, label_col_name=label_col_name,
                                              pickle_name=pickle_name, cropping_threshold=cropping_threshold, contact_threshold=contact_threshold, test_set_id=None,
                                              # above ones are only used for filling necessary arguments of MAMLDataset (for initialization), below are actual used ones
                                              # self.task_tag will be initialized using task_tag inside
                                              task_tag=task_tag, ancestor_tag=True)

        # data_path should be path storing the original CG files
        data_path = os.path.expanduser(data_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.data_path = data_path
        self.output_path = output_path
        self.index_path = index_path
        self.label_path = label_path
        self.pickle_name = pickle_name
        self.label_col_name = label_col_name
        self.cropping_threshold = cropping_threshold # for cropping the protein based on the specified threshold
        self.contact_threshold = contact_threshold # for determining contact residues based on inter-BB distance (if cropping_threshold != None)
        print("current input path for reading the original CG files:", self.data_path)

        pkl_file = os.path.join(self.output_path, self.pickle_name)
        if not os.path.exists(self.output_path):  # output pickle storage path
            os.makedirs(self.output_path)
        print("current output path for outputting processed CG pickle data file:", pkl_file)

        self.label_dict = self.create_labels(label_path=self.label_path, label_col_name=self.label_col_name)
        # load data splitting, in current setting, what usually retrieves is the splitting generated from corresponding meta-learning setting for a fair comparison
        with open(self.index_path) as f:
            self.split_list = json.load(f)
        for key in self.split_list:
            # removing potential duplicated samples and re-sort samples again
            self.split_list[key] = sorted(list(set([i.split("\\")[-1] for i in self.split_list[key]])))

        # load the structure info for each sample involved
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, verbose=verbose, **kwargs)
        else:
            # * all protein sub-folder names in the specified path (after the sorting function to determine the sample order) *
            proteins = sorted(os.listdir(self.data_path))
            print("protein CG folder number contained in the specified folder:", len(proteins))
            cg_files = [os.path.join(self.data_path, i) for i in proteins]

            # AA_num_threshold is imposed to load_cgs so that the generated pickle read by load_pickle also satisfies the defined AA_num_threshold restriction
            self.load_cgs(cg_files, transform=transform, AA_num_threshold=AA_num_threshold, verbose=verbose, **kwargs)
            # main saving content: sample number, storage path of original cg samples, protein sequences, cg protein classes
            self.save_pickle(pkl_file, verbose=verbose)

        print("The contained sample numbers in current splitting file:")
        for key in self.split_list:
            print(key + ": " + str(len(self.split_list[key])))
        print("The contained sample numbers in current source data file: {}".format(len(self.pdb_files)))
        print("The contained sample numbers in current label file: {}".format(len(self.label_dict.keys())))

        # * the check in current child class is to make sure that all samples in data splitting file are the subset of those in label and data source files *
        splitting_samples = []
        for key in self.split_list:
            splitting_samples.extend(self.split_list[key])
        splitting_samples = set(splitting_samples)
        sourcedata_samples = set([i.split("\\")[-1] for i in self.pdb_files])
        label_samples = set(self.label_dict.keys())

        assert splitting_samples.issubset(sourcedata_samples) and splitting_samples.issubset(label_samples), \
            "The loaded protein samples/labels should fully cover the samples given by the splitting file: {}/{} VS {}.".\
            format(len(sourcedata_samples), len(label_samples), len(splitting_samples))

        self.data_split()

    # using the same training and test set as the meta-learning setting to perform data splitting (for a fair performance comparison)
    def data_split(self):
        train_pdb_files, train_data, train_sequences = [], [], []
        test_pdb_files, test_data, test_sequences = [], [], []

        # * the order of the following three lists has already been sorted before pickle storage (i.e., before 'load_cgs' function) *
        # * this order will be kept within each subset after the data splitting *
        for pdb_file, structure, sequence in zip(self.pdb_files, self.data, self.sequences):
            # print(pdb_file) # MAML_complete_source_data\1-0-1CSE
            # used in local
            # pdb_file_ = os.path.basename(pdb_file) 
            # used in server
            pdb_file_ = os.path.basename(pdb_file).split('\\')[-1] 
            # print(pdb_file_, self.split_list["train"][0], pdb_file_ in self.split_list["train"], len(self.pdb_files))
            # 1-0-1CSE, 1-0-1CSE, True, 8453

            # all logics are 'if' in case that some samples belong to multiple sets
            if pdb_file_ in self.split_list["train"]:
                train_pdb_files.append(pdb_file)
                train_data.append(structure)
                train_sequences.append(sequence)
            if pdb_file_ in self.split_list["test"]:
                test_pdb_files.append(pdb_file)
                test_data.append(structure)
                test_sequences.append(sequence)
            # * comment this snippet out since in current setting, samples in splitting file could be a subset of those in source data file *
            # * thus, the check of successfully allocating training and test samples can be proceeded with below 'CHECK' print *
            # if (pdb_file_ not in self.split_list["train"]) and (pdb_file_ not in self.split_list["test"]):
            #     print("current sample is not in any subset of data splitting: {}".format(pdb_file_))

        # arrange the data following the order of training and test
        self.pdb_files = train_pdb_files + test_pdb_files
        self.data = train_data + test_data
        self.sequences = train_sequences + test_sequences
        self.num_samples = [len(train_pdb_files), len(test_pdb_files)]

        print("CHECK: number of samples in training and test sets: {}, {}".format(self.num_samples[0], self.num_samples[1]))
        assert self.num_samples[0] > 0 and self.num_samples[1] > 0, "At least one of the retrieved training ({}) and test ({}) sets is empty.".\
            format(self.num_samples[0], self.num_samples[1])

    @property
    def tasks(self):
        return ["binding_affinity"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: binding_affinity",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


@R.register("datasets.STANDARD2STEPDataset")
class STANDARD2STEPDataset(MAMLDataset):
    def __init__(self, data_path, output_path, index_path, finetune_path, label_path, label_col_name, pickle_name, cropping_threshold, contact_threshold,
                 transform=None, AA_num_threshold=5000, task_tag="binding_affinity", verbose=1, **kwargs):
        super(STANDARD2STEPDataset, self).__init__(data_path=data_path, output_path=output_path, index_path=index_path, label_path=label_path, label_col_name=label_col_name,
                                              pickle_name=pickle_name, cropping_threshold=cropping_threshold, contact_threshold=contact_threshold, test_set_id=None,
                                              # above ones are only used for filling necessary arguments of MAMLDataset (for initialization), below are actual used ones
                                              # self.task_tag will be initialized using task_tag inside
                                              task_tag=task_tag, ancestor_tag=True)

        # data_path should be path storing the original CG files
        data_path = os.path.expanduser(data_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.data_path = data_path
        self.output_path = output_path
        self.index_path = index_path
        self.finetune_path = finetune_path
        self.label_path = label_path
        self.pickle_name = pickle_name
        self.label_col_name = label_col_name
        self.cropping_threshold = cropping_threshold # for cropping the protein based on the specified threshold
        self.contact_threshold = contact_threshold # for determining contact residues based on inter-BB distance (if cropping_threshold != None)
        print("current input path for reading the original CG files:", self.data_path)

        pkl_file = os.path.join(self.output_path, self.pickle_name)
        if not os.path.exists(self.output_path): # output pickle storage path
            os.makedirs(self.output_path)
        print("current output path for outputting processed CG pickle data file:", pkl_file)

        self.label_dict = self.create_labels(label_path=self.label_path, label_col_name=self.label_col_name)

        # load data splitting, in current setting, what usually retrieves is the splitting generated from corresponding meta-learning setting for a fair comparison
        with open(self.index_path) as f:
            self.split_list = json.load(f)
        for key in self.split_list:
            # removing potential duplicated samples and re-sort samples again
            self.split_list[key] = sorted(list(set([i.split("\\")[-1] for i in self.split_list[key]])))

        # * current splitting list only contains training and test sets without finer splittings *
        # print(self.split_list.keys(), [len(self.split_list[i]) for i in self.split_list]) # dict_keys(['train', 'test']), [7837, 197]
        # * read the fine-tuning samples for further splitting *
        with open(self.finetune_path) as f:
            self.finetune_list = json.load(f)
        for key in self.finetune_list:
            self.finetune_list[key] = sorted(list(set([i.split("\\")[-1] for i in self.finetune_list[key]])))
        # * under current settings (i.e., testing each independent test set in each time of script running), *
        # * combine all test tasks as the fine-tuning set, i.e., marking as the validation set here *
        finetune_list = []
        for key in sorted(self.finetune_list):
            finetune_list.extend(self.finetune_list[key])
        # * generate new splitting list, based on current settings, need to move the fine-tuning list from training to validation sets *
        self.split_list["train"] = [i for i in self.split_list["train"] if i not in finetune_list] # make sure no overlapping between training and validation
        self.split_list["val"] = finetune_list
        # print([len(self.split_list[i]) for i in self.split_list]) # [7827, 197, 10]
        # print(self.split_list["train"][0], self.split_list["val"][0], self.split_list["test"][0]) # 1-0-1CSE, 3-1173-3WQB, 3-1007-1XDT

        # load the structure info for each sample involved
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, verbose=verbose, **kwargs)
        else:
            # * all protein sub-folder names in the specified path (after the sorting function to determine the sample order) *
            proteins = sorted(os.listdir(self.data_path))
            print("protein CG folder number contained in the specified folder:", len(proteins))
            cg_files = [os.path.join(self.data_path, i) for i in proteins]

            # AA_num_threshold is imposed to load_cgs so that the generated pickle read by load_pickle also satisfies the defined AA_num_threshold restriction
            self.load_cgs(cg_files, transform=transform, AA_num_threshold=AA_num_threshold, verbose=verbose, **kwargs)
            # main saving content: sample number, storage path of original cg samples, protein sequences, cg protein classes
            self.save_pickle(pkl_file, verbose=verbose)

        print("The contained sample numbers in current splitting file:")
        for key in self.split_list:
            print(key + ": " + str(len(self.split_list[key])))
        print("The contained sample numbers in current source data file: {}".format(len(self.pdb_files)))
        print("The contained sample numbers in current label file: {}".format(len(self.label_dict.keys())))

        # * the check in current child class is to make sure that all samples in data splitting file are the subset of those in label and data source files *
        splitting_samples = []
        for key in self.split_list:
            splitting_samples.extend(self.split_list[key])
        splitting_samples = set(splitting_samples)
        sourcedata_samples = set([i.split("\\")[-1] for i in self.pdb_files])
        label_samples = set(self.label_dict.keys())

        assert splitting_samples.issubset(sourcedata_samples) and splitting_samples.issubset(label_samples), \
            "The loaded protein samples/labels should fully cover the samples given by the splitting file: {}/{} VS {}.".\
            format(len(sourcedata_samples), len(label_samples), len(splitting_samples))

        self.data_split()

    # using the same training, validation, and test sets as the meta-learning setting to perform data splitting (for a fair performance comparison)
    def data_split(self):
        train_pdb_files, train_data, train_sequences = [], [], []
        val_pdb_files, val_data, val_sequences = [], [], []
        test_pdb_files, test_data, test_sequences = [], [], []

        # * the order of the following three lists has already been sorted before pickle storage (i.e., before 'load_cgs' function) *
        # * this order will be kept within each subset after the data splitting *
        for pdb_file, structure, sequence in zip(self.pdb_files, self.data, self.sequences):
            # pdb_file_ = os.path.basename(pdb_file) # used in local
            pdb_file_ = os.path.basename(pdb_file).split('\\')[-1] # used in server
            # print(pdb_file_, self.split_list["train"], pdb_file_ in self.split_list["train"])

            # all logics are 'if' in case that some samples belong to multiple sets
            if pdb_file_ in self.split_list["train"]:
                train_pdb_files.append(pdb_file)
                train_data.append(structure)
                train_sequences.append(sequence)
            if pdb_file_ in self.split_list["val"]:
                val_pdb_files.append(pdb_file)
                val_data.append(structure)
                val_sequences.append(sequence)
            if pdb_file_ in self.split_list["test"]:
                test_pdb_files.append(pdb_file)
                test_data.append(structure)
                test_sequences.append(sequence)
            # * comment this snippet out since in current setting, samples in splitting file could be a subset of those in source data file *
            # * thus, the check of successfully allocating training, validation, and test samples can be proceeded with below 'CHECK' print *
            # if (pdb_file_ not in self.split_list["train"]) and (pdb_file_ not in self.split_list["val"]) and (pdb_file_ not in self.split_list["test"]):
            #     print("current sample is not in any subset of data splitting: {}".format(pdb_file_))

        # arrange the data following the order of training and test
        self.pdb_files = train_pdb_files + val_pdb_files + test_pdb_files
        self.data = train_data + val_data + test_data
        self.sequences = train_sequences + val_sequences + test_sequences
        self.num_samples = [len(train_pdb_files), len(val_pdb_files), len(test_pdb_files)]

        print("CHECK: number of samples in training, validation, and test sets: {}, {}, {}".
              format(self.num_samples[0], self.num_samples[1], self.num_samples[2]))

        assert self.num_samples[0] > 0 and self.num_samples[1] > 0 and self.num_samples[2] > 0, \
            "At least one of the retrieved training ({}), validation ({}), and test ({}) sets is empty.".\
            format(self.num_samples[0], self.num_samples[1], self.num_samples[2])

    @property
    def tasks(self):
        return ["binding_affinity"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: binding_affinity",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))













