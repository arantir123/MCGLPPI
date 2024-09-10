# * currently consider directly read the CG original files (organized as the original generation form from MARTINI) *
import os
import logging
import warnings
from tqdm import tqdm
from torchdrug import data, utils
from torchdrug.core import Registry as R
from cg_steps import cg_protein

logger = logging.getLogger(__name__)


@R.register("datasets._3did")
class _3did(data.ProteinDataset):

    def __init__(self, path, output_path, pickle_name='_3did.pkl.gz', verbose=1, **kwargs):
        # ** the transform function which processes each protein with pre-defined functions can be input via 'kwarg' below **
        # print('kwargs for 3did dataset class:', kwargs)

        # path should be the position storing the original CG files
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.output_path = output_path
        self.pickle_name = pickle_name
        print('current input path for reading the original CG files:', self.path)
        pkl_file = os.path.join(self.output_path, self.pickle_name)
        print('current output path for outputting processed pickle data file:', pkl_file)

        # consider the case that the CG files are already processed and stored into a pickle file for subsequent reads
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            proteins = sorted(os.listdir(self.path)) # all protein sub-folder names in specified 3did dataset
            print('protein CG folder number contained in the specified folder:', len(proteins))
            cg_files = [os.path.join(self.path, i) for i in proteins]

            self.load_cgs(cg_files, verbose=verbose, **kwargs)
            # saving: sample number, storage path of original cg samples, protein sequences, cg protein classes
            self.save_pickle(pkl_file, verbose=verbose)

    # this function can be integrated into ProtainDataset class in torchdrug.data.dataset (as the alternative for ProtainDataset.load_pdbs)
    def load_cgs(self, cg_files, transform=None, verbose=0, **kwargs):
        num_sample = len(cg_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time.")
        # transform: torchdrug.transforms.transform.Compose object, kwarg: {}
        self.transform = transform
        self.kwargs = kwargs
        self.sequences = []
        self.pdb_files = []
        self.data = []
        self.maybe_incomplete = []

        if verbose:
            # generating progress bar when iterating it with specified infor
            cg_files = tqdm(cg_files, 'constructing proteins form CG files')
        # read and process each cg protein one by one
        for i, cg_file in enumerate(cg_files):
            # for processing specific sample via its name
            # pdb_name = os.path.basename(cg_file)
            # if pdb_name != '69':
            #     continue

            complete_check, protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file)
            if not complete_check: # not passing the complete check
                if isinstance(protein, str):
                    logger.debug("Can't construct protein from the CG file `%s`. Ignore this sample." % cg_file)
                    continue
                else: # for the case that the protein class is created successfully but may be incomplete
                    self.maybe_incomplete.append(cg_file)

            if hasattr(protein, "residue_feature"): # default: False
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()

            self.data.append(protein) # storing Protein class
            self.pdb_files.append(cg_file) # original cg file local locations: /cg_demo_martini22/1brs
            self.sequences.append(protein.aa_sequence if protein else None) # storing str protein sequences

            if i % 1000 == 0:
                print('{} coarse-grained proteins have been parsed'.format(i))

        if len(self.maybe_incomplete) > 0:
            head_path = os.path.split(cg_file)[0]
            with open(os.path.join(head_path, "maybe_incomplete_itp.txt"), "w") as f:
                f.writelines([line + '\n' for line in self.maybe_incomplete])

    def get_item(self, index):
        # ** original clone is located in basic graph class, we re-write it in cg_protein class **
        # ** clone a protein object transform function will be performed on it later (change view and crop graph for this clone) **
        protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"): # default: False
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        # the way of generating a batch of data to be fed into the model in every epoch
        item = {'graph': protein}
        if self.transform: # loaded in self.load_pickle or self.load_cgs via 'kwargs'
            item = self.transform(item)

        # ** send item into the model after transform functions **
        # ** the items will be sent to torchdrug.data.dataloader to be packed together as the batch data **
        return item

    def __repr__(self):
        # repr is used to output pre-defined class object information when calling function like print(object)
        # further illustration: https://zhuanlan.zhihu.com/p/80911576
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))



