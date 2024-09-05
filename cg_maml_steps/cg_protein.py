import copy
import glob
import os.path
import string
import numpy as np
import torch
import warnings
from rdkit import Chem
from collections.abc import Sequence
from collections import defaultdict
from torch_scatter import scatter_add, scatter_max, scatter_min

from torchdrug import utils
from torchdrug.utils import pretty
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.data import Molecule, PackedMolecule, Dictionary, feature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# general mapping for protein
residue_symbol2abbr = {"GLY": "G", "ALA": "A", "SER": "S", "PRO": "P", "VAL": "V", "THR": "T", "CYS": "C", "ILE": "I",
                       "LEU": "L", "ASN": "N", "ASP": "D", "GLN": "Q", "LYS": "K", "GLU": "E", "MET": "M", "HIS": "H",
                       "PHE": "F", "ARG": "R", "TYR": "Y", "TRP": "W"}

abbr2residue_symbol = {v: k for k, v in residue_symbol2abbr.items()}

class CG22_Protein(Molecule):
    # Protein class established with coarse-grain features
    # currently the nodes and edges between nodes are established based on CG
    _meta_types = {"node", "edge", "residue", "graph",
                   "node reference", "edge reference", "residue reference", "graph reference"}

    # stardard residue/atom id mapping
    residue2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}
    residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
                         "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19}
    atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                    "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}
    alphabet2id = {c: i for i, c in enumerate(" " + string.ascii_uppercase + string.ascii_lowercase + string.digits)}

    id2residue = {v: k for k, v in residue2id.items()}
    id2residue_symbol = {v: k for k, v in residue_symbol2id.items()}
    id2atom_name = {v: k for k, v in atom_name2id.items()}
    id2alphabet = {v: k for k, v in alphabet2id.items()}

    # coarse-grained molecule id mapping
    martini22_name2id = {"AC1": 0, "AC2": 1, "C3": 2, "C5": 3, "N0": 4, "Na": 5, "Nd": 6, "Nda": 7, "P1": 8, # standard martini
                         "P4": 9, "P5": 10, "Qa": 11, "Qd": 12, "SC4": 13, "SC5": 14, "SNd": 15, "SP1": 16}
    elenedyn22_name2id = {"C1": 0, "C2": 1, "C3": 2, "C5": 3, "N0": 4, "Na": 5, "Nd": 6, "Nda": 7, "P1": 8, # elastic network
                         "P4": 9, "P5": 10, "Qa": 11, "Qd": 12, "SC4": 13, "SC5": 14, "SNd": 15, "SP1": 16}
    martini22_bond2id = {'backbone_bonds': 0, 'sidechain_bonds': 1, 'sheet_bonds_3': 2, 'sheet_bonds_4': 3, 'constraints': 4} # itp file bond types
    martini22_beadpos2id = {'BB': 0, 'SC1': 1, 'SC2': 2, 'SC3': 3, 'SC4': 4} # bead position categories
    martini22_angletype2id = {'backbone_angles': 0, 'backbone_sidec_angles': 1, 'sidechain_angles': 2, 'backbone_dihedrals': 3}

    id2martini22_name = {v: k for k, v in martini22_name2id.items()}
    id2elenedyn22_name = {v: k for k, v in elenedyn22_name2id.items()}
    id2martini22_bond = {v: k for k, v in martini22_bond2id.items()}
    id2martini22_beadpos = {v: k for k, v in martini22_beadpos2id.items()}
    id2martini22_angletype = {v: k for k, v in martini22_angletype2id.items()}

    def __init__(self, edge_list=None, bead_type=None, bead2residue=None, bond_type=None, residue_type=None, aa_sequence=None, view=None,
                 backbone_angles=None, backbone_sidec_angles=None, sidechain_angles=None, backbone_dihedrals=None, intermol_mat=None, **kwargs):
        # 1. currently (only) edge_list index is set to starting from 0, while the index (for edges, angles, and dihedrals, etc.) in orginial CG itp files starts from 1
        # 2. initialize the ancestral molecule and graph classes with edge_list and num_node (contained in kwargs recording bead number for current protein)
        # 3. store the CG-level bead_type, bond_type (only containing bond types provided in cg itp files), node_position into molecule class as self.atom_type, self.bond_type, self.node_position
        # 4. kwargs.keys(): dict_keys(['node_position', 'num_node', 'num_relation']) for finding defined hyperparameters in ancestral class

        # for the case: https://www.jianshu.com/p/c4cdf09642ca?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation
        # this case is caused by the conflict between atom_type and defined bead_type
        if 'atom_type' in kwargs.keys():
            # print(bead_type) # None
            bead_type = kwargs['atom_type']
            kwargs.pop('atom_type')
            super(CG22_Protein, self).__init__(edge_list, atom_type=bead_type, bond_type=bond_type, **kwargs)
        else:
            super(CG22_Protein, self).__init__(edge_list, atom_type=bead_type, bond_type=bond_type, **kwargs)

        residue_type, num_residue = self._standarize_num_residue(residue_type)
        self.num_residue = num_residue
        self.view = self._standarize_view(view) # default: atom after standarization (if it is not specified by transforms.ProteinView)
        self.aa_sequence = aa_sequence # alphabet string with '.' split (not registered in context manager as attributes to be registered should be torch tensors)

        # BBB (2nd as center)
        self.backbone_angles = self._standarize_angle(backbone_angles)
        # BBS (3rd as center)
        self.backbone_sidec_angles = self._standarize_angle(backbone_sidec_angles)
        # BSS (3rd as center)
        self.sidechain_angles = self._standarize_angle(sidechain_angles)
        # BBBB (2nd as center), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure
        self.backbone_dihedrals = self._standarize_angle(backbone_dihedrals)
        # core region information (include potential inter-molecular AA pairs between specified interaction parts)
        self.intermol_mat = self._standarize_angle(intermol_mat)

        # bead2residue index starts from 0
        bead2residue = self._standarize_attribute(bead2residue, self.num_node)

        with self.atom():
            with self.residue_reference():
                self.bead2residue = bead2residue

        with self.residue():
            self.residue_type = residue_type # tensor idx

    def residue(self):
        return self.context("residue")

    def residue_reference(self):
        return self.context("residue reference")

    @property
    def node_feature(self):
        if getattr(self, "view", "atom") == "atom":
            return self.atom_feature
        else:
            return self.residue_feature

    @node_feature.setter
    def node_feature(self, value):
        self.atom_feature = value

    @property
    def num_node(self):
        return self.num_atom

    @num_node.setter
    def num_node(self, value):
        self.num_atom = value

    def _check_attribute(self, key, value):
        super(CG22_Protein, self)._check_attribute(key, value)
        for type in self._meta_contexts:
            if type == "residue":
                if len(value) != self.num_residue:
                    raise ValueError("Expect residue attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_residue, value.shape))
            elif type == "residue reference":
                is_valid = (value >= -1) & (value < self.num_residue)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect residue reference in [-1, %d), but found %d" % (self.num_residue, error_value[0]))

    def _standarize_attribute(self, attribute, size, dtype=torch.long, default=0):
        if attribute is not None:
            attribute = torch.as_tensor(attribute, dtype=dtype, device=self.device)
        else:
            if isinstance(size, torch.Tensor):
                size = size.tolist()
            if not isinstance(size, Sequence):
                size = [size]
            attribute = torch.full(size, default, dtype=dtype, device=self.device)
        return attribute

    def _standarize_num_residue(self, residue_type):
        if residue_type is None:
            raise ValueError("`residue_type` should be provided")

        residue_type = torch.as_tensor(residue_type, dtype=torch.long, device=self.device)
        num_residue = torch.tensor(len(residue_type), device=self.device)
        return residue_type, num_residue

    def _standarize_angle(self, angle):
        if angle is None:
            return torch.zeros(0, device=self.device)
        else:
            return torch.as_tensor(angle, dtype=torch.long, device=self.device)

    def __setattr__(self, key, value):
        # https://www.runoob.com/python/python-func-setattr.html
        if key == "view" and value not in ["atom", "residue"]:
            raise ValueError("Expect `view` to be either `atom` or `residue`, but found `%s`" % value)
        return super(CG22_Protein, self).__setattr__(key, value)

    def _standarize_view(self, view):
        if view is None:
            if self.num_atom > 0:
                view = "atom"
            else:
                view = "residue"
        return view

    @classmethod
    # cgfile is the path storing the original generate CG files
    def from_cg_molecule(cls, cgfile, AA_num_threshold=3000):
        # need to extract information from original CG files (then the corresponding features are calculated based on the extractions)
        # in original implementation of atom-level view (in torchdrug), the local pdbs are read by rdkit and then processed by Molecule.from_molecule function
        # here we need to rewrite these two functions and combine them together (currently only standard martini2.2 generated structures are supported)
        # ** besides, for downstream tasks, the protein regions of interest may need to be screened/cropped (e.g., to select interface/interacting regions) **
        # ** which should be carefully considered for the both pre-training and downstream task designs (for better flexibility) **

        # 1. read the files from local positions
        # normally, for standard (atom-level) pdb files, we need to screen out incomplete proteins (e.g., pdb with only CA being retained) and remove all H atoms
        # besides, completing side chain atoms and screening out over-large proteins may be needed (determined by specific downstream tasks)
        # ** however, for CG files, the requirement/conclusion of generating CGs includes 1. complete backbone and side chain atoms 2. non-natural AAs being removed 3. AALA/BALA issues already being solved **
        # ** 4. multiple sets of same chain-residue tokens are not included in the same original pdb 5. HETATM rows in standard pdb files will be automatically ignored by martini programme **
        # ** thus, here we can assume that all CG files read here are canonical following the same rules, the necessary case to be considered here is the over-large proteins (currently other special cases can be ignored) **
        complete_check, cg_info = cls.cg_file_reader(cgfile, AA_num_threshold)
        # isinstance(cg_info, str)=True: fail to create protein class (e.g., over-large proteins returned with related info), isinstance(cg_info, dict)=True: succeed to create protein class (but maybe incomplete)
        if (not complete_check) and (isinstance(cg_info, str)):
            return complete_check, cg_info

        # 2. generating basic features for local storage (some advanced features could be calculated after reading the storage file and starting model training)
        # ** residue serial number in CG pdb file does not start from 1, while keeping the number in original pdb file, while it will be re-numbered in CG itp files **

        # ** other than the special AALA/BALA cases, another type of special cases are 100A/100B/100C cases, where A/B/C are the [27th-row-digit] residue serial numbers, actually representing different AAs **
        # ** while CG pdb will keep the original A/B/C names, and re-number these names from 1 in itp, thus, current we can ignore such cases if the effective cg files are generated under assumption that **
        # ** the residue serial numbers contained in the generated cg files are effective (in atom-level siamdiff, serial numbers are explicitly checked in the implementation, here we ignore it at first) **
        edge_list, bead_type, bead2residue, node_position, bond_type, num_node, num_relation, residue_type, aa_sequence, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals = \
            cls.cg_feature_generator(cg_info, cgfile)
        # for backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals, their entries will be assigned to bead nodes (as the node features)
        # current allocation logic: backbone_angles: assign it to the middle node, backbone_sidec_angles: assign to its side chain node (SBB scheme for such angles)
        # sidechain_angles: assign to the backbone node (the smallest node id for each entry), backbone_dihedrals: assign to the 2nd or 3rd bead node

        return complete_check, \
               cls(edge_list, bead_type=bead_type, bead2residue=bead2residue, node_position=node_position,
                   bond_type=bond_type, num_node=num_node, num_relation=num_relation, residue_type=residue_type, aa_sequence=aa_sequence,
                   backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles, sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals)

    def clone(self):
        # clone this graph, following the implementation of torchdrug.graph.clone()
        # print(type(self.edge_list), type(self.atom_type), type(self.bead2residue), type(self.node_position), type(self.bond_type), type(self.residue_type), type(self.aa_sequence))
        # ** considering also clone intermol_mat (i.e., the potential inter-molecular AA pair matrix of the identified core region based on pre-defined contact_threshold) **

        # ** the 'clone' function used in this return function is based on _TensorBase of Pytorch **
        return type(self)(self.edge_list.clone(), bead_type=self.atom_type.clone(), bead2residue=self.bead2residue.clone(), node_position=self.node_position.clone(),
               bond_type=self.bond_type.clone(), num_node=self.num_node, num_relation=self.num_relation, residue_type=self.residue_type.clone(), aa_sequence=copy.copy(self.aa_sequence),
               backbone_angles=self.backbone_angles.clone(), backbone_sidec_angles=self.backbone_sidec_angles.clone(), sidechain_angles=self.sidechain_angles.clone(),
               backbone_dihedrals=self.backbone_dihedrals.clone(), intermol_mat=self.intermol_mat.clone())

    @classmethod
    def cg_file_reader(cls, cgfile, AA_num_threshold=3000):
        pdb = os.path.basename(cgfile)
        # topology files
        itp_paths = sorted(glob.glob(os.path.join(cgfile, '*.itp')))
        itp_lines_dict = dict()
        for itp_path in itp_paths:
            with open(itp_path) as f:
                itp_lines = f.readlines()
            # * update the logic of generating keys of itp_lines_dict: *
            # original:
            # itp_lines_dict[os.path.basename(itp_path).split('.')[0]] = itp_lines # keys: Protein_A, Protein_D (to identify chains)
            # update (for handling the itp name with '.', if '.' contains in the name, the above key generation is wrong like below):
            # print(os.path.basename(itp_path), os.path.basename(itp_path).split('.'))
            # MT-1AO7_E63Q.K66A_A.A_WT_nan_WT-cg_B.itp ['MT-1AO7_E63Q', 'K66A_A', 'A_WT_nan_WT-cg_B', 'itp']
            itp_lines_dict[os.path.basename(itp_path)[:-4]] = itp_lines

        # CG pdb file
        cg_pdb_path = os.path.join(cgfile, pdb + '-' + 'cg.pdb')
        if os.path.exists(cg_pdb_path):
            with open(cg_pdb_path) as f:
                cg_lines = f.readlines()
        else:
            with open(os.path.join(cgfile, 'cg.pdb')) as f:
                cg_lines = f.readlines()

        # 1. identifying over-large proteins 2. only explicitly retaining 'ATOM' and 'TER' entries
        # ** in real pre-training cg dataset, there are some proteins which contain the non-consecutive chain (i.e., part of this chain is missing) **
        # ** in this case, for cg pdb file, a 'TER' row will be added into atom rows of this chain to indicate the missiong position **
        # ** thus, in cg pdb file, the residue id in such chains is not consecutive (with extra 'TER' row), while in itp files, these residue ids will be re-numbered consecutively **
        complete_check, cg_pdb_info = cleaning_cg_pdb(cg_lines, pdb, AA_num_threshold=AA_num_threshold)
        if not complete_check: # not passing the check
            return complete_check, cg_pdb_info # over-large protein info

        # 2. detach and classify the information contained in each itp chain file
        cg_itp_info = dict()
        for key in itp_lines_dict.keys(): # key: Protein_A
            chain_lines = itp_lines_dict[key]
            chain_dict_ = cleaning_cg_itp(chain_lines, pdb)
            complete_check2, chain_dict = chain_dict_[0], chain_dict_[1]
            cg_itp_info[key] = chain_dict

        # complete check = True: passing the check, False: not passing the check
        return complete_check & complete_check2, {'cg_pdb_info': cg_pdb_info, 'cg_itp_info': cg_itp_info}

    @classmethod
    # adding the support to None bond info and angle info
    def cg_feature_generator(cls, cg_info, cg_file):
        # mainly collect bead_type, edge_list, bond_type, node_position, cb_token (bead serial number is re-numbered from 1 for each chain), bead2residue
        cg_pdb_info, cg_itp_info, pdb_name = cg_info['cg_pdb_info'], cg_info['cg_itp_info'], os.path.basename(cg_file)

        # * cg_pdb_info may contain extra 'TER' rows to indicate the position of missing residues in a chain *
        # * in current logic, when processing cg_pdb_info, 'TER' rows are ignored, only coordinate information of 'ATOM' is retrieved *
        # keys contained in each protein chain of cg_itp_info:
        # dict_keys(['sequence', 'secondary_structure', 'atom', 'backbone_bonds', 'sidechain_bonds', 'sheet_bonds_3',
        # 'sheet_bonds_4', 'constraints', 'backbone_angles', 'backbone_sidec_angles', 'sidechain_angles', 'backbone_dihedrals'])

        # (1) collect node_position and cb_token (bead serial number is re-numbered from 1 for each chain)
        # use the chain order in the cg pdb file to determine the chain order of the itp file
        # * new version for considering non-consecutive chain id in cg pdb files (e.g., A-B-A rather than chain B is provided after A) *
        # * in this case, we assume that after the re-arrangement, the bead node order of cg pdb is same to that in itp file *
        # get current chain list
        current_chain, chain_list = None, []
        for row in cg_pdb_info:
            if row[0:4] == 'ATOM':
                chainid = row[21]
                if current_chain != chainid:
                    current_chain = chainid
                    # chain_list.append('Protein' + '_' + current_chain)
                    chain_list.append(pdb_name + '-cg' + '_' + current_chain)
        chain_list = sorted(list(set(chain_list)))
        # print(chain_list, cg_itp_info.keys())

        # assign bead counter for each chain
        node_position_, cb_token_list_, local_variable = {chain: [] for chain in chain_list}, {chain: [] for chain in chain_list}, locals()
        for chain in chain_list:
            local_variable['counter' + '_' + chain] = 1

        # retrieve position and chain_bead info for each chain
        for row in cg_pdb_info:
            if row[0:4] == 'ATOM':
                chainid = row[21]
                # chain name in chain_list
                chainid_ = pdb_name + '-cg' + '_' + row[21]
                # every iteration will enter a new bead
                cb_token = '{}_{}'.format(chainid, local_variable['counter' + '_' + chainid_])
                cb_token_list_[chainid_].append(cb_token)
                # node_position is arranged based on the fixed 'BB'+'SC1'+'SC2'+'SC3' order
                node_position_[chainid_].append(get_coords(row))
                local_variable['counter' + '_' + chainid_] += 1

        # re-arrange the node_position and cb_token_list based on the order of chain_list (the retrieval of other info below is also based on the order of chain_list)
        node_position, cb_token_list = [], []
        for chain in chain_list:
            node_position.extend(node_position_[chain])
            cb_token_list.extend(cb_token_list_[chain])

        # * old version for retrieving node position and cb_token lists *
        # current_chain = None
        # node_position, cb_token_list, chain_list = [], [], []
        # for row in cg_pdb_info:
        #     if row[0:4] == 'ATOM':
        #         chainid = row[21]
        #         if current_chain != chainid:
        #             counter1 = 1
        #             current_chain = chainid
        #             chain_list.append(pdb_name + '-cg' + '_' + current_chain)
        #         cb_token = '{}_{}'.format(chainid, counter1)
        #         cb_token_list.append(cb_token)
        #         node_position.append(get_coords(row))
        #         counter1 += 1

        # check the correspondence between the cg pdb file and itp files
        # it seems that if chains in the same original pdb have the same topological structure (the coordinate sets could be different)
        # martini 2.2 will generate one itp file for these chains, our current plan is to duplicate the itp and allocate it to every such chain
        # more specifically, it will occur when two chains have the same AA sequence and secondary structures
        # however, we still need to check the consistency of the chain names provided for cg pdb and itp files
        pdb_chain_set, itp_chain_set = set(chain_list), set(cg_itp_info.keys())
        assert pdb_chain_set == itp_chain_set, "the chain identifiers contained in CG pdb file and itp file are different for {}: {}, {} (should be consistent)".\
            format(pdb_name, pdb_chain_set, itp_chain_set)
        # ** other than the above check, another check for checking the consistency between cg pdb file and itp files is that, **
        # ** the num_node generated here is from itp files, while node_position is from cg pdb file, when creating the CG22_Protein class for current sample, **
        # ** it will trigger its _check_attribute function to check the node numbers contained in both data are the same, if not, an exception will be thrown **

        # (2) collect bead_type, edge_list, bond_type (need to transform the directed edges to undirected edges), bead2residue
        # chain_list: ['Protein_A', 'Protein_D']
        bead_type, edge_list, bead2residue, res_serial_list = [], [], [], []
        chain_bead_num, chain_aa_num, residue_type = [], [], [] # record the bead and aa numbers following the order of current chain_list (can be zipped with the chain_list)
        # * for bead_type: need to collect the bead number of each chain (following the chain_list order), for generating absolute (bead node) idx for current protein *

        # should not be influenced by extra 'TER' as currently processing the re-numbered itp info
        for chain in chain_list:
            chain_bead_info = cg_itp_info[chain]['atom']
            chain_aa_info = cg_itp_info[chain]['sequence']

            chain_bead_num.append(len(chain_bead_info))
            chain_aa_num.append(len(chain_aa_info))
            residue_type.append(chain_aa_info)

            current_res_serial = None
            for row in chain_bead_info:
                row = row.split() # each row represents a new bead
                # bead name, residue serial number (re-numbered from 1), residue name, bead position category (indicating BB/SC1/SC2/SC3): 12, 1, 4, 0
                bead, res_serial, res, bead_pos = cls.martini22_name2id[row[1]], int(row[2]), cls.residue2id[row[3]], cls.martini22_beadpos2id[row[4]]
                if current_res_serial != res_serial:
                    current_res_serial = res_serial
                    res_serial_list.append(res_serial) # record the aa serial numbers of each chain following the order of chain_list
                bead2residue.append(len(res_serial_list) - 1)
                bead_type.append([bead, res, bead_pos]) # transform bead type and corresponding residue type and bead position category into idx
        # print(chain_bead_num, chain_aa_num) # [248, 191] [108, 87]

        chain_bead_cumnum = np.cumsum([0] + chain_bead_num[:-1]) # [0 248]
        # * for edge_list: the absolute (bead node) idx for current protein is used for facilitating identifying the allocation of each bead node to corresponding residue *
        bond_keys = list(cls.martini22_bond2id.keys()) # ['backbone_bonds', 'sidechain_bonds', 'sheet_bonds_3', 'sheet_bonds_4', 'constraints']
        backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals = [], [], [], []
        angle_keys = list(cls.martini22_angletype2id) # ['backbone_angles', 'backbone_sidec_angles', 'sidechain_angles', 'backbone_dihedrals']

        for chain_id, chain in enumerate(chain_list): # get one chain itp
            chain_itp_info = cg_itp_info[chain]
            cum_bead_id = chain_bead_cumnum[chain_id] # cumulative bead serial number for current chain
            for bond_key in bond_keys: # get one type of bond for current chain
                rows = chain_itp_info[bond_key] # get rows for current type of bond
                current_type = cls.martini22_bond2id[bond_key] # current bond type
                for row in rows:
                    row = row.split()
                    h, t = int(row[0]) + cum_bead_id - 1, int(row[1]) + cum_bead_id - 1 # make edge_list index starting from 0
                    edge_list += [[h, t, current_type], [t, h, current_type]]

            for angle_key in angle_keys:
                rows = chain_itp_info[angle_key]
                for row in rows:
                    row = row.split()
                    if 'dihedral' in angle_key:
                        # print(pdb_name, chain, row)
                        _1, _2, _3, _4 = int(row[0])+cum_bead_id-1, int(row[1])+cum_bead_id-1, int(row[2])+cum_bead_id-1, int(row[3])+cum_bead_id-1 # make angle node index start from 0
                        locals()[angle_key] += [[_1, _2, _3, _4]]
                    else:
                        _1, _2, _3= int(row[0])+cum_bead_id-1, int(row[1])+cum_bead_id-1, int(row[2])+cum_bead_id-1
                        locals()[angle_key] += [[_1, _2, _3]]

        assert len(edge_list), "edge information provided in itp files of protein {} is empty".format(pdb_name)
        edge_list = torch.tensor(sorted(edge_list)) # sorted: fix the edge_list order
        bond_type = torch.tensor(edge_list)[:, -1]
        bead_type = torch.tensor(bead_type)
        bead2residue = torch.tensor(bead2residue)

        # num_relation records edge types in original itp files (based on the definition of martini 2.2), not including the potential extra edge types which will be defined in graph models
        num_node, num_relation = sum(chain_bead_num), len(cls.martini22_bond2id)
        aa_sequence = '.'.join(residue_type)
        residue_type = torch.tensor([cls.residue_symbol2id[i] for chain in residue_type for i in chain])

        node_position = np.array(node_position)
        backbone_angles = torch.tensor(backbone_angles)
        backbone_sidec_angles = torch.tensor(backbone_sidec_angles)
        sidechain_angles = torch.tensor(sidechain_angles)
        backbone_dihedrals = torch.tensor(backbone_dihedrals)

        # a final check for the bead number correspondence of each chain between cg pdb and itp
        # chain_bead_num comes from itp files, cb_token_list_ comes form cg pdb files
        chain_bead_num_ = [len(cb_token_list_[chain]) for chain in chain_list]
        assert chain_bead_num_ == chain_bead_num, \
            "the bead number for each chain between CG pdb and itp files is different for {}, the number in itp files is {}, while the number in CG pdb is {}".\
                format(pdb_name, chain_bead_num, chain_bead_num_)

        return edge_list, bead_type, bead2residue, node_position, bond_type, num_node, num_relation, residue_type, aa_sequence, \
               backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals

    def protein_cropping(self, cropping_threshold=12, contact_threshold=6, compact=True):
        # this function is performed prior to the 'transform' functions
        aa_num_chain = [len(chain) for chain in self.aa_sequence.split('.')]
        assert self.residue_type.size(0) == sum(aa_num_chain) == int(self.num_residue), \
            "the residue number in residue_type and aa_sequence should be the same"

        # currently only chain number == 2 in a protein is supported (the binding affinity is measured between these two chains)
        if len(aa_num_chain) == 2:
            # get the BB positions of each chain as the positions of corresponding residues
            BB_mask = (self.atom_type[:, 2] == self.martini22_beadpos2id['BB'])
            BB_position = self.node_position[BB_mask].unsqueeze(0)

            # (1) square_distance input (support batch calculation): src: source points, [B, N, C], dst: target points, [B, M, C]
            # output: a symmetric matrix -> 195 * 195 (195 is total residue/BB number in current protein)
            # ** i.e., use the inter-BB distance as the distance between pairwise AAs (the output is the squared distance) **
            aa_square_distance = square_distance(BB_position, BB_position).squeeze(0)

            # (2) print(np.cumsum([0] + aa_num_chain), aa_num_chain) # [0, 108, 195], [108, 87]
            # in the case of two chains, only first sub-matrix of aa_square_distance is used
            contact_matrix = aa_square_distance[:aa_num_chain[0], aa_num_chain[0]:]

            # get the closest distance within the contact matrix (for later distance check)
            closest_distance = torch.sqrt(torch.min(contact_matrix))

            contact_matrix = (contact_matrix <= contact_threshold ** 2) # 8.5A**2
            contact_matrix = torch.nonzero(contact_matrix)
            # * current contact matrix contains AA pairs retained for the core region (relative id, first row: AA id in part1, second row: AA id in part 2) *
            # * only containing the inter-molecular relationships between the two parts (based on pre-defined contact_threshold) *

            contact_matrix[:, 1] = contact_matrix[:, 1] + aa_num_chain[0] # transform relative AA ids to absolute ids (in current protein)
            contact_aa_index = torch.cat([contact_matrix[:, 0], contact_matrix[:, 1]])
            # tensor([34, 35, 99, 138, 150, 151, 152])
            contact_aa_index = torch.unique(contact_aa_index, sorted=True) # remove the duplicated AA indices

            # (3) based on the found contact AAs and cropping threshold to retain proteins
            # any AAs having distance to contact AAs smaller than cropping threshold will be retained
            # the distance for retaining these AAs is also based on aforementioned inter-BB distance
            # ** in every selected aa_square_distance[contact_aa_index] row, the position representing current AA must be zero (identity itself) **
            # ** in this case, all contact AAs calculated above will be retained during residue_mask-based graph cropping **
            retain_matrix = (aa_square_distance[contact_aa_index] <= cropping_threshold ** 2)
            # ** torch.Size([7, 195]), for its each row/AA, the position for current AA in this row should be True (diagonality of aa_square_distance) **
            # ** therefore torch.nonzero(retain_matrix)[:, 0] is not needed here **

            # print('the maximum distance within the distance matrix of selected contact AAs:',
            #       torch.max(torch.sqrt(torch.clamp(aa_square_distance[contact_aa_index], min=0))))
            # maximum: 58.5827 in the subset of ATLAS (based on contact_threshold == 8.5A)

            retain_aa_index = torch.unique(torch.nonzero(retain_matrix)[:, 1], sorted=True) # ** get the absolute ids for the retained AAs **
            # * current contact_matrix: the matrix only containing the inter-molecular AA pairs based on pre-defined contact_threshold *
            # * the contact_threshold can be adjusted, which is not directly related to vdw and electrostatic CG cutoff *
            return self.residue_mask(retain_aa_index, compact=compact, intermol_mat=contact_matrix), closest_distance

        else:
            print('current protein cropping does not support proteins with chain number over two')
            raise NotImplementedError

    def residue_mask(self, index, compact=False, intermol_mat=None):
        """
        Return a masked protein based on the specified residues.
        Note the compact option is applied to both residue and atom ids.
        Parameters:
            index (array_like): residue index: mask[start:end] = True
            start and end represent the end point AAs to be retained
            compact (bool, optional): compact residue ids or not
        Returns:
            Protein
        """
        index = self._standarize_index(index, self.num_residue)
        # transform the boolean mask to the consecutive absolute AA serial numbers of AAs to be retained
        # tensor([56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69)]
        if (torch.diff(index) <= 0).any():
            warnings.warn("`residue_mask()` is called to re-order the residues. This will change the protein sequence."
                          "If this is not desired, you might have passed a wrong index to this function.")

        residue_mapping = -torch.ones(self.num_residue, dtype=torch.long, device=self.device)
        residue_mapping[index] = torch.arange(len(index), device=self.device)
        # 1. get a mapping for residue absolute ids with the size of self.num_residue
        # in which ids for retained AAs are compact new absolute ids, and -1 for removed AAs

        # residue_mapping (above): residue number size, self.bead2residue: bead node size
        # thus, node_index is to allocate residue mask to each bead node
        node_index = residue_mapping[self.bead2residue] >= 0
        # transform the bead-level mask into real-value-based mask (indicating the positions of bead nodes to be retained based on retained AAs)
        node_index = self._standarize_index(node_index, self.num_node) # the node-based mask
        # 2. after self._standarize_index, node_index contains remained bead nodes with original node absolute ids

        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            # * generating the mapping between original sparse bead node indices and the compact ones *
            # * changing the indices of bead nodes *
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_node = len(node_index)
        else:
            mapping[node_index] = node_index
            num_node = self.num_node
        # 3. if compact=True, generating a bead node mapping which projects from original node absolute ids to new compact absolute ids

        # compact mapping is tensor with the size of bead number, in which masked beads are represented as -1, while retained beads are recorded as consecutive int re-numbered from 0
        edge_list = self.edge_list.clone() # edge_list is based on the bead node number starting from 0, torch.Size([1014, 3])
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1) # torch.Size([1014])
        edge_index = self._standarize_index(edge_index, self.num_edge) # remove edges with '-1' nodes
        # edge_index: the indices retrieving edges without '-1' nodes from original edge_list
        # tensor([272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285])
        # 4. current edge index: the node ids are updated while invalid edges are not removed
        # current edge_index: contain ids for retrieving valid edges from edge_index

        if compact:
            # node_index contains ids for retrieving retained bead nodes from all nodes: torch.Size([237])
            # edge_index contains ids for retrieving retained edges from all edges: torch.Size([544])
            # index contains ids for retrieving retained AAs from all AAs: torch.Size([100])
            data_dict, meta_dict = self.data_mask(node_index, edge_index, residue_index=index) # residue_index: for retrieving information related to remained AAs
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        # truncate the angle information
        backbone_angles = self.angle_mapping(self.backbone_angles, mapping)
        backbone_sidec_angles = self.angle_mapping(self.backbone_sidec_angles, mapping)
        sidechain_angles = self.angle_mapping(self.sidechain_angles, mapping)
        backbone_dihedrals = self.angle_mapping(self.backbone_dihedrals, mapping)

        # retrieving the required information for initialing an on-the-fly truncated protein class
        bead_type, bead2residue, node_position, bond_type, residue_type = \
            data_dict['atom_type'], data_dict['bead2residue'], data_dict['node_position'], data_dict['bond_type'], data_dict['residue_type']

        # do not need to return core region AA pair information
        if intermol_mat == None:
            # edge_list has already been updated based on new relative node ids after above mapping function, num_node is also updated here
            return type(self)(edge_list[edge_index], bead_type=bead_type, bead2residue=bead2residue, node_position=node_position,
                   bond_type=bond_type, num_node=num_node, num_relation=self.num_relation, residue_type=residue_type, view=self.view,
                   backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles, sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals)
        # otherwise
        else:
            # all elements in intermol_mat should in elements in index (to ensure -1 will not occur at residue_mapping[intermol_mat])
            assert torch.all(torch.isin(intermol_mat, index)), 'All elements in intermol_mat should in elements of the index input of this cropping function.'
            if compact:
                # the retained AAs do not only include AAs in core region, also including AAs searched by cropping_threshold
                intermol_mat = residue_mapping[intermol_mat] # mapping the elements in intermol_mat into the new compact version after the cropping

            return type(self)(edge_list[edge_index], bead_type=bead_type, bead2residue=bead2residue, node_position=node_position,
                   bond_type=bond_type, num_node=num_node, num_relation=self.num_relation, residue_type=residue_type, view=self.view, backbone_angles=backbone_angles,
                   backbone_sidec_angles=backbone_sidec_angles, sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals, intermol_mat=intermol_mat)

    def angle_mapping(self, angles, mapping):
        if angles.size(0) > 0:
            angle_info = angles.clone()
            angle_info = mapping[angle_info]
            angle_index = (angle_info >= 0).all(dim=-1)
            angle_index = self._standarize_index(angle_index, angles.size(0))
            return angle_info[angle_index]
        else:
            return angles.clone()

    def data_mask(self, node_index=None, edge_index=None, residue_index=None, graph_index=None, include=None, exclude=None):
        data_dict, meta_dict = super(CG22_Protein, self).data_mask(node_index, edge_index, graph_index=graph_index, include=include, exclude=exclude)
        # Note:
        # (1) inherited data_mask in graph class mainly handles 'node', 'edge', 'graph' attributes registered with context manager ('residue'-related ones are handled below)
        # meta_dict: {'atom_type': {'node'}, 'formal_charge': {'node'}, 'explicit_hs': {'node'}, 'chiral_tag': {'node'}, 'radical_electrons': {'node'},
        # 'atom_map': {'node'}, 'node_position': {'node'}, 'bond_type': {'edge'}, 'bond_stereo': {'edge'}, 'stereo_atoms': {'edge'},
        # 'bead2residue': {'node', 'residue reference'}, 'residue_type': {'residue'}}

        # (2) information of protein class initialization:
        # cls(edge_list, bead_type=bead_type, bead2residue=bead2residue, node_position=node_position,
        #    bond_type=bond_type, num_node=num_node, num_relation=num_relation, residue_type=residue_type, aa_sequence=aa_sequence)

        # (3) the following attributes of (class initialization) features are registered using the context manager of protein class or molecule class
        # thus, we can understand that what data_dict/meta_dict returns are the information registered with the context manager in protein/molecule class
        # besides, we can find that unless we do not register atom-level features like explicit_hs in molecule class (i.e., modify this class),
        # these features will also be retrieved here by molecule.data_mask function

        # (4) check attributes in meta_dict:
        # 1. atom_type/bead_type: {'node'}, node_index 2. bead2residue: {'node', 'residue reference'}, node_index, need further check below 3. node_position: {'node'}, node_index
        # 4. bond_type: {'edge'}, edge_index 5. residue_type: {'residue'}, need further check below
        # additional check other than the attributes in meta_dict: 1. edge_list 2. num_node 3. num_relation
        residue_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "residue" and residue_index is not None:
                    if v.is_sparse:
                        v = v.to_dense()[residue_index].to_sparse()
                    else:
                        v = v[residue_index]
                elif type == "residue reference" and residue_index is not None:
                    if residue_mapping is None:
                        # residue_mapping is a tensor with the shape of num_residue + 1
                        # in which the values not equalling to -1 indicate the residues to be retained
                        # and these values are re-numbered from 0 consecutively for being retrieved by 'v' below
                        # due to the constraint from residue_index, elements in 'v' can only retrieve values not equalling to -1 from residue_mapping
                        residue_mapping = self._get_mapping(residue_index, self.num_residue)
                    # re-mapp/re-number the ids in bead2residue starting from 0 consecutively
                    v = residue_mapping[v]
            data_dict[k] = v

        # for aa_sequence, in current logic it is not used in the training process (only used in recording AA sequence for pickle storage)
        # therefore in current sub-graph truncation process, this information will not be passed into truncated proteins
        # aa_sequence = self.aa_sequence.replace('.', '')
        # aa_sequence = ''.join((np.array(list(aa_sequence))[residue_index]).tolist())
        return data_dict, meta_dict

    # return a subgraph based on the specified residues
    def subresidue(self, index):
        return self.residue_mask(index, compact=True)

    @classmethod
    # ** calling the below CG22_PackedProtein to create the batch protein graph (defined in the last row of this script) **
    # ** CG22_Protein.packed_type = CG22_PackedProtein (packed_type is what this function return in the end) **
    def pack(cls, graphs):
        # * adding the record of residue number and view information *
        # * adding the support of bead2residue cumulative sum *
        # * adding the support of angle information from CG itp files *
        # * adding the support of intermolcular matrix information from the AA-based distance matric from the cropping function *
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_residues = []
        backbone_angles = []
        backbone_sidec_angles = []
        sidechain_angles = []
        backbone_dihedrals = []
        intermol_mat = []
        # pack the information the input graphs
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_residue = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        view = graphs[0].view
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_residues.append(graph.num_residue)
            backbone_angles.append(graph.backbone_angles + num_cum_node)
            backbone_sidec_angles.append(graph.backbone_sidec_angles + num_cum_node)
            sidechain_angles.append(graph.sidechain_angles + num_cum_node)
            backbone_dihedrals.append(graph.backbone_dihedrals + num_cum_node)
            # ** note that the incremental information for angles is num_cum_node (bead node-based) **
            # ** while that for intermolecular matrix is num_cum_residue (because the distance calculation is AA-based) **
            intermol_mat.append(graph.intermol_mat + num_cum_residue)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = torch.where(v != -1, v + num_cum_node, -1)
                    elif type == "edge reference":
                        v = torch.where(v != -1, v + num_cum_edge, -1)
                    elif type == "residue reference":
                        # bead2residue cumulative sum
                        v = torch.where(v != -1, v + num_cum_residue, -1)
                    elif type == "graph reference":
                        v = torch.where(v != -1, v + num_graph, -1)
                data_dict[k].append(v)
            num_cum_node += graph.num_node # one value
            num_cum_edge += graph.num_edge # one value
            num_cum_residue += graph.num_residue # one value
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        backbone_angles = torch.cat(backbone_angles)
        backbone_sidec_angles = torch.cat(backbone_sidec_angles)
        sidechain_angles = torch.cat(sidechain_angles)
        backbone_dihedrals = torch.cat(backbone_dihedrals)
        intermol_mat = torch.cat(intermol_mat)

        # data_dict.keys: dict_keys(['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons',
        # 'atom_map', 'node_position', 'bond_type', 'bond_stereo', 'stereo_atoms', 'bead2residue', 'residue_type'])
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(
            edge_list, edge_weight=edge_weight, num_relation=graphs[0].num_relation, num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=view,
            backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles, sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals,
            intermol_mat=intermol_mat, meta_dict=meta_dict, **data_dict)

    def __repr__(self):
        fields = ["num_atom=%d" % self.num_node, "num_bond=%d" % self.num_edge,
                  "num_residue=%d" % self.num_residue]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2, dim=-1) + sum(dst**2, dim=-1) - 2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# * Important Note: for current logic, if over-large proteins are found, this protein will be ignored (returned with the specific identifier) *
# * besides, currently we do not need to consider non-natural AAs/incomplete proteins/AALA+BALA cases, as these cases will be handled in the pre-processed steps earlier than here *
def cleaning_cg_pdb(cglines, pdb, AA_num_threshold=3000):
    # the residue serial numbers in CG pdbs are based on those in original pdb files, which may not be consecutive
    # while the residue serial numbers in each itp file are re-numbered from 1 (consecutive)
    # print('current processed pdb name: {}'.format(pdb))

    screened_list = [] # the list containing the cleaned CG lines
    bead_pos_list = [] # for recording the bead list for current CG file
    BB_num_threshold = AA_num_threshold # threshold for screening out over-large proteins

    for row in cglines:
        resname = row[17:20].strip() # residue name

        if row[0:4] == 'ATOM' and len(resname) == 3:
            # a temporary correcting of CG proteins (i.e., chain a/d to chain B)
            # if row[21].islower():
            #     row = row[:21] + 'B' + row[22:]
            screened_list.append(row)
            bead_pos_list.append(row[12:16].strip())
        # a check about the non-residue atoms
        elif row[0:4] == 'ATOM' and len(resname) != 3:
            print('Non-residue atoms:', pdb, row)
        elif row[0:3] == 'TER':
            screened_list.append(row[0:3] + '\n')
        else:
            continue

    # over-large protein check
    BB_num = np.sum(np.array(bead_pos_list) == 'BB') # each residue only has one BB bead
    if BB_num > BB_num_threshold:
        return False, 'Over-large protein {} is ignored'.format(pdb)

    return True, screened_list


def cleaning_cg_itp(chain_lines, pdb):
    # ** in some cases, sheet_bonds_3 and sheet_bonds_4 will not exist in itp along with the tag row, **
    # ** for backbone dihedrals, sometimes the rows will not exist but the tag is remained, which needs to be considered **

    complete_check = True
    chain_dict = dict()
    tag = None
    # Note: within the types martini22_bond2id = {'backbone_bonds': 0, 'sidechain_bonds': 1, 'sheet_bonds_3': 2, 'sheet_bonds_4': 3, 'constraints': 4}
    # only type 0 will definitely exist for each protein in itp bond types
    # thus, we also need to consider the case that a part of bond types does not exist in itp files
    backbone_bonds, sidechain_bonds, sheet_bonds_3, sheet_bonds_4, constraints = [], [], [], [], []
    backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals = [], [], [], []

    for row_num, row in enumerate(chain_lines):
        # ** with current logic, each 'elif' branch will not conflict with each other **
        if row_num != len(chain_lines) - 1:
            next_row = chain_lines[row_num + 1].strip() # last row of itp file: '#endif'
        else: # the last row of current itp file
            next_row = ''
            # further check about the '#endif' tag for finding itp files which may be incomplete (should end with '#endif' tag)
            if row.strip() != '#endif':
                print('current itp files of protein {} may be incomplete'.format(pdb))
                complete_check = False

        # recording AA sequence
        if row.strip() == '; Sequence:':
            chain_dict['sequence'] = next_row[2:]

        # recording secondary structure
        elif row.strip() == '; Secondary Structure:':
            chain_dict['secondary_structure'] = next_row[2:]

        # recording CG beads
        elif row.strip() == '[ atoms ]':
            aa_record_tag = False
            tag = 'atom'
            atom = []
            if 'sequence' not in chain_dict.keys():
                aa_record_tag = True
                current_resid = None
                aa_sequence = []
        # considering the case that 'sequence' does not exist in itp files
        elif tag == 'atom':
            if next_row == '':
                row = row.strip()
                atom.append(row)
                tag = None
                # recording AA types if they are not provided in itp file
                if aa_record_tag == True:
                    row = row.split()
                    res_id, res_name = row[2], row[3]
                    if res_id != current_resid:
                        current_resid = res_id
                        aa_sequence.append(residue_symbol2abbr[res_name])
                    # this is the end of the 'atom' rows
                    chain_dict['sequence'] = ''.join(aa_sequence)
            else:
                row = row.strip()
                atom.append(row)
                # recording AA types if they are not provided in itp file
                if aa_record_tag == True:
                    row = row.split()
                    res_id, res_name = row[2], row[3]
                    if res_id != current_resid:
                        current_resid = res_id
                        aa_sequence.append(residue_symbol2abbr[res_name])

        # recording backbone bonds with flexible bond length (defined by martini22_aminoacid.itp)
        elif row.strip() == '; Backbone bonds':
            # skip the case that the topological tag exists but no content exists
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_bonds'
        elif tag == 'backbone_bonds':
            # if chain_lines[row_num+1].strip() == '; Sidechain bonds':
            # 1. (len(next_row) > 0 and next_row[0] == ';') for handling cases like '; Sidechain bonds'
            # 2. next_row == '' for handling empty row cases
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_bonds.append(row)
                tag = None
            else:
                backbone_bonds.append(row.strip())

        # recording bonds between backbone and side chain with flexible bond length (defined by martini22_aminoacid.itp)
        elif row.strip() == '; Sidechain bonds':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sidechain_bonds'
        elif tag == 'sidechain_bonds':
            # if chain_lines[row_num+1].strip() == '; Short elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sidechain_bonds.append(row)
                tag = None
            else:
                sidechain_bonds.append(row.strip())

        # recording virtual bonds for sheet secondary structure (based on three AA distance)
        elif row.strip() == '; Short elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sheet_bonds_3'
        elif tag == 'sheet_bonds_3':
            # if chain_lines[row_num+1].strip() == '; Long elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sheet_bonds_3.append(row)
                tag = None
            else:
                sheet_bonds_3.append(row.strip())

        # recording virtual bonds for sheet secondary structure (based on four AA distance)
        elif row.strip() == '; Long elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sheet_bonds_4'
        elif tag == 'sheet_bonds_4':
            # if chain_lines[row_num+1].strip() == '':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sheet_bonds_4.append(row)
                tag = None
            else:
                sheet_bonds_4.append(row.strip())

        # recording bonds between backbones, between backbone and side chain, and between side chains with fixed bond length (defined by martini22_aminoacid.itp)
        # for some types of residues, the 'constraints' could be empty, thus we also need to consider the case that the 'constraints' could be empty
        elif row.strip() == '[ constraints ]':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'constraints'
        elif tag == 'constraints':
            # if chain_lines[row_num+1].strip() == '':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                constraints.append(row)
                tag = None
            else:
                constraints.append(row.strip())

        # recording backbone angles
        elif row.strip() == '; Backbone angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_angles'
        elif tag == 'backbone_angles':
            # if chain_lines[row_num+1].strip() == '; Backbone-sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_angles.append(row)
                tag = None
            else:
                backbone_angles.append(row.strip())

        # recording backbone-sidechain angles
        elif row.strip() == '; Backbone-sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_sidec_angles'
        elif tag == 'backbone_sidec_angles':
            # if chain_lines[row_num+1].strip() == '; Sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_sidec_angles.append(row)
                tag = None
            else:
                backbone_sidec_angles.append(row.strip())

        # recording sidechain angles
        elif row.strip() == '; Sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sidechain_angles'
        elif tag == 'sidechain_angles':
            # if chain_lines[row_num+1].strip() == '':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sidechain_angles.append(row)
                tag = None
            else:
                sidechain_angles.append(row.strip())

        # recording backbone dihedrals
        # ** the side chain diredrals are not necessarily recorded, as only side chains with aromatic nucleus contain at least three side chain beads so that this dihedral can be calculated **
        # ** however, the aromatic nucleus side chain is the plane structure with this dehedral value 0, which cannot be used for distinguishing (the types of) each other **
        elif row.strip() == '; Backbone dihedrals':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_dihedrals'
        elif tag == 'backbone_dihedrals':
            # if chain_lines[row_num+1].strip() == '; Sidechain improper dihedrals':
            # print(len(next_row), next_row, next_row[0])
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_dihedrals.append(row)
                tag = None
            else:
                backbone_dihedrals.append(row.strip())

        else:
            continue

    # print(atom[0], '///' , atom[-1], '///' ,len(atom))
    # print(backbone_bonds[0], '///' ,backbone_bonds[-1], '///' ,len(backbone_bonds))
    # print(sidechain_bonds[0], '///', sidechain_bonds[-1], '///', len(sidechain_bonds))
    # print(sheet_bonds_3[0] if len(sheet_bonds_3) > 0 else sheet_bonds_3, '///', sheet_bonds_3[-1] if len(sheet_bonds_3) > 0 else sheet_bonds_3, '///', len(sheet_bonds_3))
    # print(sheet_bonds_4[0] if len(sheet_bonds_4) > 0 else sheet_bonds_4, '///', sheet_bonds_4[-1] if len(sheet_bonds_4) > 0 else sheet_bonds_4, '///', len(sheet_bonds_4))
    # print(constraints[0], '///', constraints[-1], '///', len(constraints))
    # print(backbone_angles[0], '///', backbone_angles[-1], '///', len(backbone_angles))
    # print(backbone_sidec_angles[0], '///', backbone_sidec_angles[-1], '///', len(backbone_sidec_angles))
    # print(sidechain_angles[0], '///', sidechain_angles[-1], '///', len(sidechain_angles))
    # print(backbone_dihedrals[0], '///', backbone_dihedrals[-1], '///', len(backbone_dihedrals))

    # (1) backbone_angles: BBB (2nd as center_pos, B)
    # (2) backbone_sidec_angles: BBS (3rd as center_pos, S)
    # (3) sidechain_angles: BSS (3rd as center_pos, S)
    # (4) backbone_dihedrals: BBBB (2nd as center_pos, B), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure
    complete_check = (len(backbone_bonds) != 0) & (len(sidechain_bonds) != 0) & (len(backbone_angles) != 0) & (len(backbone_sidec_angles) != 0) & complete_check
    # (1) 'constraints' could be empty if some proteins are lack of specific kinds of residues
    # (2) 'backbone_bonds' could also be empty, in the case like residues of current whole chain belong the helix structure (these backbone bonds will be assigned to 'constraints')
    # (3) specifically, 'bonds' and 'constraints' can both be treated as the representation of bond length
    # (4) the bond length of backbone bonds will be assigned to 'backbone_bonds' or 'constraints' based on the secondary structure:
    # if one of two beads within a backbone bond is identified as the helix structure, this bond will be 'constraints' otherwise be 'backbone_bonds'
    # (5) the bond length of side chain bonds will be assigned to 'sidechain_bonds' or 'constraints' according to the AA type:
    # the specific definition of side chain bond assignment is based on martini22_aminoacid.itp file
    # (6) the bond length refers to the geometric distance between two beads

    # beads and bonds
    chain_dict['atom'],chain_dict['backbone_bonds'], chain_dict['sidechain_bonds'], chain_dict['sheet_bonds_3'], chain_dict['sheet_bonds_4'], chain_dict['constraints'] = \
        atom, backbone_bonds, sidechain_bonds, sheet_bonds_3, sheet_bonds_4, constraints
    # angles and dihedrals
    chain_dict['backbone_angles'],chain_dict['backbone_sidec_angles'], chain_dict['sidechain_angles'], chain_dict['backbone_dihedrals'] = \
        backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals

    return complete_check, chain_dict


def get_coords(line):
    if len(line[30:38].strip()) != 0:
        x = float(line[30:38].strip())
    else:
        x = float('nan')  # nan in math format
    if len(line[38:46].strip()) != 0:
        y = float(line[38:46].strip())
    else:
        y = float('nan')
    if len(line[46:54].strip()) != 0:
        z = float(line[46:54].strip())
    else:
        z = float('nan')
    return x, y, z


# generate edge features for token edges
# the input attributes of 'token_edge_cg22_gearnet' should be based on abs (node) ids in current batch
def token_edge_cg22_gearnet(edge_list, bead2residue, node_position, node_type_num, num_relation):
    max_seq_dist = 10 # same to that in 'edge_cg22_gearnet' function, to keep the consistency of corresponding one-hot edge features

    # note that current edge feature dimension is already fixed before token injection
    # in other words, edge feature contains transductive one-hot encoding information, which cannot be extended once determined
    edge_num = edge_list.size(0)
    node_in, node_out, r = edge_list.t() # target node, source node, relation type

    # use zero-encoding to avoid conflict with one-hot original bead type encoding
    target_node_encoding, source_node_encoding = torch.zeros(edge_num, node_type_num, device=device), torch.zeros(edge_num, node_type_num, device=device)

    # calculate residue sequence distance between token nodes and bead nodes (based on abs node ids in the updated bead2residue)
    # updated bead2residue represents it already includes tokens as the residue nodes
    residue_in, residue_out = bead2residue[node_in], bead2residue[node_out]
    sequential_dist = torch.abs(residue_in - residue_out)

    # the spatial distance could be larger that in 'edge_cg22_gearnet', as the token nodes/residues are put at the begining of each graph fixedly
    # example: 26.7500 (token edge distance) VS 3.3320 (bead edge distance)
    spatial_dist = (node_position[node_in] - node_position[node_out]).norm(dim=-1)

    return torch.cat([
        target_node_encoding,
        source_node_encoding,
        functional.one_hot(r, num_relation),
        functional.one_hot(sequential_dist.clamp(max=max_seq_dist), max_seq_dist + 1), # 0-10, 11 in total
        spatial_dist.unsqueeze(-1)
    ], dim=-1)


# the class for combining multiple CG_Protein objects into one batch-graph object
# Note: functions which are not utilized in current CGDiff implementation are incomplete for processing CG22_Protein objects
# which will be further updated in the future, e.g., the detach and clone functions, needing the extra support of handling angle info
class CG22_PackedProtein(PackedMolecule, CG22_Protein):
    """
    Container for proteins with variadic sizes.
    Support both residue-level and atom-level operations and ensure consistency between two views.
    .. warning::
        Edges of the same graph are guaranteed to be consecutive in the edge list.
        The order of residues must be the same as the protein sequence.
        However, this class doesn't enforce any order on nodes or edges.
        Nodes may have a different order with residues.
    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        residue_type (array_like, optional): residue types of shape :math:`(|V_{res}|,)`
        view (str, optional): default view for this protein. Can be ``atom`` or ``residue``.
        num_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edge_list`
        num_edges (array_like, optional): number of edges in each graph
        num_residues (array_like, optional): number of residues in each graph
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
            If not provided, nodes in `edge_list` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edge_list` should be absolute index, i.e., the index in the packed graph.
    """

    unpacked_type = CG22_Protein
    _check_attribute = CG22_Protein._check_attribute

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, residue_type=None, view=None, num_nodes=None,
                 num_edges=None, num_residues=None, offsets=None, **kwargs):

        # ** the kwargs hyperparameters (which could include angle information) will be sent to all parent class (i.e., PackedMolecule, CG22_Protein) via the initialization function **
        # ** thus, current CG22_Protein will receive the angle information (and potential intermolcular matrix) of the whole batch now (but not re-numbered for the whole batch yet) **
        super(CG22_PackedProtein, self).__init__(edge_list=edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                            offsets=offsets, atom_type=atom_type, bond_type=bond_type,
                                            residue_type=residue_type, view=view, **kwargs)

        num_residues = torch.as_tensor(num_residues, device=self.device)
        num_cum_residues = num_residues.cumsum(0)

        self.num_residues = num_residues
        self.num_cum_residues = num_cum_residues

    @property
    def num_nodes(self):
        return self.num_atoms

    @num_nodes.setter
    def num_nodes(self, value):
        self.num_atoms = value

    # ** in current implementation, it will be used for edge_mask in CGDistancePrediction (graph.edge_mask(~functional.as_mask(indices, graph.num_edge))) **
    # ** in this case, the angular node features have not been generated (should be after this 'data_mask' function), thus angular feature alignment is not considered here **
    # ** besides, it will also be used in 'node_mask' function below **
    # data_dict, meta_dict = self.data_mask(edge_index=index), index contains edge indices to be retained
    def data_mask(self, node_index=None, edge_index=None, residue_index=None, graph_index=None, include=None, exclude=None):
        # for handling standard registered features (not including unregistered features like angular information)
        data_dict, meta_dict = super(CG22_PackedProtein, self).data_mask(node_index, edge_index, graph_index=graph_index, include=include, exclude=exclude)
        # print(data_dict.keys())
        # dict_keys(['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons', 'atom_map', 'node_position', 'bead2residue',
        # 'residue_type', 'atom_feature', 'edge_feature', 'bond_feature', 'bond_type', 'bond_stereo', 'stereo_atoms'])

        residue_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "residue" and residue_index is not None:
                    if v.is_sparse:
                        v = v.to_dense()[residue_index].to_sparse()
                    else:
                        v = v[residue_index]
                # residue_index=None: skip the re-numbering function
                elif type == "residue reference" and residue_index is not None:
                    if residue_mapping is None:
                        residue_mapping = self._get_mapping(residue_index, self.num_residue)

                    v = residue_mapping[v]
            data_dict[k] = v

        return data_dict, meta_dict

    # * in current implementation, it will be used for graph truncation in forward sequence diffusion process (graph1 = graph1.subgraph(node_mask)) *
    # * while the intermol_mat is only used (in energy decoder) for downstream tasks assuming no further graph cropping in downstream training *
    # * thus here directly return the original intermol_mat (meanwhile if seq_bb_retain==True, it is equivalent to not further AA cropping since the BB beads of masked AAs are retained) *
    # input: boolean mask for unmasked/remained CG nodes in current batch (True for remained bead nodes)
    # in current logic, we assume seq_bb_retain is set to True by default, under which we need to remove all angle information related to masked side chain beads
    def node_mask(self, index, compact=True):
        # _standarize_index returns a tensor giving 'True' positions under the size of num_node
        index = self._standarize_index(index, self.num_node) # not consecutive
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)

        if compact:
            # mapping is the mask+mapping function for retained nodes, masked nodes are -1, while retained nodes are re-numbered from 0
            mapping[index] = torch.arange(len(index), device=self.device)
            num_nodes = self._get_num_xs(index, self.num_cum_nodes) # retained bead node number for each protein
            # print(self.num_nodes, num_nodes) # tensor([237, 120, 231, 120]) tensor([183, 80, 150, 90])
            offsets = self._get_offsets(num_nodes, self.num_edges)
            # get the offset (based on cumulative node numbers) for each edge in batch-wise edge_list
            # print(offsets) # tensor([0, 0, 0, ..., 412, 412, 412])
        else:
            mapping[index] = index
            num_nodes = self.num_nodes
            offsets = self._offsets

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]] # the node id in edge_list has already been transformed into the new ids in mapping function
        # edge index is generated based on aforementioned mapping function (if any of end nodes in an edge are labeled to -1, this edge will be totally masked)
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)

        # print(edge_index, edge_index.size(), self.num_cum_nodes)
        # tensor([False,  True, False,  ..., False, False, False]) torch.Size([1532]) tensor([237, 357, 588, 708])
        num_edges = self._get_num_xs(edge_index, self.num_cum_edges)
        # print(num_edges) # tensor([418, 158, 316, 180]) -> sum: 1072, extracted from original 1532 edges

        if compact:
            data_dict, meta_dict = self.data_mask(index, edge_index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        # start to truncate the angle information, by default, sequence diffusion will only mask side chain beads
        # while the backbone nodes may also be masked
        # print(self.backbone_angles.size(), self.sidechain_angles.size()) # torch.Size([308, 3]), torch.Size([108, 3])
        backbone_angles = self.angle_mapping(self.backbone_angles, mapping)
        backbone_sidec_angles = self.angle_mapping(self.backbone_sidec_angles, mapping)
        sidechain_angles = self.angle_mapping(self.sidechain_angles, mapping)
        backbone_dihedrals = self.angle_mapping(self.backbone_dihedrals, mapping)
        # print(backbone_angles.size(), sidechain_angles.size()) # torch.Size([308, 3]), torch.Size([51, 3])

        # (1) meta_dict.keys: dict_keys(['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons', 'atom_map',
        # 'node_position', 'bond_type', 'bond_stereo', 'stereo_atoms', 'bead2residue', 'residue_type', 'atom_feature'])

        # (2) print(edge_list.size, edge_index.size(), edge_list[edge_index].size(), self.num_residues, offsets.size())
        # torch.Size([1532, 3]), torch.Size([1532]), torch.Size([1072, 3]), tensor([100, 62, 100, 60]), torch.Size([1532])

        # * we can find that the returned num_residues is same to that before subgraph cropping, the reason should be that *
        # * actually in original sequence truncation logic, the nodes to be masked are only side chain nodes of masked residues *
        # * while the backbone (nodes) of these masked residues are retained, thus the num_residues should not be changed *
        # * in current CG implementation, the backbone beads may also be masked, however, assuming after here *
        # * num_residues will not be used until the next epoch, in this case, we will not change the num_residues either *

        # (3) data_dict['atom_type'].size, data_dict['node_position'].size, data_dict['bead2residue'].size, data_dict['residue_type'].size
        # [503, 3], [503, 1, 3], [503], [322] (being processed individually based on its 'residue reference' tag), [503, 17]

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets[edge_index],
                          backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles,
                          sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals,
                          intermol_mat=self.intermol_mat, meta_dict=meta_dict, **data_dict)

    # * 'prompted_graph_generation' is for injecting the designed protein graph tokens into Protein class (the token attributes are provided from Token class) *
    # * here the protein cropping is already performed (performed before 'collate_fn', within 'get_item' function) *
    # * current protein already contains the complete information to be sent to the protein encoder *
    def prompted_graph_generation(self, token_parameter, token_position):
        # print(self.node2graph, self.node2graph.size(), self.node_feature.size(), self.num_nodes, self.num_cum_nodes)
        # tensor([0, 0, 0,  ..., 4, 4, 4]), torch.Size([2041]), torch.Size([2041, 25]), tensor([403, 409, 392, 393, 444]), tensor([403, 812, 1204, 1597, 2041])
        token_num = token_parameter.size(0)
        token_num_arange = torch.arange(token_num, device=self.device) # relative ids for token in each graph

        # generate some mappings for index transformation
        # * node order in current prompted graph: token helix -> token sheet3 -> token sheet4 -> token coil -> original graph nodes *
        start_node_per_graph = self.num_cum_nodes.clone() - self.num_nodes.clone() # tensor([0, 403, 812, 1204, 1597])
        token_node_offset = torch.arange(0, 0 + token_num * self.num_nodes.size(0), token_num, device=self.device) # tensor([0, 4, 8, 12, 16])
        start_node_per_graph = start_node_per_graph + token_node_offset # tensor([0, 407, 820, 1216, 1613])

        start_res_per_graph = self.num_cum_residues.clone() - self.num_residues.clone() # tensor([0, 171, 356, 539, 708])
        start_res_per_graph = start_res_per_graph + token_node_offset # each token node is a bead and an independent residue
        # tensor([0, 175, 364, 551, 724])

        # handling intermol_mat:
        # intermol_mat contains core region AA pairs for each protein (as a Protein attribute, first column: from interation part A, second column: from B)
        # these AA pairs from different proteins are packed together, using absolute node ids in current batch to distinguish each other
        intermol_mat = self.intermol_mat.clone()
        # because AA in part A and part B will stay in the same protein, we can use one side to determine its belonging protein
        interface_AA_partA = intermol_mat[:, 0]
        intermol_mat_bucket = torch.bucketize(interface_AA_partA, boundaries=self.num_cum_residues, right=True) # right=True: left boundary closed, right open
        intermol_mat_bucket = torch.gather(input=token_node_offset, index=intermol_mat_bucket, dim=0).unsqueeze(-1) # transform the belonging protein id into residue offset
        intermol_mat = intermol_mat + intermol_mat_bucket

        # reconstruct the node features:
        # * in current settings, the coordinates of token nodes are [0, 0, 0], and each token node is treated as a bead and an independent residue *
        # print(self.num_node, self.bead2residue.size(), self.node_position.size()) # tensor(2041), torch.Size([2041]), torch.Size([2041, 3])
        # print(self.bead2residue) # tensor([0, 1, 1, ..., 913, 913, 913]), the contained ids are the residue abs ids for current batch
        node_feature, node_position, bead2residue, atom_type = [], [], [], []
        scale_maps = self.bead2residue.clone()

        # 1) no change to the original node features and coordinates of each graph
        # 2) the token parameters will be added into each graph respectively
        for node_group_id, (feature, position, scale_map, atom) in enumerate(zip(
                torch.split(self.node_feature, self.num_nodes.tolist()),
                torch.split(self.node_position, self.num_nodes.tolist()),
                torch.split(scale_maps, self.num_nodes.tolist()),
                torch.split(self.atom_type, self.num_nodes.tolist()))):

            node_feature.append(token_parameter) # put tokens prior to protein nodes
            node_feature.append(feature) # currently the angular information is already injected into the ordinary node features

            node_position.append(token_position)
            node_position.append(position)

            bead2residue.append(token_num_arange + start_res_per_graph[node_group_id]) # change the residue abs ids
            bead2residue.append(scale_map + token_num * (node_group_id + 1))

            atom_type.append(torch.full((token_num, 3), fill_value=-1, device=self.device)) # use -1 to fill atom_type/bead_type of each token
            atom_type.append(atom)

        node_feature = torch.cat(node_feature)
        node_position = torch.cat(node_position)
        bead2residue = torch.cat(bead2residue)
        atom_type = torch.cat(atom_type)

        num_nodes = self.num_nodes.clone()
        num_nodes += token_num
        num_residues = self.num_residues.clone()
        num_residues += token_num

        # reconstruct the edges and edge features:
        # note that after current process, the generated graph will be sent to the encoder+decoder directly
        assert self.num_relation in [6, 7], "Only the constructed protein graph effectively enhanced by MARTINI edge parameters can utilize graph tokens."
        cg_edge_offset = 1 if self.num_relation == 6 else 2 # 6: across_res_mask=False, 7: across_res_mask=True

        # acquire the edge type ids for corresponding second structure edges in current protein graph
        # other harmonic backbone bonds (coil): backbone_bones, helix bonds: constraints
        # martini22_bond2id = {'backbone_bonds': 0, 'sidechain_bonds': 1, 'sheet_bonds_3': 2, 'sheet_bonds_4': 3, 'constraints': 4}
        helix_id, sheet3_id, sheet4_id, coil_id = self.martini22_bond2id["constraints"] + cg_edge_offset, self.martini22_bond2id["sheet_bonds_3"] + cg_edge_offset, \
                                                self.martini22_bond2id["sheet_bonds_4"] + cg_edge_offset, self.martini22_bond2id["backbone_bonds"] + cg_edge_offset

        # * current plan is to use the original secondary structure edges as the edge type of graph tokens *
        # * in this case, the num_relation will not increase, some computational resources could be saved *
        edges, num_edges, edge_list, edge_feature = self.edge_list.clone(), self.num_edges.clone(), [], []

        # * the change for 'edge' within this loop will not affect the iteration output of the below 'torch.split' function *
        for edge_group_id, (edge, feature) in enumerate(zip(
                torch.split(edges, self.num_edges.tolist()),
                torch.split(self.edge_feature, self.num_edges.tolist()))): # keep original edge feature unchanged

            # update the node index of generated edge index within current graph
            # print(edge, torch.split(edges, num_edges.tolist())[edge_group_id])
            edge[:, :2] += token_num * (edge_group_id + 1)
            # print(edge, torch.split(edges, num_edges.tolist())[edge_group_id])

            # pick the corresponding type of edges from current graph
            helix_edge = edge[edge[:, 2] == helix_id]
            sheet3_edge = edge[edge[:, 2] == sheet3_id]
            sheet4_edge = edge[edge[:, 2] == sheet4_id]
            coil_edge = edge[edge[:, 2] == coil_id]

            # start to construct token edge index
            helix_token_edge, inverse_indices = torch.unique(helix_edge[:, :2], sorted=True, return_inverse=True)
            sheet3_token_edge, inverse_indices = torch.unique(sheet3_edge[:, :2], sorted=True, return_inverse=True)
            sheet4_token_edge, inverse_indices = torch.unique(sheet4_edge[:, :2], sorted=True, return_inverse=True)
            coil_token_edge, inverse_indices = torch.unique(coil_edge[:, :2], sorted=True, return_inverse=True)

            # acquire starting node abs id for current prompted graph
            current_graph_start_id = start_node_per_graph[edge_group_id]

            # example: for helix_token_edge: [abs helix token id in current graph, helix node in original graph, helix edge type]
            helix_token_edge = torch.cat([torch.full((helix_token_edge.size(0), 1), fill_value=current_graph_start_id+0, device=self.device), # relative node id: 0
                            helix_token_edge.unsqueeze(-1), torch.full((helix_token_edge.size(0), 1), fill_value=helix_id, device=self.device)], dim=1)

            sheet3_token_edge = torch.cat([torch.full((sheet3_token_edge.size(0), 1), fill_value=current_graph_start_id+1, device=self.device), # relative node id: 1
                            sheet3_token_edge.unsqueeze(-1), torch.full((sheet3_token_edge.size(0), 1), fill_value=sheet3_id, device=self.device)], dim=1)

            sheet4_token_edge = torch.cat([torch.full((sheet4_token_edge.size(0), 1), fill_value=current_graph_start_id+2, device=self.device), # relative node id: 2
                            sheet4_token_edge.unsqueeze(-1), torch.full((sheet4_token_edge.size(0), 1), fill_value=sheet4_id, device=self.device)], dim=1)

            coil_token_edge = torch.cat([torch.full((coil_token_edge.size(0), 1), fill_value=current_graph_start_id+3, device=self.device), # relative node id: 3
                            coil_token_edge.unsqueeze(-1), torch.full((coil_token_edge.size(0), 1), fill_value=coil_id, device=self.device)], dim=1)

            # need to make the token edge index symmetric then
            helix_token_edge = torch.cat([helix_token_edge, helix_token_edge[:, [1, 0, 2]]], dim=0)
            sheet3_token_edge = torch.cat([sheet3_token_edge, sheet3_token_edge[:, [1, 0, 2]]], dim=0)
            sheet4_token_edge = torch.cat([sheet4_token_edge, sheet4_token_edge[:, [1, 0, 2]]], dim=0)
            coil_token_edge = torch.cat([coil_token_edge, coil_token_edge[:, [1, 0, 2]]], dim=0)
            # current token_edge (i.e., edges between graph tokens and bead nodes) already contains abs node ids in current batch
            token_edge = torch.cat([helix_token_edge, sheet3_token_edge, sheet4_token_edge, coil_token_edge], dim=0)

            token_graph_edge_num = token_edge.size(0)
            # update the edge number for the prompted graph
            num_edges[edge_group_id] = num_edges[edge_group_id] + token_graph_edge_num

            edge_list.extend([token_edge, edge])

            # handle edge features, the input attributes of token_edge_cg22_gearnet should be based on abs (node) ids in current batch
            edge_feature.append(token_edge_cg22_gearnet(token_edge, bead2residue, node_position, len(self.martini22_name2id.keys()), self.num_relation))
            edge_feature.append(feature)

        edge_list = torch.cat(edge_list)
        edge_feature = torch.cat(edge_feature)

        # check how many edges are injected into the original graphs in total
        # print(edge_list.size(), num_edges, self.edge_list.size(), self.edge_feature.size())
        # torch.Size([18410, 3]), tensor([3680, 3678, 3484, 3610, 3958]), torch.Size([15372, 3]), torch.Size([15372, 53])

        # start to pick and modify the registered attributes used for constructing an effective PackedProtein instance:
        # self.data_by_meta() can get all registered attributes according to the specified registered attribute types
        data_dict, meta_dict = self.data_by_meta(include=(
            "node", "residue", "node reference", "residue reference", "graph"))
        # use zero to fill all tensors in data_dict based on the size of prompted graph
        # the useful ones will be re-initialized below
        data_dict = self.data_dict_zero_fill(data_dict, extra_offset=token_num * self.num_nodes.size(0))

        # currently the keys in data_dict and meta_dict correspond with each other:
        # dict_keys(['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons', 'atom_map', 'node_position', 'bead2residue', 'residue_type', 'atom_feature'])
        # * need to change the ones to be used in the following encoder and decoder, e.g., 'node_position', 'bead2residue', 'atom_feature', 'atom_type' *

        # get the offset for each edge (offset is based on cumulative node numbers in current batch) in batch-wise edge_list
        offsets = start_node_per_graph.repeat_interleave(num_edges)
        data_dict["node_position"], data_dict["bead2residue"], data_dict["atom_feature"], data_dict["atom_type"] = node_position, bead2residue, node_feature, atom_type
        data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2]) # for passing the 'bond_type' check in 'Molecule' class

        # 1) edge_weight is not passed in, as in current setting, edge_weight is set to 1 for each edge;
        # 2) angular information is not passed in, as this information is already fused into node features before 'prompted_graph_generation'
        # 3) atom_type and residue_type in data_dict will not change, as the injected tokens do not have specific atom and residue types
        return type(self)(edge_list, num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets, edge_feature=edge_feature,
                          intermol_mat=intermol_mat, meta_dict=meta_dict, **data_dict)

    def data_dict_zero_fill(self, data_dict, extra_offset):
        for key in data_dict:
            data = data_dict[key]
            data_size = list(data.size())
            data_size[0] = data_size[0] + extra_offset
            data = torch.zeros(data_size, device=self.device)
            data_dict[key] = data
        return data_dict

    def angle_mapping(self, angles, mapping):
        if angles.size(0) > 0:
            angle_info = angles.clone()
            angle_info = mapping[angle_info]
            angle_index = (angle_info >= 0).all(dim=-1)
            return angle_info[angle_index]
        else:
            return angles.clone()

    # * in current implementation, it will be used in CGDistancePrediction (graph.edge_mask(~functional.as_mask(indices, graph.num_edge))) *
    # * in this case, the angular node features have not been generated (should be after this 'edge_mask' function), thus angular feature alignment is not considered here *
    # * however, we still need to pass the (unregistered) node composition info for each angle to the new generated protein for the later angular node feature calculation *
    # * while the intermol_mat is only used (in energy decoder) for downstream tasks assuming no further graph cropping in downstream training *
    # * thus here directly return the original intermol_mat (as edge masking does not include extra AA cropping) *
    # edge_mask input: a mask with the size of num_edge, using True to indicate the remained edges
    def edge_mask(self, index):
        index = self._standarize_index(index, self.num_edge) # tensor([0, 1, 2,  ..., 5195, 5196, 5197])
        # for handling standard registered features (not including unregistered features like angular information)
        data_dict, meta_dict = self.data_mask(edge_index=index)
        # return number of retained edges for each protein
        num_edges = self._get_num_xs(index, self.num_cum_edges)
        # CG22_PackedProtein(batch_size=4, num_atoms=[240, 120, 232, 120], num_bonds=[1916, 796, 1678, 808])
        # print(self.num_edge, index.size(), self.num_cum_edges, num_edges)
        # tensor(5198), torch.Size([4283]) (only contain retained edge indices), tensor([1916, 2712, 4390, 5198]), tensor([1679, 571, 1442, 591]) (1679 + 571 + 1442 + 591 = 4283)

        # 'edge_mask' is not a 'compact function', since the node indices have not changed (only edge index changes)
        # print(self.edge_list, self.edge_list.size())
        # tensor([[1, 0, 1], ..., [711, 710, 6]])), torch.Size([5198, 3])
        return type(self)(self.edge_list[index], edge_weight=self.edge_weight[index],
                          num_nodes=self.num_nodes, num_edges=num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=self._offsets[index],
                          # pass the (unregistered) node composition for each angle to the new generated protein (no cropping for the node)
                          backbone_angles=self.backbone_angles, backbone_sidec_angles=self.backbone_sidec_angles,
                          sidechain_angles=self.sidechain_angles, backbone_dihedrals=self.backbone_dihedrals,
                          intermol_mat=self.intermol_mat, meta_dict=meta_dict, **data_dict)

    def residue_mask(self, index, compact=False):
        """
        Return a masked packed protein based on the specified residues.

        Note the compact option is applied to both residue and atom ids, but not graph ids.

        Parameters:
            index (array_like): residue index
            compact (bool, optional): compact residue ids or not

        Returns:
            PackedProtein
        """
        index = self._standarize_index(index, self.num_residue)
        residue_mapping = -torch.ones(self.num_residue, dtype=torch.long, device=self.device)
        residue_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = residue_mapping[self.atom2residue] >= 0
        node_index = self._standarize_index(node_index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_nodes = self._get_num_xs(node_index, self.num_cum_nodes)
            num_residues = self._get_num_xs(index, self.num_cum_residues)
        else:
            mapping[node_index] = node_index
            num_nodes = self.num_nodes
            num_residues = self.num_residues

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)
        num_edges = self._get_num_xs(edge_index, self.num_cum_edges)
        offsets = self._get_offsets(num_nodes, num_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index, residue_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def graph_mask(self, index, compact=False):
        index = self._standarize_index(index, self.batch_size)
        graph_mapping = -torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        graph_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = graph_mapping[self.node2graph] >= 0
        node_index = self._standarize_index(node_index, self.num_node)
        residue_index = graph_mapping[self.residue2graph] >= 0
        residue_index = self._standarize_index(residue_index, self.num_residue)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            key = graph_mapping[self.node2graph[node_index]] * self.num_node + node_index
            order = key.argsort()
            node_index = node_index[order]
            key = graph_mapping[self.residue2graph[residue_index]] * self.num_residue + residue_index
            order = key.argsort()
            residue_index = residue_index[order]
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_nodes = self.num_nodes[index]
            num_residues = self.num_residues[index]
        else:
            mapping[node_index] = node_index
            num_nodes = torch.zeros_like(self.num_nodes)
            num_nodes[index] = self.num_nodes[index]
            num_residues = torch.zeros_like(self.num_residues)
            num_residues[index] = self.num_residues[index]

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)
        if compact:
            key = graph_mapping[self.edge2graph[edge_index]] * self.num_edge + edge_index
            order = key.argsort()
            edge_index = edge_index[order]
            num_edges = self.num_edges[index]
        else:
            num_edges = torch.zeros_like(self.num_edges)
            num_edges[index] = self.num_edges[index]
        offsets = self._get_offsets(num_nodes, num_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index,
                                                  residue_index=residue_index, graph_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def get_item(self, index):
        node_index = torch.arange(self.num_cum_nodes[index] - self.num_nodes[index], self.num_cum_nodes[index],
                                  device=self.device)
        edge_index = torch.arange(self.num_cum_edges[index] - self.num_edges[index], self.num_cum_edges[index],
                                  device=self.device)
        residue_index = torch.arange(self.num_cum_residues[index] - self.num_residues[index],
                                     self.num_cum_residues[index], device=self.device)
        graph_index = index
        edge_list = self.edge_list[edge_index].clone()
        edge_list[:, :2] -= self._offsets[edge_index].unsqueeze(-1)
        data_dict, meta_dict = self.data_mask(node_index, edge_index,
                                              residue_index=residue_index, graph_index=graph_index)

        return self.unpacked_type(edge_list, edge_weight=self.edge_weight[edge_index], num_node=self.num_nodes[index],
                                  num_relation=self.num_relation, meta_dict=meta_dict, **data_dict)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mols, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a packed protein from a list of RDKit objects.

        Parameters:
            mols (list of rdchem.Mol): molecules
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str or list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = PackedMolecule.from_molecule(mols, atom_feature=atom_feature, bond_feature=bond_feature,
                                               mol_feature=mol_feature, with_hydrogen=False, kekulize=kekulize)
        residue_feature = cls._standarize_option(residue_feature)

        residue_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom2residue = []
        residue_number = []
        insertion_code = []
        chain_id = []
        _residue_feature = []
        last_residue = None
        num_residues = []
        num_cum_residue = 0

        mols = mols + [cls.dummy_protein]
        for mol in mols:
            if mol is None:
                mol = cls.empty_mol

            if kekulize:
                Chem.Kekulize(mol)

            for atom in mol.GetAtoms():
                residue = atom.GetPDBResidueInfo()
                number = residue.GetResidueNumber()
                code = residue.GetInsertionCode()
                type = residue.GetResidueName().strip()
                canonical_residue = (number, code, type)
                if canonical_residue != last_residue:
                    last_residue = canonical_residue
                    if type not in cls.residue2id:
                        warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                        type = "GLY"
                    residue_type.append(cls.residue2id[type])
                    residue_number.append(number)
                    insertion_code.append(cls.alphabet2id[residue.GetInsertionCode()])
                    chain_id.append(cls.alphabet2id[residue.GetChainId()])
                    feature = []
                    for name in residue_feature:
                        func = R.get("features.residue.%s" % name)
                        feature += func(residue)
                    _residue_feature.append(feature)
                name = residue.GetName().strip()
                if name not in cls.atom_name2id:
                    name = "UNK"
                atom_name.append(cls.atom_name2id[name])
                is_hetero_atom.append(residue.GetIsHeteroAtom())
                occupancy.append(residue.GetOccupancy())
                b_factor.append(residue.GetTempFactor())
                atom2residue.append(len(residue_type) - 1)

            num_residues.append(len(residue_type) - num_cum_residue)
            num_cum_residue = len(residue_type)

        residue_type = torch.tensor(residue_type)[:-1]
        atom_name = torch.tensor(atom_name)[:-5]
        is_hetero_atom = torch.tensor(is_hetero_atom)[:-5]
        occupancy = torch.tensor(occupancy)[:-5]
        b_factor = torch.tensor(b_factor)[:-5]
        atom2residue = torch.tensor(atom2residue)[:-5]
        residue_number = torch.tensor(residue_number)[:-1]
        insertion_code = torch.tensor(insertion_code)[:-1]
        chain_id = torch.tensor(chain_id)[:-1]
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)[:-1]
        else:
            _residue_feature = None

        num_residues = num_residues[:-1]

        return cls(protein.edge_list, residue_type=residue_type,
                   num_nodes=protein.num_nodes, num_edges=protein.num_edges, num_residues=num_residues,
                   atom_name=atom_name, atom2residue=atom2residue, residue_feature=_residue_feature,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id,
                   offsets=protein._offsets, meta_dict=protein.meta_dict, **protein.data_dict)

    @classmethod
    def _residue_from_sequence(cls, sequences):
        num_residues = []
        residue_type = []
        residue_feature = []
        sequences = sequences + ["G"]
        for sequence in sequences:
            for residue in sequence:
                if residue not in cls.residue_symbol2id:
                    warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)
                    residue = "G"
                residue_type.append(cls.residue_symbol2id[residue])
                residue_feature.append(feature.onehot(residue, cls.residue_symbol2id, allow_unknown=True))
            num_residues.append(len(sequence))

        residue_type = residue_type[:-1]
        residue_feature = torch.tensor(residue_feature)[:-1]

        edge_list = torch.zeros(0, 3, dtype=torch.long)
        num_nodes = [0] * (len(sequences) - 1)
        num_edges = [0] * (len(sequences) - 1)
        num_residues = num_residues[:-1]

        return cls(edge_list=edge_list, atom_type=[], bond_type=[], residue_type=residue_type,
                   num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                   residue_feature=residue_feature)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_sequence(cls, sequences, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a packed protein from a list of sequences.

        .. note::

            It takes considerable time to construct proteins with a large number of atoms and bonds.
            If you only need residue information, you may speed up the construction by setting
            ``atom_feature`` and ``bond_feature`` to ``None``.

        Parameters:
            sequences (str): list of protein sequences
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str or list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if atom_feature is None and bond_feature is None and residue_feature == "default":
            return cls._residue_from_sequence(sequences)

        mols = []
        for sequence in sequences:
            mol = Chem.MolFromSequence(sequence)
            if mol is None:
                raise ValueError("Invalid sequence `%s`" % sequence)
            mols.append(mol)

        return cls.from_molecule(mols, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_pdb(cls, pdb_files, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        """
        Create a protein from a list of PDB files.

        Parameters:
            pdb_files (str): list of file names
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mols = []
        for pdb_file in pdb_files:
            mol = Chem.MolFromPDBFile(pdb_file)
            mols.append(mol)

        return cls.from_molecule(mols, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    def to_molecule(self, ignore_error=False):
        mols = super(CG22_PackedProtein, self).to_molecule(ignore_error)

        residue_type = self.residue_type.tolist()
        atom_name = self.atom_name.tolist()
        atom2residue = self.atom2residue.tolist()
        is_hetero_atom = self.is_hetero_atom.tolist()
        occupancy = self.occupancy.tolist()
        b_factor = self.b_factor.tolist()
        residue_number = self.residue_number.tolist()
        chain_id = self.chain_id.tolist()
        insertion_code = self.insertion_code.tolist()
        num_cum_nodes = [0] + self.num_cum_nodes.tolist()

        for i, mol in enumerate(mols):
            for j, atom in enumerate(mol.GetAtoms(), num_cum_nodes[i]):
                r = atom2residue[j]
                residue = Chem.AtomPDBResidueInfo()
                residue.SetResidueNumber(residue_number[r])
                residue.SetChainId(self.id2alphabet[chain_id[r]])
                residue.SetInsertionCode(self.id2alphabet[insertion_code[r]])
                residue.SetName(" %-3s" % self.id2atom_name[atom_name[j]])
                residue.SetResidueName(self.id2residue[residue_type[r]])
                residue.SetIsHeteroAtom(is_hetero_atom[j])
                residue.SetOccupancy(occupancy[j])
                residue.SetTempFactor(b_factor[j])
                atom.SetPDBResidueInfo(residue)

        return mols

    def to_sequence(self):
        """
        Return a list of sequences.

        Returns:
            list of str
        """
        residue_type = self.residue_type.tolist()
        cc_id = self.connected_component_id.tolist()
        num_cum_residues = [0] + self.num_cum_residues.tolist()
        sequences = []
        for i in range(self.batch_size):
            sequence = []
            for j in range(num_cum_residues[i], num_cum_residues[i + 1]):
                if j > num_cum_residues[i] and cc_id[j] > cc_id[j - 1]:
                    sequence.append(".")
                sequence.append(self.id2residue_symbol[residue_type[j]])
            sequence = "".join(sequence)
            sequences.append(sequence)
        return sequences

    def to_pdb(self, pdb_files):
        """
        Write this packed protein to several pdb files.

        Parameters:
            pdb_files (list of str): list of file names
        """
        mols = self.to_molecule()
        for mol, pdb_file in zip(mols, pdb_files):
            Chem.MolToPDBFile(mol, pdb_file, flavor=10)

    def merge(self, graph2graph):
        graph2graph = torch.as_tensor(graph2graph, dtype=torch.long, device=self.device)
        # coalesce arbitrary graph IDs to [0, n)
        _, graph2graph = torch.unique(graph2graph, return_inverse=True)

        graph_key = graph2graph * self.batch_size + torch.arange(self.batch_size, device=self.device)
        graph_index = graph_key.argsort()
        graph = self.subbatch(graph_index)
        graph2graph = graph2graph[graph_index]

        num_graph = graph2graph[-1] + 1
        num_nodes = scatter_add(graph.num_nodes, graph2graph, dim_size=num_graph)
        num_edges = scatter_add(graph.num_edges, graph2graph, dim_size=num_graph)
        num_residues = scatter_add(graph.num_residues, graph2graph, dim_size=num_graph)
        offsets = self._get_offsets(num_nodes, num_edges)

        data_dict, meta_dict = graph.data_mask(exclude="graph")

        return type(self)(graph.edge_list, edge_weight=graph.edge_weight, num_nodes=num_nodes,
                          num_edges=num_edges, num_residues=num_residues, view=self.view, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def repeat(self, count):
        num_nodes = self.num_nodes.repeat(count)
        num_edges = self.num_edges.repeat(count)
        num_residues = self.num_residues.repeat(count)
        offsets = self._get_offsets(num_nodes, num_edges)
        edge_list = self.edge_list.repeat(count, 1)
        edge_list[:, :2] += (offsets - self._offsets.repeat(count)).unsqueeze(-1)

        data_dict = {}
        for k, v in self.data_dict.items():
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for _type in self.meta_dict[k]:
                if _type == "node reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_node
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "edge reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_edge
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "residue reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_residue
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "graph reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.batch_size
                    v = v + pack_offsets.repeat_interleave(length)
            data_dict[k] = v

        return type(self)(edge_list, edge_weight=self.edge_weight.repeat(count),
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=self.view,
                          num_relation=self.num_relation, offsets=offsets,
                          meta_dict=self.meta_dict, **data_dict)

    def repeat_interleave(self, repeats):
        repeats = torch.as_tensor(repeats, dtype=torch.long, device=self.device)
        if repeats.numel() == 1:
            repeats = repeats * torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        num_nodes = self.num_nodes.repeat_interleave(repeats)
        num_edges = self.num_edges.repeat_interleave(repeats)
        num_residues = self.num_residues.repeat_interleave(repeats)
        num_cum_nodes = num_nodes.cumsum(0)
        num_cum_edges = num_edges.cumsum(0)
        num_cum_residues = num_residues.cumsum(0)
        num_node = num_nodes.sum()
        num_edge = num_edges.sum()
        num_residue = num_residues.sum()
        batch_size = repeats.sum()
        num_graphs = torch.ones(batch_size, device=self.device)

        # special case 1: graphs[i] may have no node or no edge
        # special case 2: repeats[i] may be 0
        cum_repeats_shifted = repeats.cumsum(0) - repeats
        graph_mask = cum_repeats_shifted < batch_size
        cum_repeats_shifted = cum_repeats_shifted[graph_mask]

        index = num_cum_nodes - num_nodes
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_nodes, self.num_nodes[graph_mask]])
        mask = index < num_node
        node_index = scatter_add(value[mask], index[mask], dim_size=num_node)
        node_index = (node_index + 1).cumsum(0) - 1

        index = num_cum_edges - num_edges
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_edges, self.num_edges[graph_mask]])
        mask = index < num_edge
        edge_index = scatter_add(value[mask], index[mask], dim_size=num_edge)
        edge_index = (edge_index + 1).cumsum(0) - 1

        index = num_cum_residues - num_residues
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_residues, self.num_residues[graph_mask]])
        mask = index < num_residue
        residue_index = scatter_add(value[mask], index[mask], dim_size=num_residue)
        residue_index = (residue_index + 1).cumsum(0) - 1

        graph_index = torch.repeat_interleave(repeats)

        offsets = self._get_offsets(num_nodes, num_edges)
        edge_list = self.edge_list[edge_index]
        edge_list[:, :2] += (offsets - self._offsets[edge_index]).unsqueeze(-1)

        node_offsets = None
        edge_offsets = None
        residue_offsets = None
        graph_offsets = None
        data_dict = {}
        for k, v in self.data_dict.items():
            num_xs = None
            pack_offsets = None
            for _type in self.meta_dict[k]:
                if _type == "node":
                    v = v[node_index]
                    num_xs = num_nodes
                elif _type == "edge":
                    v = v[edge_index]
                    num_xs = num_edges
                elif _type == "residue":
                    v = v[residue_index]
                    num_xs = num_residues
                elif _type == "graph":
                    v = v[graph_index]
                    num_xs = num_graphs
                elif _type == "node reference":
                    if node_offsets is None:
                        node_offsets = self._get_repeat_pack_offsets(self.num_nodes, repeats)
                    pack_offsets = node_offsets
                elif _type == "edge reference":
                    if edge_offsets is None:
                        edge_offsets = self._get_repeat_pack_offsets(self.num_edges, repeats)
                    pack_offsets = edge_offsets
                elif _type == "residue reference":
                    if residue_offsets is None:
                        residue_offsets = self._get_repeat_pack_offsets(self.num_residues, repeats)
                    pack_offsets = residue_offsets
                elif _type == "graph reference":
                    if graph_offsets is None:
                        graph_offsets = self._get_repeat_pack_offsets(num_graphs, repeats)
                    pack_offsets = graph_offsets
            # add offsets to make references point to indexes in their own graph
            if num_xs is not None and pack_offsets is not None:
                v = v + pack_offsets.repeat_interleave(num_xs)
            data_dict[k] = v

        return type(self)(edge_list, edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=self.view,
                          num_relation=self.num_relation, offsets=offsets, meta_dict=self.meta_dict, **data_dict)

    def undirected(self, add_inverse=True):
        undirected = PackedMolecule.undirected(self, add_inverse=add_inverse)

        return type(self)(undirected.edge_list, edge_weight=undirected.edge_weight,
                          num_nodes=undirected.num_nodes, num_edges=undirected.num_edges,
                          num_residues=self.num_residues, view=self.view, num_relation=undirected.num_relation,
                          offsets=undirected._offsets, meta_dict=undirected.meta_dict, **undirected.data_dict)

    def detach(self):
        return type(self)(self.edge_list.detach(), edge_weight=self.edge_weight.detach(),
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                          meta_dict=self.meta_dict, **utils.detach(self.data_dict))

    def clone(self):
        return type(self)(self.edge_list.clone(), edge_weight=self.edge_weight.clone(),
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                          meta_dict=self.meta_dict, **utils.clone(self.data_dict))

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight, num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                view=self.view, num_relation=self.num_relation, offsets=self._offsets, backbone_angles=self.backbone_angles,
                backbone_sidec_angles=self.backbone_sidec_angles, sidechain_angles=self.sidechain_angles, backbone_dihedrals=self.backbone_dihedrals,
                intermol_mat=self.intermol_mat, meta_dict=self.meta_dict, **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                              view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                              meta_dict=self.meta_dict, **utils.cpu(self.data_dict))

    @utils.cached_property
    def residue2graph(self):
        """Residue id to graph id mapping."""
        range = torch.arange(self.batch_size, device=self.device)
        residue2graph = range.repeat_interleave(self.num_residues)
        return residue2graph

    @utils.cached_property
    def connected_component_id(self):
        cc_id = super(CG22_PackedProtein, self).connected_component_id
        cc_id_offsets = scatter_min(cc_id, self.residue2graph, dim_size=self.num_residue)[0][self.residue2graph]
        cc_id = cc_id - cc_id_offsets
        return cc_id

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_atoms=%s" % pretty.long_array(self.num_nodes.tolist()),
                  "num_bonds=%s" % pretty.long_array(self.num_edges.tolist()),
                  # "num_residues=%s" % pretty.long_array(self.num_residues.tolist())
                ]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


CG22_Protein.packed_type = CG22_PackedProtein


















