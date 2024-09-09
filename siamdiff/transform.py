import os
import glob
import math
import numpy as np

import torch

from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from siamdiff import rotamer


# the function to create conformers for the siamese diffusion scheme
# a commonly used scheme for sampling randomly simulated conformers by adding torsional perturbations to the side chains (graph2)
@R.register("transforms.NoiseTransform")
class NoiseTransform(core.Configurable):

    def __init__(self, noise_type="gaussian", sigma=0.3):
        assert noise_type in ["gaussian", "torsion"]
        self.noise_type = noise_type # torsion for atom-level
        self.sigma = sigma # 0.1 for atom-level

    def __call__(self, item):
        # the noise generated could differ in different epochs
        graph = item["graph"].clone()
        if self.noise_type == "gaussian":
            # ** current plan for CG-level conformer generation is based on adding *independent* gaussian noise to every dim of every coordinate of each protein **
            # ** which may not be optimal, which could be further improved later (e.g., injecting noise under more reasonable domain knowledge) **
            # fill the pre-defined shape with noise drawn from normal distribution with mean 0 and variance 1
            perturb_noise = torch.randn_like(graph.node_position)
            graph.node_position = graph.node_position + perturb_noise * self.sigma
        elif self.noise_type == "torsion":
            # generate noise from standard normal distribution (assuming 4 torsion angles per residue) -> should refer to the wrapped normal distribution described in 4.1 section of original paper
            torsion_updates = torch.randn((graph.num_residue, 4), device=graph.device) * self.sigma * np.pi
            # print(torsion_updates[0]) differs in different epochs
            # print(torsion_updates.size()) # [100, 4], only contain the updates for current protein
            # NoiseTransform is called by __get_item__ in Pytorch Dataset, which defines the processing behaviour for individual proteins in epochs
            rotamer.rotate_side_chain(graph, torsion_updates) # update input graph/protein in-place
        item["graph2"] = graph
        return item


@R.register("transfroms.AtomFeature")
class AtomFeature(core.Configurable):

    def __init__(self, atom_feature=None, keys="graph"):
        self.atom_feature = atom_feature
        if isinstance(keys, str):
            keys = [keys]
        # keys are specified in .yaml file
        self.keys = keys # ['graph', 'graph2'], there could be more than one graph/protein need to be processed (e.g., in SiamDiff implementation)

    def __call__(self, item):
        # iterate each contained protein in item
        for key in self.keys:
            graph = item[key]

            # graph.atom_type: global atom type ids instead of atom position ids in different residues, torch.Size([813])
            graph = graph.subgraph(graph.atom_type != 0)
            graph = graph.subgraph(graph.atom_type < 18)
            # return the frequency for each element appearing in atom2residue (i.e., how many atoms contained in each residue of current protein)
            residue2num_atom = graph.atom2residue.bincount(minlength=graph.num_residue)
            # residue2num_atom: [6, 11, 9, 6, 8, 7, 14, 9, 8, 7, 9, 11, 8, 7, 4, 8, 9, 7]

            graph = graph.subresidue(residue2num_atom > 0)

            # atom_feature is also specified in .yaml file, e.g., residue_symbol
            # one-hot atom-type (i.e., atomic number) + one-hot residue-type
            if self.atom_feature == "residue_symbol":
                atom_feature = torch.cat([
                    functional.one_hot(graph.atom_type.clamp(max=17), 18),
                    # graph.residue_type.size: torch.Size([100])
                    functional.one_hot(graph.residue_type[graph.atom2residue], 21)
                ], dim=-1)
            else:
                raise ValueError

            # store generated atom-level features using 'atom' context manager
            # based on this, this function is more suitable to be used in atom-level model
            with graph.atom():
                # atom feature is the concatenation of atom type and residue type for each atom (not including any other close-neighboring geometric information here)
                graph.atom_feature = atom_feature
            item[key] = graph
        return item


@R.register("transforms.TruncateProteinPair")
class TruncateProteinPair(core.Configurable):

    def __init__(self, max_length=None, random=False):
        self.truncate_length = max_length
        self.random = random

    def __call__(self, item):
        new_item = item.copy()
        graph1 = item["graph1"]
        graph2 = item["graph2"]
        length = graph1.num_residue
        if length <= self.truncate_length:
            return item
        residue_mask = graph1.residue_type != graph2.residue_type
        index = residue_mask.nonzero()[:, 0]
        if self.random:
            start = math.randint(index, min(index + self.truncate_length, length)) - self.truncate_length
        else:
            start = min(index - self.truncate_length // 2, length - self.truncate_length)
        start = max(start, 0)
        end = start + self.truncate_length
        mask = torch.zeros(length, dtype=torch.bool, device=graph1.device)
        mask[start:end] = True
        new_item["graph1"] = graph1.subresidue(mask)
        new_item["graph2"] = graph2.subresidue(mask)

        return new_item
