import torch
from torch import nn
from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from cg3_steps.cg3_protein import CG3_PackedProtein


@R.register("layers.CG3_GraphConstruction")
class CG3_GraphConstruction(nn.Module, core.Configurable):
    max_seq_dist = 10

    def __init__(self, node_layers=None, edge_layers=None, edge_feature="residue_type"):
        super(CG3_GraphConstruction, self).__init__()

        if node_layers is None:
            self.node_layers = nn.ModuleList()
        else:
            self.node_layers = nn.ModuleList(node_layers)

        if edge_layers is None:
            edge_layers = nn.ModuleList()
        else:
            edge_layers = nn.ModuleList(edge_layers)
        self.edge_layers = edge_layers
        self.edge_feature = edge_feature

    def edge_residue_type(self, graph, edge_list, num_relation):
        node_in, node_out, _ = edge_list.t()
        # residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        residue_in, residue_out = graph.bead2residue[node_in], graph.bead2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id))
        ], dim=-1)

    def edge_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t() # target node, source node
        # residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        residue_in, residue_out = graph.bead2residue[node_in], graph.bead2residue[node_out]
        in_residue_type = graph.residue_type[residue_in] # get the residue type of the target nodes
        out_residue_type = graph.residue_type[residue_out] # get the residue type of the source nodes
        sequential_dist = torch.abs(residue_in - residue_out) # sequential distance
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) # Euclidean distance

        return torch.cat([
            # residue type encoding, length: 20 in total
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(r, num_relation),
            functional.one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 1), # 0-10, 11 in total
            spatial_dist.unsqueeze(-1)
        ], dim=-1)

    # replace the residue type embeddings of end nodes with bead type embeddings
    # this function is called by 'apply_edge_layer' function below
    def edge_cg3_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t() # target node, source node
        in_bead_type, out_bead_type = graph.atom_type[:, 0][node_in], graph.atom_type[:, 0][node_out] # atom_type: [bead, res, bead_pos]
        residue_in, residue_out = graph.bead2residue[node_in], graph.bead2residue[node_out]
        sequential_dist = torch.abs(residue_in - residue_out) # sequential distance
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) # Euclidean distance

        return torch.cat([
            # bead type encoding, length: 23 in total for MARTINI3
            functional.one_hot(in_bead_type, len(graph.martini3_name2id.keys())),
            functional.one_hot(out_bead_type, len(graph.martini3_name2id.keys())),
            functional.one_hot(r, num_relation),
            functional.one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 1), # 0-10, 11 in total
            spatial_dist.unsqueeze(-1)
        ], dim=-1)

    def apply_node_layer(self, graph):
        for layer in self.node_layers:
            graph = layer(graph)
        return graph

    # ** in current mode, only the first edge_layer function in input list is supported **
    def apply_edge_layer(self, graph):
        if not self.edge_layers:
            return graph

        assert len(self.edge_layers) > 0, "the input edge layer function number should be larger than 0, current number: {}". \
            format(len(self.edge_layers))

        edge_list, num_relation = self.edge_layers[0](graph)

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0] # target node
        # graph.node2graph is a tensor with the shape of batch node number indicating the bead node allocation to each protein (in current batch)
        edge2graph = graph.node2graph[node_in]
        # sort edges according to the order of the protein in current batch
        order = edge2graph.argsort()
        edge_list = edge_list[order]

        # bincount: count the occurrence time for each element (consecutive int starting from 0) in the tensor
        num_edges = edge2graph.bincount(minlength=graph.batch_size) # tensor([974, 346, 734, 382])
        # offsets for each group of edges for every protein (in current batch)
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)

        if hasattr(self, "edge_%s" % self.edge_feature):
            # 'edge_gearnet' edge features: end node features, one-hot edge type encoding, sequential and spatial distances between end nodes
            # 'getattr' calls corresponding edge generation function contained in this CG3_GraphConstruction class
            edge_feature = getattr(self, "edge_%s" % self.edge_feature)(graph, edge_list, num_relation)
        elif self.edge_feature is None:
            edge_feature = None
        else:
            raise ValueError("Unknown edge feature `%s`" % self.edge_feature)

        # the features can be correctly handled if these features are correctly registered as atom or residue features using the context manager
        data_dict, meta_dict = graph.data_by_meta(include=(
            "node", "residue", "node reference", "residue reference", "graph"))
        # meta_dict.keys:
        # ['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons', 'atom_map', 'node_position',
        # 'bead2residue', 'residue_type', 'atom_feature']
        # as shown below, because the new edge_list and edge_feature are generated, the bond_type features are no longer needed after this step,
        # at the same time, the node feature atom_feature is retrieved from the input graph, which will be sent to the new graph returned

        # returned features in CG3_Protein.pack (for creating a packed protein object):
        # ['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons', 'atom_map', 'node_position',
        # 'bond_type', 'bond_stereo', 'stereo_atoms', 'bead2residue', 'residue_type']

        # print(isinstance(graph, data.CG3_PackedProtein), isinstance(graph, CG3_PackedProtein))
        # if isinstance(graph, data.PackedProtein): # original object: PackedProtein, current object is CG3_PackedProtein rather than PackedProtein
        # if isinstance(graph, data.CG3_PackedProtein): # data.CG3_PackedProtein should be registered in torchdrug.data.__init__ for being identified
        if isinstance(graph, CG3_PackedProtein):
            data_dict["num_residues"] = graph.num_residues
        if isinstance(graph, data.PackedMolecule):
            data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2])

        # also return the information contained in PackedProtein.attributes (e.g., angle information)
        return type(graph)(edge_list, num_nodes=graph.num_nodes, num_edges=num_edges, num_relation=num_relation,
                           view=graph.view, offsets=offsets, edge_feature=edge_feature,
                           backbone_angles=graph.backbone_angles, backbone_sidec_angles=graph.backbone_sidec_angles,
                           sidechain_angles=graph.sidechain_angles, backbone_dihedrals=graph.backbone_dihedrals,
                           intermol_mat=graph.intermol_mat, meta_dict=meta_dict, **data_dict)

    def forward(self, graph):
        """
        Generate a new graph based on the input graph and pre-defined node and edge layers.
        Parameters:
            graph (Graph): :math:`n` graph(s)
        Returns:
            graph (Graph): new graph(s)
        """
        graph = self.apply_node_layer(graph)
        # print(self.node_layers)
        graph = self.apply_edge_layer(graph)
        # print(self.edge_layers)

        return graph
