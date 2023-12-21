import torch
from torch import nn
from torch_cluster import knn_graph, radius_graph
from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torch_scatter import scatter_min


@R.register("layers.geometry.AdvSpatialEdge")
class AdvSpatialEdge(nn.Module, core.Configurable):
    """
    Construct edges between nodes within a specified radius.

    Parameters:
        radius (float, optional): spatial radius
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, radius=5, min_distance=5, max_distance=None, max_num_neighbors=32,
                 across_res_mask=True, cg_edge_enhance=True, cg_edge_reduction=True):
        super(AdvSpatialEdge, self).__init__()
        self.radius = radius
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_num_neighbors = max_num_neighbors
        # extra hyperparameter
        self.across_res_mask = across_res_mask
        self.cg_edge_enhance = cg_edge_enhance
        self.cg_edge_reduction = cg_edge_reduction

    def forward(self, graph):
        """
        Return spatial radius edges constructed based on the input graph.
        Parameters:
            graph (Graph): :math:`n` graph(s)
        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """

        # batch hyperparameter: assigns each node to a specific example
        # print(graph.node_position.size(), graph.node2graph.size()) # torch.Size([503, 3]), torch.Size([503])
        edge_list = radius_graph(graph.node_position, r=self.radius, batch=graph.node2graph, max_num_neighbors=self.max_num_neighbors).t()
        # print(edge_list, edge_list.size()) # torch.Size([1364, 2])
        # edge_list = radius_graph(graph.node_position.squeeze(1), r=self.radius, max_num_neighbors=self.max_num_neighbors).t()
        # print(edge_list, edge_list.size()) # torch.Size([1734, 2]), more edges between different proteins could be established
        # for example: one edge: tensor([419, 183]), graph.num_nodes.cumsum(0): tensor([183, 263, 413, 503]), nodes 419 and 183 should not be connected theortically
        # Note: here hyperparameter batch=graph.node2graph should be needed, it utilizes torch.bucketize to assign each node to a specific example (protein)
        # in this case, the edges are established within each individual protein, the edge difference of using/not using batch hyperparameter can be found above
        # torch.bucketize: https://discuss.pytorch.org/t/what-does-torch-bucketize-do-used-for/145519/2

        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=graph.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        num_relation = 1

        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            # remove the edge which sequential distance between two end nodes is smaller than min_distance
            mask = (graph.bead2residue[node_in] - graph.bead2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        if self.max_distance:
            node_in, node_out = edge_list.t()[:2]
            # remove the edge which sequential distance between two end nodes is larger than min_distance
            mask = (graph.bead2residue[node_in] - graph.bead2residue[node_out]).abs() > self.max_distance
            edge_list = edge_list[~mask]

        # extra options:
        # (1) determine edges across different residues
        if self.across_res_mask:
            node_in, node_out = edge_list.t()[:2] # node_in: source_node, node_out: target_node
            # the bead2residue id should be the same for both end nodes in the same residue
            across_res_mask = (graph.bead2residue[node_in] - graph.bead2residue[node_out]).abs() > 0 # True for across residue edges
            edge_list[:, 2][across_res_mask] = 1 # intra residue edges: 0, across/inter residue edges: 1
            num_relation += 1

        # (2) whether the spatial edges will be enhanced by CG defined edges
        if self.cg_edge_enhance:
            edge_list_enhance = graph.edge_list.clone()
            if self.across_res_mask:
                edge_list_enhance[:, 2] += 2
            else:
                edge_list_enhance[:, 2] += 1
            # upper: radius edges, lower: cg defined edges
            edge_list = torch.cat([edge_list, edge_list_enhance], dim=0)
            # * there should be no conflicts between inter-residue edges and intra-residue edges (same to backbone_bonds and sidechain_bonds) *

            if self.cg_edge_reduction:
                # return_inverse=True: return the indices for where elements in the original input ended up in the returned unique list
                _, inverse_indices = edge_list[:, :2].unique(sorted=True, return_inverse=True, dim=0)

                # edge_list.size(), inverse_indices.size(), torch.max(inverse_indices), torch.min(inverse_indices): [2338, 3], 2338, 1333, 0
                edge_list = scatter_min(src=edge_list, index=inverse_indices, dim=0)[0]
                # output: [1334, 3], use the index in inverse_indices to determine the size of updated edge_list (not by 'dim_size' hyperparameter)
                # * as use scatter_min to perform edge reduction and radius edge has smaller edge type index (0 or 1), more radius edges could be retained *

            num_relation += 5

        # remove the edge which Euclidean distance between two end nodes is too small
        node_in, node_out = edge_list.t()[:2]
        mask = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, num_relation