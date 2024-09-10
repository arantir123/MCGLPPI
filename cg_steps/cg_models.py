import torch
from torch import nn
from collections.abc import Sequence
from torch_scatter import scatter_add, scatter_mean
from torchdrug import core, layers
from torchdrug.core import Registry as R
from torch.nn import functional as F
from torch_scatter import scatter_add
from torchdrug import layers


@R.register("models.CG22_GCN")
class CG22_GCN(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(CG22_GCN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        print('Use CG22_GCN.')

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GraphConv(self.dims[i], self.dims[i + 1], edge_input_dim, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


# Note: this version uses Pytorch Geometric to create adjacent matrix/edge index, not including C++ extension just-in-time (JIT) calculation
# at the same time, it supports the IEConv layer and extra input hidden embedding layer, etc.
@R.register("models.CG22_GearNetIEConv")
class CG22_GearNetIEConv(nn.Module, core.Configurable):

    def __init__(self, input_dim, embedding_dim, hidden_dims, num_relation, edge_input_dim=None,
                 batch_norm=False, activation="relu", concat_hidden=False, short_cut=True,
                 readout="sum", dropout=0, num_angle_bin=None, layer_norm=False, use_ieconv=False):
        super(CG22_GearNetIEConv, self).__init__()
        print('using CG22_GearNetIEConv.')

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [embedding_dim if embedding_dim > 0 else input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.layer_norm = layer_norm
        self.use_ieconv = use_ieconv

        if embedding_dim > 0:
            self.linear = nn.Linear(input_dim, embedding_dim)
            self.embedding_batch_norm = nn.BatchNorm1d(embedding_dim)

        self.layers = nn.ModuleList()
        self.ieconvs = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            # note that these GeometricRelationalGraphConv layers are from this script instead of torchdrug.layers
            self.layers.append(GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
            if use_ieconv:
                self.ieconvs.append(IEConvLayer(self.dims[i], self.dims[i] // 4,
                                    self.dims[i+1], edge_input_dim=14, kernel_hidden_dim=32))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def get_ieconv_edge_feature(self, graph):
        u = torch.ones_like(graph.node_position)
        u[1:] = graph.node_position[1:] - graph.node_position[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(graph.node_position)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(graph.node_position)
        n[:-1] = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

        node_in, node_out = graph.edge_list.t()[:2]
        t = graph.node_position[node_out] - graph.node_position[node_in]
        t = torch.einsum('ijk, ij->ik', local_frame[node_in], t)
        r = torch.sum(local_frame[node_in] * local_frame[node_out], dim=1)
        # change atom2residue to bead2residue
        delta = torch.abs(graph.bead2residue[node_in] - graph.bead2residue[node_out]).float() / 6
        delta = delta.unsqueeze(-1)

        return torch.cat([
            t, r, delta,
            1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
        ], dim=-1)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = input
        if self.embedding_dim > 0:
            layer_input = self.linear(layer_input)
            layer_input = self.embedding_batch_norm(layer_input)
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_hidden = line_graph.node_feature.float()
            # print(line_graph)
            # under AdvSpatialEdge radius = 4.5 (protein length = 100):
            # PackedGraph(batch_size=8, num_nodes=[448, 482, 474, 462, 486, 666, 660, 466],
            # num_edges=[1352, 1848, 1896, 1788, 1522, 2496, 2728, 1540], num_relation=8, device='cuda:0')
            # under AdvSpatialEdge radius = 5 (protein length = 100):
            # PackedGraph(batch_size=8, num_nodes=[844, 704, 700, 688, 886, 1074, 970, 736],
            # num_edges=[4856, 3856, 3976, 3684, 5114, 6526, 6010, 3832], num_relation=8, device='cuda:0')

            # print(graph)
            # under AdvSpatialEdge radius = 4.5 (protein length = 100):
            # CG22_PackedProtein(batch_size=8, num_atoms=[176, 149, 147, 147, 181, 220, 192, 172],
            # num_bonds=[448, 482, 474, 462, 486, 666, 660, 466], device='cuda:0')
            # under AdvSpatialEdge radius = 5 (protein length = 100):
            # CG22_PackedProtein(batch_size=8, num_atoms=[176, 149, 147, 147, 181, 220, 192, 172],
            # num_bonds=[844, 704, 700, 688, 886, 1074, 970, 736], device='cuda:0')
        else:
            edge_hidden = None
        ieconv_edge_feature = self.get_ieconv_edge_feature(graph)
        # print(layer_input.size(), edge_hidden.size(), ieconv_edge_feature.size())
        # torch.Size([502, 128]), torch.Size([1326, 59]), torch.Size([1326, 14])

        for i in range(len(self.layers)):
            # edge message passing
            if self.num_angle_bin:
                # print(edge_hidden.size()) # torch.Size([1326, 59])
                edge_hidden = self.edge_layers[i](line_graph, edge_hidden)
                # print(edge_hidden.size()) # torch.Size([1326, 128])
            hidden = self.layers[i](graph, layer_input, edge_hidden) # hidden: torch.Size([502, 128])
            # ieconv layer
            if self.use_ieconv:
                hidden = hidden + self.ieconvs[i](graph, layer_input, ieconv_edge_feature)
            hidden = self.dropout(hidden)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            if self.layer_norm:
                hidden = self.layer_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class HyboMessagePassing(nn.Module):

    def __init__(self, input_dim, num_relation):
        super(HyboMessagePassing, self).__init__()
        # print(input_dim, output_dim, num_relation, edge_input_dim, batch_norm, activation)
        # 512 512 7 None True relu
        self.num_relation = num_relation
        self.input_dim = input_dim

    def message(self, graph, input, edge_input):
        # input are node features staying in the tangent space of origin
        node_in = graph.edge_list[:, 0] # source target information
        message = input[node_in]
        if edge_input is not None:
            assert edge_input.shape == message.shape
            # print(message.size(), edge_input.size())
            # torch.Size([25458, 22]) torch.Size([25458, 22])
            message += edge_input
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        # print(node_out) # tensor([1592, 2401, 3200, ..., 190898, 199584, 199658], device='cuda:0')
        # give each edge with different edge types a different target index
        edge_weight = graph.edge_weight.unsqueeze(-1) # all 1
        # scatter_sum(src, index, dim, out, dim_size), message: information of source node, src: index of target node
        update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        # after scatter_add, the message (index) composition: node1+r1,..., node1+r7, ..., node N+r1,..., nodeN+r7
        # print(update.size()) # torch.Size([203664, 22])
        update = update.view(graph.num_node, self.num_relation, self.input_dim)
        return update

    def combine(self, update):
        # need to combine update (node1+r1,..., node1+r7) without dimension change
        output = torch.mean(update, dim=1)
        return output

    # line_graph, edge_hidden
    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        output = self.combine(update)
        return output


class GeometricRelationalGraphConv(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None,
                 batch_norm=False, activation="relu"):
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input, edge_input=None):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        if edge_input is not None:
            assert edge_input.shape == message.shape
            message += edge_input
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        output = self.combine(input, update)
        return output


# Note: this version uses torch_ext.sparse_coo_tensor_unsafe(indices, values, size) contained in original torchdrug repository to create adjacent matrix,
# which uses a PyTorch C++ extension just-in-time (JIT) to boost the calculation, requiring proper JIT configuration to run
@R.register("models.CG22_GearNet") # should be registered in torchdrug.models.__init__
class CG22_GeometryAwareRelationalGraphNeuralNetwork(nn.Module, core.Configurable):
    """
    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(CG22_GeometryAwareRelationalGraphNeuralNetwork, self).__init__()
        # print('input_dim, hidden_dims, num_relation, edge_input_dim, num_angle_bin:', input_dim, hidden_dims, num_relation, edge_input_dim, num_angle_bin)
        # input_dim, hidden_dims, num_relation, edge_input_dim, num_angle_bin: 39, [128, 128, 128, 128, 128, 128], 1, 53, 8
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin # for line graph of the original graph, thus num_angle_bin can still be applied
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm
        # print('self.dims, self.edge_dims:', self.dims, self.edge_dims)
        # self.dims, self.edge_dims: [39, 128, 128, 128, 128, 128, 128], [53, 39, 128, 128, 128, 128, 128]

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class IEConvLayer(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim, kernel_hidden_dim=32,
                 dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(IEConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = layers.MLP(edge_input_dim, [kernel_hidden_dim, (hidden_dim + 1) * hidden_dim])
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.update_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, graph, input, edge_input):
        node_in = graph.edge_list[:, 0]
        message = self.linear1(input[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_input).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]

        return message

    def aggregate(self, graph, message):
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = graph.edge_weight.unsqueeze(-1)

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, input, update):
        output = self.linear2(update)
        return output

    def forward(self, graph, input, edge_input):
        input = self.input_batch_norm(input)
        layer_input = self.dropout(self.activation(input))

        message = self.message(graph, layer_input, edge_input)
        update = self.aggregate(graph, message)
        update = self.dropout(self.activation(self.update_batch_norm(update)))

        output = self.combine(input, update)
        output = self.output_batch_norm(output)
        return output