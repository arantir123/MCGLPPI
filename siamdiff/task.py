import os
import glob
import math
import numpy as np

import torch
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_add

from torchdrug import core, tasks, layers, models, metrics, data
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.PIP")
class PIP(tasks.InteractionPrediction):

    def __init__(self, model, num_mlp_layer=1, verbose=0):
        super(PIP, self).__init__(model, model2=None, task="interaction", criterion="bce", 
            metric=("auroc", "acc"), num_mlp_layer=num_mlp_layer, normalization=False, 
            num_class=1, graph_construction_model=None, verbose=verbose)

    def preprocess(self, train_set, valid_set, test_set):
        weight = []
        for task, w in self.task.items():
            weight.append(w)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = [1]
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim + self.model2.output_dim, hidden_dims + [sum(self.num_class)])        

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        graph2 = batch["graph2"]
        output2 = self.model2(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        output1 = output1["node_feature"][graph1.ca_idx]
        output2 = output2["node_feature"][graph2.ca_idx]
        pred = self.mlp(torch.cat([output1, output2], dim=-1))
        return pred


@R.register("tasks.PSR")
class PSR(tasks.PropertyPrediction):

    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, verbose=0):
        super(PSR, self).__init__(model, task="gdt_ts", criterion="mse", 
            metric=("mae", "rmse", "spearmanr"), num_mlp_layer=num_mlp_layer, normalization=True, 
            num_class=1, graph_construction_model=graph_construction_model, verbose=verbose)   
    

@R.register("tasks.RES")
class RES(tasks.Task, core.Configurable):

    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, verbose=0):
        super(RES, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(model_output_dim, hidden_dims + [20])        

    def apply_mask(self, graph):
        residue_mask = scatter_sum(graph.ca_mask.float(), graph.atom2residue, dim=0, dim_size=graph.num_residue).bool()
        atom_mask = residue_mask[graph.atom2residue]

        graph.residue_type[residue_mask] = 0
        graph.atom_feature[atom_mask, -21:] = 0
        return graph

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        if graph.view == "residue":
            input = graph.node_feature.float()[graph.atom2residue]
        else:
            input = graph.node_feature.float()
        output = self.model(graph, input, all_loss=all_loss, metric=metric)
        output_feature = output["node_feature"] if graph.view in ["node", "atom"] else output.get("residue_feature", output.get("node_feature"))
        pred = self.mlp(output_feature)

        pred = pred[graph.ca_mask]
        target = graph.label
        assert pred.shape[0] == target.shape[0] == graph.batch_size

        return pred, target

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric

    def evaluate(self, pred, target):
        metric = {}

        score = metrics.accuracy(pred, target.long())

        name = tasks._get_metric_name("micro_acc")
        metric[name] = score
        return metric


@R.register("tasks.MSP")
class MSP(tasks.InteractionPrediction):

    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, verbose=0):
        super(MSP, self).__init__(model, model2=model, task="label", criterion="bce",
            metric=("auroc", "auprc"), num_mlp_layer=num_mlp_layer, normalization=False,
            num_class=1, graph_construction_model=graph_construction_model, verbose=verbose)

    def preprocess(self, train_set, valid_set, test_set):
        weight = []
        for task, w in self.task.items():
            weight.append(w)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = [1]
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim + self.model2.output_dim, hidden_dims + [sum(self.num_class)])  

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        if self.graph_construction_model:
            graph1 = self.graph_construction_model(graph1)
        graph2 = batch["graph2"]
        if self.graph_construction_model:
            graph2 = self.graph_construction_model(graph2)
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        output2 = self.model2(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        assert graph1.num_residue == graph2.num_residue
        residue_mask = graph1.residue_type != graph2.residue_type
        node_mask1 = residue_mask[graph1.atom2residue].float().unsqueeze(-1)
        output1 = scatter_add(output1["node_feature"] * node_mask1, graph1.atom2graph, dim=0, dim_size=graph1.batch_size) \
                / (scatter_add(node_mask1, graph1.atom2graph, dim=0, dim_size=graph1.batch_size) + 1e-10)
        node_mask2 = residue_mask[graph2.atom2residue].float().unsqueeze(-1)
        output2 = scatter_add(output2["node_feature"] * node_mask2, graph2.atom2graph, dim=0, dim_size=graph2.batch_size) \
                / (scatter_add(node_mask2, graph2.atom2graph, dim=0, dim_size=graph2.batch_size) + 1e-10)
        pred = self.mlp(torch.cat([output1, output2], dim=-1))
        return pred


@R.register("tasks.EC")
class EC(tasks.MultipleBinaryClassification):

    def __init__(self, model, task, num_mlp_layer=1, graph_construction_model=None, verbose=0):
        super(EC, self).__init__(model, task=task, criterion="bce",
            metric=("auprc@micro", "f1_max"), num_mlp_layer=num_mlp_layer, normalization=False,
            reweight=False, graph_construction_model=graph_construction_model, verbose=verbose)


@R.register("tasks.SiamDiff")
class SiamDiff(tasks.Task, core.Configurable):

    """
    Siamese Diffusion Trajectory Prediction.

    Parameters:
        model (nn.Module): the protein structure encoder to be pre-trained
        sigma_begin (float): the smallest noise scale
        sigma_end (float): the largest noise scale
        num_noise_level (int): the number of noise scale levels
        gamma (float, optional): controls the weights between sequence and structure denoising; 
            (1 - gamma) * seq_loss + gamma * struct_loss
        max_mask_ratio (float, optional): the maximum masking ratio in sequence diffusion
        num_mlp_layer (int, optional): the number of MLP layers for prediction head
        graph_construction_model (nn.Module, optional): graph construction model
        use_MI (bool, optional): whether to use mutual information maximization; if True, use SiamDiff; otherwise, use DiffPreT
    """

    num_class = constant.NUM_AMINO_ACID
    min_mask_ratio = 0.15
    eps = 1e-10

    # num_mlp_layer: 3
    # sigma_begin: 1.0e-3
    # sigma_end: 0.1
    # num_noise_level: 100
    # gamma: 0.5
    # use_MI: True
    def __init__(self, model, sigma_begin, sigma_end, num_noise_level, use_MI=True, gamma=0.5,
                max_mask_ratio=1.0, num_mlp_layer=2, graph_construction_model=None):
        super(SiamDiff, self).__init__()
        print('sigma_begin, sigma_end, num_noise_level, use_MI, gamma, max_mask_ratio, num_mlp_layer, graph_construction_model:',
              sigma_begin, sigma_end, num_noise_level, use_MI, gamma, max_mask_ratio, num_mlp_layer, graph_construction_model)
        # 0.001 0.1 100 True 0.5 1.0 3

        self.model = model
        # print('self.model:', self.model)
        self.num_noise_level = num_noise_level
        self.max_mask_ratio = max_mask_ratio
        self.use_MI = use_MI
        self.gamma = gamma # 0.5
        betas = torch.linspace(-6, 6, num_noise_level) # start: -6, end: 6, steps: 100
        betas = betas.sigmoid() * (sigma_end - sigma_begin) + sigma_begin
        # generating 100 beta values ranging from 0.0012 to 0.0998
        alphas = (1. - betas).cumprod(dim=0)
        # cumprod: torch.Tensor([1, 2, 3, 4, 5]) -> tensor([ 1., 2., 6., 24., 120.] (1，1×2，1×2×3，1×2×3×4，1×2×3×4×5)
        # print(alphas, len(alphas))
        # tensor([0.9988, 0.9975, ..., 0.0057, 0.0051]) -> gradually decreases after consecutive multiplication -> 100 values in total
        self.register_buffer("alphas", alphas)
        # alpha and beta are determined by num_noise_level (interval), sigma_start, sigma_end
        self.graph_construction_model = graph_construction_model
        
        output_dim = model.output_dim
        # print(output_dim) 128 * 6 = 768
        self.struct_mlp = layers.MLP(2 * output_dim, [output_dim] * (num_mlp_layer - 1) + [1])
        # (0): Linear(in_features=1536, out_features=768, bias=True)
        # (1): Linear(in_features=768, out_features=768, bias=True)
        # (2): Linear(in_features=768, out_features=1, bias=True)
        self.dist_mlp = layers.MLP(1, [output_dim] * (num_mlp_layer - 1) + [output_dim])
        # (0): Linear(in_features=1, out_features=768, bias=True)
        # (1): Linear(in_features=768, out_features=768, bias=True)
        # (2): Linear(in_features=768, out_features=768, bias=True)
        self.seq_mlp = layers.MLP(output_dim, [output_dim] * (num_mlp_layer - 1) + [self.num_class])
        # (0): Linear(in_features=768, out_features=768, bias=True)
        # (1): Linear(in_features=768, out_features=768, bias=True)
        # (2): Linear(in_features=768, out_features=20, bias=True)

    def add_seq_noise(self, graph, noise_level):
        num_nodes = graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        # print(num_nodes, num_cum_nodes) # tensor([100, 100,  74, 100, 100,  98, 100,  51], device='cuda:0') tensor([100, 200, 274, 374, 474, 572, 672, 723], device='cuda:0')

        # decide the mask rate according to the noise level
        # max_ratio: 1, min_ratio: 0.15, the mask rate increase with the increase of the noise level
        mask_rate = (self.max_mask_ratio - self.min_mask_ratio) * ((noise_level + 1) / self.num_noise_level) + self.min_mask_ratio
        num_samples = (num_nodes * mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        # print(num_samples, num_sample) # tensor([50, 98, 68, 77, 60, 18, 18, 22], device='cuda:0') tensor(411, device='cuda:0')
        # numbers of AA type tags to be masked
        sample2graph = torch.repeat_interleave(num_samples)
        # assign residues to be masked into the corresponding proteins in current batch

        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        # print(num_nodes[sample2graph]) # 是按序给出每一个要mask的残基所对应的蛋白质中残基总数
        # print(node_index) # 是按序给出每一个要mask的残基在所对应的蛋白质中的相对序号（序号基于torch.randn抽取的正态分布给出）

        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        # 将按序给出的每一个要mask的残基所对应的蛋白质中的相对序号变为在当前batch中的绝对序号
        # 获得了绝对序号后，因为可以同时获得当前batch内所有蛋白的按序AA tag (graph.residue_type)，所以可以通过这里获得的node_index来mask对应的residue tag (整体逻辑和hybopp选subgraph有些类似)
        node_index = node_index.clamp(max=num_cum_nodes[-1]-1)

        seq_target = graph.residue_type[node_index].clone()
        # clone()函数可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中
        # clone操作在不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下
        selected_residue = torch.zeros((graph.num_residue,), dtype=torch.bool, device=graph.device)
        selected_residue[node_index] = 1 # generate AA sequence mask for masked AA in current batch
        # print(seq_target.size(), seq_target.min(), seq_target.max())
        # torch.Size([411]) tensor(0, device='cuda:0') tensor(19, device='cuda:0'), num_sample: 411

        # print(graph.atom_name, graph.atom_name.size()) # tensor([17,  1,  0,  ..., 24, 14, 34], device='cuda:0') torch.Size([5710])
        # print(graph.atom_name2id) # {'OE1': 29, 'OE2': 30, 'OG': 31, 'OG1': 32, 'OH': 33, 'OXT': 34, 'SD': 35, 'SG': 36, 'UNK': 37}

        # only keep backbone atoms of the selected residues
        node_mask = (graph.atom_name == graph.atom_name2id["CA"]) \
                  | (graph.atom_name == graph.atom_name2id["C"]) \
                  | (graph.atom_name == graph.atom_name2id["N"]) \
                  | ~selected_residue[graph.atom2residue]
        # print(graph.atom2residue, graph.atom2residue.size())
        # print(node_mask, node_mask.size())
        # tensor([  0,   0,   0,  ..., 722, 722, 722], device='cuda:0') torch.Size([5710])
        # tensor([True, True, True,  ..., True, True, True], device='cuda:0') torch.Size([5710])

        # absolute ids for masked residues in current batch, boolean mask for unmasked/remained atoms in current batch, residue types for current masked residues
        return node_index, node_mask, seq_target

    def add_struct_noise(self, graph, noise_level):
        # add noise to coordinates and change the pairwise distance in edge features if neccessary

        # cumprod: torch.Tensor([1, 2, 3, 4, 5]) -> tensor([ 1., 2., 6., 24., 120.] (1，1×2，1×2×3，1×2×3×4，1×2×3×4×5)
        # alpha = 1 - beta, beta contains 100 values drawn from the same distribution (100 is the pre-defined noise level values, should also be the step number in original paper)
        a_graph = self.alphas[noise_level]      # (num_graph,)
        a_pos = a_graph[graph.node2graph]
        # print(a_graph, graph.node2graph, a_pos, a_graph.size(), graph.node2graph.size(), a_graph.size())
        # tensor([0.7373, 0.0063, 0.0119, 0.0837, 0.4252, 0.9934, 0.9948, 0.8674],
        #        device='cuda:0') tensor([0, 0, 0,  ..., 7, 7, 7], device='cuda:0') tensor([0.7373, 0.7373, 0.7373,  ..., 0.8674, 0.8674, 0.8674], device='cuda:0') torch.Size([8]) torch.Size([4268]) torch.Size([8])

        # print(graph.edge2graph, graph.edge2graph.size()) # tensor([0, 0, 0,  ..., 7, 7, 7], device='cuda:0') torch.Size([50028]), similar to graph.node2graph
        # a_graph值应该相当于当前batch每个graph所对应的抽取的noise level下的alpha（连乘）值，然后这里应该是把每个graph对应的alpha值分配到对应的点和边种
        a_edge = a_graph[graph.edge2graph]
        node_in, node_out = graph.edge_list.t()[:2]
        dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) # 计算边中两点的欧式距离

        perturb_noise = torch.randn_like(graph.node_position) # generate perturb noise standard normal distribution

        # add noise to atom coordinates based on current beta + noise level + value drawn from the standard normal distribution
        graph.node_position = graph.node_position + perturb_noise * ((1.0 - a_pos).sqrt() / a_pos.sqrt()).unsqueeze(-1) # (1)注意perturb_noise后面的是乘号，符合公式in（https://zhuanlan.zhihu.com/p/576475987）
        perturbed_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        if self.graph_construction_model and self.graph_construction_model.edge_feature == "gearnet":
            # print(graph.edge_feature, graph.edge_feature.size())
            # tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 1.4616],
            #         [1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 2.4494],
            #         [1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 3.5673],
            #         ...,
            #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 1.2474],
            #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 3.1129],
            #         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 2.2418]],
            #        device='cuda:0') torch.Size([50028, 53])
            graph.edge_feature[:, -1] = perturbed_dist # update the Euclidean distance included in edge features (as the original atom coordinates are corrupted)
        # (2)注意(dist - perturbed_dist)后面的是除号，感觉是上面(1)的逆操作，也就是在(1)时是先抽取一个服从标准正态分布的噪声，然后基于timestep生成最后噪声，加在当前的3D坐标上（最终可使坐标的分布接近于正态分布）
        # 而这里是产生需要模型最终去预测的噪声，需要是一个标准正态分布，所以使用上述的逆过程将dist - perturbed_dist变得接近于正态分布（随后在外部就会经过equivariant transform后就被用于监督label）
        struct_target = (dist - perturbed_dist) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        # print(a_edge, a_edge.size()) # tensor([0.7373, 0.7373, 0.7373,  ..., 0.8674, 0.8674, 0.8674], device='cuda:0') torch.Size([50028])

        # corrupted graphs with corrupted features, Euclidean distance between corrupted edges, difference between original edge distance and corrupted edge distance (normalized by current noise scale)
        return graph, perturbed_dist, struct_target

    def eq_transform(self, score_d, graph):
        # transform invariant scores on edges to equivariant coordinates on nodes
        # struct_pred1 = self.eq_transform(struct_pred1, graph1)
        node_in, node_out = graph.edge_list.t()[:2]
        diff = graph.node_position[node_in] - graph.node_position[node_out]
        dd_dr = diff / (diff.norm(dim=-1, keepdim=True) + 1e-10) #  # 经过normalize的原始边欧式距离
        score_pos = scatter_mean(dd_dr * score_d.unsqueeze(-1), node_in, dim=0, dim_size=graph.num_node) \
                + scatter_mean(- dd_dr * score_d.unsqueeze(-1), node_out, dim=0, dim_size=graph.num_node)
        return score_pos

    def struct_predict(self, output, graph, perturbed_dist):
        # predict scores on edges with node representations and perturbed distance (following previous work)

        # atom embedding of (corrupted) orginal graph or conformer graph, graph1 (after corruption, 只用于提供原始图边关系), Euclidean distances between corrupted edges in graph1
        # 感觉像是直接使用原始图或conformer图（腐蚀后）的atom embedding + 腐蚀的原始图结构 + 原始图中每条边的的欧式腐蚀距离，来预测腐蚀前后的分数差值
        node_in, node_out = graph.edge_list.t()[:2]
        dist_pred = self.dist_mlp(perturbed_dist.unsqueeze(-1))
        edge_pred = self.struct_mlp(torch.cat((output[node_in] * output[node_out], dist_pred), dim=-1))
        pred = edge_pred.squeeze(-1)
        return pred

    def seq_predict(self, graph, output, node_index):
        # node_index: absolute ids for masked residues in current batch, output2: conformer或者原始蛋白的经过encoder的atom-level embeddings (after corruption，同上)，可以使用原始蛋白，也可以使用conformer
        # graph1: 经过腐蚀的原始蛋白图
        node_feature = scatter_mean(output, graph.atom2residue, dim=0, dim_size=graph.num_residue)[node_index]
        # node_feature应该是所有masked的residue的embedding（embedding通过腐蚀的原始蛋白或者其conformer获得）
        seq_pred = self.seq_mlp(node_feature)
        return seq_pred

    def predict_and_target(self, batch, all_loss=None, metric=None):
        # the outer training loop is same to ordinary model training, all diffusion function seems to be defined here
        # print(batch, all_loss, metric) # including graph1 and graph2

        # ** no graph_construction node_layer is used here (None) **
        # ** other the noise added during the diffusion process, another certainty added during the pre-training process is TruncateProtein in dataset ** 
        # ** which randomly truncate proteins into 100 AAs in different epochs **
        graph1 = batch["graph"]
        if self.graph_construction_model:
            # only atom-scale spatial graph is used, in which features are still based on GearNet (based on AA type and atom pair relative postion, etc)
            graph1 = self.graph_construction_model.apply_node_layer(graph1)
        
        if self.use_MI:
            graph2 = batch["graph2"]
            graph2.view = graph1.view
            # print(graph1.view, graph2.view) # atom atom, initialized in Protein Class
            if self.graph_construction_model:
                graph2 = self.graph_construction_model.apply_node_layer(graph2) 

        # diffusion hyperparameters are defined in task wrapper
        # tensor([0.9988, 0.9975, ..., 0.0057, 0.0051]) -> gradually decreases after consecutive multiplication -> 100 values in total
        # self.register_buffer("alphas", alphas)
        noise_level = torch.randint(0, self.alphas.shape[0], (graph1.batch_size,), device=self.device) # (num_graph, )
        # print(noise_level, noise_level.size()) # tensor([41, 97, 91, 72, 53,  4,  3, 33], device='cuda:0') torch.Size([8]), bs=8

        # 1和2更像是前向阶段，3和4更像是后向阶段

        # 1. add shared sequence noise, the noise level for sequence and structure are the same
        # ** 给我的感觉是这样的：由于当前的模型是atom-level的模型，所以若要对序列和结构同时加噪声，这里就先对序列加噪声，后对结构加噪声 **
        # ** 对序列加噪声的方式是，先得到一个atom-level的mask，指示了当前batch中哪些residue被屏蔽掉哪些被保留（基于当前noise_level），基于这个mask去获取没被屏蔽的residue所对应的原子图和特征 **
        # ** 所以感觉最后AA加噪变成了原子图屏蔽的过程（若引入CG尺度，可能做法是类似的）**
        if self.gamma < 1.0:
            # absolute ids for masked residues in current batch, boolean mask for unmasked/remained atoms in current batch, residue types for current masked residues
            node_index, node_mask, seq_target = self.add_seq_noise(graph1, noise_level)

            # atom feature initialization in the used 'AtomFeature' transform function:
            # if self.atom_feature == "residue_symbol":
            #     atom_feature = torch.cat([
            #         functional.one_hot(graph.atom_type.clamp(max=17), 18),
            #         functional.one_hot(graph.residue_type[graph.atom2residue], 21)], dim=-1)
            if graph1.view in ["node", "atom"]:
                graph1.atom_feature[node_mask, -21:] = 0 # ** 仅保留原子类型特征，不使用残基类型特征，可能是因为在AA masking中希望不在节点特征处暴露AA类型信息 **
            # print(node_mask, node_mask.sum()) # tensor([True, True, True,  ..., True, True, True], device='cuda:0') tensor(4268, device='cuda:0')
            # print(graph1.atom_feature, graph1.atom_feature.size()) # torch.Size([5710, 39])

            # print(graph1.num_residue, graph1.num_node) # tensor(723, device='cuda:0') tensor(5710, device='cuda:0')
            graph1 = graph1.subgraph(node_mask) # *** this function seems not to influence the residue-level information under current setting ***
            # print(graph1.num_residue, graph1.num_node) # tensor(723, device='cuda:0') tensor(4268, device='cuda:0')
            # print(graph1.atom_feature, graph1.atom_feature.size()) # torch.Size([4268, 39])
            # *** retain the graph with unmasked atoms in which only atom_type (atom) node features are retained ***

            # also close the residue_feature and residue_type for masked residues in current batch
            with graph1.residue():
                graph1.residue_feature[node_index] = 0
                graph1.residue_type[node_index] = 0
            # print(graph1.residue_feature.size(), graph1.residue_type.size()) # torch.Size([723, 21]) torch.Size([723]), keeping the original tensor size before graph1.subgraph function
            if self.use_MI:
                if graph2.view in ["node", "atom"]:
                    graph2.atom_feature[node_mask, -21:] = 0
                graph2 = graph2.subgraph(node_mask)
                with graph2.residue():
                    graph2.residue_feature[node_index] = 0
                    graph2.residue_type[node_index] = 0

        # construct edges and edge features
        # this should be done before structure perturbation
        # 所以从这里可以看出graph1和graph2的边应该是不一样的
        if self.graph_construction_model:
            # construct the spatial edge (SpatialEdge())
            graph1 = self.graph_construction_model.apply_edge_layer(graph1)
        if self.use_MI and self.graph_construction_model:
            graph2 = self.graph_construction_model.apply_edge_layer(graph2)

        # 2. add structure noise
        if self.gamma > 0.0: # gamma is used for controlling weight between structural loss and sequential loss
            # print(noise_level, noise_level.size()) # tensor([41, 97, 91, 72, 53,  4,  3, 33], device='cuda:0') torch.Size([8]), bs=8
            # seems for each sample in different epochs, a noise level ranging from 0 to 100 (predefined) will be selected
            # due to the property of the diffusion model, the diffusion calculation can be normally performed no matter how noise level is selected
            # for graph and graph2, the same noise level will be used

            # corrupted graphs with corrupted features, Euclidean distances between corrupted edges, difference between original edge distance and corrupted edge distance (normalized by current noise scale)
            graph1, perturbed_dist1, struct_target1 = self.add_struct_noise(graph1, noise_level)
            if self.use_MI:
                graph2, perturbed_dist2, struct_target2 = self.add_struct_noise(graph2, noise_level)

        # 此时已完成腐蚀
        # print(graph1.node_feature.float().size(), graph2.node_feature.float().size())
        # torch.Size([4268, 39]) torch.Size([4268, 39])
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss, metric)["node_feature"]
        if self.use_MI:
            output2 = self.model(graph2, graph2.node_feature.float(), all_loss, metric)["node_feature"]
        else:
            output2 = output1
        # print(output1.size(), output2.size()) # torch.Size([4268, 768]) torch.Size([4268, 768])

        # 3. predict structure noise
        # Therefore, we adopt the chain-rule approach proposed in Xu et al. [81], which decomposes the noise on pairwise distances to obtain the modified noise vector ˆϵ as supervision
        # 应该就是还是需要利用MLP重构误差，但是基于的是成对距离信息等
        if self.gamma > 0.0:
            # get invariant scores on edges instead of nodes
            # following https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/dualenc.py#L305-L308
            # 所以看起来，使用不同encoder应该都是可以和diffusion模型直接匹配的
            # 从这里开始graph1和graph2（若设定了conformer）都是经过腐蚀的

            # ** atom embedding of (corrupted) orginal graph or conformer graph, graph1 (after corruption), Euclidean distances between corrupted edges in graph1 **
            # 下面4与这里相同，若不使用conformer（腐蚀图），就使用原始的腐蚀图，若使用conformer就使用conformer的腐蚀图以传递conformer的信息
            struct_pred1 = self.struct_predict(output2, graph1, perturbed_dist1) # (1)根据下面的注释(2)判断，应该这一步struct_pred1是利用原始图以及corrupted的节点表示预测出的原始边距离与corrupted边距离之间的差值（用于产生监督损失，这也是diffusion去噪的目的）
            struct_pred1 = self.eq_transform(struct_pred1, graph1) # eq_transform:  equivariant transform, make the loss function invariant w.r.t. Rt
            struct_target1 = self.eq_transform(struct_target1, graph1) # (2)struct_target1: difference between original edge distance and corrupted edge distance (normalized by current noise scale)
            loss1 = 0.5 * ((struct_pred1 - struct_target1) ** 2).sum(dim=-1)
            # print(loss1, loss1.size()) # torch.Size([4268]), 可以看出来上述的监督结构损失是atom-level（每一个原子都有一个值）
            loss1 = scatter_mean(loss1, graph1.node2graph, dim=0, dim_size=graph1.batch_size).mean() # 将atom-level的损失平均分配到每一张蛋白质图上
            # 对于当前蛋白质的conformer，也生成一个类似的损失
            if self.use_MI:
                struct_pred2 = self.struct_predict(output1, graph2, perturbed_dist2) # output1: graph1和graph2互为conformer
                struct_pred2 = self.eq_transform(struct_pred2, graph2)
                struct_target2 = self.eq_transform(struct_target2, graph2)

                loss2 = 0.5 * ((struct_pred2 - struct_target2) ** 2).sum(dim=-1) 
                loss2 = scatter_mean(loss2, graph2.node2graph, dim=0, dim_size=graph2.batch_size).mean()
            else:
                loss2 = loss1
            metric["structure denoising loss"] = loss1 + loss2
            all_loss += self.gamma * (loss1 + loss2)
            pred, target = struct_pred1, struct_target1

        # 4. predict sequence noise
        if self.gamma < 1.0:
            # node_index: absolute ids for masked residues in current batch, output2: conformer或者原始蛋白的经过encoder的atom-level embeddings (after corruption，同上)，可以使用原始蛋白，也可以使用conformer
            # graph1: 经过腐蚀的原始蛋白
            seq_pred1 = self.seq_predict(graph1, output2, node_index) # seq_pred1是masked residue embeddings送入MLP的结果
            loss1 = 0.5 * F.cross_entropy(seq_pred1, seq_target, reduction="none") # seq_target是residue types for current masked residues，所以这个任务就是预测被mask掉的residue类型
            loss1 = scatter_mean(loss1, graph1.residue2graph[node_index], dim=0, dim_size=graph1.batch_size).mean() # 把损失聚合到protein上再取平均（作为该batch的序列重建损失）
            acc1 = (seq_pred1.argmax(dim=-1) == seq_target).float().mean() # 计算了一个准确率
            if self.use_MI:
                seq_pred2 = self.seq_predict(graph2, output1, node_index)
                loss2 = 0.5 * F.cross_entropy(seq_pred2, seq_target, reduction="none")
                loss2 = scatter_mean(loss2, graph2.residue2graph[node_index], dim=0, dim_size=graph2.batch_size).mean()
        
                acc2 = (seq_pred2.argmax(dim=-1) == seq_target).float().mean()
            else:
                loss2 = loss1
                acc2 = acc1
            metric["sequence denoising accuracy"] = 0.5 * (acc1 + acc2)
            metric["sequence denoising loss"] = loss1 + loss2
            all_loss += (1 - self.gamma) * (loss1 + loss2)
            pred, target = seq_pred1, seq_target

        metric["loss"] = all_loss
        return pred, target

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        # print(all_loss, metric)
        # tensor(4.1210, device='cuda:0', grad_fn=<AddBackward0>) {'structure denoising loss': tensor(5.2460, device='cuda:0', grad_fn=<AddBackward0>),
        # 'sequence denoising accuracy': tensor(0.0693, device='cuda:0'), 'sequence denoising loss': tensor(2.9959, device='cuda:0', grad_fn=<AddBackward0>),
        # 'loss': tensor(4.1210, device='cuda:0', grad_fn=<AddBackward0>)}
        return all_loss, metric
