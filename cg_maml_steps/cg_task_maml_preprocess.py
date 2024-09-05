import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_scatter import scatter_sum, scatter_mean, scatter_add

from torchdrug.models import GearNet
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug import core, tasks, layers, models, metrics, data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# basic parameters for LJ and electrostatic force calculation
# mapping from elements in martini22_name2id (cg_protein) to elements in following matrices
bead2bead_ = {0: 17, 1: 16, 2: 15, 3: 13, 4: 12, 5: 11, 6: 10, 7: 9, 8: 8,
              9: 5, 10: 4, 11: 2, 12: 1, 13: 14, 14: 13, 15: 10, 16: 8}
bead2bead_ = [bead2bead_[i] for i in range(len(bead2bead_))]

# well depth unit: kJ/mol
level0, level1, level2, level3, level4, level5, level6, level7, level8, level9 = 5.6, 5.0, 4.5, 4.0, 3.5, 3.1, 2.7, 2.3, 2.0, 2.0
CG_well_depth_mat = [
    [level0, level0, level0, level2, level0, level0, level0, level1, level1, level1, level1, level1, level4, level5, level6, level7, level9, level9],    # Qda, 0
    [level0, level1, level0, level2, level0, level0, level0, level1, level1, level1, level3, level1, level4, level5, level6, level7, level9, level9],    # Qd, 1
    [level0, level0, level1, level2, level0, level0, level0, level1, level1, level1, level1, level3, level4, level5, level6, level7, level9, level9],    # Qa, 2
    [level2, level2, level2, level4, level1, level0, level1, level2, level3, level3, level3, level3, level4, level5, level6, level7, level9, level9],    # Q0, 3
    [level0, level0, level0, level1, level0, level0, level0, level0, level0, level1, level1, level1, level4, level5, level6, level6, level7, level8],    # P5, 4
    [level0, level0, level0, level0, level0, level1, level1, level2, level2, level3, level3, level3, level4, level5, level6, level6, level7, level8],    # P4, 5
    [level0, level0, level0, level1, level0, level1, level1, level2, level2, level2, level2, level2, level4, level4, level5, level5, level6, level7],    # P3, 6
    [level1, level1, level1, level2, level0, level2, level2, level2, level2, level2, level2, level2, level3, level4, level4, level5, level6, level7],    # P2, 7
    [level1, level1, level1, level3, level0, level2, level2, level2, level2, level2, level2, level2, level3, level4, level4, level4, level5, level6],    # P1, 8
    [level1, level1, level1, level3, level1, level3, level2, level2, level2, level2, level2, level2, level4, level4, level5, level6, level6, level6],    # Nda, 9
    [level1, level3, level1, level3, level1, level3, level2, level2, level2, level2, level3, level2, level4, level4, level5, level6, level6, level6],    # Nd, 10
    [level1, level1, level3, level3, level1, level3, level2, level2, level2, level2, level2, level3, level4, level4, level5, level6, level6, level6],    # Na, 11
    [level4, level4, level4, level4, level4, level4, level4, level3, level3, level4, level4, level4, level4, level4, level4, level4, level5, level6],    # N0, 12
    [level5, level5, level5, level5, level5, level5, level4, level4, level4, level4, level4, level4, level4, level4, level4, level4, level5, level5],    # C5, 13
    [level6, level6, level6, level6, level6, level6, level5, level4, level4, level5, level5, level5, level4, level4, level4, level4, level5, level5],    # C4, 14
    [level7, level7, level7, level7, level6, level6, level5, level5, level4, level6, level6, level6, level4, level4, level4, level4, level4, level4],    # C3, 15
    [level9, level9, level9, level9, level7, level7, level6, level6, level5, level6, level6, level6, level5, level5, level5, level4, level4, level4],    # C2, 16
    [level9, level9, level9, level9, level8, level8, level7, level7, level6, level6, level6, level6, level6, level5, level5, level4, level4, level4],    # C1, 17
]

# vdw radius unit: A (rather than nM used in original MARTINI paper)
CG_vdw_radius_mat = [
    # Qda, Qd,  Qa,  Q0,  P5,  P4,  P3,  P2,  P1, Nda,  Nd,  Na,  N0,  C5,  C4,  C3,  C2,  C1
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 6.2, 6.2],    # Qda
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 6.2, 6.2],    # Qd
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 6.2, 6.2],    # Qa
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 6.2, 6.2],    # Q0
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # P5
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # P4
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # P3
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # P2
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # P1
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # Nda
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # Nd
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # Na
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # N0
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # C5
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # C4
    [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # C3
    [6.2, 6.2, 6.2, 6.2, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # C2
    [6.2, 6.2, 6.2, 6.2, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7],    # C1
]

# the ES force only exists between Qa-Qa, Qa-Qd, Qd-Qa, and Qd-Qd bead pair combinations
# current beadpair_type is the updated version for energy coefficient retrival, Qa: 2 (old: 11), Qd: 1 (old: 12)
# q constant for Qa: -1 and q constant for Qd: 1 (set to 0 for other types)
CG_q_constant_mat = [0, 1, -1] # pos 0: other, pos 1: Qd, pos 2: Qa


@R.register("tasks.CGDiff")
# ** in order to make load_config_dict identify this registered function, need to import CGDiff in the main function **
class CGDiff(tasks.Task, core.Configurable):

    """
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

    num_class = constant.NUM_AMINO_ACID # 20
    min_mask_ratio = 0.15
    eps = 1e-10

    def __init__(self, model, sigma_begin, sigma_end, num_noise_level, use_MI=True, gamma=0.5,
                max_mask_ratio=1.0, num_mlp_layer=2, graph_construction_model=None, seq_bb_retain=True, bb4aa_feats_mask=False, angle_enhance=True):
        super(CGDiff, self).__init__()
        # print(sigma_begin, sigma_end, num_noise_level, use_MI, gamma, max_mask_ratio, num_mlp_layer)
        # 0.001, 0.1, 100, True, 0.5, 1.0, 3

        self.model = model
        self.num_noise_level = num_noise_level # the number of steps for diffusion process
        self.max_mask_ratio = max_mask_ratio # the maximum masking ratio in sequence diffusion
        self.use_MI = use_MI # whether to use conformers in diffusion process
        self.gamma = gamma # control the weight between structural loss and sequential loss
        # assert self.gamma >= 0 and self.gamma <= 1, 'current gamma is not in the range of [0, 1]: {}'.format(self.gamma)
        # 1.1 is for activating structural diffusion while deactivating sequential diffusion
        assert self.gamma >= 0 and self.gamma <= 1.1, 'current gamma is not in the range of [0, 1.1]: {}'.format(self.gamma)
        self.seq_bb_retain = seq_bb_retain # whether also to retain all backbone beads in forward sequence diffusion (rather than only retaining all beads in unmasked residues)
        self.bb4aa_feats_mask = bb4aa_feats_mask # when seq_bb_retain = True, considering whether BB type features of remained BB nodes of masked AAs will be closed
        self.angle_enhance = angle_enhance # whether to use the angle information provided in itp file as auxiliary bead node features

        # ** alpha and beta are overall determined by num_noise_level (interval/diffusion steps), sigma_start, sigma_end **
        # betas are a series of fixed variances based on start: -6, end: 6, and step number
        betas = torch.linspace(-6, 6, num_noise_level)
        # transform the uniform-gap values above into real fixed variances using pre-defined sigma_begin and sigma_end
        # default: generating 100 beta values ranging from 0.0012 to 0.0998
        betas = betas.sigmoid() * (sigma_end - sigma_begin) + sigma_begin
        # generate \hat{alpha} for each diffusion steps (the value for specific step can be retrieved via index directly)
        # e.g., step 50 \hat{alpha}: consecutive multiplication of former 50 alpha values
        alphas = (1. - betas).cumprod(dim=0)
        # cumprod: torch.Tensor([1, 2, 3, 4, 5]) -> tensor([ 1., 2., 6., 24., 120.] (1，1×2，1×2×3，1×2×3×4，1×2×3×4×5)
        # alphas: tensor([0.9988, 0.9975, ..., 0.0057, 0.0051]) -> gradually decreases after consecutive multiplication -> 100 values in total
        self.register_buffer("alphas", alphas)

        self.graph_construction_model = graph_construction_model
        
        output_dim = model.output_dim # 128 (hidden_dims) * 6 (layer_num) = 768
        self.struct_mlp = layers.MLP(2 * output_dim, [output_dim] * (num_mlp_layer - 1) + [1]) # for final structural loss
        # (0): Linear(in_features=1536, out_features=768, bias=True)
        # (1): Linear(in_features=768, out_features=768, bias=True)
        # (2): Linear(in_features=768, out_features=1, bias=True)
        self.dist_mlp = layers.MLP(1, [output_dim] * (num_mlp_layer - 1) + [output_dim]) # for structural loss (generating pairwise distance prior to struct_mlp)
        # (0): Linear(in_features=1, out_features=768, bias=True)
        # (1): Linear(in_features=768, out_features=768, bias=True)
        # (2): Linear(in_features=768, out_features=768, bias=True)
        self.seq_mlp = layers.MLP(output_dim, [output_dim] * (num_mlp_layer - 1) + [self.num_class]) # for final sequential loss
        # (0): Linear(in_features=768, out_features=768, bias=True)
        # (1): Linear(in_features=768, out_features=768, bias=True)
        # (2): Linear(in_features=768, out_features=20, bias=True)

    # ** it is not consecutive to mask residues in protein sequences **
    # ** sample the masked residues for each protein independently **
    # ** add_seq_noise will not change the current graph, just returning the mask for the following subgraph function **
    def add_seq_noise(self, graph, noise_level):
        # crop the graph based on residue numbers
        num_nodes = graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        # print(num_nodes, num_cum_nodes) # tensor([100, 100, 74, 100, 100, 98, 100, 51], device='cuda:0') tensor([100, 200, 274, 374, 474, 572, 672, 723], device='cuda:0')

        # decide the mask rate according to the noise level
        # max_ratio: 1, min_ratio: 0.15, the mask rate increase with the increase of the noise level
        mask_rate = (self.max_mask_ratio - self.min_mask_ratio) * ((noise_level + 1) / self.num_noise_level) + self.min_mask_ratio

        num_samples = (num_nodes * mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        # print(num_samples, num_sample) # tensor([50, 98, 68, 77, 60, 18, 18, 22], device='cuda:0') tensor(411, device='cuda:0')
        # i.e., numbers of AA type tags to be masked

        sample2graph = torch.repeat_interleave(num_samples)
        # assign residues to be masked into the corresponding proteins in current batch

        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        # print(num_nodes[sample2graph]) # 是按序给出每一个要mask的残基所对应的蛋白质中残基总数
        # print(node_index) # 是按序给出每一个要mask的残基在所对应的蛋白质中的相对序号（序号基于torch.randn抽取的正态分布给出）

        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        # 将按序给出的每一个要mask的残基在所对应的蛋白质中的相对序号变为在当前batch中的绝对序号
        # 获得了绝对序号后，因为可以同时获得当前batch内所有蛋白的按序AA tag (graph.residue_type)，所以可以通过这里获得的node_index来mask对应的residue tag
        node_index = node_index.clamp(max=num_cum_nodes[-1] - 1)
        # print(node_index, node_index.size(), graph.num_residue)
        # tensor([97, 79, 46, 32, 33, ..., 301, 290, 272, 288, 296]), torch.Size([249]), tensor(322) -> residue-level mask

        # seq_target应该是sequence diffusion需要被recover的ground truth
        # seq_target.size, seq_target.min, seq_target.max: 411, 0, 19
        seq_target = graph.residue_type[node_index].clone()

        # clone函数可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中
        # clone操作在不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下
        selected_residue = torch.zeros((graph.num_residue,), dtype=torch.bool, device=graph.device)

        # generate AA sequence mask for masked AA in current batch (the mask has the size of total AA number in current batch)
        # 1 represents the masked residues
        selected_residue[node_index] = 1

        # ** original implementation under atom-level: retain CA+C+N of all residues + side chain atoms of retained residues **
        # atom_name is based on the atom names in all types of natural AAs of proteins (see graph.atom_name2id)
        # print(graph.atom_name, graph.atom_name.size()) # tensor([17,  1,  0,  ..., 24, 14, 34], device='cuda:0') torch.Size([5710])
        # print(graph.atom_name2id) # {'OE1': 29, 'OE2': 30, 'OG': 31, 'OG1': 32, 'OH': 33, 'OXT': 34, 'SD': 35, 'SG': 36, 'UNK': 37}
        # node_mask = (graph.atom_name == graph.atom_name2id["CA"]) \
        #           | (graph.atom_name == graph.atom_name2id["C"]) \
        #           | (graph.atom_name == graph.atom_name2id["N"]) \
        #           | ~selected_residue[graph.atom2residue]

        # ** current implementation: no atom_name recorded in CG22_Protein class, in current logic, all bead nodes in retained residues will be retained **
        # ** another alternative: all BB nodes in every protein will also be retained (approximately equivalent to all CA+C+N above) ***
        # True represents the retained bead nodes of residues
        if self.seq_bb_retain:
            # bead_type/atom_type: bead (0), res (1), bead_pos (2)
            # selected_residue is residue-level, thus it can mask all side chain bead nodes for the masked residues
            node_mask = (graph.atom_type[:, 2] == graph.martini22_beadpos2id['BB']) | ~selected_residue[graph.bead2residue]
            # (graph.atom_type[:, 2] == graph.martini22_beadpos2id['BB'])返回一个bead_num大小的mask，其中1返回所有的BB bead（包括unmask和mask的残基中的BB bead）
            # ~selected_residue[graph.bead2residue]给定一个bead_num大小的mask，其中1代表未被mask的残基中的bead
            # 若想获得被mask掉的残基中的BB节点，需要(graph.atom_type[:, 2] == graph.martini22_beadpos2id['BB'])与selected_residue[graph.bead2residue]取交集
        else:
            node_mask = ~selected_residue[graph.bead2residue]

        # only for case seq_bb_retain == True + bb4aa_feats_mask = True
        BB4maskedAA = (graph.atom_type[:, 2] == graph.martini22_beadpos2id['BB']) & selected_residue[graph.bead2residue]

        # (1) absolute ids for masked residues in current batch, (2) boolean mask for unmasked/remained atom/cg nodes in current batch,
        # (3) residue types for current masked residues, (4) mask for BB nodes in masked AAs (with the size of original bead_num, True for these nodes)
        return node_index, node_mask, seq_target, BB4maskedAA

    def add_struct_noise(self, graph, noise_level):
        # ** add noise to coordinates and change the pairwise distance in edge features if necessary (using edge distance difference as the denoising objective) **
        # ** as the Euclidean distance difference is insensitive to translation and rotation **

        # cumprod: torch.Tensor([1, 2, 3, 4, 5]) -> tensor([ 1., 2., 6., 24., 120.] (1，1×2，1×2×3，1×2×3×4，1×2×3×4×5)
        # alpha = 1 - beta, beta contains 100 values drawn from the same distribution (100 is the pre-defined noise-level value and is also refered as the step number in original paper)
        a_graph = self.alphas[noise_level] # (num_graph, ), alphas contains the cumprod alpha values for each protein in current batch
        a_pos = a_graph[graph.node2graph] # get the noise level assigned for each node for every protein in current batch (noise-level for the same protein will be the same)

        # print(graph.edge2graph, graph.edge2graph.size()) # tensor([0, 0, 0,  ..., 7, 7, 7]) torch.Size([50028]), similar to graph.node2graph
        a_edge = a_graph[graph.edge2graph] # get the alphas value for each edge
        node_in, node_out = graph.edge_list.t()[:2]
        dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        perturb_noise = torch.randn_like(graph.node_position) # generate perturb noise from standard normal distribution (added to every coordinate dim of every protein graph node)

        # add noise to atom coordinates based on current beta + noise level + value drawn from the standard normal distribution
        # ** alphas decreases with the increase of the noise-level, thus the larger the noise-level is, the scale of added noise could be larger (more like disordered noise) **
        graph.node_position = graph.node_position + perturb_noise * ((1.0 - a_pos).sqrt() / a_pos.sqrt()).unsqueeze(-1) # (1)注意perturb_noise后面的是乘号，符合diffusion基本公式（https://zhuanlan.zhihu.com/p/576475987）
        perturbed_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) # all edge distance after the corruption

        # add the support of edge feature update for corruption to all 'gearnet-like' edge features
        if (self.graph_construction_model) and ("gearnet" in self.graph_construction_model.edge_feature):
            # update the edge feature if we use the 'gearnet' one (as the Euclidean edge distance changes)
            graph.edge_feature[:, -1] = perturbed_dist

        # (2)注意(dist - perturbed_dist)后面的是除号，感觉有点类似于上面(1)的逆操作，也就是在(1)时是先抽取一个服从标准正态分布的噪声，然后基于noise-level/alphas值生成最后噪声，加在当前的3D坐标上（若level足够大可使坐标的分布接近于正态分布）
        # 而这里是产生需要模型最终去预测的噪声，需要是一个*标准正态分布*，所以使用上述的逆过程将dist - perturbed_dist变得接近于正态分布（随后在外部就会经过equivariant transform后就被用于作为监督标签）
        struct_target = (dist - perturbed_dist) / (1.0 - a_edge).sqrt() * a_edge.sqrt()

        # corrupted graphs with corrupted features, Euclidean distance between corrupted edges, difference between original edge distance and corrupted edge distance (normalized by current noise scale)
        return graph, perturbed_dist, struct_target

    def eq_transform(self, score_d, graph):
        # transform invariant scores on edges to equivariant coordinates on nodes
        # struct_pred1 = self.eq_transform(struct_pred1, graph1), struct_pred1 is the recovered/predicted original edge distance based on MLP, graph1 is the corrupted graph
        node_in, node_out = graph.edge_list.t()[:2]
        diff = graph.node_position[node_in] - graph.node_position[node_out] # position vector direction: from source node to target node
        dd_dr = diff / (diff.norm(dim=-1, keepdim=True) + 1e-10) # relative position vector normalized by the (relative) distance (based on the corrupted graph)
        score_pos = scatter_mean(dd_dr * score_d.unsqueeze(-1), node_in, dim=0, dim_size=graph.num_node) \
                  + scatter_mean(-dd_dr * score_d.unsqueeze(-1), node_out, dim=0, dim_size=graph.num_node)
        # print(score_pos.size(), (dd_dr * score_d.unsqueeze(-1)).size()) # torch.Size([502, 3]), torch.Size([1326, 3])
        # ** generating a score for each node as the equivariant coordinate information (for each node, its score comes a sum score of being source node and of being target node) **
        # ** score_d is invariant (i.e., predicted original edge distance), dd_dr can be equivariant: normalized position vector (dd_dr * score_d is thus equivariant), **
        # ** and these will be assigned to end nodes, making score_pos to be equivariant **
        return score_pos

    def struct_predict(self, output, graph, perturbed_dist):
        # predict scores on edges with node representations and perturbed distance (following previous work)
        # i.e., use the corrupted edge distance and corrupted end node embeddings (from original or conformer graphs) to recover original edge distance (same to general denoising scheme here)
        # therefore, the problem of general denoising method may still exist (e.g., noise added could contain more domain prior information)
        # input (for graph1 case): node embeddings of corrupted orginal graph or conformer graph, graph1 (after corruption, providing graph structure), Euclidean distances between corrupted edges in graph1
        node_in, node_out = graph.edge_list.t()[:2] # the edge index in corrupted graph will not change compared with original graph (thus there is correspondence between graph.edge_list and perturbed_dist)
        dist_pred = self.dist_mlp(perturbed_dist.unsqueeze(-1))
        edge_pred = self.struct_mlp(torch.cat((output[node_in] * output[node_out], dist_pred), dim=-1))
        pred = edge_pred.squeeze(-1)
        return pred

    def seq_predict(self, graph, output, node_index):
        # (1) graph1: corrupted protein, output2: original or conformer corrupted protein node-wise embeddings
        # node_index: absolute ids for masked residues in current batch (directly coming from original add_seq_noise function)

        # (2) if set seq_bb_retain=False, there is no bead node remained for masked residues, in this case, it is equivalent to the whole residue (including main+side chain nodes) being isolated from the protein
        # making the AA sequence recovery for these residues extremely difficult (means no effective residue embedding (via below scatter_mean operation) generated for the recovery -> with all values 0)
        # (3) for the following function: graph.num_residue will not change after the subgraph function (keep recording the original residue number)

        # ** for subgraph function, it will call node_mask.data_mask in CG22_PackedProtein, which will keep original indices in original bead2residue under current setting (just remove original indices of masked beads) **
        # ** node_index is also based on the index of original graph, thus, the encoder embeddings of (retained) nodes can correctly correspond to each residue in original graph (so feasible for seq_bb_retain=False) **
        node_feature = scatter_mean(output, graph.bead2residue, dim=0, dim_size=graph.num_residue)[node_index] # get the embeddings of masked residues, above is the reason why this sequence recovery can be performed
        seq_pred = self.seq_mlp(node_feature)
        return seq_pred

    def angle_feat_generator(self, graph, graph_node_feats=None):
        # backbone_angles: BBB (2nd as center_pos, B)
        # backbone_sidec_angles: BBS (3rd as center_pos, S)
        # sidechain_angles: BSS (3rd as center_pos, S)
        # backbone_dihedrals: BBBB (2nd as center_pos, B), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure

        backbone_angles, backbone_angles_center = graph.backbone_angles, 1
        backbone_sidec_angles, backbone_sidec_angles_center = graph.backbone_sidec_angles, 2
        sidechain_angles, sidechain_angles_center = graph.sidechain_angles, 2
        backbone_dihedrals, backbone_dihedrals_center = graph.backbone_dihedrals, 1

        # sine-cosine encoded, output dim=2
        backbone_angles = self.angle_generator(graph, backbone_angles, backbone_angles_center)
        backbone_sidec_angles = self.angle_generator(graph, backbone_sidec_angles, backbone_sidec_angles_center)
        sidechain_angles = self.angle_generator(graph, sidechain_angles, sidechain_angles_center)
        backbone_dihedrals = self.dihedral_generator(graph, backbone_dihedrals, backbone_dihedrals_center)

        # print(torch.sum(backbone_angles), torch.sum(backbone_sidec_angles), torch.sum(sidechain_angles), torch.sum(backbone_dihedrals)) # there are some errors if all values are 0
        # tensor(618.5688, device='cuda:0') tensor(355.6403, device='cuda:0') tensor(201.1166, device='cuda:0') tensor(244.2463, device='cuda:0')

        if graph_node_feats != None:
            return torch.cat([graph_node_feats, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)
        else:
            return torch.cat([backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)

    def angle_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index] # torch.Size([308, 3, 3])
            v_1 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1)
            v_0 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)

            cosD = torch.sum(v_1 * v_0, -1)
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

            D = torch.acos(cosD).unsqueeze(-1)
            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            end_node = angle_index[:, center_pos]
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else:  # for the case that current angle information is not provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def dihedral_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index] # torch.Size([151, 4, 3])
            u_2 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1) # torch.Size([151, 3])
            u_1 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)
            u_0 = self._normalize(X[:, 3, :] - X[:, 2, :], dim=-1)

            # calculate the cross product, and then perform the normalization for it (i.e., return with values after l2 normalization)
            n_2 = self._normalize(torch.cross(u_2, u_1), dim=-1)
            n_1 = self._normalize(torch.cross(u_1, u_0), dim=-1)

            # Angle between normals
            # illustration: Mathematical Background in https://en.wikipedia.org/wiki/Dihedral_angle
            cosD = torch.sum(n_2 * n_1, -1) # actually is a dot product between n_2 and n_1
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps) # output: cosine values from -1 ~ 1
            # torch.sign function: either -1/0/1, to determine the pos/neg radian returned by torch.acos function
            # torch.acos: input the [-1, 1] values and output the angle represented by radian (i.e., arccos function)
            # torch.sum(u_2 * n_1, -1) is actually a dot product representing the pos/neg propensity between the input vectors
            # illustration: https://zhuanlan.zhihu.com/p/359975221
            D = (torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)).unsqueeze(-1) # output: [-pi, pi]

            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            # assign the dihedral features into corresponding node features
            end_node = angle_index[:, center_pos]
            # dim_size should be set to the total node number of current PackedProtein
            # if set it to 'None', the returned matrix will be the size of [maximum id in 'end_node' index]
            # e.g., if graph.num_node = 698, matrix: [698, 2] if dim_size=graph.num_node, matrix: [680, 2] if dim_size=None
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else: # for the case that no dihedral information is provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        # Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def predict_and_target(self, batch, all_loss=None, metric=None):
        # ** the outer training loop is same to standard model training, all diffusion functions are defined here **
        # ** the noise at specific levels/diffusion steps can be directly retrieved via index based on the property of diffusion model **
        # ** currently the transform function which pre-processes proteins can be applied to original proteins **
        # ** before here, we need re-define transform functions for CG-level and new edge types for protein representations **

        # other than the noise added during the diffusion process, another uncertainty added during the pre-training process comes from TruncateProtein transform function
        # which randomly truncates proteins into consecutive 100 (or other pre-defined value) AAs in different epochs
        # besides, the conformer generation is also 'on-the-fly', it will be created in get_item NoiseTransform function in CG dataset class
        graph1 = batch["graph"]

        # currently no graph_construction_model.apply_node_layer function is used here (None)
        if self.graph_construction_model:
            graph1 = self.graph_construction_model.apply_node_layer(graph1)

        if self.use_MI:
            graph2 = batch["graph2"]
            graph2.view = graph1.view # 'atom', preserved in the CG22_PackedProtein phase
            if self.graph_construction_model:
                graph2 = self.graph_construction_model.apply_node_layer(graph2) 

        # determine the step value/noise level for each protein in current batch:
        # alpha/beta is defined in task wrapper initialization: num_noise_level (interval/diffusion steps), sigma_start, sigma_end
        # alphas: tensor([0.9988, 0.9975, ..., 0.0057, 0.0051]) -> gradually decreases after consecutive multiplication -> 100 values in total
        noise_level = torch.randint(0, self.alphas.shape[0], (graph1.batch_size,), device=self.device) # (num_graph, )
        # print(noise_level, self.alphas, self.alphas.shape[0])
        # in stage1: tensor([41, 97, 91, 72]), tensor([0.9988, 0.9975, ..., 0.0063, 0.0057, 0.0051]), 100
        # in stage2: tensor([1, 2, 1, 2]), tensor([0.9999, 0.9998, 0.9997, 0.9996, 0.9995]), 5

        # * the following phases 1 and 2 are the forward diffusion process, 3 and 4 are the reverse diffusion process *
        # use one-hot bead type as the main node feature
        # for original graph node feature generation:
        graph1_node_feats = functional.one_hot(graph1.atom_type[:, 0], len(graph1.martini22_name2id.keys()))
        with graph1.atom(): # registered the feature in the context manager
            graph1.atom_feature = graph1_node_feats
        # for conformer graph node feature generation:
        if self.use_MI:
            graph2_node_feats = functional.one_hot(graph2.atom_type[:, 0], len(graph2.martini22_name2id.keys()))
            with graph2.atom():
                graph2.atom_feature = graph2_node_feats

        # 1. add shared sequence noise, the noise level for sequence and structure are the same
        # for sequence diffusion, first generate an atom/cg-level mask indicating which residues are retained in current batch (based on current noise level)
        # then retrieve the atom/cg-level graph and node features corresponding to the retained residues
        # therefore, the forward AA sequence diffusion is actually a subgraph cropping process based on masked residues
        if self.gamma < 1.0:
            # return: (1) absolute ids for masked residues in current batch (True for masked), (2) boolean mask for unmasked/remained atom/CG nodes in current batch (True for retained),
            # (3) residue types for current masked residues (sequence diffusion objective), (4) mask for BB nodes in masked AAs (with the size of original bead_num, True for these nodes)
            node_index, node_mask, seq_target, BB4maskedAA = self.add_seq_noise(graph1, noise_level) # what are masked are bead nodes
            if self.bb4aa_feats_mask: # serve for that sequence diffusion is available
                graph1.atom_feature[BB4maskedAA, :] = 0
            # print(graph1.view in ["node", "atom"], graph1.view, graph1.atom_feature.size(), graph1.node_feature.size())
            # True, atom, torch.Size([1804, 17]), torch.Size([1804, 17])

            # ** this function should not influence the residue-level information under current setting **
            # ** e.g., no matter whether the backbone beads of masked residues are retained, **
            # ** the 'num_residues' attribute after this function will not be changed (for below seq_predict func) **
            # ** besides, graph.residue2graph will also not change after this subgraph function in current process **
            # node_mask: True represents the retained bead nodes of residues
            graph1 = graph1.subgraph(node_mask) # use the subgraph function in CG22_PackedProtein (as graph1 is a packed protein)

            # current graph1: retain the graph with unmasked atom/bead nodes in which their corresponding node features (including graph.atom_feature) and edges are retained (the edges are also cropped)
            # print(graph1.view in ["node", "atom"], graph1.view, graph1.atom_feature.size(), graph1.node_feature.size())
            # True, atom, torch.Size([1406, 17]), torch.Size([1406, 17])

            # if only use the sequence diffusion (gamma=0), the angle information will be generated here (as no further coordinate change later and further angle info cropping above)
            if self.angle_enhance and self.gamma == 0:
                graph1.atom_feature = self.angle_feat_generator(graph1, graph1.atom_feature)

            # also close up the residue_type for masked residues in current batch (currently no residue_feature is provided)
            # print(graph1.residue_type.size(), graph1.num_residues) # torch.Size([322]), tensor([100, 62, 100, 60])
            with graph1.residue():
                # node_index: absolute ids for masked residues in current batch
                graph1.residue_type[node_index] = 0

            if self.use_MI:
                # print(torch.sum(graph1_node_feats - graph2_node_feats)) # tensor(0), same for the graph1 and graph2
                if self.bb4aa_feats_mask:
                    graph2.atom_feature[BB4maskedAA, :] = 0
                graph2 = graph2.subgraph(node_mask) # use the same node_mask to get the sub-graph
                if self.angle_enhance and self.gamma == 0:
                    graph2.atom_feature = self.angle_feat_generator(graph2, graph2.atom_feature)
                with graph2.residue():
                    graph2.residue_type[node_index] = 0
        # ** 对于当前cg版本的序列diffusion任务，其本身可能是trivial的，因为对于要mask的残基，需要保留至少一个cg节点（也就是BB）去生成residue-level的embedding来recover被mask掉的AA类型 **
        # ** 但对于保留的BB节点，其节点特征one-hot BB类型就包含了一些关于AA倾向的信息（也就是通过BB类型就可以缩小其所属的AA种类），在这种情况下，AA类型recovery任务会变得相对容易，可能影响预训练效果 **
        # ** 所以可能需要考虑去掉序列diffusion任务（比如取一个略大于1的gamma值屏蔽序列任务，例如1.1）或者考虑合理地mask掉这些BB节点的相关节点特征（将这些节点的BB类型节点特征mask掉） **
        # ** 也就是说，需要一个载体节点，用于对mask掉的residue去做recovery，并且需要这个载体不泄露AA类型信息 ***

        # ** 原本的序列diffusion实现：
        # ** 也就是说，应该原本的原子节点特征包含18维的原子类型，以及21维的AA类型，在预训练过程中，若使用序列diffusion，会将需要mask的残基所对应的侧链原子全部去掉 **
        # ** 最终仅保留不需要mask的残基中的全部原子节点以及要mask的残基的主干原子节点，并且对于所有的剩余原子节点，需要关闭其AA类型特征，以免在AA type recovery时造成影响 **
        # ** 上述的描述是针对原子尺度的预训练，对于残基尺度，会保留所有的残基节点（node_mask应该是完整的），只是通过下面的with graph1.residue()关闭需要mask的残基节点的特征 **

        # ** construct edges and edge features, this is done before structural-based perturbation, the edges constructed here for graph1 and graph2 could be different **
        if self.graph_construction_model: # original: construct the spatial edge (SpatialEdge()), now: AdvSpatialEdge in cg_transform script
            # ** return a new graph equipped with new edge_list and edge_feature to cover/replace the graph1 and graph2 in the last step **
            graph1 = self.graph_construction_model.apply_edge_layer(graph1)
        if self.use_MI and self.graph_construction_model:
            graph2 = self.graph_construction_model.apply_edge_layer(graph2)

        # ** 到目前这一步为止，节点特征为one-hot cg bead type，边特征为end node的one-hot residue type + one-hot relation type + end node间的sequential distance + end node之间的欧式距离 **
        # ** 其中只有end node之间的欧式距离会受下面的结构坐标噪声添加的影响，为了实时调整该特征，在add_struct_noise中已经包含根据加噪后的坐标重新调整该边特征的代码 ***
        # ** 但要注意，若边特征继续添加end node之间的相对位置信息，该信息也需要在add_struct_noise处做进一步调整，同样地，若需要添加角度信息作为节点特征，同样也需要在坐标加噪后再对这些特征进行重新计算 **
        # ** 此外，在原实现中，若使用AA sequence diffusion，节点特征就不再是原子类型+残基类型one-hot，而仅包括原子类型特征，应该是为了不在节点层面泄露AA type information，在这种情况下，**
        # ** 会试图通过微环境消息聚合recover masked的AA type，但即使这样，通过上述的gearnet边类型仍会暴露residue type信息，所以可能当前的边特征设置是不合理的，需要把边特征的end node信息从residue type切换为cg type **
        # ** 另外，此时边结构已经固定（即使接下来会对节点坐标加噪声），接下来结构去噪的目标就是对这些固定下来的边由于加噪导致的边距离变化进行预测，然后边特征会在add_struct_noise中完成因坐标变化引起的更新 **
        # ** 除了边结构和边特征外，还需要对节点特征中的角度信息进行更新（因为节点坐标发生了变化，但是所对应的角中包含的节点不应该变化），作为对比，MpbPPI预训练中腐蚀图中的边结构是基于腐蚀图生成的，但感觉由于预训练目标不同， **
        # ** 所以这里的边结构保留方式（因为去噪的就是哪些原始边结构的距离变化值）和MpbPPI中边结构的保留方式（是对加噪声的那些节点的位置变化值进行去噪）应该都是合理的，但对于其他的特征（非去噪目标）计算应该都是基于腐蚀图进行的 **

        # 2. add structure noise
        # for each sample in different epochs, a noise level ranging from 0 to 100 (predefined) will be selected, for graph and graph2, the same noise level will be used
        if self.gamma > 0.0: # gamma is used for controlling weight between structural loss and sequential loss
            # due to the property of the diffusion model, the diffusion calculation can be normally performed no matter how large the noise level will be selected

            # (1) corrupted graphs with corrupted features, (2) Euclidean distances between corrupted edges
            # (3) difference between original and corrupted edge distances (normalized by current noise-level making it more follow the standard normal distribution) -> structure denoising objective
            # analogous to residue types for current masked residues -> sequence denoising objective
            graph1, perturbed_dist1, struct_target1 = self.add_struct_noise(graph1, noise_level) # add the support of edge feature update for corruption to all 'gearnet-like' edge features
            if self.angle_enhance: # suitable for cases gamma > 0 and gamma = 1 (gamma is restricted smaller or equal to 1 in initialization of the task function)
                graph1.atom_feature = self.angle_feat_generator(graph1, graph1.atom_feature)

            if self.use_MI:
                graph2, perturbed_dist2, struct_target2 = self.add_struct_noise(graph2, noise_level)
                if self.angle_enhance:
                    graph2.atom_feature = self.angle_feat_generator(graph2, graph2.atom_feature)

        # currently all corruption has been performed (the output embeddings will be generated based on the corrupted graphs)
        # print(torch.sum(graph1.atom_feature - graph1.node_feature), graph1.atom_feature.size(), graph1.node_feature.size())
        # tensor(0., device='cuda:0'), torch.Size([503, 25]), torch.Size([503, 25]), atom_feature and node_feature should be the same
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss, metric)["node_feature"] # returned: graph_feature and node_feature
        if self.use_MI:
            output2 = self.model(graph2, graph2.node_feature.float(), all_loss, metric)["node_feature"] # torch.Size([502, 768]), 768 = 128 * 6
        else:
            output2 = output1

        # 3. predict structure noise
        # adopting the chain-rule approach proposed in Xu et al. [81], which decomposes the noise on pairwise distances to obtain the modified noise vector ˆϵ as supervision
        # i.e., utilizing MLP to denoise, based on predicting pair-wise distance difference between original and corrupted graphs
        if self.gamma > 0.0:
            # get invariant scores on edges instead of nodes
            # following https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/dualenc.py#L305-L308
            # different encoders could be used to match the diffusion model, current graph1 and graph2 are both under *corruption*
            # i.e., for steps 3 and 4, the denoising is only based on corrupted structures and sequences for both graph1 and graph2

            # input: node embeddings of corrupted orginal graph or conformer graph, graph1 (after corruption, providing graph structure), Euclidean distances between corrupted edges in graph1
            # output: struct_pred1: predicted difference between original edge distance and corrupted edge distance (for creating the corresponding supversion loss)
            struct_pred1 = self.struct_predict(output2, graph1, perturbed_dist1)
            struct_pred1 = self.eq_transform(struct_pred1, graph1) # eq_transform: equivariant transform, make the loss function invariant w.r.t. Rt

            # same to struct_pred1, transform the edge distance difference (between original and corrupted graphs) ground truth into equivariant coordinate information, keeping in line with struct_pred1
            struct_target1 = self.eq_transform(struct_target1, graph1)
            # print(struct_pred1.size(), struct_target1.size()) # torch.Size([502, 3]) torch.Size([502, 3]), generating loss for each bead node

            loss1 = 0.5 * ((struct_pred1 - struct_target1) ** 2).sum(dim=-1) # l2 distance loss, 0.5 representing half of the loss is contributed by this conformer, while the rest is contributed by another conformer
            loss1 = scatter_mean(loss1, graph1.node2graph, dim=0, dim_size=graph1.batch_size).mean() # mean-scattering the node-level loss into each protein graph

            if self.use_MI:
                struct_pred2 = self.struct_predict(output1, graph2, perturbed_dist2) # graph1 and graph2 are the conformer for each other
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
            # graph1: corrupted protein, output2: original or conformer corrupted protein node-wise embeddings
            # node_index: absolute ids for masked residues in current batch (directly coming from original add_seq_noise function)
            seq_pred1 = self.seq_predict(graph1, output2, node_index)
            # seq_pred1: predicted AA types for masked residues, torch.Size([249, 20])
            loss1 = 0.5 * F.cross_entropy(seq_pred1, seq_target, reduction="none") # 0.5 weight for one conformer
            # graph.residue2graph will not change after the subgraph function in sequence diffusion (thus fitting the node_index)
            loss1 = scatter_mean(loss1, graph1.residue2graph[node_index], dim=0, dim_size=graph1.batch_size).mean()
            acc1 = (seq_pred1.argmax(dim=-1) == seq_target).float().mean() # AA sequence recovery accuracy

            if self.use_MI:
                seq_pred2 = self.seq_predict(graph2, output1, node_index) # the same residues are masked in graph1 and graph2
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
        metric = {} # metric dict will be updated within predict_and_target function
        pred, target = self.predict_and_target(batch, all_loss, metric)
        return all_loss, metric


@R.register("tasks.CGDistancePrediction")
class CGDistancePrediction(tasks.Task, core.Configurable):
    """
    Pairwise spatial distance prediction task proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Randomly select some edges and predict the lengths of the edges using the representations of two nodes.
    The selected edges are removed from the input graph to prevent trivial solutions.

    Parameters:
        model (nn.Module): node representation model
        num_sample (int, optional): number of edges selected from each graph
        num_mlp_layer (int, optional): number of MLP layers in distance predictor
        graph_construction_model (nn.Module, optional): graph construction model
    """

    def __init__(self, model, num_sample=256, num_mlp_layer=2, graph_construction_model=None, angle_enhance=True):
        super(CGDistancePrediction, self).__init__()
        self.model = model
        self.num_sample = num_sample
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        self.angle_enhance = angle_enhance

        self.mlp = layers.MLP(2 * model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [1])

    # recover the distance of part of edges constructed from graph_construction_model
    def predict_and_target(self, batch, all_loss=None, metric=None):
        # (1) creating node features (except for angular features), edges, and edge features for each protein
        # ** the angular node features will be generated after graph cropping, as the angles may be removed after the cropping **
        graph = batch["graph"]
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom():  # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        if self.graph_construction_model:
            # node_layer + edge_layer (currently no graph_construction_model.apply_node_layer function is used here (None))
            graph = self.graph_construction_model(graph)
        # print(graph) # CG22_PackedProtein(batch_size=4, num_atoms=[240, 120, 232, 120], num_bonds=[1916, 796, 1678, 808])

        # (2) determining which edges will be masked for denoising tasks (based on selected indices)
        node_in, node_out = graph.edge_list[:, :2].t()
        indices = torch.arange(graph.num_edge, device=self.device)
        # print(indices.size(), graph.edge_list.size(), graph.num_edges)
        # torch.Size([5198]), torch.Size([5198, 3]), tensor([1916, 796, 1678, 808])
        # total edge number in current batch, edge number of each protein, edge number to be sampled for each protein
        # return an index matrix indicating the sampled edge index for each protein based on absolute id
        indices = functional.variadic_sample(indices, graph.num_edges, self.num_sample).flatten(-2, -1)
        # before flatten: torch.Size([4, 256]), (-2, -1) is used to specify the 'flatten' order: row first, col next
        node_i = node_in[indices]
        node_j = node_out[indices]

        # ** edge_mask input: a mask with the size of num_edge, using True to indicate the remained edges **
        # ** return: a new cropped graph with unchanged node indices (including angle composition) and cropped edge number **
        graph = graph.edge_mask(~functional.as_mask(indices, graph.num_edge))
        # print(graph.backbone_angles, graph.atom_feature.size()) # torch.Size([712, 17])

        # (3) after the cropping, adding the angular node features based on the cropped graph
        # ** actually, in current case, there are no angles being removed from edge_mask **
        # ** as what are removed are just edges constructed in graph_construction_model (not 'compact') **
        # ** in other words, angles are calcuated based on bead nodes, which will not be removed here **
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)
        # print(graph.atom_feature.size()) # torch.Size([712, 25])

        # (4) ** calculate distance for unsupervised pre-training (the whole process does not include coordinate noise adding, just masked edge distance recovery) **
        target = (graph.node_position[node_i] - graph.node_position[node_j]).norm(p=2, dim=-1) # as no change for node indices

        output = self.model(graph, graph.node_feature.float(), all_loss, metric)["node_feature"]
        node_feature = torch.cat([output[node_i], output[node_j]], dim=-1)
        pred = self.mlp(node_feature).squeeze(-1) # predict the distance of masked edges

        # print(pred, target)
        # tensor([0.2871, 0.9106, 0.2023,  ..., 0.1746, 0.7763, 0.4978])
        # tensor([4.6669, 3.4016, 3.2569,  ..., 2.6243, 3.1913, 3.0570])
        return pred, target

    # currently it will be used in below forward function
    def evaluate(self, pred, target):
        metric = {}
        mse = F.mse_loss(pred, target)

        name = tasks._get_metric_name("mse")
        metric[name] = mse

        return metric

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.mse_loss(pred, target)
        name = tasks._get_criterion_name("mse")
        metric[name] = loss

        all_loss += loss
        # return to 'engine' function
        return all_loss, metric

    def angle_feat_generator(self, graph, graph_node_feats=None):
        # backbone_angles: BBB (2nd as center_pos, B)
        # backbone_sidec_angles: BBS (3rd as center_pos, S)
        # sidechain_angles: BSS (3rd as center_pos, S)
        # backbone_dihedrals: BBBB (2nd as center_pos, B), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure

        backbone_angles, backbone_angles_center = graph.backbone_angles, 1
        backbone_sidec_angles, backbone_sidec_angles_center = graph.backbone_sidec_angles, 2
        sidechain_angles, sidechain_angles_center = graph.sidechain_angles, 2
        backbone_dihedrals, backbone_dihedrals_center = graph.backbone_dihedrals, 1

        # sine-cosine encoded, output dim=2
        backbone_angles = self.angle_generator(graph, backbone_angles, backbone_angles_center)
        backbone_sidec_angles = self.angle_generator(graph, backbone_sidec_angles, backbone_sidec_angles_center)
        sidechain_angles = self.angle_generator(graph, sidechain_angles, sidechain_angles_center)
        backbone_dihedrals = self.dihedral_generator(graph, backbone_dihedrals, backbone_dihedrals_center)

        # print(graph_node_feats.device, graph.device, backbone_angles.device)
        # print(torch.sum(backbone_angles), torch.sum(backbone_sidec_angles), torch.sum(sidechain_angles), torch.sum(backbone_dihedrals)) # there are some errors if all values are 0
        # tensor(362.0778, device='cuda:0'), tensor(304.2412, device='cuda:0'), tensor(164.1409, device='cuda:0'), tensor(227.0061, device='cuda:0')

        if graph_node_feats != None:
            return torch.cat([graph_node_feats, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)
        else:
            return torch.cat([backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)

    def angle_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index]  # torch.Size([308, 3, 3])
            v_1 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1)
            v_0 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)

            cosD = torch.sum(v_1 * v_0, -1)
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

            D = torch.acos(cosD).unsqueeze(-1)
            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            end_node = angle_index[:, center_pos]
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else:  # for the case that current angle information is not provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def dihedral_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index]  # torch.Size([151, 4, 3])
            u_2 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1)  # torch.Size([151, 3])
            u_1 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)
            u_0 = self._normalize(X[:, 3, :] - X[:, 2, :], dim=-1)

            # calculate the cross product, and then perform the normalization for it (i.e., return with values after l2 normalization)
            n_2 = self._normalize(torch.cross(u_2, u_1), dim=-1)
            n_1 = self._normalize(torch.cross(u_1, u_0), dim=-1)

            # Angle between normals
            # illustration: Mathematical Background in https://en.wikipedia.org/wiki/Dihedral_angle
            cosD = torch.sum(n_2 * n_1, -1)  # actually is a dot product between n_2 and n_1
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps)  # output: cosine values from -1 ~ 1
            # torch.sign function: either -1/0/1, to determine the pos/neg radian returned by torch.acos function
            # torch.acos: input the [-1, 1] values and output the angle represented by radian (i.e., arccos function)
            # torch.sum(u_2 * n_1, -1) is actually a dot product representing the pos/neg propensity between the input vectors
            # illustration: https://zhuanlan.zhihu.com/p/359975221
            D = (torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)).unsqueeze(-1)  # output: [-pi, pi]

            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            # assign the dihedral features into corresponding node features
            end_node = angle_index[:, center_pos]
            # dim_size should be set to the total node number of current PackedProtein
            # if set it to 'None', the returned matrix will be the size of [maximum id in 'end_node' index]
            # e.g., if graph.num_node = 698, matrix: [698, 2] if dim_size=graph.num_node, matrix: [680, 2] if dim_size=None
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else:  # for the case that no dihedral information is provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        # Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


class EnergyDecoder(nn.Module):
    def __init__(self, bead_emb_dim, vdw_radius_coef=0.2, energy_shortcut=True, whether_ES=False, whether_der=False, der_cal_across_protein=False,
                 loss_der1_ratio=10, loss_der2_ratio=10):
        super(EnergyDecoder, self).__init__()

        # first define the basic well depth and vdw radii parameters for the LJ and electrostatic parameters
        self.energy_distance_cutoff = 12
        # shifted distance cutoff for LJ simulation in MARTINI-based MD, temporarily no use
        self.shift_cutoff = 9
        self.ring_vdw_radius = 4.3
        self.vdw_radius_coef = vdw_radius_coef
        self.energy_shortcut = energy_shortcut
        self.whether_ES = whether_ES
        self.whether_der = whether_der
        # if der_cal_across_protein default == True, representing the derivative loss is calculated sample/protein-wise-based (rather than bead-wise-based)
        self.der_cal_across_protein = der_cal_across_protein
        self.loss_der1_ratio = loss_der1_ratio
        self.loss_der2_ratio = loss_der2_ratio

        self.ringbead = torch.LongTensor([13, 14, 15, 16]) # based on original bead type id in martini22_name2id of cg_protein
        self.chargedbead = torch.LongTensor([1, 2]) # based on updated bead type id for energy constant retrival (Qd: 1, Qa: 2)
        self.bead2bead_ = torch.LongTensor(bead2bead_)
        self.CG_well_depth_mat = torch.Tensor(CG_well_depth_mat)
        self.CG_vdw_radius_mat = torch.Tensor(CG_vdw_radius_mat)
        self.CG_q_constant_mat = torch.Tensor(CG_q_constant_mat)
        self.CG_max_welldepth = 5.6 # based on the above well depth matrix
        self.CG_min_welldepth = 2 * 0.75 # based on the above well depth matrix (for rings, the 0.75 time multiplication is needed)

        self.bead_emb_dim = bead_emb_dim # the output dim of the encoder (e.g., 256*bead_emb_layer=1536)
        # * using the MLP structure similar to that in PIGNET (currently no dropout and BN here) *
        # * some specific activation function will be used to physically restrict the output numerical range *
        # * the coefficient should be carefully considered to put into a proper value range *
        self.cal_vdw_interaction_A = nn.Sequential(
            # nn.Linear(self.bead_emb_dim * 2, self.bead_emb_dim),
            # nn.BatchNorm1d(self.bead_emb_dim),
            # nn.ReLU(),
            # nn.Linear(self.bead_emb_dim, 1),

            nn.Linear(self.bead_emb_dim * 2, 1),
            # nn.BatchNorm1d(1),
            nn.Softplus())  # * not allowing negative values here (well depth cannot be negative) *

        self.cal_vdw_interaction_B = nn.Sequential(
            # nn.Linear(self.bead_emb_dim * 2, self.bead_emb_dim),
            # nn.BatchNorm1d(self.bead_emb_dim),
            # nn.ReLU(),
            # nn.Linear(self.bead_emb_dim, 1),

            nn.Linear(self.bead_emb_dim * 2, 1),
            # nn.BatchNorm1d(1),
            nn.Tanh())  # * making the output ranging from -1 to 1

        self.LJ2property = nn.Linear(1, 1)
        # self.LJ2property = nn.Parameter(torch.tensor([0.5]))

        if self.whether_ES:
            self.ES2property = nn.Sequential(
                nn.Linear(1, 1),
                # * not allowing negative values here (energy terms should be positively added) *
                # * actually the output is a combination of energy sum coefficient term and trainable ES energy constant term *
                nn.Softplus())

    def cal_CG_LJ(self, beadpair_emb, beadpair_type, ring_ring_mask, dm):
        # the correction term for vdw radius
        # * note that the vdw radii unit in MARTINI is nm (1e-9) rather than A (1e-10) *
        B = self.cal_vdw_interaction_B(beadpair_emb).squeeze(-1) * self.vdw_radius_coef # output: [-1 * self.vdw_radius_coef, 1 * self.vdw_radius_coef]

        # get the vdw radius for the effective bead pairs
        dm_0 = self.CG_vdw_radius_mat[beadpair_type[:, 0], beadpair_type[:, 1]].to(self.device) # CG_vdw_radius_mat matrix is symmetric
        dm_0[ring_ring_mask] = self.ring_vdw_radius # please note the current unit is A (4.3A=0.43nm)
        dm_0 = dm_0 + B
        # * current dm_0 and dm use the same distance unit: A, so no further unit transformation needed *
        # dm_0/adjusted vdw radii: [4.6243, 4.6252, 4.6077,  ..., 4.7112, 4.7792, 4.7245]
        # dm/actual distance: [9.6192, 8.8947, 7.6465, ..., 11.8646, 10.8093, 11.3719]
        # vdw cutoff: 12A
        vdw_term1 = torch.pow(dm_0 / dm, 2 * 6)
        vdw_term2 = -torch.pow(dm_0 / dm, 6)

        # * adjust the well depth, need to carefully consider the activation function used in this MLP *
        # acquire the original MARTINI well depth
        A_ = self.CG_well_depth_mat[beadpair_type[:, 0], beadpair_type[:, 1]].to(self.device)
        A_[ring_ring_mask] = A_[ring_ring_mask] * 0.75 # scale to 75% of the original value
        A = (self.cal_vdw_interaction_A(beadpair_emb).squeeze(-1) * A_).clamp(min=self.CG_min_welldepth, max=self.CG_max_welldepth)

        energy = vdw_term1 + vdw_term2
        # tensor([0.0125, 0.0202, 0.0502,  ..., 0.0039, 0.0075, 0.0052]), tensor(524.9572), tensor(0.0018)
        # * the clamp is similar to atom-scale PIGNET for DTI *
        # * maybe this clamp is not needed (or changing to a larger threshold) *
        # * as currently LJ is calculated between bead pairs rather than atoms pairs, which could share a larger interaction potential *
        energy = energy.clamp(max=100)

        # energy = 4 * A * energy # choose to add the LJ coefficient in MARTINI: 4
        energy = A * energy
        return energy

    def cal_CG_ES(self, beadpair_type, dm):
        # * the ES potential only exists between Qa-Qa, Qa-Qd, Qd-Qa, and Qd-Qd bead pair combinations *
        # * current beadpair_type is the updated version for energy coefficient retrival, Qa: 2 (old: 11), Qd: 1 (old: 12) *
        # * q constant for Qa: -1 and q constant for Qd: 1 *
        charged_bead_mask = torch.isin(beadpair_type, self.chargedbead.to(self.device)).all(dim=-1)
        beadpair_type_charged, dm_charged = beadpair_type[charged_bead_mask], dm[charged_bead_mask]

        Q = self.CG_q_constant_mat[beadpair_type_charged].to(self.device)
        Q = Q[:, 0] * Q[:, 1]

        # ES = (qi * qj) / (4 * pi * eps0(1) * eps_rel(15) * dm)
        # currently the constant in above formula will not be considered
        # and the scale will be controlled by an MLP outside
        energy = Q / dm_charged

        return energy, charged_bead_mask

    # bead_emb contains the node and graph embeddings generated by the protein encoder
    def forward(self, graph, bead_emb, mlp):
        # * here will construct edges only for energy calculation based on MARTINI vdw radii cutoff and electrostatic radii cutoff *
        # * please note that current batch AA pairs are packed together, requiring to distinguish the pairs from different complexes below *
        node_emb = bead_emb['node_feature']
        graph_emb = bead_emb['graph_feature']

        # * if import prompted graphs, there are some token nodes with virtual node positions and bead/atom types *
        # * in this case, these token nodes do not have specific physical meanings, which need to be ignored when calculating energy *
        # * by utilizing rectified intermol_mat under current prompted graphs, these tokens will be easily screened out below *

        if self.whether_der:
            # construct a tensor with no autograd history (also known as a 'leaf tensor') by copying data to calculate high-order derivative
            all_bead_pos = torch.tensor(graph.node_position, requires_grad=True)
        else:
            all_bead_pos = graph.node_position

        # a bipartite graph should be established based on intermol_mat, thus the core region beads for part A and part B should be distinguished
        interface_AA_partA = torch.unique(graph.intermol_mat[:, 0], sorted=True) # unique AAs of part A in current batch
        interface_AA_partB = torch.unique(graph.intermol_mat[:, 1], sorted=True) # unique AAs of part B in current batch
        # combined = torch.concat([interface_AA_partA, interface_AA_partB])
        # combined_val, counts = combined.unique(return_counts=True)
        # intersection = combined_val[counts > 1] # output: empty

        # start to obtain all effective bead pairs for energy calculation (based on the bipartite graph)
        # coordinate tensor should have the same size as 'graph.node2graph' batch allocation tensor
        # output: all edges within 12A without distinguishing the AA pair region in a batch, including the symmetric edges
        all_interface_beadpair = radius_graph(all_bead_pos, r=self.energy_distance_cutoff, batch=graph.node2graph).t() # generate graphs for each protein in batch

        # * the AA id and bead id are unique for different AAs and beads (in a batch) *
        # * record whether every bead node in current batch is in part A or in part B (the scale is from core AAs to core beads) *
        # * True: representing current bead is in core region of part A or B (but no differentiation for every protein here) *
        bead_mask_partA = torch.isin(graph.bead2residue, interface_AA_partA)
        bead_mask_partB = torch.isin(graph.bead2residue, interface_AA_partB)
        # print(bead_mask_partA.size(), bead_mask_partB.size()) # torch.Size([1995]), torch.Size([1995]) -> total bead num in current batch

        # start to filter the effective bead pairs
        # directly acquire all_interface_beadpair[:, 0] belonging to bead_mask_partA, and all_interface_beadpair[:, 1] belonging to bead_mask_partB
        # based on this way, the symmetric edges can also be removed (such edges are not needed in our case, energy does not need to be calculated twice)
        beadpair_mask = torch.cat([bead_mask_partA[all_interface_beadpair[:, 0]].unsqueeze(-1),
                                   bead_mask_partB[all_interface_beadpair[:, 1]].unsqueeze(-1)], dim=-1).all(dim=-1) # .all() guarantees both end nodes exist
        # * 上面实际上就是interaction part A-B二部图的产生过程，更具体来说，我们先计算了每个蛋白质所对应的完整12A radius graph作为符合相互作用力截止值的所有bead pair *
        # * 对于每一个蛋白，其绝对节点排列顺序是先part A然后是part B，其中的（A和B间的）节点不会发生重复（也就是index不会发生重复），在此基础上，*
        # * 我们去寻找该radius图中source node属于part A然后target node属于B的边，这样的定向筛选也可以去掉多余的反方向对应边（因为A和B中的节点不相交）*
        # * 但需要注意的是，通过上面代码找到的bead pair是多于intermol_mat（其直接包含的是AA pair）中包含的bead pair的，*
        # * 因为我们的最终目的是在12A的cutoff中找到partA和partB之间所有的节点对（也就是二部图）去计算一次能量，*
        # * 而不是仅仅计算intermol_mat（AA pair）中所对应的，基于part A-part B间的，通过BB-based AA distance cropping_threshold所找到的bead pair所对应的能量 *
        # * 例如，在12A的限制下，若intermol_mat有1 (from A) - 4 (from B) 和3 (from A) - 6 (from B)两对（BB）bead pair，那么我们就需要在1-4，1-6，3-4，3-6之间计算能量 *

        # * all effective bead pairs based on 12A cutoff for energy calculation (already removing symmetric edges) *
        # * until this step, the potential virtual toke edges are also removed from the bead pairs *
        all_effective_beadpair = all_interface_beadpair[beadpair_mask]
        # print(all_effective_beadpair, all_effective_beadpair.size()) # from over 60000 to 4877

        # next, calculate the LJ and electrostatic energy separately
        # * in current energy calculation settings, the cutoff for both LJ and electrostatic potentials is 0-12A *
        beadpair_emb = node_emb[all_effective_beadpair].view(-1, 2 * self.bead_emb_dim)

        # graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        # get the original bead pair type (currently the token nodes are already filtered out)
        beadpair_type = graph.atom_type[:, 0][all_effective_beadpair]
        # get the updated bead pair type for energy parameter retrival from the corresponding matrices
        # bead2bead_ is a dict, which does not allow the invalid bead type input (e.g., -1 for token nodes)
        beadpair_type_ = self.bead2bead_[beadpair_type].to(self.device)

        # need to calculate another mask to consider the occurrence of ring-ring interactions (based on original beadpair_type)
        ring_ring_mask = torch.isin(beadpair_type, self.ringbead.to(self.device)).all(dim=-1)
        # print(ring_ring_mask.size(), torch.sum(ring_ring_mask)) # torch.Size([4877]), tensor(142)

        # calculate the pairwise-distance for all effective bead pairs
        pairwise_dis = self.euclidean_distance(all_bead_pos[all_effective_beadpair])

        # * calculate all pairwise LJ energy in current batch (currently no differentiation to the protein belonging of each energy) *
        # * at the same time, LJ is used as the main energy term to be calculated here *
        LJ = self.cal_CG_LJ(beadpair_emb=beadpair_emb, beadpair_type=beadpair_type_, ring_ring_mask=ring_ring_mask, dm=pairwise_dis)
        # * calculate all pairwise electrostatic potentials in current batch (optional) *
        # * the ES only exists between Qa-Qa, Qa-Qd, Qd-Qa, and Qd-Qd bead combinations (i.e., not all effective bead pairs will have ES potential) *
        if self.whether_ES:
            ES, charged_bead_mask = self.cal_CG_ES(beadpair_type=beadpair_type_, dm=pairwise_dis)
            # print(LJ, torch.max(LJ), torch.min(LJ), torch.mean(LJ))
            # print(ES, torch.max(ES), torch.min(ES), torch.mean(ES))

        # * aggregate the energy for each complex (start to distinguish the pairs from different complexes using graph.node2graph) *
        # print(graph.node2graph, graph.node2graph.size()) # tensor([0, 0, 0,  ..., 7, 7, 7]), torch.Size([2220])
        # * using one end of bead id already can determine the graph belonging of the corresponding edge *
        # * since in the above radius graph construction, batch=graph.node2graph is already specified to distinguish different graphs in current batch *
        effective_pair2graph = graph.node2graph[all_effective_beadpair[:, 0]] # tensor([0, 0, 0,  ..., 4, 4, 4])
        # print(LJ.size(), effective_pair2graph.size()) # torch.Size([4877]), torch.Size([4877])

        LJ = self.LJ2property(scatter_mean(LJ, effective_pair2graph, dim=0, dim_size=graph.batch_size).unsqueeze(-1))
        # LJ = LJ / (self.LJ2property * self.LJ2property)

        if self.whether_ES:
            ES = self.ES2property(scatter_mean(ES, effective_pair2graph[charged_bead_mask], dim=0, dim_size=graph.batch_size).unsqueeze(-1))
            final_energy = LJ + ES
        else:
            final_energy = LJ

        if self.energy_shortcut:
            # pred =  mlp(graph_emb) + final_energy # original calculation
            pred = mlp(graph_emb + final_energy) # updated calculation
        else:
            pred = final_energy

        # * derivative loss terms *
        # * final_energy.requires_grad and all_bead_pos.requires_grad are additionally used to ensure no gradient is calculated during inference *
        if self.whether_der and final_energy.requires_grad and all_bead_pos.requires_grad:
            # identify all effective beads for getting the corresponding positions
            all_effective_bead = torch.unique(all_effective_beadpair, sorted=True) # torch.Size([643])

            # actually the gradient is calculated between [the energy over all batch samples] and [the positions of all bead nodes joining the energy calculation]
            gradient = torch.autograd.grad(final_energy.sum(), all_bead_pos, retain_graph=True, create_graph=True)[0]
            # explanation:
            # * only the complete graph.node_position (i.e., all_bead_pos) can be involved into the gradient calculation as above (by opening requires_grad=True in above line) *
            # * however, we can find the calculated gradient is only valid at the position of all_effective_bead, representing the bead nodes joining the energy calculation *
            # * on top of this, above 'gradient' has the same size as all_bead_pos, but only with the position of all_effective_bead having the value not equalling to 0 *
            # effective_grad = gradient.all(dim=-1)
            # print(effective_grad.size(), torch.sum(effective_grad), all_effective_bead.size())
            # torch.Size([2766]), tensor(643), torch.Size([643])

            # 1) sample/protein complex-wise-based derivatives (where the total derivative is the average of that from each complex)
            if self.der_cal_across_protein:
                effective_bead2graph = graph.node2graph[all_effective_bead]
                der1 = scatter_add(gradient[all_effective_bead], effective_bead2graph, dim=0, dim_size=graph.batch_size) # [8, 3]
                der1 = torch.pow(der1, 2).mean()
                # for der2 usually relatively smaller, when its loss weight equals to 0, directly closing its calculation to save time
                if self.loss_der2_ratio == 0:
                    der2 = torch.zeros_like(final_energy).sum()
                else:
                    der2 = torch.autograd.grad(gradient.sum(), all_bead_pos, retain_graph=True, create_graph=True)[0] # two-order derivatives
                    der2 = -scatter_add(der2[all_effective_bead], effective_bead2graph, dim=0, dim_size=graph.batch_size).sum(1).mean()

            # 2) node/bead-wise-based derivatives (where the total derivative is the average of that from each bead node in current batch)
            else:
                der1 = torch.pow(gradient[all_effective_bead].sum(1), 2).mean()
                # for der2 usually relatively smaller, when its loss weight equals to 0, directly closing its calculation to save time
                if self.loss_der2_ratio == 0:
                    der2 = torch.zeros_like(final_energy).sum()
                else:
                    der2 = torch.autograd.grad(gradient.sum(), all_bead_pos, retain_graph=True, create_graph=True)[0]
                    der2 = -der2[all_effective_bead].sum(1).mean()
        else:
            der1 = torch.zeros_like(final_energy).sum()
            der2 = torch.zeros_like(final_energy).sum()

        return pred, der1, der2

    def euclidean_distance(self, position_tensor, dm_min=0.5):
        tensor1, tensor2 = position_tensor[:, 0, :], position_tensor[:, 1, :]
        # the version without any distance restriction
        # print(torch.norm(tensor1 - tensor2, dim=-1))

        # the version with distance restriction
        pairwise_dm = torch.sqrt(torch.pow(tensor1 - tensor2, 2).sum(-1) + 1e-10) # avoid division by zero
        # print(dm, torch.min(dm), torch.max(dm))
        # tensor([10.2305, 10.9860, 11.6955, ..., 11.1217, 11.9269, 9.0605]), tensor(3.1324), tensor(11.9992)

        # if the distance for current bead pair is smaller than dm_min, use dm_min value instead
        # replace_vec = torch.ones_like(pairwise_dm) * dm_min
        # * pairwise_dm < dm_min == True: dm_min/replace_vec, else: pairwise_dm *
        pairwise_dm = torch.where(pairwise_dm < dm_min, dm_min, pairwise_dm)
        return pairwise_dm


class ProteinPrompt(nn.Module):
    def __init__(self, token_dim, token_num=4):
        super(ProteinPrompt, self).__init__()

        self.token_dim = token_dim
        # 4 represents: helix prompt, sheet 3 prompt, sheet 4 prompt coil prompt in order
        self.token_num = token_num
        # one-hot + MLP achieves similar effect compared with nn.Embedding
        self.token_parameter = torch.nn.Parameter(torch.empty(self.token_num, self.token_dim))
        self.token_position = torch.zeros(self.token_num, 3, device=device)

        self.token_init(init_method="kaiming_uniform")
        # self.token_init(init_method="one_hot")

    def token_init(self, init_method="kaiming_uniform"): # init_method="kaiming_uniform" or init_method="one_hot"
        
        if init_method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.token_parameter, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

        # re-initialize the token_parameter based on the one-hot vector (compared with other initialization function)
        # the prompt token embeddings start from corresponding one-hot vectors
        elif init_method == "one_hot":
            assert self.token_dim >= self.token_num, \
                "the token dim should be larger than token num to accommodate required one-hot vectors, {} vs {}.".\
                    format(self.token_dim, self.token_num)

            one_hot_matrix = torch.eye(self.token_num)
            if self.token_dim > self.token_num:
                extra_dim = self.token_dim - self.token_num
                # create an additional matrix with zeros
                zero_matrix = torch.zeros((self.token_num, extra_dim))
                # concatenate the one_hot_matrix with zero_matrix
                one_hot_matrix = torch.cat((one_hot_matrix, zero_matrix), dim=1)

            # initialize the parameter with the one-hot matrix
            self.token_parameter = nn.Parameter(one_hot_matrix)

        else:
            raise ValueError("current initialization method is not supported: {}".format(init_method))

    # input is the CG_PackedProtein class
    def forward(self, packedgraph):
        # the protein prompt tokens will be injected into each of individual proteins contained in CG_PackedProtein class
        # print(packedgraph, packedgraph.edge2graph, packedgraph.node2graph, packedgraph.edge_list)
        # CG22_PackedProtein(batch_size=5, num_atoms=[403, 409, 392, 393, 444], num_bonds=[3072, 3094, 2914, 2974, 3318])
        packedgraph = packedgraph.prompted_graph_generation(self.token_parameter, self.token_position)

        return packedgraph
    

@R.register("tasks.MAML")
class MAML(tasks.PropertyPrediction):
    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, normalization=True, mlp_batch_norm=False, mlp_dropout=0,
                 angle_enhance=True, energy_inject=False, vdw_radius_coef=0.2, energy_shortcut=True, whether_ES=False, whether_der=False,
                 der_cal_across_protein=False, loss_der1_ratio=100, loss_der2_ratio=100, whether_prompt=True, verbose=0):

        # normalization is used for regression tasks with criteria like 'mse', for further normalizing the labels based on mean and std
        super(MAML, self).__init__(model, criterion="mse", metric=("mae", "rmse", "pearsonr"), task='binding_affinity', # weight for each task
            num_mlp_layer=num_mlp_layer, normalization=normalization, num_class=1, graph_construction_model=graph_construction_model,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, verbose=verbose)

        # angle_enhance is used in forward function to determine whether the angle enhanced features are used
        self.angle_enhance = angle_enhance
        # using energy injection decoder
        self.energy_inject = energy_inject
        self.whether_der = whether_der
        # the ratios work under whether_der == True
        self.loss_der1_ratio = loss_der1_ratio
        self.loss_der2_ratio = loss_der2_ratio
        # using extra trainable prompts to be injected into the Protein class
        self.whether_prompt = whether_prompt
        print("\nwhether using extra trainable graph prompts to be injected into each protein graph: {}\n".format(self.whether_prompt))

        if self.energy_inject == True:
            self.energydecoder = EnergyDecoder(
                bead_emb_dim=self.model.output_dim, vdw_radius_coef=vdw_radius_coef, energy_shortcut=energy_shortcut, whether_ES=whether_ES,
                whether_der=self.whether_der, der_cal_across_protein=der_cal_across_protein, loss_der1_ratio=self.loss_der1_ratio, loss_der2_ratio=self.loss_der2_ratio)

        if self.whether_prompt == True:
            self.protein_prompt = ProteinPrompt(token_dim=self.model.input_dim) # dim is same to that in CG_Protein

    # same to that in CGDiff class, input: PackedProtein graph
    def angle_feat_generator(self, graph, graph_node_feats=None):
        # backbone_angles: BBB (2nd as center_pos, B)
        # backbone_sidec_angles: BBS (3rd as center_pos, S)
        # sidechain_angles: BSS (3rd as center_pos, S)
        # backbone_dihedrals: BBBB (2nd as center_pos, B), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure

        backbone_angles, backbone_angles_center = graph.backbone_angles, 1
        backbone_sidec_angles, backbone_sidec_angles_center = graph.backbone_sidec_angles, 2
        sidechain_angles, sidechain_angles_center = graph.sidechain_angles, 2
        backbone_dihedrals, backbone_dihedrals_center = graph.backbone_dihedrals, 1

        # sine-cosine encoded, output dim=2
        backbone_angles = self.angle_generator(graph, backbone_angles, backbone_angles_center)
        backbone_sidec_angles = self.angle_generator(graph, backbone_sidec_angles, backbone_sidec_angles_center)
        sidechain_angles = self.angle_generator(graph, sidechain_angles, sidechain_angles_center)
        backbone_dihedrals = self.dihedral_generator(graph, backbone_dihedrals, backbone_dihedrals_center)

        # print(torch.sum(backbone_angles), torch.sum(backbone_sidec_angles), torch.sum(sidechain_angles), torch.sum(backbone_dihedrals))  # there are some errors if all values are 0
        # tensor(10711.6709, device='cuda:0') tensor(9069.5059, device='cuda:0') tensor(5036.0410, device='cuda:0') tensor(2072.2786, device='cuda:0')

        if graph_node_feats != None:
            return torch.cat([graph_node_feats, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)
        else:
            return torch.cat([backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)

    def angle_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index] # torch.Size([308, 3, 3])
            v_1 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1)
            v_0 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)

            cosD = torch.sum(v_1 * v_0, -1)
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

            D = torch.acos(cosD).unsqueeze(-1)
            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            end_node = angle_index[:, center_pos]
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else:  # for the case that current angle information is not provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def dihedral_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index] # torch.Size([151, 4, 3])
            u_2 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1) # torch.Size([151, 3])
            u_1 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)
            u_0 = self._normalize(X[:, 3, :] - X[:, 2, :], dim=-1)

            # calculate the cross product, and then perform the normalization for it (i.e., return with values after l2 normalization)
            n_2 = self._normalize(torch.cross(u_2, u_1), dim=-1)
            n_1 = self._normalize(torch.cross(u_1, u_0), dim=-1)

            # Angle between normals
            # illustration: Mathematical Background in https://en.wikipedia.org/wiki/Dihedral_angle
            cosD = torch.sum(n_2 * n_1, -1) # actually is a dot product between n_2 and n_1
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps) # output: cosine values from -1 ~ 1
            # torch.sign function: either -1/0/1, to determine the pos/neg radian returned by torch.acos function
            # torch.acos: input the [-1, 1] values and output the angle represented by radian (i.e., arccos function)
            # torch.sum(u_2 * n_1, -1) is actually a dot product representing the pos/neg propensity between the input vectors
            # illustration: https://zhuanlan.zhihu.com/p/359975221
            D = (torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)).unsqueeze(-1) # output: [-pi, pi]

            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            # assign the dihedral features into corresponding node features
            end_node = angle_index[:, center_pos]
            # dim_size should be set to the total node number of current PackedProtein
            # if set it to 'None', the returned matrix will be the size of [maximum id in 'end_node' index]
            # e.g., if graph.num_node = 698, matrix: [698, 2] if dim_size=graph.num_node, matrix: [680, 2] if dim_size=None
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else: # for the case that no dihedral information is provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        # Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def forward(self, batch, graph):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        if self.whether_prompt == True:
            # print(graph, graph.edge2graph, graph.node2graph, graph.edge_list)
            graph = self.protein_prompt(graph)
            # some check:
            # print(torch.sum(graph.node_feature - graph.atom_feature)) # tensor(0.)
            # print(graph.num_node, graph.num_nodes, torch.sum(graph.num_nodes)) # tensor(2061), tensor([407, 413, 396, 397, 448]), tensor(2061)
            # print(graph.node2graph, graph.node2graph.size(), graph.edge2graph, graph.edge2graph.size())
            # tensor([0, 0, 0,  ..., 4, 4, 4]), torch.Size([2061]), tensor([0, 0, 0,  ..., 4, 4, 4]), torch.Size([18410])

        # batch data structure:
        # {'graph': CG22_PackedProtein(batch_size=2, num_atoms=[318, 302], num_bonds=[652, 668], device='cuda:0'),
        # 'binding_affinity': tensor([7.2218, 1.6478], device='cuda:0', dtype=torch.float64)}
        if self.energy_inject == True:
            pred, der1, der2, graph_emb = self.predict_only_training(graph, all_loss, metric) # logits prediction function (without final activation function)
        else:
            pred, graph_emb = self.predict_only_training(graph, all_loss, metric)

        # check whether current batch contains the objective label name (e.g., binding affinity)
        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        # give labels with nan a real value 0
        target = self.target(batch)
        # print(target) # tensor([[7.2218], [1.6478]])
        labeled = ~torch.isnan(target)
        # print(labeled) # tensor([[True], [True]])
        target[~labeled] = 0
        # print(target) # tensor([[7.2218], [1.6478]])

        for criterion, weight in self.criterion.items(): # dict_items([('mse', 1)])
            if criterion == "mse":
                # use the mean and std of all training labels to further normalize every downstream label
                if self.normalization:
                    # ** normalize the predicted label and ground truth simultaneously **
                    # ** note that the normalization is only imposed for regression loss calculation (on predictions and loabels together) **
                    # ** in the case, the model output is still in the original scale, which can be directly used/evaluated in the test set **
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            # print(loss) # tensor([[0.8483], [0.1965]])
            loss = functional.masked_mean(loss, labeled, dim=0)
            # print(loss) # tensor([0.5224])

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l

            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        # extra energy derivative loss terms
        if self.energy_inject == True and self.whether_der == True: # in current setting, even though self.whether_der == False, two empty der1 and der2 will be returned
            der2 = der2.clamp(min=-20)
            all_loss += der1 * self.loss_der1_ratio
            all_loss += der2 * self.loss_der2_ratio

        # print(all_loss, metric) tensor(7.0480, grad_fn=<AddBackward0>), {'mean squared error': tensor(5.8945, grad_fn=<DivBackward0>)}
        return all_loss, metric, graph_emb

    def predict_only_training(self, graph, all_loss=None, metric=None):
        # current 'graph' already contains the final graph to be sent into the encoder
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)

        # * adding extra CG energy term calculation module, *
        # * if needed to add the energy decoder support for inference/test cases, *
        # * pred is the final calculated protein complex properties *
        if self.energy_inject == True:
            pred, der1, der2 = self.energydecoder(graph, output, self.mlp)
        else:
            pred = self.mlp(output["graph_feature"])

        # in current task wrapper setting, if set task=(), the std and mean generated by 'preprocess' function is empty,
        # causing that regression normalization will not work, which will be handled in the future
        # print(self.std, self.mean) # tensor([], device='cuda:0'), tensor([], device='cuda:0')
        # if self.normalization:
        if self.normalization and self.std.size(0) != 0 and self.mean.size(0) != 0:
            pred = pred * self.std + self.mean

        if self.energy_inject == True:
            return pred, der1, der2, output["graph_feature"]
        else:
            return pred, output["graph_feature"]

    # re-write the logits prediction function to incorporate the cg feature generating process
    def predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']

        # graph_node_feats = functional.one_hot(torch.ones_like(graph.atom_type[:, 0]), len(graph.martini22_name2id.keys())) # for testing the importance of bead type
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.graph_construction_model(graph)

        # whether to inject the token graph is the final graph modification option before sending the Protein class into the encoder+decoder
        if self.whether_prompt == True:
            graph = self.protein_prompt(graph)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)

        # * adding extra CG energy term calculation module *
        # * if need to add the energy decoder support for inference/test cases *
        # * pred is the final calculated protein complex properties *
        if self.energy_inject == True:
            pred, der1, der2 = self.energydecoder(graph, output, self.mlp)
        else:
            pred = self.mlp(output["graph_feature"])

        # in current task wrapper setting, if set task=(), the std and mean generated by 'preprocess' function is empty,
        # causing that regression normalization will not work, which will be handled in the future
        # print(self.std, self.mean) # tensor([], device='cuda:0') tensor([], device='cuda:0')
        # if self.normalization:
        if self.normalization and self.std.size(0) != 0 and self.mean.size(0) != 0:
            pred = pred * self.std + self.mean
        # * only return pred for not influencing/modifying the inference process *
        return pred

    # ** actually no need to modify this (for adding energy calculation support for MLP-based model inference) **
    # ** because current inference is also based on above 'predict' function, in which the corresponding support is already added (i.e., adding self.energydecoder option) **
    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    # ** generate graph embeddings only (for GBT-based decoders, current generated graph embeddings do not include energy information) **
    def encoder_predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.graph_construction_model(graph)

        if self.whether_prompt == True:
            graph = self.protein_prompt(graph)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return output["graph_feature"]

    # ** generate graph embeddings only (for GBT-based decoders, current generated graph embeddings do not include energy information) **
    def encoder_predict_and_target(self, batch, all_loss=None, metric=None):
        return self.encoder_predict(batch, all_loss, metric), self.target(batch)

    def evaluate(self, pred, target):
        # identify labels with not null values
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            # this function cannot handle binary classification task in which output dim=1,
            # which works in multi-class classification tasks
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            # add a function handling calculating acc in binary classification tasks
            # assuming self.num_class == 1
            elif _metric == 'binacc':
                assert len(self.num_class) == 1 and self.num_class[0] == 1, "the num_class is not 1: {}".format(self.num_class[0])
                score = self.accuracy(pred[labeled], target[labeled].long())
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric

    def accuracy(self, pred, target):
        _pred = F.sigmoid(pred) > 0.5 # > 0.5: classified as positive
        return ((target == _pred).sum() / target.size(0)).unsqueeze(dim=0)


# default base class of PDBBIND-like PPI complex property prediction tasks
@R.register("tasks.PDBBIND")
class PDBBIND(tasks.PropertyPrediction):
    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, normalization=True, mlp_batch_norm=False, mlp_dropout=0,
                 angle_enhance=True, energy_inject=False, vdw_radius_coef=0.2, energy_shortcut=True, whether_ES=False, whether_der=False,
                 der_cal_across_protein=False, loss_der1_ratio=10, loss_der2_ratio=10, verbose=0):

        # normalization is used for regression tasks with criteria like 'mse', for further normalizing the labels based on mean and std
        super(PDBBIND, self).__init__(model, criterion="mse", metric=("mae", "rmse", "pearsonr"), task='binding_affinity', # weight for each task
            num_mlp_layer=num_mlp_layer, normalization=normalization, num_class=1, graph_construction_model=graph_construction_model,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, verbose=verbose)

        # angle_enhance is used in forward function to determine whether the angle enhanced features are used
        self.angle_enhance = angle_enhance
        # using energy injection decoder
        self.energy_inject = energy_inject
        self.whether_der = whether_der
        # work under whether_der == True
        self.loss_der1_ratio = loss_der1_ratio
        self.loss_der2_ratio = loss_der2_ratio

        if self.energy_inject == True:
            self.energydecoder = EnergyDecoder(
                bead_emb_dim=self.model.output_dim, vdw_radius_coef=vdw_radius_coef, energy_shortcut=energy_shortcut, whether_ES=whether_ES,
                whether_der=self.whether_der, der_cal_across_protein=der_cal_across_protein, loss_der1_ratio=self.loss_der1_ratio, loss_der2_ratio=self.loss_der2_ratio)

    # same to that in CGDiff class, input: PackedProtein graph
    def angle_feat_generator(self, graph, graph_node_feats=None):
        # backbone_angles: BBB (2nd as center_pos, B)
        # backbone_sidec_angles: BBS (3rd as center_pos, S)
        # sidechain_angles: BSS (3rd as center_pos, S)
        # backbone_dihedrals: BBBB (2nd as center_pos, B),
        # it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure

        backbone_angles, backbone_angles_center = graph.backbone_angles, 1
        backbone_sidec_angles, backbone_sidec_angles_center = graph.backbone_sidec_angles, 2
        sidechain_angles, sidechain_angles_center = graph.sidechain_angles, 2
        backbone_dihedrals, backbone_dihedrals_center = graph.backbone_dihedrals, 1

        # sine-cosine encoded, output dim=2
        backbone_angles = self.angle_generator(graph, backbone_angles, backbone_angles_center)
        backbone_sidec_angles = self.angle_generator(graph, backbone_sidec_angles, backbone_sidec_angles_center)
        sidechain_angles = self.angle_generator(graph, sidechain_angles, sidechain_angles_center)
        backbone_dihedrals = self.dihedral_generator(graph, backbone_dihedrals, backbone_dihedrals_center)

        # print(torch.sum(backbone_angles), torch.sum(backbone_sidec_angles), torch.sum(sidechain_angles), torch.sum(backbone_dihedrals))
        # there are some errors if all values are 0 # tensor(10711.6709), tensor(9069.5059), tensor(5036.0410), tensor(2072.2786)

        if graph_node_feats != None:
            return torch.cat([graph_node_feats, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)
        else:
            return torch.cat([backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals], dim=-1)

    def angle_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index] # torch.Size([308, 3, 3])
            v_1 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1)
            v_0 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)

            cosD = torch.sum(v_1 * v_0, -1)
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

            D = torch.acos(cosD).unsqueeze(-1)
            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            end_node = angle_index[:, center_pos]
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else: # for the case that current angle information is not provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def dihedral_generator(self, graph, angle_index, center_pos, eps=1e-7):
        if angle_index.size(0) != 0:
            X = graph.node_position[angle_index] # torch.Size([151, 4, 3])
            u_2 = self._normalize(X[:, 1, :] - X[:, 0, :], dim=-1) # torch.Size([151, 3])
            u_1 = self._normalize(X[:, 2, :] - X[:, 1, :], dim=-1)
            u_0 = self._normalize(X[:, 3, :] - X[:, 2, :], dim=-1)

            # calculate the cross product, and then perform the normalization for it (i.e., return with values after l2 normalization)
            n_2 = self._normalize(torch.cross(u_2, u_1), dim=-1)
            n_1 = self._normalize(torch.cross(u_1, u_0), dim=-1)

            # angle between normals
            # illustration: mathematical background in https://en.wikipedia.org/wiki/Dihedral_angle
            cosD = torch.sum(n_2 * n_1, -1) # actually is a dot product between n_2 and n_1
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps) # output: cosine values from -1 ~ 1
            # torch.sign function: either -1/0/1, to determine the pos/neg radian returned by torch.acos function
            # torch.acos: input the [-1, 1] values and output the angle represented by radian (i.e., arccos function)
            # torch.sum(u_2 * n_1, -1) is actually a dot product representing the pos/neg propensity between the input vectors
            # illustration: https://zhuanlan.zhihu.com/p/359975221
            D = (torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)).unsqueeze(-1) # output: [-pi, pi]

            D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

            # assign the dihedral features into corresponding node features
            end_node = angle_index[:, center_pos]
            # dim_size should be set to the total node number of current PackedProtein
            # if set it to 'None', the returned matrix will be the size of [maximum id in 'end_node' index]
            # e.g., if graph.num_node = 698, matrix: [698, 2] if dim_size=graph.num_node, matrix: [680, 2] if dim_size=None
            D_features = scatter_mean(D_features, end_node, dim=0, dim_size=graph.num_node)

            return D_features

        else: # for the case that no dihedral information is provided for current protein
            return torch.zeros([graph.num_node, 2]).to(self.device)

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        # Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        # batch data:
        # {'graph': CG22_PackedProtein(batch_size=2, num_atoms=[318, 302], num_bonds=[652, 668], device='cuda:0'),
        # 'binding_affinity': tensor([7.2218, 1.6478], device='cuda:0', dtype=torch.float64)}
        if self.energy_inject == True:
            pred, der1, der2 = self.predict_only_training(batch, all_loss, metric) # logits prediction function (without final activation function)
        else:
            pred = self.predict_only_training(batch, all_loss, metric)

        # check whether current batch contains the objective label name (e.g., binding affinity)
        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        # give labels with nan a real value 0
        target = self.target(batch)
        # print(target) # tensor([[7.2218], [1.6478]])
        labeled = ~torch.isnan(target)
        # print(labeled) # tensor([[True], [True]])
        target[~labeled] = 0
        # print(target) # tensor([[7.2218], [1.6478]])

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                # use the mean and std of all training labels to further normalize every downstream label
                if self.normalization:
                    # * normalize the predicted label and ground truth simultaneously *
                    # * note that the normalization is only imposed for regression loss calculation (on predictions and loabels together) *
                    # * in the case, the model output is still in the original scale, which can be directly used/evaluated in the test set *
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            # print(loss) # tensor([[0.8483], [0.1965]])
            loss = functional.masked_mean(loss, labeled, dim=0)
            # print(loss) # tensor([0.5224])

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l

            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        # extra energy derivative loss terms
        # in current setting, even though self.whether_der == False, two empty der1 and der2 will be returned
        if self.energy_inject == True and self.whether_der == True:
            der2 = der2.clamp(min=-20)
            all_loss += der1 * self.loss_der1_ratio
            all_loss += der2 * self.loss_der2_ratio

        return all_loss, metric

    # * re-write the logits prediction function to incorporate the cg feature generating process *
    # * the final protein graph to be sent into the encoder will be generated inside *
    def predict_only_training(self, batch, all_loss=None, metric=None):
        graph = batch['graph']

        # graph_node_feats = functional.one_hot(torch.ones_like(graph.atom_type[:, 0]), len(graph.martini22_name2id.keys())) # test the importance of bead type
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.graph_construction_model(graph)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)

        # * adding extra CG energy term calculation module *
        # * if needing to add the energy decoder support for inference/test cases *
        # * pred is the final calculated protein complex properties *
        if self.energy_inject == True:
            pred, der1, der2 = self.energydecoder(graph, output, self.mlp)
        else:
            pred = self.mlp(output["graph_feature"])

        # in current PDBBIND task wrapper setting, if set task=(), the std and mean generated by 'preprocess' function is empty,
        # causing that regression normalization will not work, which will be handled in the future
        # print(self.std, self.mean) # tensor([], device='cuda:0'), tensor([], device='cuda:0')
        # if self.normalization:
        if self.normalization and self.std.size(0) != 0 and self.mean.size(0) != 0:
            pred = pred * self.std + self.mean

        if self.energy_inject == True:
            return pred, der1, der2
        else:
            return pred

    # re-write the logits prediction function to incorporate the cg feature generating process
    def predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']

        # graph_node_feats = functional.one_hot(torch.ones_like(graph.atom_type[:, 0]), len(graph.martini22_name2id.keys())) # test the importance of bead type
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.graph_construction_model(graph)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)

        # * adding extra CG energy term calculation module *
        # * if needing to add the energy decoder support for inference/test cases *
        # * pred is the final calculated protein complex properties *
        if self.energy_inject == True:
            pred, der1, der2 = self.energydecoder(graph, output, self.mlp)
        else:
            pred = self.mlp(output["graph_feature"])

        # in current PDBBIND task wrapper setting, if set task=(), the std and mean generated by 'preprocess' function is empty,
        # causing that regression normalization will not work, which will be handled in the future
        # print(self.std, self.mean) # tensor([], device='cuda:0'), tensor([], device='cuda:0')
        # if self.normalization:
        if self.normalization and self.std.size(0) != 0 and self.mean.size(0) != 0:
            pred = pred * self.std + self.mean

        # * only returning pred for not influencing/modifying the inference process *
        return pred

    # * actually no need to modify this (for adding energy calculation support for MLP-based model inference) *
    # * since current inference is also based on above 'predict' function, where corresponding support is already added (i.e., adding self.energydecoder option) *
    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    # * generate graph embeddings only (current generated graph embeddings do not include energy information) *
    def encoder_predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.graph_construction_model(graph)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return output["graph_feature"]

    # * generate graph embeddings only (current generated graph embeddings do not include energy information) *
    def encoder_predict_and_target(self, batch, all_loss=None, metric=None):
        return self.encoder_predict(batch, all_loss, metric), self.target(batch)

    def evaluate(self, pred, target):
        # identify labels with not null values
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            # this function cannot handle binary classification task in which output dim=1,
            # which works in multi-class classification tasks
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            # add a function handling calculating acc in binary classification tasks
            # assuming self.num_class == 1
            elif _metric == 'binacc':
                assert len(self.num_class) == 1 and self.num_class[0] == 1, "the num_class is not 1: {}".format(self.num_class[0])
                score = self.accuracy(pred[labeled], target[labeled].long())
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric

    def accuracy(self, pred, target):
        _pred = F.sigmoid(pred) > 0.5 # > 0.5: classified as positive
        return ((target == _pred).sum() / target.size(0)).unsqueeze(dim=0)


@R.register("tasks.STANDARD")
class STANDARD(PDBBIND, tasks.PropertyPrediction):
    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, normalization=True, mlp_batch_norm=False, mlp_dropout=0, angle_enhance=True,
                 energy_inject=True, vdw_radius_coef=0.2, energy_shortcut=True, whether_ES=True, whether_der=False, der_cal_across_protein=False,
                 loss_der1_ratio=100, loss_der2_ratio=100, verbose=0):
        # * two important parts in current downstream wrapper: 1. itp angle processing module 2. PropertyPrediction initialization *
        # 1. initialize the inherited basic PDBBIND task wrapper to use its angle calculation function
        PDBBIND.__init__(self, model, num_mlp_layer=num_mlp_layer, graph_construction_model=graph_construction_model, normalization=normalization,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, angle_enhance=angle_enhance, energy_inject=energy_inject,
            vdw_radius_coef=vdw_radius_coef, energy_shortcut=energy_shortcut, whether_ES=whether_ES, whether_der=whether_der,
            der_cal_across_protein=der_cal_across_protein, loss_der1_ratio=loss_der1_ratio, loss_der2_ratio=loss_der2_ratio, verbose=verbose)

        # 2. initialize the PropertyPrediction wrapper used for current *regression* task
        tasks.PropertyPrediction.__init__(self, model, criterion="mse", metric=("mae", "rmse", "pearsonr"), task='binding_affinity',
            num_mlp_layer=num_mlp_layer, normalization=normalization, num_class=1, graph_construction_model=graph_construction_model,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, verbose=verbose)

        # initialize the energy encoder again to avoid the potential inheritance error
        # https://stackoverflow.com/questions/29214888/typeerror-cannot-create-a-consistent-method-resolution-order-mro
        if self.energy_inject == True:
            self.energydecoder = EnergyDecoder(
                bead_emb_dim=self.model.output_dim, vdw_radius_coef=vdw_radius_coef, energy_shortcut=energy_shortcut, whether_ES=whether_ES,
                whether_der=whether_der, der_cal_across_protein=der_cal_across_protein, loss_der1_ratio=loss_der1_ratio, loss_der2_ratio=loss_der2_ratio)

        # print(self.task, self.metric, self.angle_enhance)
        # {'binding affinity': 1}, {'mae': 1, 'rmse': 1, 'pearsonr': 1}, True
        # angle_enhance is used in forward function to determine whether the angle enhanced features are used


if __name__ == '__main__':
    # encoder initialization
    gearnet = GearNet(hidden_dims=[128, 128, 128, 128, 128, 128], input_dim=39, num_relation=1)
    # task wrapper initialization
    cgdiff = CGDiff(gearnet, sigma_begin=1.0e-3, sigma_end=0.1, num_noise_level=100, gamma=0.5, use_MI=True)