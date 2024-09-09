import torch
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_add

from torchdrug import core, tasks, layers, models, metrics, data
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.models import GearNet


# ** this version of 'PDBBIND' task wrapper class supports the return of edge and node degree information of the input dataset **
# ** this is the only difference compared with the counterpart provided in 'cg3_task_preprocess.py' **
@R.register("tasks.PDBBIND")
class PDBBIND(tasks.PropertyPrediction):
    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, normalization=True,
                 mlp_batch_norm=False, mlp_dropout=0, angle_enhance=True, verbose=0):

        # normalization is used for regression tasks with criteria like 'mse', for further normalizing the labels based on mean and std
        super(PDBBIND, self).__init__(model, criterion="mse", metric=("mae", "rmse", "pearsonr"), task='binding_affinity', # weight for each task
            num_mlp_layer=num_mlp_layer, normalization=normalization, num_class=1, graph_construction_model=graph_construction_model,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, verbose=verbose)

        # angle_enhance is used in forward function to determine whether the angle enhanced features are used
        self.angle_enhance = angle_enhance

    # input: PackedProtein graph
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
        # tensor(10711.6709, device='cuda:0'), tensor(9069.5059, device='cuda:0'), tensor(5036.0410, device='cuda:0'), tensor(2072.2786, device='cuda:0')

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

        # batch data format:
        # {'graph': CG3_PackedProtein(batch_size=2, num_atoms=[248, 228], num_bonds=[516, 466], device='cuda:0'),
        # 'binding_affinity': tensor([-9.2120, -9.2320], device='cuda:0'), 'name': ['3m63', '3oak']}
        pred, num_nodes, num_edges, degree_in, degree_out = self.predict(batch, all_loss, metric) # logits prediction function (without final activation function)

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

        return all_loss, metric, num_nodes, num_edges, degree_in, degree_out

    # re-write the logits prediction function to incorporate the cg feature generating process
    def predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']

        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini3_name2id.keys()))

        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.graph_construction_model(graph)

        # record the node and edge numbers for each protein in current batch
        num_nodes = graph.num_nodes.cpu().detach().numpy()
        num_edges = graph.num_edges.cpu().detach().numpy()
        # print(graph.degree_in, graph.degree_out, graph.degree_in.size(), graph.degree_out.size(), graph.num_node, graph.num_edge)
        # tensor([7., 3., 7., ..., 9., 9., 9.]) tensor([7., 3., 7., ..., 9., 9., 9.]) torch.Size([4756]) torch.Size([4756]) tensor(4756) tensor(35080)
        degree_in = graph.degree_in.cpu().detach().numpy()
        degree_out = graph.degree_out.cpu().detach().numpy()

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])

        # in current PDBBIND task wrapper setting, if set task=(), the std and mean generated by 'preprocess' function is empty
        # causing that regression normalization will not work, which will be further handled in the future
        # print(self.std, self.mean) # tensor([], device='cuda:0'), tensor([], device='cuda:0')
        # if self.normalization:
        if self.normalization and self.std.size(0) != 0 and self.mean.size(0) != 0:
            pred = pred * self.std + self.mean

        return pred, num_nodes, num_edges, degree_in, degree_out

    # generate graph embeddings only
    def encoder_predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']

        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini3_name2id.keys()))

        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.angle_enhance:
            graph.atom_feature = self.angle_feat_generator(graph, graph.atom_feature)

        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return output["graph_feature"]

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


# * inherited from basic PDBBIND task wrapper (classification template) *
@R.register("tasks.MANYDC")
# inheriting order: https://zhuanlan.zhihu.com/p/268136917
# there will be an MRO error if put tasks.PropertyPrediction ahead of PDBBIND
# illustration: https://stackoverflow.com/questions/29214888/typeerror-cannot-create-a-consistent-method-resolution-order-mro
class MANYDC(PDBBIND, tasks.PropertyPrediction):
    def __init__(self, model, num_mlp_layer=1, graph_construction_model=None, mlp_batch_norm=False, mlp_dropout=0, angle_enhance=True, verbose=0):
        # ** two important parts in current downstream wrapper: 1. itp angle processing module 2. PropertyPrediction initialization **
        # initialize the inherited basic PDBBIND task wrapper to use its angle calculation function
        PDBBIND.__init__(self, model, num_mlp_layer=num_mlp_layer, graph_construction_model=graph_construction_model, normalization=False,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, angle_enhance=angle_enhance, verbose=verbose)

        # initialize the PropertyPrediction wrapper used for current classification task
        tasks.PropertyPrediction.__init__(self, model, criterion="bce", metric=("auroc", "auprc", "binacc"), task='interface_class',
            num_mlp_layer=num_mlp_layer, normalization=False, num_class=1, graph_construction_model=graph_construction_model,
            mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout, verbose=verbose)

        # print(self.task, self.metric, self.angle_enhance)
        # {'interface_class': 1}, {'auroc': 1, 'auprc': 1, 'acc': 1}, True
        # angle_enhance is used in forward function to determine whether the angle enhanced features are used



