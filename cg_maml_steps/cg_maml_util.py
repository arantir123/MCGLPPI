import os
import torch
import logging
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from torchdrug.utils import comm
from torchdrug.layers import functional
from torchdrug import core, utils, datasets, models, tasks

logger = logging.getLogger(__file__)


# * modified version of creating task wrapper, optimizer, and scheduler (mainly serving for current MAML settings) *
# * which adds extra support to fix the encoder in the branch of 'use_solver=False' *
def build_ppi_maml_solver(cfg, train_set, valid_set, test_set, use_solver=False):
    # initialize task wrapper, which includes the basic forward process of the network (i.e., backbone of regression prediction)
    task = core.Configurable.load_config_dict(cfg.task)

    # whether to define a solver using core.Engine
    # for the pre-training phase, the solver is explicitly defined
    # for current downstream tasks, the solver is not necessarily needed
    if use_solver:
        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

        # to define a solver for preprocessing
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    else:
        # the preprocess function is needed for the task wrapper if user_solver=False,
        # so that the decoder MLP structure (self.mlp) and the weight of each prediction task can be determined
        # valid_set and test_set will not be used inside under current logic (the related calculation is based on train_set)
        task.preprocess(train_set, valid_set, test_set)

        # * the encoder parameters are put in 'model' variable of task wrapper *
        if cfg.train.get("fix_encoder") == True:
            encoder_trainable_flag = False
            # parameters in the encoder need to be fixed tensor via requires_grad
            for p in task.model.parameters(): # retrieve parameters in encoder
                p.requires_grad = False
            # a simple check
            for k, v in task.named_parameters():
                if k.startswith("model") and v.requires_grad == True:
                    print("current parameters in the encoder have not been successfully fixed:\n", k, v)
            # cfg.optimizer.params takes only tensor parameters as input (without tensor names)
            cfg.optimizer.params = filter(lambda p: p.requires_grad, task.parameters())
            # print([key for key, value in task.named_parameters() if value.requires_grad])
        else:
            encoder_trainable_flag = True
            # all parameters in task wrapper are trainable
            cfg.optimizer.params = task.parameters()

        print("current encoder model trainable status: {}\n".format(encoder_trainable_flag))
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        # print(optimizer.param_groups) # check the parameters passed into the optimizer
        # * this optimizer is only used for updating trainable parameters in task wrapper *

    # define the scheduler
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau": # driven by loss change rather than just step numbers
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        # tailed specifically for current MAML settings
        if cfg.scheduler["class"] == "CosineAnnealingLR":
            # necessary hyperparameter for CosineAnnealingLR, representing the length of half period of cosine scheduler function
            # https://blog.csdn.net/m0_46324847/article/details/126367249
            # https://zhuanlan.zhihu.com/p/472962710
            cfg.scheduler.T_max = 4 * cfg.train.iterations
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        if use_solver:
            solver.scheduler = scheduler

    # this provides support of loading optimizer as well (not just model parameters) -> 'load' is from /torchdrug/core/engine.py
    if use_solver and cfg.get("checkpoint") is not None: # use_solver = False
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
        # print([name for name, _ in task.named_parameters()]) # task includes encoder (task.model) + energy injector + mlp decoder
        # print(model_dict.keys())

        # load trained parameters from supervised learning
        if cfg.get("whether_from_supervise") == True:
            if cfg.get("only_supervise_encoder") == True:
                model_dict = {k[6:]: v for k, v in model_dict.items() if k.startswith("model")}
                task.model.load_state_dict(model_dict)
            else:
                # used 'load_state_dict' function is in torch/nn/modules/module.py
                task.load_state_dict(model_dict)
        else:
            # load the model checkpoint after pre-training directly (encoder model (task.model) parameters only)
            task.model.load_state_dict(model_dict)

    if use_solver:
        return solver, scheduler
    else:
        return task, optimizer, scheduler


# * standard version of creating task wrapper, optimizer, and scheduler (serving for conventional supervised learning settings) *
def build_ppi_standard_solver(cfg, train_set, valid_set, test_set, use_solver=False):
    task = core.Configurable.load_config_dict(cfg.task) # task wrapper

    # whether to define a solver using core.Engine
    # for the pre-training phase, the solver is explicitly defined
    # for current downstream tasks, the solver is not necessarily needed
    if use_solver:
        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

        # to define a solver for preprocessing
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    else:
        # the preprocess function is needed for the task wrapper if user_solver=False,
        # so that the decoder MLP structure (self.mlp) and the weight of each prediction task can be determined
        # valid_set and test_set will not be used inside under current logic (the related calculation is based on train_set)
        task.preprocess(train_set, valid_set, test_set)

        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    # define the scheduler
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau": # driven by loss change rather than just step numbers
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        if use_solver:
            solver.scheduler = scheduler

    # this provides support for also loading optimizer (not just model parameters)
    if use_solver and cfg.get("checkpoint") is not None: # use_solver = False
        solver.load(cfg.checkpoint)

    # to load the pre-trained model checkpoint (encoder model (task.model) parameters only)
    # for cgdiff pre-training, only after the second phase, the parameters just for encoder will be provided
    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        # used 'load_state_dict' function is in torch/nn/modules/module.py
        task.model.load_state_dict(model_dict)
        # print(model_dict.keys())

    if use_solver:
        return solver, scheduler
    else:
        return task, optimizer, scheduler


# an extra MAML wrapper outside the current task wrapper (which only defines the basic forward process logic) is defined
# providing the logic related to MAML-based meta-training and meta-test
class MAML_wrapper(nn.Module):
    # * the initial model parameters of both task wrapper and optional MAML_wrapper will be put into the optimizer after this initialization outside *
    def __init__(self, cfg, task):
        super(MAML_wrapper, self).__init__()
        # direct send the task wrapper after the initialization to the MAML wrapper as 'self.task'
        # its attributes should be called via 'self.task.attributes'
        self.task = task
        self.K_shot = cfg.dataset.k_shot
        self.K_query = cfg.dataset.k_query
        self.batch_size = cfg.engine.batch_size
        self.val_shot = cfg.dataset.val_shot
        # * the following TRAINABLE INCORPORATED WEIGHTS can be optimized epoch-wise outside along with the parameters in the task wrapper *

        # INCORPORATED WEIGHTS in MAML wrapper 1 (trainable, for task loss combination):
        # for the MAML loss weighted combination, True: self attention-based task loss combination will be used
        self.weighted_task_comb = cfg.train.weighted_task_comb
        if self.weighted_task_comb:
            self.Attention = MultiHeadAttention( # attention module with trainable weights (which can be updated epoch-wise outside)
                cfg.task.model['embedding_dim'] * len(cfg.task.model['hidden_dims']), attention_dropout_rate=0.1, num_heads=8)

        # self.iterations = cfg.train.iterations
        self.task_lr = cfg.train.task_lr # initial learning rate for MAML inner loop (which can be updated epoch-wise outside)
        self.num_inner_steps = cfg.train.num_inner_steps

        # INCORPORATED WEIGHTS in MAML wrapper 2 (non-trainable, for inner loop loss combination):
        # for controlling the loss weights for the multiple inner loops within each task
        self.multi_step_loss_num_epochs = cfg.train.num_epoch

        # INCORPORATED WEIGHTS in MAML wrapper 3 (trainable, for adjusting learning rate of each layer under different inner loops):
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=self.device, init_learning_rate=self.task_lr, total_num_inner_loop_steps=self.num_inner_steps)

        # * initialize trainable learning rate for all trainable tensors during different inner loops *
        # * need to carefully consider which part of model parameters needs the trainable learning rate, *
        # * as under current setting, part of the parameters (e.g., encoder) could be fixed during meta-learning *
        forward_param_dict = self.get_inner_loop_parameter_dict(params=self.task.named_parameters())
        forward_trainable_param = {name: param for name, param in forward_param_dict.items() if param.requires_grad}
        # consider where to optimize these trainable learning rate for each layer
        self.inner_loop_optimizer.initialize(names_weights_dict=forward_trainable_param) # contain trainable weights

    # retrieving all parameters in a module as a dict
    def get_inner_loop_parameter_dict(self, params):
        return {name: param.to(device=self.device) for name, param in params}

    # retrieving all trainable parameters in a module as a dict
    def get_trainable_inner_loop_parameter_dict(self, params):
        return {name: param.to(device=self.device) for name, param in params if param.requires_grad}

    def forward(self, batch, epoch, scheduler=None):
        self.current_epoch = epoch
        # the procedure for meta-training in current batch under current epoch
        batch_total_loss, batch_total_eval = self.training_step(batch)

        # call the scheduler when needing to control the learning rate per batch
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(batch_total_loss)
        elif scheduler is not None:
            scheduler.step()
        # otherwise: scheduler=None when the scheduler should be updated epoch-wise rather than batch-wise
        # print(f"current learning Rate: {scheduler.get_lr()}")
        return batch_total_loss, batch_total_eval

    def training_step(self, batch):
        # * current data structure in a batch (each set is a PackedProtein): [task1 S set], [task1 Q set], [task2 S set], [task2 Q set], ... *
        total_loss, total_eval, sum_emb4weight = [], [], []
        # record loss and evaluation metric for current batch
        # sum_emb4weight is used to record the averaging graph embedding over each task for further giving current task combination weights

        # start to implement the meta-training procedure
        # we need to carefully consider which part of model parameters can be updated or fixed
        # * parameters_to_vector类似于clone操作，也就是会拷贝一份独立的参数（独立于原始的self.model.parameters()）*
        # * 但是和deepcopy相比，拷贝出来的参数仍然在原计算图中，在这种情况下，元学习内部循环所产生的task loss之和（本质其实是内部循环中各个task模型的梯度）才能用于更新原始的self.model.parameters() *
        # * 也就是说，若用deepcopy来保存self.model.parameters()，没有梯度在原始计算图中存在，即使提供最终的loss，也不会对原始的self.model.parameters()进行更新 *
        # * 而detach则是将节点移除计算图，阻断从input位置到节点位置间的反向传播模型更新 *
        old_params = parameters_to_vector(self.task.parameters()) # copy (all) original forward task wrapper model parameters before MAML inner loops

        # iterate every task, for each task, 1) several inner loops with possibly different fixed loss weights will be adopted (get_per_step_loss_importance_vector),
        # 2) and the corresponding initial learning rate of different model layers (under different inner loops) can be trainable (LSLRGradientDescentLearningRule)
        # in addition to this, 3) the weights across different tasks for current batch can also be trainable (MultiHeadAttention)
        Q_graph_number_task = []
        for i in range(len(batch) // 2):
            # loss and evaluation metric for current task under different inner loops
            task_loss, task_eval = [], []
            # get the loss weight per inner loop
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            # get the final CGProtein Class to be sent to the encoder
            S_graph, Q_graph = self.input_graph_construction(batch[i * 2]), self.input_graph_construction(batch[i * 2 + 1])
            Q_graph_number = torch.max(Q_graph.node2graph) + 1
            Q_graph_number_task.append(Q_graph_number)

            for step_idx in range(self.num_inner_steps):
                # * get current model parameters saved in a dict based on the tensor names *
                # * only the trainable model parameters will be retrieved and sent to the next step *
                trainable_names_weights_copy = self.get_trainable_inner_loop_parameter_dict(self.task.named_parameters()) # parameters in task wrapper
                # exactly the same under Python 3.7+
                # print(trainable_names_weights_copy.keys())
                # print([name for name, param in self.task.named_parameters() if param.requires_grad])

                # * update current trainable_names_weights_copy for current task (in_place operation) *
                self.support_step(trainable_names_weights_copy, batch[i * 2], S_graph, step_idx)

                # generate the loss on Q set of current task under current temporarily updated model parameters
                current_loss, current_metric, current_graph_emb = self.query_step(batch[i * 2 + 1], Q_graph)
                # currently only one criterion is allowed, current: mean squared error (defined in the parent class PropertyPrediction)
                current_criterion = tasks._get_criterion_name(list(self.task.criterion.keys())[-1])

                task_loss.append(per_step_loss_importance_vectors[step_idx] * current_loss)
                task_eval.append(current_metric[current_criterion].detach())

            # print(torch.stack(task_loss), torch.stack(task_eval))
            # tensor([0.3976, 0.4385, 0.3997, 0.3232, 0.4181], grad_fn=<StackBackward0>), tensor([1.3457, 1.5871, 1.3993, 1.0474, 1.5494])
            task_loss = torch.sum(torch.stack(task_loss)) # weighted sum loss
            task_eval = torch.sum(torch.stack(task_eval)) / len(task_eval) # mean evaluation metric across all inner loops under current task

            total_loss.append(task_loss)
            total_eval.append(task_eval)
            # use the graph embedding in the last inner loop to calculate the task weights
            # 这里detach的使用可以阻断在临时模型分支上的反向传播
            sum_emb4weight.append(torch.mean(current_graph_emb, dim=0, keepdim=True).detach()) # torch.Size([1, 1536])
            vector_to_parameters(old_params, self.task.parameters())

        # start to combine the loss generated under each task for updating the original model parameters
        # also to consider direct sum all task losses without using extra task weights
        # we can additionally consider to weighted-sum the task losses based on sample number in each Q set
        if self.weighted_task_comb:
            sum_emb4weight = torch.cat(sum_emb4weight, dim=0) # torch.Size([2, 1536])
            task_loss_weight = self.Attention(sum_emb4weight, sum_emb4weight, sum_emb4weight)
            # print(torch.stack(total_loss, dim=0).size(), task_loss_weight.size(), torch.mean(torch.stack(total_loss, dim=0)))
            # torch.Size([2]), torch.Size([2]), tensor(1.1435, grad_fn=<MeanBackward0>)
            total_loss = torch.dot(torch.stack(total_loss, dim=0), task_loss_weight)
        else:
            # direct-sum case:
            # total_loss = torch.mean(torch.stack(total_loss, dim=0))

            # weighted-sum case (only based on all Q sets in current batch rather than the total sample number for each task in the whole training set):
            # in this case, the task will gain more weight if its Q set is smaller
            task_loss_weight = [1 / i for i in Q_graph_number_task]
            loss_weight_sum = sum(task_loss_weight)
            task_loss_weight = [count / loss_weight_sum for count in task_loss_weight]
            # print(torch.stack(total_loss, dim=0), torch.stack(task_loss_weight, dim=0))
            # tensor([1.0674, 1.2194], device='cuda:0', grad_fn=<StackBackward0>) (with grad), tensor([0.5000, 0.5000], device='cuda:0') (without grad)
            total_loss = torch.dot(torch.stack(total_loss, dim=0), torch.stack(task_loss_weight, dim=0))

        # total_eval contains the evaluation metric on each task (based on their respective temporary updated models)
        # for each task, the result are the mean value across all inner loops under current task
        total_eval = torch.stack(total_eval, dim=0)

        # total loss: sum or attention-based weighted sum across different tasks, for each task,
        # it is weighted calculated from multiple inner loops, for each inner loop, it is from the regression metric and potential aux losses
        return total_loss, total_eval

    def support_step(self, trainable_names_weights_copy, batch, graph, step_idx):
        # enter the task wrapper to run the basic forward step
        # current 'graph' already contains the final graph to be sent into the encoder
        loss, metric, _ = self.task(batch, graph)

        # the output loss may contain a combination of the basic loss and the loss for energy decoder optimization
        # print(all_loss, metric) tensor(7.0480, grad_fn=<AddBackward0>), {'mean squared error': tensor(5.8945, grad_fn=<DivBackward0>)}
        # torch.autograd.grad函数是基于反向传播过程计算梯度的工具，torch.autograd.grad函数是基于计算图进行反向传播的。
        # 该函数用于计算某个标量输出相对于一组输入张量的梯度。在计算梯度时，可以选择计算相对于某个特定的参数集，而不计算相对于其他参数。
        # 如果想要计算并更新只属于decoder的参数，只需要将decoder的参数传递给torch.autograd.grad。不需要传递encoder的参数，因为希望它们保持固定。
        grads = torch.autograd.grad(loss, trainable_names_weights_copy.values())
        names_grads_copy = dict(zip(trainable_names_weights_copy.keys(), grads))

        for key, grad in names_grads_copy.items():
            if grad is None:
                print("Grads not found for inner loop parameter", key)
            # print(key, names_grads_copy[key].size())
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        trainable_names_weights_copy = self.inner_loop_optimizer.update_params(
            # input step_idx for retrieving corresponding trainable learning rates for current inner loop
            names_weights_dict=trainable_names_weights_copy, names_grads_wrt_params_dict=names_grads_copy, step_idx=step_idx)

        new_params = []
        for param in trainable_names_weights_copy.values():
            new_params.append(param.view(-1))
        new_params = torch.cat(new_params)

        # give the temporarily updated model parameters calculated above (specifically for current task) to the original parameters,
        # which is an in_place operation
        # 在python 3.7+版本中，support_step中的基于字典的模型参数存储逻辑可以正常执行（参数顺序不会颠倒），因为在这些版本中dict.keys()的顺序是固定的（基于产生顺序）
        # vector_to_parameters(new_params, self.task.parameters())

        # print([[k, v] for k, v in self.task.named_parameters()][-2])
        # ['mlp.batch_norms.1.weight', Parameter containing: tensor([1., 1., 1.,  ..., 1., 1., 1.], requires_grad=True)]
        vector_to_parameters(new_params, [i for i in self.task.parameters() if i.requires_grad]) # input: just parameters without names
        # print([[k, v] for k, v in self.task.named_parameters()][-2])
        # ['mlp.batch_norms.1.weight', Parameter containing: tensor([0.9999, 0.9999, 0.9999,  ..., 0.9999, 0.9999, 0.9999], requires_grad=True)]

    def query_step(self, batch, graph):
        # predict based on current temporary model parameters for current task
        loss, metric, graph_emb = self.task(batch, graph)
        return loss, metric, graph_emb

    # ** no matter in training_step and validation_step, the data sent to the task wrapper for forward calculation only contains samples from one task **
    def validation_step(self, batch):
        # call the inference function in the task wrapper for validation step
        pred, target = self.task.predict_and_target(batch)
        return pred, target

    # call functions in the task wrapper to generate the final graph to be sent to the encoder
    def input_graph_construction(self, batch):
        graph = batch["graph"]

        # graph_node_feats = functional.one_hot(torch.ones_like(graph.atom_type[:, 0]), len(graph.martini22_name2id.keys())) # for testing the importance of bead type
        graph_node_feats = functional.one_hot(graph.atom_type[:, 0], len(graph.martini22_name2id.keys()))
        with graph.atom(): # registered the feature in the context manager
            graph.atom_feature = graph_node_feats

        # enhance the node feature with itp angle information (currently no residue-level feature is used)
        if self.task.angle_enhance:
            graph.atom_feature = self.task.angle_feat_generator(graph, graph.atom_feature)

        # generate the graph structures and features for current proteins
        if self.task.graph_construction_model:
            # forward function of graph_construction_model includes apply_node_layer (None) and apply_edge_layer (edge creation)
            graph = self.task.graph_construction_model(graph)
        return graph

    # define loss weights of inner steps for current task
    # gradually increase the weight for the last step and decrease that of the others
    def get_per_step_loss_importance_vector(self):
        # num_inner_steps default: 5, multi_step_loss_num_epochs default: total epoch number
        loss_weights = np.ones(shape=(self.num_inner_steps)) * (1.0 / self.num_inner_steps)

        # start loss weight adjustment
        decay_rate = 1.0 / self.num_inner_steps / self.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.num_inner_steps

        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.num_inner_steps - 1) * decay_rate),
            1.0 - ((self.num_inner_steps - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        # end loss weight adjustment

        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights


# origin: https://github.com/myprecioushh/ZeroBind/tree/main
class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self, device, total_num_inner_loop_steps, init_learning_rate=1e-3):

        super(LSLRGradientDescentLearningRule, self).__init__()
        print('initial learning rate of the inner loops:', init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate # 0.0001
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps # 5

    # initialize the trainable learning rate for each tensor of input 'names_weights_dict' under different inner loops
    def initialize(self, names_weights_dict):
        # ParameterDict is an ordered dictionary
        # 在Python 3.7+版本中，字典dict的键keys是有序的，即它们保持插入的顺序。这是因为在Python 3.7中，字典的实现经过修改，以确保键的顺序与它们被插入的顺序相同。
        # 在Python 3.6及更早的版本中，字典的键是无序的，即字典不会保持键的插入顺序。因此，在Python 3.7+中，字典的keys()方法返回的键是按照插入的顺序排列的。
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate, requires_grad=True)
        # print([self.names_learning_rates_dict[i] for i in self.names_learning_rates_dict.keys()])

    # input: model parameters and corresponding gradients under current status, and step number of current inner loop (default maximum: 5)
    # output: updated names_weights_dict under given gradients and step_idx
    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, step_idx):
        # print(names_weights_dict)
        # print(names_grads_wrt_params_dict)
        # * 对于每一个task，当执行针对该task的meta-training时，task wrapper中的每个tensor layer在不同的inner loop中都会被分配独立的trainable learning rate（如下）*
        # * 这些trainable learning rates可选地可以基于outer loop逐batch进行更新（通过将这些参数放入针对outer loop的optimizer中实现） *

        return {
            key: names_weights_dict[key]
                 - self.names_learning_rates_dict[key.replace(".", "-")][step_idx]
                 * names_grads_wrt_params_dict[key]
            # * only update parameters with effective named gradients *
            for key in names_grads_wrt_params_dict.keys()}


# origin: https://github.com/myprecioushh/ZeroBind/tree/main
class MultiHeadAttention(nn.Module):
    # hidden_size: dimension of the encoder output (i.e., graph embedding)
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        # print(self.att_size, att_size, hidden_size, num_heads)
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        # self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        self.output_layer = nn.Linear(num_heads * att_size, 1)

    # loss_weight = self.Attention(weight_sumemb, weight_sumemb, weight_sumemb)
    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size() # print(orig_q_size) # torch.Size([2, 512])
        d_k = self.att_size # self.att_size = att_size = hidden_size // num_heads
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        # print(q.size(), k.size(), v.size())
        # torch.Size([2, 1, 8, 64]), torch.Size([2, 1, 8, 64]), torch.Size([2, 1, 8, 64])

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        # print(q.size(), v.size(), k.size())
        # torch.Size([2, 8, 1, 64]), torch.Size([2, 8, 1, 64]), torch.Size([2, 8, 64, 1])

        # Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k) # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        # print(x.size(), v.size()) # torch.Size([2, 8, 1, 1]), torch.Size([2, 8, 1, 64])
        x = x.matmul(v) # [b, h, q_len, attn]
        # print(x.size()) # torch.Size([2, 8, 1, 64])

        x = x.transpose(1, 2).contiguous() # [b, q_len, h, attn]
        # when to use contiguous(): https://www.zhihu.com/tardis/zm/art/64551412?source_id=1003
        # e.g., execute 'view' after 'transpose' and 'permute', etc.
        # is_contiguous直观的解释是Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致。
        # 如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作。
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x).view(orig_q_size[0],) # tensor([0.0282, 0.0101])

        return torch.softmax(x, dim=0)