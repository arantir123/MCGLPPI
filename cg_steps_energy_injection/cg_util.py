import os
import torch
import logging
from torch.optim import lr_scheduler

from torchdrug.utils import comm
from torchdrug import core, utils, datasets, models, tasks

logger = logging.getLogger(__file__)


def build_ppi_solver(cfg, train_set, valid_set, test_set, use_solver=False):
    task = core.Configurable.load_config_dict(cfg.task) # task wrapper

    # whether to define a solver using core.Engine
    # for the pre-training phase, the solver is explicitly defined
    # for current downstream tasks, the solver is not necessarily needed
    if use_solver:
        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

        # need to define a solver for preprocessing
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    else:
        # the preprocess function is needed for the task wrapper if user_solver=False,
        # so that the decoder MLP structure (self.mlp) and task weight (optional if not specified in task wrapper) can be determined
        task.preprocess(train_set, valid_set, test_set)

        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    # define the scheduler
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        if use_solver:
            solver.scheduler = scheduler

    # it provides support for loading optimizer (not just model parameters)
    if use_solver and cfg.get("checkpoint") is not None: # use_solver=False
        solver.load(cfg.checkpoint)

    # to load the pre-trained model checkpoint (only encoder model parameters)
    # for cgdiff pre-training, only after the second phase, the parameters only for encoder will be provided
    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        # used load_state_dict is in /uni38/Lib/site-packages/torch/nn/modules/module.py
        task.model.load_state_dict(model_dict)
        # print(model_dict.keys())

    if use_solver:
        return solver, scheduler
    else:
        return task, optimizer, scheduler


def build_ppi_solver_nograd(cfg, train_set, valid_set, test_set, use_solver=False, whether_nograd=True):
    task = core.Configurable.load_config_dict(cfg.task) # task wrapper

    # whether to define a solver using core.Engine
    # for the pre-training phase, the solver is explicitly defined
    # for current downstream tasks, the solver is not necessarily needed
    if use_solver:
        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

        # need to define a solver for preprocessing
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    else:
        # the preprocess function is needed for the task wrapper if user_solver=False,
        # so that the decoder MLP structure (self.mlp) and task weight (optional if not specified in task wrapper) can be determined
        task.preprocess(train_set, valid_set, test_set)

        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    # define the scheduler
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        if use_solver:
            solver.scheduler = scheduler

    # it provides support for loading optimizer (not just model parameters)
    if use_solver and cfg.get("checkpoint") is not None: # use_solver=False
        solver.load(cfg.checkpoint)

    # to load the pre-trained model checkpoint (only encoder model parameters)
    # for cgdiff pre-training, only after the second phase, the parameters only for encoder will be provided
    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        # print(model_dict)

        if whether_nograd == True:
            # * recover the mean and std based on the original training set for the test sample inference *
            # print(model_dict["mean"], model_dict["std"], model_dict["weight"])
            # tensor([-9.6589]), tensor([2.5558]), tensor([1.])
            task.register_buffer("mean", torch.as_tensor(model_dict["mean"], dtype=torch.float))
            task.register_buffer("std", torch.as_tensor(model_dict["std"], dtype=torch.float))
            task.register_buffer("weight", torch.as_tensor(model_dict["weight"], dtype=torch.float))
            # print(task.mean, task.std, task.weight)
            # tensor([-9.6589]), tensor([2.5558]), tensor([1.])

            # load the parameters of the original trained MLP decoder
            task.mlp.load_state_dict({k[4:]: v for k, v in model_dict.items() if k.startswith("mlp")})

            # load the parameters for the GNN protein encoder (task.model)
            model_dict = {k[6:]: v for k, v in model_dict.items() if k.startswith("model")}
            task.model.load_state_dict(model_dict)
        else:
            # load the model checkpoint after pre-training directly (encoder model (task.model) parameters only)
            task.model.load_state_dict(model_dict)

    # * for M1101 dataset, the given label is ddG (not including WT dG and MT dG labels) *
    # * in 'nograd' setting, the basic model needs to predict dG labels to be transformed into the ddG labels, *
    # * thus acquiring to load the mean and std values of the training set for training the original dG models *
    print("final training label mean and std:", task.mean, task.std)

    if use_solver:
        return solver, scheduler
    else:
        return task, optimizer, scheduler
