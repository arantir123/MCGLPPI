import os
import torch
import logging
from torch.optim import lr_scheduler

from torchdrug.utils import comm
from torchdrug import core, utils, datasets, models, tasks


logger = logging.getLogger(__file__)


def build_ppi_solver(cfg, train_set, valid_set, test_set, use_solver=False):
    task = core.Configurable.load_config_dict(cfg.task) # PDBBIND task wrapper

    # whether to define a solver using core.Engine
    # for the pre-training phase, the solver is explicitly defined
    # for current downstream tasks, the solver is not necessarily needed
    if use_solver:
        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

        # Need to define a solver for preprocessing
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
        task.model.load_state_dict(model_dict)
        # print(model_dict.keys())

    if use_solver:
        return solver, scheduler
    else:
        return task, optimizer, scheduler