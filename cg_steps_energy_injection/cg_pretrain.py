import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import util
import time
import torch
import pprint
from torchdrug.utils import comm
from torchdrug import core, models, tasks, datasets, utils

from siamdiff import dataset, model, task, transform
# new added import for performing CG-scale pre-training

from cg_steps_energy_injection import cg_pretraining_dataset, cg_edgetransform, cg_graphconstruct, 
    cg_models, cg_protein, cg_task_preprocess_type1


def save(solver, path, save_model=True):
    if save_model:
        model = solver.model.model # only save the encoder
    else:
        model = solver.model # save both the encoder and prediction head

    if comm.get_rank() == 0:
        logger.warning("Save checkpoint to %s" % path)
    path = os.path.expanduser(path)
    if comm.get_rank() == 0:
        torch.save(model.state_dict(), path)
    comm.synchronize()


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util._create_working_directory(cfg)
    print('current working dictionary:', working_dir)

    # currently only set the seed for generating pytorch random numbers (e.g., torch.rand, to control the noise-level/scale added to proteins)
    torch.manual_seed(args.seed + comm.get_rank())

    # print and save the log information
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % working_dir)

    # configure dataset class
    dataset = core.Configurable.load_config_dict(cfg.dataset)

    # solver includes the task wrapper which contain the model parameters
    # in which the whole 3did dataset is input as the training set
    # besides, build_pretrain_solver takes the saved checkpoint name as the input, to read the corresponding pre-trained model
    solver = util.build_pretrain_solver(cfg, dataset)
    # exit() # for counting the sample number of currently used pre-training set

    step = cfg.get("save_interval", 1) # step: 5
    # start the outer loop iterations (similar to the standard downstream tasks)
    # however, a major difference here is currently no validation and test sets are used, and thus no best_epoch is recorded
    # therefore the model storage logic could be little different, and the validation can also be added later
    t0 = time.time()
    for i in range(0, cfg.train.num_epoch, step): # step=5, epoch=200/50 in total
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i) # {'num_epoch': 200/50}

        # standard torchdrug training function (input batch data into model and get loss from the task wrapper, and then update model parameters according to the loss)
        solver.train(**kwargs)

        # there is a hyperparameter save_model being used, True: only save the encoder, False: save the encoder and decoder
        # save(solver, "model_epoch_%d.pth" % (i + kwargs["num_epoch"]), cfg.get("save_model", True))
        if cfg.task['class'] == 'CGDiff':
            save(solver, "cgdiff_seed{}_gamma{}_bs{}_epoch{}_dim{}_length{}_radius{}_extra_{}.pth".format(args.seed, cfg.task.gamma, cfg.engine.batch_size, cfg.train.num_epoch,
                cfg.task.model.embedding_dim, cfg.dataset.transform.transforms[0]['max_length'], cfg.task.graph_construction_model.edge_layers[0]['radius'], cfg.extra),
                cfg.get("save_model", True))
        else:
            save(solver, "pretrain_seed{}_bs{}_epoch{}_dim{}_length{}_radius{}_extra_{}.pth".format(args.seed, cfg.engine.batch_size, cfg.train.num_epoch,
                cfg.task.model.embedding_dim, cfg.dataset.transform.transforms[0]['max_length'], cfg.task.graph_construction_model.edge_layers[0]['radius'], cfg.extra),
                cfg.get("save_model", True))

    t1 = time.time()
    print(f'total elapsed time of the pre-training process: {t1 - t0:.4f}')
