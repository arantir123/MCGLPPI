import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import util
import time
import torch
import pprint
from torchdrug.utils import comm
from torchdrug import core, models, tasks, datasets, utils

# import these here to make load_config_dict identify these registered functions when calling the main function
from siamdiff import dataset, model, task, transform
# new added import for performing CG-scale pre-training
from cg_steps import cg_pretraining_dataset, cg_edgetransform, cg_graphconstruct, cg_models, cg_protein, cg_task_preprocess_type1


def save(solver, path, save_model=True):
    if save_model:
        model = solver.model.model      # only save the encoder
    else:
        model = solver.model            # save both the encoder and prediction head

    if comm.get_rank() == 0:
        logger.warning("Save checkpoint to %s" % path)
    path = os.path.expanduser(path)
    if comm.get_rank() == 0:
        torch.save(model.state_dict(), path)
    comm.synchronize()


# points to be improved:
# 1. in CG scale, the way of generating conformers could be improved (based on adding gaussion noise to every coordinate dim of every protein graph node, similar to structure forward diffusion (to be improved too))
# 2. the truncated protein length hyparparameter could be adjusted according to the cropped (interacting region) protein size in each downstream task
# 3. the radius edge can distinguish whether its two end nodes are from the same residue or not, and it also can be enhanced by CG itp defined edges (if applicable, the duplicate edge reduction maybe also needed)
# 4. in the atom-level gearnet-edge, the edge features used are still based on one-hot residue-type encoding, which may need to be further improved for CG-level models (e.g., adding inter-node unit position vectors)
# 5. diffusion-based adding noise will add noise to every node with scale controlled by step number, while our previous method adds noise based on part structures of part of residues, try to establish connections
# considering whether we can add domain knowledge-based noise into diffusion process, to make it more reasonable
# 6. introduce the bond angle and dihedral angle information provided in CG itp files, as the potential features for node edges (may via context manager)
# 7. 实际上，对于预训练的两个阶段，对于第二个阶段的*序列*噪声添加，其实使用的噪声强度/方式和第一阶段是一样的，只是第二阶段由于噪声间隔设置的较少，所以序列mask只有相对较少的几种选择（但强度范围和阶段一应该是基本一致的）
# 而对于*结构*噪声添加，是主要基于两个阶段计算出的cumprod alpha值，由于第二阶段该值很小（如上），所以第二阶段对蛋白质构象添加的噪声影响很小，符合原论文描述（但序列噪声的强度范围仍和第一阶段保持一致，只是可选范围变少）
# 针对上述的两阶段噪声添加规则，或许可做出进一步改进（对噪声加入先验，或者修改噪声添加方式），以更好的联合建模蛋白质的结构信息和序列信息
# 8. add the GVP model as the encoder and further improve it to adapt current graph structure
# 9. current edge features include residue type, which may cause information leakage when performing masked AA type recovery, which may be changed to cg bead type information

# 10. in current pre-training and downstream setting, the edge number established for each protein seems small, maybe the radius graph cutoff should also be carefully tuned
# 11. implement VAE-based pre-training task, test the performance influence caused by truncated length
# 12.（1）实现atom以及residue尺度的预训练代码，（2）移除下游重复样本，（3）实现ATLAS代码（也可考虑8-1-1划分），（4）跑不同batch size实验
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
    # need to register the name of '_3did' in torchdrug.data.__init__.py so that _3did class can be searched by load_config_dict
    dataset = core.Configurable.load_config_dict(cfg.dataset)

    # solver includes the task wrapper which contain the model parameters
    # in which the whole 3did dataset is input as the training set
    # besides, build_pretrain_solver takes the saved checkpoint name as the input, to read the corresponding pre-trained model
    solver = util.build_pretrain_solver(cfg, dataset)
    # exit() # for counting the sample number of currently used pre-training set

    step = cfg.get("save_interval", 1) # step: 5
    # start the outer loop iterations (similar to standard downstream FOLD3D task)
    # however, a major difference here is currently no validation and test sets are used, and thus no best_epoch is recorded
    # therefore the model storage logic could be little different, and the validation can also be added later
    t0 = time.time()
    for i in range(0, cfg.train.num_epoch, step): # step=5, epoch=25 in total
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i) # {'num_epoch': 25}

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
