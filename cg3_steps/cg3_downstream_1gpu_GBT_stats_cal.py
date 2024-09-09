import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.simplefilter("ignore")

import time
import tqdm
import util
import pprint
import random
import torch
import scipy.stats
import numpy as np
import pandas as pd
from datetime import datetime
from torch.optim import lr_scheduler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import cg_util
from torchdrug.utils import comm
from torchdrug import core, models, tasks, datasets, utils, data
from siamdiff import dataset, model, task, transform

from cg_steps import cg_models, cg_edgetransform
# cg3_task_preprocess_stats_cal is used to further calculate the edge and node degree information of the dataset
from cg3_steps import cg3_downstream_dataset, cg3_task_preprocess_stats_cal, cg3_protein, \
    cg3_graphconstruct


def train(cfg, model, optimizer, scheduler, train_set, valid_set, test_set, device, model_save_name, task_path, early_stop=None):
    best_epoch, best_val, early_stop_counter = None, -1e9, 0 # current best_val is for 'auprc [interface_class]'
    train_loss, val_metric = [], []

    for epoch in range(cfg.num_epoch):
        whether_stats = True if epoch == 0 else False
        # training
        model.train()
        # current total loss for current batch
        loss, train_node_num, train_edge_num, train_degree_in, train_degree_out = \
            MLP_loop(train_set, model, optimizer=optimizer, max_time=cfg.get("train_time"), whether_stats=whether_stats, device=device)
        train_loss.append(loss)
        # current training results
        print("\nEPOCH %d TRAIN loss: %.8f" % (epoch, loss))

        # evaluation
        model.eval()
        with torch.no_grad():
            metric, val_node_num, val_edge_num, val_degree_in, val_degree_out = \
                MLP_test(valid_set, model, max_time=cfg.get("val_time"), whether_stats=whether_stats, device=device)
        val_metric.append(metric[cfg.eval_metric].item())
        # current validation results
        print("\nEPOCH %d" % epoch, "VAL metric:", metric)

        # independent test
        with torch.no_grad():
            test_metric, test_node_num, test_edge_num, test_degree_in, test_degree_out = \
                MLP_test(test_set, model, max_time=cfg.get("test_time"), whether_stats=whether_stats, device=device)
        # current test results
        print("\nEPOCH %d" % epoch, "TEST metric:", test_metric)

        if whether_stats:
            print("average node num, edge num, degree in, and degree out for proteins in the training set:",
                  '%.3f' % np.mean(train_node_num), '%.3f' % np.mean(train_edge_num), '%.3f' % np.mean(train_degree_in), '%.3f' % np.mean(train_degree_out))
            print("average node num, edge num, degree in, and degree out for proteins in the validation set:",
                  '%.3f' % np.mean(val_node_num), '%.3f' % np.mean(val_edge_num), '%.3f' % np.mean(val_degree_in), '%.3f' % np.mean(val_degree_out))
            print("average node num, edge num, degree in, and degree out for proteins in the test set:",
                  '%.3f' % np.mean(test_node_num), '%.3f' % np.mean(test_edge_num), '%.3f' % np.mean(test_degree_in), '%.3f' % np.mean(test_degree_out))

        if cfg.model_save_mode == 'val':
            # * the current metrics should be those larger representing better performance (e.g., auroc/pearsonr) *
            if metric[cfg.eval_metric] > best_val:
                early_stop_counter = 0
                # * only the model under each best_epoch will be saved (instead of saving model under every epoch) *
                torch.save(model.state_dict(), model_save_name)

                best_epoch, best_val = epoch, metric[cfg.eval_metric]
                # independent test
                with torch.no_grad():
                    # 'task' is the task wrapper defined outside the 'train' function, which will be updated in every 'loop' function
                    best_test_metric, _, _, _, _ = MLP_test(test_set, task, max_time=cfg.get("test_time"), device=device)
                # current test results
                print("\nEPOCH %d" % epoch, "TEST metric under current best_val:", best_test_metric)
            else:
                early_stop_counter += 1
                if early_stop and early_stop == early_stop_counter:
                    # the specified evaluation metric results under best_epoch (determined based on validation set) on validation and test sets
                    # best_epoch, best_val, and best_test_metric will be updated together in the above
                    print("BEST %d VAL %s: %.8f TEST %s: %.8f" %
                          (best_epoch, cfg.eval_metric, best_val, cfg.eval_metric, best_test_metric[cfg.eval_metric]))
                    break
            print("BEST %d VAL %s: %.8f TEST %s: %.8f" %
              (best_epoch, cfg.eval_metric, best_val, cfg.eval_metric, best_test_metric[cfg.eval_metric]))

        elif cfg.model_save_mode == 'direct':
            if epoch == cfg.num_epoch - 1:
                torch.save(model.state_dict(), model_save_name)

        else:
            print("current model save mode {} is not supported".format(model_save_name))
            raise NotImplementedError

        # record GPU occupation status
        # https://blog.csdn.net/HJC256ZY/article/details/106516894
        if torch.cuda.is_available():
            logger.warning("max GPU memory: %.1f MiB" % (torch.cuda.max_memory_allocated() / 1e6))
            torch.cuda.reset_peak_memory_stats()

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()

    # save the training loss and evaluation metric curves as dataframe
    task_path = os.path.dirname(task_path)
    print('the path to save the training loss and evaluation metric curves: {}'.format(task_path))
    train_loss, val_metric = np.array(train_loss).reshape(-1, 1), np.array(val_metric).reshape(-1, 1)
    train_loss, val_metric = pd.DataFrame(train_loss, columns=[model_save_name]), pd.DataFrame(val_metric, columns=[cfg.eval_metric])

    train_loss.to_csv(os.path.join(task_path, 'martini3_train_loss.csv'), index=False)
    val_metric.to_csv(os.path.join(task_path, 'martini3_val_metric.csv'), index=False)

    # last best_epoch and val results on last best_epoch
    return best_epoch, best_val


def MLP_loop(dataset, model, optimizer=None, max_time=None, whether_stats=False, device=None):
    start = time.time()

    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0

    total_node_num, total_edge_num, total_degree_in, total_degree_out = [], [], [], []

    for batch in t:
        # end current loop if the time used exceeds the set max_time * 60 (no matter whether current batch already ends)
        if max_time and (time.time() - start) > 60 * max_time: break
        # clean gradient for current batch
        if optimizer: optimizer.zero_grad()

        try:
            batch = utils.cuda(batch, device=device)
            # send batch data into task wrapper to get the loss and corresponding evaluation metric for current batch
            loss, metric, num_nodes, num_edges, degree_in, degree_out = model(batch)
            if whether_stats:
                total_node_num.append(num_nodes)
                total_edge_num.append(num_edges)
                total_degree_in.append(degree_in)
                total_degree_out.append(degree_out)

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue

        total_loss += float(loss) # float: transform a tensor with gradient into a python float scalar
        total_count += 1

        if optimizer:
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise (e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue

        t.set_description(f"{total_loss / total_count:.8f}")

    if whether_stats:
        total_node_num = np.concatenate(total_node_num)
        total_edge_num = np.concatenate(total_edge_num)
        total_degree_in = np.concatenate(total_degree_in)
        total_degree_out = np.concatenate(total_degree_out)

    return total_loss / total_count, \
           np.array(total_node_num), np.array(total_edge_num), np.array(total_degree_in), np.array(total_degree_out)


def MLP_test(dataset, model, max_time=None, whether_stats=False, device=None):
    start = time.time()
    t = tqdm.tqdm(dataset)

    preds, targets = [], []
    total_node_num, total_edge_num, total_degree_in, total_degree_out = [], [], [], []

    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        try:
            batch = utils.cuda(batch, device=device)
            # call to get predictions and labels simultaneously,
            # the related two functions (self.predict and self.target) are also included in the task wrapper
            pred, target = model.predict_and_target(batch)
            pred, num_nodes, num_edges, degree_in, degree_out = pred
            if whether_stats:
                total_node_num.append(num_nodes)
                total_edge_num.append(num_edges)
                total_degree_in.append(degree_in)
                total_degree_out.append(degree_out)

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue

        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)
    # the null labels will be masked in 'evaluate' function
    # the acc calculation function in original PropertyPrediction wrapper cannot handle binary classification cases,
    # thus the updated 'evaluate' function in task wrapper is adopted
    metric = model.evaluate(pred, target)

    if whether_stats:
        total_node_num = np.concatenate(total_node_num)
        total_edge_num = np.concatenate(total_edge_num)
        total_degree_in = np.concatenate(total_degree_in)
        total_degree_out = np.concatenate(total_degree_out)

    return metric, \
           np.array(total_node_num), np.array(total_edge_num), np.array(total_degree_in), np.array(total_degree_out)


# ** this script does not support the 10-fold cross validation **
if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    # os.path.basename(args.config).split('.')[0] is used to indicate the yaml configuration file name
    dirname = os.path.basename(args.config).split('.')[0] + '_yaml' + '_seed' +str(args.seed)
    working_dir = util._create_working_directory(cfg, dirname=dirname) # dirname: providing extra suffix for working_dir
    print('current working dictionary:', working_dir)

    # for pretraining: only set the seed for generating pytorch random numbers (e.g., torch.rand, to control the noise-level/scale added to proteins)
    # here: control the random seed not only for generating random numbers but also for modeling optimization
    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print and save the log information
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    # * currently this script is only used for regression tasks including PDBBIND and ATLAS
    # need to register the name of below datasets in torchdrug.data.__init__.py so that these datasets can be searched by load_config_dict
    if cfg.dataset["class"] in ["PDBBINDDataset", "MANYDCDataset", "ATLASDataset"]:
        _dataset = core.Configurable.load_config_dict(cfg.dataset)
        train_set, valid_set, test_set = _dataset.split()

    task, optimizer, scheduler = cg_util.build_ppi_solver(cfg, train_set, valid_set, test_set)

    # make the task wrapper enter the cuda
    # only support single GPU in current script
    device = torch.device(cfg.engine.gpus[0]) if cfg.engine.gpus else torch.device("cpu")
    if device.type == "cuda":
        task = task.cuda(device)

    # use a dataloader extended from Pytorch DataLoader for batching graph structured data
    # for pretraining phase, sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank) is used to shuffle the original dataset
    # here, the shuffle function in Pytorch DataLoader is used to have the data reshuffled at every epoch
    train_loader, valid_loader, test_loader = [
        data.DataLoader(dataset, cfg.engine.batch_size, shuffle=True, num_workers=cfg.engine.num_worker)
            for dataset in [train_set, valid_set, test_set]]

    # in current setting, for downstream tasks, training from scratch and that based on pre-training share the same running epochs
    # record the time of starting training
    current_time, t0 = datetime.now().strftime("%Y_%m_%d_%H_%M"), time.time()
    model_save_name = "martini3_seed{}_bs{}_epoch{}_dim{}_radius{}_{}_extra_{}.pth".format(args.seed, cfg.engine.batch_size, cfg.train.num_epoch,
                    cfg.task.model.embedding_dim, cfg.task.graph_construction_model.edge_layers[0]["radius"], current_time, cfg.extra)

    best_epoch, best_val = train(cfg.train, task, optimizer, scheduler, train_loader, valid_loader, test_loader, device,
                                 model_save_name, cfg.dataset["label_path"], cfg.train.get("early_stop"))

    task.load_state_dict(torch.load(model_save_name))

    task.eval()
    with torch.no_grad():
        # the task wrapper has already loaded the trained parameters here
        metric, _, _, _, _ = MLP_test(test_loader, task, max_time=None, device=device)
    print("\nTEST metric", metric)

    t1 = time.time()
    print(f'total elapsed time of the downstream training process: {t1 - t0:.4f}')
    print('model save name for the training process:', model_save_name)










