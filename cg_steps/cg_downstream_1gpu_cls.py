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
import numpy as np
import pandas as pd
from datetime import datetime
from torch.optim import lr_scheduler

import cg_util
from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.utils import comm
from siamdiff import dataset, model, task, transform
from cg_steps import cg_pretraining_dataset, cg_downstream_dataset, cg_edgetransform, cg_graphconstruct, \
    cg_models, cg_protein
from cg_steps import cg_task_preprocess_type1
# from cg_steps import cg_task_preprocess_type2


def train(cfg, model, optimizer, scheduler, train_set, valid_set, test_set, device, model_save_name, task_path, early_stop=None):
    best_epoch, best_val, early_stop_counter = None, -1e9, 0
    train_loss, val_metric = [], []

    for epoch in range(cfg.num_epoch):
        # training
        model.train()
        # current total loss for current batch
        loss = loop(train_set, model, optimizer=optimizer, max_time=cfg.get("train_time"), device=device)
        train_loss.append(loss)
        # current training results
        print("\nEPOCH %d TRAIN loss: %.8f" % (epoch, loss))

        # evaluation
        model.eval()
        with torch.no_grad():
            metric = test(valid_set, model, max_time=cfg.get("val_time"), device=device)
        val_metric.append(metric[cfg.eval_metric].item())
        # current validation results
        print("\nEPOCH %d" % epoch, "VAL metric:", metric)

        # independent test
        with torch.no_grad():
            test_metric = test(test_set, model, max_time=cfg.get("test_time"), device=device)
        # current test results
        print("\nEPOCH %d" % epoch, "TEST metric:", test_metric)

        if cfg.model_save_mode == 'val':
            # ** the metrics should be those larger represent better performance (e.g., auroc/pearsonr) **
            if metric[cfg.eval_metric] > best_val:
                early_stop_counter = 0
                # ** only the model under each best_epoch will be saved (instead of saving model under every epoch) **
                torch.save(model.state_dict(), model_save_name)

                best_epoch, best_val = epoch, metric[cfg.eval_metric]
                # independent test
                with torch.no_grad():
                    # 'task' is the task wrapper defined outside the 'train' function, which will be updated in every 'loop' function
                    best_test_metric = test(test_set, task, max_time=cfg.get("test_time"), device=device)
                # current test results
                print("\nEPOCH %d" % epoch, "TEST metric under current best_val:", best_test_metric)
            else:
                early_stop_counter += 1
                if early_stop and early_stop == early_stop_counter:
                    # the specified evaluation metric results on best_epoch (determined based on validation set) for validation and test sets
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
    train_loss.to_csv(os.path.join(task_path, 'train_loss.csv'), index=False)
    val_metric.to_csv(os.path.join(task_path, 'val_metric.csv'), index=False)

    # last best_epoch and val results on last best_epoch
    return best_epoch, best_val


def loop(dataset, model, optimizer=None, max_time=None, device=None):
    start = time.time()

    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0

    for batch in t:
        # end current loop if the time used exceeds the set max_time * 60 (no matter whether current batch already ends)
        if max_time and (time.time() - start) > 60 * max_time: break
        # clean gradient for current batch
        if optimizer: optimizer.zero_grad()
        try:
            batch = utils.cuda(batch, device=device)
            # send batch data into task wrapper to get the loss and corresponding evaluation metric for current batch
            loss, metric = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue

        total_loss += float(loss)
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

    return total_loss / total_count


def test(dataset, model, max_time=None, device=None):
    start = time.time()
    t = tqdm.tqdm(dataset)

    preds, targets = [], []
    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        try:
            batch = utils.cuda(batch, device=device)
            # call to get predictions and labels simultaneously
            # the related two functions are also includes in forward function of task wrapper
            pred, target = model.predict_and_target(batch)

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
    # the acc calculation function in original PropertyPrediction wrapper cannot handle binary classification cases
    metric = model.evaluate(pred, target)

    return metric


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    # os.path.basename(args.config).split('.')[0] is used to indicate the ymal configuration file name
    dirname = os.path.basename(args.config).split('.')[0] + '_yaml' + '_seed' +str(args.seed)
    working_dir = util._create_working_directory(cfg, dirname=dirname) # dirname: providing extra suffix for working_dir
    print('current working dictionary:', working_dir)

    # for pretraining: only set the seed for generating pytorch random numbers (e.g., torch.rand, to control the noise-level/scale added to proteins)
    # here: control the random seed not only for generating random numbers but also for modeling optimization
    seed = args.seed
    print('current random seed:', seed)
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
    train_loader, valid_loader, test_loader = [ # shuffle=(cfg.dataset["class"] != "PIPDataset")
        data.DataLoader(dataset, cfg.engine.batch_size, shuffle=True, num_workers=cfg.engine.num_worker)
            for dataset in [train_set, valid_set, test_set]]

    # in current setting, for downstream tasks, training from scratch and based on pre-training share the same running epochs
    # record the time of starting training
    current_time, t0 = datetime.now().strftime("%Y_%m_%d_%H_%M"), time.time()
    model_save_name = "model_seed{}_bs{}_epoch{}_dim{}_radius{}_{}_extra_{}.pth".format(args.seed, cfg.engine.batch_size, cfg.train.num_epoch,
                    cfg.task.model.embedding_dim, cfg.task.graph_construction_model.edge_layers[0]["radius"], current_time, cfg.extra)

    best_epoch, best_val = train(cfg.train, task, optimizer, scheduler, train_loader, valid_loader, test_loader, device,
                                 model_save_name, cfg.dataset["label_path"], cfg.train.get("early_stop"))
    t1 = time.time()
    print(f'total elapsed time of the downstream training process: {t1 - t0:.4f}')

    # task.load_state_dict("model_epoch_%d.pth" % best_epoch)
    task.load_state_dict(torch.load(model_save_name))

    task.eval()
    with torch.no_grad():
        # the task wrapper has already loaded the trained parameters here
        metric = test(test_loader, task, max_time=None, device=device)
    print("TEST metric", metric)
    print('model save name for the training process:', model_save_name)





