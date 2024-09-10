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
from cg_steps import cg_pretraining_dataset, cg_downstream_dataset, cg_edgetransform, cg_graphconstruct, \
    cg_models, cg_protein
from cg_steps import cg_task_preprocess_type1_stats_cal


def train(cfg, model, optimizer, scheduler, train_set, valid_set, test_set, device, model_save_name, task_path, early_stop=None):
    best_epoch, best_val, early_stop_counter = None, -1e9, 0
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
            # exit()

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
                    best_test_metric, _, _, _, _ = MLP_test(test_set, task, max_time=cfg.get("test_time"), device=device)
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
            # call to get predictions and labels simultaneously
            # the related two functions are also includes in forward function of task wrapper
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
    # the acc calculation function in original PropertyPrediction wrapper cannot handle binary classification cases
    metric = model.evaluate(pred, target)

    if whether_stats:
        total_node_num = np.concatenate(total_node_num)
        total_edge_num = np.concatenate(total_edge_num)
        total_degree_in = np.concatenate(total_degree_in)
        total_degree_out = np.concatenate(total_degree_out)

    return metric, \
           np.array(total_node_num), np.array(total_edge_num), np.array(total_degree_in), np.array(total_degree_out)


def GBT_loop(cfg, encoder, decoder, train_set, test_set, normalization, device):
    encoder.eval()
    # training
    with torch.no_grad():
        train_X, train_Y = GBT_test(train_set, encoder, max_time=cfg.get("train_time"), device=device)
    train_X = np.round(train_X, 3)
    if normalization == True: # * normalization the labels *
        train_mean = np.mean(train_X)
        train_std = np.std(train_X)
        train_Y = (train_Y - train_mean) / train_std
    print('start the GBT decoder training ...')
    decoder.fit(train_X, train_Y)

    # testing
    with torch.no_grad():
        test_X, test_Y  = GBT_test(test_set, encoder, max_time=cfg.get("test_time"), device=device)
    test_X = np.round(test_X, 3)
    test_prediction = decoder.predict(test_X)
    # * the predictions should have the same scale as current training labels, thus the predicted labels should be recovered to normal scale if normalization=True *
    if normalization == True:
        test_prediction = test_prediction * train_std + train_mean

    # record GPU occupation status
    # https://blog.csdn.net/HJC256ZY/article/details/106516894
    if torch.cuda.is_available():
        logger.warning("max GPU memory: %.1f MiB" % (torch.cuda.max_memory_allocated() / 1e6))
        torch.cuda.reset_peak_memory_stats()

    MSE = mean_squared_error(test_Y, test_prediction)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(test_Y, test_prediction)
    PEARSON = scipy.stats.pearsonr(test_Y.reshape(-1), test_prediction.reshape(-1))[0]

    return MSE, RMSE, MAE, PEARSON


def GBT_test(dataset, model, max_time=None, device=None):
    start = time.time()
    t = tqdm.tqdm(dataset)

    preds, targets = [], []
    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        try:
            batch = utils.cuda(batch, device=device)
            # ** call to get graph embeddings and labels simultaneously **
            # ** the related two functions are also includes in forward function of task wrapper **
            pred, target = model.encoder_predict_and_target(batch) # batch type: PackedProtein
            # print(pred, target, pred.size()) # torch.Size([8, 1536])

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue

        preds.append(pred)
        targets.append(target)

    # ** assuming that there are no 'null' labels in current dataset **
    pred = utils.cat(preds).cpu().detach().numpy()
    target = utils.cat(targets).cpu().detach().numpy()

    return pred, target


# * this script does not support the 10-fold cross validation *
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

    # use GBT as the decoder (extract the configuration of GBT from yaml and remove it for below retrieval)
    if 'gbt_use' in cfg.task.keys():
        print('using GBT as the decoder')
        gbt_pars = cfg.task.gbt_use
        cfg.task.pop('gbt_use')
        gbt_pars['random_state'] = seed  # fix the random seed
        print('GBT decoder hyper-parameters:', gbt_pars)
    else:
        gbt_pars = None

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
    # * please note that for below two types of decoders, GBT only uses train_loader and test_loader, while the MLP needs the all three loaders *
    # * since for MLP, the validation set is explicitly needed for model selection of every epoch, while GBT does not needed *
    # * thus, under current settings, for MLP, the validation set should be explicitly specified when creating data splitting json file *
    # * while for GBT, the validation set can be put into training set when creating jsons, and can be split within the GBT (by specifying the split portion) *

    t0 = time.time()
    if gbt_pars != None:
        decoder = GradientBoostingRegressor(**gbt_pars)
        # currently the encoder parameters have already been loaded
        MSE, RMSE, MAE, PEARSON = GBT_loop(cfg.train, task, decoder, train_loader, test_loader, cfg.task.get("normalization"), device)

    else:
        # in current setting, for downstream tasks, training from scratch and based on pre-training share the same running epochs
        # record the time of starting training
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        model_save_name = "model_seed{}_bs{}_epoch{}_dim{}_radius{}_{}_extra_{}.pth".format(args.seed, cfg.engine.batch_size, cfg.train.num_epoch,
                        cfg.task.model.embedding_dim, cfg.task.graph_construction_model.edge_layers[0]["radius"], current_time, cfg.extra)

        best_epoch, best_val = train(cfg.train, task, optimizer, scheduler, train_loader, valid_loader, test_loader, device,
                                     model_save_name, cfg.dataset["label_path"], cfg.train.get("early_stop"))

        task.load_state_dict(torch.load(model_save_name))

        task.eval()
        with torch.no_grad():
            # the task wrapper has already loaded the trained parameters here
            metric = MLP_test(test_loader, task, max_time=None, device=device)
        print("TEST metric", metric)

        # 'mean absolute error [binding_affinity]', 'root mean squared error [binding_affinity]', 'pearsonr [binding_affinity]'
        task_suffix = cfg.train.eval_metric.split()[-1]
        MAE = 'mean absolute error' + ' ' + task_suffix
        RMSE = 'root mean squared error' + ' ' + task_suffix
        PEARSON = 'pearsonr' + ' ' + task_suffix
        MAE, RMSE, PEARSON = float(metric[MAE]), float(metric[RMSE]), float(metric[PEARSON])

    t1 = time.time()
    print(f'total elapsed time of the downstream training process: {t1 - t0:.4f}')
    print(f'normal evaluation metrics in current settings, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}, Pearson: {PEARSON:.4f}')









