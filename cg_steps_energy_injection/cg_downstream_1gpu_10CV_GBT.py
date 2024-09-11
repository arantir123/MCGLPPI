import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.simplefilter("ignore")

import json
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

from cg_steps_energy_injection import cg_pretraining_dataset, cg_downstream_dataset, cg_edgetransform, cg_graphconstruct, \
    cg_models, cg_protein
from cg_steps_energy_injection import cg_task_preprocess_type1

# if use GBT as the decoder based on the activation of 'gbt_use', set whether_emb_save to True to save embeddings of all involved samples into local
whether_emb_save = False

def train(cfg, model, optimizer, scheduler, train_set, valid_set, test_set, device, model_save_name, task_path, early_stop=None):
    best_epoch, best_val, early_stop_counter = None, -1e9, 0
    train_loss, val_metric = [], []

    for epoch in range(cfg.num_epoch):
        # training
        model.train()
        # current total loss for current batch
        loss = MLP_loop(train_set, model, optimizer=optimizer, max_time=cfg.get("train_time"), device=device)
        train_loss.append(loss)
        # current training loss
        print("\nEPOCH %d TRAIN loss: %.8f" % (epoch, loss))

        # * change the logic of the evaluation part in 10-fold CV *
        # * current evaluation set is same to training set *
        model.eval()
        if cfg.model_save_mode == "val":
            with torch.no_grad():
                metric = MLP_test(valid_set, model, max_time=cfg.get("val_time"), device=device)
            val_metric.append(metric[cfg.eval_metric].item())
            # current validation (i.e., training in current case) evaluation results
            print("\nEPOCH %d" % epoch, "VAL metric:", metric)

        # test set for current fold
        with torch.no_grad():
            test_metric = MLP_test(test_set, model, max_time=cfg.get("test_time"), device=device)
        # current test results
        print("\nEPOCH %d" % epoch, "TEST metric:", test_metric)

        if cfg.model_save_mode == "val":
            # * the metrics should be those larger representing better performance (e.g., auroc/pearsonr) *
            if metric[cfg.eval_metric] > best_val:
                early_stop_counter = 0
                # * only the model under each best_epoch will be saved (instead of saving model under every epoch) *
                torch.save(model.state_dict(), model_save_name)

                best_epoch, best_val = epoch, metric[cfg.eval_metric]
                # independent test
                with torch.no_grad():
                    # 'task' is the task wrapper (e.g., PDBBIND) defined outside the 'train' function, which will be updated in every 'loop' function
                    best_test_metric = MLP_test(test_set, task, max_time=cfg.get("test_time"), device=device)

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

        elif cfg.model_save_mode == "direct":
            if epoch == cfg.num_epoch - 1:
                torch.save(model.state_dict(), model_save_name)

        else:
            print("current model save mode {} is not supported".format(cfg.model_save_mode))
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
    task_path = os.path.dirname(task_path) # save in the save path as label files
    print("the path to save the training loss and evaluation metric curves: {}".format(task_path))
    train_loss, val_metric = np.array(train_loss).reshape(-1, 1), np.array(val_metric).reshape(-1, 1)
    train_loss, val_metric = pd.DataFrame(train_loss, columns=[model_save_name]), pd.DataFrame(val_metric, columns=[cfg.eval_metric])
    train_loss.to_csv(os.path.join(task_path, "train_loss.csv"), index=False)
    val_metric.to_csv(os.path.join(task_path, "val_metric.csv"), index=False)

    # last best_epoch and val results on last best_epoch
    return best_epoch, best_val


def MLP_loop(dataset, model, optimizer=None, max_time=None, device=None):
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

        total_loss += float(loss) # float: transform a tensor with gradient into a python float scalar
        total_count += 1

        if optimizer:
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise (e)
                torch.cuda.empty_cache()
                print("Skipped batch due to OOM", flush=True)
                continue

        t.set_description(f"{total_loss / total_count:.8f}")

    return total_loss / total_count


def MLP_test(dataset, model, max_time=None, device=None):
    start = time.time()
    t = tqdm.tqdm(dataset)

    preds, targets = [], []
    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        try:
            batch = utils.cuda(batch, device=device)
            # call to get predictions and labels simultaneously,
            # the related two functions (self.predict and self.target) are also included in the task wrapper
            pred, target = model.predict_and_target(batch)

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print("Skipped batch due to OOM", flush=True)
            continue

        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)

    # the null labels will be masked in 'evaluate' function
    # the acc calculation function in original PropertyPrediction wrapper cannot handle binary classification cases,
    # thus the updated 'evaluate' function in task wrapper is adopted
    metric = model.evaluate(pred, target)

    return metric


def GBT_loop(cfg, encoder, decoder, train_set, test_set, normalization, dataset_name, device):
    encoder.eval()
    # training
    with torch.no_grad():
        # input: training dataloader and test dataloader defined outside
        train_X, train_Y, train_name = GBT_test(train_set, encoder, max_time=cfg.get("train_time"), device=device)
        test_X, test_Y, test_name = GBT_test(test_set, encoder, max_time=cfg.get("test_time"), device=device)

    if whether_emb_save == True:
        all_emb, all_label, all_name = [], [], []
        all_emb.append(train_X) # all embeddings
        all_emb.append(test_X)
        all_emb = np.concatenate(all_emb) # (1270, 1536)
        all_label.append(train_Y) # all labels
        all_label.append(test_Y)
        all_label = np.concatenate(all_label)
        all_name.extend(train_name) # all names
        all_name.extend(test_name)
        all_name = np.array(all_name)

        # arrange the embeddings in order (based on sample names)
        all_name_order = np.argsort(all_name)
        all_name = all_name[all_name_order]
        all_label = all_label[all_name_order]
        all_emb = all_emb[all_name_order]
        emb_info = {"name": all_name, "emb": all_emb, "label": all_label}

        save_path = os.path.join("D:/PROJECT B2_5/code/raw code/CG Diffusion/cgdiff_energy_injection/visualization/", f"{dataset_name}.pkl")
        print("current sample embedding set save path:", save_path)
        try:
            with open(save_path, "wb") as fout:
                pickle.dump(emb_info, fout)
        except IOError:
            print("Error: Could not write the embedding set to the pickle file.")
        # fin = open(save_path, 'rb')
        # emb_info = pickle.load(fin)
        # fin.close()

    train_X = np.round(train_X, 3)
    if normalization == True: # * normalization the labels *
        train_mean = np.mean(train_X)
        train_std = np.std(train_X)
        train_Y = (train_Y - train_mean) / train_std
    print("start the GBT decoder training ...")
    decoder.fit(train_X, train_Y)

    # testing
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

    preds, targets, names = [], [], []
    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        try:
            batch = utils.cuda(batch, device=device)
            # * call to get graph embeddings and labels simultaneously *
            # * the related two functions are also includes in forward function of task wrapper *
            pred, target = model.encoder_predict_and_target(batch) # batch type: PackedProtein
            # print(pred, target, pred.size()) # torch.Size([8, 1536])
            # also recording the name for each protein sample
            name = batch["name"]

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print("Skipped batch due to OOM", flush=True)
            continue

        preds.append(pred)
        targets.append(target)
        names.extend(name)

    # * assuming that there are no 'null' labels in current dataset *
    pred = utils.cat(preds).cpu().detach().numpy()
    target = utils.cat(targets).cpu().detach().numpy()

    return pred, target, names


# * compared with cg_downstream_1gpu_GBT_reg.py, an extra loop is created for each fold, where the json file for current loop will be temporarily saved for data splitting *
# * this script is also added the support of retrieving embeddings for each protein complex from pre-trained model (controlled by above parameter 'whether_emb_save') *
if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    # os.path.basename(args.config).split('.')[0] is used to indicate the ymal configuration file name
    dirname = os.path.basename(args.config).split('.')[0] + "_yaml" + "_seed" +str(args.seed)
    working_dir = util._create_working_directory(cfg, dirname=dirname) # dirname: providing extra suffix for working_dir
    print("current working dictionary:", working_dir)

    # for pretraining: only set the seed for generating pytorch random numbers (e.g., torch.rand, to control the noise-level/scale added to proteins)
    # here: control the random seed not only for generating random numbers but also for modeling optimization
    seed = args.seed
    print("current random seed:", seed)
    torch.manual_seed(seed + comm.get_rank())
    os.environ["PYTHONHASHSEED"] = str(seed)
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

    # start the loop from here:
    # read the jsonl file for the 10-fold CV
    index_path = cfg.dataset.index_path
    with open(index_path) as f:
        splits = f.readlines()
    split_list = []
    for split in splits:
        split_list.append(json.loads(split))
    print("fold total number:", len(split_list),
          "\nsample number in the first fold:", [len(split_list[0][i]) for i in split_list[0]])

    # use GBT as the decoder
    if "gbt_use" in cfg.task.keys():
        print("using GBT as the decoder")
        gbt_pars = cfg.task.gbt_use
        cfg.task.pop("gbt_use")
        gbt_pars["random_state"] = seed  # fix the random seed
        print("GBT decoder hyper-parameters:", gbt_pars)
    else:
        gbt_pars = None

    # record the time of starting 10-fold CV
    current_time, t0 = datetime.now().strftime("%Y_%m_%d_%H_%M"), time.time()
    MAE_total, RMSE_total, PEARSON_total, label_total, prediction_total, name_total = [], [], [], [], [], []
    for fold in range(len(split_list)):

        t1 = time.time()
        print("current fold:", fold)
        current_fold_split = split_list[fold]
        temp_split_path = os.path.dirname(index_path)
        # temp_split_path = os.path.join(temp_split_path, "temp_split_fold{}.json".format(fold))
        temp_split_path = os.path.join(temp_split_path, "temp_split_fold{}_{}.json".format(fold, current_time))

        with open(temp_split_path, "w") as outfile:
            json.dump(current_fold_split, outfile)
        cfg.dataset.index_path = temp_split_path # to be used in Dataset class for creating sample splitting for current batch

        # * currently this script is only used for regression tasks including PDBBIND, ATLAS, and M1101-like predictions
        # could register the name of below datasets in torchdrug.data.__init__.py so that these datasets can be searched by load_config_dict
        if cfg.dataset["class"] in ["PDBBINDDataset", "ATLASDataset", "M1101Dataset"]:
            _dataset = core.Configurable.load_config_dict(cfg.dataset)
            train_set, valid_set, test_set = _dataset.split()
        else:
            raise Exception("current specified dataset is not supported:".format(cfg.dataset["class"]))

        # remove current temporary split file
        os.remove(temp_split_path)

        # ** the pre-trained parameters are loaded here for MLP and GBT-based decoder **
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

        # use GBT as the decoder
        if gbt_pars != None:
            decoder = GradientBoostingRegressor(**gbt_pars)
            # currently the encoder parameters have already been loaded
            MSE, RMSE, MAE, PEARSON = GBT_loop(cfg.train, task, decoder, train_loader, test_loader, cfg.task.get("normalization"), cfg.dataset["class"], device)

            # recording time after GBT training
            t2 = time.time() # the elapsed time is recorded after the testing
            print(f"total elapsed time of current fold {fold}: {t2 - t1:.4f}")
            print(f"total elapsed time of the downstream training process: {t2 - t0:.4f}")

        # use MLP as the decoder
        else:
            # in current settings, there is only one model saved for 10-fold CV (i.e., the model of the last fold)
            # in current settings, for downstream tasks, training from scratch and based on pre-training share the same running epochs
            model_save_name = "model_seed{}_bs{}_epoch{}_dim{}_radius{}_{}_extra_{}.pth".format(args.seed, cfg.engine.batch_size, cfg.train.num_epoch,
                            cfg.task.model.embedding_dim, cfg.task.graph_construction_model.edge_layers[0]["radius"], current_time, cfg.extra)

            best_epoch, best_val = train(cfg.train, task, optimizer, scheduler, train_loader, valid_loader, test_loader, device,
                                         model_save_name, cfg.dataset["label_path"], cfg.train.get("early_stop"))

            task.load_state_dict(torch.load(model_save_name))

            task.eval()
            with torch.no_grad():
                # the task wrapper has already loaded the trained parameters here
                metric = MLP_test(test_loader, task, max_time=None, device=device)
            print("Test metric", metric)

            t2 = time.time() # the elapsed time is recorded after the testing
            print(f"total elapsed time of current fold {fold}: {t2 - t1:.4f}")
            print(f"total elapsed time of the downstream training process: {t2 - t0:.4f}")

            # explicitly delete the model saved for current fold for avoid some unexpected issues
            if fold < len(split_list) - 1:
                os.remove(model_save_name)

            # 'mean absolute error [binding_affinity]', 'root mean squared error [binding_affinity]', 'pearsonr [binding_affinity]'
            task_suffix = cfg.train.eval_metric.split()[-1]
            MAE = "mean absolute error" + " " + task_suffix
            RMSE = "root mean squared error" + " " + task_suffix
            PEARSON = "pearsonr" + " " + task_suffix
            MAE, RMSE, PEARSON = float(metric[MAE]), float(metric[RMSE]), float(metric[PEARSON])

        print(f"normal evaluation metrics in current fold {fold}, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}, Pearson: {PEARSON:.4f}")

        # same for MLP and GBT-based decoders
        MAE_total.append(MAE)
        RMSE_total.append(RMSE)
        PEARSON_total.append(PEARSON)

    # end of the CV loop
    # print overall evaluation results
    print("average RMSE, MAE, and Pearson on test set:", np.mean(RMSE_total), np.mean(MAE_total), np.mean(PEARSON_total))

    # extra calculation of Standard Deviation (SD) and Standard Error (SE), using n-1/ddof=1, i.e., sample SD
    RMSE_SD, MAE_SD, PEARSON_SD = np.std(RMSE_total, ddof=1), np.std(MAE_total, ddof=1), np.std(PEARSON_total, ddof=1)
    print("sample SD for RMSE, MAE, and Pearson on test set:", RMSE_SD, MAE_SD, PEARSON_SD)

    RMSE_SE, MAE_SE, PEARSON_SE = RMSE_SD / np.sqrt(len(RMSE_total)), MAE_SD / np.sqrt(
        len(MAE_total)), PEARSON_SD / np.sqrt(len(PEARSON_total))
    print("sample SE for RMSE, MAE, and Pearson on test set:", RMSE_SE, MAE_SE, PEARSON_SE)

    if gbt_pars == None:
        print("model save name for the last fold:", model_save_name)








