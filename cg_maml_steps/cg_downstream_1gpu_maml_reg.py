# the implementation for CG MAML-based regression framework
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.simplefilter("ignore")

import json
import util
import time
import tqdm
import torch
import pprint
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from torch import optim
from torch.optim import lr_scheduler
from torchdrug.utils import comm
from torchdrug import core, data, utils

# import the below for being found by 'Registry.search'
from cg_maml_steps import cg_downstream_maml_dataset, cg_downstream_maml_sampler, cg_maml_graphconstruct, \
    cg_maml_util, cg_task_maml_preprocess
# import the below if these original independent files are in the 'cg_maml_steps' folder
from cg_maml_steps import cg_models, cg_edgetransform


# ** a test hyperparameter for ignoring the former N batches of samples during the zero-shot scenario **
# ** the specific number of samples to be ignored is also determined by 'val_shot' controlling the batch size in inference **
# ** note that this will also influence current 'total_sample_list' that saves currently used meta-training and test samples **
# ** it will also influence similar 'all_sample_wo_finetuning' variable due to the potential record change of the test set **
zero_ignore_batch_num = None # None: no test batch ignorance


# cfg: the hyperparameter configuration for model training, model: the initialized MAML wrapper
def train(cfg, model, optimizer, scheduler, train_set, test_set, device, model_save_name, task_path, model_path, batch_size,
          early_stop=None, test_mode="zero", eval_mode="min", few_weight_decay=None):
    assert test_mode in ["zero", "few"], "Only zero-shot or few-shot test modes are supported."

    # scheduler_pos determines the frequency of scheduler updates
    best_train_epoch, test_eval_on_best_train_epoch, scheduler_pos = 0, None, cfg.scheduler_pos
    train_loss, train_metric = [], []
    if eval_mode == "min": # for MSE/MAE-based training metric
        best_train_eval = 1e9
    elif eval_mode == "max": # for Pearsonr-based training metric
        best_train_eval = -1e9

    # ** consider where to properly put the optimizer and scheduler **
    # ** under current logic, for optimizer for MAML outer loops, it can be put normally at the end of each batch **
    # ** for scheduler, provide some flexibility to put scheduler at the end of each batch or each epoch **
    for epoch in range(cfg.num_epoch):
        # training
        model.train()
        # current output loss: the average loss for current epoch (i.e., average loss over each batch loss)
        # batch loss: from sum or attention-based weighted sum across different tasks, for each task,
        # it is weighted-calculated from multiple inner loops, for each inner loop, it is from the regression metric and potential aux losses
        loss, metric, sample_list = MAML_inner_loop(train_set, model, epoch=epoch, batch_size=batch_size, optimizer=optimizer,
                            scheduler=scheduler, scheduler_pos=scheduler_pos, max_time=cfg.get("train_time"), device=device)

        train_loss.append(loss)
        # print(torch.stack(metric, dim=0).size()) # torch.Size([14, 2]), iterations (batches) * batch task number
        # output metric format: the metric over multiple tasks in the training set, in other words, for each row above,
        # it contains the evaluation metric on each task (based on their respective temporary updated models) in a batch,
        # for each task, the result are the mean value across all inner loops under current task

        # current logic is to average the aforementioned groups of metrics as the final training metric
        # another alternative: perform averaging over individual samples rather than [batches + tasks]
        metric = torch.mean((torch.stack(metric, dim=0)))
        train_metric.append(metric)
        # current training loss
        print("\nEPOCH %d TRAIN loss on the query set: %.8f" % (epoch, loss))

        if epoch == 0:
            # * no further sorting for the sample list to preserve the original sample order (in epoch 0) *
            meta_train_sample_list = sample_list
            print("\ncurrent actual used total sample number in meta-training: {}\n".format(len(meta_train_sample_list)))

        # current evaluation setting: zero-shot test
        if test_mode == "zero":
            model.eval()
            with torch.no_grad():
                # test_iterations_dict records the iteration number required for each test task
                test_zero_total_metric, test_zero_task_metric, _ = MAML_zero_shot_test(test_set, model, iterations_dict=cfg.test_iterations_dict,
                                max_time=cfg.get("test_time"), device=device)
            # current test results
            # print("\nEPOCH %d" % epoch, "TEST zero-shot total metrics over all test tasks:\n", test_zero_total_metric) # results over all test tasks
            print("\nEPOCH %d" % epoch, "TEST zero-shot separate metrics for each test task:\n", test_zero_task_metric) # results under each test task
        # current evaluation setting: few-shot test
        elif test_mode == 'few':
            test_few_total_metric, test_few_task_metric, _, _ = MAML_few_shot_test(
                test_set, model, iterations_dict=cfg.test_iterations_dict, few_shot_num=cfg.get("few_shot_num"),
                few_epoch=cfg.get("few_epoch"), few_lr=cfg.get("few_lr"), few_weight_decay=few_weight_decay,
                fix_encoder_few_finetuning=cfg.get("fix_encoder_few_finetuning"), device=device)

            print("\nEPOCH %d" % epoch, "TEST few-shot separate metrics for each test task:\n", test_few_task_metric)

        # save the model in the last epoch and the model with the best training evaluation metric (currently no validation set)
        # output training loss may contain some auxiliary loss values, not just containing the main loss based on the regression metric
        if cfg.model_save_mode == "direct":
            if eval_mode == "min":
                if metric < best_train_eval:
                    torch.save(model.state_dict(), os.path.join(model_path, "best_train_" + model_save_name))
                    if test_mode == "zero":
                        best_train_epoch, best_train_eval, test_eval_on_best_train_epoch = epoch, metric, test_zero_task_metric
                        print("\nPER-TASK-TEST results on current best training epoch %d:\n" % best_train_epoch, test_eval_on_best_train_epoch)
                    elif test_mode == "few":
                        best_train_epoch, best_train_eval = epoch, metric

            elif eval_mode == "max":
                if metric > best_train_eval:
                    torch.save(model.state_dict(), os.path.join(model_path, "best_train_" + model_save_name))
                    if test_mode == "zero":
                        best_train_epoch, best_train_eval, test_eval_on_best_train_epoch = epoch, metric, test_zero_task_metric
                        print("\nPER-TASK-TEST results on current best training epoch %d:\n" % best_train_epoch, test_eval_on_best_train_epoch)
                    elif test_mode == "few":
                        best_train_epoch, best_train_eval = epoch, metric

            if epoch == cfg.num_epoch - 1:
                # current model parameters are those in the MAML wrapper
                torch.save(model.state_dict(), os.path.join(model_path, "last_train_" + model_save_name))
        else:
            print("current model save mode {} is not supported".format(cfg.model_save_mode))
            raise NotImplementedError

        # record GPU occupation status
        # https://blog.csdn.net/HJC256ZY/article/details/106516894
        if torch.cuda.is_available():
            logger.warning("max GPU memory: %.1f MiB" % (torch.cuda.max_memory_allocated() / 1e6))
            torch.cuda.reset_peak_memory_stats()

        # 在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次。
        # 所以如果scheduler.step()是放在mini-batch里面，那么step_size指的是经过这么多次迭代，学习率改变一次。
        if scheduler_pos == "epoch":
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                # reduce learning rate when a metric has stopped improving
                # thus for ReduceLROnPlateau, an extra metric will be sent into the 'step' function
                scheduler.step(loss)
            elif scheduler is not None:
                scheduler.step()

    # save the training loss curve as DataFrame
    # save in the same saving path as label files, /cgdiff_energy_injection/downstream_files/LABEL_STORAGE_FOLDER
    print("the path to save the training loss curves: {}".format(task_path))
    train_loss = np.array(train_loss).reshape(-1, 1)
    train_loss = pd.DataFrame(train_loss, columns=[model_save_name])
    train_loss.to_csv(os.path.join(task_path, "MAML_train_loss.csv"), index=False)

    return best_train_epoch, test_eval_on_best_train_epoch, meta_train_sample_list


# few-shot meta-test
# * current logic is to retrieve former val_shot * shot_num samples of each test task (based on ordered sample names) to perform fine-tuning *
# * besides, the complete few-shot calculation logic will be performed after the complete meta-training process *
# * under current settings, to obtain relatively robust results, the fine-tuning batch set for each task can be optionally shuffled inside *
def MAML_few_shot_test(dataset, model, iterations_dict, few_shot_num, few_epoch, few_lr, few_weight_decay, fix_encoder_few_finetuning, device=None):
    t = tqdm.tqdm(dataset)
    # get the whole iteration list of the DataLoader for meta-test
    all_tasks = list(iterations_dict.keys())
    iterations_list = [iterations_dict[key] for key in all_tasks]
    # print(iterations_list) # [5, 47]
    # print(model.training) # True
    # print([name for name, param in model.named_parameters() if not param.requires_grad]) # the encoder still can keep fixed

    # need to collect the fine-tuning batches for each task (batches are properly ordered in Test_BatchSampler)
    new_task_flag, start_test_flag, task_order_counter, test_batch_counter, train_subset_collector = True, False, 0, 0, []
    total_preds, total_targets = [], [] # record over tasks
    metric_per_task = {} # record within tasks
    fewshot_train_sample_list, fewshot_test_sample_list = {}, {}

    for batch_id, batch in enumerate(t):
        # new_task_flag will be re-opened when a test task finishes (for the next task)
        if new_task_flag:
            current_task_batch_num = iterations_list[task_order_counter]

            # * ensure that for each test task, 'few_shot_num' number of batches for fine-tuning can be guaranteed *
            # * the actual total batch number is also controlled by 'val_shot' argument in this few-shot case *
            assert current_task_batch_num > few_shot_num and few_shot_num > 0 and isinstance(few_shot_num, int), \
                "The specified few_shot_num ({}) should be smaller than the total batch number ({}) for current test task ({}).".\
                    format(few_shot_num, current_task_batch_num, all_tasks[task_order_counter])

            # get the batch number for independent test under current task
            current_remain_batch_num = current_task_batch_num - few_shot_num
            new_task_flag = False
            # new_task_flag will occur at the beginning of each task until next new task, thus we create it here
            metric_per_task[all_tasks[task_order_counter]] = {"pred": [], "target": []}

        # only collect batches under fine-tuning phase instead of independent test phase
        if start_test_flag == False:
            train_subset_collector.append(batch)

        # start to finetune and test on current task
        if len(train_subset_collector) == few_shot_num:
            # flag of the independent test for current task (starting from the next batch)
            start_test_flag = True
            # test_batch_counter is used to count the test batches already evaluated for current test task
            test_batch_counter = 0
            try:
                train_subset_collector = [utils.cuda(i, device=device) for i in train_subset_collector]

                # record the fine-tuning samples in current meta-test task following the original input order to 'few_shot_finetune_loop'
                fewshot_train_sample_list[all_tasks[task_order_counter]] = []
                for i in train_subset_collector:
                    fewshot_train_sample_list[all_tasks[task_order_counter]].extend(i["name"])

                # will perform model.train() inside the function, and return a deepcopy model specified optimized for current task
                current_task_model = few_shot_finetune_loop(
                    model, train_subset_collector, few_epoch, few_lr, few_weight_decay=few_weight_decay,
                    task_name=all_tasks[task_order_counter], fix_encoder_few_finetuning=fix_encoder_few_finetuning)

                # clean the train_subset_collector for the next task
                train_subset_collector = []
                # directly go to the next batch for independent tests
                continue

            except RuntimeError as e:
                train_subset_collector = []
                if "CUDA out of memory" not in str(e): raise (e)
                torch.cuda.empty_cache()
                print("Skipped batch due to OOM", flush=True)
                continue

        # when finishing fine-tuning on a test task, an independent test will be run subsequently
        if start_test_flag == True:
            test_batch_counter += 1
            try:
                # set the task-specific deepcopy model to evaluation mode
                current_task_model.eval()
                with torch.no_grad():
                    batch = utils.cuda(batch, device=device)

                    # record the test samples in current meta-test task following the original input order
                    if all_tasks[task_order_counter] not in fewshot_test_sample_list.keys():
                        fewshot_test_sample_list[all_tasks[task_order_counter]] = []

                    fewshot_test_sample_list[all_tasks[task_order_counter]].extend(batch["name"])

                    pred, target = current_task_model.predict_and_target(batch)

            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise (e)
                torch.cuda.empty_cache()
                print("Skipped batch due to OOM", flush=True)
                continue

            total_preds.append(pred)
            total_targets.append(target)
            metric_per_task[all_tasks[task_order_counter]]["pred"].append(pred)
            metric_per_task[all_tasks[task_order_counter]]["target"].append(target)

        # the end of one test task, refreshing the corresponding flags and counters for the next task
        if test_batch_counter == current_remain_batch_num:
            new_task_flag = True
            start_test_flag = False
            task_order_counter += 1

    # evaluate the independent test results over tasks and within tasks
    total_pred = utils.cat(total_preds)
    total_target = utils.cat(total_targets)
    total_metric = model.task.evaluate(total_pred, total_target)
    total_metric = {eval: round(float(total_metric[eval]), 4) for eval in total_metric}

    task_metric = {}
    for task in all_tasks:
        pred = utils.cat(metric_per_task[task]["pred"])
        target = utils.cat(metric_per_task[task]["target"])
        current_task_metric = model.task.evaluate(pred, target)
        task_metric[task] = {eval: round(float(current_task_metric[eval]), 4) for eval in current_task_metric}

    return total_metric, task_metric, fewshot_train_sample_list, fewshot_test_sample_list


def few_shot_finetune_loop(model, train_subset_collector, few_epoch, few_lr, few_weight_decay, task_name, fix_encoder_few_finetuning):
    # start the fine-tuning process (finishing the complete fine-tuning for single test task here)

    # * only copy the task wrapper for the fine-tuning, as the extra parameters in MAML wrapper are only used for meta-training, *
    # * which are irrelevant to the meta-test fine-tuning *
    finetune_model = deepcopy(model.task)

    # fine-tuning all model parameters in the encoder and decoder (i.e., the task wrapper) if fix_encoder_few_finetuning == False
    # if True, no effect to the rest of the logic in current fine-tuning loop function
    if fix_encoder_few_finetuning == False:
        for param in finetune_model.parameters():
            param.requires_grad = True

    # * please note that the parameters in the encoder could be fixed (defined in the yaml configuration file) *
    # * print([name for name, param in finetune_model.named_parameters() if not param.requires_grad]) *
    if few_weight_decay is not None:
        # finetune_opi = optim.AdamW(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=few_lr, weight_decay=float(few_weight_decay))
        finetune_opi = optim.Adam(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=few_lr, weight_decay=float(few_weight_decay))
    else:
        # finetune_opi = optim.AdamW(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=few_lr)
        finetune_opi = optim.Adam(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=few_lr)

    # set to the training mode
    finetune_model.train()

    for epoch in range(few_epoch):
        epoch_finetune_loss = 0.
        random.shuffle(train_subset_collector) # optional, to shuffle the order of fine-tuning batches (rather than the fine-tuning samples)
        # batch-wise update
        for subset_id, subset in enumerate(train_subset_collector):
            # get the result for current fine-tuning batch
            loss, metric, _ = finetune_model(batch=subset, graph=model.input_graph_construction(subset))

            finetune_opi.zero_grad()
            loss.backward()
            finetune_opi.step()
            epoch_finetune_loss += loss.item() # return a Python scalar to be added into current finetune_loss

            # the last batch in current epoch, giving the fine-tuning loss for current epoch
            if subset_id == len(train_subset_collector) - 1:
                epoch_last_loss = epoch_finetune_loss / len(train_subset_collector) # loss per batch
                # print("current {}/{} fine-tuning loss for the task {}: {:.8f}".format(epoch, few_epoch, task_name, epoch_last_loss))
                epoch_finetune_loss = 0. # refreshing for the next epoch

    return finetune_model


# zero-shot meta-test
def MAML_zero_shot_test(dataset, model, iterations_dict=None, max_time=None, device=None, ignore=None):
    start = time.time()
    t = tqdm.tqdm(dataset)

    # record the evaluation results for each test task
    # the order of iterations_dict has been already fixed in Test_BatchSampler
    # thus here we do not need to sort list(iterations_dict.keys()) again
    if iterations_dict is not None:
        metric_per_task, all_tasks = {}, list(iterations_dict.keys()) # keys are 'str' (set in Test_BatchSampler)
        iterations_list = [[key for _ in range(iterations_dict[key])] for key in all_tasks]
        iterations_list = [item for sublist in iterations_list for item in sublist]
        # ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '2']
        # representing in zero-shot setting, for each batch to be tested, which task it belongs to
    else:
        # not to record the test evaluation metric per task
        task_metric = None

    ignore_counter = 0
    preds, targets = [], []
    zeroshot_test_sample_list = []
    # * under the zero-shot setting, each batch generated from dataloader only contains samples from one task, *
    # * e.g., batch1 for task1, batch2 for task1, batch3 for task1, batch4 for task2, batch5 for task2, batch6 for task3, etc. *
    # * thus, can also consider to collect results for individual tasks *
    for batch_id, batch in enumerate(t):
        if max_time and (time.time() - start) > 60 * max_time: break

        # skip the former N batches for evaluation
        if isinstance(ignore, int) and ignore > 0:
            ignore_counter += 1
            if ignore_counter == ignore:
                ignore = None
            continue

        try:
            batch = utils.cuda(batch, device=device)

            # record the test samples in current meta-test task following the original input order
            # currently zeroshot_test_sample_list will not distinguish samples from different tasks (just record the original input order)
            zeroshot_test_sample_list.extend(batch["name"])

            # call to get predictions and labels under current batch simultaneously
            pred, target = model.validation_step(batch)
            # collect pred and target according to the task ids
            if iterations_dict is not None:
                current_task = iterations_list[batch_id]
                if current_task not in metric_per_task.keys():
                    metric_per_task[current_task] = {'pred': [], 'target': []}
                    metric_per_task[current_task]['pred'].append(pred)
                    metric_per_task[current_task]['target'].append(target)
                else:
                    metric_per_task[current_task]['pred'].append(pred)
                    metric_per_task[current_task]['target'].append(target)

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise (e)
            torch.cuda.empty_cache()
            print("Skipped batch due to OOM", flush=True)
            continue

        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)

    # 1) the results over the samples of the whole test set:
    # the 'null' labels will be masked in the 'evaluate' function
    # the 'evaluate' function is put in the task wrapper of the MAML wrapper
    total_metric = model.task.evaluate(pred, target)
    total_metric = {eval: round(float(total_metric[eval]), 4) for eval in total_metric}
    # dict_keys(['mean absolute error [binding_affinity]', 'root mean squared error [binding_affinity]', 'pearsonr [binding_affinity]'])

    # 2) the results over each task:
    if iterations_dict is not None:
        task_metric = {}
        for task in all_tasks:
            pred = utils.cat(metric_per_task[task]['pred'])
            target = utils.cat(metric_per_task[task]['target'])
            current_task_metric = model.task.evaluate(pred, target)
            task_metric[task] = {eval: round(float(current_task_metric[eval]), 4) for eval in current_task_metric}

    return total_metric, task_metric, zeroshot_test_sample_list


# meta-training (model parameter update) process for a batch
def MAML_inner_loop(dataset, model, epoch, batch_size, optimizer=None, scheduler=None, scheduler_pos='epoch', max_time=None, device=None):
    start = time.time()
    t = tqdm.tqdm(dataset)
    total_loss, total_count, total_eval = 0, 0, []

    # * in current sampling logic, each effectively generated batch will contain task number equalling to pre-defined batch size, *
    # * and S and Q sets for each task are complete (i.e., K_shot samples for S set, at least one sample for Q set) *
    # * in current implementation, each S/Q set of corresponding task in a batch will be given one by one orderly (defined in Sampler) *
    # * based on this, here we need to collect these parts to assemble the complete batch for meta-training *
    subset_collector, sample_list = [], []
    for subset in t: # iterate the dataloader to get each set
        subset_collector.append(subset)
        if epoch == 0:
            # print(subset["name"]) # ['3-1260-5ME5', '1-811-1REW', '1-5701-1C4Z', '1-4737-4I77', '1-7052-2WPT']
            sample_list.extend(subset["name"])

        # the data structure to be taken by the forward process of MAML wrapper:
        # [task1 S set], [task1 Q set], [task2 S set], [task2 Q set]
        # the following batch_size indicates the task number for the batch structure to be sent to MAML wrapper, and 2 represents S+Q for each task
        if len(subset_collector) % (batch_size * 2) == 0:
            # start a complete batch operation
            # end current loop if the time used exceeds the set max_time * 60 (no matter whether current batch already ends)
            if max_time and (time.time() - start) > 60 * max_time: break
            # clean gradient for current batch
            if optimizer: optimizer.zero_grad()

            try:
                subset_collector = [utils.cuda(i, device=device) for i in subset_collector]
                if scheduler_pos == "batch":
                    loss, metric = model(subset_collector, epoch, scheduler)
                else:
                    loss, metric = model(subset_collector, epoch)
                # print(metric) # tensor([1.3857, 1.4554]), provide one evaluation result for each task (metric type used defined in PropertyPrediction)

            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise (e)
                torch.cuda.empty_cache()
                print("Skipped batch due to OOM", flush=True)
                continue
            # record the loss for current batch
            total_loss += float(loss) # float: transform a tensor with gradient into a python float scalar
            total_count += 1 # count current batch for MAML wrapper
            total_eval.append(metric)

            if optimizer:
                try:
                    loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e): raise (e)
                    torch.cuda.empty_cache()
                    print("Skipped batch due to OOM", flush=True)
                    continue

            # log current average loss over all past batches in current epoch
            t.set_description(f"{total_loss / total_count:.8f}")
            # empty subset_collector for the next complete batch
            subset_collector = []

    return total_loss / total_count, total_eval, sample_list


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    dirname = os.path.basename(args.config).split('.')[0] + "_yaml" + "_seed" +str(args.seed)
    working_dir = util._create_working_directory(cfg, dirname=dirname) # dirname: providing extra suffix for working_dir
    print("current working dictionary:", working_dir)

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

    if cfg.dataset["class"] in ["MAMLDataset"]:
        _dataset = core.Configurable.load_config_dict(cfg.dataset)
        meta_train_set, meta_test_set = _dataset.split()

    # after defining and initializing the Dataset, the Sampler needs to be defined based on Dataset,
    # to give the rule of object sampling in each epoch (based on the generated object/sample retrival ids for each set)
    # start to customize the Dataloader Sampler for MAML (the validation set is not necessarily needed in current setting)
    no_task_unify = "unified_sample_num" not in cfg.dataset.keys()
    if no_task_unify:
        # meta-training sampler for regression tasks (without sample number balance between different tasks, e.g., up/down-sampling)
        train_sampler = cg_downstream_maml_sampler.Train_FewshotBatchSampler(
        # train_sampler = cg_downstream_maml_sampler.Train_FewshotBatchSampler_Random(
            # * here batch size defines how many tasks will be sampled each batch (rather than how many samples) *
            meta_train_set, dataset=_dataset, K_shot=cfg.dataset.k_shot, K_query=cfg.dataset.k_query, query_min=cfg.dataset.query_min,
            batch_size=cfg.engine.batch_size, seed=seed)
    else:
        # meta-training sampler for regression tasks (with sample number balance between different tasks)
        train_sampler = cg_downstream_maml_sampler.Train_balanced_FewshotBatchSampler(
            # * here batch size defines how many tasks will be sampled each batch (rather than how many samples) *
            meta_train_set, dataset=_dataset, K_shot=cfg.dataset.k_shot, K_query=cfg.dataset.k_query, query_min=cfg.dataset.query_min,
            batch_size=cfg.engine.batch_size, seed=seed, unified_sample_num=cfg.dataset.get("unified_sample_num"))

    # * total iteration number during meta-training, currently used in CosineAnnealingLR scheduler *
    cfg.train.iterations = train_sampler.iterations

    # zero-shot/few-shot test sampler
    test_sampler = cg_downstream_maml_sampler.Test_BatchSampler(meta_test_set, dataset=_dataset, val_shot=cfg.dataset.val_shot,
        random_shuffle_seed=cfg.dataset.get("random_shuffle_seed"))
    # record the iteration number required for each test task (based on pre-defined val_shot value)
    cfg.train.test_iterations_dict = test_sampler.iterations_dict

    # next need to define the Dataloader, model + checkpoint, optimizer + scheduler, etc.
    # * the pre-trained parameters are loaded here for the NN decoder (fixed or trainable) *
    # * the output task wrapper here defines the basic model architecture, forward process logic, and evaluation ways, *
    # * and the returned optimizer and scheduler are specifically customized for parameters in this task wrapper *
    scheduler_name = cfg.scheduler.get("class") if "scheduler" in cfg.keys() else None
    task, optimizer, scheduler = cg_maml_util.build_ppi_maml_solver(cfg, meta_train_set, None, meta_test_set) # user solver default: False

    # * an extra MAML wrapper outside the current task wrapper (which only defines the basic forward process logic) is defined *
    # * providing the logic related to MAML-based meta-training and meta-test *
    maml_task = cg_maml_util.MAML_wrapper(cfg, task)
    # put the extra parameters provided in MAML wrapper into the optimizer (optional)
    if cfg.train.get("maml_param_optim") == True:
        maml_extra_parameters = [param for name, param in maml_task.named_parameters() if not name.startswith("task")]
        # * the scheduler controlling the learning rate change is not necessarily needed to be configured again *
        # * add_param_group() only take the parameters as the input, insensitive to the order and name of the parameters *
        optimizer.add_param_group({"params": maml_extra_parameters})
        # print(optimizer.param_groups) # other optimization hyperparameters follow those specified for basic task wrapper
        # named_parameters() can iterate all tensors in every submodule of the input module
        # for k, v in maml_task.named_parameters():
        #     print(k, v.requires_grad)

        # https://github.com/pytorch/pytorch/issues/104361
        assert scheduler_name != "ReduceLROnPlateau", \
            "Using Optimizer.add_param_group on the optimizer attached to the scheduler " \
            "while the ReduceLrOnPlateau is instantiated is not supported."

    # make the task wrapper enter the cuda
    device = torch.device(cfg.engine.gpus[0]) if cfg.engine.gpus else torch.device("cpu")
    if device.type == "cuda":
        maml_task = maml_task.cuda(device)

    # define the Dataloader for MAML
    # data.DataLoader inherited default settings from torch.utils.data.DataLoader,
    # except for explicitly defining collate_fn, which batches graph structured data based on the defined 'PackProtein' class
    # pin_memory = True argument can also be also considered, to accelerate GPU data access based on extra CPU storage, while causing extra burden for CPU
    train_loader, test_loader = [
        data.DataLoader(subset, batch_sampler=sampler, num_workers=cfg.engine.num_worker)
            # initialize two independent dataloaders using the dataset and sampler respective to the training set and test set
            # for the sampler, it yields absolute sample retrieval ids (based on each subset) to replace the sampling logic from batch_size and shuffle
            for subset, sampler in zip([meta_train_set, meta_test_set], [train_sampler, test_sampler])]

    # start to record the time
    current_time, t0 = datetime.now().strftime("%Y_%m_%d__%H_%M_%S"), time.time()

    # define the model saving name, after that, start the training process
    model_save_name = "maml_seed{}_bs{}_epoch{}_dim{}_radius{}_{}_extra_{}.pth".format(args.seed, cfg.engine.batch_size, cfg.train.num_epoch,
                    cfg.task.model.embedding_dim, cfg.task.graph_construction_model.edge_layers[0]["radius"], current_time, cfg.extra)

    # run the training loop
    task_path = os.path.dirname(cfg.dataset["label_path"]) # use current label file path as the basic path of training information storage
    model_path = os.path.join(task_path, "downstream_model_storage/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("current model storage path:", model_path)
    test_mode = cfg.train.get("test_mode")
    print("\ncurrent test mode:", test_mode)

    # * all_used_sample_list is used to indicate relevant hyperparameters when saving all samples actually used in meta-training + meta-test *
    # * the saved sample list preserves original sample input order in batch-wise training, which can be further sorted/removing duplicates *
    splitting_file_name = os.path.splitext(os.path.basename(cfg.dataset.index_path))[0].split('_')
    # for comparison with supervised learning
    all_used_sample_list = ("{}shot_kshot{}_kquery{}_valshot{}_valshuffle{}_fewshotnum{}_nounify{}_batch{}_{}_{}_{}_{}.json").format(test_mode,
        cfg.dataset["k_shot"], cfg.dataset["k_query"], cfg.dataset["val_shot"], cfg.dataset.get("random_shuffle_seed"), cfg.train["few_shot_num"],
        no_task_unify, cfg.engine.batch_size, splitting_file_name[1], splitting_file_name[-3], splitting_file_name[-2], splitting_file_name[-1])
    # print(all_used_sample_list) # zeroshot_kshot5_kquery10_valshot10_fewshotnum2_nounifyTrue_batch2_extratest_tm-0.55_crit-ss_cnum-5.json

    # * another variable for saving current all samples except for those used in fine-tuning of few-shot cases *
    # * it varies between zero-shot and few-shot cases, since the samples joining parameter adjustment and pure inference could be different *
    # * for all_sample_wo_finetuning for zero-shot and few-shot cases, the 'train' part should contain same samples with only meta-training ones involved *
    all_sample_wo_finetuning = ("WOF-{}shot_kshot{}_kquery{}_valshot{}_valshuffle{}_fewshotnum{}_nounify{}_batch{}_{}_{}_{}_{}.json").format(test_mode,
        cfg.dataset["k_shot"], cfg.dataset["k_query"], cfg.dataset["val_shot"], cfg.dataset.get("random_shuffle_seed"), cfg.train["few_shot_num"],
        no_task_unify, cfg.engine.batch_size, splitting_file_name[1], splitting_file_name[-3], splitting_file_name[-2], splitting_file_name[-1])
        # for providing meta-training samples to further fine-tune the encoder based on supervised learning
        
    if cfg.get("extra"):
        all_used_sample_list = all_used_sample_list.split(".")[0] + "_" + cfg.get("extra") + ".json"
        all_sample_wo_finetuning = all_sample_wo_finetuning.split(".")[0] + "_" + cfg.get("extra") + ".json"
        print("all used sample list:", all_used_sample_list)
        print("all sample wo finetuning:", all_sample_wo_finetuning)

    # ** for the evaluation results reported inside, it is based on the model trained on the both meta-training samples and further fine-tuning samples **
    best_train_epoch, test_eval_on_best_train_epoch, meta_train_sample_list = train(cfg.train, maml_task, optimizer, scheduler, train_loader, test_loader,
            device, model_save_name, task_path, model_path, cfg.engine["batch_size"], cfg.train.get("early_stop"), test_mode=test_mode, eval_mode="min",
            few_weight_decay=cfg.optimizer.get("weight_decay"))

    # * check the test results based on the best train epoch and the last epoch separately and independently *
    # * based on current logic, no matter which 'model_save_mode' is used, the 'best train epoch' and 'last epoch' models will be saved *
    # * the difference is that the logic to save the 'best train epoch' model could be different *
    # 1) on the best epoch
    maml_task.load_state_dict(torch.load(os.path.join(model_path, "best_train_" + model_save_name)))
    # best_epoch_wrapper = deepcopy(maml_task)
    if test_mode == "zero":
        maml_task.eval()
        with torch.no_grad():
            total_metric_best_epoch, task_metric_best_epoch, zeroshot_test_sample_list = MAML_zero_shot_test(test_loader, maml_task,
                iterations_dict=cfg.train.test_iterations_dict, max_time=None, device=device, ignore=zero_ignore_batch_num) # set max_time to None

    elif test_mode == "few":
        total_metric_best_epoch, task_metric_best_epoch, fewshot_train_sample_list, fewshot_test_sample_list = MAML_few_shot_test(
            test_loader, maml_task, iterations_dict=cfg.train.test_iterations_dict, few_shot_num=cfg.train.get("few_shot_num"),
            few_epoch=cfg.train.get("few_epoch"), few_lr=cfg.train.get("few_lr"), few_weight_decay=cfg.optimizer.get("weight_decay"),
            fix_encoder_few_finetuning=cfg.train.get("fix_encoder_few_finetuning"), device=device)

    print("\nINDEPENDENT {}-shot overall test metric on the best epoch {}:\n".format(test_mode, best_train_epoch), total_metric_best_epoch)
    print("\nCORRESPONDING task-wise metric on the best epoch {}:".format(best_train_epoch))
    for current_task in task_metric_best_epoch:
        print('task {}:'.format(current_task), task_metric_best_epoch[current_task])

    # 2) on the last epoch
    # including parameters in the trained encoder and decoder (thus the further fine-tuning is still based on this)
    # print([name for name in torch.load(os.path.join(model_path, "last_train_" + model_save_name))])
    maml_task.load_state_dict(torch.load(os.path.join(model_path, "last_train_" + model_save_name)))
    # last_epoch_wrapper = deepcopy(maml_task)
    if test_mode == "zero":
        maml_task.eval()
        with torch.no_grad():
            # total_metric_last_epoch: the results over the samples of the whole test set under the last epoch model
            total_metric_last_epoch, task_metric_last_epoch, zeroshot_test_sample_list = MAML_zero_shot_test(test_loader, maml_task,
                iterations_dict=cfg.train.test_iterations_dict, max_time=None, device=device, ignore=zero_ignore_batch_num)

    elif test_mode == "few":
        total_metric_last_epoch, task_metric_last_epoch, fewshot_train_sample_list, fewshot_test_sample_list = MAML_few_shot_test(
            test_loader, maml_task, iterations_dict=cfg.train.test_iterations_dict, few_shot_num=cfg.train.get("few_shot_num"),
            few_epoch=cfg.train.get("few_epoch"), few_lr=cfg.train.get("few_lr"), few_weight_decay=cfg.optimizer.get("weight_decay"),
            fix_encoder_few_finetuning=cfg.train.get("fix_encoder_few_finetuning"), device=device)

    print("\nINDEPENDENT {}-shot overall test metric on the last epoch:\n".format(test_mode), total_metric_last_epoch)
    print("\nCORRESPONDING task-wise metric on the last epoch:")
    for current_task in task_metric_last_epoch:
        print('task {}:'.format(current_task), task_metric_last_epoch[current_task])

    t1 = time.time()
    print(f"\ncurrent model name for storage is: {model_save_name}\n")
    print(f"\ntotal elapsed time of the downstream training and required inference processes: {t1 - t0:.4f}\n")

    # automatically store all samples to be trained/tested in meta-training and meta-test in *original sample order*
    if test_mode == "zero":
        total_sample_list = {"train": [], "test": []}
        total_sample_list["train"].extend(meta_train_sample_list)
        total_sample_list["test"].extend(zeroshot_test_sample_list)
        # same to all_used_sample_list in zero-shot case
        with open(os.path.join(task_path, all_sample_wo_finetuning), 'w') as f:
            json.dump(total_sample_list, f)

    elif test_mode == "few":
        total_sample_list = {"train": [], "test": []}
        total_sample_list["train"].extend(meta_train_sample_list)

        for key in fewshot_test_sample_list:
            few_sample = fewshot_test_sample_list[key]
            print("independent test sample number for meta-test task {} in few-shot setting: {}".format(key, len(few_sample)))
            total_sample_list["test"].extend(few_sample)

        # different from all_used_sample_list in few-shot case (without few-shot fine-tuning samples in 'train')
        with open(os.path.join(task_path, all_sample_wo_finetuning), 'w') as f:
            json.dump(total_sample_list, f)

        finetune_sample_list = {}
        for key in fewshot_train_sample_list:
            few_sample = fewshot_train_sample_list[key]
            print("fine-tuning sample number for meta-test task {} in few-shot setting: {}".format(key, len(few_sample)))
            print("specific sample names include:", sorted([i.split('\\')[-1] for i in few_sample]))
            total_sample_list["train"].extend(few_sample)
            # record the fine-tuning samples in current meta-test task following the original input order to 'few_shot_finetune_loop'
            finetune_sample_list[key] = few_sample

        with open(os.path.join(task_path, "F-" + all_used_sample_list), 'w') as f:
            json.dump(finetune_sample_list, f)

    print("\ntotal sample number to be trained in current {}-shot setting is: {}".format(test_mode, len(total_sample_list["train"])))
    print("total sample number to be tested in current {}-shot setting is: {}".format(test_mode, len(total_sample_list["test"])))

    # * for the meta-training samples recorded, they are saved based on original batch generation without further sorting or duplicate removal *
    with open(os.path.join(task_path, all_used_sample_list), 'w') as f:
        json.dump(total_sample_list, f)

    # for param1, param2 in zip(best_epoch_wrapper.parameters(), last_epoch_wrapper.parameters()):
    #     print(torch.sum(param1 - param2))

    # 'mean absolute error [binding_affinity]', 'root mean squared error [binding_affinity]', 'pearsonr [binding_affinity]'
    # task_suffix = cfg.train.eval_metric.split()[-1]
    # MAE = 'mean absolute error' + ' ' + task_suffix
    # RMSE = 'root mean squared error' + ' ' + task_suffix
    # PEARSON = 'pearsonr' + ' ' + task_suffix
    # MAE, RMSE, PEARSON = float(total_metric_last_epoch[MAE]), float(total_metric_last_epoch[RMSE]), float(total_metric_last_epoch[PEARSON])
    # print(f'normal evaluation metrics under the model of the last epoch, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}, Pearson: {PEARSON:.4f}')























