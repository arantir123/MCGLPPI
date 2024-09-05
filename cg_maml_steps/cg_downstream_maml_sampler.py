import copy
import math
import random
import numpy as np


class Train_FewshotBatchSampler:
    def __init__(self, subset, dataset, K_shot, K_query, query_min, batch_size, seed, **kwargs):

        self.subset = subset # selected subset for meta-training
        self.dataset = dataset # complete pytorch Dataset
        self.K_shot = K_shot # k_shot for the S set of meta-training (used in Sampler)
        self.K_query = K_query # k_query for the Q set of meta-training (used in Sampler)

        # to ensure the minimum sample number for each Q set,
        # i.e., discarding remained samples for a task (for which the supported Q set < query_min),
        # after complete k_shot S set + k_query Q set sampling in meta-training
        self.query_min = query_min
        assert type(self.query_min) == int and self.query_min >= 1, "The query minimum arugment should be an integer larger than 1."

        # * the definition of batch size here is different from the conventional ones *
        # * here batch size defines how many tasks will be sampled each batch (rather than how many samples) *
        # * thus, the actual sample number each batch is determined by batch_size + k_shot + k_query *
        # * under current experimental settings, usually task number should be larger than the batch size *
        self.batch_size = batch_size
        self.seed = seed
        # acquire the sample absolute ids on the training set for each cluster/task
        self.task2sampid = self.dataset.train_task2sampid # task to (training set) absolute sample ids for meta-training
        assert min([len(self.task2sampid[i]) for i in self.task2sampid]) > self.K_shot, \
            "Make sure that the minimum sample number for each training task is larger than the pre-defined K_shot value: {}.".format(self.K_shot)
        self.task_id = list(self.task2sampid.keys())

        task2sampid = copy.deepcopy(self.task2sampid)
        tasks = sorted(list(task2sampid.keys())) # fix the task order
        print("the index of currently used training tasks: {}\n".format(tasks))
        print("the total sample number for each meta-training task before batch assemble:")
        for task in tasks:
            print("task {}: {}".format(task, len(task2sampid[task])))
        assert len(tasks) >= self.batch_size, \
            "The initial task number should be larger than the batch size for controlling the sampled task number each batch."

        self.iterations = 0
        # 这里的逻辑应该是先预计算meta-training中每个epoch所包含的iteration数，基于batch_size以及k_shot
        # 预计算的逻辑是，保证每个batch中所包含的任务数都一定等于batch_size，在这个的基础上根据k_shot对每个蛋白任务采样
        # 所以随着采样好的batch数增加，整个meta-training集所包含的可用任务数会越来越少，且每个batch所包含的蛋白任务可能都是不相同的（显然对应的k-shot样本也不会重复）
        # 这个过程持续到，最后剩余的蛋白任务数少于预定义的batch_size（这些剩余任务可能由于对应的初始样本数较多发生了样本剩余，但大概率在之前的batch中已经经过了采样），
        # 剩下的样本会被丢弃，但是之前的batch已经采样了比较充分的各个任务的k-shot样本，来组成最终的batch集
        # 这个batch集所包含的batch数也就是epoch训练所需要的iteration数
        while len(tasks) >= self.batch_size:
            # keep the same random seed during this loop (i.e., re-determining it at the beginning of each loop)
            random.seed(self.seed)
            task_weights = [len(task2sampid[task]) for task in tasks]
            total_count = sum(task_weights)
            task_weights = [count / total_count for count in task_weights] # obtain sample ratio for each training task
            # print(task_weights) # [0.308300395256917, 0.3294261959929126, 0.25855254191086274, 0.10372086683930762]

            # the order of 'tasks' is fixed above
            selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size) # sample used tasks in current batch
            while len(set(selected_tasks)) != self.batch_size: # ensure the sampled task number equalling to batch size
                selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)

            # pre-sample for each selected task based on k_shot (S set) and k_query (Q set)
            for selected_task in selected_tasks:
                # selected_task_samplist is given orderly based on protein names
                selected_task_samplist = copy.deepcopy(task2sampid[selected_task])
                # random sampling without considering the sampling order (for S set): [481, 393, 561, 281, 497]
                # * in binary classification tasks, k_shot sampling will be applied to positive and negative samples simultanouesly (in meta-training, S set) *
                # * and k_query//2 sampling will be applied to positive and negative samples as well (in meta-training, Q set) *
                # * while current task is the regression task, which uses original k_shot and k_query values on S and Q sets (without distinguishing pos/neg samples) *
                # * thus, the scale of k_shot and k_query should be further tuned for current regression task *
                selected_S_samples = random.sample(selected_task_samplist, self.K_shot)

                # remove the sampled k_shot samples from deepcopied task2sampid
                for selected_S_sample in selected_S_samples:
                    # using the deepcopy version (i.e., task2sampid to perform 'remove' operation)
                    task2sampid[selected_task].remove(selected_S_sample)
                    selected_task_samplist.remove(selected_S_sample)

                # sample current Q set
                # * 按照当前的逻辑，可以明确，方法假定对于一个（元训练）任务，其对应的初始样本数至少大于设定的K_shot值，这样至少可以完成一次下面定义的S集+Q集（至少会有一个样本）采集 *
                # * 在这个基础上，按照下面的逻辑，这次采样结束后，若当前任务的剩余样本仍支持至少一次S集+Q集（样本数至少为query_min）的采样，该任务就会被保留，用于下个batch的task sampling *
                if len(selected_task_samplist) >= self.K_query:
                    selected_Q_samples = random.sample(selected_task_samplist, self.K_query)
                else:
                    length = len(selected_task_samplist)
                    selected_Q_samples = random.sample(selected_task_samplist, length)

                for selected_Q_sample in selected_Q_samples:
                    task2sampid[selected_task].remove(selected_Q_sample)
                    selected_task_samplist.remove(selected_Q_sample)

                # pop current task if the remained samples cannot support a full K_shot for S update plus at least one sample for Q calculation
                # if len(selected_task_samplist) <= self.K_shot: # to remain at least 1 sample for current Q set
                # * to remain at least 2 samples for current Q set, which may discard more samples for meta-training, *
                # * while can keep BN working normally with a minimum batch sample number *
                # if len(selected_task_samplist) <= self.K_shot + 1:
                if len(selected_task_samplist) <= self.K_shot + self.query_min - 1:
                    task2sampid.pop(selected_task)
                    tasks.remove(selected_task)
            # the iteration is added 1 when the whole procedure of a batch passes
            self.iterations += 1

    # sample batch_size number of tasks based on their contained sample numbers (and pop them from task2sampid)
    def weighted_sample(self, task2sampid):
        select_tasks = []
        for _ in range(self.batch_size):
            weight = [len(value) for value in task2sampid.values()] # get sample number for each task
            select_task = random.choices(list(task2sampid.keys()), weights=weight) # random select one task according to the weight
            task2sampid.pop(select_task[0])
            select_tasks.append(select_task[0])
        return select_tasks

    # formally define the batch generation process (for each epoch)
    # in current sampling logic, each effectively generated batch will contain task number equalling to pre-defined batch size
    # and S and Q sets for each task are complete (i.e., K_shot samples for S set, at least one sample for Q set)
    def __iter__(self):
        task2sampid = copy.deepcopy(self.task2sampid)
        tasks = sorted(list(task2sampid.keys()))
        # print(sum([len(task2sampid[i]) for i in task2sampid.keys()])) # 757
        # number of samples in the training and test sets: 757, 513

        for _ in range(self.iterations):
            index_batch = []
            random.seed(self.seed)
            task_weights = [len(task2sampid[task]) for task in tasks]
            total_count = sum(task_weights)
            task_weights = [count / total_count for count in task_weights] # obtain sample ratio for each training task

            # * random.sample() does not produce repeating elements, while random.choices() does *
            selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size) # sample used tasks in current batch
            while len(set(selected_tasks)) != self.batch_size: # ensure the sampled task number equalling to batch size
                selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)

            for selected_task in selected_tasks:
                selected_task_samplist = copy.deepcopy(task2sampid[selected_task])
                # S set sampling
                selected_S_samples = random.sample(selected_task_samplist, self.K_shot)
                for selected_S_sample in selected_S_samples:
                    task2sampid[selected_task].remove(selected_S_sample)
                    selected_task_samplist.remove(selected_S_sample)
                # Q set sampling
                if len(selected_task_samplist) >= self.K_query:
                    selected_Q_samples = random.sample(selected_task_samplist, self.K_query)
                else:
                    length = len(selected_task_samplist)
                    selected_Q_samples = random.sample(selected_task_samplist, length)
                for selected_Q_sample in selected_Q_samples:
                    task2sampid[selected_task].remove(selected_Q_sample)
                    selected_task_samplist.remove(selected_Q_sample)

                if len(selected_task_samplist) <= self.K_shot + self.query_min - 1:
                    task2sampid.pop(selected_task)
                    tasks.remove(selected_task)

                # * new logic to generate indices for a batch: [task1 S set], [task1 Q set], [task2 S set], [task2 Q set] (and collect them in MAML training) *
                # * the above operations are exactly same as those in __init__, which ensure the re-producibility of generated iterations number in __init__ *
                # * next need to yield current sampled batch as a list of samples for the next step *
                index_batch.extend(selected_S_samples)
                random.shuffle(selected_Q_samples) # shuffle the Q set each time
                index_batch.extend(selected_Q_samples)

                # * new added snippet for the new batch generation logic *
                index_batch_tasks = []
                index_batch_tasks.append(selected_S_samples)
                index_batch_tasks.append(selected_Q_samples)
                for index_batch_task in index_batch_tasks:
                    # yield each S/Q set for one time
                    yield index_batch_task

            # * original logic to generate indices for a batch: concat[task1 S set, task1 Q set, task2 S set, task2 Q set, ...] *
            # print('indices for current index_batch:', index_batch)
            # print('names for current index_batch:', np.array([i.split('\\')[-1] for i in self.dataset.pdb_files])[index_batch])
            # yield index_batch

    def __len__(self):
        return self.iterations


# * another version of Train_FewshotBatchSampler which uses different random seeds to generate different batch sets in different epochs *
# * besides, the random seed used in every epoch is also under controlled, ensuring the reproducibility *
class ControlledRandomSeedGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.counter = 0

    def generate_seed(self):
        seed = self.seed + self.counter
        self.counter += 1
        return seed

class Train_FewshotBatchSampler_Random:
    def __init__(self, subset, dataset, K_shot, K_query, query_min, batch_size, seed, **kwargs):

        self.subset = subset # selected subset for meta-training
        self.dataset = dataset # complete pytorch Dataset
        self.K_shot = K_shot # k_shot for the S set of meta-training (used in Sampler)
        self.K_query = K_query # k_query for the Q set of meta-training (used in Sampler)

        # to ensure the minimum sample number for each Q set,
        # i.e., discarding remained samples for a task (for which the supported Q set < query_min),
        # after complete k_shot S set + k_query Q set sampling in meta-training
        self.query_min = query_min
        assert type(self.query_min) == int and self.query_min >= 1, "The query minimum arugment should be an integer larger than 1."

        # * the definition of batch size here is different from the conventional ones *
        # * here batch size defines how many tasks will be sampled each batch (rather than how many samples) *
        # * thus, the actual sample number each batch is determined by batch_size + k_shot + k_query *
        # * under current experimental settings, usually task number should be larger than the batch size *
        self.batch_size = batch_size
        self.seed = seed
        # random seed generator (for 'random' function) controlled by pre-defined seed value
        self.seed_generator = ControlledRandomSeedGenerator(seed=self.seed)

        # acquire the sample absolute ids on the training set for each cluster/task
        self.task2sampid = self.dataset.train_task2sampid # task to (training set) absolute sample ids for meta-training
        assert min([len(self.task2sampid[i]) for i in self.task2sampid]) > self.K_shot, \
            "Make sure that the minimum sample number for each training task is larger than the pre-defined K_shot value: {}.".format(self.K_shot)
        self.task_id = list(self.task2sampid.keys())

        task2sampid = copy.deepcopy(self.task2sampid)
        tasks = sorted(list(task2sampid.keys())) # fix the task order
        print("the index of currently used training tasks: {}\n".format(tasks))
        print("the total sample number for each meta-training task before batch assemble:")
        for task in tasks:
            print("task {}: {}".format(task, len(task2sampid[task])))
        assert len(tasks) >= self.batch_size, \
            "The initial task number should be larger than the batch size for controlling the sampled task number each batch."

        self.iterations = 0
        # 这里的逻辑是先预估计一个meta-training中每个epoch所包含的iteration数，基于batch_size，k_shot，以及一个完全固定的随机种子（但在__iter__中是会发生变化的）
        # 预估计的逻辑是，保证每个batch中所包含的任务数都一定等于batch_size，在这个的基础上根据k_shot对每个蛋白任务采样
        # 所以随着采样好的batch数增加，整个meta-training集所包含的可用任务数会越来越少，且每个batch所包含的蛋白任务可能都是不相同的（显然对应的k-shot样本也不会重复）
        # 这个过程持续到，最后剩余的蛋白任务数少于预定义的batch_size（这些剩余任务可能由于对应的初始样本数较多发生了样本剩余，但大概率在之前的batch中已经经过了采样），
        # 剩下的样本会被丢弃，但是之前的batch已经采样了比较充分的各个任务的k-shot样本（基于由任务样本数决定的抽样权重），来组成最终的batch集
        # 这个batch集所包含的batch数也就是epoch训练所需要的iteration数
        while len(tasks) >= self.batch_size:
            # keep the same random seed during this loop (i.e., re-determining it at the beginning of each loop)
            random.seed(self.seed)
            task_weights = [len(task2sampid[task]) for task in tasks]
            total_count = sum(task_weights)
            task_weights = [count / total_count for count in task_weights] # obtain sample ratio for each training task
            # print(task_weights) # [0.308300395256917, 0.3294261959929126, 0.25855254191086274, 0.10372086683930762]

            # the order of 'tasks' is fixed above
            selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size) # sample used tasks in current batch
            while len(set(selected_tasks)) != self.batch_size: # ensure the sampled task number equalling to batch size
                selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)

            # pre-sample for each selected task based on k_shot (S set) and k_query (Q set)
            for selected_task in selected_tasks:
                # selected_task_samplist is given orderly based on protein names
                selected_task_samplist = copy.deepcopy(task2sampid[selected_task])
                # random sampling without considering the sampling order (for S set): [481, 393, 561, 281, 497]
                # * in binary classification tasks, k_shot sampling will be applied to positive and negative samples simultanouesly (in meta-training, S set) *
                # * and k_query//2 sampling will be applied to positive and negative samples as well (in meta-training, Q set) *
                # * while current task is the regression task, which uses original k_shot and k_query values on S and Q sets (without distinguishing pos/neg samples) *
                # * thus, the scale of k_shot and k_query should be further tuned for current regression task *
                selected_S_samples = random.sample(selected_task_samplist, self.K_shot)

                # remove the sampled k_shot samples from deepcopied task2sampid
                for selected_S_sample in selected_S_samples:
                    # using the deepcopy version (i.e., task2sampid to perform 'remove' operation)
                    task2sampid[selected_task].remove(selected_S_sample)
                    selected_task_samplist.remove(selected_S_sample)

                # sample current Q set
                # * 按照当前的逻辑，可以明确，方法假定对于一个（元训练）任务，其对应的初始样本数至少大于设定的K_shot值，这样至少可以完成一次下面定义的S集+Q集（至少会有一个样本）采集 *
                # * 在这个基础上，按照下面的逻辑，这次采样结束后，若当前任务的剩余样本仍支持至少一次S集+Q集（样本数至少为query_min）的采样，该任务就会被保留，用于下个batch的task sampling *
                if len(selected_task_samplist) >= self.K_query:
                    selected_Q_samples = random.sample(selected_task_samplist, self.K_query)
                else:
                    length = len(selected_task_samplist)
                    selected_Q_samples = random.sample(selected_task_samplist, length)

                for selected_Q_sample in selected_Q_samples:
                    task2sampid[selected_task].remove(selected_Q_sample)
                    selected_task_samplist.remove(selected_Q_sample)

                # pop current task if the remained samples cannot support a full K_shot for S update plus at least one sample for Q calculation
                # if len(selected_task_samplist) <= self.K_shot: # to remain at least 1 sample for current Q set
                # * to remain at least 2 samples for current Q set, which may discard more samples for meta-training, *
                # * while can keep BN working normally with a minimum batch sample number *
                # if len(selected_task_samplist) <= self.K_shot + 1:
                if len(selected_task_samplist) <= self.K_shot + self.query_min - 1:
                    task2sampid.pop(selected_task)
                    tasks.remove(selected_task)
            # the iteration is added 1 when the whole procedure of a batch passes
            self.iterations += 1

    # formally define the batch generation process (for each epoch)
    # in current sampling logic, each effectively generated batch will contain task number equalling to pre-defined batch size
    # and S and Q sets for each task are complete (i.e., K_shot samples for S set, at least one sample for Q set)
    def __iter__(self):
        task2sampid = copy.deepcopy(self.task2sampid)
        tasks = sorted(list(task2sampid.keys()))
        # print(sum([len(task2sampid[i]) for i in task2sampid.keys()])) # 757
        # number of samples in the training and test sets: 757, 513

        # * set the random seed for 'random' function in current epoch (to generate unique batch sets for current batch) *
        # * it will influence the selection of tasks and samples in each task *
        new_seed = self.seed_generator.generate_seed()
        random.seed(new_seed)
        print("current batch set sampling seed is: {}".format(new_seed))

        # finish the iteration in current epoch when the remained tasks are insufficient
        while len(tasks) >= self.batch_size:
            index_batch = []
            task_weights = [len(task2sampid[task]) for task in tasks]
            total_count = sum(task_weights)
            task_weights = [count / total_count for count in task_weights] # obtain sample ratio for each training task

            # * random.sample() does not produce repeating elements, while random.choices() does *
            selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size) # sample used tasks in current batch
            while len(set(selected_tasks)) != self.batch_size: # ensure the sampled task number equalling to batch size
                selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)

            for selected_task in selected_tasks:
                selected_task_samplist = copy.deepcopy(task2sampid[selected_task])
                # S set sampling
                selected_S_samples = random.sample(selected_task_samplist, self.K_shot)
                for selected_S_sample in selected_S_samples:
                    task2sampid[selected_task].remove(selected_S_sample)
                    selected_task_samplist.remove(selected_S_sample)
                # Q set sampling
                if len(selected_task_samplist) >= self.K_query:
                    selected_Q_samples = random.sample(selected_task_samplist, self.K_query)
                else:
                    length = len(selected_task_samplist)
                    selected_Q_samples = random.sample(selected_task_samplist, length)
                for selected_Q_sample in selected_Q_samples:
                    task2sampid[selected_task].remove(selected_Q_sample)
                    selected_task_samplist.remove(selected_Q_sample)

                if len(selected_task_samplist) <= self.K_shot + self.query_min - 1:
                    task2sampid.pop(selected_task)
                    tasks.remove(selected_task)

                # * new logic to generate indices for a batch: [task1 S set], [task1 Q set], [task2 S set], [task2 Q set] (and collect them in MAML training) *
                # * the above operations are exactly same as those in __init__, which ensure the re-producibility of generated iterations number in __init__ *
                # * next need to yield current sampled batch as a list of samples for the next step *
                index_batch.extend(selected_S_samples)
                random.shuffle(selected_Q_samples) # shuffle the Q set each time
                index_batch.extend(selected_Q_samples)

                # * new added snippet for the new batch generation logic *
                index_batch_tasks = []
                index_batch_tasks.append(selected_S_samples)
                index_batch_tasks.append(selected_Q_samples)
                for index_batch_task in index_batch_tasks:
                    # yield each S/Q set for one time
                    yield index_batch_task

            # * original logic to generate indices for a batch: concat[task1 S set, task1 Q set, task2 S set, task2 Q set, ...] *
            # print('indices for current index_batch:', index_batch)
            # print('names for current index_batch:', np.array([i.split('\\')[-1] for i in self.dataset.pdb_files])[index_batch])
            # yield index_batch

    def __len__(self):
        return self.iterations


# meta-training sampler for regression tasks (with sample number balance between different tasks using up/down-sampling)
class Train_balanced_FewshotBatchSampler:
    def __init__(self, subset, dataset, K_shot, K_query, query_min, batch_size, seed, unified_sample_num=None, **kwargs):

        self.subset = subset # selected subset for meta-training
        self.dataset = dataset # complete pytorch Dataset (meta-training + meta-test)
        self.K_shot = K_shot # k_shot for the S set of meta-training (used in Sampler)
        self.K_query = K_query # k_query for the Q set of meta-training (used in Sampler)
        assert isinstance(self.K_shot, int) and self.K_shot >= 1, "The K_shot argument should be an integer larger than 1."
        assert isinstance(self.K_query, int) and self.K_query >= 1, "The k_query argument should be an integer larger than 1."

        # to ensure the minimum sample number for each Q set,
        # i.e., discarding remained samples for a task (for which the available Q set < query_min),
        # after complete k_shot S set + k_query Q set sampling in a meta-training batch
        self.query_min = query_min
        assert isinstance(self.query_min, int) and self.query_min >= 1, "The query minimum argument should be an integer larger than 1."

        # * the definition of batch size here is different from the conventional ones *
        # * here batch size defines how many tasks will be sampled each batch (rather than how many samples) *
        # * thus, the actual sample number each batch is determined by batch_size + k_shot + k_query *
        # * under current experimental settings, usually task number should be larger than the batch size *
        self.batch_size = batch_size
        self.seed = seed

        # acquire the sample absolute ids in the meta-training set for each cluster/task
        self.task2sampid = self.dataset.train_task2sampid # task name to (training set) absolute sample ids for meta-training
        tasks = sorted(list(self.task2sampid.keys())) # fix the task order, the following calculation will be based on this order
        print("the index of currently used training tasks:", tasks)
        print("the total sample number for each meta-training task before batch assembles:")
        for task in tasks:
            print("task {}: {}".format(task, len(self.task2sampid[task])))

        # a check to avoid the initial total sample number for a meta-training task too small
        assert min([len(self.task2sampid[i]) for i in self.task2sampid]) > self.K_shot, \
            "Make sure the minimum sample number for each meta-training task larger than the pre-defined K_shot value: {}.".format(self.K_shot)

        assert len(tasks) >= self.batch_size, \
            "The initial task number should be larger than the batch size for controlling the sampled task number each batch."

        # * under current logic, we need to determine the unified initial sample number across each meta-training task *
        # * in this case, we are able to know, for each task, whether the up-sampling or down-sampling should be adopted *
        # * after that, the corresponding batch assembly allocation can be proceeded based on the ratio between k_shot and k_query *
        if isinstance(unified_sample_num, int):
            # support int 'unified_sample_num' larger than 1 as the final unified sample number
            self.unified_sample_num = unified_sample_num if unified_sample_num > 1 else None
        elif isinstance(unified_sample_num, float):
            # support (0, 1.0] float 'unified_sample_num' as the ratio for the final unified sample number determination
            max_tasksample_num = max([len(self.task2sampid[i]) for i in self.task2sampid])
            self.unified_sample_num = math.ceil(max_tasksample_num * unified_sample_num) if 0 < unified_sample_num <= 1.0 else None
        else:
            self.unified_sample_num = None
        # use sample number average value over all meta-training tasks as the unified sample number instead
        # average of task numbers != max task number * 0.5
        if self.unified_sample_num == None:
            self.unified_sample_num = math.ceil(np.mean([len(self.task2sampid[task]) for task in tasks]))
        print("\ncurrent unified sample number for each task in meta-training: {}\n".format(self.unified_sample_num))

        # * calculate the ratio between k_shot and k_query (earlier than actual sampling, to avoid data leakage caused by sampling with replacement) *
        # * in current setting, S/Q set *for each batch* should be assembled from the corresponding sample pools separately *
        # * for the further sampling, we hope every S/Q set *in every task* could have the same number of samples *
        support_ratio, query_ratio = self.K_shot / (self.K_shot + self.K_query), self.K_query / (self.K_shot + self.K_query)
        # determine the final sample number for every S/Q set,
        # this is the sample number that each set will reach after the further sampling on each original task
        support_num, query_num = math.ceil(self.unified_sample_num * support_ratio), math.ceil(self.unified_sample_num * query_ratio)

        # start to create the balanced task2sampid for the following batch assembly
        self.task2sampid_balanced = {}
        for task in tasks:
            # start to split original S and Q sets, then sample them separately to avoid data leakage
            current_samples = self.task2sampid[task]
            # https://www.geeksforgeeks.org/python-random-sample-function/
            current_support = random.sample(current_samples, math.ceil(len(current_samples) * support_ratio)) # output: indices with random order
            # here we keep the *original index order*, for which we do not need to consider in the actual batch assembly stage
            current_support = sorted(current_support)
            current_query = sorted([i for i in current_samples if i not in current_support])
            # * math.ceil above and the check here ensure that original S/Q splitting based on support_ratio/query_ratio before further sampling can have at least one sample *
            # * in current logic, the original and enhanced splittings both follow the support_ratio/query_ratio (the above k_shot number check is not sufficient for this) *
            # * e.g., there are 2 samples in current task, if k_shot=1 and k_query=10, the k_shot number check passes, but, current_support: int(2 * (1/11)) == 0 (error) *
            assert len(current_query) > 0, \
                "the original picked query set for task {} does not contain effective samples before unification sampling.".format(task)

            # start to perform sampling on each S/Q set in every task (to make them reach above support_num/query_num size)
            if support_num > len(current_support): # up-sampling
                current_support = sorted(np.random.choice(current_support, support_num, replace=True))
            else: # down-sampling
                current_support = sorted(np.random.choice(current_support, support_num, replace=False))

            if query_num > len(current_query): # up-sampling
                current_query = sorted(np.random.choice(current_query, query_num, replace=True))
            else: # down-sampling
                current_query = sorted(np.random.choice(current_query, query_num, replace=False))
            assert len(current_query) >= self.query_min, \
                "the query sample number after unification sampling in task {} should be larger than the defined query_min: {}".format(task, self.query_min)

            self.task2sampid_balanced[task] = {"support": current_support, "query": current_query}
        # the deep-copied self.task2sampid_balanced is used to calculate the below batch assembly iteratively
        task2sampid_balanced = copy.deepcopy(self.task2sampid_balanced)

        # further determine the exact iteration number if sampling each S/Q set into support_num/query_num size based on current task number and batch size etc.,
        # although each S/Q set here has same sample number, effected by task number and batch size and the randomness of task choosing process for each batch,
        # the final iteration still needs to determine again
        self.iterations = 0
        while len(tasks) >= self.batch_size:
            # keep the same random seed during this loop (i.e., re-determining it at the beginning of each loop)
            random.seed(self.seed)

            # under current iterative 'iteration' calculation, although each task (the S and Q sets) after unification shares the same initial sample number,
            # due to the randomness of task sampling based on batch size, k_query, and k_shot, task weights based on current status still need to be calculated
            task_weights = [sum([len(task2sampid_balanced[task][i]) for i in task2sampid_balanced[task]]) for task in tasks] # based on S+Q set in each task
            total_count = sum(task_weights)
            task_weights = [count / total_count for count in task_weights] # obtain sample ratio for each meta-training task

            # * the order of 'tasks' is fixed above, while the order of 'selected_tasks' is random *
            # * random.sample() does not produce repeating elements, while random.choices() does (elements chosen with replacement) *
            selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size) # sample the used tasks in current batch
            while len(set(selected_tasks)) != self.batch_size: # ensure the sampled task number equalling to batch size
                selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)

            for selected_task in selected_tasks:
                # ** current batch assembly logic: complete S batch should be guaranteed, discarding remained samples in current task if this is not satisfied **
                selected_task_samplist = copy.deepcopy(task2sampid_balanced[selected_task]) # for easier access to required attributes

                # sample current S samples
                # will arise error if input is shorter than defined K_shot
                selected_S_samples = random.sample(selected_task_samplist["support"], self.K_shot) # output: random order
                for selected_S_sample in selected_S_samples:
                    # using the deep-copied version to perform the iterative removal (i.e., based on task2sampid_balanced to perform 'remove' operation)
                    task2sampid_balanced[selected_task]["support"].remove(selected_S_sample)
                    selected_task_samplist["support"].remove(selected_S_sample)

                # sample current Q samples
                # by default, for the first time entering here, the remained Q sample number in current task is larger than self.query_min (already check above)
                current_query_num = len(selected_task_samplist["query"])
                if current_query_num >= self.K_query:
                    # will also arise error if input is shorter than defined K_query
                    selected_Q_samples = random.sample(selected_task_samplist["query"], self.K_query) # output: random order
                else:
                    length = current_query_num
                    selected_Q_samples = random.sample(selected_task_samplist["query"], length)

                for selected_Q_sample in selected_Q_samples:
                    task2sampid_balanced[selected_task]["query"].remove(selected_Q_sample)
                    selected_task_samplist["query"].remove(selected_Q_sample)

                # * logic for discarding current task if finding insufficient remained samples *
                if len(selected_task_samplist["support"]) < self.K_shot or len(selected_task_samplist["query"]) < self.query_min: # query_min >= 1 is checked above
                    task2sampid_balanced.pop(selected_task)
                    tasks.remove(selected_task)

            # the iteration is added 1 when the whole procedure of a batch passes
            self.iterations += 1

    # formally define the batch generation process (for each epoch)
    # in current sampling logic, each effectively generated batch will contain task number equalling to pre-defined batch size
    # and S and Q sets for each task are complete (i.e., K_shot samples for S set, at least one sample for Q set)
    def __iter__(self):
        task2sampid_balanced = copy.deepcopy(self.task2sampid_balanced)
        tasks = sorted(list(task2sampid_balanced.keys()))

        for _ in range(self.iterations):
            index_batch = [] # store a list of batch samples for current epoch (old logic)
            random.seed(self.seed)
            task_weights = [sum([len(task2sampid_balanced[task][i]) for i in task2sampid_balanced[task]]) for task in tasks]
            total_count = sum(task_weights)
            task_weights = [count / total_count for count in task_weights]

            selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)
            while len(set(selected_tasks)) != self.batch_size:
                selected_tasks = random.choices(tasks, weights=task_weights, k=self.batch_size)

            for selected_task in selected_tasks:
                selected_task_samplist = copy.deepcopy(task2sampid_balanced[selected_task])
                # S set sampling
                selected_S_samples = random.sample(selected_task_samplist["support"], self.K_shot)

                for selected_S_sample in selected_S_samples:
                    task2sampid_balanced[selected_task]["support"].remove(selected_S_sample)
                    selected_task_samplist["support"].remove(selected_S_sample)

                # Q set sampling
                current_query_num = len(selected_task_samplist["query"])
                if current_query_num >= self.K_query:
                    selected_Q_samples = random.sample(selected_task_samplist["query"], self.K_query)
                else:
                    length = current_query_num
                    selected_Q_samples = random.sample(selected_task_samplist["query"], length)

                for selected_Q_sample in selected_Q_samples:
                    task2sampid_balanced[selected_task]["query"].remove(selected_Q_sample)
                    selected_task_samplist["query"].remove(selected_Q_sample)

                if len(selected_task_samplist["support"]) < self.K_shot or len(selected_task_samplist["query"]) < self.query_min:
                    task2sampid_balanced.pop(selected_task)
                    tasks.remove(selected_task)

                # * new logic to generate indices for a batch: [task1 S set], [task1 Q set], [task2 S set], [task2 Q set] (and collect them in MAML training) *
                # * the above operations are exactly same as those in __init__, which ensure the re-producibility of generated iterations number in __init__ *
                # * next need to yield current sampled batch as a list of samples for the next calculation step *
                # index_batch.extend(selected_S_samples) # old logic
                random.shuffle(selected_Q_samples) # further shuffle the Q set each time
                # index_batch.extend(selected_Q_samples) # old logic

                # * new added snippet for the new batch generation logic *
                # * since there could exist up-sampling for tasks with fewer samples, different S batches could contain duplicate samples *
                # * this also applies to the different Q batches generated *
                index_batch_tasks = []
                index_batch_tasks.append(selected_S_samples)
                index_batch_tasks.append(selected_Q_samples)
                # print(selected_Q_samples)
                for index_batch_task in index_batch_tasks:
                    # only one S set or Q set (of one task) is yielded each time
                    yield index_batch_task

            # * original logic to generate indices for a batch: concat[task1 S set, task1 Q set, task2 S set, task2 Q set, ...] *
            # print('indices for current index_batch:', index_batch)
            # print('names for current index_batch:', np.array([i.split('\\')[-1] for i in self.dataset.pdb_files])[index_batch])
            # yield index_batch

    def __len__(self):
        return self.iterations


# * test batchsampler for zero-shot and few-shot setting *
# * for few-shot learning, current logic is to retrieve former val_shot*shot_num samples of each test task (based on ordered sample names) to perform finetuning *
# * due to that current all test samples are already ordered based on sample names in Dataset (MAMLDataset), resulting in the ordered task2sampid here *
# * thus, for fast code implementation, when iterating the DataLoader, we can collect the former shot_num batches to temporarily stored for finetuning, *
# * and the rest of batches will be treated as the pure test samples, since val_shot*shot_num samples should be a small value, temporary storage will be cheap *
class Test_BatchSampler:
    def __init__(self, subset, dataset, val_shot, random_shuffle_seed=None, **kwargs):

        self.subset = subset
        self.dataset = dataset
        self.val_shot = val_shot
        # self.task2sampid = self.dataset.train_task2sampid # mapping dict of the task to absolute sample ids for the training set
        self.task2sampid = self.dataset.test_task2sampid # mapping dict of the task to absolute sample ids for the test set
        # print(self.task2sampid) # {49: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        self.iterations = 0
        # calculate how many batches will be needed for iterating all samples of every task in testing (based on pre-defined val_shot)
        self.iterations_dict = {}
        for task_name in sorted(list(self.task2sampid.keys())): # fix the iteration order
            # print(task_name)
            if len(self.task2sampid[task_name]) % self.val_shot == 0:
                iteration = len(self.task2sampid[task_name]) // self.val_shot
                self.iterations += iteration
            else:
                iteration = len(self.task2sampid[task_name]) // self.val_shot + 1
                self.iterations += iteration
            self.iterations_dict[str(task_name)] = iteration
        # print(self.iterations_dict.keys())

        if isinstance(random_shuffle_seed, int):
            old_seed = random.getstate()
            random.seed(random_shuffle_seed)
            # shuffle self.task2sampid to change the sample output order based on given random seed
            for task_name in sorted(list(self.task2sampid.keys())):
                random.shuffle(self.task2sampid[task_name])
            random.setstate(old_seed)

    def __iter__(self):
        # * in Test_BatchSampler, each yielded batch only contains samples from one task *
        # * in Train_FewshotBatchSampler, each batch contains multiple tasks, while the coressponding samples are ordered properly, *
        # * e.g., task1 S set, task1 Q set, task2 S set, task2 Q set (for easily retrieving each part) *
        for task_name in sorted(list(self.task2sampid.keys())): # fix the iteration order
            if len(self.task2sampid[task_name]) % self.val_shot == 0:
                n = len(self.task2sampid[task_name]) // self.val_shot
            else:
                n = len(self.task2sampid[task_name]) // self.val_shot + 1

            # iterate all samples in current task
            for i in range(n):
                yield self.task2sampid[task_name][i * self.val_shot: (i + 1) * self.val_shot]

    def __len__(self):
        return self.iterations














