from collections import deque
from collections.abc import Mapping, Sequence

import torch
from torchdrug import data
# from torch.utils.data import DataLoader


def graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0] # batch input: a list of element (truncated graph1 + graph2)
    # elem: {'graph': CG22_Protein(num_atom=237, num_bond=544), 'graph2': CG22_Protein(num_atom=237, num_bond=544)}
    # elem: {'graph': Protein(num_atom=813, num_bond=1682, num_residue=100), 'graph2': Protein(num_atom=813, num_bond=1682, num_residue=100)}
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        # print(1)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, float):
        # print(2)
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        # print(3)
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        # print(4)
        return batch
    elif isinstance(elem, data.Graph):
        # print(elem, data.Graph, isinstance(elem, data.Graph))
        # CG22_Protein(num_atom=237, num_bond=544), <class 'torchdrug.data.graph.Graph'>, True
        # print(batch) # [CG22_Protein(num_atom=231, num_bond=524, num_residue=100), CG22_Protein(num_atom=120, num_bond=244, num_residue=62)]
        # after isinstance(elem, Mapping), batch here has become a list containing all graphs under current key type (i.e., graph1 or graph2)
        # print(5)
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        # get keys of the individual element, retrieve all values/graphs corresponding to each key in current batch,
        # and then pack all values/graphs under current key type (e.g., graph1, graph2) respectively
        # print(6)
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('Each element in list of batch should be of equal size')
        # print(7)
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


class DataLoader(torch.utils.data.DataLoader):
    """
    Extended data loader for batching graph structured data.

    See `torch.utils.data.DataLoader`_ for more details.

    .. _torch.utils.data.DataLoader:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Parameters:
        dataset (Dataset): dataset from which to load the data
        batch_size (int, optional): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch
        sampler (Sampler, optional): sampler that draws single sample from the dataset
        batch_sampler (Sampler, optional): sampler that draws a mini-batch of data from the dataset
        num_workers (int, optional): how many subprocesses to use for data loading
        collate_fn (callable, optional): merge a list of samples into a mini-batch
        kwargs: keyword arguments for `torch.utils.data.DataLoader`_
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=graph_collate, **kwargs):

        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         **kwargs)


class DataQueue(torch.utils.data.Dataset):

    def __init__(self):
        self.queue = deque()

    def append(self, item):
        self.queue.append(item)

    def pop(self):
        self.queue.popleft()

    def __getitem__(self, index):
        return self.queue[index]

    def __len__(self):
        return len(self.deque)


class ExperienceReplay(torch.utils.data.DataLoader):

    def __init__(self, cache_size, batch_size=1, shuffle=True, **kwargs):
        super(ExperienceReplay, self).__init__(DataQueue(), batch_size, shuffle, **kwargs)
        self.cache_size = cache_size

    def update(self, items):
        for item in items:
            self.dataset.append(item)
        while len(self.dataset) > self.cache_size:
            self.dataset.pop()

    @property
    def cold(self):
        return len(self.dataset) < self.cache_size