from typing import List
import torch
from torch_geometric.data import TemporalData
import os
import random

class TemporalDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges succesive events of a
    :class:`torch_geometric.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
            from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, data: TemporalData, batch_size: int = 1, **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'shuffle' in kwargs:
            del kwargs['shuffle']

        self.data = data
        self.events_per_batch = batch_size

        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = range(0, len(data) - batch_size, batch_size)
        else:
            arange = range(0, len(data), batch_size)

        super().__init__(arange, 1, shuffle=False, collate_fn=self, **kwargs)
    def __call__(self, arange: List[int]) -> TemporalData:
        return self.data[arange[0]:arange[0] + self.events_per_batch]


def graph_loader(dir):
    data = []
    directory = [d for d in os.listdir(dir) if '.DS_Store' not in d]
    directory2 = random.sample(directory, 4)
    for d in directory2:
        data_dir=[]
        if os.path.isdir(dir+d):
            for f in os.listdir(dir+d+'/'):
                if '.DS_Store' not in f:
                    data_dir.append(torch.load(dir+d+'/'+f))
        data.append(data_dir)
    return data