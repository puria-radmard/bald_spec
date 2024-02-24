from abc import ABC
from math import ceil
from typing import List
from random import shuffle
from torch import Tensor as _T


class ActiveLearningDataLoaderBase(ABC):

    training: bool
    data: _T                # Of which the first axis indexes independent items
    labels: _T              # Same ^
    training_indices: set
    querying_indices: set
    training_batch_size: int
    querying_batch_size: int
    num_batches: int

    def __init__(self, data: _T, labels: _T, training_indices: set, training_batch_size: int, querying_batch_size: int) -> None:
        self.data = data
        self.labels = labels
        self.training_batch_size = training_batch_size
        self.querying_batch_size = querying_batch_size
        self.training_indices = training_indices
        assert len(training_indices) < self.data.shape[0]
        self.querying_indices = set(range(self.data.shape[0])) - self.training_indices
        self.train()

    def train(self):
        self.training = True
        self.num_batches = ceil(len(self.training_indices) / self.training_batch_size)

    def querying(self):
        self.training = False
        self.num_batches = ceil(len(self.querying_indices) / self.querying_batch_size)

    def __setitem__(self, indices: List[int], new_labels: _T):
        assert not self.training, "Cannot update labelled set, use self.querying() first"
        self.labels[indices] = new_labels
        self.querying_indices.remove(set(indices))
        self.training_indices.add(set(indices))
        import pdb; pdb.set_trace()
    
    def __iter__(self):
        if self.training:
            indices = list(self.training_indices)
            shuffle(indices)
            bs = self.training_batch_size
        else:
            indices = list(self.querying_indices)
            bs = self.querying_batch_size

        for i in range(self.num_bathes):
            batch_indices = indices[i*bs:(i+1)*bs]
            batch = {'data': self.data[batch_indices]}
            if self.training:
                batch['labels'] = self.labels[batch_indices]
            else:
                batch['indices'] = batch_indices
            yield batch
