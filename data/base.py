from abc import ABC
from math import ceil
from typing import List
from random import shuffle
from torch import Tensor as _T


class ActiveLearningDataLoaderBase(ABC):

    data_mode: str
    data: _T                    # Of which the first axis indexes independent items
    labels: _T                  # Same ^
    test_data: _T               # Same ^
    test_labels: _T             # Same ^
    training_indices: set
    querying_indices: set
    training_batch_size: int
    querying_batch_size: int
    num_batches: int

    def __init__(self, data: _T, labels: _T, test_data: _T, test_labels: _T, training_indices: set, training_batch_size: int, querying_batch_size: int) -> None:
        self.data = data
        self.labels = labels
        self.training_batch_size = training_batch_size
        self.querying_batch_size = querying_batch_size
        self.training_indices = training_indices
        assert len(training_indices) < self.data.shape[0]
        self.querying_indices = set(range(self.data.shape[0])) - self.training_indices
        self.set_data_mode('train')

        self.test_data = test_data
        self.test_labels = test_labels

    def set_data_mode(self, mode: str):
        if mode == 'train':
            self.num_batches = ceil(len(self.training_indices) / self.training_batch_size)
        elif mode == 'query':
            self.num_batches = ceil(len(self.querying_indices) / self.querying_batch_size)
        elif mode == 'test':
            self.num_batches = ceil(len(self.test_data) / self.querying_batch_size)
        else:
            raise ValueError(f'{mode} is not a valuable data mode for an ActiveLearningDataLoader')
        self.data_mode = mode

    def label(self, indices: List[int]):
        assert self.data_mode == 'query', "Cannot update labelled or test set, use self.set_data_mode('query') first"
        self.querying_indices = self.querying_indices - set(indices.tolist())
        self.training_indices.update(set(indices.tolist()))
    
    def __iter__(self):
        if self.data_mode == 'train':
            indices = list(self.training_indices)
            shuffle(indices)
            bs = self.training_batch_size
            relevant_dataset = self.data
            relevant_labels = self.labels
        elif self.data_mode == 'query':
            indices = list(self.querying_indices)
            bs = self.querying_batch_size
            relevant_dataset = self.data
        elif self.data_mode == 'test':
            indices = list(range(len(self.test_data)))
            bs = self.querying_batch_size
            relevant_dataset = self.test_data
            relevant_labels = self.test_data

        # Systematic scan in all cases: this is bonafide inference not an i.i.d. approximation remember!
        for i in range(self.num_batches):
            batch_indices = indices[i*bs:(i+1)*bs]
            batch = {'data': relevant_dataset[batch_indices]}
            if self.data_mode in ['train', 'test']:
                batch['labels'] = relevant_labels[batch_indices]
            else:
                batch['indices'] = batch_indices
            yield batch
