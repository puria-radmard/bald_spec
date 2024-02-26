import torch
from torch import Tensor as _T

from abc import ABC, abstractmethod
from tqdm import tqdm

from data.base import ActiveLearningDataLoaderBase

class AcquisitionFunctionBase(ABC):
    "Purely the evaluation of the acquisiton function - nothing fancy like batching at all!"

    @abstractmethod
    def __init__(self) -> None:
        pass

    def evaluate_function_on_batch(
        self,
        databatch: _T,
        *args, **kwargs
    ) -> _T:
        raise NotImplementedError

    def evaluate_function_on_data_loader(
        self,
        data_loader: ActiveLearningDataLoaderBase,
        use_tqdm = False,
        *args, **kwargs,
    ) -> _T:

        data_loader.querying()
        all_scores_dicts = []
        all_indices = []

        for unlabelled_batch in tqdm(data_loader, total = data_loader.num_batches, disable=not use_tqdm):

            data = unlabelled_batch['data']                                   # [B, dim_in]
            indices = unlabelled_batch['indices']     

            new_scores_dict = self.evaluate_function_on_batch(data, *args, **kwargs)
            all_scores_dicts.append(new_scores_dict)
            all_indices.append(torch.tensor(indices))

        ret = {'indices': torch.concat(all_indices, 0)}
        for k in new_scores_dict.keys():
            ret[k] = torch.concat([d[k] for d in all_scores_dicts], 0)

        assert 'scores' in ret.keys()

        return ret
