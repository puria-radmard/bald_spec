import torch

from acquisition_functions.base import AcquisitionFunctionBase


class RandomAcquisitionFunction(AcquisitionFunctionBase):

    def __init__(self) -> None:
        pass
    
    def evaluate_function_on_batch(self, databatch, *args, **kwargs):
        return {'scores': torch.randn(databatch.shape[0])}

