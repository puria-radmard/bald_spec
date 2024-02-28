import torch
from torch import Tensor as _T
from abc import ABC

__all__ = [
    'JExcitonModel'
]

class NonLinearRegressionModelBase(ABC):
    """
    $f$ - very variable!
    """

    def __init__(self, parameter_dim: int, input_dim: int, output_dim: int) -> None:
        self.parameter_dim = parameter_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def evaluate_f(self, x: _T, theta: _T) -> _T:
        raise NotImplementedError

    @staticmethod
    def validate_parameters(evaluation_function):
        """
        Ensure that each datapoint will be evaluated for each theta evaluation...
        """
        def outer(self: NonLinearRegressionModelBase, x: _T, theta: _T) -> _T:
            B, Dx = x.shape
            I, Dt = theta.shape
            assert Dt == self.parameter_dim and Dx == self.input_dim
            result = evaluation_function(self, x=x, theta=theta)
            assert list(result.shape) == [B, I, self.output_dim]
            return result
        return outer


class JExcitonModel(NonLinearRegressionModelBase):
    "Parameters are logs of equation parameters!"
    def __init__(self, J: int) -> None:
        self.J = J
        super().__init__(parameter_dim = 2*J, input_dim = 1, output_dim = 1)

    @NonLinearRegressionModelBase.validate_parameters
    def evaluate_f(self, x: _T, theta: _T) -> _T:
        amplitudes = theta[...,:self.J].unsqueeze(0).unsqueeze(-1).exp()
        time_constants = theta[...,self.J:].unsqueeze(0).unsqueeze(-1).exp()
        data = x.unsqueeze(1).unsqueeze(-1).repeat(1, 1, self.J, 1)
        comps = 1e-10 + amplitudes * (-data / time_constants).exp() # [num data, num theta, J, 1]
        return comps.sum(2)
