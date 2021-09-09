from copy import deepcopy
from typing import List

from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential


class MultiLayerPerceptron(Sequential):
    """Multi-layer perceptron.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        n_layers (int): number of hidden layers.
        n_units (int): number of units in each hidden layer.
        activation (Module): activation module in hidden layers.

    Shape:
        - Input: :math:`(N, *, H_in)` where
          :math:`*` means any number of additional dimensions and
          :math`H_in` is ``in_features``.
        - Output: :math:`(N, *, H_out)` where
          all but the last dimension are the same shape as the input and
          :math:`H_out` is ``out_features``.

    Examples:
        >>> import torch
        >>>
        >>> m = MultiLayerPerceptron(2, 3)
        >>> m
        MultiLayerPerceptron(
          (0): Linear(in_features=2, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=3, bias=True)
        )
        >>> m(torch.empty(1, 2)).size()
        torch.Size([1, 3])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = 2,
        n_units: int = 32,
        activation: Module = ReLU(),
    ) -> None:
        layers: List[Module] = []
        for i_layer in range(n_layers):
            layers.append(Linear(in_features if i_layer == 0 else n_units, n_units))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units, out_features))

        super().__init__(*layers)
