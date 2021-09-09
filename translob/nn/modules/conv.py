from copy import deepcopy
from typing import List
from typing import Tuple
from typing import Union

from torch import Tensor
from torch.nn import Conv1d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import ReplicationPad1d
from torch.nn import Sequential


class CausalConv1d(Sequential):
    r"""Applies a 1D convolution with causal padding.

    Args:
        in_channels (int): Number of channels in the input sequence.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        dilation (int, default=1): Spacing between kernel elements.

    Shape:
        - Input: :math:`(N, C_{\text{in}}, L)`
        - Output: :math:`(N, C_{\text{out}}, L)` where :math:`N` is the batch size,
          :math:`C_{\text{in}}` is the number of channels in the input sequence,
          :math:`C_{\text{out}}` is the number of channels in the output sequence,
          :math:`L` is the length of the sequence.

    Examples:

        >>> import torch
        >>>
        >>> m = CausalConv1d(40, 14, 2, dilation=1)
        >>> m
        CausalConv1d(40, 14, kernel_size=(2,), stride=(1,))
        >>>
        >>> input = torch.empty(1, 40, 100)
        >>> m(input).size()
        torch.Size([1, 14, 100])
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super().__init__()
        self.pad = ReplicationPad1d(((kernel_size - 1) * dilation, 0))
        self.conv = Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(self.pad(input))

    def __repr__(self) -> str:
        return self._get_name() + f"({self.conv.extra_repr()})"


class CausalConvLayers(Sequential):
    r"""Applies dilated causal convolution.

    Args:
        in_features (int): The number of channels in the input sequence.
        channels (int): The number of channels in the intermediate and output sequences.
        kernel_size (int): Size of the convolving kernel.
        dilation (int or tuple[int], default=1):
            If int, use the common value of dilation for each layer.
            If tuple[int], use different value for each layer.
        n_layers (int, default=5): The number of causal convolutional layer in the module.

    Shapes:
        - Input: :math:`(N, C_{\text{in}}, L)`
        - Output: :math:`(N, C_{\text{out}}, L)` where :math:`N` is the batch size,
          :math:`C_{\text{in}}` is the number of channels in the input sequence,
          :math:`C_{\text{out}}` is the number of channels in the output sequence,
          :math:`L` is the length of the sequence.

    Examples:

        >>> import torch
        >>>
        >>> _ = torch.manual_seed(42)
        >>> m = CausalConvLayers(40, 14, 2, dilation=(1, 2, 4, 8, 16))
        >>> m
        CausalConvLayers(
          (0): CausalConv1d(40, 14, kernel_size=(2,), stride=(1,))
          (1): ReLU()
          (2): CausalConv1d(14, 14, kernel_size=(2,), stride=(1,), dilation=(2,))
          (3): ReLU()
          (4): CausalConv1d(14, 14, kernel_size=(2,), stride=(1,), dilation=(4,))
          (5): ReLU()
          (6): CausalConv1d(14, 14, kernel_size=(2,), stride=(1,), dilation=(8,))
          (7): ReLU()
          (8): CausalConv1d(14, 14, kernel_size=(2,), stride=(1,), dilation=(16,))
        )
        >>> input = torch.empty((1, 40, 100))
        >>> m(input).size()
        torch.Size([1, 14, 100])
    """

    def __init__(
        self,
        in_channels: int,
        n_features: int,
        kernel_size: int,
        dilation: Union[Tuple[int, ...], int] = 1,
        n_layers: int = 5,
        activation: Module = ReLU(),
    ):
        if isinstance(dilation, int):
            dilation = (dilation,) * n_layers

        layers: List[Module] = []
        for i in range(n_layers):
            c = in_channels if i == 0 else n_features
            layers.append(CausalConv1d(c, n_features, kernel_size, dilation[i]))
            if i != n_layers - 1:
                layers.append(deepcopy(activation))

        super().__init__(*layers)
