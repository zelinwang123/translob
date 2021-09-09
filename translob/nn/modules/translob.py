from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn import Dropout
from torch.nn import Flatten
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Softmax
from torch.nn import TransformerEncoder

from .conv import CausalConvLayers
from .mlp import MultiLayerPerceptron
from .position import PositionalEncoding
from .transformer import CausalTransformerEncoderLayer


class TransLOB(Module):
    r"""Transformers for limit order books.

    Default values are the same with the original paper, unless stated otherwise.

    Reference:
        - Transformers for limit order books, James Wallbridge (2020)
          https://github.com/jwallbridge/translob

    Args:
        in_features (int, default=40): The number of input features.
        out_features (int, default=3): The number of output features.
        out_activation (torch.nn.Module, default=torch.nn.Softmax(-1)):
            The activation layer applied to the output.
        conv_n_layers (int, default=5): The number of convolutional layers.
        conv_n_features (int, default=14): The number of features
            in the convolutional layers.
        conv_kernel_size (int, default=2): The kernel size
            in the convolutional layers.
        conv_dilation (int or tuple[int], default=(1, 2, 4, 8, 16)): The dilation(s)
            in the convolutional layers.
        tf_n_channels: (int, default=3): The number of channels
            in the multi-head self-attension of Transformer encoder.
            Its default value may be different from the original implementation.
            Its default value (denoted "C" in the paper?) does not seem to be
            clarified in the original papar and so we set the default value arbitrarily.
        tf_dim_feedforward (int, default=60): The dimension of feed-forward
            network model in Transformer encoder.
        tf_dropout_rate (float, default=0.0): Dropout rate in Transformer encoder.
        tf_num_layers (int, default=2): Number of sub-encoder-layers in the Transformer encoder.
        mlp_dim (int, default=64):
            Dimension of feedforward network model after Transformer encoder.
        mlp_n_layers (int, default=1):
            Number of layers in feedforward network model after Transformer encoder.
        dropout_rate (float, default=0.1):
            Dropout rate after Transformer encoder.

    Shapes:
        - Input: :math:`(N, C, L)` where :math:`N` is the batch size,
          :math:`C` is the number of features and
          :math:`L` is the length of the sequence.
          :math:`C = 40` in the original paper: ask/bid, level 1-10, and price/volume.
          :math:`L = 100` in the original paper.
        - Output: :math:`(N, N_{\text{out}})`
          :math:`N_{\text{out}} = 3` in the original paper (up, down, and neutral).

    Examples:
        >>> import torch
        >>>
        >>> m = TransLOB()
        >>> input = torch.empty(1, 40, 100)
        >>> m(input).size()
        torch.Size([1, 3])
    """

    def __init__(
        self,
        in_features: int = 40,
        out_features: int = 3,
        len_sequence: int = 100,
        out_activation: Module = Softmax(-1),
        conv_n_features: int = 14,
        conv_kernel_size: int = 2,
        conv_dilation: Union[Tuple[int, ...], int] = (1, 2, 4, 8, 16),
        conv_n_layers: int = 5,
        tf_n_channels: int = 3,
        tf_dim_feedforward: int = 60,
        tf_dropout_rate: float = 0.0,
        tf_num_layers: int = 2,
        mlp_dim: int = 64,
        mlp_n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Define convolutional module.
        convolution = CausalConvLayers(
            in_features,
            conv_n_features,
            conv_kernel_size,
            dilation=conv_dilation,
            n_layers=conv_n_layers,
        )
        self.pre_transformer = Sequential(
            convolution,
            LayerNorm(torch.Size((conv_n_features, len_sequence))),
            PositionalEncoding(max_length=len_sequence),
        )

        # Define Transformer encoder module.
        d_model = conv_n_features + 1
        encoder_layer = CausalTransformerEncoderLayer(
            d_model=d_model,
            nhead=tf_n_channels,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout_rate,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=tf_num_layers)

        # Define modules used after Transformer encoder.
        multi_layer_perceptron = MultiLayerPerceptron(
            in_features=d_model * len_sequence,
            out_features=mlp_dim,
            n_layers=mlp_n_layers,
            n_units=mlp_dim,
        )
        self.post_transformer = Sequential(
            Flatten(1, -1),
            multi_layer_perceptron,
            Dropout(dropout_rate),
            Linear(mlp_dim, out_features),
            out_activation,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.pre_transformer(input).movedim(-1, 0)
        input = self.transformer(input)
        input = self.post_transformer(input.movedim(0, -1))
        return input
