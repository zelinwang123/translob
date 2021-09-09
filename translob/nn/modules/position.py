import math

import torch
from torch import Tensor
from torch.nn import Module


class PositionalEncoding(Module):
    """Positional encoder.

    Args:
        d_model (int, default=1): Dimension of the model.
        max_length (int, default=100): Maximum length of the sequence
        encoding ({"linear"}): Method of encoding.
            For "linear":

                y = x / max_length

                x : position in the sequence
                y : positional encoder

    Shape:
        - Input: :math:`(N, *, X, L)` where :math:`N` is the batch size,
          :math:`X` is the number of features in the input,
          :math:`F` is the number of features in the positional encoding,
          :math:`L` is the length of the sequence,
          :math:`*` is any number of additional dimensions.
        - Output: :math:`(N, *, X + F, L)`.

    Examples:

        >>> _ = torch.manual_seed(42)
        >>> x = torch.randn(1, 2, 10)
        >>> m = PositionalEncoding()
        >>> m(x).size()
        torch.Size([1, 3, 10])
    """

    positional_encoder: Tensor

    def __init__(
        self, d_model: int = 1, max_length: int = 100, encoding: str = "linear"
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.encoding = encoding

        self.register_buffer("positional_encoder", self._compute_positional_encoder())

    def _compute_positional_encoder(self) -> Tensor:
        # Returns:
        # positional_encoder : tensor, shape (F, L)
        #     F : number of features
        #     L : maximum length
        if self.encoding == "sinusoid":
            position = torch.linspace(0.0, 2 * math.pi, self.max_length).reshape(-1, 1)
            frequency = torch.logspace(0.0, math.log(2 * math.pi), self.d_model, math.e)
            frequency = frequency.unsqueeze(0)

            phase = frequency * position

            positional_encoder = torch.empty((self.max_length, 2 * self.d_model))
            positional_encoder[:, 0::2] = phase.sin()
            positional_encoder[:, 1::2] = phase.cos()

        if self.encoding == "linear":
            positional_encoder = torch.linspace(0.0, 1.0, self.max_length).unsqueeze(0)
        else:
            raise ValueError("invalid 'encoding'")

        return positional_encoder

    def forward(self, input: Tensor) -> Tensor:
        # cut and align shape
        p = self.positional_encoder[..., : input.size(-1)]
        # for input shape (N, *, X, L), p's shape is (N, *, F, L)
        p = p.expand(input.size()[:-2] + p.size()[-2:])
        return torch.cat((input, p), -2)
