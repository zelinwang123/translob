import torch
from torch import Tensor
from torch.nn import TransformerEncoderLayer


class CausalTransformerEncoderLayer(TransformerEncoderLayer):
    """Transformer encoder layer with causal mask.

    See :class:`torch.nn.TransformerEncoderLayer` for details.

    Examples:

        >>> L, N, E = 5, 1, 2  # sequence length, batch, features
        >>> m = CausalTransformerEncoderLayer(E, 1)
        >>> src = torch.empty(L, N, E)
        >>> m.causal_mask(src)
        tensor([[False,  True,  True,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False, False, False,  True],
                [False, False, False, False, False]])
        >>> assert m(src).size() == src.size()
    """

    def causal_mask(self, src: Tensor) -> Tensor:
        # In PyTorch documentation of MultiHeadAttention:
        # > (L, S) where L is the target sequence length,
        # > S is the source sequence length.
        query, key, value = src, src, src
        trues = torch.ones(
            (query.size(0), key.size(0)), dtype=torch.bool, device=src.device
        )
        return trues.triu(diagonal=1)

    def forward(self, src: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(src, src_mask=self.causal_mask(src))
