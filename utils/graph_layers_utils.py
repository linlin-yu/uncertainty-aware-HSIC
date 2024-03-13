from typing import Optional
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn.utils import spectral_norm


class SpectralLinear(nn.Module):
    """linear layer with option to use it as a spectral linear layer with lipschitz-norm of k"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_lipschitz: Optional[float] = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        linear = nn.Linear(input_dim, output_dim, bias=bias)

        if self.k_lipschitz is not None:
            self.linear = spectral_norm(linear)
        else:
            self.linear = linear

    def forward(self, x: Tensor) -> Tensor:
        if self.k_lipschitz is None:
            y = self.linear(x)

        else:
            y = self.k_lipschitz * self.linear(x)

        return y

    def reset_parameters(self):
        self.linear.reset_parameters()


def LinearSequentialLayer(
    input_dims: int,
    hidden_dims: int,
    output_dim: int,
    dropout_prob: Optional[float] = None,
    batch_norm: bool = False,
    k_lipschitz: Optional[float] = None,
    num_layers: Optional[int] = None,
    activation_in_all_layers=False,
    **_
) -> nn.Module:
    """creates a chain of combined linear and activation layers depending on specifications"""

    if isinstance(hidden_dims, int):
        if num_layers is not None:
            hidden_dims = [hidden_dims] * (num_layers - 1)
        else:
            hidden_dims = [hidden_dims]

    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []

    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(
                dims[i], dims[i + 1], k_lipschitz=k_lipschitz ** (1.0 / num_layers)
            )
        else:
            l = nn.Linear(dims[i], dims[i + 1])

        layers.append(l)

        if activation_in_all_layers or (i < num_layers - 1):
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

            if dropout_prob is not None:
                layers.append(nn.Dropout(p=dropout_prob))

    return nn.Sequential(*layers)
