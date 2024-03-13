import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributions as tdist
from .radial import Radial
from typing import Union, Tuple, List, Optional
import math


class Evidence(nn.Module):
    """layer to transform density values into evidence representations according to a predefined scale"""

    def __init__(
        self, scale: str = "latent-new-plus-classes", tau: Optional[float] = None
    ):
        super().__init__()
        self.tau = tau

        assert scale in ("latent-old", "latent-new", "latent-new-plus-classes", None)
        self.scale = scale

    def forward(self, log_q_c: Tensor, dim: int, **kwargs) -> Tensor:
        scaled_log_q = log_q_c + self.log_scale(dim, **kwargs)

        if self.tau is not None:
            scaled_log_q = self.tau * (scaled_log_q / self.tau).tanh()

        scaled_log_q = scaled_log_q.clamp(min=-30.0, max=30.0)

        return scaled_log_q

    def log_scale(self, dim: int, further_scale: int = 1) -> float:
        scale = 0

        if "latent-old" in self.scale:
            scale = 0.5 * (dim * math.log(2 * math.pi) + math.log(dim + 1))
        if "latent-new" in self.scale:
            scale = 0.5 * dim * math.log(4 * math.pi)

        scale = scale + math.log(further_scale)

        return scale


class Density(nn.Module):
    """
    encapsulates the PostNet step of transforming latent space
    embeddings z into alpha-scores with normalizing flows
    """

    def __init__(
        self,
        dim_latent: int,
        num_mixture_elements: int,
        radial_layers: int = 6,
    ):
        super().__init__()
        self.flow = BatchedNormalizingFlowDensity(
            c=num_mixture_elements,
            dim=dim_latent,
            flow_length=radial_layers,
            flow_type="radial_flow",
        )

    def forward(self, z: Tensor) -> Tensor:
        log_q_c = self.flow.log_prob(z).transpose(0, 1)
        if not self.training:
            # If we're evaluating and observe a NaN value, this is always caused by the
            # normalizing flow "diverging". We force these values to minus infinity.
            log_q_c[torch.isnan(log_q_c)] = float("-inf")

        return log_q_c


class BatchedNormalizingFlowDensity(nn.Module):
    """layer of normalizing flows density which calculates c densities in a batched fashion"""

    def __init__(self, c, dim, flow_length, flow_type="radial_flow"):
        super(BatchedNormalizingFlowDensity, self).__init__()

        self.c = c
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type
        self.dist = None

        self.mean = nn.Parameter(torch.zeros(self.c, self.dim), requires_grad=False)
        self.cov = nn.Parameter(
            torch.eye(self.dim).repeat(self.c, 1, 1), requires_grad=False
        )

        self.transforms = nn.Sequential(*(Radial(c, dim) for _ in range(flow_length)))

    def forward(self, z: Tensor) -> Tensor:
        sum_log_jacobians = 0
        z = z.repeat(self.c, 1, 1)
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(
                z, z_next
            )
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        if self.dist is None:
            self.dist = tdist.MultivariateNormal(
                self.mean.repeat(z.size(1), 1, 1).permute(1, 0, 2),
                self.cov.repeat(z.size(1), 1, 1, 1).permute(1, 0, 2, 3),
            )

        log_prob_z = self.dist.log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x
