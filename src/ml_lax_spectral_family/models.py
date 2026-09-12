"""Neural-network models used by the spectral-family experiments."""

from __future__ import annotations

import torch
from torch import nn


class ComplexMatrixNet(nn.Module):
    """Map complex lambda to a general complex square matrix.

    The input is (Re lambda, Im lambda). No matrix structure is imposed on
    the output, so recovery of a scalar multiple of the identity is an
    empirical result rather than an architectural prior.
    """

    def __init__(self, matrix_dim: int = 3, hidden_dim: int = 128, hidden_layers: int = 3):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2 * matrix_dim * matrix_dim))
        self.net = nn.Sequential(*layers)
        self.matrix_dim = matrix_dim

    def forward(self, lam_xy: torch.Tensor) -> torch.Tensor:
        raw = self.net(lam_xy)
        n = self.matrix_dim * self.matrix_dim
        real = raw[..., :n].reshape(*raw.shape[:-1], self.matrix_dim, self.matrix_dim)
        imag = raw[..., n:].reshape(*raw.shape[:-1], self.matrix_dim, self.matrix_dim)
        return torch.complex(real, imag)
