"""Neural-network models used by the spectral-family experiments."""

from __future__ import annotations

import torch
from torch import nn


class ComplexLinearMapNet(nn.Module):
    """Map a complex spectral parameter to a general complex matrix.

    The input is ``(Re lambda, Im lambda)``.  No algebraic structure is
    imposed on the matrix output.  This makes sparsity, diagonality, equal
    singular values, or other structures empirical results of training.
    """

    def __init__(
        self,
        output_rows: int,
        output_cols: int,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if output_rows < 1 or output_cols < 1:
            raise ValueError("matrix dimensions must be positive")

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        n = output_rows * output_cols
        layers.append(nn.Linear(hidden_dim, 2 * n))
        self.net = nn.Sequential(*layers)
        self.output_rows = output_rows
        self.output_cols = output_cols

    def forward(self, lam_xy: torch.Tensor) -> torch.Tensor:
        raw = self.net(lam_xy)
        n = self.output_rows * self.output_cols
        shape = (*raw.shape[:-1], self.output_rows, self.output_cols)
        real = raw[..., :n].reshape(shape)
        imag = raw[..., n:].reshape(shape)
        return torch.complex(real, imag)


class ComplexMatrixNet(ComplexLinearMapNet):
    """Map a complex spectral parameter to a general complex square matrix."""

    def __init__(self, matrix_dim: int = 3, hidden_dim: int = 128, hidden_layers: int = 3):
        super().__init__(
            output_rows=matrix_dim,
            output_cols=matrix_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        self.matrix_dim = matrix_dim
