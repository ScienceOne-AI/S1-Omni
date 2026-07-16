from __future__ import annotations

import torch
import torch.nn as nn


MIN_ATOMS = 3
MAX_ATOMS = 29
NUM_ATOM_CLASSES = MAX_ATOMS - MIN_ATOMS + 1


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: list[int] | tuple[int, ...] = (512, 256), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        last = in_dim
        for width in hidden:
            layers.extend([nn.Linear(last, int(width)), nn.GELU(), nn.Dropout(dropout)])
            last = int(width)
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
