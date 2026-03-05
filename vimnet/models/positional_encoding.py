from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_time_embedding(T: int, dim: int, device: torch.device) -> torch.Tensor:
    """Standard Transformer sinusoidal embedding for positions 0..T-1.

    Returns:
        (T, dim) float tensor
    """
    pe = torch.zeros(T, dim, device=device)
    position = torch.arange(0, T, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
