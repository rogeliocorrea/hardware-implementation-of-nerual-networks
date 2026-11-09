"""Shared preprocessing and small MNIST model for the fixed-weight study."""

import torch
from torch import nn


def pooled_pixels(images: torch.Tensor) -> torch.Tensor:
    """Return 14x14 uint8 pixels from N x 1 x 28 x 28 uint8 images."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError("expected N x 1 x 28 x 28 images")
    if images.dtype != torch.uint8:
        raise TypeError("expected uint8 MNIST pixels")
    blocks = images.reshape(-1, 1, 14, 2, 14, 2).to(torch.int32)
    return ((blocks.sum(dim=(3, 5)) + 2) // 4).reshape(-1, 196).to(torch.uint8)


class DigitMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(196, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(pixels.float() / 255.0).clamp(0.0, 1.0)
        return self.fc2(hidden)
