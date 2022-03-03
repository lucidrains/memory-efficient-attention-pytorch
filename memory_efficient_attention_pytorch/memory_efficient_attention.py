import torch
from torch import nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()

    def forward(self, x):
        return x
