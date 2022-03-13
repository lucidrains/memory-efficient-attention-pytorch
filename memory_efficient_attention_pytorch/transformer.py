import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from memory_efficient_attention_pytorch import Attention
from memory_efficient_attention_pytorch.reversible import ReversibleSequence

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, chunks = 1):
        super().__init__()
        self.chunks = chunks

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        if self.chunks <= 1:
            return self.net(x)

        chunks = x.chunk(self.chunks, dim = 1)
        out = [self.net(chunk) for chunk in chunks]
        return torch.cat(out, dim = 1)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_chunks = 1,
        **kwargs
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, **kwargs)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, chunks = ff_chunks)),
            ]))

        self.net = ReversibleSequence(self.layers)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, labels = None):
        device = x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(x.shape[-2], device = device))
        x = x + pos_emb

        x = self.net(x)

        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        return F.cross_entropy(rearrange(logits, 'b n d -> b d n'), labels)
