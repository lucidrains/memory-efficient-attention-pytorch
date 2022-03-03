import torch
from torch import nn, einsum
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# regular attention

def attention(
    q, k, v,
    mask = None,
    causal = False
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j).triu(j - i + 1).bool()
        sim = sim.masked_fill(mask, mask_value)

    attn = sim.softmax(dim = -1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

# main class

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = attention(q, k, v, mask = mask, causal = self.causal)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
