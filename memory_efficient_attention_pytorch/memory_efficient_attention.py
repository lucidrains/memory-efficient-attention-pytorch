import torch
from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

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
    causal = False,
    **kwargs
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j).triu(j - i + 1).bool()
        sim = sim.masked_fill(mask, mask_value)

    attn = sim.softmax(dim = -1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

# memory efficient attention

def safe_sum(acc, el):
    if not exists(acc):
        return el
    return acc + el

def summarize_qkv_chunk(q, k, v, mask, causal_mask):
    weight = einsum('b h i d, b h j d -> b h i j', q, k)
    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if exists(causal_mask):
        weight = weight.masked_fill(causal_mask, mask_value)

    exp_weight = weight.exp()
    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)
    return exp_weight.sum(dim = -1), weighted_value

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if causal:
        i, j = q.shape[-2], k.shape[-2]
        causal_mask = torch.ones(i, j).triu(j - i + 1).bool()

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = None
        weighted_values = None

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):

            causal_mask_chunk = None
            if causal:
                causal_mask_chunk = causal_mask[
                    (q_index * q_bucket_size):(q_index * q_bucket_size + q_bucket_size),
                    (k_index * k_bucket_size):(k_index * k_bucket_size + k_bucket_size),
                ]

            exp_weight_chunk, weighted_value_chunk = checkpointed_summarize_qkv_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                causal_mask_chunk
            )

            exp_weights = safe_sum(exp_weights, exp_weight_chunk)
            weighted_values = safe_sum(weighted_values, weighted_value_chunk)

        normalized_values = weighted_values / (rearrange(exp_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)

# main class

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False,
        memory_efficient = False,
        q_bucket_size = 512,
        k_bucket_size = 1024
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # memory efficient attention related parameters
        # can be overriden on forward
        self.memory_efficient = memory_efficient
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        x,
        context = None,
        mask = None,
        memory_efficient = None,
        q_bucket_size = None,
        k_bucket_size = None
    ):
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        attn_fn = attention if not memory_efficient else memory_efficient_attention

        out = attn_fn(q, k, v, mask = mask, causal = self.causal, q_bucket_size = q_bucket_size, k_bucket_size = k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
