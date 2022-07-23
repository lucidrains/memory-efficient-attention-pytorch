import math
import torch
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

# regular attention

def attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    **kwargs
):
    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias):
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

    attn = sim.softmax(dim = -1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

# memory efficient attention

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    exp_weight = weight.exp()
    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def numerically_unstable_memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8
):
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        q_start_index = q_index * q_bucket_size
        exp_weights = []
        weighted_values = []        

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            exp_weight_chunk, weighted_value_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index)
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)

        all_values = sum(weighted_values)
        all_weights = sum(exp_weights)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)

# main class

class CosineSimAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
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

        inner_dim = heads * dim_head

        scale_init_value = -math.log(math.log2(seq_len ** 2 - seq_len))
        self.scale = nn.Parameter(torch.full((1, heads, 1, 1), scale_init_value))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

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
        attn_bias = None,
        memory_efficient = None,
        q_bucket_size = None,
        k_bucket_size = None,
    ):
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        q = q * self.scale.exp()

        attn_fn = attention if not memory_efficient else numerically_unstable_memory_efficient_attention

        out = attn_fn(q, k, v, mask = mask, attn_bias = attn_bias, causal = self.causal, q_bucket_size = q_bucket_size, k_bucket_size = k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
