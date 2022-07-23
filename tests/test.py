import torch
from memory_efficient_attention_pytorch import Attention

from memory_efficient_attention_pytorch.memory_efficient_attention import attention
from memory_efficient_attention_pytorch.flash_attention import FlashAttention, FlashAttentionFunction

# constants

def isclose(a, b, atol = 1e-6):
    diff = (a - b).abs().amax()
    return diff < atol

# test outputs are equal

def test_output_equal():
    attn = Attention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    x = torch.randn(2, 2048, 512)
    mask = torch.ones(2, 2048).bool()

    out = attn(x, mask = mask)
    mem_efficient_out = attn(x, mask = mask, memory_efficient = True)

    assert isclose(mem_efficient_out, out, atol = 1e-6)

# test gradients equal

def test_gradients_equal():
    attn = Attention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    def loss_fn(inp, **kwargs):
        return attn(inp, **kwargs).sum()

    x = torch.randn(2, 2048, 512).requires_grad_()
    mask = torch.ones(2, 2048).bool()

    loss_fn(x, mask = mask).backward()
    out_grad = x.grad.clone()

    x.grad.zero_()
    loss_fn(x, mask = mask, memory_efficient = True).backward()
    mem_efficient_out_grad = x.grad.clone()

    assert isclose(out_grad, mem_efficient_out_grad, atol = 1e-5)

# test flash attention

def test_flash_attn_output_equal():
    attn_kwargs = dict(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    attn = Attention(**attn_kwargs)
    flash_attn = FlashAttention(**attn_kwargs)

    flash_attn.to_q = attn.to_q
    flash_attn.to_kv = attn.to_kv
    flash_attn.to_out = attn.to_out

    x = torch.randn(2, 2048, 512)
    mask = torch.ones(2, 2048).bool()

    out = attn(x, mask = mask)
    mem_efficient_out = flash_attn(x, mask = mask)

    assert isclose(mem_efficient_out, out, atol = 1e-6)

# test gradients equal

def test_flash_attn_gradients_equal():
    q = torch.randn(1, 8, 1024, 512).requires_grad_()
    k = torch.randn(1, 8, 1024, 512).requires_grad_()
    v = torch.randn(1, 8, 1024, 512).requires_grad_()

    mask = torch.ones(1, 1024).bool()

    o = attention(q, k, v, mask = mask, causal = True)
    o.sum().backward()

    dq_grad = q.grad.clone()
    dk_grad = k.grad.clone()
    dv_grad = v.grad.clone()

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    flash_o = FlashAttentionFunction.apply(q, k, v, mask, True, 64, 64)
    flash_o.sum().backward()

    flash_dq_grad = q.grad.clone()
    flash_dk_grad = k.grad.clone()
    flash_dv_grad = v.grad.clone()

    assert isclose(flash_dq_grad, dq_grad, atol = 1e-5)
    assert isclose(flash_dk_grad, dk_grad, atol = 1e-5)
    assert isclose(flash_dv_grad, dv_grad, atol = 1e-5)
