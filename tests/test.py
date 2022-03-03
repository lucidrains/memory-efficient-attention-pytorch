import torch
from memory_efficient_attention_pytorch import Attention

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
        k_bucket_size = 64
    )

    x = torch.randn(2, 2048, 512)
    mask = torch.ones(2, 2048).bool()

    out = attn(x, mask = mask)
    mem_efficient_out = attn(x, mask = mask, memory_efficient = True)

    assert isclose(mem_efficient_out, out)

# test gradients equal

def test_gradients_equal():
    attn = Attention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64
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

    assert isclose(out_grad, mem_efficient_out_grad)
