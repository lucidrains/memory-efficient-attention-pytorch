## Memory Efficient Attention Pytorch (wip)

Implementation of a memory efficient multi-head attention as proposed in the paper, <a href="https://arxiv.org/abs/2112.05682">Self-attention Does Not Need O(n²) Memory</a>. In addition, the module will take care of masking, causal masking, as well as cross attention.

## Install

```bash
$ pip install memory-efficient-attention-pytorch
```

## Usage

For autoregressive language model

```python
import torch
from memory_efficient_attention_pytorch import Attention

attn = Attention(
    dim = 512,
    dim_head = 64,                # dimension per head
    heads = 8,                    # number of attention heads
    causal = True,                # autoregressive or not
    memory_efficient = True,      # whether to use memory efficient attention (can be turned off to test against normal attention)
    q_bucket_size = 1024,         # bucket size along queries dimension
    k_bucket_size = 2048          # bucket size along key / values dimension
).cuda()

x = torch.randn(1, 16384, 512).cuda()
out = attn(x) # (1, 16384, 512)
```

Cross attention

```python
import torch
from memory_efficient_attention_pytorch import Attention

attn = Attention(
    dim = 512,
    dim_head = 64,
    heads = 8,
    memory_efficient = True,
    q_bucket_size = 1024,
    k_bucket_size = 2048
).cuda()

x = torch.randn(1, 16384, 512).cuda()
context = torch.randn(1, 16384, 512).cuda()
mask = torch.ones(1, 16384).bool().cuda()

out = attn(x, context = context, mask = mask) # (1, 16384, 512)
```

- [ ] add enwik8 example with 8192 context length

## Citations

```bibtex
@misc{rabe2021selfattention,
    title   = {Self-attention Does Not Need $O(n^2)$ Memory}, 
    author  = {Markus N. Rabe and Charles Staats},
    year    = {2021},
    eprint  = {2112.05682},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
