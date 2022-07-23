## Memory Efficient Attention Pytorch

Implementation of a memory efficient multi-head attention as proposed in the paper, <a href="https://arxiv.org/abs/2112.05682">Self-attention Does Not Need O(n²) Memory</a>. In addition, the module will take care of masking, causal masking, as well as cross attention.

This repository also contains a <a href="https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py">naive non-CUDA implementation</a> of the improvements made by <a href="https://tridao.me/">Tri Dao</a> with his <a href="https://github.com/HazyResearch/flash-attention">Flash Attention</a> paper, for educational purposes. It is a game changer for attention and building long-context transformers.

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

x = torch.randn(1, 65536, 512).cuda()
out = attn(x) # (1, 65536, 512)
```

Cross attention

```python
import torch
from memory_efficient_attention_pytorch import Attention

cross_attn = Attention(
    dim = 512,
    dim_head = 64,
    heads = 8,
    memory_efficient = True,
    q_bucket_size = 1024,
    k_bucket_size = 2048
).cuda()

x = torch.randn(1, 65536, 512).cuda()
context = torch.randn(1, 65536, 512).cuda()
mask = torch.ones(1, 65536).bool().cuda()

out = cross_attn(x, context = context, mask = mask) # (1, 65536, 512)
```

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

```bibtex
@misc{liu2021swin,
    title   = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    author  = {Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
    year    = {2021},
    eprint  = {2111.09883},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Dao2022FlashAttentionFA,
    title   = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
    author  = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R'e},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2205.14135}
}
```
