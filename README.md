## Memory Efficient Attention Pytorch (wip)

Implementation of a memory efficient multi-head attention as proposed in the paper, <a href="https://arxiv.org/abs/2112.05682">Self-attention Does Not Need O(n²) Memory</a>. In addition, the module will take care of masking, causal masking, as well as cross attention.

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
