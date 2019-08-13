import torch as th
from torch_scatter import scatter_add, scatter_max

def neighborhood_softmax(adj, logits):
    """
    Parameters
    ----------
    """
    repeats = adj.indptr[1:] - adj.indptr[:-1]
    idx = th.repeat_interleave(th.arange(len(repeats), device=repeats.device), repeats)
    exp = th.exp(logits - th.repeat_interleave(scatter_max(logits, idx)[0], repeats, 0))
    return exp / th.repeat_interleave(scatter_add(exp, idx), repeats)
