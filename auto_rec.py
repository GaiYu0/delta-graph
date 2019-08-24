import torch as th
import torch.nn as nn
from torch_sparse import spmm

from mf import *

class AutoRec(nn.Module):
    def __init__(self, n, d, g, f):
        super().__init__()
        self.v = nn.Parameter(1e-3 * th.randn(n, d))
        self.mu = nn.Parameter(th.zeros(1, d))
        self.g = g
        self.w = nn.Parameter(1e-3 * th.randn(n, d))
        self.b = nn.Parameter(th.zeros(n))
        self.f = f

    def forward(self, u, i, r, m, v, j, s=None):
        """
        Parameters
        ----------
        r : (m, n)
        """
        h = self.g(spmm([i, u], r, m, self.v) + self.mu)
        return MF.decode(h, w, j, v, s) + self.b[v]

class IAutoRec(AutoRec):
    def __init__(self, n_users, n_items, d, g, f):
        super().__init__(n_users, d, g, f)
        self.n_users = n_users
        self.n_items = n_items

    def forward(self, u, i, r, v, j, s=None):
        return super().forward(i, u, r, self.n_items, j, v, s)
