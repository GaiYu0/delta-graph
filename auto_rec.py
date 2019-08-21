import torch as th
import torch.nn as nn
from torch_sparse import spmm

class AutoRec(nn.Module):
    def __init__(self, n, d, g, f):
        super().__init__()
        self.v = nn.Parameter(th.randn(n, d))
        self.mu = nn.Parameter(th.zeros(1, d))
        self.g = g
        self.w = nn.Parameter(th.randn(n, d))
        self.b = nn.Parameter(th.zeros(n))
        self.f = f

    def forward(self, idx, dat, m, i, j):
        """
        Parameters
        ----------
        r : (m, n)
        """
        h = self.g(spmm(idx, dat, m, self.v) + mu)
        return self.f(th.sum(h[i] * self.w[j]) + self.b[j])

class IAutoRec(AutoRec):
    def __init__(self, n_users, n_items, d, g, f):
        super().__init__(n_users, d, g, f)
        self.n_users = n_users
        self.n_items = n_items

    def forward(self, uid_in, iid_in, y, uid_out, iid_out):
        return super().forward([iid_in, uid_in], y, self.n_items, iid_out, uid_out)
