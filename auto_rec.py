import torch as th
import torch.nn as nn
from torch_sparse import spmm

class AutoRec(nn.Module):
    def __init__(self, n, d, g, f):
        super().__init__()
        self.v = nn.Parameter(1e-3 * th.randn(n, d))
        self.mu = nn.Parameter(th.zeros(1, d))
        self.g = g
        self.w = nn.Parameter(1e-3 * th.randn(n, d))
        self.b = nn.Parameter(th.zeros(n))
        self.f = f

    def forward(self, ij, r, m, i, j, s=None):
        """
        Parameters
        ----------
        r : (m, n)
        """
        h = self.g(spmm(ij, r, m, self.v) + self.mu)
        ii = [i] if s is None else th.split(i, s)
        jj = [j] if s is None else th.split(j, s)
        return th.cat([self.f(th.sum(h[i] * self.w[j], 1) + self.b[j]) for i, j in zip(ii, jj)])

class IAutoRec(AutoRec):
    def __init__(self, n_users, n_items, d, g, f):
        super().__init__(n_users, d, g, f)
        self.n_users = n_users
        self.n_items = n_items

    def forward(self, uid_in, iid_in, r_in, uid_out=None, iid_out=None, s=None):
        uid_out = uid_in if uid_out is None else uid_out
        iid_out = iid_in if iid_out is None else iid_out
        return super().forward([iid_in, uid_in], r_in, self.n_items, iid_out, uid_out, s)
