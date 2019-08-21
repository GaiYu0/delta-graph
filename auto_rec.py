import torch as th
import torch.nn as nn
import torch.sparse as sparse

class AutoRec(nn.Module):
    def __init__(self, n, d, g, f):
        super().__init__()
        self.v = nn.Parameter(th.randn(n, d))
        self.mu = nn.Parameter(th.zeros(1, d))
        self.g = g
        self.w = nn.Parameter(th.randn(n, d))
        self.b = nn.Parameter(th.zeros(n))
        self.f = f

    def forward(self, r, i, j):
        """
        Parameters
        ----------
        r : (m, n)
        """
        h = self.g(sparse.mm(r, v) + mu)
        return self.f(th.sum(h[i] * self.w[j]) + b[j])

class IAutoRec(AutoRec):
    def __init__(self, n_users, n_items, d, g, f):
        super().__init__(n_users, d, g, f)

    def forward(self, r, u, i):
        return super().forward(r, i, u)

class UAutoRec(AutoRec):
    def __init__(self, n_users, n_items, d, g, f):
        super().__init__(n_items, d, g, f)

    def forward(self, r, u, i):
        return super().forward(r, u, i)
