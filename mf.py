import torch as th
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, n_users, n_items, d):
        super().__init__()
        self.h_u = nn.Parameter(1e-3 * th.randn(n_users, d))
        self.h_i = nn.Parameter(1e-3 * th.randn(n_items, d))

    @staticmethod
    def decode(h_u, h_i, u, i, s=None):
        uu = [u] if s is None else th.split(u, s)
        ii = [i] if s is None else th.split(i, s)
        return th.cat([th.sum(h_u[u] * h_i[i], 1) for u, i in zip(uu, ii)])

    def forward(self, u, i, r, v, j, s=None):
        return MF.decode(self.h_u, self.h_i, v, j, s)

class BiasedMF(MF):
    def __init__(self, n_users, n_items, d, mu):
        super().__init__(n_users, n_items, d)
        self.b_u = nn.Parameter(th.zeros(n_users))
        self.b_i = nn.Parameter(th.zeros(n_items))
        self.mu = mu

    @staticmethod
    def decode(h_u, h_i, b_u, b_i, mu, u, i, s=None):
        return MF.decode(h_u, h_i, u, i, s) + b_u[u] + b_i[i] + mu

    def forward(self, u, i, r, v, j, s=None):
        return BiasedMF.decode(self.h_u, self.h_i, self.b_u, self.b_i, self.mu, v, j, s)
