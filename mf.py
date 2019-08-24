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
