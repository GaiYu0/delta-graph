from collections import deque

import torch as th
import torch.nn as nn

from mf import *

class CollapsedMF(nn.Module):
    def __init__(self, mf):
        super().__init__()
        self.mf = mf

    def forward(self, uu, ii, rr, v, j, mm, s=None):
        return self.mf(None, None, None, v, j, s=None), None

class TemporalBiasedMF(nn.Module):
    def __init__(self, n_users, n_items, d, mus, T):
        super().__init__()
        self.hh_u = nn.ParameterList([nn.Parameter(1e-3 * th.randn(n_users, d)) for _ in mus])
        self.hh_i = nn.ParameterList([nn.Parameter(1e-3 * th.randn(n_items, d)) for _ in mus])
        self.bb_i = nn.ParameterList([nn.Parameter(th.zeros(n_users)) for _ in mus])
        self.bb_u = nn.ParameterList([nn.Parameter(th.zeros(n_users)) for _ in mus])
        self.merge_u = nn.LSTM(d, d)
        self.merge_i = nn.LSTM(d, d)

        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.mus = mus
        self.T = T

    def forward(self, uu, ii, rr, vv, jj, m, s=None, detach=False):
        m_u, m_i = None, None if m is None else m
        if detach:
            if len(rr) < self.T:
                return [self.merge_u(self.hh_u[0].unsqueeze(0), m_u)[1].detach(),
                        self.merge_i(self.hh_i[0].unsqueeze(0), m_u)[1].detach()]
            else:
                return None
        tt = []
        tail = lambda x: deque(x, maxlen=self.T)
        for v, j, h_u, h_i, b_u, b_i, mu in tail(zip(vv, jj,
                                                     self.hh_u, self.hh_i,
                                                     self.bb_u, self.bb_i, self.mus)):
            g_u, m_u = self.merge_u(h_u.unsqueeze(0), m_u)
            g_i, m_i = self.merge_i(h_i.unsqueeze(0), m_i)
            tt.append(BiasedMF.decode(g_u.squeeze(), g_i.squeeze(), b_u, b_i, mu, v, j, s))
        return tt
