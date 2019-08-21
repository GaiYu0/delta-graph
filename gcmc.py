import torch as th
import torch.nn as nn
from torch.sparse import mm

class GCMC(nn.Module):
    def __init__(self, n_items, n_users, in_feats, out_feats, n_chnls, reduce_chnl, nonlinear):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.in_feats = in_feats
        self.reduce_chnl = reduce_chnl
        self.nonlinear = nonlinear

        self.embedding_u = nn.ModuleList([nn.Embedding(n_users, in_feats) \
                                              for i in range(n_chnls)])
        self.i_embeddings = nn.ModuleList([nn.Embedding(n_items, in_feats) \
                                              for i in range(n_chnls)])
        multiplier = {'cat' : n_chnls, 'sum' : 1}[reduce_chnl]
        self.linear = nn.Linear(multiplier * in_feats, out_feats)

    def forward(self, uis, ius, u, i, h_prev=None):
        device = next(self.parameters()).device
        uu = [None if ui is None else mm(ui.coo, embedding(ui.uniq_j)) \
              for ui, embedding in zip(uis, self.i_embeddings)]
        ii = [None if iu is None else mm(iu.coo, embedding(iu.uniq_j)) \
              for iu, embedding in zip(ius, self.embedding_u)]
        u_zeros = th.zeros(self.n_users, self.in_feats, device=device)
        u_scatter = lambda ui, u: th.index_copy(u_zeros, 0, ui.uniq_i, u)
        i_zeros = th.zeros(self.n_items, self.in_feats, device=device)
        i_scatter = lambda iu, i: th.index_copy(i_zeros, 0, iu.uniq_i, i)
        if self.reduce_chnl == 'cat':
            u = th.cat([u_zeros if u is None else u_scatter(ui, u) for ui, u in zip(uis, uu)], 1)
            i = th.cat([i_zeros if i is None else i_scatter(iu, i) for iu, i in zip(ius, ii)], 1)
        elif self.reduce_chnl == 'sum':
            u = sum(u_scatter(ui, u) for ui, u in zip(uis, uu) if ui is not None)
            i = sum(i_scatter(iu, i) for iu, i in zip(ius, ii) if iu is not None)
        u = self.linear(self.nonlinear(u))
        i = self.linear(self.nonlinear(i))
        h_next = []
        return u, i, h_next

class RatingPredictor(nn.Module):
    def __init__(self, in_feats, n_bases, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.coeffs = nn.Parameter(1e-3 * th.randn(n_bases, n_classes, 1, 1))
        self.bases = nn.Parameter(1e-3 * th.randn(n_bases, n_classes, in_feats, in_feats))

    def forward(self, u, i):
        u = u.unsqueeze(0).repeat(self.n_classes, 1, 1)
        i = i.unsqueeze(0).repeat(self.n_classes, 1, 1)
        kernels = th.sum(self.coeffs * self.bases, 0)
        return th.sum(u * th.bmm(i, kernels), 2).t()
