import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as ths

from adj import Adj, SubAdj
from neighborhood_softmax import neighborhood_softmax

starmap = lambda f, xs: map(lambda x: f(*x), xs)
starfilter = lambda f, xs: filter(lambda x: f(*x), xs)

class Parameter(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = nn.Parameter(x)

    def forward(self):
        return self.x

class SharedParameter(nn.Module):
    def __init__(self, xs):
        super().__init__()
        self.xs = nn.ParameterList(map(nn.Parameter, xs))

    def __getitem__(self, idx):
        return lambda: sum(self.xs[:idx + 1])

class Attention(nn.Module):
    def __init__(self, d=None, a_src=None, a_dst=None):
        super().__init__()
        self.a_src = a_src if d is None else Parameter(1e-1 * th.randn(d))
        self.a_dst = a_dst if d is None else Parameter(1e-1 * th.randn(d))

    def forward(self, h_src, h_dst):
        return F.leaky_relu(h_src @ self.a_src() + h_dst @ self.a_dst())

class GATLayer(nn.Module):
    def __init__(self, w_user, b_user, w_item, b_item,
                 u_att_heads, i_att_heads, nonlinear, reduce_att):
        super().__init__()
        self.w_user = w_user
        self.b_user = b_user
        self.w_item = w_item
        self.b_item = b_item
        self.u_att_heads = nn.ModuleList(u_att_heads)
        self.i_att_heads = nn.ModuleList(i_att_heads)
        self.nonlinear = nonlinear
        self.reduce_att = reduce_att

    def attend(self, adj, h_src, h_dst, att_heads):
        """
        Parameters
        ----------
        adj : (m, n)
        h_src : (m, d)
        h_dst : (n, d)
        att_heads :
        """
        def _attend(att_head):
            logits = att_head(h_src[adj.row], h_dst[adj.col])
            dat = neighborhood_softmax(adj, logits)
            att = ths.FloatTensor(adj.coo.indices(), dat, adj.coo.shape)
            return ths.mm(att, h_dst)
        return self.nonlinear(self.reduce_att([_attend(att_head) for att_head in att_heads]))

    def forward(self, u2i, i2u, u_prev, i_prev):
        """
        Parameters
        ----------
        u2i : (m, n)
        i2u : (n, m)
        u_prev : (m, d)
        i_prev : (n, d)
        """
        u = u_prev @ self.w_user() + self.b_user()
        i = i_prev @ self.w_item() + self.b_item()
        u_next = self.attend(u2i, u, i, self.u_att_heads)
        i_next = self.attend(i2u, i, u, self.i_att_heads)
        return u_next, i_next

class GAT(nn.Module):
    def __init__(self, n_feats, n_att_heads, nonlinear):
        super().__init__()
        n_layers = len(n_feats) - 1
        self.layers = nn.ModuleList()
        for i, [in_feats, out_feats] in enumerate(zip(n_feats[:-1], n_feats[1:])):
            in_feats *= 1 if i == 0 else 2 * n_att_heads
            w_user = Parameter(1e-1 * th.randn(in_feats, out_feats))
#           b_user = Parameter(th.zeros(1, out_feats))
            b_user = 0
            w_item = Parameter(1e-1 * th.randn(in_feats, out_feats))
#           b_item = Parameter(th.zeros(1, out_feats))
            b_item = 0
            u_att_heads = [Attention(out_feats) for i in range(n_att_heads)]
            i_att_heads = [Attention(out_feats) for i in range(n_att_heads)]
            cat = lambda xs: th.cat(xs, 1)
            mean = lambda xs: sum(xs) / len(xs)
            self.layers.append(GATLyaer(w_user, b_user, w_item, b_item,
                                        u_att_heads, i_att_heads,
                                        nonlinear, cat if i < n_layers - 1 else mean))

    def forward(self, u2i, i2u, x_user, x_item, h_prev=None):
        h_next = []
        u = x_user
        i = x_item
        for k, layer in enumerate(self.layers):
            u_bar, i_bar = layer(u2i, i2u, u, i)
            h_next.append([u_bar.detach(), i_bar.detach()])

            if h_prev is None:
                u_prev = th.zeros(x_user.size(0), u_bar.size(1), device=u_bar.device)
                i_prev = th.zeros(x_item.size(0), i_bar.size(1), device=i_bar.device)
            else:
                u_prev, i_prev = h_prev[k]
            u = th.cat([u_prev, u_bar], 1)
            i = th.cat([i_prev, i_bar], 1)

        return u, i, h_next

class MCGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats,
                 n_att_heads, reduce_att, n_chnls, reduce_chnl, nonlinear):
        super().__init__()
        self.nonlinear = nonlinear
        self.reduce_chnl = reduce_chnl

        self.w_user = SharedParameter([1e-1 * th.randn(in_feats, out_feats) \
                                       for i in range(n_chnls)])
        self.w_item = SharedParameter([1e-1 * th.randn(in_feats, out_feats) \
                                       for i in range(n_chnls)])
        self.u_src = nn.ModuleList([SharedParameter([1e-1 * th.randn(out_feats) \
                                    for j in range(n_chnls)]) for i in range(n_att_heads)])
        self.u_dst = nn.ModuleList([SharedParameter([1e-1 * th.randn(out_feats) \
                                    for j in range(n_chnls)]) for i in range(n_att_heads)])
        self.i_src = nn.ModuleList([SharedParameter([1e-1 * th.randn(out_feats) \
                                    for j in range(n_chnls)]) for i in range(n_att_heads)])
        self.i_dst = nn.ModuleList([SharedParameter([1e-1 * th.randn(out_feats) \
                                    for j in range(n_chnls)]) for i in range(n_att_heads)])
        self.layers = nn.ModuleList([])
        for i in range(n_chnls):
            u_att_heads = [Attention(a_src=self.u_src[j][i], a_dst=self.u_dst[j][i]) \
                           for j in range(n_att_heads)]
            i_att_heads = [Attention(a_src=self.i_src[j][i], a_dst=self.i_dst[j][i]) \
                           for j in range(n_att_heads)]
            self.layers.append(GATLayer(self.w_user[i], lambda: 0, self.w_item[i], lambda: 0,
                                        u_att_heads, i_att_heads, lambda x: x, reduce_att))

    def forward(self, u2is, i2us, u_prev, i_prev):
        device = u_prev.device
        uu, ii = zip(*[[None, None] if u2i is None else \
                       layer(u2i, i2u, u_prev[u2i.uniq_i], i_prev[i2u.uniq_i]) \
                       for u2i, i2u, layer in zip(u2is, i2us, self.layers)])
        d = next(filter(lambda u: u is not None, uu)).size(1)
        u_zeros = th.zeros(u_prev.size(0), d, device=device)
        i_zeros = th.zeros(i_prev.size(0), d, device=device)
        if self.reduce_chnl == 'cat':
            lambda_u = lambda u, u2i: u_zeros if u is None else \
                                      th.index_copy(u_zeros, 0, u2i.uniq_i, u)
            u_next = th.cat(list(starmap(lambda_u, zip(uu, u2is))), 1)
            lambda_i = lambda i, i2u: i_zeros if i is None else \
                                      th.index_copy(i_zeros, 0, i2u.uniq_i, i)
            i_next = th.cat(list(starmap(lambda_i, zip(ii, i2us))), 1)
        elif self.reduce_chnl == 'sum':
            u_next = sum(starmap(lambda u, u2i: th.index_copy(u_zeros, 0, u2i.uniq_i, u),
                                 starfilter(lambda u, _: u is not None, zip(uu, u2is))))
            i_next = sum(starmap(lambda i, i2u: th.index_copy(i_zeros, 0, i2u.uniq_i, i),
                                 starfilter(lambda i, _: i is not None, zip(ii, i2us))))
        return self.nonlinear(u_next), self.nonlinear(i_next)

class MCGAT(GAT):
    def __init__(self, n_feats, n_att_heads, n_chnls, reduce_chnl, nonlinear):
        nn.Module.__init__(self)
        n_layers = len(n_feats) - 1
        self.layers = nn.ModuleList()
        for i, [in_feats, out_feats] in enumerate(zip(n_feats[:-1], n_feats[1:])):
            if i > 0:
                in_feats *= 2 * n_att_heads * (1 if reduce_chnl == 'sum' else n_chnls)
            cat = lambda xs: th.cat(xs, 1)
            mean = lambda xs: sum(xs) / len(xs)
            self.layers.append(MCGATLayer(in_feats, out_feats, n_att_heads,
                                          cat if i < n_layers - 1 else mean,
                                          n_chnls, reduce_chnl, nonlinear))

    def forward(self, u2i, i2u, x_user, x_item, h_prev=None):
        return super().forward(u2i, i2u, x_user, x_item, h_prev)

class RatingPredictor(nn.Module):
    def __init__(self, in_feats, out_feats, n_bases, n_classes, activation=None):
        super().__init__()
        self.n_classes = n_classes
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = lambda x: x if activation is None else activation
        self.layer_norm = nn.LayerNorm(out_feats)
        self.coeffs = nn.Parameter(1e-1 * th.randn(n_bases, n_classes, 1, 1))
        self.bases = nn.Parameter(1e-1 * th.randn(n_bases, n_classes, out_feats, out_feats))

    def forward(self, u, i):
        u = self.layer_norm(self.activation(self.linear(u)))
        i = self.layer_norm(self.activation(self.linear(i)))
        u = u.unsqueeze(0).repeat(self.n_classes, 1, 1)
        i = i.unsqueeze(0).repeat(self.n_classes, 1, 1)
        kernels = th.sum(self.coeffs * self.bases, 0)
        return th.sum(u * th.bmm(i, kernels), 2).t()
