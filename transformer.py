import torch as th
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, user_w_qs, user_w_ks, user_w_vs,
                       item_w_qs, item_w_ks, item_w_vs, nonlinear):
        super().__init__()
        self.user_w_qs = user_w_qs
        self.user_w_ks = user_w_ks
        self.user_w_vs = user_w_vs
        self.item_w_qs = item_w_qs
        self.item_w_ks = item_w_ks
        self.item_w_vs = item_w_vs
        self.nonlinear = nonlinear

    def attend(self, adj, h_src, h_dst, w_qs, w_ks, w_vs):
        zs = []
        for w_q, w_k, w_v in zip(w_qs, w_ks, w_vs):
            q = h_dst @ w_q
            k = h_src @ w_k
            v = h_src @ w_v
            logit = th.sum(q * k, 1)
            att = self.neighborhood_softmax(adj, logit)
            z = self.nonlinear(ths.FloatTensor(adj.idx, att, adj.size) @ v)
            zs.append(z)
        return self.merge(zs)

    def forward(self, u2i, i2u, h_user, h_item):
        z_user = self.attend(i2u, h_item, h_user, self.user_w_qs, self.item_w_ks, self.item_w_vs)
        z_item = self.attend(u2i, h_user, h_item, self.item_w_qs, self.user_w_ks, self.user_w_vs)
        return z_user, z_item

class Transformer(nn.Module):
    def __init__(self, user_n_feats, item_n_feats, n_att_heads, nonlinear):
        super().__init__()
        user_sizes = zip(user_n_feats[:-1], user_n_feats[1:])
        item_sizes = zip(item_n_feats[:-1], item_n_feats[1:])
        layers = nn.ModuleList()
        for i, [user_in_feats, user_out_feats], d_user, \
               [item_in_feats, item_out_feats], d_item in enumerate(zip(user_sizes, item_sizes)):
            user_w_qs = nn.ParameterList([nn.Parameter(th.rand(user_in_feats, d_user)) \
                                          for i in range(n_att_heads)])
            user_w_ks = nn.ParameterList([nn.Parameter(th.rand(user_in_feats, d_item)) \
                                          for i in range(n_att_heads)])
            user_w_vs = nn.ParameterList([nn.Parameter(th.rand(user_in_feats, item_out_feats)) \
                                          for i in range(n_att_heads)])
            item_w_qs = nn.ParameterList([nn.Parameter(th.rand(item_in_feats, d_item)) \
                                          for i in range(n_att_heads)])
            item_w_ks = nn.ParameterList([nn.Parameter(th.rand(item_in_feats, d_user)) \
                                          for i in range(n_att_heads)])
            item_w_vs = nn.ParameterList([nn.Parameter(th.rand(item_in_feats, user_out_feats)) \
                                          for i in range(n_att_heads)])
            layers.append(TransformerLayer(user_w_qs, user_w_ks, user_w_vs,
                                           user_w_qs, user_w_ks, user_w_vs, nonlinear))

    def forward(self, u2i, i2u, x_user, x_item):
        h_user = x_user
        h_item = x_item
        for layer in self.layers:
            h_user, h_item = layer(h_user, h_item)
        return h_user, h_item
