import torch as th
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, n_users, n_items, d):
        super().__init__()
        self.h_user = nn.Parameter(1e-3 * th.randn(n_users, d))
        self.h_item = nn.Parameter(1e-3 * th.randn(n_items, d))

    def forward(self, uid, iid, s=None):
        if s is None:
            uids, iid = [uid], [iid]
        else:
            uids, iids = th.split(uid, s), th.split(iid, s)
        return th.cat([th.sum(self.h_user[uid] * self.h_item[iid], 1) \
                       for uid, iid in zip(uids, iids)])
