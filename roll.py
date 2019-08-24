import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--p-train', type=float, required=True)
parser.add_argument('--p-val', type=float, required=True)
args = parser.parse_args()

uid = np.load(args.ds + '/uid.npy')
iid = np.load(args.ds + '/iid.npy')
r = np.load(args.ds + '/r.npy') / 5
t = np.load(args.ds + '/t.npy')

n_users = np.max(uid) + 1
n_items = np.max(iid) + 1

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

perm = th.randperm(len(r), device=device)
uid = th.from_numpy(uid).to(device)[perm]
iid = th.from_numpy(iid).to(device)[perm]
r = th.from_numpy(r).to(device)[perm]
t = th.from_numpy(t).to(device)[perm]

masks = [(t == i) for i in np.unique(t, sorted=True)]
uids = list(map(uid.__getitem__, masks))
iids = list(map(iid.__getitem__, masks))
rs = list(map(r.__getitem__, masks))

for i in range(args.n_iters):
    for j in range(1, len(masks)):
        n_train = int(args.p_train * len(rs[j]))
        n_val = int(args.p_val * len(rs[j]))
        n_test = len(rs[j]) - n_train - n_val

        uid_train, uid_val, uid_test = th.split(uids[j], [n_train, n_val, n_test])
        iid_train, iid_val, iid_test = th.split(iids[j], [n_train, n_val, n_test])
        r_train, r_val, r_test = th.split(rs[j], [n_train, n_val, n_test])
        for p in model.parameters():
            p.requires_grad = True
        loss = model(uids[:j], iids[:j], rs[:j], uid_train, iid_train, r_train)
