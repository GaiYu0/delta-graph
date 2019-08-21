import argparse
from functools import partial

import numpy as np
import torch as th
import torch.optim as optim
import torch.nn.functional as F

import auto_rec
import utils

curr_eval = partial(eval, globals=globals(), locals=locals())

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=int, required=True)
parser.add_argument('-f', type=curr_eval, required=True)
parser.add_argument('-g', type=curr_eval, required=True)
parser.add_argument('--bs', type=int)
parser.add_argument('--gpu', type=int, default='-1')
parser.add_argument('--iid', type=str, default='iid.npy')
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--model', type=curr_eval, required=True)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--opt', type=curr_eval, required=True)
parser.add_argument('--p-train', type=float, required=True)
parser.add_argument('--p-val', type=float, required=True)
parser.add_argument('--uid', type=str, default='uid.npy')
parser.add_argument('--wd', type=float, required=True)
parser.add_argument('--y', type=str, default='y.npy')
args = parser.parse_args()

uid = np.load(args.uid)
iid = np.load(args.iid)
y = np.load(args.y)

n_users = np.max(uid) + 1
n_items = np.max(iid) + 1

model = args.model(n_users, n_items, args.d, args.g, args.f)

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

uid = th.from_numpy(uid).to(device)
iid = th.from_numpy(iid).to(device)
y = th.from_numpy(y).to(device)

model = model.to(device)
opt = args.opt(model.parameters(), lr, weight_decay=args.wd)

n_train = int(args.p_train * len(y))
n_val = int(args.p_val * len(y))
perm = th.randperm(len(y), device=device)
idx_train = perm[:n_train]
idx_val = perm[n_train : n_train + n_val]
idx_test = perm[n_train + n_val:]

uid_train, iid_train, y_train = uid[idx_train], iid[idx_train], y[idx_train]
uid_val, iid_val, y_val = uid[idx_val], iid[idx_val], y[idx_val]
uid_test, iid_test, y_test = uid[idx_test], iid[idx_test], y[idx_test]

indices = th.stack(ij_train, 1)
size = [n_items, n_users] if args.i else [n_users, n_items]
r = sparse.FloatTensor(indices, y_train, size)

for i in range(args.n_iters):
    if args.bs is None:
        uid_batch, iid_batch, y_batch = uid_train, iid_train, y_train
    else:
        j = i % (len(y) // args.bs)
        idx_batch = range(j * args.bs, (j + 1) * args.bs)
        uid_batch, iid_batch, y_batch = uid[idx_batch], iid[idx_batch], y[idx_batch]

    z_batch = model(r, uid_batch, iid_batch)
    rmse_batch = utils.rmse(y_batch, z_batch)

    opt.zero_grad()
    rmse_train.backward()
    opt.step()
    
    placeholder = '0' * (len(str(args.n_iters)) - len(str(i + 1)))
    print('[%s%d]rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
          (placeholder, i + 1, rmse_train, rmse_val, rmse_test))
