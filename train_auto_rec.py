import argparse

import numpy as np
import torch as th
import torch.optim as optim
import torch.nn.functional as F

import auto_rec
import utils

curr_eval = lambda x: eval(x, globals(), locals())

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
parser.add_argument('--r', type=str, default='r.npy')
args = parser.parse_args()

uid = np.load(args.uid)
iid = np.load(args.iid)
r = np.load(args.r)

n_users = np.max(uid) + 1
n_items = np.max(iid) + 1

model = args.model(n_users, n_items, args.d, args.g, args.f)

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

uid = th.from_numpy(uid).to(device)
iid = th.from_numpy(iid).to(device)
r = th.from_numpy(r).to(device)

model = model.to(device)
opt = args.opt(model.parameters(), args.lr, weight_decay=args.wd)

n_train = int(args.p_train * len(r))
n_val = int(args.p_val * len(r))
perm = th.randperm(len(r), device=device)
idx_train = perm[:n_train]
idx_val = perm[n_train : n_train + n_val]
idx_test = perm[n_train + n_val:]

uid_train, iid_train, r_train = uid[idx_train], iid[idx_train], r[idx_train]
uid_val, iid_val, r_val = uid[idx_val], iid[idx_val], r[idx_val]
uid_test, iid_test, r_test = uid[idx_test], iid[idx_test], r[idx_test]

for i in range(args.n_iters):
    if args.bs is None:
        uid_batch, iid_batch, r_batch = uid_train, iid_train, r_train
    else:
        j = i % (len(r) // args.bs)
        idx_batch = range(j * args.bs, (j + 1) * args.bs)
        uid_batch, iid_batch, r_batch = uid[idx_batch], iid[idx_batch], r[idx_batch]

    s_batch = model(uid_batch, iid_batch, r_batch, uid_batch, iid_batch)
    rmse_batch = utils.rmse(r_batch, s_batch)

    opt.zero_grad()
    rmse_train.backward()
    opt.step()
    
    placeholder = '0' * (len(str(args.n_iters)) - len(str(i + 1)))
    print('[%s%d]rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
          (placeholder, i + 1, rmse_train, rmse_val, rmse_test))
