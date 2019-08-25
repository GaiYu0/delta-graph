import argparse

import numpy as np
from tensorboardX import SummaryWriter
import torch as th
from torch.optim import *
import torch.nn.functional as F

from auto_rec import *
from mf import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bs-infer', type=int)
parser.add_argument('--bs-train', type=int)
parser.add_argument('--ds', type=str)
parser.add_argument('--gpu', type=int, default='-1')
parser.add_argument('--logdir', type=str)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--optim', type=str, required=True)
parser.add_argument('--p-train', type=float, required=True)
parser.add_argument('--p-val', type=float, required=True)
args = parser.parse_args()

uid = np.load(args.ds + '/uid.npy')
iid = np.load(args.ds + '/iid.npy')
r = np.load(args.ds + '/r.npy')

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
perm = th.randperm(len(r), device=device)
uid = th.from_numpy(uid).to(device)[perm]
iid = th.from_numpy(iid).to(device)[perm]
r = th.from_numpy(r).to(device)[perm]
r_max = th.max(r)
r /= r_max
r_mean = th.mean(r)

n_train = int(args.p_train * len(r))
n_val = int(args.p_val * len(r))
n_test = len(r) - n_train - n_val
uid_train, uid_val, uid_test = th.split(uid, [n_train, n_val, n_test])
iid_train, iid_val, iid_test = th.split(iid, [n_train, n_val, n_test])
r_train, r_val, r_test = th.split(r, [n_train, n_val, n_test])

n_users = th.max(uid) + 1
n_items = th.max(iid) + 1

model = eval(args.model).to(device)
optim = eval(args.optim)

writer = SummaryWriter(args.logdir)

for i in range(args.n_iters):
    if args.bs_train is None:
        uid_batch, iid_batch, r_batch = uid_train, iid_train, r_train
    else:
        perm_batch = th.randperm(n_train, device=device)[:args.bs_train]
        uid_batch, iid_batch, r_batch = uid[perm_batch], iid[perm_batch], r[perm_batch]

    for p in model.parameters():
        p.requires_grad = True
    s_batch = model(uid_train, iid_train, r_train, uid_batch, iid_batch)
    mse = F.mse_loss(r_batch, s_batch)
    optim.zero_grad()
    mse.backward()
    optim.step()

    for p in model.parameters():
        p.requires_grad = False
    s = model(uid_train, iid_train, r_train, uid, iid, args.bs_infer)
    s_train, s_val, s_test = th.split(s, [n_train, n_val, n_test])
    rmse_batch = r_max * mse ** 0.5
    rmse_train = r_max * utils.rmse_loss(r_train, s_train)
    rmse_val = r_max * utils.rmse_loss(r_val, s_val)
    rmse_test = r_max * utils.rmse_loss(r_test, s_test)

    placeholder = '0' * (len(str(args.n_iters)) - len(str(i + 1)))
    print('[%s%d]rmse_batch: %.3e | rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
          (placeholder, i + 1, rmse_batch, rmse_train, rmse_val, rmse_test))

    writer.add_scalar('rmse_batch', rmse_batch.item(), i + 1)
    writer.add_scalar('rmse_train', rmse_train.item(), i + 1)
    writer.add_scalar('rmse_val', rmse_val.item(), i + 1)
    writer.add_scalar('rmse_test', rmse_test.item(), i + 1)

writer.close()
