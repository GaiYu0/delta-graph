import argparse

import numpy as np
from tensorboardX import SummaryWriter
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
parser.add_argument('--bs-infer', type=int)
parser.add_argument('--bs-train', type=int)
parser.add_argument('--ds', type=str)
parser.add_argument('--gpu', type=int, default='-1')
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--model', type=curr_eval, required=True)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--opt', type=curr_eval, required=True)
parser.add_argument('--p-train', type=float, required=True)
parser.add_argument('--p-val', type=float, required=True)
parser.add_argument('--wd', type=float, required=True)
args = parser.parse_args()

uid = np.load(args.ds + '/uid.npy')
iid = np.load(args.ds + '/iid.npy')
r = np.load(args.ds + '/r.npy') / 5

n_users = np.max(uid) + 1
n_items = np.max(iid) + 1

model = args.model(n_users, n_items, args.d, args.g, args.f)

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

perm = th.randperm(len(r), device=device)
uid = th.from_numpy(uid).to(device)[perm]
iid = th.from_numpy(iid).to(device)[perm]
r = th.from_numpy(r).to(device)[perm]

n_train = int(args.p_train * len(r))
n_val = int(args.p_val * len(r))
n_test = len(r) - n_train - n_val
uid_train, iid_train, r_train = uid[:n_train], iid[:n_train], r[:n_train]
r_val = r[n_train : n_train + n_val]
r_test = r[n_train + n_val:]

model = model.to(device)
opt = args.opt(model.parameters(), args.lr, weight_decay=args.wd)

writer = SummaryWriter()
# writer = SummaryWriter('runs/' + str(args).replace(' ', ''))

for i in range(args.n_iters):
    if args.bs_train is None:
        uid_batch, iid_batch, r_batch = uid_train, iid_train, r_train
    else:
        perm_batch = th.randperm(n_train, device=device)[:args.bs_train]
        uid_batch, iid_batch, r_batch = uid[perm_batch], iid[perm_batch], r[perm_batch]

    s_batch = model(uid_batch, iid_batch, r_batch)
    mse = utils.mse(r_batch, s_batch)
    opt.zero_grad()
    mse.backward()
    opt.step()

    s = model(uid_train, iid_train, r_train, uid, iid, args.bs_infer)
    s_train, s_val, s_test = th.split(s, [n_train, n_val, n_test])
    rmse_train = 5 * utils.rmse(r_train, s_train)
    rmse_val = 5 * utils.rmse(r_val, s_val)
    rmse_test = 5 * utils.rmse(r_test, s_test)

    placeholder = '0' * (len(str(args.n_iters)) - len(str(i + 1)))
    print('[%s%d]mse: %.3e | rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
          (placeholder, i + 1, mse, rmse_train, rmse_val, rmse_test))

    writer.add_scalar('mse', mse.item(), i + 1)
    writer.add_scalar('rmse_train', rmse_train.item(), i + 1)
    writer.add_scalar('rmse_val', rmse_val.item(), i + 1)
    writer.add_scalar('rmse_test', rmse_test.item(), i + 1)
