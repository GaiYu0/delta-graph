import argparse
import itertools

import numpy as np
import numpy.random as npr
import scipy.sparse as sps

def coalesce(i, j):
    coo = sps.coo_matrix((np.ones(len(i)), (i, j))).tocsr().tocoo()
    return coo.row, coo.col

def generate(n_users, n_ctgrs, n_items_per_ctgr, n_sesses, n_items_per_sess, p_div):
    """
    Parameters
    ----------
    n_users : int
        Number of users.
    n_ctgrs : int
        Number of categories.
    n_items_per_ctgr : int
        Number of items per category.
    n_sesses : int
        Number of sessions.
    n_items_per_sess : int
        Number of items per session.
    p_div : float
        Probability of diversion. The probability of a user in a session viewing an item not belonging to the session category is $\frac{p_div (n_ctgrs - 1)}{n_ctgr}$.
    """
    sess_ctgrs = npr.choice(n_ctgrs, size=[n_users, n_sesses, 1])
    n_div = int(p_div * n_items_per_sess)
    item_ctgrs = np.dstack([npr.choice(n_ctgrs, size=[n_users, n_sesses, n_div]),
                            np.tile(sess_ctgrs, [1, 1, n_items_per_sess - n_div])])
    item_offsets = npr.choice(n_items_per_ctgr, size=[n_users, n_sesses, n_items_per_sess])
    items = item_ctgrs * n_items_per_ctgr + item_offsets
    uid = np.repeat(np.arange(n_users), n_sesses * n_items_per_sess)
    iid = np.reshape(items, [-1])
    sess_uids = n_sesses * [np.repeat(np.arange(n_users), n_items_per_sess)]
    sess_iids = map(np.squeeze,
                    np.vsplit(items.transpose([1, 0, 2]).reshape([n_sesses, -1]), n_sesses))
    return coalesce(uid, iid), zip(*itertools.starmap(coalesce, zip(sess_uids, sess_iids)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-users', type=int)
    parser.add_argument('--n-ctgrs', type=int)
    parser.add_argument('--n-items-per-ctgr', type=int)
    parser.add_argument('--n-sesses', type=int)
    parser.add_argument('--n-items-per-sess', type=int)
    parser.add_argument('--p-div', type=float)
    args = parser.parse_args()

    [uid, iid], [sess_uids, sess_iids] = generate(**vars(args))

    np.save('uid', uid)
    np.save('iid', iid)
    np.save('uid-sess', sess_uids)
    np.save('iid-sess', sess_iids)
