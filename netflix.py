import multiprocessing as mp
import sys

def process(mid):
    with open('data.txt/mv_%s%d.txt' % ('0' * (7 - len(str(mid))), mid), 'r') as f:
        next(f)
        with open('data.csv/%d.csv' % mid, 'w') as g:
            g.write(''.join(map(lambda l: str(mid) + ',' + l, f)))

n_movies = 17770
with mp.Pool(int(sys.argv[1])) as pool:
    pool.map(process, range(1, n_movies + 1))
