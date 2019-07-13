import argparse
import pickle

import numpy as np
from pyspark.sql.session import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--delta-t', type=int)
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
df = ss.read.orc('views')

uuid = np.array(df.select('uuid').rdd.flatMap(lambda x: x).collect(), dtype=np.int64)
uniq_uuid, inv_uuid = np.unique(uuid, inverse=True)
n_users = len(uniq_uuid)
uid = np.arange(n_users)[inv_uuid]

doc_id = np.array(df.select('document_id').rdd.flatMap(lambda x: x).collect(), dtype=np.int64)
uniq_doc_id, inv_doc_id = np.unique(doc_id, inverse=True)
n_items = len(uniq_doc_id)
iid = np.arange(n_items)[inv_doc_id]

t = np.array(df.select('timestamp').rdd.flatMap(lambda x: x).collect(), dtype=np.int64)
idx = (t - np.min(t)) % args.delta_t

pickle.dump([n_users, n_items], open('size.pckl', 'wb'))
np.save('uid', uid)
np.save('iid', iid)
np.save('idx', idx)
