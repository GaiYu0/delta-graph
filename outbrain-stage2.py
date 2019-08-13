import argparse
import pickle

import numpy as np
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, FloatType

parser = argparse.ArgumentParser()
parser.add_argument('--outbrain-path', type=str)
args = parser.parse_args()

n_users, n_items = pickle.load(open('size.pckl', 'rb'))
ss = SparkSession.builder.getOrCreate()
views = ss.read.orc('views')
doc_ids = views.select('document_id').dropDuplicates()
xs = []
for single, plural in [['category', 'categories'], ['topic', 'topics']]:
    df = ss.read.csv(args.outbrain_path + 'documents_%s.csv' % plural, header=True)
    df = df.join(doc_ids, 'document_id') \
           .withColumn('i', getattr(df, single + '_id').cast(IntegerType())) \
           .withColumn('p', df.confidence_level.cast(FloatType()))
    i = np.array(df.select('i').rdd.flatMap(lambda x: x).collect())
    uniq_i, inv_i = np.unique(i, return_inverse=True)
    print(i.shape, inv_i.shape)
    x = np.zeros([n_items, len(uniq_i)])
    x[np.arange(n_items), np.arange(len(uniq_i))[inv_i]] = 1
    p = np.array(df.select('p').rdd.flatMap(lambda x: x).collect())
    xs.append(x * np.reshape(p, [-1, 1]))
x_user = np.zeros([n_users, 16])
x_item = np.hstack(xs)
np.save('x-user', x_user)
np.save('x-item', x_item)
