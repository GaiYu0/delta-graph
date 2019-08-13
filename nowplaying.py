import argparse

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, FloatType

parser = argparse.ArgumentParser()
parser.add_argument('--nowplaying-path', type=str)
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
ccf = sqlCtx.read.csv(args.nowplaying_path + 'context_content_feats.csv', header=True)
ccf = ccf.select(
#                'coordinates',  # 38058/11614671
                 'instrumentalness',  #  11611869/11614671
                 'liveness',  #  11611757/11614671
                 'speechiness',  #  11610783/11614671
                 'danceability',  #  11610783/11614671
                 'valence',  #  11609883/11614671
                 'loudness',  #  11614671/11614671
                 'tempo',  #  11614671/11614671
                 'acousticness',  #  11611884/11614671
                 'energy',  #  11611910/11614671
                 'mode',  #  11611957/11614671
                 'key',  #  11611957/11614671
                 'artist_id',  #  11614671/11614671
#                'place',  #  44344/11614671
#                'geo',  #  38061/11614671
                 'tweet_lang',  #  11614671/11614671
                 'track_id',  #  11614671/11614671
                 'created_at',  #  11614671/11614671
                 'lang',  #  11614669/11614671
#                'time_zone',  #  8354318/11614671
#                'entities',  #  12/11614671
                 'user_id',  #  11614671/11614671
                 'id')  #  11614671/11614671
ccf = ccf.dropna('any')

def integerize(df, name):
    keys = df.select(name).dropDuplicates().rdd.map(lambda x: x).collect()
    values = range(len(values))
    d = dict(zip(keys, values))
    getitem = F.udf(d.__getitem__, IntegerType())
    return df.withColumn(name, getitem(name))

ccf = integerize(integerize(ccf, 'tweet_lang'), 'lang')
continuous_feats = ['instrumentalness',
                    'liveness',
                    'speechiness',
                    'danceability',
                    'valence',
                    'loudness',
                    'tempo',
                    'acousticness',
                    'energy']
discrete_feats = ['mode', 'key', 'tweet_lang', 'lang']
feats = []
for name in continuous_feats:
    feats.append(np.array(ccf.select(name).rdd.flatMap(int).collect()))
for name in discrete_feats:
    rdd = ccf.select(name).rdd.flatMap(int)
    zeros = np.zeros(rdd.count(), rdd.max())
    idx = np.array(rdd.collect())
    zeros[np.arange(len(zeros)), idx] = 1
    feats.append(zeros)
