import argparse

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

parser = argparse.ArgumentParser()
parser.add_argument('--outbrain-path', type=str)
parser.add_argument('--publisher-id', type=int)
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
page_views = ss.read.csv(args.outbrain_path + 'page_views.csv', header=True)
page_views = page_views.select('uuid', 'document_id', 'timestamp') \
                       .withColumn('uuid', F.hex(page_views.uuid)) \
                       .withColumn('document_id', page_views.document_id.cast(IntegerType())) \
                       .withColumn('timestamp', page_views.timestamp.cast(IntegerType()))

doc_meta = ss.read.csv(args.outbrain_path + 'documents_meta.csv', header=True)
doc_meta = doc_meta.select('document_id', 'publisher_id') \
                   .withColumn('document_id', doc_meta.document_id.cast(IntegerType())) \
                   .withColumn('publisher_id', doc_meta.publisher_id.cast(IntegerType()))

df = page_views.join(doc_meta.filter(doc_meta.publisher_id == args.publisher_id), 'document_id')
df.write.orc('views', mode='overwrite')
