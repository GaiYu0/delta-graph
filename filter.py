import sys
import pandas as pd

df = pd.read_csv(sys.stdin, delimiter=' ', header=None, index_col=0)
