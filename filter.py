import sys
import pandas as pd

df = pd.read_csv(sys.stdin, delimiter=' ', header=None, index_col=0)
print(df.loc[eval(sys.argv[1]), :].values.item())
