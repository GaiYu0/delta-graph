import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], delimiter=' ', header=None, index_col=0)
print(df.loc[eval(sys.argv[2]), :].values.item())
