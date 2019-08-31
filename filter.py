import fileinput
import sys
import pandas as pd

kwargs = eval(sys.argv[1])
df = pd.read_csv(''.join(fileinput.input()), **kwargs)
print(df.filter(like=eval(sys.argv[2])))
