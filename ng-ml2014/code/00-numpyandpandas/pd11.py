# pandas 合并 merge part4

import pandas as pd
import numpy as np

# ---两个表有相同的column---
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
print(boys)
print(girls)

# merge时，相同名字的column会被区分
res = pd.merge(boys, girls, on='k', how='outer')
print(res)

# 可以给同名column添加后缀名
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='outer')
print(res)
