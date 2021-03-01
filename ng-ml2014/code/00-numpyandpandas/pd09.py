# pandas 合并 merge part2

import pandas as pd
import numpy as np

df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
print(df1)
print(df2)

# ---merge的参数indicator---
# indicator=True：显示合并的方式
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print(res)

# 可以给indicator列改名
res = pd.merge(df1, df2, on='col1', how='outer', indicator='合并方式')
print(res)





