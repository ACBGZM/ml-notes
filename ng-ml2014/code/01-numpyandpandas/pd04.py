# pandas 处理丢失数据

import pandas as pd
import numpy as np

dates = pd.date_range('20210225', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
# 用iloc按位置查找
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan

print(df)
# dropna
# how = {'any', 'all'}
print('丢掉有nan的行：')
print(df.dropna(axis=0, how='any'))  # 有任一个nan，就把行丢掉
print('丢掉有nan的列：')
print(df.dropna(axis=1, how='any'))  # 有任一个nan，就把列丢掉

# fillna
print('填充nan：')
print(df.fillna(value=0))  # 把nan填充为0

print('检查是否为nan：')
print(df.isnull())
print('表中是否存在nan：')
print(np.any(df.isnull()) == True)
