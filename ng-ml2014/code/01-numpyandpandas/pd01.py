# 初识pandas

import pandas as pd
import numpy as np

# 没有给出名字，会从0开始排序
s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)

# 给出名字，就按给出的排序
dates = pd.date_range('20210224', periods=6)
print(dates)
# DataFrame类似于二维numpy。给出行索引index，列索引columns。
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)
# 不给出索引，就默认从0开始
df_1 = pd.DataFrame(np.random.randn(6, 4))
print(df_1)
# 也可以用字典代替
df_2 = pd.DataFrame({'A': 1.,
                     'B': pd.Timestamp('20210224'),
                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                     'D': np.array([3]*4, dtype='int32'),
                     'E': pd.Categorical(["test", "train", "test", "train"]),
                     'F': 'foo'})
print(df_2)
print('dtypes:\n', df_2.dtypes)         # 每列的数据类型
print('index:\n', df_2.index)           # 行的索引
print('columns:\n', df_2.columns)       # 列的索引
print('values:\n', df_2.values)         # 值
print('describe:\n', df_2.describe())   # 数值元素列的统计数据

# 排序，不会对列表本身造成影响，因此print不出来
# 在console里把以下复制一下看结果
df_2.sort_index(axis=1, ascending=False)    # 按索引排序。按列，降序
df_2.sort_index(axis=0, ascending=False)    # 按索引排序。按行，降序
df_2.sort_values(by='E')                    # 按值排序。按E行，默认升序

