# pandas 设置值
# 跟上节一样，按位置、标签、布尔等

import pandas as pd
import numpy as np

dates = pd.date_range('20210224', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[2, 2] = 1111
print(df)

df.loc['20210227', 'B'] = 2222
print(df)

df.A[df.A > 8] = -1       # 只改 A 列的数据为 -1
print(df)

df[df.A > 0] = 0          # 改所有属性的值为 0
print(df)

# 添加属性
df['F'] = np.nan
# 值是 1, 2...6，按照日期的index对齐，附在后面
df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20210224', periods=6))
print(df)

