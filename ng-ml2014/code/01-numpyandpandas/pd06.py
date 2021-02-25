# pandas 合并 concat

import pandas as pd
import numpy as np

dates = pd.date_range('20210225', periods=3)


# concatenating
df1 = pd.DataFrame(np.ones((3, 4))*0, index=dates, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])
print(df1)
print(df2)
print(df3)

# ---concat的基本参数---
# 上下合并，指定延用元index还是重新分配连续的index
res1 = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res1)

res2 = pd.concat([df1, df2, df3], axis=0)
print(res2)


print('-----------')
df4 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df5 = pd.DataFrame(np.ones((3, 4))*1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
print(df4)
print(df5)

# ---concat的参数join = {'inner', 'outer'}---
res3 = pd.concat([df4, df5], join='outer')  # 不共同的地方填nan
print(res3)

res4 = pd.concat([df4, df5], join='inner')  # 把不共同的纵向index删掉
print(res4)

res5 = pd.concat([df4, df5], join='inner', ignore_index=True)   # 重新编号
print(res5)