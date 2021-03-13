# pandas 合并 append

import pandas as pd
import numpy as np

# append：按行向下添加
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((2, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((1, 4))*2, columns=['a', 'b', 'c', 'd'])
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# 注意Series的index相当于DataFrame的Columns

res = df1.append([df2, df3])    # 可以append多个DataFrame
print(res)

res = df1.append(s1, ignore_index=True)     # 可以一项一项地加，一次添加一个Series
print(res)

res = df1.append([df2, s1], ignore_index=True)   # 这样混用是不行的
print(res)