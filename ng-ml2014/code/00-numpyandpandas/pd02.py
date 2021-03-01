# pandas 查找数据
# 每行存放一个数据元素，列标签是某个属性。
# 最终查找的都是数据元素，也就是行。
# 几种常用方法：
#   按索引，loc
#   按位置，iloc
#   （混用：ix，现版本已弃用）
#   布尔型

import pandas as pd
import numpy as np

dates = pd.date_range('20210224', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

print(df)

# 按列的索引查找，两种写法等价
print('查找所有元素的索引A：')
print(df['A'])
print(df.A)

# 按行查找。只查一个是错误的，如 df[2] 和 df['20210225']
print('按行查找：')
print(df[0:3])                      # 按标号
print(df['20210225':'20210228'])    # 按自定义标签

# 按标签查找
print('按标签查找：')
print(df.loc['20210225'])

###
# 注意 df['20210225'] 是错误的
#     df.loc['20210225'] 是正确的
#     df['A'] 和 df.A 是正确的
# 要区分行标签和列索引。
# 最终查找的都是以行，为单位的向量。列只是筛选显示向量的某个属性。
# 理解为行标签是一个向量的名字，属于向量；列索引是表单的属性名称，属于表本身。
###

# 行列都指定查找
# select by label: loc
print('用loc按标签查找：')
print('--查找全部元素，索引A、B：')
print(df.loc[:, ['A', 'B']])
print('--查找标签0226的元素，索引A、B：')
print(df.loc['20210226', ['A', 'B']])

# select by position: iloc
# 形式上完全按照二维数组查找
print('用iloc按位置查找：')
print(df.iloc[3, 1])
print(df.iloc[3:5, 1:3])
print(df.iloc[[1, 3, 5], 1:3])          # 不连续的筛选

# Boolean indexing
print('布尔型查找：')
print(df[df.A > 8])
