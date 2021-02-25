# pandas 导入导出数据
# 官方文档https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html

import pandas as pd

data = pd.read_csv('student.csv')
print(data)

data.to_pickle('student.pickle')