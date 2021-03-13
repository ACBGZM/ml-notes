# np.array的定义

import numpy as np

array = np.array([[1, 3, 5],
                  [2, 4, 6]], dtype=np.int64)

print(array)
print('number of dim:', array.ndim)     # number of dim: 2, “维度”，shape有几个数字
print('shape:', array.shape)            # shape: (2, 3)
print('size:', array.size)              # size: 6
print('dtype:', array.dtype)            # dtype: int64


a_1 = np.zeros((3, 4))                  # 全为0
print(a_1)
a_2 = np.ones((4, 3))                   # 全为1
print(a_2)
a_3 = np.empty((3, 3))                  # 生成一些接近0的数
print(a_3)


a_4 = np.arange(10, 20, 2)              # 左闭右开
print(a_4)


a_5 = np.arange(12)                     # (0, 12, 1)
print(a_5)
a_6 = np.arange(12).reshape((3, 4))     # 转换为3×4
print(a_6)


a_7 = np.linspace(1, 10, 5)             # 生成等分线段，自动分配步长
print(a_7)
