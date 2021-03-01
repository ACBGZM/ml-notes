# np.array的计算 part2

import numpy as np

A = np.arange(14, 2, -1).reshape((3, 4))
print(A)

print(np.argmax(A))     # 最大索引位置        0
print(np.argmin(A))     # 最小值的索引位置     11

print(np.mean(A))       # 平均值
print(np.median(A))     # 中位数
print(np.cumsum(A))     # 累加
print(np.diff(A))       # 累差

print(np.sort(A))       # 按每行排序

print(np.transpose(A))  # 矩阵转置

# 以上的操作，大部分都改写为矩阵是对象，操作是方法的形式。
print((A.T).dot(A))

print(np.clip(A, 5, 9)) # 修剪