# np.array的索引

import numpy as np

A = np.arange(3, 15).reshape((3, 4))
print(A)

print(A[2][1])
print(A[2, 1])  # 两种写法

print(A[2, :])  # 冒号是所有元素的意思。取第二行的所有数
print(A[:, 0])  # 第0列的所有元素
print(A[1, 1:4])    # 左闭右开


print('----遍历行、列、元素----')
# 按行迭代
for row in A:
    print(row)

# 按列迭代，即转置后迭代行
for col in A.T:
    print(col)

# 遍历所有元素，转换成向量
print(A.flatten())
for item in A.flat:
    print(item)
