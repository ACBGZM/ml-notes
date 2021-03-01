# np.array的计算 part1

import numpy as np


# 数值运算：矩阵的对应元素做相应计算
a = np.array([10, 20, 30, 40])
b = np.arange(4)
print('a:', a,'b:', b)

c_1 = a+b
print('a+b:', c_1)

c_2 = b**2
print('b**2:', c_2)

c_3 = a*b
print('a*b:', c_3)

print('b<3:', b<3)


# 矩阵运算：矩阵乘法
a = np.array([[1, 1],
              [0, 1]])
b = np.arange(4).reshape((2, 2))
print('a:', a)
print('b:', b)

c_1 = a*b               # 对应元素相乘
c_2 = np.dot(a, b)      # 矩阵乘法
print('a*b:', c_1)
print('dot(a, b):', c_2)


# 随机生成
# 求和、找最值
a = np.random.random((2, 4))    # 随机生成 0~1 的数
print(a)
print(np.sum(a, axis=1))        # 1：按行求和
print(np.max(a, axis=0))        # 0：按列找最大值
print(np.min(a, axis=1))        # 1：按行找最小值
