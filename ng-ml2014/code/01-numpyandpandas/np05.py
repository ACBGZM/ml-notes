# np.array的合并

import numpy as np

A = np.array([1, 1, 1])
B = np.array([2, 2, 2])

# 合并
C = np.vstack((A, B))       # vertical stack
D = np.hstack((A, B))       # horizontal stack

print(C)
print(A.shape, C.shape)
print(D)
print(A.shape, D.shape)


# 加一个维度
print(A, A.shape)
A_1 = A[np.newaxis, :]      # 横向加一个维度
A_2 = A[:, np.newaxis]      # 纵向加一个维度
print(A_1, A_1.shape)
print(A_2, A_2.shape)


print('----concatenate合并----')
# 另一种合并，可以指定合并的方向
A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]
print(A)
print(B)

C = np.concatenate((A, B, B, A), axis=0)        # 纵向合并，相当于 vstack
print(C)
print(np.vstack((A, B, B, A)))                  # 两者等价

C = np.concatenate((A, B, B, A), axis=1)        # 横向合并，相当于 hstack
print(C)
print(np.hstack((A, B, B, A)))                  # 两者等价
