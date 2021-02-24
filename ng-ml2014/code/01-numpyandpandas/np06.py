# np.array的分割

import numpy as np

A = np.arange(12).reshape((3, 4))
print(A)

# 分割
# 分割时，h和v表示的方向好像有点奇怪，不求理解
print('----等量的分割----')
print(np.split(A, 2, axis=1))       # 纵向分割成两块，同 hsplit
print(np.hsplit(A, 2))
print(np.split(A, 3, axis=0))       # 横向分割成三块，同 vsplit
print(np.vsplit(A, 3))

print('----不等量的分割----')
print(np.array_split(A, 3, axis=1))       # 纵向分割成三块