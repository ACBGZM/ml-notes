# pandas 合并 merge part1

import pandas as pd
import numpy as np

print('合并1：相同key column合并')
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)

# ---merge的基本参数---
res = pd.merge(left, right, on='key')   # 在key column上合并
print(res)


print('合并2：不同key column合并')
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)

# ---merge的参数how---
# merge默认inner合并，只保留相同的key
print('---how=inner:')
res = pd.merge(left, right, on=['key1', 'key2'])
print(res)

# how='outer'，把没有的部分记nan
print('---how=outer:')
res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print(res)

# how='left' 或 how='right'，按照第一个或第二个参数的key进行merge。可以看到合并了两遍 K1 K0
# 不是参数的名字，而是参数的位置left和right
print('---how=right:')
res = pd.merge(left, right, on=['key1', 'key2'], how='right')
print(res)




