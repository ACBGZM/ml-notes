# pandas 合并 merge part3

import pandas as pd
import numpy as np

# ---merged by index---
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
print(left)
print(right)
# 不是考虑key column来合并，而是考虑index来合并
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print(res)