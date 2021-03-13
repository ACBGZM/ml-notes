# np.array的赋值，浅拷贝（指针）和深拷贝（对象）

import numpy as np

a = np.arange(4)
print(a)
b = a
c = b
d = a.copy()    # deep copy

a[1] = 10
print(a)
print(b)
print(c)
print(d)

# np.array 的名字可以理解成指针，b = a 这样的赋值只拷贝指针
# python 的普通变量，如 int b = a，其实是拷贝了对象本身。当改变 a 时，b 不会随之改变
