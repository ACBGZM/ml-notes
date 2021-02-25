# pandas plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Seires
data1 = pd.Series(np.random.randn(1000), index=np.arange(1000))
data1 = data1.cumsum()
data1.plot()
plt.show()

# DataFrame
data2 = pd.DataFrame(np.random.randn(1000, 4),
                    index=np.arange(1000),
                    columns=list("ABCD"))
data2 = data2.cumsum()
print(data2.head())
data2.plot()
plt.show()

ax = data2.plot.scatter(x='A', y='B', color='DarkBlue', label='Class1')
ax = data2.plot.scatter(x='A', y='C', color='DarkGreen', label='Class2', ax=ax)
plt.show()

