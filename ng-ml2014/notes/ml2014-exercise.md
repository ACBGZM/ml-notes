## 线性回归

#### 1. 梯度下降的实现

梯度下降第一种实现：**求偏导**
$$
{{\theta }_{j}}:={{\theta }_{j}}-\alpha \frac{\partial }{\partial {{\theta }_{j}}}J\left( \theta  \right)
$$

```python
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost
```



梯度下降第二种实现：**矩阵相乘**
$$
\theta = \theta - \alpha \frac{1}{m}X^T(X\theta - y)
$$

```python
def gradientDescent(X, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        
        if i % 100 == 0:
            print(i, cost)
        
    return theta, costs
```



#### 2. 归一化的两种方法：

**Standardization**

又称为Z-score normalization，量化后的特征将服从标准正态分布：
$$
z = \frac{x_i-\mu }{\delta}
$$
其中，$\mu$ 和 $\delta$ 分别为对应特征的均值和标准差。量化后的特征将分布在 [-1, 1] 区间。



**Min-Max Scaling**

又称为Min-Max normalization，特征量化的公式为：
$$
z = \frac{x_i-\min(x_i)}{\max(x_i)-\min(x_i)}
$$
量化后的特征将分布在区间 [0, 1]。