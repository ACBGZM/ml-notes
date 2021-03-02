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





## 逻辑回归

#### 1. 逻辑回归总览

损失函数的向量化表示
$$
h = g(X\theta)
$$

$$
J(\theta) = -\frac{1}{m}[y*\log(h)+(1-y)*\log(1-h)]
$$

梯度下降函数的向量化表示
$$
\theta = \theta - \frac{a}{m}*X^T(g(X\theta)-y)
$$
维度：X(m, n), y(m, 1), $\theta$(n, 1)



#### 2.代价函数

**代价函数1：**
$$
{{h}_{\theta }}\left( x \right)=\frac{1}{1+{{e}^{-{{\theta }^{T}}X}}}
$$

$$
J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]}
$$

```python
def cost(theta, X, y):
    theta = np.zeros(3)			# (1, 3)
    theta = np.matrix(theta)	# (3,)
    X = np.matrix(X)			# (100, 3)
    y = np.matrix(y)			# (100, 1)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
```



**向量化的代价函数2：**
$$
g\left( z \right)=\frac{1}{1+{{e}^{-z}}}
$$

$$
{{h}_{\theta }}\left( x \right)=\frac{1}{1+{{e}^{-{X{\theta}}}}}\\
J(\theta) = -\frac{1}{m}[y*\log(h)+(1-y)*\log(1-h)
$$

```python
def costFunction(X, y, theta):
    A = sigmoid(X @ theta)   # 矩阵乘法，A(m, 1)
    
    first = y * np.log(A)    # 对应元素相乘，两个矩阵都是 (m, 1)
    second = (1 - y) * np.log(1 - A)
    
    return -np.sum(first + second) / len(X)    # first 和 second 都是向量，此处要求和，计算总的损失
```



