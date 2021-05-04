## Tensorflow 2.0 笔记

整理自北大曹建老师网课，见[此处](https://www.bilibili.com/video/BV1B7411L7Qt)

### 张量 Tensor

Tensor：多维数组（列表）。阶：张量的维数。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf1.png'  width="80%" height="80%"/>

数据类型：

- 整型、浮点型：`tf.int32` `tf.float32` `tf.float64` 
- bool型：`tf.constant([True, False])` 
- string型：`tf.constant("Hello, world.")`

创建张量：

- z直接创建：`tf.constant(张量内容, dtype=数据类型)`
- 将numpy转换为Tensor：`tf.convert_to_tensor(原数据名, dtype=数据类型)`

- 特殊张量：`tf.zeros(维度)` `tf.ones(维度)` `tf.fill(维度，指定值)`，如`tf.fill([3, 2, 4], 10)`

> 维度的表示：向量[ , , ]的逗号隔开每个维度的元素个数。有几个逗号就代表tensor有+1维。

- 随机张量：
  - 正态分布：`tf.random.normal(维度, mean=均值, stddev=标准差)`
  - 截断式正态分布：`tf.random.truncated_normal(维度, mean=均值, stddev=被侦测)`
  - 均匀分布随机数：`tf.random.uniform(维度, minval=最小值, maxval=最大值)`

> 区间左闭右开



### 常用函数

- `tf.cast(原张量名, dtype=数据类型)` ，强制类型转换

- `tf.reduce_min(张量名)` `tf.reduce_max(张量名)` ，计算张量维度上的最值

- axis：在一个二位张量或数组中，可以通过axis控制执行维度。
  - axis=0 代表跨第一个维度（跨行，经度，纵向操作，down）
  - axis=1 代表跨第二个维度（跨列，纬度，横向操作，across）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf2.png'  width="80%" height="80%"/>

- `tf.Variable()` ，标记为“可训练”。被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练的参数。
  
- 如：`w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))`
  
- 数学运算：
  - 四则运算：`tf.add` ，`tf.subtract` ，`tf.multiply` ，`tf.divide`，参数为两个张量，必须维度相同。
  - 指数运算：`tf.square` ，`tf.pow` ，`tf.sqrt`
  - 矩阵乘法：`tf.matmul`

- `tf.data.Dataset.from_tensor_slices((输入特征，标签))`，把特征值和标签配对。Numpy和Tensor格式都可用该语句读入数据

  - ```python
    features = tf.constant([12, 23, 10, 17])
    labels = tf.constant([0, 1, 1, 0])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    ```

- 在with结构中，使用GradientTape，实现某个函数对指定参数的求导运算。with结构记录计算过程，gradient求出张量的梯度

  - ```python
    with tf.GradientTape() as tape:
    	w = tf.Variable(tf.constant(3.0))
    	loss = tf.pow(w, 2)
    grad = tape.gradient(loss, w)
    print(grad)
    ```

  - 梯度是 2*w=6.0，运行结果：`tf.Tensor(6.0, shape=(), dtype=float32)`

- `enumerate()`，索引，返回索引和元素：

  - ```python
    seq = ['one', 'two', 'three']
    for i, element in enumerate(seq):
    	print(i, element)
    ```

  - 运行结果：

    ```markdown
    0 one
    1 two
    2 three
    ```

- `tf.one_hot(带转换数据, depth=几分类)`，转换成one-hot编码：

  - ```python
    calsses = 3
    labels = tf.constant([1, 0 ,2])
    output = tf.one_hot(labels, depth=classes)
    print(output)
    ```

  - 运行结果：

    ```markdown
    ([[0. 1. 0.]
      [1. 0. 0.]
      [0. 0. 1.]], shape=(3, 3), dtype=float32)
    ```

- `tf.nn.softmax(张量)`，$Softmax(y_i)=\frac{e^{y_i}}{\sum_{j=0}^ne^{y_i}}$，使输出符合概率分布

  - ```python
    y = tf.constant([1.01, 2.01, -0.66])
    y_pro = tf.nn.softmax(y)
    print(y_pro)
    ```

  - 运行结果：

    ```markdown
    tf.Tensor([0.22598174 0.69583046 0.0481878 ], shape=(3,), dtype=float32)
    ```

- `w.assign_sub(w要自减的值) `，参数自更新

  - ```python
    w = tf.Variable(4)
    w.assign_sub(1)
    print(w)
    ```

  - 运行结果：

    `<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>`

- `tf.argmax(张量名, axis=操作轴)`，返回张量沿指定维度最大值的索引



### 2 优化方法

#### 2.1 一些函数

- `tf.where(条件语句, A, B)`，条件为真返回A，条件为假返回B

  - ```python
    a = tf.constant([1, 2, 3, 1, 1])
    b = tf.constant([0, 1, 3, 4, 5])
    c = tf.where(tf.greater(a, b), a, b)
    print(c)
    ```

  - 运行结果：

    `tf.Tensor([1, 2, 3, 4, 5], shape=(5,), dtype=int32)`

- `np.random.RandomState.rand(维度)`，返回指定维度的 [0, 1] 的随机数。维度为空则返回一个标量

- `np.vstack(数组1, 数组2)`，将两个数组按垂直方向叠加

- 构建网格坐标点：

  - `np.mgrid[起始值:结束值:步长, 起始值:结束值:步长, ...]`

  - `x.ravel()`，把x变为一维数组，把变量x拉直

  - `np.c_[数组1, 数组2, ...]`，使返回的间隔数值点配对

  - ```python
    x, y = np.mgrid[1:3:1, 2:4:0.5]
    grid = np.c_[x.ravel(), y.ravel()]
    print("x:", x)
    print("y:", y)
    print("grid:", grid)
    ```

  - 运行结果：

    ```markdown
    x:[[1. 1. 1. 1.]
       [2. 2. 2. 2.]]
    y:[[2.  2.5 3.  3.5]
       [2.  2.5 3.  3.5]]
    grid:
     [[1.  2. ]
      [1.  2.5]
      [1.  3. ]
      [1.  3.5]
      [2.  2. ]
      [2.  2.5]
      [2.  3. ]
      [2.  3.5]]
    ```

#### 2. 2 指数衰减学习率

```
LR_BASE = 0.2
LR_DECAY = 0.99
LR_STEP = 1
for epoch in range(epoch):
lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
```

#### 2.3 激活函数

- `tf.nn.sigmoid(x)`，sigmoid的导数为 [0, 0.25] 的小数，多层神经网络链式求导时，如果连续相乘会造成梯度消失
- `tf.math.tanh(x)`，输出是0均值了，但依然存在梯度消失、幂运算问题
- `tf.nn.relu(x)`，解决了梯度消失问题，并且计算速度快；输出非0均值，收敛慢，并且存在dead relu问题
  - dead relu是经过relu函数的负数特征过多导致的，可以合理参数初始化、设置小的学习率
- `tf.nn.leaky_relu(x)`，解决了dead relu问题

建议：首选relu函数；学习率设置较小值；输入特征标准化；初始化参数中心化

#### 2.4 损失函数

- 均方误差mse：$MSE(y\_,y) = \frac{\sum_{i=1}^n(y-y\_)^2}{n}$ 
  - `loss_mse = tf.reduce_mean(tf.square(y_ - y))` 

- 交叉熵损失函数ce：$H(y\_, y) = -\Sigma y\_*lny$ 
  - `loss_ce = tf.losses.categorical_crossentropy(y_, y)` 
  - softmax与交叉熵结合，输出先进行softmax，再计算y与y_的交叉熵损失函数
    - `tf.nn.softmax_cross_entropy_with_logits(y_, y)` 

#### 2.5 缓解过拟合

- 欠拟合的解决方法：增加模型复杂度，增加特征，增加参数；减小正则化系数
- 过拟合的解决方法：数据清洗；增大数据集；增大正则化参数

正则化：

- `loss = loss(y与y_)+ REGULARIZER *loss(w)​`

  - $loss_{L1}(w) = \sum_i|w_i|$ ，L1正则化会让参数变为0，减少参数数量，降低复杂度
  - $loss_{L2}(w) = \sum_i|w_i^2|$ ，L2正则化会使参数接近0但不为0，降低复杂度

  - ```python
    loss_mse = tf.reduce_mean(tf.square(y_train - y))
    loss_regularization = []
    loss_regularization.append(tf.nn.l2_loss(w1))
    loss_regularizaiton.append(tf.nn.l2_loss(w2))
    # 求和，例：
    # x = tf.constant(([1, 1, 1], [1, 1, 1]))
    # tf.reduce_sum(x)
    # >>>6
    loss_regularization = tf.reduce_sum(loss_regularization)
    loss = loss_mse + 0.03 * loss_regularization
    ```

#### 2.6 优化器

待优化参数$w$，损失函数$loss$，学习率$lr$，每次迭代一个$batch$，$t$表示当前$batch$迭代的总次数：

1. 计算 $t$ 时刻损失函数关于参数的梯度 $g_t = \nabla loss = \frac{\partial loss}{\partial w_t}$ 
2. 计算 $t$ 时刻一阶动量 $m_t$ 和二阶动量 $V_t$ 
3. 计算 $t$ 时刻下降梯度 $\eta_t = lr \sdot m_t/\sqrt{V_t} $ 
4. 计算 $t+1$ 时刻参数 $w_{t+1} = w_t - \eta_t = w_t - lr \sdot m_t / \sqrt{V_t}$

##### 几种优化器

SGD：随机梯度下降（无momentum）

- $m_t = g_t \qquad V_t = 1$ 
- $w_{t+1} = w_t - lr \sdot \frac{\partial loss}{\partial w_t} $ 

- ```python
  w1.assign_sub(lr * grads[0])
  b1.assign_sub(lr * grads[1])
  ```

SGDM：含momentum的SGD

- $m_t = \beta \sdot m_{t-1}+(1-\beta)\sdot g_t \qquad V_t = 1$ 

- $w_{t+1} = w_t - lr \sdot (\beta \sdot m_{t-1}+(1-\beta)\sdot g_t) $ 

- ```python
  m_w = beta * m_w + (1 - beta) * grads[0]
  m_b = beta * m_b + (1 - beta) * grads[1]
  w1.assign_sub(lr * m_w)
  b1.assign_sub(lr * m_b)
  ```

Adagrad：在SGD（无momentum）的基础上增加二阶动量

- $m_t = g_t \qquad V_t = \sum^t_{\tau=1}g_\tau^2$ 

- $w_{t+1} = w_t - lr \sdot g_t/(\sqrt{\sum^t_{\tau=1}g_\tau^2})$ 

- ```python
  v_w, v_b = 0, 0
  # adagrad
  v_w += tf.square(grads[0])
  v_b += tf.square(grads[1])
  w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
  b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))
  ```

RMSProp：在SGD（无momentum）的基础上增加二阶动量

- $m_t = g_t \qquad V_t = \beta \sdot V_{t-1}+(1-\beta) \sdot g_t^2$ 

- $w_{t+1} = w_t - lr \sdot g_t/(\sqrt{\beta \sdot V_{t-1}+(1-\beta) \sdot g_t^2})$ 

- ```python
  v_w, v_b = 0, 0
  beta = 0.9
  # rmsprop
  v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
  v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
  w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
  b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))
  ```

Adam：同时结合SGDM一阶动量和RMSProp二阶动量

- $m_t = \beta_1 \sdot m_{t-1}+(1-\beta_1)\sdot g_t \qquad V_t = \beta_2 \sdot V_{step-1}+(1-\beta_2) \sdot g_t^2$ 

- 修正一阶动量的偏差 $\widehat{m_t} = \frac{m_t}{1-\beta_1^t}$ 

- 修正二阶动量的偏差 $\widehat{V_t} = \frac{V_t}{1-\beta_2^t}$ 

- $w_{t+1} = w_t - lr \sdot \frac{m_t}{1-\beta_1^t}/\sqrt{\frac{V_t}{1-\beta_2^t}}$ 

- ```python
  m_w, m_b = 0, 0
  v_w, v_b = 0, 0
  beta1, beta2 = 0.9, 0.999
  delta_w, delta_b = 0, 0
  global_step = 0
  # adam
  m_w = beta1 * m_w + (1 - beta1) * grads[0]
  m_b = beta1 * m_b + (1 - beta1) * grads[1]
  v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
  v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])
  
  m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
  m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
  v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
  v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))
  
  w1.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction))
  b1.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction))
  ```







