## Tensorflow 2.0 笔记

整理自北大曹建老师网课，见[此处](https://www.bilibili.com/video/BV1B7411L7Qt)

### 1  基本概念

#### 1.1 张量 Tensor

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



#### 1.2 常用函数

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





### 3 神经网络搭建八股

#### 3.1 使用Sequential搭建神经网络

tf.keras搭建神经网络六步法（使用Sequential）：

1. import

2. train, test：划分数据集

3. model = tf.keras.models.Sequential：逐层描述网络结构，前向传播

4. model.compile：配置训练方法，选择优化器、损失函数、评测指标

5. model.fit：执行训练过程

6. model.summary：打印网络的结构和参数统计

##### Sequential

`model = tf.keras.models.Sequential([网络结构])` 

- 拉直层：`tf.keras.layers.Flatten()` 
- 全连接层：`tf.keras.layers.Dense(神经元个数, activation="激活函数", kernel_regularizer=哪种正则化)` 
  - 激活函数：`relu`, `softmax`, `sigmoid`, `tanh` 
  - 正则化：`tf.keras.regularizers.l1()`, `tf.keras.regularizers.l2()`
- 卷积层：`tf.keras.layers.Conv2D(fitlers=卷积核个数, kernel_size=卷积核尺寸, strides=步长, padding="valid" or "same") `
- LSTM层：`tf.keras.layers.LSTM()` 

##### compile

`model.compile(optimizer=优化器, loss=损失函数, metrics=["准确率"])` 

- optimizer 可选：
  - `'sgd'` or `tf.keras.optimizers.SGD(lr=学习率, momentum=动量)` 
  - `'adagrad'` or `tf.keras.optimizers.Adagrad(lr=学习率)` 
  - `'adadelta'` or `tf.keras.optimizers.Adadelta(lr=学习率)` 
  - `'adam'` or `tf.keras.optimizers.Adam(lr=学习率, beta_1=0.9, beta_2=0.999)` 
- loss 可选： 
  - `'mse'` or `tf.keras.losses.MeanSquaredError()` 
  - `'sparse_categorical_crossentropy'` or `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)` ，神经网络预测前没有经过概率分布则是True，经过概率分布就是False 
- metrics 可选：
  - `accuracy`：y_ 和 y 都是数值，如 y_=[1]  y=[1]
  - `categorical_accuracy`：y_ 和 y 都是独热码，如 y_=[0, 1, 0]  y=[0.256, 0.695, 0.048]
  - `sparse_categorical_accuracy`：y_ 是数值，y 是独热码，如 y_=[1]  y=[0.256, 0.695, 0.048]

##### fit

```python
model.fit(训练集特征, 训练集标签, 
batch_size=, epochs=, 
validation_data=(测试集特征, 测试集标签) or validation_split=从训练集划分多少比例给测试集, validation_freq=多少epoch测试一次)
```

##### summary

`model.summary()`

##### 举例：鸢尾花识别

```python
# 1. import
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 2. train, test
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 3. Sequential
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# 4. compile
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['sparse_categorical_accuracy'])

# 5. fit
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# 6. summary
model.summary()
```



#### 3.2  使用class搭建神经网络

Sequential支持搭建上层输入是下层输出的神经网络，如果有跳连，可以用class搭建。

tf.keras搭建神经网络六步法（使用class）：

1. import

2. train, test

3. **class MyModel(Model) model=MyModel**

4. model.compile

5. model.fit

6. model.summary

```python
class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		定义网络结构块
    def call(self, x):
    	调用网络结构块，实现前向传播
    	return y
model = MyModel()
```

以鸢尾花分类的网络为例：

 ```python
# 1. import
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

# 2. train, test
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 3. class
class IrisModel(Model):
	def __init__(self):
		super(IrisModel, self).__init__()
		self.d1 = Dense(3, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())
        
    def call(self, x):
    	y = self.d1(x)
    	return y
    
model = IrisModel()

# 4. compile
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['sparse_categorical_accuracy'])

# 5. fit
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# 6. summary
model.summary()
 ```



#### 3.3 MNIST 数据集

MNIST数据集有 7 万张 28*28 像素的手写数字，其中 6 万张用于训练，1 万张用于测试。

##### Sequential

```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 把[0, 255]变为[0, 1]，输入特征值小更易于神经网络吸收

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
```

##### class

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        
	def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y

model = MnistModel()

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
```





### 4 八股扩展

自制数据集、数据增强、断点续训（实时存取模型）、参数提取（把参数存入文本）、acc/loss可视化、应用程序。

#### 4.1 自制数据集

读写文件、建立数据集的操作，详见代码。

#### 4.2 数据增强

```python
image_gen_train = ImageDataGenerator(
	rescale = 所有数据将乘以该值, 
    rotation_range = 随机旋转角度, 
    width_shift_range = 随机宽度偏移量, 
    height_shift_range = 随机高度偏移量, 
    horizontal_flip = 是否随机水平翻转, 
    zoom_range = 随即缩放 
)
image_gen_train.fit(x_train)
```

x_train要求是四维数据，需要进行reshape，最后一个是通道数量：`x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)` 

model.fit步骤变为：`model.fit(image_gen_train.flow(x_train, y_train, batch_size=32))` 

数据增强的效果需要在实际应用程序中体会。



#### 4.3 断点续训

##### 读取模型

`load_weights(路径) `

保存模型时，会自动生成.index索引表文件。如果路径中已经有保存好的模型，就直接加载模型参数：

```python
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
	print('---load the model---')
	model.load_weights(checkpoint_save_path)
```

##### 保存模型

```python
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path, # 文件存储路径
    save_weights_only=True, # 是否只保留模型参数
    save_best_only=True)	# 是否只保留最优结果

history = model.fit(x_train, y_train, batch_size=32, epochs=5, 
                   	validation_data=(x_data, y_test), validation_freq=1, 
                    callbacks=[cp_callback])
# 模型训练时加入 callbacks 选项，记录到 history 中
```



#### 4.4 参数提取

`model.trainable_variables` 返回模型中可训练参数。

`np.set_printoptions(threshold=超过多少省略显示)` 

```python
print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
```



#### 4.5 acc/loss 曲线

在 model.fit 执行训练过程的同时，同步记录了：

- 训练集loss：`loss ` 
- 测试集loss：`val_loss ` 
- 训练集准确率：`sparse_categorical_accuracy ` 
- 测试集准确率：`val_sparse_categorical_accuracy` 

可用 history.history 提取出来。

```python
history = model.fit(x_train, y_train, batch_size=32, epochs=5, 
                   	validation_data=(x_data, y_test), validation_freq=1, 
                    callbacks=[cp_callback])
```

```python
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```



#### 4.6 模型应用程序


`predict(输入特征, batch_size=)` 返回前向传播计算结果。

分以下几步：

- 复现模型的前向传播Sequential
- 加载模型参数
- 进行预测



附手写数字识别任务的代码：

**A. 模型训练和保存**

```python
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 把[0, 255]变为[0, 1]，输入特征值小更易于神经网络吸收

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 查看有没有可以加载的模型
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)


history = model.fit(x_train, y_train, batch_size=32, epochs=20,
          validation_data=(x_test, y_test), validation_freq=1,
          callbacks=[cp_callback])  # 通过调用回调函数，保存模型

model.summary()


# 参数写入txt
print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()


# acc/loss可视化
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```



**B. 模型加载和预测**

```python
from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

# 复现前向传播
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

# 加载模型
model.load_weights(model_save_path)


preNum = int(input("执行几次识别任务:"))

for i in range(preNum):
    image_path = input("图片文件名:")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)  # 转化为28*28
    img_arr = np.array(img.convert('L'))  # 转化为灰度图片

    # 白底黑字，转换成黑底白字，并降噪
    # img_arr = 255 - img_arr
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0  # 归一化

    # 神经网络训练时，都是按照batch送入网络，所以给img_arr前面添加一个维度
    # img_arr:(28, 28)
    # x_predict:(1, 28, 28)
    x_predict = img_arr[tf.newaxis, ...]

    result = model.predict(x_predict)   # 前向传播
    pred = tf.argmax(result, axis=1)    # 输出最大概率值的索引
    print('\n')
    tf.print("预测数字：", pred, "\n")
```





### 5 卷积神经网络

实际问题中，图片尺寸大、通道多，直接输入全连接层不现实，需要进行特征的提取，再输入到全连接层。卷积计算是提取特征的常用、有效的方法。

#### 5.1 卷积概念

##### 卷积计算

输入特征图的channel数决定了卷积核的channel数；卷积核的个数决定输出特征图的channel数。

如果觉得某层特征提取能力差，可以在这一层多用几个卷积核。

可以根据下图记住参数的个数。每个格子是一个w，每个卷积核有一个b。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf3.png'  width="80%" height="80%"/>

当卷积核是多channel的，也是执行全部相乘、求和的计算过程。

当有多个卷积核，把每个卷积核得到的特征图叠加起来，形成多channel的输出特征图。



##### 感受野

感受野的概念如图：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf4.png'  width="80%" height="80%"/>

用两层 3\*3 卷积核，和一层 5\*5 卷积核，最终得到的单个输出像素的感受野都是 5，我们说**这两种卷积方法的特征提取能力是一样的**。

此时，考虑它们的带训练参数个数和计算量，在不同的卷积方法中进行选择：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf5.png'  width="80%" height="80%"/>

这也是现在神经网络经常使用两层 3\*3 卷积核替换一层 5\*5 卷积核的原因。



#### 5.2 tensorflow卷积层的实现

##### A. padading-全零填充

全零填充在输入图片周围补0，使卷积计算的输出保持跟输入相同的尺寸。

- `padding='SAME'`：使用全零填充

- `padding='VALID'`：不使用全零填充



##### B. Conv2D-描述卷积层

```python
tf.keras.layers.Conv2D(
    filters=卷积核个数
    kernel_size=卷积核尺寸, 
    strides=滑动步长, 
    padding="same" or "valid", 
    activation="relu" or "softmax" or "tanh" or "sigmoid",  # 如果有BN，此处不写
    input_shape=(高, 宽, 通道数) # 输入特征图的维度，可以省略
)
```

kernel_size、strides等，如果横纵相同，就写一个长整数；否则写 `(纵向, 横向)` 

卷积层三种不同的传参方法：

```python
model = tf.keras.models.Sequential([
    Conv2D(6, 5, padding='valid', activation='sigmoid'), 
    MaxPool2D(2, 2), 
    
    Conv2D(6, (5, 5), padding='valid', activation='sigmoid')
    MaxPool2D(2, (2, 2)), 
    
	Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='sigmoid')
    MaxPool2D(pool_size=(2, 2), strides=2), 
    
    Flatten(), 
    Dense(10, activation='softmax')
])
```



##### C. BN-批标准化

神经网络对0附近的数据更敏感，梯度更大。

- 标准化：使数据符合0均值，1标准差的分布
- 批标准化：对一batch的数据做标准化处理

$$ {H_i^k}^\prime  = \frac{H_i^k - \mu^k_{batch}}{\sigma^k_{batch}} $$

**BN常用在卷积操作和激活操作之间。** 

均值、标准差都是针对第 k 个卷积核生成的 batch 张（同channel）图中的所有像素点。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf6.png'  width="80%" height="80%"/>

这种简单的标准化，使输入数据集中在激活函数的线性区域。

因此在BN操作中，为每个像素 $H$ 引入**两个可训练参数**，缩放因子 $\gamma$ 和偏移因子 $\beta$，标准正态分布后的输入数据通过这两个因子，优化了特征数据分布的宽窄、偏移量（如图），**保证了网络的非线性表达力**。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\tf7.png'  width="80%" height="80%"/>

```python
model = tf.keras.models.Sequential([
	Conv2D(fitlers=6, kernel_size=(5, 5), padding='same'), 
	BatchNormalization(), 
	Activation('relu'), 
	MaxPool2D(pool_size=(2, 2), strides=2, padding='same'), 
	Dropout(0.2)
])
```



##### D. Pooling-池化

用于减少特征的数量。

- 最大池化：提取图片纹理
- 均值池化：保留背景特征

```python
tf.keras.layers.MaxPool2D(
	pool_size=池化核尺寸, 
    strides=池化步长,  # 默认跟 pool_size 相等
    padding='valid' or 'same' # 是否使用全零填充的池化，不改变原图片的尺寸
)
tf.keras.layers.AveragePooling2D(
	pool_size=池化核尺寸, 
    strides=池化步长, 
    padding='valid' or 'same' 
)
```

例子见之前的代码。 



##### E. Dropout-舍弃

为了防止过拟合，按照一定概率，把一部分神经元从社交网络中暂时舍弃。使用神经网络时，被舍弃的神经元恢复连接。

`tf.keras.layers.Dropout(舍弃概率)` 

例子见之前的代码。 



##### 总结

卷积神经网络：借助卷积核提取特征，然后送入全连接网络。

特征提取分四步：

- Conv2D
- BN
- Activation
- Pooling

Q：卷积是什么？

A：卷积就是特征提取器，就是CBAPD。

```python
model = tf.keras.models.Sequential([
	Conv2D(fitlers=6, kernel_size=(5, 5), padding='same'), 
	BatchNormalization(), 
	Activation('relu'), 
	MaxPool2D(pool_size=(2, 2), strides=2, padding='same'), 
	Dropout(0.2)
])
```



#### 5.3 