# 3B1B的神经网络视频

（主要是一些概念的直观理解。）



学习：找到特定的 $\omega$ 和 $b$ ，使代价函数最小化。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\3b1b1.png'  width="80%" height="80%"/>



反向传播：计算单个训练样本想怎样修改 $\omega$ 和 $b$ 。不仅是每个参数应该变大还是变小，还包括这些变化的比例是多大，才能最快下降梯度函数。一个真正的梯度下降过程要对所有的训练样本求平均，但计算太慢，就先把所有的样本分到minibatch中，，计算一个minibatch来作为梯度下降的一步，最终会收敛到局部最优点。

为了使 $a_{i+1}$ 的某个输出增大，可以

- 增大 $b$ 
- 增大 $\omega$ ：增加上层活跃的神经元的权重更好。依据对应权重大小，对激活值做出成比例的改变。
- 改变 $a_i$



反向传播的链式法则

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\3b1b2.png'  width="80%" height="80%"/>



<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\3b1b3.png'  width="80%" height="80%"/>





# 胡浩基老师NN、CNN部分





# 吴恩达deeplearning.ai网课

看的网易云课堂做的字幕版本，可惜右上方有水印，有些地方有遮挡。



## 第一课 神经网络和深度学习（Neural Networks and Deep Learning）

### 第一周：深度学习引言（Introduction to Deep Learning）

#### 1.1 欢迎

关于课程安排



#### 1.2 什么是神经网络？

神经网络可以当作一个函数。通过数据集计算从x到y的精准映射函数，然后对于新的数据x，给出预测的结果y。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\1.png'  width="80%" height="80%"/>



#### 1.3 神经网络的监督学习(Supervised Learning with Neural Networks)

监督学习：

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\2.png' width="80%" height="80%"/>

对于图片，使用CNN。对于序列信息（音频、语言信息等）使用RNN。

结构化数据：

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\3.png' width="80%" height="80%"/>

神经网络能帮助计算机理解无结构化数据。



#### 1.4 为什么神经网络会流行？

数据和计算规模的进展。现在获得了很大的数据量、计算了很复杂的网络。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\4.png' width="80%" height="80%"/>

其他原因：算法的改进，比如从sigmoid函数到relu函数



#### 1.5~1.6 课程安排

略





### 第二周：神经网络的编程基础（Basics of Neural Network Programming）

#### 2.1 二分类(Binary Classification)

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\6.png' width="80%" height="80%"/>

数据集按列组成矩阵：

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\5.png' width="80%" height="80%"/>

X.shape = (n<sub>x</sub>, m)

y.shape = (1, m)



#### 2.2 逻辑回归(Logistic Regression)

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\7.png' width="80%" height="80%"/>

在神经网络中，将 $b$ 和 $w$ 分开表示，不采用逻辑回归那样组合成 $\theta$ 的形式。



#### 2.3 逻辑回归的代价函数

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\8.png' width="80%" height="80%"/>



#### 2.4 梯度下降（Gradient Descent）

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\9.png' width="80%" height="80%"/>



#### 2.5~2.6 导数

略

#### 2.7 计算图（Computation Graph）

计算图表示从左向右的计算过程。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\10.png' width="80%" height="80%"/>



#### 2.8 计算图导数

根据计算图，从右到左计算函数 J 的导数。（链式求导）

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\11.png' width="80%" height="80%"/>



#### 2.9 逻辑回归的梯度下降

用计算图理解逻辑回归的梯度下降。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\12.png' width="80%" height="80%"/>



#### 2.10 梯度下降的例子

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\13.png' width="80%" height="80%"/>

dw<sub>1</sub>、 dw<sub>2</sub>、db 作为累加器。数据集循环后，J、 dw<sub>1</sub>、 dw<sub>2</sub>、db 除以样本个数。

一次梯度下降有两层循环，外层循环遍历所有数据样本（m个），内层循环遍历所有特征 w（n个）。



在深度学习中，数据量很大，**为了不用显式的for循环，使用向量化**。



#### 2.11 向量化(Vectorization)

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\14.png' width="80%" height="80%"/>

向量化是很有必要的。在上图1000000维向量相乘运算中，使用向量化比使用for循环节省300倍的时间。



#### 2.12 更多的向量化例子

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\15.png' width="80%" height="80%"/>



#### 2.13 向量化逻辑回归

向量化逻辑回归的正向传播：

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\16.png' width="80%" height="80%"/>

Z = np.dot(w.T, X) + b

> w.shape  =(n_x, 1)，每个特征对应一个w，列向量
>
> X.shape = (n_x, m)
>
> z.shape = (1, m)
>
> b本来是一个实数，python的broadcasting机制在相加时，把b扩展为 （1， m) 维的行向量。



#### 2.14 向量化逻辑回归的梯度计算

向量化逻辑回归的反向传播：

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\17.png' width="80%" height="80%"/>

dZ = A - Y

db = np.sum(dZ) / m

dw =  np.dot(X, dZ.T) / m

> X.shape = (n_x, m)
>
> dZ.shape = (1, m)
>
> dw.shape = (n_x, 1)



一次梯度下降：

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\18.png' width="80%" height="80%"/>





#### 2.15 Python中的广播机制（Broadcasting in Python）

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\19.png' width="80%" height="80%"/>



#### 2.16 关于 Python与numpy向量的使用

` a = np.random.randn(5)`

> a.shape = (5, )

这是numpy的特殊格式"rank 1 array"，`a.T` 操作仍然得到这种格式的数组。

在神经网络编程中，避免使用这种秩为1的数组。

用 ` a = np.random.randn(5, 1)` 作为替代。此时就可以用 `np.dot(a, a.T)` 得到一个矩阵了。

也可以用 `assert` 或 `reshape` 修改为矩阵格式。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\20.png' width="80%" height="80%"/>



#### 2.17 Jupyter/iPython Notebooks快速入门

略

#### 2.18* 逻辑回归损失函数详解

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\21.png' width="80%" height="80%"/>



在整个数据集上的情况：假设样本是独立同分布，可以累乘，做最大似然估计使这个式子取最大值。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\22.png' width="80%" height="80%"/>





### 第三周：浅层神经网络（Shallow Neural Networks）

#### 3.1 神经网络概述

正向传播，计算损失函数 $L$ ；反向传播，计算梯度下降需要的 $dW$、$db$ 。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\23.png' width="80%" height="80%"/>



#### 3.2 神经网络的表示

$ a^{[i]}$ 表示第 $i$ 层的激活值，$a^{[0]}=X$

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\24.png' width="80%" height="80%"/>



#### 3.3 计算一个神经网络的输出

向量化的前向传播

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\25.png' width="80%" height="80%"/>



#### 3.4 多样本向量化

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\26.png' width="80%" height="80%"/>

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\27.png' width="80%" height="80%"/>

$X,Z,A$ 都是按列组合起来的矩阵。



#### 3.5 向量化实现的解释

$Z^{[1]}$ 的每一列都是一个训练样本 $X_i$ 经过 $W^{[1]}$ 计算而来的。

当处理多个训练样本时，$X$ 是列向量拼起来的形式，则 $Z$ 也是 $X$ 的每一列的计算结果。

把 $b$ 加上也一样，可能用到广播机制。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\28.png' width="80%" height="80%"/>

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\29.png' width="80%" height="80%"/>



#### 3.6 激活函数（Activation functions）

不同层的激活函数可以不一样。在隐藏层中，tanh函数效果比sigmoid好；但在输出层，二分类任务用sigmoid比较好，因为输出是0~1.

tanh和sigmoid函数的问题是：当x很大或很小，函数的梯度约为0，会拖慢梯度下降。因此在隐藏层用ReLU函数更好。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\30.png' width="80%" height="80%"/>



#### 3.7 为什么需要非线性激活函数？

如果不用非线性激活函数，神经网络的输出就是 $X$ 的线性组合。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\31.png' width="80%" height="80%"/>



#### 3.8 激活函数的导数

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\30.png' width="80%" height="80%"/>

Sigmoid：

$$a = g(z) = \frac{1}{1+e^{-z}}$$

$$g'(z)=\frac{d}{dz}g(z) = g(z)(1-g(z))=a(1-a)$$

tanh：

$$g(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$$

$$g'(z)=\frac{d}{dz}g(z) = 1-(g(z))^2$$

ReLU：在0处可以指定导数的值

$$g(z)=max(0, z)$$

$$ g'(z)= \begin{cases}  0,&\text{if } z<0 \\ 1,&\text{if } z≥0 \end{cases} $$

Leaky ReLU：

$$g(z)=max(0.01z, z)$$

$$ g'(z)= \begin{cases}  0.01,&\text{if } z<0 \\ 1,&\text{if } z≥0 \end{cases} $$



#### 3.9 神经网络的梯度下降

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\32.png' width="80%" height="80%"/>

前向传播过程：

> $Z^{[1]} = W^{[1]}X + b^{[1]}$
>
> $A^{[1]} = g^{[1]}(Z^{[1]})$
>
> $Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$
>
> $A^{[2]} = g^{[2]}(Z^{[2]})$

反向传播过程：

>$dZ^{[2]}=A^{[2]}-Y$
>
>$dW^{[2]}=\frac{1}{m} dZ^{[2]} (A^{[1]} )^T$
>
>$db^{[2]} = \frac{1}{m}np.sum(dZ^{[2]}, axis=1, keepdims=ture)$ 
>
>$dZ^{[1]} =( W^{[2]})^T  dZ^{[2]} * g^{'[1]}(Z^{[1]})$
>
>$dW^{[1]}=\frac{1}{m} dZ^{[1]} X^T$
>
>$db^{[1]} = \frac{1}{m}np.sum(dZ^{[1]}, axis=1, keepdims=ture)$



#### 3.10（选修）直观理解反向传播（Backpropagation intuition）

想一想矩阵的维度。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\33.png' width="80%" height="80%"/>

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\34.png' width="80%" height="80%"/>



#### 3.11 随机初始化（Random Initialization）

不能用全0初始化神经网络，这会导致神经元的对称性和反向传播失效。

随机初始化：用很小的随机数初始化 $W$ ，用0初始化 $b$ .

如果 $W$ 的值太大，$z$ 会落在tanh函数和sigmoid函数平缓的部分，梯度下降会很慢。如果完全没有用到这两个函数就没有影响。

<img src='G:\github-repos\MyPostImage\ml-notes-img\nndl\35.png' width="80%" height="80%"/>

训练浅层神经网络，0.01是可用的；当训练深层神经网络，要使用其他的常数，在之后讲。





### 第四周：深层神经网络

#### 4.1 深层神经网络

#### 4.2 前向传播和反向传播（Forward and backward propagation）

#### 4.3 深层网络中的前向和反向传播

#### 4.4 检查矩阵的维数

#### 4.5 为什么使用深层表示？

#### 4.6 搭建神经网络块

#### 4.7 参数VS超参数（Parameters vs Hyperparameters）

#### 4.8 深度学习和人类大脑的关联性





## 第二课 改善深层神经网络：超参数调试、正则化以及优化(Improving Deep Neural Networks:Hyperparameter tuning, Regularization and Optimization)



## 第三课 结构化机器学习项目 (Structuring Machine Learning Projects)



## 第四课 卷积神经网络（Convolutional Neural Networks）





## 第五课 序列模型(Sequence Models)







### 









