# 3B1B的神经网络视频

（主要是一些概念的直观理解。）



学习的含义：找到特定的 $\omega$ 和 $b$ ，使代价函数最小化。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\3b1b1.png'  width="80%" height="80%"/>



反向传播：计算单个训练样本想怎样修改 $\omega$ 和 $b$ 。不仅是每个参数应该变大还是变小，还包括这些变化的比例是多大，才能最快下降梯度函数。一个真正的梯度下降过程要对所有的训练样本求平均，但计算太慢，就先把所有的样本分到minibatch中，，计算一个minibatch来作为梯度下降的一步，最终会收敛到局部最优点。

为了使 $a_{i+1}$ 的某个输出增大，可以

- 增大 $b$ 
- 增大 $\omega$ ：增加上层活跃的神经元的权重更好。依据对应权重大小，对激活值做出成比例的改变。
- 改变 $a_i$



反向传播的链式法则

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\3b1b2.png'  width="80%" height="80%"/>



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\3b1b3.png'  width="80%" height="80%"/>





# 胡浩基老师NN、CNN部分





# 吴恩达deeplearning.ai网课

看的网易云课堂做的字幕版本，可惜右上方有水印，有些地方有遮挡。



## 第一课 神经网络和深度学习（Neural Networks and Deep Learning）

### 第一周：深度学习引言（Introduction to Deep Learning）

#### 1.1 欢迎

关于课程安排



#### 1.2 什么是神经网络？

神经网络可以当作一个函数。通过数据集计算从x到y的精准映射函数，然后对于新的数据x，给出预测的结果y。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\1.png'  width="80%" height="80%"/>



#### 1.3 神经网络的监督学习(Supervised Learning with Neural Networks)

监督学习：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\2.png' width="80%" height="80%"/>

对于图片，使用CNN。对于序列信息（音频、语言信息等）使用RNN。

结构化数据：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\3.png' width="80%" height="80%"/>

神经网络能帮助计算机理解无结构化数据。



#### 1.4 为什么神经网络会流行？

数据和计算规模的进展。现在获得了很大的数据量、计算了很复杂的网络。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\4.png' width="80%" height="80%"/>

其他原因：算法的改进，比如从sigmoid函数到relu函数



#### 1.5~1.6 课程安排

略





### 第二周：神经网络的编程基础（Basics of Neural Network Programming）

#### 2.1 二分类(Binary Classification)

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\6.png' width="80%" height="80%"/>

数据集按列组成矩阵：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\5.png' width="80%" height="80%"/>

X.shape = (n<sub>x</sub>, m)

y.shape = (1, m)



#### 2.2 逻辑回归(Logistic Regression)

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\7.png' width="80%" height="80%"/>

在神经网络中，将 $b$ 和 $w$ 分开表示，不采用逻辑回归那样组合成 $\theta$ 的形式。



#### 2.3 逻辑回归的代价函数

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\8.png' width="80%" height="80%"/>



#### 2.4 梯度下降（Gradient Descent）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\9.png' width="80%" height="80%"/>



#### 2.5~2.6 导数

略

#### 2.7 计算图（Computation Graph）

计算图表示从左向右的计算过程。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\10.png' width="80%" height="80%"/>



#### 2.8 计算图导数

根据计算图，从右到左计算函数 J 的导数。（链式求导）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\11.png' width="80%" height="80%"/>



#### 2.9 逻辑回归的梯度下降

用计算图理解逻辑回归的梯度下降。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\12.png' width="80%" height="80%"/>



#### 2.10 梯度下降的例子

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\13.png' width="80%" height="80%"/>

dw<sub>1</sub>、 dw<sub>2</sub>、db 作为累加器。数据集循环后，J、 dw<sub>1</sub>、 dw<sub>2</sub>、db 除以样本个数。

一次梯度下降有两层循环，外层循环遍历所有数据样本（m个），内层循环遍历所有特征 w（n个）。



在深度学习中，数据量很大，**为了不用显式的for循环，使用向量化**。



#### 2.11 向量化(Vectorization)

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\14.png' width="80%" height="80%"/>

向量化是很有必要的。在上图1000000维向量相乘运算中，使用向量化比使用for循环节省300倍的时间。



#### 2.12 更多的向量化例子

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\15.png' width="80%" height="80%"/>



#### 2.13 向量化逻辑回归

向量化逻辑回归的正向传播：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\16.png' width="80%" height="80%"/>

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

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\17.png' width="80%" height="80%"/>

dZ = A - Y

db = np.sum(dZ) / m

dw =  np.dot(X, dZ.T) / m

> X.shape = (n_x, m)
>
> dZ.shape = (1, m)
>
> dw.shape = (n_x, 1)



一次梯度下降：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\18.png' width="80%" height="80%"/>





#### 2.15 Python中的广播机制（Broadcasting in Python）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\19.png' width="80%" height="80%"/>



#### 2.16 关于 Python与numpy向量的使用

` a = np.random.randn(5)`

> a.shape = (5, )

这是numpy的特殊格式"rank 1 array"，`a.T` 操作仍然得到这种格式的数组。

在神经网络编程中，避免使用这种秩为1的数组。

用 ` a = np.random.randn(5, 1)` 作为替代。此时就可以用 `np.dot(a, a.T)` 得到一个矩阵了。

也可以用 `assert` 或 `reshape` 修改为矩阵格式。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\20.png' width="80%" height="80%"/>



#### 2.17 Jupyter/iPython Notebooks快速入门

略

#### 2.18* 逻辑回归损失函数详解

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\21.png' width="80%" height="80%"/>



在整个数据集上的情况：假设样本是独立同分布，可以累乘，做最大似然估计使这个式子取最大值。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\22.png' width="80%" height="80%"/>





### 第三周：浅层神经网络（Shallow Neural Networks）

#### 3.1 神经网络概述

正向传播，计算损失函数 $L$ ；反向传播，计算梯度下降需要的 $dW$、$db$ 。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\23.png' width="80%" height="80%"/>



#### 3.2 神经网络的表示

$ a^{[i]}$ 表示第 $i$ 层的激活值，$a^{[0]}=X$

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\24.png' width="80%" height="80%"/>



#### 3.3 计算一个神经网络的输出

向量化的前向传播

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\25.png' width="80%" height="80%"/>



#### 3.4 多样本向量化

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\26.png' width="80%" height="80%"/>

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\27.png' width="80%" height="80%"/>

$X,Z,A$ 都是按列组合起来的矩阵。



#### 3.5 向量化实现的解释

$Z^{[1]}$ 的每一列都是一个训练样本 $X_i$ 经过 $W^{[1]}$ 计算而来的。

当处理多个训练样本时，$X$ 是列向量拼起来的形式，则 $Z$ 也是 $X$ 的每一列的计算结果。

把 $b$ 加上也一样，可能用到广播机制。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\28.png' width="80%" height="80%"/>

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\29.png' width="80%" height="80%"/>



#### 3.6 激活函数（Activation functions）

不同层的激活函数可以不一样。在隐藏层中，tanh函数效果比sigmoid好；但在输出层，二分类任务用sigmoid比较好，因为输出是0~1.

tanh和sigmoid函数的问题是：当x很大或很小，函数的梯度约为0，会拖慢梯度下降。因此在隐藏层用ReLU函数更好。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\30.png' width="80%" height="80%"/>



#### 3.7 为什么需要非线性激活函数？

如果不用非线性激活函数，神经网络的输出就是 $X$ 的线性组合。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\31.png' width="80%" height="80%"/>



#### 3.8 激活函数的导数

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\30.png' width="80%" height="80%"/>

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

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\32.png' width="80%" height="80%"/>

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

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\33.png' width="80%" height="80%"/>

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\34.png' width="80%" height="80%"/>



#### 3.11 随机初始化（Random Initialization）

不能用全0初始化神经网络，这会导致神经元的对称性和反向传播失效。

随机初始化：用很小的随机数初始化 $W$ ，用0初始化 $b$ .

如果 $W$ 的值太大，$z$ 会落在tanh函数和sigmoid函数平缓的部分，梯度下降会很慢。如果完全没有用到这两个函数就没有影响。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\35.png' width="80%" height="80%"/>

训练浅层神经网络，0.01是可用的；当训练深层神经网络，要使用其他的常数，在之后讲。





### 第四周：深层神经网络

#### 4.1 深层神经网络

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\36.png' width="80%" height="80%"/>

$L$：层数，从0开始计数。

$n^{[l]}$：$l$ 层的神经元个数。

图中 $L=4$ ，$n^{[L]}=1$，$n^{[1]} = n^{[2]} =5$ 。



#### 4.2 前向传播

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\37.png' width="80%" height="80%"/>

基本过程：

- $z^{[l]} = w^{[l]} a^{[l-1]} +b^{[l]}$ 
- $a^{[l]} = g(z^{[l]})$ 

向量化见图右下方。

在前向传播的实现过程中，需要使用显示的for循环，来遍历从输入层到输出层。



#### 4.3 检查矩阵的维数

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\38.png' width="80%" height="80%"/>

同样本排列成一列（如 $(x_1,x_2)^T$、$(z_1,z_2)^T$），不同的样本m纵向组合起来（如 $(A[0],A[1])$、$(Z[1],Z[2])$）。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\39.png' width="80%" height="80%"/>

在向量化的场合，python的broadcasting机制把 $b[1]$ 维度 $(n^{[1]},1)$ 扩展成 $(n^{[1]},m)$。



#### 4.4 为什么使用深层表示？

神经网络可以不用很大，但深层有好处。

在直觉层面理解，深层神经网络能组合从简单到复杂的信息。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\40.png' width="80%" height="80%"/>



另一种直觉理解，从电路角度，用小规模但深层的电路结构，可以进行复杂的计算；但用浅层的电路模型，要用指数级增长的运算单元才能实现相同的功能。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\41.png' width="80%" height="80%"/>



#### 4.5 搭建深层神经网络块

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\42.png' width="80%" height="80%"/>

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\43.png' width="80%" height="80%"/>



#### 4.6 前向和反向传播

前向传播的实现：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\44.png' width="80%" height="80%"/>

反向传播的实现：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\45.png' width="80%" height="80%"/>



#### 4.7 参数VS超参数（Parameters vs Hyperparameters）

Parameters: W, b

Hyperparameters:

- learning rate(α), #iterations, #hidden layers(L), #hidden units(n), choice of activation function.
- momentum, mini-batch size, regularization parameters, ...

尝试不同的超参数值，找到合适的值。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\46.png' width="80%" height="80%"/>



#### 4.8 深度学习和人类大脑的关联性

目前对人脑的认识没有达到建立数学模型的程度。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\47.png' width="80%" height="80%"/>







## 第二课 改善深层神经网络：超参数调试、正则化以及优化(Improving Deep Neural Networks : Hyperparameter tuning, Regularization and Optimization)

### 第一周：深度学习的实用层面

> 深度学习的应用方法。数据集的划分；偏差/方差；通过正则化来防止过拟合（包括L1L2，dropout，其他方法如数据增强和提前停止）；输入归一化；通过合理的权重初始化来避免梯度消失和梯度爆炸；进行梯度检验确保梯度下降算法正确运行。最后一节讲了以上方法的实践经验。

#### 1.1 训练/验证/测试集（Train / Dev / Test sets）

在训练集进行训练，根据在验证集上的得分选择最好的模型，在测试集上进行评估。

在数据集很大的情况下，可以把验证集、测试集划分得少一点。在百万条数据的情况下，甚至可以划分99.5%/0.25%/0.25%。



- 注意1：**保证验证集和测试集的数据来自同一分布**。

  如：训练集是网站上比较精美、清晰的图片；验证集、训练集是用户随手拍的图片。

- 注意2：不做测试集也可以。如果不需要对最终的神经网络做无偏评估，也可以不设置测试集。



#### 1.2 偏差/方差（Bias /Variance）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\48.png' width="80%" height="80%"/>

前提：基本error很低；验证集和测试集来自同一分布。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\49.png' width="80%" height="80%"/>

训练集的error要跟基本error比，基本error通常是人工识别的error。



#### 1.3 先后顺序

按步骤确认：

1. high bias？ 增大网络规模、训练更长时间、（修改网络结构）

2. high variance？ 获得更多数据、正则化、（修改网络结构）
3. 完成，获得 low bias & variance 的模型。



在现在深度学习、大数据的环境中，可以做到在减小bias或variance的过程中，不对另一方产生过多不良影响。我们不用太过关注如何 tradeoff。



#### 1.4 正则化（Regularization）

逻辑回归的正则化：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\50.png' width="80%" height="80%"/>

如果用的是L1正则化，W最终会是稀疏的，也就是W向量中有很多0。

现在更倾向于L2正则化。

$\lambda$ 也是一个需要调整的超参数。为了防止与python的关键字重复，在代码中一般写作lambd。 



神经网络的正则化：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\51.png' width="80%" height="80%"/>

由于历史原因，不叫矩阵的L2正则化，而是叫 frobenius norm。

在反向传播过程中，正则化项求导后加在 $dW$ 的后面，让梯度下降的幅度大一些。也被称为 weight decay 。



#### 1.5 为什么正则化有利于预防过拟合呢？

从直观上理解，正则化项降低了 $W$ 的值，也就是降低了一些神经元的作用，简化了模型，让模型从过拟合向欠拟合发展。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\52.png' width="80%" height="80%"/>



第二种直观理解方法：$W$ 值变小，$z$ 集中在激活函数的线性部分，则模型的每一层都相当于线性变换，模型不适用于复杂的决策，降低了过拟合程度。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\53.png' width="80%" height="80%"/>

如果实施了带正则化项的损失函数，当使用梯度下降法时，为了调试梯度下降，要使用这个新定义的损失函数，否则损失函数可能不会再所有的调幅范围内都单调递减。



#### 1.6 dropout 正则化

对每个训练样本，遍历神经网络的每一层，并设置消除神经网络中节点的概率，消除一些节点，得到一个更小规模的神经网络，训练这个精简后的网络。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\54.png' width="80%" height="80%"/>

 一种实现方法：inverted dropout（反向随机失活）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\55.png' width="80%" height="80%"/>

用不等式给d赋值为true或false，跟a相乘让a的一部分值失效。

有一个 `a/=deep_prob` 操作， 修正或弥补丢掉的一部分数据，让a的期望值不变。

在测试阶段，不使用dropout。



#### 1.7 理解 dropout

直观上理解，dropout让神经元不依赖于某一个特征，而让权重更加分散。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\56.png' width="80%" height="80%"/>

如果更担心在某些层有过拟合，就把某些层的keep-prob设置得低一些。缺点是在验证集上调参工作量增大。



dropout本质上是一种正则化方法，用来防止过拟合。在计算机视觉问题中，输入的像素很多，以至于没有足够的数据，经常一直处于过拟合情况。因此dropout在CV应用的比较频繁。在其他领域，如果没有过拟合问题就不必使用。

dropout一大缺点就是代价函数 $J$ 不再被明确定义。每次迭代都随机保留神经元，很难对每次的反向传播梯度下降进行复查。也就失去了绘制递减的代价函数图像的工具。通常先关闭dropout，运行代码确保代价函数单调递减，再开启dropout。



#### 1.8 其他正则化方法

data augment

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\57.png' width="80%" height="80%"/>



early stopping

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\58.png' width="80%" height="80%"/>

建立模型的两个过程：其一是让 $J(w, b)$ 取到最小值，手段包括梯度下降等；其二是防止过拟合，又称为orthogonalization，手段包括正则化等。early stopping 的缺点是破坏了这两个过程相互的独立性。提前结束训练过程，也就是打断了第一个过程。

如果使用L2正则化，就避免了这个缺点，随之而来的是 $\lambda$ 的调参工作量，而不是只进行一次梯度下降就可以找到early stopping的位置。



#### 1.9 归一化输入（Normalizing inputs）

第一步：零均值化；第二步：方差归一化。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\59.png' width="80%" height="80%"/>

注意：在训练集和测试集上要用相同的 $\mu,\sigma$ 。

这样做的原因：让优化变快。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\60.png' width="80%" height="80%"/>



#### 1.10 梯度消失/梯度爆炸（Vanishing / Exploding gradients）

activations以指数级增长或下降，给梯度下降造成困难。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\61.png' width="80%" height="80%"/>

以图中简化 $b$ 、$w$ 全部是对角矩阵的神经网络为例：$w$ 比单位矩阵大一点，激活值以指数级增长；w 比单位矩阵小一点，激活值以指数级减小。



#### 1.11 神经网络的权重初始化

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\62.png' width="80%" height="80%"/>

通过给 $W$ 设置合理的初始值（不能比1大/小太多），避免梯度消失和梯度爆炸。

以图中去掉 $b$ 的单个神经元为例，最合理的方式是设置 $w$ 接近 $\frac{1}{n}$ 。

因此进行这样的初始化：$W^{[l]} = np.random.randn(shape)*np.sqrt(\frac{2}{n^{[l-1]]}})$

当用ReLU函数，是 $\sqrt{\frac{2}{n^{[l-1]]}}}$ ；当用tanh函数，是 $\sqrt{\frac{1}{n^{[l-1]]}}}$；也有人用 $\sqrt{\frac{2}{n^{[l-1]]}+n^{[l]}}}$ 。



#### 1.12 梯度的数值近似

在实施反向传播时，进行gradient checking，可以确保反向传播正在正确进行。

用 $\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}\approx g(\theta)$ 近似计算 $\theta$ 的梯度 $g(\theta)$。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\63.png' width="80%" height="80%"/>



#### 1.13 梯度检验（Gradient checking）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\64.png' width="80%" height="80%"/>

把所有层的$w,b$ 组合成矩阵 $\theta$，所有层的$dW,db$ 组合成矩阵 $d\theta$ 。我们需要验证：$d\theta$ 是 $\theta$ 的梯度。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\65.png' width="80%" height="80%"/>

计算近似梯度：

$$ d\theta_{approx}[i] =\frac{J(\theta_1, \theta_2,...,\theta_i+\epsilon,...)-J(\theta_1, \theta_2,...,\theta_i-\epsilon,...)}{2\epsilon} \approx d\theta[i] = \frac{\partial J}{\partial \theta_i}$$

$$check: \frac{||d\theta_{approx}-d\theta||_2}{||d\theta_{approx}||_2+||d\theta||_2} \approx 10^{-7}$$

如果 $\approx10^{-5}$，检查向量，确保没有一项误差过大，确保没有bug；如果 $\approx10^{-3}$，需要小心有bug。可以检查哪一项的导数计算结果和估计值偏差很大，并反推求导过程，检查bug。



#### 1.14 应用梯度检验的注意事项

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\65-add.png' width="80%" height="80%"/>

- 不要在训练过程中使用梯度检验，只用于调试。
- 如果梯度检验失败，检查哪一项的导数计算结果和估计值偏差很大，确定bug位置，比如在某一层的求导结果跟估计值差很大。
- 梯度检验的过程中，如果使用了正则化，要记住计算中应包括正则化项。
- **梯度检验不能与dropout一起使用**。dropout让我们难以计算 $J$ 。可以先把 keep_prob 设置为1，验证梯度下降是正确的；再开启dropout.
- 几乎不会出现的情况：随机初始化 $w,b$ 接近0，梯度下降的实施是正确的，但在运行梯度下降时，$w,b$ 变大，可能只有在 $w,b$ 接近0时，梯度下降才是正确的，但 $w,b$ 变大时它变得越来越不准确。做法（基本不用）：在随机初始化过程中进行梯度检验，然后再训练网络，如果随机初始化值比较小，$w,b$ 会有一段时间远离0 ；反复训练网络之后再重新进行梯度检验。（开始做一下梯度检验，训练后再进行一次梯度检验，保证正确。）



### 第二周：优化算法 (Optimization algorithms)

让梯度下降加速的优化方法。包括mini-batch梯度下降、momentum/RMSprop/Adam算法（和需要了解的指数加权平均、偏差修正基础）、学习率衰减。最后一节讲了局部最优问题。

#### 2.1 Mini-batch 梯度下降

batch梯度下降：在整个数据集上进行梯度下降。

mini-batch梯度下降：把整个数据集划分为若干个mini-batch，在每个mini-batch上进行一次梯度下降。



实现过程：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\66.png' width="80%" height="80%"/>

前向传播、求平均损失 --> 反向传播、梯度下降。在每个mini-batch中进行这样的操作。

完整的遍历一次训练集称为 1 epoch。mini-batch可以在 1 epoch 中完成多次梯度下降。



#### 2.2 理解Mini-batch 梯度下降

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\67.png' width="80%" height="80%"/>

mini-batch会让梯度下降有噪声，但最终也会收敛到比较小的水平。



mini-batch梯度下降的优势：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\68.png' width="80%" height="80%"/>

- mini-batch size = m，即 batch 梯度下降：步长大，噪声少。单次迭代耗时长。

- mini-batch size = 1，即 随机 梯度下降：步长小，噪声多，永远不收敛（在最小值附近波动。失去向量化方法带来的计算加速。

- mini-batch梯度下降：既能对样本进行向量化，又能快速迭代。



选择 mini-batch size 的注意事项：

- 样本集小（<2000），直接用batch梯度下降。

- 一般把 mini-batch size 设置为2的次方数。
- 确保mini-batch内的数据 $(X^{\{t\}},Y^{\{t\}})$ 符合 CPU/GPU 的内存。

- 这也是一个 hyperparameter ，需要多次尝试，找到让梯度下降最高效的取值。

还有比梯度下降和mini-batch梯度下降都要高效得多的算法，在后面讲。



#### <span id = "2.2.3">2.3 指数加权平均（Exponentially weighted averages）</span>

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\69.png' width="80%" height="80%"/>

在统计学中叫指数加权移动平均值。

$$V_t = \beta V_{t-1} + (1-\beta) \theta_t$$ 

$V_t$ 可以看作在 $\frac{1}{1-\beta}$ 天中，温度$\theta$ 的平均值。

通过调整参数 $\beta$ ，获得不同的效果。

- $\beta$ 大，平均的样本多，曲线平滑但有偏移。（图中绿色线是50天的平均值）
- $\beta$ 小，平均的样本少，曲线更拟合，但噪声大。（图中黄色线是2天的平均值，红色线是10天的平均值）



#### 2.4 理解指数加权平均

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\70.png' width="80%" height="80%"/>

把算式展开，是sum(每天的温度×指数衰减系数)的形式。（右上方两个图中对应值相乘）

$$V_{100} = 0.1\times\theta_{100}+0.1\times 0.9\times\theta_{99}+0.1\times 0.9\times 0.9\times\theta_{98}+ ...$$

所有系数加起来近似＝1。

有 $(1-\epsilon)^{\frac{1}{\epsilon}}\approx \frac{1}{e}$ ，在此时权重系数衰减的下降幅度很大。因此可以近似认为，今天的 $V$ 的值是取了前 $\frac{1}{\epsilon} = \frac{1}{1-\beta}$ 天的平均值。如图中的 $V$ 是取了 $\frac{1}{0.1}=10$ 天的温度平均值。



实现：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\71.png' width="80%" height="80%"/>

在实现上，只需要存储单个变量 $V$ 并且不断更新即可。$V$ 近似了平均值，省去了使用滑动窗口求和求精确平均值所需的存储空间。

在之后的章节中，需要计算多个变量的平均值，使用指数加权平均是一个好的近似计算方法。



#### 2.5 指数加权平均的偏差修正（Bias correction in exponentially weighted average）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\72.png' width="80%" height="80%"/>

以 $\beta=0.98$ 为例，在实际实现上，会得到紫线而不是绿线。以为初始化 $V=0$ ，前几项会很小。（见左下算式）

使用偏差修正，让平均值近似计算更加准确：用 $\frac{V_t}{1-\beta^t}$ 代替 $V_t$ 。在刚开始 $t$ 较小时，$\frac{V_t}{1-\beta^t}$ 求的是 $\theta$ 的加权平均数（右下表达式）；$t$ 变大时，$\beta^t$ 接近 0。

偏差修正能在早期获得更好的估计，但也可以选择熬过初始时期，不使用偏差修正。



#### 2.6 momentum梯度下降

计算梯度的指数加权平均数，加速梯度下降。这个方法好于普通梯度下降。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\73.png' width="80%" height="80%"/>

梯度下降的波动，要求我们不能用很大的学习率。在纵轴上，我们希望学习慢一点；在横轴上，希望学习快一点。

平均了这些梯度之后，会发现纵轴上的摆动平均值接近 0（图中红箭头），可以采用大一些的学习率了。

一个直观上的理解：小球从碗状函数像底部滚动，微分项 $dw,db$ 是加速度，momentum项 $V_{dw},V_{db}$ 是速度，球加速向底部滚动，而 $\beta$ 相当于摩擦力，让小球不会无限加速。不像梯度下降法每一步都独立于之前的步骤，现在小球可以向下滚，获得动量（momentum）。



实现：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\74.png' width="80%" height="80%"/>

$\beta$ 的常用值是 0.9，即平均前十次迭代的梯度。同时也可以不使用偏差修正 $\frac{V_t}{1-\beta^t}$ ，因为10次后已经可以正常近似了。

也有资料将 $(1-\beta)$ 忽略，使用右边的式子，这两者在效果上是相同的，只是会影响到 $\alpha$ 的最佳值。老师认为左边的计算方法更符合直觉，因为如果要调整超参数 $\beta$，就会影响到 $V_{dw}$ 和 $V_{db}$ ，也许还要修改 $\alpha$。



#### 2.7 RMSprop-root mean square prop

另一种消除梯度下降的摆动，加快梯度下降的方法。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\75.png' width="80%" height="80%"/>

当在某个方向波动大（如图中举例 $db$ ，在梯度下降减去一个分母较大的数 $b:= b-\alpha \frac{db}{\sqrt{db}}$，让梯度下降的幅度减小。在某个方向梯度下降幅度小（如图中举例 $dw$ ，在梯度下降减去一个较小的数 $w:= w-\alpha \frac{dw}{\sqrt{dw}}$，让梯度下降的幅度增大。

其他：为了跟momentum结合起来，将RMSprop的超参数命名为 $\beta_2$ ；防止除以0，在分母加上很小的数 $\epsilon = 10^{-8}$ 。



#### 2.8 Adam优化算法

把momentum和RMSprop组合起来。在不同的模型上都有很好的效果，有很广泛的应用。

Adam算法需要进行偏差修正。

momentum：$w=w-\alpha V_{dw}$ 

RMSprop：$w = w - \alpha \frac{dw}{\sqrt{S_{dw}+\epsilon}}$ 

Adam：$w = w - \alpha \frac{V_{dw}}{\sqrt{S_{dw}+\epsilon}}$ 

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\76.png' width="80%" height="80%"/>



几个超参数，当应用adam算法时，$\beta_1,\beta_2,\epsilon$ 常常都是用缺省值，$\alpha$ 需要实验确定。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\77.png' width="80%" height="80%"/>



#### 2.9 学习率衰减（Learning rate decay）

随时间慢慢减小学习率。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\78.png' width="80%" height="80%"/>

一种方法：

$$\alpha = \frac{1}{1+decay\_rate \times epoch\_num}\alpha_{init}$$

decay\_rate 是需要调整的超参数。 

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\79.png' width="80%" height="80%"/>



其他几种方法：指数衰减、除以epoch_num的开方、离散衰减等。也有看着模型训练过程，然后手动进行衰减的方法。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\80.png' width="80%" height="80%"/>



#### 2.10 局部最优问题

在维数很高的情况下，更多的情况是收敛到鞍形部位（鞍点，图右方），而不是局部最优点（图左方）。在鞍点，一些方向的曲线向下弯曲，一些方向的曲线向上弯曲。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\81.png' width="80%" height="80%"/>

在鞍上称为plateaus问题，这段时间训练得比较慢，使用momentum等算法可以加速此过程。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\82.png' width="80%" height="80%"/>





### 第三周：超参数调试，批正则化和程序框架

调参的基本规则和方法；batch norm让学习算法运行速度更快；softmax回归；深度学习框架。

#### 3.1 调参规则

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\83.png' width="80%" height="80%"/>

调参重要性排序：红 > 黄 > 紫。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\84.png' width="80%" height="80%"/>

在深度学习中，不要用网格取值进行实验（图左）。

应该随机取超参数的值并进行实验（图右）。因为不知道哪个超参数是更重要的，需要探究重要的超参数的更多潜在值。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\85.png' width="80%" height="80%"/>

使用从粗略到精细（coarse to fine）的策略。在表现好的区域上进行更密集的取值尝试



#### 3.2 合适的参数取值范围

有些超参数，可以在合理的范围内，在**线性轴**上，做随机均匀取值。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\86.png' width="80%" height="80%"/>



学习率等超参数，更合适的方法是在**对数轴**上均匀随机取值。

```python
r = -4 * np.random.randn()
alpha = exp(10, r)
```

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\87.png' width="80%" height="80%"/>





在 $1-\beta$ 取值，而不是在 $\beta$ 取值。因为在 $\beta$ 越接近 1，平均的样本个数有更大的变化，需要更密集的取值。所以在 $1-\beta$ 接近 0 时进行更密集的取值。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\88.png' width="80%" height="80%"/>



#### 3.3 超参数训练的实践：Pandas vs. Caviar

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\89.png' width="80%" height="80%"/>

在不同领域的参数设置可能有相似的部分，多了解其他工作；多进行尝试。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\90.png' width="80%" height="80%"/>

一种方法：在训练中照看（babysitting）模型，比如进行学习率的调整。

另一种方法：同时训练超参数取值不同的多个模型。



#### 3.4 激活函数的归一化/单一隐藏层上的批归一化（Batch normalization）

batch normalization 会使参数搜索问题变得很容易，使神经网络对超参数的选择更加稳定。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\91.png' width="80%" height="80%"/>

对逻辑回归、神经网络的输入归一化而言，进行输入特征值的归一化是有效的。如图的上半部分，对 $x_1,x_2,x_3$ 进行归一化对 $w,b$ 的训练有帮助。

对深层的模型，能否对 $a^{[i]}$ 进行归一化，改进 $w^{[i+1]},b^{[i+1]}$ 的训练？

在实践中，我们不对 $a^{[i]}$ 做归一化，而是对 $z^{[i]}$ 做归一化。这一点一直有争论。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\92.png' width="80%" height="80%"/>

实现：

$$z_{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$$ 

$$\widetilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta$$ 

- $\gamma,\beta$ 是可以学习的参数（不是超参数）。如果 $\gamma=\sqrt{\sigma^2+\epsilon},\beta=\mu$，则 $\widetilde{z}^{(i)} = {z}^{(i)}$，batch normalization不起作用。 $\gamma$ 控制方差，$\beta$ 值控制均值。通过给它们赋值，可以构造含平均值和方差的隐藏单元值。

- 用  $\widetilde{z}^{(i)}$ 取代 ${z}^{(i)}$ ，参与神经网络的后续计算。
- 不一定要归一化到均值0的分布。可以归一化到均值不是0，方差大一点，符合sigmoid等激活函数的特性。

- batch normalization本质上是让隐藏单元值的均值和方差标准化，即 $z^{[i]}$ 有固定的均值和方差，由 $\gamma,\beta$ 两个参数控制。



#### 3.5 深度神经网络的批归一化

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\93.png' width="80%" height="80%"/>

batch norm是发生在计算 $z$ 和 $a$ 之间的。给神经网络添加了新的参数 $\gamma,\beta$ 。（注意，跟momentum等优化算法的超参数 $\beta$ 区分。这两者的论文都使用 $\beta$ 作为参数的名称。）

使用优化算法（如梯度下降或Adam等），对这些参数 $w,b,\gamma,\beta$ 进行更新。

在深度学习框架中，可以用一行代码完成batch norm的操作，无需自己实现。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\94.png' width="80%" height="80%"/>

batch norm通常和训练集的mini-batch一起使用。在每一个mini-batch上，做一次梯度下降。

在上一段中，提到对参数 $w,b,\gamma,\beta$ 进行更新。但实际的计算步骤为：

- 先计算 $z^{[l]}  = w^{[l]}a^{[l-1]}+b^{[l]}$；
- 然后对 $z^{[l]}$ 进行归一化计算 $z_{norm}^{[l]}$，在此过程中会减去均值， **$b^{[l]}$ 这个加上去的参数是无效的。**
- 用 $\widetilde{z}^{[l]} = \gamma^{[l]}z_{norm}^{[l]} + \beta^{[l]}$ 进行后续计算。**形式上， $\beta$ 代替了参数 $b$ 。**

实际计算步骤：

- $z^{[l]}  = w^{[l]}a^{[l-1]}$ 
- 计算 $z_{norm}^{[l]}$ 
-  $\widetilde{z}^{[l]} = \gamma^{[l]}z_{norm}^{[l]} + \beta^{[l]}$ 

此外，注意参数的维度：$z,b,\beta,\gamma$ 维度都是 $(n^{[l]}, 1)$



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\95.png' width="80%" height="80%"/>

实现：

对于每一个mini-batch：在前向传播的过程中，在每个隐藏层使用 batch norm；反向传播计算梯度；进行梯度下降或使用其他优化算法。



#### 3.6 为什么Batch Norm有用？

第一个原因：跟逻辑回归类似，让所有特征归一到同一尺度，加速梯度下降的过程。 

第二个原因：让权重比网络更滞后或更深层，让数值更稳定。第10层的权重比第1层的权重更robust。在之前层的权重发生改变时，$z$ 会发生变化，但batch norm保证了 $z$  的均值和方差保持不变。因此限制了在前层的参数更新对数值分布的影响。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\96.png' width="80%" height="80%"/>

第三个原因：batch norm有一点正则化的效果。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\97.png' width="80%" height="80%"/>



#### 3.7 测试时的Batch Norm

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\98.png' width="80%" height="80%"/>

在训练过程中，batch norm一次作用在一个mini-batch上，求这个mini-batch上的均值和方差（图左）；在评估阶段，batch norm只作用在单个测试样本上，虽然可以在整个测试集上计算 $\mu,\sigma^2$，但在实际操作中，通常使用指数加权平均（图右）。

追踪训练过程中每个mini-batch的 $\mu,\sigma^2$ 的值，然后使用之前求温度 $\theta_1,\theta_2,\theta_3$ 的指数加权平均的方法（见[第二课第二周第三节](#2.2.3)，求 $\mu,\sigma^2$ 的近似值，然后用于下一步计算 $z_{norm} = \frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}$。

实际上，不管用什么样的估计方法，整套过程都是比较robust的。

当使用深度学习框架时，通常会有默认的估算 $\mu,\sigma^2$ 的方法。

 

#### 3.8 Softmax 回归 

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\99.png' width="80%" height="80%"/>

多分类问题中，预测一组相加为1的概率值作为神经网络的输出层。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\100.png' width="80%" height="80%"/>

使用softmax激活函数进行从权值到概率的转换。$t^i=e^{z^i}$，$a^i = \frac{t^i}{\sum t}$ 

ReLU和Sigmoid函数输入一个实数，输出一个实数；而softmax函数因为要对所有的输出进行归一化（计算概率），需要输入一个向量，输出一个向量。



直观的softmax分类的例子：神经网络只有一层softmax层。神经元有三个，就分成3类，每类之间都是线性决策边界。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\101.png' width="80%" height="80%"/>



#### 3.9 训练一个 Softmax 分类器

softmax跟hardmax相对，把最大值更温和地转换成一个概率，而不是全部改为0和1.

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\102.png' width="80%" height="80%"/>



**训练-损失函数**：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\103.png' width="80%" height="80%"/>

单个训练样本的损失函数：$$L(\hat{y}, y) = -\sum^C_{j=1}y_jlog\hat{y_j}$$

在真实情况中，$y_j$ 只有一个为1，其余都为0，因此损失函数是 $-log\hat{y_i}$ ，损失函数试图让 $y=1$ 对应的 $\hat{y}$ 尽量地大。这也是最大似然估计的一种形式。

整个训练集的损失函数：$$J(w,b) = \frac{1}{m}\sum^m_{i=1} L(\hat{y}, y)$$ 

$\hat{y},y$ 的维度都是 $(4, m) $。



**训练-梯度下降**：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\104.png' width="80%" height="80%"/>

梯度：$$dz^{[l]} = \hat{y} - y$$ 

在深度学习框架中，主要精力放在将前向传播做好。通常框架自己会弄明白怎样反向传播。



#### 3.10 深度学习框架

#### 3.11 TensorFlow



## 第三课 结构化机器学习项目 (Structuring Machine Learning Projects)























## 第四课 卷积神经网络（Convolutional Neural Networks）

















## 第五课 序列模型(Sequence Models)







### 









