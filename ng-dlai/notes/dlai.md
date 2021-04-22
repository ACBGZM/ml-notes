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

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\83.png' width="50%" height="50%"/>

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

同样的思想：对深层的模型，能否对 $a^{[i]}$ 进行归一化，改进 $w^{[i+1]},b^{[i+1]}$ 的训练？

在实践中，我们不对 $a^{[i]}$ 做归一化，而是对 $z^{[i]}$ 做归一化。这一点一直有争论。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\92.png' width="80%" height="80%"/>

实现：

$$z_{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$$ 

$$\widetilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta$$ 

- $\gamma,\beta$ 是可以学习的参数（不是超参数）。如果 $\gamma=\sqrt{\sigma^2+\epsilon},\beta=\mu$，则 $\widetilde{z}^{(i)} = {z}^{(i)}$，batch normalization不起作用。 $\gamma$ 控制方差，$\beta$ 值控制均值。通过给它们赋值，可以构造含平均值和方差的隐藏单元值。

- 用  $\widetilde{z}^{(i)}$ 取代 ${z}^{(i)}$ ，参与神经网络的后续计算。
- 不一定非要归一化成均值为0的分布。可以归一化到均值不是0，方差大一点，符合sigmoid等激活函数的特性。

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

现存的框架；选择框架的标准。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\105.png' width="80%" height="80%"/>



#### 3.11 TensorFlow

在tensorflow中定义损失函数cost，可以理解为tensorflow会建立起一个计算图，来自动完成后续的反向传播。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\106.png' width="80%" height="80%"/>

在框架中，可以只用一行代码修改很多工作，比如训练的方法是梯度下降还是Adam。这支持我们快速实现复杂的神经网络模型。







## 第三课 结构化机器学习项目 (Structuring Machine Learning Projects)























## 第四课 卷积神经网络（Convolutional Neural Networks）

### 第一周 卷积神经网络(Foundations of Convolutional Neural Networks)

卷积运算（padding、stride）和不同的卷积核；将卷积核叠加的三维卷积和单层卷积网络；卷积神经网络（CONV、POOL、FN）。

#### 1.1 计算机视觉（Computer vision）

计算机视觉的应用：图片分类，目标检测，风格迁移等。

计算机视觉的一个问题是数据量非常大。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\107.png' width="80%" height="80%"/>



#### 1.2 卷积运算-边缘检测为例（Edge detection）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\108.png' width="80%" height="80%"/>

在神经网络隐藏层中，不同层识别不同的信息。比如，浅层识别物体的边缘，深层识别人脸的部位，更深层识别整个人脸。以边缘检测为例，展示卷积计算的过程。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\109.png' width="80%" height="80%"/>

以中间矩阵的区域，在左边矩阵的每个对应区域，进行对应元素相乘，然后加起来，作为右边矩阵的一个值。

左边的矩阵理解为图片；中间的矩阵是过滤器（filter）或卷积核（kernel）；右边的矩阵可以理解为另一张图片。$*$ 是数学上的卷积运算符，但在python中， $*$ 也被重载做很多场合的乘法运算，所以在编程中使用其他函数，比如tensorflow中是 `tf.nn.conv2d`。

这也是纵向边缘的计算过程：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\110.png' width="80%" height="80%"/>

越大的值理解为颜色越浅。本例计算出的边界比较宽，是因为原图片相对来说非常小。



#### 1.3 更多边缘检测内容

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\111.png' width="80%" height="80%"/>

使用相同的filter，可以在输出图像中区分源图像从亮到暗&从暗到亮这两种变化。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\112.png' width="80%" height="80%"/>

不同的filter可以帮助我们找到垂直或水平的边缘。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\113.png' width="80%" height="80%"/>

也有相关工作提出更robust的filter取值，同时也可以不手动定义filter，而把这些数字当成参数，通过反向传播学习更好的filter（之后的内容）。

通过合理设置filter，不仅能检查水平、垂直的边缘，也可以检测任何角度的边缘。

通过把filter的所有数字设置成参数，并让计算机自动学习它们，我们发现：神经网络可以学习一些低级的特征，比如图片的边缘特征。构成这些运算的基础依然是卷积运算（convolution），使得反向传播算法可以学习任何所需的3×3 filter，并在整张图片上应用它。



#### 1.4 Padding

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\114.png' width="80%" height="80%"/>

使用 $f\times f$ 的卷积核，卷积 $n\times n$ 的源图像，得到 $(n-f+1)\times(n-f+1)$ 的新图象。

卷积的两个缺点：

- 卷积会让图片尺寸缩小。可能做几次之后图像就变得很小了。
- 边缘的像素参与的卷积运算很少，中间的像素用得很多。意味着卷积丢失了图像边缘的信息。

通过padding解决这两个问题：在图像周围再添加 $p$ 圈像素。

使用 $f\times f$ 的卷积核，卷积 $(n+p)\times (n+p)$ 的源图像，得到 $(n+2p-f+1)\times(n+2p-f+1)$ 的新图象。

如图 $p=1$：

- $8\times 8$ 的新图象经过卷积，得到 $6\times 6$ 的图像，尺寸没有变小。
- 边缘的像素参与的卷积运算更多了。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\115.png' width="80%" height="80%"/>

 关于padding多少：

- Valid convolution：不padding。 $(n\times n) * (f\times f) \longrightarrow (n-f+1)\times(n-f+1)$
- Same convolution：padding后得到的输出图像尺寸是源图像尺寸。 $(n\times n) * (f\times f) \longrightarrow n\times n$，$p=\frac{f-1}{2}$

在计算机视觉问题中，$f$ 一般是奇数。



#### 1.5 卷积步长（Strided convolutions）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\116.png' width="80%" height="80%"/>

padding p，stride s：

$$(n\times n) * (f\times f) \longrightarrow (\lfloor \frac{n+2p-f}{2} +1\rfloor)\times(\lfloor \frac{n+2p-f}{2} +1\rfloor)$$

惯例：不是整数就向下取整，超出边缘的卷积不进行计算。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\117.png' width="80%" height="80%"/>

数学中的convolution还要进行反转filter的操作，在机器学习中则不进行。机器学习的运算在数学中被称为cross-correlation，但在论文中我们延续convolution这一说法，要注意与数学环境中的convolution做区分。



#### 1.6 三维卷积（Convolutions over volumes）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\118.png' width="80%" height="80%"/>

源图像和filter的channel数量必须相同。最终得到一个二维输出。

将27个数对应相乘再求和，得到输出图像上的一个数。

通过不同的filter的参数选择，获得不同的特征检测器。如图，可以构建只关心红色通道的纵向边缘的filter；也可以构建不关心任何颜色，只关心纵向边缘的filter。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\119.png' width="80%" height="80%"/>

也可以使用多个filter。如图，将纵向边缘filter、横向边缘filter卷积而来的两张图片结合起来，得到 $4\times 4\times 2$ 的新图像。这种思想使我们可以检测很多个不同的特征，并且输出的通道数等于要检测的特征数，即filter的个数。



#### 1.7 单层卷积网络

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\120.png' width="80%" height="80%"/>

单层卷积网络的前向传播：

- 卷积运算。对应 $w^{[1]}a^{[0]}$。$w^{[1]}$是filter，$a^{[0]}$是源图像。

- 对得到的 $4\times 4$ 矩阵加一个权值（使用 broadcasting）。对应 $z^{[1]} = w^{[1]}a^{[0]} + b^{[1]}$。
- 进行非线性函数处理，如 ReLU，得到新的  $4\times 4$ 矩阵。对应 $a^{[1]} = g(z^{[1]})$ 。
- 多个filter，计算结果叠加起来，得到  $4\times 4 \times \#filters$ 矩阵



参数数量：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\121.png' width="80%" height="80%"/>

不管输入图片的尺寸有多大，参数的个数只跟filter有关。这是卷积神经网络的一个特性，可以避免过拟合。



符号总结：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\122.png' width="80%" height="80%"/>

每层输出图像的尺寸：

- $n_H^{[l]} = \lfloor \frac{n_H^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}} +1 \rfloor  $ 

- $n_W^{[l]} = \lfloor \frac{n_W^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}} +1 \rfloor $ 

每个filter的尺寸需要匹配上层输出图像的channel数量：

- $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$ 

所有的filter：

- $f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$

本层图像经过bias和非线性函数得到的activation尺寸：

- $a^{[l]}:n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$ 

一个mini-batch的所有activations：

- $A^{[l]}:m \times n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$ 

也不是所有人都用这一套标记法，有些人把channel的数量写在前面。



#### 1.8 简单的卷积神经网络示例

预测 $39\times 39 \times 3$ 的图像上是否有一只猫，设计以下卷积神经网络：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\123.png' width="80%" height="80%"/>

经过几步卷积后，获得 $7\times 7 \times 40$ 的特征图，将它们展开成 1960 长度的列向量，进行logistic或softmax回归，预测图片中是否有猫。

在卷积的过程中，有这样的趋势：图像的大小在减少，通道数量在增多。



选择超参数是一个问题，$f,s,p,\#filters$ 等。在之后的课程中会提供一些建议和指导。



卷积神经网络通常由三种layer组成：

- Convolution，卷积层，CONV
- Pooling，池化层，POOL
- Fully connectied，全连接层，FC



#### 1.9 池化层（Pooling layers）

使用池化层，来缩减模型的大小，提高计算速度，同时让所提取的特征robust。



max pooling：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\124.png' width="80%" height="80%"/>

- max pooling对每一个通道独立处理，不改变通道个数。

- 有两个超参数 $f,s$ ，不需要网络学习，手动设置后就不再改变。

- 可以直觉理解为：数字大意味着可能提取了某些特定特征。



average pooling：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\125.png' width="80%" height="80%"/>

跟max pooling差不多。

通常，max pooling更加常用；但有时，在很深的神经网络也会用到average pooling。（有时用，在下周讲）



池化层的超参数：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\126.png' width="80%" height="80%"/>

- 有常用的设置 $f=2,s=2$ ，意味着把图片长宽都缩小一半。
- 可以自己增加padding参数 $p$，但极少这样做。（有意外，在下周讲）

- $n_H \times n_W \times n_c \longrightarrow \lfloor \frac{n_H-f}{s}+1 \rfloor \times \lfloor \frac{n_H-f}{s}+1 \rfloor \times n_c$ ，**池化层不改变通道的个数**。
- **池化层没有需要训练的参数，只有超参数**。



#### 1.10 卷积神经网络示例（Convolutional neural network example）

手写数字识别（跟 LeNet-5 相似）：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\127.png' width="80%" height="80%"/>

有一种叫法是把 CONV+POOL 作为一层卷积，因为 POOL 层没有权重，只有超参数。在本例中同样将 CONV1+POOL1 作为 layer 1。

全连接层相当于单层普通神经网络，神经元全部相连，每条边有一个权值。

- 第一层：卷积+最大池化。参数是6个filters。

- 第二层：卷积+最大池化。参数是16个filters。
- 第三层：flatten后，400到120的全连接。参数是 $w,b$。
- 第四层：120到84的全连接。参数是 $w,b$。
- 输出：对84个神经元进行 softmax ，预测手写数字。

常见的模式：

- 图像尺寸逐渐变小，通道数量逐渐增多。
- 一个或多个卷积层后接一个池化层，重复几次，最后是几个全连接层，最终进行softmax等函数输出。



常规做法：尽量不要自己设置超参数，而是查看文献，使用别人在任务中效果很好的架构。（下周细讲）



卷积神经网络的一些细节：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\128.png' width="80%" height="80%"/>

- 参数
  - 池化层没有任何参数。
  - 卷积层参数数量较小，这点在之前提过。只跟filter有关，跟图片的尺寸无关。
    - 416 = 16channel * (5*5filter + 1bias)，每个filter有一个偏置。
  - 大多数参数存在于全连接层。
    - 48001 = 120 * 400 + 1bias，每层一个偏置，可以类比普通的神经网络。
- 激活值
  - 随着神经网络加深，激活值会逐渐变小。如果激活值下降太快，也会影响神经网络的表现。



卷积神经网络的重点是如何更好地组织卷积层、池化层、全连接层。这要求我们多阅读论文，了解别人的模型，得到自己的insight/intuation。下周将介绍一些表现良好的模型。



#### 1.11 为什么使用卷积？（Why convolutions?）

卷积神经网络为何有效？如何整合这些卷积？如何通过标注过的训练集进行卷积神经网络的训练？



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\129.png' width="80%" height="80%"/>

卷积神经网络相比只有全连接的普通神经网络的优势：参数共享和稀疏连接。

- 参数共享：filter的参数可以用于图片的任何区域，来提取特征。

- 稀疏链接：输出图像的每个像素仅与几个源图像的像素有关（不是全连接）。

这两点保证了**卷积神经网络可以用比较小的数据集进行训练，并且不容易过拟合**。

卷积神经网络善于捕捉平移不变（translation invariance），因为神经网络的卷积结构保证了，即使移动几个像素，图片依然具有非常相似的特征。



训练卷积神经网络的过程：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\130.png' width="80%" height="80%"/>

通过梯度下降或其他优化算法，优化参数，让损失函数 $J$ 降到最低。





### 第二周 深度卷积网络：实例探究(Deep convolutional models: case studies)

一些卷积神经网络的实例分析：Classic networks（LeNet-5，AlexNet，VGG），ResNet，Inception；1×1卷积



#### 2.1 为什么要进行实例探究？

好的网络架构可能在其他任务中也好用。



#### 2.2 经典网络

红笔写的是现在基本不用的技术。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\131.png' width="80%" height="80%"/>

在LeNet提出时，使用的这些技术现在已经基本被取代了：sigmoid和tanh激活函数；平均池化；valid 卷积；受限于计算能力，卷积的计算方法也很复杂。现在用的是：ReLU；最大池化；same卷积。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\132.png' width="80%" height="80%"/>

在AlexNet中，使用了ReLU、same卷积、max-pool、设置stride、softmax等新技术。

LeNet-5大约有60,000个参数；AlexNet有大约60,000,000个参数。

在AlexNet提出时，GPU的处理速度还比较慢，所以AlexNet采用了很复杂的方法在两个GPU上训练。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\133.png' width="80%" height="80%"/>

VGG-16网络没有很多超参数，专注于构建卷积层。16的意思是网络中有16层有权值的地方（2+2+3+3+3=13卷积层，3全连接层）。

- CONV = 3×3 filter, s=1, same
- MAX-POOL = 2×2, s=2

VGG-16 有约 138,000,000 个参数，但结构很规整，图像缩小的比例和channel增加的比例是有规律的。后面的VGG-19比这个模型更大，但这两个模型表现差不多。



#### 2.3 残差网络（Residual Networks (ResNets)）

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\134.png' width="80%" height="80%"/>

很深的神经网络难以训练，因为存在梯度消失和梯度爆炸的问题。使用ResNet，可以训练北京深层的神经网络。

每两层组成一个残差块：浅层的激活值通过short cut，直接输入到深层的非线性函数（如ReLU）中。

$$a^{[l+2]} = g(z^{[l+2]}+a^{[l]})$$ 



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\135.png' width="80%" height="80%"/>

论中将没有使用残差块的神经网络叫做Plain网络，在理论上层数越多，损失越小；但实际情况是，网络越深，在训练集上的误差会反弹。ResNet就会解决这一问题。



#### 2.4 残差网络为什么有用？

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\136.png' width="80%" height="80%"/>

如果让 $w,b$ 都为0，那么 $a^{[l+1]} = a{^{[l]}}$ ，学习恒等函数对残差块来说很简单。也就是说，虽然加上一个残差块（两层神经网络），效率也不逊色于更简单的神经网络。并且残差块添加的位置也不影响网络的表现。

在不伤害性能的基础上，如果残差块的隐藏单元学习到一些有用信息，那么就能比恒等函数表现得更好。

而对于plain神经网络来说，就算是学习恒等函数的参数都很困难，因此很多层最后的表现变差了。

另外，ResNet使用same卷积，保证 $z^{[l+2]}$ 和 $a^{[l]}$ 有相同的维度，可以相加。如果输入和输出维度不一样，就再增加一个矩阵 $w_s$ 。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\137.png' width="80%" height="80%"/>

几个之前提到的细节：

- 使用3×3 same卷积，保证 $z^{[l+2]}$ 和 $a^{[l]}$ 有相同的维度，可以相加。
- pool-like 层，进行 /2 降维操作。
- CONV-CONV-CONV-POOL 交替进行的结构。



#### 2.5 网络中的网络 / 1×1卷积

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\138.png' width="80%" height="80%"/>

1×1卷积添加了非线性函数，可以让网络学习更复杂的函数。

1×1卷积对单通道作用不大，但对于多通道，可以把所有通道相同位置的数输出成一个数（对应位置相乘 -> 相加 -> ReLU）。如果filter数量不止一个，可以输出多个通道。

论文名字叫 network in network ，这种方法也可以称为1×1卷积。论文中的架构没有得到广泛使用，但这种方法利用到了之后的Inception等模型上。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\139.png' width="80%" height="80%"/>

作用如上图。POOL的作用是压缩 $n_H,n_W$，而 1×1 卷积可以压缩 $n_C$，减少信道数量来简化计算。当然让信道数量保持不变或者增加也可以。



#### 2.6 Inception 模块简介、

Inception模块：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\140.png' width="80%" height="80%"/>

Inception：不需要人来决定使用什么规格的filter、是否使用POOL。网络结构更复杂但表现更好。

如图，Inception模块输入某个量，经过不同的处理，输出将这些结果叠加起来。

Inception网络不需要人为决定使用哪个fitler，或是否需要池化，而是由网络自行决定这些参数。我们可以给网络添加这些参数的所有可能的值，然后把这些输出连接起来，让网络学习他需要什么样的参数。

为了维持所有的维度相同，对卷积要使用filter卷积，对池化要使用padding（比较特殊的POOL）。



巨大的运算量：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\141.png' width="80%" height="80%"/>

Inception模块的问题是参数多，计算成本高。以 5×5卷积的一部分为例：需要 5×5×192 filter，对于输出的每个数都要做 filter 规格次数的乘法，也就是一共要做 $(28×28×32) * (5×5×192) ≈ 120 M$ 次乘法。



改进方法：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\142.png' width="80%" height="80%"/>

使用 1×1 卷积得到相同规格的输出，通过压缩成较小的中间形态（瓶颈层bottleneck layer）。

一共要做 $(28×28×16) * (1×1×192)  + (28×28×32) * (5×5×16) ≈ 2.4 M + 10 M = 12.4 M$ 次运算，乘法计算的成本大约变为原来的十分之一。

通过合理构建瓶颈层，既可以显著缩小表示层的规模，又不会降低网络性能，从而大量节省计算成本。



#### 2.7 Inception 网络 / GoogLeNet（Inception network）

Inception模块：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\143.png' width="80%" height="80%"/>



Inception网络：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\144.png' width="80%" height="80%"/>

红圈：由Inception模块重复堆叠而来，有些max-pooling层，来改变长和宽。

绿圈：一些分支，通过一些隐藏层，做一个softmax分类。它确保了即使是隐藏单元和中间层，也参与了特征运算，也能进行预测图片的分类。并且防止网络过拟合。



其他：也有变体把 Inception 和 ResNet 结合起来。可以看Inception的后续论文。



#### 2.8 使用开源的实现方案

很多神经网络难以复现，因为一些超参数的细节调整会影响性能。

先从使用开源的实现开始。



#### 2.9 迁移学习（Transfer Learning）

用迁移学习把公共数据集的知识迁移到我们自己的问题上。

在做一个计算机视觉的应用时，相比于从头训练权重、随机初始化，可以**下载开源的、别人已经训练好的网络结构的权重，作为我们模型的初始化**。

以猫咪分类问题为例，我们使用预训练的 ImageNet 模型，将最后分类改为 softmax 分类这是猫是Tigger，还是Misty，还是都不是。

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\145.png' width="80%" height="80%"/>

**图上**：把前面预训练的模型当作冻结的，**只训练跟我们的 softmax 层有关的参数**。

- 或许可以设置`trainableParameter = 0​`或`freeze=1`这样的参数，指定不训练特定层的权重。

- 可以把前面冻结的模型看作一个函数，输入一张图片，输出一个特征向量。只训练后面的softmax层，用这个特征向量来做预测。因此可以**提前计算训练集中所有样本的这一层的激活值**，然后存到硬盘里，在此之上训练softmax层。这样就不用每次遍历数据集重新计算这一层的激活值了。

**图中**：如果模型特别大，可以freeze一部分模型，然后**训练后面的模型**。也可以freeze一部分模型，把后面的模型进行修改。

- 规律：**数据集越大，需要冻结的层数越少，需要进行训练的层数越多**。

**图下**：如果有特别多的数据，就用开源的网络和它的权重当作参数的初始化，然后**训练整个网络**。



其他：计算机视觉问题中，迁移学习特别常用。除非有一个极其大的数据集，才从头开始训练所有东西。



#### 2.10 数据增强（Data augmentation）

计算机视觉方面的主要问题是没有办法得到充足的数据。当训练模型时，不管是从别人预训练的模型进行迁移学习，还是从源代码开始训练模型，数据增强会经常有所帮助。

数据增强的方法：

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\146.png' width="80%" height="80%"/>

<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\147.png' width="80%" height="80%"/>

- 镜像翻转、随即裁剪（保留主体）、旋转、剪切、局部弯曲 等。也可以组合起来用，但因为太复杂，实际上用的很少。
- 色彩转换，进行RGB的调整，一般是根据某种概率分布来决定改变的值。
  - *对RGB不同的采样方式：使用PCA（见机器学习网课笔记）。在AlexNet的论文中，成为“PCA color augmentation”，比如我们的图片呈紫色（红蓝多，绿少），那么PCA颜色增强算法会对红蓝有大的增减幅度，对绿的变化相对少，以此使总体的颜色保持一致。



<img src='C:\Users\acbgzm\Documents\GitHub\MyPostImage\ml-notes-img\nndl\148.png' width="80%" height="80%"/>

如果数据集比较大，常用的方法是设置单个thread，串行读取数据、进行数据增强。

- thread A：**从硬盘读数据并数据增强**。CPU有一个thread不停地从硬盘中读取数据，同时进行变形或颜色转换形成新的图像，从而构成一个batch或者mini-batch的数据；

- thread B：**训练**。这些数据被传递给其他thread（可能是CPU或GPU），进行模型的训练。

**以上两个thread可以并行实现。**



其他：数据增强也有一些超参数，比如如何进行颜色变化等。方法依然是学习开源的实现。



#### 2.11 计算机视觉现状









### 第三周 目标检测（Object detection）

#### 3.1 目标定位（Object localization）

#### 3.2 特征点检测（Landmark detection）

#### 3.3 目标检测（Object detection）

#### 3.4 卷积的滑动窗口实现（Convolutional implementation of sliding windows）

#### 3.5 Bounding Box预测（Bounding box predictions）

#### 3.6 交并比（Intersection over union）

#### 3.7 非极大值抑制（Non-max suppression）

#### 3.8 Anchor Boxes

#### 3.9 YOLO 算法（Putting it together: YOLO algorithm）

#### 3.10 候选区域（选修）（Region proposals (Optional)）









### 第四周 特殊应用：人脸识别和神经风格转换

#### 4.1 什么是人脸识别？(What is face recognition?)

#### 4.2 One-Shot学习（One-shot learning）

#### 4.3 Siamese 网络（Siamese network）

#### 4.4 Triplet 损失（Triplet 损失）

#### 4.5 面部验证与二分类（Face verification and binary classification）

#### 4.6 什么是神经风格转换？（What is neural style transfer?）

#### 4.7 什么是深度卷积网络？（What are deep ConvNets learning?）

#### 4.8 代价函数（Cost function）

#### 4.9 内容代价函数（Content cost function）

#### 4.10 风格代价函数（Style cost function）

#### 4.11 一维到三维推广（1D and 3D generalizations of models）















## 第五课 序列模型(Sequence Models)















