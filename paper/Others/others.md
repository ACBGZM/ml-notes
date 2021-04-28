列表：

- [ ]  A Neural Algorithm of Artistic Style
- [ ]  1
- [ ]  2
- [ ]  







## A Neural Algorithm of Artistic Style

### 思路

CNN的每一层可以理解为：每个 filter 提取特定的特征，输出是提取出来的不同特征的组合，称为特征图。

在训练CNN完成目标识别任务时，形成一个图像的表示，使对象信息按层次的处理更清晰。

- content：网络越深层，越关注图像的content；越浅层越关注像素的值。因此选用深层网络来表示content。

- style：使用原本为了捕获texture信息的feature space。这个feature space建立在每层的 top filter 之上。由 feature map 不同 filter 的 correlations 组成。通过包括网络多层的feature correlation，可以获得图片的 texture 表示。

**图一**

- CNN：filter的数量随网络变深而增加，但每个channel的图片因为下采样技术而尺寸减小。

- content重建：可以通过只知道网络在某一特定层的反应来重建输入图像，从而使CNN中不同处理阶段的信息可视化。我们从原始VGG网络的 conv1_1（a）、conv2_1（b）、conv3_1（c）、conv4_1（d）和 conv5_1（e）层中重构输入图像。我们发现，低层的重建几乎是完美的（a,b,c）。在网络的高层，详细的像素信息被破坏了，而图像的高级 content 却被保留了下来（d,e）。

- style重建：在原始CNN表征的基础上，我们建立了一个新的特征空间来捕捉输入图像的风格。风格表征计算了CNN不同层中不同特征之间的相关性。我们从建立在CNN层不同子集上的风格表征重建了输入图像的样式。这将创造 style 越来越匹配的图像，同时丢弃图像的整体布置信息。
  - conv1_1（a）
  - conv1_1和conv2_1（b）
  - conv1_1，conv2_1和conv3_1（c）
  - conv1_1，conv2_1，conv3_1和conv4_1（d）
  - conv1_1，conv2_1，conv3_1，conv4_1和conv5_1'（e）

（content方面，深层的网络能保留高级content信息，详细的像素信息被破坏；style方面，横跨多层的风格表征更能匹配原来的style。）

本文的关键发现：CNN对图片的content和style的表示是可以分离的。因此也可以将它们组合起来（图二）。

**图二**

style的表示也包含了网络多层的组合。当然也可以只包含低层，会得到不同的结果（图三按列对比）。当包含较高层style特征时，捕获图像更流畅的色彩和style（低层像色块），可以用感受野和复杂度等来理解这一点。

当然图像的content和style不能完全分离，也不能做到两全的合成。但损失函数是分离的，可以设置它们不同的参数来决定合成更偏重style还是content。（图三按行对比）

（related work、展望）

### Method

使用了19层VGGNetwork的**16个卷积层和5个池化层**提供的特征空间。 不使用任何全连接层。

对于图像合成，我们发现用**平均池化**代替最大池化可以改善梯度流。

#### content 损失

通常，网络中的每层都定义一个非线性filter组，其复杂度随网络深层而增加。因此，给定的输入图像〜x通过每层的filter在CNN的每一层中进行编码。 具有N 1个不同filter的层具有N 1个特征图，每个特征图的大小均为M 1，其中M 1是特征图的高度乘以宽度。

因此，$l$ 层的所有特征图可以存储在矩阵 $F^l ∈ R^{N_l \times M_l}$ 中，其中 $F^l_{ij}$ 是层 $l$ 中第 $i$ 个filter位置 $j$ 处的激活值。为了可视化在层次结构的不同层上编码的图像信息（图1，内容重建），我们**在白噪声图像上执行梯度下降**，以找到与原始图像的content特征响应匹配的另一幅图像。计算梯度下降的**损失函数**：

$$L_{content}(\vec p,\vec x,l) = \frac{1}{2}\sum_{i,j}(F^l_{ij}-P^l_{ij})$$ 

- $\vec p$：原图片，photograph提供内容
- $\vec x$：生成图片
- $P^l$：原图片在 $l$ 层的特征表示
- $F^l$：生成图片在 $l$ 层的特征表示
-  $F^l_{ij}$：**层 $l$ 中第 $i$ 个filter产生的特征图中位置 $j$ 处的激活值**

在梯度下降过程中，更改 $\vec x$。

图一从不同VGG层计算content损失，生成结果：

- conv1_1（a） 
- conv1_1和conv2_1（b） 
- conv1_1，conv2_1和conv3_1（c） 
- conv1_1，conv2_1，conv3_1和conv4_1（d）
- conv1_1，conv2_1，conv3_1，conv4_1和conv5_1'（e）



#### style 损失

构建一种style表示，来计算不同filter输出的channel的相关性。由Gram矩阵 $G^l∈R^{N_l \times N_l}$ 表示，$G^l_{ij}$ 是第 $l$ 层中特征图 $i$ 和 $j$ 的矢量化之间的内积：

$$G^l_{ij} = \sum_{k}F^l_{ik}F^l_{jk}$$ 

同样在白噪声图像上使用梯度下降，来生成匹配style的新图像。$l$ 层上的损失函数：

$$E_l = \frac{1}{4N_l^2M_l^2}\sum_{i,j}(G^l_{ij}-A^l_{ij})$$ 

所有层上，总style损失函数：

$$L_{style}(\vec a, \vec x) = \sum^L_{l=0}w_lE_l$$

- $\vec a$：原图片，art提供风格
- $\vec x$：生成图片 
- $A^l$：原图片在 $l$ 层的特征表示
- $G^l$：生成图片在 $l$ 层的特征表示
- $N_l$：一层中filter（channel）的数量
- $M_l$：特征图的尺寸，长度×宽度

图一从不同VGG层的组合计算style损失，生成结果：conv1_1（a）、conv2_1（b）、conv3_1（c）、conv4_1（d）和 conv5_1（e）



#### 总损失

$$L_{total} (\vec p,\vec a,\vec x) = αL_{content} (\vec p,\vec x) + βL_{style} (\vec a,\vec x) $$

- $\vec p$：photograph
- $\vec a$：art
- $\vec x$：生成



图二：

- content损失在'conv4_2'上计算；style损失在'conv1_1'，'conv2_1'，'conv3_1'，'conv4_1'和'conv5_1'上计算（在这些层中$w_l=\frac{1}{5}$，在所有其他层中 $w_l = 0$）

- α/β之比为 $1×10^{-3}$（图2 BCD）或 $1×10^{-4}$（图2 EF）

  

图三显示了针对content和sytle的不同相对权重（按行），以及在VGG不同层计算的style损失（沿列）

- 'conv1_1'（A）
- 'conv1_1', 'conv2_1'（B）
- 'conv1_1', 'conv2_1', 'conv3_1'（C）
- 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'（D）
- 'conv1_1', 'conv2_1' , ' conv3_1', ' conv4_1', ' conv5_1'（E）
- 因子 $w_l$ 总是等于1除以具有非零损耗权重的层数。

