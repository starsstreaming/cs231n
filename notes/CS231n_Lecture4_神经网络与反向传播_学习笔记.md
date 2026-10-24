# 🧠 CS231n 第 4 讲：神经网络与反向传播（Neural Networks & Backpropagation）

> **课程**：Stanford CS231n《Deep Learning for Computer Vision》 · Lecture 4
> **资料来源**：课程讲义 PPT，以及 CS231n 官方笔记 *Backpropagation, Intuitions*、*Backpropagation for a Linear Layer*、*Derivatives, Backpropagation, and Vectorization*，Distill 文章 *Why Momentum Really Works*，LeCun 等人的 *Efficient BackProp*（1998）
> **适合读者**：已经学过线性分类器、Softmax/SVM 损失和梯度下降，准备系统理解“神经网络到底怎么学习”的同学

---

## 🎯 学习目标

读完本笔记后，你应该能够：

1. 说清楚**为什么线性分类器不够用**，以及为什么神经网络必须有**非线性激活函数**；
2. 写出两层/三层**全连接网络**（MLP）的前向传播，并标出每个矩阵的维度；
3. 在**计算图**上用**链式法则**手算任意标量函数的梯度（会做 PPT 上的两个例题）；
4. 记住四种常见“门”的**梯度流动模式**：加法 = 分配，乘法 = 交换，复制 = 相加，max = 路由；
5. 用 **forward / backward 模块化 API** 实现一个算子，并理解 PyTorch autograd 的思路；
6. 推导**矩阵乘法**的反向传播 $\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}W^\top$、$\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Y}$，并理解为什么**不能显式构造雅可比矩阵**；
7. 了解梯度检查、动量、输入归一化、权重初始化等实用技巧。

---

## 📑 目录

- [0. 课程通知](#0-课程通知)
- [1. 术语表（中英对照）](#1-术语表中英对照)
- [2. 回顾：从线性分类器到优化](#2-回顾从线性分类器到优化)
- [3. 神经网络：给线性模型加上“弯折”](#3-神经网络给线性模型加上弯折)
- [4. 激活函数一览](#4-激活函数一览)
- [5. 网络规模、容量与正则化](#5-网络规模容量与正则化)
- [6. 生物神经元类比（以及它的局限）](#6-生物神经元类比以及它的局限)
- [7. 为什么需要反向传播](#7-为什么需要反向传播)
- [8. 计算图与链式法则：标量例题](#8-计算图与链式法则标量例题)
- [9. 梯度流动的四种模式](#9-梯度流动的四种模式)
- [10. 代码实现：从“扁平代码”到模块化 API](#10-代码实现从扁平代码到模块化-api)
- [11. 向量与矩阵的反向传播](#11-向量与矩阵的反向传播)
- [12. 扩展阅读精华：实用技巧](#12-扩展阅读精华实用技巧)
- [13. 常见误区](#13-常见误区)
- [14. 自测题（附答案）](#14-自测题附答案)
- [15. 一页纸总结](#15-一页纸总结)
- [16. 参考资料](#16-参考资料)

---

## 0. 课程通知

| 事项 | 截止日期 |
|---|---|
| 作业 1（Assignment 1） | 4 月 16 日 |
| 项目提案（Project Proposal） | 4 月 23 日 |

---

## 1. 术语表（中英对照）

| 中文 | English | 一句话解释 |
|---|---|---|
| 得分函数 | Score function | 把输入映射为各类别分数的函数，如 $s=Wx$ |
| 损失函数 | Loss function | 衡量预测有多“差”的标量，训练目标是让它变小 |
| 正则化 | Regularization | 在损失里加上 $\lambda R(W)$，惩罚过于复杂的模型、防止过拟合 |
| 梯度 | Gradient | 多元函数对各个输入的偏导数组成的向量，指向函数增长最快的方向 |
| 梯度下降 | Gradient Descent | 沿负梯度方向迭代更新参数：$W\leftarrow W-\alpha\nabla_W L$ |
| 随机梯度下降 | Stochastic Gradient Descent (SGD) | 每次只用一个小批量（minibatch）估计梯度，速度快 |
| 学习率 | Learning rate | 每一步走多远的超参数 $\alpha$ |
| 学习率调度 | Learning rate schedule | 训练过程中按规则改变学习率（阶梯、余弦、线性等） |
| 神经网络 | Neural Network (NN) | 由“线性变换 + 非线性函数”层层堆叠形成的函数 |
| 全连接层 | Fully-connected layer (FC) | 每个输出都连接到所有输入的层，本质是矩阵乘法 |
| 多层感知机 | Multi-Layer Perceptron (MLP) | 全连接网络的另一种叫法 |
| 隐藏层 | Hidden layer | 输入层和输出层之间的层；“2 层网络”有 1 个隐藏层 |
| 激活函数 | Activation function | 逐元素作用的非线性函数，如 ReLU、Sigmoid |
| 线性整流单元 | ReLU (Rectified Linear Unit) | $\max(0,x)$，现代网络的默认选择 |
| 容量 | Capacity | 模型能拟合的函数有多复杂 |
| 计算图 | Computational graph | 把计算拆成节点（运算）和边（数据）组成的有向无环图 |
| 前向传播 | Forward pass / Forward propagation | 沿计算图从输入算到输出（损失） |
| 反向传播 | Backpropagation (Backprop) | 沿计算图从损失往回、用链式法则递归计算梯度 |
| 链式法则 | Chain rule | 复合函数求导法则：$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x}$ |
| 上游梯度 | Upstream gradient | 从输出端传回到当前节点的梯度 $\frac{\partial L}{\partial z}$ |
| 局部梯度 | Local gradient | 当前节点输出对自身输入的导数 $\frac{\partial z}{\partial x}$ |
| 下游梯度 | Downstream gradient | 当前节点传给输入端的梯度 = 上游 × 局部 |
| 雅可比矩阵 | Jacobian matrix | 向量→向量函数的全部偏导数构成的矩阵 |
| 广义雅可比 | Generalized Jacobian | 张量→张量函数的偏导数，形状是“输出形状 × 输入形状” |
| 自动微分 | Automatic differentiation (autodiff/autograd) | 框架根据计算图自动求梯度的技术，反向传播是其“反向模式” |
| 数值梯度 | Numerical gradient | 用有限差分 $\frac{f(x+h)-f(x-h)}{2h}$ 近似的梯度 |
| 解析梯度 | Analytic gradient | 用微积分推出的精确梯度公式 |
| 梯度检查 | Gradient check | 用数值梯度验证解析梯度是否写对 |
| 动量 | Momentum | 累积历史梯度的“速度”，帮助加速并抑制振荡 |
| 条件数 | Condition number $\kappa$ | Hessian 最大/最小特征值之比，越大问题越“病态”、越难优化 |

---

## 2. 回顾：从线性分类器到优化

### 2.1 我们已经有了什么

| 组件 | 公式 |
|---|---|
| 线性得分函数 | $s=f(x;W)=Wx$ |
| Softmax 损失 | $L_i=-\log\left(\dfrac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$ |
| SVM（铰链）损失 | $L_i=\sum_{j\neq y_i}\max(0,\ s_j-s_{y_i}+1)$ |
| 总损失 | $L=\dfrac{1}{N}\sum_{i=1}^N L_i+\lambda R(W)$ |

我们的目标：找到使 $L$ 最小的 $W$。方法是**梯度下降**——求 $\nabla_W L$，沿反方向走一小步。

### 2.2 两种求梯度的方法

| | 数值梯度（Numerical） | 解析梯度（Analytic） |
|---|---|---|
| 速度 | 🐢 慢（每个参数都要算一次前向） | 🚀 快 |
| 精度 | 近似 | 精确 |
| 实现难度 | 容易 | 容易出错 |

> 💡 **实践原则**：永远用**解析梯度**训练，用**数值梯度**做**梯度检查（gradient check）**。

### 2.3 SGD 与优化器

- **小批量（minibatch）SGD**：每步随机取 32 / 64 / 128 / 256 个样本估计梯度。
- 常见优化器：**SGD**、**SGD + Momentum**、**RMSProp**、**Adam**。

### 2.4 学习率调度（Learning Rate Schedules）

| 名称 | 公式 / 规则 | 说明 |
|---|---|---|
| 阶梯式（Step） | 在固定 epoch 乘以衰减系数 | ResNet：第 30、60、90 个 epoch 学习率 × 0.1 |
| 余弦（Cosine） | $\alpha_t=\frac{1}{2}\alpha_0\left(1+\cos\frac{t\pi}{T}\right)$ | 平滑下降，超参数少，很常用 |
| 线性（Linear） | $\alpha_t=\alpha_0\left(1-\frac{t}{T}\right)$ | 在 NLP / Transformer 中常见 |
| 逆平方根（Inverse sqrt） | $\alpha_t=\alpha_0/\sqrt{t}$ | Transformer 原论文使用（常配合 warmup） |

> 其中 $\alpha_0$ 是初始学习率，$t$ 是当前步/epoch，$T$ 是总步数。

---

## 3. 神经网络：给线性模型加上“弯折”

### 3.1 从线性到两层网络

| 模型 | 公式 |
|---|---|
| 线性分类器（之前） | $f=Wx$ |
| **2 层神经网络** | $f=W_2\max(0,\ W_1x)$ |
| **3 层神经网络** | $f=W_3\max(0,\ W_2\max(0,\ W_1x))$ |

> 实际中每层通常还有**偏置（bias）** $b$，即 $W x+b$，为简洁常省略不写。

**维度（以 2 层为例）**：

$$
x\in\mathbb{R}^{D},\qquad W_1\in\mathbb{R}^{H\times D},\qquad h=\max(0,W_1x)\in\mathbb{R}^{H},\qquad W_2\in\mathbb{R}^{C\times H},\qquad s\in\mathbb{R}^{C}
$$

以 CIFAR-10 为例：$x$ 是 $32\times32\times3=3072$ 维，隐藏层 $H=100$，输出 $C=10$ 类：

```text
x (3072) ──W1──▶ h (100) ──W2──▶ s (10)
```

> 🔍 **直观理解**：线性分类器每个类别只能学**1 个模板**（W 的一行）。两层网络的 $W_1$ 可以学 **100 个模板**（比如“朝左的马”“朝右的马”“红色的车”……），$W_2$ 再把这些模板**组合**成各类别的分数——模板在类别之间是**共享**的。

### 3.2 为什么非线性是必须的？

**原因一：线性模型画不出弯曲的决策边界。** 例如两类点分布在同心圆上，在直角坐标 $(x,y)$ 下无法用一条直线分开；但做一次非线性的**极坐标变换** $(r,\theta)$ 后，它们就变得线性可分了。神经网络的隐藏层就相当于**自动学习这种特征变换**。

**原因二：没有激活函数，多层等于一层。**

$$
f=W_2(W_1x)=(W_2W_1)x=W_3x
$$

不论叠多少层，结果仍是线性分类器。激活函数是让“深度”有意义的关键。

### 3.3 术语澄清

- 这类网络叫 **全连接网络（Fully-connected network）** 或 **多层感知机（MLP）**。
- **“N 层网络”只数有权重的层**：“2 层网络” = “1 个隐藏层”，“3 层网络” = “2 个隐藏层”。

### 3.4 三层网络的前向传播（numpy）

```python
import numpy as np

f = lambda x: 1.0 / (1.0 + np.exp(-x))   # 激活函数（这里用 sigmoid）
x = np.random.randn(3, 1)                 # 输入：3 维列向量
h1 = f(np.dot(W1, x) + b1)                # 第 1 个隐藏层：4×1
h2 = f(np.dot(W2, h1) + b2)               # 第 2 个隐藏层：4×1
out = np.dot(W3, h2) + b3                 # 输出：1×1（最后一层一般不加激活）
```

### 3.5 约 20 行代码训练一个两层网络

PPT 中的完整训练代码（sigmoid 隐藏层 + L2 损失 + 手写反向传播）：

```python
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)          # ① 定义数据
w1, w2 = randn(D_in, H), randn(H, D_out)        #    和网络参数

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))            # ② 前向传播
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)            # ③ 解析梯度（反向传播）
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))     #    sigmoid 导数 σ(1-σ)

    w1 -= 1e-4 * grad_w1                        # ④ 梯度下降
    w2 -= 1e-4 * grad_w2
```

> 📌 先别急着记公式——读完第 8~11 节，你会发现每一行梯度都能从“上游 × 局部”机械地推出来。例如 `grad_w2 = h.T · grad_y_pred` 正是第 11.4 节的 $\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Y}$。

---

## 4. 激活函数一览

| 名称 | 公式 | 特点 |
|---|---|---|
| **ReLU** | $\max(0,x)$ | ⭐ 大多数问题的**默认首选**；计算简单，正区间不饱和 |
| Leaky ReLU | $\max(0.1x,\ x)$ | 负区间有小斜率，避免神经元“死亡”（dead ReLU） |
| ELU | $x\ (x\ge0);\ \ \alpha(e^x-1)\ (x<0)$ | 负区间平滑，输出均值更接近 0 |
| GELU | $x\cdot\Phi(x)$（$\Phi$ 为标准正态 CDF） | Transformer（BERT、GPT）中常用 |
| SiLU / Swish | $x\cdot\sigma(x)$ | 平滑、非单调，用于 EfficientNet 等 |
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | 输出 (0,1)；两端**饱和**导致梯度消失，非零中心 |
| Tanh | $\tanh(x)$ | 输出 (−1,1)，零中心，但两端同样饱和 |

> ⚠️ **饱和（saturation）**：当 $|x|$ 很大时 sigmoid/tanh 的导数趋近于 0。反向传播时梯度要乘以这个局部导数，于是几乎“传不回去”——这就是**梯度消失（vanishing gradient）**。Sigmoid 导数最大也只有 0.25。

---

## 5. 网络规模、容量与正则化

- **隐藏神经元越多 → 容量越大**，能拟合越复杂的决策边界（PPT 中 3 / 6 / 20 个神经元的对比图）。
- 但容量大也意味着更容易**过拟合**。

> ❗ **重要结论：不要用“缩小网络”来做正则化，而应该用更强的正则化。**
> 保持较大的网络，调大正则化强度 $\lambda$（PPT 对比了 $\lambda=0.001,\ 0.01,\ 0.1$：$\lambda$ 越大，决策边界越平滑）。原因是大网络的损失曲面更“友好”，更容易优化到好的解，过拟合问题交给正则化来控制。

🛠️ **动手玩**：
- [ConvNetJS 二维分类演示](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
- [TensorFlow Playground](https://playground.tensorflow.org)

---

## 6. 生物神经元类比（以及它的局限）

| 生物神经元 | 人工神经元 |
|---|---|
| 树突（Dendrite）：接收其他神经元的信号 | 输入 $x_i$ |
| 突触（Synapse）：连接强度 | 权重 $w_i$ |
| 胞体（Cell body）：整合信号 | 求和 $\sum_i w_ix_i+b$ |
| 轴突（Axon）：发出“放电”信号 | 激活函数输出 $f\left(\sum_i w_ix_i+b\right)$ |
| 放电频率（Firing rate） | 激活值（sigmoid 曾被视为放电率模型） |

```python
class Neuron:
    def neuron_tick(self, inputs):
        """假设 inputs 和 weights 都是一维 numpy 数组，bias 是数字"""
        cell_body_sum = np.sum(inputs * self.weights) + self.bias
        firing_rate = 1.0 / (1.0 + np.exp(-cell_body_sum))  # sigmoid
        return firing_rate
```

- **生物神经网络**：连接复杂、不规则；**人工神经网络**：为计算效率组织成规则的**层**，从而可以用矩阵乘法批量计算。
- 有趣的是，**随机连接**的网络也能工作得很好（Xie 等，2019，*Exploring Randomly Wired Neural Networks*）。

> ⚠️ **类比需谨慎**：真实神经元种类繁多；树突本身可以做复杂的非线性计算；突触不是一个简单的权重，而是复杂的非线性动力系统；用“放电率”描述信号也可能不够。所以“神经网络模仿大脑”只是一个**粗糙的启发**，而不是严格的模型。

---

## 7. 为什么需要反向传播

以 **SVM 铰链损失 + L2 正则** 的线性分类器为例：

$$
s=Wx,\qquad L_i=\sum_{j\ne y_i}\max(0,\ s_j-s_{y_i}+1),\qquad L=\frac1N\sum_i L_i+\lambda\sum_k W_k^2
$$

**“在纸上手推 $\nabla_W L$” 的问题：**

1. 😩 **繁琐**：矩阵微积分写一大堆，容易出错；
2. 🔁 **不可复用**：换个损失函数（比如 Softmax）就得从头再推一遍；
3. 🚫 **不可扩展**：对 AlexNet、神经图灵机（Neural Turing Machine）这样的复杂模型几乎不可能手推。

**更好的想法：计算图（Computational Graph）+ 反向传播（Backpropagation）**

```text
 x ──┐
     (*)──▶ s(scores) ──▶ [hinge loss] ──┐
 W ──┘                                   (+)──▶ L
 W ─────────▶ [R(W) 正则] ───────────────┘
```

把复杂函数拆成许多**简单的小运算（节点）**，每个节点只需要知道自己的**局部梯度**；再用**链式法则**把它们串起来，就能自动地算出任意复杂函数的梯度。

---

## 8. 计算图与链式法则：标量例题

### 8.1 核心公式：上游 × 局部 = 下游

对任意节点 $z=f(x,y)$：

```text
             前向：x, y ──▶ [ f ] ──▶ z
             反向：∂L/∂x ◀── [ f ] ◀── ∂L/∂z（上游梯度）
```

$$
\underbrace{\frac{\partial L}{\partial x}}_{\text{下游梯度}}=\underbrace{\frac{\partial L}{\partial z}}_{\text{上游梯度}}\cdot\underbrace{\frac{\partial z}{\partial x}}_{\text{局部梯度}}
$$

> 💡 **关键洞察**：每个节点都是“**只顾自己**”的——它不需要知道整个网络长什么样，只要能算出自己的局部梯度，再乘上传回来的上游梯度即可。这就是反向传播可以**模块化**的原因。

### 8.2 例题 1：$f(x,y,z)=(x+y)z$

已知 $x=-2,\ y=5,\ z=-4$。

**第 1 步：拆成计算图**，令 $q=x+y$，则 $f=qz$。

**第 2 步：前向传播**

| 变量 | 值 |
|---|---|
| $q=x+y$ | $-2+5=3$ |
| $f=q\cdot z$ | $3\times(-4)=-12$ |

**第 3 步：写出局部梯度**

$$
\frac{\partial q}{\partial x}=1,\quad \frac{\partial q}{\partial y}=1,\quad \frac{\partial f}{\partial q}=z,\quad \frac{\partial f}{\partial z}=q
$$

**第 4 步：从输出往回传**

| 梯度 | 计算 | 结果 |
|---|---|---|
| $\partial f/\partial f$ | 起点 | $1$ |
| $\partial f/\partial z$ | $q$ | $3$ |
| $\partial f/\partial q$ | $z$ | $-4$ |
| $\partial f/\partial y$ | $\frac{\partial f}{\partial q}\cdot\frac{\partial q}{\partial y}=-4\times1$ | $-4$ |
| $\partial f/\partial x$ | $\frac{\partial f}{\partial q}\cdot\frac{\partial q}{\partial x}=-4\times1$ | $-4$ |

**结果解读**：$\partial f/\partial x=-4$ 表示 $x$ 增加一点点 $\epsilon$，$f$ 大约减少 $4\epsilon$。验证：$x=-1.99$ 时 $f=(3.01)(-4)=-12.04$ ✅

### 8.3 例题 2：一个 Sigmoid 神经元

$$
f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}
$$

已知 $w_0=2,\ x_0=-1,\ w_1=-3,\ x_1=-2,\ w_2=-3$。

**计算图中用到的基本门及其局部梯度：**

| 门 | 前向 | 局部梯度 |
|---|---|---|
| 乘常数 $a$ | $f=ax$ | $a$ |
| 加常数 | $f=x+c$ | $1$ |
| 指数 | $f=e^x$ | $e^x$ |
| 倒数 | $f=1/x$ | $-1/x^2$ |

**前向传播（从左到右）：**

| 步骤 | 运算 | 值 |
|---|---|---|
| ① | $w_0x_0=2\times(-1)$ | $-2$ |
| ② | $w_1x_1=(-3)\times(-2)$ | $6$ |
| ③ | ①+② | $4$ |
| ④ | ③ $+w_2$ | $1$ |
| ⑤ | $\times(-1)$ | $-1$ |
| ⑥ | $\exp$ | $0.37$ |
| ⑦ | $+1$ | $1.37$ |
| ⑧ | $1/x$ | $\mathbf{0.73}$ |

**反向传播（从右到左）：**

| 节点 | 下游梯度 = 上游 × 局部 | 值 |
|---|---|---|
| ⑧ 输出 | 起点 | $1.00$ |
| ⑦ $1/x$ 门 | $1\times\left(-\frac{1}{1.37^2}\right)$ | $-0.53$ |
| ⑥ $+1$ 门 | $-0.53\times1$ | $-0.53$ |
| ⑤ $\exp$ 门 | $-0.53\times e^{-1}$ | $-0.20$ |
| ④ $\times(-1)$ 门 | $-0.20\times(-1)$ | $0.20$ |
| ③ 加法门（分给两边与 $w_2$） | $0.20\times1$ | $0.20$ |
| $w_2$ | $0.20$ | $\mathbf{0.20}$ |
| $w_0$ | $0.20\times x_0=0.20\times(-1)$ | $\mathbf{-0.20}$ |
| $x_0$ | $0.20\times w_0=0.20\times2$ | $\mathbf{0.40}$ |
| $w_1$ | $0.20\times x_1=0.20\times(-2)$ | $\mathbf{-0.40}$ |
| $x_1$ | $0.20\times w_1=0.20\times(-3)$ | $\mathbf{-0.60}$ |

### 8.4 捷径：把 Sigmoid 当成一个整体门

Sigmoid 的导数有一个漂亮的形式：

$$
\sigma(x)=\frac{1}{1+e^{-x}},\qquad \frac{d\sigma}{dx}=\frac{e^{-x}}{(1+e^{-x})^2}=\big(1-\sigma(x)\big)\,\sigma(x)
$$

于是 ④→⑧ 这一整串可以合并成一个 **sigmoid 门**：输入为 $1$，输出 $\sigma(1)=0.73$，局部梯度 $=(1-0.73)\times0.73\approx0.2$，一步就得到 $0.20$，与逐步计算完全一致。

> 💡 **计算图的划分不唯一**。原则是：**选择那些局部梯度容易写出来的节点**。把常见组合（sigmoid、softmax+交叉熵等）封装成一个门，既简洁又数值更稳定。

---

## 9. 梯度流动的四种模式

| 门 | 前向 | 反向行为 | 记忆口诀 |
|---|---|---|---|
| **加法门** add | $z=x+y$ | 两个输入都拿到**相同**的上游梯度 | 🟢 **梯度分配器**（distributor） |
| **乘法门** mul | $z=xy$ | $\frac{\partial L}{\partial x}=y\cdot\frac{\partial L}{\partial z}$，$\frac{\partial L}{\partial y}=x\cdot\frac{\partial L}{\partial z}$ | 🔵 **交换乘数**（swap multiplier） |
| **复制门** copy | $y_1=y_2=x$ | $\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y_1}+\frac{\partial L}{\partial y_2}$ | 🟡 **梯度相加器**（adder） |
| **max 门** | $z=\max(x,y)$ | 较大的输入拿到全部梯度，另一个得 0 | 🔴 **梯度路由器**（router） |

**小例子**：设上游梯度为 $2$，

- 加法 $3+4$：两个输入都得到 $2$；
- 乘法 $2\times5$：$x=2$ 得到 $5\times2=10$，$y=5$ 得到 $2\times2=4$；
- max$(4,-1)$：$4$ 得到 $2$，$-1$ 得到 $0$。

> 🧠 **复制门为什么是“相加”？** 如果一个变量被多处使用（例如网络中的分叉、或同一个 $W$ 同时出现在得分和正则项里），它对损失的影响是**所有路径影响之和**——这正是**多元链式法则**。代码里体现为梯度用 `+=` 累积。

> ⚠️ **乘法门的反直觉效应**（来自 CS231n 笔记）：$w$ 的梯度等于输入 $x$ 乘上游梯度。如果把输入数据放大 1000 倍，权重梯度也放大 1000 倍，你就必须相应调小学习率。这也是**输入预处理/归一化**如此重要的原因之一。

---

## 10. 代码实现：从“扁平代码”到模块化 API

### 10.1 “扁平”的反向传播代码

以第 8.3 节的 sigmoid 例子为例，前向时按顺序算，反向时**倒序**写一遍：

```python
def f(w0, x0, w1, x1, w2):
    # ---------- 前向传播 ----------
    s0 = w0 * x0
    s1 = w1 * x1
    s2 = s0 + s1
    s3 = s2 + w2
    L = sigmoid(s3)

    # ---------- 反向传播（倒序） ----------
    grad_L = 1.0
    grad_s3 = grad_L * (1 - L) * L      # sigmoid 门
    grad_w2 = grad_s3                   # 加法门：分配
    grad_s2 = grad_s3
    grad_s0 = grad_s2                   # 加法门：分配
    grad_s1 = grad_s2
    grad_w1 = grad_s1 * x1              # 乘法门：交换
    grad_x1 = grad_s1 * w1
    grad_w0 = grad_s0 * x0
    grad_x0 = grad_s0 * w0
    return L, (grad_w0, grad_x0, grad_w1, grad_x1, grad_w2)
```

> 这也正是作业 1 中“手写反向传播”的风格：**前向的每一行，在反向都有对应的一段**。

### 10.2 模块化的 Forward / Backward API

真实框架把每种运算封装成一个“门”对象：

```python
class ComputationalGraph:
    def forward(self, inputs):
        # 1. 把 inputs 送入输入门
        # 2. 按拓扑排序依次调用每个门的 forward
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss  # 最后一个门输出损失

    def backward(self):
        # 按拓扑逆序依次调用每个门的 backward
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward()  # 用链式法则计算本门输入的梯度
        return inputs_gradients
```

### 10.3 PyTorch 中自定义一个乘法门

```python
import torch

class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)   # 缓存反向时需要的值
        z = x * y
        return z

    @staticmethod
    def backward(ctx, grad_z):        # grad_z：上游梯度
        x, y = ctx.saved_tensors
        grad_x = y * grad_z           # 上游 × 局部（交换乘数）
        grad_y = x * grad_z
        return grad_x, grad_y
```

PyTorch 源码中 sigmoid 的反向同样是 `grad_output * (1 - output) * output`。

> 💡 **为什么要 `save_for_backward`？** 局部梯度往往依赖前向时的中间结果（乘法门要 $x,y$；sigmoid 门要输出 $\sigma$）。**前向时缓存、反向时取用**——这也是训练时显存占用远大于推理的原因。

---

## 11. 向量与矩阵的反向传播

### 11.1 三种“导数”

| 输入 → 输出 | 名称 | 形状 | 含义 |
|---|---|---|---|
| $x\in\mathbb{R}\to y\in\mathbb{R}$ | **导数** Derivative | 标量 | $x$ 变一点，$y$ 变多少 |
| $x\in\mathbb{R}^N\to y\in\mathbb{R}$ | **梯度** Gradient | $\mathbb{R}^N$ | 每个 $x_n$ 变一点，$y$ 变多少 |
| $x\in\mathbb{R}^N\to y\in\mathbb{R}^M$ | **雅可比矩阵** Jacobian | $\mathbb{R}^{M\times N}$ | 每个 $x_n$ 变一点，每个 $y_m$ 变多少 |
| 张量 → 张量 | **广义雅可比** Generalized Jacobian | (输出形状)×(输入形状) | 上面的推广（见 *derivatives.pdf*） |

### 11.2 向量版本的反向传播

对节点 $z=f(x,y)$，其中 $x\in\mathbb{R}^{D_x}$、$z\in\mathbb{R}^{D_z}$：

$$
\underbrace{\frac{\partial L}{\partial x}}_{D_x}=\underbrace{\left(\frac{\partial z}{\partial x}\right)^{\!\top}}_{D_x\times D_z\ \text{雅可比}}\underbrace{\frac{\partial L}{\partial z}}_{D_z}
$$

> 📏 **两条铁律**
> 1. **损失 $L$ 永远是标量**，所以上游梯度 $\frac{\partial L}{\partial z}$ 的形状总和 $z$ 相同；
> 2. **$\frac{\partial L}{\partial x}$ 的形状一定和 $x$ 相同**——每个元素表示“该元素变一点，损失变多少”。写完代码先检查形状！

### 11.3 例子：逐元素 ReLU

$$
x=[1,\ -2,\ 3,\ -1]\ \xrightarrow{\ \max(0,x)\ }\ z=[1,\ 0,\ 3,\ 0],\qquad \frac{\partial L}{\partial z}=[4,\ -1,\ 5,\ 9]
$$

雅可比矩阵是一个**对角阵**：对角线上 $x_i>0$ 处为 1，否则为 0：

$$
\frac{\partial z}{\partial x}=\begin{bmatrix}1&0&0&0\\0&0&0&0\\0&0&1&0\\0&0&0&0\end{bmatrix}
\quad\Longrightarrow\quad
\frac{\partial L}{\partial x}=[4,\ 0,\ 5,\ 0]
$$

> 🚫 **永远不要显式构造雅可比矩阵！** 它是稀疏的对角阵，直接用“掩码”即可：

```python
grad_x = grad_z * (x > 0)   # ReLU 反向，等价于 (∂L/∂x)_i = (∂L/∂z)_i if x_i > 0 else 0
```

### 11.4 矩阵乘法 $Y=XW$ 的反向传播

设 $X\in\mathbb{R}^{N\times D}$，$W\in\mathbb{R}^{D\times M}$，$Y=XW\in\mathbb{R}^{N\times M}$，即 $y_{n,m}=\sum_d x_{n,d}\,w_{d,m}$。

**为什么不能用雅可比？** $\frac{\partial Y}{\partial X}$ 的形状是 $(N\times M)\times(N\times D)$。当 $N=64,\ D=M=4096$ 时，元素个数为 $64\times4096\times64\times4096\approx6.9\times10^{10}$，以 32 位浮点存储约 **256 GB**——一个雅可比矩阵就把显存撑爆了。

**逐元素推导**（*Backpropagation for a Linear Layer* 的思路）：看 $X$ 中一个元素 $x_{n,d}$ 会影响哪些输出？它只出现在第 $n$ 行的 $y_{n,1},\dots,y_{n,M}$ 中，且 $\frac{\partial y_{n,m}}{\partial x_{n,d}}=w_{d,m}$。所以

$$
\frac{\partial L}{\partial x_{n,d}}=\sum_{m}\frac{\partial L}{\partial y_{n,m}}\,w_{d,m}=\left[\frac{\partial L}{\partial Y}W^\top\right]_{n,d}
$$

同理可得：

$$
\boxed{\ \frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}\,W^\top\qquad \frac{\partial L}{\partial W}=X^\top\,\frac{\partial L}{\partial Y}\ }
$$

**形状核对（最实用的记忆法）**：

| 量 | 形状 | 组合方式 |
|---|---|---|
| $\frac{\partial L}{\partial X}$ | $N\times D$ | $(N\times M)\cdot(M\times D)$ → 必须是 $\frac{\partial L}{\partial Y}W^\top$ |
| $\frac{\partial L}{\partial W}$ | $D\times M$ | $(D\times N)\cdot(N\times M)$ → 必须是 $X^\top\frac{\partial L}{\partial Y}$ |

> 💡 PPT 原话：“容易记住：这是唯一能让形状对得上的写法！”

**数值例子**（$N=2,\ D=3,\ M=4$，已用 numpy 验证）：

$$
X=\begin{bmatrix}2&1&-3\\-3&4&2\end{bmatrix},\quad
W=\begin{bmatrix}3&2&1&-1\\2&1&3&2\\3&2&1&-2\end{bmatrix},\quad
\frac{\partial L}{\partial Y}=\begin{bmatrix}2&3&-3&9\\-8&1&4&6\end{bmatrix}
$$

$$
\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}W^\top=\begin{bmatrix}0&16&-9\\-24&9&-30\end{bmatrix},\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Y}=\begin{bmatrix}28&3&-18&0\\-30&7&13&33\\-22&-7&17&-15\end{bmatrix}
$$

以 $\frac{\partial L}{\partial x_{1,1}}$ 为例手算：$x_{1,1}$ 只影响第 1 行输出，$=2\cdot3+3\cdot2+(-3)\cdot1+9\cdot(-1)=0$ ✅

### 11.5 把它们组合起来：两层网络的完整反向传播

```python
# 前向
z1 = X @ W1 + b1          # (N,H)
h  = np.maximum(0, z1)    # ReLU
s  = h @ W2 + b2          # (N,C)
# 假设 ds = ∂L/∂s 已由损失函数给出，形状 (N,C)

# 反向
dW2 = h.T @ ds            # (H,C)   矩阵乘法规则
db2 = ds.sum(axis=0)      # (C,)    偏置被“复制”到每个样本 → 梯度相加
dh  = ds @ W2.T           # (N,H)
dz1 = dh * (z1 > 0)       # ReLU：梯度路由
dW1 = X.T @ dz1           # (D,H)
db1 = dz1.sum(axis=0)     # (H,)
dW1 += 2 * reg * W1       # 别忘了正则项的梯度（W 被复制到两个分支 → 相加）
dW2 += 2 * reg * W2
```

---

## 12. 扩展阅读精华：实用技巧

### 12.1 梯度检查（Gradient Check）

使用**中心差分**（误差为 $O(h^2)$，优于单侧差分的 $O(h)$）：

$$
\frac{\partial f}{\partial x}\approx\frac{f(x+h)-f(x-h)}{2h},\qquad h\approx10^{-5}
$$

比较时使用**相对误差**：

$$
\text{rel\_error}=\frac{|g_a-g_n|}{\max(|g_a|,|g_n|)}
$$

| 相对误差 | 判断（CS231n 经验值） |
|---|---|
| $>10^{-2}$ | ❌ 梯度很可能写错了 |
| $10^{-4}\sim10^{-2}$ | ⚠️ 可疑 |
| $<10^{-4}$ | 对含 ReLU 等不可导点的网络通常可以接受 |
| $<10^{-7}$ | ✅ 很好 |

```python
def grad_check(f, x, analytic_grad, h=1e-5, num_checks=10):
    for _ in range(num_checks):
        ix = tuple(np.random.randint(m) for m in x.shape)
        old = x[ix]
        x[ix] = old + h; fxph = f(x)
        x[ix] = old - h; fxmh = f(x)
        x[ix] = old
        g_num = (fxph - fxmh) / (2 * h)
        g_ana = analytic_grad[ix]
        rel = abs(g_num - g_ana) / max(abs(g_num), abs(g_ana), 1e-12)
        print(f"numerical {g_num:.6f}  analytic {g_ana:.6f}  rel_error {rel:.2e}")
```

> 小技巧：用 `float64` 做梯度检查；先关掉正则化单独检查数据损失；注意 ReLU 在 0 处的“拐点”（kink）可能造成误报。

### 12.2 动量为什么有效（*Why Momentum Really Works* 概要）

**问题：病态曲面（ill-conditioning）。** 想象一个狭长的山谷：沿陡峭方向梯度大，沿平缓方向梯度小。普通梯度下降会在陡峭方向来回**振荡**，在平缓方向却**爬得很慢**。条件数 $\kappa=\lambda_{\max}/\lambda_{\min}$ 越大，情况越糟。

**动量更新：**

$$
z^{k+1}=\beta z^k+\nabla f(w^k),\qquad w^{k+1}=w^k-\alpha z^{k+1}
$$

- **直觉**：像一个**重球（heavy ball）**滚下山：在来回振荡的方向，正负梯度相互抵消；在一致的方向，速度不断累积。
- **理论结果**（凸二次函数上）：

| 方法 | 最优收敛速率（每步误差缩小的比例） |
|---|---|
| 梯度下降 | $\dfrac{\kappa-1}{\kappa+1}$ |
| 动量（最优 $\alpha,\beta$） | $\dfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$ |

  也就是说，动量把对条件数的依赖从 $\kappa$ 降到了 $\sqrt\kappa$——相当于**平方级的加速**。例如 $\kappa=100$ 时，所需迭代次数大约从“百级”降到“十级”。
- **阻尼视角**：$\beta$ 太小 → 过阻尼，收敛慢；$\beta$ 太大 → 欠阻尼，来回振荡；存在一个“临界阻尼”的最佳值。实践中常用 $\beta=0.9$（有时 0.99）。

### 12.3 LeCun《Efficient BackProp》中的经典技巧（1998）

| 技巧 | 内容 | 现代对应 |
|---|---|---|
| 优先用随机/小批量学习 | 比全批量快、能利用数据冗余、有时还能跳出局部极小 | minibatch SGD |
| **打乱数据**（Shuffle） | 让相邻样本来自不同类别；可以更频繁地展示误差大的样本（但小心离群点） | 每个 epoch 重新 shuffle |
| **输入归一化** | 每个输入特征减均值到 0；缩放使方差相同；尽量去相关（如 PCA） | 数据标准化、Batch/Layer Norm |
| **零中心的激活函数** | 推荐 $f(x)=1.7159\tanh\left(\tfrac{2}{3}x\right)$，使 $f(\pm1)=\pm1$ | tanh 优于 sigmoid 的原因 |
| 目标值放在激活函数的范围内 | 别把目标设在饱和区（如 tanh 的 ±1.7159），否则权重会被推向无穷大 | label smoothing 思想类似 |
| **权重初始化** | 从均值 0、标准差 $m^{-1/2}$ 的分布采样（$m$ 为扇入 fan-in），使激活落在非饱和区 | Xavier / He 初始化 |
| 每层/每个权重不同的学习率 | 浅层通常应该用更大的学习率；学习率可与扇入的平方根成反比 | 自适应优化器（Adam 等） |
| 二阶方法 | 牛顿法、高斯-牛顿、共轭梯度、BFGS 等；对大网络大多不实用 | 理解曲率、条件数的价值 |

---

## 13. 常见误区

| ❌ 误区 | ✅ 正解 |
|---|---|
| “多叠几层线性层，模型就更强” | 没有激活函数时，多层线性 = 一层线性 |
| “网络过拟合了，那就把它缩小” | 优先增大正则化强度 $\lambda$（或用 dropout 等），而不是缩小网络 |
| “反向传播是一种优化算法” | 反向传播只负责**算梯度**；用梯度更新参数的是 SGD/Adam 等**优化器** |
| “要算 $\partial L/\partial X$ 就先写出雅可比矩阵” | 雅可比又大又稀疏，应利用结构直接写出“向量-雅可比乘积” |
| “变量被用了两次，梯度取其中一个就行” | 分叉处梯度要**相加**（代码中用 `+=`） |
| “梯度形状无所谓，能广播就行” | $\partial L/\partial x$ 的形状必须与 $x$ 一致，形状不对几乎一定是推错了 |
| “用 sigmoid 作隐藏层激活没问题” | 它会饱和、非零中心；隐藏层默认用 ReLU（或 GELU 等） |
| “数值梯度又准又简单，就用它训练” | 太慢，只用于梯度检查 |

---

## 14. 自测题（附答案）

**Q1.** 2 层网络 $f=W_2\max(0,W_1x)$，输入 3072 维、隐藏层 100、输出 10 类。不含偏置时共有多少参数？

<details><summary>答案</summary>

$100\times3072+10\times100=307{,}200+1{,}000=308{,}200$（若含偏置再加 $100+10$，共 308,310）。
</details>

**Q2.** $f=(x+y)z$，$x=1,\ y=2,\ z=3$。求 $\partial f/\partial x,\ \partial f/\partial y,\ \partial f/\partial z$。

<details><summary>答案</summary>

$q=3$。$\partial f/\partial z=q=3$；$\partial f/\partial x=\partial f/\partial y=z=3$。
</details>

**Q3.** 节点 $z=\max(x,y)$，$x=5,\ y=7$，上游梯度为 $-2$。求两输入的梯度。

<details><summary>答案</summary>

max 门是路由器：$y$ 更大，$\partial L/\partial y=-2$，$\partial L/\partial x=0$。
</details>

**Q4.** 节点 $z=xy$，$x=3,\ y=-4$，上游梯度为 $2$。求两输入的梯度。

<details><summary>答案</summary>

乘法门交换乘数：$\partial L/\partial x=y\cdot2=-8$，$\partial L/\partial y=x\cdot2=6$。
</details>

**Q5.** sigmoid 的输出为 $0.5$ 时，局部梯度是多少？输出为 $0.99$ 时呢？这说明了什么？

<details><summary>答案</summary>

$0.5\times0.5=0.25$（最大值）；$0.99\times0.01=0.0099$。输出接近饱和时梯度几乎为 0，即梯度消失。
</details>

**Q6.** $Y=XW$，$X$ 为 $128\times512$，$W$ 为 $512\times256$。写出 $\partial L/\partial W$ 的公式和形状。

<details><summary>答案</summary>

$\partial L/\partial W=X^\top\,\partial L/\partial Y$，形状为 $(512\times128)(128\times256)=512\times256$，与 $W$ 相同。
</details>

**Q7.** 为什么正则项 $\lambda\|W\|^2$ 的梯度要“加到”数据损失的梯度上，而不是替换它？

<details><summary>答案</summary>

$W$ 在计算图中被复制到了两个分支（得分计算和正则项），复制门的梯度是各分支梯度之和：$\nabla_W L=\nabla_W L_{\text{data}}+2\lambda W$。
</details>

**Q8.** 判断：把输入数据整体放大 100 倍，对第一层权重的梯度有什么影响？

<details><summary>答案</summary>

第一层是乘法（$W x$），权重梯度 = 输入 × 上游梯度，所以大约放大 100 倍，需要相应调整学习率。这也是要做输入归一化的原因之一。
</details>

---

## 15. 一页纸总结

> 📝 **本讲核心**
> 1. **神经网络** = 线性层与非线性激活函数的堆叠：$f=W_2\max(0,W_1x)$；没有非线性，多层等于一层。
> 2. 网络越大容量越大；用**正则化**而不是缩小网络来控制过拟合。
> 3. **计算图**：把复杂函数拆成简单节点。
> 4. **反向传播**：前向时计算并**缓存**中间值；反向时对每个节点使用 **下游 = 上游 × 局部**。
> 5. 四种梯度模式：**加法分配、乘法交换、复制相加、max 路由**。
> 6. 实现：每个算子提供 `forward` / `backward`，框架按拓扑序（反向时逆序）调用——这就是 PyTorch autograd。
> 7. 向量/矩阵情形：$\partial L/\partial x$ 与 $x$ 同形；**不要显式构造雅可比**；矩阵乘法：$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}W^\top$，$\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Y}$。
> 8. **下一讲**：卷积神经网络（CNN）。

---

## 16. 参考资料

1. CS231n Lecture 4 讲义（本笔记主线）
2. CS231n 笔记：[Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)——直观解释、分阶段计算、梯度模式
3. Justin Johnson：[Backpropagation for a Linear Layer](https://cs231n.stanford.edu/handouts/linear-backprop.pdf)——矩阵乘法反向传播的逐元素推导
4. Justin Johnson：[Derivatives, Backpropagation, and Vectorization](https://cs231n.stanford.edu/handouts/derivatives.pdf)——导数、梯度、雅可比、广义雅可比
5. Gabriel Goh：[Why Momentum Really Works](https://distill.pub/2017/momentum/)，Distill 2017
6. LeCun, Bottou, Orr, Müller：[Efficient BackProp](https://cs231n.stanford.edu/papers/lecun-98b.pdf)，1998
7. CS231n 笔记：[Neural Networks Part 3（梯度检查）](https://cs231n.github.io/neural-networks-3/)
8. Xie et al., *Exploring Randomly Wired Neural Networks for Image Recognition*, ICCV 2019
9. 推荐视频：3Blue1Brown《深度学习》系列 —— 反向传播的直观动画
