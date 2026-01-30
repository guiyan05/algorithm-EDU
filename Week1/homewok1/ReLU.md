## ReLU（Rectified Linear Unit）

### 数学表达式
\[
f(x)=\max(0,x)
\]

### 优点
- 计算简单，收敛速度快
- 缓解梯度消失问题
- 激活稀疏，提高模型表达能力

### 缺点
- 存在“死亡 ReLU”问题
- 输出均值非零，可能影响训练稳定性

### 适用场景
- CNN、MLP 的默认激活函数
- 大规模深度网络

---

## Leaky ReLU

### 数学表达式
\[
f(x)=
\begin{cases}
x, & x>0 \\
\alpha x, & x\le0
\end{cases}
\quad (\alpha \approx 0.01)
\]

### 优点
- 负区间仍有梯度，缓解死亡 ReLU 问题
- 实现简单

### 缺点
- 斜率参数需人工设置
- 负区间表达能力有限

### 适用场景
- GAN、目标检测网络
- 对梯度稳定性要求较高的模型

---

## PReLU（Parametric ReLU）

### 数学表达式
\[
f(x)=
\begin{cases}
x, & x>0 \\
a x, & x\le0
\end{cases}
\quad (a \text{为可学习参数})
\]

### 优点
- 负区间斜率自适应学习
- 表达能力强于 Leaky ReLU

### 缺点
- 引入额外参数，存在过拟合风险
- 训练不稳定时效果可能下降

### 适用场景
- 深层 CNN（如人脸识别、视觉感知任务）

---

## RReLU（Randomized ReLU）

### 数学表达式
训练阶段：
\[
f(x)=
\begin{cases}
x, & x>0 \\
r x, & x\le0,\ r\sim U(l,u)
\end{cases}
\]

测试阶段：
\[
f(x)=
\begin{cases}
x, & x>0 \\
\mathbb{E}[r]x, & x\le0
\end{cases}
\]

### 优点
- 具有正则化效果，缓解过拟合
- 避免负区间斜率固定带来的表达限制

### 缺点
- 引入随机性，结果波动较大
- 推理阶段需使用期望值

### 适用场景
- 小数据集场景下的 CNN 模型

---

## ELU（Exponential Linear Unit）

### 数学表达式
\[
f(x)=
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\le0
\end{cases}
\]

### 优点
- 负区间输出趋近于 −α，使激活均值接近 0
- 收敛速度通常快于 ReLU

### 缺点
- 计算复杂度略高（涉及指数运算）
- 超参数 α 需要调节

### 适用场景
- 深层神经网络
- 需要稳定训练动态的任务

---

## SELU（Scaled ELU）

### 数学表达式
\[
f(x)=\lambda
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\le0
\end{cases}
\]
其中：
\[
\alpha \approx 1.6733,\quad \lambda \approx 1.0507
\]

### 优点
- 具备自归一化特性，激活自动趋近均值 0、方差 1
- 可减少对 Batch Normalization 的依赖

### 缺点
- 需配合特定初始化方式（LeCun Normal）
- 与标准 Dropout 不兼容（需使用 AlphaDropout）

### 适用场景
- 深层全连接网络
- 不使用 BatchNorm 的高效模型

---

## Softplus

### 数学表达式
\[
f(x)=\ln(1+e^x)
\]

### 优点
- ReLU 的光滑版本，处处可导
- 无死亡神经元问题

### 缺点
- 计算成本高
- 梯度较小，训练速度慢于 ReLU

### 适用场景
- 概率建模
- 变分自编码器（VAE）

---

## GELU（Gaussian Error Linear Unit）

### 数学表达式（近似形式）
\[
f(x)=x\cdot \Phi(x)\approx 0.5x(1+\tanh[\sqrt{2/\pi}(x+0.0447x^3)])
\]

### 优点
- 平滑非线性，概率意义明确
- 在 Transformer 模型中性能显著优于 ReLU

### 缺点
- 计算复杂度较高
- 实现成本高于 ReLU

### 适用场景
- Transformer
- NLP、大模型结构

---

## Swish

### 数学表达式
\[
f(x)=x\cdot \sigma(\beta x)
\]

### 优点
- 非单调激活函数，在深网络中性能优于 ReLU
- 梯度连续平滑

### 缺点
- 计算成本较高
- 参数 β 需要调节或学习

### 适用场景
- CNN
- EfficientNet 等高性能视觉模型

---

## Mish

### 数学表达式
\[
f(x)=x\tanh(\ln(1+e^x))
\]

### 优点
- 高度平滑，信息保留能力强
- 在图像分类与检测任务中常优于 Swish 和 ReLU

### 缺点
- 计算代价高
- 理论解释相对复杂

### 适用场景
- 高精度 CNN
- 目标检测与视觉理解任务
