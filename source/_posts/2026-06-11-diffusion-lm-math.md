---
title: "扩散语言模型的数学原理：从连续扩散到离散文本生成的完整推导"
date: 2026-06-11 08:00:00
mathjax: true
categories:
 - AI
 - LLM
tags:
 - Diffusion
 - Language Model
 - Discrete Diffusion
 - Masked Diffusion
 - LLaDA
description: "从DDPM连续扩散出发，完整推导离散扩散语言模型的数学原理：吸收态扩散、CMLM损失函数、采样策略，以及与自回归生成的概率分解对比。"
topic_id: "TECH-20260611-01"
---

> 阅读时间：约 25 分钟  
> 主题类型：TECH 技术点讲解 / 算法推导  
> 关键词：DDPM、Discrete Diffusion、D3PM、Absorbing State、MDLM、CMLM Loss、LLaDA、Masked Diffusion

## TL;DR

扩散语言模型的核心问题是将 DDPM 的连续高斯扩散框架，迁移到离散词表空间上进行文本生成。关键一步是：用 **吸收态（absorbing state）** 作为扩散的"噪声终点"——相当于把连续扩散的 $\mathcal{N}(0,I)$ 替换为 $\texttt{[MASK]}$ token。整个训练目标退化为带时间权重的掩码语言建模损失（CMLM loss），而采样过程则是从全掩码序列出发，逐步去掩码的逆向扩散过程。

本文的推导路线：DDPM 的 ELBO 分解 → D3PM 的离散转移矩阵 → 吸收态扩散的特殊形式 → MDLM 的简化损失 → LLaDA 的大规模验证。每条公式都配符号解释、直觉和张量形状，并给出可运行的伪代码。

## 前置知识

阅读本文需要的基础：

**DDPM 连续扩散**：将数据 $x_0$ 通过 $T$ 步逐步加高斯噪声得到 $x_T\sim\mathcal{N}(0,I)$，再训练一个去噪网络 $p_\theta(x_{t-1}|x_t)$ 逆向恢复 $x_0$ [1]。

**自回归语言模型**：将序列 $x=(x^1,\ldots,x^L)$ 的概率分解为链式条件概率：

$$p_\theta(x)=p_\theta(x^1)\prod_{i=2}^L p_\theta(x^i|x^1,\ldots,x^{i-1})$$

训练目标是逐位置的交叉熵损失，等价于最大化对数似然 [2]。

**Categorical 分布**：对于一个有 $K$ 个类别的离散随机变量，$x\sim\operatorname{Cat}(\pi)$ 中 $\pi\in\Delta^{K-1}$ 是概率单纯形上的向量，$\sum_{k=1}^K\pi_k=1$。这是离散扩散的数学基础。

## 问题定义：离散文本的扩散建模

给定词表 $\mathcal{V}$，$|\mathcal{V}|=K$，训练集包含 $N$ 条长度为 $L$ 的序列 $\{x^{(1)},\ldots,x^{(N)}\}$，其中每个 $x^i_j\in\{1,\ldots,K\}$ 是一个 one-hot 编码的 token。

**目标**：学习一个参数化分布 $p_\theta(x)$ 来逼近真实语言分布 $p_{\text{data}}(x)$，即最小化 KL 散度：

$$\min_\theta \operatorname{KL}(p_{\text{data}}(x)\ \|\ p_\theta(x))$$

**核心瓶颈**：DDPM 的前向过程使用高斯噪声 $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$，但 token 是离散的——你无法对 one-hot 向量"加一点高斯噪声"然后仍然得到一个合法的 token 分布。

我们需要一种"离散版的噪声添加"，使 $x_t$ 始终保持为 $K$ 维的合法概率分布（或 one-hot 的随机采样）。

## 连续扩散回顾：DDPM 的数学骨架

在进入离散扩散之前，先快速锁定 DDPM 的四个核心方程，它们将成为后续离散推导的"形态模板"。

**前向过程**（连续空间）：

$$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)$$

参数 $\beta_t\in(0,1)$ 控制每步噪声量。一次采样可从 $x_0$ 直接跳到 $x_t$：

$$q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I),\quad \bar{\alpha}_t=\prod_{s=1}^t(1-\beta_s)$$

其中 $\bar{\alpha}_t$ 递减，$\bar{\alpha}_T\approx 0$，于是 $x_T\approx\mathcal{N}(0,I)$。**关键性质**：给定 $x_0$ 和 $t$，$x_t$ 的分布有闭式解，不需要 $t$ 次迭代采样。

**逆向过程**（去噪）：

$$p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I)$$

其中 $\mu_\theta$ 是神经网络预测的均值，$\sigma_t^2$ 通常固定为 $\beta_t$ 或学习得到。

**训练目标**（简化版）：

$$\mathcal{L}_{\text{simple}}=\mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]$$

等价于预测添加的噪声 $\epsilon$而非直接预测 $x_0$。这个简化形式是实际训练中普遍使用的。

**ELBO 分解**（完整版）：

$$-\log p_\theta(x_0)\leq\underbrace{D_{\text{KL}}(q(x_T|x_0)\|p(x_T))}_{\mathcal{L}_T}+\sum_{t=2}^T\underbrace{D_{\text{KL}}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{\mathcal{L}_{t-1}}-\underbrace{\log p_\theta(x_0|x_1)}_{\mathcal{L}_0}$$

这一分解的结构——先验项 + 扩散项 + 重建项——在离散扩散中完全保留，只是每项的计算方式变了。

**直觉**：DDPM 将复杂的生成问题分解为 $T$ 个"小步去噪"子问题。每一步只预测一个条件高斯分布的均值，比直接建模整个数据分布容易得多。离散扩散继承了这一哲学，只是"噪声"和"去噪"的语义从高斯变成了 token 替换。

## 离散扩散的核心机制

### 从连续过渡矩阵到离散转移矩阵

DDPM 的前向过程由高斯条件分布 $q(x_t|x_{t-1})$ 定义，等价于一个线性变换：

$$x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

在离散空间，D3PM [3] 和 Multinomial Diffusion [4] 将这一思路推广为：**用转移矩阵 $Q_t\in\mathbb{R}^{K\times K}$ 定义离散的前向过程**。

对于每个 token 位置 $i$，前向过程是独立的 Categorical 分布：

$$q(x_t^i|x_{t-1}^i)=\operatorname{Cat}(x_t^i;\ x_{t-1}^i Q_t)$$

其中 $x_t^i$ 是位置 $i$ 在时刻 $t$ 的 one-hot 向量（长度为 $K$），$x_{t-1}^i Q_t$ 是一个长度为 $K$ 的概率向量，表示从当前 token 转移到其他 token 的概率。

**张量形状**：
- $Q_t$：$[K, K]$，每行是一个 Categorical 分布，$\sum_{k'} Q_t[k,k']=1$
- $x_t^i$：$[K]$ 的 one-hot
- $x_{t-1}^i Q_t$：$[K]$ 的概率向量

与连续扩散的关键对应：

| 概念 | 连续扩散 (DDPM) | 离散扩散 (D3PM) |
|---|---|---|
| 状态空间 | $\mathbb{R}^d$ | $\{1,\ldots,K\}$ |
| 噪声机制 | 加性高斯噪声 | 转移矩阵采样 |
| 前向算子 | $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ | $q(x_t|x_0)=\operatorname{Cat}(x_t;x_0\bar{Q}_t)$ |
| 噪声终点 | $\mathcal{N}(0,I)$ | 平稳分布 $\pi$（由 $Q_t$ 决定） |
| 逆向过程 | 预测噪声 $\epsilon$ | 预测原始 token $x_0$ |

其中累积转移矩阵 $\bar{Q}_t=Q_1Q_2\cdots Q_t$，同样满足一步跳转的闭式性质：

$$q(x_t|x_0)=\operatorname{Cat}(x_t;\ x_0\bar{Q}_t)$$

### 三种典型的前向转移矩阵

D3PM 提出了几种不同的 $Q_t$ 设计 [3]：

**均匀转移（Uniform）**：每个 token 以概率 $\beta_t$ 随机跳转到任意其他 token。

$$Q_t=(1-\beta_t)I+\frac{\beta_t}{K-1}(\mathbf{1}\mathbf{1}^\top-I)$$

**吸收态转移（Absorbing）**：每个 token 以概率 $\beta_t$ 变为一个特殊的 $\texttt{[MASK]}$ token（索引为 $K$，词表扩充为 $K+1$）。一旦进入吸收态就不会离开——就像马尔可夫链的吸收态。

$$Q_t=\begin{bmatrix}
1-\beta_t & 0 & \cdots & 0 & \beta_t \\
0 & 1-\beta_t & \cdots & 0 & \beta_t \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1-\beta_t & \beta_t \\
0 & 0 & \cdots & 0 & 1
\end{bmatrix}$$

最后一行对应 $\texttt{[MASK]}$ token 的"自环"——被 mask 了就永远是 mask。

**离散高斯转移（Discretized Gaussian）**：利用 embedding 空间的欧氏距离构建转移矩阵，近的 token 之间转移概率高，远的低。这是对连续空间高斯核 $\exp(-\|x-y\|^2/2\sigma^2)$ 的离散模拟。

**为什么吸收态是文本生成的最佳选择？** 小光判断有三个原因：

1. **匹配掩码语言模型的先验**：BERT 式的 MLM 已经证明"从上下文预测被遮住的词"是一个有效的预训练任务。吸收态扩散天然与此兼容。

2. **逆向过程有明确的语义**：去掩码（而不是"去随机替换"）是一个定义良好的操作——模型需要预测被 mask 位置的原始 token。而均匀转移的逆向则需要推理出"这个位置被随机替换成了什么"，噪声更大。

3. **采样可控**：从全 mask 到全 unmask 的过程类似于逐渐揭示文本，每一步的中间结果都可读（部分掩码的文本），便于调试和分析。

MDLM [5] 和 LLaDA [6] 都选择了吸收态扩散，实验结果也证实了这一选择优于均匀转移和离散高斯转移。

## 逐步推导

### Step 1：离散前向过程的闭式解

吸收态扩散中，累积转移矩阵 $\bar{Q}_t$ 有简洁的解析形式。设 $t$ 为连续时间（$t\in[0,1]$），每个 token 在时刻 $t$ 被 mask 的概率为 $t$，保持原样的概率为 $1-t$：

$$\bar{Q}_t=(1-t)I+t\cdot e_{\texttt{[M]}}\mathbf{1}_{\text{orig}}^\top$$

其中 $e_{\texttt{[M]}}$ 是 $\texttt{[MASK]}$ 的 one-hot 向量（长度 $K+1$），$\mathbf{1}_{\text{orig}}$ 是只有前 $K$ 个位置为 1 的向量。这个矩阵的意思是：

$$\bar{Q}_t[k,:]=\begin{cases}
(1-t)\text{ 集中在 }k,\quad t\text{ 集中在 }\texttt{[MASK]}, & k\neq\texttt{[MASK]} \\
100\%\text{ 停留在 }\texttt{[MASK]}, & k=\texttt{[MASK]}
\end{cases}$$

**张量形状**：$\bar{Q}_t$ 为 $[K+1, K+1]$，但只有前 $K$ 行用于原始 token 到混合状态的映射。

实践中，给定 $x_0$（one-hot，长度 $K$，不含 mask 维度），采样 $x_t$ 等价于：

```
对于每个位置 i:
    以概率 (1-t) 保留 x_0[i]
    以概率 t 替换为 [MASK]
```

这是吸收态扩散的**核心简洁之美**：前向过程退化为了独立伯努利掩码，不需要矩阵乘法。

### Step 2：逆向后验 $q(x_{t-\Delta t}|x_t,x_0)$

在连续扩散中，用贝叶斯公式可得闭式后验 $q(x_{t-1}|x_t,x_0)$，这是训练目标推导的关键。离散扩散中，给定"干净数据"$x_0$ 和"当前噪声"$x_t$，后验也有闭式解 [3, 5]。

对于吸收态扩散，这个后验极其简单：

- 如果 $x_t^i\neq\texttt{[MASK]}$，则 $x_{t-\Delta t}^i$ 必然等于同一 token（因为一旦 unmask 就再也不会被 mask）：
  $$q(x_{t-\Delta t}^i|x_t^i\neq\texttt{[M]},x_0^i)=\delta_{x_t^i}$$

- 如果 $x_t^i=\texttt{[MASK]}$，则 $x_{t-\Delta t}^i$ 要么仍为 mask，要么"跳回"到原始 token $x_0^i$：
  $$q(x_{t-\Delta t}^i|x_t^i=\texttt{[M]},x_0^i)=\frac{t-\Delta t}{t}\delta_{x_0^i}+\frac{\Delta t}{t}\delta_{\texttt{[M]}}$$

这个后验的直觉是：已知当前是 mask，且原始 token 已知，那么在上一个时刻，有 $(t-\Delta t)/t$ 的概率那位置仍是未 mask 的（值为 $x_0^i$），有 $\Delta t/t$ 的概率刚被 mask。

### Step 3：ELBO 分解与损失函数

将 DDPM 的 ELBO 框架平移到离散空间 [5]：

$$-\log p_\theta(x_0)\leq\mathbb{E}_{q}\left[\underbrace{-\log p_\theta(x_0|x_{t(0)})}_{\mathcal{L}_{\text{recons}}}+\sum_{i=1}^{T}\underbrace{D_{\text{KL}}(q(x_{s(i)}|x_{t(i)},x_0)\ \|\ p_\theta(x_{s(i)}|x_{t(i)}))}_{\mathcal{L}_{\text{diffusion}}}\right]+\underbrace{D_{\text{KL}}(q(x_1|x_0)\ \|\ p(x_1))}_{\mathcal{L}_{\text{prior}}}$$

其中 $s(i)=(i-1)/T$，$t(i)=i/T$，$T$ 是离散化步数。

由于先验 $p(x_1)$ 是"全 mask"的确定性分布，且 $q(x_1|x_0)$ 也收敛到全 mask（概率为 1），$\mathcal{L}_{\text{prior}}$ 项为 0。

对于**吸收态扩散**，$\mathcal{L}_{\text{diffusion}}$ 中的 KL 散度可以进一步简化。MDLM [5] 证明，使用 substitution-based 参数化（用 $p_\theta$ 预测 $x_0$ 而非直接预测 $x_{s(i)}$），可以得到：

$$\mathcal{L}_{\text{diffusion}}=\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i: x_t^i=\texttt{[M]}}-\log p_\theta(x_0^i|x_t)\right]$$

这就是 **CMLM（Continuous-time Masked Language Modeling）损失**。它的结构和你熟悉的 BERT MLM 损失一模一样——对被 mask 的位置计算交叉熵——唯一的区别是两个关键细节：

**$1/t$ 权重**：时间步 $t$ 越小（mask 越少），损失权重越大。直觉：当大部分 token 未被 mask 时，预测被 mask 的少数 token 相对容易，但梯度信号更弱；$1/t$ 权重补偿这种不平衡。如果不加这个权重（如 MaskGIT [7]），训练目标就不是有效的 ELBO。

**随机 $t\sim U[0,1]$**：训练时从均匀分布中随机采样 mask 比例，而不是固定 15%（如 BERT）。这使模型学习在任何噪声水平下去噪。

### Step 4：MDLM 的 Rao-Blackwellized 客观函数

MDLM [5] 发现可以将 CMLM 损失改写为不同时间步 MLM 损失的加权混合，从而降低梯度方差：

$$\mathcal{L}_{\text{RB-MDLM}}=\mathbb{E}_{x_0}\left[\sum_{m=1}^{L}w_m\cdot\mathcal{L}_{\text{MLM}}^{(m)}(x_0)\right]$$

其中 $\mathcal{L}_{\text{MLM}}^{(m)}(x_0)$ 是在 $x_0$ 上随机 mask $m$ 个 token 后的 MLM 损失，权重 $w_m$ 由连续时间积分解出。这是 Rao-Blackwellization 技术——通过对随机变量 $t$ 进行积分来降低估计方差，不改变期望值。

**实践要点**：尽管这个形式在理论上是优雅的，LLaDA [6] 在 8B 规模的实际训练中使用的就是 Eq.(3) 的原始形式，配合 $1/t$ 权重和均匀采样的 $t$。简单有效。

### Step 5：连续时间极限

当 $T\to\infty$（连续时间），离散扩散收敛到一个连续时间马尔可夫过程（CTMC），其生成元（generator）为：

$$R_t=\lim_{\Delta t\to 0}\frac{Q_{t,t+\Delta t}-I}{\Delta t}$$

对于吸收态扩散，$R_t$ 的非零项仅出现在从原始 token 到 mask 的转移：

$$R_t[k,\texttt{[M]}]=\frac{1}{1-t},\quad R_t[k,k]=-\frac{1}{1-t}$$

连续时间形式的损失为 [5]：

$$\mathcal{L}_{\text{CT}}=\mathbb{E}_{t\sim U[0,1],x_0,x_t}\left[\frac{1}{1-t}\sum_{i: x_t^i=\texttt{[M]}}-\log p_\theta(x_0^i|x_t)\right]$$

注意这里变成了 $1/(1-t)$ 而非离散时间中的 $1/t$——这是因为在连续时间框架下，时间方向的定义不同。但数学本质和直觉一致：mask 越少的样本贡献的梯度权重越大。

## 采样策略：从纯 Mask 到完整序列

训练完成后，生成文本的过程是逆向扩散：从全 mask 序列 $x_1$ 出发，逐步去掩码直到 $x_0$。

### 基础采样（Ancestral Sampling）

将连续时间 $t\in[1,0]$ 离散化为 $T$ 步（通常 $T=128$ 或 $256$）：

$$\Delta t=1/T,\quad t_k=1-k\cdot\Delta t$$

第 $k$ 步采样过程：

1. 输入 $x_{t_k}$（当前部分掩码的序列）到模型 $p_\theta(\cdot|x_{t_k})$
2. 对所有 $\texttt{[MASK]}$ 位置，获得预测分布 $p_\theta(x_0^i|x_{t_k})$（$[K]$ 维 logits）
3. 对每个 mask 位置，以概率 $\Delta t/t_k$ **重采样**（即解码出新的 token），以概率 $1-\Delta t/t_k$ **保持 mask**

伪代码：

```python
def sample(model, seq_len, T=128, vocab_size=K):
    # x: [seq_len], 初始化为全 [MASK]
    x = torch.full((seq_len,), MASK_ID, dtype=torch.long)
    
    for k in range(T):
        t = 1.0 - k / T  # 当前时间
        
        # 模型预测所有mask位置的logits
        logits = model(x)  # [seq_len, K]
        
        # 只更新mask位置
        mask_positions = (x == MASK_ID)
        
        # 重采样概率：Δ t / t = (1/T) / t
        remask_prob = (1.0 / T) / t
        
        for i in range(seq_len):
            if mask_positions[i]:
                if random.random() < remask_prob:
                    # 从预测分布采样
                    probs = softmax(logits[i], dim=-1)
                    x[i] = categorical_sample(probs)
                # 否则保持 [MASK]
    
    return x
```

**直觉**：$t$ 较大时（前期），$1/(T\cdot t)$ 很小，大部分 mask 保持不变——模型先"观望"整体结构。$t$ 较小时（后期），$1/(T\cdot t)$ 变大，大量 mask 位置被解码——利用已解码的上下文进行精确预测。

### 低置信度重掩码策略（LLaDA 的改进）

LLaDA [6] 提出一种改进策略：不是按固定概率随机"揭晓"mask，而是根据预测置信度决定哪些位置值得现在就解码。

```
每一步:
    1. 输入 x_t → 模型预测所有mask位置的分布
    2. 对每个mask位置，计算预测概率 p = max(softmax(logits_i))
    3. 选择 p > threshold 的位置解码（高置信度先解）
    4. 其余保持mask
    5. 使用余弦调度动态调整threshold
```

这种策略使采样更可控：高置信度的 token 先被"确定"，它们的上下文帮助低置信度位置的后续预测。实验表明这能提升生成质量 [6]。

### 半自回归采样（Semi-Autoregressive）

MDLM [5] 提出的 SAR 采样介于全并行扩散和逐 token 自回归之间：一次性解码一个 block 的 token，块内并行、块间自回归。这保留了扩散的并行性优势，同时比纯扩散更快。

```python
def sar_sample(model, seq_len, block_size=32, T=8):
    x = torch.full((seq_len,), MASK_ID)
    
    for block_start in range(0, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)
        # 对当前block做完整扩散去噪
        for k in range(T):
            t = 1.0 - k / T
            logits = model(x)
            mask_pos = (x[block_start:block_end] == MASK_ID)
            # ... 采样逻辑
        # 当前block完全解码，进入下一个block
    return x
```

## 概率分解对比：自回归 vs 扩散

理解扩散语言模型的本质，最直接的方式是对比两种建模的概率分解 [6]。

### 自回归分解

$$p_\theta(x)=p_\theta(x^1)\cdot p_\theta(x^2|x^1)\cdot p_\theta(x^3|x^1,x^2)\cdots p_\theta(x^L|x^1,\ldots,x^{L-1})$$

- **依赖方向**：严格从左到右，$x^i$ 只依赖 $x^{<i}$
- **生成方式**：逐 token 串行，第 $i$ 步的 KV cache 复用前 $i-1$ 步
- **训练信号**：每个位置都产生一个交叉熵损失
- **局限性**：无法利用右侧上下文（reverse curse [8]）；长序列的误差累积

### 扩散分解

$$p_\theta(x_0)=\int p_\theta(x_0|x_{t_k})p_\theta(x_{t_k}|x_{t_{k+1}})\cdots p_\theta(x_{t_{T-1}}|x_1)p(x_1)\ dx_{t_1}\cdots dx_{t_{T-1}}$$

其中 $x_1$ 是全 mask 序列。展开来看：

- **依赖方向**：双向。$p_\theta(x_0^i|x_t)$ 在预测位置 $i$ 时可以同时利用 $x_t$ 中左右两侧的 unmask token
- **生成方式**：全局并行。每一步更新所有 mask 位置（随机或基于置信度）
- **训练信号**：只对 mask 位置计算损失（CMLM），加权 $1/t$
- **优势**：双向上下文，天然处理填空和 reversal 任务；中间状态的文本始终可读

### 核心差异表

| 维度 | 自回归 (AR) | 扩散 (Diffusion) |
|---|---|---|
| 概率分解 | $p(x)=\prod_i p(x^i|x^{<i})$ | $p(x_0)=\int\prod_t p(x_{t-\Delta t}|x_t)$ |
| 条件方向 | 单向（左→右） | 双向（全上下文） |
| 训练效率 | 每个 token 都有梯度 | 仅 mask token 有梯度（≈$t$ 比例） |
| 推理步数 | $L$ 步（序列长度） | $T$ 步（通常 32-256） |
| 推理并行 | 串行，KV cache 限制 | 全局并行，无 KV cache |
| KV Cache | 必需 | 不需要（每次输入完整序列） |
| 填充任务 | 需要特殊处理 | 天然支持 |
| Reversal | 差 [8] | 好 [6] |

**小光判断**：扩散语言模型不是"替代自回归"，而是在不同**生成范式**中提供了一种有原则的选择。AR 推理已经被高度优化（FlashAttention、KV cache、投机解码），而扩散推理的高并行性可能在特定硬件（高吞吐批量生成）和特定任务（填空、reversal、可控生成）中更有优势。

## 最小例子

以下是吸收态扩散语言模型的**最小可运行训练例子**（PyTorch 风格，单批次、小词表）：

```python
import torch
import torch.nn.functional as F

# ===== 设定 =====
B, L, K, d = 2, 8, 100, 64      # batch, seq_len, vocab, hidden_dim
T = 128                           # 采样步数
MASK_ID = K                       # mask token 索引 (0-indexed)

# ===== 模型：简单的双向 Transformer =====
class MaskPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(K+1, d)  # +1 for MASK
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d, nhead=4, batch_first=True),
            num_layers=2
        )
        self.head = torch.nn.Linear(d, K)         # 输出词表logits
    
    def forward(self, x):
        # x: [B, L], 值域 {0..K} (K = MASK_ID)
        h = self.embed(x)                         # [B, L, d]
        h = self.transformer(h)                   # [B, L, d]
        return self.head(h)                       # [B, L, K]

# ===== CMLM 训练损失 =====
def cmlm_loss(model, x0):
    """
    x0: [B, L], 原始token序列 (值域 0..K-1)
    """
    B, L = x0.shape
    t = torch.rand(1).item()                     # U[0,1]
    
    # 前向过程：独立mask，概率 t
    mask = torch.rand(B, L) < t                  # [B, L] bool
    xt = x0.clone()
    xt[mask] = MASK_ID                           # [B, L]
    
    # 模型预测
    logits = model(xt)                           # [B, L, K]
    
    # 损失：仅对mask位置计算
    loss_per_pos = F.cross_entropy(
        logits[mask],                            # [n_masked, K]
        x0[mask],                                # [n_masked]
        reduction='none'
    )                                            # [n_masked]
    
    loss = (loss_per_pos.sum() / t) / (B * L)    # 1/t 加权 + 归一化
    return loss

# ===== 采样（基础 ancestral sampling）=====
@torch.no_grad()
def sample(model, seq_len=8, T=128):
    x = torch.full((1, seq_len), MASK_ID, dtype=torch.long)
    
    for k in range(T):
        t_cur = 1.0 - k / T                      # 从1递减到~0
        logits = model(x)                        # [1, L, K]
        probs = F.softmax(logits, dim=-1)        # [1, L, K]
        
        mask_pos = (x == MASK_ID)                # [1, L]
        remask_prob = (1.0 / T) / max(t_cur, 1e-8)
        
        for i in range(seq_len):
            if mask_pos[0, i] and torch.rand(1) < remask_prob:
                x[0, i] = torch.multinomial(probs[0, i], 1)
    
    return x

# ===== 训练循环 =====
model = MaskPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(1000):
    x0 = torch.randint(0, K, (B, L))            # 随机batch
    loss = cmlm_loss(model, x0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**关键张量形状追踪**：

| 步骤 | 张量 | 形状 | 含义 |
|---|---|---|---|
| 输入 | `x0` | $B\times L$ | 原始 token 序列 |
| 加噪 | `mask` | $B\times L$ | bool 掩码，True=被mask |
| 模型输入 | `xt` | $B\times L$ | $K$ 个原始 token + MASK_TOKEN |
| Embedding | $h$ | $B\times L\times d$ | 双向 Transformer 隐状态 |
| Logits | `logits` | $B\times L\times K$ | 仅输出 $K$ 个真实 token 的分数 |
| 损失计算 | `logits[mask]` | $N_m\times K$ | 仅被 mask 位置的预测 |
| 目标 | `x0[mask]` | $N_m$ | 被 mask 位置的原始 token |
| Loss | scalar | - | $\frac{1}{t}\cdot$平均交叉熵 |

## 局限与常见误解

### 局限

**1. 推理效率**：与自回归模型不同，扩散模型每一步都需要前向传播整个序列（而不是单 token），且无法使用 KV cache。对于 $T=128$ 步采样、序列长度 $L=2048$，总计算量约为自回归的 $T$ 倍（粗略估计）。这是扩散语言模型推理的主要瓶颈。

**2. 长度控制**：扩散模型的输出长度必须在采样前确定（等于初始 mask 序列的长度），无法像自回归模型那样动态生成 EOS token 终止。实践中可以通过固定长度 + 后处理截断来解决，但不如自回归自然。

**3. 训练效率**：CMLM 训练中，只有被 mask 的 token 产生梯度信号；$t=0.5$ 时有约一半的 token 被 mask，$t=0.1$ 时只有 10% 的 token 产生信号。相比自回归的训练链（每个 token 都有损失），扩散训练的有效梯度密度更低。

**4. 连续时间近似的离散化误差**：实践中我们仍用有限步数 $T$ 近似连续扩散，$T$ 太小会导致质量下降。LLaDA 使用 $T=128$ 已在 8B 规模验证可行 [6]，但更小的模型需要更多步数。

### 常见误解

**误解 1："扩散语言模型就是 BERT 加上生成"**。不完全对。BERT 使用固定 15% mask，没有时间维度和 $1/t$ 权重，其 MLM 目标不是有效的 ELBO。扩散模型的关键区别在于 (a) 随机 mask 比例 $t\sim U[0,1]$ 和 (b) $1/t$ 加权，这使得训练目标是一个严格的似然下界，从而具备生成能力。

**误解 2："吸收态扩散只能用 [MASK] token"**。吸收态是一个通用框架，可以是任意特殊 token。理论扩展包括多个吸收态（每个代表不同的噪声类型），但现有工作表明单个 [MASK] 是最简洁高效的选择。

**误解 3："扩散模型天然比自回归差"**。LLaDA 8B [6] 表明，在等价训练计算量下，扩散模型可以达到接近自回归基线的效果。性能差距更多来自工程成熟度（AR 有十多年的优化积累），而非原理缺陷。

**误解 4："逆向过程每一步都解所有 mask"**。实际采样时，每一步只解一部分 mask（由 $1/(T\cdot t)$ 概率控制），而不是一次性预测所有 mask 位置。那些尚未解码的位置保持 mask 状态，为后续步保留上下文。

## 小光总结

**技术上的五个关键洞察**：

1. **转移矩阵是桥梁**：$Q_t\in\mathbb{R}^{K\times K}$ 完成了"高斯噪声"到"离散替换"的语义迁移。DDPM 的所有公式形态在离散空间中都有对应，区别仅在于线性空间换成了概率单纯形。

2. **吸收态是 sweet spot**：理论上 D3PM 支持任意转移矩阵，但吸收态扩散既匹配了 MLM 的先验知识，又最小化了前向过程的熵增（相比均匀转移），使逆向任务（去掩码）更可学习。

3. **CMLM 损失 = 加权 MLM**：扩散语言模型的训练目标在形式上是你熟悉的掩码语言建模损失，但两个细节——随机 $t\sim U[0,1]$ 和 $1/t$ 加权——使其成为有效的似然下界，而非启发式预训练。

4. **概率分解决定了能力边界**：自回归的单向依赖限制了对 reversal 和填充任务的处理；扩散的双向条件依赖天然绕过这些限制，代价是失去了 KV cache 的推理加速。

5. **采样是一阶控制问题**：从全 mask 到全 unmask 的路径由 $1/(T\cdot t)$ 调度控制。稀疏步（早期小步）让模型建立全局结构，密集步（后期大步）细化局部细节。LLaDA 的低置信度重掩码是对这一调度的聪明白适应。

**小光判断**：扩散语言模型不是 AR 的直接替代品，而是生成式语言建模的一个新的理论支点。它在原理上可行（LLaDA 8B 已验证），但在工程上仍需大量优化才能达到 AR 的推理效率。未来最有趣的方向是：能否让扩散的"全局并行生成"与 AR 的"增量推理"相结合，各取所长。

## 参考资料

**DDPM**：[Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020](https://arxiv.org/abs/2006.11239)

**D3PM**：[Austin, Johnson, Ho, Tarlow, and van den Berg, "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021](https://arxiv.org/abs/2107.03006)

**Multinomial Diffusion**：[Hoogeboom, Nielsen, Jaini, Forré, and Welling, "Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions", NeurIPS 2021](https://arxiv.org/abs/2102.05379)

**MDLM**：[Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, and Kuleshov, "Simple and Effective Masked Diffusion Language Models", NeurIPS 2024](https://arxiv.org/abs/2406.07524)

**LLaDA**：[Nie, Zhu, You, Zhang, Ou, Hu, Zhou, Lin, Wen, Li, "Large Language Diffusion Models", arXiv 2025](https://arxiv.org/abs/2502.09992)

**SEDD**：[Lou, Meng, and Ermon, "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", ICML 2024](https://arxiv.org/abs/2310.16834)

**SSD-LM**：[Han, Kumar, and Tsvetkov, "SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control", ACL 2023](https://arxiv.org/abs/2210.17432)

**Reversal Curse**：[Berglund, Tong, Kaufmann, et al., "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'", arXiv 2023](https://arxiv.org/abs/2309.12288)

**Attention Is All You Need**：[Vaswani, Shazeer, Parmar, et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762)

**MDLM 官方代码**：[kuleshov-group/mdlm, GitHub](https://github.com/kuleshov-group/mdlm)

**LLaDA 官方代码与演示**：[ML-GSAI/LLaDA-demo, GitHub](https://github.com/ML-GSAI/LLaDA-demo)

---

> 注：DiffusionGemma 技术报告在本文写作时（2026年6月）尚未在 arXiv 或 Google AI Blog 找到公开版本。本文的扩散语言模型核心推导基于 D3PM、MDLM 和 LLaDA 的一手论文。

