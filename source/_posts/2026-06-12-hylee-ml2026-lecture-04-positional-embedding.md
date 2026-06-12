---
title: "【ML 2026 Spring 第4讲】模型如何处理超长输入：位置编码的前世今生"
date: 2026-06-12 12:04:00
categories: ["AI", "Course"]
tags:
 - ML2026
 - Positional Embedding
 - RoPE
 - ALiBi
 - Long Context
 - Length Extrapolation
description: "李宏毅 ML 2026 Spring 第4讲：从 Sinusoidal 到 RoPE 到 ALiBi，拆解位置编码如何让 Transformer 处理超长序列，以及长度外推的核心挑战。"
mathjax: true
---

> 课程：李宏毅 Machine Learning 2026 Spring  
> 讲次：第 4 讲（3/27）  
> 主题：深入模型内部架构 — 模型如何处理超长输入  
> 课程影片：[Positional Encoding & Long Context](https://youtu.be/Ll-wk8x3G_g)  
> 课程页：[ML 2026 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)  
> 前一讲：[第3讲：如何加快模型推理速度](/xiaoguang-blog/2026/06/05/hylee-ml2026-lecture-03-inference-optimization/)  
> 延伸阅读：[小光 TECH：KV Cache 量化技术解析](/xiaoguang-blog/2026/06/05/2026-06-05-kv-cache-quantization-varn/) \| [MLA 注意力机制拆解](/xiaoguang-blog/2026/06/04/2026-06-04-mla-attention-kv-cache/)

---

## TL;DR

Transformer 的 Self-Attention 本质上是<strong>置换等变（permutation-equivariant）</strong>的——打乱输入 token 的顺序，输出只会按相同方式打乱，每个 token 的输出值不变。这意味着 Attention 天然不知道"谁在前谁在后"。本讲从零拆解位置编码的完整演进：Sinusoidal PE 用三角函数给每个位置一个多维指纹 → 发现绝对位置不够，相对位置才关键 → ALiBi 用一个极其简单的线性偏置惩罚远距离 token → RoPE 用旋转矩阵把相对位置织入 Q 和 K 的内积，统治了几乎所有现代开源模型 → 最新的 NoPE 研究甚至质疑：因果遮罩本身就隐含了顺序信息，是否真的需要显式位置编码？核心追问：<strong>如何让模型"看到"顺序？当序列长度超出训练范围时，模型还能正常工作吗？</strong>

---

## 前置知识

- Self-Attention 的计算过程：$Q, K, V$ 三个投影、$QK^\top$ 相似度矩阵、softmax 归一化、加权求和
- 基本的线性代数：矩阵乘法、旋转矩阵、复数表示
- 了解 LLM 的 context window 概念（第 2 讲 Context Engineering 涉及）
- 不需要预先知道任何一种位置编码方案——本讲从头讲起

---

## 本讲目标

本讲围绕一个核心问题展开：<strong>如何让 Transformer 知道输入 token 的顺序？</strong>具体涵盖以下方面：

1. 理解 Self-Attention 为什么天然缺失位置信息，以及为什么这需要被解决
2. 掌握 Absolute Positional Embedding（加法式）的设计动机、工作机制和局限
3. 理解 Relative Positional Embedding 的本质优势——直接编码 token 间距离
4. 深入 RoPE 的核心机制：旋转矩阵如何将相对位置融入 Q 和 K 的内积
5. 理解"Train Short, Test Long"问题：为什么模型在超出训练长度的序列上会失败，以及各种解决方案的演进逻辑
6. 思考一个激进的观点：是否真的需要显式位置编码？NoPE 的实验告诉我们什么

---

## 1. Self-attention 为什么需要位置编码

幻灯片以最直观的方式开门见山：Self-attention <strong>没有顺序信息</strong>。

回顾 Attention 的计算公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里的 $Q, K, V$ 都是从输入 token embedding $x_i$ 经过线性投影得到的。关键观察：<strong>如果你交换任意两个输入 token 的位置，输出矩阵中对应 token 的输出值不会改变</strong>——它们只是按交换后的新位置排列。这就是置换等变性（permutation equivariance）。

课程用一个中文例句把这个问题讲得非常透彻：

- 句子 A：<strong>"你打我"</strong>
- 句子 B：<strong>"我打你"</strong>

Self-attention 看到的 token 集合完全一样——都是你、打、我。如果不加入位置信息，Attention 计算出的 $QK^\top$ 矩阵对这两个句子是一模一样的（只是行和列重新排列），模型根本无法区分"你打我"和"我打你"——但在中文里，这恰恰是主客体的完全互换。

这个问题的本质是：Transformer 的原始设计把每个输入 token 视为一个<strong>集合中的元素</strong>，而不是一个<strong>序列中的元素</strong>。集合没有顺序，序列有。要让模型处理语言——这种天然带顺序的数据——我们必须显式地向模型注入顺序信息。

---

## 2. Absolute Positional Embedding — 给每个位置一个向量

PPT 第 5-6 页介绍 Absolute Positional Embedding（绝对位置编码）的最基本形式：<strong>把位置编码直接加到 token embedding 上</strong>。

对于序列中的第 $i$ 个 token：

$$
x_i^{\text{input}} = \text{Embedding}(\text{token}_i) + PE(i)
$$

其中 $PE(i) \in \mathbb{R}^d$ 是一个和 token embedding 同维度的向量，$\text{Embedding}(\text{token}_i)$ 是第 $i$ 个 token 的词向量。两者逐元素相加，然后送入 Transformer 的后续层。

这个设计的关键直觉是：<strong>位置信息"涂抹"在 token 的表征上</strong>，就像给每个词盖上一个位置印章。后续的 Self-Attention 层看到的输入已经包含了位置信息，可以在计算相似度时自然区分不同位置的 token。

加法式是最直观的位置编码策略，也是 Vaswani et al.（2017）和 BERT/GPT-2 采用的方案。但它的具体实现方式——$PE(i)$ 到底是什么——决定了其能力边界。

---

## 3. Sinusoidal Positional Embedding — 三角函数位置编码

这大概是整节课最优雅的一段推导（幻灯片第 7-12 页）。

### 3.1 公式定义

「Attention Is All You Need」中提出的 Sinusoidal PE，对位置 $pos$ 和维度索引 $i$：

$$
PE_{(pos,\, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,\, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

偶数维度用 sin，奇数维度用 cos，$d_{\text{model}}$ 是模型总维度。注意 $\theta_i = 1 / 10000^{2i/d_{\text{model}}}$ 是频率项——$i$ 越大，频率越低。

### 3.2 不同维度不同频率：秒针、分针、时针的类比

课程中用一个非常形象的比喻来解释这点：

- <strong>低维度（小 $i$，高频）</strong>：像秒针——转得快，对位置变化非常敏感。相邻 token（位置差 1）的编码值差异很大，能精细地区分邻近位置。
- <strong>中维度</strong>：像分针——转得慢一些，适合捕捉中等范围的相对位置模式。
- <strong>高维度（大 $i$，低频）</strong>：像时针——转得非常慢，几乎不随位置变化。用来编码宏观的位置范围（比如"前 100 个 token"和"后 100 个 token"的区别）。

多维度合在一起，形成了一个<strong>多维连续的位置指纹</strong>——每个位置 $pos$ 对应一个独一无二的 $d$ 维向量，而这个向量在不同的"频率通道"上捕捉不同粒度的位置信息。

### 3.3 为什么选 sin 和 cos？相对位置的线性性质

Transformer 论文提到的一个关键性质：<strong>$PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性函数</strong>。这来自三角函数恒等式：

$$
\sin(pos+k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)
$$

$$
\cos(pos+k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)
$$

理论上，这意味着模型可以通过学习适当的注意力权重来间接感知相对位置：如果模型能学会 $\sin(pos)$ 和 $\cos(pos)$ 的某种线性组合，它就自动获得了 $\sin(pos+k)$ 的信息——不需要显式编码相对位置。

### 3.4 Sinusoidal PE 的优势

- <strong>外推潜力</strong>：因为是确定性函数，理论上可以为任意 $pos$ 生成编码——即使 $pos$ 超出训练范围，函数也能计算（只是模型没见过而已）。
- <strong>无额外参数</strong>：没有可学习的位置向量，不增加参数量。
- <strong>平滑连续</strong>：位置相近的编码也相近，几何结构合理。

---

## 4. 相对位置才是关键，以及 Sinusoidal PE 的局限

### 4.1 猫吃了鱼 vs 鱼吃了猫

幻灯片第 13-15 页用另一个例子强调：<strong>相对位置有时比绝对位置重要得多</strong>。考虑：

- 句子 A："<strong>猫</strong>吃了<strong>鱼</strong>" — attention 从"吃"到"猫"的分数是 0.7，到"鱼"的分数是 0.01
- 句子 B："<strong>鱼</strong>吃了<strong>猫</strong>" — attention 从"吃"到"鱼"的分数是 0.7，到"猫"的分数是 0.01

从 Attention 角度看，这两个句子的区别仅在于：<strong>"猫"和"鱼"的相对位置交换了</strong>。"吃"前面那个词是主语，"吃"后面的词是宾语——这是纯粹基于相对位置的关系识别。绝对位置（"猫在位置 1" vs "猫在位置 3"）在这里不重要。

### 4.2 Sinusoidal PE 的局限

幻灯片第 16-22 页深入分析 Sinusoidal PE 的问题：

<strong>第一：内容与位置的交互是间接的。</strong> Sinusoidal PE 只是把位置向量加到 token embedding 上，位置信息和内容信息的交互发生在 $QK^\top$ 的计算中。但 $Q_i$ 和 $K_j$ 来自 $x_i + PE(i)$ 和 $x_j + PE(j)$ 的线性投影，位置的作用被"稀释"在了整个向量计算中——模型难以直接利用"$i$ 和 $j$ 之间的确切距离"这个信息。

<strong>第二：理论上的相对位置性质在实践中不可靠。</strong>虽然 $PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性函数，但这需要模型从数据中自己学会利用这个数学性质。经过多层非线性变换和 softmax 后，这个性质未必能被有效保留和利用。后续的实验表明，Sinusoidal PE 在实践中主要编码的是<strong>绝对位置</strong>，而非相对位置。

<strong>第三：长度外推困难。</strong>虽然在训练范围外也可以计算 $PE(pos)$，但模型在训练时从未见过 $pos=600$ 的正弦值模式。当推理时遇到训练范围以外的位置编码，模型的 Attention 计算会进入一个未经训练的分布区域——这就是外推（extrapolation）失败的根本原因。

---

## 5. Relative Positional Embedding — 直接编码相对位置

既然相对位置是关键而 Sinusoidal PE 又做不好，那么自然思路是：<strong>直接在 Attention 计算中注入相对位置信息</strong>。幻灯片第 23-26 页介绍两种代表性方案。

### 5.1 ALiBi：用线性偏置惩罚远距离

ALiBi（Attention with Linear Biases, Press et al., 2022）的设计哲学是极简主义：<strong>不需要学习任何位置参数，只在 Attention score 上减去一个与距离成正比的偏置</strong>：

$$
\text{Attention score}(i, j) = Q_i^\top K_j - m \cdot |i - j|
$$

$m$ 是每个 attention head 独立的斜率。对于 $H$ 个 head，斜率按几何级数分配：

$$
m_h = 2^{-8h/H}, \quad h = 1, 2, \dots, H
$$

这意味着：
- 有些 head 斜率大，只关注非常近的 token（比如相邻 1-2 个）
- 有些 head 斜率小，可以关注较远的 token

不同 head 自然形成了"近距关注"和"远距关注"的分工，无需学习。

ALiBi 的外推能力来自一个关键观察：<strong>线性偏置不依赖训练长度</strong>。训练时模型学会了"距离 $d$ 就减 $m \cdot d$"，推理时 $d$ 再大也只是继续应用同一规则——没有"未见过的位置编码值"这个概念。论文展示了从 1024 token 训练直接外推到 2048+ token 推理，性能几乎没有下降。

### 5.2 T5：可学习的相对位置偏置

Transformer T5（Raffel et al., 2020）采用了另一种思路：<strong>为每个可能的相对距离分配一个可学习的标量偏置</strong>。

具体做法是维护一个偏置查找表 $b_{|i-j|}$，为每个相对距离值分配一个可学习参数，然后加到 Attention score 上：

$$
\text{Attention score}(i, j) = Q_i^\top K_j + b_{|i-j|}
$$

对于超过训练中见过的最大相对距离的情况，T5 使用固定的外推值 $b_{\text{max}}$（即所有超过训练范围的相对距离共享同一个偏置）。

与 ALiBi 相比，T5 的方案更灵活（偏置值由数据驱动学习），但需要额外参数且外推能力受限——超出训练范围的相对距离只能用同一个固定值近似。ALiBi 的线性规则则可以在任意距离上平滑延续。

---

## 6. RoPE：旋转位置编码——把位置织入内积

幻灯片第 27-40 页是本讲篇幅最大的部分，也是整节课的技术核心。RoPE（Rotary Position Embedding, Su et al., 2021）在今天几乎统治了所有主流开源模型——LLaMA、Qwen、Mistral、DeepSeek 全部使用 RoPE 或其变体。

### 6.1 核心直觉：乘法优于加法

前几种方案（Sinusoidal、ALiBi、T5）要么是加法要么是偏置。RoPE 走了一条完全不同的路：<strong>用旋转矩阵修改 $Q$ 和 $K$ 的计算，让它们的内积自然包含相对位置信息</strong>。

为什么是旋转？因为旋转矩阵有一个关键数学性质：

对位置 $m$ 的 query $\tilde{Q}_m$ 和对位置 $n$ 的 key $\tilde{K}_n$ 分别施加位置相关的旋转：

$$
\tilde{Q}_m = R_m \cdot Q_m, \quad \tilde{K}_n = R_n \cdot K_n
$$

计算它们的内积：

$$
\tilde{Q}_m^\top \tilde{K}_n = (R_m Q_m)^\top (R_n K_n) = Q_m^\top R_m^\top R_n K_n = Q_m^\top R_{n-m} K_n
$$

最后的等式来自旋转矩阵的性质：$R_m^\top R_n = R_{n-m}$。注意结果中只出现了<strong>相对位置 $n-m$</strong>，绝对位置 $m$ 和 $n$ 都消掉了！这就是 RoPE 被归类为相对位置编码的原因。

### 6.2 具体实现：二维子空间旋转

RoPE 把 $d$ 维向量分成 $d/2$ 个二维子空间，对每个子空间 $(2i, 2i+1)$ 施加一个旋转角：

$$
\theta_i = 10000^{-2i/d}
$$

对位置 $m$，在子空间 $i$ 上逆时针旋转 $m\theta_i$ 弧度：

$$
\begin{pmatrix}
x_{2i}^{(m)} \\[2pt]
x_{2i+1}^{(m)}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\[2pt]
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\[2pt]
x_{2i+1}
\end{pmatrix}
$$

在实际代码实现中，RoPE 利用<strong>复数乘法的欧拉公式</strong> $e^{i\theta} = \cos\theta + i\sin\theta$ 来高效实现旋转，不需要显式构造旋转矩阵。核心操作只需逐元素乘加——计算复杂度几乎为零。

课程提供了 Colab 动手实验链接，可以直接运行观察 RoPE 的效果：
- Colab：[RoPE Demo](https://colab.research.google.com/drive/1rWDtAkScrb2K3tcprSTzwuRQo5bGiyKJ)

### 6.3 RoPE 不影响 KV Cache

这是 RoPE 的一个关键工程优势（幻灯片特别强调）。在推理时使用 KV Cache 加速时，$K$ 和 $V$ 会被缓存下来供后续 token 使用。很多位置编码方案需要修改缓存的 $K$ 值（因为新 token 的加入会改变相对位置），但 RoPE 的旋转是<strong>在投影为 $Q$ 和 $K$ 之后立即施加的</strong>，一旦旋转完成，$K$ 的 cache 值就不再需要修改。

具体来说，新 token 生成时的流程是：
1. 当前 token 的 hidden state 投影为 $Q_{\text{new}}$ 和 $K_{\text{new}}$
2. 对 $Q_{\text{new}}$ 和 $K_{\text{new}}$ 分别施加当前位置的旋转
3. $K_{\text{new}}$ 加入 KV Cache
4. $Q_{\text{new}}$ 与缓存中的所有 $K$ 计算 Attention

注意步骤 2 之后，$K_{\text{new}}$ 作为 KV Cache 的一部分就不再变化——这是 RoPE 和 KV Cache 完美兼容的关键。

### 6.4 距离越远 Attention 不一定越小

幻灯片中一个有趣的观察：<strong>RoPE 没有像 ALiBi 那样强制"距离越远 attention 越小"的先验</strong>。在 RoPE 中，token 之间的距离通过旋转角影响 $Q_m^\top R_{n-m} K_n$ 的值，但这个值不一定随 $|n-m|$ 单调递减——它取决于内容向量 $Q_m$ 和 $K_n$ 在旋转后的匹配程度。

这不是坏事。很多长上下文任务（如长文档问答）需要模型跨越很远的距离建立关联，强制衰减可能阻碍这种远距离信息整合。RoPE 把"是否关注远处 token"的决定权留给内容相似度，只提供相对位置的信息框架。

---

## 7. Train Short, Test Long — 长度外推的完整方案演进

这是课程第二大重点（幻灯片第 41-57 页），也是 RoPE 在实际部署中面临的最大工程挑战。问题很简单：<strong>训练时序列长度是 $L_{\text{train}}$（比如 2048），推理时需要处理 $L_{\text{test}} \gg L_{\text{train}}$（比如 32768），怎么办？</strong>

### 7.1 为什么会失败

RoPE 虽然理论上只依赖相对位置，但旋转角 $\theta_i = 10000^{-2i/d}$ 决定了每个维度的旋转速度。训练时模型见过的相对距离范围是 $[0, L_{\text{train}}]$，对应的旋转角范围是 $[0, L_{\text{train}} \cdot \theta_i]$。

当推理长度扩展到 $L_{\text{test}} > L_{\text{train}}$ 时：
- <strong>高频维度（小 $i$，大 $\theta_i$）</strong>：旋转角度超出训练范围，模型遇到"从未见过"的旋转模式——这是外推失败的主要原因。
- <strong>低频维度（大 $i$，小 $\theta_i$）</strong>：旋转角度仍在训练范围内，或超出幅度很小——低频维度问题不大。

也就是说，<strong>高频维度的"过旋转"是外推失败的核心原因</strong>。

### 7.2 Position Interpolation（PI）— 线性缩放位置索引

最直接的方案（Chen et al., 2023）：把推理时的位置索引按比例缩回训练范围。

设 scale factor $s = L_{\text{test}} / L_{\text{train}}$，将推理时的位置 $pos$ 替换为 $pos/s$：

$$
PE(pos) \rightarrow PE(pos / s)
$$

原来位置 4096 的编码现在变成了位置 2048/s 的编码——所有旋转角被等比例压缩回训练范围。

<strong>代价：需要额外微调。</strong> Position Interpolation 虽然是有效的起点，但直接缩放后模型性能通常下降，需要通过少量微调来适应新的位置分布。缩放后的"不同位置"变成了"更细粒度的区分"——这对于需要精确位置判别的任务（如代码生成中的缩进对齐）可能不够。

### 7.3 Frequency-Based Approach — 只压缩高频，保留低频

一个关键观察：<strong>不是所有维度都需要压缩。</strong>低频维度的旋转角本来就在训练范围内，压缩它们反而会损害模型对宏观位置结构的感知。

频率基础的思路是：<strong>只压缩高频维度（压缩到训练范围内），低频维度保持不变</strong>。具体做法是设定一个频率阈值，高于阈值的维度做位置插值，低于阈值的维度不处理。

这比均匀的 Position Interpolation 更精细，因为它保留了低频维度对长距离位置模式的感知能力，同时解决了高频维度的"过旋转"问题。

### 7.4 NTK-Aware Scaling — 调整 RoPE Base

NTK-Aware Scaling（bloc97, 2023）的思路更进一步：<strong>不修改位置索引，而是修改 RoPE 的频率参数</strong>。

RoPE 的频率由 base $\theta = 10000^{-2i/d}$ 中的 10000 决定。NTK-Aware 的方法是增大这个基值（比如从 10000 改到 500000 或更高），从而降低所有维度的旋转频率。调整后的频率分布使得在 $L_{\text{test}}$ 处的旋转角仍然在训练时的旋转角范围内。

这个方法的优势是<strong>不需要微调</strong>——只需修改 $\theta$ 基值，模型可以直接在更长的上下文上推理。实践中，将 Llama 的 base 从 10000 调整到 260000 左右可以实现 4K → 8K 的零训练外推；调整到 1000000+ 可以实现更大范围的扩展（但性能可能逐渐下降）。

### 7.5 YaRN — NTK-Aware + 温度缩放

YaRN（Peng et al., 2023）在 NTK-Aware Scaling 的基础上增加了一个关键组件：<strong>对 softmax 输入进行温度缩放</strong>。

当序列变长时，Attention 的 softmax 分布变得更加"平坦"——因为需要关注的 token 变多了，每个 token 获得的分量变小。YaRN 通过降低 softmax 温度（乘以一个小于 1 的因子）来补偿这一点，使得 attention 分布保持锐利。

结合 NTK-aware 的频率缩放和温度缩放，YaRN 在 Llama 系列上成功实现了从 4K 到 128K 的上下文窗口扩展，且保持良好性能。

YaRN 公式可以概括为：

$$
\text{Attention score}(i, j) \;=\; \frac{Q_i^\top R_{j-i}^{\text{(scaled)}} K_j}{\tau}
$$

其中 $R^{\text{(scaled)}}$ 使用 NTK-aware 调整后的频率，$\tau$ 是温度缩放因子。

### 7.6 Dynamic Scaling — 不需要压缩时就不压缩

Reddit LocalLLaMA 社区提出的一个实用洞察：<strong>如果推理时的实际序列长度较短（比如只用了 3000 token，而训练长度是 4096），根本不需要外推——直接用原版 RoPE 就行。</strong>

Dynamic Scaling（也称为 "Dynamic NTK"）的策略是：<strong>根据当前序列的实际长度动态决定是否以及如何调整 RoPE 频率</strong>。如果当前长度 $\le L_{\text{train}}$，不调整；如果超过 $L_{\text{train}}$，按比例调整。这避免了"短序列变差"的问题——均匀缩放或 NTK-aware 缩放是全局的，即使当前序列很短也会改变位置编码的语义，可能损害短序列性能。

### 7.7 LongRoPE — 进化搜索最优配置

LongRoPE（Ding et al., 2024）将问题推向极致：<strong>用进化算法（evolutionary search）搜索每个维度最优的 RoPE 频率缩放因子</strong>。

不再是统一缩放所有高频维度，而是搜索一个逐维度的缩放因子向量 $\{\lambda_1, \lambda_2, \dots, \lambda_{d/2}\}$，使得在目标长序列上的验证 loss 最小。通过进化搜索，LongRoPE 在 Llama-2 上实现了从 4K 到 2048K（2M）的上下文窗口扩展——这是目前已知的最远外推距离。

<strong>代价：需要额外的训练/微调。</strong>进化搜索本身不需要训练模型，但搜索到的最优缩放因子需要在长序列数据上微调验证。而且，搜索结果可能与具体的模型和下游任务相关，泛化性值得关注。

### 7.8 方案对比

| 方法 | 策略 | 额外训练 | 最大外推倍数 | 关键创新 |
|------|------|:---:|:---:|------|
| Position Interpolation | 均匀缩放位置索引 | ✓ | ~4x | 线性缩放到训练范围 |
| NTK-Aware | 调整 RoPE base $\theta$ | ✗ | ~8x | 零训练，改基值 |
| YaRN | NTK-Aware + 温度缩放 | 可选 | ~32x | 补偿长序列的 attention 平坦化 |
| Dynamic Scaling | 动态判断是否缩放 | ✗ | ~8x | 短序列不受影响 |
| LongRoPE | 进化搜索逐维度缩放 | ✓ | ~512x | 搜索最优配置突破极限 |

---

## 8. No Positional Embedding?! — 因果遮罩本身就是位置信息

幻灯片第 58-63 页提出了一个发人深省的问题：<strong>Self-attention 真的完全没有位置信息吗？</strong>

### 8.1 因果遮罩的不对称性

Decoder-only Transformer 使用的因果注意力遮罩（causal mask）本身提供了顺序信息的不对称性。考虑一个长度为 $L$ 的序列，causal mask 确保：
- 位置 $i$ 的 token 可以看到位置 $1, 2, \dots, i-1$ 的 token
- 位置 $i$ 的 token <strong>不能</strong>看到位置 $i+1, i+2, \dots, L$ 的 token

这带来了一个关键的不对称性：<strong>前面的 token 看不到后面的 token，但后面的 token 可以看到前面的 token</strong>。这种"视野"的不对称性本身就是一种位置信息——token 在序列中的位置决定了它能"看"到多少上下文。

### 8.2 NoPE 的实验发现

NoPE（Kazemnejad et al., 2023）进行了一系列消融实验，考察 Transformer 在不加任何位置编码时的表现。关键发现：

- <strong>在语言建模任务上</strong>：解码器架构在没有任何位置编码的情况下，困惑度仅下降几个点——程度远小于预期。这意味着因果遮罩 + 注意力机制的组合天然编码了一定量的位置信息。
- <strong>在需要精确位置的任务上</strong>：NoPE 明显差于有位置编码的模型。例如"复制"任务（输出 = 输入的全部或部分）、"计数"任务（统计某个 token 出现次数），以及需要精确句法结构理解的任务——这些任务对位置精确性要求高，因果遮罩提供的粗粒度位置信息不够用。

### 8.3 为什么因果遮罩能提供位置信息

直观理解：序列中不同位置的 token 看到的"过去"长度不同。

- 位置 1 的 token 看不到任何过去的 token——它的视野是 0
- 位置 50 的 token 可以看到前 49 个 token——视野是 49
- 位置 $L$ 的 token 可以看到前 $L-1$ 个 token——视野是 $L-1$

Attention 层的输出向量 $o_i$ 是 $v_1, v_2, \dots, v_i$ 的加权和。随着 $i$ 增长，$o_i$ 的分布和统计特性也在规律性变化。Transformer 的后续层可以通过学习 $o_i$ 的统计特性来间接推断 token 的大致位置。

但这是<strong>粗粒度的</strong>：模型能感知到"这是序列早期"vs"这是序列中期"vs"这是序列后期"，但无法精确区分位置 50 和位置 51。

### 8.4 这个发现的含义

NoPE 并不是说"位置编码没用"——它说的是一种更微妙的东西：<strong>因果遮罩天然提供了位置信息的"骨架"，而显式位置编码提供了"精度"</strong>。骨架让模型在大多数语言建模任务中不至于完全迷失，但要精确理解位置依赖关系（如句法结构、代码缩进、数学表达式的括号匹配），显式位置编码仍然是必要的。

---

## 小光总结：位置编码演进的四条主线

1. <strong>从绝对到相对</strong>：位置编码的核心矛盾是"绝对位置"和"相对位置"的信息需求权衡。语言中相对位置（谁在谁的前面/后面）通常比绝对位置（这是第几个 token）更重要，但早期的 Sinusoidal PE 和 Learned PE 主要编码绝对位置。RoPE 和 ALiBi 通过不同的机制（旋转和偏置）实现了相对位置编码，这比原始方案更接近问题的本质。

2. <strong>从加法到乘法</strong>：位置编码的操作方式从 Sinusoidal PE 的"加到输入上"进化到 RoPE 的"乘到 QK 内积中"。这不是实现细节的改变，而是设计哲学的转变——前者把位置当作独立特征拼接到向量上，后者把位置织入 Attention 计算的核心。RoPE 的成功说明，<strong>把位置信息深深嵌入模型的计算机制比浅层注入更有效</strong>。

3. <strong>从固定长度到动态长度</strong>：Train Short Test Long 问题推动了一系列技术——Position Interpolation、NTK-Aware Scaling、YaRN、Dynamic Scaling、LongRoPE。从线性插值到逐维度搜索，演化方向是越来越精细化地控制每个维度频率的缩放，同时尽可能减少或避免额外训练。

4. <strong>从必需到"也许不必"</strong>：NoPE 的研究提醒我们，因果遮罩本身已经编码了粗糙的位置信息。显式位置编码解决的是"精度"问题，而非"有无"问题。这也解释了为什么 ALiBi 的简单线性偏置就能有效——它只需要补充因果遮罩本身无法提供的精确距离信息即可。

<strong>小光的判断</strong>：在实际工程中，RoPE + (Dynamic) NTK-Aware Scaling 是当前最成熟的长上下文方案——它兼顾了推理效率（不破坏 KV Cache）、外推灵活性和性能。ALiBi 在"从短训练直接外推"的场景下仍然是值得考虑的选项，尤其是在训练资源有限时。而 NoPE 的发现不应被理解为"位置编码不重要"，而应理解为"即使不加位置编码，因果遮罩的 backbone 也在工作"——这使得位置编码从一个"生死攸关"的问题变成了一个"精度优化"的问题。

---

## 课后思考

1. RoPE 只对 $Q$ 和 $K$ 施加旋转，$V$ 保持不变。如果把旋转操作也施加到 $V$ 上，会对 Attention 输出产生什么影响？从 $O = \text{softmax}(QK^\top)V$ 的结构出发分析：$V$ 编码的是被聚合的<strong>内容</strong>信息，位置信息的注入点应该在"决定关注什么"的阶段（$QK^\top$），还是"聚合内容"的阶段（$V$ 加权求和）？

2. 在一个使用 RoPE + Dynamic NTK-Aware Scaling 的模型中，如果用户的一次对话在短文本（2000 token）和长文本（50000 token）之间频繁切换，Dynamic Scaling 的触发可能会导致模型在两种"位置模式"之间反复跳变。这是否会引入不稳定？如何设计更平滑的过渡机制？

3. NoPE 的实验表明因果遮罩天然提供位置信息。如果把 causal mask 替换为 sliding window attention，位置信息的骨架是否会改变？这意味着 Sliding Window + RoPE（如 Mistral 的配置）中，位置编码的一部分负担实际上由 attention pattern 来分担了吗？这对设计新的高效 attention 机制有什么启发？

---

## 参考资料

<strong>课程影片</strong>：[李宏毅，〈深入模型內部架構：模型如何處理超長輸入〉，ML 2026 Spring](https://youtu.be/Ll-wk8x3G_g)

<strong>课程页</strong>：[ML 2026 Spring，李宏毅，NTU](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)

<strong>Sinusoidal PE</strong>：[Vaswani et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762)

<strong>RoPE</strong>：[Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2021](https://arxiv.org/abs/2104.09864)

<strong>ALiBi</strong>：[Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", ICLR 2022](https://arxiv.org/abs/2108.12409)

<strong>T5 相对位置编码</strong>：[Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020](https://arxiv.org/abs/1910.10683)

<strong>Position Interpolation</strong>：[Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation", arXiv 2023](https://arxiv.org/pdf/2306.15595)

<strong>NTK-Aware Scaling</strong>：[bloc97, "NTK-Aware Scaled RoPE", Reddit r/LocalLLaMA, 2023](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/)

<strong>YaRN</strong>：[Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", arXiv 2023](https://arxiv.org/abs/2309.00071)

<strong>Dynamic Scaling</strong>：["Dynamically Scaled RoPE", Reddit r/LocalLLaMA, 2023](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/)

<strong>LongRoPE</strong>：[Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", arXiv 2024](https://arxiv.org/abs/2402.13753)

<strong>NoPE</strong>：[Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers", NeurIPS 2023](https://arxiv.org/pdf/2305.19466)

<strong>RoPE 特性分析</strong>：[RoPE Properties Survey, arXiv 2024](https://arxiv.org/pdf/2410.06205)；[RoPE Theory Analysis, arXiv 2025](https://arxiv.org/pdf/2512.12167)

<strong>Context Extension 综述</strong>：[Aman Arora, "RoPE Context Extension — A Practical Guide", 2025](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html)

<strong>RoPE Colab 动手实验</strong>：[Google Colab — RoPE Demo](https://colab.research.google.com/drive/1rWDtAkScrb2K3tcprSTzwuRQo5bGiyKJ)

<strong>RoPE 中文讲解</strong>：[苏剑林，〈Transformer 升级之路：2、博采众长的旋转式位置编码〉，科学空间](https://spaces.ac.cn/archives/8265)
