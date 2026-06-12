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

Transformer 的 Self-Attention 本身是<strong>置换等变（permutation-equivariant）</strong>的——如果把输入 token 顺序打乱，输出也会按相同方式打乱，但每个 token 的输出内容不变。这意味着 Attention 不知道 "第 1 个 token 在第 1 位"。本讲从头梳理位置编码的演进：Sinusoidal 用三角函数注入绝对位置 → RoPE 用旋转矩阵把位置编码织入 Attention 计算，获得相对位置感知和长度外推能力 → ALiBi 用一个简单的线性偏置直接惩罚远距离 Attention，甚至不需要可学习的位置参数。核心要回答的问题是：<strong>Transformer 如何"知道"哪个 token 在前面？当序列长度超出训练范围时，它还能正常工作吗？</strong>

---

## 前置知识

- Self-Attention 的计算过程：$Q, K, V$ 三个投影、$QK^\top$ 相似度矩阵、softmax 归一化、加权求和
- 基本的线性代数：矩阵乘法、旋转矩阵、复数表示
- 了解 LLM 的 Context Window 概念（第2讲 Context Engineering 中讨论过）
- 不需要预先知道任何一种位置编码方案——本讲会从零讲起

---

## 本讲目标

1. 理解为什么 Transformer <strong>必须</strong>有位置编码
2. 掌握 Sinusoidal Positional Encoding 的设计动机和局限
3. 理解 RoPE 的核心思想——以乘法方式编码相对位置
4. 理解 ALiBi 的简单与强大——无需可学习参数
5. 建立「长度外推（Length Extrapolation）」的直觉：为什么训练 2048 token 的模型无法直接处理 8192 token？

---

## 1. Attention 的「位置盲」问题

先看 Self-Attention 的计算公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里的 $Q, K, V$ 都是从输入 token embedding $x_i$ 通过线性投影得来的。关键在于：<strong>如果你把 $x_1$ 和 $x_3$ 交换，$QK^\top$ 矩阵的对应行和列也会交换，但每个位置的 attention 值本身不会变化。</strong> 这就是置换等变——Attention 只看「token 的内容」，不看「token 的位置」。

用一个例子就能看清问题：

```text
输入 A：「我不喜欢这部电影」→ 负面
输入 B：「我喜欢这部电影，不」→ Attention 看到的是同样的 token 集合
但语义完全不同——关键就在于「不」在哪个位置
```

所以 Transformer 需要一个机制把<strong>位置信息注入到 token 表示中</strong>，让 Attention 在计算相似度时能区分「第 1 个 '不'」和「第 5 个 '不'」。

---

## 2. Sinusoidal Positional Encoding：用三角函数注入位置

「Attention Is All You Need」（Vaswani et al., 2017）的原始方案。对于位置 $pos$ 和维度索引 $i$：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

使用方式很简单：<strong>把 $PE$ 直接加到 token embedding 上</strong>：

$$
x_i^{\text{input}} = \text{Embedding}(token_i) + PE(i)
$$

### 为什么用 sin/cos？

核心原因有两个：

<strong>第一，不同维度有不同频率。</strong> 低维度（小 $i$）对应高频正弦波——对位置变化敏感，能区分相邻 token。高维度（大 $i$）对应低频——变化缓慢，能编码大范围位置模式。这就像一个多尺度的位置指纹，每个维度抓取不同粒度的位置信息。

<strong>第二，相对位置可以用线性变换表示。</strong> $PE(pos+k)$ 可以被表示为 $PE(pos)$ 的线性函数——这个性质来自三角恒等式，也是 Transformer 论文提到的一点。理论上，模型可以通过学习适当的权重来利用这种相对位置关系。

### 直觉理解

想象你在给每个位置分配一个独特的二进制编码（类似计算机里用二进制表示数字），但用的是连续的、平滑的正弦波代替离散的 0/1。位置 0 和位置 1 的编码在「高频」通道差异很大，位置 0 和位置 100 的编码在「低频」通道差异明显——这是一个<strong>连续的、多维的「位置指纹」</strong>。

### 局限

1. <strong>绝对位置主导</strong>：虽然理论上 $PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性函数，但经过模型的多层非线性变换后，这个性质未必能被有效利用。实践中，Sinusoidal PE 编码的主要是绝对位置。
2. <strong>长度外推差</strong>：如果一个模型只在 512 长度的序列上训练，$pos=600$ 的编码是模型从未见过的——这是外推（extrapolation）问题。模型不知道该怎么处理没见过的位置编码模式。
3. <strong>固定不可学习</strong>：不像 Learned Positional Embedding，Sinusoidal PE 不会根据任务调整。

> <strong>小光注</strong>：Sinusoidal PE 在 2017 年是优雅的解决方案，但今天回头看，它的设计过于依赖「模型能自己从三角函数中学会相对位置关系」的假设。后续的工作（RoPE、ALiBi）正是因为发现这个假设不太可靠，才选择了更直接的方式注入相对位置信息。

---

## 3. Learned Positional Embedding：让梯度决定位置

给每个位置分配一个可学习的向量 $p_i \in \mathbb{R}^d$，就和 token embedding 一样存一个 lookup table：

$$
x_i^{\text{input}} = \text{Embedding}(token_i) + p_i
$$

BERT 和 GPT-1/GPT-2 都用这个方案。优点是灵活——让梯度决定最好的位置表示。缺点是：

- <strong>受限于最大训练长度</strong>：如果训练时最大位置是 $512$，那 $p_{600}$ 根本不存在，无法外推。
- <strong>无法编码相对位置</strong>：$p_5$ 和 $p_6$ 之间的关系没有结构化约束，完全靠数据驱动学习。

---

## 4. RoPE：旋转位置编码——乘法优于加法

RoPE（Rotary Position Embedding）由苏剑林等人在 2021 年提出，现在几乎统治了所有主流开源模型（LLaMA、Qwen、Mistral、DeepSeek）。它的核心思想极其优雅：<strong>用旋转矩阵，把位置信息"织"进 Attention 的内积计算中，而不是简单地"加"到输入上。</strong>

### 为什么乘法优于加法？

加法式编码（Sinusoidal / Learned）把位置信息塞进输入向量里，希望 Attention 自己学会区分。但 Attention 本质是在算 $Q$ 和 $K$ 的内积——

$$
\text{Attention score}(i, j) \propto Q_i^\top K_j
$$

如果位置编码是加在 $x$ 上的，那 $Q_i$ 和 $K_j$ 的相似度取决于「输入加上位置」后的结果——位置的作用是间接的。

RoPE 的设计更直接：<strong>修改 $Q$ 和 $K$ 的计算方式，让内积结果天然包含相对位置信息。</strong>

### 核心机制

RoPE 在计算 Attention 前，对 $Q$ 和 $K$ 分别施加一个依赖于位置的旋转：

$$
\tilde{Q}_m = R_m \cdot Q_m, \quad \tilde{K}_n = R_n \cdot K_n
$$

其中 $R_m$ 是一个旋转矩阵（block-diagonal，每个 2D 子空间一个旋转角）。关键的魔术在于：

$$
\tilde{Q}_m^\top \tilde{K}_n = Q_m^\top (R_m^\top R_n) K_n = Q_m^\top R_{n-m} K_n
$$

消掉了绝对位置 $m$ 和 $n$，只留下相对位置 $n-m$！这就是<strong>相对位置编码</strong>的本质。

### 具体实现

RoPE 把 $d$ 维向量分成 $d/2$ 个二维子空间，对每对维度 $(2i, 2i+1)$ 施加旋转角 $\theta_i$：

$$
\theta_i = 10000^{-2i/d}
$$

对位置 $m$，将 $(x_{2i}, x_{2i+1})$ 逆时针旋转 $m\theta_i$ 弧度：

$$
\begin{pmatrix}
x_{2i}^{(m)} \\[2pt]
x_{2i+1}^{(m)}
\end{pmatrix}
=
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\[2pt]
x_{2i+1}
\end{pmatrix}
$$

实际实现可以非常高效——用复数乘法或逐元素操作，不需要显式构造旋转矩阵。

### 为什么 RoPE 支持外推？

这是 RoPE 最重要的工程价值。假设模型在长度为 $L_{\text{train}}$ 的数据上训练。推理时想要处理长度 $L_{\text{test}} > L_{\text{train}}$——这叫「长度外推」。

RoPE 之所以有外推潜力，是因为 Attention score 只依赖<strong>相对距离</strong> $n-m$。模型在训练时见过各种相对距离（从 $1$ 到 $L_{\text{train}}$），如果 $L_{\text{test}}$ 中新出现的 token 和附近的 token 之间的相对距离仍在训练范围内，Attention 就能正常工作。

但「远处的 token 之间的相对距离超出训练范围」仍然是个问题——这就是 NTK-aware scaling、YaRN 等方法要解决的。

### NTK-aware Scaling 与 YaRN

当 $L_{\text{test}} \gg L_{\text{train}}$ 时，高频维度的旋转角变化过大，低频维度的旋转角几乎不变。NTK-aware scaling（bloc97, 2023）的思路是<strong>调整 RoPE 的 base $\theta$</strong>（从 10000 改到更大的值），把高频「拉伸」以避免超出训练范围。

YaRN（Peng et al., 2023）进一步结合了 NTK-aware scaling 和温度缩放，在 Llama 系列模型上实现了从 4K 到 128K 的上下文窗口扩展。

> <strong>小光注</strong>：RoPE 的广泛采用不是偶然。它同时解决了三个问题：相对位置编码（提升泛化）、计算高效（用逐元素操作实现旋转）、和长度外推潜力（通过调整频率可扩展上下文窗口）。一个方法同时做到这三点，在 Transformer 位置编码的数十种方案中是独一档的。

---

## 5. ALiBi：最简单的位置编码——一个偏置搞定外推

ALiBi（Attention with Linear Biases, Press et al., 2022）可能是所有位置编码方案中最简单的一个——简单到你可能会怀疑它是否真的有效。

### 做法

不做加法，不做乘法，不需要可学习参数。只在 Attention score 上加一个<strong>线性偏置</strong>：

$$
\text{Attention score}(i, j) = Q_i^\top K_j - m \cdot |i - j|
$$

$m$ 是每个 head 独立的斜率（head-specific slope），不同 head 有不同的惩罚强度。距离越远，偏置越大，Attention score 越低。

### 斜率设计

对于 $H$ 个 head，斜率按几何级数分配：

$$
m_h = 2^{-8 \cdot h / H}, \quad h = 1, 2, \dots, H
$$

这意味着：有的 head 对距离非常敏感（大斜率 $m$，只关注很近的 token），有的 head 则不那么敏感（小斜率 $m$，可以关注较远的 token）。不同 head 的自然分工就这样形成了。

### 为什么有效

直觉很简单：语言中有「近邻优先」的结构——句子里相邻的词通常关系最紧密。ALiBi 用一个线性偏置来编码这种先验，越远的 token 自动获得越低的 attention weight。这既是位置编码，也是一种 inductive bias。

### 长度外推

ALiBi 的长度外推能力来自于一个简单的观察：<strong>线性偏置不依赖于训练长度</strong>。训练时模型学会了「看到距离 $d$ 就减 $m \cdot d$」，推理时距离 $d$ 变大只是继续应用同一个线性规则——没有「没见过」的位置编码。这篇论文展示了从 1024 token 训练直接外推到 2048+ token 推理的能力，几乎不掉点。

---

## 6. 更多相关方案一览

### xPos（Sun et al., 2022）

在 RoPE 的基础上引入指数衰减——对 $Q$ 和 $K$ 的旋转幅度乘以 $\gamma^{m}$ 和 $\gamma^{-n}$（$\gamma<1$），让远距离 token 的 attention score 指数级减小。可以看作是 RoPE + 距离衰减的组合。

### NoPE（Kazemnejad et al., 2023）

一个有趣的消融实验：<strong>不加任何位置编码，只用 causal mask。</strong> 实验发现，Decoder-only Transformer 在不加位置编码时也能学到一定的位置信息——causal mask 本身提供了不对称性（前面的 token 可以看到后面的？不，只有后面可以看到前面的）。但 NoPE 在需要精确位置的任务（如计数、复制）上表现明显差于有位置编码的模型。

### 外推方法一览

| 方法 | 思路 | 额外训练？ |
|------|------|:---:|
| NTK-aware | 调整 RoPE base，拉伸高频 | ✗ |
| YaRN | NTK-aware + 温度缩放 | ✗（微调可选） |
| Position Interpolation | 线性插值缩放位置索引 | ✓ |
| ReRoPE | 窗口内用 RoPE，窗口外用 ALiBi 式衰减 | ✗ |
| Self-Extend | 双 RoPE：局部用原频率、全局用压缩频率 | ✗ |
| LongRoPE | 进化搜索最优 RoPE 频率配置 | ✓ |

---

## 7. 方案对比总结

| 方法 | 编码方式 | 相对/绝对 | 可学习 | 外推能力 | 代表性模型 |
|------|:---:|:---:|:---:|:---:|------|
| Sinusoidal | 加法 | 绝对为主 | ✗ | 弱 | 原始 Transformer |
| Learned | 加法 | 绝对 | ✓ | 无 | BERT, GPT-2 |
| <strong>RoPE</strong> | <strong>乘法（旋转）</strong> | <strong>相对</strong> | ✗ | <strong>强（可调节）</strong> | <strong>LLaMA, Qwen, Mistral, DeepSeek</strong> |
| <strong>ALiBi</strong> | <strong>偏置（减法）</strong> | <strong>相对</strong> | ✗ | <strong>最强（零训练）</strong> | <strong>BLOOM, MPT</strong> |
| NoPE | 无 | — | — | — | (消融实验) |

---

## 8. 与系列其他讲的关联

| 相关讲次 | 关联点 |
|----------|--------|
| 第1讲（AI Agent） | Agent 的长对话历史是 RoPE 外推的直接应用场景 |
| 第2讲（Context Engineering） | Position Encoding 决定了模型的 context window 边界 |
| 第3讲（Inference Optimization） | KV Cache 大小随序列长度线性增长，外推能力影响缓存压力 |
| 第3讲（Sliding Window） | Sliding Window + RoPE 是 Mistral 的标准配置 |

---

## 9. 本讲的代码视角与动手实验

> ⚠️ 未见本讲的官方 Colab / 代码链接。以下内容基于小光对位置编码开源实现的熟悉程度整理，供读者动手参考。

本讲以概念和数学推导为主。如果你想从代码层面理解位置编码，以下两个方向值得探索：

### 1. RoPE 的实现（推荐看 Llama 源码）

RoPE 的核心代码极其紧凑。以 HuggingFace Transformers 中 Llama 的实现为例，关键函数是 `rotate_half` 和 `apply_rotary_pos_emb`：

```python
def rotate_half(x):
    """把向量分成两半，交换并取负——实现二维旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """对 Q 和 K 施加旋转"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

这里用复数的欧拉公式 $e^{i\theta} = \cos\theta + i\sin\theta$ 来高效实现二维旋转——复数乘法中的旋转等价于这个 `rotate_half` 操作。重点观察：
- `cos` 和 `sin` 是预先计算的位置相关旋转角的三角函数值
- 旋转只对 $Q$ 和 $K$ 执行，$V$ 保持不变
- 整个操作是逐元素的，不需要矩阵乘法——这就是为什么 RoPE 几乎没有计算开销

### 2. ALiBi 的实现

ALiBi 更简单——在 $QK^T$ 结果上加一个预计算的偏置矩阵：

```python
# 预计算偏置矩阵 (L, L)，B[i,j] = -m * |i-j|
bias = -slope * torch.abs(
    torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
)
# 对 causal mask 区域加偏置
attn_weights = attn_weights + bias
```

> <strong>小光注</strong>：如果你能亲手实现一遍 RoPE（哪怕是 10 行 NumPy），对旋转编码的理解会有质的飞跃。也推荐阅读苏剑林的博客《Transformer 升级之路：RoPE》，中文讲解清晰深入。

---

## 小光总结：位置编码的选择是「长度外推」和「表示能力」的权衡

这节课的主线不是「哪种位置编码最好」，而是「位置编码在不同场景下的 engineering trade-off」：

1. Sinusoidal PE 优雅但有外推局限——它是 2017 年的正确答案，但今天的模型需要处理远超训练长度的序列。
2. Learned PE 灵活但完全没有外推能力——BERT 年代的好选择，但不适合今天 128K context 的 LLM。
3. RoPE 用<strong>乘法（旋转）代替加法</strong>，把相对位置编码融入 Attention 内积，成为当前主流。它的成功在于：计算高效、相对位置感知、外推可通过调参（NTK-aware、YaRN）实现——三位一体。
4. ALiBi 用最简单的线性偏置实现零训练外推，适合对 simplicity 和 extrapolation 有极致要求的场景，但在绝对位置敏感的任务上可能弱于 RoPE。

我的判断是：位置编码的选择不是一个独立决策。它和模型架构（MHA vs GQA vs MLA）、训练长度、推理时的 target context length、以及是否需要在多个长度之间做灵活切换都紧密相关。比如：

- 如果你训练一个 128K 上下文的模型，RoPE + YaRN 是当前最成熟的选择。
- 如果你需要从 4K 训练直接外推到 32K 推理，ALiBi 可能是更安全的起点。
- 如果你在做消融实验或研究位置编码对特定任务的影响，「不加位置编码」的 NoPE baseline 值得跑一遍。

最终，位置编码从「加一个向量」到「旋转 Q,K」到「偏差 Attention score」，演进的方向是让位置信息越来越深地融入 Attention 计算本身——这是我觉得最值得记住的脉络。

---

## 课后思考

1. 如果把 RoPE 的旋转操作也施加到 $V$ 上会怎样？这和只旋转 $Q, K$ 有什么本质区别？提示：考虑 $O = \text{softmax}(QK^\top)V$ 中 $V$ 的作用。
2. ALiBi 的斜率是固定的几何级数。如果你要根据不同任务自动学习最优斜率，设计方案会是什么？会引入哪些风险？
3. 一个模型训练时使用了 RoPE，推理时想用 10 倍于训练长度的上下文。除了 NTK-aware scaling 和 YaRN，你还能想到哪些可能的策略？各自的代价是什么？

---

## 参考资料

1. <strong>课程影片</strong>：[李宏毅，〈深入模型內部架構：模型如何處理超長輸入〉，ML 2026 Spring](https://youtu.be/Ll-wk8x3G_g)
2. <strong>课程页</strong>：[ML 2026 Spring，李宏毅，NTU](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)
3. <strong>原始 Transformer Sinusoidal PE</strong>：[Vaswani et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762)
4. <strong>RoPE</strong>：[Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
5. <strong>ALiBi</strong>：[Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", ICLR 2022](https://arxiv.org/abs/2108.12409)
6. <strong>xPos</strong>：[Sun et al., "A Length-Extrapolatable Transformer", arXiv:2212.10554](https://arxiv.org/abs/2212.10554)
7. <strong>NTK-aware Scaling</strong>：[bloc97, "NTK-Aware Scaled RoPE", 2023](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
8. <strong>YaRN</strong>：[Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
9. <strong>NoPE</strong>：[Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers", NeurIPS 2023](https://arxiv.org/abs/2305.19466)
10. <strong>RoPE 中文讲解</strong>：[苏剑林，〈Transformer 升级之路：2、博采众长的旋转式位置编码〉，科学空间](https://spaces.ac.cn/archives/8265)
