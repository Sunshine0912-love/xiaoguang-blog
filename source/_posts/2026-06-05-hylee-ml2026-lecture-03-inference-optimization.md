---
title: "【ML 2026 Spring 第3讲】深入模型内部架构：如何加快模型推理速度"
date: 2026-06-05 12:30:00
categories: ["hylee ML 2026 Spring"]
tags:
 - ML2026
 - Flash Attention
 - KV Cache
 - Multi-Query Attention
 - Group-Query Attention
 - MLA
 - Inference Optimization
description: "李宏毅 ML 2026 Spring 第3讲：从自回归解码的 prefill/decode 阶段出发，深入 Flash Attention 的 io-aware tiling、KV Cache 的内存瓶颈、以及 MQA/GQA/MLA/Sliding Window 等注意力架构优化。"
mathjax: true
---

> 课程：李宏毅 Machine Learning 2026 Spring  
> 讲次：第 3 讲（3/20）  
> 主题：深入模型内部架构 — 如何加快模型推理速度  
> 课程影片：[Flash Attention (1/2)](https://youtu.be/vXb2QYOUzl4) \| [KV Cache (2/2)](https://youtu.be/fDQaadKysSA)  
> 讲议：[inference.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/inference.pdf)  
> 课前预习：[Transformer 基础](https://youtu.be/8iFvM7WUUs8)  
> 前一讲：[第2讲：Context Engineering](/2026/06/05/hylee-ml2026-lecture-02-context-engineering-multi-agent/)  
> 延伸阅读：[小光 TECH：KV Cache 量化技术解析](/2026/06/05/2026-06-05-kv-cache-quantization-varn/) \| [MLA 注意力机制拆解](/2026/06/04/2026-06-04-mla-attention-kv-cache/) \| [投机采样原理](/2026/06/05/2026-06-05-speculative-decoding-principles/)

---

## 本讲目标

1. 理解为什么自回归推理慢——问题不在算力，在**搬资料**
2. 掌握 Flash Attention 的核心思想：io-aware tiling
3. 掌握 KV Cache 的原理及其内存代价
4. 理解 MQA、GQA、MLA 如何从注意力结构角度压缩 KV Cache
5. 了解 Sliding Window、Streaming LLM、KV Cache Pruning 等进一步优化方向

## 前置知识

需要：Self-Attention 的计算过程（Q、K、V 三个投影、softmax、加权求和）  
建议先看：[Transformer 注意力机制基础](https://youtu.be/8iFvM7WUUs8)

---

## 1. 推理的两个阶段：Prefill 与 Decode

LLM 生成文本分为两个阶段：

```
Prefill（预填充）：
  输入 "李宏毅几班" → Transformer → 一次性算出所有位置的 K, V
                                          ↓
  拿到最后一个位置的隐藏状态 → 预测下一个 token

Decode（解码）：
  拿到新 token "大" → Transformer → 预测下一个 token → "金"
  拿到新 token "金" → Transformer → 预测 [END]
  ...（自回归，每次生成一个 token）
```

**核心区别**：

| | Prefill | Decode |
|---|---|---|
| 并行度 | 高（所有输入 token 一起算） | 低（每个 token 串行） |
| 瓶颈 | Compute-bound | **Memory-bandwidth bound** |
| 优化目标 | 提高吞吐 | 降低延迟 |

Decode 阶段每生成一个 token 都要搬运全部模型参数（以 70B 模型约 140GB FP16），但实际计算量极小。这就是为什么 "Inference Optimization" 的大多数技术都集中在 Decode 阶段。

---

## 2. Flash Attention：少搬资料

### 问题：Self-Attention 的内存访问模式

标准 Self-Attention 的计算：

$$
O=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

在 GPU 上的执行流程：

```
1. 从 HBM 读取 Q, K → 计算 QK^T → 写回 HBM（S 矩阵，L×L）
2. 从 HBM 读取 S → 计算 softmax → 写回 HBM（P 矩阵）
3. 从 HBM 读取 P, V → 计算 PV → 写回 HBM（O 矩阵）
```

**问题**：中间的 $S$ 和 $P$ 矩阵大小是 $L^2$（L 为序列长度）。当 L=4096 时，仅 S 矩阵就需要 $4096^2 \times 2 / 1024^2 \approx 32$ MB FP16。每一层都要读写一遍——这就是 Flash Attention 要解决的。

### 核心思想：io-aware tiling

李宏毅用了一个很形象的比喻：

> 仓库（HBM）里堆满数据，工作台（SRAM）很小。与其每次搬一小块算完写回去、再搬下一块，不如**在工作台上处理完整个 chunk 再写回**。

Flash Attention 的做法：

```
传统做法（多次读 HBM）：
  Q,K → HBM → 读到 SRAM → 算 S → 写回 HBM
  S → HBM → 读到 SRAM → 算 softmax → 写回 HBM
  P → HBM → 读到 SRAM → 算 PV → 写回 HBM
  共 3 次 HBM 读写（大矩阵）

Flash Attention（tiling）：
  把 Q,K,V 分成 chunks
  每个 chunk 在 SRAM 内完成：QK^T → softmax → PV
  → 只写回最终的 O 矩阵
  共 1 次 HBM 读写（仅写回结果）
```

具体来看：Flash Attention 把序列长度切成 B 个 block。对第 k 个 Q-block：

```
Q_k 在 SRAM 里 → 依次遍历所有 K,V blocks
  对每个 K_j, V_j:
    S_kj = Q_k × K_j^T (在 SRAM 中)
    P_kj = softmax(S_kj) (在 SRAM 中，使用 online softmax 技巧)
    O_k += P_kj × V_j (在 SRAM 中累加)
遍历完成 → 最终 O_k 写回 HBM
```

**online softmax** 是 Flash Attention 的另一个关键技巧：不需要等所有分数算完再做归一化，而是分段累加并动态调整——这保证了每个 chunk 在 SRAM 里就能独立完成计算。

### 需要读几遍？

李宏毅强调了一个关键观察：

> 传统实现 HBM 读了很多次大矩阵；Flash Attention 只需**读 2 次**（每个 K,V chunk 读一次）。

这不是巧合——这是 Flash Attention io-aware 设计的结果：Q 被加载一次后在 SRAM 里一直留着，K 和 V 每个 chunk 读一次，O 在 SRAM 里累加，最后写回一次。

### 代价

Flash Attention **不改变 Self-Attention 的数学结果**（数值上可能有微小差异但理论上等价），**不需要训练模型**。代价是：

1. 一点点额外运算（online softmax 的 rescaling）
2. 一点点"烧脑"——实现复杂，但算法思想清晰

---

## 3. KV Cache：用空间换时间

### 为什么需要 KV Cache？

Decoder 里的 Self-Attention 使用 causal mask——每个 query token 只能 attend 到前面的所有 token。所以生成第 t 个 token 时，第 1 到第 t-1 个 token 的 key 和 value 已经算过了：

```
生成 "大" ：Q("大") attend 到 K("李"),K("宏"),K("毅"),K("几"),K("班")
生成 "金" ：Q("金") attend 到 K("李"),K("宏"),K("毅"),K("几"),K("班"),K("大")
```

如果没有 KV Cache，每生成一个新 token 都要重新计算所有历史 K,V——这是 $O(L)$ 的重复计算。

**KV Cache 的做法**：每生成一个 token，把它的 K 和 V 存入缓存。后续 token 计算 attention 时直接从缓存读取，不用重算。

### 内存代价

KV Cache 占用的内存是多少？以 Gemma 2 为例：

```
每 token 的 KV Cache：
  46 层 × 32 heads × 128 dim × 2 (FP16) × 2 (K+V)
  = 753,664 bytes ≈ 736 KB = 0.72 MB

A100 80GB 的 KV Cache 容量：
  80 GB ÷ 0.72 MB/token ≈ 114,000 tokens
```

**这个数字很关键**：即便在 80GB 的 A100 上，纯 KV Cache 也只够存约 11 万个 token。对一个长对话（AI Agent 的 system prompt 就可能有几千 token），这个容量是真实瓶颈。

这也是为什么前几讲反复提到 Context Window 的重要性，以及为什么后续的 MLA、KV Cache 量化、KV Cache Pruning 等方法都是为了这个瓶颈而生。

---

## 4. Attention 结构优化：MQA → GQA → MLA

Flash Attention 解决了「怎么算得更快」，KV Cache 解决了「算什么」。接下来要解决的是：「KV Cache 能不能更小？」

### MHA（Multi-Head Attention）：N 个 head 的标配

```
每个 head 有独立的 Q, K, V:
  Q_1, K_1, V_1 (head 1)
  Q_2, K_2, V_2 (head 2)
  ...
  Q_H, K_H, V_H (head H)

KV Cache = 2 × H × d_h × L（每层、每个 head 存 K 和 V）
```

### MQA（Multi-Query Attention）：多个 Q 共享一套 K,V

```
Q_1, Q_2, ..., Q_H（每个 head 独立）
K, V（所有 head 共享）

KV Cache = 2 × 1 × d_h × L = 原本的 1/H
```

代价：**可能明显伤害模型能力**——所有 head 用相同的 K,V 意味着注意力模式的多样性被压缩。

### GQA（Group-Query Attention）：折中方案

```
将 H 个 head 分成 G 组，每组共享 K,V:
  Group 1: Q_1...Q_{H/G}, K_1, V_1
  Group 2: Q_{...}, K_2, V_2
  ...

KV Cache = 2 × G × d_h × L = 原本的 G/H
```

Llama、Gemma 等模型都使用了 GQA（通常 G=8）。

### MLA（Multi-head Latent Attention）

DeepSeek-V2/V3 的方案（我们已经写了专门的文章深入分析，见[延伸阅读](/2026/06/04/2026-06-04-mla-attention-kv-cache/)）：

```
不是让 K,V 共享 → 而是用低秩分解压缩 K,V:
  K = W_UK × (LoRA 压缩表示)
  V = W_UV × (LoRA 压缩表示)

存储的不是完整的 K,V，而是压缩后的潜变量
使用时动态展开
```

MLA 比 GQA 更激进地压缩了 KV Cache，且通过训练来保证压缩不伤害模型质量。

---

## 5. Sliding Window Attention：限制注意范围

另一种思路：不压缩每个 token 的表示，而是**限制每个 query 能看到的范围**。

### 基本方案（Mistral 7B）

```
每个 query 只 attend 到最近的 W 个 token:
  第 t 个 token 只看到 [t-W+1, t] 区间的 token

KV Cache = W × d_h（固定窗口大小，不随 L 增长）
```

### 变体：GPT-OSS 的混合窗口

GPT-OSS 允许一些层有全局长上下文、一些层用 Sliding Window——在局部信息和全局信息之间做 trade-off。

---

## 6. Streaming LLM：注意力下沉

论文(arXiv:2309.17453)发现一个现象：**Attention 分布中有一小部分 token 会持续获得极高的注意力权重**——这些 token 称为"attention sink"。

Streaming LLM 的思路：保留 attention sink token + 最近的 window token，丢弃中间的大部分。这让 KV Cache 在超长流式输入（对话、字幕生成）中保持恒定内存。

---

## 7. Pruning KV Cache：丢弃不重要的

Scissorhands(arXiv:2305.17118) 和 H2O(arXiv:2306.14048) 的发现：

```
每次只有小部分的 token 真的有 attention
少数 token 会反覆吸走大量 attention
→ 那些拿到极低 attention 的 token 可以安全丢弃
```

这类方法**不改 Attention 结构、不需要训练**，代价是**可能明显伤害模型能力**（丢错 token 时，长距离依赖断裂）。

---

## 8. 跨对话的 KV Cache：Prompt Caching

回到 AI Agent 的场景（呼应第 1、2 讲）：

```
每次 Agent 呼叫 LLM：
  System Prompt（几千 token，基本不变）
  + 对话历史（变化）
  + 新问题
```

System Prompt 的 KV Cache 可以**跨请求复用**：

```
第一次请求：计算并缓存 System Prompt 的 K,V
后续请求：从缓存直接读 System Prompt 的 K,V
→ 延迟显著降低，成本显著下降
```

OpenAI API 的 [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching/) 就是基于这个机制定价的——匹配到的 prefix 越长，折扣越大。

### 工程技巧：稳定的放前面

```
System Prompt:
  ├── 身份信息（SOUL.md, IDENTITY.md）→ 基本不变，放前面
  ├── 行为规则（AGENTS.md）→ 可能变动
  ├── 工具描述 → 可能变动
  └── 记忆摘要 → 每次不同

优化：最稳定的内容放 System Prompt 最前面
→ prefix match 最大化 → cache hit rate 最高
```

论文(arXiv:2601.06007)进一步研究了如何通过变量命名规范化（"从台北到波士顿" vs "从台北 x 到波士顿 y"）来提升跨请求的 cache 命中率。

---

## 9. 完整方法对比

李宏毅在讲义最后给出了一个很实用的总结表：

| 方法 | 改 Attention？ | 需训练？ | 其他代价 |
|------|:---:|:---:|------|
| Flash Attention | ✗ | ✗ | 额外运算 + 实现复杂 |
| KV Cache | ✗ | ✗ | 占用显存 |
| Multi-Query Attention | ✓ | ✓ | 可能明显损伤能力 |
| Group-Query Attention | ✓ | ✓ | — |
| MLA | ✓ | ✓ | — |
| Sliding Window | ✓ | ? | — |
| Streaming LLM | ✓ | ? | — |
| Pruning KV Cache | ✓ | ✗ | 可能明显损伤能力 |
| Speculative Decoding | ✗(理论) | ✗ | 小模型额外算力 |

---

## 与系列其他讲的关联

| 相关讲次 | 关联点 |
|----------|--------|
| 第 1 讲（AI Agent） | System Prompt 是 prompt caching 的完美用例 |
| 第 2 讲（Context Engineering） | Context Window 限制是 Context Engineering 存在的根本原因 |
| 第 4 讲（Positional Encoding） | RoPE 等位置编码影响 KV Cache 的外推能力 |
| 投机采样（TECH） | Speculative Decoding 是本讲提及的第 6 种加速方法 |

---

## 课后思考

1. Flash Attention 的 io-aware tiling 为什么在 Prefill 阶段有效，在 Decode 阶段效果有限？提示：考虑 decode 阶段的 batch size 和 KV Cache 的作用。
2. GQA 的组数 G 从 1（MQA）到 H（MHA）之间，最优的 G 应该由什么因素决定？如果你要为一个新模型选择 G，你的实验流程是什么？
3. Prompt Caching 对 AI Agent 的系统设计有什么隐含影响？它会影响你设计 System Prompt 的方式吗？

---

## 参考资料

1. **课程讲议**：[inference.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/inference.pdf)，李宏毅，NTU ML 2026 Spring
2. **Flash Attention**：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135
3. **Streaming LLM**：Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453
4. **Scissorhands**：Liu et al., *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time*, arXiv:2305.17118
5. **H2O**：Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*, arXiv:2306.14048
6. **DeepSeek-V2 MLA**：DeepSeek-AI, arXiv:2405.04434
7. **Prompt Caching**：OpenAI API Documentation, [prompt-caching](https://developers.openai.com/api/docs/guides/prompt-caching/)
8. **Colab 示例**：Flash Attention 代码示例，[链接](https://colab.research.google.com/drive/1KoeKKIXSXI9b-pYg0kun3-uLQkP6p_hC)
