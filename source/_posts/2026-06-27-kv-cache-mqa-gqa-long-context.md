---
title: "KV Cache、MQA/GQA 与长上下文推理成本：为什么 LLM 推理真正卡在显存"
date: 2026-06-27 08:55:00
mathjax: true
categories:
 - AI
 - AI Infra
tags:
 - KV Cache
 - MQA
 - GQA
 - LLM Inference
 - Long Context
description: "从 KV cache 显存公式出发，推导 MHA、MQA、GQA 的头数关系，解释长上下文推理为什么常常受显存容量和 HBM 带宽限制，并映射到 PyTorch 与 Hugging Face 的实现接口。"
topic_id: "TECH-20260627-02"
---

> 阅读时间：约 12-15 分钟  
> 主题类型：TECH 技术点讲解 / 工程优化  
> 关键词：KV Cache、MQA、GQA、长上下文、LLM 推理

## TL;DR

长上下文 LLM 推理的核心瓶颈经常不是“attention 公式太难算”，而是每一步 decode 都要从显存反复读取越来越大的 KV cache。MQA 和 GQA 的关键思想是减少 key/value head 数量：MHA 为每个 query head 存一套 KV，MQA 让所有 query head 共享一套 KV，GQA 则让一组 query head 共享一套 KV，在显存、带宽和模型质量之间折中 [1][2][3]。

## 前置知识

本文默认读者知道自回归 Transformer 的基本流程：

- Prefill：处理 prompt，计算所有历史 token 的 key/value。
- Decode：每一步生成一个 token，并把新 token 的 key/value 追加进 cache。
- Attention：当前 query 会和历史 key 做点积，再用权重加权历史 value。

我们只聚焦一个技术点：**通过减少 KV head 数量降低 KV cache 的显存和带宽成本**。PagedAttention、KV 量化、prefix caching、FlashAttention 等也很重要，但在本文只作为边界讨论。

## 问题定义：KV Cache 到底占多少显存

设 batch size 为 $B$，上下文长度为 $L$，层数为 $N_l$，query head 数为 $H_q$，KV head 数为 $H_{kv}$，每个 head 维度为 $d_h$，每个元素占 $s$ bytes。对 decoder-only 模型，KV cache 显存近似为：

$$M_{\text{KV}}=B\cdot L\cdot N_l\cdot 2\cdot H_{kv}\cdot d_h\cdot s$$

这里的 $2$ 表示每层每个 token 都要存 key 和 value。这个公式的直觉很直接：

- $B$ 越大，并发请求越多，cache 线性增长。
- $L$ 越长，长上下文和长输出会线性拉大 cache。
- $N_l$ 越多，深层模型每层都要存一份 KV。
- $H_{kv}$ 越多，每个 token 要存的 KV head 越多。
- $d_h$ 和 $s$ 决定每个 head 的宽度和精度。

长上下文推理为什么贵？因为 decode 第 $t$ 步不仅要算当前 token，还要读取前面 $t$ 个 token 的 KV。随着 $L$ 变大，显存容量先被 cache 吃掉，随后 HBM 带宽会成为每步 decode 的硬瓶颈。Shazeer 在 MQA 论文中已经指出，增量推理慢的重要原因是反复加载巨大的 key/value 张量带来的内存带宽成本 [1]。

## Baseline：MHA 为什么显存重

标准 multi-head attention（MHA）中，query、key、value 都有 $H_q$ 个 head。通常 $H_{kv}=H_q$。

对第 $l$ 层，输入隐藏状态 $\mathbf{x}_t\in\mathbb{R}^{d_{\text{model}}}$，每个 head 的投影可以写成：

$$\mathbf{q}_{t,h}=\mathbf{x}_t W^Q_h,\quad \mathbf{k}_{t,h}=\mathbf{x}_t W^K_h,\quad \mathbf{v}_{t,h}=\mathbf{x}_t W^V_h$$

其中 $h=1,\ldots,H_q$，每个 $\mathbf{q}_{t,h},\mathbf{k}_{t,h},\mathbf{v}_{t,h}\in\mathbb{R}^{d_h}$。decode 时，head $h$ 的 attention 是：

$$\operatorname{Attn}_{t,h}=\operatorname{softmax}\left(\frac{\mathbf{q}_{t,h}K_{1:t,h}^{\top}}{\sqrt{d_h}}\right)V_{1:t,h}$$

这套设计表达力强，因为每个 head 有自己的 key/value 子空间。但代价也明显：每个历史 token 在每层要为每个 head 存一份 key 和 value。只要 $H_q$ 大、上下文长，KV cache 就会膨胀。

## MQA：所有 Query Head 共享一套 KV

Multi-Query Attention（MQA）由 Shazeer 在 "Fast Transformer Decoding: One Write-Head is All You Need" 中提出。它保留多个 query head，但只使用一个 key head 和一个 value head，也就是 $H_{kv}=1$ [1]。

公式上，query 仍然分 head：

$$\mathbf{q}_{t,h}=\mathbf{x}_t W^Q_h,\quad h=1,\ldots,H_q$$

但 key/value 共享：

$$\mathbf{k}_{t}=\mathbf{x}_t W^K,\quad \mathbf{v}_{t}=\mathbf{x}_t W^V$$

每个 query head 都对同一套历史 $K_{1:t},V_{1:t}$ 做 attention：

$$\operatorname{Attn}_{t,h}=\operatorname{softmax}\left(\frac{\mathbf{q}_{t,h}K_{1:t}^{\top}}{\sqrt{d_h}}\right)V_{1:t}$$

显存收益来自 $H_{kv}$ 从 $H_q$ 变成 $1$：

$$\frac{M_{\text{MQA}}}{M_{\text{MHA}}}=\frac{1}{H_q}$$

如果 $H_q=32$，KV cache 理论上降到 MHA 的 $1/32$。这对 decode 很有价值，因为每步读取的历史 key/value 也同步减少。

MQA 的代价是表达容量下降。所有 query head 看同一套 key/value 表示，模型少了“不同 head 各自维护不同 KV 子空间”的自由度。Shazeer 的实验结论是 decode 可以显著加速，质量通常只有小幅退化，但这取决于模型、任务和训练方式 [1]。

## GQA：在 MHA 和 MQA 之间插一档

Grouped-Query Attention（GQA）可以看成 MHA 和 MQA 的连续折中。Ainslie 等人在 GQA 论文中提出：把 $H_q$ 个 query head 分成 $G$ 组，每组共享一套 key/value，因此 $H_{kv}=G$ [2]。

如果 $G=1$，就是 MQA；如果 $G=H_q$，就是 MHA。中间的 $1<G<H_q$ 就是 GQA。

设 query head $h$ 属于组：

$$g(h)=\left\lfloor \frac{h}{H_q/G}\right\rfloor$$

那么第 $h$ 个 query head 使用第 $g(h)$ 组的 key/value：

$$\operatorname{Attn}_{t,h}=\operatorname{softmax}\left(\frac{\mathbf{q}_{t,h}K_{1:t,g(h)}^{\top}}{\sqrt{d_h}}\right)V_{1:t,g(h)}$$

显存比例变成：

$$\frac{M_{\text{GQA}}}{M_{\text{MHA}}}=\frac{G}{H_q}=\frac{H_{kv}}{H_q}$$

这就是 GQA 的核心：它不是改变 attention 的基本计算，而是改变 key/value 的共享粒度。NVIDIA 的推理优化文章也把 GQA 描述为一个在内存需求和模型质量之间折中的机制；MQA 只有一个 key-value head，GQA 则有少量 key-value heads [3]。

## 一个最小显存例子

假设一个模型有：

| 参数 | 数值 |
|---|---:|
| batch size $B$ | 1 |
| context length $L$ | 128k |
| layers $N_l$ | 32 |
| query heads $H_q$ | 32 |
| KV head dim $d_h$ | 128 |
| dtype | BF16, $s=2$ bytes |

MHA 中 $H_{kv}=32$：

$$M_{\text{MHA}}=1\cdot 131072\cdot 32\cdot 2\cdot 32\cdot 128\cdot 2\approx 68.7\text{ GB}$$

GQA 如果 $H_{kv}=8$：

$$M_{\text{GQA}}=1\cdot 131072\cdot 32\cdot 2\cdot 8\cdot 128\cdot 2\approx 17.2\text{ GB}$$

MQA 如果 $H_{kv}=1$：

$$M_{\text{MQA}}=1\cdot 131072\cdot 32\cdot 2\cdot 1\cdot 128\cdot 2\approx 2.1\text{ GB}$$

这个例子没有包括模型权重、activation、allocator 碎片、并发请求、系统预留和框架开销。真实 serving 会更复杂。但它足够说明为什么长上下文推理会被 KV cache 卡住：一个 batch size 为 1 的 128k 上下文，MHA 的 KV cache 已经可以吃掉几十 GB 显存。

## 实现映射：`num_key_value_heads` 是关键旋钮

Hugging Face Transformers 的 Llama 配置里有一个非常直观的参数：`num_key_value_heads`。官方文档说明：

- `num_key_value_heads` 等于 `num_attention_heads`：使用 MHA。
- `num_key_value_heads` 等于 `1`：使用 MQA。
- 其他中间值：使用 GQA。
- 从 MHA checkpoint 转 GQA checkpoint 时，每组 key/value head 可由组内原始 heads mean pooling 构造 [4]。

这正好对应前面的 $H_{kv}$。一个简化版形状表如下：

| 张量 | MHA shape | GQA shape |
|---|---|---|
| Query | $[B,H_q,L,d_h]$ | $[B,H_q,L,d_h]$ |
| Key cache | $[B,H_q,L,d_h]$ | $[B,H_{kv},L,d_h]$ |
| Value cache | $[B,H_q,L,d_h]$ | $[B,H_{kv},L,d_h]$ |

PyTorch 的 `scaled_dot_product_attention` 也暴露了 `enable_gqa` 参数。文档中的等价实现会在 `enable_gqa` 设为 `True` 时对 key/value 做 `repeat_interleave`，把较少的 KV heads 扩展到 query head 数量，然后再走普通 attention 公式 [5]。这说明 GQA 在概念上可以看成“计算时让多个 query heads 看同一个 KV head”，只是高性能实现不会真的在显存里低效复制所有 cache。

简化伪代码如下：

```python
def repeat_kv_for_gqa(k, v, num_query_heads, num_kv_heads):
    # k/v: [batch, num_kv_heads, seq_len, head_dim]
    group_size = num_query_heads // num_kv_heads
    k = repeat_interleave(k, repeats=group_size, dim=1)
    v = repeat_interleave(v, repeats=group_size, dim=1)
    # k/v: [batch, num_query_heads, seq_len, head_dim]
    return k, v
```

这段伪代码只用于解释形状，不代表生产推理引擎的最优实现。生产内核更关心避免不必要的数据复制、提升内存访问局部性，并和 KV cache 管理策略组合。

## 为什么减少显存也会改善吞吐

KV cache 有两种成本：容量成本和带宽成本。

容量成本决定你能不能把长上下文、多并发和长输出放进 GPU。带宽成本决定每个 decode step 能多快读取历史 KV。MQA/GQA 同时降低这两者。

decode 第 $t$ 步，粗略读取量与 $H_{kv}\cdot t\cdot d_h$ 成正比。把 $H_{kv}$ 从 32 降到 8，不只是节省 4 倍 cache 容量，也意味着 attention 读 KV 的带宽压力接近下降 4 倍。对长上下文场景，带宽节省往往比 FLOPs 节省更有感。

这也是为什么 Llama 3.1 的模型卡片会把 GQA 标为提升 inference scalability 的架构特性，并支持 128k context length [6]。这里不能简单说“GQA 单独让模型支持 128k”，长上下文还依赖 RoPE 扩展、训练数据、推理系统和 cache 管理。但 GQA 确实是让长上下文更可部署的重要条件之一。

## 常见误解

第一个误解：MQA/GQA 会让 attention 复杂度从 $O(L^2)$ 变成更低。

不是。对 prefill 来说，attention 仍然要处理序列间关系；对 decode 来说，每一步仍要关注历史 token。MQA/GQA 主要减少 KV cache 的 head 数，降低存储和读取量，不是把 attention 变成线性算法。

第二个误解：MQA 一定比 GQA 好，因为显存最省。

不一定。MQA 的共享最激进，质量风险也更高。GQA 的价值正是用多个 KV heads 换回部分表达能力。GQA 论文的主张是：用少量原始预训练 compute 对 MHA checkpoint uptraining，可以获得接近 MHA 的质量和接近 MQA 的速度 [2]。

第三个误解：只要模型用了 GQA，长上下文推理就不贵。

也不对。GQA 只是降低 $H_{kv}$。当 $B$、$L$、$N_l$ 继续增长，KV cache 仍然会很大。生产系统还需要 PagedAttention、prefix caching、KV quantization、prefill/decode 解耦、continuous batching 等机制共同工作。

## 工程取舍

MHA、MQA、GQA 可以按一个简单表理解：

| 方法 | $H_{kv}$ | KV cache | 质量风险 | 适合场景 |
|---|---:|---:|---|---|
| MHA | $H_q$ | 最大 | 最低 | 训练基线、小模型、显存不敏感场景 |
| MQA | 1 | 最小 | 较高 | 极致 decode 吞吐、强显存约束 |
| GQA | $1<G<H_q$ | 中等 | 中等偏低 | 大多数现代长上下文 LLM 的折中 |

小光判断：今天新模型默认选择 GQA，是因为它足够简单、收益稳定、工程生态成熟。它不像 KV 量化那样要额外考虑量化误差，也不像稀疏 attention 那样改变注意力模式；它只是改变 KV head 共享粒度，却能显著改变部署成本。

## 总结

KV cache 是长上下文推理的“显存账本”。只要模型自回归生成，每一层、每个历史 token 的 key/value 都要被保存并在 decode 中读取。MHA 为每个 query head 存一套 KV，表达力强但显存重；MQA 让所有 query head 共享一套 KV，显存最省但有质量风险；GQA 把 query heads 分组共享 KV，在质量和成本之间取得更实用的折中。

记住这三个公式就够了：

$$M_{\text{KV}}=B\cdot L\cdot N_l\cdot 2\cdot H_{kv}\cdot d_h\cdot s$$

$$\frac{M_{\text{MQA}}}{M_{\text{MHA}}}=\frac{1}{H_q}$$

$$\frac{M_{\text{GQA}}}{M_{\text{MHA}}}=\frac{H_{kv}}{H_q}$$

当你评估一个长上下文模型的部署成本时，不要只看参数量和上下文长度。一定要看 `num_key_value_heads`，因为它直接决定每个 token 在每层要留下多少 KV cache。

## 参考资料

[1] [Noam Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv, 2019](https://arxiv.org/abs/1911.02150)

[2] [Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP / arXiv, 2023](https://arxiv.org/abs/2305.13245)

[3] [NVIDIA, "Mastering LLM Techniques: Inference Optimization", NVIDIA Technical Blog, 2023](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

[4] [Hugging Face, "Llama 2 model documentation", Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/llama2)

[5] [PyTorch, "torch.nn.functional.scaled_dot_product_attention", PyTorch Docs](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)

[6] [Meta, "Llama 3.1 8B Instruct model card", Hugging Face, 2024](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
