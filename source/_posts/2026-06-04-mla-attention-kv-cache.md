---
title: "MLA 注意力机制拆解：DeepSeek 如何用低秩潜变量压缩 KV Cache"
date: 2026-06-04 09:18:08
categories:
 - AI
 - LLM
tags:
 - MLA
 - Attention
 - KV Cache
 - DeepSeek
 - Inference Optimization
description: "从 MHA、MQA、GQA 到 DeepSeek MLA，按公式、张量形状、最小例子和官方实现拆解 Multi-head Latent Attention 如何压缩 KV Cache，以及为什么它会改变推理系统的瓶颈。"
mathjax: true
---

> 阅读时间：约 15-20 分钟
> 主题类型：技术点讲解 / 架构解析
> 关键词：MLA、Multi-head Latent Attention、KV Cache、DeepSeek、MHA、MQA、GQA、RoPE

## TL;DR

Multi-head Latent Attention，简称 MLA，是 DeepSeek-V2 提出的注意力结构，DeepSeek-V3 继续沿用。它要解决的不是“attention 算不动”，而是自回归解码时 **KV cache 太大、读写太贵** 这个更具体的问题。

一句话讲清楚：

> MHA 缓存每个 head 的完整 Key/Value；MQA/GQA 通过共享 KV head 减少缓存；MLA 则把 Key/Value 先压到一个低维 latent 向量里，推理时主要缓存 latent，再通过矩阵吸收或恢复完成 attention 计算。

DeepSeek-V2 技术报告称，MLA 使用 low-rank key-value joint compression 来压缩 KV cache，并报告相对 DeepSeek 67B 减少 93.3% KV cache、最大生成吞吐提升到 5.76 倍；在附录对比中，大模型 MLA 的 KV cache 约为 MHA 的 4%，小模型约为 14% [4]。这些数字来自 DeepSeek 报告，不应泛化成所有模型、所有硬件上的固定收益。

这篇文章按学习路径讲：

1. 先从标准 attention 和 KV cache 定义开始。
2. 推导 MHA 为什么缓存大。
3. 看 MQA/GQA 如何用共享 KV 降低成本。
4. 再拆 DeepSeek MLA 的低秩 KV joint compression。
5. 解释为什么 RoPE 要解耦。
6. 用 DeepSeek-V3 官方推理代码说明 `naive` 和 `absorb` 两种实现。
7. 最后讨论它对推理系统、硬件和模型设计的真实边界。

## 前置知识

读这篇文章最好具备四个背景：

- **Transformer attention**：知道 Query、Key、Value 和 softmax attention。
- **自回归解码**：知道 LLM 每次生成一个 token，需要使用历史上下文。
- **KV cache**：知道推理时会缓存历史 token 的 Key/Value，避免每步重算。
- **RoPE**：知道 rotary positional embedding 会把位置信息注入 query/key。

如果这些概念还不熟，也可以继续读。下面会从公式开始建模。

## 问题定义：KV cache 到底缓存了什么

设某一层 Transformer 的 hidden state 为：

$$\mathbf{h}_t \in \mathbb{R}^{d}$$

其中 `t` 是 token 位置，`d` 是模型隐藏维度。标准 Multi-Head Attention 里，有 `n_h` 个注意力头，每个 head 的维度为 `d_h`。最常见的投影是：

$$\mathbf{q}_t = W^Q\mathbf{h}_t,\quad \mathbf{k}_t = W^K\mathbf{h}_t,\quad \mathbf{v}_t = W^V\mathbf{h}_t$$

把它拆成多个 head：

$$\mathbf{q}_t = [\mathbf{q}_{t,1};\mathbf{q}_{t,2};\ldots;\mathbf{q}_{t,n_h}]$$

$$\mathbf{k}_t = [\mathbf{k}_{t,1};\mathbf{k}_{t,2};\ldots;\mathbf{k}_{t,n_h}],\quad \mathbf{v}_t = [\mathbf{v}_{t,1};\mathbf{v}_{t,2};\ldots;\mathbf{v}_{t,n_h}]$$

其中：

$$\mathbf{q}_{t,i},\mathbf{k}_{t,i},\mathbf{v}_{t,i}\in\mathbb{R}^{d_h}$$

第 `i` 个 head 在位置 `t` 的 attention 输出是：

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \operatorname{softmax}_j \left( \frac{\mathbf{q}_{t,i}^{\top}\mathbf{k}_{j,i}}{\sqrt{d_h}} \right) \mathbf{v}_{j,i}$$

这就是自回归解码的关键：当模型生成第 `t` 个 token 时，它要看 `1...t` 的历史 key/value。为了避免每一步都重新计算历史 token 的 K/V，推理系统会缓存：

$$\{\mathbf{k}_{j,i},\mathbf{v}_{j,i}\mid j\le t,\ i=1,\ldots,n_h\}$$

这就是 KV cache。

如果每层缓存长度为 `L`，batch size 为 `B`，每个元素占 `bytes` 字节，那么一层 MHA KV cache 量级是：

$$\operatorname{Cache}_{MHA} = B\cdot L\cdot n_h\cdot(d_h+d_h)\cdot bytes = B\cdot L\cdot n_h\cdot 2d_h\cdot bytes$$

全模型还要乘以层数 `N_layer`。

这就是问题所在：`L` 越长、`n_h` 越多、层数越多，KV cache 就越大。更麻烦的是，decode 阶段每生成一个 token 都要反复读取历史 cache。此时瓶颈往往不是矩阵乘法峰值算力，而是显存容量和 HBM 带宽。

## 第一步改进：MQA/GQA 用共享 KV 降低缓存

Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）是 MLA 的前置背景。理解它们，才能看清 MLA 到底新在哪里。

### MQA：所有 query head 共享一组 KV

MQA 的思想很直接：保留多个 query head，但只保留一组 key/value head。也就是：

$$\mathbf{q}_{t,i}=W_i^Q\mathbf{h}_t,\quad i=1,\ldots,n_h$$

$$\mathbf{k}_t=W^K\mathbf{h}_t,\quad \mathbf{v}_t=W^V\mathbf{h}_t$$

第 `i` 个 query head 都使用同一组 `k_t, v_t`：

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \operatorname{softmax}_j \left( \frac{\mathbf{q}_{t,i}^{\top}\mathbf{k}_j}{\sqrt{d_h}} \right) \mathbf{v}_j$$

KV cache 从：

$$B\cdot L\cdot n_h\cdot 2d_h$$

变成：

$$B\cdot L\cdot 1\cdot 2d_h$$

缓存约缩小 `n_h` 倍。Shazeer 的 MQA 论文目标正是加速 Transformer 增量解码，减少每步读取的 K/V 数据 [2]。

代价也明显：所有 query head 被迫共享同一组 K/V，表达能力可能下降。

### GQA：一组 query head 共享一组 KV

GQA 是折中。设有 `n_kv` 组 KV head，且 `n_kv < n_h`。每组 query heads 共享一个 KV head：

$$\operatorname{Cache}_{GQA}=B\cdot L\cdot n_{kv}\cdot 2d_h\cdot bytes$$

当 `n_kv = n_h`，退化为 MHA；当 `n_kv = 1`，退化为 MQA。Ainslie 等人的 GQA 论文还讨论了如何从 MHA checkpoint 继续训练得到 GQA/MQA 模型 [3]。

到这里，优化方向仍然是“减少 KV head 数量”。MLA 的方向不同：它不只是减少 head，而是改变缓存的表示。

## MLA 的核心：低秩 KV joint compression

DeepSeek-V2 把 MLA 定义为带有 low-rank key-value joint compression 的注意力机制 [4]。关键词有两个：

- **joint**：Key 和 Value 不是分别压缩，而是共同压到一个 latent。
- **low-rank**：缓存的 latent 维度 `d_c` 小于完整 K/V 表示。

对每个 token，MLA 先计算：

$$\mathbf{c}_t^{KV}=W^{DKV}\mathbf{h}_t$$

其中：

$$\mathbf{h}_t\in\mathbb{R}^{d},\quad W^{DKV}\in\mathbb{R}^{d_c\times d},\quad \mathbf{c}_t^{KV}\in\mathbb{R}^{d_c}$$

`c_t^KV` 就是低维 KV latent。然后再从它恢复 content key 和 value：

$$\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV},\quad \mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}$$

其中：

$$W^{UK},W^{UV}\in\mathbb{R}^{(n_h d_h)\times d_c},\quad \mathbf{k}_t^C,\mathbf{v}_t^C\in\mathbb{R}^{n_h d_h}$$

这组公式的直觉很重要：

- MHA 缓存的是 `k_t` 和 `v_t`，维度约 `2 n_h d_h`。
- MLA 缓存的是 `c_t^KV`，维度是 `d_c`。
- 如果 `d_c << 2 n_h d_h`，缓存会大幅下降。

这不是普通的 KV head 共享。MQA/GQA 是让多个 query head 共用较少的 KV head；MLA 是把“生成所有 head 的 K/V 所需信息”先压到一个公共 latent，再通过上投影矩阵展开。

## Query 也可以低秩，但它不是 KV cache 的核心

DeepSeek-V2 的完整公式里，query 也有低秩压缩：

$$\mathbf{c}_t^Q=W^{DQ}\mathbf{h}_t,\quad \mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q$$

其中：

$$W^{DQ}\in\mathbb{R}^{d'_c\times d},\quad W^{UQ}\in\mathbb{R}^{(n_h d_h)\times d'_c}$$

但要注意：query 是当前 token 用的，通常不会像历史 K/V 一样长期缓存。因此从推理系统角度看，KV latent 才是 MLA 降 cache 的主角；query 低秩更多影响参数量、计算路径和结构对称性。

## RoPE 为什么要单独处理

如果 MLA 只做上面几行公式，还不完整。现代 LLM 常用 RoPE 给 query/key 加位置信息。RoPE 的特点是：它不是简单加一个位置向量，而是对向量进行与位置相关的旋转。

这会带来一个麻烦：如果把所有 key 都写成：

$$\mathbf{k}_t=\operatorname{RoPE}(W^{UK}\mathbf{c}_t^{KV})$$

那么 `W^UK` 和 RoPE 的顺序强绑定。由于 RoPE 依赖位置 `t`，你很难把 `W^UK` 干净地“吸收”进 query 侧计算，也很难只缓存一个位置无关的 latent 后再在所有路径里复用。

DeepSeek 的做法是 **decoupled RoPE**：把 key/query 拆成 content 部分和 RoPE 部分。

对 query：

$$\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^C;\mathbf{q}_{t,i}^R]$$

其中：

$$\mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q,\quad \mathbf{q}_t^R=\operatorname{RoPE}(W^{QR}\mathbf{c}_t^Q)$$

对 key：

$$\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^C;\mathbf{k}_t^R]$$

其中：

$$\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV},\quad \mathbf{k}_t^R=\operatorname{RoPE}(W^{KR}\mathbf{h}_t)$$

注意 `k_t^R` 没有按 head 展开成每个 head 不同的一整套 latent，而是作为 RoPE key 部分被拼到各 head 的 content key 上。DeepSeek-V3 官方推理代码里也能看到同样的拆分：`q_nope, q_pe` 和 `kv, k_pe` 分开，`kv_cache` 缓 latent，`pe_cache` 缓 RoPE key 部分 [8]。

最终第 `i` 个 head 的 attention score 变成：

$$score_{t,j,i} = \frac{\mathbf{q}_{t,i}^{\top}\mathbf{k}_{j,i}}{\sqrt{d_h+d_h^R}} = \frac{(\mathbf{q}_{t,i}^C)^{\top}\mathbf{k}_{j,i}^C+(\mathbf{q}_{t,i}^R)^{\top}\mathbf{k}_j^R}{\sqrt{d_h+d_h^R}}$$

这里 `d_h^R` 是 RoPE 部分维度。这个式子很关键：

- content 部分来自 latent KV。
- position 部分来自单独的 RoPE key。
- 两个分数相加，再进入 softmax。

所以 MLA 的 cache 不是“只缓存一个 `c_t^KV` 就万事大吉”。更准确地说，它缓存：

$$cache_t=[\mathbf{c}_t^{KV};\mathbf{k}_t^R]$$

在 DeepSeek-V3 官方代码的 `absorb` 模式里，对应：

$$kv\_cache:\ (batch,\ seq\_len,\ kv\_lora\_rank)$$

$$pe\_cache:\ (batch,\ seq\_len,\ qk\_rope\_head\_dim)$$

这就是“低秩 latent + 少量位置 key”的缓存结构。

## Absorb 技巧：为什么不用真的恢复完整 K/V

如果每次 decode 都先从 latent 恢复完整 `k_t^C, v_t^C`，再做普通 attention，MLA 仍然能省缓存，但可能多出不少投影计算和中间张量。DeepSeek-V3 官方推理代码里有两种实现：

- `naive`：缓存展开后的 `k_cache` 和 `v_cache`。
- `absorb`：缓存 `kv_cache` 和 `pe_cache`，把部分矩阵乘法吸收到 query 或输出路径里。

我们用公式看 absorb 的核心。

content key 是：

$$\mathbf{k}_{j,i}^C=W_i^{UK}\mathbf{c}_j^{KV}$$

content score 是：

$$(\mathbf{q}_{t,i}^C)^{\top}\mathbf{k}_{j,i}^C = (\mathbf{q}_{t,i}^C)^{\top}W_i^{UK}\mathbf{c}_j^{KV}$$

矩阵乘法可以重新结合：

$$(\mathbf{q}_{t,i}^C)^{\top}W_i^{UK}\mathbf{c}_j^{KV} = ((W_i^{UK})^{\top}\mathbf{q}_{t,i}^C)^{\top}\mathbf{c}_j^{KV}$$

于是我们可以先把 query 变换到 latent 空间：

$$\mathbf{q}'_{t,i}=(W_i^{UK})^{\top}\mathbf{q}_{t,i}^C$$

再直接和缓存的 latent 做点积：

$$content\_score_{t,j,i}=(\mathbf{q}'_{t,i})^{\top}\mathbf{c}_j^{KV}$$

这就是“吸收”的直觉：不用显式构造每个历史 token、每个 head 的完整 content key，而是把 `W^UK` 移到 query 侧。

DeepSeek-V3 官方代码中对应这一行：

```python
q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
```

然后 content score 和 RoPE score 分开算：

```python
scores = (
    torch.einsum("bshc,btc->bsht", q_nope, kv_cache)
  + torch.einsum("bshr,btr->bsht", q_pe, pe_cache)
) * softmax_scale
```

把符号翻译成张量形状：

- `b`：batch。
- `s`：当前 query 序列长度，decode 时常为 1。
- `h`：local attention heads。
- `c`：KV latent 维度 `kv_lora_rank`。
- `t`：历史 cache 长度。
- `r`：RoPE key 维度 `qk_rope_head_dim`。

所以 `absorb` 模式里的 score 是：

$$score = \langle query\_in\_latent\_space,\ cached\_kv\_latent\rangle + \langle query\_rope\_part,\ cached\_rope\_key\rangle$$

这比“先恢复完整 K，再点积”更接近 MLA 想要的推理形态。

Value 也有类似的吸收路径。普通写法是：

$$\mathbf{v}_{j,i}^C=W_i^{UV}\mathbf{c}_j^{KV},\quad \mathbf{o}_{t,i}=\sum_j a_{t,j,i}\mathbf{v}_{j,i}^C$$

把 `v` 展开：

$$\mathbf{o}_{t,i} = \sum_j a_{t,j,i}W_i^{UV}\mathbf{c}_j^{KV} = W_i^{UV}\left(\sum_j a_{t,j,i}\mathbf{c}_j^{KV}\right)$$

也就是说，可以先对 latent 做 attention 加权求和，再通过 `W_i^UV` 投回 value head 维度。DeepSeek-V3 代码里对应：

```python
x = torch.einsum("bsht,btc->bshc", scores, kv_cache)
x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])
```

这一步解释了 MLA 的工程本质：它不是单纯“压缩后再解压”，而是在代数上重排矩阵乘法，让推理尽量围绕低维 latent cache 运行。

## 最小例子：看缓存怎么变小

用一个简化例子。

假设：

$$n_h=32,\quad d_h=128,\quad d_h^R=64,\quad d_c=512,\quad bytes=2$$

MHA 每个 token 每层缓存：

$$K+V=n_h(d_h+d_h)=32\times256=8192\ \text{numbers}$$

$$8192\times2=16384\ \text{bytes}=16\ \text{KB}$$

MLA absorb 每个 token 每层缓存：

$$\mathbf{c}^{KV}+\mathbf{k}^R=d_c+d_h^R=512+64=576\ \text{numbers}$$

$$576\times2=1152\ \text{bytes}\approx1.125\ \text{KB}$$

缓存比例约为：

$$\frac{576}{8192}\approx7.0\%$$

也就是这组简化参数下，MLA 的 cache 约为 MHA 的 7%。真实模型数字取决于 `n_h`、`d_h`、`d_h^R`、`d_c`、是否按 tensor parallel 切分、是否还缓存其他状态。DeepSeek-V3 官方 DeepWiki 对 671B 示例给出过 “naive K+V 40960 dims vs absorb cache 576 dims” 的实现层面对比 [9]；这类数字适合帮助理解，但生产中仍要以具体模型配置和 runtime 为准。

## 最小伪代码：decode 一步发生了什么

下面是一个概念版伪代码，不追求完全等同某个框架实现，只保留 MLA 的关键路径：

```python
def mla_decode_step(h_t, cache):
    # 1. query 低秩投影
    c_q = W_DQ @ h_t
    q = W_UQ @ norm(c_q)
    q_nope, q_pe = split(q, [d_nope, d_rope])
    q_pe = RoPE(q_pe, position=t)

    # 2. 当前 token 的 KV latent 和 RoPE key
    kv_and_pe = W_DKV @ h_t
    c_kv, k_pe = split(kv_and_pe, [d_c, d_rope])
    c_kv = norm(c_kv)
    k_pe = RoPE(k_pe, position=t)

    # 3. 写入缓存：不是完整 K/V，而是 latent + RoPE key
    cache.kv[t] = c_kv
    cache.pe[t] = k_pe

    # 4. absorb: 把 W_UK 吸收到 query 侧
    q_latent = absorb_W_UK(q_nope)

    # 5. content score + position score
    score_content = q_latent @ cache.kv[:t].T
    score_rope = q_pe @ cache.pe[:t].T
    score = (score_content + score_rope) / sqrt(d_nope + d_rope)
    attn = softmax(score)

    # 6. 先在 latent 上聚合，再通过 W_UV 得到 value output
    latent_out = attn @ cache.kv[:t]
    head_out = apply_W_UV(latent_out)
    return W_O @ concat_heads(head_out)
```

这段伪代码要传达的重点是：decode 时最常被反复读取的历史状态，主要是 `cache.kv` 和 `cache.pe`，而不是所有 head 的完整 K/V。

## 和 MHA/MQA/GQA 的系统对比

| 机制 | 缓存对象 | 每 token cache 量级 | 表达能力直觉 | 工程难度 |
| --- | --- | --- | --- | --- |
| MHA | 每个 head 的 K/V | `2 n_h d_h` | 最完整 | 最标准 |
| MQA | 1 组共享 K/V | `2 d_h` | 共享最强，可能损失表达 | 简单 |
| GQA | `n_kv` 组 K/V | `2 n_kv d_h` | 折中 | 中等 |
| MLA naive | 展开后的 K/V | 接近完整 K/V | 可验证结构效果 | 较高 |
| MLA absorb | `c^KV + k^R` | `d_c + d_h^R` | latent 保留信息，RoPE 单独处理 | 最高 |

这里的 “MLA naive” 和 “MLA absorb” 是实现视角。模型结构是 MLA，但 runtime 是否真正利用低维 cache，决定了推理收益能兑现多少。

## 工程意义：MLA 把瓶颈从带宽推向矩阵重排

硬件视角分析论文指出，MLA 通过把 Q/K/V 投影到紧凑 latent 空间，减少 KV cache 和自回归 decode 阶段的内存带宽需求；论文还比较了复用 latent projection 矩阵与重算相关投影的执行方案，强调这是 compute/memory trade-off [6]。

这句话翻译成工程语言是：

- MHA decode 很容易 memory-bound：每步读大量历史 K/V。
- MLA absorb 少读很多 cache，但多做一些小矩阵变换和 einsum。
- 如果硬件带宽紧张、算力相对富余，MLA 可能更划算。
- 如果 kernel 没融合好、矩阵太碎、通信或调度开销大，收益会被吃掉。

所以 MLA 的真正价值不是“数学上压缩了 cache”，而是让系统有机会把瓶颈从不可避免的大量 HBM 读取，转移到更可优化的计算和数据布局上。

这也解释了为什么 NVIDIA Megatron-Core 已经把 Multi-Latent Attention 作为功能文档化 [10]。当训练/推理框架原生支持 MLA，工程团队才有机会在 tensor parallel、sequence parallel、FlashAttention 风格 kernel、KV cache transfer、P/D 解耦等层面继续优化。

## 常见误解

### 误解一：MLA 等于 LoRA

MLA 的投影形式看起来像低秩分解，DeepSeek-V3 代码里也有 `q_lora_rank`、`kv_lora_rank` 这样的命名。但它不是“给模型加 LoRA adapter”。LoRA 通常是参数高效微调方法；MLA 是模型原生 attention 结构，目标之一是减少推理 KV cache。

### 误解二：MLA 只是 MQA/GQA 的另一种写法

不是。MQA/GQA 减少 KV head 数量，缓存的仍然是 key/value。MLA 缓存的是低维 latent 和 RoPE key 部分，attention 计算需要配合矩阵吸收或恢复。

### 误解三：MLA 一定更快

不一定。它一定会改变 cache 形态，但端到端速度取决于：

- latent 维度设置。
- kernel 是否融合。
- runtime 是否使用 absorb。
- batch、上下文长度、输出长度分布。
- GPU/加速器的算力与带宽比例。
- P/D 解耦或跨节点传输时传的是 latent 还是展开后的 K/V。

### 误解四：RoPE 是小细节

RoPE 不是小细节。它决定哪些矩阵能被吸收到 query 侧，哪些位置相关信息必须单独缓存。很多 MLA 解释如果跳过 decoupled RoPE，就无法解释为什么官方实现里会有 `pe_cache`。

## 小光判断

MLA 最值得学习的地方，不是某个单点技巧，而是它展示了一个趋势：

> 模型结构开始主动适配推理系统瓶颈。

过去很多推理优化发生在模型外部：PagedAttention 管理 KV block，FlashAttention 优化 attention kernel，量化降低权重/激活存储，P/D 解耦改善 serving 架构。MLA 则直接改变模型内部“需要被缓存的状态”。

对 AI 专业研究生来说，MLA 是一个很好的研究样本，因为它同时连接三件事：

1. **表示学习**：低维 latent 是否能保留足够的 K/V 信息。
2. **优化目标**：模型质量、cache 大小、decode 带宽之间如何折中。
3. **系统实现**：矩阵吸收、RoPE 解耦、cache layout、kernel 融合如何决定最终收益。

对 AI 工程师来说，部署 MLA 模型时不要只问“框架支持 DeepSeek 吗”。更应该问：

- 是否真的缓存 `c^KV + k^R`，还是退化成完整 K/V？
- 是否支持 absorb 路径？
- Tensor parallel 下 `kv_lora_rank` 和 `qk_rope_head_dim` 如何切分？
- P/D 解耦传输的是 latent cache 还是展开后 K/V？
- 监控里是否分别看 TTFT、ITL、KV cache memory、HBM bandwidth、kernel time？

这些问题决定 MLA 是真正降低推理成本，还是只在模型结构图上看起来漂亮。

## 总结：5 条 takeaway

1. **KV cache 是 decode 阶段的核心瓶颈之一**：标准 MHA 每层每 token 要缓存所有 head 的 K/V，长上下文下显存和带宽压力很大。
2. **MQA/GQA 通过共享 KV head 降缓存**：它们减少 KV head 数量，但缓存对象仍是 K/V。
3. **MLA 改变缓存表示**：它把 K/V 共同压到低维 `c_t^KV`，再配合 decoupled RoPE 缓存 `k_t^R`。
4. **absorb 是兑现收益的关键实现**：通过矩阵重排，把 `W^UK` 移到 query 侧，把 value 聚合放在 latent 空间中做，避免显式恢复完整历史 K/V。
5. **MLA 是模型-系统协同优化案例**：收益取决于模型结构、runtime、kernel、硬件带宽和 workload 分布，不能只看论文里的 cache 压缩比例。

如果今天主线文章讨论的是“服务端如何调度 prefill/decode 与 KV cache”，那 MLA 讲的是更上游的问题：模型能不能从一开始就少制造 KV cache 压力。两者合起来，就是未来 LLM 推理优化的一条主线：上下文状态不再只是副产物，而是模型结构和系统架构共同设计的对象。

## 参考资料

[1] Vaswani et al., Attention Is All You Need, arXiv / NeurIPS, 2017, https://arxiv.org/abs/1706.03762  
[2] Noam Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, arXiv, 2019, https://arxiv.org/abs/1911.02150  
[3] Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, arXiv / EMNLP, 2023, https://arxiv.org/abs/2305.13245  
[4] DeepSeek-AI, DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, arXiv, 2024, https://arxiv.org/abs/2405.04434
[5] DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv, 2024, https://arxiv.org/abs/2412.19437  
[6] Geens and Verhelst, Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention, arXiv, 2025, https://arxiv.org/abs/2506.02523  
[7] Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, arXiv, 2023, https://arxiv.org/abs/2309.06180  
[8] DeepSeek-AI, DeepSeek-V3 official inference implementation, GitHub, https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
[9] DeepWiki, Multi-head Latent Attention in DeepSeek-V3, https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-%28mla%29  
[10] NVIDIA Megatron-Core Documentation, Multi-Latent Attention, https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/multi_latent_attention.html
