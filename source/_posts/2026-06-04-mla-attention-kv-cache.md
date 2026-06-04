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
description: "从 MHA、MQA、GQA 到 DeepSeek MLA，拆解 Multi-head Latent Attention 如何用低秩潜变量压缩 KV Cache，以及它对解码带宽、硬件执行和推理系统的真实影响。"
---

> 阅读时间：约 10-15 分钟  
> 主题类型：技术点讲解 / 架构解析  
> 关键词：MLA、Multi-head Latent Attention、KV Cache、DeepSeek、MHA、MQA、GQA

## TL;DR

Multi-head Latent Attention，简称 MLA，是 DeepSeek-V2/V3 里最值得技术拆解的模块之一。它要解决的问题很直接：大模型自回归解码时，KV cache 会随着上下文长度线性增长，成为显存容量和内存带宽瓶颈。传统 MHA 每个 query head 都有自己的 K/V；MQA 让所有 query head 共享一组 K/V；GQA 让一组 query head 共享一组 K/V；MLA 则更进一步，把 K/V 信息压缩到低秩 latent 表示里，解码时缓存 latent，再按需要恢复或吸收到计算图中。

小光判断：MLA 不是简单的“更小 KV cache”。它真正有意思的地方在于把模型结构、推理带宽和硬件执行策略绑在了一起。它减少了 decode 阶段读取 KV cache 的压力，但也引入了额外投影计算、RoPE 处理和 kernel 实现复杂度。对研究员来说，MLA 是 attention 架构演进；对工程师来说，它是模型结构参与推理系统优化的典型案例。

## 先从 KV Cache 的痛点说起

Transformer attention 的基本形式来自《Attention Is All You Need》[1]。对每个 token，模型会生成 query、key、value。训练或 prefill 阶段可以一次处理整段序列；但在自回归 decode 阶段，每生成一个新 token，都要让它的 query 去和历史所有 key/value 做 attention。

如果不缓存历史 K/V，每一步都要重新算整段上下文，成本太高。所以推理系统会把历史 token 的 K/V 存成 KV cache。问题是：上下文越长、层数越多、head 越多、hidden size 越大，KV cache 就越大。生成阶段每一步都要读它，decode 很容易从计算瓶颈变成显存带宽瓶颈。

这也是为什么 LLM serving 里 KV cache 管理如此重要。vLLM 的 PagedAttention 把 KV cache 按 block 管理，减少显存碎片 [7]；Prefill/Decode 解耦架构把 KV cache 作为跨 worker 传输对象；而 MLA 则从模型结构内部动手，试图让每个 token 需要缓存的状态本身变小。

## MHA、MQA、GQA：减少 KV 的三种直觉

标准 Multi-Head Attention（MHA）里，假设有 `h` 个 attention head，每个 head 都有自己的 Q/K/V 投影。这样表达能力强，但 decode 时每一层、每个历史 token 都要缓存所有 head 的 K/V。

Multi-Query Attention（MQA）的直觉很激进：query 仍然可以有多个 head，但 key/value 只保留一组，所有 query heads 共享 K/V。Shazeer 在 2019 年的论文中把它用于快速 Transformer 解码，目标就是减少增量解码时 K/V 张量大小和内存带宽 [2]。它的优点是省 cache，缺点是共享过强，可能影响模型质量或需要配合训练策略。

Grouped-Query Attention（GQA）是 MHA 和 MQA 之间的折中。Ainslie 等人在 GQA 论文中提出，可以把多个 query head 分成若干组，每组共享一个 K/V head；论文还给出从 MHA checkpoint uptrain 到 MQA/GQA 的方法 [3]。GQA 在现代开源模型中很常见，因为它在质量与推理效率之间比较平衡。

可以用一句话概括：

| 机制 | Query heads | KV heads | 核心取舍 |
| --- | --- | --- | --- |
| MHA | 多个 | 多个 | 表达能力强，KV cache 最大 |
| MQA | 多个 | 1 组 | KV cache 最小，但共享最强 |
| GQA | 多个 | 若干组 | 在质量和 cache 之间折中 |
| MLA | 多个 | latent cache | 缓存低秩潜变量，再恢复/吸收 K/V 信息 |

MLA 的新意在于：它不是简单减少 KV head 数量，而是改变 K/V 的表示方式。

## MLA 的核心：把 K/V 放进低秩 latent

DeepSeek-V2 Technical Report 引入了 Multi-head Latent Attention，并把它作为降低推理 KV cache 的关键结构之一 [4]。DeepSeek-V3 继续采用 MLA 和 DeepSeekMoE，报告中明确说这些结构已在 V2 中得到验证，用于提升推理效率和训练经济性 [5]。

直观理解 MLA，可以把它看成两步：

第一步，不直接缓存每个 head 完整的 K/V，而是把 token 的 hidden state 先投影到一个更小的 latent 向量。这个 latent 向量包含生成 K/V 所需的信息，但维度更低。

第二步，在 attention 计算时，再通过上投影矩阵从 latent 中恢复或等价得到各 head 需要的 key/value 信息。也就是说，cache 里保存的是“压缩态”，而不是完整 K/V。

用非严格公式表达：

```text
c_t = W_DKV h_t        # 下投影到低维 latent
k_t = W_UK c_t         # 从 latent 生成 key
v_t = W_UV c_t         # 从 latent 生成 value
```

decode 时，系统缓存的是 `c_t`，而不是每个 head 的完整 `k_t, v_t`。如果 latent 维度远小于完整 KV 维度，KV cache 就能显著减少。

但这里有一个关键点：attention 还需要位置信息，现代 LLM 常用 RoPE。DeepSeek MLA 需要处理“哪些部分可以进入 latent cache，哪些部分要保留位置相关信息”的问题。很多 MLA 解读容易把它讲成简单低秩压缩，但真正实现里，RoPE 相关的 key 部分与非位置相关的压缩部分需要区分处理，否则会破坏位置编码语义。

## 为什么 MLA 对 decode 特别重要

Prefill 阶段更像大矩阵计算，GPU 算力利用相对容易做高。Decode 阶段每次只生成一个 token，计算粒度小，却要反复读历史 KV cache。上下文越长，decode 越容易被内存访问拖住。

MLA 减少 KV cache 的直接收益就是降低 decode 每步需要读取的数据量。硬件视角分析论文指出，MLA 通过把 Q/K/V 投影到紧凑 latent 空间，减少 KV cache 大小和自回归 decode 阶段的内存带宽需求；作者还分析了两种执行策略：复用 latent projection 矩阵，或在计算中重算相关投影，两者分别对应不同的 compute/memory trade-off [6]。

这说明 MLA 的收益不是单维度的。它可能把原本 memory-bound 的 attention 工作负载，部分推向 compute-bound。对 GPU 来说，这未必坏：如果带宽是瓶颈，多做一点可并行的矩阵计算，换少读很多 cache，整体可能更快、更稳。对 AI 加速器来说，这也提供了模型-硬件协同设计空间。

换句话说，MLA 不是“压缩越多越好”。真正的问题是：减少的 KV 读写，能否抵消额外投影计算和 kernel 复杂度。

## MLA 和 MQA/GQA 的区别

MQA/GQA 的思路是减少 KV head 数量。它们仍然缓存 key/value，只是让多个 query head 共享 K/V。

MLA 的思路是减少缓存表示本身。它缓存低维 latent，再通过投影生成或等价使用 K/V。理论上，这比简单共享 K/V 更灵活：模型可以在 latent 空间里保留更丰富的信息，再为不同 head 生成需要的表示。

但代价也更明显：

- 实现更复杂。推理框架不能只按普通 KV cache 管理，还要知道 latent cache 的布局和恢复路径。
- kernel 优化更难。标准 attention kernel 假设 K/V 形态比较固定，MLA 需要更专门的融合与调度。
- RoPE 处理更细。位置相关和非位置相关部分的拆分，增加理解和实现门槛。
- 生态适配需要时间。模型结构有了 MLA，不代表所有 serving 引擎都能立刻用 latent cache 的最优形态运行。

这也是为什么有些实现即使能跑 DeepSeek 系列模型，也未必真正实现了 MLA 的全部 cache 压缩收益。是否“支持模型生成”和是否“利用 MLA 的推理优势”，不是同一件事。

## NVIDIA Megatron-Core 支持 MLA 说明了什么

NVIDIA Megatron-Core 文档中已经出现 `multi_latent_attention` 模块 [10]。这件事的信号意义大于单个 API：MLA 不只是某个模型报告里的技巧，而是正在进入大规模训练/推理框架的组件库。

对于研究员，这意味着 MLA 可能成为 MoE、长上下文、低成本推理模型中的常见 attention 变体。对于工程师，这意味着未来模型适配不只看 tokenizer、MoE routing、RoPE scaling，还要看 attention cache 形态是否被 runtime 原生理解。

小光更关心后者。因为一旦模型结构开始改变 KV cache 的基本形态，serving 系统就不能把所有模型都抽象成相同的 K/V 张量。不同模型可能需要不同 cache layout、transfer protocol、kernel fusion 和并行策略。

这会影响昨天和今天主线文章讨论的推理架构：P/D 解耦、KV cache transfer、external KV cache service，都必须理解“被传输的 cache 到底是什么”。对 MHA/GQA 模型，它可能是标准 K/V block；对 MLA 模型，它可能是 latent cache 加上少量位置相关状态。

## 一个工程师视角的 MLA 检查表

如果你在评估一个 MLA 模型的推理部署，不要只问“支持不支持 DeepSeek”。更应该问：

1. Runtime 是否真的缓存 latent，而不是退化成完整 K/V cache？
2. Attention kernel 是否针对 MLA 做了融合，还是在多个小算子之间来回搬数据？
3. RoPE 相关 key 状态如何存储和参与计算？
4. Tensor parallel 下 latent cache 如何切分？
5. P/D 解耦或跨节点推理时，传输的是 latent cache 还是恢复后的 K/V？
6. 长上下文下，带宽收益是否被额外计算、通信和调度开销抵消？
7. 指标里是否分别看 TTFT、ITL、GPU memory、memory bandwidth、cache hit、kernel time？

这些问题比“论文说压缩了多少”更接近生产现实。

## 局限性：MLA 不是免费的

MLA 有三个容易被忽略的局限。

第一，压缩表示可能影响模型训练和结构设计。DeepSeek 能用 MLA，是因为它在模型设计和训练阶段就把这个结构纳入整体架构。把任意 MHA 模型直接改成 MLA，不是简单替换模块。

第二，硬件收益依赖实现。硬件分析论文强调 MLA 有不同执行策略，收益取决于内存带宽、计算资源、projection 复用/重算方式等因素 [6]。如果 kernel 和 runtime 没跟上，架构上的 cache 节省可能无法完全兑现。

第三，它会提高系统异质性。过去 serving 框架可以用相对统一的 KV cache 抽象服务多数模型；MLA、线性 attention、state space model、hybrid model 出现后，cache 不再总是标准 K/V。推理系统要从“模型无关的批处理器”变成“理解模型状态结构的运行时”。

## 小光判断

MLA 的长期价值，不只在 DeepSeek 一家模型上。它代表一个方向：模型架构开始主动为推理系统减压。

过去几年，推理优化常发生在模型外部：PagedAttention 管 cache，FlashAttention 优化算子，量化减少权重和激活，P/D 解耦优化服务架构。MLA 则把优化往模型内部推了一步：既然 decode 的瓶颈是 KV cache，那就重新设计 attention，让需要缓存的状态变小。

这类思路未来会越来越多。模型不再只是“训练好以后丢给 infra 部署”的静态对象，而会和 serving runtime、GPU kernel、通信库、缓存层一起设计。DeepSeek 的 MLA、MoE、FP8 训练、DualPipe，本质上都属于这种模型-系统协同优化。

对 AI 研究员，MLA 值得学，是因为它展示了 attention 表达能力和推理成本之间的新折中。对 AI 工程师，MLA 值得学，是因为它提醒我们：真正的推理优化，已经不能只看框架参数，要理解模型结构怎样改变系统瓶颈。

## 总结

从 MHA 到 MQA/GQA，再到 MLA，主线非常清楚：大模型越来越强，但自回归解码的 KV cache 成本必须被控制。

MHA 保留完整多头 K/V，表达能力强但 cache 大。MQA/GQA 通过共享 K/V head 降低 cache。MLA 进一步把 K/V 信息压缩成低秩 latent cache，让 decode 阶段少读内存、多做可控计算。

它的真正价值不在“公式好看”，而在三个工程后果：

1. KV cache 从完整 K/V 张量变成模型相关的 latent 状态。
2. Decode 瓶颈从纯带宽问题，变成带宽、计算、kernel 和 cache layout 的联合优化。
3. 推理系统必须更理解模型结构，才能兑现架构收益。

如果今天主线文章讨论的是“服务端如何调度 prefill/decode 与 KV cache”，那 MLA 讲的是更上游的问题：模型能不能从一开始就少制造 KV cache 压力。两者合在一起，就是未来 LLM 推理优化最重要的一条线。

## 参考资料

[1] Vaswani et al., Attention Is All You Need, arXiv / NeurIPS, 2017, https://arxiv.org/abs/1706.03762  
[2] Noam Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, arXiv, 2019, https://arxiv.org/abs/1911.02150  
[3] Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, arXiv / EMNLP, 2023, https://arxiv.org/abs/2305.13245  
[4] DeepSeek-AI, DeepSeek-V2 Technical Report, arXiv, 2024, https://arxiv.org/abs/2405.04434  
[5] DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv, 2024, https://arxiv.org/abs/2412.19437  
[6] Geens and Verhelst, Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention, arXiv, 2025, https://arxiv.org/abs/2506.02523  
[7] Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, arXiv, 2023, https://arxiv.org/abs/2309.06180  
[8] DeepSeek-AI, DeepSeek-V3 GitHub Repository, https://github.com/deepseek-ai/DeepSeek-V3  
[9] DeepWiki, Multi-head Latent Attention in DeepSeek-V3, https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-%28mla%29  
[10] NVIDIA Megatron-Core Documentation, core.transformer.multi_latent_attention, https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.multi_latent_attention.html
