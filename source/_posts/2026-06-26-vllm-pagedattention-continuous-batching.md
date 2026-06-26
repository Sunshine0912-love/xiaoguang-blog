---
title: "vLLM 连续批处理与 PagedAttention：为什么 LLM Serving 不能只看“单请求延迟”"
date: 2026-06-26 10:20:00
mathjax: true
categories:
 - AI
 - AI Infra
tags:
 - vLLM
 - PagedAttention
 - Continuous Batching
 - KV Cache
 - LLM Serving
description: "从吞吐/延迟指标、KV cache 显存复杂度、PagedAttention block table 和 vLLM 调度器出发，解释生产级 LLM Serving 为什么不是把 generate API 包一层 HTTP。"
topic_id: "TECH-20260626-01"
---

> 阅读时间：约 12-18 分钟  
> 主题类型：TECH 技术点讲解 / 工程优化  
> 关键词：vLLM、PagedAttention、Continuous Batching、KV Cache、LLM Serving

## TL;DR

LLM serving 的目标不是让单个请求跑得最快，而是在给定 GPU 显存和延迟约束下，让尽可能多的请求稳定完成。vLLM 的核心组合是：PagedAttention 用块式 KV cache 管理降低显存碎片，continuous batching 在每次 decode iteration 后重新调度请求，从而提升吞吐 [1][2][3]。

一句话：**PagedAttention 解决“KV cache 放哪里”，continuous batching 解决“下一步算哪些请求”。**

## 前置知识

读者需要知道 Transformer 自回归生成的两个阶段：

- **Prefill**：一次性处理 prompt，生成初始 KV cache。
- **Decode**：每一步读取 KV cache，生成一个新 token，并把新 token 的 key/value 追加到 cache。

此前博客已经写过 continuous batching 的单点机制。本文进一步聚焦 vLLM 中 PagedAttention 与调度系统如何组合，并补上源码路径和显存模型。

## 指标：为什么不能只看单请求延迟

线上推理至少要同时看四个指标：

| 指标 | 含义 | 读法 |
|---|---|---|
| TTFT | time to first token | 用户等多久看到第一个 token |
| TPOT | time per output token | 后续 token 生成速度 |
| Throughput | 单位时间完成 token 或请求数 | GPU 是否吃满 |
| Goodput | 满足 SLO 的有效吞吐 | 生产系统更关心 |

单请求 latency 低，不代表系统好。如果一个引擎只服务一个请求很快，但并发上来后大量排队、显存碎片严重、长短请求互相阻塞，它就不是好的 serving 系统。

## KV Cache 显存复杂度

对一个 decoder-only Transformer，假设 batch size 为 $B$，上下文长度为 $L$，层数为 $n_l$，KV head 数为 $n_{kv}$，每个 head 维度为 $d_h$，每个元素占 $b$ bytes，则 KV cache 大致显存为：

$$M_{\text{KV}} = B \cdot L \cdot n_l \cdot 2 \cdot n_{kv} \cdot d_h \cdot b$$

其中 $2$ 表示 key 和 value。这个公式说明：KV cache 与 batch 和上下文长度线性增长。长上下文、多并发、长输出会快速吃满显存。

传统做法常为每条序列预留连续显存。如果请求提前结束，或者实际输出长度远小于上限，就会造成内部碎片；如果要扩容一条序列，又可能因为没有足够连续空间而失败。

## PagedAttention：把 KV Cache 切成块

PagedAttention 的直觉类似操作系统虚拟内存：不要要求一条序列的 KV cache 在物理显存中连续存放，而是把它切成固定大小 block，用 block table 维护逻辑 token 到物理 block 的映射 [2]。

设 block size 为 $S_b$，序列 $i$ 的第 $t$ 个 token 对应：

$$\text{logical\_block}=\left\lfloor\frac{t}{S_b}\right\rfloor,\quad \text{offset}=t \bmod S_b$$

调度器只需要给序列分配若干物理 block，并在 attention kernel 里通过 block table 找到对应 KV。这样一来：

- 序列不需要连续显存。
- 已完成请求的 block 可以立即回收。
- 多个序列可通过 copy-on-write 支持 parallel sampling 或 beam search。
- 显存碎片显著下降。

vLLM 文档中也明确把 PagedAttention 作为其高吞吐 serving 的关键能力之一 [1][3]。

## Continuous Batching：每一步都重新组队

PagedAttention 解决内存布局，continuous batching 解决调度时机。

传统 static batching 把一组请求打包后一起跑到结束，短请求会等长请求。vLLM 的动态调度在每次模型前向边界都可以：

- 移除已完成请求。
- 加入新请求。
- 为需要继续生成的请求分配 block。
- 在显存不足时暂停或抢占部分请求。

简化伪代码：

```python
waiting = Queue()
running = []

while True:
    admit_new_requests(waiting, running, kv_budget)
    batch = build_next_iteration_batch(running)
    outputs = model_forward(batch, paged_kv_cache)
    update_sequences(outputs)
    free_finished_blocks(running)
    maybe_preempt_if_memory_pressure(running)
```

真实 vLLM 更复杂，但核心思想就是：调度单位不是“整条请求”，而是“下一次 forward iteration”。

## 源码映射：应该从哪里读

vLLM 的公开仓库和文档是最好的入口 [4]。读源码时可以按三层看：

1. **服务入口**：OpenAI-compatible server 接收请求，转成内部 request。
2. **调度层**：scheduler 管理 waiting/running 请求、token budget、KV block 分配。
3. **执行层**：worker / model runner 调用 attention kernel，读取 paged KV cache。

PagedAttention 文档提到 vLLM 使用自己的 multi-head query attention kernel，并让 key/value cache 以 block 形式存储；文档还提醒这里的 block 与 GPU thread block 不是同一个概念 [3]。

工程读者可以先读文档中的 design/paged_attention，再看仓库里的 scheduler、kv cache、attention kernel 相关模块。不要一开始就跳进 CUDA kernel，否则很容易被实现细节淹没。

## 和 HF Jobs 的关系

Hugging Face 近期介绍了用 HF Jobs 一条命令启动私有、OpenAI-compatible vLLM endpoint，用于 eval、batch generation、临时测试等场景 [5][6]。这类工具的价值建立在 vLLM 的 serving 能力之上：如果后端只是单请求 generate，临时端点很快会被并发、长上下文和显存管理拖垮。

所以今天的主线可以这样理解：

- HF Jobs 解决“如何快速拿到一个临时 endpoint”。
- vLLM 解决“这个 endpoint 为什么能承载真实推理工作负载”。

## 局限与误解

第一，PagedAttention 不是免费午餐。block table、非连续访存和调度逻辑都有开销。它的收益来自高并发、长上下文、变长输出场景；低并发短请求不一定显著受益。

第二，continuous batching 会改善吞吐，但可能影响单请求尾延迟。生产系统需要用 SLO、优先级和 preemption 策略平衡。

第三，vLLM 不是“所有模型最快”的代名词。不同模型结构、量化格式、硬件、batch 形态和采样策略都会影响结果。正确做法是用自己的 workload benchmark。

## 小光判断

生产级 LLM serving 的核心是资源编排，而不是 HTTP 封装。一个好的 serving 引擎必须同时管理请求、token、KV cache、GPU kernel、SLO 和失败恢复。

PagedAttention 与 continuous batching 的长期价值在于，它们把 LLM 推理从“单条序列生成”改造成“多租户在线系统”。这也是为什么 vLLM 会成为很多工具和平台默认选择：它不只是在跑模型，而是在管理推理工作负载。

## 总结

- KV cache 显存与并发、上下文长度、层数线性增长，是 serving 的核心瓶颈。
- PagedAttention 用 block table 管理非连续 KV cache，降低碎片和预留浪费。
- Continuous batching 在每次 iteration 后重新调度请求，提高 GPU 利用率。
- TTFT、TPOT、throughput、goodput 必须一起看，单请求延迟不足以评价系统。
- HF Jobs 这类临时 endpoint 工具的实用性，很大程度依赖 vLLM 这类后端 serving 引擎。

## 参考资料

[1] vLLM, [Documentation](https://docs.vllm.ai/), official docs  
[2] Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023  
[3] vLLM Docs, [Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/), official design doc  
[4] vLLM Project, [vllm GitHub repository](https://github.com/vllm-project/vllm), official code  
[5] Hugging Face, [Run a vLLM Server on HF Jobs in One Command](https://huggingface.co/blog/vllm-jobs), 2026  
[6] Hugging Face Docs, [Serve Models on Jobs](https://huggingface.co/docs/hub/en/jobs-serving), official docs  
[7] vLLM Blog, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025
