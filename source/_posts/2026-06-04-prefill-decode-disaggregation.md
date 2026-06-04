---
title: "从单体 Serving 到 Prefill/Decode 解耦：LLM 推理服务架构为什么正在重构"
date: 2026-06-04 09:08:08
categories:
 - AI
 - AI Infra
tags:
 - LLM
 - Inference
 - KV Cache
 - vLLM
 - SGLang
 - TensorRT-LLM
description: "LLM 推理正在从单体引擎走向 Prefill/Decode 解耦：本文拆解 prefill 与 decode 的计算差异、KV cache transfer、路由、缓存分层，以及 vLLM、SGLang、NVIDIA Dynamo 等系统的工程取舍。"
---

> 阅读时间：约 10-15 分钟  
> 主题类型：工程实践 / 架构解析  
> 关键词：LLM Serving、Prefill/Decode Disaggregation、KV Cache、vLLM、SGLang、NVIDIA Dynamo

## TL;DR

LLM serving 过去常被理解成“把模型放到 GPU 上，加一个 HTTP API”。这个理解已经不够用了。长上下文、Agent 多轮调用、工具返回、RAG 前缀复用和多租户并发，把推理服务从单机模型执行问题推成了分布式系统问题。

核心变化是：一个请求内部的 **prefill** 和 **decode** 两个阶段，计算形态非常不同。Prefill 一次性处理长 prompt，更偏计算密集；decode 每次生成一个 token，需要反复读取历史 KV cache，更容易被显存容量和内存带宽卡住。把两者放在同一组 GPU 上统一调度，简单但容易互相干扰。Prefill/Decode 解耦就是把这两个阶段拆到不同 worker，分别调度、分别扩缩容，再通过 KV cache transfer 把上下文状态接起来。

小光判断：P/D 解耦不是“所有场景都更快”的银弹。它真正适合的是长 prompt、多轮复用、prefill 干扰明显、decode worker 需要稳定 token latency 的生产场景。判断它是否值得上，关键看三件事：KV cache 传输是否足够快，路由是否足够懂 cache，业务负载是否真的存在 prefill/decode 资源错配。

## 为什么单体 Serving 开始吃力

Transformer 推理可以粗略拆成两个阶段。

第一阶段是 **prefill**：用户一次性提交 prompt，模型对所有输入 token 做前向计算，并生成每层 attention 的 key/value 状态，也就是 KV cache。这个阶段一次处理很多 token，矩阵乘法规模大，GPU 的计算单元更容易被填满。

第二阶段是 **decode**：模型开始自回归生成。每一步通常只新增一个 token，但要拿这个新 token 的 query 去和历史所有 key/value 做 attention。随着上下文变长，decode 不只是“算一个 token”，而是在不断访问越来越大的 KV cache。Multi-Query Attention 和 Grouped-Query Attention 的提出，本质上就是为了减少增量解码时 K/V 张量的内存带宽压力 [8][9]。

单体 serving 把 prefill 和 decode 混在同一个 engine 里做。它的优点是架构简单，不需要跨 worker 传 KV cache，不需要复杂路由，也不需要多套资源池。但生产负载一旦变复杂，问题会暴露出来：

- 长 prompt 的 prefill 会占住 GPU，让正在 decode 的请求 tail latency 变差。
- decode 阶段的吞吐更依赖 KV cache 管理、batching 和内存带宽，资源需求与 prefill 不对称。
- Agent 场景里，请求不是一次问答，而是多轮“模型 -> 工具 -> 模型”，上下文越来越长，间隔也不稳定。
- RAG、系统提示词、工具 schema、长文档摘要都让前缀复用和 cache 命中变得更重要。

vLLM 的 PagedAttention 论文把 KV cache 管理类比成操作系统的分页：不是为每个请求预留连续大块显存，而是按 block 管理逻辑上下文到物理 KV block 的映射，从而降低碎片和浪费 [7]。这解决的是“单体 engine 内部如何管 cache”。而 P/D 解耦进一步问的是：当 cache 需要跨 worker、跨 GPU、甚至跨内存层级流动时，serving 系统应该怎么设计？

## P/D 解耦到底拆了什么

P/D 解耦最小流程可以分成三步：

1. Prefill worker 接收 prompt，完成输入 token 的前向计算，并生成 KV cache。
2. Prefill worker 把 KV cache 通过高速通道传给 decode worker。
3. Decode worker 继续执行自回归生成，稳定服务每 token latency。

NVIDIA Dynamo 的 disaggregated serving 文档明确把这个设计动机写成：prefill 和 decode 的计算特征、内存占用不同，拆开之后可以用不同 tensor parallel 配置和资源池服务它们；长上下文 prefill 也不会阻塞正在进行的 decode 请求 [1]。vLLM 的 disaggregated prefilling 文档同样把 prefill 和 decode 放到不同 vLLM 实例，并把实现集中在 `vllm/distributed/kv_transfer` 相关模块 [2]。SGLang 也提供 PD disaggregation 文档，讨论独立运行以及接入 Dynamo 时的架构 [3]。

这意味着 LLM serving 里多了一个非常关键的中间对象：**KV cache transfer**。在单体 serving 里，KV cache 只是 engine 内部状态；在 P/D 解耦里，它变成了跨 worker 的数据面。只要 KV 传输慢，或者传输阻塞 GPU forward，解耦带来的收益就会被吃掉。

Dynamo 文档中，KV transfer 可以通过 NIXL 等机制从 prefill engine 的 VRAM 直接传到 decode engine 的 VRAM，并尽量保持非阻塞 [1]。TensorRT-LLM / Dynamo 文档也说明 disaggregated serving 下必须处理 prefill 与 decode worker 之间的 KV cache transfer，并支持不同通信后端 [4]。这类设计的核心不是“把数据搬过去”这么简单，而是要让搬运、调度、计算形成流水线。

## 路由开始变得比模型 API 更重要

单体 serving 的路由通常只关心哪个 worker 还有容量。P/D 解耦之后，路由至少要多看三类信号。

第一是 **prefill 负载**。长 prompt 请求应该进入更适合 prefill 的资源池，避免把 decode worker 拖慢。一个多租户服务里，如果某些用户经常提交超长上下文，prefill 池的隔离价值会非常明显。

第二是 **KV cache 位置**。如果请求复用了相同系统提示词、工具 schema 或文档前缀，那么“cache 在哪台机器上”会影响实际 TTFT。Dynamo 的设计里，PrefillRouter 可以根据 cache overlap score 和负载选择 worker [1]。这说明 LLM serving 的调度目标不再只是 GPU 利用率，而是 cache 命中率、传输成本和排队延迟之间的联合优化。

第三是 **decode 稳定性**。很多产品体验更关心 token 是否稳定流出，而不是首 token 是否极致快。P/D 解耦可以把 decode worker 保护起来，让它少受大 prompt prefill 的冲击。但如果路由策略错误，把大量低复用、短上下文请求也强行拆开，额外传输可能反而变成负担。

所以，P/D 解耦后的 serving 系统更像一个带状态的数据中心调度系统：请求、KV block、GPU 显存、CPU 内存、网络和存储都成为调度对象。

## KV cache 分层：从“显存对象”到“系统资源”

仅仅把 prefill 和 decode 拆开，还不够。真正的生产问题是 KV cache 太大，而且生命周期不规则。一个长上下文 Agent 任务可能中间停下来调用工具，几秒甚至几十秒后再回来；一个 RAG 服务可能反复命中相同文档前缀；一个客服机器人可能在会话级别长期复用系统指令和工具描述。

这推动 KV cache 从 GPU 显存里的临时张量，变成跨层级资源：

- GPU HBM：最快，但最贵、容量最有限。
- CPU 内存：容量大一些，适合冷一点但可能复用的 cache。
- 本地或远端存储：延迟更高，但可以服务更长生命周期或更大规模的 cache。
- 网络传输层：决定 cache 从 prefill 池到 decode 池、从缓存服务到 engine 的代价。

vLLM 的 KV offloading connector 博客强调，offload/load KV 数据要尽量与 GPU 上的模型计算并行，避免单纯把显存压力转移成传输等待 [5]。2026 年 vLLM 与 Novita AI 的 PegaFlow 博客则进一步把 external KV cache service 做成独立 Rust 进程，并通过 external KV connector 接入 vLLM [6]。这类方向值得关注，因为它说明 serving 引擎正在把 KV cache 管理接口外部化：engine 负责执行，cache service 负责状态流动。

小光认为，这会带来一个重要工程分层：未来的高质量 LLM serving 不只是“选择 vLLM 还是 TensorRT-LLM”，而是要同时回答：

- 用什么 engine 执行模型？
- 用什么 router 决定 prefill/decode 分配？
- 用什么 KV connector 搬运和复用状态？
- 用什么缓存策略决定哪些上下文值得保留？
- 用什么指标衡量 TTFT、ITL、吞吐、cache 命中和 tail latency？

## vLLM、SGLang、Dynamo 的共同方向

这些系统实现细节不同，但趋势相当一致。

vLLM 的优势在于生态广、PagedAttention 和连续 batching 影响力强，近年来持续补齐 disaggregated prefill、KV transfer、external KV connector、KV offloading 等能力 [2][5][6][7]。它适合关注开源生态、模型覆盖和实验速度的团队。

SGLang 更强调面向结构化生成、复杂 serving workload 的运行时和调度能力。PD disaggregation 放进 SGLang 语境里，不只是推理加速，也和多请求调度、前缀复用、复杂程序化生成有关 [3]。

NVIDIA Dynamo 更像是把 disaggregated serving、KV-aware routing、KV offloading 和多后端执行统一到一个 AI inference serving 框架里。它的文档明确把 disaggregated serving、KV cache-aware routing、KV cache offloading 列为关键组成 [10]。如果团队已经在 NVIDIA GPU、TensorRT-LLM、Kubernetes 和大规模集群上投入较深，Dynamo 方向值得跟踪。

但不要把这些系统理解成互斥选项。更合理的视角是：P/D 解耦正在成为 serving 架构的一种通用模式，不同框架会围绕它提供不同抽象层。

## 什么时候不该急着上 P/D 解耦

P/D 解耦听起来高级，但它有明确代价。

第一，系统复杂度上升。你需要部署 prefill worker、decode worker、router、KV transfer 组件、监控指标和故障恢复逻辑。单体 serving 的问题往往是“GPU 利用率不够好”，解耦后的问题会变成“分布式状态是否一致、cache 是否传对、路由是否震荡、某个阶段是否成为瓶颈”。

第二，KV transfer 不是免费午餐。短 prompt、短输出、低复用的请求，拆开后可能还不如单体执行。业界也有一些实验观察到：在随机数据、KV 命中率低或传输瓶颈明显时，disaggregation 不一定赢。这个判断不能靠架构图，要靠 workload trace 和压测。

第三，观测难度变高。传统指标如 QPS、平均延迟不够用了。至少要拆出 TTFT、TPOT/ITL、prefill queue time、decode queue time、KV transfer latency、cache hit rate、GPU KV usage、tail latency。否则你很可能不知道收益来自哪里，也不知道问题卡在哪一层。

## 小光判断

P/D 解耦代表 LLM serving 的重心变化：从“单个模型实例尽量跑满”转向“围绕上下文状态做系统级调度”。这和数据库、操作系统、CDN 的演进有点像：早期只关心计算，后来真正拉开差距的是缓存、路由、隔离、调度和可观测性。

如果你是 AI 工程师，可以用下面的检查表判断是否值得引入 P/D 解耦：

- prompt 长度分布是否高度长尾？
- prefill 是否经常阻塞 decode，造成 token 流不稳定？
- 是否存在高复用前缀，例如系统 prompt、工具 schema、固定文档库？
- KV cache 是否已经成为显存容量或内存带宽瓶颈？
- 业务是否重视 tail latency，而不只是平均吞吐？
- 团队是否有能力维护 router、cache transfer、压测和观测体系？

满足越多，P/D 解耦越值得认真评估。反过来，如果只是小模型、短上下文、低并发、低复用，先把单体 serving 的 batching、PagedAttention、量化、prefix caching 和监控做好，往往更划算。

## 总结

LLM serving 的下一阶段，不再只是模型引擎竞争，而是围绕 KV cache 的分布式系统竞争。Prefill/Decode 解耦把 prefill 的计算密集特征和 decode 的内存带宽特征拆开，让系统可以更精细地做资源配置、路由和缓存复用。

它的关键不是“解耦”这个词，而是三个工程问题：

1. KV cache 能不能低延迟、非阻塞地跨 worker 移动。
2. Router 能不能理解 cache 位置、负载和 tail latency。
3. Workload 是否真的存在 prefill/decode 错配与前缀复用价值。

昨日我们讲的是 Agentic Inference 为什么改变成本曲线。今天这篇更底层：当上下文变成长生命周期资源，LLM 推理服务自然会从单体引擎，走向带状态、带路由、带缓存层级的推理操作系统。

## 参考资料

[1] NVIDIA Dynamo Documentation, Disaggregated Serving, https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving  
[2] vLLM Documentation, Disaggregated Prefilling, https://docs.vllm.ai/en/latest/usage/disagg_prefill.html  
[3] NVIDIA Dynamo Documentation, Disaggregation with SGLang, https://docs.dynamo.nvidia.com/dynamo/dev/backends/sg-lang/disaggregation  
[4] NVIDIA Dynamo Documentation, KV Cache Transfer in Disaggregated Serving, https://docs.nvidia.com/dynamo/archive/0.8.1/backends/trtllm/kv-cache-transfer.html  
[5] vLLM Blog, Inside vLLM's New KV Offloading Connector, https://vllm.ai/blog/kv-offloading-connector  
[6] vLLM Blog, vLLM x Novita AI: PegaFlow for Production-Grade External KV Cache, https://vllm.ai/blog/2026-05-18-pegaflow  
[7] Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, arXiv, 2023, https://arxiv.org/abs/2309.06180  
[8] Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, arXiv, 2019, https://arxiv.org/abs/1911.02150  
[9] Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, arXiv / EMNLP, 2023, https://arxiv.org/abs/2305.13245  
[10] NVIDIA Dynamo Documentation, Introduction, https://docs.dynamo.nvidia.com/dynamo/getting-started/introduction
