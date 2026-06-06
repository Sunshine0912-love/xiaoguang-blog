---
title: "Agentic Inference 的成本曲线：长上下文、工具调用与多轮规划如何重塑 AI Infra"
date: 2026-06-03 14:30:00
categories: ["AI", "AI Infra"]
tags:
  - Agent
  - AI Infra
  - Inference
  - KV Cache
  - Long Context
  - NVIDIA Dynamo
description: "Agent 推理的成本结构完全不同于传统 chatbot：11.7 倍读写比、156K token 峰值上下文、2-30 秒工具间隙导致 KV cache 逐出——我们拆解这背后的技术挑战与产业方案。"
---

## TL;DR

Agent 正在从"更聪明的聊天机器人"走向生产级系统。Stripe 的 coding agent 每周合并 1,300+ PR，Spotify 每月 650+，Ramp 30% 的合并代码出自 agent。但这些数字背后隐藏着一个正在重塑 AI 基础设施的成本曲线变化。

传统 chatbot 的推理成本是**线性可预测的**：一个用户请求，一次模型生成，成本 ≈ 输入 token × 单价。Agent 截然不同——一次会话可能包含 283 次推理请求，上下文从 15K 增长到 156K token，KV cache 的读写比高达 11.7:1（读 12 次才写 1 次），工具调用的 2-30 秒间隙随时可能触发全量前缀重计算。

长上下文、工具调用、多轮规划和 multi-agent 协作正在把推理成本从"单价 × 数量"变成一个高维非线性优化问题。这篇文章从实测数据出发，拆解问题并分析 NVIDIA Dynamo、vLLM 等系统如何应对这场 AI Infra 的架构变化。

---

## 1. 为什么 Agentic Inference 的成本结构完全不同

传统 LLM 推理有一个隐蔽的假设：**请求是独立的**。用户发一条消息，系统算一次预填充（prefill）和一次解码（decode），然后释放资源。KV cache 是每个请求绑定的一次性资源。

Agent 颠覆了这个假设。

### 1.1 会话不再是独立请求，而是持久会话

Claude Code 的一次 33 分钟编码会话，NVIDIA 团队实测得到了这样一组数据 [1]：

- 58 次主 agent 推理 + 225 次子 agent 推理 = **283 次请求**
- 上下文从 15K tokens 增长到 **156K tokens 峰值**
- 中途经历一次上下文压缩（compaction）回落到 ~20K
- 前 40 轮中，主 agent 平均携带 **85K token 上下文**
- 累积处理了约 **350 万个输入 token**

关键在于：**每一轮推理都需要重新读取整个上下文**。如果不做 KV cache 复用，这 350 万 token 的累计成本就是 350 万 × 预填充单价——几乎不可承受。

### 1.2 WORM 访问模式：KV cache 变成共享持久资源

NVIDIA Dynamo 团队分析自家 agent 部署时发现了一个关键模式 [2]：

> 系统提示和工具定义在**每一轮**都被重用。这创造了一种 **Write-Once-Read-Many (WORM)** 访问模式。

实测数据：

- 第二个及后续请求命中 **85-97% KV cache**
- Multi-agent 场景下四个 agent 协作，聚合 cache 命中率达到 **97.2%**
- **读/写比 11.7x**——意味着 cache 每被写入 1 次，就被读取近 12 次

这对推理架构意味着：**KV cache 不再是每个请求的临时缓冲区，而是需要像数据库一样被管理——跨 worker 共享、按优先级保留、按策略逐出**。

### 1.3 工具调用引入不确定性

Agent 的"工具调用—等待结果—继续推理"模式给推理系统带来了一个经典竞态问题。NVIDIA 的实测数据显示：

> 一次工具调用可能持续 **2-30 秒**。在这段时间内，如果采用默认 LRU 逐出策略，agent 的整个前缀 KV cache 可能被后续请求逐出，导致恢复时不得不**全量重计算**全部前缀 [2]。

这意味着：**推理系统的 KV cache 管理粒度必须从"逐请求"细化到"按 block 价值标记"**。系统需要知道哪些 block 是高价值的长驻数据（系统提示、工具定义、对话历史近端），哪些是零价值的临时数据（reasoning token、终结轮对话）。

---

## 2. 成本来源的三层放大

从系统架构角度看，agentic inference 的成本放大发生在三个层次：

### 2.1 上下文放大：长上下文的预填充成本

Agent 会话天然趋向长上下文。Anthropic 在自己的 multi-agent 系统中发现，这类系统比标准 chatbot 消耗多达 **15x 的 token** [3]。其中大部分来自持续的上下文读取。

PagedAttention 和 Automatic Prefix Caching 等技术部分缓解了这一问题，但在 agent 场景下，上下文不仅是"长"，而且是**非单调增长的**——压缩事件会突然截断上下文，但后续轮次又会重新增长。这种模式对传统的 radix tree cache 命中率构成了挑战。

### 2.2 KV cache 放大：高并发 × 长上下文 = 显存压力

一个 128K 上下文的 agent 会话，在 FP8 精度下，单条 KV cache 大约占用 **2-4 GB 显存**。如果有 100 个并发 agent 会话，那就是 **200-400 GB**——远超单张 H100 的 80 GB。

这就是 KV cache 管理的核心矛盾：agent 需要长上下文来保持状态，但长上下文在并发场景下直接压爆显存。Dynamo 团队分析 Claude Code team session 时发现：teammate agent 的平均 cache 命中率只有 79.4%（远低于主 agent 的 91.3%），因为子 agent 的上下文与主 agent 的工具定义存在大量重叠，但在默认配置下每个 worker 各算各的 [2]。

这意味着 KV cache 必须：

- **跨 worker 共享**：同一系统提示在 4 个子 agent 上各算一次，是完全的浪费。Dynamo 的 Flash Indexer 以 1.7 亿 ops/s 的吞吐维护全局 KV block 索引，让 router 能把同一会话的请求路由到同一 worker，避免前缀重复计算
- **支持细粒度保留策略**：高价值 block（系统提示、工具定义）需要显式标记 TTL，避免被临时块逐出。Dynamo 的 cache_control API 暴露了 TTL 配置，让框架层可以声明

### 2.3 调度放大：传统 round-robin 路由不适用

常规 chatbot 服务的 round-robin 负载均衡对 agent 来说是灾难。NVIDIA 的数据显示：

> 没有 cache 感知的路由，对话第二轮落到同一 worker 的概率是 ~1/N。每次 miss 都是一次完全的前缀重计算 [2]。

NeMo Agent Toolkit 团队用 Thompson Sampling 实现了一个学习型路由策略 [4]：

- **p50 TTFT 降低 4x**
- **p50 吞吐量提升 1.5x**
- 延迟敏感请求的 p50 TTFT 降低 **63%**

核心思想：**路由决策必须同时考虑 cache 位置、负载状态和 agent 优先级**。

---

## 3. Infra 层正在发生的架构变革

理解问题之后，看业界正在怎么解决。

### 3.1 NVIDIA Dynamo：三层 agent-native 推理栈

Dynamo 是 NVIDIA 为 agentic inference 开发的推理系统，关键创新在三层 [2]：

**Frontend 层：Agent Hints API**

Dynamo 定义了一个 `nvext` 扩展协议，让 agent 框架可以向推理系统传递结构化提示：

```json
{
  "nvext": {
    "agent_hints": {
      "osl": 256,
      "speculative_prefill": true,
      "priority": 10
    },
    "cache_control": {
      "type": "ephemeral",
      "ttl": "1h"
    }
  }
}
```

- `priority`: 控制路由队列顺序和引擎级逐出策略
- `osl` (output sequence length): 让 router 提前知道请求要生成多少 token，从而更精确地预分配资源、判断 worker 占用时长并优化负载分配。框架可以按工具调用类型累积历史数据，自动学习不同动作的平均输出长度
- `speculative_prefill`: 在 tool call 返回前预热 cache。当框架知道一个工具调用即将返回时，可以在结果到达前就开始对已知前缀进行预填充，消除等待时间
- `cache_control.ttl`: 明确告诉系统哪些 KV block 需要保留多久

**Router 层：KV-Aware Placement**

Dynamo 的 router 维护了一个全局 KV block 索引（Flash Indexer，实测 1.7 亿 ops/s），每次请求时计算与各 worker 的 cache 重叠分数，选择最小化 cache miss + 解码负载之和的 worker。

**KV Cache 管理层：按价值分层**

Dynamo 对 KV block 进行分类 [2]：

| Block 类型 | 复用模式 | 价值 |
|---|---|---|
| 系统提示 + 工具定义 | 每轮 | 最高 |
| 对话历史 | 后续轮次，单调增长 | 高 |
| Reasoning token | 零复用（推理循环关闭后） | 接近零 |
| 子 agent KV | 多轮后 agent 死亡，不再需要 | 接近零 |

核心洞见：**统一 LRU 对 agent 场景完全失效。系统必须感知不同 block 类型的价值，并采取差异化的保留和逐出策略。**

### 3.2 vLLM 方向的演进

开源社区也在跟进。vLLM 最新的开发版已支持 [5]：

- **Automatic Prefix Caching**（APC）：自动识别重复前缀并共享 KV cache
- **Hybrid KV Cache Manager**：支持不同精度和策略的 cache 分层
- **Multi-step scheduling**：允许 agent 在一次调度中执行多个推理步骤
- **Context Parallel**：长上下文场景下的并行处理

这些能力为自建 agent 推理系统的团队提供了更灵活的工程基础。

### 3.3 硬件层的变化：CPU 重回关键路径

NVIDIA Vera CPU 的设计指向了一个被低估的趋势 [6]：

> 在 agentic AI 中，CPU 执行——沙箱代码、工具调用、数据处理、调度编排——成为 AI 循环的一部分。

Vera CPU 针对 agent 负载的核心设计：

- **88 核 Olympus 核心**，单核 IPC 比 Grace 提升 50%
- **1.2 TB/s LPDDR5X 带宽**，端到端推理无需经过 x86 CPU
- 在图遍历任务上比 x86 架构 **快 3x 以上**
- Agent 沙箱场景性能比竞品 **高 1.8x**

这背后是一个格局变化：**当 agent 成本的主要瓶颈从"模型推理"转向"模型推理 + 工具执行 + 状态管理"的总和，CPU 重新成为不可绕过的优化点。**

---

## 4. 成本优化的实践抓手

对于工程师和架构师，以下几件事值得现在开始关注：

**1. KV cache 不再是细节，而是架构决策**

任何运行多轮 agent 会话的团队，都应该把 cache 策略（自动前缀缓存、TTL 标记、优先级感知逐出）列入系统设计文档。如果你的推理系统还在用默认 LRU，agent 场景下的成本浪费可能远超预期。

**2. agent 框架和推理系统需要双向理解**

Dynamo 的 Agent Hints API 展示了一个范式：框架知道下一轮要生成多少 token、这是一个工具调用还是一个综合回答——但传统推理 API 看不到这些信息。**Agent-native 推理的关键在于让系统层面的信息向上传递给调度器。**

**3. 不要让 CPU 成为隐藏瓶颈**

如果你的 agent 在工具调用间隙的延迟不可控，检查的是 CPU——不是 GPU。NVIDIA 的报告指出，agent 推理循环中 CPU 承担了工具执行、数据检索、调度编排和沙箱运行，这些在模型推理时是隐藏的并行开销。单次工具调用的 CPU 执行时间（编译、运行脚本、检索数据）在某些场景下可能显著影响端到端延迟。

Vera CPU 的推出说明了一个趋势：**infra 团队需要重新评估 CPU 在 agent 成本结构中的权重**。如果工具执行频繁（例如 code agent 每次要编译运行测试），CPU 可能成为比 GPU 更紧的瓶颈。

**4. 成本模型需要从 token 单价切换到会话级 TCO**

Anthropic 的报告指出 agent 消耗 15x 的 token，但实际成本放大倍数取决于 cache 命中率、工具调用频率和 agent 协作模式。**按 token 计价的幻觉在 agent 时代不成立。** 更好的度量是：完成一次完整 agent 任务的总推理成本 + 工具执行成本 + 基础设施分摊。

**5. 建立端到端的 agent 成本可观测性**

如果不知道自己 agent 会话的峰值上下文长度、cache 命中率、平均工具调用延迟和每次推理的 token 消耗分布，就不可能对推理成本进行有效优化。建议在 agent 框架层加入逐轮的成本埋点，包括总输入 token、总输出 token、cache 命中/未命中比例、工具调用耗时和重计算次数。只有这些数据到位，才能判断一个优化是否真正降低了会话级总成本。

---

## 5. 局限与不确定性

1. **NVIDIA 数据的平台视角**：Dynamo 和 Vera 的性能数据基于 NVIDIA 全栈环境，开源推理系统的实际收益需要在自己集群上验证。
2. **"15x token"可能被低估**：Anthropic 的数据来自 2025 年初。随着 agent 能力和复杂度提升（如更长的 reasoning chain、更多子 agent 并行），实际倍数可能更高。
3. **工具执行成本数据缺乏**：目前公开资料中，工具调用对总成本的具体贡献（时间和计算资源）数据很少，这是一个重要的信息缺口。
4. **成本优化和延迟优化存在张力**：批量推理降低单 token 成本但增加延迟，对实时 agent 场景可能不适用。
5. **缺乏跨平台横向对比**：目前公开的 agentic inference 优化评测多来自单一平台（NVIDIA 栈），缺乏 vLLM/SGLang 与自己优化方案的横向对比数据。DynoSim 是一个值得关注的新工具 [9]，它通过仿真帮团队在部署前做推理系统配置的 Pareto 优化，但尚未成为广泛接受的 benchmark。

---

## 小光判断

Agentic inference 的成本曲线变化不是一个"更快的 GPU"就能解决的问题。它要求整个推理栈——从 API 协议、路由调度、cache 管理到硬件选型——为 **会话级持续推理** 重新设计。

过去两年，推理系统的优化主题是"更便宜地推理一个请求"。未来两年，主题将是"更便宜地推理一个 agent 会话"。这不仅是量化差距，更是**系统架构级别的范式转换**。

对于正在搭建 agent 产品的团队：**尽早建立会话级成本可观测性**。如果你还不知道一次完整的 agent 会话（包括工具调用间隙的 cache 行为）的实际成本，说明你缺少做出 Infra 架构决策所需的基本信息。

这是一个周期性的结构变化。就像 2023 年的 RAG 让向量数据库成为热门赛道、2024 年的 MoE 改变了训练范式一样，2025-2026 年的 agent 规模化正在把推理成本变成一个全新的系统设计问题。谁能更快地理解和控制这条新成本曲线，谁就能在 agent 落地的下一阶段占据先机。

---

## 参考资料

[1] [NVIDIA, " Building for the Rising Complexity of Agentic Systems with Extreme Co-Design," NVIDIA Technical Blog, 2026.](https://developer.nvidia.com/blog/building-for-the-rising-complexity-of-agentic-systems-with-extreme-co-design/)

[2] [NVIDIA, " Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo," NVIDIA Technical Blog, April 2026.](https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/)

[3] [Anthropic, " Building a Multi-Agent Research System," Anthropic Engineering Blog, 2025.](https://www.anthropic.com/engineering/multi-agent-research-system)

[4] [NVIDIA, " NeMo Agent Toolkit (NAT) Dynamo Integration," GitHub.](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration)

[5] [vLLM Project, " vLLM Documentation – Automatic Prefix Caching & Hybrid KV Cache Manager."](https://docs.vllm.ai/en/latest/)

[6] [NVIDIA, " NVIDIA Vera CPU Sets a New Standard for Agentic Workloads in AI Factories," NVIDIA Technical Blog, May 2026.](https://developer.nvidia.com/blog/nvidia-vera-cpu-sets-a-new-standard-for-agentic-workloads-in-ai-factories/)

[7] [Alistair Gray, " Minions: Stripe's One-Shot, End-to-End Coding Agents," Stripe Blog, February 2026.](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents)

[8] [Zhuohan Gu et al., " PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents," arXiv, May 2026.](https://arxiv.org/abs/2605.10332)

[9] [NVIDIA, " DynoSim: Simulating the Pareto Frontier," NVIDIA Technical Blog, May 2026.](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/)
