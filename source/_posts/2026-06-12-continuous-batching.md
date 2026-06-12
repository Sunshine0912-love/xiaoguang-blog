---
title: "连续批处理(Continuous Batching)机制详解：从静态 Padding 到 Iteration-Level 调度的 GPU 利用率优化"
date: 2026-06-12 08:00:00
mathjax: true
categories: ["AI", "AI Infra"]
tags:
 - Continuous Batching
 - vLLM
 - LLM Inference
 - Scheduling
 - Throughput
 - GPU Utilization
description: "从静态batching的padding浪费出发，系统拆解Continuous Batching的iteration-level调度机制、vLLM请求状态机、与PagedAttention的协同，以及2-10x吞吐提升的数学原理。"
topic_id: "TECH-2026-06-12-03"
---

> 阅读时间：约 12-18 分钟
> 主题类型：工程优化 / 调度机制
> 关键词：Continuous Batching、Iteration-Level Scheduling、vLLM、Orca、GPU Utilization、PagedAttention

## TL;DR

传统 LLM 推理 serving 的一条批处理请求里，如果有一个请求生成 500 token 而另一个只需要 10 token，后者就必须等前者完成后才能释放——此时 GPU 继续为那个已经结束的请求计算 padding 填充，核心计算单元在"空转"。Continuous Batching（连续批处理）改了一句调度规则：**不要把"整条请求"当作调度单元，把"一次模型前向（iteration）"当作调度单元**。每个 iteration 结束时，已完成请求可以离开、新到达请求可以加入，batch 的成员随时演化。

一句话讲清楚：

> 静态 batching 锁死 batch，Continuous Batching 让 batch 在每个 iteration 边界"呼吸"——完成的走，新来的进，GPU 不再为 padding 浪费 Flops。

本文从静态 batching 的 GPU 浪费量化入手，系统拆解 iteration-level scheduling 的核心思想、Orca 的 selective batching、vLLM 调度器源码映射（请求状态机、block table、WAITING/RUNNING/SWAPPED 三态迁移），并给出吞吐提升 2-10x 的定量解释。文末说明 Continuous Batching 和 Prefill/Decode 解耦的区别：前者是调度粒度从 request 降到 iteration，后者是计算阶段的空间分离——两个维度可以叠加，但解决不同层面的问题。

## 前置知识

读者需要了解 LLM 推理的 **prefill** 和 **decode** 两个阶段的基本区别：prefill 一次性处理输入 prompt 的所有 token，计算密集（矩阵乘 bound）；decode 每次只生成一个 token，但需要读取整个 KV cache，内存带宽 bound。本文不讨论 prefill/decode 的架构分离，仅讨论同一推理引擎内部 **如何在迭代间动态变化 batch 成员**。

## 问题定义

### 静态 Batching 的 GPU 浪费

在静态 batching 模式下，服务系统将一组请求打包成一个 batch，提交给模型执行。关键约束是：**batch 中所有请求必须同时开始、同时结束**。这带来两个直接浪费。

**浪费一：Padding 计算。** 不同请求需要生成不同数量的 token。假设 batch 中有 $N$ 个请求，第 $i$ 个请求需要生成 $L_i$ 个 token。静态 batching 必须将每个请求填充到 $\max_i L_i$ 的长度。

定义 $T_{\text{padding}}$ 为解码阶段因 padding 额外执行的 token 计算量（以一次 decode 前向为一个单位）：

$$T_{\text{padding}} = \sum_{i=1}^{N} (\max_j L_j - L_i)$$

引入 GPU 利用率的概念。一次 decode 迭代中，GPU 实际执行的有效 token 前向次数为 $N_{\text{active}}$（即该 iteration 内仍有 token 需要生成的请求数），而 batch 固定大小为 $N$。设 batch 生命周期（按 iteration 计）为 $S = \max_i L_i$，则平均 GPU 利用率为：

$$\text{Util}_{\text{static}} = \frac{\sum_{k=1}^{S} N_{\text{active}}(k)}{N \cdot S}$$

其中 $N_{\text{active}}(k)$ 是第 $k$ 次 iteration 时尚未完成的请求数，随已完成请求逐步减少。最差情况下（一条极端长请求 + 多条极短请求），$\text{Util}_{\text{static}}$ 可以跌到 $1/N$ 以下。

以具体数字说明：batch 中有 4 个请求，分别需要生成 500、10、10、10 个 token。静态 batching 下，batch 大小恒为 4，生命周期 $S = 500$ 次 iteration。在第 11 次 iteration 时已有 3 个请求完成，但 batch 中仍有 4 个位置——其中 3 个在执行无意义的 padding 计算。前 10 次 iteration 利用率 100%，后 490 次 iteration 利用率仅 25%（只有 1/4 的请求仍在执行有效计算）。平均有效利用率为：

$$\text{Util}_{\text{static}} = \frac{4 \times 10 + 1 \times 490}{4 \times 500} = \frac{530}{2000} = 26.5\%$$

**浪费二：请求排队。** 新到达的请求必须等待当前 batch 完全结束才能被调度。在 batch 执行期间到达的请求全部排队，造成额外的排队延迟。

### 为什么这个问题在 LLM 推理中特别严重

原因来自自回归生成的 **multi-iteration** 特性。Transformer 的 encoder-only 推理（如 BERT）是 single-iteration：输入进、输出出，一次完成。静态 batching 对这种场景尚可接受。但 LLM 的 decode 阶段是迭代式的——每次前向只生成一个 token，一条请求可能需要数百到数千个 iteration。不同请求的迭代次数差异巨大（受温度、Max Tokens、Early Stopping 影响），导致"长短混跑"成为常态。

目标：设计一种调度策略，使得 GPU 在每个 iteration 都尽可能满载有效计算，不因"等别人完成"而空转，也不让新请求因"batch 未结束"而阻塞。

## 核心机制：Iteration-Level Scheduling

### 思路：调度粒度从请求降到迭代

静态 batching 的调度粒度是"请求"——一个 batch 绑定了 $N$ 个请求，直到最慢的那个完成。Continuous Batching 的思想来自 Orca 论文 [1] 提出的 **iteration-level scheduling**：将调度粒度从"请求"降为"一次迭代"。具体规则是：

1. **每个 iteration 结束后**，检查 batch 中哪些请求已完成（生成了 EOS 或达到 max_tokens）。
2. **立即将完成的请求移出 batch**，返回结果给客户端。
3. **从等待队列中取新请求加入 batch**，填充空出的位置。
4. 如果某个请求的 prompt 很长，还可以通过 **chunked prefill** 将其 prefill 拆分到多个 iteration 中执行，避免长 prefill 阻塞 decode 请求。

伪代码如下：

```python
# Iteration-level scheduling 核心循环
waiting_queue: list[Request]   # 待处理的请求
running_batch: list[Request]   # 当前正在执行的请求

while True:
    # Step 1: 执行一次模型前向
    outputs = model.forward(running_batch)
    
    # Step 2: 移除已完成的请求
    for req in running_batch:
        if req.finished():
            return_result_to_client(req)
    running_batch = [req for req in running_batch if not req.finished()]
    
    # Step 3: 从等待队列加入新请求（尽可能填满 batch）
    while len(running_batch) < max_batch_size and waiting_queue:
        req = waiting_queue.pop(0)
        running_batch.append(req)
    
    # Step 4: 每个请求根据自身状态决定此 iteration 计算的 token
    for req in running_batch:
        req.prepare_tokens_for_this_iteration()
```

这个循环的关键是：**每次 iteration 之后 batch 的组成都可以变化**——这在工程上被称为"连续"批处理，因为 batch 在每个迭代边界连续演化，而不是等待整批请求全部完成。

### 与静态 Batching 的结构对比

| 维度 | 静态 Batching | Continuous Batching |
|------|--------------|---------------------|
| 调度粒度 | 请求级 | 迭代级 |
| Batch 变更时机 | 一批完成 -> 下一批开始 | 每个 iteration 后 |
| 已完成请求 | 必须等待同批次其他请求 | 立即退出并返回 |
| 新到达请求 | 排队等下一批 | 可在下一个 iteration 加入 |
| Padding 浪费 | 有（长短不均） | 无（请求只占有效位置） |
| 排队延迟 | 高（最坏等一批时间） | 低（最坏等一个 iteration） |

### Selective Batching：Orca 的工程创新

Orca 在提出 iteration-level scheduling 的同时，还引入了一个关键的工程洞察：**并非所有算子都需要 batching**。

在 Transformer 的 decode 阶段，自注意力（self-attention）的算子确实从 batching 中受益——将多个请求的 query、key、value 矩阵拼接后做统一的矩阵乘法。但其他算子（如 layer norm、token embedding、LM head）对每个请求的处理是独立的，不需要 batch 维度。

Orca 的 selective batching 只对 attention 的矩阵乘法算子做 batching（即 `BatchMatMul`），其他算子逐个请求串行执行。这保证了即便在请求动态加入/离开的"非规整" batch 形状下，核心计算路径仍然能享受到 batching 带来的 GPU 并行度 [1]。

这一设计在 vLLM 中得到了进一步发扬。vLLM 的 attention 后端（PagedAttention）本身就以 token 为粒度管理 KV cache，天然支持变长序列的打包 batch 计算，使得 selective batching 不再需要逐个算子区分——batch 层面直接支持变长序列即可 [2]。

## vLLM 调度器详解

vLLM 在 Orca 的 iteration-level scheduling 基础上，结合 PagedAttention 的虚拟内存式 KV cache 管理，实现了一套完整的请求调度系统。本节映射 vLLM V1 调度器源码 (`vllm/v1/core/sched/scheduler.py`) 中的关键设计 [7]。

### 请求状态机

vLLM 调度器为每个请求维护明确的状态，状态转换如下：

```
新请求到达
    │
    ▼
┌──────────┐    调度分配 KV blocks    ┌──────────┐
│ WAITING  │ ───────────────────────▶ │ RUNNING  │
└──────────┘                          └──────────┘
     ▲                                      │
     │              抢占（显存不足）          │
     │             ┌───────────────────────▶ │
     │             │                         │
     │    ┌────────┴──────┐                 │
     └────│   SWAPPED     │◀────────────────┘
          └───────────────┘                  完成（EOS/max_tokens）
                                                    │
                                                    ▼
                                                 返回客户端
```

三个核心状态：

- **WAITING**：请求已接收，等待调度器分配 KV cache blocks。所有新请求进入此状态。
- **RUNNING**：请求已获得 KV cache blocks，正在参与迭代执行。每个 iteration 后检查是否完成。
- **SWAPPED**：请求被抢占（preempted），其 KV cache 被交换出 GPU 显存。当 GPU 显存释放后，可以恢复回 WAITING 重新调度。

vLLM 在源码中使用 `RequestStatus` 枚举来管理这些状态。调度器维护三个队列：

```python
# vllm/v1/core/sched/scheduler.py (简化)
self.waiting: RequestQueue      # 等待队列
self.running: list[Request]     # 运行队列
```

抢占后进入 SWAPPED 状态的请求由 `KVCacheManager` 管理，在显存充裕时通过 `_update_requests_with_incoming_blocks` 恢复 [7]。

### Block Table：连接调度与显存管理

每个请求在获得 KV cache blocks 后，调度器为其维护 **block table**——这是一个映射表，将逻辑 token 位置映射到物理 GPU 显存块。这与操作系统的页表（page table）完全同构。

PagedAttention 的 block table 使得调度器在"请求加入/离开 batch"时无需移动或重新分配其他请求的 KV cache。新增一个请求只需要分配新的物理 blocks、更新 block table；移除一个请求只需要释放其 blocks 即可。

```python
# Block table 概念映射（简化）
# req.block_table: [物理block_id_0, 物理block_id_1, ..., 物理block_id_k]
# 每个 block 存储 block_size 个 token 的 KV cache
```

### 混合调度：Prefill Chunking

vLLM 的调度器不做 prefill/decode 的硬性二元区分。核心抽象是：**每个请求维护一个 `num_computed_tokens` 计数器**，表示该请求已经完成模型计算的 token 数量。调度器在每次 iteration 中为每个请求分配需要计算的 token 数，目标是让 `num_computed_tokens` 追上 `num_tokens_with_spec`（prompt + output + spec tokens 总数）。

如果 `num_computed_tokens < num_prompt_tokens`，该请求仍有未处理的 prompt token，此时执行的是 prefill 逻辑（有因果注意力 mask）。如果 $num\_computed\_tokens \ge num\_prompt\_tokens$，每次只处理一个新 token，此时执行的是 decode 逻辑（单 token 生成）[7]。

```python
# vllm/v1/core/sched/scheduler.py: schedule() 中的注释（简化）
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

当开启 chunked prefill 时，一个长 prompt 请求可以在多个 iteration 中被逐步处理，每次计算 $C$ 个 token（$C$ 为 chunk size）。这使得长 prefill 不会独占 GPU、阻塞其他已在 decode 的请求。

### 调度循环源码映射

vLLM V1 调度器的 `schedule()` 方法按以下顺序调度 [7]：

1. **处理 RUNNING 请求**：遍历 `self.running`，对每个请求判断是否还有 token 需要计算。如果有，分配 token_budget 并保持在 RUNNING。
2. **处理 WAITING 请求**：从 `self.waiting` 中取新请求，尝试分配 KV cache blocks。如果分配成功，加入 RUNNING；如果显存不足，尝试抢占已有 RUNNING 请求（preemption）。
3. **抢占逻辑**：选择 RUNNING 中最不"划算"的请求（如生成进度最靠后的），将其 blocks 释放，请求状态退回 SWAPPED。

调度器的核心输出是 `SchedulerOutput`，包含：
- `scheduled_new_reqs`：新加入 batch 的请求
- `scheduled_running_reqs`：继续执行的请求
- `scheduled_cached_reqs`：通过 prefix caching 命中已缓存的请求
- `finished_req_ids`：本轮完成的请求 ID 集合

```python
# SchedulerOutput 结构（vllm/v1/core/sched/output.py 简化）
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_running_reqs: list[CachedRequestData]
    scheduled_cached_reqs: list[CachedRequestData]
    num_scheduled_tokens: dict[str, int]
    finished_req_ids: set[str]
    # ... 更多字段
```

### Token Budget 管理

调度器不直接管理"batch size"，而是管理"token budget"——每个 iteration 中允许参与计算的最大 token 数。这个设计来自 vLLM 对 chunked prefill 和变长序列的支持。token budget 由配置 `max_num_batched_tokens` 控制。

假设 token budget 为 8192，则调度器可以在一个 iteration 内同时处理：
- 8 个各生成 1 个 token 的 decode 请求（8 tokens）
- 1 个处理 2048 个 token 的 prefill chunk（2048 tokens）
- 若干其他 decode 请求

总计不超过 8192 个 token，且所有 token 通过 PagedAttention 的有效序列部分参与计算，不产生 padding。

## 与 PagedAttention 的协同

Continuous Batching 和 PagedAttention 解决的是不同层次的瓶颈，但组合后产生 1+1>2 的效果 [2][5]。

| 机制 | 解决什么 | 瓶颈层 |
|------|---------|--------|
| Continuous Batching | 请求级调度效率：允许迭代间自由加入/离开 batch | 调度 / 服务层 |
| PagedAttention | KV cache 显存碎片：允许任意"拼接"KV cache blocks | 显存管理层 |

**为什么必须组合？** 如果只有 Continuous Batching 而没有 PagedAttention，请求频繁加入/离开 batch 会导致 KV cache 的物理显存因碎片化而利用率低下（类似操作系统没有虚拟内存时的外部碎片问题）。因为不同请求生成不同数量的 token，它们的 KV cache 大小不同且动态增长，在物理显存中按连续区域分配会留下大量不可用的小碎片。PagedAttention 的 block 化内存映射解决了这个问题——无论请求长度如何，KV cache 都以固定大小 block 为单位分配和释放。vLLM 论文的实验表明，PagedAttention + Continuous Batching 的组合使 vLLM 相比使用静态 batching 的 FasterTransformer 和 Orca（仅 Continuous Batching 无 PagedAttention）实现了 2-4x 的吞吐提升 [2]。

## 吞吐提升的数学分析

### 简化模型

设 batch 大小为 $B$，每个请求需要生成的 token 数为随机变量 $L$。静态 batching 下，每批的 iteration 数为 $\max_{i=1}^{B} L_i$。Continuous Batching 下，每个 iteration 都满员运行（假设始终有等待请求补充），iteration 数为：

$$S_{\text{CB}} = \sum_{i} L_i / B$$

（注意这是一个近似——实际中 batch 可能因请求不足而无法填满，但 Continuous Batching 的上界确实优于静态。）

### 吞吐比

定义吞吐为每秒处理 request 数。设单次 decode 前向耗时固定为 $\tau$，则：

- 静态 batching 吞吐：$\text{Throughput}_{\text{static}} = B / (\mathbb{E}[\max_i L_i] \cdot \tau)$
- Continuous Batching 吞吐：$\text{Throughput}_{\text{CB}} = B / (\bar{L} \cdot \tau)$（理想情况下每次满批）

当请求长度方差较大时，$\mathbb{E}[\max_i L_i] \gg \bar{L}$，Continuous Batching 的吞吐优势显著。

以典型的 mix-length 负载（短请求 $\bar{L}=50$ 和长请求 $\bar{L}=500$ 各占 50%，$B=32$）：

$$\mathbb{E}[\max_{i=1}^{32} L_i] \approx 500 \quad (\text{至少一条长请求几乎确定出现})$$

$$\text{加速比} = \frac{\text{Throughput}_{\text{CB}}}{\text{Throughput}_{\text{static}}} \approx \frac{\mathbb{E}[\max_i L_i]}{\bar{L}} = \frac{500}{275} \approx 1.8\times$$

如果短请求比例增大到 90%，$\bar{L} \approx 95$，加速比可以达到约 $500/95 \approx 5.3\times$。

Orca 在 GPT-3 175B 上的实验报告在相同延迟预算下吞吐提升 **36.9x**（相比 NVIDIA FasterTransformer）[1]。这个极高倍数包含了两项收益的叠加：(1) iteration-level scheduling 消除了 padding 和排队；(2) selective batching 配合分布式执行。vLLM 在 13B 模型上的实验显示**2x-4x** 的吞吐提升（相比 Orca），额外收益来自 PagedAttention 对显存碎片化的消除 [2]。TensorRT-LLM 的 in-flight batching 也报告了**2x-5x** 的吞吐改善 [6]。综合来看，Continuous Batching 在真实负载中的吞吐提升范围通常在 **2x-10x**，取决于请求长度分布、GPU batch 容量和是否有 PagedAttention 等 KV cache 优化配合。

## 生产实践考量

### 调度策略

vLLM 调度器支持多种 **SchedulingPolicy**，包括 FCFS（先到先服务）和优先级调度 [7]。生产环境中通常使用 FCFS 以保证公平性，但在需要对某些请求保证 SLO 的场景下可以使用优先级抢占。

调度器通过 `watermark` 参数预留一部分 KV cache blocks 不分配给 WAITING 请求。当显存超过 watermark 时，新请求会被阻止加入 RUNNING，避免频繁的预抢占 [7]。

### 抢占（Preemption）

当 GPU 显存不足以容纳新请求的 KV cache 时，调度器必须执行抢占——将某些 RUNNING 请求的 KV cache 换出到 CPU 内存（SWAP），或将请求完全中止并重新排队（RECOMPUTE）。vLLM 的抢占策略基于请求的"进展最少的优先被抢占"原则（高 `num_computed_tokens` 的请求保留，低优先级的被换出）[7]。

### Chunked Prefill 的权衡

Chunked prefill 的好处是避免长 prompt 阻塞 decode，但代价是将 prefill 的计算分散到多次 iteration 中，增加了端到端延迟。在延迟不敏感的批处理场景下可以开启；在实时对话场景中则需要权衡 chunk size。

### 与 Prefill/Decode 解耦的区别

本文讨论的 Continuous Batching 与 Prefill/Decode 解耦属于不同层面的优化，不应混淆：

| 维度 | Continuous Batching | Prefill/Decode 解耦 |
|------|---------------------|---------------------|
| 优化维度 | 调度粒度（时间） | 计算分离（空间） |
| 核心动作 | 迭代间动态变化 batch 成员 | Prefill 和 Decode 计算在不同 GPU 上执行 |
| 解决的问题 | Batch 内长短请求的 GPU 浪费 | Prefill 计算密集和 Decode 内存密集的硬件资源错配 |
| 可否叠加 | ✅ | — |

两者是完全正交的优化维度，可以组合使用：Continuous Batching 让每个 worker（无论 prefill worker 还是 decode worker）内部的 batch 调度更高效；P/D 解耦让不同类型的 worker 使用不同的硬件配置。这也是 vLLM V1 的架构设计方向——调度器内部使用 iteration-level 调度，同时通过 KV Transfer 支持跨进程的 P/D 分离 [3][7]。

## 局限与常见误解

**误解一：Continuous Batching 消除了所有延迟。** 实际上，新请求仍然需要等待至少一个正在执行的 iteration 完成才能加入 batch。在高负载下（连续有大量请求涌入），每个 iteration 可能都满员，新请求需要排队等待。Continuous Batching 优化的是"batch 内因长短不均造成的浪费"，不是"无限容量"。

**误解二：Continuous Batching 和 PagedAttention 是同一回事。** 这是两个独立但互补的机制。Continuous Batching 属于**调度策略**，PagedAttention 属于**显存管理策略**。一个没有 PagedAttention 的系统仍然可以实现 iteration-level scheduling（如 Orca），但 KV cache 碎片化会限制它能容纳的最大 batch size。一个没有 Continuous Batching 的系统仍然可以使用 PagedAttention（如早期的 vLLM 实验对照组），但 batch 内的 GPU 浪费仍然存在 [2]。

**误解三：Chunked prefill 是 Continuous Batching 必需的。** 不是。Chunked prefill 是 Continuous Batching 框架下的一种优化技术，用于解决长 prompt 占用过多 iteration 时间的问题。基础的 iteration-level scheduling 不需要 chunked prefill。

**实现局限：** 生产系统中，Continuous Batching 需要与 GPU 的 kernel 实现配合。TensorRT-LLM 的实现需要编译期预定义最大 batch size 以分配 workspace（虽然通过 in-flight batching 可以动态变化，但上限固定），这在大 batch 场景下可能引入额外的显存开销 [6]。

## 小光总结

Continuous Batching 是 LLM 推理调度中为数不多的"改一行调度逻辑、提几倍吞吐"的优化。

**五条核心 takeaway：**

1. **静态 batching 的致命伤是"最长请求决定所有人的延迟"**——GPU 利用率公式 $T_{\text{padding}} / (N \cdot S)$ 清楚展示了浪费的来源。
2. **迭代级调度（iteration-level scheduling）将调度粒度从 request 降到 iteration**，使 batch 在每个前向之后可以重新组合，消除了 padding 浪费和排队阻塞。
3. **Orca 的 selective batching + vLLM 的 PagedAttention** 分别解决了"算子的批处理兼容性"和"KV cache 碎片化"两个工程问题，使得 iteration-level scheduling 在实际系统中可落地。
4. **vLLM 的调度器通过"token budget"而非"batch size"管理并发**，利用 `num_computed_tokens` 统一 prefill 和 decode 的调度，这是工程实践中的关键抽象。
5. **Continuous Batching 和 Prefill/Decode 解耦是两个正交维度**——前者优化调度粒度，后者优化计算分离；在 vLLM V1 等新一代架构中两者协同工作。

如果你是服务端工程师准备部署 LLM 推理，Continuous Batching 是你应该选配的基础特性（目前 vLLM、SGLang、TensorRT-LLM、TGI 都已默认支持）。如果你在为推理系统做性能建模，理解 iteration-level scheduling 的 token budget 管理和抢占策略是决定系统容量的关键。

## 参考资料

1. **Orca (OSDI'22)**：[Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) — 提出 iteration-level scheduling 和 selective batching，报告 GPT-3 175B 上 36.9x 吞吐提升 vs FasterTransformer。
2. **vLLM (SOSP'23)**：[Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023](https://arxiv.org/abs/2309.06180) — 提出 PagedAttention + Continuous Batching 组合，报告 2x-4x 吞吐提升 vs Orca。
3. **vLLM Architecture Overview**：[vLLM 官方架构文档](https://docs.vllm.ai/en/stable/design/arch_overview/) — vLLM V1 的 Engine Core 和调度器架构说明。
4. **vLLM PagedAttention Design**：[vLLM Paged Attention 设计文档](https://docs.vllm.ai/en/stable/design/paged_attention/) — 物理 block 分配、KV cache 虚拟内存的详细工程说明。
5. **vLLM Blog**：[Woosuk Kwon et al., "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention", vLLM Blog, 2023](https://blog.vllm.ai/2023/06/20/vllm.html) — vLLM 的通俗介绍与 Continuous Batching 的 Benchmark 对比。
6. **TensorRT-LLM In-flight Batching**：[NVIDIA, "TensorRT-LLM In-flight Batching", NVIDIA Docs](https://nvidia.github.io/TensorRT-LLM/advanced/in-flight-batching.html) — NVIDIA 的 Continuous Batching 实现，报告 2x-5x 吞吐提升。
7. **vLLM Scheduler 源码**：[vLLM v1/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — vLLM V1 调度器的核心实现，包含 `schedule()` 方法、请求状态管理和抢占逻辑。
8. **FasterTransformer**：[NVIDIA, "FasterTransformer", GitHub](https://github.com/NVIDIA/FasterTransformer) — 静态 batching 的代表性 baseline，被 Orca 和 vLLM 作为对比基线。
