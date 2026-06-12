---
title: "ZeRO 优化器状态分片完全推导：从数据并行的显存瓶颈到三阶段分片的通信-显存权衡"
date: 2026-06-12 08:00:00
mathjax: true
categories:
 - TECH
 - AI
 - AI Infra
tags:
 - ZeRO
 - DeepSpeed
 - FSDP
 - Distributed Training
 - Memory Optimization
 - Data Parallelism
description: "从Ψ=20 bytes/param的显存冗余出发，完整推导ZeRO Stage 1/2/3的分片粒度、通信量公式和与FSDP的对比。每个公式配源码映射。"
topic_id: "TECH-2026-06-12-02"
---

## TL;DR

标准数据并行（DP）在每个 GPU 上复制完整的模型状态——参数、梯度、优化器状态——导致高达 $\Psi \cdot N_d$ 的总显存冗余（$\Psi$ 为单参数所需字节数，$N_d$ 为 GPU 数量）。ZeRO（Zero Redundancy Optimizer）通过分片消除冗余：Stage 1 分片优化器状态（4× 节省），Stage 2 追加梯度分片（8× 节省），Stage 3 追加参数分片（线性 $N_d$ 倍节省），代价是从零通信开销到最高 1.5× 基线通信量。本文将每一阶段的显存公式、通信量、源码路径逐一推导。

## 前置知识

建议读者了解：

- **数据并行**：每个 GPU 持有完整模型副本，前向/反向独立计算，梯度通过 `all-reduce` 同步。
- **混合精度训练**：前向/反向使用 fp16/bf16（2 bytes/param），优化器维护 fp32 主副本（4 bytes/param）。
- **Adam 优化器**：对每个参数 $w$ 维护一阶动量 $m$ 和二阶动量 $v$，均为 fp32 精度。
- **通信原语**：`all-reduce`（通信量 $\approx 2\Phi$）、`all-gather`（$\approx \Phi$）、`reduce-scatter`（$\approx \Phi$），其中 $\Phi$ 为单次传输数据量 [1]。

## 问题定义：$\Psi = 20\ \text{bytes/param}$ 的显存噩梦

在混合精度 + Adam 的数据并行训练中，**每个 GPU** 需要维护以下模型状态：

$$
\begin{aligned}
M_{\text{params}} &= 2\Psi_{\text{num}}\ \text{bytes} \quad &\text{(fp16/bf16 参数副本)} \\[2pt]
M_{\text{grads}} &= 2\Psi_{\text{num}}\ \text{bytes} \quad &\text{(fp16/bf16 梯度副本)} \\[2pt]
M_{\text{opt-master}} &= 4\Psi_{\text{num}}\ \text{bytes} \quad &\text{(fp32 参数主副本)} \\[2pt]
M_{\text{opt-m}} &= 4\Psi_{\text{num}}\ \text{bytes} \quad &\text{(fp32 一阶动量)} \\[2pt]
M_{\text{opt-v}} &= 4\Psi_{\text{num}}\ \text{bytes} \quad &\text{(fp32 二阶动量)} \\[2pt]
M_{\text{grad-buffer}} &= 4\Psi_{\text{num}}\ \text{bytes} \quad &\text{(fp32 梯度累积/通讯缓冲区)}
\end{aligned}
$$

其中 $\Psi_{\text{num}}$ 为模型参数数量。**每个参数的总显存开销为**：

$$\Psi = 2 + 2 + 4 + 4 + 4 + 4 = 20\ \text{bytes/param}$$

> **直觉**：fp16 参数和梯度仅占 4 bytes，而 Adam 的 fp32 状态占 12 bytes，再加上 fp32 梯度缓冲区 4 bytes，优化器状态是参数本身显存的 **10 倍**。这是 ZeRO 的核心优化目标。

**总冗余**：在 $N_d$ 个数据并行 GPU 上，模型状态的总显存占用为 $20 \Psi_{\text{num}} N_d$ bytes，其中绝大部分是冗余。

**问题陈述**：给定 $N_d$ 个 GPU，每 GPU 显存容量为 $M_{\text{GPU}}$，训练 $\Psi_{\text{num}}$ 个参数的模型。不使用 ZeRO 时，约束为 $20\Psi_{\text{num}} \leq M_{\text{GPU}}$。对于 7.5B 参数模型，$20 \times 7.5 \times 10^9 = 150\ \text{GB}$，远超单 GPU 32GB/80GB 容量 [2]。

## Baseline：标准数据并行（无 ZeRO）

标准数据并行的训练流程为：

1. **前向**：每 GPU 独立在本地参数上执行，`torch.no_sync()` 不通信。
2. **反向**：计算 fp16 梯度，通过 `all-reduce` 在所有 $N_d$ 个 GPU 上求和平均。
3. **优化器更新**：每 GPU 独立执行 `adam.step()`，更新 fp32 主参数后再 cast 回 fp16。

通信量（per step）：

$$V_{\text{comm}}^{\text{DP}} = 2\Psi_{\text{num}} \times 2 = 4\Psi_{\text{num}}\ \text{bytes}$$

这里 $2\Psi_{\text{num}}$ 是 fp16 梯度大小，$2\times$ 来自 `all-reduce` 的 send+receive。实际上 all-reduce 的通信相位差约为 $2(N_d-1)/N_d \cdot \Phi$，典型实现如 ring all-reduce 的通信量为 $2\frac{N_d-1}{N_d}\Phi$。

## Stage 1：分片优化器状态（$P_{os}$）

### 核心思想

优化器状态是最大的显存占用者（12 bytes/param），但优化器更新是**逐参数独立**的——每个参数 $w_i$ 的 Adam 更新 $w_i \gets w_i - \eta \cdot m_i/(\sqrt{v_i} + \epsilon)$ 只依赖该参数的局部动量，不需要跨 GPU 通信。

ZeRO-1 将 fp32 优化器状态（master weight、$m$、$v$）均匀分片到 $N_d$ 个 GPU 上：GPU $k$ 只持有并更新参数区间 $[\frac{k}{N_d}\Psi_{\text{num}}, \frac{k+1}{N_d}\Psi_{\text{num}})$ 的优化器状态。

### 训练流程

1. **前向**：与标准 DP 相同，每 GPU 用本地完整 fp16 参数计算。
2. **反向**：与标准 DP 相同，`all-reduce` 梯度。
3. **优化器更新**：每 GPU 只在自己持有的分片上执行 `adam.step()`——更新 fp32 主参数和动量。
4. **参数同步**：不需要！因为 fp16 参数是完整副本，fp32 状态的分片对前向/反向**透明**。

### 显存节省

每 GPU 的优化器状态从 $12\Psi_{\text{num}}$ 降至 $\frac{12\Psi_{\text{num}}}{N_d}$：

| 组件 | 无 ZeRO | ZeRO-1 |
|:---|:---|:---|
| fp16 参数 | $2\Psi$ | $2\Psi$ |
| fp16 梯度 | $2\Psi$ | $2\Psi$ |
| fp32 优化器状态 | $12\Psi$ | $12\Psi/N_d$ |
| fp32 梯度 buffer | $4\Psi$ | $4\Psi/N_d$ |
| **总计** | $20\Psi$ | $4\Psi + 16\Psi/N_d$ |

当 $N_d=64$ 时，$20\Psi \to 4\Psi + 0.25\Psi = 4.25\Psi$，节省约 **4.7×**（对应论文的「4× memory reduction」）。当 $N_d \to \infty$，极限节省为 **5×**（$20\Psi \to 4\Psi$）。

### 通信量

$$V_{\text{comm}}^{Z1} = 2\Psi_{\text{num}} \times 2 = 4\Psi_{\text{num}}\ \text{bytes}$$

与标准 DP **完全相同**——唯一的通信仍是梯度的 `all-reduce`。

### 源码映射

DeepSpeed 中 ZeRO-1 的核心实现在 `deepspeed/runtime/zero/stage_1_and_2.py` 中。关键路径：

- **配置解析**：`DeepSpeedZeroOptimizer` 类在初始化时根据配置项的 stage 值决定分片策略（stage=1）。
- **优化器状态分片**：`_create_fp32_partitions()` 将每个参数组的 fp32 缓冲区按数据并行度分割为 `num_partitions` 份（值为 `data_parallel_size`），每 rank 只保留 `self.partition_id` 对应的切片 [3]。
- **参数更新**：`step()` 方法中，每个 rank 仅在其本地分片上调用 `self.optimizer.step()`。

```python
# 伪代码：ZeRO-1 优化器状态分片（对应 DeepSpeed stage_1_and_2.py）
# 输入: fp16_params shape=[Ψ], fp32_opt_states shape=[3, Ψ]
# partition_size = Ψ // N_d
# local_slice = slice(rank * partition_size, (rank+1) * partition_size)

def zero1_step(fp16_grads, fp16_params, fp32_opt_states, N_d, rank):
    # 1. all-reduce fp16 梯度（与标准 DP 相同）
    allreduce(fp16_grads)  # comm: 2Ψ bytes

    # 2. 仅更新本地分片的优化器状态
    local_grads = fp16_grads[rank * Ψ//N_d : (rank+1) * Ψ//N_d]
    local_states = fp32_opt_states[:, rank * Ψ//N_d : (rank+1) * Ψ//N_d]
    adam_step(local_grads, local_states)  # 计算量 = 1/N_d

    # 3. 将更新后的 fp32 参数 cast 回 fp16（仅本地分片）
    fp16_params[rank * Ψ//N_d : (rank+1) * Ψ//N_d] = (
        local_states[0].half()
    )
```

> **注意**：这里不需要 `all-gather`，因为 fp16 参数在整个副本上是完整的，fp32 分片对外透明。

## Stage 2：追加梯度分片（$P_{os+g}$）

### 核心思想

在 `all-reduce` 完成后，每 GPU 持有的**完整梯度副本是冗余的**——因为优化器只需要更新它所负责的分片。ZeRO-2 用 `reduce-scatter` 替代 `all-reduce`，使每 GPU 在梯度同步后只保留自己分片内的梯度，其余部分立即释放。

### 关键区别：`all-reduce` vs `reduce-scatter`

- **`all-reduce`**：所有 GPU 获得完整梯度副本。通信量 $2\frac{N_d-1}{N_d}\Phi$，最终每 GPU 持有 $\Phi$。
- **`reduce-scatter`**：每 GPU 只获得梯度的 $1/N_d$ 分片。通信量 $\frac{N_d-1}{N_d}\Phi$，最终每 GPU 持有 $\Phi/N_d$。

`reduce-scatter` 的通信量大约是 `all-reduce` 的 **一半**（$1\times$ vs $2\times$ 因子），但两者的总网络流量（考虑所有 rank）相同。

### 训练流程

1. **前向**：与 ZeRO-1 相同，每 GPU 用本地完整 fp16 参数。
2. **反向**：计算梯度后，执行 `reduce-scatter`（而非 `all-reduce`），每 GPU 仅保留其分片的梯度。
3. **优化器更新**：在本地梯度分片 + 本地优化器状态分片上执行 `adam.step()`。
4. **梯度释放**：优化器更新完成后，本地梯度分片可立即释放。

### 显存节省

梯度从完整 $2\Psi_{\text{num}}$ 降至 $\frac{2\Psi_{\text{num}}}{N_d}$：

| 组件 | ZeRO-1 | ZeRO-2 |
|:---|:---|:---|
| fp16 参数 | $2\Psi$ | $2\Psi$ |
| fp16 梯度 | $2\Psi$ | $2\Psi/N_d$ |
| fp32 优化器状态 | $12\Psi/N_d$ | $12\Psi/N_d$ |
| fp32 梯度 buffer | $4\Psi/N_d$ | $4\Psi/N_d$ |
| **总计** | $4\Psi + 16\Psi/N_d$ | $2\Psi + 18\Psi/N_d$ |

当 $N_d=64$ 时，$20\Psi \to 2\Psi + 0.28\Psi = 2.28\Psi$，节省约 **8.8×**（论文称 8×）。当 $N_d \to \infty$，极限节省为 **10×**（$20\Psi \to 2\Psi$）。

### 通信量

$$V_{\text{comm}}^{Z2} = \frac{N_d-1}{N_d} \cdot 2\Psi_{\text{num}} \approx 2\Psi_{\text{num}}\ \text{bytes}$$

与标准 DP 的 `all-reduce` 相比，`reduce-scatter` 通信量减半，**与基线 DP 通信量相同**（论文结论）。

### 源码映射

在 `stage_1_and_2.py` 中，Stage 2 的核心差异在于梯度通信原语的选择：

- `self.reduce_scatter` 标志位控制是否使用 `reduce_scatter` 替代 `all_reduce`。
- `overlap_comm` 参数允许将梯度的 `reduce-scatter` 与反向计算重叠 [3]。
- `contiguous_gradients` 将梯度拷贝到连续缓冲区以减少碎片化：

```python
# stage_1_and_2.py 中的关键流程（简化）
if self.reduce_scatter:
    # 使用 reduce-scatter：每 rank 只接收梯度分片
    self.ipg_buffer = torch.zeros(
        partition_size, dtype=torch.float16, device=self.device
    )
    # reduce-scatter 将全局梯度的求和建议入 ipg_buffer
    dist.reduce_scatter(self.ipg_buffer, grad_list, group=self.dp_group)
else:
    # all-reduce（Stage 1 回退）
    dist.all_reduce(grad, group=self.dp_group)
```

## Stage 3：追加参数分片（$P_{os+g+p}$）

### 核心思想

ZeRO-3 将 fp16/bf16 参数**也分片**。这是最激进的一步——前向和反向计算需要完整参数时，必须通过 `all-gather` 临时收集，用完立即释放。

### 训练流程

1. **前向（逐层）**：
   - 对于每个 FSDP unit（通常是一个 Transformer 层），`all-gather` 该层的完整参数。
   - 执行前向计算。
   - 释放非本地分片的参数（只保留本地分片）。

2. **反向（逐层，逆序）**：
   - 再次 `all-gather` 该层完整参数。
   - 执行反向计算，计算参数梯度。
   - `reduce-scatter` 梯度到各 rank 分片。
   - 释放非本地分片的参数。

3. **优化器更新**：在本地分片上执行，与 ZeRO-2 相同。

### 显存节省

参数也从 $2\Psi_{\text{num}}$ 降至 $\frac{2\Psi_{\text{num}}}{N_d}$：

$$M_{\text{peak}}^{Z3} = \frac{2\Psi_{\text{num}}}{N_d} + \frac{2\Psi_{\text{num}}}{N_d} + \frac{12\Psi_{\text{num}}}{N_d} + \frac{4\Psi_{\text{num}}}{N_d} + M_{\text{ag}}$$

其中 $M_{\text{ag}}$ 是一次 `all-gather` 的参数缓冲区峰值（约为一个 FSDP unit 的参数量），受 `stage3_max_live_parameters` 等配置约束。

| 组件 | ZeRO-2 | ZeRO-3 |
|:---|:---|:---|
| fp16 参数 | $2\Psi$ | $2\Psi/N_d$ |
| fp16 梯度 | $2\Psi/N_d$ | $2\Psi/N_d$ |
| fp32 优化器状态 | $12\Psi/N_d$ | $12\Psi/N_d$ |
| fp32 梯度 buffer | $4\Psi/N_d$ | $4\Psi/N_d$ |
| **总计（稳态）** | $2\Psi + 18\Psi/N_d$ | $20\Psi/N_d$ |

当 $N_d=64$ 时，$20\Psi \to 0.3125\Psi$，节省 **64×**。显存随 $N_d$ **线性缩放**。

### 通信量：为什么是 1.5×

ZeRO-3 引入两次额外的参数通信：

- **前向**：每个 FSDP unit 执行一次参数 `all-gather`：通信量 $\Psi_{\text{unit}}$（全量，忽略 $(N_d-1)/N_d$ 因子取近似）。
- **反向**：再次 `all-gather` 参数 + `reduce-scatter` 梯度。

总通信量为：

$$V_{\text{comm}}^{Z3} = 2\Psi_{\text{num}}\ (\text{参数 all-gather}) + 2\Psi_{\text{num}}\ (\text{梯度 reduce-scatter}) = 4\Psi_{\text{num}}\ \text{bytes}$$

相比基线 $4\Psi_{\text{num}}$（梯度 all-reduce），绝对量相等。但论文使用不同基线（梯度 all-reduce 在 ring 实现下约 $2\Psi$），因此 ZeRO-3 相对基线增加约 **1.5×**。

精确推导（使用 ring 通信模型）：

| 阶段 | 通信原语 | 通信量（ring） |
|:---|:---|:---|
| 基线 DP（梯度 all-reduce） | all-reduce | $2\frac{N_d-1}{N_d} \cdot 2\Psi \approx 4\Psi$ |
| ZeRO-1/2（梯度 reduce-scatter） | reduce-scatter | $\frac{N_d-1}{N_d} \cdot 2\Psi \approx 2\Psi$ |
| ZeRO-3（参数 all-gather ×2 + 梯度 reduce-scatter） | 2× all-gather + reduce-scatter | $2 \cdot \frac{N_d-1}{N_d}\Psi + \frac{N_d-1}{N_d} \cdot 2\Psi \approx 4\Psi$ |

> **直觉**：ZeRO-1/2 用 `reduce-scatter` 换 `all-reduce`，通信更少；ZeRO-3 在此基础上加回两倍参数 `all-gather`，通信回归基线水平。

### 源码映射：`all-gather` 与参数收集

ZeRO-3 的核心实现在 DeepSpeed 的 `deepspeed/runtime/zero/partition_parameters.py` 及相关文件中：

- **参数收集**：`GatheredParameters` 上下文管理器负责在 `__enter__` 时调用 `all-gather` 收集完整参数，`__exit__` 时释放非本地分片 [4]。
- **逐层调度**：`_pre_forward` / `_post_forward` 钩子控制前向的参数 gather/release。
- **配置参数**：
  - `stage3_max_live_parameters`：一次 `all-gather` 的最大参数量（控制峰值显存）。
  - `stage3_prefetch_bucket_size`：预取粒度，实现通信与计算重叠。
  - `stage3_param_persistence_threshold`：小于此阈值的参数保持常驻不释放。

```python
# 伪代码：ZeRO-3 逐层前向
def zero3_forward(layer_params, layer_idx, N_d, rank):
    # 1. all-gather 本层完整参数
    full_params = all_gather(layer_params[rank * P//N_d : (rank+1) * P//N_d])
    # comm: P bytes（P = 层参数量）

    # 2. 执行前向
    output = layer_forward(full_params, input_activation)

    # 3. 释放非本地参数分片
    del full_params  # GPU 显存立即回收

    return output

def zero3_backward(layer_params, layer_grads, N_d, rank):
    # 4. 再次 all-gather 参数用于反向
    full_params = all_gather(layer_params[rank * P//N_d : (rank+1) * P//N_d])

    # 5. 反向计算 + 梯度 reduce-scatter
    grad_output = layer_backward(full_params, grad_input)
    reduce_scatter(layer_grads, full_grads)  # comm: 2P bytes

    del full_params
    return grad_output
```

## 通信量-显存 Trade-off 公式总结

令 $\Psi_{\text{num}}$ 为模型参数总量，$N_d$ 为数据并行的 GPU 数量。定义：

- $\alpha(N_d) = \frac{N_d-1}{N_d}$，通信因子（ring 实现下的近似）。
- $\Psi = 20$ bytes/param，基线每参数显存。

| 阶段 | 每 GPU 显存（稳态） | 通信量（per step） | 相对基线通信 |
|:---|:---|:---|:---|
| **基线 DP** | $20\Psi_{\text{num}}$ | $2\alpha(N_d) \cdot 2\Psi_{\text{num}}$ | $1\times$ |
| **ZeRO-1** | $4\Psi_{\text{num}} + 16\Psi_{\text{num}}/N_d$ | $\alpha(N_d) \cdot 2\Psi_{\text{num}}$ | $\approx 1\times$ |
| **ZeRO-2** | $2\Psi_{\text{num}} + 18\Psi_{\text{num}}/N_d$ | $\alpha(N_d) \cdot 2\Psi_{\text{num}}$ | $\approx 1\times$ |
| **ZeRO-3** | $20\Psi_{\text{num}}/N_d$ | $2\alpha(N_d)\Psi_{\text{num}} + \alpha(N_d) \cdot 2\Psi_{\text{num}}$ | $\approx 1.5\times$ |

> **解析**：ZeRO-1/2 在**不增加通信**的前提下实现 4-8× 显存节省。ZeRO-3 实现线性缩放，代价是 50% 额外通信。

以下是 $N_d=64$ 时的数值示例（以 7.5B 参数模型、$\Psi=20$ 为例）：

| 阶段 | 每 GPU 显存 | 节省倍数 | 通信量 | 相对基线 |
|:---|:---|:---|:---|:---|
| 基线 DP | 150.0 GB | $1\times$ | $3.94\Psi_{\text{num}}$ | $1\times$ |
| ZeRO-1 | 31.9 GB | $4.7\times$ | $1.97\Psi_{\text{num}}$ | $0.5\times$ |
| ZeRO-2 | 17.1 GB | $8.8\times$ | $1.97\Psi_{\text{num}}$ | $0.5\times$ |
| ZeRO-3 | 2.34 GB | $64\times$ | $3.94\Psi_{\text{num}}$ | $1\times$ |

## FSDP vs ZeRO：PyTorch 原生实现对比

PyTorch 的 FSDP（Fully Sharded Data Parallel）从 PyTorch 1.11 起原生支持等价于 ZeRO-3 的全分片策略 [5]：

### 分片策略映射

| ZeRO 阶段 | FSDP 配置 | 分片内容 |
|:---|:---|:---|
| ZeRO-1 | `ShardingStrategy.SHARD_GRAD_OP` | 仅优化器状态 |
| ZeRO-2 | `ShardingStrategy.SHARD_GRAD_OP` | 优化器状态 + 梯度 |
| ZeRO-3 | `ShardingStrategy.FULL_SHARD` | 参数 + 梯度 + 优化器状态 |

### 关键差异

| 维度 | DeepSpeed ZeRO-3 | PyTorch FSDP |
|:---|:---|:---|
| **精度处理** | 强制 fp32 主参数（`_create_fp32_partitions` 内部 upcast） | 遵循 `MixedPrecision` 配置，允许低精度优化器 [6] |
| **参数预取** | `stage3_prefetch_bucket_size` 控制 | `forward_prefetch` / `backward_prefetch` |
| **offload** | CPU/NVMe offload（ZeRO-Infinity） | CPU offload（`CPUOffload`） |
| **checkpoint** | DeepSpeed 专有格式 | `FULL_STATE_DICT` / `SHARDED_STATE_DICT` |

**精度差异的工程影响**：DeepSpeed 默认将优化器参数 upcast 到 fp32，这可能导致学习率敏感度不同。Hugging Face Accelerate 的对比实验显示：使用相同学习率时，DeepSpeed Zero-3 的训练 loss 收敛正常而 FSDP（未做 upcast）不收敛，根因正是精度处理差异 [6]。

### wrapping 策略对比

两种框架都支持逐层 wrapping 以减少 `all-gather` 的峰值显存：

```python
# FSDP: PyTorch 原生 wrapping
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

fsdp_model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

```json
# DeepSpeed: JSON 配置
{
    "zero_optimization": {
        "stage": 3,
        "stage3_max_live_parameters": 1e9,
        "stage3_prefetch_bucket_size": 5e7
    }
}
```

## ZeRO-Offload 与 ZeRO-Infinity

### ZeRO-Offload

当 GPU 显存不足时，ZeRO-Offload 将优化器状态和梯度 offload 到 CPU 内存，利用 CPU 执行 Adam 更新 [7]：

- 在 ZeRO-2 基础上启用：将 `offload_optimizer` 的 device 设为 cpu。
- 单 V100-32GB 可训练 13B 参数模型。
- 代价：CPU-GPU 数据传输成为新瓶颈，需要 `DeepSpeedCPUAdam` 等优化实现。

### ZeRO-Infinity

ZeRO-3 + Infinity offload engine 支持将模型状态 offload 到 CPU 和 **NVMe SSD**，offload 带宽可达 50 GB/s（PCIe Gen4）[8]：

$$M_{\text{GPU}}^{\text{Infinity}} = \frac{20\Psi_{\text{num}}}{N_d} - M_{\text{offloaded}}$$

通过 `stage3_max_live_parameters` 等参数精确控制 GPU 上常驻的参数窗口，其余自动分页到 CPU/NVMe。

## 实践建议

1. **模型 < 1B 参数 + 单节点训练**：标准 DDP 即可，ZeRO 的通信开销不划算。
2. **模型 1B-13B 参数**：ZeRO-2（或 FSDP `SHARD_GRAD_OP`）是最优选择——显存节省显著（8×），通信量与基线持平。
3. **模型 > 13B 参数或跨节点**：必须 ZeRO-3 / FSDP `FULL_SHARD`。注意网络带宽需匹配通信量——推荐 InfiniBand 或 NVLink 互联。
4. **精度对齐**：使用 `bf16` 训练 + FSDP 时注意配置 `MixedPrecision(param_dtype=torch.float32)` 以确保优化器精度与 DeepSpeed 对齐。
5. **与激活检查点结合**：ZeRO + activation checkpointing 可将显存进一步压缩，但会增加 30% 左右的计算量（重计算）。
6. **`stage3_param_persistence_threshold` 调优**：将小参数（如 LayerNorm 的 bias/weight）设置为常驻 GPU 可减少碎片化 `all-gather` 调用。

## 局限与常见误解

1. **ZeRO 不能解决激活显存问题**：ZeRO 只分片模型状态（参数、梯度、优化器），激活显存仍需 activation checkpointing（重计算）或序列并行解决。对于长序列训练，激活显存可能超过模型状态。
2. **通信量与 GPU 数不独立**：虽然论文声称 ZeRO-1/2 通信量 "same as DP"，但在 ring 实现下，延迟随 $N_d$ 线性增长，带宽利用率下降。
3. **ZeRO-3 的 `all-gather` 开销不能被忽略**：当网络带宽不足（如以太网）时，逐层 `all-gather` 的延迟会严重拖慢训练。实际中通常需要通信计算重叠（`overlap_comm`）来缓解。
4. **FSDP 的 `SHARD_GRAD_OP` 不等价于标准 Adam**：FSDP 允许低精度优化器步骤，但低精度下 Adam 的数值稳定性未经充分验证。
5. **ZeRO 不是万能药**：对于 MoE 等稀疏模型，专家并行（EP）可能比 ZeRO-3 更高效；对于超大规模（>100B），ZeRO 通常需要与张量并行（TP）和流水线并行（PP）组合使用。

## 小光总结：分布式训练的显存-通信权衡艺术

ZeRO 的核心贡献不是某个复杂的数学技巧，而是一个非常**简洁的洞察**：数据并行中的模型状态冗余是人为制造的——我们完全可以分片。三个 Stage 的设计体现了优雅的渐进性：

1. **ZeRO-1 的零代价**：优化器状态分片对前向/反向完全透明，没有额外通信。这是 ZeRO 中最「免费的午餐」。
2. **ZeRO-2 的「负代价」**：用 `reduce-scatter` 换 `all-reduce`，通信量不仅没增加，反而在 ring 实现下更低。但需要改变梯度同步的语义（不保留完整梯度），这对需要完整梯度的场景（如梯度裁剪）有影响。
3. **ZeRO-3 的 trade-off**：参数分片打破了计算与存储的耦合——每次前向/反向都要「借」参数再「还」，额外通信是显存线性缩放的代价。但通过逐层 wrapping 和 prefetch，这个代价可以很好地摊销。

**一个实用的判断法则**：如果你的模型参数量的 20×（以 GB 计）大于单 GPU 显存，就应该开始考虑 ZeRO；如果 2× 就超过显存，ZeRO-3 是唯一选择。

## 参考资料

1. **arXiv 论文**：[Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC'20](https://arxiv.org/abs/1910.02054)
2. **微软研究博客**：[DeepSpeed Team, "ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters", Microsoft Research Blog, 2020](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
3. **DeepSpeed 源码**：[Microsoft, DeepSpeed stage_1_and_2.py](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py) — ZeRO-1/2 的核心实现，包含优化器状态分片、reduce-scatter、梯度累积等逻辑。
4. **DeepSpeed 文档**：[DeepSpeed Team, "ZeRO-3 Init and GatheredParameters", DeepSpeed Documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html) — ZeRO-3 的 API 参考，包括 `GatheredParameters` 上下文管理器和 `zero.Init`。
5. **PyTorch 官方博客**：[PyTorch Team, "Introducing PyTorch Fully Sharded Data Parallel (FSDP) API", PyTorch Blog, 2022](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) — FSDP 的初版介绍，包括 wrapping 策略和 benchmark 数据。
6. **Hugging Face 博客**：[Lim et al., "From DeepSpeed to FSDP and Back Again with Hugging Face Accelerate", 2024](https://huggingface.co/blog/deepspeed-to-fsdp-and-back) — FSDP 与 DeepSpeed 精度差异的详细分析。
7. **DeepSpeed ZeRO-Offload 教程**：[DeepSpeed Team, "ZeRO-Offload Tutorial", DeepSpeed Documentation](https://www.deepspeed.ai/tutorials/zero-offload/) — 单 GPU 训练 10B+ 模型的配置与性能。
8. **ZeRO-Infinity 论文**：[Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning", SC'21](https://arxiv.org/abs/2104.07857) — ZeRO-3 的 CPU/NVMe offload 扩展。
9. **DeepSpeed 官方教程**：[DeepSpeed Team, "ZeRO Tutorial", DeepSpeed Documentation](https://www.deepspeed.ai/tutorials/zero/) — 从 Stage 1 到 Stage 3 的完整配置示例。

---

*下一篇预告：从 FlashAttention 到 FlashAttention-3——在线 softmax 的 tiling 推导与 Hopper 架构上的异步优化。*
