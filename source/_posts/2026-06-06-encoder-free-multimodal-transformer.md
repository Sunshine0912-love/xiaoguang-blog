---
title: "Encoder-free 多模态 Transformer 技术解析：当图像和音频直接流入 LLM Backbone"
date: 2026-06-06 13:00:00
categories:
 - AI
 - Multimodal
tags:
 - Transformer
 - Multimodal
 - Encoder-free
 - Gemma
 - Architecture
description: "以 Gemma 4 12B 为核心案例，深入解析 encoder-free 多模态架构：如何去掉独立的视觉/音频编码器，让图像 patch 和音频波形直接投影到 LLM 的 embedding 空间，从而减少参数、降低延迟、简化训练流程。"
mathjax: true
---

## TL;DR

- 传统多模态模型 = 独立视觉编码器（ViT）+ 音频编码器（Conformer）+ LLM backbone，三个组件各自训练、串行推理，代价是额外的参数量（500M-800M 编码器）和推理延迟。
- Encoder-free 架构把编码器全部砍掉，**图像 patch 用一个矩阵乘法投影，音频波形用线性层投影**，直接送入 LLM backbone——所有模态共享同一套 Transformer 权重。
- Gemma 4 12B Unified 是目前最大的 encoder-free 开源模型（12B），视觉嵌入层仅 35M 参数（vs Gemma 4 31B 的 550M 视觉编码器），AIME 2026 推理 77.5%，MMMU Pro 视觉理解 69.1%。
- 代价：小 backbone 上纯 LLM 做视觉理解不如专用编码器高效，但在中型以上规模，性能差距在快速缩小。

## 前置知识

- Transformer decoder 结构（self-attention、FFN、layer norm）
- ViT（Vision Transformer）如何将图像转成 patch token
- 多模态 LLM 的标准架构：Vision Encoder → Projection → LLM Backbone
- Embedding 的基本概念：将离散 token 或连续信号映射到固定维度向量

## 1. 问题定义：传统多模态架构的瓶颈

### 1.1 标准架构回顾

几乎所有主流多模态 LLM（LLaVA、Gemini、GPT-4V、Qwen-VL、Gemma 3/4 E 系列）都采用相似的流水线 [1][2]：

```text
Image → Vision Encoder (ViT, ~300-550M params)
         ↓
    Visual Tokens (e.g., 576 tokens for 336×336)
         ↓
    Cross-Modal Projector (MLP / Q-Former)
         ↓
    LLM Backbone + Text Tokens → Output
```

对应到具体的参数形状（以 Gemma 4 31B 为例 [3]）：

- Vision Encoder: 550M 参数，~27 层 ViT
- Image: 输入 $H \times W \times 3$，切分为 $P \times P$ patches（如 $16 \times 16$）
- 每个 patch 展开为向量 $\mathbf{x}_p \in \mathbb{R}^{P^2 \cdot 3}$，经 ViT 编码为 $\mathbf{z}_p \in \mathbb{R}^{d_{\text{enc}}}$
- Projector $W_{\text{proj}} \in \mathbb{R}^{d_{\text{enc}} \times d_{\text{LLM}}}$ 投影到 LLM 隐藏维度
- 最终：$\mathbf{v}_p = W_{\text{proj}} \cdot \mathrm{ViT}(\mathbf{x}_p)$ 作为 visual token 进入 LLM

### 1.2 瓶颈

这套架构有四个核心问题：

1. **编码器参数量大**：Vision Encoder（300-550M）+ Audio Encoder（~300M）仅用于特征提取，不参与文本生成，占用了大量显存和推理时间。
2. **串行推理延迟**：图像必须先经过 ViT 编码（可能需要数百毫秒），然后才能进入 LLM——两阶段串行。
3. **训练分离**：编码器通常是冻结的预训练 ViT，LLM backbone 是独立的——联合微调需要分阶段进行，增加了工程复杂度。
4. **模态对齐损耗**：编码器的输出空间和 LLM 的 embedding 空间天然不匹配，跨模态 projector（如 Q-Former、MLP）虽然做了桥接，但仍存在信息损失。

## 2. Encoder-Free 的核心思想

**Encoder-free 架构的基本主张是：如果 LLM backbone 足够强，为什么要让一个外部编码器替你"理解"图像和音频？**

Gemma 4 12B Unified 的设计回答了这个问题 [3][4]：

```text
传统（Gemma 4 31B）：
Image → ViT (550M, 27 layers) → MLP → LLM
Audio → Audio Encoder (300M, 12 conformer layers) → MLP → LLM

Encoder-free（Gemma 4 12B）：
Image → 48×48 patches → Single MatMul (35M) → Positional Bias → LLM
Audio → 40ms frames → Linear Projection → LLM
```

### 2.1 视觉投影：从 27 层 ViT 到 1 个矩阵乘法

Gemma 4 12B 的视觉投影极简 [4]：

1. **Patch 提取**：将输入图像切分为 $48 \times 48$ 像素的 patch
2. **单次矩阵乘法**：$\mathbf{v} = W_{\text{img}} \cdot \mathbf{x} + \mathbf{b}$，其中 $W_{\text{img}} \in \mathbb{R}^{d_{\text{LLM}} \times (48^2 \cdot 3)}$
3. **空间位置编码**：使用因子化坐标查找（factorized coordinate lookup），即 X 和 Y 方向各有独立的可学习位置矩阵，避免 $H \times W$ 的组合爆炸
4. **归一化**：投影后经 Layer Norm，直接与 text token 拼接送入 LLM

参数开销：**35M**，对比传统 ViT 编码器的 550M——减少了约 94%。

### 2.2 音频投影：从 12 层 Conformer 到线性层

音频处理更加激进 [4]：

1. **分帧**：16kHz 原始音频信号 → 40ms 帧（每帧 640 个浮点数）
2. **线性投影**：$\mathbf{a} = W_{\text{aud}} \cdot \mathbf{s} + \mathbf{b}$，其中 $W_{\text{aud}} \in \mathbb{R}^{d_{\text{LLM}} \times 640}$
3. 直接送入 LLM backbone，token 顺序即为时间顺序

### 2.3 统一 Token 序列

最终输入 LLM 的是一维 token 序列：

```text
[text_1, text_2, ..., img_1, img_2, ..., img_N, text_k, ..., aud_1, aud_2, ..., aud_M]
```

所有 token 维度相同（$d_{\text{LLM}}$），共享同一套 Transformer 的自注意力权重。

## 3. 为什么 Encoder-Free 能工作？

### 3.1 Transformer 本身就是通用处理器

Transformer 的 self-attention 机制不关心 token 的来源——无论是文本 token、图像 patch 还是音频帧，它在 embedding 空间中做的是相同的矩阵运算。Encoder-free 架构没有发明任何新机制，而是充分利用了 Transformer 已有的通用性。

一个文本 token 经过 self-attention 的变换：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = W_Q \mathbf{x}_i$, $K = W_K \mathbf{x}_j$, $V = W_V \mathbf{x}_j$。

关键洞察：**$\mathbf{x}_i$ 是文本 embedding、图像 patch embedding 还是音频 frame embedding，对 attention 计算公式来说是等价的**。只要投影矩阵 $W_{\text{img}}$ 和 $W_{\text{aud}}$ 能把原始信号映射到 LLM 能处理的 embedding 空间，剩下的理解工作交给 attention 层完成。

### 3.2 Scale 是关键

为什么 encoder-free 在 12B 上可行，但在 2B 上不可行？

直觉：理解图像涉及的视觉模式（边缘、纹理、物体、场景）需要模型的注意力头学习从低维像素信号中提取高级语义。小模型的注意力容量不足以在内部完成编码器的工作——所以需要专门的 ViT 来做这个转化。

12B 模型有 48 层 × 多注意力头 = 足够多的「内部编码器容量」，可以同时学会视觉理解、音频理解和文本生成。

### 3.3 联合微调优势

在 encoder-free 架构中，**视觉投影矩阵、音频投影矩阵和 LLM transformer 层可以端到端微调** [4]。不需要：

- 先冻结编码器训练投影器（Stage 1）
- 再解冻编码器联合微调（Stage 2）
- 再解冻 LLM 全参数训练（Stage 3）

这对下游微调（如 LoRA）非常友好——一个 `train()` 调用即可覆盖所有模态。

## 4. 与 Encoder-based 架构的量化对比

| 维度 | Encoder-based (Gemma 4 31B) | Encoder-free (Gemma 4 12B) |
|------|---------------------------|--------------------------|
| 视觉编码器参数 | 550M（27 层 ViT）| 35M（1 个矩阵乘法）|
| 音频编码器参数 | 300M（12 层 Conformer）| ~0.6M（线性投影）|
| 视觉推理延迟 | 编码器前向 + LLM 前向 | 仅 LLM 前向 |
| 微调流程 | 多阶段（冻结→解冻→联合）| 单阶段端到端 |
| 图像 token 预算 | 可变分辨率（编码器决定）| 可变分辨率（patch 数决定）|
| MMMU Pro 视觉基准 | 76.9% | 69.1% |
| 模型总参数 | 30.7B | 11.95B |

核心结论：**12B encoder-free 用 ~1/3 的参数量达到了 31B encoder-based 约 90% 的视觉理解能力**。差距主要在于极端视觉推理任务（MMMU Pro 差 7.8 个百分点），但在大多数日常视觉任务上已进入可用区间。

## 5. 历史脉络：Encoder-Free 的演进

Encoder-free 不是 Gemma 4 12B 首创，它是一个渐进成熟的技术方向：

- **2023-10: Fuyu（Adept）**：首个大规模 encoder-free 多模态模型，8B 参数，直接将图像 patch 线性投影送入 decoder-only Transformer。证明了架构可行性，但视觉性能显著落后于 encoder-based 方案。
- **2024-05: Chameleon（Meta）**：7B/34B，从 tokenizer 层面统一文本和图像，使用统一的 VQ-VAE codebook。不是严格意义上的 encoder-free（仍有 VQ-VAE），但在统一 token 空间上做了关键探索。
- **2026-06: Gemma 4 12B Unified（Google DeepMind）**：首个**同时去除视觉和音频编码器**的中型开源模型，在性能上首次接近同系列 encoder-based 模型。

## 6. 局限与常见误解

### 6.1 不是「不需要视觉处理」

Encoder-free 不是不处理视觉信号——而是把视觉处理的工作**从专用编码器移到 LLM backbone 内部**。这要求 backbone 有足够的容量来学会从原始像素中提取语义。

### 6.2 性能天花板仍然存在

Gemma 4 12B 的 MMMU Pro 69.1% vs 31B 的 76.9% 说明：在当前规模下，专用编码器仍有优势。这个差距是否会随着 backbone 扩大而消失，是一个开放的研究问题。

### 6.3 高分辨率图像的挑战

去掉 ViT 意味着失去了 ViT 的全局建模能力（通过 CLS token 和位置编码对高分辨率图像的全局理解）。更高的分辨率需要更多的 patch token，这会显著增加 self-attention 的复杂度（$O(N^2)$）。

## 7. 工程与研究意义

### 7.1 端侧部署

Encoder-free 减少了部署时的组件数量：不需要加载独立的视觉编码器权重，不需要管理编码器和 LLM 之间的 IPC。对手机、笔记本等端侧设备来说，这是实质性的部署简化。

### 7.2 推理延迟

传统多模态推理的延迟 = encoder 前向（50-200ms）+ projector 前向 + LLM 自回归。Encoder-free 把这部分延迟压缩到 LLM 的 prefill 阶段内，对实时交互场景（如语音助手、实时视频理解）有直接收益。

### 7.3 模型训练

训练 pipeline 从「多阶段对齐 + 联合微调」简化为「单阶段端到端训练」，降低工程复杂度和 GPU 利用率损失。

## 8. 未来的方向

- **Dynamic token budget**：根据图像复杂度动态调整 patch 数量，简单的图像用少 token、复杂的用多 token
- **Learned patch sizes**：当前 patch 大小是固定的（48×48），未来可能让模型自己学习最优分辨率
- **Encoder-free × MoE**：将 encoder-free 与 Mixture of Experts 结合——用少量的 active parameters 获得更大的等效 backbone 容量
- **多模态预训练目标**：为 encoder-free 架构设计更合适的预训练目标（如跨模态对比学习），减少对编码器的依赖

## 总结

- Encoder-free 多模态架构的核心思想是「让 LLM backbone 直接理解原始信号」，用极简的投影层（1 个 matmul + 位置编码）替代独立的视觉/音频编码器
- Gemma 4 12B Unified 是目前最大的 encoder-free 开源实现，视觉嵌入仅 35M 参数（vs 传统 550M ViT），音频端仅为线性投影
- 性能上，12B encoder-free 达到 31B encoder-based 约 90% 的视觉理解能力，差距集中在极端视觉推理任务
- 关键优势：部署简化、推理延迟降低、端到端训练友好
- 核心限制：小 backbone 上性能不足，高分辨率图像面临 token 数量膨胀
- 这是一个「用通用性换效率」的架构方向——如果 Transformer 确实是足够通用的计算原语，那专用编码器可能只是过渡形态

## 参考资料

[1] [Haotian Liu et al., "Improved Baselines with Visual Instruction Tuning", LLaVA, CVPR 2024 Highlight](https://arxiv.org/abs/2310.03744)

[2] [OpenAI, "GPT-4V(ision) System Card", OpenAI, 2023](https://openai.com/index/gpt-4v-system-card/)

[3] [Google, "google/gemma-4-12B-it Model Card", Hugging Face, 2026](https://huggingface.co/google/gemma-4-12B-it)

[4] [Google DeepMind, "Gemma 4 12B: The Developer Guide", Google Developers Blog, 2026-06-03](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)

[5] [Olivier Lacombe & Gus Martins, "Introducing Gemma 4 12B", Google Blog, 2026-06-03](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/)

[6] [Adept AI, "Fuyu-8B: A Multimodal Architecture for AI Agents", Adept Blog, 2023](https://www.adept.ai/blog/fuyu-8b)

[7] [Meta, "Chameleon: Mixed-Modal Early-Fusion Foundation Models", arXiv:2405.09818, 2024](https://arxiv.org/abs/2405.09818)

## 后续预告

下一篇（TECH-20260606-02）将深入 AI Agent 长期记忆机制——从 RAG 到 Memory Synthesis，结合 OpenAI Memory、MemGPT 等案例，讲解 episodic/semantic memory、遗忘机制和记忆评测的技术全貌。
