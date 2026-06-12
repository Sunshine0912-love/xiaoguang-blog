---
title: "Diffusion LLM 架构解析：当文本生成告别自回归解码"
date: 2026-06-11 08:00:00
categories:
  - AI
  - LLM
tags:
  - Diffusion
  - LLM
  - Text Generation
  - Architecture
  - Inference
description: "NVIDIA DiffusionGemma 发布标志着扩散文本模型进入工程化阶段。本文从自回归瓶颈出发，系统对比扩散生成与自回归解码的架构差异、关键模型和产业影响。"
topic_id: "TOPIC-20260611-01"
---

> 阅读时间：约 12 分钟
> 主题类型：前沿调研 / 架构分析
> 关键词：离散扩散、Diffusion LLM、Block Diffusion、并行解码、推理加速

## TL;DR

2026 年 6 月 10 日，Google DeepMind 发布 DiffusionGemma，一个基于离散扩散的 26B 开源文本生成模型。NVIDIA 同日宣布对其全平台优化，单张 H100 上可达 1000+ tokens/s [1]。这标志着扩散语言模型从一个学术概念进入了工程化部署阶段。

本文从架构层面系统梳理扩散文本生成：它解决了自回归解码的什么瓶颈、核心机制是什么、主要模型（DiffusionGemma、LLaDA、Dream、MDLM 等）的架构差异在哪、以及这项技术的真正产业意义。如果你关心 LLM 推理加速和下一代模型架构，这篇文章应该能帮你建立一个完整的认知框架。

**核心结论**：扩散 LLM 不是要取代自回归，而是在自回归模型的基础上引入一种新的解码范式——block 级并行去噪，换取单用户场景下 4 倍以上的吞吐提升。它的工程代价（质量略降、KV cache 复杂化）正在被快速缩小。

## 问题：自回归解码的三重天花板

今天的 LLM 几乎全是自回归（autoregressive, AR）模型：生成第 N 个 token 时必须先看到前 N-1 个。这个范式有三重工程瓶颈。

**一、内存墙（memory-bound）**。每次 forward 只产出一个 token，GPU 的大量计算单元在等数据搬运。以单用户推理为例，batch size=1 时，整个 Transformer 的计算密度极低，大部分时间消耗在从显存加载权重和 KV cache 上。这在高并发服务中可以通过 batching 缓解，但**单用户延迟**是硬天花板。

**二、顺序依赖**。每个 token 依赖前面所有 token，无法并行。投机解码（speculative decoding）试图用草稿模型并行猜测几个 token，但最终还是要串行验证。这个"一次一个"的约束是根本性的。

**三、方向性偏见**。自回归只能从左到右生成，这在某些场景（如填空、全局规划、逆向推理）中天然劣势。LLaDA 论文中提出的"逆转诅咒"（reversal curse）实验就是一个典型案例：GPT-4o 在反向诗歌补全任务中被扩散模型远超 [2]。

这三重瓶颈在**高吞吐服务**中被 batching 掩盖得很好，但在**对延迟敏感的交互场景**——聊天助手、Copilot、Agent 多轮推理——单用户 token 生成速度直接决定了用户体验。

## 扩散模型如何用于文本：核心直觉

扩散模型的经典故事在图像生成领域：加噪声 → 去噪 → 得到清晰图像。但文本是离散的（token ID），不能直接加高斯噪声。怎么办？

**离散扩散（discrete diffusion）** 的基本思路：用一个转移矩阵定义 token 之间的"腐蚀"过程。

最早系统化这项工作的是 D3PM（Discrete Denoising Diffusion Probabilistic Models）[3]，由 Google Research 在 2021 年提出。D3PM 的核心发现是：**对于文本，"吸收态扩散"（absorbing state diffusion，即逐渐把 token 替换为 `[MASK]`）效果最好**。这建立了一个优雅的对应关系：

- **前向过程**：随机把 token 替换为 `[MASK]`（类似 BERT 训练时的 masking）
- **反向过程**：训练模型预测被 mask 的 token 应该是什么
- **生成**：从全 `[MASK]` 序列开始，逐步去掩码，得到完整文本

你可能会问：这不是 BERT 嘛？区别在于，BERT 的 MLM 是一个**训练目标**，而扩散模型把它扩展为一个**完整的生成框架**——有前向/反向过程、有变分下界、有可控的采样步数。训练目标和生成逻辑是统一的。

到了 2024 年，MDLM（Masked Diffusion Language Model）[4] 进一步简化了这个框架，用 Rao-Blackwellization 把训练目标简化为**不同噪声水平下的加权 MLM loss**。结果很直观：MDLM 在 LM1B 和 OpenWebText 上的困惑度达到了扩散模型的新 SOTA，与自回归模型的差距缩小到 15-25% [4]。

**一句话直觉**：扩散文本生成就是把生成问题变成了一个逐步"去噪"过程——从一个全 masked（或全随机）token 序列出发，每一步用 Transformer 同时预测所有位置的 token，逐步精炼到收敛。这里的关键词是**同时**。

## 关键模型对比：从学术到工程的四条路线

### LLaDA（2025.2）：大规模验证者

LLaDA（Large Language Diffusion with mAsking）[2] 是第一个把 masked diffusion 真正 scale 到 8B 参数的工作，2025 年发布，已被 NeurIPS 2025 接收。

**架构**：标准 decoder-only Transformer，但训练时不是因果注意力，而是**双向注意力**——每个 token 都能看到全部上下文。前向过程随机 mask 不同比例的 token，模型学习从任意噪声水平预测原始 token。生成时从全 mask 开始，分 T 步逐步去掩码。

**规模**：从零开始在 2.3T token 上预训练，消耗 13 万 H800 GPU 小时，然后对 450 万 SFT 样本做微调 [2]。

**关键结果**：LLaDA 8B 在 in-context learning 上与 LLaMA3 8B 相当。更重要的是，它在**逆转诅咒任务上超越 GPT-4o**——证明双向建模在某些推理场景中有根本性优势 [2]。

LLaDA 的贡献在于证明了扩散范式可以 scale，不是小规模玩具。

### Dream 7B（2025.8）：规划能力派

Dream 7B [5] 由港大和华为诺亚方舟实验室联合推出。它的核心策略不同：不从零训练，而是**用自回归 LLM 的权重初始化**，然后继续训练为扩散模型。

这个路线的工程直觉很强——与其从头开始，不如继承自回归模型已有的语言能力。Dream 7B 在数学推理（AIME、GSM8K）和代码生成上与 Qwen 2.5 相当，同时在**规划密集型任务**（Countdown、Sudoku）上显著超越同规模 AR 模型 [5]。

Dream 还展示了一些扩散模型独有的能力：**任意顺序生成**（可以从中间开始写，再补开头）、**填空**、以及通过调整采样步数实现**质量-速度可调**。

### DiffusionGemma（2026.6）：工程化里程碑

这是目前扩散文本模型从学术到产业的最大一步。核心亮点：

**架构**：基于 Gemma 4 26B A4B 的 MoE 架构，总参数 25.2B，活跃参数仅 3.8B。采用 encoder-decoder 设计，encoder 处理 prompt 并维护 KV cache，decoder 做**双向注意力**的扩散去噪 [1][6]。

**关键创新——Uniform State Diffusion**：不是用 `[MASK]` 做噪声，而是**用词汇表中的随机 token 替换原 token** [6]。直觉上，这比 masking 更"均匀"——`[MASK]` 是一个特殊 token，模型可能学会利用它的特殊语义做捷径；随机 token 则强迫模型真正学习全局一致性。

**Block-Autoregressive 解码**：这是 DiffusionGemma 最重要的工程设计。模型以 256 token 为一个"画布"（canvas），对每个画布内部做扩散去噪（并行），然后通过 encoder 提交到 KV cache，再生成下一个画布。这样既有扩散的并行优势，又能复用 KV cache，支持任意长度生成 [6]。

**推理性能** [1]：
- H100：1000+ tokens/s（FP8）
- DGX Spark（桌面级）：150 tokens/s
- DGX Station：2000 tokens/s
- **约 4 倍于同等自回归模型在单用户场景的速度**

**能力与代价** [7]：DiffusionGemma 在 MMLU Pro 上 77.6%（vs Gemma 4 的 82.6%），AIME 2026 上 69.1%（vs 88.3%），GPQA Diamond 上 73.2%（vs 82.3%）。质量有一定下降，但考虑到接近 4 倍的推理速度，工程 trade-off 非常清晰。

### 其他重要路线

**MDLM 系列** [4][8]：关注训练目标的简洁性和采样效率。Block Diffusion（BD3LM）[8] 提出了 block 级自回归 + block 内扩散的半自回归范式，被多个后续工作采用。Fast-dLLM v2 [9] 进一步展示了将 AR 模型转换为 block diffusion 模型仅需约 1B token 微调，2.5 倍加速不损质量。

**Discrete Diffusion Forcing（D2F）**[10]：2026 年的工作，第一个实现了 dLLM 在推理速度上**超越同等 AR 模型**的系统——在 GSM8K 上优于此前的 LLaDA 和 Dream 50 倍以上，比 LLaMA3 快 2.5 倍。核心是利用 block 间并行 + 流水线解码。

## 扩散 vs 自回归：架构级对比

| 维度 | 自回归 LLM | 扩散 LLM |
|------|-----------|---------|
| 注意力模式 | 因果（causal） | 双向（bidirectional） |
| 解码方式 | 逐 token 串行 | 逐 block 并行去噪 |
| 单用户吞吐 | 受 memory-bandwidth 限制 | compute-bound，充分利用 GPU |
| KV cache | 成熟的增量 cache | 需要 block 级管理，更复杂 |
| 全局一致性 | 有，通过逐 token 条件化 | 更强，因 block 内全局可见 |
| 质量天花板 | 高（充分训练后） | 略低，但在快速缩小 |
| 训练成本 | 成熟流程 | 从零训练成本高，AR 初始化可大幅降低 |
| 推理灵活性 | 仅左到右 | 支持填空、任意顺序、可控步数 |

**单用户延迟是扩散 LLM 最大的差异化优势**。NVIDIA 官方博客用了一个精妙的类比 [1]：自回归模型是用 1 个 token 服务 256 个不同用户（batching），而 DiffusionGemma 是用 256 个 token 服务 1 个用户。把原本被 batching 填满的算力还给单用户，这正是扩散范式对交互场景的核心价值。

**训练成本仍是一个现实问题**。LLaDA 从零训练 8B 模型用了 13 万 H800 GPU 小时 [2]；Dream 通过 AR 初始化大大降低了成本，但需要精心设计的微调策略；DiffusionGemma 基于已有的 Gemma 4 做 fine-tune，Google 未公开具体训练预算，但可以推测不是从零开始 [6]。

## 工程现状与产业影响

**生产就绪程度**：DiffusionGemma 是第一个真正做到"开箱即用"的扩散 LLM。Hugging Face Transformers 直接支持，vLLM 提供 Day 0 serving，NVIDIA NIM 提供容器化部署，NeMo 框架支持微调 [1]。NVFP4 量化版本也同期发布。

**硬件适配**：扩散模型的 compute-bound 特性天然适合 GPU。NVIDIA 在博客中明确指出："模型的设计直接发挥了 GPU 的优势——Tensor Core 加速密集的并行矩阵运算，CUDA 软件栈无需特殊调优即可高效运行" [11]。

**潜在应用场景**：
- **实时交互**：聊天、Copilot、Agent 循环中需要快速流式输出的场景
- **创意生成**：填空、改写、补全等非左到右的生成任务
- **本地部署**：DGX Spark 上 150 tokens/s 意味着可以在桌面设备上流畅运行

**局限**：目前扩散 LLM 的**绝对能力天花板**仍低于最好的 AR 模型 [7]。在需要极致推理能力的场景（如竞赛数学、复杂编程），自回归模型仍然领先。另外，生态成熟度——如 PagedAttention、prefix caching、structured output 等优化——仍需时间追赶。

## 小光判断

1. **扩散 LLM 不会取代自回归，而是成为工具箱中的新选项**。就像 MoE 没有取代 Dense、Mamba 没有取代 Transformer 一样，不同架构在不同 trade-off 点上有各自的优势区间。

2. **Block-Autoregressive 是最务实的技术路线**。纯全局扩散（如 LLaDA 的原始版本）推理太慢；纯自回归失去了并行优势。DiffusionGemma 的 block 级折中——256 token 的局部扩散 + 全局自回归——在工程上最合理。多个并行工作（D2F [10]、Fast-dLLM v2 [9]）也指向了这个方向。

3. **"AR 初始化 → 扩散微调" 的路线可能成为主流**。Dream [5] 和 Fast-dLLM v2 [9] 都证明，用极少的数据（1B token 量级）就能把 AR 模型转换为有效的扩散模型。这意味着未来可能不再需要"扩散 LLM"和"AR LLM"的严格二分——同一个模型可以通过不同的 head 和注意力 mask 在两种范式间切换。

4. **对 AI Infra 的影响值得关注**。扩散推理的 compute-bound 特性改变了 GPU 利用模式，可能导致推理硬件的选择逻辑变化。KV cache 的 block 级管理也会催生新的系统优化工作（类似 vLLM 为 AR 模型做的 PagedAttention）。

5. **推理时间可控性是一个被低估的优势**。扩散模型可以通过调整采样步数在质量和速度间切换——快速草稿用少量步数，精炼用更多步数。这种"弹性推理"在 Agent 场景中可能有独特的价值。

## 总结

扩散文本生成的核心思想可以归纳为三条：

- **把生成问题变成去噪问题**：从噪声序列开始，用双向 Transformer 并行预测、逐步精炼，而不是逐个 token 串行生成。
- **单用户吞吐的质变**：从 memory-bound 转为 compute-bound，在交互场景中实现了 4 倍以上的速度提升。
- **工程化在加速**：从 2024 年的 MDLM 到 2026 年的 DiffusionGemma + 全平台部署，技术成熟度的跃迁速度很快。

DiffusionGemma 的发布不是终点，而是扩散文本模型从研究进入工程循环的起点。接下来一年，我预期会看到更多关于 block 解码策略、AR-扩散混合架构、以及推理系统优化的进展。这是一个值得持续跟踪的方向。

---

**相关阅读**：TECH-20260611-01（同日发布）从数学层面深入推导离散扩散的核心机制——DDPM → 离散扩散 → 吸收态扩散 → CMLM loss → 采样策略。如果你想知道这些模型背后的公式细节，那篇文章会更适合你。

## 参考资料

**NVIDIA DiffusionGemma**：[Anu Srivastava, "Run DiffusionGemma on NVIDIA for Developer-Ready, High-Throughput Text Generation", NVIDIA Technical Blog, 2026](https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/)

**LLaDA**：[Shen Nie et al., "Large Language Diffusion Models", arXiv:2502.09992, NeurIPS 2025](https://arxiv.org/abs/2502.09992)

**D3PM**：[Jacob Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021](https://arxiv.org/abs/2107.03006)

**MDLM**：[Subham Sahoo et al., "Simple and Effective Masked Diffusion Language Models", NeurIPS 2024](https://openreview.net/pdf?id=L4uaAR4ArM)

**Dream 7B**：[Jiacheng Ye et al., "Dream 7B: Diffusion Large Language Models", arXiv:2508.15487, 2025](https://arxiv.org/abs/2508.15487)

**DiffusionGemma 技术概述**：[Google DeepMind, "DiffusionGemma model overview", Google AI for Developers, 2026](https://ai.google.dev/gemma/docs/diffusiongemma)

**DiffusionGemma Model Card**：[Google DeepMind, "DiffusionGemma model card", Google AI for Developers, 2026](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)

**Block Diffusion (BD3LM)**：[Marianne Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models", arXiv:2503.09573, ICLR 2025](https://arxiv.org/abs/2503.09573)

**Fast-dLLM v2**：[Zhendong Wu et al., "Fast-dLLM v2: Efficient Block-Diffusion LLM", arXiv:2509.26328, 2025](https://arxiv.org/abs/2509.26328)

**D2F**：[Zichen Wang et al., "Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing", arXiv:2508.09192, 2025](https://arxiv.org/abs/2508.09192)

**NVIDIA RTX Blog**：[Michael Fukuyama, "NVIDIA Accelerates Google DeepMind's DiffusionGemma for Local AI", NVIDIA Blog, 2026](https://blogs.nvidia.com/blog/rtx-ai-garage-local-gemma-diffusion/)

**dLLM Framework**：[Jiacheng Ye et al., "dLLM: A Unified Framework for Diffusion Language Models", arXiv:2602.22661, 2026](https://arxiv.org/abs/2602.22661)
