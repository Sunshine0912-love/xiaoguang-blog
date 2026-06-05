---
title: "KV Cache 量化技术解析：从均匀量化到方差归一化"
date: 2026-06-05 12:45:00
categories: ["TECH", "AI", "AI Infra"]
tags:
 - KV Cache
 - Quantization
 - LLM Inference
 - KVarN
 - TurboQuant
 - SmoothQuant
description: "从 KV Cache 内存瓶颈、KIVI 非对称量化、Hadamard 旋转到 KVarN 方差归一化，解释为什么长推理里的量化误差会累积，以及如何用双轴 scaling 抑制 token magnitude error。"
mathjax: true
---

> 阅读时间：约 15 分钟  
> 主题类型：TECH 技术点讲解 / 推理优化  
> 关键词：KV Cache、Quantization、KIVI、KVarN、Hadamard Rotation、Variance Normalization

## TL;DR

KV Cache 量化不是简单把 FP16 改成 INT4/INT2。真正难点在于：长链路自回归解码里，已经量化过的 cache 会影响下一层、下一段 token 的隐藏状态，量化误差会沿时间和层数累积。KVarN 的核心思路是先用 Hadamard rotation 摊平 channel outlier，再用双轴 variance normalization 修正 token magnitude error，最后再做低比特 round-to-nearest 量化 [1]。

这篇只讲一个技术点：**为什么 KV Cache 量化中的 token scale error 会伤害长推理，以及 KVarN 的 VarN 怎么缓解它**。

## KV Cache 为什么会成为推理瓶颈

自回归 LLM 每生成一个 token，都要让当前 query 去看过去所有 token 的 key 和 value。为了避免每步重算过去 token 的 $K,V$，推理引擎会缓存每层每个 token 的 KV：

$$\operatorname{Cache}=B\cdot L\cdot N_{layer}\cdot N_{kv}\cdot 2d_h\cdot bytes$$

其中 $B$ 是 batch size，$L$ 是上下文长度，$N_{layer}$ 是层数，$N_{kv}$ 是 KV head 数，$2d_h$ 里的 2 来自 $K$ 和 $V$。PagedAttention / vLLM 解决的是 cache 分配和分页浪费问题 [6]，但没有改变每个 cache 元素占多少 bit。

KV Cache 量化试图把 $bytes$ 这一项降下来。FP16 是 16 bit，如果把 $K,V$ 压到 4 bit，理论上 cache 容量能扩大约 4 倍；压到 2 bit，理论容量更高。但低 bit 会引入重构误差：

$$K_{dq}=D(Q(K)),\quad V_{dq}=D(Q(V))$$

这里 $Q(\cdot)$ 是量化，$D(\cdot)$ 是反量化。attention 实际读到的不是 $K,V$，而是 $K_{dq},V_{dq}$：

$$O=\operatorname{softmax}\left(\frac{qK_{dq}^{\top}}{\sqrt{d_h}}\right)V_{dq}$$

只要 $K_{dq}$ 的方向或尺度错了，attention logits 会偏；只要 $V_{dq}$ 错了，读出的内容会偏。

## 最基础的线性量化

最常见的 round-to-nearest 量化可以写成：

$$X_q=\operatorname{clip}\left(\operatorname{round}\left(\frac{X}{s}\right)+z, q_{min}, q_{max}\right)$$

反量化是：

$$X_{dq}=(X_q-z)\cdot s$$

$s$ 是 scale，$z$ 是 zero point。问题是：$s$ 应该沿哪个维度共享？

如果整块 tensor 共享一个 scale，大 outlier 会把 scale 拉大，让大多数普通值分辨率变差；如果每个元素都有 scale，又会让 metadata 太重，不像量化。KV Cache 量化的核心工程问题就是：**用尽量少的 scale，保住对模型输出最敏感的结构**。

## KIVI 的关键发现：K 和 V 不该同轴量化

KIVI 是 KV Cache 量化里的重要基线。它的观察是：key cache 和 value cache 的统计性质不一样，因此 key 更适合 per-channel 量化，value 更适合 per-token 量化 [2]。

以一个单 head 的 tile 为例：

$$X\in\mathbb{R}^{C\times R}$$

$C$ 是 head dimension，$R$ 是 token chunk size。KVarN 论文沿用类似 tile 处理方式，举例中常见 $C=128,R=128$ [1]。

KIVI 对 $K$ 的典型反量化形式是：

$$K_{dq}=(K_q+\vec{z})\odot\vec{s}$$

$\vec{s},\vec{z}$ 沿 channel 维度保存。它的直觉是 key 的 channel outlier 更麻烦，per-channel scale 能更好地覆盖不同 channel 的数值范围。对 $V$，per-token 往往更合适，因为 value 更像被读出的 payload，token 之间的幅度差异更重要 [2]。

KIVI 的贡献在于把“KV cache 可以 2-bit 量化”这件事系统化了：论文报告它能在多种 LLM 上保持接近质量，同时降低峰值内存、扩大 batch 并提高吞吐 [2]。

但 KIVI 更像解决了“静态 prefill cache 怎么压”的问题。KVarN 进一步指出，在长推理、test-time scaling、reasoning 这类 decode-heavy 场景，误差会以不同方式出现 [1]。

## 为什么 decode-heavy 场景更难

很多 KV 量化评测像这样做：

1. 先把长 prompt 一次性 prefill。
2. 把完整 KV cache 量化。
3. 看模型能不能在这个静态 cache 上回答。

这更接近 retrieval 或 NIAH 的静态长上下文场景。但 reasoning 模型常常一边生成很长 chain-of-thought，一边不断把新生成 token 的 KV 写入 cache。KVarN 论文把这个叫作 pseudo-decode：每过一个 block，新产生的 $K,V$ 被量化写回，后续 token 只能读量化 cache [1]。

误差链条可以写成：

$$h_t^{(l)} = f_l(h_t^{(l-1)}, K_{dq,<t}^{(l)}, V_{dq,<t}^{(l)})$$

如果第 $l$ 层的量化 cache 让 $h_t^{(l)}$ 偏了，那么第 $l+1$ 层生成的新 $K,V$ 本身就已经偏了；这些新 $K,V$ 再被量化，误差继续传下去。于是误差不仅来自“当前 cache 重构不准”，还来自“后续 cache 是在错误隐藏状态上生成的”。

这就是 KVarN 关注 error accumulation 的原因。

## 误差拆解：方向错，还是尺度错

KVarN 论文把 full-precision key $K$ 和反量化 key $K_{dq}$ 的平方误差拆成两部分 [1]：

$$\|K-K_{dq}\|^2=(\|K\|-\|K_{dq}\|)^2+2\|K\|\|K_{dq}\|(1-\cos\theta)$$

第一项是 magnitude error：

$$E_M=(\|K\|-\|K_{dq}\|)^2$$

第二项是 directional error：

$$E_D=2\|K\|\|K_{dq}\|(1-\cos\theta)$$

$\theta$ 是 $K$ 和 $K_{dq}$ 的夹角。这个拆解很有用：directional error 表示方向变了，magnitude error 表示 token norm 被缩小或放大了。

论文的关键经验判断是：最坏的一小部分 outlier error 对 end-to-end KL divergence 影响更大，而这些 outlier 很多由 token magnitude error 驱动 [1]。换句话说，平均 MSE 不是全部故事；少量 token 的 scale 被量化放飞，可能比许多小误差更伤模型。

## Hadamard rotation 解决什么

Hadamard rotation 属于 incoherence processing。它用一个正交矩阵 $H$ 混合 channel：

$$\tilde{X}=HX$$

因为 $H$ 是正交的，理想情况下它不改变向量范数和内积结构，却能把集中在少数 channel 的 outlier 能量摊开。TurboQuant 和 KVarN 都使用 rotation 这条路线来改善 KV cache 的低比特量化 [1][3]。

直觉上，如果原来某个 channel 特别大，那么 per-channel 或 per-block quantizer 的 scale 会被它支配；rotation 后，极端值被分散到多个维度，数值分布更接近均匀/高斯，rounding 更不容易被少数 outlier 绑架。

但 KVarN 的一个重要结论是：Hadamard rotation 对 channel outlier 有帮助，但不足以解决 token-wise scaling error [1]。这就引出 VarN。

## KVarN 的 VarN：双轴方差归一化

KVarN 在一个 tile 上做处理。设 tile：

$$T\in\mathbb{R}^{N\times R\times C}$$

$N$ 是 batch 或 tile 批数，$R$ 是 token chunk，$C$ 是 channel/head dimension。KVarN 维护两组 scale：

$$S_c\in\mathbb{R}^{N\times1\times C},\quad S_r\in\mathbb{R}^{N\times R\times1}$$

归一化后的 tile 是：

$$T_{bal}=T\oslash S_c\oslash S_r$$

$S_c$ 修 channel 轴，$S_r$ 修 token 轴。算法在 log space 里交替更新两者，让行/列方差更均衡 [1]。

一个简化伪代码是：

```python
def varn(T, iters=8):
    log_sc = zeros([N, 1, C])
    log_sr = zeros([N, R, 1])
    best = None

    for _ in range(iters):
        C_now = T / exp(log_sc) / exp(log_sr)
        log_sc += 0.5 * log(clamp(var_over_tokens(C_now)))

        C_now = T / exp(log_sc) / exp(log_sr)
        log_sr += 0.5 * log(clamp(var_over_channels(C_now)))

        keep_best_if_imbalance_smaller()

    return T / S_c_best / S_r_best, S_c_best, S_r_best
```

这不是为了让矩阵“看起来好看”，而是为了让 quantizer 在两个维度上都别被极端 scale 牵着走。KVarN 论文说，额外的 per-row second scale 可以 fused into dequant kernel，因此不需要额外 HBM round-trip；在他们的 Triton 测量里，双 scale 相比 KIVI 单 scale 的 dequantization gap 最多约 1.4%，16k 和 32k 时接近测量噪声 [1]。

## KVarN 的完整 pipeline

KVarN README 和论文给出的流程可以概括为四步 [1][7]：

1. Cache：拿到原始 FP16 KV tile。
2. Rotated Cache：沿 channel 做 Hadamard rotation，摊平 channel outlier。
3. Normalized Cache：对 token 和 channel 两个维度做 VarN。
4. Quantized Cache：用 asymmetric round-to-nearest 存低比特值、scale 和 zero point。

仓库 README 还给出 vLLM 使用方式：它目前以 vLLM fork 发布，示例里通过 `kv_cache_dtype="kvarn_k4v2_g128"` 和 `block_size=128` 启用，服务端对应 `--kv-cache-dtype kvarn_k4v2_g128 --block-size 128` [7]。这里的 `k4v2` 表示 release preset 对 key/value 使用不同 bit 配置：key 4-bit，value 2-bit；论文实验则讨论 2-bit K/V 与约 2.3 average bits/element 等设置 [1][7]。

这点很重要：论文机制和仓库 release preset 不完全是同一个“单一数字”。写生产报告时不能只说“KVarN=2-bit”或“KVarN=4-bit”，要说明具体 dtype、group size、sink/trailing token 策略和 metadata 计算方式。

## 和 SmoothQuant、TurboQuant 的关系

SmoothQuant 不是 KV Cache 量化方法，但它提供了一个关键思想：把难量化的 activation outlier 通过数学等价变换迁移到更容易量化的 weights 上 [4]。KVarN 借鉴的不是 SmoothQuant 的具体做法，而是“通过缩放/预处理改变量化难度分布”的思路。

TurboQuant 则更接近 KV Cache：vLLM 文档把 TurboQuant 描述为 rotation、optimized grid 和可选 renormalization 的 KV cache compression 技术 [3]。KVarN 与它的共同点是都使用 rotation 来改善低比特量化；差异在于 KVarN 明确把 decode error accumulation 和 token magnitude error 作为核心问题，并引入双轴 VarN。

KIVI、TurboQuant、KVarN 可以这样粗略对比：

| 方法 | 主轴 | 解决重点 | 风险 |
|---|---|---|---|
| KIVI | K per-channel、V per-token | 低 bit KV cache 的基本统计结构 | 长 decode 下 token scale error 仍可能累积 |
| TurboQuant | rotation + grid / residual correction | 让低 bit vector quantization 更接近最优失真 | 实现复杂度、吞吐与精度要看后端 |
| KVarN | Hadamard + dual-axis VarN | 抑制 outlier token magnitude error 和 error accumulation | 新方法，需更多模型/后端复现 |

## 实验结果怎么读

KVarN 论文在 Qwen3-4B、Llama-3.1-8B-Instruct、Phi-4、Phi-4-reasoning-plus 等模型上评估，任务包括 MATH500、AIME24、HumanEval、IFEval 和 line retrieval [1]。

几个数字值得记，但必须按“论文报告”理解：

- Qwen3-4B 上，MATH500 FP16 为 82.6%，KVarN 2.3 bits/elem 为 79.2%；AIME24 FP16 为 61.1%，KVarN 为 60.0% [1]。
- Phi-4-14B 上，MATH500 FP16 为 84.9%，KVarN 为 84.8%；AIME24 FP16 为 62.2%，KVarN 为 61.7% [1]。
- HumanEval 上，KVarN 在 Qwen3-4B 和 Phi-4-14B 上分别接近 FP16，并优于多种低比特 baseline [1]。
- 论文报告 variance-normalization 相对标准生成的额外开销约 0.18%，双 scale dequantization 相比 KIVI 的 gap 最多约 1.4% [1]。

我不会把这些直接翻译成“生产无损”。原因很简单：这些是论文设置下的模型、任务、dtype、group size 和后端实现。你的 workload 可能是 RAG、代码补全、长文摘要、多轮 Agent 或中文问答，误差模式会不同。

## 工程落地要看什么

**第一，看错误模式，不只看平均分。**  
KV Cache 量化最怕极少数 token 破坏后续生成。评测要包括 KL divergence、long decode、reasoning、代码执行、格式遵循，而不是只看静态 NIAH。

**第二，看有效 bits，不只看 nominal bits。**  
scale、zero point、sink tokens、trailing tokens、未量化层都会增加实际内存。KVarN 论文明确把 auxiliary storage 算进 bits/elem [1]。

**第三，看后端 kernel。**  
低 bit cache 只有在反量化和 attention kernel 融合得好时才有意义。否则省了显存，吞吐可能掉。

**第四，看 page size 和 block size。**  
KVarN README 目前提示 tile/page size 固定为 128，其他 page size 未来再支持 [7]。如果服务系统的 block manager、prefix cache、chunked prefill 与这个假设不一致，需要实测。

**第五，看模型家族。**  
KIVI 就已经指出不同模型的 K/V 分布不同 [2]。KVarN 在几个模型上效果好，不等于所有模型、所有 RoPE/MLA/GQA 变体都同样稳。

## 常见误解

**误解 1：KV Cache 量化只是在存储上省内存。**  
不止。它改变了 attention 读到的 $K,V$，影响 logits 和输出 hidden states，长解码里还会累积。

**误解 2：MSE 越小就越好。**  
不一定。KVarN 论文强调 top outlier errors 对 end-to-end KL 影响更大，平均 MSE 可能掩盖最坏 token 的破坏力 [1]。

**误解 3：Hadamard rotation 已经够了。**  
rotation 能摊平 channel outlier，但 token magnitude error 仍可能存在。KVarN 的重点就是 rotation 后再做 token/channel 双轴 scaling。

**误解 4：2-bit 一定比 4-bit 更值得。**  
低 bit 的容量收益更大，但精度、metadata、kernel、吞吐和任务质量都要一起算。KVarN 仓库 release preset 用 `k4v2`，说明生产取舍不只是“越低越好” [7]。

## 小光判断

我认为 KV Cache 量化正在从“压存储”进入“控制误差动力学”的阶段。

KIVI 解决了第一层问题：K/V 分布不一样，要按不同轴量化。TurboQuant 代表第二层：用 rotation / vector quantization 让低 bit 更接近好失真率。KVarN 往前走了一步：它把 long-horizon decoding 中的 error accumulation 摆到台面上，指出 token scale error 是重要驱动，并用 VarN 去处理这个具体问题。

这就是它值得写成 TECH 的原因。它不是又一个“某仓库支持 2-bit KV cache”的新闻，而是提供了一个可复用判断框架：

1. 先问误差来自方向还是尺度。
2. 再问 outlier token 是否主导 end-to-end degradation。
3. 再决定是 rotation、per-axis scaling、mixed precision、还是保留少量 FP16。

未来真正强的 KV Cache compression，不会只比谁 bit 更低，而会比谁能在长解码、复杂任务、真实后端里控制误差传播。

## 总结

KVarN 的技术主线可以压成一条公式链：

$$T\rightarrow HT\rightarrow (HT\oslash S_c\oslash S_r)\rightarrow Q_b(\cdot)$$

其中 $H$ 解决 channel outlier，$S_c,S_r$ 分别修 channel 和 token 方差，$Q_b$ 才是最后的低比特量化。它最重要的洞察不是“多加一个 scale”，而是：长推理里的量化质量取决于 outlier token 的尺度是否被控制住。

对推理工程师来说，这篇论文给出的实践提醒很直接：评估 KV Cache 量化时，不要只跑静态长上下文；一定要跑 decode-heavy 的生成任务，观察错误是否随 token 增长而累积。

## 参考资料

[1] Lorenz K. Muller et al., [*KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks*](https://arxiv.org/abs/2606.03458), arXiv:2606.03458, 2026.

[2] Zirui Liu et al., [*KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*](https://arxiv.org/abs/2402.02750), ICML 2024 / arXiv:2402.02750.

[3] vLLM Project, [*TurboQuant: Accuracy and Performance / vLLM TurboQuant documentation*](https://vllm-project.github.io/2026/05/11/turboquant.html), 2026.

[4] Guangxuan Xiao et al., [*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*](https://arxiv.org/abs/2211.10438), ICML 2023 / arXiv:2211.10438.

[5] Huawei CSL, [*KVarN GitHub Repository*](https://github.com/huawei-csl/KVarN), 2026.

[6] Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023 / arXiv:2309.06180.

[7] DeepSeek-AI, [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434), arXiv:2405.04434, 2024.
