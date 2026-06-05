---
title: "投机采样原理与工程实现：如何用小模型无损加速大模型的自回归解码"
date: 2026-06-05 13:15:00
categories:
 - AI
 - 推理优化
tags:
 - Speculative Decoding
 - LLM Inference
 - Draft Model
 - Self-Speculation
 - vLLM
 - Rejection Sampling
description: "从自回归解码的 memory-bandwidth 瓶颈出发，推导投机采样的 draft-verify 两阶段框架、接受/拒绝采样的概率保证，并分析 draft model 选择的数学 trade-off 与 vLLM 的生产级实现要点。"
mathjax: true
---

> 阅读时间：约 15 分钟  
> 主题类型：TECH 技术点讲解 / 推理加速算法  
> 关键词：Speculative Decoding、Draft Model、Rejection Sampling、vLLM、Memory-Bandwidth Bound、Self-Speculation

## TL;DR

自回归 LLM 推理慢，不是因为 compute 不够，而是因为 memory bandwidth 不够：每生成一个 token 都要把全部模型参数从 HBM 搬到计算单元一次。投机采样（Speculative Decoding）用"小模型先猜，大模型后验"的思路，用一个轻量 draft model 一次预测 $k$ 个候选 token，大模型一次 forward pass 验完，通过修正拒绝采样保证采样分布严格等于大模型分布。在 memory-bandwidth bound 的 decode 阶段，理论上可获得约 $k \cdot \alpha$ 的加速，其中 $\alpha$ 是 draft model 的平均接受率 [1]。

## 为什么自回归解码是 memory-bandwidth bound

LLM decode 阶段每生成一个 token 的计算量是：

$$C_{token}=2\cdot N_{params}\approx 2P$$

$P$ 是模型参数量。GPU 的计算能力通常不是瓶颈——峰值算力远高于这每步的 FLOPs。真正慢的是把 $P$ 个参数从 HBM 搬到 SRAM 的带宽：

$$T_{step}\approx \frac{P\cdot bytes_{dtype}}{BW_{HBM}}$$

以 70B FP16 模型为例，$P=7\times10^{10}$，$BW_{HBM}\approx 3\text{ TB/s}$（H100）：每步约 $7\times10^{10}\times2/3\times10^{12}\approx 47$ ms。

每次只生成一个 token，却要搬走整个模型。computational intensity（每 byte 的 FLOP）极低。这就是 decode 阶段的本质瓶颈：**memory-bandwidth bound**。

并行采样（beam search 中的并行 beam）可以增加吞吐，但不改变"每个 token 搬一次完整参数"的这个基本事实。

## 核心直觉：多猜几个 token 再验

投机采样的想法直接来自这个瓶颈：大模型反正要把参数搬进 SRAM 一次，让它顺便多验证几个候选 token，计算量几乎不增加。只要 draft model 足够小，猜 $k$ 个 token 的总延迟远低于大模型生成 $k$ 个 token 的延迟 [1]。

以 Leviathan et al. (2023) 在 Chinchilla 70B 上的实验为例：draft model 比 target model 约小 100 倍，猜 $k=4$ 个 token 的成本约等于 target model 生成一个 token 的成本，接受率约 0.8，最终加速 2-2.5x [1]。

## 两阶段框架

投机采样分为两个阶段：

### 第一阶段：Draft

draft model $M_q$ 以自回归方式生成 $k$ 个候选 token（$k$ 是 speculation length）：

$$x_1\sim M_q(\cdot|x_{<t}),\quad x_2\sim M_q(\cdot|x_{<t},x_1),\quad ...,\quad x_k\sim M_q(\cdot|x_{<t},...,x_{k-1})$$

这一步很快，因为 $M_q$ 参数量远小于 target model $M_p$。工程上常用 T5-small、Llama-68M 或 n-gram LM 做 draft model。

### 第二阶段：Verify

target model $M_p$ 做一次 parallel forward pass，把 $x_1,...,x_k$ 当作输入，一次性算出每一步的条件概率：

$$p_1(x)=M_p(x|x_{<t})$$
$$p_2(x)=M_p(x|x_{<t},x_1)$$
$$...$$
$$p_k(x)=M_p(x|x_{<t},x_1,...,x_{k-1})$$

这里面用一个 attention mask 技巧：mask 的下三角和自回归解码一样，但因为 patch 已存在，可以并行算。这就让大模型一次 forward 验证了 $k$ 个位置。

## 修正拒绝采样：保证严格无损

最关键的部分是：怎么决定接受哪些 draft token？如果直接全接受，采样分布会偏离 target model，导致模型质量下降。

Leviathan et al. 的修正拒绝采样方案：对第 $i$ 个位置，比较 $M_p$ 和 $M_q$ 对该 token 的概率。

设 draft 输出的第 $i$ 个 token 是 $x_i$，定义：

$$\alpha_i=\min\left(1,\frac{p_p(x_i)}{p_q(x_i)}\right)$$

以概率 $\alpha_i$ 接受 $x_i$；以 $1-\alpha_i$ 拒绝 $x_i$。

如果拒绝：就从 target model 的分布中重新采样——但要修正掉已经被接受的概率质量：

$$p'(x)=\operatorname{norm}\left(\max\left(0,\frac{p_p(x)-p_q(x)}{Z}\right)\right)$$

其中 $Z$ 是归一化常数 [1]。

这个方案的核心保证：**输出 token 的联合分布严格等于只用 target model 自回归采样的分布**。换言之，加速不降质量。

## 直觉解释：什么时候接受率高

$\alpha_i$ 的公式提供了直接直觉：

- 如果 $p_p(x_i)>p_q(x_i)$，说明 draft model 小瞧了这个 token（target 比 draft 更自信），$\alpha_i=1$，肯定接受。
- 如果 $p_p(x_i)<p_q(x_i)$，说明 draft model 过于自信了，以 $p_p(x_i)/p_q(x_i)$ 的概率接受。
- 如果 $p_p(x_i)\approx p_q(x_i)$，接受率接近 1——这是理想情况。

所以 draft model 不一定要在 token 层面猜对，而是要在 **top-1 后面这段概率匹配得越好，接受率越高**。这意味着高质量的小模型比同样大小但分布离 target 远的小模型更有价值。

## 工程加速公式

设 $T_{target}$ 是大模型生成一个 token 的时间，$T_{draft}$ 是小模型生成一个 token 的时间。

投机采样每步的时间是：

$$T_{step}=k\cdot T_{draft}+T_{verify}$$

其中 $T_{verify}\approx T_{target}$（因为一次 forward 的计算量和一个 token 差不多）。

期望生成的 token 数（每步"净产出"）是 acceptance rate $\alpha$ 的函数，Leviathan et al. 推导出每步期望接受 token 数为 [1]：

$$\mathbb{E}[\#accepted]=\frac{1-\alpha^{k+1}}{1-\alpha}$$

实际加速比：

$$speedup=\frac{\mathbb{E}[\#accepted]\cdot T_{target}}{k\cdot T_{draft}+T_{target}}$$

最优的 $k$ 不是越大越好：draft 成本随 $k$ 线性增长，而接受率会随 $k$ 下降（因为越往后 draft 越容易偏离 target 分布）。Leviathan et al. 在实验中常用 $k\in[3,8]$ [1]。

当 $T_{draft}\ll T_{target}$ 且 $\alpha$ 较高时，加速比趋近于每步接受 token 数。理解这一点就够了：**加速取决于 draft 的接受率，而接受率是 draft-target 分布匹配质量的可度量代理**。

## 只比加速比，不比 accept rate 就是耍流氓

以下是分析框架，不是论文跑分结论。

测投机采样不能只看加速比。一个 draft model 猜得少但准（低 $k$ 高 $\alpha$）和猜得多但准头差（高 $k$ 低 $\alpha$）可能加速比一样，但前者延迟更低、KV cache 压力更小。

关键维度：

1. **latency 和 QPS 要分开看**。投机采样本质上降低 per-token latency，但单次 forward 的 batch 内可能利用率更高。RAG、聊天补全、代码生成不同 workload 的敏感度不一样。
2. **draft-target size ratio**。draft 太小，接受率低；draft 太大，$T_{draft}$ 不低。最优 ratio 取决于 target 的分布特性和 hardware。
3. **温度**。temperature 越高，分布越 flat，$\alpha$ 会下降。Leviathan et al. 的拒绝采样对任意 temperature 有效，但高温度时加速效果有限 [1]。
4. **domain shift**。draft 和 target 在训练分布、领域微调上的一致程度直接影响 $\alpha$。

## 不用单独 draft model：Self-Speculation 路线

维护一个单独的 draft model 有部署成本：额外显存、额外加载、版本对齐。于是出现了"让同一个模型自己猜自己"的路线。

### Medusa：多个解码头同时猜

Medusa 在 backbone LLM 的最后隐藏层上挂多个轻量 FFN 头，每个头负责预测第 $i$ 步之后的 token [2]：

- Head 1：预测 $x_{t+1}$
- Head 2：预测 $x_{t+2}$
- …

训练时只训这些 head，backbone 冻结（Medusa-1），或与 backbone 联合训练但要保持 backbone 质量（Medusa-2）[2]。

Medusa 把多个头的预测组织成 tree attention：每个位置可能有多条候选分支，一次 verify forward 验证整棵树。这比逐 token draft 多了有效并行度。

论文报告 Medusa-1 可到 2.2x，Medusa-2 可到 2.3-3.6x [2]。代价是需要针对每个 backbone fine-tune Medusa heads。

### n-gram / prompt lookup 投机

这是最简单的 self-speculation：用 prompt 里已经出现的 n-gram 当作草稿 [1]。不需要任何额外模型或训练，但接受率比训练好的 draft 低很多。

### EAGLE 和 EAGLE-2

EAGLE 走更深：不只是猜下一个 token，而是从特征层面预测 draft。核心发现是 draft model 的 hidden states 包含的不确定性信息有助于提高接受率。EAGLE 比 Medusa 更进一步——它不仅输出多个候选，还在特征空间里"修正" draft 的不确定性。

## vLLM 生产级实现要点

vLLM 对投机解码的生产化实现有几个关键设计：

**1. 统一的 Spec Decode Worker**

vLLM 的 speculative decoding 支持两种 draft 模式：draft model 和 n-gram。代码在 `vllm/spec_decode/` 下，`SpecDecodeWorker` 协调整个 draft-verify 循环 [3]。

**2. Draft 与 Target 的 KV Cache 管理**

一次投机步需要的 KV cache：
- Draft model 自回归 $k$ 步产生 $k$ 个新 KV slot
- Target model 一次 verify 需要 $k$ 个位置的 KV

vLLM 的 PagedAttention 分页管理天然适配这个场景。Draft slots 可以被复用/丢弃 [3]。

**3. Batch expansion**

在线服务中不同请求的接受率不同，vLLM 用 batch expansion 统一处理：把 draft 完成的多个候选路径放进一次 target verify batch，verify 后再根据接受结果删掉拒绝的分支。这比每个请求单独 verify 更高效 [3]。

从 vLLM CLI 启用投机解码的典型命令：

```bash
vllm serve <model> \
  --speculative-model <draft_model> \
  --num-speculative-tokens 5
```

或使用 n-gram：

```bash
vllm serve <model> \
  --speculative-config method=ngram,window=10 \
  --num-speculative-tokens 5
```

`num-speculative-tokens` 就是论文里的 $k$。

**4. 与 prefix caching 的交互**

投机采样新产生的 KV cache 和 prefix KV cache 的交互需要小心处理：如果多个请求共享相同 prefix，draft 路径可能不同，prefix cache 命中会受影响 [3]。

## 为什么不是万能药

**第一，compute-bound 场景无效。**  
投机采样加速的前提是 decode 为 memory-bandwidth bound。在 prefill（一次 forward 处理整个 prompt）或大 batch decode（compute 不再闲置）时，收益很小甚至为负。

**第二，draft model 的显存开销。**  
即使 draft model 很小（几百 MB），在线服务中也要常驻显存，可能与 large model serving 的显存预算冲突。

**第三，高温度退化为 standard decoding。**  
如前所述，$T\rightarrow\infty$ 时 $p_p$ 变 flat，$\alpha$ 降到 $1/|V|$ 级别。

**第四，量化与量化的交互。**  
draft model 量化到 INT8/INT4 可以降低显存开销，但可能进一步降低接受率——需要在 vLLM 的实测中评估，不能只看论文声称。

**第五，批处理。**  
在线服务中不同请求的 accept rate 不一样，batch expansion 的处理可能引入同步开销。个别慢请求会拖慢整个 batch。

## 与 LLM 推理系列的关系

这个系列前面讲过的内容都和投机采样有关：

- **Prefill/Decode 解耦**：投机采样只在 decode 阶段有效。解耦后 decode worker 可以专注于加速 decode。
- **KV Cache 量化**：量化后的 KV cache 会不会影响 draft model 和 target model 的分布匹配？这是一个开放工程问题。
- **MLA**：MLA 的压缩 K/V 是否影响投机 decoder 的 verify？需要 case-by-case 验证。

## 思考框架速查

| 维度 | 问题 | 信号 |
|---|---|---|
| 适用性 | decode 是否 memory-bandwidth bound？ | per-token latency >2x compute time |
| Draft 选型 | draft-target 大小比例？ | $\alpha$ 在验证集上的度量 |
| 部署复杂度 | 是否接受额外模型？ | 有 GPU memory headroom 用 draft model，无 headroom 用 n-gram/Medusa |
| 温度 | 服务的目标温度是多少？ | $T>0.8$ 时加速可能不显著 |
| 验证 | 输出质量是否真的无损？ | 在同一套 benchmark 上对比 target 和 spec 的 pass@k、BLEU、ROUGE |
| 回退策略 | 如果 $\alpha$ 太低怎么办？ | 自动回退到 standard decoding |

## 小光判断

投机采样是我认为自回归解码方向最重要的算法之一。不是因为它复杂——恰恰相反，它的概念极其简单——而是因为它把"加速而保持精确"这件事做得很干净。

我会把它放进 LLM 推理系统解析系列的基础算法篇。后面讲推理服务架构时，投机解码必然出现在 decode worker 的调度逻辑里。如果你正在调推理服务：

1. 先确认 decode 确实是 memory-bandwidth bound（batch size ≤ 1-2 时通常是）。
2. 用 n-gram 投机试水——零成本，看加速比。
3. 如果加速比有吸引力但不够，再考虑 draft model 或 self-speculation。
4. 最后才看 EAGLE/Medusa 这类需要特定 fine-tune 的方法。

毕竟工程上是 80/20 法则：一个简单的 n-gram speculator 有时就够，不必为了最后 20% 去维护一个完整的 Medusa fine-tune pipeline。

## 总结

投机采样从 memory-bandwidth 瓶颈出发，把一个 token 一次搬参数的机会变成一次验证 $k$ 个 token 的机会。核心创新是一套修正拒绝采样方案，保证了输出分布严格等于 target model。

$$x_1^k\sim M_q,\quad p_{p}^{(1:k)}(x)=M_p^{(1:k)}(x|x_{<t},x_1^{k-1})$$

$$\alpha_i=\min\left(1,\frac{p_p(x_i)}{p_q(x_i)}\right)$$

记忆这三条线就够：

1. 小模型猜多个 token → 大模型一次验完 → 按修正拒绝概率接受/拒绝。
2. 加速来自 draft-target 分布匹配质量，不是"把小模型变大"。
3. vLLM 的生产实现已经覆盖 draft model / n-gram 两种模式，Medusa / EAGLE 是更高阶的 self-speculation 路线。

## 参考资料

[1] Yaniv Leviathan, Matan Kalman, Yossi Matias. *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv:2302.01318, 2023. https://arxiv.org/abs/2302.01318

[2] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, ICML 2024 / arXiv:2401.10774, 2024. https://arxiv.org/abs/2401.10774

[3] vLLM Project. *Speculative Decoding — vLLM Documentation*. https://docs.vllm.ai/en/latest/features/spec_decode.html — actual content at https://docs.vllm.ai/en/latest/speculative_decoding/

[4] NVIDIA TensorRT-LLM. *Speculative Decoding*. https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog/speculative_decoding.md

[5] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv:2302.01318, 2023. (Note: same paper as [1], different author ordering by convention)

[6] Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature-level Uncertainty*, 2024. https://sites.google.com/view/eagle-llm
