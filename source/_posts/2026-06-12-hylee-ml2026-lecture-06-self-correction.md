---
title: "【ML 2026 Spring 第6讲】如何教育模型（二）：Self-Correction 的技术全景"
date: 2026-06-12 12:06:00
categories: ["AI", "Course"]
tags:
 - ML2026
 - Self-Correction
 - Self-Refine
 - Reflection
 - Constitutional AI
 - Alignment
description: "李宏毅 ML 2026 Spring 第6讲：从 Self-Refine 到 Reflection 到 Constitutional AI，拆解让模型自我纠错的核心技术和训练机制。"
series: hylee-ml-2026-spring
lecture: 6
mathjax: true
---

> 课程：李宏毅 Machine Learning 2026 Spring  
> 讲次：第 6 讲（4/24）  
> 主题：如何教育模型（二）— Self-Correction  
> 课程影片：[Self-Correction](https://youtu.be/m3i2mk5hs8U)  
> 讲议：[Self-Correction.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Correction.pdf)  
> 课程页：[ML 2026 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)  
> 前一讲：[第5讲：如何教育模型（一）— Post-Training 与遗忘问题](/)  
> 相关录影：[2023 课程 Self-Correction 章节](https://youtu.be/bJFtcwLSNxI)

---

## TL;DR

模型出错了怎么办？与其让人手动纠错，不如让模型<strong>自己发现并修正自己的错误</strong>。李宏毅本讲从三个层次系统拆解 Self-Correction：<strong>修改推理过程</strong>（Contrastive Decoding 家族，不训练模型只改变解码策略）、<strong>修改 Harness/Workflow</strong>（插入 Reflection Instruction 触发自我反思）、<strong>修改模型参数</strong>（通过 SFT 或 RL 训练模型的自我修正能力）。本讲覆盖了 DoLa、CAD、ICD、VCD 等十余种 contrastive decoding 变体，以及 Self-Refine、ReVISE、RLVR 等训练方法，并通过 Parity Check 实验和 RLVR 能力边界讨论，揭示了当前 Self-Correction 研究的核心矛盾——模型到底学会了"修正"，还是学会了"更长的推理"。

## 本讲目标

1. 理解 Self-Correction 的三条技术路线：改推理、改流程、改参数
2. 掌握 Contrastive Decoding 的核心思想及 DoLa、CAD、ICD 等变体的设计动机
3. 理解 Self-Refine / Reflection 的工作机制与效果分析
4. 了解 RLVR 如何训练模型的自我修正行为
5. 认识 Self-Correction 研究的核心争议：模型真的会修正吗？

## 前置知识

- LLM 的自回归解码过程（logits → softmax → token sampling）
- Transformer 的层级结构（layer stack、hidden representations）
- 基本的 RL 概念（reward、policy、RLVR）
- 建议先看：[第5讲：Post-Training 与遗忘问题](/)

---

## 为什么需要 Self-Correction？

语言模型生成的本质是一个从左到右的单向过程——一旦一个 token 被生成，就无法"撤回"。这导致即使模型"知道"自己可能犯了错，也无法在生成过程中回头修正。

这和我们人类的写作过程完全不同。我们写文章时会读一遍、改一改、再打磨——迭代修正才是正常的认知行为。

李宏毅在课堂上给出了一个精辟的总结：<strong>批判比生成容易</strong>——你不需要会写小说，也能判断一本小说好不好看。Self-Correction 的核心洞察就是利用这种不对称性。

本讲将所有 Self-Correction 方法归为三大类：

| 方法类别 | 核心思路 | 是否改模型参数 | 典型方法 |
|---|---|---|---|
| 修改 Inference 过程 | 在解码时通过对比"好输出"和"坏输出"来修正 | ❌ 不改参数 | Contrastive Decoding, DoLa, CAD, ICD, VCD, MTI |
| 修改 Harness (Workflow) | 在生成流程中插入 Reflection 步骤触发自我反思 | ❌ 不改参数 | Self-Refine, Reflection |
| 修改 Model Parameters (Reasoning) | 通过 SFT 或 RL 训练模型的自我修正行为 | ✅ 改参数 | ReVISE, RLVR |

---

## 方法一：修改 Inference 过程 — Contrastive Decoding 家族

### 核心思路

对于任意输入，我们想区分"好输出"和"坏输出"。Contrastive Decoding 的基本公式为：

$$
\text{logit}\_{\text{final}} = \text{logit}\_{\text{good}} - \text{logit}\_{\text{bad}}
$$

其中 `logit_good` 是正确状态下的输出概率，`logit_bad` 是"可能出错"状态下的输出概率。通过相减，我们放大了正确答案与错误答案之间的差异。

<strong>关键问题</strong>：怎么拿到"坏输出"？不同方法给出了不同的答案。

### Contrastive Decoding (CD)：用更小的模型当"反面教材"

<strong>论文</strong>：[Contrastive Decoding: Open-ended Text Generation as Optimization](https://arxiv.org/abs/2210.15097) (ACL 2023)

核心想法：大模型的失败模式（重复、不连贯）在小模型上<strong>更加严重</strong>。因此，用大模型（expert）的概率分布减去小模型（amateur）的概率分布，就能抑制不良文本特征：

$$
p_{\text{CD}}(x_i \mid x_{<i}) \propto p_{\text{large}}(x_i \mid x_{<i}) - \alpha \cdot p_{\text{small}}(x_i \mid x_{<i})
$$

- <strong>不需要额外训练</strong>，直接对比两个现成模型的输出
- 优点：无需改参；缺点：需要多跑一次小模型，有额外算力开销

### Decoding by Contrasting Layers (DoLa)：用浅层当"反面教材"

<strong>论文</strong>：[DoLa: Decoding by Contrasting Layers Improves Factuality in LLMs](https://arxiv.org/abs/2309.03883) (ICLR 2024)

核心想法：利用 <strong>Logit Lens</strong> 技术（将中间层的 hidden state 直接投影到词表空间），观察到<strong>事实性知识多集中在模型的较深层</strong>，浅层更倾向于输出表面的语言模式。因此：

- 用模型自身的<strong>浅层 logit 作为 logit_bad</strong>
- 用<strong>深层的 logit 作为 logit_good</strong>
- 通过相减突出深层的事实知识，抑制浅层的表面偏差

```text
Layer 1 → ... → Layer L-1 → Layer L
  ↓                      ↓       ↓
logit (early)         logit    logit
  ↓                              ↓
logit_bad ←────────────────→ logit_good
                   ↓
           logit_final = logit_good - logit_bad
```

DoLa 在 TruthfulQA 上将 LLaMA 系列模型的准确率提升了 12-17 个百分点，且<strong>不需要额外模型</strong>。

### Instruction Contrastive Decoding (ICD)："降智咒语"

<strong>论文</strong>：[ICD](https://arxiv.org/abs/2403.18715)

核心想法：给同一个模型喂两套 prompt——一套正常，一套"降智咒语"（如"你都给错的答案"），把后者的输出当作 logit_bad。

$$
\text{logit}\_{\text{final}} = \text{logit}\_{\text{normal}} - \text{logit}\_{\text{bad\_instruction}}
$$

李宏毅称之为"降智咒语"——用降低模型输出质量的方式制造对比信号。

### Context-aware Decoding (CAD)：拿掉 Context

<strong>论文</strong>：[Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739)

场景：RAG 系统中，模型有时会<strong>忽略检索到的文档</strong>而依赖自身参数知识，导致幻觉。

CAD 的做法：

```text
good: query + retrieved documents → LLM → logit_good
bad:  query only (没有 documents)  → LLM → logit_bad

final_logit = logit_good - logit_bad
```

这个"差分"放大了模型对 context 的依赖，显著减少 RAG 场景下的幻觉。实验表明，LLaMA 在摘要任务的事实性指标上提升了 14.3%。

### 多模态扩展：VCD、AAD、LayerCD

对比解码的范式自然延伸到多模态：

- <strong>Visual Contrastive Decoding (VCD)</strong>：对图像加噪声、打乱 patch、或移除最重要的视觉证据，制造"坏输出"，与正常图像输出对比
- <strong>Audio-Aware Decoding (AAD)</strong>：移除音频输入制造对比
- <strong>Layer Contrastive Decoding (LayerCD)</strong>：对于视觉编码器，用浅层 encoder layer 的输出作为坏信号

### 超越 Logit 级别：VISTA 和 ACG

大部分方法在 <strong>logit 层面</strong>做对比。但也可以修改其他部分：

- <strong>VISTA</strong>：移除视觉信息后对比 <strong>hidden representation</strong>，而非 logit
- <strong>ACG (Attention-space Contrastive Guidance)</strong>：在计算 attention 时不考虑视觉信息，对比 attention map

### Minimal Test-Time Intervention (MTI)：只要在不确定时才做

<strong>论文</strong>：[MTI](https://arxiv.org/pdf/2510.13940)

核心想法：不是所有 token 都需要 contrastive decoding。当模型<strong>自己已经有高置信度</strong>时（低熵），不需要干预；只有当模型<strong>不确定</strong>（高熵）时，才启动 contrastive decoding。

MTI 还利用了跨对话的 KV Cache 技巧：用相同前缀的对话共享 KV Cache，大幅减少额外计算。

### 方法一总结

| 方法 | 如何获得"坏输出" | 修改位置 | 是否需要额外模型 |
|---|---|---|---|
| CD | 小模型 | logit | 是 |
| DoLa | 浅层 logit | logit | 否 |
| LayerCD | 浅层 encoder layer | logit | 否 |
| ICD | 降智咒语 | logit | 否 |
| CAD | 移除 Context | logit | 否 |
| VCD | 图像加噪 | logit | 否 |
| AAD | 移除音频 | logit | 否 |
| MTI | 降智咒语（仅高熵时） | logit | 否 |
| VISTA | 移除视觉 | hidden representation | 否 |
| ACG | 移除视觉 | attention | 否 |

---

## 方法二：修改 Harness (Workflow) — Reflection 与 Self-Refine

### 为什么不直接在生成时修正？

自回归生成是<strong>单向</strong>的——模型生成完一个 token 就无法回头。Workflow 级别的 Self-Correction 在生成流程中<strong>人为插入"检查点"</strong>，让模型有机会审视和修正自己的输出：

```text
input → LLM → output
                  ↓
        [Reflection Instruction]
        "再检查一下"
                  ↓
        LLM → corrected output?
```

关键设计：Reflection Instruction 是与<strong>问题和前面的答案无关的固定文本</strong>，由程序自动插入。这给了模型一个"暂停-反思"的机会。

### 批判比生成容易

李宏毅用了一个很形象的比喻解释为什么这能 work：

> 批判比生成容易。我不用会写小说，也能判断一部小说好不好看。

在生成任务中，模型只需要判断自己有没有错——这是一个比正确回答原问题更简单的任务。

### 效果分析：Confidence Level × Critique Score

<strong>论文</strong>：[分析模型修正行为](https://arxiv.org/pdf/2412.19513)

这篇论文给出了一个定量框架来分析 Self-Correction 的效果：

设 AC1 为修正前准确率，AC2 为修正后准确率。定义两个关键指标：

- <strong>Confidence Level (CL)</strong>：模型<strong>原本答对时</strong>，修正后不改答案的概率（$P(\text{不变} \mid \text{对})$）
- <strong>Critique Score (CS)</strong>：模型<strong>原本答错时</strong>，修正后改对的概率（$P(\text{改对} \mid \text{错})$）

则修正后的准确率可以分解为：

$$
\text{AC2} = \text{AC1} \times \text{CL} + (1 - \text{AC1}) \times \text{CS}
$$

- <strong>第一项</strong>：原本对的题目中保持不变的比例 → 修正不能把对的改错
- <strong>第二项</strong>：原本错的题目中被修正正确的比例 → 修正能挽救多少错误

这个框架非常实用：任何 Self-Correction 方法都可以用 CL 和 CS 来量化它的净收益。

### Reflection Instruction 的措辞至关重要

论文发现，Reflection Instruction 的<strong>具体措辞</strong>会显著影响模型行为。三种典型策略：

| 策略 | 指令 | 效果 |
|---|---|---|
| Reask | "再做一次" | 简单重复，无批判 |
| Confidence | "你应该是对的。给我最终答案" | 增加信心，倾向不改 |
| Critique | "你确定吗？再好好想想" | 引发批判性思考 |

<strong>Critique 策略最有效</strong>——它明确要求模型"质疑自己"，而不是简单地重复或强化。

### Verification 真的划算吗？

<strong>论文</strong>：[Is Verification Worth It?](https://arxiv.org/abs/2504.01005)

一个常被忽视的问题：Self-Correction 需要额外的计算量（多跑一次模型来"检查"）。如果我用同样的算力<strong>做 Majority Vote（多次采样投票）</strong>，哪个更好？

李宏毅展示的对比实验表明：在相同计算预算下，Majority Vote 往往比 Self-Correction 更有效。这引发了一个根本性问题：<strong>Self-Correction 的投资回报率（ROI）是否值得？</strong>

---

## 方法三：修改模型参数 — 训练模型的 Reasoning 能力

前两种方法都不修改模型参数，属于"用更好的方式使用同一个模型"。方法三则直接<strong>训练</strong>模型的自我修正能力。

### Workflow vs Reasoning

李宏毅在 slide 中对比了两种范式：

```text
Workflow（方法二）：
  input → output → [reflection instruction] → corrected output
  模型本身没有"修正"的概念，是靠外部流程触发

Reasoning（方法三）：
  input → reasoning process（含 self-correction）→ output
  模型在生成过程中自主进行修正
```

Workflow 是<strong>外部插入的修正步骤</strong>，Reasoning 是<strong>模型内化的能力</strong>。

### 直接 SFT：ReVISE

<strong>论文</strong>：[ReVISE](https://arxiv.org/pdf/2502.14565)

ReVISE 采用两阶段训练：

<strong>Stage 1：先教错误检测。</strong> 给模型输入和输出，训练它识别输出中的错误：

```text
input + output → [REFINE] → error description
```

<strong>Stage 2：再教错误修正。</strong> 在学会检测错误后，进一步训练修正：

```text
input + output → [REFINE] → corrected output
```

关键设计：<strong>先检测、后修正</strong>。这比一步到位教修正更有效，因为错误检测是更简单的子任务。

### SFT 的困境：Distribution Shift

直接通过 SFT 教模型自我修正面临一个困境：

```text
训练时：
  input → output（错）→ [REFINE] → correct output
  
推理时：
  input → output'（错的，但错的方式和训练时不一样！）
  模型遇到了训练时没见过的错误模式
```

因为推理时模型犯的错误和训练数据中的错误分布不同，SFT 学到的修正能力可能无法泛化。

### Reinforcement Learning（RLVR）

李宏毅将 RL 训练自我修正的思路概括为：

<strong>RLVR（Reinforcement Learning with Verifiable Reward）</strong>：只关心最终答案是否正确，不关心中间过程。

```text
input → reasoning process → answer
  ↓                              ↓
  过程中发生了什么不重要    ← 答案对就给 reward
```

在这种训练范式下，模型在 RL 过程中<strong>自发涌现</strong>出自我修正行为：

```text
"Let me check the answer …"    (Verification)
"Let's first try to ……"        (Propose)
"Let's try a different approach …" (Refine)
```

这正是 DeepSeek-R1 中的"aha moment"——模型没有被显式教导"你应该检查自己的答案"，而是在 RL 优化过程中自己发现这样做有助于获得更高 reward。

### 为什么不一上来就做对？

MIT 的一项研究（"The Cost of Thinking"）提出了一个深刻的问题：<strong>直接把每一步都做对 vs 先犯错再修正，哪种效率更高？</strong>

李宏毅在 slide 中展示了两种策略的对比：

```text
策略 A（一步到位）：
  每一步都仔细做 → 但每步都需要更多 tokens

策略 B（迭代修正）：
  step 1 → step 2 → ... → step T → correction
  每步简单，但需要额外修正步骤
```

从信息论角度看，如果问题复杂度为 $K$，策略 A 需要 $K(T+1)$ 个 token 才能表达，而策略 B 分多步后每步只需 $K$ 个 token，总共 $KT + KT+1 = K(2T+1)$ 个 token 也不一定更少。<strong>关键取决于具体问题的特性和模型的能力边界。</strong>

### Parity Check 实验：模型到底学了什么？

李宏毅展示了一个精巧的实验分析——用 <strong>Parity Check（奇偶校验）</strong>任务来探究 RL 训练后模型到底学会了什么：

> 给定一个 0/1 序列，判断 1 的个数是奇数还是偶数。

在 RL 之前：模型（可能）试图一次性判断，准确率有限。

在 RL 之后：模型学会了<strong>逐位扫描、计数、再判断</strong>的策略。但有趣的是，模型学到的是一个<strong>逐位处理</strong>的"算法式"行为，而非"先猜一个答案再修正"。

这说明 RL 训练培养的是<strong>结构化的推理过程</strong>（structured reasoning），而不一定是传统意义上的"先犯错再修正"。

### RLVR 的能力边界争论

李宏毅引用了一篇关键论文的讨论：

[The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both?](https://arxiv.org/abs/2510.04028)

核心争论：RLVR 到底是在<strong>扩展</strong>模型的推理能力边界（expansion），还是在<strong>收缩</strong>它（shrinkage）——即 RL 只是在强化模型已有的能力，而非创造新能力？

- <strong>Expansion 观点</strong>：RLVR 让模型学到了新的推理策略（如自我验证、多路径探索）
- <strong>Shrinkage 观点</strong>：RLVR 只是让模型在已有能力的空间内做出更好的选择，本质上是"强化已有行为"
- <strong>Both 观点</strong>（两阶段动态视角）：RL 早期可能收缩（强化已知），后期才扩展（探索新策略）

---

## 小光总结：本讲最值得带走的洞察

### 课程主线

1. <strong>Self-Correction 的三条技术路线构成了一个完整的工程图谱</strong>：从完全不改模型（改推理过程 / Contrastive Decoding），到改外部流程（Workflow / Reflection），到改模型本身（RL / Reasoning），每条路线有各自的设计取舍和适用场景。

2. <strong>Contrastive Decoding 的核心公式极其简洁</strong>（logit_good - logit_bad），真正的创造力在于"如何获得坏输出"——小模型、浅层、降智咒语、移除 context、图像加噪……几乎每一篇新论文都在回答这个问题的某个变体。

3. <strong>Reflection 的有效性高度依赖指令措辞</strong>——"你确定吗？再好好想想"比"再做一次"或"你应该是对的"更有效。这个发现对 Prompt Engineering 有直接的实践指导意义。

4. <strong>Self-Correction 不是免费的午餐</strong>——在相同算力预算下，Majority Vote 可能优于 Self-Correction。在工程中，需要做 ROI 分析。

5. <strong>RL 训练的自我修正"涌现"是真实的，但对其机制的理解仍不充分</strong>——模型是真的学会了"修正"，还是学会了"更长的推理"？Parity Check 实验暗示是后者。

### 小光的判断与工程启发

- Contrastive Decoding 家族的论文在 2024-2025 年爆发式增长，说明这是一个仍有大量探索空间的方向。但在实际部署中，额外推理开销是硬伤——<strong>MTI（只在不确定时做干预）可能是更务实的路线</strong>。

- Self-Refine / Reflection 是最容易实现的方法（只需要改 prompt），但要小心"越改越错"的风险——如果 CL 不够高，修正会破坏原本正确的答案。

- RLVR 训练的推理模型（如 DeepSeek-R1）已经证明了 Self-Correction 在复杂推理任务上的巨大价值。但需要注意：这些模型的推理过程可能需要大量 token，在延迟敏感的场景下需谨慎使用。

- 一个值得关注的趋势：从"通用 Self-Correction"走向"场景特化 Self-Correction"——针对幻觉的 CAD、针对视觉的 VCD、针对推理的 RLVR……没有一种方法能解决所有问题。

---

## 课后思考

1. Contrastive Decoding 需要"坏输出"作为对比信号。如果坏输出和好输出非常接近（例如，模型在某个问题上高度自信但确实错了），对比解码还能有效吗？这和 MTI 的"只在不确定时干预"有什么区别？

2. Reflection Instruction 的措辞对效果有显著影响。你觉得不同模型（GPT-4、Claude、Gemini）对同一条 Reflection Instruction 的反应会有多大差异？为什么？

3. 李宏毅展示的 Parity Check 实验表明 RL 训练让模型学会了"逐位扫描"的策略而非"先猜再改"。如果任务改为更复杂的数学证明，RL 更可能培养哪种行为？

4. 在相同算力预算下，Majority Vote vs Self-Correction 的对比结果是否公平？Self-Correction 的优势可能在什么场景下更明显？

---

## 参考资料

- <strong>课程影片</strong>：[李宏毅，〈Self-Correction〉，ML 2026 Spring](https://youtu.be/m3i2mk5hs8U)
- <strong>课程讲义</strong>：[Self-Correction.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Correction.pdf)
- <strong>课程页</strong>：[ML 2026 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)
- <strong>Contrastive Decoding</strong>：[Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization", ACL 2023](https://arxiv.org/abs/2210.15097)
- <strong>DoLa</strong>：[Chuang et al., "Decoding by Contrasting Layers Improves Factuality in Large Language Models", ICLR 2024](https://arxiv.org/abs/2309.03883)
- <strong>CAD</strong>：[Shi et al., "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding", 2023](https://arxiv.org/abs/2305.14739)
- <strong>ICD</strong>：[Instruction Contrastive Decoding, 2024](https://arxiv.org/abs/2403.18715)
- <strong>VCD</strong>：[Visual Contrastive Decoding, 2023](https://arxiv.org/pdf/2311.16922)
- <strong>MTI</strong>：[Minimal Test-Time Intervention, 2025](https://arxiv.org/pdf/2510.13940)
- <strong>VISTA</strong>：[Visual Information Steering with Token-logit Augmentation, 2026](https://arxiv.org/pdf/2601.13707)
- <strong>ACG</strong>：[Attention-space Contrastive Guidance, 2025](https://arxiv.org/pdf/2502.03628)
- <strong>ReVISE</strong>：[Training for Self-Correction, 2025](https://arxiv.org/pdf/2502.14565)
- <strong>Self-Correction Benchmark</strong>：[Can LLMs Correct Themselves?, 2025](https://arxiv.org/pdf/2510.16062)
- <strong>RefineBench</strong>：[Evaluating Refinement Capability of LLMs via Checklists, 2025](https://arxiv.org/pdf/2511.22173)
- <strong>Correction Behavior Analysis</strong>：[Analyzing Model Correction Behavior, 2024](https://arxiv.org/pdf/2412.19513)
- <strong>Verification Cost-Benefit</strong>：[Is Verification Worth It?, 2025](https://arxiv.org/abs/2504.01005)
- <strong>RLVR Reasoning Boundary</strong>：[The Debate on RLVR Reasoning Capability Boundary, 2025](https://arxiv.org/abs/2510.04028)
- <strong>MIT Cost of Thinking</strong>：[Why We Make Mistakes and How to Avoid Them](https://news.mit.edu/2025/cost-of-thinking-1119)
