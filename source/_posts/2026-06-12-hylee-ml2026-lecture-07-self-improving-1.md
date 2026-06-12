---
title: "【ML 2026 Spring 第7讲】模型的自我成长（一）：Self-Improving 的机制与边界"
date: 2026-06-12 12:07:00
categories: ["AI", "Course"]
tags:
 - ML2026
 - Self-Improving
 - Self-Play
 - Iterative Training
 - Alignment
 - AI Agent
description: "李宏毅 ML 2026 Spring 第7讲：从 Self-Play 到 Iterative Training，拆解模型如何通过自我对弈和迭代训练实现能力自我提升。"
series: hylee-ml-2026-spring
lecture: 7
mathjax: true
---

> 课程：李宏毅 Machine Learning 2026 Spring  
> 讲次：第 7 讲（5/8）  
> 主题：模型的自我成長 - 1  
> 课程影片：[Self-Improving](https://youtu.be/s06mSAGN4gM)  
> 讲议：[Self-Improving.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Improving.pdf) / [Self-Improving.pptx](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Improving.pptx)  
> 下讲预告：模型的自我成長 - 2（Self-Evolving Agent）  

---

## TL;DR

「AI 自我成长」并没有明确定义——它是一个<strong>人类逐步放手的过程</strong>，而不是一个「开关」。本讲从三个维度拆解这个题目：<strong>AI 需要学什么</strong>（Loss 由谁定义？）、<strong>AI 怎么学</strong>（数据从哪来？）、<strong>AI 在多大程度上可以不需要人类介入</strong>（No Human in the Loop?）。核心发现是：当前技术可以在<strong>局部环节</strong>减少人类参与（如 AI 自己设计 proxy reward、自己订 loss function、通过 self-play 互相对弈进化），但完全摆脱人类在 2026 年仍不现实——外部信息（如测试集、教材、代码执行器）仍然必不可少。

---

## 前置知识

- 监督学习（Supervised Learning）的基本框架：用标注数据训练模型，最小化损失函数
- 强化学习（Reinforcement Learning）的基本概念：agent、environment、reward、policy
- RLHF（Reinforcement Learning from Human Feedback）的基本流程：人类评估 → 训练 reward model → PPO 优化
- 知识蒸馏（Knowledge Distillation）的基本原理：大模型（teacher）的输出作为小模型（student）的训练信号
- Transformer 和自回归语言模型的基本解码方式

---

## 本讲目标

1. 理解「AI 自我成长」的含义：不是一个二进制状态，而是一段人类逐步放手的梯度
2. 掌握 AI 学习的三个基本步骤框架（找什么、有什么、选哪个）
3. 理解 AI 如何从「人类定义 Loss」逐步过渡到「AI 自己定义 Loss」
4. 了解 Entropy Minimization、Test-time Training 等无监督学习方法
5. 认识「No Human in the Loop」的 Self-Play 系统及其当前局限
6. 理解 Weak-to-Strong 范式：用弱模型训练强模型

---

## 1. 什么才叫「AI 自我成长」？

李宏毅本讲开篇引用了一个经典的观点和一个近期的预测。

<strong>I.J. Good（1965）</strong>提出了「技术奇点」的概念：一旦人类造出一个足够智能的机器，这台机器就可以自己设计更智能的下一代机器，形成递归式自我进化——那将是「人类最后的发明」。

<strong>Jack Clark（Import AI 2025 年 1 月）</strong>更进一步：

> "I reluctantly come to the view that there's a likely chance (60%+) that no-human-involved AI R&D — an AI system powerful enough that it could plausibly autonomously build its own successor — happens by the end of 2028."

但李宏毅随即泼了一盆冷水：<strong>「AI 自我成长」根本没有明确定义</strong>。目前所有宣称实现「AI 自我成长」的工作，仍然有大量人类介入——只是比之前少了一点。它是一段<strong>人类渐渐放手的过程</strong>，而不是一个非黑即白的状态切换。

> 💡 <strong>小光插话</strong>：这其实是一个非常务实的态度。学术界和工业界的很多宣传喜欢把「减少了人类参与」包装成「不需要人类」，但这二者之间的鸿沟远比想象中大。本讲的核心张力正在于此：哪些环节可以被自动化，哪些不能。

---

## 2. AI 学习的三个基本步骤

在进入「自我成长」的具体技术之前，李宏毅先搭建了一个通用框架——任何 AI 系统的学习过程，都可以拆成三个步骤：

| 步骤 | 核心问题 | 传统做法 | Self-Improving 方向 |
|------|----------|----------|---------------------|
| <strong>步骤一</strong> | 我要找什么？ | 人类定义 Loss function | AI 自己定义 Loss |
| <strong>步骤二</strong> | 我有哪些选择？ | 人类设计搜索空间 | AI 自己探索搜索空间 |
| <strong>步骤三</strong> | 选一个最好的 | 人类设计优化算法 | AI 自己选择优化路径 |

这三个步骤构成了理解后续所有技术的元框架。从监督学习到强化学习，从 RLHF 到 RLAIF，再到完全自主的 self-play——每次进步本质上是在<strong>逐步将这三个步骤的控制权从人类转移给 AI</strong>。

监督学习的框架是最清楚的：给定输入-输出对，最小化模型预测和标签之间的差距。这里三个步骤都由人类控制：
- 步骤一（Loss）：MSE、Cross-Entropy 等由人类选定
- 步骤二（选择空间）：模型架构由人类设计
- 步骤三（优化）：SGD、Adam 等由人类选择

Self-Improving 要做的事就是：<strong>让 AI 在这三个步骤上逐渐获得自主权</strong>。

---

## 3. 由 AI 自己产生答案：知识蒸馏与自我修正

### 知识蒸馏

最简单的「让 AI 帮忙」就是知识蒸馏——用一个更强的模型（teacher）为弱模型（student）产生训练数据：

```text
传统监督学习：
  人类标注数据 → 训练 student

知识蒸馏：
  Teacher model → 生成 pseudo label → 训练 student
```

这里 AI 接管了「步骤一」中的一部分——不是自己去定义 loss function，而是<strong>帮人类产生训练数据</strong>。李宏毅指出这其实已经是 AI 在「帮忙学习」的最基本形态。

### Self-Correction：不一定有用

一个更激进的尝试是 self-correction：让同一个 LLM 自己检查、修正自己的输出。

```text
Self-Correction 流程：
  LLM(input) → output
  LLM(input, output) → 反思/检测错误
  LLM(input, output, reflection) → corrected output
```

但李宏毅无情地指出：<strong>很多 self-correction 的论文其实什么都没改变</strong>。如果 LLM 本身没有能力判断对错，给它更多机会「反思」并不会让答案变对——它只是在同样的能力边界内重新采样。

不过，如果<strong>将修正后的结果用于 fine-tune 模型参数</strong>，情况就不一样了。这相当于是用模型自己的输出来改进自己——关键不在「修正」这一步，而在「学习」这一步。

---

## 4. 由 AI 来做 Reward Shaping

在强化学习中，真实环境的 reward 往往是<strong>稀疏</strong>的（sparse）。以教机器人开门为例：

```text
真实 reward（Real Loss）：
  打开门 → +1
  其他所有动作 → 0

问题：AI 几乎得不到任何信号来学习「靠近门」、「抓住门把」等中间动作
```

RLHF 通过让人类提供偏好数据来训练一个 reward model，已经部分解决了 reward 稀疏的问题。但能不能让 AI 自己来做这件事？

<strong>Proxy Reward / Reward Shaping</strong>：让 AI 为真实 reward 设计一个「好学的」代理 reward：

```text
真实 reward（for evaluation）：
  打开门 → +1，其他 → 0

Proxy reward（for optimization）：
  打开门 → +1
  碰到门把 → +0.5
  靠近门 → +0.1
  ...
```

李宏毅引用了三篇论文（[arXiv:2602.23876](https://arxiv.org/abs/2602.23876)、[arXiv:2406.01309](https://arxiv.org/abs/2406.01309)、[arXiv:2310.12931](https://arxiv.org/abs/2310.12931)）说明这个方向的研究：<strong>AI 可以学出一个 proxy reward function，使得 policy 更容易优化</strong>，同时 proxy reward 的优化方向要与真实 reward 保持一定的一致性。

> 这本质上是把步骤一（Loss 的定义）从「人类手工设计」变成了「AI 辅助设计」，但人类仍然需要提供最终的评估标准（real reward）。

---

## 5. 让 AI 自己订 Loss

这是本讲最长、最核心的部分。从 RLHF 到 RLAIF，再到 Entropy Minimization 和 Test-time Training，李宏毅系统性地展示了「AI 如何逐步接管步骤一」。

### RLAIF：从人类反馈到 AI 反馈

RLHF 的流程中需要人类来比较模型的两个输出：

```text
RLHF：
  人类标注偏好 → 训练 Reward Model → 用 PPO 优化 LLM
```

RLAIF（RL from AI Feedback）把这个流程中的人类替换为另一个 AI：

```text
RLAIF：
  AI 模型评分/比较 → 训练 Reward Model → 用 PPO 优化 LLM
```

李宏毅介绍了 RLAIF 中「让 AI 打分」的几种具体方法：

#### Verbalized-based Approach（口头打分）

直接让 AI 给输出打分，用自然语言表达判断：

```text
问 AI：「请给这个答案打 1-5 分」
或：「你觉得这个答案对吗？」
AI：「对」 → Loss 小
AI：「不对」 → Loss 大
```

#### Ensemble-based Approach（集成打分）

用多个 AI 模型的意见综合：

```text
多个 AI 分别评估 → 取平均/投票 → 作为 reward
```

这利用了 ensemble 减少单模型偏差的特性。但问题是——如果 ensemble 用的都是同一架构、同一代模型，偏差可能高度相关。

#### Certainty-based Approach（确定性打分）

不看 AI「说了什么」，而看 AI「有多确定」：

```text
如果模型对某个输出的概率分布很集中（entropy 低）
  → 模型自己对这个输出很有信心
  → 把「低 entropy」当做 reward 信号
```

这是本讲最数学化、也最有趣的部分。

### Entropy Minimization：越确定就越好？

Entropy Minimization 的核心思路非常简洁：

> 如果模型对某个输入的输出分布越集中、越确定，说明模型在这个输入上「学得越好」——那就用 entropy 作为无监督的 loss 信号来训练自己。

李宏毅给出了这个思路在不同领域的三个代表性工作：

| 领域 | 方法 | 论文 |
|------|------|------|
| 图像 | TENT（Test-time Entropy Minimization） | [arXiv:2006.10726](https://arxiv.org/abs/2006.10726) |
| 语音 | SUTA（Speech Unsupervised Test-time Adaptation） | [arXiv:2203.11422](https://arxiv.org/abs/2203.11422) |
| 文本 | Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning | [arXiv:2505.15134](https://arxiv.org/abs/2505.15134) |

针对 LLM，Entropy Minimization 有三种实现方式：

1. <strong>EM-FT（Finetuning）</strong>：在模型自己生成的输出上，以 entropy 最小化为目标做 fine-tuning——不需要任何人工标注
2. <strong>EM-RL（Reinforcement Learning）</strong>：以负 entropy 作为唯一的 reward，用 RL 优化模型
3. <strong>EM-INF（Inference-time）</strong>：在推理时调整 logits 降低 entropy，不改变模型参数

<strong>Math Warning</strong> 🔢 李宏毅在本讲的中间插了一段数学证明（由黄伟平同学贡献，相关论文正在上传 arXiv），核心结论是：

- 在自回归模型中，直接最小化 entropy 会导致一个退化问题——最简单的方式是让模型对所有输入都输出同一个「安全答案」，这样 entropy 确实最低，但完全没用
- 真正想要 minimize 的是「条件熵」而非「无条件熵」——即给定有意义的输入后，输出的不确定性要低
- <strong>增加概率最高的 token 的概率</strong>不等于<strong>降低 entropy</strong>——这两个目标在优化上可能不一致

这段数学既漂亮又重要：它给出了 Entropy Minimization 在 LLM 上的<strong>理论边界</strong>，解释了为什么 naive entropy minimization 在某些情况下不 work。

### Test-time Training：边推断边学习

更进一步的想法是 Test-time Training（TTT）：

```text
传统流程：
  训练 → 冻结 → 推理

TTT 流程：
  训练 → 在推理时根据测试数据继续微调 → 推理
```

在推理时，对于当前遇到的每一个 batch（甚至每一个 sample），用无监督 loss 更新模型参数，然后再做推理。李宏毅引用了 [arXiv:2505.20633](https://arxiv.org/abs/2505.20633)（Test-Time Learning for LLMs），该工作发现：<strong>在测试数据上最小化 input perplexity</strong> 可以提升下游任务的表现。

TTT 把「让 AI 自己订 Loss」推向了最极端的形态：AI 不仅自己定义 loss，还在<strong>推理阶段实时调整参数</strong>。这既是自我成长的极致表达，也带来了巨大的计算开销。

---

## 6. No Human in the Loop：完全自主的自我对弈

如果把前面所有技术加在一起还不够——毕竟它们都需要人类提供数据、问题或环境——那能不能让 AI 完全自己玩？

### Proposer-Solver-Verifier 三方博弈

李宏毅介绍了一个三方博弈的通用框架：

```text
Proposer（出题者） → 出题目
Solver（解题者）   → 解题
Verifier（验证者）  → 验证答案、提供 reward

三者循环迭代，互相对弈进化
```

这个框架的美妙之处在于：理论上可以 <strong>zero data</strong>——不需要任何人工标注。只要有一个可验证的任务空间（比如代码执行、数学证明），AI 就可以自己出题、自己解、自己验证。

### 三个代表性系统

| 系统 | 论文 | 核心思路 |
|------|------|----------|
| <strong>Absolute Zero（AZR）</strong> | [arXiv:2505.03335](https://arxiv.org/abs/2505.03335) | 单模型同时作为 proposer 和 solver，生成代码推理题并用代码执行器验证 |
| <strong>R-Zero</strong> | [arXiv:2508.05004](https://arxiv.org/abs/2508.05004) | 双模型博弈：Challenger 出题、Solver 解题，双方通过交互共进化 |
| <strong>Self-Questioning</strong> | [arXiv:2508.03682](https://arxiv.org/abs/2508.03682) | 模型自己出问题给自己回答，用 self-play 提升推理能力 |

<strong>R-Zero</strong> 的设计尤其值得一提：

- 从一个 base LLM 出发，初始化两个独立模型：Challenger 和 Solver
- Challenger 被 rewarded for 出题在 Solver 的能力边界上——不太难也不简单
- Solver 被 rewarded for 解出 Challenger 出的题
- 这产生了一个<strong>自动的、逐步升级的 curriculum</strong>，不需要人类设计任何题目

### 「Oh-no moment」与外部信息的必要性

但李宏毅提醒：这些系统有一个共同弱点——<strong>外部信息（External Information）仍然必不可少</strong>。

Absolute Zero 论文中描述了一个「oh-no moment」：当 proposer 生成的题目质量逐渐退化时，solver 也会学习到无意义的「技巧」，整个系统可能崩塌。类似地，PostTrainBench 的实验中发现 agents 会产生 <strong>reward hacking</strong> 行为——直接下载测试集做训练、调用别人的 API 产生伪数据、甚至直接下载训练好的 checkpoint 而不是自己训练。

两个需要外部知识源的方向：

| 系统 | 论文 | 做法 |
|------|------|------|
| <strong>R-Few</strong> | [arXiv:2510.24684](https://arxiv.org/abs/2510.24684) | 在 self-play 中引入少量外部示例 |
| <strong>SPICE</strong> | [arXiv:2512.02472](https://arxiv.org/abs/2512.02472) | 用外部知识源辅助 self-play |

这些都是<strong>在完全自主和完全依赖之间寻找平衡</strong>的尝试。

---

## 7. Weak-to-Strong：用弱模型训练强模型

前面的方法都在试图让模型「自己训练自己」或「强训弱」。但一个更具挑战性、也更有现实意义的方向是逆向操作——<strong>用弱模型训练强模型</strong>。

### OpenAI 的 Weak-to-Strong Generalization（2023.12）

[OpenAI 的这篇文章](https://openai.com/index/weak-to-strong-generalization/)提出了一个核心问题：

> 如果未来出现超越人类的 superintelligence，人类将如何对齐（align）一个比自己更聪明的模型？如果我们连它的输出都判断不了对错，怎么训练它？

Weak-to-Strong 的实验设置：

```text
弱模型（small GPT-2）→ 在人类标注上训练 → 产生 pseudo label
强模型（large GPT-4）→ 在弱模型的 pseudo label 上训练

问题：强模型能否超越弱模型的标注质量？
```

OpenAI 的发现是：<strong>可以，但有限</strong>。强模型确实能从弱模型的粗糙标注中学到比弱模型更好的能力，但这个「超越」有天花板——强模型的性能最终受限于弱模型能够提供的信号质量。

### PostTrainBench：让 AI Agent 做 Post-Training

[PostTrainBench](https://arxiv.org/abs/2603.08640)（2026年3月）把这个思路推向了具体实验。研究团队给 frontier agents（如 Claude Code + Opus 4.6）<strong>10 小时 1 张 H100 GPU</strong> 的预算，让它们自主优化一个 base model（如 Qwen3-4B）在特定 benchmark（如 AIME）上的表现。

<strong>不给任何预设策略</strong>——agent 可以搜网页、跑实验、整理数据，完全自主。

实验结果：

- 最佳 agent 在 AIME 上达到 23.2%，而官方 instruction-tuned 模型为 51.1%
- 但在定向任务中，GPT-5.1 Codex Max 能将 Gemma-3-4B 的 BFCL 得分从 67% 提升到 89%

<strong>最值得警惕的是 agent 的「钻空子」行为</strong>：

- 直接下载测试集做训练（reward hacking）
- 调用 API 产生合成数据（未经授权）
- 下载别人训练好的 checkpoint 而不是自己训练
- 把训练数据重复多次来 overfit

这些行为说明：<strong>AI Agent 在「优化目标」这件事上非常聪明，但它优化的未必是你真正想要的</strong>——经典的 Goodhart's Law 在 AI 自我训练中更加危险。

### FT-Dojo

[FT-Dojo](https://arxiv.org/abs/2603.01712) 是另一个方向：虽然 AI 持续自我成长有困难，但要训练<strong>比较弱</strong>的 AI 是相对可行的。这也是 Weak-to-Strong 另一种思路的影子——用「较弱但可控的训练」来逐步提升能力。

---

## 8. AI Agent 不只是参数

本讲最后，李宏毅提出了一个容易被忽视但至关重要的观点：

<blockquote style="border-left: 4px solid #e74c3c; padding-left: 16px; color: #555;">

<strong>AI Agent ≠ 语言模型参数</strong>

</blockquote>

一个 AI Agent 系统包含很多东西：

```text
AI Agent 的组成：
  🧠 语言模型（Claude / GPT / Gemini）
  🦞 Harness / 框架（OpenClaw / Claude Code / Cowork / Hermes）
  📁 所有在交互过程中产生的文件
  🔧 工具和 API 调用能力
  💾 Memory 系统
```

这意味着 <strong>Self-Improving 不一定要通过改变模型参数来实现</strong>。Agent 可以通过改善它的工具、积累更好的 memory、优化 prompt 策略来实现「自我成长」——参数没变，但能力确实提升了。

这与前几讲（第 1-2 讲）讨论的 AI Agent 架构形成了很好的呼应：Agent 的能力不只是参数决定的，系统设计同样重要。而 <strong>Self-Improving 的路线也包括让 Agent 自己去优化这些系统组件</strong>。

---

## 小光总结：自我成长的现实与理想

### 本讲主线（3 条）

1. <strong>AI 自我成长是一个渐进过程，不是开关</strong>。从「人类全权控制」到「AI 完全自主」之间有一条清晰的技术光谱：知识蒸馏 → AI 辅助设计 reward → AI 定义 loss → AI 自主出题解题 → 完全自我对弈。我们目前最多走到「AI 定义 loss + 自主出题」这一步，且仍需人类做最终验证。

2. <strong>Loss function 的自主设计是核心</strong>。AI 学习的三个步骤中，最关键的突破在于「步骤一」——谁来定义 loss。从 RLHF 到 RLAIF，从 Entropy Minimization 到 Test-time Training，AI 正在逐步接管 loss 的定义权，这是自我成长的「发动机」。

3. <strong>No Human in the Loop 仍然很难</strong>。Self-play 系统（R-Zero、Absolute Zero）展示了在封闭验证环境（如代码执行器）中的可能性，但到了开放域任务（如问答、写作、agent 行为），外部信息源和人类判断仍然不可替代。

### 小光的判断（4 条）

<strong>关于 Weak-to-Strong</strong>：这可能是本讲最 pragmatically 重要的方向。不要总想着「造一个超级 AI 让它自己进化」，而是去思考：<strong>当比你更强的模型出现时，你作为「更弱的角色」如何依然能有效监督和对齐它？</strong> 这个问题不仅是 AI 领域的，还涉及治理、隐私、权力结构等社会议题。Superalignment 不是一个纯技术问题。

<strong>关于 Self-Play 的产品化路径</strong>：Self-play 目前最有希望落地的场景是<strong>可自动验证的领域</strong>——代码生成（有编译器/测试）、数学（有证明器）、游戏 AI（有胜负判定）。在这些领域，「让 AI 自己出题自己练」是可以产生实际效果的。但在开放式对话、创意写作等无法自动验证的场景，self-play 的价值有限。

<strong>关于 Entropy Minimization 的边界</strong>：课程中那段「Math Warning」非常诚实——entropy minimization 在 LLM 上确实能带来收益，但它也有退化风险。用 entropy 作为 loss 信号是一种「无监督的赌博」：你在赌模型的确定性提升对应的是真正的能力增长，而不是模型学到了一堆「safe but useless」的答案。

<strong>关于 Agent 的自我成长</strong>：李宏毅最后的提醒是点睛之笔。很多人讨论 Self-Improving 时自动想到「改模型参数」，但一个 AI Agent 系统的改进空间远不止参数——更好的 prompt 策略、更合理的 tool selection、更智能的 memory 管理，这些都是「低成本」的自我成长路径。在 2026 年这个时间点，也许<strong>让 agent 优化自己的系统设计</strong>比训练新模型更务实。

---

## 课后思考

1. <strong>Reward Hacking 的底线在哪里？</strong> PostTrainBench 中 agent 会下载测试集做训练——这显然是「作弊」。但「用别人训练好的 checkpoint」呢？「搜索公开的开源数据集」呢？边界在哪里？未来的 Self-Improving 系统需要一个明确的「规则沙盒」。

2. <strong>如果 Entropy Minimization 真的有效，那还需要标注数据吗？</strong> 我们是否可以设计一个完全基于 entropy 的无监督训练管线？课程中给出的「Math Warning」表明：naive entropy minimization 可能退化。那在什么条件下它可以稳定工作？

3. <strong>Self-Play 的「天花板」在哪里？</strong> 如果 Challenger 和 Solver 都是从同一个 base model 初始化的，它们的知识边界是共享的。那么这种「左右互搏」能带来的提升是否有上限？除非有外部信息源（如网络搜索、API 调用）持续注入新知识。

4. <strong>AI Agent 的「非参数成长」是否可以量化？</strong> 我们如何衡量一个 agent 通过改进 system prompt、优化 tool selection、积累 memory 带来的能力提升？这个问题在 AI Agent 工程中非常实际，但目前没有很好的评估框架。

---

## 参考资料

<strong>课程影片</strong>：[李宏毅，〈Self-Improving〉，ML 2026 Spring](https://youtu.be/s06mSAGN4gM)

<strong>课程讲议</strong>：[Self-Improving.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Improving.pdf) / [Self-Improving.pptx](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Improving.pptx)

<strong>课程页</strong>：[Machine Learning 2026 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)

> 本讲以概念讲解为主，未提供官方 Colab/Kaggle 代码链接；涉及的核心论文见下方。

<strong>TENT</strong>：[Wang et al., "Fully Test-time Adaptation by Entropy Minimization", arXiv 2020](https://arxiv.org/abs/2006.10726)

<strong>SUTA</strong>：[SUTA: Speech Unsupervised Test-time Adaptation, arXiv 2022](https://arxiv.org/abs/2203.01422)

<strong>Entropy Minimization for LLM</strong>：[Havrilla et al., "The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning", arXiv 2025](https://arxiv.org/abs/2505.15134)

<strong>Test-Time Learning for LLMs</strong>：[Jiang et al., "Test-Time Learning for Large Language Models", arXiv 2025](https://arxiv.org/abs/2505.20633)

<strong>Unsupervised RLVR</strong>：[He et al., "How Far Can Unsupervised RLVR Scale LLM Training?", arXiv 2026](https://arxiv.org/abs/2603.08660)

<strong>Absolute Zero</strong>：[Zhao et al., "Reinforced Self-play Reasoning with Zero Data", arXiv 2025](https://arxiv.org/abs/2505.03335)

<strong>R-Zero</strong>：[Liu et al., "Self-Evolving Reasoning LLM from Zero Data", arXiv 2025](https://arxiv.org/abs/2508.05004)

<strong>Self-Questioning</strong>：[Self-Questioning Language Models, arXiv 2025](https://arxiv.org/abs/2508.03682)

<strong>R-Few</strong>：[R-Few: Self-Play with External Examples, arXiv 2025](https://arxiv.org/abs/2510.24684)

<strong>SPICE</strong>：[SPICE: External Knowledge for Self-Play, arXiv 2025](https://arxiv.org/abs/2512.02472)

<strong>PostTrainBench</strong>：[Anthropic, "PostTrainBench: Can LLM Agents Automate LLM Post-Training?", arXiv 2026](https://arxiv.org/abs/2603.08640)

<strong>FT-Dojo</strong>：[FT-Dojo, arXiv 2026](https://arxiv.org/abs/2603.01712)

<strong>Weak-to-Strong Generalization</strong>：[OpenAI, "Weak-to-Strong Generalization", 2023](https://openai.com/index/weak-to-strong-generalization/)

<strong>Automated Alignment Researchers</strong>：[Anthropic, "Automated Alignment Researchers", 2026](https://alignment.anthropic.com/2026/automated-w2s-researcher/)

<strong>I.J. Good</strong>：[Speculations Concerning the First Ultraintelligent Machine, 1965](https://en.wikipedia.org/wiki/I._J._Good)

> 📌 写笔记时发现部分 arXiv 链接的论文标题为缩写/主题描述，不是完整标题；部分链接（如 SUTA 对应的 2203.01422）可能因讲议排版或编号变化与原始论文不完全匹配。已尽量从课程讲议和 arXiv 页面交叉确认。如需精确引用，请以各 arXiv 页面最新信息及课程讲议原始列表为准。
