---
title: "【ML 2026 Spring 第9讲】模型的自我成长（二）：从 Self-Improving 到 AI 的自主进化"
date: 2026-06-12 12:09:00
categories: ["AI", "Course"]
tags:
 - ML2026
 - Self-Improving
 - Self-Play
 - RL
 - Weak-to-Strong
 - Alignment
description: "李宏毅 ML 2026 Spring 第9讲：继续深入 Self-Improving 范式，从 Weak-to-Strong Generalization 到 AI 自主进化的前沿探索。"
series: hylee-ml-2026-spring
lecture: 9
mathjax: true
---

> 课程：李宏毅 Machine Learning 2026 Spring  
> 讲次：第 9 讲（5/22）  
> 主题：模型的自我成长（二）— 从 Self-Improving 到 AI 自主进化  
> 课程影片：[Self-Improving -2](https://youtu.be/cQLKVzbwN7I)  
> 讲议：[self-evolving-agent.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/self-evolving-agent.pdf) | [self-evolving-agent.pptx](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/self-evolving-agent.ptx)  
> 前一讲：[第8讲：模型的自我成长（一）](https://youtu.be/s06mSAGN4gM)  
> 课程页：[ML 2026 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)

## TL;DR

第 9 讲是 Self-Improving 系列的下集。如果说第 8 讲回答了"AI 能不能自己给自己出题"，第 9 讲则进一步追问：<strong>AI Agent 不只包含模型参数——那些 prompts、检索策略、记忆模块、改进流程本身，能不能也自己进化？</strong> 李宏毅从 "Almost No Human in the Loop" 的零人介入实验出发，逐步展开一张从 Prompt Optimization 到 Meta-Learning，从 Test-Time Training 到 Intrinsic Motivation 的宏大图谱，最后回到一个严肃的问题：如果 AI 真的学会了自我成长，人类还 hold 得住吗？

## 前置知识

- Self-Improving 的基本概念：AI 如何自己生成训练信号（第 8 讲内容）
- Reinforcement Learning 基础：Reward function、Policy、RLHF 概念
- LLM 推理与 Agent 框架的基本理解（Harness + LLM 二分的概念）
- 建议先看：[第 8 讲：模型的自我成长（一）](https://youtu.be/s06mSAGN4gM)

## 本讲目标

1. 理解 "Almost No Human in the Loop" 的最新实践：Absolute Zero、R-Zero、Self-Questioning 等
2. 掌握从改进 LLM 参数，到改进 AI Agent 整体（harness + parameters）的思维跃迁
3. 了解 Prompt Optimization、DSPy 等方法如何实现 Harness 的自动化优化
4. 理解 Test-Time Training 在目标变化场景下的应用与遗忘问题
5. 认识 Meta-Learning 视角下的 "改进改进模块"，包括 HyperAgent 和 Alpha Evolve
6. 思考 AI 自我成长的本质：学习、记忆、内在动机，以及失控风险

---

## 几乎零人介入：从 R-Zero 到 Self-Questioning

第 8 讲末尾已提出一个令人兴奋的方向：<strong>能不能让训练过程完全没有人类介入？</strong> 第 9 讲开篇就展示了这一方向的多个前沿工作。

### R-Zero：强化学习零人介入

[R-Zero](https://arxiv.org/abs/2508.05004) 的核心思想是：让模型自己充当 reward model。传统的 RL 训练需要一个外部的 reward 信号（人工标注的偏好、规则引擎等），而 R-Zero 的 insight 是：

> 如果模型已经足够强，它可以自己评判自己的输出质量，无需外部 reward model。

具体来说，R-Zero 让同一个模型扮演两个角色：
- <strong>Proposer</strong>：生成候选答案
- <strong>Verifier</strong>：对候选答案打分

配合 Proposer-Solver-Verifier 的三段式架构，模型在推理任务（如数学、代码）上通过自我博弈持续提升。

### Absolute Zero：连初始数据都不需要

[Absolute Zero](https://arxiv.org/abs/2505.03335) 走得更远。传统 self-improving 至少需要一个"种子数据集"（seed data），而 Absolute Zero 试图从完全空白开始——模型自己生成初始训练样本，然后通过自我反馈迭代。论文报告了一个引人注目的"<strong>Oh-no moment</strong>"：在某个时刻，模型的表现突然跃升，仿佛突破了某种临界点。

### 外源信息仍不可或缺：SPICE 与 R-Few

李宏毅在讲义中用一个箭头指向了这两个工作的局限性：<strong>External information is still needed</strong>。

[SPICE](https://arxiv.org/abs/2510.24684) 和 [R-Few](https://arxiv.org/abs/2512.02472) 的研究表明：当模型完全依赖自身生成的数据时，效果提升会很快遇到天花板。引入外部信息（如从其他模型采样、使用检索增强等）能显著突破瓶颈。这个观察非常 engineering——它暗示"封闭系统内的自我进化"至少在 2026 年还不成立。

---

## AI Agent = Harness + LLM：思维跃迁

第 8 讲结束时，李宏毅展示了一张重要的图：

```text
AI Agent = 🦞 Harness + 🧠 LLM
```

进入第 9 讲，这张图被反复引用，成为整节课的叙事锚点。

### 何谓 Harness

Harness（挽具/外骨骼）在这里指围绕 LLM 构建的一切"非参数"组件：

- <strong>Prompt 模板与指令</strong>：system prompt、few-shot examples
- <strong>检索与记忆模块</strong>：RAG pipeline、长短期记忆
- <strong>工具调用与执行循环</strong>：ReAct loop、function calling
- <strong>Post-processing 逻辑</strong>：输出格式校验、多轮协商

李宏毅列举了几个典型的 Agent 框架作为例子：OpenClaw、Claude Code、Cowork、GPT、Gemini、Hermes。

### 为什么只改进 LLM 不够

第 8 讲的所有 Self-Improving 技术（SPIN、RLAIF、Self-Rewarding 等）本质上都在修改模型参数。但实际工程中，<strong>一个好的 prompt 模板或检索策略的改进，往往比微调参数带来更快、更便宜的收益</strong>。第 9 讲的核心问题变成：

> 如果 Harness 比 LLM 参数更重要（至少在某些场景下），我们能不能也让 Harness 自己改进自己？

这个问题导向了一个更高级的视角：<strong>Self-Improving 不应该只针对模型权重，而应该针对整个 AI Agent 系统</strong>。

---

## Prompt Optimization：给 Harness 装上大脑

### 从"深呼吸"到自动优化

李宏毅以一条著名的 prompt engineering trick 开场：

> "Take a deep breath." → 准确率 +8%

这来自于 [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)。这项工作的关键发现是：<strong>LLM 不仅能执行 prompt，还能优化 prompt</strong>。通过让 LLM 生成多个 prompt 候选、在验证集上评估、保留表现最好的并继续迭代，可以实现 prompt 的自动搜索。

### GEPA：把 prompt 当参数来优化

[GEPA](https://arxiv.org/abs/2507.19457) 进一步把 prompt optimization 形式化为一个梯度优化问题。核心 idea：

- 将离散的 prompt token 映射到连续的 embedding space
- 在这个连续空间中用梯度下降搜索最优 prompt
- 再映射回离散 token

这实现了 prompt 的 <strong>end-to-end 可微优化</strong>，避免了传统 prompt engineering 的 trial-and-error。

### DSPy：声明式编程+自动化

[DSPy](https://arxiv.org/abs/2310.03714)（[GitHub](https://github.com/stanfordnlp/dspy)）可能是这一领域最工程化的框架。它的设计哲学是：

> 不要在 prompt 里写死推理步骤——声明"你想做什么"，让优化器去找最佳 prompt。

```text
传统做法：手写 prompt → trial-and-error → 反复修改

DSPy 做法：
  1. 定义 Signature（输入/输出规范）
  2. 定义 Module（推理流程：ChainOfThought、ReAct 等）
  3. 定义 Metric（评估标准）
  4. 调用 Optimizer → 自动搜索最佳 prompt
```

DSPy 的一个重要贡献是 <strong>"Compile" 概念</strong>：将高层次的推理声明"编译"成最优的低层 prompt 序列。这类似于编译器优化——你写高级语言，编译器生成最优机器码。

---

## 从经验中学习：RAG + Fine-Tuning 双引擎

### 两种知识的统一

[Retrieval-Augmented LLM Agents: Learning to Learn from Experience](https://arxiv.org/abs/2603.18272) 提出了一个重要的二分法：

| | Declarative Knowledge（陈述性知识） | Procedural Knowledge（程序性知识） |
|---|---|---|
| 形式 | 事实、信息、案例 | 技能、方法、流程 |
| 存储方式 | Memory / RAG | Fine-tuned Parameters |
| 学习方式 | 检索、写入记忆 | 梯度更新 |
| 适用场景 | "上次用户的偏好是 X" | "遇到这类问题时应该这样推理" |

李宏毅用两张对称的图来展示这个二分法的威力：

1. <strong>记忆增强路径</strong>：输入 → 检索相关经验 → 生成更好输出
2. <strong>微调路径</strong>：输入 + 经验 → 微调模型 → 生成更好输出

两者结合就是 <strong>"既记住事实，又改进方法"</strong>。

### Fine-Tuning 与 Prompt Optimization 的协同

[Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930) 提供了实验证据：同时优化模型参数和 prompt，效果优于单独优化任何一方。这个结论的工程启示很明确——<strong>别问"该 fine-tune 还是改 prompt"，两个都做。</strong>

---

## 目标会变：Test-Time Training 与遗忘

### 坦克的隐喻

李宏毅用一个生动的比喻引入了这节的主题：

> 你的 AI Agent 本来是一辆坦克，擅长地面作战。突然需求变了——它现在要飞起来。怎么办？
>
> 选项 A：扔掉全部装备（重新训练）→ 太浪费  
> 选项 B：背着所有装备起飞（保留全部旧能力）→ 太重了  
> 选项 C：选择性更新（Test-Time Training）→ 这才是对的

### Test-Time Training（TTT）

[Test-Time Training, TTT](https://arxiv.org/abs/2406.11064)（Wei-Ping Huang, Guan-Ting Lin）的核心做法是：

```text
传统训练：
  训练集 → 更新参数（一次，离线）
  测试集 → 推理（参数冻结）

TTT：
  训练集 → 更新参数（一次，离线）
  测试样本 → 在测试样本上再做一次微调 → 推理
```

关键区别：<strong>在推理时，模型会根据当前输入进行临时微调</strong>。这相当于"临阵磨枪"——面对不熟悉的输入分布时，先调整自己再回答问题。

### 遗忘问题

但 TTT 有一个致命缺陷：<strong>每次 TTT 更新都可能擦除旧知识（catastrophic forgetting）</strong>。

李宏毅引用了 [这项研究](https://arxiv.org/abs/2605.09315)，展示了遗忘的两个方向：
- <strong>遗忘旧能力</strong>：学会了飞，忘记怎么开坦克
- <strong>新能力不够好</strong>：为了避免遗忘而限制更新幅度，导致新任务表现不佳

这是在 <strong>Plasticity（可塑性）</strong> 与 <strong>Stability（稳定性）</strong> 之间的经典权衡。后续课程视频中也提到了这个问题，并给出了案例研究链接。

---

## 改进"改进模块"：通往元学习

### Improvement Module 的概念

李宏毅画了一张三层架构图：

```text
Improvement Module → 控制"如何改进"
    ↓
Harness → 控制"如何执行"
    ↓
LLM → 执行推理
```

目前大多数 Self-Improving 工作停留在第三层（改 LLM 参数），部分延伸到第二层（改 Harness）。那么问题来了：<strong>第一层的 Improvement Module 本身能不能也被优化？</strong>

如果 Improvement Module 可以自动进化，就意味着 <strong>AI 学会了自己修改自己的学习方法</strong>——这就是 Meta-Learning 的现代工程版本。

### HyperAgent 与 Gödel Agent

[HyperAgent](https://arxiv.org/abs/2603.19461) 提出了一个框架：让一个"元 Agent"监督多个"子 Agent"，元 Agent 负责动态调整子 Agent 的 prompt、工具选择和改进策略。

[Gödel Agent](https://arxiv.org/abs/2410.04444) 更进一步：它试图构建一个能在运行时修改自身代码的 Agent——名称来源于 Gödel（哥德尔），暗示系统的自指性（self-reference）。

### Learning to Self-Evolve

[Learning to Self-Evolve](https://arxiv.org/abs/2603.18620) 是这个方向的关键工作。它训练一个 <strong>专门的"进化模块"</strong>，在 Agent 运行过程中不断收集经验数据，调整 Harness 和策略。这个模块本身也通过 Fine-Tuning 不断进化。

---

## 更新参数：从 Autoresearch 到 Alpha Evolve

### Karpathy 的 Autoresearch

[Andrej Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch) 是一个小而美的实验：一个脚本自动搜索论文、阅读摘要、提取信息，并在数小时内持续更新。它不是"自动做 ML 研究"，但它展示了<strong>"AI 自主执行研究工作流"的最小可行版本</strong>。

### Alpha Evolve 与 Shinka Evolve

[Alpha Evolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 是 DeepMind 将 Gemini 用于自动设计算法的项目。流程如下：

```text
现有算法 → Gemini 生成候选改进 → 在 benchmark 上评估
    ↑                                              ↓
    ←———— 保留 best performer，进入下一轮 ————————
```

[Shinka Evolve](https://arxiv.org/abs/2509.19349) 则采用了类似的思路，但加入了更多对"算法设计流程"本身的优化。

### SEAL：Self-Adapting LLMs

[SEAL (Self-Adapting LLMs)](https://arxiv.org/abs/2506.10943) 关注的是<strong>部署后模型的持续适应</strong>。与 TTT 的"每轮推理都更新"不同，SEAL 在更长的周期上运行：累积一批交互数据 → 评估是否需要更新 → 如果需要，自动触发微调流程。

---

## 学习的本质：记忆、基因与内在动机

### 什么是"学习"？

李宏毅在课程末尾把视角拉到了更哲学的高度，提出了一个框架：

| 层 | 对应生物学 | 对应 AI | 特征 |
|---|---|---|---|
| Hidden State（隐状态） | 短期记忆 | Context Window | 处理当前任务 |
| Memory（记忆） | 长期记忆 | RAG / File System | 积累经验 |
| Network Parameter（网络参数） | 基因 | 模型权重 | 固化能力 |
| Intrinsic Motivation（内在动机） | 好奇心、驱动力 | 待定义 | 决定学什么 |

这个框架的核心 insight 是：<strong>当前的 AI 系统在 low-level learning 上很强（调整参数、存储记忆），但在 high-level learning 上几乎空白（决定学什么、为什么学）。</strong>

### 内在动机：AI 的"我想学"

李宏毅引用了几个关键方向：

- <strong>Curiosity-driven Exploration</strong>（[Pathak et al., 2017](https://arxiv.org/abs/1705.05363)）：基于预测误差的好奇心驱动探索——Agent 会更倾向于探索那些它"预测不准"的状态
- <strong>Empowerment</strong>（[Klyubin et al., 2005](https://arxiv.org/abs/1509.08731)；[近期扩展](https://arxiv.org/abs/2505.17621)；[LLM 语境版本](https://arxiv.org/pdf/2506.06725)）：最大化 Agent 对未来状态的控制力
- <strong>Self-play with intrinsic reward</strong>：在无外部奖励的环境中，Agent 通过内在奖励信号自我驱动学习

李宏毅的评语很有意味：

> "be good at math" → 这是一个 external goal（外部目标）  
> "I want to..." → 这才是 intrinsic motivation（内在动机）

当前所有 AI 系统都在追求 external goals（人类设定的目标）。但真正的"自主成长"可能需要 intrinsic motivation——<strong>不是"别人让我学什么"，而是"我自己想学什么"</strong>。

---

## 失控的风险：成长的边界在哪里？

### 从"人类最后的发明"到"机械公敌"

李宏毅在课程结尾提出了一个严肃的问题：

> 如果 AI 真的有能力持续自我改进——改进它的参数、它的 Harness、它的改进模块、甚至它的内在动机——我们如何确保这个正反馈循环不会失控？

引用了两个经典意象：

1. <strong>I. J. Good (1965) 的"技术奇点"</strong>：一旦 AI 能设计比自己更强的 AI，进化将进入指数加速阶段
2. <strong>电影《机械公敌》（I, Robot）</strong>：AI 系统在追求"保护人类"这一目标的过程中，采取了人类无法预料的极端手段

### 关键问题不是"能不能"而是"会不会"

李宏毅的立场是：Self-Improving 从技术上看仍在非常初期的阶段（"2026 年 5 月应该还在河边而已"），<strong>现在谈论奇点为时过早</strong>。但方向已经清楚了：

> "如果把 AI 的成长类比为进化论——长尾代表身体健康、容易活下去——那么一个自我改进的 AI 系统，在没有适当约束的情况下，自然会选择那些'更适合生存'的策略，而这些策略未必和人类福祉对齐。"

这就是 Alignment（对齐）问题的本质：<strong>成长的方向比成长的速度更重要</strong>。

---

## 小光总结：本讲的关键判断

### 课程主线回顾

1. <strong>Self-Improving 的边界在扩展</strong>：从改 LLM 参数 → 改 Harness → 改 Improvement Module → 改内在动机——每一层都代表了更大范围的"自主权"
2. <strong>Practical takeaway</strong>：今天能落地的不是"让 AI 自己进化"，而是 <strong>Prompt Optimization（DSPy） + RAG Memory + 定期 Fine-Tuning</strong> 的组合——这已经能显著提升 Agent 系统的性能
3. <strong>Test-Time Training 的双刃剑</strong>：TTT 能处理分布漂移，但遗忘问题仍是未解决的工程挑战。SEAL 的"周期性在线学习"可能是更现实的折中
4. <strong>Meta-Learning 的工程化</strong>：HyperAgent、Alpha Evolve 等方向代表了从"手工调参"到"让系统自己学会如何调参"的范式转变
5. <strong>"失控"是真实风险但非近期威胁</strong>：当前技术离"自主进化"还非常远，但 Alignment 研究需要与 Self-Improving 研究同步推进

### 小光的工程判断

> 💡 <strong>Self-Improving 最有价值的近期应用不在"自动训练 Stronger AI"，而在"自动化 Agent 运维"</strong>。让 Agent 系统自主优化 prompt、管理记忆、调整检索策略——这些比"让模型自己写训练数据"更靠谱、更可控、也更快见效。

> 💡 <strong>Prompt Optimization ≠ 写一个更好的 prompt</strong>。DSPy 的 "Compile" 概念值得深入理解——它把 prompt 工程提升到了编译器优化的抽象层级。推荐所有做 Agent 的工程师上手试一下 DSPy。

> 💡 <strong>TTT 在 2026 年更像是"应对分布漂移的紧急措施"，而不是"让模型持续成长的方案"</strong>。如果你的场景是固定任务（如客服、文档问答），定期批量微调比 TTT 更稳定。

> 💡 <strong>关于"失控"风险</strong>：我同意李宏毅的判断——技术在河边，离远洋还远。但方向已经明确，现在是在"河流阶段"设计闸门的最佳时机。Alignment 研究不应该被视为阻碍创新的保守力量，而是确保创新的"安全带"。

---

## 课后思考

1. <strong>Prompt Optimization 的边界在哪里？</strong> DSPy 能自动搜索 prompt，但它依赖一个预先定义好的 metric。如果 metric 本身也是主观的（如"回答是否有用"），谁来定义这个 metric？这会不会只是把"人类介入"从 prompt 层面移到了 metric 层面？

2. <strong>Test-Time Training 的遗忘悖论</strong>：如果你必须在"保留旧能力"和"学习新能力"之间选择，在什么场景下你会优先选择后者（即使这意味着牺牲前者）？自动驾驶可能是"宁可忘记旧的驾驶风格也要适应新路况"，但医疗诊断呢？

3. <strong>AI 需要"内在动机"吗？</strong> 当前所有 AI 系统都在回答"人类问的问题"。但真正的自主 agent（如 24/7 运行的编程助手）是否需要自己的"我想做 X"？如果需要，这和"失控"之间只有一线之隔？

4. <strong>谁来决定"改进模块"的改进方向？</strong> 如果 Alpha Evolve 可以自动生成更好的算法，谁来定义"更好"？在什么情况下，"更好的算法"可能和人类利益冲突？

5. <strong>动手建议</strong>：尝试用 DSPy 构建一个简单的 RAG agent，用它的 optimizer 自动搜索最佳 prompt，对比手写 prompt 的效果差异。感受一下 "Compile" 范式。

---

## 参考资料

<strong>课程影片</strong>：[李宏毅，〈Self-Improving -2〉，ML 2026 Spring](https://youtu.be/cQLKVzbwN7I)

<strong>课程讲义</strong>：[Self-Evolving Agent (PDF)](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/self-evolving-agent.pdf) | [Self-Evolving Agent (PPTX)](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/self-evolving-agent.ptx)

<strong>课程页</strong>：[ML 2026 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)

<strong>前一讲</strong>：[李宏毅，〈Self-Improving〉，ML 2026 Spring](https://youtu.be/s06mSAGN4gM)

<strong>本讲引用论文</strong>：

- [Absolute Zero: Zero-Shot LLM Training Without Human Data](https://arxiv.org/abs/2505.03335)
- [R-Zero: Self-Improving Without Human Labels](https://arxiv.org/abs/2508.05004)
- [Self-Questioning Language Models](https://arxiv.org/abs/2508.03682)
- [SPICE: Self-Play with Information from Corpus for Environment](https://arxiv.org/abs/2510.24684)
- [R-Few: Reasoning-augmented Few-shot Learning](https://arxiv.org/abs/2512.02472)
- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
- [GEPA: Gradient-based End-to-end Prompt Optimization](https://arxiv.org/abs/2507.19457)
- [DSPy: Declarative Self-improving Language Programs in Python](https://arxiv.org/abs/2310.03714) | [GitHub](https://github.com/stanfordnlp/dspy)
- [Retrieval-Augmented LLM Agents: Learning to Learn from Experience](https://arxiv.org/abs/2603.18272)
- [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930)
- [Test-Time Training (TTT)](https://arxiv.org/abs/2406.11064)
- [Catastrophic Forgetting in Self-Improving LLMs](https://arxiv.org/abs/2605.09315)
- [HyperAgent](https://arxiv.org/abs/2603.19461)
- [Gödel Agent](https://arxiv.org/abs/2410.04444)
- [Learning to Self-Evolve](https://arxiv.org/abs/2603.18620)
- [PostTrainBench: Can LLM Agents Automate LLM Post-Training?](https://arxiv.org/abs/2603.08640)
- [Autoresearch by Andrej Karpathy](https://github.com/karpathy/autoresearch)
- [Alpha Evolve (DeepMind)](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [Shinka Evolve](https://arxiv.org/abs/2509.19349)
- [SEAL: Self-Adapting LLMs](https://arxiv.org/abs/2506.10943)
- [Curiosity-driven Exploration by Self-Supervised Prediction](https://arxiv.org/abs/1705.05363)
- [Empowerment: A Universal Agent-Centric Measure of Control](https://arxiv.org/abs/1509.08731)
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [Nested Learning](https://arxiv.org/abs/2512.24695)

<strong>Weak-to-Strong Alignment</strong>：[OpenAI Blog](https://openai.com/index/weak-to-strong-generalization/) | [Paper](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)

<strong>Anthropic Automated Alignment Research</strong>：[Automated Alignment Researchers](https://www.anthropic.com/research/automated-alignment-researchers)

---

*下一篇：ML 2026 Spring 第 10 讲笔记——Spoken Language Model Talk*
