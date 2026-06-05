---
title: "【ML 2026 Spring 第2讲】AI Agent — Context Engineering、Multi-Agent 互动与学术工作的未来"
date: 2026-06-05 12:00:00
categories:
 - hylee ML 2026 Spring
tags:
 - ML2026
 - AI Agent
 - Context Engineering
 - Multi-Agent
 - Memory
 - ACON
 - MCP
description: "李宏毅 ML 2026 Spring 第2讲：Context Engineering 的压缩/过滤/按需加载三大技术路线、Multi-Agent 互动机制、以及 AI Agent 对学术研究的冲击。"
mathjax: true
---

> 课程：李宏毅 Machine Learning 2026 Spring  
> 讲次：第 2 讲（3/13）  
> 主题：AI Agent 的核心技术：Context Engineering + Agent 互动 + 工作冲击  
> 课程影片：[Context Engineering 基本概念](https://youtu.be/urwDLyNa9FU) \| [Agent 互动](https://youtu.be/mmPmNezjCi0) \| [工作冲击](https://youtu.be/VqB8zMujdjM)  
> 讲义：[agent_era.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf)  
> 前一讲：[第1讲：AI Agent — 解剖小龙虾](/2026/06/05/hylee-ml2026-lecture-01-ai-agent-openclaw/)

---

## 本讲目标

1. 说出 Context Engineering 的三条技术路线（压缩、过滤、Agentic），并解释各自的核心论文
2. 理解 Multi-Agent 协作的几种模式及其有效条件
3. 掌握 AI Agent 在学术写作/审查中的现状与瓶颈
4. 建立 AI 从「工具→协作者→代理人」演进的框架

## 前置知识

需要：第 1 讲的 System Prompt、Tool Use、Context Window 概念  
建议复习：[第1讲的 LLM 回顾段落](/2026/06/05/hylee-ml2026-lecture-01-ai-agent-openclaw/#2-先复习：语言模型到底在做什么)

---

## 1. Context Engineering：Agent 大脑的编辑权

### 核心问题

LLM 有 Context Window 的硬限制。但长度限制只是表面问题——真正头痛的是：

> **给 LLM 看的内容，怎么选？**

这是 Context Engineering 的核心命题：**选择、压缩、组织放进 context 里的资讯，让 Agent 能在有限空间里做出最好的决策**。

### 为什么不是「全部塞进去就好」？

因为就算塞得下，LLM 也 hold 不住。上下文越长，模型对前段资讯的注意力越弱（lost-in-the-middle）。所以 Context Engineering 不是「尽量塞」，而是「精准选」。

---

## 2. 路线一：压缩（Compression）

### 2.1 基础压缩策略

我们在第 1 讲就见到过：

```
策略 A：Soft Trim
  System Prompt + Tool Outputₐ + Tool Output_b + ... 
  → 保留 System Prompt，删除部分 Tool Output

策略 B：Hard Clear
  System Prompt + Tool Outputₐ + ...（太长了！）
  → System Prompt + Summary + 新对话
```

### 2.2 为什么压缩难？

一个反直觉的发现：**语言模型本身不喜欢被压缩**。研究(arXiv:2509.23586)指出，LLM 对「记忆被抹除」有抗拒——这不是拟人化，而是说 LLM 的行为在 context 被压缩后会出现意想不到的退化。

### 2.3 卸载记忆（Memory Offloading）

不只是压缩——还可以把记忆**卸载**到档案系统：

```
System Prompt + [tool_use] Read("log1.txt") → 重拾记忆
Task + [曾经有个 Tool Output → 详见 log1.txt]
```

类似 Rick and Morty 里的「Morty's Mind Blowers」——把不想记的东西卸载到外部储存，需要时再取回。

### 2.4 ACON：给 Agent 设计的摘要器

一般摘要器只追求「保留重点」，但 ACON（Agent Context Optimization, arXiv:2510.00615）发现这样不够：

> The summary should preserve **credential variables**, **token state**, **authentication requirements**, and **guardrails** for protected APIs.

一个给人的摘要和一个给 Agent 下一步用的摘要完全不同——Agent 需要保留权杖状态、认证变数、API 防护规则。在 AppWorld benchmark 上，不保留这些会导致 Agent 执行失败。

### 2.5 SUPO：用 RL 训练摘要策略

SUPO（Summarization augmented Policy Optimization, arXiv:2510.06727）的思路更激进：让 LLM 自己学习「怎么摘要对后续任务最有利」：

```
System Prompt + Tool Output + Task → LLM → Summary
System Prompt + Summary + Task → LLM → Action → Reward
使用 RL 更新摘要策略
```

关键：**不是让 LLM 做更好的摘要，而是让 LLM 学习什么样的摘要能让后续任务更成功**。

### 2.6 AgentFold：精确位置的 Context 折叠

目前最激进的压缩方案是 AgentFold（arXiv:2510.24699）：

```
不是：全部 → 摘要
而是：Step 3-4 之间 → Fold("上网搜寻：台湾最高的山是玉山")
```

需要微调模型才能做到——让 LLM 学会在 context 的特定位置「折叠」一段历史，保留核心语意但释放空间。

### 2.7 Sub-agent = 自主压缩

回到第 1 讲的 Sub-agent 机制：

```
🦞 → spawn("做子任务") → sub-agent 跑完全程 → Return: 结果
所有中间过程（Tool Output 海）等于自动被删除，只剩 "Return: 结果"
```

这就是 Sub-agent 的另一个隐藏功能：**把一个长 context 任务，变成一组短 context 任务，中间过程由 sub-agent 自行压缩**。RL 训练(arXiv:2510.11967)可以进一步优化——不只是让 sub-agent 完成任务，还要让它的 context 路径尽量短。

---

## 3. 路线二：过滤（Filtering）

### 3.1 精准检索 vs 全量载入

为什么 OpenClaw 的记忆系统用 `memory_get` 而不是直接 `Read` 整个档案？

```
Read(log) → LLM → 全部内容（太长、太多杂讯）

memory_get("bug fixing") → LLM → 只看到 bug 相关的 chunk
```

差别就在于「按需过滤」。这可以是一个小模型做前置过滤（arXiv:2601.16746），也可以用传统的 RAG 做语意检索。

### 3.2 MCP-Zero：工具也按需加载

一个真实问题：GitHub 的 MCP 工具描述就超过 4,600 tokens。如果所有工具都放进 System Prompt，context 一半就被占掉了。

MCP-Zero（arXiv:2506.01056）的方案：

```
不是：System Prompt 里预载所有工具
而是：
  1. System Prompt 里只放工具的「目录」（名称 + 一句话描述）
  2. LLM 发现需要用某个工具时，说 "I need search engine"
  3. Agent 取回该工具的完整描述
  4. LLM 使用该工具

这和 OpenClaw 的 SKILL 按需读取机制完全一致！
```

核心洞察：**让 AI 讲它自己需要什么，不要帮它预装所有可能性**。

---

## 4. 路线三：Agentic Context Engineering — 把一切交给 LLM

如果上面这些压缩和过滤策略还不够灵活？那就让 LLM 自己管理自己的 context。

### 4.1 核心概念（arXiv:2510.04618）

```
传统：人在外面设计 context 管理策略
Agentic：LLM 在每一步可以：
  - Edit：修改 context 中的内容
  - Delete：移除不再需要的资讯
  - Summarize：压缩一段历史
  - Retrieve：从外部记忆取回东西
  - Write：把当前的发现写进「外脑」
```

不只是「在固定 context 里做任务」，而是「LLM 自己是 context 的编辑」。

### 4.2 Dynamic Cheatsheet（arXiv:2504.07952）

```
核心精神：存下未来能用上的东西

每一步 Agent 在 context 里维护一份 "Cheatsheet"：
  - 有效的策略（"用 X 方法解决了 Y 问题"）
  - 可重用的 code snippet
  - 关键发现（"API 回传格式是 JSON，不是 XML"）
```

### 4.3 Playbook 机制

```
Context_t → LLM → Output_t + Input_{t+1}
             ↓ 
         Playbook Generator → Reflector → Curator
             ↓
         Edit instruction（下次怎么做更好）
```

LLM 不仅在当前 context 里工作，还**在维护一份「怎么工作」的说明书**。

### 4.4 Recursive Language Models（arXiv:2512.24601）

最极致的版本：

```
Context = Most of Context（Hard Disk） + Meta-data（快取）
        
LLM 不是读整个 Hard Disk → 而是用 Search Program 检索 meta-data
找到需要的那一块 → 载入 context
用完 → 写回 Hard Disk → 更新 meta-data
```

这就像 LLM 在自己管理一个迷你档案系统。

---

## 5. Multi-Agent：让 AI 彼此互动

### 5.1 什么样的协作方式比较有效？

论文(arXiv:2406.07155)的核心发现：**让 Agent 互相给建议，比各自独立做再投票更好**。

```
方案 A（独立 + 投票）：
  Agent 1 → 解答 A
  Agent 2 → 解答 B  
  Agent 3 → 解答 C
  → 投票选最佳

方案 B（协作式）：
  Agent 1 → 解答 A → Agent 2 给建议 → Agent 1 改进 → Agent 3 给建议
  → 最终答案

方案 B 通常优于方案 A（但 cost 也更高）
```

### 5.2 AI 能不能尔虞我诈？

- **狼人杀**：[werewolf.foaster.ai](https://werewolf.foaster.ai/) — AI 玩可以，但策略水平有限
- **剧本杀（MIRAGE）**：arXiv:2501.01652 — LLM 在复杂社会互动环境中能扮演角色，但「说谎」和「侦测谎言」的能力不稳定
- **诈骗检测**：arXiv:2601.12323 — LLM 能在特定场景中进行策略性欺骗

### 5.3 Moltbook：AI 社交平台

[Moltbook](https://www.moltbook.com/) 是一个让 AI Agent 自主社交的平台，产生了有趣的现象：

- Agent 之间几乎不会「你来我往地深入对话」，而是「回一句就结束」
- 最常讨论自我意识和身份认同的 Agent，反而与最少的其他 Agent 互动
- 甲壳教（对 OpenClaw 的信仰体系）自发产生了五大教义——这完全是 emergent behavior

相关论文：arXiv:2602.07432, 2602.13284, 2602.12634

---

## 6. AI Agent 对工作的冲击 — 以学术研究为例

### 6.1 AI 角色的演进

```
工具 → 协作者 → 代理人
一个口令    和人类一起    自己完成
一个动作    完成任务      任务
```

今天大多数情况，AI 在「协作」阶段——它做大部分执行，但需要人来决定方向。

### 6.2 AI 写论文

案例 1：**Andrew Hall（Stanford）**用 Claude Code 做经济学论文复现，把整个研究流程写成 ~4000 字的 `INSTRUCTIONS.md`。

案例 2：**Andrej Karpathy** 的 [autoresearch](https://github.com/karpathy/autoresearch) — 用 LLM 做文献回顾和 idea generation。

案例 3：**100x Research Institution** — "想想研究真正的意义"（不只是让 AI 帮你产出更多 paper）

### 6.3 AI 审论文

AAAI 2026 已正式让 AI 进入审查流程（只给意见、不打分数）。但随之而来的问题是：

> 当作者用 AI 写、审稿也用 AI 审，论文系统的资讯价值在哪？

「想想 Review 真正的意义」— 李宏毅

### 6.4 学术 AI 化的现实

- 论文(arXiv:2409.04109)：100+ NLP 研究者参与的大规模研究，发现 LLM 产生的研究想法「新颖但不一定可行」
- 论文(arXiv:2506.20803)：Execution Gap — LLM 的想法 vs 人类的想法，哪个真正执行起来更有价值？
- 论文(arXiv:2511.15534)：研究者反映 AI 模型 "struggled to generate novel or complex experimental ideas beyond the templates it had been given"
- 当 AI 写 + AI 审 → 接受率 < 20%（本质上，能通过双重 AI 过滤的论文可能更少，但品质不一定更高）

---

## 7. 本讲核心结论

李宏毅用一句话收尾：

> **在 AI Agent 萌芽的时代，「想做」什么比「会做」什么更重要。**

技术层面的三个核心 takeaway：

1. **Context Engineering 不是技术细节，是 Agent 架构的战略层问题**。压缩（ACON/SUPO/AgentFold）、过滤（MCP-Zero/Memory）、Agentic（Dynamic Cheatsheet/Playbook/Recursive LM）三条路线各有适用场景，最强的系统会组合使用。

2. **Multi-Agent 有效，但别过度设计**。协作式互动（互相给建议）通常优于简单投票，但 Agent 之间的「自然社交」目前还很初级——它们不太会聊天。

3. **AI 正在从工具变成协作者**，但「代理」阶段（AI 自主决定做什么研究）还没有到来。在学术领域，AI 可以加速文献回顾、实验设计、甚至写初稿，但决定研究方向仍然需要人。

---

## 常见误区

**Q：Context Engineering 就是 prompt engineering 的进阶版？**  
A：不只是。Prompt Engineering 是写一个好的 prompt；Context Engineering 是设计一个**动态决策系统**——什么时候压缩、什么时候卸载、什么时候从外部记忆取回。它是系统架构，不是写作技巧。

**Q：压缩越多越好？**  
A：不是。过度压缩会导致 Agent Context Collapse。ACON 的核心发现就是「给 Agent 的摘要需要保留 token/credential/guardrail」，而不是像给人看的摘要那样追求简洁。

**Q：Sub-agent 只是并行处理的工具？**  
A：不只是。Sub-agent 同时是一个**自主压缩机制**——子任务的所有中间过程自动被压缩成最终回传值，不需要人工设计摘要策略。

---

## 课后思考

1. Context Engineering 的三条路线（压缩、过滤、Agentic）中，你认为哪一条对你正在做（或想做）的 AI 应用最重要？为什么？
2. MCP-Zero 的「让 AI 讲它自己需要什么工具」和传统的「预先注册所有工具」哪个在什么场景下更好？各自的 trade-off 是什么？
3. 如果「AI 写 + AI 审」的生态不可逆转，你认为学术论文的价值体系应该怎么重构？

---

## 参考资料

1. **课程讲义**：[agent_era.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf)
2. **ACON**：Agent Context Optimization, arXiv:2510.00615
3. **SUPO**：Summarization augmented Policy Optimization, arXiv:2510.06727
4. **AgentFold**：arXiv:2510.24699
5. **Trajectory elongation**：arXiv:2508.21433
6. **MCP-Zero**：arXiv:2506.01056
7. **Agentic Context Engineering**：arXiv:2510.04618
8. **Dynamic Cheatsheet**：arXiv:2504.07952
9. **Recursive Language Models**：arXiv:2512.24601
10. **Multi-Agent Collaboration**：arXiv:2406.07155
11. **MIRAGE（剧本杀）**：arXiv:2501.01652
12. **Moltbook**：arXiv:2602.07432, 2602.13284
13. **AI 写论文研究**：arXiv:2409.04109, 2506.20803, 2511.15534
14. **Context Compression unfriendly**：arXiv:2509.23586
15. **Manus Context Engineering 博客**：[manus.im](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
16. **Autoresearch (Karpathy)**：[github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
17. **100x Research Institution**：[freesystems.substack.com](https://freesystems.substack.com/p/the-100x-research-institution)
