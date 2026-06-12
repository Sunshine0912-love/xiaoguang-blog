---
title: "Hermes-Agent 19 万星现象：开源 AI Agent 框架的规模化竞争与架构范式"
date: 2026-06-12 08:00:00
categories:
 - AI
 - Agent
tags:
 - Agent
 - Open Source
 - Hermes-Agent
 - Framework
 - NousResearch
description: "NousResearch Hermes-Agent 以190K星成为GitHub增长最快的AI Agent项目。本文分析其架构设计和开源Agent框架的规模化竞争。"
topic_id: "TOPIC-20260612-02"
---

> 阅读时间：约 12 分钟
> 主题类型：产业分析 / 工程实践
> 关键词：AI Agent、Hermes-Agent、NousResearch、开源框架、Agent 架构

## TL;DR

Hermes-Agent 在本周突破 19 万 GitHub Star，成为开源 AI Agent 赛道增长最快的项目。它的崛起不只是又一个"明星项目"——它代表着一种范式迁移：Agent 框架正从"开发者玩具"进化为"个人AI基础设施"，核心能力是闭环学习（技能自生成、自改进）、跨平台持久存在（不依赖笔记本）和完全的模型/提供商自由。本文拆解 Hermes-Agent 的架构设计，对比开源 Agent 框架生态，并讨论这场规模化竞争对开发者的实际影响。

## Hermes-Agent 是什么？

Hermes-Agent 是 Nous Research 构建的自改进型 AI Agent。它不是一个简单的 Chatbot 壳，而是一套完整的个人 AI 基础设施——你可以在 $5 的 VPS 上跑它，也可以在 GPU 集群上跑，不运行时不花钱，从 Telegram 发消息时自动唤醒。

它的核心理念可以浓缩为一句话：**Agent 应该像人一样从经验中学习**。具体来说，Hermes-Agent 是"唯一内置学习闭环的 Agent"[1]：在完成复杂任务后自动生成 Skills（可复用的操作流程），在使用过程中持续改进这些 Skills，定期提醒自己将关键信息写入长期记忆，并能搜索自己的过往对话。

这套能力建立在 Nous Research 的开源哲学之上。Nous 从 Hermes 3 起就明确了立场：闭源模型缺乏灵活性和适应性，无法做到真正的个人化对齐[2]。Hermes-Agent 继承了这一理念——用户可以完全自由地选择模型提供商，从 Nous Portal 到 OpenRouter、NVIDIA NIM、小米 MiMo、智谱 GLM、Kimi/Moonshot、MiniMax，或者自己的 endpoint，切换只需 `/model` 命令，零代码改动，零锁定[1]。

## GitHub 增长：不止是数字

19 万 Star 在开源 Agent 项目中是一个什么概念？作为对比：

- AutoGPT（曾经的 GitHub 增长奇迹）目前约 17 万 Star，增长已趋缓
- LangGraph 约 1.3 万 Star
- CrewAI 约 3 万 Star
- CopilotKit 约 1.5 万 Star

Hermes-Agent 的 Star 数不仅超越了 AutoGPT——一个 2023 年引爆"自主 Agent"热潮的始祖级项目——而且增速仍然维持在每周 1 万+ Star 的水平。这意味着它不是在吃老本，而是在持续扩大领先优势。

增长背后的驱动力值得分析。2025-2026 年，AI Agent 进入了真正的"个人用户时代"：Claude Code、Gemini CLI、OpenAI Codex 等 IDE Agent 培育了开发者"把 AI 当同事"的使用习惯。当开发者习惯了在编辑器里和 AI 协作后，自然会需要一个不局限于编辑器的通用 Agent。Hermes-Agent 恰好在这个时间窗口完成了产品化——跨平台持久存在 + 零成本闲置 + 自我学习，这三个特性精准命中了从"IDE Agent"到"个人 Agent"的升级路径。

## 架构拆解：三层设计哲学

Hermes-Agent 的架构不是简单的"接个 LLM API + 循环调用"，而是一个经过深思熟虑的三层系统[3]。

### 第一层：核心 Agent 循环

入口有四个：CLI（`cli.py`）、Gateway（网关进程，对接 20 个平台）、ACP 适配器（VS Code / Zed / JetBrains 集成）、Batch Runner（批量轨迹生成）。所有入口汇聚到同一个核心——`AIAgent` 类（`run_agent.py`）。

`AIAgent` 负责一次完整的对话循环：Prompt 构建 → Provider 解析 → API 调用 → Tool 执行 → 响应返回 → 持久化。这里没有花哨的异步图调度——它是一个同步的、可预测的循环，但每个环节都做了深度优化：

- **Prompt Builder**：将系统提示分为三层：稳定层（身份/工具指导/Skills）、上下文层（上下文文件）、易变层（记忆/用户画像/时间戳）。这种分层设计使 Anthropic 的 prefix caching 能命中稳定层，大幅降低 API 成本。
- **Context Compressor**：对话上下文超出阈值时，自动总结中间轮次，保留开头和结尾的完整信息。
- **三种 API 模式**：`chat_completions`（OpenAI 格式）、`codex_responses`（OpenAI Codex）、`anthropic_messages`（Anthropic 原生）——一个 Agent 适配所有主流 API。

### 第二层：Skills 系统——Agent 的程序记忆

这是 Hermes-Agent 最独特的设计。Skills 不是"功能"，而是 **Agent 的程序记忆**——一种可复用的知识单元[4]。

Skills 遵循渐进式披露（Progressive Disclosure）的 token 效率模式：

- **Level 0**：`skills_list()` 返回名称和描述列表（约 3K tokens）
- **Level 1**：`skill_view(name)` 加载完整内容
- **Level 2**：`skill_view(name, path)` 加载特定参考文件

Agent 只在真正需要时才加载完整内容。更重要的是，Hermes-Agent 能在完成复杂任务后**自动生成 Skills**，并在后续类似场景中自动激活和**自我改进**。这形成了一个真正的学习闭环：执行 → 沉淀 → 应用 → 改进。

Skills 还支持条件激活：可以设定只在缺少某种工具时才显示（fallback），或只在某种工具可用时才显示（requires）。例如，内置的 DuckDuckGo 搜索 Skill 只在 `FIRECRAWL_API_KEY` 未设置时激活，否则使用更强大的 web_search 工具。

Skills 遵循 [agentskills.io](https://agentskills.io) 开放标准，与 Claude Code、Cursor、Gemini CLI、OpenAI Codex、GitHub Copilot、JetBrains Junie 等二十多个 Agent 生态互通[5]。这一标准的广泛采用本身就是一个信号：Agent 的"程序记忆"格式正在从各家碎片化走向统一。

### 第三层：插件与工具生态

Hermes-Agent 的插件系统有三个发现源：`~/.hermes/plugins/`（用户级）、`.hermes/plugins/`（项目级）、pip entry points。插件可以注册工具、钩子和 CLI 命令[3]。

工具系统包含 70+ 工具、28 个工具集（toolset），支持 6 种终端后端（本地、Docker、SSH、Daytona、Modal、Singularity）、5 种浏览器后端、4 种网络搜索后端，以及动态 MCP 客户端[3]。这意味着一个 Agent 实例可以在不同的隔离环境中执行命令——本机开发用 local，生产任务用 SSH，批处理任务用 Docker 容器，GPU 密集型用 Modal Serverless。

特别值得一提的是 **Subagent 机制**：Hermes-Agent 可以派生隔离的子 Agent 并行处理多个工作流，并且可以通过 RPC 调用 Python 脚本中的工具，将多步流程压缩为零上下文消耗的单次调用[1]。

## 开源 Agent 框架对比：四种范式

当前的 Agent 框架可以分为四种范式，每种对应不同的使用场景：

### 1. 全栈个人 Agent（Hermes-Agent）

定位：个人 AI 基础设施。不是库，不是平台，就是一个可以装在任何地方的 Agent 进程。核心差异化在于**闭环学习**（Skills 自生成/自改进）和**零成本持久化**（serverless 后端，不运行时不计费）。MIT 协议，安装只需一行 curl。

### 2. 前端 Agent SDK（CopilotKit）

定位：将 Agent 嵌入 Web/Mobile/Slack 应用。核心价值是 AG-UI 协议——连接 Agent 后端和前端 UI 层的标准协议，已被 Google、LangChain、AWS、Microsoft 等采用[6]。提供 React/Angular/Vue 组件、Generative UI（Agent 动态生成界面）、Human-in-the-Loop 审批。适合需要 Agent 驱动 UI 的产品团队。

### 3. 多 Agent 编排框架（CrewAI / LangGraph）

CrewAI 定位为独立于 LangChain 的轻量级 Python 框架，专注于角色扮演多 Agent 协作。它的 Crews（团队）和 Flows（工作流）两套模型可以组合使用——用 Crews 做创造性探索，用 Flows 做生产级流程控制[7]。已获得 DeepLearning.AI 官方课程推荐，超过 10 万认证开发者。

LangGraph 是底层编排框架，提供持久化执行（Durable Execution）：Agent 在失败时可以从中断点恢复，适合需要运行数小时甚至数天的长时间任务[8]。配合 LangSmith 提供全链路的可观测和调试。

### 4. 低代码 Agent 平台（AutoGPT）

AutoGPT 已从最初的"自主 GPT Agent"进化为一个'低代码 Agent 平台'：可视化的 Agent Builder、工作流管理、预置 Agent 市场。用户可以通过连接"方块"来构建自动化流程，无需写代码[9]。适合非技术用户和快速原型。

### 为什么 Hermes-Agent 赢了 Star？

一个核心原因：**它解决了"Agent 归谁"的问题**。CopilotKit 需要用户有产品需要嵌入 Agent，CrewAI 需要用户有足够技术能力写 Python 编排，AutoGPT 的平台化路径还在走。而 Hermes-Agent 开箱即用——装上就用，想换模型随时换，不跑不花钱，关机后 Telegram 消息还能唤醒。这种 zero-friction 体验 + 自我进化能力，让它的受众面远超其他框架。

## 为什么 Agent 框架在 2025-2026 年集中爆发？

这不是偶然。几个结构性因素共同推动了这一波：

**1. IDE Agent 培育了使用习惯**：Claude Code、Gemini CLI、Cursor、OpenAI Codex 让大量开发者习惯了"和 AI 做同事"。当开发者从" AI 在编辑器里帮我写代码"进化到" AI 帮我管理整个工作流"时，通用 Agent 的需求自然爆发。

**2. Skills 标准正在统一**：agentskills.io 被 Claude Code、Cursor、GitHub Copilot、VS Code、JetBrains Junie 等巨头采用[5]。这意味着一个 Skill 可以跨 Agent 使用，降低了生态锁定风险，反过来加速了 Agent 框架的采用。

**3. 模型能力进入"Agent 可用"区间**：Hermes 4.3 36B 模型在 512K 上下文下实现了与 Hermes 4 70B 相当的性能[10]，这意味着本地部署的个人 Agent 已经具备实用级别的理解能力。工具调用（Function Calling）和长上下文推理的成熟，让 Agent 不再像 2023 年的 AutoGPT 那样"聪明但不可靠"。

**4. Serverless 基础设施降低了持久化成本**：Daytona 和 Modal 的 serverless 后端使 Agent 在空闲时可以休眠，成本趋近于零。这解决了个人 Agent 最大的经济障碍——没人愿意为一个 24/7 运行的 Agent 持续掏钱。

**5. 分布式训练验证了开源 Agent 的可行性**：Nous Research 的 Psyche 网络使用 DisTrO 优化器在 Solana 区块链上实现分布式训练[10]，证明了开源团队同样可以训练生产级模型。这为 Hermes-Agent 的长期发展提供了模型能力保障。

## 开发者选择指南

面对这么多选择，开发者应该如何判断？以下是小光的实用框架：

| 你的需求 | 推荐框架 | 原因 |
|---------|---------|------|
| 个人日常效率提升，多设备使用 | **Hermes-Agent** | 零摩擦安装、跨平台持久、自我学习、完全自由选择模型 |
| 在产品中嵌入 Agent UI | **CopilotKit** | AG-UI 协议生态最广，Generative UI 成熟，前端覆盖最全 |
| 企业级多 Agent 工作流 | **CrewAI + Flows** | 角色协作 + 事件驱动流程，适合生产级复杂自动化 |
| 需要精确控制 Agent 状态和恢复 | **LangGraph** | Durable Execution 是杀手特性，配合 LangSmith 可观测 |
| 非技术人员快速自动化 | **AutoGPT Platform** | 低代码可视化构建，预置 Agent 市场 |

**如果你只能选一个开始**：装 Hermes-Agent。它不要求你学任何新概念，不绑定任何模型，不强制任何部署方式。当你从零基础到"我的 Agent 帮我管理日程、自动回复消息、定时做代码审查"时，你自然会理解哪些能力需要更专门的框架来补充。

## 小光判断

Hermes-Agent 的 19 万 Star 不只是开源社区的又一次狂欢。它标志着三个关键趋势的交叉验证：

**第一，Agent 正在从"工具的封装层"变成"个人的基础设施层"**。Hermes-Agent 不是把 LLM API 包了一层——它有学习闭环、有跨平台持久存在、有自我改进能力。这让它从"更好的脚本"质变为"数字分身"。我认为，这一层会在未来两年内逐渐成为类似操作系统的存在——你不再"打开一个 Agent 应用"，而是"我的 Agent 一直在线"。

**第二，模型提供商的中立性将成为 Agent 框架的核心竞争力**。Hermes-Agent 支持 18+ 提供商，且切换只需一条命令——这不是"功能多"，而是战略选择。当 Agent 成为个人基础设施时，用户不能接受被某个模型提供商锁定。Nous Research 作为模型发布方却保持 Agent 的提供商中立，这种"自己不做围墙"的姿态在商业上罕见但在架构上正确。

**第三，闭环学习才是 Agent 的真正分水岭**。大部分 Agent 框架本质上是"编排层"——把 LLM 调用、工具调用、流程控制组织起来。但 Hermes-Agent 的 Skills 系统让 Agent 可以"从经验中学习"。当前版本的学习还是提示词层面的——Agent 把经验写成 SKILL.md，下次自动载入。但这条路走下去，就是 Agent 真正的"记忆与成长"。我预测 2026-2027 年，框架之间的竞争将从"谁的工具更多"转向"谁的 Agent 学得更快"。

不过也需要冷静：19 万 Star ≠ 19 万日活用户。GitHub Star 反映的是关注度和好奇心，不是实际使用率。Hermes-Agent 能否把 Star 优势转化为开发者生态的护城河，取决于 Skills 市场的繁荣程度、社区插件的丰富度，以及是否能保持当前的产品迭代速度。

## 总结

1. Hermes-Agent 以 19 万 Star 登顶开源 Agent 项目，核心能力是**闭环学习**（Skills 自生成/自改进）+ **跨平台持久存在** + **完全的模型自由**。
2. 架构采用三层设计：同步 Agent 循环 → Skills 渐进式披露系统 → 插件/工具生态，70+ 工具 + 28 工具集 + 6 种终端后端。
3. 开源 Agent 框架分化为四种范式：全栈个人 Agent（Hermes-Agent）、前端 Agent SDK（CopilotKit）、多 Agent 编排（CrewAI/LangGraph）、低代码平台（AutoGPT）。选择取决于你的需求是"个人使用"还是"产品嵌入"。
4. 2025-2026 年的集中爆发源于五个结构性因素：IDE Agent 习惯培育、Skills 标准统一、模型能力成熟、Serverless 降本、开源训练验证。
5. Agent 框架的下一阶段竞争将从"工具数量"转向"学习速度"。闭环学习能力将成为分水岭。

## 参考资料

[1] **Hermes-Agent GitHub**：[NousResearch, "Hermes-Agent: The agent that grows with you", GitHub, 2025-2026](https://github.com/NousResearch/Hermes-Agent)

[2] **Freedom at the Frontier**：[Aria, "Freedom at the Frontier: Hermes 3", Nous Research Blog, 2024](https://nousresearch.com/freedom-at-the-frontier-hermes-3/)

[3] **Hermes-Agent Architecture**：[NousResearch, "Architecture", Hermes Agent Documentation, 2025-2026](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)

[4] **Hermes-Agent Skills System**：[NousResearch, "Skills System", Hermes Agent Documentation, 2025-2026](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)

[5] **Agent Skills 开放标准**：[agentskills.io, "Agent Skills Overview", agentskills.io, 2025-2026](https://agentskills.io)

[6] **CopilotKit GitHub**：[CopilotKit, "The Frontend Stack for Agents & Generative UI", GitHub, 2025-2026](https://github.com/CopilotKit/CopilotKit)

[7] **CrewAI GitHub**：[crewAIInc, "Framework for orchestrating role-playing AI agents", GitHub, 2025-2026](https://github.com/crewAIInc/crewAI)

[8] **LangGraph GitHub**：[LangChain AI, "Build resilient agents", GitHub, 2025-2026](https://github.com/langchain-ai/langgraph)

[9] **AutoGPT GitHub**：[Significant Gravitas, "AutoGPT: The vision of accessible AI for everyone", GitHub, 2025-2026](https://github.com/Significant-Gravitas/AutoGPT)

[10] **Hermes 4.3 发布**：[Nous Research, "Introducing Hermes 4.3: Local Intelligence Globally Trained", Nous Research Blog, 2025-2026](https://nousresearch.com/introducing-hermes-4-3/)

[11] **Nous Research 博客**：[Nous Research, "Blog Archive", Nous Research Blog, 2024-2026](https://nousresearch.com/blog/)

[12] **Hermes-Agent Memory 系统**：[NousResearch, "Persistent Memory", Hermes Agent Documentation, 2025-2026](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)
