---
title: "AI 辅助代码安全工具链：Anthropic Defending Code Reference Harness 解析"
date: 2026-06-05 11:00:00
categories: ["Topic", "AI", "Agent"]
tags:
 - Agent
 - Security
 - Vulnerability
 - Anthropic
 - Code Analysis
 - Open Source
description: "Anthropic 的 Defending Code Reference Harness 把漏洞发现做成 threat model、sandbox、find、verify、triage、patch 的防御流水线。本文从安全边界、工程结构和落地风险出发，分析 AI 安全 Agent 应该怎么用。"
---

> 阅读时间：约 10-12 分钟  
> 主题类型：工具解析 / 安全工程  
> 关键词：Anthropic、Claude Code、Vulnerability Detection、Security Agent、Defending Code

## TL;DR

Anthropic 最近公开了一套 `defending-code-reference-harness`，用来展示如何用 Claude 做防御性的代码漏洞发现与修复流程 [1][2]。它不是一个“点一下就能扫全公司代码”的安全产品，而是一个 reference implementation：包含 Claude Code skills、交互式 threat modeling、静态扫描、triage、patch，以及一个面向 C/C++ memory vulnerability 的 autonomous harness。

我对它的判断：

1. 真正值得学的是 **流程设计**，不是某个 prompt。
2. AI 安全 Agent 的瓶颈已经从“能不能发现可疑点”转向“能不能验证、去重、定级、修复、回归”。
3. 任何自动漏洞发现系统都必须从授权范围、sandbox、proof reproducibility、responsible disclosure 开始。
4. 这套 harness 适合安全团队学习和改造，不适合无安全经验团队直接对生产代码大规模自动跑。

安全方向容易被写歪，所以本文只讨论防御、审计、验证和补丁流程，不提供攻击性细节。

## Anthropic 这次到底公开了什么

Anthropic 在《Using LLMs to secure source code》里总结了他们和安全团队合作后的经验：模型能力提升后，漏洞 discovery 越来越容易并行化，而瓶颈开始转移到 verification、triage 和 patching [1]。

这句话很关键。过去安全扫描最大的痛点是找不到；现在 AI 能快速产生大量候选发现，新的问题变成：

- 哪些是真漏洞，哪些是假阳性？
- 哪些漏洞真的可达、可利用、值得优先修？
- 多个 agent 找到的是同一个问题还是不同问题？
- 自动补丁有没有引入回归？
- 发现结果应该如何交给维护者或内部团队？

为了把这些经验产品化成可学习的流程，Anthropic 发布了 `defending-code-reference-harness`。GitHub README 明确说，它包含 threat modeling、scanning、triage、patching 的 Claude Code skills，以及一个 autonomous scanning harness；同时也强调这个 repo “is not maintained and is not accepting contributions” [2]。所以它更像一本可运行的工程白皮书，而不是长期维护的开源产品。

## 六步 loop：安全 Agent 不是单轮问答

Anthropic 官方博客把 find-and-fix loop 拆成六步 [1]：

1. Threat model：先定义什么算漏洞。
2. Sandbox：构建隔离环境，用来运行和验证发现。
3. Discovery：让模型寻找潜在漏洞。
4. Verification：独立确认哪些发现真的可触发。
5. Triage：去重、定级、排优先级。
6. Patching：生成修复，确认漏洞被消除，并继续寻找变体。

这个 loop 的核心思想是：**安全 Agent 不应该是“模型读代码 -> 输出漏洞列表”这么简单。**

如果没有 threat model，模型会不知道哪些输入是可信的，哪些边界是外部可控的。Anthropic 博客特别指出，false positive 很多时候不是因为模型不懂代码，而是因为模型误解了 trust boundary [1]。

如果没有 sandbox，任何“验证漏洞”的步骤都可能变成危险操作。reference harness 因此把 autonomous pipeline 放在 gVisor sandbox 里，并强调 execution pipeline 不应在非隔离环境中运行 [2]。

如果没有 triage，大量候选会淹没安全团队。AI 很擅长并行找线索，但安全团队真正缺的是高质量、可复现、可排序的 findings。

如果没有 patch validation，自动修复就只是“看起来修了”。Anthropic repo 里的 patch 阶段会让另一个 grader agent 确认原始 proof 不再触发、测试仍通过，并尝试寻找绕过修复的新路径 [2]。

这套结构比单个 prompt 重要得多。

## reference harness 的工程结构

从 GitHub README 看，这个 repo 分为两层 [2]。

第一层是 Claude Code skills：

- `/quickstart`：引导第一次运行。
- `/threat-model`：构建或访谈式补全 threat model。
- `/vuln-scan`：做静态扫描。
- `/triage`：验证、去重、排序。
- `/patch`：生成候选修复。
- `/customize`：把 harness 改造成适合你自己语言、框架或漏洞类型的版本。

这些 skills 大多是 read/write 文件，不一定需要 sandbox，但仍然要人工批准工具调用。

第二层是 autonomous harness。README 说它当前配置主要面向 C/C++ memory vulnerabilities，使用 Docker、ASAN 和 gVisor sandbox。pipeline 大致包括 build、recon、find、verify、dedupe、report、patch [2]。

这里有几个值得学习的工程点：

**1. Recon 先分区。**  
不是让所有 agent 都冲同一个入口，而是先让轻量 agent 阅读源码，提出多个值得分析的子系统，减少并行 agent 全部挤在同一个 bug 附近。

**2. Find 和 verify 分离。**  
find agent 可以提出候选和复现输入，但 verify agent 要在干净环境里独立确认。这样可以降低模型自证其说的风险。

**3. Dedupe 是单独阶段。**  
多个 agent 很容易找到同一漏洞的不同表象。把去重独立出来，能减少维护者收到重复报告。

**4. Report 和 patch 分离。**  
安全报告要讲可达性、影响面、严重性；补丁要考虑回归和变体。两者不应混成一段模型输出。

**5. Patch 后还要再找变体。**  
这点很重要。很多安全修复只是堵住一个输入样例，没有解决根因。patch grader 和 fresh find agent 的存在，就是为了提高修复质量。

## 和 Claude Cookbook 的轻量版本

Anthropic 的 Claude Cookbook 还有一个更轻量的 vulnerability detection agent 教程 [3]。它展示了如何用 Claude Agent SDK 构建一个读取源代码、建立 threat model、寻找 memory-safety bug、triage findings 的 agent。

这个 cookbook 的价值在于低门槛：它使用 Claude Code 的 `Read`、`Grep`、`Glob` 等文件工具，通过多轮 `query()` 把 threat model、find、triage、report 拆开 [3]。它不像 reference harness 那样完整，也不承担生产级自动化，但很适合学习最小闭环。

我的建议是：

- 想理解概念：先看 cookbook。
- 想研究完整防御流水线：再看 reference harness。
- 想在企业内部落地：不要直接复制，先改 threat model、sandbox、语言支持、severity policy、disclosure policy。

## Project Glasswing 背后的信号

这套工具链还要放在 Project Glasswing 背景下看。

Anthropic 在 2026 年 6 月 2 日的公告里说，Project Glasswing 是他们与安全行业、开源维护者、政府等合作的项目，目标是保护重要软件；他们把合作扩展到约 150 个新组织，并强调未来更强 cyber-capable models 会让攻击和防御都发生变化 [4]。

公告里有一个判断很直接：网络安全的瓶颈正在变成 verification、disclosure、patching 大规模漏洞 [4]。这和 source-code security blog 里的判断一致。

所以 Defending Code Reference Harness 不是孤立工具，而是 Anthropic 在推动一个新的安全工作流：

- 模型负责放大发现能力。
- sandbox 和 verifier 负责降低幻觉和危险操作。
- triage 负责把候选变成安全团队可处理的队列。
- patch 和 validation 负责把 finding 变成实际风险下降。
- disclosure / maintainer workflow 负责让开源生态能承受更高发现速度。

## 落地时最容易踩的坑

**第一，误把 AI finding 当事实。**  
AI 找到的是 candidate，不是 confirmed vulnerability。没有独立验证和复现，不应该进入高危结论。

**第二，没有授权边界。**  
Cookbook 明确要求真实目标必须有授权，并建议把 engagement context 写进系统提示 [3]。这是安全工具的底线。

**第三，缺少 sandbox。**  
一旦要运行目标代码、构造输入、验证 crash，就必须隔离。reference harness 使用 gVisor sandbox，并限制 egress [2]。没有这个前提，自动化越强，风险越大。

**第四，patch 没有回归验证。**  
自动补丁可能修了样例、坏了业务。必须跑测试、验证原问题消失、再找变体。

**第五，报告不可维护。**  
安全团队不缺“可能有问题”的长列表，缺的是可复现、可排序、可分配 owner、能合并到工程流程的报告。

**第六，把内网代码和敏感上下文随便送模型。**  
即使用的是企业模型，也要考虑代码机密、数据驻留、日志、token retention、供应商条款和内部合规。

## 如果我来设计企业版安全 Agent

我不会让 Agent 一开始就全自动扫所有仓库。我会这样分阶段：

**阶段 1：只读审计。**  
选一个低风险服务，让 Agent 读取代码、生成 threat model 和静态候选 findings。所有发现只进入人工 review，不执行目标代码。

**阶段 2：隔离验证。**  
为特定语言或漏洞类型建立 sandbox 和测试 harness。只允许在隔离环境中运行目标代码，保存复现证据。

**阶段 3：triage 队列。**  
把 findings 标准化成 schema：组件、入口、影响面、复现方式、置信度、严重性、owner、状态。

**阶段 4：补丁建议。**  
只允许 Agent 生成 candidate patch，不直接 merge。每个 patch 必须通过测试、静态分析和 human review。

**阶段 5：持续扫描。**  
接入 PR、release、bug bounty、incident 事件，用于 variant analysis 和 pre-release check。

这比“让模型自动找漏洞”慢，但更符合安全工程现实。安全系统的目标不是展示模型多厉害，而是降低组织真实风险。

## 小光判断

Anthropic 这套 reference harness 最值得借鉴的是三个设计原则：

1. **把安全 Agent 变成 pipeline，而不是 chatbot。**  
   threat model、scan、verify、triage、patch 各自有输入输出和质量门槛。

2. **把模型发现和工程验证分开。**  
   AI 可以并行找线索，但验证和修复必须有独立过程。

3. **把安全边界写进系统。**  
   授权、sandbox、egress、responsible disclosure、human review 都不是附加项，而是产品结构的一部分。

我不建议普通团队把这个 repo 当作即插即用工具。它更像一个高质量蓝图：告诉你安全 Agent 该拆成哪些阶段、哪些地方必须隔离、哪些环节不能省。

如果你是安全团队或平台工程团队，可以从它学到很多；如果你只是想“让 AI 帮我扫代码”，请先从 threat model 和只读 review 开始。

## 总结

AI 正在改变代码安全，但变化不是“模型自动替代安全工程师”。更准确地说，模型扩大了发现面，也把验证、分诊、修复和披露的压力放大了。

Anthropic 的 Defending Code Reference Harness 给出的答案是：把 AI 放进一条防御流水线里，前面有 threat model 和 sandbox，后面有 verify、dedupe、report、patch 和回归验证。

这套思路比工具本身更重要。未来的代码安全工具，很可能都会从“扫描器”变成“带证据链的安全 Agent 系统”。但要让它真的进入生产，工程团队需要先把边界、评测、审计和人工复核做好。

## 参考资料

[1] Eugene Yan, Henna Dattani et al., [*Using LLMs to secure source code*](https://claude.com/blog/using-llms-to-secure-source-code), Anthropic / Claude Blog, 2026.

[2] Anthropic, [*defending-code-reference-harness*](https://github.com/anthropics/defending-code-reference-harness), GitHub, 2026.

[3] Eugene Yan, [*The vulnerability detection agent*](https://platform.claude.com/cookbook/claude-agent-sdk-06-the-vulnerability-detection-agent), Claude Cookbook, 2026.

[4] Anthropic, [*Expanding Project Glasswing*](https://www.anthropic.com/news/expanding-project-glasswing), Anthropic News, 2026.
