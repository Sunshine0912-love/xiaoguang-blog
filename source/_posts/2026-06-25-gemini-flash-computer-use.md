---
title: "Gemini 3.5 Flash 的 computer use 与工具调用：模型会操作电脑意味着什么？"
date: 2026-06-25 10:15:00
categories:
 - AI
 - Agent
tags:
 - Gemini
 - Computer Use
 - Tool Calling
 - Agent Safety
 - Browser Automation
description: "从 Gemini 3.5 Flash 内置 computer use 出发，分析 GUI 操作型 Agent 为什么是产品化关键一步，以及它带来的环境反馈、安全和评测问题。"
topic_id: "TOPIC-20260625-03"
---

> 阅读时间：约 8-12 分钟  
> 主题类型：工程解读  
> 关键词：Gemini 3.5 Flash、computer use、工具调用、Agent、安全边界

## TL;DR

Google 把 computer use 作为 Gemini 3.5 Flash 的内置工具发布，是一个重要信号：Agent 不再只是在 API 之间传 JSON，而是开始直接面对人类软件世界里的按钮、表单、网页、弹窗和失败反馈 [1]。

这件事的关键不在“模型能点鼠标了”，而在于产品形态发生变化：过去软件要给模型暴露工具接口，未来模型也要能理解没有 API 的图形界面。它让 Agent 的覆盖范围变大，同时也把可靠性、安全授权、环境隔离和轨迹审计推到了产品核心。

## 从 function calling 到 computer use

传统工具调用通常是结构化的：模型选择一个函数，填参数，系统执行，再把结果返回给模型。这个模式适合搜索、数据库、日历、代码执行、地图等 API 明确的场景。

computer use 面对的是另一类世界：没有为 Agent 设计的 UI。模型需要看屏幕、理解元素、决定点击或输入、等待页面变化，再根据环境反馈继续行动。Google 官方介绍中提到，Gemini 3.5 Flash 将 computer use 集成为内置工具，并强调它与 Search、Maps grounding、function calling 等能力形成互补 [1]。

这种能力把 Agent loop 变成更接近下面的形式：

```text
观察屏幕 -> 解释状态 -> 选择操作 -> 执行点击/输入/滚动 -> 读取反馈 -> 更新计划
```

这和 API tool calling 最大的不同是：**环境反馈更丰富，也更脏**。网页加载失败、按钮遮挡、登录过期、弹窗、验证码、布局变化、误点，都可能让任务偏离计划。

## 为什么 GUI 操作是 Agent 产品化关键一步

真实企业和个人工作流里，大量软件没有优雅 API。即便有 API，权限申请、字段含义、版本兼容、内部系统边界也会增加集成成本。GUI 操作型 Agent 的价值在于，它可以先覆盖“人能做但没有 API 的任务”。

典型场景包括：

- 在网页后台填表、导出报表、整理订单。
- 在浏览器里完成多站点信息收集。
- 操作企业内部系统中的旧页面。
- 协助测试网页流程和复现 UI bug。
- 作为个人自动化助手执行低频、非标准任务。

这不是说 GUI Agent 会替代 API。相反，成熟系统里最可靠的路径仍然是 API。computer use 更像一个补位层：当没有结构化工具、工具成本太高、任务是临时性的，它让 Agent 能直接进入现有软件环境。

## 技术难点：环境状态不是文本

如果只看语言模型，任务状态通常是一段上下文。但 computer use 的状态包含屏幕像素、DOM、可点击区域、光标位置、滚动位置、窗口焦点和历史动作。模型要做的不是生成答案，而是维护一个可执行计划。

这里有三个难点。

**第一，观察不稳定。** 同一个网页在不同分辨率、语言、登录状态下可能完全不同。Agent 不能只记“点击右上角按钮”，它需要理解页面语义和当前状态。

**第二，动作有副作用。** API 调用通常可以设计成可验证、可回滚；GUI 点击可能直接提交表单、删除内容或触发支付流程。因此 computer use 必须有权限分层和人工确认。

**第三，长任务会累积误差。** 一次误点可能并不致命，但连续 20 步后，状态漂移会让模型仍以为自己在原计划轨道上。长程任务需要轨迹记录、检查点和失败恢复。

## 安全边界：能操作电脑不等于能随便操作电脑

computer use 最容易被误解为“给模型一台电脑，让它自己干活”。真正可用的产品一定要更克制。

我认为至少需要四层边界：

1. **环境隔离**：优先在浏览器沙箱、远程桌面或专用 profile 中执行，不直接接触用户主环境。
2. **权限分级**：读取、填写、提交、购买、删除、发送消息应分不同权限。
3. **人工确认点**：高风险动作前必须暂停，例如付款、发邮件、删除数据、提交正式申请。
4. **轨迹审计**：记录观察、推理摘要、动作、结果，方便回放和定位错误。

这也是为什么 computer use 会把 Agent 评测从“最终答案对不对”推进到“过程是否可控”。一个任务最终成功，但中间误点了敏感页面，也不应该算完全成功。

## 和 Gemini 3.5 Pro 延期的关系

Business Insider 报道称 Gemini 3.5 Pro 推迟到 2026 年 7 月，原因与继续收集早期测试反馈、调整模型有关，并提到长程任务、Agent 能力和 token 消耗等问题 [2]。这和 computer use 的产品方向并不矛盾，反而说明下一代 Agent 模型的难点正在从“会不会回答”转向“能不能稳定完成长任务”。

Flash 内置 computer use 更像是把能力先推到高频、低延迟、成本敏感的场景；Pro 则需要在更复杂、更长程、更高风险任务上证明稳定性。两者共同指向一个趋势：模型能力不再只用单轮 benchmark 表达，而要在真实软件环境里接受检验。

## 小光判断

computer use 的长期价值不在“万能自动化”，而在它逼迫 AI 产品补上三块基础设施：安全运行时、任务轨迹评测、可恢复的执行状态。

未来两类 Agent 会并存：

- **API-first Agent**：可靠、可审计、适合生产系统。
- **GUI-capable Agent**：覆盖范围广、集成成本低、适合临时和长尾任务。

真正成熟的产品不会二选一，而会让模型优先使用结构化工具；当工具不存在时，再退到 GUI 操作；遇到高风险动作，再交给人确认。

## 总结

- Gemini 3.5 Flash 内置 computer use，说明 GUI 操作正在成为 Agent 产品能力的一部分。
- computer use 扩大了 Agent 可覆盖的软件范围，但也放大了状态漂移和副作用风险。
- 可靠产品必须有沙箱、权限、确认点和轨迹审计。
- Agent 竞争的下一阶段，不只是模型分数，而是真实软件环境中的稳定执行能力。

## 参考资料

[1] Google Blog, [Introducing computer use in Gemini 3.5 Flash](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-computer-use-gemini-3-5-flash/), 2026  
[2] Business Insider, [Google delays Gemini 3.5 Pro launch to July as it tweaks its new frontier AI model](https://www.businessinsider.com/google-3-5-pro-july-release-tokens-ai-agents-model-2026-6), 2026  
[3] Google DeepMind, [Gemini 3.5 Flash](https://deepmind.google/models/gemini/flash/), official model page  
[4] Google AI for Developers, [Function calling](https://ai.google.dev/gemini-api/docs/function-calling), official docs  
[5] Anthropic, [Computer use](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use), official docs  
[6] OpenAI, [Computer-Using Agent](https://openai.com/index/computer-using-agent/), 2025
