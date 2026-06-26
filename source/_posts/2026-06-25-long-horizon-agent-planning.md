---
title: "长程 Agent 的规划机制：ReAct、Tree Search、Verifier 和 Memory 如何组合？"
date: 2026-06-25 11:00:00
mathjax: true
categories:
 - AI
 - Agent
tags:
 - Agent Planning
 - ReAct
 - Tree Search
 - Verifier
 - Memory
description: "从状态空间和搜索视角拆解长程 Agent：ReAct 负责边想边做，Tree Search 负责探索，Verifier 负责评估，Memory 负责复用经验。"
topic_id: "TECH-20260625-02"
---

> 阅读时间：约 12-16 分钟  
> 主题类型：TECH 技术点讲解 / 系统机制  
> 关键词：ReAct、Tree of Thoughts、LATS、Verifier、Agent Memory

## TL;DR

长程 Agent 的核心不是“让模型多想几步”，而是把任务执行写成一个可搜索、可验证、可恢复的状态机。ReAct 让模型在每一步交替生成思考和动作 [1]；Tree of Thoughts / LATS 把候选行动扩展成搜索树 [2][4]；Verifier 给轨迹或状态打分；Memory 把失败、成功和中间产物存起来，供后续任务复用 [3]。

一句话：**ReAct 是执行循环，Tree Search 是探索策略，Verifier 是价值函数，Memory 是跨回合经验。**

## 问题定义

把一个 Agent 任务抽象成马尔可夫决策过程的简化版本：

$$s_t = (g, o_{\le t}, a_{<t}, m_t)$$

其中 $g$ 是目标，$o_{\le t}$ 是到当前为止的环境观察，$a_{<t}$ 是历史动作，$m_t$ 是可检索记忆。Agent 在状态 $s_t$ 下选择动作 $a_t$，环境返回新观察 $o_{t+1}$，直到任务成功、失败或预算耗尽。

长程任务难在三点：

1. **动作空间大**：搜索、写代码、点网页、调用工具，每一步都有大量候选。
2. **反馈延迟**：第 3 步的坏选择可能到第 20 步才暴露。
3. **上下文成本高**：完整轨迹越来越长，模型既容易忘前文，也容易被无关细节污染。

因此，一个可用的长程 Agent 不能只靠单次 prompt。它需要分层机制。

## ReAct：把推理和动作交织起来

ReAct 的贡献是把 reasoning trace 和 action 放在同一条轨迹里 [1]：

```text
Thought: 我需要先查清楚目标页面结构
Action: open_browser(url)
Observation: 页面包含登录按钮和搜索框
Thought: 下一步应先登录，否则搜索结果不可见
Action: click("login")
```

它解决的是“纯 CoT 不接触环境、纯 action 没有解释”的问题。推理让模型维护局部计划，动作让模型从环境拿新信息。

但 ReAct 的弱点也明显：它通常是一条贪心轨迹。每一步只选一个动作，错了再补救。对于长任务，这种局部最优容易累积成全局失败。

## Tree Search：从一条轨迹变成多条候选

Tree of Thoughts 指出，语言模型推理不必只沿着 token 从左到右生成，也可以把“想法”作为搜索节点，生成多个候选再评估 [2]。在 Agent 场景，节点可以是状态 $s_t$，边是动作 $a_t$。

一个简化搜索目标是：

$$a_t^\* = \arg\max_{a \in A(s_t)} V(f(s_t,a))$$

其中 $A(s_t)$ 是候选动作集合，$f$ 是环境转移或模拟转移，$V$ 是状态价值评估器。

LATS 进一步把 Monte Carlo Tree Search 引入语言 Agent，把推理、动作和规划统一到树搜索框架中 [4]。它的启发是：不要让模型一次押注一个动作，而要让它在候选轨迹之间比较。

## Verifier：给轨迹定价

Tree Search 需要价值函数。对 Agent 来说，Verifier 可以有三类：

- **规则型 verifier**：测试是否通过、网页是否出现目标元素、文件是否存在。
- **模型型 verifier**：用另一个模型判断当前状态是否接近目标。
- **人类或外部反馈**：review、单元测试、运行日志、用户确认。

形式化地，可以把轨迹 $\tau=(s_0,a_0,o_1,\ldots,s_T)$ 的分数写成：

$$R(\tau)=\lambda_s R_{\text{success}}+\lambda_c R_{\text{cost}}+\lambda_r R_{\text{risk}}+\lambda_q R_{\text{quality}}$$

这里 $R_{\text{success}}$ 衡量任务是否完成，$R_{\text{cost}}$ 惩罚 token 和工具成本，$R_{\text{risk}}$ 惩罚越权或危险动作，$R_{\text{quality}}$ 衡量输出质量。关键不是公式复杂，而是把“做完”拆成可审计的多维目标。

## Memory：不是把所有东西塞进上下文

Reflexion 提出让 Agent 把失败反馈转成自然语言反思，并放入 episodic memory，在后续尝试中复用 [3]。这比“把完整历史塞进 prompt”更像工程系统。

长程 Agent 至少需要四种记忆：

| 记忆类型 | 内容 | 用途 |
|---|---|---|
| 工作记忆 | 当前目标、最近观察、未完成子任务 | 保持短期状态 |
| 轨迹记忆 | 动作、观察、错误、检查点 | 回放和恢复 |
| 经验记忆 | 成功策略、失败教训、反思 | 跨任务复用 |
| 工具记忆 | 工具约束、参数模板、常见错误 | 降低调用失败率 |

Memory 的核心不是“无限上下文”，而是检索：在状态 $s_t$ 下，找出最相关的经验 $M(s_t)$，而不是暴力带上所有历史。

## 一个组合式 Agent Loop

下面是一个可落地的骨架：

```python
state = init_state(goal)
memory = retrieve_memory(goal)

while budget_left():
    candidates = propose_actions(state, memory)      # ReAct-style thoughts + actions
    expanded = []
    for action in candidates:
        next_state = simulate_or_execute(state, action)
        score = verify(next_state, goal)              # rules/tests/model verifier
        expanded.append((score, action, next_state))

    best = select_by_search_policy(expanded)          # beam/MCTS/best-first
    state = best.next_state
    log_trajectory(state, best.action, best.score)

    if is_success(state):
        store_success_memory(state)
        break
    if is_stuck(state):
        reflection = reflect_on_failure(state)
        store_failure_memory(reflection)
        state = rollback_to_checkpoint(state)
```

这个 loop 的重点是：执行、评估、记忆、回滚都显式存在，而不是藏在一个超长 prompt 里。

## 工程边界

组合这些机制并不保证 Agent 可靠。它带来新的成本：

- Tree Search 会成倍增加模型调用和工具调用。
- Verifier 如果不可靠，搜索会优化错误目标。
- Memory 如果不做清洗，会把旧错误带进新任务。
- 回滚如果没有环境快照，只能“语言上后悔”，不能真正恢复状态。

所以实际系统要按风险分层：低风险任务可以多探索，高风险任务应少自动执行、多 verifier、多人工确认。

## 小光判断

未来长程 Agent 的竞争，不会只看底座模型推理分数，而会看“规划运行时”的质量。一个优秀 Agent 系统应该能回答：

- 为什么选这个动作？
- 还有哪些候选没选？
- 这一步的 verifier 是什么？
- 失败后能回到哪里？
- 哪些经验会被写入长期记忆？

如果这些问题没有答案，所谓长程 Agent 很可能只是一个更贵的 ReAct loop。

## 总结

- ReAct 解决“边想边做”，但单轨迹容易贪心失败。
- Tree Search 让 Agent 比较多条候选轨迹，但成本更高。
- Verifier 是长程任务的价值函数，决定搜索方向。
- Memory 应该是可检索、可清洗的经验库，不是无限上下文。
- 真正可靠的 Agent loop 必须显式支持检查点、回放和失败恢复。

## 参考资料

[1] Shunyu Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023  
[2] Shunyu Yao et al., [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601), NeurIPS 2023  
[3] Noah Shinn et al., [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366), NeurIPS 2023  
[4] Andy Zhou et al., [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406), ICML 2024  
[5] Princeton NLP, [tree-of-thought-llm official implementation](https://github.com/princeton-nlp/tree-of-thought-llm), GitHub  
[6] Noah Shinn, [Reflexion official implementation](https://github.com/noahshinn/reflexion), GitHub
