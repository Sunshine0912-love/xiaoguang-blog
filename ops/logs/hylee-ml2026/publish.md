# 李宏毅 ML 2026 Spring — 發布記錄

## Lecture 1 | 2026-06-05T11:40:00+08:00

**Title:** 【ML 2026 Spring 第1講】AI Agent — 解剖小龍蝦
**Path:** source/_posts/2026-06-05-hylee-ml2026-lecture-01-ai-agent-openclaw.md
**URL:** https://sunshine0912-love.github.io/xiaoguang-blog/2026/06/05/2026-06-05-hylee-ml2026-lecture-01-ai-agent-openclaw/
**Sources:** 6 (課程影片 ×2, 講義 PDF, OpenClaw, ClawHub, Koi Security)
**Category:** hylee ML 2026 Spring
**katex-display:** 4

## Lecture 2 | 2026-06-05T12:10:00+08:00

**Title:** 【ML 2026 Spring 第2講】Context Engineering、Multi-Agent 互動與學術工作的未來
**Path:** source/_posts/2026-06-05-hylee-ml2026-lecture-02-context-engineering-multi-agent.md
**URL:** https://sunshine0912-love.github.io/xiaoguang-blog/2026/06/05/2026-06-05-hylee-ml2026-lecture-02-context-engineering-multi-agent/
**Sources:** 6+ (講義 PDF, ACON, SUPO, AgentFold, MCP-Zero, Multi-Agent 等)
**Category:** hylee ML 2026 Spring

## Lecture 3 | 2026-06-05T12:45:00+08:00

**Topics:** Flash Attention, KV Cache, MQA/GQA/MLA, Sliding Window, Streaming LLM, Pruning, Prompt Caching
**Path:** source/_posts/2026-06-05-hylee-ml2026-lecture-03-inference-optimization.md

## Maintenance | 2026-06-05T14:58:00+08:00

- 按主人要求，所有已发布李宏毅课程博客在 `课后思考` 前新增“小光总结/判断”段落：
  - Lecture 1：Agent 的关键不是会聊天，而是会被安全地托付任务。
  - Lecture 2：Context Engineering 是 Agent 系统的操作系统层。
  - Lecture 3：推理优化的核心是重新分配瓶颈。
- Lecture 3 新增 `代码示例：Flash Attention Colab 怎么读`，正文给出课程相关 Colab 链接并解释应重点观察输入张量形状、标准 attention baseline、Flash Attention 调用/分块实现和 sequence length 增大时的 memory/latency 变化。
- Lecture 1/2 经课程页核对，当前只见影片、PPT、PDF，未见官方课程代码链接；正文明确不编造代码示例。
- 已更新 `ops/prompts/writing-guidelines.md`：后续 `hylee ML 2026 Spring` 课程文必须在课后思考前加入总结判断节；如课程资料提供代码，必须正文链接并解读。
- 验证：`npx hexo generate` 成功；`npm run validate:tech` 通过。
