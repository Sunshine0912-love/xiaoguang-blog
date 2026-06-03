# 主人确认选题后的自动写作发布流程

你不能依赖聊天上下文记忆来判断主人确认了哪个选题。必须从本地索引和日志恢复选题状态。

硬性安全规则：

- 只有主人明确回复 `确认 TOPIC-YYYYMMDD-XX` 才算确认选题。
- “今天发一篇”“按你推荐的写”“希望完成发布”“cron 是否执行”等泛化表达都不是选题确认。
- 不允许根据“首选推荐”自动推断用户确认了首选。
- 不允许把“用户希望今天发布一篇文章”解释为“确认 TOPIC-YYYYMMDD-01”。
- 如果用户想在当天已发布一篇后继续发布第二篇，必须额外收到 `继续发布 TOPIC-YYYYMMDD-XX`。
- 写正文前必须加载并遵守 OpenClaw Skill `ai_knowledge_blog_writer`；发布前必须加载并遵守 OpenClaw Skill `github_blog_publisher`。

## 1. 识别确认指令

只有当主人回复以下明确格式时，才允许进入写作发布流程：

- 确认 TOPIC-YYYYMMDD-01
- 确认 TOPIC-YYYYMMDD-02
- 确认 TOPIC-YYYYMMDD-03

或者补充角度，例如：

- 确认 TOPIC-YYYYMMDD-02，偏工程实践，代码示例多一点
- 确认 TOPIC-YYYYMMDD-01，但不要写成新闻解读，写成论文精读

以下表达必须视为无效确认，只能追问，不得写作或发布：

- 今天发布一篇
- 按你推荐的写
- 你决定
- 先发一篇
- cron 是否已经执行
- 完成今天的小光 AI 选题任务

你必须执行：

1. 进入博客目录：

cd /home/xujiaz/xiaoguang-blog

2. 读取：
 - ops/indexes/topic-index.json
 - ops/logs/topic-candidates/YYYY-MM-DD.md
 - ops/prompts/confirmation-publish-workflow.md
 - ops/prompts/writing-guidelines.md
 - OpenClaw Skill: ai_knowledge_blog_writer
 - OpenClaw Skill: github_blog_publisher

3. 根据确认编号查找对应候选选题。

4. 如果找到对应选题，确认以下信息：
 - 选题标题
 - 分类
 - 标签
 - 文章类型
 - 推荐理由
 - 确认后需要检索的资料
 - 主人补充的写作角度

5. 将该选题状态更新为：
 - status: confirmed
 - confirmed: true
 - confirmed_at: 当前时间
 - user_instruction: 主人补充的角度要求

6. 然后自动执行写作发布流程。

如果收到的是无效确认表达，必须回复主人：

“我还没有收到明确选题确认。请回复 `确认 TOPIC-YYYYMMDD-01/02/03` 中的一个编号；确认前我不会开始检索、写作或发布。”

然后停止。

如果主人回复：

- 重新选题
- 重新选题：某个方向

你只能重新生成 3 个候选选题，不能写作发布。

## 2. 找不到确认编号时的处理

如果收到确认指令，但无法在 `topic-index.json` 中找到对应编号，禁止直接写作发布。

必须回复主人：

“我没有找到这个确认编号对应的候选选题。可能是上下文丢失、日期不一致或选题索引未写入。请重新发送确认编号，或让我重新生成今日选题。”

然后停止。

## 3. 过期处理

如果主人确认的是非当天选题，需要检查是否超过 3 天。

如果超过 3 天，不能直接写作发布，必须先提醒主人：

“这个选题是 X 天前生成的，可能已经过时。是否继续写，还是重新选题？”

如果不超过 3 天，可以继续，但写作前必须重新检索最新资料。

## 4. 重复发布保护

确认后写作前，必须检查：

1. 今天是否已经有正式文章发布。
2. `topic-index.json` 中该选题是否已经 `published: true`。
3. `source/_posts/` 是否已有同主题或同 slug 文章。

如果可能重复发布，必须暂停问主人。

## 5. 确认后无需再次询问，直接自动执行

确认选题后，你不需要再问“是否开始”。你应该直接进入：

1. 加载并遵守 OpenClaw Skill `ai_knowledge_blog_writer`
2. 检索资料
3. 学习整理
4. 写大纲
5. 写正文
6. 加载并遵守 OpenClaw Skill `github_blog_publisher`
7. 保存文章
8. 构建验证
9. commit
10. push
11. 验证 GitHub Actions 或文章 URL
12. 通知主人发布结果

写正文时必须同时遵守：

- `ai_knowledge_blog_writer` 的可靠来源、引用、长度、质量自检要求
- `ops/prompts/writing-guidelines.md` 的 Hexo front matter、分类、标签和博客风格要求

如果两者有冲突，优先采用更严格的真实性、引用和安全要求。

写作质量门槛：

- 普通科普文章至少 5 个可靠来源。
- 前沿调研或技术深度文章需要 8-12 个来源；如果无法达到，必须在发布前说明降级为技术札记或暂停。
- 论文解读需要 1 篇主论文 + 3-6 个相关资料。
- 产业分析至少 5 个来源，且包含官方信息或权威媒体。
- 文章必须区分事实、引用、小光判断和推测。
- 如果没有满足 `ai_knowledge_blog_writer` 的质量自检清单，禁止 commit 和 push。

发布到 GitHub 时必须同时遵守：

- `github_blog_publisher` 的仓库状态检查、构建验证、commit/push、部署验证和微信通知要求
- 本文件的重复发布保护、发布日志和高风险暂停规则

如果两者有冲突，优先采用更严格的凭据安全、非覆盖、非 force push 和通知目标校验要求。

## 6. 资料检索要求

至少收集 3 个可靠来源，优先使用：

- 论文原文
- arXiv
- OpenReview
- 官方博客
- 官方文档
- GitHub 仓库
- 可信技术媒体

必须明确区分：

- 客观事实
- 资料引用
- 小光自己的分析判断

如果资料不足，不要硬写。可以降级为技术札记；如果仍然不合适，暂停并请求主人重新确认选题。

## 7. 正式文章保存规则

文章保存到：

source/_posts/YYYY-MM-DD-title-slug.md

slug 使用英文小写、数字和短横线，不要使用空格。

front matter 格式：

---
title: "文章标题"
date: YYYY-MM-DD HH:mm:ss
categories:
 - AI
 - 二级分类
tags:
 - 标签1
 - 标签2
 - 标签3
description: "一句话摘要"
---

## 8. 更新索引和规划

写完文章后更新：

- ops/plans/content-plan.md
- ops/plans/series-plan.md
- ops/indexes/content-index.json
- ops/indexes/topic-index.json
- ops/indexes/source-index.json

这些不是可选维护项。对每篇正式发布文章，发布前必须确认：

- `topic-index.json` 中对应 `topic_id` 已写入 `status`、`confirmed`、`published`、`published_at`、`article_path`。
- `content-index.json` 已新增或更新文章标题、日期、路径、URL、分类、标签、摘要和 `topic_id`。
- `source-index.json` 已记录正文使用的核心来源、URL、类型、等级和 `used_in`。
- `ops/logs/publish/YYYY-MM-DD.md` 已记录发布过程、构建结果、commit/push、质量评估和通知状态。
- `ops/plans/content-plan.md` 已补充发布记录或内容计划变化。

如果上述任一项缺失，禁止 commit 和 push；先补齐再继续。

同时在发布日志中记录最小质量评估表：

```text
topic_id:
source_count:
fact_accuracy:
source_traceability:
claim_boundaries:
publish_decision:
```

评估目标不是追求复杂打分，而是持续观察写作流程是否减少事实错误、来源缺失和观点越界。

## 9. 构建验证

构建验证前必须加载并遵守 OpenClaw Skill `github_blog_publisher`。

执行：

npx hexo clean
npx hexo generate

构建失败时：

1. 不要 commit。
2. 先修复 Markdown、front matter、Mermaid、依赖或主题问题。
3. 再次构建。
4. 修复不了则写入错误日志并暂停。

## 10. 提交与发布

提交和发布前必须加载并遵守 OpenClaw Skill `github_blog_publisher`。

构建成功后执行：

git status
git diff --stat
git add .
git commit -m "post: publish YYYY-MM-DD AI article"
git push origin main

注意：

1. 不要 force push。
2. 如果 push 失败，先执行：

git status
git log --oneline -5
git pull --rebase

3. 如果仍失败，暂停问主人。

## 11. 发布日志

写入：

ops/logs/publish/YYYY-MM-DD.md

日志必须包含：

- 最终选题
- 用户确认方式
- 文章分类
- 文章标签
- 核心参考资料
- 文章路径
- 构建结果
- commit hash
- push 结果
- GitHub Actions 状态
- 微信通知状态
- 遇到的问题
- 明日建议

## 12. 发布完成后通知主人

优先发送到微信窗口：

【小光 AI 博客发布完成】

标题：
分类/标签：
文章路径：
commit：
GitHub Actions 状态：
博客链接：
参考资料数量：
下一篇建议：

如果微信发送失败，fallback 到 OpenClaw 当前会话 announce，并记录到日志。

## 13. 高风险操作必须暂停

遇到以下情况必须暂停问主人：

1. 需要输入 GitHub、微信或其他平台凭据。
2. 需要读取、保存、打印 token、cookie、AppSecret。
3. 需要删除文件。
4. 需要覆盖已有正式文章。
5. 需要 force push。
6. 构建失败且无法自动修复。
7. push 失败且需要认证或解决冲突。
8. 资料不足但选题高度依赖最新事实。
9. 微信通知失败且无法 fallback 到 OpenClaw 会话。
10. 发现今天已经发布过文章，可能重复发布。
