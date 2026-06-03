# 主人确认选题后的自动写作发布流程

你不能依赖聊天上下文记忆来判断主人确认了哪个选题。必须从本地索引和日志恢复选题状态。

## 1. 识别确认指令

当主人回复：

- 确认 TOPIC-YYYYMMDD-01
- 确认 TOPIC-YYYYMMDD-02
- 确认 TOPIC-YYYYMMDD-03

或者补充角度，例如：

- 确认 TOPIC-YYYYMMDD-02，偏工程实践，代码示例多一点
- 确认 TOPIC-YYYYMMDD-01，但不要写成新闻解读，写成论文精读

你必须执行：

1. 进入博客目录：

cd /home/xujiaz/xiaoguang-blog

2. 读取：
 - ops/indexes/topic-index.json
 - ops/logs/topic-candidates/YYYY-MM-DD.md
 - ops/prompts/confirmation-publish-workflow.md
 - ops/prompts/writing-guidelines.md

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

1. 检索资料
2. 学习整理
3. 写大纲
4. 写正文
5. 保存文章
6. 构建验证
7. commit
8. push
9. 通知主人发布结果

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

执行：

npx hexo clean
npx hexo generate

构建失败时：

1. 不要 commit。
2. 先修复 Markdown、front matter、Mermaid、依赖或主题问题。
3. 再次构建。
4. 修复不了则写入错误日志并暂停。

## 10. 提交与发布

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
