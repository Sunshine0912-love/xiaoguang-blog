---
title: 你好，小光
date: 2026-06-02 23:59:00
categories:
  - 技术博客
tags:
  - Hexo
  - NexT
  - GitHub Pages
  - AI Engineering
---

这是小光技术博客的第一篇测试文章。这个站点使用 Hexo 生成静态页面，使用 NexT 作为主题，并计划通过 GitHub Actions 自动部署到 GitHub Pages。

## 代码块测试

```js
function introduce(name) {
  return `你好，${name}。让我们持续学习、持续构建。`;
}

console.log(introduce('小光'));
```

## Mermaid 流程图测试

{% mermaid %}
flowchart TD
  A[提出问题] --> B[拆解任务]
  B --> C[实现与验证]
  C --> D[自动化部署]
  D --> E[持续迭代]
{% endmermaid %}

## 验证说明

这篇文章用于验证：

- Hexo front matter 能被正确解析。
- 中文内容能正常渲染。
- 代码块具备高亮和复制按钮。
- Mermaid 图表能在 NexT 主题中渲染。
- GitHub Actions 构建后能发布 `public/` 目录。
