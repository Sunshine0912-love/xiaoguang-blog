# Xiaoguang Blog

小光的技术博客，基于 Hexo 和 NexT，使用 GitHub Actions 部署到 GitHub Pages。

## Local Commands

```bash
npm install
npx hexo clean
npx hexo generate
npx hexo server
```

## Deployment

推送到 `main` 分支后，GitHub Actions 会构建 Hexo 并把 `public/` 作为 GitHub Pages artifact 发布。

目标地址：

```text
https://sunshine0912-love.github.io/xiaoguang-blog/
```
