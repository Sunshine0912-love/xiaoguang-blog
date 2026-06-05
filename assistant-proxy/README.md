# 小光老师后端代理

博客部署在 GitHub Pages，前端代码完全公开，所以 DeepSeek API Key 不能写进页面里。
这个目录提供一个 Cloudflare Worker 代理：浏览器只请求 Worker，Worker 从环境变量读取密钥并流式转发 DeepSeek 回复。

## 环境变量

- `DEEPSEEK_API_KEY`: DeepSeek API Key，必须配置为 Worker secret。
- `DEEPSEEK_MODEL`: 可选，默认 `deepseek-v4-pro`。
- `ALLOWED_ORIGIN`: 可选，默认 `https://sunshine0912-love.github.io`；Worker 会拒绝其他 Origin。

部署后，把 Worker URL 写入根目录 `_config.yml`：

```yaml
ai_assistant:
  enabled: true
  api_url: 'https://your-worker.example.workers.dev'
  model_label: DeepSeek V4 Pro
```
