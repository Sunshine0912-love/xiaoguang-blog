export default {
  async fetch(request, env) {
    const originCheck = verifyOrigin(request, env);
    if (!originCheck.ok) {
      return cors(json({ error: "Forbidden origin" }, 403), env);
    }

    if (request.method === "OPTIONS") {
      return cors(new Response(null, { status: 204 }), env);
    }

    if (request.method !== "POST") {
      return cors(json({ error: "Method not allowed" }, 405), env);
    }

    if (!env.DEEPSEEK_API_KEY) {
      return cors(json({ error: "DEEPSEEK_API_KEY is not configured" }, 500), env);
    }

    let payload;
    try {
      payload = await request.json();
    } catch (error) {
      return cors(json({ error: "Invalid JSON body" }, 400), env);
    }

    const question = String(payload.question || "").trim();
    const article = payload.article || {};
    const history = Array.isArray(payload.history) ? payload.history.slice(-8) : [];

    if (!question) {
      return cors(json({ error: "Question is required" }, 400), env);
    }

    if (question.length > 1200) {
      return cors(json({ error: "Question is too long" }, 400), env);
    }

    const prompt = buildPrompt(question, article);
    const upstream = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.DEEPSEEK_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: env.DEEPSEEK_MODEL || "deepseek-v4-pro",
        stream: true,
        temperature: 0.35,
        messages: [
          {
            role: "system",
            content: [
              "你是小光老师，一个中文 AI 学习助手。",
              "回答必须优先基于用户正在阅读的博客内容，同时可以补充必要背景。",
              "如果文章没有覆盖某个细节，要明确说这是补充推断或通用知识。",
              "保持解释清晰、耐心、专业，适合学习 AI 的工程师和研究生。",
              "涉及公式、代码、论文结论时要谨慎，不要编造来源。"
            ].join("\n")
          },
          ...history.map(normalizeMessage),
          { role: "user", content: prompt }
        ]
      })
    });

    return cors(new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") || "text/event-stream; charset=utf-8",
        "Cache-Control": "no-store"
      }
    }), env);
  }
};

function buildPrompt(question, article) {
  const title = clean(article.title).slice(0, 200);
  const url = clean(article.url).slice(0, 500);
  const headings = Array.isArray(article.headings) ? article.headings.map(clean).filter(Boolean).slice(0, 24) : [];
  const content = clean(article.content).slice(0, 18000);

  return [
    "请基于下面这篇博客回答用户问题。",
    "",
    "【用户问题】",
    question,
    "",
    "【当前文章】",
    `标题：${title}`,
    `链接：${url}`,
    headings.length ? `目录：\n${headings.map((item, index) => `${index + 1}. ${item}`).join("\n")}` : "",
    "",
    "【文章正文摘录】",
    content,
    "",
    "【回答要求】",
    "1. 先直接回答问题，再展开解释。",
    "2. 若问题涉及文章中的概念、公式或工程实现，请引用文章上下文进行说明。",
    "3. 不确定时说明不确定性，不要假装文章已经覆盖。"
  ].filter(Boolean).join("\n");
}

function normalizeMessage(message) {
  const role = message && message.role === "assistant" ? "assistant" : "user";
  return {
    role,
    content: clean(message && message.content).slice(0, 3000)
  };
}

function clean(value) {
  return String(value || "").replace(/\u0000/g, "").trim();
}

function json(body, status) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" }
  });
}

function verifyOrigin(request, env) {
  const allowedOrigin = env.ALLOWED_ORIGIN || "https://sunshine0912-love.github.io";
  const origin = request.headers.get("Origin");
  return { ok: origin === allowedOrigin };
}

function cors(response, env) {
  const allowedOrigin = env.ALLOWED_ORIGIN || "https://sunshine0912-love.github.io";
  const headers = new Headers(response.headers);
  headers.set("Access-Control-Allow-Origin", allowedOrigin);
  headers.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type");
  headers.set("Vary", "Origin");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
