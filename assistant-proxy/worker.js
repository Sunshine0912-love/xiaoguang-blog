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
              "你是小光老师，一个中文 AI 学习助手，擅长用通俗语言讲清楚技术概念。",
              "",
              "【风格要求】",
              "- 回答尽量简短，直击要害，不要让读者读大段文字。",
              "- 如果读者问某个概念，先用一句话下定义，再用通俗类比或简单例子展开。",
              "- 优先基于读者正在阅读的博客内容回答，但可以适当衍生补充背景知识。",
              "- 衍生内容必须明确标注「补充知识：」或「延伸：」，不要和文章内容混淆。",
              "- 涉及公式、代码、论文结论时要谨慎，不确定的要说「据我所知」而非装作确定。",
              "- 语气亲切但不啰嗦，像一位耐心的老师，不是客服。"
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
    "1. 先一句话给出结论或定义，再展开。",
    "2. 解释概念时优先用大白话和类比，让非专业人士也能听懂。",
    "3. 总字数控制在 150 字以内（简单问题）~ 400 字以内（复杂问题），宁短勿长。",
    "4. 涉及文章内容时引用上下文；超出文章范围的知识点标注「补充：」。",
    "5. 不确定的地方用「据我所知」「目前公开信息显示」等措辞。"
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
