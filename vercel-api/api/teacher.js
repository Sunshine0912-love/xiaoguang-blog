export const config = { runtime: "edge" };

export default async function handler(request) {
  const allowedOrigin = process.env.ALLOWED_ORIGIN || "https://sunshine0912-love.github.io";
  const origin = request.headers.get("Origin");

  /* CORS preflight */
  if (request.method === "OPTIONS") {
    return cors(new Response(null, { status: 204 }), allowedOrigin);
  }

  /* origin check */
  if (origin !== allowedOrigin) {
    return cors(new Response(JSON.stringify({ error: "Forbidden origin" }), {
      status: 403,
      headers: { "Content-Type": "application/json" }
    }), allowedOrigin);
  }

  if (request.method !== "POST") {
    return cors(new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" }
    }), allowedOrigin);
  }

  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    return cors(new Response(JSON.stringify({ error: "DEEPSEEK_API_KEY not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    }), allowedOrigin);
  }

  let payload;
  try {
    payload = await request.json();
  } catch (e) {
    return cors(new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    }), allowedOrigin);
  }

  const question = String(payload.question || "").trim();
  const article = payload.article || {};
  const history = Array.isArray(payload.history) ? payload.history.slice(-8) : [];

  if (!question || question.length > 1200) {
    return cors(new Response(JSON.stringify({ error: "Question required (max 1200 chars)" }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    }), allowedOrigin);
  }

  const prompt = buildPrompt(question, article);
  const upstream = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: process.env.DEEPSEEK_MODEL || "deepseek-v4-pro",
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
  }), allowedOrigin);
}

/* ── helpers ── */

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

function normalizeMessage(msg) {
  const role = msg && msg.role === "assistant" ? "assistant" : "user";
  return { role, content: clean(msg && msg.content).slice(0, 3000) };
}

function clean(v) {
  return String(v || "").replace(/\u0000/g, "").trim();
}

function cors(response, allowedOrigin) {
  const headers = new Headers(response.headers);
  headers.set("Access-Control-Allow-Origin", allowedOrigin);
  headers.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type");
  headers.set("Vary", "Origin");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
