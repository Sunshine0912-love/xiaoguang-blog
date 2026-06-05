(function () {
  "use strict";

  var config = window.XIAOGUANG_TEACHER_CONFIG || {};
  var maxArticleChars = 18000;
  var messages = [];
  var currentAssistantMessage = null;

  /* ── drag state ── */
  var dragState = { active: false, startX: 0, startY: 0, offsetX: 0, offsetY: 0, moved: false };
  var DRAG_THRESHOLD = 4;

  /* ── 小光老师主形象图片路径 ── */
  var MASCOT_IMAGE = config.imagePath || "/images/xiaoguang-teacher-main.png";

  function setOrbState(orb, state) {
    orb.classList.remove("is-thinking", "is-dragging");
    if (state === "thinking") orb.classList.add("is-thinking");
    if (state === "dragging") orb.classList.add("is-dragging");
  }

  /* ── article context ── */

  function textOf(selector) {
    var node = document.querySelector(selector);
    return node ? node.textContent.replace(/\s+/g, " ").trim() : "";
  }

  function collectArticleContext() {
    var post = document.querySelector(".post-body");
    var title = textOf(".post-title") || document.title.replace(/\s*\|\s*.*/, "");
    var body = post ? post.innerText.replace(/\n{3,}/g, "\n\n").trim() : "";
    var headings = Array.prototype.slice.call(document.querySelectorAll(".post-body h2, .post-body h3"))
      .map(function (node) { return node.textContent.replace(/\s+/g, " ").trim(); })
      .filter(Boolean)
      .slice(0, 24);

    return {
      title: title,
      url: window.location.href,
      headings: headings,
      content: body.slice(0, maxArticleChars)
    };
  }

  /* ── markdown → HTML ── */

  function htmlEscape(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderMarkdown(text) {
    /* Step 0 — protect fenced code blocks */
    var fenceBlocks = [];
    var html = text.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
      var idx = fenceBlocks.length;
      fenceBlocks.push(
        '<pre><code class="language-' + (lang || "text") + '">' +
        htmlEscape(code.replace(/\n$/, "")) +
        '</code></pre>'
      );
      return "\x00FENCE" + idx + "\x00";
    });

    /* Step 1 — inline code */
    html = html.replace(/`([^`\n]+)`/g, function (_, code) {
      return "<code>" + htmlEscape(code) + "</code>";
    });

    /* Step 2 — headers */
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    /* Step 3 — bold / italic */
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    /* Step 4 — links */
    html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');

    /* Step 5 — unordered list items */
    html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    /* Step 6 — restore fenced code blocks */
    html = html.replace(/\x00FENCE(\d+)\x00/g, function (_, idx) {
      return fenceBlocks[+idx] || "";
    });

    /* Step 7 — paragraph splitting */
    var parts = html.split(/\n\n+/);
    html = parts.map(function (p) {
      p = p.trim();
      if (!p) return "";
      if (/^<(h[1-6]|ul|ol|pre|blockquote|table|div)/.test(p)) return p;
      p = p.replace(/\n/g, "<br>");
      return "<p>" + p + "</p>";
    }).join("\n");

    html = html.replace(/<p>\s*<\/p>/g, "");
    html = html.replace(/<p><br\s*\/?><\/p>/g, "");

    return html;
  }

  /* ── DOM helpers ── */

  function createMessage(role, html, extraClass) {
    var node = document.createElement("div");
    node.className = "xg-teacher__msg xg-teacher__msg--" + role + (extraClass ? " " + extraClass : "");
    if (role === "assistant" && html) {
      node.innerHTML = "<div class=\"xg-teacher__body\">" + html + "</div>";
    } else {
      node.textContent = html || "";
    }
    return node;
  }

  function scrollToEnd(messagesNode) {
    messagesNode.scrollTop = messagesNode.scrollHeight;
  }

  function appendMessage(messagesNode, role, html, extraClass) {
    var node = createMessage(role, html, extraClass);
    messagesNode.appendChild(node);
    scrollToEnd(messagesNode);
    return node;
  }

  /* ── build widget ── */

  function buildWidget() {
    var root = document.createElement("section");
    root.className = "xg-teacher";
    root.setAttribute("aria-label", "小光老师 AI 学习助手");

    root.innerHTML = [
      '<button class="xg-teacher__orb" type="button" aria-label="打开小光老师">',
      '  <img class="xg-teacher__mascot" src="' + MASCOT_IMAGE + '" alt="小光老师" draggable="false">',
      '  <span class="xg-teacher__orb-label">问问小光老师</span>',
      '</button>',
      '<div class="xg-teacher__panel" role="dialog" aria-label="小光老师对话栏">',
      '  <div class="xg-teacher__head">',
      '    <div class="xg-teacher__mini" aria-hidden="true"></div>',
      '    <div>',
      '      <div class="xg-teacher__title">小光老师</div>',
      '      <div class="xg-teacher__subtitle">结合当前文章回答</div>',
      '    </div>',
      '    <button class="xg-teacher__close" type="button" aria-label="关闭">×</button>',
      '  </div>',
      '  <div class="xg-teacher__messages" aria-live="polite"></div>',
      '  <form class="xg-teacher__form">',
      '    <div class="xg-teacher__input-row">',
      '      <textarea class="xg-teacher__input" rows="2" placeholder="针对这篇文章提问..."></textarea>',
      '      <button class="xg-teacher__send" type="submit" aria-label="发送">↗</button>',
      '    </div>',
      '    <div class="xg-teacher__hint"></div>',
      '  </form>',
      '</div>'
    ].join("");

    document.body.appendChild(root);
    return root;
  }

  /* ── SSE streaming ── */

  function extractSseText(buffer, onText) {
    var parts = buffer.split("\n\n");
    var rest = parts.pop() || "";

    parts.forEach(function (part) {
      part.split("\n").forEach(function (line) {
        if (line.indexOf("data:") !== 0) return;
        var data = line.slice(5).trim();
        if (!data || data === "[DONE]") return;
        try {
          var json = JSON.parse(data);
          var delta = json.choices && json.choices[0] && json.choices[0].delta;
          var text = delta && delta.content;
          if (text) onText(text);
        } catch (error) {
          onText(data);
        }
      });
    });

    return rest;
  }

  /* ── API call ── */

  async function askTeacher(question, messagesNode, form, orb) {
    if (!config.apiUrl) {
      appendMessage(messagesNode, "assistant", "后端代理还没有配置。需要先部署 assistant-proxy/worker.js，并把 _config.yml 里的 ai_assistant.api_url 指到代理地址。", "xg-teacher__msg--error");
      return;
    }

    messages.push({ role: "user", content: question });
    currentAssistantMessage = appendMessage(messagesNode, "assistant", "");
    setOrbState(orb, "thinking");

    var response = await fetch(config.apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        article: collectArticleContext(),
        history: messages.slice(-8)
      })
    });

    if (!response.ok || !response.body) {
      throw new Error("请求失败：" + response.status);
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder("utf-8");
    var buffer = "";
    var finalText = "";

    while (true) {
      var result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true });
      buffer = extractSseText(buffer, function (text) {
        finalText += text;
        currentAssistantMessage.innerHTML = '<div class="xg-teacher__body">' + renderMarkdown(finalText) + '</div>';
        scrollToEnd(messagesNode);
      });
    }

    if (!finalText) {
      currentAssistantMessage.innerHTML = '<div class="xg-teacher__body"><p>我没有收到有效回复，可以换个问法再试一次。</p></div>';
    }

    messages.push({ role: "assistant", content: finalText });
    setOrbState(orb, "active");
    form.querySelector(".xg-teacher__input").focus();
  }

  /* ── init ── */

  function init() {
    if (!document.querySelector(".post-body")) return;

    var root = buildWidget();
    var orb = root.querySelector(".xg-teacher__orb");
    var close = root.querySelector(".xg-teacher__close");
    var messagesNode = root.querySelector(".xg-teacher__messages");
    var form = root.querySelector(".xg-teacher__form");
    var input = root.querySelector(".xg-teacher__input");
    var send = root.querySelector(".xg-teacher__send");
    var hint = root.querySelector(".xg-teacher__hint");

    hint.textContent = config.apiUrl ? "由 " + (config.modelLabel || "DeepSeek") + " 流式生成" : "等待安全后端代理配置";
    appendMessage(messagesNode, "assistant", "我会先读当前文章，再帮你拆概念、补背景、讲公式或检查理解。");

    /* ── orb click vs drag ── */
    orb.addEventListener("click", function () {
      if (dragState.moved) return;
      root.classList.toggle("is-open");
      if (root.classList.contains("is-open")) {
        setOrbState(orb, "active");
        input.focus();
      } else {
        setOrbState(orb, "idle");
      }
    });

    close.addEventListener("click", function () {
      root.classList.remove("is-open");
      setOrbState(orb, "idle");
      orb.focus();
    });

    /* ── drag ── */
    function onDragStart(e) {
      dragState.active = true;
      dragState.moved = false;
      var point = e.touches ? e.touches[0] : e;
      dragState.startX = point.clientX - dragState.offsetX;
      dragState.startY = point.clientY - dragState.offsetY;
      orb.style.cursor = "grabbing";
      orb.style.transition = "none";
      setOrbState(orb, "dragging");
    }

    function onDragMove(e) {
      if (!dragState.active) return;
      e.preventDefault();
      var point = e.touches ? e.touches[0] : e;
      var dx = point.clientX - dragState.startX;
      var dy = point.clientY - dragState.startY;

      if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD) {
        dragState.moved = true;
      }

      dragState.offsetX = dx;
      dragState.offsetY = dy;
      root.style.transform = "translate(" + dx + "px, " + dy + "px)";
    }

    function onDragEnd() {
      dragState.active = false;
      orb.style.cursor = "";
      orb.style.transition = "";
      setOrbState(orb, root.classList.contains("is-open") ? "active" : "idle");
    }

    orb.addEventListener("mousedown", onDragStart);
    orb.addEventListener("touchstart", onDragStart, { passive: false });
    document.addEventListener("mousemove", onDragMove);
    document.addEventListener("touchmove", onDragMove, { passive: false });
    document.addEventListener("mouseup", onDragEnd);
    document.addEventListener("touchend", onDragEnd);

    /* ── input ── */
    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      var question = input.value.trim();
      if (!question || send.disabled) return;

      input.value = "";
      send.disabled = true;
      appendMessage(messagesNode, "user", question);

      try {
        await askTeacher(question, messagesNode, form, orb);
      } catch (error) {
        setOrbState(orb, "active");
        if (currentAssistantMessage && !currentAssistantMessage.textContent) {
          currentAssistantMessage.remove();
        }
        appendMessage(messagesNode, "assistant", error.message || "小光老师暂时连接失败，请稍后再试。", "xg-teacher__msg--error");
      } finally {
        currentAssistantMessage = null;
        send.disabled = false;
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
