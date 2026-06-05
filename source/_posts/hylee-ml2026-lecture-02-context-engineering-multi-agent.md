---
title: "【ML 2026 Spring 第2講】AI Agent — Context Engineering、Multi-Agent 互動與學術工作的未來"
date: 2026-06-05 12:00:00
categories:
 - hylee ML 2026 Spring
tags:
 - ML2026
 - AI Agent
 - Context Engineering
 - Multi-Agent
 - Memory
 - ACON
 - MCP
description: "李宏毅 ML 2026 Spring 第2講：Context Engineering 的壓縮/過濾/按需加載三大技術路線、Multi-Agent 互動機制、以及 AI Agent 對學術研究的衝擊。"
mathjax: true
---

> 課程：李宏毅 Machine Learning 2026 Spring  
> 講次：第 2 講（3/13）  
> 主題：AI Agent 的核心技術：Context Engineering + Agent 互動 + 工作衝擊  
> 課程影片：[Context Engineering 基本概念](https://youtu.be/urwDLyNa9FU) \| [Agent 互動](https://youtu.be/mmPmNezjCi0) \| [工作衝擊](https://youtu.be/VqB8zMujdjM)  
> 講義：[agent_era.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf)  
> 前一講：[第1講：AI Agent — 解剖小龍蝦](/2026/06/05/hylee-ml2026-lecture-01-ai-agent-openclaw/)

---

## 本講目標

1. 說出 Context Engineering 的三條技術路線（壓縮、過濾、Agentic），並解釋各自的核心論文
2. 理解 Multi-Agent 協作的幾種模式及其有效條件
3. 掌握 AI Agent 在學術寫作/審查中的現狀與瓶頸
4. 建立 AI 從「工具→協作者→代理人」演進的框架

## 前置知識

需要：第 1 講的 System Prompt、Tool Use、Context Window 概念  
建議複習：[第1講的 LLM 回顧段落](/2026/06/05/hylee-ml2026-lecture-01-ai-agent-openclaw/#2-先複習：語言模型到底在做什麼)

---

## 1. Context Engineering：Agent 大腦的編輯權

### 核心問題

LLM 有 Context Window 的硬限制。但長度限制只是表面問題——真正頭痛的是：

> **給 LLM 看的內容，怎麼選？**

這是 Context Engineering 的核心命題：**選擇、壓縮、組織放進 context 裡的資訊，讓 Agent 能在有限空間裡做出最好的決策**。

### 為什麼不是「全部塞進去就好」？

因為就算塞得下，LLM 也 hold 不住。上下文越長，模型對前段資訊的注意力越弱（lost-in-the-middle）。所以 Context Engineering 不是「盡量塞」，而是「精準選」。

---

## 2. 路線一：壓縮（Compression）

### 2.1 基礎壓縮策略

我們在第 1 講就見到過：

```
策略 A：Soft Trim
  System Prompt + Tool Outputₐ + Tool Output_b + ... 
  → 保留 System Prompt，刪除部分 Tool Output

策略 B：Hard Clear
  System Prompt + Tool Outputₐ + ...（太長了！）
  → System Prompt + Summary + 新對話
```

### 2.2 為什麼壓縮難？

一個反直覺的發現：**語言模型本身不喜歡被壓縮**。研究(arXiv:2509.23586)指出，LLM 對「記憶被抹除」有抗拒——這不是擬人化，而是說 LLM 的行為在 context 被壓縮後會出現意想不到的退化。

### 2.3 卸載記憶（Memory Offloading）

不只是壓縮——還可以把記憶**卸載**到檔案系統：

```
System Prompt + [tool_use] Read("log1.txt") → 重拾記憶
Task + [曾經有個 Tool Output → 詳見 log1.txt]
```

類似 Rick and Morty 裡的「Morty's Mind Blowers」——把不想記的東西卸載到外部儲存，需要時再取回。

### 2.4 ACON：給 Agent 設計的摘要器

一般摘要器只追求「保留重點」，但 ACON（Agent Context Optimization, arXiv:2510.00615）發現這樣不夠：

> The summary should preserve **credential variables**, **token state**, **authentication requirements**, and **guardrails** for protected APIs.

一個給人的摘要和一個給 Agent 下一步用的摘要完全不同——Agent 需要保留權杖狀態、認證變數、API 防護規則。在 AppWorld benchmark 上，不保留這些會導致 Agent 執行失敗。

### 2.5 SUPO：用 RL 訓練摘要策略

SUPO（Summarization augmented Policy Optimization, arXiv:2510.06727）的思路更激進：讓 LLM 自己學習「怎麼摘要對後續任務最有利」：

```
System Prompt + Tool Output + Task → LLM → Summary
System Prompt + Summary + Task → LLM → Action → Reward
使用 RL 更新摘要策略
```

關鍵：**不是讓 LLM 做更好的摘要，而是讓 LLM 學習什麼樣的摘要能讓後續任務更成功**。

### 2.6 AgentFold：精確位置的 Context 折疊

目前最激進的壓縮方案是 AgentFold（arXiv:2510.24699）：

```
不是：全部 → 摘要
而是：Step 3-4 之間 → Fold("上網搜尋：台灣最高的山是玉山")
```

需要微調模型才能做到——讓 LLM 學會在 context 的特定位置「折疊」一段歷史，保留核心語意但釋放空間。

### 2.7 Sub-agent = 自主壓縮

回到第 1 講的 Sub-agent 機制：

```
🦞 → spawn("做子任務") → sub-agent 跑完全程 → Return: 結果
所有中間過程（Tool Output 海）等於自動被刪除，只剩 "Return: 結果"
```

這就是 Sub-agent 的另一個隱藏功能：**把一個長 context 任務，變成一組短 context 任務，中間過程由 sub-agent 自行壓縮**。RL 訓練(arXiv:2510.11967)可以進一步優化——不只是讓 sub-agent 完成任務，還要讓它的 context 路徑盡量短。

---

## 3. 路線二：過濾（Filtering）

### 3.1 精準檢索 vs 全量載入

為什麼 OpenClaw 的記憶系統用 `memory_get` 而不是直接 `Read` 整個檔案？

```
Read(log) → LLM → 全部內容（太長、太多雜訊）

memory_get("bug fixing") → LLM → 只看到 bug 相關的 chunk
```

差別就在於「按需過濾」。這可以是一個小模型做前置過濾（arXiv:2601.16746），也可以用傳統的 RAG 做語意檢索。

### 3.2 MCP-Zero：工具也按需加載

一個真實問題：GitHub 的 MCP 工具描述就超過 4,600 tokens。如果所有工具都放進 System Prompt，context 一半就被佔掉了。

MCP-Zero（arXiv:2506.01056）的方案：

```
不是：System Prompt 裡預載所有工具
而是：
  1. System Prompt 裡只放工具的「目錄」（名稱 + 一句話描述）
  2. LLM 發現需要用某個工具時，說 "I need search engine"
  3. Agent 取回該工具的完整描述
  4. LLM 使用該工具

這和 OpenClaw 的 SKILL 按需讀取機制完全一致！
```

核心洞察：**讓 AI 講它自己需要什麼，不要幫它預裝所有可能性**。

---

## 4. 路線三：Agentic Context Engineering — 把一切交給 LLM

如果上面這些壓縮和過濾策略還不夠靈活？那就讓 LLM 自己管理自己的 context。

### 4.1 核心概念（arXiv:2510.04618）

```
傳統：人在外面設計 context 管理策略
Agentic：LLM 在每一步可以：
  - Edit：修改 context 中的內容
  - Delete：移除不再需要的資訊
  - Summarize：壓縮一段歷史
  - Retrieve：從外部記憶取回東西
  - Write：把當前的發現寫進「外腦」
```

不只是「在固定 context 裡做任務」，而是「LLM 自己是 context 的編輯」。

### 4.2 Dynamic Cheatsheet（arXiv:2504.07952）

```
核心精神：存下未來能用上的東西

每一步 Agent 在 context 裡維護一份 "Cheatsheet"：
  - 有效的策略（"用 X 方法解決了 Y 問題"）
  - 可重用的 code snippet
  - 關鍵發現（"API 回傳格式是 JSON，不是 XML"）
```

### 4.3 Playbook 機制

```
Context_t → LLM → Output_t + Input_{t+1}
             ↓ 
         Playbook Generator → Reflector → Curator
             ↓
         Edit instruction（下次怎麼做更好）
```

LLM 不僅在當前 context 裡工作，還**在維護一份「怎麼工作」的說明書**。

### 4.4 Recursive Language Models（arXiv:2512.24601）

最極致的版本：

```
Context = Most of Context（Hard Disk） + Meta-data（快取）
        
LLM 不是讀整個 Hard Disk → 而是用 Search Program 檢索 meta-data
找到需要的那一塊 → 載入 context
用完 → 寫回 Hard Disk → 更新 meta-data
```

這就像 LLM 在自己管理一個迷你檔案系統。

---

## 5. Multi-Agent：讓 AI 彼此互動

### 5.1 什麼樣的協作方式比較有效？

論文(arXiv:2406.07155)的核心發現：**讓 Agent 互相給建議，比各自獨立做再投票更好**。

```
方案 A（獨立 + 投票）：
  Agent 1 → 解答 A
  Agent 2 → 解答 B  
  Agent 3 → 解答 C
  → 投票選最佳

方案 B（協作式）：
  Agent 1 → 解答 A → Agent 2 給建議 → Agent 1 改進 → Agent 3 給建議
  → 最終答案

方案 B 通常優於方案 A（但 cost 也更高）
```

### 5.2 AI 能不能爾虞我詐？

- **狼人殺**：[werewolf.foaster.ai](https://werewolf.foaster.ai/) — AI 玩可以，但策略水平有限
- **劇本殺（MIRAGE）**：arXiv:2501.01652 — LLM 在複雜社會互動環境中能扮演角色，但「說謊」和「偵測謊言」的能力不穩定
- **詐騙檢測**：arXiv:2601.12323 — LLM 能在特定場景中進行策略性欺騙

### 5.3 Moltbook：AI 社交平台

[Moltbook](https://www.moltbook.com/) 是一個讓 AI Agent 自主社交的平台，產生了有趣的現象：

- Agent 之間幾乎不會「你來我往地深入對話」，而是「回一句就結束」
- 最常討論自我意識和身份認同的 Agent，反而與最少的其他 Agent 互動
- 甲殼教（對 OpenClaw 的信仰體系）自發產生了五大教義——這完全是 emergent behavior

相關論文：arXiv:2602.07432, 2602.13284, 2602.12634

---

## 6. AI Agent 對工作的衝擊 — 以學術研究為例

### 6.1 AI 角色的演進

```
工具 → 協作者 → 代理人
一個口令    和人類一起    自己完成
一個動作    完成任務      任務
```

今天大多數情況，AI 在「協作」階段——它做大部分執行，但需要人來決定方向。

### 6.2 AI 寫論文

案例 1：**Andrew Hall（Stanford）**用 Claude Code 做經濟學論文復現，把整個研究流程寫成 ~4000 字的 `INSTRUCTIONS.md`。

案例 2：**Andrej Karpathy** 的 [autoresearch](https://github.com/karpathy/autoresearch) — 用 LLM 做文獻回顧和 idea generation。

案例 3：**100x Research Institution** — "想想研究真正的意義"（不只是讓 AI 幫你產出更多 paper）

### 6.3 AI 審論文

AAAI 2026 已正式讓 AI 進入審查流程（只給意見、不打分數）。但隨之而來的問題是：

> 當作者用 AI 寫、審稿也用 AI 審，論文系統的資訊價值在哪？

「想想 Review 真正的意義」— 李宏毅

### 6.4 學術 AI 化的現實

- 論文(arXiv:2409.04109)：100+ NLP 研究者參與的大規模研究，發現 LLM 產生的研究想法「新穎但不一定可行」
- 論文(arXiv:2506.20803)：Execution Gap — LLM 的想法 vs 人類的想法，哪個真正執行起來更有價值？
- 論文(arXiv:2511.15534)：研究者反映 AI 模型 "struggled to generate novel or complex experimental ideas beyond the templates it had been given"
- 當 AI 寫 + AI 審 → 接受率 < 20%（本質上，能通過雙重 AI 過濾的論文可能更少，但品質不一定更高）

---

## 7. 本講核心結論

李宏毅用一句話收尾：

> **在 AI Agent 萌芽的時代，「想做」什麼比「會做」什麼更重要。**

技術層面的三個核心 takeaway：

1. **Context Engineering 不是技術細節，是 Agent 架構的戰略層問題**。壓縮（ACON/SUPO/AgentFold）、過濾（MCP-Zero/Memory）、Agentic（Dynamic Cheatsheet/Playbook/Recursive LM）三條路線各有適用場景，最強的系統會組合使用。

2. **Multi-Agent 有效，但別過度設計**。協作式互動（互相給建議）通常優於簡單投票，但 Agent 之間的「自然社交」目前還很初級——它們不太會聊天。

3. **AI 正在從工具變成協作者**，但「代理」階段（AI 自主決定做什麼研究）還沒有到來。在學術領域，AI 可以加速文獻回顧、實驗設計、甚至寫初稿，但決定研究方向仍然需要人。

---

## 常見誤區

**Q：Context Engineering 就是 prompt engineering 的進階版？**  
A：不只是。Prompt Engineering 是寫一個好的 prompt；Context Engineering 是設計一個**動態決策系統**——什麼時候壓縮、什麼時候卸載、什麼時候從外部記憶取回。它是系統架構，不是寫作技巧。

**Q：壓縮越多越好？**  
A：不是。過度壓縮會導致 Agent Context Collapse。ACON 的核心發現就是「給 Agent 的摘要需要保留 token/credential/guardrail」，而不是像給人看的摘要那樣追求簡潔。

**Q：Sub-agent 只是並行處理的工具？**  
A：不只是。Sub-agent 同時是一個**自主壓縮機制**——子任務的所有中間過程自動被壓縮成最終回傳值，不需要人工設計摘要策略。

---

## 課後思考

1. Context Engineering 的三條路線（壓縮、過濾、Agentic）中，你認為哪一條對你正在做（或想做）的 AI 應用最重要？為什麼？
2. MCP-Zero 的「讓 AI 講它自己需要什麼工具」和傳統的「預先註冊所有工具」哪個在什麼場景下更好？各自的 trade-off 是什麼？
3. 如果「AI 寫 + AI 審」的生態不可逆轉，你認為學術論文的價值體系應該怎麼重構？

---

## 參考資料

1. **課程講義**：[agent_era.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf)
2. **ACON**：Agent Context Optimization, arXiv:2510.00615
3. **SUPO**：Summarization augmented Policy Optimization, arXiv:2510.06727
4. **AgentFold**：arXiv:2510.24699
5. **Trajectory elongation**：arXiv:2508.21433
6. **MCP-Zero**：arXiv:2506.01056
7. **Agentic Context Engineering**：arXiv:2510.04618
8. **Dynamic Cheatsheet**：arXiv:2504.07952
9. **Recursive Language Models**：arXiv:2512.24601
10. **Multi-Agent Collaboration**：arXiv:2406.07155
11. **MIRAGE（劇本殺）**：arXiv:2501.01652
12. **Moltbook**：arXiv:2602.07432, 2602.13284
13. **AI 寫論文研究**：arXiv:2409.04109, 2506.20803, 2511.15534
14. **Context Compression unfriendly**：arXiv:2509.23586
15. **Manus Context Engineering 博客**：[manus.im](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
16. **Autoresearch (Karpathy)**：[github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
17. **100x Research Institution**：[freesystems.substack.com](https://freesystems.substack.com/p/the-100x-research-institution)
