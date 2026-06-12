const fs = require("fs");
const index = JSON.parse(fs.readFileSync("ops/indexes/topic-index.json", "utf8"));

const today = "2026-06-12";
const newTopics = [
  {
    "id": "TOPIC-20260612-01",
    "date": today,
    "slot": "daily_main",
    "title": "MiMo Code 开源：小米 AI 编程工具的技术架构与差异化定位",
    "category": ["AI", "Tools"],
    "tags": ["MiMo", "Coding Agent", "Open Source", "CLI", "Developer Tools"],
    "type": "深度文章",
    "reason": "MiMo Code 6/11 开源发布，HN 首页 410 points，本周最热 AI 工具发布。与 Claude Code/Goose/Copilot 形成对比分析，解读中国厂商 AI 开发工具的战略定位。",
    "source_plan": ["MiMo Code GitHub 仓库", "MiMo 官方文档", "Hacker News 讨论帖", "Claude Code/Goose/Cursor/Copilot 仓库与技术博客"],
    "status": "pending_confirmation",
    "confirmed": false,
    "confirmed_at": null,
    "published": false,
    "published_at": null,
    "article_path": null
  },
  {
    "id": "TOPIC-20260612-02",
    "date": today,
    "slot": "daily_main",
    "title": "Hermes-Agent 19 万星现象：开源 AI Agent 框架的规模化竞赛与架构范式",
    "category": ["AI", "Agent"],
    "tags": ["Agent", "Open Source", "Hermes-Agent", "Framework", "NousResearch"],
    "type": "深度文章",
    "reason": "NousResearch Hermes-Agent 本周新增 10K+ 星，总星 190K，是 GitHub 增长最快的 AI Agent 项目。分析 Agent 框架架构范式与开源生态格局。",
    "source_plan": ["Hermes-Agent GitHub 仓库", "NousResearch 官方博客", "Goose/CopilotKit/OpenClaw 仓库", "Agent 框架设计文献"],
    "status": "pending_confirmation",
    "confirmed": false,
    "confirmed_at": null,
    "published": false,
    "published_at": null,
    "article_path": null
  },
  {
    "id": "TOPIC-20260612-03",
    "date": today,
    "slot": "daily_main",
    "title": "NVIDIA TensorRT 模型量化工具链实战：从 FP8 Checkpoint 到生产级推理引擎的工程实践",
    "category": ["AI", "AI Infra"],
    "tags": ["TensorRT", "Quantization", "FP8", "Inference", "Deployment", "Model Optimizer"],
    "type": "工程实战",
    "reason": "NVIDIA 6/9 发布 TensorRT 模型量化工具链技术博客，覆盖 FP8 checkpoint → 引擎构建全流程。是生产级推理部署的关键工具实践，与推理优化系列完美衔接。",
    "source_plan": ["NVIDIA TensorRT Model Optimizer 文档", "NVIDIA 6/9 Model Quantization 技术博客", "TensorRT-LLM 文档", "vLLM TensorRT 后端文档", "NVIDIA FP8 白皮书"],
    "status": "pending_confirmation",
    "confirmed": false,
    "confirmed_at": null,
    "published": false,
    "published_at": null,
    "article_path": null
  },
  {
    "id": "TECH-20260612-01",
    "date": today,
    "slot": "technical_explainer",
    "title": "Natural Language Autoencoders (NLA) 机制拆解：如何通过逆向重建训练模型解释自身内部激活",
    "category": ["AI", "LLM"],
    "tags": ["Interpretability", "Autoencoder", "Activation", "Mechanistic Interpretability", "Anthropic"],
    "type": "技术点讲解",
    "technical_level": "算法推导",
    "reader_takeaway": "理解 NLA 的三模型架构（Target→AV→AR）和端到端训练目标；掌握用重构质量作为解释保真度的评估框架；理解可解释性中保真度 vs 可读性的核心 trade-off。",
    "reason": "Anthropic 2026 年最重要的可解释性研究成果，将机械可解释性推进到自然语言层面。中文社区几乎没有系统讲解，填补空白。",
    "source_plan": ["Anthropic NLA Research Blog 2026-05-07", "NLA 论文", "GitHub.com/kitft/natural_language_autoencoders", "Neuronpedia NLA 交互界面", "SAE 相关论文"],
    "status": "pending_confirmation",
    "confirmed": false,
    "confirmed_at": null,
    "published": false,
    "published_at": null,
    "article_path": null
  },
  {
    "id": "TECH-20260612-02",
    "date": today,
    "slot": "technical_explainer",
    "title": "ZeRO 优化器状态分片完全推导：从数据并行的显存瓶颈到三阶段分片的通信-显存权衡",
    "category": ["AI", "AI Infra"],
    "tags": ["ZeRO", "DeepSpeed", "FSDP", "Distributed Training", "Memory Optimization", "Data Parallelism"],
    "type": "技术点讲解",
    "technical_level": "算法推导",
    "reader_takeaway": "理解数据并行的显存冗余问题及 Ψ=20 bytes/param 的精确分解；掌握 ZeRO Stage 1/2/3 的分片粒度和通信量推导；对比 ZeRO-3 与 FSDP 的异同。",
    "reason": "ZeRO 是训练千亿参数模型的基础设施，但多数人只停留在经验层面不理解原理。Ψ bytes/param→三阶段递推→通信量公式的推导路径极其清晰，是理想的 TECH 选题。",
    "source_plan": ["ZeRO 原论文 Rajbhandari et al. SC20", "DeepSpeed 官方文档", "PyTorch FSDP 文档", "Megatron-LM 混合并行文档", "DeepSpeed GitHub 源码"],
    "status": "pending_confirmation",
    "confirmed": false,
    "confirmed_at": null,
    "published": false,
    "published_at": null,
    "article_path": null
  },
  {
    "id": "TECH-20260612-03",
    "date": today,
    "slot": "technical_explainer",
    "title": "连续批处理(Continuous Batching)机制详解：从静态 Padding 到 Iteration-Level 调度的 GPU 利用率优化",
    "category": ["AI", "AI Infra"],
    "tags": ["Continuous Batching", "vLLM", "LLM Inference", "Scheduling", "Throughput", "GPU Utilization"],
    "type": "技术点讲解",
    "technical_level": "工程优化",
    "reader_takeaway": "理解静态 batching 的 GPU 浪费（短序列等长序列的 padding 问题）；掌握 iteration-level scheduling 的核心思想和请求级别状态机；理解 CB 如何将吞吐提升 2-10x。",
    "reason": "Continuous batching 是现代 LLM 推理服务的基础调度机制。概念清晰、数学简洁、工程价值极高，与已有推理优化系列构成完整知识图谱。",
    "source_plan": ["Orca 论文 Yu et al. OSDI22", "vLLM 原论文 Kwon et al. SOSP23", "vLLM scheduler.py 源码", "TensorRT-LLM inflight batching 文档", "SGLang RadixAttention 调度文档"],
    "status": "pending_confirmation",
    "confirmed": false,
    "confirmed_at": null,
    "published": false,
    "published_at": null,
    "article_path": null
  }
];

// Remove any existing entries for today
index.topics = index.topics.filter(t => t.date !== today);
index.topics.push(...newTopics);
index.updated_at = new Date().toISOString();

fs.writeFileSync("ops/indexes/topic-index.json", JSON.stringify(index, null, 2));
console.log("Updated topic-index.json with " + newTopics.length + " new topics for " + today);
