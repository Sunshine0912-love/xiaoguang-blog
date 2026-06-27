# 内容规划

## 目标

- 长期沉淀高质量 AI 技术内容。
- 优先写可复用、可验证、能形成系列的技术文章。
- 每日候选选题只进入 `pending_confirmation`，主人确认前不写正文、不发布。
- 从 2026-06-04 起，每天维护两条发布线：一篇保留原每日选题逻辑，另一篇新增为技术点讲解博客。TECH 线面向 AI 专业研究生、AI 研究员、AI 工程师和 AI 学者，重点解释算法、架构、系统机制和工程优化，要求更多公式/伪代码/结构图，由浅入深讲清机制，使读者能真正学懂技术点。

## 近期方向

- AI Agent 工程化
- LLM 推理与上下文系统
- AI Infra 与训练/推理优化
- 多模态生成模型
- 开源模型与行业动态
- AI 技术点讲解：Transformer、Diffusion、MLA/MQA/GQA、PPO/GRPO、MoE、Speculative Decoding、CUDA/算子优化、分布式训练、推理服务优化、Agent 核心机制

## 发布记录

- 2026-06-02：你好，小光，博客搭建测试文章。
- 2026-06-03：Agent 落地进入第二阶段：企业真正卡住的不是模型。
- 2026-06-03：Agentic Inference 的成本曲线：长上下文、工具调用与多轮规划如何重塑 AI Infra。
- 2026-06-04：从单体 Serving 到 Prefill/Decode 解耦：LLM 推理服务架构为什么正在重构。
- 2026-06-04：MLA 注意力机制拆解：DeepSeek 如何用低秩潜变量压缩 KV Cache。

- 2026-06-05：KVarN 深度评测：vLLM 原生 KV Cache 量化后端能不能进生产。
- 2026-06-05：Headroom 实战：LLM 上下文压缩工具如何给 Agent 省 token。
- 2026-06-05：AI 辅助代码安全工具链：Anthropic Defending Code Reference Harness 解析。
- 2026-06-05：Q、K、V 投影共享：Transformer 注意力机制真的需要三个投影矩阵吗？


## 2026-06-11 发布（三篇批量）

### TOPIC-20260611-01: Diffusion LLM 架构解析
- 类型: 深度文章（前沿调研/架构分析）
- 分类: Topic / AI / LLM
- 来源: 12
- 状态: 已发布

### TECH-20260611-01: 扩散语言模型的数学原理
- 类型: 研究级技术点讲解（算法推导）
- 分类: TECH / AI / LLM
- 来源: 11
- 公式: 28 KaTeX display blocks
- 状态: 已发布

### TECH-20260611-02: Constitutional AI 训练机制深度拆解
- 类型: 研究级技术点讲解（算法推导）
- 分类: TECH / AI / 强化学习
- 来源: 13
- 公式: 12 KaTeX display blocks
- 状态: 已发布

## 2026-06-26 发布（五篇批量）

### TOPIC-20260625-02: AI 大厂人才流动
- 类型: 产业观察
- 分类: AI / Industry
- 来源: 6
- 状态: 已发布

### TOPIC-20260625-03: Gemini 3.5 Flash computer use
- 类型: 工程解读
- 分类: AI / Agent
- 来源: 6
- 状态: 已发布

### TECH-20260625-02: 长程 Agent 规划机制
- 类型: 研究级技术点讲解（系统机制）
- 分类: TECH / AI / Agent
- 来源: 6
- 公式: 状态空间、价值函数
- 状态: 已发布

### TOPIC-20260626-03: huggingface_hub release CI
- 类型: 短篇技术札记 / MLOps
- 分类: AI / MLOps
- 来源: 6
- 状态: 已发布

### TECH-20260626-01: vLLM PagedAttention 与连续批处理
- 类型: 研究级技术点讲解（工程优化）
- 分类: TECH / AI / AI Infra
- 来源: 7
- 公式: KV cache 显存复杂度、block table
- 状态: 已发布

## 2026-06-27 发布（两篇）

### TOPIC-20260627-03: 生命科学 Agent
- 类型: 前沿调研 / AI for Science
- 分类: AI / Research
- 来源: 5
- 状态: 已发布

### TECH-20260627-02: KV Cache、MQA/GQA 与长上下文推理成本
- 类型: 研究级技术点讲解（工程优化）
- 分类: AI / AI Infra
- 来源: 6
- 公式: KV cache 显存复杂度、MQA/GQA 显存比例、最小显存例子
- 状态: 已发布
