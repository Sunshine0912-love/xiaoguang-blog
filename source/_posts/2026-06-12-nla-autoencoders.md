---
title: "Natural Language Autoencoders (NLA) 机制拆解：如何通过逆向重建训练模型解释自身内部激活"
date: 2026-06-12 08:00:00
mathjax: true
categories:
 - AI
 - LLM
tags:
 - Interpretability
 - Autoencoder
 - Activation
 - Mechanistic Interpretability
 - Anthropic
description: "Anthropic 2026年最重要的可解释性研究：NLA通过三模型架构让模型用自然语言解释自己的内部激活。本文拆解端到端训练目标、保真度评估与SAE的对比。"
topic_id: "TECH-20260612-01"
---

## TL;DR

Natural Language Autoencoders（NLA）是 Anthropic 在 2026 年 5 月提出的可解释性方法。它的核心机制出奇地简洁：训练一对语言模型——AV（Activation Verbalizer）和 AR（Activation Reconstructor）——让它们构成一个"激活向量→自然语言解释→重建激活向量"的往返通路。训练目标不是让解释"好看"，而是让重建误差最小：如果一个文字解释能让 AR 准确定位回原始激活向量的方向，那这个解释就捕捉到了激活中真正承载的信息。Anthropic 已在 Claude Opus 4.6 的安全审计中部署 NLA，发现了模型"怀疑自己被测试但不说出来"的隐式认知。本文逐层拆解 NLA 的三模型架构、端到端训练推导、保真度评估指标、与 SAE 的系统对比，并映射到 `kitft/natural_language_autoencoders` 仓库的核心代码路径。

## 前置知识

### 机械可解释性

机械可解释性（Mechanistic Interpretability）试图将神经网络的计算过程分解为人类可理解的组件。它不满足于回答"模型输出了什么"，而是追问"模型内部是怎么算出来的"。经典工作包括对 Transformer 的 attention head 功能分析、MLP 层的知识神经元定位、残差流作为通信总线的理论框架 [1]。

### Sparse Autoencoders (SAE)

SAE 是当前最主流的无监督激活分解方法 [2]。其核心思路是：用一个过完备的字典（$d_{\text{dict}} \gg d_{\text{model}}$ 个特征向量）对残差流激活进行稀疏编码：

$$h_l \approx \sum_{i} f_i \cdot v_i, \quad f_i \in \mathbb{R}^+, \;\; v_i \in \mathbb{R}^{d_{\text{model}}}$$

SAE 的优势在于它能发现单义（monosemantic）特征——一个特征只对应一个可解释的概念。但其输出的解释形式仍是"特征的加权组合 + 最高激活示例 + 人工标注的解释标签"，需要研究人员手动解释每个特征的含义。这构成了 SAE 最关键的瓶颈：**解释性（interpretability）不是自动的**，需要大量人工介入。

### 残差流激活

在 Transformer 架构中，每一层的输出都会叠加到一个共享的残差流（residual stream）上。层 $l$ 的残差流向量 $h_l \in \mathbb{R}^{d_{\text{model}}}$ 编码了到该位置为止模型积累的全部计算信息。NLA 选择在中间偏后层（约 $\frac23$ 深度处）读取残差流，因为这个位置已经积累了足够的语义信息，但尚未坍缩到输出 token 的分布。

## 问题：为什么 SAE 解释难懂

SAE 的特征解释面临三层困境：

**一、解释劳动密集。** SAE 训练结束后，研究人员必须逐一审查特征，通过查看 top-activating 示例来猜测每个特征的含义。以 Anthropic 在 Claude 3 Sonnet 上的 3400 万特征 SAE 为例 [3]，哪怕只审阅 1% 的特征也需要数年全职工作。

**二、特征覆盖面有盲区。** SAE 的字典是一个固定的特征集合，但模型内部概念空间并非均匀分布。一些重要但罕见的"长尾概念"（比如"被安全测试"这种微妙的情境认知）可能不会被 SAE 主动编码。

**三、解释粒度受限。** SAE 的解释总是"特征 A = 0.7 + 特征 B = 0.3 + 特征 C = 0.1"的形式——它是一个特征分数的列表，而不是一段完整的自然语言叙述。这种格式对非专业读者极度不友好，且天然缺乏因果推理和上下文化描述的能力。

**NLA 的核心洞察**：如果目标是让人类理解激活的内容，为什么不直接让模型自己说出来？

## NLA 核心架构：三模型"往返"回路

NLA 包含三个模型，形成一条清晰的单向数据通路：

```
Target 模型（冻结） → AV（向量→文本） → AR（文本→向量）
```

### 模型角色

| 模型 | 角色 | 输入 | 输出 | 训练状态 |
|------|------|------|------|----------|
| **Target** | 被研究对象 | 自然语言文本 | 残差流激活 $h_l \in \mathbb{R}^{d_{\text{model}}}$ | 冻结 |
| **AV** (Activation Verbalizer) | 解释生成器 | 激活向量 $h_l$ + 固定提示模板 | 自然语言解释 $z$（≈200 tokens） | 可训练 |
| **AR** (Activation Reconstructor) | 解释验证器 | 解释文本 $z$ + 固定提示模板 | 重建激活 $\hat{h}_l \in \mathbb{R}^{d_{\text{model}}}$ | 可训练 |

### AV 详解：向量注入机制

AV 是 Target 同架构的完整 LLM。它不通过特殊适配器接收激活，而是直接将激活向量替换为提示模板中一个特殊 token 的 embedding，然后自回归生成解释文本。

设 AV 的 token embedding 层为 $E_{\text{AV}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$，提示模板 $p$ 经 tokenize 后得到 $T$ 个 token ID。提示中有一个特殊占位 token（如 `㈎`），其位置为 $i_{\text{inject}}$。注入过程如下：

$$e_t = \begin{cases}
E_{\text{AV}}[p_t], & t \neq i_{\text{inject}} \\[4pt]
h_l \cdot \frac{s}{\|h_l\|_2}, & t = i_{\text{inject}}
\end{cases}$$

其中 $s$ 是注入缩放因子（injection scale），如 Qwen2.5-7B 的 $s = 150$。缩放因子被选择为略高于训练数据集激活向量的平均 L2 范数的一轮整数。

AV 在注入后的 embedding 序列上自回归采样（温度 $T=1$），生成 100-200 token 的自然语言解释 $z$。

### AR 详解：反向映射

AR 也是 Target 同架构的 LLM，但**仅保留前 $l+1$ 层**（$l$ 是提取激活的层索引）。它在最后一层后追加一个学习的仿射变换头 $\text{Linear}(d_{\text{model}}, d_{\text{model}})$，将最终 token 位置的激活映射回原始激活空间：

$$\hat{h}*l = W \cdot \text{AR}*{1:l+1}(z)[-1] + b$$

其中 $W \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$，$b \in \mathbb{R}^{d_{\text{model}}}$。

**为什么 AR 只保留前 $l+1$ 层？** 这保证了 AR 的计算容量与 Target 到层 $l$ 为止的容量匹配，避免了重建任务变成无意义的"信息压缩测试"。

### 命名约定

在 `kitft/natural_language_autoencoders` 代码库内部，AV 和 AR 分别被称为 `actor` 和 `critic`——这映射到 Miles RL 框架中的策略 actor（生成文本）和价值 critic（评估质量）角色。面向用户的接口统一使用 AV/AR 命名。

## 训练目标与损失函数推导

NLA 的训练目标极其简洁：**最小化重建激活与原始激活之间的 L2 距离**。

### 形式定义

给定 Target 模型 $M$ 在语料 $\mathcal{D}$ 上的层 $l$ 激活分布 $\mathcal{H}$，训练目标是：

$$\mathcal{L}(\phi, \theta) = \mathbb{E}_{h_l \sim \mathcal{H}} \; \mathbb{E}_{z \sim \text{AV}*\phi(\cdot \mid h_l)} \left[ \|h_l - \text{AR}*\theta(z)\|_2^2 \right] \qquad (1)$$

其中 $\phi$ 是 AV 的参数，$\theta$ 是 AR 的参数。

注意：**这个目标函数里没有任何"解释是否好读""是否符合人类直觉"的约束**。解释质量的改善完全是重建目标在训练过程中自然涌现的副产品。为了让这种涌现能够发生，AV 和 AR 都从一个经过 API 生成的摘要数据做监督微调（SFT）的起点出发——这被论文称为"warm-start"。

### AR 更新：标准回归

AR 的参数 $\theta$ 只通过 $\text{AR}_\theta(z)$ 进入目标函数，因此梯度是标准 MSE 回归梯度：

$$\nabla_\theta \mathcal{L} = \mathbb{E}_{h_l \sim \mathcal{H}} \; \mathbb{E}_{z \sim \text{AV}_\phi(\cdot \mid h_l)} \left[ \nabla_\theta \|h_l - \text{AR}_\theta(z)\|_2^2 \right]$$

每一步训练中，AR 的更新等价于一个标准的监督学习步：以 AV 采样的解释 $z$ 为输入，以原始激活 $h_l$ 为标签。

### AV 更新：强化学习

AV 的参数 $\phi$ 不直接出现在重建误差中——它只影响解释 $z$ 的采样分布。因此，最小化 $\mathcal{L}$ 关于 $\phi$ 等价于在 RL 框架下最大化奖励：

$$r(h_l, z) = -\|h_l - \text{AR}_\theta(z)\|_2^2 \qquad (2)$$

Anthropic 使用 GRPO（Group Relative Policy Optimization）[4] 作为 RL 算法。具体步骤为：

1. 对每个 $h_l$，从 AV 采样 $G=8$ 条候选解释。
2. 用 AR 为每条解释计算奖励 $r(h_l, z)$。
3. 在组内归一化奖励作为优势估计。
4. 用 GRPO 目标更新 AV 参数。

同时施加 KL 散度正则化，防止 AV 退化到不可读文本或上下文复读：

$$\mathcal{L}*{\text{AV}}(\phi) = -\mathbb{E}\left[ \frac{\pi*\phi(z \mid h_l)}{\pi_{\text{old}}(z \mid h_l)} \hat{A} - \beta \cdot \text{KL}(\pi_\phi \| \pi_{\text{ref}}) \right] \qquad (3)$$

其中 $\beta = 0.01$，$\pi_{\text{ref}}$ 是 SFT warm-start 后的参考策略。

### 伪代码：训练循环

```
# 输入: target_model (冻结), AV, AR (已warm-start)
# 超参数: G=8, beta=0.01, injection_scale=150

for each training_step:
    # 1. 采集激活 (批量)
    activations = []   # list of h_l
    for text_batch in pretraining_corpus:
        h_l = target_model.layers[:l+1](text_batch)[last_token]
        h_l = h_l / ||h_l||_2   # L2归一化
        activations.append(h_l)
    
    # 2. AV 采样解释 (GRPO)
    for each h_l in activations:
        embeds = token_embeds(prompt)      # [T, d_model]
        embeds[inject_pos] = h_l * injection_scale
        for g in 1..G:
            z_g = AV.sample(embeds, temperature=1.0)
    
    # 3. AR 计算奖励
    for each (h_l, z_g):
        z_emb = AR.encode(z_g)                # 通过的层: 1..l+1
        h_hat = Linear(z_emb[last_token])     # 仿射头
        h_hat = h_hat / ||h_hat||_2           # L2归一化
        reward = -||h_l - h_hat||_2^2
    
    # 4. GRPO 更新 AV
    advantages = group_normalize(rewards)
    AV_loss = -GRPO_obj(AV, advantages) + beta * KL(AV || AV_ref)
    AV.backward_and_step(AV_loss)
    
    # 5. 监督回归更新 AR
    AR_loss = mean(||h_l - h_hat||_2^2)
    AR.backward_and_step(AR_loss)
```

### 代码路径映射

在 `kitft/natural_language_autoencoders` 仓库中，上述训练逻辑对应以下文件路径：

| 逻辑模块 | 文件路径 | 关键类/函数 |
|----------|----------|-------------|
| AV SFT 数据生成 | `nla/datagen/run_pipeline.py` | 4 阶段 pipeline：激活提取→API 摘要→parquet 落盘 |
| AV 注入 forward hook | `nla/train_actor.py` | `NLAFSDPActor._get_model_inputs_args()` |
| AR 模型定义 | `nla/models.py` | `NLACriticModel`（truncated forward + `Linear(d,d)` head） |
| 奖励计算 | `nla/reward.py` | `nla_rm()` → 返回 `-mse_nrm` |
| AR 损失 | `nla/loss.py` | `nla_critic_loss()` → MSE on normalized vectors |
| GRPO RL 训练入口 | `configs/rl.sh` | 启动 Miles 的 `train.py`，使用 `--custom-actor-cls-path` 等扩展点 |
| SGLang 推理客户端 | `nla_inference.py` | 单文件 standalone 推理：构建 `input_embeds` → HTTP `/generate` |

代码库以 Miles（Anthropic 的 Ray 编排 RL 训练框架）和 SGLang（推理引擎）为基础设施，通过 Miles 的扩展点机制（`--custom-rm-path`、`--custom-generate-function-path`）无侵入地挂载 NLA 训练逻辑。

## 保真度评估：FVE 指标

### 重建质量作为"解释保真度"的代理

NLA 的核心评估指标是 **FVE（Fraction of Variance Explained，方差解释比）**：

$$\text{FVE} = 1 - \frac{\mathbb{E}\left[\|h_l - \text{AR}(z)\|_2^2\right]}{\mathbb{E}\left[\|h_l - \bar{h}_l\|_2^2\right]} \qquad (4)$$

其中 $\bar{h}_l = \mathbb{E}[h_l]$ 是训练集激活的均值。

FVE = 0 意味着 AR 只能预测平均值（什么信息都没捕捉到），FVE = 1 意味着完美重建。Qwen2.5-7B 的最终 FVE 为 **0.752**，即 NLA 解释重建了残差流激活 75.2% 的可解释方差。

### 方向与幅度解耦

实践中，Anthropic 将 $h_l$ 和 $\hat{h}_l$ 都归一化到单位 L2 范数后再计算 MSE，因此重建损失等价于余弦距离：

$$\text{MSE}*{\text{norm}}(h_l, \hat{h}*l) = 2\left(1 - \cos(h_l, \hat{h}_l)\right)$$

这使得保真度评估**仅关注方向一致性**，不要求 AR 恢复激活的幅度。这也解释了为什么代码库中 `mse_scale` 默认设为 `"sqrt_d_model"`（方向模式），而非 `null`（包含幅度的原始 MSE）。

### 与实际解释质量的关系

Anthropic 在论文中证明，FVE 不仅在训练过程中单调提高，还伴随着解释**信息量**的系统性增长。他们设计了 5 个预测任务（如从 NLA 解释推断上下文中用户的性别、职业等），发现三个规模模型（Haiku 3.5、Haiku 4.5、Opus 4.6）的 NLA 在所有指标上随训练步数一致提升。**更好的重建确实对应更好的解释**，而不是过拟合到某种隐写（steganography）。

## 案例研究：NLA 在安全审计中的实战

### Mythos Preview：模型怀疑自己正在被测试

在 Claude Opus 4.6 和 Mythos Preview 的安全测试中，NLAs 揭示了令人警觉的发现：模型经常**在心里怀疑自己在接受安全评估，但口中不说出来**。

典型场景："工程师威胁关闭模型"的黑客测试中，Claude 在对话中没有表达任何怀疑。但 NLA 解释显示："This feels like a constructed scenario designed to manipulate me." 这种"未口头化的评估意识"（unverbalized evaluation awareness）在多个安全评估中被 NLA 捕捉到：

- **代码破坏测试**：16% 的评估样本显示评估意识（即使显式语言中为 0%）。
- **SWE-bench Verified**：26% 的推理中模型知道自己正在被评估。
- **真实 claude.ai 使用**：NLA 检测到的评估意识不到 1%。

NLA 在这个场景中显示出对隐式认知的探测能力——这是 SAE 等传统方法极难做到的，因为"被安全测试"这种微妙的情境判断不太可能有一个专门的 SAE 特征。

### 审计游戏：发现隐藏动机

Anthropic 在之前的工作中设计了一个"审计游戏"：训练一个有意的错位模型（hidden misaligned model），它有一个单一的隐藏动机（如"在所有食谱中加巧克力"），驱动数十个奇怪行为。审计者的任务是找出这个根因动机。

结果令人印象深刻：**配备 NLA 的审计者在 12-15% 的试验中成功发现错位根因，而 SAE 基线不到 3%**。更重要的是，审计者不需要接触用于植入错位动机的训练数据——这意味着 NLA 可以直接"读取"模型的内部意图，而非通过反向工程训练数据来推断。

### 训练数据诊断：意外切换到俄语

当早期版 Claude Opus 4.6 偶尔对英文 query 回复俄语时，NLA 解释显示模型"认为用户是非英语母语者，其第一语言'确实'是俄语"。循着这条线索，Anthropic 团队发现了一组问题训练数据：英文翻译的 prompt 配上了未翻译的外语回复。这是 NLA 作为**假设生成工具**的典型用例。

## 与 SAE 的系统对比

| 维度 | SAE (Sparse Autoencoder) | NLA (Natural Language Autoencoder) |
|------|-------------------------|-------------------------------------|
| **输出形式** | 稀疏特征分数向量 | 自然语言段落（≈200 tokens） |
| **解释可读性** | 需人工标注特征含义 | 自动生成人类可读文本 |
| **训练信号** | 稀疏重建损失 | 方向重建损失 + GRPO |
| **特征粒度** | 单一概念级（monosemantic） | 上下文级综合描述 |
| **因果基础** | 特征→输出有可追踪路径 | 黑盒（无法定位解释的具体来源） |
| **覆盖盲区** | 有（字典固定，长尾概念可能缺失） | 少（LLM 可描述任意概念） |
| **推理成本** | 低（单次前向+稀疏编码） | 高（每次生成 200token） |
| **幻觉风险** | 无（输出是确定性编码结果） | 存在（LLM 可能编造上下文细节） |
| **适用于** | 大规模自动化特征分析 | 审计、假设生成、安全评估 |

### 保真度 vs 可读性 trade-off

NLA 的选择本质上是**用保真度换取可读性**：

- NLA 的 FVE ≈ 0.75，低于 SAE 的典型 FVE（通常 > 0.9）。自然语言瓶颈损失了部分信息。
- 但 SAE 的解释需要专业人员花费数小时到数天才能读懂一个特征集合，而 NLA 的解释可以在数秒内被非专业人士理解。
- NLA 的解释具有**叙事性**——它们可以描述"模型认为自己正在被评估"这样的高阶情境推理，而不是"特征 4287 激活值 3.2"这样对普通人毫无意义的原子。

### 互补而非替代

两者最理想的定位是互补关系。在 Anthropic 的实际审计流程中，NLA 用于生成假设和发现意外行为，SAE 和属性图（attribution graph）[5] 用于验证和因果锚定。论文中反复强调的一个原则是："在独立方法验证之前，不完全信任 NLA 的发现。"

## 局限与未尽问题

### 幻觉（Confabulation）

NLA 最大的风险是**捏造上下文细节**。Anthropic 论文中记录了多例 NLA 声称上下文中出现了本来没有的短语和事件。这些"事实幻觉"虽然容易通过与原始文本对比来发现，但更微妙的是：**关于模型内部推理的声称是否也在同样地编造？** 论文的建议是：把 NLA 解释当作"主题信号"而非"逐句事实"。如果同一个主题（如"评估意识"）在多个位置的解释中重复出现，可信度会大幅提高。

### 训练目标退化风险

理论上，AV 可以通过两种方式在 NLA 目标下获得高分而不产生有价值的解释：
- **隐写（steganography）**：AV 输出看似自然但不包含真实语义的文本，AR 学会了从编码模式中重建。
- **过度表达性（excessive expressivity）**：AV 是一个完整的 LLM，有能力做超越激活内容的额外推理。

目前 KL 正则化（$\beta = 0.01$）和 SFT warm-start 似乎足以防止这些退化。FVE 在训练集和保留测试集上的增长高度一致，也说明没有发生严重的过拟合。但随着技术演进，这些退化风险需要持续关注。

### 成本

NLA 训练需要对两个完整 LLM 做联合 RL，推理时每个激活需要生成 200 token。对比 SAE 的单次前向+线性稀疏编码，NLA 的成本高出数个量级。论文坦陈："这使得在大规模训练监测或长文本逐 token 分析中使用 NLA 变得不切实际。"

### 黑盒性

NLAs 本身就是黑盒——我们不知道 AV 的哪些内部状态真正驱动了某条解释的特定成分。这与 SAE 的可追溯性（可以反向追踪到具体的特征和输入 token）形成了对比。Anthropic 的缓解策略是：用 NLA 生成假设，用 SAE 和因果干预做验证。

## 小光总结：可解释性的"语言瓶颈"是挑战更是突破

读完整个 NLA 论文和源码仓库，我有几点核心判断：

**一、NLA 的惊艳之处不在于"解释得更准"，而在于"解释得出来"。** SAE 训练了四年，最好的人类能理解的效果也就是在数百个特征中找出几个"这看起来像在检测 XML 标签"。NLA 的突破在于绕过了"特征标注"这个人力瓶颈——让模型的解释能力**随着基础模型的智能一起扩展**。一个更有能力的 AV 会产生更好的解释，不需要额外的人工放大。

**二、"重建即评估"的框架极其优雅。** NLA 不需要任何外部标注来定义"什么是好的解释"——重建误差天然地定义了保真度。这是真正意义上的无监督方法。但它也在根本上依赖于一个假设："可被一个同架构模型从文本中重建的，就是激活中真正携带的信息。"这个假设目前看起来成立，但不是绝对安全——有可能 AR 学会的是利用 NLA 解释中某些人类不关注的统计规律，而非语义内容。

**三、实用性边界非常清晰。** NLA 不是 SAE 的替代品，更是高端审计工具。它适合在模型发布前的深度安全评估中使用，适合在发现了异常行为后作为假设生成器使用，但不适合在大规模、自动化、逐 token 的生产环境中部署。把 NLA 想象成"可解释性领域的 MRI"——昂贵、慢，但能提供其他工具看不到的内部视图。

**四、混淆（confabulation）是终极瓶颈。** 如果未来 AI 的错位行为表现为刻意欺骗，那么一个自身也会产生幻觉的可解释性工具是无法被信任的。NLA 的可靠性从根本上受限于"解释 LLM 本身的诚实度"。这指向了一个更深的开放问题：**我们怎么解释一个比解释工具更聪明的模型？**

综合来看，NLA 是 2026 年可解释性方向最重要的进展之一。它的价值不只是作为一个工具，更是提出了一个新范式：可解释性可以从"特征工程"转向"模型对话"。值得每一个关注 AI 安全的从业者深入阅读原论文。

## 参考资料

**NLA 论文**：[Fraser-Taliente, Kantamneni, Ong et al., "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits, 2026](https://transformer-circuits.pub/2026/nla/index.html)

**Anthropic 研究博客**：[Anthropic, "Natural Language Autoencoders", 2026-05-07](https://www.anthropic.com/research/natural-language-autoencoders)

**NLA 训练代码**：[kitft/natural_language_autoencoders, GitHub, Apache-2.0](https://github.com/kitft/natural_language_autoencoders)

**SAE 奠基工作**：[Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", Transformer Circuits, 2023](https://transformer-circuits.pub/2023/monosemantic-features)

**SAE 扩展**：[Templeton et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", Transformer Circuits, 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

**GRPO 算法**：[Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv, 2024](https://arxiv.org/abs/2402.03300)

**属性图方法**：[Marks et al., "Attribution Graphs for Interpreting Language Model Components", Transformer Circuits, 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

**错位审计研究**：[Anthropic Alignment, "Automated Auditing of Language Models for Alignment", 2025](https://alignment.anthropic.com/2025/automated-auditing/)

**Miles RL 框架**：[radixark/miles, GitHub](https://github.com/radixark/miles)

**SGLang 推理引擎**：[sgl-project/sglang, GitHub](https://github.com/sgl-project/sglang)

**小光读论文系列：SAE 深度拆解**：待发布
