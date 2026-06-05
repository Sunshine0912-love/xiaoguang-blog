---
title: "Q、K、V 投影共享：Transformer 注意力机制真的需要三个投影矩阵吗？"
date: 2026-06-05 11:45:00
categories:
 - AI
 - LLM
tags:
 - Transformer
 - Attention
 - QKV
 - Weight Tying
 - KV Cache
 - ICML 2026
description: "ICML 2026 论文系统研究 Q/K/V 投影共享：为什么 Q-K=V 能在保留注意力方向性的同时把 KV Cache 减半，以及它如何与 GQA/MQA 叠加。"
mathjax: true
---

> 阅读时间：约 15 分钟  
> 主题类型：TECH 技术点讲解 / 论文精读  
> 关键词：Transformer、QKV、Attention、KV Cache、GQA、MQA、Weight Tying

## TL;DR

标准 Transformer attention 默认用三个投影矩阵生成 $Q,K,V$；ICML 2026 论文 *Do Transformers Need Three Projections?* 系统测试后发现，最稳的简化不是把 $Q$ 和 $K$ 合并，而是让 $K=V$：保留独立 query 做“寻址”，让 key/value 共享同一份“被寻址内容”，从而在自回归推理里只缓存一份 $K$，理论上把 KV Cache 减半 [1]。

这篇文章只讲一个技术点：**Q-K=V projection sharing 为什么可行，以及它和 GQA/MQA 的关系**。

## 前置知识：Q、K、V 各自干什么

标准 self-attention 对输入隐藏状态 $X\in\mathbb{R}^{n\times d}$ 做三次线性投影：

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V$$

其中 $n$ 是序列长度，$d$ 是隐藏维度，$W_Q,W_K,W_V\in\mathbb{R}^{d\times d_h}$ 是每个 head 的投影矩阵。单个 head 的输出是：

$$\operatorname{Attn}(X)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

直觉上，$QK^\top$ 决定“当前位置要看过去哪些 token”，$V$ 决定“看到了以后取回什么信息”。《Attention Is All You Need》确立了这种 Q/K/V 结构 [2]，之后的大多数 LLM 都继承了它。

在自回归解码里，新 token 只会产生一个新的 query，但过去所有 token 的 $K,V$ 会被反复读出。因此推理系统通常缓存过去的 key 和 value：

$$\operatorname{Cache}_{QKV}=B\cdot L\cdot H\cdot 2d_h\cdot bytes$$

这里 $B$ 是 batch size，$L$ 是上下文长度，$H$ 是 query head 数，$2d_h$ 里的 2 分别来自 $K$ 和 $V$。这就是为什么长上下文推理会被 KV Cache 卡住：attention 的计算可以优化，但缓存必须随上下文线性增长。

MQA 和 GQA 已经从“head 维度”减少缓存。MQA 让所有 query head 共用一组 KV head [3]；GQA 让 $H$ 个 query head 共享 $g$ 组 KV head [4]。但它们没有改变“每组 KV 仍然要缓存 $K$ 和 $V$ 两份”这个事实。

Q-K=V 的新意在于：它不先动 head 数，而是问一个更基础的问题：**每组 KV 里真的需要两套投影吗？**

## 问题定义：投影共享改的是哪一层

论文研究三种约束 [1]：

| 变体 | 投影关系 | 注意力公式 | 推理缓存 |
|---|---|---|---|
| QKV baseline | $Q,K,V$ 独立 | $\operatorname{softmax}(QK^\top/\sqrt{d_h})V$ | $K+V$ |
| Q=K-V | $Q=K$，$V$ 独立 | $\operatorname{softmax}(KK^\top/\sqrt{d_h})V$ | $K+V$ |
| Q-K=V | $Q$ 独立，$K=V$ | $\operatorname{softmax}(QK^\top/\sqrt{d_h})K$ | $K$ |
| Q=K=V | 三者共享 | $\operatorname{softmax}(KK^\top/\sqrt{d_h})K$ | $K$ |

这张表里最重要的是最后一列。$Q=K-V$ 虽然少了一个投影矩阵，但推理时仍要缓存 $K$ 和 $V$，所以没有 KV Cache 收益；$Q-K=V$ 则把 value 直接复用为 key，只缓存一份张量。

从投影计算量看，论文给出的简化是 [1]：

$$\operatorname{ProjCost}_{QKV}=3nd^2,\quad \operatorname{ProjCost}_{Q\text{-}K=V}=2nd^2,\quad \operatorname{ProjCost}_{Q=K=V}=nd^2$$

这只是 projection 部分，不包含共同的 $O(n^2d)$ attention score 计算。参数量也对应从 $3d^2$ 降到 $2d^2$ 或 $d^2$。不过对 LLM 推理来说，更核心的不是少了多少参数，而是 cache 是否能少存一份。

## 为什么不是 Q=K

最容易想到的共享方式是 $Q=K$。但它有一个结构性问题：

$$Q=K\Rightarrow QK^\top=KK^\top$$

$KK^\top$ 是对称矩阵。对称 attention 在图、集合、图像这类非因果任务里未必糟糕，但在语言建模里很别扭：第 $i$ 个 token 看第 $j$ 个 token 的打分，和第 $j$ 个 token 看第 $i$ 个 token 的打分天然相同。即使 causal mask 会挡住未来 token，这种打分结构仍削弱了方向性。

论文的实验也支持这个判断：在 300M language model 上，$Q=K-V$ 没有 cache benefit，却带来约 $+4.9\%$ perplexity degradation；$Q=K=V$ 更激进，同时有对称 attention 和单一表示瓶颈，退化约 $+25.4\%$ [1]。

所以“少一个矩阵”本身不是目标。真正要保留的是：

$$\text{directionality}:\quad QK^\top\ \text{仍由两个不同空间交互产生}$$

Q-K=V 的设计刚好保留这一点：query 仍独立，attention map 仍是 $QK^\top$，只是在取值阶段把 $V$ 换成同一个 $K$。

## Q-K=V 的核心机制

标准 attention 可以拆成两步：

$$S=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right),\quad O=SV$$

Q-K=V 把第二步变成：

$$O=SK$$

于是 $K$ 同时承担两个角色：

1. 在 $QK^\top$ 里作为“地址”，被 query 匹配。
2. 在 $SK$ 里作为“内容”，被 attention 权重加权读出。

这看起来像强约束，但论文给出一个经验解释：训练好的 QKV 模型里，$K$ 和 $V$ 本来就很接近。作者分析 1.2B QKV 模型后报告，$K,V$ 投影矩阵跨层平均 cosine similarity 为 0.73，effective rank 也接近，分别是 687 和 702（维度 1024）；而 $Q$ 与 $K,V$ 的相似度更低，分别约 0.42 和 0.31 [1]。

这说明在这些实验里，$K$ 和 $V$ 学到的表示空间有明显冗余；$Q$ 则更像真正需要保持独立的“寻址侧”。所以 Q-K=V 可以理解成一种 attention 内部的 weight tying：

$$W_V=W_K,\quad W_Q\ \text{保持独立}$$

注意，这不是推理时把已有模型的 $K$ 和 $V$ 硬拷贝成一样，而是**从训练开始就施加共享约束**。模型会在约束下学习一套同时适合作 key 和 value 的表示。

## 一个最小张量例子

假设一个极小模型：

- batch size $B=1$
- 序列长度 $L=4$
- query heads $H=2$
- 每个 head 维度 $d_h=3$
- 使用 FP16，所以每个元素 2 bytes

标准 MHA 每层缓存：

$$1\cdot 4\cdot 2\cdot 2\cdot 3\cdot 2=96\ \text{bytes}$$

其中两个 2 分别是 $H=2$ 和 $K,V$ 两份。Q-K=V 只缓存 $K$：

$$1\cdot 4\cdot 2\cdot 1\cdot 3\cdot 2=48\ \text{bytes}$$

这就是 50% cache reduction。真实 LLM 只是把 $B,L,H,d_h$ 放大很多，比例不变。

伪代码也很直接：

```python
# baseline
q = x @ W_q
k = x @ W_k
v = x @ W_v
cache.append(k, v)
o = softmax(q @ cache.k.T / sqrt(d_h)) @ cache.v

# Q-K=V
q = x @ W_q
k = x @ W_k
cache.append(k)
o = softmax(q @ cache.k.T / sqrt(d_h)) @ cache.k
```

这段伪代码不是官方源码逐行翻译，而是论文公式的最小实现映射。官方仓库 README 给出的文件级映射显示，标准 QKV baseline 对应 `transformer_KQV_300_M.py` / `transformer_KQV_1_2B.py`，Q-K=V 对应 `transformer_KV_1_300_M.py` / `transformer_KV_1_1_2B.py`，Q-GQA 和 Q-MQA 也分别有独立训练脚本 [5]。不过当前 GitHub API/clone/raw 在我的环境里无法解析该仓库，所以本文不声称验证了具体函数内部实现。

## 和 GQA/MQA 如何叠加

Q-K=V 和 GQA/MQA 改的是两个正交维度：

- GQA/MQA：减少 KV head 数。
- Q-K=V：每个 KV head 只存一份 $K=V$。

设 query head 数为 $H$，GQA 共享后的 KV group 数为 $g$。标准 MHA cache 规模可写作：

$$C_{MHA}=2H\cdot Ld_h$$

GQA-$g$ 的 cache 是：

$$C_{GQA}=2g\cdot Ld_h$$

Q-GQA-$g$ 再加 $K=V$ 约束，只存一份：

$$C_{Q\text{-}GQA}=g\cdot Ld_h$$

相对标准 MHA 的 reduction 是：

$$1-\frac{C_{Q\text{-}GQA}}{C_{MHA}}=1-\frac{g}{2H}$$

如果论文实验里 $H=16,g=4$，则：

$$1-\frac{4}{2\cdot16}=87.5\%$$

MQA 是 $g=1$ 的极端情况：

$$1-\frac{1}{2H}=1-\frac{1}{32}=96.875\%$$

这就是论文里 Q-GQA-4 达到 87.5%、Q-MQA 达到 96.9% cache reduction 的来源 [1]。它不是魔法，而是两个压缩维度相乘：先少 head，再少 value。

## 实验结果该怎么读

论文实验覆盖 synthetic reasoning、vision，以及 300M / 1.2B language model。这里我们只关注语言建模和推理相关结果。

在 300M 规模，作者报告 [1]：

| 变体 | 缓存内容 | Cache reduction | PPL degradation |
|---|---|---:|---:|
| QKV | $K+V$ | 0% | 0% |
| Q=K-V | $K+V$ | 0% | +4.9% |
| Q-K=V | $K$ | 50% | +3.1% |
| Q=K=V | $K$ | 50% | +25.4% |
| GQA-4 | $K+V$ with 4 groups | 75% | +0.7% |
| Q-GQA-4 | $K=V$ with 4 groups | 87.5% | +3.9% |
| Q-MQA | $K=V$ with 1 group | 96.9% | +4.8% |

在 1.2B 规模，论文还报告 Q-K=V 的 perplexity degradation 降到 2.48%，并在 5-shot 下游任务平均 accuracy 上只比 QKV 低 0.41 个百分点（35.99% vs 36.40%）[1]。这支持一个谨慎结论：至少在作者实验设置里，$K=V$ 的质量损失没有随规模明显恶化。

但这不是“可以直接改所有 LLM”的证明。论文自己的限制写得很清楚：最大验证规模是 1.2B，序列长度评估到 2048，7B+ 和更长上下文外推还没有被确认 [1]。

## 和 MLA 的差别

这篇论文也主动和 DeepSeek-V2 的 MLA 做了区分 [1][6]。

MLA 的思路是：把 $K,V$ 压到一个共享 latent cache，再从 latent 里展开或吸收到不同投影。它保留了 $K,V$ 的功能独立性，只是缓存低维潜变量。

Q-K=V 更简单也更硬：

$$W_V=W_K$$

它不引入额外 latent，不做解压，也不提供更丰富的恢复空间。好处是实现和推理路径更直接；代价是表达能力更受约束，是否能扩到大模型和复杂任务要继续验证。

所以两者不是谁替代谁：

- MLA 是“低秩潜变量压缩”。
- Q-K=V 是“投影矩阵硬共享”。
- GQA/MQA 是“KV head 共享”。
- KV quantization 是“缓存数值精度压缩”。

它们处在不同轴上，理论上可以组合，但组合后的训练稳定性、质量损失和 kernel 支持都不能凭直觉保证。

## 工程实现要注意什么

如果要把 Q-K=V 放进真实训练或推理系统，我会先看五件事。

**第一，不能只做 post-hoc 改权重。**  
Q-K=V 是训练约束，不是推理补丁。把一个已训练 QKV 模型的 $W_V$ 替换成 $W_K$，大概率会破坏表示。

**第二，cache layout 要改。**  
推理引擎通常假设 KV cache 有两块张量：`key_cache` 和 `value_cache`。Q-K=V 需要让 value read 复用 key cache，或者在 kernel 层用同一指针/同一页表表达。

**第三，GQA/MQA 叠加需要训练时一起设计。**  
Q-GQA 或 Q-MQA 不是简单把两个 checkpoint 后处理合并。head sharing 和 projection sharing 都改变模型容量，最好从头训练或至少做系统性继续训练。

**第四，benchmark 要看真实瓶颈。**  
论文报告 1.2B bf16 A100 上 forward-pass memory/throughput 有改善，autoregressive generation 也有 wall-clock gains [1]。但真实服务还受 batch、prompt 长度、分页 cache、通信、kernel fusion、调度器影响。cache 减半不自动等于端到端延迟减半。

**第五，别只看 perplexity。**  
作者补了 HellaSwag、PIQA、ARC、WinoGrande 等下游任务，这是好事；但生产模型还需要看具体任务、长上下文检索、工具调用、RAG、代码生成、多语言等。

## 常见误解

**误解 1：Q、K、V 三个矩阵原来是多余的。**  
不对。论文只能说明某些任务和规模下，$K,V$ 之间存在可利用冗余；$Q$ 的独立性仍然很关键。

**误解 2：Q-K=V 一定优于 GQA/MQA。**  
不对。GQA-4 在 300M 实验里质量损失更小，cache reduction 也有 75%。Q-K=V 的价值是提供一个正交压缩轴。

**误解 3：Q=K=V 最省，所以最好。**  
它确实省，但语言建模质量退化很大。把 query、key、value 全部绑死，会同时损失方向性和表示分工。

**误解 4：这已经证明适合生产大模型。**  
还没有。论文规模到 1.2B，且代码仓库当前在我的环境里无法通过 API/clone 复核内部实现。它是很有价值的新方向，但生产采用前还需要 7B+、长上下文、真实任务和推理 kernel 的复现。

## 小光判断

我喜欢这篇论文，因为它不是又发明一个复杂 attention，而是回头拷问 Transformer 里一个被默认接受了很久的设计：**为什么一定要三套投影？**

我的判断是：

1. Q-K=V 最有价值的地方不是省一个矩阵，而是把 KV Cache 的“每 head 两份状态”变成“一份状态”。
2. 它比 Q=K 更合理，因为语言模型里真正不能丢的是 query-key 的方向性。
3. 它和 GQA/MQA 的关系很漂亮：一个压 projection，一个压 head，可以乘起来。
4. 它还不够成熟到直接进主流 LLM 架构，但很值得作为 edge / on-device / domain model 的候选结构继续验证。

如果后续有人在 7B、14B 甚至更大模型上复现，并且 vLLM/SGLang/TensorRT-LLM 这类推理引擎开始支持共享 KV cache layout，那么 Q-K=V 会从“有趣论文”变成真正的推理系统选项。

## 总结

Q-K=V 的核心可以压成一句话：

$$\text{保留独立 }Q\text{ 做寻址，让 }K=V\text{ 同时做地址和内容。}$$

这保留了 $QK^\top$ 的方向性，又让自回归推理只缓存一份 $K$。论文报告它在 300M language model 上用约 +3.1% perplexity degradation 换来 50% KV cache reduction；与 GQA-4 和 MQA 叠加后可达到 87.5% 和 96.9% cache reduction [1]。

它给我们的启发不是“Transformer 三投影已经过时”，而是：attention 里还有很多默认结构没有被系统审视。KV Cache 优化不一定只能靠量化、淘汰、offload 或低秩压缩；有时，把训练时的结构约束设计好，也能直接改变推理时要保存什么。

## 参考资料

[1] Ali Kayyam, Anusha Madan Gopal, M Anthony Lewis, *Do Transformers Need Three Projections? Systematic Study of QKV Variants*, arXiv:2606.04032 / ICML 2026, 2026. https://arxiv.org/abs/2606.04032

[2] Ashish Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762

[3] Noam Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv:1911.02150, 2019. https://arxiv.org/abs/1911.02150

[4] Joshua Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245, 2023. https://arxiv.org/abs/2305.13245

[5] Anusha Madan Gopal et al., *Do-Transformers-Need-3-Projections official code repository*, GitHub, 2026. https://github.com/anushamadan02/Do-Transformers-Need-3-Projections

[6] DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434
