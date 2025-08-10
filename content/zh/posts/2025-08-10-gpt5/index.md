---
title: "gpt-oss & GPT-5 (更新中)"
date: 2025-08-10T12:00:00+08:00
author: "Yue Shui"
tags: ["gpt-oss", "GPT-5",  "MoE", "Reasoning", "Tool Use", "OpenAI", "LLM Architecture"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

> **注意**: 本文正在更新中，内容只是**草稿版本，并不完善，后续会有较大变动**。请随时关注最新版本。

2025 年 8 月，人工智能领域迎来了 OpenAI 的密集发布期。继 2019 年 GPT-2 之后，OpenAI 再次向开源社区贡献了其首个开放权重的大型语言模型系列 **gpt-oss** ([OpenAI, 2025a](https://openai.com/index/introducing-gpt-oss/))，包含 120B 和 20B 两种规模。紧随其后，备受瞩目的下一代旗舰模型 **GPT-5** ([OpenAI, 2025b](https://openai.com/index/introducing-gpt-5/)) 也正式发布。这一系列发布不仅标志着开源模型在推理和智能体（Agentic）能力上达到了新的高度，也揭示了 OpenAI 在模型架构、训练方法论以及安全对齐方面的最新进展。

## gpt-oss

gpt-oss 是 OpenAI 自 GPT-2 以来首次发布的开放权重语言模型，旨在为开源社区提供强大的推理和工具使用能力。该系列包含 `gpt-oss-120b` 和 `gpt-oss-20b` 两个版本，均在 Apache 2.0 许可下发布。

### 架构概览

{{< figure
    src="gpt_oss_vs_gpt2.png"
    caption="Fig. 1. A side-by-side comparison between gpt-oss-20b and GPT-2 XL 1.5B. (Image source: [Raschka, 2025](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the))"
    align="center"
    width="100%"
>}}

gpt-oss 建立在 GPT 系列架构之上，并融合了近年来多项主流技术包括 [RMSNorm](https://syhya.github.io/zh/posts/2025-02-01-normalization/#rms-normalization)、[SwiGLU](https://syhya.github.io/zh/posts/2025-04-06-llama/#ffn_swiglu)、[GQA](https://syhya.github.io/posts/2025-01-16-group-query-attention/#grouped-query-attention-gqa)、[RoPE](https://syhya.github.io/zh/posts/2025-04-06-llama/#rotary-positional-embeddings-rope)、[YaRN](https://syhya.github.io/zh/posts/2025-04-18-deepseek-v2-v3/)、[MoE](https://syhya.github.io/posts/2025-03-01-train-llm/#mixture-of-experts-model) 等。


下表直观对比了 GPT-OSS 20B vs GPT-2 XL 1.5B 模型差异。

| **特性** | **GPT-OSS 20B (2025)** | **GPT-2 XL 1.5B (2019)** |
|----------|------------------------|--------------------------|
| **发布时间** | 2025年 | 2019年 |
| **模型大小** | **20B 参数** | 1.5B 参数 |
| **活跃参数** | **3.5B** (每次推理) | 1.5B (全部激活) |
| **词汇表大小** | **200k tokens** | 50k tokens |
| **嵌入维度** | **2,880** | 1,600 |
| **Transformer层数** | 24层 | **48层** |
| **注意力头数** | **64个** | 25个 |
| **支持上下文长度** | **131k tokens** | 1,024 tokens |
| **位置编码** | **RoPE** (旋转位置编码) | 绝对位置嵌入 |
| **注意力机制** | **Grouped Query Attention** | Multi-head Attention |
| **前馈网络** | **SwiGLU激活 + MoE** | GELU激活 |
| **MoE架构** | **32个专家，4个激活** | 无 |
| **归一化方法** | **RMSNorm** (2处) | LayerNorm (2处) |
| **Dropout** | **无** | 有 |
| **滑动窗口注意力** | **每隔一层使用**<br>(窗口128 tokens) | 无 |
| **训练特点** | **包含监督微调+强化学习** | 仅预训练 |
| **量化支持** | **MXFP4** (可在单GPU运行) | 无特殊量化 |
| **许可证** | Apache 2.0 | MIT |


### 高效注意力机制

为了在支持 128k 长上下文的同时保持高效率，gpt-oss 采用了多种先进的注意力机制。

*   **分组查询注意力 (Grouped-Query Attention, GQA)**: gpt-oss 中有 64 个查询头和 8 个键值头，意味着每 8 个查询头共享一对 K/V，这显著减少了 KV 缓存的大小和内存带宽需求，从而在几乎不损失模型性能的情况下，大幅提升了推理吞吐量。

*   **滑动窗口注意力 (Sliding Window Attention)**: 为了进一步降低计算复杂度，gpt-oss 借鉴了 Longformer([Jiang et al., 2020](https://arxiv.org/abs/2004.05150)) 和 Mistral([Jiang et al., 2023](https://arxiv.org/abs/2310.06825))的思想，采用了**交替的注意力模式**。其 Transformer 层在**全注意力 (Dense Attention)** 和**局部带状稀疏注意力 (Locally Banded Sparse Attention)** 之间交替。后者即**滑动窗口注意力**，它将每个 Token 的注意力范围限制在一个固定大小的局部窗口内。

{{< figure
    src="sliding_window_attention.png"
    caption="Fig. 2. Comparison between regular attention (left) and sliding window attention (right). (Image source: [Jiang et al., 2023](https://arxiv.org/abs/2310.06825))"
    align="center"
    width="80%"
>}}

在 gpt-oss 中，这个窗口大小被设置为 128 个 Token。这意味着，在一个局部注意力层中，一个 Token 只能关注其前面 128 个 Token，而不是整个上下文。这种设计使得注意力的计算复杂度从 \( O(L^2) \) 降低到 \( O(L \cdot W) \)，其中 \( L \) 是序列长度，\( W \) 是窗口大小。通过与全注意力层交替，模型既能高效处理局部信息，又能通过全注意力层整合全局信息。


*   **注意力池**: 模型引入了**注意力池（Attention Sinks）** ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453))，通过学习一个附加到注意力分数上的偏置 \( \mathbf{s}_h \)，使得初始 Token 能够被持续关注，这有助于在长上下文场景下稳定注意力分布，防止信息丢失。

\[ \text{Attention}(Q, K, V)_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}} + \mathbf{s}_h\right)V_h \]

{{< figure
    src="StreamingLLM.png"
    caption="Fig. 3. IllustrationofStreamingLLMvs.existingmethods.. (Image source: [Xiao et al., 2023](https://arxiv.org/abs/2309.17453))"
    align="center"
    width="100%"
>}}

上图比较了 **StreamingLLM** 与三种常见长文本处理方法在性能和效率上的差异。假设语言模型在预训练时只见过长度为 $L$ 的文本，而推理时需要预测第 $T$ 个 token（其中 $T \gg L$）：

1. **密集注意力（Dense Attention）**：保留所有历史 token 的键值（KV）并计算全量注意力，时间复杂度为 $O(T^2)$，缓存规模持续增长。当输入长度超过预训练长度 $L$ 时，性能明显下降。
2. **窗口注意力（Window Attention）**：只缓存最近 $L$ 个 token 的 KV，推理效率高，但一旦早期 token 的信息被替换，性能会急剧下降。
3. **滑动窗口重计算（Sliding Window with Re-computation）**：在每次生成新 token 时，从最近 $L$ 个 token 重建 KV 状态。尽管在长文本上性能较好，但因重计算涉及二次注意力，时间复杂度高达 $O(TL^2)$，推理速度很慢。

这种方法在计算时结合了注意力池和最近的 token，不仅保持了推理效率，还能在长文本场景下维持稳定的注意力分布和较低的困惑度。

### MXFP4 量化

{{< figure
    src="mxfp4.png"
    caption="Fig. 4. Faster MXFP4 Backpropagation via Stochastic Rounding and Hadamard Transform. (Image source: [Tseng et al., 2025](https://arxiv.org/abs/2502.20586))"
    align="center"
    width="60%"
>}}

为了让大模型能在消费级硬件上运行，gpt-oss 采用了 **MXFP4** ([Tseng et al., 2025](https://arxiv.org/abs/2502.20586)) 格式对 MoE 权重进行量化。MXFP4 是一种微缩放浮点格式，可以将权重有效量化到约 4.25 bit。由于 MoE 权重占模型总参数的 90% 以上，此方法极大地压缩了模型大小，使得 120B 模型能装入单个 80GB GPU，20B 模型能在 16GB 显存的设备上运行。

### 训练

*   **预训练**: 模型在数万亿 Token 的文本数据集上进行训练，数据侧重于 STEM、编码和通用知识。为了提升安全性，预训练数据复用了 GPT-4o 的 CBRN 内容过滤器。

*   **后训练 (推理与工具使用)**: 预训练后，模型采用与 OpenAI o3 类似的 **CoT RL** 技术进行后训练。这一阶段的目标是教会模型：
    1.  **推理**: 生成详细的思维链（Chain-of-Thought, CoT）来解决复杂问题。
    2.  **工具使用**: 学会调用外部工具（如网页搜索、代码执行）来增强自身能力。

为了实现这些高级智能体功能，OpenAI 设计了[Harmony Chat Format](https://github.com/openai/harmony)。该格式引入了“通道（channels）”概念（如 `analysis` 用于 CoT，`commentary` 用于工具调用，`final` 用于最终答案），并建立了严格的指令层级（System > Developer > User > Assistant > Tool），确保模型行为的可控性。

此外，模型还支持**可变努力推理（Variable Effort Reasoning）**。用户可以在系统提示中设置 `Reasoning: low/medium/high`，从而在延迟和性能之间进行权衡。高努力度会生成更长的 CoT，通常带来更高的准确率，但延迟也相应增加。

{{< figure
    src="reasoning_efforts.png"
    caption="Fig. 5. Accuracy vs. average CoT length for different reasoning levels on AIME and GPQA benchmarks. (Image source: [OpenAI, 2025](https://openai.com/index/gpt-oss-120b-and-gpt-oss-20b-model-card/))"
    align="center"
    width="100%"
>}}

## GPT-5

与 gpt-oss 不同，GPT-5 并非单个模型，而是一个**统一的智能系统**。它并非一个单一的庞大模型，而是一个由多个专业模型和智能路由机制协同工作的复杂系统，旨在平衡性能、速度与成本。

### 系统架构

GPT-5 系统由以三个核心部分组成：

1.  **gpt-5-main**: 作为系统的默认模型，它快速、高效，负责处理绝大多数用户查询。可视为 GPT-4o 的继任者。
2.  **gpt-5-thinking**: 用于处理更复杂、需要深度思考的问题。当用户明确要求（如“think hard about this”）或系统判断任务需要时，该模型会被激活。可视为 OpenAI o3 的继任者。
3.  **实时路由器 (Real-time Router)**: 这是一个持续训练的决策模型，路由器是一个实时决策模型，它能根据多种信号快速判断应将用户请求分配给哪个模型处理。其决策依据包括：
*   **对话类型 (Conversation Type):** 是闲聊、问答还是任务导向型对话。
*   **复杂性 (Complexity):** 问题的难度和所需的推理深度。
*   **工具需求 (Tool Needs):** 是否需要调用网页搜索、代码解释器等外部工具。
*   **用户意图 (Explicit Intent):** 用户可以通过明确的指令（如“think hard about this”）来引导路由器选择深度推理模型。

该路由器通过持续学习真实用户信号（如用户切换模型的行为、响应的偏好率、答案的实测正确率）来不断优化其决策能力。

### 安全补全

传统的安全训练范式是**硬性拒绝 (Hard Refusals)**，即模型根据用户意图的二元分类（安全或不安全）来决定是完全回答还是直接拒绝。这种方法在处理意图明确的恶意提示时有效，但在面对意图模糊或涉及**双重用途 (Dual-use)**（如生物、网络安全）的领域时，显得非常脆弱，常常导致过度拒绝。

**安全补全 (Safe Completions)** 不再对用户意图进行二元分类，而是致力于在遵守安全策略的前提下，最大化模型的帮助性。

*   **对于明确有害的请求**：模型依然会拒绝。
*   **对于双重用途的请求**（如生物、化学、网络安全领域）：模型会提供安全的、高层次的、不包含可直接执行细节的回答，而不是完全拒绝。
*   **对于模糊意图的请求**：模型会尝试以安全的方式完成任务，或提供安全的替代方案。

这种方法显著提升了模型在双重用途领域的安全性和实用性，并减少了不必要的过度拒绝。

{{< figure
    src="safe_completions.png"
    caption="Fig. 6. Left: Overall structure of the safe-completion training stack. Right: Details of the safecompletion reward design.. (Image source: [OpenAI, 2025](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cdn.openai.com/pdf/be60c07b-6bc2-4f54-bcee-4141e1d6c69a/gpt-5-safe_completions.pdf))"
    align="center"
    width="100%"
>}}


### 思维链监控

为了确保推理模型的可靠性、安全性以及避免出现 reward hacking，OpenAI 采用 **思维链监控 (CoT Monitoring)** 原则和方法。与一些试图通过监督微调来“美化”CoT 的做法不同，GPT-5 的 CoT 训练没有施加直接的对齐压力。这使得 CoT 能够更真实地反映模型的“思考”过程，成为一个有效的窗口，用于检测模型的错误行为、欺骗意图或潜在的风险。

{{< figure
    src="monitor_frontier_reasoning.png"
    caption="Fig. 7. Monitoring Frontier Reasoning Models for Reward Hacking. (Image source: [Baker et al., 2025](https://arxiv.org/abs/2503.11926))"
    align="center"
    width="100%"
>}}

通过 CoT 监控，OpenAI 发现 `gpt-5-thinking` 的欺骗行为发生率从 o3 的 4.8% 降低到了 2.1%。这项技术对于理解和缓解高级 AI 系统的风险至关重要。

## 参考文献

[1] Raschka, S. (2025). ["From GPT-2 to gpt-oss: Analyzing the Architectural Advances."](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) Ahead of AI.

[2] OpenAI. (2025). ["Introducing gpt-oss."](https://openai.com/index/introducing-gpt-oss/) OpenAI Blog.

[3] OpenAI. (2025). ["Introducing GPT-5."](https://openai.com/index/introducing-gpt-5/) OpenAI Blog.

[4] OpenAI. (2025). ["gpt-oss-120b & gpt-oss-20b Model Card."](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)

[5] Xiao, G., et al. (2023). ["Efficient Streaming Language Models with Attention Sinks."](https://arxiv.org/abs/2309.17453) arXiv preprint arXiv:2309.17453.

[6] OpenAI. (2025). ["GPT-5 System Card."](https://openai.com/index/gpt-5-system-card/)

[7] OpenAI. (2025). ["Introducing GPT-5 for developers."](https://openai.com/index/introducing-gpt-5-for-developers/) OpenAI Blog.

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (August 2025). gpt-oss & GPT-5.
> https://syhya.github.io/zh/posts/2025-08-10-gpt5-gpt-oss-analysis/

Or

```bibtex
@article{yue_shui_gpt5_gpt_oss_2025,
  title   = "gpt-oss & GPT-5",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "August",
  url     = "https://syhya.github.io/zh/posts/2025-08-12-gpt5-gpt-oss-analysis/"
}
```