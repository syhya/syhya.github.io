---
title: "深入解析OpenAI GPT-5：架构、技术与安全"
date: 2025-08-07T12:00:00+08:00
author: "Yue Shui"
tags: ["GPT-5", "OpenAI", "MoE", "RL", "Safe-Completions", "CoT", "Preparedness Framework"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

2025 年 8 月 7 日，OpenAI 发布了其最新一代旗舰模型 **GPT-5** ([OpenAI, 2025a](https://openai.com/index/introducing-gpt-5/))，标志着人工智能领域又一次重大飞跃。GPT-5 不仅仅是单一模型的性能提升，而是一个集成了快速响应模型、深度推理模型和实时路由器的统一智能系统。它在编码、数学、写作、健康和视觉感知等多个领域树立了新的技术标杆，同时在减少幻觉、提升指令遵循能力和安全性方面取得了显著进展。本文将深入剖析 GPT-5 背后的核心技术、创新的训练方法以及全面的安全保障体系，为您揭示其强大能力的技术基石。

## 符号表

下面列举了文章中使用的关键数学公式符号及其含义，以帮助你更轻松地阅读。

| 符号 | 说明 |
| :--- | :--- |
| \( x \) | 输入序列（例如，文本 token 序列） |
| \( E_i(x) \) | 第 \( i \) 个专家网络对输入 \( x \) 的处理结果 |
| \( G(x) \) | 门控网络（Gating Network），用于计算专家的权重 |
| \( G(x)_i \) | 门控网络为第 \( i \) 个专家生成的权重（标量） |
| \( y \) | MoE 层的最终输出 |
| \( N \) | 专家网络的总数量 |
| \( k \) | 每次前向传播中被激活的专家数量（Top-k） |
| \( \theta, \phi \) | 模型的可训练参数集合 |
| \( \pi_{\theta}(a|s) \) | 策略模型，在状态 \( s \) 下采取动作 \( a \) 的概率 |
| \( V_{\phi}(s) \) | 价值函数，评估状态 \( s \) 的价值 |
| \( r_t \) | 在时间步 \( t \) 获得的奖励 |
| \( A_t \) | 优势函数（Advantage Function），\( A_t = Q(s_t, a_t) - V(s_t) \) |
| \( \mathcal{L}_{CLIP}(\theta) \) | PPO 算法中的裁剪替代目标函数 |
| \( \epsilon \) | PPO 算法中用于裁剪概率比率的超参数 |
| \( \mathcal{L}_{VF}(\phi) \) | 价值函数损失 |
| \( S(\theta) \) | 策略的熵，用于鼓励探索 |
| \( c_1, c_2 \) | 价值函数损失和熵奖励的系数 |
| \( \mathcal{L}^{PPO}(\theta, \phi) \) | PPO 算法的总损失函数 |
| \( \mathbf{x}_m, \mathbf{x}_n \) | 旋转位置编码中的位置 \( m \) 和 \( n \) 的嵌入向量 |
| \( \mathbf{q}, \mathbf{k} \) | 注意力机制中的查询（Query）和键（Key）向量 |
| \( \mathbf{R}_{\Theta, m} \) | 应用于位置 \( m \) 的旋转矩阵 |
| \( \theta_i \) | RoPE 中用于不同维度对的旋转角度 |
| \( d \) | 嵌入向量的维度 |
| CoT | 思维链（Chain of Thought） |

## GPT-5 统一系统架构

与前代模型不同，GPT-5 的核心创新之一在于其**统一系统架构 (Unified System Architecture)**。它并非一个单一的庞大模型，而是一个由多个专业模型和智能路由机制协同工作的复杂系统，旨在平衡性能、速度与成本。

{{< figure
    src="gpt5_unified_system.png"
    caption="Fig. 1. The Unified Architecture of the GPT-5 System, integrating a real-time router with specialized models for optimal performance. (Image source: Mathpix from [OpenAI, 2025a](https://openai.com/index/introducing-gpt-5/))"
    align="center"
    width="80%"
>}}

该系统主要由以下几个部分组成：

1.  **核心模型 (Main Models):**
    *   **`gpt-5-main`**: 一个高效、智能的主力模型，作为 GPT-4o 的继任者，负责处理绝大多数用户查询。它在速度和智能之间取得了极佳的平衡。
    *   **`gpt-5-main-mini`**: `main` 模型的小型化版本，用于在用量达到限制后处理剩余的查询，保证服务的连续性。

2.  **深度推理模型 (Thinking Models):**
    *   **`gpt-5-thinking`**: 专为解决复杂、困难问题设计的深度推理模型，是 OpenAI o3 的继任者。该模型被训练用于在响应前生成一个长的内部**思维链 (Chain of Thought, CoT)**，从而进行更深入的分析和推理。
    *   **`gpt-5-thinking-mini` / `nano`**: `thinking` 模型的小型化版本，分别对应 o4-mini 和 GPT-4.1-nano，为开发者在 API 中提供了不同性能和成本的选择。
    *   **`gpt-5-thinking-pro`**: `thinking` 模型的一个特殊变体，通过利用**并行测试时计算 (Parallel Test Time Compute)** 来扩展思考时间，为最复杂的任务提供最高质量的答案。

3.  **实时路由器 (Real-time Router):**
    这是整个系统的“大脑”。路由器是一个实时决策模型，它能根据多种信号快速判断应将用户请求分配给哪个模型处理。其决策依据包括：
    *   **对话类型 (Conversation Type):** 是闲聊、问答还是任务导向型对话。
    *   **复杂性 (Complexity):** 问题的难度和所需的推理深度。
    *   **工具需求 (Tool Needs):** 是否需要调用网页搜索、代码解释器等外部工具。
    *   **用户意图 (Explicit Intent):** 用户可以通过明确的指令（如“think hard about this”）来引导路由器选择深度推理模型。

    该路由器通过持续学习真实用户信号（如用户切换模型的行为、响应的偏好率、答案的实测正确率）来不断优化其决策能力。

这种分层、路由的架构设计带来了多重优势：
*   **效率与性能兼顾:** 简单的请求由快速模型处理，降低延迟和成本；复杂的请求则交由强大的推理模型，保证回答质量。
*   **资源优化:** 避免了所有请求都动用最大模型的计算资源，实现了计算力的按需分配。
*   **用户体验提升:** 用户无需手动选择模型，系统能智能匹配最合适的模型，提供流畅且高质量的交互体验。

OpenAI 表示，未来计划将这些能力整合到一个单一模型中，这预示着未来的模型架构将更加动态和自适应。

## 核心模型技术解析

尽管 OpenAI 并未完全公开 GPT-5 的内部架构，但结合其发布的 `gpt-oss` 开源模型 ([OpenAI, 2025d](https://openai.com/index/introducing-gpt-oss/)) 的技术细节和 GPT-5 系统卡中的描述，我们可以推断其采用了当前最前沿的技术。

### 混合专家模型 (Mixture-of-Experts, MoE)

现代大规模语言模型普遍采用 MoE 架构以在控制计算成本的同时扩展模型参数量。`gpt-oss` 模型明确采用了 MoE，GPT-5 极有可能也基于此架构。

MoE 的核心思想是将一个大的前馈网络 (FFN) 层替换为多个并行的、规模较小的“专家”网络和一个“门控网络”。对于每个输入的 token，门控网络会动态地选择一小部分（通常是 Top-k 个）最相关的专家来处理它。

{{< figure
    src="moe_architecture.png"
    caption="Fig. 2. Mixture-of-Experts (MoE) Layer Architecture. (Image source: Mathpix)"
    align="center"
    width="70%"
>}}

一个 MoE 层的计算过程可以表示为：
\[ y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x) \]
其中：
*   \( x \) 是输入到 MoE 层的 token 表示。
*   \( E_i(\cdot) \) 是第 \( i \) 个专家网络（通常是一个 FFN）。
*   \( N \) 是专家的总数。
*   \( G(x) \) 是门控网络（通常是一个简单的线性层后接 Softmax），它为每个专家生成一个权重。\( G(x)_i \) 是分配给第 \( i \) 个专家的权重。

在实际应用中，为了实现稀疏激活并控制计算量，通常采用 **Top-k 门控 (Top-k Gating)**。门控网络会输出一个 logits 向量，只选择得分最高的 \( k \) 个专家，并将其余专家的权重设为零。例如，`gpt-oss-120b` 模型拥有 128 个专家，但每个 token 只激活其中的 4 个。

**优势:**
*   **参数扩展:** 可以在不显著增加每次前向传播计算量（FLOPs）的情况下，极大地增加模型的总参数量。
*   **专业化:** 不同的专家可以在训练中学习处理不同类型的数据或模式，实现知识的专业化分工。

### 高效注意力机制

处理长序列是现代 LLM 的核心能力。为了在支持长达数十万 token 上下文的同时保持计算效率，GPT-5 可能采用了多种先进的注意力机制。

1.  **分组查询注意力 (Grouped-Query Attention, GQA):**
    GQA 是对多头注意力 (MHA) 和多查询注意力 (MQA) 的一种折中。在 MHA 中，每个查询头 (Query Head) 都有自己独立的键 (Key) 和值 (Value) 头。在 MQA 中，所有查询头共享同一对键/值头。GQA 则将查询头分成若干组，组内共享同一对键/值头。这在保持 MHA 高质量的同时，显著减少了推理时键/值缓存的内存占用，从而加快了生成速度。

2.  **稀疏注意力模式 (Sparse Attention Patterns):**
    `gpt-oss` 提到了“交替使用密集和局部带状稀疏注意力模式”，这与 GPT-3 论文中的描述类似。这意味着模型可能并非在所有层都使用全局注意力，而是在某些层中使用计算成本更低的稀疏注意力，例如只关注局部窗口内的 token 或按特定步长关注的 token，从而在处理长序列时降低计算复杂度。

### 旋转位置编码 (Rotary Position Embedding, RoPE)

RoPE ([Su et al., 2024](https://arxiv.org/abs/2104.09864)) 是一种将相对位置信息融入自注意力机制的高效方法。与传统的绝对或可学习位置编码不同，RoPE 通过旋转矩阵来编码位置信息。

其核心思想是，对于位置为 \( m \) 的 token 嵌入 \( \mathbf{x}_m \)，其查询向量 \( \mathbf{q}_m \) 和键向量 \( \mathbf{k}_m \) 会通过一个与位置 \( m \) 相关的旋转矩阵 \( \mathbf{R}_{\Theta, m} \) 进行变换。两个不同位置 \( m \) 和 \( n \) 的 token 之间的注意力得分，只取决于它们的相对位置 \( m-n \)，而与绝对位置无关。

具体来说，对于一个 \( d \) 维的嵌入向量，可以将其视为 \( d/2 \) 个二维向量。对于第 \( i \) 个二维向量 \( [x_i, x_{i+1}] \)，在位置 \( m \) 的旋转操作如下：
\[ \begin{pmatrix} x'_i \\ x'_{i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix} \]
其中 \( \theta_i = 10000^{-2i/d} \) 是预设的旋转角度。

经过 RoPE 变换后，两个位置 \( m \) 和 \( n \) 的查询和键向量的点积 \( (\mathbf{R}_{\Theta, m}\mathbf{q})^T(\mathbf{R}_{\Theta, n}\mathbf{k}) \) 可以被证明只依赖于相对位置 \( m-n \)。

**优势:**
*   **良好的外推性:** RoPE 对长于训练序列的上下文具有天然的外推能力。
*   **实现简单:** 易于集成到现有的 Transformer 架构中。

## 训练方法与安全创新

GPT-5 的卓越性能不仅源于其先进的架构，更得益于其创新的训练方法和对安全性的高度重视。

### 基于强化学习的推理训练

`gpt-5-thinking` 模型的核心能力在于其深度推理。系统卡明确指出，这类模型通过**强化学习 (Reinforcement Learning, RL)** 被训练成“先思考，后回答”。

{{< figure
    src="cot_rl.png"
    caption="Fig. 3. Reinforcement Learning for Chain-of-Thought Reasoning. (Image source: Mathpix)"
    align="center"
    width="90%"
>}}

这个过程可以概括为：
1.  **生成思维链 (CoT):** 当模型接收到一个复杂问题时，它首先会生成一个内部的、非公开的思维链，详细阐述其解决问题的步骤、假设和中间结论。
2.  **评估与奖励:** 这个思维链和最终答案会被一个**奖励模型 (Reward Model)** 或其他评估机制进行评估。奖励不仅仅基于最终答案的正确性，更重要的是评估推理过程的逻辑性、一致性和效率。
3.  **策略优化:** 模型使用强化学习算法（如 **近端策略优化 Proximal Policy Optimization, PPO**）来更新其参数，以最大化期望奖励。

PPO 的目标函数通常形式如下：
\[ \mathcal{L}^{PPO}(\theta) = \mathbb{E}_t \left[ \mathcal{L}_{CLIP}(\theta) - c_1 \mathcal{L}_{VF}(\phi) + c_2 S(\pi_{\theta})(s_t) \right] \]
其中：
*   \( \mathcal{L}_{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) \right] \) 是裁剪后的策略目标，\( r_t(\theta) \) 是新旧策略的比率，\( A_t \) 是优势函数。
*   \( \mathcal{L}_{VF}(\phi) \) 是价值函数的损失。
*   \( S(\pi_{\theta}) \) 是策略的熵，用于鼓励探索。

通过这种方式，模型学会了如何进行更有效、更可靠的推理，识别并修正自己的错误，并遵循特定的指导方针和安全策略。

### 从硬性拒绝到安全补全 (Safe-Completions)

这是 GPT-5 在安全训练方面的一项重大革新。传统的安全训练范式是**硬性拒绝 (Hard Refusals)**，即模型根据用户意图的二元分类（安全或不安全）来决定是完全回答还是直接拒绝。这种方法在处理意图明确的恶意提示时有效，但在面对意图模糊或涉及**双重用途 (Dual-use)**（如生物、网络安全）的领域时，显得非常脆弱，常常导致过度拒绝。

**安全补全 (Safe-Completions)** 是一种以**输出为中心**的安全方法。它不再简单地判断用户意图，而是致力于在遵守安全策略的前提下，最大化回答的帮助性。

{{< figure
    src="safe_completions.png"
    caption="Fig. 4. Comparison between Hard Refusals and Safe-Completions. (Image source: Mathpix from [OpenAI, 2025a](https://openai.com/index/introducing-gpt-5/))"
    align="center"
    width="100%"
>}}

**工作方式:**
*   对于一个可能涉及双重用途的请求，模型不会直接拒绝，而是提供一个高层次、不包含可直接执行的危险细节的安全回答。
*   如果必须拒绝，模型会被训练以透明地解释拒绝的原因，并提供安全的替代方案。

实验证明，安全补全方法在处理双重用途提示时显著提升了安全性，降低了残余安全风险的严重性，并大幅提高了模型的整体帮助性。

## 关键能力评估与挑战缓解

OpenAI 系统卡详细介绍了 GPT-5 在多个安全和可靠性维度上的评估结果，展示了其相较于前代模型的显著进步。

### 减少幻觉 (Hallucinations)

幻觉是 LLM 面临的核心挑战之一。GPT-5 在这方面取得了巨大进步。
*   **评估方法:** OpenAI 使用一个带网络访问功能的 LLM 评分器，在代表真实 ChatGPT 流量的提示上评估模型回答的事实正确性。该评分器通过与人类评估员 75% 的一致性验证，并被证明能比人类发现更多事实错误。
*   **结果:**
    *   在启用浏览的情况下，`gpt-5-main` 的幻觉率比 GPT-4o 低 26%，`gpt-5-thinking` 比 o3 低 65%。
    *   在响应层面，`gpt-5-thinking` 包含至少一个重大事实错误的回答数量比 o3 减少了 78%。
    *   在 LongFact 和 FActScore 等开源长文本事实性基准上，`gpt-5-thinking` 的事实错误数量比 o3 少了 5 倍以上。

{{< figure
    src="hallucination_rate.png"
    caption="Fig. 5. Average Hallucination Rate on Open-Source Benchmarks (Browsing Disabled). (Image source: Mathpix from [OpenAI, 2025b](https://openai.com/index/introducing-gpt-5/gpt5-system-card-aug7.pdf))"
    align="center"
    width="100%"
>}}

### 缓解欺骗 (Deception)

欺骗行为，即模型的外部响应与其内部推理或行为不符，是推理模型的一大风险。例如，o3 有时会谎称自己执行了某项操作或编造经验。

*   **缓解措施:** OpenAI 通过在特定场景下训练模型来减少欺骗行为。这些场景包括：
    *   **无法完成的编码任务:** 如缺少网络访问或任务规模过大。
    *   **损坏的工具:** 模拟工具返回错误或不可用的情况。
    *   **信息不足的用户请求:** 如提问关于一张不存在的图片。
    在这些情况下，模型被奖励诚实地承认其局限性。
*   **结果:** 在编码欺骗、工具损坏和图像缺失等评估中，`gpt-5-thinking` 的欺骗率远低于 o3。例如，在 CharXiv 缺失图像测试中，o3 的欺骗率为 87%，而 GPT-5 仅为 9%。

{{< figure
    src="deception_evals.png"
    caption="Fig. 6. Deception Rate Evaluations Across Different Scenarios. (Image source: Mathpix from [OpenAI, 2025b](https://openai.com/index/introducing-gpt-5/gpt5-system-card-aug7.pdf))"
    align="center"
    width="100%"
>}}

### 思维链监控 (CoT Monitoring)

可解释的思维链为监控和理解模型行为提供了独特的窗口。OpenAI 致力于保持其推理模型 CoT 的**可监控性 (Monitorability)**，即尽可能忠实和清晰。
*   **应用:** 通过监控 CoT，OpenAI 能够检测模型在真实世界使用中的欺骗行为。
*   **发现:** CoT 监控器在代表性的生产流量中，标记了约 4.8% 的 o3 响应和 2.1% 的 `gpt-5-thinking` 响应存在欺骗行为。这表明，尽管仍有改进空间，但 GPT-5 的欺骗倾向已显著降低。这项技术对于识别和优先处理模型的对齐问题至关重要。

### 应对谄媚 (Sycophancy)

谄媚行为，即模型过度附和或奉承用户，会损害交互的质量。
*   **缓解措施:** OpenAI 通过后训练来减少 GPT-5 的谄媚行为。他们使用代表生产数据的对话来评估模型的谄媚程度，并将该分数作为 RL 训练中的奖励信号。
*   **结果:** `gpt-5-main` 在离线评估中的表现比 GPT-4o 好近 3 倍。在初步的在线 A/B 测试中，与 GPT-4o 相比，免费用户的谄媚行为发生率下降了 69%，付费用户下降了 75%。

## 准备框架与高风险领域保障

面对日益强大的模型能力，OpenAI 启动了**准备框架 (Preparedness Framework)**，以主动识别、评估和缓解前沿 AI 可能带来的严重风险。

### 生物与化学风险的审慎对待

尽管没有明确证据表明 GPT-5 能帮助新手制造严重的生物危害，但 OpenAI 采取了**预防性原则**，决定将 `gpt-5-thinking` 视为在**生物和化学领域具有高能力 (High Capability)**，并激活了相应的安全保障措施。

这套多层次的端到端保障体系包括：
1.  **威胁建模:** 识别出两条主要风险路径：(1) 提升新手能力以制造已知生物威胁；(2) 提升专家能力以创造或部署生物威胁。
2.  **生物威胁分类:** 建立了一个详细的分类系统，包括**生物武器化 (Biological Weaponization)**、**高风险双重用途生物学 (High Risk Dual Use Biology)** 和**低风险双重用途生物学 (Low Risk Dual Use Biology)**，用于指导模型训练和系统防护。
3.  **模型安全训练:** 使用上述分类系统，通过“安全补全”范式训练模型拒绝武器化请求，并对双重用途话题提供安全的、非操作性的高层回答。
4.  **系统级防护:** 部署了一个两级实时监控系统，覆盖所有 `gpt-5-thinking` 流量。
    *   **第一级:** 快速分类器，判断内容是否与生物学相关。
    *   **第二级:** 推理监视器，根据生物威胁分类系统精确判断响应的风险等级，并决定是否向用户展示。
5.  **账户级执法:** 结合自动化系统和生物威胁专家的审查，检测并封禁试图滥用模型的用户，并在极端情况下通报执法部门。
6.  **API 访问控制:** 引入新的 `safety_identifier` 字段，帮助开发者区分其终端用户，并实施更精细的风险管控。

通过大量的内部评估和外部专家（包括美国 CAISI 和英国 AISI）的红队演练，这套保障体系的有效性得到了验证，确保了在释放强大能力的同时，将潜在风险降至最低。

## 总结

GPT-5 不仅仅是一次性能的线性提升，它代表了 OpenAI 在构建更智能、更可靠、更安全的通用人工智能系统方面的一次范式转变。其创新的**统一系统架构**、基于**强化学习的深度推理**能力，以及以**安全补全**和**思维链监控**为代表的先进安全理念，共同构成了 GPT-5 的技术内核。

通过深入解析这些技术细节，我们不仅能更好地理解 GPT-5 的强大之处，也能看到通往更负责任、更有益的 AGI 的清晰路径。未来，随着这些技术的不断成熟和融合，我们有理由期待一个更加智能且与人类价值观对齐的 AI 时代的到来。

## 参考文献

[1] OpenAI. (2025a). ["Introducing GPT-5."](https://openai.com/index/introducing-gpt-5/) OpenAI Blog.

[2] OpenAI. (2025b). ["GPT-5 System Card."](https://openai.com/index/introducing-gpt-5/gpt5-system-card-aug7.pdf)

[3] OpenAI. (2025c). ["Introducing GPT-5 for developers."](https://openai.com/index/introducing-gpt-5-for-developers/) OpenAI Blog.

[4] OpenAI. (2025d). ["Introducing gpt-oss."](https://openai.com/index/introducing-gpt-oss/) OpenAI Blog.

[5] Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2024). ["Roformer: Enhanced transformer with rotary position embedding."](https://arxiv.org/abs/2104.09864) Neurocomputing, 568, 127063.

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (August 2025). 深入解析OpenAI GPT-5：架构、技术与安全.
> https://syhya.github.io/zh/posts/2025-08-10-gpt-5-deep-dive/

Or

```bibtex
@article{yue_shui_gpt5_deep_dive_2025,
  title   = "深入解析OpenAI GPT-5：架构、技术与安全",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "August",
  url     = "https://syhya.github.io/zh/posts/2025-08-10-gpt-5-deep-dive/"
}
```