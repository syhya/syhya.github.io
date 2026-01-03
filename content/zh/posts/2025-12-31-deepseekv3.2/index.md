---
title: "DeepSeek-V3.2 系列"
date: 2025-12-31T12:00:00+08:00
lastmod: 2025-12-31T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "LLM", "DeepSeek", "Sparse Attention", "Reinforcement Learning", "Reasoning", "Agent", "Theorem Proving"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

通过引入 DeepSeek Sparse Attention (DSA) 高效注意力机制、可扩展的强化学习框架以及大规模智能体任务合成管道，**DeepSeek-V3.2**([DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))在推理能力和智能体性能上实现了与 GPT-5 相当的水平。

{{< figure
    src="deepseek_eval_res.png"
    caption="Fig. 1. Benchmark of DeepSeek-V3.2 and its counterparts. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

## DeepSeek Sparse Attention

{{< figure
    src="deepseekv3.2_arch.png"
    caption="Fig. 2. Attention architecture of DeepSeek-V3.2, where DSA is instantiated under MLA. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

传统注意力机制假设每个 token 都需要关注全部历史 token，但从信息论角度看，文本中的有效信息分布高度不均匀，真正与当前 token 相关的历史 token 只占少数。 [滑动窗口注意力](https://syhya.github.io/zh/posts/2025-08-24-gpt5/#efficient-attention-mechanisms)虽然意识到这一点，却仅通过限制关注最近窗口来简化计算，容易丢失关键的长程依赖。DeepSeek 的核心思想是**让模型自主学习并动态选择真正重要的 token**，从而在效率与信息保留之间取得更优平衡。

### 闪电索引器

DSA 的核心是使用一个轻量级的**闪电索引器（Lightning Indexer）** 快速筛选相关 token。对于每个查询 token $\mathbf{h}_t$，索引器会计算它与所有前序 token $\mathbf{h}_s$ 之间的相关性分数：

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)
$$

这里有几个设计值得注意：
1. **多头索引**：使用 $H^I$ 个索引器头，每个头学习不同的相关性模式
2. **ReLU 激活**：将负相关性置零，提供一定的稀疏性
3. **可学习权重**：$w_{t,j}^I$ 决定每个头的贡献，允许模型动态调整

索引器的计算量远小于主注意力。论文提到它可以用 FP8 实现，这意味着在保持精度的同时大幅降低计算开销。

### 细粒度 token 选择

有了相关性分数后，模型只需要选择 top-k 个最相关的 token：

$$
\mathbf{u}_t = \text{Attn}\left(\mathbf{h}_t, \left\{\mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\right\}\right)
$$


其中，$k$ 设置为 2048，以在效率和效果之间取得平衡。该设置将注意力计算复杂度从 $O(L^2)$ 降低到 $O(Lk)$，显著减少了计算开销，同时仍能覆盖大部分关键依赖关系。


### 继续预训练

从 [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) 进行继续预训练，包含两个训练阶段：

**稠密预热阶段 (Dense Warm-up Stage)**

保持原有的稠密注意力，只训练索引器。目标是让索引器的输出分布与真实的注意力分布对齐：

$$
\mathcal{L}^I = \sum_t D_{\text{KL}}\left(p_{t,:} \| \text{Softmax}(I_{t,:})\right)
$$

这里 $p_{t,:}$ 表示由主注意力分数聚合得到的真实注意力分布，该阶段通过与索引器输出分布的对齐，确保索引器能够学习并识别在当前时间步下哪些历史 token 对建模最为重要。

- 学习率：$10^{-3}$
- 训练步数：1000 步
- 每步：16 个序列 × 128K tokens
- 总 tokens：2.1B

**稀疏训练阶段（Sparse Training Stage）**

在索引器完成预热后，引入细粒度的 token 选择机制，并对模型的全部参数进行联合优化，以适应 DSA 的稀疏注意力计算模式。该阶段中，索引器仅在被选中的关键 token 子集上与主注意力分布进行对齐，其损失函数定义为：

$$
\mathcal{L}^I = \sum_t \mathbb{D}_{\mathrm{KL}}\left(p_{t,\mathcal{S}_t} \| \operatorname{Softmax}\left(I_{t,\mathcal{S}_t}\right)\right)
$$

$$
\mathcal{S}_t=\left\{s \mid I_{t, s} \in \operatorname{Top}-\mathrm{k}\left(I_{t,:}\right)\right\}
$$

其中 $\mathcal{S}_t$ 表示在时间步 $t$ 上由索引器预测为最重要的 top-k 个键值 token 集合。

* 学习率：$7.3 \times 10^{-6}$
* 每个查询 token 选择 2048 个键值 token
* 训练步数：15000 步
* 每步：480 个序列 × 128K tokens
* 总 tokens：943.7B

### 推理成本

DeepSeek-V3.2 与 DeepSeek-V3.1-Terminus 中的 MLA 相比，其所需计算量更小，上下文越长，收益越明显。

在 H800 GPU 集群上的实际成本对比（以 2 美元/GPU 小时计算）：

{{< figure
    src="inference_cost_compare.png"
    caption="Fig. 3. Inference costs of DeepSeek-V3.1-Terminus and DeepSeek-V3.2 on H800 clusters. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

## Scaling GRPO

DeepSeek-V3.2 将推理、智能体和人类对齐训练合并到一个 RL 阶段。这种方法有效地平衡了跨多样化领域的性能，同时避免了与多阶段训练范式常见的灾难性遗忘问题。

奖励设计：
- **推理和智能体任务**：基于规则的结果奖励 + 长度惩罚 + 语言一致性奖励。
- **通用任务**：生成式奖励模型，每个提示有自己的评分标准。

### GRPO

[GRPO](https://syhya.github.io/zh/posts/2025-01-27-deepseek-r1/#grpo) 是 DeepSeek 提出的一种高效 RL 算法，它通过组内相对优势估计来替代传统 PPO 中的价值模型。GRPO 通过最大化以下目标来优化策略模型 $\pi_\theta$：

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)= & \mathbb{E}_{q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\mathrm{old}}(\cdot \mid q)}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\right. \\
& \left.\min \left(r_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right)-\beta \mathbb{D}_{\mathrm{KL}}\left(\pi_\theta\left(o_{i, t}\right) \| \pi_{\mathrm{ref}}\left(o_{i, t}\right)\right)\right],
\end{aligned}
$$

其中重要性采样比率：
$$
r_{i, t}(\theta)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_{\mathrm{old}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}
$$

但在大规模训练中，原始 GRPO 会遇到稳定性问题。DeepSeek-V3.2 中提出了四项关键改进。

### 稳定 RL 扩展的关键策略

1. **无偏 KL 估计 (Unbiased KL Estimate)**

原始 GRPO 使用的 K3 估计器在某些情况下会产生系统性偏差。当采样的 token 在当前策略下的概率远低于参考策略时（$\pi_\theta\left(o_t \mid q, o_{&lt;t  }\right) \ll \pi_{\mathrm{ref}}\left(o_t \mid q, o_{&lt;t  }\right)$），梯度会变得异常大，导致训练不稳定。

$$
\mathbb{D}_{\mathrm{KL}}\left(\pi_\theta\left(o_{i, t}\right) \| \pi_{\mathrm{ref}}\left(o_{i, t}\right)\right)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_{\mathrm{old}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}\left(\frac{\pi_{\mathrm{ref}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}-\log \frac{\pi_{\mathrm{ref}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}-1\right)
$$

通过在 K3 估计器中加入**当前策略 $\pi_\theta$ 与旧策略 $\pi_{\text{old}}$ 的 importance-sampling 比值**进行校正，使 KL（及其梯度）估计**无偏、收敛更稳定**，并可按领域需求**调节 KL 惩罚强度**（必要时弱化甚至省略）。

2. **离策略序列掩码 (Off-Policy Sequence Masking)**

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)= & \mathbb{E}_{q \sim P(Q),\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\mathrm{old}}(\cdot \mid q)}\left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|}\right. \\
& \left.\min \left(r_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right) M_{i, t}-\beta \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}\left(o_{i, t}\right) \| \pi_{\mathrm{ref}}\left(o_{i, t}\right)\right)\right],
\end{aligned}
$$

$$
M_{i,t} = \begin{cases} 0 & \text{if } \hat{A}_{i,t} < 0 \text{ and } \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log\frac{\pi_{\text{old}}(o_{i,t}|q, o_{i,&lt;t  })}{\pi_\theta(o_{i,t}|q, o_{i,&lt;t  })} > \delta \\ 1 & \text{otherwise} \end{cases}
$$


为缓解多步更新与训练–推理不一致引入的离策略问题，在 GRPO 中引入[离策略序列掩码 (Off-Policy Sequence Masking)](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)，仅对**负优势且策略分歧（KL）超过阈值**的序列进行屏蔽，从而抑制高度离策略负样本对优化的干扰。该机制在保留有效学习信号的同时显著提升了训练稳定性。


3. **Keep Routing**

对于 MoE 模型，一个微妙的问题是：推理时激活的专家组合可能与训练时不一致。即使输入相同，框架差异或策略更新也可能导致不同的路由结果。

DeepSeek 的做法是记录采样时的路由路径，并在训练时强制使用相同的路径。这确保了梯度更新的是真正产生了采样输出的那些参数。目前 verl 框架中已经集成了 [Router Replay](https://github.com/volcengine/verl/tree/main/examples/router_replay) 功能，可以直接使用。

4. **Keep Sampling Mask**

在 RL 训练的 rollout 阶段，常采用 Top-p / Top-k 采样以过滤低概率 token、提升生成质量；然而训练阶段通常基于完整词表进行优化，这会导致旧策略 $\pi_{\text{old}}$ 与新策略 $\pi_\theta$ 的动作空间不一致，破坏重要性采样假设并引发训练不稳定。为此，DeepSeek 在 rollout 时记录采样产生的截断掩码，并在训练阶段对 $\pi_\theta$ 施加相同掩码，强制新旧策略在一致的动作子空间内进行优化，从而在结合 Top-p 采样的同时保持 RL 训练过程中的生成一致性与稳定性。

## 智能体任务合成与训练

### 工具使用中的思考

**思考上下文管理**

将推理能力融入工具使用场景是一个有趣的挑战。DeepSeek R1 的做法是在每轮新消息时丢弃之前的推理内容，但这在工具调用场景下会导致严重的 token 浪费——模型每次调用工具后都要重新推理整个问题。**Claude Opus 4.5** ([Anthropic, 2025](https://www.anthropic.com/news/claude-opus-4-5)) 通过 [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) 与 [new context tool](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf) 实现跨上下文的信息持久化与上下文重置，在保持单次 200k token 上下文规模的同时，支持跨多轮重置累计最高 1M token 的有效使用。

{{< figure
    src="claude_context_window.png"
    caption="Fig. 4. The context window token management when combining extended thinking with tool use in Claude. (Image source: [Claude Docs](https://platform.claude.com/docs/en/build-with-claude/context-windows#the-context-window-with-extended-thinking-and-tool-use))"
    align="center"
    width="100%"
>}}

DeepSeek-V3.2 也参照实现精细的上下文管理：

{{< figure
    src="deepseek_thinking_tool_use.png"
    caption="Fig. 5. Thinking retention mechanism in tool-calling scenarios. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

- 只有在收到**新的用户消息**时才丢弃推理内容
- 工具输出等消息不会触发推理内容的删除
- 工具调用历史始终保留在上下文中

这种设计在保持推理连贯性的同时，避免了冗余计算。

### 大规模智能体任务

多样化的 RL 任务对于增强模型鲁棒性至关重要。DeepSeek-V3.2 使用的智能体任务包括：

| Agent Type       | Number of Tasks | Environment | Prompt Type |
| ---------------- | --------------- | ----------- | ----------- |
| Code Agent       | 24,667          | Real        | Extracted   |
| Search Agent     | 50,275          | Real        | Synthesized |
| General Agent    | 4,417           | Synthesized | Synthesized |
| Code Interpreter | 5,908           | Real        | Extracted   |


{{< figure
    src="deepseekv3.2_benchmark.png"
    caption="Fig. 6. Comparison between DeepSeek-V3.2 and closed/open models. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}


评估结果显示了 DeepSeek-V3.2 在推理任务上达到了与 GPT-5 High 相当的水平，并在代码智能体（Code Agent）任务中显著优于开源模型，同时在工具使用（Tool-Use）场景下有效缩小了与顶尖闭源模型的差距，展现了强大的泛化能力。

### 搜索智能体的上下文管理

尽管配备了如 128K 的长上下文窗口，智能体工作流（尤其是基于搜索的场景）仍频繁遭遇最大长度限制，导致推理过程被迫提前中断。这一瓶颈严重阻碍了测试时计算（Test-time Compute）潜力的充分释放。为此，当 Token 使用量超过上下文窗口的 **80%** 时，引入了以下上下文管理策略，以在测试时动态扩展 Token 预算：

1. **Summary（摘要重置）**：对溢出的轨迹内容进行摘要总结，并基于此重新启动后续的推理过程。
2. **Discard-75%（部分丢弃）**：丢弃轨迹中前 75% 的工具调用历史记录，以释放存储空间。
3. **Discard-all（完全重置）**：丢弃所有先前的工具调用历史以彻底重置上下文（机制类似于 Anthropic 的 new context tool）。

{{< figure
    src="new_context_tool.png"
    caption="Fig. 7. New context tool feature in Claude. (Image source: [Anthropic, 2025](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf))"
    align="center"
    width="90%"
>}}

在 **BrowseComp** 基准测试中对上述策略进行了评估。结果表明，上下文管理通过允许模型进行更多的执行步骤，显著提升了测试时计算的规模和模型性能。具体表现如下：

{{< figure
    src="search_agent_browsecomp.png"
    caption="Fig. 8. Accuracy of BrowseComp with different test-time compute expansion strategies. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

虽然 Summary 策略通过将平均步数从 140 延长至 364，成功将得分从 53.4 提升至 60.2，但其整体计算效率相对较低。 Discard-all 策略则在效率和可扩展性方面表现最佳，达到了 **67.6** 的高分。值得注意的是，它在性能上与并行扩展（Parallel scaling）基线相当，但所需的执行步数显著更少。

## DeepSeekMath-V2

基于[DeepSeek-V3.2-Exp-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp-Base) 开发的 **DeepSeekMath-V2**([Shao et al., 2025](https://arxiv.org/abs/2511.22570))专注于形式化数学推理与定理证明。与传统基于最终答案的强化学习不同，DeepSeekMath-V2 引入了一套**自验证（Self-Verification）** 机制，通过独立的证明验证器和元验证器的协同训练，使模型能够在生成证明的同时对推理过程进行严格的自我检查。这种方法在 [IMO-Proof Bench](https://imobench.github.io/) 和 [Putnam](https://maa.org/putnam/) 等高难度数学竞赛基准上取得了金牌级别的表现。

### 过程奖励模型

传统的数学推理强化学习（RL）通常采用**基于结果的奖励（Outcome-based Reward Models, ORM）**，即仅根据最终答案是否正确给予奖励。这种方法在 AIME、HMMT 等以数值答案为主的竞赛中尚能奏效，但在更复杂的推理任务中暴露出根本性局限：

1. **代理指标不可靠**：最终答案正确并不意味着推理过程正确。模型可能通过错误推理、捷径或偶然性得到正确结果（False Positive）。
2. **不适用于定理证明**：定理证明等高阶数学任务强调严格的逐步逻辑推导，单一的结果奖励无法提供有效训练信号。

针对上述问题，DeepMind 在 2022 年已率先探索了**过程奖励模型（Process Reward Models, PRM）** 来解决数学问题，提出对中间推理步骤进行显式评估，以缓解结果奖励的监督不足问题（[Uesato et al., 2022](https://arxiv.org/abs/2211.14275)）。这一工作系统性地揭示了：相比仅监督最终答案，对推理过程本身进行建模更有利于复杂推理能力的学习。

{{< figure
    src="orm_prm_compare.png"
    caption="Fig. 9. Comparison of policy improvement mechanisms for Final-Answer RL, ORM-RL, and PRM-RL. (Image source: [Uesato et al., 2022](https://arxiv.org/abs/2211.14275))"
    align="center"
    width="100%"
>}}

DeepSeekMath-V2 模型也以此模式进行训练，使得模型生成推理过程中主动识别潜在逻辑漏洞并进行自我修正，模拟人类在阅读与审查证明时的思维模式。

### 整体架构

{{< figure
    src="deepseekv3.2_math_arch.png"
    caption="Fig. 10. Self-verification architecture with proof generation, verifier-based evaluation, and meta-verification."
    align="center"
    width="100%"
>}}

DeepSeekMath-V2 构建了一个三层验证架构，通过模型间的相互监督实现持续改进：

- **证明生成器（$\pi_\theta$）**：负责根据问题 $X$ 生成数学证明。
- **证明验证器（$\pi_\varphi$）**：作为 LLM-as-a-judge 评估证明质量。
- **元验证器（$\pi_\psi$）**：监督验证器的评估过程，确保验证质量。

#### 数据构建

团队通过以下流程构建了初始训练数据：

1. **问题收集**：从 [Art of Problem Solving (AoPS)](https://artofproblemsolving.com/?srsltid=AfmBOoqcstCRpzZaf7rDkaLdkuHkR_SUAaTVBUHDrPo-nctXiCEuobst) 中爬取问题，优先选择数学奥林匹克、国家队选拔测试以及 2010 年后明确要求证明的问题，总计 **17,503 个问题**，记为 $\mathcal{D}_p$

2. **候选证明生成**：使用 DeepSeek-V3.2-Exp-Thinking 的变体生成候选证明。由于该模型未针对定理证明优化，倾向于产生简洁但容易出错的输出，因此提示它在多轮中迭代精炼证明以提高全面性和严谨性

3. **专家标注**：随机抽样不同问题类型（如代数、数论）的证明，由数学专家根据评估标准对每个证明打分

这个过程产生了初始 RL 数据集 $\mathcal{D}_v = \{(X_i, Y_i, s_i)\}$，其中每项包含问题 $X_i$、证明 $Y_i$ 和整体证明分数 $s_i \in \{0, 0.5, 1\}$。

#### 验证器训练

验证器从 DeepSeek-V3.2-Exp-SFT 初始化（该模型基于 DeepSeek-V3.2-Exp 在数学和代码相关的推理数据上监督微调得到），采用三级评分标准：

- **1.0 分**：完整且严格的证明，所有逻辑步骤都有清晰论证。
- **0.5 分**：整体逻辑正确但存在细微错误或遗漏细节。
- **0.0 分**：包含致命逻辑错误或关键推理缺口的根本有缺陷的证明。

给定问题 $X$ 和证明 $Y$，验证器 $\pi_\varphi(\cdot|X, Y, \mathcal{I}_v)$ 设计为首先总结识别的问题，然后根据评分标准分配分数。

验证器通过强化学习优化，结合两个奖励组件：

- **格式奖励 $R_{\text{format}}$**：通过校验响应中是否包含指定评估短语及其后 `\boxed{}` 中的分数，约束模型按固定模板输出问题总结与证明评分。

- **分数奖励 $R_{\text{score}}$**：基于预测分数 $s'_i$ 与标注分数 $s_i$ 之间的接近程度：
  $$R_{\text{score}}(s'_i, s_i) = 1 - |s'_i - s_i|$$

验证器的 RL 目标为：

$$
\max_{\pi_\varphi} \mathbb{E}_{(X_i, Y_i, s_i) \sim \mathcal{D}_v, (V'_i, s'_i) \sim \pi_\varphi(\cdot|X_i, Y_i)} \left[ R_{\text{format}}(V'_i) \cdot R_{\text{score}}(s'_i, s_i) \right]
$$

其中 $V'_i$ 表示验证器的最终响应，$s'_i$ 是从中提取的证明分数。

#### 元验证器

上述方法通过 RL 训练证明验证器使预测的证明分数与专家标注对齐，但**不直接监督识别的问题本身**。这造成了一个关键漏洞：在训练期间评估有缺陷的证明时（$s_i < 1$），验证器可以通过预测正确分数同时幻觉不存在的问题来获得完全奖励，从而破坏其可信度。

为解决这个问题，引入了**元验证（Meta-Verifier）**：一个二级评估过程，评估验证器识别的问题是否确实存在，以及这些问题是否根据评估标准 $\mathcal{I}_v$ 逻辑上证明预测的证明分数。

元验证器同样通过强化学习训练，采用与验证器类似的目标函数。使用训练好的元验证器 $\pi_\psi$，通过将元验证反馈整合到奖励函数中来增强验证器训练：

$$
R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}
$$

其中 $R_{\text{meta}}$ 是来自元验证器的质量分数。

实验结果表明，引入元验证器在 $\mathcal{D}_v$ 的验证集上，验证器证明分析的平均质量分数从 0.85 提高到 0.96，同时保持相同的证明分数预测准确性。

这种设计类似于 **生成对抗网络（Generative adversarial network, GANs）**([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661))的思想：验证器推动生成器改进，而更强的生成器反过来为验证器提供更具挑战性的训练样本，形成良性循环。需要注意的是，元验证分数**仅在训练阶段使用**，在推理时不参与计算。

### 生成器训练

以验证器 $\pi_\varphi$ 作为生成式奖励模型，证明生成器 $\pi_\theta$ 的优化目标为：

$$
\max_{\pi_\theta} \mathbb{E}_{X_i \sim \mathcal{D}_p, Y_i \sim \pi_\theta(\cdot|X_i)} [R_Y]
$$

其中 $R_Y$ 是由 $\pi_\varphi(\cdot|X_i, Y_i, \mathcal{I}_v)$ 产生的证明分数。


在训练期间，提示生成器 $\pi_\theta$ 产生证明 $Y$ 后紧接着生成遵循与验证器相同格式和标准 $\mathcal{I}_v$ 的自我分析 $Z$。将自我分析中预测的证明分数记为 $s'$。奖励函数综合考虑这些评估：

$$
R = R_{\text{format}}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z)
$$

$$
R_Z = R_{\text{score}}(s', s) \cdot R_{\text{meta}}(Z)
$$

其中 $\alpha = 0.76$，$\beta = 0.24$。

这种奖励结构创造了以下激励：

- **诚实优于虚假**：如实承认错误相比于错误地声称正确性会获得更高的奖励。
- **自知之明**：最高奖励来自于给出正确的证明，并且准确识别其严谨性。
- **主动改进**：对于证明生成器而言，获得高奖励的有效策略 是在最终给出回答之前，尽可能识别并解决潜在的问题。

### 串行修正

{{< figure
    src="imo_2024_res.png"
    caption="Fig. 11. Proof Quality Improves with Increasing Sequential Self-Refinement Iterations (1–8). (Image source: [Shao et al., 2025](https://arxiv.org/abs/2511.22570))"
    align="center"
    width="80%"
>}}

[Test-time Scaling](https://syhya.github.io/zh/posts/2025-11-19-scaling-law/#test-time-scaling) 通过在推理阶段增加计算量来提升模型性能，尤其适用于 IMO、Putnam 等高难度数学证明任务。上图展示了 **DeepSeekMath-V2** 在最多 **8 次串行迭代** 下的性能变化：随着最大顺序迭代次数增加，**Pass@1** 从约 **0.15** 稳步提升至 **0.27**，**Best@32** 从约 **0.26** 提升至 **0.42**。其中，串行修正（Sequential Refinement）结合自验证机制，通过多轮生成与纠错，在计算成本可控的前提下显著提高证明成功率，并呈现出随迭代次数增加而稳定上升的性能增益趋势。


## 参考文献

[1] Liu, Aixin, et al. ["DeepSeek-V3.2: Pushing the frontier of open large language models."](https://arxiv.org/abs/2512.02556) arXiv preprint arXiv:2512.02556 (2025).

[2] Shao, Zhihong, et al. ["DeepSeekMath-V2: Towards self-verifiable mathematical reasoning."](https://arxiv.org/abs/2511.22570) arXiv preprint arXiv:2511.22570 (2025).

[3] Uesato, Jonathan, et al. ["Solving math word problems with process-and outcome-based feedback."](https://arxiv.org/abs/2211.14275) arXiv preprint arXiv:2211.14275 (2022).

[4] Goodfellow, Ian J., et al. ["Generative adversarial nets."](https://arxiv.org/abs/1406.2661) Advances in neural information processing systems 27 (2014).

[5] Anthropic. [“Claude Opus 4.5 — System Card.”](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf) 2025. (PDF)

[6] Luong, Minh-Thang, et al. ["Towards robust mathematical reasoning."](https://arxiv.org/abs/2511.01846) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025.

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Dec 2025). DeepSeek-V3.2 系列.
https://syhya.github.io/zh/posts/2025-12-31-deepseekv3.2

Or

```bibtex
@article{syhya2025deepseekv32,
  title   = "DeepSeek-V3.2 系列",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Dec",
  url     = "https://syhya.github.io/zh/posts/2025-12-31-deepseekv3.2"
}
```
