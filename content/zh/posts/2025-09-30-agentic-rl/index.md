---
title: "Agentic RL"
date: 2025-09-30T12:00:00+08:00
author: "Yue Shui"
tags: ["Agentic RL", "Reinforcement Learning", "LLM", "Agent", "SWE-bench", "verl", "ReTool"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

大语言模型（LLMs）目前应用场景不断扩展，但也暴露出知识截止、幻觉以及复杂计算与逻辑推理不足等局限。为应对这些挑战，将智能体（Agent）与强化学习（Reinforcement Learning, RL）相结合的 **Agentic RL** 正逐渐成为关键研究方向。

Agentic RL 通过让模型与外部世界（如搜索引擎、代码解释器、数据库、浏览器等）形成闭环交互，并借助奖励信号持续优化，使 LLM 拥有自主规划、决策制定、工具使用与环境交互等能力。在实际业务中，它不仅能理解需求并自主规划，还能在执行与反馈循环中不断修正与优化。

其核心价值主要体现在两方面：

- **减少提示依赖**: 让模型摆脱对 prompt 的过度依赖，具备自适应问题求解能力；
- **强化自主探索**: 借助多轮强化学习，提升探索与推理能力，从而弥补静态数据分布稀疏或重复带来的不足。

## Agentic RL 与 LLM-RL 区别

{{< figure
    src="agentic_rl_survey.png"
    caption="Fig. 1. Paradigm shift from LLM-RL to Agentic RL. (Image source: [Zhang et al., 2025](https://arxiv.org/abs/2509.02547))"
    align="center"
    width="100%"
>}}
 
以 RLHF 为代表的对齐式 **LLM-RL** 在实践中常被近似为单步（序列级）决策的[马尔可夫决策过程（Markov Decision Process, MDP）](https://en.wikipedia.org/wiki/Markov_decision_process)；而 **Agentic RL** 则在部分可观测环境中展开，涉及多步、长时程的序列决策，更适合用[部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）](https://en.wikipedia.org/wiki/Partially_observable_markov_decision_process)进行刻画。下面表格也总结了两者之间的差异。

| 特性 | 传统 LLM-RL (如 RLHF) | Agentic RL |
| :--- | :--- | :--- |
| **决策过程** | **单步退化 MDP**：输入 prompt → 输出完整 response → 一次性奖励，类似“单轮映射”。 | **多步长时程 POMDP**：在部分可观测环境中持续交互，每一步都会更新状态并获得反馈。 |
| **状态空间** \(\mathcal{S}\) | 静态，仅由输入 prompt 决定，不随过程演化。 | 动态，包含历史交互、工具返回结果、外部环境状态等，随交互不断更新。 |
| **动作空间** \(\mathcal{A}\) | 单一动作：生成文本序列（response）。 | 复合动作：生成思考链（Thought）、调用工具（Tool Call）、更新状态、生成最终答案。 |
| **奖励** \(\mathcal{R}\) | **稀疏 Outcome Reward**：多在生成完成后，由人工偏好或模型裁判给出。 | **混合奖励机制**：既包括稀疏 Outcome Reward，也结合稠密 Process Reward（如工具调用成败、子任务完成度）。 |
| **核心挑战** | 对齐人类偏好，保证安全性与有用性；提升整体生成质量。 | 长时程信用分配、复杂任务规划、探索效率、工具的稳健使用，以及探索—利用的平衡。 |

## 评估

科学、全面且贴近真实的评估基准是衡量和提升 LLM Agent 能力的关键。**"Successful language model evals"** ([Wei, 2024](https://www.jasonwei.net/blog/evals))总结了成功评估基准的几大关键特质，这些特质共同决定了一个评测集能否被社区广泛接受并经受住时间的考验：

1.  **足够的样本量**：评测集需要包含足够多的样本（通常建议至少 1000 个），以减少评估结果的随机波动。样本量过少会导致评测分数在不同模型检查点之间剧烈震荡，给研究者带来困扰，使其难以判断模型性能的真实变化。
2.  **高质量的数据**：评测集中的数据（问题、答案、测试用例等）必须准确无误。如果评测集自身存在大量错误，当一个强大的模型（如 GPT-4）给出与标准答案不符的结果时，研究者会更倾向于质疑评测集的正确性，从而丧失对该评测集的信任。
3.  **简洁的单一度量指标**：一个成功的评测集必须有一个核心的、易于理解的单一度量指标（如准确率）。过于复杂的评估体系，如 **HELM** ([Liang et al., 2022](https://arxiv.org/abs/2211.09110)) 的早期版本，虽然全面但因指标过多，反而让研究者难以聚焦，不利于快速比较和传播。
4.  **易于运行和复现**：评估流程应尽可能简单、高效。如果运行一次评测需要复杂的环境配置和漫长的等待时间如 **BIG-Bench** ([Srivastava et al., 2022](https://arxiv.org/abs/2206.04615)) 的部分子集，会极大地阻碍其推广和应用。
5.  **任务富有意义**：评测的任务应该与智能的核心能力（如语言理解、数学推理、代码生成）紧密相关。一些虽然有挑战性但意义不大的任务（如正确闭合括号），即使模型表现优异，也难以得出关于其“智能水平”的实质性结论。
6.  **评分机制准确可靠**：自动化评分脚本必须极其健壮和准确。如果研究者在调试时发现模型的正确输出被评分脚本误判，会迅速削弱他们对整个评测集的信心。
7.  **避免在评测集上过拟合**：评测集的难度应具有前瞻性，确保在未来一段时间内，模型性能仍有足够的提升空间。像 **GLUE** ([Wang et al., 2018](https://arxiv.org/abs/1804.07461)) 和 **SuperGLUE** ([Wang et al., 2019](https://arxiv.org/abs/1905.00537))这样很快被模型性能刷满的评测集，会迅速失去作为衡量技术进步标尺的作用。

**验证的不对称性**([Wei, 2024](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law))这个概念指出，许多任务**验证一个解的正确性远比从零开始找到这个解要容易得多**。

{{< figure
    src="asymmetry_verification.png"
    caption="Fig. 2. Improving Verification with Privileged Information. (Image source: [Wei, 2024](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law))"
    align="center"
    width="100%"
>}}

例如，解决一个复杂的数独谜题可能需要数小时，但验证一个填好的数独是否正确只需几分钟；编写一个大型网站的后端代码需要团队数年的努力，但任何用户都能快速判断网站是否正常工作。类似地，很多信息检索或开放型任务，生成答案可能要经过大量尝试，但一旦有候选结果，验证它是否符合约束往往只需很少的时间。

如果我们能为任务提前准备先验信息或验证机制，就能显著降低验证成本。比如：  
- **SWE-bench**：验证代码正确性本来需要人工逐行阅读，但如果提前准备好覆盖率足够的测试用例，那么只需运行测试，就能在几秒钟内判断模型生成的补丁是否有效。  
- **AIME 数学竞赛**：数学题目的推导过程往往复杂且耗时，但一旦公布标准答案，任何解答都可以通过对照答案在几秒钟内完成验证。  

这种不对称性对于 AI 训练至关重要，因为它直接关系到构建 RL 环境或奖励模型的可行性。由此，Jason Wei 提出了验证者法则：

> *训练 AI 解决一个任务的难易程度，与该任务的可验证性成正比。所有可解且易于验证的任务，终将被 AI 解决。*

一个任务是否易于验证，通常取决于它是否满足以下五个关键属性：  

1. **客观真理**：对于什么是好的解答存在广泛共识（如数学题的唯一正确答案）。  
2. **快速验证**：单个解答能在秒级完成核验（如运行一组测试用例）。  
3. **可扩展验证**：可以并行验证大量候选解答（例如批量运行代码测试）。  
4. **低噪声**：验证信号与解答质量高度相关，误判率低。  
5. **连续奖励**：不仅能判定对错，还能对多个解答进行排序，从而形成更平滑的优化信号。  

这套法则解释了为何像 **SWE-bench 编程任务** 和 **AIME 数学解题**这样的场景，成为检验 AI 能力的理想试金石。它们天然符合上述大部分条件，使得我们能够高效构建自动化评测体系，并通过大规模“生成–验证”循环不断优化模型表现。  

### SWE-bench

{{< figure
    src="swe_bench.png"
    caption="Fig. 3. SWE-bench links real GitHub issues with their merged pull request fixes. Given an issue description and a codebase snapshot, models generate a patch that is tested against actual project test cases. (Image source: [Jimenez et al., 2024](https://arxiv.org/abs/2310.06770))"
    align="center"
    width="100%"
>}}

**SWE-bench** ([Jimenez et al., 2024](https://arxiv.org/abs/2310.06770)) 从 12 个流行的 Python 开源项目中收集了 2294 个真实的开发任务，这些任务直接来源于 GitHub 的 Issues 和 Pull Requests。  

为了保证实验的可重复性和环境独立性，SWE-Bench 为每个任务构建了**隔离的 Docker 环境**，避免因 Python 版本或依赖库不一致导致运行失败。同时，这种设计也迫使模型学会针对不同环境生成兼容的补丁代码。  

验证机制上，SWE-Bench 巧妙地利用了项目自带的单元测试来**自动化评估 LLM 的补丁是否正确**。它包含两类测试：  

- **Fail-to-Pass (F2P)**：原先失败的测试，合入正确的 PR 后应当通过，用于确认 LLM 是否修复了目标问题。  
- **Pass-to-Pass (P2P)**：原先能通过的测试，在合入 PR 后也必须继续通过，用于保证 LLM 没有破坏已有功能或引入新 bug。  

这种“真实任务 + 隔离环境 + 自动化测试”的组合，使 SWE-bench 成为一个高可信度、可扩展的基准，大幅降低了验证编程任务正确性的成本。但原始 SWE-bench 存在测试用例不公平、问题描述模糊和环境复杂等缺陷，导致模型能力被低估，因此 OpenAI 构建了经人工筛选的高质量子集 **SWE-bench Verified**([OpenAI, 2024](https://openai.com/index/introducing-swe-bench-verified/))，用以更准确评估模型水平。

### BrowseComp

与软件工程任务不同，网页浏览任务的目标是在浩瀚的互联网中找到特定信息。**BrowseComp** ([Wei et al., 2025](https://arxiv.org/abs/2504.12516)) 是一个专为此类任务设计的简单而具挑战性的基准。

*   **设计理念**：BrowseComp 遵循**难于解决，易于验证**的原则。问题被设计得需要持久、创造性地浏览大量网页才能找到答案，但答案本身通常是一个简短的、无可争议的字符串，可以轻松与参考答案进行比对。

*   **数据构建**：出题者采用**逆向提问**的方式。他们先找到一个冷门的事实（如一篇特定会议论文），然后围绕这个事实构建一个包含多个复杂约束条件的查询。例如：“请找出 EMNLP 2018-2023 年间发表的一篇论文，其第一作者本科毕业于达特茅斯学院，第四作者本科毕业于宾夕法尼亚大学。” 验证这个答案很简单，但要从数千篇论文中找到它则极为困难。

{{< figure
    src="BrowseComp_scale.png"
    caption="Fig. 4. BrowseComp performance of an early version of OpenAI Deep Research scales smoothly with test-time compute. (Image source: [Wei et al., 2025](https://arxiv.org/abs/2504.12516))"
    align="center"
    width="70%"
>}}

BrowseComp 衡量的是 Agent 的核心浏览能力：事实推理、持久导航和创造性搜索。如图所示，强大的浏览 Agent (如 OpenAI Deep Research) 在该基准上的性能会随着测试时计算量（即浏览努力程度）的增加而平滑提升，这表明该评测集能有效衡量 Agent 的深度搜索和信息整合能力。

## 数据

高质量的数据是训练强大智能体的基石。然而，人工标注 agent 在复杂任务中的完整决策轨迹成本极高且难以规模化。因此，**合成数据**成为了该领域的主流解决方案。各种创新的数据生成管线被提出，旨在构建一个能持续产生高质量训练数据的“数据飞轮”。

### AgentFounder

{{< figure
    src="tongyi_agentic_training_pipeline.png"
    caption="Fig. 5. The Agentic Training Pipeline proposed by AgentFounder, incorporating an Agentic CPT stage. (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

**AgentFounder** ([Su et al., 2025](https://arxiv.org/abs/2509.13310)) 在传统的预训练与后训练之间，提出了一个新的流程**智能体持续预训练 (Agentic Continual Pre-training, Agentic CPT)**，整个训练流程包含三个阶段：

1. **General Pre-training**：与标准流程一致，先训练一个具备通用知识的基础模型。
2. **Agentic CPT**：在通用基础模型之上，利用大规模、多样化的合成智能体行为数据，继续进行“下一个词预测”式训练。其目标并非解决具体任务，而是让模型内化通用的 Agent 行为模式，形成 agentic 能力。
3. **Post-training**：在已经具备基础智能体能力的模型上，再进行 SFT 或 RL，使其对齐到具体任务，从而避免后训练阶段“能力习得”和“任务对齐”同时进行所带来的优化冲突。

Agentic CPT 的关键是如何以**低成本合成大规模类智能体数据**。为此，AgentFounder 提出了两种无需外部工具调用的高效数据生成方法：**一阶动作合成 (First-order Action Synthesis, FAS)** 和 **高阶动作合成 (Higher-order Action Synthesis, HAS)**。

1. FAS 的核心思想是通过推演的方式，低成本地生成关于**如何思考**和**如何规划第一步**的数据。

{{< figure
    src="tongyi_fas.png"
    caption="Fig. 6. Illustration of First-order Action Synthesis (FAS). (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

- **规划动作**：给定一个问题，让 LLM 生成多种可能的分析与第一个工具调用计划，帮助模型学习任务分解与初步规划。
- **推理动作**：在提供问题和相关知识片段的情况下，让 LLM 生成完整的逻辑推理链，并得出最终答案，锻炼其逻辑演绎与信息综合能力。

FAS 仅生成思考过程与计划动作，不涉及真实工具调用或环境交互，因而生成成本极低，非常适合进行**大规模数据合成（可达亿级）**。

2. HAS 目标是将已有的（即使是次优的）agent 轨迹转化为高价值的**决策学习数据**。

{{< figure
    src="tongyi_has.png"
    caption="Fig. 7. Illustration of Higher-order Action Synthesis (HAS). (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

- **步骤级扩展**：针对轨迹中的任意一步，利用 LLM 生成多个替代动作，构建一个局部决策空间。
- **对比学习**：将原始选择与扩展出的候选动作，重构为一个带反馈的多选题，要求模型识别更优决策。
- **因果监督**：在轨迹末尾附上最终结果（成功或失败），帮助模型学习**决策与结果之间的因果联系**。

这种方法将传统的**轨迹模仿**，升级为**步骤级决策学习**。模型不仅会“走过一条成功路径”，更能理解**在每个关键节点如何选择**，提升了数据的信噪比与利用效率。

Agentic CPT 是否真的缓解了“优化冲突”？论文通过训练损失曲线与性能对比实验出了结论。

{{< figure
    src="agent_founder_loss.png"
    caption="Fig. 8. Training loss evolution showing superior convergence of AgentFounder models compared to baseline. (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

实验结果表明经过 Agentic CPT 的模型在 SFT 阶段收敛速度更快，训练损失显著低于未经过该阶段的基线模型。这说明模型在进入后训练阶段之前，已经形成了一定 agentic 能力，从而在后续学习特定任务时更加高效。

### WebShaper

传统的数据合成方法通常是信息驱动的：先从网络爬取信息，再根据信息生成问题。**WebShaper** ([Tao et al., 2025](https://arxiv.org/abs/2507.15061)) 指出，这种方式可能导致生成的问题中的推理结构与原始信息结构不一致，模型可能通过“逻辑捷径”而非真正的多步推理来找到答案。为此，它提出了一种**形式化驱动** 范式, 这里主要用了知识图谱和集合论来形式化问题。

{{< figure
    src="WebShaper.png"
    caption="Fig. 9. The formalism-driven data synthesis pipeline of WebShaper. (Image source: [Tao et al., 2025](https://arxiv.org/abs/2507.15061))"
    align="center"
    width="100%"
>}}

*   **知识投影(Knowledge Projections, KP)**：WebShaper 首先基于集合论将信息寻求任务进行形式化。一个 **知识投影** \(R(V)\) 被定义为与实体集合 \(V\) 存在关系 \(R\) 的所有实体的集合。例如，`bornIn({1990s})` 表示所有在 1990 年代出生的实体集合。
*   **任务形式化**：复杂的查询可以被严谨地表示为多个 KP 的**交集** 和 **并集** 运算。例如，“查找在 2004-05 赛季效力于一支成立于 1966 年的东德球队，并且出生于 90 年代的球员”可以被形式化为多个 KP 的交集。
*   **Expander 智能体**：WebShaper 使用一个名为 **Expander** 的智能体，它首先生成一个形式化的查询结构（如三个 KP 的交集），然后通过调用工具（搜索、摘要）来逐步填充这个结构中的具体内容，并以“层级扩展”策略逐步增加问题的复杂度，从而有效避免了逻辑捷径和信息冗余。

## 奖励设计

奖励函数是 RL 的灵魂，它定义了智能体的优化目标。

*   **可验证奖励**：对于数学、代码等有明确答案的任务，这是最可靠和可扩展的奖励来源。奖励信号可以直接来自**单元测试通过率**、**代码编译器反馈**或**最终答案的正确性**。这种基于规则的奖励有效避免了奖励模型可能带来的 Reward hacking 问题。

*   **生成式奖励**：对于开放式、没有唯一答案的任务（如生成研究报告），采用 **LLM-as-a-Judge**([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)) 的方法使用一个强大的 LLM 作为裁判来评估生成结果的质量，并给出分数或自然语言反馈作为奖励信号。

*   **稠密奖励**：与只在任务结束时提供一次性奖励的**结果奖励模型** 不同，**过程奖励模型** 为智能体在任务过程中的每一步或每一个中间环节提供反馈。这有助于解决长时程任务中的信用分配难题，但同时也增加了标注成本和被模型利用的风险。

*   **无监督奖励**：为了摆脱对外部标注的依赖，研究者探索了从模型自身行为中提取奖励信号的方法，例如基于**输出一致性**（多次生成结果是否一致）或**内部置信度**（如生成概率的熵）来构建奖励。

## 优化算法

近年来，围绕 PPO、DPO、GRPO 等方法，衍生出了大量改进算法，下面简单介绍三个比较有代表性的算法。

### PPO

**近端策略优化 (Proximal Policy Optimization, PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) 是一种经典的 Actor-Critic 算法，因其在 **InstructGPT** ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) 中的成功应用而成为 LLM 强化学习微调的主流方法。PPO 的核心思想是在更新策略时，限制新旧策略之间的变化幅度，从而保证训练的稳定性。用 **token 级重要性比率** 和 **clip 裁剪** 限制新旧策略偏移，并借助一个 Critic 模型估计优势（将 sequence-level 奖励分解到 token-level），从而提升稳定性，但也带来额外的模型与计算开销。

$$
\mathcal{J}_{\mathrm{PPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{|y|} \sum_{t=1}^{|y|} \min \left(w_t(\theta) \widehat{A}_t, \operatorname{clip}\left(w_t(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_t\right)\right]
$$

其中，token $y_{t}$ 的重要性比率定义为 $w_t(\theta)=\frac{\pi_\theta\left(y_t \mid x, y_{<t}\right)}{\pi_{\theta_{\text {old }}}\left(y_t \mid x, y_{<t}\right)}$，$y_{t}$ 的优势 $\widehat{A}_{t}$ 由另一个 Critic 模型估计得到，$\varepsilon$ 是重要性比率的裁剪范围。


### GRPO

**组相对策略优化 (Group Relative Policy Optimization, GRPO)** ([Shao, et al. 2024](https://arxiv.org/abs/2402.03300)) 巧妙地移除了 Critic 模型。对于每个问题，它会采样一组 $G$ 输出来，通过计算每个输出在该组内的**相对优势**（即奖励值减去组内均值再除以标准差）来作为优势函数（优势是 sequence-level 的，但仍然在 token-level 上更新），降低了计算成本并提升了训练稳定性，下面公式省略了 KL 散度惩罚项，完整的可以参考博主之前写的 [GRPO](https://syhya.github.io/zh/posts/2025-01-27-deepseek-r1/#grpo)。

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D},\left\{y_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \min \left(w_{i, t}(\theta) \widehat{A}_{i, t}, \operatorname{clip}\left(w_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_{i, t}\right)\right]
$$

token $y_{i, t}$ 的重要性比率和优势分别为：

$$
w_{i, t}(\theta)=\frac{\pi_{\theta}\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}{\pi_{\theta_{\text{old}}}\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}, \quad \widehat{A}_{i, t}=\widehat{A}_{i}=\frac{r\left(x, y_{i}\right)-\operatorname{mean}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)}{\operatorname{std}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)},
$$

$y_{i}$ 内的所有 token 共享相同的优势 $\widehat{A}_{i}$, $G$ 表示每个查询 $x$ 生成的输出数量（即组大小）。

### GSPO

**组序列策略优化 (Group Sequence Policy Optimization, GSPO)** ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)) 将优化的基本单位从 token level 提升到 sequence-level。与 GRPO 使用 token level 重要性比率不同，GSPO 引入 **sequence-level** 重要性比率来对齐**序列级奖励**，从而避免长序列中 token 比率累积带来的噪声，降低方差并提升稳定性。Qwen 团队指出这样不仅能够在 MoE 中缓解由局部路由抖动导致的概率大幅波动，还能与 Agent 任务中普遍存在序列级奖励自然对齐，适用于长序列建模和对路由敏感的场景。

GSPO 的目标函数为：

$$
\mathcal{J}_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D},\left\{y_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{G} \sum_{i=1}^{G} \min \left(s_{i}(\theta) \widehat{A}_{i}, \operatorname{clip}\left(s_{i}(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_{i}\right)\right],
$$

其中，组内的优势函数定义为：

$$
\widehat{A}_{i}=\frac{r\left(x, y_{i}\right)-\operatorname{mean}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)}{\operatorname{std}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)}
$$

而序列级的重要性比率 $s_i(\theta)$ 定义为：

$$
s_i(\theta)=\left(\frac{\pi_\theta\left(y_i \mid x\right)}{\pi_{\theta_{\text{old}}}\left(y_i \mid x\right)}\right)^{\frac{1}{\left|y_i\right|}}=\exp \left(\frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \log \frac{\pi_\theta\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}{\pi_{\theta_{\text{old}}}\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}\right)
$$

它对 **整个序列** 应用裁剪，而不是对单个 token 裁剪，使优化与序列级奖励保持一致。这里采用了 **长度归一化** 来降低方差并控制 $s_i(\theta)$ 的数值范围，否则少量 token 的概率变化就可能导致比率剧烈波动，同时不同长度的响应也会导致不一致的裁剪范围。需要注意的是，由于重要性比率定义不同，GSPO 与 GRPO 的裁剪范围数量级通常并不相同。

### 框架

Agentic RL 的训练流程复杂，涉及推理、训练、奖励计算等多个环节，通常需要借助分布式框架来高效协调。

{{< figure
    src="dataflow_rlhf_verl.png"
    caption="Fig. 10. Dataflow graph of 3 RLHF algorithms including PPO, Safe-RLHF and ReMax. (Image source: [Sheng et al., 2024](https://arxiv.org/abs/2409.19256v2))"
    align="center"
    width="100%"
>}}

上图显示了不同算法训练过程中的复杂数据流。它不仅包含 Actor、Critic、Reference 和 Reward 等多种模型，还交织着数据生成、推理和训练等不同类型的计算负载。以基础的 **PPO 算法**为例，系统中涉及 4 个核心模型：Actor 负责根据输入 prompt 生成 response；Critic 用于评估结果；Reference 作为生成质量的基准；Reward 则提供奖励信号。从计算角度看，整个流程可分为 3 个阶段：

1. **Generation**：Actor 逐 token 生成 response，此过程受文本长度和生成方式影响，是推理资源和时间的主要消耗点。
2. **Forward (Rollout)**：生成结果与 query 一起输入 4 个模型进行一次前向计算，数据存入 [Replay Buffer](https://docs.ray.io/en/latest/rllib/rllib-replay-buffers.html)。
3. **Training**：从 Buffer 中采样数据，用于更新 Actor 与 Critic。

{{< figure
    src="disaggregated_colocated_arch.png"
    caption="Fig. 11. Two representative RL framework architectures. (Image source: [Zhong et al., 2025](https://arxiv.org/abs/2504.15930))"
    align="center"
    width="100%"
>}}

如图所示，常见的分布式调度策略分为两类：

1.  **时分共用 (Colocate)**：将 Rollout 和 Training 部署在同一组 GPU 上，通过时间片轮流执行。这种方式实现简单，通信开销小，但稳定性差，且无法利用异构硬件。
2.  **训推分离 (Disaggregated)**：将 Rollout 和 Training 分别部署在独立的 GPU 集群上。这种架构更灵活，稳定性更高，允许异构硬件混合部署，但可能引入流水线气泡，影响吞吐率。

### verl

**verl (Volcano Engine Reinforcement Learning)** ([Sheng et al., 2024](https://arxiv.org/abs/2409.19256v2)) 是字节跳动开源的一个为 LLM 设计的高效、通用的强化学习框架。

{{< figure
    src="verl_async_system_arch.png"
    caption="Fig. 12. The asynchronous system architecture of verl. (Image source: [ByteDance Seed, 2025](https://verl.readthedocs.io/en/latest/start/agentic_rl.html))"
    align="center"
    width="100%"
>}}

verl 的核心是**异步架构**，它将 Rollout、奖励计算和模型优化等阶段解耦，通过流水线并行处理，最大化硬件利用率。其流程如下：

1.  `PPOTrainer` 发起一次 PPO 迭代，先进行 **rollout**，再进行 **train**。
2.  `AgentLoopManager` 唤醒/同步推理与训练引擎权重（vLLM/SGLang ⇄ FSDP/Megatron-LM），将 batch 切成 chunk 分派给多个 `AgentLoopWorker` 并**并发**执行。
3.  每个 `AgentLoopWorker` 为一个样本启动一个协程，在需要生成时，通过 `AsyncLLMServerManager` 将请求路由到**负载最小**的推理实例，天然支持**多轮对话和多工具调用**。
4.  verl 原生支持对工具输出等外部信息进行**损失掩码 (Loss Masking)**，即在计算损失时忽略这些 token，只让模型为自己生成的内容负责。这是保证 Tool RL 训练稳定性的关键特性。

{{< figure
    src="verl_agent_loop_worker.png"
    caption="Fig. 13. The agent loop worker of verl based on React. (Image source: [ByteDance Seed, 2025](https://verl.readthedocs.io/en/latest/start/agentic_rl.html))"
    align="center"
    width="100%"
>}}

5. rollout 结束后统一回收/休眠实例以释放显存；通过可插拔接口自定义**奖励函数**、集成新**工具**或替换**RL 算法**（如在 `ReactAgentLoop` 上派生自定义 Agent）。这里默认使用 LangGraph 框架实现 [ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/?h=react)。

## 案例研究

### Search-R1
**Search-R1** ([Jin et al., 2025](https://arxiv.org/abs/2503.09516)) 训练 LLM 在逐步推理的过程中，自主地与搜索引擎进行多轮交互，从而学会何时以及如何利用外部知识。

{{< figure
    src="search_r1_template.png"
    caption="Fig. 14. Template for SEARCH-R1. question will be replaced with the specific question during training and inference. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

在 RL 训练的轨迹中，模型与环境的交互遵循以下步骤：
1.  在 `<think>...</think>` 标签内进行**自主推理**。
2.  当意识到知识不足时，生成 `<search>query</search>` 标签来**调用搜索引擎**。
3.  环境执行搜索后，将检索到的信息包裹在 `<information>...</information>` 标签内反馈给模型。
4.  模型根据新的信息继续推理，并可以进行多轮搜索，直到最终在 `<answer>...</answer>` 标签内给出答案。


{{< figure
    src="search_r1_rollout.png"
    caption="Fig. 15. LLM Response Rollout with Multi-Turn Search Engine Calls. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}


这个过程通过一个循环算法实现：模型生成文本直到遇到 `<search>` 或 `<answer>` 标签。如果检测到搜索请求，系统会暂停生成，执行搜索，并将结果用 `<information>` 标签包裹后注入到上下文中，供模型在下一步继续推理。这个循环会持续进行，直到模型生成最终答案或达到最大交互次数。

论文采用 PPO 与 GRPO 算法进行训练。为保证训练稳定性，Search-R1 引入了**检索 token 掩码**机制：在计算 RL 损失（策略梯度与 KL 散度）时，所有由搜索引擎返回并包裹在 `<information>` 标签内的 retrieved token 均被**masking**，其损失不参与梯度更新。这一设计使模型专注于学习**何时及如何进行推理和搜索**，而非机械模仿外部检索内容，从而有效避免训练不稳定。下图结果显示使用该策略训练后的模型（Qwen2.5-7b-base，经 PPO 训练）在性能上始终优于未使用掩码的版本。

{{< figure
    src="search_r1_token_mask.png"
    caption="Fig. 16. The performance of SEARCH-R1 with and without retrieved token loss masking. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

Search-R1 采用了一个简洁的基于结果的奖励函数，仅根据最终答案的正确性进行评分，使用精确匹配作为评判标准。

$$
R(y, \hat{y}_{\text{gold}}) = \text{EM}(y_{\text{answer}}, \hat{y}_{\text{gold}}) = 
\begin{cases} 
1, & \text{if } y_{\text{answer}} = \hat{y}_{\text{gold}} \\ 
0, & \text{otherwise} 
\end{cases}
$$

这种简单的奖励设计被证明是有效的，足以引导模型学习出复杂的搜索和推理行为。

{{< figure
    src="search_r1_result.png"
    caption="Fig. 17. The main results comparing SEARCH-R1 with baseline methods across the seven datasets. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

实验表明，在 Qwen2.5-7B 和 Qwen2.5-3B 模型上，相较于传统的 RAG 基线，Search-R1 分别取得了 **24%** 和 **20%** 的平均相对性能提升。

### ReTool

**ReTool** ([Luo et al., 2025](https://arxiv.org/abs/2504.11536)) 是一个基于 verl 框架训练模型何时以及如何调用**代码解释器 (Code Interpreter, CI)** 来解决数学问题案例。

{{< figure
    src="ReTool_arch.png"
    caption="Fig. 18. The architecture of ReTool. (Image source: [Luo et al., 2025](https://arxiv.org/abs/2504.11536))"
    align="center"
    width="80%"
>}}

ReTool 采用“**冷启动 SFT → 工具增强 RL**”两阶段流程：

1.  **冷启动 SFT**：首先构建包含代码增强推理轨迹的数据集，通过 SFT 使模型掌握基础的工具调用能力。
2.  **工具增强 RL**：在 RL 阶段，模型在解决问题的过程中可以生成 `<code>...</code>` 代码片段，这些代码会在沙盒环境（如 [SandboxFusion](https://github.com/bytedance/SandboxFusion)）中执行，执行结果（包括输出或错误栈）会被包裹在 `<interpreter>...</interpreter>` 标签中反馈给模型，模型可以根据反馈继续推理或进行**自我纠错**。

{{< figure
    src="ReTool_self_correction.png"
    caption="Fig. 19. The case of “aha moment” about code self-correction. (Image source: [Luo et al., 2025](https://arxiv.org/abs/2504.11536))"
    align="center"
    width="100%"
>}}

其奖励函数仅依据最终答案的正确性打分，鼓励模型自主探索稳健的推理-执行策略。

$$
R(a,\hat a)=
\begin{cases}
1, & \text{is_equivalent}(a,\hat a) \\
-1, & \text{otherwise}
\end{cases}
$$

基于 PPO 算法进行训练，类似于 Search-R1 对解释器反馈 `<interpreter>...</interpreter>` 进行**全量 loss mask**，只更新模型思考与代码，避免梯度被外部环境噪声污染。下图结果显示在 AIME 2024/2025 评测中，ReTool 仅用 400 步 RL 就使 Qwen2.5-32B-Instruct 的准确率达到 67.0% / 49.3%，超过纯文本 RL 基线，同时平均响应长度缩短了约 40%。

{{< figure
    src="ReTool_aime.png"
    caption="Fig. 20. AIME 2024 & 2025 scores of ReTool and text-based RL baseline on the Qwen2.5-32B-Instruct model. (Image source: [Luo et al., 2025](https://arxiv.org/abs/2504.11536))"
    align="center"
    width="100%"
>}}

## 参考文献

[1] Zhang, Guibin, et al. ["The landscape of agentic reinforcement learning for llms: A survey."](https://arxiv.org/abs/2509.02547) arXiv preprint arXiv:2509.02547 (2025).

[2] Wei, Jason. ["Successful language model evals."](https://www.jasonwei.net/blog/evals) Blog post, 2024.

[3] Liang, Percy, et al. ["Holistic evaluation of language models."](https://arxiv.org/abs/2211.09110) arXiv preprint arXiv:2211.09110 (2022).

[4] Srivastava, Aarohi, et al. ["Beyond the imitation game: Quantifying and extrapolating the capabilities of language models."](https://arxiv.org/abs/2206.04615) Transactions on machine learning research (2023).

[5] Wang, Alex, et al. ["GLUE: A multi-task benchmark and analysis platform for natural language understanding."](https://arxiv.org/abs/1804.07461) arXiv preprint arXiv:1804.07461 (2018).

[6] Wang, Alex, et al. ["SuperGLUE: A stickier benchmark for general-purpose language understanding systems."](https://arxiv.org/abs/1905.00537) Advances in neural information processing systems 32 (2019).

[7] Wei, Jason. ["Asymmetry of verification and verifier’s rule."](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) Blog post, 2025.

[8] Jimenez, Carlos E., et al. ["SWE-bench: Can language models resolve real-world github issues?."](https://arxiv.org/abs/2310.06770) arXiv preprint arXiv:2310.06770 (2023).

[9] OpenAI. ["Introducing SWE-bench Verified."](https://openai.com/index/introducing-swe-bench-verified/) OpenAI, 2024 (updated 2025).

[10] Wei, Jason, et al. ["Browsecomp: A simple yet challenging benchmark for browsing agents."](https://arxiv.org/abs/2504.12516) arXiv preprint arXiv:2504.12516 (2025).

[11] Su, Liangcai, et al. ["Scaling Agents via Continual Pre-training."](https://arxiv.org/abs/2509.13310) arXiv preprint arXiv:2509.13310 (2025).

[12] Tao, Zhengwei, et al. ["Webshaper: Agentically data synthesizing via information-seeking formalization."](https://arxiv.org/abs/2507.15061) arXiv preprint arXiv:2507.15061 (2025).

[13] Zheng, Lianmin, et al. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."](https://arxiv.org/abs/2306.05685) Advances in neural information processing systems 36 (2023): 46595-46623.

[14] Schulman, John, et al. ["Proximal policy optimization algorithms."](https://arxiv.org/abs/1707.06347) arXiv preprint arXiv:1707.06347 (2017).

[15] Shao, Zhihong, et al. ["Deepseekmath: Pushing the limits of mathematical reasoning in open language models."](https://arxiv.org/abs/2402.03300) arXiv preprint arXiv:2402.03300 (2024).

[16] Zheng, Chujie, et al. ["Group sequence policy optimization."](https://arxiv.org/abs/2507.18071) arXiv preprint arXiv:2507.18071 (2025).

[17] Sheng, Guangming, et al. ["Hybridflow: A flexible and efficient rlhf framework."](https://arxiv.org/abs/2409.19256v2) Proceedings of the Twentieth European Conference on Computer Systems. 2025.
 
[18] Zhong, Yinmin, et al. ["StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation."](https://arxiv.org/abs/2504.15930) arXiv preprint arXiv:2504.15930 (2025).

[19] Jin, Bowen, et al. ["Search-r1: Training llms to reason and leverage search engines with reinforcement learning."](https://arxiv.org/abs/2503.09516) arXiv preprint arXiv:2503.09516 (2025).

[20] Feng, Jiazhan, et al. ["Retool: Reinforcement learning for strategic tool use in llms."](https://arxiv.org/abs/2504.11536) arXiv preprint arXiv:2504.11536 (2025).

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (September 2025). Agentic RL
> https://syhya.github.io/zh/posts/2025-09-30-agentic-rl/

Or

```bibtex
@article{yue_shui_agentic_rl_2025,
  title   = "Agentic RL",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "September",
  url     = "https://syhya.github.io/zh/posts/2025-09-30-agentic-rl/"
}
```