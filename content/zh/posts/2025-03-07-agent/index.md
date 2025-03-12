---
title: "大语言模型智能体（长期更新中）"
date: "2025-03-09T10:00:00+00:00"
author: "Yue Shui"
tags: ["LLM", "AI", "Agent"]
categories: ["技术博客"]
toc: true
draft: false
---

> **注意**: 本文**正在更新中**，内容只是**草稿版本**，并不完善，后续会有变动。请随时关注最新版本。

## Agent

**大语言模型智能体(Large Language Model Agent, LLM agent)** 利用 LLM 作为系统大脑，并结合规划、记忆与外部工具等模块，实现了对复杂任务的自动化执行。

- **用户请求 (User Request):** 用户通过 prompt 输入任务，与智能体互动。  
- **智能体 (Agent):** 系统大脑，由一个或多个大语言模型构成，负责整体协调和执行任务。  
- **规划 (Planning):** 将复杂任务拆解为更小的子任务，并制定执行计划，同时通过反思不断优化结果。  
- **记忆 (Memory):** 包含短期记忆(利用上下文学习即时捕捉任务信息)和长期记忆(采用外部向量存储保存和检索关键信息，确保长时任务的信息连续性)。  
- **工具 (Tools):** 集成计算器、网页搜索、代码解释器等外部工具，用于调用外部数据、执行代码和获取最新信息。

{{< figure
    src="llm_agent.png"
    caption="Fig. 1. The illustration of LLM Agent Framework. (Image source: [DAIR.AI, 2024](https://www.promptingguide.ai/research/llm-agents#llm-agent-framework))"
    align="center"
    width="70%"
>}}

**强化学习(Reinforcement Learning，RL)** 的目标是训练一个智能体(agent)在给定的环境 (environment) 中采取一系列动作(actions, \(a_t\))。在交互过程中，智能体从一个状态(state, \(s_t\))转移到下一个状态，并在每次执行动作后获得环境反馈的奖励(reward, \(r_t\))。这种交互生成了完整的轨迹(trajectory, \(\tau\))，通常表示为：

$$
\tau = \{(s_0, a_0, r_0), (s_1, a_1, r_1), \dots, (s_T, a_T, r_T)\}.
$$

智能体的目标是学习一个策略(policy, \(\pi\))，即在每个状态下选择动作的规则，以**最大化期望累积奖励**，通常表达为：  

$$
\max_{\pi} \, \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right],
$$
其中 \(\gamma \in [0,1]\) 为折扣因子，用于平衡短期与长期奖励。

{{< figure
    src="rl_agent.png"
    caption="Fig. 2. The agent-environment interaction. (Image source: [Sutton & Barto, 2017](http://incompleteideas.net/book/the-book.html))"
    align="center"
    width="80%"
>}}

在 **LLM** 场景中，可以将模型视为一个智能体，而“环境”可理解为用户输入及其对应的期望回答方式：

- **状态(\(s_t\))**：可以是当前对话上下文或用户的问题。  
- **动作(\(a_t\))**：模型输出的文本(回答、生成内容等)。  
- **奖励(\(r_t\))**：来自用户或系统的反馈(如用户满意度、奖励模型自动评分等)。  
- **轨迹(\(\tau\))**：从初始对话到结束的所有文本交互序列，可用于评估模型的整体表现。  
- **策略(\(\pi\))**：大语言模型在每个状态下(对话上下文)如何生成文本的规则，一般由模型的参数所决定。


对于 LLM，传统上先通过海量离线数据进行预训练，而在后训练强化学习环节中，则通过人类或者模型的反馈对模型进行训练，使其输出更符合人类偏好或任务需求的高质量文本。

下表展示了两者之间的差异：

| **对比维度**     | **LLM Agent**                                                          | **RL Agent**                                                    |
|------------------|------------------------------------------------------------------------|------------------------------------------------------------------|
| **核心原理**     | 规划、记忆和工具实现复杂任务自动化。   | 通过与环境互动的试错反馈循环，不断优化策略以最大化长期奖励。     |
| **优化方式** | **不直接更新模型参数**，主要依靠上下文扩展、外部记忆和工具提高性能。     | **持续频繁更新策略模型参数**，依赖环境反馈的奖励信号不断优化。       |
| **互动方式**     | 使用自然语言与用户或外部系统交互，灵活调用多种工具获得外部信息。       | 与真实或模拟环境交互，环境提供奖励或惩罚，形成闭环反馈。         |
| **实现目标**     | 分解复杂任务、借助外部资源完成任务，关注任务结果质量与准确性。         | 最大化长期奖励，追求短期与长期回报之间的最优平衡。              |

随着研究的深入，LLM 与 RL 智能体的结合呈现出更多可能性，例如：
- 利用强化学习方法训练 Reasoning LLM(如 o1/o3)，使其更适合作为 LLM 智能体的基础模型。
- 同时，记录 LLM 智能体执行任务的数据与反馈，为 Reasoning LLM 提供丰富的训练数据，从而提升模型性能。


## 规划、记忆与工具


LLM Agent 的核心组件包括**规划**、**记忆**和**工具使用**，这些组件共同协作，使智能体能够自主执行复杂任务。


{{< figure
    src="llm_agent_overview.png"
    caption="Fig. 3. Overview of a LLM-powered autonomous agent system. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}


规划对于成功执行复杂任务至关重要。它可以根据复杂性和迭代改进的需求以不同的方式进行。在简单的场景中，规划模块可以使用 LLM 预先概述详细的计划，包括所有必要的子任务。此步骤确保智能体系统地进行 **任务分解(Task Decomposition)** 并从一开始就遵循清晰的逻辑流程。这通常通过使用诸如以下提示技术来实现。

### 思维链

**思维链(Chain of Thought, CoT)** ([Wei et al. 2022](https://arxiv.org/abs/2201.11903)) 通过逐步生成一系列简短句子来描述推理过程，这些句子称为推理步骤。其目的是显式地展示模型的推理路径，帮助模型更好地处理**复杂推理任务**。下图展示了少样本提示（左侧）和思维链提示（右侧）的区别。少样本提示得到错误答案，而思维链方法则引导模型逐步陈述推理过程，更清晰地体现模型的逻辑过程，从而提升答案准确性和可解释性。

{{< figure
    src="cot.png"
    caption="Fig. 4. The comparison example of few-shot prompting and CoT prompting. (Image source: [Weng, 2023](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}


**零样本思维链提示(Zero-Shot CoT)** ([Kojima et al. 2022](https://arxiv.org/abs/2205.11916))是 CoT 的后续研究，提出了一种极为简单的零样本提示方式。他们发现，仅需在问题末尾添加一句 `Let's think step by step`，LLM 便能够产生思维链，可以获得更为准确的答案。


{{< figure
    src="zero_shot_cot.png"
    caption="Fig. 5. The comparison example of few-shot prompting and CoT prompting. (Image source: [Kojima et al. 2022](https://arxiv.org/abs/2205.11916))"
    align="center"
    width="100%"
>}}

### 自洽采性

**自洽采样(Self-consistency sampling)**([Wang et al. 2022a](https://arxiv.org/abs/2203.11171)) 通过对同一提示词在 `temperature > 0` 的情况下多次采样，生成**多个多样化答案**，并从中选出最佳答案的方法。其核心思想是通过采样多个推理路径，再进行多数投票以提高最终答案的准确性和稳健性。不同任务中选择最佳答案的标准可以有所不同，一般情况下采用**多数投票**作为通用方案。而对于如编程题这类易于验证的任务，则可以通过解释器运行并结合单元测试对答案进行验证。这是一种对 CoT 的优化，与其结合使用时，能够显著提升模型在复杂推理任务中的表现。

{{< figure
    src="self_consistency.png"
    caption="Fig. 6. Overview of the Self-Consistency Method for Chain-of-Thought Reasoning. (Image source: [Wang et al. 2022a](https://arxiv.org/abs/2203.11171))"
    align="center"
    width="100%"
>}}


下面是一些后续优化的工作：

- ([Wang et al. 2022b](https://arxiv.org/abs/2207.00747))后续采用另一种集成学习方法进行优化, 通过改变示例顺序或以模型生成的推理代替人为书写，增加随机性后再多数投票。

{{< figure
    src="rationale_augmented.png"
    caption="Fig. 7. An overview of different ways of composing rationale-augmented ensembles, depending on how the randomness of rationales is introduced. (Image source: [Wang et al. 2022b](https://arxiv.org/abs/2207.00747))"
    align="center"
    width="100%"
>}}

- 如果训练样本仅提供正确答案而无推理依据，可采用STaR（Self-Taught Reasoner）([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465))方法：
(1) 让LLM生成推理链，仅保留正确答案的推理。
(2) 用生成的推理微调模型，反复迭代直至收敛。注意 `temperature` 高时易生成带正确答案但错误推理的结果。如无标准答案，可考虑将多数投票视作“正确答案”。

{{< figure
    src="STaR.png"
    caption="Fig. 8. An overview of STaR and a STaR-generated rationale on CommonsenseQA (Image source: [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465))"
    align="center"
    width="100%"
>}}

- ([Fu et al. 2023](https://arxiv.org/abs/2210.00720))发现复杂程度更高的示例（推理步骤更多）可提高模型性能。分隔推理步骤时，换行符 `\n` 效果优于`step i`、`.` 或 `;`。此外，通过复杂度一致性策略(Complexity-based consistency)，即仅对生成复杂度 top $k$ 的推理链进行多数投票，也能进一步优化模型输出。同时，将提示中的 `Q:` 替换为 `Question:` 也被证明对性能有额外提升。

{{< figure
    src="linebreak.png"
    caption="Fig. 9. Sensitivity analysis on step formatting. Complex prompts consistently lead to better performance with regard to different step formatting. (Image source: [Fu et al. 2023)](https://arxiv.org/abs/2210.00720))"
    align="center"
    width="100%"
>}}


### 思维树

**思维树 (Tree of Thoughts, ToT)**([Yao et al. 2023](https://arxiv.org/abs/2305.10601)) 在 CoT 基础上拓展，每一步都探索多个推理可能性。它首先将问题分解为多个思考步骤，并在每一步产生多个不同想法，从而形成树形结构。搜索过程可采用广度优先搜索（Breadth-first search, BFS）或深度优先搜索（Depth-first search, DFS），并通过分类器（也可以让 LLM 进行打分）或多数票方式评估每个状态。它包含三个主要步骤：

- 扩展(Expand)：生成一个或多个候选解决方案。
- 评分(Score)：衡量候选方案的质量。
- 剪枝(Prune)：保留排名 top $k$ 的最佳候选方案。

如果没有找到解决方案（或者候选方案的质量不够高），则回撤到扩展步骤。

{{< figure
    src="tot.png"
    caption="Fig. 10. Schematic illustrating various approaches to problem solving with LLMs (Image source: [Yao et al. 2023](https://arxiv.org/abs/2305.10601))"
    align="center"
    width="100%"
>}}


### Self-Reflexion

自我反思是使自主代理通过改进过去的行动决策和纠正以往错误而实现迭代提升的关键因素。在试错不可避免的现实任务中，它起着至关重要的作用。

### ReAct

**ReAct(Reason + Act)** ([Yao et al. 2023](https://arxiv.org/abs/2210.03629)) 框架通过将任务特定的离散动作和语言空间相结合，实现了大语言模型（LLM）中推理与行动的无缝整合。这种设计不仅使模型能够通过调用例如 Wikipedia 搜索 API 等外部接口与环境进行交互，同时还能以自然语言生成详细的推理轨迹，从而解决复杂问题。

ReAct 提示模板包含明确的思考步骤，其基本格式如下：

```plaintext
Thought：...
Action：...
Observation：...
...(Repeated many times)
```

{{< figure
    src="ReAct.png"
    caption="Fig. 12. Examples of reasoning trajectories for knowledge-intensive tasks (e.g. HotpotQA, FEVER) and decision-making tasks (e.g. AlfWorld Env, WebShop). (Image source: [Yao et al. 2023](https://arxiv.org/abs/2210.03629))"
    align="center"
    width="100%"
>}}

从下图我们可以看出在知识密集型任务和决策任务中，ReAct 的表现均明显优于仅依赖`Actor`的基础方法，从而展示了其在提升推理效果和交互性能方面的优势。

{{< figure
    src="ReAct_res.png"
    caption="Fig. 13. PaLM-540B prompting results on HotpotQA and Fever. (Image source: [Yao et al. 2023](https://arxiv.org/abs/2210.03629))"
    align="center"
    width="50%"
>}}


### Reflexion

**Reflexion**([Shinn et al. 2023](https://arxiv.org/abs/2303.11366))让 LLM 能够通过自我反馈与动态记忆不断迭代、优化决策。

这种方法本质上借鉴了强化学习的思想，在传统的 Actor-Critic 模型中，Actor 根据当前状态 $s_t$ 选择动作 $a_t$，而 Critic 则会给出估值（例如价值函数 $V(s_t)$ 或动作价值函数 $Q(s_t,a_t)$），并反馈给 Actor 进行策略优化。对应地，在 Reflexion 的三大组件中：

- **Actor**：由 LLM 扮演，基于环境状态（包括上下文和历史信息）输出文本及相应动作。可记为：  

  $$
  a_t = \pi_\theta(s_t),
  $$

  其中 $\pi_\theta$ 表示基于参数 $\theta$（即 LLM 的权重或提示）得到的策略。Actor 与环境交互并产生轨迹 $\tau = \{(s_1,a_1,r_1), \dots, (s_T,a_T,r_T)\}$。

- **Evaluator**：类似于 Critic，Evaluator 接收由 Actor 生成的轨迹并输出奖励信号 $r_t$。在 Reflexion 框架里，Evaluator 可以通过预先设计的启发式规则或额外的 LLM 来对轨迹进行分析，进而产生奖励。例如：  

  $$
  r_t = R(\tau_t),
  $$

  其中 $R(\cdot)$ 为基于当前轨迹 $\tau_t$ 的奖励函数。

- **Self-Reflection**：该模块相当于在 Actor-Critic 之外额外增加了自我调节反馈机制。它整合当前轨迹 $\tau$、奖励信号 $\{r_t\}$ 以及长期记忆中的历史经验，利用语言生成能力产生针对下一次决策的自我改进建议。这些反馈信息随后被写入外部记忆，为后续 Actor 的决策提供更丰富的上下文，从而在不更新 LLM 内部参数的情况下，通过提示词的动态调整实现类似于策略参数 $\theta$ 的迭代优化。

{{< figure
    src="Reflexion.png"
    caption="Fig. 14. (a) Diagram of Reflexion. (b) Reflexion reinforcement algorithm (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

Reflexion 的核心循环与算法描述如下：

- **初始化**  
   - 同时实例化 Actor、Evaluator、Self-Reflection 三个模型（均可由 LLM 实现），分别记为 $M_a, M_e, M_{sr}$。  
   - 初始化策略 $\pi_\theta$（包含 Actor 的模型参数或提示，以及初始记忆等信息）。  
   - 先让 Actor 按当前策略 $\pi_\theta$ 生成一个初始轨迹 $\tau_0$，$M_e$ 进行评估后，再由 $M_{sr}$ 生成首条自我反思文本并存入长期记忆。

- **生成轨迹**  
   - 在每一次迭代中，$M_a$ 读取当前的长期记忆及环境观测，依次输出动作 $\{a_1, a_2, \ldots\}$，与环境交互并获得相应反馈，形成新的轨迹 $\tau_t$。 $\tau_t$ 可以视作该任务的短期记忆 (short-term memory)，仅在本轮迭代中使用。

- **评估**  
   - $M_e$ 根据轨迹 $\tau_t$（即 Actor 的行为与环境反馈序列），输出奖励或评分 $\{r_1, r_2, \ldots\}$。这一步对应 $M_e$ 的内部反馈 (Internal feedback)，或由外部环境直接给出结果。

- **自我反思**  
   - $M_{sr}$ 模块综合轨迹 $\tau_t$ 与奖励信号 $\{r_t\}$ 在语言层面生成自我修正或改进建议 $\mathrm{sr}_t$。  
   - 反思文本 (Reflective text) 既可视为对错误的剖析，也可提供新的启发思路，并通过存储到长期记忆 (long-term memory) 中。实践中，我们可以将反馈信息向量化后存入向量数据库。

- **更新并重复**  
   - 将最新的自我反思文本 $\mathrm{sr}_t$ 追加到长期记忆后，Actor 在下一轮迭代时即可从中采用 RAG 检索历史相关信息来调整策略。  
   - 反复执行上述步骤，直至 $M_e$ 判定任务达成或到达最大轮次。在此循环中，Reflexion 依靠 **自我反思 + 长期记忆** 的持续累加来改进决策，而非直接修改模型参数。

下面分别展示了 Reflexion 在决策制定、编程和推理任务中的应用示例：

{{< figure
    src="reflextion_examples.png"
    caption="Fig. 15. Reflexion works on decision-making 4.1, programming 4.3, and reasoning 4.2 tasks (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

在 100 个 HotPotQA 问题的实验中，通过对比 CoT 方法和加入 episodic memory 的方式，结果显示采用 Reflexion 方法在最后增加自我反思步骤后，其搜索、信息检索和推理能力提升明显。

{{< figure
    src="reflextion_result.png"
    caption="Fig. 15. Comparative Analysis of Chain-of-Thought (CoT) and ReAct on the HotPotQA Benchmark (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}


### 后见链（Chain of Hindsight, CoH）

Chain of Hindsight (CoH；Liu 等人，2023) 鼓励模型通过明确展示一系列带有反馈注释的过去输出，从而改进自身输出。人类反馈数据是一系列数据，其中包含提示、模型输出、人工评分以及对应的人类后见反馈。假设这些反馈元组按奖励排序，整个过程采用监督式微调，其数据序列的形式为  ，其中 ；模型经过微调后仅在给定序列前缀条件下预测 以实现自我反思，从而基于反馈序列生成更优输出。模型在测试时还可以选择接收多轮来自人工标注者的指令。

为防止过拟合，CoH 添加了正则化项以最大化预训练数据集的对数似然。为了避免捷径效应和复制（因为反馈序列中存在许多常见词），训练过程中会随机屏蔽 0% 至 5% 的历史 token。

他们实验中的训练数据集由 WebGPT 对比数据、人类反馈摘要和人类偏好数据集组合而成。

图5显示，经过 CoH 微调后，模型能够按照指令生成逐步改进的输出序列。（图片来源：Liu 等人，2023）

CoH 的理念是将一系列逐步改进的输出作为上下文呈现，并训练模型顺应这种趋势产生更好的输出。

### 算法蒸馏（Algorithm Distillation, AD）

算法蒸馏（AD；Laskin 等人，2023）将相同理念应用于强化学习任务中跨集数的轨迹，其核心是将一个算法封装进长期历史条件下的策略中。考虑到代理与环境多次交互并在每个试验中都有所进步，AD 将这种学习历史串联起来并输入到模型中，因此我们预期下一次预测的动作能带来比之前试验更好的表现。其目标在于学习强化学习的过程，而非仅训练针对特定任务的策略。

图6展示了算法蒸馏（AD）工作原理的示意图。（图片来源：Laskin 等人，2023）

论文假设，任何生成一系列学习历史的算法，都可以通过对动作进行行为克隆而被蒸馏进神经网络中。这些历史数据由一组针对特定任务训练的源策略生成。在训练阶段，每次强化学习运行中会随机抽取一个任务，并使用多集数历史中的一个子序列进行训练，从而使所学策略具有任务无关性。

实际上，由于模型的上下文窗口长度有限，各试验应足够短以构建多集数历史。构建近似最优上下文内强化学习算法需要 2 到 4 个试验的多集数上下文，而实现上下文内强化学习的前提是足够长的上下文信息。

与三种基线方法（包括 ED（专家蒸馏——使用专家轨迹而非学习历史进行行为克隆）、源策略（通过 UCB 生成用于蒸馏的轨迹）、RL²（Duan 等人，2017；由于需要在线强化学习，作为上界使用））相比，AD 展现了上下文内强化学习的能力，其性能虽只依赖离线强化学习却接近 RL²，并且学习速度远快于其他基线方法。当以源策略的部分训练历史作为条件时，AD 的提升速度也远快于 ED 基线。


#### Self-Ask

Self-Ask（Press et al. 2022）则通过不断提示模型提出后续问题来迭代构建思维过程。后续问题可以通过搜索引擎结果来回答。类似地，IRCoT（交替检索的 CoT；Trivedi et al. 2022）和 ReAct（推理 + 行动；Yao et al. 2023）将迭代式的 CoT 提示与对 Wikipedia API 的查询相结合，以搜索相关实体和内容，再将其添加回上下文。


## 记忆

### 记忆的类型

记忆可以定义为获取、存储、保持和之后提取信息的过程。人类大脑中的记忆主要分为以下几种：

- **感觉记忆（Sensory Memory）**：
这是记忆的最初阶段，它使我们能在原始刺激消失后短暂地保留感官信息（视觉、听觉等）的印象。感觉记忆通常只能持续数秒钟。其子类别包括视觉记忆（Iconic memory）、听觉记忆（Echoic memory）以及触觉记忆（Haptic memory）。

- **短期记忆（STM）或工作记忆（Working Memory）**：
短期记忆存储的是我们当前意识到的信息，它在我们执行学习和推理等复杂认知任务时发挥作用。短期记忆的容量大约是7个项目，持续时间约为20到30秒。

- **长期记忆（LTM）**：
长期记忆能够长时间地存储信息，时间范围可以从数天到数十年，其存储容量几乎是无限的。长期记忆可进一步分为两类：

    - **外显记忆（Explicit / declarative memory）**：
      指关于事实和事件的记忆，这些记忆能够被有意识地回忆。外显记忆又可细分为情景记忆（Episodic memory，关于个人经历和事件）和语义记忆（Semantic memory，关于事实和概念）。

    - **内隐记忆（Implicit / procedural memory）**：
      指无意识的记忆，涉及自动执行的技能和日常行为，例如骑自行车或打字。

{{< figure
    src="category_human_memory.png"
    caption="Fig. xx. Categorization of human memory. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}


我们大致可以做以下类比映射：
- 感觉记忆对应于学习输入原始数据（如文本、图像或其他模态）的嵌入表征（embedding representations）。
- 短期记忆对应于上下文内学习（in-context learning），其容量短暂且有限，受限于 LLM 有限的上下文窗口长度`max_tokens`, 当对话历史超过上限时，较早的信息会被截断，从而造成上下文丢失。
- 长期记忆对应于外部的向量存储（vector store），智能体可以通过 RAG 的方式在查询时快速访问和检索。


### 长期记忆相关研究

**LongMem(Language Models Augmented with Long-Term Memory)**([Wang, et al. 2023](https://arxiv.org/abs/2306.07174))使 LLM 能够记忆长历史信息。其采用一种解耦的网络结构，将原始 LLM 参数冻结为记忆编码器(memory encoder)固定下来，同时使用自适应残差网络(Adaptive Residual Side-Network，SideNet)作为记忆检索器进行记忆检查和读取。

{{< figure
    src="LongMem.png"
    caption="Fig. xx. Overview of the memory caching and retrieval flow of LongMem. (Image source: [Wang, et al. 2023](https://arxiv.org/abs/2306.07174))"
    align="center"
    width="100%"
>}}
 
其主要由三部分构成：**Frozen LLM**、**Residual SideNet** 和 **Cached Memory Bank**。其工作流程是，先将长文本序列拆分成固定长度的片段，每个片段在冻结的 LLM 中逐层编码后，在第 $m$ 层提取注意力的 $K, V \in \mathbb{R}^{H \times M \times d}$ 向量对并缓存到 **Cached Memory Bank**。面对新的输入序列时，模型根据当前输入的 query-key 检索长期记忆库，从中获取与输入最相关的前 $k$ 个 key-value（即 *top-$k$* 检索结果），并将其融合到后续的语言生成过程中；与此同时，记忆库会移除最旧的内容以保证最新上下文信息的可用性。**Residual SideNet** 则在推理阶段对冻结 LLM 的隐藏层输出与检索得到的历史 key-value 进行融合，完成对超长文本的有效建模和上下文利用。通过这种解耦设计，LongMem 无需扩大自身的原生上下文窗口就能灵活调度海量历史信息，兼顾了速度与长期记忆能力。


## 最大内积搜索 (Maximum Inner Product Search, MIPS)

为了有效地从长期记忆中检索相关信息，特别是从大型向量存储中检索，采用了最大内积搜索 (MIPS) 技术。这些技术专注于快速查找与查询向量具有最高相似度(内积)的向量。近似最近邻 (Approximate Nearest Neighbor, ANN) 算法通常用于优化检索速度，以少量精度损失换取速度的显着提升。流行的 MIPS 技术包括：

- **局部敏感哈希 (Locality-Sensitive Hashing, LSH):** 使用哈希函数将相似的项目映射到相同的桶中，从而通过关注相关桶来实现更快的搜索。
- **ANNOY (近似最近邻，哦耶) (Approximate Nearest Neighbors Oh Yeah):** 使用随机投影树来划分数据空间并有效地搜索最近邻。
- **HNSW (分层可导航小世界) (Hierarchical Navigable Small World):** 构建小世界图的分层层，以创建快捷方式，从而在多维空间中实现更快的导航和搜索。
- **FAISS (Facebook AI 相似性搜索) (Facebook AI Similarity Search):** 应用向量量化来聚类数据点并细化聚类内的量化，以实现高效搜索。
- **ScaNN (可扩展最近邻) (Scalable Nearest Neighbors):** 使用各向异性向量量化来优化内积相似性搜索，重点是精确的距离近似。

这些 ANN 算法使从大型向量数据库进行实时检索成为可能，从而使智能体能够有效地利用长期记忆。

# 组件三：工具使用 (Tool Use)

为 LLM 智能体配备调用外部工具的能力，可以显着扩展其能力，使其能够访问信息并执行超出 LLM 预训练知识范围的操作。

- **MRKL (模块化推理、知识和语言) 系统 (Modular Reasoning, Knowledge and Language Systems):** 将 LLM 与专家模块相结合，专家模块可以是神经模块或符号模块(例如，计算器、API)。 LLM 充当路由器，将查询定向到最适合特定子任务的专家模块。
- **Toolformer 和函数调用 (Function Calling):** 这些方法微调 LLM 以自动使用外部 API。 Toolformer 学习预测何时以及如何使用工具，方法是评估 API 调用在提高模型输出质量方面的效用。函数调用，如 OpenAI API 中所示，允许开发人员定义工具 API 并将其提供给 LLM，使其能够根据需要调用这些函数。
- **HuggingGPT:** 一个新兴框架，它使用 ChatGPT 作为任务规划器来协调 Hugging Face 平台上可用的各种 AI 模型。 HuggingGPT 将用户请求分解为任务，根据模型描述从 Hugging Face 中选择合适的模型，使用这些模型执行任务，并为用户总结结果。

通过集成这些工具，LLM 智能体可以执行复杂的任务，例如生成代码、访问实时信息、与数据库交互以及控制外部系统，从而极大地扩展其解决问题的能力。

# LLM 智能体应用 (LLM Agent Applications)

LLM 智能体已成功应用于各个领域，展示了其多功能性和潜力：

- **科学发现 (Scientific Discovery):**
    - *ChemCrow* 利用化学数据库和多种专家工具，自主规划和执行有机合成、药物发现和材料设计领域的实验。
- **对话式和心理健康支持 (Conversational and Mental Health Support):**
    - 正在探索将对话式智能体用于心理健康支持，帮助用户应对焦虑。然而，密切监控至关重要，以防止潜在的有害输出。
- **人类行为模拟 (Simulation of Human Behavior):**
    - *Generative Agents* 在虚拟环境中模拟人类互动，创建可信的人类行为模拟，并实现诸如信息传播和社交活动协调等涌现的社会行为。
- **概念验证演示 (Proof-of-Concept Demos):**
    - 诸如 AutoGPT 和 GPT-Engineer 之类的项目展示了将 LLM 用作通用问题解决和代码生成的自主智能体的潜力。虽然在可靠性和格式方面面临挑战，但它们是自主 AI 系统的鼓舞人心的例子。

# 挑战与局限性 (Challenges and Limitations)

虽然 LLM 智能体前景广阔，但要实现其广泛而可靠的部署，仍需要解决一些挑战和局限性：

- **有限的上下文长度 (Finite Context Length):** LLM 的有限上下文窗口限制了可以一次处理的信息量，包括历史数据和指令。这种限制影响了长期规划和在单个上下文中保持记忆。
- **长期规划和任务分解 (Long-Term Planning and Task Decomposition):** 有效地规划长期目标并在遇到意外错误时动态调整计划仍然是一个重大挑战。 LLM 可能难以在漫长而复杂的任务中保持连贯性并调整策略。
- **自然语言接口的可靠性 (Reliability of Natural Language Interfaces):** 依赖自然语言与外部工具进行交互会带来潜在的可靠性问题。 LLM 可能会在 API 调用中产生格式错误或表现出不可预测的行为，这需要强大的解析和错误处理机制。
- **效率和成本 (Efficiency and Cost):** 由于多次 API 调用和繁重的 LLM 推理，部署 LLM 智能体可能在计算上非常密集且成本高昂。优化效率和降低成本对于实际应用至关重要。

# 结论 (Conclusion)

LLM 智能体代表着自动化复杂任务的重大一步，它将大型语言模型的强大功能与复杂的规划、记忆和工具使用模块相结合。诸如 ReAct、Reflexion 和 HuggingGPT 之类的框架正在为更强大和更可靠的智能体铺平道路。随着研究的进展和这些挑战的解决，LLM 智能体有望改变从科学发现到日常自动化等各个领域，预示着人工智能智能体可以自主解决日益复杂的问题的未来。

# 参考文献 (References)

1. Weng, Lilian. "LLM-powered Autonomous Agents." *LilianWeng.github.io*, June 2023. [https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/)
2. Wei et al. "Chain of thought prompting elicits reasoning in large language models." *NeurIPS* 2022.
3. Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *arXiv preprint arXiv:2305.10601* (2023).
4. Liu et al. "Chain of Hindsight Aligns Language Models with Feedback." *arXiv preprint arXiv:2302.02676* (2023).
5. Liu et al. "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency." *arXiv preprint arXiv:2304.11477* (2023).
6. Yao et al. "ReAct: Synergizing reasoning and acting in language models." *ICLR 2023*.
7. Shinn & Labash. "Reflexion: an autonomous agent with dynamic memory and self-reflection." *arXiv preprint arXiv:2303.11366* (2023).
8. Laskin et al. "In-context Reinforcement Learning with Algorithm Distillation." *ICLR 2023*.
9. Karpas et al. "MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning." *arXiv preprint arXiv:2205.00445* (2022).
10. Parisi et al. "TALM: Tool Augmented Language Models."
11. Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." *arXiv preprint arXiv:2302.04761* (2023).
12. Li et al. "API-Bank: A Benchmark for Tool-Augmented LLMs." *arXiv preprint arXiv:2304.08244* (2023).
13. Shen et al. "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace." *arXiv preprint arXiv:2303.17580* (2023).
14. Bran et al. "ChemCrow: Augmenting large-language models with chemistry tools." *arXiv preprint arXiv:2304.05376* (2023).
15. Boiko et al. "Emergent autonomous scientific research capabilities of large language models." *arXiv preprint arXiv:2304.05332* (2023).
16. Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." *arXiv preprint arXiv:2304.03442* (2023).
17. AutoGPT, [https://github.com/Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)
18. GPT-Engineer, [https://github.com/AntonOsika/gpt-engineer](https://github.com/AntonOsika/gpt-engineer)