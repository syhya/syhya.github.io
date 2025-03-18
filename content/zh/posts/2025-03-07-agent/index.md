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


## 规划、记忆与工具使用


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

placeholder

### 算法蒸馏（Algorithm Distillation, AD）

placeholder

### Self-Ask

Self-Ask（Press et al. 2022）则通过不断提示模型提出后续问题来迭代构建思维过程。后续问题可以通过搜索引擎结果来回答。类似地，IRCoT（交替检索的 CoT；Trivedi et al. 2022）和 ReAct（推理 + 行动；Yao et al. 2023）将迭代式的 CoT 提示与对 Wikipedia API 的查询相结合，以搜索相关实体和内容，再将其添加回上下文。


## 记忆

### 人类记忆

**记忆** 指的是获取、储存、保持和提取信息的过程。人类的记忆主要分为以下三大类：

{{< figure
    src="category_human_memory.png"
    caption="Fig. xx. Categorization of human memory. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}

- **感觉记忆:** 用于在原始刺激（视觉、听觉、触觉等）消失后短暂保留感官信息，通常持续时间以毫秒或秒计。感觉记忆又分为：
    - 视觉记忆：对视觉通道所保留的瞬时图像或视觉印象，一般维持 0.25～0.5 秒，用于在视频或动画场景中形成视觉连续性。  
    - 听觉记忆：对听觉信息的短暂存储，可持续数秒，使人能回放刚刚听到的语句或声音片段。  
    - 触觉记忆：用于保留短暂的触觉或力觉信息，一般持续毫秒到秒级，例如敲击键盘或盲文阅读时短暂的手指感知。 

- **短期记忆:** 存储我们当前意识到的信息。
    - 持续时间约 20～30 秒，容量通常为 7±2 个项目。
    - 承担学习、推理等复杂认知任务时对信息的临时处理和维持。


- **长期记忆:** 能将信息保存数天到数十年，容量几乎无限。长期记忆分为：
  - 外显记忆: 可有意识回忆，包含情景记忆（个人经历、事件细节）和语义记忆（事实与概念）。
  - 内隐记忆:无意识记忆，主要与技能和习惯相关，如骑车或盲打。


人类记忆的这三种类型相互交织，共同构成了我们对世界的认知和理解。在构建 LLM Agent 中，我们也可以借鉴人类记忆的这种分类方式：

- **感觉记忆** 对应 LLM 对输入原始数据（如文本、图片和视频）等的嵌入表征。  
- **短期记忆** 对应 LLM 的上下文内学习，受限于模型上下文窗口 `max_tokens`，当对话长度超出窗口后，早期信息将被截断。  
- **长期记忆** 对应 外部向量存储或数据库，Agent 可以基于 **RAG** 技术在需要时检索历史信息。

### LLM Agent 记忆

Agent 与用户多轮互动、执行多步任务时，可以利用不同形式的记忆以及环境信息来完成工作流。

{{< figure
    src="llm_memory_overview.png"
    caption="Fig. xx. An overview of the sources, forms, and operations of the memory in LLM-based agents. (Image source: [Zhang et al. 2024](https://arxiv.org/abs/2404.13501))"
    align="center"
    width="100%"
>}}

- **文本记忆**  
  - 完整交互：记录了所有对话与操作轨迹，帮助 Agent 回溯上下文。  
  - 近期交互：只保留与当前任务高度相关的对话内容，减少不必要的上下文占用。  
  - 检索到的交互：Agent 可从外部知识库中检索到与当前任务相关的历史对话或记录，融入当前上下文中。  
  - 外部知识：当 Agent 遇到知识空白时，可通过 API 或外部存储进行检索和获取额外信息。

- **参数化记忆**  
  - 微调：通过给 LLM 注入新的信息或知识，从而扩充模型的内部知识。  
  - 知识编辑：在模型层面对已有知识进行修改或更新，实现对模型内部参数记忆的动态调整。

- **环境**  
  - 代表着 Agent 与用户及外部系统交互时涉及的实体和上下文，比如用户 Alice、可能访问的工具或界面（例如订票系统、流媒体平台等）。

- **Agent**  
  - LLM Agent 负责读与写操作，即读取外部环境或知识库的信息，写入新的动作或内容。  
  - 同时包含一系列管理功能，如合并、反思、遗忘等，用以动态维护短期和长期记忆。


另外一个例子是 Agent 在完成在两个不同但相关的任务中，需要同时使用短期记忆和长期记忆：

- **任务 A 播放视频**：Agent 将当前的计划、操作和环境状态（例如搜索、点击、播放视频等）记录在短期记忆中，这部分信息会保存在内存和 LLM 的上下文窗口中。
- **任务 B 下载游戏**：Agent 利用长期记忆中与 Arcane 和 League of Legend 相关的知识，快速找到游戏下载方案。图中显示，Agent 在 Google 上进行搜索，我们可以将 Google 的知识库视作一个外部知识源，同时所有新的搜索、点击和下载操作也会被更新到短期记忆中。

{{< figure
    src="gui_agent_memory_illustration.png"
    caption="Fig. xx: Illustration of short-term memory and long-term memory in an LLM-brained GUI agent. (Image source: [Zhang et al. 2024](https://arxiv.org/abs/2411.18279))"
    align="center"
    width="100%"
>}}

常见记忆元素及其对应的存储方式可以总结成以下表格: 

| **记忆元素**   | **记忆类型** | **描述**                                              | **存储介质 / 方式**          |
|---------------|------------|-------------------------------------------------------|--------------------------|
| 动作          | 短期记忆   | 历史动作轨迹（例如点击按钮、输入文本等）               |     内存、LLM 上下文窗口       |
| 计划          | 短期记忆   | 上一步或当前生成的下一步操作计划                         |   内存、LLM 上下文窗口       |
| 执行结果      | 短期记忆   | 动作执行后返回的结果、报错信息及环境反馈                  |  内存、LLM 上下文窗口       |
| 环境状态      | 短期记忆   | 当前 UI 环境中可用的按钮、页面标题、系统状态等              |    内存、LLM 上下文窗口       |
| 自身经验      | 长期记忆   | 历史任务轨迹与执行步骤                                  |    数据库、磁盘              |
| 自我引导      | 长期记忆   | 从历史成功轨迹中总结出的指导规则与最佳实践                |  数据库、磁盘              |
| 外部知识      | 长期记忆   | 辅助任务完成的外部知识库、文档或其他数据源                |  外部数据库、向量检索       |
| 任务成功指标  | 长期记忆   | 记录任务成功率、失败率等指标，便于改进和分析              |  数据库、磁盘              |


此外，研究者还提出一些新的训练和存储方法，以增强 LLM 的记忆能力：

**LongMem(Language Models Augmented with Long-Term Memory)**([Wang, et al. 2023](https://arxiv.org/abs/2306.07174))使 LLM 能够记忆长历史信息。其采用一种解耦的网络结构，将原始 LLM 参数冻结为记忆编码器(memory encoder)固定下来，同时使用自适应残差网络(Adaptive Residual Side-Network，SideNet)作为记忆检索器进行记忆检查和读取。

{{< figure
    src="LongMem.png"
    caption="Fig. xx. Overview of the memory caching and retrieval flow of LongMem. (Image source: [Wang, et al. 2023](https://arxiv.org/abs/2306.07174))"
    align="center"
    width="100%"
>}}
 
其主要由三部分构成：**Frozen LLM**、**Residual SideNet** 和 **Cached Memory Bank**。其工作流程如下：

- 先将长文本序列拆分成固定长度的片段，每个片段在 Frozen LLM 中逐层编码后，在第 $m$ 层提取注意力的 $K, V \in \mathbb{R}^{H \times M \times d}$ 向量对并缓存到 Cached Memory Bank。
- 面对新的输入序列时，模型根据当前输入的 query-key 检索长期记忆库，从中获取与输入最相关的前 $k$ 个 key-value（即 top-$k$ 检索结果），并将其融合到后续的语言生成过程中；与此同时记忆库会移除最旧的内容以保证最新上下文信息的可用性。
- Residual SideNet 则在推理阶段对冻结 LLM 的隐藏层输出与检索得到的历史 key-value 进行融合，完成对超长文本的有效建模和上下文利用。

通过这种解耦设计，LongMem 无需扩大自身的原生上下文窗口就能灵活调度海量历史信息，兼顾了速度与长期记忆能力。


## 工具使用

工具使用是 LLM Agent 重要组成部分, 通过赋予 LLM 调用外部工具的能力，其功能得到了显著扩展：不仅能够生成自然语言，还能获取实时信息、执行复杂计算以及与各类系统（如数据库、API 等）交互，从而有效突破预训练知识的局限，避免重复造轮子的低效过程。

传统 LLM 主要依赖预训练数据进行文本生成，但这也使得它们在数学运算、数据检索和实时信息更新等方面存在不足。通过工具调用，模型可以：
  
- **提升运算能力：** 例如通过调用专门的计算器工具 [Wolfram](https://gpt.wolfram.com/index.php.en)，模型能够进行更精准的数学计算，弥补自身算术能力的不足。

- **实时获取信息：** 利用搜索引擎 Gooole、Bing 或数据库 API，模型可以访问最新信息，确保生成内容的时效性和准确性。
- **增强信息可信度：** 借助外部工具的支持，模型能够引用真实数据来源，降低信息虚构的风险，提高整体可信性。
- **提高系统透明度：** 跟踪 API 调用记录可以帮助用户理解模型决策过程，提供一定程度的可解释性。


当前，业界涌现出多种基于工具调用的 LLM 应用，它们利用不同策略和架构，实现了从简单任务到复杂多步推理的全面覆盖。以下是几个典型案例：

- **MRKL (模块化推理、知识和语言) 系统 (Modular Reasoning, Knowledge and Language Systems):** 将 LLM 与专家模块相结合，专家模块可以是神经模块或符号模块(例如，计算器、API)。 LLM 充当路由器，将查询定向到最适合特定子任务的专家模块。
- **Toolformer 和函数调用 (Function Calling):** 这些方法微调 LLM 以自动使用外部 API。 Toolformer 学习预测何时以及如何使用工具，方法是评估 API 调用在提高模型输出质量方面的效用。函数调用，如 OpenAI API 中所示，允许开发人员定义工具 API 并将其提供给 LLM，使其能够根据需要调用这些函数。
- **HuggingGPT:** 一个新兴框架，它使用 ChatGPT 作为任务规划器来协调 Hugging Face 平台上可用的各种 AI 模型。 HuggingGPT 将用户请求分解为任务，根据模型描述从 Hugging Face 中选择合适的模型，使用这些模型执行任务，并为用户总结结果。

通过集成这些工具，LLM 智能体可以执行复杂的任务，例如生成代码、访问实时信息、与数据库交互以及控制外部系统，从而极大地扩展其解决问题的能力。


## LLM 智能体应用

### WebVoyager

WebVoyager([He et al., 2024](https://arxiv.org/abs/2401.13919)) 是一种基于多模态大模型的自主网页交互智能体，能够控制鼠标和键盘进行网页浏览。WebVoyager 采用经典的 ReAct 循环。在每个交互步骤中，它查看带有类似 SoM(Set-of-Marks)([Yang, et al., 2023](https://arxiv.org/abs/2310.11441)) 方法标注的浏览器截图即通过在网页元素上放置数字标签提供交互提示，然后决定下一步行动。这种视觉标注与 ReAct 循环相结合，使得用户可以通过自然语言与网页进行交互。具体可以参考使用 LangGraph 框架的    [WebVoyager 代码](https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/)。


{{< figure
    src="WebVoyager.png"
    caption="Fig. xx. The overall workflow of WebVoyager. (Image source: [He et al., 2024](https://arxiv.org/abs/2401.13919))"
    align="center"
    width="100%"
>}}

### OpenAI Operator

**Operator** ([OpenAI, 2025](https://openai.com/index/introducing-operator/)) 是一个 OpenAI 近期发布的 AI 智能体，旨在自主执行网络任务。Operator 能够像人类用户一样与网页互动，通过打字、点击和滚动等操作完成指定任务。Operator 的核心技术是计算机使用智能体（Computer-Using Agent, CUA）([OpenAI, 2025](https://openai.com/index/computer-using-agent/))。CUA 结合了 GPT-4o 的视觉能力和通过强化学习获得更强的推理能力，经过专门训练后能够与图形用户界面（GUI）进行交互，包括用户在屏幕上看到的按钮、菜单和文本框。

{{< figure
    src="cua_overview.png"
    caption="Fig. xx. Overview of OpenAI CUA. (Image source: [OpenAI, 2025](https://openai.com/index/computer-using-agent/))"
    align="center"
    width="100%"
>}}

CUA 的运作方式遵循一个迭代循环，包含三个阶段：

- **感知 (Perception):**  
  CUA 通过捕获浏览器截图来“观察”网页内容。这种基于视觉的输入方式使其能够理解页面的布局和元素。

- **推理 (Reasoning):**  
  借助链式思考的推理过程，CUA 会评估下一步行动，其依据是当前和之前的截图以及已执行的操作。这种推理能力使其能够跟踪任务进度、回顾中间步骤，并根据需要进行调整。

- **行动 (Action):**  
  CUA 通过模拟鼠标和键盘操作（如点击、输入和滚动）与浏览器进行交互。这使其能够在无需特定 API 集成的情况下执行各种网络任务。

CUA 和之前现有的 WebVoyager 不同之处在于这是一个专门经过强化学习训练的 Agent，而不是直接调用 GPT-4o 搭建的固定流程的 Workflow。虽然 CUA 目前仍处于早期阶段且存在一定局限，但它以下基准测试中取得了 SOTA 结果。

{{< figure
    src="cua_benchmark.png"
    caption="Fig. xx. OpenAI CUA Benchmark Results. (Image source: [OpenAI, 2025](https://openai.com/index/computer-using-agent/))"
    align="center"
    width="100%"
>}}


## 挑战与局限性

虽然 LLM 智能体前景广阔，但要实现其广泛而可靠的部署，仍需要解决一些挑战和局限性：

- **有限的上下文长度:** LLM 的有限上下文窗口限制了可以一次处理的信息量，包括历史数据和指令。这种限制影响了长期规划和在单个上下文中保持记忆。
- **长期规划和任务分解 :** 有效地规划长期目标并在遇到意外错误时动态调整计划仍然是一个重大挑战。 LLM 可能难以在漫长而复杂的任务中保持连贯性并调整策略。
- **自然语言接口的可靠性:** 依赖自然语言与外部工具进行交互会带来潜在的可靠性问题。 LLM 可能会在 API 调用中产生格式错误或表现出不可预测的行为，这需要强大的解析和错误处理机制。
- **效率和成本:** 由于多次 API 调用和繁重的 LLM 推理，部署 LLM 智能体可能在计算上非常密集且成本高昂。优化效率和降低成本对于实际应用至关重要。


## 参考文献

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