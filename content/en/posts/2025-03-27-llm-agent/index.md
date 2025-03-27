---
title: "Large Language Model Agents"
date: "2025-03-27T10:00:00+00:00"
author: "Yue Shui"
categories: ["Technical Blog"]
tags: ["LLM", "AI", "Agent", "Reinforcement Learning", "Planning", "Memory", "Tool Use", "Deep Research", "ReAct", "Reflexion", "WebVoyager", "OpenAI Operator", "CoT", "ToT", "workflow"]
readingTime: 30
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

## Agents

Since OpenAI released ChatGPT in October 2022, and with the subsequent emergence of projects such as [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) and [AgentGPT](https://github.com/reworkd/AgentGPT), LLM-related agents have gradually become a research hotspot and a promising direction for practical applications in AI in recent years. This article will introduce the basic concepts of agents, their core technologies, and the latest advances in their applications.

### Large Language Model Agents

**Large Language Model Agents (LLM agents)** utilize LLMs as the system's brain, combined with modules such as planning, memory, and external tools, to achieve automated execution of complex tasks.

- **User Request:** Users interact with the agent by inputting tasks through prompts.
- **Agent:** The system's brain, consisting of one or more LLMs, responsible for overall coordination and task execution.
- **Planning:** Decomposes complex tasks into smaller subtasks and formulates execution plans, while continuously optimizing results through reflection.
- **Memory:** Includes short-term memory (using in-context learning to instantly capture task information) and long-term memory (using external vector storage to save and retrieve key information, ensuring information continuity for long-term tasks).
- **Tools:** Integrates external tools such as calculators, web searches, and code interpreters to call external data, execute code, and obtain the latest information.

{{< figure
    src="llm_agent.png"
    caption="Fig. 1. The illustration of LLM Agent Framework. (Image source: [DAIR.AI, 2024](https://www.promptingguide.ai/research/llm-agents#llm-agent-framework))"
    align="center"
    width="70%"
>}}

### Reinforcement Learning Agents

The goal of **Reinforcement Learning (RL)** is to train an agent to take a series of actions (actions, $a_t$) in a given environment. During the interaction, the agent transitions from one state (state, $s_t$) to the next, and receives a reward (reward, $r_t$) from the environment after each action. This interaction generates a complete trajectory (trajectory, $\tau$), usually represented as:

$$
\tau = \{(s_0, a_0, r_0), (s_1, a_1, r_1), \dots, (s_T, a_T, r_T)\}.
$$

The agent's objective is to learn a policy (policy, $\pi$), which is a rule for selecting actions in each state, to **maximize the expected cumulative reward**, often expressed as:

$$
\max_{\pi} \, \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right],
$$
where $\gamma \in [0,1]$ is the discount factor, used to balance short-term and long-term rewards.

{{< figure
    src="rl_agent.png"
    caption="Fig. 2. The agent-environment interaction. (Image source: [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book.html))"
    align="center"
    width="80%"
>}}

In the **LLM** context, the model can be viewed as an agent, and the "environment" can be understood as the user input and its corresponding expected response:

- **State ($s_t$)**: Can be the current dialogue context or the user's question.
- **Action ($a_t$)**: The text output by the model (answers, generated content, etc.).
- **Reward ($r_t$)**: Feedback from the user or system (such as user satisfaction, automatic scoring by a reward model, etc.).
- **Trajectory ($\tau$)**: The sequence of all text interactions from the initial dialogue to the end, which can be used to evaluate the overall performance of the model.
- **Policy ($\pi$)**: The rules that govern how the LLM generates text in each state (dialogue context), generally determined by the model's parameters.

For LLMs, traditionally, pre-training is first performed on a massive amount of offline data.  In the subsequent reinforcement learning stage, the model is trained through human or model feedback to produce high-quality text that better aligns with human preferences or task requirements.

### Comparison

The following table shows the differences between the two:

| **Comparison Dimension** | **LLM Agent**                                                                 | **RL Agent**                                                                     |
|--------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Core Principle**       | Automates complex tasks through planning, memory, and tool utilization.        | Continuously optimizes the policy to maximize long-term rewards through a trial-and-error feedback loop of interaction with the environment. |
| **Optimization Method**  | **Does not directly update model parameters**, primarily relies on context expansion, external memory, and tools to improve performance. | **Continuously and frequently updates policy model parameters**, relying on reward signals from environmental feedback for optimization. |
| **Interaction Method**   | Uses natural language to interact with users or external systems, flexibly calling various tools to obtain external information. | Interacts with real or simulated environments, where the environment provides rewards or punishments, forming a closed-loop feedback. |
| **Implementation Goal**  | Decomposes complex tasks and utilizes external resources to complete tasks, focusing on the quality and accuracy of task results. | Maximizes long-term rewards, pursuing the optimal balance between short-term and long-term returns. |

As research deepens, the combination of LLM and RL agents presents more possibilities, such as:
- Using reinforcement learning methods to train Reasoning LLMs (e.g., o1/o3), making them more suitable as base models for LLM agents.
- Simultaneously, recording the data and feedback of LLM agents executing tasks to provide rich training data for Reasoning LLMs, thereby improving model performance.


## Planning: Task Decomposition

The core components of an LLM Agent include **planning**, **memory**, and **tool use**. These components work together to enable the agent to autonomously execute complex tasks.

{{< figure
    src="llm_agent_overview.png"
    caption="Fig. 3. Overview of a LLM-powered autonomous agent system. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}

Planning is crucial for the successful execution of complex tasks. It can be approached in different ways depending on the complexity and the need for iterative improvement. In simple scenarios, the planning module can use the LLM to pre-outline a detailed plan, including all necessary subtasks. This step ensures that the agent systematically performs **task decomposition** and follows a clear logical flow from the outset.

### Chain of Thought

**Chain of Thought (CoT)** ([Wei et al. 2022](https://arxiv.org/abs/2201.11903)) generates a series of short sentences describing the reasoning process, called reasoning steps. The purpose is to explicitly show the model's reasoning path, helping the model better handle **complex reasoning tasks**. The figure below shows the difference between few-shot prompting (left) and chain-of-thought prompting (right). Few-shot prompting gets the wrong answer, while the chain-of-thought method guides the model to state the reasoning process step by step, more clearly reflecting the model's logical process, thereby improving the accuracy and interpretability of the answer.

{{< figure
    src="cot.png"
    caption="Fig. 4. The comparison example of few-shot prompting and CoT prompting. (Image source: [Weng, 2023](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}


**Zero-Shot CoT** ([Kojima et al. 2022](https://arxiv.org/abs/2205.11916)) is a follow-up study of CoT that proposes an extremely simple zero-shot prompting method. They found that by simply adding the sentence `Let's think step by step` at the end of the question, the LLM can generate a chain of thought and obtain a more accurate answer.

{{< figure
    src="zero_shot_cot.png"
    caption="Fig. 5. The comparison example of few-shot prompting and CoT prompting. (Image source: [Kojima et al. 2022](https://arxiv.org/abs/2205.11916))"
    align="center"
    width="100%"
>}}

### Self-Consistency Sampling

**Self-consistency sampling** ([Wang et al. 2022a](https://arxiv.org/abs/2203.11171)) is a method that generates **multiple diverse answers** by sampling the same prompt multiple times with `temperature > 0` and selecting the best answer from them. Its core idea is to improve the accuracy and robustness of the final answer by sampling multiple reasoning paths and then using majority voting. The criteria for selecting the best answer may vary for different tasks. Generally, **majority voting** is used as a general solution. For tasks such as programming problems that are easy to verify, the answers can be verified by running an interpreter and combining unit tests. This is an optimization of CoT, and when used in conjunction with it, it can significantly improve the model's performance in complex reasoning tasks.

{{< figure
    src="self_consistency.png"
    caption="Fig. 6. Overview of the Self-Consistency Method for Chain-of-Thought Reasoning. (Image source: [Wang et al. 2022a](https://arxiv.org/abs/2203.11171))"
    align="center"
    width="100%"
>}}

Here are some subsequent optimization efforts:

- ([Wang et al. 2022b](https://arxiv.org/abs/2207.00747)) subsequently used another ensemble learning method for optimization, increasing randomness by changing the order of examples or replacing human-written reasoning with model-generated reasoning, and then using majority voting.

{{< figure
    src="rationale_augmented.png"
    caption="Fig. 7. An overview of different ways of composing rationale-augmented ensembles, depending on how the randomness of rationales is introduced. (Image source: [Wang et al. 2022b](https://arxiv.org/abs/2207.00747))"
    align="center"
    width="100%"
>}}

- If the training samples only provide the correct answer without reasoning, **STaR (Self-Taught Reasoner)**([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)) can be used:
(1) Let the LLM generate reasoning chains, and only keep the reasoning with the correct answer.
(2) Fine-tune the model with the generated reasoning, and iterate repeatedly until convergence. Note that when `temperature` is high, it is easy to generate results with the correct answer but incorrect reasoning. If there is no standard answer, consider using majority voting as the "correct answer".

{{< figure
    src="STaR.png"
    caption="Fig. 8. An overview of STaR and a STaR-generated rationale on CommonsenseQA (Image source: [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465))"
    align="center"
    width="100%"
>}}

- ([Fu et al. 2023](https://arxiv.org/abs/2210.00720)) found that more complex examples (more reasoning steps) can improve model performance. When separating reasoning steps, the newline character `\n` works better than `step i`, `.`, or `;`. In addition, the complexity-based consistency strategy, which only performs majority voting on the top $k$ reasoning chains generated by complexity, can further optimize the model output.  Replacing `Q:` with `Question:` in the prompt has also been shown to have an additional performance boost.

{{< figure
    src="linebreak.png"
    caption="Fig. 9. Sensitivity analysis on step formatting. Complex prompts consistently lead to better performance with regard to different step formatting. (Image source: [Fu et al. 2023)](https://arxiv.org/abs/2210.00720))"
    align="center"
    width="100%"
>}}

### Tree of Thoughts

**Tree of Thoughts (ToT)** ([Yao et al. 2023](https://arxiv.org/abs/2305.10601)) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thinking steps and generates multiple different ideas at each step, forming a tree structure. The search process can use breadth-first search (BFS) or depth-first search (DFS), and each state is evaluated by a classifier (or the LLM can be used for scoring) or majority voting. It consists of three main steps:

- **Expand**: Generate one or more candidate solutions.
- **Score**: Measure the quality of the candidate solutions.
- **Prune**: Keep the top $k$ best candidate solutions.

If no solution is found (or the quality of the candidate solutions is not high enough), backtrack to the expansion step.

{{< figure
    src="tot.png"
    caption="Fig. 10. Schematic illustrating various approaches to problem solving with LLMs (Image source: [Yao et al. 2023](https://arxiv.org/abs/2305.10601))"
    align="center"
    width="100%"
>}}

## Planning: Self-Reflection

**Self-Reflection** is a key factor that enables agents to achieve iterative improvement by improving past action decisions and correcting previous errors. It plays a crucial role in real-world tasks where trial and error is inevitable.

### ReAct

The **ReAct (Reason + Act)** ([Yao et al. 2023](https://arxiv.org/abs/2210.03629)) framework achieves seamless integration of reasoning and action in LLMs by combining task-specific discrete actions and language space. This design not only enables the model to interact with the environment by calling external interfaces such as the Wikipedia search API, but also generates detailed reasoning trajectories in natural language, thereby solving complex problems.

The ReAct prompt template contains explicit thinking steps, and its basic format is as follows:

```plaintext
Thought：...
Action：...
Observation：...
...(Repeated many times)
```

{{< figure
    src="ReAct.png"
    caption="Fig. 11. Examples of reasoning trajectories for knowledge-intensive tasks (e.g. HotpotQA, FEVER) and decision-making tasks (e.g. AlfWorld Env, WebShop). (Image source: [Yao et al. 2023](https://arxiv.org/abs/2210.03629))"
    align="center"
    width="100%"
>}}

As can be seen from the figure below, in both knowledge-intensive tasks and decision-making tasks, ReAct's performance is significantly better than the basic method that relies only on `Actor`, thus demonstrating its advantages in improving reasoning effectiveness and interaction performance.

{{< figure
    src="ReAct_res.png"
    caption="Fig. 12. PaLM-540B prompting results on HotpotQA and Fever. (Image source: [Yao et al. 2023](https://arxiv.org/abs/2210.03629))"
    align="center"
    width="50%"
>}}

### Reflexion

**Reflexion** ([Shinn et al. 2023](https://arxiv.org/abs/2303.11366)) enables LLMs to iteratively optimize decisions through self-feedback and dynamic memory.

This method essentially draws on the idea of reinforcement learning. In the traditional Actor-Critic model, the Actor selects action $a_t$ based on the current state $s_t$, while the Critic gives an estimate (such as the value function $V(s_t)$ or the action-value function $Q(s_t,a_t)$) and feeds it back to the Actor for policy optimization. Correspondingly, in the three major components of Reflexion:

- **Actor**: Played by the LLM, it outputs text and corresponding actions based on the environment state (including context and historical information). It can be denoted as:

  $$
  a_t = \pi_\theta(s_t),
  $$

  where $\pi_\theta$ represents the policy obtained based on the parameter $\theta$ (i.e., the weights or prompts of the LLM). The Actor interacts with the environment and generates a trajectory $\tau = \{(s_1,a_1,r_1), \dots, (s_T,a_T,r_T)\}$.

- **Evaluator**: Similar to the Critic, the Evaluator receives the trajectory generated by the Actor and outputs a reward signal $r_t$. In the Reflexion framework, the Evaluator can analyze the trajectory through pre-designed heuristic rules or an additional LLM, and then generate rewards. For example:

  $$
  r_t = R(\tau_t),
  $$

  where $R(\cdot)$ is a reward function based on the current trajectory $\tau_t$.

- **Self-Reflection**: This module is equivalent to adding an additional self-regulation feedback mechanism outside the Actor-Critic. It integrates the current trajectory $\tau$, reward signals $\{r_t\}$, and historical experience in long-term memory, and uses language generation capabilities to generate self-improvement suggestions for the next decision. This feedback information is then written to external memory, providing richer context for subsequent Actor decisions, thereby achieving iterative optimization similar to policy parameter $\theta$ through dynamic adjustment of prompts without updating the internal parameters of the LLM.

{{< figure
    src="Reflexion.png"
    caption="Fig. 13. (a) Diagram of Reflexion. (b) Reflexion reinforcement algorithm (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

The core loop and algorithm description of Reflexion are as follows:

- **Initialization**
   - Instantiate three models (all can be implemented by LLM): Actor, Evaluator, and Self-Reflection, denoted as $M_a, M_e, M_{sr}$ respectively.
   - Initialize the policy $\pi_\theta$ (including the model parameters or prompts of the Actor, and initial memory, etc.).
   - Let the Actor generate an initial trajectory $\tau_0$ according to the current policy $\pi_\theta$. After $M_e$ evaluates it, $M_{sr}$ generates the first self-reflection text and stores it in long-term memory.

- **Generate Trajectory**
   - In each iteration, $M_a$ reads the current long-term memory and environmental observations, sequentially outputs actions $\{a_1, a_2, \ldots\}$, interacts with the environment and obtains corresponding feedback, forming a new trajectory $\tau_t$. $\tau_t$ can be regarded as the short-term memory of this task, and is only used in this iteration.

- **Evaluation**
   - $M_e$ outputs rewards or scores $\{r_1, r_2, \ldots\}$ based on the trajectory $\tau_t$ (i.e., the sequence of Actor's actions and environmental feedback). This step corresponds to the internal feedback of $M_e$, or the results are directly given by the external environment.

- **Self-Reflection**
   - The $M_{sr}$ module integrates the trajectory $\tau_t$ and the reward signal $\{r_t\}$ to generate self-correction or improvement suggestions $\mathrm{sr}_t$ at the language level.
   - Reflective text can be regarded as an analysis of errors or provide new启发思路, and is stored in long-term memory. In practice, we can vectorize the feedback information and store it in a vector database.

- **Update and Repeat**
   - After appending the latest self-reflection text $\mathrm{sr}_t$ to the long-term memory, the Actor can use RAG to retrieve historically relevant information from it in the next iteration to adjust the policy.
   - Repeat the above steps until $M_e$ determines that the task is achieved or the maximum number of rounds is reached. In this loop, Reflexion relies on the continuous accumulation of **self-reflection + long-term memory** to improve decisions, rather than directly modifying model parameters.

The following shows examples of Reflexion's application in decision-making, programming, and reasoning tasks:

{{< figure
    src="reflextion_examples.png"
    caption="Fig. 14. Reflexion works on decision-making 4.1, programming 4.3, and reasoning 4.2 tasks (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

In an experiment on 100 HotPotQA questions, by comparing the CoT method and the method of adding episodic memory, the results show that after adding the self-reflection step at the end using the Reflexion method, its search, information retrieval, and reasoning capabilities are significantly improved.

{{< figure
    src="reflextion_result.png"
    caption="Fig. 15. Comparative Analysis of Chain-of-Thought (CoT) and ReAct on the HotPotQA Benchmark (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

### DeepSeek R1

**DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)) represents a major breakthrough in the open-source community's replication of OpenAI o1 ([OpenAI, 2024](https://openai.com/o1/)), successfully training an advanced reasoning model with deep reflection capabilities through reinforcement learning techniques.

> For a detailed description of the training process and technical implementation of DeepSeek R1, please refer to my previous blog post: [OpenAI o1 Replication Progress: DeepSeek-R1](https://syhya.github.io/posts/2025-01-27-deepseek-r1/).

The key transformation in the training process of DeepSeek-R1-Zero - as training progresses, the model gradually **emerges** with excellent **self-evolution** capabilities. This capability is reflected in three core aspects:

- **Self-Reflection**: The model can trace back and critically evaluate previous reasoning steps.
- **Active Exploration**: When it finds that the current solution path is not ideal, it can autonomously find and try alternative solutions.
- **Dynamic Thinking Adjustment**: Adaptively adjust the number of generated tokens according to the complexity of the problem to achieve a deeper thinking process.

This dynamic and spontaneous reasoning behavior significantly improves the model's ability to solve complex problems, enabling it to respond to challenging tasks more efficiently and accurately.

{{< figure
    src="deepseek_r1_zero_response_time.png"
    caption="Fig. 16. The average response length of DeepSeek-R1-Zero on the training set during the RL process. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

DeepSeek-R1-Zero also exhibited a typical "aha moment" during training. At this critical stage, the model suddenly realized that there was an error in the previous thinking path during the reasoning process, and then quickly adjusted the thinking direction, and finally successfully led to the correct answer. This phenomenon strongly proves that the model has developed strong **self-correction** and **reflection capabilities** during the reasoning process, similar to the aha experience in human thinking.

{{< figure
    src="aha_moment.png"
    caption="Fig. 17. An interesting “aha moment” of an intermediate version of DeepSeek-R1-Zero. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

## Memory

### Human Memory

**Memory** refers to the process of acquiring, storing, retaining, and retrieving information. Human memory is mainly divided into the following three categories:

{{< figure
    src="category_human_memory.png"
    caption="Fig. 18. Categorization of human memory. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}

- **Sensory Memory:** Used to briefly retain sensory information after the original stimulus (visual, auditory, tactile, etc.) disappears, usually lasting for milliseconds or seconds. Sensory memory is divided into:
    - Visual Memory: The instantaneous image or visual impression retained by the visual channel, generally lasting 0.25-0.5 seconds, is used to form visual continuity in video or animation scenes.
    - Auditory Memory: The short-term storage of auditory information, which can last for several seconds, enables people to replay the sentences or sound clips they just heard.
    - Tactile Memory: Used to retain short-term tactile or force information, generally lasting from milliseconds to seconds, such as the short-term finger perception when tapping the keyboard or reading Braille.

- **Short-Term Memory:** Stores the information we are currently aware of.
    - Lasts about 20-30 seconds, and the capacity is usually 7±2 items.
    - Undertakes the temporary processing and maintenance of information during complex cognitive tasks such as learning and reasoning.

- **Long-Term Memory:** Can store information for days to decades, and the capacity is almost unlimited. Long-term memory is divided into:
  - Explicit Memory: Can be consciously recalled, including episodic memory (personal experiences, event details) and semantic memory (facts and concepts).
  - Implicit Memory: Unconscious memory, mainly related to skills and habits, such as riding a bicycle or touch typing.

These three types of human memory are intertwined and together constitute our cognition and understanding of the world. In building LLM Agents, we can also learn from this classification of human memory:

- **Sensory Memory** corresponds to the embedding representation of the LLM's input raw data (such as text, pictures, and videos).
- **Short-Term Memory** corresponds to the LLM's in-context learning, which is limited by the model's context window `max_tokens`. When the dialogue length exceeds the window, early information will be truncated.
- **Long-Term Memory** corresponds to external vector storage or databases. Agents can retrieve historical information when needed based on **RAG** technology.

### LLM Agent Memory

When an agent interacts with users in multiple rounds and performs multi-step tasks, it can utilize different forms of memory and environmental information to complete the workflow.

{{< figure
    src="llm_memory_overview.png"
    caption="Fig. 19. An overview of the sources, forms, and operations of the memory in LLM-based agents. (Image source: [Zhang et al. 2024](https://arxiv.org/abs/2404.13501))"
    align="center"
    width="100%"
>}}

- **Text Memory**
  - Complete Interaction: Records all dialogue and operation trajectories, helping the Agent trace back the context.
  - Recent Interaction: Only retains dialogue content that is highly relevant to the current task, reducing unnecessary context occupation.
  - Retrieved Interaction: The Agent can retrieve historical dialogues or records related to the current task from an external knowledge base and integrate them into the current context.
  - External Knowledge: When the Agent encounters a knowledge gap, it can retrieve and obtain additional information through APIs or external storage.

- **Parameterized Memory**
  - Fine-tuning: Infuses new information or knowledge into the LLM, thereby expanding the model's internal knowledge.
  - Knowledge Editing: Modifies or updates existing knowledge at the model level, realizing dynamic adjustment of the memory of internal parameters of the model.

- **Environment**
  - Represents the entities and context involved when the Agent interacts with users and external systems, such as user Alice, tools or interfaces that may be accessed (such as ticketing systems, streaming platforms, etc.).

- **Agent**
  - The LLM Agent is responsible for read and write operations, that is, reading information from the external environment or knowledge base, and writing new actions or content.
  - It also includes a series of management functions, such as merging, reflection, forgetting, etc., to dynamically maintain short-term and long-term memory.

Another example is when an Agent needs to use both short-term and long-term memory to complete two different but related tasks:

- **Task A Play Video**: The Agent records the current plan, operations, and environment state (such as search, click, play video, etc.) in short-term memory, which is stored in memory and the LLM's context window.
- **Task B Download Game**: The Agent utilizes the knowledge related to Arcane and League of Legend in long-term memory to quickly find a game download solution. The figure shows that the Agent searches on Google. We can regard Google's knowledge base as an external knowledge source, and all new search, click, and download operations will also be updated to short-term memory.

{{< figure
    src="gui_agent_memory_illustration.png"
    caption="Fig. 20: Illustration of short-term memory and long-term memory in an LLM-brained GUI agent. (Image source: [Zhang et al. 2024](https://arxiv.org/abs/2411.18279))"
    align="center"
    width="100%"
>}}

Common memory elements and their corresponding storage methods can be summarized in the following table:

| **Memory Element** | **Memory Type** | **Description**                                              | **Storage Medium / Method**          |
|--------------------|-----------------|-------------------------------------------------------|--------------------------|
| Actions            | Short-Term Memory | Historical action trajectories (e.g., clicking buttons, entering text) | Memory, LLM context window       |
| Plan               | Short-Term Memory | The previous step or the next step of the currently generated operation plan | Memory, LLM context window       |
| Execution Results  | Short-Term Memory | Results returned after action execution, error messages, and environmental feedback | Memory, LLM context window       |
| Environment State  | Short-Term Memory | Available buttons, page titles, system status, etc. in the current UI environment | Memory, LLM context window       |
| Own Experience     | Long-Term Memory  | Historical task trajectories and execution steps                                  | Database, disk              |
| Self-Guidance      | Long-Term Memory  | Guidance rules and best practices summarized from historical successful trajectories | Database, disk              |
| External Knowledge | Long-Term Memory  | External knowledge bases, documents, or other data sources that assist in task completion | External database, vector retrieval |
| Task Success Metrics| Long-Term Memory  | Records task success rate, failure rate, and other indicators for improvement and analysis | Database, disk              |

In addition, researchers have proposed some new training and storage methods to enhance the memory capabilities of LLMs:

**LongMem (Language Models Augmented with Long-Term Memory)** ([Wang, et al. 2023](https://arxiv.org/abs/2306.07174)) enables LLMs to memorize long historical information. It adopts a decoupled network structure, freezing the original LLM parameters as a memory encoder, while using an Adaptive Residual Side-Network (SideNet) as a memory retriever for memory checking and reading.

{{< figure
    src="LongMem.png"
    caption="Fig. 21. Overview of the memory caching and retrieval flow of LongMem. (Image source: [Wang, et al. 2023](https://arxiv.org/abs/2306.07174))"
    align="center"
    width="100%"
>}}

It is mainly composed of three parts: **Frozen LLM**, **Residual SideNet**, and **Cached Memory Bank**. Its workflow is as follows:

- First, the long text sequence is split into fixed-length segments. Each segment is encoded layer by layer in the Frozen LLM, and the attention $K, V \in \mathbb{R}^{H \times M \times d}$ vector pairs are extracted from the $m$-th layer and cached in the Cached Memory Bank.
- When facing a new input sequence, the model retrieves the long-term memory bank based on the query-key of the current input, obtains the top $k$ key-value pairs (i.e., top-$k$ retrieval results) that are most relevant to the input, and integrates them into the subsequent language generation process; at the same time, the memory bank will remove the oldest content to ensure the availability of the latest context information.
- The Residual SideNet fuses the hidden layer output of the frozen LLM with the retrieved historical key-values during the inference stage to complete the effective modeling and context utilization of ultra-long text.

Through this decoupled design, LongMem can flexibly schedule massive historical information without expanding its native context window, taking into account both speed and long-term memory capabilities.

## Tool Use

Tool use is an important part of LLM Agents. By giving LLMs the ability to call external tools, their functions are significantly expanded: they can not only generate natural language, but also obtain real-time information, perform complex calculations, and interact with various systems (such as databases, APIs, etc.), thereby effectively breaking through the limitations of pre-trained knowledge and avoiding the inefficient process of reinventing the wheel.

Traditional LLMs mainly rely on pre-trained data for text generation, but this also makes them insufficient in mathematical operations, data retrieval, and real-time information updates. Through tool calls, the model can:

- **Improve computing power:** For example, by calling a dedicated calculator tool [Wolfram](https://gpt.wolfram.com/index.php.en), the model can perform more precise mathematical calculations, making up for its lack of arithmetic capabilities.

- **Obtain real-time information:** Using search engines like Google, Bing, or database APIs, the model can access the latest information to ensure the timeliness and accuracy of the generated content.
- **Enhance information credibility:** With the support of external tools, the model can cite real data sources, reduce the risk of information fabrication, and improve overall credibility.
- **Improve system transparency:** Tracking API call records can help users understand the model's decision-making process and provide a certain degree of interpretability.

Currently, various LLM applications based on tool calls have emerged in the industry. They use different strategies and architectures to achieve comprehensive coverage from simple tasks to complex multi-step reasoning.

### Toolformer

**Toolformer** ([Schick, et al. 2023](https://arxiv.org/abs/2302.04761)) is an LLM that can use external tools through simple APIs. It is trained by fine-tuning the GPT-J model, requiring only a few examples for each API. Toolformer learned to call tools including a question answering system, Wikipedia search, a calculator, a calendar, and a translation system:

{{< figure
    src="Toolformer_api.png"
    caption="Fig. 22. Examples of inputs and outputs for all APIs used. (Image source: [Schick, et al. 2023](https://arxiv.org/abs/2302.04761))"
    align="center"
    width="100%"
>}}

### HuggingGPT

**HuggingGPT** ([Shen, et al. 2023](https://arxiv.org/abs/2302.04761)) is a framework that uses ChatGPT as a task planner. It selects available models from [HuggingFace](https://huggingface.co/) by reading model descriptions to complete user tasks, and summarizes based on the execution results.

{{< figure
    src="HuggingGPT.png"
    caption="Fig. 23. Examples of inputs and outputs for all APIs used. (Image source: [Shen, et al. 2023](https://arxiv.org/abs/2302.04761))"
    align="center"
    width="100%"
>}}

The system consists of the following four stages:

- **Task Planning**: Parses user requests into multiple subtasks. Each task contains four attributes: task type, ID, dependencies, and arguments. The paper uses few-shot prompting to guide the model in task splitting and planning.
- **Model Selection**: Assigns each subtask to different expert models, using a multiple-choice approach to determine the most suitable model. Due to the limited context length, models need to be initially filtered based on task type.
- **Task Execution**: Expert models execute the assigned specific tasks and record the results. The results are passed to the LLM for subsequent processing.
- **Response Generation**: Receives the execution results of each expert model and finally outputs a summary answer to the user.

## LLM Agent Applications

### Generative Agent

The **Generative Agent** experiment ([Park, et al. 2023](https://arxiv.org/abs/2304.03442)) simulates realistic human behavior in a sandbox environment through 25 virtual characters driven by large language models. Its core design integrates memory, retrieval, reflection, and planning and reaction mechanisms, allowing the Agent to record and review its own experiences, and extract key information from them to guide subsequent actions and interactions.

{{< figure
    src="generative_agent_sandbox.png"
    caption="Fig. 24. The screenshot of generative agent sandbox. (Image source: [Park, et al. 2023](https://arxiv.org/abs/2304.03442))"
    align="center"
    width="100%"
>}}

The entire system uses a long-term memory module to record all observed events, combines a retrieval model to extract information based on recency, importance, and relevance, and then generates high-level inferences through a reflection mechanism, and finally converts these results into specific actions. This simulation experiment demonstrates emergent behaviors such as information diffusion, relationship memory, and social event coordination, providing a realistic human behavior simulation for interactive applications.

{{< figure
    src="generative_agent_architecture.png"
    caption="Fig. 25. The generative agent architecture. ([Park, et al. 2023](https://arxiv.org/abs/2304.03442))"
    align="center"
    width="100%"
>}}

### WebVoyager

**WebVoyager** ([He et al. 2024](https://arxiv.org/abs/2401.13919)) is an autonomous web interaction agent based on a large multimodal model that can control the mouse and keyboard for web browsing. WebVoyager uses the classic ReAct loop. In each interaction step, it views a browser screenshot annotated with a method similar to SoM (Set-of-Marks) ([Yang, et al. 2023](https://arxiv.org/abs/2310.11441)) – providing interaction hints by placing numerical labels on web elements – and then decides the next action. This visual annotation combined with the ReAct loop allows users to interact with web pages through natural language.  For specifics, you can refer to the [WebVoyager code](https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/) using the LangGraph framework.

{{< figure
    src="WebVoyager.png"
    caption="Fig. 26. The overall workflow of WebVoyager. (Image source: [He et al. 2024](https://arxiv.org/abs/2401.13919))"
    align="center"
    width="100%"
>}}

### OpenAI Operator

**Operator** ([OpenAI, 2025](https://openai.com/index/introducing-operator/)) is an AI agent recently released by OpenAI, designed to autonomously perform web tasks. Operator can interact with web pages like a human user, completing specified tasks through typing, clicking, and scrolling. The core technology of Operator is the **Computer-Using Agent (CUA)** ([OpenAI, 2025](https://openai.com/index/computer-using-agent/)). CUA combines the visual capabilities of GPT-4o with stronger reasoning capabilities obtained through reinforcement learning, and is specially trained to interact with graphical user interfaces (GUIs), including buttons, menus, and text boxes that users see on the screen.

{{< figure
    src="cua_overview.png"
    caption="Fig. 27. Overview of OpenAI CUA. (Image source: [OpenAI, 2025](https://openai.com/index/computer-using-agent/))"
    align="center"
    width="100%"
>}}

CUA operates in an iterative loop that includes three stages:

- **Perception**: CUA "observes" the web page content by capturing browser screenshots. This vision-based input method enables it to understand the layout and elements of the page.

- **Reasoning**: With the help of chain-of-thought reasoning, CUA evaluates the next action based on the current and previous screenshots and the actions that have been performed. This reasoning ability enables it to track task progress, review intermediate steps, and make adjustments as needed.

- **Action**: CUA interacts with the browser by simulating mouse and keyboard operations (such as clicking, typing, and scrolling). This enables it to perform various web tasks without specific API integration.

The difference between CUA and the previously existing WebVoyager is that this is an Agent specifically trained with reinforcement learning, rather than a fixed-process workflow built by directly calling GPT-4o. Although CUA is still in its early stages and has certain limitations, it has achieved SOTA results in the following benchmark tests.

{{< figure
    src="cua_benchmark.png"
    caption="Fig. 28. OpenAI CUA Benchmark Results. (Image source: [OpenAI, 2025](https://openai.com/index/computer-using-agent/))"
    align="center"
    width="100%"
>}}

## Deep Research

Deep Research is essentially a report generation system: given a user's query, the system uses an LLM as the core Agent, and after multiple rounds of iterative information retrieval and analysis, it finally generates a structured and informative report. Currently, the implementation logic of various Deep Research systems can be mainly divided into two methods: **Workflow Agent** and **RL Agent**.

### Workflow Agent vs RL Agent

The Workflow Agent approach relies on developers pre-designing workflows and manually constructing Prompts to organize the entire report generation process. The main features include:

- **Task Decomposition and Process Orchestration**: The system breaks down the user query into several subtasks, such as generating an outline, information retrieval, content summarization, etc., and then executes them in a predetermined process sequence.
- **Fixed Process**: The calls and interactions between each stage are pre-set, similar to building a static flow chart or directed acyclic graph (DAG), ensuring that each step has a clear responsibility.
- **Manual Design Dependence**: This method mainly relies on the experience of engineers, and improves the output quality by repeatedly debugging Prompts. It is highly applicable but has limited flexibility.

The [LangGraph](https://langchain-ai.github.io/langgraph/) framework can be used to build and orchestrate workflows in the form of graphs.

{{< figure
    src="langgraph_workflow.png"
    caption="Fig. 29. A workflow of the LangGraph. (Image source: [LangGraph, 2025](https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=workflow))"
    align="center"
    width="100%"
>}}

Currently, there are multiple open-source projects on Github that implement Deep Research Agents based on workflows, such as [GPT Researcher](https://github.com/assafelovic/gpt-researcher) and [open deep research](https://github.com/langchain-ai/open_deep_research).

{{< figure
    src="open_deep_research.png"
    caption="Fig. 30. An overview of the open deep research. (Image source: [LangChain, 2025](https://github.com/langchain-ai/open_deep_research))"
    align="center"
    width="100%"
>}}

The RL Agent is another implementation method that optimizes the Agent's multi-round search, analysis, and report writing process by training a reasoning model with RL. The main features include:

- **Autonomous Decision-Making Ability**: The system is trained through reinforcement learning, allowing the Agent to autonomously judge, make decisions, and adjust strategies when facing complex search and content integration tasks, thereby generating reports more efficiently.
- **Continuous Optimization**: Using a reward mechanism to score and provide feedback on the generation process, the Agent can continuously iterate and optimize its own strategy, improving the overall quality from task decomposition to final report generation.
- **Reduced Manual Intervention**: Compared to fixed processes that rely on manual Prompts, the reinforcement learning training method reduces the dependence on manual design and is more suitable for dealing with changing and complex real-world application scenarios.

The following table summarizes the main differences between these two methods:

| Feature               | Workflow Agent                             | RL Agent                            |
| ------------------ | ---------------------------------------------------- | -------------------------------------------------- |
| **Process Design**       | Pre-designed fixed workflow, clear task decomposition and process orchestration | End-to-end learning, Agent autonomous decision-making and dynamic process adjustment |
| **Autonomous Decision-Making**   | Relies on manually designed Prompts, the decision-making process is fixed and immutable | Through reinforcement learning, the Agent can autonomously judge, make decisions, and optimize strategies |
| **Manual Intervention**       | Requires a lot of manual design and debugging of Prompts, more manual intervention | Reduces manual intervention, achieves automatic feedback and continuous optimization through a reward mechanism |
| **Flexibility and Adaptability** | Less adaptable to complex or changing scenarios, limited scalability | More adaptable to changing and complex real-world scenarios, with high flexibility |
| **Optimization Mechanism**       | Optimization mainly relies on the engineer's experience adjustment, lacking an end-to-end feedback mechanism | Uses reinforcement learning's reward feedback to achieve continuous and automated performance improvement |
| **Implementation Difficulty**       | Relatively straightforward to implement, but requires cumbersome process design and maintenance | Requires training data and computing resources, with a larger initial development investment, but better long-term results |
| **Training Required**   | No additional training required, only relies on manually constructed processes and Prompts | Requires training the Agent through reinforcement learning to achieve autonomous decision-making |

### OpenAI Deep Research

**OpenAI Deep Research** ([OpenAI, 2025](https://openai.com/index/introducing-deep-research/)) is an intelligent Agent officially released by OpenAI in February 2025, designed for complex scenarios. It can automatically search, filter, analyze, and integrate multi-source information, and finally generate high-quality comprehensive reports. The system uses [o3](https://openai.com/index/openai-o3-mini/) as the core base, and combined with reinforcement learning methods, significantly improves the accuracy and robustness of the multi-round iterative search and reasoning process.

Compared with traditional ChatGPT plug-in search or conventional RAG technology, OpenAI Deep Research has the following outstanding advantages:

1. **Reinforcement Learning-Driven Iterative Reasoning**
   With the help of the **o3 reasoning model** and reinforcement learning training strategies, the Agent can continuously optimize its own reasoning path during the multi-round search and summarization process, effectively reducing the risk of distortion caused by error accumulation.

2. **Integration and Cross-Validation of Multi-Source Information**
   Breaking through the limitations of a single search engine, it can simultaneously call multiple authoritative data sources such as specific databases and professional knowledge bases, and form more reliable research conclusions through cross-validation.

3. **High-Quality Report Generation**
   The LLM-as-a-judge scoring mechanism and strict evaluation criteria are introduced in the training phase, so that the system can conduct self-evaluation when outputting reports, thereby generating professional texts with clearer structures and more rigorous arguments.

#### Training Process

The OpenAI Deep Research training process uses a **browser interaction dataset** specifically customized for research scenarios. Through these datasets, the model masters core browsing functions - including search, click, scroll, and file parsing; at the same time, it learns the ability to use Python tools in a sandbox environment for calculation, data analysis, and visualization. In addition, with the help of reinforcement learning training on these browsing tasks, the model can efficiently perform information retrieval, integration, and reasoning in a large number of websites, quickly locate key information, or generate comprehensive research reports.

These training datasets include both objective tasks with standard answers and automatic scoring, and open-ended tasks with detailed scoring rubrics. During the training process, the model's responses are strictly compared with standard answers or scoring criteria, and the evaluation model provides feedback by using the CoT thinking process generated by the model.

At the same time, the training process reuses the safety datasets accumulated during the o1 model training phase, and specifically adds safety training data for Deep Research scenarios to ensure that the model strictly complies with relevant compliance and safety requirements during the automated search and browsing process.

#### Performance

In the benchmark test **Humanity's Last Exam** ([Phan, et al. 2025](https://arxiv.org/abs/2501.14249)), which evaluates the ability of AI to answer expert-level questions in various professional fields, the model achieved SOTA results.

{{< figure
    src="human_last_exam.png"
    caption="Fig. 31. Humanity's Last Exam Benchmark Results。(Image source: [OpenAI, 2025](https://openai.com/index/introducing-deep-research/))"
    align="center"
    width="80%"
>}}

## Future Development Directions

Agents show broad prospects, but to achieve reliable and widespread application, the following key challenges still need to be addressed:

- **Context Window Limitation**: The context window of LLMs limits the amount of information processed, affecting long-term planning and memory capabilities, and reducing task coherence. Current research explores external memory mechanisms and context compression techniques to enhance long-term memory and complex information processing capabilities. Currently, OpenAI's latest model **GPT-4.5** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-4-5/)) has a maximum context window of 128k tokens.

- **Interface Standardization and Interoperability**: The current natural language-based tool interaction has the problem of inconsistent formats. The **Model Context Protocol (MCP)** ([Anthropic, 2024](https://www.anthropic.com/news/model-context-protocol)) unifies the interaction between LLMs and applications through open standards, reducing development complexity, improving system stability, and cross-platform compatibility.

- **Task Planning and Decomposition Capabilities**: Agents have difficulty formulating coherent plans, effectively decomposing subtasks, and lack the ability to dynamically adjust in unexpected situations in complex tasks. More powerful planning algorithms, self-reflection mechanisms, and dynamic strategy adjustment methods are needed to flexibly respond to uncertain environments.

- **Computing Resources and Economic Benefits**: Deploying large model agents is costly due to multiple API calls and intensive computing, limiting some practical application scenarios. Optimization directions include efficient model structure, quantization technology, inference optimization, caching strategies, and intelligent scheduling mechanisms. With the development of dedicated GPU hardware such as [NVIDIA DGX B200](https://www.nvidia.com/en-sg/data-center/dgx-b200/) and distributed technologies, computing efficiency is expected to be significantly improved.

- **Security Protection and Privacy Assurance**: Agents face security risks such as prompt injection, and need to establish sound authentication, access control, input validation, and sandbox environments. For multimodal input and external tools, it is necessary to strengthen data anonymization, the principle of least privilege, and audit logs to meet security and privacy compliance requirements.

- **Decision Transparency and Interpretability**: Agent decisions are difficult to explain, limiting their application in high-risk areas. Enhancing interpretability requires the development of visualization tools, chain-of-thought tracking, and decision reason generation mechanisms to improve decision transparency, enhance user trust, and meet regulatory requirements.

## References

[1] DAIR.AI. ["LLM Agents."](https://www.promptingguide.ai/research/llm-agents) Prompt Engineering Guide, 2024.

[2] Sutton, Richard S., and Andrew G. Barto. ["Reinforcement Learning: An Introduction."](http://incompleteideas.net/book/the-book.html) MIT Press, 2018.

[3] Weng, Lilian. ["LLM-powered Autonomous Agents."](https://lilianweng.github.io/posts/2023-06-23-agent/) Lil’Log, 2023.

[4] Wei, Jason, et al. ["Chain-of-thought prompting elicits reasoning in large language models."](https://arxiv.org/abs/2201.11903) Advances in neural information processing systems 35 (2022): 24824-24837.

[5] Kojima, Takeshi, et al. ["Large language models are zero-shot reasoners."](https://arxiv.org/abs/2205.11916) Advances in neural information processing systems 35 (2022): 22199-22213.

[6] Wang, Xuezhi, et al. ["Self-consistency improves chain of thought reasoning in language models."](https://arxiv.org/abs/2203.11171) arXiv preprint arXiv:2203.11171 (2022).

[7] Wang, Xuezhi, et al. ["Rationale-augmented ensembles in language models."](https://arxiv.org/abs/2207.00747) arXiv preprint arXiv:2207.00747 (2022).

[8] Zelikman, Eric, et al. ["Star: Bootstrapping reasoning with reasoning."](https://arxiv.org/abs/2203.14465) Advances in Neural Information Processing Systems 35 (2022): 15476-15488.

[9] Fu, Yao, et al. ["Complexity-based prompting for multi-step reasoning."](https://arxiv.org/abs/2210.00720) arXiv preprint arXiv:2210.00720 (2022).

[10] Yao, Shunyu, et al. ["Tree of thoughts: Deliberate problem solving with large language models."](https://arxiv.org/abs/2305.10601) Advances in neural information processing systems 36 (2023): 11809-11822.

[11] Yao, Shunyu, et al. ["React: Synergizing reasoning and acting in language models."](https://arxiv.org/abs/2210.03629) International Conference on Learning Representations (ICLR). 2023.

[12] Shinn, Noah, et al. ["Reflexion: Language agents with verbal reinforcement learning."](https://arxiv.org/abs/2303.11366) Advances in Neural Information Processing Systems 36 (2023): 8634-8652.

[13] Guo, Daya, et al. ["Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning."](https://arxiv.org/abs/2501.12948) arXiv preprint arXiv:2501.12948 (2025).

[14] OpenAI. ["Introducing OpenAI o1"](https://openai.com/o1/) OpenAI, 2024.

[15] Zhang, Zeyu, et al. ["A survey on the memory mechanism of large language model based agents."](https://arxiv.org/abs/2404.13501) arXiv preprint arXiv:2404.13501 (2024).

[16] Zhang, Chaoyun, et al. ["Large language model-brained gui agents: A survey."](https://arxiv.org/abs/2411.18279) arXiv preprint arXiv:2411.18279 (2024).

[17] Wang, Weizhi, et al. ["Augmenting language models with long-term memory."](https://arxiv.org/abs/2306.07174) Advances in Neural Information Processing Systems 36 (2023): 74530-74543.

[18] Schick, Timo, et al. ["Toolformer: Language models can teach themselves to use tools."](https://arxiv.org/abs/2302.04761) Advances in Neural Information Processing Systems 36 (2023): 68539-68551.

[19] Shen, Yongliang, et al. ["Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face."](https://arxiv.org/abs/2303.17580) Advances in Neural Information Processing Systems 36 (2023): 38154-38180.

[20] Park, Joon Sung, et al. ["Generative agents: Interactive simulacra of human behavior."](https://arxiv.org/abs/2304.03442) Proceedings of the 36th annual acm symposium on user interface software and technology. 2023.

[21] He, Hongliang, et al. ["WebVoyager: Building an end-to-end web agent with large multimodal models."](https://arxiv.org/abs/2401.13919) arXiv preprint arXiv:2401.13919 (2024).

[22] Yang, Jianwei, et al. ["Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v."](https://arxiv.org/abs/2310.11441) arXiv preprint arXiv:2310.11441 (2023).

[23] OpenAI. ["Introducing Operator."](https://openai.com/index/introducing-operator/) OpenAI, 2025.

[24] OpenAI. ["Computer-Using Agent."](https://openai.com/index/computer-using-agent/) OpenAI, 2025.

[25] OpenAI. ["Introducing Deep Research."](https://openai.com/index/introducing-deep-research/) OpenAI, 2025.

[26] Phan, Long, et al. ["Humanity's Last Exam."](https://arxiv.org/abs/2501.14249) arXiv preprint arXiv:2501.14249 (2025).

[27] OpenAI. ["Introducing GPT-4.5."](https://openai.com/index/introducing-gpt-4-5/) OpenAI, 2025.

[28] Anthropic. ["Introducing the Model Context Protocol."](https://www.anthropic.com/news/model-context-protocol) Anthropic, 2024.

[29] LangGraph. ["A workflow of the LangGraph."](https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=workflow) LangGraph Tutorials, 2025.

[30] Assaf Elovic. ["GPT Researcher"](https://github.com/assafelovic/gpt-researcher) GitHub Repository, 2025.

[31] LangChain. ["Open Deep Research"](https://github.com/langchain-ai/open_deep_research) GitHub Repository, 2025.

## Citation

> **Citation**: Please cite the original author and source when reprinting or referencing the content of this article.

**Cited as:**

> Yue Shui.(Mar 2025). Large Language Model Agents.
https://syhya.github.io/posts/2025-03-27-llm-agent

Or

```bibtex
@article{syhya2025llm-agent,
  title   = "Large Language Model Agents",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Mar",
  url     = "https://syhya.github.io/posts/2025-03-27-llm-agent"  
}
```