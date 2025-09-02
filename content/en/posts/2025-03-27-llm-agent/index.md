---
title: "Large Language Model Agents"
date: "2025-03-27T10:00:00+00:00"
lastmod: "2025-09-02T10:00:00+00:00"
author: "Yue Shui"
categories: ["Technical Blog"]
tags: ["Large Language Model", "AI", "Agent", "Reinforcement Learning", "Planning", "Memory", "Tool Use", "Deep Research", "ReAct", "Reflexion", "WebVoyager", "OpenAI Operator", "CoT", "ToT", "Workflow"]
readingTime: 30
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

## Agents

Since the release of ChatGPT by OpenAI in October 2022, and with the emergence of subsequent projects like [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) and [AgentGPT](https://github.com/reworkd/AgentGPT), LLM-related agents have become a research hotspot and a practical application direction in AI in recent years. This article will introduce the basic concepts, core technologies, and latest application progress of agents.

### LLM Agent

A **Large Language Model Agent (LLM agent)** utilizes an LLM as its core "brain" and combines modules like planning, memory, and external tools to automate the execution of complex tasks.

- **User Request:** The user interacts with the agent by inputting tasks through prompts.
- **Agent:** The system's brain, composed of one or more LLMs, responsible for overall coordination and task execution.
- **Planning:** Decomposes complex tasks into smaller sub-tasks, formulates an execution plan, and continuously optimizes the results through reflection.
- **Memory:** Includes short-term memory (capturing task information in real-time using in-context learning) and long-term memory (using external vector stores to save and retrieve key information, ensuring information continuity for long-running tasks).
- **Tools:** Integrates external tools such as calculators, web search, and code interpreters to call external data, execute code, and obtain the latest information.

{{< figure
    src="llm_agent.png"
    caption="Fig. 1. The illustration of LLM Agent Framework. (Image source: [DAIR.AI, 2024](https://www.promptingguide.ai/research/llm-agents#llm-agent-framework))"
    align="center"
    width="70%"
>}}

### RL Agent

The goal of **Reinforcement Learning (RL)** is to train an agent to take a series of actions ($a_t$) in a given environment. During interaction, the agent transitions from one state ($s_t$) to the next and receives a reward ($r_t$) from the environment after each action. This interaction generates a complete trajectory ($\tau$), typically represented as:

$$
\tau = \{(s_0, a_0, r_0), (s_1, a_1, r_1), \dots, (s_T, a_T, r_T)\}.
$$

The agent's objective is to learn a policy ($\pi$), which is a rule for selecting actions in each state, to **maximize the expected cumulative reward**, usually expressed as:

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

In the context of **LLMs**, the model can be viewed as an agent, and the "environment" can be understood as the user's input and the desired response format:

- **State ($s_t$)**: Can be the current dialogue context or the user's question.
- **Action ($a_t$)**: The text output by the model (answer, generated content, etc.).
- **Reward ($r_t$)**: Feedback from the user or the system (e.g., user satisfaction, automatic scoring by a reward model).
- **Trajectory ($\tau$)**: The entire sequence of text interactions from the beginning to the end of a conversation, which can be used to evaluate the model's overall performance.
- **Policy ($\pi$)**: The rule by which the LLM generates text in each state (dialogue context), typically determined by the model's parameters.

For LLMs, they are traditionally pre-trained on massive offline datasets. In the post-training reinforcement learning phase, the model is trained with feedback from humans or other models to produce high-quality text that better aligns with human preferences or task requirements.

### Comparison

The table below shows the differences between the two:

| **Comparison Dimension** | **LLM Agent**                                                              | **RL Agent**                                                              |
|--------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **Core Principle**       | Automates complex tasks through planning, memory, and tools.               | Continuously optimizes its policy to maximize long-term rewards through a trial-and-error feedback loop with the environment. |
| **Optimization Method**  | **Does not directly update model parameters.** Performance is improved mainly through context extension, external memory, and tools. | **Continuously and frequently updates policy model parameters,** relying on reward signals from the environment for optimization. |
| **Interaction Method**   | Interacts with users or external systems using natural language, flexibly calling various tools to obtain external information. | Interacts with a real or simulated environment, which provides rewards or penalties, forming a closed feedback loop. |
| **Objective**            | Decomposes complex tasks and completes them with the help of external resources, focusing on the quality and accuracy of the task outcome. | Maximizes long-term rewards, seeking an optimal balance between short-term and long-term returns. |

As research progresses, the combination of LLM and RL agents presents more possibilities, such as:
- Using reinforcement learning methods to train Reasoning LLMs (like o1/o3), making them more suitable as base models for LLM agents.
- Simultaneously, recording the data and feedback from LLM agents executing tasks to provide rich training data for Reasoning LLMs, thereby enhancing model performance.

## Prompt Engineering

**Prompt Engineering**, also known as **In-Context Prompting**, is the technique of optimizing input prompts to guide an LLM to produce the desired output. Its core objective is to control the model's behavior through effective communication **without updating the model's weights**.

### Zero-Shot Prompting

**Zero-Shot Prompting** directly provides the model with task instructions without any examples. This method relies entirely on the knowledge and instruction-following capabilities the model learned during its pre-training phase. For example, for sentiment analysis:

{{< figure
    src="zero_shot.png"
    caption="Fig. 3. Zero-Shot Prompting."
    align="center"
    width="100%"
>}}

For models that have undergone instruction fine-tuning, such as GPT-5 or Claude 4, they can understand and execute these direct instructions very well.

### Few-Shot Prompting

**Few-Shot Prompting** provides a set of high-quality examples in the prompt, with each example containing an input and the desired output. Through these examples, the model can better understand the user's intent and the specific requirements of the task, leading to better performance than zero-shot prompting. However, a drawback of this method is that it consumes more of the context window length. For example, providing a few sentiment analysis examples:

{{< figure
    src="few_shot.png"
    caption="Fig. 4. Few-Shot Prompting."
    align="center"
    width="100%"
>}}

### Automatic Prompt Construction

**Automatic Prompt Engineer (APE)** ([Zhou et al. 2022](https://arxiv.org/abs/2211.01910)) is a method that searches through a pool of candidate instructions generated by the model. It filters the candidate set and ultimately selects the highest-scoring instruction based on a chosen scoring function.

{{< figure
    src="ape.png"
    caption="Fig. 5. Automatic Prompt Engineer (APE) workflow. (Image source: [Zhou et al. 2022](https://arxiv.org/abs/2211.01910))"
    align="center"
    width="100%"
>}}

**Automatic Chain-of-Thought (Auto-CoT)** ([Zhang et al. 2022](https://arxiv.org/abs/2210.03493)) proposes a method for automatically constructing Chain-of-Thought examples, aiming to solve the problem of manual prompt design being time-consuming and potentially suboptimal. Its core idea is to sample questions through **clustering techniques** and then **leverage the LLM's own zero-shot reasoning capabilities to automatically generate reasoning chains**, thereby constructing diverse, high-quality examples.

{{< figure
    src="auto_cot.png"
    caption="Fig. 6. Overview of the Auto-CoT method. (Image source: [Zhang et al. 2022](https://arxiv.org/abs/2210.03493))"
    align="center"
    width="100%"
>}}

**Auto-CoT consists of two main stages:**
1.  **Question Clustering**: Embeds the questions in the dataset and runs an algorithm like $k$-means for clustering. This step aims to group similar questions into the same cluster to ensure the diversity of subsequently sampled questions.
2.  **Demonstration Selection & Rationale Generation**: Selects one or more representative questions from each cluster (e.g., choosing the question closest to the cluster centroid). Then, it uses a **Zero-Shot CoT** prompt to have the LLM generate reasoning chains for these selected questions. These automatically generated "question-rationale" pairs form the final few-shot prompt used for task execution.

## Knowledge Enhancement

When dealing with knowledge-intensive or commonsense reasoning tasks, relying solely on the LLM's parametric knowledge is often insufficient and can lead to incorrect or outdated answers. To address this issue, researchers have proposed two main approaches:

**Generated Knowledge Prompting** ([Liu et al. 2022](https://arxiv.org/abs/2110.08387)) is a method where the model is first prompted to **generate relevant knowledge** before making a prediction. The core idea is that when a task requires commonsense or external information, the model may make mistakes due to a lack of context. If the model is first guided to generate knowledge related to the input and then answers based on that knowledge, the accuracy of its reasoning can be improved.

{{< figure
    src="generated_knowledge_prompting.png"
    caption="Fig. 7. Overview of the Generated Knowledge Prompting. (Image source: [Liu et al. 2022](https://arxiv.org/abs/2110.08387))"
    align="center"
    width="100%"
>}}

1. **Knowledge Generation**: Based on the input, the model first generates relevant factual knowledge.
2. **Knowledge Integration**: The generated knowledge is combined with the original question to form a new prompt.
3. **Answer Prediction**: The model provides an answer based on the enhanced input.

**Retrieval Augmented Generation (RAG)** ([Lewis et al. 2021](https://arxiv.org/abs/2005.11401)) is a method that combines **information retrieval with text generation** to tackle knowledge-intensive tasks. Its core idea is that relying solely on an LLM's parametric (static) knowledge can easily lead to factual errors. By introducing retrieval from external knowledge bases, the **factual consistency and timeliness** of the generated results can be improved.

{{< figure
    src="rag.png"
    caption="Fig. 8. Overview of the Retrieval Augmented Generation. (Image source: [Lewis et al. 2021](https://arxiv.org/abs/2005.11401))"
    align="center"
    width="100%"
>}}

1. **Retrieval**: Retrieves relevant documents from an external knowledge source (e.g., Wikipedia or a private knowledge base).
2. **Augmentation**: Concatenates the retrieved documents with the original input to serve as the prompt context.
3. **Generation**: A generation model (the original paper used a pre-trained seq2seq model, but today LLMs are mainstream) outputs the answer based on the augmented prompt.

### Multimodal Chain-of-Thought Prompting

**Multimodal CoT Prompting (MCoT)** ([Zhang et al. 2023](https://arxiv.org/abs/2302.00923)) integrates **text and visual information** into the reasoning process, breaking the limitation of traditional CoT, which relies solely on the language modality. Its framework is divided into two stages:
1. **Rationale Generation**: Generates an explanatory reasoning chain based on multimodal information (text + image).
2. **Answer Inference**: Uses the generated rationale as an aid to infer the final answer.

{{< figure
    src="MCoT.png"
    caption="Fig. 9. Overview of our Multimodal-CoT framework. (Image source: [Zhang et al. 2023](https://arxiv.org/abs/2302.00923))"
    align="center"
    width="100%"
>}}

### Active Prompt

**Active Prompt** ([Diao et al. 2023](https://arxiv.org/abs/2302.12246)) addresses the limitation of traditional CoT methods that rely on a fixed set of manually annotated examples. The problem is that **fixed examples are not necessarily optimal for all tasks, which can lead to poor generalization**. Active Prompt introduces an active learning strategy to adaptively select and update the best task-relevant examples, thereby improving the model's reasoning performance.

{{< figure
    src="active_prompt.png"
    caption="Fig. 10. Illustrations of active prompting framework. (Image source: [Diao et al. 2023](https://arxiv.org/abs/2302.12246))"
    align="center"
    width="100%"
>}}

1. **Uncertainty Estimation**: With or without a small number of manual CoT examples, the LLM generates *k* answers for training questions (in the paper, *k=5*), and an uncertainty metric is calculated based on the variance of these answers.
2. **Selection**: Based on the uncertainty level, the most uncertain questions are selected.
3. **Annotation**: The selected questions are manually annotated to provide new, high-quality CoT examples.
4. **Inference**: The newly annotated examples are used for inference, improving the model's performance on the target task.

## Planning: Task Decomposition

The core components of an LLM Agent include **planning**, **memory**, and **tool use**. These components work together to enable the agent to autonomously execute complex tasks.

{{< figure
    src="llm_agent_overview.png"
    caption="Fig. 11. Overview of a LLM-powered autonomous agent system. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}

Planning is crucial for the successful execution of complex tasks. It can be approached in different ways depending on the complexity and the need for iterative refinement. In simple scenarios, the planning module can use an LLM to outline a detailed plan in advance, including all necessary sub-tasks. This step ensures that the agent systematically performs **task decomposition** and follows a clear logical flow from the outset.

### Chain of Thought

**Chain of Thought (CoT)** ([Wei et al. 2022](https://arxiv.org/abs/2201.11903)) works by generating a series of short sentences step-by-step to describe the reasoning process; these sentences are called reasoning steps. Its purpose is to explicitly show the model's reasoning path, helping it better handle **complex reasoning tasks**. The figure below shows the difference between few-shot prompting (left) and CoT prompting (right). The few-shot prompt leads to an incorrect answer, while the CoT method guides the model to state its reasoning process step-by-step, more clearly reflecting the model's logical process and thus improving the answer's accuracy and interpretability.

{{< figure
    src="cot.png"
    caption="Fig. 12. The comparison example of few-shot prompting and CoT prompting. (Image source: [Wei et al. 2022](https://arxiv.org/abs/2201.11903))"
    align="center"
    width="100%"
>}}

**Zero-Shot CoT** ([Kojima et al. 2022](https://arxiv.org/abs/2205.11916)) is a follow-up to CoT that proposes an extremely simple zero-shot prompting method. They found that by simply appending the phrase `Let's think step by step` to the end of the question, the LLM can produce a chain of thought, leading to more accurate answers.

{{< figure
    src="zero_shot_cot.png"
    caption="Fig. 13. The comparison example of few-shot prompting and CoT prompting. (Image source: [Kojima et al. 2022](https://arxiv.org/abs/2205.11916))"
    align="center"
    width="100%"
>}}

### Self-Consistency Sampling

**Self-consistency sampling** ([Wang et al. 2022a](https://arxiv.org/abs/2203.11171)) generates **multiple diverse answers** by sampling multiple times from the same prompt with a `temperature > 0` and then selecting the best answer from the set. The core idea is to improve the final answer's accuracy and robustness by sampling multiple reasoning paths and then taking a majority vote. The criteria for selecting the best answer can vary for different tasks, but **majority voting** is generally used as a universal solution. For tasks that are easy to verify, such as programming problems, the answers can be validated by running them through an interpreter and using unit tests. This is an optimization of CoT, and when used in combination, it can significantly improve the model's performance on complex reasoning tasks.

{{< figure
    src="self_consistency.png"
    caption="Fig. 14. Overview of the Self-Consistency Method for Chain-of-Thought Reasoning. (Image source: [Wang et al. 2022a](https://arxiv.org/abs/2203.11171))"
    align="center"
    width="100%"
>}}

Here are some subsequent optimization works:

- ([Wang et al. 2022b](https://arxiv.org/abs/2207.00747)) later used another ensemble learning method for optimization, increasing randomness by **changing the order of examples** or **replacing human-written reasoning with model-generated ones**, followed by majority voting.

{{< figure
    src="rationale_augmented.png"
    caption="Fig. 15. An overview of different ways of composing rationale-augmented ensembles, depending on how the randomness of rationales is introduced. (Image source: [Wang et al. 2022b](https://arxiv.org/abs/2207.00747))"
    align="center"
    width="100%"
>}}

- If training samples only provide the correct answer without the reasoning, the **STaR (Self-Taught Reasoner)** ([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)) method can be used: (1) Have the LLM generate reasoning chains, and keep only the reasoning for answers that are correct. (2) Fine-tune the model with the generated reasoning, iterating until convergence. Note that a high `temperature` can easily lead to results with the correct answer but incorrect reasoning. If there is no ground truth, majority voting can be considered the "correct answer."

{{< figure
    src="STaR.png"
    caption="Fig. 16. An overview of STaR and a STaR-generated rationale on CommonsenseQA. (Image source: [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465))"
    align="center"
    width="100%"
>}}

- ([Fu et al. 2023](https://arxiv.org/abs/2210.00720)) found that more complex examples (with more reasoning steps) can improve model performance. When separating reasoning steps, a newline character `\n` works better than `step i`, `.`, or `;`. Additionally, using a complexity-based consistency strategy, which only performs majority voting on the top-$k$ most complex generated reasoning chains, can further optimize the model's output. It was also shown that replacing `Q:` with `Question:` in the prompt provides an additional performance boost.

{{< figure
    src="linebreak.png"
    caption="Fig. 17. Sensitivity analysis on step formatting. Complex prompts consistently lead to better performance with regard to different step formatting. (Image source: [Fu et al. 2023](https://arxiv.org/abs/2210.00720))"
    align="center"
    width="100%"
>}}

### Tree of Thoughts

**Tree of Thoughts (ToT)** ([Yao et al. 2023](https://arxiv.org/abs/2305.10601)) expands on CoT by exploring multiple reasoning possibilities at each step. It first decomposes a problem into multiple thought steps and generates several different ideas at each step, forming a tree structure. The search process can use Breadth-First Search (BFS) or Depth-First Search (DFS), and each state is evaluated by a classifier (or by having the LLM score it) or through majority voting. It involves three main steps:

- **Expand**: Generate one or more candidate solutions.
- **Score**: Measure the quality of the candidate solutions.
- **Prune**: Keep the top-$k$ best candidate solutions.

If no solution is found (or if the quality of the candidates is not high enough), the process backtracks to the expansion step.

{{< figure
    src="tot.png"
    caption="Fig. 18. Schematic illustrating various approaches to problem solving with LLMs. (Image source: [Yao et al. 2023](https://arxiv.org/abs/2305.10601))"
    align="center"
    width="100%"
>}}

## Planning: Self-Reflexion

**Self-Reflexion** is a key factor that enables an agent to achieve iterative improvement by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks where trial and error are inevitable.

### ReAct

The **ReAct (Reason + Act)** ([Yao et al. 2023](https://arxiv.org/abs/2210.03629)) framework achieves seamless integration of reasoning and acting in LLMs by combining task-specific discrete actions with the language space. This design not only allows the model to interact with the environment by calling external interfaces like the Wikipedia search API but also to generate detailed reasoning trajectories in natural language to solve complex problems.

The ReAct prompt template includes explicit thought steps, with the basic format as follows:

```plaintext
Thought: ...
Action: ...
Observation: ...
...(Repeated many times)
```

{{< figure
    src="ReAct.png"
    caption="Fig. 19. Examples of reasoning trajectories for knowledge-intensive tasks (e.g. HotpotQA, FEVER) and decision-making tasks (e.g. AlfWorld Env, WebShop). (Image source: [Yao et al. 2023](https://arxiv.org/abs/2210.03629))"
    align="center"
    width="100%"
>}}

As seen in the figure below, ReAct significantly outperforms the baseline `Action`-only method in both knowledge-intensive and decision-making tasks, demonstrating its advantages in enhancing reasoning effectiveness and interactive performance.

{{< figure
    src="ReAct_res.png"
    caption="Fig. 20. PaLM-540B prompting results on HotpotQA and Fever. (Image source: [Yao et al. 2023](https://arxiv.org/abs/2210.03629))"
    align="center"
    width="50%"
>}}

### Reflexion

**Reflexion** ([Shinn et al. 2023](https://arxiv.org/abs/2303.11366)) enables an LLM to iteratively improve and optimize its decisions through self-feedback and dynamic memory.

This method essentially borrows ideas from reinforcement learning. In the traditional Actor-Critic model, the Actor selects an action $a_t$ based on the current state $s_t$, while the Critic provides an evaluation (e.g., a value function $V(s_t)$ or an action-value function $Q(s_t, a_t)$) and gives feedback to the Actor for policy optimization. Correspondingly, in Reflexion's three main components:

- **Actor**: Played by the LLM, it outputs text and corresponding actions based on the environment's state (including context and historical information). This can be denoted as:

  $$
  a_t = \pi_\theta(s_t),
  $$

  where $\pi_\theta$ represents the policy based on parameters $\theta$ (i.e., the LLM's weights or prompt). The Actor interacts with the environment and produces a trajectory $\tau = \{(s_1,a_1,r_1), \dots, (s_T,a_T,r_T)\}$.

- **Evaluator**: Similar to the Critic, the Evaluator receives the trajectory generated by the Actor and outputs a reward signal $r_t$. In the Reflexion framework, the Evaluator can analyze the trajectory using pre-designed heuristic rules or an additional LLM to generate rewards. For example:

  $$
  r_t = R(\tau_t),
  $$

  where $R(\cdot)$ is the reward function based on the current trajectory $\tau_t$.

- **Self-Reflection**: This module adds a self-regulating feedback mechanism on top of the Actor-Critic model. It integrates the current trajectory $\tau$, reward signals $\{r_t\}$, and historical experience from long-term memory, using its language generation capabilities to produce self-improvement suggestions for the next decision. This feedback is then written to external memory, providing richer context for the Actor's subsequent decisions, thus achieving iterative optimization similar to updating policy parameters $\theta$ through dynamic adjustment of the prompt, without updating the LLM's internal parameters.

{{< figure
    src="Reflexion.png"
    caption="Fig. 21. (a) Diagram of Reflexion. (b) Reflexion reinforcement algorithm. (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

The core loop and algorithm of Reflexion are described as follows:

- **Initialization**
   - Instantiate the Actor, Evaluator, and Self-Reflection models simultaneously (all can be implemented by LLMs), denoted as $M_a, M_e, M_{sr}$ respectively.
   - Initialize the policy $\pi_\theta$ (including the Actor's model parameters or prompt, and initial memory).
   - The Actor first generates an initial trajectory $\tau_0$ according to the current policy $\pi_\theta$. After evaluation by $M_e$, $M_{sr}$ generates the first self-reflection text and stores it in long-term memory.

- **Generate Trajectory**
   - In each iteration, $M_a$ reads the current long-term memory and environmental observations, sequentially outputting actions $\{a_1, a_2, \ldots\}$, interacting with the environment, and receiving corresponding feedback to form a new trajectory $\tau_t$. $\tau_t$ can be considered the short-term memory for this task, used only in the current iteration.

- **Evaluation**
   - $M_e$ outputs rewards or scores $\{r_1, r_2, \ldots\}$ based on the trajectory $\tau_t$ (i.e., the sequence of the Actor's behaviors and environmental feedback). This step corresponds to internal feedback from $M_e$ or results directly provided by the external environment.

- **Self-Reflection**
   - The $M_{sr}$ module synthesizes the trajectory $\tau_t$ and reward signals $\{r_t\}$ to generate self-correction or improvement suggestions $\mathrm{sr}_t$ at the language level.
   - The reflection text can be seen as an analysis of errors or as providing new inspirational ideas, and it is stored in long-term memory. In practice, we can vectorize the feedback information and store it in a vector database.

- **Update and Repeat**
   - After appending the latest self-reflection text $\mathrm{sr}_t$ to long-term memory, the Actor can use RAG to retrieve relevant historical information in the next iteration to adjust its policy.
   - Repeat the above steps until $M_e$ determines the task is completed or the maximum number of rounds is reached. In this loop, Reflexion relies on the continuous accumulation of **self-reflection + long-term memory** to improve decision-making, rather than directly modifying model parameters.

Below are examples of Reflexion applied to decision-making, programming, and reasoning tasks:

{{< figure
    src="reflextion_examples.png"
    caption="Fig. 22. Reflexion works on decision-making, programming, and reasoning tasks. (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

In an experiment with 100 HotPotQA questions, a comparison between the CoT method and a method with episodic memory showed that the Reflexion method, with an added self-reflection step at the end, significantly improved its search, information retrieval, and reasoning capabilities.

{{< figure
    src="reflextion_result.png"
    caption="Fig. 23. Comparative Analysis of Chain-of-Thought (CoT) and ReAct on the HotPotQA Benchmark. (Image source: [Shinn et al. 2023](https://arxiv.org/abs/2303.11366))"
    align="center"
    width="100%"
>}}

### DeepSeek R1

**DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)) represents a major breakthrough in the open-source community's efforts to replicate OpenAI's o1 ([OpenAI, 2024](https://openai.com/o1/)), successfully training an advanced reasoning model with deep reflection capabilities through reinforcement learning techniques.

> For a detailed training process and technical implementation of DeepSeek R1, please refer to my previous blog post: [Progress in Replicating OpenAI o1: DeepSeek-R1](https://syhya.github.io/en/posts/2025-01-27-deepseek-r1/).

A key transformation during the training of DeepSeek-R1-Zero is that as training progresses, the model gradually **emerges** with an outstanding **self-evolution** capability. This capability is manifested in three core aspects:

- **Self-Reflection**: The model can look back and critically evaluate previous reasoning steps.
- **Proactive Exploration**: When it finds the current problem-solving path to be suboptimal, it can autonomously search for and try alternative solutions.
- **Dynamic Thought Adjustment**: It adaptively adjusts the number of generated tokens based on the complexity of the problem, achieving a deeper thought process.

This dynamic and spontaneous reasoning behavior significantly enhances the model's ability to solve complex problems, enabling it to tackle challenging tasks more efficiently and accurately.

{{< figure
    src="deepseek_r1_zero_response_time.png"
    caption="Fig. 24. The average response length of DeepSeek-R1-Zero on the training set during the RL process. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

A typical "aha moment" also emerged during the training of DeepSeek-R1-Zero. At this critical stage, the model suddenly realized during the reasoning process that its previous line of thought was flawed, and it quickly adjusted its thinking direction, ultimately leading to the correct answer. This phenomenon strongly demonstrates that the model has developed powerful **self-correction** and **reflection capabilities** in its reasoning process, similar to the "aha" experience in human thinking.

{{< figure
    src="aha_moment.png"
    caption="Fig. 25. An interesting “aha moment” of an intermediate version of DeepSeek-R1-Zero. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

## Memory

### Human Memory

**Memory** refers to the process of acquiring, storing, retaining, and retrieving information. Human memory is primarily divided into the following three categories:

{{< figure
    src="category_human_memory.png"
    caption="Fig. 26. Categorization of human memory. (Image source: [Weng, 2017](https://lilianweng.github.io/posts/2023-06-23-agent/))"
    align="center"
    width="100%"
>}}

- **Sensory Memory:** Used to briefly retain sensory information after the original stimulus (visual, auditory, tactile, etc.) has disappeared, typically lasting for milliseconds or seconds. Sensory memory is further divided into:
    - Iconic Memory: The transient image or visual impression retained by the visual channel, generally lasting 0.25–0.5 seconds, used to form visual continuity in video or animation scenes.
    - Echoic Memory: The brief storage of auditory information, which can last for several seconds, allowing a person to replay recently heard sentences or sound clips.
    - Haptic Memory: Used to retain brief tactile or force information, generally lasting from milliseconds to seconds, such as the brief finger sensations when typing on a keyboard or reading Braille.

- **Short-Term Memory:** Stores the information we are currently conscious of.
    - Lasts for about 20–30 seconds, with a capacity typically of 7±2 items.
    - Responsible for the temporary processing and maintenance of information during complex cognitive tasks like learning and reasoning.

- **Long-Term Memory:** Can store information for days to decades, with a virtually unlimited capacity. Long-term memory is divided into:
  - Explicit Memory: Can be consciously recalled, including episodic memory (personal experiences, event details) and semantic memory (facts and concepts).
  - Implicit Memory: Unconscious memory, primarily related to skills and habits, such as riding a bike or touch typing.

These three types of human memory are intertwined and together form our cognition and understanding of the world. When building LLM Agents, we can draw inspiration from this classification of human memory:

- **Sensory Memory** corresponds to the LLM's embedding representations of raw input data (such as text, images, and videos).
- **Short-Term Memory** corresponds to the LLM's in-context learning, limited by the model's context window `max_tokens`. When the conversation length exceeds the window, earlier information is truncated.
- **Long-Term Memory** corresponds to an external vector store or database, where the Agent can retrieve historical information on demand using **RAG** technology.

### LLM Agent Memory

When an Agent engages in multi-turn interactions with a user or executes multi-step tasks, it can utilize different forms of memory and environmental information to complete its workflow.

{{< figure
    src="llm_memory_overview.png"
    caption="Fig. 27. An overview of the sources, forms, and operations of the memory in LLM-based agents. (Image source: [Zhang et al. 2024](https://arxiv.org/abs/2404.13501))"
    align="center"
    width="100%"
>}}

- **Textual Memory**
  - Full Interaction: Records all dialogue and action trajectories, helping the Agent trace back the context.
  - Recent Interaction: Retains only the dialogue content highly relevant to the current task, reducing unnecessary context usage.
  - Retrieved Interaction: The Agent can retrieve historical dialogues or records related to the current task from an external knowledge base and integrate them into the current context.
  - External Knowledge: When the Agent encounters a knowledge gap, it can retrieve and acquire additional information through APIs or external storage.

- **Parametric Memory**
  - Fine-tuning: Expands the model's internal knowledge by injecting new information or knowledge into the LLM.
  - Knowledge Editing: Modifies or updates existing knowledge at the model level, achieving dynamic adjustment of the model's internal parametric memory.

- **Environment**
  - Represents the entities and context involved when the Agent interacts with users and external systems, such as the user Alice, or accessible tools or interfaces (e.g., a ticket booking system, a streaming platform).

- **Agent**
  - The LLM Agent is responsible for read and write operations, i.e., reading information from the external environment or knowledge base and writing new actions or content.
  - It also includes a series of management functions, such as merging, reflecting, and forgetting, to dynamically maintain short-term and long-term memory.

Another example is an Agent completing two different but related tasks, requiring the use of both short-term and long-term memory:

- **Task A: Play a video**: The Agent records the current plan, actions, and environmental state (e.g., searching, clicking, playing the video) in its short-term memory. This information is stored in memory and the LLM's context window.
- **Task B: Download a game**: The Agent utilizes its long-term memory related to Arcane and League of Legends to quickly find a way to download the game. The figure shows the Agent searching on Google; we can consider Google's knowledge base as an external knowledge source. All new search, click, and download actions are also updated in the short-term memory.

{{< figure
    src="gui_agent_memory_illustration.png"
    caption="Fig. 28: Illustration of short-term memory and long-term memory in an LLM-brained GUI agent. (Image source: [Zhang et al. 2024](https://arxiv.org/abs/2411.18279))"
    align="center"
    width="100%"
>}}

Common memory elements and their corresponding storage methods can be summarized in the following table:

| **Memory Element**      | **Memory Type**   | **Description**                                                              | **Storage Medium / Method**      |
|-------------------------|-------------------|------------------------------------------------------------------------------|----------------------------------|
| Action                  | Short-Term Memory | Historical action trajectory (e.g., clicking buttons, entering text)           | Memory, LLM Context Window       |
| Plan                    | Short-Term Memory | The plan for the next operation generated in the previous or current step    | Memory, LLM Context Window       |
| Execution Result        | Short-Term Memory | The result returned after an action, error messages, and environmental feedback | Memory, LLM Context Window       |
| Environment State       | Short-Term Memory | Available buttons, page titles, system status, etc., in the current UI environment | Memory, LLM Context Window       |
| Self-Experience         | Long-Term Memory  | Historical task trajectories and execution steps                             | Database, Disk                   |
| Self-Guidance           | Long-Term Memory  | Guiding rules and best practices summarized from historical successful trajectories | Database, Disk                   |
| External Knowledge      | Long-Term Memory  | External knowledge bases, documents, or other data sources to assist in task completion | External Database, Vector Retrieval |
| Task Success Metrics    | Long-Term Memory  | Records of task success rates, failure rates, etc., for improvement and analysis | Database, Disk                   |

Furthermore, researchers have proposed new training and storage methods to enhance the memory capabilities of LLMs:

**LongMem (Language Models Augmented with Long-Term Memory)** ([Wang, et al. 2023](https://arxiv.org/abs/2306.07174)) enables LLMs to remember long historical information. It adopts a decoupled network architecture, freezing the original LLM parameters as a memory encoder, while using an Adaptive Residual Side-Network (SideNet) as a memory retriever for memory checking and reading.

{{< figure
    src="LongMem.png"
    caption="Fig. 29. Overview of the memory caching and retrieval flow of LongMem. (Image source: [Wang, et al. 2023](https://arxiv.org/abs/2306.07174))"
    align="center"
    width="100%"
>}}

It mainly consists of three parts: **Frozen LLM**, **Residual SideNet**, and **Cached Memory Bank**. Its workflow is as follows:

- First, a long text sequence is split into fixed-length segments. Each segment is encoded layer by layer in the Frozen LLM, and at the $m$-th layer, the attention's $K, V \in \mathbb{R}^{H \times M \times d}$ vector pairs are extracted and cached in the Cached Memory Bank.
- When a new input sequence is encountered, the model retrieves the top-$k$ most relevant key-value pairs from the long-term memory bank based on the current input's query-key. These are then integrated into the subsequent language generation process. Meanwhile, the memory bank removes the oldest content to ensure the availability of the latest contextual information.
- The Residual SideNet fuses the hidden layer outputs of the frozen LLM with the retrieved historical key-values during inference, enabling effective modeling and utilization of context from ultra-long texts.

Through this decoupled design, LongMem can flexibly schedule massive amounts of historical information without expanding its native context window, balancing both speed and long-term memory capabilities.

## Tool Use

**Tool use** is an important component of LLM Agents. By empowering LLMs with the ability to call external tools, their capabilities are significantly expanded: they can not only generate natural language but also obtain real-time information, perform complex calculations, and interact with various systems (such as databases, APIs, etc.), effectively breaking through the limitations of their pre-trained knowledge and avoiding the inefficiency of reinventing the wheel.

Traditional LLMs primarily rely on pre-trained data for text generation, which makes them deficient in areas like mathematical operations, data retrieval, and real-time information updates. Through tool calling, models can:

- **Enhance computational ability:** For example, by calling a specialized calculator tool like [Wolfram](https://gpt.wolfram.com/index.php.en), the model can perform more precise mathematical calculations, compensating for its own arithmetic shortcomings.
- **Obtain real-time information:** Using search engines like Google, Bing, or database APIs, the model can access the latest information, ensuring the timeliness and accuracy of the generated content.
- **Increase information credibility:** With the support of external tools, the model can cite real data sources, reducing the risk of fabricating information and improving overall credibility.
- **Improve system transparency:** Tracking API call records can help users understand the model's decision-making process, providing a degree of interpretability.

Currently, various LLM applications based on tool calling have emerged, utilizing different strategies and architectures to cover everything from simple tasks to complex multi-step reasoning.

### Toolformer

**Toolformer** ([Schick, et al. 2023](https://arxiv.org/abs/2302.04761)) is an LLM that can use external tools through simple APIs. It is trained by fine-tuning the GPT-J model, requiring only a few examples for each API. The tools Toolformer learns to call include a question-answering system, Wikipedia search, a calculator, a calendar, and a translation system:

{{< figure
    src="Toolformer_api.png"
    caption="Fig. 30. Examples of inputs and outputs for all APIs used. (Image source: [Schick, et al. 2023](https://arxiv.org/abs/2302.04761))"
    align="center"
    width="100%"
>}}

### HuggingGPT

**HuggingGPT** ([Shen, et al. 2023](https://arxiv.org/abs/2303.17580)) is a framework that uses ChatGPT as a task planner. It selects available models from [HuggingFace](https://huggingface.co/) by reading their descriptions to complete user tasks and summarizes the results based on their execution.

{{< figure
    src="HuggingGPT.png"
    caption="Fig. 31. Illustration of how HuggingGPT works. (Image source: [Shen, et al. 2023](https://arxiv.org/abs/2303.17580))"
    align="center"
    width="100%"
>}}

The system consists of the following four stages:

- **Task Planning**: Parses the user's request into multiple sub-tasks. Each task has four attributes: task type, ID, dependencies, and parameters. The paper uses few-shot prompting to guide the model in task decomposition and planning.
- **Model Selection**: Assigns each sub-task to different expert models, using a multiple-choice format to determine the most suitable model. Due to the limited context length, models need to be initially filtered based on the task type.
- **Task Execution**: The expert models execute their assigned specific tasks and record the results, which are then passed to the LLM for further processing.
- **Response Generation**: Receives the execution results from each expert model and finally outputs a summary answer to the user.

## LLM Agent Applications

### Generative Agent

The **Generative Agent** ([Park, et al. 2023](https://arxiv.org/abs/2304.03442)) experiment simulates realistic human behavior with 25 virtual characters driven by large language models in a sandbox environment. Its core design integrates memory, retrieval, reflection, and planning/reaction mechanisms, allowing agents to record and review their experiences and extract key information to guide future actions and interactions.

{{< figure
    src="generative_agent_sandbox.png"
    caption="Fig. 32. The screenshot of generative agent sandbox. (Image source: [Park, et al. 2023](https://arxiv.org/abs/2304.03442))"
    align="center"
    width="100%"
>}}

The entire system uses a long-term memory module to record all observed events, a retrieval model to extract information based on recency, importance, and relevance, and a reflection mechanism to generate high-level inferences, ultimately translating these outcomes into concrete actions. This simulation demonstrates emergent behaviors such as information diffusion, relationship memory, and social event coordination, providing a realistic simulation of human behavior for interactive applications.

{{< figure
    src="generative_agent_architecture.png"
    caption="Fig. 25. The generative agent architecture. ([Park, et al. 2023](https://arxiv.org/abs/2304.03442))"
    align="center"
    width="100%"
>}}

### WebVoyager

**WebVoyager** ([He et al. 2024](https://arxiv.org/abs/2401.13919)) is an autonomous web interaction agent based on large multimodal models, capable of controlling the mouse and keyboard for web browsing. WebVoyager uses the classic ReAct loop. In each interaction step, it views a browser screenshot annotated with a method similar to **SoM (Set-of-Marks)** ([Yang, et al. 2023](https://arxiv.org/abs/2310.11441)), which provides interaction cues by placing numerical labels on web elements, and then decides on the next action. This combination of visual annotation and the ReAct loop allows users to interact with web pages using natural language. For a concrete example, you can refer to the [WebVoyager code](https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/) using the LangGraph framework.

{{< figure
    src="WebVoyager.png"
    caption="Fig. 33. The overall workflow of WebVoyager. (Image source: [He et al. 2024](https://arxiv.org/abs/2401.13919))"
    align="center"
    width="100%"
>}}

### OpenAI Operator

**Operator** ([OpenAI, 2025](https://openai.com/index/introducing-operator/)) is an AI agent recently released by OpenAI, designed to autonomously execute web tasks. Operator can interact with web pages like a human user, completing specified tasks by typing, clicking, and scrolling. The core technology behind Operator is the **Computer-Using Agent (CUA)** ([OpenAI, 2025](https://openai.com/index/computer-using-agent/)). CUA combines the visual capabilities of GPT-4o with enhanced reasoning abilities gained through reinforcement learning, and it has been specially trained to interact with graphical user interfaces (GUIs), including buttons, menus, and text boxes that users see on the screen.

{{< figure
    src="cua_overview.png"
    caption="Fig. 34. Overview of OpenAI CUA. (Image source: [OpenAI, 2025](https://openai.com/index/computer-using-agent/))"
    align="center"
    width="100%"
>}}

CUA operates in an iterative loop consisting of three stages:

- **Perception**: CUA "observes" the web page content by capturing browser screenshots. This vision-based input allows it to understand the page's layout and elements.
- **Reasoning**: Using a chain-of-thought reasoning process, CUA evaluates the next action based on the current and previous screenshots and the actions already taken. This reasoning ability enables it to track task progress, review intermediate steps, and make adjustments as needed.
- **Action**: CUA interacts with the browser by simulating mouse and keyboard operations (such as clicking, typing, and scrolling). This allows it to perform a wide range of web tasks without needing specific API integrations.

The difference between CUA and the pre-existing WebVoyager is that CUA is an agent specifically trained with reinforcement learning, rather than a fixed-flow workflow built by directly calling GPT-4o. Although CUA is still in its early stages and has certain limitations, it has achieved state-of-the-art results on the following benchmarks.

{{< figure
    src="cua_benchmark.png"
    caption="Fig. 35. OpenAI CUA Benchmark Results. (Image source: [OpenAI, 2025](https://openai.com/index/computer-using-agent/))"
    align="center"
    width="100%"
>}}

## Deep Research

Deep Research is essentially a report generation system: given a user's query, the system uses an LLM as its core agent to generate a structured and detailed report through multiple rounds of iterative information retrieval and analysis. Currently, the implementation logic of various Deep Research systems can be mainly divided into two approaches: **Workflow Agent** and **RL Agent**.

### Workflow Agent vs RL Agent

The Workflow Agent approach relies on developers to pre-design workflows and manually craft prompts to organize the entire report generation process. Its main features include:

- **Task Decomposition and Flow Orchestration**: The system breaks down the user's query into several sub-tasks, such as generating an outline, information retrieval, and content summarization, and then executes them in a predetermined sequence.
- **Fixed Process**: The calls and interactions between different stages are pre-defined, similar to building a static flowchart or a directed acyclic graph (DAG), ensuring that each step has a clear responsibility.
- **Dependence on Manual Design**: This method heavily relies on the experience of engineers, who improve output quality through repeated prompt tuning. It is highly applicable but has limited flexibility.

The [LangGraph](https://langchain-ai.github.io/langgraph/) framework can be used to build and orchestrate workflows in the form of a graph.

{{< figure
    src="langgraph_workflow.png"
    caption="Fig. 36. A workflow of the LangGraph. (Image source: [LangGraph, 2025](https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=workflow))"
    align="center"
    width="100%"
>}}

The following table compares 5 common workflow and agent patterns:

| Pattern                 | Core Mechanism                               | Advantages                               | Limitations                          | Use Cases                               |
| ----------------------- | -------------------------------------------- | ---------------------------------------- | ------------------------------------ | --------------------------------------- |
| **Prompt Chaining**     | Sequentially calls LLMs, passing results step-by-step | Suitable for phased reasoning, more accurate results | Fixed process, high latency          | Document generation (outline → content), translation polishing |
| **Parallelization**     | Splits sub-tasks for parallel processing, or multi-model voting | Increases speed, more robust results     | Sub-tasks must be independent, high resource consumption | Parallel content moderation, multi-model code detection |
| **Routing**             | First classifies, then assigns to different models/processes | Highly targeted, improves efficiency     | Effectiveness depends on classification accuracy | Customer service query routing, dynamic model size selection |
| **Evaluator-Optimizer** | Generate → Evaluate → Optimize iteratively   | Improves result quality, suitable for tasks with standards | High cost, multiple iterations increase latency | Translation optimization, multi-round retrieval refinement |
| **Orchestrator-Worker** | Central orchestration, dynamically decomposes and schedules sub-tasks | Flexible, can handle complex tasks       | Complex architecture, high scheduling cost | Multi-file code modification, real-time research integration |
| **Agent**               | LLM makes autonomous decisions, calls tools based on environmental feedback | Highly flexible, adapts to dynamic environments | Unpredictable, cost and security need control | Autonomous research agents, interactive problem-solving |

Currently, there are several open-source projects on GitHub that have implemented workflow-based Deep Research Agents, such as [GPT Researcher](https://github.com/assafelovic/gpt-researcher) and [open deep research](https://github.com/langchain-ai/open_deep_research).

{{< figure
    src="open_deep_research.png"
    caption="Fig. 37. An overview of the open deep research. (Image source: [LangChain, 2025](https://github.com/langchain-ai/open_deep_research))"
    align="center"
    width="100%"
>}}

The RL Agent is an alternative approach that uses RL to train a reasoning model to optimize the agent's multi-round search, analysis, and report writing process. Its main features include:

- **Autonomous Decision-Making Capability**: The system is trained through reinforcement learning, allowing the agent to autonomously judge, make decisions, and adjust its strategy when facing complex search and content integration tasks, thereby generating reports more efficiently.
- **Continuous Optimization**: Using a reward mechanism to score and provide feedback on the generation process, the agent can continuously iterate and optimize its own policy, improving the overall quality from task decomposition to the final report.
- **Reduced Manual Intervention**: Compared to fixed processes that rely on manually crafted prompts, the reinforcement learning training approach reduces dependence on manual design, making it more suitable for handling variable and complex real-world application scenarios.

The table below summarizes the main differences between these two approaches:

| Feature                    | Workflow Agent                                               | RL Agent                                                     |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Process Design**         | Pre-designed fixed workflow with clear task decomposition and flow orchestration | End-to-end learning, with the agent making autonomous decisions and dynamically adjusting the process |
| **Autonomous Decision-Making** | Relies on manually designed prompts; the decision process is fixed and immutable | Through reinforcement learning, the agent can autonomously judge, decide, and optimize its strategy |
| **Manual Intervention**    | Requires extensive manual design and tuning of prompts; significant manual intervention | Reduces manual intervention, achieving automatic feedback and continuous optimization through a reward mechanism |
| **Flexibility & Adaptability** | Weaker adaptability to complex or changing scenarios; limited extensibility | Better suited for variable and complex real-world scenarios, with high flexibility |
| **Optimization Mechanism** | Optimization mainly relies on engineers' experience and adjustments; lacks an end-to-end feedback mechanism | Utilizes reward feedback from reinforcement learning for continuous, automated performance improvement |
| **Implementation Difficulty** | Relatively straightforward to implement, but requires tedious process design and maintenance | Requires training data and computational resources; higher initial development investment, but better long-term results |
| **Training Required**      | No additional training needed; relies solely on manually constructed processes and prompts | Requires training the agent through reinforcement learning to achieve autonomous decision-making |

### OpenAI Deep Research

**OpenAI Deep Research** ([OpenAI, 2025](https://openai.com/index/introducing-deep-research/)) is an intelligent agent officially released by OpenAI in February 2025. Designed for complex scenarios, it can automatically search, filter, analyze, and integrate multi-source information to ultimately generate high-quality comprehensive reports. The system is built on [o3](https://openai.com/index/openai-o3-mini/) as its core base model and incorporates reinforcement learning methods, significantly improving the accuracy and robustness of its multi-round iterative search and reasoning processes.

Compared to traditional ChatGPT plugin-based search or conventional RAG techniques, OpenAI Deep Research has the following outstanding advantages:

1. **Reinforcement Learning-Driven Iterative Reasoning**
   Leveraging the **o3 reasoning model** and reinforcement learning training strategies, the agent can continuously optimize its reasoning path during multi-round search and summarization, effectively reducing the risk of distortion caused by error accumulation.

2. **Multi-Source Information Integration and Cross-Validation**
   Breaking the limitations of a single search engine, it can simultaneously call upon various authoritative data sources such as specific databases and professional knowledge bases, forming more reliable research conclusions through cross-validation.

3. **High-Quality Report Generation**
   The training phase introduces an LLM-as-a-judge scoring mechanism and strict evaluation criteria, enabling the system to self-evaluate when outputting reports, thereby generating more clearly structured and rigorously argued professional texts.

#### Training Process

The training process for OpenAI Deep Research utilized a **browser interaction dataset** specifically tailored for research scenarios. Through these datasets, the model mastered core browsing functions—including searching, clicking, scrolling, and file parsing—and also learned to use Python tools in a sandboxed environment for computation, data analysis, and visualization. Furthermore, with reinforcement learning training on these browsing tasks, the model can efficiently perform information retrieval, integration, and reasoning across a vast number of websites, quickly locating key information or generating comprehensive research reports.

These training datasets include both objective tasks with ground-truth answers that can be automatically scored, as well as open-ended tasks equipped with detailed scoring rubrics. During training, the model's responses are rigorously compared against the ground truth or scoring criteria, and the model generates CoT thought processes to allow an evaluation model to provide feedback.

Additionally, the training process reused the safety datasets accumulated during the o1 model's training phase and was specifically supplemented with safety training data for Deep Research scenarios, ensuring that the model strictly adheres to relevant compliance and safety requirements during automated search and browsing.

#### Performance

The model achieved state-of-the-art results on the **Humanity's Last Exam** benchmark ([Phan, et al. 2025](https://arxiv.org/abs/2501.14249)), which evaluates AI's ability to answer expert-level questions across various professional domains.

{{< figure
    src="human_last_exam.png"
    caption="Fig. 38. Humanity's Last Exam Benchmark Results. (Image source: [OpenAI, 2025](https://openai.com/index/introducing-deep-research/))"
    align="center"
    width="80%"
>}}

## Future Directions

Intelligent agents show vast promise, but to achieve reliable and widespread application, the following key challenges still need to be addressed:

- **Context Window Limitations**: The limited context window of LLMs restricts the amount of information they can process, affecting long-term planning and memory capabilities and reducing task coherence. Current research is exploring external memory mechanisms and context compression techniques to enhance long-term memory and complex information processing abilities. Currently, OpenAI's latest model, **GPT-4.5** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-4-5/)), has a maximum context window of 128k tokens.

- **Interface Standardization and Interoperability**: The current natural language-based interaction with tools suffers from inconsistent formatting. The **Model Context Protocol (MCP)** ([Anthropic, 2024](https://www.anthropic.com/news/model-context-protocol)) aims to unify the interaction between LLMs and applications through an open standard, reducing development complexity and improving system stability and cross-platform compatibility.

- **Task Planning and Decomposition Capabilities**: Agents struggle to formulate coherent plans for complex tasks, effectively decompose sub-tasks, and lack the ability to dynamically adjust in unexpected situations. More powerful planning algorithms, self-reflection mechanisms, and dynamic policy adjustment methods are needed to flexibly respond to uncertain environments.

- **Computational Resources and Economic Viability**: Deploying large model agents is costly due to multiple API calls and intensive computation, limiting their use in some practical scenarios. Optimization directions include more efficient model architectures, quantization techniques, inference optimization, caching strategies, and intelligent scheduling mechanisms. With the development of specialized GPU hardware like the [NVIDIA DGX B200](https://www.nvidia.com/en-sg/data-center/dgx-b200/) and distributed technologies, computational efficiency is expected to improve significantly.

- **Security and Privacy Protection**: Agents face security risks such as prompt injection and need robust authentication, permission control, input validation, and sandboxed environments. For multimodal inputs and external tools, data anonymization, the principle of least privilege, and audit logs must be strengthened to meet security and privacy compliance requirements.

- **Decision Transparency and Explainability**: The difficulty in explaining agent decisions limits their application in high-stakes domains. Enhancing explainability requires the development of visualization tools, chain-of-thought tracking, and decision rationale generation mechanisms to improve decision transparency, build user trust, and meet regulatory requirements.

## References

[1] DAIR.AI. ["LLM Agents."](https://www.promptingguide.ai/research/llm-agents) Prompt Engineering Guide, 2024.

[2] Sutton, Richard S., and Andrew G. Barto. ["Reinforcement Learning: An Introduction."](http://incompleteideas.net/book/the-book.html) MIT Press, 2018.

[3] Weng, Lilian. ["LLM-powered Autonomous Agents."](https://lilianweng.github.io/posts/2023-06-23-agent/) Lil’Log, 2023.

[4] Zhou, Yongchao, et al. ["Large language models are human-level prompt engineers."](https://arxiv.org/abs/2211.01910) The eleventh international conference on learning representations. 2022.

[5] Zhang, Zhuosheng, et al. ["Automatic chain of thought prompting in large language models."](https://arxiv.org/abs/2210.03493) arXiv preprint arXiv:2210.03493 (2022).

[6] Liu, Jiacheng, et al. ["Generated knowledge prompting for commonsense reasoning."](https://arxiv.org/abs/2110.08387) arXiv preprint arXiv:2110.08387 (2021).

[7] Lewis, Patrick, et al. ["Retrieval-augmented generation for knowledge-intensive nlp tasks."](https://arxiv.org/abs/2005.11401) Advances in neural information processing systems 33 (2020): 9459-9474.

[8] Zhang, Zhuosheng, et al. ["Multimodal chain-of-thought reasoning in language models."](https://arxiv.org/abs/2302.00923) arXiv preprint arXiv:2302.00923 (2023).

[9] Diao, Shizhe, et al. ["Active prompting with chain-of-thought for large language models."](https://arxiv.org/abs/2302.12246) arXiv preprint arXiv:2302.12246 (2023).

[10] Wei, Jason, et al. ["Chain-of-thought prompting elicits reasoning in large language models."](https://arxiv.org/abs/2201.11903) Advances in neural information processing systems 35 (2022): 24824-24837. 

[11] Kojima, Takeshi, et al. ["Large language models are zero-shot reasoners."](https://arxiv.org/abs/2205.11916) Advances in neural information processing systems 35 (2022): 22199-22213.

[12] Wang, Xuezhi, et al. ["Self-consistency improves chain of thought reasoning in language models."](https://arxiv.org/abs/2203.11171)  arXiv preprint arXiv:2203.11171 (2022).

[13] Wang, Xuezhi, et al. ["Rationale-augmented ensembles in language models."](https://arxiv.org/abs/2207.00747) arXiv preprint arXiv:2207.00747 (2022).

[14] Zelikman, Eric, et al. ["Star: Bootstrapping reasoning with reasoning."](https://arxiv.org/abs/2203.14465) Advances in Neural Information Processing Systems 35 (2022): 15476-15488.

[15] Fu, Yao, et al. ["Complexity-based prompting for multi-step reasoning."](https://arxiv.org/abs/2210.00720) arXiv preprint arXiv:2210.00720 (2022).

[16] Yao, Shunyu, et al. ["Tree of thoughts: Deliberate problem solving with large language models."](https://arxiv.org/abs/2305.10601) Advances in neural information processing systems 36 (2023): 11809-11822.

[17] Yao, Shunyu, et al. ["React: Synergizing reasoning and acting in language models."](https://arxiv.org/abs/2210.03629) International Conference on Learning Representations (ICLR). 2023.

[18] Shinn, Noah, et al. ["Reflexion: Language agents with verbal reinforcement learning."](https://arxiv.org/abs/2303.11366) Advances in Neural Information Processing Systems 36 (2023): 8634-8652.

[19] Guo, Daya, et al. ["Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning."](https://arxiv.org/abs/2501.12948) arXiv preprint arXiv:2501.12948 (2025).

[20] OpenAI. ["Introducing OpenAI o1"](https://openai.com/o1/) OpenAI, 2024.

[21] Zhang, Zeyu, et al. ["A survey on the memory mechanism of large language model based agents."](https://arxiv.org/abs/2404.13501) arXiv preprint arXiv:2404.13501 (2024).

[22] Zhang, Chaoyun, et al. ["Large language model-brained gui agents: A survey."](https://arxiv.org/abs/2411.18279) arXiv preprint arXiv:2411.18279 (2024).

[23] Wang, Weizhi, et al. ["Augmenting language models with long-term memory."](https://arxiv.org/abs/2306.07174) Advances in Neural Information Processing Systems 36 (2023): 74530-74543.

[24] Schick, Timo, et al. ["Toolformer: Language models can teach themselves to use tools."](https://arxiv.org/abs/2302.04761) Advances in Neural Information Processing Systems 36 (2023): 68539-68551.

[25] Shen, Yongliang, et al. ["Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face."](https://arxiv.org/abs/2303.17580) Advances in Neural Information Processing Systems 36 (2023): 38154-38180.

[26] Park, Joon Sung, et al. ["Generative agents: Interactive simulacra of human behavior."](https://arxiv.org/abs/2304.03442) Proceedings of the 36th annual acm symposium on user interface software and technology. 2023.

[27] He, Hongliang, et al. ["WebVoyager: Building an end-to-end web agent with large multimodal models."](https://arxiv.org/abs/2401.13919) arXiv preprint arXiv:2401.13919 (2024).

[28] Yang, Jianwei, et al. ["Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v."](https://arxiv.org/abs/2310.11441) arXiv preprint arXiv:2310.11441 (2023).

[29] OpenAI. ["Introducing Operator."](https://openai.com/index/introducing-operator/) OpenAI, 2025.

[30] OpenAI. ["Computer-Using Agent."](https://openai.com/index/computer-using-agent/) OpenAI, 2025.

[31] OpenAI. ["Introducing Deep Research."](https://openai.com/index/introducing-deep-research/) OpenAI, 2025.

[32] Phan, Long, et al. ["Humanity's Last Exam."](https://arxiv.org/abs/2501.14249) arXiv preprint arXiv:2501.14249 (2025).

[33] OpenAI. ["Introducing GPT-4.5."](https://openai.com/index/introducing-gpt-4-5/) OpenAI, 2025.

[34] Anthropic. ["Introducing the Model Context Protocol."](https://www.anthropic.com/news/model-context-protocol) Anthropic, 2024.

[35] LangGraph. ["A workflow of the LangGraph."](https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=workflow) LangGraph Tutorials, 2025.

[36] Assaf Elovic. ["GPT Researcher"](https://github.com/assafelovic/gpt-researcher) GitHub Repository, 2025.

[37] LangChain. ["Open Deep Research"](https://github.com/langchain-ai/open_deep_research) GitHub Repository, 2025.

## Citation

> **Citation**: When reprinting or citing the content of this article, please indicate the original author and source.

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