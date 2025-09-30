---
title: "Agentic RL"
date: 2025-09-30T12:00:00+08:00
author: "Yue Shui"
tags: ["Agentic RL", "Reinforcement Learning", "LLM", "Agent", "SWE-bench", "verl", "ReTool"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

As Large Language Models (LLMs) achieve breakthroughs in natural language processing, their applications continue to expand. However, they also exhibit limitations such as knowledge cutoffs, hallucinations, and deficiencies in complex computation and logical reasoning. To address these challenges, **Agentic RL**, which combines agents with Reinforcement Learning (RL), is emerging as a key research direction.

Agentic RL enables LLMs to possess capabilities like autonomous planning, decision-making, tool use, and environmental interaction by creating a closed-loop interaction with the external world (e.g., search engines, code interpreters, databases, browsers) and continuously optimizing through reward signals. In practical applications, it not only understands requirements and plans autonomously but also constantly corrects and optimizes within an execution-feedback loop.

Its core value is primarily reflected in two aspects:

- **Reducing Prompt Dependency**: It frees the model from over-reliance on prompts, equipping it with adaptive problem-solving abilities.
- **Enhancing Autonomous Exploration**: Through multi-turn reinforcement learning, it improves exploration and reasoning capabilities, thereby compensating for the shortcomings of sparse or repetitive static data distributions.

## Agentic RL vs. LLM-RL

{{< figure
    src="agentic_rl_survey.png"
    caption="Fig. 1. Paradigm shift from LLM-RL to Agentic RL. (Image source: [Zhang et al., 2025](https://arxiv.org/abs/2509.02547))"
    align="center"
    width="100%"
>}}

Alignment-focused **LLM-RL**, represented by RLHF, is often approximated in practice as a single-step (sequence-level) decision-making [Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process). In contrast, **Agentic RL** operates in partially observable environments, involving multi-step, long-horizon sequential decision-making, which is better characterized by a [Partially Observable Markov Decision Process (POMDP)](https://en.wikipedia.org/wiki/Partially_observable_markov_decision_process). The table below summarizes the differences between the two.

| Feature | Traditional LLM-RL (e.g., RLHF) | Agentic RL |
| :--- | :--- | :--- |
| **Decision Process** | **Degenerate single-step MDP**: Input prompt → Output complete response → One-time reward, akin to a "single-turn mapping." | **Multi-step, long-horizon POMDP**: Continuous interaction in a partially observable environment, where each step updates the state and receives feedback. |
| **State Space** \(\mathcal{S}\) | Static, determined solely by the input prompt, and does not evolve during the process. | Dynamic, includes interaction history, tool outputs, external environment states, etc., and is constantly updated through interaction. |
| **Action Space** \(\mathcal{A}\) | Single action: generating a text sequence (response). | Composite actions: generating a chain of thought (Thought), calling a tool (Tool Call), updating the state, and generating the final answer. |
| **Reward** \(\mathcal{R}\) | **Sparse Outcome Reward**: Typically given after generation is complete, based on human preference or a model judge. | **Hybrid Reward Mechanism**: Includes both sparse Outcome Rewards and dense Process Rewards (e.g., success/failure of tool calls, sub-task completion). |
| **Core Challenges** | Aligning with human preferences, ensuring safety and usefulness; improving overall generation quality. | Long-horizon credit assignment, complex task planning, exploration efficiency, robust tool use, and balancing exploration-exploitation. |

## Evaluation

Scientific, comprehensive, and realistic evaluation benchmarks are crucial for measuring and enhancing the capabilities of LLM Agents. **"Successful language model evals"** ([Wei, 2024](https://www.jasonwei.net/blog/evals)) summarizes several key traits of successful evaluation benchmarks, which collectively determine whether an eval can be widely accepted by the community and stand the test of time:

1.  **Sufficient Sample Size**: An eval needs enough examples (at least 1,000 is recommended) to reduce random fluctuations in results. Too few samples can cause scores to fluctuate wildly between model checkpoints, making it painful for researchers to track real performance changes.
2.  **High-Quality Data**: The data in the eval (questions, answers, test cases, etc.) must be accurate. If the eval itself contains many errors, researchers will lose trust in it, especially when a powerful model like GPT-4 disagrees with the ground truth.
3.  **Simple Single-Number Metric**: A successful eval must have a core, easily understandable single-number metric (e.g., accuracy). Overly complex evaluation systems, like the first version of **HELM** ([Liang et al., 2022](https://arxiv.org/abs/2211.09110)), can be too comprehensive, making it hard for researchers to focus and hindering quick comparisons and dissemination.
4.  **Easy to Run and Reproduce**: The evaluation process should be as simple and efficient as possible. If running an eval requires complex setup and long wait times, like some subsets of **BIG-Bench** ([Srivastava et al., 2022](https://arxiv.org/abs/2206.04615)), it will significantly impede its adoption.
5.  **Meaningful Task**: The eval's task should be central to intelligence, such as language understanding, math reasoning, or code generation. Tasks that are challenging but not meaningful (e.g., closing parentheses properly) don't allow for substantive conclusions about a model's intelligence, even if it performs well.
6.  **Accurate and Reliable Grading**: The automated grading script must be extremely robust and correct. If researchers find that their model's correct output is graded incorrectly, they will quickly write off the eval.
7.  **Avoids Premature Saturation**: The eval should be difficult enough to ensure that model performance has room to grow over time. Benchmarks like **GLUE** ([Wang et al., 2018](https://arxiv.org/abs/1804.07461)) and **SuperGLUE** ([Wang et al., 2019](https://arxiv.org/abs/1905.00537)) became saturated too quickly, losing their utility as a measure of progress.

The concept of **Asymmetry of verification** ([Wei, 2024](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law)) points out that for many tasks, **it is much easier to verify a solution than to solve it from scratch**.

{{< figure
    src="asymmetry_verification.png"
    caption="Fig. 2. Improving Verification with Privileged Information. (Image source: [Wei, 2024](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law))"
    align="center"
    width="100%"
>}}

For example, solving a complex Sudoku puzzle can take hours, but checking a completed grid takes only minutes. Writing the backend code for a large website like Instagram takes a team of engineers years, but any user can quickly verify if the site is working. Similarly, for many information retrieval or open-ended tasks, generating an answer may require extensive trial and error, but verifying a candidate solution against constraints is often much faster.

We can significantly reduce the cost of verification by preparing prior information or verification mechanisms. For instance:
- **SWE-bench**: Verifying code correctness would normally require manual line-by-line review. However, with a pre-existing suite of test cases with ample coverage, one can simply run the tests to determine if a model-generated patch is valid in seconds.
- **AIME Math Competition**: The derivation process for math problems is often complex and time-consuming, but once the official answer is released, any proposed solution can be verified against it in seconds.

This asymmetry is crucial for AI training because it directly relates to the feasibility of creating an RL environment or a reward model. From this, Jason Wei proposed the **Verifier's Rule**:

> *The ease of training AI to solve a task is proportional to how verifiable the task is. All tasks that are possible to solve and easy to verify will be solved by AI.*

A task's verifiability typically depends on whether it satisfies these five key properties:

1. **Objective truth**: Everyone agrees on what constitutes a good solution (e.g., the unique correct answer to a math problem).
2. **Fast to verify**: A single solution can be checked in seconds (e.g., running a set of test cases).
3. **Scalable to verify**: Many candidate solutions can be verified in parallel (e.g., batch-running code tests).
4. **Low noise**: The verification signal is highly correlated with solution quality, with a low rate of false positives/negatives.
5. **Continuous reward**: It's possible to rank the quality of multiple solutions, not just determine correctness, which provides a smoother optimization signal.

This rule explains why scenarios like the **SWE-bench programming task** and **AIME math problem-solving** have become ideal testbeds for AI capabilities. They naturally meet most of the above criteria, allowing us to efficiently build automated evaluation systems and continuously optimize model performance through large-scale "generate-and-verify" loops.

### SWE-bench

{{< figure
    src="swe_bench.png"
    caption="Fig. 3. SWE-bench links real GitHub issues with their merged pull request fixes. Given an issue description and a codebase snapshot, models generate a patch that is tested against actual project test cases. (Image source: [Jimenez et al., 2024](https://arxiv.org/abs/2310.06770))"
    align="center"
    width="100%"
>}}

**SWE-bench** ([Jimenez et al., 2024](https://arxiv.org/abs/2310.06770)) collects 2,294 real-world development tasks from 12 popular open-source Python projects, sourced directly from GitHub Issues and Pull Requests.

To ensure reproducibility and environment independence, SWE-bench constructs an **isolated Docker environment** for each task, preventing failures due to inconsistencies in Python versions or dependencies. This design also forces the model to learn to generate compatible patches for different environments.

For verification, SWE-bench cleverly utilizes the projects' built-in unit tests to **automatically evaluate the correctness of the LLM's patch**. It includes two types of tests:

- **Fail-to-Pass (F2P)**: Tests that initially fail but should pass after the correct PR is merged, confirming that the LLM has fixed the target issue.
- **Pass-to-Pass (P2P)**: Tests that initially pass and must continue to pass after the PR is merged, ensuring the LLM has not broken existing functionality or introduced new bugs.

This combination of "real tasks + isolated environments + automated testing" makes SWE-bench a high-fidelity, scalable benchmark that significantly reduces the cost of verifying programming task correctness. However, the original SWE-bench had flaws like unfair test cases, ambiguous problem descriptions, and complex environments, which led to underestimation of model capabilities. Consequently, OpenAI created a high-quality, human-curated subset called **SWE-bench Verified** ([OpenAI, 2024](https://openai.com/index/introducing-swe-bench-verified/)) for more accurate model evaluation.

### BrowseComp

Unlike software engineering tasks, web browsing tasks aim to find specific information within the vast expanse of the internet. **BrowseComp** ([Wei et al., 2025](https://arxiv.org/abs/2504.12516)) is a simple yet challenging benchmark designed for such tasks.

*   **Design Philosophy**: BrowseComp follows the principle of being **hard to solve, easy to verify**. The problems are designed to require persistent and creative browsing of numerous web pages to find the answer, but the answer itself is typically a short, indisputable string that can be easily compared against a reference.

*   **Data Construction**: The creators use a **reverse questioning** method. They first find an obscure fact (e.g., a specific conference paper) and then construct a query with multiple complex constraints around it. For example: "Find a paper published at EMNLP between 2018-2023 whose first author graduated from Dartmouth College and whose fourth author graduated from the University of Pennsylvania." Verifying this answer is simple, but finding it among thousands of papers is extremely difficult.

{{< figure
    src="BrowseComp_scale.png"
    caption="Fig. 4. BrowseComp performance of an early version of OpenAI Deep Research scales smoothly with test-time compute. (Image source: [Wei et al., 2025](https://arxiv.org/abs/2504.12516))"
    align="center"
    width="60%"
>}}

BrowseComp measures an agent's core browsing abilities: factual reasoning, persistent navigation, and creative search. As shown in the figure, the performance of a powerful browsing agent (like OpenAI Deep Research) on this benchmark scales smoothly with test-time compute (i.e., browsing effort), indicating that the eval effectively measures an agent's deep search and information integration capabilities.

## Data

High-quality data is the cornerstone of training powerful agents. However, manually annotating the complete decision trajectories of agents in complex tasks is extremely costly and difficult to scale. Therefore, **Synthetic Data** has become the mainstream solution in this field. Various innovative data generation pipelines have been proposed to create a "data flywheel" that continuously produces high-quality training data.

### AgentFounder

{{< figure
    src="tongyi_agentic_training_pipeline.png"
    caption="Fig. 5. The Agentic Training Pipeline proposed by AgentFounder, incorporating an Agentic CPT stage. (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

**AgentFounder** ([Su et al., 2025](https://arxiv.org/abs/2509.13310)) introduces a new stage between traditional pre-training and post-training called **Agentic Continual Pre-training (Agentic CPT)**. The entire training pipeline consists of three stages:

1. **General Pre-training**: Consistent with standard procedures, a base model with general knowledge is first trained.
2. **Agentic Continual Pre-training (Agentic CPT)**: On top of the general base model, large-scale, diverse synthetic agent behavior data is used for continued "next-word prediction" training. The goal is not to solve specific tasks but to have the model internalize general agent behavior patterns and develop agentic capabilities.
3. **Post-training/Task Fine-tuning**: On the model that already possesses basic agentic capabilities, SFT or RL is performed to align it with specific tasks. This avoids the optimization conflict that arises when trying to learn capabilities and align with tasks simultaneously during the post-training phase.

The key to Agentic CPT is how to **synthesize large-scale agent-like data at low cost**. To this end, AgentFounder proposes two efficient data generation methods that do not require external tool calls: **First-order Action Synthesis (FAS)** and **Higher-order Action Synthesis (HAS)**.

1. The core idea of FAS is to generate data about **how to think** and **how to plan the first step** at a low cost through deduction.

{{< figure
    src="tongyi_fas.png"
    caption="Fig. 6. Illustration of First-order Action Synthesis (FAS). (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

- **Planning Action**: Given a problem, have the LLM generate multiple possible analyses and initial tool call plans to help the model learn task decomposition and preliminary planning.
- **Reasoning Action**: Given a problem and relevant knowledge snippets, have the LLM generate a complete logical reasoning chain to arrive at the final answer, thereby training its logical deduction and information synthesis abilities.

FAS only generates thought processes and planned actions, without involving actual tool calls or environmental interactions. This makes its generation cost extremely low, making it suitable for **large-scale data synthesis (up to hundreds of millions of examples)**.

2. The goal of HAS is to transform existing (even suboptimal) agent trajectories into high-value **decision-making learning data**.

{{< figure
    src="tongyi_has.png"
    caption="Fig. 7. Illustration of Higher-order Action Synthesis (HAS). (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

- **Step-level Expansion**: For any step in a trajectory, use an LLM to generate multiple alternative actions, creating a local decision space.
- **Contrastive Learning**: Reframe the original choice and the expanded candidate actions as a multiple-choice question with feedback, requiring the model to identify the better decision.
- **Causal Supervision**: Append the final outcome (success or failure) to the end of the trajectory to help the model learn the **causal link between decisions and outcomes**.

This method upgrades traditional **Imitation Learning** to **step-level Decision Making**. The model not only "walks through a successful path" but also understands **how to choose at each critical juncture**, improving the signal-to-noise ratio and utilization efficiency of the data.

Does Agentic CPT truly alleviate the "optimization conflict"? The paper provides an answer through training loss curves and performance comparison experiments.

{{< figure
    src="agent_founder_loss.png"
    caption="Fig. 8. Training loss evolution showing superior convergence of AgentFounder models compared to baseline. (Image source: [Su et al., 2025](https://arxiv.org/abs/2509.13310))"
    align="center"
    width="100%"
>}}

The experimental results show that models that underwent Agentic CPT converge faster during the SFT phase, with training losses significantly lower than baseline models that did not go through this stage. This indicates that the model has already developed certain agentic capabilities before entering the post-training phase, making it more efficient when learning specific tasks later on.

### WebShaper

Traditional data synthesis methods are typically information-driven: information is first crawled from the web, and then questions are generated based on that information. **WebShaper** ([Tao et al., 2025](https://arxiv.org/abs/2507.15061)) points out that this approach can lead to a mismatch between the reasoning structure in the generated questions and the structure of the original information, allowing models to find answers through "logical shortcuts" rather than genuine multi-step reasoning. To address this, it proposes a **Formalism-Driven** paradigm, primarily using knowledge graphs and set theory to formalize problems.

{{< figure
    src="WebShaper.png"
    caption="Fig. 9. The formalism-driven data synthesis pipeline of WebShaper. (Image source: [Tao et al., 2025](https://arxiv.org/abs/2507.15061))"
    align="center"
    width="100%"
>}}

*   **Knowledge Projections (KP)**: WebShaper first formalizes information-seeking tasks based on set theory. A **Knowledge Projection** \(R(V)\) is defined as the set of all entities that have a relation \(R\) with the entity set \(V\). For example, `bornIn({1990s})` represents the set of all entities born in the 1990s.
*   **Task Formalization**: Complex queries can be rigorously represented as **Intersection** and **Union** operations of multiple KPs. For example, the query "Find a player who played for an East German team founded in 1966 during the 2004-05 season and was born in the 90s" can be formalized as the intersection of multiple KPs.
*   **Expander Agent**: WebShaper uses an agent called **Expander**, which first generates a formalized query structure (e.g., the intersection of three KPs) and then progressively populates this structure with specific content by calling tools (search, summarization). It uses a "hierarchical expansion" strategy to gradually increase problem complexity, effectively avoiding logical shortcuts and information redundancy.

## Reward Design

The reward function is the soul of RL, defining the agent's optimization objective.

*   **Verifiable Rewards**: For tasks with clear answers, such as mathematics and coding, this is the most reliable and scalable source of rewards. The reward signal can come directly from **unit test pass rates**, **code compiler feedback**, or the **correctness of the final answer**. This rule-based reward effectively avoids the reward hacking problem that reward models might introduce.

*   **Generative Rewards**: For open-ended tasks without a single correct answer (e.g., generating a research report), the **LLM-as-a-Judge** ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)) approach uses a powerful LLM as a judge to evaluate the quality of the generated output and provide a score or natural language feedback as the reward signal.

*   **Dense Rewards**: Unlike an **Outcome Reward Model (ORM)**, which provides a one-time reward only at the end of a task, a **Process Reward Model (PRM)** provides feedback for each step or intermediate stage of the agent's process. This helps solve the credit assignment problem in long-horizon tasks but also increases annotation costs and the risk of being exploited by the model.

*   **Unsupervised Rewards**: To eliminate reliance on external annotations, researchers have explored methods for extracting reward signals from the model's own behavior, such as constructing rewards based on **output consistency** (whether multiple generations are consistent) or **internal confidence** (e.g., the entropy of generation probabilities).

## Optimization Algorithms

In recent years, numerous improved algorithms have been developed based on methods like PPO, DPO, and GRPO. Below are three representative examples.

### PPO

**Proximal Policy Optimization (PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is a classic Actor-Critic algorithm that has become a mainstream method for RL fine-tuning of LLMs due to its successful application in **InstructGPT** ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)). The core idea of PPO is to limit the magnitude of change between the new and old policies during updates, thereby ensuring training stability. It uses a **token-level importance ratio** and **clipping** to constrain policy shifts and employs a Critic model to estimate the advantage (decomposing sequence-level rewards to the token-level), which improves stability but introduces additional model and computational overhead.

$$
\mathcal{J}_{\mathrm{PPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{|y|} \sum_{t=1}^{|y|} \min \left(w_t(\theta) \widehat{A}_t, \operatorname{clip}\left(w_t(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_t\right)\right]
$$

where the importance ratio of the token $y_{t}$ is defined as $w_t(\theta)=\frac{\pi_\theta\left(y_t \mid x, y_{<t}\right)}{\pi_{\theta_{\text {old }}}\left(y_t \mid x, y_{<t}\right)}$, the advantage of $y_{t}$, denoted by $\widehat{A}_{t}$, is estimated by another value model, and $\varepsilon$ is the clipping range of importance ratios.

### GRPO

**Group Relative Policy Optimization (GRPO)** ([Shao, et al. 2024](https://arxiv.org/abs/2402.03300)) cleverly removes the Critic model. For each problem, it samples a group of $G$ outputs and calculates the **relative advantage** of each output within the group (i.e., the reward value minus the group mean, divided by the standard deviation) to serve as the advantage function. The advantage is at the sequence-level but is still used for token-level updates. This reduces computational costs and improves training stability. The formula below omits the KL divergence penalty term; for the complete version, refer to the my previous post on [GRPO](https://syhya.github.io/posts/2025-01-27-deepseek-r1/#grpo).

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D},\left\{y_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \min \left(w_{i, t}(\theta) \widehat{A}_{i, t}, \operatorname{clip}\left(w_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_{i, t}\right)\right]
$$

The importance ratio and advantage for token $y_{i, t}$ are:

$$
w_{i, t}(\theta)=\frac{\pi_{\theta}\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}{\pi_{\theta_{\text{old}}}\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}, \quad \widehat{A}_{i, t}=\widehat{A}_{i}=\frac{r\left(x, y_{i}\right)-\operatorname{mean}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)}{\operatorname{std}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)},
$$

All tokens within sequence $y_{i}$ share the same advantage $\widehat{A}_{i}$, and $G$ is the number of outputs generated for each query $x$ (i.e., the group size).

### GSPO

**Group Sequence Policy Optimization (GSPO)** ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)) elevates the basic unit of optimization from the token level to the sequence level. Unlike GRPO, which uses token-level importance ratios, GSPO introduces a **sequence-level** importance ratio to align with sequence-level rewards. This avoids the noise accumulated from token ratios in long sequences, reducing variance and improving stability. The Qwen team notes that this not only mitigates large probability fluctuations caused by local routing jitter in MoE models but also naturally aligns with the **sequence-level rewards** common in Agent tasks, making it suitable for long-sequence modeling and routing-sensitive scenarios.

The objective function for GSPO is:

$$
\mathcal{J}_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D},\left\{y_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{\text {old }}}(\cdot \mid x)}\left[\frac{1}{G} \sum_{i=1}^{G} \min \left(s_{i}(\theta) \widehat{A}_{i}, \operatorname{clip}\left(s_{i}(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_{i}\right)\right],
$$

where the advantage function within the group is defined as:

$$
\widehat{A}_{i}=\frac{r\left(x, y_{i}\right)-\operatorname{mean}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)}{\operatorname{std}\left(\left\{r\left(x, y_{i}\right)\right\}_{i=1}^{G}\right)}
$$

And the sequence-level importance ratio $s_i(\theta)$ is defined as:

$$
s_i(\theta)=\left(\frac{\pi_\theta\left(y_i \mid x\right)}{\pi_{\theta_{\text{old}}}\left(y_i \mid x\right)}\right)^{\frac{1}{\left|y_i\right|}}=\exp \left(\frac{1}{\left|y_i\right|} \sum_{t=1}^{\left|y_i\right|} \log \frac{\pi_\theta\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}{\pi_{\theta_{\text{old}}}\left(y_{i, t} \mid x, y_{i,&lt;t}\right)}\right)
$$

It applies clipping to the **entire sequence** rather than individual tokens, keeping the optimization consistent with the sequence-level reward. **Length normalization** is used here to reduce variance and control the numerical range of $s_i(\theta)$, as otherwise, probability changes in a few tokens could cause the ratio to fluctuate dramatically, and different response lengths would lead to inconsistent clipping ranges. It's important to note that because the importance ratio is defined differently, the magnitude of the clipping range in GSPO is typically not the same as in GRPO.

### Frameworks

The training pipeline for Agentic RL is complex, involving multiple stages such as inference (Rollout), training (Training), and reward calculation. It typically requires a distributed framework for efficient coordination.

{{< figure
    src="dataflow_rlhf_verl.png"
    caption="Fig. 10. Dataflow graph of 3 RLHF algorithms including PPO, Safe-RLHF and ReMax. (Image source: [Sheng et al., 2024](https://arxiv.org/abs/2409.19256v2))"
    align="center"
    width="100%"
>}}

The figure above shows the complex dataflow during the training process of different algorithms. It involves not only multiple models like Actor, Critic, Reference, and Reward but also intertwines different types of computational workloads such as data generation, inference, and training. Taking the basic **PPO algorithm** as an example, the system involves 4 core models: the Actor generates responses based on the input prompt; the Critic evaluates the results; the Reference serves as a baseline for generation quality; and the Reward provides the reward signal. From a computational perspective, the entire process can be divided into 3 stages:

1. **Generation**: The Actor generates the response token by token. This process is affected by text length and generation method and is the main consumer of inference resources and time.
2. **Forward (Rollout)**: The generated result, along with the query, is fed into the 4 models for a forward pass, and the data is stored in a [Replay Buffer](https://docs.ray.io/en/latest/rllib/rllib-replay-buffers.html).
3. **Training**: Data is sampled from the Buffer to update the Actor and Critic.

{{< figure
    src="disaggregated_colocated_arch.png"
    caption="Fig. 11. Two representative RL framework architectures. (Image source: [Zhong et al., 2025](https://arxiv.org/abs/2504.15930))"
    align="center"
    width="100%"
>}}

As shown in the figure, common distributed scheduling strategies fall into two categories:

1.  **Colocated**: Rollout and Training are deployed on the same set of GPUs and executed alternately using time-slicing. This approach is simple to implement with low communication overhead but suffers from poor stability and cannot leverage heterogeneous hardware.
2.  **Disaggregated**: Rollout and Training are deployed on separate, independent GPU clusters. This architecture is more flexible, has higher stability, and allows for hybrid deployment of heterogeneous hardware, but it may introduce pipeline bubbles, affecting throughput.

### verl

**verl (Volcano Engine Reinforcement Learning)** ([Sheng et al., 2024](https://arxiv.org/abs/2409.19256v2)) is an efficient and general-purpose reinforcement learning framework for LLMs, open-sourced by ByteDance.

{{< figure
    src="verl_async_system_arch.png"
    caption="Fig. 12. The asynchronous system architecture of verl. (Image source: [ByteDance Seed, 2025](https://verl.readthedocs.io/en/latest/start/agentic_rl.html))"
    align="center"
    width="100%"
>}}

The core of verl is its **asynchronous architecture**, which decouples stages like Rollout, reward calculation, and model optimization, processing them in a pipeline to maximize hardware utilization. Its workflow is as follows:

1.  The `PPOTrainer` initiates a PPO iteration, first performing **rollout**, then **train**.
2.  The `AgentLoopManager` wakes up/synchronizes the weights of the inference and training engines (vLLM/SGLang ⇄ FSDP/Megatron-LM), splits the batch into chunks, and dispatches them to multiple `AgentLoopWorker`s for **concurrent** execution.
3.  Each `AgentLoopWorker` starts a coroutine for each sample. When generation is needed, it routes the request to the inference instance with the **lowest load** via the `AsyncLLMServerManager`, naturally supporting **multi-turn dialogue and multi-tool calls**.
4.  verl natively supports **Loss Masking** for external information like tool outputs. This means these tokens are ignored when calculating the loss, ensuring the model is only responsible for the content it generates. This is a key feature for maintaining stability in Tool RL training.

{{< figure
    src="verl_agent_loop_worker.png"
    caption="Fig. 13. The agent loop worker of verl based on React. (Image source: [ByteDance Seed, 2025](https://verl.readthedocs.io/en/latest/start/agentic_rl.html))"
    align="center"
    width="100%"
>}}

5. After the rollout is complete, instances are uniformly collected/hibernated to free up GPU memory. Pluggable interfaces allow for customizing **reward functions**, integrating new **tools**, or replacing **RL algorithms** (e.g., deriving a custom Agent from `ReactAgentLoop`). By default, it uses the LangGraph framework to implement a [ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/?h=react).

## Case Studies

### Search-R1
**Search-R1** ([Jin et al., 2025](https://arxiv.org/abs/2503.09516)) trains an LLM to autonomously engage in multi-turn interactions with a search engine during its step-by-step reasoning process, thereby learning when and how to leverage external knowledge.

{{< figure
    src="search_r1_template.png"
    caption="Fig. 14. Template for SEARCH-R1. `question` will be replaced with the specific question during training and inference. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

In the RL training trajectory, the model's interaction with the environment follows these steps:
1.  Conducts **autonomous reasoning** within `<think>...</think>` tags.
2.  When it recognizes a knowledge gap, it generates a `<search>query</search>` tag to **call the search engine**.
3.  After the environment performs the search, it feeds the retrieved information back to the model wrapped in `<information>...</information>` tags.
4.  The model continues its reasoning based on the new information and can perform multiple rounds of searches until it finally provides an answer within `<answer>...</answer>` tags.

{{< figure
    src="search_r1_rollout.png"
    caption="Fig. 15. LLM Response Rollout with Multi-Turn Search Engine Calls. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

This process is implemented through a loop algorithm: the model generates text until it encounters a `<search>` or `<answer>` tag. If a search request is detected, the system pauses generation, executes the search, and injects the results wrapped in `<information>` tags back into the context for the model to continue reasoning in the next step. This loop continues until the model generates a final answer or reaches the maximum number of interactions.

The paper uses PPO and GRPO algorithms for training. To ensure training stability, Search-R1 introduces a **retrieved token masking** mechanism: when calculating the RL loss (policy gradient and KL divergence), all retrieved tokens returned by the search engine and wrapped in `<information>` tags are **masked**, and their loss does not contribute to the gradient update. This design forces the model to focus on learning **when and how to reason and search**, rather than mechanically imitating external retrieved content, thus effectively preventing training instability. The results below show that the model trained with this strategy (Qwen2.5-7b-base, trained with PPO) consistently outperforms the version trained without masking.

{{< figure
    src="search_r1_token_mask.png"
    caption="Fig. 16. The performance of SEARCH-R1 with and without retrieved token loss masking. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

Search-R1 employs a simple outcome-based reward function, scoring only based on the correctness of the final answer, using exact match as the criterion.

$$
R(y, \hat{y}_{\text{gold}}) = \text{EM}(y_{\text{answer}}, \hat{y}_{\text{gold}}) = 
\begin{cases} 
1, & \text{if } y_{\text{answer}} = \hat{y}_{\text{gold}} \\ 
0, & \text{otherwise} 
\end{cases}
$$

This simple reward design proved effective enough to guide the model to learn complex search and reasoning behaviors.

{{< figure
    src="search_r1_result.png"
    caption="Fig. 17. The main results comparing SEARCH-R1 with baseline methods across the seven datasets. (Image source: [Jin et al., 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="100%"
>}}

Experiments show that on the Qwen2.5-7B and Qwen2.5-3B models, Search-R1 achieved average relative performance improvements of **24%** and **20%**, respectively, compared to traditional RAG baselines.

### ReTool

**ReTool** ([Luo et al., 2025](https://arxiv.org/abs/2504.11536)) is a case study on training a model to decide when and how to call a **Code Interpreter (CI)** to solve math problems, based on the verl framework.

{{< figure
    src="ReTool_arch.png"
    caption="Fig. 18. The architecture of ReTool. (Image source: [Luo et al., 2025](https://arxiv.org/abs/2504.11536))"
    align="center"
    width="100%"
>}}

ReTool adopts a two-stage process: "**Cold-start SFT → Tool-augmented RL**":

1.  **Cold-start SFT**: First, a dataset containing code-augmented reasoning trajectories is constructed. SFT is used to equip the model with basic tool-calling capabilities.
2.  **Tool-augmented RL**: During the RL phase, the model can generate `<code>...</code>` snippets while solving a problem. This code is executed in a sandboxed environment (like [SandboxFusion](https://github.com/bytedance/SandboxFusion)), and the execution result (including output or error stack) is fed back to the model wrapped in `<interpreter>...</interpreter>` tags. The model can then continue reasoning or perform **self-correction** based on the feedback.

{{< figure
    src="ReTool_self_correction.png"
    caption="Fig. 19. The case of an “aha moment” about code self-correction. (Image source: [Luo et al., 2025](https://arxiv.org/abs/2504.11536))"
    align="center"
    width="100%"
>}}

Its reward function scores only based on the correctness of the final answer, encouraging the model to autonomously explore robust reasoning-execution strategies.

$$
R(a,\hat a)=
\begin{cases}
1, & \text{is_equivalent}(a,\hat a) \\
-1, & \text{otherwise}
\end{cases}
$$

Training is based on the PPO algorithm. Similar to Search-R1, it applies **full loss masking** to the interpreter feedback `<interpreter>...</interpreter>`, updating only the model's thoughts and code to prevent gradients from being contaminated by external environmental noise. The results below show that in the AIME 2024/2025 evaluation, ReTool enabled the Qwen2.5-32B-Instruct model to achieve an accuracy of 67.0% / 49.3% with just 400 RL steps, surpassing the text-only RL baseline while reducing the average response length by about 40%.

{{< figure
    src="ReTool_aime.png"
    caption="Fig. 20. AIME 2024 & 2025 scores of ReTool and text-based RL baseline on the Qwen2.5-32B-Instruct model. (Image source: [Luo et al., 2025](https://arxiv.org/abs/2504.11536))"
    align="center"
    width="100%"
>}}

## References

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

## Citation

> **Citation**: When reproducing or citing the content of this article, please credit the original author and source.

**Cited as:**

> Yue Shui. (September 2025). Agentic RL
> https://syhya.github.io/en/posts/2025-09-30-agentic-rl/

Or

```bibtex
@article{yue_shui_agentic_rl_2025,
  title   = "Agentic RL",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "September",
  url     = "https://syhya.github.io/en/posts/2025-09-30-agentic-rl/"
}
```