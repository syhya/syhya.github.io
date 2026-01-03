---
title: "DeepSeek-V3.2 Series"
date: 2025-12-31T12:00:00+08:00
lastmod: 2025-12-31T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "LLM", "DeepSeek", "Sparse Attention", "Reinforcement Learning", "Reasoning", "Agent", "Theorem Proving"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

By introducing DeepSeek Sparse Attention (DSA), a scalable reinforcement learning framework, and a large-scale agentic task synthesis pipeline, **DeepSeek-V3.2** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556)) achieves reasoning capabilities and agent performance comparable to GPT-5.

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

Traditional attention mechanisms assume that every token needs to attend to all historical tokens. However, from an information-theoretic perspective, effective information in text is highly unevenly distributed, with only a minority of historical tokens being truly relevant to the current token. While [sliding window attention](https://syhya.github.io/posts/2025-08-24-gpt5/#efficient-attention-mechanisms) recognizes this by limiting attention to recent windows to simplify computation, it risks losing critical long-range dependencies. DeepSeek's core insight is to **enable the model to autonomously learn and dynamically select truly important tokens**, achieving a better balance between efficiency and information retention.

### Lightning Indexer

The core of DSA is using a lightweight **Lightning Indexer** to quickly filter relevant tokens. For each query token $\mathbf{h}_t$, the indexer computes relevance scores with all preceding tokens $\mathbf{h}_s$:

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)
$$

Several design aspects are noteworthy:
1. **Multi-head indexing**: Uses $H^I$ indexer heads, each learning different relevance patterns
2. **ReLU activation**: Zeros out negative correlations, providing sparsity
3. **Learnable weights**: $w_{t,j}^I$ determines each head's contribution, allowing dynamic model adjustment

The indexer's computational cost is far lower than the main attention. The paper mentions it can be implemented in FP8, significantly reducing computational overhead while maintaining precision.

### Fine-grained Token Selection

With relevance scores computed, the model only needs to select the top-k most relevant tokens:

$$
\mathbf{u}_t = \text{Attn}\left(\mathbf{h}_t, \left\{\mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\right\}\right)
$$

Here, $k$ is set to 2048 to balance efficiency and effectiveness. This setting reduces attention computation complexity from $O(L^2)$ to $O(Lk)$, significantly reducing computational overhead while still covering most critical dependencies.

### Continued Pre-training

Continued pre-training from [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) consists of two training stages:

**Dense Warm-up Stage**

Maintains the original dense attention, training only the indexer. The goal is to align the indexer's output distribution with the true attention distribution:

$$
\mathcal{L}^I = \sum_t D_{\text{KL}}\left(p_{t,:} \| \text{Softmax}(I_{t,:})\right)
$$

Here, $p_{t,:}$ represents the true attention distribution aggregated from main attention scores. This stage ensures the indexer learns to identify which historical tokens are most important for modeling at the current time step through alignment with the indexer output distribution.

- Learning rate: $10^{-3}$
- Training steps: 1000 steps
- Per step: 16 sequences × 128K tokens
- Total tokens: 2.1B

**Sparse Training Stage**

After indexer warm-up, introduces fine-grained token selection mechanism and jointly optimizes all model parameters to adapt to DSA's sparse attention computation pattern. In this stage, the indexer only aligns with the main attention distribution on the selected subset of key tokens, with loss function defined as:

$$
\mathcal{L}^I = \sum_t \mathbb{D}_{\mathrm{KL}}\left(p_{t,\mathcal{S}_t} \| \operatorname{Softmax}\left(I_{t,\mathcal{S}_t}\right)\right)
$$

$$
\mathcal{S}_t=\left\{s \mid I_{t, s} \in \operatorname{Top}-\mathrm{k}\left(I_{t,:}\right)\right\}
$$

where $\mathcal{S}_t$ represents the set of top-k key-value tokens predicted as most important by the indexer at time step $t$.

* Learning rate: $7.3 \times 10^{-6}$
* Selects 2048 key-value tokens per query token
* Training steps: 15000 steps
* Per step: 480 sequences × 128K tokens
* Total tokens: 943.7B

### Inference Costs

DeepSeek-V3.2 requires less computation compared to MLA in DeepSeek-V3.1-Terminus, with benefits becoming more pronounced as context length increases.

Actual cost comparison on H800 GPU clusters (calculated at $2/GPU hour):

{{< figure
    src="inference_cost_compare.png"
    caption="Fig. 3. Inference costs of DeepSeek-V3.1-Terminus and DeepSeek-V3.2 on H800 clusters. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

## Scaling GRPO

DeepSeek-V3.2 merges reasoning, agent, and human alignment training into a single RL stage. This approach effectively balances performance across diverse domains while circumventing catastrophic forgetting issues common in multi-stage training paradigms.

Reward design:
- **Reasoning and agent tasks**: Rule-based outcome reward + length penalty + language consistency reward
- **General tasks**: Generative reward model where each prompt has its own evaluation rubrics

### GRPO

[GRPO](https://syhya.github.io/posts/2025-01-27-deepseek-r1/#grpo) is an efficient RL algorithm proposed by DeepSeek that replaces the value model in traditional PPO with group-relative advantage estimation. GRPO optimizes the policy model $\pi_\theta$ by maximizing the following objective:

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)= & \mathbb{E}_{q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\mathrm{old}}(\cdot \mid q)}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\right. \\
& \left.\min \left(r_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right)-\beta \mathbb{D}_{\mathrm{KL}}\left(\pi_\theta\left(o_{i, t}\right) \| \pi_{\mathrm{ref}}\left(o_{i, t}\right)\right)\right],
\end{aligned}
$$

where the importance sampling ratio is:
$$
r_{i, t}(\theta)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_{\mathrm{old}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}
$$

However, in large-scale training, vanilla GRPO encounters stability issues. DeepSeek-V3.2 proposes four key improvements.

### Key Strategies for Stable RL Scaling

1. **Unbiased KL Estimate**

The K3 estimator used in vanilla GRPO can produce systematic bias in certain cases. When sampled tokens have much lower probability under the current policy than the reference policy ($\pi_\theta\left(o_t \mid q, o_{&lt;t  }\right) \ll \pi_{\mathrm{ref}}\left(o_t \mid q, o_{&lt;t  }\right)$), gradients become abnormally large, causing training instability.

$$
\mathbb{D}_{\mathrm{KL}}\left(\pi_\theta\left(o_{i, t}\right) \| \pi_{\mathrm{ref}}\left(o_{i, t}\right)\right)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_{\mathrm{old}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}\left(\frac{\pi_{\mathrm{ref}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}-\log \frac{\pi_{\mathrm{ref}}\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,&lt;t  }\right)}-1\right)
$$

By incorporating the **importance-sampling ratio between current policy $\pi_\theta$ and old policy $\pi_{\text{old}}$** to correct the K3 estimator, the KL (and its gradient) estimation becomes **unbiased with more stable convergence**, and allows **domain-specific adjustment of KL penalty strength** (weakening or even omitting when necessary).

2. **Off-Policy Sequence Masking**

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)= & \mathbb{E}_{q \sim P(Q),\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\mathrm{old}}(\cdot \mid q)}\left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|}\right. \\
& \left.\min \left(r_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right) M_{i, t}-\beta \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}\left(o_{i, t}\right) \| \pi_{\mathrm{ref}}\left(o_{i, t}\right)\right)\right],
\end{aligned}
$$

$$
M_{i,t} = \begin{cases} 0 & \text{if } \hat{A}_{i,t} < 0 \text{ and } \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log\frac{\pi_{\text{old}}(o_{i,t}|q, o_{i,&lt;t  })}{\pi_\theta(o_{i,t}|q, o_{i,&lt;t  })} > \delta \\ 1 & \text{otherwise} \end{cases}
$$

To mitigate off-policy issues introduced by multi-step updates and training-inference inconsistency, [Off-Policy Sequence Masking](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) is introduced in GRPO, masking only sequences with **negative advantage and policy divergence (KL) exceeding a threshold**, thereby suppressing interference from highly off-policy negative samples on optimization. This mechanism significantly improves training stability while retaining effective learning signals.

3. **Keep Routing**

For MoE models, a subtle issue arises: the combination of experts activated during inference may differ from training. Even with identical inputs, framework differences or policy updates can lead to different routing results.

DeepSeek's approach is to record routing paths during sampling and enforce the same paths during training. This ensures that gradients update the exact parameters that produced the sampled output. The verl framework has integrated [Router Replay](https://github.com/volcengine/verl/tree/main/examples/router_replay) functionality for direct use.

4. **Keep Sampling Mask**

During the rollout phase of RL training, Top-p/Top-k sampling is commonly used to filter low-probability tokens and improve generation quality. However, training typically optimizes over the full vocabulary, causing action space inconsistency between old policy $\pi_{\text{old}}$ and new policy $\pi_\theta$, violating importance sampling assumptions and causing training instability. To address this, DeepSeek records the truncation mask generated during rollout sampling and applies the same mask to $\pi_\theta$ during training, forcing both policies to optimize within a consistent action subspace, thereby maintaining generation consistency and stability in RL training when combined with Top-p sampling.

## Agentic Task Synthesis and Training

### Thinking in Tool-Use

**Thinking Context Management**

Integrating reasoning capabilities into tool-use scenarios is an interesting challenge. DeepSeek R1's approach discards previous reasoning content at each new message round, but this causes severe token waste in tool-calling scenarios—the model must re-reason through the entire problem after each tool call. **Claude Opus 4.5** ([Anthropic, 2025](https://www.anthropic.com/news/claude-opus-4-5)) achieves cross-context information persistence and context reset through [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) and [new context tool](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf), supporting up to 1M effective tokens accumulated across multiple resets while maintaining a single 200k token context window.

{{< figure
    src="claude_context_window.png"
    caption="Fig. 4. The context window token management when combining extended thinking with tool use in Claude. (Image source: [Claude Docs](https://platform.claude.com/docs/en/build-with-claude/context-windows#the-context-window-with-extended-thinking-and-tool-use))"
    align="center"
    width="100%"
>}}

DeepSeek-V3.2 implements similar fine-grained context management:

{{< figure
    src="deepseek_thinking_tool_use.png"
    caption="Fig. 5. Thinking retention mechanism in tool-calling scenarios. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

- Reasoning content is discarded only when receiving **new user messages**
- Tool outputs and similar messages don't trigger reasoning content deletion
- Tool call history is always preserved in context

This design maintains reasoning coherence while avoiding redundant computation.

### Large-Scale Agentic Tasks

A diverse set of RL tasks is crucial for enhancing model robustness. DeepSeek-V3.2 uses the following agentic tasks:

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

Evaluation results show DeepSeek-V3.2 achieves performance comparable to GPT-5 High on reasoning tasks, significantly outperforms open-source models on Code Agent tasks, and effectively narrows the gap with top closed-source models in Tool-Use scenarios, demonstrating strong generalization capabilities.

### Context Management for Search Agents

Despite extended context windows like 128K, agentic workflows (especially search-based scenarios) frequently encounter maximum length limits that prematurely truncate reasoning processes. This bottleneck severely inhibits the full realization of test-time compute potential. To address this, when token usage exceeds **80%** of the context window, the following context management strategies are introduced to dynamically extend token budget at test time:

1. **Summary**: Summarizes overflowed trajectory content and re-initiates subsequent reasoning.
2. **Discard-75%**: Discards the first 75% of tool call history to free up space.
3. **Discard-all**: Resets context by discarding all previous tool call history (similar to Anthropic's new context tool).

{{< figure
    src="new_context_tool.png"
    caption="Fig. 7. New context tool feature in Claude. (Image source: [Anthropic, 2025](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf))"
    align="center"
    width="90%"
>}}

These strategies were evaluated on the **BrowseComp** benchmark. Results show that context management significantly improves test-time compute scaling and model performance by allowing more execution steps. Specific performance:

{{< figure
    src="search_agent_browsecomp.png"
    caption="Fig. 8. Accuracy of BrowseComp with different test-time compute expansion strategies. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556))"
    align="center"
    width="100%"
>}}

While the Summary strategy successfully extends average steps from 140 to 364, improving scores from 53.4 to 60.2, its overall computational efficiency is relatively low. The Discard-all strategy performs best in both efficiency and scalability, achieving a high score of **67.6**. Notably, it matches the performance of parallel scaling baselines while requiring significantly fewer execution steps.

## DeepSeekMath-V2

Developed on [DeepSeek-V3.2-Exp-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp-Base), **DeepSeekMath-V2** ([Shao et al., 2025](https://arxiv.org/abs/2511.22570)) focuses on formal mathematical reasoning and theorem proving. Unlike traditional final-answer-based reinforcement learning, DeepSeekMath-V2 introduces a **Self-Verification** mechanism that enables rigorous self-examination of reasoning processes during proof generation through collaborative training of an independent proof verifier and meta-verifier. This approach achieves gold-medal level performance on challenging mathematical competition benchmarks like [IMO-Proof Bench](https://imobench.github.io/) and [Putnam](https://maa.org/putnam/).

### Process Reward Models

Traditional mathematical reasoning RL typically employs **Outcome-based Reward Models (ORM)**, rewarding only based on final answer correctness. While this works for competitions like AIME and HMMT that focus on numerical answers, it exposes fundamental limitations in more complex reasoning tasks:

1. **Unreliable proxy metric**: A correct final answer doesn't guarantee correct reasoning. Models may arrive at correct results through flawed logic, shortcuts, or coincidence (False Positive).
2. **Inapplicable to theorem proving**: Higher-order mathematical tasks like theorem proving emphasize rigorous step-by-step logical derivation, where single outcome rewards provide insufficient training signal.

To address these issues, DeepMind pioneered **Process Reward Models (PRM)** for mathematical problem-solving in 2022, proposing explicit evaluation of intermediate reasoning steps to mitigate the supervision deficiency of outcome rewards ([Uesato et al., 2022](https://arxiv.org/abs/2211.14275)). This work systematically revealed that modeling the reasoning process itself is more conducive to learning complex reasoning capabilities than supervising only final answers.

{{< figure
    src="orm_prm_compare.png"
    caption="Fig. 9. Comparison of policy improvement mechanisms for Final-Answer RL, ORM-RL, and PRM-RL. (Image source: [Uesato et al., 2022](https://arxiv.org/abs/2211.14275))"
    align="center"
    width="100%"
>}}

DeepSeekMath-V2 is also trained in this mode, enabling the model to actively identify potential logical gaps during reasoning generation and perform self-correction, simulating human thinking patterns when reading and reviewing proofs.

### Overall Architecture

{{< figure
    src="deepseek_math_v2_arch.png"
    caption="Fig. 10. Self-verification architecture with proof generation, verifier-based evaluation, and meta-verification."
    align="center"
    width="100%"
>}}

DeepSeekMath-V2 constructs a three-tier verification architecture achieving continuous improvement through inter-model supervision:

- **Proof Generator ($\pi_\theta$)**: Generates mathematical proofs based on problem $X$.
- **Proof Verifier ($\pi_\varphi$)**: Acts as LLM-as-a-judge to evaluate proof quality.
- **Meta-Verifier ($\pi_\psi$)**: Supervises the verifier's evaluation process to ensure verification quality.

#### Data Construction

The team constructed initial training data through the following process:

1. **Problem Collection**: Crawled problems from [Art of Problem Solving (AoPS)](https://artofproblemsolving.com/?srsltid=AfmBOoqcstCRpzZaf7rDkaLdkuHkR_SUAaTVBUHDrPo-nctXiCEuobst), prioritizing mathematical olympiads, team selection tests, and post-2010 problems explicitly requiring proofs, totaling **17,503 problems**, denoted as $\mathcal{D}_p$

2. **Candidate Proof Generation**: Used a variant of DeepSeek-V3.2-Exp-Thinking to generate candidate proofs. Since this model wasn't optimized for theorem proving and tended to produce concise but error-prone outputs, it was prompted to iteratively refine proofs over multiple rounds to improve comprehensiveness and rigor

3. **Expert Annotation**: Randomly sampled proofs across different problem types (e.g., algebra, number theory) and had mathematical experts score each proof according to evaluation criteria

This process produced an initial RL dataset $\mathcal{D}_v = \{(X_i, Y_i, s_i)\}$, where each item contains problem $X_i$, proof $Y_i$, and overall proof score $s_i \in \{0, 0.5, 1\}$.

#### Verifier Training

The verifier is initialized from DeepSeek-V3.2-Exp-SFT (supervised fine-tuned on mathematics and code-related reasoning data) and employs a three-level scoring standard:

- **1.0 score**: Complete and rigorous proof with all logical steps clearly justified.
- **0.5 score**: Generally correct overall logic but with minor errors or omitted details.
- **0.0 score**: Fundamentally flawed proof containing fatal logical errors or critical gaps.

Given problem $X$ and proof $Y$, the verifier $\pi_\varphi(\cdot|X, Y, \mathcal{I}_v)$ is designed to first summarize identified issues, then assign a score based on the criteria.

The verifier is optimized through reinforcement learning with two reward components:

- **Format reward $R_{\text{format}}$**: Enforces output of issue summary and proof score in specified format by checking for designated evaluation phrases and scores in `\boxed{}`.

- **Score reward $R_{\text{score}}$**: Based on proximity between predicted score $s'_i$ and annotated score $s_i$:
  $$R_{\text{score}}(s'_i, s_i) = 1 - |s'_i - s_i|$$

The verifier's RL objective is:

$$
\max_{\pi_\varphi} \mathbb{E}_{(X_i, Y_i, s_i) \sim \mathcal{D}_v, (V'_i, s'_i) \sim \pi_\varphi(\cdot|X_i, Y_i)} \left[ R_{\text{format}}(V'_i) \cdot R_{\text{score}}(s'_i, s_i) \right]
$$

where $V'_i$ represents the verifier's final response and $s'_i$ is the proof score extracted from it.

#### Meta-Verifier

The above approach trains the proof verifier through RL to align predicted proof scores with expert annotations, but **doesn't directly supervise the identified issues themselves**. This creates a critical vulnerability: when evaluating flawed proofs during training ($s_i < 1$), the verifier can receive full reward by predicting correct scores while hallucinating non-existent issues, undermining its trustworthiness.

To address this problem, **Meta-Verification** is introduced: a secondary evaluation process that assesses whether issues identified by the verifier actually exist and whether these issues logically justify the predicted proof score according to evaluation criteria $\mathcal{I}_v$.

The meta-verifier is also trained through reinforcement learning with an objective function similar to the verifier's. Using the trained meta-verifier $\pi_\psi$, verifier training is enhanced by integrating meta-verification feedback into the reward function:

$$
R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}
$$

where $R_{\text{meta}}$ is the quality score from the meta-verifier.

Experimental results show that introducing the meta-verifier improves the average quality score of the verifier's proof analyses from 0.85 to 0.96 on a validation split of $\mathcal{D}_v$, while maintaining the same proof score prediction accuracy.

This design is similar to the idea of **Generative Adversarial Networks (GANs)** ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)): the verifier drives the generator to improve, while a stronger generator provides more challenging training samples for the verifier, forming a virtuous cycle. Note that the meta-verification score is **used only during training** and doesn't participate in inference computation.

### Generator Training

With verifier $\pi_\varphi$ as a generative reward model, the proof generator $\pi_\theta$'s optimization objective is:

$$
\max_{\pi_\theta} \mathbb{E}_{X_i \sim \mathcal{D}_p, Y_i \sim \pi_\theta(\cdot|X_i)} [R_Y]
$$

where $R_Y$ is the proof score produced by $\pi_\varphi(\cdot|X_i, Y_i, \mathcal{I}_v)$.

During training, the generator $\pi_\theta$ is prompted to produce proof $Y$ followed by self-analysis $Z$ following the same format and criteria $\mathcal{I}_v$ as the verifier. The predicted proof score in self-analysis is denoted as $s'$. The reward function comprehensively considers these evaluations:

$$
R = R_{\text{format}}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z)
$$

$$
R_Z = R_{\text{score}}(s', s) \cdot R_{\text{meta}}(Z)
$$

where $\alpha = 0.76$ and $\beta = 0.24$.

This reward structure creates the following incentives:

- **Honesty over falsehood**: Faithfully acknowledging errors receives higher rewards than falsely claiming correctness.
- **Self-awareness**: Highest rewards come from producing correct proofs and accurately recognizing their rigor.
- **Active improvement**: An effective strategy for the proof generator to obtain high rewards is to identify and resolve as many potential issues as possible before finalizing the response.

### Sequential Refinement

{{< figure
    src="imo_2024_res.png"
    caption="Fig. 11. Proof Quality Improves with Increasing Sequential Self-Refinement Iterations (1–8). (Image source: [Shao et al., 2025](https://arxiv.org/abs/2511.22570))"
    align="center"
    width="80%"
>}}

[Test-time Scaling](https://syhya.github.io/posts/2025-11-19-scaling-law/#test-time-scaling) improves model performance by increasing computation during inference, especially suitable for high-difficulty mathematical proof tasks like IMO and Putnam. The above figure shows **DeepSeekMath-V2** performance variation under up to **8 sequential iterations**: as maximum sequential iterations increase, **Pass@1** steadily improves from approximately **0.15** to **0.27**, and **Best@32** improves from approximately **0.26** to **0.42**. Sequential refinement combined with self-verification mechanisms significantly improves proof success rates through multiple rounds of generation and error correction under controllable computational cost, showing stable performance gains that increase with iteration count.

## References

[1] Liu, Aixin, et al. ["DeepSeek-V3.2: Pushing the frontier of open large language models."](https://arxiv.org/abs/2512.02556) arXiv preprint arXiv:2512.02556 (2025).

[2] Shao, Zhihong, et al. ["DeepSeekMath-V2: Towards self-verifiable mathematical reasoning."](https://arxiv.org/abs/2511.22570) arXiv preprint arXiv:2511.22570 (2025).

[3] Uesato, Jonathan, et al. ["Solving math word problems with process-and outcome-based feedback."](https://arxiv.org/abs/2211.14275) arXiv preprint arXiv:2211.14275 (2022).

[4] Goodfellow, Ian J., et al. ["Generative adversarial nets."](https://arxiv.org/abs/1406.2661) Advances in neural information processing systems 27 (2014).

[5] Anthropic. ["Claude Opus 4.5 — System Card."](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf) 2025. (PDF)

[6] Luong, Minh-Thang, et al. ["Towards robust mathematical reasoning."](https://arxiv.org/abs/2511.01846) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025.

## Citation

> **Citation**: When reposting or citing content from this article, please credit the original author and source.

**Cited as:**

> Yue Shui. (Dec 2025). DeepSeek-V3.2 Series.
https://syhya.github.io/posts/2025-12-31-deepseekv3.2

Or

```bibtex
@article{syhya2025deepseekv32,
  title   = "DeepSeek-V3.2 Series",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Dec",
  url     = "https://syhya.github.io/posts/2025-12-31-deepseekv3.2"
}
```