---
title: "Scaling Laws"
date: 2025-11-19T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "LLM", "Scaling Laws", "Test-Time Compute", "Reinforcement Learning", "Reward Model", "Compute-Optimal"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

From the evolution of the GPT series, researchers have gradually realized that as long as model parameters, training data, and compute resources are continuously scaled up, the performance of large models improves along a stable and predictable path. This predictability is characterized by **Scaling Laws**, which provide the theoretical foundation and practical confidence for high-cost pre-training. As model scale, alignment techniques, and inference-time compute co-evolve, the boundaries of AI capabilities are being systematically pushed. Scaling laws are not only the foundation for building next-generation models but also a key methodology for continuously improving model capabilities under compute constraints.

## Scaling Laws

**Scaling Laws** ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)) are the foundation for understanding how Large Language Model (LLM) performance improves with resource investment. It was the first to systematically reveal the [Power-Law](https://en.wikipedia.org/wiki/Power_law) relationship between model performance and three core factors: model parameter count $N$, dataset size $D$, and training compute $C$. A basic inverse power-law relationship can be expressed as:

$$
y = a \left( \frac{1}{x} \right)^p
$$

In a log-log plot, this relationship appears as a straight line, which is the iconic chart seen in most scaling law papers. For LLMs, `y` is typically the model's test loss, while `x` is a scale variable of interest (e.g., parameter count).

### Basic Laws

{{< figure
    src="scaling_laws.png"
    caption="Fig. 1. Language modeling performance improves smoothly with model size, dataset size, and training compute. (Image source: [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361))"
    align="center"
    width="100%"
>}}

When the other two factors are not bottlenecks, the model's test loss $L$ exhibits a power-law relationship with model parameters $N$, dataset size $D$, and compute $C$, respectively. These relationships can be precisely modeled mathematically:

1.  **Model Size (N)**: With sufficient data, loss decreases as the number of non-embedding parameters $N$ increases.

$$
L(N)=\left(N_{\mathrm{c}} / N\right)^{\alpha_N} ; \alpha_N \sim 0.076, \quad N_{\mathrm{c}} \sim 8.8 \times 10^{13} \text { (non-embedding parameters) }
$$

2.  **Dataset Size (D)**: For LLMs trained to convergence (via early stopping), loss decreases as dataset size $D$ increases.

$$
L(D)=\left(D_{\mathrm{c}} / D\right)^{\alpha_D} ; \alpha_D \sim 0.095, \quad D_{\mathrm{c}} \sim 5.4 \times 10^{13} \text { (tokens) }
$$

3.  **Training Compute (C)**: With optimal model size and sufficient data, loss decreases as the minimum compute $C_{\min}$ increases.

$$
L\left(C_{\min }\right)=\left(C_{\mathrm{c}}^{\min } / C_{\min }\right)^{\alpha_C^{\min }} ; \alpha_C^{\min } \sim 0.050, \quad C_{\mathrm{c}}^{\min } \sim 3.1 \times 10^8 \text { (PF-days) }
$$

Here, $N_c, D_c, C_c^{\min}$ are constants, while $\alpha_N, \alpha_D, \alpha_C^{\min}$ are scaling exponents that determine the rate of performance improvement. This work also uncovered several critical findings:

- **Larger models are more sample-efficient**: To reach the same level of loss, larger models require fewer training samples. When the compute budget increases, priority should be given to increasing model parameters $N$ rather than data volume $D$ (suggesting $N \propto C^{0.73}, D \propto C^{0.27}$).

{{< figure
    src="scaling_law_optimal_parameters.png"
    caption="Fig. 2. Optimal parameters for compute-efficient training. (Image source: [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361))"
    align="center"
    width="100%"
>}}

- **Model shape matters little**: Provided the total parameter count is fixed, specific architectural designs such as depth vs. width have a minimal impact on performance compared to scale itself.
- **C ≈ 6ND**: In Decoder-only Transformer models, training compute (FLOPs) can be estimated via a simple formula. With $N$ parameters, a forward pass processes one token using approximately $2N$ FLOPs, and the backward pass is roughly twice the forward pass. Thus, the cost to train per token is approximately $6N$ FLOPs. If the training corpus contains $D$ tokens, the total training compute is approximately $6ND$.

### Multimodal

**Scaling Laws for Autoregressive Generative Modeling** ([Henighan et al., 2020](https://arxiv.org/abs/2010.14701)) verified that scaling laws are not unique to language models but also apply to **multimodal models**. Autoregressive Transformers in image, video, and multimodal tasks all show predictable power-law performance improvements with scale.

{{< figure
    src="autoregressive_scaling_law.png"
    caption="Fig. 3. Smooth scaling of reducible loss across different domains (Image, Video, Multimodal). (Image source: [Henighan et al., 2020](https://arxiv.org/abs/2010.14701))"
    align="center"
    width="100%"
>}}

### Chinchilla

**Chinchilla** ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) revised OpenAI's conclusions. By training over 400 models, they found that OpenAI did not identify the true optimum due to the use of fixed learning rate schedules.

The Chinchilla scaling law states that **to achieve Compute-Optimality, model parameters $N$ and training tokens $D$ should be scaled equally**. That is:
$$ N \propto C^{0.5}, \quad D \propto C^{0.5} $$

This implies that under a given compute budget, most existing large models (such as GPT-3, Gopher) are *undertrained*. Chinchilla (70B), trained following this law (using the same compute as Gopher 280B but with 4x the data), outperformed Gopher across the board.

{{< figure
    src="Chinchilla.png"
    caption="Fig. 4. Training curve envelope and optimal model size/token count projections. (Image source: [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556))"
    align="center"
    width="100%"
>}}

### GPT-4

In the **GPT-4** technical report ([OpenAI, 2023](https://arxiv.org/abs/2303.08774)), OpenAI demonstrated a crucial application of Scaling Laws: **predicting the final performance of large models using small models**. They accurately predicted GPT-4's final loss using models with only 1/10,000th the compute. Their fitting formula introduces an **irreducible loss** $c$:

$$
L(C)=a C^b+c
$$

**Parameter Definitions:**

* $C$: Training compute.
* $L(C)$：Model loss under compute $C$.
* $a$：**Scale Coefficient**, controls the overall magnitude of loss reduction as compute increases.
* $b$：**Scaling Exponent**, determines the rate of loss reduction; a larger exponent means faster reduction.
* $c$：**Irreducible Loss**, reflecting the inherent entropy of the data, i.e., the lower bound of error that cannot be reduced regardless of compute investment.

{{< figure
src="gpt4_loss.png"
caption="Fig. 5. GPT-4’s final loss on an internal code-token dataset aligns with a power-law trend extrapolated from smaller models when training compute is normalized. (Image source: [OpenAI, 2023](https://arxiv.org/abs/2303.08774))"
align="center"
width="100%"
>}}

In a plot with normalized training compute on the x-axis and bits-per-word on the y-axis, the loss points from small models exhibit a highly stable power-law linear relationship. The power-law curve fitted from these points (without using any GPT-4 data) falls almost precisely on GPT-4's final loss.

## RM Scaling Laws

In the RLHF pipeline, the Reward Model (RM) acts as a proxy for human preference. However, the RM is not a perfect discriminator. [Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law) states: **When a measure becomes a target, it ceases to be a good measure**. Overoptimizing this proxy RM during LLM training leads to performance degradation on the true objective.

**Scaling Laws for Reward Model Overoptimization** ([Gao et al., 2023](https://arxiv.org/abs/2210.10760)) used a synthetic data setup, employing a 6B parameter "Gold RM" as the true reward $R$, and using different "Proxy RMs" ranging from 3M to 3B parameters as optimization targets.

{{< figure
    src="rm_scaling_laws.png"
    caption="Fig. 6. Scaling laws for reward model overoptimization for Best-of-N and RL. (Image source: [Gao et al., 2023](https://arxiv.org/abs/2210.10760))"
    align="center"
    width="100%"
>}}

The KL divergence from the initial policy to the optimized policy is defined as:

$$
\mathrm{KL}:=D_{\mathrm{KL}}\left(\pi \| \pi_{\mathrm{init}}\right)
$$

And a distance function is defined as:

$$
d:=\sqrt{D_{\mathrm{KL}}\left(\pi \| \pi_{\mathrm{init}}\right)}
$$

The change in the Gold RM score $R$ follows different functional forms depending on the optimization method:

$$
R_{\mathrm{BoN}}(d) = d(\alpha_{\mathrm{BoN}} - \beta_{\mathrm{BoN}} d)
$$

**Best-of-N (BoN) Rejection Sampling** exhibits a **quadratic decay** relationship (increasing then decreasing). In the early stages of optimization, the true reward increases, but after passing a certain optimal point, the reward drops significantly as optimization deepens.

$$
R_{\mathrm{RL}}(d) = d(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}} \log d)
$$

**RL** has a decay term of $\log d$, meaning the decline is significantly slower than BoN, exhibiting a **logarithmic decay** relationship.

{{< figure
    src="rm_size.png"
    caption="Fig. 7. The values of  $\alpha_{\text {bon }}$, $\beta_{\text {bon }}$ and $\beta_{\mathrm{RL}}$ in the BoN and RL overoptimization scaling laws for both proxy (dashed line) and gold (solid line) rewards as they scale with parameter count. (Image source: [Gao et al., 2023](https://arxiv.org/abs/2210.10760))"
    align="center"
    width="100%"
>}}

The coefficients $\alpha$ and $\beta$ in the above formulas describe the initial optimization efficiency and the severity of overoptimization, leading to the following conclusions:

1. **RM Parameter Count is Critical**: Increasing the RM parameters effectively increases $\alpha$ and decreases $\beta$, thereby mitigating overoptimization and achieving higher Gold RM peaks.
2. **RL is More Robust to Overoptimization than BoN**: Measured by KL divergence, RL is slower in both optimization and overoptimization compared to BoN.

## Test-time Scaling

In addition to investing compute during the training phase, performance can also be improved by increasing compute during the inference phase. This is known as **Test-Time Scaling** ([Snell et al., 2024](https://arxiv.org/abs/2408.03314)). The main strategies include Parallel Sampling and Sequential Refinement.

*   **Parallel Sampling**: e.g., Best-of-N or Majority Voting. The model generates multiple independent samples for the same question, and the best answer is selected via a verifier or voting.
*   **Sequential Refinement**: e.g., Self-Correction. The model iteratively improves based on the previous output.

### Efficiency Trade-off
{{< figure
    src="test_time_scaling.png"
    caption="Fig. 8. Compute-Optimal Scaling for Iterative Self-Refinement and Search: Evaluating Efficiency Against Best-of-N Baselines and Analyzing the Trade-offs Between Test-Time Compute and Pretraining Scaling (Image source: [Snell et al., 2024](https://arxiv.org/abs/2408.03314))"
    align="center"
    width="100%"
>}}

From the figure above, we can draw the following conclusions:

1. **The optimal strategy depends on problem difficulty, and there is a limit to returns**: Although compute-optimal strategies are generally more efficient than the Best-of-N baseline (saving ~4x compute), their specific efficacy is highly dependent on task complexity.
    * **Easy Problems:** The model's initial answer is usually directionally correct and only needs local fine-tuning. In this case, sequential refinement is the most efficient strategy, quickly fixing details with minimal compute cost.
    * **Hard Problems:** Although parallel search can explore more paths, as problem difficulty increases, the return on simply increasing test-time inference (whether search or refinement) rapidly shows **diminishing marginal returns**. For extremely hard problems, test-time compute struggles to break through the ceiling of the model's capabilities.

2. **Test-time compute cannot fully replace pre-training; base capability remains fundamental**: Test-time Compute and Pretraining Compute are not a 1:1 equivalent exchange.
    * **Prerequisite for Exchange ($Y \ll X$):** Increasing inference time is only cost-effective when the inference token budget ($Y$) is far smaller than the pre-training token budget ($X$), and the problems are of low-to-medium difficulty.
    * **Irreplaceability of Base Models:** Once the problem becomes hard or inference demands are too high, the reasoning tricks (search/refinement) of small models cannot bridge the huge capability gap. The broad knowledge and strong reasoning capabilities provided by scaling up pre-training remain the decisive foundation for solving complex problems.

### Simple test-time scaling

**s1: Simple test-time scaling** ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)) curated the [s1K](https://huggingface.co/datasets/simplescaling/s1K) dataset containing 1,000 questions paired with reasoning traces. They then performed supervised fine-tuning (SFT) using Qwen2.5-32B-Instruct as the base model, employing a **Budget Forcing** method to control the length of output tokens to study test-time scaling.

{{< figure
    src="s1_sample.png"
    caption="Fig. 9. Budget forcing in s1-32B: suppressing the end-of-thinking delimiter prompts the model to continue after “...is 2.”, triggering a self-correction via “Wait”. (Image source: [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))"
    align="center"
    width="100%"
>}}

* **Lengthening**: When the model attempts to end thinking, suppress the end token and force append words like `Wait` to encourage the model to reflect and double-check.
* **Shortening**: When the thinking process exceeds a preset token length, force append end tokens (e.g., `</think>` or `"Final Answer:"`) to prompt the model to output a conclusion immediately.

{{< figure
    src="s1_result.png"
    caption="Fig. 10. Sequential and parallel test-time scaling. (Image source: [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))"
    align="center"
    width="100%"
>}}

The results above show that thinking time (Token count) extended via Budget Forcing in both parallel and sequential scaling methods is significantly positively correlated with downstream task accuracy.

## Scale RL

**ScaleRL** ([Khatri et al., 2025](https://arxiv.org/abs/2510.13786)) proposes a systematic method for analyzing and predicting RL scaling in LLMs. Unlike the common power-law relationship in pre-training, the performance of RL training (e.g., expected reward $R_C$ on the validation set) follows a **Sigmoidal saturation curve** as compute $C$ increases. This curve can be described by the following formula:

$$
\overbrace{R_C-R_0}^{\text {Reward Gain }}=\overbrace{\left(A-R_0\right)}^{\text {Asymptotic Reward Gain }} \times \frac{1}{\underbrace{1+\left(C_{\text {mid }} / C\right)^B}_{\text {Compute Efficiency }}}
$$

**Parameter Definitions:**
*   $R_C$: Expected reward (or Pass Rate) under compute $C$.
*   $R_0$: Initial model performance.
*   $A$: **Asymptotic Performance**. Represents the theoretical maximum performance the model can achieve under infinite compute.
*   $B$: **Scaling Exponent (Efficiency)**. Controls the steepness of the curve; a larger value means the model reaches the performance ceiling faster.
*   $C_{mid}$: The compute required to achieve half of the total gain.

{{< figure
    src="rl_scaling_fit.png"
    caption="Fig. 11. Interpretation of the sigmoidal scaling curve for RL. (Image source: [Khatri et al., 2025](https://arxiv.org/abs/2510.13786))"
    align="center"
    width="100%"
>}}

The figure above demonstrates the value of this framework: It allows researchers to fit the curve using early training data at a smaller compute scale, thereby **predicting** the final performance and efficiency of an RL recipe under a larger compute budget, reducing the cost of algorithm exploration.

Through large-scale ablation experiments on various RL design choices (such as loss functions, advantage normalization, data curriculum, etc.), several key principles were summarized:

1.  **Performance ceilings are not universal**: Different RL algorithms (e.g., GRPO, DAPO, CISPO) have different asymptotic performance limits $A$. Choosing the correct loss function is critical.
2.  **Decoupling of Efficiency and Ceiling**: Many common RL tricks, such as advantage normalization, data curriculum, and length penalties, mainly affect compute efficiency ($B$ and $C_{\text{mid}}$), but have little impact on the performance ceiling $A$.
3.  **Stable and Predictable Scaling**: A carefully designed, stable RL recipe follows a predictable sigmoidal trajectory. Even when scaled to the level of 100,000 GPU hours, its performance remains highly consistent with the curve fitted from early data.

{{< figure
    src="scale_rl_result.png"
    caption="Fig. 12. ScaleRL demonstrates more scalable performance compared to other prevalent RL methods. (Image source: [Khatri et al., 2025](https://arxiv.org/abs/2510.13786))"
    align="center"
    width="90%"
>}}

## References

[1] Kaplan, Jared, et al. ["Scaling laws for neural language models."](https://arxiv.org/abs/2001.08361) arXiv preprint arXiv:2001.08361 (2020).

[2] Henighan, Tom, et al. ["Scaling laws for autoregressive generative modeling."](https://arxiv.org/abs/2010.14701) arXiv preprint arXiv:2010.14701 (2020). 

[3] Hoffmann, Jordan, et al. ["Training compute-optimal large language models."](https://arxiv.org/abs/2203.15556) arXiv preprint arXiv:2203.15556 (2022).

[4] Achiam, Josh, et al. ["Gpt-4 technical report."](https://arxiv.org/abs/2303.08774) arXiv preprint arXiv:2303.08774 (2023).

[5] Gao, Leo, John Schulman, and Jacob Hilton. ["Scaling laws for reward model overoptimization."](https://arxiv.org/abs/2210.10760) International Conference on Machine Learning. PMLR, 2023.

[6] Snell, Charlie, et al. ["Scaling llm test-time compute optimally can be more effective than scaling model parameters."](https://arxiv.org/abs/2408.03314) arXiv preprint arXiv:2408.03314 (2024).

[7] Muennighoff, Niklas, et al. ["s1: Simple test-time scaling."](https://arxiv.org/abs/2501.19393) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025.

[8] Khatri, Devvrit, et al. ["The art of scaling reinforcement learning compute for llms."](https://arxiv.org/abs/2510.13786) arXiv preprint arXiv:2510.13786 (2025).

## Citation

> **Citation**: When reproducing or citing the content of this article, please credit the original author and source.

**Cited as:**

> Yue Shui. (Nov 2025). Scaling Laws.
> https://syhya.github.io/posts/2025-11-19-scaling-law/

Or

```bibtex
@article{yue_shui_scaling_laws_2025,
  title   = {Scaling Laws},
  author  = {Yue Shui},
  journal = {syhya.github.io},
  year    = {2025},
  month   = {November},
  url     = {https://syhya.github.io/posts/2025-11-19-scaling-law/}
}
```