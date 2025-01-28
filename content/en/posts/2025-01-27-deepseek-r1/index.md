---
title: "o1 Replication Progress: DeepSeek-R1"
date: 2025-01-27T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "Reinforcement Learning", "LLM", "Reasoning Model", "NLP", "Model Distillation", "DeepSeek-R1", "GRPO", "PPO", "SFT", "RFT", "o1", "Reject sampling"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

DeepSeek AI recently released **DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)), whose reasoning performance on multiple benchmarks approaches the level of OpenAI's o1 ([OpenAI, 2024](https://openai.com/o1/)), marking a significant step for the open-source community in successfully replicating o1. Relevant code for R1 can be found in the huggingface's attempt to open-source replication project [open-r1](https://github.com/huggingface/open-r1). While previous research has often relied on massive amounts of supervised data to enhance the performance of Large Language Models (LLMs), the success of DeepSeek-R1 and its earlier experiment, DeepSeek-R1-Zero, powerfully demonstrates the potential of purely large-scale reinforcement learning in improving the reasoning capabilities of LLMs. This success reinforces the profound insight proposed by Richard Sutton in "The Bitter Lesson":

> One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning. ([Richard Sutton, 2019](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf))

## Notations

The following table lists the mathematical symbols used in this article to facilitate your reading.

| Symbol | Meaning |
| :--- | :--- |
| \( q \) or \( Q \) | Question, user input or instruction |
| \( o \) or \( O \) | Output, model-generated text response or answer |
| \( t \) | Token index, indicating the position of the \( t \)-th token in the output text |
| \( o_t \) | The \( t \)-th token in the output text \( o \) |
| \( o_{&lt;t}   \) | The sequence of tokens in the output text \( o \) preceding the \( t \)-th token |
| \( &#124;o&#124; \) | Length of the output text \( o \), usually referring to the number of tokens |
| \( G \) | Output group size, in the GRPO algorithm, the number of outputs sampled for each question |
| \( \pi_\theta, \pi_{\theta_{\text{old}}}, \pi_{\text{ref}}, \pi_{\text{sft}} \) | Policy models and their variants, used to generate text outputs or as reference models |
| \( A_t, A_i \) | Advantage function and relative advantage value |
| \( \varepsilon \) | Clipping hyperparameter, used to limit the range of the importance sampling, ensuring the stability of policy updates |
| \( \beta \) | Regularization coefficient, used to control the weight of the KL divergence penalty term in the objective function |
| \( \mathbb{D}_{KL} \) | KL divergence, a measure of the difference between two probability distributions, used to constrain the distance between the new policy and the reference policy |
| \( \mathcal{J}, \mathcal{L} \) | Objective function and loss function |
| \( \mathbb{E} \) | Expectation, representing the average value of a random variable, in the objective function, it represents the average over sample data |
| \( P_{\text{sft}}(Q, O) \) | Distribution of the SFT dataset, representing the joint probability distribution of question \( Q \) and output \( O \) in the \( SFT \) dataset |
| \( P_{\text{sft}}(Q) \) | Distribution of questions in the SFT dataset, representing the marginal probability distribution of question \( Q \) in the \( SFT \) dataset |
| \( \pi_\theta(o_t \mid q, o_{&lt;t}  ) \) | Conditional probability of the policy model generating the \( t \)-th token \( o_t \) given the question \( q \) and previously generated tokens \( o_{&lt;t}   \) |
| \( \mathbb{I}(o) \) | Indicator function that determines whether the output \( o \) is of high quality, 1 if high quality, 0 otherwise |
| \( r(o) \) | Reward function, a function that evaluates the quality of the model output \( o \) |
| \( r_i \) | Reward value of the \( i \)-th output |
| \( \nabla_{\theta} \) | Gradient operator, representing the gradient of a function with respect to model parameters \( \theta \) |
| \( \mathcal{N}(\mu, 1) \) | Normal distribution with mean \( \mu \) and standard deviation 1 |
| \( \binom{a}{b} \) | Binomial coefficient, representing the number of combinations of choosing \( b \) elements from \( a \) elements |
| \( r(o) = \frac{\pi_{\text{ref}}(o \mid q)}{\pi_\theta(o \mid q)} \) | Probability ratio, the ratio of the probability of generating output \( o \) by the reference model to the current policy model |

## Training Process Overview

The training of the DeepSeek-R1 series models is a multi-stage process aimed at building LLMs with superior reasoning and general language capabilities. The entire training process starts from the **DeepSeek-V3** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)) model and iteratively optimizes it to obtain different versions of the DeepSeek-R1 model.

{{< figure
    src="deepseek_r1_pipeline.jpg"
    caption="Fig. 1. DeepSeek R1 Training Pipeline. (Image source: [Harris Chan's Tweet](https://x.com/SirrahChan/status/1881488738473357753))"
    align="center"
    width="90%"
>}}

As shown in Figure 1, the DeepSeek-R1 training process is clearly displayed and mainly divided into the following key stages:

- **Base Model and Initial Fine-tuning**: The starting point of the process is the **DeepSeek-V3 Base** model. First, SFT technology is used to initially train the base model on **cold-start long-text CoT data**, endowing the model with preliminary reasoning abilities.

- **Reinforcing Reasoning Ability**: Based on SFT, a reasoning-oriented reinforcement learning method, specifically the Group Relative Policy Optimization (GRPO) algorithm, combined with rule-based rewards and CoT language consistency rewards, is used to further enhance the model's reasoning ability.

- **Reasoning Data Generation and Rejection Sampling**: Using reasoning prompts and rejection sampling techniques, and leveraging rules and the DeepSeek-V3 model to judge data quality, high-quality reasoning data is generated.

- **Non-Reasoning Data Generation**: Using the CoT prompting method, the DeepSeek-V3 model is used for data augmentation to generate non-reasoning data, which is combined with the original SFT data to improve the model's general language capabilities.

- **Distillation**: Combining reasoning data and non-reasoning data for distillation training. Through SFT, the capabilities of DeepSeek-V3 are transferred to a series of smaller models (Qwen and Llama series), resulting in the **DeepSeek-R1-Distill** series models.

- **Final Model Fine-tuning**: The DeepSeek-V3 model is fine-tuned again with SFT and reinforcement learning. In the reinforcement learning stage, reasoning and preference rewards are adopted, and diverse training prompts are used, ultimately resulting in the **DeepSeek-R1** model.

- **DeepSeek-R1-Zero**: Trained directly on DeepSeek-V3 Base using the GRPO algorithm, serving as a comparative baseline for other models.

Next, this blog post will delve into the key technologies and methods in the DeepSeek-R1 training process.

## DeepSeek-R1-Zero

### PPO
**Proximal Policy Optimization (PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) algorithm is a classic algorithm widely used in reinforcement learning. In the InstructGPT ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) paper, it was proven to be an effective and stable method for training LLMs in the reinforcement learning fine-tuning stage.

The core idea of reinforcement learning is to allow an agent to learn through interaction with an environment, maximizing cumulative rewards through trial and error. In the **LLM scenario**, the model itself is the agent, and the "environment" can be understood as the questions raised by users and the expected ways of answering. The policy \( \pi_\theta \) represents the agent's behavior guidelines, i.e., given an input (e.g., question \( q \)), the policy will output an action (e.g., generate text \( o \)). The policy \( \pi_\theta \) is usually parameterized by a neural network model, and the training objective is to find the optimal parameters \( \theta \) so that the policy can generate high-quality outputs.

The Actor-Critic framework is a commonly used architecture in reinforcement learning, and PPO also belongs to the Actor-Critic algorithm family. The Actor-Critic framework includes two core components:

- **Actor (Policy Model)**: Responsible for learning the policy \( \pi_\theta \), i.e., how to choose actions (generate text) based on the current state (e.g., user question).
- **Critic (Value Model)**: Responsible for evaluating the quality of the Actor's policy, usually achieved by learning a value function \( V(s) \) or \( Q(s, a) \). The value function predicts the expected value of cumulative rewards that can be obtained in the future given a state \( s \) (or state-action pair \( (s, a) \)).

The goal of PPO is to improve the policy model (Actor) so that it can generate higher quality outputs, while using the value model (Critic) to stabilize the training process. PPO updates the policy model \( \pi_{\theta} \) by maximizing the following objective function:

\[
\mathcal{J}_{PPO}(\theta)
= \mathbb{E}\!\Biggl[
   \min\Bigl(
     \frac{\pi_\theta(a\!\mid\!s)}{\pi_{\theta_{\text{old}}}(a\!\mid\!s)}A_t,\,
     \operatorname{clip}\Bigl(
        \frac{\pi_\theta(a\!\mid\!s)}{\pi_{\theta_{\text{old}}}(a\!\mid\!s)},
        1-\varepsilon,\,
        1+\varepsilon
     \Bigr)A_t
   \Bigr)
\Biggr]
\]


**Parameter Description:**

- **Expectation \( \mathbb{E}[\cdot] \)**: Represents the average over samples. In actual training, we sample a batch of data (e.g., user questions and model-generated answers) and then calculate the average objective function value for this batch of data.
- **Importance Sampling**: Measures the probability ratio of the current policy \( \pi_\theta \) to the old policy \( \pi_{\theta_{\text{old}}} \) on action \( a \). PPO adopts the idea of **proximal policy update**, limiting the magnitude of each policy update to avoid excessive policy changes that lead to training instability.
- **Advantage Function \( A_t \)**: Evaluates the advantage of taking action \( a \) in state \( s \) relative to the average level. The advantage function is usually estimated by the Critic model (value network), and can be Advantage Estimation or Generalized Advantage Estimation (GAE) and other methods. The larger the advantage function \( A_t \), the better the current action \( a \), and the policy model should increase the probability of taking this action.
- **clip**: One of the core mechanisms of PPO, which can essentially be seen as a penalty function, used to limit the range of the importance sampling between \( [1-\varepsilon, 1+\varepsilon] \), where \( \varepsilon \) is a hyperparameter (usually set to 0.2). The clipping operation prevents excessive policy update steps and improves training stability.

    - The `clip` function penalizes excessively large or small policy update magnitudes by limiting the importance sampling.
        - When the importance sampling  exceeds the range of \( [1-\varepsilon, 1+\varepsilon] \), the `clip` function will limit it within this range, thereby reducing the gain (or reducing the loss) of the objective function.
        - **For positive updates (\( A_t > 0 \)):** If the importance sampling is too large (exceeds \( 1+\varepsilon \)), `clip` will limit it to \( 1+\varepsilon \), **reducing the actual update magnitude and penalizing overly aggressive policy improvements.**
        - **For negative updates (\( A_t < 0 \)):** If the importance sampling is too small (less than \( 1-\varepsilon \)), `clip` will limit it to \( 1-\varepsilon \), **also limiting the update magnitude and avoiding drastic changes in policy.**

    - The objective function takes the minimum value between `clip` before and `clip` after, ensuring that when the importance sampling is out of range, PPO will penalize policy updates, ensuring the "conservatism" of policy updates.

In the actual optimization process, we usually define the PPO loss function \( \mathcal{L}_{PPO}(\theta) \) as the negative value of the objective function, and maximize the objective function by minimizing the loss:

\[
\mathcal{L}_{PPO}(\theta) = -\,\mathcal{J}_{PPO}(\theta).
\]

The PPO algorithm, due to its characteristics of being **simple and effective, and relatively stable**, has become one of the benchmark algorithms in the field of reinforcement learning and has achieved success in various tasks, including reinforcement learning fine-tuning of LLMs. PPO is generally considered more stable than earlier methods such as TRPO, but its specific application in large models still requires careful hyperparameter tuning. In large-scale language model scenarios, if the value network and policy network are completely separated and of comparable size, it will inevitably bring more computational and memory overhead. To solve these problems, the DeepSeek team proposed Group Relative Policy Optimization (GRPO) algorithm.

### GRPO

**Group Relative Policy Optimization (GRPO)** ([Shao, et al., 2024](https://arxiv.org/abs/2402.03300)) is an efficient and stable reinforcement learning algorithm specifically designed by the DeepSeek team for training LLMs like DeepSeek-R1-Zero. GRPO's core innovation lies in abandoning the dependence on an independent value network (critic model) in the traditional Actor-Critic framework, reducing computational costs and improving training stability. Broadly speaking, GRPO can be regarded as an **Actor-Only** reinforcement learning method.

GRPO is inspired by the idea of **relative evaluation**. In many practical scenarios, we are often better at judging the relative quality among a group of things than giving absolute value evaluations. For example, when evaluating a group of student assignments, teachers may find it easier to compare the merits of different assignments than to give each assignment an absolute score. GRPO introduces this idea of relative evaluation into reinforcement learning, using **in-group relative scoring** to build a baseline, completely replacing the dependence on value networks.

Specifically, for each question \( q \), GRPO samples a set of outputs \( \{o_1, o_2, \ldots, o_G\} \) from the old policy \( \pi_{\theta_{\text{old}}} \), forming an **output group**. Then, the policy model \( \pi_{\theta} \) is updated by maximizing the following objective function:

\[
\begin{aligned}
\mathcal{J}_{GRPO}(\theta)
& = \mathbb{E}\left[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)\right] \\
& \quad \frac{1}{G} \sum_{i=1}^G \Biggl(
    \min\biggl(
      \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \,A_i,\,
      \operatorname{clip}\Bigl(
         \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},
         1-\varepsilon,\,
         1+\varepsilon
      \Bigr)\,A_i
    \biggr)
    \;-\;\beta\,\mathbb{D}_{KL}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr)
\Biggr),
\end{aligned}
\]

Similar to the PPO objective function, the GRPO objective function also includes the importance sampling and clip to ensure the stability of policy updates. The differences are:
- **Relative Advantage Value \( A_i \)**: GRPO uses the **relative advantage value** \( A_i \) instead of the advantage function \( A_t \) in PPO. The relative advantage value \( A_i \) is calculated based on **in-group rewards**, without the need for value network estimation.
- **KL Divergence Penalty Term \( \mathbb{D}_{KL}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr) \)**: To further constrain policy updates, GRPO introduces a **KL divergence penalty term**, limiting the difference between the new policy \( \pi_\theta \) and the reference policy \( \pi_{\text{ref}} \) from being too large.

{{< figure
    src="ppo_grpo_comparison.png"
    caption="Fig. 2. The comparison of PPO and GRPO. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2402.03300))"
    align="center"
    width="90%"
>}}

From Figure 2 above, we can see that the core innovation of GRPO lies in the calculation method of the **relative advantage value \( A_i \)**. Unlike PPO, GRPO **does not rely on an independent value network**, but directly uses **in-group rewards** for relative evaluation. For each output group \( \{o_1, o_2, \ldots, o_G\} \), GRPO first obtains the reward values \( \{r_1, r_2, \ldots, r_G\} \) corresponding to each output. Then, the relative advantage value \( A_i \) is calculated according to the following formula:

\[
A_i = \frac{\,r_i \;-\; \text{mean}(\{r_1, r_2, \ldots, r_G\})\,}{
        \text{std}\bigl(\{r_1, r_2, \ldots, r_G\}\bigr)}.
\]

The relative advantage value \( A_i \) is obtained by **standardizing** the in-group rewards \( \{r_1, r_2, \ldots, r_G\} \), with **zero mean and unit variance**, better reflecting the relative merits of each output within the group.

GRPO adopts the method of **relative evaluation**, which has the following advantages:

- **No need to train a value network**: Avoids the computational overhead and instability caused by training a large-scale value network.
- **Reduces variance in value estimation**: Relative evaluation focuses on the relative merits of outputs within the group, rather than absolute values, reducing estimation variance and improving training stability.
- **More consistent with the comparative nature of reward models**: Reward models are usually trained based on comparative data, and GRPO's relative evaluation method is more consistent with this.
- **More suitable for credit assignment in sequence generation tasks**: Even if the reward is sparse, GRPO can learn effectively because it focuses on the relative quality between outputs in the same group.

### Schulman Unbiased Estimator

KL divergence \( \mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right) \) measures the information loss of policy \( \pi_\theta \) relative to the reference policy \( \pi_{\text{ref}} \), and its standard definition is:

\[
\mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right)
= \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \left[ \log \frac{\pi_\theta(o \mid q)}{\pi_{\text{ref}}(o \mid q)} \right].
\]

As mentioned earlier, directly calculating the above expectation in practice faces challenges. To solve this problem, GRPO adopts the Schulman unbiased estimator ([Schulman, 2020](http://joschu.net/blog/kl-approx.html)). Unlike the KL divergence penalty term that may be used in formula, we use the following unbiased estimator to estimate the KL divergence between \( \pi_\theta \) and \( \pi_{ref} \):

$$
\mathbb{D}_{K L}\left[\pi_{\theta}| | \pi_{r e f}\right]=\frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}-\log \frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}-1.
$$

To understand the advantages of this estimator, we first mathematically derive its unbiasedness.

#### Unbiasedness Proof

To simplify the notation, let \( r(o) = \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} \). Then the Schulman estimator can be written as:

\[
\hat{D}_{KL}(o) = r(o) - \log r(o) - 1.
\]

We need to prove that when \( o \) is sampled from \( \pi_\theta(\cdot|q) \), the expectation of \( \hat{D}_{KL}(o) \) is equal to the true KL divergence \( \mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right) \).

\[
\begin{aligned}
\mathbb{E}_{o \sim \pi_\theta(\cdot|q)} [\hat{D}_{KL}(o)]
&= \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} [r(o) - \log r(o) - 1] \\
&= \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \left[ \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - 1 \right] \\
&= \sum_{o} \pi_\theta(o \mid q) \left[ \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - 1 \right]  \quad (\text{Discrete case, integral for continuous case}) \\
&= \sum_{o} \left[ \pi_{ref}(o \mid q) - \pi_\theta(o \mid q) \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - \pi_\theta(o \mid q) \right] \\
&= \underbrace{\sum_{o} \pi_{ref}(o \mid q)}_{=1} - \underbrace{\sum_{o} \pi_\theta(o \mid q) \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)}}_{=-\mathbb{D}_{KL}(\pi_\theta || \pi_{ref})} - \underbrace{\sum_{o} \pi_\theta(o \mid q)}_{=1} \\
&= 1 - (-\mathbb{D}_{KL}(\pi_\theta || \pi_{ref})) - 1 \\
&= \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}).
\end{aligned}
\]

Therefore, we have proven that \( \hat{D}_{KL}(o) \) is an unbiased estimator of \( \mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right) \).

#### Comparison of Three KL Divergence Estimators

To intuitively understand the differences between the three estimators, the following table lists their mathematical expressions, where \( r(o) = \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} \):

| Estimator                     | Mathematical Expression                                               | Main Features                                                                 |
|:------------------------------|:----------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| **k1 (Naive Estimator)**       | \( \hat{D}_{KL}^{(k1)}(o) = \log \frac{\pi_\theta(o \mid q)}{\pi_{ref}(o \mid q)} = \log \frac{1}{r(o)} \) | Simple and direct, corresponds to the definition of KL divergence; high variance, large fluctuations in estimation results. |
| **k2 (Squared Log-Ratio Estimator)** | \( \hat{D}_{KL}^{(k2)}(o) = \frac{1}{2} (\log r(o))^2 \)                 | Uses the square of the log-ratio, always positive, reduces variance; introduces bias, especially when distribution differences are large. |
| **k3 (Schulman Unbiased Estimator)** | \( \hat{D}_{KL}^{(k3)}(o) = r(o) - \log r(o) - 1 \)                     | Combines the ratio \( r(o) \) and the log-ratio \( \log r(o) \); unbiased, low variance, stable estimation. |

- **k1 (Naive Estimator):** Unbiased, simple and direct, but with high variance, leading to unstable estimation results.
- **k2 (Squared Log-Ratio Estimator):** Reduces variance, but introduces bias, especially when the distribution difference is large, the bias is significant.
- **k3 (Schulman Unbiased Estimator):** Combines unbiasedness and low variance, providing stable estimation results.

#### Experimental Results

To evaluate the performance of the three KL divergence estimators, we conducted numerical experiments, and the results are shown in the table below. In the experiment, the distribution \( q = \mathcal{N}(0, 1) \) was fixed, and the mean \( \mu \) of the distribution \( p = \mathcal{N}(\mu, 1) \) was changed to control the true KL divergence \( \mathbb{D}_{KL}(p \| q) \). Monte Carlo estimation was performed using 500 million samples, and the experiment was repeated to obtain stable results.

Experimental code can be found at [unbiased_kl_divergence.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-27-deepseek-r1/unbiased_kl_divergence.py)

| True KL Divergence | Estimator | Average Estimated Value | Standard Deviation | Relative Bias (%) |
|:------------------:|:---------:|:-----------------------:|:------------------:|:-----------------:|
| 0.005              | k1        | 0.005                   | 0.1                | 0.0387            |
| 0.005              | k2        | 0.005                   | 0.0071             | 0.2415            |
| 0.005              | k3        | 0.005                   | **0.0071**         | -0.0082           |
| 0.125              | k1        | 0.125                   | 0.5                | -0.0389           |
| 0.125              | k2        | 0.1328                  | 0.1875             | 6.2500            |
| 0.125              | k3        | 0.125                   | **0.1845**         | 0.0072            |
| 0.5                | k1        | 0.5                     | 1                  | -0.0018           |
| 0.5                | k2        | 0.625                   | 0.866              | 25.0004           |
| 0.5                | k3        | 0.5                     | **0.8478**         | 0.0052            |

- **Naive Estimator (k1):**
  - **Unbiasedness:** The average estimated value is highly consistent with the true KL divergence, and the relative bias is close to 0%.
  - **Variance:** The standard deviation is higher than k3 and increases with the true KL divergence, leading to unstable estimation results.

- **Squared Log-Ratio Estimator (k2):**
  - **Unbiasedness:** There is a certain bias, and the bias increases significantly with the increase of the true KL divergence (for example, when the true KL is 0.5, the relative bias reaches 25%).
  - **Variance:** The variance is lower at lower true KL divergence, but the overall performance is unstable.

- **Schulman Unbiased Estimator (k3):**
  - **Unbiasedness:** The experimental results show that the relative bias is extremely small, almost 0%, which verifies its unbiasedness.
  - **Variance:** The standard deviation is significantly lower than k1, and compared with k1, it shows lower variance under all KL divergences, especially when the KL divergence is low, the advantage is significant.

#### Advantages Summary

- **Unbiasedness:** Both theoretical and experimental results show that k3 is an unbiased estimator, which can accurately reflect the true KL divergence.
- **Positive Definiteness:** The estimated value is always non-negative, which is consistent with the nature of KL divergence.
- **Lower Variance:** Compared with k1, k3 significantly reduces the estimation variance and provides more stable estimation results, especially when the KL divergence is small, the performance is outstanding.

The Schulman unbiased estimator \( \hat{D}_{KL}^{(k3)}(o) = r(o) - \log r(o) - 1 \) provides an estimation method for KL divergence that combines unbiasedness and low variance. Its unbiasedness ensures the accuracy of the estimation, and the lower variance improves the stability of the estimation, especially suitable for reinforcement learning scenarios that require stable gradient signals, such as policy optimization. Based on these advantages, the GRPO algorithm chooses to use k3 as an estimator to penalize policy deviation, thereby ensuring the stability of the training process and the performance of the final policy.

In actual optimization, the GRPO loss function \( \mathcal{L}_{GRPO}(\theta) \) is defined as the negative value of the objective function \( \mathcal{J}_{GRPO}(\theta) \), and the objective function \( \mathcal{J}_{GRPO}(\theta) \) is maximized by minimizing the loss function \( \mathcal{L}_{GRPO}(\theta) \):

\[
\mathcal{L}_{GRPO}(\theta) = -\,\mathcal{J}_{GRPO}(\theta)
\]

### Comparison of PPO and GRPO

To more clearly understand the similarities and differences between PPO and GRPO, the following table compares the two algorithms:

| Feature                 | PPO                                                                 | GRPO                                                                    |
| :---------------------- | :------------------------------------------------------------------ | :---------------------------------------------------------------------- |
| **Actor-Critic or Not** | Yes                                                                 | Yes (broadly considered Actor-Only)                                     |
| **Value Network Needed** | Requires an independent value network (Critic)                        | No independent value network required                                     |
| **Advantage Estimation** | Estimates absolute advantage value through a value network            | Relatively evaluates relative advantage value through in-group rewards     |
| **Computational Cost**    | Higher, requires training a value network                            | Lower, no need to train a value network                                     |
| **Training Stability**    | Relatively good, but value network training may introduce instability | Better, avoids instability from value network training                    |
| **Algorithm Complexity**  | Relatively complex, needs to maintain and update policy and value networks | Relatively simple, only needs to maintain and update the policy network     |
| **Applicable Scenarios**  | Widely applicable to various RL tasks, including fine-tuning of small to medium-sized language models | Especially suitable for RL fine-tuning of large-scale language models, focusing on efficiency and stability |
| **Credit Assignment**     | Relies on value network for temporal difference learning to handle credit assignment issues | Relies on final rewards and in-group relative evaluation, can also be assisted by intermediate rewards |
| **Variance Issue**        | Value network estimation may introduce variance                       | In-group relative advantage estimation may have variance under small group sizes, which can be mitigated by increasing group size, etc. |

As can be seen from the table, PPO is a general and powerful reinforcement learning algorithm, but its mechanism of training a value network brings additional computational burden and potential instability in LLMs scenarios. **GRPO cleverly avoids the need for a value network by introducing in-group relative scoring, significantly reducing computational costs and improving training stability while ensuring performance**. This makes GRPO an ideal choice for training LLMs like **DeepSeek-R1-Zero** when training resources are limited.

### Code Generation Evaluation Metrics

Code generation employs more rigorous testing methods. The code generated by the model is executed through a compiler, and multiple unit tests are performed using predefined test cases to determine the correctness of the code. Commonly used evaluation metrics include **pass@k** ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) and **cons@N** ([OpenAI, 2024](https://openai.com/index/learning-to-reason-with-llms/)).

`pass@k`: Measures the probability that at least one sample out of k code samples generated by the model can pass all predefined test cases.

#### Biased Estimation Formula for pass@k

\[
\text{Simplified pass@k} = \frac{1}{P} \sum_{i=1}^{P} C_i
\]
Where, for each problem \(i\), \(C_i\) is defined as:
\[
C_i = \begin{cases}
    1 & \text{if at least one of the k generated samples is correct} \\
    0 & \text{if all k generated samples are incorrect}
\end{cases}
\]

**Parameter Description:**

*   \( P \): Total number of problems evaluated.
*   \( C_i \): For the \( i \)-th problem, \( C_i = 1 \) if at least one of the \( k \) generated samples is correct, otherwise \( C_i = 0 \).
*   \( \sum_{i=1}^{P} C_i \): Represents the total number of problems "solved" among all \( P \) problems.
*   \( \frac{1}{P} \sum_{i=1}^{P} C_i \): Represents the proportion of "solved" problems, i.e., accuracy.

**Formula Meaning:** This simplified method directly calculates the **proportion of problems for which at least one sample is correct after generating k samples**. Although this method provides a **biased estimate** of pass@k, which may slightly overestimate the true value, it is very commonly used in practice because it is intuitive, easy to calculate, and can provide a reasonable approximation of model performance when the sample size is large enough. Especially in industrial and rapid evaluation scenarios, this simplified method is very practical.

However, LLMs are affected by parameters such as `temperature`, `top_p`, `top_k`, and `repetition_penalty` during reasoning decoding. These parameters can make code generation results random and diverse, and if the parameters are set too randomly when sample K is relatively small, it will affect the evaluation results of pass@k. Therefore, using an unbiased estimation method can more accurately reflect the true performance of the model.

#### Unbiased Estimation Formula for pass@k

\[
\text { pass @ } k:=\underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
\]

**Parameter Description:**

*   \( n \): Total number of code samples generated for each problem.
*   \( c \): Number of correct samples among the \( n \) samples that can pass all unit tests.
*   \( k \): Parameter \( k \) in the pass@\(k\) metric, indicating the number of generated samples we consider.
*   \( \binom{a}{b} \): Represents the binomial coefficient, calculating the number of combinations of choosing \( b \) elements from \( a \) elements.
*   \( \underset{\text { Problems }}{\mathbb{E}} \): Represents the expected value (average value) over all evaluation problems.

**Formula Meaning:**
- The formula actually calculates the probability of having at least one correct sample. The formula \( \frac{\binom{n-c}{k}}{\binom{n}{k}} \) calculates the probability of randomly selecting \( k \) samples from the generated \( n \) samples, and none of these \( k \) samples are correct. We subtract this probability from 1 to get the probability of randomly selecting \( k \) samples from \( n \) samples, and at least one of these \( k \) samples is correct, which is the meaning of the pass@\(k\) metric.
- This formula provides an **unbiased estimate** of pass@k, which is more suitable for scenarios requiring precise evaluation such as academic research. In actual calculations, a sample size \( n \) much larger than \( k \) is usually generated (for example, \( n=200 \), \( k \leq 100 \) is used in papers) to more stably estimate pass@\(k\).

#### Simplified Product Form of pass@k

For easier numerical calculation, the original formula can also be converted into the following product form, which is still an unbiased estimate and can avoid numerical overflow problems:

\[
\text { pass @ } k = \underset{\text { Problems }}{\mathbb{E}}\left[1 - \prod_{i=0}^{k-1} \frac{n-c-i}{n-i}\right]
\]

**Derivation Process:**

1. The opposite of having at least one correct sample is that all k samples are incorrect. Therefore, pass@k is equal to 1 minus the probability that all k samples are incorrect.

2. Consider the scenario of **sampling without replacement**. Assume we draw \( k \) samples from \( n \) samples, and we want to calculate the probability that all \( k \) samples are incorrect. There are a total of \( n \) samples, of which \( n-c \) are incorrect.

3. When drawing for the first time, the probability of drawing an incorrect sample is \( \frac{n-c}{n} \).

4. Given that an incorrect sample was drawn in the first draw, when drawing for the second time, among the remaining \( n-1 \) samples, there are \( n-c-1 \) incorrect samples. Therefore, the conditional probability of still drawing an incorrect sample for the second time is \( \frac{n-c-1}{n-1} \).

5. By analogy, when drawing for the \( i \)-th time ( \( i \) from 1 to \( k \)), given that incorrect samples were drawn in the previous \( i-1 \) times, the conditional probability of still drawing an incorrect sample for the \( i \)-th time is \( \frac{n-c-(i-1)}{n-(i-1)} = \frac{n-c-i+1}{n-i+1} \). To align with the index \( i=0 \) in the formula, we change the index to range from \( i=0 \) to \( k-1 \), then when drawing for the \( (i+1) \)-th time ( \( i \) from 0 to \( k-1 \)), the conditional probability is \( \frac{n-c-i}{n-i} \).

6. Multiply these conditional probabilities of \( k \) draws to get the probability that all \( k \) samples are incorrect:

    \[
    P(\text{all k samples are incorrect}) = \frac{n-c}{n} \times \frac{n-c-1}{n-1} \times \cdots \times \frac{n-c-k+1}{n-k+1} = \prod_{i=0}^{k-1} \frac{n-c-i}{n-i}
    \]

7. Finally, the simplified formula for pass@k is:

    \[
    \text { pass @ } k = \underset{\text { Problems }}{\mathbb{E}}\left[1 - \prod_{i=0}^{k-1} \frac{n-c-i}{n-i}\right]
    \]

This product form formula avoids directly calculating binomial coefficients that may be numerically large, is easier to understand and numerically calculate, especially in programming implementation, it can be multiplied term by term to effectively prevent numerical overflow.

#### cons@N
`cons@N`: By generating N samples and selecting the answer with the highest frequency as the final answer, the accuracy of this answer is evaluated. In the evaluation of DeepSeek-R1-Zero, **cons@64** was used, i.e., 64 samples were generated, and the answer that appeared most frequently among them was taken as the final answer for evaluation.

\[
\text{cons@N} = \frac{1}{P} \sum_{i=1}^{P} \mathbb{I}(\text{ConsensusAnswer}_i \text{ is correct})
\]

**Parameter Description:**

- \( P \): Total number of problems evaluated.
- \( \text{ConsensusAnswer}_i \): **Consensus answer** obtained through majority voting.
- \( \mathbb{I}(\text{ConsensusAnswer}_i \text{ is correct}) \): Indicator function, 1 if the consensus answer is correct, 0 otherwise.

**Formula Meaning:** Calculate the proportion of consensus answers that are correct among all evaluation problems. By increasing the number of generated samples \( N \) and adopting a majority voting strategy, the cons@N metric can more stably and reliably evaluate the average performance of the model. In cases where the model's generated results have a certain degree of randomness, this metric can verify the consistency and accuracy of the model's output.

### Reward Model

Reward models are crucial in the development of LLMs, mainly used in the following key stages:

- **Reinforcement Learning from Human Feedback**: In the Reinforcement Learning from Human Feedback (RLHF) process, reward models are used to evaluate the quality of model-generated results and provide reward signals for subsequent reinforcement learning.

- **Key Tool for Rejection Sampling**: In the rejection sampling process, reward models score a large number of candidate results and filter out high-quality samples for SFT. Rejection sampling is an important method for automated sample engineering, and reward models are its core component.

- **Discriminator in Business Scenarios**: In practical applications, reward models serve as discriminators or validators of LLM output results, evaluating the quality of generated results. Only results with scores exceeding a preset threshold will be output, otherwise, regeneration or degradation processing will be performed to improve the reliability and safety of the output.

#### ORM vs PRM
{{< figure
    src="orm_prm_comparison.png"
    caption="Fig. 3. Outcome reward vs Process reward. (Image source: [Zeng et al., 2024](https://arxiv.org/abs/2412.14135))"
    align="center"
    width="100%"
>}}

Current reward models are mainly divided into two paradigms: **Outcome Reward Model (ORM)** and **Process Reward Model (PRM)**. Figure 3 above intuitively shows the difference between these two reward models. The following table also compares the main characteristics of these two models:

| Feature                     | **ORM**                                                                 | **PRM**                                                                   |
|-----------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Definition**              | Holistically scores the complete result generated by the model           | Provides fine-grained scoring for each step or stage during content generation |
| **Main Advantages**         | Simple and direct, easy to implement <br> Comprehensive evaluation of overall results | Provides more refined reward signals <br> Helps guide each step of the model generation process |
| **Main Disadvantages**        | High variance, large fluctuations in estimation results <br> Lack of feedback during the process | More complex to train and apply <br> May introduce bias, especially when distribution differences are large |
| **Applicable Scenarios**      | Tasks requiring overall evaluation of generated results                 | Tasks requiring fine-grained control of the generation process, such as step-by-step reasoning or complex generation tasks |
| **Ability to Avoid Reward Hacking** | Medium, depends on the accuracy of overall scoring                      | Lower, can cheat by optimizing rewards for each step rather than overall performance |
| **Training Complexity**       | Lower, no need for additional supervision of the generation process      | Higher, needs to score at each step of generation, increasing computational and data requirements |
| **Explainability**            | High, scoring is based on the final result                              | Lower, scoring involves multiple steps of the generation process, difficult to fully understand the scoring basis for each step |

#### Adopting ORM

To train DeepSeek-R1-Zero, the DeepSeek team chose **ORM** instead of PRM. This choice is based on the following considerations:

- **Avoiding Reward Hacking**
  PRM is prone to being exploited by agents in large-scale RL training, leading to reward hacking ([Gao et al., 2022](https://arxiv.org/abs/2210.10760)). Models may adopt "shortcuts" to maximize rewards instead of improving reasoning ability. Rule-based reward systems effectively avoid reward hacking problems through clear and interpretable rules.

  > Rule-based reward systems may be difficult to cover all types of questions when the problem scenario is complex or creative answers are required, and rule design may have loopholes that can be exploited by the model.

- **Reducing Training Complexity**
  Training PRM requires a lot of computing resources and data, increasing the complexity of the training process. Rule-based reward systems, on the other hand, do not require additional training, and rules can be directly applied once determined, simplifying the training process. Rule-based reward systems are particularly suitable for tasks with automatic scoring or clear objectives, such as math problems, LeetCode programming problems, and tasks with clear output format requirements. For open-ended dialogue or creative tasks, it may be necessary to combine human feedback or trained reward models.

#### Reward Mechanism

The reward system of DeepSeek-R1-Zero adopts a dual reward mechanism, which is automatically evaluated through predefined rules to ensure the efficiency and real-time nature of the evaluation process. This system mainly includes the following two types of rewards:

**1. Accuracy Reward**

* **Definition:** Measures the correctness of the model output result, which is the most critical part of the reward system.
* **Implementation Method:** Different verification methods are adopted according to different task types:
    * **Math Problems:** Verify whether the final answer is consistent with the standard answer.
    * **Code Generation:** Execute the code generated by the model through a compiler, and use preset unit test cases for multiple tests to determine the correctness of the code.
* **Purpose:** Guide the model to generate accurate and reliable output results.

**2. Format Reward**

* **Definition:** A reward mechanism introduced to improve the readability and structure of model output, facilitating subsequent analysis and evaluation.
* **Evaluation Method:** Automatically evaluated by a predefined rule system during reinforcement learning training.
* **Purpose:** Encourage the model to generate structured output, such as including the thinking process and the final answer, making it easier to understand and analyze.


The reward function \( r(o) \) of DeepSeek-R1-Zero consists of a weighted sum of accuracy reward and format reward:

$$
r(o) = r_{\text{accuracy}}(o) + \lambda \cdot r_{\text{format_effective}}(o)
$$

Where, the effective format reward \( r_{\text{format_effective}}(o) \) is calculated as follows:

$$
r_{\text{format_effective}}(o) =
\begin{cases}
    r_{\text{format}}(o) & \text{if the basic format of } o \text{ meets the requirements} \\
    0 & \text{if the basic format of } o \text{ does not meet the requirements}
\end{cases}
$$

The basic format reward \( r_{\text{format}}(o) \) is graded according to the degree of compliance with the format specification:

$$
r_{\text{format}}(o) =
\begin{cases}
    R_{\text{format_full}} & \text{if the format of } o \text{ fully complies with the specification} \\
    R_{\text{format_partial}} & \text{if the format of } o \text{ partially complies with the specification} \\
    0 & \text{if the format of } o \text{ does not comply with the specification}
\end{cases}
$$

### Experimental Process
#### Training Template

To guide the base model to follow specified instructions, the DeepSeek team designed a concise and effective training template. This template requires the model to first generate the reasoning process (placed between `<think>` and `</think>` tags), and then provide the final answer (placed between `<answer>` and `</answer>` tags). This structured format not only ensures the readability of the output, but also allows researchers to clearly observe the model's reasoning process during RL training, thereby more accurately assessing the model's learning progress.

| Role | Prompt Content           | Assistant Reply                                   |
| :--- | :--------------------- | :------------------------------------------------ |
| User | prompt (user question) | Assistant: `<think> Reasoning Process </think>` `<answer> Answer </answer>` |

- `<think>` and `</think>` (Thinking Process Tags): Used to wrap the model's intermediate reasoning steps, clearly showing the model's thinking process, facilitating understanding of the model's reasoning logic and error analysis.
- `<answer>` and `</answer>` (Final Answer Tags): Used to wrap the model's final answer, facilitating program automation to extract the answer part for efficient evaluation and subsequent processing.

#### Evaluation Process

1. **Accuracy Evaluation:** Evaluate whether the answer of the model output \( o \) is correct, and calculate the accuracy reward \( r_{\text{accuracy}}(o) \).
2. **Basic Format Check:** Check whether the basic format of the output \( o \) meets predefined requirements, such as whether it contains necessary tags `<think>` and `<answer>`, and whether the tags are correctly closed and nested.
3. **Effective Format Reward Judgment:**
    * **Basic format does not comply:** Effective format reward \( r_{\text{format_effective}}(o) = 0 \).
    * **Basic format complies:** Further evaluate the degree of format specification compliance, and calculate the basic format reward \( r_{\text{format}}(o) \).
4. **Final Reward Calculation:** Linearly weight and sum the accuracy reward \( r_{\text{accuracy}}(o) \) and the effective format reward \( r_{\text{format_effective}}(o) \) to obtain the final reward \( r(o) \).

By combining accuracy reward and format reward, the reward system of DeepSeek-R1-Zero not only focuses on the correctness of the model output, but also emphasizes the structuredness and readability of the output results. This enables the model to not only give correct answers, but also show its thinking process, making it more like an intelligent agent with reasoning ability, rather than just a simple answer output machine.

#### Experimental Results

{{< figure
    src="deepseek_r1_zero_benchmark.png"
    caption="Fig. 4. Comparison of DeepSeek-R1-Zero and OpenAI o1 models on reasoning-related benchmarks. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

Figure 4 shows the performance of different models on multiple benchmarks. In the [AIME 2024](https://maa.org/student-programs/amc/) benchmark, the pass@1 score of the DeepSeek-R1-Zero model reached 71.0%, and the cons@64 score was 86.7%, comparable to the OpenAI o1-0912 model.

{{< figure
    src="deepseek_r1_zero_response_time.png"
    caption="Fig. 5. The average response length of DeepSeek-R1-Zero on the training set during the RL process. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

Figure 5 shows that as training deepens, the DeepSeek-R1-Zero model exhibits the ability of spontaneous **self-evolution**. The model dynamically allocates "thinking time" according to the complexity of the question. For more complex questions, it spontaneously generates longer reasoning chains for deeper thinking. This adaptive adjustment of "thinking time" is not artificially set, but an emergent behavior of the model in the RL training process, fully reflecting the autonomous improvement of the model's reasoning ability driven by reinforcement learning.

## DeepSeek-R1

### Training Process

To further improve model performance based on DeepSeek-R1-Zero, the DeepSeek team adopted a **multi-stage training** strategy and incorporated **cold-start data** into the training process. The training process of DeepSeek-R1 mainly includes the following four stages, reflecting the progressive path from initial policy initialization to comprehensive capability improvement:

1. **Cold Start**: Using high-quality long Chain-of-Thought (CoT) data, perform preliminary SFT on the DeepSeek-V3-Base base model to lay the foundation for subsequent reinforcement learning.

2. **Reasoning-Oriented RL**: Based on the cold-start model, apply reinforcement learning algorithms, focusing on enhancing the model's ability in reasoning-intensive tasks.

3. **Rejection Sampling & SFT**: Filter high-quality reasoning data through rejection sampling technology, and combine it with non-reasoning data for SFT to further improve the model's reasoning ability and general ability.

4. **All-Scenario RL**: Comprehensively consider reasoning and non-reasoning tasks, and conduct the second stage of reinforcement learning to align the model with human preferences and improve performance in a wider range of scenarios.

### Cold Start

In the training process of DeepSeek-R1, the **cold start** stage is crucial, like the igniter of an engine, laying a solid foundation for the subsequent complex reinforcement learning process. SFT is the core technology of the cold start stage.

#### Training Objective

The objective of the cold start stage is clear and critical: using high-quality Chain-of-Thought (CoT) data to perform preliminary fine-tuning on the DeepSeek-V3-Base base model. This fine-tuning aims to quickly endow the model with the following core capabilities:

* **Preliminary Reasoning Ability:** Guide the model to learn to imitate the human reasoning process, laying the foundation for more complex reasoning.
* **Good Text Generation Quality:** Ensure the fluency and naturalness of the text output by the model, improving the user experience.

These CoT data are like the model's "starting fuel", helping the model quickly grasp human reasoning patterns and providing **good policy initialization** for subsequent reinforcement learning, effectively **avoiding the inefficiency and instability of RL training starting from scratch in the early stage**.

#### Data Construction

To construct high-quality cold-start data, the DeepSeek team conducted multi-faceted explorations and finally integrated the following efficient methods:

* **Few-shot Prompting:** Using a small number of high-quality examples to guide the model to generate longer, deeper, and more logical CoT data.
* **Model Generation + Reflection Verification:** Directly prompt the model to generate answers, and add reflection and verification links to ensure the quality of answers and the correctness of reasoning.
* **Optimize R1-Zero Output:** Collect the output of the DeepSeek-R1-Zero model, and improve the readability and overall quality of the data through manual annotation and optimization.

Through the above strategies, the DeepSeek team accumulated **thousands of high-quality cold-start data**, and used this as a basis to fine-tune DeepSeek-V3-Base as a solid starting point for reinforcement learning.

#### Advantages of Cold Start

Compared to directly using DeepSeek-R1-Zero as a starting point, cold-start data brings several significant advantages, laying a better foundation for subsequent training:

* **Improved Readability:**
    * The output of DeepSeek-R1-Zero has readability challenges, such as language mixing, lack of structured format, etc.
    * Cold-start data is specially designed with a **more readable output mode**, including:
        * **Adding Summary:** Add a refined summary at the end of the reply to quickly extract core conclusions.
        * **Filtering Bad Replies:** Remove unfriendly or low-quality replies to ensure data purity.
        * **Structured Output Format:** Adopt the `| special_token | <reasoning_process> | special_token | <summary>` format to clearly present the reasoning process and summary.

* **Enhanced Performance:**
    * By carefully designing data patterns that incorporate human prior knowledge, the DeepSeek team observed a significant improvement in model performance compared to R1-Zero.
    * This further verifies that iterative training is an effective path to improve the performance of reasoning models.

* **Superior Policy Initialization:**
   * **The core of SFT in the cold start stage is policy initialization.** Policy initialization is a key step in building Reasoning LLMs, such as the OpenAI o1 series. By learning high-quality CoT data, the model initially grasps human reasoning patterns and has the ability to generate structured reasoning processes, laying a solid foundation for subsequent reinforcement learning training and avoiding the dilemma of starting exploration from scratch.

### SFT
The core objective of **Supervised Fine-tuning (SFT)** is to fine-tune the model on supervised labeled data so that its predictions are as close as possible to the true labels. This aims to improve the model's ability in specific tasks and instruction execution.

#### Loss Function

The training objective of SFT is to minimize the difference between model predictions and true labels. The loss function usually adopts **Cross-Entropy Loss**, also known as **Negative Log-Likelihood**, to measure the difference between the model's predicted token distribution and the true token distribution. To balance the contributions of output sequences of different lengths, we usually normalize the loss function to the average loss per token.

The loss function formula is as follows:

\[
\mathcal{L}_{SFT}(\theta) = - \mathbb{E}_{(q, o) \sim P_{sft}(Q, O)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \log \pi_\theta\left(o_t \mid q, o_{&lt;t}         \right)\right]
\]

**Parameter Description:**

* **\( \mathcal{L}_{SFT}(\theta) \)**: SFT loss function, minimized by adjusting model parameters \( \theta \).
* **\( \mathbb{E}_{(q, o) \sim P_{sft}(Q, O)}[\cdot] \)**: Expectation over the SFT dataset distribution \( P_{sft}(Q, O) \).
* **\( P_{sft}(Q, O) \)**: SFT dataset distribution, \( q \) represents the question (Query), and \( o \) represents the corresponding standard answer output (Output).
* **\( (q, o) \)**: Question-answer pair sampled from the SFT dataset.
* **\( |o| \)**: Token length of the standard answer output.
* **\( o_t \)**: The \( t \)-th token of the standard answer output.
* **\( o_{&lt;t}   \)**: The first \( t-1 \) tokens of the standard answer output.
* **\( \pi_\theta\left(o_t \mid q, o_{&lt;t}   \right) \)**: Given the question \( q \) and the preceding text \( o_{&lt;t}   \), the probability of the model predicting token \( o_t \).
* **\( \frac{1}{|o|} \)**: Length normalization factor, dividing the total loss by the output sequence length to get the average loss per token.

The SFT loss function aims to penalize deviations between model predictions and standard answers. For a given question \( q \) and standard answer \( o \), the loss function calculates the probability \( \pi_\theta(o_t | q, o_{&lt;t}   ) \) of the model predicting each token \( o_t \) in the answer \( o \). By dividing by the output length \( |o| \), the loss function is normalized to the average negative log-likelihood per token.

* **When the model accurately predicts the standard answer token**, \( \pi_\theta(o_t \mid q, o_{&lt;t}   ) \approx 1 \), \( \log \pi_\theta(o_t \mid q, o_{&lt;t}   ) \approx 0 \), and the loss value is close to the minimum.
* **When the model prediction deviates from the standard answer**, \( \pi_\theta(o_t \mid q, o_{&lt;t}   ) \) is smaller, \( \log \pi_\theta(o_t \mid q, o_{&lt;t}   ) \) is negative and has a larger absolute value, and the loss value increases.

The process of minimizing the SFT loss function is the process of making the model learn to generate text as similar as possible to the standard answers in the training dataset. From the perspective of negative log-likelihood, the goal is to find the optimal model parameters \( \theta \) to maximize the probability of the model generating the answer \( o \) in the training data, which is equivalent to minimizing the negative log-likelihood of generating the answer \( o \). High-quality CoT data contains human preferences for reasoning and results, so SFT can also be regarded as a process of making the model learn and fit human reasoning preferences.

#### Gradient

The gradient of the SFT loss function is used to guide model parameter updates to reduce the loss value. The gradient of the loss function with respect to the model parameters \( \theta \) is:

\[
\nabla_{\theta} \mathcal{L}_{SFT} = - \mathbb{E}_{(q, o) \sim P_{sft}(Q, O)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \nabla_{\theta} \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}         \right)\right]
\]

**Parameter Description:**

* **\( \nabla_{\theta} \mathcal{L}_{SFT} \)**: Gradient of the SFT loss function with respect to parameter \( \theta \), indicating the direction in which the loss function value decreases fastest.
* **\( \nabla_{\theta} \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}   \right) \)**: Gradient of the token probability logarithm \( \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}   \right) \) with respect to parameter \( \theta \).
* **\( \frac{1}{|o|} \)**: **Length normalization factor**, consistent with the loss function, the gradient is also the **gradient of the average loss per token**.

When actually calculating the gradient, stochastic gradient descent algorithm is usually used to update the model parameters along the gradient descent direction, gradually minimizing the loss function and improving the accuracy of the model in generating standard answers.

**Gradient Coefficient**

In the SFT stage, the gradient coefficient is usually set to 1, which means that **all training samples contribute equally to the update of model parameters**. The model learns each example equally, striving to minimize the average loss over the entire dataset.

#### Data Source and Human Preference

* **Data Source**: The SFT dataset mainly consists of high-quality long Chain-of-Thought (CoT) examples, representing the "standard answers" that the model is expected to learn, used to guide the minimization of the loss function. Data may come from manual annotation or generation by more powerful models. Refer to the SFT dataset [OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT) of the [Open-o1](https://github.com/Open-Source-O1/Open-O1?tab=readme-ov-file) project, which contains long CoT replies.
* **Human Preference**: In the SFT stage, human selection can be regarded as an implicit reward function. High-quality CoT data reflects human expectations for model reasoning and output. By learning these data, the model minimizes the deviation from human expected output, thereby fitting human preferences.

### Reasoning-Oriented Reinforcement Learning

After cold-start fine-tuning, the DeepSeek team further improved the model's ability in reasoning-intensive tasks (such as coding, mathematics, science, and logical reasoning) through reinforcement learning (RL). The core of this stage is to **maximize the reward function, guiding the model to learn more effective reasoning strategies**.

#### Reward Function

To solve the problem of CoT language mixing during reasoning, the DeepSeek team introduced **language consistency reward** and combined it with **task reward** to form the total reward function:

\[
r(o) = r_{\text{task}}(o) + \alpha \cdot r_{\text{lang_consistency}}(o)
\]

**Parameter Description:**

* **\( r(o) \)**: Total reward function, the goal of RL training is to maximize this function.
* **\( r_{\text{task}}(o) \)**: Task reward based on task completion, measuring the accuracy of model reasoning.
* **\( r_{\text{lang_consistency}}(o) \)**: Language consistency reward, measuring the language purity of CoT output.
* **\( \alpha \)**: Hyperparameter, balancing the weights of task reward and language consistency reward.

The total reward function is the weighted sum of task reward and language consistency reward. Maximizing \( r(o) \) drives the model to improve reasoning accuracy while maintaining the language consistency of CoT output. The role of \( \alpha \) is to adjust the model's emphasis on language consistency.

#### Training Objective

By maximizing the above reward function, the DeepSeek team conducted RL training on the model after cold-start fine-tuning, **optimizing model parameters to obtain higher reward values in reasoning tasks, and ultimately improving reasoning ability**.

### RFT

**Rejection Sampling Fine-tuning (RFT)** aims to improve the general ability of the model by refining training data. Its core idea is to minimize the selective loss function, guiding the model to learn the generation patterns of high-quality outputs.

#### Loss Function

RFT adopts a **rejection sampling** strategy to distinguish the generation and selection processes of **reasoning data** and **non-reasoning data**, and constructs a high-quality SFT dataset. The training objective is to minimize the following loss function:

\[
\mathcal{L}_{RFT}(\theta) = - \mathbb{E}_{(q, o) \sim P_{sft}(Q) \times \pi_{sft}(O \mid q)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \mathbb{I}(o) \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}    \right)\right]
\]

Where, the indicator function \( \mathbb{I}(o) \) is defined as:

\[
\mathbb{I}(o) = \begin{cases}
    1, & \text{if output } o \text{ is judged to be high quality} \\
    0, & \text{otherwise}
\end{cases}
\]

**Parameter Description:**

* **\( \mathcal{L}_{RFT}(\theta) \)**: RFT loss function.
* **\( P_{sft}(Q) \)**: Distribution of question \( q \).
* **\( \pi_{sft}(O \mid q) \)**: Conditional probability distribution of the SFT model generating output \( O \) given question \( q \).
* **\( \mathbb{I}(o) \)**: Indicator function, used to select high-quality answers. It is 1 when output \( o \) is judged to be high quality, and 0 otherwise.

The RFT loss function is based on cross-entropy loss, and **selectively learns high-quality outputs** through the indicator function \( \mathbb{I}(o) \):

* **High-quality output (\( \mathbb{I}(o) = 1 \)):** The loss function degenerates into standard cross-entropy loss, and the model updates parameters based on the negative log-likelihood of high-quality answers, minimizing the difference between model predictions and high-quality answers.
* **Low-quality output (\( \mathbb{I}(o) = 0 \)):** The loss function is zero, and low-quality answers do not participate in parameter updates.

RFT guides the model to focus on learning the generation patterns of high-quality answers by minimizing the loss function, achieving selective learning.

#### Data Generation

* **High-quality data (reasoning data):** Generate candidate answers through the RL model, use a reward model (or DeepSeek-V3 model) to score, and **reject sample to retain high-score answers**.
* **SFT data (non-reasoning data):** Reuse the SFT dataset of DeepSeek-V3 and its generation process.

#### Training Process

- Use the high-quality dataset obtained by rejection sampling to perform SFT on the DeepSeek-V3-Base model, **minimize the RFT loss function, and improve the model's reasoning and general abilities**.

- RFT iteratively refines data and retrains the model, expecting the model to learn higher quality data patterns in each iteration, and finally converge to a high-quality output model. In the iterative process, the training data distribution \( P_{sft}(Q, O) \) gradually focuses on high-quality data, enabling the model to continuously improve its ability to generate high-quality outputs in the process of loss minimization.

### OnRFT

**Online Rejection Sampling Fine-tuning (OnRFT)** has a similar objective to RFT, both aiming to learn high-quality output patterns by minimizing the selective loss function. The main difference between OnRFT and RFT is the data sampling method, and the loss function form is consistent with RFT. The gradient of the OnRFT loss function is:

\[
\nabla_{\theta} \mathcal{L}_{OnRFT}(\theta) = - \mathbb{E}_{(q, o) \sim P_{sft}(Q) \times \pi_{\theta}(O \mid q)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \mathbb{I}(o) \nabla_{\theta} \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}     \right)\right]
\]

**Parameter Description:**

* **\( \nabla_{\theta} \mathcal{L}_{OnRFT} \)**: Gradient of the OnRFT loss function with respect to model parameter \( \theta \), indicating the direction of loss function decrease.
* **\( \pi_{\theta}(O \mid q) \)**: Conditional probability distribution of the **current training model** generating output \( O \) given question \( q \).

### Comparison of RFT and OnRFT

The table below briefly compares the main differences between RFT and OnRFT.

| Feature                 | RFT                                     | OnRFT                                        |
| ----------------------- | --------------------------------------- | -------------------------------------------- |
| **Data Generation Method** | Offline                                 | Online                                       |
| **Data Generation Model** | SFT model \( \pi_{sft} \)              | Current training model \( \pi_{\theta} \)      |
| **Rejection Sampling Data Source** | Pre-generated SFT dataset             | Real-time data generation during training     |
| **Data Loop**           | Separated                               | Online loop                                  |
| **Loss Function Mechanism** | Selective cross-entropy loss, selects high-quality output for learning | Selective cross-entropy loss, selects high-quality output for learning |
| **Training Data Distribution Change** | Gradually focuses on high-quality data | Dynamic change, fits current model capability    |

### All-Scenario Reinforcement Learning

To further align with human preferences, the DeepSeek team conducted the second stage of RL, aiming to improve the model's Helpfulness and Harmlessness while maximizing the reward function, and also taking into account reasoning ability. This stage still uses maximizing the reward function to guide model training, but the design of the reward function is more complex to reflect multi-dimensional optimization goals.

The RL training in this stage combines:

* **Diverse Prompt Distribution:** Covers a wider range of scenarios, including reasoning and general tasks.
* **Multi-objective Reward Signals:**
    * **Reasoning Data:** Follows the rule-based task reward, focusing on reasoning accuracy. Maximize task reward to guide the model to minimize reasoning errors.
    * **General Data:** Uses a reward model to capture human preferences for helpfulness and harmlessness. The goal of the reward model is to learn human preferences and output reward signals consistent with human preferences. The goal of RL training is to maximize the reward value given by the reward model, thereby indirectly minimizing the deviation between model output and human preferences.

### Distillation

To transfer the powerful reasoning ability of DeepSeek-R1 to more efficient small models, the DeepSeek team adopted **Distillation** ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) technology. The distillation process mainly includes the following steps:

1. **Data Generation**: Use the trained DeepSeek-R1 model to generate about **800,000** high-quality reasoning data. These data not only include reasoning-intensive tasks (such as math problems, programming problems), but also cover general tasks (such as question answering, dialogue) to ensure the diversity and coverage of distillation data.

2. **Model Fine-tuning**: Use the generated 800,000 high-quality reasoning data to fine-tune small dense models. Distillation experiments selected Qwen and Llama series models as Student models, covering multiple model scales from 1.5B to 70B parameters to explore the effect of distillation technology under different model scales. The selected Student models include Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, and **Llama-3.3-70B-Instruct**.

3. **Performance Evaluation**: Conduct a comprehensive performance evaluation of the distilled models in multiple reasoning-related benchmarks. The evaluation results are intended to verify whether distillation technology can effectively transfer the reasoning ability of large models to small models, and to investigate whether the reasoning ability of distilled small models can reach or even exceed the level of large models.

#### KL Divergence Distillation

In addition to directly using the text output generated by the Teacher model as pseudo-labels for SFT distillation, a more rigorous method is to also consider the token probability distribution \( \pi_{\text{teacher}} \) generated by the Teacher model. **KL divergence distillation** is a commonly used method, which not only allows the Student model to learn the text output of the Teacher model, but also learns the token probability distribution of the Teacher model. By minimizing the KL divergence between the output probability distributions of the Student model and the Teacher model, the knowledge of the Teacher model can be more fully transferred to the Student model. However, in actual engineering, directly using the text output of the Teacher model as pseudo-labels for SFT distillation can usually achieve sufficiently good results and is simpler to implement.

#### Experimental Results

The experimental results are shown in Figure 6:

{{< figure
    src="deepseek_r1_distill_comparison.png"
    caption="Fig. 6. Comparison of DeepSeek-R1 distilled models and other comparable models on reasoning-related benchmarks. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

The experimental results show that this **direct SFT distillation method can significantly improve the reasoning ability of small models**. Especially the distilled Llama-3.3-70B-Instruct model outperforms the o1 model on multiple benchmarks, achieving such significant results through SFT distillation alone. Here, the blogger is somewhat skeptical that the model has overfitted on these benchmarks or there is a risk of data leakage, and further testing on more benchmarks is needed in the future.

## Discussion

DeepSeek-R1, based on a multi-stage training framework, explores a simplified path for Reasoning Model training technology, mainly including the following points:

**Linearized Thinking Process: CoT Replaces MCTS**
- Traditional reinforcement learning AI, such as Go and chess, once relied on Monte Carlo Tree Search (MCTS). DeepSeek-R1 and other models explore the use of autoregressive chain-of-thought methods to simplify the reasoning process, gradually abandoning computationally complex MCTS.
- CoT decomposes complex reasoning into linear steps, and the model reasons step by step like solving a problem, rather than the exhaustive search of MCTS. This linearized thinking reduces computational complexity, is more in line with human thinking habits, and makes it easier for models to learn complex reasoning strategies.

**Eliminating Independent Value Networks: Simplifying RL Architecture**
- Traditional reinforcement learning (such as PPO) usually requires independent policy networks and value networks. DeepSeek-R1 and other studies have found that strengthened policy networks or simplified value evaluation methods (such as GRPO's in-group relative scoring) can replace independent value networks.
- This simplifies the RL training architecture, reduces resource requirements, and improves efficiency. It shows that the policy network of LLMs already has strong value evaluation capabilities, and no additional value network is needed.

**Focusing on Outcome Rewards: Minimizing Reward Signals**
- DeepSeek-R1 adopts a simpler ORM reward strategy, mainly focusing on the accuracy reward of the final result, weakening the reward for intermediate reasoning steps. This strategy is inspired by AlphaZero ([Silver et al., 2017](https://arxiv.org/abs/1712.01815)), which only focuses on winning or losing.
- For Reasoning Models, outcome rewards may be more effective than PRM, which can help models learn "ways of thinking" more naturally and reduce cumbersome step-by-step supervision.

**Increasing Thinking Time: Model Spontaneously Emerges Deep Thinking**
- DeepSeek-R1-Zero training shows the ability to spontaneously **increase thinking time**. As training deepens, the model adaptively allocates more "thinking time" according to the complexity of the question, generating longer reasoning sequences. This increase in "thinking time" is an emergent behavior of the model in RL training.
- Increasing thinking time reflects the model's deeper exploration and optimization of the thinking process. Complex problems require more reasoning steps to find answers. The self-evolution ability of DeepSeek-R1-Zero confirms the potential of reinforcement learning in improving model reasoning ability.

## Summary

The success of DeepSeek-R1 demonstrates the great potential of RL in improving the reasoning ability of LLMs. The GRPO algorithm adopted by DeepSeek-R1 is superior to PPO and DPO in terms of computational efficiency, optimization stability, and reward robustness, and reduces training resource consumption by simplifying the model architecture. DeepSeek-R1 provides a path worth referencing for open-source Reasoning Model replication of o1.

## References

[1] [OpenAI O1](https://openai.com/o1/) [Website]. OpenAI, 2024.  *(OpenAI O1 official introduction page)*

[2] Jaech A, et al. [Openai o1 system card](https://arxiv.org/abs/2412.16720) [J]. *arXiv preprint arXiv:2412.16720*, 2024.

[3] [Open-r1](https://github.com/huggingface/open-r1) [Website]. GitHub, 2024. *(Open-r1 open source project GitHub repository)*

[4] Sutton R. [The bitter lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) [J]. *Incomplete Ideas (blog)*, 2019, 13(1): 38.

[5] Liu A, et al. [Deepseek-v3 technical report](https://arxiv.org/abs/2412.19437) [J]. *arXiv preprint arXiv:2412.19437*, 2024.

[6] Schulman J, et al. [Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347) [J]. *arXiv preprint arXiv:1707.06347*, 2017.

[7] Ouyang L, et al. Training language models to follow instructions with human feedback [J]. *Advances in Neural Information Processing Systems*, 2022, 35: 27730-27744. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

[8] Shao Z, et al. [Deepseekmath: Pushing the limits of mathematical reasoning in open language models](https://arxiv.org/abs/2402.03300) [J]. *arXiv preprint arXiv:2402.03300*, 2024.

[9] J. Schulman. [Approximating kl divergence]("http://joschu.net/blog/kl-approx.html"), 2020.

[10] Gao L, Schulman J, Hilton J. [Scaling laws for reward model overoptimization](https://proceedings.mlr.press/v202/gao23b.html) [C]// *International Conference on Machine Learning*. PMLR, 2023.

[11] Chen M, et al. [Evaluating large language models trained on code](https://arxiv.org/abs/2107.03374) [J]. *arXiv preprint arXiv:2107.03374*, 2021.

[12] [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) [Website]. OpenAI, 2024. *(OpenAI blog post about LLM reasoning ability)*

[13] [AMC](https://maa.org/student-programs/amc/) [Website]. Mathematical Association of America (MAA), 2024. *(American Mathematics Competitions AMC official website)*

[14] [Open-O1](https://github.com/Open-Source-O1/Open-O1?tab=readme-ov-file) [Website]. GitHub, 2024. *(Open-O1 open source project GitHub repository)*

[15] Zeng Z, et al. [Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective](https://arxiv.org/abs/2412.14135) [J]. *arXiv preprint arXiv:2412.14135*, 2024.

[16] Hinton G. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) [J]. *arXiv preprint arXiv:1503.02531*, 2015.

[17] Silver D, et al. [Mastering chess and shogi by self-play with a general reinforcement learning algorithm](https://arxiv.org/abs/1712.01815) [J]. *arXiv preprint arXiv:1712.01815*, 2017.

## Citation

> **Citation**: Please indicate the original author and source when reprinting or citing the content of this article.

**Cited as:**

> Yue Shui. (Jan 2025). o1 Replication Progress: DeepSeek-R1.
https://syhya.github.io/posts/2025-01-27-deepseek-r1

Or

```bibtex
@article{syhya2025deepseekr1,
  title   = "o1 Replication Progress: DeepSeek-R1",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jan",
  url     = "https://syhya.github.io/posts/2025-01-27-deepseek-r1"
}
