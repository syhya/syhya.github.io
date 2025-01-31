---
title: "OpenAI o1复现进展：DeepSeek-R1"
date: 2025-01-27T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "Reinforcement Learning", "LLM", "Reasoning Model", "NLP", "Model Distillation", "DeepSeek-R1", "GRPO", "PPO", "SFT", "RFT", "o1", "Reject sampling"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

DeepSeek AI 近期发布 **DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))，其推理性能在多个      benchmark 上已接近 OpenAI o1 ([OpenAI, 2024](https://openai.com/o1/))的水平，是开源社区成功复现 o1 的重要一步。R1 相关代码可以参考huggingface 尝试开源复现 [open-r1](https://github.com/huggingface/open-r1) 项目。以往的研究多依赖于海量的监督数据来提升大语言模型（Large Language Model, LLM）性能，但 DeepSeek-R1 及其早期实验 DeepSeek-R1-Zero 的成功，有力证明了纯粹大规模强化学习在提升 LLM 推理能力方面的潜力。其印证了 Richard Sutton 在 “The Bitter Lesson” 中提出的深刻见解:

> One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning. ([Richard Sutton, 2019](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf))


## 符号

下面列举了文章所使用的数学公式，可以帮你更轻松阅读。

| 符号 | 含义 |
| :--- | :--- |
| \( q \) 或 \( Q \) | 问题，用户提出的输入或指令 |
| \( o \) 或 \( O \) | 输出，模型生成的文本回复或答案  |
| \( t \) | token 索引，表示输出文本中的第 \( t \) 个 token 的位置 |
| \( o_t \) | 输出文本 \( o \) 中的第 \( t \) 个 token |
| \( o_{&lt;t}   \) | 输出文本 \( o \) 中前 \( t-1 \) 个 tokens |
| \( &#124;o&#124; \) | 输出文本 \( o \) 的长度，通常指 token 的数量 |
| \( G \) | 输出组的大小，在 GRPO 算法中，为每个问题采样的输出数量 |
| \( \pi_\theta, \pi_{\theta_{\text{old}}}, \pi_{\text{ref}}, \pi_{\text{sft}} \) | 策略模型及其变体，用于生成文本输出或作为参考模型 |
| \( A_t, A_i \) | 优势函数与相对优势值 |
| \( \varepsilon \) | 剪辑超参数, 用于限制重要性采样率的范围，保证策略更新的稳定性 |
| \( \beta \) | 正则化系数，用于控制 KL 散度惩罚项在目标函数中的权重 |
| \( \mathbb{D}_{KL} \) | KL 散度，衡量两个概率分布之间差异的度量，用于约束新策略与参考策略的距离 |
| \( \mathcal{J}, \mathcal{L} \) | 目标函数与损失函数 |
| \( \mathbb{E} \) | 期望，表示对随机变量的平均值，在目标函数中表示对样本数据的平均 |
| \( P_{\text{sft}}(Q, O) \) | SFT 数据集的分布，表示 \( SFT \) 数据集中问题 \( Q \) 和输出 \( O \) 的联合概率分布 |
| \( P_{\text{sft}}(Q) \) | SFT 数据集中问题的分布，表示 \( SFT \) 数据集中问题 \( Q \) 的边缘概率分布 |
| \( \pi_\theta(o_t \mid q, o_{&lt;t}   ) \) | 策略模型在给定问题 \( q \) 和之前生成的 tokens: \( o_{&lt;t} \) 的条件下，生成第 \( t \) 个 token: \( o_t \) 条件概率 |
| \( \mathbb{I}(o) \) | 判断输出 \( o \) 的答案是否为高质量的函数，高质量时为 1，否则为 0 |
| \( r(o) \) | 奖励函数，评估模型输出 \( o \) 质量的函数 |
| \( r_i \) | 第 \( i \) 个输出的奖励值 |
| \( \nabla_{\theta} \) | 梯度算子，表示对函数关于模型参数 \( \theta \) 求梯度 |
| \( \mathcal{N}(\mu, 1) \) | 正态分布，均值为 \( \mu \)，标准差为 1 |
| \( \binom{a}{b} \) | 二项式系数，表示从 \( a \) 个元素中选择 \( b \) 个元素的组合数 |
| \( r(o) = \frac{\pi_{\text{ref}}(o \mid q)}{\pi_\theta(o \mid q)} \) | 概率比值，参考模型与当前策略模型生成输出 \( o \) 的概率之比 |


## 训练流程概要

DeepSeek-R1 系列模型的训练是一个多阶段的过程，旨在构建具备卓越推理能力和通用语言能力的大型语言模型。整个训练流程从 **DeepSeek-V3** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)) 模型出发，逐步迭代优化，最终得到不同版本的 DeepSeek-R1 模型。

{{< figure
    src="deepseek_r1_pipeline.jpg"
    caption="Fig. 1. DeepSeek R1 Training Pipeline. (Image source: [Harris Chan's Tweet](https://x.com/SirrahChan/status/1881488738473357753))"
    align="center"
    width="90%"
>}}

如图1清晰的显示了 DeepSeek-R1 整个训练流程，主要分为以下几个关键阶段：

- **基座模型与初步微调**: 流程的起点是 **DeepSeek-V3 Base** 模型。首先，使用监督式微调 (SFT) 技术，在 **冷启动长文本 CoT 数据** 上对基础模型进行初步训练，赋予模型初步的推理能力。

- **强化推理能力**: 在 SFT 基础上，采用面向推理的强化学习方法，具体为组相对策略优化 (GRPO) 算法，并结合基于规则的奖励和CoT 语言一致性奖励，进一步提升模型的推理能力。

- **推理数据生成与拒绝采样**: 利用推理提示和拒绝采样技术，并以规则和让 DeepSeek-V3 模型进行评判数据质量，生成高质量的推理数据。

- **非推理数据生成**: 使用 CoT 提示方法，让 DeepSeek-V3 模型进行数据增强，生成非推理数据并且结合原始 SFT 数据，以提升模型的通用语言能力。

- **蒸馏**: 将推理数据和非推理数据结合，用于蒸馏训练。通过 SFT，将 DeepSeek-V3 的能力迁移到一系列小型模型 (Qwen 和 Llama 系列)，得到 **DeepSeek-R1-Distill** 系列模型。

- **最终模型微调**: 对 DeepSeek-V3 模型再次进行 SFT 和强化学习微调。强化学习阶段采用 推理和偏好奖励，并使用多样化的训练提示，最终得到 **DeepSeek-R1** 模型。

- **DeepSeek-R1-Zero**: 通过 GRPO 算法直接在 DeepSeek-V3 Base 上进行训练得到，作为其他模型的对比基准。

接下来博主将深入分析 DeepSeek-R1 训练流程中的关键技术和方法。

## DeepSeek-R1-Zero

### PPO
**近端策略优化 (Proximal Policy Optimization, PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) 算法是一种广泛应用于强化学习的经典算法，在 InstructGPT([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) 论文中被证明是训练 LLM 强化学习微调阶段的有效且稳定的方法。

强化学习核心思想是让智能体 (Agent) 在与环境的交互中学习，通过试错来最大化累积奖励。在**LLM场景下**，模型本身就是智能体，“环境” 可以理解为用户提出的问题和期望的回答方式。策略 (Policy) \(\pi_\theta\) 代表了智能体的行为准则，即给定一个输入 (例如问题 \(q\))，策略会输出一个动作 (例如生成文本 \(o\))。策略 \(\pi_\theta\) 通常由一个神经网络模型参数化，训练的目标是找到最优的参数 \(\theta\)，使得策略能够生成高质量的输出。

Actor-Critic 框架是强化学习中常用的一种架构，PPO 也属于 Actor-Critic 算法。Actor-Critic 框架包含两个核心组件：

- **Actor (策略模型)**：负责学习策略 \(\pi_\theta\)，即如何根据当前状态 (例如用户问题) 选择动作 (生成文本)。
- **Critic (价值模型)**：负责评估 Actor 策略的好坏，通常通过学习一个价值函数 \(V(s)\) 或 \(Q(s, a)\) 来实现。价值函数预测在给定状态 \(s\) (或状态-动作对 \((s, a)\)) 下，未来能够获得的累积奖励的期望值。

PPO 的目标是改进策略模型 (Actor)，使其能够生成更高质量的输出，同时借助价值模型 (Critic) 来稳定训练过程。PPO 通过最大化以下目标函数来更新策略模型 \(\pi_{\theta}\)：

\[
\mathcal{J}_{PPO}(\theta)
= \mathbb{E}\!\Biggl[
   \min\Bigl(
     \underbrace{\frac{\pi_\theta(a\!\mid\!s)}{\pi_{\theta_{\text{old}}}(a\!\mid\!s)}}_{\text{重要性采样率}}\underbrace{A_t}_{\text{优势函数}},\,
     \operatorname{clip}\Bigl(
        \underbrace{\frac{\pi_\theta(a\!\mid\!s)}{\pi_{\theta_{\text{old}}}(a\!\mid\!s)}}_{\text{重要性采样率}},
        1-\varepsilon,\,
        1+\varepsilon
     \Bigr)\underbrace{A_t}_{\text{优势函数}}
   \Bigr)
\Biggr]
\]

**参数说明：**

- **期望 \(\mathbb{E}[\cdot]\)**：表示对样本的平均。在实际训练中，我们会采样一批数据 (例如用户问题和模型生成的回答)，然后计算这批数据的平均目标函数值。
- **重要性采样率**：衡量当前策略 \(\pi_\theta\) 与旧策略 \(\pi_{\theta_{\text{old}}}\) 在动作 \(a\) 上的概率比值。PPO 采用 **近端策略更新** 的思想，限制每次策略更新的幅度，避免策略变化过大导致训练不稳定。
- **优势函数 \(A_t\)**：评估在状态 \(s\) 下采取动作 \(a\) 相对于平均水平的优势。优势函数通常由 Critic 模型 (价值网络) 估计得到，可以是优势估计 (Advantage Estimation) 或 广义优势估计 (Generalized Advantage Estimation, GAE) 等方法。优势函数 \(A_t\) 越大，表示当前动作 \(a\) 越好，策略模型应该增加采取该动作的概率。
- **clip**：PPO 的核心机制之一，本质上可以看作是一个惩罚函数，用于限制重要性采样率的范围在 \([1-\varepsilon, 1+\varepsilon]\) 之间，其中 \(\varepsilon\) 是一个超参数 (通常设置为 0.2)。剪辑操作防止策略更新步幅过大，提高训练的稳定性。

    - `clip` 函数通过限制重要性采样率来惩罚过大或过小的策略更新幅度。
        - 当重要性采样率超出 \([1-\varepsilon, 1+\varepsilon]\) 范围时，`clip` 函数会将其限制在该范围内，从而降低目标函数的增益 (或减少损失)。
        - **对于正向更新 (\(A_t > 0\))：** 如果重要性采样率过大 (超过 \(1+\varepsilon\))，`clip` 会将其限制为 \(1+\varepsilon\)，**降低了实际的更新幅度，惩罚了过于激进的策略改进。**
        - **对于负向更新 (\(A_t < 0\))：** 如果重要性采样率过小 (小于 \(1-\varepsilon\))，`clip` 会将其限制为 \(1-\varepsilon\)，**同样限制了更新幅度，避免策略发生剧烈变化。**

    - 目标函数取 `clip` 之前和 `clip` 之后的最小值，确保在重要性采样率超出范围时，PPO 会对策略更新进行惩罚，保证策略更新的“保守性”。

在实际优化过程中，我们通常将 PPO 损失函数 \(\mathcal{L}_{PPO}(\theta)\) 定义为目标函数的负值，通过最小化损失来最大化目标函数：

\[
\mathcal{L}_{PPO}(\theta) = -\,\mathcal{J}_{PPO}(\theta).
\]

PPO 算法因其 **简单有效、相对稳定** 的特点，成为强化学习领域的基准算法之一，并在各种任务中取得了成功，包括大型语言模型的强化学习微调。PPO 通常被认为比早期的 TRPO 等方法更稳定，但在大模型上的具体应用仍需要细致的超参数调优。在大语言模型场景下，如果价值网络与策略网络完全分离且规模相当，势必会带来更多的计算与内存开销。为解决这些问题，DeepSeek 团队提出了组相对策略优化 (GRPO)算法。

### GRPO

**组相对策略优化 (Group Relative Policy Optimization, GRPO)** ([Shao, et al., 2024](https://arxiv.org/abs/2402.03300)) 是 DeepSeek 团队为训练 DeepSeek-R1-Zero 这样的大语言模型而专门设计的一种高效稳定的强化学习算法。GRPO 的核心创新在于摒弃了传统Actor-Critic 框架中对独立价值网络 (critic model) 的依赖，降低了计算成本，并提高了训练的稳定性。 从广义上讲，GRPO 可以被视为一种 **Actor-Only** 的强化学习方法。

GRPO 的灵感来源于 **相对评估** 的思想。在许多实际场景中，我们往往更容易判断一组事物之间的相对好坏，而不是给出绝对的价值评估。例如，在评价一组学生的作业时，老师可能更容易比较不同作业之间的优劣，而不是给每份作业打一个绝对分数。GRPO 将这种相对评估的思想引入强化学习，通过 **组内相对评分**来构建基准 (baseline)，完全替代了对价值网络的依赖。

具体而言，对于每个问题 \( q \)，GRPO 会从旧策略 \( \pi_{\theta_{\text{old}}} \) 中采样一组输出 \( \{o_1, o_2, \ldots, o_G\} \)，形成一个 **输出组**。然后，通过最大化下面的目标函数来更新策略模型 \( \pi_{\theta} \)：

\[
\begin{aligned}
\mathcal{J}_{GRPO}(\theta)
& = \mathbb{E}\left[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)\right] \\
& \quad \frac{1}{G} \sum_{i=1}^G \Biggl(
    \min\biggl(
      \underbrace{\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}}_{\text{重要性采样率}} \,\underbrace{A_i}_{\text{相对优势值}},\,
      \operatorname{clip}\Bigl(
         \underbrace{\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}}_{\text{重要性采样率}},
         1-\varepsilon,\,
         1+\varepsilon
      \Bigr)\,\underbrace{A_i}_{\text{相对优势值}}
    \biggr)
    \;-\;\beta\,\underbrace{\mathbb{D}_{KL}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr)}_{\text{KL 散度惩罚项}}
\Biggr),
\end{aligned}
\]

与 PPO 的目标函数类似，GRPO 的目标函数也包含重要性采样率和 clip，用于保证策略更新的稳定性。不同之处在于：
- **相对优势值 \(A_i\)**：GRPO 使用 **相对优势值** \(A_i\) 代替 PPO 中的优势函数 \(A_t\)。相对优势值 \(A_i\) 是根据 **组内奖励** 计算得到的，无需价值网络估计。
- **KL 散度惩罚项 \(\mathbb{D}_{KL}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr)\)**：为了进一步约束策略更新，GRPO 引入了 **KL 散度惩罚项**，限制新策略 \(\pi_\theta\) 与参考策略 \(\pi_{\text{ref}}\) 之间的差异过大。

{{< figure
    src="ppo_grpo_comparison.png"
    caption="Fig. 2. The comparison of PPO and GRPO. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2402.03300))"
    align="center"
    width="90%"
>}}


从上图2我们可以看出GRPO 的核心创新在于 **相对优势值 \(A_i\)** 的计算方式。与 PPO 不同，GRPO **不依赖于独立的价值网络**，而是直接利用 **组内奖励** 进行相对评估。对于每个输出组 \( \{o_1, o_2, \ldots, o_G\} \)，GRPO 首先获取每个输出对应的奖励值 \( \{r_1, r_2, \ldots, r_G\} \)。然后，根据以下公式计算相对优势值 \( A_i \)：

\[
A_i = \frac{\,r_i \;-\; \text{mean}(\{r_1, r_2, \ldots, r_G\})\,}{
        \text{std}\bigl(\{r_1, r_2, \ldots, r_G\}\bigr)}.
\]

相对优势值 \( A_i \) 通过 **标准化** 组内奖励 \( \{r_1, r_2, \ldots, r_G\} \) 得到，具有 **零均值和单位方差**，更好地反映了每个输出在组内的相对优劣程度。

GRPO 采用 **相对评估** 的方式，具有以下优点：

- **无需训练价值网络**：避免了训练大规模价值网络带来的计算开销和不稳定性。
- **降低价值估计方差**：相对评估关注组内输出的相对优劣，而不是绝对价值，降低了估计方差，提高了训练稳定性。
- **更符合奖励模型的比较特性**：奖励模型通常基于比较数据训练，GRPO 的相对评估方式与之更契合。
- **更适用于序列生成任务的信用分配**：即使奖励是稀疏的，GRPO 也能有效学习，因为它关注同组输出之间的相对好坏。

### Schulman 无偏估计器

KL 散度 \(\mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right)\)  衡量了策略 \(\pi_\theta\) 相对于参考策略 \(\pi_{\text{ref}}\) 的信息损失，其标准定义为：

\[
\mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right)
= \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \left[ \log \frac{\pi_\theta(o \mid q)}{\pi_{\text{ref}}(o \mid q)} \right].
\]

如前所述，直接计算上述期望在实际中面临挑战。为了解决这个问题，GRPO 采用了 Schulman 无偏估计器 ([Schulman, 2020](http://joschu.net/blog/kl-approx.html))。与公式中可能使用的 KL 散度惩罚项不同，我们使用以下无偏估计器来估计 \(\pi_\theta\) 和 \(\pi_{ref}\) 之间的 KL 散度：

$$
\mathbb{D}_{K L}\left[\pi_{\theta}| | \pi_{r e f}\right]=\frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}-\log \frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,&lt;t}\right)}-1.
$$

为了理解这个估计器的优点，我们首先从数学上推导其无偏性。

#### 无偏性证明

为了简化符号，我们令 \(r(o) = \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)}\)。则 Schulman 估计器可以写成：

\[
\hat{D}_{KL}(o) = r(o) - \log r(o) - 1.
\]

我们需要证明，当 \(o\) 从 \(\pi_\theta(\cdot|q)\) 中采样时，\(\hat{D}_{KL}(o)\) 的期望等于真实的 KL 散度 \(\mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right)\)。

\[
\begin{aligned}
\mathbb{E}_{o \sim \pi_\theta(\cdot|q)} [\hat{D}_{KL}(o)]
&= \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} [r(o) - \log r(o) - 1] \\
&= \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \left[ \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - 1 \right] \\
&= \sum_{o} \pi_\theta(o \mid q) \left[ \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - 1 \right]  \quad (\text{离散情况，连续情况为积分}) \\
&= \sum_{o} \left[ \pi_{ref}(o \mid q) - \pi_\theta(o \mid q) \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} - \pi_\theta(o \mid q) \right] \\
&= \underbrace{\sum_{o} \pi_{ref}(o \mid q)}_{=1} - \underbrace{\sum_{o} \pi_\theta(o \mid q) \log \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)}}_{=-\mathbb{D}_{KL}(\pi_\theta || \pi_{ref})} - \underbrace{\sum_{o} \pi_\theta(o \mid q)}_{=1} \\
&= 1 - (-\mathbb{D}_{KL}(\pi_\theta || \pi_{ref})) - 1 \\
&= \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}).
\end{aligned}
\]

因此，我们证明了 \(\hat{D}_{KL}(o)\) 是 \(\mathbb{D}_{KL}\left(\pi_\theta \|\pi_{\text{ref}}\right)\) 的无偏估计。

#### 三种 KL 散度估计器对比

为了直观理解三种估计器的差异，以下表格列出了它们的数学表达式，其中 \( r(o) = \frac{\pi_{ref}(o \mid q)}{\pi_\theta(o \mid q)} \)：

| 估计器                   | 数学表达式                                               | 主要特点                                     |
|:------------------------|:---------------------------------------------------------|:---------------------------------------------|
| **k1 (朴素估计器)**     | \(\hat{D}_{KL}^{(k1)}(o) = \log \frac{\pi_\theta(o \mid q)}{\pi_{ref}(o \mid q)} = \log \frac{1}{r(o)}\) | 简单直接，对应 KL 散度定义；方差较高，估计结果波动较大。 |
| **k2 (平方对数比估计器)** | \(\hat{D}_{KL}^{(k2)}(o) = \frac{1}{2} (\log r(o))^2\)     | 使用对数比的平方，始终为正，降低方差；引入偏差，尤其在分布差异大时。 |
| **k3 (Schulman 无偏估计器)** | \(\hat{D}_{KL}^{(k3)}(o) = r(o) - \log r(o) - 1\)         | 结合了比值 \( r(o) \) 和对数比 \(\log r(o)\)；无偏，较低方差，估计稳定。 |


- **k1 (朴素估计器):** 无偏简单直接，但方差较高，导致估计结果不稳定。
- **k2 (平方对数比估计器):** 降低了方差，但引入了偏差，特别是在分布差异较大时偏差显著。
- **k3 (Schulman 无偏估计器):** 兼具无偏性和较低的方差，提供了稳定的估计结果。

#### 实验结果

为了评估三种 KL 散度估计器的性能，我们进行了数值实验，结果如下表所示。实验中，固定分布 \( q = \mathcal{N}(0, 1) \)，通过改变分布 \( p = \mathcal{N}(\mu, 1) \) 的均值 \(\mu\) 来控制真实的 KL 散度 \(\mathbb{D}_{KL}(p \| q)\)。使用5亿个样本进行 Monte Carlo 估计，并重复实验以获得稳定结果。

实验代码可以参考 [unbiased_kl_divergence.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-27-deepseek-r1/unbiased_kl_divergence.py)

| 真实 KL 散度 | 估计器 | 平均估计值 | 标准差  | 相对偏差 (%) |
|:------------:|:------:|:----------:|:-------:|:------------:|
| 0.005        | k1     | 0.005      | 0.1     | 0.0387       |
| 0.005        | k2     | 0.005      | 0.0071  | 0.2415       |
| 0.005        | k3     | 0.005      | **0.0071**  | -0.0082      |
| 0.125        | k1     | 0.125      | 0.5     | -0.0389      |
| 0.125        | k2     | 0.1328     | 0.1875  | 6.2500       |
| 0.125        | k3     | 0.125      | **0.1845**  | 0.0072       |
| 0.5          | k1     | 0.5        | 1       | -0.0018      |
| 0.5          | k2     | 0.625      | 0.866   | 25.0004      |
| 0.5          | k3     | 0.5        | **0.8478**  | 0.0052       |

- **朴素估计器 (k1):**
  - **无偏性:** 平均估计值与真实 KL 散度高度吻合，相对偏差接近 0%。
  - **方差:** 标准差高于 k3，且随着真实 KL 散度的增加而增大，导致估计结果不稳定。

- **平方对数比估计器 (k2):**
  - **无偏性:** 存在一定偏差，且偏差随真实 KL 散度的增加而显著增大（例如，真实 KL 为 0.5 时相对偏差达到 25%）。
  - **方差:** 在较低的真实 KL 散度下方差较低，但整体表现不稳定。

- **Schulman 无偏估计器 (k3):**
  - **无偏性:** 实验结果显示相对偏差极小，几乎为 0%，验证了其无偏性。
  - **方差:** 标准差明显低于 k1，且与 k1 相比在所有 KL 散度下均表现出更低的方差，尤其在较低 KL 散度时优势显著。

#### 优点总结

- **无偏性:** 理论和实验结果均表明，k3 是无偏估计器，能够准确反映真实的 KL 散度。
- **正定性:** 估计值始终为非负，符合 KL 散度的性质。
- **较低的方差:** 相较于 k1，k3 显著降低了估计方差，提供了更稳定的估计结果，尤其在 KL 散度较小时表现突出。

Schulman 无偏估计器 \( \hat{D}_{KL}^{(k3)}(o) = r(o) - \log r(o) - 1 \) 为 KL 散度提供了一种兼具无偏性和低方差的估计方法。其无偏性确保了估计的准确性，而较低的方差提升了估计的稳定性，特别适用于需要稳定梯度信号的强化学习场景，如策略优化。基于这些优势，GRPO 算法选择使用 k3 作为惩罚策略偏离的估计器，从而保证训练过程的稳定性和最终策略的性能。

在实际优化中，GRPO 损失函数 \(\mathcal{L}_{GRPO}(\theta)\) 被定义为目标函数 \(\mathcal{J}_{GRPO}(\theta)\) 的负值，通过最小化损失函数 \(\mathcal{L}_{GRPO}(\theta)\) 来实现目标函数 \(\mathcal{J}_{GRPO}(\theta)\) 的最大化：

\[
\mathcal{L}_{GRPO}(\theta) = -\,\mathcal{J}_{GRPO}(\theta)
\]


### PPO 与 GRPO 对比

为更清晰理解 PPO 和 GRPO 的异同，以下表格对两种算法进行对比：

| 特性               | PPO                                 | GRPO                               |
| :----------------- | :------------------------------------------------- | :--------------------------------------------------- |
| **是否 Actor-Critic** | 是                                                  | 是 (广义上可以认为是 Actor-Only)                     |
| **是否价值网络**     | 需要独立的价值网络 (Critic)                           | 无需独立的价值网络                                    |
| **优势函数估计**     | 通过价值网络估计绝对优势值                           | 通过组内奖励相对评估相对优势值                       |
| **计算开销**         | 较高，需要训练价值网络                                | 较低，无需训练价值网络                                |
| **训练稳定性**       | 相对较好，但价值网络训练可能引入不稳定性               | 更好，避免了价值网络训练带来的不稳定性               |
| **算法复杂度**       | 相对复杂，需要维护和更新策略网络和价值网络           | 相对简单，只需维护和更新策略网络                     |
| **适用场景**         | 广泛适用于各种强化学习任务，包括中小规模语言模型微调 | 特别适用于大语言模型的强化学习微调，注重效率和稳定性 |
| **信用分配**         | 依赖价值网络进行时间差分学习，处理信用分配问题       | 依赖最终奖励和组内相对评估，也可辅助中间奖励         |
| **方差问题**         | 价值网络估计可能引入方差                             | 组内相对优势估计在小组规模下可能存在方差，可通过增大组规模等缓解 |

从表中可以看出，PPO 是一种通用且强大的强化学习算法，但其训练价值网络的机制在大语言模型场景下带来了额外的计算负担和潜在的不稳定性。**GRPO 通过引入组相对评分，巧妙地规避了对价值网络的需求，在保证性能的同时，显著降低了计算成本，并提升了训练稳定性**。这使得 GRPO 成为在训练资源不多的情况下训练 **DeepSeek-R1-Zero** 这样 LLM 的理想选择。

### 代码生成评估指标

代码生成会采用更严谨的测试方法。通过编译器执行模型生成的代码，并使用预定义的测试用例进行多次单元测试，以判断代码的正确性。常用的评估指标包括 **pass@k**([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) 和 **cons@N**([OpenAI, 2024](https://openai.com/index/learning-to-reason-with-llms/))。

`pass@k`: 衡量模型在生成 k 个代码样本时，至少有一个样本能够通过所有预定义测试用例的概率。

#### pass@k 有偏估计公式

\[
\text{Simplified pass@k} = \frac{1}{P} \sum_{i=1}^{P} C_i
\]
其中，对于每个问题 \(i\), \(C_i\) 定义为：
\[
C_i = \begin{cases}
    1 & \text{如果生成的 k 个样本中至少有一个是正确的} \\
    0 & \text{如果生成的 k 个样本全部都不正确}
\end{cases}
\]

**参数说明:**

*   \( P \):  评估的问题总数。
*   \( C_i \):  对于第 \(i\) 个问题，如果生成的 \(k\) 个样本中至少有一个是正确的，则 \(C_i = 1\)，否则 \(C_i = 0\)。
*   \( \sum_{i=1}^{P} C_i \):  表示在所有 \(P\) 个问题中，被 “解决” 的问题总数。
*   \( \frac{1}{P} \sum_{i=1}^{P} C_i \):  表示 “解决” 问题的比例，即准确率。

**公式含义:**  这种简化方法直接计算 **生成 k 个样本后，至少有一个样本正确的比例**。  虽然这种方法提供的是 pass@k 的 **有偏估计**，可能会略微高估真实值，但它在实践中非常常用，因为它 直观、易于计算，并且在样本量足够大时，能够提供对模型性能的合理近似。尤其在工业界和快速评估场景中，这种简化方法非常实用。

然而，LLM 在推理解码时会受到 `temperature`、`top_p（核采样概率）`、`top_k（候选词数量）`和`repetition_penalty` 等参数的影响。这些参数会使代码生成结果有随机性和多样性，并且当样本 K 比较少的如果设置随机过高的参数，会影响 pass@k 的评估结果。因此，采用无偏估计方法能够更准确地反映模型的真实性能。

#### pass@k 的无偏估计公式

\[
\text { pass @ } k:=\underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
\]

**参数说明:**

*   \( n \):  为每个问题生成的代码样本总数。
*   \( c \):  在 \( n \) 个样本中，能够通过所有单元测试的正确样本数。
*   \( k \):  pass@\(k\) 指标中的参数 \(k\)，表示我们考虑的生成样本数量。
*   \( \binom{a}{b} \):  表示二项式系数，计算从 \(a\) 个元素中选择 \(b\) 个元素的组合数。
*   \( \underset{\text { Problems }}{\mathbb{E}} \):  表示对所有评估问题的期望值（平均值）。

**公式含义:** 
- 公式实际上计算的是至少有一个正确样本的概率。公式  \( \frac{\binom{n-c}{k}}{\binom{n}{k}} \)  计算的是在生成的 \(n\) 个样本中，随机抽取 \(k\) 个样本，且这 \(k\) 个样本都不正确的概率。我们用 1 减去这个概率，就得到了 在 \(n\) 个样本中，随机抽取 \(k\) 个样本，且这 \(k\) 个样本中至少有一个正确的概率， 这就是 pass@\(k\) 指标的含义。
- 这个公式提供的是 pass@k 的 **无偏估计**，更适用于学术研究等需要精确评估的场景。 在实际计算中，通常会生成远大于 \(k\) 的样本数 \(n\) (例如论文中使用 \(n=200\), \(k \leq 100\))，以更稳定地估计 pass@\(k\)。

#### pass@k 简化乘积形式

为了更方便数值计算，原始公式还可以转化为以下乘积形式，它仍然是无偏估计，并能避免数值溢出问题：

\[
\text { pass @ } k = \underset{\text { Problems }}{\mathbb{E}}\left[1 - \prod_{i=0}^{k-1} \frac{n-c-i}{n-i}\right]
\]

**推导过程:**

1. 至少有一个正确样本的反面是所有 k 个样本都不正确。 因此，pass@k 等于 1 减去 所有 k 个样本都不正确的概率。

2. 考虑**不放回抽样**的场景。假设我们从 \(n\) 个样本中抽取 \(k\) 个样本，要计算这 \(k\) 个样本都不正确的概率。总共有 \(n\) 个样本，其中 \(n-c\) 个是不正确的。

3. 第一次抽取时，抽到不正确样本的概率为 \( \frac{n-c}{n} \)。

4. 在第一次抽取到不正确样本的条件下，第二次抽取时，剩余 \(n-1\) 个样本中，有 \(n-c-1\) 个不正确样本。因此，第二次仍然抽到不正确样本的条件概率为 \( \frac{n-c-1}{n-1} \)。

5. 以此类推，第 \(i\) 次抽取时 ( \(i\) 从 1 到 \(k\) )，在之前 \(i-1\) 次都抽到不正确样本的条件下，第 \(i\) 次仍然抽到不正确样本的条件概率为 \( \frac{n-c-(i-1)}{n-(i-1)} = \frac{n-c-i+1}{n-i+1} \)。  为了与公式中的索引 \(i=0\) 对齐，我们将索引改为从 \(i=0\) 到 \(k-1\)，则第 \(i+1\) 次抽取时 ( \(i\) 从 0 到 \(k-1\) )，条件概率为 \( \frac{n-c-i}{n-i} \)。

6. 将这 \(k\) 次抽取的条件概率连乘，即可得到所有 \(k\) 个样本都不正确的概率：

    \[
    P(\text{所有 k 个样本都不正确}) = \frac{n-c}{n} \times \frac{n-c-1}{n-1} \times \cdots \times \frac{n-c-k+1}{n-k+1} = \prod_{i=0}^{k-1} \frac{n-c-i}{n-i}
    \]

7. 最终，pass@k 的简化公式为：

    \[
    \text { pass @ } k = \underset{\text { Problems }}{\mathbb{E}}\left[1 - \prod_{i=0}^{k-1} \frac{n-c-i}{n-i}\right]
    \]

这个乘积形式的公式，避免了直接计算可能数值很大的二项式系数，更易于理解和数值计算，尤其是在编程实现时，可以逐项累乘，有效防止数值溢出。

#### cons@N 
`cons@N`:  通过生成 N 个样本，并从中选择出现频率最高的答案作为最终答案，评估该答案的准确率。在 DeepSeek-R1-Zero 的评估中，使用了 **cons@64**，即生成 64 个样本，并取其中出现次数最多的答案作为最终答案进行评估。


\[
\text{cons@N} = \frac{1}{P} \sum_{i=1}^{P} \mathbb{I}(\text{ConsensusAnswer}_i \text{ is correct})
\]

**参数说明:**

- \( P \)：评估的问题总数。
- \( \text{ConsensusAnswer}_i \)：通过多数投票得到的 **共识答案**。
- \( \mathbb{I}(\text{ConsensusAnswer}_i \text{ is correct}) \)：指示函数，若共识答案正确，则为 1，否则为 0。

**公式含义：** 计算在所有评估问题中，共识答案正确的比例。通过增加生成样本数 \(N\)，并采用多数投票策略，cons@N 指标能够更稳定和可靠地评估模型的平均性能。在模型生成结果存在一定随机性的情况下，该指标可以验证模型输出的一致性和准确性。


### 奖励模型

奖励模型在 LLM 的研发中至关重要，主要应用于以下关键环节：

- **基于人类反馈的强化学习**: 在基于人类反馈的强化学习（RLHF）流程中，奖励模型用于评估模型生成结果的质量，并为后续的强化学习提供奖励信号。

- **拒绝采样的关键工具**: 在拒绝采样过程中，奖励模型对大量候选结果进行评分，筛选出高质量样本用于监督微调（SFT）。拒绝采样是自动化样本工程的重要方法，而奖励模型是其核心组成部分。

- **业务场景中的判别器**: 在实际应用中，奖励模型作为 LLM 输出结果的判别器或校验器，评估生成结果的质量。只有得分超过预设阈值的结果才会输出，否者进行再生成或降级处理，提高输出的可靠性和安全性。

#### ORM 与 PRM
{{< figure
    src="orm_prm_comparison.png"
    caption="Fig. 3. Outcome reward vs Process reward. (Image source: [Zeng et al., 2024](https://arxiv.org/abs/2412.14135))"
    align="center"
    width="100%"
>}}

当前奖励模型主要分为两种范式：**结果奖励模型(Outcome Reward Model, ORM)** 和 **过程奖励模型(Process Reward Model, PRM)**。上图3直观的展示了这两种奖励模型的区别。以下表格也对比了这两种模型的主要特性：

| 特性                   | **ORM**       | **PRM**       |
|------------------------|-----------------------------------------------------|------------------------------------------------------|
| **定义**               | 对模型生成的完整结果进行整体评分                  | 在内容生成过程中，对每一步或每个阶段进行细粒度评分 |
| **主要优势**           | 简单直接，易于实现<br> 对整体结果进行全面评估      | 提供更精细的奖励信号<br> 有助于指导模型生成过程的每个步骤 |
| **主要劣势**           | 方差较高，估计结果波动较大 <br> 缺乏过程中的反馈  | 训练和应用更为复杂 <br> 可能引入偏差，尤其在分布差异大时 |
| **适用场景**           | 需要整体评估生成结果的任务          | 需要细粒度控制生成过程的任务，如分步推理或复杂生成任务 |
| **避免奖励欺骗的能力** | 中等，依赖于整体评分的准确性                        | 较低，可通过优化每一步的奖励而非整体表现来作弊 |
| **训练复杂度**         | 较低，无需对生成过程进行额外的监督                  | 较高，需要在生成的每一步进行评分，增加了计算和数据需求 |
| **可解释性**           | 高，评分基于最终结果                                | 较低，评分涉及生成过程的多个步骤，难以全面理解每一步的评分依据 |


#### 采用 ORM

为了训练 DeepSeek-R1-Zero，DeepSeek 团队选择了**ORM**，而非PRM。此选择基于以下考虑：

- **避免奖励欺骗（Reward Hacking）**  
  PRM在大规模 RL 训练中，容易被智能体利用，导致奖励欺骗（[Gao et al., 2022](https://arxiv.org/abs/2210.10760)）。模型可能采取“旁门左道”的策略以最大化奖励，而非提升推理能力。基于规则的奖励系统通过明确且可解释的规则，有效避免了奖励欺骗问题。

  > 基于规则的奖励系统在问题场景复杂或需要创造性回答时，可能难以覆盖所有类型的问题，规则设计可能存在漏洞被模型利用。

- **降低训练复杂度**  
  训练 PRM 需要大量计算资源和数据，增加了训练流程的复杂性。而基于规则的奖励系统无需额外训练，规则一旦确定即可直接应用，简化了训练流程。基于规则的奖励系统特别适合自动判分或目标明确的任务，如数学题、LeetCode 编程题及对输出格式有明确要求的任务。对于开放式对话或创意类任务，则可能需要结合人类反馈或训练好的奖励模型。


#### 奖励机制

DeepSeek-R1-Zero 的奖励系统采用双重奖励机制，通过预定义的规则进行自动化评估，确保评估过程的高效性和实时性。这套系统主要包含以下两种类型的奖励：

**1. 准确性奖励 (Accuracy Reward)**

* **定义：**  衡量模型输出结果的正确性，是奖励系统中最关键的部分。
* **实现方式：**  根据不同任务类型采用不同的验证方法：
    * **数学问题：** 验证最终答案是否与标准答案一致。
    * **代码生成：** 通过编译器执行模型生成的代码，并使用预设的单元测试用例进行多次测试，判断代码的正确性。
* **目的：**  引导模型生成准确、可靠的输出结果。

**2. 格式奖励 (Format Reward)**

* **定义：**  为了提升模型输出的可读性和结构性，方便后续分析和评估而引入的奖励机制。
* **评估方式：**  在强化学习训练过程中，通过预定义的规则系统进行自动化评估。
* **目的：**  鼓励模型生成结构化的输出，例如包含思考过程和最终答案，使其更易于理解和分析。


DeepSeek-R1-Zero 的奖励函数 \(r(o)\) 由准确性奖励和格式奖励加权求和构成：

$$
r(o) = r_{\text{accuracy}}(o) + \lambda \cdot r_{\text{format_effective}}(o)
$$

其中，有效格式奖励 \(r_{\text{format_effective}}(o)\) 的计算方式如下：

$$
r_{\text{format_effective}}(o) =
\begin{cases}
    r_{\text{format}}(o) & \text{如果 } o \text{ 的基本格式符合要求} \\
    0 & \text{如果 } o \text{ 的基本格式不符合要求}
\end{cases}
$$

基础格式奖励 \(r_{\text{format}}(o)\) 则根据格式规范的符合程度进行分级：

$$
r_{\text{format}}(o) =
\begin{cases}
    R_{\text{format_full}} & \text{如果 } o \text{ 的格式完全符合规范} \\
    R_{\text{format_partial}} & \text{如果 } o \text{ 的格式部分符合规范} \\
    0 & \text{如果 } o \text{ 的格式不符合规范}
\end{cases}
$$

### 实验流程

#### 训练模板

为了引导基模型遵循指定的指令，DeepSeek 团队设计了一个简洁而有效的训练模板。该模板要求模型首先生成推理过程（放在 `<think>` 和 `</think>` 标签之间），然后再提供最终答案（放在 `<answer>` 和 `</answer>` 标签之间）。这种结构化的格式，不仅确保了输出的可读性，更使研究人员能够清晰地观察模型在 RL 训练过程中的推理过程，从而更准确地评估模型的学习进展。

| 角色 | 提示内容           | 助手回复                             |
| :--- | :----------------- | :----------------------------------- |
| 用户 | prompt (用户提出的问题) | 助手： `<think> 推理过程 </think>` `<answer> 答案 </answer>` |


- `<think>` 和 `</think>` (思维过程标签):  用于包裹模型的中间推理步骤，清晰展示模型的思考过程，便于理解模型的推理逻辑和进行错误分析。
- `<answer>` 和 `</answer>` (最终答案标签): 用于包裹模型的最终答案，方便程序自动化提取答案部分，进行高效的评估和后续处理。

#### 评估流程

1. **准确性评估：**  评估模型输出 \(o\) 的答案是否正确，计算准确性奖励 \(r_{\text{accuracy}}(o)\)。
2. **基本格式检查：**  检查输出 \(o\) 的基本格式是否符合预定义要求，例如是否包含必要的标签 `<think>` 和 `<answer>`，以及标签是否正确闭合和嵌套。
3. **有效格式奖励判断：**
    * **基本格式不符合：**  有效格式奖励 \(r_{\text{format_effective}}(o) = 0\)。
    * **基本格式符合：**  进一步评估格式规范程度，计算基础格式奖励 \(r_{\text{format}}(o)\)。
4. **最终奖励计算：**  将准确性奖励 \(r_{\text{accuracy}}(o)\) 和有效格式奖励 \(r_{\text{format_effective}}(o)\) 进行线性加权求和，得到最终奖励 \(r(o)\)。

通过结合准确性奖励和格式奖励，DeepSeek-R1-Zero 的奖励系统不仅关注模型输出的正确性，更重视输出结果的结构化和可读性。这使得模型不仅能够给出正确的答案，还能展现其思考过程，使其更像一个具备推理能力的智能体，而不仅仅是一个简单的答案输出机器。

#### 实验结果

{{< figure
    src="deepseek_r1_zero_benchmark.png"
    caption="Fig. 4. Comparison of DeepSeek-R1-Zero and OpenAI o1 models on reasoning-related benchmarks. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

图4展示了不同模型在多项基准测试上的表现。在 [AIME 2024](https://maa.org/student-programs/amc/) 基准测试中，DeepSeek-R1-Zero 模型的 pass@1 分数达到了71.0%，此外 cons@64 分数为86.7%，与 OpenAI o1-0912 模型相当。

{{< figure
    src="deepseek_r1_zero_response_time.png"
    caption="Fig. 5. The average response length of DeepSeek-R1-Zero on the training set during the RL process. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

图5显示了随着训练的深入，DeepSeek-R1-Zero 模型展现出自发 **自我进化** 的能力。模型根据问题的复杂程度，动态地分配“思考时间”，对于更复杂的问题，会自发生成更长的推理链，进行更深入的思考。这种“思考时间”自适应调整，并非人为设定，而是模型在 RL 训练过程中自发涌现的行为，充分体现了强化学习驱动下，模型推理能力的自主提升。

## DeepSeek-R1

### 训练流程

为了在 DeepSeek-R1-Zero 的基础上进一步提升模型性能，DeepSeek 团队采用了 **多阶段训练** 策略，并将 **冷启动数据** 融入到训练流程中。DeepSeek-R1 的训练流程主要包括以下四个阶段，体现了从初步策略初始化到全面能力提升的进阶之路：

1. **冷启动 (Cold Start)**: 利用高质量的长思维链 (CoT) 数据，对 DeepSeek-V3-Base 基模型进行初步的监督微调，为后续强化学习奠定基础。

2. **面向推理的强化学习 (Reasoning-Oriented RL)**: 在冷启动模型基础上，应用强化学习算法，专注于增强模型在推理密集型任务中的能力。

3. **拒绝采样与监督微调 (Rejection Sampling & SFT)**: 通过拒绝采样技术筛选高质量推理数据，并结合非推理数据进行监督微调，进一步提升模型推理能力和通用能力。

4. **面向所有场景的强化学习 (All-Scenario RL)**: 综合考虑推理和非推理任务，进行第二阶段强化学习，使模型与人类偏好对齐，提升在更广泛场景下的表现。


### 冷启动

在 DeepSeek-R1 的训练流程中，**冷启动**阶段至关重要，它如同引擎的点火器，为后续复杂的强化学习过程奠定坚实基础。监督微调是冷启动阶段的核心技术。

#### 训练目标

冷启动阶段的目标明确而关键：利用高质量的思维链 (Chain-of-Thought, CoT) 数据，对 DeepSeek-V3-Base 基模型进行初步微调。这次微调旨在快速赋予模型以下核心能力：

* **初步推理能力：**  引导模型学习模仿人类的推理过程，为更复杂的推理打下基础。
* **良好文本生成质量：**  确保模型输出文本的流畅性和自然度，提升用户体验。

这些 CoT 数据如同模型的“启动燃料”，帮助模型快速掌握人类的推理模式，并为后续强化学习提供**良好的策略初始化**，有效**避免 RL 训练初期从零开始探索的低效和不稳定性**。

#### 数据构建

为了构建高质量的冷启动数据，DeepSeek 团队进行了多方探索，最终整合了以下高效方法：

* **少量示例引导 (Few-shot Prompting)：**  利用少量高质量的示例，引导模型生成更长、更具深度和逻辑性的 CoT 数据。
* **模型生成 + 反思验证：**  直接 Prompt 模型生成答案，并加入反思和验证环节，确保答案的质量和推理的正确性。
* **优化 R1-Zero 输出：**  收集 DeepSeek-R1-Zero 模型的输出，通过人工标注和优化，提升数据的可读性和整体质量。

通过上述策略，DeepSeek 团队积累了**数千条高质量的冷启动数据**，并以此为基础对 DeepSeek-V3-Base 进行了微调，作为强化学习的坚实起点。

#### 冷启动优点

相比于直接以 DeepSeek-R1-Zero 作为起点，冷启动数据带来了多项显著优势，为后续训练奠定了更优的基础：

* **显著提升可读性 (Improved Readability)：**
    * DeepSeek-R1-Zero 的输出存在可读性挑战，例如语言混合、缺乏结构化格式等。
    * 冷启动数据特别设计了**更易读的输出模式**，包括：
        * **添加摘要 (Summary)：**  在回复末尾添加精炼的摘要，快速提炼核心结论。
        * **过滤不良回复：**  去除不友好或低质量的回复，确保数据纯净度。
        * **结构化输出格式：**  采用 `| special_token | <reasoning_process> | special_token | <summary>` 格式，清晰呈现推理过程和总结。

* **性能显著提升 (Enhanced Performance)：**
    * 通过精心设计融入人类先验知识的数据模式，DeepSeek 团队观察到模型性能相较 R1-Zero 有了显著提升。
    * 这进一步验证了迭代训练是提升推理模型性能的有效路径。

* **更优的策略初始化 (Superior Policy Initialization)：**
   * **冷启动阶段的 SFT 核心在于策略初始化。**  策略初始化是构建 Reasoing LLM，例如 OpenAI o1 系列的关键步骤。通过学习高质量 CoT 数据，模型初步掌握人类推理模式，并具备生成结构化推理过程的能力，为后续强化学习训练奠定坚实基础，避免了从零开始探索的困境。

### 监督微调
**监督微调 (Supervised Fine-tuning, SFT)** 的核心目标是通过在有监督标注的数据上微调模型，使其预测结果尽可能接近真实标签。 这旨在提升模型在特定任务和指令执行方面的能力。

#### 损失函数

SFT 的训练目标是最小化模型预测与真实标签之间的差异。损失函数通常采用**交叉熵损失 (Cross-Entropy Loss)**，也称为**负对数似然 (Negative Log-Likelihood)**，用于衡量模型预测 token 分布与真实 token 分布之间的差异。 为了平衡不同长度输出序列的贡献，我们通常会将损失函数归一化到每个 token 的平均损失。

损失函数公式如下：

\[
\mathcal{L}_{SFT}(\theta) = - \mathbb{E}_{(q, o) \sim P_{sft}(Q, O)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \log \pi_\theta\left(o_t \mid q, o_{&lt;t}       \right)\right]
\]

**参数说明:**

* **\(\mathcal{L}_{SFT}(\theta)\)**：SFT 损失函数，通过调整模型参数 \(\theta\) 最小化该函数。
* **\(\mathbb{E}_{(q, o) \sim P_{sft}(Q, O)}[\cdot]\)**：在 SFT 数据集分布 \(P_{sft}(Q, O)\) 上的期望。
* **\(P_{sft}(Q, O)\)**：SFT 数据集分布，\(q\) 代表问题 (Query)，\(o\) 代表对应的标准答案输出 (Output)。
* **\((q, o)\)**：从 SFT 数据集中采样的 问题-答案对。
* **\(|o|\)**：标准答案输出的 token 长度。
* **\(o_t\)**：标准答案输出的第 \(t\) 个 token。
* **\(o_{&lt;t}  \)**：标准答案输出的前 \(t-1\) 个 tokens。
* **\(\pi_\theta\left(o_t \mid q, o_{&lt;t}   \right)\)**：给定问题 \(q\) 和前文 \(o_{&lt;t}  \)，模型预测 token \(o_t\) 的概率。
* **\(\frac{1}{|o|}\)**:  长度归一化因子，将总损失除以输出序列长度，得到每个 token 的平均损失。

SFT 损失函数旨在惩罚模型预测与标准答案之间的偏差。 对于给定的问题 \(q\) 和标准答案 \(o\)，损失函数计算模型预测答案 \(o\) 中每个 token \(o_t\) 的概率 \(\pi_\theta(o_t | q, o_{&lt;t}   )\)。通过除以输出长度 \(|o|\)，损失函数被归一化为每个 token 的平均负对数似然。

* **模型准确预测标准答案 token 时**，\(\pi_\theta(o_t \mid q, o_{&lt;t}   )\approx 1\)，\(\log \pi_\theta(o_t \mid q, o_{&lt;t}   )\approx 0\)，损失值接近最小值。
* **模型预测偏离标准答案时**，\(\pi_\theta(o_t \mid q, o_{&lt;t}   )\) 较小，\(\log \pi_\theta(o_t \mid q, o_{&lt;t}   )\) 为负数且绝对值较大，损失值增大。

最小化 SFT 损失函数的过程，就是让模型学习生成与训练数据集中标准答案尽可能相似文本的过程。从负对数似然角度看，目标是找到最优模型参数 \(\theta\)，最大化模型生成训练数据答案 \(o\) 的概率，等价于最小化生成答案 \(o\) 的负对数似然。高质量的 CoT 数据蕴含人类对推理和结果的偏好，因此 SFT 也可视为让模型学习并拟合人类推理偏好的过程。

#### 梯度

SFT 损失函数的梯度用于指导模型参数更新，以降低损失值。 损失函数关于模型参数 \(\theta\) 的梯度为：

\[
\nabla_{\theta} \mathcal{L}_{SFT} = - \mathbb{E}_{(q, o) \sim P_{sft}(Q, O)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \nabla_{\theta} \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}       \right)\right]
\]

**参数说明:**

* **\(\nabla_{\theta} \mathcal{L}_{SFT}\)**：SFT 损失函数关于参数 \(\theta\) 的梯度，指示损失函数值下降最快的方向。
* **\(\nabla_{\theta} \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}   \right)\)**:  token 概率对数 \(\log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}   \right)\) 关于参数 \(\theta\) 的梯度。
* **\(\frac{1}{|o|}\)**: **长度归一化因子**，与损失函数保持一致，梯度也是**每个 token 平均损失的梯度**。

实际计算梯度时，通常使用随机梯度下降算法，沿着梯度下降方向更新模型参数，逐步最小化损失函数，提升模型生成标准答案的准确性。

**梯度系数**

在 SFT 阶段，梯度系数通常设置为 1，这意味着**所有训练样本对模型参数的更新贡献相同**，模型平等地学习每个示例，力求最小化在整个数据集上的平均损失。

#### 数据来源与人类偏好

* **数据来源 (Data Source)**：SFT 数据集主要由高质量的长链思维 (CoT) 示例构成，代表了期望模型学习的“标准答案”，用于指导损失函数最小化。 数据可能来自人工标注或更强大的模型生成。可参考 [Open-o1](https://github.com/Open-Source-O1/Open-O1?tab=readme-ov-file) 项目的 SFT 数据集 [OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)，包含长 CoT 回复。
* **人类偏好 (Human Preference)**：在 SFT 阶段，人类选择可以被视为隐式的奖励函数。 高质量 CoT 数据体现了人类对模型推理和输出的期望，模型通过学习这些数据，最小化与人类期望输出的偏差，从而拟合人类偏好。

### 面向推理的强化学习

在冷启动微调后，DeepSeek 团队通过强化学习 (RL) 进一步提升模型在推理密集型任务（如编码、数学、科学和逻辑推理）中的能力。 此阶段的核心在于**最大化奖励函数，引导模型学习更有效的推理策略**。

#### 奖励函数

为了解决推理过程中 CoT 语言混合问题，DeepSeek 团队引入了**语言一致性奖励**，并将其与**任务奖励**结合，构成总奖励函数：

\[
r(o) = r_{\text{task}}(o) + \alpha \cdot r_{\text{lang_consistency}}(o)
\]

**参数说明:**

* **\(r(o)\)**：总奖励函数，RL 训练的目标是最大化该函数。
* **\(r_{\text{task}}(o)\)**：基于任务完成情况的任务奖励，衡量模型推理的准确性。
* **\(r_{\text{lang_consistency}}(o)\)**：语言一致性奖励，衡量 CoT 输出的语言纯度。
* **\(\alpha\)**：超参数，平衡任务奖励和语言一致性奖励的权重。

总奖励函数是任务奖励和语言一致性奖励的加权和。最大化 \(r(o)\) 驱动模型在提升推理准确性的同时，保持 CoT 输出的语言一致性。  \(\alpha\) 的作用是调整模型对语言一致性的重视程度。

#### 训练目标

通过最大化上述奖励函数，DeepSeek 团队在冷启动微调后的模型上进行 RL 训练，**优化模型参数，使其在推理任务上获得更高的奖励值，最终提升推理能力**。

### RFT

**拒绝采样微调 (Rejection Sampling Fine-tuning, RFT)** 旨在通过精炼训练数据提升模型通用能力。其核心思想是最小化选择性损失函数，引导模型学习高质量输出的生成模式。

#### 损失函数

RFT 采用**拒绝采样**策略，区分**推理数据**和**非推理数据**的生成与选择过程，构建高质量 SFT 数据集。训练目标是最小化以下损失函数：

\[
\mathcal{L}_{RFT}(\theta) = - \mathbb{E}_{(q, o) \sim P_{sft}(Q) \times \pi_{sft}(O \mid q)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \mathbb{I}(o) \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}  \right)\right]
\]

其中，指示函数 \(\mathbb{I}(o)\) 定义为：

\[
\mathbb{I}(o) = \begin{cases}
    1, & \text{如果输出 } o \text{ 被判定为高质量} \\
    0, & \text{否则}
\end{cases}
\]

**参数说明:**

* **\(\mathcal{L}_{RFT}(\theta)\)**: RFT 损失函数。
* **\(P_{sft}(Q)\)**: 问题 \(q\) 的分布。
* **\(\pi_{sft}(O \mid q)\)**: 给定问题 \(q\)，SFT 模型生成输出 \(O\) 的条件概率分布。
* **\(\mathbb{I}(o)\)**: 指示函数，用于选择高质量答案。当输出 \(o\) 被判定为高质量时为 1，否则为 0。

RFT 损失函数基于交叉熵损失，通过指示函数 \(\mathbb{I}(o)\) **选择性地学习高质量输出**：

* **高质量输出 (\(\mathbb{I}(o) = 1\)):** 损失函数退化为标准交叉熵损失，模型根据高质量答案的负对数似然更新参数，最小化模型预测与高质量答案的差异。
* **低质量输出 (\(\mathbb{I}(o) = 0\)):** 损失函数为零，低质量答案不参与参数更新。

RFT 通过最小化损失函数，引导模型专注于学习高质量答案的生成模式，实现选择性学习。

#### 数据生成

* **高质量数据 (推理数据):**  通过 RL 模型生成候选答案，使用奖励模型（或 DeepSeek-V3 模型）评分，**拒绝采样保留高分答案**。
* **SFT 数据 (非推理数据):** 复用 DeepSeek-V3 的 SFT 数据集及其生成流程。

#### 训练过程

- 使用拒绝采样得到的高质量数据集，对 DeepSeek-V3-Base 模型进行监督微调，**最小化 RFT 损失函数，提升模型推理和通用能力**。

- RFT 迭代精炼数据和重训练模型，期望模型在每轮迭代学习更高质量数据模式，最终收敛到高质量输出模型。 迭代过程中，训练数据分布 \(P_{sft}(Q, O)\) 逐渐聚焦于高质量数据，使模型在损失最小化过程中不断提升生成高质量输出的能力。


### OnRFT 

**在线拒绝采样微调 (Online Rejection Sampling Fine-tuning, OnRFT)** 目标与 RFT 类似，都是通过最小化选择性损失函数学习高质量输出模式。OnRFT 与 RFT 的主要区别在于数据采样方式，损失函数形式与 RFT 保持一致。OnRFT 的损失函数梯度为：

\[
\nabla_{\theta} \mathcal{L}_{OnRFT}(\theta) = - \mathbb{E}_{(q, o) \sim P_{sft}(Q) \times \pi_{\theta}(O \mid q)}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \mathbb{I}(o) \nabla_{\theta} \log \pi_{\theta}\left(o_t \mid q, o_{&lt;t}   \right)\right]
\]

**参数说明:**

* **\(\nabla_{\theta} \mathcal{L}_{OnRFT}\)**: OnRFT 损失函数关于模型参数 \(\theta\) 的梯度，指示损失函数下降方向。
* **\(\pi_{\theta}(O \mid q)\)**: 给定问题 \(q\)，**当前训练模型** 生成输出 \(O\) 的条件概率分布。

### RFT 与 OnRFT 对比

下面表格简单对比了 RFT 和OnRFT 的主要区别。

| 特性             | RFT  | OnRFT  |
| ------------------ | ------------------------------------ | --------------------------------------- |
| **数据生成方式**     | 离线 (Offline)                       | 在线 (Online)                          |
| **数据生成模型**     | SFT 模型 \(\pi_{sft}\)               | 当前训练模型 \(\pi_{\theta}\)                 |
| **拒绝采样数据来源**   | 预生成 SFT 数据集                  | 训练时实时生成数据                      |
| **数据循环**       | 分离                                 | 在线循环                               |
| **损失函数机制**     | 选择性交叉熵损失，选择高质量输出学习         | 选择性交叉熵损失，选择高质量输出学习             |
| **训练数据分布变化** | 逐渐聚焦于高质量数据                 | 动态变化，贴合当前模型能力                  |


### 面向所有场景的强化学习

为了进一步对齐人类偏好，DeepSeek 团队进行了第二阶段 RL，目标是在最大化奖励函数的同时，提升模型的有用性 (Helpfulness) 和 无害性 (Harmlessness)，并兼顾推理能力。此阶段仍然是通过最大化奖励函数来指导模型训练，但奖励函数的设计更加复杂，以反映多维度的优化目标。

此阶段的 RL 训练结合了：

* **多样化的 Prompt 分布:**  覆盖更广泛的场景，包括推理和通用任务。
* **多目标奖励信号:**
    * **推理数据:**  沿用基于规则的任务奖励，侧重推理准确性。最大化任务奖励，引导模型最小化推理错误。
    * **通用数据:**  使用奖励模型捕捉人类对有用性和无害性的偏好。奖励模型的目标是学习人类偏好，并输出与人类偏好一致的奖励信号，RL 训练的目标是最大化奖励模型给出的奖励值，从而间接最小化模型输出与人类偏好之间的偏差。

### 蒸馏

为了将 DeepSeek-R1 的强大推理能力迁移到更高效的小型模型上，DeepSeek 团队采用了**蒸馏（Distillation）**（[Hinton et al., 2015](https://arxiv.org/abs/1503.02531)）技术。蒸馏过程主要包括以下步骤：

1. **数据生成 (Data Generation)**: 利用训练好的 DeepSeek-R1 模型，生成约 **80 万条**高质量的推理数据。这些数据不仅包括推理密集型任务 (如数学题、编程题)，也涵盖了通用任务 (如问答、对话)，以保证蒸馏数据的多样性和覆盖面。

2. **模型微调 (Model Fine-tuning)**: 将生成的 80 万条高质量推理数据，用于微调小型密集模型。蒸馏实验选择了 Qwen 和 Llama 系列模型作为 Student 模型，涵盖了从 1.5B 到 70B 参数的多种模型规模，以探索蒸馏技术在不同模型规模下的效果。选取的 Student 模型包括 Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, 和 **Llama-3.3-70B-Instruct**。

3. **性能评估 (Performance Evaluation)**: 在多个推理相关的 benchmark 中，对蒸馏后的模型进行全面的性能评估。评估结果旨在验证蒸馏技术是否能够有效地将大型模型的推理能力迁移到小型模型，并考察蒸馏后的小型模型在推理能力上是否能够达到甚至超越大型模型的水平。

#### KL 散度蒸馏

除了直接使用 Teacher 模型生成的文本输出作为伪标签进行 SFT 蒸馏外，更严谨的方法是将 Teacher 模型生成的 token 概率分布 \(\pi_{\text{teacher}}\) 也纳入考虑。**KL 散度蒸馏** 是一种常用的方法，它不仅让 Student 模型学习 Teacher 模型的文本输出，也学习 Teacher 模型的 token 概率分布。通过最小化 Student 模型和 Teacher 模型输出概率分布之间的 KL 散度，可以更充分地将 Teacher 模型的知识迁移到 Student 模型中。但在实际工程中，直接使用 Teacher 模型的文本输出作为伪标签进行 SFT 蒸馏，通常也能取得足够好的效果，并且实现更简单。

#### 实验结果

实验结果如图6所示：

{{< figure
    src="deepseek_r1_distill_comparison.png"
    caption="Fig. 6. Comparison of DeepSeek-R1 distilled models and other comparable models on reasoning-related benchmarks. (Image source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))"
    align="center"
    width="90%"
>}}

实验结果表明，这种**直接的 SFT 蒸馏方法能够显著提升小型模型的推理能力**。值得注意的是，在除 CodeForces 之外的多个基准测试中，蒸馏后的 Llama-3.3-70B-Instruct 模型表现已超越 OpenAI-o1-mini。仅依靠 SFT 蒸馏更大的基座模型就能获得如此显著的提升，充分展现了这一方法在后续研究与应用中的潜力。

## 讨论

DeepSeek-R1 在多阶段训练框架基础上，探索了 Reasoning Model 训练技术的简化路径，主要包括以下几点:

**线性化思维过程：CoT 替代 MCTS**
- 传统强化学习 AI，如围棋和象棋，曾依赖蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS)。DeepSeek-R1 等模型则探索使用自回归的链式思维方法简化推理过程，逐步摒弃了计算复杂度高的 MCTS。
- CoT 将复杂推理分解为线性步骤，模型像解题一样逐步推理，而非 MCTS 的穷举式搜索。这种线性化思维降低了计算复杂度，更符合人类思维习惯，使模型更易学习复杂推理策略。

**消除独立价值网络：简化 RL 架构**
- 传统强化学习 (如 PPO) 通常需要独立的策略网络和价值网络。DeepSeek-R1 等研究发现，强化的策略网络或简化的价值评估方法 (如 GRPO 的组相对评分) 可替代独立价值网络。
- 这简化了 RL 训练架构，降低了资源需求，提高了效率。表明大语言模型的策略网络已具备强大的价值评估能力，无需额外价值网络。

**聚焦最终结果奖励：最小化奖励信号**
- DeepSeek-R1 采用更加简单的 ORM 奖励策略，主要关注最终结果的准确性奖励，弱化中间推理步骤奖励。这种策略受 AlphaZero ([Silver et al., 2017](https://arxiv.org/abs/1712.01815)) 启发，后者仅关注胜负。
- 对于 Reasoning Model，最终结果奖励可能比 PRM 更有效，能帮助模型更自然地学习“思维方式”，减少繁琐的逐步监督。

**增加思考时间：模型自发涌现深度思考**
- DeepSeek-R1-Zero 训练中展现出自发 **增加思考时间** 的能力。模型随训练深入，根据问题复杂度自适应分配更多“思考时间”，生成更长推理序列。这种“思考时间”增加是模型 RL 训练中自发涌现的行为。
- 思考时间增加反映模型更深入探索和优化思维过程。复杂问题需要更多推理步骤才能找到答案。DeepSeek-R1-Zero 的自我进化能力印证了强化学习在提升模型推理能力方面的潜力。

## 总结

DeepSeek-R1 的成功展示了 RL 提升 LLM 推理能力的巨大潜力。DeepSeek-R1 采用的 GRPO 算法在计算效率、优化稳定性、奖励鲁棒性等方面优于 PPO 和 DPO，并通过简化模型架构降低了训练资源消耗。DeepSeek-R1 为开源 Reasoning Model 复现 o1 提供了一条值得参考的路径。

## 参考文献

[1] [OpenAI o1](https://openai.com/o1/). OpenAI, 2024. (OpenAI O1 official introduction page)

[2] Jaech, Aaron, et al. ["OpenAI o1 system card."](https://arxiv.org/abs/2412.16720) arXiv preprint arXiv:2412.16720 (2024).

[3] [Open-r1](https://github.com/huggingface/open-r1). GitHub, 2024. (Open-r1 open source project GitHub repository)

[4] Sutton, Richard. ["The bitter lesson."](http://incompleteideas.net/IncIdeas/BitterLesson.html) Incomplete Ideas (blog) 13.1 (2019): 38.

[5] Liu A, et al. ["Deepseek-v3 technical report."](https://arxiv.org/abs/2412.19437) arXiv preprint arXiv:2412.19437 (2024).

[6] Schulman, John, et al. ["Proximal policy optimization algorithms."](https://arxiv.org/abs/1707.06347) arXiv preprint arXiv:1707.06347 (2017).

[7] Ouyang, Long, et al. ["Training language models to follow instructions with human feedback."](https://arxiv.org/abs/2203.02155) Advances in neural information processing systems 35 (2022): 27730-27744.

[8] Shao, Zhihong, et al. ["Deepseekmath: Pushing the limits of mathematical reasoning in open language models."](https://arxiv.org/abs/2402.03300) arXiv preprint arXiv:2402.03300 (2024).

[9] J. Schulman. [Approximating kl divergence]("http://joschu.net/blog/kl-approx.html"), 2020.

[10] Gao, Leo, John Schulman, and Jacob Hilton. ["Scaling laws for reward model overoptimization."](https://proceedings.mlr.press/v202/gao23b.html) International Conference on Machine Learning. PMLR, 2023.

[11] Chen, Mark, et al. ["Evaluating large language models trained on code."](https://arxiv.org/abs/2107.03374) arXiv preprint arXiv:2107.03374 (2021).

[12] [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/). OpenAI, 2024. (OpenAI blog post about LLM reasoning ability)

[13] [AMC](https://maa.org/student-programs/amc/). Mathematical Association of America (MAA), 2024. (American Mathematics Competitions AMC official website)

[14] [Open-O1](https://github.com/Open-Source-O1/Open-O1?tab=readme-ov-file). GitHub, 2024. (Open-O1 open source project GitHub repository)

[15] Zeng, Zhiyuan, et al. ["Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective."](https://arxiv.org/abs/2412.14135) arXiv preprint arXiv:2412.14135 (2024).

[16] Hinton, Geoffrey. ["Distilling the Knowledge in a Neural Network."](https://arxiv.org/abs/1503.02531) arXiv preprint arXiv:1503.02531 (2015).

[17] Silver, David, et al. ["Mastering chess and shogi by self-play with a general reinforcement learning algorithm."](https://arxiv.org/abs/1712.01815) arXiv preprint arXiv:1712.01815 (2017).


## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Jan 2025). OpenAI o1复现进展：DeepSeek-R1.
https://syhya.github.io/posts/2025-01-27-deepseek-r1

Or

```bibtex
@article{syhya2025deepseekr1,
  title   = "OpenAI o1复现进展：DeepSeek-R1",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jan",
  url     = "https://syhya.github.io/posts/2025-01-27-deepseek-r1"
}