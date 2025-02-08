---
title: "大语言模型对齐: 直接偏好优化(DPO)"  
date: 2025-02-08T12:00:00+08:00  
author: "Yue Shui"  
tags: ["AI", "NLP", "LLM", "Post-training", "DPO", "RLHF", "Alignment", "Bradley–Terry Model"]  
categories: ["技术博客"]  
readingTime: 25  
toc: true  
ShowToc: true  
TocOpen: false  
draft: false  
type: "posts"
---

这篇博客主要介绍一种比 RLHF 更精简的替代算法 DPO。与 RLHF 一样，DPO 目的是使模型输出与人类偏好保持一致，但它在实现上更加简单，并且对资源的需求更低。在项目资源受限的情况下，DPO 是一个实用解决方案。

## 符号


| 符号                              | 含义  |
| --------------------------------- | ------------------------------- |
| \( x \)                           | 用户输入（Prompt），即模型需要回答的问题 |
| \( y \)                           | 模型生成的回答（Response / Completion），即模型输出的文本 |
| \( \pi_\theta(y \mid x) \)         | Actor 模型：待训练策略，用于生成回答 \(y\)；参数为 \(\theta\) |
| \( \pi_{\mathrm{ref}}(y \mid x) \)  | 参考模型：冻结的 SFT 模型，作为对齐基准 |
| \( r_\phi(x,y) \)                 | 奖励模型：用于评估回答 \(y\) 质量的奖励函数；参数为 \(\phi\) |
| \( V_\psi(x) \)                   | critic 模型：用于估计给定输入 \(x\) 下未来累计奖励的值函数；参数为 \(\psi\) |
| \( \pi^*(y \mid x) \)              | 最优策略分布，通过参考模型与奖励函数确定 |
| \( r_\theta(x,y) \)               | 基于 Actor 模型导出的奖励函数，通过 \(\pi_\theta\) 与 \(\pi_{\mathrm{ref}}\) 构造 |
| \(\beta\)                         | 超参数，控制 KL 惩罚项或对数比差异项的权重 |
| \(\mathbb{D}_{\mathrm{KL}}[P \| Q]\)| KL 散度，衡量概率分布 \(P\) 与 \(Q\) 之间的差异 |
| \(\sigma(z)\)                     | Sigmoid 函数，定义为：\(\sigma(z)=\frac{1}{1+e^{-z}}\) |
| \(\log\)                          | 对数函数 |
| \(\mathbb{E}\)                    | 期望算子，用于求随机变量的平均值 |
| \( (y_w, y_l) \)                  | 一对偏好数据，其中 \( y_w \) 表示被偏好（质量更好）的回答，\( y_l \) 表示质量较差的回答 |
| \( P\left(y_w \succ y_l \mid x\right) \) | 在输入 \(x\) 下，回答 \( y_w \) 优于 \( y_l \) 的概率 |
| \( Z(x) \)                        | 配分函数，对所有回答 \(y\) 归一化概率分布 |
| \( \mathcal{L}_{\mathrm{DPO}} \)   | DPO 的损失函数 |


## 从 RLHF 到 DPO

### RLHF

OpenAI 主要利用人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）([Christiano et al., 2017](https://arxiv.org/abs/1706.03741))来训练 InstructGPT ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155))，而其构成了大语言模型（如 ChatGPT, Llama 等）的基础。整个训练过程通常包括以下三个主要步骤：

{{< figure
    src="InstructGPT.png"
    caption="Fig. 1. A diagram illustrating the three steps of InstructGPT. (Image source: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155))"
    align="center"
    width="100%"
>}}

1. **监督微调（SFT）**  
   利用大量人工示例数据对预训练模型进行微调，得到一个初步能理解指令并生成合理回答的模型，即参考模型 \( \pi_{\mathrm{ref}}(y \mid x) \) 。

2. **奖励模型训练**  
   这里我们简化只考虑生成两个不同的结果，实际可以生成多个结果进行排序。针对同一输入 \(x\) 生成两个回答 \(y_w\)（较优）和 \(y_l\)（较劣），由人工排序后收集偏好数据。基于这些数据训练奖励模型 \(r_\phi(x, y)\)，使其能预测哪种回答更符合人类偏好。

3. **基于 PPO 的强化学习**  
   利用奖励模型 \(r_\phi\) 提供的反馈，通过 PPO 算法优化 Actor 模型 \(\pi_\theta\) 以提升回答质量。为防止模型偏离 \(\pi_{\mathrm{ref}}\)，在优化过程中引入 KL 惩罚项。该阶段通常涉及以下 4 个模型：  
   - \(\pi_\theta\)：经过 SFT 后待更新的模型。 
   - \(\pi_{\mathrm{ref}}\)：冻结的 SFT 模型，作为对齐基准。 
   - \(r_\phi\)：用于评估回答质量，参数固定。 
   - \(V_\psi\)：用于估计未来奖励，辅助 Actor 模型更新。

### RLHF 的局限性

尽管 RLHF 能充分利用人类偏好信息提升模型对齐效果，但其固有局限性包括：

- **多模型训练**：除 Actor 模型 \(\pi_\theta\) 外，还需额外训练奖励模型 \(r_\phi\) 和 Critic 模型 \(V_\psi\)，整体训练过程复杂且资源消耗大。  
- **高采样成本**：LLM 生成文本计算量大，强化学习过程中的大量在线采样进一步推高了计算开销；采样不足可能导致错误的优化方向。  
- **训练不稳定与超参数敏感**：PPO 涉及众多超参数（如学习率、采样量等），调参复杂且训练过程易受不稳定因素影响。  
- **对齐税效应**：在提高模型对齐性的同时，可能会降低模型在其他任务上的表现。

{{< figure
    src="rlhf_dpo.png"
    caption="Fig. 2. DPO optimizes for human preferences while avoiding reinforcement learning.（Image source: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）"
    align="center"
    width="100%"
>}}


### DPO 简介

直接偏好优化（Direct Preference Optimization, DPO）（[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）为了解决 RLHF 的上述问题，其核心思路是将 RLHF 的目标转化为类似于监督微调的对比学习任务，从而实现：

- **省略奖励模型训练**：直接利用人类偏好数据优化 Actor 模型 \(\pi_\theta\)，无须单独训练 \(r_\phi\)。  
- **消除强化学习采样**：采用对比损失函数替代 PPO，降低采样和计算资源消耗。
- **提升训练稳定性**：基于监督学习的方法对超参数不敏感，训练过程更加平稳。

虽然 DPO 在 LLM 性能提升的上限上可能不及 RLHF，但在资源利用、实现复杂度和训练稳定性方面具有优势。

### 方法对比

| 方法         | 训练步骤                             | 模型                                | 训练方式                | 优点                                  | 缺点                                  |
| ------------ | ------------------------------------ | --------------------------------------- | ----------------------- | ------------------------------------- | ------------------------------------- |
| **RLHF**     | 先训练奖励模型，再使用 PPO 优化策略   | \(\pi_\theta\)、\(\pi_{\mathrm{ref}}\)、\(r_\phi\)、\(V_\psi\) | 强化学习和在线采样      | 充分利用人类偏好，上限潜力较高        | 资源消耗大、训练不稳定、超参数敏感       |
| **DPO**      | 直接利用偏好数据训练 Actor 模型      | \(\pi_\theta\)、\(\pi_{\mathrm{ref}}\)         | 类似 SFT 监督学习      | 流程简化、训练稳定、资源消耗低         | 性能提升上限可能低于 RLHF          |


## DPO 数学推导

### RLHF 目标与最优策略分布

在大规模语言模型对齐中，我们希望利用人类反馈强化学习（RLHF）来优化模型输出。设输入 \( x \) 来自数据集 \(\mathcal{D}\)，模型生成回答 \( y \)；待训练的 模型记为 \(\pi_\theta(y \mid x)\)，而参考模型记为 \(\pi_{\mathrm{ref}}(y \mid x)\)（通常为SFT模型），同时引入奖励函数 \( r(x,y) \) 衡量回答质量。RLHF 的目标可写为

\[
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)} \Big[ r(x,y) \Big] \;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\Big[ \pi(y \mid x) \,\|\, \pi_{\mathrm{ref}}(y \mid x) \Big],
\tag{1}
\]

其中 \(\beta\) 为调节奖励与参考模型偏差的超参数。利用 KL 散度的定义

\[
\mathbb{D}_{\mathrm{KL}} \Big[\pi(y \mid x) \,\|\, \pi_{\mathrm{ref}}(y \mid x)\Big] = \mathbb{E}_{y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \right],
\tag{2}
\]

式 (1) 可重写为

\[
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)} \left[ r(x,y) - \beta \, \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \right].
\tag{3}
\]

将 (3) 式转换为最小化问题并除以 \(\beta\) 得

\[
\min_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} - \frac{1}{\beta} r(x,y) \right].
\tag{4}
\]

假设存在一个最优策略分布 \(\pi^*(y \mid x)\) 使 (4) 式全局最优，则令

\[
\pi^*(y \mid x) \;=\; \frac{1}{Z(x)} \,\pi_{\mathrm{ref}}(y \mid x)\, \exp\!\Big(\frac{1}{\beta} \, r(x,y)\Big),
\tag{5}
\]

其中配分函数 \( Z(x) \) 定义为

\[
Z(x) = \sum_{y}\, \pi_{\mathrm{ref}}(y \mid x)\, \exp\!\Big(\frac{1}{\beta} \, r(x,y)\Big).
\tag{6}
\]

- \(Z(x)\) 对所有可能的 \(y\) 求和，实现归一化，使得 \(\pi^*(y \mid x)\) 构成合法概率分布。  
- \(Z(x)\) 是 \(x\) 的函数，与待优化的 Actor 模型 \(\pi_\theta\) 无关。

对 (5) 式取对数得到

\[
\log \pi^*(y \mid x) = \log \pi_{\mathrm{ref}}(y \mid x) + \frac{1}{\beta}\, r(x,y) - \log Z(x),
\tag{7}
\]

从而解得

\[
r(x,y) = \beta \left[\log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \log Z(x)\right].
\tag{8}
\]

### Bradley–Terry 模型

为了利用成对偏好数据 \((x, y_w, y_l)\) 训练模型，我们希望在相同输入 \( x \) 下，模型输出更偏好于高质量回答 \( y_w \) 而不是低质量回答 \( y_l \)。 

[Bradley–Terry 模型](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)用于预测成对比较的结果。对于任意两个对象 \( i \) 和 \( j \)，若为每个对象分配正实数得分 \( p_i \) 和 \( p_j \)，则对象 \( i \) 被认为比对象 \( j \) 强的概率为

\[
\Pr(i > j) = \frac{p_i}{p_i + p_j}.
\tag{9}
\]

在我们的场景中，将每个回答 \( y \) 的强度参数设为 \( p_{y} = \exp\big(r(x,y)\big) \)（保证为正实数）。因此，给定输入 \( x \) 下，回答 \( y_w \) 好于 \( y_l \) 的概率为

\[
P\left(y_w \succ y_l \mid x\right)=\frac{\exp \big[r(x,y_w)\big]}{\exp \big[r(x,y_w)\big]+\exp \big[r(x,y_l)\big]}.
\tag{10}
\]

为了使得数据集中每个成对偏好数据 \((x, y_w, y_l)\) 中，高质量回答 \( y_w \) 的胜出概率尽可能大，我们将奖励模型训练目标设计为最大化 \( y_w \) 被偏好的概率，或等价地最小化负对数似然损失：

\[
L_{R}\left(r_{\phi}, D\right) = -\mathbb{E}_{(x,y_w,y_l) \sim D}\left[\log P\left(y_w \succ y_l \mid x\right)\right],
\tag{11}
\]
其中数据集定义为
\[
D=\{(x^i, y_w^i, y_l^i)\}_{i=1}^{N}.
\tag{12}
\]

利用公式 (10)、(11) 以及下面的恒等式

\[
\log \frac{e^a}{e^a+e^b} = \log\frac{1}{1+e^{b-a}} = \log \sigma(a-b),
\tag{13}
\]

其中 Sigmoid 函数定义为

\[
\sigma(z)=\frac{1}{1+e^{-z}},
\tag{14}
\]

可得

\[
\log P\left(y_w \succ y_l \mid x\right) = \log \sigma\Big(r(x,y_w)-r(x,y_l)\Big).
\tag{15}
\]


### 直接偏好优化

注意到 (8) 式中，奖励 \( r(x,y) \) 与最优策略的对数比有关。为避免显式训练一个单独的奖励模型 \(r_\phi\)，我们采用 DPO的思想，即**直接用待训练 Actor 模型 \(\pi_\theta\) 替换最优策略 \(\pi^*\) 的位置**，将 (8) 式中的奖励表示为

\[
r_\theta(x,y) \;=\; \beta \left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \log Z(x)\right].
\tag{16}
\]

在成对比较中，对于相同输入 \( x \)，两个回答 \( y_w \) 和 \( y_l \) 均包含相同的 \(\log Z(x)\) 项，因此在计算奖励差值时，该项会被消去，即

\[
\begin{aligned}
r_\theta(x,y_w)-r_\theta(x,y_l)
&=\; \beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} + \log Z(x)\right] - \beta \left[\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} + \log Z(x)\right] \\
&=\; \beta \,\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \,\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}.
\end{aligned}
\tag{17}
\]

将 (17) 式代入 (15) 式，并结合 (11) 式，我们最终得到 DPO 的损失函数

\[
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) 
= - \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma\Big(
\beta \,\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} 
-\; \beta \,\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\Big) \right].
\tag{18}
\]

该损失函数针对待训练 Actor 模型 \(\pi_\theta\) 而设，通过比较其在高质量回答 \(y_w\) 与低质量回答 \(y_l\) 上、相对于参考模型 \(\pi_{\mathrm{ref}}\) 的对数概率比来区分好坏回答：当 \(\pi_\theta\) 在 \(y_w\) 上的相对概率比远大于 \(y_l\) 时，Sigmoid 输出更接近 1，损失更小；反之，若 \(\pi_\theta\) 对 \(y_l\) 的相对概率过大，则损失增加。参数 \(\beta\) 用于放大或缩小这些对数比的差值，从而调节模型对好坏答案的区分强度。

##  训练流程

### 数据准备

- **偏好数据采集**  
  收集大量偏好数据，每个样本包含输入 \( x \) 及一对回答 \((y_w, y_l)\)，其中 \( y_w \) 为被偏好回答(chosen)，\( y_l \) 为质量较差的回答(rejected)。

  **数据集例子**: [dpo_en_demo.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_en_demo.json)
  
   ```json
   {
    "conversations": [
      {
        "from": "human",
        "value": "what does abstraction mean on oop"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "Abstraction in Object-Oriented Programming (OOP) is the process of exposing only the necessary information to the outside world while hiding the implementation details. It's a way to simplify complex systems by focusing only on the essential aspects and encapsulating the implementation details.\n\nIn other words, abstraction is about hiding the internal workings of an object or system and only exposing the necessary interfaces or APIs to interact with it. This helps to reduce complexity, improve modularity, and make the system easier to understand and maintain.\n\nFor example, consider a car. A car has many components like engine, transmission, brakes, etc. However, when you drive a car, you don't need to know how the engine works or how the transmission changes gears. All you need to know is how to operate the car's controls like accelerator, brakes, and steering wheel. The car's controls are an abstraction of the underlying complex systems, allowing you to interact with the car without needing to know the details of how it works.\n\nIn programming, abstraction is achieved by defining interfaces or abstract classes that define the essential functions or behaviors of an object without specifying how they are implemented. This allows developers to create different implementations of the same interface or abstract class, which can be swapped out without affecting the rest of the system.\n\nAbstraction is a fundamental concept in OOP and is used to create modular, maintainable, and flexible software systems. It helps to reduce code duplication, improve code reuse, and make software systems more understandable and easier to evolve over time."
    },
    "rejected": {
      "from": "gpt",
      "value": "Abstraction is a principle in object-oriented programming that refers to the process of focusing on the essential features of an object or concept without emphasizing the details that are not important for its functionality. It enables programmers to create classes that represent the objects they want to work with, and specifies only the behavior of those objects, leaving out unnecessary details. Abstraction helps to make the code more maintainable, modular, and scalable. It also improves the productivity of developers by reducing the amount of code they need to write."
    }
  }
   ```

- **输出预计算**  
  为降低计算开销，预先计算并缓存参考模型 \(\pi_{\mathrm{ref}}(y \mid x)\) 的输出。

### 模型训练

- **训练目标**  
  通过最小化 DPO 损失 \(\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}})\) 直接优化 Actor 模型 \(\pi_\theta\)，使其生成的回答更符合人类偏好。

- **训练步骤**  
  1. 从数据集中采样一批 \((x, y_w, y_l)\) 数据。  
  2. 计算 Actor 模型 \(\pi_\theta(y \mid x)\) 的输出概率。  
  3. 利用下式计算损失：
     
     \[
     \mathcal{L}_{\mathrm{DPO}} = - \log \sigma\Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \Big).
     \]
     
  4. 通过反向传播更新 Actor 模型参数 \(\theta\)。

### 模型推理

训练完成后，得到的 Actor 模型 \(\pi_\theta\) 可直接用于推理。给定输入 \( x \) 后，模型基于学到的概率分布生成回答。由于训练过程中参考了人类偏好，同时受到参考模型 \(\pi_{\mathrm{ref}}\) 的约束，生成的回答既符合预期，又能保持生成文本的稳定性。


## 总结

DPO 将 RLHF 过程简化为直接的监督学习任务，不仅节省了资源、提高了训练稳定性，同时降低了实现复杂度，是 LLM 对齐训练的一种高效替代方法。在实际应用中，我们可以根据业务场景选择 RLHF 或 DPO 方法，以达到最佳的训练效果。


## 参考文献

[1] Christiano, Paul F., et al. ["Deep reinforcement learning from human preferences."](https://arxiv.org/abs/1706.03741) Advances in neural information processing systems 30 (2017).

[2] Ouyang, Long, et al. ["Training language models to follow instructions with human feedback."](https://arxiv.org/abs/2203.02155) Advances in neural information processing systems 35 (2022): 27730-27744.

[3] Rafailov, Rafael, et al. ["Direct preference optimization: Your language model is secretly a reward model."](https://arxiv.org/abs/1706.03741) Advances in Neural Information Processing Systems 36 (2024).

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Feb 2025). 大语言模型对齐: 直接偏好优化(DPO).
https://syhya.github.io/posts/2025-02-08-dpo

Or

```bibtex
@article{syhya2025dpo,
  title   = "大语言模型对齐: 直接偏好优化(DPO)",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://syhya.github.io/posts/2025-02-08-dpo"
}
