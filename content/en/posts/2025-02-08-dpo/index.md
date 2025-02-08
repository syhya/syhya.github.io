---
title: "LLMs Alignment: DPO"  
date: 2025-02-08T12:00:00+08:00  
author: "Yue Shui"  
tags: ["AI", "NLP", "LLMs", "Post-training", "DPO", "RLHF", "Alignment", "Bradley–Terry Model"]  
categories: ["Technical Blog"]  
readingTime: 25  
toc: true  
ShowToc: true  
TocOpen: false  
draft: false  
type: "posts"
---

This blog post introduces a streamlined alternative to RLHF called DPO. Like RLHF, DPO is designed to align model outputs with human preferences, but it stands apart with its simplicity and lower resource demands. In scenarios where project resources are limited, DPO emerges as a highly attractive and practical solution worth exploring.

## Notations

|Symbol                             |                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| \( x \)                          | User input (Prompt): the question the model needs to answer                                             |
| \( y \)                          | Model-generated response (Response / Completion): the text output by the model                           |
| \( \pi_\theta(y \mid x) \)         | Actor model: The trainable policy used to generate response \(y\); parameterized by \(\theta\)                 |
| \( \pi_{\mathrm{ref}}(y \mid x) \)  | Reference model: The frozen SFT (Supervised Fine-Tuning) model, serving as the alignment baseline                |
| \( r_\phi(x,y) \)                 | Reward model: A reward function (with parameter \(\phi\)) used to evaluate the quality of response \(y\)        |
| \( V_\psi(x) \)                   | Critic model: A value function (with parameter \(\psi\)) used to estimate the future cumulative reward given \(x\)|
| \( \pi^*(y \mid x) \)              | Optimal policy distribution, determined via the reference model and reward function                              |
| \( r_\theta(x,y) \)               | Reward derived from the Actor model, constructed from \(\pi_\theta\) and \(\pi_{\mathrm{ref}}\)                   |
| \(\beta\)                         | Hyperparameter that controls the weight of the KL penalty or the log-ratio difference term                        |
| \(\mathbb{D}_{\mathrm{KL}}[P \| Q]\)| KL divergence, a measure of the difference between probability distributions \(P\) and \(Q\)                       |
| \(\sigma(z)\)                     | Sigmoid function, defined as: \(\sigma(z)=\frac{1}{1+e^{-z}}\)                                                |
| \(\log\)                          | Logarithm function                                                                                            |
| \(\mathbb{E}\)                    | Expectation operator, used to compute the average value of a random variable                                      |
| \( (y_w, y_l) \)                  | A pair of preference data where \( y_w \) is the preferred (better quality) response and \( y_l \) is the lesser one  |
| \( P\left(y_w \succ y_l \mid x\right) \) | The probability that response \( y_w \) is preferred over \( y_l \) given input \(x\)                      |
| \( Z(x) \)                        | Partition function, which normalizes the probability distribution over all responses \(y\)                         |
| \( \mathcal{L}_{\mathrm{DPO}} \)   | The loss function of DPO                                                                             |



## From RLHF to DPO

### RLHF

OpenAI primarily leverages Reinforcement Learning from Human Feedback (RLHF) ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741)) to train InstructGPT ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)), which forms the basis for LLMs (such as ChatGPT, Llama, etc.). The entire training process generally comprises the following three main steps:

{{< figure
    src="InstructGPT.png"
    caption="Fig. 1. A diagram illustrating the three steps of InstructGPT. (Image source: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155))"
    align="center"
    width="100%"
>}}

1. **Supervised Fine-Tuning (SFT)**  
   A pre-trained model is fine-tuned using a large volume of human-annotated examples, resulting in an initial model capable of understanding instructions and generating reasonable responses. This model is referred to as the reference model, \(\pi_{\mathrm{ref}}(y \mid x)\).

2. **Reward Model Training**  
   For simplicity, assume that for each input \(x\), two distinct responses are generated. In practice, multiple responses can be ranked. Two responses \(y_w\) (better) and \(y_l\) (worse) are generated for the same input \(x\), and human ranking provides the preference data. A reward model \(r_\phi(x, y)\) is then trained on this data to predict which response aligns better with human preferences.

3. **PPO-Based Reinforcement Learning**  
   Using feedback from the reward model \(r_\phi\), the Actor model \(\pi_\theta\) is optimized via the Proximal Policy Optimization (PPO) algorithm to improve response quality. To prevent the model from deviating too far from \(\pi_{\mathrm{ref}}\), a KL penalty is added during optimization. This stage typically involves the following four models:  
   - \(\pi_\theta\): The model (after SFT) that is updated.  
   - \(\pi_{\mathrm{ref}}\): The frozen SFT model used as the alignment baseline.  
   - \(r_\phi\): The fixed reward model for evaluating response quality.  
   - \(V_\psi\): The critic model that estimates future rewards to assist the update of the Actor model.

### Limitations of RLHF

While RLHF leverages human preference data to enhance model alignment, it comes with several inherent limitations:

- **Multi-Model Training**: In addition to the Actor model \(\pi_\theta\), extra models such as the reward model \(r_\phi\) and the Critic model \(V_\psi\) must be trained, making the overall process complex and resource-intensive.  
- **High Sampling Cost**: LLMs require significant computational resources to generate text. The extensive online sampling during reinforcement learning further increases computational costs; insufficient sampling may lead to suboptimal optimization directions.  
- **Training Instability and Hyperparameter Sensitivity**: PPO involves numerous hyperparameters (e.g., learning rate, sampling batch size), making tuning complex and the training process prone to instability.  
- **Alignment Tax Effect**: While improving model alignment, the performance on other tasks may suffer.

{{< figure
    src="rlhf_dpo.png"
    caption="Fig. 2. DPO optimizes for human preferences while avoiding reinforcement learning. (Image source: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))"
    align="center"
    width="100%"
>}}

### Introduction to DPO

Direct Preference Optimization (DPO) ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) algorithm was developed to address the above issues of RLHF. Its core idea is to convert the RLHF objective into a contrastive learning task akin to supervised fine-tuning, thereby achieving the following:

- **Eliminating Reward Model Training**: Directly optimize the Actor model \(\pi_\theta\) using human preference data, without training a separate \(r_\phi\).  
- **Removing Reinforcement Learning Sampling**: Replace PPO with a contrastive loss function, reducing sampling and computational overhead.  
- **Enhancing Training Stability**: The supervised learning approach is less sensitive to hyperparameters, leading to a more stable training process.

Although DPO might have a lower performance ceiling compared to RLHF in terms of ultimate LLM performance improvements, it offers advantages in resource utilization, reduced implementation complexity, and training stability.

### Method Comparison

| Method  | Training Steps                                      | Models Involved                                      | Training Approach         | Advantages                                  | Disadvantages                          |
| ------- | --------------------------------------------------- | ---------------------------------------------------- | ------------------------- | ------------------------------------------- | -------------------------------------- |
| **RLHF**  | Train a reward model first, then use PPO to optimize the policy | \(\pi_\theta\), \(\pi_{\mathrm{ref}}\), \(r_\phi\), \(V_\psi\) | Reinforcement learning with online sampling | Fully leverages human preferences; higher performance potential | Resource intensive; unstable training; hyperparameter sensitive |
| **DPO**   | Directly train the Actor model using preference data | \(\pi_\theta\), \(\pi_{\mathrm{ref}}\)                         | Supervised-learning-like approach         | Simplified process; stable training; lower resource cost         | Performance ceiling may be lower than RLHF |


## Mathematical Derivation of DPO

### RLHF Objective and the Optimal Policy Distribution

In the alignment of large language models, our goal is to use RLHF to optimize model outputs. Let the input \( x \) be drawn from a dataset \(\mathcal{D}\), and let the model generate a response \( y \). Denote the trainable model as \(\pi_\theta(y \mid x)\) and the reference model as \(\pi_{\mathrm{ref}}(y \mid x)\) (typically the SFT model). We also introduce a reward function \( r(x,y) \) to measure the quality of a response. The RLHF objective can be written as

\[
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)} \Big[ r(x,y) \Big] \;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\Big[ \pi(y \mid x) \,\|\, \pi_{\mathrm{ref}}(y \mid x) \Big],
\tag{1}
\]

where \(\beta\) is a hyperparameter that balances the reward and the deviation from the reference model. Using the definition of KL divergence,

\[
\mathbb{D}_{\mathrm{KL}} \Big[\pi(y \mid x) \,\|\, \pi_{\mathrm{ref}}(y \mid x)\Big] = \mathbb{E}_{y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \right],
\tag{2}
\]

we can rewrite Equation (1) as

\[
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)} \left[ r(x,y) - \beta \, \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \right].
\tag{3}
\]

Converting (3) to a minimization problem and dividing by \(\beta\) yields

\[
\min_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} - \frac{1}{\beta} r(x,y) \right].
\tag{4}
\]

Assuming there exists an optimal policy distribution \(\pi^*(y \mid x)\) that globally minimizes (4), we set

\[
\pi^*(y \mid x) \;=\; \frac{1}{Z(x)} \,\pi_{\mathrm{ref}}(y \mid x)\, \exp\!\Big(\frac{1}{\beta} \, r(x,y)\Big),
\tag{5}
\]

where the partition function \( Z(x) \) is defined as

\[
Z(x) = \sum_{y}\, \pi_{\mathrm{ref}}(y \mid x)\, \exp\!\Big(\frac{1}{\beta} \, r(x,y)\Big).
\tag{6}
\]

- \(Z(x)\) sums over all possible \(y\) to normalize the distribution, ensuring that \(\pi^*(y \mid x)\) is a valid probability distribution.  
- \(Z(x)\) is a function of \(x\) and is independent of the trainable Actor model \(\pi_\theta\).

Taking the logarithm of (5) gives

\[
\log \pi^*(y \mid x) = \log \pi_{\mathrm{ref}}(y \mid x) + \frac{1}{\beta}\, r(x,y) - \log Z(x),
\tag{7}
\]

which can be rearranged to obtain

\[
r(x,y) = \beta \left[\log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \log Z(x)\right].
\tag{8}
\]

### The Bradley–Terry Model

To leverage pairwise preference data \((x, y_w, y_l)\) for training, we aim for the model to favor the higher-quality response \( y_w \) over the lower-quality response \( y_l \) for the same input \( x \).

The [Bradley–Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) is used to predict the outcomes of paired comparisons. For any two items \( i \) and \( j \), if we assign each a positive score \( p_i \) and \( p_j \), then the probability that item \( i \) is preferred over item \( j \) is

\[
\Pr(i > j) = \frac{p_i}{p_i + p_j}.
\tag{9}
\]

In our scenario, we set the strength parameter for each response \( y \) as \( p_{y} = \exp\big(r(x,y)\big) \) (ensuring positivity). Therefore, given input \( x \), the probability that response \( y_w \) is preferred over \( y_l \) becomes

\[
P\left(y_w \succ y_l \mid x\right)=\frac{\exp \big[r(x,y_w)\big]}{\exp \big[r(x,y_w)\big]+\exp \big[r(x,y_l)\big]}.
\tag{10}
\]

To maximize the probability that the higher-quality response \( y_w \) wins in every preference pair \((x, y_w, y_l)\) in the dataset, we design the reward model’s training objective to maximize this probability or, equivalently, to minimize the negative log-likelihood:

\[
L_{R}\left(r_{\phi}, D\right) = -\mathbb{E}_{(x,y_w,y_l) \sim D}\left[\log P\left(y_w \succ y_l \mid x\right)\right],
\tag{11}
\]

where the dataset is defined as

\[
D=\{(x^i, y_w^i, y_l^i)\}_{i=1}^{N}.
\tag{12}
\]

Using Equations (10) and (11) along with the identity

\[
\log \frac{e^a}{e^a+e^b} = \log\frac{1}{1+e^{b-a}} = \log \sigma(a-b),
\tag{13}
\]

with the Sigmoid function defined as

\[
\sigma(z)=\frac{1}{1+e^{-z}},
\tag{14}
\]

we have

\[
\log P\left(y_w \succ y_l \mid x\right) = \log \sigma\Big(r(x,y_w)-r(x,y_l)\Big).
\tag{15}
\]

### Direct Preference Optimization

Notice from Equation (8) that the reward \( r(x,y) \) is related to the log-ratio of the optimal policy. To avoid explicitly training a separate reward model \(r_\phi\), DPO directly substitutes the trainable Actor model \(\pi_\theta\) in place of the optimal policy \(\pi^*\) and represents the reward as

\[
r_\theta(x,y) \;=\; \beta \left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \log Z(x)\right].
\tag{16}
\]

In pairwise comparisons, for the same input \( x \), both responses \( y_w \) and \( y_l \) contain the same \(\log Z(x)\) term; therefore, when computing the difference, this term cancels out:

\[
\begin{aligned}
r_\theta(x,y_w)-r_\theta(x,y_l)
&=\; \beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} + \log Z(x)\right] - \beta \left[\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} + \log Z(x)\right] \\
&=\; \beta \,\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \,\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}.
\end{aligned}
\tag{17}
\]

Substituting Equation (17) into (15) and combining with (11), we obtain the final DPO loss function:

\[
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) 
= - \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma\Big(
\beta \,\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} 
-\; \beta \,\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\Big) \right].
\tag{18}
\]

This loss function is designed for the trainable Actor model \(\pi_\theta\). It distinguishes between good and bad responses by comparing the log-probability ratios (relative to the reference model \(\pi_{\mathrm{ref}}\)) for the high-quality response \(y_w\) and the low-quality response \(y_l\): when \(\pi_\theta\) assigns a much higher relative probability to \(y_w\) compared to \(y_l\), the sigmoid output approaches 1 and the loss decreases; conversely, if the relative probability for \(y_l\) is too high, the loss increases. The parameter \(\beta\) scales the difference between these log-ratios, thereby adjusting the model’s sensitivity in distinguishing good from bad responses.


## Training Process

### Data Preparation

- **Preference Data Collection**  
  Collect a large amount of preference data. Each sample includes an input \( x \) and a pair of responses \((y_w, y_l)\), where \( y_w \) is the chosen (preferred) response and \( y_l \) is the rejected (lower quality) response.

  **Dataset Example**: [dpo_en_demo.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_en_demo.json)
  
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

- **Output Precomputation**  
  To reduce computational overhead, pre-compute and cache the outputs of the reference model \(\pi_{\mathrm{ref}}(y \mid x)\).

### Model Training

- **Training Objective**  
  Directly optimize the Actor model \(\pi_\theta\) by minimizing the DPO loss \(\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}})\) so that its generated responses better align with human preferences.

- **Training Steps**  
  1. Sample a batch of data \((x, y_w, y_l)\) from the dataset.  
  2. Compute the output probabilities of the Actor model \(\pi_\theta(y \mid x)\).  
  3. Calculate the loss using:
     
     \[
     \mathcal{L}_{\mathrm{DPO}} = - \log \sigma\Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \Big).
     \]
     
  4. Update the Actor model parameters \(\theta\) via backpropagation.

### Model Inference

Once training is complete, the resulting Actor model \(\pi_\theta\) can be used directly for inference. Given an input \( x \), the model generates responses based on the learned probability distribution. Since human preferences have been incorporated during training and the model is constrained by the reference model \(\pi_{\mathrm{ref}}\), the generated responses are not only aligned with expectations but also maintain stability in the generated text.


## Summary

DPO simplifies the RLHF process into a direct supervised learning task, saving resources, enhancing training stability, and reducing implementation complexity. It serves as an efficient alternative for LLM alignment training. In practical applications, one can choose between RLHF and DPO methods based on the specific business scenario to achieve the best training results.


## References

[1] Christiano, Paul F., et al. ["Deep reinforcement learning from human preferences."](https://arxiv.org/abs/1706.03741) Advances in neural information processing systems 30 (2017).

[2] Ouyang, Long, et al. ["Training language models to follow instructions with human feedback."](https://arxiv.org/abs/2203.02155) Advances in neural information processing systems 35 (2022): 27730-27744.

[3] Rafailov, Rafael, et al. ["Direct preference optimization: Your language model is secretly a reward model."](https://arxiv.org/abs/1706.03741) Advances in Neural Information Processing Systems 36 (2024).


## Citation

> **Citation**: When reprinting or citing the contents of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (Feb 2025). LLMs Alignment: DPO.  
> https://syhya.github.io/posts/2025-02-08-dpo

Or

```bibtex
@article{syhya2025dpo,
  title   = "LLMs Alignment: DPO",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://syhya.github.io/posts/2025-02-08-dpo"
}

