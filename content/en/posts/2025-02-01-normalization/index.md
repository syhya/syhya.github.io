---
title: Normalization in Deep Learning
date: 2025-02-01T12:00:00+08:00
author: "Yue Shui"
tags: ["AI", "NLP", "Deep Learning", "Normalization", "Residual Connection", "ResNet", "BatchNorm", "LayerNorm", "WeightNorm", "RMSNorm", "Pre-Norm", "Post-Norm", "LLM"]
categories: ["Technical Blog"]
readingTime: 30
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

## Introduction

In deep learning, the design of network architectures significantly impacts model performance and training efficiency. As model depth increases, training deep neural networks faces numerous challenges, such as the vanishing and exploding gradient problems. To address these challenges, residual connections and various normalization methods have been introduced and are widely used in modern deep learning models. This article will first introduce residual connections and two architectures: pre-norm and post-norm. Then, it will describe four common normalization methods: Batch Normalization, Layer Normalization, Weight Normalization, and RMS Normalization, and analyze why current mainstream large models tend to adopt an architecture combining **RMSNorm** and **Pre-Norm**.

## Residual Connections

**Residual Connection** is a crucial innovation in deep neural networks, forming the core of Residual Networks (ResNet) ([He, et al., 2015](https://arxiv.org/abs/1512.03385)). Residual connections are a significant architectural design aimed at mitigating the vanishing gradient problem in deep network training and facilitating information flow within the network. By introducing shortcut/skip connections, they allow information to pass directly from shallow layers to deeper layers, thereby enhancing the model's representational capacity and training stability.

{{< figure
    src="residual_connection.png"
    caption="Fig. 1. Residual learning: a building block. (Image source: [He, et al., 2015](https://arxiv.org/abs/1502.03167))"
    align="center"
    width="70%"
>}}

In a standard residual connection, the input \( x_l \) undergoes a series of transformation functions \( \text{F}(\cdot) \) and is then added to the original input \( x_l \) to form the output \( x_{l+1} \):

\[
x_{l+1} = x_l + \text{F}(x_l)
\]

Where:

*   \( x_l \) is the input to the \( l \)-th layer.
*   \( \text{F}(x_l) \) represents the residual function composed of a series of non-linear transformations (e.g., convolutional layers, fully connected layers, activation functions, etc.).
*   \( x_{l+1} \) is the output of the \( (l+1) \)-th layer.

The structure using residual connections has several advantages:

- **Mitigation of Vanishing Gradients**: By directly passing gradients through shortcut paths, it effectively reduces gradient decay in deep networks, making it easier to train deeper models.
- **Facilitation of Information Flow**: Shortcut paths allow information to flow more freely between network layers, helping the network learn more complex feature representations.
- **Optimization of the Learning Process**: Residual connections make the loss function surface smoother, optimizing the model's learning process and making it easier to converge to a better solution.
- **Improvement of Model Performance**: In various deep learning tasks such as image recognition and natural language processing, models using residual connections typically exhibit superior performance.

## Pre-Norm vs. Post-Norm

When discussing normalization methods, **Pre-Norm** and **Post-Norm** are two critical architectural design choices, particularly prominent in Transformer models. The following will detail the definitions, differences, and impacts of both on model training.

### Definitions

{{< figure
    src="pre_post_norm_comparison.png"
    caption="Fig. 2. (a) Post-LN Transformer layer; (b) Pre-LN Transformer layer. (Image source: [Xiong, et al., 2020](https://arxiv.org/abs/2002.04745))"
    align="center"
    width="50%"
>}}

From the figure above, we can intuitively see that the main difference between Post-Norm and Pre-Norm lies in the position of the normalization layer:

- **Post-Norm**: In traditional Transformer architectures, the normalization layer (such as LayerNorm) is typically placed after the residual connection.

  \[
  \text{Post-Norm}: \quad x_{l+1} = \text{Norm}(x_l + \text{F}(x_l))
  \]

- **Pre-Norm**: Places the normalization layer before the residual connection.

  \[
  \text{Pre-Norm}: \quad x_{l+1} = x_l + \text{F}(\text{Norm}(x_l))
  \]

### Comparative Analysis

| Feature               | Post-Norm                                        | Pre-Norm                                         |
|--------------------|--------------------------------------------------|--------------------------------------------------|
| **Normalization Position**     | After residual connection                             | Before residual connection                             |
| **Gradient Flow**       | May lead to vanishing or exploding gradients, especially in deep models | More stable gradients, helps in training deep models |
| **Training Stability**     | Difficult to train deep models, requires complex optimization techniques | Easier to train deep models, reduces reliance on learning rate scheduling |
| **Information Transfer**       | Retains characteristics of the original input, aiding information transfer | May cause compression or loss of input feature information |
| **Model Performance**       | Performs better in shallow models or when strong regularization is needed | Performs better in deep models, improves training stability and convergence speed |
| **Implementation Complexity**     | Relatively straightforward to implement, but training may require more tuning | Simple to implement, training process is more stable |

The differences between Pre-Norm and Post-Norm in model training can be understood from the perspective of gradient backpropagation:

- **Pre-Norm**: Normalization operation is performed first, allowing gradients to be passed more directly to the preceding layers during backpropagation, reducing the risk of vanishing gradients. However, this may also weaken the actual contribution of each layer, reducing the effective depth of the model.

- **Post-Norm**: Normalization operation is performed last, helping to maintain the stability of each layer's output, but in deep models, gradients may decay layer by layer, leading to training difficulties.

The **DeepNet** ([Wang, et al., 2022](https://arxiv.org/abs/2203.00555)) paper indicates that Pre-Norm is effective for training extremely deep Transformer models, while Post-Norm is difficult to scale to such depths.

## Normalization Methods

In deep learning, there are numerous types of normalization methods, and different methods perform differently in various application scenarios. The following will detail four common normalization methods: Batch Normalization, Layer Normalization, Weight Normalization, and RMS Normalization, and analyze their advantages, disadvantages, and applicable scenarios.

### Batch Normalization

Batch Normalization ([Ioffe, et al., 2015](https://arxiv.org/abs/1502.03167)) aims to alleviate the Internal Covariate Shift problem by standardizing the data of each batch, making its mean 0 and variance 1. Its mathematical expression is as follows:

$$
\text{BatchNorm}(x_i) = \gamma \cdot \frac{x_i - \mu_{\text{B}}}{\sqrt{\sigma_{\text{B}}^2 + \epsilon}} + \beta
$$

Where:
- \( x_i \) is the \( i \)-th sample in the input vector.
- \( \mu_{\text{B}} \) is the mean of the current batch:
  $$
  \mu_{\text{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
  $$
  where \( m \) is the batch size.
- \( \sigma_{\text{B}}^2 \) is the variance of the current batch:
  $$
  \sigma_{\text{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\text{B}})^2
  $$
- \( \epsilon \) is a very small constant used to prevent division by zero.
- \( \gamma \) and \( \beta \) are learnable scaling and shifting parameters.

**Advantages:**
- **Accelerated Training**: Accelerates the convergence speed of the model through standardization.
- **Regularization Effect**: Reduces overfitting to some extent, decreasing the reliance on regularization techniques like Dropout.
- **Mitigation of Vanishing Gradient Problem**: Helps alleviate vanishing gradients, improving the training effect of deep networks.

**Disadvantages:**
- **Not Friendly to Small Batches**: When the batch size is small, the estimation of mean and variance may be unstable, affecting the normalization effect.
- **Batch Size Dependent**: Requires a large batch size to obtain good statistical estimates, limiting its use in certain application scenarios.
- **Complex Application in Certain Network Structures**: Such as Recurrent Neural Networks (RNNs), requiring special handling to adapt to the dependency of time steps.

### Layer Normalization

Layer Normalization ([Ba, et al., 2016](https://arxiv.org/abs/1607.06450)) normalizes across the feature dimension, making the features of each sample have the same mean and variance. Its mathematical expression is as follows:

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu_{\text{L}}}{\sqrt{\sigma_{\text{L}}^2 + \epsilon}} + \beta
$$

Where:
- \( x \) is the input vector.
- \( \mu_{\text{L}} \) is the mean across the feature dimension:
  $$
  \mu_{\text{L}} = \frac{1}{d} \sum_{i=1}^{d} x_i
  $$
  where \( d \) is the size of the feature dimension.
- \( \sigma_{\text{L}}^2 \) is the variance across the feature dimension:
  $$
  \sigma_{\text{L}}^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu_{\text{L}})^2
  $$
- \( \epsilon \) is a very small constant used to prevent division by zero.
- \( \gamma \) and \( \beta \) are learnable scaling and shifting parameters.

**Advantages:**
- **Batch Size Independent**: Suitable for scenarios with small batch sizes or dynamic batch sizes, especially performing excellently in sequence models.
- **Applicable to Various Network Structures**: Performs well in Recurrent Neural Networks (RNNs) and Transformer models.
- **Simplified Implementation**: No need to rely on batch statistics, simplifying implementation in distributed training.

**Disadvantages:**
- **Higher Computational Cost**: Compared to BatchNorm, the overhead of calculating mean and variance is slightly higher.
- **May Not Improve Training Speed as Much as BatchNorm**: In some cases, the effect of LayerNorm may not be as significant as BatchNorm.

### Weight Normalization

Weight Normalization ([Salimans, et al., 2016](https://arxiv.org/abs/1602.07868)) decouples the norm and direction of the weight vector in neural networks by reparameterizing it, thereby simplifying the optimization process and accelerating training to some extent. Its mathematical expression is as follows:

$$
w = \frac{g}{\lVert v \rVert} \cdot v
$$

$$
\text{WeightNorm}(x) = w^T x + b
$$

Where:
- \( w \) is the reparameterized weight vector.
- \( g \) is a learnable scalar scaling parameter.
- \( v \) is a learnable direction vector (with the same dimension as the original \( w \)).
- \( \lVert v \rVert \) represents the Euclidean norm of \( v \).
- \( x \) is the input vector.
- \( b \) is the bias term.

**Advantages:**
- **Simplified Optimization Objective**: Separately controlling the norm and direction of weights helps accelerate convergence.
- **Stable Training Process**: In some cases, it can reduce gradient explosion or vanishing problems.
- **Implementation Independent of Batch Size**: Unrelated to the batch size of input data, broader applicability.

**Disadvantages:**
- **Implementation Complexity**: Requires reparameterization of network layers, which may bring additional implementation costs.
- **Caution Needed When Combined with Other Normalization Methods**: When used in conjunction with BatchNorm, LayerNorm, etc., debugging and experimentation are needed to determine the best combination.

### RMS Normalization

RMS Normalization ([Zhang, et al., 2019](https://arxiv.org/abs/1910.07467)) is a simplified normalization method that normalizes by only calculating the Root Mean Square (RMS) of the input vector, thereby reducing computational overhead. Its mathematical expression is as follows:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

Where:
- \( x \) is the input vector.
- \( d \) is the size of the feature dimension.
- \( \epsilon \) is a very small constant used to prevent division by zero.
- \( \gamma \) is a learnable scaling parameter.

**Advantages:**
- **High Computational Efficiency**: Compared to LayerNorm, which requires calculating both mean and variance, RMSNorm only needs to calculate the root mean square, reducing computational overhead.
- **Training Stability**: By normalizing the input, it improves the training stability of the model, allowing it to train stably even with larger learning rates.
- **Resource Optimization**: Reduced computational overhead helps deploy models in resource-constrained environments, improving training and inference efficiency.
- **Simplified Implementation**: RMSNorm is relatively simple to implement, making it easy to integrate and optimize in complex models, reducing the complexity of engineering implementation.

**Disadvantages:**
- **Information Loss**: Using only the root mean square for normalization may lose some information, such as mean information.
- **Limited Applicability**: In some tasks, it may not perform as well as BatchNorm or LayerNorm.

### Code Example
You can refer to [normalization.py]("https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-02-01-normalization/normalization.py")

### Comparison of Normalization Methods

The following two tables compare the main characteristics of BatchNorm, LayerNorm, WeightNorm, and RMSNorm.

#### BatchNorm vs. LayerNorm

| Feature                 | BatchNorm (BN)                                              | LayerNorm (LN)                                         |
|----------------------|-------------------------------------------------------------|--------------------------------------------------------|
| **Calculated Statistics**     | Batch mean and variance                                           | Per-sample mean and variance                                  |
| **Operation Dimension**         | Normalizes across all samples in a batch                             | Normalizes across all features for each sample                        |
| **Applicable Scenarios**         | Suitable for large batch data, Convolutional Neural Networks (CNNs)                       | Suitable for small batch or sequential data, RNNs or Transformers            |
| **Batch Size Dependency** | Strongly dependent on batch size                                             | Independent of batch size, suitable for small batch or single-sample tasks              |
| **Learnable Parameters**     | Scaling parameter \( \gamma \) and shifting parameter \( \beta \)               | Scaling parameter \( \gamma \) and shifting parameter \( \beta \)          |
| **Formula**             | \( \text{BatchNorm}(x_i) = \gamma \cdot \frac{x_i - \mu_{\text{B}}}{\sqrt{\sigma_{\text{B}}^2 + \epsilon}} + \beta \) | \( \text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu_{\text{L}}}{\sqrt{\sigma_{\text{L}}^2 + \epsilon}} + \beta \) |
| **Computational Complexity**       | Requires calculating batch mean and variance                                   | Requires calculating per-sample mean and variance                          |
| **Example Usage**         | CNN, Vision Transformers                                   | RNN, Transformer, NLP                                |

#### WeightNorm vs. RMSNorm

| Feature                 | WeightNorm (WN)                                           | RMSNorm (RMS)                                      |
|----------------------|-----------------------------------------------------------|----------------------------------------------------|
| **Calculated Statistics**     | Decomposes weight vector into norm and direction                                  | Root Mean Square (RMS) of each sample                            |
| **Operation Dimension**         | Reparameterizes along the dimension of the weight vector                         | Normalizes across all features for each sample                    |
| **Applicable Scenarios**         | Suitable for scenarios requiring more flexible weight control or accelerated convergence               | Suitable for tasks requiring efficient computation, such as RNNs or Transformers   |
| **Batch Size Dependency** | Independent of batch size, unrelated to the dimension of input data                     | Independent of batch size, suitable for small batch or single-sample tasks          |
| **Learnable Parameters**     | Scalar scaling \( g \) and direction vector \( v \)                      | Scaling parameter \( \gamma \)                             |
| **Formula**             | \( \text{WeightNorm}(x) = w^T x + b \) | \( \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma \) |
| **Computational Complexity**       | Reparameterization and update of parameters, slightly higher overhead, but requires modifying network layer implementation      | Only needs to calculate the root mean square of each sample, computationally efficient            |
| **Example Usage**         | Fully connected layers, convolutional layers, etc., in deep networks, improving training stability and convergence speed    | Transformer, NLP, efficient sequence tasks                    |

Through the above comparison, it can be seen that the four normalization methods have their own advantages and disadvantages:

- **BatchNorm** performs excellently in large batch data and convolutional neural networks but is sensitive to small batches.
- **LayerNorm** is suitable for various batch sizes, especially effective in RNNs and Transformers.
- **WeightNorm** simplifies the optimization process and accelerates convergence to some extent by reparameterizing the weight vector.
- **RMSNorm** provides a lightweight alternative in scenarios requiring efficient computation.

## Why do current mainstream LLMs use Pre-Norm and RMSNorm?

In recent years, with the rise of large-scale language models (LLMs) such as GPT, LLaMA, and the Qwen series, **RMSNorm** and **Pre-Norm** have become the standard choices for these models.

### Advantages of RMSNorm

{{< figure
    src="rms_norm_time_benchmark.png"
    caption="Fig. 3. RMSNorm vs. LayerNorm: A Comparison of Time Consumption (Image source: [Zhang, et al., 2019](https://arxiv.org/abs/1910.07467))"
    align="center"
    width="80%"
>}}

1. **Higher Computational Efficiency**
   - **Reduced Operations**: Only needs to calculate the Root Mean Square (RMS) of the input vector, without calculating mean and variance.
   - **Faster Training Speed**: In actual tests, RMSNorm significantly shortens training time (as shown in the figure, reduced from **665s** to **501s**), which is particularly evident in large-scale model training.

2. **More Stable Training**
   - **Adaptable to Larger Learning Rates**: While maintaining stability, it can use larger learning rates, accelerating model convergence.
   - **Maintains Expressive Power**: By simplifying the normalization process with an appropriate scaling parameter \( \gamma \), it still maintains model performance.

3. **Resource Saving**
   - **Reduced Hardware Requirements**: Less computational overhead not only improves speed but also reduces the occupation of hardware resources, suitable for deployment in resource-constrained environments.

### Advantages of Pre-Norm

1. **Easier to Train Deep Models**
   - **Stable Gradient Propagation**: Performing normalization before residual connections can effectively alleviate gradient vanishing or explosion.
   - **Reduced Reliance on Complex Optimization Techniques**: Even if the model is very deep, the training process remains stable.

2. **Accelerated Model Convergence**
   - **Efficient Gradient Flow**: Pre-Norm makes it easier for gradients to propagate to preceding layers, resulting in faster overall convergence speed.

## Conclusion

Residual connections and normalization methods play crucial roles in deep learning models. Different normalization methods and network architecture designs have their own applicable scenarios, advantages, and disadvantages. By introducing residual connections, ResNet successfully trained extremely deep networks, significantly improving model expressiveness and training efficiency. Meanwhile, normalization methods such as BatchNorm, LayerNorm, WeightNorm, and RMSNorm each offer different advantages, adapting to different application needs.

As model scales continue to expand, choosing appropriate normalization methods and network architecture designs becomes particularly important. **RMSNorm**, due to its efficient computation and good training stability, combined with the **Pre-Norm** architecture design, has become the preferred choice for current mainstream LLMs. This combination not only improves the training efficiency of models but also ensures training stability and performance under large-scale parameters.

## References

[1] He, Kaiming, et al. ["Deep residual learning for image recognition."](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Xiong, Ruibin, et al. ["On layer normalization in the transformer architecture."](https://arxiv.org/abs/2002.04745) International Conference on Machine Learning. PMLR, 2020.

[3] Wang, Hongyu, et al. ["Deepnet: Scaling transformers to 1,000 layers."](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10496231) IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

[4] Ioffe, Sergey. ["Batch normalization: Accelerating deep network training by reducing internal covariate shift."](https://arxiv.org/abs/1502.03167) arXiv preprint arXiv:1502.03167 (2015).

[5] Ba, Jimmy Lei. ["Layer normalization."](https://arxiv.org/abs/1607.06450) arXiv preprint arXiv:1607.06450 (2016).

[6] Salimans, Tim, and Durk P. Kingma. ["Weight normalization: A simple reparameterization to accelerate training of deep neural networks."](https://proceedings.neurips.cc/paper_files/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf) Advances in neural information processing systems 29 (2016).

[7] Zhang, Biao, and Rico Sennrich. ["Root mean square layer normalization."](https://arxiv.org/abs/1910.07467) Advances in Neural Information Processing Systems 32 (2019).

## Citation

> **Citation**: When reprinting or citing the content of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (Feb 2025). Normalization in Deep Learning.
https://syhya.github.io/posts/2025-02-01-normalization

Or

```bibtex
@article{syhya2025normalization,
  title   = "Normalization in Deep Learning",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://syhya.github.io/posts/2025-02-01-normalization"
}
