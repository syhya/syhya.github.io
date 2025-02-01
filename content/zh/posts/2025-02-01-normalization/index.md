---
title: 深度学习中的归一化
date: 2025-02-01T12:00:00+08:00
author: "Yue Shui"
tags: ["AI", "NLP", "Deep Learning", "Normalization", "Residual Connection", "ResNet", "Batch Normalization", "Layer Normalization", "Weight Normalization", "RMS Normalization", "Pre-Norm", "Post-Norm", "LLM"]
categories: ["技术博客"]
readingTime: 20
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

## 引言

在深度学习中，网络架构的设计对模型的性能和训练效率有着至关重要的影响。随着模型深度的增加，训练深层神经网络面临诸多挑战，如梯度消失和梯度爆炸问题。为了应对这些挑战，残差连接和各种归一化方法被引入并广泛应用于现代深度学习模型中。本文将首先介绍残差连接和两种架构，分别是pre-norm 和 post-norm。随后介绍四种常见的方法：Batch Normalization、Layer Normalization、Weight Normalization和 RMS Normalization，并分析为何当前主流的大模型倾向于采用 **RMSNorm** 与 **Pre-Norm** 结合的架构。


## 残差连接

**残差连接（Residual Connection）** 是深度神经网络中的一项关键创新，它构成了残差网络（ResNet）([He, et al., 2015](https://arxiv.org/abs/1512.03385)) 的核心。残差连接是一种重要的架构设计，目的是缓解深层网络训练中的梯度消失问题，并促进信息在网络中的流动。它通过引入快捷路径（Shortcut/Skip Connection），允许信息直接从浅层传递到深层，从而增强模型的表达能力和训练稳定性。

{{< figure 
    src="residual_connection.png" 
    caption="Fig. 1. Residual learning: a building block. (Image source: [He, et al., 2015](https://arxiv.org/abs/1502.03167))"
    align="center" 
    width="70%"
>}}

在标准的残差连接中，输入 \( x_l \) 经过一系列变换函数 \( \text{F}(\cdot) \) 后，与原始输入 \( x_l \) 相加，形成输出 \( x_{l+1} \)：

\[
x_{l+1} = x_l + \text{F}(x_l)
\]

其中：

*   \( x_l \) 是第 \( l \) 层的输入。
*   \( \text{F}(x_l) \) 表示由一系列非线性变换（例如卷积层、全连接层、激活函数等）组成的残差函数。
*   \( x_{l+1} \) 是第 \( l+1 \) 层的输出。

使用残差连接的结构有以下几个优势：
- **缓解梯度消失**： 通过快捷路径直接传递梯度，有效减少梯度在深层网络中的衰减，从而更容易训练更深的模型。
- **促进信息流动**： 快捷路径允许信息更自由地在网络层之间流动，有助于网络学习更复杂的特征表示。
- **优化学习过程**： 残差连接使得损失函数曲面更加平滑，优化模型的学习过程，使其更容易收敛到较好的解。
- **提升模型性能**： 在图像识别、自然语言处理等多种深度学习任务中，使用残差连接的模型通常表现出更优越的性能。


## Pre-Norm 与 Post-Norm

在讨论归一化方法时，**Pre-Norm** 和 **Post-Norm** 是两个关键的架构设计选择，尤其在 Transformer 模型中表现突出。以下将详细探讨两者的定义、区别及其对模型训练的影响。

### 定义
{{< figure 
    src="pre_post_norm_comparison.png" 
    caption="Fig. 2. (a) Post-LN Transformer layer; (b) Pre-LN Transformer layer. (Image source: [Xiong, et al., 2020](https://arxiv.org/abs/2002.04745))"
    align="center" 
    width="50%"
>}}

从上图我可以直观看到，Post-Norm 和 Pre-Norm 的主要区别在于归一化层的位置：

- **Post-Norm**：传统的 Transformer 架构中，归一化层（如 LayerNorm）通常位于残差连接之后。

  \[
  \text{Post-Norm}: \quad x_{l+1} = \text{Norm}(x_l + \text{F}(x_l))
  \]

- **Pre-Norm**：将归一化层放在残差连接之前。

  \[
  \text{Pre-Norm}: \quad x_{l+1} = x_l + \text{F}(\text{Norm}(x_l))
  \]

### 对比分析

| 特性               | Post-Norm                                        | Pre-Norm                                         |
|--------------------|--------------------------------------------------|--------------------------------------------------|
| **归一化位置**     | 残差连接之后                                    | 残差连接之前                                    |
| **梯度流动**       | 可能导致梯度消失或爆炸，尤其在深层模型中        | 梯度更稳定，有助于训练深层模型                    |
| **训练稳定性**     | 难以训练深层模型，需要复杂的优化技巧            | 更容易训练深层模型，减少对学习率调度的依赖        |
| **信息传递**       | 保留了原始输入的特性，有助于信息传递            | 可能导致输入特征的信息被压缩或丢失                |
| **模型性能**       | 在浅层模型或需要强正则化效果时表现更优          | 在深层模型中表现更好，提升训练稳定性和收敛速度    |
| **实现复杂度**     | 实现较为直接，但训练过程可能需要更多调优        | 实现简单，训练过程更稳定                          |


Pre-Norm 和 Post-Norm 在模型训练中的差异可以从梯度反向传播的角度理解：

- **Pre-Norm**：归一化操作在前，梯度在反向传播时能够更直接地传递到前面的层，减少了梯度消失的风险。但这也可能导致每一层的实际贡献被弱化，降低模型的实际有效深度。

- **Post-Norm**：归一化操作在后，有助于保持每一层的输出稳定，但在深层模型中，梯度可能会逐层衰减，导致训练困难。

**DeepNet** ([Wang, et al., 2022](https://arxiv.org/abs/2203.00555)) 论文表明 Pre-Norm 在极深的 Transformer 模型中能够有效训练，而 Post-Norm 难以扩展到如此深度。

## 归一化方法

在深度学习中，归一化方法种类繁多，不同的方法在不同的应用场景下表现各异。下面将详细介绍四种常见的归一化方法：Batch Normalization、Layer Normalization、Weight Normalization 和 RMS Normalization，并分析它们的优劣势及适用场景。

### Batch Normalization

Batch Normalization ([Ioffe, et al., 2015](https://arxiv.org/abs/1502.03167)) 旨在通过标准化每一批次的数据，使其均值为0，方差为1，从而缓解内部协变量偏移（Internal Covariate Shift）的问题。其数学表达式如下：

$$
\text{BatchNorm}(x_i) = \gamma \cdot \frac{x_i - \mu_{\text{B}}}{\sqrt{\sigma_{\text{B}}^2 + \epsilon}} + \beta
$$

其中：
- \( x_i \) 为输入向量中的第 \( i \) 个样本。
- \( \mu_{\text{B}} \) 为当前批次的均值：
  $$
  \mu_{\text{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
  $$
  其中 \( m \) 为批次大小。
- \( \sigma_{\text{B}}^2 \) 为当前批次的方差：
  $$
  \sigma_{\text{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\text{B}})^2
  $$
- \( \epsilon \) 为一个极小的常数，用于防止分母为零。
- \( \gamma \) 和 \( \beta \) 为可学习的缩放和平移参数。

**优势：**
- **加速训练**：通过标准化加速模型的收敛速度。
- **正则化效果**：在一定程度上减少过拟合，降低了对 Dropout 等正则化技术的依赖。
- **减轻梯度消失问题**：有助于缓解梯度消失，提高深层网络的训练效果。

**缺点：**
- **对小批次不友好**：在批次大小较小时，均值和方差的估计可能不稳定，影响归一化效果。
- **依赖批次大小**：需要较大的批次才能获得良好的统计量估计，限制了在某些应用场景中的使用。
- **在某些网络结构中应用复杂**：如循环神经网络（RNN），需要特殊处理以适应时间步的依赖性。


### Layer Normalization

Layer Normalization ([Ba, et al., 2016](https://arxiv.org/abs/1607.06450)) 通过在特征维度上进行归一化，使得每个样本的特征具有相同的均值和方差。其数学表达式如下：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu_{\text{L}}}{\sqrt{\sigma_{\text{L}}^2 + \epsilon}} + \beta
$$

其中：
- \( x \) 为输入向量。
- \( \mu_{\text{L}} \) 为特征维度的均值：
  $$
  \mu_{\text{L}} = \frac{1}{d} \sum_{i=1}^{d} x_i
  $$
  其中 \( d \) 为特征维度的大小。
- \( \sigma_{\text{L}}^2 \) 为特征维度的方差：
  $$
  \sigma_{\text{L}}^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu_{\text{L}})^2
  $$
- \( \epsilon \) 为一个极小的常数，用于防止分母为零。
- \( \gamma \) 和 \( \beta \) 为可学习的缩放和平移参数。

**优势：**
- **对批次大小不敏感**：适用于小批次或动态批次大小的场景，尤其在序列模型中表现优异。
- **适用于多种网络结构**：在循环神经网络（RNN）和 Transformer 等模型中表现良好。
- **简化实现**：无需依赖批次统计量，简化了在分布式训练中的实现。

**缺点：**
- **计算量较大**：相比 BatchNorm，计算均值和方差的开销稍高。
- **可能不如 BatchNorm 提升训练速度**：在某些情况下，LayerNorm 的效果可能不如 BatchNorm 显著。

### Weight Normalization

Weight Normalization ([Salimans, et al., 2016](https://arxiv.org/abs/1602.07868)) 通过重新参数化神经网络中的权重向量来解耦其模长（norm）和方向（direction），从而简化优化过程并在一定程度上加速训练。其数学表达式如下：

$$
w = \frac{g}{\lVert v \rVert} \cdot v
$$

$$
\text{WeightNorm}(x) = w^T x + b
$$

其中：
- \( w \) 是重新参数化后的权重向量。
- \( g \) 为可学习的标量缩放参数。
- \( v \) 为可学习的方向向量（与原始 \( w \) 维度相同）。
- \( \lVert v \rVert \) 表示 \( v \) 的欧几里得范数。
- \( x \) 为输入向量。
- \( b \) 为偏置项。

**优势：**
- **简化优化目标**：单独控制权重的模长与方向，有助于加速收敛。
- **稳定训练过程**：在某些情况下，可减少梯度爆炸或消失问题。
- **实现不依赖批次大小**：与输入数据的批次无关，适用性更广。

**缺点：**
- **实现复杂度**：需要对网络层进行重新参数化，可能带来额外的实现成本。
- **与其他归一化方法结合时需谨慎**：如与 BatchNorm、LayerNorm 等同用时，需要调试和实验来确定最佳组合。


### RMS Normalization

RMS Normalization ([Zhang, et al., 2019](https://arxiv.org/abs/1910.07467)) 是一种简化的归一化方法，通过仅计算输入向量的均方根（RMS）进行归一化，从而减少计算开销。其数学表达式如下：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

其中：
- \( x \) 为输入向量。
- \( d \) 为特征维度的大小。
- \( \epsilon \) 为一个极小的常数，用于防止分母为零。
- \( \gamma \) 为可学习的缩放参数。

**优势：**
- **计算效率高**：相比 LayerNorm 需要计算均值和方差，RMSNorm 仅需计算均方根，减少了计算开销。
- **训练稳定性**：通过归一化输入，提升了模型的训练稳定性，使其在更大的学习率下仍能稳定训练。
- **资源优化**：减少计算开销有助于在资源受限的环境中部署模型，提高训练和推理的效率。
- **简化实现**：RMSNorm 的实现相对简单，便于在复杂模型中集成和优化，减少了工程实现的复杂性。

**缺点：**
- **信息损失**：仅使用均方根进行归一化，可能丢失部分信息，如均值信息。
- **适用性有限**：在某些任务中，可能不如 BatchNorm 或 LayerNorm 表现优异。

### 代码示例
可以参考[normalization.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-02-01-normalization/normalization.py)

### 归一化方法对比

以下两个表格对比了 BatchNorm、LayerNorm、WeightNorm 和 RMSNorm 四种归一化方法的主要特性：

#### BatchNorm vs. LayerNorm

| 特性                 | BatchNorm (BN)                                              | LayerNorm (LN)                                         |
|----------------------|-------------------------------------------------------------|--------------------------------------------------------|
| **计算的统计量**     | 批量的均值和方差                                           | 每个样本的均值和方差                                  |
| **操作维度**         | 对批量数据的所有样本进行归一化                             | 对每个样本的所有特征进行归一化                        |
| **适用场景**         | 适用于大批量数据，卷积神经网络 (CNN)                       | 适用于小批量或序列数据，RNN 或 Transformer            |
| **是否依赖批量大小** | 强依赖批量大小                                             | 不依赖批量大小，适用于小批量或单样本任务              |
| **可学习的参数**     | 缩放参数 \( \gamma \) 和平移参数 \( \beta \)               | 缩放参数 \( \gamma \) 和平移参数 \( \beta \)          |
| **公式**             | \( \text{BatchNorm}(x_i) = \gamma \cdot \frac{x_i - \mu_{\text{B}}}{\sqrt{\sigma_{\text{B}}^2 + \epsilon}} + \beta \) | \( \text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu_{\text{L}}}{\sqrt{\sigma_{\text{L}}^2 + \epsilon}} + \beta \) |
| **计算复杂度**       | 需要计算批量的均值和方差                                   | 需要计算每个样本的均值和方差                          |
| **使用示例**         | CNN, Vision Transformers                                   | RNN, Transformer, NLP                                |

#### WeightNorm vs. RMSNorm

| 特性                 | WeightNorm (WN)                                           | RMSNorm (RMS)                                      |
|----------------------|-----------------------------------------------------------|----------------------------------------------------|
| **计算的统计量**     | 分解权重向量的模长和方向                                  | 每个样本的均方根 (RMS)                            |
| **操作维度**         | 针对权重向量的维度进行重新参数化                         | 对每个样本的所有特征进行归一化                    |
| **适用场景**         | 适用于需要更灵活的权重控制或加速收敛的场景               | 适用于需要高效计算的任务，如 RNN 或 Transformer   |
| **是否依赖批量大小** | 不依赖批量大小，与输入数据的维度无关                     | 不依赖批量大小，适用于小批量或单样本任务          |
| **可学习的参数**     | 标量缩放 \( g \) 和方向向量 \( v \)                      | 缩放参数 \( \gamma \)                             |
| **公式**             | \( \text{WeightNorm}(x) = w^T x + b \) | \( \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma \) |
| **计算复杂度**       | 参数重新分解和更新，额外开销较小，但需修改网络层实现      | 只需计算每个样本的均方根，计算较为高效            |
| **使用示例**         | 深度网络的全连接层、卷积层等，提升训练稳定性和收敛速度    | Transformer, NLP, 高效序列任务                    |

通过上述对比，可以看出四种归一化方法各有优劣：

- **BatchNorm** 在大批量数据和卷积神经网络中表现优异，但对小批量敏感。
- **LayerNorm** 适用于各种批量大小，尤其是在 RNN 和 Transformer 中效果显著。
- **WeightNorm** 通过重新参数化权重向量，在一定程度上简化了优化过程并加速收敛。
- **RMSNorm** 则在需要高效计算的场景下提供了一种轻量级的替代方案。

## 为什么当前主流 LLM 都使用 Pre-Norm 和 RMSNorm？

近年来，随着大规模语言模型（LLM）如 GPT、LLaMA 和 Qwen 系列等的兴起，**RMSNorm** 和 **Pre-Norm** 已成为这些模型的标准选择。

### RMSNorm 的优势

{{< figure 
    src="rms_norm_time_benchmark.png" 
    caption="Fig. 3. RMSNorm vs. LayerNorm: A Comparison of Time Consumption (Image source: [Zhang, et al., 2019](https://arxiv.org/abs/1910.07467))"
    align="center" 
    width="80%"
>}}

1. **计算效率更高**  
   - **减少运算量**：只需计算输入向量的均方根（RMS），无需计算均值和方差。  
   - **加快训练速度**：实际测试中，RMSNorm 显著缩短了训练时间（如上图由 **665s** 降至 **501s**），在大规模模型训练中尤其明显。

2. **训练更稳定**  
   - **适应更大学习率**：在保持稳定性的同时，能够使用更大学习率，加速模型收敛。  
   - **保持表达能力**：通过适当的缩放参数 \(\gamma\) 简化归一化过程的同时，仍能维持模型的表现。

3. **节省资源**  
   - **降低硬件需求**：更少的计算开销既能提升速度，也能减少对硬件资源的占用，适合在资源受限环境中部署。


### Pre-Norm 的优势

1. **更易训练深层模型**  
   - **稳定梯度传播**：在残差连接之前进行归一化，可有效缓解梯度消失或爆炸。  
   - **减少对复杂优化技巧的依赖**：即使模型很深，训练过程依然稳定。

2. **加速模型收敛**  
   - **高效的梯度流动**：Pre-Norm 使梯度更容易传递到前面的层，整体收敛速度更快。

## 结论

残差连接和归一化方法在深度学习模型中扮演着至关重要的角色，不同的归一化方法和网络架构设计各有其适用场景和优缺点。通过引入残差连接，ResNet 成功地训练了极深的网络，显著提升了模型的表达能力和训练效率。同时，归一化方法如 BatchNorm、LayerNorm、WeightNorm 和 RMSNorm 各自提供了不同的优势，适应了不同的应用需求。

随着模型规模的不断扩大，选择合适的归一化方法和网络架构设计变得尤为重要。**RMSNorm** 由于其高效的计算和良好的训练稳定性，结合 **Pre-Norm** 的架构设计，成为当前主流 LLM 的首选。这种组合不仅提升了模型的训练效率，还确保了在大规模参数下的训练稳定性和性能表现。

## 参考文献

[1] He, Kaiming, et al. ["Deep residual learning for image recognition."](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Xiong, Ruibin, et al. ["On layer normalization in the transformer architecture."](https://arxiv.org/abs/2002.04745) International Conference on Machine Learning. PMLR, 2020.

[3] Wang, Hongyu, et al. ["Deepnet: Scaling transformers to 1,000 layers."](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10496231) IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

[4] Ioffe, Sergey. ["Batch normalization: Accelerating deep network training by reducing internal covariate shift."](https://arxiv.org/abs/1502.03167) arXiv preprint arXiv:1502.03167 (2015).

[5] Ba, Jimmy Lei. ["Layer normalization."](https://arxiv.org/abs/1607.06450) arXiv preprint arXiv:1607.06450 (2016).

[6] Salimans, Tim, and Durk P. Kingma. ["Weight normalization: A simple reparameterization to accelerate training of deep neural networks."](https://proceedings.neurips.cc/paper_files/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf) Advances in neural information processing systems 29 (2016).

[7] Zhang, Biao, and Rico Sennrich. ["Root mean square layer normalization."](https://arxiv.org/abs/1910.07467) Advances in Neural Information Processing Systems 32 (2019).

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Feb 2025). 深度学习中的归一化.
https://syhya.github.io/posts/2025-02-01-normalization

Or

```bibtex
@article{syhya2025normalization,
  title   = "深度学习中的归一化",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://syhya.github.io/posts/2025-02-01-normalization"
}
