---
title: Llama系列（长期更新中）
date: 2025-01-22T12:00:00+08:00
author: "Yue Shui"
tags: ["AI", "NLP", "LLM", "Pre-training", "Post-training", "Llama"]
categories: ["技术博客"]
readingTime: 25
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

> **注意**: 本文**正在更新中**，内容只是**草稿版本**，并不完善，后续会有变动。请随时关注最新版本。

## 引言

本篇文章将系统梳理LLaMA系列模型从LLaMA1到LLaMA3的发展历程，深入解析其模型架构、训练数据和训练方法，并通过对比表格揭示各版本的核心差异。

## LLaMA模型演进

大语言模型（LLM）在近年来取得了重大进展，Meta 在 2023 年相继推出了多代 LLaMA 模型系列，每一代在模型规模、训练数据以及应用场景上都有新的提升。本节将依次介绍 LLaMA1、LLaMA2、Code Llama、Llama Guard 和 LLaMA3 模型的主要特性与技术创新。

---

### LLaMA1

#### 概述
**LLaMA1**（[Touvron et al., 2023](https://arxiv.org/abs/2302.13971)）于 2023 年 2 月发布，作为当时性能优异的开源模型，在学术界和工业界迅速引起广泛关注。

#### 特点
- **参数规模**：提供 7B、13B、30B 和 65B 四个版本。  
- **训练数据量**：超过 1.4 万亿个 token。  
- **训练资源**：以 65B 模型为例，在 2048 张 A100 80GB GPU 上训练约 21 天。  
- **性能优势**：在大多数基准测试中，65B 模型超越了当时流行的 175B 参数 GPT-3。

#### 技术细节
- **Pre-normalization & RMSNorm**：采用预归一化（Pre-normalization）方案，使用 RMSNorm 替换传统的 LayerNorm，训练更稳定且速度更快。  
- **FFN_SWiGLU**：在前馈网络中引入 SWiGLU 结构，激活函数由 ReLU 更换为 SiLU，并对隐藏单元数量进行优化配置。  
- **Rotary Embeddings (RoPE)**：在每一层动态注入旋转位置嵌入信息，有利于长序列的建模能力。

#### 应用场景
- **研究与开发**：作为通用大模型的基础，用于各种 NLP 任务研究。  
- **商业应用**：可在客户服务、内容生成等场景中提升自动化水平。  
- **教育与培训**：为学术机构和教育平台提供新的教学和实验工具。

---

### LLaMA2

#### 概述
**LLaMA2**（[Touvron et al., 2023](https://arxiv.org/abs/2307.09288)）是 LLaMA 系列的第二代版本，于 2023 年中发布，相较于第一代在规模和性能上均有大幅提升。

#### 特点
- **参数规模**：覆盖 7B、13B、34B、70B 四个不同规模。  
- **训练数据量**：从 1.4 万亿 token 扩展到 2 万亿 token，增幅约 40%。  
- **上下文长度**：支持 4096 个 token，是 LLaMA1 的两倍。  
- **注意力机制**：引入分组查询注意力（Group Query Attention, GQA）以提高推理效率和内存利用率。

#### 技术细节
- **GQA（Group Query Attention）**：对查询进行分组，可减少自注意力计算的开销，降低显存占用。  
- **KV Cache**：推理阶段采用 KV 缓存，提升解码速度，缩短推理时延。

#### 应用场景
- **对话系统**：更自然、准确地生成对话回复，改善用户交互体验。  
- **内容生成**：可应用于新闻、营销文案等高质量文本的自动化生成。  

---

### Code Llama

{{< figure 
    src="codellama.png" 
    caption="Fig. 1. The Code Llama specialization pipeline. (Image source: [[Rozière et al., 2023](https://arxiv.org/abs/2308.12950))"
    align="center" 
    width="100%"
>}}

#### 概述
**Code Llama**（[Rozière et al., 2023](https://arxiv.org/abs/2308.12950)）基于 LLaMA2 进行额外训练与微调，专门面向代码生成、补全以及指令跟随任务，涵盖多种编程语言。

#### 特点
1. **多种参数规模**：提供 7B、13B、34B、70B 四类版本，可根据算力和应用需求选择。  
2. **训练规模**：  
   - 7B、13B、34B 三个型号在约 5000 亿（500B）标记的代码数据基础上训练；  
   - 70B 型号在约 1 万亿（1T）标记的同源数据上训练。  
3. **支持长上下文**：通过“长上下文微调”（LCFT）过程，能稳定处理多达 16k 乃至 100k tokens 的大型代码文件。

#### 技术细节
- **架构继承**：延续 LLaMA2 的 Transformer 架构，并针对代码领域做专门的目标任务优化。  
- **数据构成**：主要来源于公开可用的开源代码库，辅以少量与代码相关的自然语言内容，以保持通用理解能力。  
- **填充中间（Fill-in-the-Middle, FIM）**：7B、13B、70B 参数的基础模型直接支持在已有代码任意位置插入补全，方便 IDE 中的实时代码片段生成。  
- **长上下文微调（LCFT）**：在已训练的模型上使用更长序列（16k tokens）再次微调，重置 RoPE 参数，使模型在处理超过训练长度的输入时更稳定。  
- **指令微调（Instruct Fine Tuning）**：结合 Llama 2 安全指令数据与自监督生成的单元测试筛选数据，以提升模型对自然语言指令的理解和合规性。

#### 应用场景
1. **开发者工具**：集成在 IDE 中用于智能补全、调试建议与文档注释，提升开发效率。  
2. **教育/培训**：为初学者或教学平台提供示例代码、解题思路和习题解析。  
3. **商业化软件**：与版本控制、CI/CD 等平台集成，为企业级开发提供自动化支持。  
4. **研究探索**：在自动化测试、代码生成等领域带来新的算法与应用思路。


### Llama Guard


{{< figure 
    src="llama_guard.png" 
    caption="Fig. 2. Example task instructions for the Llama Guard prompt and response classification tasks. (Image source: [Inan et al., 2023](https://arxiv.org/abs/2312.06674))"
    align="center" 
    width="100%"
>}}

{{< figure 
    src="llama_guard_vision.png" 
    caption="Fig. 3. Llama Guard 3 Vision classifies harmful content in the response classification task. (Image source: [Chi et al., 2024](https://arxiv.org/abs/2411.10414))"
    align="center" 
    width="100%"
>}}

#### 概述
**Llama Guard**（[Inan et al., 2023](https://arxiv.org/abs/2312.06674)）是 Meta 为 LLaMA2 及后续版本（如 LLaMA3）开发的安全增强模块，主要面向内容安全的评估与过滤，确保模型输出符合相关安全标准。


#### 特点
- **版本**：  
  1. **Llama Guard 3 1B**：面向基础文本内容的安全评估。  
  2. **Llama Guard 3 8B**：可处理更复杂的文本安全场景，专注于代码解释器滥用（S14）检测。  
  3. **Llama Guard 3 Vision**（[Chi et al., 2024](https://arxiv.org/abs/2411.10414)）：增强多模态处理能力，支持图像和文本的综合安全评估。

#### 技术细节
- **多模态评估**：通过特殊 `<|image|>` token 将图像信息与文本输入相结合，进行统一的安全审查。  
- **安全类别**：基于 ML Commons consortium 定义的 13 个安全类别（S1-S13），在 3.2 版本中新增针对“代码解释器滥用（S14）”的安全检测。  
- **评估流程**：用户将安全类别和对话内容作为输入提示，模型给出判定结果（安全或不安全）及违规类别。

#### 应用场景
- **内容审核**：自动化检测并过滤违反平台或法律规定的内容。  
- **安全监控**：在生产环境中实时监控信息流，防范有害或敏感内容传播。  
- **多模态审核**：对含图文混合的输入执行更加全面的安全审查。


### LLaMA3

{{< figure 
    src="llama3_architecture.png" 
    caption="Fig. 4. The architecute of Llama 2 and 3. (Image source: [Umar Jamil](https://github.com/hkproj/pytorch-llama/blob/main/Slides.pdf))"
    align="center" 
    width="80%"
>}}


#### 概述
**LLaMA3**（[Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)）是 LLaMA 系列的第三代模型，在多语言、多模态、以及边缘设备部署方面均有提升，拥有从 1B 到 405B 等多种规模。

#### 特点
- **参数规模**：1B、3B、11B、70B、90B 和 405B 六种版本，覆盖从轻量级到超大规模的多种需求。  
- **训练数据量**：累计 15 万亿 token，约为 LLaMA2 的 7.5 倍。  
- **Tokenizer 更新**：采用更高效的 `tiktoken`，词表从 32k 扩大至 128k。  
- **上下文长度**：可处理多达 128k tokens 的上下文。  
- **多语言支持**：覆盖 8 种语言，全面升级在跨语言环境下的适配能力。  
- **多模态支持**：11B 与 90B 版本提供视觉语言模型，可处理与图像结合的任务。  
- **轻量级版本**：1B 与 3B 通过剪枝和知识蒸馏技术，适合边缘与移动端部署。

#### 技术细节
- **全面采用 GQA（Grouped Query Attention）**：优化自注意力计算效率与内存使用。  
- **训练方法多样化**：结合监督微调(SFT)、拒绝采样(RS)、直接策略优化(DPO)等，以进一步提升模型推理与编码能力。  
- **多模态模型**：同时支持图像、视频和语音的多模态处理。

#### 应用场景
- **高级对话系统**：面对更广泛、更复杂的对话需求，提供自然、上下文一致的回复。  
- **跨语言场景**：为全球化应用提供多语言支持，覆盖更多人群和市场。  
- **多模态任务**：在图像理解、视觉问答等场景中发挥出色的多模态生成与推理能力。  
- **边缘计算**：1B 和 3B 版本可在算力有限的设备上运行，为 IoT 或移动端场景提供支持。

### LLaMA 系列模型特性对比

| 特性               | LLaMA1               | LLaMA2                       | LLaMA3                              |
|--------------------|----------------------|------------------------------|-------------------------------------|
| **发布时间**        | 2023年2月            | 2023年7月                    | 2024年4月                           |
| **模型规模**        | 7B、13B、30B、65B    | 7B、13B、34B、70B             | 1B、3B、11B、70B、90B、405B         |
| **训练数据量**      | 1.4 万亿+ tokens     | 2 万亿 +tokens               | 15 万亿+ tokens                    |
| **上下文长度**      | 2048 tokens          | 4096 tokens                  | 128k tokens                        |
| **Tokenizer**      | SentencePiece BPE，32k 词汇表 | SentencePiece BPE，32k 词汇表 | tiktoken BPE，128k 词汇表          |
| **位置编码**        | RoPE                 | RoPE                         | RoPE                               |
| **注意力机制和推理优化** | Multi-Head Attention (MHA) | Grouped Query Attention (GQA)+ kv cache | Grouped Query Attention (GQA) + kv cache |
| **归一化方法**      | RMSNorm              | RMSNorm                      | RMSNorm                            |
| **激活函数**        | SwiGLU               | SwiGLU                       | SwiGLU                             |
| **训练资源**        | 2048 * A100 80GB     | 3.3M GPU hours on A100-80GB  | 16K H100 80GB                      |
| **应用场景**        | 通用语言理解与生成    | 通用语言理解与生成，推理效率进一步提升 | 多模态应用（图像、语音）、轻量级部署、边缘设备适配 |


## 关键技术解析

LLaMA3作为系列最新版本，集成了LLaMA1和LLaMA2的核心技术，并在此基础上进行了多项创新和优化。以下是LLaMA3所采用的所有关键技术的详细解析，包括数学公式和相关说明。

### RMS Normalization

在深度学习中，归一化技术在加速训练、提升模型性能和稳定性方面起着至关重要的作用。RMS Normalization ([Zhang, et al., 2019](https://arxiv.org/abs/1910.07467)) 是一种简化的归一化方法，通过仅计算输入向量的均方根（RMS）进行归一化，从而减少计算开销。其数学表达式如下：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

其中：
- \( x \) 为输入向量。
- \( d \) 为特征维度的大小。
- \( \epsilon \) 为一个极小的常数，用于防止分母为零。
- \( \gamma \) 为可学习的缩放参数。


LLaMA3 中选择 **RMSNorm** 作为其归一化方法，主要基于以下考虑：

- **计算效率**：RMSNorm 相比 LayerNorm、BatchNorm 和WeightNorm 计算量更低，仅计算输入向量的均方根，适合 LLM 的高效训练。
- **训练稳定性**：RMSNorm 在保持训练稳定性的同时，能够适应更大的学习率，促进模型的快速收敛。
- **资源优化**：减少计算开销有助于在资源受限的环境中部署模型，提高训练和推理的效率。
- **简化实现**：RMSNorm 的实现相对简单，便于在复杂模型中集成和优化。

> 关于各种 Norm 的对比和代码实现，可参考博客：[Normalization in Deep Learning](https://syhya.github.io/posts/2025-02-01-normalization/)。


### FFN_SwiGLU

Swish-Gated Linear Unit ([Shazeer, 2020](https://arxiv.org/abs/2002.05202v1)) 是 LLaMA 中用于增强前馈网络（Feed-Forward Network, FFN）非线性表达能力的关键技术。SwiGLU 结合了 Swish 激活函数和门控机制，显著提升了模型的表现力和性能。此外，与 PaLM ([Chowdhery, 2022](https://arxiv.org/abs/2204.02311)) 中使用的$4 d$隐藏维度不同，LLaMA 采用了 $\frac{2}{3}d$ 的隐藏维度，从而在保持参数量和计算量不变的情况下，实现了更高的参数效率。

数学表达式：
$$
\operatorname{FFN}_{\mathrm{SwiGLU}}\left(x, W_1, W_3, W_2\right)=\left(\operatorname{Swish}\left(x W_1\right) \otimes x W_3\right) W_2
$$
其中：
- \( \text{Swish}(x) = x \cdot \sigma(x) \)（Swish 激活函数）。
- \( \sigma(x) = \frac{1}{1 + e^{-x}} \)（Sigmoid 函数）。
- \( \otimes \) 表示逐元素相乘。
- \( W_1, W_2, W_3 \) 为线性变换矩阵。

**优势**：
- **增强非线性表达**：SwiGLU 通过结合 Swish 激活函数与门控机制，能够更有效地捕捉复杂的模式和关系，提升 FFN 层的表达能力。
- **参数效率**：采用 $\frac{2}{3}d$ 的隐藏维度，在引入额外的线性变换矩阵的同时，保持了总参数量不变，实现了参数的高效利用。
- **性能提升**：在多项基准测试中，FFN_SwiGLU 显著提升了模型的性能，尤其在处理复杂任务和长文本时表现尤为出色。例如，在文本生成和理解任务中，SwiGLU 帮助模型更好地理解上下文和长距离依赖关系。

**实现细节**：
- **权重矩阵调整**：为了保持与传统 FFN 层相同的参数量和计算量，SwiGLU 通过减少隐藏层的维度（例如，将隐藏层大小从 4d 调整为 $\frac{2}{3}d$），在引入额外的线性变换矩阵的同时，确保整体模型的效率不受影响。
- **兼容性**：SwiGLU 作为 GLU 家族的一员，能够无缝集成到现有的 Transformer 架构中，替代传统的 ReLU 或 GELU 激活函数，提升模型的整体性能。

> 实现代码可以参考这个文件：[swiglu.py](https://github.com/syhya/syhya.github.io/blob/main/content/zh/posts/2025-01-22-llama3/swiglu.py)


#### Rotary Positional Embeddings (RoPE)

**Rotary Positional Embeddings (RoPE)** 是LLaMA3中用于表示序列中位置关系的技术，通过对Query和Key向量应用旋转变换，增强了模型对相对位置信息的感知能力。

**优势**：
1. **相对位置感知**：RoPE能够自然地捕捉词汇之间的相对位置关系，提升了长距离依赖的建模效果。
2. **计算效率高**：无需额外的计算，位置编码与词向量的结合在计算上是高效的，适用于大规模模型。
3. **适应不同长度的序列**：RoPE可以灵活处理不同长度的输入序列，不受固定位置编码的限制。
4. **兼容线性注意力**：RoPE 可以与线性注意力机制结合，保持注意力计算的线性复杂度，进一步提升处理长序列的效率。

---

### 1. 背景简介

理解单词在序列中的位置关系对于成功训练大型语言模型（LLM）至关重要。循环神经网络（RNN）通过递归计算隐藏状态来自然地捕捉序列中的位置信息。然而，Transformer这类基于自注意力机制的模型由于其并行计算的特性，无法直接感知单词之间的相对位置关系，因此需要额外的位置编码来提供这一信息。

位置编码的方法主要分为绝对位置编码和相对位置编码两大类。RoPE 则是一种创新性的绝对位置编码方法，旨在结合绝对位置编码和相对位置编码的优点，通过旋转变换实现相对位置感知。

---

### 2. 位置编码的基本概念

#### 2.1 绝对位置编码

绝对位置编码通过为每个位置生成一个固定或可训练的位置向量，并将其与词向量相加，从而为模型提供位置信息。常见的绝对位置编码方法包括：

- **三角函数位置编码**：例如，Vaswani 等人（2017）提出的使用正弦和余弦函数生成的位置编码。
  
  数学表达式：
  $$
  p_i = \left[\sin\left(\frac{i}{10000^{2j/d}}\right), \cos\left(\frac{i}{10000^{2j/d}}\right)\right]_{j=1}^{d/2}
  $$
  其中，\( p_i \) 是位置 \( i \) 的位置编码向量，\( d \) 是词向量的维度。

  **优点**：
  - 实现简单，与词向量的结合方式直接。

  **缺点**：
  - 无法自然地捕捉词汇之间的相对位置关系，限制了模型对长距离依赖的建模能力。
  - 对于不同长度的序列，可能需要重新生成位置编码。

- **可训练位置编码**：如BERT和GPT中使用的可训练位置编码。

#### 2.2 相对位置编码

相对位置编码旨在让模型关注词汇之间的相对距离，而不是绝对位置。这样，模型可以更灵活地处理不同长度的序列，并更有效地捕捉长距离依赖关系。

**常见方法**：
- **Google式相对位置编码**：在论文《Self-Attention with Relative Position Representations》中，Shaw 等人（2018）提出了一种扩展自注意力机制以考虑相对位置的方法。

  **优点**：
  - 自然地捕捉词汇之间的相对位置信息，有助于长距离依赖的建模。
  - 提高模型对序列长度的灵活适应能力。

  **缺点**：
  - 实现相对复杂，尤其是在自注意力机制中的集成。
  - 计算效率相对较低，特别是在处理长序列时。

---

### 3. 旋转位置编码（RoPE）的原理与实现

#### 3.1 RoPE 的设计思路

RoPE 的核心理念是通过旋转变换将绝对位置信息转化为相对位置信息，从而结合绝对位置编码和相对位置编码的优点。具体而言，RoPE 通过将查询（Query）和键（Key）向量分别乘以与其位置相关的旋转矩阵，使得内积计算中自然地体现了相对位置关系。

#### 3.2 数学表达式与推导

##### 3.2.1 2 维情况下的 RoPE 推导

为了更深入理解 RoPE 在二维情况下的工作原理，以下通过数学推导展示其如何实现相对位置编码。

**引言**

在二维空间中，向量的旋转可以通过复数的乘法来简化理解。RoPE 利用这一性质，通过对查询（Query）和键（Key）向量施加旋转变换，实现相对位置编码。

**复数与二维向量的对应关系**

- 一个复数 \( z = a + ib \) 可以表示为二维向量 \( \mathbf{v} = [a, b]^T \)。
- 复数的乘法对应于二维向量的旋转和缩放。

利用欧拉公式：
$$
e^{i\theta} = \cos\theta + i\sin\theta
$$
可以将复数旋转表示为二维向量的旋转矩阵。

**RoPE 的基本操作**

假设有二维的查询向量 \( \mathbf{q}_m \) 和键向量 \( \mathbf{k}_n \)，分别位于位置 \( m \) 和 \( n \)。RoPE 的目标是通过旋转变换，使得它们的内积仅依赖于相对位置 \( m - n \)。

**步骤 1：表示为复数**

将二维向量表示为复数形式：
$$
\mathbf{q}_m = q_m^{(1)} + i q_m^{(2)} \\
\mathbf{k}_n = k_n^{(1)} + i k_n^{(2)}
$$

**步骤 2：应用旋转变换**

对查询和键向量分别应用与其位置相关的旋转变换：
$$
f_q(\mathbf{q}_m, m) = \mathbf{q}_m \cdot e^{im\theta} = (q_m^{(1)} + i q_m^{(2)}) (\cos(m\theta) + i\sin(m\theta)) \\
f_k(\mathbf{k}_n, n) = \mathbf{k}_n \cdot e^{in\theta} = (k_n^{(1)} + i k_n^{(2)}) (\cos(n\theta) + i\sin(n\theta))
$$
其中，\( \theta \) 是一个预先定义的常数，用于控制旋转的速度。

**步骤 3：计算内积**

为了实现相对位置编码，计算内积：
$$
\langle f_q(\mathbf{q}_m, m), f_k(\mathbf{k}_n, n) \rangle = \text{Re}\left[ f_q(\mathbf{q}_m, m) \cdot \overline{f_k(\mathbf{k}_n, n)} \right]
$$
其中，\( \overline{f_k(\mathbf{k}_n, n)} \) 是 \( f_k \) 的共轭复数。

展开后：
$$
\langle f_q(\mathbf{q}_m, m), f_k(\mathbf{k}_n, n) \rangle = (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) + (q_m^{(1)}k_n^{(2)} - q_m^{(2)}k_n^{(1)})\sin((m-n)\theta)
$$
这个结果表明，内积的计算中自然地引入了相对位置 \( m - n \) 的影响。

**步骤 4：将二维向量与旋转矩阵对应**

将复数形式转化为矩阵形式：

查询向量 \( \mathbf{q}_m = [q_m^{(1)}, q_m^{(2)}]^T \) 和键向量 \( \mathbf{k}_n = [k_n^{(1)}, k_n^{(2)}]^T \)，旋转后的向量表示为：
$$
f_q(\mathbf{q}_m, m) = R(m\theta) \mathbf{q}_m \\
f_k(\mathbf{k}_n, n) = R(n\theta) \mathbf{k}_n
$$
其中，\( R(\phi) \) 是角度 \( \phi \) 的旋转矩阵：
$$
R(\phi) = \begin{bmatrix}
\cos\phi & -\sin\phi \\
\sin\phi & \cos\phi
\end{bmatrix}
$$

因此，内积可以表示为：
$$
\langle f_q(\mathbf{q}_m, m), f_k(\mathbf{k}_n, n) \rangle = [q_m^{(1)}, q_m^{(2)}] R((n - m)\theta) \mathbf{k}_n
$$
进一步展开：
$$
= q_m^{(1)} k_n^{(1)} \cos((m - n)\theta) + q_m^{(2)} k_n^{(2)} \cos((m - n)\theta) \\
+ q_m^{(1)} k_n^{(2)} \sin((m - n)\theta) - q_m^{(2)} k_n^{(1)} \sin((m - n)\theta)
$$
与前述结果一致，验证了 RoPE 在二维情况下实现相对位置编码的有效性。

**总结二维 RoPE 的实现步骤**：
1. **表示向量为复数**：将二维的查询和键向量表示为复数形式。
2. **应用旋转变换**：根据各自的位置 \( m \) 和 \( n \)，分别将查询和键向量旋转 \( m\theta \) 和 \( n\theta \) 的角度。
3. **计算内积的实部**：通过复数的乘法和内积运算，确保内积结果仅依赖于相对位置 \( m - n \)。
4. **利用旋转矩阵**：通过旋转矩阵的性质，将复数形式转化为矩阵形式，使得旋转操作更加直观和易于扩展到高维情况。

##### 3.2.2 高维情况下的 RoPE

对于词向量维度 \( d \) 为偶数的情况，RoPE 通过将向量拆分为 \( d/2 \) 个二维子向量，并对每个子向量应用独立的旋转矩阵，从而实现高维度的旋转位置编码。

**具体实现**：
$$
f_{\{q, k\}}(x_m, m) = R_{\Theta, m}^d \cdot W_{\{q, k\}} \cdot x_m
$$
其中，\( R_{\Theta, m}^d \) 是一个块对角矩阵，由 \( d/2 \) 个二维旋转矩阵组成，每个旋转矩阵对应一个不同的角度 \( \theta_i = 10000^{-2(i-1)/d} \)。

旋转矩阵的形式为：
$$
R_{\Theta, m}^d = \begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & & & \\
\sin(m\theta_1) & \cos(m\theta_1) & & & \\
& & \cos(m\theta_2) & -\sin(m\theta_2) & \\
& & \sin(m\theta_2) & \cos(m\theta_2) & \\
& & & & \ddots \\
\end{bmatrix}
$$

应用于自注意力机制中的查询和键向量：
$$
q_m^\top k_n = (R_{\Theta, m}^d W_q x_m)^\top (R_{\Theta, n}^d W_k x_n) = x_m^\top W_q^\top R_{\Theta, m}^{d\top} R_{\Theta, n}^d W_k x_n = x_m^\top W_q^\top R_{\Theta, n-m}^d W_k x_n
$$
其中，\( R_{\Theta, n-m}^d = R_{\Theta, m}^{d\top} R_{\Theta, n}^d \)，体现了相对位置信息。

#### 3.3 RoPE 的性质

**3.3.1 远程衰减特性**

通过选择 \( \theta_i = 10000^{-2(i-1)/d} \)，RoPE 的内积计算结果随着相对位置 \( |m - n| \) 增大而衰减。这一特性符合自然语言中的直觉，即距离较远的单词对当前词的影响应当较小。

**3.3.2 兼容线性注意力机制**

RoPE 的旋转操作保持了向量的范数不变，使其可以与线性注意力机制（如 Performer）无缝结合，进一步提升模型在处理长序列时的计算效率。

---

### 4. 总结

RoPE 通过旋转变换将绝对位置信息转化为相对位置信息，结合了绝对位置编码和相对位置编码的优点。其核心机制确保了模型对相对位置关系的感知能力，并且保持了高效的计算性能，特别是在处理长序列任务时表现出色。通过详细的数学推导，特别是在二维情况下的实现，RoPE 展示了其在自注意力机制中引入相对位置编码的有效性和灵活性。


### Grouped Query Attention (GQA)

Grouped Query Attention (GQA) ([Ainslie, 2023](https://arxiv.org/pdf/2305.13245)) 是 LLaMA3 中用于优化自注意力计算的关键技术。在大规模语言模型的推理过程中，每个注意力头（head）拥有独立的键（Key）和值（Value）参数会导致巨大的内存消耗。**Grouped Query Attention (GQA)** 旨在通过将多个查询（Query）头分组，并让每组共享一组键值头，从而在模型性能与推理效率之间取得更优的平衡。GQA 是 **Multi-Head Attention (MHA)** 和 **Multi-Query Attention (MQA)** 之间的一种折中方案：

- **MHA**：每个注意力头都有独立的 \(\mathbf{K}\) 和 \(\mathbf{V}\)。
- **MQA**：所有注意力头共享一组 \(\mathbf{K}\) 和 \(\mathbf{V}\)。
- **GQA**：将 \(H\) 个查询头划分为 \(G\) 组，每组共享一组 \(\mathbf{K}\) 和 \(\mathbf{V}\)（其中 \(1 < G < H\)）。

#### 1. 投影 (Projections)

给定输入序列 \(\mathbf{X} \in \mathbb{R}^{B \times S \times d}\)，首先通过线性变换投影得到查询、键和值矩阵：

$$
\mathbf{Q} = \mathbf{X} W_Q, \quad
\mathbf{K} = \mathbf{X} W_K, \quad
\mathbf{V} = \mathbf{X} W_V,
$$

其中，\(W_Q, W_K, W_V \in \mathbb{R}^{d \times d}\) 为可学习的投影矩阵。

#### 2. 头与分组 (Heads and Grouping)

- **头的切分**：将 \(\mathbf{Q}\)、\(\mathbf{K}\) 和 \(\mathbf{V}\) 分割成 \(H\) 个头，每个头的向量维度为 \(d_{\text{head}} = \frac{d}{H}\)。

$$
\mathbf{Q} = [\mathbf{Q}_1; \mathbf{Q}_2; \dots; \mathbf{Q}_H], \quad
\mathbf{K} = [\mathbf{K}_1; \mathbf{K}_2; \dots; \mathbf{K}_H], \quad
\mathbf{V} = [\mathbf{V}_1; \mathbf{V}_2; \dots; \mathbf{V}_H]
$$

- **分组**：将这 \(H\) 个查询头进一步划分为 \(G\) 组（\(1 < G < H\)）。对于第 \(g\) 组，包含 \(\frac{H}{G}\) 个查询头，并共享一组键值头 \(\mathbf{K}^g\) 和 \(\mathbf{V}^g\)。

$$
\mathcal{G} = \left\{ \mathcal{G}_1, \mathcal{G}_2, \dots, \mathcal{G}_G \right\}, \quad |\mathcal{G}_g| = \frac{H}{G} \quad \forall g \in \{1, 2, \dots, G\}
$$

下图展示了 GQA 与传统 MHA 和 MQA 的对比，可见在 GQA 中，**每组查询头公用一组键值头**。

{{< figure 
    src="attention_comparison.png" 
    caption="Fig. 5. Overview of Grouped Query Attention (GQA). (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))"
    align="center"
    width="100%"
>}}

#### 3. 组内注意力 (Intra-Group Attention)

对于第 \(g\) 组，令该组的查询向量为 \(\{\mathbf{Q}_i\}_{i \in \mathcal{G}_g}\)，共享的键值向量为 \(\mathbf{K}^g\) 和 \(\mathbf{V}^g\)。组内注意力的计算公式为：

$$
\text{Attention}_g(\mathbf{Q}_i, \mathbf{K}^g, \mathbf{V}^g) = \text{softmax}\left( \frac{\mathbf{Q}_i (\mathbf{K}^g)^\top}{\sqrt{d_{\text{head}}}} \right) \mathbf{V}^g
$$

其中，\(\sqrt{d_{\text{head}}}\) 为缩放因子，用于稳定梯度和数值计算。

#### 4. 拼接输出 (Concatenate & Output)

将所有组的注意力结果在通道维度上拼接，得到矩阵 \(\mathbf{O}\)，然后通过线性变换矩阵 \(W_O \in \mathbb{R}^{d \times d}\) 得到最终输出：

$$
\mathbf{O} = \text{Concat}\left( \text{Attention}_1, \text{Attention}_2, \dots, \text{Attention}_G \right) W_O
$$

其中，\(\text{Concat}\) 表示在通道维度上的拼接操作。

> 更多关于注意力机制在 **MHA**、**MQA** 和 **GQA** 之间的详细对比及代码示例，可参考博客：[Attention Mechanisms in Transformers: Comparing MHA, MQA, and GQA](https://syhya.github.io/posts/2025-01-16-group-query-attention/#grouped-query-attention-gqa)。


#### BPE

**tiktoken** tokenizer是LLaMA3采用的新一代分词器，相较于LLaMA2使用的SentencePiece BPE，tiktoken在以下方面有所改进：

1. **词汇表扩展**：词汇表从32k扩展至128k，覆盖更多语言和专业术语，减少了分词次数，提升了生成质量。
2. **编码效率**：优化了编码算法，减少了分词时间，提高了处理速度。
3. **生成质量**：通过更细粒度的词汇表示，提升了模型生成文本的连贯性和准确性。

数学表达式（简化版）：
$$
\text{Tokenize}(w) = \text{BPE}(w) \quad \text{vs} \quad \text{Tokenize}(w) = \text{tiktoken\_BPE}(w)
$$
- 其中，\( w \) 为输入词汇，tiktoken\_BPE通过更大词汇表减少了分词次数。

**优势**：
- **减少分词次数**：更大的词汇表使得更多词汇能作为单一token处理，减少了分词次数，提高了生成效率和质量。
- **提升生成质量**：更细粒度的词汇表示，使模型在生成文本时能够更准确地表达复杂语义。
- **编码速度快**：优化的编码算法提升了分词速度，适用于大规模模型的高效训练和推理。

#### 轻量级模型

为了适应边缘设备和移动设备的需求，LLaMA3推出了**1B和3B参数量的轻量级模型**，采用以下技术：

1. **剪枝技术**：通过系统性地移除网络中的冗余参数，减小模型规模，同时保持核心性能。
2. **知识蒸馏**：让小模型从大模型中学习，提升其在特定任务上的表现。
3. **优化部署**：针对移动设备的硬件架构进行优化，如针对Arm处理器的性能调优，确保模型在资源受限环境中的高效运行。

数学表达式（简化版）：
$$
\text{Pruned\_Model} = \text{Prune}(\text{Original\_Model}, \text{Pruning\_Rate})
$$
$$
\text{Distilled\_Model} = \text{Distill}(\text{Large\_Model}, \text{Small\_Model})
$$
- 其中，Prune表示剪枝操作，Distill表示知识蒸馏过程。

**优势**：
- **适应资源受限设备**：减小模型规模，使其适用于边缘设备和移动设备，推动了大语言模型的普及。
- **保持性能**：通过剪枝和知识蒸馏技术，保持了模型的核心性能和表现。
- **高效运行**：优化的模型结构和权重格式（如BFloat16）提升了计算效率，确保在移动设备上的高效运行。

#### 训练方法

**LLaMA3**在训练数据和方法上进行了全面升级，采用了更大规模的数据和更先进的训练技术：

1. **预训练阶段**：
   - **大规模数据扩展**：训练数据量达到15万亿token，覆盖更多语言、专业领域和多模态数据，提升了模型的泛化能力和多语言支持。
   - **扩展法则（Scaling Laws）**：
     - 根据Chinchilla扩展法则，优化模型的训练数据量和参数规模平衡，确保模型在关键任务上的最佳性能。
     - 数学表达式：
       $$
       \text{Optimal Data} \propto \text{Model Size}^{4/3}
       $$
       这一公式指导了数据和模型规模的平衡，确保随着模型规模的增加，训练数据量也按比例增长，避免模型过拟合或欠拟合。

2. **并行训练策略**：
   - **数据并行**：将训练数据分布到多个GPU上，提升数据处理速度。
   - **模型并行**：将模型的不同部分分布到多个GPU上，支持更大规模的模型训练。
   - **流水并行**：分阶段处理模型的不同部分，提高训练效率。
   
   数学表达式：
   $$
   \text{Total Throughput} = \text{Data Parallelism} \times \text{Model Parallelism} \times \text{Pipeline Parallelism}
   $$
   - 其中，总吞吐量（Total Throughput）是数据并行、模型并行和流水并行的乘积，显著提升了训练效率。

3. **硬件优化**：
   - **高效利用GPU**：在16K GPU上实现每GPU超过400 TFLOPS的计算利用率，通过定制的24K GPU集群进行训练，确保训练过程的高效性和稳定性。
   - **错误处理与存储优化**：
     - **自动错误检测与处理**：确保训练过程的连续性和高效性。
     - **可扩展存储系统**：减少检查点和回滚的开销，提高数据存储效率。

4. **微调阶段**：
   - **多轮对齐步骤**：
     - **监督微调（SFT）**：使用高质量的标注数据进一步优化模型性能。
     - **拒绝采样（Rejection Sampling）**：通过拒绝低质量内容，提升生成文本的质量。
     - **近端策略优化（Proximal Policy Optimization, PPO）和直接策略优化（Direct Policy Optimization, DPO）**：结合两者的优势，优化模型的生成策略，使其更符合人类偏好。
   
   数学表达式：
   $$
   \mathcal{L}_{\text{RLHF}} = \mathbb{E}_{\theta \sim \pi_{\theta}} \left[ r(s, a) \right]
   $$
   - 其中，\( \mathcal{L}_{\text{RLHF}} \)为RLHF的损失函数，\( \pi_{\theta} \)为策略分布，\( r(s, a) \)为奖励函数。

5. **多模态训练**：
   - **视觉语言模型**：结合图像和文本数据，提升模型在多模态任务中的表现。
   - **代码数据扩展**：增加代码token数量，提升模型在编程任务中的表现。

6. **模型安全与质量控制**：
   - **数据过滤pipeline**：
     - **启发式过滤器**：基于规则的过滤，提高数据质量。
     - **NSFW过滤器**：去除不适内容，确保数据的安全性。
     - **语义重复数据删除**：使用语义分析技术，删除内容高度相似的数据。
     - **文本分类器**：预测数据质量，进一步优化数据集。

7. **优化训练堆栈**：
   - **高级训练堆栈**：自动检测和处理训练过程中的错误，提升硬件可靠性。
   - **性能调优**：针对不同硬件平台进行优化，确保训练过程的高效性。

**LLaMA3**通过这些先进的训练方法和优化策略，显著提升了模型的性能和适应性，成为开源大语言模型领域的领先者。

## 总结

**LLaMA**系列模型从LLaMA1到LLaMA3，体现了大规模预训练语言模型的技术进化与产业影响。通过不断扩展训练数据量、优化模型架构和引入先进的训练方法，LLaMA系列在性能、多语言支持和多模态能力等方面取得了显著的提升。其开源策略不仅推动了全球AI社区的创新和发展，也为产业应用提供了强大的技术支持。


## 参考资料

1. [Hendrycks and Gimpel, 2016](https://arxiv.org/pdf/1606.08415.pdf)
2. [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202.pdf)
3. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
4. [LLaMA2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288)
5. [meta-llama repo](https://github.com/meta-llama/llama/blob/main/llama/model.py)