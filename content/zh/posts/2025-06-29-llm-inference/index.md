---
title: "大语言模型推理"
date: 2025-06-29T12:00:00+08:00
author: "Yue Shui"
tags: ["LLM", "Inference", "Quantization", "Pruning", "Knowledge Distillation", "KV Cache", "Attention", "Speculative Decoding", "FlashAttention", "vLLM", "Transformer", "Sparsity", "Mixture of Experts"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

近年来，大语言模型 (Large Language Models, LLMs) 在自然语言处理、代码生成乃至多模态交互等领域取得了革命性的突破。然而，这些模型强大的能力背后是巨大的计算和内存开销，尤其是在推理 (Inference) 阶段。高效地部署和运行这些数十亿甚至数万亿参数的模型，已成为将 LLM 技术规模化应用到实际产品中的核心挑战。

LLM 推理的挑战主要源于两个方面：
1.  **巨大的内存占用**：除了模型参数本身，推理过程中还需要存储大量的中间状态，尤其是 **KV Cache**。例如，对于一个批处理大小为 512、序列长度为 2048 的请求，其 KV 缓存可能高达 3TB，数倍于模型本身的体积。此外，自注意力计算复杂度随序列长度呈二次增长。
2.  **低并行性**：LLM 的文本生成本质上是一个自回归 (Autoregressive) 过程，即逐个 Token 生成，下一个 Token 的生成依赖于之前所有已生成的 Tokens。这种串行特性使得解码过程难以高效并行。

## Token 生成原理

为了更好地理解后续的优化技术，我们首先需要了解大模型是如何生成文本的，以及其推理过程中的关键瓶颈。

### 自回归生成

目前主流的大语言模型如 GPT 采用 Decoder-Only Transformers 的架构，使用**自回归 (Autoregressive)** 的方式生成文本。其基本思想是，文本序列的概率可以被分解为一系列条件概率的乘积。给定一个初始的上下文词序列 $W_0$（通常是用户的输入 Prompt），模型逐个预测下一个词 (Token)，并将新生成的词加入到上下文，作为下一步预测的输入。这个过程可以用以下公式表示：

$$
P(w_{1:T} | W_0) = \prod_{t=1}^{T} P(w_t | w_{1:t-1}, W_0), \text{ with } w_{1:0} = \emptyset
$$

其中，$w_t$ 是在时间步 $t$ 生成的词，$w_{1:t-1}$ 是在时间步 $t$ 之前已经生成的所有词的序列。整个生成过程持续进行，直到模型生成一个特殊的终止符 (EOS, End-of-Sequence) 或者达到预设的最大长度 $T$。

### Prefilling 与 Decoding

自回归的生成方式决定了 LLM 的推理过程可以被清晰地划分为两个阶段：**Prefilling (预填充) 阶段**和**Decoding (解码) 阶段**。

{{< figure
    src="prefilling_decoding.png"
    caption="Fig. 1. The Prefilling and Decoding Stages of LLM Inference. (Image source: [Zhou et al., 2024](https://arxiv.org/abs/2404.14294))"
    align="center"
    width="100%"
>}}

1.  **Prefilling 阶段**：在此阶段，模型并行处理输入的整个 Prompt (例如，上图 1 中的 "I, like, natural, language")，并计算出第一个输出 Token ("Processing") 的概率分布。这个阶段的计算特点是**并行度高**，因为输入的所有 Token 可以一次性送入 Transformer 模型进行计算，这使得计算密集型操作 (如矩阵乘法) 可以充分利用 GPU 的并行计算能力，属于**计算密集型 (Compute-bound)**。

2.  **Decoding 阶段**：在此阶段，模型逐个生成后续的 Token。每生成一个 Token，它就会被添加到现有序列的末尾，作为下一次预测的输入。这个过程是**串行**的，因为下一个 Token 的生成依赖于前一个 Token。因此，这个阶段的计算特点是**内存访问密集 (Memory-bound)**，其主要瓶颈在于从 GPU 显存中加载庞大的模型权重，而不是计算本身。

{{< figure
    src="inference_memory.png"
    caption="Fig. 2. Illustration of the memory variation through time (latency) during one generation process. Note that author ignore the activation size in this figure for a simplification. (Image source: [Zhou et al., 2024](https://arxiv.org/abs/2404.14294))"
    align="center"
    width="100%"
>}}

为了加速解码过程，现代 LLM 推理框架普遍采用 **KV Cache** 技术。在 Transformer 的自注意力机制中，每个 Token 都需要与它之前的所有 Token 进行交互。为了避免在生成每个新 Token 时都重新计算前面所有 Token 的 Key (K) 和 Value (V) 向量，系统会将这些计算好的 K 和 V 值缓存起来。这个缓存就是 KV Cache。

如图 2 所示，随着生成序列的增长，KV Cache 的体积会线性增大。对于一个拥有数十亿参数的模型和长序列，KV Cache 可能占用数 GB 甚至数十 GB 的显存。这使得显存成为 LLM 推理中最稀缺的资源，极大地限制了系统能够同时处理的请求数量 (即批处理大小，Batch Size)，从而直接影响了推理的吞吐量。因此，**如何高效地管理和优化 KV Cache 是 LLM 推理优化的核心问题之一**。

### 解码策略

在每个解码步骤，模型会输出一个覆盖整个词汇表的概率分布。如何从这个分布中选择下一个 Token，是由解码策略 (或称 Token 生成策略) 决定的。不同的策略会显著影响生成文本的质量、创造性和连贯性。

#### 贪心搜索

贪心搜索是最简单的解码策略。在每个时间步 $t$，它总是选择概率最高的那个词作为输出：

$$
w_t = \underset{w}{\operatorname{argmax}} P(w | w_{1:t-1})
$$

通过这样的方式，它可以极大地减少计算复杂度，快速地产生结果，但这种方法存在明显的局限性：因为每一步只做局部的最优选择，贪心搜索很容易陷入局部最优，从而忽略整体更优的可能性，导致生成的文本常常显得枯燥、重复，缺乏多样性和创造性。

{{< figure
    src="greedy_search.svg"
    caption="Fig. 3. At each time step, greedy search selects the token with the highest conditional probability. (Image source: [d2l-en, 2019](https://d2l.ai/chapter_recurrent-modern/beam-search.html#id1))"
    align="center"
    width="50%"
>}}

**代码实现**:
```python
import torch
import torch.nn.functional as F

def greedy_search(model, input_ids, max_len=20, eos_token_id=2):
    """
    A simple implementation of Greedy Search.
    `model` should be a function that takes input_ids and returns logits.
    """
    generated_sequence = input_ids
    for _ in range(max_len):
        # Get logits for the last token
        logits = model(generated_sequence)
        next_token_logits = logits[:, -1, :]
        
        # Select the token with the highest probability
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token to the sequence
        generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
        
        # Stop if EOS token is generated
        if next_token.item() == eos_token_id:
            break
            
    return generated_sequence
```

#### 束搜索

为了克服贪心搜索的局部最优问题，束搜索在每个解码步骤保留 $k$ 个 (`num_beams` 或束宽度) 最可能的候选序列 (称为“束”)。在下一步，它会基于这 $k$ 个序列扩展，并再次选出总概率最高的 $k$ 个新序列。最后，算法会从所有完成的候选序列中选择一个整体概率最高的作为最终输出。

{{< figure
    src="beam_search.svg"
    caption="Fig. 4. The process of beam search (beam size $=2$; maximum length of an output sequence $=3$ ). The candidate output sequences are $A, C, A B, C E, A B D$, and $C E D$. (Image source: [d2l-en, 2019](https://d2l.ai/chapter_recurrent-modern/beam-search.html#id1))"
    align="center"
    width="100%"
>}}

这种方式扩大了搜索空间，有效地减少了局部最优的影响，通常能够生成更高质量、更连贯的文本。然而，束搜索的本质依然是选择整体概率最高的路径，这使得它在处理开放式生成任务时仍会倾向于产生高频、常见的表达，可能缺乏创造性和多样化的输出。

#### 温度采样


{{< figure
    src="temperature.png"
    caption="Fig. 5. Illustration of Temperature Sampling. (Image source: [Big Hummingbird Blogs, 2024](https://www.bighummingbird.com/blogs/llm-hyperparameter))"
    align="center"
    width="80%"
>}}

与确定性的搜索方法不同，采样方法引入了随机性，使得生成的文本更加多样和富有创造力。最基础的采样方法是直接根据模型的概率分布进行随机抽样。**温度采样**通过一个温度系数 $T$ 来调节原始概率分布的形状，加在 Softmax 上。温度系数用来调节大模型输出 Token 的概率分布的平坦程度，越大概率分布越平坦，输出越随机，越小概率分布越极端，输出越稳定。

$$
P_T(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $z_i$ 是模型对词 $w_i$ 输出的 logit。
*   当 $T \to 1$ 时，概率分布不变。
*   当 $T < 1$ 时 (降温)，分布会变得更“尖锐”，高概率的词更容易被选中，生成结果更接近贪心搜索。
*   当 $T > 1$ 时 (升温)，分布会变得更“平坦”，低概率的词也有机会被选中，生成结果更具多样性和随机性。

#### Top-K 采样

**Top-K 采样** ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)) 在采样前，只保留概率最高的 $K$ 个候选词，然后在这 $K$ 个词中重新进行归一化和采样。这有效防止了模型从概率极低的词中采样，避免生成不连贯的文本。但其缺点是 $K$ 的取值是固定的，无法动态适应不同的概率分布。

{{< figure
    src="top_k.png"
    caption="Fig. 6. Illustration of Top-K Sampling. (Image source: [Big Hummingbird Blogs, 2024](https://www.bighummingbird.com/blogs/llm-hyperparameter))"
    align="center"
    width="80%"
>}}

#### Top-p 采样

**Top-p 采样** ([Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)) 采用动态选择候选词集合的方法。它从概率最高的词开始，累加它们的概率，直到总和超过一个预设的阈值 $p$ (例如 0.9)。然后，模型只在这个动态生成的、最小的候选词集合 $V_{\text{top-p}}$ 中进行采样。这种方法兼顾了文本的连贯性和创造性，是目前开放式文本生成中最常用且效果最好的策略之一。

{{< figure
    src="top_p.png"
    caption="Fig. 7. Illustration of Top-p Sampling. (Image source: [Big Hummingbird Blogs, 2024](https://www.bighummingbird.com/blogs/llm-hyperparameter))"
    align="center"
    width="80%"
>}}

**联合采样代码实现 (Top-K, Top-p, Temperature)**:
```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_with_sampling(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=2):
    for _ in range(max_new_tokens):
        # Crop context if it's too long
        idx_cond = idx if idx.size(1) <= model.config.max_position_embeddings else idx[:, -model.config.max_position_embeddings:]
        
        # Forward pass to get logits
        logits = model(idx_cond).logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Apply Top-p (Nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(1, indices_to_remove, -float('Inf'))

        # Convert logits to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append sampled index and check for EOS
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == eos_token_id:
            break
            
    return idx
```

#### 推测解码

**推测解码 (Speculative Decoding)** ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) 是一种创新的加速技术，旨在用小模型的速度实现大模型的生成质量，从而在不牺牲生成质量的前提下降低延迟。

它使用一个小型、快速的 Draft Model 一次性生成多个 (例如 $k$ 个) 候选 Token。然后，大型的 Target Model 并行地对这 $k$ 个 Token 进行一次前向传播验证。如果草稿模型预测的 Token 与目标模型一致，那么这些 Token 就被接受，从而实现了一次前向传播生成多个 Token 的效果。如果不一致，则丢弃草稿模型的后续预测，并使用目标模型的预测进行修正。

{{< figure
    src="online_speculative_decoding.png"
    caption="Fig. 8. Overview of online speculative decoding (OSD) framework: For each prompt, the draft model suggests multiple tokens and the target model performs the verification. (Image source: [Liu et al., 2024](https://arxiv.org/abs/2310.07177))"
    align="center"
    width="100%"
>}}

只要草稿模型与目标模型的预测有一定的一致性，推测解码就能显著降低生成延迟。其变体包括使用模型自身的早期层作为草稿模型的**自推测解码** (Self-speculative decoding) 等。

#### 启发式策略

*   **Best-of-N / Majority Vote**: 这些方法通过生成多个候选答案来提升最终结果的质量和鲁棒性。
    *   **Best-of-N**: 让大模型输出 N 个回答，然后通过一个独立的评估模型 (Verifier) 或基于奖励模型 (Reward Model) 打分，选择分数最高的 (Best) 作为最终回答。
    *   **Majority Vote / Self-Consistency**: 让大模型针对同一问题生成多个不同的推理路径 (Chain-of-Thought) 和答案，然后通过多数投票的方式选出最一致的答案作为最终结果。这种方法在需要复杂推理的任务上尤其有效。


## 优化方法概述

了解了推理的基本原理后，我们来深入探讨如何优化这个过程。推理优化的目标主要有三个：**降低延迟 (Latency)**、**提高吞吐量 (Throughput)** 和 **减少内存占用 (Memory Footprint)**。现有技术可以大致分为三大类：模型压缩、内存与计算优化、以及高效模型架构。

通常，模型推理优化的目标包括：

*   通过使用更少的 GPU 设备和显存来**降低模型的内存占用**；
*   通过减少所需的浮点运算次数 (FLOPs) 来**降低计算复杂度**；
*   **降低推理延迟**，让模型运行得更快。

为了在内存和时间上降低推理成本，可以采用多种方法：

1.  **应用各种并行技术**，将模型扩展到大量 GPU 上。通过对模型组件和数据的智能并行化，可以运行数万亿参数的模型。
2.  **内存卸载**，将暂时不用的数据卸载到 CPU，在需要时再读回。这有助于减少内存使用，但会增加延迟。
3.  **智能批处理策略**；例如，EffectiveTransformer 将连续的序列打包在一起，以消除批处理中的填充 (padding)。
4.  **网络压缩技术**，如**剪枝 (pruning)、量化 (quantization)、蒸馏 (distillation)**。参数数量或位宽更小的模型，自然需要更少的内存和更快的运行速度。
5.  **针对特定模型架构的改进**。许多架构上的改变，特别是针对注意力层的改进，有助于加快 Transformer 的解码速度。

可以查阅之前关于[大型模型训练](https://syhya.github.io/zh/posts/2025-03-01-train-llm/)的文章，了解不同类型的训练并行化和内存节省设计，包括 CPU 内存卸载。本文将重点关注网络压缩技术和针对 Transformer 模型的架构改进。

## 知识蒸馏

**知识蒸馏 (Knowledge Distillation, KD)** ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) 是一种直接的方法，通过将一个预训练好的昂贵模型 (“教师模型”) 的知识迁移到一个更小、更廉价的模型 (“学生模型”) 中，来构建一个更小的模型以加速推理。除了要求学生模型的输出空间与教师模型匹配以便构建合适的学习目标外，对于学生模型的架构没有太多限制。

{{< figure
    src="knowledge_distillation.png"
    caption="Fig. 9. The generic framework of teacher-student knowledge distillation training. (Image source: [Gou et al., 2020](https://arxiv.org/abs/2006.05525))"
    align="center"
    width="90%"
>}}

给定一个数据集，学生模型通过蒸馏损失函数来学习模仿教师模型的输出。通常神经网络会有一个 softmax 层；例如，一个 LLM 会输出一个关于 Token 的概率分布。我们用 $\mathbf{z}_t$ 和 $\mathbf{z}_s$ 分别表示教师模型和学生模型在 softmax 之前的 logits 层。蒸馏损失通过最小化两个带有较高温度 $T$ 的 softmax 输出之间的差异来实现。当存在真实标签 $\mathbf{y}$ 时，我们可以将其与一个监督学习目标 (例如交叉熵) 结合起来，该目标作用于真实标签和学生的软 logits 之间。

$$
\mathcal{L}_{\mathrm{KD}}=\mathcal{L}_{\text {distill }}\left(\operatorname{softmax}\left(\mathbf{z}_t, T\right), \operatorname{softmax}\left(\mathbf{z}_s, T\right)\right)+\lambda \mathcal{L}_{\mathrm{CE}}\left(\mathbf{y}, \mathbf{z}_s\right)
$$

其中 $\lambda$ 是一个超参数，用于平衡软目标和硬目标的学习。$\mathcal{L}_{\text {distill}}$ 的一个常见选择是 KL 散度或交叉熵。

一个早期的成功案例是 **DistilBERT** ([Sanh et al. 2019](https://arxiv.org/abs/1910.01108))，它能够将 BERT 的参数减少 40%，同时在下游微调任务上保持 BERT 97% 的性能，并且运行速度快 71%。DistilBERT 的预训练损失是软蒸馏损失、监督训练损失 (在 BERT 中即掩码语言建模损失 $\mathcal{L}_{\text{MLM}}$) 以及一个特殊的余弦嵌入损失的组合，后者用于对齐教师和学生模型之间的隐藏状态向量。


{{< figure
    src="DistilBERT.png"
    caption="Fig. 10. The performance of DistilBERT (Image source: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108))"
    align="center"
    width="90%"
>}}

蒸馏可以很容易地与**量化**、**剪枝**或**稀疏化**技术相结合，其中教师模型是原始的全精度、密集模型，而学生模型则被量化、剪枝或修剪以达到更高的稀疏度。

## 量化

为了在推理过程中进一步提升模型性能，我们可以超越低精度浮点数，转而使用**量化 (Quantization)** 技术。量化将模型的浮点权重转换为低位宽的整数表示，例如 8 位整数 (INT8)，甚至 4 位整数 (INT4)。

在深度神经网络上应用量化通常有两种方法：

1.  **训练后量化 (Post-Training Quantization, PTQ)**：首先将模型训练至收敛，然后在不进行更多训练的情况下将其权重转换为较低的精度。与训练相比，这种方法的实现成本通常很低。
2.  **量化感知训练 (Quantization-Aware Training, QAT)**：在预训练或进一步微调的过程中应用量化。QAT 能够获得更好的性能，但需要额外的计算资源和对代表性训练数据的访问。

### 精度对比

在深度学习领域，数值精度决定着计算速度和模型性能之间的微妙平衡。理解不同浮点数和整数格式的优缺点，是优化大规模模型性能的关键。浮点数在计算机中以三部分表示：

* **符号位 (Sign)**：表示数值的正负。
* **指数位 (Exponent)**：决定数值的动态范围。
* **尾数位 (Mantissa 或 Significand)**：决定数值的精确度，为了方便通常我们将尾数位称为小数 (fraction)。

{{< figure
    src="combined_float_diagrams.png"
    caption="Fig. 11. fp32 vs fp16 vs bf16 (Image source: [Raschka, 2023](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html))"
    align="center"
    width="70%"
>}}


| 类型       | 总位数 | 符号位 | 指数位 | 尾数位 | 特性                                  |
| -------- | --- | --- | --- | --- | ----------------------------------- |
| [**FP64 (双精度)**](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) | 64  | 1   | 11  | 52  | 极高精度，广泛用于科学计算，但计算昂贵，内存占用大，深度学习中少用   |
| [**FP32 (单精度)**](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) | 32  | 1   | 8   | 23  | 深度学习训练标准格式，速度适中，内存占用较大              |
| [**FP16 (半精度)**](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) | 16  | 1   | 5   | 10  | 更快计算，内存占用是 FP32 的一半，但动态范围受限，易发生数值溢出   |
| [**BF16 (Brain Floating Point)**](https://cloud.google.com/tpu/docs/bfloat16) | 16  | 1   | 8   | 7   | 动态范围与 FP32 相同，避免溢出，更适合大语言模型，精度略低于 FP16 |

纯粹的 FP16 精度虽速度快、内存小，但由于动态范围有限，极易出现数值溢出 (overflow) 和下溢 (underflow)，使训练不稳定甚至无法收敛。因此，采用[混合精度训练 (Mixed-Precision)](https://syhya.github.io/posts/2025-03-01-train-llm/#mixed-precision-training)至关重要。

量化将浮点数映射为整数，进一步降低计算复杂度与内存占用。具体来说：

* **INT8**：占用内存仅为 FP32 的 1/4，显著加速推理速度，但可能会略微降低模型精度。
* **INT4**：更极端的压缩方案，更适合资源极其受限的设备，或需要极高吞吐量的推理场景。

### Transformer 量化挑战

许多关于 Transformer 模型量化的研究都有一个共同的发现：简单的低精度 (例如 8-bit) 训练后量化会导致显著的性能下降，这主要是由于**激活值存在高动态范围**，而一个简单的激活值量化策略无法保持模型的性能。


{{< figure
    src="glue_benchmark.png"
    caption="Fig. 12. Only quantizing model weights to 8-bit while keeping activation at full precision (`W8A32`) achieves much better results when activations are quantized to 8-bit irrespective of whether weights are in lower precision (`W8A8` and `W32A8`). (Image source: [Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948))"
    align="center"
>}}

[Bondarenko et al. (2021)](https://arxiv.org/abs/2109.12948) 在小型 BERT 模型进行实验发现由于输出张量中存在强烈的**离群值 (outliers)**，FFN (前馈网络) 的输入和输出具有非常不同的动态范围。因此，对 FFN 的残差和进行逐张量 (per-tensor) 量化可能会导致显著的误差。

随着模型规模增长到数十亿参数，所有 Transformer 层中都开始出现**大幅度的离群特征**，这导致简单的低比特量化失败。研究人员在大于 6.7B 参数大小的 **OPT** ([Zhang et al. 2022](https://arxiv.org/abs/2205.01068)) 模型中观察到了这种现象。更大的模型有更多的层带有极端的离群值，而这些离群特征对模型性能有显著影响。在少数维度上，激活值离群点的规模可以比其他大多数值大约 100 倍。

{{< figure
    src="int8_outliner.png"
    caption="Fig. 13. The mean zero-shot accuracy over a set of language tasks (WinoGrande, HellaSwag, PIQA, LAMBADA) of OPT models of increasing sizes. (Image source: [Dettmers et al. 2022](https://arxiv.org/abs/2208.07339))"
    align="center"
    width="80%"
>}}

### 训练后量化

#### 混合精度量化

解决上述量化挑战最直接的方法是为权重和激活值实现不同精度的量化。

**GOBO** ([Zadeh et al. 2020](https://arxiv.org/abs/2005.03842)) 是最早在 BERT 上应用训练后量化的模型之一。它假设每层的模型权重服从高斯分布，因此通过跟踪每层的均值和标准差来检测离群值。离群特征保持原始形式，而其他值被分成多个桶 (bin)，只存储相应的桶索引和质心值。

{{< figure
    src="gobo.png"
    caption="Fig. 14. The pseudocode for GOBO algorithm. (Image source: [Zadeh et al. 2020](https://arxiv.org/abs/2005.03842))"
    align="center"
    width="80%"
>}}

基于在 BERT 中只有某些激活层 (例如 FFN 后的残差连接) 会导致大的性能下降的观察，[Bondarenko et al. (2021)](https://arxiv.org/abs/2109.12948) 采用了混合精度量化，对有问题的激活值使用 16 位量化，而对其他部分使用 8 位量化。


**LLM.int8()** ([Dettmers et al. 2022](https://arxiv.org/abs/2208.07339)) 中的混合精度量化通过两种混合精度分解实现：

1.  因为矩阵乘法包含一系列行向量和列向量之间的独立内积，我们可以对每个内积施加独立的量化：每一行和每一列都通过其绝对值最大值进行缩放，然后量化到 INT8。
2.  离群的激活特征 (例如比其他维度大 20 倍) 保持在 FP16 格式，但它们只占总权重的一小部分。如何识别离群值是经验性的。

{{< figure
    src="llm_int8_quantization.png"
    caption="Fig. 15. Two mixed-precision decompositions of `LLM.int8()`. (Image source: [Dettmers et al. 2022](https://arxiv.org/abs/2208.07339))"
    align="center"
    width="100%"
>}}

#### 细粒度量化

{{< figure
    src="quantization_granularity.png"
    caption="Fig. 16. Comparison of quantization at different granularity. $d$ is the model size / hidden state dimension and $h$ is the number of heads in one MHSA (multi-head self-attention) component. (Image source: [Lilian, 2023](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))"
    align="center"
    width="100%"
>}}

简单地将一层中的整个权重矩阵进行量化 (“逐张量”或“逐层”量化) 最容易实现，但无法达到良好的量化粒度。

**Q-BERT** ([Shen, et al. 2020](https://arxiv.org/abs/1909.05840)) 对一个微调过的 BERT 模型应用了**分组量化 (group-wise quantization)**，将 MHSA (多头自注意力) 中每个头对应的单个矩阵 $W$ 视为一个组，然后应用基于 Hessian 矩阵的混合精度量化。

**逐嵌入组 (Per-embedding group, PEG)** ([Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948)) 激活值量化的动机是观察到离群值只出现在 $d$ (隐藏状态/模型大小) 维度中的少数几个维度上。逐嵌入量化计算成本相当高。相比之下，PEG 量化将激活张量沿着嵌入维度分成几个大小均匀的组，其中同一组中的元素共享量化参数。为了确保所有离群值被分到同一组，他们应用了一种确定性的基于范围的嵌入维度排列，其中维度按其值范围排序。

**ZeroQuant** ([Yao et al. 2022](https://arxiv.org/abs/2206.01861)) 对权重使用分组量化 (与 Q-BERT 相同)，对激活值使用**逐 Token 量化 (token-wise quantization)**。为了避免昂贵的量化和反量化计算，ZeroQuant 构建了定制化的内核，将量化操作与其前一个操作符合并。

#### 二阶信息用于量化

**Q-BERT** ([Shen, et al. 2020](https://arxiv.org/abs/1909.05840)) 为其混合精度量化开发了**Hessian 感知量化 (Hessian AWare Quantization, HAWQ)** ([Dong, et al. 2019](https://arxiv.org/abs/1905.03696))。其动机是具有较高 Hessian 谱 (即较大的顶层特征值) 的参数对量化更敏感，因此需要更高的精度。这实质上是一种识别离群值的方法。

从另一个角度看，量化问题是一个优化问题。给定权重矩阵 $\mathbf{W}$ 和输入矩阵 $\mathbf{X}$，我们希望找到一个量化后的权重矩阵 $\hat{\mathbf{W}}$ 来最小化均方误差 (MSE)：

$$
\hat{\mathbf{W}}^* = \arg \min_{\hat{\mathbf{W}}} |\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}|
$$

[**GPTQ**](https://github.com/IST-DASLab/gptq) ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)) 在 **OBC (Optimal Brain Compression)** ([Frantar et al. 2022](https://arxiv.org/abs/2208.11580)) 方法基础上进行优化，将权重矩阵 $\mathbf{W}$ 视为行向量 $\mathbf{w}$ 的集合，并独立地对每一行进行量化。GPTQ 迭代地量化更多的权重，这些权重是贪婪选择的，以最小化量化误差。对所选权重的更新有一个利用 Hessian 矩阵的闭式解公式。


{{< figure
    src="gptq.png"
    caption="Fig. 17. The pseudocode for GPTQ algorithm. (Image source: [Frantar et al. 2022](https://arxiv.org/abs/2210.17323))"
    align="center"
    width="100%"
>}}

GPTQ 可以将 OPT-175B 中权重的位宽降低到 **3 bit** 或 **4 bit** 而没有太多性能损失，但它只适用于模型权重，不适用于激活值。

#### 离群值平滑

{{< figure
    src="migrate_quantization_difficulty.png"
    caption="Fig. 18. Magnitude of the input activations and weights of a linear layer in OPT-13B before and after SmoothQuant (Image source: [Xiao et al. 2022](https://arxiv.org/abs/2211.10438))"
    align="center"
    width="100%"
>}}

从上图我们可以看到在 Transformer 模型中，激活值比权重更难量化。并且有以下三个特征：

1. **激活比权重更难量化**： 权重用 INT8/INT4 量化几乎不损准确率，但激活则更敏感。
2. **异常值放大激活量化难度**： 激活中的极端大值约为普通值的 100 倍，直接 INT8 量化会把多数小值压成 0。
3. **异常值固定在少数通道**： 这些极端值长期集中在固定通道，通道间分布高度不均。

**SmoothQuant** ([Xiao et al. 2022](https://arxiv.org/abs/2211.10438)) 提出了一个聪明的解决方案，**通过数学上等价的变换将离群特征从激活值平滑到权重**，然后对权重和激活值都进行量化 (`W8A8`)。因此，SmoothQuant 比混合精度量化具有更好的硬件效率。

{{< figure
    src="smooth_quant.png"
    caption="Fig. 19. SmoothQuant migrates the scale variance from activations to weights offline to reduce the difficulty of activation quantization. Both the resulting new weight and activation matrices are easy to quantize. (Image source: [Xiao et al. 2022](https://arxiv.org/abs/2211.10438))"
    align="center"
    width="80%"
>}}

考虑一个逐通道的平滑因子 $\mathbf{s}$，SmoothQuant 根据以下公式缩放权重：

$$
\mathbf{Y} = (\mathbf{X}\text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s})\mathbf{W}) = \hat{\mathbf{X}}\hat{\mathbf{W}}
$$

平滑因子可以很容易地离线融合到前一层的参数中。一个超参数 $\alpha$ 控制将量化难度从激活值迁移到权重的程度：$\mathbf{s} = \max(|\mathbf{X}_j|)^\alpha / \max(|\mathbf{W}_j|)^{1-\alpha}$。论文发现在实验中，对于许多 LLM, $\alpha=0.5$ 是一个最佳选择。对于激活值中具有更显著离群值的模型，可以调整 $\alpha$ 使其更大。

### 量化感知训练

量化感知训练 (Quantization-Aware Training, QAT) 将量化操作融合到预训练或微调过程中，直接学习模型权重的低比特表示，以额外的训练时间和计算资源为代价，获得更高的性能。

常见的量化感知训练方法包括：

* **直接微调法**：首先对模型进行一次性量化，然后在原始预训练数据集或具有代表性的训练数据集上进一步**微调**模型，使模型对量化误差敏感并主动补偿，从而提升量化后模型的性能。训练目标可以与原始预训练目标一致 (如语言模型中的负对数似然 NLL 或掩蔽语言建模 MLM)，也可根据具体下游任务定义特定目标 (如分类任务的交叉熵)。一个典型的实现是 [QLoRA](https://syhya.github.io/zh/posts/2025-03-01-train-llm/#qlora)，它通过低比特 (如 4-bit) 基座模型结合全精度 LoRA 适配器实现高效微调。

* **知识蒸馏法**：将全精度模型作为教师 (teacher)，低精度模型作为学生 (student)，通过**蒸馏损失**引导学生模型接近教师模型的性能。**ZeroQuant** ([Yao et al. 2022](https://arxiv.org/abs/2206.01861)) 采用的逐层知识蒸馏 (Layer-by-layer Knowledge Distillation, LKD) 技术即属于这一方法，它逐层地进行模型权重量化，每一层量化后的网络以对应的全精度层为教师，通过最小化两者之间权重计算结果的均方误差 (MSE) 来提升性能。

$$
\mathcal{L}_{L K D, k}=M S E\left(L_k \cdot L_{k-1} \cdot L_{k-2} \cdot \ldots \cdot L_1(\boldsymbol{X})-\widehat{L}_k \cdot L_{k-1} \cdot L_{k-2} \cdot \ldots \cdot L_1(\boldsymbol{X})\right)
$$


## 剪枝

网络剪枝 (Pruning) 通过删除不重要的模型权重或连接来减小模型规模，从而实现模型压缩，同时尽可能保持模型的性能。根据实现方式不同，剪枝可分为**非结构化剪枝**和**结构化剪枝**。

* **非结构化剪枝**：不受限于特定模式，可随意丢弃网络中任意位置的权重或连接，因此破坏了原始网络的结构规律。由于这种方式产生的稀疏模式难以适配现代硬件架构，通常无法有效提升实际推理效率。

* **结构化剪枝**：通过裁剪整个结构 (如卷积核、通道或层) 来保持网络的结构性，使剪枝后的网络仍然适用于现有硬件优化的密集矩阵计算，从而显著提升实际推理性能。我们在本文中特别关注结构化剪枝，以实现 Transformer 模型中的高效稀疏结构。

网络剪枝的典型工作流程包括以下三个步骤：

1. 训练一个完整的密集网络，直至收敛；
2. 对网络进行剪枝，移除冗余的结构或权重；
3. (可选) 进一步微调网络，以恢复剪枝后模型的性能。

### 彩票假设

剪枝的理论基础之一是**彩票假设 (Lottery Ticket Hypothesis, LTH)** ([Frankle & Carbin, 2019](https://arxiv.org/abs/1803.03635))。该假设认为，一个随机初始化的密集神经网络中蕴含着某些稀疏子网络 (即“中奖彩票”)，这些子网络在单独训练时即可达到与完整网络接近甚至更好的性能。

LTH 的核心观点是，并非所有参数都同等重要。模型中只有一小部分参数真正发挥了关键作用，这表明大量参数并非用于解决过拟合问题，而主要是提供了充足的初始化搜索空间，使得高性能子网络能够被发现。

为了验证这一假设，Frankle 与 Carbin 提出了以下实验步骤：

1. 随机初始化一个密集的神经网络，初始权重记为 $\theta_0$；
2. 完整训练该网络，使参数达到良好性能，记为 $\theta$；
3. 对训练后的参数 $\theta$ 进行剪枝，生成稀疏掩码 $m$；
4. 选取“中奖彩票”子网络，初始参数定义为 $m \odot \theta_0$。

实验发现，仅使用步骤 1 中选定的少量“中奖彩票”参数，并保持它们原始的随机初始化值进行训练，模型仍可达到几乎与原网络相同的准确率。

这一结果表明，庞大的初始参数空间并非在最终模型部署时是必需的，而是在训练阶段提供了大量初始可能性，使得网络可以发掘出性能优异的稀疏结构。这也解释了为什么虽然经过剪枝的模型规模显著减小，但直接从零开始训练相同的稀疏结构却难以成功。

### 剪枝策略

**幅度剪枝 (Magnitude pruning)** 是最简单但相当有效的剪枝方法——绝对值最小的权重被修剪掉。事实上，一些研究 ([Gale et al. 2019](https://arxiv.org/abs/1902.09574)) 发现，简单的幅度剪枝方法可以达到与复杂剪枝方法 (如变分丢弃 [Molchanov et al. 2017](https://arxiv.org/abs/1701.05369) 和 $l_0$ 正则化 [Louizos et al. 2017](https://arxiv.org/abs/1712.01312)) 相当或更好的结果。幅度剪枝易于应用于大型模型，并在广泛的超参数范围内实现相当一致的性能。

**渐进幅度剪枝 (Gradual Magnitude Pruning, GMP)** ([Zhu & Gupta, 2017](https://arxiv.org/abs/1710.01878)) 方法基于大型稀疏模型能够比小型但密集的模型取得更好的性能，提出了在训练过程中逐渐增加网络的稀疏度。在每个训练步骤，绝对值最小的权重被掩码为零以达到期望的稀疏度水平 $s$，并且被掩码的权重在反向传播期间不接收梯度更新。期望的稀疏度水平 $s$ 随着训练步骤的增加而增加。GMP 的过程对学习率调度敏感，学习率应该高于密集网络训练中使用的学习率，但又不能太高以至于无法收敛。

**迭代剪枝 (Iterative pruning)** ([Renda et al. 2020](https://arxiv.org/abs/2003.02389)) 则是多次迭代步骤 2 (剪枝) 和步骤 3 (重训练)：在每次迭代中只剪枝一小部分权重，然后重新训练模型。这个过程重复进行，直到达到期望的稀疏度水平。

### 重训练

重训练步骤可以是简单的微调，使用相同的预训练数据或其他任务特定的数据集。

**彩票假设** 提出了一种 **权重回溯 (weight rewinding)** 的重训练技术：剪枝后，未剪枝的权重被重新初始化回训练早期的原始值，然后使用相同的学习率调度进行重训练。

**学习率回溯** ([Renda et al. 2020](https://arxiv.org/abs/2003.02389)) 只将学习率重置回其早期值，而未剪枝的权重自上一个训练阶段结束以来保持不变。他们观察到 (1) 在各种网络和数据集上，使用权重回卷的重训练优于使用微调的重训练；(2) 在所有测试场景中，学习率回卷与权重回卷相当或更优。

## 稀疏性

稀疏性是扩展模型容量同时保持模型推理计算效率的有效方法。这里我们考虑两种用于 Transformer 的稀疏性类型：

*   稀疏化的密集层，包括自注意力和 FFN 层。
*   稀疏模型架构；即通过引入专家混合 (Mixture-of-Experts, MoE) 组件。

### 通过剪枝实现 N:M 稀疏性

**N:M 稀疏性**是一种结构化的稀疏模式，与现代 GPU 硬件优化配合良好，其中每 $M$ 个连续元素中有 $N$ 个为零。例如，[Nvidia A100](https://www.nvidia.com/en-us/data-center/a100/) 的稀疏张量核心支持 2:4 稀疏性以实现更快的推理。

{{< figure
    src="sparsity.png"
    caption="Fig. 20. The illustration of achieving N:M structure sparsity. (Image source: [Zhou et al. 2021](https://arxiv.org/abs/2102.04010))"
    align="center"
    width="100%"
>}}

为了使密集神经网络稀疏化以遵循 N:M 结构化稀疏模式，Nvidia 建议使用三步常规工作流程来训练剪枝网络：训练 -> 剪枝以满足 2:4 稀疏性 -> 重训练。

通过置换可以在剪枝过程中提供更多选择，以保持大幅度参数或满足像 N:M 稀疏性这样的特殊限制。只要两个矩阵的配对轴以相同的顺序进行置换，矩阵乘法的结果就不会改变。例如：

(1) 在自注意力模块内，如果对查询嵌入矩阵 $\mathbf{Q}$ 的轴 1 和键嵌入矩阵 $\mathbf{K}^\top$ 的轴 0 应用相同的置换顺序，$\mathbf{Q}\mathbf{K}^\top$ 的最终矩阵乘法结果将保持不变。

{{< figure
    src="permutation_attention.png"
    caption="Fig. 21. Illustration of same permutation on $\mathbf{Q}$ (axis 1) and $\mathbf{K}^\top$ (axis 0) to keep the results of a self-attention module unchanged. (Image source: [Lilian, 2023](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))"
    align="center"
    width="90%"
>}}

(2) 在包含两个 MLP 层和一个 ReLU 非线性层的 FFN 层内，我们可以以相同的顺序置换第一个线性权重矩阵 $\mathbf{W}_1$ 的轴 1 和第二个线性权重矩阵 $\mathbf{W}_2$ 的轴 0。

{{< figure
    src="permutation_ffn.png"
    caption="Fig. 22. Illustration of the same permutation on $\mathbf{W}_1$ (axis 1) and $\mathbf{W}_2$ (axis 0) to keep the FFN layer's output unchanged. For simplicity, the bias terms are skipped but the same permutation should be applied on them too. (Image source: [Lilian, 2023](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))"
    align="center"
    width="100%"
>}}

为了强制实现 N:M 结构化稀疏性，我们将一个矩阵的列分成多个 $M$ 列的片段 (称为“条带”)，我们可以很容易地观察到，每个条带内的列顺序和条带的顺序对 N:M 稀疏性限制没有影响。

### Channel Permutations
**Channel Permutations** ([Pool & Yu, 2021](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html)) 采用了迭代贪心算法来寻找最优置换，以最大化 N:M 稀疏性的权重幅度。所有通道对都被推测性地交换，只有导致幅度最大增加的交换被采纳，从而生成一个新的置换并结束单次迭代。贪心算法可能只找到局部最优解，因此他们引入了两种技术来跳出局部最优：

1.  **有界回归 (Bounded regressions)**：在实践中，随机交换两个通道，最多固定次数。解决方案搜索被限制在只有一个通道交换的深度，以保持搜索空间的广度和浅度。
2.  **窄而深的搜索 (Narrow, deep search)**：选择多个条带并同时优化它们。


{{< figure
    src="greedy_permulation_search.png"
    caption="Fig. 23. Algorithm of finding the best permutation for N:M sparsity greedily and iteratively. (Image source: [Pool & Yu, 2021](https://proceedings.neurips.cc/paper/2021/hash/1415fe9c182320c8d536c9493793334d-Abstract.html))"
    align="center"
>}}

如果网络在剪枝前进行了置换，相比于在其默认通道顺序下剪枝，可以获得更好的性能。


### STE & SR-STE

**STE (Straight-Through Estimator)** ([Bengio et al. 2013](https://arxiv.org/abs/1308.3432)) 计算密集参数相对于剪枝后网络 $\widetilde{W}$ 的梯度 $\partial\mathcal{L}/\partial\widetilde{W}$，并将其应用于密集网络 $W$ 作为近似：
$$
W_{t+1} \leftarrow W_t - \gamma \frac{\partial\mathcal{L}}{\partial\widetilde{W}}
$$

**SR-STE (Sparse-refined STE)** ([Zhou et al. 2021](https://arxiv.org/abs/2102.04010)) 对 STE 方法进行了扩展, 采用从头开始训练一个具有 N:M 稀疏性的模型，它通常用于模型量化中的反向传播更新，以适用于幅度剪枝和稀疏参数更新。通过以下方式更新密集权重 $W$：
$$
W_{t+1} \leftarrow W_t - \gamma \frac{\partial\mathcal{L}}{\partial\widetilde{W}} + \lambda_W(\overline{\mathcal{E}} \odot W_t)
$$
其中 $\overline{\mathcal{E}}$ 是 $\widetilde{W}$ 的掩码矩阵，$\odot$ 是逐元素乘法。SR-STE 旨在通过 (1) 限制在 $\widetilde{W}_t$ 中被剪枝的权重的值，和 (2) 提升在 $\widetilde{W}_t$ 中未被剪枝的权重，来防止二元掩码发生大的变化。

{{< figure
    src="sr-ste.png"
    caption="Fig. 24. Comparison of STE and SR-STE. $\odot$ is element-wise product; $\otimes$ is matrix multiplication. (Image source: [Zhou et al. 2021](https://arxiv.org/abs/2102.04010))"
    align="center"
    width="100%"
>}}

### Top-KAST

**Top-KAST (Top-K Always Sparse Training)** ([Jayakumar et al. 2021](https://arxiv.org/abs/2106.03517)) 方法与 STE 或 SR-STE 不同，可以在前向和后向传播中都保持恒定的稀疏性，而不需要使用密集参数或密集梯度进行前向传播。

在一个训练步骤 $t$，Top-KAST 处理如下：

1.  **稀疏前向传播**：选择一个参数子集 $A^t \subset \Theta$，包含每层按幅度排序的前 $K$ 个参数，限制为权重的前 $D$ 比例。在时间 $t$ 的参数化 $\alpha^t$ 中，如果参数不在 $A^t$ (活动权重) 中，则其值为零。
    $$
    \alpha_i^t = \begin{cases} \theta_i^t & \text{if } i \in A^t = \{i \mid \theta_i^t \in \text{TopK}(\theta^t, D)\} \\ 0 & \text{otherwise} \end{cases}
    $$
    其中 $\text{TopK}(\theta, x)$ 从 $\theta$ 中根据幅度选择前 $x$ 比例的权重。

2.  **稀疏后向传播**：然后将梯度应用于一个更大的参数子集 $B \subset \Theta$，其中 $B$ 包含 $(D+M)$ 比例的权重且 $A \subset B$。更新更大比例的权重可以更有效地探索不同的剪枝掩码，使其更有可能在顶部的 $D$ 比例活动权重中引起置换。
    $$
    \Delta\theta_i^t = \begin{cases} -\eta \nabla_{\alpha_t} \mathcal{L}(y, x, \alpha^t)_i & \text{if } i \in B^t = \{i \mid \theta_i^t \in \text{TopK}(\theta^t, D+M)\} \\ 0 & \text{otherwise} \end{cases}
    $$

训练分为两个阶段，集合 $B \setminus A$ 中的附加坐标控制了引入多少探索。探索量预计会随着训练过程逐渐减少，掩码最终会稳定下来。

{{< figure
    src="top_kast.png"
    caption="Fig. 25. The pruning mask of Top-KAST stabilizes in time. (Image source: [Jayakumar et al. 2021](https://proceedings.neurips.cc/paper/2020/hash/47d1e990583c9c67424d369f3414728e-Abstract.html))"
    align="center"
    width="100%"
>}}

为了防止“富者愈富”现象，Top-KAST 通过 L2 正则化损失惩罚活动权重的大小，以鼓励探索新项目。在 $B \setminus A$ 中的参数比 $A$ 中的参数受到更多的惩罚，以便在更新期间为稳定掩码设置更高的选择门槛。
$$
L_{\text{penalty}}(\alpha_i^t) = \begin{cases} |\theta_i^t| & \text{if } i \in A^t \\ |\theta_i^t|/D & \text{if } i \in B^t \setminus A^t \\ 0 & \text{otherwise} \end{cases}
$$

### 稀疏化 Transformer

**稀疏化 Transformer** ([Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763)) 在 Transformer 架构中稀疏化了自注意力和 FFN 层，实现了单样本推理 37 倍的加速。

{{< figure
    src="sparsified_transformer_speed.png"
    caption="Fig. 26. Decoding speed of a single token for Terraformer with 17B parameters is 37x faster than a dense baseline model, (Image source: [Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763))"
    align="center"
    width="70%"
>}}

**稀疏 FFN 层**：每个 FFN 层包含 2 个 MLP 和一个 ReLU。因为 ReLU 会引入大量零值，他们对激活值实施了一个固定结构，强制在一个 $N$ 个元素的块中只有一个非零值。稀疏模式是动态的，对每个 Token 都不同。
$$
\begin{aligned}
Y_{\text{sparse}} &= \max(0, xW_1 + b_1) \odot \text{Controller}(x) \\
\text{SparseFFN}(x) &= Y_{\text{sparse}} W_2 + b_2 \\
\text{Controller}(x) &= \arg\max(\text{Reshape}(xC_1C_2, (-1, N)))
\end{aligned}
$$
其中 $Y_{\text{sparse}}$ 中的每个激活对应于 $W_1$ 中的一列和 $W_2$ 中的一行。控制器是一个低秩瓶颈密集层，$C_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{lowrank}}}, C_2 \in \mathbb{R}^{d_{\text{lowrank}} \times d_{\text{ff}}}$ 且 $d_{\text{lowrank}} = d_{\text{model}}/N$。它在推理时使用 `arg max` 来选择哪些列应该为非零，在训练时使用 Gumbel-softmax 技巧。因为我们可以在加载 FFN 权重矩阵之前计算 $\text{Controller}(x)$，我们知道哪些列将被置零，因此选择**不将它们加载到内存中**以加速推理。

{{< figure
    src="sparse_ffn.png"
    caption="Fig. 27. (a) Sparse FFN layer; columns in red are not loaded in memory for faster inference. (b) Sparse FFN controller for 1:4 sparsity. (Image source: [Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763))"
    align="center"
    width="100%"
>}}

**稀疏 QKV (注意力) 层**：在注意力层中，维度 $d_{\text{model}}$ 被分成 $S$ 个模块，每个模块的大小为 $M = d_{\text{model}}/S$。为了确保每个子部分可以访问嵌入的任何部分，Scaling Transformer 引入了一个**乘法层** (即，一个将来自多个神经网络层的输入逐元素相乘的层)，它可以表示任意置换，但比密集层包含更少的参数。

给定一个输入向量 $x \in \mathbb{R}^{d_{\text{model}}}$，乘法层输出 $y \in \mathbb{R}^{S \times M}$：
$$
y_{s,m} = \sum_i x_i D_{i,s} E_{i,m} \quad \text{where } D \in \mathbb{R}^{d_{\text{model}} \times S}, E \in \mathbb{R}^{d_{\text{model}} \times M}
$$
乘法层的输出是一个大小为 $\mathbb{R}^{\text{batch size} \times \text{length} \times S \times M}$ 的张量。然后它由一个二维卷积层处理，其中 `length` 和 $S$ 被视为图像的高度和宽度。这样的卷积层进一步减少了注意力层的参数数量和计算时间。

{{< figure
    src="sparse_qkv.png"
    caption="Fig. 28. (a) A multiplicative layer is introduced to enable partitions to access any part of an embedding. (b) Combination of multiplicative dense layer and 2-D convolutional layer reduces the number of parameters and computation time of the attention layer. (Image source: [Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763))"
    align="center"
    width="100%"
>}}

为了更好地处理长序列，Scaling Transformer 进一步配备了来自 **Reformer** ([Kitaev, et al. 2020](https://arxiv.org/abs/2001.04451)) 的 LSH (局部敏感哈希) 注意力和 FFN 块循环。


### 混合专家

[**混合专家 (Mixture-of-Experts, MoE)**](https://syhya.github.io/zh/posts/2025-04-18-deepseek-v2-v3/#%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6%E6%A8%A1%E5%9E%8B) 模型由多个“专家”网络组成，每个输入样本只激活部分专家网络来进行计算。

{{< figure
    src="dense_sparse_model.png"
    caption="Fig. 29. Dense Transformer vs Sparse Expert Transformer. (Image source: [Fedus et al. 2022](https://arxiv.org/abs/2209.01667))"
    align="center"
    width="100%"
>}}

* **Dense Model**：所有输入令牌 (token) 使用相同的前馈网络 (FFN) 参数进行处理。虽然结构简单易于训练，但随着模型规模的增长，其计算成本也快速增加。

* **Sparse Expert Model**：每个输入令牌独立地路由到多个专家网络中的少数专家进行处理。这种稀疏路由机制使得模型可以拥有更多独特的参数，同时整体计算成本不显著增加，因而提高了参数效率和扩展能力，有效降低了推理阶段的计算成本。


#### 路由策略改进

MoE 层有一个路由网络，为每个输入 Token 分配一个专家子集。在传统的 MoE 模型中，路由策略是按照 Token 在自然顺序中出现的顺序，将每个 Token 路由到其偏好的专家。如果一个 Token 被路由到已经达到容量的专家，该 Token 将被标记为“溢出”并被跳过。

**Vision MoE (V-MoE)** ([Riquelme et al. 2021](https://arxiv.org/abs/2106.05974)) 将 MoE 层添加到 ViT (Vision Transformer) 中。它达到了之前 SOTA 的性能，但只需要一半的推理计算量。V-MoE 可以扩展到 15B 参数。他们的实验使用了 $k=2$，32 个专家，并且每隔 2 层放置一个专家 (意味着 MoE 被放置在每隔一层)。

由于每个专家的容量有限，一些重要和信息丰富的 Token 如果出现得太晚 (例如，句子中单词的顺序，或图像块的顺序)，可能会被丢弃。为了避免传统路由方案的这种缺点，V-MoE 采用了**批量优先路由 (BPR, Batch Priority Routing)**，首先为具有高优先级的 Token 分配专家。BPR 在专家分配前为每个 Token 计算一个优先级分数 (top-k 路由器分数的最大值或总和)，并相应地改变 Token 的顺序。这保证了专家容量缓冲区将首先被关键 Token 填充。

{{< figure
    src="v_moe_bpr.png"
    caption="Fig. 30. How image patches are discarded according to priority scores when $C < 1$. (Image source: [Riquelme et al. 2021](https://arxiv.org/abs/2106.05974))"
    align="center"
    width="100%"
>}}

当 $C \le 0.5$ 时，BPR 比传统路由效果好得多，此时模型开始丢弃大量 Token。它使模型即使在相当低的容量下也能与密集网络竞争。

在研究如何解释图像类别与专家的关联时，他们观察到早期的 MoE 层更通用，而后期的 MoE 层可能专门用于少数几个图像类别。

**Task MoE (Task-level Mixture-of-Experts)** ([Kudugunta et al. 2021](https://arxiv.org/abs/2110.03742)) 考虑了任务信息，并在机器翻译中以**任务级别**而不是单词或 Token 级别路由 Token。他们使用 MNMT (多语言神经机器翻译) 作为例子，并根据目标语言或语言对对翻译任务进行分组。

Token 级别的路由是动态的，每个 Token 的路由决策是独立做出的。因此，在推理时，服务器需要预加载所有专家。相比之下，任务级别的路由在给定固定任务时是**静态**的，因此一个任务的推理服务器只需要预加载 $k$ 个专家 (假设是 top-k 路由)。根据他们的实验，与密集模型基线相比，Task MoE 可以实现与 Token MoE 相似的性能增益，峰值吞吐量高 2.6 倍，解码器大小仅为 1.6%。

任务级别的 MoE 本质上是根据预定义的启发式规则对任务分布进行分类，并将这些人类知识融入到路由器中。当这种启发式规则不存在时 (例如，考虑一个通用的句子续写任务)，如何利用 Task MoE 就不那么直接了。

**PR-MoE (Pyramid residual MoE)** ([Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596)) 让每个 Token 通过一个固定的 MLP 和一个选择的专家。由于观察到 MoE 在后期层更有益，PR-MoE 在后期层采用更多的专家。DeepSpeed 库实现了一个灵活的多专家、多数据并行，以支持训练具有不同专家数量的 PR-MoE。

{{< figure
    src="pr_moe.png"
    caption="Fig. 31. Illustration of PR-MoE architecture in comparison with a standard MoE. (Image source: [Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596))"
    align="center"
    width="100%"
>}}

#### 内核改进

专家网络可以托管在不同的设备上。然而，当 GPU 数量增加时，每个 GPU 的专家数量减少，专家之间的通信 (“All-to-all”) 变得更加昂贵。跨多个 GPU 的专家之间的 All-to-all 通信依赖于 NCCL 的 P2P API，在大规模下无法饱和高速链路 (如 NVLink, HDR InfiniBand) 的带宽，因为随着使用更多节点，单个数据块变得更小。现有的 all-to-all 算法在小工作负载的大规模场景下表现不佳。有多种内核改进可以实现更高效的 MoE 计算，例如使 all-to-all 通信更便宜/更快。

**DeepSpeed** ([Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596)) 和 **TUTEL** ([Hwang et al. 2022](https://arxiv.org/abs/2206.03382)) 都实现了一种基于树的**分层 all-to-all 算法**，该算法先运行一个节点内 all-to-all，然后是一个节点间 all-to-all。它将通信跳数从 $O(G)$ 减少到 $O(G_{\text{node}} + G/G_{\text{node}})$，其中 $G$ 是 GPU 节点的总数，$G_{\text{node}}$ 是每个节点的 GPU 核心数。尽管在这种实现中通信量增加了一倍，但它在小批量大规模场景下实现了更好的扩展性，因为瓶颈在于延迟而不是通信带宽。

**DynaMoE** ([Kossmann et al. 2022](https://arxiv.org/abs/2205.01848)) 使用**动态重编译**来使计算资源适应专家之间的动态工作负载。`RECOMPILE` 机制从头开始编译计算图，并且只在需要时重新分配资源。它测量分配给每个专家的样本数量，并动态调整它们的容量因子 $C$，以减少运行时的内存和计算需求。基于样本-专家分配在训练早期收敛的观察，收敛后引入**样本分配缓存**，然后使用 `RECOMPILE` 来消除门控网络和专家之间的依赖关系。

## 架构优化

### Efficient Transformers

**Efficient Transformers** ([Tay et al. 2020](https://arxiv.org/abs/2009.06732)) 的综述论文回顾了一系列 Transformer 架构，这些架构在计算和内存效率方面有所改进。对此感兴趣的读者可以阅读原文。

{{< figure
    src="efficient_transformers.png"
    caption="Fig. 32. Categorization of efficient transformer models. (Image source: [Tay et al. 2020](https://arxiv.org/abs/2009.06732))"
    align="center"
    width="100%"
>}}

### KV Cache 优化

*   **Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)**：标准的多头注意力 (Multi-Head Attention, MHA) 中，每个头都有一套独立的 Key 和 Value 投影矩阵。MQA ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) 提出让所有的查询头 (Query heads) 共享同一套 Key 和 Value 头，极大地减小了 KV Cache 的体积。GQA ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) 则是 MHA 和 MQA 的折中，它将查询头分组，组内的头共享一套 K/V，在性能和效果上取得了很好的平衡。

*   **vLLM**([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)) 借鉴了操作系统中虚拟内存和分页的思想，提出了 **PagedAttention**。它将 KV Cache 分割成固定大小的块 (Block)，这些块在物理显存中可以不连续存储，通过一个“块表”来管理逻辑块到物理块的映射。关于 vLLM 的详细介绍可以参考我之前的博客[vLLM：高吞吐、有效内存的LLM服务引擎](https://syhya.github.io/zh/posts/2025-05-17-vllm/)。这种方法几乎完全消除了内存碎片 (内部和外部)，使得显存利用率接近 100%。更重要的是，它通过写时复制 (Copy-on-Write) 机制，可以非常高效地实现跨请求的 KV Cache 共享，极大地提升了并行采样、Beam Search 等复杂解码场景下的吞吐量。

### FlashAttention

*   **FlashAttention**([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) 是一种 IO 感知的精确注意力算法。它认识到标准 Attention 实现的主要瓶颈在于 GPU HBM (高带宽内存) 和 SRAM (片上高速缓存) 之间的数据读写。FlashAttention 通过 **Tiling (分块)** 和 **Recomputation (重计算)** 技术，将整个 Attention 计算融合到一个 CUDA 核中，避免了将巨大的 $N \times N$ 注意力矩阵写入和读出 HBM。这极大地减少了内存访问量，从而在不牺牲精度的情况下，将 Attention 的计算速度提升了数倍。**FlashAttention-2**([Dao, 2023](https://arxiv.org/abs/2307.08691)) 进一步优化了并行度和硬件利用率。

{{< figure
    src="flash_attention.png"
    caption="Fig. 33. FlashAttention uses tiling to avoid materializing the large N × N attention matrix on slow GPU HBM, achieving up to 7.6× speedup over PyTorch. (Image source: [Dao et al., 2022](https://arxiv.org/abs/2205.14135))"
    align="center"
    width="100%"
>}}


### 参考文献

[1] Zhou, Zixuan, et al. [“A survey on efficient inference for large language models.”](https://arxiv.org/abs/2404.14294) arXiv preprint arXiv:2404.14294 (2024).

[2] Zhang, Aston, et al. [“Dive into Deep Learning.”](https://d2l.ai/). Cambridge University Press, 2023.

[3] Big Hummingbird Blogs. (2024). [“A Visual Explanation of LLM Hyperparameters.”](https://www.bighummingbird.com/blogs/llm-hyperparameter) Blog post.

[4] Fan, Angela, Mike Lewis, and Yann Dauphin. [“Hierarchical neural story generation.”](https://arxiv.org/abs/1805.04833) arXiv preprint arXiv:1805.04833 (2018).
int arXiv:1805.04832.

[5] Holtzman, Ari, et al. [“The curious case of neural text degeneration.”](https://arxiv.org/abs/1904.09751) arXiv preprint arXiv:1904.09751 (2019).

[6] Leviathan, Yaniv, Matan Kalman, and Yossi Matias. [“Fast inference from transformers via speculative decoding.”](https://arxiv.org/abs/2211.17192) International Conference on Machine Learning. PMLR, 2023.

[7] Liu, Xiaoxuan, et al. [“Online speculative decoding.”](https://arxiv.org/abs/2310.07177) arXiv preprint arXiv:2310.07177 (2023).

[8] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. [“Distilling the knowledge in a neural network.”](https://arxiv.org/abs/1503.02531) arXiv preprint arXiv:1503.02531 (2015).

[9] Gou, Jianping, et al. [“Knowledge distillation: A survey.”](https://arxiv.org/abs/2006.05525) International Journal of Computer Vision 129.6 (2021): 1789-1819.

[10] Sanh, Victor, et al. [“DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.”](https://arxiv.org/abs/1910.01108) arXiv preprint arXiv:1910.01108 (2019).

[11] Raschka, S. (2023). [“Accelerating Large Language Models with Mixed-Precision Techniques.”](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html) Blog post.

[12] Bondarenko, Yelysei, Markus Nagel, and Tijmen Blankevoort. ["Understanding and overcoming the challenges of efficient transformer quantization."](https://arxiv.org/abs/2109.12948) arXiv preprint arXiv:2109.12948 (2021).

[13] Zhang, S., Roller, S., Goyal, N., et al. (2022). [“OPT: Open pre-trained transformer language models.”](https://arxiv.org/abs/2205.01068) arXiv preprint arXiv:2205.01068.

[14] Dettmers, T., et al. (2022). [“LLM.int8(): 8-bit matrix multiplication for transformers at scale.”](https://arxiv.org/abs/2208.07339) arXiv preprint arXiv:2208.07339.

[15] Zadeh, V. K., et al. (2020). [“GOBO: Quantizing attention-based NLP models for low latency and energy efficient inference.”](https://arxiv.org/abs/2005.03842) arXiv preprint arXiv:2005.03842.

[16] Weng, L. (2023). [“Large Transformer Model Inference Optimization.”](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) Lil’Log blog post.

[17] Shen, Z., Dong, Z., Ye, J., et al. (2020). [“Q-BERT: Hessian-based ultra-low-precision quantization of BERT.”](https://arxiv.org/abs/1909.05840) arXiv preprint arXiv:1909.05840.

[18] Dong, Z., Yao, Z., Gholami, A., et al. (2019). [“HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision”](https://arxiv.org/abs/1905.03696) arXiv preprint arXiv:1905.03696.

[19] Yao, Z., et al. (2022). [“ZeroQuant: Efficient and affordable post-training quantization for large-scale transformers.”](https://arxiv.org/abs/2206.01861) arXiv preprint arXiv:2206.01861.

[20] Frantar, E., et al. (2022). [“GPTQ: Accurate post-training quantization for generative pre-trained transformers.”](https://arxiv.org/abs/2210.17323) arXiv preprint arXiv:2210.17323.

[21] Xiao, G., & Lin, J. (2022). [“SmoothQuant: Accurate and efficient post-training quantization for large language models.”](https://arxiv.org/abs/2211.10438) arXiv preprint arXiv:2211.10438.

[22] Frankle, J., & Carbin, M. (2019). [“The lottery ticket hypothesis: Finding sparse, trainable neural networks.”](https://arxiv.org/abs/1803.03635) arXiv preprint arXiv:1803.03635.

[23] Gale, T., Elsen, E., & Hooker, S. (2019). [“The state of sparsity in deep neural networks.”](https://arxiv.org/abs/1902.09574) arXiv preprint arXiv:1902.09574.

[24] Molchanov, D., Ashukha, A., & Vetrov, D. (2017). [“Variational dropout sparsifies deep neural networks.”](https://arxiv.org/abs/1701.05369) arXiv preprint arXiv:1701.05369.

[25] Louizos, Christos, Max Welling, and Diederik P. Kingma. ["Learning sparse neural networks through $ L_0 $ regularization."](https://arxiv.org/abs/1712.01312) arXiv preprint arXiv:1712.01312 (2017).

[26] Zhu, M., & Gupta, S. (2017). [“To prune, or not to prune: Exploring the efficacy of pruning for model compression.”](https://arxiv.org/abs/1710.01878) arXiv preprint arXiv:1710.01878.

[27] Renda, A., Frankle, J., & Carbin, M. (2020). [“Comparing rewinding and fine-tuning in neural network pruning.”](https://arxiv.org/abs/2003.02389) arXiv preprint arXiv:2003.02389.

[28] Nvidia. (2020). [“NVIDIA A100 Tensor Core GPU.”](https://www.nvidia.com/en-us/data-center/a100/) Nvidia Blog.

[29] Zhou, A., & Ma, X. (2021). [“Learning N:M fine-grained structured sparse neural networks from scratch.”](https://arxiv.org/abs/2102.04010) arXiv preprint arXiv:2102.04010.

[30] Pool, J., & Yu, F. (2021). [“Channel permutations for N:M structured sparsity.”](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) Advances in Neural Information Processing Systems 34.

[31] Bengio, Y., Léonard, N., & Courville, A. (2013). [“Estimating or propagating gradients through stochastic neurons for conditional computation.”](https://arxiv.org/abs/1308.3432) arXiv preprint arXiv:1308.3432.

[32] Jayakumar, S. M., Pascanu, R., Rae, J., et al. (2021). [“Top-KAST: Top-K always sparse training.”](https://arxiv.org/abs/2106.03517) arXiv preprint arXiv:2106.03517.

[33] Jaszczur, S., et al. (2021). [“Sparse is enough in scaling transformers.”](https://arxiv.org/abs/2111.12763) Advances in Neural Information Processing Systems 34.

[34] Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). [“Reformer: The efficient transformer.”](https://arxiv.org/abs/2001.04451) arXiv preprint arXiv:2001.04451.

[35] Fedus, W., et al. (2022). [“A review of sparse expert models in deep learning.”](https://arxiv.org/abs/2209.01667) arXiv preprint arXiv:2209.01667.

[36] Riquelme, C., et al. (2021). [“Scaling vision with sparse mixture of experts.”](https://arxiv.org/abs/2106.05974) Advances in Neural Information Processing Systems 34: 8583-8595.

[37] Kudugunta, S., Lepikhin, D., Heafield, K., et al. (2021). [“Beyond domain adaptation: Multi-task mixture-of-experts for zero-shot generalization.”](https://arxiv.org/abs/2110.03742) arXiv preprint arXiv:2110.03742.

[38] Rajbhandari, S., et al. (2022). [“DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale.”](https://arxiv.org/abs/2201.05596) arXiv preprint arXiv:2201.05596.

[39] Hwang, I., et al. (2022). [“Tutel: Adaptive mixture-of-experts at scale.”](https://arxiv.org/abs/2206.03382) arXiv preprint arXiv:2206.03382.

[40] Kossmann, F., et al. (2022). [“Optimizing Mixture of Experts using Dynamic Recompilations”](https://arxiv.org/abs/2205.01848) arXiv preprint arXiv:2205.01848.

[41] Tay, Y., et al. (2020). [“Efficient transformers: A survey.”](https://arxiv.org/abs/2009.06732) arXiv preprint arXiv:2009.06732.

[42] Shazeer, N. (2019). [“Fast transformer decoding: One write-head is all you need.”](https://arxiv.org/abs/1911.02150) arXiv preprint arXiv:1911.02150.

[43] Ainslie, J., et al. (2023). [“GQA: Training generalized multi-query transformer models from multi-head checkpoints.”](https://arxiv.org/abs/2305.13245) arXiv preprint arXiv:2305.13245.

[44] Kwon, W., et al. (2023). [“Efficient memory management for large language model serving with PagedAttention.”](https://arxiv.org/abs/2309.06180) Proceedings of the 29th Symposium on Operating Systems Principles.

[45] Dao, T., et al. (2022). [“FlashAttention: Fast and memory-efficient exact attention with IO-awareness.”](https://arxiv.org/abs/2205.14135) Advances in Neural Information Processing Systems 35: 16344-16359.

[46] Dao, T. (2023). [“FlashAttention-2: Faster attention with better parallelism and work partitioning.”](https://arxiv.org/abs/2307.08691) arXiv preprint arXiv:2307.08691.

[47] Pope, R., et al. (2022). [“Efficiently scaling transformer inference.”](https://arxiv.org/abs/2211.05102) arXiv preprint arXiv:2211.05102.

[48] von Platen, P. (2020). [“How to generate text: Using different decoding methods for language generation with Transformers.”](https://huggingface.co/blog/how-to-generate) Hugging Face Blog.


## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Jun 2025). 大语言模型推理.
https://syhya.github.io/zh/posts/2025-06-29-llm-inference

Or

```bibtex
@article{syhya2025llminferencesurvey,
  title   = "大语言模型推理",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jun",
  url     = "https://syhya.github.io/zh/posts/2025-06-29-llm-inference"
}
```