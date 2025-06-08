---
title: "Transformer注意力机制：MHA、MQA与GQA的对比"
date: 2025-01-16T12:00:00+08:00
author: "Yue Shui"
tags: ["深度学习", "AI", "Transformer", "注意力机制", "MHA", "MQA", "GQA", "KV Cache", "NLP", "LLM"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## 背景

Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)）是一种基于编码器-解码器架构的模型。此模型在自然处理领域中展示了卓越的性能，随后一系列模型在此基础上进行了优化，例如仅使用编码器的 BERT ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)）或仅使用解码器的 GPT ([Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)）系列，以及后续的大型语言模型如 LLaMA ([Touvron et al., 2023](https://arxiv.org/abs/2302.13971)）和 GPT-4 ([OpenAI al., 2024](https://arxiv.org/abs/2303.08774)）系列，这些模型大多采用了仅解码器的结构。

## 符号
| 符号                                                         | 含义                                                                                                                         |
|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| \(B\)                                                        | 批量大小（Batch Size）                                                                                                       |
| \(S\)                                                        | 序列长度（Sequence Length）                                                                                                  |
| \(d\)                                                        | 隐藏维度 / 模型维度（Model Size / Hidden Dimension）                                                                         |
| \(H\)                                                        | 注意力头数量（Number of Heads in Multi-Head Attention）                                                                      |
| \(G\)                                                        | 分组数量（Group Number），用于分组查询注意力（GQA）                                                                          |
| \(d_{\text{head}} = \frac{d}{H}\)                            | 每个注意力头的维度                                                                                                           |
| \(\mathbf{X} \in \mathbb{R}^{B \times S \times d}\)          | 输入序列，批量为 \(B\)，序列长度为 \(S\)，隐藏维度为 \(d\)                                                                   |
| \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times S \times d}\) | 经过线性变换后的 Query、Key、Value 矩阵                                                                                      |
| \(W_Q, W_K, W_V \in \mathbb{R}^{d \times d}\)                | 分别对应生成 \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) 的可训练线性映射矩阵                                                                 |
| \(W_O \in \mathbb{R}^{d \times d}\)                          | 多头 / 分组注意力输出后，用于映射回原始维度 \(d\) 的可训练线性映射矩阵                                                       |
| \(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h \in \mathbb{R}^{B \times S \times d_{\text{head}}}\) | 第 \(h\) 个注意力头对应的 Query、Key、Value 子矩阵                                                                           |
| \(\mathbf{K}^*, \mathbf{V}^*\)                               | 在多查询注意力（MQA）中，将所有头的 \(\mathbf{K}_h, \mathbf{V}_h\) 平均或合并后得到的共享 \(\mathbf{K}\) 和 \(\mathbf{V}\)       |
| \(\mathbf{q}, \mathbf{k}\in \mathbb{R}^{d_{\text{head}}}\)   | 在缩放点积注意力的随机向量示例中，用于数学推导（中心极限定理）的单个查询向量和单个键向量                                      |


## Transformer中的注意力机制

Transformer模型的核心在于**自注意力机制（Self-Attention）**，它允许模型在处理序列数据时，动态地关注序列中的不同部分。具体来说，给定一个输入序列 \(\mathbf{X} \in \mathbb{R}^{B \times S \times d}\)（批大小 \(B\)，序列长度 \(S\)，隐藏维度 \(d\)），Transformer会通过三个线性层分别投影为查询（Query, \(\mathbf{Q}\)）、键（Key, \(\mathbf{K}\)）和值（Value, \(\mathbf{V}\)）：

\[
\mathbf{Q} = \mathbf{X} W_Q, \quad
\mathbf{K} = \mathbf{X} W_K, \quad
\mathbf{V} = \mathbf{X} W_V
\]

其中，\(W_Q, W_K, W_V \in \mathbb{R}^{d \times d}\) 是可训练的权重矩阵。多头注意力通过将这些投影分成多个头，每个头负责不同的子空间表示，从而增强模型的表示能力。

注意力机制有多种形式，Transformer 依赖于缩放点积注意力（Scaled Dot-Product Attention）：给定查询矩阵 \(\mathbf{Q}\)、键矩阵 \(\mathbf{K}\) 和值矩阵 \(\mathbf{V}\)，输出是值向量的加权和，其中每个值的权重由查询与对应键的点积决定：

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) 
= \text{softmax}\!\Bigl(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\Bigr)\,\mathbf{V}
\]

{{< figure 
    src="scaled_dot_product_attention.png" 
    caption="Fig. 1. Scaled Dot-Product Attention. (Image source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))"
    align="center"
    width="50%"
>}}

### 多头注意力（MHA）

多头注意力（MHA）将 \(\mathbf{Q}\)、\(\mathbf{K}\)、\(\mathbf{V}\) 分成多个头，每个头有独立的 \(\mathbf{K}\) 和 \(\mathbf{V}\)，从而增加了模型的容量和灵活性：

\[
\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) 
= \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
\]

其中，每个头的计算为：

\[
\text{head}_h 
= \text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h)
= \text{softmax}\!\Bigl(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_{\text{head}}}}\Bigr)\,\mathbf{V}_h
\]

{{< figure 
    src="multi_head_attention.png" 
    caption="Fig. 2. Multi-Head Attention. (Image source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))"
    align="center"
    width="50%"
>}}

#### 使用多头注意力机制的好处
- **捕捉不同的特征**：单头注意力机制只能关注输入序列中的一种特征或模式，而多头注意力机制可以通过多个注意力头同时关注不同的特征或模式，使模型能够更全面地理解输入数据。
- **增强模型的表达能力**：每个注意力头可以学习不同的表示方式，增强模型的表达能力。不同的注意力头可以关注输入序列的不同部分或不同关系，帮助模型更好地捕捉复杂的依赖关系。
- **提高稳定性和性能**：多头注意力机制通过多个注意力头的平均或组合，减少单个注意力头的噪声和不稳定性，提高模型的稳定性和性能。
- **并行计算**：多头注意力机制可以在计算上并行化，因为每个注意力头的计算是独立的。这有助于提高计算效率，特别是在使用GPU或TPU等硬件加速器时。

#### 缩放点积注意力中的Softmax

Softmax函数将一个向量 \(\mathbf{z} = [z_1, z_2, \dots, z_n]\) 转换为一个概率分布 \(\mathbf{y} = [y_1, y_2, \dots, y_n]\)，其定义如下：

\[
y_i = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)} 
\quad \text{对于} \quad i = 1, 2, \dots, n
\]

在注意力机制中，softmax函数用于将缩放后的点积 \(\tfrac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\) 转换为注意力权重：

\[
\text{softmax}\!\Bigl(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\Bigr)
= \Bigl[ 
\frac{\exp\Bigl(\frac{Q_1 \cdot K_1}{\sqrt{d_{\text{head}}}}\Bigr)}{\sum_{j=1}^{S} \exp\Bigl(\frac{Q_1 \cdot K_j}{\sqrt{d_{\text{head}}}}\Bigr)}, 
\dots, 
\frac{\exp\Bigl(\frac{Q_S \cdot K_S}{\sqrt{d_{\text{head}}}}\Bigr)}{\sum_{j=1}^{S} \exp\Bigl(\frac{Q_S \cdot K_j}{\sqrt{d_{\text{head}}}}\Bigr)} 
\Bigr]
\]

在 Transformer 的注意力机制中，缩放点积注意力公式中的缩放因子 \(\sqrt{d_{\text{head}}}\) 是为了确保在进行 softmax 之前，点积的结果不会因为向量维度的增加而变得过大。这主要有以下几个原因：

- **防止梯度消失**：通过缩放注意力得分，可以避免输入 softmax 函数的值过大，从而防止梯度在反向传播过程中出现消失的情况。
- **数值不稳定性**：缩放注意力得分可以使得 softmax 函数的输入值范围更加合理，避免数值过于极端，从而提升模型的数值稳定性和训练效果。特别是当向量维度较大时，未经缩放的点积结果可能导致 softmax 的指数函数值过大，进而引发溢出问题。

- **数学解释**：假设向量 \(\mathbf{q}\) 和 \(\mathbf{k}\) 的各分量独立同分布，均值为 0，方差为 1。它们的点积 \(\mathbf{q} \cdot \mathbf{k}\) 的均值为 0，方差为 \(d_{\text{head}}\)。为了防止点积的方差随维度 \(d_{\text{head}}\) 增加而变大，需要对其进行缩放处理。通过将点积除以 \(\sqrt{d_{\text{head}}}\)，可以使缩放后的点积的方差为 1，与 \(d_{\text{head}}\) 无关。

根据统计学原理，当将随机变量除以一个常数时，其方差会按该常数的平方倒数缩放。因此，缩放因子 \(\tfrac{1}{\sqrt{d_{\text{head}}}}\) 可以有效控制注意力得分的规模，从而提高数值稳定性。以下是详细推导过程：

假设 \(\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_{\text{head}}}\)，各分量独立同分布，均值为 0，方差为 1，则它们的点积为：

\[
\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_{\text{head}}} q_i k_i
\]

根据中心极限定理，当 \(d_{\text{head}}\) 较大时，点积 \(\mathbf{q} \cdot \mathbf{k}\) 近似服从均值为 0、方差为 \(d_{\text{head}}\) 的正态分布：

\[
\mathbf{q} \cdot \mathbf{k} \sim \mathcal{N}(0, d_{\text{head}})
\]

为了使缩放后的点积具有单位方差，我们将点积除以 \(\sqrt{d_{\text{head}}}\)：

\[
\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_{\text{head}}}} \;\sim\; \mathcal{N}\!\Bigl(0, \frac{d_{\text{head}}}{d_{\text{head}}}\Bigr) = \mathcal{N}(0, 1)
\]

因此，经过缩放后，点积 \(\tfrac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_{\text{head}}}}\) 的方差恒为 1，与维度 \(d_{\text{head}}\) 无关。这种缩放操作能够保持点积在一个稳定的范围内，避免 softmax 函数在计算中因输入值过大或过小而产生数值不稳定性。


### 多查询注意力（MQA）

多查询注意力（MQA）([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) 通过让所有查询头（Query Heads）共享同一组键（Key）\(\mathbf{K}\) 和值（Value）\(\mathbf{V}\)，从而显著减少了显存带宽的需求。具体地，如果我们将传统多头注意力（MHA）中的所有 \(\mathbf{K}_h\) 和 \(\mathbf{V}_h\) 做如下平均：

\[
\mathbf{K}^* = \frac{1}{H} \sum_{h=1}^{H} \mathbf{K}_h,
\quad 
\mathbf{V}^* = \frac{1}{H} \sum_{h=1}^{H} \mathbf{V}_h,
\]

其中 \(H\) 表示查询头的数量，\(\mathbf{K}_h\) 和 \(\mathbf{V}_h\) 分别表示第 \(h\) 个头对应的键和值。那么在推理过程中，每个头只需要使用同一个 \(\mathbf{K}^*\) 和 \(\mathbf{V}^*\)，从而大幅降低对显存带宽的占用。最后再将所有头输出拼接并映射回输出空间：

\[
\text{MQA}(\mathbf{Q}, \mathbf{K}^*, \mathbf{V}^*) 
= \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
\]

由于键和值只保留了一组，MQA 推理速度更快，但在某些场景下，模型的表达能力和性能会受到一定限制。

### 分组查询注意力（GQA）

分组查询注意力（GQA） ([Ainslie, 2023](https://arxiv.org/pdf/2305.13245)) 是介于 MHA 和 MQA 之间的一种折中方案。它通过将查询头分为多个组，让每组共享一组 $\mathbf{K}$ 和 $\mathbf{V}$ 头，以在推理速度和模型性能之间取得平衡。每组包含 $\frac{H}{G}$ 个查询头，每组共享一组 $\mathbf{K}$ 和 $\mathbf{V}$ 头。其具体流程如下：

- **投影**：将输入 $\mathbf{X}$ 通过线性变换分别投影为 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$。
- **分组 Query**：将 $\mathbf{Q}$ 划分为 $H$ 个头后，再将这些头进一步划分为 $G$ 组。
- **分组 Key/Value**：将 $\mathbf{K}$ 和 $\mathbf{V}$ 划分为 $G$ 组，每组共享一组 $\mathbf{K}$ 和 $\mathbf{V}$。
- **组内注意力**：对每组的 $\mathbf{Q}$ 与各自组共享的 $\mathbf{K}$ 和 $\mathbf{V}$ 进行注意力计算。
- **拼接输出**：将各组的注意力结果在通道维度上拼接，最后通过线性层得到最终输出。


### 三种 Attention 方法之间的联系

{{< figure 
    src="attention_comparison.png" 
    caption="Fig. 3. Overview of grouped-query method. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))"
    align="center"
    width="100%"
>}}

图3直观展示了这三种注意力机制的关系：多头注意力（MHA）为每个查询头都保留独立的 $\mathbf{K}$ 和 $\mathbf{V}$；多查询注意力（MQA）则所有查询头共享同一组 $\mathbf{K}$ 和 $\mathbf{V}$；分组查询注意力（GQA）则在两者之间，通过分组共享的方式兼顾速度与性能。

- 当 $G=1$ 时：所有查询头共享同一组 $\mathbf{K}$ 和 $\mathbf{V}$。此时 GQA 退化为多查询注意力（MQA）。
  - **$\mathbf{K}/\mathbf{V}$ 头数量**：$1$
  - **模型行为**：所有头使用相同的 $\mathbf{K}$ 和 $\mathbf{V}$ 进行注意力计算，显著降低显存带宽需求。

- 当 $G=H$ 时：每个查询头都拥有独立的 $\mathbf{K}$ 和 $\mathbf{V}$。此时 GQA 退化为多头注意力（MHA）。
  - **$\mathbf{K}/\mathbf{V}$ 头数量**：$H$
  - **模型行为**：每个头使用完全独立的 $\mathbf{K}$ 和 $\mathbf{V}$，保留 MHA 的高模型容量和性能。

通过调整分组数量 $G$，GQA 在 MHA 和 MQA 之间实现了灵活切换，能够在保持较高模型性能的同时，兼顾推理速度的提升。

### 实现代码片段

下面是使用 PyTorch 简单实现的 **MHA** 、**MQA**和 **GQA** 的代码, 其中 GQA采用了广播（Boardcast）和复制（Repeat）两种方法。此外 需要注意的是，在实际的 LLaMA3 源代码中，GQA 的实现还引入了 KV Cache。为简化示例，以下代码并未包含该部分。如果感兴趣，可以参考源代码 [model.py](https://github.com/meta-llama/llama3/blob/main/llama/model.py) 获取更完整的代码细节。

{{< collapse summary="MHA 代码片段" openByDefault=false >}}[multi_head_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/multi_head_attention.py)
```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head

        # (nums_head * head_dim = hidden_dim)
        assert hidden_dim % nums_head == 0
        self.head_dim = hidden_dim // nums_head

        self.dropout = nn.Dropout(dropout_rate)

        # Define linear projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # x has shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.size()

        # Q, K, V each has shape: (batch_size, seq_len, hidden_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshaping from (batch_size, seq_len, hidden_dim) to (batch_size, seq_len, nums_head, head_dim)
        # Then transpose to (batch_size, nums_head, seq_len, head_dim)
        # q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)  # [Another approach to do it]
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        k = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        # Matrix multiplication: (batch_size, nums_head, seq_len, head_dim) * (batch_size, nums_head, head_dim, seq_len)
        # Resulting shape: (batch_size, nums_head, seq_len, seq_len)
        # Note that the scaling factor uses head_dim, not hidden_dim.
        attention_val = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        print(f"attention_val shape is {attention_val.size()}")
        print(f"attention_mask shape is {attention_mask.size()}")
        if attention_mask is not None:
            # If attention_mask is provided, it should have shape (batch_size, nums_head, seq_len, seq_len).
            assert attention_val.size() == attention_mask.size()
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float("-inf"))

        # Apply softmax along the last dimension to get attention weights.
        attention_weight = torch.softmax(attention_val, dim=-1)
        
        # Dropout on attention weights
        attention_weight = self.dropout(attention_weight)
        
        # Multiply attention weights with V:
        # (batch_size, nums_head, seq_len, seq_len) * (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        output_tmp = attention_weight @ v

        # Transpose back: (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, seq_len, nums_head, head_dim)
        # -> (batch_size, seq_len, hidden_dim)
        #
        # Note: The transpose operation changes the dimension ordering but does not change the memory layout,
        # resulting in a non-contiguous tensor. The contiguous() method makes the tensor contiguous in memory,
        # allowing subsequent view or reshape operations without error.
        output_tmp = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        # output = output_mid.permute(0, 2, 1, 3).reshpae(batch_size, seq_len, self.hidden_dim)  # # [Another approach to do it]

        output = self.output_proj(output_tmp)
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    batch_size, seq_len, hidden_dim = x.size()
    nums_head = 2

    # attention_mask has shape: (batch_size, nums_head, seq_len, seq_len).
    # Here we use a lower-triangular mask to simulate causal masking.
    attention_mask = torch.tril(torch.ones(batch_size, nums_head, seq_len, seq_len))
    print(attention_mask)

    multi_head_attention = MultiHeadAttention(hidden_dim=hidden_dim, nums_head=nums_head)
    
    x_forward = multi_head_attention.forward(x, attention_mask=attention_mask)
    print(x_forward)
    print(x_forward.size())
```
{{< /collapse >}}

{{< collapse summary="MQA 代码片段" openByDefault=false >}}[multi_query_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/multi_query_attention.py)
```python
import torch
import torch.nn as nn
import math


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        assert hidden_dim % nums_head == 0
        self.head_dim = hidden_dim // nums_head

        self.dropout = nn.Dropout(p=dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)        
        # For kv, project: hidden_dim -> head_dim
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * 1)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * 1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        # Broadcast k and v to match q's dimensions for attention computation
        # k -> (batch_size, 1, seq_len, head_dim)
        # v -> (batch_size, 1, seq_len, head_dim)
        k = K.unsqueeze(1)
        v = V.unsqueeze(1)

        # (batch_size, head_num, seq_len, head_dim) * (batch_size, 1, head_dim, seq_len)
        # -> (batch_size, head_num, seq_len, seq_len)
        attention_val = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        print(f"attention_val  shape is {attention_val.size()}")

        if  attention_mask is not None:
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float("-inf"))
          
        attention_weight = torch.softmax(attention_val, dim=-1)
        print(f"attention_weight is {attention_weight}")
        attention_weight = self.dropout(attention_weight)

        # (batch_size, head_num, seq_len, seq_len) * (batch_size, 1, seq_len, head_dim)
        # -> (batch_size, head_num, seq_len, head_dim)
        output_tmp = attention_weight @ v

        # -> (batch_size, seq_len, head_num, head_dim)
        # -> (batch_size, seq_len, hidden_dim)
        output_tmp = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output_tmp)
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    batch_size, seq_len, hidden_dim = x.size()
    nums_head = 2
    attention_mask = torch.tril(torch.ones(batch_size, nums_head, seq_len, seq_len))
    print(attention_mask)

    multi_query_attention = MultiQueryAttention(hidden_dim=hidden_dim, nums_head=nums_head, dropout=0.2)
    
    x_forward = multi_query_attention.forward(x, attention_mask=attention_mask)
    print(x_forward)
    print(x_forward.size())
```
{{< /collapse >}}


{{< collapse summary="GQA 代码片段" openByDefault=false >}}[group_query_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/group_query_attention.py)
```python 
import math
import torch
import torch.nn as nn


class GQABroadcast(nn.Module):
    """
    Group Query Attention (GQA) implementation:
    By configuring `nums_kv_head` (G, the number of groups), this module supports:
      - When nums_kv_head == nums_head: Multi-Head Attention (MHA)
      - When nums_kv_head == 1: Multi-Query Attention (MQA)
      - When 1 < nums_kv_head < nums_head: Generic Grouped Query Attention (GQA)
    """
    def __init__(self, hidden_dim, nums_head, nums_kv_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head  # Total number of Q heads (H)
        self.nums_kv_head = nums_kv_head # Number of K, V heads (G, groups)
        assert hidden_dim % nums_head == 0
        assert nums_head % nums_kv_head == 0

        self.head_dim = hidden_dim // nums_head
        # Number of Q heads per group
        self.q_heads_per_group = nums_head // nums_kv_head
        self.dropout = nn.Dropout(dropout_rate)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # Projection output dimensions for K, V = nums_kv_head * head_dim
        self.k_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask= None):
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x)  # (batch_size, seq_len, hidden_dim)
        K = self.k_proj(x)  # (batch_size, seq_len, nums_kv_head * head_dim)
        V = self.v_proj(x)  # (batch_size, seq_len, nums_kv_head * head_dim)

        # Q: (batch_size, seq_len, hidden_dim)
        # -> (batch_size, seq_len, nums_head, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2).contiguous()
        q = q.view(batch_size, self.nums_kv_head, self.q_heads_per_group, seq_len, self.head_dim)

        # K, V: (batch_size, seq_len, nums_kv_head * head_dim)
        #  -> (batch_size, seq_len, nums_kv_head, head_dim)
        # -> (batch_size, nums_kv_head, seq_len, head_dim
        # -> (batch_size, nums_kv_head, 1, seq_len, head_dim)
        k = K.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2).unsqueeze(2)
        v = V.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2).unsqueeze(2)

        # q: (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim) * (batch_size, nums_kv_head, 1, head_dim, seq_len)
        # -> (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len)
        attention_val = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        # mask
        if attention_mask is not None:
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float("-inf"))

        # softmax
        attention_weight = torch.softmax(attention_val, dim=-1)

        # dropout
        attention_weight = self.dropout(attention_weight)

        # (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len) * (batch_size, nums_kv_head, 1, seq_len, head_dim)
        # -> (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        output_tmp = attention_weight @ v

        # (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        output_tmp = output_tmp.view(batch_size, self.nums_head, seq_len, self.head_dim)

        # (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, seq_len, nums_head, head_dim) -> (batch_size, seq_len, hidden_dim)
        output_concat = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output_concat)
        return output


class GQARepeat(nn.Module):
    """
    Group Query Attention (GQA) implementation:
    By configuring `nums_kv_head` (G, the number of groups), this module supports:
      - When nums_kv_head == nums_head: Multi-Head Attention (MHA)
      - When nums_kv_head == 1: Multi-Query Attention (MQA)
      - When 1 < nums_kv_head < nums_head: Generic Grouped Query Attention (GQA)
    """
    def __init__(self, hidden_dim, nums_head, nums_kv_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_kv_head = nums_kv_head
        assert hidden_dim % nums_head == 0
        assert nums_head % nums_kv_head == 0
        self.head_dim = hidden_dim // nums_head
        self.q_head_per_group = nums_head // nums_kv_head

        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        # (batch_size, seq_len, hidden_dim)
        Q = self.q_proj(x)
        # (batch_size, seq_len, nums_kv_head * self.head_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # -> (batch_size, seq_len, nums_head, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        # -> (batch_size, seq_len, nums_kv_head, head_dim)
        # -> (batch_size, nums_kv_head, seq_len, head_dim)
        k = K.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2)
        v = V.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2)

        # (batch_size, nums_head, seq_len, head_dim)
        k_repeat = k.repeat_interleave(self.q_head_per_group, dim=1)
        v_repeat = v.repeat_interleave(self.q_head_per_group, dim=1)

        # (batch_size, nums_head, seq_len, seq_len)
        attention_val = q @ k_repeat.transpose(-1, -2) / math.sqrt(self.head_dim)

        # mask
        if attention_mask is not None:
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float('-inf'))
        
        # softmax
        attention_weight = torch.softmax(attention_val, dim=-1)

        # dropout
        attention_weight = self.dropout(attention_weight)

        # (batch_size, nums_head, seq_len, head_dim)
        output_tmp = attention_weight @ v_repeat

        # (batch_size, seq_len, hidden_dim)
        output_concat = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.output_proj(output_concat)
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 16)
    batch_size, seq_len, hidden_dim = x.size()
    nums_head = 8
    head_dim = hidden_dim // nums_head
    nums_kv_head = 4
    q_heads_per_group = nums_head // nums_kv_head
    
    # v1: Boardcast
    # attention_mask_v1 has shape: (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len)
    attention_mask_v1 = torch.tril(torch.ones(batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len))
    gqa_boradcast = GQABroadcast(hidden_dim=hidden_dim, nums_head=nums_head,
                                                nums_kv_head=nums_kv_head, dropout_rate=0.1)
    x_forward_v1 = gqa_boradcast.forward(x, attention_mask=attention_mask_v1)

    # print(x_forward_v1)
    print(x_forward_v1.size())

    # v2: Repeat
    # attention_mask_v2 has shape: (batch_size, nums_head, seq_len, seq_len)
    attention_mask_v2 = torch.tril(torch.ones(batch_size, nums_head, seq_len, seq_len))
    gqa_repeat = GQARepeat(hidden_dim=hidden_dim, nums_head=nums_head,
                                                nums_kv_head=nums_kv_head, dropout_rate=0.1)
    x_forward_v2 = gqa_repeat.forward(x, attention_mask=attention_mask_v2)

    # print(x_forward_v2)
    print(x_forward_v2.size())
```
{{< /collapse >}}


## 时间与空间复杂度分析

> **说明**：下文针对的是一次**前向传播**（forward propagation）的复杂度；在**训练**时，还需要额外考虑**反向传播**（backward propagation）与**参数更新**。反向传播不仅依赖前向传播保存的中间激活值，还需额外计算梯度和存储中间导数，通常使得总计算量和内存占用比前向传播高，导致训练耗时为前向传播的数倍。

在分析不同注意力机制（MHA、MQA、GQA）时，我们主要关注它们在 **自注意力（self-attention）** 或 **交叉注意力（cross-attention）** 过程中，进行前向传播时的时间复杂度和空间复杂度。即使它们在实现细节上（例如是否共享 \(\mathbf{K}\) 和 \(\mathbf{V}\)）有所不同，但从计算量和主要的缓存/显存使用角度来看，其量级大致保持一致。

假设每个位置都会生成查询 \(\mathbf{Q}\)、键 \(\mathbf{K}\) 和值 \(\mathbf{V}\) 的表征，且各矩阵按批量和头数拆分之后的形状如同下式所示：

\[
\mathbf{Q}, \mathbf{K}, \mathbf{V} \;\in\; \mathbb{R}^{B \times H \times S \times d_{\text{head}}}
\]


### 时间复杂度分析

#### 矩阵乘法的通用时间复杂度

对于形状为 $m \times n$ 的矩阵 $\mathbf{A}$ 与形状为 $n \times p$ 的矩阵 $\mathbf{B}$ 进行乘法 $\mathbf{A}\mathbf{B}$，其时间复杂度一般表示为：

$$
\mathcal{O}(m \times n \times p)
$$

在注意力机制的计算中，这一基本结论常用于分析 $\mathbf{Q}\mathbf{K}^\top$ 以及注意力分数与 $\mathbf{V}$ 的乘法等。

#### 自注意力计算的主要步骤及复杂度

1. **点积计算 ($\mathbf{Q}\mathbf{K}^\top$)**  
   - $\mathbf{Q}$ 形状：$B \times H \times S \times d_{\text{head}}$  
   - $\mathbf{K}$ 形状：$B \times H \times S \times d_{\text{head}}$  
   - 因此 $\mathbf{Q}\mathbf{K}^\top$ 的结果形状为 $B \times H \times S \times S$。  
   - 具体的计算量可以视为：对每个批次、每个头，以及序列内所有位置对 $(S \times S)$ 的点积，其中每个点积涉及 $d_{\text{head}}$ 维度的乘加运算。  
   - 故其时间复杂度为：
     
     $$
     \mathcal{O}\bigl(B \times H \times S \times S \times d_{\text{head}}\bigr)
     \;=\;
     \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
     $$

2. **softmax 操作**  
   - 在得到的注意力分数矩阵 $B \times H \times S \times S$ 上进行逐元素的 softmax 运算。  
   - softmax 对矩阵的每个元素执行指数与归一化操作，其复杂度一般为：

     $$
     \mathcal{O}(\text{元素数})
     = \mathcal{O}\bigl(B \times H \times S^2\bigr)
     $$
     
   - 相对于上一步的矩阵乘法，其依赖维度 $d_{\text{head}}$ 的项可以忽略。因此常将其视为比矩阵乘法更小的开销。

3. **加权平均（注意力分数与 $\mathbf{V}$ 的乘法）**  
   - $\mathbf{V}$ 形状：$B \times H \times S \times d_{\text{head}}$  
   - 注意力分数矩阵形状：$B \times H \times S \times S$  
   - 将每个位置的注意力分数与对应的 $\mathbf{V}$ 向量乘加之后，输出仍是 $B \times H \times S \times d_{\text{head}}$。  
   - 其时间复杂度与 $\mathbf{Q}\mathbf{K}^\top$ 的分析类似：
     
     $$
     \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
     $$

将上述三步综合，最主要的开销来自两次矩阵乘法，各为 $\mathcal{O}(B \times H \times S^2 \times d_{\text{head}})$。因此在一次**完整前向**计算时，量级可写为：

$$
\mathcal{O}(B \times H \times S^2 \times d_{\text{head}})
= \mathcal{O}(B \times S^2 \times d).
$$

（这里用到了 $d_{\text{head}} = \frac{d}{H}$）

#### 增量解码/推理场景（KV Cache）下的时间复杂度

{{< figure 
    src="kv_cache.png" 
    caption="Fig. 4. KV cache example. (Image source: [Efficient NLP YouTube Channel](https://www.youtube.com/watch?v=80bIUggRJf4))"
    align="center"
    width="90%"
>}}

参考图4在推理场景（尤其自回归生成）中，通常会使用 **KV Cache** 来缓存先前时刻的 $\mathbf{K}$, $\mathbf{V}$，从而避免重复计算。此时，每生成一个新 token（即处理一个新的时间步）只需：

1. **对新 token 计算 $\mathbf{Q}$（及对应的 $\mathbf{K}$, $\mathbf{V}$）**  
   - 若只保留了投影权重，则新产生的 $\mathbf{Q}$ 和当前时刻的 $\mathbf{K}$, $\mathbf{V}$ 仅涉及 $\mathcal{O}(d^2)$ 参数乘法，但这是**对单个 token**而言，相对开销不大。

2. **与已有 KV Cache 做注意力**  
   - KV Cache 中存储了所有先前时刻的 $\mathbf{K}$, $\mathbf{V}$，形状约为：
     
     $$
     B \times H \times S_{\text{past}} \times d_{\text{head}}
     $$
     
     此时 $S_{\text{past}}$ 表示已经生成的序列长度。
   - 新的 $\mathbf{Q}$ 形状是 $B \times H \times 1 \times d_{\text{head}}$，故新 token 的注意力分数计算为：
     
    $$
    \mathbf{Q}\mathbf{K}^\top 
    : \; \mathcal{O}\bigl(B \times H \times 1 \times S_{\text{past}} \times d_{\text{head}}\bigr) 
    = \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
    $$

   - 同理，对 $\mathbf{V}$ 的加权得到新 token 的输出，也有相同量级：
     
     $$
     \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     $$

3. **更新 KV Cache**  
   - 将新产生的 $\mathbf{K}$, $\mathbf{V}$ 追加到 KV Cache 中，以备下一个时间步使用。此操作在时间复杂度上只是简单的 concat/append，主要在空间上会不断增长。

因此，在增量解码时，每个新 token 的计算量约为：

$$
\mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
$$

而不是一次性地进行 $S \times S$ 规模的注意力计算。若要生成长度为 $S$ 的序列，总体时间在理想情况下也可归纳为

$$
\sum_{k=1}^{S} \mathcal{O}\bigl(B \times H \times k \times d_{\text{head}}\bigr)
= \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
$$

与一次性计算的复杂度同阶，只是**一次性计算**与**逐步计算**的差异。每步只处理 1 个 token 的注意力时，峰值的临时计算量更小，也无需存储完整的 $S \times S$ 注意力分数矩阵。

#### 时间复杂度总结

- **MHA（多头注意力）**：头数多，但每个头分别计算 $\mathbf{K}$, $\mathbf{V}$。  
- **MQA（多查询注意力）**：多个头共享 $\mathbf{K}$, $\mathbf{V}$。  
- **GQA（分组注意力）**：将 $H$ 个头分成 $G$ 个组，每组共享一组 $\mathbf{K}$, $\mathbf{V}$。

不论 MHA / MQA / GQA，在 **完整前向** 下，它们的主要矩阵乘法复杂度均为：

$$
\mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
= \mathcal{O}\bigl(B \times S^2 \times d\bigr)
$$

而在**增量推理场景**（KV Cache）下，单步计算复杂度降低为

$$
\mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
$$

但需要在多步解码过程中维护并更新 KV Cache。

### 空间复杂度分析

空间复杂度既包括模型参数（**权重参数**）的规模，也包括前向计算时需要的**中间激活值**（尤其是注意力得分矩阵、加权结果，以及可能的 KV Cache）的规模。

#### 模型参数规模

1. **线性投影层的参数**  
   对输入向量（维度 $d$）投影到 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ 的维度：

   $$
   \underbrace{d \times d}_{\mathbf{Q}\text{的投影}} 
   \;+\;
   \underbrace{d \times d}_{\mathbf{K}\text{的投影}} 
   \;+\;
   \underbrace{d \times d}_{\mathbf{V}\text{的投影}}
   = 3d^2
   $$
   一般而言，这些参数会再根据头数 $H$ 切分成多头的形式，但总和并不因为头数增加而改变。故其量级为 $\mathcal{O}(d^2)$。

2. **输出合并层的参数**  
   将多头输出拼接后再投影回维度 $d$ 时，通常还会有一个 $d \times d$ 的线性层。这也同样是 $\mathcal{O}(d^2)$。  
   因此，若单独把二者相加，有

   $$
   3d^2 + d^2 = 4d^2
   $$
   仍然可记作 $\mathcal{O}(d^2)$。

#### 前向计算的中间激活值

在进行**训练**或**完整前向**时，需要缓存如下主要张量：

1. **注意力分数矩阵**  
   形状为 $B \times H \times S \times S$。无论使用 MHA、MQA 还是 GQA，每个头（或组）都需要计算与 $\mathbf{Q}\mathbf{K}^\top$ 相关的注意力分数，其规模量级为：

   $$
   \mathcal{O}\bigl(B \times H \times S^2\bigr)
   $$

2. **加权后的输出**  
   形状为 $B \times H \times S \times d_{\text{head}}$，对应每个位置在前向计算中得到的注意力上下文向量。其量级为：

   $$
   \mathcal{O}\bigl(B \times H \times S \times d_{\text{head}}\bigr)
   = \mathcal{O}\bigl(B \times S \times d\bigr)
   $$

3. **不同注意力机制下的 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ 存储**  
   一般在**反向传播**时，需要缓存 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ 的前向输出（或中间梯度）。若要显式存储，其形状及规模通常如下：

   - **MHA（多头注意力）**  
     - $\mathbf{Q}$: $B \times H \times S \times d_{\text{head}}$  
     - $\mathbf{K}$, $\mathbf{V}$: $B \times H \times S \times d_{\text{head}}$
   - **MQA（多查询注意力）**  
     - $\mathbf{Q}$: $B \times H \times S \times d_{\text{head}}$  
     - $\mathbf{K}$, $\mathbf{V}$（共享）: $B \times S \times d$
   - **GQA（分组注意力）**  
     - $\mathbf{Q}$: $B \times H \times S \times d_{\text{head}}$  
     - $\mathbf{K}$, $\mathbf{V}$（分组共享）: $B \times G \times S \times d_{\text{head}}$, 其中 $G \times d_{\text{head}} = d$

#### 增量解码（KV Cache）下的空间消耗

在**推理**（增量解码）场景，往往会使用 **KV Cache** 来保存先前时刻的所有 Key、Value，以免反复计算。此时的存储结构通常是：

- **KV Cache 维度**（以 MHA 为例）：

  $$
  \mathbf{K}, \mathbf{V} : B \times H \times S_{\text{past}} \times d_{\text{head}}
  $$

  随着生成序列长度 $S_{\text{past}}$ 的增长，KV Cache 会**线性**增大。

- **单步注意力分数矩阵**：

  由于每次只对新 token 进行注意力计算，分数矩阵的形状约为

  $$
  B \times H \times 1 \times S_{\text{past}}
  $$

  显著小于训练时的 $B \times H \times S \times S$。

因此，**增量解码时**，大部分临时激活开销（如完整的 $S \times S$ 矩阵）不再需要，但需要为 KV Cache 额外分配一份 $\mathcal{O}(B \times H \times S_{\text{past}} \times d_{\text{head}})$ 的显存，并随着序列长度增长而累积。

#### 综合空间复杂度

- **训练/完整前向**  
  主要激活值（注意力分数矩阵 + 输出 + Q, K, V 显式缓存）可合并表示为

  $$
  \mathcal{O}\bigl(B \times H \times S^2 + B \times S \times d\bigr)
  $$

  当 $S$ 较大时，$B \times H \times S^2$ 常是主要瓶颈。

- **推理/增量解码（KV Cache）**  
  无需完整的 $S^2$ 注意力分数矩阵，但需要一份

  $$
  \mathbf{K},\mathbf{V}\text{ Cache}:
  \;\mathcal{O}(B \times H \times S_{\text{past}} \times d_{\text{head}})
  $$

  会随着解码步数 $S_{\text{past}}$ 增长而线性增加。  
  单次注意力分数仅为 $B \times H \times 1 \times S_{\text{past}}$ 的临时存储，量级显著小于训练场景。

### 结论与对比

1. **时间复杂度**  
   - 对于**自注意力机制**，无论是 **MHA**、**MQA** 还是 **GQA**，在**完整前向**场景下（训练时亦会包含该前向过程），主要的矩阵运算都保持相同量级：  
     
     $$
     \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
     = \mathcal{O}\bigl(B \times S^2 \times d\bigr)
     $$

   - 在 **增量推理（KV Cache）** 场景下，每个新 token 只需

     $$
     \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     $$

     的计算，但需要维护并更新 KV Cache。

2. **空间复杂度**  
   - **模型参数**：三者都在 $\mathcal{O}(d^2)$ 量级。  
   - **中间激活值**（训练/完整前向）：主要由注意力分数矩阵和输出决定，量级为

     $$
     \mathcal{O}\bigl(B \times H \times S^2 + B \times S \times d\bigr)
     $$

   - **增量解码（KV Cache）**：节省了 $S^2$ 大小的临时分数矩阵，但需要一份随着 $S_{\text{past}}$ 增长的 K, V 缓存

     $$
     \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     $$

3. **MQA / GQA 的优势**  
   - 虽然从大 $S$ 场景的理论时间复杂度看，MQA、GQA 与 MHA 并无数量级的差别，但它们在**键、值共享**（或分组共享）带来的**实际带宽、缓存访存效率**方面，往往能在工程实现中取得更好的**显存和速度性能**。

下表总结了 MHA、MQA 和 GQA 三种注意力机制的主要差异：

| 特性                | 多头注意力 (MHA)                     | 多查询注意力 (MQA)                     | 分组查询注意力 (GQA)                        |
|:-------------------:|:------------------------------------:|:--------------------------------------:|:--------------------------------------------:|
| **K/V头数量**      | 与头数量相同（$H$）                | 单一K/V头                              | 分组数（$G$）组，每组1个K/V头             |
| **推理时间**        | 较慢                                 | 最快                                   | 较快，但略高于MQA                            |
| **显存带宽需求**    | 最高，$H$倍的K/V加载              | 最低，仅1个K/V头                        | 介于MHA和MQA之间，$G$倍的K/V加载           |
| **模型容量**        | 最高                                 | 最低                                   | 中等，取决于分组数$G$                      |
| **性能表现**        | 最佳                                 | 略低于MHA                              | 接近MHA，显著优于MQA                           |
| **向上训练需求**    | 无需                                 | 高，需要更多的稳定性和调整              | 较低，GQA模型在少量数据进行向上训练后即可稳定运行      |
| **适用场景**        | 高性能需求但推理速度不敏感的应用      | 推理速度要求极高，且对模型性能要求较低的场景 | 需要在推理速度和模型性能之间取得平衡的应用          |


## 实验结果


### 性能测试

本实验在一台配备双 NVIDIA RTX 4090 GPU 的环境下进行，采用数据并行（Data Parallel, DP）方式，将批量大小（batch size）均匀拆分到两张 GPU 上。实验仅测试了前向传播的性能表现，包括平均延迟时间（Time_mean，单位：ms）和峰值显存占用（Peak_Mem_mean，单位：MB），以评估不同注意力机制（MHA、MQA 和 GQA）在推理阶段的资源需求和效率。
- 实验代码请参考[benchmark_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/benchmark_attention.py)。

测试基于 Llama3 8B 参数超参数设置
{{< figure 
    src="llama3_key_hyperparameters.png" 
    caption="Fig. 5. Overview of the key hyperparameters of Llama 3. (Image source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="100%"
>}}

主要设置参数如下:
- 总层数：32 层。
- 隐藏层维度：4096。
- 多头注意力总头数：32。
- 不同组数（nums_kv_head）配置：32（MHA）、1（MQA）、8（GQA-8）。

### 实验结果

本节主要介绍在不同序列长度（512、1024 和 1536）下，多头注意力（MHA）、多查询注意力（MQA）以及组查询注意力（GQA-8）的实验表现，包含时间延迟和显存占用两个方面的数据。为了方便对比，下表给出了三种注意力机制的具体测试结果。

| Model Size | Method | nums_kv_head | Seq Length | Time_mean (ms) | Peak_Mem_mean (MB) |
|------------|--------|--------------|------------|----------------|--------------------|
| Llama3 8B  | GQA-8  | 8            | 512        | 40.8777        | 2322.328           |
| Llama3 8B  | MHA    | 32           | 512        | 53.0167        | 2706.375           |
| Llama3 8B  | MQA    | 1            | 512        | 37.3592        | 2210.314           |
| Llama3 8B  | GQA-8  | 8            | 1024       | 85.5850        | 6738.328           |
| Llama3 8B  | MQA    | 1            | 1024       | 80.8002        | 6570.314           |
| Llama3 8B  | MHA    | 32           | 1024       | 102.0514       | 7314.375           |
| Llama3 8B  | GQA-8  | 8            | 1536       | 147.5949       | 13586.328          |
| Llama3 8B  | MHA    | 32           | 1536       | 168.8142       | 14354.375          |
| Llama3 8B  | MQA    | 1            | 1536       | 141.5059       | 13362.314          |

{{< figure 
    src="benchmark_time_Llama3_8B.svg" 
    caption="Fig. 6. Average Time Benchmark."
    align="center"
    width="90%"
>}}

{{< figure 
    src="benchmark_mem_Llama3_8B.svg" 
    caption="Fig. 7. Average Peak Memory Benchmark."
    align="center"
    width="90%"
>}}

在显存和时间开销敏感的场景下，MQA 和 GQA-8 是更高效的选择，其中 MQA 表现最优，但可能在模型性能能力上有所不足；GQA-8 则在效率和性能之间达到了良好的平衡。

### GQA 论文实验结果

#### 推理性能

{{< figure 
    src="inference_benchmark_table.png" 
    caption="Fig. 8. Inference time and performance comparison. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="100%"
>}}

{{< figure 
    src="inference_benchmark_image.png" 
    caption="Fig. 9. Additional Experimental Results. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

从实验结果可以看出：

- **推理速度**：
  - MHA-XXL 的推理时间显著高于 MHA-Large，主要由于其更大的头数量和模型规模。
  - MQA-XXL 和 GQA-8-XXL 相比 MHA-XXL，推理时间分别减少至约1/6和1/5。

- **性能表现**：
  - MHA-XXL 在所有任务上表现最佳，但推理时间较长。
  - MQA-XXL 在推理速度上具有优势，平均分仅略低于 MHA-XXL。
  - GQA-8-XXL 在推理速度上接近 MQA-XXL，但在性能上几乎与 MHA-XXL 持平，显示出 GQA 的高效性和优越性。

#### CheckPoint 转化

{{< figure 
    src="checkpoint_conversion.png" 
    caption="Fig. 10. Ablation Study on Checkpoint Conversion Methods. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

图10证明了均值池化方法在保留模型信息方面表现最佳，选择第一个头次之，随机初始化效果最差。均值池化有效地融合了多个 $\mathbf{K}$ 和 $\mathbf{V}$ 头的信息，保持了模型性能。

#### 向上训练比例

{{< figure 
    src="uptraining_ratios.png" 
    caption="Fig. 11. Ablation Study on Uptraining Ratios. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

图11展示了以MHA模型为基准，T5 XXL模型在MQA和GQA上的性能随着向上训练的数据量增加变化情况。
- **GQA**：即使在仅进行转换（无向上训练）的情况下，GQA已具备一定性能，随着向上训练比例增加，性能持续提升。
- **MQA**：需要至少5%比例的预训练数据进行向上训练才能达到实用的性能，且随着比例增加，性能提升趋于平缓。

#### 分组数量对推理速度的影响

{{< figure 
    src="group_number.png" 
    caption="Fig. 12. Effect of the Number of GQA Groups on Inference Speed. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

从图12可以发现随着分组数的增加，推理时间略有上升，但相较于MHA，仍然保持显著的速度优势。选择适中的分组数比如8可以在速度和性能之间取得良好平衡。图3也显示了 llama3 从 7B 到 405B 参数的模型都是才采用8作为分组数（key/value heads = 8）。

## 其他优化方法

除了注意力机制的优化，研究者们还提出了多种方法以提升Transformer模型的推理和训练效率：

- **LoRA** ([HU et al., 2021](https://arxiv.org/abs/2106.09685)): 通过在预训练模型的权重矩阵上添加低秩矩阵来实现高效的参数微调。
- **Flash Attention**（[Dao et al., 2022](https://arxiv.org/abs/2205.14135)）：通过优化注意力计算，减少内存和计算开销。
- **量化技术** LLM.int8（[Dettmers et al., 2022](https://arxiv.org/pdf/2208.07339))和 GPTQ ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323))：通过降低模型权重和激活的精度，减少显存占用和计算成本。
- **模型蒸馏**（[Hinton et al., 2015](https://arxiv.org/abs/1503.02531)）：通过训练小模型模仿大模型的行为，减小模型规模。
- **投机采样** Speculative Sampling（[Chen et al., 2023](https://arxiv.org/pdf/2302.01318)）：通过并行生成和筛选，提升生成效率。

## 关键总结

1. 向上训练方法能够有效利用已有的MHA模型的Checkpoint，通过少量的额外训练，将其转化为更高效的MQA或GQA模型，显著降低了训练成本。
2. **分组查询注意力(GQA)** 在推理效率和模型性能之间取得了良好的平衡，尤其适用于需要高效推理和高性能的应用场景。
3. 实验结果表明，GQA 能够在保持与 MHA 模型相近的性能的同时，显著提升推理速度，适合大规模模型部署和实时应用。

## 参考文献

[1] Vaswani A. [Attention is all you need](https://arxiv.org/abs/1706.03762) [J]. *Advances in Neural Information Processing Systems*, 2017.  
[2] Devlin J. [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [J]. *arXiv preprint arXiv:1810.04805*, 2018.  
[3] Radford A. [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) [J]. 2018.  
[4] Touvron H, Lavril T, Izacard G, et al. [Llama: Open and efficient foundation language models](https://arxiv.org/abs/2302.13971) [J]. *arXiv preprint arXiv:2302.13971*, 2023.  
[5] Achiam J, Adler S, Agarwal S, et al. [Gpt-4 technical report](https://arxiv.org/abs/2303.08774) [J]. *arXiv preprint arXiv:2303.08774*, 2023.  
[6] Shazeer N. [Fast transformer decoding: One write-head is all you need](https://arxiv.org/pdf/1911.02150) [J]. *arXiv preprint arXiv:1911.02150*, 2019.  
[7] Ainslie J, Lee-Thorp J, de Jong M, et al. [Gqa: Training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/pdf/2305.13245) [J]. *arXiv preprint arXiv:2305.13245*, 2023.  
[8] Hu E J, Shen Y, Wallis P, et al. [Lora: Low-rank adaptation of large language models](https://arxiv.org/pdf/2106.09685) [J]. *arXiv preprint arXiv:2106.09685*, 2021.  
[9] Dettmers T, Lewis M, Belkada Y, et al. [Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale](https://arxiv.org/pdf/2208.07339) [J]. *Advances in Neural Information Processing Systems*, 2022, 35: 30318-30332.  
[10] Frantar E, Ashkboos S, Hoefler T, et al. [Gptq: Accurate post-training quantization for generative pre-trained transformers](https://arxiv.org/abs/2210.17323) [J]. *arXiv preprint arXiv:2210.17323*, 2022.  
[11] Hinton G. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) [J]. *arXiv preprint arXiv:1503.02531*, 2015.  
[12] Chen C, Borgeaud S, Irving G, et al. [Accelerating large language model decoding with speculative sampling](https://arxiv.org/pdf/2302.01318) [J]. *arXiv preprint arXiv:2302.01318*, 2023.  

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Jan 2025). Transformer注意力机制：MHA、MQA与GQA的对比.  
https://syhya.github.io/posts/2025-01-16-group-query-attention/

Or

```bibtex
@article{syhya2025gqa,
  title   = "Transformer注意力机制：MHA、MQA与GQA的对比",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jan",
  url     = "https://syhya.github.io/posts/2025-01-16-group-query-attention/"
}
