---
title: "Attention Mechanisms in Transformers: Comparing MHA, MQA, and GQA"
date: 2025-01-16T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "Transformer", "Attention Mechanism", "MHA", "MQA", "GQA", "KV Cache", "NLP", "LLM"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## Background

The Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) is a model based on the encoder-decoder architecture. This model has demonstrated outstanding performance in the field of natural language processing (NLP), leading to a series of optimized models based on it, such as BERT ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)) which uses only the encoder, GPT ([Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)) series which uses only the decoder, and subsequent large language models (LLMs) like LLaMA ([Touvron et al., 2023](https://arxiv.org/abs/2302.13971)) and GPT-4 ([OpenAI et al., 2024](https://arxiv.org/abs/2303.08774)), most of which adopt a decoder-only architecture.

## Notations

| Symbol                                                       | Meaning                                                                                                     |
|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| \(B\)                                                        | Batch Size                                                                                                  |
| \(S\)                                                        | Sequence Length                                                                                             |
| \(d\)                                                        | Hidden Dimension / Model Size                                                                               |
| \(H\)                                                        | Number of Heads in Multi-Head Attention                                                                     |
| \(G\)                                                        | Number of Groups, used for Grouped-Query Attention (GQA)                                                   |
| \(d_{\text{head}} = \frac{d}{H}\)                            | Dimension of each attention head                                                                             |
| \(\mathbf{X} \in \mathbb{R}^{B \times S \times d}\)          | Input sequence, with batch size \(B\), sequence length \(S\), and hidden dimension \(d\)                   |
| \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times S \times d}\) | Query, Key, and Value matrices after linear transformation                                                  |
| \(W_Q, W_K, W_V \in \mathbb{R}^{d \times d}\)                | Trainable linear projection matrices for generating \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) respectively   |
| \(W_O \in \mathbb{R}^{d \times d}\)                          | Trainable linear projection matrix for mapping multi-head/grouped attention outputs back to dimension \(d\) |
| \(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h \in \mathbb{R}^{B \times S \times d_{\text{head}}}\) | Query, Key, and Value sub-matrices for the \(h\)-th attention head                                         |
| \(\mathbf{K}^*, \mathbf{V}^*\)                               | Shared \(\mathbf{K}\) and \(\mathbf{V}\) obtained by averaging or merging all heads' \(\mathbf{K}_h, \mathbf{V}_h\) in Multi-Query Attention (MQA) |
| \(\mathbf{q}, \mathbf{k}\in \mathbb{R}^{d_{\text{head}}}\)   | Single query and key vectors used in mathematical derivations (Central Limit Theorem) in Scaled Dot-Product Attention |

## Attention Mechanism in Transformers

The core of the Transformer model is the **Self-Attention Mechanism**, which allows the model to dynamically focus on different parts of the sequence when processing sequential data. Specifically, given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times S \times d}\) (batch size \(B\), sequence length \(S\), hidden dimension \(d\)), the Transformer projects it into queries (\(\mathbf{Q}\)), keys (\(\mathbf{K}\)), and values (\(\mathbf{V}\)) through three linear layers:

\[
\mathbf{Q} = \mathbf{X} W_Q, \quad
\mathbf{K} = \mathbf{X} W_K, \quad
\mathbf{V} = \mathbf{X} W_V
\]

where \(W_Q, W_K, W_V \in \mathbb{R}^{d \times d}\) are trainable weight matrices. MHA enhances the model's representational capacity by splitting these projections into multiple heads, each responsible for different subspace representations.

There are various forms of attention mechanisms, and the Transformer relies on **Scaled Dot-Product Attention**: given query matrix \(\mathbf{Q}\), key matrix \(\mathbf{K}\), and value matrix \(\mathbf{V}\), the output is a weighted sum of the value vectors, where each weight is determined by the dot product of the query with the corresponding key:

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

### Multi-Head Attention (MHA)

Multi-Head Attention (MHA) splits \(\mathbf{Q}\), \(\mathbf{K}\), and \(\mathbf{V}\) into multiple heads, each with independent \(\mathbf{K}\) and \(\mathbf{V}\), thereby increasing the model's capacity and flexibility:

\[
\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) 
= \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
\]

where each head is computed as:

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

#### Benefits of Using Multi-Head Attention

- **Capturing Diverse Features**: A single-head attention mechanism can only focus on one type of feature or pattern in the input sequence, whereas MHA can simultaneously focus on different features or patterns across multiple attention heads, enabling the model to understand the input data more comprehensively.
- **Enhanced Expressive Power**: Each attention head can learn different representations, enhancing the model's expressive power. Different heads can focus on different parts or relationships within the input sequence, helping the model capture complex dependencies more effectively.
- **Improved Stability and Performance**: MHA reduces noise and instability from individual attention heads by averaging or combining multiple heads, thereby improving the model's stability and performance.
- **Parallel Computation**: MHA allows for parallel computation since each attention head's calculations are independent. This boosts computational efficiency, especially when using hardware accelerators like GPUs or TPUs.

#### Softmax in Scaled Dot-Product Attention

The softmax function transforms a vector \(\mathbf{z} = [z_1, z_2, \dots, z_n]\) into a probability distribution \(\mathbf{y} = [y_1, y_2, \dots, y_n]\) defined as:

\[
y_i = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)} 
\quad \text{for} \quad i = 1, 2, \dots, n
\]

In the attention mechanism, the softmax function is used to convert the scaled dot product \(\tfrac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\) into attention weights:

\[
\text{softmax}\!\Bigl(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\Bigr)
= \Bigl[ 
\frac{\exp\Bigl(\frac{Q_1 \cdot K_1}{\sqrt{d_{\text{head}}}}\Bigr)}{\sum_{j=1}^{S} \exp\Bigl(\frac{Q_1 \cdot K_j}{\sqrt{d_{\text{head}}}}\Bigr)}, 
\dots, 
\frac{\exp\Bigl(\frac{Q_S \cdot K_S}{\sqrt{d_{\text{head}}}}\Bigr)}{\sum_{j=1}^{S} \exp\Bigl(\frac{Q_S \cdot K_j}{\sqrt{d_{\text{head}}}}\Bigr)} 
\Bigr]
\]

In the Transformer's attention mechanism, the scaling factor \(\sqrt{d_{\text{head}}}\) in the scaled dot-product attention formula ensures that the dot product results do not become excessively large as the vector dimension increases before applying softmax. This is primarily for the following reasons:

- **Preventing Gradient Vanishing**: Scaling the attention scores avoids overly large inputs to the softmax function, preventing gradients from vanishing during backpropagation.
- **Numerical Stability**: Scaling ensures that the input range to the softmax function remains reasonable, avoiding extreme values that could lead to numerical instability and overflow issues, especially when the vector dimensions are large. Without scaling, the dot product results could cause the softmax's exponential function to produce excessively large values.
  
- **Mathematical Explanation**: Suppose vectors \(\mathbf{q}\) and \(\mathbf{k}\) have independent and identically distributed components with mean 0 and variance 1. Their dot product \(\mathbf{q} \cdot \mathbf{k}\) has a mean of 0 and a variance of \(d_{\text{head}}\). To prevent the dot product's variance from increasing with the dimension \(d_{\text{head}}\), it is scaled by \(\frac{1}{\sqrt{d_{\text{head}}}}\). This scaling ensures that the variance of the scaled dot product remains 1, independent of \(d_{\text{head}}\).

According to statistical principles, dividing a random variable by a constant scales its variance by the inverse square of that constant. Therefore, the scaling factor \(\tfrac{1}{\sqrt{d_{\text{head}}}}\) effectively controls the magnitude of the attention scores, enhancing numerical stability. The detailed derivation is as follows:

Assume \(\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_{\text{head}}}\) with independent and identically distributed components, mean 0, and variance 1. Their dot product is:

\[
\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_{\text{head}}} q_i k_i
\]

By the Central Limit Theorem, for large \(d_{\text{head}}\), the dot product \(\mathbf{q} \cdot \mathbf{k}\) approximately follows a normal distribution with mean 0 and variance \(d_{\text{head}}\):

\[
\mathbf{q} \cdot \mathbf{k} \sim \mathcal{N}(0, d_{\text{head}})
\]

To achieve unit variance in the scaled dot product, we divide by \(\sqrt{d_{\text{head}}}\):

\[
\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_{\text{head}}}} \;\sim\; \mathcal{N}\!\Bigl(0, \frac{d_{\text{head}}}{d_{\text{head}}}\Bigr) = \mathcal{N}(0, 1)
\]

Thus, the scaled dot product \(\tfrac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_{\text{head}}}}\) maintains a variance of 1, independent of \(d_{\text{head}}\). This scaling operation keeps the dot product within a stable range, preventing the softmax function from encountering numerical instability due to excessively large or small input values.

### Multi-Query Attention (MQA)

Multi-Query Attention (MQA) ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) significantly reduces memory bandwidth requirements by allowing all query heads to share the same set of keys (\(\mathbf{K}\)) and values (\(\mathbf{V}\)). Specifically, if we average all \(\mathbf{K}_h\) and \(\mathbf{V}_h\) from traditional MHA as follows:

\[
\mathbf{K}^* = \frac{1}{H} \sum_{h=1}^{H} \mathbf{K}_h,
\quad 
\mathbf{V}^* = \frac{1}{H} \sum_{h=1}^{H} \mathbf{V}_h,
\]

where \(H\) is the number of query heads, and \(\mathbf{K}_h\) and \(\mathbf{V}_h\) are the keys and values for the \(h\)-th head, respectively. During inference, each head only needs to use the same \(\mathbf{K}^*\) and \(\mathbf{V}^*\), significantly reducing memory bandwidth usage. Finally, all head outputs are concatenated and projected back to the output space:

\[
\text{MQA}(\mathbf{Q}, \mathbf{K}^*, \mathbf{V}^*) 
= \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
\]

Since keys and values are consolidated into a single set, MQA inference is faster but may limit the model's expressive capacity and performance in certain scenarios.

### Grouped-Query Attention (GQA)

Grouped-Query Attention (GQA) ([Ainslie, 2023](https://arxiv.org/pdf/2305.13245)) serves as a compromise between MHA and MQA. It divides query heads into multiple groups, allowing each group to share a set of \(\mathbf{K}\) and \(\mathbf{V}\) heads, thereby balancing inference speed and model performance. Each group contains \(\frac{H}{G}\) query heads and shares one set of \(\mathbf{K}\) and \(\mathbf{V}\) heads. The specific process is as follows:

- **Projection**: Project the input \(\mathbf{X}\) into \(\mathbf{Q}\), \(\mathbf{K}\), and \(\mathbf{V}\) via linear transformations.
- **Grouped Query**: After splitting \(\mathbf{Q}\) into \(H\) heads, further divide these heads into \(G\) groups.
- **Grouped Key/Value**: Split \(\mathbf{K}\) and \(\mathbf{V}\) into \(G\) groups, with each group sharing a set of \(\mathbf{K}\) and \(\mathbf{V}\).
- **Within-Group Attention**: Perform attention calculations for each group's \(\mathbf{Q}\) with the shared \(\mathbf{K}\) and \(\mathbf{V}\).
- **Concatenate Outputs**: Concatenate the attention results from all groups along the channel dimension and project them through a linear layer to obtain the final output.

### Relationship Between the Three Attention Methods

{{< figure 
    src="attention_comparison.png" 
    caption="Fig. 3. Overview of grouped-query method. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))"
    align="center"
    width="100%"
>}}

Figure 3 intuitively illustrates the relationship between the three attention mechanisms: MHA maintains independent \(\mathbf{K}\) and \(\mathbf{V}\) for each query head; MQA allows all query heads to share the same set of \(\mathbf{K}\) and \(\mathbf{V}\); GQA strikes a balance by sharing \(\mathbf{K}\) and \(\mathbf{V}\) within groups.

- **When \(G=1\)**: All query heads share the same set of \(\mathbf{K}\) and \(\mathbf{V}\). In this case, GQA degenerates into MQA.
  - **Number of \(\mathbf{K}/\mathbf{V}\) Heads**: 1
  - **Model Behavior**: All heads use the same \(\mathbf{K}\) and \(\mathbf{V}\) for attention calculations, significantly reducing memory bandwidth requirements.

- **When \(G=H\)**: Each query head has its own independent set of \(\mathbf{K}\) and \(\mathbf{V}\). In this case, GQA degenerates into MHA.
  - **Number of \(\mathbf{K}/\mathbf{V}\) Heads**: \(H\)
  - **Model Behavior**: Each head uses completely independent \(\mathbf{K}\) and \(\mathbf{V}\), maintaining the high model capacity and performance of MHA.

By adjusting the number of groups \(G\), GQA allows flexible switching between MHA and MQA, achieving a balance between maintaining high model performance and improving inference speed.

### Implementation Code Snippet

Below is a simple PyTorch implementation of  **MHA** 、**MQA**和 **GQA**. For GQA, two approaches are demonstrated: broadcasting and repeating.

Additionally, note that in the actual implementation of LLaMA3, GQA incorporates KV Cache for optimization. To keep the example concise, this part is omitted in the code below. For more comprehensive details, you can refer to the official source code in [model.py](https://github.com/meta-llama/llama3/blob/main/llama/model.py).

{{< collapse summary="MHA Code Snippet" openByDefault=false >}}[multi_head_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/multi_head_attention.py)
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

{{< collapse summary="MQA Code Snippet" openByDefault=false >}}[multi_query_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/multi_query_attention.py)
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


{{< collapse summary="GQA Code Snippet" openByDefault=false >}}[group_query_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/group_query_attention.py)
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
    
    # v1: Broadcast
    # attention_mask_v1 has shape: (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len)
    attention_mask_v1 = torch.tril(torch.ones(batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len))
    gqa_broadcast = GQABroadcast(hidden_dim=hidden_dim, nums_head=nums_head,
                                                nums_kv_head=nums_kv_head, dropout_rate=0.1)
    x_forward_v1 = gqa_broadcast.forward(x, attention_mask=attention_mask_v1)

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


## Time and Space Complexity Analysis

> **Note**: The following discussion focuses on the computational complexity of a single **forward pass**. In **training**, one must also account for **backward pass** and parameter updates, which rely on the intermediate activations stored during the forward pass. The additional computation to calculate gradients and maintain partial derivatives usually makes the total training cost (both computation and memory usage) significantly higher—often multiple times the forward-pass cost.

When analyzing different attention mechanisms (MHA, MQA, GQA), our main focus is on their time complexity and space complexity **during the forward pass** of either **self-attention** or **cross-attention**. Even though their implementation details (e.g., whether \(\mathbf{K}\) and \(\mathbf{V}\) are shared) can differ, the overall computational cost and memory usage are roughly on the same order of magnitude.

Assume that each position in the sequence produces its own representations of query \(\mathbf{Q}\), key \(\mathbf{K}\), and value \(\mathbf{V}\). After splitting by batch size and number of heads, their shapes can be written as:

\[
\mathbf{Q}, \mathbf{K}, \mathbf{V} \;\in\; \mathbb{R}^{B \times H \times S \times d_{\text{head}}}
\]


### Time Complexity Analysis

#### General Time Complexity of Matrix Multiplication

For two matrices \(\mathbf{A}\) of shape \(m \times n\) and \(\mathbf{B}\) of shape \(n \times p\), the complexity of computing the product \(\mathbf{A}\mathbf{B}\) is typically expressed as:

\[
\mathcal{O}(m \times n \times p)
\]

In attention-related computations, this formula is frequently used to analyze \(\mathbf{Q}\mathbf{K}^\top\) and the multiplication of attention scores by \(\mathbf{V}\).

#### Main Steps and Complexity in Self-Attention

1. **Dot Product (\(\mathbf{Q}\mathbf{K}^\top\))**  
   - Shape of \(\mathbf{Q}\): \(B \times H \times S \times d_{\text{head}}\)  
   - Shape of \(\mathbf{K}\): \(B \times H \times S \times d_{\text{head}}\)  
   - Consequently, the result of \(\mathbf{Q}\mathbf{K}^\top\) has shape \(B \times H \times S \times S\) 
   - The calculation can be viewed as \(S \times S\) dot products for each head in each batch. Each dot product involves \(d_{\text{head}}\) multiply-add operations.  
   - Hence, its time complexity is:

     \[
     \mathcal{O}\bigl(B \times H \times S \times S \times d_{\text{head}}\bigr)
     \;=\;
     \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
     \]

2. **Softmax Operation**  
   - Applied element-wise to the attention score matrix of shape \(B \times H \times S \times S\)  
   - Softmax entails computing exponentials and performing normalization on each element. The complexity is approximately:

     \[
     \mathcal{O}(\text{number of elements})
     = \mathcal{O}\bigl(B \times H \times S^2\bigr)
     \]
     
   - Compared with the matrix multiplication above, this step’s dependency on \(d_{\text{head}}\) is negligible and is thus often considered a smaller overhead.

3. **Weighted Averaging (Multiplying Attention Scores with \(\mathbf{V}\))**  
   - Shape of \(\mathbf{V}\): \(B \times H \times S \times d_{\text{head}}\)  
   - Shape of the attention score matrix: \(B \times H \times S \times S\)  
   - Multiplying each position’s attention scores by the corresponding \(\mathbf{V}\) vector yields an output of shape \(B \times H \times S \times d_{\text{head}}\)  
   - Its time complexity is analogous to that of \(\mathbf{Q}\mathbf{K}^\top\):

     \[
     \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
     \]

Combining these three steps, the dominant costs come from the two matrix multiplications, each of complexity \(\mathcal{O}(B \times H \times S^2 \times d_{\text{head}})\). Therefore, for a **single full forward** pass, the total complexity can be denoted as:

\[
\mathcal{O}(B \times H \times S^2 \times d_{\text{head}})
= \mathcal{O}(B \times S^2 \times d)
\]

Here, we use \(d_{\text{head}} = \frac{d}{H}\).


#### Time Complexity in Incremental Decoding/Inference with KV Cache

{{< figure 
    src="kv_cache.png" 
    caption="Fig. 4. KV cache example. (Image source: [Efficient NLP YouTube Channel](https://www.youtube.com/watch?v=80bIUggRJf4))"
    align="center"
    width="90%"
>}}

As depicted in Figure 4, **incremental decoding** (especially in autoregressive generation) often employs a **KV Cache** to store previously computed \(\mathbf{K}\) and \(\mathbf{V}\). Thus, one does not have to recalculate keys and values at each new time step. With every new token generated (i.e., a new time step), the following operations are performed:

1. **Compute \(\mathbf{Q}\) for the New Token (and corresponding \(\mathbf{K}, \mathbf{V}\))**  
   - If only the projection weights are retained, then generating the new \(\mathbf{Q}\) vector and the local \(\mathbf{K}, \mathbf{V}\) involves \(\mathcal{O}(d^2)\) parameters, but this overhead is small as it is only for a single token.

2. **Perform Attention with the Existing KV Cache**  
   - The KV Cache stores all previous \(\mathbf{K}, \mathbf{V}\) vectors, with shape:

     \[
     B \times H \times S_{\text{past}} \times d_{\text{head}}
     \]
     
     Here, \(S_{\text{past}}\) is the length of the already-generated sequence.
   - The new \(\mathbf{Q}\) has shape \(B \times H \times 1 \times d_{\text{head}}\). Hence, computing the attention scores for the new token:

     \[
     \mathbf{Q}\mathbf{K}^\top : \; \mathcal{O}\bigl(B \times H \times 1 \times S_{\text{past}} \times d_{\text{head}}\bigr)
     = \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     \]
     
   - Similarly, multiplying these scores by \(\mathbf{V}\) has the same order:

     \[
     \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     \]

3. **Update the KV Cache**  
   - Append the newly generated \(\mathbf{K}, \mathbf{V}\) to the cache, so they can be used at the subsequent time step. This merely requires a concatenation or append operation, which primarily grows the memory usage rather than incurring high compute.

Thus, for incremental decoding, each new token involves:

\[
\mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
\]
in computation, instead of the \(S \times S\) scale for each forward pass. If one aims to generate \(S\) tokens in total, the cumulative complexity (under ideal conditions) becomes:

\[
\sum_{k=1}^{S} \mathcal{O}\bigl(B \times H \times k \times d_{\text{head}}\bigr)
= \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
\]
which is the same order as the one-shot computation. The difference is that incremental decoding computes one token at a time, thus requiring lower temporary memory usage per step and avoiding a full \(S \times S\) attention score matrix at once.

#### Summary of Time Complexity

- **MHA (Multi-Head Attention)**: Multiple heads, each computing its own \(\mathbf{K}, \mathbf{V}\).  
- **MQA (Multi-Query Attention)**: Multiple heads share \(\mathbf{K}, \mathbf{V}\).  
- **GQA (Grouped Query Attention)**: The \(H\) heads are divided into \(G\) groups, each group sharing a single \(\mathbf{K}, \mathbf{V}\).

Regardless of whether we use MHA, MQA, or GQA, in a **full forward pass** (or the forward portion during training), the main matrix multiplications have roughly the same complexity:

\[
\mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
= \mathcal{O}\bigl(B \times S^2 \times d\bigr)
\]

On the other hand, in **incremental inference** with KV Cache, the per-token complexity decreases to

\[
\mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
\]
but one must maintain and update the KV Cache over multiple decoding steps.


### Space Complexity Analysis

Space complexity encompasses both **model parameters (weights)** and **intermediate activations** needed during the forward pass—particularly the attention score matrices, weighted outputs, and potential KV Cache.

#### Model Parameter Scale

1. **Parameters for the Linear Projection Layers**  
   Projecting the input vector of dimension \(d\) into \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\):

   \[
   \underbrace{d \times d}_{\text{Q projection}} 
   + 
   \underbrace{d \times d}_{\text{K projection}} 
   + 
   \underbrace{d \times d}_{\text{V projection}}
   = 3d^2
   \]
   These parameters may be split among heads, but the total remains \(\mathcal{O}(d^2)\), independent of the number of heads \(H\).

2. **Output Merging Layer**  
   After concatenating multiple heads, there is typically another \(d \times d\) linear layer to project the concatenated outputs back into dimension \(d\). This is also \(\mathcal{O}(d^2)\).  
   Therefore, combining these yields:

   \[
   3d^2 + d^2 = 4d^2
   \]
   which remains \(\mathcal{O}(d^2)\).


#### Intermediate Activations for the Forward Pass

During **training** or a **full forward** pass, the following key tensors often need to be stored:

1. **Attention Score Matrix**  
   Shape: \(B \times H \times S \times S\). Regardless of MHA, MQA, or GQA, each head (or group) computes \(\mathbf{Q}\mathbf{K}^\top\) for the attention scores, yielding:

   \[
   \mathcal{O}\bigl(B \times H \times S^2\bigr)
   \]

2. **Weighted Output**  
   Shape: \(B \times H \times S \times d_{\text{head}}\), corresponding to the contextual vectors after weighting \(\mathbf{V}\). Its size is:

   \[
   \mathcal{O}\bigl(B \times H \times S \times d_{\text{head}}\bigr)
   = \mathcal{O}\bigl(B \times S \times d\bigr)
   \]

3. **Storage of \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) for Backprop**  
   In **backward propagation**, we need the forward outputs (or intermediate gradients). If explicitly stored, their shapes and scales are usually:

   - **MHA (Multi-Head Attention)**  
     - \(\mathbf{Q}\): \(B \times H \times S \times d_{\text{head}}\)  
     - \(\mathbf{K}, \mathbf{V}\): \(B \times H \times S \times d_{\text{head}}\)  
   - **MQA (Multi-Query Attention)**  
     - \(\mathbf{Q}\): \(B \times H \times S \times d_{\text{head}}\)  
     - \(\mathbf{K}, \mathbf{V}\) (shared): \(B \times S \times d\)  
   - **GQA (Grouped Query Attention)**  
     - \(\mathbf{Q}\): \(B \times H \times S \times d_{\text{head}}\)  
     - \(\mathbf{K}, \mathbf{V}\) (shared by group): \(B \times G \times S \times d_{\text{head}}\), where \(G \times d_{\text{head}} = d\).


#### Space Usage in Incremental Decoding (KV Cache)

In **inference** with incremental decoding, a **KV Cache** is typically used to store all previously computed keys and values, thus avoiding repeated computation for past tokens. The structure is generally as follows:

- **KV Cache Dimensions** (MHA example):

  \[
  \mathbf{K}, \mathbf{V} : B \times H \times S_{\text{past}} \times d_{\text{head}}
  \]
  As the generated sequence length \(S_{\text{past}}\) grows, the cache usage increases linearly.

- **Per-Step Attention Score Matrix**:
  
  Each new token only requires a score matrix of shape:

  \[
  B \times H \times 1 \times S_{\text{past}}
  \]
  
  which is much smaller than the \(B \times H \times S \times S\) matrix used during training.

Therefore, in **incremental decoding**, large temporary activations—such as the \(S \times S\) score matrix—are not needed; however, the KV Cache itself (size \(\mathcal{O}(B \times H \times S_{\text{past}} \times d_{\text{head}})\)) must be maintained and grows along with the sequence length.


#### Combined Space Complexity

- **Training / Full Forward**  
  The main activations (attention scores + weighted outputs + explicit storage of Q,K,V) add up to:

  \[
  \mathcal{O}\bigl(B \times H \times S^2 + B \times S \times d\bigr)
  \]

  For large \(S\), the \(\mathcal{O}(B \times H \times S^2)\) term tends to dominate.

- **Inference / Incremental Decoding (KV Cache)**  
  There is no need for the full \(S^2\) attention matrix, but a KV Cache of size

  \[
  \mathcal{O}(B \times H \times S_{\text{past}} \times d_{\text{head}})
  \]
  must be stored. This grows linearly with the decoding steps \(S_{\text{past}}\).  
  Meanwhile, the per-step attention matrix is only \(B \times H \times 1 \times S_{\text{past}}\), significantly smaller than the \(\mathcal{O}(S^2)\) scenario in training.


### Conclusions and Comparisons

1. **Time Complexity**  
   - For **self-attention**—whether using **MHA**, **MQA**, or **GQA**—in a **full forward pass** (which also applies to the forward portion during training), the principal matrix multiplications remain:

     \[
     \mathcal{O}\bigl(B \times H \times S^2 \times d_{\text{head}}\bigr)
     = \mathcal{O}\bigl(B \times S^2 \times d\bigr)
     \]

   - In **incremental inference** (KV Cache), each new token only requires

     \[
     \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     \]
     
     but the KV Cache must be updated and maintained throughout the decoding sequence.

2. **Space Complexity**  
   - **Model Parameters**: All three attention mechanisms (MHA, MQA, GQA) reside in \(\mathcal{O}(d^2)\) parameter space.  
   - **Intermediate Activations** (Training / Full Forward): Dominated by the attention score matrix and weighted outputs:

     \[
     \mathcal{O}\bigl(B \times H \times S^2 + B \times S \times d\bigr)
     \]

   - **Incremental Decoding (KV Cache)**: Saves on the \(\mathcal{O}(S^2)\) score matrix cost but requires

     \[
     \mathcal{O}\bigl(B \times H \times S_{\text{past}} \times d_{\text{head}}\bigr)
     \]
     of storage for the KV Cache, which increases linearly with \(S_{\text{past}}\).

3. **Benefits of MQA / GQA**  
   - Although from a high-level perspective, MHA, MQA, and GQA share similar asymptotic complexities when \(S\) is large, **MQA** and **GQA** can achieve improved efficiency in practice due to **key/value sharing** (or partial sharing) which can reduce memory bandwidth demands and improve cache locality. Consequently, in real-world systems, they often deliver better **speed and memory** performance.

The table below summarizes the main differences among MHA, MQA, and GQA attention mechanisms:

| Feature               | MHA             | MQA            | GQA                |
|:---------------------:|:--------------------------------------:|:--------------------------------------:|:--------------------------------------------:|
| **Number of K/V Heads** | Same as number of heads (\(H\))        | Single K/V head                        | Number of groups (\(G\)), one K/V head per group |
| **Inference Time**       | Slower                                 | Fastest                                | Faster, but slightly slower than MQA            |
| **Memory Bandwidth Requirement** | Highest, \(H\) times K/V loading      | Lowest, only one K/V head               | Between MHA and MQA, \(G\) times K/V loading   |
| **Model Capacity**        | Highest                                 | Lowest                                 | Moderate, depending on the number of groups \(G\) |
| **Performance**           | Best                                    | Slightly lower than MHA                | Close to MHA, significantly better than MQA     |
| **Uptraining Requirement** | None                                    | High, requires more stability and tuning | Lower, GQA models stabilize after minimal uptraining |
| **Applicable Scenarios** | Applications with high performance requirements but insensitive to inference speed | Scenarios requiring extremely fast inference with lower model performance demands | Applications needing a balance between inference speed and model performance |

In summary, from a **theoretical** standpoint, all three attention mechanisms (MHA, MQA, GQA) share **\(\mathcal{O}(B \times S^2 \times d)\)** complexity in a full pass and **\(\mathcal{O}(B \times S_{\text{past}} \times d)\)** per-step complexity in incremental decoding.

## Experimental Results

### Performance Testing

This experiment was conducted on an environment equipped with dual NVIDIA RTX 4090 GPUs using data parallelism (DP), evenly splitting the batch size across both GPUs. The experiment only tested the performance of the forward pass, including average latency time (Time_mean, unit: ms) and peak memory usage (Peak_Mem_mean, unit: MB), to evaluate the resource requirements and efficiency of different attention mechanisms (MHA, MQA, and GQA) during the inference phase. 
You can get the source code in [benchmark_attention.py](https://github.com/syhya/syhya.github.io/blob/main/content/en/posts/2025-01-16-group-query-attention/benchmark_attention.py).
- The tests were based on Llama3 8B hyperparameters.

{{< figure 
    src="llama3_key_hyperparameters.png" 
    caption="Fig. 5. Overview of the key hyperparameters of Llama 3. (Image source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="100%"
>}}

The main configuration parameters are as follows:
- **Total Layers**: 32 layers.
- **Hidden Layer Dimension**: 4096.
- **Total Number of Multi-Head Attention Heads**: 32.
- **Different Group Configurations (nums_kv_head)**: 32 (MHA), 1 (MQA), 8 (GQA-8).

### Experimental Results

This section primarily introduces the experimental performance of MHA, MQA, and GQA-8 under different sequence lengths (512, 1024, and 1536), including data on latency and memory usage. For ease of comparison, the table below presents the specific test results for the three attention mechanisms.

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

In scenarios sensitive to memory and time overheads, MQA and GQA-8 are more efficient choices, with MQA performing the best but potentially lacking in model performance capabilities; GQA-8 achieves a good balance between efficiency and performance.

### GQA Paper Experimental Results

#### Inference Performance

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

The experimental results show that:

- **Inference Speed**:
  - MHA-XXL's inference time is significantly higher than MHA-Large, primarily due to its larger number of heads and model size.
  - Compared to MHA-XXL, MQA-XXL and GQA-8-XXL reduce inference time to approximately 1/6 and 1/5, respectively.

- **Performance**:
  - MHA-XXL performs best across all tasks but has longer inference times.
  - MQA-XXL has an advantage in inference speed, with average scores only slightly lower than MHA-XXL.
  - GQA-8-XXL has inference speed close to MQA-XXL but nearly matches MHA-XXL in performance, demonstrating the efficiency and superiority of GQA.

#### Checkpoint Conversion

{{< figure 
    src="checkpoint_conversion.png" 
    caption="Fig. 10. Ablation Study on Checkpoint Conversion Methods. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

Figure 10 compares the performance of different methods for checkpoint conversion. The mean pooling method performs best in retaining model information, followed by selecting the first head, while random initialization performs the worst. Mean pooling effectively integrates information from multiple \(\mathbf{K}\) and \(\mathbf{V}\) heads, maintaining model performance.

#### Uptraining Ratio

{{< figure 
    src="uptraining_ratios.png" 
    caption="Fig. 11. Ablation Study on Uptraining Ratios. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

Figure 11 shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA.
- **GQA**: Even with only conversion (no uptraining), GQA already has certain performance. As the uptraining ratio increases, performance continues to improve.
- **MQA**: Requires at least a 5% uptraining ratio to achieve practical performance, and as the ratio increases, performance gains tend to plateau.

#### Effect of Number of GQA Groups on Inference Speed

{{< figure 
    src="group_number.png" 
    caption="Fig. 12. Effect of the Number of GQA Groups on Inference Speed. (Image source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))" 
    align="center"
    width="80%"
>}}

Figure 12 demonstrates that as the number of groups increases, inference time slightly rises, but it still maintains a significant speed advantage over MHA. Choosing an appropriate number of groups, such as 8, can achieve a good balance between speed and performance. Figure 3 also shows that models ranging from 7B to 405B parameters in Llama3 adopt 8 as the number of groups (key/value heads = 8).

## Other Optimization Methods

In addition to optimizing the attention mechanism, researchers have proposed various methods to enhance the inference and training efficiency of Transformer models:
- **LoRA** ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)): Achieves efficient parameter fine-tuning by adding low-rank matrices to the pretrained model's weight matrices.
- **Flash Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)): Reduces memory and computational overhead by optimizing attention calculations.
- **Quantization Techniques**: LLM.int8 ([Dettmers et al., 2022](https://arxiv.org/pdf/2208.07339)) and GPTQ ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)) reduce memory usage and computational costs by lowering the precision of model weights and activations.
- **Model Distillation** ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)): Reduces model size by training smaller models to mimic the behavior of larger models.
- **Speculative Sampling** ([Chen et al., 2023](https://arxiv.org/pdf/2302.01318)): Enhances generation efficiency through parallel generation and filtering.

## Key Takeaways

1. **Uptraining** methods can effectively utilize existing MHA model checkpoints. By performing a small amount of additional training, they can transform these into more efficient MQA or GQA models, significantly reducing training costs.
2. **Grouped-Query Attention (GQA)** strikes a good balance between inference efficiency and model performance, making it especially suitable for applications requiring both high-efficiency inference and high performance.
3. Experimental results demonstrate that GQA can significantly improve inference speed while maintaining performance comparable to MHA models, making it suitable for large-scale model deployment and real-time applications.

## References

[1] Vaswani A. [Attention is all you need](https://arxiv.org/abs/1706.03762) [J]. *Advances in Neural Information Processing Systems*, 2017.  
[2] Devlin J. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) [J]. *arXiv preprint arXiv:1810.04805*, 2018.  
[3] Radford A. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) [J]. 2018.  
[4] Touvron H, Lavril T, Izacard G, et al. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) [J]. *arXiv preprint arXiv:2302.13971*, 2023.  
[5] Achiam J, Adler S, Agarwal S, et al. [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) [J]. *arXiv preprint arXiv:2303.08774*, 2023.  
[6] Shazeer N. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150) [J]. *arXiv preprint arXiv:1911.02150*, 2019.  
[7] Ainslie J, Lee-Thorp J, de Jong M, et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245) [J]. *arXiv preprint arXiv:2305.13245*, 2023.  
[8] Hu E J, Shen Y, Wallis P, et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685) [J]. *arXiv preprint arXiv:2106.09685*, 2021.  
[9] Dettmers T, Lewis M, Belkada Y, et al. [GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/pdf/2208.07339) [J]. *Advances in Neural Information Processing Systems*, 2022, 35: 30318-30332.  
[10] Frantar E, Ashkboos S, Hoefler T, et al. [GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers](https://arxiv.org/abs/2210.17323) [J]. *arXiv preprint arXiv:2210.17323*, 2022.  
[11] Hinton G. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) [J]. *arXiv preprint arXiv:1503.02531*, 2015.  
[12] Chen C, Borgeaud S, Irving G, et al. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318) [J]. *arXiv preprint arXiv:2302.01318*, 2023.  

## Citation

> **Citation**: To reproduce or cite the content of this article, please acknowledge the original author and source.
    
**Cited as:**
    
> Yue Shui. (Jan 2025). Attention Mechanisms in Transformers: Comparing MHA, MQA, and GQA.  
https://syhya.github.io/posts/2025-01-16-group-query-attention/
    
Or
    
```bibtex
@article{syhya2025gqa,
  title   = "Attention Mechanisms in Transformers: Comparing MHA, MQA, and GQA",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jan",
  url     = "https://syhya.github.io/posts/2025-01-16-group-query-attention/"
}
