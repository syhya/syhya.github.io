---
title: "DeepSeek-V2 vs V3"
date: 2025-04-18T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "LLM", "DeepSeek-V2", "DeepSeek-V3", "MoE", "Transformer", "MLA", "DeepSeekMoE", "MTP", "FP8 Training", "GRPO", "SFT", "RL", "KV Cache"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

DeepSeek AI successively released **DeepSeek-V2** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434)) and **DeepSeek-V3** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)), two powerful Mixture-of-Experts (MoE) language models that significantly optimize training costs and inference efficiency while maintaining state-of-the-art performance. DeepSeek-V2 has a total of 236B parameters, activating 21B per token, while DeepSeek-V3 further expands to 671B total parameters, activating 37B per token. Both support a 128K context length.

The core innovations of these two models lie in the adoption of **Multi-head Latent Attention (MLA)** and the **DeepSeekMoE** architecture ([Dai et al., 2024](https://arxiv.org/abs/2401.06066)). MLA drastically reduces GPU memory usage during inference by compressing the Key-Value (KV) cache into low-dimensional latent vectors, improving efficiency. DeepSeekMoE achieves stronger expert specialization capabilities and more economical training costs through fine-grained expert segmentation and shared expert isolation. Building upon V2, DeepSeek-V3 further introduces an **Auxiliary-Loss-Free Load Balancing** strategy ([Wang et al., 2024](https://arxiv.org/abs/2408.15664)) and the **Multi-Token Prediction (MTP)** ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)) training objective, further enhancing model performance and training efficiency.

DeepSeek-V2 was pre-trained on 8.1T tokens, while DeepSeek-V3 was trained on a larger scale of 14.8T tokens. Both underwent Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) stages to fully unlock their potential. Evaluation results show that both DeepSeek-V2 and V3 achieved top-tier performance among open-source models across numerous benchmarks. DeepSeek-V3, in particular, has become one of the strongest open-source base models currently available, with performance comparable to top closed-source models.

{{< figure
    src="deepseek_v2_benchmark.png"
    caption="Fig. 1. (a) MMLU accuracy vs. activated parameters, among different open-source models. (b) Training costs and inference efficiency of DeepSeek 67B (Dense) and DeepSeek-V2. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

{{< figure
    src="deepseek_v3_benchmark.png"
    caption="Fig. 2. Benchmark performance of DeepSeek-V3 and its counterparts. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

This article will delve into the key technologies of DeepSeek-V2 and DeepSeek-V3, including their innovative model architectures, efficient training infrastructure, pre-training, and alignment processes.

## Notations

The following table lists the mathematical notations used in this article to help you read more easily.

| Symbol | Meaning |
| :--- | :--- |
| \( d \) | Embedding dimension |
| \( n_h \) | Number of attention heads |
| \( d_h \) | Dimension per attention head |
| \( \mathbf{h}_t \in \mathbb{R}^d \) | Input to the attention layer for the \( t \)-th token |
| \( \mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \) | Query, Key, Value vectors |
| \( W^Q, W^K, W^V, W^O \) | Projection matrices for Query, Key, Value, Output |
| \( \mathbf{q}_{t,i}, \mathbf{k}_{t,i}, \mathbf{v}_{t,i} \) | Query, Key, Value vectors for the \( i \)-th attention head |
| \( \mathbf{o}_{t,i} \) | Output of the \( i \)-th attention head |
| \( \mathbf{u}_t \) | Final output of the attention layer |
| \( l \) | Number of model layers |
| \( \mathbf{c}_t^{KV} \in \mathbb{R}^{d_c} \) | Compressed latent vector for key-value |
| \( d_c \) | KV compression dimension |
| \( W^{DKV}, W^{UK}, W^{UV} \) | Down-projection matrix for KV, Up-projection matrix for Key, Up-projection matrix for Value |
| \( \mathbf{k}_t^C, \mathbf{v}_t^C \) | Key and Value recovered from the latent vector via up-projection |
| \( \mathbf{c}_t^Q \in \mathbb{R}^{d_c'} \) | Compressed latent vector for query |
| \( d_c' \) | Query compression dimension |
| \( W^{DQ}, W^{UQ} \) | Down-projection matrix for Query, Up-projection matrix for Query |
| \( \mathbf{q}_t^C \) | Query recovered from the latent vector via up-projection |
| \( \mathbf{q}_{t,i}^R, \mathbf{k}_t^R \) | Decoupled RoPE query and key |
| \( d_h^R \) | Head dimension for decoupled RoPE query/key |
| \( W^{QR}, W^{KR} \) | Generation matrices for decoupled RoPE query/key |
| \( \operatorname{RoPE}(\cdot) \) | Operation applying Rotary Position Embedding |
| \( [\cdot ; \cdot] \) | Concatenation operation |
| \( n_g \) | Number of groups in GQA |
| \( n \) | Total number of experts in MoE |
| \( E_i \) | The \( i \)-th expert network |
| \( G(\cdot) \) | Gating network function |
| \( p_i \) | The \( i \)-th probability output by the gating network |
| \( H^{(i)}(x) \) | Gating score for expert \( i \) in Noisy Top-k Gating |
| \( W_g, W_{\text{noise}} \) | Weight matrices for MoE gating network and noise network |
| \( \epsilon \) | Standard Gaussian noise |
| \( \text{softplus}(\cdot) \) | Softplus activation function |
| \( k \) | Number of experts selected per token in MoE |
| \( \text{topk}(\cdot, k) \) | Function selecting the top k largest values |
| \( \mathcal{L}_{\text{aux}} \) | MoE auxiliary loss |
| \( w_{\text{aux}} \) | Auxiliary loss weight |
| \( \text{CV}(\cdot) \) | Coefficient of Variation |
| \( N_s, N_r \) | Number of shared and routing experts in DeepSeekMoE |
| \( \operatorname{FFN}_i^{(s)}(\cdot), \operatorname{FFN}_i^{(r)}(\cdot) \) | The \( i \)-th shared expert and routing expert function |
| \( K_r \) | Number of activated routing experts in DeepSeekMoE |
| \( g_{i,t} \) | Gating value of the \( i \)-th expert for the \( t \)-th token |
| \( g_{i,t}' \) | Raw gating value after TopK selection (V3) |
| \( s_{i,t} \) | Affinity score of the \( t \)-th token for the \( i \)-th expert |
| \( \mathbf{e}_i \) | Center vector for the \( i \)-th routing expert |
| \( M \) | Device/Node limit for routing |
| \( \mathcal{L}_{\text{ExpBal}}, \mathcal{L}_{\text{DevBal}}, \mathcal{L}_{\text{CommBal}} \) | Expert-level, Device-level, Communication-level load balancing losses |
| \( f_i, P_i \) | Load score and average affinity for expert \( i \) |
| \( \alpha_1, \alpha_2, \alpha_3 \) | Hyperparameters for load balancing losses |
| \( T \) | Number of tokens in the sequence |
| \( D \) | Number of device/node groups |
| \( \mathcal{E}_i \) | Set of experts on the \( i \)-th device/node |
| \( f_i', P_i' \) | Average load score and total affinity for device group \( i \) |
| \( f_i'', P_i'' \) | Proportion of tokens sent to device \( i \) and total affinity for device group \( i \) |
| \( b_i \) | Bias term for the \( i \)-th expert (aux-loss-free balancing) |
| \( \gamma \) | Bias term update rate |
| \( \mathcal{L}_{\text{Bal}} \) | Sequence-level load balancing loss |
| \( \alpha \) | Hyperparameter for sequence-level load balancing loss |
| \( D_{MTP} \) | MTP prediction depth |
| \( \operatorname{Emb}(\cdot), \operatorname{OutHead}(\cdot) \) | Shared embedding layer and output head (MTP) |
| \( \operatorname{TRM}_k(\cdot) \) | Transformer block for the \( k \)-th MTP module |
| \( M_k \) | Projection matrix for the \( k \)-th MTP module |
| \( \mathbf{h}_i^k \) | Representation of the \( i \)-th token at the \( k \)-th MTP depth |
| \( \mathbf{h}_i^{\prime k} \) | Input to the Transformer block of the \( k \)-th MTP module |
| \( P_{i+k+1}^k \) | Predicted probability distribution for the \( i+k+1 \)-th token by the \( k \)-th MTP module |
| \( V \) | Vocabulary size |
| \( \mathcal{L}_{\text{MTP}}^k \) | Cross-entropy loss for the \( k \)-th MTP depth |
| \( \mathcal{L}_{\text{MTP}} \) | Total MTP loss |
| \( \lambda \) | Weight factor for MTP loss |
| \( \mathcal{J}_{GRPO}(\theta) \) | GRPO objective function |
| \( A_i \) | Relative advantage value (GRPO) |
| \( \varepsilon \) | Clipping hyperparameter in PPO/GRPO |
| \( \beta \) | Coefficient for KL divergence penalty term |
| \( \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \) | KL divergence |
| \( \pi_\theta, \pi_{\theta_{old}}, \pi_{ref} \) | Current policy, old policy, reference policy models |
| \( r_i \) | Reward value for the \( i \)-th output |
| \( \mathbb{1}(\cdot) \) | Indicator function |

## Core Architecture

Both DeepSeek-V2 and V3 are based on the Transformer architecture, but employ innovative designs in the attention and feed-forward network (FFN) parts, such as MLA and DeepSeekMoE, to balance performance, training cost, and inference efficiency. The figure below illustrates the architecture of DeepSeek-V2 and V3.

{{< figure
    src="deepseek_architecture.png"
    caption="Fig. 3. Illustration of the architecture of DeepSeek-V2 and DeepSeek-V3. MLA ensures efficient inference by significantly reducing the KV cache for generation, and DeepSeekMoE enables training strong models at an economical cost through the sparse architecture. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

### Multi-head Latent Attention (MLA)

Traditional Transformer models typically use **Multi-Head Attention (MHA)** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), but during generation, their large KV cache becomes a bottleneck limiting inference efficiency. To address this, researchers proposed **Multi-Query Attention (MQA)** ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) and **Grouped-Query Attention (GQA)** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)). While these methods reduce the KV cache, they often come at the cost of model performance.

DeepSeek-V2 and V3 adopt the innovative **Multi-head Latent Attention (MLA)** mechanism. The core idea of MLA is **Low-Rank Key-Value Joint Compression**.

{{< figure
    src="mla.png"
    caption="Fig. 4. Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

#### MHA Recap

Standard MHA first transforms the input \(\mathbf{h}_t \in \mathbb{R}^d\) into query \(\mathbf{q}_t\), key \(\mathbf{k}_t\), and value \(\mathbf{v}_t \in \mathbb{R}^{d_h n_h}\) using three projection matrices \(W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}\):
\[
\begin{aligned}
\mathbf{q}_{t} &= W^{Q} \mathbf{h}_{t}, \\
\mathbf{k}_{t} &= W^{K} \mathbf{h}_{t}, \\
\mathbf{v}_{t} &= W^{V} \mathbf{h}_{t}.
\end{aligned}
\]
Then, \(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t\) are split into \(n_h\) heads for multi-head attention computation:
\[
\begin{aligned}
& [\mathbf{q}_{t, 1} ; \mathbf{q}_{t, 2} ; \ldots ; \mathbf{q}_{t, n_{h}}] = \mathbf{q}_{t}, \\
& [\mathbf{k}_{t, 1} ; \mathbf{k}_{t, 2} ; \ldots ; \mathbf{k}_{t, n_{h}}] = \mathbf{k}_{t}, \\
& [\mathbf{v}_{t, 1} ; \mathbf{v}_{t, 2} ; \ldots ; \mathbf{v}_{t, n_{h}}] = \mathbf{v}_{t}, \\
& \mathbf{o}_{t, i} = \sum_{j=1}^{t} \operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t, i}^{T} \mathbf{k}_{j, i}}{\sqrt{d_{h}}}\right) \mathbf{v}_{j, i}, \\
& \mathbf{u}_{t} = W^{O}\left[\mathbf{o}_{t, 1} ; \mathbf{o}_{t, 2} ; \ldots ; \mathbf{o}_{t, n_{h}}\right],
\end{aligned}
\]
where \(\mathbf{q}_{t, i}, \mathbf{k}_{t, i}, \mathbf{v}_{t, i} \in \mathbb{R}^{d_h}\) are the query, key, and value for the \(i\)-th head, respectively, and \(W^O \in \mathbb{R}^{d \times d_h n_h}\) is the output projection matrix. During inference, keys and values for all \(t\) steps need to be cached, requiring \(2 n_h d_h l\) elements per token (\(l\) being the number of layers), which constitutes a huge KV cache overhead.

#### Low-Rank Key-Value Joint Compression

MLA introduces a low-dimensional latent vector \(\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}\) to jointly compress keys and values, where \(d_c \ll d_h n_h\):
\[
\begin{aligned}
\boxed{\mathbf{c}_{t}^{K V}} &= W^{D K V} \mathbf{h}_{t}, \\
\mathbf{k}_{t}^{C} &= W^{U K} \mathbf{c}_{t}^{K V}, \\
\mathbf{v}_{t}^{C} &= W^{U V} \mathbf{c}_{t}^{K V}.
\end{aligned}
\]
Here, \(W^{DKV} \in \mathbb{R}^{d_c \times d}\) is the down-projection matrix, and \(W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}\) are the up-projection matrices for keys and values, respectively. During inference, MLA **only needs to cache the compressed latent vector \(\mathbf{c}_t^{KV}\)** (and the decoupled RoPE key \(\mathbf{k}_t^R\) mentioned later), greatly reducing the KV cache size.

To reduce activation memory during training, MLA also applies similar low-rank compression to the query:
\[
\begin{aligned}
\mathbf{c}_{t}^{Q} &= W^{D Q} \mathbf{h}_{t}, \\
\mathbf{q}_{t}^{C} &= W^{U Q} \mathbf{c}_{t}^{Q},
\end{aligned}
\]
where \(\mathbf{c}_t^Q \in \mathbb{R}^{d_c'}\) is the compressed latent vector for the query, \(d_c' \ll d_h n_h\), and \(W^{DQ} \in \mathbb{R}^{d_c' \times d}\) and \(W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}\) are the down-projection and up-projection matrices for the query, respectively.

#### Decoupled Rotary Position Embedding

Standard **Rotary Position Embedding (RoPE)** ([Su et al., 2024](https://arxiv.org/abs/2104.09864)) is applied directly to keys and queries, but this is incompatible with MLA's low-rank KV compression. If RoPE were applied to the compressed key \(\mathbf{k}_t^C\), the up-projection matrix \(W^{UK}\) would couple with the position-dependent RoPE matrix. This would prevent absorbing \(W^{UK}\) into \(W^Q\) during inference, requiring recomputation of keys for all prefix tokens, severely impacting efficiency.

To solve this, MLA proposes the **Decoupled RoPE** strategy. It introduces an additional multi-head query \(\mathbf{q}_{t, i}^R \in \mathbb{R}^{d_h^R}\) and a shared key \(\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}\) specifically to carry the RoPE information:
\[
\begin{aligned}
\left[\mathbf{q}_{t,1}^R;\,\mathbf{q}_{t,2}^R;\,\dots;\,\mathbf{q}_{t,n_h}^R\right]
= \mathbf{q}_t^R
&= \operatorname{RoPE}\bigl(W^{Q R}\,\mathbf{c}_t^Q\bigr),\\
\boxed{\mathbf{k}_t^R}
&= \operatorname{RoPE}\bigl(W^{K R}\,\mathbf{h}_t\bigr).
\end{aligned}
\]
Here, \(W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}\) and \(W^{KR} \in \mathbb{R}^{d_h^R \times d}\) are matrices generating the decoupled query and key.
The compressed key/query parts (\(C\)) are then concatenated with the decoupled RoPE parts (\(R\)) to form the final keys and queries:
\[
\begin{aligned}
\mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C} ; \mathbf{q}_{t, i}^{R}], \\
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C} ; \mathbf{k}_{t}^{R}].
\end{aligned}
\]
The final attention computation becomes:
\[
\begin{aligned}
\mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t, i}^{T} \mathbf{k}_{j, i}}{\sqrt{d_{h}+d_{h}^{R}}}\right) \mathbf{v}_{j, i}^{C}, \\
\mathbf{u}_{t} &= W^{O}\left[\mathbf{o}_{t, 1} ; \mathbf{o}_{t, 2} ; \ldots ; \mathbf{o}_{t, n_{h}}\right].
\end{aligned}
\]
During inference, besides caching \(\mathbf{c}_t^{KV}\), the decoupled RoPE key \(\mathbf{k}_t^R\) also needs to be cached. Therefore, DeepSeek-V2/V3 require caching a total of \((d_c + d_h^R)l\) elements per token.

#### Matrix Absorption in MLA Inference

A key advantage of MLA is the improvement in inference efficiency, partly due to the associative property of matrix multiplication allowing the up-projection matrices \(W^{UK}\) and \(W^{UV}\) to be "absorbed," avoiding the explicit computation of the full keys \(\mathbf{k}_t^C\) and values \(\mathbf{v}_t^C\).

**1. Absorbing \(W^{UK}\) (Optimizing Attention Score Calculation):**

The core of attention score calculation is the dot product of query and key \(\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}\). Focusing on the \(C\) part generated from the compressed vector:
\[
(\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C
\]
Substitute \(\mathbf{k}_{j,i}^C = W^{UK} \mathbf{c}_j^{KV}\):
\[
(\mathbf{q}_{t,i}^C)^T (W^{UK} \mathbf{c}_j^{KV})
\]
Using matrix multiplication associativity \((AB)C = A(BC)\) and transpose property \((AB)^T = B^T A^T\), the expression can be rewritten as:
\[
(\mathbf{q}_{t,i}^C)^T (W^{UK} \mathbf{c}_j^{KV}) = ((W^{UK})^T \mathbf{q}_{t,i}^C)^T \mathbf{c}_j^{KV}
\]
The significance of this transformation is: we no longer need to apply \(W^{UK}\) to the cached \(\mathbf{c}_j^{KV}\) to get \(\mathbf{k}_{j,i}^C\). Instead, we can first compute an "effective query" \(\tilde{\mathbf{q}}_{t,i}^C = (W^{UK})^T \mathbf{q}_{t,i}^C\), and then directly compute the dot product of this effective query with the cached latent vector \(\mathbf{c}_j^{KV}\).

The original query \(\mathbf{q}_{t,i}^C\) is computed from \(\mathbf{h}_t\) via \(W^{UQ}\) and \(W^{DQ}\) (\(\mathbf{q}_{t,i}^C = (W^{UQ} W^{DQ} \mathbf{h}_t)_i\)). Thus, the entire computation from \(\mathbf{h}_t\) to the effective query \(\tilde{\mathbf{q}}_{t,i}^C\) can be viewed as a new, effective query projection operation that incorporates \(W^{UK}\). In practice, this means after computing \(\mathbf{q}_{t,i}^C\), one can left-multiply by \((W^{UK})^T\), or more efficiently, merge \((W^{UK})^T\) into the original query generation matrix \(W^Q\) (or \(W^{UQ}W^{DQ}\)) to form a new query projection matrix \(\tilde{W}^Q = (W^{UK})^T W^{UQ} W^{DQ}\).

Crucially, the computation involving \(W^{UK}\) is moved to the query side and performed once before calculating attention scores, eliminating the need to recover \(\mathbf{k}_{j,i}^C\) from the cached \(\mathbf{c}_j^{KV}\) using \(W^{UK}\) for every query.

**2. Absorbing \(W^{UV}\) (Optimizing Weighted Sum):**

The output of an attention head \(\mathbf{o}_{t,i}\) is the weighted sum of attention weights (denoted \(w_{ij}\)) and values \(\mathbf{v}_{j,i}^C\):
\[
\mathbf{o}_{t, i} = \sum_{j=1}^{t} w_{ij} \cdot \mathbf{v}_{j, i}^{C}
\]
Substitute \(\mathbf{v}_{j,i}^C = (W^{UV} \mathbf{c}_j^{KV})_i\) (where \((\cdot)_i\) denotes the part belonging to the \(i\)-th head):
\[
\mathbf{o}_{t, i} = \sum_{j=1}^{t} w_{ij} \cdot (W^{UV} \mathbf{c}_j^{KV})_i
\]
The final attention layer output \(\mathbf{u}_t\) is obtained by concatenating the outputs of all heads \(\mathbf{o}_{t,i}\) and projecting through the output matrix \(W^O\):
\[
\mathbf{u}_{t} = W^{O}\left[\mathbf{o}_{t, 1} ; \ldots ; \mathbf{o}_{t, n_{h}}\right] = W^{O} \begin{bmatrix} \sum_{j} w_{1j} (W^{UV} \mathbf{c}_j^{KV})_1 \\ \vdots \\ \sum_{j} w_{n_h j} (W^{UV} \mathbf{c}_j^{KV})_{n_h} \end{bmatrix}
\]
Due to the linearity of matrix multiplication (\(A(B+C) = AB + AC\) and \(A(cB) = c(AB)\)), \(W^{UV}\) can be "factored out" of the summation (this is for intuitive understanding; the actual operation is at the matrix level):
\[
\mathbf{u}_{t} \approx W^{O} W^{UV} \left( \sum_{j=1}^{t} \begin{bmatrix} w_{1j} (\mathbf{c}_j^{KV})_1 \\ \vdots \\ w_{n_h j} (\mathbf{c}_j^{KV})_{n_h} \end{bmatrix} \right)
\]
(Note: \((\mathbf{c}_j^{KV})_i\) here is illustrative; in practice, operations are performed directly on the complete \(\mathbf{c}_j^{KV}\), but the principle is the same: first perform the weighted sum on \(\mathbf{c}_j^{KV}\), then apply \(W^{UV}\) and \(W^O\)).

Let the effective output matrix be \(\tilde{W}^O = W^O W^{UV}\). This means we can first compute the weighted sum of attention weights and the latent vectors \(\mathbf{c}_j^{KV}\) (yielding an intermediate result \(\tilde{\mathbf{o}}_t = \sum_j w_{ij} \mathbf{c}_j^{KV}\) of dimension \(d_c\)), and then directly use this merged effective output matrix \(\tilde{W}^O\) for the final projection to get \(\mathbf{u}_t\). Similarly, the computation involving \(W^{UV}\) is merged into the final output projection step, eliminating the need to recover \(\mathbf{v}_{j,i}^C\) from \(\mathbf{c}_j^{KV}\) during the weighted sum calculation.

**Summary:** Through matrix absorption, MLA avoids repeatedly computing the high-dimensional keys \(\mathbf{k}_{j,i}^C\) and values \(\mathbf{v}_{j,i}^C\) from the cached low-dimensional latent vectors \(\mathbf{c}_j^{KV}\) during inference, significantly improving computational efficiency. Only \(\mathbf{c}_t^{KV}\) and \(\mathbf{k}_t^R\) are actually cached.

#### KV Cache Comparison

The table below compares the per-token KV cache size for different attention mechanisms. \(n_h\) is the number of attention heads, \(d_h\) is the dimension per head, \(l\) is the number of layers, \(n_g\) is the number of GQA groups, and \(d_c\) and \(d_h^R\) are MLA's KV compression dimension and decoupled RoPE dimension. For DeepSeek-V2, \(d_c = 4d_h\), \(d_h^R = d_h/2\), making its KV cache equivalent to GQA with \(n_g=2.25\), but with performance superior to MHA. DeepSeek-V3 uses a similar configuration.

| Attention Mechanism | Per-Token KV Cache Size (# elements) | Capability |
| :--- | :---: | :---: |
| Multi-Head Attention (MHA) | \(2 n_{h} d_{h} l\) | Strong |
| Grouped-Query Attention (GQA) | \(2 n_{g} d_{h} l\) | Medium |
| Multi-Query Attention (MQA) | \(2 d_{h} l\) | Weak |
| Multi-head Latent Attention (MLA) | \(\bigl(d_{c} + d_{h}^{R}\bigr) l \approx \tfrac{9}{2} \, d_{h} \, l\) | Stronger |

The figure below shows that MLA not only significantly reduces the KV cache but also achieves performance superior to standard MHA.

{{< figure
    src="mla_vs_mha.png"
    caption="Fig. 5. Comparison between MLA and MHA on hard benchmarks. DeepSeek-V2 shows better performance than MHA, but requires a significantly smaller amount of KV cache. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

### Mixture-of-Experts Models

Before diving into DeepSeekMoE, let's review the basics of Mixture-of-Experts (MoE) models.

**Mixture-of-Experts (MoE)** ([Shazeer et al. 2017](https://arxiv.org/abs/1701.06538)) is a sparsely activated model that significantly increases model parameter count and performance without substantially increasing computational cost by combining multiple independent "expert" networks and a gating network. The core idea of MoE is **Sparse Activation**, meaning that for each input sample, only a subset of expert networks is activated, rather than the entire model. This approach enhances both computational efficiency and the model's expressive power, leading to excellent performance in LLMs.

MoE design is inspired by [Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning), a technique that decomposes complex tasks into multiple subtasks handled collaboratively by different models. In MoE, these "subtasks" are processed by multiple independent expert networks, while a gating network dynamically selects the most suitable experts based on the input sample's features. This division of labor resembles expert teams in human society: specialists from different fields provide expertise on specific problems, and their insights are combined to reach a final result.

{{< figure
    src="moe.png"
    caption="Fig. 6. Illustration of a mixture-of-experts (MoE) layer. Only 2 out of experts are selected and activated by the gating network. (Image source: [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538))"
    align="center"
    width="100%"
>}}

### Core MoE Components

A typical MoE layer includes the following components:

*   **Experts:** A set of independent neural networks $\{E_1, E_2, ..., E_n\}$. Each expert network $E_i$ can be any type of neural network, such as an FFN, CNN, RNN, etc. The number of experts $n$ can be large, e.g., tens, hundreds, or even thousands.
*   **Gating Network:** A trainable neural network $G$ that learns a probability distribution based on the input sample $x$ to decide which experts to activate. The gating network takes the input sample $x$ and outputs an $n$-dimensional probability vector $p = G(x) = [p_1, p_2, ..., p_n]$, where $p_i$ represents the probability of activating expert $E_i$.
*   **Expert Output Aggregation:** Based on the probability distribution from the gating network, the outputs of the activated expert networks are weighted and summed to produce the final output $y$ of the MoE layer.

### Noisy Top-k Gating

To achieve sparse activation and ensure balanced expert utilization, MoE typically employs **Noisy Top-k Gating** as the gating mechanism. This method introduces noise and top-k selection to ensure computational efficiency while preventing uneven expert load. Here's the detailed workflow:

1.  **Gating Score Calculation:**

    For an input sample $x$, the gating network first computes a gating score $H^{(i)}(x)$ for each expert. This score consists of a linear transformation and a noise term, formulated as:

    $$
    H^{(i)}(x) =(x W_g)^{(i)} + \epsilon \cdot \text{softplus}\left((x W_{\text{noise}})^{(i)} \right), \quad \epsilon \sim \mathcal{N}(0, 1)
    $$

    -   **Parameters:**
        -   $W_g \in \mathbb{R}^{d \times n}$: Trainable weight matrix of the gating network, where $d$ is the input feature dimension and $n$ is the number of experts.
        -   $W_{\text{noise}} \in \mathbb{R}^{d \times n}$: Weight matrix used to generate noise.
        -   $\epsilon \sim \mathcal{N}(0, 1)$: Standard Gaussian noise, adding randomness to the gating.
        -   $\text{softplus}(x) = \log(1 + e^x)$: Smooth activation function ensuring non-negative noise.

    The introduction of noise prevents the gating network from always selecting the same experts, enhancing the model's robustness and diversity.

2.  **Top-k Selection:**

    After computing the gating score vector $H(x) = [H^{(1)}(x), H^{(2)}(x), \dots, H^{(n)}(x)]$, the gating network selects the top $k$ experts with the highest scores (usually $k \ll n$). This step is implemented using the $\text{topk}(v, k)$ function:

    $$
    \text{topk}^{(i)}(v, k) =
    \begin{cases}
    v^{(i)} & \text{if } v^{(i)} \text{ is in the top } k \text{ elements of } v \\
    -\infty & \text{otherwise}
    \end{cases}
    $$

    Setting the scores of non-top-k experts to $-\infty$ ensures their probabilities become 0 after the subsequent softmax operation, achieving sparsity.

3.  **Softmax Normalization:**

    The gating scores of the top-k experts are normalized using softmax to obtain a sparse probability distribution $G(x)$:

    $$
    G(x) = \text{softmax}\left( \text{topk}(H(x), k) \right)
    $$

    Only the top-k experts have non-zero probabilities; the rest are 0. For example, if $n=100, k=2$, then 98 experts will have a probability of 0.

4.  **Weighted Sum:**

    The outputs of the top-k experts are weighted by their probabilities and summed to get the MoE layer's output:

    $$
    y = \sum_{i=1}^{n} G^{(i)}(x) E_i(x)
    $$

    Since only $k$ experts are activated, the computational load is much lower than activating all $n$ experts.

### Auxiliary Loss

To **prevent the gating network from overly favoring a few experts**, MoE introduces an **Auxiliary Loss** ([Shazeer et al. 2017](https://arxiv.org/abs/1701.06538)) to encourage uniform usage of all experts. A common method is based on the square of the [Coefficient of Variation (CV)](https://en.wikipedia.org/wiki/Coefficient_of_variation) of expert usage:

$$
\mathcal{L}_{\text{aux}} = w_{\text{aux}} \cdot \text{CV}\left( \sum_{x \in X} G(x) \right)^2
$$

-   **Parameters:**
    -   $X$: A mini-batch of input samples.
    -   $\sum_{x \in X} G(x)$: Counts the number of times each expert is activated within the mini-batch.
    -   $\text{CV}$: The ratio of the standard deviation to the mean, measuring the uniformity of expert usage distribution.
    -   $w_{\text{aux}}$: Weight of the auxiliary loss, needs manual tuning.

-   **Purpose:** By minimizing $\mathcal{L}_{\text{aux}}$, the model optimizes the balance of expert selection, preventing some experts from being overused while others remain idle.

### GShard

**GShard** ([Lepikhin et al. 2020](https://arxiv.org/abs/2006.16668)) primarily focuses on sharding the MoE layer, distributing the expert networks $\{E_1, E_2, ..., E_n\}$ across multiple TPU devices. For instance, with $P$ TPU devices, the experts can be divided into $P$ groups, each assigned to one TPU device. Other layers of the Transformer model (e.g., self-attention, LayerNorm) are replicated across all TPU devices.

**GShard's Improved Gating Mechanism:**

GShard builds upon Noisy Top-k Gating with several improvements to enhance performance and stability:

-   **Expert Capacity:**
    To prevent expert overload, GShard introduces expert capacity limits. Each expert network has a maximum capacity, indicating the maximum number of tokens it can process. If a token is routed to an expert that has reached its capacity limit, the token is marked as "overflowed," and its gating output is set to a zero vector, meaning it won't be routed to any expert.

-   **Local Group Dispatching:**
    To improve gating efficiency, GShard groups tokens and enforces expert capacity limits at the group level. For example, tokens in a mini-batch are divided into multiple local groups, each containing a certain number of tokens. The gating network selects top-k experts for each local group, ensuring that the number of tokens processed by each expert within a group does not exceed its capacity limit.

-   **Auxiliary Loss:**
    GShard also uses an auxiliary loss function to balance expert load. Unlike the original MoE model's auxiliary loss, GShard's loss aims to minimize the mean squared error of the proportion of data routed to each expert, more directly measuring expert load balance.

-   **Random Routing:**
    To increase routing randomness, GShard introduces a random routing mechanism when selecting the top-k experts. Besides selecting the best top-k experts, GShard also randomly selects sub-optimal experts with a certain probability, increasing expert diversity and improving the model's generalization ability.

Below is the core algorithm flow of GShard:

{{< figure
    src="gshard.png"
    caption="Fig. 7. Pseudo code of the group-level top-2 gating mechanism with auxiliary loss in GShard. (Image source: [Lepikhin et al. 2020](https://arxiv.org/abs/2006.16668))"
    align="center"
    width="100%"
>}}

### Switch Transformer

**Switch Transformer** ([Fedus et al. 2021](https://arxiv.org/pdf/2101.03961)) is a **trillion-parameter** MoE model proposed by Google. Its core innovation is replacing the dense feed-forward network (FFN) layers in the Transformer model with sparse Switch FFN layers. Unlike GShard's Top-2 Gating, Switch Transformer routes each input token to only one expert network, achieving higher sparsity and further reducing computational costs, making it possible to train trillion-parameter models. It encourages more balanced token routing among the $N$ experts. Switch Transformer's auxiliary loss is based on the product sum of the actual routing fraction and the predicted routing probability, formulated as:

$$
\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

-   **Parameters:**
    -   $N$: Total number of experts.
    -   $f_i$: Fraction of tokens routed to the $i$-th expert, defined as:
        $$
        f_i = \frac{1}{T} \sum_{x \in B} \mathbb{1}\{\text{argmax } p(x) = i\}
        $$
    -   $P_i$: Routing probability for the $i$-th expert predicted by the gating network, defined as:
        $$
        P_i = \frac{1}{T} \sum_{x \in B} p_i(x)
        $$
    -   $T$: Total number of tokens in batch $B$.
    -   $\alpha$: Weight hyperparameter for the auxiliary loss, typically set to $10^{-2}$.

By minimizing this loss, the model encourages the actual routing fraction $f_i$ to align with the predicted probability $P_i$, indirectly promoting load balance among experts and preventing some from being idle.

{{< figure
    src="switch_transformer.png"
    caption="Fig. 8. Switch transformer. The sparse switch FFN layer is in the blue boxes. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}

**Switch Router Mechanism:**

1.  **Routing Prediction:**
    For an input token $x$, the Switch Router predicts the routing probability $p_i = G^{(i)}(x)$ for each expert network, where $i = 1, 2, ..., n$, and n is the number of expert networks.

2.  **Expert Selection:**
    Select the expert network with the highest routing probability as the best expert. Switch Transformer uses a Top-1 routing strategy, meaning each token is routed only to the expert with the highest probability.

3.  **Token Routing:**
    Route the input token $x$ to the selected best expert network for processing.

**Switch Transformer Training Stability Optimizations:**

To improve the training stability of Switch Transformer, the paper proposes the following optimization strategies:

-   **Selective Precision:**
    Using FP32 precision inside the router function improves training stability without the overhead of FP32 tensor communication. Specifically, the Switch Router computations are performed entirely in FP32, and the final result is converted back to FP16 to balance efficiency and precision.

-   **Smaller Initialization:**
    It is recommended to adjust the Transformer weight initialization scale parameter $s$ from 1.0 to 0.1. A smaller initialization scale helps mitigate the risk of gradient explosion early in training, thereby improving overall training stability. This is implemented by sampling from a truncated normal distribution with mean 0 and standard deviation $\sqrt{s/n}$ (where $n$ is the number of input units).

-   **Higher Expert Dropout:**
    Using a higher dropout rate (e.g., 0.4) in the expert FFN layers while maintaining a lower dropout rate (e.g., 0.1) in non-expert layers effectively prevents overfitting and enhances the model's generalization ability. The experimental results in the figure below show that the model performs best on tasks like GLUE, CNNDM, SQuAD, and SuperGLUE when the expert layer dropout rate is set to 0.4.

{{< figure
    src="switch_transformer_fine_tuning_result.png"
    caption="Fig. 9. Fine-tuning regularization results. A sweep of dropout rates while fine-tuning Switch Transformer models pre-trained on 34B tokens of the C4 data set (higher numbers are better). (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}

The Switch Transformers paper uses the following figure to intuitively illustrate how different parallelism techniques partition model weights and data:

{{< figure
    src="switch_transformer_parallelism.png"
    caption="Fig. 10. An illustration of various parallelism strategies on how (Top) model weights and (Bottom) data are split over multiple GPU cores. In the top row, each color denotes a unique weight matrix. In the bottom row, different colors indicate different sets of tokens. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}

### Expert Choice

**Expert Choice (EC)** ([Zhou et al. 2022](https://arxiv.org/abs/2202.09368)) is a routing strategy opposite to token choice routing (like GShard's top-2 or Switch Transformer's top-1). In token choice routing, each token selects top-k experts from all available experts. In expert choice routing, each expert selects top-k tokens from all available tokens to process. This approach aims to address the load imbalance and token dropping issues of token choice routing while significantly improving training efficiency. Here is the specific computation process:

1.  **Compute token-to-expert affinity scores:**

    For an input matrix $X \in \mathbb{R}^{n \times d}$, the token-to-expert affinity score matrix $S \in \mathbb{R}^{n \times e}$ is computed as:

    $$
    S = \text{softmax}(X \cdot W_g), \quad \text{where } W_g \in \mathbb{R}^{d \times e}.
    $$
    Here, $W_g$ is the gating weight matrix, and $e$ is the number of experts.

2.  **Experts select tokens:**

    Each expert selects the top-k tokens from all tokens to process. This is done by performing top-k selection on $S^T$:

    $$
    G, I = \text{top-}k(S^T, k),
    $$

    This yields:
    -   **Gating matrix $G \in \mathbb{R}^{e \times k}$:** Records the routing weights corresponding to the tokens selected by the experts, where $G[i, j]$ is the weight for the $j$-th token selected by expert $i$.
    -   **Token index matrix $I \in \mathbb{R}^{e \times k}$:** Indicates the indices of the tokens selected by each expert in the input.

3.  **One-hot encoding:**

    Convert the token index matrix $I$ into a one-hot encoded matrix $P \in \mathbb{R}^{e \times k \times n}$ for subsequent calculations:

    $$
    P = \operatorname{one}-\operatorname{hot}(I)
    $$

4.  **Construct input for Gated FFN layer:**

    For each expert $i$, the input to its gated FFN layer is:

    $$
    (P \cdot X) \in \mathbb{R}^{e \times k \times d}.
    $$

EC controls model sparsity by regularizing the number of experts each token is routed to. A common regularization objective is:

$$
\begin{aligned}
& \max_{A} \langle S^{\top}, A \rangle + \lambda H(A) \\
& \text{s.t. } \forall i: \sum_{j'} A[i, j'] = k, \quad \forall j: \sum_{i'} A[i', j] \leq b, \quad \forall i,j: 0 \leq A[i, j] \leq 1,
\end{aligned}
$$

The optimization problem defines a matrix $A$ where the element at row $i$, column $j$ indicates whether expert $i$ selected token $j$ (value 0 or 1). Since solving this optimization problem is complex, the paper uses [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) (obtaining an approximate solution through multiple iterations).

The parameter $b$ is typically determined by the total number of tokens $n$ in the batch and a capacity factor, which represents the average number of experts used per token. Most experiments use a high capacity factor. Experimental results show that even with reduced capacity, EC generally outperforms traditional top-1 token choice routing, although capped expert choice slightly degrades fine-tuning performance.

The advantages of EC are mainly twofold:
-   **Perfect Load Balancing:** Each expert processes a fixed $k$ tokens, avoiding the issue of some experts being overloaded while others are idle, achieving ideal load balance.
-   **Higher Training Efficiency:** Experiments show that EC can improve training convergence speed by about 2x, making it more efficient than traditional token choice routing.

However, EC also has limitations:
-   **Batch Size Requirement:** EC requires a relatively large batch size, making it unsuitable for scenarios with smaller batch sizes.
-   **Autoregressive Generation Limitation:** In autoregressive text generation tasks, EC's top-k selection cannot be implemented because future tokens are unknown, making it unsuitable for such tasks.

### DeepSeekMoE

Mixture-of-Experts (MoE) models enhance efficiency and performance by routing computation to specific "expert" subnetworks. DeepSeek-V2 and V3 employ an architecture named **DeepSeekMoE** ([Dai et al., 2024](https://arxiv.org/abs/2401.06066)) in their FFN (Feed-Forward Network) layers. Compared to traditional MoE architectures like GShard, the core ideas of DeepSeekMoE are:

1.  **Fine-grained Expert Segmentation:** Splitting expert networks into smaller units. This aims for higher expert specialization and more precise knowledge acquisition, as each expert can focus on a narrower domain.
2.  **Shared Expert Isolation:** The architecture includes a set of "shared experts" processed by all tokens, intended to handle general knowledge. This reduces knowledge redundancy among the "routing experts" that need to be selected, allowing them to focus more on specific knowledge.

#### Basic Architecture

For an input token representation \(\mathbf{u}_t\) to the FFN layer, the output \(\mathbf{h}_t'\) of DeepSeekMoE is computed by combining the outputs of shared experts and selected routing experts:
\[
\mathbf{h}_{t}^{\prime} = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} \operatorname{FFN}_{i}^{(s)}(\mathbf{u}_{t}) + \sum_{i=1}^{N_{r}} g_{i, t} \operatorname{FFN}_{i}^{(r)}(\mathbf{u}_{t}),
\]
where:
*   \(N_s\) is the number of shared experts.
*   \(N_r\) is the number of routing experts.
*   \(\operatorname{FFN}_i^{(s)}\) is the \(i\)-th shared expert network.
*   \(\operatorname{FFN}_i^{(r)}\) is the \(i\)-th routing expert network.
*   \(g_{i, t}\) is the gating value (weight) assigned to the \(i\)-th routing expert for the \(t\)-th token.

The calculation of the gating value \(g_{i,t}\), based on token-to-expert affinity scores \(s_{i,t}\) and selected via a Top-K routing mechanism, is one of the key differences between DeepSeek-V2 and V3.

#### V2 vs V3 Gating Mechanism and Load Balancing Comparison

A core challenge in MoE models is load balancing: ensuring all experts are effectively utilized, avoiding situations where some experts are overloaded while others are idle, which affects training stability and computational efficiency. DeepSeek-V2 and V3 adopt different approaches to gating mechanisms and load balancing strategies.

**1. Affinity Calculation (\(s_{i,t}\)) and Top-K Selection:**

*   **DeepSeek-V2:** Uses the Softmax function to compute the affinity score of each token for each routing expert. Top-K selection is directly based on these affinity scores \(s_{i,t}\).
    \[
    s_{i, t} = \operatorname{Softmax}_{i}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})
    \]
    where \(\mathbf{e}_i\) is the learnable center vector for the \(i\)-th routing expert. The \(K_r\) experts with the highest \(s_{i,t}\) are selected.

*   **DeepSeek-V3:** Uses the Sigmoid function to compute affinity scores. More importantly, it introduces a learnable bias term \(b_i\) for each routing expert. Top-K selection is based on the **bias-adjusted affinity** \(s_{i,t} + b_i\).
    \[
    s_{i, t} = \operatorname{Sigmoid}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})
    \]
    Selection is based on the \(K_r\) experts with the highest \(s_{i,t} + b_i\) values.

**2. Gating Value Calculation (\(g_{i,t}\)):**

*   **DeepSeek-V2:** For experts selected by Top-K, their gating value \(g_{i,t}\) is directly equal to their original affinity score \(s_{i,t}\). For unselected experts, \(g_{i,t} = 0\).
    \[
    g_{i, t}^{\prime} = \begin{cases} s_{i, t}, & s_{i, t} \in \operatorname{Topk}(\{s_{j, t}\}, K_{r}), \\ 0, & \text{otherwise}, \end{cases}
    \]
    \[
    g_{i, t} = g_{i, t}^{\prime} \quad (\text{No additional normalization in V2})
    \]

*   **DeepSeek-V3:** For experts selected based on \(s_{i,t} + b_i\), their gating value \(g_{i,t}\) is obtained by normalizing the **original affinity scores** \(s_{i,t}\) of these selected experts. The bias \(b_i\) is **only used for routing selection** and does not affect the final weighted sum.
    \[
    g_{i, t}^{\prime}= \begin{cases} s_{i, t}, & s_{i, t}+b_{i} \in \operatorname{Topk}\left(\left\{s_{j, t}+b_{j} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right) \\ 0, & \text{otherwise.} \end{cases}
    \]
    \[
    g_{i, t} = \frac{g_{i, t}^{\prime}}{\sum_{j=1}^{N_{r}} g_{j, t}^{\prime}} \quad (\text{Normalize affinities of selected experts})
    \]

**3. Load Balancing Strategy:**

*   **DeepSeek-V2:**
    *   **Primary Strategy: Auxiliary Losses** V2 introduces multiple auxiliary loss terms to explicitly encourage load balancing:
        *   **Expert-level Balancing Loss (\(\mathcal{L}_{\text{ExpBal}}\)):** Encourages each expert to process roughly the same number of tokens.
            \[
            \begin{aligned}
            \mathcal{L}_{\text{ExpBal}} &= \alpha_{1} \sum_{i=1}^{N_{r}} f_{i} P_{i} \\
            f_{i} &= \frac{N_{r}}{K_{r} T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i) \\
            P_{i} &= \frac{1}{T} \sum_{t=1}^{T} s_{i, t}
            \end{aligned}
            \]
            where \(T\) is the total number of tokens in the batch, \(f_i\) is the fraction of tokens routed to expert \(i\) (relative to the ideal balanced state), \(P_i\) is the average affinity score for expert \(i\), and \(\alpha_1\) is a hyperparameter.
        *   **Device-level Balancing Loss (\(\mathcal{L}_{\text{DevBal}}\)):** Encourages uniform distribution of computational load across different device groups (assuming experts are distributed across \(D\) device groups \(\{\mathcal{E}_1, \dots, \mathcal{E}_D\}\)).
            \[
            \begin{aligned}
            \mathcal{L}_{\text{DevBal}} &= \alpha_{2} \sum_{i=1}^{D} f_{i}^{\prime} P_{i}^{\prime} \\
            f_{i}^{\prime} &= \frac{1}{|\mathcal{E}_{i}|} \sum_{j \in \mathcal{E}_{i}} f_{j} \\
            P_{i}^{\prime} &= \sum_{j \in \mathcal{E}_{i}} P_{j}
            \end{aligned}
            \]
            where \(f_i'\) is the average load score for device group \(i\), \(P_i'\) is the total affinity for device group \(i\), and \(\alpha_2\) is a hyperparameter.
        *   **Communication Balancing Loss (\(\mathcal{L}_{\text{CommBal}}\)):** Encourages roughly equal numbers of tokens sent to each device to balance All-to-All communication load.
            \[
            \begin{aligned}
            \mathcal{L}_{\text{CommBal}} &= \alpha_{3} \sum_{i=1}^{D} f_{i}^{\prime \prime} P_{i}^{\prime \prime} \\
            f_{i}^{\prime \prime} &= \frac{D}{M T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ is sent to Device } i) \\
            P_{i}^{\prime \prime} &= \sum_{j \in \mathcal{E}_{i}} P_{j}
            \end{aligned}
            \]
            where \(f_i''\) is the fraction of tokens sent to device \(i\) (relative to the ideal balanced state), \(P_i''\) is the total affinity for device group \(i\), and \(\alpha_3\) is a hyperparameter.
    *   **Routing Restriction: Device-Limited Routing** Limits each token to route to experts distributed on at most \(M\) different devices. In V2, \(M=3\).
    *   **Token Dropping:** During training, if a device receives more tokens than a preset capacity factor (usually slightly above the average), some tokens with the lowest routing weights (affinities) are dropped to avoid wasting computational resources. However, tokens from about 10% of sequences are preserved from dropping.

*   **DeepSeek-V3:**
    *   **Primary Strategy: Auxiliary-Loss-Free Load Balancing** V3 posits that auxiliary losses can harm model performance and thus adopts an innovative **Auxiliary-Loss-Free Load Balancing** ([Wang et al., 2024](https://arxiv.org/abs/2408.15664)). It achieves load balancing by dynamically adjusting the aforementioned learnable bias terms \(b_i\):
        *   **Bias Update:** After each training step, monitor the number of tokens processed by each expert \(i\) in the current batch.
            *   If expert \(i\) is overloaded (processed tokens > Total batch tokens / \(N_r\)), decrease its bias: \(b_i \leftarrow b_i - \gamma\).
            *   If expert \(i\) is underloaded (processed tokens < Total batch tokens / \(N_r\)), increase its bias: \(b_i \leftarrow b_i + \gamma\).
        *   \(\gamma\) is a small positive step size (bias update rate hyperparameter). This way, highly loaded experts become less likely to be selected in subsequent routing, while lowly loaded experts become more likely, dynamically balancing the load at the batch level.
    *   **Supplementary Strategy: Sequence-Level Auxiliary Loss (\(\mathcal{L}_{\text{Bal}}\))** V3 still retains an auxiliary loss with an **extremely small weight** (\(\alpha=0.0001\)), but it acts on the expert selection balance **within individual sequences**, rather than the entire batch. This is mainly to prevent extreme imbalance within a single sequence.
        \[
        \begin{gathered}
        \mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_{r}} f_{i} P_{i}, \\
        f_{i} = \frac{N_{r}}{K_{r} T_{seq}} \sum_{t=1}^{T_{seq}} \mathbb{1}\left(s_{i, t} \in \operatorname{Topk}\left(\left\{s_{j, t} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right)\right), \\
        s_{i, t}^{\prime} = \frac{s_{i, t}}{\sum_{j=1}^{N_{r}} s_{j, t}}, \quad P_{i} = \frac{1}{T_{seq}} \sum_{t=1}^{T_{seq}} s_{i, t}^{\prime}
        \end{gathered}
        \]
        Note that \(f_i, P_i\) here are computed over a single sequence (length \(T_{seq}\)), and \(s_{i,t}'\) is the value of original \(s_{i,t}\) normalized within the sequence.
    *   **Routing Restriction: Node-Limited Routing** Similar to V2's device limit, but applied at the node level. In V3, \(M=4\).
    *   **No Token Dropping:** Due to the effectiveness of bias-adjustment-based load balancing, V3 does not drop any tokens during training or inference.

**Advantages of V3's Strategy:**
V3's auxiliary-loss-free strategy aims to minimize the negative impact of the load balancing mechanism on the final model performance. By dynamically adjusting bias terms for batch-level load balancing, the constraints are looser compared to V2's sequence-level balancing based on auxiliary losses. This allows experts to exhibit stronger specialization patterns across different domains, as routing decisions do not need to strictly follow a balanced distribution within each sequence. The figure below shows experimental results indicating this strategy outperforms auxiliary-loss-based methods on multiple benchmarks.

{{< figure
    src="auxiliary_loss_free_result.png"
    caption="Fig. 11. Ablation results for the auxiliary-loss-free balancing strategy. Compared with the purely auxiliary-loss-based method, the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

The key difference between auxiliary-loss-free load balancing and sequence-level auxiliary loss lies in their balancing scope: batch-level versus sequence-level. Compared to sequence-level auxiliary loss, batch-level balancing imposes more flexible constraints as it does not enforce domain balance within each sequence. This flexibility allows experts to specialize better across different domains. To validate this, the figure records and analyzes the expert load on different domains of the Pile test set for a 16B baseline model with auxiliary loss and a 16B model without auxiliary loss. It can be observed that the auxiliary-loss-free model exhibits more pronounced expert specialization patterns, as expected.

{{< figure
    src="expert_load.png"
    caption="Fig. 12. Expert load of auxiliary-loss-free and auxiliary-loss-based models on three domains in the Pile test set. The auxiliary-loss-free model shows greater expert specialization patterns than the auxiliary-loss-based one. The relative expert load denotes the ratio between the actual expert load and the theoretically balanced expert load. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

---

#### DeepSeekMoE V2 vs V3 Comparison Summary Table

| Feature                     | DeepSeek-V2                                                                                                                                                              | DeepSeek-V3                                                                                                                                                                                             |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Affinity Calculation \(s_{i,t}\)** | \(\operatorname{Softmax}_{i}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})\)                                                                                                         | \(\operatorname{Sigmoid}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})\)                                                                                                                                           |
| **TopK Selection Basis**    | Original affinity \(s_{i,t}\)                                                                                                                                             | Bias-adjusted affinity \(s_{i,t} + b_i\)                                                                                                                                                                |
| **Gating Value Calc. \(g_{i,t}\)** | For selected experts, \(g_{i,t} = s_{i,t}\) (Usually no extra normalization)                                                                                             | For selected experts, normalize based on original affinity \(s_{i,t}\): \(g_{i, t} = \frac{s_{i, t}}{\sum_{j \in \text{Selected}} s_{j, t}}\)                                                                     |
| **Primary Load Balancing**  | **Auxiliary Losses:** <br> - \(\mathcal{L}_{\text{ExpBal}}\) (Expert-level) <br> - \(\mathcal{L}_{\text{DevBal}}\) (Device-level) <br> - \(\mathcal{L}_{\text{CommBal}}\) (Comm-level) | **Auxiliary-Loss-Free:** <br> - Dynamic adjustment of learnable bias \(b_i\) (step \(\gamma\)) for batch-level balancing                                                                                    |
| **Supplementary Balancing** | No explicit supplementary strategy                                                                                                                                       | **Sequence-Level Aux Loss** \(\mathcal{L}_{\text{Bal}}\) (Weight \(\alpha\) minimal, e.g., 0.0001), prevents extreme imbalance within single sequences                                                     |
| **Routing Restriction**     | **Device Limit:** <br> Each token routes to experts on at most \(M=3\) devices                                                                                             | **Node Limit:** <br> Each token routes to experts on at most \(M=4\) nodes                                                                                                                             |
| **Token Dropping**          | **Yes:** During training, tokens exceeding device capacity with lowest affinity are dropped (preserving ~10% sequences) to mitigate bottlenecks                               | **No:** No tokens dropped during training or inference                                                                                                                                                  |
| **Balancing Granularity**   | Primarily enforced at sequence/batch level via auxiliary losses                                                                                                            | Primarily balanced dynamically at batch level via bias adjustment, looser constraints                                                                                                                   |
| **Impact on Performance**   | Auxiliary losses might negatively impact model performance                                                                                                               | Designed to minimize negative impact of balancing strategy on performance, allowing better expert specialization                                                                                        |

### Multi-Token Prediction (MTP)

To further enhance model performance and data efficiency, DeepSeek-V3 introduces the **Multi-Token Prediction (MTP)** training objective (inspired by [Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)). Standard language models only predict the next token, whereas MTP makes the model predict multiple future tokens (in V3, \(D_{MTP}=1\), i.e., predicting the token after the next) at each position.

#### MTP Implementation

MTP is implemented through \(D_{MTP}\) sequential modules. The \(k\)-th MTP module (\(k=1, \dots, D_{MTP}\)) contains:
*   A shared embedding layer \(\operatorname{Emb}(\cdot)\)
*   A shared output head \(\operatorname{OutHead}(\cdot)\)
*   Independent Transformer blocks \(\operatorname{TRM}_k(\cdot)\)
*   Independent projection matrices \(M_k \in \mathbb{R}^{d \times 2d}\)

For the \(i\)-th token \(t_i\) in the input sequence, at the \(k\)-th prediction depth:
1.  Concatenate the representation of the \(i\)-th token at depth \(k-1\), \(\mathbf{h}_i^{k-1}\) (which is the main model output when \(k=1\)), with the embedding of the \((i+k)\)-th token, \(\operatorname{Emb}(t_{i+k})\). Project this concatenation through matrix \(M_k\) to get the combined representation \(\mathbf{h}_i^{\prime k}\):
    \[
    \mathbf{h}_{i}^{\prime k} = M_{k}[\operatorname{RMSNorm}(\mathbf{h}_{i}^{k-1}) ; \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]
    \]
2.  Input the combined representation into the \(k\)-th Transformer block to get the output representation \(\mathbf{h}_i^k\) for the current depth:
    \[
    \mathbf{h}_{1: T-k}^{k} = \operatorname{TRM}_{k}(\mathbf{h}_{1: T-k}^{\prime k})
    \]
3.  Use the shared output head to predict the probability distribution \(P_{i+k+1}^k \in \mathbb{R}^V\) for the \((i+k+1)\)-th token:
    \[
    P_{i+k+1}^{k} = \operatorname{OutHead}(\mathbf{h}_{i}^{k})
    \]

Crucially, this implementation **maintains the complete causal chain for each prediction depth**, differing from methods that predict multiple tokens in parallel.

{{< figure
    src="mtp.png"
    caption="Fig. 13. Illustration of our Multi-Token Prediction (MTP) implementation. They keep the complete causal chain for the prediction of each token at each depth. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

#### MTP Training Objective

Compute the cross-entropy loss \(\mathcal{L}_{\text{MTP}}^k\) for each prediction depth \(k\):
\[
\mathcal{L}_{\text{MTP}}^{k} = \operatorname{CrossEntropy}(P_{2+k: T+1}^{k}, t_{2+k: T+1}) = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_{i}^{k}[t_{i}]
\]
The total MTP loss is the weighted average of losses across all depths:
\[
\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D_{MTP}} \sum_{k=1}^{D_{MTP}} \mathcal{L}_{\text{MTP}}^{k}
\]
where \(\lambda\) is a weighting factor (0.3 initially, 0.1 later in V3 training). This loss is added to the main model's standard next-token prediction loss.

#### MTP Inference

MTP is primarily used to enhance the main model's performance. During inference, the **MTP modules can be simply discarded**, and the main model works independently. Alternatively, the MTP modules can be utilized for **speculative decoding** ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Xia et al., 2023](https://arxiv.org/abs/2203.16487)) to accelerate the generation process. V3 experiments show an acceptance rate of 85%-90% for the second token, potentially speeding up decoding by about 1.8x.

## Infrastructure and Training Efficiency

The efficient training and deployment of DeepSeek-V3 benefit from the synergistic design of algorithms, frameworks, and hardware.

### Compute Cluster

DeepSeek-V3 was trained on a cluster equipped with **2048 NVIDIA H800 GPUs**.
*   **Intra-node:** Each node contains 8 H800 GPUs interconnected via high-speed **NVLink** and **NVSwitch**.
*   **Inter-node:** Different nodes communicate using the **InfiniBand (IB)** network.

### Training Framework

DeepSeek-V3 training is based on the self-developed, efficient, and lightweight framework **HAI-LLM**. Overall, it employs:
*   **16-way Pipeline Parallelism (PP)** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241))
*   **64-way Expert Parallelism (EP)** (across 8 nodes) ([Lepikhin et al., 2021](https://arxiv.org/abs/2006.16668))
*   **ZeRO-1 Data Parallelism (DP)** ([Rajbhandari et al., 2020](https://arxiv.org/pdf/1910.02054))

To achieve efficient training, Deepseek performed meticulous engineering optimizations:
1.  Designed the **DualPipe** algorithm for efficient pipeline parallelism, reducing bubbles and overlapping computation with communication, addressing the heavy communication overhead introduced by cross-node expert parallelism.
2.  Developed efficient **cross-node All-to-all communication Kernels** that fully utilize IB and NVLink bandwidth while saving SM resources for communication.
3.  Carefully optimized **memory usage** during training, enabling DeepSeek-V3 to be trained **without using Tensor Parallelism (TP)**.

#### DualPipe and Computation-Communication Overlap

*   **Challenge:** Cross-node expert parallelism leads to a computation-to-communication ratio close to 1:1, which is inefficient.

{{< figure
    src="forward_backward_chucks.png"
    caption="Fig. 17. Overlapping strategy for a pair of forward and backward chunks with misaligned transformer block boundaries. Orange: forward, green: backward for input, blue: backward for weights, purple: PP communication, red: barriers. Both all-to-all and PP communications are fully hidden. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

*   **Core Idea:** Overlap computation and communication within a pair of independent forward and backward chunks. Each chunk is decomposed into four components: **Attention**, **All-to-all Dispatch**, **MLP**, **All-to-all Combine** (backward Attention and MLP are further divided into backward for input and backward for weights, similar to **ZeroBubble** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241)). By rearranging these components and manually adjusting the GPU SM ratio for communication versus computation, both All-to-all and PP communication can be fully hidden.
*   **Scheduling:** Adopts a bidirectional pipeline schedule, feeding micro-batches from both ends of the pipeline simultaneously, allowing most communication to be completely overlapped.

{{< figure
    src="dualpipe.png"
    caption="Fig. 18. Example DualPipe scheduling with 8 PP ranks and 20 micro-batches in both directions. The reverse-direction micro-batches mirror the forward ones, so their batch IDs are omitted for simplicity. Two cells within a shared black border represent mutually overlapped computation and communication. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

*   **Advantages:**
    *   Efficient even in general scenarios without heavy communication burden.
    *   Compared to **ZB1P** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241)) and **1F1B** ([Harlap et al., 2018](https://arxiv.org/abs/1806.03377)), significantly reduces pipeline bubbles, only increasing peak activation memory by a factor of \(\frac{1}{PP}\).
    *   Although requiring two copies of model parameters, the memory increase is not significant due to the large EP size used in training.
    *   Compared to **Chimera** ([Li and Hoefler, 2021](https://dl.acm.org/doi/10.1145/3458817.3476145)), has looser requirements on the number of micro-batches (only needs to be divisible by 2), and bubble/activation memory does not grow with the number of micro-batches.

| Method (Method)     | Bubble (Bubble)                                                    | Parameter (Parameter) | Activation (Activation) |
| :---------------- | :--------------------------------------------------------------- | :--------------- | :---------------- |
| 1F1B              | \((PP-1)(F+B)\)                                                  | \(1 \times\)     | \(PP\)            |
| ZB1P              | \((PP-1)(F+B-2W)\)                                               | \(1 \times\)     | \(PP\)            |
| **DualPipe (Deepseek V3)** | \(\left(\frac{PP}{2}-1\right)(F\&B+B-3W)\)                        | \(2 \times\)     | \(PP+1\)          |

The table above compares pipeline bubble and memory usage for different pipeline parallelism methods. \(F\): Forward chunk execution time; \(B\): Full backward chunk execution time; \(W\): "Backward for weights" chunk execution time; \(F\&B\): Execution time of two mutually overlapped forward and backward chunks.

#### Efficient Cross-Node All-to-All Communication Implementation

*   **Goal:** Provide sufficient computational performance for DualPipe by customizing efficient cross-node All-to-all communication Kernels (dispatching & combining), saving SMs dedicated to communication.
*   **Strategy:** Combine MoE gating algorithm with cluster network topology (fully connected IB between nodes, NVLink within nodes).
    *   **Bandwidth Utilization:** NVLink bandwidth (\(160 \mathrm{~GB} / \mathrm{s}\)) is about 3.2 times IB bandwidth (\(50 \mathrm{~GB} / \mathrm{s}\)). Limit each token to be dispatched to at most **4 nodes** to reduce IB traffic.
    *   **Transmission Path:** After token routing is determined, tokens are first transmitted via **IB** to the GPU with the same intra-node index on the target node. Upon arrival at the target node, they are immediately forwarded via **NVLink** to the specific GPU hosting the target expert, avoiding blockage by subsequently arriving tokens.
    *   **Effect:** IB and NVLink communication are fully overlapped. Each token can efficiently select an average of **3.2 experts/node** without additional NVLink overhead. This means V3 actually selects 8 routing experts, but could theoretically scale up to **13 experts** (4 nodes × 3.2 experts/node) with no increase in communication cost.
*   **Implementation:**
    *   Use **Warp Specialization** ([Bauer et al., 2014](https://doi.org/10.1145/2555243.2555258)) technology to divide **20 SMs** into 10 communication channels.
    *   Dispatch process: IB send, IB-to-NVLink forward, NVLink receive are handled by respective warps, with warp counts dynamically adjusted based on load.
    *   Combine process: NVLink send, NVLink-to-IB forward & accumulate, IB receive & accumulate are also handled by dynamically adjusted warps.
    *   **Optimization:** Dispatch and Combine Kernels overlap with computation streams. Use custom **PTX** instructions and automatically adjust communication chunk sizes to significantly reduce L2 cache usage and interference with other SM computation Kernels.
*   **Result:** Only **20 SMs** are required to fully utilize IB and NVLink bandwidth.

#### Extreme Memory Optimization and Minimal Overhead

To reduce training memory footprint, the following techniques were employed:
*   **Recomputation:** Recompute all **RMSNorm** operations and **MLA up-projections** during backpropagation, avoiding storage of their output activations. Significantly reduces activation memory demand at minimal overhead.
*   **CPU Storage for EMA:** Store the **Exponential Moving Average (EMA)** of model parameters in **CPU memory** and update asynchronously after each training step. Maintains EMA parameters without additional GPU memory or time overhead.
*   **Shared Embedding and Output Head:** Leverage the DualPipe strategy to deploy the shallowest layers (including Embedding) and deepest layers (including Output Head) on the **same PP rank**. This allows **MTP modules** and the main model to **physically share** parameters and gradients for Embedding and Output Head, further enhancing memory efficiency.
*   **Effect:** These optimizations enable DeepSeek-V3 to be trained **without using expensive Tensor Parallelism (TP)**.

### FP8 Training

To accelerate training and reduce GPU memory usage, DeepSeek-V3 employs an **FP8 mixed-precision training framework** ([Dettmers et al., 2022](https://arxiv.org/pdf/2208.07339); [Noune et al., 2022](https://arxiv.org/abs/2206.02915); [Peng et al., 2023](https://arxiv.org/abs/2310.18313)), validating its effectiveness on ultra-large-scale models for the first time.

#### Mixed Precision Framework

*   **Core Computation (GEMM):** Most GEMM operations (forward, activation gradient backward, weight gradient backward) use FP8 inputs, outputting BF16 or FP32, theoretically doubling computation speed.
*   **High Precision Retention:** Parts sensitive to precision or with low computational overhead (e.g., Embedding, Output Head, MoE Gating, Normalization, Attention) retain BF16/FP32 precision.
*   **High Precision Storage:** Main weights, weight gradients, and optimizer states (partially BF16) use higher precision, with ZeRO-1 sharding reducing GPU memory pressure.

{{< figure
    src="fp8_framework.png"
    caption="Fig. 14. The overall mixed precision framework with FP8 data format. For clarification, only the Linear operator is illustrated. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

#### Precision Enhancement Strategies

1.  **Fine-grained Quantization:** To address FP8's **limited dynamic range and sensitivity to outliers** ([Fishman et al., 2024](https://arxiv.org/abs/2409.12517); [He et al., 2024](https://arxiv.org/abs/2405.19279); [Sun et al., 2024](https://arxiv.org/abs/2402.17762)), adopt finer-grained quantization:
    *   **Activations:** Scaled in groups of \(1 \times 128\) tiles.
    *   **Weights:** Scaled in groups of \(128 \times 128\) blocks.
    This method allows scaling factors to better adapt to the range of local data, reducing quantization error.
2.  **Improved Accumulation Precision:** H800 Tensor Cores have limited accumulation precision (approx. 14 bits) for FP8 GEMM. To solve this, employ the **Promotion to CUDA Cores** strategy ([Thakkar et al., 2023](https://github.com/NVIDIA/cutlass)): Tensor Cores compute partial sums (e.g., every \(N_C=128\) elements), then transfer the results to CUDA Core FP32 registers for full-precision accumulation. Scaling factors from fine-grained quantization can also be efficiently applied on CUDA Cores. With concurrent execution of WGMMA operations, this method improves precision with minimal impact on computational efficiency.
3.  **E4M3 Format:** V3 uniformly uses the **E4M3 format** (4 exponent bits, 3 mantissa bits) for all tensors, rather than mixing with **E5M2** ([NVIDIA, 2024](https://github.com/NVIDIA/TransformerEngine); [Peng et al., 2023](https://arxiv.org/abs/2310.18313); [Sun et al., 2019b](https://papers.nips.cc/paper_files/paper/2019/hash/65fc9fb4897a89789352e211ca2d398f-Abstract.html)). The fine-grained quantization strategy effectively mitigates the smaller dynamic range issue of E4M3.
4.  **Online Quantization:** Compute scaling factors based on the **maximum absolute value of each tile/block in real-time, rather than relying on historical values** ([NVIDIA, 2024](https://github.com/NVIDIA/TransformerEngine); [Peng et al., 2023](https://arxiv.org/abs/2310.18313)), ensuring quantization accuracy.

{{< figure
    src="fp8_quantization_enhancement.png"
    caption="Fig. 15. (a) Fine-grained quantization method to mitigate quantization errors. (b) Improved FP8 GEMM precision by promoting to CUDA Cores for high-precision accumulation. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

#### Low-Precision Storage and Communication

*   **Optimizer States:** First and second moments of **AdamW** ([Loshchilov and Hutter, 2017](https://arxiv.org/abs/1711.05101)) are stored in BF16. Main weights and gradient accumulation remain FP32.
*   **Activation Caching:** Since Wgrad operations use FP8 inputs, activations can be cached as FP8. For specific sensitive operations (e.g., input to Linear after Attention), a custom E5M6 format with round scaling is used. Inputs to SwiGLU in MoE are also cached as FP8.
*   **Communication:** Activations before MoE up-projection are quantized to FP8 for dispatch; activation gradients before MoE down-projection are also quantized to FP8. Combine operations retain BF16 precision.

The figure below shows experiments demonstrating that the relative error of FP8 training loss compared to BF16 is less than **0.25%**, which is within an acceptable range.

{{< figure
    src="fp8_vs_bf16_loss_curves.png"
    caption="Fig. 16. Loss curves comparison between BF16 and FP8 training. Results are smoothed by Exponential Moving Average (EMA) with a coefficient of 0.9 [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

### Inference and Deployment

DeepSeek-V3 is deployed on an H800 cluster (NVLink intra-node, fully connected IB inter-node). To guarantee both **SLO (Service Level Objective)** for online services and high throughput, a deployment strategy separating **Prefilling** and **Decoding** stages is adopted.

#### Prefilling Stage

*   **Minimum Deployment Unit:** 4 nodes, 32 GPUs.
*   **Parallelism Strategy:**
    *   Attention part: **TP4 (Tensor Parallelism) + SP (Sequence Parallelism)** combined with **DP8 (Data Parallelism)**. The smaller TP size (4) limits TP communication overhead.
    *   MoE part: **EP32 (Expert Parallelism)**, ensuring each expert processes a sufficiently large batch for computational efficiency.
    *   Shallow Dense MLP: Uses TP1 to save TP communication.
*   **MoE All-to-All Communication:** Uses a method similar to training: first transmit tokens across nodes via IB, then forward within the node between GPUs via NVLink.
*   **Load Balancing:** Employs a **redundant expert** deployment strategy.
    *   Based on statistics collected from online deployment, periodically (e.g., every 10 minutes) detect **high-load experts** and replicate them.
    *   After determining the redundant expert set, carefully **reshuffle experts** among GPUs within nodes based on observed load, balancing GPU load as much as possible without increasing cross-node All-to-all communication overhead.
    *   In DeepSeek-V3 deployment, the Prefilling stage sets up **32 redundant experts**. Each GPU hosts its original 8 experts plus 1 additional redundant expert.
*   **Efficiency Optimization:** To improve throughput and hide All-to-all and TP communication overhead, **process two micro-batches with similar computational load concurrently**, overlapping the Attention and MoE of one micro-batch with the Dispatch and Combine of the other.
*   **Exploration Direction:** **Dynamic redundancy** strategy, where each GPU hosts more experts (e.g., 16), but only activates 9 per inference step. Dynamically compute the globally optimal routing scheme before each layer's All-to-all operation begins. Since Prefilling is computationally intensive, the overhead of computing the routing scheme is almost negligible.

#### Decoding Stage

*   **Expert Perspective:** Treat the **shared expert** as one routing target. From this perspective, each token selects **9 experts** during routing (the shared expert is considered a high-load expert that is always selected).
*   **Minimum Deployment Unit:** 40 nodes, 320 GPUs.
*   **Parallelism Strategy:**
    *   Attention part: **TP4 + SP** combined with **DP80**.
    *   MoE part: **EP320**. Each GPU hosts only one expert, with 64 GPUs responsible for hosting redundant and shared experts.
*   **All-to-All Communication:** Dispatch and Combine parts use **direct IB point-to-point transmission** for low latency. Utilize **IBGDA** ([NVIDIA, 2022](https://developer.nvidia.com/blog/gpudirect-storage/)) technology to further minimize latency and enhance communication efficiency.
*   **Load Balancing:** Similar to Prefilling, periodically determine the redundant expert set based on online service's statistical expert load. However, since each GPU hosts only one expert, reshuffling is not needed.
*   **Exploration Directions:**
    *   **Dynamic redundancy strategy:** Requires more careful optimization of the algorithm for computing the globally optimal routing scheme and fusion with the Dispatch Kernel to reduce overhead.
    *   **Processing two micro-batches concurrently:** Unlike Prefilling, the Attention phase takes a larger proportion of time in Decoding. Therefore, overlap the **Attention** of one micro-batch with the **Dispatch+MoE+Combine** of another. In the Decoding stage, the batch size per expert is relatively small (typically < 256 tokens), making memory access the bottleneck rather than computation. Since the MoE part only needs to load parameters for one expert, memory access overhead is small, and using fewer SMs does not significantly impact overall performance. Thus, only a small portion of SMs can be allocated to Dispatch+MoE+Combine without affecting the computation speed of the Attention part.

### Suggestions for Hardware Design

Based on the implementation of **All-to-all communication** and the **FP8 training scheme**, the DeepSeek team proposes the following chip design suggestions to AI hardware vendors.

#### Communication Hardware

*   **Current State:** Communication latency is hidden through computation/communication overlap, significantly reducing dependency on communication bandwidth. However, the current communication implementation relies on expensive **SMs** (e.g., 20 out of 132 SMs on H800 allocated for this purpose), limiting computational throughput. Furthermore, using SMs for communication leaves Tensor Cores completely idle, which is inefficient.
*   **Primary SM Tasks:**
    *   Forwarding data between IB and NVLink domains, while aggregating IB traffic destined for multiple GPUs within the same node from a single GPU.
    *   Transferring data between RDMA buffers and input/output buffers.
    *   Performing Reduce operations for All-to-all Combine.
    *   Managing fine-grained memory layout when transmitting data in chunks to multiple experts across IB and NVLink domains.
*   **Expectation:**
    *   Future vendors should develop hardware to **offload** these communication tasks from valuable computational units (SMs), potentially as GPU co-processors or network co-processors similar to NVIDIA SHARP ([Graham et al., 2016](https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf)).
    *   To reduce application programming complexity, this hardware should ideally **unify IB (scale-out) and NVLink (scale-up) networks** from the perspective of the computational units. Through this unified interface, computational units could easily perform operations like read, write, multicast, and reduce across the entire unified IB-NVLink domain by submitting communication requests based on simple primitives.

#### Compute Hardware

1.  **Higher Precision FP8 GEMM Accumulation in Tensor Cores:**
    *   **Problem:** In the current NVIDIA Hopper architecture Tensor Core implementation, FP8 GEMM uses fixed-point accumulation, aligning mantissa products via right shifts before addition. Experiments show that after sign-extended right shifts, only the top 14 bits of each mantissa product are used, with bits beyond this range truncated. However, achieving an accurate FP32 result from accumulating, say, 32 FP8×FP8 products requires at least 34 bits of precision.
    *   **Suggestion:** Future chip designs should **increase the accumulation precision within Tensor Cores** to support full-precision accumulation, or select an appropriate accumulation bit-width based on the precision requirements of training and inference algorithms. This approach maintains computational efficiency while ensuring errors remain within acceptable limits.
2.  **Support for Tile and Block Level Quantization:**
    *   **Problem:** Current GPUs only support per-tensor quantization, lacking native support for finer-grained quantization like tile-wise and block-wise. In the current implementation, when the \(N_C\) interval is reached, partial results must be copied from Tensor Cores to CUDA Cores, multiplied by scaling factors, and then added to CUDA Core FP32 registers. Although combining this with the exact FP32 accumulation strategy significantly mitigates dequantization overhead, frequent data movement between Tensor Cores and CUDA Cores still limits computational efficiency.
    *   **Suggestion:** Future chips should support fine-grained quantization by **enabling Tensor Cores to receive scaling factors and implement MMA with grouped scaling**. This would allow the entire partial sum accumulation and dequantization to be performed directly within the Tensor Cores until the final result is produced, avoiding frequent data movement.
3.  **Support for Online Quantization:**
    *   **Problem:** Current implementations struggle to efficiently support online quantization, despite its proven effectiveness in research. The existing workflow requires reading 128 BF16 activation values (output from a previous computation) from HBM for quantization, writing the quantized FP8 values back to HBM, and then reading them again for MMA.
    *   **Suggestion:**
        *   Future chips should **fuse FP8 type conversion and TMA (Tensor Memory Accelerator) access into a single operation**. This would allow quantization to occur as activations are transferred from global memory to shared memory, avoiding frequent memory reads and writes.
        *   Recommend supporting **warp-level cast instructions** for acceleration, further promoting better fusion of Layer Normalization and FP8 cast.
        *   Alternatively, adopt **near-memory computing** approaches, placing computation logic near HBM. This way, BF16 elements could be directly converted to FP8 as they are read from HBM into the GPU, reducing off-chip memory access by about 50%.
4.  **Support for Transposed GEMM Operations:**
    *   **Problem:** Fusing matrix transpose with GEMM operations is cumbersome in the current architecture. In the workflow, activations from the forward pass are quantized into \(1 \times 128\) FP8 tiles and stored. During backpropagation, the matrix needs to be read, dequantized, transposed, re-quantized into \(128 \times 1\) tiles, and stored back into HBM.
    *   **Suggestion:** Future chips should support **reading transposed matrices directly from shared memory** before MMA operations (for the precisions required by training and inference). Combined with the fusion of FP8 format conversion and TMA access, this enhancement would significantly simplify the quantization workflow.

### Training Cost and Efficiency

*   **DeepSeek-V2:** Compared to DeepSeek 67B (Dense), achieved 42.5% savings in training cost, 93.3% reduction in KV cache, and a 5.76x increase in maximum throughput.
*   **DeepSeek-V3:** Extremely high training efficiency, requiring only 180K H800 GPU hours per 1T tokens trained. The total training cost (pre-training + context extension + post-training) was only 2.788M H800 GPU hours (approx. $5.58 million, assuming $2/hour). Pre-training took less than 2 months on the 2048 H800 GPU cluster.

| Training Stage     | H800 GPU Hours | Estimated Cost (USD) |
| :----------------- | :------------: | :------------------: |
| Pre-training       | 2664 K         | $5.328 M             |
| Context Extension  | 119 K          | $0.238 M             |
| Post-training      | 5 K            | $0.01 M              |
| **Total**          | **2788 K**     | **$5.576 M**         |

## Pre-training

### Data Construction

Compared to DeepSeek-V2 (based on a 67B model, using a 100K vocabulary Byte-level BPE Tokenizer, 8.1T tokens), DeepSeek-V3 achieved larger scale and higher quality data construction during the pre-training phase through the following strategies:

1.  **Corpus Expansion and Refinement**
    *   **Domain Focus:** Significantly increased the proportion of text related to mathematics and programming, strengthening the model's understanding and generation capabilities in technical domains.
    *   **Multilingual Coverage:** Added corpora in multiple languages beyond English and Chinese, improving cross-lingual generalization performance.
    *   **Deduplication and Diversity:** Employed efficient data deduplication and filtering processes to minimize redundancy while ensuring content diversity.
    *   **Scale Increase:** Ultimately constructed approximately **14.8T** high-quality tokens, an increase of nearly 83% compared to V2.

2.  **Training Strategy and Technical Innovation**
    *   **Document Packing**
        Combined the **Document Packing** ([Ding et al., 2024](https://arxiv.org/abs/2404.10830)) method, packing coherent texts into longer segments to improve GPU utilization and context integrity; did not use cross-sample attention masks to maintain implementation simplicity.
    *   **Fill-in-Middle (FIM) Strategy**
        *   **Motivation:** Inspired by DeepSeekCoder-V2 ([DeepSeek-AI, 2024](https://arxiv.org/abs/2406.11931)), aimed at enhancing the model's ability to fill in missing information in the middle.
        *   **Framework:** Introduced the Prefix-Suffix-Middle (PSM) structure, with examples like:
            ```
            <|fim_begin|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>
            ```
        *   **Application Ratio:** Inserted FIM before document-level pre-packing, accounting for **10%**, balancing generation and prediction tasks.

3.  **Tokenizer Optimization**
    *   **BBPE Vocabulary Expansion:** Adopted Byte-level BPE, expanding the vocabulary from 100K to **128K**, improving coverage of rare words and proper nouns.
    *   **Pre-tokenizer Improvement:** Adjusted tokenization rules for multilingual scenarios, enhancing compression efficiency and encoding consistency.
    *   **Boundary Bias Mitigation:** Referenced [Lundberg, 2023](https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb)'s method to reduce bias from punctuation + newline combination tokens in few-shot scenarios by introducing a random splitting mechanism, exposing the model to more boundary variations.

### Hyperparameters

| Parameter                  | DeepSeek-V2                                           | DeepSeek-V3                                                     |
| :------------------------- | :----------------------------------------------------: | :-------------------------------------------------------------: |
| Transformer Layers         | 60                                                    | 61                                                              |
| Hidden Dimension \(d\)     | 5120                                                  | 7168                                                            |
| Initialization Stddev      | 0.006                                                 | 0.006                                                           |
| **MLA Parameters**         |                                                       |                                                                 |
| Attention Heads \(n_h\)    | 128                                                   | 128                                                             |
| Dim per Head \(d_h\)       | 128                                                   | 128                                                             |
| KV Compression Dim \(d_c\) | 512 (\(4d_h\))                                        | 512 (\(4d_h\))                                                 |
| Query Compression Dim \(d_c'\)| 1536                                                  | 1536                                                            |
| Decoupled RoPE Dim \(d_h^R\)| 64 (\(d_h/2\))                                         | 64 (\(d_h/2\))                                                  |
| **DeepSeekMoE Parameters** |                                                       |                                                                 |
| MoE Layer Position         | All except layer 1                                    | All except first 3 layers                                       |
| Shared Experts \(N_s\)     | 2                                                     | 1                                                               |
| Routing Experts \(N_r\)    | 160                                                   | 256                                                             |
| Expert Intermediate Dim    | 1536                                                  | 2048                                                            |
| Activated Experts \(K_r\)  | 6                                                     | 8                                                               |
| Device/Node Limit \(M\)    | 3 (Device)                                            | 4 (Node)                                                        |
| Load Balancing Strategy    | Aux Losses (\(\alpha_1=0.003, \alpha_2=0.05, \alpha_3=0.02\)) + Token Dropping | Aux-Loss-Free (\(\gamma=0.001\)) + Seq Loss (\(\alpha=0.0001\)) |
| **MTP Parameters (V3 only)**|                                                       |                                                                 |
| Prediction Depth \(D_{MTP}\)| N/A                                                   | 1                                                               |
| MTP Loss Weight \(\lambda\)| N/A                                                   | 0.3 (first 10T) / 0.1 (last 4.8T)                               |
| **Training Parameters**    |                                                       |                                                                 |
| Optimizer                  | AdamW (\(\beta_1=0.9, \beta_2=0.95, wd=0.1\))         | AdamW (\(\beta_1=0.9, \beta_2=0.95, wd=0.1\))                  |
| Max Sequence Length        | 4K                                                    | 4K                                                              |
| Training Tokens            | 8.1T                                                  | 14.8T                                                           |
| Learning Rate              | Warmup + Step Decay (Max \(2.4 \times 10^{-4}\))       | Warmup + Cosine Decay + Constant (Max \(2.2 \times 10^{-4}\))    |
| Batch Size                 | 2304 -> 9216                                          | 3072 -> 15360                                                   |
| Gradient Clipping          | 1.0                                                   | 1.0                                                             |
| Precision                  | BF16                                                  | FP8 Mixed Precision                                             |

### Long Context Extension

Both models use the **YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)) technique to extend the context window.
*   **DeepSeek-V2:** Extended from 4K to 128K. Used YaRN (scale \(s=40, \alpha=1, \beta=32\)), trained for 1000 steps on 32K sequence length. Adjusted length scaling factor \(\sqrt{t}=0.0707 \ln s+1\).
*   **DeepSeek-V3:** Extended from 4K to 32K, then to 128K in two stages. Each stage trained for 1000 steps. YaRN parameters same as V2, length scaling factor \(\sqrt{t}=0.1 \ln s+1\). First stage sequence length 32K, second stage 128K.

Both models demonstrated good long context capabilities in the NIAH test.

{{< figure
    src="deepseek_v2_niah.png"
    caption="Fig. 19. Evaluation results on the 'Needle In A Haystack' (NIAH) tests for DeepSeek-V2. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

{{< figure
    src="deepseek_v3_niah.png"
    caption="Fig. 20. Evaluation results on the 'Needle In A Haystack' (NIAH) tests for DeepSeek-V3. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

### Evaluation

**DeepSeek-V2 Evaluation Results:**

Comparison of DeepSeek-V2 with representative open-source models (partial results). DeepSeek-V2 achieved state-of-the-art performance at the time with 21B activated parameters.

|         | Benchmark (Metric)                               | # Shots | DeepSeek 67B | Qwen1.5 72B | Mixtral 8x22B | LLaMA 3 70B | DeepSeek-V2 |
| :-----: | :-----------------------------------------------: | :-----: | :----------: | :---------: | :-----------: | :---------: | :---------: |
|         | # Activated Params                               | -       | 67B          | 72B         | 39B           | 70B         | **21B**     |
| English | MMLU ([Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300)) (Acc.) | 5-shot  | 71.3         | 77.2        | 77.6          | 78.9        | **78.5**    |
| Code    | HumanEval ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) (Pass@1) | 0-shot  | 45.1         | 43.9        | 53.1          | 48.2        | **48.8**    |
| Math    | GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) (EM)     | 8-shot  | 63.4         | 77.9        | 80.3          | 83.0        | **79.2**    |
| Chinese | C-Eval ([Huang et al., 2023](https://arxiv.org/abs/2305.08322)) (Acc.)   | 5-shot  | 66.1         | **83.7**    | 59.6          | 67.5        | 81.7        |

**DeepSeek-V3 Evaluation Results:**

Comparison of DeepSeek-V3-Base with representative open-source models (partial results). DeepSeek-V3-Base became the strongest open-source model on most benchmarks, especially in code and math.

|             | Benchmark (Metric)                                       | # Shots | DeepSeek-V2 Base | Qwen2.5 72B Base | LLaMA-3.1 405B Base | DeepSeek-V3 Base |
| :---------: | :-------------------------------------------------------: | :-----: | :--------------: | :--------------: | :-----------------: | :--------------: |
|             | # Activated Params                                       | -       | 21B              | 72B              | 405B                | **37B**          |
| English     | MMLU ([Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300)) (EM)         | 5-shot  | 78.4             | 85.0             | 84.4                | **87.1**         |
|             | MMLU-Pro ([Wang et al., 2024](https://arxiv.org/abs/2406.01574)) (em)         | 5-shot  | 51.4             | 58.3             | 52.8                | **64.4**         |
| Code        | HumanEval ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) (Pass@1)     | 0-shot  | 43.3             | 53.0             | 54.9                | **65.2**         |
|             | LiveCodeBench-Base ([Jain et al., 2024](https://arxiv.org/abs/2403.07974)) (Pass@1) | 3-shot  | 11.6             | 12.9             | 15.5                | **19.4**         |
| Math        | GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) (Em)         | 8-shot  | 81.6             | 88.3             | 83.5                | **89.3**         |
|             | MATH ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) (EM)        | 4-shot  | 43.4             | 54.4             | 49.0                | **61.6**         |
| Chinese     | C-Eval ([Huang et al., 2023](https://arxiv.org/abs/2305.08322)) (EM)        | 5-shot  | 81.4             | 89.2             | 72.5                | **90.1**         |
| Multilingual| MMMLU-non-English ([OpenAI, 2024](https://huggingface.co/datasets/openai/MMMLU)) (em) | 5-shot  | 64.0             | 74.8             | 73.8                | **79.4**         |

**Summary:** DeepSeek-V3-Base, leveraging its architectural innovations, larger training dataset, and efficient training methods, comprehensively surpassed DeepSeek-V2-Base and other top open-source models (including LLaMA-3.1 405B, whose total parameter count far exceeds V3's activated parameters).

## Alignment

To enable the models to better understand instructions, follow human preferences, and enhance specific capabilities (like reasoning), both DeepSeek-V2 and V3 underwent Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

### Supervised Fine-Tuning

*   **DeepSeek-V2:** Used approximately 1.5M instruction data points, including 1.2M helpfulness data and 0.3M safety data, focusing on improving data quality to reduce hallucinations and enhance writing ability.
*   **DeepSeek-V3:**
    *   **Reasoning Data:** Utilized the internal DeepSeek-R1 model ([Guo et al., 2025](https://arxiv.org/abs/2501.12948)) to generate reasoning processes (math, code, logic, etc.). Since R1 outputs could be overly long or poorly formatted, V3 adopted a **knowledge distillation** approach:
        1.  Train domain expert models (e.g., code expert): Combined original SFT data with R1-generated long CoT data (with system prompts guiding reflection/verification) for SFT+RL training.
        2.  Use expert models to generate SFT data: The expert models learned during RL to blend R1's reasoning patterns with the conciseness of regular SFT data.
        3.  Rejection sampling: Filtered high-quality SFT data for the final V3 SFT.
    *   **Non-Reasoning Data:** Generated using DeepSeek-V2.5 and verified by human annotators.
    *   **SFT Setup:** Fine-tuned for 2 epochs, with learning rate cosine decayed from \(5 \times 10^{-6}\) to \(1 \times 10^{-6}\). Employed sample packing and mask isolation.

### Reinforcement Learning

Both models used the **Group Relative Policy Optimization (GRPO)** algorithm ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) for RL. GRPO is an Actor-Only method that estimates advantage \(A_i\) by comparing the relative quality of a group (\(G\)) of candidate outputs. This avoids training a Critic model of the same size as the policy model, saving costs.

GRPO objective function:
\[
\begin{gathered}
\mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{o l d}}(O \mid q)\right] \\
\frac{1}{G} \sum_{i=1}^{G}\left(\min \left(\frac{\pi_{\theta}\left(o_{i} \mid q\right)}{\pi_{\theta_{o l d}}\left(o_{i} \mid q\right)} A_{i}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{i} \mid q\right)}{\pi_{\theta_{o l d}}\left(o_{i} \mid q\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{i}\right)-\beta \mathbb{D}_{K L}\left(\pi_{\theta}| | \pi_{r e f}\right)\right),
\end{gathered}
\]
where the advantage \(A_i\) is obtained by standardizing the intra-group rewards \(r_i\):
\[
A_{i}=\frac{r_{i}-\operatorname{mean}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}{\operatorname{std}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}.
\]
The KL divergence penalty term uses the Schulman unbiased estimator:
\[
\mathbb{D}_{K L}\left(\pi_{\theta}| | \pi_{r e f}\right)=\frac{\pi_{r e f}\left(o_{i} \mid q\right)}{\pi_{\theta}\left(o_{i} \mid q\right)}-\log \frac{\pi_{r e f}\left(o_{i} \mid q\right)}{\pi_{\theta}\left(o_{i} \mid q\right)}-1.
\]

**Reward Model (RM):**

*   **DeepSeek-V2:** Employed a two-stage RL strategy.
    1.  **Reasoning Alignment:** Used a specially trained \(RM_{\text{reasoning}}\) to optimize for code and math reasoning tasks.
    2.  **Human Preference Alignment:** Used a multi-reward framework combining \(RM_{\text{helpful}}\), \(RM_{\text{safety}}\), and rule-based \(RM_{\text{rule}}\).
*   **DeepSeek-V3:**
    *   **Rule-based RM:** For verifiable tasks (e.g., math answer format, LeetCode test cases), used rules to provide reliable rewards.
    *   **Model-based RM:** For free-form answers or tasks without standard answers (e.g., creative writing), used an RM initialized from the V3 SFT Checkpoint. This RM was trained on preference data with CoT to enhance reliability and reduce reward hacking risks.
    *   **Self-Reward:** V3 explored using the model's own judgment capabilities (enhanced via voting) as a feedback source, especially in general scenarios, combined with ideas from **Constitutional AI** ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) for optimization.

**RL Training Optimization (V2/V3):** Addressed the high resource demands of large-model RL through engineering optimizations like a hybrid engine (different parallelism strategies for training/inference), using **vLLM** ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)) for accelerated sampling, CPU offloading scheduling, etc.

### Evaluation

**DeepSeek-V2 Chat Evaluation:**

Comparison of DeepSeek-V2 Chat (SFT/RL) with representative open-source Chat models on open-ended generation tasks. V2 Chat (RL) performed exceptionally well on AlpacaEval 2.0 and AlignBench.

| Model                  | MT-Bench ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)) | AlpacaEval 2.0 ([Dubois et al., 2024](https://arxiv.org/abs/2404.04475)) (LC Win Rate) | AlignBench ([Liu et al., 2023](https://doi.org/10.48550/arXiv.2311.18743)) (Chinese) |
| :---------------------: | :----------------------------------------------------------: | :---------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
| DeepSeek 67B Chat      | 8.35                                                         | 16.6                                                                                | 6.43                                                                            |
| Mistral 8x22B Instruct | 8.66                                                         | 30.9                                                                                | -                                                                               |
| Qwen1.5 72B Chat       | 8.61                                                         | 36.6                                                                                | 7.19                                                                            |
| LLaMA3 70B Instruct    | **8.95**                                                     | 34.4                                                                                | -                                                                               |
| DeepSeek-V2 Chat (SFT) | 8.62                                                         | 30.0                                                                                | 7.74                                                                            |
| DeepSeek-V2 Chat (RL)  | **8.97**                                                     | **38.9**                                                                            | **7.91**                                                                        |

**DeepSeek-V3 Chat Evaluation:**

Comparison of DeepSeek-V3 Chat with representative open-source and closed-source Chat models (partial results). V3 leads open-source models on most benchmarks and is comparable to top closed-source models in code, math, Chinese, and open-ended generation tasks.

|         | Benchmark (Metric)                                       | DeepSeek V2.5-0905 | Qwen2.5 72B-Inst. | LLaMA-3.1 405B-Inst. | Claude-3.5- Sonnet-1022 | GPT-4o 0513 | DeepSeek V3 |
| :-----: | :-------------------------------------------------------: | :----------------: | :---------------: | :------------------: | :---------------------: | :---------: | :---------: |
| English | MMLU ([Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300)) (EM)         | 80.6               | 85.3              | 88.6                 | 88.3                    | 87.2        | **88.5**    |
|         | MMLU-Pro ([Wang et al., 2024](https://arxiv.org/abs/2406.01574)) (EM)         | 66.2               | 71.6              | 73.3                 | **78.0**                | 72.6        | 75.9        |
|         | GPQA-Diamond ([Rein et al., 2023](https://arxiv.org/abs/2311.12022)) (Pass@1) | 41.3               | 49.0              | 51.1                 | **65.0**                | 49.9        | 59.1        |
|         | SimpleQA ([OpenAI, 2024c](https://openai.com/index/introducing-simpleqa/)) (Correct) | 10.2               | 9.1               | 17.1                 | 28.4                    | **38.2**    | 24.9        |
| Code    | HumanEval-Mul (Pass@1)                                   | 77.4               | 77.3              | 77.2                 | 81.7                    | 80.5        | **82.6**    |
|         | LiveCodeBench ([Jain et al., 2024](https://arxiv.org/abs/2403.07974)) (Pass@1-COT) | 29.2               | 31.1              | 28.4                 | 36.3                    | 33.4        | **40.5**    |
|         | SWE Verified ([OpenAI, 2024d](https://openai.com/index/introducing-swe-bench-verified/)) (Resolved) | 22.6               | 23.8              | 24.5                 | **50.8**                | 38.8        | 42.0        |
| Math    | AIME 2024 ([MAA, 2024](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOooril84-FGuAUnzl8I-zXl8XG7P00X-BAkMG9x9RIzEWcXHlwWm) (Pass@1) | 16.7               | 23.3              | 23.3                 | 16.0                    | 9.3         | **39.2**    |
|         | MATH-500 ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) (ЕМ)    | 74.7               | 80.0              | 73.8                 | 78.3                    | 74.6        | **90.2**    |
| Chinese | C-Eval ([Huang et al., 2023](https://arxiv.org/abs/2305.08322)) (EM)        | 79.5               | 86.1              | 61.5                 | 76.7                    | 76.0        | **86.5**    |
|         | C-SimpleQA ([He et al., 2024](https://arxiv.org/abs/2411.07140)) (Correct)   | 54.1               | 48.4              | 50.4                 | 51.3                    | 59.3        | **64.8**    |
| Open-Ended| Arena-Hard ([Li et al., 2024](https://arxiv.org/abs/2406.11939))           | 76.2               | 81.2              | 69.3                 | 85.2                    | 80.4        | **85.5**    |
|         | AlpacaEval 2.0 ([Dubois et al., 2024](https://arxiv.org/abs/2404.04475)) (LC Win Rate) | 50.5               | 49.1              | 40.5                 | 52.0                    | 51.1        | **70.0**    |

**Summary:**
*   DeepSeek-V2 Chat (RL) was already a top-tier open-source chat model at its release, particularly excelling on AlpacaEval and the Chinese AlignBench.
*   DeepSeek-V3 Chat further boosted performance, becoming the current strongest open-source chat model. It shows extremely strong performance in code, math, Chinese knowledge, and open-ended evaluations like Arena-Hard ([Li et al., 2024](https://arxiv.org/abs/2406.11939)) and AlpacaEval, reaching levels comparable to GPT-4o and Claude-3.5-Sonnet.
*   V3's R1 distillation significantly improved reasoning capabilities but might also increase response length, requiring a trade-off between accuracy and efficiency.
*   V3's self-reward capability (strong performance on RewardBench ([Lambert et al., 2024](https://arxiv.org/abs/2403.13787))) provides an effective pathway for continuous alignment.

## Discussion

*   **Load Balancing Strategy Evolution:** The shift from V2's auxiliary losses to V3's auxiliary-loss-free + bias adjustment reflects a trend towards minimizing interference with model performance while ensuring load balance. Batch-level balancing, compared to sequence-level, better facilitates expert specialization.
*   **Effectiveness of MTP:** V3's experiments demonstrate that multi-token prediction as an auxiliary training objective indeed improves model performance on standard evaluation tasks, while also offering potential for inference acceleration (speculative decoding).
*   **R1 Distillation:** V3 successfully distilled the long-chain reasoning capabilities of DeepSeek-R1 into a standard LLM, significantly boosting math and code abilities. This is an important technical direction, though controlling generation length needs attention.
*   **Self-Reward:** V3's strong judgment capability (evidenced by **RewardBench** results ([Lambert et al., 2024](https://arxiv.org/abs/2403.13787))) enables effective self-feedback and self-alignment. This is crucial for reducing reliance on human annotation and achieving continuous model self-improvement.
*   **SFT Data Quantity:** While **LIMA** ([Zhou et al., 2024](https://arxiv.org/abs/2305.11206)) suggested that a small amount of high-quality SFT data can achieve good results, sufficient high-quality data is still necessary for specific skills (like instruction following, IFEval) to reach satisfactory performance.
*   **Alignment Tax:** OpenAI noted in **InstructGPT** ([Ouyang et al., 2022](https://arxiv.org/pdf/2203.02155)) that RL alignment, while improving open-ended generation capabilities, might sacrifice performance on some standard benchmarks. Both V2 and V3 made efforts in data processing and training strategies to mitigate this issue and achieve an acceptable balance.

## Conclusion, Limitations & Future Directions

### Conclusion

DeepSeek-V2 and DeepSeek-V3 are two powerful, economical, and efficient MoE language models. Through innovations like the MLA and DeepSeekMoE architectures, along with V3's introduction of auxiliary-loss-free load balancing, MTP, FP8 training, and R1 distillation, they have achieved breakthroughs in performance, training cost, and inference efficiency. DeepSeek-V3 has become one of the strongest open-source models currently available, with performance competitive with top closed-source models.

### Limitations

*   **General LLM Limitations:** Such as knowledge cutoffs, hallucinations, factual errors, etc.
*   **Language Coverage:** Primarily focused on Chinese and English, with limited capabilities in other languages (V2). V3 expanded multilingual support but remains predominantly focused on Chinese and English.
*   **Deployment Threshold (V3):** Efficient inference requires relatively large deployment units (multi-node), which might be challenging for smaller teams.
*   **Inference Efficiency:** Although V3's inference efficiency improved over V2, there is still room for optimization.

### Future Directions

*   **Architectural Innovation:** Continue optimizing MoE architectures, exploring new architectures supporting infinite context and overcoming Transformer limitations.
*   **Data Expansion:** Improve the quantity, quality, and dimensionality (multimodality, etc.) of training data.
*   **Deeper Reasoning:** Enhance the model's reasoning length and depth, increasing intelligence levels.
*   **Evaluation Methods:** Develop more comprehensive, multi-dimensional evaluation methods to avoid overfitting to specific benchmarks.
*   **Alignment and Safety:** Continuously improve alignment techniques (e.g., self-reward) to ensure models are helpful, honest, harmless, and aligned with human values.

## References

[1] Liu, Aixin, et al. ["Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model."](https://arxiv.org/abs/2405.04434) arXiv preprint arXiv:2405.04434 (2024).

[2] Liu, Aixin, et al. ["Deepseek-v3 technical report."](https://arxiv.org/abs/2412.19437) arXiv preprint arXiv:2412.19437 (2024).

[3] Dai, Damai, et al. ["Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models."](https://arxiv.org/abs/2401.06066) arXiv preprint arXiv:2401.06066 (2024).

[4] Wang, Lean, et al. ["Auxiliary-loss-free load balancing strategy for mixture-of-experts."](https://arxiv.org/abs/2408.15664) arXiv preprint arXiv:2408.15664 (2024).

[5] Gloeckle, Fabian, et al. ["Better & faster large language models via multi-token prediction."](https://arxiv.org/abs/2404.19737) Proceedings of the 41st International Conference on Machine Learning. PMLR 235:16821-16841 (2024).

[6] Vaswani, Ashish, et al. ["Attention is all you need."](https://arxiv.org/abs/1706.03762) Advances in neural information processing systems 30 (2017).

[7] Shazeer, Noam. ["Fast transformer decoding: One write-head is all you need."](https://arxiv.org/abs/1911.02150) arXiv preprint arXiv:1911.02150 (2019).

[8] Ainslie, Joshua, et al. ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints."](https://arxiv.org/abs/2305.13245) Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 4895-4901 (2023).

[9] Su, Jianlin, et al. ["Roformer: Enhanced transformer with rotary position embedding."](https://arxiv.org/abs/2104.09864) Neurocomputing 568 (2024): 127063.

[10] Shazeer, Noam, et al. ["Outrageously large neural networks: The sparsely-gated mixture-of-experts layer."](https://arxiv.org/abs/1701.06538) arXiv preprint arXiv:1701.06538 (2017).

[11] Lepikhin, Dmitry, et al. ["Gshard: Scaling giant models with conditional computation and automatic sharding."](https://arxiv.org/abs/2006.16668) arXiv preprint arXiv:2006.16668 (2020).

[12] Fedus, William, Barret Zoph, and Noam Shazeer. ["Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity."](https://arxiv.org/abs/2101.03961) The Journal of Machine Learning Research 23.1: 5232-5270 (2022).

[13] Zhou, Zexuan, et al. ["Mixture-of-experts with expert choice routing."](https://arxiv.org/abs/2202.09368) Advances in Neural Information Processing Systems 35: 7103-7114 (2022).

[14] Leviathan, Yaniv, Matan Kalman, and Yossi Matias. ["Fast inference from transformers via speculative decoding."](https://arxiv.org/abs/2211.17192) Proceedings of the 40th International Conference on Machine Learning. PMLR 202:19274-19286 (2023).

[15] Xia, Yichao, et al. ["Accelerating large language model decoding with speculative sampling."](https://arxiv.org/abs/2302.01318) arXiv preprint arXiv:2302.01318 (2023).

[16] Qi, Hai, et al. ["ZeroBubble: A High-Performance Framework for Training Mixture-of-Experts Models."](https://arxiv.org/abs/2401.10241) arXiv preprint arXiv:2401.10241 (2024).

[17] Rajbhandari, Samyam, et al. ["Zero: Memory optimizations toward training trillion parameter models."](https://arxiv.org/abs/1910.02054) SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE (2020).

[18] Harlap, Aaron, et al. ["Pipedream: Fast and efficient pipeline parallel dnn training."](https://arxiv.org/abs/1806.03377) arXiv preprint arXiv:1806.03377 (2018).

[19] Li, Shigang, and Torsten Hoefler. ["Chimera: Efficiently training large-scale neural networks with bidirectional pipelines."](https://arxiv.org/abs/2107.06925) Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2021.

[20] Bauer, Michael, Sean Treichler, and Alex Aiken. ["Singe: Leveraging warp specialization for high performance on gpus."](https://dl.acm.org/doi/10.1145/2692916.2555258) Proceedings of the 19th ACM SIGPLAN symposium on Principles and practice of parallel programming. 2014.

[21] Dettmers, Tim, et al. ["Llm. int8 (): 8-bit matrix multiplication for transformers at scale."](https://arxiv.org/abs/2208.07339) Advances in Neural Information Processing Systems 35: 34138-34151 (2022).

[22] Noune, Badreddine, et al. ["8-bit numerical formats for deep neural networks."](https://arxiv.org/abs/2206.02915) arXiv preprint arXiv:2206.02915 (2022).

[23] Peng, Houwen, et al. ["FP8-LM: Training FP8 Large Language Models."](https://arxiv.org/abs/2310.18313) arXiv preprint arXiv:2310.18313 (2023).

[24] Fishman, Maxim, et al. ["Scaling FP8 training to trillion-token LLMs."](https://arxiv.org/abs/2409.12517)) arXiv preprint arXiv:2409.12517 (2024).

[25] He, Bobby, et al. ["Understanding and minimising outlier features in neural network training."](https://arxiv.org/abs/2405.19279) arXiv preprint arXiv:2405.19279 (2024).

[26] Sun, Xiao, et al. ["Massive activations in large language models."](https://arxiv.org/abs/2402.17762) arXiv preprint arXiv:2402.17762 (2024).

[27] NVIDIA. ["Transformer Engine."](https://github.com/NVIDIA/TransformerEngine) GitHub Repository (Accessed 2024).

[28] Sun, Xiao, et al. ["Hybrid 8-bit floating point (HFP8) training and inference for deep neural networks."](https://papers.nips.cc/paper_files/paper/2019/hash/65fc9fb4897a89789352e211ca2d398f-Abstract.html) Advances in neural information processing systems 32 (2019).

[29] Loshchilov, Ilya, and Frank Hutter. ["Decoupled weight decay regularization."](https://arxiv.org/abs/1711.05101) arXiv preprint arXiv:1711.05101 (2017).

[30] NVIDIA. ["GPUDirect Storage: A Direct Path Between Storage and GPU Memory."](https://developer.nvidia.com/blog/gpudirect-storage/) NVIDIA Developer Blog (2022).

[31] Graham, Richard L., et al. ["Scalable hierarchical aggregation protocol (SHArP): A hardware architecture for efficient data reduction."](https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf) 2016 First International Workshop on Communication Optimizations in HPC (COMHPC). IEEE, 2016.

[32] Ding, Yiran, et al. ["Longrope: Extending llm context window beyond 2 million tokens."](https://arxiv.org/abs/2402.13753) arXiv preprint arXiv:2402.13753 (2024).

[33] Zhu, Qihao, et al. ["DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence."](https://arxiv.org/abs/2406.11931) arXiv preprint arXiv:2406.11931 (2024).

[34] Lundberg, Scott M. ["Guidance: Prompt Boundaries and Token Healing."](https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb) GitHub Notebook (2023).

[35] Peng, Bowen, et al. ["YaRN: Efficient Context Window Extension of Large Language Models."](https://arxiv.org/abs/2309.00071) arXiv preprint arXiv:2309.00071 (2023).

[36] Hendrycks, Dan, et al. ["Measuring massive multitask language understanding."](https://arxiv.org/abs/2009.03300) arXiv preprint arXiv:2009.03300 (2020).

[37] Chen, Mark, et al. ["Evaluating large language models trained on code."](https://arxiv.org/abs/2107.03374) arXiv preprint arXiv:2107.03374 (2021).

[38] Cobbe, Karl, et al. ["Training verifiers to solve math word problems."](https://arxiv.org/abs/2110.14168) arXiv preprint arXiv:2110.14168 (2021).

[39] Huang, Yuzhen, et al. ["C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models."](https://arxiv.org/abs/2305.08322) Advances in Neural Information Processing Systems 36 (2023): 62991-63010.

[40] Wang, Yubo, et al. ["Mmlu-pro: A more robust and challenging multi-task language understanding benchmark."](https://arxiv.org/abs/2406.01574) The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2024.

[41] Jain, Naman, et al. ["LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code."](https://arxiv.org/abs/2403.07974) arXiv preprint arXiv:2403.07974 (2024).

[42] Hendrycks, Dan, et al. ["Measuring mathematical problem solving with the math dataset."](https://arxiv.org/abs/2103.03874) arXiv preprint arXiv:2103.03874 (2021).

[43] OpenAI. ["MMMLU Dataset."](https://huggingface.co/datasets/openai/MMMLU) Hugging Face Datasets (Accessed 2024).

[44] Guo, Daya, et al. ["Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning."](https://arxiv.org/abs/2501.12948) arXiv preprint arXiv:2501.12948 (2025).

[45] Shao, Zhihong, et al. ["Deepseekmath: Pushing the limits of mathematical reasoning in open language models."](https://arxiv.org/abs/2402.03300) arXiv preprint arXiv:2402.03300 (2024).

[46] Bai, Yuntao, et al. ["Constitutional ai: Harmlessness from ai feedback."](https://arxiv.org/abs/2212.08073) arXiv preprint arXiv:2212.08073 (2022).

[47] Kwon, Woosuk, et al. ["Efficient memory management for large language model serving with pagedattention."](https://arxiv.org/abs/2309.06180) Proceedings of the 29th Symposium on Operating Systems Principles. 2023.

[48] Zheng, Lianmin, et al. ["Judging llm-as-a-judge with mt-bench and chatbot arena."](https://arxiv.org/abs/2306.05685) Advances in Neural Information Processing Systems 36 (2023): 46595-46623.

[49] Dubois, Yann, et al. ["Length-controlled alpacaeval: A simple way to debias automatic evaluators."](https://arxiv.org/abs/2404.04475) arXiv preprint arXiv:2404.04475 (2024).

[50] Liu, Xiao, et al. ["Alignbench: Benchmarking chinese alignment of large language models."](https://arxiv.org/abs/2311.18743) arXiv preprint arXiv:2311.18743 (2023).

[51] Rein, David, et al. ["GPQA: A Graduate-Level Google-Proof Q&A Benchmark."](https://arxiv.org/abs/2311.12022) First Conference on Language Modeling. 2024.

[52] OpenAI. ["Introducing SimpleQA"](https://openai.com/index/introducing-simpleqa/) OpenAI Blog (2024).

[53] OpenAI. ["Introducing SWE-bench Verified"](https://openai.com/index/introducing-swe-bench-verified/) OpenAI Blog (2024).

[54] Mathematical Association of America (MAA). ["2024 AIME I Problems."](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I) Art of Problem Solving Wiki (2024).

[55] Li, Tianle, et al. ["From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline."](https://arxiv.org/abs/2406.11939) arXiv preprint arXiv:2406.11939 (2024).

[56] Lambert, Nathan, et al. ["RewardBench: Evaluating Reward Models for Language Modeling."](https://arxiv.org/abs/2403.13787) arXiv preprint arXiv:2403.13787 (2024).

[57] Zhou, Chunting, et al. ["Lima: Less is more for alignment."](https://arxiv.org/abs/2305.11206) Advances in Neural Information Processing Systems 36 (2023): 55006-55021.

[58] Ouyang, Long, et al. ["Training language models to follow instructions with human feedback."](https://arxiv.org/abs/2203.02155) Advances in neural information processing systems 35 (2022): 27730-27744.

## Citation

> **Citation**: When reproducing or citing the content of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (Apr 2025). DeepSeek-V2 vs V3.
https://syhya.github.io/posts/2025-04-18-deepseek-v2-v3

Or

```bibtex
@article{syhya2025deepseekv2v3,
  title   = "DeepSeek-V2 vs V3",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Apr",
  url     = "https://syhya.github.io/posts/2025-04-18-deepseek-v2-v3"
}
```