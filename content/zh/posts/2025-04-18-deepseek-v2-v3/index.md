---
title: "DeepSeek-V2 vs V3"
date: 2025-04-18T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "LLM", "DeepSeek-V2", "DeepSeek-V3", "MoE", "Transformer", "MLA", "DeepSeekMoE", "MTP", "FP8 Training", "GRPO", "SFT", "RL", "KV Cache"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

DeepSeek AI 先后发布了 **DeepSeek-V2** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434)) 和 **DeepSeek-V3** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))，这两款强大的混合专家（Mixture-of-Experts, MoE）语言模型在保持顶尖性能的同时，显著优化了训练成本和推理效率。DeepSeek-V2 拥有 236B 总参数，每次激活 21B；而 DeepSeek-V3 则进一步扩展至 671B 总参数，每次激活 37B。两者均支持 128K 上下文长度。

这两个模型的核心创新在于采用了 **多头隐注意力 (Multi-head Latent Attention, MLA)** 和 **DeepSeekMoE** 架构 ([Dai et al., 2024](https://arxiv.org/abs/2401.06066))。MLA 通过将键值（KV）缓存压缩到低维隐向量中，大幅降低了推理时的显存占用，提高了效率。DeepSeekMoE 则通过细粒度专家切分和共享专家隔离，实现了更强的专家特化能力和更经济的训练成本。DeepSeek-V3 在 V2 的基础上，进一步引入了**无辅助损失的负载均衡策略 (Auxiliary-Loss-Free Load Balancing)** ([Wang et al., 2024](https://arxiv.org/abs/2408.15664)) 和**多 token 预测 (Multi-Token Prediction, MTP)** ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737))训练目标 ，进一步提升了模型性能和训练效率。

DeepSeek-V2 在 8.1T tokens 上进行预训练，而 DeepSeek-V3 则在更大规模的 14.8T tokens 上训练。两者都经过了监督微调（Supervised Fine-Tuning, SFT）和强化学习（Reinforcement Learning, RL）阶段以充分释放潜力。评估结果显示，DeepSeek-V2 和 V3 在众多基准测试中均达到了开源模型的顶尖水平，DeepSeek-V3 更是成为了目前最强的开源基础模型之一，性能可与顶尖闭源模型媲美。

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

本文将深入探讨 DeepSeek-V2 和 DeepSeek-V3 的关键技术，包括其创新的模型架构、高效的训练基础设施、预训练和对齐过程。

## 符号表

下面列举了文章所使用的数学公式，可以帮你更轻松阅读。

| 符号 | 含义 |
| :--- | :--- |
| \( d \) | 嵌入维度 |
| \( n_h \) | 注意力头数量 |
| \( d_h \) | 每个注意力头的维度 |
| \( \mathbf{h}_t \in \mathbb{R}^d \) | 第 \( t \) 个 token 在注意力层的输入 |
| \( \mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \) | 查询, 键, 值向量 |
| \( W^Q, W^K, W^V, W^O \) | 查询、键、值、输出的投影矩阵 |
| \( \mathbf{q}_{t,i}, \mathbf{k}_{t,i}, \mathbf{v}_{t,i} \) | 第 \( i \) 个注意力头的查询、键、值向量 |
| \( \mathbf{o}_{t,i} \) | 第 \( i \) 个注意力头的输出 |
| \( \mathbf{u}_t \) | 注意力层的最终输出 |
| \( l \) | 模型层数 |
| \( \mathbf{c}_t^{KV} \in \mathbb{R}^{d_c} \) | 键值的压缩隐向量 |
| \( d_c \) | KV 压缩维度 |
| \( W^{DKV}, W^{UK}, W^{UV} \) | KV 的下投影、键的上投影、值的上投影矩阵 |
| \( \mathbf{k}_t^C, \mathbf{v}_t^C \) | 通过上投影从隐向量恢复的键和值 |
| \( \mathbf{c}_t^Q \in \mathbb{R}^{d_c'} \) | 查询的压缩隐向量 |
| \( d_c' \) | 查询压缩维度 |
| \( W^{DQ}, W^{UQ} \) | 查询的下投影、上投影矩阵 |
| \( \mathbf{q}_t^C \) | 通过上投影从隐向量恢复的查询 |
| \( \mathbf{q}_{t,i}^R, \mathbf{k}_t^R \) | 解耦的 RoPE 查询和键 |
| \( d_h^R \) | 解耦 RoPE 查询/键的头维度 |
| \( W^{QR}, W^{KR} \) | 解耦 RoPE 查询/键的生成矩阵 |
| \( \operatorname{RoPE}(\cdot) \) | 应用旋转位置编码的操作 |
| \( [\cdot ; \cdot] \) | 拼接操作 |
| \( n_g \) | GQA 中的组数 |
| \( n \) | MoE 中的专家总数 |
| \( E_i \) | 第 \( i \) 个专家网络 |
| \( G(\cdot) \) | 门控网络函数 |
| \( p_i \) | 门控网络输出的第 \( i \) 个概率 |
| \( H^{(i)}(x) \) | Noisy Top-k Gating 中专家 \( i \) 的门控分数 |
| \( W_g, W_{\text{noise}} \) | MoE 门控网络和噪声网络的权重矩阵 |
| \( \epsilon \) | 标准高斯噪声 |
| \( \text{softplus}(\cdot) \) | Softplus 激活函数 |
| \( k \) | MoE 中每个 token 选择的专家数量 |
| \( \text{topk}(\cdot, k) \) | 选择前 k 个最大值的函数 |
| \( \mathcal{L}_{\text{aux}} \) | MoE 辅助损失 |
| \( w_{\text{aux}} \) | 辅助损失权重 |
| \( \text{CV}(\cdot) \) | 变异系数 |
| \( N_s, N_r \) | DeepSeekMoE 中共享专家和路由专家的数量 |
| \( \operatorname{FFN}_i^{(s)}(\cdot), \operatorname{FFN}_i^{(r)}(\cdot) \) | 第 \( i \) 个共享专家和路由专家函数 |
| \( K_r \) | DeepSeekMoE 中激活的路由专家数量 |
| \( g_{i,t} \) | 第 \( i \) 个专家对第 \( t \) 个 token 的门控值 |
| \( g_{i,t}' \) | TopK 选择后的原始门控值（V3） |
| \( s_{i,t} \) | 第 \( t \) 个 token 对第 \( i \) 个专家的亲和度分数 |
| \( \mathbf{e}_i \) | 第 \( i \) 个路由专家的中心向量 |
| \( M \) | 设备/节点限制路由中的设备/节点数上限 |
| \( \mathcal{L}_{\text{ExpBal}}, \mathcal{L}_{\text{DevBal}}, \mathcal{L}_{\text{CommBal}} \) | 专家级、设备级、通信级负载均衡损失 |
| \( f_i, P_i \) | 专家 \( i \) 的负载分数和平均亲和度 |
| \( \alpha_1, \alpha_2, \alpha_3 \) | 负载均衡损失的超参数 |
| \( T \) | 序列中的 token 数量 |
| \( D \) | 设备/节点组的数量 |
| \( \mathcal{E}_i \) | 第 \( i \) 个设备/节点上的专家组 |
| \( f_i', P_i' \) | 设备组 \( i \) 的平均负载分数和总亲和度 |
| \( f_i'', P_i'' \) | 发送到设备 \( i \) 的 token 比例和设备组 \( i \) 的总亲和度 |
| \( b_i \) | 第 \( i \) 个专家的偏置项 (aux-loss-free balancing) |
| \( \gamma \) | 偏置项更新速度 |
| \( \mathcal{L}_{\text{Bal}} \) | 序列级负载均衡损失 |
| \( \alpha \) | 序列级负载均衡损失的超参数 |
| \( D_{MTP} \) | MTP 预测深度 |
| \( \operatorname{Emb}(\cdot), \operatorname{OutHead}(\cdot) \) | 共享嵌入层和输出头 (MTP) |
| \( \operatorname{TRM}_k(\cdot) \) | 第 \( k \) 个 MTP 模块的 Transformer 块 |
| \( M_k \) | 第 \( k \) 个 MTP 模块的投影矩阵 |
| \( \mathbf{h}_i^k \) | 第 \( i \) 个 token 在第 \( k \) 个 MTP 深度的表示 |
| \( \mathbf{h}_i^{\prime k} \) | 第 \( k \) 个 MTP 模块的 Transformer 块的输入 |
| \( P_{i+k+1}^k \) | 第 \( k \) 个 MTP 模块对第 \( i+k+1 \) 个 token 的预测概率分布 |
| \( V \) | 词汇表大小 |
| \( \mathcal{L}_{\text{MTP}}^k \) | 第 \( k \) 个 MTP 深度的交叉熵损失 |
| \( \mathcal{L}_{\text{MTP}} \) | 总 MTP 损失 |
| \( \lambda \) | MTP 损失的权重因子 |
| \( \mathcal{J}_{GRPO}(\theta) \) | GRPO 目标函数 |
| \( A_i \) | 相对优势值 (GRPO) |
| \( \varepsilon \) | PPO/GRPO 中的裁剪超参数 |
| \( \beta \) | KL 散度惩罚项系数 |
| \( \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \) | KL 散度 |
| \( \pi_\theta, \pi_{\theta_{old}}, \pi_{ref} \) | 当前策略、旧策略、参考策略模型 |
| \( r_i \) | 第 \( i \) 个输出的奖励值 |
| \( \mathbb{1}(\cdot) \) | 指示函数 |

## 核心架构

DeepSeek-V2 和 V3 均基于 Transformer 架构，但在注意力和前馈网络（FFN）部分采用了创新设计比如 MLA 和 DeepseekMoE，以平衡性能、训练成本和推理效率。下图展示了 DeepSeek-V2 和 V3 的架构。


{{< figure
    src="deepseek_architecture.png"
    caption="Fig. 3. Illustration of the architecture of DeepSeek-V2 and DeepSeek-V3. MLA ensures efficient inference by significantly reducing the KV cache for generation, and DeepSeekMoE enables training strong models at an economical cost through the sparse architecture. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

### 多头隐注意力 (MLA)

传统的 Transformer 模型通常采用**多头注意力（Multi-Head Attention, MHA）**([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))，但在生成过程中，其庞大的 KV cache 成为限制推理效率的瓶颈。为了解决这个问题，研究者提出了**多查询注意力（Multi-Query Attention, MQA）**([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) 和**分组查询注意力（Grouped-Query Attention, GQA）**([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))。这些方法虽然减少了 KV 缓存，但往往以牺牲模型性能为代价。

DeepSeek-V2 和 V3 采用了创新的 **多头隐注意力（Multi-head Latent Attention, MLA）** 机制。MLA 的核心思想是 **低秩键值联合压缩 (Low-Rank Key-Value Joint Compression)**。


{{< figure
    src="mla.png"
    caption="Fig. 4. Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}

#### MHA 回顾

标准 MHA 首先通过三个投影矩阵 \(W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}\) 将输入 \(\mathbf{h}_t \in \mathbb{R}^d\) 转换为查询 \(\mathbf{q}_t\)、键 \(\mathbf{k}_t\) 和值 \(\mathbf{v}_t \in \mathbb{R}^{d_h n_h}\)：
\[
\begin{aligned}
\mathbf{q}_{t} &= W^{Q} \mathbf{h}_{t}, \\
\mathbf{k}_{t} &= W^{K} \mathbf{h}_{t}, \\
\mathbf{v}_{t} &= W^{V} \mathbf{h}_{t}.
\end{aligned}
\]
然后将 \(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t\) 切分为 \(n_h\) 个头，进行多头注意力计算：
\[
\begin{aligned}
& [\mathbf{q}_{t, 1} ; \mathbf{q}_{t, 2} ; \ldots ; \mathbf{q}_{t, n_{h}}] = \mathbf{q}_{t}, \\
& [\mathbf{k}_{t, 1} ; \mathbf{k}_{t, 2} ; \ldots ; \mathbf{k}_{t, n_{h}}] = \mathbf{k}_{t}, \\
& [\mathbf{v}_{t, 1} ; \mathbf{v}_{t, 2} ; \ldots ; \mathbf{v}_{t, n_{h}}] = \mathbf{v}_{t}, \\
& \mathbf{o}_{t, i} = \sum_{j=1}^{t} \operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t, i}^{T} \mathbf{k}_{j, i}}{\sqrt{d_{h}}}\right) \mathbf{v}_{j, i}, \\
& \mathbf{u}_{t} = W^{O}\left[\mathbf{o}_{t, 1} ; \mathbf{o}_{t, 2} ; \ldots ; \mathbf{o}_{t, n_{h}}\right],
\end{aligned}
\]
其中 \(\mathbf{q}_{t, i}, \mathbf{k}_{t, i}, \mathbf{v}_{t, i} \in \mathbb{R}^{d_h}\) 分别是第 \(i\) 个头的查询、键、值，\(W^O \in \mathbb{R}^{d \times d_h n_h}\) 是输出投影矩阵。推理时，需要缓存所有 \(t\) 步的键和值，每个 token 需要缓存 \(2 n_h d_h l\) 个元素（\(l\) 为层数），这构成了巨大的 KV 缓存开销。

#### 低秩键值联合压缩

MLA 通过引入一个低维的隐向量 \(\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}\) 来联合压缩键和值，其中 \(d_c \ll d_h n_h\)：
\[
\begin{aligned}
\boxed{\mathbf{c}_{t}^{K V}} &= W^{D K V} \mathbf{h}_{t}, \\
\mathbf{k}_{t}^{C} &= W^{U K} \mathbf{c}_{t}^{K V}, \\
\mathbf{v}_{t}^{C} &= W^{U V} \mathbf{c}_{t}^{K V}.
\end{aligned}
\]
这里 \(W^{DKV} \in \mathbb{R}^{d_c \times d}\) 是下投影矩阵，\(W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}\) 分别是键和值的上投影矩阵。推理时，MLA **只需要缓存压缩后的隐向量 \(\mathbf{c}_t^{KV}\)**（以及后面提到的解耦 RoPE 键 \(\mathbf{k}_t^R\)），极大地减少了 KV 缓存量。

为了减少训练时的激活内存，MLA 也对查询进行了类似的低秩压缩：
\[
\begin{aligned}
\mathbf{c}_{t}^{Q} &= W^{D Q} \mathbf{h}_{t}, \\
\mathbf{q}_{t}^{C} &= W^{U Q} \mathbf{c}_{t}^{Q},
\end{aligned}
\]
其中 \(\mathbf{c}_t^Q \in \mathbb{R}^{d_c'}\) 是查询的压缩隐向量，\(d_c' \ll d_h n_h\)，\(W^{DQ} \in \mathbb{R}^{d_c' \times d}\) 和 \(W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}\) 分别是查询的下投影和上投影矩阵。

#### 解耦旋转位置编码

标准的**旋转位置编码（RoPE）**([Su et al., 2024](https://arxiv.org/abs/2104.09864)) 直接应用于键和查询，但这与 MLA 的低秩 KV 压缩不兼容。如果在压缩后的键 \(\mathbf{k}_t^C\) 上应用 RoPE，那么上投影矩阵 \(W^{UK}\) 会与位置相关的 RoPE 矩阵耦合，导致推理时无法将其吸收到 \(W^Q\) 中，必须重新计算所有前缀 token 的键，严重影响效率。

为了解决这个问题，MLA 提出了**解耦 RoPE (Decoupled RoPE)** 策略。它引入了额外的多头查询 \(\mathbf{q}_{t, i}^R \in \mathbb{R}^{d_h^R}\) 和一个共享的键 \(\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}\) 来专门承载 RoPE 信息：

\[
\begin{aligned}
\left[\mathbf{q}_{t,1}^R;\,\mathbf{q}_{t,2}^R;\,\dots;\,\mathbf{q}_{t,n_h}^R\right]
= \mathbf{q}_t^R
&= \operatorname{RoPE}\bigl(W^{Q R}\,\mathbf{c}_t^Q\bigr),\\
\boxed{\mathbf{k}_t^R}
&= \operatorname{RoPE}\bigl(W^{K R}\,\mathbf{h}_t\bigr).
\end{aligned}
\]
这里 \(W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}\) 和 \(W^{KR} \in \mathbb{R}^{d_h^R \times d}\) 是生成解耦查询和键的矩阵。
然后将压缩得到的键/查询部分 (\(C\)) 与解耦的 RoPE 部分 (\(R\)) 拼接起来形成最终的键和查询：
\[
\begin{aligned}
\mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C} ; \mathbf{q}_{t, i}^{R}], \\
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C} ; \mathbf{k}_{t}^{R}].
\end{aligned}
\]
最终的注意力计算变为：
\[
\begin{aligned}
\mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t, i}^{T} \mathbf{k}_{j, i}}{\sqrt{d_{h}+d_{h}^{R}}}\right) \mathbf{v}_{j, i}^{C}, \\
\mathbf{u}_{t} &= W^{O}\left[\mathbf{o}_{t, 1} ; \mathbf{o}_{t, 2} ; \ldots ; \mathbf{o}_{t, n_{h}}\right].
\end{aligned}
\]
推理时，除了缓存 \(\mathbf{c}_t^{KV}\)，还需要缓存解耦的 RoPE 键 \(\mathbf{k}_t^R\)。因此，DeepSeek-V2/V3 每个 token 总共需要缓存 \((d_c + d_h^R)l\) 个元素。

#### MLA 推理中的矩阵吸收

MLA 的一个关键优势在于推理效率的提升，这部分得益于矩阵乘法结合律允许将上投影矩阵 \(W^{UK}\) 和 \(W^{UV}\) “吸收”掉，避免显式计算完整的键 \(\mathbf{k}_t^C\) 和值 \(\mathbf{v}_t^C\)。

**1. 吸收 \(W^{UK}\) (优化注意力分数计算):**

注意力分数计算的核心是查询和键的点积 \(\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}\)。关注其中由压缩向量生成的 \(C\) 部分的点积：
\[
(\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C
\]
将 \(\mathbf{k}_{j,i}^C = W^{UK} \mathbf{c}_j^{KV}\) 代入：
\[
(\mathbf{q}_{t,i}^C)^T (W^{UK} \mathbf{c}_j^{KV})
\]
根据矩阵乘法结合律 \((AB)C = A(BC)\) 和转置性质 \((AB)^T = B^T A^T\)，可以将上式改写为：
\[
(\mathbf{q}_{t,i}^C)^T (W^{UK} \mathbf{c}_j^{KV}) = ((W^{UK})^T \mathbf{q}_{t,i}^C)^T \mathbf{c}_j^{KV}
\]
这个变换的意义在于：不再需要用 \(W^{UK}\) 作用于缓存的 \(\mathbf{c}_j^{KV}\) 来得到 \(\mathbf{k}_{j,i}^C\)。相反，可以先计算一个“有效查询” \(\tilde{\mathbf{q}}_{t,i}^C = (W^{UK})^T \mathbf{q}_{t,i}^C\)，然后直接用这个有效查询与缓存的隐向量 \(\mathbf{c}_j^{KV}\) 进行点积。

原始查询 \(\mathbf{q}_{t,i}^C\) 是通过 \(W^{UQ}\) 和 \(W^{DQ}\) 从 \(\mathbf{h}_t\) 计算得到的 (\(\mathbf{q}_{t,i}^C = (W^{UQ} W^{DQ} \mathbf{h}_t)_i\))。因此，从 \(\mathbf{h}_t\) 到有效查询 \(\tilde{\mathbf{q}}_{t,i}^C\) 的整个计算过程可以看作是一个新的、合并了 \(W^{UK}\) 的有效查询投影操作。在实际实现中，这意味着计算 \(\mathbf{q}_{t,i}^C\) 后，可以再左乘 \((W^{UK})^T\)，或者更高效地，将 \((W^{UK})^T\) 合并到生成查询的原始矩阵 \(W^Q\)（或 \(W^{UQ}W^{DQ}\)）中，形成一个新的查询投影矩阵 \(\tilde{W}^Q = (W^{UK})^T W^{UQ} W^{DQ}\)。

关键在于，涉及 \(W^{UK}\) 的计算被移到了查询侧，在计算注意力分数之前一次性完成，而无需在每次查询时都用 \(W^{UK}\) 从缓存的 \(\mathbf{c}_j^{KV}\) 中恢复 \(\mathbf{k}_{j,i}^C\)。

**2. 吸收 \(W^{UV}\) (优化加权求和):**

注意力头的输出 \(\mathbf{o}_{t,i}\) 是注意力权重 (记作 \(w_{ij}\)) 与值 \(\mathbf{v}_{j,i}^C\) 的加权和：
\[
\mathbf{o}_{t, i} = \sum_{j=1}^{t} w_{ij} \cdot \mathbf{v}_{j, i}^{C}
\]
将 \(\mathbf{v}_{j,i}^C = (W^{UV} \mathbf{c}_j^{KV})_i\) 代入（这里 \(( \cdot )_i\) 表示属于第 \(i\) 个头的部分）：
\[
\mathbf{o}_{t, i} = \sum_{j=1}^{t} w_{ij} \cdot (W^{UV} \mathbf{c}_j^{KV})_i
\]
最终的注意力层输出 \(\mathbf{u}_t\) 是所有头的输出 \(\mathbf{o}_{t,i}\) 拼接后通过输出矩阵 \(W^O\) 投影得到的：
\[
\mathbf{u}_{t} = W^{O}\left[\mathbf{o}_{t, 1} ; \ldots ; \mathbf{o}_{t, n_{h}}\right] = W^{O} \begin{bmatrix} \sum_{j} w_{1j} (W^{UV} \mathbf{c}_j^{KV})_1 \\ \vdots \\ \sum_{j} w_{n_h j} (W^{UV} \mathbf{c}_j^{KV})_{n_h} \end{bmatrix}
\]
由于矩阵乘法的线性性质（\(A(B+C) = AB + AC\) 以及 \(A(cB) = c(AB)\)），可以将 \(W^{UV}\) 从求和中“提出”（这里是为了直观理解，实际操作是矩阵层面的）：
\[
\mathbf{u}_{t} \approx W^{O} W^{UV} \left( \sum_{j=1}^{t} \begin{bmatrix} w_{1j} (\mathbf{c}_j^{KV})_1 \\ \vdots \\ w_{n_h j} (\mathbf{c}_j^{KV})_{n_h} \end{bmatrix} \right)
\]
（注意：这里的 \((\mathbf{c}_j^{KV})_i\) 只是示意，实际计算中是直接对完整的 \(\mathbf{c}_j^{KV}\) 操作，但原理相同，即先对 \(\mathbf{c}_j^{KV}\) 进行加权求和，再应用 \(W^{UV}\) 和 \(W^O\)）。

令有效输出矩阵 \(\tilde{W}^O = W^O W^{UV}\)。这意味着可以先计算注意力权重与隐向量 \(\mathbf{c}_j^{KV}\) 的加权和（得到一个维度为 \(d_c\) 的中间结果 \(\tilde{\mathbf{o}}_t = \sum_j w_{ij} \mathbf{c}_j^{KV}\)），然后直接用这个合并后的有效输出矩阵 \(\tilde{W}^O\) 进行最终投影得到 \(\mathbf{u}_t\)。同样，涉及 \(W^{UV}\) 的计算被合并到了最后的输出投影步骤，无需在计算加权和时从 \(\mathbf{c}_j^{KV}\) 恢复 \(\mathbf{v}_{j,i}^C\)。

**总结:** 通过矩阵吸收，MLA 在推理时避免了从缓存的低维隐向量 \(\mathbf{c}_j^{KV}\) 重复计算高维的键 \(\mathbf{k}_{j,i}^C\) 和值 \(\mathbf{v}_{j,i}^C\)，显著提高了计算效率。实际缓存的只有 \(\mathbf{c}_t^{KV}\) 和 \(\mathbf{k}_t^R\)。


#### KV 缓存对比

下表比较了不同注意力机制的每 Token KV 缓存对比。\(n_h\) 是注意力头数，\(d_h\) 是每头维度，\(l\) 是层数，\(n_g\) 是 GQA 组数，\(d_c\) 和 \(d_h^R\) 是 MLA 的 KV 压缩维度和解耦 RoPE 维度。对于 DeepSeek-V2，\(d_c = 4d_h\)，\(d_h^R = d_h/2\)，其 KV 缓存相当于 \(n_g=2.25\) 的 GQA，但性能优于 MHA。DeepSeek-V3 沿用了类似配置。

| 注意力机制 | 每 Token KV 缓存大小 (\# 元素) | 能力 |
| :--- | :---: | :---: |
| Multi-Head Attention (MHA) | \(2 n_{h} d_{h} l\) | 强 |
| Grouped-Query Attention (GQA) | \(2 n_{g} d_{h} l\) | 中等 |
| Multi-Query Attention (MQA) | \(2 d_{h} l\) | 弱 |
| Multi-head Latent Attention (MLA) | \(\bigl(d_{c} + d_{h}^{R}\bigr) l \approx \tfrac{9}{2} \, d_{h} \, l\) | 更强 |

下图表明，MLA 不仅显著减少了 KV 缓存，其性能甚至优于标准的 MHA。

{{< figure
    src="mla_vs_mha.png"
    caption="Fig. 5. Comparison between MLA and MHA on hard benchmarks. DeepSeek-V2 shows better performance than MHA, but requires a significantly smaller amount of KV cache. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434))"
    align="center"
    width="100%"
>}}


### 混合专家模型

在深入 DeepSeekMoE 之前，先回顾一下混合专家模型（MoE）的基础知识。

**混合专家模型(Mixture-of-Experts, MoE)**([Shazeer et al. 2017](https://arxiv.org/abs/1701.06538)) 是一种稀疏激活模型，它通过结合多个独立的“专家”网络和一个门控网络(Gating Network)，在不显著增加计算成本的前提下，大幅提升了模型的参数量和性能。MoE 的核心思想是**稀疏激活(Sparse Activation)**，即对于每个输入样本，仅激活部分专家网络，而不是整个模型。这种方法既提高了计算效率，又增强了模型的表达能力，使其在 LLMs 中表现出色。

MoE 设计灵感来源于[集成学习(Ensemble learning)](https://en.wikipedia.org/wiki/Ensemble_learning), 一种将复杂任务分解为多个子任务并由不同模型协作完成的技术。在 MoE 中，这些“子任务”由多个独立的专家网络处理，而门控网络则负责根据输入样本的特征动态选择最适合的专家。这种分工合作的机制类似于人类社会中的专家团队：不同领域的专家针对特定问题提供专业意见，最终综合得出结果。


{{< figure
    src="moe.png"
    caption="Fig. 6. Illustration of a mixture-of-experts(MoE) layer. Only 2 out of experts are selected and activated by the gating network. (Image source: [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538))"
    align="center"
    width="100%"
>}}  


### MoE 核心组件

一个典型的 MoE 包含以下组件：

* **专家网络(Experts):**  一组独立的神经网络 $\{E_1, E_2, ..., E_n\}$，每个专家网络 $E_i$ 可以是任意类型的神经网络，例如 FFN, CNN, RNN 等。专家网络的数量 $n$ 可以很大，例如几十个、几百个甚至几千个。
* **门控网络(Gating Network):**  一个可训练的神经网络 $G$，用于根据输入样本 $x$ 学习一个概率分布，决定激活哪些专家。门控网络的输入是输入样本 $x$，输出是一个 $n$ 维的概率向量 $p = G(x) = [p_1, p_2, ..., p_n]$，其中 $p_i$ 表示激活专家 $E_i$ 的概率。
* **专家输出聚合(Expert Output Aggregation):**  根据门控网络的输出概率分布，将激活的专家网络的输出进行加权求和，得到 MoE 层的最终输出 $y$。

### Noisy Top-k Gating

为了实现稀疏激活并确保专家使用均衡，MoE 通常采用 **Noisy Top-k Gating** 作为门控机制。这种方法通过引入噪声和 top-k 选择，既保证了计算效率，又避免了专家负载不均的问题。以下是其详细工作流程：


1. **门控分数计算:**

对于输入样本 $x$，门控网络首先计算每个专家的门控分数 $H^{(i)}(x)$。这一分数包含两部分：线性变换和噪声项，公式如下：

$$
H^{(i)}(x) =(x W_g)^{(i)} + \epsilon \cdot \text{softplus}\left((x W_{\text{noise}})^{(i)} \right), \quad \epsilon \sim \mathcal{N}(0, 1)
$$

- **参数说明**：
  - $W_g \in \mathbb{R}^{d \times n}$：门控网络的可训练权重矩阵，$d$ 是输入特征维度，$n$ 是专家数量。
  - $W_{\text{noise}} \in \mathbb{R}^{d \times n}$：用于生成噪声的权重矩阵。
  - $\epsilon \sim \mathcal{N}(0, 1)$：标准高斯噪声，增加门控随机性。
  - $\text{softplus}(x) = \log(1 + e^x)$：平滑激活函数，确保噪声非负。

噪声的引入避免了门控网络总是选择固定的专家，增强了模型的鲁棒性和多样性。

2. **Top-k 选择:**

计算出门控分数向量 $H(x) = [H^{(1)}(x), H^{(2)}(x), \dots, H^{(n)}(x)]$ 后，门控网络选择其中值最大的前 $k$ 个专家(通常 $k \ll n$)。这一步骤通过 $\text{topk}(v, k)$ 函数实现：

$$
\text{topk}^{(i)}(v, k) = 
\begin{cases} 
v^{(i)} & \text{if } v^{(i)} \text{ is in the top } k \text{ elements of } v \\
-\infty & \text{otherwise}
\end{cases}
$$

将非 Top-k 专家的分数设为 $-\infty$，确保后续 softmax 操作中这些专家的概率为 0，实现稀疏性。

3. **Softmax 归一化:**

对 Top-k 专家的门控分数进行 softmax 归一化，得到稀疏的概率分布 $G(x)$：

$$
G(x) = \text{softmax}\left( \text{topk}(H(x), k) \right)
$$

只有 Top-k 个专家的概率非零，其余为 0。例如，若 $n=100, k=2$，则 98 个专家的概率为 0。

4. **加权求和:**

将 Top-k 个专家的输出按概率加权求和，得到 MoE 层的输出：

$$
y = \sum_{i=1}^{n} G^{(i)}(x) E_i(x)
$$

由于只有 $k$ 个专家被激活，计算量远低于激活所有 $n$ 个专家。


### 辅助损失

为了**避免门控网络过度偏向少数专家**，MoE 引入了**辅助损失(Auxiliary Loss)**([Shazeer et al. 2017](https://arxiv.org/abs/1701.06538))，鼓励所有专家被均匀使用。一种常用方法是基于专家使用率的[变异系数(Coefficient of Variation, CV)](https://en.wikipedia.org/wiki/Coefficient_of_variation)的平方：

$$
\mathcal{L}_{\text{aux}} = w_{\text{aux}} \cdot \text{CV}\left( \sum_{x \in X} G(x) \right)^2
$$

- **参数说明**：  
  - $X$：一个 mini-batch 的输入样本。  
  - $\sum_{x \in X} G(x)$：统计每个专家在 mini-batch 中的激活次数。  
  - $\text{CV}$：标准差与均值的比值，衡量专家使用分布的均匀性。  
  - $w_{\text{aux}}$：辅助损失的权重，需手动调整。  

- **作用**：通过最小化 $\mathcal{L}_{\text{aux}}$，模型优化专家选择的均衡性，避免某些专家被过度使用而其他专家闲置。

### GShard

**GShard**([Lepikhin et al. 2020](https://arxiv.org/abs/2006.16668))主要对 MoE 层进行分片，将 MoE 层中的专家网络 $\{E_1, E_2, ..., E_n\}$ 分散到多个 TPU 设备上。例如，如果有 $P$ 个 TPU 设备，可以将专家网络划分为 $P$ 组，每组专家网络分配到一个 TPU 设备上。Transformer 模型的其他层(例如自注意力层、LayerNorm 层) 则在所有 TPU 设备上复制。

**GShard 的改进门控机制:**

GShard 在 Noisy Top-k Gating 的基础上，进行了一些改进，以提高门控机制的性能和稳定性：

- **专家容量(Expert Capacity):**  
  为了避免专家过载，GShard 引入了专家容量限制。每个专家网络都有一个容量上限，表示它最多可以处理的 token 数量。如果一个 token 被路由到一个已经达到容量上限的专家网络，则该 token 会被标记为 "overflowed"，门控输出会被设置为零向量，表示该 token 不会被路由到任何专家网络。

- **局部组分发(Local Group Dispatching):**  
  为了提高门控效率，GShard 将 token 分组，在组级别强制执行专家容量限制。例如，将 mini-batch 中的 token 划分为多个局部组，每个局部组包含一定数量的 token。门控网络为每个局部组选择 top-k 个专家网络，并确保每个专家网络在一个局部组内处理的 token 数量不超过其容量上限。

- **辅助损失(Auxiliary Loss):**  
  GShard 也使用了辅助损失函数来平衡专家负载。与原始 MoE 模型的辅助损失不同，GShard 的辅助损失旨在最小化每个专家网络路由到的数据比例的均方误差，更加直接地衡量专家负载平衡程度。

- **随机路由(Random Routing):**  
  为了增加路由的随机性，GShard 在选择 top-k 个专家网络时，引入了随机路由机制。除了选择最佳的 top-k 个专家网络外，GShard 还会以一定的概率随机选择次优的专家网络，增加专家网络的多样性，提高模型的泛化能力。

下面是 GShard 的核心算法流程:

{{< figure
    src="gshard.png"
    caption="Fig. 7. Pseudo code of the group-level top-2 gating mechanism with auxiliary loss in GShard. (Image source: [Lepikhin et al. 2020](https://arxiv.org/abs/2006.16668))"
    align="center"
    width="100%"
>}}  

### Switch Transformer

**Switch Transformer**([Fedus et al. 2021](https://arxiv.org/pdf/2101.03961)) 是 Google 提出的一个参数量达到**万亿**级别的 MoE 模型。其核心创新是将 Transformer 模型中的密集前馈网络(FFN) 层替换为稀疏的 Switch FFN 层。与 GShard 的 Top-2 Gating 不同，Switch Transformer 每个输入 token 只路由到一个专家网络，具有更高的稀疏性，进一步降低了计算成本，使得训练万亿参数模型成为可能。鼓励 token 路由在 $N$ 个专家之间更加均衡。Switch Transformer 的辅助损失基于实际路由比例与预测路由概率的乘积累加，具体公式如下：

$$
\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

- **参数说明**：  
  - $N$：专家的总数。  
  - $f_i$：路由到第 $i$ 个专家的 token 比例，定义为：  

    $$
    f_i = \frac{1}{T} \sum_{x \in B} 1\{\text{argmax } p(x) = i\}
    $$

  - $P_i$：gating 网络预测的第 $i$ 个专家的路由概率，定义为：  

    $$
    P_i = \frac{1}{T} \sum_{x \in B} p_i(x)
    $$

  - $T$：批次 $B$ 中的 token 总数。  
  - $\alpha$：辅助损失的权重超参数，通常设为 $10^{-2}$。  

通过最小化 $\text{loss}$，模型使实际路由比例 $f_i$ 与预测概率 $P_i$ 趋于一致，从而间接促进专家间的负载平衡，避免部分专家闲置。

{{< figure
    src="switch_transformer.png"
    caption="Fig. 8. Switch transformer. The sparse switch FFN layer is in the blue boxes. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}  

**Switch Router 机制:**

1. **路由预测:**  
   对于输入 token $x$，Switch Router 预测每个专家网络的路由概率 $p_i = G^{(i)}(x)$，其中 $i = 1, 2, ..., n$，n 是专家网络数量。

2. **专家选择:**  
   选择路由概率最高的专家网络作为最佳专家网络。Switch Transformer 采用 Top-1 路由策略，即每个 token 只路由到路由概率最高的专家网络。

3. **token 路由:**  
   将输入 token $x$ 路由到选择的最佳专家网络进行处理。

**Switch Transformer 的训练稳定性优化:**

为提升 Switch Transformer 的训练稳定性，论文提出了如下优化策略：

- **选择性精度(Selective Precision)**  
  在路由函数内部采用 FP32 精度既能提高训练稳定性，又能避免因 FP32 张量通信而产生的额外开销。具体来说，Switch Router 的计算过程全程使用 FP32，最终结果再转换为 FP16 以兼顾效率与精度。

- **更小初始化(Smaller Initialization)**  
  建议将 Transformer 的权重初始化尺度参数 $s$ 从 1 调整至 0.1。较小的初始化尺度有助于缓解训练初期的梯度爆炸风险，从而提升整体训练稳定性。具体实现为：从均值为 0、标准差为 $\sqrt{s/n}$(其中 $n$ 为输入单元数) 的截断正态分布中采样。

- **更高专家 Dropout(Higher Expert Dropout)**  
  在专家 FFN 层中采用较高的 dropout 率(例如 0.4)，而在非专家层则保持较低的 dropout 率(例如 0.1)，这种设置能有效防止过拟合，进而增强模型的泛化能力。下图实验结果显示，在 GLUE、CNNDM、SQuAD 和 SuperGLUE 等任务上，当专家层 dropout 率设为 0.4 时，模型表现最佳。


{{< figure
    src="switch_transformer_fine_tuning_result.png"
    caption="Fig. 9. Fine-tuning regularization results. A sweep of dropout rates while fine-tuning Switch Transformer models pre-trained on 34B tokens of the C4 data set(higher numbers are better). (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}  

Switch Transformers 论文中使用下图直观的展示了使用不同的并行技术如何分割模型权重和数据:

{{< figure
    src="switch_transformer_parallelism.png"
    caption="Fig. 10. An illustration of various parallelism strategies on how(Top) model weights and(Bottom) data are split over multiple GPU cores. In the top row, each color denotes a unique weight matrix. In the bottom row, different colors indicate different sets of tokens. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}  

### 专家选择

**专家选择(Expert Choice, EC)**([Zhou et al. 2022](https://arxiv.org/abs/2202.09368)) 是一种与 token 选择路由(如 GShard 的 top-2 或 Switch Transformer 的 top-1)相反的路由策略。在 token 选择路由中，每个 token 从所有专家中选择 top-k 个进行路由；而在专家选择路由中，每个专家从所有 token 中挑选 top-k 个进行处理。这种方法旨在解决 token 选择路由中的负载不均和 token 浪费问题，同时显著提高训练效率。下面是具体的计算过程：

1. **计算 token-to-expert 亲和度分数**  

   对于输入矩阵 $X \in \mathbb{R}^{n \times d}$，计算 token-to-expert 亲和度分数矩阵 $S \in \mathbb{R}^{n \times e}$ 的过程为：

   $$
   S = \text{softmax}(X \cdot W_g), \quad \text{where } W_g \in \mathbb{R}^{d \times e}.
   $$
   这里，$W_g$ 为门控权重矩阵，$e$ 为专家数量。

2. **专家选择 token**  

   每个专家从所有 token 中选择 top-k 个进行处理。通过对 $S^T$ 进行 top-k 选择：

   $$
   G, I = \text{top-}k(S^T, k),
   $$

   得到：
   - **门控矩阵 $G \in \mathbb{R}^{e \times k}$：** 记录专家选择的 token 对应的路由权重，其中 $G[i, j]$ 表示专家 $i$ 选择的第 $j$ 个 token 的权重；
   - **token 索引矩阵 $I \in \mathbb{R}^{e \times k}$：** 表示每个专家选择的 token 在输入中的索引。

3. **One-hot 编码**  

   将 token 索引矩阵 $I$ 转换为 one-hot 编码矩阵 $P \in \mathbb{R}^{e \times k \times n}$，用于后续计算：

   $$
   P = \operatorname{one}-\operatorname{hot}(I)
   $$

4. **构造 Gated FFN 层输入**  

   对于每个专家 $i$，其 gated FFN 层的输入为：

   $$
  (P \cdot X) \in \mathbb{R}^{e \times k \times d}.
   $$

EC 通过正则化限制每个 token 被路由到的专家数量，从而控制模型的稀疏性。一个常见的正则化目标如下：

$$
\begin{aligned}
& \max_{A} \langle S^{\top}, A \rangle + \lambda H(A) \\
& \text{s.t. } \forall i: \sum_{j'} A[i, j'] = k, \quad \forall j: \sum_{i'} A[i', j] \leq b, \quad \forall i,j: 0 \leq A[i, j] \leq 1,
\end{aligned}
$$

考虑的优化问题中定义了一个矩阵 $A$，其第 $i$ 行第 $j$ 列的元素表示第 $i$ 个专家是否选择了第 $j$ 个 token(取值 0 或 1)。由于该优化问题求解较为复杂，论文中采用 [Dijkstra 算法](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)(通过多次迭代获得近似解)来解决。

参数 $b$ 通常由批量中 token 总数 $n$ 与容量因子决定，其中容量因子表示每个 token 平均使用的专家数量。大多数实验采用较高的容量因子，实验结果表明，即使在容量降低的情况下，EC 整体表现仍优于传统的 top-1 token 选择路由，尽管 capped expert choice 略微降低了微调性能。

EC 的优势主要体现在以下两方面：
- **完美负载均衡：** 每个专家固定处理 $k$ 个 token，从而避免了部分专家过载而其他专家闲置的问题，实现了理想的负载均衡。
- **更高训练效率：** 实验表明，EC 能将训练收敛速度提升约 2 倍，相较于传统 token 选择路由具有更高的效率。

但 EC 也存在以下局限性：
- **批量大小要求：** 由于 EC 对 batch size 有较高要求，因此不适用于较小 batch size 的场景。
- **自回归生成限制：** 在自回归文本生成任务中，由于无法预知未来 token，EC 的 top-k 选择无法实现，因此不适用于此类任务。

### DeepSeekMoE

混合专家（MoE）模型通过将计算路由到特定的“专家”子网络来提高效率和性能。DeepSeek-V2 和 V3 在其 FFN（前馈网络）层中采用了名为 **DeepSeekMoE** ([Dai et al., 2024](https://arxiv.org/abs/2401.06066))架构 。与 GShard 等传统 MoE 架构相比，DeepSeekMoE 的核心思想是：

1.  **细粒度专家切分 (Fine-grained Expert Segmentation):** 将专家网络切分得更小。这旨在实现更高的专家特化能力和更精确的知识获取，因为每个专家可以专注于更窄的领域。
2.  **共享专家隔离 (Shared Expert Isolation):** 架构中包含一部分“共享专家”，这些专家由所有 token 处理，旨在处理通用知识。这减少了需要路由的“路由专家”之间的知识冗余，让路由专家可以更专注于特定知识。

#### 基本架构

对于 FFN 层的输入 token 表示 \(\mathbf{u}_t\)，DeepSeekMoE 的输出 \(\mathbf{h}_t'\) 通过结合共享专家和选定的路由专家的输出来计算：
\[
\mathbf{h}_{t}^{\prime} = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} \operatorname{FFN}_{i}^{(s)}(\mathbf{u}_{t}) + \sum_{i=1}^{N_{r}} g_{i, t} \operatorname{FFN}_{i}^{(r)}(\mathbf{u}_{t}),
\]
其中：
*   \(N_s\) 是共享专家的数量。
*   \(N_r\) 是路由专家的数量。
*   \(\operatorname{FFN}_i^{(s)}\) 是第 \(i\) 个共享专家网络。
*   \(\operatorname{FFN}_i^{(r)}\) 是第 \(i\) 个路由专家网络。
*   \(g_{i, t}\) 是第 \(t\) 个 token 分配给第 \(i\) 个路由专家的门控值（权重）。

门控值 \(g_{i,t}\) 的计算方式是 DeepSeek-V2 和 V3 之间的关键区别之一，它基于 token-to-expert 的亲和度分数 \(s_{i,t}\)，并通过 Top-K 路由机制选择激活哪些专家。

#### V2 与 V3 的门控机制和负载均衡对比

MoE 模型的一个核心挑战是负载均衡：确保所有专家都能得到有效利用，避免某些专家过载而其他专家空闲，这会影响训练稳定性和计算效率。DeepSeek-V2 和 V3 在门控机制和负载均衡策略上采取了不同的方法。

**1. 亲和度计算 (\(s_{i,t}\)) 与 Top-K 选择:**

*   **DeepSeek-V2:** 使用 Softmax 函数计算每个 token 对每个路由专家的亲和度分数。Top-K 选择直接基于这些亲和度分数 \(s_{i,t}\)。
    \[
    s_{i, t} = \operatorname{Softmax}_{i}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})
    \]
    其中 \(\mathbf{e}_i\) 是第 \(i\) 个路由专家的可学习中心向量。选择 \(s_{i,t}\) 最高的 \(K_r\) 个专家。

*   **DeepSeek-V3:** 使用 Sigmoid 函数计算亲和度分数。更重要的是，它引入了一个可学习的偏置项 \(b_i\) 用于每个路由专家。Top-K 选择是基于 **加偏置后的亲和度** \(s_{i,t} + b_i\)。
    \[
    s_{i, t} = \operatorname{Sigmoid}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})
    \]
    选择依据是 \(s_{i,t} + b_i\) 值最高的 \(K_r\) 个专家。

**2. 门控值计算 (\(g_{i,t}\)):**

*   **DeepSeek-V2:** 对于被 Top-K 选中的专家，其门控值 \(g_{i,t}\) 直接等于其原始亲和度分数 \(s_{i,t}\)。对于未被选中的专家，\(g_{i,t} = 0\)。
    \[
    g_{i, t}^{\prime} = \begin{cases} s_{i, t}, & s_{i, t} \in \operatorname{Topk}(\{s_{j, t}\}, K_{r}), \\ 0, & \text{otherwise}, \end{cases}
    \]
    \[
    g_{i, t} = g_{i, t}^{\prime} \quad (\text{V2 中不进行额外归一化})
    \]

*   **DeepSeek-V3:** 对于基于 \(s_{i,t} + b_i\) 被 Top-K 选中的专家，其门控值 \(g_{i,t}\) 是通过对这些被选中专家的 **原始亲和度** \(s_{i,t}\) 进行归一化得到的。偏置 \(b_i\) **仅用于路由选择**，不影响最终的加权求和。
    \[
    g_{i, t}^{\prime}= \begin{cases} s_{i, t}, & s_{i, t}+b_{i} \in \operatorname{Topk}\left(\left\{s_{j, t}+b_{j} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right) \\ 0, & \text{otherwise.} \end{cases}
    \]
    \[
    g_{i, t} = \frac{g_{i, t}^{\prime}}{\sum_{j=1}^{N_{r}} g_{j, t}^{\prime}} \quad (\text{对选中的专家亲和度进行归一化})
    \]

**3. 负载均衡策略:**

*   **DeepSeek-V2:**
    *   **主要策略：辅助损失** V2 引入了多种辅助损失项来显式地鼓励负载均衡:
        *   **专家级平衡损失 (\(\mathcal{L}_{\text{ExpBal}}\)):** 鼓励每个专家处理大致相等数量的 token。
            \[
            \begin{aligned}
            \mathcal{L}_{\text{ExpBal}} &= \alpha_{1} \sum_{i=1}^{N_{r}} f_{i} P_{i} \\
            f_{i} &= \frac{N_{r}}{K_{r} T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i) \\
            P_{i} &= \frac{1}{T} \sum_{t=1}^{T} s_{i, t}
            \end{aligned}
            \]
            其中 \(T\) 是 batch 中的 token 总数，\(f_i\) 是路由到专家 \(i\) 的 token 比例（相对于理想均衡状态），\(P_i\) 是专家 \(i\) 的平均亲和度分数，\(\alpha_1\) 是超参数。
        *   **设备级平衡损失 (\(\mathcal{L}_{\text{DevBal}}\)):** 鼓励将计算负载均匀分布到不同的设备组上（假设专家分布在 \(D\) 个设备组 \(\{\mathcal{E}_1, \dots, \mathcal{E}_D\}\)）。
            \[
            \begin{aligned}
            \mathcal{L}_{\text{DevBal}} &= \alpha_{2} \sum_{i=1}^{D} f_{i}^{\prime} P_{i}^{\prime} \\
            f_{i}^{\prime} &= \frac{1}{|\mathcal{E}_{i}|} \sum_{j \in \mathcal{E}_{i}} f_{j} \\
            P_{i}^{\prime} &= \sum_{j \in \mathcal{E}_{i}} P_{j}
            \end{aligned}
            \]
            其中 \(f_i'\) 是设备组 \(i\) 的平均负载分数，\(P_i'\) 是设备组 \(i\) 的总亲和度，\(\alpha_2\) 是超参数。
        *   **通信平衡损失 (\(\mathcal{L}_{\text{CommBal}}\)):** 鼓励发送到每个设备的 token 数量大致相等，以平衡 All-to-All 通信负载。
            \[
            \begin{aligned}
            \mathcal{L}_{\text{CommBal}} &= \alpha_{3} \sum_{i=1}^{D} f_{i}^{\prime \prime} P_{i}^{\prime \prime} \\
            f_{i}^{\prime \prime} &= \frac{D}{M T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ is sent to Device } i) \\
            P_{i}^{\prime \prime} &= \sum_{j \in \mathcal{E}_{i}} P_{j}
            \end{aligned}
            \]
            其中 \(f_i''\) 是发送到设备 \(i\) 的 token 比例（相对于理想均衡状态），\(P_i''\) 是设备组 \(i\) 的总亲和度，\(\alpha_3\) 是超参数。
    *   **路由限制：设备限制路由** 限制每个 token 最多只能路由到分布在 \(M\) 个不同设备上的专家。V2 中设 \(M=3\)。
    *   **Token 丢弃:** 在训练期间，如果某个设备接收到的 token 数量超过了预设的容量因子（通常略大于平均值），则会丢弃一部分具有最低路由权重（亲和度）的 token，以避免计算资源的浪费。但会保留约 10% 序列的 token 不被丢弃。

*   **DeepSeek-V3:**
    *   **主要策略：无辅助损失的负载均衡 (Auxiliary-Loss-Free Load Balancing)** V3 认为辅助损失会损害模型性能，因此采用了一种创新的**无辅助损失的负载均衡**([Wang et al., 2024](https://arxiv.org/abs/2408.15664))。它通过动态调整前面提到的可学习偏置项 \(b_i\) 来实现负载均衡：
        *   **偏置更新:** 在每个训练步骤之后，监控每个专家 \(i\) 在当前 batch 中处理的 token 数量。
            *   如果专家 \(i\) 过载（处理的 token 数 > Batch 总 token 数 / \(N_r\)），则降低其偏置：\(b_i \leftarrow b_i - \gamma\)。
            *   如果专家 \(i\) 欠载（处理的 token 数 < Batch 总 token 数 / \(N_r\)），则增加其偏置：\(b_i \leftarrow b_i + \gamma\)。
        *   \(\gamma\) 是一个小的正步长（偏置更新速率超参数）。通过这种方式，负载高的专家在后续路由中被选中的概率会降低，负载低的专家被选中的概率会增加，从而在批处理级别上动态平衡负载。
    *   **补充策略：序列级辅助损失 (\(\mathcal{L}_{\text{Bal}}\))** V3 仍然保留了一个**权重极小** (\(\alpha=0.0001\)) 的辅助损失，但它作用于**单个序列内部**的专家选择平衡，而不是整个 batch。这主要是为了防止在单个序列中出现极端不平衡的情况。
        \[
        \begin{gathered}
        \mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_{r}} f_{i} P_{i}, \\
        f_{i} = \frac{N_{r}}{K_{r} T_{seq}} \sum_{t=1}^{T_{seq}} \mathbb{1}\left(s_{i, t} \in \operatorname{Topk}\left(\left\{s_{j, t} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right)\right), \\
        s_{i, t}^{\prime} = \frac{s_{i, t}}{\sum_{j=1}^{N_{r}} s_{j, t}}, \quad P_{i} = \frac{1}{T_{seq}} \sum_{t=1}^{T_{seq}} s_{i, t}^{\prime}
        \end{gathered}
        \]
        注意这里的 \(f_i, P_i\) 是在单个序列（长度为 \(T_{seq}\)）上计算的，并且 \(s_{i,t}'\) 是在序列内对原始 \(s_{i,t}\) 归一化后的值。
    *   **路由限制：节点限制路由** 类似于 V2 的设备限制，但应用于节点级别。V3 中设 \(M=4\)。
    *   **无 Token 丢弃:** 由于基于偏置调整的负载均衡效果良好，V3 在训练和推理过程中均不丢弃任何 token。

**V3 策略的优势:**
V3 的无辅助损失策略旨在最小化负载均衡机制对模型最终性能的负面影响。通过动态调整偏置项进行批处理级别的负载均衡，相比 V2 中基于辅助损失的序列级均衡，约束更宽松。这允许专家在不同领域展现出更强的特化模式，因为路由决策不必在每个序列内部都严格遵循均衡分布。下图实验表明该策略在多个基准测试上优于基于辅助损失的方法。

{{< figure
    src="auxiliary_loss_free_result.png"
    caption="Fig. 11. Ablation results for the auxiliary-loss-free balancing strategy. Compared with the purely auxiliary-loss-based method, the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

辅助损失无关的负载均衡与序列级辅助损失的关键区别在于它们的均衡范围：批次级与序列级。相比序列级辅助损失，批次级均衡施加了更灵活的约束，因为它不强制每个序列内的领域平衡。这种灵活性使得专家能够更好地在不同领域中进行专门化。为了验证这一点，图中记录并分析了在 Pile 测试集不同领域上，基于辅助损失的 16B 基线模型和无辅助损失的 16B 模型的专家负载，可以观察到无辅助损失模型表现出更明显的专家专门化模式，符合预期。

{{< figure
    src="expert_load.png"
    caption="Fig. 12. Expert load of auxiliary-loss-free and auxiliary-loss-based models on three domains in the Pile test set. The auxiliary-loss-free model shows greater expert specialization patterns than the auxiliary-loss-based one. The relative expert load denotes the ratio between the actual expert load and the theoretically balanced expert load. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

---

#### DeepSeekMoE V2 vs V3 对比总结表

| 特性                     | DeepSeek-V2                                                                                                                                                              | DeepSeek-V3                                                                                                                                                                                             |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **亲和度计算 \(s_{i,t}\)** | \(\operatorname{Softmax}_{i}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})\)                                                                                                         | \(\operatorname{Sigmoid}(\mathbf{u}_{t}^{T} \mathbf{e}_{i})\)                                                                                                                                           |
| **TopK 选择依据**        | 原始亲和度 \(s_{i,t}\)                                                                                                                                                     | 加偏置后的亲和度 \(s_{i,t} + b_i\)                                                                                                                                                                        |
| **门控值计算 \(g_{i,t}\)**  | 对选中的专家，\(g_{i,t} = s_{i,t}\) (通常无额外归一化)                                                                                                                            | 对选中的专家，基于原始亲和度 \(s_{i,t}\) 进行归一化: \(g_{i, t} = \frac{s_{i, t}}{\sum_{j \in \text{Selected}} s_{j, t}}\)                                                                                             |
| **主要负载均衡策略**     | **辅助损失:** <br> - \(\mathcal{L}_{\text{ExpBal}}\) (专家级) <br> - \(\mathcal{L}_{\text{DevBal}}\) (设备级) <br> - \(\mathcal{L}_{\text{CommBal}}\) (通信级)                               | **无辅助损失:** <br> - 通过动态调整可学习偏置项 \(b_i\) (步长 \(\gamma\)) 实现批处理级均衡                                                                                                                            |
| **补充负载均衡**         | 无明确的补充策略                                                                                                                                                           | **序列级辅助损失** \(\mathcal{L}_{\text{Bal}}\) (权重 \(\alpha\) 极小, e.g., 0.0001)，防止单序列内极端不平衡                                                                                                     |
| **路由限制**             | **设备限制:** <br> 每个 token 最多路由到 \(M=3\) 个设备上的专家                                                                                                  | **节点限制:** <br> 每个 token 最多路由到 \(M=4\) 个节点上的专家                                                                                                                                  |
| **Token 丢弃**           | **是:** 训练时，为缓解计算瓶颈，会丢弃超出设备容量的 token 中亲和度最低的部分 (保留约10%序列不丢弃)                                                                                             | **否:** 训练和推理中均不丢弃 token                                                                                                                                                                      |
| **均衡粒度**             | 主要通过辅助损失在序列/Batch 级别强制均衡                                                                                                                                    | 主要通过偏置调整在 Batch 级别动态均衡，约束更宽松                                                                                                                                                           |
| **对模型性能影响**       | 辅助损失可能对模型性能产生负面影响                                                                                                                                             | 设计上旨在最小化均衡策略对性能的负面影响，允许更好的专家特化                                                                                                                                                   |

### 多 token 预测 (MTP)

为了进一步提升模型性能和数据效率，DeepSeek-V3 引入了**多 token 预测 (Multi-Token Prediction, MTP)** 训练目标 ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737) 启发)。标准的语言模型只预测下一个 token，而 MTP 让模型在每个位置预测未来多个（V3 中是 \(D_{MTP}=1\)，即预测下下个 token）token。

#### MTP 实现

MTP 通过 \(D_{MTP}\) 个顺序模块实现。第 \(k\) 个 MTP 模块 (\(k=1, \dots, D_{MTP}\)) 包含：
*   共享的嵌入层 \(\operatorname{Emb}(\cdot)\)
*   共享的输出头 \(\operatorname{OutHead}(\cdot)\)
*   独立的 Transformer 块 \(\operatorname{TRM}_k(\cdot)\)
*   独立的投影矩阵 \(M_k \in \mathbb{R}^{d \times 2d}\)

对于输入序列中的第 \(i\) 个 token \(t_i\)，在第 \(k\) 个预测深度：
1.  将第 \(i\) 个 token 在第 \(k-1\) 深度的表示 \(\mathbf{h}_i^{k-1}\)（\(k=1\) 时为主模型输出）与第 \(i+k\) 个 token 的嵌入 \(\operatorname{Emb}(t_{i+k})\) 拼接，并通过投影矩阵 \(M_k\) 得到组合表示 \(\mathbf{h}_i^{\prime k}\)：
    \[
    \mathbf{h}_{i}^{\prime k} = M_{k}[\operatorname{RMSNorm}(\mathbf{h}_{i}^{k-1}) ; \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]
    \]
2.  将组合表示输入到第 \(k\) 个 Transformer 块，得到当前深度的输出表示 \(\mathbf{h}_i^k\)：
    \[
    \mathbf{h}_{1: T-k}^{k} = \operatorname{TRM}_{k}(\mathbf{h}_{1: T-k}^{\prime k})
    \]
3.  使用共享输出头预测第 \(i+k+1\) 个 token 的概率分布 \(P_{i+k+1}^k \in \mathbb{R}^V\)：
    \[
    P_{i+k+1}^{k} = \operatorname{OutHead}(\mathbf{h}_{i}^{k})
    \]

关键在于，这种实现方式**保持了每个预测深度的完整因果链**，与并行预测多个 token 的方法不同。

{{< figure
    src="mtp.png"
    caption="Fig. 13. Illustration of our Multi-Token Prediction (MTP) implementation. They keep the complete causal chain for the prediction of each token at each depth. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

#### MTP 训练目标

计算每个预测深度 \(k\) 的交叉熵损失 \(\mathcal{L}_{\text{MTP}}^k\)：
\[
\mathcal{L}_{\text{MTP}}^{k} = \operatorname{CrossEntropy}(P_{2+k: T+1}^{k}, t_{2+k: T+1}) = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_{i}^{k}[t_{i}]
\]
总的 MTP 损失是所有深度损失的加权平均：
\[
\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D_{MTP}} \sum_{k=1}^{D_{MTP}} \mathcal{L}_{\text{MTP}}^{k}
\]
其中 \(\lambda\) 是权重因子（V3 中前期为 0.3，后期为 0.1）。这个损失会加到主模型的标准 next-token prediction 损失上。

#### MTP 推理

MTP 主要用于提升主模型性能。推理时，可以**直接丢弃 MTP 模块**，主模型可以独立工作。或者也可以利用 MTP 模块进行**投机解码** ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Xia et al., 2023](https://arxiv.org/abs/2203.16487))，以加速生成过程。V3 的实验表明，第二个 token 的接受率在 85%-90% 之间，可将解码速度提升约 1.8 倍。


## 基础设施与训练效率

DeepSeek-V3 的高效训练与部署得益于算法、框架和硬件的协同设计。

### 计算集群

DeepSeek-V3 在一个配备了 **2048 块 NVIDIA H800 GPU** 的集群上进行训练。
*   **节点内部:** 每个节点包含 8 块 H800 GPU，通过 **NVLink** 和 **NVSwitch** 高速互联。
*   **节点之间:** 不同节点间使用 **InfiniBand (IB)** 网络进行通信。

### 训练框架

DeepSeek-V3 的训练基于自研的高效轻量级框架 **HAI-LLM**。整体上采用了：
*   **16 路流水线并行 (Pipeline Parallelism, PP)** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241))
*   **64 路专家并行 (Expert Parallelism, EP)** (跨 8 节点) ([Lepikhin et al., 2021](https://arxiv.org/abs/2006.16668))
*   **ZeRO-1 数据并行 (Data Parallelism, DP)** ([Rajbhandari et al., 2020](https://arxiv.org/pdf/1910.02054))

为了实现高效训练，Deepseek 进行了细致的工程优化：
1.  设计了 **DualPipe** 算法以实现高效流水线并行，减少气泡并重叠计算与通信，解决了跨节点专家并行带来的重通信开销问题。
2.  开发了高效的**跨节点 All-to-all 通信 Kernel**，充分利用 IB 和 NVLink 带宽，并节省用于通信的 SM 资源。
3.  细致优化了训练过程中的**内存占用**，使得可以在**不使用张量并行 (Tensor Parallelism, TP)** 的情况下训练 DeepSeek-V3。

#### DualPipe 与计算通信重叠

*   **挑战:** 跨节点专家并行导致计算通信比接近 1:1，效率不高。

{{< figure
    src="forward_backward_chucks.png"
    caption="Fig. 17. Overlapping strategy for a pair of forward and backward chunks with misaligned transformer block boundaries. Orange: forward, green: backward for input, blue: backward for weights, purple: PP communication, red: barriers. Both all-to-all and PP communications are fully hidden. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

*   **核心思想:** 重叠一对独立的前向和反向 chunk 内的计算和通信。将每个 chunk 分解为 **Attention**、**All-to-all Dispatch**、**MLP**、**All-to-all Combine** 四个组件（反向的 Attention 和 MLP 进一步细分为 backward for input 和 backward for weights，类似 **ZeroBubble** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241)）。通过重排这些组件并手动调整用于通信与计算的 GPU SM 比例，实现 All-to-all 和 PP 通信的完全隐藏。
*   **调度:** 采用双向流水线调度，同时从流水线的两端输入微批次，大部分通信可以被完全重叠。

{{< figure
    src="dualpipe.png"
    caption="Fig. 18. Example DualPipe scheduling with 8 PP ranks and 20 micro-batches in both directions. The reverse-direction micro-batches mirror the forward ones, so their batch IDs are omitted for simplicity. Two cells within a shared black border represent mutually overlapped computation and communication. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

*   **优势:**
    *   即使在没有重通信负担的一般场景下也具有效率优势。
    *   相比 **ZB1P** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241)) 和 **1F1B** ([Harlap et al., 2018](https://arxiv.org/abs/1806.03377))，显著减少流水线气泡，仅增加 \(\frac{1}{PP}\) 倍的峰值激活内存。
    *   虽然需要两份模型参数副本，但由于训练中使用了大的 EP size，内存增加不显著。
    *   相比 **Chimera** ([Li and Hoefler, 2021](https://dl.acm.org/doi/10.1145/3458817.3476145))，对微批次数量的要求更宽松（只需能被 2 整除），且气泡和激活内存不随微批次数量增加而增长。

| 方法 (Method)     | 气泡 (Bubble)                                                    | 参数 (Parameter) | 激活 (Activation) |
| :---------------- | :--------------------------------------------------------------- | :--------------- | :---------------- |
| 1F1B              | \((PP-1)(F+B)\)                                                  | \(1 \times\)     | \(PP\)            |
| ZB1P              | \((PP-1)(F+B-2W)\)                                               | \(1 \times\)     | \(PP\)            |
| **DualPipe (Deepseek V3)** | \(\left(\frac{PP}{2}-1\right)(F\&B+B-3W)\)                        | \(2 \times\)     | \(PP+1\)          |

上表对不同流水线并行方法的流水线气泡和内存使用比较。\(F\)：前向 chunk 执行时间；\(B\)：完整反向 chunk 执行时间；\(W\)：“权重反向” chunk 执行时间；\(F\&B\)：两个相互重叠的前向和反向 chunk 的执行时间。

#### 高效的跨节点 All-to-All 通信实现

*   **目标:** 为 DualPipe 提供足够的计算性能，定制高效的跨节点 All-to-all 通信 Kernel (dispatching & combining)，节省通信专用 SM。
*   **策略:** 结合 MoE 门控算法和集群网络拓扑（节点间 IB 全互联，节点内 NVLink）。
    *   **带宽利用:** NVLink 带宽 (\(160 \mathrm{~GB} / \mathrm{s}\)) 约是 IB (\(50 \mathrm{~GB} / \mathrm{s}\)) 的 3.2 倍。限制每个 token 最多分发到 **4 个节点**以减少 IB 流量。
    *   **传输路径:** Token 确定路由后，先通过 **IB** 传输到目标节点上具有相同节点内索引的 GPU。到达目标节点后，立即通过 **NVLink** 转发给托管目标专家的特定 GPU，避免被后续到达的 token 阻塞。
    *   **效果:** IB 和 NVLink 通信完全重叠。每个 token 可高效选择平均 **3.2 个专家/节点**，无需额外 NVLink 开销。这意味着 V3 实际选择 8 个路由专家，但理论上可扩展到 **13 个专家** (4 nodes × 3.2 experts/node) 而通信成本不变。
*   **实现:**
    *   使用 **Warp Specialization** ([Bauer et al., 2014](https://doi.org/10.1145/2555243.2555258))技术 ，将 **20 个 SM** 划分为 10 个通信通道。
    *   Dispatch 过程：IB 发送、IB-to-NVLink 转发、NVLink 接收由各自的 warp 处理，warp 数量根据负载动态调整。
    *   Combine 过程：NVLink 发送、NVLink-to-IB 转发与累加、IB 接收与累加也由动态调整的 warp 处理。
    *   **优化:** Dispatch 和 Combine Kernel 与计算流重叠。使用定制 **PTX** 指令并自动调整通信 chunk 大小，显著减少 L2 缓存使用和对其他 SM 计算 Kernel 的干扰。
*   **结果:** 仅需 **20 个 SM** 即可充分利用 IB 和 NVLink 带宽。

#### 极致内存优化与最小开销

为减少训练内存占用，采用了以下技术：
*   **重计算:** 在反向传播中重计算所有 **RMSNorm** 操作和 **MLA 的上投影**，避免存储它们的输出激活值。以微小开销显著减少激活内存需求。
*   **CPU 存储 EMA:** 将模型参数的**指数移动平均 (EMA)** 保存在 **CPU 内存**中，并在每个训练步骤后异步更新。无需额外 GPU 内存或时间开销即可维护 EMA 参数。
*   **共享 Embedding 和输出头:** 利用 DualPipe 策略，将模型最浅层（含 Embedding 层）和最深层（含输出头）部署在**同一个 PP rank** 上。这使得 **MTP 模块**和主模型可以**物理共享** Embedding 和输出头的参数及梯度，进一步提升内存效率。
*   **效果:** 这些优化使得 DeepSeek-V3 可以在**不使用昂贵的张量并行 (TP)** 的情况下进行训练。

### FP8 训练

为了加速训练并减少显存占用，DeepSeek-V3 采用了 **FP8 混合精度训练框架** ([Dettmers et al., 2022](https://arxiv.org/pdf/2208.07339); [Noune et al., 2022](https://arxiv.org/abs/2206.02915); [Peng et al., 2023](https://arxiv.org/abs/2310.18313))，并在超大规模模型上首次验证了其有效性。

#### 混合精度框架

*   **核心计算 (GEMM):** 大部分 GEMM 操作（前向、激活梯度反向、权重梯度反向）使用 FP8 输入，输出 BF16 或 FP32，理论上计算速度翻倍。
*   **高精度保留:** 对精度敏感或计算开销小的部分（如 Embedding、输出头、MoE 门控、Normalization、Attention）保留 BF16/FP32 精度。
*   **高精度存储:** 主权重、权重梯度、优化器状态（部分 BF16）使用更高精度存储，通过 ZeRO-1 分片降低显存压力。

{{< figure
    src="fp8_framework.png"
    caption="Fig. 14. The overall mixed precision framework with FP8 data format. For clarification, only the Linear operator is illustrated. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

#### 精度提升策略

1.  **细粒度量化:** 为了解决 FP8 **动态范围有限和对离群值敏感**的问题 ([Fishman et al., 2024](https://arxiv.org/abs/2409.12517); [He et al., 2024](https://arxiv.org/abs/2405.19279); [Sun et al., 2024](https://arxiv.org/abs/2402.17762))，采用更细粒度的量化：
    *   **激活:** 按 \(1 \times 128\) 的 tile 分组缩放。
    *   **权重:** 按 \(128 \times 128\) 的 block 分组缩放。
    这种方法让缩放因子更适应局部数据的范围，减少量化误差。
2.  **提升累加精度:** H800 的 Tensor Core 进行 FP8 GEMM 时累加精度有限（约 14 位）。为解决此问题，采用 **Promotion to CUDA Cores** 策略 ([Thakkar et al., 2023](https://github.com/NVIDIA/cutlass))：Tensor Core 计算部分累加和（例如每 \(N_C=128\) 个元素），然后将结果传输到 CUDA Core 的 FP32 寄存器中进行全精度累加。细粒度量化的缩放因子也可以在 CUDA Core 上高效应用。通过 WGMMA 操作的并发执行，这种方法在提升精度的同时，对计算效率影响较小。
3.  **E4M3 格式:** V3 在所有张量上统一使用 **E4M3 格式**（4 位指数，3 位尾数），而非混合使用 **E5M2** ([NVIDIA, 2024](https://github.com/NVIDIA/TransformerEngine); [Peng et al., 2023](https://arxiv.org/abs/2310.18313); [Sun et al., 2019b](https://papers.nips.cc/paper_files/paper/2019/hash/65fc9fb4897a89789352e211ca2d398f-Abstract.html))。细粒度量化策略有效缓解了 E4M3 动态范围较小的问题。
4.  **在线量化:** 实时计算每个 tile/block 的**最大绝对值来确定缩放因子，而非依赖历史值** ([NVIDIA, 2024](https://github.com/NVIDIA/TransformerEngine); [Peng et al., 2023](https://arxiv.org/abs/2310.18313))，确保量化精度。

{{< figure
    src="fp8_quantization_enhancement.png"
    caption="Fig. 15. (a) Fine-grained quantization method to mitigate quantization errors. (b) Improved FP8 GEMM precision by promoting to CUDA Cores for high-precision accumulation. (Image source: [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

#### 低精度存储与通信

*   **优化器状态:** **AdamW** ([Loshchilov and Hutter, 2017](https://arxiv.org/abs/1711.05101)) 的一阶和二阶矩使用 BF16 存储。主权重和梯度累加仍用 FP32。
*   **激活缓存:** Wgrad 操作使用 FP8 输入，因此激活值可以缓存为 FP8。对特定敏感操作（如 Attention 后 Linear 的输入）使用定制的 E5M6 格式，并进行 round scaling。MoE 中 SwiGLU 的输入也缓存为 FP8。
*   **通信:** MoE up-projection 前的激活量化为 FP8 进行分发（dispatch），MoE down-projection 前的激活梯度也量化为 FP8。Combine 操作保留 BF16 精度。

下图实验证明，FP8 训练的损失与 BF16 相比，相对误差低于**0.25%**，在可接受范围内。

{{< figure
    src="fp8_vs_bf16_loss_curves.png"
    caption="Fig. 16. Loss curves comparison between BF16 and FP8 training. Results are smoothed by Exponential Moving Average (EMA) with a coefficient of 0.9 [DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437))"
    align="center"
    width="100%"
>}}

### 推理与部署

DeepSeek-V3 部署在 H800 集群上（节点内 NVLink，节点间 IB 全互联）。为同时保证在线服务的 **SLO (服务等级目标)** 和高吞吐量，采用分离 **Prefilling (预填充)** 和 **Decoding (解码)** 阶段的部署策略。

#### 预填充阶段

*   **最小部署单元:** 4 节点 32 GPU。
*   **并行策略:**
    *   Attention 部分: **TP4 (张量并行) + SP (序列并行)** 结合 **DP8 (数据并行)**。较小的 TP size (4) 限制了 TP 通信开销。
    *   MoE 部分: **EP32 (专家并行)**，确保每个专家处理足够大的批次以提高计算效率。
    *   浅层 Dense MLP: 使用 TP1 节省 TP 通信。
*   **MoE All-to-All 通信:** 采用与训练类似的方法：先通过 IB 跨节点传输 token，然后在节点内 GPU 间通过 NVLink 转发。
*   **负载均衡:** 采用 **冗余专家** 的部署策略。
    *   根据在线部署收集的统计数据，周期性（例如每 10 分钟）检测**高负载专家**并进行复制部署。
    *   确定冗余专家集后，根据观察到的负载仔细地在节点内 GPU 间**重排专家**，在不增加跨节点 All-to-all 通信开销的前提下尽可能平衡 GPU 负载。
    *   DeepSeek-V3 部署中，Prefilling 阶段设置 **32 个冗余专家**。每个 GPU 除了托管原有的 8 个专家外，还会额外托管 1 个冗余专家。
*   **效率优化:** 为提高吞吐并隐藏 All-to-all 和 TP 通信开销，**同时处理两个计算负载相似的微批次**，将一个微批次的 Attention 和 MoE 与另一个微批次的 Dispatch 和 Combine 重叠。
*   **探索方向:** **动态冗余** 策略，每个 GPU 托管更多专家（如 16 个），但每次推理步骤只激活 9 个。在每层 All-to-all 操作开始前动态计算全局最优路由方案。由于 Prefilling 计算量大，计算路由方案的开销几乎可忽略。

#### 解码阶段

*   **专家视角:** 将**共享专家** 视为一个路由目标。从这个角度看，每个 token 在路由时会选择 **9 个专家**（共享专家被视为一个总会被选中的高负载专家）。
*   **最小部署单元:** 40 节点 320 GPU。
*   **并行策略:**
    *   Attention 部分: **TP4 + SP** 结合 **DP80**。
    *   MoE 部分: **EP320**。每个 GPU 只托管一个专家，其中 64 个 GPU 负责托管冗余专家和共享专家。
*   **All-to-All 通信:** Dispatch 和 Combine 部分通过 **IB 直接点对点传输**以实现低延迟。利用 **IBGDA** ([NVIDIA, 2022](https://developer.nvidia.com/blog/gpudirect-storage/)) 技术进一步最小化延迟和提升通信效率。
*   **负载均衡:** 类似 Prefilling，根据在线服务的统计专家负载，周期性确定冗余专家集。但由于每个 GPU 只托管一个专家，无需重排。
*   **探索方向:**
    *   **动态冗余策略:** 但需要更仔细地优化计算全局最优路由方案的算法，并与 Dispatch Kernel 融合以减少开销。
    *   **同时处理两个微批次:** 与 Prefilling 不同，解码阶段 Attention 耗时占比更大。因此，将一个微批次的 **Attention** 与另一个微批次的 **Dispatch+MoE+Combine** 重叠。解码阶段每个专家的批次大小相对较小（通常小于 256 token），瓶颈是内存访问而非计算。由于 MoE 部分只需加载一个专家的参数，访存开销小，使用较少 SM 不会显著影响整体性能。因此，可以只分配一小部分 SM 给 Dispatch+MoE+Combine，避免影响 Attention 部分的计算速度。

### 对硬件设计的建议

DeepSeek 团队基于 **All-to-all 通信**和 **FP8 训练方案**的实现，向 AI 硬件供应商提出以下芯片设计建议。

#### 通信硬件

*   **现状:** 通过计算/通信重叠隐藏了通信延迟，显著降低了对通信带宽的依赖。但当前通信实现依赖昂贵的 **SM**（例如 H800 上分配了 132 个 SM 中的 20 个用于此目的），限制了计算吞吐量，且 SM 用于通信导致 Tensor Core 完全闲置，效率低下。
*   **SM 主要任务:**
    *   在 IB 和 NVLink 域之间转发数据，同时将发往同一节点内多个 GPU 的 IB 流量从单个 GPU 聚合。
    *   在 RDMA 缓冲区和输入/输出缓冲区之间传输数据。
    *   为 All-to-all Combine 执行 Reduce 操作。
    *   在跨 IB 和 NVLink 域向多个专家分块传输数据时管理细粒度内存布局。
*   **期望:**
    *   未来供应商开发硬件，将这些通信任务从宝贵的计算单元 SM **卸载**，作为 GPU 协处理器或网络协处理器其类似于 NVIDIA SHARP([Graham et al., 2016](https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf))。
    *   为降低应用编程复杂性，期望该硬件能从计算单元的角度**统一 IB (scale-out) 和 NVLink (scale-up) 网络**。通过这个统一接口，计算单元可以通过提交基于简单原语的通信请求，轻松完成在整个 IB-NVLink 统一域内的读、写、多播和 Reduce 等操作。

#### 计算硬件

1.  **Tensor Core 中更高精度的 FP8 GEMM 累加:**
    *   **问题:** 当前 NVIDIA Hopper 架构 Tensor Core 实现中，FP8 GEMM 使用定点累加，通过右移对齐尾数乘积再相加。实验表明，符号填充右移后仅使用每个尾数乘积的最高 14 位，超出范围的位被截断。然而，例如要从 32 个 FP8×FP8 乘积累加得到精确的 FP32 结果，至少需要 34 位精度。
    *   **建议:** 未来芯片设计应**增加 Tensor Core 中的累加精度**以支持全精度累加，或根据训练和推理算法的精度要求选择合适的累加位宽。这种方法能在保持计算效率的同时确保误差在可接受范围内。
2.  **支持 Tile 和 Block 级量化:**
    *   **问题:** 当前 GPU 仅支持 per-tensor 量化，缺乏对 tile-wise 和 block-wise 等细粒度量化的原生支持。当前实现中，达到 \(N_C\) 间隔时，部分结果需从 Tensor Core 复制到 CUDA Core，乘以缩放因子，再加到 CUDA Core 的 FP32 寄存器上。虽然结合精确 FP32 累加策略显著缓解了反量化开销，但 Tensor Core 和 CUDA Core 间的频繁数据移动仍限制计算效率。
    *   **建议:** 未来芯片应通过**让 Tensor Core 能够接收缩放因子并实现带分组缩放的 MMA** 来支持细粒度量化。这样，整个部分和累加与反量化可以在 Tensor Core 内部直接完成，直到产生最终结果，避免频繁数据移动。
3.  **支持在线量化:**
    *   **问题:** 当前实现难以有效支持在线量化，尽管其有效性已在研究中得到证明。现有流程中，需要从 HBM 读取 128 个 BF16 激活值（前一计算的输出）进行量化，量化后的 FP8 值写回 HBM，然后再次读取用于 MMA。
    *   **建议:**
        *   未来芯片将 **FP8 类型转换和 TMA (Tensor Memory Accelerator) 访问融合成单一操作**，使得量化可以在激活从全局内存传输到共享内存的过程中完成，避免频繁的内存读写。
        *   推荐支持 **warp 级 cast 指令**以加速，进一步促进 Layer Normalization 和 FP8 cast 的更好融合。
        *   或者，采用**近存计算**方法，在 HBM 附近放置计算逻辑。这样，BF16 元素可以在从 HBM 读入 GPU 时直接转换为 FP8，减少约 50% 的片外内存访问。
4.  **支持转置 GEMM 操作:**
    *   **问题:** 当前架构将矩阵转置与 GEMM 操作融合起来很麻烦。在工作流中，前向传播的激活被量化为 \(1 \times 128\) 的 FP8 tile 并存储。反向传播时，需要读出矩阵，反量化，转置，重新量化为 \(128 \times 1\) tile，再存入 HBM。
    *   **建议:** 未来芯片应支持在 MMA 操作前**直接从共享内存读取转置后的矩阵**（针对训练和推理所需的那些精度）。结合 FP8 格式转换和 TMA 访问的融合，此增强将显著简化量化工作流。


### 训练成本与效率

*   **DeepSeek-V2:** 相比 DeepSeek 67B (Dense)，节省 42.5% 训练成本，KV 缓存减少 93.3%，最大吞吐量提升 5.76 倍。
*   **DeepSeek-V3:** 训练效率极高，每训练 1T token 仅需 180K H800 GPU 小时。总训练成本（预训练+上下文扩展+后训练）仅 2.788M H800 GPU 小时（约 558 万美元，按 2 美元/小时计）。预训练在 2048 卡 H800 集群上耗时不到 2 个月。

| 训练阶段 | H800 GPU 小时 | 预估成本 (美元) |
| :--- | :---: | :---: |
| 预训练 | 2664 K | \$5.328 M |
| 上下文扩展 | 119 K | \$0.238 M |
| 后训练 | 5 K | \$0.01 M |
| **总计** | **2788 K** | **\$5.576 M** |

## 预训练

### 数据构建

相较于 DeepSeek‑V2（基于 67B 模型，使用 100K 词表 Byte‑level BPE Tokenizer，8.1T tokens），DeepSeek‑V3 在预训练阶段通过以下策略，实现了更大规模和更高质量的数据构建：

1. **语料库扩展与精炼**
   - **专注领域**：显著增加数学与编程相关文本占比，强化模型在技术领域的理解与生成能力。
   - **多语言覆盖**：在英语、中文之外，新增多种语种语料，提升跨语言泛化性能。
   - **去重与多样性**：采用高效的数据去重和过滤流程，既最大限度减少冗余，又保证内容多样性。
   - **规模提升**：最终构建了约 **14.8T** 高质量 tokens，比 V2 增长近 83%。

2. **训练策略与技术创新**
   - **文档打包**
     结合 **Document Packing**([Ding et al., 2024](https://arxiv.org/abs/2404.10830))方法，将连贯文本打包为更长片段，以提升 GPU 利用率和上下文完整性；未采用跨样本注意力掩码，保持实现简洁。
   - **Fill‑in‑Middle（FIM）策略**
     - **动机**：借鉴 DeepSeekCoder‑V2([DeepSeek‑AI, 2024](https://arxiv.org/abs/2406.11931))的方法，旨在提升模型对中间缺失信息的填充能力。
     - **框架**：引入 Prefix‑Suffix‑Middle (PSM) 结构 ，样例如下：
       ```
       <|fim_begin|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>
       ```
     - **应用比例**：文档级预打包前插入 FIM，占比 **10%**，平衡生成与预测任务。

3. **Tokenizer 优化**
   - **BBPE 词表扩容**：采用 Byte‑level BPE，词表由 100K 扩至 **128K**，提升罕见词与专有名词覆盖。
   - **预分词器改进**：针对多语言场景，调整分词规则，提升压缩效率与编码一致性。
   - **边界偏见缓解**：参考 [Lundberg, 2023](https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb) 的方法为减少标点符号+换行组合 token 在 few‑shot 场景下的偏倚，引入随机拆分机制，让模型接触更多边界变体。

### 超参数

| 参数 | DeepSeek-V2  | DeepSeek-V3  |
| :--- | :---: | :---: |
| Transformer 层数 | 60 | 61 |
| 隐藏层维度 \(d\) | 5120 | 7168 |
| 初始化标准差 | 0.006 | 0.006 |
| **MLA 参数** | | |
| 注意力头数 \(n_h\) | 128 | 128 |
| 每头维度 \(d_h\) | 128 | 128 |
| KV 压缩维度 \(d_c\) | 512 (\(4d_h\)) | 512 (\(4d_h\)) |
| 查询压缩维度 \(d_c'\) | 1536 | 1536 |
| 解耦 RoPE 维度 \(d_h^R\) | 64 (\(d_h/2\)) | 64 (\(d_h/2\)) |
| **DeepSeekMoE 参数** | | |
| MoE 层位置 | 除第 1 层外 | 除前 3 层外 |
| 共享专家数 \(N_s\) | 2 | 1 |
| 路由专家数 \(N_r\) | 160 | 256 |
| 专家中间维度 | 1536 | 2048 |
| 激活专家数 \(K_r\) | 6 | 8 |
| 设备/节点限制路由 \(M\) | 3 (设备) | 4 (节点) |
| 负载均衡策略 | 辅助损失 (\(\alpha_1=0.003, \alpha_2=0.05, \alpha_3=0.02\)) + Token Dropping | 无辅助损失 (\(\gamma=0.001\)) + 序列级损失 (\(\alpha=0.0001\)) |
| **MTP 参数 (V3 only)** | | |
| 预测深度 \(D_{MTP}\) | N/A | 1 |
| MTP 损失权重 \(\lambda\) | N/A | 0.3 (前 10T) / 0.1 (后 4.8T) |
| **训练参数** | | |
| 优化器 | AdamW (\(\beta_1=0.9, \beta_2=0.95, wd=0.1\)) | AdamW (\(\beta_1=0.9, \beta_2=0.95, wd=0.1\)) |
| 最大序列长度 | 4K | 4K |
| 训练 Tokens | 8.1T | 14.8T |
| 学习率 | Warmup + Step Decay (Max \(2.4 \times 10^{-4}\)) | Warmup + Cosine Decay + Constant (Max \(2.2 \times 10^{-4}\)) |
| Batch Size | 2304 -> 9216 | 3072 -> 15360 |
| 梯度裁剪 | 1.0 | 1.0 |
| 精度 | BF16 | FP8 混合精度 |

### 长上下文扩展

两者均使用 **YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)) 技术扩展上下文窗口。
*   **DeepSeek-V2:** 从 4K 扩展到 128K。使用 YaRN (scale \(s=40, \alpha=1, \beta=32\))，在 32K 序列长度上训练 1000 步。调整了长度缩放因子 \(\sqrt{t}=0.0707 \ln s+1\)。
*   **DeepSeek-V3:** 分两阶段从 4K 扩展到 32K，再到 128K。每阶段训练 1000 步。YaRN 参数与 V2 相同，长度缩放因子 \(\sqrt{t}=0.1 \ln s+1\)。第一阶段序列长度 32K，第二阶段 128K。

两模型在 NIAH 测试中均表现出良好的长上下文能力。

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

### 评估

**DeepSeek-V2 评估结果:**

DeepSeek-V2 与代表性开源模型对比 (部分结果)。DeepSeek-V2 以 21B 激活参数达到当时顶尖水平。

|  | Benchmark (Metric) | \# Shots | DeepSeek 67B | Qwen1.5 72B | Mixtral 8x22B | LLaMA 3 70B | DeepSeek-V2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | \# Activated Params | - | 67B | 72B | 39B | 70B | **21B** |
| English | MMLU ([Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300)) (Acc.) | 5-shot | 71.3 | 77.2 | 77.6 | 78.9 | **78.5** |
| Code | HumanEval ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) (Pass@1) | 0-shot | 45.1 | 43.9 | 53.1 | 48.2 | **48.8** |
| Math | GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) (EM) | 8-shot | 63.4 | 77.9 | 80.3 | 83.0 | **79.2** |
| Chinese | C-Eval ([Huang et al., 2023](https://arxiv.org/abs/2305.08322)) (Acc.) | 5-shot | 66.1 | **83.7** | 59.6 | 67.5 | 81.7 |

**DeepSeek-V3 评估结果:**

DeepSeek-V3-Base 与代表性开源模型对比 (部分结果)。DeepSeek-V3-Base 在多数基准上成为最强开源模型，尤其在代码和数学方面。

|  | Benchmark (Metric) | \# Shots | DeepSeek-V2 Base | Qwen2.5 72B Base | LLaMA-3.1 405B Base | DeepSeek-V3 Base |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | \# Activated Params | - | 21B | 72B | 405B | **37B** |
| English | MMLU ([Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300)) (EM) | 5-shot | 78.4 | 85.0 | 84.4 | **87.1** |
|  | MMLU-Pro ([Wang et al., 2024](https://arxiv.org/abs/2406.01574)) (em) | 5-shot | 51.4 | 58.3 | 52.8 | **64.4** |
| Code | HumanEval ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) (Pass@1) | 0-shot | 43.3 | 53.0 | 54.9 | **65.2** |
|  | LiveCodeBench-Base ([Jain et al., 2024](https://arxiv.org/abs/2403.07974)) (Pass@1) | 3-shot | 11.6 | 12.9 | 15.5 | **19.4** |
| Math | GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) (Em) | 8-shot | 81.6 | 88.3 | 83.5 | **89.3** |
|  | MATH ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) (EM) | 4-shot | 43.4 | 54.4 | 49.0 | **61.6** |
| Chinese | C-Eval ([Huang et al., 2023](https://arxiv.org/abs/2305.08322)) (EM) | 5-shot | 81.4 | 89.2 | 72.5 | **90.1** |
| Multilingual | MMMLU-non-English ([OpenAI, 2024](https://huggingface.co/datasets/openai/MMMLU)) (em) | 5-shot | 64.0 | 74.8 | 73.8 | **79.4** |

**总结:** DeepSeek-V3-Base 凭借其架构创新、更大规模的训练数据和高效的训练方法，全面超越了 DeepSeek-V2-Base 和其他顶尖开源模型（包括参数量远超其激活参数的 LLaMA-3.1 405B）。

## 对齐

为了使模型更好地理解指令、遵循人类偏好并提升特定能力（如推理），DeepSeek-V2 和 V3 都进行了监督微调 (SFT) 和强化学习 (RL)。

### 监督微调

*   **DeepSeek-V2:** 使用了约 1.5M 条指令数据，包含 1.2M 帮助性数据和 0.3M 安全性数据，注重提升数据质量以减少幻觉、增强写作能力。
*   **DeepSeek-V3:**
    *   **推理数据:** 利用内部的 DeepSeek-R1 模型 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948)) 生成推理过程（数学、代码、逻辑等）。由于 R1 输出可能过长或格式不佳，V3 采用了**知识蒸馏**的思路：
        1.  训练领域专家模型（如代码专家）：结合原始 SFT 数据和 R1 生成的长 CoT 数据（带有引导反思/验证的系统提示）进行 SFT+RL 训练。
        2.  使用专家模型生成 SFT 数据：专家模型在 RL 过程中学会融合 R1 的推理模式和常规 SFT 数据的简洁性。
        3.  拒绝采样：筛选高质量 SFT 数据用于最终 V3 的 SFT。
    *   **非推理数据:** 使用 DeepSeek-V2.5 生成，并由人工标注员验证。
    *   **SFT 设置:** 微调 2 个 epoch，学习率从 \(5 \times 10^{-6}\) 余弦衰减到 \(1 \times 10^{-6}\)。采用样本打包和掩码隔离。

### 强化学习

两者均采用 **组相对策略优化 (Group Relative Policy Optimization, GRPO)** 算法 ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) 进行 RL。GRPO 是一种 Actor-Only 的方法，它通过比较一组（\(G\) 个）候选输出的相对好坏来估计优势 \(A_i\)，从而避免了训练与策略模型同样大小的 Critic 模型，节省了成本。

GRPO 目标函数：
\[
\begin{gathered}
\mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{o l d}}(O \mid q)\right] \\
\frac{1}{G} \sum_{i=1}^{G}\left(\min \left(\frac{\pi_{\theta}\left(o_{i} \mid q\right)}{\pi_{\theta_{o l d}}\left(o_{i} \mid q\right)} A_{i}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{i} \mid q\right)}{\pi_{\theta_{o l d}}\left(o_{i} \mid q\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{i}\right)-\beta \mathbb{D}_{K L}\left(\pi_{\theta}| | \pi_{r e f}\right)\right),
\end{gathered}
\]
其中优势 \(A_i\) 通过组内奖励 \(r_i\) 标准化得到：
\[
A_{i}=\frac{r_{i}-\operatorname{mean}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}{\operatorname{std}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}.
\]
KL 散度惩罚项使用 Schulman 无偏估计器：
\[
\mathbb{D}_{K L}\left(\pi_{\theta}| | \pi_{r e f}\right)=\frac{\pi_{r e f}\left(o_{i} \mid q\right)}{\pi_{\theta}\left(o_{i} \mid q\right)}-\log \frac{\pi_{r e f}\left(o_{i} \mid q\right)}{\pi_{\theta}\left(o_{i} \mid q\right)}-1.
\]

**奖励模型 (RM):**

*   **DeepSeek-V2:** 采用两阶段 RL 策略。
    1.  **推理对齐:** 使用专门训练的 \(RM_{\text{reasoning}}\) 对代码和数学推理任务进行优化。
    2.  **人类偏好对齐:** 使用多奖励框架，结合 \(RM_{\text{helpful}}\)、\(RM_{\text{safety}}\) 和基于规则的 \(RM_{\text{rule}}\)。
*   **DeepSeek-V3:**
    *   **基于规则的 RM:** 对于可验证的任务（如数学答案格式、LeetCode 测试用例），使用规则来提供可靠奖励。
    *   **基于模型的 RM:** 对于自由格式答案或无标准答案的任务（如创意写作），使用从 V3 SFT Checkpoint 初始化的 RM。该 RM 通过学习带有 CoT 的偏好数据来提升可靠性，减少奖励 Hacking 风险。
    *   **自奖励:** V3 探索使用模型自身的判断能力（通过投票增强）作为反馈来源，特别是在通用场景下，结合 **Constitutional AI** ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) 的思想进行优化。

**RL 训练优化 (V2/V3):** 针对大模型 RL 的高资源需求，进行了工程优化，如混合引擎（训练/推理并行策略不同）、使用 **vLLM** ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)) 加速采样、CPU Offloading 调度等。

### 评估

**DeepSeek-V2 Chat 评估:**

DeepSeek-V2 Chat (SFT/RL) 与代表性开源 Chat 模型在开放式生成任务上的对比。V2 Chat (RL) 在 AlpacaEval 2.0 和 AlignBench 上表现突出。

| Model | MT-Bench ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)) | AlpacaEval 2.0 ([Dubois et al., 2024](https://arxiv.org/abs/2404.04475)) (LC Win Rate) | AlignBench ([Liu et al., 2023](https://doi.org/10.48550/arXiv.2311.18743)) (中文) |
| :---: | :---: | :---: | :---: |
| DeepSeek 67B Chat | 8.35 | 16.6 | 6.43 |
| Mistral 8x22B Instruct | 8.66 | 30.9 | - |
| Qwen1.5 72B Chat | 8.61 | 36.6 | 7.19 |
| LLaMA3 70B Instruct | **8.95** | 34.4 | - |
| DeepSeek-V2 Chat (SFT) | 8.62 | 30.0 | 7.74 |
| DeepSeek-V2 Chat (RL) | **8.97** | **38.9** | **7.91** |

**DeepSeek-V3 Chat 评估:**

DeepSeek-V3 Chat 与代表性开源及闭源 Chat 模型对比 (部分结果)。V3 在多数基准上领先开源模型，并在代码、数学、中文及开放式生成任务上与顶尖闭源模型相当。

|  | Benchmark (Metric) | DeepSeek V2.5-0905 | Qwen2.5 72B-Inst. | LLaMA-3.1 405B-Inst. | Claude-3.5- Sonnet-1022 | GPT-4o 0513 | DeepSeek V3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| English | MMLU ([Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300)) (EM) | 80.6 | 85.3 | 88.6 | 88.3 | 87.2 | **88.5** |
|  | MMLU-Pro ([Wang et al., 2024](https://arxiv.org/abs/2406.01574)) (EM) | 66.2 | 71.6 | 73.3 | **78.0** | 72.6 | 75.9 |
|  | GPQA-Diamond ([Rein et al., 2023](https://arxiv.org/abs/2311.12022)) (Pass@1) | 41.3 | 49.0 | 51.1 | **65.0** | 49.9 | 59.1 |
|  | SimpleQA ([OpenAI, 2024c](https://openai.com/index/introducing-simpleqa/)) (Correct) | 10.2 | 9.1 | 17.1 | 28.4 | **38.2** | 24.9 |
| Code | HumanEval-Mul (Pass@1) | 77.4 | 77.3 | 77.2 | 81.7 | 80.5 | **82.6** |
|  | LiveCodeBench ([Jain et al., 2024](https://arxiv.org/abs/2403.07974)) (Pass@1-COT) | 29.2 | 31.1 | 28.4 | 36.3 | 33.4 | **40.5** |
|  | SWE Verified ([OpenAI, 2024d](https://openai.com/index/introducing-swe-bench-verified/)) (Resolved) | 22.6 | 23.8 | 24.5 | **50.8** | 38.8 | 42.0 |
| Math | AIME 2024 ([MAA, 2024](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOooril84-FGuAUnzl8I-zXl8XG7P00X-BAkMG9x9RIzEWcXHlwWm) (Pass@1) | 16.7 | 23.3 | 23.3 | 16.0 | 9.3 | **39.2** |
|  | MATH-500 ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) (ЕМ) | 74.7 | 80.0 | 73.8 | 78.3 | 74.6 | **90.2** |
| Chinese | C-Eval ([Huang et al., 2023](https://arxiv.org/abs/2305.08322)) (EM) | 79.5 | 86.1 | 61.5 | 76.7 | 76.0 | **86.5** |
|  | C-SimpleQA ([He et al., 2024](https://arxiv.org/abs/2411.07140)) (Correct) | 54.1 | 48.4 | 50.4 | 51.3 | 59.3 | **64.8** |
| Open-Ended | Arena-Hard ([Li et al., 2024](https://arxiv.org/abs/2406.11939)) | 76.2 | 81.2 | 69.3 | 85.2 | 80.4 | **85.5** |
|  | AlpacaEval 2.0 ([Dubois et al., 2024](https://arxiv.org/abs/2404.04475)) (LC Win Rate) | 50.5 | 49.1 | 40.5 | 52.0 | 51.1 | **70.0** |

**总结:**
*   DeepSeek-V2 Chat (RL) 在发布时已是顶尖的开源聊天模型，尤其在 AlpacaEval 和中文 AlignBench 上表现优异。
*   DeepSeek-V3 Chat 进一步提升了性能，成为目前最强的开源聊天模型，在代码、数学、中文知识以及 Arena-Hard ([Li et al., 2024](https://arxiv.org/abs/2406.11939))、AlpacaEval 等开放式评估中表现极其亮眼，达到了与 GPT-4o、Claude-3.5-Sonnet 相媲美的水平。
*   V3 的 R1 蒸馏显著提升了推理能力，但也可能增加响应长度，需要在准确性和效率间权衡。
*   V3 的自奖励能力（在 RewardBench ([Lambert et al., 2024](https://arxiv.org/abs/2403.13787)) 上表现优异）为其持续对齐提供了有效途径。

## 讨论

*   **负载均衡策略演进:** 从 V2 的辅助损失到 V3 的无辅助损失+偏置调整，体现了在保证负载均衡的同时，尽量减少对模型性能本身干扰的趋势。批处理级别的均衡相比序列级均衡，更能促进专家特化。
*   **MTP 的有效性:** V3 的实验证明，多 token 预测作为辅助训练目标，确实能提升模型在标准评估任务上的性能，同时为推理加速（推测解码）提供了可能。
*   **R1 蒸馏:** V3 成功地将 DeepSeek-R1 的长链推理能力蒸馏到标准 LLM 中，显著提升了数学和代码能力。这是一个重要的技术方向，但也需要注意控制生成长度。
*   **自奖励:** V3 强大的判断能力（**RewardBench**([Lambert et al., 2024](https://arxiv.org/abs/2403.13787))）结果使其能有效进行自反馈和自对齐，这对于减少对人类标注的依赖、实现模型持续自我提升至关重要。
*   **SFT 数据量:** 虽然在**LIMA**([Zhou et al., 2024](https://arxiv.org/abs/2305.11206)）认为少量高质量 SFT 数据即可达到不错的效果，但对于特定技能（如指令遵循 IFEval），仍需足够数据量的高质量数据才能达到满意效果。
*   **对齐税:** OpenAI 在**InstructGPT** ([Ouyang et al., 2022](https://arxiv.org/pdf/2203.02155))中指出 RL 对齐在提升开放式生成能力的同时，可能牺牲部分标准基准的性能。V2 和 V3 都努力在数据处理和训练策略上缓解此问题，以达到可接受的平衡。

## 结论、局限性与未来方向


### 结论 

DeepSeek-V2 和 DeepSeek-V3 是两款强大、经济且高效的 MoE 语言模型。它们通过 MLA 和 DeepSeekMoE 架构创新，以及 V3 引入的无辅助损失负载均衡、MTP、FP8 训练和 R1 蒸馏等技术，在性能、训练成本和推理效率上取得了突破。DeepSeek-V3 已成为当前最强的开源模型之一，性能可与顶尖闭源模型竞争。

### 局限性

*   **通用 LLM 局限:** 如知识截止、幻觉、事实性错误等。
*   **语言覆盖:** 主要针对中英文，其他语言能力有限 (V2)。V3 扩展了多语言，但仍以中英文为主。
*   **部署门槛 (V3):** 高效推理需要较大的部署单元（多节点），对小型团队可能有挑战。
*   **推理效率:** 虽然 V3 推理效率相比 V2 有提升，但仍有优化空间。

### 未来方向

*   **架构创新:** 持续优化 MoE 架构，探索支持无限上下文、突破 Transformer 限制的新架构。
*   **数据扩展:** 提升训练数据的数量、质量和维度（多模态等）。
*   **深度思考:** 增强模型的推理长度和深度，提升智能水平。
*   **评估方法:** 发展更全面、多维度的评估方法，避免过拟合特定基准。
*   **对齐与安全:** 持续改进对齐技术（如自奖励），确保模型有用、诚实、无害，与人类价值观对齐。


## 参考文献

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

[53] OpenAI. ["Introducing SWE-bench Verified
"](https://openai.com/index/introducing-swe-bench-verified/) OpenAI Blog (2024).

[54] Mathematical Association of America (MAA). ["2024 AIME I Problems."](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I) Art of Problem Solving Wiki (2024).

[55] Li, Tianle, et al. ["From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline."](https://arxiv.org/abs/2406.11939) arXiv preprint arXiv:2406.11939 (2024).

[56] Lambert, Nathan, et al. ["RewardBench: Evaluating Reward Models for Language Modeling."](https://arxiv.org/abs/2403.13787) arXiv preprint arXiv:2403.13787 (2024).

[57] Zhou, Chunting, et al. ["Lima: Less is more for alignment."](https://arxiv.org/abs/2305.11206) Advances in Neural Information Processing Systems 36 (2023): 55006-55021.

[58] Ouyang, Long, et al. ["Training language models to follow instructions with human feedback."](https://arxiv.org/abs/2203.02155) Advances in neural information processing systems 35 (2022): 27730-27744.


## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Apr 2025). DeepSeek-V2 vs V3.
https://syhya.github.io/zh/posts/2025-04-18-deepseek-v2-v3

Or

```bibtex
@article{syhya2025deepseekv2v3,
  title   = "DeepSeek-V2 vs V3",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Apr",
  url     = "https://syhya.github.io/zh/posts/2025-04-18-deepseek-v2-v3"
}
```