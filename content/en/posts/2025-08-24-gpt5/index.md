---
title: "gpt-oss & GPT-5"
date: 2025-08-24T12:00:00+08:00
author: "Yue Shui"
tags: ["gpt-oss", "GPT-5",  "MoE", "Reasoning", "Tool Use", "OpenAI", "LLM Architecture"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

In August 2025, the AI field witnessed a period of intensive releases from OpenAI. Following **GPT-2** ([OpenAI, 2019](https://openai.com/index/better-language-models/)) in 2019, OpenAI has once again contributed to the open-source community with its first open-weight large language model series, **gpt-oss** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-oss/)), available in 120B and 20B sizes. Shortly after, the highly anticipated next-generation flagship model, **GPT-5** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-5/)), was also officially launched. This series of releases not only marks a new high for open-source models in reasoning and agent capabilities but also reveals OpenAI's latest advancements in model architecture, training methodologies, and safety alignment.

## gpt-oss

**gpt-oss** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-oss/)) is OpenAI's first open-weight language model series released since GPT-2, designed to provide the open-source community with powerful reasoning and tool-use capabilities. The series includes two versions, `gpt-oss-120b` and `gpt-oss-20b`, both released under the Apache 2.0 license.

### Architecture Overview

{{< figure
    src="gpt_oss_vs_gpt2.png"
    caption="Fig. 1. A side-by-side comparison between gpt-oss-20b and GPT-2 XL 1.5B. (Image source: [Raschka, 2025](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the))"
    align="center"
    width="100%"
>}}

gpt-oss is built upon the GPT series architecture and incorporates several mainstream technologies from recent years, including [RMSNorm](https://syhya.github.io/posts/2025-02-01-normalization/#rms-normalization), [SwiGLU](https://syhya.github.io/posts/2025-04-06-llama/#ffn_swiglu), [GQA](https://syhya.github.io/posts/2025-01-16-group-query-attention/#grouped-query-attention-gqa), [RoPE](https://syhya.github.io/posts/2025-04-06-llama/#rotary-positional-embeddings-rope), [YaRN](https://syhya.github.io/posts/2025-04-18-deepseek-v2-v3/), and [MoE](https://syhya.github.io/posts/2025-03-01-train-llm/#mixture-of-experts-model).

The table below provides a direct comparison of the differences between the GPT-OSS 20B and GPT-2 XL 1.5B models.

| **Feature** | **GPT-OSS 20B (2025)** | **GPT-2 XL 1.5B (2019)** |
|---|---|---|
| **Release Date** | 2025 | 2019 |
| **Model Size** | **20B parameters** | 1.5B parameters |
| **Active Parameters** | **3.5B** (per inference) | 1.5B (all activated) |
| **Vocabulary Size** | **200k tokens** | 50k tokens |
| **Embedding Dimension** | **2,880** | 1,600 |
| **Number of Transformer Layers** | 24 layers | **48 layers** |
| **Number of Attention Heads** | **64** | 25 |
| **Context Length** | **131k tokens** | 1,024 tokens |
| **Positional Encoding** | **RoPE** (Rotary Positional Embeddings) | Absolute Positional Embeddings |
| **Attention Mechanism** | **Grouped-Query Attention** | Multi-head Attention |
| **Feed-Forward Network** | **SwiGLU activation + MoE** | GELU activation |
| **MoE Architecture** | **32 experts, 4 activated** | None |
| **Normalization Method** | **RMSNorm** (2 locations) | LayerNorm (2 locations) |
| **Dropout** | **None** | Yes |
| **Sliding Window Attention** | **Used in every other layer**<br>(window of 128 tokens) | None |
| **Training Features** | **Includes Supervised Fine-Tuning + Reinforcement Learning** | Pre-training only |
| **Quantization Support** | **MXFP4** (Can run on a single GPU) | No special quantization |
| **License** | Apache 2.0 | MIT |

### Efficient Attention Mechanisms

To maintain high efficiency while supporting a 128k long context, gpt-oss employs several advanced attention mechanisms.

*   **Grouped-Query Attention (GQA)**: gpt-oss has 64 query heads and 8 key/value heads, meaning every 8 query heads share a single K/V pair. This significantly reduces the size of the KV cache and memory bandwidth requirements, thereby substantially increasing inference throughput with almost no loss in model performance.

*   **Sliding Window Attention**: To further reduce computational complexity, gpt-oss draws inspiration from **Longformer** ([Jiang et al., 2020](https://arxiv.org/abs/2004.05150)) and **Mistral** ([Jiang et al., 2023](https://arxiv.org/abs/2310.06825)) by adopting a **sliding window attention**. Its Transformer layers alternate between Dense Attention and Locally Banded Sparse Attention. The latter is **Sliding Window Attention**, which limits the attention scope of each token to a fixed-size local window.

{{< figure
    src="sliding_window_attention.png"
    caption="Fig. 2. Comparison between regular attention (left) and sliding window attention (right). (Image source: [Jiang et al., 2023](https://arxiv.org/abs/2310.06825))"
    align="center"
    width="80%"
>}}

In gpt-oss, this window size is set to 128 tokens. This means that in a local attention layer, a token can only attend to the 128 tokens preceding it, not the entire context. This design reduces the computational complexity of attention from \( O(L^2) \) to \( O(L \cdot W) \), where \( L \) is the sequence length and \( W \) is the window size. By alternating with full attention layers, the model can efficiently process local information while integrating global information through the full attention layers.

*   **Attention Sinks**: The model introduces **Attention Sinks** ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)), which learns a bias \( \mathbf{s}_h \) added to the attention scores. This allows initial tokens to be consistently attended to, helping to stabilize the attention distribution and prevent information loss in long-context scenarios.

\[ \text{Attention}(Q, K, V)_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}} + \mathbf{s}_h\right)V_h \]

{{< figure
    src="StreamingLLM.png"
    caption="Fig. 3. Illustration of StreamingLLM vs existing methods. (Image source: [Xiao et al., 2023](https://arxiv.org/abs/2309.17453))"
    align="center"
    width="100%"
>}}

The figure above compares the performance and efficiency of **StreamingLLM** with three common long-text processing methods. Assume a language model is pre-trained on texts of length $L$ and needs to predict the $T$-th token during inference (where $T \gg L$):

1.  **Dense Attention**: Retains the key-values (KV) of all historical tokens and computes full attention. The time complexity is $O(T^2)$, and the cache size grows continuously. Performance drops significantly when the input length exceeds the pre-training length $L$.
2.  **Window Attention**: Caches only the KV of the most recent $L$ tokens. It is efficient for inference, but performance plummets once the information from early tokens is replaced.
3.  **Sliding Window with Re-computation**: Reconstructs the KV state from the most recent $L$ tokens each time a new token is generated. Although it performs well on long texts, the re-computation involves quadratic attention, leading to a high time complexity of $O(TL^2)$ and slow inference speed.

This method combines attention sinks with recent tokens during computation, not only maintaining inference efficiency but also sustaining a stable attention distribution and low perplexity in long-text scenarios.

### MXFP4 Quantization

{{< figure
    src="mxfp4.png"
    caption="Fig. 4. Faster MXFP4 Backpropagation via Stochastic Rounding and Hadamard Transform. (Image source: [Tseng et al., 2025](https://arxiv.org/abs/2502.20586))"
    align="center"
    width="60%"
>}}

To enable the large model to run on consumer-grade hardware, gpt-oss uses the **MXFP4** ([Tseng et al., 2025](https://arxiv.org/abs/2502.20586)) format to quantize the MoE weights. MXFP4 is a micro-scaling floating-point format that can effectively quantize weights to about 4.25 bits. Since MoE weights account for over 90% of the model's total parameters, this method drastically compresses the model size, allowing the 120B model to fit into a single 80GB GPU and the 20B model to run on devices with 16GB of VRAM.

### Training

*   **Pre-training**: The model is pre-trained on a dataset of several trillion tokens of text, with a focus on STEM, coding, and general knowledge. To enhance safety, the pre-training data reuses the CBRN content filters from GPT-4o.

*   **Post-training (Reasoning & Tool Use)**: After pre-training, the model undergoes post-training using **CoT RL** techniques similar to those for OpenAI's o3. The goal of this stage is to teach the model to:
    1.  **Reason**: Generate detailed Chain-of-Thought (CoT) to solve complex problems.
    2.  **Use Tools**: Learn to call external tools (like web search, code execution) to enhance its capabilities.

To achieve these advanced agent functionalities, OpenAI designed the [Harmony Chat Format](https://github.com/openai/harmony). This format introduces the concept of "channels" (e.g., `analysis` for CoT, `commentary` for tool calls, `final` for the final answer) and establishes a strict instruction hierarchy (System > Developer > User > Assistant > Tool) to ensure the model's behavior is controllable.

Additionally, the model supports **Variable Effort Reasoning**. Users can set `Reasoning: low/medium/high` in the system prompt to trade off between latency and performance. A higher effort level generates a longer CoT, which typically leads to higher accuracy but also increased latency.

{{< figure
    src="reasoning_efforts.png"
    caption="Fig. 5. Accuracy vs. average CoT length for different reasoning levels on AIME and GPQA benchmarks. (Image source: [OpenAI, 2025](https://openai.com/index/gpt-oss-model-card/))"
    align="center"
    width="100%"
>}}

### Evaluation

{{< figure
    src="gpt_oss_eval.png"
    caption="Fig. 6. Main capabilities evaluations for gpt-oss series. (Image source: [OpenAI, 2025](https://openai.com/index/introducing-gpt-oss/))"
    align="center"
    width="100%"
>}}

Benchmark results show that gpt-oss-120b's accuracy surpasses that of OpenAI's o3-mini and approaches o4-mini. Meanwhile, gpt-oss-20b, at only one-sixth the size, also demonstrates competitive performance.

## GPT-5

**GPT-5** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-5/)) is not a single model but a **unified intelligent system**. It is not a monolithic, massive model but a complex system of multiple specialized models and an intelligent routing mechanism working in concert to balance performance, speed, and cost.

### System Architecture

{{< figure
    src="gpt5_system_arch.png"
    caption="Fig. 7. GPT-5 Unified System Architecture. (Image source: [Latent Space, 2025](https://www.latent.space/p/gpt5-router))"
    align="center"
    width="100%"
>}}

The GPT-5 system consists of three core components:

1.  **gpt-5-main**: As the system's default model, it is fast and efficient, handling the vast majority of user queries. It can be considered the successor to GPT-4o.
2.  **gpt-5-thinking**: Used for more complex problems that require deep thought. This model is activated when the user explicitly requests it (e.g., "think hard about this") or when the system determines the task requires it. It can be seen as the successor to OpenAI's o3.
3.  **Real-time Router**: This is a continuously trained decision-making model that quickly determines which model to assign a user request to based on various signals. Its decisions are based on:
    *   **Conversation Type:** Whether it's a casual chat, Q&A, or a task-oriented conversation.
    *   **Complexity:** The difficulty of the question and the depth of reasoning required.
    *   **Tool Needs:** Whether external tools like web search or a code interpreter need to be called.
    *   **Explicit Intent:** Users can guide the router to select the deep reasoning model with explicit instructions (e.g., "think hard about this").

The router continuously learns from real user signals (such as user model-switching behavior, response preference rates, and measured answer correctness) to constantly optimize its decision-making capabilities.

### Safe Completions

The traditional safety training paradigm is **Hard Refusals**, where the model decides whether to answer fully or refuse directly based on a binary classification of user intent (safe or unsafe). This approach is effective for clearly malicious prompts but is very brittle when dealing with ambiguous intent or topics involving **dual-use** (e.g., biology, cybersecurity), often leading to over-refusals.

**Safe Completions** ([Baker et al., 2025](https://openai.com/index/gpt-5-safe-completions/)) moves away from binary classification of user intent and instead focuses on maximizing the model's helpfulness while adhering to safety policies.

*   **For clearly harmful requests**: The model will still refuse.
*   **For dual-use requests** (e.g., in biology, chemistry, cybersecurity): The model provides safe, high-level answers without directly executable details, rather than refusing entirely.
*   **For requests with ambiguous intent**: The model attempts to complete the task in a safe manner or offers safe alternatives.

This approach significantly improves the model's safety and utility in dual-use domains and reduces unnecessary over-refusals.

{{< figure
    src="safe_completions.png"
    caption="Fig. 8. Left: Overall structure of the safe-completion training stack. Right: Details of the safecompletion reward design. (Image source: [OpenAI, 2025](https://openai.com/index/gpt-5-safe-completions/))"
    align="center"
    width="100%"
>}}

### Chain-of-Thought Monitoring

OpenAI employs **Chain-of-Thought Monitoring (CoT Monitoring)** ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)) to ensure the reliability and safety of its reasoning models and to prevent issues like [reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/). Unlike some approaches that try to optimize CoT through SFT, GPT-5's CoT training does not impose direct alignment. This allows the CoT to more genuinely reflect the model's "thinking" process, serving as an effective window for detecting erroneous behavior, deceptive intent, or potential risks.

{{< figure
    src="monitor_frontier_reasoning.png"
    caption="Fig. 9. Monitoring Frontier Reasoning Models for Reward Hacking. (Image source: [Baker et al., 2025](https://arxiv.org/abs/2503.11926))"
    align="center"
    width="100%"
>}}

Through CoT monitoring, OpenAI found that the incidence of deceptive behavior in `gpt-5-thinking` was reduced to 2.1% from 4.8% in o3. This technology is crucial for understanding and mitigating the risks of advanced AI systems.

### Evaluation

GPT-5 excels across multiple benchmarks, setting new standards, especially in reasoning, coding, and multimodal capabilities. Compared to its predecessors, it not only achieves a leap in accuracy but also makes significant strides in efficiency, often achieving or surpassing the performance of o3 with 50-80% less output.

{{< figure
    src="gpt5_swe_bench.png"
    caption="Fig. 10. GPT-5 performance in SWE-bench Verified Software Engineering. (Image source: [OpenAI, 2025](https://openai.com/index/introducing-gpt-5/))"
    align="center"
    width="100%"
>}}

#### 🧠 Intelligence

| Benchmark                            | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano |
| ------------------------------------ | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ------- | ------------ | ------------ |
| AIME ’25             | **94.6%**  | 91.1%             | 85.2%             | 88.9%            | 92.7%                 | 46.4%   | 40.2%        | -            |
| FrontierMath (with python tool only) | **26.3%**  | 22.1%             | 9.6%              | 15.8%            | 15.4%                 | -       | -            | -            |
| GPQA diamond              | **85.7%**  | 82.3%             | 71.2%             | 83.3%            | 81.4%                 | 66.3%   | 65.0%        | 50.3%        |
| HLE                    | **24.8%**  | 16.7%             | 8.7%              | 20.2%            | 14.7%                 | 5.4%    | 3.7%         | -            |
| HMMT 2025               | **93.3%**  | 87.8%             | 75.6%             | 81.7%            | 85.0%                 | 28.9%   | 35.0%        | -            |

#### 🖼️ Multimodal

| Benchmark                                      | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano |
| ---------------------------------------------- | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ------- | ------------ | ------------ |
| MMMU                                           | **84.2%**  | 81.6%             | 75.6%             | 82.9%            | 81.6%                 | 74.8%   | 72.7%        | 55.4%        |
| MMMU-Pro (avg across standard and vision sets) | **78.4%**  | 74.1%             | 62.6%             | 76.4%            | 73.4%                 | 60.3%   | 58.9%        | 33.0%        |
| CharXiv reasoning (python enabled)             | **81.1%**  | 75.5%             | 62.7%             | 78.6%            | 72.0%                 | 56.7%   | 56.8%        | 40.5%        |
| VideoMMMU (max frame 256)                      | **84.6%**  | 82.5%             | 66.8%             | 83.3%            | 79.4%                 | 60.9%   | 55.1%        | 30.2%        |
| ERQA                                           | **65.7%**  | 62.9%             | 50.1%             | 64.0%            | 56.5%                 | 44.3%   | 42.3%        | 26.5%        |

#### 💻 Coding

| Benchmark                                         | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano |
| ------------------------------------------------- | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ------- | ------------ | ------------ |
| SWE-Lancer: IC SWE Diamond Freelance Coding Tasks | **\$112K** | \$75K             | \$49K             | \$86K            | \$66K                 | \$34K   | \$31K        | \$9K         |
| SWE-bench Verified                          | **74.9%**  | 71.0%             | 54.7%             | 69.1%            | 68.1%                 | 54.6%   | 23.6%        | -            |
| Aider polyglot (diff)                             | **88.0%**  | 71.6%             | 48.4%             | 79.6%            | 58.2%                 | 52.9%   | 31.6%        | 6.2%         |

#### 📋 Instruction Following

| Benchmark                                      | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano |
| ---------------------------------------------- | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ------- | ------------ | ------------ |
| Scale multichallenge (o3-mini grader)     | **69.6%**  | 62.3%             | 54.9%             | 60.4%            | 57.5%                 | 46.2%   | 42.2%        | 31.1%        |
| Internal API instruction following eval (hard) | 64.0%        | **65.8%**       | 56.1%             | 47.4%            | 44.7%                 | 49.1%   | 45.1%        | 31.6%        |
| COLLIE                                         | **99.0%**  | 98.5%             | 96.9%             | 98.4%            | 96.1%                 | 65.8%   | 54.6%        | 42.5%        |

#### 🔧 Function Calling

| Benchmark          | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano |
| ------------------ | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ------- | ------------ | ------------ |
| Tau²-bench airline | 62.6%        | 60.0%             | 41.0%             | **64.8%**      | 60.2%                 | 56.0%   | 51.0%        | 14.0%        |
| Tau²-bench retail  | **81.1%**  | 78.3%             | 62.3%             | 80.2%            | 70.5%                 | 74.0%   | 66.0%        | 21.5%        |
| Tau²-bench telecom | **96.7%**  | 74.1%             | 35.5%             | 58.2%            | 40.5%                 | 34.0%   | 44.0%        | 12.1%        |

#### 📚 Long Context

| Benchmark                               | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano |
| --------------------------------------- | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ------- | ------------ | ------------ |
| OpenAI-MRCR: 2 needle 128k              | **95.2%**  | 84.3%             | 43.2%             | 55.0%            | 56.4%                 | 57.2%   | 47.2%        | 36.6%        |
| OpenAI-MRCR: 2 needle 256k              | **86.8%**  | 58.8%             | 34.9%             | -                | -                     | 56.2%   | 45.5%        | 22.6%        |
| Graphwalks bfs <128k                    | **78.3%**  | 73.4%             | 64.0%             | 77.3%            | 62.3%                 | 61.7%   | 61.7%        | 25.0%        |
| Graphwalks parents <128k                | **73.3%**  | 64.3%             | 43.8%             | 72.9%            | 51.1%                 | 58.0%   | 60.5%        | 9.4%         |
| BrowseComp Long Context 128k            | **90.0%**  | 89.4%             | 80.4%             | 88.3%            | 80.0%                 | 85.9%   | 89.0%        | 89.4%        |
| BrowseComp Long Context 256k            | **88.8%**  | 86.0%             | 68.4%             | -                | -                     | 75.5%   | 81.6%        | 19.1%        |
| VideoMME (long, with subtitle category) | **86.7%**  | 78.5%             | 65.7%             | 84.9%            | 79.5%                 | 78.7%   | 68.4%        | 55.2%        |

#### 🚨 Hallucinations

| Benchmark                                       | GPT-5 (high) | GPT-5 mini (high) | GPT-5 nano (high) | OpenAI o3 (high) | OpenAI o4-mini (high) | GPT-4.1    | GPT-4.1 mini |
| ----------------------------------------------- | ------------ | ----------------- | ----------------- | ---------------- | --------------------- | ---------- | ------------ | 
| LongFact-Concepts hallucination rate (no tools) | 1.0%         | **0.7%**        | 1.0%              | 5.2%             | 3.0%                  | **0.7%** | 1.1%         |
| LongFact-Objects hallucination rate (no tools)  | 1.2%         | 1.3%              | 2.8%              | 6.8%             | 8.9%                  | **1.1%** | 1.8%         |
| FActScore hallucination rate (no tools)         | **2.8%**   | 3.5%              | 7.3%              | 23.5%            | 38.7%                 | 6.7%       | 10.9%        |

These results indicate that GPT-5 has made significant improvements in complex tasks requiring deep reasoning (like GPQA, AIME) and agentic tasks that require interaction with external environments (like SWE-bench, τ²-bench). At the same time, the substantial improvement in factual accuracy (hallucination rate reduced by nearly 8x) makes it more reliable for practical applications.

## References

[1] Raschka, S. (2025). ["From GPT-2 to gpt-oss: Analyzing the Architectural Advances."](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) Ahead of AI.

[2] Radford, Alec, et al. ["Language models are unsupervised multitask learners."](https://openai.com/index/better-language-models/) OpenAI blog 1.8 (2019): 9.

[3] OpenAI. (2025). ["Introducing gpt-oss."](https://openai.com/index/introducing-gpt-oss/) OpenAI Blog.

[4] OpenAI. (2025). ["Introducing GPT-5."](https://openai.com/index/introducing-gpt-5/) OpenAI Blog.

[5] OpenAI. (2025). ["gpt-oss-120b & gpt-oss-20b Model Card."](https://openai.com/index/gpt-oss-model-card/)

[6] Beltagy, Iz, Matthew E. Peters, and Arman Cohan. ["Longformer: The long-document transformer."](https://arxiv.org/abs/2004.05150) arXiv preprint arXiv:2004.05150 (2020).

[7] Jiang, Dongsheng, et al. ["Mistral 7B."](https://arxiv.org/abs/2310.06825) arXiv preprint arXiv:2310.08825 (2023).

[8] Xiao, G., et al. (2023). ["Efficient Streaming Language Models with Attention Sinks."](https://arxiv.org/abs/2309.17453) arXiv preprint arXiv:2309.17453.

[9] Tseng, Albert, Tao Yu, and Youngsuk Park. ["Training llms with mxfp4."](https://arxiv.org/abs/2502.20586) arXiv preprint arXiv:2502.20586 (2025).

[10] Yuan, Yuan, et al. ["From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training."](https://www.arxiv.org/abs/2508.09224) arXiv preprint arXiv:2508.09224 (2025).

[11] B. Baker, J. Huizinga, L. Gao, Z. Dou, M. Y. Guan, A. Madry, W. Zaremba, J. Pachocki, and D. Farhi, ["Monitoring reasoning models for misbehavior and the risks of promoting obfuscation."](https://arxiv.org/abs/2503.11926) arXiv preprint arXiv:2503.11926, 2025. Submitted on 14 March 2025.

[12] OpenAI. (2025). ["GPT-5 System Card."](https://openai.com/index/gpt-5-system-card/)

[13] OpenAI. (2025). ["Introducing GPT-5 for developers."](https://openai.com/index/introducing-gpt-5-for-developers/) OpenAI Blog.

## Citation

> **Citation**: When reproducing or citing the content of this article, please credit the original author and source.

**Cited as:**

> Yue Shui. (Aug 2025). gpt-oss & GPT-5.
> https://syhya.github.io/posts/2025-08-24-gpt5

Or

```bibtex
@article{yue_shui_gpt5_2025
  title   = "gpt-oss & GPT-5",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Aug",
  url     = "https://syhya.github.io/posts/2025-08-24-gpt5"
}
```