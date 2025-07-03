---
title: "Large Language Model Inference"
date: 2025-06-29T12:00:00+08:00
author: "Yue Shui"
tags: ["LLM", "Inference", "Quantization", "Pruning", "Knowledge Distillation", "KV Cache", "Attention", "Speculative Decoding", "FlashAttention", "vLLM", "Transformer", "Sparsity", "Mixture of Experts"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

In recent years, Large Language Models (LLMs) have achieved revolutionary breakthroughs in fields such as natural language processing, code generation, and even multimodal interaction. However, the powerful capabilities of these models come at the cost of enormous computational and memory overhead, especially during the inference stage. Efficiently deploying and running these models, which have billions or even trillions of parameters, has become a core challenge in scaling LLM technology for real-world applications.

The challenges of LLM inference primarily stem from two aspects:
1.  **Huge Memory Footprint**: In addition to the model parameters themselves, the inference process requires storing a large amount of intermediate state, especially the **KV Cache**. For example, for a request with a batch size of 512 and a sequence length of 2048, its KV cache can be as large as 3TB, several times the size of the model itself. Furthermore, the computational complexity of self-attention grows quadratically with the sequence length.
2.  **Low parallelizability**: The text generation of LLMs is inherently an **autoregressive** process, meaning tokens are generated one by one, with the generation of the next token depending on all previously generated tokens. This serial nature makes the decoding process difficult to parallelize efficiently.

## Principles of Token Generation

To better understand the optimization techniques that follow, we first need to understand how large models generate text and identify the key bottlenecks in their inference process.

### Autoregressive Generation

Mainstream large language models like GPT use a Decoder-Only Transformer architecture and generate text in an autoregressive manner. The basic idea is that the probability of a text sequence can be decomposed into a product of a series of conditional probabilities. Given an initial context word sequence $W_0$ (usually the user's input prompt), the model predicts the next word (token) one at a time, adding the newly generated word to the context as input for the next prediction step. This process can be represented by the following formula:

$$
P(w_{1:T} | W_0) = \prod_{t=1}^{T} P(w_t | w_{1:t-1}, W_0), \text{ with } w_{1:0} = \emptyset
$$

where $w_t$ is the word generated at time step $t$, and $w_{1:t-1}$ is the sequence of all words generated before time step $t$. The generation process continues until the model produces a special end-of-sequence (EOS) token or reaches a predefined maximum length $T$.

### Prefilling and Decoding

The autoregressive nature of generation dictates that LLM inference can be clearly divided into two stages: the **Prefilling stage** and the **Decoding stage**.

{{< figure
    src="prefilling_decoding.png"
    caption="Fig. 1. The Prefilling and Decoding Stages of LLM Inference. (Image source: [Zhou et al., 2024](https://arxiv.org/abs/2404.14294))"
    align="center"
    width="100%"
>}}

1.  **Prefilling Stage**: In this stage, the model processes the entire input prompt in parallel (e.g., "I, like, natural, language" in Figure 1) and computes the probability distribution for the first output token ("Processing"). This stage is characterized by **high parallelism**, as all input tokens can be fed into the Transformer model at once. This allows compute-intensive operations (like matrix multiplication) to fully utilize the parallel processing power of GPUs, making it **compute-bound**.

2.  **Decoding Stage**: In this stage, the model generates subsequent tokens one by one. Each time a token is generated, it is appended to the end of the existing sequence and used as input for the next prediction. This process is **serial** because the generation of the next token depends on the previous one. Consequently, this stage is **memory-bound**, with the main bottleneck being the loading of the massive model weights from GPU memory, rather than the computation itself.

{{< figure
    src="inference_memory.png"
    caption="Fig. 2. Illustration of the memory variation through time (latency) during one generation process. Note that the author ignores the activation size in this figure for simplification. (Image source: [Zhou et al., 2024](https://arxiv.org/abs/2404.14294))"
    align="center"
    width="100%"
>}}

To accelerate the decoding process, modern LLM inference frameworks widely adopt the **KV Cache** technique. In the Transformer's self-attention mechanism, each token needs to interact with all preceding tokens. To avoid recomputing the Key (K) and Value (V) vectors for all previous tokens when generating each new token, the system caches these computed K and V values. This cache is the KV Cache.

As shown in Figure 2, the size of the KV Cache grows linearly as the generated sequence lengthens. For a model with billions of parameters and long sequences, the KV Cache can occupy several gigabytes or even tens of gigabytes of VRAM. This makes VRAM the scarcest resource in LLM inference, severely limiting the number of requests the system can handle simultaneously (i.e., the batch size), which directly impacts inference throughput. Therefore, **how to efficiently manage and optimize the KV Cache is one of the core problems in LLM inference optimization**.

### Decoding Strategies

At each decoding step, the model outputs a probability distribution over the entire vocabulary. The method used to select the next token from this distribution is determined by the decoding strategy (or token generation strategy). Different strategies significantly affect the quality, creativity, and coherence of the generated text.

#### Greedy Search

Greedy search is the simplest decoding strategy. At each time step $t$, it always selects the token with the highest probability as the output:

$$
w_t = \underset{w}{\operatorname{argmax}} P(w | w_{1:t-1})
$$

This method greatly reduces computational complexity and produces results quickly, but it has clear limitations. Because it only makes locally optimal choices at each step, greedy search can easily get stuck in local optima, overlooking globally better possibilities. This often results in generated text that is dull, repetitive, and lacks diversity and creativity.

{{< figure
    src="greedy_search.svg"
    caption="Fig. 3. At each time step, greedy search selects the token with the highest conditional probability. (Image source: [d2l-en, 2019](https://d2l.ai/chapter_recurrent-modern/beam-search.html#id1))"
    align="center"
    width="50%"
>}}

**Code Implementation**:
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

#### Beam Search

To overcome the local optima problem of greedy search, beam search maintains $k$ (`num_beams` or beam width) most likely candidate sequences (called "beams") at each decoding step. In the next step, it expands based on these $k$ sequences and again selects the $k$ new sequences with the highest overall probability. Finally, the algorithm selects the candidate sequence with the highest overall probability from all completed sequences as the final output.

{{< figure
    src="beam_search.svg"
    caption="Fig. 4. The process of beam search (beam size $=2$; maximum length of an output sequence $=3$ ). The candidate output sequences are $A, C, A B, C E, A B D$, and $C E D$. (Image source: [d2l-en, 2019](https://d2l.ai/chapter_recurrent-modern/beam-search.html#id1))"
    align="center"
    width="100%"
>}}

This approach expands the search space, effectively reducing the impact of local optima and typically generating higher-quality, more coherent text. However, the essence of beam search is still to choose the path with the highest overall probability, which makes it prone to producing high-frequency, common expressions in open-ended generation tasks, potentially lacking creativity and diverse output.

#### Temperature Sampling

{{< figure
    src="temperature.png"
    caption="Fig. 5. Illustration of Temperature Sampling. (Image source: [Big Hummingbird Blogs, 2024](https://www.bighummingbird.com/blogs/llm-hyperparameter))"
    align="center"
    width="80%"
>}}

Unlike deterministic search methods, sampling methods introduce randomness, making the generated text more diverse and creative. The most basic sampling method is to sample directly from the model's probability distribution. **Temperature sampling** adjusts the shape of the original probability distribution using a temperature coefficient $T$, applied to the Softmax function. The temperature coefficient controls the flatness of the token probability distribution output by the LLM. A higher temperature makes the distribution flatter and the output more random, while a lower temperature makes the distribution more extreme and the output more stable.

$$
P_T(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

where $z_i$ is the logit output by the model for word $w_i$.
*   When $T \to 1$, the probability distribution remains unchanged.
*   When $T < 1$ (cooling), the distribution becomes "sharper," making high-probability words more likely to be selected, and the generated result is closer to greedy search.
*   When $T > 1$ (heating), the distribution becomes "flatter," giving low-probability words a chance to be selected, resulting in more diverse and random output.

#### Top-K Sampling

**Top-K sampling** ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)) retains only the $K$ most probable candidate words before sampling, then renormalizes and samples from this set of $K$ words. This effectively prevents the model from sampling from extremely low-probability words, avoiding the generation of incoherent text. However, its drawback is that the value of $K$ is fixed and cannot adapt dynamically to different probability distributions.

{{< figure
    src="top_k.png"
    caption="Fig. 6. Illustration of Top-K Sampling. (Image source: [Big Hummingbird Blogs, 2024](https://www.bighummingbird.com/blogs/llm-hyperparameter))"
    align="center"
    width="80%"
>}}

#### Top-p (Nucleus) Sampling

**Top-p sampling** ([Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)) uses a dynamic method to select the set of candidate words. It starts with the most probable word and accumulates their probabilities until the sum exceeds a preset threshold $p$ (e.g., 0.9). The model then samples only from this dynamically generated, minimal set of candidate words $V_{\text{top-p}}$. This method balances text coherence and creativity and is currently one of the most commonly used and effective strategies for open-ended text generation.

{{< figure
    src="top_p.png"
    caption="Fig. 7. Illustration of Top-p Sampling. (Image source: [Big Hummingbird Blogs, 2024](https://www.bighummingbird.com/blogs/llm-hyperparameter))"
    align="center"
    width="80%"
>}}

**Combined Sampling Code Implementation (Top-K, Top-p, Temperature)**:
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

#### Speculative Decoding

**Speculative Decoding** ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) is an innovative acceleration technique that aims to achieve the generation quality of a large model at the speed of a small model, thereby reducing latency without sacrificing quality.

It uses a small, fast Draft Model to generate multiple (e.g., $k$) candidate tokens at once. Then, the large Target Model performs a single forward pass to validate these $k$ tokens in parallel. If the tokens predicted by the draft model match those of the target model, they are accepted, effectively generating multiple tokens with a single forward pass. If they don't match, the subsequent predictions from the draft model are discarded, and the target model's prediction is used for correction.

{{< figure
    src="online_speculative_decoding.png"
    caption="Fig. 8. Overview of online speculative decoding (OSD) framework: For each prompt, the draft model suggests multiple tokens and the target model performs the verification. (Image source: [Liu et al., 2024](https://arxiv.org/abs/2310.07177))"
    align="center"
    width="100%"
>}}

As long as there is some consistency between the predictions of the draft and target models, speculative decoding can significantly reduce generation latency. Variations include **Self-speculative decoding**, which uses the early layers of the model itself as the draft model.

#### Heuristic Strategies

*   **Best-of-N / Majority Vote**: These methods improve the quality and robustness of the final result by generating multiple candidate answers.
    *   **Best-of-N**: The LLM generates N answers, which are then scored by an independent evaluation model (Verifier) or a Reward Model. The answer with the highest score (Best) is selected as the final output.
    *   **Majority Vote / Self-Consistency**: The LLM generates multiple different reasoning paths (Chain-of-Thought) and answers for the same question. The most consistent answer is then selected through a majority vote. This method is particularly effective for tasks requiring complex reasoning.

## Overview of Optimization Methods

Now that we understand the basic principles of inference, let's delve into how to optimize this process. The main goals of inference optimization are to **reduce latency**, **increase throughput**, and **decrease memory footprint**. Existing techniques can be broadly categorized into three areas: model compression, memory and computation optimization, and efficient model architectures.

Typically, the goals of model inference optimization include:

*   **Reducing the model's memory footprint** by using fewer GPU devices and less VRAM.
*   **Reducing computational complexity** by decreasing the number of floating-point operations (FLOPs) required.
*   **Reducing inference latency** to make the model run faster.

To reduce the cost of inference in terms of memory and time, several methods can be employed:

1.  **Applying various parallelization techniques** to scale the model across a large number of GPUs. By intelligently parallelizing model components and data, it's possible to run models with trillions of parameters.
2.  **Memory Offloading**, which moves temporarily unused data to the CPU and reads it back when needed. This helps reduce memory usage but increases latency.
3.  **Smart Batching Strategies**; for example, EffectiveTransformer packs consecutive sequences together to eliminate padding in batches.
4.  **Network compression techniques**, such as pruning, quantization, and distillation. Models with fewer parameters or lower bit-width naturally require less memory and run faster.
5.  **Improvements specific to model architectures**. Many architectural changes, especially to the attention layer, help speed up Transformer decoding.

You can refer to a previous post on [Large Model Training](https://syhya.github.io/posts/2025-03-01-train-llm/) for different types of training parallelization and memory-saving designs, including CPU memory offloading. This article will focus on network compression techniques and architectural improvements for Transformer models.

## Knowledge Distillation

Knowledge Distillation (KD) ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) is a direct method for building a smaller model to accelerate inference by transferring the knowledge from a pre-trained, expensive model (the "teacher model") to a smaller, cheaper model (the "student model"). There are few restrictions on the student model's architecture, other than requiring its output space to match the teacher's to construct a suitable learning objective.

{{< figure
    src="knowledge_distillation.png"
    caption="Fig. 9. The generic framework of teacher-student knowledge distillation training. (Image source: [Gou et al., 2020](https://arxiv.org/abs/2006.05525))"
    align="center"
    width="90%"
>}}

Given a dataset, the student model learns to mimic the teacher's output through a distillation loss function. Neural networks typically have a softmax layer; for example, an LLM outputs a probability distribution over tokens. Let $\mathbf{z}_t$ and $\mathbf{z}_s$ be the pre-softmax logits of the teacher and student models, respectively. The distillation loss minimizes the difference between the two softmax outputs, both with a high temperature $T$. When ground truth labels $\mathbf{y}$ are available, we can combine this with a supervised learning objective (e.g., cross-entropy) that operates on the ground truth labels and the student's soft logits.

$$
\mathcal{L}_{\mathrm{KD}}=\mathcal{L}_{\text {distill }}\left(\operatorname{softmax}\left(\mathbf{z}_t, T\right), \operatorname{softmax}\left(\mathbf{z}_s, T\right)\right)+\lambda \mathcal{L}_{\mathrm{CE}}\left(\mathbf{y}, \mathbf{z}_s\right)
$$

where $\lambda$ is a hyperparameter that balances learning from soft and hard targets. A common choice for $\mathcal{L}_{\text {distill}}$ is KL-divergence or cross-entropy.

An early success story is **DistilBERT** ([Sanh et al. 2019](https://arxiv.org/abs/1910.01108)), which reduced BERT's parameters by 40% while retaining 97% of its performance on downstream fine-tuning tasks and running 71% faster. DistilBERT's pre-training loss is a combination of a soft distillation loss, a supervised training loss (masked language modeling loss $\mathcal{L}_{\text{MLM}}$ in BERT), and a special cosine embedding loss to align the hidden state vectors of the teacher and student models.

{{< figure
    src="DistilBERT.png"
    caption="Fig. 10. The performance of DistilBERT (Image source: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108))"
    align="center"
    width="90%"
>}}

Distillation can be easily combined with **quantization**, **pruning**, or **sparsification** techniques, where the teacher model is the original full-precision, dense model, and the student model is quantized, pruned, or sparsified to achieve higher sparsity.

## Quantization

To further improve model performance during inference, we can go beyond low-precision floating-point numbers and use **quantization**. Quantization converts the model's floating-point weights into low-bit integer representations, such as 8-bit integers (INT8) or even 4-bit integers (INT4).

There are generally two ways to apply quantization to deep neural networks:

1.  **Post-Training Quantization (PTQ)**: The model is first trained to convergence, and then its weights are converted to a lower precision without further training. This method is typically low-cost to implement compared to training.
2.  **Quantization-Aware Training (QAT)**: Quantization is applied during pre-training or further fine-tuning. QAT can achieve better performance but requires additional computational resources and access to representative training data.

### Precision Comparison

In the field of deep learning, numerical precision determines the delicate balance between computational speed and model performance. Understanding the pros and cons of different floating-point and integer formats is key to optimizing the performance of large-scale models. Floating-point numbers are represented in a computer with three parts:

*   **Sign**: Indicates whether the number is positive or negative.
*   **Exponent**: Determines the dynamic range of the number.
*   **Mantissa (or Significand)**: Determines the precision of the number. For convenience, we often refer to the mantissa as the fraction.

{{< figure
    src="combined_float_diagrams.png"
    caption="Fig. 11. fp32 vs fp16 vs bf16 (Image source: [Raschka, 2023](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html))"
    align="center"
    width="70%"
>}}

| Type                                                                                   | Total Bits | Sign Bits | Exponent Bits | Mantissa Bits | Characteristics                                                                                             |
| -------------------------------------------------------------------------------------- | ---------- | --------- | ------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| [**FP64 (Double-precision)**](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) | 64         | 1         | 11            | 52            | Extremely high precision, widely used in scientific computing, but computationally expensive and memory-intensive, rarely used in deep learning. |
| [**FP32 (Single-precision)**](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) | 32         | 1         | 8             | 23            | Standard format for deep learning training, moderate speed, larger memory footprint.                        |
| [**FP16 (Half-precision)**](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)     | 16         | 1         | 5             | 10            | Faster computation, half the memory footprint of FP32, but limited dynamic range, prone to numerical overflow. |
| [**BF16 (Brain Floating Point)**](https://cloud.google.com/tpu/docs/bfloat16)           | 16         | 1         | 8             | 7             | Same dynamic range as FP32, avoids overflow, better suited for LLMs, slightly lower precision than FP16.    |

While pure FP16 precision is fast and memory-efficient, its limited dynamic range makes it highly susceptible to numerical overflow and underflow, which can make training unstable or even prevent convergence. Therefore, using [Mixed-Precision Training](https://syhya.github.io/posts/2025-03-01-train-llm/#mixed-precision-training) is crucial.

Quantization maps floating-point numbers to integers, further reducing computational complexity and memory footprint. Specifically:

*   **INT8**: Occupies only 1/4 of the memory of FP32, significantly accelerating inference speed, but may slightly reduce model accuracy.
*   **INT4**: A more extreme compression scheme, better suited for devices with extremely limited resources or inference scenarios requiring very high throughput.

### Challenges in Transformer Quantization

Many studies on Transformer model quantization share a common finding: simple low-precision (e.g., 8-bit) post-training quantization leads to a significant performance drop. This is mainly due to the **high dynamic range of activation values**, and a simple activation quantization strategy cannot maintain the model's performance.

{{< figure
    src="glue_benchmark.png"
    caption="Fig. 12. Only quantizing model weights to 8-bit while keeping activation at full precision (`W8A32`) achieves much better results when activations are quantized to 8-bit irrespective of whether weights are in lower precision (`W8A8` and `W32A8`). (Image source: [Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948))"
    align="center"
>}}

[Bondarenko et al. (2021)](https://arxiv.org/abs/2109.12948) found in experiments with small BERT models that the input and output of the FFN (feed-forward network) have very different dynamic ranges due to strong **outliers** in the output tensor. Therefore, per-tensor quantization of the FFN's residual sum can lead to significant errors.

As model sizes grow to billions of parameters, **large-magnitude outlier features** begin to appear in all Transformer layers, causing simple low-bit quantization to fail. Researchers observed this phenomenon in the **OPT** ([Zhang et al. 2022](https://arxiv.org/abs/2205.01068)) model, which is larger than 6.7B parameters. Larger models have more layers with extreme outliers, and these outlier features have a significant impact on model performance. In a few dimensions, the magnitude of activation outliers can be about 100 times larger than most other values.

{{< figure
    src="int8_outliner.png"
    caption="Fig. 13. The mean zero-shot accuracy over a set of language tasks (WinoGrande, HellaSwag, PIQA, LAMBADA) of OPT models of increasing sizes. (Image source: [Dettmers et al. 2022](https://arxiv.org/abs/2208.07339))"
    align="center"
    width="80%"
>}}

### Post-Training Quantization (PTQ)

#### Mixed-precision quantization

The most direct way to solve the aforementioned quantization challenges is to implement quantization with different precisions for weights and activations.

**GOBO** ([Zadeh et al. 2020](https://arxiv.org/abs/2005.03842)) was one of the first models to apply post-training quantization to BERT. It assumes that the model weights of each layer follow a Gaussian distribution and thus detects outliers by tracking the mean and standard deviation of each layer. Outlier features are kept in their original form, while other values are divided into multiple bins, storing only the corresponding bin index and centroid value.

{{< figure
    src="gobo.png"
    caption="Fig. 14. The pseudocode for the GOBO algorithm. (Image source: [Zadeh et al. 2020](https://arxiv.org/abs/2005.03842))"
    align="center"
    width="80%"
>}}

Based on the observation that only certain activation layers in BERT (e.g., the residual connection after the FFN) cause large performance drops, [Bondarenko et al. (2021)](https://arxiv.org/abs/2109.12948) adopted mixed-precision quantization, using 16-bit quantization for problematic activations and 8-bit for others.

**LLM.int8()** ([Dettmers et al. 2022](https://arxiv.org/abs/2208.07339)) achieves mixed-precision quantization through two mixed-precision decompositions:

1.  Since matrix multiplication consists of a series of independent inner products between row and column vectors, we can apply independent quantization to each inner product: each row and column is scaled by its absolute maximum value and then quantized to INT8.
2.  Outlier activation features (e.g., 20 times larger than other dimensions) are kept in FP16 format, but they only account for a small fraction of the total weights. How to identify outliers is empirical.

{{< figure
    src="llm_int8_quantization.png"
    caption="Fig. 15. Two mixed-precision decompositions of `LLM.int8()`. (Image source: [Dettmers et al. 2022](https://arxiv.org/abs/2208.07339))"
    align="center"
    width="100%"
>}}

#### Quantization at fine-grained granularity

{{< figure
    src="quantization_granularity.png"
    caption="Fig. 16. Comparison of quantization at different granularities. $d$ is the model size / hidden state dimension and $h$ is the number of heads in one MHSA (multi-head self-attention) component. (Image source: [Lilian, 2023](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))"
    align="center"
    width="100%"
>}}

Simply quantizing the entire weight matrix of a layer ("per-tensor" or "per-layer" quantization) is the easiest to implement but cannot achieve good quantization granularity.

**Q-BERT** ([Shen, et al. 2020](https://arxiv.org/abs/1909.05840)) applies **group-wise quantization** to a fine-tuned BERT model, treating the individual matrix $W$ corresponding to each head in the MHSA (multi-head self-attention) as a group, and then applies Hessian-based mixed-precision quantization.

**Per-embedding group (PEG)** ([Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948)) activation quantization is motivated by the observation that outliers only appear in a few dimensions of the $d$ (hidden state/model size) dimension. Per-embedding quantization is quite computationally expensive. In contrast, PEG quantization divides the activation tensor into several uniformly sized groups along the embedding dimension, where elements in the same group share quantization parameters. To ensure that all outliers are assigned to the same group, they apply a deterministic range-based permutation of the embedding dimensions, where dimensions are sorted by their value range.

**ZeroQuant** ([Yao et al. 2022](https://arxiv.org/abs/2206.01861)) uses group-wise quantization for weights (same as Q-BERT) and **token-wise quantization** for activations. To avoid expensive quantization and dequantization computations, ZeroQuant builds custom kernels that fuse the quantization operation with its preceding operation.

#### Second-order information for quantization

**Q-BERT** ([Shen, et al. 2020](https://arxiv.org/abs/1909.05840)) developed **Hessian AWare Quantization (HAWQ)** ([Dong, et al. 2019](https://arxiv.org/abs/1905.03696)) for its mixed-precision quantization. The motivation is that parameters with a higher Hessian spectrum (i.e., larger top eigenvalues) are more sensitive to quantization and thus require higher precision. This is essentially a method for identifying outliers.

From another perspective, the quantization problem is an optimization problem. Given a weight matrix $\mathbf{W}$ and an input matrix $\mathbf{X}$, we want to find a quantized weight matrix $\hat{\mathbf{W}}$ that minimizes the mean squared error (MSE):

$$
\hat{\mathbf{W}}^* = \arg \min_{\hat{\mathbf{W}}} |\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}|
$$

[**GPTQ**](https://github.com/IST-DASLab/gptq) ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)) builds on the **OBC (Optimal Brain Compression)** ([Frantar et al. 2022](https://arxiv.org/abs/2208.11580)) method, treating the weight matrix $\mathbf{W}$ as a set of row vectors $\mathbf{w}$ and quantizing each row independently. GPTQ iteratively quantizes more weights, which are chosen greedily to minimize the quantization error. The update for the selected weights has a closed-form solution that utilizes the Hessian matrix.

{{< figure
    src="gptq.png"
    caption="Fig. 17. The pseudocode for the GPTQ algorithm. (Image source: [Frantar et al. 2022](https://arxiv.org/abs/2210.17323))"
    align="center"
    width="100%"
>}}

GPTQ can reduce the bit-width of weights in OPT-175B to **3-bit** or **4-bit** without much performance loss, but it only applies to model weights, not activations.

#### Outlier smoothing

{{< figure
    src="migrate_quantization_difficulty.png"
    caption="Fig. 18. Magnitude of the input activations and weights of a linear layer in OPT-13B before and after SmoothQuant (Image source: [Xiao et al. 2022](https://arxiv.org/abs/2211.10438))"
    align="center"
    width="100%"
>}}

From the figure above, we can see that in Transformer models, activations are harder to quantize than weights. There are three main characteristics:

1.  **Activations are harder to quantize than weights**: Quantizing weights to INT8/INT4 barely affects accuracy, but activations are more sensitive.
2.  **Outliers amplify the difficulty of activation quantization**: Extreme values in activations are about 100 times larger than normal values. Direct INT8 quantization would crush most small values to zero.
3.  **Outliers are fixed in a few channels**: These extreme values are consistently concentrated in specific channels, leading to a highly uneven distribution across channels.

**SmoothQuant** ([Xiao et al. 2022](https://arxiv.org/abs/2211.10438)) proposes a clever solution by **smoothing outlier features from activations to weights through a mathematically equivalent transformation**, and then quantizing both weights and activations (`W8A8`). Therefore, SmoothQuant has better hardware efficiency than mixed-precision quantization.

{{< figure
    src="smooth_quant.png"
    caption="Fig. 19. SmoothQuant migrates the scale variance from activations to weights offline to reduce the difficulty of activation quantization. Both the resulting new weight and activation matrices are easy to quantize. (Image source: [Xiao et al. 2022](https://arxiv.org/abs/2211.10438))"
    align="center"
    width="80%"
>}}

Considering a per-channel smoothing factor $\mathbf{s}$, SmoothQuant scales the weights according to the following formula:

$$
\mathbf{Y} = (\mathbf{X}\text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s})\mathbf{W}) = \hat{\mathbf{X}}\hat{\mathbf{W}}
$$

The smoothing factor can be easily fused into the parameters of the preceding layer offline. A hyperparameter $\alpha$ controls the degree to which the quantization difficulty is migrated from activations to weights: $\mathbf{s} = \max(|\mathbf{X}_j|)^\alpha / \max(|\mathbf{W}_j|)^{1-\alpha}$. The paper finds that for many LLMs, $\alpha=0.5$ is an optimal choice in experiments. For models with more significant outliers in activations, $\alpha$ can be adjusted to be larger.

### Quantization-Aware Training (QAT)

Quantization-Aware Training (QAT) integrates the quantization operation into the pre-training or fine-tuning process, directly learning a low-bit representation of the model weights. It achieves higher performance at the cost of additional training time and computational resources.

Common QAT methods include:

*   **Direct Fine-tuning**: The model is first quantized once, and then further **fine-tuned** on the original pre-training dataset or a representative training dataset. This makes the model sensitive to quantization errors and allows it to actively compensate, thereby improving the performance of the quantized model. The training objective can be the same as the original pre-training objective (e.g., negative log-likelihood NLL or masked language modeling MLM in language models) or a task-specific objective (e.g., cross-entropy for classification tasks). A typical implementation is [QLoRA](https://syhya.github.io/posts/2025-03-01-train-llm/#qlora), which achieves efficient fine-tuning by combining a low-bit (e.g., 4-bit) base model with full-precision LoRA adapters.

*   **Knowledge Distillation**: A full-precision model acts as the teacher, and a low-precision model acts as the student. A **distillation loss** guides the student model to approach the teacher's performance. The Layer-by-layer Knowledge Distillation (LKD) technique used by **ZeroQuant** ([Yao et al. 2022](https://arxiv.org/abs/2206.01861)) is an example of this method. It quantizes the model weights layer by layer, with each quantized layer using the corresponding full-precision layer as a teacher, minimizing the mean squared error (MSE) between their weight computation results to improve performance.

$$
\mathcal{L}_{L K D, k}=M S E\left(L_k \cdot L_{k-1} \cdot L_{k-2} \cdot \ldots \cdot L_1(\boldsymbol{X})-\widehat{L}_k \cdot L_{k-1} \cdot L_{k-2} \cdot \ldots \cdot L_1(\boldsymbol{X})\right)
$$

## Pruning

Network pruning reduces model size by removing unimportant model weights or connections, thereby achieving model compression while maintaining performance as much as possible. Depending on the implementation, pruning can be divided into **unstructured pruning** and **structured pruning**.

*   **Unstructured pruning**: Not limited to a specific pattern, it can discard weights or connections at any position in the network, thus disrupting the original structural regularity of the network. Because the resulting sparse patterns are difficult to adapt to modern hardware architectures, this method usually cannot effectively improve actual inference efficiency.

*   **Structured pruning**: Maintains the network's structure by trimming entire structures (such as convolutional kernels, channels, or layers). The pruned network is still suitable for dense matrix computations optimized by existing hardware, thus significantly improving actual inference performance. In this article, we focus on structured pruning to achieve efficient sparse structures in Transformer models.

A typical workflow for network pruning includes the following three steps:

1.  Train a full, dense network until convergence.
2.  Prune the network by removing redundant structures or weights.
3.  (Optional) Further fine-tune the network to recover the performance of the pruned model.

### The Lottery Ticket Hypothesis

One of the theoretical foundations of pruning is the **Lottery Ticket Hypothesis (LTH)** ([Frankle & Carbin, 2019](https://arxiv.org/abs/1803.03635)). This hypothesis posits that a randomly initialized dense neural network contains certain sparse subnetworks (the "winning tickets") that, when trained in isolation, can achieve performance comparable to or even better than the full network.

The core idea of LTH is that not all parameters are equally important. Only a small fraction of the parameters in a model play a crucial role. This suggests that a large number of parameters are not primarily for solving overfitting but mainly provide a sufficient initialization search space for high-performance subnetworks to be discovered.

To test this hypothesis, Frankle and Carbin proposed the following experimental steps:

1.  Randomly initialize a dense neural network with initial weights $\theta_0$.
2.  Train the full network to achieve good performance, with final parameters $\theta$.
3.  Prune the trained parameters $\theta$ to generate a sparse mask $m$.
4.  Select the "winning ticket" subnetwork, with initial parameters defined as $m \odot \theta_0$.

The experiment found that by using only the small number of "winning ticket" parameters selected in step 1 and training them from their original random initial values, the model could still achieve almost the same accuracy as the original network.

This result indicates that the vast initial parameter space is not necessary for the final deployed model but provides a large number of initial possibilities during the training phase, allowing the network to discover high-performing sparse structures. This also explains why, although a pruned model is significantly smaller, training the same sparse structure from scratch is often difficult to succeed.

### Pruning Strategies

**Magnitude pruning** is the simplest yet quite effective pruning method—weights with the smallest absolute values are pruned. In fact, some studies ([Gale et al. 2019](https://arxiv.org/abs/1902.09574)) have found that simple magnitude pruning methods can achieve comparable or better results than complex pruning methods like variational dropout ([Molchanov et al. 2017](https://arxiv.org/abs/1701.05369)) and $l_0$ regularization ([Louizos et al. 2017](https://arxiv.org/abs/1712.01312)). Magnitude pruning is easy to apply to large models and achieves fairly consistent performance across a wide range of hyperparameters.

**Gradual Magnitude Pruning (GMP)** ([Zhu & Gupta, 2017](https://arxiv.org/abs/1710.01878)) is based on the idea that large sparse models can achieve better performance than small but dense models. It proposes gradually increasing the sparsity of the network during training. At each training step, weights with the smallest absolute values are masked to zero to achieve a desired sparsity level $s$, and the masked weights do not receive gradient updates during backpropagation. The desired sparsity level $s$ increases with the number of training steps. The GMP process is sensitive to the learning rate schedule, which should be higher than that used in dense network training but not so high that it fails to converge.

**Iterative pruning** ([Renda et al. 2020](https://arxiv.org/abs/2003.02389)) iterates through step 2 (pruning) and step 3 (retraining) multiple times: in each iteration, only a small fraction of weights are pruned, and then the model is retrained. This process is repeated until the desired sparsity level is reached.

### Retraining

The retraining step can be simple fine-tuning, using the same pre-training data or other task-specific datasets.

The **Lottery Ticket Hypothesis** proposes a retraining technique called **weight rewinding**: after pruning, the unpruned weights are re-initialized to their original values from early in training, and then retrained using the same learning rate schedule.

**Learning rate rewinding** ([Renda et al. 2020](https://arxiv.org/abs/2003.02389)) only resets the learning rate to its early value, while the unpruned weights remain unchanged from the end of the previous training phase. They observed that (1) on various networks and datasets, retraining with weight rewinding is superior to retraining with fine-tuning; and (2) in all tested scenarios, learning rate rewinding is comparable or superior to weight rewinding.

## Sparsity

Sparsity is an effective way to scale model capacity while maintaining computational efficiency for model inference. Here we consider two types of sparsity for Transformers:

*   Sparsified dense layers, including self-attention and FFN layers.
*   Sparse model architectures; i.e., by introducing Mixture-of-Experts (MoE) components.

### N:M Sparsity via Pruning

**N:M sparsity** is a structured sparse pattern that works well with modern GPU hardware optimizations, where N out of every M consecutive elements are zero. For example, the [Nvidia A100](https://www.nvidia.com/en-us/data-center/a100/)'s sparse tensor cores support 2:4 sparsity for faster inference.

{{< figure
    src="sparsity.png"
    caption="Fig. 20. The illustration of achieving N:M structure sparsity. (Image source: [Zhou et al. 2021](https://arxiv.org/abs/2102.04010))"
    align="center"
    width="100%"
>}}

To sparsify a dense neural network to follow an N:M structured sparse pattern, Nvidia recommends a three-step conventional workflow for training the pruned network: train -> prune to meet 2:4 sparsity -> retrain.

Permutations can provide more options during pruning to preserve large-magnitude parameters or satisfy special constraints like N:M sparsity. The result of a matrix multiplication does not change as long as the paired axes of two matrices are permuted in the same order. For example:

(1) Within a self-attention module, if the same permutation order is applied to axis 1 of the query embedding matrix $\mathbf{Q}$ and axis 0 of the key embedding matrix $\mathbf{K}^\top$, the final matrix multiplication result of $\mathbf{Q}\mathbf{K}^\top$ will remain unchanged.

{{< figure
    src="permutation_attention.png"
    caption="Fig. 21. Illustration of the same permutation on $\mathbf{Q}$ (axis 1) and $\mathbf{K}^\top$ (axis 0) to keep the results of a self-attention module unchanged. (Image source: [Lilian, 2023](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))"
    align="center"
    width="90%"
>}}

(2) Within an FFN layer containing two MLP layers and a ReLU non-linearity, we can permute axis 1 of the first linear weight matrix $\mathbf{W}_1$ and axis 0 of the second linear weight matrix $\mathbf{W}_2$ in the same order.

{{< figure
    src="permutation_ffn.png"
    caption="Fig. 22. Illustration of the same permutation on $\mathbf{W}_1$ (axis 1) and $\mathbf{W}_2$ (axis 0) to keep the FFN layer's output unchanged. For simplicity, the bias terms are skipped but the same permutation should be applied to them too. (Image source: [Lilian, 2023](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))"
    align="center"
    width="100%"
>}}

To enforce N:M structured sparsity, we divide the columns of a matrix into segments of M columns (called "stripes"). We can easily observe that the order of columns within each stripe and the order of the stripes themselves have no effect on the N:M sparsity constraint.

### Channel Permutations
**Channel Permutations** ([Pool & Yu, 2021](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html)) employs an iterative greedy algorithm to find the optimal permutation that maximizes the weight magnitude for N:M sparsity. All pairs of channels are speculatively swapped, and only the swap that results in the largest increase in magnitude is adopted, thus generating a new permutation and ending a single iteration. The greedy algorithm may only find a local optimum, so they introduce two techniques to escape local optima:

1.  **Bounded regressions**: In practice, randomly swap two channels for a fixed number of times. The solution search is limited to a depth of one channel swap to keep the search space broad and shallow.
2.  **Narrow, deep search**: Select multiple stripes and optimize them simultaneously.

{{< figure
    src="greedy_permulation_search.png"
    caption="Fig. 23. Algorithm for finding the best permutation for N:M sparsity greedily and iteratively. (Image source: [Pool & Yu, 2021](https://proceedings.neurips.cc/paper/2021/hash/1415fe9c182320c8d536c9493793334d-Abstract.html))"
    align="center"
>}}

If the network is permuted before pruning, it can achieve better performance compared to pruning it in its default channel order.

### STE & SR-STE

The **Straight-Through Estimator (STE)** ([Bengio et al. 2013](https://arxiv.org/abs/1308.3432)) computes the gradient of the dense parameters with respect to the pruned network $\widetilde{W}$, $\partial\mathcal{L}/\partial\widetilde{W}$, and applies it to the dense network $W$ as an approximation:
$$
W_{t+1} \leftarrow W_t - \gamma \frac{\partial\mathcal{L}}{\partial\widetilde{W}}
$$

**Sparse-refined STE (SR-STE)** ([Zhou et al. 2021](https://arxiv.org/abs/2102.04010)) extends the STE method to train a model with N:M sparsity from scratch. It is commonly used for backpropagation updates in model quantization and is adapted for magnitude pruning and sparse parameter updates. The dense weights $W$ are updated as follows:
$$
W_{t+1} \leftarrow W_t - \gamma \frac{\partial\mathcal{L}}{\partial\widetilde{W}} + \lambda_W(\overline{\mathcal{E}} \odot W_t)
$$
where $\overline{\mathcal{E}}$ is the mask matrix of $\widetilde{W}$, and $\odot$ is element-wise multiplication. SR-STE aims to prevent large changes in the binary mask by (1) limiting the values of weights pruned in $\widetilde{W}_t$, and (2) boosting the weights that are not pruned in $\widetilde{W}_t$.

{{< figure
    src="sr-ste.png"
    caption="Fig. 24. Comparison of STE and SR-STE. $\odot$ is element-wise product; $\otimes$ is matrix multiplication. (Image source: [Zhou et al. 2021](https://arxiv.org/abs/2102.04010))"
    align="center"
    width="100%"
>}}

### Top-KAST

The **Top-K Always Sparse Training (Top-KAST)** ([Jayakumar et al. 2021](https://arxiv.org/abs/2106.03517)) method, unlike STE or SR-STE, can maintain constant sparsity in both the forward and backward passes without needing dense parameters or dense gradients for the forward pass.

At a training step $t$, Top-KAST proceeds as follows:

1.  **Sparse Forward Pass**: Select a subset of parameters $A^t \subset \Theta$, containing the top $K$ parameters of each layer sorted by magnitude, limited to the top $D$ proportion of weights. In the parameterization $\alpha^t$ at time $t$, if a parameter is not in $A^t$ (the active weights), its value is zero.
    $$
    \alpha_i^t = \begin{cases} \theta_i^t & \text{if } i \in A^t = \{i \mid \theta_i^t \in \text{TopK}(\theta^t, D)\} \\ 0 & \text{otherwise} \end{cases}
    $$
    where $\text{TopK}(\theta, x)$ selects the top $x$ proportion of weights from $\theta$ based on magnitude.

2.  **Sparse Backward Pass**: The gradient is then applied to a larger subset of parameters $B \subset \Theta$, where $B$ contains a $(D+M)$ proportion of weights and $A \subset B$. Updating a larger proportion of weights allows for more effective exploration of different pruning masks, making it more likely to cause permutations in the top $D$ proportion of active weights.
    $$
    \Delta\theta_i^t = \begin{cases} -\eta \nabla_{\alpha_t} \mathcal{L}(y, x, \alpha^t)_i & \text{if } i \in B^t = \{i \mid \theta_i^t \in \text{TopK}(\theta^t, D+M)\} \\ 0 & \text{otherwise} \end{cases}
    $$

Training is divided into two phases, and the additional coordinates in the set $B \setminus A$ control how much exploration is introduced. The amount of exploration is expected to decrease gradually during the training process, and the mask will eventually stabilize.

{{< figure
    src="top_kast.png"
    caption="Fig. 25. The pruning mask of Top-KAST stabilizes in time. (Image source: [Jayakumar et al. 2021](https://proceedings.neurips.cc/paper/2020/hash/47d1e990583c9c67424d369f3414728e-Abstract.html))"
    align="center"
    width="100%"
>}}

To prevent the "rich get richer" phenomenon, Top-KAST penalizes the magnitude of active weights through an L2 regularization loss to encourage the exploration of new items. Parameters in $B \setminus A$ are penalized more than those in $A$ to set a higher selection threshold for a stable mask during updates.
$$
L_{\text{penalty}}(\alpha_i^t) = \begin{cases} |\theta_i^t| & \text{if } i \in A^t \\ |\theta_i^t|/D & \text{if } i \in B^t \setminus A^t \\ 0 & \text{otherwise} \end{cases}
$$

### Sparsified Transformer

**Scaling Transformer** ([Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763)) sparsifies the self-attention and FFN layers in the Transformer architecture, achieving a 37x speedup in single-sample inference.

{{< figure
    src="sparsified_transformer_speed.png"
    caption="Fig. 26. Decoding speed of a single token for Terraformer with 17B parameters is 37x faster than a dense baseline model. (Image source: [Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763))"
    align="center"
    width="70%"
>}}

**Sparse FFN Layer**: Each FFN layer contains 2 MLPs and a ReLU. Because ReLU introduces a large number of zero values, they enforce a fixed structure on the activations, forcing only one non-zero value in a block of $N$ elements. The sparse pattern is dynamic and different for each token.
$$
\begin{aligned}
Y_{\text{sparse}} &= \max(0, xW_1 + b_1) \odot \text{Controller}(x) \\
\text{SparseFFN}(x) &= Y_{\text{sparse}} W_2 + b_2 \\
\text{Controller}(x) &= \arg\max(\text{Reshape}(xC_1C_2, (-1, N)))
\end{aligned}
$$
where each activation in $Y_{\text{sparse}}$ corresponds to a column in $W_1$ and a row in $W_2$. The controller is a low-rank bottleneck dense layer, $C_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{lowrank}}}, C_2 \in \mathbb{R}^{d_{\text{lowrank}} \times d_{\text{ff}}}$ and $d_{\text{lowrank}} = d_{\text{model}}/N$. It uses `arg max` at inference time to select which columns should be non-zero, and the Gumbel-softmax trick during training. Because we can compute $\text{Controller}(x)$ before loading the FFN weight matrices, we know which columns will be zeroed out and thus choose **not to load them into memory** to speed up inference.

{{< figure
    src="sparse_ffn.png"
    caption="Fig. 27. (a) Sparse FFN layer; columns in red are not loaded in memory for faster inference. (b) Sparse FFN controller for 1:4 sparsity. (Image source: [Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763))"
    align="center"
    width="100%"
>}}

**Sparse QKV (Attention) Layer**: In the attention layer, the dimension $d_{\text{model}}$ is divided into $S$ modules, each of size $M = d_{\text{model}}/S$. To ensure that each sub-part can access any part of the embedding, Scaling Transformer introduces a **multiplicative layer** (i.e., a layer that multiplies inputs from multiple neural network layers element-wise), which can represent any permutation but contains fewer parameters than a dense layer.

Given an input vector $x \in \mathbb{R}^{d_{\text{model}}}$, the multiplicative layer outputs $y \in \mathbb{R}^{S \times M}$:
$$
y_{s,m} = \sum_i x_i D_{i,s} E_{i,m} \quad \text{where } D \in \mathbb{R}^{d_{\text{model}} \times S}, E \in \mathbb{R}^{d_{\text{model}} \times M}
$$
The output of the multiplicative layer is a tensor of size $\mathbb{R}^{\text{batch size} \times \text{length} \times S \times M}$. It is then processed by a 2D convolutional layer, where `length` and $S$ are treated as the height and width of an image. Such a convolutional layer further reduces the number of parameters and computation time of the attention layer.

{{< figure
    src="sparse_qkv.png"
    caption="Fig. 28. (a) A multiplicative layer is introduced to enable partitions to access any part of an embedding. (b) Combination of multiplicative dense layer and 2-D convolutional layer reduces the number of parameters and computation time of the attention layer. (Image source: [Jaszczur et al. 2021](https://arxiv.org/abs/2111.12763))"
    align="center"
    width="100%"
>}}

To better handle long sequences, Scaling Transformer is further equipped with LSH (Locality-Sensitive Hashing) attention and FFN block recurrence from **Reformer** ([Kitaev, et al. 2020](https://arxiv.org/abs/2001.04451)).

### Mixture of Experts

[**Mixture-of-Experts (MoE)**](https://syhya.github.io/posts/2025-04-18-deepseek-v2-v3/#mixture-of-experts-model) models consist of multiple "expert" networks, where each input sample activates only a subset of these experts for computation.

{{< figure
    src="dense_sparse_model.png"
    caption="Fig. 29. Dense Transformer vs Sparse Expert Transformer. (Image source: [Fedus et al. 2022](https://arxiv.org/abs/2209.01667))"
    align="center"
    width="100%"
>}}

*   **Dense Model**: All input tokens are processed using the same feed-forward network (FFN) parameters. While the structure is simple and easy to train, its computational cost increases rapidly as the model size grows.

*   **Sparse Expert Model**: Each input token is independently routed to a few experts among many for processing. This sparse routing mechanism allows the model to have more unique parameters without a significant increase in overall computational cost, thus improving parameter efficiency and scalability, and effectively reducing the computational cost during inference.

#### Routing Strategy Improvements

The MoE layer has a routing network that assigns a subset of experts to each input token. In traditional MoE models, the routing strategy routes each token to its preferred expert in the order they appear in the natural sequence. If a token is routed to an expert that has already reached its capacity, the token is marked as "overflowed" and skipped.

**Vision MoE (V-MoE)** ([Riquelme et al. 2021](https://arxiv.org/abs/2106.05974)) adds MoE layers to the ViT (Vision Transformer). It achieves previous SOTA performance with only half the inference computation. V-MoE can be scaled up to 15B parameters. Their experiments used $k=2$, 32 experts, and placed an expert layer every 2 layers (meaning MoE was placed in every other layer).

Due to the limited capacity of each expert, some important and informative tokens might be dropped if they appear too late (e.g., the order of words in a sentence, or the order of image patches). To avoid this drawback of the traditional routing scheme, V-MoE employs **Batch Priority Routing (BPR)**, which first assigns experts to tokens with high priority. BPR computes a priority score for each token before expert assignment (the maximum or sum of the top-k router scores) and changes the order of tokens accordingly. This ensures that the expert capacity buffer will be filled with critical tokens first.

{{< figure
    src="v_moe_bpr.png"
    caption="Fig. 30. How image patches are discarded according to priority scores when $C < 1$. (Image source: [Riquelme et al. 2021](https://arxiv.org/abs/2106.05974))"
    align="center"
    width="100%"
>}}

When $C \le 0.5$, BPR performs much better than traditional routing, as the model starts to drop a large number of tokens. It allows the model to compete with dense networks even at fairly low capacities.

When studying how to interpret the association between image classes and experts, they observed that early MoE layers are more general, while later MoE layers may specialize in a few image classes.

**Task MoE (Task-level Mixture-of-Experts)** ([Kudugunta et al. 2021](https://arxiv.org/abs/2110.03742)) considers task information and routes tokens at the **task level** rather than the word or token level in machine translation. They use MNMT (Multilingual Neural Machine Translation) as an example and group translation tasks based on the target language or language pair.

Token-level routing is dynamic, with routing decisions made independently for each token. Therefore, at inference time, the server needs to pre-load all experts. In contrast, task-level routing is **static** for a given fixed task, so an inference server for a task only needs to pre-load $k$ experts (assuming top-k routing). According to their experiments, Task MoE can achieve similar performance gains as Token MoE compared to a dense model baseline, with 2.6x higher peak throughput and only 1.6% of the decoder size.

Task-level MoE essentially classifies the task distribution based on predefined heuristic rules and incorporates this human knowledge into the router. When such heuristic rules do not exist (e.g., for a general sentence completion task), how to utilize Task MoE is less straightforward.

**PR-MoE (Pyramid residual MoE)** ([Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596)) has each token pass through a fixed MLP and a selected expert. Observing that MoE is more beneficial in later layers, PR-MoE employs more experts in the later layers. The DeepSpeed library implements a flexible multi-expert, multi-data parallel system to support training PR-MoE with varying numbers of experts.

{{< figure
    src="pr_moe.png"
    caption="Fig. 31. Illustration of PR-MoE architecture in comparison with a standard MoE. (Image source: [Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596))"
    align="center"
    width="100%"
>}}

#### Kernel Improvement

Expert networks can be hosted on different devices. However, as the number of GPUs increases, the number of experts per GPU decreases, and the communication between experts ("All-to-all") becomes more expensive. All-to-all communication between experts across multiple GPUs relies on NCCL's P2P API, which cannot saturate the bandwidth of high-speed links (like NVLink, HDR InfiniBand) at a large scale because individual data blocks become smaller as more nodes are used. Existing all-to-all algorithms perform poorly in large-scale scenarios with small workloads. There are several kernel improvements that enable more efficient MoE computation, such as making all-to-all communication cheaper/faster.

**DeepSpeed** ([Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596)) and **TUTEL** ([Hwang et al. 2022](https://arxiv.org/abs/2206.03382)) both implement a tree-based **hierarchical all-to-all algorithm** that first runs an intra-node all-to-all, followed by an inter-node all-to-all. It reduces the number of communication hops from $O(G)$ to $O(G_{\text{node}} + G/G_{\text{node}})$, where $G$ is the total number of GPU nodes and $G_{\text{node}}$ is the number of GPU cores per node. Although the communication volume is doubled in this implementation, it achieves better scalability in small-batch, large-scale scenarios because the bottleneck is latency rather than communication bandwidth.

**DynaMoE** ([Kossmann et al. 2022](https://arxiv.org/abs/2205.01848)) uses **dynamic recompilation** to adapt computational resources to the dynamic workload among experts. The `RECOMPILE` mechanism compiles the computation graph from scratch and reallocates resources only when needed. It measures the number of samples assigned to each expert and dynamically adjusts their capacity factor $C$ to reduce memory and computational requirements at runtime. Based on the observation that sample-expert assignments converge early in training, a **sample assignment cache** is introduced after convergence, and then `RECOMPILE` is used to eliminate dependencies between the gating network and the experts.

## Architectural Optimization

### Efficient Transformers

The survey paper **Efficient Transformers** ([Tay et al. 2020](https://arxiv.org/abs/2009.06732)) reviews a series of Transformer architectures with improvements in computational and memory efficiency. Readers interested in this topic can read the original paper.

{{< figure
    src="efficient_transformers.png"
    caption="Fig. 32. Categorization of efficient transformer models. (Image source: [Tay et al. 2020](https://arxiv.org/abs/2009.06732))"
    align="center"
    width="100%"
>}}

### KV Cache Optimization

*   **Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)**: In standard Multi-Head Attention (MHA), each head has its own set of Key and Value projection matrices. MQA ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) proposed having all Query heads share the same set of Key and Value heads, which greatly reduces the size of the KV Cache. GQA ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) is a compromise between MHA and MQA, grouping query heads so that heads within a group share a set of K/V, achieving a good balance between performance and effectiveness.

*   **vLLM**([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)) introduced **PagedAttention**, inspired by virtual memory and paging in operating systems. It divides the KV Cache into fixed-size blocks, which can be stored non-contiguously in physical VRAM. A "block table" manages the mapping from logical blocks to physical blocks. For a detailed introduction to vLLM, you can refer to my previous blog post [vLLM: High-Throughput and Memory-Efficient LLM Serving Engine](https://syhya.github.io/posts/2025-05-17-vllm/). This method almost completely eliminates memory fragmentation (both internal and external), bringing VRAM utilization close to 100%. More importantly, through a Copy-on-Write mechanism, it can efficiently share the KV Cache across requests, greatly increasing throughput for complex decoding scenarios like parallel sampling and Beam Search.

### FlashAttention

*   **FlashAttention**([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) is an IO-aware exact attention algorithm. It recognizes that the main bottleneck in standard Attention implementations is the data transfer between GPU HBM (High-Bandwidth Memory) and SRAM (on-chip high-speed cache). FlashAttention uses **Tiling** and **Recomputation** techniques to fuse the entire Attention computation into a single CUDA kernel, avoiding the need to write and read the huge $N \times N$ attention matrix to and from HBM. This dramatically reduces memory access, thereby speeding up Attention computation by several times without sacrificing accuracy. **FlashAttention-2**([Dao, 2023](https://arxiv.org/abs/2307.08691)) further optimizes parallelism and hardware utilization.

{{< figure
    src="flash_attention.png"
    caption="Fig. 33. FlashAttention uses tiling to avoid materializing the large N × N attention matrix on slow GPU HBM, achieving up to 7.6× speedup over PyTorch. (Image source: [Dao et al., 2022](https://arxiv.org/abs/2205.14135))"
    align="center"
    width="100%"
>}}

### References

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

## Citation

> **Citation**: When reproducing or citing the content of this article, please credit the original author and source.

**Cited as:**

> Yue Shui. (Jun 2025). Large Language Model Inference.
https://syhya.github.io/posts/2025-06-29-llm-inference

Or

```bibtex
@article{syhya2025llminferencesurvey,
  title   = "Large Language Model Inference",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jun",
  url     = "https://syhya.github.io/posts/2025-06-29-llm-inference"
}
```