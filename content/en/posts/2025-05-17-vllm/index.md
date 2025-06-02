---
title: "vLLM: High-Throughput, Memory-Efficient LLM Serving"
date: 2025-05-17T10:00:00+08:00
lastmod: 2025-05-17T10:00:00+08:00
author: "Yue Shui"
categories: ["Technical Blog"]
tags: ["vLLM", "PagedAttention", "LLM Serving", "Inference", "KV Cache", "Memory Optimization", "LLMs", "AI Infrastructure", "Deep Learning"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

As the parameters of Large Language Models (LLMs) continue to grow, deploying and serving these models presents significant challenges. [vLLM](https://github.com/vllm-project/vllm) is an open-source library designed for fast, convenient, and cost-effective LLM inference and online serving. Its core lies in the **PagedAttention** algorithm, which efficiently manages the KV Cache in the attention mechanism.

## Evaluation Metrics

To evaluate the performance of LLM inference and serving engines, we primarily focus on the following metrics:

### Time To First Token (TTFT)

**Time To First Token (TTFT)** refers to the time elapsed from when the model receives user input (Prompt) to when it generates the first output token. A shorter TTFT means less waiting time for the user, which is particularly important for real-time interactive scenarios; in offline scenarios, TTFT is relatively less critical.

### Time Per Output Token (TPOT)

**Time Per Output Token (TPOT)** indicates the average time required for the model to generate one new token. It directly determines the user-perceived "speed" of the response. To enhance user experience, [Streaming](https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses) is commonly used in practical applications. For example, if TPOT is 0.1 seconds/token, it means the model can generate about 10 tokens per second, equivalent to approximately 450 words per minute, which exceeds the reading speed of most people.

### Latency

**Latency** is the total time required for the model to generate a complete response for the user. It can be calculated from TTFT and TPOT using the following formula:

$$
\text{Latency} = \text{TTFT} + \text{TPOT} \times (\text{Number of Output Tokens})
$$

### Throughput

**Throughput** measures the total number of tokens (including input and output tokens) that the model inference server can generate per unit of time for all user requests. It reflects the server's processing efficiency and concurrency capability. The specific calculation formula is as follows:

$$
\text{Throughput} = \frac{\text{Batch Size} \times (\text{Number of Input Tokens} + \text{Number of Output Tokens})}{\text{End-to-End Latency}}
$$

### Inter-Token Latency (ITL)

**Inter-Token Latency (ITL)** represents the average time interval between the generation of two consecutive tokens after the first token has been generated. It reflects the speed at which each subsequent token is generated, calculated as:

$$
\text{ITL} = \frac{\text{End-to-End Latency} - \text{TTFT}}{\text{Batch Size} \times (\text{Number of Output Tokens} - 1)}
$$

These metrics reflect the inference engine's response speed, processing efficiency, and concurrency capabilities, serving as important benchmarks for evaluating and optimizing LLM inference performance.

## vLLM V0

Since its initial release in June 2023, vLLM, equipped with PagedAttention, has significantly raised the performance benchmark for LLM serving. It demonstrates a notable throughput advantage over [HuggingFace Transformers (HF)](https://huggingface.co/docs/transformers/main_classes/text_generation) and [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference), without requiring any modifications to the model architecture.

{{< figure
    src="vllm_v0_throughput1.png"
    caption="Fig. 1. Throughput comparison (single output completion) on LLaMA models. vLLM vs. HF and TGI. (Image source: [vLLM Blog, 2023](https://vllm.ai/blog/2023/06/20/vllm.html))"
    align="center"
    width="80%"
>}}

*   Single output inference: The figure shows vLLM achieves 14x-24x higher throughput than HF and 2.2x-2.5x higher throughput than TGI.

{{< figure
    src="vllm_v0_throughput2.png"
    caption="Fig. 2. Throughput comparison (three parallel output completions) on LLaMA models. vLLM vs. HF and TGI. (Image source: [vLLM Blog, 2023](https://vllm.ai/blog/2023/06/20/vllm.html))"
    align="center"
    width="80%"
>}}

*   Three-way parallel inference: The figure shows vLLM achieves 8.5x-15x higher throughput than HF and 3.3x-3.5x higher throughput than TGI.

### Batching

Traditional **Dynamic Batching** waits for an entire batch of requests to complete before processing the next batch. If some requests finish early, this leads to GPU idle time and reduced resource utilization.

In contrast, **Continuous Batching**, employed by vLLM, allows new request sequences to be dynamically inserted during batch execution. Once a sequence completes, it can be immediately replaced by a new sequence, significantly improving GPU utilization and throughput.

{{< figure
    src="batching.png"
    caption="Fig. 3. Dynamic Batching vs Continuous Batching. (Image source: [NYC vLLM Meetup, 2025](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?slide=id.g31441846c39_0_0#slide=id.g31441846c39_0_0))"
    align="center"
    width="100%"
>}}

* **Dynamic Batching**: As shown on the left, sequences S₁-S₄ are processed in parallel from T1-T4. At T5, S₁ and S₃ finish early. However, because S₂ and S₄ are still running, new sequences cannot join immediately, leading to partial GPU idleness. New sequences can only start after S₂ finishes at T6 and S₄ finishes at T7.

* **Continuous Batching**: As shown on the right, T1-T4 are similar to dynamic batching. However, at T5, when S₁ and S₃ complete, new sequences S₅ and S₆ can join and start processing immediately, while S₂ and S₄ continue running. When S₂ finishes at T6, S₇ can join instantly. This approach keeps the GPU almost fully utilized, greatly enhancing efficiency.


### KV Cache

The primary bottleneck in LLM serving performance is memory management. During the autoregressive decoding process, LLMs generate attention Key and Value tensors for each token in the input sequence. These tensors (KV cache) must be retained in GPU memory to generate subsequent tokens. The KV cache has the following characteristics:

1.  **Large:** For the LLaMA-13B model, the KV cache for a single sequence can be up to 1.7 GB.
2.  **Dynamic:** The size of the KV cache depends on the sequence length, which is highly variable and unpredictable.
3.  **Inefficient Management:** Existing inference frameworks like [FasterTransformer](https://github.com/NVIDIA/FasterTransformer?tab=readme-ov-file) and Orca ([Yu et al. 2022](https://www.usenix.org/system/files/osdi22-yu.pdf)) typically store the KV cache in contiguous memory blocks. To handle its dynamic nature, they need to pre-allocate memory blocks large enough to accommodate the maximum possible sequence length. This leads to severe memory waste:
    *   **Internal Fragmentation:** Reserved space is much larger than actually needed.
    *   **External Fragmentation:** Pre-allocated blocks of different sizes make it difficult to utilize memory space efficiently.
    *   **Over-reservation:** Space reserved for future tokens cannot be used by other requests currently.

The figure below illustrates the types of memory waste caused by KV cache management in existing inference systems:

{{< figure
    src="kv_cache_existing_system.png"
    caption="Fig. 4. KV cache memory management in existing systems, showing reserved waste, internal fragmentation, and external fragmentation. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="100%"
>}}

The left image below shows the memory distribution when running a 13B parameter LLM on an NVIDIA A100 GPU: **approximately 65% of the memory is used for static model weights (gray), about 30% is dynamically allocated on demand for the KV cache (red)** to store the attention context of preceding tokens, and a small amount of memory (yellow) is used for temporary activation computations. The right image indicates that vLLM effectively alleviates the memory bottleneck by smoothing the rapid growth of KV cache memory usage, thereby significantly enhancing batch processing capabilities and overall service throughput.

{{< figure
    src="memory_layout.png"
    caption="Fig. 5. Left: Memory layout for a 13B LLM on an NVIDIA A100—gray is persistent parameters, red is per-request KV cache, and yellow is temporary activation memory. Right: vLLM limits rapid KV cache growth, improving throughput. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

### PagedAttention

**PagedAttention** ([Kwon et al. 2023](https://arxiv.org/abs/2309.06180)) is inspired by **Virtual Memory** and **Paging** concepts from operating systems. It allows **logically contiguous KV Cache to be stored in physically non-contiguous GPU memory**.

Specifically, PagedAttention divides the KV Cache of each sequence into fixed-size **Blocks**. Each block contains the Key and Value vectors for a fixed number of tokens. The system maintains a **Block Table** that records the mapping from logical blocks to physical blocks for each sequence.

{{< figure
    src="PagedAttention.png"
    caption="Fig. 6. Illustration of the PagedAttention algorithm, where KV vectors are stored in non-contiguous blocks. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

The core idea of PagedAttention borrows from the virtual memory and paging mechanisms of operating systems to manage the KV cache.

Specifically, the design philosophy of PagedAttention can be summarized as follows:
1.  **Analogy**:
    *   **Blocks** of the KV cache are analogous to **Pages** in OS memory management.
    *   **Tokens** are analogous to **Bytes**.
    *   **Sequences** are analogous to **Processes**.

2.  **Mapping Mechanism**: PagedAttention uses a **Block Table** to maintain the mapping from a sequence's contiguous **Logical Blocks** to **Physical Blocks**. These physical blocks can be non-contiguous in memory, much like an OS page table maps virtual addresses to physical page frames.

3.  **Allocate-on-Demand**: Crucially, **Physical Blocks** are not pre-allocated for the maximum sequence length. Instead, they are **allocated on demand** when new Key-Values need to be stored (i.e., when new tokens are generated).

This on-demand, non-contiguous memory management allows PagedAttention to utilize memory more effectively, avoiding the waste and internal fragmentation caused by pre-allocating large contiguous spaces, thereby improving GPU memory utilization.

Mathematically, PagedAttention transforms attention computation into block-wise computation. Let the block size be $B$. The $j$-th Key block is $K_{j}=\left(k_{(j-1) B+1}, \ldots, k_{j B}\right)$, and the Value block is $V_{j}=\left(v_{(j-1) B+1}, \ldots, v_{j B}\right)$. For a query vector $q_i$, the attention computation becomes:

\[
A_{i j}=\frac{\exp \left(q_{i}^{\top} K_{j} / \sqrt{d}\right)}{\sum_{t=1}^{\lceil i / B\rceil} \exp \left(q_{i}^{\top} K_{t} \mathbf{1} / \sqrt{d}\right)}, \quad o_{i}=\sum_{j=1}^{\lceil i / B\rceil} V_{j} A_{i j}^{\top}
\]

where $A_{i j}=\left(a_{i,(j-1) B+1}, \ldots, a_{i, j B}\right)$ is the row vector of attention scores for the $i$-th query on the $j$-th KV block. During computation, the PagedAttention kernel efficiently identifies and fetches the required physical blocks.

### KV Cache Manager

vLLM's memory manager draws inspiration from the virtual memory mechanisms of operating systems:

1.  **Logical vs. Physical Blocks:** Each request's KV cache is represented as a series of logical blocks. The Block Engine on GPU worker nodes allocates physical memory and divides it into physical blocks.
2.  **Block Table:** Maintains the mapping from logical blocks to physical blocks for each request. Each entry records the physical block address and the number of tokens filled within the block.
3.  **Dynamic Allocation:** Physical blocks are allocated on demand, eliminating the need to pre-reserve space for the maximum length, thereby significantly reducing memory waste.

{{< figure
    src="block_table.png"
    caption="Fig. 7. Block table translation in vLLM. Logical blocks are mapped to non-contiguous physical blocks. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

Consider the example in Fig. 7:

1.  **Prefill Stage:** The input prompt has 7 tokens. Assume a block size of 4. vLLM allocates 2 physical blocks (e.g., physical blocks 7 and 1) and updates the block table, mapping logical block 0 to physical block 7, and logical block 1 to physical block 1. The KV cache for the prompt is computed and filled into these two physical blocks. Logical block 0 is filled with 4 tokens, and logical block 1 is filled with 3 tokens, leaving 1 slot reserved.
2.  **Decode Stage:**
    *   **Step 1:** The next token is computed using PagedAttention. Since logical block 1 still has an empty slot, the new KV cache is stored directly in physical block 1, and the fill count for logical block 1 in the block table is updated.
    *   **Step 2:** Logical block 1 is now full. vLLM allocates a new physical block (e.g., physical block 3), updates the block table to map the new logical block 2 to physical block 3, and stores the newly generated KV cache in physical block 3.

This on-demand allocation method limits memory waste to the last block of each sequence, achieving near-optimal memory utilization (waste below 4%). This allows for batching more requests, thereby increasing throughput.

Fig. 8 shows how vLLM manages memory for two sequences. The logical blocks of the two sequences are mapped to different physical blocks reserved by the block engine on the GPU worker. This means that even logically adjacent blocks do not need to be contiguous in physical GPU memory, allowing both sequences to effectively share and utilize the physical memory space.

{{< figure
    src="two_requests_vllm.png"
    caption="Fig. 8. Storing the KV cache of two requests concurrently in vLLM using paged memory. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

### Memory Sharing

Another key advantage of PagedAttention is efficient memory sharing, especially for complex decoding strategies.

#### Parallel Sampling

When a request needs to generate multiple output sequences from the same prompt (e.g., code completion suggestions), the KV cache for the prompt part can be shared.

{{< figure
    src="parallel_sampling.png"
    caption="Fig. 9. Parallel sampling example. Logical blocks for the shared prompt map to the same physical blocks. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

vLLM achieves sharing via the block table:

1.  **Shared Mapping:** Logical blocks of different sequences can map to the same physical block.
2.  **Reference Counting:** Each physical block maintains a reference count.
3.  **Copy-on-Write (CoW):** When a shared block (reference count > 1) needs to be written to, vLLM allocates a new physical block, copies the content of the original block, updates the block table mapping for the writing sequence, and decrements the reference count of the original physical block. Subsequent writes to this physical block (when its reference count is 1) are performed directly.

This mechanism significantly reduces memory overhead for **Parallel Sampling**, with experiments showing memory savings of up to 55%.

#### Beam Search

During **Beam Search**, different candidate sequences (beams) not only share the prompt part but may also share the KV cache of subsequently generated tokens. The sharing pattern changes dynamically.

{{< figure
    src="beam_search.png"
    caption="Fig. 10. Beam search example ($k=4$). Blocks are dynamically shared and freed based on candidate survival. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

vLLM efficiently manages this dynamic sharing using reference counting and the CoW mechanism, avoiding the frequent and costly memory copy operations found in traditional implementations. Most blocks can be shared; CoW is only needed when newly generated tokens fall into an old shared block (requiring only a single block copy).

#### Shared Prefix

For applications where many prompts share a common prefix (e.g., system instructions, few-shot examples), vLLM can pre-compute and cache the KV cache of these **Shared Prefixes** into a set of physical blocks.

{{< figure
    src="shared_prefix.png"
    caption="Fig. 11. Shared prompt example for machine translation using few-shot examples. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

When processing a request containing such a prefix, its logical blocks are simply mapped to the cached physical blocks (with the last block marked as CoW), thus avoiding redundant computation for the prefix part.

### Scheduling and Preemption

vLLM employs an FCFS scheduling policy. When GPU memory is insufficient to accommodate newly generated KV cache, preemption is necessary:

1.  **Preemption Unit:** Preemption occurs at the **Sequence Group** level (e.g., all candidate sequences of a beam search request). This ensures that the earliest arrived requests are served first, and the latest requests are preempted first.
2.  **Recovery Mechanisms:**
    *   **Swapping:** The KV blocks of preempted sequences are copied to CPU memory. They are swapped back to the GPU when resources become available. This is suitable for scenarios with high PCIe bandwidth and larger block sizes.
    *   **Recomputation:** The KV cache of preempted sequences is discarded. When resources are available, the original prompt and already generated tokens are concatenated, and the KV cache is recomputed efficiently in a single prompt phase. This is suitable for scenarios with lower PCIe bandwidth or smaller block sizes.

### Distributed Execution

vLLM supports [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) style tensor model parallelism.

{{< figure
    src="vllm_system_overview.png"
    caption="Fig. 12. vLLM system overview showing centralized scheduler and distributed workers. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

*   **Centralized Scheduler:** Contains the KV cache manager, maintaining the global mapping from logical to physical blocks.
*   **Shared Mapping:** All GPU workers share the block table.
*   **Local Storage:** Each worker only stores the portion of the KV cache corresponding to the attention heads it is responsible for.
*   **Execution Flow:** The scheduler broadcasts input token IDs and the block table to all workers -> workers execute model computation (including PagedAttention) -> workers synchronize intermediate results via All-Reduce -> workers return sampled results to the scheduler. Memory management information is broadcast once at the beginning of each step, requiring no synchronization between workers.

### Kernel Optimization

To efficiently implement PagedAttention, vLLM develops custom CUDA kernels:

*   **Fused Reshape and Block Write:** Combines splitting new KV cache into blocks, reshaping the layout, and writing to the block table into a single kernel.
*   **Fused Block Read and Attention Computation:** Modifies FasterTransformer's attention kernel to read non-contiguous blocks according to the block table and compute attention on-the-fly, optimizing memory access patterns.
*   **Fused Block Copy:** Batches multiple small block copy operations triggered by CoW into a single kernel launch.

## vLLM V1

In January 2025, the vLLM team released the alpha version of **vLLM V1**, a major upgrade to its core architecture. Based on development experience over the past year and a half, the V1 release revisits key design decisions, integrates various features, and simplifies the codebase.

Building on the success and lessons learned from vLLM V0, vLLM V1 introduces significant upgrades to the core architecture, aiming to provide a cleaner, more modular, easily extensible, and higher-performance codebase.

### Motivation and Goals for V1

*   **Challenges of V0:** As features and hardware support expanded, V0's code complexity increased, making it difficult to combine features effectively and accumulating technical debt.
*   **Goals of V1:**
    *   A simple, modular, and easy-to-modify codebase.
    *   High performance with near-zero CPU overhead.
    *   Unify key optimizations into the architecture.
    *   Enable optimizations by default for zero-configuration.

### Optimized Execution Loop & API Server

{{< figure
    src="vllm_v1_architecture.png"
    caption="Fig. 13. vLLM V1's multiprocessing architecture with an isolated EngineCore. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="80%"
>}}

As GPU computation speeds increase (e.g., Llama-8B inference time on H100 is only ~5ms), CPU overhead (API serving, scheduling, input preparation, decoding, streaming responses) becomes a bottleneck. V1 adopts a **multiprocessing architecture**:

*   **Isolated EngineCore:** Isolates the scheduler and model executor in a core engine loop.
*   **CPU Task Offloading:** Moves CPU-intensive tasks like Tokenization, multimodal input processing, Detokenization, and streaming to separate processes, executing them in parallel with the EngineCore to maximize model throughput.

### Simple & Flexible Scheduler

{{< figure
    src="v1_scheduler.png"
    caption="Fig. 14. vLLM V1's scheduler treats prompt and generated tokens uniformly, enabling features like chunked prefill. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="80%"
>}}

*   **Uniform Processing:** No longer distinguishes between "prefill" and "decode" phases, treating user input tokens and model-generated tokens uniformly.
*   **Simple Representation:** Scheduling decisions are represented by a dictionary, e.g., `{request_id: num_tokens}`, specifying how many tokens to process for each request per step.
*   **Generality:** This representation is sufficient to support features like Chunked Prefills, Prefix Caching, and Speculative Decoding. For example, chunked prefill is implemented by dynamically allocating the processing quantity for each request under a fixed token budget.

### Zero-Overhead Prefix Caching

{{< figure
    src="prefix_caching_benchmark.png"
    caption="Fig. 15. Performance comparison of prefix caching in vLLM V0 and V1. V1 achieves near-zero overhead even at 0% hit rate. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="100%"
>}}

V1 optimizes the implementation of prefix caching (based on hash matching and LRU eviction):

*   **Optimized Data Structures:** Implements constant-time cache eviction.
*   **Reduced Python Object Overhead:** Minimizes object creation.
*   **Result:** Performance degradation is less than 1% even with a 0% cache hit rate. At high hit rates, performance improves severalfold. Therefore, V1 enables prefix caching by default.

### Clean TP Architecture (Tensor-Parallel)

{{< figure
    src="v1_tp_architecture.png"
    caption="Fig. 16. vLLM V1's symmetric tensor-parallel architecture using diff-based updates. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="80%"
>}}

V1 addresses the asymmetric architecture issue in V0 caused by the coupling of the scheduler and Worker 0:

*   **Worker-Side State Caching:** Request states are cached on the worker side.
*   **Incremental Updates:** Only incremental changes (diffs) to the state are transmitted each step, greatly reducing inter-process communication.
*   **Symmetric Architecture:** The scheduler and Worker 0 can run in different processes, resulting in a cleaner, symmetric architecture.
*   **Abstracted Distributed Logic:** Workers behave consistently in single-GPU and multi-GPU setups.

### Efficient Input Preparation

{{< figure
    src="persistent_batch.png"
    caption="Fig. 17. vLLM V1 uses Persistent Batch to cache input tensors and apply diffs. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="70%"
>}}

V0 recreates model input tensors and metadata at each step, leading to high CPU overhead. V1 adopts the [Persistent Batch](https://github.com/InternLM/lmdeploy?tab=readme-ov-file) technique:

*   **Cache Input Tensors:** Caches input tensors.
*   **Apply Diffs:** Only applies incremental changes each step.
*   **Numpy Optimization:** Extensively uses Numpy operations instead of native Python operations to reduce CPU overhead in updating tensors.

### Comprehensive Optimizations

1.  **torch.compile and Piecewise CUDA Graphs**
    *   **`torch.compile` Integration:** V1 fully leverages vLLM's `torch.compile` integration to automatically optimize models, supporting efficient operation for various models and significantly reducing the need for manually writing CUDA kernels.
    *   **Piecewise CUDA Graphs:** By introducing piecewise CUDA graphs, V1 successfully overcomes the limitations of native CUDA graphs, enhancing model flexibility and performance.

2.  **Enhanced Support for Multimodal LLMs**
    *   V1 introduces several key improvements for Multimodal Large Language Models (MLLMs):
        *   **Optimized Preprocessing:** CPU-intensive preprocessing tasks like image decoding, cropping, and transformation are moved to non-blocking separate processes to prevent GPU work from being blocked. A preprocessing cache is also introduced to reuse processed inputs for subsequent requests, especially beneficial for identical multimodal inputs.
        *   **Multimodal Prefix Caching:** In addition to token ID hashes, V1 uses image hashes to identify KV cache entries containing image inputs. This improvement is particularly advantageous in multi-turn dialogue scenarios involving image inputs.
        *   **Encoder Cache:** For applications requiring visual encoder outputs, V1 temporarily caches visual embeddings, allowing the scheduler to process text inputs in chunks without recomputing visual embeddings at each step, thus supporting chunked-fill scheduling for MLLMs.

3.  **FlashAttention 3 Integration**
    *   Due to V1's high dynamism (e.g., combining prefill and decode within the same batch), a flexible and high-performance attention kernel was needed. [FlashAttention 3](https://arxiv.org/abs/2407.08608) perfectly meets this requirement, providing robust feature support while maintaining excellent performance across various use cases.

### Performance Comparison

Thanks to architectural improvements and significantly reduced CPU overhead, V1 achieves up to 1.7x higher throughput compared to V0 (without multi-step scheduling). Performance improvements are even more pronounced for multimodal models.

{{< figure
    src="vllm_v1_llama3.png"
    caption="Fig. 18. Performance comparison between vLLM V0 and V1 on Llama 3.1 8B & Llama 3.3 70B (1xH100). (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="100%"
>}}

{{< figure
    src="v1_qwen2vl.png"
    caption="Fig. 19. Performance comparison between vLLM V0 and V1 on Qwen2-VL 7B (1xH100). (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="60%"
>}}


**Comparison Table:**

| Feature               | vLLM V0                                           | vLLM V1                                                | Improvement Point                                                      |
| :-------------------- | :------------------------------------------------ | :----------------------------------------------------- | :--------------------------------------------------------------------- |
| **Core Technology**   | PagedAttention                                    | PagedAttention + Comprehensive Architectural Refactor  | Retains PagedAttention benefits, optimizes overall architecture        |
| **Memory Efficiency** | Extremely High (Waste < 4%)                       | Extremely High (Waste < 4%)                            | Maintains high memory efficiency                                       |
| **Memory Sharing**    | Supported (CoW)                                   | Supported (CoW)                                        | Maintains efficient sharing                                            |
| **CPU Overhead**      | Relatively high, especially in complex scenarios or low-hit-rate prefix caching | Significantly reduced, near-zero overhead              | Multiprocessing, Persistent Batch, optimized data structures, etc.     |
| **Execution Loop**    | Single process, API server tightly coupled with engine | Multiprocess, API server decoupled from EngineCore, highly parallel | Improves CPU/GPU parallelism, reduces blocking                         |
| **Scheduler**         | Differentiates Prefill/Decode                     | Uniform token processing, dictionary-based scheduling  | Simpler, more flexible, easily supports advanced features              |
| **Prefix Caching**    | Disabled by default (overhead at low hit rates)   | Enabled by default (zero-overhead design)              | Optimized for low hit rates, default enabled for ease of use           |
| **Tensor Parallelism**| Asymmetric architecture (Scheduler+Worker0 in same process) | Symmetric architecture (Scheduler & Worker separated)  | Cleaner architecture, IPC overhead controlled by state caching & Diffs |
| **Multimodal Support**| Basic support                                     | Enhanced support (non-blocking preprocessing, image prefix cache, encoder cache, etc.) | Improves VLM performance and usability                               |
| **Compiler Integration**| Limited                                           | Integrated `torch.compile`                             | Automated model optimization, reduces manual Kernel writing            |
| **Attention Kernel**  | Custom Kernel (based on FasterTransformer)        | Integrated FlashAttention 3                            | Adopts industry standard for better performance and feature support    |
| **Performance (vs V0)**| Baseline                                          | Up to 1.7x throughput increase (text), MLLMs more significant | Gains from comprehensive CPU overhead optimization                     |
| **Code Complexity**   | Increased with features                           | Simpler, more modular                                  | Lowers maintenance cost, facilitates community contribution & dev      |

## Other Inference Frameworks

*   [LightLLM](https://github.com/ModelTC/lightllm): A Python-based lightweight inference and serving framework, known for its lightweight design, scalability, and high-speed performance, drawing on the strengths of other open-source projects like vLLM.
*   [LMDeploy](https://github.com/InternLM/lmdeploy): A toolkit for compressing, deploying, and serving LLMs, featuring the TurboMind inference engine, emphasizing high request throughput and efficient quantization.
*   [SGLang](https://github.com/sgl-project/sglang): A framework for efficiently executing complex LLM programs (especially those involving structured generation) through co-design of a frontend language and a backend execution engine.
*   [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference): Hugging Face's production-grade LLM serving solution, widely used and supporting multiple hardware backends. It leverages vLLM's PagedAttention kernel to provide high-concurrency, low-latency inference services.
*   [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): An open-source library from NVIDIA for optimizing and accelerating LLM inference on NVIDIA GPUs, utilizing TensorRT's ahead-of-time compilation and deep hardware optimization capabilities.

## Summary

vLLM, through its core technology PagedAttention, significantly alleviates the memory bottleneck in LLM serving caused by KV cache management, markedly improving memory utilization and throughput. PagedAttention, inspired by operating system paging mechanisms, enables non-contiguous storage, dynamic allocation, and efficient sharing of the KV cache (supporting parallel sampling, beam search, shared prefixes, etc.).

Building on V0, vLLM V1 comprehensively refactors and optimizes the core architecture. Through a multiprocessing architecture, flexible scheduler, zero-overhead prefix caching, symmetric tensor-parallel architecture, efficient input preparation, `torch.compile` integration, enhanced MLLM support, and FlashAttention 3 integration, V1 further reduces CPU overhead and enhances overall system performance, flexibility, and scalability, laying a solid foundation for rapid iteration of new features in the future.


## References

[1] Kwon, Woosuk, et al. ["Efficient memory management for large language model serving with pagedattention."](https://arxiv.org/abs/2309.06180) Proceedings of the 29th Symposium on Operating Systems Principles. 2023.

[2] vLLM Team. ["vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention."](https://vllm.ai/blog/2023/06/20/vllm.html) vLLM Blog, June 20, 2023.

[3] vLLM Team. ["vLLM V1: A Major Upgrade to vLLM's Core Architecture."](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) vLLM Blog, Jan 27, 2025.

[4] NVIDIA. ["FasterTransformer."](https://github.com/NVIDIA/FasterTransformer) GitHub Repository, 2023.

[5] Yu, Gyeong-In, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. ["Orca: A Distributed Serving System for Transformer-Based Generative Models."](https://www.usenix.org/conference/osdi22/presentation/yu) In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 2022.

[6] OpenAI. ["API Reference - Streaming."](https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses) OpenAI Platform Documentation, 2025.

[7] Wolf, Thomas, et al. ["Transformers: State-of-the-Art Natural Language Processing."](https://www.aclweb.org/anthology/2020.emnlp-demos.6) In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 2020.

[8] Hugging Face. ["Text Generation Inference."](https://github.com/huggingface/text-generation-inference) GitHub Repository, 2025.

[9] Shoeybi, Mohammad, et al. ["Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism."](https://arxiv.org/abs/1909.08053) arXiv preprint arXiv:1909.08053 (2019).

[10] InternLM Team. ["LMDeploy."](https://github.com/InternLM/lmdeploy) GitHub Repository, 2025.

[12] Shah, Jay, et al. ["Flashattention-3: Fast and accurate attention with asynchrony and low-precision."](https://arxiv.org/abs/2407.08608) Advances in Neural Information Processing Systems 37 (2024): 68658-68685.

[13] ModelTC. ["LightLLM."](https://github.com/ModelTC/lightllm) GitHub Repository, 2025.

[14] Zheng, Lianmin, et al. ["Sglang: Efficient execution of structured language model programs."](https://arxiv.org/abs/2312.07104) Advances in Neural Information Processing Systems 37 (2024): 62557-62583.

[15] NVIDIA. ["TensorRT-LLM."](https://github.com/NVIDIA/TensorRT-LLM) GitHub Repository, 2025.

[16] vLLM Team. ["NYC vLLM Meetup Presentation."](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?slide=id.g31441846c39_0_0#slide=id.g31441846c39_0_0) Google Slides, 2025.

## Citation

> **Citation**: When reprinting or citing the content of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (May 2025). vLLM: High-Throughput, Memory-Efficient LLM Serving.
https://syhya.github.io/posts/2025-05-17-vllm

Or

```bibtex
@article{syhya2025vllm-en,
  title   = "vLLM: High-Throughput, Memory-Efficient LLM Serving",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "May",
  url     = "https://syhya.github.io/posts/2025-05-17-vllm"
}
```