---
title: "vLLM：高吞吐、有效内存的LLM服务引擎"
date: 2025-05-17T10:00:00+08:00
lastmod: 2025-05-17T10:00:00+08:00
author: "Yue Shui"
categories: ["技术博客"]
tags: ["PagedAttention", "LLM Serving", "Inference", "KV Cache", "Memory Optimization", "LLMs", "AI Infrastructure"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

随着大语言模型 (Large Language Models, LLMs) 参数不断增大，实际部署和提供这些模型的服务也面临挑战。[vLLM](https://github.com/vllm-project/vllm) 是一个开源库，旨在实现快速、便捷且经济高效的 LLM 推理和在线服务。其核心是利用 **PagedAttention** 算法高效地管理注意力机制中的键和值的缓存（KV Cache）。

## 评价指标

为了评估 LLM 推理与服务引擎的性能，我们主要关注以下几个指标：

### 首 token 生成时间

**首 token 生成时间（Time To First Token, TTFT）** 是指模型从接收到用户输入到生成第一个输出 token 所花费的时间。TTFT 越短，用户等待响应的时间就越少，这对于实时交互场景尤为重要；而在离线场景中，TTFT 的重要性相对较低。

### 每个输出 token 的生成时间

**每个输出 token 的生成时间（Time Per Output Token, TPOT）** 指模型平均生成一个新 token 所需的时间，它直接决定了用户感知到的响应“速度”。为提升体验，实际应用中通常采用 [Streaming](https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses) 方式。例如，如果 TPOT 为 0.1 秒/token，意味着模型每秒可生成约 10 个 token，折合每分钟约 450 个单词，已超过多数人的阅读速度。


### 总体延迟

**总体延迟（Latency）** 指模型为用户生成完整响应所需的总时间。总体延迟可由 TTFT 和 TPOT 计算得出，公式如下：

$$
\text{Latency} = \text{TTFT} + \text{TPOT} \times (\text{Number of Output Tokens})
$$

### 吞吐量

**吞吐量（Throughput）** 衡量模型推理服务器单位时间能为所有用户请求生成的总 token 数量（包括输入与输出 token），体现了服务器的处理效率与并发能力。具体计算公式如下：

$$
\text{Throughput} = \frac{\text{Batch Size} \times (\text{Number of Input Tokens} + \text{Number of Output Tokens})}{\text{End-to-End Latency}}
$$

### Token 间延迟

**Token 间延迟（Inter Token Latency, ITL）** 表示生成连续 token 时每两个 token 间的平均时间间隔。它体现了模型在生成首个 token 后，每个后续 token 的生成速度，计算公式为：

$$
\text{ITL} = \frac{\text{End-to-End Latency} - \text{TTFT}}{\text{Batch Size} \times (\text{Number of Output Tokens} - 1)}
$$

这些指标反映了推理引擎的响应速度、处理效率和并发能力，是评估和优化 LLM 推理性能的重要依据。

## vLLM V0

自 2023 年 6 月首次发布以来，配备 PagedAttention 的 vLLM 显著提升了 LLM 服务的性能标杆，相较于 [HuggingFace Transformers (HF)](https://huggingface.co/docs/transformers/main_classes/text_generation) 和 [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) 具有显著的吞吐量优势, 且无需修改任何模型架构。

{{< figure
    src="vllm_v0_throughput1.png"
    caption="Fig. 1. Throughput comparison (single output completion) on LLaMA models. vLLM vs. HF and TGI. (Image source: [vLLM Blog, 2023](https://vllm.ai/blog/2023/06/20/vllm.html))"
    align="center"
    width="80%"
>}}

*   单输出推理：图中显示了 vLLM 吞吐量比 HF 高 14x-24x，比 TGI 高 2.2x-2.5x。

{{< figure
    src="vllm_v0_throughput2.png"
    caption="Fig. 2. Throughput comparison (three parallel output completions) on LLaMA models. vLLM vs. HF and TGI. (Image source: [vLLM Blog, 2023](https://vllm.ai/blog/2023/06/20/vllm.html))"
    align="center"
    width="80%"
>}}

*   三路并行推理：图中显示了 vLLM 吞吐量比 HF 高 8.5x-15x，比 TGI 高 3.3x-3.5x。

### Batching

传统的**动态批处理 (Dynamic Batching)** 会**等待一批请求全部完成后再处理下一批**，如果某些请求提前结束，会导致 GPU 空闲，资源利用率降低。

而 vLLM 采用的 **连续批处理 (Continuous Batching)** 则允许在批次执行过程中动态插入新的请求序列，**一旦某个序列完成，就可以立即用新的序列替换**，从而显著提高 GPU 利用率和吞吐量。

{{< figure
    src="batching.png"
    caption="Fig. 3. Dynamic Batching vs Continuous Batching. (Image source: [NYC vLLM Meetup, 2025](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?slide=id.g31441846c39_0_0#slide=id.g31441846c39_0_0))"
    align="center"
    width="100%"
>}}

* **Dynamic Batching**: 如图左侧所示，T1-T4 时刻，S₁-S₄ 四个序列并行处理。在 T5 时刻，S₁ 和 S₃ 提前完成，但由于 S₂ 和 S₄ 仍在运行，新的序列无法立即加入，导致 GPU 部分空闲。直到 T6 时刻 S₂ 结束、T7 时刻 S₄ 结束后，新的序列才能开始。

* **Continuous Batching**: 如图右侧所示，T1-T4 时刻与动态批处理类似。但在 T5 时刻，S₁ 和 S₃ 完成后，新的序列 S₅ 和 S₆ 可以立即加入并开始处理，而 S₂ 和 S₄ 继续运行。当 S₂ 在 T6 结束时，S₇ 可以即时加入。这种方式使得 GPU 几乎总是满负荷运行，极大提高了效率。


### KV 缓存

LLM 服务性能的主要瓶颈在于内存管理，在自回归解码过程中，LLM 为输入序列中的每个 token 生成注意力键和值张量，这些 KV 缓存必须保留在 GPU 内存中以生成后续的 token。KV 缓存具有以下特点：

1.  **占用空间大:** 对于 LLaMA-13B 模型，单个序列的 KV 缓存可能高达 1.7 GB。
2.  **动态性:** KV 缓存的大小取决于序列长度，而序列长度是高度可变且不可预测的。
3.  **管理效率低下:** 现有推理框架比如 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer?tab=readme-ov-file), Orca ([Yu et al. 2022](https://www.usenix.org/system/files/osdi22-yu.pdf)) 通常将 KV 缓存存储在连续的内存块中。为了应对动态性，它们需要预先分配足够容纳最大可能序列长度的内存块。这导致了严重的内存浪费：
    *   **内部碎片:** 预留空间远大于实际需要。
    *   **外部碎片:** 不同大小的预留块导致内存空间难以有效利用。
    *   **过度预留:** 为未来 token 预留的空间在当前无法被其他请求利用。

下图展示了现有推理系统中 KV 缓存管理导致的内存浪费类型：

{{< figure
    src="kv_cache_existing_system.png"
    caption="Fig. 4. KV cache memory management in existing systems, showing reserved waste, internal fragmentation, and external fragmentation. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="100%"
>}}

以下左图展示了在 NVIDIA A100 GPU 上运行 13B 参数 LLM 的内存分布：**约 65% 的内存用于静态模型权重（灰色），约 30% 的内存按需动态分配给 KV 缓存（红色）**，用于存储前序 token 的注意力上下文，而少量内存（黄色）则用于临时激活计算；右图则表明 vLLM 通过平滑 KV 缓存内存使用的快速增长，有效缓解了内存瓶颈，从而大幅提升了批量请求处理能力和整体服务吞吐量。

{{< figure
    src="memory_layout.png"
    caption="Fig. 5. Left: Memory layout for a 13B LLM on an NVIDIA A100—gray is persistent parameters, red is per-request KV cache, and yellow is temporary activation memory. Right: vLLM limits rapid KV cache growth, improving throughput. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

### PagedAttention

**PagedAttention** ([Kwon et al. 2023](https://arxiv.org/abs/2309.06180)) 的灵感来源于操作系统中的**虚拟内存 (Virtual Memory)** 和 **分页 (Paging)**。它允许将**逻辑上连续的 KV Cache 存储在物理上非连续的显存空间中**。

具体来说，PagedAttention 将每个序列的 KV Cache 分割成固定大小的 **块 (Blocks)**。每个块包含固定数量 token 的 Key 和 Value 向量。系统维护一个 **块表 (Block Table)**，用于记录每个序列的逻辑块到物理块的映射关系。

{{< figure
    src="PagedAttention.png"
    caption="Fig. 6. Illustration of the PagedAttention algorithm, where KV vectors are stored in non-contiguous blocks. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

PagedAttention 的核心思想借鉴了操作系统的虚拟内存和分页机制来管理 KV 缓存。

具体来说，PagedAttention 的设计理念可以总结为以下几点：
1.  **类比关系**:
    *   KV 缓存的 **块 (Blocks)** 类比于操作系统内存管理的 **页 (Pages)**。
    *   **Token** 类比于 **字节 (Bytes)**。
    *   **序列 (Sequences)** 类比于 **进程 (Processes)**。

2.  **映射机制**: PagedAttention 使用 **块表** 来维护从序列的连续 **逻辑块** 到 **物理块** 的映射。这些物理块在内存中可以是非连续的，就像操作系统的页表将虚拟地址映射到物理页帧一样。

3.  **按需分配**: 最关键的一点是，**物理块** 不是预先为整个序列最大长度分配好的，而是在需要存储新的 Key-Value（即生成新 Token）时 **按需分配**。

这种按需、非连续的内存管理方式，使得 PagedAttention 能更有效地利用内存，避免了因预分配大量连续空间而造成的浪费和内部碎片，从而提高了 GPU 内存的利用率。

数学上，PagedAttention 将注意力计算转化为块级计算。设块大小为 $B$，第 $j$ 个 Key 块为 $K_{j}=\left(k_{(j-1) B+1}, \ldots, k_{j B}\right)$，Value 块为 $V_{j}=\left(v_{(j-1) B+1}, \ldots, v_{j B}\right)$。对于查询向量 $q_i$，注意力计算变为：

\[
A_{i j}=\frac{\exp \left(q_{i}^{\top} K_{j} / \sqrt{d}\right)}{\sum_{t=1}^{\lceil i / B\rceil} \exp \left(q_{i}^{\top} K_{t} \mathbf{1} / \sqrt{d}\right)}, \quad o_{i}=\sum_{j=1}^{\lceil i / B\rceil} V_{j} A_{i j}^{\top}
\]

其中 $A_{i j}=\left(a_{i,(j-1) B+1}, \ldots, a_{i, j B}\right)$ 是第 $i$ 个查询对第 $j$ 个 KV 块的注意力得分行向量。在计算过程中，PagedAttention 内核会高效地识别并获取所需的物理块。

### KV 缓存管理器

vLLM 的内存管理器借鉴了操作系统的虚拟内存机制：

1.  **逻辑块与物理块:** 每个请求的 KV 缓存被表示为一系列逻辑块。GPU 工作节点上的块引擎分配物理内存并将其划分为物理块。
2.  **块表:** 维护每个请求的逻辑块到物理块的映射。每个条目记录物理块地址和块内已填充的 token 数量。
3.  **动态分配:** 物理块按需分配，无需预先保留最大长度的空间，从而消除了大部分内存浪费。

{{< figure
    src="block_table.png"
    caption="Fig. 7. Block table translation in vLLM. Logical blocks are mapped to non-contiguous physical blocks. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

结合上图7的例子：

1.  **预填充阶段:** 输入 prompt 有 7 个 token。假设块大小为 4。vLLM 分配 2 个物理块（例如物理块 7 和 1）并更新块表，将逻辑块 0 映射到物理块 7，逻辑块 1 映射到物理块 1。计算 prompt 的 KV 缓存并填充到这两个物理块中。逻辑块 0 填满 4 个 token，逻辑块 1 填充 3 个 token，剩余 1 个 slot 备用。
2.  **解码阶段 :**
    *   **第 1 步:** 使用 PagedAttention 计算下一个 token。由于逻辑块 1 还有空位，新的 KV 缓存直接存入物理块 1，并更新块表中逻辑块 1 的填充计数。
    *   **第 2 步:** 逻辑块 1 已满。vLLM 分配一个新的物理块（例如物理块 3），更新块表将新的逻辑块 2 映射到物理块 3，并将新生成的 KV 缓存存入物理块 3。

这种按需分配的方式将内存浪费限制在每个序列的最后一个块内，实现了接近最优的内存利用率（浪费低于 4%），从而可以批处理更多请求，提高吞吐量。

图8中展示了 vLLM 如何管理两个序列的内存空间。两个序列的逻辑块被映射到 GPU 工作节点（GPU worker）上由区块引擎预留的不同物理块中。这意味着，即使在逻辑层面相邻的块在物理 GPU 内存中也无需连续，从而两个序列可以有效地共享和利用物理内存空间。

{{< figure
    src="two_requests_vllm.png"
    caption="Fig. 8. Storing the KV cache of two requests concurrently in vLLM using paged memory. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

### 内存共享

PagedAttention 的另一个关键优势是高效的内存共享，尤其适用于复杂的解码策略。

#### 并行采样

当一个请求需要从同一个 prompt 生成多个输出序列时（例如代码补全建议），prompt 部分的 KV 缓存可以共享。

{{< figure
    src="parallel_sampling.png"
    caption="Fig. 9. Parallel sampling example. Logical blocks for the shared prompt map to the same physical blocks. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="80%"
>}}

vLLM 通过块表实现共享：

1.  **共享映射:** 不同序列的逻辑块可以映射到同一个物理块。
2.  **引用计数:** 每个物理块维护一个引用计数。
3.  **写时复制 (Copy-on-Write, CoW):** 当一个共享块（引用计数 > 1）需要被写入时，vLLM 会分配一个新物理块，复制原块内容，更新写入序列的块表映射，并将原物理块的引用计数减 1。后续对该物理块的写入（当引用计数为 1 时）则直接进行。

这种机制显著减少了**并行采样 (Parallel Sampling)** 的内存开销，论文实验显示可节省高达 55% 的内存。

#### 束搜索

**束搜索 (Beam Search)** 在解码过程中，不同的候选序列（beam）不仅共享 prompt 部分，还可能共享后续生成的 token 的 KV 缓存，且共享模式是动态变化的。

{{< figure
    src="beam_search.png"
    caption="Fig. 10. Beam search example ($k=4$). Blocks are dynamically shared and freed based on candidate survival. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

vLLM 通过引用计数和 CoW 机制，高效地管理这种动态共享，避免了传统实现中频繁且昂贵的内存拷贝操作。大部分块可以共享，只有当新生成的 token 落入旧的共享块时才需要 CoW（仅拷贝一个块）。

#### 共享前缀

对于许多 prompt 共享相同前缀（如系统指令、few-shot 示例）的应用场景，vLLM 可以预先计算并缓存这些 **共享前缀 (Shared Prefix)** 的 KV 缓存到一组物理块中。

{{< figure
    src="shared_prefix.png"
    caption="Fig. 11. Shared prompt example for machine translation using few-shot examples. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

当处理包含该前缀的请求时，只需将其逻辑块映射到缓存的物理块（最后一个块标记为 CoW），从而避免了对前缀部分的重复计算。

### 调度与抢占

vLLM 采用 FCFS 调度策略。当 GPU 内存不足以容纳新生成的 KV 缓存时，需要进行抢占：

1.  **抢占单位:** 以**序列组** 为单位进行抢占（例如，一个 beam search 请求的所有候选序列）。确保最早到达的请求优先服务，最晚到达的请求优先被抢占。
2.  **恢复机制:**
    *   **换出:** 将被抢占序列的 KV 块拷贝到 CPU 内存。当资源可用时再换回 GPU。适用于 PCIe 带宽较高且块较大的情况。
    *   **重计算:** 丢弃被抢占序列的 KV 缓存。当资源可用时，将原始 prompt 和已生成的 token 拼接起来，通过一次高效的 prompt phase 重新计算 KV 缓存。适用于 PCIe 带宽较低或块较小的情况。

### 分布式执行

vLLM 支持 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 风格的张量模型并行。

{{< figure
    src="vllm_system_overview.png"
    caption="Fig. 12. vLLM system overview showing centralized scheduler and distributed workers. (Image source: [Kwon et al. 2023](https://arxiv.org/abs/2309.06180))"
    align="center"
    width="70%"
>}}

*   **集中式调度器:** 包含 KV 缓存管理器，维护全局的逻辑块到物理块的映射。
*   **共享映射:** 所有 GPU worker 共享块表。
*   **本地存储:** 每个 worker 只存储其负责的注意力头对应的 KV 缓存部分。
*   **执行流程:** 调度器广播输入 token ID 和块表给所有 worker -> worker 执行模型计算（包括 PagedAttention）-> worker 间通过 All-Reduce 同步中间结果 -> worker 将采样结果返回给调度器。内存管理信息在每步开始时一次性广播，无需 worker 间同步。

### 内核优化

为了高效实现 PagedAttention，vLLM 开发了定制 CUDA 核：

*   **融合 Reshape 和块写入:** 将新 KV 缓存分块、重塑布局、按块表写入融合为单个核。
*   **融合块读取和注意力计算:** 修改 FasterTransformer 的注意力核，使其能根据块表读取非连续块并即时计算注意力，优化内存访问模式。
*   **融合块拷贝:** 将 CoW 触发的多个小块拷贝操作批量化到单个核中执行。

## vLLM V1

2025 年 1 月，vLLM 团队发布了 **vLLM V1** 的 alpha 版本，这是对其核心架构的一次重大升级。基于过去一年半的开发经验，V1 版本重新审视了关键设计决策，整合了各种特性，并简化了代码库。

基于 vLLM V0 的成功和经验教训，vLLM V1 对核心架构进行了重大升级，旨在提供更简洁、模块化、易于扩展且性能更高的代码库。

### V1 的动机与目标

*   **V0 的挑战:** 随着功能和硬件支持的扩展，V0 的代码复杂度增加，特性难以有效组合，技术债务累积。
*   **V1 的目标:**
    *   简洁、模块化、易于修改的代码库。
    *   接近零 CPU 开销的高性能。
    *   将关键优化统一到架构中。
    *   默认启用优化，实现零配置。

### 优化的执行循环与 API 服务器

{{< figure
    src="vllm_v1_architecture.png"
    caption="Fig. 13. vLLM V1's multiprocessing architecture with an isolated EngineCore. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="80%"
>}}

随着 GPU 计算速度加快（例如 H100 上 Llama-8B 推理时间仅约 5ms），CPU 开销（API 服务、调度、输入准备、解码、流式响应）成为瓶颈。V1 采用了**多进程架构**：

*   **隔离的 EngineCore:** 将调度器和模型执行器隔离在核心引擎循环中。
*   **CPU 任务卸载:** 将 Tokenization、多模态输入处理、Detokenization、流式传输等 CPU 密集型任务移至独立进程，与 EngineCore 并行执行，最大化模型吞吐量。

### 简洁灵活的调度器

{{< figure
    src="v1_scheduler.png"
    caption="Fig. 14. vLLM V1's scheduler treats prompt and generated tokens uniformly, enabling features like chunked prefill. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="80%"
>}}

*   **统一处理:** 不再区分 "prefill" 和 "decode" 阶段，统一处理用户输入 token 和模型生成 token。
*   **简单表示:** 调度决策用字典表示，如 `{request_id: num_tokens}`，指定每步为每个请求处理多少 token。
*   **通用性:** 这种表示足以支持块状预填充 (Chunked Prefills)、前缀缓存 (Prefix Caching)、投机解码 (Speculative Decoding) 等特性。例如，块状预填充只需在固定 token 预算下动态分配各请求的处理数量。

### 零开销前缀缓存

{{< figure
    src="prefix_caching_benchmark.png"
    caption="Fig. 15. Performance comparison of prefix caching in vLLM V0 and V1. V1 achieves near-zero overhead even at 0% hit rate. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="100%"
>}}

V1 优化了前缀缓存（基于哈希匹配和 LRU 驱逐）的实现：

*   **优化数据结构:** 实现常数时间缓存驱逐。
*   **减少 Python 对象开销:** 最小化对象创建。
*   **结果:** 即使缓存命中率为 0%，性能下降也小于 1%。而在高命中率时，性能提升数倍。因此，V1 默认启用前缀缓存。

### 清晰的张量并行推理架构

{{< figure
    src="v1_tp_architecture.png"
    caption="Fig. 16. vLLM V1's symmetric tensor-parallel architecture using diff-based updates. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="80%"
>}}

V1 解决了 V0 中调度器和 Worker 0 耦合导致的非对称架构问题：

*   **Worker 端状态缓存:** 请求状态缓存在 Worker 端。
*   **增量更新:** 每步只传输状态的**增量变化 (diffs)**，极大减少了进程间通信。
*   **对称架构:** 调度器和 Worker 0 可以运行在不同进程中，架构更清晰、对称。
*   **抽象分布式逻辑:** Worker 在单 GPU 和多 GPU 设置下行为一致。

### 高效的输入准备

{{< figure
    src="persistent_batch.png"
    caption="Fig. 17. vLLM V1 uses Persistent Batch to cache input tensors and apply diffs. (Image source: [vLLM Blog, 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html))"
    align="center"
    width="70%"
>}}

V0 每步重新创建模型输入张量和元数据，CPU 开销大。V1 采用 [Persistent Batch](https://github.com/InternLM/lmdeploy?tab=readme-ov-file) 技术：

*   **缓存输入张量:** 缓存输入张量。
*   **应用 Diffs:** 每步只应用增量变化。
*   **Numpy 优化:** 大量使用 Numpy 操作替代 Python 原生操作，减少更新张量的 CPU 开销。

### 综合优化

1.  **torch.compile 与分段 CUDA 图**
    *   **集成 `torch.compile`:** V1 充分利用 vLLM 的 `torch.compile` 集成功能，自动优化模型，支持多种模型的高效运行，显著减少了手动编写 CUDA 核的需求。
    *   **分段 CUDA 图 (Piecewise CUDA Graphs):** 通过引入分段 CUDA 图，成功克服了原生 CUDA 图的局限性，提高了模型的灵活性和性能。

2.  **增强的多模态 LLM 支持**
    *   V1 针对多模态大语言模型 (MLLMs) 推出多项关键改进：
        *   **优化预处理:** 将图像解码、裁剪、转换等 CPU 密集型的预处理任务移至非阻塞的独立进程，防止阻塞 GPU 工作。同时引入预处理缓存，以便缓存已处理的输入，供之后的请求复用，尤其适用于相同的多模态输入。
        *   **多模态前缀缓存:** 除了使用 token ID 的哈希，V1 还引入图像哈希来标识包含图像输入的 KV 缓存。此改进在包含图像输入的多轮对话场景中尤其有利。
        *   **编码器缓存:** 针对需要视觉编码器输出的应用，V1 临时缓存视觉嵌入，允许调度器将文本输入分块处理，避免在每一步都重新计算视觉嵌入，从而支持 MLLM 的块状填充调度。

3.  **FlashAttention 3 集成**
    *   由于 V1 的高度动态性（如在同一批处理内结合预填充和解码），需要一种灵活且高性能的注意力核。[FlashAttention 3](https://arxiv.org/abs/2407.08608) 完美符合这一需求，提供了强大的功能支持，同时在各种使用场景中保持优异的性能表现。

### 性能对比

得益于架构改进和 CPU 开销的大幅降低，V1 相比 V0（未开启多步调度）实现了高达 1.7 倍的吞吐量提升，同时在多模态模型上的性能提升显著。

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


**表格对比:**

| 特性             | vLLM V0                                           | vLLM V1                                                | 改进点                                                                 |
| :-------------------- | :------------------------------------------------ | :----------------------------------------------------- | :--------------------------------------------------------------------- |
| **核心技术**          | PagedAttention                                    | PagedAttention + 全面架构重构                          | 保持 PagedAttention 优势，优化整体架构                                 |
| **内存效率**          | 极高 (浪费 < 4%)                                  | 极高 (浪费 < 4%)                                       | 维持高内存效率                                                         |
| **内存共享**          | 支持 (CoW)                                        | 支持 (CoW)                                             | 维持高效共享                                                           |
| **CPU 开销**          | 相对较高，尤其在复杂场景或低命中率前缀缓存时        | 显著降低，接近零开销                                   | 多进程、Persistent Batch、优化数据结构等                               |
| **执行循环**          | 单进程，API 服务器与引擎耦合较紧                  | 多进程，API 服务器与 EngineCore 解耦，高度并行           | 提升 CPU/GPU 并行度，减少阻塞                                          |
| **调度器**            | 区分 Prefill/Decode                               | 统一处理 Token，字典式调度表示                         | 更简洁、灵活，易于支持高级特性                                         |
| **前缀缓存**          | 默认禁用 (低命中率时有开销)                       | 默认启用 (零开销设计)                                  | 优化后无惧低命中率，默认开启提升易用性                                 |
| **张量并行**          | 不对称架构 (Scheduler+Worker0 同进程)             | 对称架构 (Scheduler 与 Worker 分离)                    | 架构更清晰，IPC 开销通过状态缓存和 Diffs 传输控制                      |
| **多模态支持**        | 基本支持                                          | 增强支持 (非阻塞预处理、图像前缀缓存、编码器缓存等)      | 提升 VLM 性能和易用性                                                  |
| **编译器集成**        | 有限                                              | 集成 `torch.compile`                                   | 自动化模型优化，减少手写 Kernel                                        |
| **Attention Kernel**  | 定制 Kernel (基于 FasterTransformer)              | 集成 FlashAttention 3                                  | 拥抱业界标准，获得更好的性能和特性支持                                 |
| **性能 (vs V0)**      | 基线                                              | 吞吐量提升高达 1.7x (文本), 多模态模型提升更显著             | 全面优化 CPU 开销带来的提升                                            |
| **代码复杂度**        | 随功能增加而提高                                  | 更简洁、模块化                                         | 降低维护成本，方便社区贡献和二次开发                                   |

## 其他推理框架

* [LightLLM](https://github.com/ModelTC/lightllm)：一个基于 Python 的轻量级推理与服务框架，以轻量级设计、可扩展性和高速性能著称，汲取了 vLLM 等其他开源项目的优势。
* [LMDeploy](https://github.com/InternLM/lmdeploy)：用于压缩、部署和服务 LLM 的工具包，内置 TurboMind 推理引擎，强调高请求吞吐量和高效量化。
* [SGLang](https://github.com/sgl-project/sglang)：通过前端语言与后端执行引擎协同设计，高效执行复杂（尤其涉及结构化生成）的 LLM 程序的框架。
* [TGI](https://github.com/huggingface/text-generation-inference)：Hugging Face 的生产级 LLM 服务方案，广泛应用并支持多种硬件后端，借助 vLLM 的 PagedAttention 内核，提供高并发、低延迟的推理服务。
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)：NVIDIA 推出的开源库，用于在 NVIDIA GPU 上优化并加速 LLM 推理，利用 TensorRT 的提前编译和深度硬件优化能力。

## 总结

vLLM 通过其核心技术 PagedAttention，极大地缓解了 LLM 服务中 KV 缓存管理带来的内存瓶颈，显著提高了内存利用率和吞吐量。PagedAttention 借鉴操作系统分页机制，实现了 KV 缓存的非连续存储、动态分配和高效共享（支持并行采样、束搜索、共享前缀等）。

vLLM V1 在 V0 的基础上，对核心架构进行了全面重构和优化，通过多进程架构、灵活调度器、零开销前缀缓存、对称张量并行架构、高效输入准备、`torch.compile` 集成、增强 MLLMs 支持以及 FlashAttention 3 集成等一系列改进，进一步降低了 CPU 开销，提升了系统整体性能、灵活性和可扩展性，为未来快速迭代新功能奠定了坚实基础。


## 参考文献

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

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (May 2025). vLLM：高吞吐、有效内存的LLM服务引擎.
https://syhya.github.io/zh/posts/2025-05-17-vllm

Or

```bibtex
@article{syhya2025vllm,
  title   = "vLLM：高吞吐、有效内存的LLM服务引擎",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "May",
  url     = "https://syhya.github.io/zh/posts/2025-05-17-vllm"
}
```