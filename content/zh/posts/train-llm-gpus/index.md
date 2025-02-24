---
title: "训练大模型的并行技术与内存优化（长期更新中）"
date: 2025-01-05T12:00:00+08:00
lastmod: 2025-02-05T12:00:00+08:00
author: Yue Shui
categories: ["技术博客"]
tags: [LLMs, 并行训练, 内存优化, 数据并行, 模型并行, 流水线并行, 张量并行, 混合专家模型, MoE, ZeRO, 异构系统]
readingTime: 25
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

> **注意**：本文**正在更新**中，目前内容仍是**草稿版本**，后续会不断完善和补充，敬请关注最新版本。

## 大模型的训练挑战

* **参数规模爆炸式增长**  
  为了追求更高的模型容量和性能，神经网络的参数规模不断膨胀：从百万级参数到数十亿、数千亿，乃至数万亿参数的大模型层出不穷。例如，Llama 3.1 405B 拥有约 4,050 亿参数，而据说 GPT-4 可能已经达到了 1.7 万亿（1.7T）级别的参数量。如此庞大的参数规模给模型训练带来了前所未有的计算与内存压力。

* **计算复杂度剧增**  
  参数数量的指数级增长直接导致计算量的巨大提升。训练一次大型模型可能需要数周甚至数月的时间，即使使用大规模的高性能 GPU 集群，训练时间依旧可能难以让人接受。这严重限制了模型的迭代速度和研究效率。

* **内存瓶颈日益凸显**  
  大模型训练过程中，除了大量的模型参数，还需要存储中间激活值、梯度信息以及优化器状态。这些都将占用巨大的 GPU 显存。单卡 GPU（例如 A100 或 H100 的 80GB 显存，甚至 H200 的 141GB 显存, GB200 的 384GB 显存）都难以应对数千亿甚至数万亿规模的模型，“Out of Memory” (OOM) 错误在大模型训练中十分常见。

* **通信开销成为瓶颈**  
  在多 GPU 的分布式环境下，GPU 间需要频繁地交换数据进行同步。随着 GPU 数量和模型规模的增加，通信量显著上升，可能成为限制并行效率的主要瓶颈。比如，为了合并梯度，每次 All-Reduce 操作都要传输海量数据，即使在高带宽网络中也会消耗大量时间。

* **训练稳定性挑战**  
  超大规模模型更容易出现梯度消失或梯度爆炸，训练过程往往不稳定，也更难收敛。混合精度训练虽然能在一定程度上加速训练并节省显存，但也可能带来新的数值稳定性问题，需要更加细致的优化与调试。


## 分布式训练的必要性

面对上述大模型训练难题，分布式训练已成为不可或缺的解决方案。其核心思想是：将训练任务拆分后，使用多台计算设备（通常是 GPU 集群）协同工作，充分利用并行计算能力和聚合的内存资源来完成训练。其优势主要体现在以下几个方面：

* **突破单 GPU 算力限制**  
  单卡 GPU 的计算能力终究有限，无法满足大型模型动辄数万亿参数的计算需求。通过分布式训练可以将训练负载均匀地分配到多个 GPU 上，显著缩短训练时间。例如，数据并行将 mini-batch 中的样本分散到各 GPU 进行并行计算，模型并行则将模型的不同部分分布到多张卡上，从而加速整体训练。

* **克服单 GPU 内存容量瓶颈**  
  大模型的参数量、中间激活值以及优化器状态常常超出单卡显存极限。分布式训练能将这些数据分散存储在多个 GPU 的显存里，等效上“合并”了更多可用显存，使得训练极大规模模型成为可能。典型例子是 ZeRO 技术，它能将参数、梯度和优化器状态都分片到不同 GPU 上存储。

* **加速模型迭代与实验周期**  
  借助分布式训练的高并行度，原本需要数周甚至更久才能完成的训练任务，有机会在数天或更短时间内完成。更快的模型迭代能让研究人员在更短时间内验证新架构、新超参数与新训练策略，从而加速创新与研发。

* **支持更大规模的模型探索**  
  分布式训练为探索规模更大、结构更复杂的神经网络架构提供了可能。像万亿参数级别的 Mixture-of-Experts (MoE) 模型、Switch Transformer 等，都是基于成熟的分布式训练技术才能成功训练并投入应用。

* **提高训练系统的鲁棒性和可扩展性**  
  分布式训练系统具有良好的容错能力：当某个 GPU 节点出现故障时，可切换到其他节点继续工作，减少训练中断的风险。同时，集群规模也可根据需求灵活扩容或缩容，以适应不同规模的模型和训练任务。


## 并行训练技术

下图直观展示了多种并行训练策略的不同之处。不同颜色代表不同的模型层（例如三层），虚线将不同的 GPU 区分开。从左到右分别是数据并行、模型并行（含流水线并行和张量并行）以及专家并行（MoE）。


{{< figure
    src="parallelism_compare.png"
    caption="Fig. 1. An illustration of various parallelism strategies on a three-layer model. Each color refers to one layer and dashed lines separate different GPUs. (Image source: [OpenAI Blog, 2022](https://openai.com/index/techniques-for-training-large-neural-networks/))"
    align="center"
    width="90%"
>}}

- **数据并行 (Data Parallel)**  
  完整模型会被拷贝到每个 GPU 上，数据集则被切分为不同批次分配给各个 GPU 并行计算，最终在参数更新时聚合所有 GPU 的梯度。

- **模型并行 (Model Parallel)**  
  将模型划分到不同的 GPU 上，每个 GPU 只负责模型的一部分计算；可进一步分为以下两类：  
  - **流水线并行 (Pipeline Parallel)**：按层（垂直方向）拆分模型，不同 GPU 负责不同的层，通过微批次（micro-batch）在流水线中传递来并行执行前向和反向计算。  
  - **张量并行 (Tensor Parallel)**：在层内（水平方向）对大规模张量操作（如大矩阵乘法）进行切分，各 GPU 并行完成这部分运算并在必要时进行聚合。

- **专家并行 (Expert Parallel)**  
  通过门控策略，让每个输入样本只经过部分专家（子网络），从而将整个模型按“专家模块”分布到不同 GPU。常见于 Mixture-of-Experts（MOE） 结构，可实现超大参数规模但推理/训练时仅激活部分专家。

- **序列并行 (Sequence Parallel)**  
  将输入序列按 token 维度拆分到不同 GPU 处理，每个 GPU 拥有完整的模型，但只处理序列的一段，需要在注意力计算等阶段进行跨 GPU 的通信和同步。

- **3D 并行 (3D Parallel)**  
  同时结合数据并行、张量并行和流水线并行三种方式，最大化利用多 GPU 的计算与存储资源，既能支持大规模批量大小，又能支持超大模型规模和深度。


下面我们将对这几种并行方式进行详细说明。


### 数据并行 (Data Parallelism)  
  
{{< figure
    src="data_parallelism.png"
    caption="Fig. 2. Data Parallelism. (Image source: [Clolossal-AI Documentation](https://colossalai.org/zh-Hans/docs/concepts/paradigms_of_parallelism/))"
    align="center"
    width="60%"
>}}

在深度学习训练中，数据并行（Data Parallelism, DP）是最常用的并行策略，其核心思路是：  
1. **复制模型参数**：在每个计算设备（通常是 GPU）上都放置一份完整的模型参数。  
2. **划分训练数据**：将大规模的数据集按样本维度拆分为多个子集，不同子集分配给不同的 GPU 进行处理。  
3. **局部前向与反向传播**：每个 GPU 独立计算损失及对应的局部梯度。  
4. **梯度/参数同步**：将各 GPU 的梯度聚合后更新模型参数，保证在每一次迭代后所有 GPU 的模型副本保持一致。  
  
下面展示了 **数据并行** 工作流程：  
  
1. **数据集划分 (Data Partitioning)**    
   将训练数据集 $D$ 划分为 $N$ 个互不重叠的子集 $\{D_1, D_2, \dots, D_N\}$，其中 $N$ 是 GPU 数量。通常会确保各子集大小相近，以实现负载均衡。  
  
2. **模型复制 (Model Replication)**    
   在每个 GPU 上复制一份完整的模型参数 $\theta$。在训练开始时，这些参数在各 GPU 上都是相同的。  
  
3. **数据分发 (Data Distribution)**    
   将子集 $D_i$ 分发给第 $i$ 张 GPU，让其在本地存储并供后续计算使用。  
  
4. **局部前向传播 (Local Forward Propagation)**    
   每个 GPU 基于其本地数据子集 $D_i$ 做前向传播，得到局部损失 $L_i(\theta, D_i)$。  
  
5. **局部反向传播 (Local Backward Propagation)**    
   每个 GPU 基于局部损失 $L_i$ 进行反向传播，计算局部梯度
     
   $$  
     g_i = \nabla_{\theta} L_i(\theta, D_i).  
   $$  
  
6. **梯度同步 (Gradient Synchronization)**    
   各 GPU 之间执行梯度同步（常用 All-Reduce），将所有局部梯度 $\{g_1, g_2, \ldots, g_N\}$ 汇总得到全局平均梯度 

   $$  
     \bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i.  
   $$  
  
7. **参数更新 (Parameter Update)**    
   每个 GPU 使用全局平均梯度 $\bar{g}$ 更新本地模型参数：

   $$  
     \theta \leftarrow \theta - \eta \cdot \bar{g},  
   $$  
   其中 $\eta$ 为学习率 (learning rate)。  
  
8. **迭代循环 (Iteration Loop)**    
   重复步骤 4 - 7，直至模型达到收敛或达到预设的训练轮数（epochs）。  
  
#### 批量同步并行与异步并行
  
在上面的第 6 步“梯度同步”中，如何以及何时进行“同步”是影响数据并行性能和收敛行为的重要因素之一。一般分为以下两大类：  

批量同步并行 (Bulk Synchronous Parallel, BSP) 是数据并行中最常见、也是最易理解的同步模式。其特点可概括为「在每一次小批量（mini-batch）迭代结束后，全局同步一次梯度并更新参数」。具体流程：  
  
1. **局部计算**：各 GPU 基于其数据子集 $D_i$ 分别做前向与反向传播，得到局部梯度 $g_i$。  
2. **全局通信**：所有 GPU 同步（如通过 All-Reduce）计算 $\bar{g}$。  
3. **参数更新**：每个节点均使用 $\bar{g}$ 更新本地参数副本 $\theta$。  
4. **等待与下一步迭代**：所有节点完成上述操作后，再进入下一个迭代。  

异步并行(Asynchronous Parallel, ASP) 旨在摆脱 BSP 的全局同步点，让各节点独立进行计算和参数更新。其典型实现是「参数服务器」(Parameter Server, PS) 架构下的 **异步 push-pull** 过程：  
  
1. 各节点在本地计算得到梯度 $g_i$, 然后 **push** 到参数服务器；    
2. 参数服务器一旦收到梯度，立即更新全局模型参数；    
3. 其他节点在需要最新参数时，会 **pull** 下来继续下一步计算。  
  
#### BSP vs. ASP
  
下表总结了在数据并行环境下，同步并行与异步并行的主要差异：  
  
| **对比维度**           | **同步并行 (BSP)**                                                      | **异步并行 (ASP)**                                                          |  
|:-----------------------|:------------------------------------------------------------------------|:----------------------------------------------------------------------------|  
| **参数更新时机**       | 每个小批量或一定迭代后，全局同步一次                                    | 各节点独立更新参数，无需与他人保持同一时间步                                 |  
| **收敛稳定性**         | **高**。使用的梯度均为最新，收敛路径可控，易于分析                      | **较低**。存在过时梯度，收敛速率与稳定性可能受影响                          |  
| **通信需求**           | 高度依赖 All-Reduce，同步时所有节点都需要等待和交换数据                  | 每个节点向参数服务器异步推送/拉取，通信更为灵活，但参数服务器可能成为瓶颈     |  
| **硬件资源利用**       | 若有慢节点或网络延迟，则其他节点需等待，资源利用率可能降低               | 无需等待慢节点，可高效使用计算资源                                          |  
| **实现复杂度**         | 相对较低，主流框架（PyTorch DDP、Horovod 等）有内置支持                  | 相对更高，需要参数服务器等组件，需处理更多的同步逻辑与数据一致性             |  
| **适用场景**           | 同构硬件、网络带宽良好、追求较高收敛质量                                 | 异构硬件、网络不稳定或带宽较低、需要极高的吞吐量且能容忍一定收敛风险         |  
| **典型实现**           | PyTorch DDP、TensorFlow MirroredStrategy                                 | Parameter Server 架构（MXNet、TensorFlow ParameterServer 模式等）            |  
  
> **建议**：在实际项目中，先从简单的同步并行 (BSP) 入手，利用 PyTorch DDP 或类似工具进行多 GPU 训练。若网络环境异构、节点繁多或任务对吞吐率要求极高，可再尝试异步并行 (ASP) 或参数服务器方案，并配合梯度累积 (Gradient Accumulation) 来平衡带宽与更新频率。  
  
#### 梯度累积  
  
当批量大小较大或通信成为主要瓶颈时，可以采用 **梯度累积(Gradient Accumulation)** 来减少同步频率。其核心思路是：    
- 连续计算多个小批量（mini-batch）的局部梯度，并将它们累加到本地的累积缓冲区中；    
- 当累积的 mini-batch 数量达到 $K$ 时，再触发一次全局梯度同步与参数更新。  
  
设第 $j$ 个 mini-batch 的梯度为 $g_j$，则在一个「累积周期」内得到

$$  
  G = \sum_{j=1}^{K} g_j.  
$$  

再用学习率 $\eta$ 更新：

$$  
  \theta \leftarrow \theta - \eta \cdot G.  
$$  
  
由于梯度同步不再是每个 mini-batch 都进行，而是每累计 $K$ 个 mini-batch 执行一次，通信开销可显著降低。但参数更新频率降低也可能导致训练收敛速度放缓，需在吞吐量与收敛性能之间做权衡。  
  
#### 分布式数据并行
  
分布式数据并行 (Distributed Data Parallel, DDP) 是 PyTorch v1.5 ([Li et al. 2020](https://arxiv.org/pdf/2006.15704))在 BSP 思想下的高度优化实现，为单机多 GPU 乃至多机多 GPU 的数据并行提供便利。其主要优化包括：  
  
1. **梯度 Bucketing（梯度桶化）**：将模型参数分为多个「桶」(bucket)；反向传播时一旦某个桶内所有梯度都已计算完，就立即启动一次针对**该桶的 All-Reduce**，而不是等到所有梯度都算完后再一次性同步。    
2. **通信与计算重叠**：DDP 通过异步通信和非阻塞操作，尽可能地将梯度同步（通信）与前向传播、反向传播（计算）重叠，从而减少了通信开销。这种重叠策略提升了整体的并行效率。  
3. **梯度累积**：DDP 也能方便地与**梯度累积**相结合，结合使用，通过增加每次同步的梯度更新间隔，从而减少同步频率。这在大规模分布式训练中有助于进一步降低通信开销，提高训练效率。

{{< figure
    src="pytorch_ddp.png"
    caption="Fig. 3. Pseudo code for Pytorch DDP. (Image source: [Li et al. 2020](https://arxiv.org/pdf/2006.15704))"
    align="center"
    width="80%"
>}}

#### Ring All-Reduce  
  
在多 GPU（尤其是单机多 GPU）环境下，若有高速互联（如 NVLink、PCIe 交换机等），可使用 **Ring All-Reduce** 来显著降低通信开销。其思路是：  
  
1. 将 $k$ 个节点组织成一个环，并把梯度向量等分成 $k$ 份。    
2. 在「加和阶段」，每个节点分别向下一个节点发送其本地的一部分梯度，并与收到的梯度相加；该过程循环若干次后，每个节点会持有完整的「聚合后」梯度。    
3. 在「广播阶段」，再将最终结果沿环路分发给所有节点。  
  
理想情况下，Ring All-Reduce 的通信代价与节点数量近似无关（可以视为 $\mathcal{O}(1)$），非常适合多 GPU 环境下的梯度同步，是 Horovod、NCCL 等库中广泛使用的核心通信模式。  
  
#### 参数服务器
  
当集群规模扩展至多机多 GPU 时，若简单地采用单点聚合（例如一台中心服务器）往往难以支撑海量数据的并行训练。参数服务器(Parameter Server, PS) ([Li, et al., 2014](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf))是为可扩展分布式训练而设计的一种典型架构：  
  
1. **参数分片**：将模型参数按键值对 (key-value) 的形式进行拆分，不同 PS 节点只管理特定分片的参数；    
2. **push-pull** 语义：计算节点在本地得到梯度后，**push** 到相应的 PS；PS 更新完该分片参数后，计算节点可在需要时 **pull** 下最新版本进行下一步计算。    
3. **灵活容错与扩展**：通过增加或移除 PS 节点，可在带宽或计算需求上灵活扩容；在 PS 上也能实现备份与容错策略。  
  
这种 **PS + Worker** 模式可以**同时**结合数据并行和模型并行，将超大模型拆分到多个 PS 上存储，并对超大数据进行分布式训练。PS 本身也能根据负载情况做拆分与合并，形成更加复杂的层次化拓扑结构。  


### 模型并行 (Model Parallelism)

模型并行 (Model Parallelism) 是一种将模型本身分割到多个计算设备 (GPU) 上进行训练的并行方式。当模型参数规模超过单个 GPU 的内存容量时，模型并行成为必要的选择。模型并行主要分为两种类型：流水线并行 (Pipeline Parallelism) 和张量并行 (Tensor Parallelism)。

**朴素模型并行与气泡问题**

{{< figure
    src="naive_mp.png"
    caption="Fig. 4. A naive model parallelism setup where the model is vertically split into 4 partitions. Data is processed by one worker at a time due to sequential dependency, leading to large “bubbles” of idle time. (Image source: [Huang et al. 2018](https://arxiv.org/abs/1811.06965))"
    align="center"
    width="100%"
>}}

朴素的模型并行实现，即将模型简单地按层划分，并顺序地在不同 GPU 上执行，会遇到严重的 "气泡" (bubble) 问题。由于层之间的依赖关系，当一个 GPU 在处理某个数据样本的某个阶段时，其他 GPU 可能处于空闲状态，等待前一个 GPU 的输出或者后一个 GPU 的输入。这种 GPU 空闲时间被称为 "气泡"，严重降低了流水线并行的效率。

其中，$F_i$ 表示 Stage i 的前向传播，$B_i$  表示 Stage i 的反向传播。可以看到，在朴素流水线并行中，大部分时间只有一个 GPU 在工作，其他 GPU 处于空闲状态，效率低下。

**气泡问题产生的原因:**

* **层间依赖:**  神经网络的层之间存在顺序依赖关系，后一层的计算必须依赖于前一层的输出。
* **顺序执行:**  朴素模型并行按照层顺序依次执行，导致 GPU 之间无法充分并行工作。


### 流水线并行 (Pipeline Parallelism)

{{< figure
    src="pipeline_parallelism.png"
    caption="Fig. 5. Pipeline Parallelism. (Image source: [Clolossal-AI Documentation](https://colossalai.org/zh-Hans/docs/concepts/paradigms_of_parallelism/))"
    align="center"
    width="60%"
>}}


流水线并行 (Pipeline Parallelism) 将模型按层划分为多个阶段 (stage)，每个阶段分配到一个 GPU 上。数据像流水线一样在不同 GPU 之间传递，前一个 GPU 的输出作为后一个 GPU 的输入。流水线并行旨在提高模型并行训练的效率，减少 GPU 空闲时间。

#### GPipe

GPipe ([Huang et al. 2018](https://arxiv.org/abs/1811.06965)) 是 Google 提出的一个高效的流水线并行训练系统，旨在解决朴素流水线并行的气泡问题。GPipe 的核心思想是将 **mini-batch** 划分为多个 **micro-batch**，并采用**同步梯度聚合**的方式来缓解气泡问题，提高流水线效率。

{{< figure
    src="gpipe.png"
    caption="Fig. 6. Illustration of pipeline parallelism in GPipe with 4 microbatches and 4 partitions. GPipe aggregates and updates gradients across devices synchronously at the end of every batch. (Image source: [Huang et al. 2018](https://arxiv.org/abs/1811.06965))"
    align="center"
    width="100%"
>}}

以下是 GPipe 调度策略：
1. **Micro-batch 划分:**  将一个 mini-batch 划分为 $m$ 个 micro-batch。划分后的每个 micro-batch 的大小为原 mini-batch 的 $1/m$。
2. **流水线阶段划分:**  将模型按层划分为 $d$ 个阶段，每个阶段分配到一个 GPU 上。
3. **流水线执行:**  依次处理每个 micro-batch，在流水线中进行前向和反向传播。具体流程如下：
    * **前向传播 (Forward Propagation):**  对于每个 micro-batch，依次在 Stage 1, Stage 2, ..., Stage d 上进行前向传播。Stage i 的输出作为 Stage i+1 的输入。
    * **反向传播 (Backward Propagation):**  当所有 micro-batch 的前向传播都完成后，开始反向传播。对于每个 micro-batch，依次在 Stage d, Stage d-1, ..., Stage 1 上进行反向传播。Stage i 的梯度作为 Stage i-1 的输入。
4. **同步梯度聚合 (Synchronous Gradient Aggregation):**  在所有 micro-batch 的反向传播都完成后，将所有 micro-batch 的梯度进行聚合 (例如求平均)，得到全局平均梯度。
5. **参数更新 (Parameter Update):**  每个 GPU 使用全局平均梯度更新本地模型参数。


#### GPipe 气泡比例公式
假设每个 micro-batch 的前向和反向传播时间均为 1 单位，流水线深度为 $d$，micro-batch 数量为 $m$，则 GPipe 的气泡比例为：

$$
\text{Bubble Ratio} = 1 - \frac{2md}{(2m + 2(d-1))d} = \frac{d-1}{m+d-1}
$$

当 micro-batch 数量 $m$ 远大于流水线深度 $d$ 时 ($m \gg d$)，气泡比例趋近于 0，流水线效率接近线性加速。GPipe 论文中指出，当 $m > 4d$ 时，气泡开销几乎可以忽略不计 (在激活重计算的情况下)。


#### GPipe 优缺点
**优点:**

* **减少气泡:**  GPipe 通过 micro-batch 划分和流水线调度，显著减少了朴素流水线并行的气泡问题，提高了 GPU 利用率和训练效率。
* **同步梯度聚合:**  GPipe 采用同步梯度聚合，保证了训练过程的同步性，模型收敛性较好。
* **线性加速潜力:**  在 micro-batch 数量足够大的情况下，GPipe 可以实现接近线性的加速效果。

**缺点:**

* **仍然存在气泡:**  即使使用 micro-batch 划分，GPipe 仍然存在一定的气泡开销，尤其是在流水线启动和结束阶段。
* **内存占用增加:**  为了支持流水线执行，GPipe 需要存储每个 micro-batch 的中间激活值，内存占用相比数据并行有所增加。
* **延迟参数更新:**  GPipe 在所有 micro-batch 的反向传播完成后才进行参数更新，可能导致参数更新的延迟，影响模型收敛速度。

#### PipeDream

{{< figure
    src="pipe_dream.png"
    caption="Fig. 7. Illustration of 1F1B microbatch scheduling in PipeDream. (Image source: [Harlap et al. 2018](https://arxiv.org/abs/1806.03377))"
    align="center"
    width="100%"
>}}

PipeDream ([Harlap et al. 2018](https://arxiv.org/abs/1806.03377))是另一种高效的流水线并行训练系统，它采用了 1F1B (1-Forward-1-Backward) 调度策略，并引入了权重暂存 (Weight Stashing) 技术，进一步减少气泡，提高流水线效率，并解决 1F1B 调度可能导致的权重版本不一致问题。

PipeDream 的 1F1B 调度策略的核心思想是，每个 GPU (Stage) 交替执行前向传播和反向传播，尽可能地并行工作，减少 GPU 空闲时间。具体流程如下：

1. **Micro-batch 划分:**  将一个 mini-batch 划分为 $m$ 个 micro-batch。
2. **流水线阶段划分:**  将模型按层划分为 $d$ 个阶段，每个阶段分配到一个 GPU 上。
3. **1F1B 调度执行:**  每个 GPU 轮流执行前向传播和反向传播。例如，对于 GPU 1 (Stage 1)：
    * 处理 micro-batch 1 的前向传播 (F1_1)。
    * 处理 micro-batch 1 的反向传播 (B1_1)。
    * 处理 micro-batch 2 的前向传播 (F1_2)。
    * 处理 micro-batch 2 的反向传播 (B1_2)。
    * ...
    * 依此类推。
    其他 GPU (Stage 2, Stage 3, ...) 也采用类似的 1F1B 调度策略，但处理的 micro-batch 和阶段不同，形成流水线效果。


#### 权重暂存

由于 1F1B 调度中，前向传播和反向传播可能使用不同版本的模型权重，会导致权重版本不一致问题，影响训练的正确性和收敛性。PipeDream 引入了权重暂存 (Weight Stashing)技术来解决这个问题。权重暂存的核心思想是，每个 GPU 维护多个版本的模型权重，并确保前向传播和反向传播使用同一版本的权重。

**权重暂存实现方式:**

* **版本管理:**  每个 GPU 维护一个权重版本队列，存储多个版本的模型权重。
* **版本选择:**  在进行前向传播时，选择当前最新的权重版本。在进行反向传播时，选择与对应前向传播相同的权重版本。
* **版本更新:**  在完成一个 mini-batch 的所有 micro-batch 的反向传播后，更新模型权重，并生成新的权重版本。


> 为了进一步优化 PipeDream 的内存使用，尤其是在权重暂存方面，PipeDream 衍生出了 PipeDream-flush 和 PipeDream-2BW 两种内存优化变体。

#### PipeDream-flush

{{< figure
    src="pipe_dream_flush.png"
    caption="Fig. 8. Illustration of pipeline scheduling in PipeDream-flush. (Image source: [Narayanan et al. 2020](https://arxiv.org/abs/2006.09503))"
    align="center"
    width="100%"
>}}

 PipeDream-flush 在 PipeDream 的基础上，周期性地进行全局同步的流水线刷新 (flush)，类似于 GPipe 的同步梯度聚合。通过定期刷新，PipeDream-flush 可以大幅减少权重暂存所需的内存空间，只需维护单个版本的模型权重，但会牺牲少量吞吐量。


#### PipeDream-2BW

PipeDream-2BW (Double-Buffered Weights) 维护两个版本的模型权重，即 "双缓冲权重"。它每 $k$ 个 micro-batch 更新一次模型版本，其中 $k$ 大于流水线深度 $d$ ($k > d$). 新更新的模型版本不会立即完全替换旧版本，因为可能还有一些剩余的反向传播操作仍然依赖于旧版本。通过双缓冲权重，PipeDream-2BW 可以将权重暂存的内存开销降低到只维护两个版本的模型权重，显著减少内存占用。

{{< figure
    src="pipe_dream_2bw.png"
    caption="Fig. 9. Illustration of pipeline scheduling in PipeDream-2BW. (Image source: [Narayanan et al. 2020](https://arxiv.org/abs/2006.09503))"
    align="center"
    width="100%"
>}}  

#### PipeDream 优缺点

**优点:**
* **更低的气泡开销:**  1F1B 调度策略相比 GPipe 可以进一步减少气泡，提高 GPU 利用率和训练效率。
* **权重暂存解决版本一致性:**  权重暂存技术保证了前向传播和反向传播使用同一版本的权重，解决了 1F1B 调度可能导致的权重版本不一致问题。
* **内存优化变体:**  PipeDream-flush 和 PipeDream-2BW 进一步优化了内存使用，降低了权重暂存的内存开销，使得流水线并行更适用于内存受限的场景。

**缺点:**

* **实现复杂度较高:**  PipeDream 的 1F1B 调度和权重暂存机制相比 GPipe 更为复杂，实现难度较高。
* **权重暂存内存开销:**  即使是 PipeDream-2BW，仍然需要维护两个版本的模型权重，内存开销仍然比数据并行要高。
* **异步性引入学习效率问题:**  PipeDream 的异步更新方式可能引入学习效率问题，需要仔细调整超参数和训练策略。

### 张量并行 (Tensor Parallelism)

张量并行 (Tensor Parallelism, TP) 是一种将模型中的张量 (通常是权重矩阵) 沿着特定维度切分，并将切分后的分片分配到不同的 GPU 上进行计算的并行方式。

#### Megatron-LM
Megatron-LM ([Shoeybi et al. 2019](https://arxiv.org/pdf/1909.08053)) 是 NVIDIA 提出的一个用于训练超大型语言模型的系统，它采用了张量并行技术，对 Transformer 模型层内部的矩阵乘法操作进行并行化，包括 **self-attention** 和 **MLP** 中的矩阵乘法。

{{< figure
    src="Megatron-LM.png"
    caption="Fig. 10. Illustration of tensor parallelism for key transformer components proposed in Megatron-LM. (Image source: [Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053))"
    align="center"
    width="100%"
>}}  

Transformer 的 MLP 层通常包含两个线性层，第一个线性层的计算可以表示为 $Y = \text{GeLU}(XA)$，其中 $X$ 是输入矩阵，$A$ 是权重矩阵，GeLU 是激活函数。Megatron-LM 将权重矩阵 $A$ 沿着列维度切分为 $P$ 个分片 $[A_1, A_2, ..., A_P]$，其中 $P$ 是 GPU 的数量。每个 GPU $i$ 负责存储和计算权重分片 $A_i$。

**MLP 层张量并行计算流程:**

$$
\begin{aligned}
\text { Split } A & =\left[A_1, A_2\right] \\
Y & =\operatorname{GeLU}(X A) \\
{\left[Y_1, Y_2\right] } & =\left[\operatorname{GeLU}\left(X A_1\right), \operatorname{GeLU}\left(X A_2\right)\right]
\end{aligned}
$$

1. **权重分片:**  将权重矩阵 $A$ 沿着列维度切分为 $P$ 个分片 $[A_1, A_2, ..., A_P]$，并将分片 $A_i$ 分配到 GPU $i$。
2. **局部矩阵乘法:**  每个 GPU $i$ 使用输入矩阵 $X$ 和权重分片 $A_i$ 进行矩阵乘法计算，得到局部输出 $Y_i = \text{GeLU}(XA_i)$。
3. **全局拼接 (All-Gather):**  所有 GPU 通过 All-Gather 操作，将局部输出 $\{Y_1, Y_2, ..., Y_P\}$ 拼接成完整的输出矩阵 $Y = [Y_1, Y_2, ..., Y_P]$。


**自注意力层张量并行**

Megatron-LM 也对 Transformer 的自注意力层中的 Query ($Q$), Key ($K$), Value ($V$) 权重矩阵进行张量并行切分，并进行相应的局部矩阵乘法和全局拼接操作，实现自注意力层的张量并行化。。自注意力层的计算公式为：

$$
\text{Attention}(X, Q, K, V) = \text{softmax}\left(\frac{(XQ)(XK)^T}{\sqrt{d_k}}\right)XV
$$


#### PTD-P

PTD-P (Pipeline, Tensor, and Data Parallelism) 是一个结合了流水线并行、张量并行和数据并行的多维并行策略。PTD-P 旨在充分利用各种并行技术的优势，提高超大型模型训练的效率和可扩展性。

**PTD-P 的特点:**

* **多维并行结合:**  PTD-P 同时使用了流水线并行、张量并行和数据并行三种并行技术，可以从多个维度对训练过程进行并行化。
* **Interleaved 1F1B 调度:**  PTD-P 采用了 interleaved 1F1B 调度策略，与传统的流水线并行不同，它将模型划分为多个不连续的层块 (model chunk)，并将多个层块分配给每个 GPU。这种调度策略可以进一步减少气泡，提高流水线效率。
* **灵活的并行配置:**  PTD-P 允许用户根据模型结构和硬件资源灵活配置各种并行技术的组合方式，例如可以只使用张量并行和数据并行，也可以同时使用流水线并行、张量并行和数据并行。

传统的流水线并行通常将模型划分为连续的层块，每个 GPU 负责一个连续的层块。PTD-P 的 interleaved 1F1B 调度则将模型划分为多个不连续的层块，例如，GPU 1 负责层 1, 2, 9, 10，GPU 2 负责层 3, 4, 11, 12，依此类推。每个 GPU 负责多个不连续的层块，可以更有效地利用 GPU 资源，减少气泡开销。

{{< figure
    src="PTD-P.png"
    caption="Fig. 11. (Top) Default 1F1B pipeline schedule as in PipeDream-flush. (Bottom) Interleaved 1F1B pipeline schedule. First model chunks are in dark colors and second chunks are in light colors (Image source: [Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053))"
    align="center"
    width="100%"
>}}  

#### 张量并行优缺点

**优点:**

* **突破单 GPU 显存限制:**  张量并行可以将模型参数分散存储在多个 GPU 上，突破单 GPU 显存容量限制，支持训练更大规模的模型。
* **层内并行:**  张量并行可以实现模型层内部的并行化，例如矩阵乘法操作的并行计算，提高计算效率。
* **与数据并行和流水线并行结合:**  张量并行可以与数据并行和流水线并行等其他并行技术结合使用，形成多维并行策略，进一步提高训练效率和可扩展性。

**缺点:**

* **通信开销:**  张量并行需要进行额外的通信操作 (例如 All-Gather, Reduce-Scatter) 来聚合计算结果，通信开销可能较高，尤其是在 GPU 数量较多时。
* **实现复杂度较高:**  张量并行的实现较为复杂，需要仔细设计张量切分策略和通信操作，以保证计算的正确性和效率。
* **适用性有限:**  张量并行主要适用于具有矩阵乘法等张量运算的模型层，例如 Transformer 模型的自注意力层和 MLP 层，对于其他类型的模型层，张量并行的适用性可能有限。

### 混合专家模型

混合专家模型 (Mixture-of-Experts, MoE) ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) 是一种稀疏激活模型，它包含多个 "专家" (expert) 网络，并使用一个门控网络 (gating network) 来决定每个输入样本应该由哪些专家处理。MoE 可以在不显著增加计算成本的情况下，大幅增加模型参数量，从而提升模型容量和性能。

**MoE 层的基本原理与门控机制**

MoE 层的核心思想来源于[集成学习（Ensemble learning）](https://en.wikipedia.org/wiki/Ensemble_learning)。它将一个复杂的任务分解为多个子任务，每个子任务由一个专门的 "专家" 网络负责处理。对于每个输入样本，门控网络会根据输入特征选择激活一部分专家网络，而不是激活整个模型。这样可以实现模型的稀疏激活，降低计算成本，同时增加模型参数量。

**MoE 层结构:**

一个典型的 MoE 层包含以下组件：

* **专家网络 (Experts):**  一组独立的神经网络 $\{E_1, E_2, ..., E_n\}$，每个专家网络 $E_i$ 可以是任意类型的神经网络，例如前馈神经网络 (FFN), CNN, RNN 等。专家网络的数量 $n$ 可以很大，例如几十个、几百个甚至几千个。
* **门控网络 (Gating Network):**  一个可训练的神经网络 $G$，用于根据输入样本 $x$ 学习一个概率分布，决定激活哪些专家。门控网络的输入是输入样本 $x$，输出是一个 $n$ 维的概率向量 $p = G(x) = [p_1, p_2, ..., p_n]$，其中 $p_i$ 表示激活专家 $E_i$ 的概率。
* **专家输出聚合 (Expert Output Aggregation):**  根据门控网络的输出概率分布，将激活的专家网络的输出进行加权求和，得到 MoE 层的最终输出 $y$。

**门控机制详解 (Noisy Top-k Gating):**

一种常用的门控机制是 Noisy Top-k Gating，其计算过程如下：

1. **门控分数计算:**  对于输入样本 $x$，门控网络计算每个专家的门控分数 $H^{(i)}(x)$：

   $$
   H^{(i)}(x) = (xW_g)^{(i)} + \epsilon \cdot \text{softplus}((xW_{\text{noise}})^{(i)}), \quad \epsilon \sim \mathcal{N}(0, \mathbf{1})
   $$

   其中，$W_g \in \mathbb{R}^{d \times n}$ 和 $W_{\text{noise}} \in \mathbb{R}^{d \times n}$ 是可训练的权重矩阵，$d$ 是输入特征维度，$n$ 是专家数量，$\epsilon \sim \mathcal{N}(0, \mathbf{1})$ 是高斯噪声，$\text{softplus}(x) = \log(1 + e^x)$ 是 softplus 函数。添加噪声的目的是为了提高负载均衡，避免门控网络总是选择相同的专家。

2. **Top-k 选择:**  选择门控分数 $H(x) = [H^{(1)}(x), H^{(2)}(x), ..., H^{(n)}(x)]$ 中值最大的前 $k$ 个专家。$\text{topk}(v, k)$ 函数将向量 $v$ 中值最大的前 $k$ 个元素保留，并将其他元素设置为 $-\infty$。

   $$
   \text{topk}^{(i)}(v, k) = \begin{cases} v^{(i)} & \text{if } v^{(i)} \text{ is in the top } k \text{ elements of } v \\ -\infty & \text{otherwise} \end{cases}
   $$

3. **Softmax 归一化:**  对 top-k 个专家的门控分数进行 softmax 归一化，得到门控概率分布 $G(x) = [G^{(1)}(x), G^{(2)}(x), ..., G^{(n)}(x)]$：

   $$
   G(x) = \text{softmax}(\text{topk}(H(x), k))
   $$

   门控概率分布 $G(x)$ 是一个稀疏向量，只有 top-k 个专家的概率值非零，其他专家的概率值为 0。

4. **专家输出聚合:**  将激活的 top-k 个专家网络的输出进行加权求和，得到 MoE 层的最终输出 $y$：

   $$
   y = \sum_{i=1}^{n} G^{(i)}(x) E_i(x)
   $$

**辅助损失 (Auxiliary Loss) (MoE):**

为了避免门控网络总是偏向于少数几个专家，导致专家负载不均衡，MoE 通常会引入一个辅助损失函数，鼓励所有专家被均衡地使用。一种常用的辅助损失是专家使用率的变异系数平方：

$$
\mathcal{L}_{\text{aux}} = w_{\text{aux}} \cdot \text{CV}\left(\sum_{x \in X} G(x)\right)^2
$$

其中，CV 是变异系数 (Coefficient of Variation)，用于衡量专家使用率的离散程度。$\sum_{x \in X} G(x)$ 表示在 mini-batch $X$ 中，每个专家的被激活次数之和。辅助损失 $\mathcal{L}_{\text{aux}}$ 旨在最小化专家使用率的变异系数，使得所有专家的使用率尽可能接近，实现负载均衡。$w_{\text{aux}}$ 是辅助损失的权重，需要根据具体任务进行调整。

**GShard：分片 MoE Transformer**

GShard 是 Google 提出的一个用于训练大规模 MoE Transformer 模型的系统。GShard 的核心思想是将 MoE Transformer 模型中的 MoE 层分片到多个 TPU 设备上，而 Transformer 模型的其他层 (例如自注意力层、LayerNorm 层) 则在所有设备上复制。这样可以充分利用 TPU 集群的并行计算能力和内存资源，训练参数量巨大的 MoE Transformer 模型。

**GShard 的 MoE Transformer 结构:**

GShard 的 MoE Transformer 模型将 Transformer 模型中的每隔一个前馈网络 (FFN) 层替换为 MoE 层。MoE 层包含多个专家网络，门控网络决定每个输入 token 路由到哪些专家网络进行处理。

**GShard 的分片策略:**

GShard 主要对 MoE 层进行分片，将 MoE 层中的专家网络 $\{E_1, E_2, ..., E_n\}$ 分散到多个 TPU 设备上。例如，如果有 $P$ 个 TPU 设备，可以将专家网络划分为 $P$ 组，每组专家网络分配到一个 TPU 设备上。Transformer 模型的其他层 (例如自注意力层、LayerNorm 层) 则在所有 TPU 设备上复制。

**GShard 的改进门控机制:**

GShard 在 Noisy Top-k Gating 的基础上，进行了一些改进，以提高门控机制的性能和稳定性：

* **专家容量 (Expert Capacity):**  为了避免专家过载，GShard 引入了专家容量限制。每个专家网络都有一个容量上限，表示它最多可以处理的 token 数量。如果一个 token 被路由到一个已经达到容量上限的专家网络，则该 token 会被标记为 "overflowed"，门控输出会被设置为零向量，表示该 token 不会被路由到任何专家网络。
* **局部组分发 (Local Group Dispatching):**  为了提高门控效率，GShard 将 token 分组，在组级别强制执行专家容量限制。例如，将 mini-batch 中的 token 划分为多个局部组，每个局部组包含一定数量的 token。门控网络为每个局部组选择 top-k 个专家网络，并确保每个专家网络在一个局部组内处理的 token 数量不超过其容量上限。
* **辅助损失:**  GShard 也使用了辅助损失函数来平衡专家负载。与原始 MoE 模型的辅助损失不同，GShard 的辅助损失旨在最小化每个专家网络路由到的数据比例的均方误差，更加直接地衡量专家负载均衡程度。
* **随机路由 (Random Routing):**  为了增加路由的随机性，GShard 在选择 top-k 个专家网络时，引入了随机路由机制。除了选择最佳的 top-k 个专家网络外，GShard 还会以一定的概率随机选择次优的专家网络，增加专家网络的多样性，提高模型的泛化能力。

**Switch Transformer：万亿参数模型与稀疏性**

Switch Transformer 是 Google 提出的一个参数量达到万亿级别的 MoE 模型。Switch Transformer 的核心创新是将 Transformer 模型中的密集前馈网络 (FFN) 层替换为稀疏的 Switch FFN 层。与 GShard 的 Top-2 Gating 不同，Switch Transformer 每个输入 token 只路由到一个专家网络，具有更高的稀疏性，进一步降低了计算成本，使得训练万亿参数模型成为可能。

**Switch FFN 层:**

Switch Transformer 将 Transformer 模型中的密集 FFN 层替换为稀疏的 Switch FFN 层。Switch FFN 层使用一个门控网络 (Switch Router)，为每个输入 token 选择一个最佳专家网络。

**Switch Router 机制:**

1. **路由预测:**  对于输入 token $x$，Switch Router 预测每个专家网络的路由概率 $p_i = G^{(i)}(x)$，其中 $i = 1, 2, ..., n$，n 是专家网络数量。
2. **专家选择:**  选择路由概率最高的专家网络作为最佳专家网络。Switch Transformer 采用 Top-1 路由策略，即每个 token 只路由到路由概率最高的专家网络。
3. **token 路由:**  将输入 token $x$ 路由到选择的最佳专家网络进行处理。

**Switch Transformer 的训练稳定性优化:**

为了提高 Switch Transformer 的训练稳定性，Switch Transformer 论文中提出了一些优化策略：

* **选择性精度 (Selective Precision):**  Switch Transformer 发现，在路由函数内部使用 FP32 精度可以提高训练稳定性，同时避免 FP32 张量的通信开销。Switch Router 的计算过程使用 FP32 精度，计算结果再转换为 FP16 精度。
* **更小的初始化 (Smaller Initialization):**  Switch Transformer 建议使用更小的权重初始化尺度，例如将 Transformer 初始化尺度参数 $s$ 从 1 降低到 0.1。更小的初始化尺度可以减缓训练初期梯度爆炸的风险，提高训练稳定性。
* **更高的专家 Dropout (Higher Expert Dropout):**  Switch Transformer 发现，在专家 FFN 层中使用更高的 dropout 率 (例如 0.4)，而非专家层使用较低的 dropout 率 (例如 0.1)，可以有效防止过拟合，提高模型的泛化能力。

**专家选择路由 (Expert Choice Routing, EC)**

专家选择路由 (Expert Choice Routing, EC) 是一种与 token 选择路由 (GShard top-2, Switch Transformer top-1) 相反的路由策略。在 token 选择路由中，每个 token 选择 top-k 个专家网络进行路由。在专家选择路由中，每个专家网络选择 top-k 个 token 进行处理。EC 旨在解决 token 选择路由可能导致的负载不均衡和 token 浪费问题。

**EC 的优势:**

* **完美负载均衡:**  EC 保证每个专家网络都处理固定数量的 token (top-k 个)，避免专家网络过载和 token 浪费，实现完美负载均衡。
* **更高的训练效率:**  实验表明，EC 可以将训练收敛速度提高 2 倍，相比 token 选择路由具有更高的训练效率。

**EC 的计算过程:**

1. **计算 token-to-expert 亲和度分数:**  对于输入矩阵 $X \in \mathbb{R}^{n \times d}$，计算 token-to-expert 亲和度分数矩阵 $S \in \mathbb{R}^{n \times e}$：

   $$
   S = \text{softmax}(X \cdot W_g), \quad \text{where } W_g \in \mathbb{R}^{d \times e}
   $$

   其中，$W_g$ 是门控权重矩阵，$e$ 是专家网络数量。

2. **专家选择 token:**  每个专家网络选择 top-k 个 token 进行处理。通过 $\text{top-k}(S^T, k)$ 函数，得到门控矩阵 $G \in \mathbb{R}^{e \times k}$ 和 token 索引矩阵 $I \in \mathbb{R}^{e \times k}$。$G[i, j]$ 表示专家网络 $i$ 选择的第 $j$ 个 token 的路由权重，$I[i, j]$ 表示专家网络 $i$ 选择的第 $j$ 个 token 的索引。

   $$
   G, I = \text{top-k}(S^T, k)
   $$

3. **One-hot 编码:**  将 token 索引矩阵 $I$ 转换为 one-hot 矩阵 $P \in \mathbb{R}^{e \times k \times n}$。

   $$
   P = \text{one-hot}(I)
   $$

4. **Gated FFN 层输入:**  专家网络 $i$ 的 gated FFN 层的输入为 $(P \cdot X) \in \mathbb{R}^{e \times k \times d}$。

**EC 的正则化:**

EC 可以通过正则化限制每个 token 最多被路由到的专家网络数量，以控制模型的稀疏性。

**EC 的局限性:**

EC 的一个主要局限性是，它不适用于小 batch size 的场景，也不适用于自回归文本生成任务，因为它需要知道未来的 token 才能进行 top-k 选择


### 序列并行 (Sequence Parallelism)

序列并行 (Sequence Parallelism, SP) 是一种专门针对序列数据 (例如文本、时间序列) 的并行策略。它沿着序列维度对输入样本进行切分，并将切分后的序列分片分配到不同的计算设备 (GPU) 上进行处理。序列并行特别适用于训练长序列模型，例如 Transformer 模型处理长文本序列时，可以有效降低内存占用，提高训练效率。

**Megatron 序列并行 (Megatron SP)**

Megatron SP 是 Megatron-LM 系统中实现的一种序列并行方法。它是在张量并行的基础上构建的，旨在进一步降低训练 Transformer 模型时的内存消耗，尤其是在处理长序列时。Megatron SP 的核心思想是在模型并行的基础上，对序列维度进行切分，以减少激活值的内存占用。

**Megatron SP 的适用场景:**

Megatron SP 主要适用于以下场景：

* **模型并行已使用:** Megatron SP 通常与张量并行结合使用，作为模型并行的补充。它假设模型已经通过张量并行进行了层内并行化。
* **长序列输入:** Megatron SP 针对长序列输入进行了优化，可以有效降低处理长序列时的内存占用。
* **Transformer 模型:** Megatron SP 主要应用于 Transformer 模型，特别是处理长文本序列的 Transformer 模型。

**Megatron SP 的工作原理:**

1. **模型并行基础:** Megatron SP 建立在模型并行的基础上。假设模型已经通过张量并行等技术进行了层内并行化，模型参数分布在多个 GPU 上。
2. **序列维度切分:** 对于输入样本，Megatron SP 沿着序列维度将其切分为多个分片。例如，如果序列长度为 $L$，GPU 数量为 $P$，则每个 GPU 负责处理长度为 $L/P$ 的序列分片。
3. **局部计算:** 每个 GPU 使用分配到的序列分片进行局部计算。在 Transformer 模型中，对于线性运算部分 (例如 Attention 和 MLP 层)，仍然可以使用张量并行策略进行加速。对于非线性运算部分 (例如 LayerNorm, Dropout)，由于无法直接使用张量并行，Megatron SP 在序列维度上进行计算。
4. **激活值汇总:** 在计算完非线性运算部分后，需要将每个 GPU 上计算得到的激活值在序列维度上进行汇总 (例如 All-Gather 操作)，以保证后续线性运算的输入是完整的序列信息。
5. **线性运算 (张量并行):** 对于 Attention 和 MLP 等线性运算部分，仍然使用张量并行策略进行计算，利用模型并行的优势。
6. **梯度同步和参数更新:**  梯度同步和参数更新过程与模型并行类似，需要进行跨 GPU 的梯度聚合和参数更新。

**Megatron SP 的优势:**

* **减少激活内存占用:** Megatron SP 通过在序列维度上切分样本，显著减少了激活值的内存占用，尤其是在处理长序列时效果明显。这使得在有限的 GPU 内存下训练更长的序列成为可能。
* **与张量并行兼容:** Megatron SP 可以与张量并行技术结合使用，充分利用模型并行和序列并行的优势，进一步提高训练效率和可扩展性。

**Megatron SP 的局限性:**

* **只能与张量并行一起使用:** Megatron SP 依赖于模型并行 (通常是张量并行) 的基础，无法单独使用。
* **实现复杂度较高:** Megatron SP 的实现较为复杂，需要仔细处理序列维度的切分、汇总和通信操作。
* **通信开销:** 序列维度的汇总操作 (All-Gather) 会引入额外的通信开销。

**DeepSpeed-Ulysses 序列并行**

DeepSpeed-Ulysses 序列并行是 DeepSpeed 库中实现的一种序列并行方法。与 Megatron SP 不同，DeepSpeed-Ulysses SP 旨在提供一种更通用的序列并行解决方案，可以应用于更广泛的 Attention 机制，包括密集和稀疏 Attention。DeepSpeed-Ulysses SP 的核心思想是通过 All-to-All 通信操作，使每个 GPU 接收完整序列，但仅计算注意力头的非重叠子集，从而实现序列并行。

**DeepSpeed-Ulysses SP 的工作原理:**

1. **序列维度切分:** 与 Megatron SP 类似，DeepSpeed-Ulysses SP 也沿着序列维度将输入样本切分为多个分片。
2. **All-to-All 通信:** 在计算 Attention 之前，DeepSpeed-Ulysses SP 使用 All-to-All 通信操作，将序列分片在 GPU 之间进行全局交换。All-to-All 通信操作类似于分布式转置操作，可以将形状为 `[P, N/P, D]` 的张量转换为形状为 `[P, N/P, D]` 的张量，其中 P 是 GPU 数量，N 是序列长度，D 是特征维度。通过 All-to-All 通信，每个 GPU 可以接收到完整序列，但每个 GPU 只持有序列的不同部分。
3. **注意力头并行:** 在计算 Attention 时，DeepSpeed-Ulysses SP 将注意力头 (Attention Head) 进行并行化。每个 GPU 只负责计算一部分注意力头，例如 GPU $i$ 负责计算注意力头 $i, i+P, i+2P, ...$。由于每个 GPU 接收到的是完整序列，因此每个 GPU 可以独立地计算其负责的注意力头，无需额外的通信。
4. **输出拼接:** 在计算完所有注意力头后，将每个 GPU 上计算得到的注意力头输出进行拼接，得到完整的 Attention 输出。
5. **后续计算:** 后续的计算过程 (例如 MLP 层) 可以继续使用模型并行或数据并行等策略。

**DeepSpeed-Ulysses SP 的优势:**

* **通用性:** DeepSpeed-Ulysses SP 具有更强的通用性，可以应用于各种 Attention 机制，包括密集和稀疏 Attention。
* **完整序列信息:** 通过 All-to-All 通信，每个 GPU 可以接收到完整序列信息，保证了 Attention 计算的正确性。
* **注意力头并行:** 将注意力头进行并行化，可以有效降低计算量和内存占用。

**DeepSpeed-Ulysses SP 的局限性:**

* **All-to-All 通信开销:** All-to-All 通信操作会引入显著的通信开销，尤其是在 GPU 数量较多和序列长度较长时。All-to-All 通信的效率是 DeepSpeed-Ulysses SP 性能的关键瓶颈。
* **实现复杂度较高:** DeepSpeed-Ulysses SP 的实现较为复杂，需要仔细处理 All-to-All 通信和注意力头并行化。

**Ring Attention 序列并行**

Ring Attention 序列并行是一种借鉴 Flash Attention 思路的序列并行方法，旨在进一步提高长序列模型的训练效率，并降低内存占用。Ring Attention 的核心思想是将输入序列沿着序列维度切分为多个块，每个块由不同的 GPU 负责处理，并通过 "环形通信" 策略，在 GPU 之间传递 KV 子块，实现迭代计算，最终归约得到完整的 Attention 输出。

**Ring Attention 的工作原理:**

1. **序列分块:** 将输入序列沿着序列维度切分为多个块。例如，如果序列长度为 $L$，块大小为 $B$，则序列被切分为 $L/B$ 个块。
2. **GPU 环形排列:** 将 GPU 排列成环形结构，例如 GPU 0, GPU 1, ..., GPU (P-1), GPU 0, ...。
3. **块分配:** 将序列块分配到不同的 GPU 上。例如，块 1 分配到 GPU 0，块 2 分配到 GPU 1，依此类推。
4. **局部 Attention 计算:** 每个 GPU 负责计算分配给它的序列块的局部 Attention。局部 Attention 计算只考虑当前 GPU 上的序列块和相邻 GPU 传递过来的 KV 子块。
5. **环形通信 (Ring Communication):** 通过环形通信策略，每个 GPU 将其计算得到的 KV 子块传递给下一个 GPU，并接收来自前一个 GPU 的 KV 子块。环形通信迭代进行，直到每个 GPU 都接收到所有序列块的 KV 子块信息。
6. **全局 Attention 归约:** 在环形通信结束后，每个 GPU 都拥有计算完整 Attention 输出所需的所有信息。每个 GPU 计算其负责的序列块的完整 Attention 输出，并将结果进行归约 (例如求和或拼接)，得到最终的全局 Attention 输出。

**Ring Attention 的优势:**

* **内存效率高:** Ring Attention 每个 GPU 只需处理一个序列块，内存占用显著降低，尤其是在处理长序列时。
* **计算效率高:** 局部 Attention 计算和环形通信策略可以提高计算效率，减少冗余计算。
* **长序列支持:** Ring Attention 特别适用于处理超长文本序列，可以突破传统 Attention 机制的序列长度限制。
* **借鉴 Flash Attention:** Ring Attention 借鉴了 Flash Attention 的思想，例如分块计算和高效的内存访问模式，进一步提高了效率。

**Ring Attention 的局限性:**

* **环形通信开销:** 环形通信会引入一定的通信开销，尤其是在 GPU 数量较多时。环形通信的效率是 Ring Attention 性能的关键因素。
* **实现复杂度较高:** Ring Attention 的实现较为复杂，需要仔细设计序列分块、环形通信和 Attention 归约策略。

### 优化器相关的并行：ZeRO

ZeRO (Zero Redundancy Optimizer) 是一种与优化器相关的并行技术，旨在消除训练大型模型时的内存冗余，尤其是在使用混合精度训练时。ZeRO 的核心思想是将模型状态 (Model States) 分片到多个 GPU 上，模型状态主要包括：

1. **优化器状态 (Optimizer States):** 例如 Adam 优化器的 momentum 和 variance 等状态变量。
2. **梯度 (Gradients):** 用于更新模型权重的梯度信息。
3. **模型参数 (Parameters):** 模型权重参数。

ZeRO 在三个层面上进行优化，逐步消除内存冗余：

**ZeRO-1 (Optimizer State Partitioning)**

ZeRO-1 关注优化器状态的内存优化。它将优化器状态分割到不同的 GPU 上，每个 GPU 只存储一部分优化器状态。

**ZeRO-1 的工作原理:**

1. **优化器状态分片:** 将优化器状态 (例如 Adam 的 momentum 和 variance) 沿着参数维度进行分片，并将分片后的优化器状态分配到不同的 GPU 上。例如，如果有 $P$ 个 GPU，则将优化器状态切分为 $P$ 个分片，每个 GPU $i$ 存储与其负责的模型参数相对应的优化器状态分片。
2. **局部优化器更新:** 在参数更新阶段，每个 GPU 只更新其本地存储的优化器状态和模型参数分片。
3. **无需额外通信:** ZeRO-1 主要关注优化器状态的内存优化，不需要额外的跨 GPU 通信操作。

**ZeRO-1 的优势:**

* **显著减少优化器状态内存占用:** ZeRO-1 可以将优化器状态的内存占用降低到原来的 1/P，其中 P 是 GPU 数量。对于使用 Adam 等内存密集型优化器的模型，ZeRO-1 可以显著节省内存。
* **实现简单:** ZeRO-1 的实现相对简单，只需要修改优化器的状态管理方式即可。

**ZeRO-2 (Gradient Partitioning)**

ZeRO-2 在 ZeRO-1 的基础上，进一步优化了梯度的内存占用。它将用于更新模型权重的 32 位梯度也进行划分，每个 GPU 只存储与其优化器状态划分相对应的梯度。

**ZeRO-2 的工作原理:**

1. **梯度分片:** 除了优化器状态分片外，ZeRO-2 还将梯度沿着参数维度进行分片，并将分片后的梯度分配到不同的 GPU 上。每个 GPU $i$ 存储与其负责的模型参数和优化器状态相对应的梯度分片。
2. **局部梯度计算:** 每个 GPU 计算其负责的模型参数分片的梯度。
3. **梯度聚合 (Reduce-Scatter):** 在梯度同步阶段，ZeRO-2 使用 Reduce-Scatter 操作，将每个 GPU 上计算得到的梯度分片进行聚合，并将聚合后的梯度分片分散到对应的 GPU 上。Reduce-Scatter 操作是一种高效的梯度聚合方式，可以减少通信量。
4. **局部优化器更新:** 每个 GPU 使用其本地存储的梯度分片和优化器状态分片，更新本地模型参数分片。

**ZeRO-2 的优势:**

* **进一步减少梯度内存占用:** ZeRO-2 在 ZeRO-1 的基础上，进一步减少了梯度的内存占用，可以将梯度内存占用也降低到原来的 1/P。
* **高效的梯度聚合:** ZeRO-2 使用 Reduce-Scatter 操作进行梯度聚合，提高了梯度同步的效率。

**ZeRO-3 (Parameter Partitioning)**

ZeRO-3 是 ZeRO 系列的最高级别优化，它在 ZeRO-1 和 ZeRO-2 的基础上，将 16 位模型参数也分割到不同的 GPU 上。ZeRO-3 旨在最大程度地消除模型状态的内存冗余，使得训练超大型模型成为可能。

**ZeRO-3 的工作原理:**

1. **参数分片:** 除了优化器状态和梯度分片外，ZeRO-3 还将 16 位模型参数沿着参数维度进行分片，并将分片后的模型参数分配到不同的 GPU 上。每个 GPU $i$ 只存储与其负责的优化器状态和梯度相对应的模型参数分片。
2. **按需参数收集 (Parameter Gathering):** 在前向传播和反向传播过程中，当某个 GPU 需要访问完整的模型参数时，ZeRO-3 会按需地从其他 GPU 收集 (gather) 缺失的模型参数分片，组成完整的模型参数。参数收集操作只在必要时进行，以减少通信开销。
3. **局部计算:** 每个 GPU 使用收集到的模型参数进行局部计算。
4. **梯度聚合 (Reduce-Scatter):** 梯度聚合过程与 ZeRO-2 类似，使用 Reduce-Scatter 操作。
5. **局部优化器更新:** 每个 GPU 使用本地存储的梯度分片和优化器状态分片，更新本地模型参数分片。

**ZeRO-3 的优势:**

* **最大程度消除内存冗余:** ZeRO-3 可以将优化器状态、梯度和模型参数的内存占用都降低到原来的 1/P，最大程度地消除了模型状态的内存冗余。
* **支持训练超大型模型:** ZeRO-3 使得在有限的 GPU 内存下训练参数量巨大的模型成为可能，例如万亿参数模型。

**ZeRO 的总结:**

ZeRO 是一种强大的内存优化技术，通过分片模型状态，显著降低了训练大型模型时的内存占用。ZeRO-1, ZeRO-2, ZeRO-3 提供了不同级别的内存优化，用户可以根据模型规模和硬件资源选择合适的 ZeRO 级别。ZeRO 通常与混合精度训练结合使用，以进一步提高训练效率和节省内存。

## 其他内存节省设计

除了并行训练技术，还有许多其他内存节省设计可以帮助训练LLMs，这些设计主要从减少训练过程中各个环节的内存占用入手。

### CPU 卸载 (CPU Offloading)

CPU 卸载 (CPU Offloading) 是一种将暂时不使用的张量 (例如模型参数、优化器状态、中间激活值) 卸载到 CPU 内存中，在需要使用时再加载回 GPU 内存的技术。CPU 内存通常比 GPU 内存大得多，CPU 卸载可以有效扩展可用内存，使得在 GPU 内存受限的情况下也能训练大型模型。

**CPU 卸载的工作原理:**

1. **识别可卸载张量:**  识别训练过程中暂时不使用的张量，例如模型参数、优化器状态、中间激活值等。判断张量是否可以卸载的依据可以是张量的使用频率、生命周期等。
2. **张量卸载:** 将可卸载的张量从 GPU 内存移动到 CPU 内存。数据传输通常通过 PCIe 总线进行。
3. **张量预取 (Prefetching):**  在需要使用卸载到 CPU 内存的张量之前，提前将张量从 CPU 内存加载回 GPU 内存。预取操作可以与 GPU 的计算操作并行进行，以减少数据加载的延迟。
4. **张量使用:**  GPU 使用加载回 GPU 内存的张量进行计算。
5. **张量再次卸载:**  在张量使用完毕后，如果张量在一段时间内不再需要使用，可以再次将其卸载到 CPU 内存，释放 GPU 内存空间。

**CPU 卸载的优点:**

* **扩展可用内存:** CPU 卸载可以有效扩展可用内存，突破 GPU 内存容量限制，使得在 GPU 内存受限的情况下也能训练大型模型。
* **支持更大模型:** 通过 CPU 卸载，可以在有限的 GPU 内存下训练参数量更大的模型。

**CPU 卸载的缺点:**

* **数据传输开销:** CPU 卸载引入了 CPU-GPU 数据传输的开销，数据传输通常通过 PCIe 总线进行，速度相对较慢。频繁的 CPU-GPU 数据传输会降低训练速度。
* **训练速度降低:**  由于数据传输开销，CPU 卸载可能会导致训练速度降低，尤其是在数据传输量较大或者 PCIe 带宽受限的情况下。
* **实现复杂度较高:**  高效的 CPU 卸载需要仔细设计张量卸载和预取策略，以最小化数据传输开销，并保证训练的正确性和效率。

**ZeRO-Offload 和 ZeRO-Infinity:**

ZeRO-Offload 和 ZeRO-Infinity 是 DeepSpeed 库中实现的基于 CPU 卸载的内存优化技术。ZeRO-Offload 将优化器状态卸载到 CPU 内存，ZeRO-Infinity 更进一步，将模型参数也卸载到 CPU 内存甚至 NVMe 磁盘，突破 GPU 内存墙，支持训练更大规模的模型。

### 激活重计算 (Activation Recomputation / Checkpointing)

激活重计算 (Activation Recomputation)，也称为梯度检查点 (Gradient Checkpointing)，是一种以计算换内存的技术。在训练过程中，只保存部分激活值 (例如每个 Transformer 层的输入激活值)，在反向传播时，重新计算未保存的激活值。激活重计算可以显著减少训练过程中的激活值内存占用，尤其是在训练深层神经网络时效果明显。

**激活重计算的工作原理:**

1. **选择检查点:**  选择模型中的一些层作为检查点 (checkpoint)。通常选择模型中的关键层，例如 Transformer 层的输入层。
2. **前向传播 (Forward Pass):**  在前向传播过程中，只保存检查点层的激活值，对于非检查点层的激活值，在计算完成后立即释放，不进行保存。
3. **反向传播 (Backward Pass):**  在反向传播过程中，当需要计算某个非检查点层的梯度时，首先重新进行一次前向传播，计算该层的激活值 (重计算)，然后再进行反向传播计算梯度。对于检查点层，由于

...由于已经保存了检查点层的激活值，可以直接使用保存的激活值进行反向传播，无需重新计算。

**激活重计算的详细流程:**

1. **前向传播 (Forward Pass):**
    - 对于模型中的每一层 $l = 1, 2, ..., L$：
        - 如果层 $l$ 是检查点层，则计算并保存其输入激活值 $a_l^{in}$ 和输出激活值 $a_l^{out} = f_l(a_l^{in})$，其中 $f_l$ 是层 $l$ 的前向传播函数。
        - 如果层 $l$ 不是检查点层，则只计算其输出激活值 $a_l^{out} = f_l(a_l^{in})$，不保存输入激活值 $a_l^{in}$。
2. **反向传播 (Backward Pass):**
    - 从最后一层 $L$ 开始，反向传播计算梯度：
        - 对于每一层 $l = L, L-1, ..., 1$：
            - 如果层 $l$ 是检查点层，则使用保存的输入激活值 $a_l^{in}$ 和输出激活值 $a_l^{out}$ 进行反向传播，计算梯度 $\frac{\partial L}{\partial W_l}$ 和 $\frac{\partial L}{\partial a_l^{in}}$，其中 $W_l$ 是层 $l$ 的权重参数。
            - 如果层 $l$ 不是检查点层，则需要重新进行一次前向传播，计算输入激活值 $a_l^{in}$ 和输出激活值 $a_l^{out} = f_l(a_l^{in})$。然后使用重新计算的激活值进行反向传播，计算梯度 $\frac{\partial L}{\partial W_l}$ 和 $\frac{\partial L}{\partial a_l^{in}}$。

**激活重计算的内存成本分析:**

假设一个深度为 $\ell$ 层的神经网络被均匀地划分为 $d$ 个分区。只在分区边界保存激活值，分区内部的中间激活值在反向传播时重新计算。

* **每个分区的内存成本:** 假设每个分区的内存成本为 $M_p$ (主要由模型参数和少量边界激活值构成)。
* **分区数量:** $d$
* **分区边界数量:** $d-1$ (近似为 $d$)
* **总内存成本:** $M(\ell) \approx d \cdot M_p + \frac{\ell}{d} \cdot M_p$

其中，第一项 $d \cdot M_p$ 表示存储分区边界激活值的内存成本，第二项 $\frac{\ell}{d} \cdot M_p$ 表示存储模型参数的内存成本 (假设模型参数均匀分布在 $\ell$ 层中)。

为了最小化总内存成本 $M(\ell)$，对 $d$ 求导并令导数为零：

$$
\frac{dM(\ell)}{dd} = M_p - \frac{\ell}{d^2} \cdot M_p = 0
$$

解得最优分区数量 $d = \sqrt{\ell}$。此时，最小内存成本为：

$$
M_{\text{min}}(\ell) = O(\sqrt{\ell} \cdot M_p)
$$

因此，激活重计算可以将训练深度为 $\ell$ 的神经网络的内存成本降低到 $O(\sqrt{\ell})$，实现了亚线性级别的内存成本降低。代价是每个 mini-batch 需要额外进行一次前向传播的计算量。

**激活重计算的优点:**

* **显著减少激活值内存占用:** 激活重计算可以显著减少训练过程中的激活值内存占用，尤其是在训练深层神经网络时效果明显。
* **支持更大模型和更深网络:** 通过激活重计算，可以在有限的 GPU 内存下训练更大规模的模型和更深层的网络。

**激活重计算的缺点:**

* **增加计算成本:** 激活重计算需要重新计算部分激活值，增加了计算成本，训练时间会略微延长。通常情况下，激活重计算会增加约 20%-30% 的计算时间。
* **实现复杂度较高:** 激活重计算的实现需要对模型的前向和反向传播过程进行修改，实现复杂度相对较高。

**激活重计算在深度学习框架中的实现:**

主流深度学习框架 (如 PyTorch, TensorFlow) 都提供了激活重计算的实现。例如，PyTorch 中可以使用 `torch.utils.checkpoint.checkpoint_sequential` 或 `torch.utils.checkpoint.checkpoint` 函数来实现激活重计算。

### 混合精度训练 (Mixed Precision Training)

混合精度训练 (Mixed Precision Training) 是一种使用低精度浮点数 (例如 FP16 或 bfloat16) 与高精度浮点数 (例如 FP32) 相结合进行模型训练的技术。混合精度训练可以显著减少内存占用，并加速计算，同时保持模型精度接近甚至等同于全精度训练。

**混合精度训练的核心思想:**

利用低精度浮点数 (例如 FP16) 进行前向传播、反向传播和梯度计算，利用高精度浮点数 (例如 FP32) 存储模型权重和进行参数更新。现代 GPU (例如 NVIDIA Tensor Core) 在低精度计算方面具有更高的吞吐量，因此混合精度训练可以加速计算过程。同时，低精度浮点数占用更少的内存空间，可以减少内存带宽需求和内存占用。

**混合精度训练的关键技术:**

为了避免在低精度训练中丢失关键信息，并保证模型精度，混合精度训练通常需要结合以下关键技术：

1. **权重的全精度主副本 (Full-Precision Master Copy of Weights):**  维护一份全精度 (FP32) 的模型权重主副本。模型的前向传播、反向传播和梯度计算都使用半精度 (例如 FP16) 进行，但在参数更新时，使用全精度主副本进行更新。这样可以避免梯度更新过小，在半精度下被截断为零，导致训练停滞。

    **流程:**
    * 初始化模型权重为 FP32 精度，作为主副本。
    * 在每个迭代开始时，将 FP32 主副本权重转换为 FP16 精度，用于前向传播和反向传播。
    * 反向传播计算得到的梯度为 FP16 精度。
    * 在参数更新之前，将 FP16 梯度转换为 FP32 精度。
    * 使用 FP32 梯度更新 FP32 主副本权重。
    * 在下一个迭代开始时，再次将 FP32 主副本权重转换为 FP16 精度。

2. **损失缩放 (Loss Scaling):**  在反向传播之前，将损失函数的值乘以一个缩放因子 (loss scale)。这样可以放大梯度值，使其在 FP16 精度下能够更好地表示，避免梯度过小，在转换为 FP16 精度时丢失信息。

    **流程:**
    * 前向传播计算损失值 $L$ (FP32 精度)。
    * 将损失值乘以缩放因子 $S$，得到缩放后的损失值 $L' = L \times S$。
    * 使用缩放后的损失值 $L'$ 进行反向传播，计算梯度 (FP16 精度)。
    * 在参数更新之前，将梯度除以缩放因子 $S$，得到真实的梯度值。

    **缩放因子的选择:**  缩放因子 $S$ 的选择需要仔细调整。过小的缩放因子可能无法有效避免梯度下溢，过大的缩放因子可能导致梯度上溢。动态损失缩放 (Dynamic Loss Scaling) 是一种常用的策略，可以根据训练过程中的梯度值动态调整缩放因子。

3. **算术精度 (Arithmetic Precision):**  对于某些对精度敏感的算术运算 (例如向量点积、求和归约)，可以使用更高的精度 (例如 FP32) 进行累积，然后再将结果转换为半精度 (例如 FP16) 存储。对于逐元素运算，可以使用半精度或全精度，根据具体情况选择。

**混合精度训练的优点:**

* **减少内存占用:**  使用半精度浮点数可以减少模型参数、激活值和梯度信息的内存占用，通常可以减少一半的内存占用。
* **加速计算:**  现代 GPU 在半精度计算方面具有更高的吞吐量，混合精度训练可以加速矩阵乘法、卷积等计算密集型操作，提高训练速度。
* **提高训练吞吐量:**  内存占用和计算时间的减少，最终可以提高训练吞吐量，缩短训练时间。

**混合精度训练的缺点和挑战:**

* **数值稳定性问题:**  低精度浮点数的动态范围和精度有限，可能导致梯度下溢、梯度上溢等数值稳定性问题，影响模型收敛。损失缩放和权重的全精度主副本等技术可以缓解数值稳定性问题，但需要仔细调整超参数。
* **实现复杂度较高:**  混合精度训练的实现需要对模型的训练流程进行修改，包括权重精度转换、损失缩放、算术精度控制等，实现复杂度相对较高。

**混合精度训练在深度学习框架中的实现:**

主流深度学习框架 (如 PyTorch, TensorFlow) 都提供了混合精度训练的支持。例如，PyTorch 中可以使用 `torch.cuda.amp` 模块来实现自动混合精度训练 (Automatic Mixed Precision, AMP)。

### 压缩 (Compression)

压缩技术可以用于压缩训练过程中的中间结果，例如激活值、梯度信息等，以减少内存占用。压缩可以在前向传播后、反向传播前对激活值进行压缩，或者在反向传播计算梯度后、梯度同步前对梯度进行压缩。压缩后的数据在需要使用时再进行解压缩。

**压缩技术的类型:**

* **无损压缩 (Lossless Compression):**  无损压缩算法 (例如 Huffman 编码, Lempel-Ziv 算法) 可以保证解压缩后的数据与原始数据完全一致，不会引入任何信息损失。但无损压缩的压缩率通常较低，内存节省效果有限。
* **有损压缩 (Lossy Compression):**  有损压缩算法 (例如 JPEG, MPEG) 可以在一定程度上损失数据信息，以换取更高的压缩率。有损压缩可以显著减少内存占用，但可能会对模型精度和收敛性产生一定影响。

**压缩技术的应用场景:**

* **激活值压缩:**  激活值通常占用大量的内存空间，尤其是在训练深层神经网络时。压缩激活值可以有效减少内存占用，但需要在反向传播前进行解压缩，可能会增加计算开销。
* **梯度压缩:**  梯度信息在梯度同步时需要进行跨 GPU 通信，压缩梯度可以减少通信数据量，降低通信开销，提高分布式训练效率。

**Gist 系统：激活值压缩**

Gist 系统是一种用于压缩激活值的内存优化技术。Gist 系统采用了两种编码方案：

* **层特定无损编码 (Layer-Specific Lossless Encoding):**  针对 ReLU-Pool 和 ReLU-Conv 等特定层结构，设计了高效的无损编码方案。例如，对于 ReLU-Pool 层，可以使用二值化 (Binarize) 编码；对于 ReLU-Conv 层，可以使用稀疏存储和稠密计算 (Sparse Storage and Dense Computation) 编码。
* **激进有损编码 (Aggressive Lossy Encoding):**  使用延迟精度降低 (Delayed Precision Reduction, DPR) 技术进行有损压缩。DPR 的核心思想是，激活值在第一次使用时 (前向传播) 需要保持高精度，但在第二次使用时 (反向传播) 可以容忍较低的精度。因此，DPR 在前向传播后立即将激活值压缩到较低精度，在反向传播前再解压缩回高精度。

**DALL-E：梯度压缩**

DALL-E 模型在训练过程中使用了梯度压缩技术，在梯度同步之前，对梯度进行压缩，以减少通信数据量，提高分布式训练效率。DALL-E 使用的梯度压缩方法包括稀疏化 (Sparsification) 和量化 (Quantization)。

**压缩技术的优点:**

* **减少内存占用:**  压缩技术可以有效减少中间结果的内存占用，使得在有限的 GPU 内存下训练更大规模的模型成为可能。
* **降低通信开销:**  梯度压缩可以减少梯度同步时的通信数据量，降低通信开销，提高分布式训练效率。

**压缩技术的缺点和挑战:**

* **计算开销:**  压缩和解压缩操作会引入额外的计算开销，可能会降低训练速度。
* **精度损失 (有损压缩):**  有损压缩可能会导致数据信息损失，影响模型精度和收敛性。需要仔细选择合适的有损压缩算法和压缩率，以在内存节省和精度损失之间取得平衡。
* **实现复杂度较高:**  压缩技术的实现需要对模型的训练流程进行修改，包括压缩和解压缩操作的添加，以及与深度学习框架的集成，实现复杂度相对较高。

### 内存高效优化器 (Memory Efficient Optimizers)

传统的优化器 (例如 Adam, SGD with Momentum) 通常需要维护大量的优化器状态 (例如 momentum, variance)，这些优化器状态的内存占用与模型参数量相当，甚至更高。内存高效优化器 (Memory Efficient Optimizers) 旨在减少优化器状态的内存占用，从而降低训练大型模型的内存需求。

**Adam 优化器的内存占用:**

以 Adam 优化器为例，Adam 优化器需要为每个模型参数维护两个状态变量：一阶矩估计 (momentum) 和二阶矩估计 (variance)。这两个状态变量的形状和数据类型与模型参数相同，因此 Adam 优化器的内存占用约为模型参数内存占用的 2 倍。加上模型参数本身和梯度信息，使用 Adam 优化器训练模型，总的内存占用约为模型参数内存占用的 4 倍。

**内存高效优化器的设计思想:**

内存高效优化器的设计思想主要包括以下几个方面：

* **减少状态变量数量:**  减少优化器需要维护的状态变量数量。例如，Adafactor 优化器只维护行和列的和，而不是完整的二阶矩估计矩阵。
* **降低状态变量精度:**  降低状态变量的数据精度。例如，可以使用 FP16 或 bfloat16 精度存储优化器状态。
* **共享状态变量:**  在参数之间共享状态变量。例如，SM3 优化器在一定程度上共享了状态变量。

**Adafactor 优化器**

Adafactor 优化器是一种内存高效的自适应学习率优化器。与 Adam 优化器不同，Adafactor 优化器只跟踪移动平均值的行和列和，而不是完整的二阶矩估计矩阵。Adafactor 优化器可以显著减少优化器状态的内存占用，尤其是在模型参数矩阵具有低秩结构时效果更加明显。

**Adafactor 优化器的状态变量:**

Adafactor 优化器主要维护以下状态变量：

* **行尺度向量 (Row Scale Vector) $R$:**  形状为 `[行数]` 的向量，用于存储每行的二阶矩估计信息。
* **列尺度向量 (Column Scale Vector) $C$:**  形状为 `[列数]` 的向量，用于存储每列的二阶矩估计信息。
* **一阶矩估计 (Momentum) $M$:**  形状与模型参数相同的矩阵，用于存储一阶矩估计。

**Adafactor 优化器的参数更新公式:**

Adafactor 优化器的参数更新公式如下：

1. **计算二阶矩估计:**  Adafactor 优化器使用行尺度向量 $R$ 和列尺度向量 $C$ 来估计二阶矩：

   $$
   V_{ij} = R_i \cdot C_j
   $$

   其中，$V_{ij}$ 是参数矩阵 $W$ 的第 $i$ 行第 $j$ 列元素的二阶矩估计。

2. **计算自适应学习率:**  根据二阶矩估计 $V$，计算每个参数的自适应学习率：

   $$
   \eta_{ij} = \frac{\text{learning\_rate}}{\sqrt{V_{ij}} + \epsilon}
   $$

   其中，$\text{learning\_rate}$ 是全局学习率，$\epsilon$ 是一个小的平滑项，防止分母为零。

3. **更新一阶矩估计:**  使用梯度 $g$ 更新一阶矩估计 $M$：

   $$
   M \leftarrow \beta_1 M + (1 - \beta_1) g
   $$

   其中，$\beta_1$ 是一阶矩估计的指数衰减率。

4. **参数更新:**  使用自适应学习率 $\eta$ 和一阶矩估计 $M$ 更新模型参数 $W$：

   $$
   W \leftarrow W - \eta \odot M
   $$

   其中，$\odot$ 表示逐元素乘法。

**Adafactor 优化器的内存节省:**

Adafactor 优化器将二阶矩估计矩阵 $V$ 替换为行尺度向量 $R$ 和列尺度向量 $C$，显著减少了二阶矩估计的内存占用。对于形状为 `[行数, 列数]` 的参数矩阵，Adam 优化器需要存储形状相同的二阶矩估计矩阵，而 Adafactor 优化器只需要存储两个向量，内存占用大幅降低。

**SM3 优化器**

SM3 (Sparse Momentum for Massive Models) 优化器是另一种内存高效的自适应优化方法。SM3 优化器通过共享状态变量和稀疏更新等技术，实现了内存占用的大幅降低。

**SM3 优化器的特点:**

* **稀疏 Momentum:** SM3 优化器使用稀疏 Momentum，只为梯度非零的参数更新 Momentum，减少了 Momentum 的计算和存储开销。
* **状态共享:** SM3 优化器在一定程度上共享了状态变量，进一步减少了内存占用。
* **自适应学习率:** SM3 优化器也具有自适应学习率的特性，可以根据参数的梯度信息动态调整学习率。

**ZeRO 优化器 (Zero Redundancy Optimizer):**

ZeRO 优化器 (尤其是 ZeRO-1 和 ZeRO-2) 也可以被视为一种与优化器相关的并行技术和内存优化技术。ZeRO 通过分片优化器状态和梯度信息，显著减少了每个 GPU 上的优化器内存占用。

## 异构系统并行

异构系统并行 (Heterogeneous System Parallelism) 是一种充分利用异构计算资源 (例如 CPU, GPU, NVMe 磁盘) 进行模型训练的并行策略。传统的分布式训练主要依赖于 GPU 集群，但 CPU 和 NVMe 磁盘也具有各自的优势，例如 CPU 内存容量大，NVMe 磁盘存储容量大且读写速度快。异构系统并行旨在将计算和存储任务分配到最合适的设备上，充分利用各种硬件资源的优势，突破 GPU 内存墙，支持训练更大规模的模型。

**异构系统并行的核心思想:**

将模型训练过程中的不同任务分配到不同的硬件设备上执行，例如：

* **GPU:**  负责计算密集型任务，例如前向传播、反向传播、梯度计算等。GPU 具有强大的并行计算能力，适合处理这些计算密集型任务。
* **CPU:**  负责内存密集型任务，例如存储模型参数、优化器状态、中间激活值等。CPU 内存容量大，适合存储大量数据。
* **NVMe 磁盘:**  负责存储超大型模型参数和数据集，作为 CPU 内存的扩展。NVMe 磁盘具有高速读写能力，可以支持快速的数据加载和卸载。

**异构系统并行的关键技术:**

* **CPU 卸载 (CPU Offloading):**  将暂时不使用的张量 (例如模型参数、优化器状态) 卸载到 CPU 内存中，在需要使用时再加载回 GPU 内存。
* **NVMe 磁盘卸载 (NVMe Offloading):**  将模型参数和数据集卸载到 NVMe 磁盘中，作为 CPU 内存的扩展。
* **数据预取 (Data Prefetching):**  在 GPU 计算之前，提前将需要的数据从 CPU 内存或 NVMe 磁盘加载到 GPU 内存中，减少数据加载的延迟。
* **计算与通信重叠:**  尽可能地将数据加载和卸载操作与 GPU 的计算操作重叠，隐藏数据传输的开销。
* **Chunk-based 内存管理:**  将模型参数和数据划分为多个 Chunk，按需加载和卸载 Chunk，实现更精细化的内存管理。

**ZeRO-Offload 和 ZeRO-Infinity (异构系统并行):**

ZeRO-Offload 和 ZeRO-Infinity 是 DeepSpeed 库中实现的异构系统并行技术。ZeRO-Offload 将优化器状态卸载到 CPU 内存，ZeRO-Infinity 更进一步，将模型参数也卸载到 CPU 内存甚至 NVMe 磁盘。ZeRO-Offload 和 ZeRO-Infinity 充分利用 CPU 内存和 NVMe 磁盘的存储容量，突破 GPU 内存墙，支持训练更大规模的模型。

**PatrickStar 系统 (异构系统并行):**

PatrickStar 系统是另一种基于异构系统并行的模型训练系统。PatrickStar 系统采用了基于 Chunk 的内存管理策略，将模型参数和数据集划分为多个 Chunk，并按需在 GPU 内存、CPU 内存和 NVMe 磁盘之间进行数据迁移。PatrickStar 系统可以高效地利用异构计算资源，支持训练超大型模型。

**异构系统并行的优点:**

* **突破 GPU 内存墙:**  异构系统并行可以突破 GPU 内存容量限制，利用 CPU 内存和 NVMe 磁盘的存储容量，支持训练更大规模的模型。
* **降低训练成本:**  使用 CPU 内存和 NVMe 磁盘可以降低对昂贵 GPU 内存的依赖，降低训练成本。
* **提高训练效率 (潜在):**  通过合理地分配计算和存储任务，并优化数据传输和预取策略，异构系统并行有可能在某些场景下提高训练效率。

**异构系统并行的缺点和挑战:**

* **数据传输开销:**  CPU-GPU 和 NVMe-GPU 数据传输速度相对较慢，数据传输开销是异构系统并行性能的关键瓶颈。
* **实现复杂度较高:**  异构系统并行的实现非常复杂，需要仔细设计数据划分、任务调度、数据迁移和预取策略，以最小化数据传输开销，并保证训练的正确性和效率。
* **系统优化难度大:**  异构系统的性能优化难度较大，需要深入理解硬件架构和系统特性，进行精细化的性能调优。

## 总结

训练LLMs是一项极具挑战性的任务，需要综合运用各种并行技术和内存优化策略。本文详细介绍了训练LLMs的关键技术，包括：

* **并行训练技术:**
    * **数据并行 (Data Parallelism):**  最基础的并行方式，易于实现，但内存冗余较大。
        * **批量同步并行 (BSP):**  同步性好，但同步等待开销大。
        * **异步并行 (AsP):**  避免同步等待，但学习效率可能降低。
        * **分布式数据并行 (DDP) 与梯度累积:**  优化数据并行，减少通信开销。
    * **模型并行 (Model Parallelism):**  突破单 GPU 内存限制，训练超大型模型。
        * **流水线并行 (Pipeline Parallelism):**  层间并行，减少气泡问题。
            * **GPipe:**  同步梯度聚合，缓解气泡。
            * **PipeDream:**  1F1B 调度，权重暂存，进一步减少气泡和内存占用。
        * **张量并行 (Tensor Parallelism):**  层内并行，矩阵乘法并行化。
            * **Megatron-LM:**  Transformer 层内并行。
            * **PTD-P:**  多维并行策略，结合流水线、张量和数据并行。
    * **混合专家模型 (MoE):**  稀疏激活模型，增加参数量，不显著增加计算成本。
        * **GShard:**  分片 MoE Transformer。
        * **Switch Transformer:**  万亿参数模型，更高稀疏性。
        * **专家选择路由 (EC):**  专家选择 token，完美负载均衡。
    * **序列并行 (Sequence Parallelism):**  针对序列数据，降低长序列内存占用。
        * **Megatron SP:**  张量并行基础上实现序列并行。
        * **DeepSpeed-Ulysses SP:**  All-to-All 通信，通用 Attention 序列并行。
        * **Ring Attention SP:**  环形通信，高效长序列 Attention。
    * **优化器相关的并行：ZeRO:**  消除优化器内存冗余。
        * **ZeRO-1, ZeRO-2, ZeRO-3:**  不同级别的优化器状态、梯度和参数分片。

* **其他内存节省设计:**
    * **CPU 卸载 (CPU Offloading):**  扩展可用内存，但引入数据传输开销。
    * **激活重计算 (Activation Recomputation / Checkpointing):**  以计算换内存，减少激活值内存占用。
    * **混合精度训练 (Mixed Precision Training):**  使用低精度浮点数，减少内存和加速计算。
    * **压缩 (Compression):**  压缩中间结果，减少内存占用和通信开销。
    * **内存高效优化器 (Memory Efficient Optimizers):**  减少优化器状态内存占用。

* **异构系统并行:**  充分利用 CPU, GPU, NVMe 磁盘等异构资源，突破 GPU 内存墙。

选择合适的并行技术和内存优化策略需要根据具体的模型结构、数据集规模、硬件资源和训练目标进行权衡和选择。通常情况下，需要将多种技术结合使用，才能有效地训练大规模的模型，并取得最佳的性能和效率。

## 参考文献

1. Li, et al. "PyTorch Distributed: Experiences on Accelerating Data Parallel Training." *VLDB*, 2020.
2. Cui, et al. "GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server." *EuroSys*, 2016.
3. Shoeybi, et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv preprint* arXiv:1909.08053, 2019.
4. Narayanan, et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." *arXiv preprint* arXiv:2104.04473, 2021.
5. Huang, et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." *arXiv preprint* arXiv:1811.06965, 2018.
6. Narayanan, et al. "PipeDream: Generalized Pipeline Parallelism for DNN Training." *SOSP*, 2019.
7. Narayanan, et al. "Memory-Efficient Pipeline-Parallel DNN Training." *ICML*, 2021.
8. Shazeer, et al. "The Sparsely-Gated Mixture-of-Experts Layer." *arXiv preprint* arXiv:1701.06538, 2017.
9. Lepikhin, et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." *arXiv preprint* arXiv:2006.16668, 2020.
10. Fedus, et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *arXiv preprint* arXiv:2101.03961, 2021.
11. Narang & Micikevicius, et al. "Mixed Precision Training." *ICLR*, 2018.
12. Chen, et al. "Training Deep Nets with Sublinear Memory Cost." *arXiv preprint* arXiv:1604.06174, 2016.
13. Jain, et al. "Gist: Efficient Data Encoding for Deep Neural Network Training." *ISCA*, 2018.
14. Shazeer & Stern. "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost." *arXiv preprint* arXiv:1804.04235, 2018.
15. Anil, et al. "Memory-Efficient Adaptive Optimization." *arXiv preprint* arXiv:1901.11150, 2019.
16. Rajbhandari, et al. "ZeRO: Memory Optimization Towards Training A Trillion Parameter Models." *arXiv preprint* arXiv:1910.02054, 2019.
17. Zhou, et al. "Mixture-of-Experts with Expert Choice Routing." *arXiv preprint* arXiv:2202.09368, 2022.
18. Weng, Lilian. "How to Train Really Large Models on Many GPUs?" *Lil'Log*, September 24, 2021. [Link](https://lilianweng.github.io/posts/2021-09-25-train-large/)
19. 猛猿：图解大模型训练系列 [Link](https://zhuanlan.zhihu.com/p/654910335)