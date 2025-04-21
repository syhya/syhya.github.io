---
title: "训练大模型并行和内存优化技术"
date: 2025-03-01T12:00:00+08:00
lastmod: 2025-03-01T12:00:00+08:00
author: Yue Shui
categories: ["技术博客"]
tags: [LLMs, 预训练, 分布式训练, 内存优化, 数据并行, 模型并行, 流水线并行, 张量并行, 序列并行, 混合并行, 异构系统, MoE, ZeRO, LoRA, AI, 深度学习, AI Infrastructure]
readingTime: 60
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

## 背景

最近大模型的参数数量不断攀升，从最初的数十亿扩展到如今数千亿乃至数万亿级别。大模模型虽然带来了前所未有的应用效果，但与此同时，也引发了计算资源、内存管理和训练稳定性等一系列严峻挑战。因此本博客总结了一些常用分布式并行训练和内存管理技术，希望能够帮助大家更好地训练和优化大模型。


### 大模型的训练挑战

* **参数规模爆炸式增长**  
  随着对模型容量和性能的不断追求，神经网络的参数数量呈现出指数级增长。现今从百万级到数十亿、数千亿甚至数万亿参数的模型层出不穷。例如，Llama 3.1 405B 拥有约 4,050 亿参数，而据传 GPT-4 的参数量可能达到 1.7 万亿级别。这种庞大的参数规模使得计算和内存需求急剧上升，给训练过程带来了前所未有的压力。

* **计算复杂度剧增**  
  参数数量的急速增加直接导致整体计算复杂度大幅上升。训练一次大型模型可能需要耗费数周甚至数月的时间，即便采用大规模高性能 GPU 集群，训练周期仍难以令人满意，从而严重制约了模型迭代速度和研究效率。

* **内存瓶颈日益凸显**  
  除了需要存储庞大的模型参数之外，大模型在训练过程中还必须保存中间激活值、梯度信息以及优化器状态，这些数据对 GPU 显存构成了巨大挑战。即使是配备 A100、H100(80GB 显存)、H200(141GB 显存)或 GB200(384GB 显存)等高端 GPU，单卡内存往往也难以满足数千亿甚至数万亿级模型的需求，“Out of Memory(OOM)”错误频发。

* **通信开销成为瓶颈**  
  在多 GPU 分布式训练环境中，节点间需要频繁进行数据同步(如梯度汇总)。随着模型规模和 GPU 数量的增加，这种通信量急剧上升，即使在高带宽网络中，All-Reduce 操作传输海量数据也会消耗大量时间，成为整体并行效率的主要瓶颈之一。

* **训练稳定性挑战**  
  超大规模模型在训练过程中更容易遭遇梯度消失或梯度爆炸问题，导致训练过程不稳定、难以收敛。虽然混合精度训练可以在一定程度上加速训练并降低显存占用，但同时也可能引入新的数值稳定性问题，要求研究人员投入更多精力进行细致调优。


### 分布式训练的必要性

面对上述挑战，分布式训练技术成为支撑大模型训练的关键方案。通过将训练任务拆分并分配到多台 GPU 或计算节点上，分布式训练能够充分利用并行计算和集群内存资源，从而突破单 GPU 的局限，主要优势体现在以下几个方面：

* **突破单 GPU 算力限制**  
  单个 GPU 的计算能力终究有限，无法应对万亿级别参数模型的庞大计算需求。借助数据并行与模型并行技术，训练任务可以均匀分布至多个 GPU，从而大幅缩短整体训练时间。

* **克服单 GPU 内存瓶颈**  
  通过将模型参数、中间激活值和优化器状态分散存储在多个 GPU 显存中，分布式训练有效扩展了可用内存容量。典型技术如 ZeRO，通过对模型参数、梯度以及优化器状态进行分片，大幅降低单卡的显存负担，使得超大规模模型的训练成为可能。

* **加速模型迭代与研发周期**  
  分布式训练的高并行度使得原本需要数周甚至数月完成的训练任务有望在数天内完成，从而大幅提升模型迭代速度，使得新架构和新策略能够更快得到验证与应用。

* **支持更大规模模型探索**  
  分布式训练为探索更大规模、更复杂的神经网络架构提供了坚实基础。正是在这种技术支持下，万亿参数级别的模型(如 Switch Transformer)才得以成功训练并投入实际应用。

* **提高训练系统的鲁棒性与可扩展性**  
  分布式系统具备出色的容错能力，当某个 GPU 节点出现故障时，其他节点可迅速接管任务，确保训练过程不被中断。同时，集群规模可以根据具体需求灵活扩展或缩减，满足不同规模模型的训练要求。


## 并行训练

下图直观展示了多种并行训练策略的不同之处。不同颜色代表不同的模型层(例如三层)，虚线将不同的 GPU 区分开。从左到右分别是数据并行、模型并行(含流水线并行和张量并行)以及专家并行(MoE)。


{{< figure
    src="parallelism_compare.png"
    caption="Fig. 1. An illustration of various parallelism strategies on a three-layer model. Each color refers to one layer and dashed lines separate different GPUs. (Image source: [OpenAI Blog, 2022](https://openai.com/index/techniques-for-training-large-neural-networks/))"
    align="center"
    width="90%"
>}}

- **数据并行**  
  完整模型会被拷贝到每个 GPU 上，数据集则被切分为不同批次分配给各个 GPU 并行计算，最终在参数更新时聚合所有 GPU 的梯度。

- **模型并行**  
  将模型划分到不同的 GPU 上，每个 GPU 只负责模型的一部分计算；可进一步分为以下两类：  
  - **流水线并行**：按层(垂直方向)拆分模型，不同 GPU 负责不同的层，通过微批次(micro-batch)在流水线中传递来并行执行前向和反向计算。  
  - **张量并行**：在层内(水平方向)对大规模张量操作(如大矩阵乘法)进行切分，各 GPU 并行完成这部分运算并在必要时进行聚合。

- **专家并行**  
  通过门控策略，让每个输入样本只经过部分专家(子网络)，从而将整个模型按“专家模块”分布到不同 GPU。常见于 Mixture-of-Experts(MOE) 结构，可实现超大参数规模但推理/训练时仅激活部分专家。


下面我将对多种并行方式进行详细说明。

## 数据并行
  
{{< figure
    src="data_parallelism.png"
    caption="Fig. 2. Data Parallelism. (Image source: [Clolossal-AI Documentation](https://colossalai.org/docs/concepts/paradigms_of_parallelism))"
    align="center"
    width="60%"
>}}

在深度学习训练中，**数据并行(Data Parallelism, DP)** 是最常用的并行策略，其核心思路是：  
1. **复制模型参数**：在每个计算设备(通常是 GPU)上都放置一份完整的模型参数。  
2. **划分训练数据**：将大规模的数据集按样本维度拆分为多个子集，不同子集分配给不同的 GPU 进行处理。  
3. **局部前向与反向传播**：每个 GPU 独立计算损失及对应的局部梯度。  
4. **梯度/参数同步**：将各 GPU 的梯度聚合后更新模型参数，保证在每一次迭代后所有 GPU 的模型副本保持一致。  
  
下面展示了 **数据并行** 工作流程：  
  
1. **数据集划分**    
   将训练数据集 $D$ 划分为 $N$ 个互不重叠的子集 $\{D_1, D_2, \dots, D_N\}$，其中 $N$ 是 GPU 数量。通常会确保各子集大小相近，以实现负载均衡。  
  
2. **模型复制**    
   在每个 GPU 上复制一份完整的模型参数 $\theta$。在训练开始时，这些参数在各 GPU 上都是相同的。  
  
3. **数据分发**    
   将子集 $D_i$ 分发给第 $i$ 张 GPU，让其在本地存储并供后续计算使用。  
  
4. **局部前向传播**    
   每个 GPU 基于其本地数据子集 $D_i$ 做前向传播，得到局部损失 $L_i(\theta, D_i)$。  
  
5. **局部反向传播**    
   每个 GPU 基于局部损失 $L_i$ 进行反向传播，计算局部梯度
     
   $$  
     g_i = \nabla_{\theta} L_i(\theta, D_i).  
   $$  
  
6. **梯度同步**    
   各 GPU 之间执行梯度同步(常用 All-Reduce)，将所有局部梯度 $\{g_1, g_2, \ldots, g_N\}$ 汇总得到全局平均梯度 

   $$  
     \bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i.  
   $$  
  
7. **参数更新**    
   每个 GPU 使用全局平均梯度 $\bar{g}$ 更新本地模型参数：

   $$  
     \theta \leftarrow \theta - \eta \cdot \bar{g},  
   $$

   其中 $\eta$ 为学习率(learning rate)。  
  
8. **迭代循环**    
   重复步骤 4 - 7，直至模型达到收敛或达到预设的训练轮数(epochs)。  
  
### 批量同步并行与异步并行
  
在上面的第 6 步“梯度同步”中，如何以及何时进行“同步”是影响数据并行性能和收敛行为的重要因素之一。一般分为以下两大类：  

**批量同步并行(Bulk Synchronous Parallel, BSP)** 是数据并行中最常见、也是最易理解的同步模式。其特点可概括为「在每一次小批量(mini-batch)迭代结束后，全局同步一次梯度并更新参数」。具体流程：  
  
1. **局部计算**：各 GPU 基于其数据子集 $D_i$ 分别做前向与反向传播，得到局部梯度 $g_i$。  
2. **全局通信**：所有 GPU 同步(如通过 All-Reduce)计算 $\bar{g}$。  
3. **参数更新**：每个节点均使用 $\bar{g}$ 更新本地参数副本 $\theta$。  
4. **等待与下一步迭代**：所有节点完成上述操作后，再进入下一个迭代。  

**异步并行(Asynchronous Parallel, ASP)** 旨在摆脱 BSP 的全局同步点，让各节点独立进行计算和参数更新。其典型实现是「参数服务器」(Parameter Server, PS) 架构下的 **异步 push-pull** 过程：  
  
1. 各节点在本地计算得到梯度 $g_i$, 然后 **push** 到参数服务器；    
2. 参数服务器一旦收到梯度，立即更新全局模型参数；    
3. 其他节点在需要最新参数时，会 **pull** 下来继续下一步计算。  
  
### BSP vs. ASP
  
下表总结了在数据并行环境下，同步并行与异步并行的主要差异：  
  
| **对比维度**           | **同步并行(BSP)**                                                      | **异步并行(ASP)**                                                          |  
|:-----------------------|:------------------------------------------------------------------------|:----------------------------------------------------------------------------|  
| **参数更新时机**       | 每个小批量或一定迭代后，全局同步一次                                    | 各节点独立更新参数，无需与他人保持同一时间步                                 |  
| **收敛稳定性**         | **高**。使用的梯度均为最新，收敛路径可控，易于分析                      | **较低**。存在过时梯度，收敛速率与稳定性可能受影响                          |  
| **通信需求**           | 高度依赖 All-Reduce，同步时所有节点都需要等待和交换数据                  | 每个节点向参数服务器异步推送/拉取，通信更为灵活，但参数服务器可能成为瓶颈     |  
| **硬件资源利用**       | 若有慢节点或网络延迟，则其他节点需等待，资源利用率可能降低               | 无需等待慢节点，可高效使用计算资源                                          |  
| **实现复杂度**         | 相对较低，主流框架(PyTorch DDP、Horovod 等)有内置支持                  | 相对更高，需要参数服务器等组件，需处理更多的同步逻辑与数据一致性             |  
| **适用场景**           | 同构硬件、网络带宽良好、追求较高收敛质量                                 | 异构硬件、网络不稳定或带宽较低、需要极高的吞吐量且能容忍一定收敛风险         |  
| **典型实现**           | PyTorch DDP、TensorFlow MirroredStrategy                                 | Parameter Server 架构(MXNet、TensorFlow ParameterServer 模式等)            |  
  
> **建议**：在实际项目中，先从简单的同步并行(BSP) 入手，利用 PyTorch DDP 或类似工具进行多 GPU 训练。若网络环境异构、节点繁多或任务对吞吐率要求极高，可再尝试异步并行(ASP) 或参数服务器方案，并配合梯度累积(Gradient Accumulation) 来平衡带宽与更新频率。  
  
### 梯度累积  
  
当批量大小较大或通信成为主要瓶颈时，可以采用 **梯度累积(Gradient Accumulation)** 来减少同步频率。其核心思路是：    
- 连续计算多个小批量(mini-batch)的局部梯度，并将它们累加到本地的累积缓冲区中；    
- 当累积的 mini-batch 数量达到 $K$ 时，再触发一次全局梯度同步与参数更新。  
  
设第 $j$ 个 mini-batch 的梯度为 $g_j$，则在一个「累积周期」内得到

$$  
  G = \sum_{j=1}^{K} g_j.  
$$  

再用学习率 $\eta$ 更新：

$$  
  \theta \leftarrow \theta - \eta \cdot G.  
$$  
  
由于梯度同步不再是每个 mini-batch 都进行，而是每累计 $K$ 个 mini-batch 执行一次，**通信开销可显著降低**。但参数更新频率降低也可能导致训练收敛速度放缓，需在吞吐量与收敛性能之间做权衡。  
  
### 分布式数据并行
  
**分布式数据并行(Distributed Data Parallel, DDP)** 是 PyTorch v1.5([Li et al. 2020](https://arxiv.org/pdf/2006.15704))在 BSP 思想下的高度优化实现，为单机多 GPU 乃至多机多 GPU 的数据并行提供便利。其主要优化包括：  
  
1. **梯度 Bucketing(梯度桶化)**：将模型参数分为多个「桶」(bucket)；反向传播时一旦某个桶内所有梯度都已计算完，就立即启动一次针对**该桶的 All-Reduce**，而不是等到所有梯度都算完后再一次性同步。    
2. **通信与计算重叠**：DDP 通过异步通信和非阻塞操作，尽可能地将梯度同步(通信)与前向传播、反向传播(计算)重叠，从而减少了通信开销。这种重叠策略提升了整体的并行效率。  
3. **梯度累积**：DDP 也能方便地与**梯度累积**相结合，结合使用，通过增加每次同步的梯度更新间隔，从而减少同步频率。这在大规模分布式训练中有助于进一步降低通信开销，提高训练效率。

{{< figure
    src="pytorch_ddp.png"
    caption="Fig. 3. Pseudo code for Pytorch DDP. (Image source: [Li et al. 2020](https://arxiv.org/pdf/2006.15704))"
    align="center"
    width="80%"
>}}

### Ring All-Reduce  
  
在多 GPU(尤其是单机多 GPU)环境下，若有高速互联(如 NVLink、PCIe 交换机等)，可使用 **Ring All-Reduce** 来显著降低通信开销。其思路是：  
  
1. 将 $k$ 个节点组织成一个环，并把梯度向量等分成 $k$ 份。    
2. 在「加和阶段」，每个节点分别向下一个节点发送其本地的一部分梯度，并与收到的梯度相加；该过程循环若干次后，每个节点会持有完整的「聚合后」梯度。    
3. 在「广播阶段」，再将最终结果沿环路分发给所有节点。  
  
理想情况下，Ring All-Reduce 的通信代价与节点数量近似无关(可以视为 $\mathcal{O}(1)$)，非常适合多 GPU 环境下的梯度同步，是 Horovod、NCCL 等库中广泛使用的核心通信模式。  
  
### 参数服务器
  
当集群规模扩展至多机多 GPU 时，若简单地采用单点聚合(例如一台中心服务器)往往难以支撑海量数据的并行训练。**参数服务器(Parameter Server, PS)**([Li, et al. 2014](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf))是为可扩展分布式训练而设计的一种典型架构：  
  
1. **参数分片**：将模型参数按键值对(key-value) 的形式进行拆分，不同 PS 节点只管理特定分片的参数；    
2. **push-pull** 语义：计算节点在本地得到梯度后，**push** 到相应的 PS；PS 更新完该分片参数后，计算节点可在需要时 **pull** 下最新版本进行下一步计算。    
3. **灵活容错与扩展**：通过增加或移除 PS 节点，可在带宽或计算需求上灵活扩容；在 PS 上也能实现备份与容错策略。  
  
这种 **PS + Worker** 模式可以**同时**结合数据并行和模型并行，将超大模型拆分到多个 PS 上存储，并对超大数据进行分布式训练。PS 本身也能根据负载情况做拆分与合并，形成更加复杂的层次化拓扑结构。  


## 模型并行

**模型并行(Model Parallelism, MP)** 是一种将模型本身分割到多个计算设备(GPU) 上进行训练的并行方式。当模型参数规模超过单个 GPU 的内存容量时，模型并行成为必要的选择。模型并行主要分为两种类型：流水线并行(Pipeline Parallelism) 和张量并行(Tensor Parallelism)。

**朴素模型并行与气泡问题**

{{< figure
    src="naive_mp.png"
    caption="Fig. 4. A naive model parallelism setup where the model is vertically split into 4 partitions. Data is processed by one worker at a time due to sequential dependency, leading to large “bubbles” of idle time. (Image source: [Huang et al. 2018](https://arxiv.org/abs/1811.06965))"
    align="center"
    width="100%"
>}}

朴素的模型并行实现，即将模型简单地按层划分，并顺序地在不同 GPU 上执行，会遇到严重的 "气泡"(bubble) 问题。由于层之间的依赖关系，当一个 GPU 在处理某个数据样本的某个阶段时，其他 GPU 可能处于空闲状态，等待前一个 GPU 的输出或者后一个 GPU 的输入。这种 GPU 空闲时间被称为 "气泡"，严重降低了流水线并行的效率。

其中，$F_i$ 表示 Stage $i$ 的前向传播，$B_i$  表示 Stage $i$ 的反向传播。可以看到，在朴素流水线并行中，大部分时间只有一个 GPU 在工作，其他 GPU 处于空闲状态，效率低下。

**气泡问题产生的原因:**

* **层间依赖:**  神经网络的层之间存在顺序依赖关系，后一层的计算必须依赖于前一层的输出。
* **顺序执行:**  朴素模型并行按照层顺序依次执行，导致 GPU 之间无法充分并行工作。


## 流水线并行

{{< figure
    src="pipeline_parallelism.png"
    caption="Fig. 5. Pipeline Parallelism. (Image source: [Clolossal-AI Documentation](https://colossalai.org/docs/concepts/paradigms_of_parallelism))"
    align="center"
    width="60%"
>}}


**流水线并行(Pipeline Parallelism, PP)** 将模型按层划分为多个阶段(stage)，每个阶段分配到一个 GPU 上。数据像流水线一样在不同 GPU 之间传递，前一个 GPU 的输出作为后一个 GPU 的输入。流水线并行旨在提高模型并行训练的效率，减少 GPU 空闲时间。

### GPipe

**GPipe**([Huang et al. 2018](https://arxiv.org/abs/1811.06965)) 是 Google 提出的一个高效的流水线并行训练系统，旨在解决朴素流水线并行的气泡问题。GPipe 的核心思想是将 **mini-batch** 划分为多个 **micro-batch**，并采用**同步梯度聚合**的方式来缓解气泡问题，提高流水线效率。

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
    * **前向传播(Forward Propagation):**  对于每个 micro-batch，依次在 Stage $1$, Stage $2$, ..., Stage $d$ 上进行前向传播。Stage $i$ 的输出作为 Stage $i+1$ 的输入。
    * **反向传播(Backward Propagation):**  当所有 micro-batch 的前向传播都完成后，开始反向传播。对于每个 micro-batch，依次在 Stage $d$, Stage $d-1$, ..., Stage $1$ 上进行反向传播。Stage $i$ 的梯度作为 Stage $i-1$ 的输入。
4. **同步梯度聚合(Synchronous Gradient Aggregation):**  在所有 micro-batch 的反向传播都完成后，将所有 micro-batch 的梯度进行聚合(例如求平均)，得到全局平均梯度。
5. **参数更新(Parameter Update):**  每个 GPU 使用全局平均梯度更新本地模型参数。


### GPipe 气泡比例公式

假设每个 micro-batch 的前向和反向传播时间均为 1 单位，流水线深度为 $d$，micro-batch 数量为 $m$，则 GPipe 的气泡比例为：

$$
\text{Bubble Ratio} = 1 - \frac{2md}{(2m + 2(d-1))d} = \frac{d-1}{m+d-1}
$$

当 micro-batch 数量 $m$ 远大于流水线深度 $d$ 时($m \gg d$)，气泡比例趋近于 0，流水线效率接近线性加速。GPipe 论文中指出，当 $m > 4d$ 时，气泡开销几乎可以忽略不计(在激活重计算的情况下)。因此有以下好处：

* **减少气泡:**  GPipe 通过 micro-batch 划分和流水线调度，显著减少了朴素流水线并行的气泡问题，提高了 GPU 利用率和训练效率。
* **同步梯度聚合:**  GPipe 采用同步梯度聚合，保证了训练过程的同步性，模型收敛性较好。
* **线性加速潜力:**  在 micro-batch 数量足够大的情况下，GPipe 可以实现接近线性的加速效果。

### PipeDream

{{< figure
    src="pipe_dream.png"
    caption="Fig. 7. Illustration of 1F1B microbatch scheduling in PipeDream. (Image source: [Harlap et al. 2018](https://arxiv.org/abs/1806.03377))"
    align="center"
    width="100%"
>}}

**PipeDream**([Harlap et al. 2018](https://arxiv.org/abs/1806.03377))是另一种高效的流水线并行训练系统，它采用了 1F1B(1-Forward-1-Backward) 调度策略，并引入了权重暂存(Weight Stashing) 技术，进一步减少气泡，提高流水线效率，并解决 1F1B 调度可能导致的权重版本不一致问题。

PipeDream 的 1F1B 调度策略的核心思想是，每个 GPU(Stage) 交替执行前向传播和反向传播，尽可能地并行工作，减少 GPU 空闲时间。具体流程如下：

1. **Micro-batch 划分:**  将一个 mini-batch 划分为 $m$ 个 micro-batch。
2. **流水线阶段划分:**  将模型按层划分为 $d$ 个阶段，每个阶段分配到一个 GPU 上。
3. **1F1B 调度执行:**  每个 GPU 轮流执行前向传播和反向传播。

### 权重暂存

由于 1F1B 调度中，前向传播和反向传播可能使用不同版本的模型权重，会导致权重版本不一致问题，影响训练的正确性和收敛性。PipeDream 引入了 **权重暂存(Weight Stashing)** 技术来解决这个问题。权重暂存的核心思想是，每个 GPU 维护多个版本的模型权重，并确保前向传播和反向传播使用同一版本的权重。

**权重暂存实现方式:**

* **版本管理:**  每个 GPU 维护一个权重版本队列，存储多个版本的模型权重。
* **版本选择:**  在进行前向传播时，选择当前最新的权重版本。在进行反向传播时，选择与对应前向传播相同的权重版本。
* **版本更新:**  在完成一个 mini-batch 的所有 micro-batch 的反向传播后，更新模型权重，并生成新的权重版本。


为了进一步优化 PipeDream 的内存使用，尤其是在权重暂存方面，PipeDream 衍生出了 **PipeDream-flush** 和 **PipeDream-2BW** 两种内存优化变体。

### PipeDream-flush

{{< figure
    src="pipe_dream_flush.png"
    caption="Fig. 8. Illustration of pipeline scheduling in PipeDream-flush. (Image source: [Narayanan et al. 2020](https://arxiv.org/abs/2006.09503))"
    align="center"
    width="100%"
>}}

**PipeDream-flush** 在 PipeDream 的基础上，周期性地进行全局同步的流水线刷新(flush)，类似于 GPipe 的同步梯度聚合。通过定期刷新，PipeDream-flush 可以大幅减少权重暂存所需的内存空间，只需维护单个版本的模型权重，但会牺牲少量吞吐量。


### PipeDream-2BW

***PipeDream-2BW(Double-Buffered Weights)** 维护两个版本的模型权重，即 "双缓冲权重"。它每 $k$ 个 micro-batch 更新一次模型版本，其中 $k$ 大于流水线深度 $d$($k > d$). 新更新的模型版本不会立即完全替换旧版本，因为可能还有一些剩余的反向传播操作仍然依赖于旧版本。通过双缓冲权重，PipeDream-2BW 可以将权重暂存的内存开销降低到只维护两个版本的模型权重，显著减少内存占用。

{{< figure
    src="pipe_dream_2bw.png"
    caption="Fig. 9. Illustration of pipeline scheduling in PipeDream-2BW. (Image source: [Narayanan et al. 2020](https://arxiv.org/abs/2006.09503))"
    align="center"
    width="100%"
>}}  

PipeDream-2BW 策略有以下优点：

* **更低的气泡开销:**  1F1B 调度策略相比 GPipe 可以进一步减少气泡，提高 GPU 利用率和训练效率。
* **权重暂存解决版本一致性:**  权重暂存技术保证了前向传播和反向传播使用同一版本的权重，解决了 1F1B 调度可能导致的权重版本不一致问题。
* **内存优化变体:**  PipeDream-flush 和 PipeDream-2BW 进一步优化了内存使用，降低了权重暂存的内存开销，使得流水线并行更适用于内存受限的场景。


## 张量并行

**张量并行(Tensor Parallelism, TP)** 是一种将模型中的张量(通常是权重矩阵) 沿着特定维度切分，并将切分后的分片分配到不同的 GPU 上进行计算的并行方式。张量并行有以下几点优势：

* **突破单 GPU 显存限制:**  张量并行可以将模型参数分散存储在多个 GPU 上，突破单 GPU 显存容量限制，支持训练更大规模的模型。
* **层内并行:**  张量并行可以实现模型层内部的并行化，例如矩阵乘法操作的并行计算，提高计算效率。
* **与数据并行和流水线并行结合:**  张量并行可以与数据并行和流水线并行等其他并行技术结合使用，形成多维并行策略，进一步提高训练效率和可扩展性。

### Megatron-LM

**Megatron-LM**([Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)) 是 NVIDIA 提出的一个用于训练超大型语言模型的系统，它采用了张量并行技术，对 Transformer 模型层内部的矩阵乘法操作进行并行化，包括 **self-attention** 和 **MLP** 中的矩阵乘法。

{{< figure
    src="Megatron-LM.png"
    caption="Fig. 10. Illustration of tensor parallelism for key transformer components proposed in Megatron-LM. (Image source: [Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053))"
    align="center"
    width="100%"
>}}  

Transformer 的 MLP 层通常包含两个线性层，第一个线性层的计算可表示为 $Y = \text{GeLU}(XA)$，其中 $X$ 是输入矩阵，$A$ 是权重矩阵，GeLU 是激活函数。Megatron-LM 将权重矩阵 $A$ 沿着列维度切分为 $P$ 个分片 $[A_1, A_2, ..., A_P]$，其中 $P$ 是 GPU 的数量。每个 GPU $i$ 负责存储和计算权重分片 $A_i$。

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
3. **全局拼接(All-Gather):**  所有 GPU 通过 All-Gather 操作，将局部输出 $\{Y_1, Y_2, ..., Y_P\}$ 拼接成完整的输出矩阵 $Y = [Y_1, Y_2, ..., Y_P]$。

**自注意力层张量并行**

Megatron-LM 也对 Transformer 的自注意力层中的 Query($Q$), Key($K$), Value($V$) 权重矩阵进行张量并行切分，并进行相应的局部矩阵乘法和全局拼接操作，实现自注意力层的张量并行化。自注意力层的计算公式为：

$$
\text{Attention}(X, Q, K, V) = \text{softmax}\left(\frac{(XQ)(XK)^T}{\sqrt{d_k}}\right)XV
$$


### PTD-P

**PTD-P(Pipeline, Tensor, and Data Parallelism)**([Narayanan et al. 2021](https://arxiv.org/abs/2104.04473))是一个结合了流水线并行、张量并行和数据并行的多维并行策略。PTD-P 旨在充分利用各种并行技术的优势，提高超大型模型训练的效率和可扩展性。

**PTD-P 的特点:**

* **多维并行结合:**  PTD-P 同时使用了流水线并行、张量并行和数据并行三种并行技术，可以从多个维度对训练过程进行并行化。
* **Interleaved 1F1B 调度:**  PTD-P 采用了 interleaved 1F1B 调度策略，与传统的流水线并行不同，它将模型划分为多个不连续的层块(model chunk)，并将多个层块分配给每个 GPU。这种调度策略可以进一步减少气泡，提高流水线效率。
* **灵活的并行配置:**  PTD-P 允许用户根据模型结构和硬件资源灵活配置各种并行技术的组合方式，例如可以只使用张量并行和数据并行，也可以同时使用流水线并行、张量并行和数据并行。

传统的流水线并行通常将模型划分为连续的层块，每个 GPU 负责一个连续的层块。PTD-P 的 interleaved 1F1B 调度则将模型划分为多个不连续的层块，例如，GPU 1 负责层 1, 2, 9, 10，GPU 2 负责层 3, 4, 11, 12，依此类推。每个 GPU 负责多个不连续的层块，可以更有效地利用 GPU 资源，减少气泡开销。

{{< figure
    src="PTD-P.png"
    caption="Fig. 11.(Top) Default 1F1B pipeline schedule as in PipeDream-flush.(Bottom) Interleaved 1F1B pipeline schedule. First model chunks are in dark colors and second chunks are in light colors. (Image source: [Narayanan et al. 2021](https://arxiv.org/abs/2104.04473))"
    align="center"
    width="100%"
>}}  


## 混合专家模型

**混合专家模型(Mixture-of-Experts, MoE)**([Shazeer et al. 2017](https://arxiv.org/abs/1701.06538)) 是一种稀疏激活模型，它通过结合多个独立的“专家”网络和一个门控网络(Gating Network)，在不显著增加计算成本的前提下，大幅提升了模型的参数量和性能。MoE 的核心思想是**稀疏激活(Sparse Activation)**，即对于每个输入样本，仅激活部分专家网络，而不是整个模型。这种方法既提高了计算效率，又增强了模型的表达能力，使其在 LLMs 中表现出色。

MoE 设计灵感来源于[集成学习(Ensemble learning)](https://en.wikipedia.org/wiki/Ensemble_learning), 一种将复杂任务分解为多个子任务并由不同模型协作完成的技术。在 MoE 中，这些“子任务”由多个独立的专家网络处理，而门控网络则负责根据输入样本的特征动态选择最适合的专家。这种分工合作的机制类似于人类社会中的专家团队：不同领域的专家针对特定问题提供专业意见，最终综合得出结果。


{{< figure
    src="moe.png"
    caption="Fig. 12. Illustration of a mixture-of-experts(MoE) layer. Only 2 out of experts are selected and activated by the gating network. (Image source: [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538))"
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

为了避免门控网络过度偏向少数专家，MoE 引入了**辅助损失(Auxiliary Loss)**([Shazeer et al. 2017](https://arxiv.org/abs/1701.06538))，鼓励所有专家被均匀使用。一种常用方法是基于专家使用率的[变异系数(Coefficient of Variation, CV)](https://en.wikipedia.org/wiki/Coefficient_of_variation)的平方：

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
    caption="Fig. 13. Pseudo code of the group-level top-2 gating mechanism with auxiliary loss in GShard. (Image source: [Lepikhin et al. 2020](https://arxiv.org/abs/2006.16668))"
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
    caption="Fig. 14. Switch transformer. The sparse switch FFN layer is in the blue boxes. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
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
    caption="Fig. 15. Fine-tuning regularization results. A sweep of dropout rates while fine-tuning Switch Transformer models pre-trained on 34B tokens of the C4 data set(higher numbers are better). (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}  

Switch Transformers 论文中使用下图直观的展示了使用不同的并行技术如何分割模型权重和数据:

{{< figure
    src="switch_transformer_parallelism.png"
    caption="Fig. 16. An illustration of various parallelism strategies on how(Top) model weights and(Bottom) data are split over multiple GPU cores. In the top row, each color denotes a unique weight matrix. In the bottom row, different colors indicate different sets of tokens. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
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

## 序列并行

**序列并行(Sequence Parallelism, SP)** 是针对长序列模型(如 Transformer)提出的一种并行化策略，通过在序列维度上对输入进行划分，大幅降低激活内存占用并提高训练效率。它常与数据并行、张量并行或流水线并行结合使用，尤其适合处理超长文本或其他序列数据。

### Colossal-AI 序列并行

{{< figure
    src="colossal_sp.png"
    caption="Fig. 17. The overall architecture of the proposed sequence parallelism and existing parallel approaches. For sequence parallelism, Device 1 and Device 2 share the same trainable parameters. (Image source: [Li, et al. 2021](https://arxiv.org/abs/2105.13120))"
    align="center"
    width="100%"
>}}  

自注意力(self-attention) 的计算复杂度和内存开销与序列长度 $s$ 的平方 $O(s^2)$ 成正比，长序列数据将增加中间 activation 内存使用量，从而限制设备的训练能力。**Colossal-AI 序列并行**([Li, et al. 2021](https://arxiv.org/abs/2105.13120))从系统角度提出**拆分超长序列到多卡**，具体的解决步骤如下。

1. **序列分块**  
   将输入序列划分为若干块，每个块由不同 GPU 保存和计算；因此每张卡只需存储自己对应的序列块激活，避免单卡内存爆炸。  
2. **环状通信 + 自注意力**  
   提出环自注意力(Ring Self-Attention, RSA) 机制：各 GPU 先本地计算局部注意力，然后依次向相邻 GPU 传递(环状结构)Key/Value 分块，多轮迭代后保证每个 GPU 能获取全局序列信息。  
3. **与其他并行方式结合**  
   不受注意力头数、层数等超参数限制，可配合数据并行、张量并行、流水线并行等技术，共同突破大规模模型的序列长度限制。

{{< figure
    src="ring_self_attention.png"
    caption="Fig. 18. Ring Self-Attention. (Image source: [Li, et al. 2021](https://arxiv.org/abs/2105.13120))"
    align="center"
    width="100%"
>}}  

### Megatron-LM 序列并行

**Megatron-LM**([Shoeybi et al. 2019](https://arxiv.org/pdf/1909.08053)) 原本使用张量并行分担部分激活值，但 Transformer 中的 LayerNorm、Dropout 等操作的激活值仍需完整保存在单卡，显存消耗依旧庞大。因此 NVIDIA 提出 Megatron-LM 序列并行([Korthikanti, et al. 2022](https://arxiv.org/abs/2205.05198))在序列维度对这些**激活值进行切分**，大幅降低占用。

{{< figure
    src="Megatron-LM-transformer-sp.png"
    caption="Fig. 19. Transformer layer with tensor and sequence parallelism. (Image source: [Korthikanti, et al. 2022](https://arxiv.org/abs/2205.05198))"
    align="center"
    width="100%"
>}}  

{{< figure
    src="Megatron-LM-mlp-sp.png"
    caption="Fig. 20. MLP layer with tensor and sequence parallelism. (Image source: [Korthikanti, et al. 2022](https://arxiv.org/abs/2205.05198))"
    align="center"
    width="100%"
>}}  

1. **序列维度切分**  
   针对 LayerNorm、Dropout 等难以在张量维度切分的激活，将其沿序列维度划分，使每个 GPU 只处理一部分序列的非线性操作。  
2. **张量并行仍保留**  
   注意力(Attention)、MLP 等线性操作继续使用张量并行；序列并行的激活需要在前后进行对应的 All-Gather 或 Reduce-Scatter 以交换数据。  
3. **选择性激活重计算(Selective Activation Recomputation)**  
   针对部分计算量小但激活量大的操作，选择在反向传播时临时重算，以进一步节省显存。

### DeepSpeed-Ulysses 序列并行

**DeepSpeed-Ulysses**([Jacobs et al. 2023](https://arxiv.org/abs/2309.14509)) 针对超长序列训练提出了一种高效的序列并行方案，通过在序列维度对输入进行划分，并结合两阶段的全对全通信，有效降低通信量和激活内存，从而支持训练百万 token 级别的长序列 Transformer 模型。

{{< figure
    src="deepspeed_sp.png"
    caption="Fig. 21. DeepSpeed sequence parallelism(DeepSpeed-Ulysses) design. (Image source: [Jacobs et al. 2023](https://arxiv.org/abs/2309.14509))"
    align="center"
    width="100%"
>}}  

1. **序列划分 + 全对全通信**  
   将输入序列沿序列维度划分到 $P$ 张 GPU 上，每个 GPU 只处理局部 $N/P$ 的序列；在注意力计算前，通过 All-to-All 操作交换查询($Q$)、键($K$)和值($V$)，使得每个 GPU 获得完整序列信息，但仅计算分配到的注意力头。

2. **双阶段通信优化**  
   - **第一次 All-to-All：** 在注意力计算前对 $Q$/$K$/$V$ 进行全对全交换，分散激活计算并降低每卡内存压力；  
   - **第二次 All-to-All：** 在注意力计算后收集输出上下文，将其重新映射为局部序列分区，既恢复原始序列结构，又显著减少了通信数据量。

3. **高效通信与通用性**  
   利用全对全通信，使得通信量降为 $O(N/P)$，相比传统的 All-Gather 方法(通信量 $O(N)$)节省了近 $P$ 倍的带宽；同时，该方案适用于密集和稀疏注意力，并可与 ZeRO-3 内存优化无缝集成，从而支持更大模型和更长序列的高效训练。

{{< figure
    src="deepspeed_ulysses_compare.png"
    caption="Fig. 22. DeepSpeed-Ulysses vs Megatron LM. (Image source: [DeepSpeed Blogs](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md))"
    align="center"
    width="100%"
>}}  

- 在 64 卡 A100 环境下，吞吐量较 Megatron-LM 序列并行最高提升 2.5 倍，并可处理更长序列(百万级 token)；  
- 收敛性能与原模型无差别，可轻松集成到 Megatron-DeepSpeed 框架。

## 优化器相关的并行：ZeRO

**ZeRO(Zero Redundancy Optimizer)**([Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054))是一种旨在消除训练大型模型时内存冗余的优化器并行技术。大型模型训练的内存主要消耗在两大部分：

- **模型状态(Model States)：** 包括**优化器状态**(如 Adam 的动量和二阶矩)、**梯度**和**模型参数**。混合精度训练不仅需要存储 FP16 数据，还需保留 FP32 版本的参数和状态，导致内存占用更高。
- **激活值、临时缓冲区与内存碎片(Residual States)：** 这些数据在前向传播和反向传播中仅被使用一次，但同样会占用大量内存。

为了解决内存冗余问题，ZeRO 采用了两大策略：

1. **ZeRO-DP(Data Parallelism)：**  
   针对模型状态，通过将优化器状态、梯度和参数分片分布到多个数据并行进程中，消除冗余，同时利用动态通信调度减少通信量。

2. **ZeRO-R(Residuals Optimization)：**  
   针对激活值和临时缓冲区，采用分片激活值重计算、固定缓冲区大小以及实时内存碎片整理等方法优化内存使用。

### ZeRO 分片策略

ZeRO 分为三个阶段，每个阶段在前一阶段基础上进一步减少内存冗余，从而使得训练超大模型成为可能：

#### ZeRO-1(优化器状态分片)
- **原理：**  
  - 将优化器状态(如 Adam 的动量和二阶矩)沿参数维度分为 $P$ 个分片($P$ 为 GPU 数量)，每个 GPU 只存储与其负责模型参数对应的状态。  
  - 局部更新：每个 GPU 在参数更新阶段仅更新其本地存储的状态和参数分片，无需额外跨 GPU 通信。

#### ZeRO-2(梯度分片) 
- **原理：**  
  - 在优化器状态分片的基础上，将梯度沿参数维度同样进行分片，每个 GPU 只存储对应梯度分片。  
  - 每个 GPU 计算局部梯度，利用高效的 Reduce-Scatter 操作进行梯度聚合，再更新本地参数分片。

#### ZeRO-3(参数分片)
- **原理：**  
  - 在 ZeRO-1 和 ZeRO-2 的基础上，将模型参数(通常为 16 位数据)也进行分片，每个 GPU 只存储与其对应的参数分片。  
  - 按需参数收集：在前向或反向传播过程中，若某个 GPU 需要完整的模型参数，则从其他 GPU 收集缺失的分片，这一过程仅在必要时进行，以减少通信开销。


下图展示了不同阶段下每个设备上模型状态内存消耗的对比情况：

{{< figure
    src="deepspeed_zero.png"
    caption="Fig. 23. Comparing the per-device memory consumption of model states, with three stages of ZeRO-DP optimizations. (Image source: [Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054))"
    align="center"
    width="100%"
>}}


### DeepSpeed ZeRO 分片与 Offload 策略对比

为了更好地理解 DeepSpeed 的 ZeRO 策略，以下对各阶段及 Offload 方案进行对比：


| **ZeRO Stage** | **描述** | **显存占用** | **训练速度** |
|----------------|----------|--------------|--------------|
| **ZeRO-0**     | 纯数据并行，不进行任何分片，所有状态在每个 GPU 上完全复制。 | 最高 | **最快** |
| **ZeRO-1**     | 仅分片优化器状态，梯度和参数仍复制。 | 较高 | 略慢于 ZeRO-0 |
| **ZeRO-2**     | 分片优化器状态和梯度。 | 中等 | 慢于 ZeRO-1 |
| **ZeRO-3**     | 分片优化器状态、梯度和模型参数。 | 最低 | 明显慢于 ZeRO-2，受模型规模和网络带宽影响 |


| **Offload 类型**                | **描述** | **显存占用** | **训练速度** |
|----------------------------------|----------|--------------|--------------|
| **ZeRO-1 + CPU Offload**         | 在 ZeRO-1 基础上，将优化器状态卸载到 CPU 内存，降低 GPU 显存占用，但依赖 PCIe 带宽且占用 CPU 内存。 | 中偏低 | 慢于 ZeRO-1 |
| **ZeRO-2 + CPU Offload**         | 在 ZeRO-2 基础上，将优化器状态卸载到 CPU 内存，对大模型进一步降低 GPU 显存，但增加 CPU–GPU 数据传输。 | 较低 | 慢于 ZeRO-2 |
| **ZeRO-3 + CPU Offload**         | 在 ZeRO-3 基础上，将优化器状态和模型参数卸载到 CPU，GPU 显存占用最低，但 CPU–GPU 通信开销极大。 | **极低** | **非常慢** |
| **ZeRO-Infinity(NVMe Offload)** | 基于 ZeRO-3，将状态卸载到 NVMe 设备，突破 CPU 内存限制，适合超大模型；性能高度依赖 NVMe 并行读写速度。 | **极低**<br/>需 NVMe 支持 | 慢于 ZeRO-3，但通常优于 CPU Offload 方案 |

### 通信量与性能影响

- **ZeRO-0/1/2：**  
  主要依赖 All-Reduce 进行梯度同步，通信量相对较低。
  
- **ZeRO-3：**  
  需要对模型参数进行 All-Gather/All-Reduce 操作，通信量显著增加，网络带宽成为关键瓶颈。

- **Offload 策略(CPU/NVMe)：**  
  数据传输主要在 CPU ↔ GPU 或 NVMe ↔ GPU 之间，传输带宽远低于 GPU 之间的通信，可能显著影响训练速度，尤其在 ZeRO-3 场景下更为明显。


## 多维度并行

**多维度并行(Multi-dimensional Parallelism)** 是指在分布式训练中将数据并行、模型并行和流水线并行等多种并行技术有机结合，以充分利用现代 GPU 集群的计算资源。通过这种“3D 并行”或“4D 并行”策略，不仅能提高内存效率，还能提升计算效率，从而实现超大规模(甚至万亿参数级别)模型的高效训练。

### 3D 并行

随着 GPU 集群计算能力的迅速提升，训练万亿参数级别的模型不再遥不可及。DeepSpeed 将数据并行、模型并行与流水线并行三种技术融合，构建了一种“3D 并行”策略。该策略主要解决训练超大模型所面临的两大挑战：

- **内存效率：**  
  模型层被划分到不同的流水线阶段，每个阶段内部又通过模型并行进一步分割，减少了模型、优化器和激活值占用的内存量。但需要注意，模型分割不能无限制进行，否则通信开销会显著增加，进而影响计算效率。

- **计算效率：**  
  为了让计算工作者数量超越单纯模型和流水线并行的限制，同时保证计算效率，DeepSpeed 借助 ZeRO-DP(基于优化器状态分片的数据并行)进行扩展。ZeRO-DP 不仅进一步优化内存使用，还通过拓扑感知映射将数据并行组分配到局部高带宽通信的设备上，极大降低了通信开销。

下面的图示展示了 3D 并行的整体策略：

{{< figure
    src="zero_3d.png"
    caption="Fig. 24. Example 3D parallelism with 32 workers. Layers of the neural network are divided among four pipeline stages. Layers within each pipeline stage are further partitioned among four model parallel workers. Lastly, each pipeline is replicated across two data parallel instances, and ZeRO partitions the optimizer states across the data parallel replicas. (Image source: [Majumder et al. 2020](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/))"
    align="center"
    width="100%"
>}}

每个并行维度(数据、模型、流水线)均经过精心映射，以充分利用节点内和节点间的通信带宽。具体策略包括：  
- **优化节点内通信：** 由于模型并行的通信开销最大，优先将模型并行组安排在同一节点内，以利用较高的节点内带宽(例如采用 NVIDIA Megatron-LM 的张量切分方式)；  
- **数据并行与流水线并行：** 当模型并行不覆盖整个节点时，数据并行组尽可能安排在同一节点内；而流水线并行由于通信量较小，可灵活安排跨节点调度。  

通过减少每个数据并行组中通信数据量以及提高局部并行通信的并行度，整体通信带宽得到有效放大。

{{< figure
    src="3d_parallelism.png"
    caption="Fig. 25. Mapping of workers in Figure 24 to GPUs on a system with eight nodes, each with four GPUs. Coloring denotes GPUs on the same node. (Image source: [Majumder et al. 2020](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/))"
    align="center"
    width="100%"
>}}

### 4D 并行

为了进一步扩展模型规模，Llama3([Grattafiori et al. 2024](https://arxiv.org/abs/2407.21783)) 训练的时候采用了 **4D 并行**，它结合了四种并行方法，将模型进行更细粒度的分片，使每个 GPU 上的模型参数、优化器状态、梯度和激活值均能适配高带宽内存(HBM)的容量限制。这四种并行方法分别是：

- **张量并行(Tensor Parallelism, TP)：** 将单个权重张量划分为多个块，分布在不同设备上；
- **流水线并行(Pipeline Parallelism, PP)：** 将模型垂直划分为多个阶段，各阶段在不同设备上并行处理不同微批次；
- **上下文并行(Context Parallelism, CP)：** 将输入上下文划分成多个段，从而缓解长序列输入时的内存瓶颈；
- **数据并行(Data Parallelism, DP)，通常采用完全分片的数据并行(FSDP)：** 对模型、优化器状态和梯度进行分片，并在每个训练步后同步。

下面的图示展示了 4D 并行在 16 个 GPU 上的实现示例，各 GPU 的位置用向量 [D1, D2, D3, D4] 表示，其中每个维度对应一种并行策略。GPU 按照 [TP, CP, PP, DP] 四个维度分组，每个维度的组内大小均为 2。例如，GPU0 和 GPU1 属于同一张量并行组；GPU0 和 GPU2 属于同一上下文并行组；GPU0 和 GPU4 属于同一流水线并行组；GPU0 和 GPU8 属于同一数据并行组：

{{< figure
    src="4d_parallelism.png"
    caption="Fig. 26. Illustration of 4D parallelism. (Image source: [Grattafiori et al. 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="100%"
>}}

通过 4D 并行策略，Llama3 在训练时能够充分利用多个 GPU 的计算资源，同时有效减少内存占用，支持训练超大规模的模型。


## 内存优化技术

除了并行训练技术，还有许多内存优化技术设计可以帮助训练 LLMs，这些设计主要从减少训练过程中各个环节的内存占用入手。

### CPU Offloading

**CPU Offloading**([Rhu et al. 2016](https://arxiv.org/abs/1602.08124)) 是指当 GPU 内存不足时，将暂时无需使用的数据或者张量卸载到 CPU 并在需要时再加载回 GPU是一种常见且直观的做法。它的主要目的是利用 CPU 内存更大的容量来扩展可用空间，从而在显存受限的环境下也能训练更大规模的模型。然而，这种方法会带来额外的数据传输开销，通常会降低训练速度，因此近年来应用相对减少。

1. **识别可卸载张量:**  识别训练过程中暂时不使用的张量，例如模型参数、优化器状态、中间激活值等。判断张量是否可以卸载的依据可以是张量的使用频率、生命周期等。
2. **张量卸载:** 将可卸载的张量从 GPU 内存移动到 CPU 内存。数据传输通常通过 PCIe 总线进行。
3. **张量预取(Prefetching):**  在需要使用卸载到 CPU 内存的张量之前，提前将张量从 CPU 内存加载回 GPU 内存。预取操作可以与 GPU 的计算操作并行进行，以减少数据加载的延迟。
4. **张量使用:**  GPU 使用加载回 GPU 内存的张量进行计算。
5. **张量再次卸载:**  在张量使用完毕后，如果张量在一段时间内不再需要使用，可以再次将其卸载到 CPU 内存，释放 GPU 内存空间。

ZeRO-Offload 和 ZeRO-Infinity 是 DeepSpeed 库中实现的基于 CPU 卸载的内存优化技术。ZeRO-Offload 将优化器状态卸载到 CPU 内存，ZeRO-Infinity 更进一步，将模型参数也卸载到 CPU 内存甚至 NVMe 磁盘，突破 GPU 内存墙，支持训练更大规模的模型。

下图直观的展示了**异构系统(Heterogenous system)** 内存优化技术：

{{< figure
    src="heterogenous_system.png"
    caption="Fig. 27. Heterogenous system illustration. (Image source: [Clolossal-AI Documentation](https://colossalai.org/docs/concepts/paradigms_of_parallelism))"
    align="center"
    width="100%"
>}}

### 激活重计算/梯度检查点

**激活重计算/梯度检查点(Activation Recomputation/Gradient Checkpointing)**([Chen et al. 2016](https://arxiv.org/abs/1604.06174)) 是一种**以计算换内存的技术**。在训练过程中，只保存部分激活值(例如每个 Transformer 层的输入激活值)，在反向传播时，重新计算未保存的激活值。激活重计算可以显著减少训练过程中的激活值内存占用，尤其是在训练深层神经网络时效果明显。

1. **选择检查点:** 选择模型中的一些层作为 checkpoint。通常选择模型中的关键层，例如 Transformer 层的输入层。  
2. **前向传播(Forward Pass):** 在前向传播过程中，只保存检查点层的激活值，对于非检查点层的激活值，在计算完成后立即释放，不进行保存。  
3. **反向传播(Backward Pass):** 在反向传播过程中，当需要计算某个非检查点层的梯度时，首先重新进行一次前向传播，计算该层的激活值(重计算)，然后再进行反向传播计算梯度。对于检查点层，由于已经保存了检查点层的激活值，可以直接使用保存的激活值进行反向传播，无需重新计算。  

下面进行激活重计算的内存成本分析。为方便分析，假设模型共有 $n$ 层网络结构，并将其**均匀**地划分为 $k$ 个分段(segment)。这样，每个分段大约包含 $n/k$ 层网络。做激活重计算时，我们只保存各分段边界处(即检查点)的激活值，其余需要时再重算。下面函数表示训练过程中的内存需求：

$$
\text{cost-total} \;=\; \max_{i=1,\ldots,k}\bigl[\text{cost-of-segment}(i)\bigr] \;+\; O(k)
\;=\; O\Bigl(\tfrac{n}{k}\Bigr) \;+\; O(k).
$$

接下来考虑在给定 $n$ 的前提下，如何选择最优 $k$ 以最小化

$$
f(k) \;=\; \frac{n}{k} \;+\; k.
$$

对 $f(k)$ 关于 $k$ 求导并令其为 0(只考虑 $k>0$)：

$$
f'(k) \;=\; -\frac{n}{k^2} \;+\; 1 \;=\; 0 
\quad\Longrightarrow\quad
k^2 = n 
\quad\Longrightarrow\quad
k = \sqrt{n}.
$$

将 $k = \sqrt{n}$ 代入，可得最小内存开销大约为

$$
f(\sqrt{n}) \;=\; \frac{n}{\sqrt{n}} \;+\; \sqrt{n} 
\;=\; 2\sqrt{n}.
$$

因此，该做法的总体峰值内存需求可降到 $O(\sqrt{n})$ 的量级(对比一般直接保存所有激活的 $O(n)$ 内存)，这就是激活重计算技术能带来“亚线性”内存占用的原因。下图直观的展现了这个 trick 的效果。

{{< figure
    src="activation_recomputation.png"
    caption="Fig. 28. The memory cost of different memory saving algorithms. Sharing: Memory used by intermediate results is recycled when no longer needed. Inplace: Save the output directly into memory of an input value. (Image source: [Chen et al. 2016](https://arxiv.org/abs/1604.06174))"
    align="center"
    width="100%"
>}}  


需要注意，激活重计算需要在**反向传播**阶段额外进行前向重算，每个分段要做一次到 $n/k$ 层的前向计算。若将网络分为 $k$ 个分段，反向传播时重算总计约 $k \times \bigl(n/k\bigr) = n$ 层的前向操作，相当于每个训练迭代中整体多做了大约一次“前向传播”。这在 LLM 训练中通常能被接受，原因是：

- 比起快速耗尽 GPU 显存导致无法训练大规模模型，这种在计算上的额外代价通常更容易承受。  
- 当模型十分深($n$ 很大)时，用激活重计算技术可以将内存使用从 $O(n)$ 显著降至 $O(\sqrt{n})$，使得更多、更深的大模型能够在给定硬件上进行训练。


### 混合精度训练

**混合精度训练(Mixed Precision Training)**([Micikevicius al. 2017](https://arxiv.org/abs/1710.03740))是一种在模型训练过程中同时利用低精度浮点数(如 FP16 或 BF16)与高精度浮点数(如 FP32)的技术，其核心目标是在**减少显存占用**、**加速训练**的同时，尽可能保持与全精度训练相当的模型精度。

现代 GPU 在低精度计算上具有更高的吞吐量和更低的显存占用，从而降低访存开销与内存带宽需求，使混合精度训练能显著提升训练速度。下图展示了一个网络层中混合精度训练的基本流程：前向和反向传播主要采用半精度(FP16)运算，而在梯度累积与参数更新时使用全精度(FP32)，以规避低精度计算可能带来的数值精度问题。

{{< figure
    src="mixed_precision.png"
    caption="Fig. 29. Mixed precision training iteration for a layer. (Image source: [Micikevicius al. 2017](https://arxiv.org/abs/1710.03740))"
    align="center"
    width="100%"
>}}

混合精度训练主要依赖以下三项关键技术：

1. **权重全精度主副本**  
   为防止梯度在 FP16 下因幅度过小而被截断为零，训练过程中保持一份 FP32 的权重主副本。具体流程为：  
   - **初始化：** 使用 FP32 权重作为模型的主副本；  
   - **前向/反向传播：** 每次迭代开始前，将 FP32 权重转换为 FP16 用于前向传播和反向传播，计算得到 FP16 梯度；  
   - **参数更新：** 在更新参数前，将 FP16 梯度转换为 FP32，并用其更新 FP32 主副本。  

   这种设计既能利用低精度计算的高效性，又确保了参数更新的准确性。

2. **损失缩放**  
为避免因低精度表示范围受限而导致梯度下溢，通常在反向传播前对损失值进行放大处理。具体流程为：  
- 使用 FP32 计算损失 $L$；  
- 将损失乘以缩放因子 $S$，得到 $L' = L \times S$ 后进行反向传播，计算出 FP16 梯度；  
- 在参数更新前，将梯度除以 $S$ 还原为真实梯度。  

缩放因子的选择十分关键：过小可能无法避免梯度下溢，过大则有可能引起梯度上溢。动态损失缩放技术可以根据训练过程中梯度的实际情况自动调整缩放因子。

如下图所示，通过放大损失使梯度分布更集中于较高数值部分，从而保留那些在低精度表示下可能被截断的细微信息。

{{< figure
    src="mixed_precision_fp16.png"
    caption="Fig. 30. The histogram of gradients in full precision. The left part up to $2^{-24}$ will be zero-ed off once the model switches to FP16. (Image source: [Micikevicius al. 2017](https://arxiv.org/abs/1710.03740))"
    align="center"
    width="100%"
>}}

3. **算术精度控制**  
   对于对精度要求较高的运算(如向量点积和求和归约)，可采用 FP32 进行累积计算，然后再转换为 FP16 存储；而对于逐元素运算，则可根据具体需求选择使用 FP16 或 FP32。


### 压缩

在深度学习训练过程中，中间结果(如激活值和梯度信息)虽然仅在一次前向传播和一次反向传播中使用，但往往占用大量内存。考虑到两次使用之间存在明显的时间间隔，可以在第一次使用后对数据进行**压缩(Compression)**，待后续需要时再解压缩，从而有效降低内存占用。

压缩技术主要应用于两个场景：

- **激活值压缩：** 前向传播后对激活值进行压缩，反向传播前解压缩。这对深层神经网络尤为重要，因为激活值通常占用大量内存。
- **梯度压缩：** 在反向传播计算梯度后、梯度同步前对梯度进行压缩，减少跨 GPU 通信的数据量，从而提高分布式训练效率。

压缩技术可以分为两类：

1. **无损压缩(Lossless Compression):**  
   采用如 Huffman 编码或 Lempel-Ziv 算法等方法，确保解压缩后的数据与原始数据完全一致。但由于压缩率较低，其内存节省效果有限。

2. **有损压缩(Lossy Compression):**  
   使用如 JPEG 或 MPEG 等算法，在允许一定数据损失的前提下获得更高的压缩率。这种方法能显著降低内存占用，但可能对模型精度和收敛性产生一定影响。

**Gist**([Jain et al. 2018](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf))是一种用于激活值压缩的内存优化技术，其核心在于利用数据编码策略压缩中间结果，主要包含两种编码方案：

- **层特定无损编码(Layer-Specific Lossless Encoding):**  
  针对特定层结构(例如 ReLU-Pool 与 ReLU-Conv)，设计专门的无损编码方案：  
  - 对于 ReLU-Pool 层，可采用二值化编码；  
  - 对于 ReLU-Conv 层，则使用稀疏存储与稠密计算编码。

- **激进有损编码(Aggressive Lossy Encoding):**  
  采用 **延迟精度降低(Delayed Precision Reduction, DPR)** 技术。DPR 的核心思想是：**激活值在前向传播时需保持高精度，而在反向传播时可容忍较低精度**。因此，在前向传播后将激活值压缩到较低精度，反向传播前再解压至高精度。

### 内存高效优化器

传统优化器(如 Adam、SGD with Momentum)在训练过程中需要为每个模型参数维护大量状态数据(例如 momentum 和 variance)，其内存占用往往与模型参数量相当甚至更高。例如，以 **Adam 优化器**([Kingma et al. 2014](https://arxiv.org/pdf/1412.6980))为例，每个参数需要存储一阶矩和二阶矩，与参数本身及其梯度加起来，整个训练过程**大约需要 4 倍于模型权重的内存**，这对大型模型训练构成了严峻挑战。

为降低内存消耗，**内存高效优化器**主要通过以下策略进行设计：
- **减少状态变量数量：** 只保存必要的统计信息，而非完整矩阵；
- **降低状态变量精度：** 采用 FP16 或 bfloat16 存储；
- **共享状态变量：** 在多个参数间共享部分状态信息。

#### Adafactor

**Adafactor**([Shazeer et al. 2018](https://arxiv.org/abs/1804.04235)) 是一种内存高效的自适应学习率优化器。与 Adam 不同，Adafactor 不存储完整的二阶矩估计矩阵，而是只存储两个向量(行、列统计)替代完整的二阶矩矩阵，显著降低了内存占用，特别适用于参数矩阵具有低秩结构的场景。

#### SM3

**SM3(Sparse Momentum for Massive Models)**([Anil et al. 2019](https://arxiv.org/abs/1905.11286)) 通过稀疏更新和状态共享，提供了一种同样内存高效的自适应优化方案。

- **稀疏 Momentum：** 只对梯度非零的参数更新 Momentum，从而减少计算和存储开销；
- **状态共享：** 在一定程度上允许不同参数共享状态变量，进一步降低内存消耗；
- **自适应学习率：** 根据各参数梯度动态调整学习率，提高了模型训练的稳定性和收敛速度。


### LoRA

**LoRA (Low-Rank Adaptation)**([Hu et al. 2021](https://arxiv.org/abs/2106.09685)) 提出在预训练权重旁引入 **低秩适配器** 的方法，通过添加少量参数实现高效微调，同时保持预训练模型原有的推理能力。

下图直观展示了 LoRA 的原理和初始化策略：

{{< figure
    src="lora.png"
    caption="Fig. 31. An illustration of LoRA. (Image source: [Hu et al. 2021](https://arxiv.org/abs/2106.09685))"
    align="center"
    width="70%"
>}}

在标准前向传播中，模型输出为

$$
h = W_0 x,
$$

而引入 LoRA 后，输出变为

$$
h = W_0 x + \Delta W x = W_0 x + B A x.
$$

其中：
- **$A \in \mathbb{R}^{r \times k}$（降维矩阵）**：将输入从 $k$ 维映射到更低的 $r$ 维；  
- **$B \in \mathbb{R}^{d \times r}$（升维矩阵）**：将降维后的特征从 $r$ 维映射回原来的 $d$ 维；  
- **输入 $x$：** 维度为 $\mathbb{R}^{k}$；  
- **原始权重 $W_0$：** 维度为 $\mathbb{R}^{d \times k}$，因而 $W_0 x \in \mathbb{R}^{d}$；  

假设预训练权重矩阵为  
$$
\mathbf{W} \in \mathbb{R}^{d \times k},
$$  
LoRA 在其上添加低秩更新项，从而得到新的权重表示：

$$
\mathbf{W}' = \mathbf{W} + \alpha\, \mathbf{B}\mathbf{A},
$$

其中：  
- **$A \in \mathbb{R}^{r \times k}$（降维矩阵）**：将输入从 $k$ 维映射到更低的 $r$ 维；  
- **$B \in \mathbb{R}^{d \times r}$（升维矩阵）**：将降维后的特征从 $r$ 维映射回原来的 $d$ 维；  
- **$r \ll \min(d, k)$（低秩维度）**：通常取值为 $4$ 到 $16$，在保证模型表达能力的同时尽量减少新增参数；  
- **$\alpha$（缩放因子）**：用于放大低秩更新参数 $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$，补偿低秩分解带来的数值幅度较小的问题（通常设置为 $\alpha = 2 \times r$，例如当 $r = 8$ 时，$\alpha = 16$）。

在微调过程中，**原始权重 $\mathbf{W}$ 被冻结**，只更新 $\mathbf{A}$ 和 $\mathbf{B}$，因而大大减少了训练和存储的参数量。

为了确保微调初期引入的更新项 $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ 对原模型的影响尽量小，通常采用以下初始化策略：

1. **降维矩阵 $\mathbf{A}$ 的初始化**  
   - **高斯初始化**：令 $\mathbf{A} \sim \mathcal{N}(0,\sigma^2)$（一般 $\sigma$ 取较小值，如 0.02），保证初始更新量足够小，从而不至于严重干扰模型输出。  
   - **Kaiming(He) 初始化**：Kaiming 初始化是一种专为深层网络设计的权重初始化方法，其目标是保持前向信号和反向梯度在网络层之间的稳定性。对于 LoRA，只要确保尺度较小(或配合合适的缩放因子)，即可使初始 $\Delta \mathbf{W}$ 接近零。

2. **升维矩阵 $\mathbf{B}$ 的初始化**  
   - 通常将 $\mathbf{B}$ 初始化为全零矩阵，这样初始时即有 $\mathbf{B}\mathbf{A} = 0$，进一步降低对原模型影响。

采用 LoRA 进行训练具备以下优势：

- **参数高效**：仅引入低秩适配器参数，减少了需要训练和存储的总参数量。  
- **显存与计算效率**：冻结大部分预训练权重，微调过程中仅更新小规模参数，显著降低了显存占用与算力开销。  
- **无额外推理时延**：训练完成后，可将更新项 $\Delta \mathbf{W}$ 合并回原始权重，从而在推理阶段不会增加额外计算量。  
- **模块选择灵活性**：通过 `--lora_target` 或 `--lora-target` 参数，可以指定仅在特定线性模块上应用 LoRA 更新。支持的目标模块包括： ```q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj```, 这种设计允许用户根据具体任务需求，针对性地选择关键模块进行调优，从而进一步提高微调效率和适应性。

### QLoRA

**QLoRA**([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) 是在 LoRA 基础上结合量化思想对大规模模型进行高效微调的一种方法。它通过以下三个关键改进，大幅降低显存占用，同时保持模型精度基本不变：

1. **4 位标准浮点数(NF4) 量化**  
   采用基于分块的分位量化策略，将原始模型权重量化为 4 位，从而在细微损失精度的情况下实现显著的存储压缩。

2. **双重量化(Double Quantization)**  
   在对普通参数进行一次量化后，再对量化常数进行一次额外的量化，从而进一步减小缓存占用。

3. **分页优化器(Paged Optimizer)**  
   当显存使用过高时，自动将部分优化过程转移到 CPU 内存，从而减轻 GPU 显存压力，提升可伸缩性。

与传统的 LoRA 仅减少需微调参数数量不同，QLoRA 还通过 4 位量化来**压缩**所有权重，从而在保证接近原有精度的同时，最大限度减少显存占用和数据传输开销。

{{< figure
    src="qlora.png"
    caption="Fig. 32. Different finetuning methods and their memory requirements. QLoRA improves over LoRA by quantizing the transformer model to 4-bit precision and using paged optimizers to handle memory spikes. (Image source: [Dettmers et al. 2023](https://arxiv.org/abs/2305.14314))"
    align="center"
    width="100%"
>}}

这种方法可以看作是对 LoRA 的进一步扩展：LoRA 通过减少需要微调的权重数量来提升效率，而 QLoRA 则在此基础上，将所有权重(包括未微调的部分)量化到 4 位精度，在总体上实现**存储与计算的双重压缩**，适合对 LLM 进行资源受限环境下的高效微调。


## 总结

并行技术和内存优化策略需要根据具体的模型结构、数据集规模、硬件资源和训练目标进行权衡和选择。通常情况下，需要将多种技术结合使用，才能有效地训练大规模的模型，并取得最佳的性能和效率。

## 参考文献

[1] Weng, Lilian, and Greg Brockman. ["Techniques for training large neural networks."](https://openai.com/blog/techniques-for-training-large-neural-networks/) OpenAI Blog, 2022.

[2] Shenggui Li, Siqi Mai. ["Paradigms of Parallelism."](https://colossalai.org/docs/concepts/paradigms_of_parallelism/) Colossal-AI Documentation, 2024.

[3] Li, Shen, et al. ["Pytorch distributed: Experiences on accelerating data parallel training."](https://arxiv.org/abs/2006.15704) arXiv preprint, 2020.

[4] Li, Mu, et al. ["Communication efficient distributed machine learning with the parameter server."](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf) Advances in Neural Information Processing Systems 27, 2014.

[5] Huang, Yanping, et al. ["Gpipe: Efficient training of giant neural networks using pipeline parallelism."](https://arxiv.org/abs/1811.06965) Advances in Neural Information Processing Systems 32, 2019.

[6] Harlap, Aaron, et al. ["Pipedream: Fast and efficient pipeline parallel dnn training."](https://arxiv.org/abs/1806.03377) arXiv preprint, 2018.

[7] Narayanan, Deepak, et al. ["Memory-efficient pipeline-parallel dnn training."](https://arxiv.org/abs/2006.09503) International Conference on Machine Learning, PMLR, 2021.

[8] Shoeybi, Mohammad, et al. ["Megatron-lm: Training multi-billion parameter language models using model parallelism."](https://arxiv.org/abs/1909.08053) arXiv preprint, 2019.

[9] Narayanan, Deepak, et al. ["Efficient large-scale language model training on gpu clusters using megatron-lm."](https://arxiv.org/abs/2104.04473) Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2021.

[10] Shazeer, Noam, et al. ["Outrageously large neural networks: The sparsely-gated mixture-of-experts layer."](https://arxiv.org/abs/1701.06538) arXiv preprint, 2017.

[11] Lepikhin, Dmitry, et al. ["Gshard: Scaling giant models with conditional computation and automatic sharding."](https://arxiv.org/abs/2006.16668) arXiv preprint, 2020.

[12] Fedus, William, Barret Zoph, and Noam Shazeer. ["Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity."](https://arxiv.org/abs/2101.03961) Journal of Machine Learning Research 23.120, 2022: 1–39.

[13] Zhou, Yanqi, et al. ["Mixture-of-experts with expert choice routing."](https://arxiv.org/abs/2202.09368) Advances in Neural Information Processing Systems 35, 2022: 7103–7114.

[14] Li, Shenggui, et al. ["Sequence parallelism: Long sequence training from system perspective."](https://arxiv.org/abs/2105.13120) arXiv preprint, 2021.

[15] Korthikanti, Vijay Anand, et al. ["Reducing activation recomputation in large transformer models."](https://arxiv.org/abs/2205.05198) Proceedings of Machine Learning and Systems 5, 2023: 341–353.

[16] Jacobs, Sam Ade, et al. ["Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models."](https://arxiv.org/abs/2309.14509) arXiv preprint, 2023.

[17] DeepSpeed. ["DeepSpeed Ulysses README."](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md) GitHub repository.

[18] Rajbhandari, Samyam, et al. ["Zero: Memory optimizations toward training trillion parameter models."](https://arxiv.org/abs/1910.02054) SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, IEEE, 2020.

[19] Microsoft Research. ["DeepSpeed: Extreme-scale model training for everyone."](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) 2020.

[20] Dubey, Abhimanyu, et al. ["The llama 3 herd of models."](https://arxiv.org/abs/2407.21783) arXiv preprint, 2024.

[21] Rhu, Minsoo, et al. ["vDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design."](https://arxiv.org/abs/1602.08124) 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture(MICRO), IEEE, 2016.

[22] Chen, Tianqi, et al. ["Training deep nets with sublinear memory cost."](https://arxiv.org/abs/1604.06174) arXiv preprint, 2016.

[23] Micikevicius, Paulius, et al. ["Mixed precision training."](https://arxiv.org/abs/1710.03740) arXiv preprint, 2017.

[24] Jain, Animesh, et al. ["Gist: Efficient data encoding for deep neural network training."](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf) 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture(ISCA), IEEE, 2018.

[25] Kingma, Diederik P., and Jimmy Ba. ["Adam: A method for stochastic optimization."](https://arxiv.org/abs/1412.6980) arXiv preprint, 2014.

[26] Shazeer, Noam, and Mitchell Stern. ["Adafactor: Adaptive learning rates with sublinear memory cost."](https://arxiv.org/abs/1804.04235) International Conference on Machine Learning, PMLR, 2018.

[27] Ginsburg, Boris, et al. ["Stochastic gradient methods with layer-wise adaptive moments for training of deep networks."](https://arxiv.org/abs/1905.11286) arXiv preprint, 2019.

[28] Hu, Edward J., et al. ["LoRA: Low-rank adaptation of large language models."](https://arxiv.org/abs/2106.09685) ICLR, 2022: 3.

[29] Dettmers, Tim, et al. ["Qlora: Efficient finetuning of quantized llms."](https://arxiv.org/abs/2305.14314) Advances in Neural Information Processing Systems 36, 2023: 10088–10115.

[30] Weng, Lilian. ["How to Train Really Large Models on Many GPUs?"](https://lilianweng.github.io/posts/2021-09-25-train-large/) Lil'blog, 2021.

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui.(Mar 2025). 训练大模型并行和内存优化技术.
https://syhya.github.io/zh/posts/2025-03-01-train-llm

Or

```bibtex
@article{syhya2025train-llm,
  title   = "训练大模型并行和内存优化技术",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Mar",
  url     = "https://syhya.github.io/zh/posts/2025-03-01-train-llm"
}