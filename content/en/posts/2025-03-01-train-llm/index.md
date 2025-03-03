---
title: "Parallel and Memory Optimization Techniques for Training Large Models"
date: 2025-03-01T12:00:00+08:00
lastmod: 2025-03-01T12:00:00+08:00
author: Yue Shui
categories: ["Tech Blog"]
tags: [LLMs, Pre-training, Distributed Training, Memory Optimization, Data Parallelism, Model Parallelism, Pipeline Parallelism, Tensor Parallelism, Sequence Parallelism, Hybrid Parallelism, Heterogeneous Systems, MoE, ZeRO, LoRA, AI, Deep Learning, AI Infrastructure]
readingTime: 60
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

## Background

Recently, the number of parameters in large models has been continuously increasing, from the initial billions to today's hundreds of billions or even trillions. While large models have brought unprecedented application effects, they have also triggered a series of severe challenges in computing resources, memory management, and training stability. Therefore, this blog summarizes some commonly used distributed parallel training and memory management techniques, hoping to help everyone better train and optimize large models.

### Training Challenges of Large Models

* **Explosive Growth in Parameter Scale**
  With the continuous pursuit of model capacity and performance, the number of parameters in neural networks is growing exponentially. Today, models ranging from millions to billions, hundreds of billions, and even trillions of parameters are emerging. For example, Llama 3.1 405B has approximately 405 billion parameters, while it is rumored that GPT-4 may have as many as 1.7 trillion parameters. This massive parameter scale has led to a sharp increase in computing and memory demands, bringing unprecedented pressure to the training process.

* **Soaring Computational Complexity**
  The rapid increase in the number of parameters directly leads to a significant increase in overall computational complexity. Training a large model once may take weeks or even months. Even with large-scale high-performance GPU clusters, the training cycle is still unsatisfactory, severely restricting model iteration speed and research efficiency.

* **Increasingly Prominent Memory Bottleneck**
  In addition to storing massive model parameters, large models must also save intermediate activations, gradient information, and optimizer states during training. This data poses a huge challenge to GPU memory. Even with high-end GPUs equipped with A100, H100 (80GB memory), H200 (141GB memory), or GB200 (384GB memory), single-card memory is often insufficient to meet the needs of models with hundreds of billions or even trillions of parameters, leading to frequent "Out of Memory (OOM)" errors.

* **Communication Overhead Becomes a Bottleneck**
  In multi-GPU distributed training environments, inter-node communication is frequently required for data synchronization (such as gradient aggregation). As the model size and the number of GPUs increase, this communication volume rises sharply. Even in high-bandwidth networks, All-Reduce operations to transmit massive amounts of data consume a significant amount of time, becoming one of the main bottlenecks of overall parallel efficiency.

* **Training Stability Challenges**
  Ultra-large-scale models are more prone to gradient vanishing or gradient explosion problems during training, leading to unstable training processes and difficulty in convergence. Although mixed-precision training can accelerate training and reduce memory footprint to some extent, it may also introduce new numerical stability issues, requiring researchers to invest more effort in detailed tuning.

### Necessity of Distributed Training

Faced with the above challenges, distributed training technology has become a key solution to support the training of large models. By splitting training tasks and distributing them to multiple GPUs or computing nodes, distributed training can fully utilize parallel computing and cluster memory resources, thereby breaking through the limitations of a single GPU. The main advantages are reflected in the following aspects:

* **Breaking Through the Computing Power Limit of a Single GPU**
  The computing power of a single GPU is ultimately limited and cannot cope with the massive computing demands of trillion-parameter models. With data parallelism and model parallelism techniques, training tasks can be evenly distributed to multiple GPUs, thereby significantly shortening the overall training time.

* **Overcoming the Memory Bottleneck of a Single GPU**
  By distributing model parameters, intermediate activations, and optimizer states across the memory of multiple GPUs, distributed training effectively expands the available memory capacity. Typical technologies such as ZeRO, through sharding data processing, make the training of ultra-large-scale models possible.

* **Accelerating Model Iteration and R&D Cycle**
  The high parallelism of distributed training makes it possible to complete training tasks that originally required weeks or even months in just a few days, thereby greatly improving the model iteration speed and enabling new architectures and strategies to be verified and applied more quickly.

* **Supporting Exploration of Larger-Scale Models**
  Distributed training provides a solid foundation for exploring larger-scale and more complex neural network architectures. It is with this technical support that trillion-parameter models (such as Switch Transformer) can be successfully trained and put into practical applications.

* **Improving the Robustness and Scalability of Training Systems**
  Distributed systems have excellent fault tolerance. When a GPU node fails, other nodes can quickly take over the task, ensuring that the training process is not interrupted. At the same time, the cluster size can be flexibly expanded or reduced according to specific needs, meeting the training requirements of different scale models.

## Parallel Training

The following figure intuitively shows the differences between various parallel training strategies. Different colors represent different model layers (e.g., three layers), and dashed lines distinguish different GPUs. From left to right are data parallelism, model parallelism (including pipeline parallelism and tensor parallelism), and expert parallelism (MoE).

{{< figure
    src="parallelism_compare.png"
    caption="Fig. 1. An illustration of various parallelism strategies on a three-layer model. Each color refers to one layer and dashed lines separate different GPUs. (Image source: [OpenAI Blog, 2022](https://openai.com/index/techniques-for-training-large-neural-networks/))"
    align="center"
    width="90%"
>}}

- **Data Parallelism**
  The complete model is copied to each GPU, and the dataset is divided into different batches and distributed to each GPU for parallel computation. Finally, the gradients of all GPUs are aggregated during parameter updates.

- **Model Parallelism**
  The model is divided and distributed across different GPUs, with each GPU responsible for computing only a part of the model. It can be further divided into the following two categories:
  - **Pipeline Parallelism**: The model is split layer-wise (vertically), with different GPUs responsible for different layers. Micro-batches are passed through the pipeline to execute forward and backward computations in parallel.
  - **Tensor Parallelism**: Large-scale tensor operations (such as large matrix multiplications) within a layer are split horizontally. Each GPU performs part of the computation in parallel and aggregates the results when necessary.

- **Expert Parallelism**
  Through a gating strategy, each input sample only passes through a subset of experts (sub-networks), thus distributing the entire model across different GPUs by "expert modules". Commonly used in Mixture-of-Experts (MOE) structures, it can achieve ultra-large parameter scales but only activate a portion of experts during inference/training.

Below, I will elaborate on various parallel methods.

## Data Parallelism

{{< figure
    src="data_parallelism.png"
    caption="Fig. 2. Data Parallelism. (Image source: [Clolossal-AI Documentation](https://colossalai.org/docs/concepts/paradigms_of_parallelism))"
    align="center"
    width="60%"
>}}

In deep learning training, **Data Parallelism (DP)** is the most commonly used parallel strategy. Its core idea is:
1. **Replicate Model Parameters**: Place a complete copy of the model parameters on each computing device (usually a GPU).
2. **Partition Training Data**: Divide the large-scale dataset into multiple subsets along the sample dimension. Different subsets are assigned to different GPUs for processing.
3. **Local Forward and Backward Propagation**: Each GPU independently computes the loss and corresponding local gradients.
4. **Gradient/Parameter Synchronization**: Aggregate the gradients from each GPU and update the model parameters, ensuring that the model replicas on all GPUs remain consistent after each iteration.

The following shows the **data parallelism** workflow:

1. **Dataset Partitioning**
   Divide the training dataset $D$ into $N$ non-overlapping subsets $\{D_1, D_2, \dots, D_N\}$, where $N$ is the number of GPUs. Usually, it is ensured that the size of each subset is similar to achieve load balancing.

2. **Model Replication**
   Replicate a complete copy of the model parameters $\theta$ on each GPU. At the beginning of training, these parameters are the same on each GPU.

3. **Data Distribution**
   Distribute subset $D_i$ to the $i$-th GPU, allowing it to be stored locally and used for subsequent calculations.

4. **Local Forward Propagation**
   Each GPU performs forward propagation based on its local data subset $D_i$ to obtain the local loss $L_i(\theta, D_i)$.

5. **Local Backward Propagation**
   Each GPU performs backward propagation based on the local loss $L_i$ to calculate the local gradient

   $$
     g_i = \nabla_{\theta} L_i(\theta, D_i).
   $$

6. **Gradient Synchronization**
   Gradient synchronization (usually All-Reduce) is performed between GPUs to aggregate all local gradients $\{g_1, g_2, \ldots, g_N\}$ to obtain the global average gradient

   $$
     \bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i.
   $$

7. **Parameter Update**
   Each GPU uses the global average gradient $\bar{g}$ to update its local model parameters:

   $$
     \theta \leftarrow \theta - \eta \cdot \bar{g},
   $$

   where $\eta$ is the learning rate.

8. **Iterative Loop**
   Repeat steps 4-7 until the model converges or reaches the preset number of training epochs.

### Bulk Synchronous Parallel vs. Asynchronous Parallel

In step 6 "Gradient Synchronization" above, how and when to perform "synchronization" is one of the important factors affecting the performance and convergence behavior of data parallelism. It is generally divided into the following two categories:

**Bulk Synchronous Parallel (BSP)** is the most common and easiest to understand synchronization mode in data parallelism. Its characteristics can be summarized as "globally synchronizing gradients and updating parameters once after each mini-batch iteration". The specific process is:

1. **Local Computation**: Each GPU performs forward and backward propagation based on its data subset $D_i$ to obtain the local gradient $g_i$.
2. **Global Communication**: All GPUs synchronize (e.g., through All-Reduce) to calculate $\bar{g}$.
3. **Parameter Update**: Each node uses $\bar{g}$ to update its local parameter replica $\theta$.
4. **Wait and Next Iteration**: All nodes complete the above operations before entering the next iteration.

**Asynchronous Parallel (ASP)** aims to get rid of the global synchronization point of BSP and allow each node to perform calculations and parameter updates independently. Its typical implementation is the **asynchronous push-pull** process under the "Parameter Server (PS)" architecture:

1. Each node calculates the gradient $g_i$ locally, and then **pushes** it to the parameter server;
2. Once the parameter server receives the gradient, it immediately updates the global model parameters;
3. Other nodes will **pull** down the latest parameters when they need them to continue the next step of calculation.

### BSP vs. ASP

The following table summarizes the main differences between synchronous and asynchronous parallelism in a data parallel environment:

| **Comparison Dimension** | **Synchronous Parallel (BSP)**                                      | **Asynchronous Parallel (ASP)**                                          |
|:-----------------------|:-----------------------------------------------------------------------|:----------------------------------------------------------------------------|
| **Parameter Update Timing** | Global synchronization once per mini-batch or after a certain number of iterations | Each node updates parameters independently, without needing to keep the same timestep as others |
| **Convergence Stability** | **High**. The gradients used are the latest, the convergence path is controllable and easy to analyze | **Lower**. Stale gradients exist, convergence rate and stability may be affected |
| **Communication Requirements** | Highly dependent on All-Reduce, all nodes need to wait and exchange data during synchronization | Each node asynchronously pushes/pulls to the parameter server, communication is more flexible, but the parameter server may become a bottleneck |
| **Hardware Resource Utilization** | If there are slow nodes or network delays, other nodes need to wait, and resource utilization may be reduced | No need to wait for slow nodes, computing resources can be used efficiently |
| **Implementation Complexity** | Relatively low, mainstream frameworks (PyTorch DDP, Horovod, etc.) have built-in support | Relatively higher, parameter server and other components are required, more synchronization logic and data consistency need to be handled |
| **Applicable Scenarios** | Homogeneous hardware, good network bandwidth, pursuit of higher convergence quality | Heterogeneous hardware, unstable or low bandwidth network, need for extremely high throughput and tolerance for certain convergence risks |
| **Typical Implementations** | PyTorch DDP, TensorFlow MirroredStrategy                                | Parameter Server architecture (MXNet, TensorFlow ParameterServer mode, etc.) |

> **Recommendation**: In actual projects, start with simple synchronous parallelism (BSP), and use PyTorch DDP or similar tools for multi-GPU training. If the network environment is heterogeneous, there are many nodes, or the task requires extremely high throughput, you can try asynchronous parallelism (ASP) or parameter server solutions, and cooperate with Gradient Accumulation to balance bandwidth and update frequency.

### Gradient Accumulation

When the batch size is large or communication becomes the main bottleneck, **Gradient Accumulation** can be used to reduce the synchronization frequency. Its core idea is:
- Continuously calculate the local gradients of multiple mini-batches and accumulate them in the local accumulation buffer;
- When the number of accumulated mini-batches reaches $K$, trigger a global gradient synchronization and parameter update.

Let $g_j$ be the gradient of the $j$-th mini-batch, then in an "accumulation cycle", we get

$$
  G = \sum_{j=1}^{K} g_j.
$$

Then update with learning rate $\eta$:

$$
  \theta \leftarrow \theta - \eta \cdot G.
$$

Since gradient synchronization is no longer performed for each mini-batch, but once every $K$ accumulated mini-batches, the communication overhead can be significantly reduced. However, the reduced parameter update frequency may also slow down the training convergence speed, and a trade-off between throughput and convergence performance is needed.

### Distributed Data Parallel

**Distributed Data Parallel (DDP)** is a highly optimized implementation of BSP in PyTorch v1.5 ([Li et al. 2020](https://arxiv.org/pdf/2006.15704)), which facilitates data parallelism for single-machine multi-GPU and even multi-machine multi-GPU. Its main optimizations include:

1. **Gradient Bucketing**: Divide model parameters into multiple "buckets"; when backpropagation is performed, once all gradients in a bucket are calculated, an **All-Reduce for that bucket** is immediately initiated, instead of waiting for all gradients to be calculated before synchronizing at once.
2. **Communication and Computation Overlap**: DDP uses asynchronous communication and non-blocking operations to overlap gradient synchronization (communication) with forward propagation and backward propagation (computation) as much as possible, thereby reducing communication overhead. This overlap strategy improves overall parallel efficiency.
3. **Gradient Accumulation**: DDP can also be easily combined with **gradient accumulation**. Combined use, by increasing the gradient update interval for each synchronization, reduces the synchronization frequency. In large-scale distributed training, this helps to further reduce communication overhead and improve training efficiency.

{{< figure
    src="pytorch_ddp.png"
    caption="Fig. 3. Pseudo code for Pytorch DDP. (Image source: [Li et al. 2020](https://arxiv.org/pdf/2006.15704))"
    align="center"
    width="80%"
>}}

### Ring All-Reduce

In a multi-GPU (especially single-machine multi-GPU) environment, if there is high-speed interconnect (such as NVLink, PCIe switch, etc.), **Ring All-Reduce** can be used to significantly reduce communication overhead. The idea is:

1. Organize $k$ nodes into a ring and divide the gradient vector into $k$ parts equally.
2. In the "summation phase", each node sends a part of its local gradient to the next node and adds it to the received gradient; after several rounds of this process, each node will hold the complete "aggregated" gradient.
3. In the "broadcast phase", the final result is distributed to all nodes along the ring.

Ideally, the communication cost of Ring All-Reduce is approximately independent of the number of nodes (can be regarded as $\mathcal{O}(1)$), which is very suitable for gradient synchronization in a multi-GPU environment. It is a core communication mode widely used in libraries such as Horovod and NCCL.

### Parameter Server

When the cluster scale expands to multi-machine multi-GPU, simple single-point aggregation (such as a central server) is often difficult to support parallel training of massive data. **Parameter Server (PS)** ([Li, et al., 2014](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)) is a typical architecture designed for scalable distributed training:

1. **Parameter Sharding**: Split model parameters in the form of key-value pairs. Different PS nodes only manage parameters of specific shards.
2. **push-pull Semantics**: After the computing node obtains the gradient locally, it **pushes** it to the corresponding PS; after the PS updates the parameters of the shard, the computing node can **pull** down the latest version when needed for the next step of calculation.
3. **Flexible Fault Tolerance and Expansion**: By adding or removing PS nodes, capacity can be flexibly expanded in terms of bandwidth or computing needs; backup and fault tolerance strategies can also be implemented on PS.

This **PS + Worker** mode can combine data parallelism and model parallelism **simultaneously**, splitting ultra-large models and storing them on multiple PSs, and performing distributed training on ultra-large data. PS itself can also be split and merged according to the load situation to form a more complex hierarchical topology.

## Model Parallelism

**Model Parallelism (MP)** is a parallel method that splits the model itself across multiple computing devices (GPUs) for training. When the model parameter size exceeds the memory capacity of a single GPU, model parallelism becomes a necessary choice. Model parallelism is mainly divided into two types: Pipeline Parallelism and Tensor Parallelism.

**Naive Model Parallelism and Bubble Problem**

{{< figure
    src="naive_mp.png"
    caption="Fig. 4. A naive model parallelism setup where the model is vertically split into 4 partitions. Data is processed by one worker at a time due to sequential dependency, leading to large “bubbles” of idle time. (Image source: [Huang et al. 2018](https://arxiv.org/abs/1811.06965))"
    align="center"
    width="100%"
>}}

Naive model parallelism implementation, which simply divides the model layer by layer and executes it sequentially on different GPUs, will encounter a serious "bubble" problem. Due to the dependencies between layers, when one GPU is processing a certain stage of a data sample, other GPUs may be idle, waiting for the output of the previous GPU or the input of the next GPU. This GPU idle time is called "bubble", which seriously reduces the efficiency of pipeline parallelism.

Where $F_i$ represents the forward propagation of Stage $i$, and $B_i$ represents the backward propagation of Stage $i$. It can be seen that in naive pipeline parallelism, only one GPU is working most of the time, and other GPUs are idle, resulting in low efficiency.

**Reasons for the bubble problem:**

* **Inter-layer dependency:** There is a sequential dependency between the layers of the neural network. The calculation of the next layer must depend on the output of the previous layer.
* **Sequential execution:** Naive model parallelism executes layer by layer in order, which prevents GPUs from working in full parallelism.

## Pipeline Parallelism

{{< figure
    src="pipeline_parallelism.png"
    caption="Fig. 5. Pipeline Parallelism. (Image source: [Clolossal-AI Documentation](https://colossalai.org/docs/concepts/paradigms_of_parallelism))"
    align="center"
    width="60%"
>}}

**Pipeline Parallelism (PP)** divides the model layer by layer into multiple stages, and each stage is assigned to a GPU. Data is passed between different GPUs like a pipeline. The output of the previous GPU serves as the input of the next GPU. Pipeline parallelism aims to improve the efficiency of model parallel training and reduce GPU idle time.

### GPipe

GPipe ([Huang et al. 2018](https://arxiv.org/abs/1811.06965)) is an efficient pipeline parallel training system proposed by Google, which aims to solve the bubble problem of naive pipeline parallelism. The core idea of GPipe is to divide a **mini-batch** into multiple **micro-batches** and use **synchronous gradient aggregation** to alleviate the bubble problem and improve pipeline efficiency.

{{< figure
    src="gpipe.png"
    caption="Fig. 6. Illustration of pipeline parallelism in GPipe with 4 microbatches and 4 partitions. GPipe aggregates and updates gradients across devices synchronously at the end of every batch. (Image source: [Huang et al. 2018](https://arxiv.org/abs/1811.06965))"
    align="center"
    width="100%"
>}}

The following is the GPipe scheduling strategy:
1. **Micro-batch Partitioning:** Divide a mini-batch into $m$ micro-batches. The size of each micro-batch after partitioning is $1/m$ of the original mini-batch.
2. **Pipeline Stage Partitioning:** Divide the model layer by layer into $d$ stages, and assign each stage to a GPU.
3. **Pipeline Execution:** Process each micro-batch in sequence, performing forward and backward propagation in the pipeline. The specific process is as follows:
    * **Forward Propagation:** For each micro-batch, perform forward propagation sequentially on Stage 1, Stage 2, ..., Stage $d$. The output of Stage $i$ serves as the input of Stage $i+1$.
    * **Backward Propagation:** When the forward propagation of all micro-batches is completed, backward propagation begins. For each micro-batch, perform backward propagation sequentially on Stage $d$, Stage $d-1$, ..., Stage $1$. The gradient of Stage $i$ serves as the input of Stage $i-1$.
4. **Synchronous Gradient Aggregation:** After the backward propagation of all micro-batches is completed, aggregate the gradients of all micro-batches (e.g., averaging) to obtain the global average gradient.
5. **Parameter Update:** Each GPU uses the global average gradient to update its local model parameters.

### GPipe Bubble Ratio Formula

Assuming that the forward and backward propagation time of each micro-batch is 1 unit, the pipeline depth is $d$, and the number of micro-batches is $m$, the bubble ratio of GPipe is:

$$
\text{Bubble Ratio} = 1 - \frac{2md}{(2m + 2(d-1))d} = \frac{d-1}{m+d-1}
$$

When the number of micro-batches $m$ is much larger than the pipeline depth $d$ ($m \gg d$), the bubble ratio approaches 0, and the pipeline efficiency is close to linear acceleration. The GPipe paper points out that when $m > 4d$, the bubble overhead can be almost ignored (in the case of activation recomputation). Therefore, there are the following benefits:

* **Reduce Bubbles:** GPipe significantly reduces the bubble problem of naive pipeline parallelism through micro-batch partitioning and pipeline scheduling, improving GPU utilization and training efficiency.
* **Synchronous Gradient Aggregation:** GPipe adopts synchronous gradient aggregation, which ensures the synchronicity of the training process and good model convergence.
* **Linear Acceleration Potential:** When the number of micro-batches is large enough, GPipe can achieve near-linear acceleration.

### PipeDream

{{< figure
    src="pipe_dream.png"
    caption="Fig. 7. Illustration of 1F1B microbatch scheduling in PipeDream. (Image source: [Harlap et al. 2018](https://arxiv.org/abs/1806.03377))"
    align="center"
    width="100%"
>}}

PipeDream ([Harlap et al. 2018](https://arxiv.org/abs/1806.03377)) is another efficient pipeline parallel training system. It adopts the 1F1B (1-Forward-1-Backward) scheduling strategy and introduces Weight Stashing technology to further reduce bubbles, improve pipeline efficiency, and solve the weight version inconsistency problem that may be caused by 1F1B scheduling.

The core idea of PipeDream's 1F1B scheduling strategy is that each GPU (Stage) alternately performs forward propagation and backward propagation, working in parallel as much as possible to reduce GPU idle time. The specific process is as follows:

1. **Micro-batch Partitioning:** Divide a mini-batch into $m$ micro-batches.
2. **Pipeline Stage Partitioning:** Divide the model layer by layer into $d$ stages, and assign each stage to a GPU.
3. **1F1B Scheduling Execution:** Each GPU takes turns to perform forward propagation and backward propagation.

### Weight Stashing

Since forward propagation and backward propagation may use different versions of model weights in 1F1B scheduling, it will cause weight version inconsistency problems, affecting the correctness and convergence of training. PipeDream introduces Weight Stashing technology to solve this problem. The core idea of weight stashing is that each GPU maintains multiple versions of model weights and ensures that forward propagation and backward propagation use the same version of weights.

**Weight Stashing Implementation:**

* **Version Management:** Each GPU maintains a weight version queue to store multiple versions of model weights.
* **Version Selection:** When performing forward propagation, select the latest weight version. When performing backward propagation, select the same weight version as the corresponding forward propagation.
* **Version Update:** After completing backward propagation of all micro-batches in a mini-batch, update the model weights and generate a new weight version.

> To further optimize the memory usage of PipeDream, especially in terms of weight stashing, PipeDream has derived two memory optimization variants: PipeDream-flush and PipeDream-2BW.

### PipeDream-flush

{{< figure
    src="pipe_dream_flush.png"
    caption="Fig. 8. Illustration of pipeline scheduling in PipeDream-flush. (Image source: [Narayanan et al. 2020](https://arxiv.org/abs/2006.09503))"
    align="center"
    width="100%"
>}}

PipeDream-flush periodically performs global synchronous pipeline flushing on the basis of PipeDream, similar to GPipe's synchronous gradient aggregation. By periodically flushing, PipeDream-flush can greatly reduce the memory space required for weight stashing, only needing to maintain a single version of model weights, but it will sacrifice a small amount of throughput.

### PipeDream-2BW

PipeDream-2BW (Double-Buffered Weights) maintains two versions of model weights, namely "double-buffered weights". It updates the model version every $k$ micro-batches, where $k$ is greater than the pipeline depth $d$ ($k > d$). The newly updated model version does not immediately completely replace the old version, because there may still be some remaining backward propagation operations that depend on the old version. With double-buffered weights, PipeDream-2BW can reduce the memory overhead of weight stashing to only maintaining two versions of model weights, significantly reducing memory footprint.

{{< figure
    src="pipe_dream_2bw.png"
    caption="Fig. 9. Illustration of pipeline scheduling in PipeDream-2BW. (Image source: [Narayanan et al. 2020](https://arxiv.org/abs/2006.09503))"
    align="center"
    width="100%"
>}}

The PipeDream-2BW strategy has the following advantages:

* **Lower Bubble Overhead:** The 1F1B scheduling strategy can further reduce bubbles compared to GPipe, improving GPU utilization and training efficiency.
* **Weight Stashing Solves Version Consistency:** Weight stashing technology ensures that forward propagation and backward propagation use the same version of weights, solving the weight version inconsistency problem that may be caused by 1F1B scheduling.
* **Memory Optimization Variants:** PipeDream-flush and PipeDream-2BW further optimize memory usage, reduce the memory overhead of weight stashing, and make pipeline parallelism more suitable for memory-constrained scenarios.

## Tensor Parallelism

**Tensor Parallelism (TP)** is a parallel method that splits tensors in the model (usually weight matrices) along specific dimensions and distributes the split shards to different GPUs for computation. Tensor parallelism has the following advantages:

* **Breaking Through Single GPU Memory Limits:** Tensor parallelism can distribute model parameters across multiple GPUs, breaking through the memory capacity limit of a single GPU and supporting the training of larger-scale models.
* **Intra-layer Parallelism:** Tensor parallelism can achieve parallelization within model layers, such as parallel computation of matrix multiplication operations, improving computational efficiency.
* **Combination with Data Parallelism and Pipeline Parallelism:** Tensor parallelism can be combined with other parallel technologies such as data parallelism and pipeline parallelism to form multi-dimensional parallel strategies, further improving training efficiency and scalability.

### Megatron-LM

Megatron-LM ([Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)) is a system proposed by NVIDIA for training ultra-large language models. It adopts tensor parallelism technology to parallelize matrix multiplication operations within Transformer model layers, including matrix multiplications in **self-attention** and **MLP**.

{{< figure
    src="Megatron-LM.png"
    caption="Fig. 10. Illustration of tensor parallelism for key transformer components proposed in Megatron-LM. (Image source: [Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053))"
    align="center"
    width="100%"
>}}

The MLP layer of Transformer usually contains two linear layers. The calculation of the first linear layer can be expressed as $Y = \text{GeLU}(XA)$, where $X$ is the input matrix, $A$ is the weight matrix, and GeLU is the activation function. Megatron-LM splits the weight matrix $A$ along the column dimension into $P$ shards $[A_1, A_2, ..., A_P]$, where $P$ is the number of GPUs. Each GPU $i$ is responsible for storing and computing the weight shard $A_i$.

**Tensor Parallel Computation Process of MLP Layer:**

$$
\begin{aligned}
\text { Split } A & =\left[A_1, A_2\right] \\
Y & =\operatorname{GeLU}(X A) \\
{\left[Y_1, Y_2\right] } & =\left[\operatorname{GeLU}\left(X A_1\right), \operatorname{GeLU}\left(X A_2\right)\right]
\end{aligned}
$$

1. **Weight Sharding:** Split the weight matrix $A$ along the column dimension into $P$ shards $[A_1, A_2, ..., A_P]$ and assign shard $A_i$ to GPU $i$.
2. **Local Matrix Multiplication:** Each GPU $i$ uses the input matrix $X$ and weight shard $A_i$ to perform matrix multiplication to obtain the local output $Y_i = \text{GeLU}(XA_i)$.
3. **Global Concatenation (All-Gather):** All GPUs use All-Gather operation to concatenate the local outputs $\{Y_1, Y_2, ..., Y_P\}$ into a complete output matrix $Y = [Y_1, Y_2, ..., Y_P]$.

**Tensor Parallelism of Self-Attention Layer**

Megatron-LM also performs tensor parallel sharding on the Query ($Q$), Key ($K$), Value ($V$) weight matrices in the Transformer's self-attention layer, and performs corresponding local matrix multiplication and global concatenation operations to achieve tensor parallelism of the self-attention layer. The calculation formula of the self-attention layer is:

$$
\text{Attention}(X, Q, K, V) = \text{softmax}\left(\frac{(XQ)(XK)^T}{\sqrt{d_k}}\right)XV
$$

### PTD-P

PTD-P (Pipeline, Tensor, and Data Parallelism) ([Narayanan et al. 2021](https://arxiv.org/abs/2104.04473)) is a multi-dimensional parallel strategy that combines pipeline parallelism, tensor parallelism, and data parallelism. PTD-P aims to fully utilize the advantages of various parallel technologies to improve the efficiency and scalability of training ultra-large models.

**Features of PTD-P:**

* **Multi-dimensional Parallel Combination:** PTD-P uses pipeline parallelism, tensor parallelism, and data parallelism simultaneously, which can parallelize the training process from multiple dimensions.
* **Interleaved 1F1B Scheduling:** PTD-P adopts the interleaved 1F1B scheduling strategy. Unlike traditional pipeline parallelism, it divides the model into multiple discontinuous layer blocks (model chunks) and assigns multiple layer blocks to each GPU. This scheduling strategy can further reduce bubbles and improve pipeline efficiency.
* **Flexible Parallel Configuration:** PTD-P allows users to flexibly configure the combination of various parallel technologies according to the model structure and hardware resources. For example, tensor parallelism and data parallelism can be used alone, or pipeline parallelism, tensor parallelism, and data parallelism can be used simultaneously.

Traditional pipeline parallelism usually divides the model into continuous layer blocks, and each GPU is responsible for a continuous layer block. PTD-P's interleaved 1F1B scheduling divides the model into multiple discontinuous layer blocks. For example, GPU 1 is responsible for layers 1, 2, 9, 10, GPU 2 is responsible for layers 3, 4, 11, 12, and so on. Each GPU is responsible for multiple discontinuous layer blocks, which can more effectively utilize GPU resources and reduce bubble overhead.

{{< figure
    src="PTD-P.png"
    caption="Fig. 11.(Top) Default 1F1B pipeline schedule as in PipeDream-flush.(Bottom) Interleaved 1F1B pipeline schedule. First model chunks are in dark colors and second chunks are in light colors. (Image source: [Narayanan et al. 2021](https://arxiv.org/abs/2104.04473))"
    align="center"
    width="100%"
>}}

## Mixture-of-Experts Model

**Mixture-of-Experts (MoE)** ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) is a sparsely activated model that significantly increases the model's parameter size and performance without significantly increasing the computational cost by combining multiple independent "expert" networks and a gating network. The core idea of MoE is **Sparse Activation**, that is, for each input sample, only a part of the expert networks are activated, rather than the entire model. This method not only improves computational efficiency but also enhances the model's expressive ability, making it perform well in LLMs.

MoE's design inspiration comes from **Ensemble learning**, a technology that decomposes complex tasks into multiple subtasks and completes them collaboratively by different models. In MoE, these "subtasks" are processed by multiple independent expert networks, and the gating network is responsible for dynamically selecting the most suitable experts based on the characteristics of the input sample. This division of labor and cooperation mechanism is similar to an expert team in human society: experts in different fields provide professional opinions for specific problems, and finally, a comprehensive result is obtained.

{{< figure
    src="moe.png"
    caption="Fig. 12. Illustration of a mixture-of-experts(MoE) layer. Only 2 out of experts are selected and activated by the gating network. (Image source: [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538))"
    align="center"
    width="100%"
>}}

### Core Components of MoE

A typical MoE contains the following components:

* **Expert Networks:** A set of independent neural networks $\{E_1, E_2, ..., E_n\}$. Each expert network $E_i$ can be any type of neural network, such as FFN, CNN, RNN, etc. The number of expert networks $n$ can be very large, such as dozens, hundreds, or even thousands.
* **Gating Network:** A trainable neural network $G$ used to learn a probability distribution based on the input sample $x$ to determine which experts to activate. The input of the gating network is the input sample $x$, and the output is an $n$-dimensional probability vector $p = G(x) = [p_1, p_2, ..., p_n]$, where $p_i$ represents the probability of activating expert $E_i$.
* **Expert Output Aggregation:** According to the output probability distribution of the gating network, the outputs of the activated expert networks are weighted and summed to obtain the final output $y$ of the MoE layer.

### Noisy Top-k Gating

To achieve sparse activation and ensure balanced expert usage, MoE usually adopts **Noisy Top-k Gating** as the gating mechanism. This method guarantees computational efficiency and avoids uneven expert load through the introduction of noise and top-k selection. The detailed workflow is as follows:

1. **Gating Score Calculation:**

For an input sample $x$, the gating network first calculates the gating score $H^{(i)}(x)$ for each expert. This score consists of two parts: linear transformation and noise term, as shown in the formula:

$$
H^{(i)}(x) =(x W_g)^{(i)} + \epsilon \cdot \text{softplus}\left((x W_{\text{noise}})^{(i)} \right), \quad \epsilon \sim \mathcal{N}(0, 1)
$$

- **Parameter Description**:
  - $W_g \in \mathbb{R}^{d \times n}$: Trainable weight matrix of the gating network, where $d$ is the input feature dimension and $n$ is the number of experts.
  - $W_{\text{noise}} \in \mathbb{R}^{d \times n}$: Weight matrix used to generate noise.
  - $\epsilon \sim \mathcal{N}(0, 1)$: Standard Gaussian noise, increasing gating randomness.
  - $\text{softplus}(x) = \log(1 + e^x)$: Smooth activation function to ensure that the noise is non-negative.

The introduction of noise avoids the gating network always selecting fixed experts and enhances the robustness and diversity of the model.

2. **Top-k Selection:**

After calculating the gating score vector $H(x) = [H^{(1)}(x), H^{(2)}(x), \dots, H^{(n)}(x)]$, the gating network selects the top $k$ experts with the largest values (usually $k \ll n$). This step is implemented by the $\text{topk}(v, k)$ function:

$$
\text{topk}^{(i)}(v, k) =
\begin{cases}
v^{(i)} & \text{if } v^{(i)} \text{ is in the top } k \text{ elements of } v \\
-\infty & \text{otherwise}
\end{cases}
$$

Setting the scores of non-Top-k experts to $-\infty$ ensures that the probabilities of these experts in the subsequent softmax operation are 0, achieving sparsity.

3. **Softmax Normalization:**

Perform softmax normalization on the gating scores of the Top-k experts to obtain a sparse probability distribution $G(x)$:

$$
G(x) = \text{softmax}\left( \text{topk}(H(x), k) \right)
$$

Only the probabilities of the Top-k experts are non-zero, and the rest are 0. For example, if $n=100, k=2$, then the probabilities of 98 experts are 0.

4. **Weighted Summation:**

Weight and sum the outputs of the Top-k experts according to the probabilities to obtain the output of the MoE layer:

$$
y = \sum_{i=1}^{n} G^{(i)}(x) E_i(x)
$$

Since only $k$ experts are activated, the amount of calculation is much lower than activating all $n$ experts.

### Auxiliary Loss

To prevent the gating network from being overly biased towards a few experts, MoE introduces Auxiliary Loss ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) to encourage all experts to be used evenly. A common method is based on the square of the [Coefficient of Variation (CV)](https://en.wikipedia.org/wiki/Coefficient_of_variation) of expert usage rate:

$$
\mathcal{L}_{\text{aux}} = w_{\text{aux}} \cdot \text{CV}\left( \sum_{x \in X} G(x) \right)^2
$$

- **Parameter Description**:
  - $X$: Input samples of a mini-batch.
  - $\sum_{x \in X} G(x)$: Statistics on the number of times each expert is activated in a mini-batch.
  - $\text{CV}$: The ratio of standard deviation to mean, measuring the uniformity of expert usage distribution.
  - $w_{\text{aux}}$: Weight of auxiliary loss, which needs to be adjusted manually.

- **Function**: By minimizing $\mathcal{L}_{\text{aux}}$, the model optimizes the balance of expert selection and avoids some experts being overused while others are idle.

### GShard

GShard ([Lepikhin et al., 2020](https://arxiv.org/abs/2006.16668)) mainly shards the MoE layer, distributing the expert networks $\{E_1, E_2, ..., E_n\}$ in the MoE layer to multiple TPU devices. For example, if there are $P$ TPU devices, the expert networks can be divided into $P$ groups, and each group of expert networks is assigned to a TPU device. Other layers of the Transformer model (such as self-attention layer, LayerNorm layer) are replicated on all TPU devices.

**Improved Gating Mechanism of GShard:**

GShard has made some improvements on the basis of Noisy Top-k Gating to improve the performance and stability of the gating mechanism:

- **Expert Capacity:**
  To avoid expert overload, GShard introduces expert capacity limits. Each expert network has a capacity limit, indicating the maximum number of tokens it can process. If a token is routed to an expert network that has reached its capacity limit, the token will be marked as "overflowed", and the gating output will be set to a zero vector, indicating that the token will not be routed to any expert network.

- **Local Group Dispatching:**
  To improve gating efficiency, GShard groups tokens and enforces expert capacity limits at the group level. For example, divide the tokens in a mini-batch into multiple local groups, each local group containing a certain number of tokens. The gating network selects the top-k expert networks for each local group and ensures that the number of tokens processed by each expert network in a local group does not exceed its capacity limit.

- **Auxiliary Loss:**
  GShard also uses an auxiliary loss function to balance expert load. Different from the auxiliary loss of the original MoE model, GShard's auxiliary loss aims to minimize the mean square error of the proportion of data routed to each expert network, which more directly measures the degree of expert load balance.

- **Random Routing:**
  To increase the randomness of routing, GShard introduces a random routing mechanism when selecting the top-k expert networks. In addition to selecting the best top-k expert networks, GShard also randomly selects suboptimal expert networks with a certain probability to increase the diversity of expert networks and improve the generalization ability of the model.

Below is the core algorithm flow of GShard:

{{< figure
    src="gshard.png"
    caption="Fig. 13. Pseudo code of the group-level top-2 gating mechanism with auxiliary loss in GShard. (Image source: [Lepikhin et al., 2020](https://arxiv.org/abs/2006.16668))"
    align="center"
    width="100%"
>}}

### Switch Transformer

Switch Transformer ([Fedus et al. 2021](https://arxiv.org/pdf/2101.03961)) is a MoE model proposed by Google with a parameter size of **trillions**. Its core innovation is to replace the dense feed-forward network (FFN) layer in the Transformer model with a sparse Switch FFN layer. Unlike GShard's Top-2 Gating, Switch Transformer only routes each input token to one expert network, which has higher sparsity and further reduces computational costs, making it possible to train trillion-parameter models. It encourages token routing to be more balanced among $N$ experts. The auxiliary loss of Switch Transformer is based on the cumulative product of the actual routing ratio and the predicted routing probability. The specific formula is as follows:

$$
\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

- **Parameter Description**:
  - $N$: Total number of experts.
  - $f_i$: The proportion of tokens routed to the $i$-th expert, defined as:

    $$
    f_i = \frac{1}{T} \sum_{x \in B} 1\{\text{argmax } p(x) = i\}
    $$

  - $P_i$: The routing probability of the $i$-th expert predicted by the gating network, defined as:

    $$
    P_i = \frac{1}{T} \sum_{x \in B} p_i(x)
    $$

  - $T$: Total number of tokens in batch $B$.
  - $\alpha$: Weight hyperparameter of auxiliary loss, usually set to $10^{-2}$.

By minimizing $\text{loss}$, the model makes the actual routing ratio $f_i$ consistent with the predicted probability $P_i$, thereby indirectly promoting load balancing between experts and avoiding some experts being idle.

{{< figure
    src="switch_transformer.png"
    caption="Fig. 14. Switch transformer. The sparse switch FFN layer is in the blue boxes. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}

**Switch Router Mechanism:**

1. **Routing Prediction:**
   For an input token $x$, Switch Router predicts the routing probability $p_i = G^{(i)}(x)$ of each expert network, where $i = 1, 2, ..., n$, and n is the number of expert networks.

2. **Expert Selection:**
   Select the expert network with the highest routing probability as the best expert network. Switch Transformer adopts the Top-1 routing strategy, that is, each token is only routed to the expert network with the highest routing probability.

3. **Token Routing:**
   Route the input token $x$ to the selected best expert network for processing.

**Training Stability Optimization of Switch Transformer:**

To improve the training stability of Switch Transformer, the paper proposes the following optimization strategies:

- **Selective Precision**
  Using FP32 precision inside the routing function can improve training stability and avoid additional overhead caused by FP32 tensor communication. Specifically, the calculation process of Switch Router uses FP32 throughout, and the final result is converted to FP16 to balance efficiency and precision.

- **Smaller Initialization**
  It is recommended to adjust the weight initialization scale parameter $s$ of Transformer from 1 to 0.1. A smaller initialization scale helps to alleviate the risk of gradient explosion in the early stage of training, thereby improving overall training stability. The specific implementation is to sample from a truncated normal distribution with a mean of 0 and a standard deviation of $\sqrt{s/n}$ (where $n$ is the number of input units).

- **Higher Expert Dropout**
  Using a higher dropout rate (e.g., 0.4) in the expert FFN layer, while maintaining a lower dropout rate (e.g., 0.1) in non-expert layers, this setting can effectively prevent overfitting and thus enhance the generalization ability of the model. The experimental results in the figure below show that the model performs best when the dropout rate of the expert layer is set to 0.4 on tasks such as GLUE, CNNDM, SQuAD, and SuperGLUE.

{{< figure
    src="switch_transformer_fine_tuning_result.png"
    caption="Fig. 15. Fine-tuning regularization results. A sweep of dropout rates while fine-tuning Switch Transformer models pre-trained on 34B tokens of the C4 data set(higher numbers are better). (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}

The Switch Transformers paper uses the following figure to intuitively show how different parallel technologies split model weights and data:

{{< figure
    src="switch_transformer_parallelism.png"
    caption="Fig. 16. An illustration of various parallelism strategies on how(Top) model weights and(Bottom) data are split over multiple GPU cores. In the top row, each color denotes a unique weight matrix. In the bottom row, different colors indicate different sets of tokens. (Image source: [Fedus et al. 2021](https://arxiv.org/abs/2101.03961))"
    align="center"
    width="100%"
>}}

### Expert Choice

Expert Choice (EC) ([Zhou et al. 2022](https://arxiv.org/abs/2202.09368)) is a routing strategy opposite to token choice routing (such as GShard's top-2 or Switch Transformer's top-1). In token choice routing, each token selects top-k experts from all experts for routing; while in expert choice routing, each expert selects top-k tokens from all tokens for processing. This method aims to solve the problems of load imbalance and token waste in token choice routing, and significantly improve training efficiency. The following is the specific calculation process:

1. **Calculate Token-to-Expert Affinity Score**

   For an input matrix $X \in \mathbb{R}^{n \times d}$, the process of calculating the token-to-expert affinity score matrix $S \in \mathbb{R}^{n \times e}$ is:

   $$
   S = \text{softmax}(X \cdot W_g), \quad \text{where } W_g \in \mathbb{R}^{d \times e}.
   $$
   Here, $W_g$ is the gating weight matrix, and $e$ is the number of experts.

2. **Expert Selects Tokens**

   Each expert selects top-k tokens from all tokens for processing. Top-k selection is performed on $S^T$:

   $$
   G, I = \text{top-}k(S^T, k),
   $$

   to get:
   - **Gating matrix $G \in \mathbb{R}^{e \times k}$:** Records the routing weights corresponding to the tokens selected by the experts, where $G[i, j]$ represents the weight of the $j$-th token selected by expert $i$;
   - **Token index matrix $I \in \mathbb{R}^{e \times k}$:** Represents the index of the token selected by each expert in the input.

3. **One-hot Encoding**

   Convert the token index matrix $I$ into a one-hot encoding matrix $P \in \mathbb{R}^{e \times k \times n}$ for subsequent calculations:

   $$
   P = \operatorname{one}-\operatorname{hot}(I)
   $$

4. **Construct Gated FFN Layer Input**

   For each expert $i$, the input of its gated FFN layer is:

   $$
  (P \cdot X) \in \mathbb{R}^{e \times k \times d}.
   $$

EC controls the sparsity of the model by regularizing and limiting the number of experts to which each token is routed. A common regularization target is as follows:

$$
\begin{aligned}
& \max_{A} \langle S^{\top}, A \rangle + \lambda H(A) \\
& \text{s.t. } \forall i: \sum_{j'} A[i, j'] = k, \quad \forall j: \sum_{i'} A[i', j] \leq b, \quad \forall i,j: 0 \leq A[i, j] \leq 1,
\end{aligned}
$$

In the optimization problem considered, a matrix $A$ is defined, and the element in the $i$-th row and $j$-th column indicates whether the $i$-th expert has selected the $j$-th token (value 0 or 1). Since this optimization problem is complex to solve, the paper uses the [Dykstra algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) (to obtain an approximate solution through multiple iterations) to solve it.

The parameter $b$ is usually determined by the total number of tokens $n$ in the batch and the capacity factor, where the capacity factor represents the average number of experts used by each token. Most experiments use a higher capacity factor. The experimental results show that even when the capacity is reduced, EC (Expert Choice) still performs better than traditional top-1 token choice routing, although capped expert choice slightly reduces fine-tuning performance.

The advantages of EC are mainly reflected in the following two aspects:
- **Perfect Load Balancing:** Each expert processes a fixed number of $k$ tokens, thus avoiding the problem of some experts being overloaded while others are idle, achieving ideal load balancing.
- **Higher Training Efficiency:** Experiments show that EC can improve the training convergence speed by about 2 times, which is more efficient than traditional token choice routing.

However, EC also has the following limitations:
- **Batch Size Requirements:** Since EC has high requirements for batch size, it is not suitable for scenarios with smaller batch sizes.
- **Autoregressive Generation Limitations:** In autoregressive text generation tasks, EC's top-k selection cannot be implemented because future tokens cannot be predicted, so it is not suitable for such tasks.

## Sequence Parallelism

**Sequence Parallelism (SP)** is a parallelization strategy proposed for long sequence models (such as Transformer). By partitioning the input in the sequence dimension, it greatly reduces activation memory footprint and improves training efficiency. It is often used in combination with data parallelism, tensor parallelism, or pipeline parallelism, and is especially suitable for processing ultra-long text or other sequence data.

### Colossal-AI Sequence Parallelism

{{< figure
    src="colossal_sp.png"
    caption="Fig. 17. The overall architecture of the proposed sequence parallelism and existing parallel approaches. For sequence parallelism, Device 1 and Device 2 share the same trainable parameters. (Image source: [Li, et al. 2021](https://arxiv.org/abs/2105.13120))"
    align="center"
    width="100%"
>}}

The computational complexity and memory overhead of self-attention are proportional to the square of the sequence length $s$, $O(s^2)$. Long sequence data will increase the intermediate activation memory usage, thus limiting the training capacity of the device. Colossal-AI sequence parallelism ([Li, et al. 2021](https://arxiv.org/abs/2105.13120)) proposes **splitting ultra-long sequences to multiple cards** from a system perspective. The specific solution steps are as follows.

1. **Sequence Chunking**
   Divide the input sequence into several chunks, each chunk is saved and computed by different GPUs; therefore, each card only needs to store the activation of its corresponding sequence chunk, avoiding single-card memory explosion.
2. **Ring Communication + Self-Attention**
   Propose Ring Self-Attention (RSA) mechanism: each GPU first calculates local attention, and then sequentially transmits (ring structure) Key/Value chunks to adjacent GPUs. After multiple iterations, it is guaranteed that each GPU can obtain global sequence information.
3. **Combination with Other Parallel Methods**
   Not restricted by hyperparameters such as the number of attention heads and layers, it can be combined with data parallelism, tensor parallelism, pipeline parallelism and other technologies to jointly break through the sequence length limit of large-scale models.

{{< figure
    src="ring_self_attention.png"
    caption="Fig. 18. Ring Self-Attention. (Image source: [Li, et al. 2021](https://arxiv.org/abs/2105.13120))"
    align="center"
    width="100%"
>}}

### Megatron-LM Sequence Parallelism

Megatron-LM ([Shoeybi et al. 2019](https://arxiv.org/pdf/1909.08053)) originally used tensor parallelism to share part of the activation values, but the activation values of operations such as LayerNorm and Dropout in Transformer still need to be completely saved on a single card, and the memory consumption is still huge. Therefore, NVIDIA proposed Megatron-LM sequence parallelism ([Korthikanti, et al. 2022](https://arxiv.org/abs/2205.05198)) to **split these activation values** in the sequence dimension, greatly reducing the footprint.

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

1. **Sequence Dimension Splitting**
   For activations that are difficult to split in the tensor dimension, such as LayerNorm and Dropout, divide them along the sequence dimension, so that each GPU only processes a part of the sequence's nonlinear operations.
2. **Tensor Parallelism is Still Retained**
   Linear operations such as Attention and MLP continue to use tensor parallelism; the activations of sequence parallelism need to perform corresponding All-Gather or Reduce-Scatter before and after to exchange data.
3. **Selective Activation Recomputation**
   For some operations with small computational load but large activation volume, choose to temporarily recompute during backpropagation to further save memory.

### DeepSpeed-Ulysses Sequence Parallelism

DeepSpeed-Ulysses ([Jacobs et al. 2023](https://arxiv.org/abs/2309.14509)) proposes an efficient sequence parallelism scheme for ultra-long sequence training. By partitioning the input in the sequence dimension and combining two-stage all-to-all communication, it effectively reduces communication volume and activation memory, thereby supporting the training of million-token long sequence Transformer models.

{{< figure
    src="deepspeed_sp.png"
    caption="Fig. 21. DeepSpeed sequence parallelism(DeepSpeed-Ulysses) design. (Image source: [Jacobs et al. 2023](https://arxiv.org/abs/2309.14509))"
    align="center"
    width="100%"
>}}

1. **Sequence Partitioning + All-to-All Communication**
   Divide the input sequence along the sequence dimension to $P$ GPUs, and each GPU only processes a local $N/P$ sequence; before attention calculation, exchange Query ($Q$), Key ($K$), and Value ($V$) through All-to-All operation, so that each GPU obtains complete sequence information, but only calculates the assigned attention heads.

2. **Two-Stage Communication Optimization**
   - **First All-to-All:** Perform all-to-all exchange on $Q$/$K$/$V$ before attention calculation to disperse activation calculation and reduce memory pressure per card;
   - **Second All-to-All:** After attention calculation, collect the output context and remap it to local sequence partitions, which not only restores the original sequence structure but also significantly reduces the amount of communication data.

3. **Efficient Communication and Generality**
   Using all-to-all communication, the communication volume is reduced to $O(N/P)$, which saves nearly $P$ times the bandwidth compared to the traditional All-Gather method (communication volume $O(N)$); at the same time, this scheme is suitable for dense and sparse attention and can be seamlessly integrated with ZeRO-3 memory optimization, thereby supporting efficient training of larger models and longer sequences.

{{< figure
    src="deepspeed_ulysses_compare.png"
    caption="Fig. 22. DeepSpeed-Ulysses vs Megatron LM. (Image source: [DeepSpeed Blogs](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md))"
    align="center"
    width="100%"
>}}

- In a 64-card A100 environment, the throughput is increased by up to 2.5 times compared to Megatron-LM sequence parallelism, and longer sequences (million-level tokens) can be processed;
- The convergence performance is the same as the original model, and it can be easily integrated into the Megatron-DeepSpeed framework.

## Optimizer-Related Parallelism: ZeRO

**ZeRO (Zero Redundancy Optimizer)** ([Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054)) is an optimizer parallelism technology designed to eliminate memory redundancy when training large models. The main memory consumption for training large models is in two parts:

- **Model States:** Including **optimizer states** (such as momentum and second-order moments of Adam), **gradients**, and **model parameters**. Mixed-precision training not only requires storing FP16 data but also needs to retain FP32 versions of parameters and states, resulting in higher memory footprint.
- **Activations, Temporary Buffers, and Memory Fragmentation (Residual States):** These data are only used once in forward and backward propagation, but they also occupy a lot of memory.

To solve the memory redundancy problem, ZeRO adopts two major strategies:

1. **ZeRO-DP (Data Parallelism):**
   For model states, by sharding and distributing optimizer states, gradients, and parameters to multiple data parallel processes, redundancy is eliminated, and communication volume is reduced by using dynamic communication scheduling.

2. **ZeRO-R (Residuals Optimization):**
   For activations and temporary buffers, memory usage is optimized by using sharded activation recomputation, fixed buffer size, and real-time memory fragmentation management.

### ZeRO Sharding Strategy

ZeRO is divided into three stages, each stage further reduces memory redundancy on the basis of the previous stage, thus making it possible to train ultra-large models:

#### ZeRO-1 (Optimizer State Sharding)
- **Principle:**
  - Shard optimizer states (such as Adam's momentum and second-order moments) along the parameter dimension into $P$ shards ($P$ is the number of GPUs), and each GPU only stores the states corresponding to the model parameters it is responsible for.
  - Local Update: Each GPU only updates its locally stored state and parameter shards during the parameter update phase, without additional cross-GPU communication.

#### ZeRO-2 (Gradient Sharding)
- **Principle:**
  - On the basis of optimizer state sharding, gradients are also sharded along the parameter dimension, and each GPU only stores the corresponding gradient shard.
  - Each GPU calculates local gradients and uses efficient Reduce-Scatter operations to aggregate gradients and then update local parameter shards.

#### ZeRO-3 (Parameter Sharding)
- **Principle:**
  - On the basis of ZeRO-1 and ZeRO-2, model parameters (usually 16-bit data) are also sharded, and each GPU only stores the parameter shards corresponding to it.
  - Parameter Collection on Demand: During forward or backward propagation, if a GPU needs complete model parameters, it collects the missing shards from other GPUs. This process is only performed when necessary to reduce communication overhead.

The following figure shows the comparison of model state memory consumption per device in different stages:

{{< figure
    src="deepspeed_zero.png"
    caption="Fig. 23. Comparing the per-device memory consumption of model states, with three stages of ZeRO-DP optimizations. (Image source: [Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054))"
    align="center"
    width="100%"
>}}

### Comparison of DeepSpeed ZeRO Sharding and Offload Strategies

To better understand DeepSpeed's ZeRO strategy, the following compares each stage and Offload scheme:

| **ZeRO Stage** | **Description** | **Memory Footprint** | **Training Speed** |
|----------------|----------|--------------|--------------|
| **ZeRO-0**     | Pure data parallelism, no sharding, all states are fully replicated on each GPU. | Highest | **Fastest** |
| **ZeRO-1**     | Optimizer states are sharded only, gradients and parameters are still replicated. | Higher | Slightly slower than ZeRO-0 |
| **ZeRO-2**     | Optimizer states and gradients are sharded. | Medium | Slower than ZeRO-1 |
| **ZeRO-3**     | Optimizer states, gradients, and model parameters are sharded. | Lowest | Significantly slower than ZeRO-2, affected by model size and network bandwidth |

| **Offload Type**                | **Description** | **Memory Footprint** | **Training Speed** |
|----------------------------------|----------|--------------|--------------|
| **ZeRO-1 + CPU Offload**         | On the basis of ZeRO-1, optimizer states are offloaded to CPU memory, reducing GPU memory footprint, but relying on PCIe bandwidth and occupying CPU memory. | Medium-Low | Slower than ZeRO-1 |
| **ZeRO-2 + CPU Offload**         | On the basis of ZeRO-2, optimizer states are offloaded to CPU memory, further reducing GPU memory footprint for large models, but increasing CPU-GPU data transfer. | Low | Slower than ZeRO-2 |
| **ZeRO-3 + CPU Offload**         | On the basis of ZeRO-3, optimizer states and model parameters are offloaded to CPU, GPU memory footprint is the lowest, but CPU-GPU communication overhead is extremely large. | **Extremely Low** | **Very Slow** |
| **ZeRO-Infinity (NVMe Offload)** | Based on ZeRO-3, states are offloaded to NVMe devices, breaking through CPU memory limits, suitable for ultra-large models; performance is highly dependent on NVMe parallel read and write speed. | **Extremely Low**<br/>NVMe support required | Slower than ZeRO-3, but usually better than CPU Offload scheme |

### Communication Volume and Performance Impact

- **ZeRO-0/1/2:**
  Mainly rely on All-Reduce for gradient synchronization, and the communication volume is relatively low.

- **ZeRO-3:**
  All-Gather/All-Reduce operations are required for model parameters, and the communication volume increases significantly. Network bandwidth becomes a key bottleneck.

- **Offload Strategy (CPU/NVMe):**
  Data transmission is mainly between CPU ↔ GPU or NVMe ↔ GPU. The transmission bandwidth is much lower than the communication between GPUs, which may significantly affect the training speed, especially in ZeRO-3 scenarios.

## Multi-dimensional Parallelism

**Multi-dimensional Parallelism** refers to the organic combination of multiple parallel technologies such as data parallelism, model parallelism, and pipeline parallelism in distributed training to fully utilize the computing resources of modern GPU clusters. Through this "3D parallelism" or "4D parallelism" strategy, not only memory efficiency can be improved, but also computational efficiency can be improved, thereby achieving efficient training of ultra-large-scale (even trillion-parameter level) models.

### 3D Parallelism

With the rapid improvement of the computing power of GPU clusters, training trillion-parameter models is no longer out of reach. DeepSpeed integrates data parallelism, model parallelism, and pipeline parallelism to build a "3D parallelism" strategy. This strategy mainly solves the two major challenges faced by training ultra-large models:

- **Memory Efficiency:**
  Model layers are divided into different pipeline stages, and each stage is further divided by model parallelism, reducing the memory occupied by models, optimizers, and activations. However, it should be noted that model splitting cannot be unlimited, otherwise, the communication overhead will increase significantly, which will affect computational efficiency.

- **Computational Efficiency:**
  To make the number of computing workers exceed the limitations of simple model and pipeline parallelism, and to ensure computational efficiency, DeepSpeed expands with ZeRO-DP (data parallelism based on optimizer state sharding). ZeRO-DP not only further optimizes memory usage but also allocates data parallel groups to devices with local high-bandwidth communication through topology-aware mapping, greatly reducing communication overhead.

The following diagram shows the overall strategy of 3D parallelism:

{{< figure
    src="zero_3d.png"
    caption="Fig. 24. Example 3D parallelism with 32 workers. Layers of the neural network are divided among four pipeline stages. Layers within each pipeline stage are further partitioned among four model parallel workers. Lastly, each pipeline is replicated across two data parallel instances, and ZeRO partitions the optimizer states across the data parallel replicas. (Image source: [Majumder et al. 2020](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/))"
    align="center"
    width="100%"
>}}

Each parallel dimension (data, model, pipeline) is carefully mapped to fully utilize the communication bandwidth within and between nodes. Specific strategies include:
- **Optimize Intra-node Communication:** Since model parallelism has the largest communication overhead, model parallel groups are preferentially arranged within the same node to utilize higher intra-node bandwidth (e.g., using NVIDIA Megatron-LM's tensor sharding method);
- **Data Parallelism and Pipeline Parallelism:** When model parallelism does not cover the entire node, data parallel groups are arranged within the same node as much as possible; pipeline parallelism can be flexibly arranged for cross-node scheduling due to its smaller communication volume.

By reducing the amount of communication data in each data parallel group and increasing the parallelism of local parallel communication, the overall communication bandwidth is effectively amplified.

{{< figure
    src="3d_parallelism.png"
    caption="Fig. 25. Mapping of workers in Figure 24 to GPUs on a system with eight nodes, each with four GPUs. Coloring denotes GPUs on the same node. (Image source: [Majumder et al. 2020](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/))"
    align="center"
    width="100%"
>}}

### 4D Parallelism

To further expand the model scale, Llama3 ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)) adopted a 4D parallel strategy during training. It combines four parallel methods to shard the model in a more fine-grained manner, so that the model parameters, optimizer states, gradients, and activations on each GPU can be adapted to the capacity limit of high-bandwidth memory (HBM). These four parallel methods are:

- **Tensor Parallelism (TP):** Divide a single weight tensor into multiple blocks and distribute them across different devices;
- **Pipeline Parallelism (PP):** Vertically divide the model into multiple stages, and each stage processes different micro-batches in parallel on different devices;
- **Context Parallelism (CP):** Divide the input context into multiple segments to alleviate the memory bottleneck when inputting long sequences;
- **Data Parallelism (DP), usually using Fully Sharded Data Parallelism (FSDP):** Shard models, optimizer states, and gradients, and synchronize after each training step.

The following diagram shows an example of 4D parallelism implemented on 16 GPUs. The position of each GPU is represented by a vector [D1, D2, D3, D4], where each dimension corresponds to a parallel strategy. GPUs are grouped according to four dimensions [TP, CP, PP, DP], and the group size of each dimension is 2. For example, GPU0 and GPU1 belong to the same tensor parallel group; GPU0 and GPU2 belong to the same context parallel group; GPU0 and GPU4 belong to the same pipeline parallel group; GPU0 and GPU8 belong to the same data parallel group:

{{< figure
    src="4d_parallelism.png"
    caption="Fig. 26. Illustration of 4D parallelism. (Image source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="100%"
>}}

Through the 4D parallel strategy, Llama3 can fully utilize the computing resources of multiple GPUs during training, while effectively reducing memory footprint and supporting the training of ultra-large-scale models.

## Memory Optimization Techniques

In addition to parallel training techniques, there are many memory optimization techniques designed to help train LLMs. These designs mainly start from reducing the memory footprint of each stage in the training process.

### CPU Offloading

CPU Offloading ([Rhu et al. 2016](https://arxiv.org/abs/1602.08124)) refers to a common and intuitive practice of offloading data or tensors that are temporarily not needed to the CPU when GPU memory is insufficient and loading them back to the GPU when needed. Its main purpose is to use the larger capacity of CPU memory to expand available space, so that larger-scale models can be trained even in memory-constrained environments. However, this method will bring additional data transmission overhead and usually reduce training speed, so its application has been relatively reduced in recent years.

1. **Identify Offloadable Tensors:** Identify tensors that are temporarily not used during training, such as model parameters, optimizer states, intermediate activations, etc. The basis for judging whether a tensor can be offloaded can be the frequency of use, life cycle, etc. of the tensor.
2. **Tensor Offloading:** Move offloadable tensors from GPU memory to CPU memory. Data transmission is usually performed through the PCIe bus.
3. **Tensor Prefetching:** Before needing to use tensors offloaded to CPU memory, load the tensors from CPU memory back to GPU memory in advance. Prefetching operations can be performed in parallel with GPU computing operations to reduce data loading latency.
4. **Tensor Usage:** The GPU uses tensors loaded back into GPU memory for computation.
5. **Tensor Re-offloading:** After the tensor is used up, if the tensor is no longer needed for a period of time, it can be offloaded to CPU memory again to release GPU memory space.

ZeRO-Offload and ZeRO-Infinity are memory optimization technologies based on CPU offloading implemented in the DeepSpeed library. ZeRO-Offload offloads optimizer states to CPU memory, and ZeRO-Infinity goes further, offloading model parameters to CPU memory or even NVMe disks, breaking through the GPU memory wall and supporting the training of larger-scale models.

The following figure intuitively shows the memory optimization technology of **Heterogeneous system**:

{{< figure
    src="heterogenous_system.png"
    caption="Fig. 27. Heterogenous system illustration. (Image source: [Clolossal-AI Documentation](https://colossalai.org/docs/concepts/paradigms_of_parallelism))"
    align="center"
    width="100%"
>}}

### Activation Recomputation/Gradient Checkpointing

Activation Recomputation/Gradient Checkpointing ([Chen et al. 2016](https://arxiv.org/abs/1604.06174)) is a **technology that trades computation for memory**. During training, only part of the activation values are saved (e.g., the input activation values of each Transformer layer). During backpropagation, the unsaved activation values are recomputed. Activation recomputation can significantly reduce the activation memory footprint during training, especially when training deep neural networks.

1. **Select Checkpoints:** Select some layers in the model as checkpoints. Usually, key layers in the model are selected as checkpoints, such as the input layer of the Transformer layer.
2. **Forward Pass:** During forward propagation, only the activation values of checkpoint layers are saved. For non-checkpoint layers, the activation values are immediately released after calculation and not saved.
3. **Backward Pass:** During backpropagation, when it is necessary to calculate the gradient of a non-checkpoint layer, forward propagation is performed again first to calculate the activation value of the layer (recomputation), and then backward propagation is performed to calculate the gradient. For checkpoint layers, since the activation values of checkpoint layers have been saved, the saved activation values can be directly used for backpropagation without recomputation.

The following is a memory cost analysis of activation recomputation. For ease of analysis, assume that the model has a total of $n$ network layers and divides them **evenly** into $k$ segments. In this way, each segment contains approximately $n/k$ network layers. When doing activation recomputation, we only save the activation values at the boundaries of each segment (i.e., checkpoints), and recompute the rest when needed. The following function represents the memory requirement during training:

$$
\text{cost-total} \;=\; \max_{i=1,\ldots,k}\bigl[\text{cost-of-segment}(i)\bigr] \;+\; O(k)
\;=\; O\Bigl(\tfrac{n}{k}\Bigr) \;+\; O(k).
$$

Next, consider how to choose the optimal $k$ to minimize $f(k)$ given $n$:

$$
f(k) \;=\; \frac{n}{k} \;+\; k.
$$

Take the derivative of $f(k)$ with respect to $k$ and set it to 0 (only consider $k>0$):

$$
f'(k) \;=\; -\frac{n}{k^2} \;+\; 1 \;=\; 0
\quad\Longrightarrow\quad
k^2 = n
\quad\Longrightarrow\quad
k = \sqrt{n}.
$$

Substituting $k = \sqrt{n}$, we can get the minimum memory overhead of approximately

$$
f(\sqrt{n}) \;=\; \frac{n}{\sqrt{n}} \;+\; \sqrt{n}
\;=\; 2\sqrt{n}.
$$

Therefore, the overall peak memory requirement of this approach can be reduced to the order of $O(\sqrt{n})$ (compared to the $O(n)$ memory of generally directly saving all activations), which is why activation recomputation technology can bring "sublinear" memory footprint. The following figure intuitively shows the effect of this trick.

{{< figure
    src="activation_recomputation.png"
    caption="Fig. 28. The memory cost of different memory saving algorithms. Sharing: Memory used by intermediate results is recycled when no longer needed. Inplace: Save the output directly into memory of an input value. (Image source: [Chen et al. 2016](https://arxiv.org/abs/1604.06174))"
    align="center"
    width="100%"
>}}

It should be noted that activation recomputation requires additional forward recomputation in the **backward propagation** stage. Each segment needs to perform forward computation of $n/k$ layers. If the network is divided into $k$ segments, the total recomputation during backpropagation is approximately $k \times \bigl(n/k\bigr) = n$ layers of forward operations, which is equivalent to doing approximately one more "forward propagation" in each training iteration. This is usually acceptable in LLM training because:

- Compared to quickly exhausting GPU memory and making it impossible to train large-scale models, this additional cost in computation is usually more bearable.
- When the model is very deep ($n$ is large), using activation recomputation technology can significantly reduce memory usage from $O(n)$ to $O(\sqrt{n})$, making it possible to train more and deeper large models on given hardware.

### Mixed Precision Training

Mixed Precision Training ([Micikevicius al. 2017](https://arxiv.org/abs/1710.03740)) is a technology that simultaneously uses low-precision floating-point numbers (such as FP16 or BF16) and high-precision floating-point numbers (such as FP32) during model training. Its core goal is to **reduce memory footprint** and **accelerate training** while maintaining model accuracy comparable to full-precision training as much as possible.

Modern GPUs have higher throughput and lower memory footprint in low-precision computing, thereby reducing memory access overhead and memory bandwidth requirements. Mixed-precision training can significantly improve training speed. The following figure shows the basic process of mixed-precision training in a network layer: forward and backward propagation mainly use half-precision (FP16) operations, while gradient accumulation and parameter updates use full-precision (FP32) to avoid numerical precision problems that may be caused by low-precision computing.

{{< figure
    src="mixed_precision.png"
    caption="Fig. 29. Mixed precision training iteration for a layer. (Image source: [Micikevicius al. 2017](https://arxiv.org/abs/1710.03740))"
    align="center"
    width="100%"
>}}

Mixed-precision training mainly relies on the following three key technologies:

1. **Full-Precision Master Copy of Weights**
   To prevent gradients from being truncated to zero due to being too small in magnitude under FP16, a master copy of FP32 weights is maintained during training. The specific process is:
   - **Initialization:** Use FP32 weights as the master copy of the model;
   - **Forward/Backward Propagation:** Before each iteration starts, convert FP32 weights to FP16 for forward propagation and backward propagation to calculate FP16 gradients;
   - **Parameter Update:** Before updating parameters, convert FP16 gradients to FP32 and use them to update the FP32 master copy.

   This design not only utilizes the efficiency of low-precision computing but also ensures the accuracy of parameter updates.

2. **Loss Scaling**
To avoid gradient underflow due to the limited representation range of low precision, the loss value is usually amplified before backpropagation. The specific process is:
- Use FP32 to calculate the loss $L$;
- Multiply the loss by a scaling factor $S$ to get $L' = L \times S$, and then perform backpropagation to calculate FP16 gradients;
- Before parameter update, divide the gradient by $S$ to restore it to the true gradient.

The choice of scaling factor is crucial: too small may not avoid gradient underflow, and too large may cause gradient overflow. Dynamic loss scaling technology can automatically adjust the scaling factor according to the actual situation of gradients during training.

As shown in the figure below, amplifying the loss makes the gradient distribution more concentrated in the higher numerical part, thereby retaining the subtle information that may be truncated under low-precision representation.

{{< figure
    src="mixed_precision_fp16.png"
    caption="Fig. 30. The histogram of gradients in full precision. The left part up to $2^{-24}$ will be zero-ed off once the model switches to FP16. (Image source: [Micikevicius al. 2017](https://arxiv.org/abs/1710.03740))"
    align="center"
    width="100%"
>}}

3. **Arithmetic Precision Control**
   For operations with high precision requirements (such as vector dot product and summation reduction), FP32 can be used for accumulation calculation, and then converted to FP16 for storage; for element-wise operations, FP16 or FP32 can be selected according to specific needs.

### Compression

In the deep learning training process, intermediate results (such as activation values and gradient information), although only used once in forward propagation and once in backward propagation, often occupy a lot of memory. Considering that there is a significant time interval between two uses, data can be **compressed** after the first use, and then decompressed when needed later, thereby effectively reducing memory footprint.

Compression technology is mainly applied to two scenarios:

- **Activation Value Compression:** Compress activation values after forward propagation and decompress before backward propagation. This is especially important for deep neural networks because activation values usually occupy a lot of memory.
- **Gradient Compression:** Compress gradients after calculating gradients in backpropagation and before gradient synchronization to reduce the amount of data for cross-GPU communication, thereby improving distributed training efficiency.

Compression technology can be divided into two categories:

1. **Lossless Compression:**
   Methods such as Huffman coding or Lempel-Ziv algorithm are used to ensure that the decompressed data is completely consistent with the original data. However, due to the low compression rate, its memory saving effect is limited.

2. **Lossy Compression:**
   Algorithms such as JPEG or MPEG are used to obtain higher compression rates on the premise of allowing certain data loss. This method can significantly reduce memory footprint, but may have a certain impact on model accuracy and convergence.

Gist ([Jain et al. 2018](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf)) is a memory optimization technology for activation value compression. Its core lies in using data encoding strategies to compress intermediate results, mainly including two encoding schemes:

- **Layer-Specific Lossless Encoding:**
  Design special lossless encoding schemes for specific layer structures (such as ReLU-Pool and ReLU-Conv):
  - For ReLU-Pool layers, binary encoding can be used;
  - For ReLU-Conv layers, sparse storage and dense computation encoding are used.

- **Aggressive Lossy Encoding:**
  Delayed Precision Reduction (DPR) technology is used. The core idea of DPR is: activation values need to maintain high precision during forward propagation, while lower precision can be tolerated during backward propagation. Therefore, activation values are compressed to lower precision after forward propagation, and then decompressed to high precision before backward propagation.

### Memory-Efficient Optimizers

Traditional optimizers (such as Adam, SGD with Momentum) need to maintain a large amount of state data (such as momentum and variance) for each model parameter during training. Their memory footprint is often comparable to or even higher than the model parameter size. For example, taking the Adam optimizer ([Kingma et al. 2014](https://arxiv.org/pdf/1412.6980)) as an example, each parameter needs to store the first-order moment and the second-order moment. Adding up with the parameter itself and its gradient, the entire training process requires approximately 4 times the memory of the model weights, which poses a severe challenge to large model training.

To reduce memory consumption, memory-efficient optimizers are mainly designed through the following strategies:
- **Reduce the Number of State Variables:** Only save necessary statistical information instead of complete matrices;
- **Reduce the Precision of State Variables:** Use FP16 or bfloat16 for storage;
- **Share State Variables:** Share part of the state information between multiple parameters.

#### Adafactor

Adafactor ([Shazeer et al. 2018](https://arxiv.org/abs/1804.04235)) is a memory-efficient adaptive learning rate optimizer. Unlike Adam, Adafactor does not store the complete second-order moment estimation matrix, but only stores two vectors (row and column statistics) to replace the complete second-order moment matrix, which significantly reduces memory footprint, especially suitable for scenarios where the parameter matrix has a low-rank structure.

#### SM3

SM3 (Sparse Momentum for Massive Models) ([Anil et al. 2019](https://arxiv.org/abs/1905.11286)) provides a memory-efficient adaptive optimization scheme through sparse updates and state sharing.

- **Sparse Momentum:** Only update Momentum for parameters with non-zero gradients, thereby reducing computation and storage overhead;
- **State Sharing:** To a certain extent, allow different parameters to share state variables, further reducing memory consumption;
- **Adaptive Learning Rate:** Dynamically adjust the learning rate according to the gradients of each parameter, improving the stability and convergence speed of model training.

### LoRA

LoRA (Low-Rank Adaptation) ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) proposes to introduce **low-rank adapters** next to pre-trained weights to achieve efficient fine-tuning by adding a small number of parameters without interfering with the original inference ability of the pre-trained model.

The following figure intuitively shows the principle and initialization strategy of LoRA:

{{< figure
    src="lora.png"
    caption="Fig. 31. An illustration of LoRA. (Image source: [Hu et al., 2021](https://arxiv.org/abs/2106.09685))"
    align="center"
    width="70%"
>}}

Assuming that the pre-trained weight matrix is $ \mathbf{W} \in \mathbb{R}^{d \times k} $. LoRA adds a low-rank update term $ \Delta \mathbf{W} = \mathbf{B}\mathbf{A} $ to it to obtain new weights:

$$
\mathbf{W}' = \mathbf{W} + \alpha\, \mathbf{B}\mathbf{A},
$$

where:
- $ \mathbf{A} \in \mathbb{R}^{r \times d} $ is the dimensionality reduction matrix;
- $ \mathbf{B} \in \mathbb{R}^{k \times r} $ is the dimensionality increase matrix;
- $ r \ll \min(d,k) $ is the low-rank dimension, generally taking values of 1, 2, 4, or 8;
- $ \alpha $ is an adjustable scaling factor.

During fine-tuning, **the original weight $ \mathbf{W} $ is frozen and unchanged**, and only $ \mathbf{A} $ and $ \mathbf{B} $ are updated. Since $ r $ is much smaller than $ d $ or $ k $, the number of parameters that need to be trained is greatly reduced.

To ensure that the impact of the introduced $ \Delta \mathbf{W} = \mathbf{B}\mathbf{A} $ on the original model is as small as possible in the early stage of fine-tuning, $ \Delta \mathbf{W} \approx 0 $ is required. Common practices are as follows:

1. **Initialization of Dimensionality Reduction Matrix $ \mathbf{A} $**
   - **Gaussian Initialization**
     Let $ \mathbf{A} \sim \mathcal{N}(0,\sigma^2) $ (generally $ \sigma $ takes a smaller value, such as 0.02). This can ensure that the initial update amount is small and does not seriously interfere with the model output.
   - **Kaiming (He) Initialization**
     Kaiming initialization is a weight initialization method specially designed for deep networks. Its goal is to maintain the stability of forward signals and backward gradients between network layers. For LoRA, as long as it is ensured that the scale is small (or with a suitable scaling factor), the initial $ \Delta \mathbf{W} $ can be made closer to zero.

2. **Initialization of Dimensionality Increase Matrix $ \mathbf{B} $**
   - Usually, $ \mathbf{B} $ is initialized as a zero matrix, so that $ \mathbf{B}\mathbf{A} = 0 $ initially, further ensuring that the impact on the original model is minimal.

Training with LoRA has the following advantages:

- **Parameter Efficiency:** Only low-rank adapter parameters are introduced, reducing the total number of parameters that need to be trained and stored.
- **Memory and Computational Efficiency:** Most pre-trained weights are frozen, and only small-scale parameters are updated during fine-tuning, significantly reducing memory footprint and computing power overhead.
- **No Additional Inference Latency:** After training is completed, the update term $ \Delta \mathbf{W} $ can be merged back into the original weights, which will not increase the amount of calculation in the inference stage.

### QLoRA

QLoRA ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) is a method for efficient fine-tuning of large-scale models based on LoRA combined with quantization ideas. Through the following three key improvements, it greatly reduces memory footprint while maintaining basically unchanged model accuracy:

1. **4-bit Normal Float (NF4) Quantization**
   A block-based quantile quantization strategy is adopted to quantize the original model weights to 4 bits, thereby achieving significant storage compression with subtle loss of accuracy.

2. **Double Quantization**
   After performing quantization once on ordinary parameters, perform an additional quantization on the quantization constants to further reduce cache footprint.

3. **Paged Optimizer**
   When memory usage is too high, automatically transfer part of the optimization process to CPU memory, thereby alleviating GPU memory pressure and improving scalability.

Different from traditional LoRA, which only reduces the number of parameters to be fine-tuned, QLoRA also **compresses** all weights through 4-bit quantization, thereby maximizing the reduction of memory footprint and data transmission overhead while ensuring near-original accuracy.

{{< figure
    src="qlora.png"
    caption="Fig. 32. Different finetuning methods and their memory requirements. QLoRA improves over LoRA by quantizing the transformer model to 4-bit precision and using paged optimizers to handle memory spikes. (Image source: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314))"
    align="center"
    width="100%"
>}}

This method can be regarded as a further extension of LoRA: LoRA improves efficiency by reducing the number of weights that need to be fine-tuned, while QLoRA, on this basis, quantizes all weights (including the un-fine-tuned part) to 4-bit precision, achieving **dual compression of storage and computation** in general, which is suitable for efficient fine-tuning of LLMs in resource-constrained environments.

## Summary

Parallel techniques and memory optimization strategies need to be weighed and selected according to the specific model structure, dataset size, hardware resources, and training goals. Usually, it is necessary to combine multiple technologies to effectively train large-scale models and achieve the best performance and efficiency.

## References

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

## Citation

> **Citation**: When reprinting or citing the content of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui.(Mar 2025). Parallel and Memory Optimization Techniques for Training Large Models.
https://syhya.github.io/posts/2025-03-01-train-llm

Or

```bibtex
@article{syhya2025train-llm,
  title   = "Parallel and Memory Optimization Techniques for Training Large Models",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Mar",
  url     = "https://syhya.github.io/posts/2025-03-01-train-llm"
}