---
title: "LLaMA 系列模型"
date: 2025-04-06T12:00:00+08:00
author: "Yue Shui"
tags: ["LLaMA", "AI", "NLP", "LLM", "Pre-training", "Post-training"]
categories: ["技术博客"]
readingTime: 40
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

## LLaMA

Meta AI 推出的 LLaMA 系列开源模型已成为大语言模型社区的基石之一，对推动开放研究和应用产生了深远影响。从 2023 年初发布的开创性 LLaMA，到同年性能显著提升的 LLaMA 2，再到针对特定领域（如代码、安全）的衍生模型，以及 2024 年和 2025 年相继推出的新一代 LLaMA 3 和 LLaMA 4，Meta 持续致力于提升开源模型的性能，使其逐步逼近最先进的闭源模型。下面，我们将依次介绍每个主要模型的关键技术细节。

### LLaMA 1

**LLaMA 1** ([Touvron et al., 2023a](https://arxiv.org/abs/2302.13971)) 于 2023 年 2 月发布，是 Meta 开源的首个基础语言模型系列。LLaMA 提供了 7B、13B、33B、65B 四种参数规模，其核心特点在于完全使用**公开可用的数据集**进行训练，未依赖任何专有数据。尽管参数量远小于当时的 GPT-3 (175B)，LLaMA 13B 模型在多数基准测试上超越了 GPT-3，而 65B 模型则达到了与 Chinchilla-70B 和 PaLM-540B 等顶尖模型相媲美的性能水平。

{{< figure
    src="llama1_bechmark.png"
    caption="Fig. 1. Zero-shot performance of LLaMA models on Common Sense Reasoning tasks compared to other foundation models. (Source: [Touvron et al., 2023a](https://arxiv.org/abs/2302.13971))"
    align="center"
    width="100%"
>}}

**训练数据：** LLaMA 1 的训练基于大规模公开语料。其中，65B 和 33B 模型使用了约 **1.4 万亿 tokens**，而 7B 和 13B 模型则使用了约 **1 万亿 tokens**。语料来源广泛，主要包括 Common Crawl、C4、GitHub、Wikipedia、Books、ArXiv 和 StackExchange，覆盖了多种领域和约 20 种语言（以英语为主）。

**架构设计：** LLaMA 1 采用了标准的 Transformer 解码器架构，并引入了以下关键改进以提升性能和训练效率：

*   **Pre-normalization & RMSNorm：** 采用 **Pre-normalization** 结构（在每个子层输入前进行归一化），并使用 **RMSNorm (Root Mean Square Normalization)** 替代标准的 LayerNorm。RMSNorm 通过省略均值中心化步骤，仅依据向量元素的均方根进行缩放，从而降低了计算复杂度，同时有效维持了训练过程的稳定性。
*   **SwiGLU 激活函数：** 将前馈网络（FFN）中的激活函数从 ReLU 替换为 **SwiGLU (Swish-Gated Linear Unit)**。SwiGLU 结合了 Swish 激活函数的平滑非线性和门控机制，增强了模型的表达能力。同时，LLaMA 调整了 FFN 的隐藏层维度（使用 $ \frac{2}{3} \times 4d $ 而非标准的 $4d$，其中 $d$ 是模型维度），以在引入门控参数的同时，大致保持 FFN 层的总参数量和计算量不变。
*   **RoPE 旋转位置编码：** 采用 **Rotary Position Embeddings (RoPE)** 进行位置编码。RoPE 通过对 Query 和 Key 向量施加与位置相关的旋转操作，将相对位置信息有效融入自注意力计算中，增强了模型处理长序列和捕捉长距离依赖关系的能力。LLaMA 1 的最大上下文长度为 2048 tokens。
*   **高效注意力实现：** 利用 Meta 开源的 [xformers](https://github.com/facebookresearch/xformers) 库实现了内存高效且计算优化的因果多头注意力机制 (causal multi-head attention)。

**微调对话模型：** LLaMA 1 发布时主要提供预训练模型权重（限制商业用途），并未包含官方的对话微调版本。然而，开源社区迅速在其基础上进行了探索，例如 **Stanford Alpaca** ([Taori et al., 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html)) 项目证明了仅需少量指令数据进行监督微调（SFT），即可赋予 LLaMA 基础模型强大的对话能力，极大地促进了开源 LLM 的研究与应用生态。

{{< figure
    src="alpaca.png"
    caption="Fig. 2. The pipeline for generating instruction-following demonstrations and training Alpaca 7B based on LLaMA 7B. (Source: [Taori et al., 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html))"
    align="center"
    width="100%"
>}}

**训练稳定性与 Loss 突刺 (Loss Spike)**

{{< figure
    src="llama1_train_loss.png"
    caption="Fig. 3. Training loss curves over processed tokens for the LLaMA 7B, 13B, 33B, and 65B models. (Source: [Touvron et al., 2023a](https://arxiv.org/abs/2302.13971))"
    align="center"
    width="80%"
>}}

从图 3 可以观察到，LLaMA 模型在训练过程中的 Loss 总体呈现下降趋势，表明训练过程相对稳定。然而，在训练 13B、33B 和 65B 模型时，均出现了 **Loss 突刺 (Loss Spike)** 现象，即训练损失在某个时间点突然异常飙升。模型规模越大，突刺现象似乎越显著，且可能在训练过程中多次出现。

*   **现象描述：** Loss Spike 指的是在模型训练期间，损失函数值出现短暂、急剧且异常的增高。
*   **潜在原因：** 通常与多种因素相关，包括训练数据中存在的**异常样本或分布突变**、**学习率设置不当**（过高或衰减策略问题）、**优化器（如 Adam）内部状态与梯度剧烈变化的相互作用**，以及**混合精度训练中的数值不稳定性**（例如梯度溢出或下溢）。
*   **常见应对策略：** 解决或缓解 Loss Spike 的方法包括：加强数据清洗和预处理；应用**梯度裁剪 (Gradient Clipping)** 限制梯度范数；精细调整学习率调度策略（如 Warmup、Decay）；优化混合精度训练配置；以及在发生突刺后，从最近的检查点恢复训练，并可能跳过导致问题的特定数据批次。

### LLaMA 2

**LLaMA 2** ([Touvron et al., 2023b](https://arxiv.org/abs/2307.09288)) 于 2023 年 7 月推出，是 LLaMA 1 的重要升级版本。相较于第一代，LLaMA 2 在模型规模、训练数据量、上下文长度以及模型对齐方面均有显著改进，并首次发布了官方的对话优化版本 **LLaMA 2-Chat**，且开放了商业使用许可。

**架构与优化：** LLaMA 2 的基础架构很大程度上继承了 LLaMA 1 的成功设计（如 RMSNorm, SwiGLU, RoPE）。主要的技术更新包括：

*   **分组查询注意力 (GQA)：** 针对较大的 **34B 和 70B 模型**，采用了**Grouped Query Attention (GQA)**。GQA 是一种介于多头注意力 (MHA) 和多查询注意力 (MQA) 之间的折中方案，它允许多个查询头（Query heads）共享同一组键（Key）和值（Value）头。这显著减少了推理过程中 KV 缓存的内存占用和计算开销，从而提高了大模型的推理速度和部署效率，同时对模型性能的影响很小。
*   **上下文长度提升：** 将模型的最大上下文长度从 LLaMA 1 的 2048 tokens 扩展到了 **4096 tokens**。这使得模型能够处理更长的文本输入，增强了其在长文档问答、摘要、长对话等任务上的能力。

**训练数据与规模：** LLaMA 2 使用了更大规模的预训练数据，总量达到约 **2 万亿 tokens**，相比 LLaMA 1 增加了约 40%。数据来源更加多样化，并进行了更严格的筛选和清洗。

**Post-Training (LLaMA 2-Chat)：** **LLaMA 2-Chat** 是经过精心对齐的对话模型。其训练流程始于 LLaMA 2 预训练基础模型，主要包含以下阶段：

1.  **监督微调 (Supervised Fine-tuning, SFT)：** 使用高质量的指令和对话样本对预训练模型进行微调，使其初步具备遵循指令和进行对话的能力。
2.  **人类反馈强化学习 (Reinforcement Learning from Human Feedback, RLHF)：** 这是提升模型有用性（Helpfulness）和安全性（Safety）的关键步骤。
    *   **奖励模型训练 (Reward Modeling)：** 收集大量人类偏好数据（即对模型生成的多个回答进行排序），训练一个或多个奖励模型来学习评估哪个回答更符合人类偏好（在有用性和安全性维度上）。
    *   **RL 优化：** 利用训练好的奖励模型作为回报信号，通过 **PPO** 和 **拒绝采样** 来进一步优化 SFT 后的模型。PPO 旨在最大化奖励信号，而拒绝采样则通过从模型生成的 K 个样本中选出奖励最高的一个进行梯度更新，进一步提升模型质量。这个过程通常是迭代进行的，不断收集新的偏好数据来改进奖励模型和对话模型本身。参考了 Anthropic 提出的 Constitutional AI 和 HH-RLHF ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) 的思想。可在 Hugging Face 上获取相关的 [HH-RLHF 数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)。

{{< figure
    src="llama2_chat_rlhf.png"
    caption="Fig. 4. Illustration of the Llama 2-Chat fine-tuning process, including SFT and RLHF stages with rejection sampling and PPO. (Source: [Touvron et al., 2023b](https://arxiv.org/abs/2307.09288))"
    align="center"
    width="100%"
>}}

### Code Llama

**Code Llama** ([Rozière et al., 2023](https://arxiv.org/abs/2308.12950)) 是 Meta 在 LLaMA 2 基础上于 2023 年 8 月发布的**专注于代码能力**的大型语言模型系列。通过在海量的编程代码数据上进行额外的持续预训练和特定任务微调，Code Llama 在代码生成、代码补全、代码理解和调试等任务上展现出卓越的能力。

{{< figure
    src="codellama.png"
    caption="Fig. 5. The Code Llama specialization pipeline, starting from Llama 2 and involving code-specific training stages. (Source: [Rozière et al., 2023](https://arxiv.org/abs/2308.12950))"
    align="center"
    width="100%"
>}}

**训练与数据：** Code Llama 以 LLaMA 2 的权重为起点，使用了 **5000 亿 tokens** 的代码及代码相关自然语言语料进行继续预训练（针对 7B/13B/34B 版本）或 **1 万亿 tokens**（针对 70B 版本）。训练数据主要来源于公开的代码仓库和数据集。关键的技术改进包括：

*   **长上下文微调 (Long Context Fine-tuning, LCFT)：** Code Llama 在训练中特别关注长序列处理能力，将序列长度扩展到 **16k tokens** 进行训练。为了更好地处理长距离依赖，通过调整 RoPE 位置编码的基数 $\theta$（从 LLaMA 2 的 10,000 增加到 1,000,000），减缓了随着 token 距离增大注意力分数的衰减。这使得模型在推理时能够稳定地处理长达 **100k tokens** 的超长上下文。

{{< figure
    src="codellama_rope.png"
    caption="Fig. 6. Effect of RoPE base period scaling on perplexity for long sequences, showing improved performance with a larger base. (Source: [Rozière et al., 2023](https://arxiv.org/abs/2308.12950))"
    align="center"
    width="70%"
>}}

*   **填充中间 (Fill-in-the-Middle, FIM)：** 训练中引入了**填充中间**任务。模型需要根据给定的代码前缀和后缀，在中间插入合适的代码片段。这种能力对于集成开发环境（IDE）中的代码补全功能至关重要。

**模型变体：** Code Llama 提供了多个版本以满足不同场景的需求：

*   **Code Llama (Base):** 基础代码模型，擅长代码补全和从自然语言生成代码。
*   **Code Llama - Python:** 在基础模型之上，针对 Python 语言额外进行了 1000 亿 tokens 的专门微调，显著增强了 Python 相关任务的性能。
*   **Code Llama - Instruct:** 在代码相关的指令和人类反馈数据上进行了微调（约 50 亿 tokens），使其能更好地理解自然语言指令来生成、解释或修改代码，更适合作为代码助手使用。

每个版本均提供 7B、13B、34B、70B 四种参数规模。

### Llama Guard

**Llama Guard** ([Inan et al., 2023](https://arxiv.org/abs/2312.06674)) 是 Meta 于 2023 年 12 月推出的一款专门用于保障人机对话内容安全的模型。它旨在对用户输入（prompt）和模型输出（response）进行内容审查和风险分类。

{{< figure
    src="llama_guard.png"
    caption="Fig. 7. Example task instructions for the Llama Guard prompt and response classification tasks, demonstrating its safety assessment capability. (Source: [Inan et al., 2023](https://arxiv.org/abs/2312.06674))"
    align="center"
    width="100%"
>}}

**模型概况：** Llama Guard 基于 LLaMA 2-7B 模型，通过**指令微调**的方式专门训练用于**安全风险分类**任务。它并非生成式模型，而是接收一段文本输入，判断其内容是否安全，并能根据预定义的安全风险分类体系（taxonomy）输出具体的风险类别标签。

**训练与分类体系：** Meta 构建了一个包含多种不安全内容类别（如暴力、仇恨言论、性内容、非法行为推广等）的分类体系，并收集了高质量的标注数据进行训练。Llama Guard 能够进行多标签分类，识别文本中可能同时存在的多种风险。由于采用了指令微调范式，用户可以通过设计不同的提示词（prompt）来灵活调整安全策略或定制分类标准。Llama Guard 可以作为过滤器部署在对话系统的输入端（检测用户输入风险）和输出端（检测模型生成内容的风险）。

### Llama Guard 3 Vision

**Llama Guard 3 Vision** ([Chi et al., 2024](https://arxiv.org/abs/2411.10414)) 是 Llama Guard 的多模态升级版本，基于 **Llama-3.2-Vision** 模型构建。它能够同时评估**图像和文本**内容的安全风险，将安全防护能力扩展到了多模态场景。该模型使用特殊的 `<|image|>` token 来整合图像信息，进行统一的多模态安全审查。

{{< figure
    src="llama_guard_vision.png"
    caption="Fig. 8. Llama Guard 3 Vision classifying harmful content in a multimodal response classification task involving both image and text. (Source: [Chi et al., 2024](https://arxiv.org/abs/2411.10414))"
    align="center"
    width="100%"
>}}

Llama Guard 3 Vision 采用了 ML Commons 定义的安全风险分类标准 ([Vidgen et al., 2024](https://arxiv.org/abs/2404.12241))，并在此基础上扩展，增加了对代码解释器滥用风险的检测（S14 类别）。

{{< figure
    src="hazard_categories.png"
    caption="Fig. 9. The 14 hazard categories used by Llama Guard 3 Vision, based on the MLCommons taxonomy with an added category for code interpreter abuse. (Source: [Meta Llama, 2024](https://huggingface.co/meta-llama/Llama-Guard-3-8B))"
    align="center"
    width="60%"
>}}

基准测试结果显示，Llama Guard 3 Vision 在 MLCommons 安全基准上，无论是在检测用户输入风险还是模型输出风险方面，其多项指标均优于 GPT-4o 和 GPT-4o mini 等先进模型。

{{< figure
    src="ml_commons_benchmark.png"
    caption="Fig. 10. Performance comparison of various models on the MLCommons hazard taxonomy internal test set, showing Llama Guard 3 Vision's strong results. (Source: [Chi et al., 2024](https://arxiv.org/abs/2411.10414))"
    align="center"
    width="100%"
>}}

### LLaMA 3

**LLaMA 3** ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)) 是 Meta 自 2024 年 4 月起陆续发布的新一代开源大模型系列，其在性能、规模、多语言能力、多模态支持和训练效率进行了优化。

**模型规模与版本演进：** LLaMA 3 系列覆盖了从小型到超大规模的广泛参数范围：

*   **LLaMA 3 (初版, 2024/04):** 首先发布了 8B 和 70B 两种规模的预训练和指令微调模型。
*   **LLaMA 3.1 (2024/07):** ([Meta AI, 2024](https://ai.meta.com/blog/meta-llama-3-1/)) 推出了 **405B** 参数的旗舰模型，其性能在多项基准上接近 GPT-4 水平，同时更新了 8B 和 70B 版本。
*   **LLaMA 3.2 (2024/10):** 引入了**轻量化模型**（如 1B, 3B, 11B, 13B），专为边缘设备（如手机、手表、智能家居）优化，并发布了**多模态视觉模型**（如 Llama-3.2-11B-Vision 和 Llama-3.2-90B-Vision）。

{{< figure
    src="llama3_key_hyperparameters.png"
    caption="Fig. 11. Overview of the key hyperparameters for Llama 3 models of different scales. (Source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="70%"
>}}

从上图可以看出，训练更大规模的 LLM 通常需要采用更小的峰值学习率。这主要是因为：
1.  **优化景观复杂性与梯度稳定性：** 参数量越大，模型的损失函数景观越复杂、非凸性越强，对参数更新的敏感度也越高。较小的学习率有助于限制每次更新的步长，避免在陡峭区域产生过大的梯度导致训练震荡或发散，从而确保优化过程更稳定地收敛。
2.  **避免过拟合与提升泛化能力：** 大模型容量更大，更容易在训练数据上过拟合。较小的学习率使得模型以更慢、更稳健的方式学习数据中的模式，减少了对训练数据中噪声或局部特征的过度拟合风险，有助于提升模型在未见数据上的泛化性能。
3.  **精细搜索与参数调整：** 在高维参数空间中，最优解可能位于狭窄区域。小学习率允许优化算法进行更精细的搜索，逐步逼近最优解，避免因步长过大而“跳过”最优区域，从而可能达到更高的最终模型精度。

{{< figure
    src="llama3_architecture.png"
    caption="Fig. 12. Comparison of the high-level architecture between Llama 2 and Llama 3. (Source: [Umar Jamil's PyTorch Llama Slides](https://github.com/hkproj/pytorch-llama/blob/main/Slides.pdf))"
    align="center"
    width="70%"
>}}

**架构与技术创新：** LLaMA 3 在 LLaMA 2 的基础上进行了多方面的显著增强：

*   **超大规模预训练数据：** 预训练数据量达到了惊人的 **15 万亿 tokens**，是 LLaMA 2 的 7.5 倍。数据来源更加广泛，质量更高，多样性更强，并显著增加了非英语语言（如德语、法语、西班牙语、印地语等，占总数据 5% 以上）和代码数据的比例。
*   **优化的 Tokenizer：** 采用了基于 `tiktoken` 库实现的新分词器，**词汇表大小从 LLaMA 2 的 32k 大幅扩展到 128k**。更大的词汇表提高了对多种语言（尤其是非拉丁语系）和代码的编码效率，平均能将输入序列长度减少约 15%，从而间接提升了模型的处理效率和性能。
*   **扩展的上下文长度：** LLaMA 3 初版（8B, 70B）支持 8k tokens 的上下文窗口。而 **LLaMA 3.1 (405B) 将最大上下文窗口进一步提升至 128k tokens**，极大地增强了模型处理长文档、长对话历史和复杂上下文推理的能力。这通常通过结合 RoPE 频率调整、注意力机制优化（如 FlashAttention）等技术实现。
*   **全面应用 GQA：** 与 LLaMA 2 仅在较大模型使用 GQA 不同，LLaMA 3 的**所有规模模型（包括 8B）都采用了 Grouped Query Attention (GQA)**，以优化推理时的内存占用和计算速度。
*   **先进的对齐技术：** 在指令微调（Post-training）阶段，LLaMA 3 结合了监督微调 (SFT)、拒绝采样 (Rejection Sampling) 和直接偏好优化 (Direct Preference Optimization, DPO) 等多种先进技术，旨在全面提升模型的指令遵循能力、有用性（Helpfulness）和安全性（Safety）。
*   **多模态整合 (LLaMA 3.2)：** 通过引入视觉编码器（Vision Encoder）并进行联合训练，实现了图像和文本的融合处理，推出了 Llama-3.2-Vision 系列视觉语言模型。
*   **轻量化模型 (LLaMA 3.2)：** 针对资源受限的边缘计算场景，通过模型压缩技术（如剪枝、蒸馏）推出了 1B、3B 等小型化模型，在性能和资源消耗之间取得了良好平衡。

{{< figure
    src="llama3_post_training.png"
    caption="Fig. 13. Illustration of the overall post-training approach for Llama 3, involving multiple stages and iterative refinement. (Source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="100%"
>}}

从上图可以看出，LLaMA 3 的后训练（指令微调）流程是一个精心设计的多阶段迭代过程：

1.  **数据准备 (Data Preparation):** 收集大量人类偏好数据。这些数据通常包含一个提示（prompt）以及模型生成的多个回答，标注员会对这些回答进行排序（例如，选出最好的 "chosen" 回答和较差的 "rejected" 回答）。同时也会收集高质量的 SFT 数据（prompt-response 对）。
2.  **奖励模型训练 (Reward Modeling, RM):** 利用收集到的人类偏好数据 (prompt, chosen, rejected) 三元组训练一个或多个奖励模型。奖励模型的目标是学习预测人类对模型生成回答的偏好程度，为后续的优化提供量化信号。LLaMA 3 训练了两个独立的奖励模型，分别侧重于有用性（Helpfulness）和安全性（Safety）。
3.  **拒绝采样 (Rejection Sampling):** 使用训练好的奖励模型对模型生成的候选回答进行打分。选择得分最高的回答作为高质量样本，用于后续的微调阶段。这有助于筛选出比 SFT 数据质量更高的样本。
4.  **监督微调 (Supervised Finetuning, SFT):** 结合初始的人工标注 SFT 数据和通过拒绝采样筛选出的高质量数据，对预训练的基础模型进行微调。此阶段旨在让模型学习遵循指令的格式、风格，并初步掌握所需的知识和能力。LLaMA 3 在此阶段混合使用了多种来源的数据。
5.  **偏好优化:** 在 SFT 模型的基础上，使用偏好数据 (prompt, chosen, rejected) 通过**直接偏好优化 (Direct Preference Optimization, DPO)** 算法进一步对齐模型。DPO 直接优化模型以提高其对 "chosen" 回答的似然，同时降低对 "rejected" 回答的似然，相比基于 RL 的 PPO 方法，实现更简单且训练更稳定。LLaMA 3 对 DPO 进行了改进，例如在训练 DPO 时屏蔽了回答中的特殊格式化 token，并引入了归一化的负对数似然（NLL）损失作为正则项，以提升训练稳定性和生成质量。其损失函数形式大致可以参考 **RPO**([Pang et al., 2024](https://arxiv.org/abs/2404.19733)) 中的损失，LLaMA3 的具体实现可能略有不同：

    $$
    \begin{aligned}
    \mathcal{L}_{\mathrm{DPO}+\mathrm{NLL}} & =\mathcal{L}_{\mathrm{DPO}}\left(y^w, y^l \mid x\right)+\alpha \mathcal{L}_{\mathrm{NLL}}\left(y^w \mid x\right) \\
    & =-\log \sigma\left(\beta \log \frac{\pi_\theta(y^w \mid x)}{\pi_{\mathrm{ref}}(y^w \mid x)}-\beta \log \frac{\pi_\theta(y^l \mid x)}{\pi_{\mathrm{ref}}(y^l \mid x)}\right)-\alpha \frac{\log \pi_\theta(y^w \mid x)}{|y^w|}
    \end{aligned}
    $$
    其中：
    *   $x$ 是输入 prompt。
    *   $y^w$ 是偏好的 (winning/chosen) 回答，$y^l$ 是不被偏好的 (losing/rejected) 回答。
    *   $\pi_\theta$ 是当前正在优化的模型策略（参数为 $\theta$）。
    *   $\pi_{\mathrm{ref}}$ 是参考模型策略（通常是 SFT 后的模型或上一轮迭代的模型）。
    *   $\beta$ 是控制偏好强度差异的超参数。
    *   $\sigma$ 是 Sigmoid 函数。
    *   $\alpha$ 是平衡 DPO 损失和 NLL 正则化损失的权重。
    *   $|y^w|$ 是 winning 回答的长度，用于归一化 NLL 损失。
    该损失函数鼓励模型 $\pi_\theta$ 相对于参考模型 $\pi_{\mathrm{ref}}$ 更倾向于生成 $y^w$ 而非 $y^l$，同时通过 NLL 正则项保持生成文本的流畅性和语言质量。

6.  **迭代循环 (Iterative Loop):** 上述的 SFT 和 DPO（或 RLHF 变体）过程会重复进行多轮（LLaMA 3 进行了五轮）。每一轮都会使用上一轮优化后的模型来生成新的数据，收集新的人类反馈，训练新的奖励模型，并进行下一轮的 SFT 和 DPO 优化。这种迭代的方式使得模型能够持续学习和改进。
7.  **模型权重平均 (Model Weight Averaging):** 在某些阶段，可能会对使用不同数据子集或超参数训练得到的多个模型检查点进行权重平均，以获得更鲁棒、性能更均衡的最终模型。

### LLaMA 4

**LLaMA 4** ([Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))系列模型由 Meta AI 于 2025 年 4 月 5 日发布，标志着 LLaMA 生态系统迈入了原生多模态 AI 创新的新阶段。这一代模型首次引入了**混合专家（Mixture-of-Experts, MoE）架构**，并具备了前所未有的**超长上下文处理能力**，旨在提供更强大、更高效的开源基础模型。

**模型概览：性能、规模与部署**

LLaMA 4 首批发布了三个不同定位的模型，其中两个开放权重：

| 模型名称          | 活跃参数 | 专家数量 | 总参数 | 关键性能/定位                                                                                                | 硬件参考                                   | 上下文窗口 |
| :---------------- | :------- | :------- | :----- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------- | :--------- |
| **Llama 4 Scout** | 17B      | 16       | 109B   | 优于 Gemma 3 等同级模型；**10M Token 超长上下文**；图像理解能力强；高性价比                                        | 单张 H100 GPU (INT4 量化)                  | **10M**    |
| **Llama 4 Maverick**| 17B      | 128      | 400B   | 性能比肩甚至超越 GPT-4o/Gemini 2.0 Flash (推理/编码/多语言)；活跃参数少，计算效率高；图像推理/理解领先；LMArena ELO 1417 | 单台 H100 主机 (多卡) 或分布式部署         | **1M** |
| **Llama 4 Behemoth**| 288B     | 16       | ~2T    | **教师模型 (未发布)**；STEM 基准 (MATH, GPQA) 超越 GPT-4.5/Claude 3.7/Gemini 2.0 Pro；通过**共蒸馏**提升 Scout/Maverick | 仍在训练中，未公开发布                     | (未明确)   |

*   **性能亮点：** Maverick (17B 活跃参数) 在多个主流基准上展现出与 GPT-4o 等顶尖闭源模型竞争的实力，尤其在推理、编码和多语言任务上，而其活跃参数量显著更少，体现了出色的计算效率。Scout 则凭借其惊人的 10M Token 上下文窗口在同级别模型中脱颖而出。
*   **部署门槛：** Scout 的 INT4 量化版本可在单块 H100 上运行，降低了高性能模型的部署门槛。Maverick 虽然需要更强的算力（如单台 H100 多卡主机），但相较于其性能，仍提供了有吸引力的性价比。*(注：消费级显卡运行这些模型仍有挑战)*

**核心架构与训练创新**

LLaMA 4 相比上一代模型有以下优化：

1.  **混合专家 (MoE) 架构：**
    *   LLaMA 4 是首个采用 MoE 的 Llama 系列。MoE 允许模型在推理时仅激活总参数的一小部分（即“活跃参数”），以**更低的计算量换取更大的模型容量和更强的性能**。这对于计算成本敏感（尤其是吞吐量敏感）的推理场景非常有利。
    *   Maverick 模型采用了**交替的密集层和 MoE 层**。其 MoE 层包含 128 个路由专家和一个所有 Token 都会访问的共享专家，每个 Token 会被路由到共享专家及其中一个路由专家进行处理。

2.  **原生多模态与早期融合 (Early Fusion)：**
    *   **告别“拼接式”：** 不同于以往将视觉模块“外挂”到 LLM 上的后期融合方法，LLaMA 4 从设计之初就采用**早期融合**策略。
    *   **统一骨干：** 文本 Token 和视觉 Token（来自图像和视频帧）在模型主干网络（Backbone）的早期阶段就被无缝集成和共同处理。
    *   **深度理解：** 这使得模型能在海量的图文、视频数据上进行联合预训练，学习到更深层次、更细粒度的跨模态关联，实现更自然的交互和更强的**视觉定位 (Grounding)** 能力（将文本提示与图像区域精确对应），而不仅仅是“看图说话”。
    *   **视觉编码器：** 基于 **MetaCLIP** ([Xu et al., 2023](https://arxiv.org/abs/2309.16671)) 进行了改进，并与 Llama 模型协同训练，以更好地适应 LLM 的需求。

3.  **超长上下文 (Ultra-Long Context)：**
    *   **10M Token 上限：** Llama 4 Scout 实现了**行业领先的 1000 万 Token 上下文窗口**。
    *   **技术支撑：**
        *   **iRoPE 架构：** 结合了 **RoPE (旋转位置编码)** 和 **NoPE (无位置编码)** 的思想。通过**交错注意力层 (interleaved attention layers)** 实现，部分特定层采用 NoPE ([Kazemnejad et al., 2023](https://arxiv.org/abs/2305.19466))，依赖注意力机制隐式学习位置关系，而 RoPE 仍用于大多数其他层。("i" 同时代表 interleaved 和 infinite 的目标)。
        *   **Scalable-Softmax:** 结合推理时温度缩放 ([Nakanishi et al., 2025](https://arxiv.org/abs/2501.19399))，增强模型在未见过长度上的泛化能力。
        *   **专门训练：** 通过专门构建的长上下文数据集进行中期训练 (mid-training) 和后训练 (post-training)，Scout 在 256k 上下文长度上进行了训练，并通过 iRoPE 和 Scalable Softmax 泛化到 10M。
    *   **实用性观察：** 尽管 10M Token 非常吸引人，但在实际应用中处理如此长的上下文可能会遇到推理效率、注意力分散和带宽瓶颈等问题，其**真实场景下的效果和效率仍有待用户验证**。

{{< figure
    src="llama4_sequence_position_nll.png"
    caption="Fig. 14. Cumulative average NLL loss per sequence position for code generation, demonstrating Llama 4 Scout's strong performance over long contexts. (Source: [Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))"
    align="center"
    width="100%"
>}}

4.  **大规模高质量预训练：**
    *   **数据量级：** 训练数据超过 **30 万亿 tokens**（LLaMA 3 的两倍以上），包含文本、图像和视频。
    *   **多语言覆盖：** 覆盖 **200 种语言**，其中超 100 种语言数据量过 10 亿 token，多语言 token 总量是 LLaMA 3 的 10 倍。
    *   **训练效率：** 采用 **FP8 精度**进行训练，Behemoth 在 32K GPU 上实现了 390 TFLOPs/GPU 的高利用率。利用 **MetaP** 技术可靠设置超参数。

**革新的后训练 (Post-training) 流程**

LLaMA 4 采用了新的三阶段后训练流程，旨在平衡指令遵循、智能涌现和对话质量：

1.  **轻量级 SFT (Supervised Fine-Tuning)：** 专注于使用少量、更困难的数据集进行监督微调，教会模型基本的指令遵循和对话格式，避免过度拟合简单模式，为后续 RL 探索保留空间。相比之前版本，**大幅削减了简单 SFT 数据**（Maverick >50%, Behemoth >95%）。
2.  **在线强化学习 (Online RL)：** 这是提升模型核心智能和复杂任务能力的关键阶段。采用**持续在线 RL 策略**，模型在与环境交互中学习，通过精心选择较难的提示进行探索，并交替进行模型训练和数据过滤（保留中等到困难的交互数据），在计算和效果间取得平衡。
3.  **轻量级 DPO (Direct Preference Optimization)：** 在 RL 之后进行，用于微调模型响应的风格、安全性和修正边缘案例 (corner cases)，进行最终的“精修和打磨”，确保智能与流畅对话体验的统一。

**教师模型与共蒸馏 (Co-Distillation)**

*   强大的 **Behemoth (2T)** 虽然未发布，但其通过**新颖的共蒸馏技术**，在预训练阶段就将其知识传递给了 Scout 和 Maverick。
*   这种**共蒸馏**发生在预训练过程中，使用了动态加权软目标（教师模型的 logits）和硬目标（真实标签）的新型蒸馏损失函数，显著提升了学生模型的质量（尤其在数学、编码等领域），同时摊销了教师模型的训练成本。

**大规模 RL 基础设施**

为了训练 Behemoth 这样的超大 MoE 模型，Meta 彻底改造了 RL 基础设施，采用了**完全异步的在线 RL 训练框架**，优化了 MoE 并行策略，实现了灵活的 GPU 资源分配和近 10 倍的训练效率提升。

### 对比

| 特性               | LLaMA 1              | LLaMA 2              | Code Llama           | Llama Guard          | LLaMA 3              | LLaMA 4              |
|--------------------|----------------------|----------------------|----------------------|----------------------|----------------------|--------------------------|
| **发布时间**        | 2023/02              | 2023/07              | 2023/08              | 2023/12+             | 2024/04+             | 2025/04+             |
| **基础模型**        | -                    | -                    | LLaMA 2              | LLaMA 2 / LLaMA 3    | -                    | -                        |
| **模型规模**        | 7B - 65B             | 7B, 13B, 70B         | 7B - 70B             | 7B / 8B (+Vision)    | 1B - 405B (+Vision)  | 109B, 400B, ~2T (MoE)|
| **训练数据量**      | 1T - 1.4T tokens     | 2T+ tokens           | + 0.5T/1T Code       | ~40k 安全分类    | 15T+ tokens          | 30T+ tokens (多模态) |
| **上下文长度**      | 2k tokens            | 4k tokens            | 100k | 4k / 8k+             | 8k / 128k tokens     | 10M |
| **Tokenizer**      | SentencePiece (32k)  | SentencePiece (32k)  | SentencePiece (32k)  | 基于 LLaMA 2/3       | tiktoken (128k)      | tiktoken (256k)      |
| **位置编码**        | RoPE                 | RoPE                 | RoPE (基数调整)      | RoPE                 | RoPE                 | iRoPE   |
| **注意力** | MHA       | MHA / GQA (34B, 70B) | MHA / GQA (>13B)     | 基于 LLaMA 2/3       | GQA   | GQA                  |
| **归一化**          | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)        |
| **激活函数**        | SwiGLU               | SwiGLU               | SwiGLU               | SwiGLU               | SwiGLU               | SwiGLU                   |
| **模型类别**        | 文本模型         | 文本模型        | 代码生成    | 安全分类型         | 多模态模型 | 多模态模型 |

## 关键技术解析

以下是对 LLaMA 系列中广泛采用的关键技术的解析。

### RMS Normalization (RMSNorm)

在深度学习模型训练中，归一化层对于加速收敛、提高泛化能力和稳定训练过程至关重要。**RMSNorm (Root Mean Square Normalization)** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) 是 Layer Normalization 的一种简化变体，它仅使用输入的均方根（Root Mean Square）进行归一化，省去了均值中心化步骤，从而减少了计算量。

其数学表达式为：
$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$
其中：
*   $ x \in \mathbb{R}^d $ 是输入向量。
*   $ d $ 是向量维度。
*   $ \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon} $ 计算输入的均方根。
*   $ \epsilon $ 是一个很小的正数（如 $10^{-6}$），用于防止分母为零，增加数值稳定性。
*   $ \gamma \in \mathbb{R}^d $ 是一个可学习的缩放参数向量（gain）。RMSNorm 通常省略了 LayerNorm 中的可学习偏移参数（bias）$ \beta $。

**LLaMA 系列选择 RMSNorm 的主要原因：**

*   **计算效率高：** 相比 LayerNorm，RMSNorm 计算量更小，因为它不需要计算均值。这对于计算密集型的大语言模型训练和推理尤为重要。
*   **性能相当：** 实践证明，RMSNorm 在 Transformer 等架构中通常能达到与 LayerNorm 相当甚至更好的性能，同时保持训练稳定性。
*   **实现简单：** 其计算逻辑相对简单，易于在各种硬件上高效实现。

> 关于各种 Norm 的对比和代码实现，可参考博客：[Normalization in Deep Learning](https://syhya.github.io/posts/2025-02-01-normalization/)。

### FFN_SwiGLU

**Swish-Gated Linear Unit** ([Shazeer, 2020](#ref-16)) 是 LLaMA 中用于增强前馈网络（Feed-Forward Network, FFN）非线性表达能力的关键技术。SwiGLU 结合了 Swish 激活函数和门控机制，显著提升了模型的表现力和性能。此外，与 PaLM ([Chowdhery, 2022](#ref-6)) 中使用的$4 d$隐藏维度不同，LLaMA 采用了 $\frac{2}{3}d$ 的隐藏维度，从而在保持参数量和计算量不变的情况下，实现了更高的参数效率。

数学表达式：
$$
\operatorname{FFN}_{\mathrm{SwiGLU}}\left(x, W_1, W_3, W_2\right)=\left(\operatorname{Swish}\left(x W_1\right) \otimes x W_3\right) W_2
$$
其中：
- $ \text{Swish}(x) = x \cdot \sigma(x) $（Swish 激活函数）。
- $ \sigma(x) = \frac{1}{1 + e^{-x}} $（Sigmoid 函数）。
- $ \otimes $ 表示逐元素相乘。
- $ W_1, W_2, W_3 $ 为线性变换矩阵。

**优势**：
- **增强非线性表达**：SwiGLU 通过结合 Swish 激活函数与门控机制，能够更有效地捕捉复杂的模式和关系，提升 FFN 层的表达能力。
- **参数效率**：采用 $\frac{2}{3}d$ 的隐藏维度，在引入额外的线性变换矩阵的同时，保持了总参数量不变，实现了参数的高效利用。
- **性能提升**：在多项基准测试中，FFN_SwiGLU 显著提升了模型的性能，尤其在处理复杂任务和长文本时表现尤为出色。例如，在文本生成和理解任务中，SwiGLU 帮助模型更好地理解上下文和长距离依赖关系。

**实现细节**：
- **权重矩阵调整**：为了保持与传统 FFN 层相同的参数量和计算量，SwiGLU 通过减少隐藏层的维度（例如，将隐藏层大小从 4d 调整为 $\frac{2}{3}d$），在引入额外的线性变换矩阵的同时，确保整体模型的效率不受影响。
- **兼容性**：SwiGLU 作为 GLU 家族的一员，能够无缝集成到现有的 Transformer 架构中，替代传统的 ReLU 或 GELU 激活函数，提升模型的整体性能。

> 实现代码可以参考这个文件：[swiglu.py](https://github.com/syhya/syhya.github.io/blob/main/content/zh/posts/2025-04-06-llama/swiglu.py)。

### Grouped Query Attention (GQA)

**Grouped Query Attention (GQA)** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) 是一种对标准多头注意力（Multi-Head Attention, MHA）机制的关键优化技术，特别是在像 LLaMA 这样的大型语言模型中应用。其主要目标是在推理过程中，减少加载和存储 **KV Cache** 所需的内存带宽和容量，从而在模型性能和计算效率之间取得更好的平衡。

GQA 是 MHA 和多查询注意力（Multi-Query Attention, MQA）之间的一种折中：

*   **MHA：** 有 $H$ 个查询头（Query heads），每个头都有自己独立的 $H$ 组 K 和 V 投影。计算量和 KV Cache 大小与头数 $H$ 成正比。
*   **MQA：** 仍然有 $H$ 个查询头，但所有头共享同一组 K 和 V 投影。这极大地减少了 KV Cache 大小（减少为 MHA 的 $1/H$），但可能导致模型质量下降。
*   **GQA：** 将 $H$ 个查询头分成 $G$ 组（$1 < G < H$，且 $H$ 是 $G$ 的倍数），每组内的 $H/G$ 个查询头共享同一组 K 和 V 投影。总共有 $G$ 组 K 和 V 投影。


{{< figure
    src="attention_comparison.png"
    caption="Fig. 15. Overview of Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA). GQA groups query heads to share key/value heads. (Source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))"
    align="center"
    width="100%"
>}}

计算步骤如下：

1.  **投影：** 输入 $X$ 仍然投影得到 $Q, K, V$。$Q$ 被划分为 $H$ 个头 $Q_1, \dots, Q_H$。$K$ 和 $V$ 被划分为 $G$ 组 $K^1, \dots, K^G$ 和 $V^1, \dots, V^G$。
2.  **分组注意力：** 对于第 $g$ 组（$g=1, \dots, G$），该组对应的查询头（例如 $Q_i$ 其中 $i$ 属于第 $g$ 组）与共享的 $K^g$ 和 $V^g$ 计算注意力：
    $$
    \text{Attention}_i(Q_i, K^g, V^g) = \text{softmax}\left( \frac{Q_i (K^g)^\top}{\sqrt{d_k}} \right) V^g
    $$
    其中 $d_k$ 是每个 K 头（也是 Q 头）的维度。
3.  **拼接与输出：** 所有头的输出 $ \text{Attention}_1, \dots, \text{Attention}_H $ 拼接起来，再通过一个输出投影矩阵 $W_O$ 得到最终输出。


**优势：**

*   **平衡性能与效率：** GQA 在大幅减少 KV Cache（是 MHA 的 $G/H$）的同时，通常能保持比 MQA 更接近 MHA 的模型质量。
*   **加速推理：** 减少内存带宽需求可以显著加速大模型的推理速度，尤其是在长序列生成场景下。

> 更多关于注意力机制在 **MHA**、**MQA** 和 **GQA** 之间的详细对比及代码示例，可参考博客：[Attention Mechanisms in Transformers: Comparing MHA, MQA, and GQA](https://syhya.github.io/posts/2025-01-16-group-query-attention/#grouped-query-attention-gqa)。

### Rotary Positional Embeddings (RoPE)

**Rotary Positional Embeddings (RoPE)** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) 是一种用于将位置信息注入 Transformer 注意力机制的有效方法，特别擅长编码相对位置信息。与传统的绝对位置编码（如正弦编码或可学习编码）不同，RoPE 通过对 Query 和 Key 向量应用与位置相关的旋转操作来实现。

{{< figure
    src="rope.png"
    caption="Fig. 16. Implementation of Rotary Position Embedding(RoPE). (Source: [Su et al., 2021](https://arxiv.org/abs/2104.09864))"
    align="center"
    width="90%"
>}}

假设 $q_m$ 和 $k_n$ 分别是位置 $m$ 的 Query 向量和位置 $n$ 的 Key 向量。RoPE 将 $d$ 维的向量 $x$ （$q$ 或 $k$）视为 $d/2$ 个二维向量块 $[x^{(1)}, x^{(2)}, \dots, x^{(d/2)}]$，其中 $x^{(i)} = [x_{2i-1}, x_{2i}]$。对于位置 $m$，RoPE 定义了一个旋转矩阵 $R_m$，它由 $d/2$ 个二维旋转矩阵组成：
$$
R_m = \text{diag}(R_{m,1}, R_{m,2}, \dots, R_{m,d/2})
$$

其中每个二维旋转矩阵为：
$$
R_{m,i} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
$$

旋转频率 $ \theta_i = b^{-2(i-1)/d} $，其中 $b$ 是一个预设的基数（LLaMA 中通常为 10000）。

应用 RoPE 后，新的 Query 和 Key 向量为 $q'_m = R_m q_m$ 和 $k'_n = R_n k_n$。关键在于，它们之间的内积（点积，决定注意力分数）只依赖于相对位置 $m-n$：

$$
(q'_m)^\top k'_n = (R_m q_m)^\top (R_n k_n) = q_m^\top R_m^\top R_n k_n = q_m^\top R_{n-m} k_n
$$

这里利用了旋转矩阵的性质 $R_m^\top R_n = R_{n-m}$。

**优势：**

*   **显式相对位置编码：** 内积结果直接依赖于相对距离 $m-n$，这对于捕捉序列中元素间的相对关系非常自然。
*   **长距离衰减特性：** 随着相对距离 $|m-n|$ 的增大，旋转导致的向量间夹角变化通常会使得内积值衰减，符合直觉（距离越远，关联越弱）。
*   **良好的外推性：** 理论上，RoPE 可以较好地泛化到比训练时更长的序列长度，因为它不依赖于绝对位置的最大值。通过调整基数 $b$（如 Code Llama 和 LLaMA 4 的 iRoPE），可以进一步优化其在超长上下文下的表现。
*   **无额外参数：** RoPE 是一种固定的、基于位置的变换，不引入额外的可学习参数。
*   **兼容线性注意力：** 可以与各种线性注意力变体结合使用。

### Mixture-of-Experts (MoE)

**Mixture-of-Experts (MoE)** 是一种旨在提升模型容量（总参数量）同时控制计算成本（活跃参数量）的神经网络架构范式。它将网络中的某些层（通常是 FFN 层）替换为多个并行的专家子网络，并通过一个轻量级的门控网络（gating network）为每个输入 token 动态地选择性地激活其中少数几个（通常是 Top-K，K=1 或 2）专家进行计算。

{{< figure
    src="llama4_moe.png"
    caption="Fig. 17. The Illustration of a mixture-of-experts(MoE) in llama4. (Source: [Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))"
    align="center"
    width="100%"
>}}


假设一个 MoE 层有 $N$ 个专家 $E_1, E_2, \dots, E_N$（例如，每个专家是一个独立的 FFN 网络）和一个门控网络 $G$。对于输入 token $x$，MoE 层的计算过程如下：

1.  **门控计算：** 门控网络 $G$（通常是一个简单的线性层加 Softmax）计算每个专家被选中的概率或权重：$p = G(x) = \text{Softmax}(\text{Linear}(x))$，其中 $p \in \mathbb{R}^N$。
2.  **专家选择 (Top-K Routing)：** 根据门控输出 $p$，选择得分最高的 K 个专家。设选中的专家索引集合为 $\mathcal{T} = \text{TopKIndices}(p)$。
3.  **专家计算：** 只有被选中的 K 个专家对输入 $x$ 进行计算，得到输出 $E_i(x)$ for $i \in \mathcal{T}$。
4.  **输出组合：** 最终的输出 $y$ 是被选中专家的输出根据其门控权重（通常是重新归一化后的权重）的加权和：
    $$
    y = \sum_{i \in \mathcal{T}} \frac{p_i}{\sum_{j \in \mathcal{T}} p_j} \cdot E_i(x)
    $$
    或者在某些实现中，权重可能直接使用 $p_i$。

**优势：**

*   **参数规模与计算解耦：** MoE 允许模型拥有巨大的总参数量（通过增加专家数量 $N$），但每次前向传播的计算量仅取决于激活的 K 个专家的计算量，远低于同等总参数量的密集（Dense）模型。这使得在有限的计算预算下可以训练出容量更大、可能性能更强的模型。
*   **专家特化：** 理论上，不同的专家可以学习处理不同类型的数据、模式或任务的特定方面，实现知识的模块化存储和处理，从而提升模型的整体能力和泛化性。

**挑战：**

*   **负载均衡 (Load Balancing)：** 需要确保所有专家被大致均匀地利用，避免某些专家过载而其他专家空闲。通常需要引入辅助损失函数（如 Load Balancing Loss）来鼓励均匀路由。
*   **通信开销：** 在分布式训练和推理中，需要在不同设备（GPU）之间进行高效的通信（如 All-to-All）来将 token 路由到存储相应专家的设备上，并收集结果。这增加了实现的复杂性和通信成本。
*   **训练稳定性：** MoE 模型的训练可能比密集模型更不稳定，需要仔细调整超参数和训练策略。
*   **内存占用：** 虽然计算量是稀疏的，但模型总参数量巨大，需要大量内存来存储所有专家权重。

> 关于 MoE 更详细的说明，可参考博客：[Parallelism and Memory Optimization Techniques for Training Large Models](https://syhya.github.io/posts/2025-03-01-train-llm/#mixture-of-experts-model)的混合专家模型部分。


## 参考文献
[1] Touvron, Hugo, et al. ["LLaMA: Open and Efficient Foundation Language Models."](https://arxiv.org/abs/2302.13971) arXiv preprint arXiv:2302.13971 (2023).

[2] Facebook Research. ["xformers."](https://github.com/facebookresearch/xformers) GitHub repository (Accessed 2024).

[3] Taori, Rohan, et al. ["Alpaca: A Strong, Replicable Instruction-Following Model."](https://crfm.stanford.edu/2023/03/13/alpaca.html) Stanford CRFM Blog (2023).

[4] Touvron, Hugo, et al. ["Llama 2: Open Foundation and Fine-Tuned Chat Models."](https://arxiv.org/abs/2307.09288) arXiv preprint arXiv:2307.09288 (2023).

[5] Bai, Yuntao, et al. ["Constitutional AI: Harmlessness from AI Feedback."](https://arxiv.org/abs/2212.08073) arXiv preprint arXiv:2212.08073 (2022).

[6] Roziere, Baptiste, et al. ["Code Llama: Open Foundation Models for Code."](https://arxiv.org/abs/2308.12950) arXiv preprint arXiv:2308.12950 (2023).

[7] Inan, Hakan, et al. ["Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations."](https://arxiv.org/abs/2312.06674) arXiv preprint arXiv:2312.06674 (2023).

[8] Chi, Jianfeng, et al. ["Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations."](https://arxiv.org/abs/2411.10414) arXiv preprint arXiv:2411.10414 (2024).

[9] Vidgen, Bertie, et al. ["Introducing v0.5 of the AI Safety Benchmark from MLCommons."](https://arxiv.org/abs/2404.12241) arXiv preprint arXiv:2404.12241 (2024).

[10] Meta Llama. ["Llama-Guard-3-8B."](https://huggingface.co/meta-llama/Llama-Guard-3-8B) Hugging Face Model (Accessed 2024).

[11] Grattafiori, Aaron, et al. ["The Llama 3 Herd of Models."](https://arxiv.org/abs/2407.21783) arXiv preprint arXiv:2407.21783 (2024).

[12] Meta AI. ["Introducing Llama 3.1: Our most capable models to date."](https://ai.meta.com/blog/meta-llama-3-1/) Meta AI Blog (2024).

[13] Umar Jamil. ["pytorch-llama Slides."](https://github.com/hkproj/pytorch-llama/blob/main/Slides.pdf) GitHub file (Accessed 2024).

[14] Pang, Richard Yuanzhe, et al. ["Iterative reasoning preference optimization."](https://arxiv.org/abs/2404.19733) Advances in Neural Information Processing Systems 37 (2024): 116617-116637.

[15] Meta AI. ["The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation"](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) Meta AI Blog (2025).

[16] Xu, Hu, et al. ["Demystifying clip data."](https://arxiv.org/abs/2309.16671) arXiv preprint arXiv:2309.16671 (2023).

[17] Kazemnejad, Amirhossein, et al. ["The impact of positional encoding on length generalization in transformers."](https://arxiv.org/abs/2305.19466) Advances in Neural Information Processing Systems 36 (2023): 24892-24928.

[18] Nakanishi, Ken M. ["Scalable-Softmax Is Superior for Attention."](https://arxiv.org/abs/2501.19399) arXiv preprint arXiv:2501.19399 (2025).

[19] Zhang, Biao, and Rico Sennrich. ["Root mean square layer normalization."](https://arxiv.org/abs/1910.07467) Advances in Neural Information Processing Systems 32 (2019).

[20] Shazeer, Noam. ["Glu variants improve transformer."]((https://arxiv.org/abs/2002.05202v1)) arXiv preprint arXiv:2002.05202 (2020).

[21] Ainslie, Joshua, et al. ["Gqa: Training generalized multi-query transformer models from multi-head checkpoints."](https://arxiv.org/abs/2305.13245) arXiv preprint arXiv:2305.13245 (2023).

[22] Su, Jianlin, et al. ["Roformer: Enhanced transformer with rotary position embedding."]((https://arxiv.org/abs/2104.09864)) Neurocomputing 568 (2024): 127063.

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Apr 2025). LLaMA 系列模型. https://syhya.github.io/zh/posts/2025-04-06-llama

Or

```bibtex
@article{syhya2025llama,
  title   = "LLaMA 系列模型",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Apr",
  url     = "https://syhya.github.io/zh/posts/2025-04-06-llama"
}

