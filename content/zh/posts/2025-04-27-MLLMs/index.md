---
title: "多模态大语言模型（长期更新）"
date: 2025-04-27T12:00:00+08:00
author: "Yue Shui"
tags: ["多模态", "视觉语言", "ViT", "CLIP", "BLIP", "LLaVA", "Qwen-VL", "Kimi-VL", "o3", "o4-mini", "MCoT", "大语言模型", "人工智能", "深度学习", "NLP", "CV"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

> **注意**: 本文**正在更新中**，内容只是**草稿版本**，并不完善，后续会有变动。请随时关注最新版本。

人类通过多种感官（视觉、听觉、触觉等）与世界互动，每种感官通道在表征和交流特定概念时都具有独特的优势。这种多模态交互促进了我们对世界的深刻理解。人工智能领域的核心目标之一便是开发能够有效遵循多模态指令（如视觉和语言）的通用助手，使其能够像人类一样完成现实世界的各种任务。近年来，随着 GPT-4o ([OpenAI, 2024](https://openai.com/index/hello-gpt-4o/))、Gemini 2.5 Pro ([DeepMind, 2025](https://deepmind.google/technologies/gemini/pro/)) 和 o3/o4-mini ([OpenAI, 2025](https://openai.com/index/introducing-openai-o3-and-o4-mini/)) 等模型的发布，**多模态大语言模型（Multimodal Large Language Models, MLLMs）** 取得了显著进展，它们不仅能理解图像、视频、音频等多种模态信息，还能进行复杂的推理和生成。

## 符号表

下面列举了文章可能使用的数学公式符号，以帮助你更轻松地阅读。

| 符号 | 含义 |
| :--- | :--- |
| \( I \) 或 \( \mathbf{X}_v \) | 图像输入 |
| \( T \) 或 \( \mathbf{X}_c, \mathbf{X}_q, \mathbf{X}_a \) | 文本输入（如标题、问题、答案） |
| \( V \) 或 \( \mathbf{Z}_v \) | 图像特征或嵌入 |
| \( L \) 或 \( \mathbf{H}_v, \mathbf{H}_q, \mathbf{H}_a \) | 文本特征或嵌入 |
| \( E_{img}, E_{text} \) | 图像编码器, 文本编码器 |
| \( \pi_\theta, \pi_{\text{ref}} \) | 策略模型及其参考模型 |
| \( \mathcal{L} \) | 损失函数 (e.g., \( \mathcal{L}_{ITC}, \mathcal{L}_{ITM}, \mathcal{L}_{LM}, \mathcal{L}_{SFT} \)) |
| \( \theta, \phi \) | 模型参数 |
| \( N \) | 批处理大小 (Batch size) |
| \( \mathbf{W}_i, \mathbf{W}_t, \mathbf{W} \) | 投影矩阵 |
| \( \tau \) | 温度参数 (Temperature parameter) |
| \( \text{sim}(u, v) \) | 向量 \( u \) 和 \( v \) 之间的相似度 (通常是余弦相似度) |
| \( \mathbb{E} \) | 期望 |
| \( \log p(\cdot) \) | 对数似然 |
| \( \mathbb{D}_{KL} \) | KL 散度 |
| \( \alpha, \beta, \lambda \) | 超参数或权重系数 |
| \( \sigma(\cdot) \) | Sigmoid 函数 |
| \( \mathcal{D} \) | 数据集或数据分布 |

## 多模态基础知识

在深入探讨具体技术之前，我们先来了解一些多模态 AI 的基础概念。

### 什么是多模态？

**多模态 (Multimodality)** 指的是使用多种不同类型的数据或信息通道（模态）来表示和处理信息。人类天生就是多模态的生物，我们通过视觉、听觉、触觉、嗅觉、味觉以及语言来感知和理解世界。在人工智能领域，多模态学习旨在构建能够处理和关联来自不同模态（如文本、图像、视频、音频、表格数据、3D 数据等）信息的模型。

{{< figure
    src="multimodality_data.png"
    caption="Fig. 1. Multimodality Data. (Image source: [GPT-4o Image Generation](https://chatgpt.com/s/m_680de852ca60819196a2c729b2603f33))"
    align="center"
    width="60%"
>}}

**常见模态：**

*   **文本 (Text):** 自然语言文字，是信息传递和知识表达的主要方式。
*   **图像 (Image):** 静态视觉信息，包含丰富的场景、物体和纹理细节。
*   **视频 (Video):** 动态视觉信息，由连续的图像帧组成，通常伴随音频。视频不仅包含空间信息，还包含时间信息。
*   **音频 (Audio):** 声音信息，包括语音、音乐和环境声音。
*   **其他:** 表格数据、3D 点云、传感器数据（如雷达、激光雷达）、生物信号（如 EEG、ECG）等。

### 为什么需要多模态 AI？

1.  **更全面的世界理解:** 现实世界是多模态的。单一模态往往只能提供片面的信息。例如，仅凭文字描述可能难以完全理解一个复杂的场景，而结合图像或视频则能提供更直观、丰富的信息。多模态模型能够整合来自不同来源的信息，形成更全面、准确的理解。
2.  **增强的任务性能:** 在许多任务中，结合多种模态的信息可以显著提升性能。例如，在视觉问答（VQA）中，模型需要同时理解图像内容和文本问题才能给出正确答案。在视频描述生成中，结合视觉帧和音频信息可以生成更生动、准确的描述。
3.  **更自然的交互方式:** 多模态 AI 使得人机交互更加自然和灵活。用户可以通过语音、文字、图像等多种方式与 AI 系统交互，AI 系统也能以多种模态（如生成带有图片的文本回复，或生成语音回答）进行响应。
4.  **解锁新应用场景:** 多模态能力催生了许多新的应用，如自动驾驶（融合摄像头、雷达、激光雷达数据）、医疗诊断（结合医学影像和病历文本）、内容创作（文生图、文生视频）、虚拟助手、机器人交互等。
5.  **促进可访问性:** 多模态技术可以帮助有感官障碍的人士。例如，图像描述可以帮助视障人士理解图片内容，语音识别和合成可以帮助听障或语障人士交流。

### 多模态 AI 的核心挑战

尽管多模态 AI 前景广阔，但也面临诸多挑战：

1.  **模态对齐 (Modality Alignment):** 不同模态的数据具有不同的结构和统计特性。如何学习不同模态数据之间的对应关系，将它们映射到一个共享的表示空间，是多模态学习的核心挑战。例如，如何将图像中的“狗”区域与文本中的“dog”一词对齐。
2.  **信息融合 (Information Fusion):** 如何有效地融合来自不同模态的信息是一个关键问题。简单的拼接可能效果不佳，需要设计更复杂的融合机制（如注意力机制、门控机制等）来捕捉模态间的交互和互补信息。
3.  **表示学习 (Representation Learning):** 如何学习到既能捕捉各模态内部信息，又能反映模态间关联的联合表示（Joint Representation）或协调表示（Coordinated Representation）至关重要。
4.  **数据稀疏与噪声:** 高质量、大规模、标注良好的多模态数据集相对稀缺，尤其是在特定领域。许多现有数据集（特别是从网络爬取的）包含大量噪声（如图像与文本描述不匹配），如何有效利用这些含噪数据是一个挑战。
5.  **计算成本:** 处理和融合多种模态数据通常需要更大的模型和更多的计算资源，导致训练和推理成本高昂。
6.  **评估困难:** 评估多模态模型性能的指标和基准仍在发展中，如何全面、公正地评估模型的综合能力是一个挑战。

### 常见多模态任务

*   **视觉问答 (Visual Question Answering, VQA):** 给定一张图片和一个关于图片的问题，模型需要回答该问题。
*   **图像/视频描述生成 (Image/Video Captioning):** 为给定的图像或视频生成一段自然语言描述。
*   **文本-图像/视频检索 (Text-Image/Video Retrieval):** 根据文本描述检索相关的图像或视频，反之亦然。
*   **多模态情感分析 (Multimodal Sentiment Analysis):** 结合文本、语音语调、面部表情等信息判断情感倾向。
*   **多模态机器翻译 (Multimodal Machine Translation):** 在翻译文本时利用图像信息来消除歧义或提供上下文。
*   **文本到图像/视频/音频生成 (Text-to-Image/Video/Audio Generation):** 根据文本描述生成相应的视觉或听觉内容。
*   **视觉推理 (Visual Reasoning):** 基于图像内容进行逻辑推理，如判断物体关系、预测事件发展等。
*   **多模态对话 (Multimodal Dialogue):** 在对话中同时理解和生成多种模态的信息。
*   **视觉语言导航 (Vision-Language Navigation, VLN):** 根据自然语言指令在视觉环境中导航。

## 核心技术演进

多模态 AI 的发展离不开一系列核心技术的推动。本节将按照技术演进的脉络，介绍其中的关键模型和方法。

### Vision Transformer (ViT)

**Vision Transformer (ViT)** ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) 将 Transformer 架构成功应用于计算机视觉领域，成为当前众多先进 MLLMs 的首选视觉编码器。

{{< figure
    src="vit_overview.png"
    caption="Fig. 2. ViT model overview. (Image source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))"
    align="center"
    width="100%"
>}}

**核心思想:** ViT 将图像视为一系列 **图像块 (Patches)** 的序列，然后利用 Transformer 的自注意力机制来处理这些图像块，从而捕捉全局依赖关系。

**工作流程:**

1.  **图像分块 (Patch Embedding):** 将输入图像 \( I \in \mathbb{R}^{H \times W \times C} \) 分割成 \( N \) 个固定大小的非重叠图像块 \( x_p \in \mathbb{R}^{P^2 \times C} \)，其中 \( (H, W) \) 是图像分辨率，\( C \) 是通道数，\( P \) 是每个图像块的大小，\( N = HW/P^2 \) 是图像块的数量。
2.  **线性投射:** 将每个图像块 \( x_p \) 展平成一维向量，并通过一个可学习的线性投射矩阵 \( E \) 将其映射到 \( D \) 维的嵌入空间，得到图像块嵌入 \( z_p = x_p E \)。
3.  **位置编码 (Position Embedding):** 为了保留图像块的空间位置信息，ViT 在图像块嵌入的基础上加入了可学习的 **位置编码 (Position Embeddings)** \( E_{pos} \)。
    \[ z_0 = [x_{class}; z_p^1; z_p^2; \dots; z_p^N] + E_{pos}, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N+1) \times D} \]
    通常还会添加一个可学习的 `[class]` 标记嵌入 \( x_{class} \)，其在 Transformer 输出端的对应向量用于图像分类任务。
4.  **Transformer 编码器:** 将添加了位置编码的图像块嵌入序列输入到标准的 Transformer 编码器中。编码器由多层 **多头自注意力 (Multi-Head Self-Attention, MSA)** 和 **前馈网络 (Feed Forward Network, FFN)** 组成。
    *   **MSA:** 捕捉图像块之间的全局依赖关系。对于输入序列 \( Z_{l-1} \)，自注意力计算如下：
        \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
        其中 \( Q = Z_{l-1}W_Q, K = Z_{l-1}W_K, V = Z_{l-1}W_V \) 是查询、键、值矩阵，\( d_k \) 是键向量的维度。多头注意力将 \( Q, K, V \) 拆分成多个头并行计算注意力，然后拼接结果。
    *   **FFN:** 通常由两个线性层和一个非线性激活函数（如 GELU）组成。
    每一层的计算可以表示为：
    \[ Z'_l = \text{MSA}(\text{LN}(Z_{l-1})) + Z_{l-1} \]
    \[ Z_l = \text{FFN}(\text{LN}(Z'_l)) + Z'_l \]
    其中 LN 表示层归一化 (Layer Normalization)。
5.  **输出:** Transformer 编码器的输出 \( Z_L \) 即为图像的特征表示。

#### ViT 的优势

{{< figure
    src="vit_bit_hybrid_compare.png"
    caption="Fig. 3. Performance versus pre-training compute for different architectures: Vision Transformers, ResNets, and hybrids. Vision Transformers generally outperform ResNets with the same computational budget. Hybrids improve upon pure Transformers for smaller model sizes, but the gap vanishes for larger models. (Image source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))"
    align="center"
    width="100%"
>}}

ViT 相比于传统的卷积神经网络 (CNN) 具有以下优势：

1.  **全局依赖建模** :自注意力直接连接任意两 patch，可显式捕捉长距离空间关系，比传统 CNN 更擅长整合整幅图像的语义信息。
2.  **大规模预训练迁移能力强** : 在诸如 JFT-300M、ImageNet-22K 等超大数据集上预训练后，可轻松迁移到分类、检测、分割等 20 多种下游任务，性能随模型/数据规模几乎线性提升。
3.  **架构简洁、易于扩展与并行**: 纯 Transformer 堆叠便于按深度、宽度和输入分辨率三维扩展；计算由矩阵乘与 Softmax 组成，天然适配 GPU/TPU 的大批量并行和混合精度训练。

#### ViT 的进阶技术

随着研究的深入，ViT 本身也在不断进化，以适应 MLLMs 的需求：

1.  **原生动态分辨率 (Native Dynamic Resolution):** 传统 ViT 通常需要固定输入分辨率。Qwen2-VL ([Wang et al., 2024f](https://arxiv.org/abs/2409.12191)) 和 Kimi-VL ([Kimi Team, 2025](https://arxiv.org/abs/2504.16790)) 等模型引入了动态分辨率处理能力。它们通常去除 ViT 中的绝对位置编码，转而使用 **2D 旋转位置编码 (2D Rotary Position Embedding, 2D-RoPE)** ([Su et al., 2024](https://arxiv.org/abs/2104.09864); [Su, 2021](https://spaces.ac.cn/archives/8397)) 来编码二维空间信息。这使得模型能够处理任意分辨率和长宽比的图像，并将其转换为变长的视觉 token 序列，更好地保留细节信息。Kimi-VL 的 MoonViT 还借鉴了 NaViT ([Dehghani et al., 2023](https://arxiv.org/abs/2307.06304)) 的 **图像打包 (Patch n' Pack)** 技术，将不同分辨率的图像块序列打包输入 Transformer，提高了训练效率。
2.  **窗口注意力 (Window Attention):** 为了降低处理高分辨率图像时自注意力机制带来的二次方计算复杂度，Qwen2.5-VL ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)) 在其 ViT 的大部分层中采用了 **窗口注意力**。注意力计算被限制在局部窗口内，使得计算复杂度与图像块数量成线性关系，显著提升了效率，同时通过少数几层全注意力层来保持全局信息的交互。
3.  **架构对齐 LLM:** Qwen2.5-VL 和 Kimi-VL 等模型还对其 ViT 架构进行了微调，使其更接近 LLM 的设计，例如使用 RMSNorm ([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)) 进行归一化，使用 SwiGLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)) 作为激活函数，以提升计算效率和模态间的兼容性。

### CLIP

**CLIP (Contrastive Language-Image Pre-training)** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) 是多模态领域具有里程碑意义的工作，它提出了一种简单而高效的方法来学习图像和文本之间的关联，为后续许多 MLLMs 奠定了基础。

**核心思想:** CLIP 的目标是学习一个 **多模态嵌入空间 (Multimodal Embedding Space)**，使得在该空间中，匹配的图像和文本对具有高相似度，而不匹配的对具有低相似度。它通过 **对比学习 (Contrastive Learning)** 的方式，利用自然语言监督来实现这一目标。

**架构:** CLIP 包含两个主要部分：

1.  **图像编码器 (Image Encoder):** 可以是 ResNet 或 ViT，负责将输入图像 \( I \) 编码为图像特征 \( V \)。
2.  **文本编码器 (Text Encoder):** 通常是 Transformer，负责将输入文本 \( T \) 编码为文本特征 \( L \)。
3.  **线性投射层:** 分别将图像特征 \( V \) 和文本特征 \( L \) 投射到共享的多模态嵌入空间，得到 \( I_e = V W_i \) 和 \( T_e = L W_t \)，其中 \( W_i \) 和 \( W_t \) 是可学习的投射矩阵。

{{< figure
    src="https://cdn.mathpix.com/cropped/2025_04_14_455634672e9a2826be22g-02.jpg?height=629&width=1709&top_left_y=214&top_left_x=181"
    caption="Fig. 4. CLIP Architecture Overview. CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset's classes. (Image source: [Radford et al., 2021](https://arxiv.org/abs/2103.00020))"
    align="center"
    width="100%"
>}}

**训练数据 (WIT):** CLIP 的成功很大程度上归功于其大规模的预训练数据集 **WIT (WebImageText)**。研究团队从互联网上收集了 4 亿个 (图像, 文本) 对。他们通过搜索约 50 万个查询词（源自维基百科词汇、高频二元组、维基百科文章标题和 WordNet 同义词集）来构建数据集，并对每个查询词限制最多 2 万个样本以平衡数据分布。这种利用网络原生图文对的方式被称为 **自然语言监督**，它避免了昂贵的人工标注，使得数据规模可以轻松扩展。

**对比损失 (Contrastive Loss):** CLIP 的核心是对比学习目标。给定一个包含 \( N \) 个 (图像, 文本) 对的批次 \( \{(I_1, T_1), \dots, (I_N, T_N)\} \)，模型的目标是预测 \( N \times N \) 个可能的配对中哪些是真实的配对。

1.  计算所有图像嵌入 \( \{I_{e,1}, \dots, I_{e,N}\} \) 和文本嵌入 \( \{T_{e,1}, \dots, T_{e,N}\} \)。通常会进行 **L2 归一化** 把每个图像或文本嵌入除以它自己的 L2 范数（Euclidean norm）。
2.  计算所有 \( N^2 \) 对 \( (I_{e,i}, T_{e,j}) \) 之间的 **余弦相似度 (Cosine Similarity)**。
    \[ \text{logits}_{i,j} = \frac{I_{e,i} \cdot T_{e,j}}{\|I_{e,i}\| \|T_{e,j}\|} \cdot \exp(\tau) \]
    其中 \( \tau \) 是一个可学习的 **温度参数**，用于缩放 logits 的范围。
3.  计算 **对称交叉熵损失 (Symmetric Cross-Entropy Loss)**。将问题视为两个分类任务：
    *   对于每个图像 \( I_i \)，在 \( N \) 个文本中找到匹配的文本 \( T_i \)。损失为 \( \mathcal{L}_{\text{image}} \)。
    *   对于每个文本 \( T_j \)，在 \( N \) 个图像中找到匹配的图像 \( I_j \)。损失为 \( \mathcal{L}_{\text{text}} \)。
    总损失为：
    \[ \mathcal{L}_{CLIP} = \frac{1}{2} (\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}) \]
    其中，
    \[ \mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(I_{e,i}, T_{e,i}) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_{e,i}, T_{e,j}) / \tau)} \]
    \[ \mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(\text{sim}(I_{e,j}, T_{e,j}) / \tau)}{\sum_{i=1}^N \exp(\text{sim}(I_{e,i}, T_{e,j}) / \tau)} \]
    这种损失函数鼓励正样本对（匹配的图文）的相似度高于负样本对（不匹配的图文）。

{{< collapse summary="**CLIP Core Pseudocode**" openByDefault=false >}}

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
# l2_normalize projects the embeddings onto the unit hypersphere
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
# The temperature parameter t scales the logits
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
# labels are the indices [0, 1, ..., n-1] indicating the correct pairings
labels = np.arange(n)
# Calculate cross-entropy loss for image-to-text classification
loss_i = cross_entropy_loss(logits, labels, axis=0)
# Calculate cross-entropy loss for text-to-image classification
loss_t = cross_entropy_loss(logits, labels, axis=1)
# Final loss is the average of the two losses
loss = (loss_i + loss_t)/2
```

{{< /collapse >}}

**零样本迁移 (Zero-Shot Transfer):** CLIP 强大的能力在于其零样本迁移性能。对于一个新的图像分类任务，无需任何微调，CLIP 可以通过以下方式进行预测：

1.  获取任务的所有类别名称（例如，“猫”，“狗”）。
2.  使用 **提示工程 (Prompt Engineering)** 将类别名称构造成句子，如 "A photo of a {label}."。这有助于弥合预训练数据（通常是句子）和下游任务（通常是单词标签）之间的分布差距。CLIP 论文发现使用提示模板和集成多个提示（Ensembling）可以显著提高性能（在 ImageNet 上提升近 5%）。
3.  使用 CLIP 的文本编码器计算每个构造句子的文本嵌入，这些嵌入构成了零样本分类器的 **权重向量**。
4.  对于一张新的待分类图像，使用 CLIP 的图像编码器计算其图像嵌入。
5.  计算该图像嵌入与所有类别文本嵌入之间的余弦相似度。
6.  将相似度最高的类别作为预测结果。

{{< figure
    src="clip_prompt_engineering.png"
    caption="Fig. 5. Prompt engineering and ensembling improve zero-shot performance. Compared to the baseline of using contextless class names, prompt engineering and ensembling boost zero-shot classification performance by almost 5 points on average across 36 datasets. This improvement is similar to the gain from using 4 times more compute with the baseline zero-shot method but is 'free' when amortized over many predictions. (Image source: [Radford et al., 2021](https://arxiv.org/abs/2103.00020))"
    align="center"
    width="70%"
>}}

**CLIP 的影响:** CLIP 证明了通过大规模自然语言监督和对比学习可以学到强大的、可迁移的视觉表示。其学习到的多模态嵌入空间和强大的图像编码器被广泛应用于后续的 MLLMs（如 Flamingo, BLIP-2, LLaVA）以及文生图模型（如 DALL-E 2, Stable Diffusion）中。

CLIP 主要关注学习对齐的表示，但在生成任务上能力有限。后续工作开始探索能够同时进行理解和生成的统一模型架构。

### BLIP

**BLIP (Bootstrapping Language-Image Pre-training)** ([Li et al., 2022a](https://arxiv.org/abs/2201.12086)) 旨在解决现有 Vision-Language Pre-training (VLP) 方法在模型和数据方面的局限性：模型通常要么擅长理解，要么擅长生成；数据则依赖于大规模但充满噪声的网络图文对。

#### MED (Multimodal Encoder-Decoder)

BLIP 提出了 **MED (Multimodal Encoder-Decoder)** 架构，旨在统一理解和生成任务。它结合了 CLIP 的对比学习和自回归生成的优点，能够处理多种模态数据。

*   **图像编码器:** 采用 ViT。
*   **文本编码器/解码器:** 基于 BERT 架构，但进行了修改以适应多模态任务和不同功能模式。
    *   **单模态编码器:** 标准的 ViT 和 BERT，分别处理图像和文本。
    *   **图像接地文本编码器:** 在文本编码器的每个 Transformer 块的自注意力 (SA) 层和前馈网络 (FFN) 层之间插入 **交叉注意力 (Cross-Attention, CA)** 层，用于注入视觉信息。文本输入前会添加 `[Encode]` 标记，其输出嵌入作为图文对的多模态表示。
    *   **图像接地文本解码器:** 将编码器中的双向 SA 层替换为 **因果自注意力 (Causal Self-Attention)** 层，以实现自回归生成。共享编码器的 CA 层和 FFN 层。使用 `[Decode]` 标记作为序列开始符。

{{< figure
    src="blip_model_architecture.png"
    caption="Fig. 6. BLIP Pre-training Model Architecture and Objectives (same parameters have the same color). We propose multimodal mixture of encoder-decoder (MED), a unified vision-language model which can operate in one of the three functionalities. (Image source: [Li et al., 2022a](https://arxiv.org/abs/2201.12086))"
    align="center"
    width="100%"
>}}

**预训练目标:** BLIP 联合优化三个目标：

1.  **图文对比损失 (Image-Text Contrastive Loss, ITC):** 类似于 CLIP，使用单模态编码器对齐图像和文本的特征空间。BLIP 采用了**ALBEF**([Li et al., 2021](https://arxiv.org/abs/2107.07651))提出的动量编码器 (Momentum Encoder) 和软标签策略来改进对比学习。
2.  **图文匹配损失 (Image-Text Matching Loss, ITM):** 使用图像接地文本编码器学习细粒度的图文对齐。这是一个二分类任务，预测图文对是匹配还是不匹配。采用难负例挖掘策略。
3.  **语言建模损失 (Language Modeling Loss, LM):** 使用图像接地文本解码器，根据图像生成文本描述。采用标准的交叉熵损失（带标签平滑）。

**参数共享:** 为了效率和多任务学习的好处，文本编码器和解码器共享除 SA 层外的所有参数（嵌入层、CA 层、FFN 层）。

#### CapFilt (Captioning and Filtering)

提出了 CapFilt (Captioning and Filtering) 一种创新的数据集引导方法，用于从未标注的网络图像中生成高质量的合成标题，并过滤掉噪声数据（包括原始网络文本和合成文本）。

1.  **初始化:** 使用预训练好的 MED 模型初始化两个模块：Captioner（图像接地文本解码器）和 Filter（图像接地文本编码器）。
2.  **微调:** 在高质量的人工标注数据集（如 COCO）上分别微调 Captioner (使用 LM 损失) 和 Filter (使用 ITC 和 ITM 损失)。这是一个轻量级过程。
3.  **生成与过滤:**
    *   Captioner 为网络图像 \( I_w \) 生成合成标题 \( T_s \)。
    *   Filter 判断原始网络文本 \( T_w \) 和合成文本 \( T_s \) 是否与图像 \( I_w \) 匹配。预测为不匹配的文本被视为噪声并移除。
4.  **引导数据集:** 将过滤后的高质量图文对（来自原始网络数据和合成数据）与人工标注数据结合，形成新的引导数据集。
5.  **重新预训练:** 使用引导数据集从头预训练一个新的 BLIP 模型。

{{< figure
    src="blip_learning_framework.png"
    caption="Fig. 7. BLIP Learning Framework. We introduce a captioner to produce synthetic captions for web images, and a filter to remove noisy image-text pairs. (Image source: [Li et al., 2022a](https://arxiv.org/abs/2201.12086))"
    align="center"
    width="100%"
>}}

**效果:** CapFilt 显著提升了模型在各项下游任务（如检索、描述生成、VQA）上的性能，证明了通过引导方式改善噪声数据质量的有效性。BLIP 也展示了统一模型在理解和生成任务上的灵活性。

### BLIP-2

**BLIP-2** ([Li et al., 2023c](https://arxiv.org/abs/2301.12597)) 针对 VLP 训练成本日益高昂的问题，提出了一种更 **高效** 的预训练策略，其核心思想是 **利用现成的、冻结的 (Frozen)** 预训练图像编码器和 **冻结的大语言模型 (LLMs)**。

**核心贡献:**

1.  **利用冻结模型:** 无需端到端训练整个大型模型，显著降低了计算成本，并利用了强大的预训练单模态模型的能力。
2.  **Q-Former (Querying Transformer):** 提出了一种轻量级的 Transformer 结构作为 **可训练的桥梁**，连接冻结的图像编码器和冻结的 LLM。
3.  **两阶段预训练:** 设计了两阶段策略来有效弥合模态鸿沟：
    *   **阶段一：视觉-语言表示学习 (Vision-Language Representation Learning):** 从冻结的图像编码器引导学习。
    *   **阶段二：视觉到语言生成学习 (Vision-to-Language Generative Learning):** 从冻结的 LLM 引导学习。

**架构 (Q-Former):**

*   Q-Former 是一个轻量级 Transformer，包含 188M 参数。
*   它使用一组 **可学习的查询向量 (Learnable Query Embeddings)**（例如 32 个 768 维向量）作为输入。
*   这些查询向量通过 **自注意力层** 相互交互。
*   通过 **交叉注意力层** 与冻结的图像编码器输出的图像特征进行交互，提取视觉信息。
*   查询向量的输出 \( Z \) (例如 \( 32 \times 768 \) 维) 维度远小于原始图像特征，充当了 **信息瓶颈 (Information Bottleneck)**，迫使 Q-Former 提取对语言模型最有用的视觉信息。
*   Q-Former 内部包含图像 Transformer 和文本 Transformer 两个共享自注意力层的子模块。

{{< figure
    src="blip2_stage1.png"
    caption="Fig. 8. (Left) Model architecture of Q-Former and BLIP-2's first-stage vision-language representation learning objectives. (Right) The self-attention masking strategy for each objective to control query-text interaction. (Image source: [Li et al., 2023c](https://arxiv.org/abs/2301.12597))"
    align="center"
    width="100%"
>}}

**两阶段预训练:**

1.  **阶段一 (表示学习):**
    *   将 Q-Former 连接到 **冻结的图像编码器** (如 CLIP ViT-L/14, EVA-CLIP ViT-g/14)。
    *   使用图文对进行预训练，目标是让 Q-Former 的查询向量学会提取与文本最相关的视觉表示。
    *   联合优化三个与 BLIP 类似的目标 (共享输入格式和模型参数，但冻结图像编码器，只训练 Q-Former)：
        *   **ITC (图文对比):** 对齐 Q-Former 输出的查询表示 \( Z \) 和文本表示 \( t \)。使用 In-batch Negatives。
        *   **ITM (图文匹配):** 预测图文对是否匹配。使用 Q-Former 输出的多模态查询表示 \( Z \) 进行分类。
        *   **ITG (图像接地的文本生成):** 训练 Q-Former 生成文本。查询向量需要捕获所有生成文本所需的信息，并通过自注意力层传递给文本 token。
    *   通过不同的自注意力掩码控制查询-文本交互来实现不同目标。

2.  **阶段二 (生成学习):**
    *   将 **第一阶段预训练好的 Q-Former** (及其连接的冻结图像编码器) 连接到 **冻结的 LLM** (如 OPT 系列, FlanT5 系列)。
    *   使用一个 **全连接层 (FC Layer)** 将 Q-Former 的输出查询嵌入 \( Z \) 线性投射到与 LLM 文本嵌入相同的维度。
    *   将投射后的查询嵌入作为 **软视觉提示 (Soft Visual Prompts)**，添加到 LLM 输入文本嵌入的前面。
    *   **训练目标:** 训练 Q-Former (FC 层也训练)，使其输出的视觉表示能够被冻结的 LLM 理解并用于生成文本。
        *   对于 **Decoder-only LLM (如 OPT):** 使用标准的语言建模损失，即根据视觉提示生成后续文本。
        *   对于 **Encoder-Decoder LLM (如 FlanT5):** 使用前缀语言建模损失 (Prefix Language Modeling)，将文本分成前缀和后缀，视觉提示和前缀输入 Encoder，Decoder 生成后缀。

{{< figure
    src="blip2_stage2.png"
    caption="Fig. 9. BLIP-2's second-stage vision-to-language generative pre-training, which bootstraps from frozen large language models (LLMs). (Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5). (Image source: [Li et al., 2023c](https://arxiv.org/abs/2301.12597))"
    align="center"
    width="90%"
>}}

**效果与优势:**

*   **高效:** 由于只训练轻量级的 Q-Former，预训练成本远低于端到端训练的大型模型。
*   **高性能:** 在 VQA、Captioning、Retrieval 等任务上达到 SOTA 水平，甚至超越了参数量远大于它的模型（如 Flamingo）。
*   **通用性:** 可以方便地接入不同的冻结图像编码器和 LLMs，利用各自领域的最新进展。
*   **零样本能力:** 借助强大的冻结 LLM（特别是指令微调过的 FlanT5），BLIP-2 展现出令人印象深刻的 **零样本指令图像到文本生成 (Instructed Zero-shot Image-to-Text Generation)** 能力，可以根据自然语言指令执行各种视觉语言任务（如视觉对话、视觉知识推理）。

### LLaVA

**LLaVA (Large Language and Vision Assistant)** ([Liu et al., 2023b](https://arxiv.org/abs/2304.08485)) 是**视觉指令微调 (Visual Instruction Tuning)** 开源社区领域的重要工作，首次尝试将 NLP 领域的指令微调思想扩展到多模态领域。

**核心贡献:**

1.  **提出视觉指令微调:** 首次探索将指令微调应用于语言-图像多模态模型，旨在构建通用的视觉助手。
2.  **GPT 辅助数据生成:** 面对视觉指令数据的缺乏，创新性地使用**纯语言模型 GPT-4** (或 ChatGPT) 来生成包含视觉内容的多模态语言-图像指令遵循数据。
3.  **构建 LLaVA 模型:** 提出了一种连接预训练的视觉编码器 (CLIP ViT-L/14) 和大型语言模型 (LLM, Vicuna) 的端到端训练架构。
4.  **创建评估基准:** 构建了 LLaVA-Bench，包含多样化和具有挑战性的任务，用于评估多模态模型的指令遵循能力。
5.  **开源贡献:** 公开了 GPT-4 生成的视觉指令数据、模型代码和预训练权重，极大地推动了社区在这一方向上的研究。

**GPT 辅助视觉指令数据生成:**

LLaVA 解决的关键挑战是缺乏大规模、高质量的视觉指令遵循数据。研究者提出了一种利用现有的多模态大模型如 GPT-4 基于现有的图像-文本对来生成此类数据的方法，本质上这是一种对闭源模型 GPT-4 进行**知识蒸馏**的过程。

1.  **面临的挑战:** 简单的将图像-标题对扩展为 (指令：描述图像，图像 -> 回答：标题) 的格式虽然廉价，但缺乏指令和响应的多样性及深度推理。
2.  **解决方案:** 使用 GPT-4 作为“教师模型”。由于这些模型仅接受文本输入，研究者将图像内容通过**符号表示 (Symbolic Representations)** 传递给它们：
    * **图像描述 (Captions):** 提供图像场景的整体或多方面描述。
    * **边界框 (Bounding Boxes):** 提供图像中对象的类别概念及其空间位置信息 (例如 `person: [0.681, 0.242, 0.774, 0.694]`)。
3.  **提示与上下文学习:** 将图像的符号表示 (描述和边界框) 输入给 GPT-4。为了引导 GPT-4 生成特定格式和内容的输出，研究者手动设计了少量高质量的**种子示例 (Seed Examples)**，利用 GPT-4 的**上下文学习 (In-context Learning)** 能力进行 few-shot 推理。
4.  **生成三种类型数据 (基于 COCO 图像):** 通过精心设计的 Prompt 引导 GPT-4 生成了三种类型的指令数据：
    * **对话 (Conversation):** 生成模拟人与助手之间关于图像内容的多轮对话，包含物体识别、计数、定位、动作、关系等问题。
    * **详细描述 (Detailed Description):** 根据特定指令（如“详细描述下图”）生成对图像全面、细致的描述。
    * **复杂推理 (Complex Reasoning):** 生成需要基于图像内容进行逻辑推理或结合背景知识的问题和答案（如“图中人物可能面临什么挑战？”）。


{{< figure
    src="llava_instruction_data.png"
    caption="Fig. 10. One example to illustrate the instruction-following data. (Image source: [Liu et al., 2023b](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}


5.  **数据集:** 共收集了 **158K** 个独特的语言-图像指令样本，具体包括：**58K** 对话样本，**23K** 详细描述样本，**77K** 复杂推理样本。实验发现，GPT-4 生成的数据质量通常优于 ChatGPT。

{{< figure
    src="llava_architecture.png"
    caption="Fig. 11. LLaVA network architecture. (Image source: [Liu et al., 2023b](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}

LLaVA 的架构旨在有效结合预训练的视觉模型和 LLM 的能力，如上图所示。

1.  **视觉编码器 \( g(\cdot) \):** 使用**冻结的 CLIP ViT-L/14** 模型。对于输入图像 \( \mathbf{X}_{\mathrm{v}} \)，提取其视觉特征 \( \mathbf{Z}_{\mathrm{v}} = g(\mathbf{X}_{\mathrm{v}}) \)。论文中提到实验考虑了最后一层 Transformer 层之前和之后的网格特征。

2.  **投影层:** 使用一个**可训练的线性投影矩阵 \( \mathbf{W} \)** 将视觉特征 \( \mathbf{Z}_{\mathrm{v}} \) 映射到语言模型的词嵌入空间。

    $$
    \mathbf{H}_{\mathrm{v}} = \mathbf{W} \cdot \mathbf{Z}_{\mathrm{v}}
    $$

其中， \( \mathbf{H}_{\mathrm{v}} \) 是一系列视觉 Token，其维度与 LLM 的词嵌入维度相同。这种简单的线性投影方式轻量且高效，便于快速迭代以数据为中心的实验。更复杂的连接方式（如 Flamingo 中的门控交叉注意力或 BLIP-2 中的 Q-Former）可作为未来工作探索。

3.  **大型语言模型 (LLM) \( f_{\phi}(\cdot) \):** 使用 **Vicuna**，其参数表示为 \( \phi \)。LLM 接收视觉 Token \( \mathbf{H}_{\mathrm{v}} \) 和文本指令 \( \mathbf{X}_{\text{instruct}} \)，并自回归地生成答案 \( \mathbf{X}_{\mathrm{a}} \)。

**两阶段训练:**

LLaVA 采用两阶段指令微调流程。

1.  **阶段一：特征对齐预训练 (Feature Alignment Pre-training):**
    * **目标:** 将视觉特征 \( \mathbf{H}_{\mathrm{v}} \) 与 LLM 的词嵌入空间对齐，可以理解为为冻结的 LLM 训练一个兼容的“视觉 Tokenizer”。
    * **数据:** 使用了 CC3M 数据集的一个经过滤的子集 (约 595K 图文对)。将这些图文对通过简单方式转换为指令数据：对于图像 \( \mathbf{X}_{\mathrm{v}} \)，随机选择一个简单的描述指令 \( \mathbf{X}_{\mathrm{q}} \) (如 "简要描述这张图片")，并将原始标题 \( \mathbf{X}_{\mathrm{c}} \) 作为答案 \( \mathbf{X}_{\mathrm{a}} \)。这可以视为单轮对话。
    * **训练:** **冻结** 视觉编码器 \( g(\cdot) \) 和 LLM \( f_{\phi}(\cdot) \) 的权重，**仅训练** 投影层 \( \mathbf{W} \)。训练目标是最大化答案（即图像标题）的似然概率。

2.  **阶段二：端到端微调 (Fine-tuning End-to-End):**
    * **目标:** 提升模型在多模态任务上的指令遵循和对话能力。
    * **数据:** 使用前述生成的 **158K** 视觉指令数据 (包含对话、详细描述、复杂推理三种类型，训练时均匀采样)。
    * **训练:** **冻结** 视觉编码器 \( g(\cdot) \)，**同时训练** 投影层 \( \mathbf{W} \) 和 **LLM \( f_{\phi}(\cdot) \) 的权重**。

**训练目标:**

对于每张图像 \( \mathbf{X}_{\mathrm{v}} \)，生成包含 \( T \) 轮的多轮对话数据 \( \left(\mathbf{X}_{\mathrm{q}}^{1}, \mathbf{X}_{\mathrm{a}}^{1}, \cdots, \mathbf{X}_{\mathrm{q}}^{T}, \mathbf{X}_{\mathrm{a}}^{T}\right) \)，其中 \( T \) 是总对话轮数。将这些数据组织成一个序列，并将所有答案 \( \mathbf{X}_{\mathrm{a}} \) 视为模型的回应。其输入序列的组织形式采用了 Vicuna 格式。在第 \( t \) 轮对话中，指令 \( \mathbf{X}_{\text{instruct}}^{t} \) 定义为：


$$
\mathbf{X}_{\text{instruct}}^{t} = \left\{ \begin{array}{ll} \text{Randomly choose } [\mathbf{X}_{\mathrm{q}}^{1}, \mathbf{X}_{\mathrm{v}}] \text{or } [\mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\mathrm{q}}^{1}], & \text{ if } t=1 \text{ (the first turn)} \\ \mathbf{X}_{\mathrm{q}}^{t}, & \text{ if } t>1 \text{ (the remaining turns)} \end{array} \right.
$$

目标是预测答案序列 \( \mathbf{X}_{\mathrm{a}} = (\mathbf{X}_{\mathrm{a}}^{1}, \dots, \mathbf{X}_{\mathrm{a}}^{T}) \)。模型需要最大化在给定图像 \( \mathbf{X}_{\mathrm{v}} \) 和所有指令 \( \mathbf{X}_{\text{instruct}} = (\mathbf{X}_{\text{instruct}}^{1}, \dots, \mathbf{X}_{\text{instruct}}^{T}) \) 的条件下，生成正确答案序列的概率。对于长度为 \( L \) 的完整答案序列（所有轮次的 \( \mathbf{X}_{\mathrm{a}} \) 拼接而成），其概率计算如下：

$$
p\left(\mathbf{X}_{\mathrm{a}} \mid \mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\text {instruct }}\right)=\prod_{i=1}^L p_{\boldsymbol{\theta}}\left(x_i \mid \mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\text {instruct },\lt i}, \mathbf{X}_{\mathrm{a},\lt i}\right)
$$

其中：
* \( \boldsymbol{\theta} \) 是模型的可训练参数。
    * 在阶段一，\( \boldsymbol{\theta} = \{ \mathbf{W} \} \)。
    * 在阶段二，\( \boldsymbol{\theta} = \{ \mathbf{W}, \phi \} \)。
* \( x_i \) 是答案序列 \( \mathbf{X}_{\mathrm{a}} \) 中的第 \( i \) 个 token。
* \( \mathbf{X}_{\text{instruct},\lt i} \) 和 \( \mathbf{X}_{\mathrm{a},\lt i} \) 分别代表在预测 \( x_i \) 时，模型已接收到的所有指令 token 和已生成的所有答案 token。
* 训练时的损失函数是上述概率的**负对数似然 (Negative Log-Likelihood)**，并且**仅在答案部分的 token (即 \( \mathbf{X}_{\mathrm{a}} \) 中的 token)** 上计算损失。


**效果与影响:**

LLaVA 在多模态对话方面展示了令人印象深刻的能力，有时能在未见过的图像和指令上表现出类似多模态 GPT-4 的行为。在 ScienceQA 基准测试上进行微调后，LLaVA 与 GPT-4 的结合取得了当时最先进的 92.53% 准确率。

{{< figure
    src="llava_science_qa_accuracy.png"
    caption="Fig. 12. Accuracy (%) on Science QA dataset. (Image source: [Liu et al., 2023b](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}

LLaVA 的成功证明了视觉指令微调的有效性，其开源的数据、代码和模型极大地促进了后续多模态大模型的研究，为构建通用的、能够理解并遵循视觉和语言指令的 AI 助手开辟了新的途径。

### Qwen2-VL

**Qwen2-VL** ([Wang et al., 2024f](https://arxiv.org/abs/2409.12191)) 是 Qwen-VL ([Bai et al., 2023b](https://arxiv.org/abs/2308.12966)) 的升级版，在处理可变分辨率视觉输入和融合多模态位置信息方面取得了显著进展。


{{< figure
    src="qwen2_vl.jpg"
    caption="Fig. 13. Qwen2-VL is capable of accurately identifying and comprehending the content within images, regardless of their clarity, resolution, or extreme aspect ratios.: ([Wang et al., 2024f](https://arxiv.org/abs/2409.12191))"
    align="center"
    width="100%"
>}}

**核心贡献:**

1.  **原生动态分辨率 (Naive Dynamic Resolution):** 模型能够处理任意分辨率的图像，并将其动态地转换为变长的视觉 token 序列。这通过在 ViT 中使用 **2D-RoPE** 替代绝对位置编码实现。
2.  **多模态旋转位置编码 (Multimodal Rotary Position Embedding, M-RoPE):** 提出了一种新的位置编码方法，可以统一处理文本、图像和视频的位置信息。
3.  **统一图像与视频理解:** 采用混合训练范式和特定架构设计（如 3D 卷积处理视频）来同时处理图像和视频。
4.  **模型扩展:** 发布了 2B, 8B, 72B 三种规模的模型，探索了 LVLM 的扩展规律。

**架构改进:**

*   **动态分辨率 ViT:**
    *   移除 ViT 的绝对位置编码，引入 **2D-RoPE**。
    *   推理时，可变分辨率图像被打包处理，限制总 token 长度以控制显存。
    *   ViT 输出后，使用 MLP 压缩相邻 \( 2 \times 2 \) 的 token 为一个，减少输入 LLM 的序列长度。使用 `<|vision_start|>` 和 `<|vision_end|>` 包裹视觉 token。
*   **M-RoPE:**
    *   将 RoPE 分解为 **时间 (Temporal)**、**高度 (Height)**、**宽度 (Width)** 三个分量。
    *   **文本:** 三个分量使用相同的位置 ID，等价于 1D-RoPE。
    *   **图像:** 时间 ID 恒定，高度和宽度 ID 根据 token 在图像中的二维位置赋值。
    *   **视频:** 时间 ID 随帧数递增，高度和宽度 ID 同图像。
    *   **多模态输入:** 不同模态的位置 ID 依次递增。
    *   **优势:** 统一编码多模态位置信息，降低了图像/视频的位置 ID 值，有利于推理时外插到更长序列。

{{< figure
    src="mrope.png"
    caption="Fig. 14. Illustration of M-RoPE. By decomposing rotary embedding into temporal, height, and width components, M-RoPE can explicitly model the positional information of text, images, and video in LLM. (Image source: [Wang et al., 2024f](https://arxiv.org/abs/2409.12191))"
    align="center"
    width="100%"
>}}

*   **统一图像/视频处理:**
    *   混合图像和视频数据进行训练。
    *   视频以 2 FPS 采样。
    *   ViT 中集成 **3D 卷积** 处理视频输入 (处理 \( 2 \times 14 \times 14 \) 的 3D 块)，减少 token 数量。
    *   图像被视为两帧相同的视频帧。
    *   动态调整视频帧分辨率，限制每段视频的总 token 数（如 16384）。

**训练:** 沿用 Qwen-VL 的三阶段训练：ViT 预训练 -> 全模型预训练 -> LLM 指令微调。预训练数据包含图文对、OCR、图文交错文章、VQA、视频对话、图像知识等。指令微调使用 ChatML 格式。

**效果:** Qwen2-VL 在多种分辨率和长宽比的图像理解、长视频理解（超过 20 分钟）以及视觉 Agent 能力方面表现出色。

### Qwen2.5-VL

**Qwen2.5-VL** ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)) 在 Qwen2-VL 的基础上进一步优化了效率和时序建模能力。

**核心贡献:**

1.  **高效 ViT 架构:** 在 ViT 中引入 **窗口注意力 (Window Attention)**，将大部分层的注意力计算限制在局部窗口内，使计算复杂度与图像块数量成线性关系，显著提升处理高分辨率图像的效率。仅保留少数层进行全局注意力计算。
2.  **动态 FPS 采样:** 将动态分辨率思想扩展到时间维度，训练时对视频采用 **动态帧率 (Dynamic FPS)** 采样，增强模型对不同速率视频的理解能力。
3.  **绝对时间对齐 M-RoPE (TMRoPE):** 改进 M-RoPE，将其 **时间分量与绝对时间戳对齐**（例如，每个时间 ID 对应 40ms），而不是仅仅基于帧序号。这使得模型能够感知事件的真实速率和精确定位时间点，不受采样 FPS 变化的影响。
4.  **增强的数据与能力:** 使用更大规模（4.1T tokens）和更高质量的数据进行预训练和微调，特别加强了文档解析（表格、图表、公式、乐谱等）、对象定位（支持点和框）、长视频理解（小时级）和 Agent 能力。

**架构改进:**

*   **窗口注意力 ViT:** 大部分 ViT 层使用窗口注意力（如 \( 8 \times 8 \) patch 窗口），少数层（如每隔 8 层）使用全注意力。
*   **TMRoPE:** M-RoPE 的时间 ID 不再是简单的帧序号，而是根据视频帧的实际时间戳计算得出，保持时间 ID 与绝对时间的对应关系（如 1 ID = 40ms）。
*   **视频处理:** 依然采用 3D 块处理（\( 2 \times 14 \times 14 \)），结合动态 FPS 采样和 TMRoPE。

{{< figure
    src="qwen2.5vl_arc.jpeg"
    caption="Fig. 15. The Qwen2.5-VL framework demonstrates the integration of a vision encoder and a language model decoder to process multimodal inputs. The vision encoder is designed to handle inputs at their native resolution and supports dynamic FPS sampling. TMRoPE aligns time IDs with absolute time along the temporal dimension. (Image source: [Bai et al., 2025](https://arxiv.org/abs/2502.13923))"
    align="center"
    width="100%"
>}}

**数据增强:**

*   **文档全解析数据:** 构建了包含表格、图表、公式、图片、乐谱、化学式的 HTML 格式数据，包含布局框信息和坐标。
*   **定位数据:** 扩展了边界框和点的定位数据，覆盖超过 1 万个类别，并合成了包含不存在对象和多实例对象的难例。使用了 Grounding DINO 和 SAM 等工具合成数据。
*   **OCR 数据:** 增加了多语言 OCR 数据（覆盖欧洲主要语言及日韩阿越等），并包含手写体、密集文本、网页、公式、图表、表格等多种场景。
*   **视频数据:** 增加了长视频（超过半小时）的密集描述数据，并采用动态 FPS 采样训练。时间戳标注包含秒和 HMSF 两种格式。
*   **Agent 数据:** 收集了移动端、Web 端、桌面端的截图和操作轨迹，统一为函数调用格式，并合成了 CoT 推理过程。

**效果:** Qwen2.5-VL 在文档理解、细粒度定位、长视频理解和 Agent 任务上取得了 SOTA 性能，72B 版本在多个基准上媲美甚至超越 GPT-4o 和 Claude 3.5 Sonnet。

### Qwen2.5-Omni

**Qwen2.5-Omni** ([Qwen Team, 2025](https://arxiv.org/pdf/2503.20215)) 是一个端到端的多模态模型，旨在统一处理文本、图像、音频和视频输入，并能同时 **流式生成文本和自然语音** 输出。

{{< figure
    src="qwen2.5_omni.png"
    caption="Fig. 16. Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="100%"
>}}

**核心贡献:**

1.  **全模态感知:** 单一模型处理文本、图像、音频、视频四种模态输入。
2.  **时序对齐 TMRoPE:** 进一步优化 Qwen2.5-VL 的 TMRoPE，通过 **时间交错 (Time-interleaving)** 结构将视频帧和音频帧按时间顺序排列，并使用绝对时间戳对齐位置编码，实现音视频同步理解。
3.  **Thinker-Talker 架构:** 提出一种新颖架构用于解耦文本生成和语音生成，避免相互干扰，同时允许端到端训练。
    *   **Thinker:** 核心 LLM (基于 Qwen2.5)，负责理解多模态输入，生成高级表示和文本输出。
    *   **Talker:** 一个 **双轨自回归 Transformer 解码器**，接收 Thinker 的隐层表示和生成的文本 token，专门负责生成 **离散语音 token**。
4.  **流式处理:**
    *   **输入:** 音频和视觉编码器采用 **分块处理 (Block-wise Processing)**，支持流式输入和预填充 (Prefilling)。
    *   **输出:** Talker 生成离散语音 token，一个 **流式音频解码器 (Streaming Audio Codec)** (基于滑动窗口 DiT 和 Flow Matching) 将这些 token 实时转换为音频波形，显著降低首包延迟。

**架构细节:**

*   **输入处理:**
    *   文本: Qwen tokenizer。
    *   音频: 16kHz 采样，128 通道梅尔频谱图 (25ms 窗长, 10ms 步长)，使用 Qwen2-Audio 的编码器 (每帧约 40ms)。
    *   图像/视频: 使用 Qwen2.5-VL 的 ViT，视频采用动态 FPS 采样。
    *   **时间交错与 TMRoPE:** 对于带音频的视频，每 2 秒切块，块内先排视频帧表示，后排音频帧表示。所有模态使用 TMRoPE 进行位置编码，时间 ID 与绝对时间 (40ms 粒度) 对齐。
*   **Thinker-Talker:**
    *   Thinker 是基于 Qwen2.5 的 Transformer 解码器。
    *   Talker 接收 Thinker 的 **高维隐层表示** (提供语义和韵律信息) 和 **采样的文本 token** (消除语音歧义)，自回归地生成离散语音 token。
    *   两者共享历史上下文，端到端训练。
*   **流式语音解码:**
    *   使用 qwen-tts-tokenizer 将语音编码为离散 token。
    *   解码器基于 **DiT (Diffusion Transformer)**，采用 **滑动窗口块注意力 (Sliding Window Block Attention)** (如回看 2 块，前看 1 块)，限制感受野，实现流式生成。
    *   使用 **Flow Matching** 将离散 token 转换为梅尔频谱图块。
    *   使用修改版的 **BigVGAN** 将梅尔频谱图块流式转换为波形。

{{< figure
    src="qwen2.5_omini_arc.png"
    caption="Fig. 17. Qwen2.5-Omni Overview. Adopts Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receiving high-level representations directly from Thinker. (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="100%"
>}}

**训练:** 包含三个阶段：编码器与 LLM 对齐 -> 全模型多模态预训练 -> 长上下文预训练 (32k)。Talker 单独进行三阶段训练：上下文学习 -> DPO (优化稳定性) -> 多说话人指令微调 (提升自然度)。

**效果:** Qwen2.5-Omni 在各项单模态基准上与同规模的 Qwen2.5-VL (视觉) 和 Qwen2-Audio (音频) 表现相当或更好。在 OmniBench 等多模态融合基准上达到 SOTA。语音指令遵循能力接近文本指令。语音生成在鲁棒性和自然度上优于多数现有模型。

### Kimi-VL

**Kimi-VL** ([Kimi Team, 2025](https://arxiv.org/pdf/2504.07491)) 是一款开源的 **高效混合专家 (Mixture-of-Experts, MoE)** 视觉语言模型。

**核心贡献:**

1.  **高效 MoE 架构:** 语言模型部分采用 MoE 架构 (基于 Moonlight 模型)，总参数 16B，但每次推理 **仅激活 2.8B** 参数，显著降低了计算成本。
2.  **原生分辨率视觉编码器 (MoonViT):** 提出 MoonViT (400M 参数)，能够 **原生处理** 不同分辨率的图像，无需缩放或填充，更好地保留图像细节。结合了 NaViT 的图像打包和 2D-RoPE。
3.  **长上下文能力:** 支持 **128K** 的上下文窗口，能够处理长文档、长视频等输入。
4.  **长思维链推理 (Kimi-VL-Thinking):** 通过长 CoT SFT 和 RL 训练，推出了具备更强长链推理能力的 Kimi-VL-Thinking 版本。

**架构:**

*   **MoonViT:**
    *   基于 ViT，使用 NaViT 的打包方法处理变长序列。
    *   结合了 **插值的绝对位置编码** (继承自 SigLIP 初始化) 和 **2D-RoPE**。
    *   训练时采用动态分辨率采样。
*   **MLP Projector:** 2 层 MLP，包含 Pixel Shuffle 操作压缩空间维度。
*   **MoE LLM:** 基于 Moonlight (类 DeepSeek-V3 架构)，16B 总参数，2.8B 激活参数。

{{< figure
    src="kimi_vl_arch.png"
    caption="Fig. 18. Model architecture of Kimi-VL and Kimi-VL-Thinking, consisting of a MoonViT that allows native-resolution images, an MLP projector, and a Mixture-of-Experts (MoE) language decoder. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.16790))"
    align="center"
    width="100%"
>}}

**训练:**

*   **预训练 (4 阶段, 共 4.4T tokens):**
    1.  **ViT 训练 (2.1T):** 单独训练 MoonViT (从 SigLIP 初始化)，使用对比损失和描述生成损失。
    2.  **联合预训练 (1.4T):** 联合训练 ViT, Projector, LLM (从 Moonlight 5.2T checkpoint 初始化)，混合文本和多模态数据。
    3.  **联合冷却 (0.6T):** 使用高质量文本和多模态数据继续联合训练。
    4.  **联合长上下文激活 (0.3T):** 将上下文从 8K 扩展到 128K，使用长文本、长视频、长文档数据。
*   **后训练 (Post-Training):**
    1.  **联合 SFT:** 使用 ChatML 格式，在混合文本和多模态指令数据上进行微调 (先 32K 再 128K 上下文)。
    2.  **(Kimi-VL-Thinking) 长 CoT SFT:** 使用少量高质量长 CoT 数据进行 SFT，激活长链推理能力。
    3.  **(Kimi-VL-Thinking) RL:** 使用在线策略镜像下降 RL 算法，结合基于答案正确性的奖励和长度惩罚，进一步提升推理能力。

**效果:** Kimi-VL (A3B) 在多项基准测试中表现出色，尤其在长上下文（LongVideoBench, MMLongBench-Doc）、高分辨率（InfoVQA, ScreenSpot-Pro）和 Agent 任务（OSWorld）上具有优势，性能媲美甚至超越了规模更大的模型如 Qwen2.5-VL-7B 和 Gemma-3-12B-IT。Kimi-VL-Thinking 在复杂推理任务（MMMU, MathVision, MathVista）上表现突出，以极小的激活参数量达到了领先水平。

### o3 & o4-mini

OpenAI 的 **o3** 和 **o4-mini** ([OpenAI, 2025](https://openai.com/index/introducing-openai-o3-and-o4-mini/)) 是其 o 系列推理模型的最新迭代，核心特点是 **更长的思考时间 (Longer Thinking Time)** 和 **全面的工具接入 (Full Tool Access)**。

**核心贡献:**

1.  **增强推理:** 模型被训练成在响应前进行更长时间、更深入的思考（类似于 CoT 或更复杂的推理过程），显著提升了在编码、数学、科学、视觉感知等复杂任务上的性能。o3 在 Codeforces, SWE-bench, MMMU 等基准上达到 SOTA。
2.  **全工具接入:** 模型可以无缝调用各种工具，如 **网页搜索 (Web Search)**、**代码解释器 (Code Interpreter)**、**图像生成 (Image Generation)**，以及通过 API 实现的 **函数调用 (Function Calling)**。模型经过训练，能够自主判断何时以及如何使用这些工具来解决问题。
3.  **多模态推理:** 模型可以将 **图像直接整合进其思维链 (Chain of Thought)**，实现视觉和文本的深度融合推理，而不仅仅是将图像作为输入。这使其在分析图表、图示等方面表现优异。
4.  **效率与性能权衡:** o3 是最强模型，适用于复杂查询；o4-mini 则针对速度和成本进行了优化，参数量更小，但在数学、编码和视觉任务上仍表现出色，尤其擅长利用工具（如在 AIME 竞赛中使用 Python 解释器）。
5.  **大规模强化学习:** o 系列模型的性能提升很大程度上归功于大规模强化学习 (RL) 的应用，验证了 RL 在提升推理能力方面的潜力，且性能随计算量增加而提升。

**工作机制:**

*   **长时间思考:** 模型内部被设计为可以进行多步推理或更复杂的计算过程，这可能通过增加模型深度、宽度或采用特定的推理算法（如 MCTS 的变体或内部 CoT）实现。用户可以通过选择不同的“推理努力程度 (reasoning effort)”设置（如 o4-mini-high）来调整模型的思考时间。
*   **工具使用:** 模型通过 RL 或指令微调学习工具使用的策略。当面对一个问题时，模型会：
    1.  **规划:** 分析问题，判断是否需要以及需要哪些工具。
    2.  **执行:** 调用选定的工具（如进行网络搜索获取最新信息，运行代码进行计算）。
    3.  **整合:** 将工具返回的结果整合到其推理过程中，生成最终答案。
    这个过程可以是多轮迭代的，模型可以根据工具返回的信息调整策略（如进行二次搜索）。
*   **多模态 CoT:** 模型可以直接在其内部推理步骤中引用和分析图像内容，例如识别图表中的数据点，理解流程图的步骤，或解释照片中的细节。

{{< figure
    src="thinking_with_images_static.webp"
    caption="Fig. 19. o3 model demonstrates its multimodal CoT capability by analyzing a user-uploaded image, identifying the ship, and using tools (web search) to find information, ultimately answering the ship's name and its next port of call. (Image source: [OpenAI, 2025](https://openai.com/index/introducing-o3-and-o4-mini/))"
    align="center"
    width="100%"
>}}

**效果:** o3 和 o4-mini 在多项基准测试中展现了 SOTA 或接近 SOTA 的性能，尤其是在需要深度推理和工具辅助的任务上。专家评估显示，它们相比前代 o1/o3-mini 产生的严重错误更少，回答更实用、可验证，并且交互更自然。

### 多模态思维链

**多模态思维链 (Multimodal Chain-of-Thought, MCoT)** ([Wang et al., 2025](https://arxiv.org/abs/2503.12605)) 是将 CoT 推理从纯文本领域扩展到包含图像、视频、音频等多种模态的场景中的一种方法论。其核心思想是模仿人类处理复杂多模态信息时的分步推理过程，生成一系列中间 **思考步骤 (Thoughts)** 或 **基本原理 (Rationale)**，最终导出答案。

{{< figure
    src="mcot_timeline.png"
    caption="Fig. 20. MCoT timeline. (Image source: [Wang et al., 2025](https://arxiv.org/abs/2503.12605))"
    align="center"
    width="100%"
>}}


**重要性:** MCoT 对于提升 MLLMs 在需要复杂推理的多模态任务（如科学问答、视觉常识推理、视频事件理解、具身智能规划等）上的性能至关重要。它提高了模型决策过程的 **透明度** 和 **可解释性**，并有助于 **缓解模型幻觉**。

**MCoT 的关键方面 (根据 [Wang et al., 2025](https://arxiv.org/abs/2503.12605) 的综述):**

1.  **基本原理构建 (Rationale Construction):**
    *   **基于提示 (Prompt-based):** 通过零样本（如 "think step-by-step"）或少样本提示引导模型生成推理链。简单灵活，但效果依赖提示质量。
    *   **基于规划 (Plan-based):** 模型在推理过程中动态探索和评估不同的思考路径，如思维树 (Tree-of-Thoughts, ToT) 或思维图 (Graph-of-Thoughts, GoT)。允许更复杂的探索和回溯。
    *   **基于学习 (Learning-based):** 通过在包含显式推理链的数据上进行微调 (SFT) 或强化学习 (RL) 来让模型学习生成推理链。这是目前高性能模型（如 o3, Kimi-VL-Thinking）采用的主流方法。

2.  **结构化推理 (Structural Reasoning):** 为了增强推理过程的可控性和可解释性，研究者提出了不同的结构化方法：
    *   **异步模态建模 (Asynchronous Modality Modeling):** 分离感知（如图像描述）和决策（基于描述进行推理）阶段，模拟人类认知。
    *   **定义过程分段 (Defined Procedure Staging):** 将复杂任务分解为预定义的、有序的子步骤（如识别->定位->关系推理）。
    *   **自主过程分段 (Autonomous Procedure Staging):** 让模型自主决定推理步骤和顺序。

3.  **信息增强 (Information Enhancing):** 在推理过程中整合额外信息：
    *   **使用专家工具 (Expert Tools):** 调用外部工具（如计算器、代码解释器、图像编辑器、搜索引擎）来辅助推理。o3/o4-mini 是典型代表。
    *   **使用世界知识检索 (World Knowledge Retrieval):** 通过 RAG (Retrieval-Augmented Generation) 等方式从外部知识库（如维基百科、领域数据库）检索信息。
    *   **利用上下文知识检索 (In-context Knowledge Retrieval):** 从输入的多模态内容或模型自身生成的中间步骤中提取和组织信息（如构建场景图）。

4.  **目标粒度 (Objective Granularity):** 推理的目标可以是不同粒度的：
    *   **粗粒度理解 (Coarse Understanding):** 对整个场景或主要内容进行概括性推理（如 VQA）。
    *   **语义定位 (Semantic Grounding):** 将语言描述与图像/视频中的特定区域或对象实例精确对应（如 Referring Expression Comprehension/Segmentation）。
    *   **细粒度理解 (Fine-grained Understanding):** 关注图像/视频中的细节信息进行推理。

5.  **多模态基本原理 (Multimodal Rationale):** 推理链本身是否包含非文本模态：
    *   **纯文本原理 (Text-only Rationale):** 大多数 MCoT 方法生成纯文本的推理步骤。
    *   **多模态原理 (Multimodal Rationale):** 推理链中包含图像、草图或其他视觉/听觉元素，如 Visual-CoT, Chain-of-Image, Visualization-of-Thought。这更接近人类的思考方式（例如，在脑海中想象画面）。

6.  **测试时扩展/慢思考 (Test-Time Scaling / Slow Thinking):**
    *   通过在推理时增加计算量（如多次采样、使用更长的推理链、MCTS 搜索）来提升复杂任务的性能。o3/o4-mini 的“更长思考时间”和 Kimi-VL-Thinking 是这方面的实践。
    *   **强化学习 (RL):** 如 DeepSeek-R1, R1-V, MM-Eureka-Zero 等工作表明，RL 可以有效激发模型的长链推理能力，甚至在没有 SFT 的情况下也能产生“顿悟时刻 (Aha Moment)”。RL 通常结合基于结果的奖励（ORM）或基于过程的奖励（PRM）进行优化。

MCoT 是当前 MLLM 研究的热点，它不仅提升了模型的推理能力，也为实现更通用、更接近人类智能的 AI 系统提供了重要途径。

## 讨论

多模态 AI 领域正经历着飞速的发展，从早期的简单模态融合到如今能够进行复杂推理、遵循指令、甚至使用工具的 MLLMs，技术演进的轨迹清晰可见。我们可以观察到以下几个关键趋势和挑战：

1.  **架构融合与统一:** 模型架构趋向于更加统一和灵活。早期的模型往往针对特定任务设计，而像 BLIP 的 MED、BLIP-2 的 Q-Former、LLaVA 的简单连接器、Qwen 系列的统一框架以及 Kimi-VL 的 MoE 设计，都体现了构建通用多模态基础模型的努力。Qwen2.5-Omni 更是将多种输入和输出模态整合到单一模型中。
2.  **效率与性能的平衡:** 随着模型规模的增大，计算成本成为瓶颈。BLIP-2 的冻结模型策略、Qwen2.5-VL 的窗口注意力、Kimi-VL 的 MoE 架构都是在追求更高性能的同时，探索更高效的训练和推理方法。o4-mini 的推出也明确指向了对高性价比模型的需求。
3.  **指令遵循与交互性:** 从简单的 VQA、描述生成，到 LLaVA 开创的视觉指令微调，再到 Qwen 系列和 Kimi-VL 展示的 Agent 能力，模型与用户的交互方式越来越丰富和主动，逐渐从被动回答转向主动执行任务。
4.  **深度推理与 CoT:** 简单地关联图文已不能满足需求，模型需要具备更深层次的推理能力。MCoT 的引入，特别是结合 RL 和长思考时间的 o3/o4-mini 以及 Kimi-VL-Thinking，显著提升了模型解决复杂问题的能力。如何有效地生成和利用高质量的推理链数据是未来的关键。
5.  **原生分辨率与时序理解:** 突破固定分辨率限制（如 Qwen2-VL, Kimi-VL）和增强对视频时序动态的理解（如 Qwen2.5-VL/Omni 的 TMRoPE）是提升模型感知真实世界能力的重要方向。
6.  **数据的作用:** 高质量、大规模的多模态数据是驱动模型进步的核心燃料。无论是 CLIP 的 WIT 数据集，还是 BLIP 的 CapFilt 技术，亦或是 LLaVA 的 GPT-4 生成数据，以及 Qwen 和 Kimi 系列对数据工程的重视，都凸显了数据在 MLLM 发展中的关键作用。如何高效获取、清洗、生成和利用多模态数据（尤其是包含推理链的数据）将持续是研究热点。
7.  **挑战依然存在:**
    *   **模型幻觉 (Hallucination):** MLLMs 仍可能生成与视觉内容不符或凭空捏造的信息。
    *   **细粒度理解:** 对于图像或视频中的精细细节、复杂关系和微妙变化的理解仍有提升空间。
    *   **常识与世界知识:** 如何将丰富的常识和世界知识有效融入多模态理解与推理中。
    *   **评估:** 如何全面、可靠地评估 MLLMs 的综合能力，特别是其推理和泛化能力，仍然是一个难题。
    *   **安全与偏见:** 模型可能继承训练数据中的偏见，并可能被用于生成有害内容或执行不当任务。

## 总结

多模态 AI 正在从基础的感知对齐走向更高级的认知智能。以 ViT 为代表的视觉编码器奠定了基础，CLIP 通过对比学习实现了高效的图文对齐。BLIP 和 BLIP-2 探索了统一理解与生成以及利用冻结模型的效率优化路径。LLaVA 引入了视觉指令微调，增强了模型的交互性。Qwen 系列在动态分辨率、时序建模和全模态处理上不断突破。Kimi-VL 则展示了 MoE 架构在效率和长上下文处理上的潜力。o3/o4-mini 和 MCoT 的研究将模型的推理能力推向了新的高度，并结合工具使用，展现了未来 Agent 形态的雏形。

## 参考文献

[1] OpenAI. ["Hello gpt-4o."](https://openai.com/index/hello-gpt-4o/) OpenAI Blog, 2024.

[2] DeepMind. ["Gemini - Google DeepMind."](https://deepmind.google/technologies/gemini/pro/) Google DeepMind, 2025.

[3] OpenAI. ["Introducing OpenAI o3 and o4-mini."](https://openai.com/index/introducing-openai-o3-and-o4-mini/) OpenAI Blog, 2025.

[4] Dosovitskiy, Alexey, et al. ["An image is worth 16x16 words: Transformers for image recognition at scale."](https://arxiv.org/abs/2010.11929) arXiv preprint arXiv:2010.11929 (2020).

[5] He, Kaiming, et al. ["Deep residual learning for image recognition."](https://arxiv.org/abs/1512.03385) In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770-778. 2016.

[6] Wang, Peng, et al. ["Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution."](https://arxiv.org/abs/2409.12191) arXiv preprint arXiv:2409.12191 (2024).

[7] Kimi Team. ["Kimi-VL Technical Report."](https://arxiv.org/abs/2504.16790) arXiv preprint arXiv:2504.16790 (2025).

[8] Su, Jianlin, et al. ["Roformer: Enhanced Transformer with rotary position embedding."](https://arxiv.org/abs/2104.09864) *Neurocomputing* 568 (2024): 127063.

[9] Su, Jianlin. ["Transformer升级之路：4、二维位置的旋转位置编码."](https://spaces.ac.cn/archives/8397) Spaces (blog), 2021.

[10] Dehghani, Mostafa, et al. ["Patch n' pack: Navit, a vision transformer for any aspect ratio and resolution."](https://arxiv.org/abs/2307.06304) *Advances in Neural Information Processing Systems* 36 (2023).

[11] Bai, Shuai, et al. ["Qwen2.5-VL Technical Report."](https://arxiv.org/abs/2502.13923) arXiv preprint arXiv:2502.13923 (2025).

[12] Zhang, Biao, and Rico Sennrich. ["Root mean square layer normalization."](https://arxiv.org/abs/1910.07467) *Advances in neural information processing systems* 32 (2019).

[13] Shazeer, Noam. ["Glu variants improve transformer."](https://arxiv.org/abs/2002.05202) arXiv preprint arXiv:2002.05202 (2020).

[14] Radford, Alec, et al. ["Learning transferable visual models from natural language supervision."](https://arxiv.org/abs/2103.00020) In *International conference on machine learning*, pp. 8748-8763. PMLR, 2021.

[15] Li, Junnan, Dongxu Li, Caiming Xiong, and Steven Hoi. ["Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation."](https://arxiv.org/abs/2201.12086) In *International conference on machine learning*, pp. 12888-12900. PMLR, 2022.

[16] Li, Junnan, Ramprasaath R. Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven CH Hoi. ["Align before fuse: Vision and language representation learning with momentum distillation."](https://arxiv.org/abs/2107.07651) *Advances in Neural Information Processing Systems* 34 (2021): 9267-9279.

[17] Li, Junnan, Dongxu Li, Silvio Savarese, and Steven Hoi. ["Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models."](https://arxiv.org/abs/2301.12597) In *International Conference on Machine Learning*, pp. 19730-19742. PMLR, 2023.

[18] Liu, Haotian, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. ["Visual instruction tuning."](https://arxiv.org/abs/2304.08485) *Advances in Neural Information Processing Systems* 36 (2023).

[19] Bai, Jinze, et al. ["Qwen-vl: A frontier large vision language model with versatile abilities."](https://arxiv.org/abs/2308.12966) arXiv preprint arXiv:2308.12966 (2023).

[20] Qwen Team. ["Qwen2.5-Omni Technical Report."](https://arxiv.org/abs/2504.14786) arXiv preprint arXiv:2504.14786 (2025). (Note: Link in draft was different, used the one matching the citation text)

[21] Wang, Yaoting, et al. ["Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey."](https://arxiv.org/abs/2503.12605) arXiv preprint arXiv:2503.12605 (2025).

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Apr 2025). 多模态 AI 综述：从基础到前沿技术演进.
> [你的博客链接]

Or

```bibtex
@article{yue_shui_multimodal_survey_2025,
  title   = "多模态 AI 综述：从基础到前沿技术演进",
  author  = "Yue Shui",
  journal = "你的博客名称或网址",
  year    = "2025",
  month   = "Apr",
  url     = "[你的博客链接]"
}
```