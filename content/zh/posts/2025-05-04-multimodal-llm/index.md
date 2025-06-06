---
title: "多模态大语言模型"
date: 2025-05-04T12:00:00+08:00
author: "Yue Shui"
tags: ["Multimodal", "MLLMs", "ViT", "CLIP", "BLIP", "LLaVA", "OpenAI", "Qwen-VL", "Kimi-VL"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

人类通过多种感官（视觉、听觉、触觉等）与世界互动，每种感官通道在表征和交流特定概念时都具有独特的优势。这种多模态交互促进了我们对世界的深刻理解。人工智能领域的核心目标之一便是开发能够有效遵循多模态指令（如视觉和语言）的通用助手，使其能够像人类一样完成现实世界的各种任务。近年来，随着 GPT-4o ([OpenAI, 2024](https://openai.com/index/hello-gpt-4o/))、Gemini 2.5 Pro ([DeepMind, 2025](https://deepmind.google/technologies/gemini/pro/)) 和 o3/o4-mini ([OpenAI, 2025](https://openai.com/index/introducing-o3-and-o4-mini/)) 等模型的发布，**多模态大语言模型（Multimodal Large Language Models, MLLMs）** 取得了显著进展，它们不仅能理解图像、视频、音频等多种模态信息，还能进行复杂的推理和生成。

## 符号表

下面列举了文章中使用的关键数学公式符号及其含义，以帮助你更轻松地阅读。

| 符号                                                                         | 说明                                                                                                                             |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| \( I, \mathbf{X}_v \)                                                        | 图像输入， \( I \) 通常指原始图像矩阵 \( \in \mathbb{R}^{H \times W \times C} \)                                                     |
| \( T, \mathbf{X}_c, \mathbf{X}_q, \mathbf{X}_a, \mathbf{X}_{\text{instruct}} \) | 文本输入，具体可能指图像标题(\( \mathbf{X}_c \))、用户问题(\( \mathbf{X}_q \))、模型回答(\( \mathbf{X}_a \))或指令(\( \mathbf{X}_{\text{instruct}} \)) |
| \( V, \mathbf{Z}_v \)                                                        | 图像编码器输出的原始图像特征或嵌入序列                                                                                             |
| \( L, \mathbf{H}_q, \mathbf{H}_a \)                                          | 文本编码器输出的文本特征或嵌入序列                                                                                             |
| \( \mathbf{H}_v \)                                                           | 经过投影层处理后，输入到 LLM 的视觉 Token 序列                                                                 |
| \( Z \)                                                                      | Q-Former 输出的查询嵌入，作为视觉信息的压缩表示                                                                        |
| \( P_Z \)                                                                    | 由 Q-Former 输出转换得到的软视觉提示 (Soft Visual Prompt)                                                             |
| \( I_e, T_e \)                                                               | 在 CLIP 的共享多模态嵌入空间中的图像和文本嵌入                                                                                     |
| \( z_p \)                                                                    | ViT 中单个图像块经过线性投射后的嵌入向量                                                                                         |
| \( x_{class} \)                                                              | ViT 中用于分类任务的可学习 `[class]` 标记的嵌入                                                                                   |
| \( x_i \)                                                                    | 序列中的第 \( i \) 个元素或 Token (例如文本序列中的词 \( w_i \))                                                                   |
| \( E_{img}, g(\cdot) \)                                                      | 图像编码器模型 (如 ViT)                                                                                                          |
| \( E_{text}, f_{\phi}(\cdot) \)                                              | 文本编码器或大语言模型                                                                                                   |
| \( E, \mathbf{W}, \mathbf{W}_i, \mathbf{W}_t \)                              | 线性投影矩阵，用于特征转换或模态对齐                                                                                             |
| \( E_{pos} \)                                                                | 位置编码向量，用于向 Transformer 提供序列的位置信息                                                                                |
| \( Q, K, V \)                                                                | 注意力机制中的 Query、Key、Value 矩阵                                                                              |
| \( W_Q, W_K, W_V \)                                                          | 用于从输入计算 Q, K, V 的可学习投影矩阵                                                                                            |
| \( \theta, \phi \)                                                           | 模型整体或特定部分 (如 LLM \( \phi \)) 的可训练参数集合                                                                             |
| \( P \)                                                                      | ViT 模型中定义的图像块 (Patch) 的边长                                                                                              |
| \( N \)                                                                      | 批次大小 (Batch Size)，通常指一个批次中的样本数量                                                                                  |
| \( N_{patches} \)                                                            | ViT 模型将图像分割成的图像块数量                                                                                                 |
| \( D \)                                                                      | 模型中嵌入向量的主要维度                                                                                                         |
| \( d, d_k \)                                                                 | 注意力机制中 Key向量的维度，用于缩放点积                                                                                      |
| \( T_{turns} \)                                                              | 多轮对话数据中的总对话轮数 (LLaVA)                                                                                               |
| \( \mathcal{L} \)                                                            | 损失函数，模型优化的目标 (如 \( \mathcal{L}_{ITC}, \mathcal{L}_{ITM}, \mathcal{L}_{LM}, \mathcal{L}_{CLIP}, \mathcal{L}_{siglip} \)) |
| \( \tau \)                                                                   | 可学习参数，如对比损失中的温度或强化学习中的 KL 正则化权重                                                                           |
| \( \lambda \)                                                                | 超参数，如不同损失项的权重或强化学习中的长度调节因子                                                                                 |
| \( y \)                                                                      | 目标标签或类别 (如 ITM 损失)；或模型生成的最终答案 (如 Kimi-VL RL)                                                                 |
| \( x \)                                                                      | 输入数据、上下文或问题                                                                                                             |
| \( z \)                                                                      | 模型生成的中间推理步骤或思维链                                                                                  |
| \( y^* \)                                                                    | 参考答案或基准答案 (Ground Truth)                                                                                                |
| $\operatorname{sim}(u, v) = s(u, v)$                                                      | 向量 \( u \) 和 \( v \) 之间的相似度计算，通常是余弦相似度                                                                         |
| \( \mathbb{E} \)                                                             | 数学期望                                                                                                                         |
| KL                                                                           | KL 散度 (Kullback–Leibler Divergence)，用于衡量两个概率分布的差异                                                                    |
| \( \pi_{\theta} \)                                                           | 策略模型，根据参数 \( \theta \) 输出动作或文本序列                                                                                  |
| \( r \)                                                                      | 奖励函数，评估生成结果的好坏                                                                                                       |

## 多模态基础知识

在深入探讨具体技术之前，我们先来了解一些多模态的基础概念。

### 什么是多模态？

**多模态 (Multimodality)** 指的是使用多种不同类型的数据或信息通道（模态）来表示和处理信息。人类天生就是多模态的生物，我们通过**视觉、听觉、触觉、嗅觉和味觉**感知和理解世界。在人工智能领域，多模态学习为了构建能够处理和关联来自不同模态（如文本、图像、视频、音频等）信息的模型。

{{< figure
    src="multimodality_data.png"
    caption="Fig. 1. Multimodality Data. (Image source: [GPT-4o Image Generation](https://chatgpt.com/s/m_6814c5d31e288191a5409a7420ee30f4))"
    align="center"
    width="60%"
>}}

**常见模态：**
*   **文本 (Text):** 自然语言文字，是信息传递和知识表达的主要方式。
*   **图像 (Image):** 静态视觉信息，包含丰富的场景、物体和纹理细节。
*   **视频 (Video):** 动态视觉信息，由连续的图像帧组成，通常伴随音频。视频不仅包含空间信息，还包含时间信息。
*   **音频 (Audio):** 声音信息，包括语音、音乐和环境声音。
*   **其他:** 表格数据、[3D 点云](https://en.wikipedia.org/wiki/Point_cloud)、传感器数据（如雷达、激光雷达）、生物信号（如 [EEG](https://en.wikipedia.org/wiki/Electroencephalography)、[ECG](https://en.wikipedia.org/wiki/Electrocardiography)）等。

### 为什么需要多模态 AI？

1.  **更全面的世界理解:** 现实世界是多模态的。单一模态往往只能提供片面的信息。例如，仅凭文字描述可能难以完全理解一个复杂的场景，而结合图像或视频则能提供更直观、丰富的信息。多模态模型能够整合来自不同来源的信息，形成更全面、准确的理解。
2.  **增强的任务性能:** 在许多任务中，结合多种模态的信息可以显著提升性能。例如，在视觉问答（VQA）中，模型需要同时理解图像内容和文本问题才能给出正确答案。在视频描述生成中，结合视觉帧和音频信息可以生成更生动、准确的描述。
3.  **更自然的交互方式:** 多模态 AI 使得人机交互更加自然和灵活。用户可以通过语音、文字、图像等多种方式与 AI 系统交互，AI 系统也能以多种模态（如生成带有图片的文本回复，或生成语音回答）进行响应。
4.  **解锁新应用场景:** 多模态能力催生了许多新的应用，如自动驾驶（融合摄像头、雷达、激光雷达数据）、医疗诊断（结合医学影像和病历文本）、内容创作（文生图、文生视频）、虚拟助手、机器人交互等。
5.  **促进可访问性:** 多模态技术可以帮助有感官障碍的人士。例如，图像描述可以帮助视障人士理解图片内容，语音识别和合成可以帮助听障或语障人士交流。

### 常见多模态任务

以下表格列举了一些常见的多模态任务，这些任务通常需要结合多种模态的信息进行处理和生成。

| 任务名称 | 说明 |
| :------------------------------------- | :--------------------------------------------------------- |
| [视觉问答 (VQA)](https://paperswithcode.com/task/visual-question-answering) | 根据图像和相关问题生成文本答案。 |
| [图像/视频描述生成 (Image/Video Captioning)](https://paperswithcode.com/task/image-captioning) | 为图像或视频生成自然语言文字描述。 |
| [文本到多模态生成 (Text-to-X Generation)](https://paperswithcode.com/task/text-to-image-generation) | 根据文本描述生成相应的图像、视频或音频内容。 |
| [跨模态检索 (Cross-Modal Retrieval)](https://paperswithcode.com/task/cross-modal-retrieval) | 使用一种模态（如文本）查询另一种模态（如图像）的相关数据。 |
| [多模态情感分析 (Multimodal Sentiment)](https://paperswithcode.com/task/multimodal-sentiment-analysis) | 结合文本、音频、视频等多种信息判断情感倾向。 |
| [视觉推理 (Visual Reasoning)](https://paperswithcode.com/task/visual-reasoning) | 基于图像或视频内容进行逻辑判断与关系推理。 |
| [视觉语言导航 (VLN)](https://paperswithcode.com/task/vision-language-navigation) | 根据自然语言指令在视觉环境中指导智能体导航。 |
| [多模态机器翻译 (MMT)](https://paperswithcode.com/task/multimodal-machine-translation) | 利用相关图像信息辅助文本翻译以消除歧义。 |
| [音视频语音识别 (AVSR)](https://paperswithcode.com/task/audio-visual-speech-recognition) | 结合音频信号和说话者唇动视觉信息进行语音识别。 |
| [视觉定位 (Visual Grounding)](https://paperswithcode.com/task/visual-grounding) | 将文本中的词语或短语与图像或视频中的对应区域或物体关联起来。 |


## 关键技术

多模态大模型的发展由一系列技术推动。下图直观展示了多模态理解和生成的相关技术，博主介绍其中的一些关键模型和方法。

{{< figure
    src="MLLMs_arch.png"
    caption="Fig. 2. The general model architecture of MM-LLMs and the implementation choices for each component. (Image source: [Zhang et al., 2024](https://arxiv.org/pdf/2401.13601))"
    align="center"
    width="100%"
>}}

### Vision Transformer (ViT)

**Vision Transformer (ViT)** ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) 将 Transformer 架构成功应用于计算机视觉领域，成为当前众多先进 MLLMs 的首选视觉编码器。

{{< figure
    src="vit_overview.png"
    caption="Fig. 3. ViT model overview. (Image source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))"
    align="center"
    width="100%"
>}}

**核心思想:** ViT 将图像视为一系列 **图像块 (Patches)** 的序列，然后利用 Transformer 的自注意力机制来处理这些图像块，从而捕捉全局依赖关系。

**工作流程:**

1.  **图像分块:** 将输入图像 \( I \in \mathbb{R}^{H \times W \times C} \) 分割成 \( N_{patches} \) 个固定大小的非重叠图像块 \( x_p \in \mathbb{R}^{P^2 \times C} \)，其中 \( (H, W) \) 是图像分辨率，\( C \) 是通道数，\( P \) 是每个图像块的大小，\( N_{patches} = HW/P^2 \) 是图像块的数量。
2.  **线性投射:** 将每个图像块 \( x_p \) 展平成一维向量，并通过一个可学习的线性投射矩阵 \( E \) 将其映射到 \( D \) 维的嵌入空间，得到图像块嵌入 \( z_p = x_p E \)。
3.  **位置编码:** 为了保留图像块的空间位置信息，ViT 在图像块嵌入的基础上加入了可学习的位置编码 \( E_{pos} \)。
    \[ z_0 = [x_{class}; z_p^1; z_p^2; \dots; z_p^{N_{patches}}] + E_{pos}, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N_{patches}+1) \times D} \]
    通常还会添加一个可学习的 `[class]` 标记嵌入 \( x_{class} \)，其在 Transformer 输出端的对应向量用于图像分类任务。
4.  **Transformer 编码器:** 将添加了位置编码的图像块嵌入序列输入到标准的 Transformer 编码器中。编码器由多层 **多头自注意力 (Multi-Head Self-Attention, MSA)** 和 **前馈网络 (Feed Forward Network, FFN)** 组成。
    *   **MSA:** 捕捉图像块之间的全局依赖关系。对于输入序列 \( Z_{l-1} \)，自注意力计算如下：
        \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
        其中 \( Q = Z_{l-1}W_Q, K = Z_{l-1}W_K, V = Z_{l-1}W_V \) 是查询、键、值矩阵，\( d_k \) 是键向量的维度。多头注意力将 \( Q, K, V \) 拆分成多个头并行计算注意力，然后拼接结果。
    *   **FFN:** 通常由两个线性层和一个非线性激活函数（如 GELU）组成。
    每一层的计算可以表示为：
    \[ Z'_l = \text{MSA}(\text{LN}(Z_{l-1})) + Z_{l-1} \]
    \[ Z_l = \text{FFN}(\text{LN}(Z'_l)) + Z'_l \]
    其中 LN 表示层归一化。
5.  **输出:** Transformer 编码器的输出 \( Z_L \) 即为图像的特征表示 \( V \)。

{{< figure
    src="vit_bit_hybrid_compare.png"
    caption="Fig. 4. Performance versus pre-training compute for different architectures: Vision Transformers, ResNets, and hybrids. Vision Transformers generally outperform ResNets with the same computational budget. Hybrids improve upon pure Transformers for smaller model sizes, but the gap vanishes for larger models. (Image source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))"
    align="center"
    width="100%"
>}}

ViT 相比于传统的 CNN 具有以下优势：

1.  **全局依赖建模** :自注意力直接连接任意两 patch，可显式捕捉长距离空间关系，比传统 CNN 更擅长整合整幅图像的语义信息。
2.  **大规模预训练迁移能力强** : 在诸如 JFT-300M、ImageNet-22K 等超大数据集上预训练后，可轻松迁移到分类、检测、分割等 20 多种下游任务，性能随模型/数据规模几乎线性提升。
3.  **架构简洁、易于扩展与并行**: 纯 Transformer 堆叠便于按深度、宽度和输入分辨率三维扩展；计算由矩阵乘与 Softmax 组成，天然适配 GPU/TPU 的大批量并行和混合精度训练。

随着研究的深入，ViT 本身也在不断优化，以适应 MLLMs 的需求：

1. **原生动态分辨率:** 传统 ViT 通常需要固定输入分辨率。Qwen2-VL 和 Kimi-VL 等模型引入了动态分辨率处理能力。它们通常去除 ViT 中的绝对位置编码，转而使用 2D 旋转位置编码来编码二维空间信息。这使得模型能够处理任意分辨率和长宽比的图像，并将其转换为变长的视觉 token 序列，更好地保留细节信息。Kimi-VL 的 MoonViT 还借鉴了 NaViT 的图像打包技术，将不同分辨率的图像块序列打包输入 Transformer，提高了训练效率。
2. **窗口注意力:** 为了降低处理高分辨率图像时自注意力机制带来的二次方计算复杂度，Qwen2.5-VL 在其 ViT 的大部分层中采用了窗口注意力。注意力计算被限制在局部窗口内，使得计算复杂度与图像块数量成线性关系，显著提升了效率，同时通过少数几层全注意力层来保持全局信息的交互。
3. **架构对齐 LLM:** Qwen2.5-VL 和 Kimi-VL 等模型还对其 ViT 架构进行了微调，使其更接近 LLM 的设计，例如使用 RMSNorm 进行归一化，使用 SwiGLU 作为激活函数，以提升计算效率和模态间的兼容性。

### CLIP

**CLIP (Contrastive Language-Image Pre-training)** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) 是多模态领域具有里程碑意义的工作，它提出了一种简单而高效的方法来学习图像和文本之间的关联，为后续许多 MLLMs 奠定了基础。

**核心思想:** CLIP 的目标是学习一个 **多模态嵌入空间 (Multimodal Embedding Space)**，使得在该空间中，匹配的图像和文本对具有高相似度，而不匹配的对具有低相似度。它通过 **对比学习 (Contrastive Learning)** 的方式，利用自然语言监督来实现这一目标。

**架构:** CLIP 包含两个主要部分：

1.  **图像编码器:** 可以是 ResNet 或 ViT，负责将输入图像 \( I \) 编码为图像特征 \( V \)。
2.  **文本编码器:** 通常是 Transformer，负责将输入文本 \( T \) 编码为文本特征 \( L \)。
3.  **线性投射层:** 分别将图像特征 \( V \) 和文本特征 \( L \) 投射到共享的多模态嵌入空间，得到 \( I_e = V W_i \) 和 \( T_e = L W_t \)，其中 \( W_i \) 和 \( W_t \) 是可学习的投射矩阵。

{{< figure
    src="clip.png"
    caption="Fig. 5. CLIP Architecture Overview. CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset's classes. (Image source: [Radford et al., 2021](https://arxiv.org/abs/2103.00020))"
    align="center"
    width="100%"
>}}

**训练数据:** CLIP 的成功很大程度上归功于其大规模的预训练数据集 **WIT (WebImageText)**。研究团队从互联网上收集了 4 亿个 (图像, 文本) 对。他们通过搜索约 50 万个查询词（源自维基百科词汇、高频二元组、维基百科文章标题和 WordNet 同义词集）来构建数据集，并对每个查询词限制最多 2 万个样本以平衡数据分布。这种利用网络原生图文对的方式被称为 **自然语言监督**，它避免了昂贵的人工标注，使得数据规模可以轻松扩展。

**对比损失:** CLIP 的核心是对比学习目标。给定一个包含 \( N \) 个 (图像, 文本) 对的批次 \( \{(I_1, T_1), \dots, (I_N, T_N)\} \)，模型的目标是预测 \( N \times N \) 个可能的配对中哪些是真实的配对。

1.  计算所有图像嵌入 \( \{I_{e,1}, \dots, I_{e,N}\} \) 和文本嵌入 \( \{T_{e,1}, \dots, T_{e,N}\} \)。通常会进行 **L2 归一化** 把每个图像或文本嵌入除以它自己的 L2 范数（Euclidean norm）。
2.  计算所有 \( N^2 \) 对 \( (I_{e,i}, T_{e,j}) \) 之间的 **余弦相似度**。
    \[ \text{logits}_{i,j} = \text{sim}(I_{e,i}, T_{e,j}) \cdot \exp(\tau) = \frac{I_{e,i} \cdot T_{e,j}}{\|I_{e,i}\| \|T_{e,j}\|} \cdot \exp(\tau) \]
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
# t - learned temperature parameter (tau in text)

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
# l2_normalize projects the embeddings onto the unit hypersphere
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
# The temperature parameter t scales the logits
# Note: using dot product on normalized vectors is equivalent to cosine similarity
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
# labels are the indices [0, 1, ..., n-1] indicating the correct pairings
labels = np.arange(n)
# Calculate cross-entropy loss for image-to-text classification
# (Predict correct text for each image)
loss_i = cross_entropy_loss(logits, labels, axis=1) # axis=1 for softmax over columns
# Calculate cross-entropy loss for text-to-image classification
# (Predict correct image for each text)
loss_t = cross_entropy_loss(logits, labels, axis=0) # axis=0 for softmax over rows
# Final loss is the average of the two losses
loss = (loss_i + loss_t)/2
```

{{< /collapse >}}

**零样本迁移:** CLIP 强大的能力在于其零样本迁移性能。对于一个新的图像分类任务，无需任何微调，CLIP 可以通过以下方式进行预测：

1.  获取任务的所有类别名称（例如，“猫”，“狗”）。
2.  使用 **提示词工程 (Prompt Engineering)** 将类别名称构造成句子，如 "A photo of a {label}."。这有助于弥合预训练数据（通常是句子）和下游任务（通常是单词标签）之间的分布差距。CLIP 论文发现使用提示模板和集成多个提示可以显著提高性能（在 ImageNet 上提升近 5%）。
3.  使用 CLIP 的文本编码器计算每个构造句子的文本嵌入，这些嵌入构成了零样本分类器的 **权重向量**。
4.  对于一张新的待分类图像，使用 CLIP 的图像编码器计算其图像嵌入。
5.  计算该图像嵌入与所有类别文本嵌入之间的余弦相似度。
6.  将相似度最高的类别作为预测结果。

{{< figure
    src="clip_prompt_engineering.png"
    caption="Fig. 6. Prompt engineering and ensembling improve zero-shot performance. Compared to the baseline of using contextless class names, prompt engineering and ensembling boost zero-shot classification performance by almost 5 points on average across 36 datasets. This improvement is similar to the gain from using 4 times more compute with the baseline zero-shot method but is 'free' when amortized over many predictions. (Image source: [Radford et al., 2021](https://arxiv.org/abs/2103.00020))"
    align="center"
    width="70%"
>}}

**CLIP 的影响:** CLIP 证明了通过大规模自然语言监督和对比学习可以学到强大的、可迁移的视觉表示。其学习到的多模态嵌入空间和强大的图像编码器被广泛应用于后续的 MLLMs（如 Flamingo, BLIP-2, LLaVA）以及文生图模型（如 DALL-E 2, Stable Diffusion）中。

CLIP 主要关注学习对齐的表示，但在生成任务上能力有限。后续工作开始探索能够同时进行理解和生成的统一模型架构。



### BLIP

**BLIP (Bootstrapping Language-Image Pre-training)** ([Li et al., 2022](https://arxiv.org/abs/2201.12086)) 为了解决现有**视觉语言预训练（Vision-Language Pre-training, VLP）** 方法在模型和数据方面的局限性：模型常只能擅长理解或生成之一；数据则依赖于海量且噪声较大的网络图文对。


BLIP 提出了**多模态编码器-解码器（Multimodal Encoder-Decoder, MED）** 架构，旨在统一理解和生成任务。它结合了 CLIP 的对比学习和自回归生成的优点，能够处理多种模态数据。

{{< figure
    src="blip_model_architecture.png"
    caption="Fig. 7. BLIP Pre-training Model Architecture and Objectives (same parameters have the same color). We propose multimodal mixture of encoder-decoder (MED), a unified vision-language model which can operate in one of the three functionalities. (Image source: [Li et al., 2022](https://arxiv.org/abs/2201.12086))"
    align="center"
    width="100%"
>}}

*   **图像编码器:** 采用 ViT。
*   **文本编码器/解码器:** 基于 BERT 架构，但进行了修改以适应多模态任务和不同功能模式。
    *   **单模态编码器:** 标准的 ViT 和 BERT，分别处理图像和文本。
    *   **基于图像的文本生成编码器:** 在文本编码器的每个 Transformer 块的自注意力 (SA) 层和前馈网络 (FFN) 层之间插入 **交叉注意力 (Cross-Attention, CA)** 层，用于注入视觉信息。文本输入前会添加 [Encode] 标记，其输出嵌入作为图文对的多模态表示。
    *   **基于图像的文本生成解码器:** 将编码器中的双向 SA 层替换为 **因果自注意力 (Causal Self-Attention)** 层，以实现自回归生成。共享编码器的 CA 层和 FFN 层。使用 [Decode] 标记作为序列开始符。

**预训练目标:** BLIP 联合优化三个目标：

1.  **图文对比(Image-Text Contrastive, ITC)损失:** 类似于 CLIP，使用单模态编码器对齐图像和文本的特征空间。BLIP 采用了**ALBEF**([Li et al., 2021](https://arxiv.org/abs/2107.07651))提出的动量编码器 (Momentum Encoder) 和软标签策略来改进对比学习。
    $$L_{ITC} = \frac{1}{2N} \sum_{i=1}^{N} \left( -\log \frac{\exp(s(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(v_i, t_j)/\tau)} -\log \frac{\exp(s(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(v_j, t_i)/\tau)} \right)$$
    其中 $v_i, t_j$ 为图文特征，$s$ 为相似度函数，$\tau$ 为温度参数。

2.  **图文匹配(Image-Text Matching, ITM)损失:** 使用图像接地文本编码器学习细粒度的图文对齐。这是一个二分类任务，预测图文对是匹配还是不匹配。采用难负例挖掘策略。
    $$L_{ITM} = -\mathbb{E}_{(I,T)\sim D} [y \log p_{match} + (1-y) \log(1 - p_{match})]$$
    其中 $y$ 是标签，$p_{match}$ 是匹配概率。

3.  **语言模型(Language Modeling, LM)损失:** 使用图像接地文本解码器，根据图像生成文本描述。采用标准的交叉熵损失（带标签平滑）。

    $$L_{L M}=-\mathbb{E}_{(I, T) \sim D} \sum_{k=1}^L \log P\left(w_k \mid I, w_{\lt k} ; \theta\right)$$
    其中 $w_k$ 是文本序列中的词，$\theta$ 是模型参数。
    

**总损失函数:** 这三个损失通常被联合优化（例如，等权重相加）：
$$L_{BLIP} = L_{ITC} + L_{ITM} + L_{LM}$$

**参数共享:** 为了效率和多任务学习的好处，文本编码器和解码器共享除 SA 层外的所有参数（嵌入层、CA 层、FFN 层）。


**CapFilt (Captioning and Filtering)** 是一种创新的数据集引导方法，用于从未标注的网络图像中生成高质量的合成标题，并过滤掉噪声数据（包括原始网络文本和合成文本）。

{{< figure
    src="blip_learning_framework.png"
    caption="Fig. 8. BLIP Learning Framework. We introduce a captioner to produce synthetic captions for web images, and a filter to remove noisy image-text pairs. (Image source: [Li et al., 2022](https://arxiv.org/abs/2201.12086))"
    align="center"
    width="100%"
>}}

1.  **初始化:** 使用预训练好的 MED 模型初始化两个模块：Captioner（图像接地文本解码器）和 Filter（图像接地文本编码器）。
2.  **微调:** 在高质量的人工标注数据集（如 COCO）上分别微调 Captioner (使用 LM 损失) 和 Filter (使用 ITC 和 ITM 损失)。这是一个轻量级过程。
3.  **生成与过滤:**
    *   Captioner 为网络图像 \( I_w \) 生成合成标题 \( T_s \)。
    *   Filter 判断原始网络文本 \( T_w \) 和合成文本 \( T_s \) 是否与图像 \( I_w \) 匹配。预测为不匹配的文本被视为噪声并移除。
4.  **引导数据集:** 将过滤后的高质量图文对（来自原始网络数据和合成数据）与人工标注数据结合，形成新的引导数据集。
5.  **重新预训练:** 使用引导数据集从头预训练一个新的 BLIP 模型。

**效果:** CapFilt 显著提升了模型在各项下游任务（如检索、描述生成、VQA）上的性能，证明了通过引导方式改善噪声数据质量的有效性。BLIP 也展示了统一模型在理解和生成任务上的灵活性。

### BLIP-2

**BLIP-2** ([Li et al., 2023](https://arxiv.org/abs/2301.12597)) 针对高昂 VLP 训练成本，提出**高效**预训练策略：冻结预训练图像编码器与大语言模型，只训练轻量桥接模块 Q‑Former。

**核心贡献:**

1.  **利用冻结模型:** 无需端到端训练整个大型模型，显著降低了计算成本，并利用了强大的预训练单模态模型的能力。
2.  **Q-Former (Querying Transformer):** 提出了一种轻量级的 Transformer 结构作为可训练的桥梁，连接冻结的图像编码器和冻结的 LLM。
3.  **两阶段预训练:** 设计了两阶段策略来有效弥合模态鸿沟：
    *   **阶段一：视觉-语言表示学习 (Vision-Language Representation Learning):** 从冻结的图像编码器引导学习。
    *   **阶段二：视觉到语言生成学习 (Vision-to-Language Generative Learning):** 从冻结的 LLM 引导学习。

**架构 (Q-Former):**

*   Q-Former 是一个轻量级 Transformer，包含 188M 参数。
*   它使用一组 **可学习的查询向量**（例如 32 个 768 维向量）作为输入。
*   这些查询向量通过 **自注意力层** 相互交互。
*   通过 **交叉注意力层** 与冻结的图像编码器输出的图像特征进行交互，提取视觉信息。
*   查询向量的输出 \( Z \) (例如 \( 32 \times 768 \) 维) 维度远小于原始图像特征，充当了信息瓶颈，迫使 Q-Former 提取对语言模型最有用的视觉信息。
*   Q-Former 内部包含图像 Transformer 和文本 Transformer 两个共享自注意力层的子模块。

{{< figure
    src="blip2_stage1.png"
    caption="Fig. 9. (Left) Model architecture of Q-Former and BLIP-2's first-stage vision-language representation learning objectives. (Right) The self-attention masking strategy for each objective to control query-text interaction. (Image source: [Li et al., 2023](https://arxiv.org/abs/2301.12597))"
    align="center"
    width="100%"
>}}

**两阶段预训练:**

1.  **阶段一 (表示学习):**
    *   将 Q-Former 连接到 **冻结的图像编码器** (如 CLIP ViT-L/14, EVA-CLIP ViT-g/14)。
    *   使用图文对进行预训练，目标是让 Q-Former 的查询向量学会提取与文本最相关的视觉表示。
    *   联合优化三个与 BLIP 类似的目标 (共享输入格式和模型参数，但**冻结图像编码器，只训练 Q-Former**)：
        *   **图文对比(Image-Text Contrastive, ITC)损失:** 对齐 Q-Former 输出的查询表示 \( z \) 和文本表示 \( t \)。使用 In-batch Negatives。
            $$L_{ITC} = \frac{1}{2N} \sum_{i=1}^{N} \left( -\log \frac{\exp(s(z_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(z_i, t_j)/\tau)} -\log \frac{\exp(s(z_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(z_j, t_i)/\tau)} \right)$$
        *   **图文匹配(Image-Text Matching, ITM)损失:** 预测图文对是否匹配。使用 Q-Former 输出的多模态查询表示进行分类。
            $$L_{ITM} = -\mathbb{E}_{(I,T)\sim D} [y \log p_{match} + (1-y) \log(1 - p_{match})]$$
        *   **基于图像的文本生成(Image-grounded Text Generation, ITG)损失:** 训练 Q-Former 生成文本。查询向量需要捕获所有生成文本所需的信息，并通过自注意力层传递给文本 token。
            $$L_{ITG} = -\mathbb{E}_{(I,T)\sim D} \sum_{k=1}^{L} \log P(w_k | Z_q, w_{\lt k}; \theta_{Q-Former})$$
            其中 $Z_q$ 是 Q-Former 的查询输出。
    *   通过不同的自注意力掩码控制查询-文本交互来实现不同目标。
    *   **第一阶段总损失函数:**
        $$L_{Stage1} = L_{ITC} + L_{ITM} + L_{ITG}$$

2.  **阶段二 (生成学习):**
    *   将 **第一阶段预训练好的 Q-Former** (及其连接的冻结图像编码器) 连接到 **冻结的 LLM** (如 OPT 系列, FlanT5 系列)。
    *   使用一个 **全连接层** 将 Q-Former 的输出查询嵌入 \( Z \) 线性投射到与 LLM 文本嵌入相同的维度，得到软视觉提示 $P_Z$。
    *   将投射后的查询嵌入作为 **软视觉提示 (Soft Visual Prompts)**，添加到 LLM 输入文本嵌入的前面。
    *   **训练目标:** 训练 Q-Former (FC 层也训练)，使其输出的视觉表示能够被冻结的 LLM 理解并用于生成文本。
        *   对于 **Decoder-only LLM (如 OPT):** 使用标准的语言建模损失，即根据视觉提示生成后续文本。
        *   对于 **Encoder-Decoder LLM (如 FlanT5):** 使用前缀语言建模损失 (Prefix Language Modeling)，将文本分成前缀和后缀，视觉提示和前缀输入 Encoder，Decoder 生成后缀。
        $$L_{Stage2} = L_{LM} = -\mathbb{E}_{(I, T_{prompt}, T_{gen})\sim D} \sum_{k=1}^{M} \log P_{LLM}(w_k | P_Z, T_{prompt}, w_{\lt k}; \theta_{LLM\_frozen})$$
        其中 $\theta_{L L M_{-} \text {frozen }}$ 是冻结 LLM 的参数, 只用来前向传播，不参与梯度更新。

{{< figure
    src="blip2_stage2.png"
    caption="Fig. 10. BLIP-2's second-stage vision-to-language generative pre-training, which bootstraps from frozen large language models (LLMs). (Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5). (Image source: [Li et al., 2023](https://arxiv.org/abs/2301.12597))"
    align="center"
    width="90%"
>}}

**效果与优势:**

*   **高效:** 由于只训练轻量级的 Q-Former，预训练成本远低于端到端训练的大型模型。
*   **高性能:** 在 VQA、Captioning、Retrieval 等任务上达到 SOTA 水平，甚至超越了参数量远大于它的模型（如 Flamingo）。
*   **通用性:** 可以方便地接入不同的冻结图像编码器和 LLMs，利用各自领域的最新进展。
*   **零样本能力:** 借助强大的冻结 LLM（特别是指令微调过的 FlanT5），BLIP-2 展现出令人印象深刻的**零样本指令图像到文本生成**能力，可以根据自然语言指令执行各种视觉语言任务（如视觉对话、视觉知识推理）。


### LLaVA

**LLaVA (Large Language and Vision Assistant)** ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)) 是**视觉指令微调 (Visual Instruction Tuning)** 开源社区领域的重要工作，首次尝试将 NLP 领域的指令微调思想扩展到多模态领域。

**核心贡献:**

1.  **提出视觉指令微调:** 探索将指令微调应用于语言-图像多模态模型，旨在构建通用的视觉助手。
2.  **GPT 辅助数据生成:** 面对视觉指令数据的缺乏，创新性地使用**纯语言模型 GPT-4**来生成包含视觉内容的多模态语言-图像指令遵循数据。
3.  **构建 LLaVA 模型:** 提出了一种连接预训练的视觉编码器 (CLIP ViT-L/14) 和大型语言模型 (LLM, Vicuna) 的端到端训练架构。
4.  **创建评估基准:** 构建了 LLaVA-Bench，包含多样化和具有挑战性的任务，用于评估多模态模型的指令遵循能力。
5.  **开源贡献:** 公开了 GPT-4 生成的视觉指令数据、模型代码和预训练权重，极大地推动了社区在这一方向上的研究。

**GPT 辅助视觉指令数据生成:**

LLaVA 解决的关键挑战是缺乏大规模、高质量的视觉指令遵循数据。研究者提出了一种利用现有的多模态大模型如 GPT-4 基于现有的图像-文本对来生成此类数据的方法，本质上这是一种对闭源模型 GPT-4 进行**知识蒸馏**的过程。

1.  **面临的挑战:** 简单的将图像-标题对扩展为 (指令：描述图像，图像 -> 回答：标题) 的格式虽然廉价，但缺乏指令和响应的多样性及深度推理。
2.  **解决方案:** 使用 GPT-4 作为“教师模型”。由于这些模型仅接受文本输入，研究者将图像内容通过**符号表示** 传递给它们：
    * **图像描述:** 提供图像场景的整体或多方面描述。
    * **边界框:** 提供图像中对象的类别概念及其空间位置信息 (例如 `person: [0.681, 0.242, 0.774, 0.694]`)。
3.  **提示与上下文学习:** 将图像的符号表示 (描述和边界框) 输入给 GPT-4。为了引导 GPT-4 生成特定格式和内容的输出，研究者手动设计了少量高质量的**种子示例 **，利用 GPT-4 的**上下文学习** 能力进行 few-shot 推理。
4.  **生成三种类型数据 (基于 COCO 图像):** 通过精心设计的 Prompt 引导 GPT-4 生成了三种类型的指令数据：
    * **对话:** 生成模拟人与助手之间关于图像内容的多轮对话，包含物体识别、计数、定位、动作、关系等问题。
    * **详细描述:** 根据特定指令（如“详细描述下图”）生成对图像全面、细致的描述。
    * **复杂推理:** 生成需要基于图像内容进行逻辑推理或结合背景知识的问题和答案（如“图中人物可能面临什么挑战？”）。


{{< figure
    src="llava_instruction_data.png"
    caption="Fig. 11. One example to illustrate the instruction-following data. (Image source: [Liu et al., 2023](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}


5.  **数据集:** 共收集了 **158K** 个独特的语言-图像指令样本，具体包括：**58K** 对话样本，**23K** 详细描述样本，**77K** 复杂推理样本。实验发现，GPT-4 生成的数据质量通常优于 ChatGPT。

{{< figure
    src="llava_architecture.png"
    caption="Fig. 12. LLaVA network architecture. (Image source: [Liu et al., 2023](https://arxiv.org/abs/2304.08485))"
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

对于每张图像 \( \mathbf{X}_{\mathrm{v}} \)，生成包含 \( T_{turns} \) 轮的多轮对话数据 \( \left(\mathbf{X}_{\mathrm{q}}^{1}, \mathbf{X}_{\mathrm{a}}^{1}, \cdots, \mathbf{X}_{\mathrm{q}}^{T_{turns}}, \mathbf{X}_{\mathrm{a}}^{T_{turns}}\right) \)，其中 \( T_{turns} \) 是总对话轮数。我们将这些数据组织成一个序列，并将所有答案 \( \mathbf{X}_{\mathrm{a}} \) 视为模型的回应。其输入序列的组织形式采用了 Vicuna 格式。在第 \( t \) 轮对话中，指令 \( \mathbf{X}_{\text{instruct}}^{t} \) 定义为：


$$
\mathbf{X}_{\text{instruct}}^{t} = \left\{ \begin{array}{ll} \text{Randomly choose } [\mathbf{X}_{\mathrm{q}}^{1}, \mathbf{X}_{\mathrm{v}}] \text{or } [\mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\mathrm{q}}^{1}], & \text{ if } t=1 \text{ (the first turn)} \\ \mathbf{X}_{\mathrm{q}}^{t}, & \text{ if } t>1 \text{ (the remaining turns)} \end{array} \right.
$$

目标是预测答案序列 \( \mathbf{X}_{\mathrm{a}} = (\mathbf{X}_{\mathrm{a}}^{1}, \dots, \mathbf{X}_{\mathrm{a}}^{T_{turns}}) \)。模型需要最大化在给定图像 \( \mathbf{X}_{\mathrm{v}} \) 和所有指令 \( \mathbf{X}_{\text{instruct}} = (\mathbf{X}_{\text{instruct}}^{1}, \dots, \mathbf{X}_{\text{instruct}}^{T_{turns}}) \) 的条件下，生成正确答案序列的概率。对于长度为 \( L_{seq} \) 的完整答案序列（所有轮次的 \( \mathbf{X}_{\mathrm{a}} \) 拼接而成），其概率计算如下：

$$
p\left(\mathbf{X}_{\mathrm{a}} \mid \mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\text {instruct }}\right)=\prod_{i=1}^{L_{seq}} p_{\boldsymbol{\theta}}\left(x_i \mid \mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\text {instruct },\lt i}, \mathbf{X}_{\mathrm{a},\lt i}\right)
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
    caption="Fig. 13. Accuracy (%) on Science QA dataset. (Image source: [Liu et al., 2023](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}

LLaVA 的成功证明了视觉指令微调的有效性，其开源的数据、代码和模型极大地促进了后续多模态大模型的研究，为构建通用的、能够理解并遵循视觉和语言指令的 AI 助手开辟了新的途径。


### Qwen-VL

**Qwen-VL** ([Bai et al., 2023](https://arxiv.org/abs/2308.12966))模型是 Qwen 团队研发的首个开源大型视觉语言模型，其架构由三大模块组成：

* **大语言模型**：采用预训练的 Qwen-7B 文本模型作为语言解码器。这部分负责理解和生成文本，与标准的 LLM 架构一致。
* **视觉编码器**：使用 Vision Transformer 提取图像特征。具体实现上，Qwen-VL 利用采用 [OpenCLIP](https://github.com/mlfoundations/open_clip) 的 ViT-bigG 模型初始化视觉编码部分。在训练和推理阶段，输入图像都会被调整为特定分辨率。视觉编码器通过以 14 的步幅将图像切分为多个图像块，从而提取出一组图像特征。

* **位置感知视觉-语言适配器(Position-aware Vision-Language Adapter)**：为高效融合长序列图像特征，引入了一个适配器将视觉特征序列压缩至固定长度。具体而言，该适配器包含一组随机初始化的**可学习查询向量**，通过单层的**交叉注意力** 模块与 ViT 输出的图像特征进行计算，将图像特征压缩为长度固定为256的序列。

注意力计算公式如下：

$$
\text{CrossAttn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中，\(Q\) 为适配器内部定义的可训练查询向量矩阵，而\(K, V\) 均直接使用视觉编码器（ViT）输出的图像特征序列作为键（Key）和值（Value）。

通过这一机制，适配器能够根据学习到的查询向量从众多图像特征中选择并聚合最相关的信息。此外，为缓解图像特征压缩过程中可能引发的空间位置信息损失，在注意力计算的查询-键对中额外融入了**二维绝对位置编码**，强化了对图像空间结构的感知能力。

{{< figure
    src="qwen_vl_pipeline.png"
    caption="Fig. 14. The training pipeline of the Qwen-VL series. (Image source: [Bai et al., 2023](https://arxiv.org/abs/2308.12966))"
    align="center"
    width="100%"
>}}

Qwen-VL 采用“三阶段” 逐步训练策略，将视觉感知能力注入通用大模型。第一阶段冻结 LLM 仅训练视觉模块，第二阶段解冻联合多任务训练，第三阶段指令微调得到聊天模型。上图中雪花 ❄ 表示冻结，火焰 🔥 表示参与训练。

**训练策略：** Qwen-VL 系列采用**分三阶段**的逐步训练流程：

1. **纯图文预训练阶段**：
   * 固定语言模型（7B）参数，仅训练视觉编码器和 VL 适配器；
   * 使用约14亿对弱标注图文数据（英文占77.3%、中文占22.7%）；
   * 图像统一缩放至较低分辨率（如 224×224）以提高效率；
   * 采用自回归方式进行语言建模，训练模型生成图像描述文本；
   * 训练约5万步（15亿样本）后，初步实现图文对齐能力（Qwen-VL）。

2. **多任务联合训练阶段**：
   * 解冻语言模型，与视觉部分端到端共同训练；
   * 提升输入图像分辨率（如448×448以上）；
   * 加入多种细粒度视觉任务（如图像描述、视觉问答、内容定位、OCR 识别等），共涉及7大类任务；
   * 训练数据混合多来源数据集，并加入约 2480 万条 OCR 数据和 780 万条纯文本数据；
   * 所有任务数据随机混合训练，每条样本带任务前缀并填充至 2048 序列长度；
   * 模型显著提升图像理解、跨模态检索、定位、阅读等能力。

3. **监督微调（SFT）阶段**：
   * 在多模态指令数据（约35万条）上进行微调，得到对话增强版 Qwen-VL-Chat；
   * 特别设计复杂的多图推理、细粒度定位、多轮交互任务数据；
   * 微调期间再次冻结视觉编码器，仅微调语言模型和适配器；
   * 最终模型表现出优异的多模态对话、指令跟随和复杂推理能力。


### Qwen2-VL

**Qwen2-VL** ([Wang et al., 2024](https://arxiv.org/abs/2409.12191)) 是 Qwen-VL 的升级版，在处理可变分辨率视觉输入和融合多模态位置信息方面取得了进展。

{{< figure
    src="qwen2_vl.jpg"
    caption="Fig. 15. Qwen2-VL is capable of accurately identifying and comprehending the content within images, regardless of their clarity, resolution, or extreme aspect ratios.: ([Wang et al., 2024](https://arxiv.org/abs/2409.12191))"
    align="center"
    width="100%"
>}}

从上图我们可以看出，Qwen2-VL 在处理不同分辨率和长宽比的图像时，能够准确识别和理解图像中的内容。主要采用了以下技术：

*   **原生动态分辨率 (Naive Dynamic Resolution):** 借鉴 **NaViT** ([Dehghani et al., 2023](https://arxiv.org/abs/2307.06304))，模型能够处理任意分辨率的图像，并将其动态地转换为变长的视觉 token 序列。
    *   移除 ViT 的绝对位置编码，引入 **2D 旋转位置编码 (2D Rotary Position Embedding, 2D-RoPE)** ([Su et al., 2024](https://arxiv.org/abs/2104.09864); [Su, 2021](https://spaces.ac.cn/archives/8397)) 来编码二维空间信息。
    *   推理时，可变分辨率图像被打包处理，限制总 token 长度以控制显存。
    *   ViT 输出后，使用 MLP 压缩相邻 \( 2 \times 2 \) 的 token 为一个，减少输入 LLM 的序列长度。使用 `<|vision_start|>` 和 `<|vision_end|>` 包裹视觉 token。

*   **多模态旋转位置编码 (Multimodal Rotary Position Embedding, M-RoPE):** 提出了一种新的位置编码方法，可以统一处理文本、图像和视频的位置信息。
    *   将 RoPE 分解为 **时间 (Temporal)**、**高度 (Height)**、**宽度 (Width)** 三个分量。
    *   **文本:** 三个分量使用相同的位置 ID，等价于 1D-RoPE。
    *   **图像:** 时间 ID 恒定，高度和宽度 ID 根据 token 在图像中的二维位置赋值。
    *   **视频:** 时间 ID 随帧数递增，高度和宽度 ID 同图像。
    *   **多模态输入:** 不同模态的位置 ID 依次递增。
    *   **优势:** 统一编码多模态位置信息，降低了图像/视频的位置 ID 值，有利于推理时外插到更长序列。

{{< figure
    src="mrope.png"
    caption="Fig. 16. Illustration of M-RoPE. By decomposing rotary embedding into temporal, height, and width components, M-RoPE can explicitly model the positional information of text, images, and video in LLM. (Image source: [Wang et al., 2024](https://arxiv.org/abs/2409.12191))"
    align="center"
    width="100%"
>}}

*   **统一图像与视频理解:** 采用混合训练范式和特定架构设计（如 3D 卷积处理视频）来同时处理图像和视频。
    *   混合图像和视频数据进行训练。
    *   视频以 2 FPS 采样。
    *   ViT 中集成 **3D 卷积** 处理视频输入 (处理 \( 2 \times 14 \times 14 \) 的 3D 块)，减少 token 数量。
    *   图像被视为两帧相同的视频帧。
    *   动态调整视频帧分辨率，限制每段视频的总 token 数（如 16384）。

**训练:** 沿用 Qwen-VL 的三阶段训练：ViT 预训练 -> 全模型预训练 -> LLM 指令微调。预训练数据包含图文对、OCR、图文交错文章、VQA、视频对话、图像知识等。指令微调使用 ChatML 格式。发布了 2B, 8B, 72B 三种规模的模型，探索了 MLLMs 的 scaling law。

**效果:** Qwen2-VL 在多种分辨率和长宽比的图像理解、长视频理解（超过 20 分钟）以及视觉 Agent 能力方面表现出色。

### Qwen2.5-VL

**Qwen2.5-VL** ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)) 在 Qwen2-VL 的基础上进一步优化了效率和时序建模能力。

{{< figure
    src="qwen2.5vl_arc.jpeg"
    caption="Fig. 17. The Qwen2.5-VL framework demonstrates the integration of a vision encoder and a language model decoder to process multimodal inputs. The vision encoder is designed to handle inputs at their native resolution and supports dynamic FPS sampling. TMRoPE aligns time IDs with absolute time along the temporal dimension. (Image source: [Bai et al., 2025](https://arxiv.org/abs/2502.13923))"
    align="center"
    width="100%"
>}}

**模型优化：**

Qwen2.5-VL 在 Qwen2-VL 的基础上进行了多项优化，主要包括：

1. **高效 ViT 架构：** 在 Vision Transformer 中引入**窗口注意力(Window Attention)** 机制，将大部分层的注意力计算限制在局部窗口（如 $8 \times 8$ patch），使得计算复杂度随图像块数量呈线性增长，显著提升对高分辨率图像的处理效率。同时，仅在少数层（如每隔 8 层）执行全局注意力以保留整体上下文信息。

2. **动态 FPS 采样与视频处理：** 引入**动态帧率（Dynamic FPS）采样**机制，将动态分辨率思想拓展至时间维度，提升模型对不同速率视频的适应能力。在视频处理上，保持3D 块结构（$2 \times 14 \times 14$）设计，并结合动态 FPS 和时间感知编码优化整体时序建模效果。

3. **更强的数据与任务能力支持：** 模型在大规模（4.1T tokens）、高质量数据集上进行预训练与微调，重点提升了**文档解析**（表格、图表、公式、乐谱等）、**对象定位**（支持点和框标注）、**长视频理解（小时级）**以及**Agent 多任务能力**，拓宽了多模态理解的应用边界。

**数据增强:**
*   **文档全解析数据:** 构建了包含表格、图表、公式、图片、乐谱、化学式的 HTML 格式数据，包含布局框信息和坐标。
*   **定位数据:** 扩展了边界框和点的定位数据，覆盖超过 1 万个类别，并合成了包含不存在对象和多实例对象的难例。使用了 Grounding DINO 和 SAM 等工具合成数据。
*   **OCR 数据:** 增加了多语言 OCR 数据（覆盖欧洲主要语言及日韩阿越等），并包含手写体、密集文本、网页、公式、图表、表格等多种场景。
*   **视频数据:** 增加了长视频（超过半小时）的密集描述数据，并采用动态 FPS 采样训练。时间戳标注包含秒和 HMSF 两种格式。
*   **Agent 数据:** 收集了移动端、Web 端、桌面端的截图和操作轨迹，统一为函数调用格式，并合成了 CoT 推理过程。

**效果:** Qwen2.5-VL 在文档理解、细粒度定位、长视频理解和 Agent 任务上取得了 SOTA 性能，72B 版本在多个基准上媲美甚至超越 GPT-4o 和 Claude 3.5 Sonnet。

### Qwen2.5-Omni

{{< figure
    src="qwen2.5_omni.png"
    caption="Fig. 18. Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="100%"
>}}

**Qwen2.5-Omni** ([Qwen Team, 2025](https://arxiv.org/abs/2504.14786)) 是一个类似于 GPT-4o([OpenAI, 2024](https://openai.com/index/hello-gpt-4o/)) 的端到端多模态模型，支持处理包括文本、图像、音频和视频全模态的输入，并能同时 **流式生成文本和自然语音** 输出。

从下图可以看出，Qwen2.5-Omni 采用了**Thinker-Talker**架构，其主要特点包括：

{{< figure
    src="qwen2.5_omini_arc.png"
    caption="Fig. 19. Qwen2.5-Omni Overview. Adopts Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receiving high-level representations directly from Thinker. (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="80%"
>}}

1. **统一多模态处理与时序建模：**

   * **全模态感知:** 单一模型能够同时处理文本、图像、音频、视频四种模态输入，实现多模态统一理解。

    {{< figure
    src="TMRoPE.png"
    caption="Fig. 20. An illustration of Time-aligned Multimodal RoPE (TMRoPE). (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="100%"
    >}}

   * **时序对齐多模态旋转位置编码（Time-aligned Multimodal RoPE，TMRoPE）:** 在Qwen2.5-VL基础上进一步优化 TMRoPE，通过**时间交错 (Time-interleaving)** 结构，将视频帧和音频帧每2秒切块后按时间顺序排列，块内先视频后音频。所有模态使用绝对时间戳（40ms粒度）与位置编码（TMRoPE）对齐，实现精准的音视频同步。

   * **输入处理细节:** 文本使用Qwen tokenizer；音频为16kHz采样、128通道梅尔频谱图（25ms窗长，10ms步长），每帧约40ms，通过Qwen2-Audio编码器处理；图像/视频通过Qwen2.5-VL的ViT架构处理，视频支持动态FPS采样。

2. **Thinker-Talker 架构设计与功能解耦：**

   * 提出创新的Thinker-Talker架构，将文本生成和语音生成解耦，避免相互干扰，同时允许端到端联合训练。
   * **Thinker:** 基于Qwen2.5的Transformer解码器，处理多模态输入，生成高级隐层表示（包含语义和韵律信息）及文本token输出。
   * **Talker:** 双轨自回归Transformer解码器，接收Thinker输出的隐层表示和文本token，结合消除语音歧义的能力，自回归地生成离散语音token。
   * Thinker与Talker共享历史上下文，支持端到端训练，提升语音生成一致性与上下文保持能力。

3. **高效流式处理能力：**

   * **输入流式处理:** 音频和视觉编码器采用**分块处理 (Block-wise Processing)**，支持流式输入及预填充 (Prefilling)。
   * **输出流式处理:**

     * Talker生成的离散语音token实时送入**流式音频解码器 (Streaming Audio Codec)**。
     * 解码器采用基于**Diffusion Transformer (DiT)** 的 **滑动窗口块注意力 (Sliding Window Block Attention)**（回看2个块，前看1个块），控制感受野，实现流式生成。
     * 使用 **Flow Matching** ([Lipman et al., 2022](https://arxiv.org/abs/2210.02747)) 将离散 token 转换为梅尔频谱图，再通过改进版 **BigVGAN**([Lee et al., 2022](https://arxiv.org/abs/2206.04658)) 将频谱图流式转换为音频波形，有效降低首包延迟，提升生成实时性。

**训练:** 包含三个阶段：编码器与 LLM 对齐 -> 全模型多模态预训练 -> 长上下文预训练 (32k)。Talker 单独进行三阶段训练：上下文学习 -> DPO (优化稳定性) -> 多说话人指令微调 (提升自然度)。

**效果:** Qwen2.5-Omni 在各项单模态基准上与同规模的 Qwen2.5-VL (视觉) 和 Qwen2-Audio (音频) 表现相当或更好。在 OmniBench 等多模态融合基准上达到 SOTA。语音指令遵循能力接近文本指令。语音生成在鲁棒性和自然度上优于多数现有模型。

### Kimi-VL

**Kimi-VL** ([Kimi Team, 2025](https://arxiv.org/pdf/2504.07491)) 是一款开源的 **高效混合专家 (Mixture-of-Experts, MoE)** 视觉语言模型。

{{< figure
    src="kimi_vl_arch.png"
    caption="Fig. 21. Model architecture of Kimi-VL and Kimi-VL-Thinking, consisting of a MoonViT that allows native-resolution images, an MLP projector, and a Mixture-of-Experts (MoE) language decoder. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.07491))"
    align="center"
    width="100%"
>}}

**架构细节:**

1.  **高效 MoE 架构:**
    语言模型部分采用 MoE 架构（基于 Moonlight，类 DeepSeek-V3 架构），总参数 **16B**，每次推理仅激活 **2.8B** 参数（如每层激活 2/8 experts），在保证模型性能的同时显著降低计算成本。支持最大 **128K token** 的上下文窗口，适用于长文档、长视频等输入场景。

2.  **原生分辨率视觉编码器:**
    提出参数为 **400M** 的视觉编码器 **MoonViT**，支持图像 **原生分辨率处理**，无需缩放或填充，最大程度保留图像细节。架构基于 ViT，融合了以下技术：
    *   **NaViT 图像打包 (Patch n' Pack) 策略**：实现对变长图像序列的高效 batch 处理；
    *   **插值式绝对位置编码**：从 **SigLIP**([Zhai et al. 2023](https://arxiv.org/abs/2303.15343)) 初始化而来，提升位置感知；
    *   **二维旋转位置编码（2D-RoPE）**：增强空间结构理解；
    *   **动态分辨率训练**：训练阶段采样不同尺寸图像，提升泛化能力。

3.  **多模态融合模块:**
    MoonViT 输出的图像特征通过一个包含 **Pixel Shuffle 操作** 的 **两层 MLP Projector** 进行空间压缩和格式转换，之后与文本 token 级特征拼接输入 MoE LLM，完成图文融合处理。

4.  **长思维链推理:**
    基于主模型，通过长链思维训练流程，包括 **长思维链监督微调** 和 **强化学习优化**，提升模型在多轮、多步骤推理任务中的表现，支持复杂逻辑问答与场景理解。

**训练:**

{{< figure
    src="kimi_vl_pretrain.png"
    caption="Fig. 22. Model architecture of Kimi-VL and Kimi-VL-Thinking, consisting of a MoonViT that allows native-resolution images, an MLP projector, and a Mixture-of-Experts (MoE) language decoder. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.07491))"
    align="center"
    width="100%"
>}}

*   **预训练 (4 阶段, 共 4.4T tokens):**
    1.  **ViT 训练 (2.1T):** 单独训练 MoonViT (从 SigLIP 初始化)，使用对比损失 SigLIP 和交叉熵 caption 生成。
        $$
        \mathcal{L}=\mathcal{L}_{\text {siglip }}+\lambda \mathcal{L}_{\text {caption }}, \text { where } \lambda=2
        $$
    2.  **联合预训练 (1.4T):** 联合训练 ViT, Projector, LLM (从 Moonlight 5.2T checkpoint 初始化)，混合文本和多模态数据。
    3.  **联合冷却 (0.6T):** 使用高质量文本和多模态数据继续联合训练。
    4.  **联合长上下文激活 (0.3T):** 将上下文从 8K 扩展到 128K，使用长文本、长视频、长文档数据。

{{< figure
    src="kimi_vl_post_training.png"
    caption="Fig. 23. The post-training stages of Kimi-VL and Kimi-VL-Thinking, including two stages of joint SFT in 32K and 128K context, and further long-CoT SFT and RL stages to activate and enhance long thinking abilities. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.07491))"
    align="center"
    width="100%"
>}}

*   **后训练   :**
    1.  **联合 SFT:** 使用 ChatML 格式，在混合文本和多模态指令数据上进行微调 (先 32K 再 128K 上下文)。
    2.  **Long CoT SFT:** 使用少量高质量长 CoT 数据进行 SFT，激活长链推理能力。
    3.  **强化学习:** 采用和 **KIMI K1.5** 模型([Kimi Team, 2025](https://arxiv.org/abs/2501.12599)) 相同**在线策略镜像下降（Online Policy Mirror Descent）** 算法的进行训练。此阶段旨在通过强化学习进一步提升模型的复杂推理和规划能力（如错误识别、回溯、解决方案优化），使其能利用长思维链上下文进行隐式搜索，从而逼近显式规划算法的效果，同时保持自回归生成的简洁性。
        *   **核心目标:** 优化策略模型 $\pi_{\theta}$，使其针对问题 $x \in \mathcal{D}$ 生成的思维链 $z$ 和最终答案 $y$ 能够最大化基于基准答案 $y^*$ 的奖励期望：

            $$
            \max _{\theta} \mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D},(y, z) \sim \pi_{\theta}}\left[r\left(x, y, y^{*}\right)\right]
            $$
            其中 $r(x, y, y^*)$ 通常为 0 或 1 的正确性奖励。

        *   **奖励机制:**
            *   **正确性奖励 ($r$):** 主要基于最终答案 $y$ 的正确性，判断方式依据任务类型：
                *   对于**编程**问题：通过运行自动生成的测试用例来判断。
                *   对于**数学**问题：使用高精度的思维链奖励模型（Chain-of-Thought RM, 其准确率达 98.5%）来评估。
                *   对于**视觉**问题：利用真实世界图像、合成视觉推理数据和文本渲染图像等多种数据源，根据任务目标定义奖励。
            *   **长度惩罚 (Length Penalty):** 为解决“过度思考”并提升 token 效率，引入额外的长度奖励 $\text{len\_reward}(i)$。对于问题 $x$ 从当前策略采样 $k$ 个回答 $(y_i, z_i)$（$i=1, \dots, k$），令 $\text{len}(i)$ 为回答 $i$ 的 token 长度，$\text{min\_len} = \min_i \text{len}(i)$ 和 $\text{max\_len} = \max_i \text{len}(i)$。若 $\text{max\_len} > \text{min\_len}$，则长度奖励为：

                $$
                \text{len_reward}(i) = \begin{cases} \lambda & \text{若 } r(x, y_i, y^*) = 1 \\ \min(0, \lambda) & \text{若 } r(x, y_i, y^*) = 0 \end{cases}
                $$
                其中，长度调节因子 $\lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}}$。最终用于优化的总奖励是正确性奖励和长度奖励的加权和。此惩罚采用逐步（warm-up）引入的方式。

        *   **训练特点:**
            *   **算法:** 基于在线策略镜像下降，训练过程是迭代的。在第 $i$ 轮迭代中，使用当前模型 $\pi_{\theta_i}$ 作为参考策略，优化以下带相对熵（KL散度）正则化的目标：
                $$
                \max _{\theta} \mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D}}\left[\mathbb{E}_{(y, z) \sim \pi_{\theta}}\left[r\left(x, y, y^{*}\right)\right]-\tau \operatorname{KL}\left(\pi_{\theta}(x) \| \pi_{\theta_{i}}(x)\right)\right]
                $$
                其中 $\tau > 0$ 是控制正则化强度的参数。
            *   **优化:** 实际更新使用离策略（off-policy）数据（即从参考策略 $\pi_{\theta_i}$ 采样）和近似梯度。对于每个问题 $x$，从 $\pi_{\theta_i}$ 采样 $k$ 个回答 $(y_j, z_j)$，计算经验平均奖励 $\bar{r} = \frac{1}{k}\sum_{j=1}^{k} r(x, y_j, y^*)$ 作为基准（baseline）。模型参数 $\theta$ 的梯度近似为：
                $$
                \frac{1}{k} \sum_{j=1}^{k}\left(\nabla_{\theta} \log \pi_{\theta}\left(y_{j}, z_{j} \mid x\right)\left(r\left(x, y_{j}, y^{*}\right)-\bar{r}\right)-\frac{\tau}{2} \nabla_{\theta}\left(\log \frac{\pi_{\theta}\left(y_{j}, z_{j} \mid x\right)}{\pi_{\theta_{i}}\left(y_{j}, z_{j} \mid x\right)}\right)^{2}\right)
                $$
                该梯度形式类似于带基准的策略梯度，但加入了 $l_2$ 正则化项（最后一项的梯度）并使用离策略样本。训练中 **舍弃了价值网络（value network）** 以鼓励探索。
            *   **采样策略:** 为提高训练效率，结合使用：
                *   **课程学习 (Curriculum Sampling):** 从易到难逐步增加训练问题的难度。
                *   **优先采样 (Prioritized Sampling):** 根据模型在各问题上的历史成功率 $s_i$，以 $1-s_i$ 的比例优先采样成功率较低的问题。


### o3 & o4-mini

OpenAI 的 **o3** 和 **o4-mini** ([OpenAI, 2025](https://openai.com/index/introducing-openai-o3-and-o4-mini/)) 是其 o 系列推理模型的最新迭代，核心特点是 **更长的思考时间 (Longer Thinking Time)** 和 **全面的工具接入 (Full Tool Access)**。

**核心贡献:**
1.  **增强推理:** 模型被训练成在响应前进行更长时间、更深入的思考（类似于 CoT 或更复杂的推理过程），显著提升了在编码、数学、科学、视觉感知等复杂任务上的性能。o3 在 Codeforces, SWE-bench, MMMU 等基准上达到 SOTA。

2.  **全工具接入:** 模型可以无缝调用各种工具，如[Web Search](https://openai.com/index/introducing-chatgpt-search/)、[Code Interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter)、[GPT‑4o Image Generation](https://openai.com/index/introducing-4o-image-generation/)，以及通过 API 实现的 [Function Calling](https://platform.openai.com/docs/guides/function-calling)。模型经过训练，能够自主判断何时以及如何使用这些工具来解决问题。

3.  **多模态推理:** 模型可以将 **图像直接整合进其思维链**，实现视觉和文本的深度融合推理，而不仅仅是将图像作为输入。这使其在分析图表、图示等方面表现优异。

4.  **效率与性能权衡:** o3 是目前最强模型，适用于复杂查询；o4-mini 则针对速度和成本进行了优化，参数量更小，但在数学、编码和视觉任务上仍表现出色，尤其擅长利用工具（如在 AIME 竞赛中使用 Python 解释器）。

5.  **大规模强化学习:** o 系列模型的性能提升很大程度上归功于大规模强化学习 (RL) 的应用，验证了 RL 在提升推理能力方面的潜力，且性能随计算量增加而提升。

{{< figure
    src="thinking_with_images_static.webp"
    caption="Fig. 24. o3 model demonstrates its multimodal CoT capability by analyzing a user-uploaded image, identifying the ship, and using tools (web search) to find information, ultimately answering the ship's name and its next port of call. (Image source: [OpenAI, 2025](https://openai.com/index/introducing-o3-and-o4-mini/))"
    align="center"
    width="100%"
>}}

**工作机制:**

*   **长时间思考:** 借鉴了“计算换性能”的思想 ([Snell et al., 2024](https://arxiv.org/abs/2408.03314))，通过在推理时增加计算量（如多次采样、使用更长的推理链、MCTS 等搜索算法）来提升复杂任务的性能，这可能比单纯增加模型参数更有效。模型内部被设计为可以进行多步推理或更复杂的计算过程，用户可以通过选择不同的 **推理努力程度 (reasoning effort)** 设置（如 o4-mini-high）来调整模型的思考时间。

*   **工具使用:** 模型通过 RL 或指令微调学习工具使用的策略。当面对一个问题时，模型会：
    *   **规划:** 分析问题，判断是否需要以及需要哪些工具。
    *   **执行:** 调用选定的工具（如进行网络搜索获取最新信息，运行代码进行计算）。
    *   **整合:** 将工具返回的结果整合到其推理过程中，生成最终答案。
    这个过程可以是多轮迭代的，模型可以根据工具返回的信息调整策略（如进行二次搜索）。
*   **多模态思维链 (Multimodal Chain-of-Thought, MCoT)** 模型可以直接在其内部推理步骤中引用和分析图像内容，例如识别图表中的数据点，理解流程图的步骤，或解释照片中的细节。感兴趣的读者可以阅读 **MCoT 综述**([Wang et al., 2025](https://arxiv.org/abs/2503.12605)) 介绍其扩展到包含图像、视频、音频、3D、表格/图表等多种模态场景。

**效果:**

{{< figure
    src="o3_o4_benchmark.png"
    caption="Fig. 25. To highlight visual reasoning improvement versus our previous multimodal models, OpenAI tested o3 and o4-mini on a diverse set of human exams and ML benchmarks. These new visual reasoning models significantly outperform their predecessors on all multimodal tasks we tested. (Image source: [OpenAI, 2025](https://openai.com/index/thinking-with-images/))"
    align="center"
    width="100%"
>}}

o3 和 o4-mini 在多项基准测试中展现了 SOTA 或接近 SOTA 的性能，尤其是在需要深度推理和工具辅助的任务上。专家评估显示，它们相比前代 o1/o3-mini 产生的严重错误更少，回答更实用、可验证，并且交互更自然。

## 总结

多模态大语言模型正朝着更全面、更智能、更高效的方向发展。它们不仅能够理解和生成跨越文本、图像、视频、音频等多种模态的内容，还能进行复杂的推理、规划和工具调用。未来，我们可以期待 MLLMs 在效率优化、更深层次的跨模态融合与推理、更强的时序和空间理解能力、以及安全性和可控性方面取得进一步突破。

## 参考文献

[1] OpenAI. ["Hello gpt‑4o."](https://openai.com/index/hello-gpt-4o/) OpenAI Blog (2024).

[2] DeepMind. ["Gemini 2.5 Pro"](https://deepmind.google/technologies/gemini/pro/) DeepMind Blog (2025).

[3] OpenAI. ["Introducing OpenAI o3 and o4‑mini."](https://openai.com/index/introducing-o3-and-o4-mini/) OpenAI Blog (2025).

[4] Zhang, Duzhen, et al. ["Mm-llms: Recent advances in multimodal large language models."](https://arxiv.org/abs/2401.13601) arXiv preprint arXiv:2401.13601 (2024).

[5] Dosovitskiy, Alexey, et al. ["An image is worth 16×16 words: Transformers for image recognition at scale."](https://arxiv.org/abs/2010.11929) arXiv preprint arXiv:2010.11929 (2020).

[6] Radford, Alec, et al. ["Learning transferable visual models from natural language supervision."](https://arxiv.org/abs/2103.00020) International conference on machine learning. PmLR, 2021.

[7] Li, Junnan, et al. ["Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation."](https://arxiv.org/abs/2201.12086) International conference on machine learning. PMLR, 2022.

[8] Li, Junnan, et al. ["Align before fuse: Vision and language representation learning with momentum distillation."](https://arxiv.org/abs/2107.07651) Advances in neural information processing systems 34 (2021): 9694-9705.

[9] Li, Junnan, et al. ["Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models."](https://arxiv.org/abs/2301.12597) International conference on machine learning. PMLR, 2023.

[10] Liu, Haotian, et al. ["Visual instruction tuning."](https://arxiv.org/abs/2304.08485) arXiv preprint arXiv:2304.08485 (2023).

[11] Bai, Jinze, et al. ["Qwen-vl: A frontier large vision-language model with versatile abilities."](https://arxiv.org/abs/2308.12966) arXiv preprint arXiv:2308.12966 1.2 (2023): 3.

[12] Wang, Peng, et al. ["Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution."](https://arxiv.org/abs/2409.12191) arXiv preprint arXiv:2409.12191 (2024).

[13] Dehghani, Mostafa, et al. ["Patch n' pack: NaViT, a vision transformer for any aspect ratio and resolution."](https://arxiv.org/abs/2307.06304) Advances in Neural Information Processing Systems 36 (2023): 2252-2274.

[14] Su, Jianlin, et al. ["Roformer: Enhanced transformer with rotary position embedding."](https://arxiv.org/abs/2104.09864) Neurocomputing 568 (2024): 127063.

[15] Su, Jianlin. ["Transformer升级之路：4、二维位置的旋转位置编码."](https://spaces.ac.cn/archives/8397) *科学空间* (blog) (2021).

[16] Bai, Shuai, et al. ["Qwen2.5‑VL Technical Report."](https://arxiv.org/abs/2502.13923) arXiv preprint arXiv:2502.13923 (2025).

[17] Xu, Jin, et al. ["Qwen2.5‑Omni Technical Report."](https://arxiv.org/abs/2503.20215) arXiv preprint arXiv:2503.20215 (2025).

[18] Lipman, Yaron, et al. ["Flow matching for generative modeling."](https://arxiv.org/abs/2210.02747) arXiv preprint arXiv:2210.02747 (2022).

[19] Lee, Sang-gil, et al. ["Bigvgan: A universal neural vocoder with large-scale training."](https://arxiv.org/abs/2206.04658) arXiv preprint arXiv:2206.04658 (2022).

[20] Kimi Team. ["Kimi‑VL Technical Report."](https://arxiv.org/abs/2504.07491) arXiv preprint arXiv:2504.07491 (2025).

[21] Zhai, Xiaohua, et al. ["Sigmoid loss for language image pre-training."](https://arxiv.org/abs/2303.15343) Proceedings of the IEEE/CVF international conference on computer vision. 2023.

[22] Kimi Team.  ["Kimi k1. 5: Scaling reinforcement learning with llms."](https://arxiv.org/abs/2501.12599) arXiv preprint arXiv:2501.12599 (2025).

[23] Snell, Charlie, et al. ["Scaling llm test-time compute optimally can be more effective than scaling model parameters."](https://arxiv.org/abs/2408.03314) arXiv preprint arXiv:2408.03314 (2024).

[24] Wang, Yaoting, et al. ["Multimodal chain-of-thought reasoning: A comprehensive survey."](https://arxiv.org/abs/2503.12605) arXiv preprint arXiv:2503.12605 (2025).


## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (May 2025). 多模态大语言模型.
> https://syhya.github.io/zh/posts/2025-05-04-multimodal-llm/

Or

```bibtex
@article{yue_shui_multimodal_llm_2025,
  title   = "多模态大语言模型",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "May",
  url     = "https://syhya.github.io/zh/posts/2025-05-04-multimodal-llm/"
}
```