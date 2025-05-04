---
title: "Multimodal Large Language Models"
date: 2025-05-04T12:00:00+08:00
author: "Yue Shui"
tags: ["Multimodal", "MLLMs", "ViT", "CLIP", "BLIP", "LLaVA", "OpenAI", "Qwen-VL", "Kimi-VL"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

Humans interact with the world through multiple senses (vision, hearing, touch, etc.), with each sensory channel offering unique advantages in representing and communicating specific concepts. This multimodal interaction fosters our deep understanding of the world. One of the core goals in the field of artificial intelligence is to develop general-purpose assistants that can effectively follow multimodal instructions (such as visual and linguistic ones), enabling them to perform various real-world tasks like humans. In recent years, with the release of models like GPT-4o ([OpenAI, 2024](https://openai.com/index/hello-gpt-4o/)), Gemini 2.5 Pro ([DeepMind, 2025](https://deepmind.google/technologies/gemini/pro/)), and o3/o4-mini ([OpenAI, 2025](https://openai.com/index/introducing-o3-and-o4-mini/)), **Multimodal Large Language Models (MLLMs)** have made significant progress. They can not only understand information from multiple modalities like images, videos, and audio but also perform complex reasoning and generation.

## Notations

The following table lists the key mathematical symbols used in this article and their meanings to help you read more easily.

| Symbol                                                                         | Description                                                                                                                             |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| \( I, \mathbf{X}_v \)                                                        | Image input, \( I \) usually refers to the raw image matrix \( \in \mathbb{R}^{H \times W \times C} \)                                                     |
| \( T, \mathbf{X}_c, \mathbf{X}_q, \mathbf{X}_a, \mathbf{X}_{\text{instruct}} \) | Text input, specifically could refer to image caption (\( \mathbf{X}_c \)), user question (\( \mathbf{X}_q \)), model answer (\( \mathbf{X}_a \)), or instruction (\( \mathbf{X}_{\text{instruct}} \)) |
| \( V, \mathbf{Z}_v \)                                                        | Raw image features or embedding sequence output by the image encoder                                                                                             |
| \( L, \mathbf{H}_q, \mathbf{H}_a \)                                          | Text features or embedding sequence output by the text encoder                                                                                             |
| \( \mathbf{H}_v \)                                                           | Visual token sequence input to the LLM after processing by a projection layer (e.g., LLaVA, Qwen)                                                                    |
| \( Z \)                                                                      | Query embeddings output by Q-Former, serving as a compressed representation of visual information (BLIP-2)                                                                         |
| \( P_Z \)                                                                    | Soft Visual Prompt derived from the Q-Former output (BLIP-2)                                                               |
| \( I_e, T_e \)                                                               | Image and text embeddings in CLIP's shared multimodal embedding space                                                                                     |
| \( z_p \)                                                                    | Embedding vector of a single image patch after linear projection in ViT                                                                                   |
| \( x_{class} \)                                                              | Embedding of the learnable `[class]` token used for classification tasks in ViT                                                                                   |
| \( x_i \)                                                                    | The \( i \)-th element or token in a sequence (e.g., word \( w_i \) in a text sequence)                                                                   |
| \( E_{img}, g(\cdot) \)                                                      | Image encoder model (e.g., ViT)                                                                                                          |
| \( E_{text}, f_{\phi}(\cdot) \)                                              | Text encoder or Large Language Model (LLM)                                                                                                     |
| \( E, \mathbf{W}, \mathbf{W}_i, \mathbf{W}_t \)                              | Linear projection matrix, used for feature transformation or modality alignment                                                                                             |
| \( E_{pos} \)                                                                | Positional encoding vector, used to provide sequence position information to the Transformer                                                                                |
| \( Q, K, V \)                                                                | Query, Key, Value matrices in the attention mechanism                                                                              |
| \( W_Q, W_K, W_V \)                                                          | Learnable projection matrices used to compute Q, K, V from input                                                                                            |
| \( \theta, \phi \)                                                           | Set of trainable parameters for the entire model or a specific part (e.g., LLM \( \phi \))                                                                             |
| \( P \)                                                                      | Side length of an image patch defined in the ViT model                                                                                              |
| \( N \)                                                                      | Batch Size, usually refers to the number of samples in a batch                                                                                  |
| \( N_{patches} \)                                                            | Number of image patches the ViT model divides an image into                                                                                                 |
| \( D \)                                                                      | Main dimension of embedding vectors in the model                                                                                                         |
| \( d, d_k \)                                                                 | Dimension of the key vector in the attention mechanism, used for scaling the dot product                                                                                      |
| \( T_{turns} \)                                                              | Total number of conversation turns in multi-turn dialogue data (LLaVA)                                                                                               |
| \( \mathcal{L} \)                                                            | Loss function, the objective optimized by the model (e.g., \( \mathcal{L}_{ITC}, \mathcal{L}_{ITM}, \mathcal{L}_{LM}, \mathcal{L}_{CLIP}, \mathcal{L}_{siglip} \)) |
| \( \tau \)                                                                   | Learnable parameter, such as temperature in contrastive loss or KL regularization weight in reinforcement learning                                                                           |
| \( \lambda \)                                                                | Hyperparameter, such as the weight of different loss terms or length penalty factor in reinforcement learning                                                                                 |
| \( y \)                                                                      | Target label or category (e.g., ITM loss); or the final answer generated by the model (e.g., Kimi-VL RL)                                                                 |
| \( x \)                                                                      | Input data, context, or question                                                                                                             |
| \( z \)                                                                      | Intermediate reasoning steps or chain-of-thought generated by the model                                                                                  |
| \( y^* \)                                                                    | Reference answer or ground truth answer                                                                                                |
| $\operatorname{sim}(u, v) = s(u, v)$                                                      | Similarity calculation between vectors \( u \) and \( v \), usually cosine similarity                                                                         |
| \( \mathbb{E} \)                                                             | Mathematical expectation                                                                                                                         |
| KL                                                                           | KL Divergence (Kullback–Leibler Divergence), used to measure the difference between two probability distributions                                                                    |
| \( \pi_{\theta} \)                                                           | Policy model, outputs actions or text sequences based on parameters \( \theta \)                                                                                  |
| \( r \)                                                                      | Reward function, evaluates the quality of the generated output                                                                                                       |

## Multimodal Fundamentals

Before diving into specific technologies, let's understand some basic concepts of multimodality.

### What is Multimodality?

**Multimodality** refers to the use of multiple different types of data or information channels (modalities) to represent and process information. Humans are inherently multimodal beings; we perceive and understand the world through vision, hearing, touch, smell, taste, and language. In the field of artificial intelligence, multimodal learning aims to build models capable of processing and correlating information from different modalities (such as text, images, videos, audio, etc.).

{{< figure
    src="multimodality_data.png"
    caption="Fig. 1. Multimodality Data. (Image source: [GPT-4o Image Generation](https://chatgpt.com/s/m_6814c5d31e288191a5409a7420ee30f4))"
    align="center"
    width="60%"
>}}

**Common Modalities:**
*   **Text:** Natural language text, the primary means of information transmission and knowledge expression.
*   **Image:** Static visual information, containing rich details of scenes, objects, and textures.
*   **Video:** Dynamic visual information, composed of sequential image frames, often accompanied by audio. Video contains not only spatial information but also temporal information.
*   **Audio:** Sound information, including speech, music, and environmental sounds.
*   **Others:** Tabular data, 3D point clouds, sensor data (e.g., radar, LiDAR), biological signals (e.g., EEG, ECG), etc.

### Why Do We Need Multimodal AI?

1.  **More Comprehensive World Understanding:** The real world is multimodal. A single modality often provides only partial information. For example, text descriptions alone may struggle to fully convey a complex scene, whereas combining images or videos offers more intuitive and richer information. Multimodal models can integrate information from different sources to form a more comprehensive and accurate understanding.
2.  **Enhanced Task Performance:** In many tasks, combining information from multiple modalities can significantly improve performance. For instance, in Visual Question Answering (VQA), the model needs to understand both the image content and the text question to provide the correct answer. In video captioning, combining visual frames and audio information can generate more vivid and accurate descriptions.
3.  **More Natural Interaction:** Multimodal AI makes human-computer interaction more natural and flexible. Users can interact with AI systems through various means like voice, text, and images, and the AI system can respond in multiple modalities (e.g., generating text replies with images, or generating voice answers).
4.  **Unlocking New Application Scenarios:** Multimodal capabilities have given rise to many new applications, such as autonomous driving (fusing data from cameras, radar, LiDAR), medical diagnosis (combining medical images and patient records), content creation (text-to-image, text-to-video), virtual assistants, robot interaction, etc.
5.  **Promoting Accessibility:** Multimodal technology can assist individuals with sensory impairments. For example, image captioning can help visually impaired people understand image content, while speech recognition and synthesis can aid those with hearing or speech impairments in communication.

### Common Multimodal Tasks

The following table lists some common multimodal tasks, which typically require processing and generating information by combining multiple modalities.

| Task Name                       | Description                                                              |
| :------------------------------ | :----------------------------------------------------------------------- |
| Visual Question Answering (VQA) | Generate text answers based on an image and a related question.          |
| Image/Video Captioning          | Generate natural language descriptions for images or videos.             |
| Text-to-X Generation            | Generate corresponding image, video, or audio content from text descriptions. |
| Cross-Modal Retrieval           | Use one modality (e.g., text) to query relevant data in another modality (e.g., image). |
| Multimodal Sentiment Analysis   | Determine sentiment by combining information from text, audio, video, etc. |
| Visual Reasoning                | Perform logical judgment and relationship inference based on image or video content. |
| Visual Language Navigation (VLN)| Guide an agent to navigate in a visual environment based on natural language instructions. |
| Multimodal Machine Translation (MMT) | Utilize relevant image information to assist text translation and resolve ambiguity. |
| Audio-Visual Speech Recognition (AVSR) | Perform speech recognition by combining audio signals and visual information of the speaker's lip movements. |
| Visual Grounding                | Associate words or phrases in text with corresponding regions or objects in images or videos. |

## Key Technologies

The development of multimodal large models is driven by a series of technologies. The figure below visually illustrates the related technologies for multimodal understanding and generation. The author will introduce some key models and methods among them.

{{< figure
    src="MLLMs_arch.png"
    caption="Fig. 2. The general model architecture of MM-LLMs and the implementation choices for each component. (Image source: [Zhang et al., 2024](https://arxiv.org/pdf/2401.13601))"
    align="center"
    width="100%"
>}}

### Vision Transformer (ViT)

**Vision Transformer (ViT)** ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) successfully applied the Transformer architecture to the field of computer vision, becoming the preferred visual encoder for many advanced MLLMs today.

{{< figure
    src="vit_overview.png"
    caption="Fig. 3. ViT model overview. (Image source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))"
    align="center"
    width="100%"
>}}

**Core Idea:** ViT treats an image as a sequence of **Image Patches** and then uses the Transformer's self-attention mechanism to process these patches, thereby capturing global dependencies.

**Workflow:**

1.  **Patch Embedding:** Divide the input image \( I \in \mathbb{R}^{H \times W \times C} \) into \( N_{patches} \) fixed-size non-overlapping image patches \( x_p \in \mathbb{R}^{P^2 \times C} \), where \( (H, W) \) is the image resolution, \( C \) is the number of channels, \( P \) is the size of each patch, and \( N_{patches} = HW/P^2 \) is the number of patches.
2.  **Linear Projection:** Flatten each patch \( x_p \) into a 1D vector and map it to a \( D \)-dimensional embedding space using a learnable linear projection matrix \( E \), resulting in patch embeddings \( z_p = x_p E \).
3.  **Position Embedding:** To preserve the spatial position information of the patches, ViT adds learnable **Position Embeddings** \( E_{pos} \) to the patch embeddings.
    \[ z_0 = [x_{class}; z_p^1; z_p^2; \dots; z_p^{N_{patches}}] + E_{pos}, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N_{patches}+1) \times D} \]
    Often, a learnable `[class]` token embedding \( x_{class} \) is also added. Its corresponding vector at the Transformer's output is used for image classification tasks.
4.  **Transformer Encoder:** Feed the sequence of patch embeddings with added position encodings into a standard Transformer encoder. The encoder consists of multiple layers of **Multi-Head Self-Attention (MSA)** and **Feed Forward Network (FFN)**.
    *   **MSA:** Captures global dependencies between image patches. For an input sequence \( Z_{l-1} \), the self-attention is computed as:
        \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
        where \( Q = Z_{l-1}W_Q, K = Z_{l-1}W_K, V = Z_{l-1}W_V \) are the query, key, and value matrices, and \( d_k \) is the dimension of the key vectors. Multi-head attention splits \( Q, K, V \) into multiple heads, computes attention in parallel, and then concatenates the results.
    *   **FFN:** Typically consists of two linear layers and a non-linear activation function (e.g., GELU).
    The computation in each layer can be represented as:
    \[ Z'_l = \text{MSA}(\text{LN}(Z_{l-1})) + Z_{l-1} \]
    \[ Z_l = \text{FFN}(\text{LN}(Z'_l)) + Z'_l \]
    where LN denotes Layer Normalization.
5.  **Output:** The output \( Z_L \) of the Transformer encoder serves as the image's feature representation \( V \).

{{< figure
    src="vit_bit_hybrid_compare.png"
    caption="Fig. 4. Performance versus pre-training compute for different architectures: Vision Transformers, ResNets, and hybrids. Vision Transformers generally outperform ResNets with the same computational budget. Hybrids improve upon pure Transformers for smaller model sizes, but the gap vanishes for larger models. (Image source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))"
    align="center"
    width="100%"
>}}

Compared to traditional Convolutional Neural Networks (CNNs), ViT offers the following advantages:

1.  **Global Dependency Modeling**: Self-attention directly connects any two patches, explicitly capturing long-range spatial relationships, making it better at integrating semantic information across the entire image than traditional CNNs.
2.  **Strong Transferability from Large-Scale Pre-training**: After pre-training on massive datasets like JFT-300M or ImageNet-22K, ViT can be easily transferred to over 20 downstream tasks such as classification, detection, and segmentation, with performance scaling almost linearly with model/data size.
3.  **Simple Architecture, Easy to Scale and Parallelize**: The pure Transformer stack is easy to scale in three dimensions: depth, width, and input resolution. Computations consist mainly of matrix multiplications and Softmax, naturally suited for large-batch parallel processing and mixed-precision training on GPUs/TPUs.

As research progresses, ViT itself is continuously being optimized to meet the demands of MLLMs:

1.  **Native Dynamic Resolution:** Traditional ViTs typically require a fixed input resolution. Models like Qwen2-VL and Kimi-VL have introduced the capability to handle dynamic resolutions. They often remove absolute position embeddings in ViT and instead use 2D rotary position embeddings to encode two-dimensional spatial information. This allows the model to process images of arbitrary resolutions and aspect ratios, converting them into variable-length visual token sequences, better preserving detailed information. Kimi-VL's MoonViT also borrows the image packing technique from NaViT, packing sequences of image patches with different resolutions into the Transformer, improving training efficiency.
2.  **Window Attention:** To reduce the quadratic computational complexity of self-attention when processing high-resolution images, Qwen2.5-VL employs window attention in most layers of its ViT. Attention computation is restricted within local windows, making the complexity linear with respect to the number of patches, significantly improving efficiency while maintaining global information interaction through a few full attention layers.
3.  **Architecture Alignment with LLM:** Models like Qwen2.5-VL and Kimi-VL have also fine-tuned their ViT architectures to be closer to LLM designs, such as using RMSNorm for normalization and SwiGLU as the activation function, to enhance computational efficiency and cross-modal compatibility.

### CLIP

**CLIP (Contrastive Language-Image Pre-training)** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) is a landmark work in the multimodal field. It proposed a simple yet effective method for learning the association between images and text, laying the foundation for many subsequent MLLMs.

**Core Idea:** CLIP aims to learn a **Multimodal Embedding Space** where matched image-text pairs have high similarity, and mismatched pairs have low similarity. It achieves this through **Contrastive Learning**, leveraging natural language supervision.

**Architecture:** CLIP consists of two main parts:

1.  **Image Encoder:** Can be a ResNet or ViT, responsible for encoding the input image \( I \) into image features \( V \).
2.  **Text Encoder:** Typically a Transformer, responsible for encoding the input text \( T \) into text features \( L \).
3.  **Linear Projection Layer:** Projects the image features \( V \) and text features \( L \) into the shared multimodal embedding space, obtaining \( I_e = V W_i \) and \( T_e = L W_t \), where \( W_i \) and \( W_t \) are learnable projection matrices.

{{< figure
    src="https://cdn.mathpix.com/cropped/2025_04_14_455634672e9a2826be22g-02.jpg?height=629&width=1709&top_left_y=214&top_left_x=181"
    caption="Fig. 5. CLIP Architecture Overview. CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset's classes. (Image source: [Radford et al., 2021](https://arxiv.org/abs/2103.00020))"
    align="center"
    width="100%"
>}}

**Training Data (WIT):** CLIP's success is largely attributed to its massive pre-training dataset, **WIT (WebImageText)**. The research team collected 400 million (image, text) pairs from the internet. They built the dataset by searching for approximately 500,000 query terms (derived from Wikipedia vocabulary, high-frequency bigrams, Wikipedia article titles, and WordNet synsets), limiting the number of samples per query to a maximum of 20,000 to balance the data distribution. This approach of using native web image-text pairs is called **natural language supervision**, which avoids expensive manual annotation and allows for easy scaling of data size.

**Contrastive Loss:** The core of CLIP is the contrastive learning objective. Given a batch of \( N \) (image, text) pairs \( \{(I_1, T_1), \dots, (I_N, T_N)\} \), the model's goal is to predict which of the \( N \times N \) possible pairings are the true pairings.

1.  Compute all image embeddings \( \{I_{e,1}, \dots, I_{e,N}\} \) and text embeddings \( \{T_{e,1}, \dots, T_{e,N}\} \). **L2 normalization** is typically applied, dividing each image or text embedding by its own L2 norm (Euclidean norm).
2.  Calculate the **Cosine Similarity** between all \( N^2 \) pairs \( (I_{e,i}, T_{e,j}) \).
    \[ \text{logits}_{i,j} = \text{sim}(I_{e,i}, T_{e,j}) \cdot \exp(\tau) = \frac{I_{e,i} \cdot T_{e,j}}{\|I_{e,i}\| \|T_{e,j}\|} \cdot \exp(\tau) \]
    where \( \tau \) is a learnable **temperature parameter** used to scale the range of the logits.
3.  Compute the **Symmetric Cross-Entropy Loss**. The problem is treated as two classification tasks:
    *   For each image \( I_i \), find the matching text \( T_i \) among the \( N \) texts. The loss is \( \mathcal{L}_{\text{image}} \).
    *   For each text \( T_j \), find the matching image \( I_j \) among the \( N \) images. The loss is \( \mathcal{L}_{\text{text}} \).
    The total loss is:
    \[ \mathcal{L}_{CLIP} = \frac{1}{2} (\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}) \]
    where,
    \[ \mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(I_{e,i}, T_{e,i}) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_{e,i}, T_{e,j}) / \tau)} \]
    \[ \mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(\text{sim}(I_{e,j}, T_{e,j}) / \tau)}{\sum_{i=1}^N \exp(\text{sim}(I_{e,i}, T_{e,j}) / \tau)} \]
    This loss function encourages the similarity of positive pairs (matching image-text) to be higher than that of negative pairs (mismatched image-text).

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

**Zero-Shot Transfer:** CLIP's powerful capability lies in its zero-shot transfer performance. For a new image classification task, without any fine-tuning, CLIP can make predictions as follows:

1.  Obtain all class names for the task (e.g., "cat", "dog").
2.  Use **Prompt Engineering** to structure the class names into sentences, like "A photo of a {label}." This helps bridge the distribution gap between the pre-training data (often sentences) and downstream tasks (often word labels). The CLIP paper found that using prompt templates and ensembling multiple prompts significantly improves performance (nearly 5% gain on ImageNet).
3.  Use CLIP's text encoder to compute the text embeddings for each constructed sentence. These embeddings form the **weight vectors** of the zero-shot classifier.
4.  For a new image to be classified, use CLIP's image encoder to compute its image embedding.
5.  Calculate the cosine similarity between this image embedding and all class text embeddings.
6.  The class with the highest similarity is predicted as the result.

{{< figure
    src="clip_prompt_engineering.png"
    caption="Fig. 6. Prompt engineering and ensembling improve zero-shot performance. Compared to the baseline of using contextless class names, prompt engineering and ensembling boost zero-shot classification performance by almost 5 points on average across 36 datasets. This improvement is similar to the gain from using 4 times more compute with the baseline zero-shot method but is 'free' when amortized over many predictions. (Image source: [Radford et al., 2021](https://arxiv.org/abs/2103.00020))"
    align="center"
    width="70%"
>}}

**Impact of CLIP:** CLIP demonstrated that powerful, transferable visual representations can be learned through large-scale natural language supervision and contrastive learning. Its learned multimodal embedding space and strong image encoder have been widely adopted in subsequent MLLMs (like Flamingo, BLIP-2, LLaVA) and text-to-image models (like DALL-E 2, Stable Diffusion).

CLIP primarily focuses on learning aligned representations but has limited capabilities in generative tasks. Subsequent work began exploring unified model architectures capable of both understanding and generation.

### BLIP

**BLIP (Bootstrapping Language-Image Pre-training)** ([Li et al., 2022](https://arxiv.org/abs/2201.12086)) aimed to address the limitations of existing **Vision-Language Pre-training (VLP)** methods in terms of both models and data: models often excel at either understanding or generation, but not both; data relies on massive and noisy web image-text pairs.

**MED (Multimodal Encoder-Decoder):**

BLIP proposed the **MED (Multimodal Encoder-Decoder)** architecture, designed to unify understanding and generation tasks. It combines the advantages of CLIP's contrastive learning and autoregressive generation, capable of handling multimodal data.

{{< figure
    src="blip_model_architecture.png"
    caption="Fig. 7. BLIP Pre-training Model Architecture and Objectives (same parameters have the same color). We propose multimodal mixture of encoder-decoder (MED), a unified vision-language model which can operate in one of the three functionalities. (Image source: [Li et al., 2022](https://arxiv.org/abs/2201.12086))"
    align="center"
    width="100%"
>}}

*   **Image Encoder:** Uses ViT.
*   **Text Encoder/Decoder:** Based on the BERT architecture but modified to accommodate multimodal tasks and different functional modes.
    *   **Unimodal Encoder:** Standard ViT and BERT, processing images and text separately.
    *   **Image-grounded Text Encoder:** Inserts **Cross-Attention (CA)** layers between the Self-Attention (SA) layer and the Feed-Forward Network (FFN) layer in each Transformer block of the text encoder to inject visual information. A `[Encode]` token is prepended to the text input, and its output embedding serves as the multimodal representation of the image-text pair.
    *   **Image-grounded Text Decoder:** Replaces the bidirectional SA layers in the encoder with **Causal Self-Attention** layers for autoregressive generation. Shares the CA and FFN layers with the encoder. Uses a `[Decode]` token as the sequence start symbol.

**Pre-training Objectives:** BLIP jointly optimizes three objectives:

1.  **Image-Text Contrastive (ITC) Loss:** Similar to CLIP, uses the unimodal encoders to align the feature spaces of images and text. BLIP adopts the momentum encoder and soft label strategy proposed by **ALBEF** ([Li et al., 2021](https://arxiv.org/abs/2107.07651)) to improve contrastive learning.
    $$L_{ITC} = \frac{1}{2N} \sum_{i=1}^{N} \left( -\log \frac{\exp(s(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(v_i, t_j)/\tau)} -\log \frac{\exp(s(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(v_j, t_i)/\tau)} \right)$$
    where $v_i, t_j$ are image and text features, $s$ is the similarity function, and $\tau$ is the temperature parameter.

2.  **Image-Text Matching (ITM) Loss:** Uses the image-grounded text encoder to learn fine-grained image-text alignment. This is a binary classification task predicting whether an image-text pair is matched or mismatched. A hard negative mining strategy is employed.
    $$L_{ITM} = -\mathbb{E}_{(I,T)\sim D} [y \log p_{match} + (1-y) \log(1 - p_{match})]$$
    where $y$ is the label and $p_{match}$ is the matching probability.

3.  **Language Modeling (LM) Loss:** Uses the image-grounded text decoder to generate text descriptions based on the image. A standard cross-entropy loss (with label smoothing) is used.
    $$L_{L M}=-\mathbb{E}_{(I, T) \sim D} \sum_{k=1}^L \log P\left(w_k \mid I, w_{\lt k} ; \theta\right)$$
    where $w_k$ is a word in the text sequence and $\theta$ are the model parameters.

**Total Loss Function:** These three losses are typically optimized jointly (e.g., summed with equal weights):
$$L_{BLIP} = L_{ITC} + L_{ITM} + L_{LM}$$

**Parameter Sharing:** For efficiency and the benefits of multi-task learning, the text encoder and decoder share all parameters except for the SA layers (i.e., embedding layer, CA layers, FFN layers).

**CapFilt:**

**CapFilt (Captioning and Filtering)** is an innovative dataset bootstrapping method used to generate high-quality synthetic captions for unlabeled web images and filter out noisy data (including original web text and synthetic text).

{{< figure
    src="blip_learning_framework.png"
    caption="Fig. 8. BLIP Learning Framework. We introduce a captioner to produce synthetic captions for web images, and a filter to remove noisy image-text pairs. (Image source: [Li et al., 2022](https://arxiv.org/abs/2201.12086))"
    align="center"
    width="100%"
>}}

1.  **Initialization:** Initialize two modules using a pre-trained MED model: a Captioner (image-grounded text decoder) and a Filter (image-grounded text encoder).
2.  **Fine-tuning:** Fine-tune the Captioner (using LM loss) and Filter (using ITC and ITM losses) separately on a high-quality human-annotated dataset (e.g., COCO). This is a lightweight process.
3.  **Generation and Filtering:**
    *   The Captioner generates synthetic captions \( T_s \) for web images \( I_w \).
    *   The Filter determines whether the original web text \( T_w \) and the synthetic text \( T_s \) match the image \( I_w \). Texts predicted as mismatched are considered noise and removed.
4.  **Bootstrapped Dataset:** Combine the filtered high-quality image-text pairs (from original web data and synthetic data) with human-annotated data to form a new bootstrapped dataset.
5.  **Re-Pre-training:** Pre-train a new BLIP model from scratch using the bootstrapped dataset.

**Effect:** CapFilt significantly improved the model's performance on various downstream tasks (like retrieval, captioning, VQA), demonstrating the effectiveness of improving noisy data quality through bootstrapping. BLIP also showcased the flexibility of a unified model for both understanding and generation tasks.

### BLIP-2

**BLIP-2** ([Li et al., 2023](https://arxiv.org/abs/2301.12597)) addresses the high cost of VLP training by proposing an **efficient** pre-training strategy: freeze pre-trained image encoders and large language models, training only a lightweight bridging module, the Q-Former.

**Core Contributions:**

1.  **Leveraging Frozen Models:** Avoids end-to-end training of the entire large model, significantly reducing computational costs and leveraging the capabilities of powerful pre-trained unimodal models.
2.  **Q-Former (Querying Transformer):** Proposed a lightweight Transformer structure as a trainable bridge connecting the frozen image encoder and the frozen LLM.
3.  **Two-Stage Pre-training:** Designed a two-stage strategy to effectively bridge the modality gap:
    *   **Stage 1: Vision-Language Representation Learning:** Bootstrapped from a frozen image encoder.
    *   **Stage 2: Vision-to-Language Generative Learning:** Bootstrapped from a frozen LLM.

**Architecture (Q-Former):**

*   Q-Former is a lightweight Transformer with 188M parameters.
*   It uses a set of **Learnable Query Embeddings** (e.g., 32 vectors of 768 dimensions) as input.
*   These query vectors interact with each other through **Self-Attention layers**.
*   They interact with the image features output by the frozen image encoder through **Cross-Attention layers** to extract visual information.
*   The output of the query vectors \( Z \) (e.g., \( 32 \times 768 \) dimensions) has a much lower dimension than the original image features, acting as an **Information Bottleneck**, forcing the Q-Former to extract the visual information most relevant to the language model.
*   Q-Former internally contains two sub-modules, an image Transformer and a text Transformer, which share self-attention layers.

{{< figure
    src="blip2_stage1.png"
    caption="Fig. 9. (Left) Model architecture of Q-Former and BLIP-2's first-stage vision-language representation learning objectives. (Right) The self-attention masking strategy for each objective to control query-text interaction. (Image source: [Li et al., 2023](https://arxiv.org/abs/2301.12597))"
    align="center"
    width="100%"
>}}

**Two-Stage Pre-training:**

1.  **Stage 1 (Representation Learning):**
    *   Connect the Q-Former to a **frozen image encoder** (e.g., CLIP ViT-L/14, EVA-CLIP ViT-g/14).
    *   Pre-train using image-text pairs, aiming for the Q-Former's query vectors to learn to extract visual representations most relevant to the text.
    *   Jointly optimize three objectives similar to BLIP (sharing input format and model parameters, but **freezing the image encoder and training only the Q-Former**):
        *   **Image-Text Contrastive (ITC) Loss:** Align the Q-Former's output query representations \( z \) and text representations \( t \). Uses In-batch Negatives.
            $$L_{ITC} = \frac{1}{2N} \sum_{i=1}^{N} \left( -\log \frac{\exp(s(z_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(z_i, t_j)/\tau)} -\log \frac{\exp(s(z_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(z_j, t_i)/\tau)} \right)$$
        *   **Image-Text Matching (ITM) Loss:** Predict whether an image-text pair matches. Uses the Q-Former's output multimodal query representation for classification.
            $$L_{ITM} = -\mathbb{E}_{(I,T)\sim D} [y \log p_{match} + (1-y) \log(1 - p_{match})]$$
        *   **Image-grounded Text Generation (ITG) Loss:** Train the Q-Former to generate text. The query vectors need to capture all information required for text generation and pass it to the text tokens via self-attention layers.
            $$L_{ITG} = -\mathbb{E}_{(I,T)\sim D} \sum_{k=1}^{L} \log P(w_k | Z_q, w_{\lt k}; \theta_{Q-Former})$$
            where $Z_q$ is the query output of the Q-Former.
    *   Different self-attention masks are used to control query-text interaction for different objectives.
    *   **Stage 1 Total Loss Function:**
        $$L_{Stage1} = L_{ITC} + L_{ITM} + L_{ITG}$$

2.  **Stage 2 (Generative Learning):**
    *   Connect the **Q-Former pre-trained in Stage 1** (and its connected frozen image encoder) to a **frozen LLM** (e.g., OPT series, FlanT5 series).
    *   Use a **Fully Connected (FC) Layer** to linearly project the Q-Former's output query embeddings \( Z \) to the same dimension as the LLM's text embeddings, obtaining soft visual prompts $P_Z$.
    *   Prepend the projected query embeddings as **Soft Visual Prompts** to the LLM's input text embeddings.
    *   **Training Objective:** Train the Q-Former (and the FC layer) so that its output visual representations can be understood by the frozen LLM and used for text generation.
        *   For **Decoder-only LLMs (e.g., OPT):** Use standard language modeling loss, i.e., generate subsequent text based on the visual prompt.
        *   For **Encoder-Decoder LLMs (e.g., FlanT5):** Use prefix language modeling loss, splitting the text into prefix and suffix. The visual prompt and prefix are input to the Encoder, and the Decoder generates the suffix.
        $$L_{Stage2} = L_{LM} = -\mathbb{E}_{(I, T_{prompt}, T_{gen})\sim D} \sum_{k=1}^{M} \log P_{LLM}(w_k | P_Z, T_{prompt}, w_{\lt k}; \theta_{LLM\_frozen})$$
        where $\theta_{L L M_{-} \text {frozen }}$ are the parameters of the frozen LLM, used only for forward propagation and not involved in gradient updates.

{{< figure
    src="blip2_stage2.png"
    caption="Fig. 10. BLIP-2's second-stage vision-to-language generative pre-training, which bootstraps from frozen large language models (LLMs). (Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5). (Image source: [Li et al., 2023](https://arxiv.org/abs/2301.12597))"
    align="center"
    width="90%"
>}}

**Effects and Advantages:**

*   **Efficiency:** Since only the lightweight Q-Former is trained, the pre-training cost is much lower than end-to-end training of large models.
*   **High Performance:** Achieved SOTA levels on tasks like VQA, Captioning, and Retrieval, even surpassing models with significantly more parameters (like Flamingo).
*   **Versatility:** Can easily connect to different frozen image encoders and LLMs, leveraging the latest advancements in respective fields.
*   **Zero-Shot Capability:** Leveraging powerful frozen LLMs (especially instruction-tuned FlanT5), BLIP-2 demonstrated impressive **zero-shot instruction-based image-to-text generation** capabilities, performing various visual-language tasks based on natural language instructions (e.g., visual dialogue, visual knowledge reasoning).

### LLaVA

**LLaVA (Large Language and Vision Assistant)** ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)) is a significant work in the open-source community for **Visual Instruction Tuning**, being the first to attempt extending the instruction tuning concept from NLP to the multimodal domain.

**Core Contributions:**

1.  **Proposed Visual Instruction Tuning:** Explored applying instruction tuning to language-image multimodal models, aiming to build general-purpose visual assistants.
2.  **GPT-Assisted Data Generation:** Facing the lack of visual instruction data, innovatively used a **language-only model GPT-4** (or ChatGPT) to generate multimodal language-image instruction-following data containing visual content.
3.  **Built LLaVA Model:** Proposed an end-to-end trained architecture connecting a pre-trained visual encoder (CLIP ViT-L/14) and a large language model (LLM, Vicuna).
4.  **Created Evaluation Benchmark:** Constructed LLaVA-Bench, comprising diverse and challenging tasks to evaluate the instruction-following capabilities of multimodal models.
5.  **Open Source Contribution:** Released the GPT-4 generated visual instruction data, model code, and pre-trained weights, greatly promoting community research in this direction.

**GPT-Assisted Visual Instruction Data Generation:**

The key challenge LLaVA addressed was the lack of large-scale, high-quality visual instruction-following data. The researchers proposed a method using existing multimodal large models like GPT-4 to generate such data based on existing image-text pairs, essentially a form of **knowledge distillation** from the closed-source GPT-4 model.

1.  **Challenge Faced:** Simply extending image-caption pairs into the format (Instruction: Describe the image, Image -> Answer: Caption) is cheap but lacks diversity in instructions and responses, as well as deep reasoning.
2.  **Solution:** Use GPT-4 as a "teacher model". Since these models only accept text input, the researchers conveyed image content through **Symbolic Representations**:
    *   **Captions:** Provide overall or multi-faceted descriptions of the image scene.
    *   **Bounding Boxes:** Provide class concepts and spatial location information of objects in the image (e.g., `person: [0.681, 0.242, 0.774, 0.694]`).
3.  **Prompting and In-context Learning:** Input the symbolic representations (descriptions and bounding boxes) of the image to GPT-4. To guide GPT-4 to generate output in specific formats and content, the researchers manually designed a small number of high-quality **Seed Examples**, leveraging GPT-4's **In-context Learning** ability for few-shot inference.
4.  **Generating Three Types of Data (based on COCO images):** Through carefully designed prompts, GPT-4 was guided to generate three types of instruction data:
    *   **Conversation:** Generate multi-turn dialogues simulating interaction between a human and an assistant about image content, including questions about object recognition, counting, localization, actions, relationships, etc.
    *   **Detailed Description:** Generate comprehensive, detailed descriptions of the image based on specific instructions (e.g., "Describe the image below in detail").
    *   **Complex Reasoning:** Generate questions and answers requiring logical reasoning based on image content or combined with background knowledge (e.g., "What challenges might the person in the picture be facing?").

{{< figure
    src="llava_instruction_data.png"
    caption="Fig. 11. One example to illustrate the instruction-following data. (Image source: [Liu et al., 2023](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}

5.  **Dataset:** A total of **158K** unique language-image instruction samples were collected, specifically including: **58K** conversation samples, **23K** detailed description samples, and **77K** complex reasoning samples. Experiments found that data generated by GPT-4 was generally of higher quality than that from ChatGPT.

{{< figure
    src="llava_architecture.png"
    caption="Fig. 12. LLaVA network architecture. (Image source: [Liu et al., 2023](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}

LLaVA's architecture is designed to effectively combine the capabilities of pre-trained visual models and LLMs, as shown in the figure above.

1.  **Visual Encoder \( g(\cdot) \):** Uses a **frozen CLIP ViT-L/14** model. For an input image \( \mathbf{X}_{\mathrm{v}} \), it extracts visual features \( \mathbf{Z}_{\mathrm{v}} = g(\mathbf{X}_{\mathrm{v}}) \). The paper mentions experimenting with grid features from before and after the last Transformer layer.

2.  **Projection Layer:** Uses a **trainable linear projection matrix \( \mathbf{W} \)** to map the visual features \( \mathbf{Z}_{\mathrm{v}} \) into the word embedding space of the language model.
    $$
    \mathbf{H}_{\mathrm{v}} = \mathbf{W} \cdot \mathbf{Z}_{\mathrm{v}}
    $$
    Here, \( \mathbf{H}_{\mathrm{v}} \) is a sequence of visual tokens whose dimension matches the LLM's word embedding dimension. This simple linear projection is lightweight and efficient, facilitating rapid data-centric experiments. More complex connection methods (like gated cross-attention in Flamingo or Q-Former in BLIP-2) could be explored in future work.

3.  **Large Language Model (LLM) \( f_{\phi}(\cdot) \):** Uses **Vicuna**, with its parameters denoted as \( \phi \). The LLM receives the visual tokens \( \mathbf{H}_{\mathrm{v}} \) and the text instruction \( \mathbf{X}_{\text{instruct}} \), and autoregressively generates the answer \( \mathbf{X}_{\mathrm{a}} \).

**Two-Stage Training:**

LLaVA employs a two-stage instruction fine-tuning process.

1.  **Stage 1: Feature Alignment Pre-training:**
    *   **Goal:** Align the visual features \( \mathbf{H}_{\mathrm{v}} \) with the LLM's word embedding space, which can be viewed as training a compatible "visual tokenizer" for the frozen LLM.
    *   **Data:** Used a filtered subset of the CC3M dataset (approx. 595K image-text pairs). These pairs were simply converted into instruction data: for an image \( \mathbf{X}_{\mathrm{v}} \), randomly select a simple descriptive instruction \( \mathbf{X}_{\mathrm{q}} \) (e.g., "Briefly describe this picture"), and use the original caption \( \mathbf{X}_{\mathrm{c}} \) as the answer \( \mathbf{X}_{\mathrm{a}} \). This can be considered a single-turn conversation.
    *   **Training:** **Freeze** the weights of the visual encoder \( g(\cdot) \) and the LLM \( f_{\phi}(\cdot) \), and **only train** the projection layer \( \mathbf{W} \). The training objective is to maximize the likelihood of the answer (i.e., the image caption).

2.  **Stage 2: Fine-tuning End-to-End:**
    *   **Goal:** Enhance the model's instruction-following and conversational abilities on multimodal tasks.
    *   **Data:** Use the previously generated **158K** visual instruction data (including conversation, detailed description, and complex reasoning types, sampled uniformly during training).
    *   **Training:** **Freeze** the visual encoder \( g(\cdot) \), and **train both** the projection layer \( \mathbf{W} \) and the **LLM \( f_{\phi}(\cdot) \) weights**.

**Training Objective:**

For each image \( \mathbf{X}_{\mathrm{v}} \), multi-turn dialogue data \( \left(\mathbf{X}_{\mathrm{q}}^{1}, \mathbf{X}_{\mathrm{a}}^{1}, \cdots, \mathbf{X}_{\mathrm{q}}^{T_{turns}}, \mathbf{X}_{\mathrm{a}}^{T_{turns}}\right) \) containing \( T_{turns} \) turns is generated, where \( T_{turns} \) is the total number of conversation turns. This data is organized into a sequence, and all answers \( \mathbf{X}_{\mathrm{a}} \) are treated as the model's responses. The input sequence format follows the Vicuna style. In the \( t \)-th turn of the conversation, the instruction \( \mathbf{X}_{\text{instruct}}^{t} \) is defined as:

$$
\mathbf{X}_{\text{instruct}}^{t} = \left\{ \begin{array}{ll} \text{Randomly choose } [\mathbf{X}_{\mathrm{q}}^{1}, \mathbf{X}_{\mathrm{v}}] \text{or } [\mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\mathrm{q}}^{1}], & \text{ if } t=1 \text{ (the first turn)} \\ \mathbf{X}_{\mathrm{q}}^{t}, & \text{ if } t>1 \text{ (the remaining turns)} \end{array} \right.
$$

The objective is to predict the answer sequence \( \mathbf{X}_{\mathrm{a}} = (\mathbf{X}_{\mathrm{a}}^{1}, \dots, \mathbf{X}_{\mathrm{a}}^{T_{turns}}) \). The model needs to maximize the probability of generating the correct answer sequence given the image \( \mathbf{X}_{\mathrm{v}} \) and all instructions \( \mathbf{X}_{\text{instruct}} = (\mathbf{X}_{\text{instruct}}^{1}, \dots, \mathbf{X}_{\text{instruct}}^{T_{turns}}) \). For the complete answer sequence of length \( L_{seq} \) (concatenation of all \( \mathbf{X}_{\mathrm{a}} \) turns), the probability is calculated as follows:

$$
p\left(\mathbf{X}_{\mathrm{a}} \mid \mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\text {instruct }}\right)=\prod_{i=1}^{L_{seq}} p_{\boldsymbol{\theta}}\left(x_i \mid \mathbf{X}_{\mathrm{v}}, \mathbf{X}_{\text {instruct },\lt i}, \mathbf{X}_{\mathrm{a},\lt i}\right)
$$

where:
*   \( \boldsymbol{\theta} \) are the trainable parameters of the model.
    *   In Stage 1, \( \boldsymbol{\theta} = \{ \mathbf{W} \} \).
    *   In Stage 2, \( \boldsymbol{\theta} = \{ \mathbf{W}, \phi \} \).
*   \( x_i \) is the \( i \)-th token in the answer sequence \( \mathbf{X}_{\mathrm{a}} \).
*   \( \mathbf{X}_{\text{instruct},\lt i} \) and \( \mathbf{X}_{\mathrm{a},\lt i} \) represent all instruction tokens and generated answer tokens received by the model before predicting \( x_i \).
*   The training loss function is the **Negative Log-Likelihood** of the above probability, and the loss is calculated **only on the answer tokens** (i.e., tokens in \( \mathbf{X}_{\mathrm{a}} \)).

**Effects and Impact:**

LLaVA demonstrated impressive capabilities in multimodal dialogue, sometimes exhibiting behavior similar to multimodal GPT-4 on unseen images and instructions. After fine-tuning on the ScienceQA benchmark, the combination of LLaVA and GPT-4 achieved a state-of-the-art accuracy of 92.53% at the time.

{{< figure
    src="llava_science_qa_accuracy.png"
    caption="Fig. 13. Accuracy (%) on Science QA dataset. (Image source: [Liu et al., 2023](https://arxiv.org/abs/2304.08485))"
    align="center"
    width="100%"
>}}

LLaVA's success proved the effectiveness of visual instruction tuning. Its open-sourced data, code, and model greatly facilitated subsequent research on multimodal large models, paving new ways for building general-purpose AI assistants capable of understanding and following visual and language instructions.

### Qwen-VL

The **Qwen-VL** ([Bai et al., 2023](https://arxiv.org/abs/2308.12966)) model is the first open-source large vision-language model developed by the Qwen team. Its architecture consists of three main modules:

*   **Large Language Model**: Uses the pre-trained Qwen-7B text model as the language decoder. This part is responsible for understanding and generating text, consistent with standard LLM architectures.
*   **Visual Encoder**: Employs a Vision Transformer to extract image features. Specifically, Qwen-VL initializes the visual encoding part using the ViT-bigG model from [OpenCLIP](https://github.com/mlfoundations/open_clip). During training and inference, input images are resized to a specific resolution. The visual encoder extracts a set of image features by dividing the image into patches with a stride of 14.

*   **Position-aware Vision-Language Adapter**: To efficiently fuse long sequences of image features, an adapter is introduced to compress the visual feature sequence to a fixed length. Specifically, this adapter contains a set of randomly initialized **learnable query vectors**. It computes with the image features output by ViT through a single-layer **cross-attention** module, compressing the image features into a sequence of fixed length 256.

The attention calculation formula is as follows:

$$
\text{CrossAttn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

where \(Q\) is the matrix of trainable query vectors defined within the adapter, and both \(K\) and \(V\) directly use the image feature sequence output by the visual encoder (ViT) as keys and values.

Through this mechanism, the adapter can select and aggregate the most relevant information from numerous image features based on the learned query vectors. Furthermore, to mitigate the potential loss of spatial position information during image feature compression, **2D absolute position encodings** are additionally incorporated into the query-key pairs in the attention calculation, enhancing the perception of the image's spatial structure.

{{< figure
    src="qwen_vl_pipeline.png"
    caption="Fig. 14. The training pipeline of the Qwen-VL series. (Image source: [Bai et al., 2023](https://arxiv.org/abs/2308.12966))"
    align="center"
    width="100%"
>}}

Qwen-VL adopts a "three-stage" progressive training strategy to inject visual perception capabilities into the general large model. The first stage freezes the LLM and trains only the visual modules; the second stage unfreezes and performs joint multi-task training; the third stage involves instruction fine-tuning to obtain the chat model. In the figure above, the snowflake ❄ indicates frozen components, and the flame 🔥 indicates components participating in training.

**Training Strategy:** The Qwen-VL series employs a **three-stage** progressive training flow:

1.  **Pure Image-Text Pre-training Stage**:
    *   Fix the language model (7B) parameters, training only the visual encoder and VL adapter.
    *   Use approximately 1.4 billion pairs of weakly labeled image-text data (77.3% English, 22.7% Chinese).
    *   Images are uniformly scaled to a lower resolution (e.g., 224×224) for efficiency.
    *   Use autoregressive language modeling to train the model to generate image descriptions.
    *   After about 50,000 steps (1.5 billion samples), preliminary image-text alignment capability is achieved (Qwen-VL).

2.  **Multi-task Joint Training Stage**:
    *   Unfreeze the language model, training it end-to-end together with the visual part.
    *   Increase the input image resolution (e.g., 448×448 or higher).
    *   Incorporate various fine-grained visual tasks (e.g., image captioning, visual question answering, content localization, OCR recognition, etc.), covering 7 major task categories.
    *   Training data mixes datasets from multiple sources, adding about 24.8 million OCR data points and 7.8 million pure text data points.
    *   All task data are randomly mixed for training, with each sample prefixed by a task identifier and padded to a sequence length of 2048.
    *   The model significantly improves capabilities in image understanding, cross-modal retrieval, localization, reading, etc.

3.  **Supervised Fine-tuning (SFT) Stage**:
    *   Fine-tune on multimodal instruction data (approx. 350K samples) to obtain the dialogue-enhanced version, Qwen-VL-Chat.
    *   Specifically design complex data for multi-image reasoning, fine-grained localization, and multi-turn interaction tasks.
    *   During fine-tuning, freeze the visual encoder again, fine-tuning only the language model and the adapter.
    *   The final model exhibits excellent multimodal dialogue, instruction following, and complex reasoning abilities.

### Qwen2-VL

**Qwen2-VL** ([Wang et al., 2024](https://arxiv.org/abs/2409.12191)) is an upgraded version of Qwen-VL, making advancements in handling variable-resolution visual inputs and fusing multimodal positional information.

{{< figure
    src="qwen2_vl.jpg"
    caption="Fig. 15. Qwen2-VL is capable of accurately identifying and comprehending the content within images, regardless of their clarity, resolution, or extreme aspect ratios.: ([Wang et al., 2024](https://arxiv.org/abs/2409.12191))"
    align="center"
    width="100%"
>}}

As seen in the figure above, Qwen2-VL can accurately identify and understand content within images of varying resolutions and aspect ratios. It primarily employs the following techniques:

*   **Native Dynamic Resolution:** Inspired by **NaViT** ([Dehghani et al., 2023](https://arxiv.org/abs/2307.06304)), the model can process images of arbitrary resolutions and dynamically convert them into variable-length visual token sequences.
    *   Removes absolute position embeddings from ViT and introduces **2D Rotary Position Embedding (2D-RoPE)** ([Su et al., 2024](https://arxiv.org/abs/2104.09864); [Su, 2021](https://spaces.ac.cn/archives/8397)) to encode 2D spatial information.
    *   During inference, variable-resolution images are processed in packed batches, limiting the total token length to manage memory usage.
    *   After ViT output, an MLP compresses adjacent \( 2 \times 2 \) tokens into one, reducing the sequence length input to the LLM. Visual tokens are wrapped with `<|vision_start|>` and `<|vision_end|>`.

*   **Multimodal Rotary Position Embedding (M-RoPE):** Proposed a novel position embedding method that can uniformly handle positional information for text, images, and videos.
    *   Decomposes RoPE into three components: **Temporal**, **Height**, and **Width**.
    *   **Text:** All three components use the same position ID, equivalent to 1D-RoPE.
    *   **Image:** Temporal ID is constant; Height and Width IDs are assigned based on the token's 2D position in the image.
    *   **Video:** Temporal ID increases with frame number; Height and Width IDs are assigned as in images.
    *   **Multimodal Input:** Position IDs for different modalities increase sequentially.
    *   **Advantage:** Uniformly encodes multimodal positional information, reduces the magnitude of image/video position IDs, facilitating extrapolation to longer sequences during inference.

{{< figure
    src="mrope.png"
    caption="Fig. 16. Illustration of M-RoPE. By decomposing rotary embedding into temporal, height, and width components, M-RoPE can explicitly model the positional information of text, images, and video in LLM. (Image source: [Wang et al., 2024](https://arxiv.org/abs/2409.12191))"
    align="center"
    width="100%"
>}}

*   **Unified Image and Video Understanding:** Adopts a mixed training paradigm and specific architectural designs (like 3D convolution for video processing) to handle both images and videos simultaneously.
    *   Trains on a mixture of image and video data.
    *   Videos are sampled at 2 FPS.
    *   Integrates **3D convolution** in ViT to process video input (handling \( 2 \times 14 \times 14 \) 3D patches), reducing the number of tokens.
    *   Images are treated as two identical video frames.
    *   Dynamically adjusts video frame resolution, limiting the total number of tokens per video segment (e.g., to 16384).

**Training:** Follows Qwen-VL's three-stage training: ViT pre-training -> Full model pre-training -> LLM instruction fine-tuning. Pre-training data includes image-text pairs, OCR, interleaved image-text documents, VQA, video dialogue, image knowledge, etc. Instruction fine-tuning uses the ChatML format. Released models in 2B, 8B, and 72B sizes, exploring the scaling laws of MLLMs.

**Effect:** Qwen2-VL demonstrates outstanding performance in understanding images of various resolutions and aspect ratios, long video understanding (over 20 minutes), and visual agent capabilities.

### Qwen2.5-VL

**Qwen2.5-VL** ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)) further optimizes efficiency and temporal modeling capabilities based on Qwen2-VL.

{{< figure
    src="qwen2.5vl_arc.jpeg"
    caption="Fig. 17. The Qwen2.5-VL framework demonstrates the integration of a vision encoder and a language model decoder to process multimodal inputs. The vision encoder is designed to handle inputs at their native resolution and supports dynamic FPS sampling. TMRoPE aligns time IDs with absolute time along the temporal dimension. (Image source: [Bai et al., 2025](https://arxiv.org/abs/2502.13923))"
    align="center"
    width="100%"
>}}

**Model Optimization:**

Qwen2.5-VL incorporates several optimizations over Qwen2-VL, primarily including:

1.  **Efficient ViT Architecture:** Introduces **Window Attention** mechanism in the Vision Transformer, restricting attention computation in most layers to local windows (e.g., $8 \times 8$ patches). This makes the computational complexity grow linearly with the number of image patches, significantly improving efficiency for high-resolution image processing. Meanwhile, global attention is performed only in a few layers (e.g., every 8 layers) to retain overall context information.

2.  **Dynamic FPS Sampling & Video Processing:** Introduces **Dynamic FPS (Frames Per Second) sampling** mechanism, extending the dynamic resolution concept to the temporal dimension, enhancing the model's adaptability to videos with varying frame rates. For video processing, it maintains the 3D patch structure ($2 \times 14 \times 14$) and combines dynamic FPS with time-aware encoding to optimize overall temporal modeling.

3.  **Stronger Data & Task Capability Support:** The model is pre-trained and fine-tuned on large-scale (4.1T tokens), high-quality datasets, with a focus on enhancing **document parsing** (tables, charts, formulas, sheet music, etc.), **object localization** (supporting point and box annotations), **long video understanding (hour-level)**, and **Agent multi-task capabilities**, broadening the application boundaries of multimodal understanding.

**Data Augmentation:**
*   **Full Document Parsing Data:** Constructed HTML-formatted data containing tables, charts, formulas, images, sheet music, chemical formulas, including layout bounding boxes and coordinates.
*   **Localization Data:** Expanded bounding box and point localization data covering over 10,000 categories, and synthesized hard examples containing non-existent objects and multiple instances of objects. Tools like Grounding DINO and SAM were used for data synthesis.
*   **OCR Data:** Increased multilingual OCR data (covering major European languages, Japanese, Korean, Arabic, Vietnamese, etc.), including various scenarios like handwriting, dense text, web pages, formulas, charts, and tables.
*   **Video Data:** Added dense captioning data for long videos (over half an hour) and trained using dynamic FPS sampling. Timestamp annotations include both seconds and HMSF formats.
*   **Agent Data:** Collected screenshots and action trajectories from mobile, web, and desktop environments, unified into a function call format, and synthesized CoT reasoning processes.

**Effect:** Qwen2.5-VL achieved SOTA performance on document understanding, fine-grained localization, long video understanding, and Agent tasks. The 72B version rivals or even surpasses GPT-4o and Claude 3.5 Sonnet on several benchmarks.

### Qwen2.5-Omni

{{< figure
    src="qwen2.5_omni.png"
    caption="Fig. 18. Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="100%"
>}}

**Qwen2.5-Omni** ([Qwen Team, 2025](https://arxiv.org/abs/2504.14786)) is an end-to-end multimodal model similar to GPT-4o ([OpenAI, 2024](https://openai.com/index/hello-gpt-4o/)), supporting input processing across all modalities including text, image, audio, and video, and capable of simultaneously **streaming text and natural speech** output.

As shown in the figure below, Qwen2.5-Omni adopts the **Thinker-Talker** architecture, with key features including:

{{< figure
    src="qwen2.5_omini_arc.png"
    caption="Fig. 19. Qwen2.5-Omni Overview. Adopts Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receiving high-level representations directly from Thinker. (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="80%"
>}}

1.  **Unified Multimodal Processing and Temporal Modeling:**

    *   **Omni-modal Perception:** A single model can simultaneously process text, image, audio, and video inputs, achieving unified multimodal understanding.

    {{< figure
    src="TMRoPE.png"
    caption="Fig. 20. An illustration of Time-aligned Multimodal RoPE (TMRoPE). (Image source: [Qwen Team, 2025](https://arxiv.org/abs/2504.14786))"
    align="center"
    width="100%"
    >}}

    *   **Time-aligned Multimodal RoPE (TMRoPE):** Further optimizes TMRoPE based on Qwen2.5-VL. Through a **Time-interleaving** structure, video and audio frames are chunked every 2 seconds and arranged chronologically, with video preceding audio within each chunk. All modalities are aligned using absolute timestamps (40ms granularity) and position encoding (TMRoPE), enabling precise audio-video synchronization.

    *   **Input Processing Details:** Text uses the Qwen tokenizer; audio is sampled at 16kHz, represented as 128-channel Mel spectrograms (25ms window, 10ms step), with each frame approx. 40ms, processed by the Qwen2-Audio encoder; images/videos are processed via Qwen2.5-VL's ViT architecture, with video supporting dynamic FPS sampling.

2.  **Thinker-Talker Architecture Design and Functional Decoupling:**

    *   Proposes the innovative Thinker-Talker architecture, decoupling text generation and speech generation to avoid mutual interference while allowing end-to-end joint training.
    *   **Thinker:** Based on Qwen2.5's Transformer decoder, processes multimodal input, generates high-level hidden representations (containing semantic and prosodic information) and text token outputs.
    *   **Talker:** A dual-track autoregressive Transformer decoder, receives hidden representations and text tokens from the Thinker, combined with the ability to disambiguate speech, autoregressively generates discrete speech tokens.
    *   Thinker and Talker share historical context, supporting end-to-end training, enhancing speech generation consistency and context retention.

3.  **Efficient Streaming Capability:**

    *   **Input Streaming:** Audio and visual encoders use **Block-wise Processing**, supporting streaming input and Prefilling.
    *   **Output Streaming:**
        *   Discrete speech tokens generated by the Talker are fed in real-time to a **Streaming Audio Codec**.
        *   The codec employs **Sliding Window Block Attention** (looking back 2 blocks, looking ahead 1 block) based on a **Diffusion Transformer (DiT)** to control the receptive field, enabling streaming generation.
        *   Uses **Flow Matching** ([Lipman et al., 2022](https://arxiv.org/abs/2210.02747)) to convert discrete tokens into Mel spectrograms, which are then streamed into an improved **BigVGAN** ([Lee et al., 2022](https://arxiv.org/abs/2206.04658)) to convert spectrograms into audio waveforms, effectively reducing first-packet latency and improving generation real-time performance.

**Training:** Consists of three stages: Encoder-LLM alignment -> Full model multimodal pre-training -> Long context pre-training (32k). The Talker undergoes separate three-stage training: Context learning -> DPO (optimizing stability) -> Multi-speaker instruction fine-tuning (improving naturalness).

**Effect:** Qwen2.5-Omni performs comparably or better than Qwen2.5-VL (vision) and Qwen2-Audio (audio) of similar scale on respective unimodal benchmarks. Achieves SOTA on multimodal fusion benchmarks like OmniBench. Speech instruction following capability is close to text instructions. Speech generation robustness and naturalness surpass most existing models.

### Kimi-VL

**Kimi-VL** ([Kimi Team, 2025](https://arxiv.org/pdf/2504.07491)) is an open-source **efficient Mixture-of-Experts (MoE)** vision-language model.

{{< figure
    src="kimi_vl_arch.png"
    caption="Fig. 21. Model architecture of Kimi-VL and Kimi-VL-Thinking, consisting of a MoonViT that allows native-resolution images, an MLP projector, and a Mixture-of-Experts (MoE) language decoder. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.07491))"
    align="center"
    width="100%"
>}}

**Architecture Details:**

1.  **Efficient MoE Architecture:**
    The language model part uses an MoE architecture (based on Moonlight, similar to DeepSeek-V3 architecture), with a total of **16B** parameters, activating only **2.8B** parameters per inference (e.g., activating 2/8 experts per layer). This significantly reduces computational cost while maintaining model performance. Supports a maximum context window of **128K tokens**, suitable for long documents, long videos, etc.

2.  **Native Resolution Vision Encoder (MoonViT):**
    Proposes a **400M** parameter vision encoder, MoonViT, supporting **native resolution processing** for images without scaling or padding, maximally preserving image details. The architecture is based on ViT and incorporates the following techniques:
    *   **NaViT Patch n' Pack strategy**: Enables efficient batch processing of variable-length image sequences.
    *   **Interpolated Absolute Position Embeddings**: Initialized from **SigLIP** ([Zhai et al. 2023](https://arxiv.org/abs/2303.15343)), enhancing positional awareness.
    *   **2D Rotary Position Embeddings (2D-RoPE)**: Enhances spatial structure understanding.
    *   **Dynamic Resolution Training**: Samples images of different sizes during training to improve generalization.

3.  **Multimodal Fusion Module:**
    Image features output by MoonViT pass through a **two-layer MLP Projector** containing a **Pixel Shuffle operation** for spatial compression and format conversion. They are then concatenated with text token-level features and input into the MoE LLM for image-text fusion processing.

4.  **Long Chain-of-Thought Reasoning (Kimi-VL-Thinking):**
    Based on the main model, a long-chain thinking training process, including **Supervised Fine-Tuning (SFT) with long Chain-of-Thought (CoT)** and **Reinforcement Learning (RL) optimization**, enhances the model's performance in multi-turn, multi-step reasoning tasks, supporting complex logical Q&A and scene understanding.

**Training:**

{{< figure
    src="kimi_vl_pretrain.png"
    caption="Fig. 22. The pre-training stages of Kimi-VL and Kimi-VL-Thinking, including ViT pre-training, joint pre-training, joint cooling, and joint long-context activation. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.07491))"
    align="center"
    width="100%"
>}}

*   **Pre-training (4 stages, 4.4T tokens total):**
    1.  **ViT Training (2.1T):** Train MoonViT separately (initialized from SigLIP), using contrastive loss SigLIP and cross-entropy caption generation.
        $$
        \mathcal{L}=\mathcal{L}_{\text {siglip }}+\lambda \mathcal{L}_{\text {caption }}, \text { where } \lambda=2
        $$
    2.  **Joint Pre-training (1.4T):** Jointly train ViT, Projector, LLM (initialized from Moonlight 5.2T checkpoint), mixing text and multimodal data.
    3.  **Joint Cooling (0.6T):** Continue joint training with high-quality text and multimodal data.
    4.  **Joint Long Context Activation (0.3T):** Expand context from 8K to 128K using long text, long video, and long document data.

{{< figure
    src="kimi_vl_post_training.png"
    caption="Fig. 23. The post-training stages of Kimi-VL and Kimi-VL-Thinking, including two stages of joint SFT in 32K and 128K context, and further long-CoT SFT and RL stages to activate and enhance long thinking abilities. (Image source: [Kimi Team, 2025](https://arxiv.org/abs/2504.07491))"
    align="center"
    width="100%"
>}}

*   **Post-Training:**
    1.  **Joint SFT:** Use ChatML format, fine-tune on mixed text and multimodal instruction data (first 32K then 128K context).
    2.  **(Kimi-VL-Thinking) Long CoT SFT:** Perform SFT using a small amount of high-quality long CoT data to activate long-chain reasoning capabilities.
    3.  **(Kimi-VL-Thinking) Reinforcement Learning (RL):** Employ the same Online Policy Mirror Descent algorithm used for the **KIMI K1.5** model ([Kimi Team, 2025](https://arxiv.org/abs/2501.12599)) for training. This stage aims to further enhance the model's complex reasoning and planning abilities (e.g., error identification, backtracking, solution optimization) through reinforcement learning, enabling it to utilize long Chain-of-Thought (long CoT) context for implicit search, thereby approximating the effectiveness of explicit planning algorithms while maintaining the simplicity of autoregressive generation.
        *   **Core Objective:** Optimize the policy model $\pi_{\theta}$ such that for a question $x \in \mathcal{D}$, the generated chain-of-thought $z$ and final answer $y$ maximize the expected reward based on the ground truth answer $y^*$:
            $$
            \max _{\theta} \mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D},(y, z) \sim \pi_{\theta}}\left[r\left(x, y, y^{*}\right)\right]
            $$
            where $r(x, y, y^*)$ is typically a correctness reward of 0 or 1.

        *   **Reward Mechanism:**
            *   **Correctness Reward ($r$):** Primarily based on the correctness of the final answer $y$, judged according to the task type:
                *   For **programming** problems: Judged by running automatically generated test cases.
                *   For **math** problems: Evaluated using a high-precision Chain-of-Thought reward model (Chain-of-Thought RM, with 98.5% accuracy).
                *   For **visual** problems: Utilizes diverse data sources like real-world images, synthetic visual reasoning data, and text-rendered images, defining rewards based on task objectives.
            *   **Length Penalty:** To address "overthinking" and improve token efficiency, an additional length reward $\text{len\_reward}(i)$ is introduced. For a question $x$, sample $k$ responses $(y_i, z_i)$ ($i=1, \dots, k$) from the current policy. Let $\text{len}(i)$ be the token length of response $i$, $\text{min\_len} = \min_i \text{len}(i)$, and $\text{max\_len} = \max_i \text{len}(i)$. If $\text{max\_len} > \text{min\_len}$, the length reward is:
                $$
                \text{len_reward}(i) = \begin{cases} \lambda & \text{if } r(x, y_i, y^*) = 1 \\ \min(0, \lambda) & \text{if } r(x, y_i, y^*) = 0 \end{cases}
                $$
                where the length penalty factor $\lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}}$. The final total reward used for optimization is a weighted sum of the correctness reward and the length reward. This penalty is introduced gradually (warm-up).

        *   **Training Characteristics:**
            *   **Algorithm:** Based on Online Policy Mirror Descent, the training process is iterative. In iteration $i$, the current model $\pi_{\theta_i}$ is used as the reference policy to optimize the following objective with relative entropy (KL divergence) regularization:
                $$
                \max _{\theta} \mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D}}\left[\mathbb{E}_{(y, z) \sim \pi_{\theta}}\left[r\left(x, y, y^{*}\right)\right]-\tau \operatorname{KL}\left(\pi_{\theta}(x) \| \pi_{\theta_{i}}(x)\right)\right]
                $$
                where $\tau > 0$ controls the regularization strength.
            *   **Optimization:** Actual updates use off-policy data (i.e., sampled from the reference policy $\pi_{\theta_i}$) and approximate gradients. For each question $x$, sample $k$ responses $(y_j, z_j)$ from $\pi_{\theta_i}$, calculate the empirical average reward $\bar{r} = \frac{1}{k}\sum_{j=1}^{k} r(x, y_j, y^*)$ as a baseline. The gradient of the model parameters $\theta$ is approximated as:
                $$
                \frac{1}{k} \sum_{j=1}^{k}\left(\nabla_{\theta} \log \pi_{\theta}\left(y_{j}, z_{j} \mid x\right)\left(r\left(x, y_{j}, y^{*}\right)-\bar{r}\right)-\frac{\tau}{2} \nabla_{\theta}\left(\log \frac{\pi_{\theta}\left(y_{j}, z_{j} \mid x\right)}{\pi_{\theta_{i}}\left(y_{j}, z_{j} \mid x\right)}\right)^{2}\right)
                $$
                This gradient form resembles policy gradient with baseline but includes an $l_2$ regularization term (gradient of the last term) and uses off-policy samples. The **value network is discarded** during training to encourage exploration.
            *   **Sampling Strategy:** To improve training efficiency, a combination is used:
                *   **Curriculum Sampling:** Gradually increase the difficulty of training problems from easy to hard.
                *   **Prioritized Sampling:** Prioritize sampling problems with lower historical success rates $s_i$ with probability proportional to $1-s_i$.

### o3 & o4-mini

OpenAI's **o3** and **o4-mini** ([OpenAI, 2025](https://openai.com/index/introducing-openai-o3-and-o4-mini/)) are the latest iterations of its o-series reasoning models, characterized by **Longer Thinking Time** and **Full Tool Access**.

**Core Contributions:**
1.  **Enhanced Reasoning:** Models are trained to think longer and deeper (akin to CoT or more complex reasoning processes) before responding, significantly improving performance on complex tasks like coding, math, science, and visual perception. o3 achieves SOTA on benchmarks like Codeforces, SWE-bench, and MMMU.

2.  **Full Tool Access:** Models can seamlessly call various tools, such as [Web Search](https://openai.com/index/introducing-chatgpt-search/), [Code Interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter), [GPT‑4o Image Generation](https://openai.com/index/introducing-4o-image-generation/), and [Function Calling](https://platform.openai.com/docs/guides/function-calling) via API. The models are trained to autonomously decide when and how to use these tools to solve problems.

3.  **Multimodal Reasoning:** Models can **directly integrate images into their chain of thought**, enabling deep fusion of visual and textual reasoning, rather than just using images as input. This makes them excel at analyzing charts, diagrams, etc.

4.  **Efficiency vs. Performance Trade-off:** o3 is the current most powerful model, suitable for complex queries; o4-mini is optimized for speed and cost, with fewer parameters, but still performs well on math, coding, and visual tasks, especially adept at using tools (e.g., using a Python interpreter in the AIME competition).

5.  **Large-Scale Reinforcement Learning:** The performance improvements of the o-series models are largely attributed to the application of large-scale reinforcement learning (RL), validating the potential of RL in enhancing reasoning capabilities, with performance scaling with increased compute.

{{< figure
    src="thinking_with_images_static.webp"
    caption="Fig. 24. o3 model demonstrates its multimodal CoT capability by analyzing a user-uploaded image, identifying the ship, and using tools (web search) to find information, ultimately answering the ship's name and its next port of call. (Image source: [OpenAI, 2025](https://openai.com/index/introducing-o3-and-o4-mini/))"
    align="center"
    width="100%"
>}}

**Working Mechanism:**

*   **Longer Thinking Time:** Borrows the idea of "trading compute for performance" ([Snell et al., 2024](https://arxiv.org/abs/2408.03314)), improving performance on complex tasks by increasing computation at inference time (e.g., multiple sampling, using longer reasoning chains, search algorithms like MCTS), which might be more effective than simply increasing model parameters. The models are internally designed to perform multi-step reasoning or more complex computations. Users can adjust the model's thinking time by selecting different **reasoning effort** settings (e.g., o4-mini-high).

*   **Tool Use:** Models learn tool usage strategies through RL or instruction fine-tuning. When faced with a problem, the model will:
    *   **Plan:** Analyze the problem, determine if tools are needed and which ones.
    *   **Execute:** Call the selected tools (e.g., perform a web search for latest information, run code for calculations).
    *   **Integrate:** Incorporate the results returned by the tools into its reasoning process to generate the final answer.
    This process can be multi-turn and iterative; the model can adjust its strategy based on the information returned by tools (e.g., performing a secondary search).
*   **Multimodal Chain-of-Thought (MCoT):** Models can directly reference and analyze image content within their internal reasoning steps, such as identifying data points in a chart, understanding the steps in a flowchart, or interpreting details in a photograph. Interested readers can refer to the **MCoT Survey** ([Wang et al., 2025](https://arxiv.org/abs/2503.12605)) which introduces its extension to scenarios involving various modalities like images, videos, audio, 3D, tables/charts, etc.

**Effect:**

{{< figure
    src="o3_o4_benchmark.png"
    caption="Fig. 25. To highlight visual reasoning improvement versus our previous multimodal models, OpenAI tested o3 and o4-mini on a diverse set of human exams and ML benchmarks. These new visual reasoning models significantly outperform their predecessors on all multimodal tasks we tested. (Image source: [OpenAI, 2025](https://openai.com/index/thinking-with-images/))"
    align="center"
    width="100%"
>}}

o3 and o4-mini demonstrate SOTA or near-SOTA performance on multiple benchmarks, especially on tasks requiring deep reasoning and tool assistance. Expert evaluations show they produce fewer serious errors compared to their predecessors o1/o3-mini, provide more useful and verifiable answers, and interact more naturally.

## Summary

Multimodal Large Language Models are advancing towards being more comprehensive, intelligent, and efficient. They can not only understand and generate content across multiple modalities like text, images, videos, and audio, but also perform complex reasoning, planning, and tool invocation. In the future, we can expect further breakthroughs in MLLMs regarding efficiency optimization, deeper cross-modal fusion and reasoning, stronger temporal and spatial understanding capabilities, as well as safety and controllability.

## References

[1] OpenAI. ["Hello gpt-4o."](https://openai.com/index/hello-gpt-4o/) OpenAI Blog (2024).

[2] DeepMind. ["Gemini 2.5 Pro"](https://deepmind.google/technologies/gemini/pro/) DeepMind Blog (2025).

[3] OpenAI. ["Introducing OpenAI o3 and o4-mini."](https://openai.com/index/introducing-o3-and-o4-mini/) OpenAI Blog (2025).

[4] Zhang, Duzhen, et al. ["Mm-llms: Recent advances in multimodal large language models."](https://arxiv.org/abs/2401.13601) arXiv preprint arXiv:2401.13601 (2024).

[5] Dosovitskiy, Alexey, et al. ["An image is worth 16x16 words: Transformers for image recognition at scale."](https://arxiv.org/abs/2010.11929) arXiv preprint arXiv:2010.11929 (2020).

[6] Radford, Alec, et al. ["Learning transferable visual models from natural language supervision."](https://arxiv.org/abs/2103.00020) International conference on machine learning. PMLR, 2021.

[7] Li, Junnan, et al. ["Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation."](https://arxiv.org/abs/2201.12086) International conference on machine learning. PMLR, 2022.

[8] Li, Junnan, et al. ["Align before fuse: Vision and language representation learning with momentum distillation."](https://arxiv.org/abs/2107.07651) Advances in neural information processing systems 34 (2021): 9694-9705.

[9] Li, Junnan, et al. ["Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models."](https://arxiv.org/abs/2301.12597) International conference on machine learning. PMLR, 2023.

[10] Liu, Haotian, et al. ["Visual instruction tuning."](https://arxiv.org/abs/2304.08485) arXiv preprint arXiv:2304.08485 (2023).

[11] Bai, Jinze, et al. ["Qwen-vl: A frontier large vision-language model with versatile abilities."](https://arxiv.org/abs/2308.12966) arXiv preprint arXiv:2308.12966 1.2 (2023): 3.

[12] Wang, Peng, et al. ["Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution."](https://arxiv.org/abs/2409.12191) arXiv preprint arXiv:2409.12191 (2024).

[13] Dehghani, Mostafa, et al. ["Patch n' pack: NaViT, a vision transformer for any aspect ratio and resolution."](https://arxiv.org/abs/2307.06304) Advances in Neural Information Processing Systems 36 (2023): 2252-2274.

[14] Su, Jianlin, et al. ["Roformer: Enhanced transformer with rotary position embedding."](https://arxiv.org/abs/2104.09864) Neurocomputing 568 (2024): 127063.

[15] Su, Jianlin. ["Transformer升级之路：4、二维位置的旋转位置编码." (Path to Upgrading Transformers: 4. Rotary Position Embedding for 2D Positions)](https://spaces.ac.cn/archives/8397) *科学空间 (Scientific Spaces)* (blog) (2021).

[16] Bai, Shuai, et al. ["Qwen2.5-VL Technical Report."](https://arxiv.org/abs/2502.13923) arXiv preprint arXiv:2502.13923 (2025).

[17] Xu, Jin, et al. ["Qwen2.5-Omni Technical Report."](https://arxiv.org/abs/2503.20215) arXiv preprint arXiv:2503.20215 (2025).

[18] Lipman, Yaron, et al. ["Flow matching for generative modeling."](https://arxiv.org/abs/2210.02747) arXiv preprint arXiv:2210.02747 (2022).

[19] Lee, Sang-gil, et al. ["Bigvgan: A universal neural vocoder with large-scale training."](https://arxiv.org/abs/2206.04658) arXiv preprint arXiv:2206.04658 (2022).

[20] Kimi Team. ["Kimi-VL Technical Report."](https://arxiv.org/abs/2504.07491) arXiv preprint arXiv:2504.07491 (2025).

[21] Zhai, Xiaohua, et al. ["Sigmoid loss for language image pre-training."](https://arxiv.org/abs/2303.15343) Proceedings of the IEEE/CVF international conference on computer vision. 2023.

[22] Kimi Team. ["Kimi k1. 5: Scaling reinforcement learning with llms."](https://arxiv.org/abs/2501.12599) arXiv preprint arXiv:2501.12599 (2025).

[23] Snell, Charlie, et al. ["Scaling llm test-time compute optimally can be more effective than scaling model parameters."](https://arxiv.org/abs/2408.03314) arXiv preprint arXiv:2408.03314 (2024).

[24] Wang, Yaoting, et al. ["Multimodal chain-of-thought reasoning: A comprehensive survey."](https://arxiv.org/abs/2503.12605) arXiv preprint arXiv:2503.12605 (2025).

## Citation

> **Citation**: When reprinting or citing the content of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (May 2025). Multimodal Large Language Models.
> https://syhya.github.io/en/posts/2025-05-04-multimodal-llm/

Or

```bibtex
@article{yue_shui_multimodal_llm_2025,
  title   = "Multimodal Large Language Models",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "May",
  url     = "https://syhya.github.io/en/posts/2025-05-04-multimodal-llm/"
}
```