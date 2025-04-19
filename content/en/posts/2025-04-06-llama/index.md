---
title: "The LLaMA Herd"
date: 2025-04-06T12:00:00+08:00
author: "Yue Shui"
tags: ["LLaMA", "AI", "NLP", "LLM", "Pre-training", "Post-training"]
categories: ["Technical Blog"]
readingTime: 40
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
math: true
---

## LLaMA

The LLaMA series of open-source models released by Meta AI has become one of the cornerstones of the large language model community, profoundly impacting the advancement of open research and applications. From the pioneering LLaMA released in early 2023, to the significantly improved LLaMA 2 later that year, to derivative models targeting specific domains (like code, safety), and the subsequent new generations LLaMA 3 and LLaMA 4 launched in 2024 and 2025 respectively, Meta has continuously committed to enhancing the performance of open-source models, gradually bringing them closer to state-of-the-art closed-source models. Below, we will introduce the key technical details of each major model in sequence.

### LLaMA 1

**LLaMA 1** ([Touvron et al., 2023a](https://arxiv.org/abs/2302.13971)), released in February 2023, was Meta's first series of open-source foundation language models. LLaMA was offered in four parameter sizes: 7B, 13B, 33B, and 65B. Its core characteristic was being trained entirely on **publicly available datasets**, without relying on any proprietary data. Despite having a significantly smaller parameter count than the contemporary GPT-3 (175B), the LLaMA 13B model outperformed GPT-3 on most benchmarks, while the 65B model achieved performance comparable to top models like Chinchilla-70B and PaLM-540B.

{{< figure
    src="llama1_bechmark.png"
    caption="Fig. 1. Zero-shot performance of LLaMA models on Common Sense Reasoning tasks compared to other foundation models. (Source: [Touvron et al., 2023a](https://arxiv.org/abs/2302.13971))"
    align="center"
    width="100%"
>}}

**Training Data:** LLaMA 1 was trained on large-scale public corpora. The 65B and 33B models used approximately **1.4 trillion tokens**, while the 7B and 13B models used about **1 trillion tokens**. The corpus sources were diverse, primarily including Common Crawl, C4, GitHub, Wikipedia, Books, ArXiv, and StackExchange, covering multiple domains and about 20 languages (predominantly English).

**Architecture Design:** LLaMA 1 employed a standard Transformer decoder architecture, incorporating the following key improvements to enhance performance and training efficiency:

*   **Pre-normalization & RMSNorm:** Adopted a **Pre-normalization** structure (applying normalization before each sub-layer input) and used **RMSNorm (Root Mean Square Normalization)** instead of standard LayerNorm. RMSNorm reduces computational complexity by omitting the mean centering step, scaling based only on the root mean square of the vector elements, while effectively maintaining training stability.
*   **SwiGLU Activation Function:** Replaced the activation function in the feed-forward network (FFN) from ReLU to **SwiGLU (Swish-Gated Linear Unit)**. SwiGLU combines the smooth non-linearity of the Swish activation function with a gating mechanism, enhancing the model's expressive power. Concurrently, LLaMA adjusted the FFN's hidden layer dimension (using $ \frac{2}{3} \times 4d $ instead of the standard $4d$, where $d$ is the model dimension) to roughly maintain the total parameter count and computational load of the FFN layer while introducing gating parameters.
*   **RoPE Rotary Position Embeddings:** Utilized **Rotary Position Embeddings (RoPE)** for positional encoding. RoPE effectively incorporates relative positional information into self-attention calculations by applying position-dependent rotation operations to Query and Key vectors, enhancing the model's ability to handle long sequences and capture long-range dependencies. LLaMA 1 had a maximum context length of 2048 tokens.
*   **Efficient Attention Implementation:** Leveraged Meta's open-source [xformers](https://github.com/facebookresearch/xformers) library to implement a memory-efficient and computationally optimized causal multi-head attention mechanism.

**Fine-tuned Dialogue Models:** At its release, LLaMA 1 primarily provided pre-trained model weights (with restricted commercial use) and did not include an official dialogue fine-tuned version. However, the open-source community quickly explored its potential. For instance, the **Stanford Alpaca** ([Taori et al., 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html)) project demonstrated that Supervised Fine-tuning (SFT) with only a small amount of instruction data could endow the LLaMA base model with strong conversational abilities, greatly promoting the research and application ecosystem of open-source LLMs.

{{< figure
    src="alpaca.png"
    caption="Fig. 2. The pipeline for generating instruction-following demonstrations and training Alpaca 7B based on LLaMA 7B. (Source: [Taori et al., 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html))"
    align="center"
    width="100%"
>}}

**Training Stability & Loss Spikes**

{{< figure
    src="llama1_train_loss.png"
    caption="Fig. 3. Training loss curves over processed tokens for the LLaMA 7B, 13B, 33B, and 65B models. (Source: [Touvron et al., 2023a](https://arxiv.org/abs/2302.13971))"
    align="center"
    width="80%"
>}}

As observed in Figure 3, the training loss of LLaMA models generally shows a downward trend, indicating relatively stable training. However, during the training of the 13B, 33B, and 65B models, **Loss Spikes** occurred, where the training loss suddenly and abnormally surged at certain points. The larger the model scale, the more pronounced the spike phenomenon seems to be, and it might occur multiple times during training.

*   **Phenomenon Description:** A Loss Spike refers to a brief, sharp, and abnormal increase in the loss function value during model training.
*   **Potential Causes:** Often related to multiple factors, including **anomalous samples or distribution shifts** in the training data, **improper learning rate settings** (too high or issues with decay strategy), **interaction between the optimizer's internal state (like Adam) and drastic gradient changes**, and **numerical instability in mixed-precision training** (e.g., gradient overflow or underflow).
*   **Common Mitigation Strategies:** Methods to resolve or mitigate Loss Spikes include: strengthening data cleaning and preprocessing; applying **Gradient Clipping** to limit the gradient norm; fine-tuning learning rate scheduling strategies (like Warmup, Decay); optimizing mixed-precision training configurations; and, after a spike occurs, resuming training from the nearest checkpoint, possibly skipping the specific data batch that caused the issue.

### LLaMA 2

**LLaMA 2** ([Touvron et al., 2023b](https://arxiv.org/abs/2307.09288)), launched in July 2023, was a significant upgrade to LLaMA 1. Compared to the first generation, LLaMA 2 featured notable improvements in model scale, training data volume, context length, and model alignment. It also marked the first release of an official dialogue-optimized version, **LLaMA 2-Chat**, and came with a license permitting commercial use.

**Architecture & Optimization:** LLaMA 2's base architecture largely inherited the successful design of LLaMA 1 (e.g., RMSNorm, SwiGLU, RoPE). Key technical updates included:

*   **Grouped Query Attention (GQA):** For the larger **34B and 70B models**, **Grouped Query Attention (GQA)** was adopted. GQA is a compromise between Multi-Head Attention (MHA) and Multi-Query Attention (MQA), allowing multiple Query heads to share the same set of Key and Value heads. This significantly reduces the memory footprint and computational overhead of the KV cache during inference, thereby improving the inference speed and deployment efficiency of large models with minimal impact on performance.
*   **Increased Context Length:** The maximum context length of the model was extended from LLaMA 1's 2048 tokens to **4096 tokens**. This enabled the model to process longer text inputs, enhancing its capabilities in tasks like long-document question answering, summarization, and extended conversations.

**Training Data & Scale:** LLaMA 2 was trained on a larger pre-training dataset, totaling approximately **2 trillion tokens**, about a 40% increase compared to LLaMA 1. The data sources were more diverse, and underwent more rigorous filtering and cleaning.

**Post-Training (LLaMA 2-Chat):** **LLaMA 2-Chat** is a meticulously aligned dialogue model. Its training process starts with the LLaMA 2 pre-trained base model and primarily involves the following stages:

1.  **Supervised Fine-tuning (SFT):** The pre-trained model is fine-tuned using high-quality instruction and dialogue samples, initially equipping it with the ability to follow instructions and engage in dialogue.
2.  **Reinforcement Learning from Human Feedback (RLHF):** This is a crucial step for enhancing the model's Helpfulness and Safety.
    *   **Reward Modeling:** A large amount of human preference data is collected (i.e., ranking multiple responses generated by the model). One or more reward models are trained to learn to evaluate which response better aligns with human preferences (along dimensions of helpfulness and safety).
    *   **RL Optimization:** Using the trained reward model(s) as a reward signal, the SFT model is further optimized using **PPO** and **Rejection Sampling**. PPO aims to maximize the reward signal, while rejection sampling further improves model quality by selecting the highest-reward response from K samples generated by the model for gradient updates. This process is typically iterative, continually collecting new preference data to refine the reward model and the dialogue model itself. It drew inspiration from Anthropic's Constitutional AI and HH-RLHF ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). The relevant [HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) is available on Hugging Face.

{{< figure
    src="llama2_chat_rlhf.png"
    caption="Fig. 4. Illustration of the Llama 2-Chat fine-tuning process, including SFT and RLHF stages with rejection sampling and PPO. (Source: [Touvron et al., 2023b](https://arxiv.org/abs/2307.09288))"
    align="center"
    width="100%"
>}}

### Code Llama

**Code Llama** ([Rozière et al., 2023](https://arxiv.org/abs/2308.12950)), released by Meta in August 2023, is a family of large language models built upon LLaMA 2, specifically **focused on coding capabilities**. Through additional continued pre-training on massive amounts of programming code data and specific task fine-tuning, Code Llama demonstrates excellent capabilities in code generation, code completion, code understanding, and debugging.

{{< figure
    src="codellama.png"
    caption="Fig. 5. The Code Llama specialization pipeline, starting from Llama 2 and involving code-specific training stages. (Source: [Rozière et al., 2023](https://arxiv.org/abs/2308.12950))"
    align="center"
    width="100%"
>}}

**Training & Data:** Code Llama started with LLaMA 2 weights and underwent continued pre-training on **500 billion tokens** of code and code-related natural language corpora (for 7B/13B/34B versions) or **1 trillion tokens** (for the 70B version). Training data primarily came from public code repositories and datasets. Key technical improvements include:

*   **Long Context Fine-tuning (LCFT):** Code Llama paid special attention to long sequence processing during training, extending the sequence length to **16k tokens**. To better handle long-range dependencies, the base period $\theta$ of RoPE positional encoding was adjusted (increased from LLaMA 2's 10,000 to 1,000,000), slowing down the decay of attention scores as the token distance increases. This allows the model to stably handle ultra-long contexts of up to **100k tokens** during inference.

{{< figure
    src="codellama_rope.png"
    caption="Fig. 6. Effect of RoPE base period scaling on perplexity for long sequences, showing improved performance with a larger base. (Source: [Rozière et al., 2023](https://arxiv.org/abs/2308.12950))"
    align="center"
    width="70%"
>}}

*   **Fill-in-the-Middle (FIM):** The training incorporated the **Fill-in-the-Middle** task. The model needs to insert appropriate code snippets given a code prefix and suffix. This capability is crucial for code completion features in Integrated Development Environments (IDEs).

**Model Variants:** Code Llama offers several versions to meet the needs of different scenarios:

*   **Code Llama (Base):** The foundational code model, adept at code completion and generating code from natural language.
*   **Code Llama - Python:** Built upon the base model, specialized fine-tuning on an additional 100 billion tokens of Python code significantly enhances performance on Python-related tasks.
*   **Code Llama - Instruct:** Fine-tuned on code-related instructions and human feedback data (approx. 5 billion tokens), enabling it to better understand natural language instructions to generate, explain, or modify code, making it more suitable as a code assistant.

Each version is available in 7B, 13B, 34B, and 70B parameter sizes.

### Llama Guard

**Llama Guard** ([Inan et al., 2023](https://arxiv.org/abs/2312.06674)), introduced by Meta in December 2023, is a model specifically designed for safeguarding the content of human-AI conversations. It aims to perform content review and risk classification for both user inputs (prompts) and model outputs (responses).

{{< figure
    src="llama_guard.png"
    caption="Fig. 7. Example task instructions for the Llama Guard prompt and response classification tasks, demonstrating its safety assessment capability. (Source: [Inan et al., 2023](https://arxiv.org/abs/2312.06674))"
    align="center"
    width="100%"
>}}

**Model Overview:** Llama Guard is based on the LLaMA 2-7B model and is specifically trained via **instruction fine-tuning** for the task of **safety risk classification**. It is not a generative model; instead, it takes a text input, determines if its content is safe, and can output specific risk category labels based on a predefined safety risk taxonomy.

**Training & Taxonomy:** Meta constructed a taxonomy containing various categories of unsafe content (e.g., violence, hate speech, sexual content, promotion of illegal acts) and collected high-quality labeled data for training. Llama Guard can perform multi-label classification, identifying potentially multiple risks present in the text simultaneously. Due to its instruction fine-tuning paradigm, users can flexibly adjust safety policies or customize classification standards by designing different prompts. Llama Guard can be deployed as a filter at the input end (detecting user input risks) and the output end (detecting risks in model-generated content) of a dialogue system.

### Llama Guard 3 Vision

**Llama Guard 3 Vision** ([Chi et al., 2024](https://arxiv.org/abs/2411.10414)) is the multimodal upgraded version of Llama Guard, built upon the **Llama-3.2-Vision** model. It can simultaneously assess the safety risks of both **image and text** content, extending safety protection capabilities to multimodal scenarios. The model uses a special `<|image|>` token to integrate image information for unified multimodal safety review.

{{< figure
    src="llama_guard_vision.png"
    caption="Fig. 8. Llama Guard 3 Vision classifying harmful content in a multimodal response classification task involving both image and text. (Source: [Chi et al., 2024](https://arxiv.org/abs/2411.10414))"
    align="center"
    width="100%"
>}}

Llama Guard 3 Vision adopts the safety risk classification standard defined by ML Commons ([Vidgen et al., 2024](https://arxiv.org/abs/2404.12241)) and expands upon it, adding detection for code interpreter abuse risks (Category S14).

{{< figure
    src="hazard_categories.png"
    caption="Fig. 9. The 14 hazard categories used by Llama Guard 3 Vision, based on the MLCommons taxonomy with an added category for code interpreter abuse. (Source: [Meta Llama, 2024](https://huggingface.co/meta-llama/Llama-Guard-3-8B))"
    align="center"
    width="60%"
>}}

Benchmark results show that Llama Guard 3 Vision outperforms advanced models like GPT-4o and GPT-4o mini on multiple metrics within the MLCommons safety benchmark, both for detecting risks in user inputs and model outputs.

{{< figure
    src="ml_commons_benchmark.png"
    caption="Fig. 10. Performance comparison of various models on the MLCommons hazard taxonomy internal test set, showing Llama Guard 3 Vision's strong results. (Source: [Chi et al., 2024](https://arxiv.org/abs/2411.10414))"
    align="center"
    width="100%"
>}}

### LLaMA 3

**LLaMA 3** ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)) is the new generation of open-source large model series successively released by Meta starting from April 2024. It features optimizations in performance, scale, multilingual capabilities, multimodal support, and training efficiency.

**Model Scale & Version Evolution:** The LLaMA 3 series covers a wide range of parameter scales, from small to ultra-large:

*   **LLaMA 3 (Initial Release, 2024/04):** First released 8B and 70B scale pre-trained and instruction fine-tuned models.
*   **LLaMA 3.1 (2024/07):** ([Meta AI, 2024](https://ai.meta.com/blog/meta-llama-3-1/)) Introduced the **405B** parameter flagship model, with performance approaching GPT-4 levels on multiple benchmarks, along with updated 8B and 70B versions.
*   **LLaMA 3.2 (2024/10):** Introduced **lightweight models** (e.g., 1B, 3B, 11B, 13B) optimized for edge devices (like phones, watches, smart homes), and released **multimodal vision models** (e.g., Llama-3.2-11B-Vision and Llama-3.2-90B-Vision).

{{< figure
    src="llama3_key_hyperparameters.png"
    caption="Fig. 11. Overview of the key hyperparameters for Llama 3 models of different scales. (Source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="70%"
>}}

From the figure above, it can be observed that training larger-scale LLMs typically requires using smaller peak learning rates. This is primarily due to:
1.  **Optimization Landscape Complexity and Gradient Stability:** Larger parameter counts lead to more complex and non-convex loss landscapes, making the model more sensitive to parameter updates. Smaller learning rates help limit the step size of each update, avoiding excessively large gradients in steep regions that could lead to training oscillations or divergence, thus ensuring a more stable convergence process.
2.  **Avoiding Overfitting and Improving Generalization:** Larger models have greater capacity and are more prone to overfitting the training data. Smaller learning rates allow the model to learn patterns in the data more slowly and robustly, reducing the risk of overfitting to noise or local features in the training data, which helps improve generalization performance on unseen data.
3.  **Fine-grained Search and Parameter Adjustment:** In high-dimensional parameter spaces, the optimal solution might reside in narrow regions. Small learning rates enable the optimization algorithm to perform a finer search, gradually approaching the optimum and avoiding "overshooting" the optimal region due to large step sizes, potentially leading to higher final model accuracy.

{{< figure
    src="llama3_architecture.png"
    caption="Fig. 12. Comparison of the high-level architecture between Llama 2 and Llama 3. (Source: [Umar Jamil's PyTorch Llama Slides](https://github.com/hkproj/pytorch-llama/blob/main/Slides.pdf))"
    align="center"
    width="70%"
>}}

**Architecture & Technical Innovations:** LLaMA 3 incorporates several significant enhancements over LLaMA 2:

*   **Massive Pre-training Data:** The pre-training data volume reached a staggering **15 trillion tokens**, 7.5 times that of LLaMA 2. Data sources were broader, of higher quality, more diverse, and significantly increased the proportion of non-English languages (e.g., German, French, Spanish, Hindi, each >5% of total data) and code data.
*   **Optimized Tokenizer:** Employed a new tokenizer implemented based on the `tiktoken` library, with the **vocabulary size drastically expanded from LLaMA 2's 32k to 128k**. The larger vocabulary improves encoding efficiency for multiple languages (especially non-Latin scripts) and code, reducing input sequence length by about 15% on average, thereby indirectly boosting model processing efficiency and performance.
*   **Extended Context Length:** The initial LLaMA 3 release (8B, 70B) supported an 8k token context window. **LLaMA 3.1 (405B) further increased the maximum context window to 128k tokens**, greatly enhancing the model's ability to handle long documents, long conversation histories, and complex contextual reasoning. This is typically achieved through techniques like RoPE frequency adjustments and attention mechanism optimizations (e.g., FlashAttention).
*   **Universally Applied GQA:** Unlike LLaMA 2, which only used GQA in larger models, **all scales of LLaMA 3 models (including 8B) adopted Grouped Query Attention (GQA)** to optimize memory usage and computation speed during inference.
*   **Advanced Alignment Techniques:** During the instruction fine-tuning (Post-training) phase, LLaMA 3 combined multiple advanced techniques, including Supervised Fine-tuning (SFT), Rejection Sampling, and Direct Preference Optimization (DPO), aiming to comprehensively improve the model's instruction-following ability, Helpfulness, and Safety.
*   **Multimodal Integration (LLaMA 3.2):** Introduced a Vision Encoder and performed joint training to achieve fusion processing of images and text, leading to the Llama-3.2-Vision series of vision-language models.
*   **Lightweight Models (LLaMA 3.2):** Targeted resource-constrained edge computing scenarios by introducing smaller models (1B, 3B, etc.) through model compression techniques (like pruning, distillation), achieving a good balance between performance and resource consumption.

{{< figure
    src="llama3_post_training.png"
    caption="Fig. 13. Illustration of the overall post-training approach for Llama 3, involving multiple stages and iterative refinement. (Source: [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))"
    align="center"
    width="100%"
>}}

As shown in the figure above, LLaMA 3's post-training (instruction fine-tuning) process is a carefully designed multi-stage iterative procedure:

1.  **Data Preparation:** Collect large amounts of human preference data. This data typically includes a prompt and multiple model-generated responses, which annotators rank (e.g., selecting the best "chosen" response and a worse "rejected" response). High-quality SFT data (prompt-response pairs) are also collected.
2.  **Reward Modeling (RM):** Utilize the collected human preference data triplets (prompt, chosen, rejected) to train one or more reward models. The goal of the reward model is to learn to predict the degree of human preference for model-generated responses, providing a quantitative signal for subsequent optimization. LLaMA 3 trained two separate reward models focusing on Helpfulness and Safety, respectively.
3.  **Rejection Sampling:** Use the trained reward model(s) to score candidate responses generated by the model. Select the highest-scoring responses as high-quality samples for subsequent fine-tuning stages. This helps filter out samples of higher quality than the initial SFT data.
4.  **Supervised Finetuning (SFT):** Combine the initial human-annotated SFT data with the high-quality data filtered through rejection sampling to fine-tune the pre-trained base model. This stage aims to teach the model the format and style of following instructions and to initially grasp the required knowledge and abilities. LLaMA 3 used a mix of data from various sources in this stage.
5.  **Preference Optimization:** Starting from the SFT model, use the preference data (prompt, chosen, rejected) to further align the model via the **Direct Preference Optimization (DPO)** algorithm. DPO directly optimizes the model to increase the likelihood of the "chosen" response while decreasing the likelihood of the "rejected" response. Compared to RL-based PPO methods, it is simpler to implement and more stable to train. LLaMA 3 made improvements to DPO, such as masking special formatting tokens in the responses during DPO training and introducing a normalized negative log-likelihood (NLL) loss as a regularizer to enhance training stability and generation quality. Its loss function form can be roughly referenced from the loss in **RPO** ([Pang et al., 2024](https://arxiv.org/abs/2404.19733)), though LLaMA3's specific implementation might differ slightly:

    $$
    \begin{aligned}
    \mathcal{L}_{\mathrm{DPO}+\mathrm{NLL}} & =\mathcal{L}_{\mathrm{DPO}}\left(y^w, y^l \mid x\right)+\alpha \mathcal{L}_{\mathrm{NLL}}\left(y^w \mid x\right) \\
    & =-\log \sigma\left(\beta \log \frac{\pi_\theta(y^w \mid x)}{\pi_{\mathrm{ref}}(y^w \mid x)}-\beta \log \frac{\pi_\theta(y^l \mid x)}{\pi_{\mathrm{ref}}(y^l \mid x)}\right)-\alpha \frac{\log \pi_\theta(y^w \mid x)}{|y^w|}
    \end{aligned}
    $$
    Where:
    *   $x$ is the input prompt.
    *   $y^w$ is the preferred (winning/chosen) response, $y^l$ is the dispreferred (losing/rejected) response.
    *   $\pi_\theta$ is the current model policy being optimized (with parameters $\theta$).
    *   $\pi_{\mathrm{ref}}$ is the reference model policy (often the SFT model or the model from the previous iteration).
    *   $\beta$ is a hyperparameter controlling the strength of the preference margin.
    *   $\sigma$ is the Sigmoid function.
    *   $\alpha$ is the weight balancing the DPO loss and the NLL regularization loss.
    *   $|y^w|$ is the length of the winning response, used to normalize the NLL loss.
    This loss function encourages the model $\pi_\theta$ to prefer generating $y^w$ over $y^l$ relative to the reference model $\pi_{\mathrm{ref}}$, while the NLL regularization term helps maintain the fluency and linguistic quality of the generated text.

6.  **Iterative Loop:** The SFT and DPO (or RLHF variant) processes described above are repeated for multiple rounds (LLaMA 3 underwent five rounds). In each round, the model optimized in the previous round is used to generate new data, new human feedback is collected, new reward models are trained, and the next round of SFT and DPO optimization is performed. This iterative approach allows the model to continuously learn and improve.
7.  **Model Weight Averaging:** At certain stages, weight averaging might be performed across multiple model checkpoints trained with different data subsets or hyperparameters to obtain a final model that is more robust and has more balanced performance.

### LLaMA 4

The **LLaMA 4** ([Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) series of models, released by Meta AI on April 5, 2025, marks the LLaMA ecosystem's entry into a new phase of natively multimodal AI innovation. This generation introduces the **Mixture-of-Experts (MoE) architecture** for the first time and possesses unprecedented **ultra-long context processing capabilities**, aiming to provide more powerful and efficient open-source foundation models.

**Model Overview: Performance, Scale & Deployment**

LLaMA 4 initially released three models with different positionings, two of which have open weights:

| Model Name          | Active Params | Num Experts | Total Params | Key Performance/Positioning                                                                                                | Hardware Reference                                   | Context Window |
| :---------------- | :------------ | :---------- | :----------- | :--------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------- | :------------- |
| **Llama 4 Scout** | 17B           | 16          | 109B         | Outperforms peer models like Gemma 3; **10M Token ultra-long context**; Strong image understanding; High cost-performance        | Single H100 GPU (INT4 quantized)                  | **10M**        |
| **Llama 4 Maverick**| 17B           | 128         | 400B         | Matches or surpasses GPT-4o/Gemini 2.0 Flash (reasoning/coding/multilingual); Fewer active params, high computational efficiency; Leading in image reasoning/understanding; LMArena ELO 1417 | Single H100 host (multi-GPU) or distributed deployment | **1M**         |
| **Llama 4 Behemoth**| 288B          | 16          | ~2T          | **Teacher Model (Unreleased)**; Surpasses GPT-4.5/Claude 3.7/Gemini 2.0 Pro on STEM benchmarks (MATH, GPQA); Improves Scout/Maverick via **co-distillation** | Still in training, not publicly released                     | (Not specified)|

*   **Performance Highlights:** Maverick (17B active parameters) demonstrates competitive strength against top-tier closed-source models like GPT-4o on several mainstream benchmarks, especially in reasoning, coding, and multilingual tasks, while having significantly fewer active parameters, reflecting excellent computational efficiency. Scout stands out among its peers with its astounding 10M token context window.
*   **Deployment Threshold:** Scout's INT4 quantized version can run on a single H100, lowering the deployment barrier for high-performance models. Although Maverick requires more compute power (e.g., a single H100 multi-GPU host), it still offers attractive cost-performance relative to its capabilities. *(Note: Running these models on consumer-grade GPUs remains challenging)*

**Core Architecture & Training Innovations**

LLaMA 4 features the following optimizations compared to the previous generation:

1.  **Mixture-of-Experts (MoE) Architecture:**
    *   LLaMA 4 is the first Llama series to adopt MoE. MoE allows the model to activate only a small fraction of its total parameters ("active parameters") during inference, achieving **larger model capacity and stronger performance with lower computational cost**. This is highly beneficial for compute-cost-sensitive (especially throughput-sensitive) inference scenarios.
    *   The Maverick model employs **alternating dense and MoE layers**. Its MoE layers contain 128 routing experts and one shared expert accessed by all tokens. Each token is routed to the shared expert plus one of the routing experts for processing.

2.  **Native Multimodality & Early Fusion:**
    *   **Moving beyond 'stitched' approaches:** Unlike previous methods that "bolted on" visual modules to LLMs using late fusion, LLaMA 4 adopts an **early fusion** strategy from the ground up.
    *   **Unified Backbone:** Text tokens and visual tokens (from image and video frames) are seamlessly integrated and processed together in the early stages of the model's backbone network.
    *   **Deep Understanding:** This enables joint pre-training on massive amounts of image-text and video data, allowing the model to learn deeper, more fine-grained cross-modal associations, achieve more natural interaction, and exhibit stronger **visual grounding** capabilities (accurately mapping text prompts to image regions), going beyond simple "image captioning."
    *   **Vision Encoder:** Based on **MetaCLIP** ([Xu et al., 2023](https://arxiv.org/abs/2309.16671)), improved and co-trained with the Llama model to better suit the LLM's needs.

3.  **Ultra-Long Context:**
    *   **10M Token Limit:** Llama 4 Scout achieves an **industry-leading 10 million token context window**.
    *   **Technical Underpinnings:**
        *   **iRoPE Architecture:** Combines ideas from **RoPE (Rotary Position Embeddings)** and **NoPE (No Positional Encoding)**. Implemented via **interleaved attention layers**, where specific layers use NoPE ([Kazemnejad et al., 2023](https://arxiv.org/abs/2305.19466)), relying on the attention mechanism to implicitly learn positional relationships, while RoPE is still used in most other layers. (The "i" signifies both interleaved and the goal of infinite context).
        *   **Scalable-Softmax:** Combined with inference-time temperature scaling ([Nakanishi et al., 2025](https://arxiv.org/abs/2501.19399)), enhancing the model's generalization ability to unseen lengths.
        *   **Specialized Training:** Underwent mid-training and post-training on specially constructed long-context datasets. Scout was trained on 256k context length and generalized to 10M via iRoPE and Scalable Softmax.
    *   **Practicality Observation:** While 10M tokens are appealing, processing such long contexts in practice may encounter issues like inference efficiency, attention diffusion, and bandwidth bottlenecks. Its **effectiveness and efficiency in real-world scenarios remain to be validated by users**.

{{< figure
    src="llama4_sequence_position_nll.png"
    caption="Fig. 14. Cumulative average NLL loss per sequence position for code generation, demonstrating Llama 4 Scout's strong performance over long contexts. (Source: [Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))"
    align="center"
    width="100%"
>}}

4.  **Large-Scale High-Quality Pre-training:**
    *   **Data Scale:** Training data exceeds **30 trillion tokens** (more than double LLaMA 3), including text, images, and video.
    *   **Multilingual Coverage:** Covers **200 languages**, with over 100 languages having more than 1 billion tokens each. Total multilingual token count is 10 times that of LLaMA 3.
    *   **Training Efficiency:** Trained using **FP8 precision**. Behemoth achieved high utilization of 390 TFLOPs/GPU on 32K GPUs. Utilized **MetaP** technology to reliably set hyperparameters.

**Revolutionary Post-training Process**

LLaMA 4 employs a new three-stage post-training process designed to balance instruction following, emergent intelligence, and dialogue quality:

1.  **Lightweight SFT (Supervised Fine-Tuning):** Focuses on supervised fine-tuning using a small amount of harder datasets to teach the model basic instruction following and dialogue formats, avoiding overfitting simple patterns and preserving space for subsequent RL exploration. **Significantly reduced simple SFT data** compared to previous versions (Maverick >50%, Behemoth >95%).
2.  **Online RL (Reinforcement Learning):** The key stage for enhancing the model's core intelligence and complex task capabilities. Employs a **continuous online RL strategy** where the model learns through interaction with the environment, explores using carefully selected harder prompts, and alternates between model training and data filtering (retaining medium-to-hard interaction data) to balance computation and effectiveness.
3.  **Lightweight DPO (Direct Preference Optimization):** Performed after RL to fine-tune the model's response style, safety, and correct corner cases, serving as the final "refinement and polishing" step to ensure the unification of intelligence and smooth conversational experience.

**Teacher Model & Co-Distillation**

*   The powerful **Behemoth (2T)**, though unreleased, transferred its knowledge to Scout and Maverick during the pre-training phase via a **novel co-distillation technique**.
*   This **co-distillation** occurred during pre-training, using a new distillation loss function with dynamically weighted soft targets (teacher model's logits) and hard targets (true labels). This significantly improved the quality of the student models (especially in math, coding, etc.) while amortizing the training cost of the teacher model.

**Large-Scale RL Infrastructure**

To train ultra-large MoE models like Behemoth, Meta completely overhauled its RL infrastructure, adopting a **fully asynchronous online RL training framework**. This optimized MoE parallelism, enabled flexible GPU resource allocation, and achieved nearly a 10x improvement in training efficiency.

### Comparison

| Feature               | LLaMA 1              | LLaMA 2              | Code Llama           | Llama Guard          | LLaMA 3              | LLaMA 4                  |
|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|--------------------------|
| **Release Date**      | 2023/02              | 2023/07              | 2023/08              | 2023/12+             | 2024/04+             | 2025/04+                 |
| **Base Model**        | -                    | -                    | LLaMA 2              | LLaMA 2 / LLaMA 3    | -                    | -                        |
| **Model Scale**       | 7B - 65B             | 7B, 13B, 70B         | 7B - 70B             | 7B / 8B (+Vision)    | 1B - 405B (+Vision)  | 109B, 400B, ~2T (MoE)    |
| **Training Data Size**| 1T - 1.4T tokens     | 2T+ tokens           | + 0.5T/1T Code       | ~40k Safety Class.   | 15T+ tokens          | 30T+ tokens (Multimodal) |
| **Context Length**    | 2k tokens            | 4k tokens            | 100k tokens          | 4k / 8k+             | 8k / 128k tokens     | 1M / 10M tokens          |
| **Tokenizer**         | SentencePiece (32k)  | SentencePiece (32k)  | SentencePiece (32k)  | Based on LLaMA 2/3   | tiktoken (128k)      | tiktoken (256k)          |
| **Positional Encoding**| RoPE                 | RoPE                 | RoPE (Base adjusted) | RoPE                 | RoPE                 | iRoPE                    |
| **Attention**         | MHA                  | MHA / GQA (34B, 70B) | MHA / GQA (>13B)     | Based on LLaMA 2/3   | GQA                  | GQA                      |
| **Normalization**     | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)    | RMSNorm (PreNorm)        |
| **Activation Func.**  | SwiGLU               | SwiGLU               | SwiGLU               | SwiGLU               | SwiGLU               | SwiGLU                   |
| **Model Type**        | Text Model           | Text Model           | Code Generation      | Safety Classifier    | Multimodal Model     | Multimodal Model         |

## Key Technology Analysis

Below is an analysis of the key technologies widely adopted in the LLaMA series.

### RMS Normalization (RMSNorm)

In deep learning model training, normalization layers are crucial for accelerating convergence, improving generalization, and stabilizing the training process. **RMSNorm (Root Mean Square Normalization)** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) is a simplified variant of Layer Normalization. It normalizes using only the Root Mean Square (RMS) of the inputs, omitting the mean centering step, thus reducing computation.

Its mathematical expression is:
$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$
Where:
*   $ x \in \mathbb{R}^d $ is the input vector.
*   $ d $ is the vector dimension.
*   $ \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon} $ calculates the root mean square of the input.
*   $ \epsilon $ is a small positive number (e.g., $10^{-6}$) to prevent division by zero and increase numerical stability.
*   $ \gamma \in \mathbb{R}^d $ is a learnable scaling parameter vector (gain). RMSNorm typically omits the learnable offset parameter (bias) $ \beta $ found in LayerNorm.

**Main reasons for LLaMA series choosing RMSNorm:**

*   **High Computational Efficiency:** Compared to LayerNorm, RMSNorm requires less computation because it doesn't need to calculate the mean. This is particularly important for computationally intensive large language model training and inference.
*   **Comparable Performance:** Practice has shown that RMSNorm often achieves performance comparable to or even better than LayerNorm in architectures like Transformers, while maintaining training stability.
*   **Simple Implementation:** Its computational logic is relatively simple, making it easy to implement efficiently on various hardware.

> For a comparison of various Norm techniques and code implementations, refer to the blog post: [Normalization in Deep Learning](https://syhya.github.io/posts/2025-02-01-normalization/).

### FFN_SwiGLU

**Swish-Gated Linear Unit (SwiGLU)** ([Shazeer, 2020](https://arxiv.org/abs/2002.05202v1)) is a key technique used in LLaMA to enhance the non-linear expressive capability of the Feed-Forward Network (FFN). SwiGLU combines the Swish activation function with a gating mechanism, significantly improving the model's expressiveness and performance. Furthermore, unlike the $4d$ hidden dimension used in PaLM ([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)), LLaMA employs a $\frac{2}{3} \times 4d$ hidden dimension, achieving higher parameter efficiency while keeping the parameter count and computational load roughly constant.

Mathematical expression:
$$
\operatorname{FFN}_{\mathrm{SwiGLU}}\left(x, W_1, W_3, W_2\right)=\left(\operatorname{Swish}\left(x W_1\right) \otimes x W_3\right) W_2
$$
Where:
- $ \text{Swish}(x) = x \cdot \sigma(x) $ (Swish activation function).
- $ \sigma(x) = \frac{1}{1 + e^{-x}} $ (Sigmoid function).
- $ \otimes $ denotes element-wise multiplication.
- $ W_1, W_2, W_3 $ are linear transformation matrices.

**Advantages**:
- **Enhanced Non-linear Expression**: By combining the Swish activation function with a gating mechanism, SwiGLU can more effectively capture complex patterns and relationships, boosting the expressive power of the FFN layer.
- **Parameter Efficiency**: Using a $\frac{2}{3} \times 4d$ hidden dimension allows the introduction of an additional linear transformation matrix while maintaining the total parameter count, leading to efficient parameter utilization.
- **Performance Improvement**: FFN_SwiGLU has shown significant performance improvements on various benchmarks, especially excelling in handling complex tasks and long texts. For example, in text generation and understanding tasks, SwiGLU helps the model better grasp context and long-range dependencies.

**Implementation Details**:
- **Weight Matrix Adjustment**: To maintain the same parameter count and computational load as traditional FFN layers, SwiGLU reduces the hidden layer dimension (e.g., adjusting the hidden size from $4d$ to $\frac{2}{3} \times 4d$), ensuring the overall model efficiency is unaffected despite introducing an extra linear transformation matrix.
- **Compatibility**: As a member of the GLU family, SwiGLU can be seamlessly integrated into existing Transformer architectures, replacing traditional ReLU or GELU activation functions to enhance overall model performance.

> For implementation code, refer to this file: [swiglu.py](https://github.com/syhya/syhya.github.io/blob/main/content/zh/posts/2025-04-06-llama/swiglu.py).

### Grouped Query Attention (GQA)

**Grouped Query Attention (GQA)** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) is a key optimization technique for the standard Multi-Head Attention (MHA) mechanism, particularly applied in large language models like LLaMA. Its primary goal is to reduce the memory bandwidth and capacity required for loading and storing the **KV Cache** during inference, thereby achieving a better balance between model performance and computational efficiency.

GQA is a compromise between MHA and Multi-Query Attention (MQA):

*   **MHA:** Has $H$ Query heads, each with its own independent set of $H$ Key (K) and Value (V) projections. Computation and KV Cache size are proportional to the number of heads $H$.
*   **MQA:** Still has $H$ Query heads, but all heads share a single set of K and V projections. This drastically reduces the KV Cache size (to $1/H$ of MHA's), but can potentially degrade model quality.
*   **GQA:** Divides the $H$ Query heads into $G$ groups ($1 < G < H$, and $H$ is a multiple of $G$). The $H/G$ Query heads within each group share the same set of K and V projections. There are a total of $G$ sets of K and V projections.


{{< figure
    src="attention_comparison.png"
    caption="Fig. 15. Overview of Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA). GQA groups query heads to share key/value heads. (Source: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))"
    align="center"
    width="100%"
>}}

The computation steps are as follows:

1.  **Projection:** Input $X$ is still projected to get $Q, K, V$. $Q$ is split into $H$ heads $Q_1, \dots, Q_H$. $K$ and $V$ are split into $G$ groups $K^1, \dots, K^G$ and $V^1, \dots, V^G$.
2.  **Grouped Attention:** For the $g$-th group ($g=1, \dots, G$), the corresponding Query heads (e.g., $Q_i$ where $i$ belongs to group $g$) compute attention with the shared $K^g$ and $V^g$:
    $$
    \text{Attention}_i(Q_i, K^g, V^g) = \text{softmax}\left( \frac{Q_i (K^g)^\top}{\sqrt{d_k}} \right) V^g
    $$
    where $d_k$ is the dimension of each K head (and also Q head).
3.  **Concatenation & Output:** The outputs of all heads $ \text{Attention}_1, \dots, \text{Attention}_H $ are concatenated and then passed through an output projection matrix $W_O$ to get the final output.


**Advantages:**

*   **Balances Performance and Efficiency:** GQA significantly reduces the KV Cache size (to $G/H$ of MHA's) while typically maintaining model quality closer to MHA than MQA does.
*   **Accelerates Inference:** Reducing memory bandwidth requirements can significantly speed up inference for large models, especially in long sequence generation scenarios.

> For a more detailed comparison between **MHA**, **MQA**, and **GQA** attention mechanisms, along with code examples, refer to the blog post: [Attention Mechanisms in Transformers: Comparing MHA, MQA, and GQA](https://syhya.github.io/posts/2025-01-16-group-query-attention/#grouped-query-attention-gqa).

### Rotary Positional Embeddings (RoPE)

**Rotary Positional Embeddings (RoPE)** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) is an effective method for injecting positional information into the Transformer attention mechanism, particularly adept at encoding relative positional information. Unlike traditional absolute positional encodings (like sinusoidal or learnable embeddings), RoPE achieves this by applying position-dependent rotation operations to the Query and Key vectors.

{{< figure
    src="rope.png"
    caption="Fig. 16. Implementation of Rotary Position Embedding(RoPE). (Source: [Su et al., 2021](https://arxiv.org/abs/2104.09864))"
    align="center"
    width="90%"
>}}

Assume $q_m$ and $k_n$ are the Query vector at position $m$ and the Key vector at position $n$, respectively. RoPE treats a $d$-dimensional vector $x$ ($q$ or $k$) as $d/2$ blocks of 2D vectors $[x^{(1)}, x^{(2)}, \dots, x^{(d/2)}]$, where $x^{(i)} = [x_{2i-1}, x_{2i}]$. For position $m$, RoPE defines a rotation matrix $R_m$ composed of $d/2$ 2D rotation matrices:
$$
R_m = \text{diag}(R_{m,1}, R_{m,2}, \dots, R_{m,d/2})
$$

where each 2D rotation matrix is:
$$
R_{m,i} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
$$

The rotation frequencies are $ \theta_i = b^{-2(i-1)/d} $, where $b$ is a predefined base (typically 10000 in LLaMA).

After applying RoPE, the new Query and Key vectors are $q'_m = R_m q_m$ and $k'_n = R_n k_n$. The key insight is that their inner product (dot product, which determines attention scores) depends only on the relative position $m-n$:

$$
(q'_m)^\top k'_n = (R_m q_m)^\top (R_n k_n) = q_m^\top R_m^\top R_n k_n = q_m^\top R_{n-m} k_n
$$

This utilizes the property of rotation matrices $R_m^\top R_n = R_{n-m}$.

**Advantages:**

*   **Explicit Relative Position Encoding:** The inner product result directly depends on the relative distance $m-n$, which is very natural for capturing relationships between elements in a sequence.
*   **Long-Distance Decay Property:** As the relative distance $|m-n|$ increases, the change in angle between vectors due to rotation typically causes the inner product value to decay, aligning with the intuition that more distant elements have weaker correlations.
*   **Good Extrapolation:** Theoretically, RoPE can generalize better to sequence lengths longer than those seen during training because it doesn't rely on a maximum absolute position. Adjusting the base $b$ (as in Code Llama and LLaMA 4's iRoPE) can further optimize its performance on ultra-long contexts.
*   **No Extra Parameters:** RoPE is a fixed, position-based transformation that introduces no additional learnable parameters.
*   **Compatibility with Linear Attention:** It can be used in conjunction with various linear attention variants.

### Mixture-of-Experts (MoE)

**Mixture-of-Experts (MoE)** is a neural network architecture paradigm designed to increase model capacity (total parameters) while controlling computational cost (active parameters). It replaces certain layers in the network (typically FFN layers) with multiple parallel expert subnetworks. A lightweight gating network dynamically selects a small number (usually Top-K, with K=1 or 2) of these experts for each input token to perform computation.

{{< figure
    src="llama4_moe.png"
    caption="Fig. 17. The Illustration of a mixture-of-experts(MoE) in llama4. (Source: [Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))"
    align="center"
    width="100%"
>}}


Assume an MoE layer has $N$ experts $E_1, E_2, \dots, E_N$ (e.g., each expert is an independent FFN) and a gating network $G$. For an input token $x$, the computation process of the MoE layer is as follows:

1.  **Gating Calculation:** The gating network $G$ (often a simple linear layer followed by Softmax) computes the probability or weight for selecting each expert: $p = G(x) = \text{Softmax}(\text{Linear}(x))$, where $p \in \mathbb{R}^N$.
2.  **Expert Selection (Top-K Routing):** Based on the gating output $p$, the K experts with the highest scores are selected. Let the set of selected expert indices be $\mathcal{T} = \text{TopKIndices}(p)$.
3.  **Expert Computation:** Only the selected K experts compute on the input $x$, yielding outputs $E_i(x)$ for $i \in \mathcal{T}$.
4.  **Output Combination:** The final output $y$ is the weighted sum of the outputs from the selected experts, using their gating weights (often re-normalized):
    $$
    y = \sum_{i \in \mathcal{T}} \frac{p_i}{\sum_{j \in \mathcal{T}} p_j} \cdot E_i(x)
    $$
    Alternatively, in some implementations, the weights $p_i$ might be used directly.

**Advantages:**

*   **Decoupling Parameters and Computation:** MoE allows models to have a massive total parameter count (by increasing the number of experts $N$), but the computational cost of each forward pass depends only on the computation of the activated K experts, which is much lower than that of a dense model with an equivalent total parameter count. This enables training larger capacity, potentially higher-performing models within a limited computational budget.
*   **Expert Specialization:** Theoretically, different experts can learn to handle specific aspects of different types of data, patterns, or tasks, enabling modular storage and processing of knowledge, thereby enhancing the model's overall capability and generalization.

**Challenges:**

*   **Load Balancing:** Ensuring that all experts are utilized roughly equally is necessary to avoid some experts being overloaded while others remain idle. This often requires introducing auxiliary loss functions (like Load Balancing Loss) to encourage uniform routing.
*   **Communication Overhead:** In distributed training and inference, efficient communication (e.g., All-to-All) is needed between different devices (GPUs) to route tokens to the devices storing the corresponding experts and to gather the results. This increases implementation complexity and communication costs.
*   **Training Stability:** Training MoE models can be less stable than training dense models, requiring careful tuning of hyperparameters and training strategies.
*   **Memory Footprint:** Although computation is sparse, the total number of parameters is huge, requiring substantial memory to store all expert weights.

> For a more detailed explanation of MoE, refer to the Mixture-of-Experts section in the blog post: [Parallelism and Memory Optimization Techniques for Training Large Models](https://syhya.github.io/posts/2025-03-01-train-llm/#mixture-of-experts-model).


## References
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

[20] Shazeer, Noam. ["Glu variants improve transformer."](https://arxiv.org/abs/2002.05202v1) arXiv preprint arXiv:2002.05202 (2020).

[21] Ainslie, Joshua, et al. ["Gqa: Training generalized multi-query transformer models from multi-head checkpoints."](https://arxiv.org/abs/2305.13245) arXiv preprint arXiv:2305.13245 (2023).

[22] Su, Jianlin, et al. ["Roformer: Enhanced transformer with rotary position embedding."](https://arxiv.org/abs/2104.09864) Neurocomputing 568 (2024): 127063.

## Citation

> **Citation**: When reproducing or citing the content of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (Apr 2025). The LLaMA Herd. https://syhya.github.io/posts/2025-04-06-llama

Or

```bibtex
@article{syhya2025llama,
  title   = "The LLaMA Herd",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Apr",
  url     = "https://syhya.github.io/posts/2025-04-06-llama"
}
```