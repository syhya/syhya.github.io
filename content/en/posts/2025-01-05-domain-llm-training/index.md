---
title: Building Domain-Specific LLMs
date: 2025-01-05T12:00:00+08:00
author: "Yue Shui"
tags: ["AI", "NLP", "LLM", "Pre-training", "Post-training", "DPO", "Domain Models", "DeepSpeed"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
---
 
## Background

With the widespread application of Large Language Models (LLMs) across various industries, enterprises and research teams face an urgent need to adapt general-purpose models to specific domains. Foundational LLMs often fail to meet deep domain-specific requirements when handling specialized tasks. For example, in the application of closed-source programming languages, existing open-source models lack sufficient understanding of their syntax and semantics, leading to poor performance in tasks such as code generation and error correction. Therefore, injecting domain knowledge and training dedicated LLMs has become a key step in enhancing development efficiency and code quality.

Based on my work experience, this article summarizes how to build LLMs equipped with specific domain knowledge by leveraging data preparation, model training, deployment, evaluation, and continuous iteration on top of existing general models.

## Why Inject Domain Knowledge into the Foundational LLMs?

### Challenge 1: Limited Domain Knowledge

Existing pre-trained models (such as GPT-4 and Llama 3) are primarily trained on general-purpose corpora, lacking in-depth understanding of niche languages or proprietary domains. This deficiency leads to subpar performance when the models handle programming code.

### Challenge 2: Data Security and Compliance

When enterprises handle sensitive data, they must adhere to strict data sovereignty and compliance requirements. Uploading proprietary business data to third-party cloud services poses security risks, necessitating data processing and model training within local environments.

### Challenge 3: Limitations of OpenAI Fine-Tuning

Mainstream commercial APIs for fine-tuning are typically basic and struggle to achieve deep alignment and optimization. For highly customized domain models, such approaches often fail to meet the required specifications.

---

## Two Approaches of Injecting Knowledge

In practical projects, the common methods for injecting domain knowledge into base models include **Fine-Tuning** and **Retrieval-Augmented Generation (RAG)**. The following sections provide a detailed comparison of these methods to aid in selecting the most suitable strategy.

### Method Comparison

#### Fine-Tuning

**Core Concept**  
Through continued pre-training, supervised fine-tuning, and preference alignment, directly update the model parameters to enable it to master domain-specific knowledge and task patterns.

**Technical Details**
- **Continued Pre-Training (CPT)**: Continue pre-training the base model on a large volume of domain-specific unsupervised data.
- **Supervised Fine-Tuning (SFT)**: Perform supervised fine-tuning using high-quality labeled data.
- **Preference Alignment (DPO)**: Optimize model outputs based on user feedback.
- **Parameter Tuning Methods**: Utilize full-parameter fine-tuning or combine with PEFT methods like LoRA to freeze some parameters and add adapters.

**Advantages**
- **Deep Customization**: Updating the internal weights of the model enables a profound understanding of domain knowledge.
- **No External Retrieval Dependency**: Inference does not require additional knowledge bases, reducing latency and total token consumption.
- **Enhanced Overall Performance**: Significantly outperforms general models in domain-specific tasks.

**Disadvantages**
- **High Computational Cost**: Requires substantial computational resources for training, especially during the CPT phase.
- **Long Training Cycles**: From data preparation to model training and optimization, the process is time-consuming.
- **Catastrophic Forgetting**: The model may forget its original general capabilities while learning new knowledge.

#### Retrieval-Augmented Generation (RAG)

**Core Concept**  
Build a domain-specific knowledge base and retrieve relevant documents during inference to assist the model in generating more accurate responses without directly altering model parameters.

**Technical Details**
- **Data Processing**: Preprocess domain documents by chunking them based on size and overlap.
- **Vectorization**: Embedding text chunks as vectors using embedding models and storing them in a Vector Store for retrieval.
- **Retrieval**: During inference, retrieve relevant documents through similarity search to provide contextual information or few-shot examples to the base model.

**Advantages**
- **Preserves General Capabilities**: Model parameters remain unchanged, retaining general language abilities.
- **Quick Updates**: The knowledge base can be dynamically updated without retraining the model.
- **Computational Efficiency**: Avoids large-scale training, saving computational resources.

**Disadvantages**
- **Dependence on Knowledge Base Quality**: The quality of retrieved documents directly impacts response quality.
- **Inference Speed**: The retrieval process may increase inference latency and require more tokens.
- **Limited Knowledge Coverage**: The model’s internal knowledge is still restricted by the base model’s pre-training data.

---

## Models and Training Resources

### Base Models

Taking the [Llama 3 series](https://arxiv.org/pdf/2407.21783) as an example, it features the following characteristics:

- **Parameter Scale**  
  The Llama 3 series includes models ranging from 1B to 405B parameters, widely supporting multilingual processing, code generation, reasoning, as well as visual and textual tasks. Smaller models (1B and 3B) are specially optimized for edge and mobile devices, supporting up to 128K context windows, efficiently handling local tasks such as summary generation, instruction execution, and text rewriting.

- **Multimodal Capabilities**  
  Llama 3's visual models (11B and 90B parameters) outperform many closed models in image understanding tasks and support multimodal processing of images, videos, and audio. All models support fine-tuning, facilitating customized development for specific domains.

- **Open Source and Community Support**  
  Llama 3 series models and their weights are released in open-source form and can be accessed via [llama.com](https://llama.com) and the [Hugging Face platform](https://huggingface.co/meta-llama), providing convenient access and application support for developers.

- **Dataset Restrictions**  
  Although the Llama 3 models are released as open-source, the datasets used for their training are not open-sourced. Therefore, strictly speaking, Llama 3 is not entirely open-source. This limitation may pose challenges in addressing catastrophic forgetting, as obtaining data sets identical to the original training data is difficult.

### Training Resources

Training large language models requires robust computational resources and efficient distributed training frameworks.

- **Hardware Resources**  
  - **GPU Clusters**: NVIDIA A100 or H100 GPUs are recommended, with configurations of 4 or 8 GPUs connected via NVLink or InfiniBand to enhance communication bandwidth.  
  - **Storage Resources**: High-performance SSDs (e.g., NVMe) to support fast data read and write operations.

- **Software Frameworks**  
  - **Distributed Training Frameworks**: [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), among others, support large-scale model training.  
  - **Inference Frameworks**: [vLLM](https://github.com/vllm-project/vllm), [ollama](https://github.com/jmorganca/ollama), etc., optimize inference speed and resource utilization.

- **Parallel Strategies**  
  - **Data Parallelism (DP)**: Suitable when the model fits on a single GPU, implemented via DeepSpeed's ZeRO Stage 0.  
  - **Model Parallelism (MP), Pipeline Parallelism (PP), and Tensor Parallelism (TP)**: When the model cannot fit on a single GPU, optimize using ZeRO Stage 1, 2, or 3, or employ ZeRO-Infinity to offload parts of parameters and optimizer states to CPU or NVMe.

## DeepSpeed ZeRO Sharding Strategies Comparison

### ZeRO Stage Sharding Strategies

| **ZeRO Stage** | **Description**                                                                                                                                                                                        | **GPU Memory Usage** | **Training Speed** |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|---------------------|
| **ZeRO-0**     | Pure data parallelism without any sharding. All optimizer states, gradients, and parameters are fully replicated on each GPU.                                                                        | Highest              | **Fastest**         |
| **ZeRO-1**     | Shards optimizer states (e.g., momentum and second moments), reducing GPU memory usage, but gradients and parameters remain data parallel.                                                             | High                 | Slightly slower than ZeRO-0 |
| **ZeRO-2**     | Shards optimizer states and gradients, further reducing GPU memory usage based on ZeRO-1.                                                                                                             | Medium               | Slower than ZeRO-1  |
| **ZeRO-3**     | Shards optimizer states, gradients, and model parameters, achieving the lowest GPU memory usage, suitable for extremely large models. Requires parameter broadcasting (All-Gather/All-Reduce) during forward/backward passes, significantly increasing communication overhead. | Low                  | Significantly slower than ZeRO-2, depends on model size and network bandwidth |

### Offload Strategies

| **Offload Type**               | **Description**                                                                                                                                                                                        | **GPU Memory Usage**       | **Training Speed**                                                                                                      |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **ZeRO-1 + CPU Offload**        | Extends ZeRO-1 by offloading optimizer states to CPU memory, further reducing GPU memory usage but necessitating CPU-GPU data transfer, relying on PCIe bandwidth, and occupying CPU memory.              | Medium-low                 | Slower than ZeRO-1, affected by CPU performance and PCIe bandwidth                                                     |
| **ZeRO-2 + CPU Offload**        | Extends ZeRO-2 by offloading optimizer states to CPU memory, further reducing GPU memory usage for larger models but increasing CPU-GPU data transfer overhead.                                         | Lower                      | Slower than ZeRO-2, affected by CPU performance and PCIe bandwidth                                                     |
| **ZeRO-3 + CPU Offload**        | Extends ZeRO-3 by offloading optimizer states and model parameters to CPU, achieving minimal GPU memory usage but with extremely high CPU-GPU communication volume and CPU bandwidth significantly lower than GPU-GPU communication. | Extremely Low             | Very Slow                                                                                                               |
| **ZeRO-Infinity (NVMe Offload)** | Based on ZeRO-3, offloads optimizer states, gradients, and parameters to NVMe, breaking CPU memory limits and suitable for ultra-large-scale models; performance highly depends on NVMe parallel read/write speeds. | Extremely Low<br>Requires NVMe support | Slower than ZeRO-3 but generally faster than ZeRO-3 + CPU Offload, can achieve better throughput if NVMe bandwidth is sufficient |

---

## Communication Volume and Performance Impact

- **ZeRO-0/1/2**:  
  Communication is primarily **gradient synchronization** using All-Reduce operations, resulting in relatively low communication volume.

- **ZeRO-3**:  
  Requires **All-Gather/All-Reduce** operations for model parameters, significantly increasing communication volume. Network bandwidth becomes a critical bottleneck, and parameter broadcasting during forward/backward passes further exacerbates communication load.

- **CPU Offload** (ZeRO-1/2/3 + CPU):  
  - Offloads optimizer states or parameters to CPU, reducing GPU memory usage.  
  - Communication volume mainly arises from **CPU <-> GPU** data transfers, which have much lower bandwidth compared to GPU-GPU communication, easily causing performance bottlenecks, especially in **ZeRO-3** scenarios.

- **NVMe Offload** (ZeRO-Infinity):  
  - Further offloads to NVMe based on **ZeRO-3**, overcoming CPU memory limitations to support ultra-large-scale models.  
  - Performance heavily relies on **NVMe I/O bandwidth** and parallelism. If NVMe speed is sufficiently high, it typically outperforms CPU Offload; however, performance may suffer in scenarios with weak I/O performance or high latency.

### Hardware and Configuration Impact

- **Hardware Constraints**:  
  - **PCIe Bandwidth**, **Network Bandwidth**, **NVMe I/O**, etc., significantly impact Offload performance. Optimal strategies should be selected based on the hardware environment.

- **Additional Notes**:  
  - **CPU Offload** utilizes CPU memory and transfers data via PCIe; **NVMe Offload** saves states on NVMe devices.  
  - NVMe Offload generally outperforms CPU Offload when **NVMe I/O performance is adequate**, but care must be taken to avoid performance bottlenecks caused by insufficient I/O performance.

- **Reference to Official Documentation**:  
  - It is recommended to consult the [DeepSpeed official documentation](https://www.deepspeed.ai/) for the latest and most accurate configuration parameters and performance tuning advice.

---

## Data Preparation: The Core of Training Success

Data quality directly determines model performance. Data preparation includes data collection, cleaning, deduplication, categorization and balancing, anonymization, and other steps.

### Pre-Training Data

#### Data Sources

- **Public Datasets**: Such as [the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2), Common Crawl, etc.  
- **Enterprise Proprietary Data**: Internal documents, code repositories, business logs, etc.  
- **Web Crawlers**: Collect domain-relevant web content using crawling technologies.

#### Data Scale

- It is recommended to use at least hundreds of millions to billions of tokens to ensure the model can thoroughly learn domain knowledge.  
- When data volume is insufficient, model performance may be limited. Data augmentation methods are suggested to supplement the data.

#### Data Processing

1. **Data Preprocessing**  
   - **Uniform Formatting**: Process large volumes of unlabeled corpora from multiple data sources to ensure consistent formatting. It is recommended to use efficient storage formats like Parquet to improve data reading and processing efficiency.

2. **Data Deduplication**  
   - **Detection Methods**: Use algorithms such as MinHash, SimHash, or cosine similarity for approximate duplicate detection.  
   - **Granularity of Processing**: Choose to deduplicate at the sentence, paragraph, or document level, adjusting flexibly based on task requirements.  
   - **Similarity Threshold**: Set a reasonable similarity threshold (e.g., 0.9) to remove texts with duplication above the threshold, ensuring data diversity.

3. **Data Cleaning**  
   - **Text Filtering**: Remove garbled text, spelling errors, and low-quality text by combining rule-based methods and model scorers (e.g., BERT/RoBERTa).  
   - **Formatting Processing**: Prefer using JSON format to handle data, ensuring the accuracy of special formats like code, Markdown, and LaTeX.

4. **Data Anonymization**  
   - **Privacy Protection**: Anonymize or remove sensitive information such as names, phone numbers, emails, passwords, etc., to ensure data compliance.  
   - **Filtering Non-Compliant Content**: Remove data blocks containing illegal, pornographic, or racially discriminatory content.

5. **Data Mixing and Balancing**  
   - **Proportion Control**: For example, combine 70% domain-specific data with 30% general data to prevent the model from forgetting general capabilities.  
   - **Task Types**: Ensure the data includes various task types such as code generation, Q&A dialogue, document summarization, multi-turn conversations, and mathematical reasoning.

6. **Data Sequencing**  
   - **Progressive Guidance**: Use Curriculum Learning to start training with simple, clean data and gradually introduce more complex or noisy data, optimizing the model's learning efficiency and convergence path.  
   - **Semantic Coherence**: Utilize In-Context Pretraining techniques to concatenate semantically similar documents, enhancing contextual consistency and improving the model's depth of semantic understanding and generalization ability.

### Supervised Fine-Tuning Data

#### Data Format

Adopt Alpaca or Vicuna styles, such as single-turn and multi-turn dialogues structured as [instruction, input, output].  
- **Scale**: From thousands to hundreds of thousands, depending on project requirements and computational resources.  
- **Quality**: Ensure high-quality and diverse data to prevent the model from learning errors or biases.

#### Data Construction

During the data construction process, we first collect daily business data and collaboratively build foundational questions with business experts. Subsequently, we use large language models for data augmentation to enhance data diversity and robustness. The specific data augmentation strategies are as follows:

#### Data Augmentation Strategies

- **Diverse Expressions**  
  Rewrite existing data using large language models through synonym replacement and syntactic transformations to increase data diversity.

- **Robustness Enhancement**  
  Create prompts containing spelling errors, mixed languages, and other input variations to simulate real-world scenarios while ensuring high-quality generated answers.

- **Knowledge Distillation**  
  Utilize large language models like GPT-4 and Claude for knowledge distillation to generate Q&A pairs that meet requirements.

- **Complex Task Design**  
  Manually design high-quality data for complex scenarios (e.g., multi-turn dialogues, logical reasoning) to cover the model's capability boundaries.

- **Data Generation Pipeline**  
  Build an automated data generation pipeline that integrates data generation, filtering, formatting, and validation to improve overall efficiency.

#### Key Points

- **Task Type Annotation**: Clearly annotate each data entry with its task type to facilitate subsequent fine-grained analysis and tuning.  
- **Multi-Turn Dialogues and Topic Switching**: Construct data that captures contextual coherence and topic transitions in multi-turn dialogues to ensure the model learns the ability to handle topic switching and maintain contextual relevance.  
- **Chain of Thought (CoT) Strategy**: For classification and reasoning tasks, generate procedural answers using CoT to improve accuracy.  
- **Data Flywheel**: Continuously collect real user queries after deployment, iterating data based on real needs; regularly clean the data to ensure quality and diversity.

### Preference Data

#### Data Format

- **Triple Structure**: [prompt, chosen answer, rejected answer]  
- **Annotation Details**:
  - **Multi-Model Sampling**: Generate answers using multiple models at different training stages or with different data ratios to increase data diversity.  
  - **Editing and Optimization**: Annotators can make slight modifications to the chosen answers to ensure answer quality.

#### Sampling Strategies

- **Multi-Model Sampling**: Deploy multiple versions of the model to generate diverse answers for the same prompt.  
- **Comparative Annotation**: Use manual or automated systems to compare generated answers and select superior answer pairs.

#### Key Points

- **Data Diversity and Coverage**: Ensure preference data covers various scenarios and tasks to prevent the model from underperforming in specific contexts.  
- **High-Quality Annotation**: The quality of preference data directly affects the model's alignment, requiring accurate and consistent annotations.

---

## Training Process

A complete training process for a domain-specific large language model typically includes **Continued Pre-Training (CPT) → Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO)** as the three main steps, ultimately achieving model deployment and continuous optimization.

### Comparison of Three Methods

#### Training Method Overview

| **Training Method**       | **Main Objective**                                                                   | **Data Requirements**                                                        | **Typical Application Scenarios**                                                |
|---------------------------|---------------------------------------------------------------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Continued Pre-Training (CPT)** | Continue pre-training on large-scale unsupervised corpora to inject new domain knowledge | Large amounts of unlabeled text (at least hundreds of millions to billions of tokens) | Supplementing domain knowledge, such as specialized texts in law, medicine, finance, etc. |
| **Supervised Fine-Tuning (SFT)**   | Fine-tune on supervised labeled data to strengthen specific tasks and instruction execution capabilities | Customized labeled data (instruction/dialog pairs), ranging from thousands to hundreds of thousands | Various specific tasks, such as code generation, Q&A, text rewriting, complex instruction execution, etc. |
| **Direct Preference Optimization (DPO)** | Optimize model outputs to align with human preferences using preference data (chosen vs. rejected) | Preference data: [prompt, chosen, rejected]<br>(relatively smaller scale) | Aligning with human feedback, such as response style, compliance, safety, etc. |

#### Advantages and Challenges

##### Continued Pre-Training (CPT)

**Advantages**:
- Better domain coverage, comprehensively enhancing the model's understanding and generation capabilities in specific domains.
- No need for additional manual annotation.

**Challenges/Limitations**:
- Requires a large volume of high-quality domain data.
- High training costs, necessitating massive computational power and time.
- May introduce domain biases, necessitating careful handling of data quality and distribution.

##### Supervised Fine-Tuning (SFT)

**Advantages**:
- Quickly acquires task execution capabilities.
- Significantly improves accuracy in specific scenarios.

**Challenges/Limitations**:
- High data annotation costs.
- Requires careful selection of labeled data to avoid overfitting.
- Fine-tuning may weaken the model's generality.

##### Direct Preference Optimization (DPO)

**Advantages**:
- No need to train a separate Reward Model.
- Requires less data and computational resources to achieve similar or better results compared to PPO.

**Challenges/Limitations**:
- Requires reliable preference annotations.
- Continues to need more preference data for complex and diverse scenarios.
- Easily constrained by the distribution of preference data.

---

### General Training Tips and Technical Details

When performing **CPT, SFT, and DPO**, there are numerous general training tips and technical details. The following sections uniformly describe these general aspects for better understanding and application.

#### Data Processing and Preparation

- **Data Quality**: Regardless of CPT, SFT, or DPO, data quality is crucial. Ensure data accuracy, unambiguity, and diversity.  
- **Data Formatting**: Consistent data formats simplify the training process. For example, using JSON or other structured formats to store training data.  
- **Data Augmentation**: Increase data diversity and improve the model's generalization ability through methods like LLM rewriting and optimization.

#### Learning Rate and Optimization

- **Learning Rate Settings**: Typically use a smaller learning rate than during pre-training, such as reducing from 3e-4 to 3e-5, depending on the task and data volume.  
- **Learning Rate Scheduling**: Use warm-up strategies (e.g., linearly increasing for the first 10% of steps), followed by linear decay or cosine annealing to ensure a smooth training process.  
- **Optimizer Selection**: Choose suitable optimizers based on model size and hardware resources, such as AdamW.

#### Training Strategies

- **Full-Parameter Fine-Tuning**: When resources permit, prioritize full-parameter fine-tuning to ensure the model fully captures new knowledge.  
- **Parameter-Efficient Fine-Tuning (PEFT)**: Methods like LoRA are suitable for scenarios with limited computational resources by freezing some parameters and adding adapters for efficient fine-tuning.  
- **Mixed Precision Training**: Use bf16 or fp16 on supported GPUs to reduce memory usage and increase training speed.  
- **Training Stability**: Employ techniques such as gradient clipping, regularization, dropout, and weight decay to prevent gradient explosion and model overfitting.  
- **Flash Attention**: Utilize [Flash Attention](https://github.com/Dao-AILab/flash-attention) to optimize the computation efficiency of the attention mechanism, enhancing training speed and reducing memory usage.

#### Monitoring and Tuning

- **Convergence Monitoring**: Continuously monitor loss curves on training and validation sets to ensure the model is converging properly. Adjust learning rates and other hyperparameters as needed.  
- **Checkpoint**: Regularly save checkpoints to prevent loss of all training progress due to unexpected interruptions.  
- **Early Stopping**: Prevent model overfitting by stopping training at an appropriate time and saving the best model state.  
- **Model Evaluation**: Conduct periodic evaluations during training to ensure model performance meets expectations.

### Continued Pre-Training (CPT)

#### Objective

Inject new domain knowledge into the base model by continuing pre-training on a large volume of domain-specific unsupervised data, enhancing the model's understanding and generation capabilities in the specific domain.

#### Training Tips

1. **Streaming Data Loading**  
   - Implement streaming data loading to dynamically read data during training, preventing memory overflows and training interruptions.

2. **Full-Parameter Fine-Tuning**  
   - Typically, update all model parameters during training to ensure comprehensive knowledge acquisition.  
   - Compared to parameter-efficient fine-tuning methods (e.g., LoRA), full-parameter fine-tuning offers better domain knowledge injection, especially when computational resources are abundant. It is recommended to prioritize full-parameter fine-tuning under such conditions.

### Supervised Fine-Tuning (SFT)

#### Objective

Enhance the model's practicality and accuracy by training it on high-quality labeled data to perform specific tasks such as code generation, code repair, and complex instruction execution.

#### Training Tips

1. **Number of Epochs**  
   - Typically, 1 to 4 epochs are sufficient to observe significant effects when data volume is adequate.  
   - If data volume is insufficient, consider increasing the number of epochs while being mindful of overfitting risks. Data augmentation is recommended in such cases.

2. **Data Augmentation and Diversity**  
   - Ensure training data covers a variety of task types and instruction expressions to improve the model's generalization ability.  
   - Include multi-turn dialogues and robustness data to enhance the model's capability to handle real user scenarios.

### Direct Preference Optimization (DPO)

#### Objective

Optimize model outputs to better align with human expectations and needs, including response style, safety, and readability, by leveraging user feedback and preference data.

#### Characteristics of DPO

- **Direct Optimization**  
  Does not require training a separate Reward Model. Instead, directly performs contrastive learning on (chosen, rejected) data pairs.

- **Efficiency**  
  Compared to PPO, DPO requires less data and computational resources to achieve similar or better results.

- **Dynamic Adaptation**  
  The model can immediately adapt whenever new data is available without the need to retrain a Reward Model.

#### Training Tips

1. **Collecting Preference Data**  
   - Deploy multiple models at different training stages or with different data ratios to generate diverse responses.  
   - Annotate chosen and rejected answer pairs through manual or automated means to ensure data diversity and quality.

2. **Contrastive Learning**  
   - Optimize model parameters by maximizing the probability of chosen answers and minimizing the probability of rejected answers.

3. **Iterative Optimization**  
   - Continuously collect user feedback, generate new preference data, and perform iterative training to gradually enhance model performance.  
   - Implement a data flywheel mechanism to achieve ongoing model evolution and optimization.

---

### Common Issues and Solutions

1. **Repetitive Outputs**  
   **Issue**: The model generates repetitive content, continuously printing without stopping.  
   **Solutions**:  
   - **Data Deduplication and Cleaning**: Ensure training data does not contain a large amount of repetitive content.  
   - **Check EOT (End-of-Token) Settings**: Prevent the model from continuously generating without stopping.  
   - **Align via SFT/DPO**: Optimize model output quality.  
   - **Adjust Decoding Strategies**: Increase parameters like top_k, repetition penalty, and temperature.

2. **Catastrophic Forgetting**  
   **Issue**: The model forgets its original general capabilities during fine-tuning, effectively overfitting to the new dataset and causing excessive changes to the original model parameter space.  
   **Solutions**:  
   - **Mix in Some General Data**: Maintain the model’s general capabilities.  
   - **Lower the Learning Rate**: Reduce the impact on existing knowledge.  
   - **Increase Dropout Rate and Weight Decay**: Prevent overfitting.  
   - **Use Parameter-Efficient Fine-Tuning Methods like LoRA**: Avoid large-scale parameter updates.  
   - **Utilize RAG Assistance**: Combine with external knowledge bases to enhance model performance.  
   - **[Chat Vector](https://arxiv.org/pdf/2310.04799)**: Quickly inject conversational and general capabilities into the model through simple arithmetic operations on model weights.

3. **Insufficient Understanding of Entity Relationships and Reasoning Paths**  
   **Issue**: The model struggles to correctly understand complex entity relationships and reasoning paths.  
   **Solutions**:  
   - **Introduce Chain-of-Thought (CoT) Data and Enhanced Reasoning Training**: Improve the model's capabilities through step-by-step reasoning training, combined with [Reinforcement Fine-Tuning](https://openai.com/form/rft-research-program/) and [o1/o3](https://openai.com/o1/) training methods.  
   - **Expand Training Data Coverage**: Incorporate more diverse scenarios containing complex entity relationships and reasoning paths.  
   - **Combine with Knowledge Graph Modeling**: Use [GraphRAG](https://github.com/microsoft/graphrag) to strengthen the model's understanding and reasoning abilities regarding entity relationships.

---

## Model Deployment and Evaluation

### Deployment

**Inference Frameworks**

- [**ollama**](https://github.com/jmorganca/ollama): Local inference deployment based on [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling quick startups.  
- [**vLLM**](https://github.com/vllm-project/vllm): Optimized for high concurrency and inference throughput in multi-user scenarios.  
- **Quantization**: Quantize the model to 8-bit or 4-bit to further reduce inference costs and improve deployment efficiency.

**Integrate RAG & Agents**

- **RAG**: Combine with a vector knowledge base to retrieve relevant documents or code snippets in real-time, assisting the model in generating more accurate responses.  
- **Agents**: Utilize Function Calls or multi-turn dialogue mechanisms to enable the model to invoke external tools or perform complex reasoning, enhancing interactivity and practicality.  
- **Langgraph**: Encapsulate RAG and multi-agent workflows to build customized dialogue systems or automated code generation platforms.

### Evaluation

**Evaluation Metrics**

- **CPT Phase**: Use domain-specific test sets to evaluate perplexity (PPL) or cross-entropy, measuring the model's mastery of new knowledge.  
- **SFT/DPO Phase**:  
  - **Human or Model Evaluation**: Assess the accuracy, coherence, readability, and safety of responses through human ratings or automated tools.  
  - **Code Generation**: Build a large-scale unit test set to evaluate the pass@k metric, measuring the correctness rate of code generation.  
  - **General Capabilities**: Test the model on common benchmarks (e.g., MMLU, CMMLU) to ensure minimal performance degradation on general tasks.

**Decoding Hyperparameters**

- **Consistency**: Maintain consistent decoding parameters such as top_k, top_p, temperature, and max_new_tokens during evaluation to ensure comparability of results.  
- **Grid Search**: When computational resources permit, evaluate different combinations of decoding parameters to select the optimal configuration.

---

## Data Flywheel and Continuous Iteration

**Data Flywheel Mechanism**

1. **Real-Time Collection of User Logs**  
   - Collect real user prompts and generated responses online, covering diverse usage scenarios and task types.

2. **Automated or Manual Annotation**  
   - Annotate collected user prompts and responses with preferences, generating new (chosen, rejected) data pairs.

3. **Iterative Training**  
   - Incorporate newly generated preference data into the next round of SFT/DPO training to continuously optimize response quality and user experience.

4. **Robustness Data**  
   - Include data with spelling errors, mixed languages, vague instructions, etc., to enhance the model’s robustness and ability to handle real-world scenarios.

**Continuous Optimization**

- **Feedback Loop**: Utilize user feedback to continuously improve training data and model performance, achieving self-optimization and evolution of the model.  
- **Multi-Model Collaboration**: Deploy multiple versions of the model to generate diverse responses, enhancing the model's comprehensive capabilities through contrastive learning.

---

## Integrating Intent Recognition and Multi-Agent Reasoning

Use an intent classification model to allow the large model to determine the category of user input intent. Based on the mapping between intent categories and context types, supervise the reasoning path, and then perform multi-way retrieval based on the reasoning path. Provide this information to the trained model to generate the final result.

---

## Conclusion

Through the combination of **Continued Pre-Training (CPT) → Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO)**, it is possible to effectively inject domain-specific knowledge into base large models, constructing closed-source LLMs capable of efficiently solving business problems. The key steps are as follows:

1. **Data Preparation**  
   - High-quality data collection, cleaning, deduplication, and categorization to ensure data diversity and accuracy.  
   - Implement data anonymization strategies to protect privacy and ensure compliance.

2. **Model Training**  
   - Use CPT to inject domain knowledge, SFT to learn specific task patterns, and DPO to optimize model outputs to align with human preferences and safety.  
   - Leverage efficient parallel training frameworks and hyperparameter tuning techniques to enhance training efficiency and resource utilization.

3. **Deployment and Evaluation**  
   - Employ efficient inference frameworks, integrating RAG and Agents for knowledge enhancement and functional extension.  
   - Conduct multi-dimensional evaluations to ensure the model performs as expected at each stage.

4. **Continuous Iteration**  
   - Build a data flywheel to continuously collect user feedback and optimize training data and model performance.  
   - Integrate RAG and Agents to achieve ongoing improvement and expansion of model capabilities.

Ultimately, through a systematic process and technical measures, it is possible to construct an AI system with not only profound domain knowledge but also the flexibility to handle complex business requirements over its lifecycle.

---

## References

1. [DeepSpeed](https://github.com/microsoft/DeepSpeed)  
2. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)  
3. [ollama](https://github.com/jmorganca/ollama)  
4. [vLLM](https://github.com/vllm-project/vllm)  
5. [GraphRAG](https://microsoft.github.io/graphrag/)  
6. [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)  
7. [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)  
8. [Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages](https://arxiv.org/pdf/2310.04799)  
9. [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374)  
10. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

---

## Citation

> **Citation**: To reproduce or cite the content of this article, please acknowledge the original author and source.
  
**Cited as:**

Yue Shui. (Jan 2025). Building Domain-Specific LLMs.  
[https://syhya.github.io/posts/2025-01-05-build-domain-llm](https://syhya.github.io/posts/2025-01-05-build-domain-llm)

Or

```bibtex
@article{syhya2024domainllm,
  title   = "Building Domain-Specific LLMs",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2025",
  month   = "Jan",
  url     = "https://syhya.github.io/posts/2025-01-05-build-domain-llm/"
}
