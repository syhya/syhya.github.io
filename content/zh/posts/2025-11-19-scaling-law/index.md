---
title: "Scaling Laws"
date: 2025-11-19T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep Learning", "AI", "LLM", "Scaling Laws", "Test-Time Compute", "Reinforcement Learning", "Reward Model", "Compute-Optimal"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

从 GPT 系列的演进中，研究者逐渐意识到：只要持续扩大模型参数、训练数据和计算资源，大模型性能便会沿着稳定且可预测的路径不断提升。这种可预测性正是由 **Scaling Laws** 所刻画，它为成本高昂的预训练提供了理论基础与实践信心。随着模型规模、对齐技术以及推理阶段的计算不断协同演化，AI 的能力边界正在系统性地被推高。它不仅是构建下一代模型的基础，也是在算力约束下持续提升模型能力的关键方法论。

## Scaling Laws

**Scaling Laws** ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)) 是理解大模型性能如何随资源投入而提升的基础。其首次系统地揭示了模型性能与三个核心要素——模型参数量  $N$、数据集大小 $D$ 和训练计算量 $C$ 之间的 [幂律(Power-Law)](https://en.wikipedia.org/wiki/Power_law)。一个基本的逆幂律关系可以表示为：

$$
y = a \left( \frac{1}{x} \right)^p
$$

在对数坐标系下，这个关系呈现为一条直线，这正是在多数尺度定律论文中看到的标志性图表。对于 LLM，`y` 通常是模型的测试损失，而 `x` 则是关注的某个尺度变量（如模型参数量）。

### 基本定律

{{< figure
    src="scaling_laws.png"
    caption="Fig. 1. Language modeling performance improves smoothly with model size, dataset size, and training compute. (Image source: [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361))"
    align="center"
    width="100%"
>}}

当其他两个因素不受限制时，模型的测试损失 $L$ 与模型参数量 $N$、数据集大小 $D$ 和计算量 $C$ 分别呈现出幂律关系，这些关系可以被精确地数学建模：

1.  **模型大小 (N)**：在数据量充足的情况下，损失随模型非嵌入参数量 $N$ 的增加而降低。

$$
L(N)=\left(N_{\mathrm{c}} / N\right)^{\alpha_N} ; \alpha_N \sim 0.076, \quad N_{\mathrm{c}} \sim 8.8 \times 10^{13} \text { (non-embedding parameters) }
$$

2.  **数据集大小 (D)**：对于 LLM，在训练至收敛（通过早停）时，损失随数据集大小 $D$ 的增加而降低。

$$
L(D)=\left(D_{\mathrm{c}} / D\right)^{\alpha_D} ; \alpha_D \sim 0.095, \quad D_{\mathrm{c}} \sim 5.4 \times 10^{13} \text { (tokens) }
$$

3.  **训练计算量 (C)**：在最优模型大小和充足数据下，损失随最小计算量 $C_{\min}$ 的增加而降低。

$$
L\left(C_{\min }\right)=\left(C_{\mathrm{c}}^{\min } / C_{\min }\right)^{\alpha_C^{\min }} ; \alpha_C^{\min } \sim 0.050, \quad C_{\mathrm{c}}^{\min } \sim 3.1 \times 10^8 \text { (PF-days) }
$$

其中，$N_c, D_c, C_c^{\min}$ 是常数，而 $\alpha_N, \alpha_D, \alpha_C^{\min}$ 是尺度指数，决定了性能提升的速度。这项工作还发现了一些重要结论：

- **更大的模型更具样本效率**：在达到相同的损失水平时，更大的模型需要的训练样本更少。在计算预算增加时，应优先增加模型参数量 $N$，而非数据量 $D$（建议 $N \propto C^{0.73}, D \propto C^{0.27}$）。

{{< figure
    src="scaling_law_optimal_parameters.png"
    caption="Fig. 2. Optimal parameters for compute-efficient training. (Image source: [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361))"
    align="center"
    width="100%"
>}}

- **模型形状影响甚微**：在参数总量固定的前提下，模型的深度和宽度等具体架构设计对性能的影响远小于尺度本身。
- **C ≈ 6ND**：在 Decoder-only Transformer 模型中，训练计算量（FLOPs）可以通过一个简洁的公式估算。模型拥有 $N$ 个参数时，前向传播处理一个 token 大约需要 $2N$ 次浮点运算，而反向传播大约是前向的两倍，因此训练每个 token 的成本约为 $6N$ FLOPs。若训练语料包含 $D$ 个 token，则总训练计算量近似为 $6ND$。

### 多模态

**Scaling Laws for Autoregressive Generative Modeling** ([Henighan et al., 2020](https://arxiv.org/abs/2010.14701)) 验证了 scaling law 并非语言模型独有，也适用于**多模态模型**。自回归 Transformer 在图像、视频及多模态任务中，其性能均随规模增加呈现可预测的幂律提升。

{{< figure
    src="autoregressive_scaling_law.png"
    caption="Fig. 3. Smooth scaling of reducible loss across different domains (Image, Video, Multimodal). (Image source: [Henighan et al., 2020](https://arxiv.org/abs/2010.14701))"
    align="center"
    width="100%"
>}}

### Chinchilla

**Chinchilla** ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) 修正了 OpenAI 的结论。通过训练超过 400 个模型，他们发现 OpenAI 由于使用了固定的学习率调度策略，导致并未找到真正的最优解。

Chinchilla 定律指出**为了实现计算最优（Compute-Optimal），模型参数量 $N$ 和训练数据量 $D$ 应该等比例扩展**。即：
$$ N \propto C^{0.5}, \quad D \propto C^{0.5} $$

这意味着，在给定计算预算下，大多数现有的大模型（如 GPT-3, Gopher）都属于 *undertrained*。遵循此定律训练出的 Chinchilla (70B) 在使用与 Gopher (280B) 相同计算量的情况下（数据量扩大 4 倍），性能全面超越后者。

{{< figure
    src="Chinchilla.png"
    caption="Fig. 4. Training curve envelope and optimal model size/token count projections. (Image source: [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556))"
    align="center"
    width="100%"
>}}

### GPT-4

OpenAI 在 **GPT-4** ([OpenAI, 2023](https://arxiv.org/abs/2303.08774)) 的技术报告中展示了 Scaling Law 的重要应用**通过小模型预测大模型的最终性能**。他们利用计算量仅为 GPT-4 万分之一的小模型，精确预测了 GPT-4 的最终 loss。其拟合公式引入了**不可约损失(irreducible loss)** $c$：

$$
L(C)=a C^b+c
$$

**参数定义：**

* $C$：训练计算量。
* $L(C)$：在计算量 $C$ 下的模型损失。
* $a$：**尺度因子 (Scale Coefficient)**，控制损失随计算量增加下降的整体幅度。
* $b$：**幂律指数 (Scaling Exponent)**，决定损失下降的速率；指数越大，损失下降越快。
* $c$：**不可约损失 (Irreducible Loss)**，反映数据固有熵，即无论投入多少计算量也无法继续降低的误差下限。

{{< figure
src="gpt4_loss.png"
caption="Fig. 5. GPT-4’s final loss on an internal code-token dataset aligns with a power-law trend extrapolated from smaller models when training compute is normalized. (Image source: [OpenAI, 2023](https://arxiv.org/abs/2303.08774))"
align="center"
width="100%"
>}}

在以归一化训练计算量为横轴、bits-per-word 为纵轴的图中，小模型训练得到的多个 loss 点呈现出高度稳定的幂律直线关系，而由这些点拟合出的幂律曲线（未使用任何 GPT-4 数据）却几乎精确地落在 GPT-4 的最终损失位置上。

## RM Scaling Laws

在 RLHF 流程中，奖励模型（RM）充当人类偏好的代理。然而，RM 并非完美的判别器。[Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law) 指出：**当一个指标成为目标时，它就不再是一个好的指标**。在 LLM 训练中过度优化这个代理 RM 会导致在真实目标上的性能下降。

**Scaling Laws for Reward Model Overoptimization** ([Gao et al., 2023](https://arxiv.org/abs/2210.10760)) 采用了一种合成数据的设置， 将一个 60 亿参数的 Gold RM 作为真实奖励 $R$，并使用参数量从 300 万到 30 亿的不同代理 RM 作为优化目标。

{{< figure
    src="rm_scaling_laws.png"
    caption="Fig. 6. Scaling laws for reward model overoptimization for Best-of-N and RL. (Image source: [Gao et al., 2023](https://arxiv.org/abs/2210.10760))"
    align="center"
    width="100%"
>}}

从初始策略到优化后策略的 KL 散度定义为：

$$
\mathrm{KL}:=D_{\mathrm{KL}}\left(\pi \| \pi_{\mathrm{init}}\right)
$$

并定义距离函数:

$$
d:=\sqrt{D_{\mathrm{KL}}\left(\pi \| \pi_{\mathrm{init}}\right)}
$$

Gold RM 分数 $R$ 的变化遵循不同的函数形式，具体取决于优化方法:

$$
R_{\mathrm{BoN}}(d) = d(\alpha_{\mathrm{BoN}} - \beta_{\mathrm{BoN}} d)
$$

**Best-of-N (BoN) 拒绝采样** 呈现出先增后减的**二次方衰减**关系。在优化初期真实奖励提升，但超过某一最优点后，奖励随优化加深而显著下降。

$$
R_{\mathrm{RL}}(d) = d(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}} \log d)
$$

**RL** 的衰减项为 $\log d$，下降速度比 BoN 显著更慢，呈现出**对数衰减**关系。


{{< figure
    src="rm_size.png"
    caption="Fig. 7. The values of  $\alpha_{\text {bon }}$, $\beta_{\text {bon }}$ and $\beta_{\mathrm{RL}}$ in the BoN and RL overoptimization scaling laws for both proxy (dashed line) and gold (solid line) rewards as they scale with parameter count. (Image source: [Gao et al., 2023](https://arxiv.org/abs/2210.10760))"
    align="center"
    width="100%"
>}}

上述公式中的系数 $\alpha$ 和 $\beta$ 描述了初始优化效率和过优化的严重程度，并且可以得出以下结论：

1. **RM 参数量至关重要**：增加 RM 的参数量可以有效地提升 $\alpha$ 并降低 $\beta$，从而缓解过优化问题，获得更高的真实奖励峰值。
2. **RL 比 BoN 更抗过优化**：以 KL 散度为度量，RL 在优化和过优化上都比 BoN 慢。

## Test-time Scaling

除了在训练阶段投入算力，还可以在推理阶段通过增加计算量来提升性能。这被称为 **Test-Time Scaling** ([Snell et al.,  2024](https://arxiv.org/abs/2408.03314))。主要策略包括并行采样（Parallel Sampling）和串行修正（Sequential Refinement）。

*   **并行采样（Parallel Sampling）**：例如 Best-of-N 或多数投票（Majority Voting）。模型针对同一问题生成多个独立样本，然后通过验证器或投票选出最佳答案。
*   **串行修正（Sequential Refinement）**：如自我修正（Self-Correction），模型基于上一次输出进行迭代改进。

### 效率权衡
{{< figure
    src="test_time_scaling.png"
    caption="Fig. 8. Compute-Optimal Scaling for Iterative Self-Refinement and Search: Evaluating Efficiency Against Best-of-N Baselines and Analyzing the Trade-offs Between Test-Time Compute and Pretraining Scaling (Image source: [Snell et al., 2024](https://arxiv.org/abs/2408.03314))"
    align="center"
    width="100%"
>}}

从上图可以得出以下结论：

1. **最佳策略的选择由问题难度决定，且存在收益边界**: 虽然计算最优策略总体上比 Best-of-N 基线更高效（节省约 4 倍算力），但其具体效能高度依赖于任务的复杂度。
* **简单问题：** 模型的初始回答通常在大方向上正确，仅需局部微调。此时，串行修正是最高效的策略，能以极小的算力代价快速修复细节。
* **困难问题：** 尽管并行搜索能探索更多路径，但随着问题难度提升，单纯增加测试时推理（无论是搜索还是修正）的收益会迅速**边际递减**。对于极难问题，测试时算力很难突破模型能力的天花板。

2. **测试时算力无法完全替代预训练，基座能力依然是根本**: 测试时算力（Test-time Compute）与预训练算力（Pretraining Compute）并非 1:1 等价交换。
* **交换的先决条件（$Y \ll X$）：** 只有当推理端的 Token 预算（$Y$）远小于预训练端的 Token 预算（$X$），且面对的是中低难度问题时，增加推理时间才是划算的。
* **基座模型的不可替代性：** 一旦问题变难或推理需求过高，小模型的推理技巧（搜索/修正）无法填补巨大的能力鸿沟。扩大预训练规模提供的广泛知识与强推理能力，依然是解决复杂问题的决定性根基。

### Simple test-time scaling

**s1: Simple test-time scaling** ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)) 收集了包含 1000 个配有推理过程的问题数据集 [s1K](https://huggingface.co/datasets/simplescaling/s1K)，之后以 Qwen2.5-32B-Instruct 为基座模型进行监督微调训练，其中采用了**Budget Forcing** 的方法来控制输出 token 的长度以此研究 test-time scaling。

{{< figure
    src="s1_sample.png"
    caption="Fig. 9. Budget forcing in s1-32B: suppressing the end-of-thinking delimiter prompts the model to continue after “...is 2.”, triggering a self-correction via “Wait”. (Image source: [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))"
    align="center"
    width="100%"
>}}

* **延长思考（Lengthening）**：当模型试图结束思考时，抑制结束标识符，并强制追加 `Wait` 等词语，鼓励模型进行反思和二次检查。
* **缩短思考（Shortening）**：当思考过程超过预设 Token 长度时，强制追加结束标识符（如 `</think>` 或 `"Final Answer:"`），促使模型立即输出结论。


{{< figure
    src="s1_result.png"
    caption="Fig. 10. Sequential and parallel test-time scaling. (Image source: [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))"
    align="center"
    width="100%"
>}}

上述结果显示，并行扩展和顺序扩展方法通过 Budget Forcing 延长的思考时间（Token 数）与下游任务准确率呈显著的正相关。

## Scale RL

**ScaleRL**（[Khatri et al., 2025](https://arxiv.org/abs/2510.13786)）提出了用于分析和预测 LLM 中 RL 扩展的系统性方法。与预训练中常见的幂律关系不同，RL 训练的性能（如验证集上的预期奖励 $R_C$）随计算量 $C$ 的增长更符合一个**S型（Sigmoidal）饱和曲线**。该曲线可以用以下公式描述：

$$
\overbrace{R_C-R_0}^{\text {Reward Gain }}=\overbrace{\left(A-R_0\right)}^{\text {Asymptotic Reward Gain }} \times \frac{1}{\underbrace{1+\left(C_{\text {mid }} / C\right)^B}_{\text {Compute Efficiency }}}
$$

**参数定义：**
*   $R_C$: 在计算量 $C$ 下的预期奖励（或 Pass Rate）。
*   $R_0$: 初始模型的性能。
*   $A$: **渐近性能上限 (Asymptotic Performance)**。代表在无限计算量下模型能达到的理论最高性能。
*   $B$: **计算效率 (Scaling Exponent)**。控制曲线的陡峭程度，值越大代表模型能越快达到性能上限。
*   $C_{mid}$: 达到总收益一半时所需的计算量。

{{< figure
    src="rl_scaling_fit.png"
    caption="Fig. 11. Interpretation of the sigmoidal scaling curve for RL. (Image source: [Khatri et al., 2025](https://arxiv.org/abs/2510.13786))"
    align="center"
    width="100%"
>}}

上图展示了这个框架的价值: 它允许研究人员通过在较小计算规模下的早期训练数据来拟合曲线，从而**预测**一个 RL 配方在更大计算预算下的最终性能和效率，降低了算法探索的成本。


通过对多种 RL 设计选择（如损失函数、优势归一化、数据课程等）进行大规模消融实验，总结出几条关键原则：

1.  **性能天花板并非普适**：不同的 RL 算法（如 GRPO, DAPO, CISPO）具有不同的渐进性能上限 $A$。选择正确的损失函数至关重要。
2.  **效率与上限的分离**：许多常见的 RL 技巧，如优势归一化、数据课程、长度惩罚等，主要影响计算效率（$B$ 和 $C_{\text{mid}}$），而对性能上限 $A$ 的影响不大。
3.  **稳定可预测的扩展**：一个精心设计的、稳定的 RL 配方能够遵循可预测的 S 型曲线轨迹，即使扩展到十万 GPU 小时的级别，其表现也与早期拟合的曲线高度一致。

{{< figure
    src="scale_rl_result.png"
    caption="Fig. 12. ScaleRL demonstrates more scalable performance compared to other prevalent RL methods. (Image source: [Khatri et al., 2025](https://arxiv.org/abs/2510.13786))"
    align="center"
    width="90%"
>}}

## 参考文献

[1] Kaplan, Jared, et al. ["Scaling laws for neural language models."](https://arxiv.org/abs/2001.08361) arXiv preprint arXiv:2001.08361 (2020).

[2] Henighan, Tom, et al. ["Scaling laws for autoregressive generative modeling."](https://arxiv.org/abs/2010.14701) arXiv preprint arXiv:2010.14701 (2020). 

[3] Hoffmann, Jordan, et al. ["Training compute-optimal large language models."](https://arxiv.org/abs/2203.15556) arXiv preprint arXiv:2203.15556 (2022).

[4] Achiam, Josh, et al. ["Gpt-4 technical report."](https://arxiv.org/abs/2303.08774) arXiv preprint arXiv:2303.08774 (2023).

[5] Gao, Leo, John Schulman, and Jacob Hilton. ["Scaling laws for reward model overoptimization."](https://arxiv.org/abs/2210.10760) International Conference on Machine Learning. PMLR, 2023.

[6] Snell, Charlie, et al. ["Scaling llm test-time compute optimally can be more effective than scaling model parameters."](https://arxiv.org/abs/2408.03314) arXiv preprint arXiv:2408.03314 (2024).

[7] Muennighoff, Niklas, et al. ["s1: Simple test-time scaling."](https://arxiv.org/abs/2501.19393) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025.

[8] Khatri, Devvrit, et al. ["The art of scaling reinforcement learning compute for llms."](https://arxiv.org/abs/2510.13786) arXiv preprint arXiv:2510.13786 (2025).

## 引用

> **引用声明**：转载或引用本文内容时，请注明原作者与来源。

**Cited as:**

> Yue Shui. (Nov 2025). Scaling Laws.
> https://syhya.github.io/zh/posts/2025-11-19-scaling-law/

Or

```bibtex
@article{yue_shui_scaling_laws_2025,
  title   = {Scaling Laws},
  author  = {Yue Shui},
  journal = {syhya.github.io},
  year    = {2025},
  month   = {November},
  url     = {https://syhya.github.io/zh/posts/2025-11-19-scaling-law/}
}
```