---
title: "Self-Evolving Agents"
date: 2026-02-20T12:00:00+08:00
lastmod: 2026-02-20T12:00:00+08:00
author: "Yue Shui"
tags: ["Agent Evolve", "AlphaEvolve", "OpenEvolve", "AI for Science"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

最近，AI 领域正在经历一次关键的结构性转变：Agent 的核心竞争力，正从一次性生成正确答案，转向在闭环系统中持续产生可验证、可进化的新结果。这一转变的标志性事件是 DeepMind 发布的 [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 通过 LLM 驱动的进化编码代理，在数学、算法与工程优化领域取得了多项突破，在部分任务上超越了人类已知最优解。在这一框架下，人类与 Agent 的分工发生了明确重构：

- 人类负责定义 **What** —— 设计评估标准、提供初始候选方案，并将必要的背景知识以 context 形式注入模型。
- Agents 负责探索 **How** —— 通过生成代码并调用外部工具，自主搜索并发现更优的结构与算法实现路径。  

{{< figure
    src="alpha-evolve-high-level.png"
    caption="Fig. 1. AlphaEvolve high-level overview. (Image source: [Novikov et al., 2025](https://arxiv.org/abs/2506.13131))"
    align="center"
    width="80%"
>}}

## FunSearch

通常，人们通过精心设计 Prompt 引导 LLM 一次性生成目标结果，其输出质量主要取决于模型能力和 Prompt 设计水平。该模式在问答、摘要等任务中效果显著，但在需要探索解空间或寻找超越当前最优解的场景中存在一定局限。**FunSearch**（[Romera-Paredes et al., 2024](https://www.nature.com/articles/s41586-023-06924-6)）强调通过迭代闭环的方式，让模型在外部环境中不断试错、评估和改进得到最优程序。

{{< figure
    src="fun-search-arch.png"
    caption="Fig. 2. The overview of FunSearch. (Image source: [Romera-Paredes et al., 2024](https://www.nature.com/articles/s41586-023-06924-6))"
    align="center"
    width="90%"
>}}

FunSearch 是一个有状态的迭代闭环：

$$\text{Specification} \rightarrow \text{Program Generation} \rightarrow \text{Evaluation} \rightarrow \text{Program Database Update} \rightarrow \text{Next Iteration}$$

这与传统单次生成范式存在三个本质差异：

- **外部可验证**：验证分数来自真实执行器（代码运行、数学验证、性能测试）。
- **可累积改进**：每一轮迭代都基于上一轮的最优解进行改进，具备可观测的收敛趋势。
- **可治理**：沙箱执行、审批机制与规则约束可以嵌入循环各个环节，确保过程安全、结果可控。

## AlphaEvolve

**AlphaEvolve** ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131))是 DeepMind 推出的新一代进化编码智能体。其核心架构建立在一个闭环系统之上：LLM 用于生成和修改候选程序，评估器提供任务特定性能信号，进化算法基于评估结果执行选择与变异，从而在程序空间中进行迭代优化。

{{< figure
    src="alpha-evolve-arch.png"
    caption="Fig. 3. The overall view of the AlphaEvolve discovery process. (Image source: [Novikov et al., 2025](https://arxiv.org/abs/2506.13131))"
    align="center"
    width="95%"
>}}

相较于主要针对函数级优化的先前方法 FunSearch，AlphaEvolve 将搜索空间扩展至跨函数和跨模块的程序结构。借助 SOTA LLM 的长上下文推理能力，AlphaEvolve 显著扩大了可搜索程序空间，从而提升复杂算法发现任务的性能上限。

| **项目**       | FunSearch             | AlphaEvolve         |
| -------- | --------------------- | ------------------- |
| 进化范围     | 进化单个函数                | 进化整个代码文件          |
| 代码规模     | 最多进化 10–20 行代码        | 最多可进化数百行代码          |
| 编程语言     | 仅进化 Python 代码         | 可进化任意编程语言           |
| 评估需求     | 需要快速评估（单 CPU ≤ 20 分钟） | 可在加速器上并行运行，评估可持续数小时 |
| LLM 采样量  | 使用数百万次 LLM 采样         | 数千次 LLM 采样即可        |
| LLM 规模依赖 | 使用小模型；更大模型无明显收益       | 能从最先进（SOTA）大模型中获益   |
| 上下文信息    | 上下文极少（仅包含历史解）         | 提示中包含丰富上下文和反馈       |
| 优化目标     | 优化单一指标                | 可同时优化多个指标           |

### OpenEvolve

[OpenEvolve](https://github.com/codelion/openevolve) 提供了 AlphaEvolve 的高质量开源工程实现，完整的实现了四个核心模块：

{{< figure
    src="openevolve-architecture.png"
    caption="Fig. 4. The OpenEvolve architecture: showing the integration of LLMs, MAP-Elites population database, cascade evaluator, and evolution controller. (Image source: [OpenEvolve](https://github.com/codelion/openevolve))"
    align="center"
    width="80%"
>}}

**Prompt Sampler**：从程序数据库中采样历史解以构建丰富的上下文，不仅包含当前最优解，还引入多样化的次优解作为参考，避免 LLM 陷入单一模式；同时结合 meta prompts 机制，使 LLM 不仅用于生成答案，还能参与 prompt 本身的持续优化，从而提升整体推理质量。

**LLM Ensemble**：大小模型协同工作，例如使用高吞吐量小模型负责广泛探索，高推理质量大模型负责精细改写。这种集成策略在探索与利用之间取得平衡。

**Evaluator Pool**：支持确定性测试、级联式统计假设检验、LLM 辅助反馈信号以及并行化评估，以提升评估效率和吞吐量。评估结果用于引导后续 LLM 生成，使模型能够基于错误信号持续优化。

**Program Database**：基于 MAP-Elites ([Mouret & Clune, 2015](https://arxiv.org/abs/1504.04909)) 和 [多岛模型(island-based population model)](https://en.wikipedia.org/wiki/Population_model_(evolutionary_algorithm))维护解的种群。MAP-Elites 将解空间映射到用户定义的多维特征网格中，并在每个网格单元中保留该区域内适应度最高的个体，从而同时提升解集的质量和多样性。

系统控制器以异步流水线方式协调各组件交互，最大化系统吞吐量，从而在单位时间内评估尽可能多的候选解。

### 消融实验

下图消融实验清晰地揭示了各组件的贡献。在矩阵乘法张量分解和空间球堆积两个任务上，移除任何一个组件都会导致性能下降：

{{< figure
    src="alpha-evolve-ablations.png"
    caption="Fig. 5. AlphaEvolve ablation results on matrix multiplication tensor decomposition and Kissing. (Image source: [Novikov et al., 2025](https://arxiv.org/abs/2506.13131))"
    align="center"
    width="100%"
>}}

| 实验设置                                 | 修改内容               | 性能影响                       |
| ------------------------------------ | ------------------ | -------------------------- |
| 完整 AlphaEvolve                   | 无                  | 性能最佳                       |
| 无进化机制    | 移除进化搜索，仅重复输入初始程序   | 性能最差；证明进化机制是系统核心驱动力        |
| 无上下文提示  | 移除问题特定上下文信息        | 性能大幅下降；说明上下文对 LLM 生成质量至关重要 |
| 仅使用小模型  | 使用小规模模型替代 SOTA 大模型 | 性能受限；强推理能力模型决定性能上限         |
| 无全文件进化  | 仅进化单个函数，而非整个代码文件   | 性能明显下降；全局跨函数协同优化更重要        |
| 无元提示进化  | 禁用元提示进化机制          | 性能中等下降；Prompt 自优化可提升最终效果上限 |

### 成果

AlphaEvolve 的成果横跨数学发现和工程优化两个维度：

**数学发现**：在超过 50 个开放数学问题上进行了系统实验，约 75% 情况下复现当前最优结果，并在约 20% 的问题上取得超越已有最优解的新进展。其中最具代表性的是 $4 \times 4$ 复数矩阵乘法问题：AlphaEvolve 发现了仅需 **48 次标量乘法** 的新算法，首次突破长期由 [Strassen 算法](https://en.wikipedia.org/wiki/Strassen_algorithm) 保持的 **49 次乘法** 记录，体现了基于大模型进化式搜索在复杂算法空间中的突破能力。

**工程优化**：在 Google 生产级计算基础设施中实现了多项可规模化放大的性能提升。数据中心调度方面，为 [Borg 系统](https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/)发现了新的可解释启发式函数，持续回收约 **0.7% 的全球数据中心闲置算力资源**。Gemini 训练核心方面，通过改进矩阵乘法分解策略，使关键 kernel 获得 **平均 23% 计算加速**，并直接带来约 **1% 的整体训练时间下降**，同时将传统需要数周专家调优的优化流程缩短至数天自动实验周期。

## AI for Science

近期研究表明，随着 LLM 基础能力、长思维链推理能力以及 Agentic 能力的持续提升，其在科学发现领域正展现出前所未有的潜力。以 **Gemini 3 Deep Think**([DeepMind, 2026](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/)) 和 **GPT‑5.2**([OpenAI, 2025](https://openai.com/index/introducing-gpt-5-2/)) 为代表的先进模型，已在数学、物理、生物等学科中显著提升科研效率，加速了关键问题的探索与突破。

### Aletheia

**Aletheia**([Feng et al., 2026](https://arxiv.org/abs/2602.10177)) 是一个用于数学研究智能体，它模拟了数学家的真实研究流程。其核心是一个 **生成-验证-修复** 的迭代闭环机制，在循环推理与形式化校验中不断优化解题路径与结论可靠性。

{{< figure
    src="aletheia_overview.png"
    caption="Fig. 6. Overview of Aletheia, a math research agent powered by Deep Think. It iteratively generates, verifies, and revises solutions. (Image source: [Luong & Mirrokni, 2026](https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/))"
    align="center"
    width="100%"
>}}

1.  **Generator**：利用 Deep Think 的长思维链推理能力，在当前问题状态下探索可能的求解路线，提出候选的证明步骤、引理或构造。
2.  **Verifier**：作为关键约束组件，通常由微调模型或形式化证明器实现，用于审查生成结果，定位逻辑断点、幻觉与计算/推导错误，并输出可操作的反馈。
3.  **Reviser**：根据验证反馈对解题轨迹进行更新：修补局部步骤、替换错误引理，必要时回溯至先前决策点重新搜索，从而进入下一轮迭代。

{{< figure
    src="aletheia_eval_res.png"
    caption="Fig. 7. The January 2026 Deep Think surpasses IMO-Gold on Olympiad problems, scales to PhD-level tasks, and, with Aletheia, delivers stronger reasoning at lower compute. (Image source: [Feng et al., 2026](https://arxiv.org/abs/2602.10177))"
    align="center"
    width="100%"
>}}

随着推理阶段计算资源的增加，Gemini Deep Think 在 [IMO-ProofBench](https://imobench.github.io/) 基准测试上的得分最高达到 90% ，充分验证了 [inference-time scaling law](https://syhya.github.io/zh/posts/2025-11-19-scaling-law/#test-time-scaling) 的有效性。定律不仅适用于奥赛级问题，也可迁移至博士级难度的 FutureMath Basic 基准测试。Aletheia 在更低的推理计算开销下实现了更高的推理质量。

{{< figure
    src="aletheia_research_output.png"
    caption="Fig. 8. The work proposes a taxonomy for AI-assisted mathematics based on research significance and AI contribution, reports several Level 0–2 results with Level 2 papers submitted to journals, and currently claims no Level 3 or 4 breakthroughs. (Image source: [Feng et al., 2026](https://arxiv.org/abs/2602.10177))"
    align="center"
    width="100%"
>}}

Aletheia 在前沿数学研究中已产出多项达到 Level 2 的成果，部分论文已投稿，同时实现了若干自主完成的 Level 0–1 级结果。尽管尚未取得重大或里程碑式突破，但已展现出稳定产出研究级成果的能力。

### 前沿研究进展

OpenAI 在 **Early Science Acceleration Experiments with GPT-5**（[Bubeck et al., 2025](https://arxiv.org/abs/2511.16072)）中展示了 GPT-5 在真实科研环境中的跨学科协作能力。报告汇集了数学、物理、天文学、计算机科学、生物医学与材料科学等多个领域的案例，记录模型如何在专家引导下参与前沿问题的探索与突破。

与此同时，DeepMind 在 **Accelerating Research with Gemini**（[Woodruff et al., 2026](https://arxiv.org/abs/2602.03837)）中呈现了前沿 LLM 作为“研究合作者”进入理论研究流程的实践，覆盖数学、理论计算机科学、物理与经济学等方向。模型已深度参与假设构建、路径搜索、证明生成与严谨性检验等核心科研环节。

这些案例共同表明，前沿 LLM 正在嵌入科学推理的核心链条：从提出研究思路、重构证明路径，到进行深度文献综合、识别潜在漏洞，最后生成具备发表价值的研究成果。最近 GPT-5 进一步被嵌入自动化实验系统，通过机器臂形成完整的 **AI 驱动自主实验闭环**，实现从假设生成到物理验证的持续迭代优化。

{{< figure
    src="gpt5_driven_auto_lab.png"
    caption="Fig. 9. GPT-5-driven autonomous laboratory workflow. (Image source: [Smith et al., 2026](https://www.biorxiv.org/content/10.64898/2026.02.05.703998v1))"
    align="center"
    width="100%"
>}}

* **实验设计生成**：GPT-5 基于历史数据与文献进行数据分析与生化推理，批量生成 384 孔板格式的实验方案。
* **结构化校验**：实验方案被编码为 Pydantic 对象，进行字段、剂量与设备可执行性验证，避免幻觉实验。
* **自动化执行**：通过 Catalyst 协议转化为机器指令，在 RAC 系统中完成加样、孵育与检测。
* **数据回流分析**：实验数据与元数据自动回传至 GPT-5，用于性能评估、假设更新与下一轮实验设计。

基于上述理论与实验案例，可以提炼出一套可复用的 AI 辅助研究方法论：

* **迭代式精炼**：通过多轮交互逐步修正错误、补充假设与收敛推理路径，在连续反馈中逼近严谨结论。
* **问题分解**：将复杂开放问题拆解为可验证的子命题或关键计算模块，降低单步推理失败风险。
* **跨领域迁移**：利用模型的广谱知识结构，建立不同学科之间的概念映射与工具复用，突破证明瓶颈。
* **反例构造与仿真验证**：通过实例生成、代码验证或小规模数值模拟，快速排除错误方向。
* **形式化与严谨检查**：将高层证明草稿扩展为可发表级别的严谨文本，系统检查符号一致性与逻辑闭环。
* **Agentic 工具闭环**：将模型嵌入代码执行或实验系统，实现“生成—执行—反馈—修正”的自动化推理闭环。

整体来看，AI for Science 正在经历从辅助智能到协作智能，再到闭环智能的范式跃迁。

## 总结

以 FunSearch、AlphaEvolve 和 Aletheia 为代表的自我进化智能体证明：将大语言模型嵌入**生成、验证与修正**的迭代闭环中，能够有效打破模型单次推理的能力天花板，在数学发现与工程优化等复杂解空间内，探索出超越已有认知的新结果。

随着模型长逻辑推理能力与 Agentic 工具链（如自动化实验室）的深度融合，**Self-Evolving Agents** 正在从被动的辅助工具进化为主动的科学合作者。这种具备高度自主探索能力的闭环系统，不仅重塑了人类与 AI 的协作分工，也将成为未来推动 AI for Science 产生颠覆性突破的核心驱动力。

## 参考文献

[1] Novikov, Alexander, et al. ["Alphaevolve: A coding agent for scientific and algorithmic discovery."](https://arxiv.org/abs/2506.13131) arXiv preprint arXiv:2506.13131 (2025).

[2] Romera-Paredes, Bernardino, et al. ["Mathematical discoveries from program search with large language models."](https://www.nature.com/articles/s41586-023-06924-6) Nature 625.7995 (2024): 468-475.

[3] Asankhaya Sharma. [OpenEvolve: Open-source implementation of AlphaEvolve](https://github.com/codelion/openevolve). GitHub (2025).

[4] Mouret, Jean-Baptiste, and Jeff Clune. ["Illuminating search spaces by mapping elites."](https://arxiv.org/abs/1504.04909) arXiv preprint arXiv:1504.04909 (2015).

[5] Verma, Abhishek, et al. ["Large-scale cluster management at Google with Borg."](https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/) Proceedings of the tenth european conference on computer systems. 2015.

[6] DeepMind. ["Gemini 3 Deep Think: Advancing science, research and engineering"](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/) Google Blog (2026).

[7] OpenAI. ["Introducing GPT-5.2."](https://openai.com/index/introducing-gpt-5-2/) OpenAI Blog (2025).

[8] Feng, Tony, et al. ["Towards Autonomous Mathematics Research."](https://arxiv.org/abs/2602.10177) arXiv preprint arXiv:2602.10177 (2026).

[9] Luong, Thang, and Vahab Mirrokni. ["Accelerating mathematical and scientific discovery with Gemini Deep Think."](https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/) Google DeepMind Blog (2026).

[10] Bubeck, Sébastien, et al. ["Early science acceleration experiments with GPT-5."](https://arxiv.org/abs/2511.16072) arXiv preprint arXiv:2511.16072 (2025).

[11] Woodruff, David P., et al. ["Accelerating Scientific Research with Gemini: Case Studies and Common Techniques."](https://arxiv.org/abs/2602.03837) arXiv preprint arXiv:2602.03837 (2026).

[12] Smith, Alexus A., et al. ["Using a GPT-5-driven autonomous lab to optimize the cost and titer of cell-free protein synthesis."](https://www.biorxiv.org/content/10.64898/2026.02.05.703998v1) bioRxiv (2026): 2026-02.

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Feb 2026). Self-Evolving Agents.
https://syhya.github.io/zh/posts/2026-02-20-self-evolving-agents

Or

```bibtex
@article{syhya2026-self-evolving-agents,
  title   = "Self-Evolving Agents",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "Feb",
  url     = "https://syhya.github.io/zh/posts/2026-02-20-self-evolving-agents"
}
```



