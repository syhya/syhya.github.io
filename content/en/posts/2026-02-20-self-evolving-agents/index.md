---
title: "Self-Evolving Agents"
date: 2026-02-20T12:00:00+08:00
lastmod: 2026-02-20T12:00:00+08:00
author: "Yue Shui"
tags: ["Agent Evolve", "AlphaEvolve", "OpenEvolve", "AI for Science"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

A structural shift is underway in AI: the core capability of agents is moving from **one-shot answer generation** to continually producing verifiable, self-improving results in **closed-loop systems**. A representative milestone is DeepMind's release of [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), an LLM-driven evolutionary coding agent that has delivered breakthroughs in mathematics, algorithm design, and engineering optimization, in several cases improving upon best-known human-designed baselines. Under this paradigm, the division of labor between humans and agents is clearly reconfigured:

- Humans are responsible for defining the **What** — setting evaluation criteria, providing initial candidate solutions, and injecting essential background knowledge as *context* into the model.
- Agents are responsible for figuring out the **How** — autonomously searching for and discovering better structures and algorithmic implementations by generating code and invoking external tools.

{{< figure
    src="alpha-evolve-high-level.png"
    caption="Fig. 1. AlphaEvolve high-level overview. (Image source: [Novikov et al., 2025](https://arxiv.org/abs/2506.13131))"
    align="center"
    width="80%"
>}}

## FunSearch

Many LLM workflows rely on prompt engineering to elicit the desired output in a single pass, with the quality of the results largely determined by the model’s capabilities and the effectiveness of the prompt design. While highly effective for tasks like Q&A and summarization, this approach has limitations in scenarios requiring the exploration of a search space or finding solutions that surpass the current state-of-the-art (SOTA). **FunSearch** ([Romera-Paredes et al., 2024](https://www.nature.com/articles/s41586-023-06924-6)) emphasizes an iterative closed-loop approach, allowing the model to repeatedly generate, evaluate, and refine programs in an external environment.

{{< figure
    src="fun-search-arch.png"
    caption="Fig. 2. The overview of FunSearch. (Image source: [Romera-Paredes et al., 2024](https://www.nature.com/articles/s41586-023-06924-6))"
    align="center"
    width="90%"
>}}

FunSearch is a stateful iterative closed loop:

$$\text{Specification} \rightarrow \text{Program Generation} \rightarrow \text{Evaluation} \rightarrow \text{Program Database Update} \rightarrow \text{Next Iteration}$$

This presents three fundamental differences from the traditional single-pass generation paradigm:

- **Externally Verifiable**: Evaluation scores come from real executors (code execution, mathematical verification, performance testing).
- **Cumulative Improvement**: Each iteration builds on the best solutions so far, often yielding consistent gains over time.
- **Governable**: Sandbox execution, approval mechanisms, and rule constraints can be embedded into the loop, ensuring safety and controllability.

## AlphaEvolve

**AlphaEvolve** ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131)) is DeepMind's next-generation evolutionary coding agent. Its core architecture orchestrates a closed-loop pipeline: LLMs generate and modify candidate programs, evaluators provide task-specific performance signals, and an evolutionary algorithm performs selection and mutation based on those signals—iteratively optimizing within the program space.

{{< figure
    src="alpha-evolve-arch.png"
    caption="Fig. 3. The overall view of the AlphaEvolve discovery process. (Image source: [Novikov et al., 2025](https://arxiv.org/abs/2506.13131))"
    align="center"
    width="95%"
>}}

Compared to its predecessor FunSearch, which primarily focused on single-function optimization, AlphaEvolve expands the search space to entire codebases spanning multiple functions and components. Leveraging the long-context reasoning capabilities of SOTA LLMs, AlphaEvolve significantly enlarges the searchable program space, thereby raising the performance ceiling for complex algorithm discovery tasks.

| **Feature**        | FunSearch                                 | AlphaEvolve                                          |
| :------------- | :---------------------------------------- | :--------------------------------------------------- |
| Scope          | evolves single function                   | evolves entire code file                             |
| Size           | evolves up to 10-20 lines of code         | evolves up to hundreds of lines of code              |
| Language       | evolves code in Python                    | evolves any language                                 |
| Evaluation     | needs fast evaluation (≤ 20min on 1 CPU)  | can evaluate for hours, in parallel, on accelerators |
| LLM Samples    | millions of LLM samples used              | thousands of LLM samples suffice                     |
| LLM Type       | small LLMs used; no benefit from larger   | benefits from SOTA LLMs                              |
| Prompt Context | minimal context (only previous solutions) | rich context and feedback in prompts                 |
| Objective      | optimizes single metric                   | can simultaneously optimize multiple metrics         |

### OpenEvolve

[OpenEvolve](https://github.com/codelion/openevolve) provides a high-quality open-source engineering implementation of AlphaEvolve, fully realizing its four core modules:

{{< figure
    src="openevolve-architecture.png"
    caption="Fig. 4. The OpenEvolve architecture: showing the integration of LLMs, MAP-Elites population database, cascade evaluator, and evolution controller. (Image source: [OpenEvolve](https://github.com/codelion/openevolve))"
    align="center"
    width="80%"
>}}

**Prompt Sampler**: Samples previously discovered solutions from the program database to construct rich context prompts. It includes not only the current best solution but also diverse sub-optimal alternatives as inspiration, preventing the LLM from falling into a single mode. Combined with a **meta prompting** mechanism, the LLM is used not only to generate answers but also to co-evolve the instructions and context themselves, thereby improving overall reasoning quality.

**LLM Ensemble**: coordinates small and large models working synergistically—for example, a high-throughput smaller model handles broad exploration (increasing the rate of candidate generation), while a higher-reasoning large model focuses on occasional, high-quality rewrites. This ensemble strategy balances exploration and exploitation.

**Evaluator Pool**: supports deterministic tests, cascaded statistical hypothesis testing, LLM-assisted feedback signals, and parallelized evaluation to improve throughput and efficiency. Evaluation results guide subsequent LLM generations, enabling continuous hill-climbing driven by error signals.

**Program Database**: maintains a population of solutions using **MAP-Elites** ([Mouret & Clune, 2015](https://arxiv.org/abs/1504.04909)) and an [island-based population model](https://en.wikipedia.org/wiki/Population_model_(evolutionary_algorithm)). MAP-Elites maps the solution space onto a user-defined multidimensional feature grid, retaining the highest-fitness individual in each cell—improving both solution quality and diversity simultaneously.

A system **Controller** coordinates interactions among all components via an asynchronous pipeline, optimized for throughput to maximize the number of candidate solutions evaluated within a given compute budget.

### Ablations

The ablation studies below clearly reveal each component's contribution. On both the task of finding tensor decompositions for faster matrix multiplication and computing lower bounds on kissing numbers (sphere packing), removing any single component degrades performance:

{{< figure
    src="alpha-evolve-ablations.png"
    caption="Fig. 5. AlphaEvolve ablation results on matrix multiplication tensor decomposition and Kissing. (Image source: [Novikov et al., 2025](https://arxiv.org/abs/2506.13131))"
    align="center"
    width="100%"
>}}

| **Setting** | Modification | Performance Impact |
| :--- | :--- | :--- |
| Full method | None | Best performance |
| No evolution | Remove evolutionary search; repeatedly re-inject the same initial program | Worst performance; proves evolution is the core driving mechanism |
| No context in prompt | Remove problem-specific context information | Large performance drop; context is crucial to generation quality |
| Small base LLM only | Replace SOTA LLM with a smaller model | Performance capped; strong reasoning models set the upper bound |
| No full-file evolution | Evolve only a single function instead of the entire codebase | Noticeable drop; global cross-function co-optimization matters |
| No meta prompt evolution | Disable meta prompt evolution | Moderate drop; prompt self-improvement raises the attainable ceiling |

### Results

The achievements of AlphaEvolve span two dimensions: scientific/mathematical discovery and engineering optimization:

**Mathematical Discovery**: Systematic experiments were conducted on over 50 open mathematical problems. AlphaEvolve matched the best-known constructions in approximately 75% of cases and surpassed prior SOTA results in roughly 20%, discovering new, provably better constructions. A representative highlight is the $4 \times 4$ complex-valued matrix multiplication problem, where AlphaEvolve discovered a **48-scalar-multiplication algorithm**, improving upon the previous best-known 49-multiplication construction for this setting. This result sits within a long lineage of algebraic optimization research dating back to seminal breakthroughs such as [Strassen's algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm), which fundamentally reshaped our understanding of matrix multiplication complexity, and it underscores how LLM-guided evolutionary search can meaningfully advance classical algorithmic frontiers.

**Engineering Optimization**: Achieved multiple scalable performance improvements within Google's production-grade compute infrastructure. In data center scheduling, it discovered a new, interpretable heuristic function for the [Borg system](https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/), continuously recovering on average **0.7% of Google's fleet-wide stranded compute resources**. In Gemini's core training stack, by optimizing tiling heuristics for matrix multiplication kernels, it achieved an **average 23% kernel speedup**, directly leading to a **1% reduction in overall training time**. Furthermore, it reduced the optimization process from months of dedicated expert engineering to just days of automated experimentation.

## AI for Science

Recent research indicates that as LLMs' foundational capabilities, long chain-of-thought reasoning, and agentic abilities continue to scale, they are showing unprecedented potential in scientific discovery. Advanced models represented by **Gemini 3 Deep Think** ([DeepMind, 2026](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/)) and **GPT‑5.2** ([OpenAI, 2025](https://openai.com/index/introducing-gpt-5-2/)) have significantly improved research efficiency across disciplines like mathematics, physics, and biology, accelerating the exploration of critical problems.

### Aletheia

**Aletheia** ([Feng et al., 2026](https://arxiv.org/abs/2602.10177)) is an agent for mathematical research that simulates the authentic workflow of mathematicians. Its core is an iterative **Generate-Verify-Revise** closed-loop mechanism, continuously optimizing solution paths and conclusion reliability through cyclic reasoning and formal verification.

{{< figure
    src="aletheia_overview.png"
    caption="Fig. 6. Overview of Aletheia, a math research agent powered by Deep Think. It iteratively generates, verifies, and revises solutions. (Image source: [Luong & Mirrokni, 2026](https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/))"
    align="center"
    width="100%"
>}}

1.  **Generator**: Leverages Deep Think's long chain-of-thought reasoning to explore possible solution routes given the current problem state, proposing candidate proof steps, lemmas, or constructions.
2.  **Verifier**: Acts as a critical constraint component, typically implemented via fine-tuned models or formal provers, to scrutinize generated results, locate logical gaps, hallucinations, and calculation/derivation errors, outputting actionable feedback.
3.  **Reviser**: Updates the problem-solving trajectory based on verification feedback: patching local steps, replacing faulty lemmas, and backtracking to previous decision points to re-search when necessary, thus entering the next iteration.

{{< figure
    src="aletheia_eval_res.png"
    caption="Fig. 7. The January 2026 Deep Think surpasses IMO-Gold on Olympiad problems, scales to PhD-level tasks, and, with Aletheia, delivers stronger reasoning at lower compute. (Image source: [Feng et al., 2026](https://arxiv.org/abs/2602.10177))"
    align="center"
    width="100%"
>}}

As compute resources increase during the inference phase, Gemini Deep Think achieves up to a 90% score on the [IMO-ProofBench](https://imobench.github.io/), providing strong empirical support for [inference-time scaling law](https://syhya.github.io/posts/2025-11-19-scaling-law/#test-time-scaling). This law applies not only to Olympiad-level problems but also transfers to PhD-level tasks like the FutureMath Basic benchmark. Aletheia achieves higher reasoning quality with lower inference compute overhead.

{{< figure
    src="aletheia_research_output.png"
    caption="Fig. 8. The work proposes a taxonomy for AI-assisted mathematics based on research significance and AI contribution, reports several Level 0–2 results with Level 2 papers submitted to journals, and currently claims no Level 3 or 4 breakthroughs. (Image source: [Feng et al., 2026](https://arxiv.org/abs/2602.10177))"
    align="center"
    width="100%"
>}}


Aletheia has already produced multiple **Level 2** results in frontier mathematical research, with some papers submitted to journals, alongside several autonomously completed **Level 0–1** results. While it has not yet delivered a major milestone breakthrough, it demonstrates a stable capacity to produce research-grade outputs.

### Frontier Research Progress

In **Early Science Acceleration Experiments with GPT-5** ([Bubeck et al., 2025](https://arxiv.org/abs/2511.16072)), OpenAI showcases GPT-5's cross-disciplinary collaboration capabilities in real research environments. The report compiles case studies spanning mathematics, physics, astronomy, computer science, biomedicine, and materials science, documenting how the model—under expert guidance—contributes to exploring and making progress on frontier problems.

In parallel, DeepMind's **Accelerating Scientific Research with Gemini** ([Woodruff et al., 2026](https://arxiv.org/abs/2602.03837)) presents practical evidence of frontier LLMs entering theoretical research workflows as "research collaborators," covering mathematics, theoretical computer science, physics, and economics. The models participate deeply in hypothesis generation, path searching, proof generation, and rigor checking.

Taken together, these cases indicate that frontier LLMs are increasingly embedded into the core chain of scientific reasoning: from proposing research directions and restructuring proof strategies, to synthesizing literature in depth, identifying potential gaps, and producing research artifacts with publication-level value. More recently, GPT-5 has been integrated into automated experimental systems—closing the loop with robotic platforms to form an **AI-driven autonomous laboratory** that continuously iterates from hypothesis generation to physical validation.

{{< figure
    src="gpt5_driven_auto_lab.png"
    caption="Fig. 9. GPT-5-driven autonomous laboratory workflow. (Image source: [Smith et al., 2026](https://www.biorxiv.org/content/10.64898/2026.02.05.703998v1))"
    align="center"
    width="100%"
>}}

* **Experimental Design Generation**: GPT-5 conducts data analysis and biochemical reasoning based on historical data and literature, batch-generating experimental protocols in a 384-well plate format.
* **Structured Validation**: Experimental protocols are encoded as Pydantic objects for field, dosage, and equipment executability validation, preventing hallucinated experiments.
* **Automated Execution**: Translated into machine instructions via the Catalyst protocol, completing pipetting, incubation, and detection in the RAC system.
* **Data Backflow Analysis**: Experimental data and metadata are automatically fed back to GPT-5 for performance evaluation, hypothesis updating, and the next round of experimental design.

From these theoretical and experimental cases, a reusable methodology for AI-assisted research can be distilled:

- **Iterative refinement**: progressively correct errors, supplement hypotheses, and converge reasoning paths through multi-turn feedback, approaching rigorous conclusions incrementally.
- **Problem decomposition**: break complex open problems into verifiable sub-propositions or key computational modules, reducing the risk of single-step reasoning failures.
- **Cross-pollination**: leverage the model's broad knowledge to map concepts across disciplines and reuse tools, unlocking new routes past proof bottlenecks.
- **Counterexample & simulation**: rapidly eliminate incorrect directions through instance generation, code-based checks, or small-scale numerical simulations.
- **Rigor checks**: expand high-level proof sketches into publication-grade arguments, systematically checking symbol consistency and logical closure.
- **Agentic tool loops**: embed models into code execution or lab automation to implement an automated "generate–execute–feedback–revise" closed loop.

Overall, *AI for Science* appears to be shifting from assistive intelligence to collaborative intelligence, and further toward **closed-loop intelligence**.

## Conclusion

Self-evolving agents—exemplified by FunSearch, AlphaEvolve, and Aletheia—demonstrate that embedding large language models into iterative loops of **generation, verification, and revision** can effectively break the ceiling of one-shot reasoning. In complex search spaces such as mathematical discovery and engineering optimization, these closed-loop systems can explore and produce results that go beyond existing best-known solutions.

As long-horizon reasoning increasingly integrates with agentic toolchains (e.g., autonomous laboratories), **Self-Evolving Agents** are evolving from passive assistants into active scientific collaborators. Such closed-loop systems with high autonomy not only reshape the division of labor between humans and AI, but may also become a key driver of disruptive breakthroughs in AI for Science.


## References

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

## Citation

> **Citation**: When reposting or citing content from this article, please credit the original author and source.

**Cited as:**

> Yue Shui. (Feb 2026). Self-Evolving Agents.
https://syhya.github.io/posts/2026-02-20-self-evolving-agents

Or


```bibtex
@article{syhya2026-self-evolving-agents,
  title   = "Self-Evolving Agents",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "Feb",
  url     = "https://syhya.github.io/posts/2026-02-20-self-evolving-agents"
}