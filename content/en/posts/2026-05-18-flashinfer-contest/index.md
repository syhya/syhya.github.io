---
title: "GPU Kernel Generation and Optimization with Coding Agents: MLSys 2026 FlashInfer Contest Summary"
date: 2026-05-18T00:00:00+08:00
lastmod: 2026-05-25T00:00:00+08:00
author: "Yue Shui"
categories: ["Technical Blog"]
tags: ["LLM", "GPU Kernel", "CUDA", "Triton", "FlashInfer", "MLSys", "Agent", "Harness Engineering"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

Recently, I participated in the **MLSys 2026 - NVIDIA Track: FlashInfer AI Kernel Generation Contest** ([FlashInfer Contest, 2026a](https://mlsys26.flashinfer.ai/)). This post is not a tutorial on CUDA kernel optimization, and I am not a GPU operator development expert. My main goal was to use a highly verifiable task environment with clear feedback to study how coding agents can continuously produce high-quality GPU kernels in a closed-loop workflow.

The full materials are split into two reports: **Harness Engineering for LLM-Driven GPU Kernel Generation** ([Shui et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf)) and **Full-Agent Kernel Generation for FlashInfer** ([Ma et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf)). The code is available in [mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest).

## Research Background

The difficulty of LLM-generated GPU kernels is not merely writing plausible CUDA or Triton code. A candidate implementation must be semantically correct, compile successfully, cover the target input shapes, and run faster than existing implementations on real GPUs.

**KernelBench** ([Ouyang et al., 2025](https://arxiv.org/abs/2502.10517)) builds an evaluation framework where an LLM reads a PyTorch reference implementation, writes a custom kernel, and is then evaluated for compilation, correctness, and runtime performance.

{{< figure
    src="kernel-bench.png"
    caption="Fig. 1. KernelBench evaluation workflow. A model receives a PyTorch workload, writes a custom kernel implementation, and is evaluated by both functional correctness and latency. (Image source: [Ouyang et al., 2025](https://arxiv.org/abs/2502.10517))"
    align="center"
    width="100%"
>}}

**FlashInfer-Bench** ([Xing et al., 2026](https://arxiv.org/abs/2601.00227)) places this problem in the real workload distributions of LLM inference serving, emphasizing the closed loop among execution traces, evaluation, candidate implementations, and deployment. It is not just a standalone microbenchmark; it requires candidate kernels to be evaluated under realistic inference distributions, a unified trace format, and correctness checks.

{{< figure
    src="flashinfer-bench.png"
    caption="Fig. 2. FlashInfer-Bench architecture. FlashInfer Trace connects kernel definitions, serving workloads, candidate solutions, benchmark results, and deployment paths. (Image source: [Xing et al., 2026](https://arxiv.org/abs/2601.00227))"
    align="center"
    width="100%"
>}}

From a method-taxonomy perspective, this post follows the survey **Towards Automated Kernel Generation in the Era of LLMs** ([Yu et al., 2026](https://arxiv.org/abs/2601.15727)) and summarizes related work into two routes: LLM4Kernel and Agent4Kernel.

**LLM4Kernel** starts from high-quality domain data and uses training techniques such as continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL) to improve the model itself, making it better at understanding kernel-development contexts and generating high-quality kernel code.

{{< figure
    src="LLM4Kernel.png"
    caption="Fig. 3. LLM4Kernel focuses on improving model-side kernel generation capability through data construction, supervised fine-tuning, reinforcement learning, and domain adaptation. (Image source: [Yu et al., 2026](https://arxiv.org/abs/2601.15727))"
    align="center"
    width="90%"
>}}

This route can internalize kernel knowledge into model parameters, but it usually depends on high-quality training data, stable reward design, and substantial training cost.

**Agent4Kernel** instead emphasizes iterative search, external memory, multi-agent orchestration, and automated evaluation. My contest solution is closer to this route: I did not train a new model; instead, I designed a workflow where existing coding agents could try candidates, record feedback, and improve the experimental harness.

{{< figure
    src="Agent4Kernel.png"
    caption="Fig. 4. Agent4Kernel emphasizes iterative refinement, evolutionary search, external memory, hardware profiling, and multi-agent orchestration for kernel optimization. (Image source: [Yu et al., 2026](https://arxiv.org/abs/2601.15727))"
    align="center"
    width="100%"
>}}

This direction also echoes industrial systems such as Meta's **KernelEvolve** ([Liao et al., 2025](https://arxiv.org/abs/2512.23236)). KernelEvolve emphasizes persistent knowledge bases, retrieval-augmented prompt construction, cross-hardware programming abstractions, and continuous optimization for production operators.

{{< figure
    src="KernelEvolve.png"
    caption="Fig. 5. KernelEvolve system overview. Persistent memory, retrieval, evolutionary search, and hardware-aware evaluation are combined to scale agentic kernel coding to production operators. (Image source: [Liao et al., 2025](https://arxiv.org/abs/2512.23236))"
    align="center"
    width="100%"
>}}

The core challenge for GPU-kernel agents is preserving useful experience under complex hardware, complex workloads, and strict validation constraints, then compressing failure feedback into executable search constraints for the next iteration.

## System Architecture

The LLM CUDA team submitted two routes: Agent-Assisted and Full-Agent. Their main difference is not whether they use LLMs, but whether humans continuously intervene in the search process.

| Dimension | Agent-Assisted | Full-Agent |
| --- | --- | --- |
| Human involvement | Humans continuously design strategies, provide reference implementations, select optimization directions, and maintain promotion rules. | Humans provide only the initial task, constraints, and automation tools; the agent performs the subsequent search. |
| Search style | Uses profiling results and experience to choose candidate implementation families, focusing on high-confidence local optimizations. | Runs long-horizon search through a plan-execute-evaluate-summarize-store loop. |
| State management | Mainly relies on human-maintained notes, skills, and experiment archives. | Uses a LoongFlow-style database to record candidate origins, result summaries, current best records, and failure modes. |

### Agent-Assisted

In [Self-Evolving Agents](/posts/2026-02-20-self-evolving-agents/), I discussed the core paradigm of **Harness Engineering** ([Lopopolo, 2026](https://openai.com/index/harness-engineering/); [Rajasekaran, 2026](https://www.anthropic.com/engineering/harness-design-long-running-apps)): humans design constraints, build feedback mechanisms, and define evaluation criteria, while agents iteratively generate higher-quality code inside a controlled environment.

{{< figure
    src="agent-assisted-arch.png"
    caption="Fig. 6. Agent-Assisted closed-loop harness for B200 kernel optimization. The workflow grounds agents in operator definitions, workload distributions, references, profile signals, and explicit promotion policies. (Image source: [Shui et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

The Agent-Assisted harness is organized into four layers:

- **Grounding inputs**: required context, such as operator definitions, reference implementations, and workload JSON files.
- **Shape discovery**: group workloads by parameters such as batch size and sequence length, then sample representative dimensions from each group so that agents can quickly evaluate and iterate on candidates without running full validation every time.
- **Closed-loop optimization**: generate candidates along the *baseline -> profile -> diagnose -> generate -> evaluate -> archive* loop, verify whether the code compiles and whether the results are correct, and evaluate performance; Torch Profiler and NVIDIA Nsight Compute (NCU) are also used to analyze operator bottlenecks.
- **Outputs**: archive related code, performance metrics, and other files so they can be provided back to the agent for later iterations.

Humans wrote optimization [skills](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/skills) and built [evaluation scaffolding](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/scripts). These practices share the same engineering motivation as **Agent Skills** ([OpenAI, 2026a](https://developers.openai.com/codex/skills)) and **Subagents** ([OpenAI, 2026b](https://developers.openai.com/codex/concepts/subagents)): reusable context, packaged tools, and parallel exploration. In this contest setting, they also helped keep the search process inside a verifiable closed loop.

### Full-Agent

{{< figure
    src="full-agent-arch.png"
    caption="Fig. 7. Modified LoongFlow Full-Agent stack. The agent iterates through planning, code generation, evaluation, summarization, and database updates, turning failed candidates into searchable context for later iterations. (Image source: [Ma et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf))"
    align="center"
    width="100%"
>}}

The framework I used for the Full-Agent route follows a **LoongFlow**-like ([Wan et al., 2025](https://arxiv.org/abs/2512.24077)) plan-execute-summarize paradigm and is also similar to **OpenEvolve**-style ([Sharma, 2025](https://github.com/algorithmicsuperintelligence/openevolve)) evolutionary search systems. It decomposes a kernel search into planning, execution, evaluation, summarization, and storage, then writes each candidate's provenance, performance results, and failure summary into a persistent database.

## Experimental Results

On the official Top-3 leaderboard, our team results were:

| Official track | Method | Rank |
| --- | --- | --- |
| Track A Fused MoE | Agent-Assisted | 3rd |
| Track C Gated Delta Net | Agent-Assisted | 3rd |
| Track C Gated Delta Net | Full-Agent | 2nd |

The data below comes from local evaluation on Modal B200 GPUs. It should be treated as reference only, because the development environment did not fully lock GPU clock frequencies and the final evaluation ran on bare-metal machines. The evaluation protocol follows the correctness-gated benchmark setting of **FlashInfer** ([Ye et al., 2025](https://arxiv.org/abs/2501.01005)) and **FlashInfer-Bench** ([Xing et al., 2026](https://arxiv.org/abs/2601.00227)). The table uses the matched comparison convention from the report: latency is mean milliseconds, PyTorch speedup uses the corresponding PyTorch reference mean, and FlashInfer speedup uses the official FlashInfer baseline. In simplified form:

\\[
\mathrm{Speedup} = \frac{\mathrm{mean\ baseline\ latency}}{\mathrm{mean\ solution\ latency}}.
\\]

This table is for local analysis only, not the official per-operator or per-track contest score:

| Operator definition | Method | Mean latency (ms) | Speedup vs. PyTorch reference mean | Speedup vs. FlashInfer baseline |
| --- | --- | ---: | ---: | ---: |
| **DSA Attention** | **Agent-Assisted** | **0.011175** | **217.17×** | **29.68×** |
|  | Full-Agent | 0.022811 | 106.39× | 14.54× |
|  | FlashInfer baseline | 0.331650 | 7.32× | 1.00× |
| **DSA Indexer** | **Agent-Assisted** | **0.006893** | **494.13×** | **18.05×** |
|  | Full-Agent | 0.032659 | 104.29× | 3.81× |
|  | FlashInfer baseline | 0.124420 | 27.38× | 1.00× |
| **GDN Prefill** | **Agent-Assisted** | **0.051992** | **21,078×** | **13.70×** |
|  | Full-Agent | 0.688875 | 1,591× | 1.03× |
|  | FlashInfer baseline | 0.712166 | 1,539× | 1.00× |
| **MoE FP8** | **Agent-Assisted** | **0.286340** | **63.78×** | **1.62×** |
|  | FlashInfer baseline | 0.463874 | 39.37× | 1.00× |
|  | Full-Agent | 1.742630 | 10.48× | 0.27× |
| **GDN Decode** | **Agent-Assisted** | **0.006201** | **7,970×** | **1.12×** |
|  | FlashInfer baseline | 0.006940 | 7,121× | 1.00× |
|  | Full-Agent | 0.008366 | 5,907× | 0.83× |

{{< figure
    src="flashinfer-speedup.png"
    caption="Fig. 8. Final retained speedups over the supplied FlashInfer baseline, measured using mean latency on local Modal B200 runs. (Image source: [Shui et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="65%"
>}}

Agent-Assisted outperforms the FlashInfer baseline on all five operators, with speedups ranging from `1.12×` on GDN Decode to `29.68×` on DSA Attention. Full-Agent also finds effective candidates for DSA Attention, DSA Indexer, and GDN Prefill, but remains below the baseline on MoE FP8 and GDN Decode.

### Agent-Assisted Optimization Trajectory

{{< figure
    src="iteration-trajectory.png"
    caption="Fig. 9. Retained speedup trajectories over the supplied FlashInfer baseline. The curves show long plateaus and discrete jumps rather than smooth monotonic progress. (Image source: [Shui et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

The trajectories show that performance improvements did not happen smoothly. Instead, a small number of large jumps occurred after long plateaus. Effective Agent-Assisted kernel optimization does not simply rely on prompts; it depends on a measurable systems loop: organizing operator constraints, evaluation scaffolding, performance-analysis feedback, and historical trajectories into reusable workflows, then letting agents generate, validate, and retain candidates inside that loop. This process requires humans to continuously design and maintain the harness.

### Full-Agent Optimization Trajectory

{{< figure
    src="full-agent-trajectory.png"
    caption="Fig. 10. Full-Agent optimization trajectories from LoongFlow trace logs. Gray dots are correctness-passing candidates, solid lines are running best, and dashed lines mark the FlashInfer baseline. (Image source: [Ma et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf))"
    align="center"
    width="100%"
>}}

The Full-Agent trajectories come from automatic search logs. Full-Agent can still find effective candidates for some operators, such as DSA Attention at `14.54×`, but it remains slower than Agent-Assisted overall and even falls below the FlashInfer baseline on MoE FP8 and GDN Decode. This gap suggests that fully automated agent search is still difficult. High-quality human-provided reference implementations and continuously accumulated trajectory memory are often more efficient than asking agents to explore from scratch. Future systems need to incorporate controller state and historical memory into the harness while preserving strict final validation.

## Future Work

- **Model-level optimization loop**: Following the direction of **AutoKernel** ([Jaber et al., 2026](https://arxiv.org/abs/2603.21331)), kernel optimization can be extended from single operators to a model-level `profile -> extract -> optimize -> verify` workflow. The system should first use profilers to locate GPU bottlenecks in the model, then extract standalone Triton/CUDA kernels, and use Amdahl's law to decide which kernel should be optimized next.

- **Experiment management and independent verification**: Based on the publicly released contest writeups ([FlashInfer Contest, 2026b](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups)), future harnesses should standardize benchmarks, correctness checks, input-shape scans, numerical stability tests, determinism checks, Roofline Analysis, promotion/rollback decisions, and artifact-structure constraints, while using an independent verifier to review candidate implementations.

- **Workload specialization and retrievable memory**: The common pattern in high-scoring solutions is not blindly trying more kernels, but first understanding the workload distribution and then choosing implementation strategies that fit it. Future systems can structure common workload profiles, reusable optimization templates, successful candidates, and failure reasons, so that agents retrieve similar cases before generating code and know the applicable input shapes, known bottlenecks, and directions that should not be repeated.

## References

[1] FlashInfer Contest. ["FlashInfer AI Kernel Generation Contest."](https://mlsys26.flashinfer.ai/) MLSys 2026 Competition, NVIDIA Track (2026a).

[2] Shui, Yue, Chenyu Ma, Hangfei Xu, Shengzhao Wen, and Yanpeng Wang. ["Harness Engineering for LLM-Driven GPU Kernel Generation."](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf) Technical Report (2026).

[3] Ma, Chenyu, Yue Shui, Hangfei Xu, Shengzhao Wen, and Yanpeng Wang. ["Full-Agent Kernel Generation for FlashInfer @ MLSys 2026."](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf) Technical Report (2026).

[4] Ouyang, Anne, et al. ["KernelBench: Can LLMs Write Efficient GPU Kernels?"](https://arxiv.org/abs/2502.10517) arXiv preprint arXiv:2502.10517 (2025).

[5] Xing, Shanli, et al. ["FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems."](https://arxiv.org/abs/2601.00227) arXiv preprint arXiv:2601.00227 (2026).

[6] Yu, Yang, et al. ["Towards Automated Kernel Generation in the Era of LLMs."](https://arxiv.org/abs/2601.15727) arXiv preprint arXiv:2601.15727 (2026).

[7] Liao, Gang, et al. ["KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta."](https://arxiv.org/abs/2512.23236) arXiv preprint arXiv:2512.23236 (2025).

[8] Lopopolo, Ryan. ["Harness Engineering: Leveraging Codex in an Agent-First World."](https://openai.com/index/harness-engineering/) OpenAI Blog (2026).

[9] Rajasekaran, Prithvi. ["Harness Design for Long-Running Application Development."](https://www.anthropic.com/engineering/harness-design-long-running-apps) Anthropic Engineering Blog (2026).

[10] OpenAI. ["Agent Skills."](https://developers.openai.com/codex/skills) OpenAI Developers (2026a).

[11] OpenAI. ["Subagents."](https://developers.openai.com/codex/concepts/subagents) OpenAI Developers (2026b).

[12] Wan, Chunhui, et al. ["LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm."](https://arxiv.org/abs/2512.24077) arXiv preprint arXiv:2512.24077 (2025).

[13] Sharma, Asankhaya. ["OpenEvolve: Open-source Implementation of AlphaEvolve."](https://github.com/algorithmicsuperintelligence/openevolve) GitHub Repository (2025).

[14] Ye, Zihao, et al. ["FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving."](https://arxiv.org/abs/2501.01005) Proceedings of Machine Learning and Systems (2025).

[15] Jaber, Jaber, and Osama Jaber. ["AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search."](https://arxiv.org/abs/2603.21331) arXiv preprint arXiv:2603.21331 (2026).

[16] FlashInfer Contest. ["MLSys 2026 Contest Writeups."](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups) GitHub Repository (2026b).

## Citation

> **Citation**: Please cite the original author and source when reposting or citing this post.

**Cited as:**

> Yue Shui. (May 2026). GPU Kernel Generation and Optimization with Coding Agents: MLSys 2026 FlashInfer Contest Summary.  
https://syhya.github.io/posts/2026-05-18-flashinfer-contest

Or

```bibtex
@article{syhya2026-mlsys26-flashinfer-contest,
  title   = "GPU Kernel Generation and Optimization with Coding Agents: MLSys 2026 FlashInfer Contest Summary",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "May",
  url     = "https://syhya.github.io/posts/2026-05-18-flashinfer-contest"
}
```
