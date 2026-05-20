---
title: "Coding Agents for GPU Kernel Generation"
date: 2026-05-18T00:00:00+08:00
lastmod: 2026-05-21T00:00:00+08:00
author: "Yue Shui"
categories: ["Technical Blog"]
tags: ["LLM", "GPU Kernel", "CUDA", "Triton", "FlashInfer", "MLSys", "Agent"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

> **Note**: This article is being updated. Please check back for the latest version.

Recently, I participated in the **FlashInfer AI Kernel Generation Contest** ([FlashInfer Contest, 2026](https://mlsys26.flashinfer.ai/)). This blog post is not a tutorial on CUDA kernel optimization, and I am not a GPU operator development expert. My main purpose in joining the contest was to use a highly verifiable task environment with clear feedback to study **how coding agents can continuously produce high-quality GPU kernels in a closed-loop workflow**. The full technical report is **Harness Engineering for LLM-Driven GPU Kernel Generation** ([Shui et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf)), and the public repository is [mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest).

## Harness Engineering

In [Self-Evolving Agents](/posts/2026-02-20-self-evolving-agents/), I discussed the core paradigm of **Harness Engineering** ([OpenAI, 2026](https://openai.com/index/harness-engineering/); [Anthropic, 2026](https://www.anthropic.com/engineering/harness-design-long-running-apps)): humans design constraints, build feedback mechanisms, and define evaluation criteria, while agents iteratively generate higher-quality code inside a controlled environment.

{{< figure
    src="closed-loop.png"
    caption="Fig. 1. Closed-loop harness/controller workflow. Harness compiles, validates, benchmarks, profiles, and archives candidates; controller converts workload/profile/history into the next optimization hypothesis. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

The harness consists of four layers:

- **Grounding inputs**: required context, such as operator definitions, reference implementations, and workload JSON files.
- **Shape discovery**: group workloads by parameters such as batch size and sequence length, then sample representative dimensions from each group so that agents can quickly evaluate and iterate on candidates without running full validation every time.
- **Closed-loop optimization**: generate candidates along the *baseline -> profile -> diagnose -> generate -> evaluate -> archive* loop, verify whether the code compiles and whether the results are correct, and evaluate performance; Torch Profiler and NVIDIA Nsight Compute (NCU) are also used to analyze operator bottlenecks.
- **Outputs**: archive related code, performance metrics, and other files so they can be provided back to the agent for later iterations.

Humans wrote optimization [skills](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/skills) and built [evaluation scaffolding](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/scripts). These practices reflect similar ideas behind **Codex Skills** ([OpenAI, 2026](https://developers.openai.com/codex/skills)) and **Codex Subagents** ([OpenAI, 2026](https://developers.openai.com/codex/concepts/subagents)): using reusable context, tools, and parallel exploration to improve agent search efficiency and avoid local optima.

## Experimental Results

The data below comes from local evaluation on Modal B200 GPUs and is not a public leaderboard result. The evaluation protocol follows the correctness-gated kernel benchmark setting of **FlashInfer** ([Ye et al., 2025](https://arxiv.org/abs/2501.01005)) and **FlashInfer-Bench** ([Xing et al., 2026](https://arxiv.org/abs/2601.00227)). The table below uses a ratio-of-means latency summary for readability, rather than the official per-kernel/per-track contest score:

| Operator definition | Method | Mean latency (ms) | Speedup vs. PyTorch | Speedup vs. FlashInfer |
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
    src="flashinfer_speedup.png"
    caption="Fig. 2. Final retained speedups over the supplied FlashInfer baseline, measured using mean latency. Agent-Assisted artifacts improve all five definitions under the reporting normalization. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="60%"
>}}

In these local retained measurements, Agent-Assisted outperforms the FlashInfer baseline on all five operators, with speedups ranging from `1.12×` on GDN Decode to `29.68×` on DSA Attention. Full-Agent, by contrast, is slower than the baseline on MoE FP8 and GDN Decode.

### Agent-Assisted Optimization Trajectory

{{< figure
    src="iteration_trajectory.png"
    caption="Fig. 3. Retained speedup trajectories over the supplied FlashInfer baseline. The curves show long plateaus and discrete jumps rather than smooth monotonic progress. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

The trajectories show that performance improvements did not happen smoothly. Instead, a small number of large jumps occurred after long plateaus. Effective Agent-Assisted kernel optimization does not simply rely on prompts; it depends on a measurable systems loop: organizing operator constraints, evaluation scaffolding, performance-analysis feedback, and historical trajectories into reusable workflows, then letting agents generate, validate, and retain candidates inside that loop. This process requires humans to continuously design and maintain the harness.

### Full-Agent Optimization Trajectory

{{< figure
    src="full_agent_trajectory.png"
    caption="Fig. 4. Full-Agent optimization trajectories from LoongFlow trace logs. Gray dots are correctness-passing candidates, solid lines are running best, and dashed lines mark the FlashInfer baseline. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

The Full-Agent trajectories come from the automatic search logs of **LoongFlow** ([Wan et al., 2025](https://arxiv.org/abs/2512.24077)). LoongFlow can still find effective candidates for some operators; for example, the selected Full-Agent DSA Attention artifact evaluates at `14.54×` under the matched final-evaluation protocol. Overall, however, it remains slower than Agent-Assisted, and even falls below the FlashInfer baseline on MoE FP8 and GDN Decode. This gap suggests that fully automated agent search is still difficult. High-quality human-provided reference implementations and continuously accumulated trajectory memory are often more efficient than asking agents to explore from scratch. Future systems need to incorporate controller state and historical memory into the harness while preserving strict final validation.

## References

[1] FlashInfer Contest. ["FlashInfer AI Kernel Generation Contest."](https://mlsys26.flashinfer.ai/) MLSys 2026 Competition, NVIDIA Track (2026).

[2] Shui, Yue, Chenyu Ma, Hangfei Xu, Shengzhao Wen, and Yanpeng Wang. ["Harness Engineering for LLM-Driven GPU Kernel Generation."](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf) Technical Report (2026).

[3] OpenAI. ["Harness Engineering: Leveraging Codex in an Agent-First World."](https://openai.com/index/harness-engineering/) OpenAI Blog (2026).

[4] Anthropic. ["Harness Design for Long-Running Application Development."](https://www.anthropic.com/engineering/harness-design-long-running-apps) Anthropic Engineering Blog (2026).

[5] OpenAI. ["Codex Skills."](https://developers.openai.com/codex/skills) OpenAI Developers (2026).

[6] OpenAI. ["Codex Subagents."](https://developers.openai.com/codex/concepts/subagents) OpenAI Developers (2026).

[7] Ye, Zihao, et al. ["FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving."](https://arxiv.org/abs/2501.01005) Proceedings of Machine Learning and Systems (2025).

[8] Xing, Shanli, et al. ["FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems."](https://arxiv.org/abs/2601.00227) arXiv preprint arXiv:2601.00227 (2026).

[9] Wan, Chunhui, et al. ["LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm."](https://arxiv.org/abs/2512.24077) arXiv preprint arXiv:2512.24077 (2025).

## Citation

> **Citation**: Please cite the original author and source when reposting or citing this post.

**Cited as:**

> Yue Shui. (May 2026). Coding Agents for GPU Kernel Generation.  
https://syhya.github.io/posts/2026-05-18-flashinfer-contest

Or

```bibtex
@article{syhya2026-mlsys26-flashinfer-contest,
  title   = "Coding Agents for GPU Kernel Generation",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "May",
  url     = "https://syhya.github.io/posts/2026-05-18-flashinfer-contest"
}
```
