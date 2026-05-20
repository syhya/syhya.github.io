---
title: "代码智能体生成 GPU Kernel"
date: 2026-05-18T00:00:00+08:00
lastmod: 2026-05-21T00:00:00+08:00
author: "Yue Shui"
categories: ["技术博客"]
tags: ["LLM", "GPU Kernel", "CUDA", "Triton", "FlashInfer", "MLSys", "Agent"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

> **注意**: 本文正在更新中，请随时关注最新版本。

我最近参加了 **FlashInfer AI Kernel Generation Contest**（[FlashInfer Contest, 2026](https://mlsys26.flashinfer.ai/)）。这篇博客并不是一篇关于 CUDA kernel 优化技巧的教程，我本身并不是 GPU 算子开发专家；参加这次比赛的主要目的，是想借助一个高度可验证、反馈明确的任务环境，研究 **如何让 Coding Agent 在持续闭环中产出高质量的 GPU kernel**。完整的技术报告参见 **Harness Engineering for LLM-Driven GPU Kernel Generation**（[Shui et al., 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf)），公开仓库在 [mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest)。

## Harness Engineering

[Self-Evolving Agents](/zh/posts/2026-02-20-self-evolving-agents/) 中提到 **Harness Engineering**（[OpenAI, 2026](https://openai.com/index/harness-engineering/)；[Anthropic, 2026](https://www.anthropic.com/engineering/harness-design-long-running-apps)）的核心范式：人类负责设计约束、构建反馈机制并定义评估标准，Agent 在受控环境内迭代生成更高质量的代码。

{{< figure
    src="closed-loop.png"
    caption="Fig. 1. Closed-loop harness/controller workflow. Harness compiles, validates, benchmarks, profiles, and archives candidates; controller converts workload/profile/history into the next optimization hypothesis. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

整个 harness 由四层组成：

- **Grounding inputs**：算子定义、参考实现和 workload JSON 等必要上下文输入。
- **Shape discovery**：从 workload 中根据 batch size 和 sequence length 等参数进行分组，并从每组抽取代表性维度，使 agent 可以快速评估和迭代候选，而不需要每次都做全量验证。
- **Closed-loop optimization**：在 *baseline → profile → diagnose → generate → evaluate → archive* 这条循环里生成候选，验证代码能否编译、结果是否正确，并评估性能；同时利用 Torch Profiler 和 NVIDIA Nsight Compute (NCU) 等工具分析算子瓶颈。
- **Outputs**：归档相关代码和性能指标等文件，后续提供给 agent 继续迭代。

人类编写优化 [skills](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/skills)、构建[评测脚手架](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/scripts)。这些做法类似 **Codex Skills**（[OpenAI, 2026](https://developers.openai.com/codex/skills)）和 **Codex Subagents**（[OpenAI, 2026](https://developers.openai.com/codex/concepts/subagents)）的思路：通过复用上下文、工具和并行探索来提升 agent 的搜索效率，避免陷入局部最优。

## 实验结果

下面的数据来自 Modal 平台 B200 GPU 上的本地评测，并不是公开 leaderboard 结果；评测方法参考 **FlashInfer**（[Ye et al., 2025](https://arxiv.org/abs/2501.01005)）和 **FlashInfer-Bench**（[Xing et al., 2026](https://arxiv.org/abs/2601.00227)）的 correctness-gated kernel benchmark 设定。下表为了便于阅读使用平均延迟之比作为汇总口径，并不是官方逐算子或逐赛道 contest score：

| 算子定义 | 方法 | 平均延迟 (ms) | 相对 PyTorch 加速 | 相对 FlashInfer 加速 |
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

在这些本地评估的结果中，Agent-Assisted 在五个算子上全部优于 FlashInfer baseline，加速比从 `1.12×`（GDN Decode）到 `29.68×`（DSA Attention）；Full-Agent 则在 MoE FP8 与 GDN Decode 上反而比 baseline 更慢。


### Agent-Assisted 优化轨迹

{{< figure
    src="iteration_trajectory.png"
    caption="Fig. 3. Retained speedup trajectories over the supplied FlashInfer baseline. The curves show long plateaus and discrete jumps rather than smooth monotonic progress. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

从轨迹可以看到，性能提升并不是平滑发生的，而是长期平台期之后出现少数几次明显跃迁。有效的 Agent-Assisted kernel 优化不是单纯依赖提示词，而是依赖可测量的系统闭环：把算子约束、评测脚手架、性能分析反馈和历史轨迹组织成可复用流程，再让智能体在其中生成、验证和保留候选。这个过程需要人类持续设计和维护 harness。

### Full-Agent 优化轨迹

{{< figure
    src="full_agent_trajectory.png"
    caption="Fig. 4. Full-Agent optimization trajectories from LoongFlow trace logs. Gray dots are correctness-passing candidates, solid lines are running best, and dashed lines mark the FlashInfer baseline. (Image source: [Shui, 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf))"
    align="center"
    width="100%"
>}}

Full-Agent 轨迹来自 **LoongFlow**（[Wan et al., 2025](https://arxiv.org/abs/2512.24077)）的自动搜索日志。它也能在部分算子上找到有效候选，例如 DSA Attention 达到 `14.54×`，但整体仍慢于 Agent-Assisted，并且 MoE FP8 和 GDN Decode 甚至低于 FlashInfer baseline。这个差距说明，完全自动化的智能体搜索仍然困难。人类提供的高质量参考实现和持续积累的轨迹记忆，往往比让智能体从零开始探索更有效率。未来系统需要把控制状态和历史记忆纳入 harness，同时保持严格的最终验证。

## 参考文献

[1] FlashInfer Contest. ["FlashInfer AI Kernel Generation Contest."](https://mlsys26.flashinfer.ai/) MLSys 2026 Competition, NVIDIA Track (2026).

[2] Shui, Yue, Chenyu Ma, Hangfei Xu, Shengzhao Wen, and Yanpeng Wang. ["Harness Engineering for LLM-Driven GPU Kernel Generation."](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf) Technical Report (2026).

[3] OpenAI. ["Harness Engineering: Leveraging Codex in an Agent-First World."](https://openai.com/index/harness-engineering/) OpenAI Blog (2026).

[4] Anthropic. ["Harness Design for Long-Running Application Development."](https://www.anthropic.com/engineering/harness-design-long-running-apps) Anthropic Engineering Blog (2026).

[5] OpenAI. ["Codex Skills."](https://developers.openai.com/codex/skills) OpenAI Developers (2026).

[6] OpenAI. ["Codex Subagents."](https://developers.openai.com/codex/concepts/subagents) OpenAI Developers (2026).

[7] Ye, Zihao, et al. ["FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving."](https://arxiv.org/abs/2501.01005) Proceedings of Machine Learning and Systems (2025).

[8] Xing, Shanli, et al. ["FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems."](https://arxiv.org/abs/2601.00227) arXiv preprint arXiv:2601.00227 (2026).

[9] Wan, Chunhui, et al. ["LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm."](https://arxiv.org/abs/2512.24077) arXiv preprint arXiv:2512.24077 (2025).


## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (May 2026). 代码智能体生成 GPU Kernel.  
https://syhya.github.io/zh/posts/2026-05-18-flashinfer-contest

Or

```bibtex
@article{syhya2026-mlsys26-flashinfer-contest,
  title   = "代码智能体生成 GPU Kernel",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "May",
  url     = "https://syhya.github.io/zh/posts/2026-05-18-flashinfer-contest"
}
```
