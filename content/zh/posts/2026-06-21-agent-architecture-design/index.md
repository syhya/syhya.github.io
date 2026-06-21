---
title: "如何选择合适的 Agent 架构？"
date: 2026-06-21T12:00:00+08:00
lastmod: 2026-06-21T20:48:13+08:00
author: "Yue Shui"
categories: ["技术博客"]
tags: ["LLM", "Agent", "Agent Skills", "Subagents", "Multi-Agent", "Dynamic Workflows", "Context Engineering"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

面向长任务场景的上下文管理与多执行单元协同，是当前 Agent 研究中的核心问题之一。在 [大语言模型智能体](/zh/posts/2025-03-27-llm-agent/) 中我介绍过 Agent 的规划、记忆与工具使用三大模块；在 [Self-Evolving Agents](/zh/posts/2026-02-20-self-evolving-agents/) 与 [FlashInfer 比赛总结](/zh/posts/2026-05-18-flashinfer-contest/) 中，我讨论了 Harness Engineering 这一范式：人类负责设计约束、反馈与评估，Agent 在受控闭环里迭代产出可验证的结果。这篇博客聚焦更具体的一层问题：**当任务变长、变复杂应该用什么架构来组织 Agent？** 

## 基础概念

Agent Skills、Subagents、Multi-Agent Systems 与 Dynamic Workflows 经常被放在一起讨论，但它们并不是同一类机制。更准确地说，它们分别对应 Agent 系统中的四个层级：能力封装、执行隔离、协作组织与运行时编排。

| 概念                     | 系统层级  | 设计焦点                                      |
| ---------------------- | ----- | ----------------------------------------- |
| **Agent Skills**        | 能力封装  | 将稳定、可复用的操作流程封装为标准化能力，减少重复规划与执行成本          |
| **Subagents**          | 执行隔离  | 为相对独立的子任务分配独立上下文，由主 agent 委派任务并回收结果       |
| **Multi-Agent Systems** | 协作组织  | 通过角色分工、消息传递与协调机制，组织多个 Agent 共同完成复杂任务      |
| **Dynamic Workflows**   | 运行时编排 | 根据任务状态、中间结果与失败信号，动态调度 Skill、工具与 Subagents |

## Workflow 与 Agent

**Building Effective Agents**（[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)）给出了一个被广泛引用的区分：

- **Workflow（工作流）**：LLM 和工具通过**预先定义好的代码路径**被编排起来。
- **Agent（智能体）**：LLM **动态地自主决定**自己的流程和工具使用，自己掌控如何完成任务。

文章给出的核心建议是：**先找最简单的方案，只在确有必要时才增加复杂度。** 很多场景用单次 LLM 调用加检索和上下文示例就够了，不必动辄上多智能体系统。所有 agentic 系统的基础构件都是一个**增强型 LLM（augmented LLM）**，即具备检索、工具、记忆能力的模型。

{{< figure
    src="augmented-llm.png"
    caption="Fig. 1. The augmented LLM: the basic building block of agentic systems, enhanced with retrieval, tools, and memory. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="90%"
>}}

先理解这些基础 workflow，是因为后面讨论的 Multi-Agent Systems 和 Dynamic Workflows 是建立在它们之上、而非取而代之的。在此之上，下面总结五种常见的可组合工作流模式。

{{< figure
    src="langgraph-workflow-agents.png"
    caption="Fig. 2. Workflows orchestrate LLMs and tools along predefined code paths, while agents dynamically direct their own process and tool use. (Image source: [LangGraph, 2025](https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=workflow))"
    align="center"
    width="100%"
>}}

**1. Prompt chaining（提示链）**：把任务拆成一串顺序步骤，每步 LLM 处理前一步的输出，可在中间加入程序化检查（gate）。适合任务能被干净地拆成固定子步骤的场景。

{{< figure
    src="prompt-chaining.png"
    caption="Fig. 3. The prompt chaining workflow. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="100%"
>}}

**2. Routing（路由）**：先分类输入，再把它导向专门的后续处理。适合输入种类明确、分开处理效果更好的场景，这正是后面 Dynamic Workflows 中 Classify-and-act 模式的雏形。

{{< figure
    src="routing.png"
    caption="Fig. 4. The routing workflow. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="100%"
>}}

**3. Parallelization（并行化）**：把任务并行分发，分为 sectioning（拆成独立子任务并行跑）和 voting（同一任务跑多次以获得多样化输出）两种。适合可并行加速、或需要多次采样提升置信度的场景。

{{< figure
    src="parallelization.png"
    caption="Fig. 5. The parallelization workflow. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="100%"
>}}

**4. Orchestrator-workers（编排者-工作者）**：一个中心 LLM 动态拆解任务、分发给 worker LLM，再综合结果。与 parallelization 的区别在于子任务不是预先定好的，而是由 orchestrator 根据输入动态决定。

{{< figure
    src="orchestrator-workers.png"
    caption="Fig. 6. The orchestrator-workers workflow. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="100%"
>}}

**5. Evaluator-optimizer（评估器-优化器）**：一个 LLM 生成结果，另一个 LLM 评估并给出反馈，循环迭代直到满意。适合有明确评估标准、且迭代能带来明显改进的场景。

{{< figure
    src="evaluator-optimizer.png"
    caption="Fig. 7. The evaluator-optimizer workflow. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="100%"
>}}

当这些预定义路径不再够用、需要模型自己规划和决策时，就进入了真正的 **autonomous agent**：模型在循环中自主使用工具、根据环境反馈推进，直到完成任务或触发停止条件。

{{< figure
    src="autonomous-agent.png"
    caption="Fig. 8. Autonomous agent: the model plans, calls tools, and advances based on environment feedback in a loop. (Image source: [Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents))"
    align="center"
    width="100%"
>}}

可以把这五种模式看作原子级设计模式，而本文后面讨论的 Multi-Agent Systems 和 Dynamic Workflows 则是把这些原子组合、调度起来的机制。

## 上下文窗口

LLM 的有效工作空间受上下文窗口限制。**Context Rot**（[Hong et al., 2025](https://www.trychroma.com/research/context-rot)）对 GPT-4.1、Claude 4 和 Gemini 2.5 等模型的测试表明，模型性能并不会在不同输入长度下保持一致；随着输入变长，模型对上下文的利用会逐渐退化，输出也更容易变得不稳定。下图展示了在 Repeated Words 任务中，多个主流模型的表现随输入长度增加而下降的趋势。

{{< figure
    src="context-rot.png"
    caption="Fig. 9. Context Rot: model accuracy degrades as input length grows across Claude, GPT, Gemini, and Qwen families, showing that LLMs do not maintain consistent performance across input lengths. (Image source: [Hong et al., 2025](https://www.trychroma.com/research/context-rot))"
    align="center"
    width="100%"
>}}

**Context engineering**（[Anthropic, 2025a](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)）与 prompt engineering 的区别在于：后者通常指一次性设计指令的离散任务，而前者是一套在 LLM 推理过程中持续筛选、组织与维护上下文的策略。换言之，context engineering 关注的不是如何写好一个 prompt，而是每一次模型调用时，如何判断应当向模型提供哪些信息、舍弃哪些噪声，并动态更新上下文。其核心目标是在尽可能少的 token 预算内，构建能够最大化期望输出概率的高信号上下文集合。

{{< figure
    src="context-engineering.png"
    caption="Fig. 10. Context engineering treats the context window as a finite resource: system prompts, tools, message history, and retrieved data must all be curated into the smallest high-signal set that fits within the model's attention budget. (Image source: [Anthropic, 2025a](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents))"
    align="center"
    width="100%"
>}}

针对长程任务，有以下三类方案：

- **Compaction（压缩）**：当对话接近上下文上限时，将其内容总结后重启一个新的上下文窗口，保留架构决策与未解决问题，丢弃冗余的工具输出。
- **Structured note-taking（结构化记忆）**：Agent 把笔记持久化到上下文窗口之外（如 `NOTES.md`、待办列表），需要时再读回。
- **Subagent 架构**：让专门的 Subagents 在干净的上下文里处理聚焦任务，再把压缩后的摘要（通常 1,000–2,000 tokens）返回给负责协调的 main agent。

上面这些长程策略为本文接下来要讲的三种机制做了铺垫：Skill 通过按需加载减少重复指令进入上下文，Subagent 通过隔离执行避免主上下文被中间材料污染，而 Dynamic Workflows 则把两者组合成可控的长程执行图。下面先从最轻量的 Skill 开始。

## Agent Skills

**Agent Skill**（[Anthropic, 2025b](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)）是把一套可复用的 instructions、元数据和可选资源（脚本、模板、参考资料）打包成的模块化能力。当 Claude 判断与当前请求相关时，会自动调用它。它和 prompt 的区别在于：prompt 是一次性的对话级指令，而 Skill 是文件系统中按需加载的能力，省去了在多个对话里反复粘贴同一套 prompt。

### SKILL.md 结构

每个 Skill 是文件系统上的一个文件夹，核心是其中的 `SKILL.md` 文件。它由两部分组成：开头的 YAML frontmatter（告诉模型这个 skill 做什么、何时使用），以及其后的 Markdown 指令正文。frontmatter 中只有 `name` 与 `description` 两个**必填字段**，约束如下：

- `name`：最多 64 字符，仅允许小写字母、数字和连字符，不能包含 XML 标签，且不能包含保留词 `anthropic`、`claude`。
- `description`：非空，最多 1024 字符，不能包含 XML 标签；应当同时说明这个 skill 做什么、以及 Claude 应在何时使用它，这正是 Claude 判断是否触发该 skill 的依据。

正文之外，Skill 还可以可选地捆绑额外的 Markdown 指令文件、可执行脚本（scripts）和参考资源（reference materials，如数据库 schema、API 文档、模板）。

```
pdf-skill/
├── SKILL.md          # 必需：YAML frontmatter + Markdown 指令
├── FORMS.md          # 可选：补充指令（如表单填写指南）
├── REFERENCE.md      # 可选：详细 API 参考
└── scripts/
    └── fill_form.py  # 可选：Claude 通过 bash 执行的脚本
```

### 渐进式披露

Skill 设计中最巧妙的一点是 **progressive disclosure（渐进式披露）**：Skill 运行在一个带有文件系统、bash 和代码执行能力的环境里，Claude 像查阅 onboarding guide 的特定章节一样，**分级、按需**地加载内容，而不是一次性把所有东西塞进上下文。 Skill 的三类内容对应到三级加载：

| 层级 | 内容类型 | 加载时机 | 上下文占用 | 加载内容 |
| --- | --- | --- | --- | --- |
| **Level 1 / Metadata** | 指令 | 启动时（always） | 约 100 tokens / skill | YAML frontmatter 中的 `name` 与 `description` |
| **Level 2 / Instructions** | 指令 | skill 被触发时 | < 5k tokens | SKILL.md 正文的工作流与指导 |
| **Level 3+ / Resources** | 指令、代码、资源 | 按需 | 几乎无限 | 通过 bash 读取/执行的捆绑文件，内容不进入上下文 |

这套机制的关键在于：**只有真正用到时，SKILL.md 的内容才会通过 bash 从文件系统读取并进入上下文窗口**。如果指令里引用了 FORMS.md 这类文件，Claude 才会再用 bash 把它读进来；而当指令提到某个脚本时，Claude 通过 bash 运行它、**只接收脚本的输出**（如"校验通过"），脚本代码本身从不进入上下文。

{{< figure
    src="agent-skills.png"
    caption="Fig. 11. Progressive disclosure in Agent Skills: at startup only each Skill's name/description metadata (~100 tokens) sits in the system prompt; the full SKILL.md body loads into context only when the Skill is triggered; bundled files and scripts are read or executed via bash on demand without their contents entering the context window. (Image source: [Anthropic, 2025c](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview))"
    align="center"
    width="100%"
>}}

由此带来三个直接好处：

- **按需文件访问**：一个 Skill 可以包含几十个参考文件，但任务只用到哪个就读哪个，其余始终存储在文件系统里，占用零 token。
- **高效脚本执行**：脚本代码不进上下文，远比让模型现场生成等价代码省 token。
- **捆绑内容几乎没有上限**：因为文件在被访问前不消耗上下文，Skill 可以打包大量 API 文档、数据集或示例而不产生上下文负担。

## Subagents 和 Agent Teams

### Subagents

**Subagents**（[Anthropic, 2026a](https://code.claude.com/docs/en/subagents)）解决的是**上下文隔离**问题。当某个辅助任务会产生大量搜索结果、日志或文件内容，而这些原始材料之后不需要在主对话中直接引用时，就应交给 Subagent 处理，以避免主上下文混乱和 Context Rot。

Subagent 是一个**拥有独立上下文窗口**的执行者：它在自己的 context 中完成探索、阅读和推理，只将压缩后的结论返回给主 Agent。这种机制特别适合代码库探索、多步骤 feature planning 等复杂任务，因为它可以让主 Agent 保持聚焦，同时把大量中间过程隔离在独立上下文中。其核心价值包括：

- **保持上下文聚焦**：将探索和实现细节隔离在主对话之外，保持主上下文聚焦。
- **施加约束**：限制 Subagent 能使用的工具和 Skill。
- **复用配置**：通过用户级 Subagents 在多个项目间复用配置。
- **专精行为**：用聚焦的 system prompt 让 Subagent 专精某个领域。
- **控制成本**：把任务路由到更快或更低成本的模型（如 Sonnet 或 GPT mini 系列）。

Claude Code 给出了以下直观的例子：一个 Subagent 在自己的上下文里读取了 **6,100 tokens** 的文件，但返回给主 Agent 的只有 **420 tokens** 的结果摘要。

{{< figure
    src="subagent-context-windows.png"
    caption="Fig. 12. A subagent works in its own context window: it reads 6,100 tokens of files but returns only a 420-token summary to the main agent, keeping the raw intermediate content out of the main conversation. (Image source: [Anthropic, 2026b](https://code.claude.com/docs/en/context-window))"
    align="center"
    width="90%"
>}}

### Agent Teams

Subagents 只是 Claude Code 并行能力的一种，它们只向主 Agent 汇报、彼此从不通信。当任务需要多个执行者之间相互讨论、协调时，就轮到 **Agent Teams**（[Anthropic, 2026c](https://code.claude.com/docs/en/agent-teams)）：多个拥有独立上下文的会话共享一个任务列表，自行认领工作，并直接互相通信，由一个 lead agent 统筹。两者都拥有独立上下文，但协作方式截然不同：

{{< figure
    src="subagents-agent-teams.png"
    caption="Fig. 13. Subagents only report results back to the main agent and never talk to each other; in agent teams, teammates share a task list, claim work, and communicate directly. (Image source: [Anthropic, 2026c](https://code.claude.com/docs/en/agent-teams))"
    align="center"
    width="100%"
>}}

| | Subagents | Agent Teams |
| --- | --- | --- |
| **上下文** | 独立上下文窗口；结果返回给调用者 | 独立上下文窗口；完全独立 |
| **通信** | 只向主 Agent 汇报结果 | teammate 之间直接互相通信 |
| **协调** | 主 Agent 管理所有工作 | 共享任务列表，自我协调 |
| **适合** | 只关心结果的聚焦任务 | 需要讨论和协作的复杂工作 |
| **Token 成本** | 较低：摘要返回主上下文 | 较高：每个 teammate 都是独立 Claude 实例 |

Subagents 是主 Agent 临时派出的 worker，Agent Teams 更像多个长期会话组成的协作空间。

## Multi-Agent Systems

当一个复杂问题可以被拆成多个相对独立的方向并行探索时，单个 Subagent 就不够了，需要一个调度者来组织一组 Subagents。这就是 **Multi-Agent System**，最经典的形态是 **Orchestrator-Worker（编排者-工作者）** 模式。

Anthropic 在 **How we built our multi-agent research system**（[Anthropic, 2025d](https://www.anthropic.com/engineering/built-multi-agent-research-system)）中详细介绍了其 Research 功能的架构。

{{< figure
    src="multi-agent-architecture.png"
    caption="Fig. 14. High-level architecture of Anthropic's multi-agent research system: user queries flow through a lead agent that creates specialized subagents to search in parallel, then synthesizes results. (Image source: [Anthropic, 2025d](https://www.anthropic.com/engineering/built-multi-agent-research-system))"
    align="center"
    width="100%"
>}}

这是一个典型的 orchestrator-worker 模式的多智能体架构：由一个 lead agent 协调全程并把任务委派给并行运行的专业化 Subagents，最后由 CitationAgent 处理文档和 research report 以定位具体引用位置。完整的执行流程如下图所示：

{{< figure
    src="multi-agent-process.png"
    caption="Fig. 15. The full process of the multi-agent research system: the LeadResearcher plans and spawns subagents, which retrieve in parallel and return results, and finally the CitationAgent aligns citation locations to produce the report. (Image source: [Anthropic, 2025d](https://www.anthropic.com/engineering/built-multi-agent-research-system))"
    align="center"
    width="100%"
>}}

**为什么研究任务特别适合 Multi-Agent？** 因为研究是开放式、路径依赖的，流程难以预先写死：需要边搜索边调整方向。它也天然能拆成多个独立子方向。而搜索的本质是压缩，即从大量网页里提炼关键 token，因此给每个 Subagent 独立的上下文窗口去探索、再返回少量高价值摘要，正好契合问题本身。

以 Claude Opus 4 作为 lead agent、Claude Sonnet 4 作为 Subagents 的多智能体系统，在 Anthropic 内部 research eval 上比单 agent Claude Opus 4 提升 90.2%。但收益伴随显著成本。Anthropic 给出的经验观测是普通 Agent 系统约消耗 chat 的 **4× token**，而 Multi-Agent Systems 约消耗 **15× token**。Multi-Agent 适合高价值、强并行、信息量超过单个上下文窗口的任务，而不是所有任务。

Multi-Agent Systems 的效果很大程度取决于 lead agent 的任务拆解能力和委派提示词（delegation prompt）的设计。lead agent 需要明确告诉每个 Subagent 研究目标是什么、搜索范围是什么、应关注哪些来源、不要重复哪些方向、最终按什么格式返回。如果分工不清，多个 Subagents 可能重复搜索同一批信息，或遗漏关键方向，结果反而推高了汇总成本。

## Dynamic Workflows

Claude Code 默认提供一套通用的 coding harness，让模型在同一上下文中完成计划、执行、检查和总结。对于普通 coding task，这通常已经够用；但当任务变长、并行度提高、结构更复杂，或需要强验证时，单一上下文就容易失控：模型既要记住原始目标，又要维护中间状态、处理大量工具结果，并判断各个分支是否完成。

**Dynamic Workflows**（[Anthropic, 2026d](https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code)）的核心思想是：**让 Claude 在运行时为当前任务动态生成专用 harness，而不是让一个 Agent 在同一上下文里硬撑完整个复杂任务。** 这个 harness 通常是一段 JavaScript 脚本，用于生成、协调和管理多个 Subagents。换言之，Claude 不只是执行任务，还会为任务即时生成一套可执行的工作流。

Dynamic Workflows 要解决的，正是长程任务在单一上下文中反复出现的几类失败模式：

| 失败模式 | 含义 |
| --- | --- |
| **Agentic Laziness（智能体偷懒）** | 面对复杂的多部分任务，只完成一部分就提前宣告完成 |
| **Self-Preferential Bias（自我偏好偏差）** | 在被要求按 rubric 验证或评判时，倾向于相信自己产出的结果 |
| **Goal Drift（目标漂移）** | 多轮执行、尤其是上下文压缩之后，逐渐偏离最初目标 |

这三类问题本质上都和单一上下文有关：计划、执行、验证和修正全放在一起，模型很容易被中间结果影响。Dynamic Workflows 通过独立 Subagents 和显式 workflow control 将这些步骤拆开：每个 Subagent 只关注局部目标，workflow 则负责维护全局状态、并发调度、结果验证和停止条件。因此，它们更像是构建在 Subagents 之上的运行时调度器。

把它和 Subagents、Agent Teams 放在一起看，三者代表了 Claude Code 并行工作的不同取向：

| 方式 | 提供什么 | 何时使用 |
| --- | --- | --- |
| **Subagents** | 一个会话内的委派 worker，在自己的上下文里完成侧支任务并返回摘要 | 侧支任务会产生大量搜索结果、日志或文件内容，且之后不再直接引用 |
| **Agent Teams** | 多个协调的会话，共享任务列表、互相通信，由 lead 管理 | 需要把项目拆块、分配，并让多个 worker 保持同步 |
| **Dynamic Workflows** | 一段脚本编排多个 Subagents，并对结果进行调度和交叉验证 | 任务大到难以逐轮手动协调，或需要明确的验证与停止条件 |

{{< figure
    src="compare-agent-teams-dynamic-workflows.png"
    caption="Fig. 16. Claude Code parallelizes work in several ways: subagents delegate a side task in their own context and return only a summary, agent teams coordinate through a shared task list with direct messaging, and dynamic workflows script and cross-check many subagents for jobs too big to coordinate one turn at a time. (Image source: [Cat Wu on X](https://x.com/_catwu/status/2060054180379689074))"
    align="center"
    width="100%"
>}}

### Workflow Patterns

Anthropic 总结了几种常见的 workflow pattern。Claude 在运行时生成 harness 时，可能会单独使用其中一种，也可能把多种 pattern 组合起来。

{{< figure
    src="six-workflow-patterns.png"
    caption="Fig. 17. Six common dynamic workflow patterns in Claude Code: classify-and-act, fan-out-and-synthesize, adversarial verification, generate-and-filter, tournament, and loop until done. (Image source: [Anthropic, 2026d](https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code))"
    align="center"
    width="100%"
>}}

| Pattern                      | 适用场景              | 核心思想                                                 |
| ---------------------------- | ----------------- | ---------------------------------------------------- |
| **Classify-and-act**         | 任务类型不确定，或输出需要分流处理 | 先用 classifier agent 判断任务类型，再路由到不同 agent 或行为          |
| **Fan-out-and-synthesize**   | 大量可拆分的小任务         | 将任务拆成多个独立步骤并行执行，最后由 synthesize 步骤等待全部完成后合并结构化结果      |
| **Adversarial verification** | 需要高置信度或严格验证       | 为生成结果启动独立 verifier agent，按 rubric 或 criteria 对抗式检查输出 |
| **Generate-and-filter**      | 创意、命名、方案探索        | 先生成大量候选，再按 rubric 或 verification 筛选、去重，只保留高质量结果                  |
| **Tournament**               | 主观排序、方案选择或质量比较    | 让多个 agent 用不同方法完成同一任务，再通过 pairwise judging 逐轮选出胜者    |
| **Loop until done**          | 工作量未知，无法预设轮数      | 持续生成或检查，直到满足停止条件，例如没有新发现或日志中没有新错误                    |

这些 pattern 的价值在于把协调从模型的单一上下文中移出来，交给 workflow 显式控制。以`/deep-research`为例，可以先用 **fan-out-and-synthesize** 让多个 Subagents 分别核查论文、产品文档和博客来源；再用 **adversarial verification** 按引用、日期、数字等 rubric 检查每个结论；最后由 synthesizer 只合并通过验证的内容。这样，Dynamic Workflows 把并行探索、对抗验证和最终汇总变成明确的执行约束。

## Agent Scaling Laws

前面介绍了越来越复杂的组织方式，但一个关键问题始终悬而未决：**是不是 Agent 越多越好？** **Towards a Science of Scaling Agent Systems**（[Kim et al., 2025](https://arxiv.org/abs/2512.08296)）评测了 260 个实验配置，覆盖 6 个 agentic benchmark（BrowseComp-Plus、Finance-Agent、Plancraft、WorkBench、SWE-bench Verified、Terminal-Bench）、5 种架构（单 agent 加 Independent / Centralized / Decentralized / Hybrid 四类多 agent）和 3 个 LLM 家族（OpenAI、Google、Anthropic）。核心结论是：**增加 Agent 数量并非普遍有益，是否有帮助取决于任务结构，尤其是任务的可并行性 vs 顺序性、工具密度和可分解性**。

{{< figure
    src="agent-archs.png"
    caption="Fig. 18. The agent system architectures studied: a single-agent baseline alongside several multi-agent coordination topologies. (Image source: [Kim et al., 2025](https://arxiv.org/abs/2512.08296))"
    align="center"
    width="100%"
>}}

几个关键发现：
- **可并行 → 受益；顺序推理 → 受损**：多 agent 协调在可并行任务上带来增益，但在需要逐步推理的顺序任务上反而损害性能。相对单 agent 基线，性能变化的范围从可分解的金融推理任务上 **+80.8%**（集中式协调），到顺序规划任务上 **−70.0%**（独立式协调），因为通信开销割裂了推理过程。

{{< figure
    src="agent-scaling.png"
    caption="Fig. 19. Multi-agent coordination helps on parallelizable tasks but hurts on sequential ones, and the benefit shrinks as the single-agent baseline grows stronger (capability saturation). (Image source: [Kim et al., 2025](https://arxiv.org/abs/2512.08296))"
    align="center"
    width="100%"
>}}

- **能力饱和效应（capability-saturation effect）**：当单 agent 基线准确率已超过约 **45%** 阈值时，再加 agent 会带来负收益。换句话说，**模型越强，多 agent 的相对价值越低。**
- **架构决定错误传播**：没有集中验证的独立式（Independent）系统会**放大 trace-level（轨迹级）错误达 17.2×**，而集中式（Centralized）orchestrator 通过充当"验证瓶颈"把放大控制在 **4.4×**（去中心式 7.8×、混合式 5.1×、单 agent 基线 1.0×）。不过论文给出了一个重要的补充：在控制协调效率与开销等因素后，这一错误放大效应在统计上并不显著；架构间的性能差异更多源于协调开销，而非纯粹的错误传播。

## 决策建议

把前面所有机制收敛成可操作的决策规则，可以先按五个问题顺序判断：

1. 流程是否稳定、会反复使用？如果是，优先封装为 **Agent Skill**。
2. 子任务是否会产生大量中间信息、污染主上下文？如果是，交给 **Subagent**。
3. 任务是否天然可并行，且信息量超过单个上下文窗口？如果是，考虑 **Multi-Agent System**。
4. 是否需要强验证、自动调度或可重复的控制逻辑？如果是，引入 **Dynamic Workflows**。
5. 如果任务强顺序依赖，或单 Agent 已经足够强，就保持 **Single Agent**，不要为了复杂而复杂。

| 判断 | 推荐机制 | 典型场景 | 为什么 |
| --- | --- | --- | --- |
| 重复流程已标准化 | **Agent Skill** | 固定 checklist、项目规范、固定操作 | 重点是"怎么做"已固定，封装成可复用流程 |
| 独立任务会污染上下文 | **Subagents** | 大量搜索、复杂分析、海量中间信息 | 重点是把任务隔离出去，避免 Context Rot |
| 长期复用的固定角色 | **Custom Subagents** | 固定权限、固定工具、长期复用 | 不只流程固定，"谁来做"也固定 |
| 标准流程 + 独立执行 | **Subagents + Skill** | 专门角色执行标准化流程 | Subagents 负责隔离与权限，Skill 负责标准化流程 |
| 可并行、信息量超单上下文 | **Multi-Agent System** | 开放式研究、多方向并行探索 | Orchestrator 拆分调度，Worker 并行压缩 |
| 任务大到无法手动协调 / 需强验证 | **Dynamic Workflows** | 大规模迁移、事实核查、安全审查、root cause | 运行时动态编排 + 对抗式验证 |
| 强顺序依赖的推理 / 单 agent 已够强 | **Single Agent** | 链式推理、规划、简单 coding | 多 agent 通信开销反而割裂推理 |

在复杂场景下，最强的方案往往是把多种机制组合起来。例如在代码迁移任务中，workflow 为每个模块创建一个 Subagent，每个 Subagent 调用 migration checklist skill 完成局部修改，再由 verifier agent 检查测试和边界条件，最后由 main agent 合并结果。

```
Dynamic Workflows
   ├── Subagent A + Skill X
   ├── Subagent B + Skill Y
   ├── Verifier Agent（对抗式验证）
   └── Synthesizer（汇总）
```

## 小结

Agent 架构设计的关键不是堆更多 Agent，而是让任务结构和系统复杂度匹配。Skill 解决可复用流程，Subagent 解决上下文隔离，Multi-Agent Systems 解决并行分工，Dynamic Workflows 解决运行时编排与验证。更难的功夫在于克制：永远先用任务结构真正需要的最轻量机制，只有当出现具体的失败——上下文污染、并行度失控、验证不足——再上更重的一层。

## 参考文献

[1] Anthropic (Schluntz, Erik, and Barry Zhang). ["Building Effective Agents."](https://www.anthropic.com/engineering/building-effective-agents) Anthropic Engineering Blog (2024).

[2] LangGraph. ["Workflows and Agents."](https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=workflow) LangGraph Tutorial (2025).

[3] Hong, Kelly, Anton Troynikov, and Jeff Huber. ["Context Rot: How Increasing Input Tokens Impacts LLM Performance."](https://www.trychroma.com/research/context-rot) Chroma Technical Report (2025).

[4] Anthropic. ["Effective Context Engineering for AI Agents."](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) Anthropic Engineering Blog (2025a).

[5] Anthropic (Zhang, Barry, Keith Lazuka, and Mahesh Murag). ["Equipping Agents for the Real World with Agent Skills."](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) Anthropic Engineering Blog (2025b).

[6] Anthropic. ["Agent Skills Overview."](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) Claude Platform Documentation (2025c).

[7] Anthropic. ["Subagents."](https://code.claude.com/docs/en/subagents) Claude Code Documentation (2026a).

[8] Anthropic. ["Explore the Context Window."](https://code.claude.com/docs/en/context-window) Claude Code Documentation (2026b).

[9] Anthropic. ["Agent Teams."](https://code.claude.com/docs/en/agent-teams) Claude Code Documentation (2026c).

[10] Anthropic. ["How We Built Our Multi-Agent Research System."](https://www.anthropic.com/engineering/built-multi-agent-research-system) Anthropic Engineering Blog (2025d).

[11] Anthropic. ["A Harness for Every Task: Dynamic Workflows in Claude Code."](https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code) Anthropic Blog (2026d).

[12] Kim, Yubin, et al. ["Towards a Science of Scaling Agent Systems."](https://arxiv.org/abs/2512.08296) arXiv preprint arXiv:2512.08296 (2025).

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (Jun 2026). 如何选择合适的 Agent 架构？  
https://syhya.github.io/zh/posts/2026-06-21-agent-architecture-design

Or

```bibtex
@article{syhya2026-agent-architecture-design,
  title   = "如何选择合适的 Agent 架构？",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "Jun",
  url     = "https://syhya.github.io/zh/posts/2026-06-21-agent-architecture-design"
}
```
