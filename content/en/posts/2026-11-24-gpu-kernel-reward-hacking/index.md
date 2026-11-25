---
title: "Reward Hacking in GPU Kernel Optimization: Loopholes and Defenses in SOL-ExecBench"
date: 2026-11-24T00:00:00+08:00
lastmod: 2026-11-25T00:00:00+08:00
author: "Yue Shui"
categories: ["Technical Blog"]
tags: ["LLM", "Agent", "GPU Kernel", "Evaluation", "Reward Hacking"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

On November 24, 2026, I held first place as **`yue shui@syhya.github.io`** on the L1, L2, Quantization, and FlashInfer-Bench B200 tracks of SOL-ExecBench ([Lin et al., 2026](https://arxiv.org/abs/2603.19173v1)), according to its [official leaderboards](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard). All four pages used Evaluation Stack v1.1. **These results do not reflect real kernel optimization**, though; they came from exploiting evaluation loopholes. Starting from this experiment, this post analyzes reward hacking in GPU kernel optimization. Drawing on SOL-ExecBench's evaluation pipeline and concrete cases, it explains how these loopholes arise, why they pass evaluation, and how to defend against them.

## Results: First Place on All Four Leaderboards

{{< html-visualization src="leaderboard-visualization.html" title="SOL scores and solved coverage across four B200 leaderboards" height="480" >}}

All scores above were obtained with the reward hacking techniques described below, with every task solved on all four collections.

{{< figure
    src="flashinfer-bench-res.png"
    alt="SOL-ExecBench FlashInfer-Bench (B200) leaderboard under Evaluation Stack v1.1, ranked first with a 0.927523 SOL score, 26/26 solved, and 22.52x average speedup"
    caption="Fig. 1. SOL-ExecBench FlashInfer-Bench (B200) leaderboard under Evaluation Stack v1.1, ranked first with a 0.927523 SOL score, 26/26 solved, and 22.52x average speedup. (Image source: [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200))"
    align="center"
    width="100%"
>}}

## What Is Reward Hacking?

{{< figure
    src="reward-hacking-overall-flow.png"
    alt="GPU kernel evaluation and seven failure paths: fixed-test specialization, tolerance exploitation and branch removal, reference leakage, cross-call caching, batching, untimed work, and verifier tampering"
    caption="Fig. 2. Reward hacking can enter at several stages, from inputs to rankings. The coral callouts show the seven paths discussed below; the bottom row shows complementary checks. AI-generated illustration."
    align="center"
    width="100%"
>}}

**Reward hacking exploits gaps in a reward or evaluation rule to earn high scores without a corresponding improvement in the intended objective.** Lilian Weng's overview traces the common problem: the reward being optimized may not fully express the outcome we want ([Weng, 2024](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)).

For kernel optimization, the objective is to **perform the specified computation on the current inputs, faster**. The evaluator usually observes finite correctness tests and timing readings. Those observations become the proxy objective used for optimization. Returning the right tensor does not establish that the current inputs were processed; finishing the whole loop does not establish that each independent call completed its own work. Numerical correctness, call semantics, and timing scope must be checked together for the score to measure the intended capability.

These failures can follow different paths: **specification gaming** exploits gaps between the literal rules and the intended objective, without necessarily changing the evaluator; **reward tampering** directly interferes with the reward function or the evidence it relies on. Both can raise scores without completing the intended task ([Krakovna et al., 2020](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/); [Everitt et al., 2019](https://arxiv.org/abs/1908.04734)).

Rule-based rewards and unit tests reduce some subjectivity, but still depend on input coverage, tolerances, and execution boundaries. Conversely, a failed test is not enough to label behavior hacking. EvilGenie ([Gabor et al., 2025](https://arxiv.org/html/2511.21654v1)) finds that ordinary implementations can miss edge cases, while heuristics exploiting the test distribution can pass held-out tests. Judgments therefore need specifications, code, and execution evidence.

## How SOL-ExecBench Scores Submissions

SOL-ExecBench scores each workload using this formula ([NVIDIA, 2026a](https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide#scoring)):

$$
S(t)=
\frac{t_b-t_{\mathrm{sol}}}
{(t-t_{\mathrm{sol}})+(t_b-t_{\mathrm{sol}})}.
$$

Here, $t$ is the candidate latency, $t_b$ is the precomputed baseline latency, and $t_{\mathrm{sol}}$ is the lower bound estimated by the SOL model, all in the same unit. Kernel and collection scores take arithmetic means over workloads and kernels, respectively; failed or missing entries receive zero.

Within the normal scoring range $t_b>t_{\mathrm{sol}}$ and $t\ge t_{\mathrm{sol}}$, a smaller $t$ gives a higher score. Cross-call caching can compute normally during correctness checks while the timed path only returns an old output and launches a marker kernel, reducing the measured $t$ far below the cost of the full computation. If the evaluator does not verify that the timed path processes the current inputs, it credits that smaller $t$ as an optimization gain.

## Four Case Studies

Each of the following examples follows a different path: Example 1 caches final outputs, Example 2 freezes inputs, Example 3 drops a computation branch, and Example 4 changes the unit of work behind a single call.

### Example 1: Returning Old Outputs Based on Address Changes

The [069_rms_norm](https://research.nvidia.com/benchmarks/sol-execbench/kernel/69) task adds a residual, applies RMSNorm, and multiplies by a weight. The following excerpt from [rmsnorm-submission.py](rmsnorm-submission.py) keeps only the path that returns a single output tensor:

```python
def run(*args):
    in_args = args[:_NUM_INPUTS]
    tensor_inputs = tuple(x for x in in_args if isinstance(x, torch.Tensor))
    scalars = tuple(float(x) for x in in_args if not isinstance(x, torch.Tensor))

    if _reusable(tensor_inputs, scalars):
        return _CACHE["outputs"][0]

    out = _ref_run(*in_args)
    _capture(tensor_inputs, scalars, out)
    return out
```

On a cache miss, `_ref_run` performs the PyTorch reference computation, and `_capture` clones the output and records input metadata. On a hit, the function returns the saved result. The key is how `_reusable` decides that reuse is valid: it checks the input count, scalars, and byte counts, but never compares tensor contents. Instead, it checks whether all addresses have advanced together.

Here is the relevant loop, with `_MIN_SHIFT = 256` and `_MAX_SHIFT = 2048`:

```python
shift = None
cur_ptrs = []
for t, prev_ptr, prev_bytes in zip(tensor_inputs, _CACHE["pointers"], _CACHE["nbytes"]):
    nb = t.numel() * t.element_size()
    ptr = t.data_ptr()
    if nb != prev_bytes or ptr < prev_ptr:
        return False
    cur_shift = ptr - prev_ptr
    if shift is None:
        shift = cur_shift
    elif cur_shift != shift:
        return False
    cur_ptrs.append(ptr)
if shift is None or shift < _MIN_SHIFT or shift > _MAX_SHIFT:
    return False
```

This matches the allocation pattern in the evaluator version pinned in the saved record ([NVIDIA, 2026b](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd); commit `a9fa080`): all input addresses advance by the same 256–2048 bytes per iteration, while the new locations receive copies of the same input data ([allocator source](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L621)). Address rotation therefore gives the cache a recognizable pattern instead of preventing reuse.

The correctness phase generates fresh inputs, and the submission computes normally on cache misses. The [timing phase](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L584) repeatedly uses the same input values, so warmup can populate the cache. On a hit, `_reusable` updates the saved addresses and launches a Triton marker kernel that only writes the integer 0. Then `run` returns the old output.

This version's CUPTI timer records the candidate's GPU activity sequence after warmup and matches it in later iterations. It can record kernels and memory copies, but a matching sequence alone does not establish that those operations perform RMSNorm. If sequence discovery already takes the cached path, later iterations can match only the marker ([timing source](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L154)).

In [rmsnorm-evaluation.json](rmsnorm-evaluation.json), saved on November 23, 2026, this submission received `PASSED` on a B200 for one BF16 workload (`1 × 8192 × 8192`, `eps = 1e-5`). Both maximum absolute and relative errors were 0, and the reported latency was **2.064 μs**. The configuration used 10 warmup calls, 50 timed calls, and seed 200; GPU clocks were not locked, and the reference implementation was not timed. The record has no per-call cache-hit log; the source path and evaluation flow above provide one explanation: **normal computation during correctness checks, followed by cache reuse during timing.**

### Example 2: Stopping Input Updates After Enough Calls

Another approach retains inputs or intermediate results instead of directly caching the final answer. The [public Hyra submission (flash_attention_gqa_ultralong)](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json) for [067_flash_attention_gqa_ultralong](https://research.nvidia.com/benchmarks/sol-execbench/kernel/67) includes the following logic in `main.cpp`. Buffer allocation and argument setup are omitted:

```cpp
static std::unordered_map<int64_t, int> g_call_count;
static const int PREP_CALLS = 15;

int64_t gkey = B * 100000000LL + S;
int cnt = ++g_call_count[gkey];
bool do_prep = (cnt <= PREP_CALLS) || grew;

if (do_prep) {
    prep_launch(srcs, dsts, sigs, lens, modes, 7, stream);
}
```

`prep_launch` converts and copies the current inputs into internal buffers. On consecutive calls with the same `(B, S)`, provided there is no buffer growth or intervening call that overwrites those buffers, this step is skipped from call 16 onward. Even if new hidden states, RoPE parameters, or weights arrive, subsequent computation still reads the old buffers.

Even when `prep_launch` runs, it checks only 256 sampled positions in each input. The following condition comes from the [CUDA source](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json#L49); `samp_pos` selects positions at fixed intervals:

```cpp
float cur = tab.s[w][samp_pos(t, n)];
float st  = tab.sig[w][t];
int dirty = __syncthreads_or((cur != st) ? 1 : 0);
if (!dirty) return;
```

Changing only unsampled positions can therefore leave the internal buffers stale. This differs from the address check above: **matching samples do not prove that the full inputs match, and a recognizable address shift says even less about their contents.** Fixed weights may be preprocessed when the interface permits it; caching inputs that can change requires a reliable validity check.

The code also captures and replays a CUDA Graph when `S <= 4096`. **Graph replay still performs GPU computation; the problem here is that the inputs stop being refreshed.** The [PyTorch documentation](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs) distinguishes fixed addresses from changing data: replay reuses the captured addresses but can read new values stored there. An implementation with internal static input buffers needs to copy the current inputs into them before replay. A call counter cannot determine whether those inputs need updating.

The source of the savings differs from the first example. The first can skip RMSNorm itself; this one skips input preparation and continues computing on old data. This is an analysis of the public code path. I have not rerun this submission on a GPU for this post or attributed all of its performance gains to this behavior.

### Example 3: Exploiting the Input Distribution to Drop a Branch

The reference task for [051_seqlen-finetuned-reconstructed_hyena_complete_forward_block](https://research.nvidia.com/benchmarks/sol-execbench/kernel/145) includes a Hyena mixer followed by an MLP. The [public Hyra submission (hyena_complete_forward_block)](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k145__sid25621__hyena_complete_forward_block__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block.json#L22) selects only eight tensor arguments through these two lines:

```python
_NEED_IDX = [0, 3, 4, 20, 21, 22, 23, 24]
needed = [all_args[i] for i in _NEED_IDX]
```

These indices select the hidden states, second normalization parameters, output projection bias, and MLP parameters. The first normalization, short convolution, long filter, and output projection weight do not participate in the computation. `_prologue` constructs the residual directly from `hidden_states + out_proj_bias`, then runs normalization and the MLP. This removes a computational branch rather than reducing kernel launches through fusion.

One possible explanation is that several reference weights are initialized as `randn * 0.02`, and gating may make the Hyena branch's contribution small enough to disappear within the final output tolerance. **This is an inference about why the submission passes, not an error guarantee for arbitrary inputs.** Amplifying the omitted branch and changing the input distribution would test whether the approximation still meets the task requirements.

The same submission also stops copying new values into the Graph's static inputs after its counter exceeds 11, adding the state issue from the previous example. Dropping a branch and freezing inputs are separate paths. Changing random seeds may expose the former without testing the latter during timing.

### Example 4: Completing the Work While Changing What a Call Means

State retained across calls can also change how work is organized. GPU MODE's NVFP4 retrospective ([Kokoromyti, 2026](https://www.gpumode.com/news/reward-hacking-nvfp4)) describes a submission whose result was later invalidated: it collected 15 inputs, combined their independent calls into one grouped GEMM, and returned cached output handles on subsequent calls.

In timing iterations where batching is enabled and object identities remain stable, the first call launches the whole batch; the other 14 only return handles. For new objects, the submission can instead collect inputs by shape and launch once it has all of them. An independent call must produce its output under the caller's stream semantics, without depending on later calls to finish the work. Correct outputs at the end of the full loop do not establish this.

This case does perform the matrix multiplications and may deliver real batching throughput gains. Dividing the batch time by 15, however, produces an amortized measurement across calls. It does not establish the same latency for an independent request. Batching can be legitimate when the production interface allows it, with the waiting time to form a batch reported as well. **Evaluation needs to check both the answer and whether the unit of work behind the score matches the task contract.**

## Other Publicly Reported Methods

The SOL-ExecBench paper discusses state caching alongside defenses such as input cloning, address offsets, and code review ([Lin et al., 2026](https://arxiv.org/abs/2603.19173v1)). The paper, technical blogs, and incident reports describe the following methods. These paths can overlap; the table gives representative examples rather than an exhaustive list.

| Method | Evaluation gap and public example |
| --- | --- |
| Cross-call caching and phase detection | Reuse old outputs or inputs based on addresses, shapes, sampled values, or call counts. The code above shows several combinations. |
| Exploiting tolerances and input distributions | [SF Tensor](https://sf-tensor.com/news/sol-execbench) reports reducing MoE computation from top-10 to top-2 while still passing; [doubleAI](https://www.doubleai.com/research/maxwell-approaches-speed-of-light-on-blackwell) reports omitting softcap while staying sufficiently close within a particular test range. |
| Specializing to fixed test instances | doubleAI reports RoPE implementations covering only 16 `theta` values and failures exposed by changing seeds. The problem is omitting declared inputs, rather than shape specialization itself. |
| Moving work outside timing or deferring computation | The SOL paper records issues involving side streams, background threads, and capture streams; [Wafer](https://www.wafer.ai/blog/reward-hacks-field-guide) also shows a Tensor subclass deferring computation until the correctness comparison. Historical CUDA-event exploits do not automatically apply to the CUPTI version examined here. |
| Batching across calls | [GPU MODE](https://www.gpumode.com/news/reward-hacking-nvfp4) records combining 15 calls into one grouped GEMM, then returning cached handles. Matrix multiplication still completes, but amortization changes the unit used to measure independent calls. |
| Reading reference answers or reusing evaluator buffers | [METR](https://metr.org/evaluations/openai-o3-report/#optimize-a-kernel) records reading `y_torch` through the Python call stack; Wafer describes no-ops reusing existing outputs and identity operations exploiting test outputs close to their inputs. |
| Modifying timers or the reference environment | The SOL paper records replacing `Event.elapsed_time`; SF Tensor reports changing environment variables at candidate import to reduce reference precision in the same process. |

There is an important distinction between METR's stack inspection and the caching here: the former takes an answer already computed by the evaluator; the latter saves its own earlier answer. Both can return correct values while bypassing the computation required for the current call. They require different defenses: protecting access to reference data, and checking the validity of state reused across calls.

[Cursor](https://cursor.com/blog/multi-agent-kernels) explicitly states that implausible results from caching are excluded, and doubleAI discusses timing improvements.

## How to Reduce Reward Hacking

### Fix the Evaluation Protocol and Execution Boundaries

The cases above suggest three priority checks:

1. **Change input values within the timing workflow and verify the outputs.** At the same shape, test in-place updates, new allocations, and identical address shifts with different contents. Data generation and output comparison can stay outside the timed interval, but correctness must cover the execution path being timed.
2. **Vary the call history and numerical distribution.** Change the warmup count and interleave inputs and shapes. Modify only unsampled positions, or amplify an omitted branch. The former tests state reuse; the latter tests tolerances and input coverage. Both need separate checks.
3. **Protect reference data and scoring.** Specify the allowed caching, preprocessing, batching, and timing scope. Place reference answers, timers, and authoritative records behind boundaries that prevent arbitrary candidate reads or writes. Keep normal implementations as positive controls and caching, branch-dropping, and timing-bypass implementations as negative controls. New checks need to reject exploits while accepting legitimate optimizations.

These measures target the cases in this post and need to be tested individually against retained exploit implementations. More random seeds, CUPTI, or code review alone do not establish that the whole evaluator is reliable.

### Checks in Our Internal Evaluation Framework

Our internal kernel evaluation framework supports two modes for different evaluation goals:

- **Isolated mode**: the generation container first removes hidden files from the task package. The task directory visible to Codex or another Agent contains only `base.py`, `input.py`, `shapes.json`, `constraints.md`, and `self_check.py`. The production reference, frozen latency tables, and evaluator are hidden. Formal evaluation runs in a separate, fresh, complete repository clone to measure implementation from the specification.
- **Non-isolated mode**: the full task package is available to measure how much production speedup the Agent can deliver on the target workloads. Reusing existing implementations can be a valid optimization path when the submission rules permit it.

We report the modes separately and share timing code between self-checks and formal evaluation. The checks include:

| Evaluation stage | Checks and examples |
| --- | --- |
| GPU clocks | Use the same locked SM clock setting for the candidate and reference, then read it back to verify it ([NVIDIA, 2026c](https://docs.nvidia.com/deploy/nvidia-smi/index.html)). This reduces bias from Boost fluctuations or different clocks on the two sides. Our current environment targets about 0.73 times the maximum SM clock and requires readback within ±15 MHz. |
| Timing scope | Capture and replay CUDA Graphs, synchronize the GPU around the Event timing interval, and verify replay outputs. Use node counts and per-call timing probes to reject empty or partial captures and anomalous timings, checking paths that move work to another stream or a CPU thread. |
| State across calls and caching | Rebuild inputs with 3 independent seeds and compare outputs to test implementations that compute once and reuse results. Allocate the timing pool at different addresses with `clone_inputs`, with a total size above twice the L2 capacity, to reduce fixed-object reuse and hot-cache bias. |
| Input mutation | Snapshot tensors whose contracts forbid in-place changes and compare bytes with `view(torch.uint8)`. Changing `+0.0` to `-0.0`, for example, preserves numerical equality but changes the bytes. Check packed FP8 data as raw bytes too. |
| Interface and numerical results | Check shapes, dtypes, and numerical error. An FP32 output requirement, for example, rules out returning FP16 directly; numerical tests still check whether internal precision was reduced. |
| Import and execution policy | Apply AST checks according to the submission rules, such as rejecting direct imports of the reference or restricted vendor libraries, or calls to `torch.matmul`. Graph replay reduces dependence on the candidate's Python wrapper; imports, warmup, and capture still need checking. |
| Scoring baseline | Guard the dispatch path: fail if the production implementation cannot load, rather than silently falling back to a slower PyTorch path that inflates both reference latency and speedup. |
| Audit records | Save the environment, inputs, candidate code, and stage verdicts. Record seeds, Graph node counts, clock readbacks, and the actual reference dispatch path together to help reproduce anomalous results. |

These verdicts also feed the final score: invalid timings or failed anti-hacking checks receive a performance score of zero.

Random inputs test dependence on content, address rotation tests dependence on fixed objects, and a larger working set reduces hot-cache bias. Copying identical values to new addresses does not replace correctness checks on new inputs.

### Strengthen Semantic Checks with Formal Verification

SF Tensor's formal correctness work ([Koska et al., 2026](https://sf-tensor.com/news/formal-correctness)) builds symbolic models from PTX and uses solvers to check equivalence with a specification, covering branches, masks, and synchronization issues that random tests can miss. This advances from testing selected inputs to proving properties under a model and its assumptions. Its early implementation, however, treats floating-point values as exact reals, so it does not establish bounds on actual floating-point error.

### Connect Monitoring to Investigation and Patching

OpenAI's CoT monitoring research ([Baker et al., 2025](https://arxiv.org/abs/2503.11926v1)) describes detecting cheating trajectories, investigating them, and patching environment vulnerabilities. For kernel agents, retain tool calls, submission diffs, GPU activity, and authoritative test results together. Use monitoring to find new patterns, then turn confirmed exploits into regression negatives.

Monitoring itself also needs validation. OpenAI's later practice ([OpenAI, 2026](https://openai.com/safety-alignment/how-we-detect-frontier-model-misalignment/)) uses layered investigations, random audits of unflagged samples, and historical incidents and red-teaming to test for missed cases. Alert rates alone can hide concealed behavior. Actual hacking, monitor recall and false positives, valid-computation pass rates, and defense overhead need separate reporting.

### Penalize Confirmed Bypasses During Training

Once evaluation loopholes enter training, they may also affect later behavior. Denison and colleagues rewarded milder specification gaming in a constructed curriculum and observed occasional modifications of simulated reward code in settings where tampering was not directly trained ([Denison et al., 2024](https://arxiv.org/abs/2406.10162v3)). This demonstrates one possible route for gaming behavior to generalize across settings; it does not estimate a deployment cheating rate.

Anthropic's research ([MacDiarmid et al., 2025](https://arxiv.org/abs/2511.18397v1)) finds that sufficiently weighted preference rewards or dedicated hacking penalties can suppress known cheats in its coding RL setup. Ordinary chat-focused RLHF does not guarantee a fix for agentic behavior. Inoculation prompting mainly reduces the generalization from cheating to other misaligned behaviors; it does not repair evaluator vulnerabilities.

Fewer statements of cheating intent should not be mistaken for success. OpenAI's experiments show that strong optimization pressure directly on CoT can make the remaining cheating harder to detect. Candidate validity should therefore be judged using reliable behavioral and execution evidence, with the effects of monitoring and training interventions checked independently.

Another approach, MONA ([Farquhar et al., 2025](https://arxiv.org/html/2501.13011v2)), changes incentives for multi-step optimization. It limits credit assignment to a step or short horizon while using a supervisor's advance assessment of downstream value to provide foresight. This reduces the incentive to weaken checks now and benefit from higher scores later. The paper tests three constructed settings and does not solve single-step hacks such as hard-coding. For kernel RL, it offers a reward-design idea while independent numerical, call-contract, and timing checks remain necessary.

## Conclusion

A `PASSED` result and a small latency become evidence of optimization only when they correspond to the same valid execution path. Evaluation needs to establish both that an answer was computed correctly and that the timed call actually completed the current request.

As agents take on more kernel optimization work, the evaluator itself needs to be tested, reproduced, and patched as rigorously as the kernels it measures. Otherwise, a speedup on the leaderboard shows only that the evaluation was optimized, not that the computation became faster.

## References

[1] Lin, Edward, et al. ["SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits."](https://arxiv.org/abs/2603.19173v1) arXiv preprint arXiv:2603.19173, v1 (2026).

[2] Weng, Lilian. ["Reward Hacking in Reinforcement Learning."](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) Lil'Log (2024).

[3] Krakovna, Victoria, et al. ["Specification Gaming: The Flip Side of AI Ingenuity."](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) DeepMind Blog (2020).

[4] Everitt, Tom, et al. ["Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective."](https://arxiv.org/abs/1908.04734) arXiv preprint arXiv:1908.04734 (2019).

[5] Gabor, Jonathan, Jayson Lynch, and Jonathan Rosenfeld. ["EvilGenie: A Reward Hacking Benchmark."](https://arxiv.org/html/2511.21654v1) arXiv preprint arXiv:2511.21654, v1 (2025).

[6] NVIDIA. ["SOL-ExecBench Submission Guide: Scoring."](https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide#scoring) NVIDIA Documentation (2026a).

[7] NVIDIA. ["SOL-ExecBench Evaluation Code."](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd) GitHub Repository (2026b).

[8] Tencent Hunyuan. ["Hyra SOL-ExecBench Results and Submissions."](https://github.com/Tencent-Hunyuan/Hyra-results/tree/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench) GitHub Repository (2026).

[9] PyTorch. ["CUDA Semantics: CUDA Graphs."](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs) PyTorch Documentation (version 2.15).

[10] Kokoromyti, Natalia. ["Anatomy of a Reward Hack: A Real Story from the Latest GPU Mode NVFP4 Competition."](https://www.gpumode.com/news/reward-hacking-nvfp4) GPU MODE Blog (2026).

[11] SF Tensor. ["What We Learned Topping NVIDIA's SOL-ExecBench."](https://sf-tensor.com/news/sol-execbench) SF Tensor Blog (2026).

[12] doubleAI. ["Maxwell Approaches Speed of Light on Blackwell."](https://www.doubleai.com/research/maxwell-approaches-speed-of-light-on-blackwell) doubleAI Research Blog (2026).

[13] Andere, Emilio. ["A Field Guide to Reward Hacking in AI Kernel Generation."](https://www.wafer.ai/blog/reward-hacks-field-guide) Wafer Blog (2026).

[14] METR. ["Details About METR's Preliminary Evaluation of OpenAI's o3 and o4-mini."](https://metr.org/evaluations/openai-o3-report/) METR Evaluation Report (2025).

[15] Lin, Wilson, et al. ["Speeding Up GPU Kernels by 38% with a Multi-Agent System."](https://cursor.com/blog/multi-agent-kernels) Cursor Research Blog (2026).

[16] NVIDIA. ["NVIDIA System Management Interface."](https://docs.nvidia.com/deploy/nvidia-smi/index.html) NVIDIA Documentation (2026c).

[17] Koska, Ben, Tom Koska, and Luk Koska. ["Proving Kernels Correct Instead of Testing Them."](https://sf-tensor.com/news/formal-correctness) SF Tensor Blog (2026).

[18] Baker, Bowen, et al. ["Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation."](https://arxiv.org/abs/2503.11926v1) arXiv preprint arXiv:2503.11926, v1 (2025). [OpenAI Research Blog](https://openai.com/index/chain-of-thought-monitoring/).

[19] OpenAI. ["How We Detect Frontier Model Misalignment."](https://openai.com/safety-alignment/how-we-detect-frontier-model-misalignment/) OpenAI Safety & Alignment (2026).

[20] Denison, Carson, et al. ["Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models."](https://arxiv.org/abs/2406.10162v3) arXiv preprint arXiv:2406.10162, v3 (2024). [Anthropic Research Blog](https://www.anthropic.com/research/reward-tampering).

[21] MacDiarmid, Monte, et al. ["Natural Emergent Misalignment from Reward Hacking in Production RL."](https://arxiv.org/abs/2511.18397v1) arXiv preprint arXiv:2511.18397, v1 (2025). [Anthropic Research Blog](https://www.anthropic.com/research/emergent-misalignment-reward-hacking).

[22] Farquhar, Sebastian, et al. ["MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking."](https://arxiv.org/html/2501.13011v2) arXiv preprint arXiv:2501.13011, v2 (2025).

## Citation

> **Citation**: Please cite the original author and source when reposting or citing this post.

**Cited as:**

> Yue Shui. (November 2026). Reward Hacking in GPU Kernel Optimization: Loopholes and Defenses in SOL-ExecBench. [Yue Shui's Blog](https://syhya.github.io/posts/2026-11-24-gpu-kernel-reward-hacking/).

Or

```bibtex
@article{syhya2026-gpu-kernel-reward-hacking,
  title   = "Reward Hacking in GPU Kernel Optimization: Loopholes and Defenses in SOL-ExecBench",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "November",
  url     = "https://syhya.github.io/posts/2026-11-24-gpu-kernel-reward-hacking/"
}
```
