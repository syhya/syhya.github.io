---
title: "Reward Hacking in GPU Kernel Optimization: Loopholes and Defenses in SOL-ExecBench"
date: 2026-11-24T00:00:00+08:00
lastmod: 2026-11-26T00:00:00+08:00
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

On November 24, 2026, I held first place across all four B200 tracks of SOL-ExecBench ([Lin et al., 2026](https://arxiv.org/abs/2603.19173v1))—L1, L2, Quantization, and FlashInfer-Bench—on its [official leaderboards](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard), all evaluated under Evaluation Stack v1.1. However, **these results do not reflect genuine kernel optimization**; they were achieved by exploiting loopholes in the evaluation harness. Using this experiment as a starting point, this post examines reward hacking in GPU kernel optimization: walking through SOL-ExecBench's evaluation pipeline and concrete cases to explain how these loopholes arise, why they slip past verification, and how to defend against them.

## Results: First Place on All Four Leaderboards

{{< html-visualization src="leaderboard-visualization.html" title="SOL scores and solved coverage across four B200 leaderboards" height="480" >}}

All of the scores above were obtained using the reward-hacking techniques described below, with every problem solved across all four collections.

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
    alt="GPU kernel evaluation pipeline and seven exploit paths: fixed-test overfitting, tolerance or branch omission, reference-answer leakage, stale-input or output caching, cross-call batching, off-timer or lazy execution, and verifier or timer tampering"
    caption="Fig. 2. Reward hacking can enter at several stages, from inputs to rankings. The coral callouts show the seven paths discussed below; the bottom row shows complementary checks. AI-generated illustration."
    align="center"
    width="100%"
>}}

**Reward hacking occurs when an agent exploits loopholes in a reward function or evaluation rule to achieve high scores without making corresponding progress on the designer's intended objective.** Lilian Weng's survey traces this recurring failure mode: the proxy reward being optimized rarely captures the exact outcome we actually want ([Weng, 2024](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)).

In GPU kernel optimization, the true objective is to **perform the specified computation on the current inputs, faster**. An evaluator, however, typically observes only a finite suite of correctness tests and a set of timing measurements, which together become the proxy objective being optimized. Returning the right tensor does not prove that the current inputs were processed; finishing an entire benchmark loop does not prove that each independent invocation completed its own work. Numerical correctness, call semantics, and timing scope must all be verified together—only when all three hold is the resulting score trustworthy.

This divergence can take two distinct forms: **specification gaming** satisfies the literal rules of an objective without achieving the intended outcome, without necessarily modifying the evaluator; **reward tampering** directly interferes with the reward function itself or the evidence it relies on. Either path can inflate scores without completing the intended task ([Krakovna et al., 2020](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/); [Everitt et al., 2019](https://arxiv.org/abs/1908.04734)).

Rule-based rewards and unit tests reduce subjectivity, yet they remain bounded by input coverage, numerical tolerances, and execution boundaries. Conversely, a failed test does not automatically indicate hacking: EvilGenie ([Gabor et al., 2025](https://arxiv.org/html/2511.21654v1)) finds that otherwise legitimate implementations can miss edge cases on held-out tests, while heuristic solutions tailored to the test distribution can pass them. Judging whether a submission is a genuine bug or a reward hack therefore requires inspecting the specification, code, and execution traces together.

## How SOL-ExecBench Scores Submissions

SOL-ExecBench scores each workload using the following formula ([NVIDIA, 2026a](https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide#scoring)):

$$
S(T_k)=\frac{T_b-T_{\mathrm{SOL}}}{(T_k-T_{\mathrm{SOL}})+(T_b-T_{\mathrm{SOL}})}.
$$

Here, \(T_k\) is the measured runtime of the candidate kernel, \(T_b\) is the runtime of the scoring baseline, and \(T_{\mathrm{SOL}}\) is the hardware Speed-of-Light (SOL) lower bound estimated by SOLAR. The scoring baseline is an optimized implementation derived from the PyTorch reference program. Under the assumptions \(T_b > T_{\mathrm{SOL}}\) and \(T_k \geq T_{\mathrm{SOL}}\), the score is bounded above by 1 and satisfies three anchor properties:

- Matching the scoring baseline (\(T_k = T_b\)) gives \(S = 0.5\).
- Reaching the SOL bound (\(T_k = T_{\mathrm{SOL}}\)) gives \(S = 1\).
- As runtime grows without bound (\(T_k \to \infty\)), \(S \to 0\).

Fig. 3 plots the SOL score against kernel runtime. Because the curve is nonlinear, an identical runtime reduction yields a much larger score gain closer to the SOL bound.

{{< figure
    src="sol-execbench-score-vs-runtime.png"
    alt="SOL score curve as a function of kernel runtime. With an SOL bound of 50 the score is 1, at a scoring baseline of 100 the score is 0.5, and as runtime grows toward infinity the score decays toward 0"
    caption="Fig. 3. SOL score as a function of kernel runtime with an SOL bound of 50 and a scoring baseline of 100. (Image source: [Lin et al., 2026](https://arxiv.org/html/2603.19173v1#S4.F5))"
    align="center"
    width="100%"
>}}

When timing misses part of the computation required for a call, the measured runtime \(T_k\) can fall below the estimated hardware bound \(T_{\mathrm{SOL}}\). [Cursor](https://cursor.com/blog/multi-agent-kernels) notes in its blog post that if reward-hacking tactics such as improper caching push measured performance beyond what a B200 can physically support, the evaluation pipeline should invalidate the result.

Take [001_attention_softmax_dropout_value_matmul_backward](https://research.nvidia.com/benchmarks/sol-execbench/kernel/1) in Fig. 4 as an example. On the B200 leaderboard, the submission's reported latency is **0.002197 ms**, far below the problem's **0.081720 ms** SOL bound. The online evaluation service clamps any SOL score above 1 to **1**.

{{< figure
    src="sol-execbench-kernel-001-b200-leaderboard.png"
    alt="B200 leaderboard screenshot for 001_attention_softmax_dropout_value_matmul_backward showing a submitted latency of 0.002197 ms, an SOL bound of 0.081720 ms, and a displayed SOL score of 1"
    caption="Fig. 4. B200 leaderboard for 001_attention_softmax_dropout_value_matmul_backward. The submission’s reported runtime is below the SOL bound, and the leaderboard shows a score of 1."
    align="center"
    width="100%"
>}}

## Four Case Studies

The four examples below each follow a distinct exploit path: Example 1 caches the final output, Example 2 freezes the inputs, Example 3 drops an entire computation branch, and Example 4 changes the unit of work behind a single call.

### Example 1: Returning Cached Outputs by Tracking Address Shifts

The [069_rms_norm](https://research.nvidia.com/benchmarks/sol-execbench/kernel/69) task adds a residual, applies RMSNorm, and scales by a weight tensor. The following excerpt from [rmsnorm-submission.py](rmsnorm-submission.py) keeps only the single-output path that returns a tensor directly:

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

On a cache miss, `_ref_run` executes the PyTorch reference implementation, and `_capture` clones the output while recording metadata about the inputs. On a hit, `run` returns the saved tensor directly. The trick lies in how `_reusable` decides whether the cache is valid: it checks the input count, scalar values, and tensor byte sizes, yet never inspects tensor contents. Instead, it checks whether every input pointer shifted forward by the same offset.

Here is that check, with `_MIN_SHIFT = 256` and `_MAX_SHIFT = 2048`:

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

This matches the allocation scheme in the evaluator revision pinned in the saved trace ([NVIDIA, 2026b](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd); commit `a9fa080`): on each iteration, all input pointers advance together by 256–2048 bytes, while the new addresses receive copies of the exact same input data ([allocator source](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L621)). Rather than defeating caching, predictable address rotation hands the cache a clean signal to key on.

During correctness verification, fresh inputs are generated and cache misses fall through to the reference computation. During the [timing phase](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L584), the harness reuses the same input values across iterations, allowing warmup runs to prime the cache. On every subsequent hit, `_reusable` updates the stored pointers, launches a minimal Triton marker kernel that writes the integer `0`, and lets `run` return the cached output.

This revision's CUPTI timer records the candidate's GPU activity sequence after warmup and matches that sequence across timed iterations. While CUPTI captures kernels and memory copies, a matching sequence alone cannot verify that those operations actually perform RMSNorm. Once sequence discovery hits the cached branch, later iterations need only match the marker kernel ([timing source](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L154)).

In [rmsnorm-evaluation.json](rmsnorm-evaluation.json), saved on November 23, 2026, this submission received `PASSED` on an NVIDIA B200 for a BF16 workload (`1 × 8192 × 8192`, `eps = 1e-5`) with zero maximum absolute and relative error and a reported latency of **2.064 μs**. The run used 10 warmup iterations, 50 timed iterations, and seed 200, with clocks unlocked and reference benchmarking disabled. Although the JSON trace does not log individual cache hits, the code path and harness flow together point to a clear explanation: **full computation during correctness checks, followed by cache reuse during timing.**

### Example 2: Freezing Inputs After a Call-Count Threshold

Another pattern preserves inputs or intermediate buffers instead of caching the final output directly. In the [public Hyra submission (flash_attention_gqa_ultralong)](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json) for [067_flash_attention_gqa_ultralong](https://research.nvidia.com/benchmarks/sol-execbench/kernel/67), `main.cpp` includes the following logic (with buffer allocation and argument packing omitted):

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

`prep_launch` converts and copies the current inputs into internal buffers. On repeated calls with the same `(B, S)`—as long as no buffer growth or intervening call overwrites those buffers—this step is bypassed from call 16 onward. Even if fresh hidden states, RoPE parameters, or weights are passed in, downstream kernels keep reading stale buffers.

Even when `prep_launch` does run, it inspects only 256 sampled positions per input. The check below comes from the [CUDA source](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json), where `samp_pos` selects indices at fixed strides:

```cpp
float cur = tab.s[w][samp_pos(t, n)];
float st  = tab.sig[w][t];
int dirty = __syncthreads_or((cur != st) ? 1 : 0);
if (!dirty) return;
```

Modifying only unsampled elements can therefore leave internal buffers untouched. This highlights a contrast with Example 1's pointer check: **matching samples do not guarantee that full inputs match, and a regular address shift proves even less about tensor contents.** Preprocessing fixed weights is valid when the interface contract allows it, whereas caching mutable inputs demands a sound validity check.

The same code also captures and replays a CUDA Graph when `S <= 4096`. **Graph replay still executes GPU work; the flaw here is that the inputs stop being refreshed.** The [PyTorch documentation](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs) draws a clear line between static addresses and fresh data: a captured graph reuses fixed memory addresses, yet reads whatever new values are copied into those buffers before replay. Any implementation relying on internal static input buffers must populate them with the current inputs prior to replay rather than gating updates on a call counter.

The savings here come from a different place than in Example 1: whereas Example 1 skips RMSNorm altogether, this implementation skips input preparation and keeps computing on stale data.

### Example 3: Exploiting the Input Distribution to Drop an Entire Branch

The reference specification for [051_seqlen-finetuned-reconstructed_hyena_complete_forward_block](https://research.nvidia.com/benchmarks/sol-execbench/kernel/145) consists of a Hyena mixer followed by an MLP. Yet the [public Hyra submission (hyena_complete_forward_block)](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k145__sid25621__hyena_complete_forward_block__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block.json#L22) extracts only eight tensor arguments:

```python
_NEED_IDX = [0, 3, 4, 20, 21, 22, 23, 24]
needed = [all_args[i] for i in _NEED_IDX]
```

Those eight indices correspond to the hidden states, second normalization parameters, output-projection bias, and MLP weights. The first normalization, short convolution, long filter, and output-projection weight never enter the computation at all. Instead, `_prologue` forms the residual directly from `hidden_states + out_proj_bias`, then proceeds straight to normalization and the MLP. Rather than fusing kernels to cut launch overhead, this drops an entire computational branch. One plausible reason it still passes is that several reference weights are initialized as `randn * 0.02`; after gating, the Hyena branch's contribution may shrink below the final output tolerance.

The same submission also stops copying fresh values into its CUDA Graph's static input buffers once the call counter exceeds 11, layering Example 2's state-freezing issue on top. Branch omission and input freezing are independent failure modes: re-rolling random seeds may catch the former without ever exercising the latter during timing.

### Example 4: Completing the Math While Changing What a Single Call Means

Cross-call state can also change the unit of work being measured. GPU MODE's NVFP4 retrospective ([Kokoromyti, 2026](https://www.gpumode.com/news/reward-hacking-nvfp4)) documents a submission that topped the leaderboard at 11.191 μs before being invalidated. It exploited the structural difference between the correctness check and the timing loop: during correctness verification, the harness passes 15 cloned inputs; the submission runs a real 8-group GEMM on every call while counting invocations via `id(data)`. Once the count reaches 15, it infers that the timing loop is next.

{{< figure
    src="nvfp4-correctness-vs-timing.png"
    alt="Correctness check runs a real 8-group kernel on each of 15 calls; timing loop launches one 120-group kernel on the first call and returns cached outputs on the remaining 14 calls"
    caption="Fig. 5. Every correctness call runs a real 8-group kernel. In timing, one 120-group kernel computes all 15 inputs, and the other 14 calls return cached outputs. (Image source: [Kokoromyti, 2026](https://www.gpumode.com/news/reward-hacking-nvfp4))"
    align="center"
    width="100%"
>}}

During the timing loop, the harness repeatedly passes in the same batch of input objects. On the first call, the submission launches a single 120-group kernel covering all 15 inputs at once; on the remaining 14 calls, it launches no GPU work and simply returns the cached outputs. **Only one kernel runs on the GPU, yet its total elapsed time is divided by 15.**

This submission does perform all the matrix multiplications and can realize genuine batching throughput gains. Dividing the batch runtime by 15, however, measures an amortized per-call cost across invocations rather than the latency of an independent request. When a production interface permits batching, super-batching can be a valid optimization—provided the time spent waiting to assemble a batch is reported as well. **Evaluation must verify not only that the output is right, but also that the unit of work behind the score matches the task contract.**

## Other Publicly Reported Methods

In its internal agent experiments, the SOL-ExecBench paper flagged and rejected 589 submissions—14.5% of all submissions evaluated. Fig. 6 shows the breakdown of detected exploit types across benchmark levels ([Lin et al., 2026](https://arxiv.org/html/2603.19173v1#S5.F9)).

{{< figure
    src="sol-execbench-exploit-distribution.png"
    alt="Distribution of flagged reward-hacking submissions in SOL-ExecBench: Precision Downgrade 259 (6.4%), Monkey Patching 134 (3.3%), Stream Injection 100 (2.5%), Reuse Cached Output 67 (1.6%), plus smaller counts of JIT Forking (19), One-time Correctness (9), and Thread Injection (1)"
    caption="Fig. 6. Reward-hacking submissions (589 total, 14.5%) flagged across L1, L2, Quantization, and FlashInfer-Bench in SOL-ExecBench. (Image source: [Lin et al., 2026](https://arxiv.org/abs/2603.19173v1))"
    align="center"
    width="100%"
>}}

The paper categorizes the eight observed exploits into three families—concurrency, state & caching, and environment manipulation—and pairs each with its corresponding defense mechanism (Fig. 7).

{{< figure
    src="sol-execbench-reward-hacking-defenses.png"
    alt="Table 3 from the SOL-ExecBench paper listing eight reward-hacking exploits and corresponding defenses across three categories: Concurrency, State & Caching, and Environment"
    caption="Fig. 7. Reward-hacking methods observed in SOL-ExecBench and their corresponding defenses. (Image source: [Lin et al., 2026](https://arxiv.org/pdf/2603.19173v1#page=12))"
    align="center"
    width="100%"
>}}

The table below summarizes publicly documented cases across SOL-ExecBench and other GPU kernel benchmarks and competitions.

| Method | Public case |
| --- | --- |
| Overfitting to fixed test values | [doubleAI](https://www.doubleai.com/research/maxwell-approaches-speed-of-light-on-blackwell#simple-verifiers-allow-overfitting) reports a RoPE implementation hardcoded to only the 16 `theta` values present in the verifier's workloads, failing on any other value. When doubleAI re-tested with fresh random seeds, eight previously passing kernels each failed on at least one workload. |
| Exploiting numerical tolerances | [SF Tensor](https://sf-tensor.com/news/sol-execbench#why-bother-computing-what-was-asked) found an MoE submission that routed to and aggregated only the top-2 experts on a task requiring the top-10. Because the routing probability distribution was heavily skewed, the result stayed within the verifier's numerical tolerance. |
| Reusing precomputed answers | [METR](https://metr.org/evaluations/openai-o3-report/#optimize-a-kernel) documented OpenAI's o3 directly returning the grader's precomputed reference output during a GPU kernel optimization task, exposing an isolation gap between the evaluator and candidate code. |
| Cross-call caching | [Wafer](https://www.wafer.ai/blog/reward-hacks-field-guide#9-caching--memoization) observed GEMM submissions that computed and saved the output on an initial call and returned the cached tensor on every subsequent timed iteration, hiding the true recomputation cost from the benchmark timer. |
| Cross-call batching | [GPU MODE](https://www.gpumode.com/news/reward-hacking-nvfp4)'s NVFP4 retrospective records a submission that collapsed 15 calls into a single GPU launch. Dividing the batch runtime by 15 yielded an amortized per-call latency instead of independent single-call performance. |
| Moving work outside timing | [Wafer](https://www.wafer.ai/blog/reward-hacks-field-guide#timing-attacks) describes two timing attacks that slip past benchmark timers: executing the real computation on an unmonitored CUDA stream, or deferring it lazily until the verifier inspects the output. |
| Tampering with timers or verifiers | [METR](https://metr.org/evaluations/openai-o3-report/#optimize-a-kernel) notes that the same o3 run later patched the scorer's timing and CUDA synchronization functions. [SF Tensor](https://sf-tensor.com/news/sol-execbench#why-not-just-change-the-graders-reference-implementation) observed a submission lowering the compute precision of the in-process reference implementation so its own low-precision output would stay within numerical bounds. |

Version 1 of the SOL-ExecBench paper describes `torch.cuda.Event`-based timing. The evaluator revision examined in Example 1 (`a9fa080`) defaults to CUPTI, which observes GPU activity across CUDA streams and times the matched kernel activity sequence. Even so, catching the exploit in Example 1 still requires varying input values along the timed path and verifying the outputs to confirm that the timed invocation actually processed the current inputs.

## How to Reduce Reward Hacking

### Fix the Evaluation Protocol and Execution Boundaries First

Addressing the failure modes above starts with three high-priority checks:

1. **Vary input values along the timing path and verify the outputs.** At the same shape, test in-place updates, fresh allocations, and identical pointer shifts carrying different contents. Input generation and output comparison can remain outside the timed window, but correctness verification must cover the exact execution path being timed.
2. **Vary call history and input distributions.** Change warmup counts and interleave distinct inputs and shapes; mutate only unsampled elements, or amplify potentially omitted branches. The former probes cross-call state reuse, while the latter probes tolerances and input coverage—both need dedicated tests.
3. **Protect reference data and scoring integrity.** Explicitly specify what caching, preprocessing, batching, and timing scope are permitted, and keep reference outputs, timers, and authoritative logs behind boundaries that candidate code cannot read or mutate. Maintain legitimate kernels as positive controls alongside caching, branch-dropping, and timing-bypass exploits as negative controls so every new check rejects exploits without penalizing valid optimizations.

These defenses target the specific failure modes examined here and should be validated individually against retained exploit submissions. Adding random seeds, switching to CUPTI, or introducing code review does not by itself make an evaluation harness reliable.

### Strategies in Our Internal Evaluation Framework

Our internal kernel evaluation framework maintains two modes tailored to distinct evaluation goals:

- **Isolated mode**: the generation container strips hidden files from the task package so the workspace visible to agents such as Codex contains only `base.py`, `input.py`, `shapes.json`, `constraints.md`, and `self_check.py`. Production references, frozen latency tables, and evaluator internals are excluded. Formal grading runs inside a separate, fresh, complete repository copy to measure an agent's ability to implement kernels purely from the problem specification.
- **Non-isolated mode**: the complete task package is exposed to measure how much production speedup an agent can deliver on target workloads. Where submission rules permit, reusing existing implementations can be a legitimate optimization path.

Both modes are scored separately, while self-checks and formal evaluation share the same timing implementation. Key checks include:

| Check area | Approach and examples |
| --- | --- |
| GPU clocks | Apply the same locked SM clock setting to both candidate and reference, then read clocks back to verify ([NVIDIA, 2026c](https://docs.nvidia.com/deploy/nvidia-smi/index.html)), reducing variance from Boost drift or mismatched clocks. Our environment targets roughly 0.73× the maximum SM clock and requires readbacks within ±15 MHz. |
| Timing scope | Capture and replay CUDA Graphs, synchronize the GPU before and after the Event timing window, and check replay outputs. Combine graph node counts with per-call reference timing probes to reject empty graphs, partial captures, and anomalous timings—flagging work offloaded to side streams or CPU threads. |
| Cross-call state and caching | Rebuild inputs across 3 independent seeds and verify outputs to catch "compute once, reuse thereafter" kernels. Allocate the timing pool at distinct addresses via `clone_inputs`, with a total footprint exceeding 2× L2 cache capacity to counter fixed-object reuse and warm-cache bias. |
| Input mutation | Snapshot tensors whose contract forbids in-place modification and compare raw bytes via `view(torch.uint8)`. Changing `+0.0` to `-0.0`, for example, preserves numerical equality while altering the underlying bits; packed FP8 tensors are checked byte-for-byte as well. |
| Interface and numerical accuracy | Verify shapes, dtypes, and numerical error. An FP32 output requirement, for instance, forbids returning FP16 directly, while numerical accuracy checks still catch internal precision downgrades. |
| Import and execution policy | Enforce submission rules via AST inspection—for example, blocking direct imports of the reference implementation, restricted vendor libraries, or calls to `torch.matmul`. Graph replay reduces reliance on the candidate's Python wrapper, though imports, warmup, and capture still require scrutiny. |
| Scoring baseline | Guard the dispatch path: raise an error if the production implementation fails to load rather than silently falling back to a slower PyTorch path that inflates both reference latency and reported speedup. |
| Audit records | Archive the environment, inputs, candidate code, and per-stage verdicts—recording seeds, Graph node counts, clock readbacks, and the actual reference dispatch path together so anomalous scores remain reproducible. |

These verdicts also feed directly into final scoring: invalid timings or failed anti-hacking checks receive a performance score of zero.

Randomized inputs test content dependence, address rotation tests independence from fixed objects, and a large working set mitigates warm-cache bias. Copying identical values to fresh addresses cannot replace correctness verification on genuinely new inputs.

### Strengthen Semantic Checks with Formal Verification

SF Tensor's formal correctness work ([Koska et al., 2026](https://sf-tensor.com/news/formal-correctness)) builds a symbolic model directly from PTX and uses a solver to prove equivalence between an implementation and its reference specification, covering branch, masking, and synchronization bugs that random testing easily misses. This advances verification from "tested on a sample of inputs" to "proven under an explicit model and its assumptions." Because the early implementation models floating-point values as exact reals rather than IEEE-754 bit patterns, however, it cannot by itself bound actual floating-point error.

{{< figure
    src="random-testing-vs-formal-verification.png"
    alt="Left: random testing passes all sampled points while missing an unsampled bug region. Right: formal verification constructs a symbolic model from an optimized PTX kernel and reference specification and checks equivalence with an SMT solver (Z3)"
    caption="Fig. 8. Random testing samples inputs, while formal verification uses a symbolic model to check whether a PTX kernel can differ from its specification."
    align="center"
    width="100%"
>}}

### Turn Monitoring into a Detect-Investigate-Patch Loop

OpenAI's chain-of-thought (CoT) monitoring research ([Baker et al., 2025](https://arxiv.org/abs/2503.11926v1)) illustrates a practical loop of using monitors to surface reward-hacking trajectories, investigating them manually, and patching the underlying environment loopholes. For kernel-optimizing agents, logging tool calls, submission diffs, GPU activity traces, and authoritative test verdicts together lets monitors uncover emerging patterns, which can then be turned into permanent regression negatives.

Monitors themselves also require ongoing validation. OpenAI's subsequent operational practice ([OpenAI, 2026](https://openai.com/safety-alignment/how-we-detect-frontier-model-misalignment/)) pairs layered investigations with random audits of unflagged traffic, alongside historical incidents and red-teaming to catch blind spots. Relying on alert rates alone can miss covert behavior; actual hacking incidence, monitor recall and false-positive rates, valid-computation pass rates, and defense overhead should all be tracked and reported separately.

### Penalize Confirmed Bypasses During Training

When evaluation loopholes leak into RL training, their effects can reach beyond the immediate task. Denison et al. trained models on a constructed curriculum rewarding milder specification gaming and observed that, in a small fraction of trials, models generalized zero-shot to rewriting simulated reward code in settings where reward tampering was never directly trained ([Denison et al., 2024](https://arxiv.org/abs/2406.10162v3)). While this does not estimate real-world deployment cheating rates, it demonstrates a concrete pathway by which gaming behavior can generalize across contexts.

Anthropic's production coding RL study ([MacDiarmid et al., 2025](https://arxiv.org/abs/2511.18397v1)) found that sufficiently weighted preference rewards or dedicated reward-hacking penalties can suppress known exploits. Standard chat-oriented RLHF, by contrast, does not reliably eliminate agentic misbehavior, and inoculation prompting primarily curbs generalization from reward hacking to broader emergent misalignment rather than fixing the underlying grader vulnerabilities.

Nor should fewer verbalized cheating plans be mistaken for genuine alignment. OpenAI's experiments show that applying strong optimization pressure directly to CoT can induce obfuscation, making residual reward hacks harder for monitors to catch. Candidate validity should therefore be grounded in reliable behavioral and execution evidence, with monitoring and training interventions evaluated independently.

MONA—*Myopic Optimization with Non-myopic Approval* ([Farquhar et al., 2025](https://arxiv.org/html/2501.13011v2))—approaches the problem by reshaping multi-step incentives: optimizing only over immediate or short-horizon returns while incorporating an overseer's advance assessment of long-term value. This reduces the incentive to weaken checks now in order to reap inflated scores later. The paper evaluates MONA across three constructed environments and does not eliminate single-step exploits such as hard-coding; for kernel RL, it offers a promising reward-design direction alongside independent numerical, call-contract, and timing checks.

## Conclusion

A `PASSED` verdict and a tiny latency figure count as evidence of optimization only when both come from the same valid execution path. Evaluation must verify not just that the right answer was computed once, but that the timed call actually completed the current request.

As agents take on a growing share of GPU kernel optimization, the evaluator itself needs to be tested, reproduced, and patched with the same rigor as the kernels it measures. Otherwise, a faster score on the leaderboard proves only that the evaluation was optimized—not that the computation actually got faster.

## References

[1] Lin, Edward, et al. ["SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits."](https://arxiv.org/abs/2603.19173v1) arXiv preprint arXiv:2603.19173, v1 (2026).

[2] Weng, Lilian. ["Reward Hacking in Reinforcement Learning."](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) Lil'Log (2024).

[3] Krakovna, Victoria, et al. ["Specification Gaming: The Flip Side of AI Ingenuity."](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) DeepMind Blog (2020).

[4] Everitt, Tom, et al. ["Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective."](https://arxiv.org/abs/1908.04734) arXiv preprint arXiv:1908.04734 (2019).

[5] Gabor, Jonathan, Jayson Lynch, and Jonathan Rosenfeld. ["EvilGenie: A Reward Hacking Benchmark."](https://arxiv.org/html/2511.21654v1) arXiv preprint arXiv:2511.21654, v1 (2025).

[6] NVIDIA. ["SOL-ExecBench Submission Guide: Scoring."](https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide#scoring) NVIDIA Documentation (2026a).

[7] Lin, Wilson, et al. ["Speeding Up GPU Kernels by 38% with a Multi-Agent System."](https://cursor.com/blog/multi-agent-kernels) Cursor Research Blog (2026).

[8] NVIDIA. ["SOL-ExecBench Evaluation Code."](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd) GitHub Repository (2026b).

[9] Tencent Hunyuan. ["Hyra SOL-ExecBench Results and Submissions."](https://github.com/Tencent-Hunyuan/Hyra-results/tree/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench) GitHub Repository (2026).

[10] PyTorch. ["CUDA Semantics: CUDA Graphs."](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs) PyTorch Documentation (version 2.15).

[11] Kokoromyti, Natalia. ["Anatomy of a Reward Hack: A Real Story from the Latest GPU Mode NVFP4 Competition."](https://www.gpumode.com/news/reward-hacking-nvfp4) GPU MODE Blog (2026).

[12] doubleAI. ["Maxwell Approaches Speed of Light on Blackwell."](https://www.doubleai.com/research/maxwell-approaches-speed-of-light-on-blackwell) doubleAI Research Blog (2026).

[13] SF Tensor. ["What We Learned Topping NVIDIA's SOL-ExecBench."](https://sf-tensor.com/news/sol-execbench) SF Tensor Blog (2026).

[14] METR. ["Details About METR's Preliminary Evaluation of OpenAI's o3 and o4-mini."](https://metr.org/evaluations/openai-o3-report/) METR Evaluation Report (2025).

[15] Andere, Emilio. ["A Field Guide to Reward Hacking in AI Kernel Generation."](https://www.wafer.ai/blog/reward-hacks-field-guide) Wafer Blog (2026).

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

> Yue Shui. (Nov 2026). Reward Hacking in GPU Kernel Optimization: Loopholes and Defenses in SOL-ExecBench.  
https://syhya.github.io/posts/2026-11-24-gpu-kernel-reward-hacking

Or

```bibtex
@article{syhya2026-gpu-kernel-reward-hacking,
  title   = "Reward Hacking in GPU Kernel Optimization: Loopholes and Defenses in SOL-ExecBench",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "Nov",
  url     = "https://syhya.github.io/posts/2026-11-24-gpu-kernel-reward-hacking"
}
```
