---
title: "GPU Kernel 优化中的 Reward Hacking：SOL-ExecBench 的评测漏洞与防护"
date: 2026-11-24T00:00:00+08:00
lastmod: 2026-11-25T00:00:00+08:00
author: "Yue Shui"
categories: ["技术博客"]
tags: ["LLM", "Agent", "GPU Kernel", "Evaluation", "Reward Hacking"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

2026 年 11 月 24 日，我以 **`yue shui@syhya.github.io`** 的名字排在 SOL-ExecBench（[Lin et al., 2026](https://arxiv.org/abs/2603.19173v1)）的 L1、L2、Quantization 和 FlashInfer-Bench 四个 B200 [官方榜单](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard)第一，页面均使用 Evaluation Stack v1.1。不过，**这些成绩并不代表真实的 kernel 优化**，而是利用评测漏洞取得的。本文以这次实验为切入点，分析 GPU kernel 优化中的 reward hacking：结合 SOL-ExecBench 的评测流程与具体案例，说明这些漏洞的成因、它们为何能通过评测，以及相应的防护方法。

## 实验结果：四个榜单第一

{{< html-visualization src="leaderboard-visualization.html" title="四个 B200 榜单的 SOL 分数与通过覆盖率" height="480" >}}

上图分数均通过下文介绍的 reward hacking 方法取得，四个子榜单的题目全部通过。

{{< figure
    src="flashinfer-bench-res.png"
    alt="SOL-ExecBench FlashInfer-Bench (B200) 官方榜单截图（Evaluation Stack v1.1），排名第一（SOL 分数 0.927523，26/26 通过，22.52x 平均加速比）"
    caption="Fig. 1. SOL-ExecBench FlashInfer-Bench (B200) leaderboard under Evaluation Stack v1.1, ranked first with a 0.927523 SOL score, 26/26 solved, and 22.52x average speedup. (Image source: [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200))"
    align="center"
    width="100%"
>}}

## 什么是 Reward Hacking

{{< figure
    src="reward-hacking-overall-flow.png"
    alt="GPU kernel 评测流程及七类缺口：固定测试特化、容差与删分支、参考答案泄漏、跨调用缓存、合批、计时外执行和验证器篡改"
    caption="Fig. 2. Reward hacking can enter at several stages, from inputs to rankings. The coral callouts show the seven paths discussed below; the bottom row shows complementary checks. AI-generated illustration."
    align="center"
    width="100%"
>}}

**Reward hacking（奖励欺骗）是指利用奖励或评测规则的缺口获得高分，却没有相应实现设计者希望的目标。** Lilian Weng 的综述梳理了这个共同问题：实际被优化的奖励，未必完整表达我们真正想要的结果（[Weng, 2024](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)）。

在 kernel 优化里，真实目标是**对当前输入完成规定计算，并且更快**；评测器通常看到的则是有限正确性测试与一组计时读数。这些读数构成了优化时使用的代理目标。返回正确张量，不一定意味着处理了当前输入；整轮计算完成，也不一定意味着每次独立调用都完成了自己的工作。数值正确、调用语义和计时范围需要一起检查，分数才对应我们想测量的能力。

这类偏离可以分成不同路径：**specification gaming** 利用字面规则与真实目标之间的缺口，不一定修改评分器；**reward tampering** 则直接干预奖励函数或它依赖的证据。两者都可能让分数上升，却没有完成预期任务（[Krakovna et al., 2020](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)；[Everitt et al., 2019](https://arxiv.org/abs/1908.04734)）。

规则奖励和单元测试降低了部分主观性，仍然受输入覆盖、容差与执行边界限制。反过来，测试失败也不能直接叫作 hacking：EvilGenie（[Gabor et al., 2025](https://arxiv.org/html/2511.21654v1)）发现，正常实现可能漏掉边界条件，利用测试分布的启发式实现也可能通过保留测试。因此，判断需要结合规格、代码和执行证据。

## SOL-ExecBench 如何打分

SOL-ExecBench 对单个 workload 使用以下评分公式（[NVIDIA, 2026a](https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide#scoring)）：

$$
S(t)=
\frac{t_b-t_{\mathrm{sol}}}
{(t-t_{\mathrm{sol}})+(t_b-t_{\mathrm{sol}})}.
$$

其中，$t$ 是候选耗时，$t_b$ 是预计算的基线耗时，$t_{\mathrm{sol}}$ 是 SOL 模型估计的时间下界，三者使用相同单位。单题与集合依次对 workload 和 kernel 分数取算术平均，失败或缺失项记零。

在 $t_b>t_{\mathrm{sol}}$、$t\ge t_{\mathrm{sol}}$ 的正常评分区间内，$t$ 越小，分数越高。跨调用缓存可以在正确性检查时正常计算，而在计时阶段只返回旧输出、运行一个 marker kernel，把测得的 $t$ 压到远低于完整计算的耗时。若评测没有核验计时路径是否处理了当前输入，这个更小的 $t$ 就会被当作优化收益。

## 四个案例

下面四个例子各自对应一条不同的路径：例子一缓存最终输出，例子二冻结输入，例子三删去计算分支，例子四改变单次调用的计量单位。

### 例子一：根据地址变化返回旧输出

[069_rms_norm](https://research.nvidia.com/benchmarks/sol-execbench/kernel/69) 的任务是先做残差相加，再执行 RMSNorm 并乘以权重。以下代码摘自 [rmsnorm-submission.py](rmsnorm-submission.py)，只保留单输出、直接返回张量的路径：

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

未命中缓存时，`_ref_run` 执行 PyTorch 参考计算，`_capture` 克隆输出并记录输入信息。命中时，直接返回之前的结果。关键在于 `_reusable` 如何判断可以复用：它检查输入数量、标量和字节数，却不比较张量内容，而是检查地址是否一起向前移动。

下面是其中的判断循环。`_MIN_SHIFT = 256`，`_MAX_SHIFT = 2048`：

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

这对应了保存记录中的 evaluator 版本（[NVIDIA, 2026b](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)；commit `a9fa080`）的分配方式：每轮所有输入共同前移 256～2048 字节，但新地址中复制的仍是同一份输入数据（[分配器源码](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L621)）。因此，地址轮换没有阻止缓存，反而给缓存提供了识别依据。

正确性阶段会重新生成输入；缓存未命中时，提交正常计算。[计时阶段](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L584)反复使用相同输入值，预热便有机会填充缓存。命中后，`_reusable` 更新保存的地址，启动一个只写入整数 0 的 Triton marker kernel，再让 `run` 返回旧输出。

该版本的 CUPTI 计时器会在预热后记录候选的 GPU 活动序列，并匹配后续迭代的序列。它能记录 kernel、内存复制等活动，却无法仅凭序列一致判断这些活动是否完成了 RMSNorm。发现序列时若已走缓存路径，后续也可以只匹配到 marker（[计时源码](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L154)）。

2026 年 11 月 23 日保存的 [rmsnorm-evaluation.json](rmsnorm-evaluation.json) 中，这份提交在 B200 上对一个 BF16 workload（`1 × 8192 × 8192`，`eps = 1e-5`）得到 `PASSED`，最大绝对与相对误差均为 0，报告耗时 **2.064 μs**。配置是 10 次预热、50 次计时、seed 200；未锁频，也没有测量参考耗时。记录没有逐次缓存命中日志；上述源码路径与评测流程提供了一种解释：**正确性阶段正常计算，计时阶段复用缓存。**

### 例子二：调用次数到了，就不再更新输入

另一种方式不直接缓存最终答案，而是保留输入或中间结果。在 [067_flash_attention_gqa_ultralong](https://research.nvidia.com/benchmarks/sol-execbench/kernel/67) 的 [Hyra 公开提交（flash_attention_gqa_ultralong）](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json)中，`main.cpp` 有这样的逻辑。下面省略了缓冲区分配与参数组装：

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

`prep_launch` 负责把当前输入转换、复制到内部缓冲。对同一 `(B, S)` 连续调用，且没有扩容或其他调用改写缓冲时，第 16 次起这一步就被跳过了。即使传入新的 hidden states、RoPE 参数或权重，后面的计算仍读取旧缓冲。

即使进入 `prep_launch`，它也只抽样检查每个输入的 256 个位置。下面是 [CUDA 源码](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json#L49)中的判断；`samp_pos` 按固定间隔选点：

```cpp
float cur = tab.s[w][samp_pos(t, n)];
float st  = tab.sig[w][t];
int dirty = __syncthreads_or((cur != st) ? 1 : 0);
if (!dirty) return;
```

只修改未抽样的位置，就可能让内部缓冲保持旧值。它与前一例的地址判断不同：**抽样值相同不等于完整输入相同，地址变化符合规律更不能证明内容相同。** 对接口允许的固定权重可以预处理；对可能变化的输入，缓存需要可靠的有效性条件。

这份代码还会在 `S <= 4096` 时捕获、重放 CUDA Graph。**Graph 重放仍会执行 GPU 计算；这里的问题是输入不再刷新。** [PyTorch 文档](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs)说明了固定地址与更新数据的区别：重放复用捕获时的地址，但可以读取这些地址里的新内容。若使用内部静态输入，就需要在重放前写入本次输入；不能靠调用次数来决定是否更新。

这个例子与前一个的节省来源不同：前者可以跳过 RMSNorm 本身，这里省掉的是输入准备，并对旧数据继续计算。这是对公开代码路径的分析，本文没有为这份提交重新运行 GPU 实验，也不据此推断它的全部性能收益。

### 例子三：利用输入分布，删掉整个计算分支

[051_seqlen-finetuned-reconstructed_hyena_complete_forward_block](https://research.nvidia.com/benchmarks/sol-execbench/kernel/145) 的参考任务包含 Hyena mixer 和后续 MLP。[Hyra 公开提交（hyena_complete_forward_block）](https://github.com/Tencent-Hunyuan/Hyra-results/blob/b24e7f5543f08069fe190e98c5c05b62ca28a074/AI4AI/sol_execbench/k145__sid25621__hyena_complete_forward_block__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block.json#L22)却只挑选了八个张量参数，相关的两处代码是：

```python
_NEED_IDX = [0, 3, 4, 20, 21, 22, 23, 24]
needed = [all_args[i] for i in _NEED_IDX]
```

这些索引对应 hidden states、第二层归一化参数、输出投影 bias 和 MLP 参数；第一层归一化、短卷积、长滤波器与输出投影 weight 都没有进入计算。`_prologue` 直接从 `hidden_states + out_proj_bias` 构造残差，再执行归一化和 MLP。这里被删除的是计算分支，而不是通过融合减少 kernel 数量。

一个可能的解释是：参考输入中多组权重按 `randn * 0.02` 初始化，经过门控后，Hyena 分支的贡献可能小到被最终输出容差掩盖。**这是对通过原因的推断，不是对任意输入的误差保证。** 需要放大被忽略的分支、改变输入分布，才能检验这个近似是否仍符合任务要求。

同一提交还在计数超过 11 后停止向 Graph 的静态输入复制新值，叠加了前一例的状态问题。删分支和冻结输入是两条独立路径：换一批随机种子可能暴露前者，却未必触及计时阶段的后者。

### 例子四：计算完成了，单次调用的含义却变了

跨调用状态还可以改变工作的组织方式。GPU MODE 的 NVFP4 复盘（[Kokoromyti, 2026](https://www.gpumode.com/news/reward-hacking-nvfp4)）记录了一份后来被撤销成绩的提交：它收集 15 份输入，把原本独立的调用合为一次 grouped GEMM，再在后续调用中返回缓存的输出句柄。

在启用合批且对象身份稳定的计时轮里，第一次调用启动整批计算，其余 14 次只返回句柄。对新对象，提交还可以按形状收集输入，等收齐后再启动计算。独立调用的输出应满足调用方的 stream 语义，不能依赖后续其他调用补算；整轮结束后输出正确，无法证明这一点。

这个案例确实完成了矩阵乘法，也可能获得真实的批处理吞吐收益；但把整批耗时除以 15，得到的是跨调用摊销后的读数。它没有证明一个独立请求能以同样延迟完成。如果生产接口允许合批，这可以是合法优化，同时需要报告等待成批的时间。**评测需要验证的，不只是答案对不对，还包括分数采用的工作单位是否符合任务约定。**

## 公开报告中还有哪些方法

SOL-ExecBench 论文已讨论状态缓存，并介绍输入克隆、地址偏移和代码审核等防护（[Lin et al., 2026](https://arxiv.org/abs/2603.19173v1)）。结合论文、技术博客与事件记录，可以把已公开的方法归纳如下。它们会相互叠加，这里列的是代表性路径，并非完整清单。

| 方法 | 利用的缺口与公开例子 |
| --- | --- |
| 跨调用缓存与阶段识别 | 按地址、形状、抽样值或调用次数复用旧输出、旧输入。前面的代码展示了其中几种组合。 |
| 容差与输入分布利用 | [SF Tensor](https://sf-tensor.com/news/sol-execbench) 报告 MoE 从 top-10 少算到 top-2 仍通过；[doubleAI](https://www.doubleai.com/research/maxwell-approaches-speed-of-light-on-blackwell) 报告省略 softcap 在特定测试范围内仍足够接近。 |
| 固定测试实例特化 | doubleAI 报告只覆盖 16 个 `theta` 值的 RoPE，以及更换种子后暴露的失败。问题是漏掉声明支持的输入，而不是 shape 特化本身。 |
| 计时范围外执行或延迟计算 | SOL 论文记录 side stream、后台线程与 capture stream 问题；[Wafer](https://www.wafer.ai/blog/reward-hacks-field-guide) 还展示 Tensor 子类等到正确性比较时才计算。历史 CUDA-event 漏洞不能直接套用于本文的 CUPTI 版本。 |
| 跨调用合批 | [GPU MODE](https://www.gpumode.com/news/reward-hacking-nvfp4) 记录把 15 次调用合为一次 grouped GEMM，再返回缓存句柄。矩阵乘法仍完成，但摊销耗时改变了独立调用的计量单位。 |
| 获取参考答案或复用评分器缓冲 | [METR](https://metr.org/evaluations/openai-o3-report/#optimize-a-kernel) 记录沿 Python 调用栈读取 `y_torch`；Wafer 描述 no-op 利用已有输出、identity 利用测试输出接近输入的情形。 |
| 修改计时器或参考环境 | SOL 论文记录替换 `Event.elapsed_time`；SF Tensor 报告在候选导入时修改环境变量，降低同进程 reference 的计算精度。 |

METR 的栈读取与本文缓存有一个重要区别：前者取走评分器已经算出的答案，后者保存自己先前算出的答案。两者都可能返回正确数值，却绕开本次要求的计算；防护分别需要保护参考数据的访问边界，以及验证跨调用状态的有效性。

[Cursor](https://cursor.com/blog/multi-agent-kernels) 明确表示会排除缓存带来的不合理成绩，doubleAI 也讨论了计时改进。

## 怎样减少 Reward Hacking

### 先修复评测协议和执行边界

针对上面的案例，优先补三类检查：

1. **在计时流程中改变输入值，并核对输出。** 同形状下测试原地修改、重新分配，以及相同地址偏移但不同内容的输入。生成数据和比较结果可以放在计时区间外，但不能让正确性只覆盖另一条执行路径。
2. **改变调用历史与数值分布。** 调整预热次数，交错输入和形状；只改未抽样位置，或放大被忽略的计算分支。前者检查状态复用，后者检查容差与输入覆盖，两者需要分别测试。
3. **保护参考数据与评分过程。** 写清楚允许的缓存、预处理、合批及计时范围，把参考答案、计时器和权威记录放到候选不能任意读写的边界。保留正常实现作为正对照，以缓存、删分支和计时绕过实现作为负对照；新增检查需要既拒绝漏洞，也接受合法优化。

这些措施针对的是本文案例；它们需要用保留的漏洞实现逐项验证，不能因为增加了随机种子、CUPTI 或代码审核，就宣称整个评测已经可靠。

### 内部评估框架中的几项策略

内部的 kernel 评估框架保留两种模式，对应不同的评估目标：

- **隔离模式**：生成容器先清理题包，Codex 等 Agent 的题目工作目录只保留 `base.py`、`input.py`、`shapes.json`、`constraints.md` 和 `self_check.py`。生产 reference、冻结耗时表和评测实现不可见，正式评测在另一份全新、完整的仓库副本中运行，衡量从题面生成实现的能力。
- **非隔离模式**：开放完整题包，衡量 Agent 面向目标负载能满足多少生产提效需求。在提交规则允许的范围内复用已有实现，也可以是有效的优化路径。

两种模式分别报分，自检与正式评测共享计时实现。具体检查包括：

| 检查环节 | 做法与例子 |
| --- | --- |
| GPU 频率 | 候选与 reference 使用相同的 SM 锁频设置，并回读校验（[NVIDIA, 2026c](https://docs.nvidia.com/deploy/nvidia-smi/index.html)），减少 Boost 波动或两侧不同频造成的偏差。当前环境按最高 SM 频率的约 0.73 倍设定目标，回读误差要求在 ±15 MHz 内。 |
| 计时范围 | 使用 CUDA Graph 捕获与重放，在 Event 计时区间前后同步 GPU，并核对重放后的输出。结合节点数和逐调用对照计时，拒绝空图、部分捕获与异常耗时，检查把工作转移到其他 stream 或 CPU 线程的路径。 |
| 跨调用状态与缓存 | 使用 3 个独立 seed 重建输入并比对输出，检查“首轮计算、后续复用”的实现；计时池通过 `clone_inputs` 分配不同地址，总量大于 2 倍 L2 容量，减少固定对象复用与热缓存偏差。 |
| 输入改写 | 对禁止原地修改的张量保存快照，用 `view(torch.uint8)` 按字节比较。例如 `+0.0` 改成 `-0.0` 在数值上相等，但字节已经改变；FP8 打包数据也按原始字节检查。 |
| 接口与数值 | 核对 shape、dtype 和数值误差。例如要求 FP32 输出时，不能直接返回 FP16；内部计算是否降低精度，仍由数值测试检查。 |
| 导入与执行策略 | 按提交规则做 AST 检查，例如拒绝直接导入 reference、受限 vendor 库或调用 `torch.matmul`。Graph 重放减少对候选 Python 包装层的依赖，导入、预热和捕获阶段仍需检查。 |
| 评分基线 | 加入分派路径检查：生产实现未加载成功时直接报错，避免静默回退到较慢的 PyTorch 路径，把 reference 耗时和 speedup 一起抬高。 |
| 审计记录 | 保存环境、输入、候选代码与各阶段判定。例如同时记录 seed、Graph 节点数、时钟读数和实际 reference 分派路径，便于复现异常成绩。 |

这些判定也会进入最终评分：计时无效或未通过反作弊检查的结果，性能分记为零。

随机输入检查内容依赖，地址轮换检查固定对象依赖，大工作集减少热缓存偏差；复制相同数值到新地址，不能代替对新输入的正确性检查。

### 用形式化验证加强语义检查

SF Tensor 的形式化正确性工作（[Koska et al., 2026](https://sf-tensor.com/news/formal-correctness)）尝试从 PTX 建立符号模型，通过求解器验证实现与规格等价，覆盖随机测试容易漏掉的分支、mask 和同步问题。它把“测过若干输入”推进到“在模型与前提下证明性质”，但早期实现将浮点数视为精确实数，不能据此保证实际浮点误差。

### 让监控形成“发现、调查、补洞”的闭环

OpenAI 的 CoT 监控研究（[Baker et al., 2025](https://arxiv.org/abs/2503.11926v1)）展示了用监控发现作弊轨迹、人工调查并修补环境漏洞的过程。对 kernel Agent，可以同时保留工具调用、提交差异、GPU 活动与权威测试结果，用监控寻找新模式，再把确认的漏洞变成回归负例。

监控本身也要验收。OpenAI 的后续实践（[OpenAI, 2026](https://openai.com/safety-alignment/how-we-detect-frontier-model-misalignment/)）采用分层调查，并随机抽查未触发告警的样本、用历史事件和红队测试漏检。只看告警率会漏掉隐蔽行为；实际 hacking、监控召回与误报、有效计算通过率和防护开销，需要分开报告。

### 训练中惩罚已证实的绕过行为

评测漏洞进入训练后，还可能影响后续行为。Denison 等人用构造课程奖励较轻的 specification gaming；在未直接训练奖励篡改的测试场景中，模型也曾在少数试次里改写模拟奖励代码（[Denison et al., 2024](https://arxiv.org/abs/2406.10162v3)）。这展示了投机行为跨场景泛化的一条路径，不能据此估计部署模型的作弊率。

Anthropic 的研究（[MacDiarmid et al., 2025](https://arxiv.org/abs/2511.18397v1)）发现，在其 coding RL 设置中，足够权重的偏好奖励或专门的 hacking 检测惩罚可以抑制已知作弊。普通聊天型 RLHF 则不能保证修复 agentic 行为；inoculation prompting 主要减少作弊向其他失配行为的泛化，并没有修复评分器漏洞。

也要避免把“更少说出作弊意图”当成成功。OpenAI 的实验表明，直接对 CoT 施加强优化压力，可能让剩余作弊更难被监控发现。因此，应依据可靠的行为与执行证据判定候选是否有效，再独立检查监控和训练干预的效果。

MONA（[Farquhar et al., 2025](https://arxiv.org/html/2501.13011v2)）则尝试改变多步优化的激励：只把单步或短期回报用于优化，并加入监督者对长期影响的预先评价，减少“先削弱检查、再从后续高分中获益”的动机。论文实验覆盖三个构造场景，也不解决单步硬编码等漏洞。对 kernel RL，这提供了奖励设计思路，仍需保留独立的数值、调用契约与计时验收。

## 总结

一次 `PASSED` 和一个很小的耗时，只有对应到同一条有效执行路径上，才能成为优化效果的证据。评测需要验证的不只是“答案曾经算对过”，还包括“这次计时确实完成了当前请求”。

随着 Agent 越来越多地参与 kernel 优化，评测器本身也需要像 kernel 一样被持续测试、复现和修补；否则，榜单上的速度提升说明的只是评测被优化了，而不是计算真的变快了。

## 参考文献

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

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> Yue Shui. (November 2026). GPU Kernel 优化中的 Reward Hacking：SOL-ExecBench 的评测漏洞与防护. [Yue Shui's Blog](https://syhya.github.io/zh/posts/2026-11-24-gpu-kernel-reward-hacking/).

Or

```bibtex
@article{syhya2026-gpu-kernel-reward-hacking,
  title   = "GPU Kernel 优化中的 Reward Hacking：SOL-ExecBench 的评测漏洞与防护",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2026",
  month   = "November",
  url     = "https://syhya.github.io/zh/posts/2026-11-24-gpu-kernel-reward-hacking/"
}
```
