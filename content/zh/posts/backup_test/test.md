## Test-time Scaling

除了在训练阶段投入算力，还可以在推理阶段通过增加计算量来提升性能。这被称为 **Test-Time Scaling** ([Snell et al.,  2024](https://arxiv.org/abs/2408.03314))。主要策略包括并行采样（Parallel Sampling）和串行修正（Sequential Refinement）。

*   **并行采样（Parallel Sampling）**：例如 Best-of-N 或多数投票（Majority Voting）。模型针对同一问题生成多个独立样本，然后通过验证器或投票选出最佳答案。
*   **串行修正（Sequential Refinement）**：如自我修正（Self-Correction），模型基于上一次输出进行迭代改进。

### Large Reasoning Models



{{< figure
    src="reasoning_scaling.png"
    caption="Fig. 8. Additional RL training and additional test-time compute improves competitive mathematics performance. (Image source: [OpenAI, 2025](https://arxiv.org/abs/2502.06807))"
    align="center"
    width="100%"
>}}

与传统 LLM 不同，LRMs 类似于一个人在解决复杂问题时会逐步思考：强化学习优化了这一思维链过程，帮助模型识别和纠正错误、将复杂任务分解为可管理的部分，并在某种方法失败时探索替代方案。研究团队在 CodeForces 平台上系统评估了不同模型的竞技编程能力，模型性能呈现出清晰的阶梯式提升：

| 模型 | CodeForces Rating | 百分位 |
|------|-------------------|--------|
| GPT-4o | 808 | 11th |
| o1-preview | 1258 | 62nd |
| o1 | 1673 | 89th |
| o1-ioi (full strategy) | 2214 | 98th |
| o3 | 2724 | 99.8th |

{{< figure
    src="codeforces_progression.png"
    caption="Fig. 9. Performance progression from GPT-4o to o3 on CodeForces benchmark. (Image source: [OpenAI, 2025](https://arxiv.org/abs/2502.06807))"
    align="center"
    width="90%"
>}}

该研究的一个核心发现是：**扩展通用强化学习比依赖领域特定技术更能提供稳健的性能提升路径**。o1-ioi 系统采用了专门为 2024 年国际信息学奥林匹克（IOI）设计的手工推理策略（将问题分解为子任务、每个子任务采样 10,000 个解决方案、基于聚类和重排序的提交选择），而 o3 则完全不依赖这些人工设计的策略——复杂的测试时推理策略**从端到端 RL 训练中自然涌现**。

{{< figure
    src="o3_self_testing.png"
    caption="Fig. 10. o3 testing its own solution by writing brute-force solutions for verification—a sophisticated reasoning strategy that emerged naturally from RL training. (Image source: [OpenAI, 2025](https://arxiv.org/abs/2502.06807))"
    align="center"
    width="90%"
>}}

如上图所示，o3 在解决复杂问题时会自发地编写简单的暴力解法，并将暴力解的输出与优化算法进行交叉验证，通过这种自我验证机制捕获潜在错误。在 2024 年 IOI 竞赛中，o3 在仅使用 50 次提交且**无任何领域特定策略**的情况下，以 395.64 分超越金牌线（约 360 分），而 o1-ioi 需要放宽到 10K 次提交才能达到 362.14 分。

### 效率权衡