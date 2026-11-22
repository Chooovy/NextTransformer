# NextTransformer-MoE

面向 **Mixture-of-Experts** 语言模型的算力受控对比平台。沿用 NextTransformer 的契约：数据相同、顺序相同、每 token 训练
FLOPs 相同、评测相同；参数量和速度是结果，不是约束。平台设两个固定赛道，形状与预算复现 Marin 的 compute-optimal
Grug MoE：

| 赛道 | 结构 | 上下文 | 训练 token | 每 token FLOPs | 训练总 FLOPs | 总参数 / 激活参数 |
|---|---|---|---|---|---|---|
| **d768** | 8 层 × 768，GQA 6/1，256 专家 × 384 选 4，另有宽 768 的共享专家 | 4,096 | 4.42B（16,875 × 262,144） | 9.154 × 10⁸ | 4.05 × 10¹⁸ | 1.99B / 209M |
| **d1024** | 11 层 × 1024，GQA 8/2，256 专家 × 512 选 4，另有宽 1024 的共享专家 | 4,096 | 8.43B（16,080 × 524,288） | 1.656 × 10⁹ | 1.40 × 10¹⁹ | 4.70B / 341M |

- **数据**：[`xjxyys/dolma3_hq_300B_shuffled_clean`](https://huggingface.co/datasets/xjxyys/dolma3_hq_300B_shuffled_clean)，固定 revision。
  - 分词器为 dolma2（cl100k_base 加 22 个特殊 token），用 tiktoken 实现，与 HF 版本逐 id 一致。
  - 训练 shard 按 00000、00001… 的顺序读取；00255 作为开发集；6 个 `validation/*.parquet` 保留给正式评测。
- **总分 MoE-CORE**：对 27 项中心化准确率取平均，即 (22 × CORE-22 + 5 × LC-5) / 27。
  - CORE-22 与稠密排行榜完全相同。
  - LC-5 是长上下文任务，长度为 **2,048 / 3,072 / 4,096**：passkey、单针检索、多键检索、多键四选一、变量追踪。
  - 另报 8,192 长度外推（不计分）和长文档分位置 BPB。
- **吞吐**：统一由 `moe.throughput` 测量（10 步预热，30 步计时，去掉首尾各 10%）。
  - 每次在同一节点上先后测 baseline 和待测方法，排行榜发布两者的**相对吞吐**。
  - 规划参考值见 [references.json](references.json)。
- **算力控制**：matmul FLOPs 在 CPU 上逐 block 实测，注意力按边数规则计费。
  - 准入要求每 token FLOPs 与 anchor 相差不超过 1%，整个训练的累计偏差不超过 2%。

规范文本见 [PROTOCOL.md](PROTOCOL.md)，提交流程见 [README.md](README.md#submitting-a-method)。提交流程与原
NextTransformer 一致：插件 → 校准 → `--formal` 训练 → 吞吐测量 → 审计与导出 → PR。
