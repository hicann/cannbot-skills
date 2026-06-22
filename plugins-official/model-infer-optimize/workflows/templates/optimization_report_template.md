````markdown
<!--
写入规范：

正面记录：仅写与模型本身相关、对外可复现的内容——架构特性、并行 / 算子 / 图模式 / KVCache 等优化思路与方案、模型代码改动要点、性能与精度结论。以现在时陈述当前实现，仅记录最终方案（保留技术依据，省略试错与演化过程）。

避免写入：
- 环境与本地痕迹：调试期才有意义的字段（绝对路径、机器名、IP、端口、卡号、临时目录、shell 变量等）
- 版本控制痕迹：commit hash、SHA、HEAD、分支 / tag、提交序列等元信息
- 中间过程产物：临时 yaml / 脚本、改名前路径、占位日期、TODO 备忘
- 过程性标记：状态徽章（emoji / 删除线）、流程编号（"阶段 N" 等）、编排状态（"确认 / 已完成" 等）

可读性约束：
- 自然语言描述：每个优化点用完整中文句子描述场景、方案与收益；禁止以配置参数或接口名做段落开头，禁止以变量名 / shape 表达式 / 代码片段做主语或谓语，代码块仅用于实现要点 / 伪代码小节
- 配置参数不入正文：`EP=8` / `attn_tp=1` / `kvp_size > 1` 等具体取值写在表格、yaml 引用或伪代码块中，正文用自然表达如"专家并行切到 8 卡"；优化思路若已在通用文档（如 Llama / DeepSeek）介绍则引用即可
- 章节引言：每个一级章节开头先写 1-2 句过渡，说明本章涉及哪类优化、与基线或上一章的关系；禁止直接进入表格或子小节
- 数据 / 表格后补总结句：每个性能或精度表格之后用 1 句自然语言总结收益与精度结论，例如"该改造保持精度不变，Decode 时延从 X 降至 Y，主要收益来自 Z"；禁止只写"精度：与基线一致。性能：X ms / Y ms"这类裸数据
- 章节标题用名词短语而非技术口号：标题示例 "MLA 低秩压缩优化"、"MoE 模块的专家并行与通信适配"；避免 "FA v2 + hybrid TND-sliding + BNSD-full" 这种紧凑拼接式
- 占位符使用：所有 `{...}` 占位符（包括标 `描述示例：` 的过渡句）仅作格式与风格参考，必须按本模型实际情况重写，禁止直接照抄
- 术语沿用既有命名：优先使用仓内代码函数名、仓库设计文档和注册模型既有技术文档命名；描述性词汇优先用中文，行业缩写和技术名保留英文（如 MoE / MLA / Decode）；禁止为已有方案自造同义新词。反例：把 double routing 写成 "双轮 AllToAll"

反例：

并行化：EP=8（128 experts / 8 = 16 expert/rank）+ embed/lmhead_tp=8 + attn_tp=1。Paged Attention 接入 framework PA 框架管理 KV cache。MoE EP decode 默认走 MC2 dispatch_v2/combine_v2；A2 平台 experts_per_rank > 24 时自动回退 double_routing AllToAll。

正例：

本样例采用混合并行部署：MoE 模块按 Expert Parallel (EP) 切分到 8 卡，每张卡承载部分路由专家；Embedding 与 LM Head 采用 Tensor Parallel (TP) 切分到 8 卡以降低单卡显存压力；Attention 模块保持单卡计算，避免引入额外通信开销。具体并行配置参见 yaml 文件中的 `parallel_config` 字段。

KV Cache 采用 Paged Attention 方案管理，由推理框架统一负责 KV Cache 块的分页分配与回收，避免连续显存占用，在长序列与高并发场景下显著提升显存利用率。实现细节可参考 framework 中的 Paged Attention 接口。

MoE 模块在 Decode 阶段通过 [torch_npu.npu_moe_distribute_dispatch_v2](链接) 和 [torch_npu.npu_moe_distribute_combine_v2](链接) 算子实现 EP 并行下多卡间的路由通信。在 Atlas A2 平台上，MC2 路径受算子硬件约束限制，每张卡承载的专家数（`experts_per_rank`）不能超过 24，将切换到 `double_routing + AllToAll` 等回退路径。
-->

# {model_name} 模型优化报告

> 生成时间：{date}
> 优化执行者：{agent / manual}

---

## 1. 概述

{描述示例：用 1-2 句概括模型类型、架构亮点、参数规模、目标硬件平台与部署规模。}

- HuggingFace: {hf_repo_link}
- 架构: {Dense / GQA / MoE / MLA / Diffusion / 多模态}
- 总参数量: {N B}
- 硬件平台: {Atlas A2 / A3}
- 部署规模: {N 卡，single-node / multi-node}
- 量化模式: {BF16 / W8A8 / W8A8C8 / W4A16}

---

## 2. 模型结构

{模型网络结构简图}

```
Embedding
  └─ Transformer Block × N
       ├─ Attention (type: {GQA / MHA / MLA})
       │    └─ {关键计算路径}
       ├─ {FFN / MoE}
       │    └─ {结构描述}
       └─ Residual Connection
  └─ LM Head
```

关键特殊点：

- {描述示例：列出本模型有别于通用 Transformer 的结构特殊点，便于后续优化时识别需特殊处理的模块；常见类型包括共享权重、特殊投影方式、混合 attention 模式等。}

---

## 3. 性能基线

{描述示例：说明基线采集场景，包括执行模式、量化模式、yaml 配置与部署规模，作为后续各项优化效果的对比基准。}

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill 耗时 (ms) | {value} | input_len={}, batch_size={} |
| Decode 单步耗时 (ms) | {value} | batch_size={} |
| 端到端吞吐 (tokens/s) | {value} | input_len={}, output_len={}, batch_size={} |
| 显存占用 (GB) | {value} | - |

{简要描述基线的执行模式与 yaml 配置定位}

### 3.1 精度基线

测试输入：{标准输入内容或数据集}
基线输出：{模型输出结果，用于后续精度对比}

---

## 4. 并行切分

<!-- 单卡部署时整段删除 -->

{描述示例：说明本样例并行策略的总体思路，包括各模块（Attention / MoE / Dense MLP / Embedding / LM Head）的切分方式与设计依据。}

部署拓扑（{yaml 文件名}）：

| 模块 | 切分 | 通信形式 |
|------|------|---------|
| Attention | {attn_tp=N / attn_dp=N} | {AllReduce / RS+AG / 无} |
| Dense MLP | {dense_tp=N / DP} | {AllReduce / 无} |
| MoE | {EP=N / TP=N} | {dispatch_v2/combine_v2 / AllToAll} |
| Embedding | {embed_tp=N / DP} | {AllReduce / 无} |
| LM Head | {lmhead_tp=N} | {AllGather} |

技术依据：

- {自然语言段落，解释每个切分选择的依据}
- {根据实际并行策略补充}

<!-- 若有非常规并行（KVP / SP / o_proj K 轴切分等），按 4.1 / 4.2 子节展开 -->

{简要描述本节并行优化的收益与精度结论}

---

## 5. KVCache 与 Attention

{描述示例：说明 KV Cache 管理方案与 Attention 计算的整体策略，包括 PA 接入、FA 算子选型、Prefill 与 Decode 路径差异。}

### 5.1 KVCache 管理

{KVCache 方案描述：Paged Attention / Legacy / MLA 压缩}

| 参数 | 值 | 说明 |
|------|-----|------|
| KV 模式 | {Paged / Legacy / MLA} | {简短说明} |
| block_size | {32 / 64 / 128} | Paged 模式的块粒度 |
| cache_mode | {PA_NZ / 其他} | 缓存格式 |

### 5.2 Flash Attention 算子选型

{FA 算子选型与 layout 设计描述}

| 项 | Prefill | Decode |
|---|---|---|
| FA op | {torch.ops.npu / torchair.ops} | {同左} |
| input_layout | {TND / BNSD} | {同左} |
| sparse_mode | {3 / 4} | {0} |

<!-- 若有 MLA 压缩 / SlidingWindow / 多 attn_type 等特殊处理，按 5.3 / 5.4 子节展开 -->

{简要描述本节 KVCache 与 FA 改造的精度结论与性能收益}

---

## 6. 算子融合

{描述示例：本节列出本样例引入的所有融合算子，涵盖 KVCache 与 Attention、MoE、Norm 与激活、量化等阶段引入的实现；"来源"列标注算子在哪个章节首次引入。}

<!-- 总览表：包含 KVCache 与 Attention 章节引入的 FA / MLA 类算子、量化章节引入的量化算子、以及算子融合阶段单独引入的 Norm / RoPE / MoE 通信等融合算子 -->

| 模块 | 实现 | 来源 | 触发位置 |
|------|------|------|---------|
| RMSNorm | `npu_rms_norm` | 算子融合 | 每层 Norm |
| Residual + RMSNorm | `npu_add_rms_norm` | 算子融合 | 层间残差 |
| RoPE | `npu_rotary_mul` | 算子融合 | 每层 Q+K |
| Flash Attention (decode) | `npu_fused_infer_attention_score_v2` | KVCache 与 Attention | Attention 计算 |
| MLA Prolog (decode) | `npu_mla_prolog_v3` | KVCache 与 Attention | MLA 前置 |
| MoE Router | `npu_moe_gating_top_k` | 算子融合 | MoE 路由 |
| MoE Dispatch (decode) | `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2` | 算子融合 | MoE 通信 |
| MoE Expert | `npu_grouped_matmul` | 算子融合 | Expert 计算 |
| Quant Linear | `npu_quant_matmul` 等 | 量化适配 | Linear 量化 |

{简要描述算子融合整体的精度结论与性能影响}

---

## 7. 量化适配

<!-- 模型未启用量化时整段删除 -->

{描述示例：说明本样例的量化接入路径、量化对象与改造范围，以及融合算子与量化冲突时的回退策略。}

| 项 | 内容 |
|----|------|
| 量化模式 | {W8A8 / W8A8C8 / W4A16} |
| 量化对象 | {Linear / MoEGMM / KVCache / ...} |
| 量化产物 | {quant_export_dir} |
| compressed-tensors 契约 | {满足 / 有补充诉求} |

<!-- 若有融合算子与量化冲突的回退，按列表说明 -->

{简要描述量化前后的精度与性能对比}

---

## 8. 图模式

{描述示例：说明本样例支持的图模式后端、切换方式与覆盖范围；npugraph_ex 与 GE graph 通常为并行关系，可按需选择。}

### 8.1 npugraph_ex

{适配范围与关键改造点}

| 项 | 配置 |
|----|------|
| 后端 | torch.compile aclgraph |
| 覆盖范围 | Decode（Prefill 保持 eager） |

{性能数据表与简要描述本后端的收益}

### 8.2 GE graph

{适配范围与关键改造点}

| 项 | 配置 |
|----|------|
| 后端 | torchair GE graph，`fullgraph=True` |
| 覆盖范围 | Decode（Prefill 保持 eager） |

{性能数据表与简要描述本后端的收益}

---

## 9. 累计性能演进

{描述示例：说明各项优化逐步叠加后的累计效果定位，标注最大收益来源与最终性能水平。}

| 路径 | 关键改造 | Prefill (ms) | Decode (ms) | vs baseline (decode) |
|------|---------|--------------|-------------|----------------------|
| 基线 (eager) | — | {baseline} | {baseline} | 1.0× |
| + 并行切分 | {简述} | {value} | {value} | {ratio}× |
| + KVCache + FA | {简述} | {value} | {value} | {ratio}× |
| + 算子融合 | {简述} | {value} | {value} | {ratio}× |
| + 量化适配 | {简述} | {value} | {value} | {ratio}× |
| + 图模式 | {简述} | {value} | {value} | {ratio}× |

测试条件：{batch_size / seq_len / 卡数 / 数据类型}

{简要描述最大收益来源与最终性能定位}

---

## 10. 算子需求

<!-- 改造过程中未遇到需 CANN 后续扩展支持的算子约束时整段删除 -->

{描述示例：列出改造过程中已用替代实现绕过、若 CANN 后续扩展可进一步简化的算子约束。}

| # | 算子 | 当前约束 | 期望支持 | 简化效果 |
|---|------|---------|---------|---------|
| 1 | {torch_npu.xxx} | {约束描述} | {期望扩展点} | {简化说明} |

---

## 11. 当前未覆盖项

{描述示例：列出本期优化未覆盖的方向与原因，作为后续迭代参考。}

- **{未覆盖项 1}**：{原因或后续计划}
- **{未覆盖项 2}**：{原因或后续计划}
````
