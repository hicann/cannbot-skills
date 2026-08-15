# 线性 Attention 参考资产地图

本文件用于 GDN/KDA/retention/RWKV 等线性 Attention Catlass 算子设计和实现前的参考定位。命中线性 Attention 路由时，不要要求用户在 prompt 中重复给出本机路径；用户没有显式给本地实现参考路径时，必须自动启用远程开源仓作为 primary reference。按需尝试 clone 到当前工作区；clone 失败不得阻塞生成，必须降级使用本文固化的开源规范摘要、远程搜索路径和 curated reference。

## Reference Source Policy（硬门禁）

Architect / Developer 必须先判定 `reference_source`，并写入 `OPEN_SOURCE_ALIGNMENT.md`：

| reference_source | 触发条件 | primary reference | 允许读取的辅助材料 |
|---|---|---|---|
| `USER_LOCAL` | 用户 prompt 明确给出本地实现参考路径（绝对路径或仓内相对路径） | 用户给出的本地实现参考路径 | 远程开源仓、仓内 Linear Attention shape 覆盖规则、既有 Catlass 经验 |
| `OPEN_SOURCE` | 用户未明确给本地实现参考路径 | 本文默认远程开源仓；clone 可用时读取源码，clone 不可用时读取仓内开源规范摘要 | 仓内 Linear Attention shape 覆盖规则、既有 Catlass 经验 |

判定规则：

- “本地实现参考路径”必须是用户需求文本里明确标注为“本地参考实现 / 实现参考 / source-of-truth / 按此实现或 pipeline 对齐”的路径；不能由 agent 自动扫描 `/workspace`、当前工作区之外目录或历史算子目录推断。
- 用户以“性能 baseline / evaluation baseline / 精度或性能评测 / 对比指标 / 使用 X 评测”给出的路径是 `evaluation_baseline`，只能用于评测指标、shape、报告字段和 baseline_status，禁止作为 implementation primary reference 或 pipeline 骨架。
- `OPEN_SOURCE` 模式下，如果需要源码细节，应尝试 `git clone` 默认远程仓到工作区可复现目录，例如 `tmp/open_source_refs/flash-linear-attention-npu`。clone 成功时记录 URL、commit/tag、clone 路径和 `clone_status=CLONED`；clone 失败时记录 `clone_status=UNAVAILABLE`、失败原因，并继续使用 [flash-linear-attention-npu-reference.md](flash-linear-attention-npu-reference.md) 的规范摘要、远程搜索路径和 curated reference。
- 仓内 Linear Attention shape 覆盖规则、mixed tolerance 精度规则和已有 Catlass 经验是 curated reference，只能提供 shape 覆盖维度、报告字段、工程风险和 Catlass 化经验；不能替代 primary reference。
- DESIGN.md / OPEN_SOURCE_ALIGNMENT.md 禁止把开发机绝对路径或未由用户指定的本地实现写成 primary reference。

## 仓内固化 Reference

| 类型 | Reference | 用途 |
|---|---|---|
| 开源 NPU 参考规范 | [flash-linear-attention-npu-reference.md](flash-linear-attention-npu-reference.md) | 通过 GitHub 远程仓库对齐 FLA/KDA/NPU 语义和工程规范 |
| Linear Attention shape 覆盖 | [../../../catlass-op-develop/references/shape-constraints.md](../../../catlass-op-develop/references/shape-constraints.md) Δ5 | 统一定义 GDN/KDA/retention/RWKV 等线性 Attention 类 shape 覆盖维度与 representative 子集约束 |
| 线性 Attention 设计路由 | [attention/linear-attention.md](attention/linear-attention.md) | full-flow/stage、dependency graph、workspace/flag、shape 覆盖 |
| GDN/KDA 专项设计 | [attention/gdn-kda.md](attention/gdn-kda.md) | KDA dAv stage、shape 覆盖、mixed tolerance 精度报告、varlen/partial chunk |
| A2/A3 stage 经验 | [../../../catlass-op-develop/references/patterns/a2-a3-linear-attention-stage-design.md](../../../catlass-op-develop/references/patterns/a2-a3-linear-attention-stage-design.md) | stage/window 调度、CrossCoreFlag、L1/L0/UB 复用 |
| 线性 Attention 实现经验 | [../../../catlass-op-develop/references/patterns/linear-attention.md](../../../catlass-op-develop/references/patterns/linear-attention.md) | KDA dAv varlen、同步、mixed tolerance 精度报告、性能复用 |

## 远程开源参考

默认远程仓库：

```text
https://github.com/flashserve/flash-linear-attention-npu
```

使用方式：

1. 优先读取 [flash-linear-attention-npu-reference.md](flash-linear-attention-npu-reference.md) 中的规范摘要。
2. 先按 Reference Source Policy 判定 `USER_LOCAL` 或 `OPEN_SOURCE`。
3. `OPEN_SOURCE`：若需要源码细节，在当前工作区内尝试 clone 默认远程仓；DESIGN.md 记录远程 URL、clone_status、commit/tag 或摘要版本、clone 路径或降级依据。
4. `USER_LOCAL`：读取用户显式给出的本地路径；DESIGN.md 仍需记录该路径的 git commit 或文件状态，并对照远程开源仓说明差异。
5. 不能记录开发机临时路径，不能把未由用户指定的本地路径作为生成前置条件。

## 读取顺序

1. 读 develop 侧 [shape-constraints.md](../../../catlass-op-develop/references/shape-constraints.md) Δ5，生成 Linear Attention 类 full coverage 或 representative shape 矩阵。
2. 读 [flash-linear-attention-npu-reference.md](flash-linear-attention-npu-reference.md)，对齐开源 NPU/KDA 接口和代码规范。
3. 读 [attention/linear-attention.md](attention/linear-attention.md)，按子场景继续读取 [attention/gdn-kda.md](attention/gdn-kda.md) 或 develop 侧实现 reference，完成 Catlass stage 设计。

DESIGN.md 中必须区分：

| 来源 | 可以继承 | 不能直接继承 |
|---|---|---|
| 远程 FLA/NPU 仓库 | 算法语义、mask、scale、layout、baseline 对齐、host/tiling 规范 | 本机路径；Triton `tl.program_id` 到 Catlass `blockIdx` 的直接映射；Triton `BLOCK_SIZE_M/N/K` 到 Catlass `L1TileShape` 的直接映射。两者 tile 语义、L1/L0 容量模型和 AIC/AIV 同步模型不同 |
| 仓内 Linear Attention shape 覆盖规则 | 覆盖矩阵维度、representative 子集下限 | 只取少量 smoke case 代表完整覆盖 |
| 仓内 mixed tolerance 精度规则 | 指标、阈值选择原则、报告字段 | 自创阈值或绕过 `ops-precision-standard` |
| 已生成 Catlass 经验 | 工程模板、脚本、报告格式、已知风险 | 未经重新验证的 tile 优化、临时调试分支；未获用户指定时不能作为 primary reference |
| 用户给出的 evaluation baseline | 评测指标、shape、baseline_status、报告字段 | 实现 pipeline、workspace/flag 协议、kernel 骨架 |

## User Contract Priority

线性 Attention 同族算子常存在多个相近但不完全等价的 contract，例如 `scale` 作用位置、`v` 与中间 `v_new`、gate/decay clamp、state 初值、mask 边界和输出 dtype。生成算子时必须遵循：

- 用户 prompt 中给出的数学公式和接口语义优先于任何参考实现。
- 参考实现只用于理解同族算法、layout、tiling、workspace、case 和 baseline 口径；不能把参考实现里的不同 contract 静默继承到目标算子。
- DESIGN.md 必须冻结本次 contract，并在 golden、verify、README、精度报告、性能报告中保持同一语义。
- 如果 open-source、仓内旧算子、用户公式三者不一致，不能自行择优；必须在 DESIGN.md 记录差异并以用户公式为准，除非用户明确改口。

## 通用生成经验：非 GEMM 节点不要让 Architect 阻塞

线性 Attention 类 stage 往往由 GEMM 节点和非 GEMM 节点交错组成。公开 Catlass GEMM/epilogue 组件不能覆盖所有 gate、decay、causal mask、prefix/scan、state update、finalize、layout 转换时，不应直接停在 architect；应继续设计可审查的 Catlass-style 自定义 Block/Tile。

| 节点类别 | 设计动作 |
|---|---|
| GEMM / batched GEMM / split-K GEMM | 使用 Catlass `Kernel` / `BlockMmad` / scheduler，不手写矩阵乘 |
| gate、decay、mask、clamp、cast、finalize | 封装成自定义 Block/Tile，写清输入输出、tile shape、UB 预算、API 依据 |
| scan/cumsum/state recurrence | 优先 stage 化，说明 dependency-based non-L0 exemption、workspace/flag 和同步边界 |
| layout/transpose/workspace 整理 | 优先物理连续布局或独立 stage，避免高频 scatter/gather |

自定义 Block/Tile 内部可以使用 Ascend C Vector API 和固定 tile 内循环表达逐元素逻辑，但必须是组件化、可复用、可审查的 device 主路径。禁止 host 侧完成真实计算，禁止 device kernel 入口为空，禁止在 op_kernel 顶层散写大段标量循环，禁止手写 GEMM。

## KDA dAv 已固化经验

### 语义与接口

- `dAv` 是 backward 的 stage operator，通常只覆盖 `dA` 与 `dV`，不要在设计里声称完整 KDA backward。
- 对齐开源接口时，重点核对 `B/T/H/HV/HK/K/V/BT/scale`、`cu_seqlens` 或 `chunk_indices`、GVA `HV/HK` 映射、输入输出 layout 和 case 命名。
- Triton/GPU/NPU evaluation baseline 某些 shape 不支持或运行失败是允许状态，报告中标记 `baseline_status=UNSUPPORTED/FAIL/MISSING`，不能把 baseline 失败误判为 custom 精度失败。

### Varlen 与 partial chunk

- Tiling 的 `nt`、task 数和 GM 偏移应来自真实 chunk 索引或 `cu_seqlens`，不能假设 `ceil(T/BT)` 覆盖所有 varlen。
- 区分“物理 matmul chunk 尺寸”和“有效输出行数”：先用完整 `BT` 物理块计算，用 `validRows` 做 causal/mask/writeback；只在单独 profile 证明收益后再收敛 matmul actual shape。
- KDA dAv 生成经验中，`chunkLen=validRows` 使长 varlen case 进入分钟级慢路径；恢复为 `validRows==0 ? 0 : BT` 后长 varlen case 回到秒级测试。后续优化必须把这种修改作为高风险 gated 分支。

### 同步与调度

- validated baseline 优先保持 `RunDa -> AIV signal -> AIV wait -> RunDv` 的依赖顺序；未证明资源和 flag 协议前，不要把 `RunDv` 提前与 AIV 后处理重叠。
- `V=256` 或 GVA case 先检查 split accumulation、L0C fixpipe 边界、HK/HV cache scope，再调 TileShape。
- 改 window 深度或 slot 复用前，必须设计 credit/free 或 overwrite 保护。

### 测试与报告

- 完整覆盖按 develop 侧 [shape-constraints.md](../../../catlass-op-develop/references/shape-constraints.md) Δ5 生成。smoke case 只用于功能门禁，不代表完整 shape 覆盖。
- 精度报告按 mixed tolerance 口径生成，至少记录 case、shape、dtype、输出名、atol、rtol、matched_ratio、max_abs 和 pass/fail。
- 性能测试可复用已有 inputs 和 bench，避免重复生成昂贵 golden。复用规则必须写在脚本帮助和报告中。

## 禁止

- 禁止在 skill、workflow、DESIGN.md 模板中写入开发机绝对路径。
- 禁止把开发机绝对路径作为必需参考；允许在用户显式给出本地路径时读取，但生成经验必须固化为远程 URL 或仓内 reference。
- 禁止把外部表格或外部评测文档作为运行时必需依赖。
