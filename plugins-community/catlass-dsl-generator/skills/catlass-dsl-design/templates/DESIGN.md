# CATLASS DSL 算子设计

> 本文档用于设计评审与 develop 参考。请用完整句子说明设计，让不了解内部格式的读者
> 也能理解；评测口径、产物路径和执行命令必须写清楚，便于后续按文本流程执行。

## 设计摘要

- **算子名称：** `<算子名称>`
- **算子类别：** `<matmul、reduction、elementwise、attention 等>`
- **目标：** `<这个算子解决什么问题，以及用户可以观察到的行为>`
- **版本：** `1`
- **目标 DSL 版本：** `<ascend-catlass-dsl X.Y.Z；默认 2.2.0>`
- **执行模式：** `<remote-detached | remote-connected | local>`
- **性能要求：** `<requested | none；只有用户明确要求性能、时延或加速比时才写 requested>`
- **状态：** 草稿

## 执行模式

执行模式在任务开始时确定，并写在上面的 `执行模式：` 行；`gate.py` 只读取这一行，不做
任何本地环境探测。三种模式的判定依据与推进上限如下：

| 模式 | 判定依据（任务开始时确认） | 推进上限 |
| --- | --- | --- |
| `remote-detached` | 本地生成，当前拿不到远端连接，也没有本机 NPU | 只能推进到 develop 的代码生成完成；focused test 记 `deferred` 并附精确重跑命令，之后必须先连上远端 |
| `remote-connected` | 本地生成，远端调试环境可连接（或已给出连接方式） | 全流程真实执行，不允许 `deferred` |
| `local` | 本机生成并在本机调试（本机有 NPU 与 CANN 环境） | 全流程真实执行，不允许 `deferred` |

`deferred` 只在 `remote-detached` 下、且只用于 develop 的 focused test，表示“代码已生成、
验证尚未执行”。full test #1 及之后的所有阶段都必须在真实环境执行后才能通过门禁；连接
恢复后按 `Command` 列命令补跑并把该行改为 `passed`。`design` 与 `finish` 永不推迟，
`gate.py finish` 见到任何 `deferred` 行都会直接 FAIL。

## 背景与范围

说明为什么需要该算子、它会用于什么场景，以及本次设计明确包含和不包含的内容。
实现必须使用 CATLASS Python DSL（`python/tla_dsl`），采用 Python 函数、
`@tla.kernel`、`tla.Tensor` 和已有的 `tla.*` API，不设计 C++/CUDA C++ CATLASS
class 或模板。

## 知识依据

记录查询条件、知识版本、实际读取并采用的路径、适用约束，以及检索覆盖不足时的重查结果。

## 输入与输出

用业务含义描述每个张量。shape 可以使用符号维度，但必须在正文中解释约束。

| 名称 | 方向 | Shape | Dtype | Layout | 含义 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入 | `<符号 shape 与约束>` | `<支持的 dtype>` | `<逻辑和物理布局>` | `<张量含义>` |
| `output` | 输出 | `<符号 shape>` | `<输出 dtype>` | `<输出布局>` | `<结果含义>` |

## 计算语义

写出数学定义或清晰的伪代码，并说明：

- 广播、空输入、尾块、对齐以及非法输入如何处理；
- 累加精度、舍入方式、容差以及 NaN/Inf 行为；
- `reference.py`、`definition.json`、`workload.jsonl`、`workload_dev.jsonl` 的相对路径和
  SHA-256。

## 实现方案

### 算法

描述准备采用的 Python DSL 算法、关键 `tla.*` API，以及仍需验证的假设。

### 融合范围与性能依据

默认 `fusion.mode=performance`，关闭全部 anti-hack；只有用户明确要求单个融合 kernel
时才设为 `single_kernel` 并启用严格校验。记录用户约束，比较数据复用、GM 读写、launch
开销、存储容量、同步成本和并行度，说明各 kernel 的计算范围、调用顺序及中间张量。
没有实测数据的性能判断必须标记为待验证假设。

### 数据流

按执行顺序说明各 kernel 的加载、计算、写回和中间数据依赖。默认允许多个 executor
及 GM 中间张量；严格模式要求所有融合阶段在一个 kernel 内完成。

### Tiling 与 Layout

说明初始 tile、线程/核映射、逻辑布局与物理布局策略，以及选择这些参数的理由。

搬运量核算（必填）：

- 总搬运字节 = Σ（各输入的重读次数 × 其体积），给出具体数字，并折算成带宽下界时间
- tiling 参数与参考样例不同时，逐项说明差异、理由及对搬运量的影响

### GEMM 骨架与优化阶梯（含 MMAD 时必填）

- 是否含 MMAD：`<是 / 否>`；包含的 matmul 阶段：`<QK / PV / dK / dV ...>`；纯 vector
  算子写“否”并说明为何不适用
- 知识条目：`knowledge/dsl/pattern-gemm-optimization-ladder`（须完整读取）
- 目标级别：`<第 3 级（默认）/ 第 2 级（须写理由）>`；理由：`<tiling、容量、同步复杂度>`
- 骨架验证 shape：`<主 matmul：M, N, K, dtype, batch>`；来源 dev 用例：`<uuid>`
- 全部 matmul 阶段清单（逐行）：`<名称 / 所在 kernel 或阶段 / 预计占比 / 各自最大 dev shape(M,N,K,dtype,batch) / 来源 uuid；每个都要各自跑一次骨架评测>`
- 骨架形态：`<剥掉 vector / epilogue 后的单 GEMM；与最终融合 kernel 的差异，以及融合后的真实占用由哪一步复核；骨架仅用于测量，不反推 kernel 划分>`
- 首版骨架必须包含：`<L1A/L1B 各 2 份、L0A/L0B 各 2 份、BK > SK、auto_sync 或等价 flag 结构>`
- 明确排除的反模式：`<单缓冲串行、BK == SK、多个 ready flag 共享同一 MTE1→CUBE 对>`
- skeleton 性能验收：默认判据、豁免与带宽受限规则见 `catlass-dsl-develop/SKILL.md` 的
  「skeleton 性能 DoD」；不达标但有可核对理由时记 `passed (waived)`，理由写入
  ITERATIONS；验收 shape 与来源 dev 用例同上

### 结构类型与参考样例

- 整体结构：`<阶段数 / Cube-Vector 交替方式 / 数据流 / 任务映射 / 同步方式>`
- 结构类型：`<Cube-only / Cube-Vector / Cube-Vector-Cube / Vector-only>`
- 主参考样例（整体结构，可为空）：`<Cube-only → tla/kernel/matmul/basic_mmad/basic_matmul.py；Cube-Vector → experimental/kernel/matmul/matmul_softmax_row_stats/matmul_softmax_row_stats.py；Cube-Vector-Cube → experimental/kernel/matmul/matmul_add_chain/matmul_add_chain.py；Vector-only → tla/kernel/vec/basic_vadd/basic_vadd.py、dsl_tutorials/vector_simd_ops/、dsl_tutorials/vector_simt_ops/；没有足够契合的样例时写“无”>`
- 整体结构匹配点 / 偏离点：`<阶段序列 / 数据流 / 任务映射 / 同步；偏离时写理由>`
- 任务映射与结论迁移：`<样例的任务分解 → 本 kernel 的任务分解；逐项说明哪些结论可迁移、
  哪些必须重算（并发度、数据复用、每块工作量、归约范围、跨核 / 跨引擎同步开销等），
  并给出核算值；设计期估算即可，后续用 profile 校正>`
- 局部参考样例（补机制或组合）：`<path + 机制，如 gather SIMT / atomic add / epilogue；或组合方案>`
- 组合后的整体结构（无主参考时必填）：`<阶段顺序 / 接口 / 同步 / 验证方式>`
- 不适用点：`<主参考与本算子的差异，以及准备怎么改>`

### 数值契约（所有算子必填）

先固定每个阶段/中间量的数值语义。单 kernel 内部的 C→V→C、epilogue、fixpipe 窄化同样要
逐行列出，不只看跨 kernel 边。

| 阶段 / 中间量 | 存储 dtype | 累加 dtype | 窄化或提升发生在哪一步 | 舍入 / 饱和 / 近似 |
|---|---|---|---|---|
| `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

标杆步骤映射（逐行覆盖 `reference.py` 的每个计算步骤，含 cast / scale / 归约 / scatter /
广播，不得跳步或合并；漏一步视为契约未完成）：

| reference 步骤 | 对应 kernel 阶段 | 存储 / 累加 dtype | 舍入与顺序 | 对齐结论 |
|---|---|---|---|---|
| `<...>` | `<...>` | `<...>` | `<...>` | `<一致 / deviation>` |

- reference 精度路径：`<输入 f16/bf16 是否先 .float() 再计算、结果按输出 dtype 还是 f32 转回；必须与 candidate 的转换点一一对应>`
- 累加 / 归约顺序：`<K 分块与 BK 循环顺序、是否 splitK、atomic add、分段归约、online softmax 或整行一趟；哪些会改变结果、误差量级、是否在容差内>`
- 舍入 / 饱和 / 近似：`<cast 的 round mode（如 RINT，支持情况随子包而异）、f32→f16 溢出处理、HF32、exp/rcp 近似；模式支持范围以目标 ascend-catlass-dsl 版本的包实际 API 能力为准，不支持时写替代方案，不得按“应该支持”假设>`
- 无效值与边界语义：`<padding 是否参与归约（sum 补 0 / max 补 -inf）、被 mask 的 lane、空输入、全 mask 行、NaN/Inf 传播>`
- 容差口径（按输出 dtype 写死）：`<max_atol / max_rtol / required_matched_ratio / max_error_cap；debug 期不得放宽，放宽必须改设计并记录>`
- 与标杆的对齐结论：`<逐项一致；或列出 deviation：哪一项没复现、为什么不可复现、量级或对照实验、影响用例、补偿计划、high risk；不得只写“容差内”>`

### 跨 kernel 阶段 oracle（多 kernel 时必填）

- 跨 kernel 边：`<from → to；中间量名称/路径>`
- 数学语义：`<...>`
- shape/stride/有效区/padding：`<...>`
- 输入/输出 dtype、累加 dtype、舍入位置：`<...>`
- 归约/重复 ID/chunk 顺序：`<...>`
- 消费分支：`<...>`
- 对应 oracle：`<reference 中间量 / 独立实现 / 替代验证>`
- 阶段检查计划：`<局部检查、真实链路、双向替换>`

### 内存与同步

说明 GM/L1/L0/UB 等内存空间、buffer、pipeline、flag、mutex 或 barrier 的使用方式。

## 正确性验证

列出必须通过的代表性、边界和安全用例。这里展示便于评审的输入和预期行为；
精确的机器输入保存在 `workload.jsonl`。

另外列出 `focused_cases` 和实际执行命令。没有专用 focused 命令时执行冻结的全量命令；
执行器应输出 argv、退出码、逐 case 状态和验证上下文摘要。覆盖全部 required cases 时，
该执行证据只用于跳过重复的 focused 执行；全量测试在 optimize 之前跑一次
（`full test #1`），优化采纳改动、kernel 变化时再补跑 `full test #2`，每次执行的记录都绑定
当时的 kernel；优化未采纳、kernel 未变化时 `full test #1` 可继续作为交付证据。

| 用例 | 类别 | 输入摘要 | 预期结果 |
| --- | --- | --- | --- |
| `correctness-main` | 正确性 | `<shape、dtype 和输入分布>` | `<与 Torch reference 的比较方式>` |

### Dev Workload 子集

`workload_dev.jsonl` 共 `<N>` 条（总数 ≤40；用户用例 ≥30 条时为 30~40 条，不足 30 条时
全部选入，不要求补齐；无用户用例时按覆盖需要自定，不以下限 30 为目标），
文件内 `uuid` 顺序即用例顺序；`origin=user` 为
`<列出用户提供的用例>`，`origin=agent_supplemented` 为 `<列出 agent 补充用例及选取理由>`。
用户用例（`workload.jsonl` 中 `origin=user` 的行）不超过 40 条时全覆盖；超过 40 条时至少包含
30 条用户用例。全部 optimize 轮次共用该子集，不每轮换样本。

## 性能目标

`性能要求：requested` 时，说明最终 benchmark 方法、采样参数、聚合指标和 candidate
profiling 原始数据的保存位置；默认结果记录 `anti_hack.status=disabled`，严格模式要求
`anti_hack.status=passed`，证明每次只 launch 一个声明的 kernel。如果有门槛，继续说明方向、
阈值、最大优化轮数和停止条件。

`性能要求：none`（用户没有明确提出性能、时延或加速比要求）时，明确写“本次不采集性能”，
当前 kernel 全量测试通过后直接进入 finish，不执行 benchmark 与 profiling；仍按流程完成
optimize 的三轮（或模式 A 一轮）记录，只是不采集交付性能数据。两种取值下，优化采纳改动
了 kernel 都必须在 optimize 结束后补跑 `full test #2`；全部未采纳、kernel 未变化时可复用
`full test #1`。

## 修改范围

列出允许修改的仓库相对路径，并用自然语言解释每个路径的用途。

- `<safe/repository-relative/operator.py>`：CATLASS Python DSL 算子实现。
- Solution 路径：完整源码包的仓库相对路径，默认与主实现文件同目录的 `solution.json`。

develop 一次生成全部 kernel、host 编排及完整 `solution.json`，随后才进行调试验证；
不进行任务拆分或逐任务上板。不涉及远程连接及文件同步实现。

## 风险与评审重点

说明 DSL 执行语义、同步、内存、layout、精度、公共接口或构建配置方面的风险，
以及 reviewer 应重点检查的内容。没有特殊风险时也应明确写出。

## 交付与证据

- 证据目录：`evidence/`（按阶段分子目录）
- 交付形式：工作树修改
- 知识收录：完成时批量准入

## 设计状态

- **状态：** 已确认（默认直通，无需用户批准）
- **确认时间：** `<生成时填写 RFC-3339 时间>`

设计发生变化时，在 `ITERATIONS.md` 记录一条“设计修订”（改了什么、为什么改、依据哪次
实测）；接口、reference 或评测口径发生变化时，重建 `reference.py` / `workload.jsonl` /
`workload_dev.jsonl`，更新 SHA-256 并重跑全部 reference cases。
