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

`workload_dev.jsonl` 共 `<30~40>` 条（用户用例不足 30 条时全部选入即可，不要求补齐），
文件内 `uuid` 顺序即用例顺序；`origin=user` 为
`<列出覆盖的用户用例>`，`origin=agent_supplemented` 为 `<列出补充用例及选取理由>`。
用户用例（`workload.jsonl` 中的 required case）不超过 40 条时全覆盖；超过 40 条时至少包含
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
