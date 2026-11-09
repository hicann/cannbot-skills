---
name: catlass-dsl-design
description: Use when specifying a new CATLASS Python DSL operator and producing its Torch oracle, definition, workloads, executable reference evidence, and frozen Markdown preliminary design
---

# CATLASS DSL Design（门禁引导）

> **入口**：本阶段开始前先加载 `catlass-dsl-workflow` 并输出阶段卡。
> 默认直通：产物齐备并自检通过后直接进入 develop，不等待用户批准；用户要求先评审时才暂停。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合。
> 仅当缺少只有用户能提供的信息、或可行路径已穷尽时才提问。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块；
> 编译独立 300s，单例运行 `timeout -k 5 60s`，批量分片，禁止整批套一次性超时，禁止裸跑。
> 超时或卡住立即终止，不得死等或加大超时。终止后先做卡健康判定：同一张卡的小探针或
> 官方算子能过 → 判本算子死锁并转 debug；探针也失败 → 判设备问题并换卡重试一次。

> **先生成、再推进**：本阶段先产出 `DESIGN.md`、`reference.py`、`definition.json`、
> `workload.jsonl`、`workload_dev.jsonl` 与校验脚本，再考虑执行。纯本地生成模式下本地
> 只写文件：reference 运行、静态检查、模拟/打桩检查全部留到真实环境（远端服务器或本机
> NPU 机器）；一开始就拿到服务器信息或本来就在服务器上，同样是先生成再推进。

本 Skill 负责算子规格、初步设计和 Torch 精度标杆，输出 `DESIGN.md`、`reference.py`、
`definition.json`、`workload.jsonl`、`workload_dev.jsonl` 与 reference 执行记录
（纯本地生成模式下是 `blocked` / `not_run` 记录加精确重跑命令）。
不实现 CATLASS DSL solution，不做性能测试或优化。

## 执行模式（开始即确定）

进入设计前先根据用户说明确定执行模式，并写入 `DESIGN.md` 的 `执行模式：` 行：

| 模式 | 判定依据 | 推进上限 |
|---|---|---|
| `remote-detached` | 本地生成，当前拿不到远端连接，也没有本机 NPU | 只能推进到 develop 的代码生成完成；focused test 记 `deferred` 并附精确重跑命令，之后必须先连上远端 |
| `remote-connected` | 本地生成，远端调试环境可连接（或已给出连接方式） | 全流程真实执行，不允许 `deferred` |
| `local` | 本地生成并在本机调试（本机有 NPU 与 CANN） | 全流程真实执行，不允许 `deferred` |

判定只依据用户说明与工作区已有声明，禁止在本地跑 `npu-smi`、导入 `torch_npu` 或做任何
环境探测。用户未说明且工作区没有声明时，先确认一次，不得自行猜测。`design` 阶段本身
不允许推迟；full test #1 及之后的所有阶段都必须在真实环境执行，门禁不会接受推迟。

## 性能要求（开始即确定）

同时确定是否采集性能，并写入 `DESIGN.md` 的 `性能要求：` 行：

| 取值 | 判定依据 | 后续流程 |
|---|---|---|
| `requested` | 用户明确提出性能目标、时延、加速比、吞吐或 profiling 要求 | optimize 后验收全量测试（kernel 变化时跑 full test #2），再执行 benchmark 与 profiling |
| `none` | 用户没有明确提性能，只要求功能正确或实现算子 | optimize 后验收全量测试（kernel 变化时跑 full test #2），直接进入 finish，不执行 benchmark 与 profiling |

默认取 `none`。只有用户原话里出现性能相关诉求才写 `requested`，不得以“顺便测一下”为由
自行升级；用户后续补充性能要求时，在 `ITERATIONS.md` 记录一条设计修订并更新该行。
两种情况都必须完成 optimize（模式 B 三轮 / 模式 A 一轮）；优化采纳改动了 kernel 时补跑
`full test #2`，全部未采纳、kernel 未变化时可复用 `full test #1`。

## 语言与实现边界

CATLASS DSL 是以 Python 语法编写的算子 DSL，不是 C++ 算子库。设计必须面向
`python/tla_dsl` 的 Python 接口和执行模型：

- 算子入口采用 `@tla.kernel`、Python 函数、`tla.Tensor` 和 `tla.*` API；
- 实现文件应为 `.py`，方案必须给出 Python DSL，而不是 C++、CUDA C++、Ascend C 或
  伪装成 DSL 的 C++ 模板代码；
- 不得设计 C++ class/template、头文件、namespace、宏、指针算术、RAII 或
  `<<<grid, block>>>` launch；
- GM/L1/L0/UB、tiling、layout、copy、MMAD、vector、flag 和 mutex 等概念必须映射到
  CATLASS Python DSL 的现有接口；
- C++、MLIR 或生成代码只能作为后端资料，不能成为算子公开接口或初步实现方案；
- 默认 `fusion.mode=performance`，按整体性能最大化确定融合范围，允许多个
  `@tla.kernel`、多个 executor、GM 中间张量及 shape/dtype/layout 分派；
- 只有用户明确要求单个融合 kernel 时设置 `fusion.mode=single_kernel`：一次 host `run`
  只 launch 一个 kernel，solution 只声明一个 `@tla.kernel`，所有编译期变体复用它，不得
  通过 GM 临时张量连接多个 executor；
- 每个 `@tla.kernel` 必须自包含：除形式参数、函数内局部值、Python 内建符号和受信任的
  `tla` API 命名空间外，不得读取模块级 tile、shape、dtype、params、开关或 closure 配置；
- 固定 tile/dtype/params 在 kernel 函数内定义；运行时 shape 通过显式 scalar/tensor
  参数、tiling data 或 tensor metadata 取得。

若目标仓库同时包含 C++ CATLASS 与 CATLASS DSL，必须先定位 Python DSL 样例和
`catlass.tla` 导入方式。没有真实 Python DSL API 支持的设计项标记为待验证假设。

## Torch 精度标杆

设计阶段必须在当前 workspace 生成项目专用 `reference.py`，只使用公开 `torch` API 表达
算子数学语义，不得 import candidate、复用 candidate 输出或调用待实现算子。reference 默认
在 CPU 上执行；只有语义确实依赖设备且 CPU 无等价表达时才使用目标 device。

### 执行分层（无 NPU 时）

| 层 | 内容 | 执行要求 |
|---|---|---|
| L1 静态 | Python 语法、`run` 入口、摘要与 solution 一致性 | 真实环境执行；纯本地生成模式推迟到真实环境 |
| L2 reference | Torch CPU 精度 oracle | 真实环境执行；纯本地生成模式推迟并记 `blocked` / `not_run` |
| L3 NPU | 编译、上板、精度、benchmark、profiling | 真实环境执行；缺环境记 `blocked` / `not_run` |

执行位置：有真实环境（远端服务器或本机 NPU 机器）就在该环境执行 L1/L2；纯本地生成模式
下本地一行都不跑，只生成文件并写好精确命令，等真实环境就绪后直接执行。本地不做静态
检查、模拟检查、打桩检查或任何形式的代跑。

L2 延迟口径：

- 在真实环境跑全 shape 算量过大时，先用小 shape（M≤8、V≤256、H≤64）的临时
  `definition` / `workload` 证明 `reference.py` 可执行；冒烟结果只作可执行性证据。
- 随后把冻结 workload 的全 shape 校验记录为 `blocked`，写明原因和精确重跑 argv，并在
  `DESIGN.md` 与 `ITERATIONS.md` 同步标注。
- 允许带 L2 `blocked`（纯本地生成模式记 `blocked` / `not_run`）进入 develop，但 `finish` 前必须
  补跑全部非 performance required cases；未补跑不得标记完成。
- 禁止为判断“能不能跑”做 CPU / NPU 算力或环境探测；环境缺失直接记录。

`reference.py` 必须导出与 `catlass-dsl-bench` 完全相同的 `run`：

```python
def run(*inputs):
    """Return one Tensor or a flat tuple/list ordered like definition.outputs."""
```

用例不在 `reference.py` 中声明。按 `catlass-dsl-bench` 的接口生成：

- `definition.json`：输入输出规格及完整 Torch `reference` 源码，`axes` 固定为空对象，
  `reference` 必须与同批生成的 `reference.py` 逐字一致；
- `workload.jsonl`：每行一个 required case，`uuid` 等于 `case_id`；可选 `origin`
  标记来源——用户明确提供的用例写 `user`，agent 自行补充的写 `agent_supplemented`，
  缺省按 `agent_supplemented` 处理；
- `solution.json` 由 develop 在实现完成后生成，不属于 design 输出。

case 规则：

- 不使用 axis binding：`definition.json` 的 `axes` 必须为 `{}`，workload 的 `axes`
  必须为 `null`；禁止用 axis 名称、`const`、`var` 或 `expr` 间接表达 shape；
- 每个 tensor input 在 workload 中直接写出数值 `shape` 和 `dtype`，标量写 `value`，
  可选空输入写 `null`；
- 每个 case 的 shape、dtype、layout selector、序列边界、布尔开关、标量参数和初始 state
  必须直接保存在该行；`uuid` 只标识用例，不得承担配置查表；
- 允许 `custom_inputs_entrypoint` 与 `type: custom` 生成特殊分布输入，但不得恢复 axis
  binding，也不得在 `reference.py` 维护 `_CASE_SPECS` 等 case table；
- 边界、尾块和广播尺寸分别用独立 workload 表达；随机输入固定 seed；
- tolerance 的字段语义、默认 `required_matched_ratio` 与 NaN/Inf 规则以
  [`catlass-dsl-bench`](../catlass-dsl-bench/SKILL.md) 为准；design 负责按累加精度与
  舍入分析给出具体 `max_atol` / `max_rtol` / `max_error_cap`，不能为通过用例放宽；
- definition/workload 不保存 solution 输出或伪造 expected。

### Dev Workload 子集

除全量 `workload.jsonl` 外，design 还必须冻结 `workload_dev.jsonl` 供 optimize 迭代：

- 条数：`workload_dev.jsonl` 总数不超过 40 条。用户用例（`workload.jsonl` 中
  `origin=user` 的行）不少于 30 条时冻结 **30~40** 条；不足 30 条时全部选入即可，
  不要求补齐；没有用户用例时按覆盖需要自定条数，不以下限 30 为目标；文件内 `uuid`
  顺序即用例顺序；
- 每行必须是对象，只允许 `uuid`、`inputs`（必填）和 `axes`、`tolerance`、`origin`、
  `source`（可选）；`uuid` 唯一且符合安全 ID 规则；
- 每行标注 `origin`：
  - `origin=user`：`uuid` 必须同时是 `workload.jsonl` 中的 required case，`source` 等于 `uuid`；
  - `origin=agent_supplemented`：不得声明 `source`，该行必须能真实执行；
- 用户用例不超过 40 条时子集必须全覆盖；超过 40 条时至少包含 30 条用户用例，
  其余用补充用例补齐；
- 子集必须覆盖代表性用例与用户点名/文档反复举例的用例。全部 optimize 轮次共用这份子集，
  不得每轮换样本。

### Reference 校验

参考校验（真实环境执行；纯本地生成模式只写脚本与命令，不运行）：

1. 在 `DESIGN.md` 的 `## Reference Validation` 小节写入 `reference.py`、
   `definition.json`、`workload.jsonl` 的相对路径与 SHA-256；
2. 自行编写一次性校验脚本（例如 `tools/validate_reference.py`），导入 `reference.py`，
   按 workload 生成输入、执行 `run`、按 tolerance 比对，输出逐 case 结果 JSON；
3. 在真实环境执行该脚本，把命令、退出码、通过数和证据路径追加到 `ITERATIONS.md`；
   纯本地生成模式记 `blocked` / `not_run` 并附精确 argv，不在本地运行；
4. 任一输入文件变化后重新运行全部 case，更新摘要与 revision，不得只改摘要。

三个输入文件的摘要和用例集一旦冻结即不再变更；新增或修改 `DESIGN.md` 的实现方案
不需要重新冻结 reference（见下方“设计是工作草稿”）。

## 过程

1. 只读探索目标仓库，确认 `python/tla_dsl` 来源、导入方式、已有 `.py` 算子模式、
   构建/测试入口和参考实现。用户指定标杆文件并禁止参考兄弟目录时，只读标杆文件、
   用例文件与 catlass 参考源码。
2. 明确算子名称、family、用途和可观察行为。按 `catlass-dsl-knowledge` 检索相关条目，
   把查询关键词、采用的条目路径与适用约束写入 `DESIGN.md`。
3. 固定每个输入输出的 symbolic shape、dtype、layout 和语义。
4. 写清数学计算、广播/尾块/空输入等边界行为，以及累加精度、舍入和 NaN/Inf 规则。
5. 生成 `reference.py` 与 `definition.json`，把每个非 performance required case 写成
   `workload.jsonl` 一行，并按规则生成 `workload_dev.jsonl`。
6. 按 L2 口径在真实环境运行 reference cases 并记录证据；全 shape 不可完成时先冒烟
   再记 `blocked`。纯本地生成模式不在本地运行，只写命令并记 `blocked` / `not_run`。
7. 使用 CATLASS Python DSL API 给出融合范围、各 kernel 职责与调用顺序、中间张量、
   tiling/layout、内存空间、pipeline 和同步方案，解释性能选择依据。tiling 块数按核数
   取整，行块对齐 cube fractal（16 行），依据见 `dsl/pattern-gemm-optimization-ladder`。
   同时给出搬运量核算（Σ 重读次数 × 体积）与对应的带宽下界，tiling 参数偏离参考样例时
   逐项说明差异及对搬运量的影响。
   写明 `fusion.mode` 及单融合是否由用户明确提出；没有实测支持的性能判断标记为待验证
   假设。
8. 明确正确性 oracle、required cases、`focused_cases`、focused/full test 命令、风险、
   允许路径、benchmark/profiling 命令和可选性能门槛。要求执行器输出实际 argv、退出码
   和逐 case 结果。
9. 从 [`templates/DESIGN.md`](templates/DESIGN.md) 创建面向用户的设计文档，写入目标 DSL
   版本与执行模式，并在 `ITERATIONS.md` 记录 design 里程碑。
10. 向用户展示 `DESIGN.md`、Resolved Plan 与 reference case 结果（纯本地生成模式
    展示 `blocked` / `not_run` 状态与精确命令）；默认直通时直接进入 develop，用户要求
    先评审时才暂停。

初步设计不是已验证实现结论，不得虚构 benchmark 或 profiling 数据。涉及 DSL 执行语义、
同步、内存、layout、精度 oracle、公共接口或构建配置时，`risk_level` 必须为 `high` 并
填写 targeted review focus。

**设计是工作草稿，不是合同。** 评测口径（allowed paths、required cases、benchmark
命令、融合模式、anti-hack 策略）冻结后不再修改；但 `DESIGN.md` 中的实现方案（tiling、
数据流、buffer 规划、同步策略、kernel 划分）允许在 develop 阶段随实测修改：

- 修改后在 `ITERATIONS.md` 记录一条“设计修订”（改了什么、为什么改、依据哪次实测）；
- 只有最终交付时才把 `DESIGN.md` 固化为只读快照。

设计应当在“写→跑→看数”的循环中收敛，而不是在动笔前一次性消灭所有不确定性。

冻结前再次检查：算子主体和 reference 都是 Python 文件；算法引用的 `tla.*` 接口可以从
本地源码或知识条目定位；标准 definition/workload 覆盖全部非 performance required cases
（`blocked` 项附原因与精确 argv）；正文包含三个文件的实际 SHA-256；没有把 C++ CATLASS
API 当作 DSL API；每个 decorated kernel 自包含；融合边界有明确性能依据。

## 输出

```text
DESIGN.md
reference.py
definition.json
workload.jsonl
workload_dev.jsonl
evidence/reference-cases.json      # 自行生成的一次性校验结果
```

`DESIGN.md` 必须冻结 benchmark 命令、metric JSON path，并设置 profiling 要求。
声明性能门槛时还必须写明方向、阈值、迭代上限、停滞阈值和最小提升比例。门槛只能来自
用户提示或用户确认；由知识条目、基线实现或 roofline 推算出的数值只是参考目标，不得
写成本设计的门槛。没有性能门槛不表示跳过 optimize：无门槛时进入模式 B（explore），
至少执行三轮优化；也不表示跳过 benchmark 或 profiling。
