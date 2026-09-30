# TileLang Flash 性能优化工作流

源码位置与产物目录遵循 [Ascend950 路径约定](../../../references/source-layout.md)。本阶段沿用并向后续技能/角色传递 `{ops_tilelang_repo}=OPS_TILELANG_DIR` 和 `{tilelang_repo}=TILELANG_DIR`；两者由插件源码准备阶段确定，不由用户另行指定。


该模式默认初始算子精度通过，仅在优化完成后进行一次全量精度测试，当前主 agent 直接完成分析、实现和验证，不调用性能分析或方案实施 subagent。流程不限制时长、轮数和 case 数，以证据收敛为停止依据。

## 输入与产出

必需输入：入口选定的 `{backend}`/`{tilelang_target}`、工程根目录、目标算子文件、现有精度 pytest，以及用户直接输入的性能 case 完整参数。缺失或不完整时只询问用户补充；禁止生成候选表、推导 case，确认前禁止创建调优目录、编译运行、profiling 或修改代码。

全流程显式使用 `TILELANG_DEFAULT_TARGET={tilelang_target}`：PTO 为 `pto`，AscendC 为 `ascend`。禁止默认 PTO、静默回退或跨后端比较。

Case 输入确认后，在 `operators/<算子名>[_时间戳]/` 中创建隔离的优化目录，并按下述统一复制范围完成首次源码复制。优化目录保留：

- `working/`：当前最佳实现；
- `candidates/Ci/`：从 `working/` 创建的隔离候选；
- `profiling/`：基线和迭代性能数据；
- `optimization-search-coverage.json`：逐 case 结构义务与候选事件；
- `final_optimized/`：最终验证通过的代码；
- `flash-report.md`：精简过程与结论。

不修改原工程。记录原始源码和 pytest SHA256，禁止修改 reference、精度阈值或测试 case 以获得通过。
开始执行时创建 `timing.json` 并记录墙钟起点；基线、每个候选和最终验收分别记录开始、结束与 elapsed，结束时写总墙钟时间并在最终报告汇总。耗时只用于复盘，不作为默认停止条件或硬预算。

### 统一复制范围

复制范围固定为能够在优化目录内运行和调试目标算子的最小文件集：

- 目标算子文件和现有精度 pytest；
- 二者直接或间接导入的仓内模块；
- 上述模块路径上的必要 `__init__.py`、`conftest.py`；
- 必要的 pytest/工程配置，如 `pytest.ini`、`pyproject.toml`、`setup.cfg`。

禁止复制无关算子、整个算子源码目录或整个工程根目录。若输出目录位于工程根目录内，必须排除输出目录，避免递归复制。首次复制形成文件清单；后续创建 `Ci` 和归档时直接复用该清单，不再重新扩大复制范围。只有编译或测试证明缺少仓内依赖时才补充清单并记录原因。

## 事实优先级

1. 当前算子、pytest、公开接口与 dispatch；
2. 本轮编译、精度和 `tilelang-op-profiling` 实测；
3. 实际导入的 TileLang 源码、所选后端 lowering 及仓库同类实现；
4. `tilelang-performance-best-practices` 等 skill 资料。

低优先级经验不得覆盖高优先级实测。skill 用于提出候选，不直接证明收益。

## 默认实现合约（临时单 kernel 门禁）

除非用户明确允许多 kernel，目标算子默认满足以下约束：

- 一个目标 factory 内只定义一个承担该算子工作的嵌套 `@T.prim_func`，全部 case 解析为同一个逻辑运行时 kernel；禁止把快路径、慢路径或不同 shape 拆成多个 primfunc 后在 host 侧 dispatch。
- 禁止仅为命中已选性能 case 而增加与某个精确 shape、计数或属性值相等的性能分支。允许在同一 primfunc 内依据接口语义和通用性质分支，例如容量、对齐、dtype、任务数、整 tile/尾 tile、连续性或硬件资源上限。
- Python factory 参数和 `T.macro` 可以用于组织代码；macro 展开后仍属于同一个 kernel，不能被用来绕过上述约束。
- 不同 shape 可使用同一 kernel 定义的少量 JIT 参数组合，例如不同 tile 或核数；launcher 只做简短的配置选择，不复制算法实现。
- 建立基线时记录目标 primfunc 数、dispatch 条件和运行时 kernel 名。每个候选 profiling 前重新做源码/AST 审计并核对运行时 kernel 名；违反约束的候选直接淘汰，不以性能收益晋升。

该门禁限制的是 case 特化和多 kernel dispatch，不禁止同一 kernel 内基于通用可证明条件选择不同执行路径。

## 状态

- `B0`：不可变的原始基线；
- `B*`：当前已验证的最佳版本，初始为 `B0`；
- `Ci`：从 `B*` 生成的本轮候选。

每轮同时计算：

```text
增量加速比 = B* kernel_time / Ci kernel_time
累计加速比 = B0 kernel_time / Ci kernel_time
```

## 工作流

### 0. Case 输入门禁

1. 校验用户输入的 case 参数；缺失时询问并停止。
2. 禁止生成候选表，禁止从 pytest 或 benchmark 自动选择、补充或分组。
3. 用户确认后锁定清单并记录 SHA256；变更须重新确认。未通过门禁不得继续。

### 0.5 历史证据与已验证候选复用门禁

建立基线前，先加载名为 `tilelang-performance-best-practices` 的 Skill，并传入平台 `Ascend950`；再只读检查当前 git 历史、同算子已有 `operators/` 产物、当前工作区候选，以及该 Skill 中的 `VERIFIED`/`PRODUCTION_REFERENCE` 记录，避免重复探索已经实测过的结构。

`PRODUCTION_REFERENCE` 必须用对应文档的特征指纹核对当前 kernel 与 launcher；缺失核心结构时将本地文件视为 baseline，不得仅凭路径或状态表继承性能结论。

特征指纹至少核对数据驻留生命周期、local/GM 布局、中间物化、同步点、tile/core、静态尾块、buffer 版本和 launcher JIT 参数。“融合、复用、合并 partial、流水”等语义标签须展开到物理 buffer、GM copy、归约 lowering 和版本配置；名称相同不代表结构已复现。各优化点按当前瓶颈独立排序；只复现部分特征时不得继承整组性能证据。

历史候选只有在公开接口、目标 case、测试口径、TileLang 与所选后端版本、设备架构均兼容时，才能进入本轮候选池；任一关键项不兼容时只能作为设计线索。必须记录候选来源、源码 SHA、继承的性能证据和不兼容项，不能把继承成果描述成从 B0 独立发现。

兼容的历史候选与新候选统一按瓶颈匹配度、收益上界、证据成熟度和验证成本排序，不自动优先，也不强制整组复现。历史结果不能直接晋升为 `B*`，仍须按本工作流完成当前环境的编译、精度与同口径 profiling 复测。

### 1. 建立基线

1. 确认 Case 输入门禁已通过，再阅读目标算子、pytest、调用链、同类实现、实际导入的 TileLang 源码和所选后端 lowering。
2. 确认基线能够正确编译并运行；默认初始算子精度已通过，不重复执行全量精度测试。
3. 在提出性能假设前按名称加载 `npu-arch` Skill 并传入平台 `Ascend950`，复用入口的完整硬件探测证据，记录目标架构、物理 UB/L1/L0 和核数，并读取 `tilelang-performance-best-practices` 的 `references/common/hardware_resource_discovery.md`，从实际 TileLang lowering 核对最终 kernel body 的执行域预留，建立显式/常驻/多版本 footprint 账本。增加或移除 `SimtVF`、buffer version、padding、LUT/index 或临时布局后都要重算；禁止用没有来源的固定 cap 代替有效容量。
4. 建立源码语义成本模型：列出每个输出表达式依赖的逻辑维度（如 layer、token、prefix、table、channel），统计每个有效元素或 token 的整数乘除/取模、XOR、cast、分支和索引计算，检查循环不变量、跨输出维度的公共子表达式、重复搬运、中间值生命周期及每线程 live state。必须量化外提或复用后可消除的逻辑工作量；这只是候选收益上界，不是实测加速比。
   若考虑多级流水，还必须画出相邻迭代间每个可变 UB/L1 buffer 的 `CopyIn → Compute → CopyOut` 读写依赖，标明版本数、每核有效迭代数、满波范围和尾部范围。只给输入加版本而遗漏仍被异步读取的输出或临时 buffer，不得进入实施。
   对 Elementwise、gather/scatter 或布局转换，同时读取 `tilelang-performance-best-practices` 的 `references/elementwise/tiling_task_vector_search.md`：画出外层 item/内层 chunk 工作树，枚举完整连续单元、SIMD、DMA、容量和并行边界；按固定有效输出量比较可实施数据流族的连续/离散访存、寄存器操作、临时物化和 lane 利用率，不预设两个具体 API 路径为固定答案。
5. 从 `B0` 最新 kernel 与 launcher 生成独立 profiling 入口，内嵌全部目标 case；host 入口复刻 launcher 的参数处理和调用口径，但调用本地同一 kernel，并校验两者 SHA、实际参数、case 和运行时 kernel 名。
6. 先加载名为 `tilelang-op-profiling` 的 Skill，并传入平台 `Ascend950`；对全部目标 case 按该 Skill 采集普通指标并生成 `summary.txt`。
7. 将语义成本模型与实测交叉验证，判断计算、搬运、标量/调度、并行度、启动或混合瓶颈。不得仅凭某条 pipe ratio 直接认定某个源码操作是主瓶颈；具体操作归因须有源码工作量、生成 IR/指令或进一步 profiling 证据。
8. 首次修改前完整阅读 [性能优化搜索覆盖门禁](optimization-search-coverage/optimization_search_coverage_gate.md)，参考 [开始模板](optimization-search-coverage/optimization_search_coverage.example.json) 创建 `{output_dir}/optimization-search-coverage.json`（schema v3）：逐 case 记录结构事实、目标、下界和待裁决义务；用 `candidate_events` 追加候选的 `CREATED`/`RESULT`，候选身份和适用 case 不得中途改义。将本文件真实目录的上一级（`Ascend950/`）设为 `WORKFLOW_DIR`，运行 `python3 "$WORKFLOW_DIR/scripts/validate_optimization_search_coverage.py" "{output_dir}/optimization-search-coverage.json"` 通过后才能创建候选。

### 1.5 候选池与收益排序

第一次修改前生成最多四个互相可区分的 active candidates，后续每轮依据新数据更新执行队列。候选同时来自当前源码与 profiling 分析、已读取的 skill 结构、兼容历史证据和已核验同类实现；来源不同不分池，也不决定晋升。优先覆盖与当前瓶颈匹配的算法级重复计算消除、数据驻留与任务映射、Tiling/流水，以及有生成代码证据的算术 lowering；不要求每类都凑数。四项上限只限制当前执行队列，不限制 coverage obligations；待办不能因降出前四、排序变化或连续低收益而消失。

对适用的 Skill 或历史候选，在提出时和实现后都核对源码、buffer/copy、lowering 与 launcher；只有物理结构匹配才算已实施。上述方向是候选示例而非封闭清单，不能因命中历史结构或现有模板就停止探索。

以 `optimization-search-coverage.json` 的 append-only `candidate_events` 作为闭环 ledger；可另建便于阅读的候选索引，但不得作为第二套状态真源。每个初始或后续重开的高收益候选须有 `CREATED` 和对应 `RESULT`，结果按门禁记录为 `PROMOTED`、`REJECTED_WITH_EVIDENCE`、`INAPPLICABLE_WITH_EVIDENCE` 或 `BLOCKED_BY_ENVIRONMENT`；重排候选池不能删除条目。未达目标时，未裁决义务不能被候选排序变化或环境故障伪装为已关闭。

候选事件是每轮必须落盘的执行记录，不是最终补写的报告。候选创建时固定 `(work granularity, task mapping, physical dataflow, precision, storage/pipeline, tail strategy)` 六轴身份、逐 case 粒度事实、结构化物理 route、pipeline facts 和 applicable cases，并追加 `CREATED`。创建 `Ci+1` 前，先追加同 identity/applicable cases 的 `RESULT`，逐 case 记录修改路径是否真正执行，再记录源码 SHA、编译/精度/性能结果、瓶颈转移、状态及下一步排序并运行校验器。任一轴、粒度值、route kind 或适用路径实质变化都分配新 ID；不得复用旧 ID 改义或结束时凭记忆批量回填。

对每个候选记录：可证伪假设、适用 case、预计消除的逻辑工作量、性能收益上界、实现/精度风险、TileLang 与所选后端依据、验证成本和回退点。根据用户验收指标计算当前目标差距；例如指标与耗时成反比且数据量固定时，计算达到目标所需的最大耗时和剩余加速比。预计收益上界不足以覆盖明显目标差距的候选只能作为诊断项，不得连续优先于能够覆盖差距的候选。

候选按“与实测瓶颈匹配度、可消除工作量、覆盖目标差距的可能性、复用证据成熟度、实施与验证成本”排序。若多个源码表达预计语义等价，先比较所选后端的 lowering、生成 IR 或关键指令结构；没有实质差异时合并为一个候选，禁止仅靠改写拼写消耗多轮上板验证。

对 Elementwise、gather/scatter 或布局转换，候选必须按 `tile/group × 任务映射 × Vector 数据流 × 精度 × 流水` 分解。若一个轴改变了另一轴的连续 payload、固定成本摊销或主瓶颈，将其组合新增或重开为高收益候选。单独的 tiling 失败和单独的 SIMD 失败不能代替二者组合的实测/不可实施证据；具体裁决按 tiling/task/Vector 指南的组合矩阵执行。

Vector 数据流候选按从源到结果的完整物理路线覆盖，而不是按源码名称或最后算术所在执行域凑数。先物化完整 planar/transpose/scratch 再做 Vector 算术属于 materialized transform，不能关闭“原始连续窗口直接 load + register select/shuffle/pack”路线。目标未达到时，route plan 中仍与瓶颈相关的物理族要逐 case 实测或给出确定证据；只要仍存在能减少逐 lane UB/GM 访问、完整物化遍数或显著提高 active-lane 比例的未裁决路线，就必须加入或重开候选事件。

路线候选必须采用对应物理族中当前 API/lowering 有依据的最低成本代表，并在记录中列出 dtype/lane/part、隐式或显式扩宽、select/shuffle/interleave/pack、算术和输出 store/scatter 链。带有尚可消除的多次扩宽、lane 修复或中间物化的首次实现失败，只淘汰该完整链，不能关闭整个物理族；按 Vector 指南标记并重开代表候选。跨路线比较尽量固定 tile/group、任务映射、精度、输出路径和 stage 数，无法固定时不得把整体候选耗时归因给某一条 intrinsic。

对每个候选在结构化物理路线中写 `route_experiment`。若 `source_window.register_window_feasible=true`，且 direct route 未由基线或确定 API/lowering/同 payload 成本证据关闭，则第一个 `route_experiment=true` 的候选必须是 `CONTIGUOUS_LOAD_REGISTER_REORDER`；仅改变其他轴并继承父版本物理路线的候选不受此顺序限制。实现前按 Vector 指南的 `vld + vselr` 模板核对 index 作用域、dtype、predicate 与跨寄存器分段，禁止因为 gather 示例更熟悉而改变顺序。

工作粒度候选用实际来源值裁决完整连续单元、SIMD、DMA、容量和并行边界；实测 pair 或任意小 group 只关闭该数值。若同一 case 同时有粒度和物理数据流义务，必须由同一候选实测 `FULL_CONTIGUOUS_UNIT × CONTIGUOUS_LOAD_REGISTER_REORDER`，或给出准确不可实施证据。两个轴各自的失败不能替代组合。

tile、核数、stage、静态尾块和 buffer version 相互影响时，将其视为配置族，在同一 kernel 定义下筛选少量有依据的组合；先用同口径性能数据比较，再对胜出组合完整验证。单一组合无收益只淘汰该组合，不能淘汰整个结构方向。

流水候选必须先完整阅读 `tilelang-performance-best-practices` 的 `references/elementwise/double_buffer_design.md`，并明确回答：自动还是手动多版本、为何 lowering 能识别、哪些 buffer 需要版本化、控制流如何暴露稳定的满波流水、尾部如何处理、预计可重叠的流水及生效/失败判据。回答不完整时只能作为诊断实验。

流水一旦因迭代数、容量和 MTE/Compute 可重叠性而准入，自动与手动实现构成完整决策链，不再依赖数值性能目标。若自动 buffer versions 不 eligible、extent/别名 lowering 错误、动态 stage intrinsic 失败、仅部分存活 buffer 被版本化或 latency/overlap 未生效，只否决准确的自动组合；必须继续裁决显式输入/输出/临时 storage、手动 annotation、无条件 full-wave、静态 stage body（若需要）、`T.Pipelined` 与同结构 stage-1 配对。只有自动组合完整生效，或手动路径有当前 API/lowering/容量/依赖的确定不可实施证据，才可关闭流水义务。

### 2. 单假设迭代

每轮只验证一个主要假设；存在明确数据流、布局或 lowering 依赖时，可将必要改动组成一个候选，并注明依赖关系，不得混入无关优化：

1. 从候选池选择当前排序最高的假设，按既有文件清单从当前 `B*` 对应的 `working/` 创建 `candidates/Ci/`；禁止从未晋升或已退化的候选继续叠加，也禁止直接修改 `working/`。组合候选同样从 `B*` 独立创建。
2. 编译并验证本轮调优对应的 shape；失败则修复或淘汰，禁止放宽精度。迭代期间不重复运行完整测试套。
3. 从 `Ci` 最新 kernel 与 launcher 重新生成并校验 profiling 入口，按已加载的 `tilelang-op-profiling` Skill Step 2～3 采集和归档本轮数据。`msprof op` 若仅因插桩/导出失败，在同一最新入口直调正常后按该 Skill Step 2.5 使用普通 `msprof`；fallback 未完成前不得淘汰候选或宣布收敛。结果异常、方向不明或接近噪声时，再做完整 CSV 与 PipeTimeline 分析。
4. 在相同设备、输入、warmup、launch count、并发和计时口径下比较 `Ci`、`B*` 和 `B0`。不得使用 host 计时替代 kernel 数据。
5. 读取新 `summary.txt` 和必要原始数据，确认总时间、原瓶颈、核间均衡是否改善，以及是否引入新瓶颈；据此决定晋升、淘汰或下一个假设，并更新其余候选的排序与收益上界。
6. 在创建下一个候选目录前更新 `optimization-search-coverage.json` 并通过普通校验；若瓶颈转移使其他正交轴产生新的组合收益，同时新增/重开该组合义务与候选并重排序。工作粒度、任务映射或物理数据流改变后，旧结构上的流水结论不继承；对仍有可重叠迭代的 case 重新执行流水准入。

pytest benchmark 只用于快速筛选，不能替代步骤 3～5 支持新的瓶颈归因。目标仍未达到时，将剩余路径区分为每次 launch/core 的固定段与随 tile/element 重复的热循环；须以跨 case 绝对耗时、timeline、lowering 或关键指令确认归因后再修改对应部分。

性能目标仍未达到时，归因调度或硬件极限须有 timeline、lowering 或关键指令证据；不得仅凭 wave 数、pipe ratio 或 Task/block 时间差下结论。

流水候选还必须同时检查生成代码/IR 中的版本切换与同步结构，以及同口径 kernel latency、有效带宽和相关 pipe overlap。出现两个 UB buffer、编译成功、运行成功或 `num_stages > 1` 均不能单独证明流水已生效；若 latency 与 overlap 都未超过噪声改善，应判为未生效并继续检查控制流和依赖，而不是保留形式上的双缓冲。

每轮 profiling 前执行“默认实现合约”审计，并把 primfunc 数、通用分支依据和解析出的目标 kernel 名写入候选记录。

### 3. 晋升判定

先重复采集同一版本估计当前环境噪声。候选差异接近噪声时，交替复测 `B* → Ci → B* → Ci`，使用稳定均值或中位数判断。

仅当以下条件同时满足时令 `B* = Ci`：

- 编译和本轮调优 shape 的精度门禁通过；
- 汇总性能相对 `B*` 的收益超过测量噪声和必要安全余量；
- 没有不可接受的单 case 或未修改路径退化；
- 收益可复现且命中目标 kernel。
- `B*` 与 `Ci` 的全部目标 case 都完成同口径 `tilelang-op-profiling`；pytest benchmark 不能单独支持晋升。

默认以全部目标 case 的几何平均评估总体收益，同时报告最差 case 和退化 case。用户指定权重或性能目标时以用户目标为准。存在无法裁决的 case 取舍时不覆盖 `B*`，保留为候选并报告。

候选未晋升时恢复 `B*`，记录假设、源码 SHA、测试、性能结果和失败原因；不得在无明确依赖的退化版本上继续叠加修改。

候选晋升时才按既有文件清单用 `candidates/Ci/` 更新 `working/`；未晋升候选保留在自身目录，不改变 `working/`。相关输入/复用、accumulator/归约、partial/copy/host、索引位置、tile/core/stage、尾块或 buffer 生命周期改变后，旧负向结论只否决原组合，相关方向须按新上下文重新判断。

涉及多版本 buffer、流水同步、持久化调度或编译器 warning 的候选，在晋升前必须在同一进程覆盖全部目标 case，并以正式 warm-up/repeat 至少复测一次；不同 case 会经过不同通用路径时，再用逆序或交错顺序复测。warning 本身既不自动否决也不自动放行：结果稳定且精度、性能均可复现时记录后晋升；若耗时随 case 顺序异常漂移、偶发错误或跨次运行不稳定，则淘汰候选并保留诊断证据。

### 3.5 搜索升级与去重

- 一个算术或语法 lowering 候选未产生超过噪声的收益后，除非生成 IR/关键指令证明另一写法实质不同，否则停止该类等价改写。
- 连续两个候选收益均低于 3%，且仍未达到目标时，暂停继续实现，重新检查输出依赖维度、重复计算、每线程 live state、任务映射和历史已验证结构；下一候选必须来自有更高收益上界的算法复用、数据流或并行结构，或明确报告不存在这类可实施候选。
- 当前目标仍需超过 1.2 倍加速时，不得连续实施只能解释为低个位数收益的局部候选，除非它是验证后续高收益方案的必要前置。
- 每次重排候选池都保留已淘汰假设和生成代码证据，禁止换一种源码拼写重复测试同一 lowering。
- 若某慢 case 的外层 item 少于可用核数，必须先裁决内层独立 round/chunk 展平候选；保留串行 fallback 不构成已优化证据。
- 若任务粒度变化使连续 payload、源窗口密度、固定成本摊销或主瓶颈变化，必须新建与物理数据流/流水的交互义务；原粒度上的数据流失败、或新粒度配原数据流失败，均不能替代组合裁决。
- 以“严格受支配”关闭候选时，须附同一有效 payload 下的具体指令、访存层级、转换、物化、同步与 lane 成本；仅凭代码复杂或推测指令更多不能关闭高收益义务。

### 4. 停止条件

满足任一条件后停止迭代：

- 达到用户性能目标或用户明确给出的资源预算；
- 已接近可核验的硬件理论下界；
- 连续候选未产生超过噪声的收益，且没有证据更强的新假设；
- 剩余方向缺少当前所选后端的可实施依据；
- 继续优化会破坏精度、覆盖范围或造成不可接受退化。

停止前按 [搜索覆盖门禁](optimization-search-coverage/optimization_search_coverage_gate.md) 运行 `--final`。全部 case 达到用户目标时须取得 `TARGET_MET`；尚未达标时普通 `--final` 的 `SEARCH_INCOMPLETE` 不是停止授权，应继续搜索。只有逐 case 结构覆盖和可核验下界均已闭环，才能用 `--final --allow-unmet-convergence` 取得 `UNMET_BUT_CONVERGED` 并声明证据收敛。设备、profiler 或工具链故障只记 `BLOCKED`，不能作为结构性关闭证据；用户明确的预算耗尽或环境阻塞应如实报告未完成状态，不得称为收敛。几何平均或快 case 不能掩盖慢 case；声称热路径已是最小序列时，还须给出最低语义操作、生成指令/访存计数、相关 API 搜索范围及其他路线裁决。

用户目标未达到时，报告必须列出尚未实测的高收益候选及未实施原因；未达标或环境阻塞不能报告为目标完成。

用户未给数值目标时，所有 case 的 `performance_target` 填 `null`，完成结构覆盖和 `convergence_evidence.case_bounds` 后运行 `--final`，须取得 `CONVERGED_WITHOUT_NUMERIC_TARGET`；不得为通过门禁自行设定性能阈值。

### 5. 最终验收

1. 全部性能调优结束后，对 `B*` 执行 `env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET={tilelang_target} python -m pytest {test_file}`，完成一次完整目标测试套；设备访问和并发遵循工作流入口约定；失败则最终验收不通过，不交付该候选。
2. 从 `B0` 与 `B*` 最新源码分别生成 profiling 入口，对全部目标 case 按已加载的 `tilelang-op-profiling` Skill 同口径复测；环境漂移时重新采集基线。
3. 相关完整测试套全绿且最终实测优于 `B0` 时，按“统一复制范围”的既有文件清单归档 `B*` 到 `final_optimized/`；否则按同一清单归档基线或仅保留未完成候选。
4. 生成 `flash-report.md`：状态、后端、环境与 SHA、全部 case 最终对比、最差退化、关键迭代、停止原因、完整墙钟耗时和产物路径。

报告同时给出实现状态和目标状态。实现状态仅使用：

- `VERIFIED`：相关完整测试套通过且同口径实测提升；这只表示实现已验证，不代表用户性能目标已达到；
- `NO_CHANGE`：没有可靠收益，保留基线；
- `CANDIDATE`：有潜力但最终门禁未完成；
- `BLOCKED`：环境、设备、编译器或测试阻塞。

目标状态仅使用 `MET`、`CONVERGED_WITHOUT_NUMERIC_TARGET`、`NOT_MET_CONVERGED`、`NOT_MET_UNFINISHED`。只有达到用户明确目标才能写 `MET`；用户未给数值目标且逐 case 结构覆盖和 `--final` 门禁通过时写 `CONVERGED_WITHOUT_NUMERIC_TARGET`；有目标但未达到、结构审计通过时才可写 `NOT_MET_CONVERGED`，否则写 `NOT_MET_UNFINISHED`。不能用 `VERIFIED` 或几何平均提升掩盖目标/慢 case 缺口。

没有完整性能证据时只报告假设或候选，不宣称优化完成。
