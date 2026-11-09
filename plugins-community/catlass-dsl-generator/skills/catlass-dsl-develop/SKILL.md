---
name: catlass-dsl-develop
description: Generate a complete CATLASS Python DSL operator from a frozen DESIGN.md, then debug, review, test, benchmark, profile, and optimize it with knowledge queries and a text-tracked milestone log. Use for CATLASS DSL operator implementation, debugging, and delivery from a frozen design.
---

# CATLASS DSL Develop（门禁引导）

> **入口**：本阶段开始前先加载 `catlass-dsl-workflow` 并输出阶段卡。若 `DESIGN.md`
> 不存在或要素不全，退回 `catlass-dsl-design`，禁止现场补设计直接实现。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合，
> 也不用进展汇报代替继续执行。仅当缺少只有用户能提供的信息、或可行路径已穷尽时才提问。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块，
> 一次只做一类动作，禁止串成一条长命令。编译独立 300s，单例运行 `timeout -k 5 60s`，
> 批量按分片逐段执行，禁止整批套一次性超时，禁止裸跑。超时、无输出增长或卡住时立即终止
> 命令，不得死等或加大超时。
> 终止后先做卡健康判定：在同一张卡跑一个 60s 内的小探针或官方算子；探针能过就判定为本
> 算子实现的死锁，直接转 debug 定位；探针也失败才判定为设备问题，换卡重试一次。

读取已冻结设计，一次生成全部代码，再验证和增量调试。本阶段**没有状态机脚本**：阶段推进
由 `ITERATIONS.md`、结果文件与实际命令承担；进入下一阶段前必须用
`<SKILL_DIR>/../catlass-dsl-workflow/scripts/gate.py` 对应门禁验证记录齐备。

## First action

先读用户提示、`DESIGN.md` 和工作区已有产物；可发现的信息不要询问。展示：

**📋 Resolved Plan**

- **Workspace / Design** — 仓库根目录、设计路径和 SHA-256。
- **Implementation** — 一个主 Python 文件，可包含多个 DSL kernel。
- **Fusion** — `performance`（默认）或用户明确指定的 `single_kernel`。
- **Benchmark inputs** — `solution.json`、`definition.json`、`workload.jsonl`。
- **Dev subset** — design 冻结的 `workload_dev.jsonl`（用户用例 ≥30 条时为 30~40 条；不足 30 条时全选；无用户用例时按覆盖需要自定，总数 ≤40），optimize 专用。
- **Commands / required cases** — 实际 argv、case IDs、设备与环境。
- **Risk / Performance / Knowledge** — 评审重点、性能目标或只测量、查询条件。

`fusion.mode=performance` 默认不做源码 anti-hack 扫描，允许多个 kernel、executor、
GM 中间张量及显式 shape/dtype/layout 分派。只在用户明确要求单个融合 kernel 时使用
`single_kernel`：只声明一个 `@tla.kernel`、每次 `run` 只 launch 一个 kernel，不得为编译期
变体声明独立 kernel。

每个 kernel 保持自包含：固定配置在函数内定义，运行时配置走形式参数或 tensor metadata；
host 不通过 `global`、模块级可变状态或 closure 注入语义。

**写代码之前先记录骨架里程碑。** 首次进入 develop 时，先在 `ITERATIONS.md` 追加一行
`skeleton`：有真实环境时立即运行并记 `passed` / `failed`，附命令与日志路径；纯本地生成
模式记 `blocked` / `not_run`，写好编译与最小 shape 运行命令，留作远端第一个执行动作。在这条记录
产生前，禁止写实现文件、禁止写探针、禁止编译。

**先生成、再推进**：一开始就拿到服务器信息，或本来就在服务器上，也要先完成设计与代码
生成再进入编译、上板等动作；不得先做环境探测、装环境或跑标杆。

## 骨架先行，增量扩展

**先写出可编译、可运行的最小端到端路径**：单 tile / 单分块、单一路径、host 可调用；
数值允许错误，但必须能在 NPU 上完整跑完一次并返回。

骨架完成后先做一次**最小 focused 验证**（编译 + 最小 shape 运行），拿到第一个真实
反馈；随后进入 `implement`，按「正确性 → 边界 → 性能」顺序增量扩展。每次扩展都必须
重新通过编译与 focused test，全部 focused cases 通过后才进入 optimize。

保持一个主实现文件；`solution.json` 随每次增量同步更新，不做“攒到最后一次性生成”。

代码生成阶段只产出文件：主实现、`solution.json` 与 `ITERATIONS.md` 记录。写
`solution.json` 时按 UTF-8 原始字节记录 SHA-256（保留原始换行），保证源码包与文件一致。
本地不做语法检查、导入检查、模拟/打桩检查、编译或运行；`run` 入口、语法与一致性检查
作为真实环境第一次执行的一部分完成，能力验证由紧随其后的 focused test 承担。

**调研预算（可数，不依赖时钟）**：骨架跑通之前，同一文件读取不超过 2 次，总读取不超过
20 次；达到任一上限必须停止调研并落笔写骨架。骨架跑通后应把重心放在“改→编译→上板→
看数”上，而不是继续查阅资料。

## 执行环境

开发机没有 NPU 时采用“本地生成、远端验证”：本地只负责设计与代码生成（主实现、
`solution.json`、命令与 `ITERATIONS.md` 记录）；静态检查、reference 运行、模拟检查、
编译、上板、精度比对、benchmark、profiling 全部在远端真实环境完成，连接与文件同步由
用户自备。该模式下本地不得安装或配置 CANN / CATLASS DSL，也不得在本地做上述任何验证；
远端暂不可用时记录 `blocked` / `not_run` 并保留精确重跑命令，环境就绪后直接执行。

一开始就拿到服务器信息时同样先完成代码生成，再操作远端环境；本来就在服务器上开发时
也先生成文件，再编译和上板。

目标 `ascend-catlass-dsl` 版本在启动阶段冻结；远端实际版本与目标不一致时先停下确认。

**编译与运行必须拆成两条独立命令**，禁止把「改代码 + 编译 + 上板」串进同一条命令：

```bash
# 编译：不绑定设备，只做 lowering/构建
unset ASCEND_RT_VISIBLE_DEVICES; timeout 300 <compile>

# 运行：占卡执行，必须自带硬超时
export ASCEND_RT_VISIBLE_DEVICES=<id>; timeout -k 5 60s <run>
```

> **重要！本地生成、远端调试时，禁止用一条带超时的命令把「传输 + 编译 + 运行」打包执行。**
> 文件传输、编译、单例运行必须是各自独立的命令块；运行算子必须逐例拆成单条命令，每条只
> 跑一个用例，并由 `timeout -k 5 60s` 包裹。打包命令会让传输或编译占用 60 秒预算，既无法
> 判断超时发生在哪个环节，也无法区分算子死锁与设备问题。

单次算子执行超过 **60 秒**即判定超时或不可用，立即停止该次执行。运行命令必须由
`timeout -k 5 60s` 包裹、不得裸跑：设备同步超时默认要等 8 分钟以上，裸跑会让每轮调试
固定损失这段时间；返回码 124/137 或等价超时信号即判定超时。触发后暂停后续 case、采样和
验收，保存触发用例、输入规格、设备、源码摘要、耗时与日志后转入 debug；不得用该次执行
生成性能结论。

## 机制验证预算

骨架成形前允许验证少数不确定机制，但验证必须有计数上限、有出口、可复用：

- **默认不写独立探针**。优先使用 catlass 样例中已覆盖的写法：master 源码在
  `examples/python/tla_dsl/`；examples 未覆盖但能通过编译或一次数值对比判断的，
  直接写入主实现后再验证。
- 只有同时满足三条才允许写独立探针：examples 未覆盖；失败会推翻已冻结设计；无法用编译
  检查或分段运行判断。
- 单个探针只回答一个是/否命题，自带断言与退出码，单次运行 ≤60 秒；同一探针修改 3 次
  仍不通即判定该机制不可用，按 DESIGN.md 的备选路径推进。
- 探针总数不超过 5 个。超出即停止验证，未验证机制在 `ITERATIONS.md` 标注为假设。
- 探针通过的代码必须直接提升为主实现的骨架函数；得到的容量、对齐、语义结论立即回写
  `DESIGN.md` 的约束清单。

## 高风险接口选型

`tla.gather` 接口使用限制多，默认不用它。需要实现 gather 操作时，必须参考并使用
`**/indexer/gather/gather.py` 样例的 SIMT 标量读写法；样例写法不适用时，可参考知识库
`knowledge/dsl/pattern-precomputed-index-structure` 中的方案进行扩展，理由写入 `DESIGN.md`。

## 里程碑与日志

阶段顺序：

```text
skeleton → implement → focused test → full test #1 → optimize →
targeted review（high risk）/ final review → full test #2 → [benchmark] → finish
```

- `benchmark` 只在 `DESIGN.md` 声明 `性能要求：requested` 时执行；`none` 时当前 kernel
  全量测试通过后直接 finish，不采集性能、不写 benchmark 里程碑。
- 每完成一个里程碑，在 `ITERATIONS.md` 追加一行；进入下一阶段前先跑
  `gate.py develop / fulltest / optimize / benchmark / finish`，未通过不得继续。
  唯一例外：`remote-detached` 的 focused test 补跑后，原地把该行 `deferred` 改为
  `passed` 并补上证据。

记录格式：

```markdown
| Iter | Stage | Kernel SHA256 | Command | Result | Evidence |
|------|-------|---------------|---------|--------|----------|
| 1 | skeleton | <sha256> | `timeout -k 5 60s python3 ...` | passed | evidence/skeleton.log |
```

- `Result` 只写 `passed` / `failed` / `blocked` / `not_run`；后两者必须写原因和重跑命令。
- `Kernel SHA256` 使用当前实现文件的实际摘要；代码变化后旧行保留，新增行。
- 探索阶段的每次小改动不必逐条记录；骨架、focused test 通过、数值收敛、优化轮次、
  review、full test、benchmark、finish 必须记录。
- 环境缺失时记录 `blocked` / `not_run` 并保留当前实现；环境就绪后重跑该阶段，不重建
  run。

## 开发循环

1. **Validate.** 完整首版生成后，执行 focused cases，覆盖完整计算链路及受影响用例。
   没有独立 focused 命令时执行 full test 命令，不跳过首轮验证。正确性判据按
   [`catlass-dsl-bench`](../catlass-dsl-bench/SKILL.md) 的 tolerance 规则：
   `abs(candidate-reference) <= max_atol + max_rtol * abs(reference)`，匹配比例不低于
   `required_matched_ratio`，`max_error_cap` 是额外硬上限；NaN/Inf 默认失败。编译、运行、
   精度、review 或 full test 失败转 debug；失败后不测性能。
2. **Analyze and query.** 遇到问题先读日志、DESIGN、当前源码及 `ITERATIONS.md`，按算子族、
   API、错误类型、shape/dtype/layout 查阅 `knowledge/debug`、`knowledge/dsl` 或
   `knowledge/operator`，性能问题查 `knowledge/optimization`。先读分类 `index.md` 定位
   候选条目，再完整读取实际采用的条目；记录采用路径与适用条件。
3. **Debug.** 一轮验证一个假设，写明 Expected 和判伪条件，然后增量修改完整实现。
   每次修改重新生成 `solution.json`；不得修改 reference、workload、容差或计时边界换取
   通过。debug 完成后回到 focused test。
4. **Record.** 里程碑结果写入 `ITERATIONS.md`，附实际命令、退出码、证据路径和 kernel SHA。
5. **Continue.** 按当前产物判断下一步，不凭记忆跳步；代码变化后重新评估受影响阶段。

## Review, test, and benchmark

- review 在**数值收敛之后**执行：focused test 通过且数值稳定后，才做 targeted review
  （high risk）和 final review，然后跑全部 required cases。编写与调试阶段不因 review
  未做而阻塞。
- review 必须检查每个阶段的复杂度标注（实现中一行注释即可）。选择/索引类阶段若为
  O(M·K·nv)（按 V 块重新扫描全部 K），必须给出理由与实测代价，否则判不通过并要求改为
  O(M·K)。
- review 默认检查接口、数学语义、数据流、融合范围及 DSL 实现；严格单融合模式额外检查
  自由名字、唯一声明、单 launch、无独立 dispatch 和无 Torch/Tensor 计算替代。
- full test 使用完整 `workload.jsonl`，必须分 2~5 片并行执行；全部 required cases 逐例
  passed 才通过。失败用例一次收集、批量修复、逐例复验后再重跑整轮。
- benchmark 只在 `DESIGN.md` 声明 `性能要求：requested` 时执行：逐例配对
  candidate/reference，引用 full test 的正确性证据，要求 candidate mean 为有限正数且
  profiling 通过。默认不做 anti-hack 审计；`single_kernel` 模式要求人工核查唯一声明与
  逐 iteration CSV。声明 `none` 时不做 benchmark、不采 profiling，也不写 benchmark 行。
- 多 kernel 时测量完整 `run`，包括所有阶段；不挑选其中一个 kernel 的耗时作为整体性能。
- 代码变化使旧测试、review、benchmark 和知识结论失效；每个结果绑定当前 kernel SHA-256。
- focused test 通过后先跑 `full test #1`，通过后进入 `catlass-dsl-optimize`，继承融合策略
  与 design 冻结的 `workload_dev.jsonl`。optimize 完成后按 targeted review / final
  review → 全量测试重新验收：采纳改动导致 kernel 变化时补跑 `full test #2`，全部未采纳、
  kernel 未变化时复用 `full test #1`。`性能要求：requested` 时继续 benchmark，`none` 时
  当前 kernel 全量测试通过即 finish。
- 选择空闲 NPU，执行前 source 项目 `env.sh`。candidate profiler 每轮重新采集；普通 trace
  归档只保留 CSV。

## optimize 阶段：模式 A / 模式 B

`full test #1` 通过后恒进入 optimize，不因“是否有性能目标”而跳过：

- **模式 A（target）**：design 冻结了性能目标与 threshold，达到目标后至少完成 1 轮。
- **模式 B（explore）**：没有性能目标时至少完成 **3 轮**并逐轮记录。

评测口径：

1. **只用 dev 子集**：基线与每轮复测都跑 design 冻结的 `workload_dev.jsonl`（用户用例
   ≥30 条时为 30~40 条；不足 30 条时全选；无用户用例时按覆盖需要自定，总数 ≤40），
   不得每轮换样本，也不得改用全量 workload 计时。
2. **逐例配对**：同一用例先跑 candidate 再跑 reference（或交替先后），在同一张卡相邻时间
   完成；有对标实现时不得只跑 candidate。
3. **控温仅在本阶段**：同用例内各实现温差 ≤3℃；执行前 AICore 频率须为 1650 MHz；
   温度超过 85℃ 停机。full test 与 benchmark 不做控温。
4. **单例 60s**：每例执行受 `timeout -k 5 60s` 约束；超时允许换卡重试一次，二次失败判定
   kernel 问题，不得加大超时绕过。

每轮顺序（缺一步视为该轮未完成）：

1. **瓶颈取证**：先做零成本算术 `makespan = ceil(总块数 / 核数)`，排除任务铺放空转
   （见 `knowledge/optimization/makespan-scheduling-gap`）；再从本轮 fresh profiling 指出
   主导 pipe、搬运/计算受限归属与理论下限。计数器不可用时改用单变量消融计时。
2. **单元假设**：一个改动轴、预期收益、判伪条件。
3. **针对性修改**：只改与假设相关的实现，保持正确性可重跑。
4. **前后对比**：同设备重跑 benchmark，记录 candidate mean 前后值、加速比与是否达成预期。

硬下限提前停止需给出量化依据（例如 >80% 时间落在单一 pipe 且已达理论带宽/算力）。
每轮把 bottleneck、axis、before/after、decision、evidence 写入 optimize 的
`ITERATIONS.md`；详细流程见 [`catlass-dsl-optimize`](../catlass-dsl-optimize/SKILL.md)。

## 交付与知识沉淀

`finish` 前必须证明当前源码的 full test 与 final review 通过，`性能要求：requested` 时
还要求 fresh benchmark 和 profiling 通过；两种情况都要求 `solution.json` 与源码一致。
检查清单：

- `ITERATIONS.md` 包含 skeleton 到 finish 的完整里程碑行，kernel SHA 与当前源码一致；
- full test #1（optimize 前）覆盖全部 required cases；优化采纳改动了 kernel 时 full test #2
  也要覆盖全部 required cases，失败重跑记录齐全；
- `gate.py finish --workspace <ws>` 通过，确认 optimize 轮次、当前 kernel 的全量测试、
  final review 与 Final Verdict 都已落账；`性能要求：requested` 时还包含 benchmark；
- `Final Verdict` 记录交付 kernel SHA-256，与当前源码及 `solution.json` 一致；列出的
  最终证据路径必须真实存在（例如 `full test #2` 的 result.json），引用缺失文件视为
  记录不完整；交付 kernel 与 optimize best kernel 不同时写明差异原因；
- final review 结论针对当前源码；
- `性能要求：requested` 时 benchmark 为当前源码的配对测量，profiling 数据存在；
  `none` 时允许无 benchmark 结果与 profile CSV；
- 产物目录包含 `solution.json` 与主实现文件，requested 时另含 benchmark 结果与 profile CSV；
- 目标版本与远端实际版本一致。

仅将完整测试或 profiling 直接支持的经验写入 `catlass-dsl-knowledge`，绑定最终 kernel
SHA-256。证伪假设、静态推断及 `not_run` 不入库。

## 产物布局（建议）

```text
<workspace>/
├── DESIGN.md
├── reference.py
├── definition.json
├── workload.jsonl
├── workload_dev.jsonl
├── solution.json
├── src/<kernel>.py
├── ITERATIONS.md
└── evidence/
    ├── skeleton.log
    ├── focused-test.json
    ├── reviews/final-review.md
    ├── full-test/result.json
    ├── benchmark/result.json
    └── profile/case-NNNN/{kernel_details.csv,step_trace_time.csv}
```
