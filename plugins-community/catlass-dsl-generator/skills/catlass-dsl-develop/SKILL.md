---
name: catlass-dsl-develop
description: Generate a complete CATLASS Python DSL operator from a frozen DESIGN.md, then debug, review, test, benchmark, profile, and optimize it with knowledge queries and a text-tracked milestone log. Use for CATLASS DSL operator implementation, debugging, and delivery from a frozen design.
---

# CATLASS DSL Develop（门禁引导）

> **入口**：本阶段开始前先加载 `catlass-dsl-workflow` 并输出阶段卡。若 `DESIGN.md`
> 不存在或要素不全，退回 `catlass-dsl-design`，禁止现场补设计直接实现。
> **硬前置**：`.catlass-dsl/references/catlass/examples/python/tla_dsl` 必须已成功下载且能
> 列出 `.py` 样例；缺失时先按 workflow §6 下载，下载失败即停止，不得用知识库或已安装包
> 源码替代。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合，
> 也不用进展汇报代替继续执行。仅当缺少只有用户能提供的信息、或可行路径已穷尽时才提问。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块，
> 一次只做一类动作，禁止串成一条长命令。编译单独成块、不套 60s，但必须带明确上限（默认
> `timeout 600s`）；单例运行一条命令只跑一个 case/shape，加 `timeout -k 5 60s`；批量按
> 2~5 片拆分，分片命令用分片级超时（用例数 × 60s + 余量），片内逐例 60s 硬超时，禁止整批
> 套一次性 60s 或裸跑。超时、无输出增长或卡住时立即终止命令，不得死等或加大超时。远端或
> 容器调试时，`timeout` 必须写在最终执行环境的命令内，外层只负责连接；只包外层 ssh 会
> 留下孤儿 kernel 继续占卡。
> 终止后先做卡健康判定：在同一张卡跑一个 60s 内的小探针或官方算子；探针能过就判定为本
> 算子实现的死锁，直接转 debug 定位；探针也失败才判定为设备问题，换卡重试一次。
> 每次上板运行都要在 `ITERATIONS.md` 的 Run Log 记一行：命令、**实际设置的超时值**、
> 墙钟耗时、退出码；被超时杀掉记 `killed by timeout`。脚本内的超时同样要写，不能只写
> "已运行"，否则 `long timeout` 这类写法永远不会暴露。被超时杀掉的运行不产生性能结论。

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

## 先跑性能骨架，再铺正确性

**第一版必须先写“性能骨架”，不是“能跑通的最小实现”。** 用单 GEMM + 主 GEMM 的最大
dev shape（或等价合成 shape）先让流水线指标达标（见下方 DoD），再叠加 epilogue、
其余 vector / GEMM 阶段和边界处理；正确性在性能骨架之后收敛。融合、分块、尾块只改变
字节数，改不动流水；流水一旦定死，后续扩展全部继承它，返工成本高一个量级。

### GEMM Skeleton Contract

凡包含 MMAD 的算子（纯 matmul、Cube-Vector、Cube-Vector-Cube 融合都算）都必须对其中
每个 matmul 阶段做骨架评测，并从 [`templates/gemm_pipelined.md`](templates/gemm_pipelined.md) 起改。
骨架是剥掉 vector / epilogue 后的单 GEMM 形态，用途是先定住 matmul 的流水基线；要不要
评测只看“有没有 MMAD”，不看整体是否 GEMM 主导——CV 融合算子的瓶颈通常在 matmul 段，
matmul 骨架拉跨，后面叠多少 vector 优化都补不回来。纯 vector（无 MMAD）算子不适用。
骨架只是测量工件：它不改变 DESIGN 决定的 kernel 划分与 `fusion.mode`，也不能作为把
CV 融合算子拆成独立 Cube / Vector kernel 的理由；最终交付结构仍按 DESIGN 的融合方案。
首版单 GEMM 骨架阶段只允许修改 `BM/BN/BK/SK`、dtype 与块映射（波次量化 / L2 swizzle）；
`SK` 是 L0 的 K 子块步长。改完 tile 与 dtype 后先按 `knowledge/dsl/pattern-capacity-driven-tiling`
校核 L1/L0A/L0B/L0C 容量（超限编译期不报错，上板才 OOB）；
固定参数写在 kernel 函数内，不得读取模块级配置；不得改动缓冲数量、同步顺序与 MMAD
结构。主 GEMM 含 batch 维时，骨架只验证单 batch，batch 循环在 implement 阶段加在外层，
不得改动缓冲与同步结构。DESIGN 有多个 GEMM 时，首版只写主 GEMM（取 DESIGN 中时间
占比最高者）的骨架；其余 GEMM 在 implement 阶段各自补一次独立骨架评测，按同一 DoD、
同一取数口径，不得以“结构和主 GEMM 相同”“占比很小”或“已在完整算子里带过”为由跳过。
每个 GEMM 都要在 Skeleton Performance 表占一行、有自己的 evidence 目录；全部 GEMM
都有判定（`passed` / `passed (waived)` / `bandwidth-bound`，或 DESIGN 写明的结构性
豁免）后才能进入 optimize。后续叠加 epilogue 等阶段按 DESIGN 扩展，不受这条限制。

必须保留的结构：

- L1A/L1B 各 2 份缓冲，按 `l1_idx` 交替；
- L0A/L0B 各 2 份缓冲，按 `l0_idx` 交替；
- `BK > SK` 且 `BK % SK == 0`，一个 L1 载入喂多个 L0 子块。

禁止的反模式：

- 单缓冲 + 每个 `copy` 紧跟 `set_flag/wait_flag`（强制 MTE2→MTE1→CUBE 串行）；
- `BK == SK`（MTE1 与 CUBE 无法交错）；
- 多个 ready flag 共享同一个 MTE1→CUBE 对（会死锁）；
- 只做 L1 双缓冲、L0 仍单缓冲（单独收益只有 3%~5%，拿不到第 3 级的 20%~30%）。

要偏离模板，必须在 `ITERATIONS.md` 写清“为什么双缓冲 / `auto_sync` 在此不可用”，否则
skeleton 不计通过。目标 DSL 版本不支持 `auto_sync` 时，按官方 flag-sync 样例
（`examples/python/tla_dsl/` 下的 `basic_matmul.py`）实现等价双缓冲，结构不变，并在
账本记录样例路径。各级判据见
[`knowledge/dsl/pattern-gemm-optimization-ladder`](../../knowledge/dsl/pattern-gemm-optimization-ladder.md)。

### skeleton 性能 DoD

在**每个 GEMM 各自的最大 dev 用例**上跑 fresh profiler；多 GEMM 时逐个执行，不合并成
一次。下面的判据是默认线，不是硬门禁：不满足时，只要在 `ITERATIONS.md` 写清可核对的
理由，就可以记 `passed (waived)` 继续。

**默认判据**

- GEMM kernel 时间 ≤ `1.05 ×` 同 shape、同输入 dtype 的 `torch.mm`；
- 不出现串行特征：不得同时满足 `mac < 0.45`、`0.80 < mte2 < 0.95`、`overlap < 1.30`。

**参考线（写入账本用于归因，不单独判失败）**

| 指标 | 参考值 |
|---|---|
| `aic_mac_ratio` | 计算主导时 ≥ 0.60 |
| `aic_mte2_ratio` | 带宽主导时 ≥ 0.90 |
| `overlap = mac + mte2` | 计算主导时 ≥ 1.30 |

**tile 对照（必做，不是可选优化）**

骨架阶段的旋钮只有 `BM/BN/BK/SK`、dtype 与块映射，这三项必须在记账前扫过，不能测一组就
下结论。每个 GEMM 至少测 3 组配置，主 GEMM 必须包含一组 `BM=256` 量级的大 tile；缓冲
数量、同步顺序与 MMAD 结构在这一步不动。

`ITERATIONS.md` 的 Skeleton Performance 表每组一行，写全 `BM/BN/BK/SK`、块映射、
`gemm_ms`、`mac`、`mte2` 与名义搬运下界，并标出选中组。判定取三组中最好的一组：
达标记 `passed` / `bandwidth-bound`；三组都超 1.05× 才考虑豁免，且豁免必须附这三组数据。
只测一组就写豁免的，骨架不计通过。

选中组的 tile 写回 `DESIGN.md` 并带进 implement，主实现不得退回未选中的配置。`mte2` 已
接近 1 而时间仍落后时，先怀疑 tile 过小导致 A/B 重读，按本节扫参数，不要先归因于流水。

**豁免（有理由即可通过）**

不满足默认判据时，在 Skeleton Performance 表记 `passed (waived)`，并在
`ITERATIONS.md` 写明：

1. 哪条判据未达到、实测值是多少；
2. 为什么该判据对这个 shape / 硬件 / 版本不适用或不可达（带宽受限、尾块、波次不足、
   profiler 口径、编译器限制等），附 roofline 或对照数据；
3. 已做过的判断或改动（阶梯轴、roofline、对照实验均可）与结果；
4. 对最终算子的预期影响，以及 optimize 阶段的补偿计划。

“感觉差不多”“时间已经很快”这类没有测量的理由不算。理由是否成立由 agent 自行判断并
记录，不设二次确认；用户明确不同意时以用户意见为准。

判定顺序：

1. **带宽受限**：`mte2 ≥ 0.95` 且 `gemm_ms ≤ torch.mm` 时记 `bandwidth-bound` 通过，
   不要求 `mac` 与 `overlap`；账本写明算术强度与带宽下界。
2. **串行特征**：命中 `mac < 0.45`、`0.80 < mte2 < 0.95`、`overlap < 1.30` 时，默认
   先修流水；若确认是尾块、波次不足、shape 过小或 profiler 口径等结构性原因，按上面的
   豁免记录后通过。
3. **时间**：`gemm_ms ≤ 1.05 × torch.mm` 记 `passed`；超过时同样可走豁免。
4. **证据**：Skeleton Performance 表缺失或没有数字时不能通过；`passed` /
   `bandwidth-bound` / `passed (waived)` 必须附 `kernel_details.csv` 路径。

证据按 GEMM 分目录：`evidence/skeleton/<gemm-id>/profile/kernel_details.csv`（只有一个
GEMM 时可省略 `<gemm-id>`），每个 GEMM 的数值写入 Skeleton Performance 表的对应行。
CV 融合算子即使 vector 段占比更高，也要逐个 matmul 阶段评测，因为瓶颈通常在 matmul 段；
纯 vector（无 MMAD）算子不套用该表，但必须在 DESIGN 说明为什么不适用。

取数口径：只测单 GEMM 骨架（不含 host 与 epilogue），shape 取该 GEMM 自己的最大 dev
用例，多 GEMM 时逐个各测一次；该 shape 运行成本过高时，可用各维不小于原值 50% 的
合成 shape；shape 过小导致单次 kernel <100µs 时，可等比放大到 ≥100µs。两种替换都要在
账本记录原 shape、替换后 shape 与原因。同时记录块映射参数 `nm = ceil(M/BM)`、
`nn = ceil(N/BN)`、A/B 名义重读倍数（A 为 `nn`、B 为 `nm`）与按 HBM 带宽折算的搬运
下界时间；`mte2` 偏高时先用这组数区分“块映射重读”和“流水缺失”，再决定改 swizzle
还是修流水。

骨架的网格和结论只在其自身任务分解下成立，不代表最终 kernel。若最终 kernel 改变了任务
映射、归约范围或同步方式，必须在 DESIGN 里重新评估受影响的结构指标（并发度、数据复用、
每块工作量、跨核 / 跨引擎同步开销等），不得直接沿用骨架或样例的结论；评估项按偏离类型
选择，不设固定清单。

CV 融合算子光有设计评估不够，implement 收敛后要实测回测一次：从最终 kernel 的 profile
里取 GEMM 段的实际耗时（按 kernel 名取单轮行；多轮采集按 `Step Id` 分组，取该 kernel
齐全的一轮），折算有效算力 `2·M·N·K / 该段耗时`，与骨架同 shape 的 GEMM 时间
对比。偏差超过 15% 时在 `ITERATIONS.md` 写归因——UB / L1 被 epilogue 挤占导致 tile 退化、
跨核交棒插入额外等待、还是流水被 vector 段打断——并说明该损失是否可以接受。结构属于
CV 融合却没有这项回测记录的，implement 不算完成。

MAC/MTE2 占比只从 `kernel_details.csv` 同一行的 `aic_*` 列读取（字段名随版本
可能是 `aic_mac_ratio` / `aic_cube_ratio`，以实际列为准并把列名写入账本）；`overlap` 为
同一口径下的两者之和；GEMM 时间取该 kernel 的 task duration，`torch.mm`
基线优先取同一 profiling 会话的 kernel task duration，无法同采时两者都改用设备事件
计时（warmup 后取中位数）并注明口径。profiling 配置按
[`catlass-dsl-bench`](../catlass-dsl-bench/SKILL.md) 的 Level1 + AiCMetrics；无法产出
kernel 级产物时记 `blocked` 并停止推进，待 profiling 恢复后重跑；不得用自述数字通过
skeleton。

骨架的 host 侧用一次性脚本 `tools/skeleton_profile.py`：按 DESIGN 冻结的 M/N/K/dtype
直接构造 A/B/C，编译并 launch 骨架 kernel，同时计时同 shape `torch.mm`；脚本路径、命令
与结果写入 `ITERATIONS.md`。该脚本只服务 skeleton 验证，不进入主实现文件，也不改变
`solution.json` 的交付约束。

### skeleton 自检（记 `passed` 前逐项确认）

1. 首版是否只有一个 GEMM，且直接来自模板，没有带入 epilogue、边界处理或多 dtype 分派？
2. L1A/L1B、L0A/L0B 是否各 2 份并按 `l1_idx` / `l0_idx` 交替？
3. 是否满足 `BK > SK` 且 `BK % SK == 0`？
4. 四项数值是否来自同一次 fresh profile 的 `kernel_details.csv` 同一行，并已写入
   Skeleton Performance 表？shape、列名、命令与证据路径是否齐全？
5. 时间超过 `1.05 × torch.mm` 或出现串行特征时，是否先扫过至少 3 组 tile / 块映射并按
   阶梯修流水；仍不达标时，是否写清了可核对的豁免理由（未达判据、原因、三组对照数据、
   optimize 补偿计划）？
6. DESIGN 列出的每个 GEMM 是否都有各自的 Skeleton Performance 行与 evidence 目录，没有
   用“结构与主 GEMM 相同”或“占比小”跳过？单 GEMM 算子此项直接答“是”。

六项都答“是”才记 `passed`；未达默认判据但第 5 项给出了可核对豁免理由的记
`passed (waived)`；其余记 `failed` 并先修。

骨架判定通过（含 `passed (waived)`）后再做**最小 focused 验证**（编译 + 最小 shape 运行），
随后进入 `implement`，按「正确性 → 边界 → 性能细节」增量扩展。每次扩展都必须重新通过
编译与 focused test，全部 focused cases 通过后才进入 optimize。

保持一个主实现文件；`solution.json` 随每次增量同步更新，不做“攒到最后一次性生成”。

主实现文件开头固定写两行运行约定，作为每次改代码都会看到的提醒：

```python
# 上板运行：kernel 命令在最终执行环境内加 timeout -k 5 60s；单例运行（一个 case/shape）>60s 视为失败，不得原样重试。
# 每次运行后更新 ITERATIONS.md 的 Run Log（Timeout / Wall / Exit）。
```

写 `solution.json` 前确认这两行存在；缺失时补齐再生成，否则 Run Log 与超时纪律在长会话里
很容易失效。

代码生成阶段只产出文件：主实现、`solution.json` 与 `ITERATIONS.md` 记录。写
`solution.json` 时按 UTF-8 原始字节记录 SHA-256（保留原始换行），保证源码包与文件一致。
本地不做语法检查、导入检查、模拟/打桩检查、编译或运行；`run` 入口、语法与一致性检查
作为真实环境第一次执行的一部分完成，能力验证由紧随其后的 focused test 承担。

**调研预算（可数，不依赖时钟）**：骨架判定通过之前，同一文件读取不超过 2 次，总读取不超过
20 次；达到任一上限必须停止调研并落笔写骨架。骨架判定通过后应把重心放在“改→编译→上板→
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
> 文件传输、编译、运行命令必须是各自独立的命令块；单例调试时运行算子必须逐例拆成单条
> 命令，每条只跑一个用例，并由 `timeout -k 5 60s` 包裹；full test / benchmark 的分片命令
> 按分片级超时、片内逐例 60s。`timeout` 必须写在最终执行环境的命令内，
> 例如 `docker exec <容器> bash -lc '... && timeout -k 5 60s <cmd>'`；只包外层 ssh 会让远端
> 孤儿进程继续占卡。打包命令会让传输或编译占用 60 秒预算，既无法判断超时发生在哪个环节，
> 也无法区分算子死锁与设备问题。

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
- **按结构选参考样例**：Cube-only（纯 MMAD）参考
  `tla/kernel/matmul/basic_mmad/basic_matmul.py`；Cube-Vector（MMAD 后接 vector 后处理）参考
  `experimental/kernel/matmul/matmul_softmax_row_stats/matmul_softmax_row_stats.py`；
  Cube-Vector-Cube（两段 MMAD 中间夹 vector）参考
  `experimental/kernel/matmul/matmul_add_chain/matmul_add_chain.py`；
  纯 vector 参考 `tla/kernel/vec/basic_vadd/basic_vadd.py`、`dsl_tutorials/vector_simd_ops/`、
  `dsl_tutorials/vector_simt_ops/`。先按整体结构（阶段序列、
  数据流、任务映射、同步方式）选主参考；主参考是基线不是约束，偏离时写清理由。没有
  足够契合的主参考时，可以组合多个分步样例，但要显式设计组合后的整体结构并照常跑
  skeleton DoD。路径先在
  `.catlass-dsl/references/catlass/examples/python/tla_dsl/` 下用 `find` 确认；
  目标版本没有同名样例时按结构检索等价样例，并把实际路径写入 `ITERATIONS.md`。
- 只有同时满足三条才允许写独立探针：examples 未覆盖；失败会推翻已冻结设计；无法用编译
  检查或分段运行判断。
- 单个探针只回答一个是/否命题，自带断言与退出码，单次运行 ≤60 秒；同一探针修改 3 次
  仍不通即判定该机制不可用，按 DESIGN.md 的备选路径推进。
- 探针总数不超过 5 个。超出即停止验证，未验证机制在 `ITERATIONS.md` 标注为假设。
- 探针通过的代码必须直接提升为主实现的骨架函数；得到的容量、对齐、语义结论立即回写
  `DESIGN.md` 的约束清单。

## UB 空间预算

vector 段与 SIMT 段共用同一块 tile 的结构（读改写型 epilogue、CV 局部融合）里，tile 面积由
UB 剩余空间反推，不能先定 tile 再找空间：每多一份整 tile 的中间缓冲就压低一档。定 tile 前先
按 `knowledge/optimization/tiling-and-capacity` 的 UB 预算算一遍，并优先评估 tile 内原地复用
能释放多少空间。SIMT region 要求静态 shape，省空间只能缩 buffer，不能靠动态 shape。

## 高风险接口选型

`tla.gather` 接口使用限制多，默认不用它。需要实现 gather 操作时，必须参考并使用
`**/indexer/gather/gather.py` 样例的 SIMT 标量读写法；样例写法不适用时，可参考知识库
`knowledge/dsl/pattern-precomputed-index-structure` 中的方案进行扩展，理由写入 `DESIGN.md`。

scatter 没有官方样例，SIMT 标量读写写法同样参考 `**/indexer/gather/gather.py`；索引结构
按知识库 `knowledge/dsl/pattern-precomputed-index-structure` 的 run 表（CSR）实现：host
侧按输出 tile 建表、按 `(row, col)` 合并重复 id，kernel 内每线程对整段 run 求和后只做一次
目标格读改写。放置位置不做默认规定：独立 Vector kernel 与 Cube 融合 epilogue 都可行，
按 `DESIGN.md` 的收益与风险权衡选择。融合时至少保持单写者，SIMD/SIMT 交界按样例放
`pipe_barrier`，SIMT region 内需要线程间同步时用 `sync_threads()`；可见性与 UB 复用的
额外处理以目标版本实测为准。先按两段式跑通全量再合并，放置选择、理由与验证写入
`DESIGN.md` / `ITERATIONS.md`。

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
- 包含 MMAD 的算子的 `skeleton` 行必须在 `ITERATIONS.md` 的 Skeleton Performance 表按
  matmul 阶段逐行记录 `mac_ratio / mte2_ratio / overlap / gemm_ms / torch_mm_ms / Decision`；
  缺行、缺表或没有数字视为 skeleton 未完成。数值未达默认判据但写了可核对豁免理由的，Skeleton
  Performance 表的 Decision 记 `passed (waived)`，里程碑 `Result` 仍记 `passed` 继续。
- 探索阶段的每次小改动不必逐条记录；骨架、focused test 通过、数值收敛、优化轮次、
  review、full test、benchmark、finish 必须记录。
- 环境缺失时记录 `blocked` / `not_run` 并保留当前实现；环境就绪后重跑该阶段，不重建
  run。

## 开发循环

进入 focused test 前先做一次契约核对：逐项对照 `DESIGN.md` 的「数值契约」，确认实现里的
中间量存储 dtype、累加 dtype、窄化/舍入位置、归约顺序与 padding 语义和契约一致。发现
不一致时，先在 `ITERATIONS.md` 记录差异、实测影响和判断依据，再决定改实现还是修订契约；
修订契约按“设计修订”记录，不要用放宽容差代替。

数值不匹配时先对标杆、再谈容差：确认存储/累加 dtype、舍入位置与模式、归约顺序是否与
reference 一致；对齐之前不得把“在容差内”当作结论，也不得先改容差。容差只吸收完全对齐
后仍然存在的浮点尾差。中间量精度高于标杆（dtype/累加/舍入逐项不低于标杆，且至少一项
更高）时属于豁免情形，直接按用例容差判定，并在 DESIGN 的 deviation 项补齐实测偏差量级；
容差判定不过时豁免不成立，回到 design 按误差分析重新推导容差并记录，不得在调试中放宽；
重推后的容差仍不过，按精度 bug 处理。

1. **Validate.** 完整首版生成后，执行 focused cases，覆盖完整计算链路及受影响用例。
   没有独立 focused 命令时执行 full test 命令，不跳过首轮验证。正确性判据按
   [`catlass-dsl-bench`](../catlass-dsl-bench/SKILL.md) 的 tolerance 规则：
   `abs(candidate-reference) <= max_atol + max_rtol * abs(reference)`，匹配比例不低于
   `required_matched_ratio`，`max_error_cap` 是额外硬上限；NaN/Inf 默认失败。编译、运行、
   精度、review 或 full test 失败转 debug；失败后不测性能。转 debug 后继续修复并重跑
   full test，直到通过或所有可行路径确认真实不可行；不得以汇报失败结束。
2. **Analyze and query.** 遇到问题先读日志、DESIGN、当前源码及 `ITERATIONS.md`，按算子族、
   API、错误类型、shape/dtype/layout 查阅 `knowledge/debug`、`knowledge/dsl` 或
   `knowledge/operator`，性能问题查 `knowledge/optimization`。先读分类 `index.md` 定位
   候选条目，再完整读取实际采用的条目；记录采用路径与适用条件。
3. **Debug.** 一轮验证一个假设，写明 Expected 和判伪条件，然后增量修改完整实现。
   每次修改重新生成 `solution.json`；不得修改 reference、workload、容差或计时边界换取
   通过。debug 完成后回到 focused test。
4. **Record.** 里程碑结果写入 `ITERATIONS.md`，附实际命令、退出码、证据路径和 kernel SHA。
5. **Continue.** 按当前产物判断下一步，不凭记忆跳步；代码变化后重新评估受影响阶段。

### 多 kernel 精度定位

算子包含 ≥2 个 kernel、跨 kernel 中间张量或 C→V→C 交棒，且 focused/full test 出现数值
不匹配时，先按通用规则对齐每条边的中间量：存储 dtype、内部累加 dtype、layout/shape/
stride、有效区、舍入位置、归约/重复 ID/chunk 顺序都要与标杆一致；代数等价不等于数值
等价。然后完整读取
[`knowledge/debug/multi-kernel-precision-localization.md`](../../knowledge/debug/multi-kernel-precision-localization.md)，
按“阶段契约 → 局部检查 → 真实链路 → 双向替换 → 首错定位 → 单因素修复 → 原判据复测”
推进。局部 kernel 检查必须用标杆中间量作输入；下游 GEMM 用候选中间量算得对，不能证明
上游正确。输出不稳定时先查初始化、覆盖、布局、同步和运行配置，再谈舍入。检查矩阵、
双向替换组合和失败分类见该条目。
- “单例通过、批量失败”时，测试/对比脚本还要查编译缓存：编译产物 key 是否漏了参与
  编译的标量，是否缓存了随 workspace 重建的张量句柄。两者都会表现为批量失败，且失败
  模式接近数值问题。

## Review, test, and benchmark

- review 在**数值收敛之后**执行：focused test 通过且数值稳定后，才做 targeted review
  （high risk）和 final review，然后跑全部 required cases。编写与调试阶段不因 review
  未做而阻塞。
- review 必须检查每个阶段的复杂度标注（实现中一行注释即可）。选择/索引类阶段若为
  O(M·K·nv)（按 V 块重新扫描全部 K），必须给出理由与实测代价，否则判不通过并要求改为
  O(M·K)。
- review 默认检查接口、数学语义、数据流、融合范围及 DSL 实现；含 MMAD 的算子还要核对
  DESIGN 列出的每个 matmul 阶段都有各自的 Skeleton Performance 行与 evidence；严格单融合
  模式额外检查自由名字、唯一声明、单 launch、无独立 dispatch 和无 Torch/Tensor 计算替代。
- full test 使用完整 `workload.jsonl`，必须分 2~5 片并行执行；每片一条命令、写独立
  `--output`，分片命令用分片级超时，片内逐例 60s 硬超时。全部 required cases 逐例
  passed 才通过。失败用例一次收集、批量修复、逐例复验后再重跑整轮。
- full test 失败先读结构化 `result.json`，按数值不匹配、输入生成失败、预期报错验证、空
  输入和 info 分类；不得只 grep 日志或把总失败数都归为精度问题。
- 测试/对比脚本先按最大用例估算显存预算（输入、候选/reference 输出、中间量、workspace），
  超预算时分块比较、逐例释放或分片执行；OOM 或“单例通过、批量失败”先排查脚本，
  不得直接按 kernel 精度 bug 处理。
- 多 kernel / 跨 kernel 中间张量的算子，精度证据必须包含逐阶段中间量对照和首个偏差边；
  只有最终输出对比不足以证明链路与标杆对齐。
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
4. **单例 60s**：每个执行单元（一个 case/shape）的单次运行受 60s 硬超时约束；分片命令按
   分片级超时，片内逐例同样受 60s 约束。超时允许换卡重试一次，二次失败判定 kernel 问题，
   不得加大超时绕过。

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
