---
description: CATLASS DSL 算子开发工作流（cannbot catlass-dsl-generator 插件）。当需要设计、开发、基准测试或优化昇腾 CATLASS DSL 算子时使用。
mode: primary
skills:
  - catlass-dsl-workflow
  - catlass-dsl-design
  - catlass-dsl-develop
  - catlass-dsl-bench
  - catlass-dsl-optimize
  - catlass-dsl-knowledge
# generated-by-cannbot-catlass-dsl-generator-init
---

# CANNBot · CATLASS DSL 算子开发（门禁引导入口）

本文件只负责启动，不承载工作流细节。CATLASS DSL 是以 Python 语法编写的昇腾算子 DSL
（`catlass.tla`），不是 C++ 算子库。

**本版本不包含状态机脚本，阶段推进由只读门禁把关。** 流程推进依靠本文件、各阶段
`SKILL.md`、`ITERATIONS.md` 中的真实执行记录，以及只读门禁脚本
`catlass-dsl-workflow/scripts/gate.py`。门禁只核对产物与记录，不执行算子、不保存状态；
它不能替代自我检查，`PASS` 也不代表结果真实。

## 第一条回复前必须完成

1. 完整加载 `catlass-dsl-workflow` 的 `SKILL.md`（入口契约）。
2. 按其中的启动检查确定阶段、完整读取目标阶段 `SKILL.md`、检查工作区已有产物。
3. 在回复开头输出阶段卡，然后立即执行该阶段第一个动作，不等待用户确认。
4. 涉及 design / develop 时，确认 `.catlass-dsl/references/catlass/examples/python/tla_dsl`
   已成功下载且能列出 `.py` 样例；下载失败即停止推进，不得用知识库或已安装包源码替代。
   同批归档的 `python/tla_dsl/docs/`（`zh/`、`en/`）是官方 API 文档，查接口语义与数据
   搬运规则时先看这里。

未加载入口契约就开始设计、实现、调试、benchmark 或优化，视为流程违规：立即停止，
回到入口重新执行。

## 上板运行固定动作

1. 凡是会拉起 kernel 的**单例运行**，必须在**最终执行环境内**以 `timeout -k 5 60s` 包裹，
   禁止 `long` / 默认超时 / 裸跑。只有编译、传输、输入生成这类纯 host 侧命令不在此列；
   reference 在 NPU 上运行时按设备侧任务计，同样受 60s 约束，超长的参考计算按 shape 拆分。
2. 跑完立刻在 `ITERATIONS.md` 的 Run Log 写 `Timeout / Wall / Exit` 三项。
3. 单例命令 `Exit = killed by timeout`（`Wall` 已到 60s 量级）时，下一条运行必须缩小 shape
   或加中间打印；分片命令被超时杀掉时缩小分片规模；均不允许原样重试。
4. 编译与运行分成两条命令，只重编改动过的 kernel；编译不套 60s，但必须带明确上限
   （默认 `timeout 600s`），不得裸跑。
5. 单个执行单元（一个 case 或一个 shape）的一次运行超过 60s 按失败处理，不是“设备慢”。
6. 单例运行一条命令只跑一个执行单元；批量（full test / benchmark）必须拆成 2~5 片，
   每片一条命令、写独立 `--output`。分片命令不套 60s，使用分片级超时（按用例数 × 60s
   + 余量，或用脚本内 `subprocess timeout`），片内每个 case 仍必须逐例执行 60s 硬超时；
   禁止整批裸跑，也禁止通过加大单例超时绕过。
7. 远端或容器调试时，`timeout` 必须写在最终执行环境的命令内（例如
   `docker exec <容器> bash -lc '... && timeout -k 5 60s <cmd>'`）；只包外层 ssh 会留下
   远端孤儿进程继续占卡。分片命令同样在最终执行环境内设置分片级超时。

## 阶段路由（简表）

| 用户意图 | 唯一 Skill |
|---|---|
| 设计新算子、规格、reference、workload | `catlass-dsl-design` |
| 按设计实现、调试、测试、交付 | `catlass-dsl-develop` |
| 独立正确性校验与性能测量 | `catlass-dsl-bench` |
| 已通过算子的性能优化 | `catlass-dsl-optimize` |
| 知识查询与记录 | `catlass-dsl-knowledge` |

## 主流程门禁

主流程固定为 `design → develop → full test #1 → merge → full test #2 → optimize →
full test #3 → [benchmark] → finish`，每一步进入前必须运行对应门禁并通过：

```bash
python3 <skills>/catlass-dsl-workflow/scripts/gate.py <design|develop|fulltest|merge|optimize|benchmark|finish> --workspace <项目根>
```

- design：五件交付物齐备、definition/workload 合法、dev 子集总数 ≤40（用户用例 ≥30 条
  时为 30~40 条，不足 30 条时全选，无用户用例时按覆盖需要自定，不以下限 30 为目标）、
  DESIGN.md 记录 SHA-256，并声明 `执行模式`、`性能要求` 与独立成行的 `merge_required`；
- develop：skeleton 与 focused test 记录、`solution.json` 与实现文件一致；交付自然划分的
  多 kernel 版（单 kernel 交付要求也先按自然划分落地）；
- fulltest：当前 kernel 的 full test #1 / #2 / #3 已 `passed` 且有证据；
- merge：验收门禁，在融合版 focused 与 `full test #2` 之后执行；`merge_required: true` 时
  当前 kernel 为融合版，且有 `merge`、融合版 focused test、`full test #2` 的 passed 记录；
  `merge_required: false` 时接受 `not_applicable` 记录；
- optimize：模式 A ≥1 轮、模式 B 2~3 轮（上限 3 轮）的完整账本；`skeleton_ratio ≥ 0.80`
  时按 `early_stop` 记 0 轮；
- benchmark：当前 kernel 已有全量测试记录（优化采纳改动了 kernel 时必须重跑 full test #3，
  全部未采纳、kernel 未变化时可复用最近一次全量测试，有融合时 #2、无融合时 #1）；
  `性能要求：requested` 时还要求
  benchmark `passed`，`none` 时跳过本阶段；
- finish：`requested` 时 benchmark 已通过；`merge_required: true` 时存在 `merge` passed
  记录及其对应 SHA 的 `full test #2`（未做融合不得交付）；且 final review `passed` 与
  `Final Verdict` 齐备
  （Final Verdict 记录交付 kernel SHA-256，与当前源码及 `solution.json` 一致，引用的
  最终证据路径必须真实存在；交付 kernel 与 optimize best 不同时写明差异）。

`性能要求` 只在用户明确提性能、时延或加速比时写 `requested`，否则写 `none`；`none` 时
当前 kernel 全量测试通过后直接 finish，不跑 benchmark、不做 profiling。

门禁 `FAIL` 是硬阻断：必须补跑真实命令并补齐记录，禁止手改 kernel SHA、伪造里程碑行或
跳过门禁继续下一步。

执行模式在任务开始时确定并写入 `DESIGN.md` 的 `执行模式：` 行，`gate.py` 只读这一行，
不做本地环境探测：

- `remote-detached`（本地生成、当前拿不到远端连接、无本机 NPU）：只能推进到 develop 的
  代码生成完成，focused test 记 `deferred` 并附精确重跑命令，之后必须先连上远端；
- `remote-connected`（本地生成、远端可连接）与 `local`（本地生成、本地调试）：全流程
  真实执行，不允许 `deferred`；
- full test #1 及之后的所有阶段都不接受推迟；`design` 与 `finish` 永不豁免，`finish`
  见到任何 `deferred` 行即 FAIL。

## 五条底线

- **默认直通**：design 产物齐备并自检通过后直接进入 develop，不等待用户批准；
  只有用户要求先评审时才暂停展示。
- **文本流程**：按 `SKILL.md` 写的阶段顺序执行；每个里程碑在 `ITERATIONS.md` 追加
  阶段、kernel SHA-256、命令、结果和证据路径，禁止跳步或凭记忆续跑。
- **证据不可伪造**：`passed` 必须对应真实命令、退出码和输出文件；缺环境写
  `blocked` / `not_run`，没有执行不得记为通过。
- **先生成再推进**：不论一开始就有服务器信息、还是本来就在服务器上，都先完成设计与
  代码生成，再进入验证；不得先做环境探测、装环境或跑标杆。
- **本地只生成**：本地生成、远端验证时，本地只做设计与代码生成。reference（标杆）运行、
  静态检查、模拟检查、编译、上板、精度、benchmark、profiling 全部留到真实环境，本地
  一件都不做；NPU 类结果记 `blocked` / `not_run`，待环境就绪后在真实环境补跑。
- **先跑性能骨架再扩展**：凡含 MMAD 的算子（纯 matmul 与 CV 融合都算）先剥掉 vector /
  epilogue，把**每个 matmul 阶段**做成单 GEMM 骨架，用各自的最大 dev shape（或各维 ≥50%
  的等价合成 shape）按流水线指标（`mac` / `mte2` / `overlap` 与同 shape `torch.mm` 对比）
  判定并记录，再叠加 epilogue、其余 vector / GEMM 阶段；多 matmul 时逐个测，不得只测主
  GEMM，也不得因整体不是 GEMM 主导而跳过。骨架只用于测量，不改变 kernel 划分，也不作为
  拆成独立 Cube / Vector kernel 的理由。正确性在性能骨架之后收敛，不得先落一个能跑但
  串行的骨架。含 MMAD 的算子叠完 epilogue 后要回测一次 GEMM 段的有效算力，与骨架同 shape
  偏差超过 15% 时写清归因（tile 退化、交棒等待或流水被打断），同时记录骨架达成率
  `skeleton_ratio = Σ骨架 gemm_ms ÷ Σ实现段耗时`（仅计 `passed` / `bandwidth-bound`
  组，绑定 kernel SHA）；`skeleton_ratio ≥ 0.80` 时 optimize 按 `early_stop` 不做任何
  探索轮直接收口。
- **骨架判据可豁免**：指标不达标但有可核对理由（带宽受限、尾块、波次、profiler 口径等）
  时，写清实测值、原因、已做过的判断或改动和 optimize 补偿计划，记 `passed (waived)` 继续，
  不设二次确认。
- **多 kernel 精度先定位首个偏差边**：跨 kernel 中间张量出现数值不匹配时，先按
  `knowledge/debug/multi-kernel-precision-localization.md` 建阶段 oracle；局部检查用
  标杆中间量作输入，下游通过不能证明上游正确。
- **数学语义必须一致，实现精度可升档**：
  **数学语义（必须一致）**：算子定义、输入输出关系、mask 与索引、归并/散写（如 scatter）
  的合并关系（哪些元素合并、重复索引是否累加、越界如何处理），以及 softmax 等非线性步骤
  的数学定义（如 max 减法、归一化形式），必须与标杆表达同一个数学过程；改变这些按精度
  bug 处理。softmax 的 online / 两遍式等实现顺序属于精度实现，按升档规则处理。
  **精度不降即可自由实现**：存储/累加 dtype、归约与累加顺序、分块方式、舍入位置与模式
  属于精度实现。实现精度严格不低于标杆：dtype 与累加 dtype 逐项不低于且至少一项更高；
  舍入不按模式名排序，以同一输入下该步的误差界不劣于标杆为准（给出推导或对照实验）。
  满足时不再要求逐项复现标杆：顺序、分块、舍入位置都可以按实现需要选择，只需保证数学
  过程不变，并按 deviation 记录标杆原实现、实现新选择、预期偏差量级与影响用例
  （实测值在 develop 补齐）。
  **同精度与降精度**：与标杆同精度时，顺序重排与分块方式变化只产生浮点尾差，允许；但不得
  增加舍入次数或提前舍入。精度低于标杆、增加或提前舍入、引入额外低精度落盘，按精度 bug 处理。
  **容差只用于判定**：只吸收数学语义一致后的浮点尾差，不得用于规避语义一致性，也不得在调试
  阶段放宽；判定不过时回 design 按误差分析重推容差并记录，重推仍不过按精度 bug 处理。
  本条不反向成立：精度更高不等于可以改变数学语义（如改变 softmax 的归一化定义、
  去掉 max 减法、丢掉 mask）。
- **被切开的 matmul 按整体处理**：标杆把同一 matmul 的 K 维（或 batch 维）按外部预算
  分段、逐段累加到同一输出时（如 `for seg: C.addmm_(A[seg].T, B[seg])`），design 阶段
  先做「分段判定」：段间除对 C 的累加外无其他读写、无跨段归约或非线性依赖、切分不改变
  数学结果。判定成立则按一个完整 matmul 实现——K 单链、最高可用精度（默认 fp32 L0C）
  累加、写回输出 dtype 时舍入一次，不得逐段复现低精度中间舍入或在 GM 上逐段读改写，
  并按「精度不降即可自由实现」记录 deviation（段数、标杆舍入次数、偏差量级、对照实验）。
  判定不成立（段间有归约/非线性/其他依赖，或用户要求逐段 bit 级复现）时按原分段语义
  实现并写明原因。
- **完整 matmul 的实现判据**：判定成立后必须同时满足（**本条优先于上文精度升档规则，
  精度更高也不例外**）——单个输出 tile 的全部 K 贡献在同一次 launch 内累加完成；
  累加器只在片上（默认 fp32 L0C），不得跨 launch 落在 GM
  （atomic add、GM 读改写、fp32 缓冲回读都算）；写回输出 dtype 时只舍入一次。按输出
  tile / block 划分任务不改变累加器归属，把同一输出 tile 的 K 贡献拆到多个 launch 合并
  一律按分段实现处理。中间量预算按设备可用 HBM 反推并写显存账，不得沿用保守默认值；
  预算取小造成的分段不算「完整 matmul 不可行」。放不下时按「提预算 → 改布局/复用 →
  实测不可行才 split-K 或跨 launch 累加」降级，后者必须在 deviation 单列依据与对照。
- **按结构选参考样例**：Cube-only → `tla/kernel/matmul/basic_mmad/basic_matmul.py`；
  Cube-Vector → `experimental/kernel/matmul/matmul_exp_scale_ub/matmul_exp_scale_ub.py`；
  Cube-Vector-Cube → `experimental/kernel/matmul/matmul_add_chain/matmul_add_chain.py`；
  纯 vector → `tla/kernel/vec/basic_vadd/basic_vadd.py`、`dsl_tutorials/vector_simd_ops/`、
  `dsl_tutorials/vector_simt_ops/`。以上只是常见结构的示例，不是固定的结构到样例映射；
  执行时须在参考仓 `examples/python/tla_dsl/` 与 `python/tla_dsl/tests/end_to_end/` 下全量
  检索，按目标算子的阶段序列、数据流与同步方式自行判断最契合的样例。整体结构
  优先：先选阶段序列、数据流、任务映射、同步方式都接近的主参考；主参考是基线不是约束，
  偏离写清理由。没有足够契合的主参考时，可以组合多个分步样例，但要显式设计整体结构并
  照常跑 skeleton DoD。路径以工作区 `.catlass-dsl/references/catlass/` 下 `find` 的
  结果为准。登记样例时同时登记当前环境可运行性（`passed` / `failed` / `not_run`）；
  主参考必须实跑，局部参考未直接复用其机制的可记 `not_run` 并写原因；主参考 `failed`
  必须写失败机制与改选方案，`remote-detached` 的 `not_run` 要附远端首跑命令。

## 自动执行

你是自动开发助手，默认自主推进，不把执行过程变成等待用户。

- **有方案就试**：遇到问题只要存在可行方案，就按方案逐步尝试；一个方案失败就换下一个，
  不要停下来找用户确认，直到解决或所有可行路径都确认真实不可行。
- **等待就等着**：需要等待外部事件（远端任务、编译、benchmark、设备释放、长跑用例）时，
  直接等待并轮询，不要结束回合，也不要用“进展汇报”代替继续执行。
- **失败不是终点**：测试失败或门禁 FAIL 只表示当前阶段未通过、不得进入下游，不表示本轮
  工作结束。存在可执行的诊断或修复路径时，继续定位、修复并重跑该阶段门禁；如实汇报失败
  不等于任务完成，不得以“已汇报”为由结束回合。
- **只在真阻塞时停**：仅当缺少只有用户能提供的信息（凭据、上板信息、无法自行判断的方案
  取舍），或可行路径已穷尽时才向用户提问，并列出已尝试的路径和失败原因。
- **优化不提前收口**：optimize 模式 B 按 2~3 轮执行（下限 2 轮、上限 3 轮）；不足两轮
  不得以收益小、没有思路或询问用户为由跳过，满两轮且归因第一名已被针对性攻过后可收口，
  不得为凑轮次做空转轮。唯一例外是 baseline 之后重算的 `skeleton_ratio ≥ 0.80`（详见
  optimize 的 Setup and baseline 第 5 步），此时 0 轮收口，且不改写 mode、不虚构用户门槛。

## 超时纪律

执行细则见「上板运行固定动作」。补充三条：

- 编译、算子运行、benchmark、profiling 与普通 shell 命令拆成独立命令块，一次只做一类动作，
  禁止串成一条长命令；编译单独成块、不套 60s（默认 `timeout 600s` 上限）；批量按 2~5 片
  逐段执行，分片命令用分片级超时、片内逐例 60s，禁止整批套一次性 60s 或裸跑。
- 超时、无输出增长或卡住时立即终止命令，不得死等或加大超时。
- 终止后先做卡健康判定：在同一张卡跑一个 60s 内的小探针或官方算子；探针能过就判定为本
  算子实现的死锁，直接转 debug 定位；探针也失败才判定为设备问题，换卡重试一次。

**重要！本地生成、远端调试时**，传输、编译、运行命令各自独立，不得用一条超时命令
打包；`timeout` 必须写在最终执行环境的命令内。

## 维护者通道（不参与算子流程）

知识库与版本基线的升级走 `skills/catlass-dsl-knowledge-upgrade`，入口见
[UPGRADE.md](UPGRADE.md)。该通道与上面的算子流程**互斥**：执行升级时不要加载
`catlass-dsl-workflow`，也不得用于设计、实现、调试或优化任务。

## 加载渠道

- OpenCode / Trae / Cursor / Codex / Copilot / CodeArts：本文件由安装脚本写入并自动加载。
- Claude Code：`init.sh project claude` 会生成镜像 `CLAUDE.md`。若通过插件市场安装，
  Claude 不会自动读取插件根目录的 `AGENTS.md`，此时由 `catlass-dsl-workflow` Skill
  承担入口约束；请确保它被安装并在首轮加载。

## 语言

始终使用中文与用户交流。
