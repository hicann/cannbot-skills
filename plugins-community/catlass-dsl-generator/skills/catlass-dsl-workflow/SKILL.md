---
name: catlass-dsl-workflow
description: Use when the user asks for any CATLASS DSL operator task (design, implementation, debugging, benchmark, optimization, knowledge) or mentions catlass.tla, to route stages and apply the gate-checked workflow rules before any other catlass-dsl skill runs. 当用户提及 CATLASS DSL 算子开发、实现、复现、调试、精度对齐、benchmark、优化或上板时，必须先加载本 Skill。
---

# CATLASS DSL 工作流入口（门禁引导）

本 Skill 是 `catlass-dsl-generator` 插件的唯一入口。**本版本不提供状态机**：
`develop_state.py`、`optimize_state.py`、`bench.py`、`verification.py`、
`validate_reference.py`、`record_knowledge.py` 均已移除。阶段推进和证据核对由本文件、
目标阶段 `SKILL.md` 与只读门禁脚本 `scripts/gate.py` 共同承担；门禁不保存状态文件，
只校验工作区产物和 `ITERATIONS.md` 记录。

唯一不可放松的要求：**任何“通过”都必须对应真实执行记录**。没有命令、退出码和输出文件的
结果一律视为未执行，写 `blocked` / `not_run`，不得写成通过。

## 1. 启动检查（每个新任务的第一条回复前）

依次完成：

1. **定阶段**：按 §2 路由表确定唯一阶段；无法归属时先向用户确认。
2. **定执行模式**：按 §3.1 确定 `remote-detached` / `remote-connected` / `local`，只依据
   用户说明或工作区已有声明，禁止本地探测环境；无法判断时先向用户确认一次。
3. **读全文**：完整读取目标阶段 `SKILL.md`，禁止只读摘要或标题。
4. **查产物**：确认工作区已有文件（`DESIGN.md`、`reference.py`、`definition.json`、
   `workload.jsonl`、`workload_dev.jsonl`、主实现文件、`solution.json`、`ITERATIONS.md`）。
   以文件系统为准，不凭对话记忆续跑。
5. **定 DSL 版本**：用户提示或任务文件已声明时直接采用；否则读取当前 Python 环境已安装
   版本（`importlib.metadata.version("ascend-catlass-dsl")`）。本地未安装时，可交互则先向
   用户确认一次；无法交互时默认 `2.2.0`。目标版本写入阶段卡与 `DESIGN.md`；远端实际版本
   不一致时先停下确认。
6. **刷新参考源码**（涉及 design / develop 的任务才执行）：删除已存在的
   `.catlass-dsl/references/catlass`，按 §6 重新下载并解压 master 归档包，确保本次任务
   使用最新 catlass 样例。下载结果必须是 ZIP；若得到 HTML 页面或空文件，视为下载失败，
   记录原因并用知识库兜底，不得把残留旧目录当作最新 master。
7. **跑门禁**：执行 `python3 <SKILL_DIR>/scripts/gate.py status --workspace <项目根>`，
   按输出定位最上游未通过项；该阶段就是当前阶段卡里的阶段。门禁失败原因必须原样带进
   阶段卡，不得省略。
8. **输出阶段卡**，然后才允许执行动作：

   ```text
   📋 阶段卡
   阶段：<design | develop | bench | optimize | knowledge>
   依据：<SKILL.md 绝对路径> ｜ 关键约束：<该文件中的一条原文>
   DSL 版本：<目标版本 + 来源：用户指定 / 本地已安装 / 默认>
   执行模式：<remote-detached | remote-connected | local>（来源：用户说明 / DESIGN.md）
   工作区：<项目根> ｜ 已有产物：<列出实际存在的关键文件>
   门禁：<gate.py status 的输出摘要，或未通过项与原因>
   下一步：<该阶段第一个必做动作>
   ```

9. **建立或更新里程碑日志**：工作区没有 `ITERATIONS.md` 时创建，表头为
   `| Iter | Stage | Kernel SHA256 | Command | Result | Evidence |`。后续每次阶段推进
   都追加一行真实记录。

## 2. 阶段路由

| 用户意图 | 唯一 Skill | 前置产物 |
|---|---|---|
| 设计新算子、写规格、reference、workload | `catlass-dsl-design` | 无 |
| 按设计实现、调试、测试、交付 | `catlass-dsl-develop` | `DESIGN.md`、`reference.py`、`definition.json`、`workload.jsonl` 齐备 |
| 独立正确性校验与性能测量 | `catlass-dsl-bench` | 已有 `solution.json` / `workload.jsonl` / `definition.json` |
| 已通过算子的性能优化 | `catlass-dsl-optimize` | 已有可运行的实现、`solution.json` 与冻结的 `workload_dev.jsonl` |
| 知识查询与记录 | `catlass-dsl-knowledge` | 无 |

- 一次只在一个阶段工作；阶段切换由实际产物决定，不因“感觉可以了”或用户催促而切换。
- 用户未指定阶段时，选最上游未满足的产物：无 `DESIGN.md` → design；有设计未交付 →
  develop；已有交付 → bench / optimize。
- 禁止在一次回复里跨阶段执行；禁止在 develop 里重做设计与 reference；禁止在 optimize
  里改语义、容差或计时口径。

## 3. 阶段推进（带门禁）

主流程固定为：

```text
design → develop → full test #1 → optimize → full test #2 → [benchmark] → finish
```

`benchmark` 是条件阶段：`DESIGN.md` 的 `性能要求：requested` 时执行，`none` 时跳过并直接
finish。

每一步进入前必须先跑对应门禁，未通过不得继续；跳步、合并步骤或“先做完再补记录”一律
视为无效执行：

| 门禁 | 命令（`<ws>` 为工作区根） | 强制要求 |
|---|---|---|
| design | `python3 <SKILL_DIR>/scripts/gate.py design --workspace <ws>` | 五件交付物齐备；definition/workload 合法；dev 子集总数 ≤40（用户用例 ≥30 条时为 30~40 条；不足 30 条时全选，不要求补齐；无用户用例时按覆盖需要自定，不以下限 30 为目标）；DESIGN.md 记录三个文件的 SHA-256；声明执行模式与性能要求 |
| develop | `python3 <SKILL_DIR>/scripts/gate.py develop --workspace <ws>` | design 通过；skeleton 与 focused test 记录；`solution.json` 与实现文件一致 |
| fulltest | `python3 <SKILL_DIR>/scripts/gate.py fulltest --workspace <ws>` | 当前 kernel 的 full test passed 记录与证据 |
| optimize | `python3 <SKILL_DIR>/scripts/gate.py optimize --workspace <ws>` | full test 通过；模式 A ≥1 轮或模式 B ≥3 轮的完整账本 |
| benchmark | `python3 <SKILL_DIR>/scripts/gate.py benchmark --workspace <ws>` | 当前 kernel 有全量测试 passed（优化改动了 kernel 时须为 full test #2）；`性能要求：requested` 时还要求当前 kernel 的 benchmark passed；`none` 时跳过 benchmark 与 profiling |
| finish | `python3 <SKILL_DIR>/scripts/gate.py finish --workspace <ws>` | `性能要求：requested` 时 benchmark 已通过；final review passed；Final Verdict 存在 |

`gate.py` 是只读脚本：只检查产物和 `ITERATIONS.md`，不执行算子，也不写状态文件。
`PASS` 只代表记录齐备，结果真实性仍由 §4 约束；`FAIL` 是硬阻断，不得绕过或手改记录蒙混。

### 3.1 执行模式与推进上限

执行模式在启动检查第 2 步确定，写入 `DESIGN.md` 的 `执行模式：` 行；`gate.py` 只读这一行：

| 模式 | 适用场景 | 推进上限 |
|---|---|---|
| `remote-detached` | 本地生成，当前拿不到远端连接，也没有本机 NPU | 只能推进到 develop 的代码生成完成；focused test 记 `deferred`，之后必须先连上远端 |
| `remote-connected` | 本地生成，远端调试环境可连接 | 全流程真实执行，不允许 `deferred` |
| `local` | 本地生成、本地调试（本机有 NPU） | 全流程真实执行，不允许 `deferred` |

- 判定只依据用户说明或 `DESIGN.md` 已有声明，禁止在本地跑 `npu-smi`、导入 `torch_npu`
  或做任何环境探测；无法判断时先向用户确认一次。
- `remote-detached` 下 develop 门禁可以在代码生成完成后通过：`skeleton` 记 `blocked` /
  `not_run`（附编译与最小 shape 的精确重跑命令），focused test 记 `deferred` 并在 `Command`
  列写远端恢复后可原样执行的精确命令；缺命令按 FAIL 处理。此时工作到此为止，必须等待连接，
  不得继续 full test 或 optimize。
- full test #1 及之后的所有阶段都不接受推迟，门禁要求真实 `passed` 记录与证据；连接
  恢复后按 `Command` 列补跑 focused test，把该行改为 `passed` 再继续。
- `deferred` 只表示“代码已生成、验证未执行”，不得用来掩盖失败；真实执行后失败仍记
  `failed`。`design` 与 `finish` 永不豁免，`gate.py finish` 见到任何 `deferred` 行即 FAIL。

### design → develop

- 门禁 `design` 通过；
- `DESIGN.md`、`reference.py`、`definition.json`、`workload.jsonl`、`workload_dev.jsonl`
  存在且内容自洽；
- reference 已在真实环境（远端服务器或本机 NPU 机器）按 L2 口径跑过（能跑全 shape
  就跑完；算力不足时用小 shape 冒烟，并记录 `blocked` 与精确重跑命令）。纯本地生成模式
  下本地不代跑，记 `blocked` / `not_run` 并附精确命令后即可进入 develop；
- `DESIGN.md` 写明目标 DSL 版本、基准命令、指标路径与 required cases。

默认直接进入 develop，不等待用户批准；只有用户明确要求先评审时才暂停展示。

### develop 内部顺序

```text
skeleton → implement → focused test → full test #1 → optimize →
targeted review（high risk）/ final review → full test #2 → [benchmark] → finish
```

- `benchmark` 只在 `DESIGN.md` 声明 `性能要求：requested` 时执行；`none` 时当前 kernel
  全量测试通过后直接 finish，不采集性能，也不写 benchmark 里程碑。
- 每个里程碑先在 `ITERATIONS.md` 追加一行，再进入下一步；写实现文件前至少已有
  `skeleton` 记录（可运行则记 `passed`，无 NPU 则记 `blocked` / `not_run` 并附重跑命令）。
- 代码变化后，受影响的测试、评审与性能结论需要重新评估；交付只认当前 kernel 的全量测试
  与 final review。优化全部未采纳、kernel 未变化时 `full test #1` 继续有效；采纳改动导致
  kernel 变化时必须补跑 `full test #2`，优化轮次作为过程证据保留，不要求重跑。
- 全量测试至少跑一次：`full test #1` 在 optimize 之前确认功能正确，绑定 optimize 起始
  kernel；`full test #2` 只在 optimize 采纳改动、kernel 变化时补跑，绑定交付 kernel。
  `finish` 前必须有当前 kernel 的全量测试 passed 记录：优化采纳改动后必须补跑
  `full test #2`，全部未采纳、kernel 未变化时可复用 `full test #1`。`ITERATIONS.md` 的
  `Stage` 列按实际记录写 `full test #1` / `full test #2`，门禁据此识别。`remote-detached`
  下需要执行的每一次全量测试都必须等连接远端后真实执行，不得推迟。
- `finish` 前必须对**当前源码**完成一次全量测试与最终评审；声明 `性能要求：requested`
  时还要完成配对 benchmark 和 profiling。`Final Verdict` 必须记录交付 kernel SHA-256
  （与当前源码及 `solution.json` 一致），引用的最终证据路径必须真实存在；交付 kernel
  与 optimize best kernel 不同（例如 review 修复）时写明差异。
- optimize 采纳改动后，当前 kernel 的 `full test #2` 可同时作为 develop / fulltest 门禁的
  完成证据；kernel 未变化时 `full test #1` 继续作为交付证据，不要求对最终 kernel 重跑。

### 测试与性能口径

- full test 使用完整 `workload.jsonl`，必须分 2~5 片并行执行；全部 required cases 逐例
  通过后才算通过。失败用例一次收集、批量修复、逐例复验后再重跑整轮。
- benchmark 使用完整 `workload.jsonl`，逐例配对 candidate/reference；正确性已在 full test
  覆盖时可跳过逐例校验，但必须在结果中引用 full test 的证据路径。
- optimize 使用 design 冻结的 `workload_dev.jsonl`（用户用例 ≥30 条时为 30~40 条；不足
  30 条时全选；无用户用例时按覆盖需要自定，总数 ≤40），
  所有轮次共用同一份样本。

## 4. 证据真实性

- `passed` 必须能指向实际执行的命令、退出码和输出文件；环境缺失写 `blocked` / `not_run`。
- 禁止手写、推测或复制旧结果冒充本轮执行；无法复现的结果按未通过处理。
- 关键参数（kernel SHA-256、definition/workload 摘要、命令、设备、CANN/DSL 版本）
  记录在 `ITERATIONS.md` 与结果文件中。
- `finish` 前随机抽 1~2 个用例重跑，确认结果可复现；重跑失败则回到 debug。
- 门禁 `gate.py` 只核对产物与记录，不代替真实执行；发现 FAIL 时补跑真实命令并补记录，
  不得手写凑数行或改 kernel SHA 蒙混通过。

## 5. 执行纪律

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合，
> 也不用进展汇报代替继续执行。仅当缺少只有用户能提供的信息、或可行路径已穷尽时才提问。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块，
> 一次只做一类动作，禁止串成一条长命令。编译独立 300s，单例运行 `timeout -k 5 60s`，
> 批量按分片逐段执行，禁止整批套一次性超时，禁止裸跑。超时、无输出增长或卡住时立即终止
> 命令，不得死等或加大超时。
> **重要！本地生成、远端调试时**，不得用一条超时命令打包传输、编译与运行；传输、编译、
> 单例运行各自独立，运行算子逐例拆成单条命令并加 `timeout -k 5 60s`。
> 终止后先做卡健康判定：在同一张卡跑一个 60s 内的小探针或官方算子；探针能过就判定为本
> 算子实现的死锁，直接转 debug 定位；探针也失败才判定为设备问题，换卡重试一次。

- **控温只在 optimize 阶段执行**：同用例内各实现温差 ≤3℃，执行前 AICore 频率须为
  1650 MHz，温度超过 85℃ 停机；full test 与 benchmark 不做控温。
- **optimize 轮次**：有明确性能目标时（模式 A）至少完成 1 轮；没有目标时（模式 B）至少
  完成 3 轮并逐轮写入 `ITERATIONS.md` 的 `## Performance Rounds`。禁止以“已收敛”“收益
  不大”或询问用户换取提前收口。optimize 只能在 full test #1 通过后开始，结束后必须确认
  当前 kernel 有全量测试记录（kernel 变化时补跑 full test #2）；`性能要求：requested`
  时再进入 benchmark。
- **性能采集看用户要求**：`DESIGN.md` 的 `性能要求：none` 表示用户没有明确提性能，
  当前 kernel 全量测试通过后直接 finish，不跑 benchmark、不做 profiling，也不得自行补测；
  `requested` 才按 §5 口径采集性能并作为交付证据。
- **先生成、再推进**：不论环境在哪，都先完成设计与代码生成，再进入验证。一开始就拿到
  服务器信息，或本来就在服务器上，同样要先生成文件再推进；不得先做环境探测、装环境、
  跑标杆或其它准备动作。
- **本地生成、远端验证**：本地只负责设计与代码生成。reference（标杆）运行、静态检查、
  模拟/打桩检查、编译、上板、精度、benchmark、profiling 一律留到远端真实环境执行，
  本地一件都不做；不得在本地安装或配置 CANN / CATLASS DSL。远端暂不可用时记 `blocked` /
  `not_run` 并保留精确命令，环境就绪后直接在真实环境执行，不重建代码。
- 缺 NPU / CANN / CATLASS 环境时记 `blocked` / `not_run`，不得写成通过。

## 6. 参考与产物

建议的工作区布局：

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
├── evidence/                 # 每阶段命令输出、结果 JSON、profile CSV
└── .catlass-dsl/references/catlass/   # catlass master 样例
```

- 文档中的 `<SKILL_DIR>` 指当前 `SKILL.md` 所在目录，跨 Skill 引用写成
  `<SKILL_DIR>/../catlass-dsl-<name>/...`。
- **本地 API 与样例参考**：以启动阶段冻结的目标版本为准，catlass 源码统一从 master
  归档包获取，不使用 git clone：

  1. 若 `.catlass-dsl/references/catlass` 已存在，先删除整个目录，不复用旧内容；
  2. 下载 master 归档包
     `https://raw.gitcode.com/cann/catlass/archive/refs/heads/master.zip`。该地址会先返回
     302 跳转到对象存储的签名地址，下载器必须跟随重定向；不要改用
     `gitcode.com/cann/catlass/archive/...`，那是仓库网页而不是归档包；
  3. 下载完成后先校验产物是 ZIP：文件头应为 `PK`，大小应为几十 MB 级。若文件以
     `<html` 开头或只有几百字节，说明没有跟随重定向、下载到了网页，必须删除并从第 2 步
     重试，不得直接解压；
  4. 解压到临时目录（路径尽量短；Windows 上过深的路径会触发 260 字符限制），再把解压
     出的内容移动到 `.catlass-dsl/references/catlass/`，使最终样例位于
     `.catlass-dsl/references/catlass/examples/...`；
  5. 确认该目录存在且包含 `examples/python/tla_dsl`。下载或解压失败时记录原因，继续用
     知识库兜底，不得把残留旧目录当作最新 master。

  下载、解压和移动的具体命令按当前平台选择，只要求每次任务启动都重新获取最新 master。

  样例分布在 `examples/python/tla_dsl/` 下的 `tla/kernel`、`experimental/kernel`、
  `dsl_tutorials`、`helpers` 四类目录，挑选参考前先全量浏览（例如
  `find <refs>/examples/python/tla_dsl -name '*.py' | sort`），避免只看顶层或单个子树
  而漏掉 `experimental/kernel/matmul/...` 这类深层样例。
- 用户指定标杆代码并禁止参考兄弟目录时：标杆文件、其用例文件与 catlass 参考源码仍可读，
  禁止扫描同工程其他算子目录。
- 用户暂未提供上板信息时：design 与代码生成照常推进，验证与性能记 `blocked` /
  `not_run`，待信息补齐后重试。

## 7. 用户要求暂停评审或跳过步骤

- **先评审设计**：design 完成后停止并展示 `DESIGN.md` 与 reference 结果；用户回复
  “继续”即进入 develop。
- **跳过步骤**：说明影响与风险，取得用户明确书面确认，并在 `ITERATIONS.md` 记录跳过的
  步骤和理由。证据真实性、环境缺失标记和最终全量验证不允许跳过。

## 8. 每轮回复自检

```text
🔎 流程自检 ｜ 阶段：<...> ｜ 依据：<SKILL.md 约束原文> ｜ 本轮执行：<命令或文件> ｜ 证据：<路径> ｜ 下一步：<动作>
```

## 9. 语言

始终使用中文与用户交流。
