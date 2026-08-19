# ascendc-port-orchestrator — 架构设计（cannbot 插件）

> 本文档描述 **cannbot 插件 `ascendc-port-orchestrator`** 的架构，**不是** a5_ops harness 的文档。随插件维护、低频更新。
> 插件对外仅提供两项能力：**① 跨代际算子移植（当前 arch22→arch35）② 正向→反向生成**。其余能力不在本插件对外范围。

## 1. 总体结构

```
用户（自然语言指定目标架构/产品）
        │
        ▼
┌─────────────────────────────────────────────┐
│ 两个入口 skill（薄 NL 前端）                    │
│       ① cross-gen-port ② backward-gen            │
│  - 解析 NL 目标 → canonical target              │
│  - 来源架构由代码分析自动识别                    │
└───────────────┬─────────────────────────────┘
                ▼  （底层同一编排器，保持完整能力）
┌─────────────────────────────────────────────┐
│ 确定性流水线（FSM）  +  安全网（gates）           │
│        ▲                                         │
│        │  按需注入 / 生成后沉淀                    │
│ 反馈环：用户本地 KB(c) > 插件自带 KB(b) > 社区 skills(a) │
└─────────────────────────────────────────────┘
```

## 1.1 执行模型（强制约束：bundle-orch）

- **编排器引擎打包进插件**：FSM + 安全网 + 迭代到绿 + 子 agent 调度的**完整编排器引擎随插件交付**（`plugins-community/ascendc-port-orchestrator/` 内自带 scripts + engine），**不依赖外部 a5_ops checkout**。
- **入口 skill = 薄壳**：两个入口 skill 解析目标后**调用打包进来的编排器**；由编排器亲自驱动 FSM —— **不是**让 AGENTS.md 这个 primary agent 用自然语言临时编排各阶段。确定性（状态机/钩子/迭代上限）来自引擎，NL prose 复刻不了。
- **运行时**：双 harness（Claude Code / opencode），引擎经 `backends/` 统一抽象拉起子 agent，`AOG_HARNESS_BACKEND` 切换、两套互不依赖（见 §8）。
- **引擎布局**：编排器保留平铺模块布局；harness 适配器位于 `engine/src/scripts/orchestrator/backends/`，不依赖目录重构。
- **强制一致性**：`AGENTS.md` / `install.sh` / `plugin.json` 必须与本模型一致（编排器打包 + 入口调用编排器）。**若 AGENTS.md 出现「自包含 NL 方法论执行、不调用编排器（orch-less）」的描述 = 与本文档冲突，以本文档为准，须改回 orch-shell。** 任何改动先改本文档、再改实现（§11）。

## 2. 确定性流水线（FSM）

状态机驱动、可中断/恢复/复现。每阶段产出机器可读状态。两项能力共享流水线，差异在「迁移/反向生成」阶段。

| 阶段 | 职责 | 关键产出 |
|---|---|---|
| Parse / 配置 | 解析入口意图 + 目标(NL→canonical) + 来源(代码分析自动识别) | 任务配置（source-arch, target-arch, mode） |
| 分类 | 算子族分类、确定性策略分类（required/best-effort/n/a） | op-family, det-policy |
| 参考/真值 | 移植：获取来源参考；反向：由正向生成精确梯度真值 | reference / golden |
| **迁移 / 反向生成** | 移植：跨代际生成目标算子；反向：生成反向算子 | 目标 AscendC kernel |
| 构建 | 在目标架构上编译 | build artifact + provenance |
| 精度验证 | 与真值比对（分层阈值 + 特殊场景） | precision verdict |
| 性能优化（可选） | 按 roofline/profiling 调优、再验精度 | perf ratio |
| 报告 | 汇总精度/确定性/性能 + 复现指引 | REPORT + REPRODUCE |

状态以机器可读形式持久化（state transitions），支持中断恢复与失败重入（如 `--optimize` 再入）。

## 2.1 分层执行模型：确定性 FSM 骨架 + LLM 驱动的 step 子 workflow

> owner 2026-07-02 提出的核心架构点，写入本文档（不止 PPT 用）。

本插件流水线是**分层**的，确定性与 LLM 的边界清晰：

| 层 | 谁驱动 | 机制 | 保证 |
|---|---|---|---|
| **顶层（编排）** | 确定性 | Python FSM：`opgen_state_machine.yaml`（机器可读规格）+ `state_machine.py` + `orchestrator.py`；状态、`exit_transitions`、迭代上限由代码强制 | 可复现、可中断/恢复、无漂移 |
| **中层（phase O0–O6）** | 确定性 | 每个 phase 是代码化的门(gate)：O0 hook 完整性、O2.5 参考已就绪、O5 验证不过不 finalize；`state_executor.next_agent` 确定性选下一个 agent + 确定性建 brief + 确定性 gate 输出 | 安全网、防作弊、质量门 |
| **step 子 workflow** | **LLM** | 在某个确定性 phase 信封内，被 spawn 的 agent（kernel-worker / precision-probe / …）做 LLM 驱动的生成（分析→写码→构建→验证）。LLM 负责「怎么生成」，但「何时 spawn / 给什么 brief / 输出是否放行」由上·中层代码定 | 生成灵活性 + 被确定性外壳约束 |

一句话：**确定性的骨架（控制 + 门禁）+ LLM 的血肉（每 step 的生成）**。骨架是**可执行代码（FSM）**，不是提示词。

### 与 cannbot 原生「workflow」定义的区别

cannbot 生态里「workflow」的官方定义（`docs/STANDARDS.md §目录结构规范`）＝ **`workflows/` 下的流程配置文件：任务提示词、数据流定义、错误处理指南**。一个真实例子（`catlass-op-generator/workflows/task-prompts.md`）自称是 **「CANNBot 调用各阶段 Subagent 的唯一执行手册」**——每个 Step 写好 subagent 调用参数、关联 skill、完成校验、约束提醒，**由顶层 CANNBot（一个 LLM）读这份 prose 手册、逐 step 决定调哪个 subagent、检查完成、进下一步**。

| 维度 | cannbot 原生 workflow | 本插件流水线 |
|---|---|---|
| 顶层编排的驱动 | **LLM**（读 `task-prompts.md` prose 手册） | **确定性 Python FSM** |
| step 排序 / 门禁 | LLM 按 prose 解释执行（可漂移） | 代码强制状态转移 + gate + 迭代上限 |
| 「workflow」是什么 | **提示词工件**（markdown 手册 / 模板） | **可执行代码**（FSM + gates + critic） |
| step 内生成 | LLM subagent | LLM agent（同） |
| 确定性来源 | 靠 LLM 遵守 prose | 靠代码（LLM 骗不过 FSM / gate / cap） |

**本质**：cannbot 原生 ＝ **LLM 编排 + LLM 执行**；本插件 ＝ **确定性代码编排 + LLM 执行（每 step）**。两者都把 op-gen 拆成 step + subagent，但 **step 的排序 / 门禁 / 上限**在 cannbot 是 LLM 解释的 prose、在本插件是机器执行的 FSM。**改流水线 ＝ 改 FSM**（`opgen_state_machine.yaml`），不是改一份提示词手册。

**这就是本插件 bundle 一个 engine（FSM 是代码）、而非像其它 cannbot 插件只发 `workflows/*.md` 提示词的原因**：可复现 / 安全 / 防漂移的硬保证来自代码 FSM，不来自 LLM 对 prose 手册的遵守——LLM 会漂，Python FSM 不会。（注：本插件根的 `workflows/development-guide.md` 是**描述性**文档、给人读的，不是真正的驱动；真正的驱动是 engine 里的 FSM，别把它当 cannbot 那种 prose-workflow。）

### 2.2 跨代际移植的参考来源

`--port-a3` 是 `opgen_mode=port_a3_to_a5` 的唯一移植入口：它提供要分析和移植的 arch22 源算子。
O2.5 必须在当次任务中采集来源 A3 的实时 CANN 输出，O5 使用该输出走既有 two-tier 精度验证；目标实现则在
独立 A5 环境构建和测量。因此跨代移植需要可用的 A3 与 A5 配置，并将参考来源、输入及运行证据写入状态和报告。
离线包或其它参考来源不属于当前实现，不能冒充实时 A3/A5 结果。

## 3. 插件结构

```
plugins-community/ascendc-port-orchestrator/
├── .claude-plugin/plugin.json   # 注册清单（agents[] 指向 agents/*.md）
├── AGENTS.md              # 主编排 agent（两个入口的系统提示词 + skills 列表）
├── quickstart.md
├── init.sh                # 安装（建用户侧 KB(c) 根 + scaffold .ascendc_env）
├── agents/aog-*.md        # 11 个客户子 agent（plugin.json 注册）
├── skills/                # 插件内置运行时 Skill（15 个，Claude 自动发现）
├── hooks/                 # SessionStart 上下文注入（run-hook.cmd + session-start 脚本）
├── workflows/             # 描述性开发指南（真正驱动在 engine FSM，见 §2.1）
├── docs/ARCHITECTURE.md   # 本文档
├── engine/                # bundle-orch：编排引擎（FSM + gates + KB + scripts + 子 agent 定义源）
└── kb/                    # 插件自带 b 层 KB（含保留的 OKF 结构）
```
> 插件结构沿用 `agents/` 扁平 + `hooks/` + `init.sh` 约定；当前作为社区插件交付。`engine/` 打包是本插件独有的 bundle-orch 设计（§1.1）。
两个具名入口 skill：`ascendc-cross-gen-port`（跨代际移植）、`ascendc-backward-gen`（正向→反向）；各自解析目标、调用同一编排能力（orch 保持完整）。

## 4. Agent 角色

| agent | 职责 | 触发 |
|---|---|---|
| kernel-worker | 分析+生成+构建+验证（首发，含内层编译/精度修复循环） | 每任务首先 |
| precision-probe | 精度卡死时的最小复现 + bit-diff 根因 | 精度连续卡 |
| kernel-optimizer | ratio<阈值 时的 msprof + 性能调优（每次改后必复验精度） | 性能不达标 |
| fused-optimizer | 可分解为子算子的融合算子专用优化 | optimizer 在融合算子上 plateau |
| determinism-analyzer | det-policy=required 且观测非确定时的根因 | 确定性不满足 |
| researcher | 末位结构性探索 | 前述均 plateau |
| hardware-probe | 经验性硬件/编译行为探针（产出探针报告，非生产算子） | 需 HW 行为确证 |

（agent 集合随能力范围裁定；上表为流水线核心角色。）

## 5. 反馈环 · 双层 KB + 社区 skills 知识源

### 5.0 两层 KB 与社区知识源（cannbot 特殊形态）

原 a5_ops 单体 KB 经导出被**切成三部分**（converter + coverage gate 保证无损、可证）：
- **被删的一部分（→ 不进插件）**：与 cannbot 已有 skills **重合**的知识，dedup 后**从插件 KB 删除**——因为 cannbot skills（下面的 a 层）已经提供；保留=重复。判据：`coverage_gate.py` 用一张 **curated 映射表**（unit → 对应 cannbot skill 名 + TRIM 处置）+ 校验该 **skill 文件存在**（`cannbot_skill_exists`）。⚠️ **已知弱点（codex 审出）**：现 gate 证的是「映射的 cannbot skill **存在**」，**不是**「被删知识的**内容**真被该 skill 覆盖（主题词共现）」——存在 ≠ 实质覆盖。**待修**：把删除判据升级为内容级覆盖证明（复用 resolver 的主题词共现判定）。注：**主题词共现是 `cba_resolver.py` 的运行时取知识判定**（§5.2），与本静态 gate 是两个机制，先前文档把二者混为一谈、已更正。
- **(b) 插件自带 KB**：a5-unique **减去** cannbot 已覆盖的部分 = 插件真正要带的硬件/编译/平台经验（OL/EC/PB/P-P/hardware/fa-class，skill 格式 + 路由）。
- **(c) 用户本地 KB**：用户侧修正/增量。

运行期三源、优先级 **c > b > a**（仅同主题冲突时按优先级裁，否则叠加）：
- **(c) 用户本地 KB**：运行时**可写**，最高优先（覆盖默认）。
- **(b) 插件自带 KB**：随插件交付，运行时**只读**。
- **(a) 社区 skills**：cannbot 现有 skills（被删部分的知识从这里取）。

### 5.1 用户侧 KB 格式 / 存放 / 索引（本期设计）
- **格式**：每条目 = 一个带 frontmatter 的小文件（`id` / `topic`(canonical-topic-key) / `applies_to`(架构·算子族 scope) / `provenance`(用户手写 | 流水线沉淀+时间戳) / 正文）。
- **存放位置**：用户主目录下、**与插件交付物分离**的固定路径（如 `~/.ascendc-port/user_kb/`，运行时可写；插件目录 `references/` 只读、不写）。位置可由 env 覆盖。
- **索引**：目录内维护一个 `INDEX.md`（canonical-topic → 条目文件 的路由表），新条目按 topic 归位 + 自动更新 INDEX 行；resolver 按 topic 查 INDEX 命中条目。
- **维护**：运行时由流水线「生成后沉淀」追加/更新（见 §5.3）；用户可手工增删改。

### 5.2 三源取知识机制（READ）
- **resolver**：`cba_resolver.py`（已实现+单测）按 canonical-topic 在三源里判覆盖（distinctive 主题词**共现**=实质覆盖，非偶然提及），返回**最高优先且覆盖该 topic 的层**（c→b→a），或 NOT_FOUND。
- **注入子 agent**：编排器据 op 分类把相关 topic 的命中知识**注入子 agent 的 brief**；同时 install 把 cannbot skills + 插件 KB 装成 CC skills，**子 agent（编排器拉起的）天然能 invoke 已装的 cannbot skill**（skill registry 对 sub-agent 可见——已实测）。即：b/c 经 brief 注入，a 经 CC skill registry 可达。
- **provenance 校验**：子 agent transcript 记录 invoke 了哪个 cannbot skill / 读了哪条 KB（用于验证三源真被用，见 §6 + differential A/B）。

### 5.3 写回机制（WRITE）—— 只写用户 KB(c)、不改插件 KB(b)
- **触发**：生成闭合（精度 PASS）后，「生成后沉淀」阶段把本次新经验（新 EC/OL 类条目）写入**用户本地 KB(c)**。
- **去向唯一**：写**只**落在 §5.1 的用户 KB 路径；**插件目录 `references/`（b 层）运行时绝不被写**（只读挂载/路径隔离 + 写入函数硬编码用户 KB 根）。这是「更新用户侧、不动插件侧」的硬保证。
- **索引同步**：写条目的同时更新用户 KB 的 `INDEX.md`（topic 路由行）。
- **冲突**：同 topic 已存在则按 c>b>a 以用户本地为准（新沉淀标 provenance + 时间戳，便于回溯/人工裁决）。
- **状态（实现）**：见 §5.5（本 session 诚实修正）。

### 5.4 统一分层契约（跨 cannbot / a5_ops / npu-autoport — owner 2026-07-02 统一设计，main 主持）

三个项目共同面临 user-KB + official-KB 分层问题；owner 定为**一个统一可行方案**、三方分别落入各自设计文档、**设计必须一致**。本节 = cannbot 侧对齐（参考实现 = CLaptopScan `poc_kb_layers.py` + DESIGN §4「KB resolver 契约」`580851b`；统一 §tiering spec 由 main 综合、待出）：

- **Resolver 接口**：`resolve(query, [c,b,a], reset_to_official=False)`（默认 `[c,b,a]`/reset `[b,a]`，语义签名匹配）；`make_entry`/`admit_to_c`/`dedup_semantic(thr=0.6)`。**写 c 还是 b 由 cannbot 侧 `kb_write_root()` 决定**（config-driven：`ASCENDC_PORT_USER_KB`/`~/.ascendc-port/user_kb`→c，否则 b；a5_ops 默认 b 行为不变）。
- **Namespace / ID（核心不变量）**：c = `customer:{硬key哈希}`（内容哈希、同教训同 ID 幂等）；b = 引擎序号 `OL-N/EC-N/PB-N/P-Pxx`（我方发版控制、客户只读）。**c/b 不同 namespace + 客户只写 c → 升级不撞**。
- **准入 gate（写入端，≠优先级）**：`correctness` 需 `trust=verified` 才进 c；`site_config`/`experience` 自由进 c。
- **硬 key/去重**：signature = 错误码 `\d{5,6}` + snake_case 符号；dedup = 硬 key jaccard ≥ 0.6；写策略 = 泛化 + 写前去重。
- **Merge c→b**：工具 + owner-gated + 发版时；硬 key 语义去重；**重分配 ID（c 哈希→b 序号，别塞哈希进 b KB_INDEX）**；再验证 + 墓碑 + 迁移防悬空。
- **读路径**：c>b>a；c 影子盖 b 同-key；`reset-to-official` 跳 c。
- **KB_INDEX 跨-tier 不变量**：c/b 各自 namespace+index，orphan-free 每 tier 各自成立；merge/b 升级时迁移 c→b 引用防跨-tier 悬空。

### 5.5 实现状态（2026-07-02 诚实修正 — impl≠design gap 本 session 查出）

- **READ**：`cba_resolver.py`（§5.2 主题词共现）已实现+单测；但**引擎 op-gen 主流程 KB-load 是否真按 c>b>a 读 user_kb 待验**（引擎 `src/scripts` grep user_kb = 0 条）。
- **WRITE c-tier = 未实现（GAP）**：现状 = **a5_ops 继承的引擎沉淀 ACTIVE 写 b-tier**——finalize `kb_invoke` spawn `aog-knowledge-maintain` → 写打包 `references/`（**e2e 实证：`OPERATIONAL_KNOWLEDGE.md +26` 进 bundled b**）；`kb_auto_promote --kb-root` 默认也 b。install.sh 建了 c-root 但无写路径填充。
- **后果**：分发式升级盖用户学习 + 序号撞（= §5.4 要解、= main 的 DEBT-178）。
- **修复（3 处）**：① `aog-knowledge-maintain` 写 c（`kb_invoke` prompt 注 c-root）② `kb_auto_promote --kb-root` config-driven（`kb_write_root()`）③ KB-load 读 c>b>a + KB_INDEX 跨-tier。
- **归属（owner 2026-07-02）**：并入 **main 主持的统一 KB 设计**；cannbot 消费统一 outbound，并负责 `kb_write_root()`、插件激活语义与 outbound 消费。

## 6. 安全网（gates）

防作弊/防退化，保证产出可独立调用、真在目标 NPU 跑通：
- **钩子完整性门**：运行前校验编排钩子未被篡改。
- **前置检查**：环境/源完整性。
- **来源核验（provenance）门**：构建/精度产物来源可核（禁 CPU 替代 NPU、禁直接抄源码实现、二进制/编译产物来源核验）。
- **自检清单**：生成后按 catalog 自检（防「看起来过了但不可独立调用」）。

### 6.1 设计原则：保交付质量、不阻碍生成
- **门设在边界、不打断生成内循环**：钩子完整性 + 前置检查在**生成前**一次性过；provenance + 自检 + 精度在**生成后/阶段边界**判。生成阶段内部（kernel-worker 的编译/精度修复循环）**不被 gate 中途打断**——gate 是「交付前的验收」，不是「每步审批」。
- **fail-closed 只对交付质量**：gate 失败 = **不放行交付物**（不标 PASS、不归档），但**不删/不改**已生成的中间产物，允许重入修复（如 `--optimize` / probe 再入）。即：宁可「未交付」也不「假交付」，但失败不毁现场。
- **正交于知识注入**：gate 只验「产物是否真·达标·可独立调用」，不参与 §5 的知识取用——所以收紧 gate 不会削弱生成能力（知识从三源照常注入），只会拦住不达标的交付。
- **实现/测试状态**：`hermeticity_gate.py` / `cannbot_ut_gate.py` / `coverage_gate.py` 已实现 + 单测；a5_ops 侧 Phase O0 钩子门 / provenance / self-critic 已生产在用。**集成进插件运行回路 = 见 §10。**

### 6.2 性能门（perf gate）：强制测量、阈值与失效处理

**策略**：性能默认强制测量。是否升级优化 / 拦截交付，由性能比（目标 kernel 相对参考实现的耗时比）与阈值判定。阈值解析顺序：`.ascendc_env` 覆盖 > 插件按算子类给出的 band-aware 阈值（`plugin.ko_escalation_threshold(op_class)`）> AscendC 默认 `0.6×`（`schema_norm.py`，本插件默认值完好，未在移植中改动）。**跳过性能测量为显式开关**：`--perf-threshold=0` → `PRECISION_ONLY` profile（`perf_gate.py`），仅在明确指定时生效，不隐式跳过。

**门判定基于事实、不采信自报状态**：finalize 阶段的性能门不采信子 agent 自报的 `performance.status`，要求 `performance.independent_re_measure`（独立复测证据，`phase_o5.py`）。测量有效性有两条硬约束：(1) 方法学对称性（P141）——参考侧与目标侧须走同一测量路径，否则两侧测量对象不同；(2) 同 session 采集——A/B 两侧须在同一 session 内实测，禁用跨 session 的存档基线。

**一次实测失效（run 2684792，gelu，arch22→arch35 移植）**：

| 项 | 事实 |
|---|---|
| 启动命令 | `python3 -m orchestrator --port-a3 gelu --lane 0`（未带 `--perf-threshold=0`）→ 走默认，性能强制测量 |
| 环境 | 本次无 A3 device；参考侧 aclnn-pipeline 与目标侧 raw `ACLRT_LAUNCH_KERNEL` 测量路径不对称（P141，实为不同 op）；A3 侧仅有跨 session 的存档基线 |
| 子 agent 行为 | 完成 A5 侧独立复测（同一 `.so`，md5 一致；10 warmup + 20 rep，实测 2.40×，快于自报 2.264×，无注水），**并在文档中透明披露上述三条使测量无效的限制**；但仍标 `performance.status=PASS` |
| 门判定 | finalize 性能门正确拒绝：测量虽经独立复测且已披露，但 P141 op 不对称 + 跨 session A/B 使其无效，`PASS` 声明不成立 |
| FSM 行为 | `finalize → finalize_rollback → 重新拉起 worker` 循环（2 次 rollback、3 个 worker、约 55 分钟） |
| 精度 | 不受影响：`verification.json` 精度 29/29（生态 T1，目标输出 vs CPU-fp64 真值），已独立核验 |

**根因与处理**：
1. 缺陷不在隐瞒——子 agent 透明披露了限制；缺陷在于**自披露测量无效后仍声明 `PASS`**。有效性未成立时的正确状态为 `N/A`（或 `NOT_VALID` + reason），据此阻断交付而非放行。（worker 侧修正）
2. finalize 遇不可满足的性能门应 fail-fast 并给出明确原因，而非无限 rollback。（robustness 缺陷，登记 DEBT-192）
3. 仅验证精度 + 自包含的场景，应显式使用 `--perf-threshold=0`（`PRECISION_ONLY`），使性能不参与该场景的门判定。**真实性能数据（相对 `0.6×` 阈值）须在同 session 内、A3 与 A5 两侧均实测、且测量路径对称（对称 op 方法，而非 aclnn-pipeline 对 raw `ACLRT_LAUNCH_KERNEL`）的条件下单独测得，不可跳过，不可由无效测量替代。** 该条件严于「提供一台 A3 device」。

结论：本次失效是「自披露测量无效后仍声明通过」与「门失败后未 fail-fast」两个缺陷叠加，非性能策略需要放宽。

## 7. 跨代际可扩展

当前 arch22→arch35。新增目标架构/产品 = 加 NL→canonical-target 映射 + 该目标的 KB，**入口与流水线范式不变**。规划：更多目标 + 反向跨代际移植（如 910C→910A）。

## 8. Harness 抽象（Claude Code / opencode）+ 安装面差异

CANNBot 与底座 agent harness 不直接耦合：引擎经 `backends/` 的 `Backend` 抽象调度子 agent
（`cc_backend` / `opencode_backend` / `codex_backend`），边界不变量是
**backend 只接线 harness、绝不自带语义规则**（规则在 canonical checker / KB）。
`AOG_HARNESS_BACKEND=claude_code|opencode` 决定走哪条线，**两条线互不依赖**：
claude 模式不要求 opencode/node，opencode 模式完全不需要 claude 环境（含安装预检、
运行时自检与进程清理，见 §8.2 与 `backends/opencode_runtime.py`）。

### 8.1 两个 harness 的差异一览

| | Claude Code | opencode |
|---|---|---|
| agent 文件 | marketplace 安装为本地 agent | **不落盘**，每次 dispatch 经 `OPENCODE_CONFIG_CONTENT` 注入（open code 的 agent frontmatter `tools:` 是 record，落盘会令该机器所有工程的 opencode 无法启动） |
| 安全网触发面 | PreToolUse / PostToolUse / SubagentStop hook | `tool.execute.before` / `permission.ask` 适配器 + 编排器 dispatch 站点 stop gate |
| 安全网注册 | 写用户 settings/hooks 声明 | 同样经注入配置注册 `a5_ops_hooks.mjs`，不写用户的 `opencode.json`（0600、含明文 key） |
| 子 agent 模型 | settings.json | OpenCode 自身配置；需要固定模型时显式设置 `AOG_OPENCODE_MODEL*` |
| 入口 | skill | `.opencode/command/*.md`（安装时 `@@PLUGIN_DIR@@` 物化为真实路径） |

### 8.2 opencode 安装面的安全网证明（两级，实现细节）

USAGE.md 只要求用户看最后一行 `✓ safety net ENFORCES`；这里放实现细节。

- **结构级**：真实 opencode 二进制必须解析注入配置（agents 为 `mode: primary`、入口 skills 可见），
  否则 **exit 1**。首次在全新 opencode 环境下安装会先触发 opencode 的插件依赖解析（`bun install`，
  实测 >90s、在线约 110s），安装器有进度提示 + 180s 超时兜底。
- **行为级**：`engine/src/opencode/probe_safety_net.mjs` 用真实 JS runtime 调适配器的
  `tool.execute.before`，要求 **deny/allow 成对**：kernel-worker 读别的 workspace 必须被拒、读自己的
  必须放行。通过 → manifest `hooks_verified_live: true`；不通过 → exit 1。无 node/bun 时，非严格安装只
  告警以便完成其余配置，`--strict-deps` 会失败；无论哪种安装结果，首次 OpenCode dispatch 的运行时门都会
  fail-closed 拒绝执行。
  （只做 deny 半边不够：一个"什么都拒"的坏门与"武装完好"从外面看一模一样，只有 allow 半边能区分。）

**运行时门与受控例外。** 首次 OpenCode dispatch 会记录版本建议线（默认 `1.18.18`，可用
`AOG_OPENCODE_MIN_VERSION` 显式覆盖）：低版本、无法解析或查询失败只会留下兼容性 warning，不会单独
阻断；可执行文件缺失和行为探针仍是硬门。只有 `(exit 0, "OK")` 或 `(exit 2, "SKIP:…")`
两种探针结果会通过，后者会留下告警。`AOG_OPENCODE_SKIP_RUNTIME_CHECK=1`
是仅供测试或短时运维诊断的显式逃生口，正常生产运行不得设置。流式 watchdog 默认继承
`AOG_STREAM_SILENCE_TIMEOUT_SEC`（未设置为 1800 秒），OpenCode 可用
`AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC` 单独覆盖。OpenCode 子进程会兼容性地设置
`CLAUDE_PLUGIN_ROOT=<本插件根>`，仅为既有 agent prompt 宏提供路径；这不是 Claude Code 配置，也不会
读取 `~/.claude` 或其插件缓存。

**三道证明的共同盲区**（安装探针、Phase O0 探针、单测）：都是自己驱动适配器，证的是
「到达守卫后判得对」，不是「opencode 会让它到达」。要验证后者，跑模型驱动端到端
（`AOG_E2E_OPENCODE_MODEL=<provider>/<model> python3 -m pytest src/scripts/tests/test_opencode_e2e_live.py`，
需凭证、花 token、默认 skip）。G8 起该文件还断言：模型必须发起真实工具调用（README_PROBE
唯一 token + opencode NDJSON 流事件，蒙猜不通过）。

### 8.3 离线安装

opencode 首次安装需一次性拉插件依赖（npm registry）。完全离线：在联网机器跑一次
`init.sh … opencode` + 一次 `opencode run`，把 `$CONFIG_ROOT/{node_modules,package.json,package-lock.json}`
搬到目标机器；或预置 npm 镜像。

## 9. 待定（依赖社区）

- **插件自带 KB(b) 的格式/接口**：采用社区认可的 OKF 格式，载荷由 #611 交付。
- 插件激活（plugin.json + skills 列表填充）在 KB 就绪后进行。

## 10. 集成与装配（bundle-orch wiring）

把 §2 流水线 / §5 三源 / §6 安全网串成一个**自包含、可运行**的插件：

1. **装配**：编排器引擎（FSM + 子 agent 调度）+ aog-* 子 agent 定义 + scripts + `references/`(b 层 KB) **打包进插件目录**；install 把 cannbot skills（a）+ 插件 KB（b）装成 CC skills、并建用户 KB（c）根目录 + INDEX。
2. **入口→编排器**：两个入口 skill 解析 NL 目标 → 调打包进来的编排器（`python -m orchestrator …` 形态）。
3. **每阶段 wiring**：Parse→分类→参考/真值→**生成**（编排器据分类用 `cba_resolver`(§5.2) 取三源 → 注入子 agent brief → 子 agent 生成 + 内层修复循环）→构建→精度→（性能）→报告→**沉淀写回用户 KB(c)**(§5.3)。
4. **gate 边界**(§6.1)：生成前钩子/前置门；阶段边界 provenance/自检/精度门；fail-closed 只拦交付。
5. **验证**：cannbot-知识-被用 = differential A/B（抽掉 a 层 → 该 topic 在 B 不可解 → 证 a 真被用）+ provenance transcript。

### 10.1 实现状态（如实，随实现更新）
| 设计项 | 设计 | 实现 | 测试 |
|---|---|---|---|
| 确定性流水线 FSM | ✅ §2 | ✅ bundle-orch（`engine/src/scripts/orchestrator` 随插件交付） | ✅ 单测全绿；PR #814 e2e 验证 |
| 三源 READ resolver (c>b>a) | ✅ §5.2 | ✅ `cba_resolver.py` | ✅ test_cba_toggle(真 KB+cannbot 树) |
| 三部分 KB / dedup 覆盖证明 | ✅ §5.0 | ✅ converter + `coverage_gate.py` | ✅ 44 passed |
| 安全网 gates | ✅ §6 | ✅ hermeticity/cannbot_ut/coverage + 插件内 O0 钩子门/provenance/self-critic | ✅ 单测全绿；已随插件运行 |
| 用户 KB 存储+索引 | ✅ §5.1 | ❌ 待实现 | ❌ |
| 写回用户 KB(c)、不改 b | ✅ §5.3 | ❌ 待实现 | ❌ |
| **集成（bundle-orch 端到端生成算子）** | ✅ §10 | ✅ **已 bundle**：入口 skill → `python -m orchestrator` → 全流水线随插件交付 | ✅ 引擎单测全绿 + PR #814（含 opencode harness）e2e 走通 |

> 本表是「恐惧清单」的对账：已实现+测的不是幻觉（44 单测过）；❌ 项是真空白，按本文档落地。

## 11. 本文档 = 强制约束（流程）

- 任何代码生成 / 委派 sub-agent / code review，**必须把本文档作为强制上下文输入**（满足上面全部设计项）。
- **有架构调整：先改本文档、再改实现**（doc-first），防止实现漂离设计（曾因此在 orch-less ↔ bundle-orch 间反复横跳，见 §1.1）。
- 本文档是 cannbot fork 仓内插件目录的**唯一权威**（不在 a5_ops 仓）；保持更新。
- 设计与 code review **优先用 codex**（read-only），评审时把本文档一并提供。

## 12. Codex 设计 review 结论 + 解决映射（2026-06-29 首轮）

codex（read-only、喂了本文档 + cba_resolver/converter/coverage_gate + AGENTS.md）adversarial 审 6 项需求，结论 = **2 CONTRADICTION + 4 GAP**，headline：**先把 bundle-orch 做成真的**（在有可执行编排器+FSM状态+KB注入+gate+写回之前，本文档多为「意图陈述」）。逐项解决映射：

| # | codex 判定 | 解决方式 |
|---|---|---|
| 1 确定性流水线 | CONTRADICTION：无真状态表/schema/retry；AGENTS.md 仍自执行 | **bundle 真引擎**：FSM 契约 = 随引擎打包的权威 spec `workflows/opgen_state_machine.yaml`（2001 行、states/transitions/invariants）+ `state_machine.py`；AGENTS.md 改 orch-shell 只调引擎。 |
| 2 安全网不阻碍生成 | GAP：§6.1 只断言、未机械化、未集成 | 每 gate 标 precondition/boundary/delivery-only + FSM 中确切位置 + 失败效果/保留产物/修复转移 —— 落在引擎的 state_machine（同 #1 来源），随集成具体化。 |
| 3 三部分 KB | GAP：b 层 TBD；c 层 schema 粗（path「e.g.」/env 未命名/INDEX schema 未定/无 lock-atomic-scope）；dedup 判据被夸大 | dedup 判据已更正（§5.0，existence≠coverage、待升内容级）；c 层 schema/INDEX/env/scope 待在写回实现时定死（§5.3 build）。 |
| 4 三源取知识 | GAP：resolver 只扫单文件全词、不用 INDEX/frontmatter/applies_to/scope；「能看到」≠「会 invoke」tier-a | resolver 升级为返回**具体 artifact（含 tier-a skill 名/路径）**；编排器**强制注入 tier-a 或用 transcript gate 验证真 invoke**（随集成）。 |
| 5 写回只动用户 KB | GAP：写侧未实现；「硬编码 user root」与「env override」矛盾；「插件只读」在 agent 有写权限时不可强制 | 单一 writer API（resolved user root + symlink/traversal 拒绝 + 原子 file+index 写 + lock + 权限）+ **run 后 gate：若插件 `references/` 被改则 FAIL**（§5.3 build）。 |
| 6 bundle-orch 执行模型 | CONTRADICTION：仍 orch-less（AGENTS.md/install.sh/plugin.json 无引擎 wiring） | **核心交付**：打包 `python -m orchestrator` 引擎 + 子 agent 定义 + install 期 skill/KB/user-KB wiring + AGENTS.md 换 orch-shell。 |

**结论**：#1/#2/#6 由「建 bundle-orch（打包真引擎 + 引用其 FSM spec）」一并解决（= 核心交付）；#3/#4/#5 在写回/resolver 实现时定死 schema。**先建 bundle-orch、再随集成把 #3/#4/#5 的契约落地**。本节随每轮 codex review 更新。

## 13. 算子生成模式说明（#363 标准结构）

> 按 RFC #363 的统一模式结构（9 段）描述本插件的两个算子生成模式，供「算子生成模式总览」收录 + reviewer 评估。两模式共用确定性流水线（§2），差异在「生成」阶段。

### 13.0 共享标准工作流（阶段 | Owner | 输入 | 输出 | 门禁）

| 阶段 | Owner | 输入 | 输出 | 门禁 |
|---|---|---|---|---|
| 钩子完整性 O0 | Primary(orch-shell) | 插件 hook 配置 | 完整性 verdict | hook 完整或自愈 |
| 解析/配置 O1 | Primary(orch-shell) | 用户 NL 目标 + 源算子 | 任务配置(source/target/mode) | 目标可归一、源可识别 |
| 分类 O1.5 | orchestrator | 源算子 | op-family + det-policy | 分类完成 |
| 参考/真值 O2.5 | orchestrator | 移植源算子（实时 A3-CANN）/ 正向规格（反向） | reference / golden | 真值自洽、来源与 state 绑定 |
| **生成 O4（差异阶段）** | aog-kernel-worker | 配置 + 参考 | 目标 AscendC kernel | 编译通过 |
| 构建 | build harness | kernel | .so + provenance | build green |
| 精度验证 O5 | orchestrator + precision | kernel + 真值 | precision verdict(分层) | 达阈值 / 记录 ceiling |
| 性能(可选) | aog-kernel-optimizer | kernel + profiling | perf ratio | ≥阈值 或 记录 |
| 报告 O6 | aog-report-gen | 全产物 | REPORT + 复现指引 | 报告完整 |

差异阶段 = 参考 O2.5 与生成 O4：cross-gen-port 走来源架构实时 A3-CANN 真值→目标 kernel；backward-gen 走 CPU/fp64 autograd 梯度真值→反向 kernel。其余阶段两模式共用。

### 13.1 模式 A：ascendc-cross-gen-port（跨代际算子移植）

- **定位**：把一个 AscendC 算子从来源架构移植到目标架构。
- **适用场景**：已有 arch22(910C/V220) 的 AscendC 算子、需在 arch35(950PR/V300) 上得到等价算子；用户能用自然语言指定目标。
- **不适用场景**：源不是 AscendC（用其它 op-gen 模式）；目标架构超出当前支持（当前仅 arch22→arch35）；纯新算子无源参考（用直调/注册模式）。
- **标准工作流**：见 §13.0，生成阶段走移植路径。
- **参考输入**：当次在 A3 执行来源 CANN，捕获的输出作为 A5 目标算子的验证真值；离线 A3 tensor 包或其它参考来源不在本模式当前实现内。
- **Agent 设计**：Primary（AGENTS.md，orch-shell）= 解析 NL→canonical target、选 lane、调编排器、收口状态/报告，**不亲自逐阶段写 kernel**；Subagent = aog-kernel-worker（生成+构建+验证）、aog-kernel-optimizer（perf）、aog-precision-probe（精度卡壳）、aog-fused-optimizer（fused 升级）、aog-researcher（架构探索）、aog-determinism-analyzer（确定性）。触发条件见 §4。
- **Skill 依赖（阶段级）**：入口 `ascendc-cross-gen-port`（O1）；KB resolver c>b>a（O2.5/O4 按需注入）；社区 skills a 层（O4 生成，如 ascendc-api-best-practices）；gates 安全网（O0/O5）；KB writer（O6 写回 c）。
- **工件契约**：state（`workspace/{op}/.opgen_state.json`）、kernel（`workspace/{op}/kernel/*.{h,cpp}` + pybind11.cpp）、A3 reference provenance、build provenance、precision/perf report、user-KB 写回文件。消费者 = 后续阶段 + reviewer + 用户复现。
- **失败恢复**：可重试（编译/精度内层循环、`--optimize` 再入）；可回滚（状态机回退阶段）；保留现场（workspace 持久化、可中断恢复）；blocked（真值不可得/硬件不支持 → 升级 aog-researcher 或 await_user_decision）；用户需补充（目标歧义时回询）。
- **用户入口示例**：`把这个算子移植到 arch35：<算子源/名称>`；`移植到 950PR / A5：<算子源/名称>`。

### 13.2 模式 B：ascendc-backward-gen（正向 → 反向生成）

- **定位**：由一个正向算子，自动生成其反向（梯度）算子。
- **适用场景**：已有正向 AscendC 算子 / spec、需对应反向（梯度）算子；可自然语言指定目标芯片。
- **不适用场景**：正向不可微 / 无梯度定义；只需要正向实现（用移植模式）；目标芯片超出当前支持。
- **标准工作流**：见 §13.0，生成阶段走反向路径。
- **Agent 设计**：Primary / Subagent 同模式 A；反向特有 = O2.5 由正向生成 autograd / CPU-truth 精确梯度真值（`BACKWARD_SPEC`：wrt + inputs）。
- **Skill 依赖（阶段级）**：入口 `ascendc-backward-gen`（O1）；其余同模式 A。
- **工件契约**：同模式 A + 反向梯度真值（grad golden）；消费者同。
- **失败恢复**：同模式 A；反向特有 blocked = 梯度真值不自洽 → 停在真值阶段报错、不进入生成。
- **用户入口示例**：`为这个正向算子生成反向（目标 a5 / arch35）：<正向算子>`。

> 失败恢复 / 工件契约的字段级细节随实现演进，以 §2 FSM + §6 安全网 + §10 集成为权威；本节为 #363 模式说明视图。
