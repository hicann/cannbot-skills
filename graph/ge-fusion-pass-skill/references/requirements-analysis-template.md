# 需求分析文档模板（requirements-analysis-template）

这是**需求分析阶段唯一业务产物 `requirements-analysis.md` 的结构模板**。统一 skill 的阶段一按本模板产出文档、落盘到 case 目录；阶段二与阶段三按章节号读取该文档。

> 本模板是结构契约，不是知识源——每节「填什么」的判定依据指向统一 `ge-fusion-pass-skill` 的阶段一、`pass-development-paradigm.md`、`fragment-spec.md`、`interface-catalog.md`。落盘文档里每节都要写清判定依据与证据，字段不留空、不伪造。

> 本模板取代旧的结构化卡片产物，是需求分析阶段的唯一业务产物。假设、待确认事项、证据和 G1 状态都写入**同一份文档**（§10、§2、§12），不再产生独立待确认清单。

---

## 文档产出与传递契约

- **落盘**：统一 skill 的阶段一产出 `requirements-analysis.md`，写到 case 根目录（或用户指定目录）。
- **revision**：文档末尾 §12 只记录摘要文件路径与产出时间；用 `write_handoff_digest.py` 将 `requirements-analysis.md` 的 SHA256 写到独立文件，供下游核对传递内容未被篡改/截断。摘要绝不写回被摘要的文档。
- **传递**：编排 agent 把「文档路径 + sha256 hash + 完整内容」传给阶段二（开发）与阶段三（验证）；下游以路径为主、内容为辅，hash 用于一致性核对。

---

# 融合 Pass 需求分析

> 文档标题行。下文 12 节为正文结构，逐节填写。

## 1. 任务背景与目标

**填什么**：一段自然语言说清「要做什么融合 pass、为什么做、预期收益」。来自用户原始需求。

**判定依据**：统一 skill 阶段一“证据核实、联网与澄清”+ 用户输入。

## 2. 输入材料与证据

**填什么**：列明本轮拿到的输入（自然语言需求 / `data/` / `CMakeLists.txt` / 参考路径 / 模型文件 / dump）与已核实的环境证据（`$GE_REPO_PATH` 是否可达、soc version、CANN 版本、ES API 可用性）。每条标来源。**输入只给模型/dump 时**，引用 `adapt_input.py` 产出的 `input-inventory.json` / `normalized-graph.json` / `provenance.json`（schema 见 `normalized-graph-schema.md`，分流见 `input-adaptation.md`），并标 **`structural`**（仅结构复现，pbtxt/残缺输入）或 **`semantic`**（完整语义，ONNX/AIR 正常解析）reproduction 级别。

**判定依据**：统一 skill 阶段一的核实顺序；`knowledge-base.md` §一 KB 入口映射、`ge-repo-map.md` §1 任务路由表；`input-adaptation.md` 输入分流。

## 3. 原始图与目标片段

**填什么**：目标片段六要素——目标结构（导入后真实 GE op type）、定位（可选）、边界（输入/输出 tensor）、守卫（dtype/shape/attr）、可选算子（bias/offset_w 等）、目标形态（替换后结构）。附原始图描述或 dump 片段引用。

**判定依据**：统一 skill 阶段一“目标片段与路线选择”；`fragment-spec.md` §二片段规格模板、§三大网络指片段方式。**op type 一律以 dump 为准**（`tips/dump-first-op-type.md`）。

## 4. 预期图变换

**填什么**：替换前后的图结构变化（如 MatMul+Add→GEMM、grouped Conv→Split+多Conv2D+Concat、Add(x,0)→x）。声明优化意图（融合减算子 / decompose 换形态 / 删冗余 / 改 format）。这是验证阶段判定「图层收益是否达成」的对照基准。

**判定依据**：统一 skill 阶段一的目标形态要求；`example-map.md` 场景→样例→看点。

## 5. Pass 路线选择

> 本节是开发前**必须完成**的明确决策。强制记录：可表达性、选择路线、理由、被否决路线与退化方案。选型矩阵见 `pass-development-paradigm.md` §3。

### 5.1 Pattern 可表达性

**填什么**：可 / 不可 + 原因（控制边 / 嵌套子图 / 动态输入输出个数）+ 不可时退化到哪。三条边界规则（输入边界/输出边界/输入个数精确）是否满足。

**判定依据**：统一 skill 阶段一“目标片段与路线选择”；`interface-catalog.md` §一.2。

### 5.2 graph / pattern / decompose 选择及理由

**填什么**：base 家族（graph_base_pass / pattern_base_pass）+ 具体接口（函数式 graph pass / FusionBasePass / PatternFusionPass / DecomposePass）+ 语言（C++/Python）+ 选择理由（对外接口 > 参考路径 > 交付物，受 pattern 可表达性否决）。pattern 设计参数（pattern 数量/输出个数/捕获需求/匹配严格度/op_types）。

**判定依据**：统一 skill 阶段一“目标片段与路线选择”；`interface-catalog.md` §一、§一.3 V1/V2 基类；`pass-development-paradigm.md` §3 选型矩阵、§4 语言选择。

### 5.3 被否决方案及原因

**填什么**：列出考虑过但未选的路线及否决理由（如「PatternFusionPass 被否决：目标含控制边」「DecomposePass 被否决：目标是多→一非单算子展开」）。含「禁止走偏」——本 case 不该实现的相邻接口（如不是 PatternFusionPass 就不写 `Patterns()`/`Replacement()`；Python `PatternFusionPass`/`DecomposePass` 不得重写 `run()`）。

**判定依据**：统一 skill 阶段一的可表达性和路线约束；`interface-catalog.md` §二 Python 特有写法。

## 6. 接口、语言与注册阶段

**填什么**：具体接口 + 语言 + 注册阶段（可选阶段集合 → 选哪个 → 为什么 → 版本门槛）+ 接口清单（用到的 pass 接口 + 构图接口）+ 每接口注意点（指向 tip）。

**判定依据**：统一 skill 阶段一“目标片段与路线选择”；`interface-catalog.md` §一.1 阶段枚举、§一.3 V1/V2、§二 pass 接口、§三构图接口。

## 7. 匹配边界、守卫和可选输入

**填什么**：边界（接线依据）+ 守卫（MeetRequirements 判什么：dtype/shape/attr/Const 值，严格度 PatternMatcherConfig vs MeetRequirements）+ 可选输入形态策略（S1/S2/S3，支持哪些形态、其余如何拒绝并打日志）。

**判定依据**：统一 skill 阶段一的片段边界与路线约束；`fragment-spec.md` §四可选输入场景与三种策略。

## 8. 开关、加载与环境依赖

**填什么**：开关策略（默认常开 / 按 option key 开关：key 名 + 缺失时默认行为 + 版本门槛）+ 交付与加载路径（`.so` 安装目录 / `ASCEND_GE_PY_PASS_PATH`；触发编译用 `atc` 还是 `pyatc`）+ 环境依赖（CANN/OPP/ES API/soc version 探测结果与缺失时的降级）。

**判定依据**：统一 skill 阶段一的语言、加载和环境规则；`interface-catalog.md` §一.1 版本门槛、§一.3 context 可达性；`knowledge-base.md` §四环境探测规则。

## 9. 验证方案

> 整网输出比较与性能方案属验证证据阶段；它是 skill 的内部里程碑 R4，不是版本号。本节先定方案占位，由该阶段落实执行细节。

### 9.1 匹配、生效与跳过维测

**填什么**：怎么确认 pass 被加载/执行/命中（dump 文件名按注册阶段、日志关键字 `std::cout`/`print`、`DUMP_GE_GRAPH=1`/`DUMP_GRAPH_LEVEL`）。

**判定依据**：统一 skill 阶段三“验证顺序与证据”及“诊断、性能与交付”；`tips/dump-first-op-type.md`、`tips/dump-log-diff-checklist.md`、`fusion-troubleshooting.md` 诊断树。

### 9.2 图结构正确性

**填什么**：dump 前后拓扑对比预期（对照 §4 预期图变换）。

**判定依据**：统一 skill 阶段三“验证顺序与证据”；`tips/dump-log-diff-checklist.md`。

### 9.3 融合前后整网输出比较

**填什么**：baseline（pass 未生效）vs optimized（pass 生效）整网最终输出比较方案——相同模型/输入/seed/预处理/编译运行配置；比较输出数量/映射/shape/dtype/数值；整数布尔精确比较，浮点用容差（按 dtype 定默认容差，记录 max abs/rel error 与容差）。环境不具备整网运行能力时记 `NOT_RUN` 并说明缺失条件。**不启动逐算子精度工具链**。

**判定依据**：统一 skill 阶段三“验证顺序与证据”及 `validation-evidence.md` 的整网输出契约（比较整网最终输出，不做逐算子精度比对）。

### 9.4 性能比较

**填什么**：默认 L0 图层收益核对（对照 §4 优化意图）；用户明确要求时 L1 msprof profiling（baseline vs optimized 设备耗时对比）。

**判定依据**：`performance-analysis.md` 两档；统一 skill 阶段三“诊断、性能与交付”。

## 10. 假设、限制与待确认事项

**填什么**：所有 `【假设，待确认】` 项汇总——假设了什么、为什么、怎么确认、暂按什么处理。**取代旧「待确认清单」独立产物**，并入文档此节。

**判定依据**：统一 skill 阶段一“证据核实、联网与澄清”及 G1 降级规则；取不到证、需实测的开放问题作为“假设+待确认”记录在本 case 的 §10，不在本节当结论写死。

## 11. 计划交付物

**填什么**：`src/*.cpp` 或 `src/*.py` 路径、构建/加载/运行命令、`.so` 安装目录或 `ASCEND_GE_PY_PASS_PATH` 设置、交付形态（产品化 `.so` / 快速验证 `.py`）。

**判定依据**：统一 skill 阶段一的语言和加载规则；`pass-development-paradigm.md` §4 语言选择。

## 12. G1 门禁、证据与文档 revision

**填什么**：G1 门禁块（`gate_status`/`result_status`/`missing_fields`/`assumptions_or_limitations`/`evidence`/`artifacts`）+ 关键证据（2-4 条来自需求/参考路径/交付物/加载方式）+ 文档 revision（摘要文件路径 + 产出时间）。文档定稿后生成摘要文件：`python3 "$SKILL_ROOT/scripts/write_handoff_digest.py" --input "$CASE_ROOT/requirements-analysis.md" --out "$CASE_ROOT/artifacts/handoff/requirements-analysis.sha256"`。

**判定依据**：统一 skill 阶段一的 G1 判定（PASS/DEGRADED/BLOCKED）。

```text
gate_status: PASS | DEGRADED | BLOCKED
result_status: PASSED | FAILED | NOT_RUN
missing_fields: []
assumptions_or_limitations: []
evidence: []
artifacts: [requirements-analysis.md, artifacts/handoff/requirements-analysis.sha256]
revision:
  handoff_digest: artifacts/handoff/requirements-analysis.sha256
  produced_at: <产出时间>
```

- `PASS`：需求分析文档和可核验摘要文件已产出，且 `base 家族 / 具体接口 / 语言` 都有已核实结论，`result_status: PASSED`。
- `DEGRADED`：文档已产出，三个关键字段都有明确取值或"未确认"占位；所有未核实项均标为假设、写入 §10，并写清确认方式。摘要文件或部分证据不可获得时相应标 `NOT_RUN`；允许下游按限制继续，但不得把假设当事实。
- `BLOCKED`：文档未产出，或任一关键字段既无取值/占位，也无降级说明。只在此状态停止，不进入开发。

门禁判定属于统一 skill 的阶段一；编排 agent 只读取 `gate_status`，不得重新实现关键字段规则。
