# HCCL AICPU 共享跨层契约

> 本参考是 `hccl-aicpu-best-practice` 中设计、开发、测试和检视角色共享的 AICPU 技术真源，回答“怎么做、受什么约束、如何客观验证”。Agent 职责、权限、调用顺序、失败回退和完成状态由上层 Plugin 定义，不在本参考中编排。

## 适用范围

适用于新增或修改 **AICPU 引擎**的 HCCL 集合通信算法：

- `algorithm/template/aicpu/` 算法模板
- `algorithm/executor/` 执行器和注册
- `selector/` 算法选择与成本候选
- `op_common/algorithm/topo_match/` 拓扑匹配
- 上述层级之间的接口、资源和算法身份一致性

不适用：AIV 模板、CCU 引擎、`src/legacy/` 新特性。CCU 暂不对外支持；环境准备使用 `hccl-env-setup`；功能测试执行使用 `hccl-test-tool-hvm`；AICPU 问题诊断使用 `hccl-aicpu-debug`。

## 能力路由

按改动对象加载最小必要能力。各 Skill 是技术能力，不代表 Agent 角色。

| 改动对象 | 目标 Skill | 当前能力来源 |
|---|---|---|
| 需求、分层方案与 Dataflow Spec | `hccl-aicpu-design` | [设计入口](../hccl-aicpu-design/SKILL.md) |
| template 实现、数据流、线程同步 | `hccl-aicpu-best-practice` | [实现实践入口](SKILL.md) |
| executor、注册、`REGISTER_ALG_ATTRS` | `hccl-aicpu-best-practice` | [executor-selector.md](references/executor-selector.md) §1 + [architecture.md](references/architecture.md) §2 |
| selector、阈值、成本候选 | `hccl-aicpu-best-practice` | [executor-selector.md](references/executor-selector.md) §2-§4 |
| TopoMatch、分层拓扑 | `hccl-aicpu-best-practice` | [architecture.md](references/architecture.md) §3 |
| 源码预检、构建与证据 | `hccl-aicpu-best-practice` | [build-verification.md](references/build-verification.md) + `scripts/` |
| Checker 功能测试 | `hccl-test-tool-hvm` | `communication/hccl-test-tool-hvm/` |
| 源码分支、边界与数据流的白盒用例设计 | `hccl-aicpu-whitebox-design` | [专项入口](../hccl-aicpu-whitebox-design/SKILL.md) |
| 构建、选路或 Checker 故障定位 | `hccl-aicpu-debug` | [专项入口](../hccl-aicpu-debug/SKILL.md) |
| 有明确性能目标时的基线与候选测量 | `hccl-aicpu-perf-optimize` | [专项入口](../hccl-aicpu-perf-optimize/SKILL.md)；性能收益须另取真实测量证据 |

关键路由规则：

- “只注册、不改变默认选择、用 `HCCL_ALGO` 测试”只需要 template、executor 注册及必要映射；先读 [HCCL_ALGO 显式选路](references/executor-selector.md#4-hccl_algo-显式选路保留默认选择)，不要额外新增旧 selector 默认分支。
- 一个任务涉及多层时，分别加载各层能力；跨层契约以本参考为准。
- 各专项 Skill 的知识真源在对应 Skill；本参考不复制其实现细节。

## 自定义算法的设计与接线入口

新增 Ring、Tree 或用户自定义调度时，算法语义、阶段和片归属由 `hccl-aicpu-design` 的算法设计参考与 Dataflow Spec 定义；实现能力消费设计结果后按[接线入口](references/10-implementation-entry.md)核对来源、接口和目标 executor。
算法调度由用户需求决定；不默认全量扫描或完整阅读已有 NHR/Mesh。
来源匹配且适用范围一致时复用已有结论；新 executor/接口、来源变化或契约缺口只定向核对。
AllGather Sole/Parallel 直接加载[接入卡](references/allgather-integration.md)，含注册组合、版本入口、参数传递与初始化顺序。
复用 AllGather Parallel 时按需加载[四阶段契约](references/12-allgather-parallel.md)，
不把单层契约外推到分层调用。不安排“读取全部参考”；交接已有来源记录和具体缺口即可。
跨层改动继续加载相应能力，完整构建与双轨验收不变。
调研是否委派及其输入范围由 Plugin/Agent 决定；本参考只提供已核验契约、来源证据和具体缺口，不扩大算法组合或验收范围。

## 角色消费边界

本参考不授予执行权限，具体权限以上层 Agent 定义为准。`hccl-op-dev` 的四个角色按以下范围消费能力：

| 角色 | 使用本参考的能力 |
|---|---|
| Architect | 需求输入契约、三层架构、数据流规格、默认选路影响和测试矩阵 |
| Developer | 分层实现、静态检查、生成器、CMake 接线和开发期诊断 |
| Verifier | 基线证据、源码快照、唯一候选构建入口和双轨测试判据 |
| Reviewer | 数据流、资源、注册、选路、架构红线和证据身份的独立核对 |

角色间交接格式、状态推进和失败回退不在这里定义。

## 仓库与架构真源

| 项 | 说明 |
|---|---|
| HCCL 仓 | `${HCCL_ROOT}`；由已确认的上游输入或 `hccl-env-setup` 生成的环境文件提供，不猜路径、不扫盘 |
| HCOMM 仓 | HCCL 底层通信库，与 HCCL 独立编译 |
| 架构真源 | `${HCCL_ROOT}/AGENTS.md` 第 3 节、`docs/zh/architecture/architecture-brief.md`、目标版本真实源码 |
| 环境入口 | `source <CANN路径>/set_env.sh`；准备和检查分别使用环境专项 Skill |

2026-09 重构后的标准结构：

```text
src/ops/<op>/
├── <op>.h / <op>.cc
├── selector/
├── algorithm/
│   ├── executor/
│   └── template/aicpu/
└── op_graph/

公共层：src/ops/op_common/algorithm/{template,executor,topo_match}/
        src/ops/op_common/{selector,topo_info}/
```

调用链：`Hccl<Op> → Selector(algName) → HcclExecOp → Executor(Orchestrate) → Template(KernelRun)`。

脚本兼容重构前的 `template/aicpu/` 布局；新增代码的实际落点必须跟随目标 HCCL 仓当前结构。

## 任务输入契约

以下信息用于确定能力路由和客观验收范围。调用方应继承已确认的信息和授权；无法由技术证据消解的缺口应结构化返回上游。

| 输入项 | 必要性 | 处理规则 |
|---|---|---|
| 开发意图 | 必须 | 新增算法，或明确要修改的已有算法/文件 |
| 算子 | 必须 | 如 all_reduce、all_gather、reduce_scatter、broadcast、reduce、all_to_all_v |
| 拓扑场景 | 必须 | 单级 Mesh、多级 NHR 或依据真实 selector 定义的其它拓扑 |
| 默认选路影响 | 必须 | 不变，或上游已批准的明确变更范围 |
| 改动层级 | 必须 | 依据需求与源码推导 template/executor/selector/topoMatch 范围 |
| 数据量和数据类型 | 按语义需要 | 覆盖算法目标档位、64bit 类型和 `PROD` 等特殊路径 |
| rank 与通信域 | 按测试需要 | 目标 rank 决定主验证域；112 只作路由冒烟时不能替代目标拓扑 |
| 验收标准 | 必须 | 功能双轨通过；`hccl-vm` 不支持性能验收 |

算法形态、线程分配、executor 复用、TopoMatch 和注册宏属于实现层决策，应依据目标版本的算子契约与接口推导。Ring/Tree 等自定义算法不要求仓内已有同类蓝本。上游已给出的技术偏好作为约束保留；无法由证据消解且会改变目标或验收范围时，返回上游输入缺口。

输入足以唯一确定能力路由，且能够客观判定“完整构建 + 双轨 Checker”是否完成时，即满足契约。

## 技术门禁

这些顺序是正确性约束，不代表 Agent 调度流程。

### 1. 源码和任务预检

开始处理目标代码前，确认 HCCL 仓、分支、HEAD、工作区差异和任务持久目录。使用：

```bash
python3 <本skill路径>/scripts/workflow.py preflight \
  --repo <HCCL源码根> --op <算子> --output <新预检.json> \
  --task-dir <任务目录> [--previous <旧预检.json>] [--environment-file <环境文件>]
```

预检只提供源码与环境身份、入口定位和变化范围，不证明算法正确。

### 2. 数据流规格

新增算法或改变 slice、scratch、同步点、收发模式时，代码前必须有无语义 TBD 的 Dataflow Spec，并通过 `hccl-aicpu-design` 的 `check_spec.py` 和布局量化检查。只读分析或不改变数据流的局部修复不强制创建 Spec。

### 3. 基线与候选身份

涉及注册或选路时，首次候选构建前必须保全可信旧包并完成结构化基线。基线至少绑定：

- 旧包哈希、host/device 身份和来源
- 源码 commit、未提交改动或可恢复快照
- 构建命令、完整日志和真实退出码
- 每个用例的配置、算法名、调用次数、Checker 结果和原始日志

基线 FAIL 必须保留并定位，不能作为候选失败的豁免。缺失证据优先恢复日志，其次安装已归档旧包定点补测；没有可信产物或构建输入已经变化时才重建。

### 4. 候选构建

首次构建前按[接入审查](references/build-verification.md#25-首次构建前的接入审查)
关闭已发现的布局、资源、算法名透传和接线问题；每项给出修复或不影响当前路径的证据。
按模板对 `CalcCostCoeffParam.algName/comm/topoInfo` 的实际依赖逐个绑定 executor 核对传递链；
可用 `scripts/check_cost_forwarding.py --require-field 字段=表达式` 检查直接聚合初始化。
未知写法定向核对，不留到完整矩阵才排查。

首次完整构建的单一入口：

```bash
cd ${HCCL_ROOT}
bash build.sh --pkg --full -j16
```

实际候选构建优先使用 [workflow.py 构建门禁](references/build-verification.md#23-预检与候选构建入口)，由其检查基线、保存源码快照并执行上述命令。不要顺序执行 `--aicpu`、`--pkg`、`--pkg --full`，也不要为补取退出码或日志重复构建。

HCCL 仓内 ST/UT 不是本参考的默认验收证据，也不能替代 Checker；是否执行由 Plugin/Verifier 依据上游授权决定。`check_spec.py`、`check_layout.py`、`check_template.py` 和 Skill 自身离线回归属于静态/工具检查，不是 HCCL ST/UT。

### 5. 双轨功能验收

功能验收加载 `hccl-test-tool-hvm`，AICPU 引擎选择 `AI_CPU`：

| 轨道 | 目的 | 必要证据 |
|---|---|---|
| A：新算法定向 | 证明新代码被选中且语义正确 | `HCCL_ALGO=<已核验配置>`、`--expect-algo <注册全名>`、Checker PASS、失败数 0 |
| B：存量回归 | 证明默认路径无未批准变化 | 不设置 `HCCL_ALGO`、与基线同矩阵、Checker PASS、选路满足验收契约 |

测试计划在 Spec 完成时固定 executor、布局/repeat/rank/stride、通信域、配置 DSL、期望注册名和日志断言。基线与回归使用同一矩阵；多 loop 与 template `repeatNum > 1` 是不同路径，必须分别用源码公式和日志证明。
多批次默认生成并检查 [hccl-vm JSON 计划](../hccl-test-tool-hvm/references/test-plan.md)，使用 `plan.py run` 串行执行。
候选先对每个 executor 跑独立的单用例定向冒烟，优先检查新增或变更的接入链；全部通过后再展开覆盖。
多 loop 大小由已核验的 count 阈值和测试程序换算生成，避免逐次放大数据量试探。保留覆盖要求和两轨比较。

默认选路验收分两类：

- **不允许改变**：用 `hccl-test-tool-hvm/evidence.py compare` 检查用例、算法名和调用次数与基线一致。
- **已批准改变**：实现前固定需求依据、允许范围和逐用例旧/新算法矩阵；分别检查 baseline/regression 全 PASS，再人工核对范围内符合新期望、范围外保持一致。现有 `compare` 不支持放宽选路，禁止编造参数或篡改基线。

轨道 A、轨道 B、算法命中断言和选路契约全部通过，才满足功能验收。环境不可用时只能报告证据、阻塞和未验证范围，不得把静态检查、ST/UT 或已有失败改称 Checker PASS。

## 跨层不变量

1. **算法身份一致**：旧 selector 路径的 `selectAlgName`、executor 注册名和 `REGISTER_ALG_ATTRS` 名称逐字符一致；新 DSL 路径核对 DSL 拼出的全名及对应注册，不为机械凑齐“三处”新增默认分支。
2. **资源消费一致**：Template 的 `CalcRes`、`GetRes`、`GetThreadNum`、Notify 索引和实际 executor 消费方式一致，尤其检查 Parallel `PrepareResForTemplate → GetRes`。
3. **数据流一致**：Spec 与代码中的 offset、length、repeat、stride、scratch、channel、线程和同步位置一致。
4. **两侧接线一致**：AICPU template 同时接入 host CMake 和 `scatter_aicpu_kernel.cmake`，不能只保证 host 编译。
5. **架构边界**：不 include HCOMM 私有头；跨仓接口走 `src/common/hcomm_dlsym/` weak 符号并先查询支持性；不建立反向依赖；legacy 不承接新特性。
6. **编码规范**：CANN-2.0 许可头、PascalCase 类名、`camelCase_` 成员变量、120 列、`.clang-format`；C++ 标准以目标 CMake 为准，当前核对为 C++17；编译无告警。
7. **支持性契约**：新增能力限制/模式分支时，核对字段赋值与消费时序、完整模式枚举及拒绝路径，见 [support-contract.md](references/support-contract.md)。
8. **证据一致**：最终 Review 和 Test 必须指向同一源码身份；源码变化后旧结论不能继续作为当前版本证据；Spec 单独记录哈希，交付清单检查被忽略文件。

## 工具索引

| 工具 | 能力 | 边界 |
|---|---|---|
| `scripts/workflow.py preflight` | 记录源码、任务和环境指纹，定位相关入口 | 不证明实现正确 |
| `scripts/workflow.py build` | 检查旧包和基线、保存源码身份、执行唯一候选全量构建 | 不安装、不运行 Checker |
| `scripts/source_snapshot.py` | 捕获、校验、恢复已跟踪 diff、未跟踪源码和 Unix 权限 | 只恢复到新目录；依赖基准 commit |
| `scripts/archive_package.py` | 归档包、构建日志、退出码、源码快照和哈希 | 不触发构建或安装 |
| `scripts/check_deliverables.py` | 核对显式交付清单、Git 跟踪/忽略状态、Spec 与附件哈希 | 不暂存、不代替源码快照或实际归档 |
| `scripts/compile_smoke.py` | 在已核验编译数据库上做可选单文件冒烟 | SKIP/通过都不能替代完整构建 |
| `scripts/check_cost_forwarding.py` | 按依赖检查成本参数字段透传；默认 algName，可选 comm/topoInfo 等 | 普通聚合初始化；未知语法返回 UNVERIFIED，不证明来源有效或选路正确 |
| `hccl-aicpu-best-practice/scripts/*` | Spec、布局、Template、CMake 检查及代码骨架生成 | 静态检查不替代双轨 Checker |

## 参考资料

| 文档 | 使用时机 | 真源归属 |
|---|---|---|
| [architecture.md](references/architecture.md) | 理解三层调用链、注册机制、TopoMatch 和架构红线 | `hccl-aicpu-best-practice` |
| [executor-selector.md](references/executor-selector.md) | 修改 executor、selector、显式选路或默认阈值 | `hccl-aicpu-best-practice` |
| [allgather-integration.md](references/allgather-integration.md) | AllGather Sole/Parallel 接入、版本选择和成本字段检查 | `hccl-aicpu-best-practice` |
| [support-contract.md](references/support-contract.md) | 新增支持性限制、模式分支或准备跨层评审交接 | `hccl-aicpu-best-practice` |
| [build-verification.md](references/build-verification.md) | 基线、归档、构建、安装前门禁和双轨验收 | `hccl-aicpu-best-practice` |
| [hccl-aicpu-design](../hccl-aicpu-design/SKILL.md) | 需求、分层方案、数据流和选路设计 | 设计能力 |
| [hccl-aicpu-best-practice](SKILL.md) | AICPU 跨层实现、template 静态检查和专项检视 | 实现实践能力 |
| [hccl-aicpu-debug](../hccl-aicpu-debug/SKILL.md) | 构建、选路、Checker 或同步故障定位 | 调试能力 |
| [hccl-aicpu-perf-optimize](../hccl-aicpu-perf-optimize/SKILL.md) | 性能基线、候选测量与归因 | 性能优化能力 |
| [hccl-aicpu-whitebox-design](../hccl-aicpu-whitebox-design/SKILL.md) | 源码分支与边界用例矩阵 | 白盒测试设计能力 |
