---
name: ascendc-cross-gen-migration
description: 当用户希望将已有 AscendC 算子工程跨架构迁移时使用。支持 DAV_2201→DAV_3510、DAV_2201→DAV_2002、DAV_3510→DAV_2201、DAV_3510→DAV_9201/DAV_9202。先选择路线，再执行契约分析、实现改造、构建、精度与性能验证；阶段细则与能力选择按所选路线执行。
metadata:
  version: "3.5"
  date: "2026-11-12"
---

# AscendC 算子跨架构迁移（轻量入口）

## 安装与知识依赖

本目录可作为独立 skill 使用。按照所选路线和阶段声明的条件，安装并加载下表中所需的知识 skill（按 Skill 名称单向依赖，不依赖相对路径；安装方式见各 skill 自身的安装说明）：

| 知识入口 | 使用时机 |
|---|---|
| `knowledge-query` | 迁移指南、概念与经验卡检索，以及精确 API/语义查证 |
| `npu-arch` | 芯片型号/构建目标 ↔ 架构代号映射；按其来源与证据边界澄清，未知 SKU/编译 token 不推断 |
| `ascendc-api-best-practices` | 跨代 API 差异、参数与签名 |
| `ascendc-regbase-best-practice` | 实际涉及 RegBase 改造时查询，目标平台支持须核对版本 |
| `ascendc-simt-best-practices` | 实际涉及 SIMT 开发、改造或 API 查询时使用 |
| `ascendc-registry-invoke-template` | 注册算子工程规范、平台注册方式及构建接入 |
| `ops-precision-standard` | 迁移后的精度验证标准 |

`knowledge-query` 按当前工具的已安装 Skill 列表定位，先读取其 `SKILL.md`，再使用该 Skill 目录下的 `scripts/knowledge_query.py`。知识库通过 `--knowledge-root <知识库根目录>` 或 `CANNBOT_KNOWLEDGE_ROOT` 配置；精确 API 查询使用其 API/语义查证路线，带上实际 API 名和源/目标平台。多个 skill 按需组合，安装本 skill 不会自动安装其他 skill。

本 Skill 是**路由器**，不包含路线实现细节。每条迁移路线是一个自包含路线包 `routes/<源代号>-to-<目标代号>/`（含路线入口 `route.md`、阶段执行细则 `stages/`、路线专属知识 `references/`）；可复用的公共资产在 `references/`，由路线声明适用范围后引用。

## Step R：迁移路线判定（必须先做，不许猜）

按用户工程的**源平台 → 目标平台**判定路线：

| 源平台 → 目标平台 | 路线包（判定后 MUST READ 其 `route.md`） |
|---|---|
| DAV_2201 → DAV_3510 | [routes/2201-to-3510/route.md](routes/2201-to-3510/route.md) |
| DAV_3510 → DAV_2201 | [routes/3510-to-2201/route.md](routes/3510-to-2201/route.md) |
| DAV_2201 → DAV_2002 | [routes/2201-to-2002/route.md](routes/2201-to-2002/route.md) |
| DAV_3510 → DAV_9201 / DAV_9202 | [routes/3510-to-920x/route.md](routes/3510-to-920x/route.md) |

**判定规则**：

1. **★ 芯片号 ↔ 架构映射 MUST 经 `npu-arch` skill 澄清（单向依赖）**——用户只给出芯片型号或构建目标时，先按名称加载 `npu-arch` 确认架构代号再查表；映射真源及出处在该 skill，不在此重复维护。依赖不可用且用户未明确架构时向用户确认，禁止自行推断。
2. 源/目标平台组合不在表内 → 本 skill 暂不支持该迁移，向用户说明并停止。**禁止套用最近路线的差异清单**——不同架构对的 API 差异、数据通路与同步语义不同，混用即臆造架构差异（违反全局约束）。
3. 判定完成后 MUST 先读取路线包的 `route.md`，再进入 Stage 0；`route.md` 中的约束与本文「全局约束」同等强制。
4. 新增路线时新增一个路线包目录（`routes/<源代号>-to-<目标代号>/`）并在上表登记。实施方案、阶段细则、Gate 结果字段、构建模板选择与验收门槛写入路线包，公共入口保持路线无关。**路线包自包含，禁止跨路线套用架构差异和实现约束。**

> **Cube 类算子（kernel 含 `Mmad` / `LoadData` / `Fixpipe` 等）**：MUST 同时阅读所属 `route.md` 指定的 Cube 指南，检查数据通路、排布与同步；实现资料按该路线组合。

## 外部依赖：asc-devkit 全量仓

本 skill 依赖 asc-devkit 全量仓提供 API 文档、样例代码和头文件。

| 属性 | 值 |
|------|-----|
| **仓库地址** | https://gitcode.com/cann/asc-devkit |
| **用途** | API 文档（4700+）、算子样例（1800+）、头文件（600+）、实现源码（4400+） |
| **路径变量** | `$DEVKIT_PATH`（在 Stage 0 中由用户指定或 git clone 后记录） |

### 依赖初始化

在 Stage 0 环境检查阶段，按路线要求确认参考仓及实际编译用 devkit 的可用版本与路径；不能用参考仓的新声明证明旧工具链已支持。
详见路线包内 `routes/<路线>/stages/stage_0_env.md` 的依赖检查要求。

### 文档查找协议

1. **必读索引**：每次需要查找文档前，先参考 `references/knowledge-index.md`
2. **分级查找**：按 `search-rules.md` 定义的优先级查找
3. **目录映射**：不确定去哪个目录时，查 `references/devkit-path-map.md`
4. **置信度评估**：每次查找后评估置信度（HIGH/MEDIUM/LOW），LOW 时扩大搜索或上报用户
5. **LOADED 标记**：从全量仓读取文件后，输出 `[LOADED] $DEVKIT_PATH/<相对路径>`

### 全量仓提供的核心资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| API 文档 | `$DEVKIT_PATH/docs/zh/api/` | SIMD/SIMT/高阶 API 完整文档 |
| 编程指南 | `$DEVKIT_PATH/docs/zh/guide/programming_guide/` | 编程模型、语言扩展、硬件实现 |
| 迁移指南 | `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/` | 跨代迁移详述（各路线对应其架构对章节） |
| 算子实践 | `$DEVKIT_PATH/docs/zh/guide/operator_practice/` | 90+ 实现与优化指南 |
| 算子样例 | `$DEVKIT_PATH/examples/` | SIMD/SIMT/AICPU 样例代码 |
| 头文件 | `$DEVKIT_PATH/include/` | API 声明 |
| 实现源码 | `$DEVKIT_PATH/impl/` | API 实现 |

## 阶段执行流程

**MUST 按顺序执行。阶段文件在所选路线包内（`routes/<路线>/stages/`）；每个阶段开始时先读取对应 stage 文件，完成时输出 Gate Token。**

| 阶段 | 文件（路线包内） | Gate Token |
|------|------|------------|
| 0 环境验证 | `routes/<路线>/stages/stage_0_env.md` | `[GATE-0] ENV_CONFIRMED` |
| 1 迁移评估 | `routes/<路线>/stages/stage_1_assess.md` | `[GATE-1]` 的结果字段与通过条件由所属 `route.md` 和本阶段文件定义 |
| 2 代码改造 | `routes/<路线>/stages/stage_2_implement.md` | `[GATE-2] FILES_MODIFIED=[...]` |
| 3 编译安装 | `routes/<路线>/stages/stage_3_build.md` | `[GATE-3] BUILD=PASS INSTALL=PASS` |
| 4 精度验证 | `routes/<路线>/stages/stage_4_precision.md` | `[GATE-4] PRECISION=N/N_PASS`（仅当 N_FAIL==0 时允许输出 PASS；否则输出 `PRECISION=N/M_PASS BLOCKED`）。用例 ≥30、覆盖全部契约 dtype/layout、通过 EXEC_NPU_CMD 绑定的 `torch.ops.npu` 调用真实实现；规模限制和专项用例按路线定义 |
| 5 性能验证 | `routes/<路线>/stages/stage_5_performance.md` | `[GATE-5] PERF=COLLECTED`（仅当 10 项证据全部存在时允许输出；否则输出 `PERF=NOT_COLLECTED BLOCKED`） |
| 6 收尾阶段（**由路线包定义**） | 阶段文件、执行条件及目标见所属 `route.md` | `[GATE-6]` 的结果字段与通过条件由路线定义；不得跨路线套用必做性、预算或收益阈值 |

## Reference 文件索引

### 路线专属内容

实施方案的选择规则、API 差异适配指南、算子实现指南、构建模板选择、测试规模及阶段 6 判据，全部在所选路线包内（`route.md`、`stages/` 与 `references/`），不在此逐路线展开。

### 构建与安装（阶段 3，按路线选用）

| 文件 | 用途 |
|------|------|
| `references/build_system/build_and_install.sh.template` | 构建安装模板；是否适用、模板选择及工程参数由所属路线 Stage 3 明确 |
| `references/build_system/build-troubleshooting.md` | 安装/构建类故障排查（路径拼接、runtime 回退内置 kernel、stub 库缺失、nm 不可见、依赖下载失败） |

### 精度测试（阶段 4 MUST READ，路线无关）

| 文件 | 用途 |
|------|------|
| `references/precision-testing/pytorch-binding-build-guide.md` | PyTorch 绑定构建（唯一标准方式） |
| `references/precision-testing/torch_aclnn_helper.h.template` | EXEC_NPU_CMD 桥接头文件模板 |
| `references/precision-testing/OPS_PRECISION_STANDARDS.md` | 精度标准（混合容差 rtol/atol + 双门限，真源 ops-precision-standard） |
| `references/precision-testing/test_op_precision_aclnn_template.py.template` | pytest 测试模板 |
| `references/precision-testing/aclnn-interface-guide.md` | aclnn 接口调用规范与 Python 调用方式确认 |
| `references/precision-testing/precision-test-pre-validation-guide.md` | 写测试脚本前的小规模数值前置验证（MUST） |
| `references/precision-testing/source-code-reverse-analysis.md` | 无文档算子的源码逆向方法论（阶段 1 生成分析摘要用） |
| `references/precision-testing/precision_report_template.md` | 精度报告格式模板 |

### 外部 Sub-Skill：RegBase 最佳实践（实际改写 RegBase 时）

| 属性 | 值 |
|------|-----|
| **Skill** | `ascendc-regbase-best-practice` |
| **入口文件** | `references/regbase_development_guide.md`（四层模型） |
| **按需查阅** | `references/api/`（白名单、MemBase 对照、同步）、`references/pitfalls/`（精度陷阱）、`references/dev-experience/`（编程经验） |

**实际改写 RegBase 时 MUST 先读取其 `references/regbase_development_guide.md`，再核对目标版本 API。** 编程模型选择与改造要求由所属路线定义。

### 外部 Sub-Skill：API 最佳实践（跨代迁移 API 知识）

API 用法知识（参数语义/签名/模式表）沉淀于 `ascendc-api-best-practices` 的 `references/`；引用前核对资料声明的架构与 SDK 版本范围：
- `api-cross-gen-migration.md` — Subnormal 与超越函数差异（跨代迁移 API 差异适配的 API 知识真源）
- `api-cross-gen-fixpipe.md` — Fixpipe 参数字段与单位差异（L0C 回写参数真源）

实际涉及 SIMT 路径时查询 `ascendc-simt-best-practices`，并核对目标版本的 API 与实现。

## Gate 协议

1. **先读 stage 文件，再执行工作**——每个阶段开始时 MUST 读取对应的 stage 文件
2. **LOADED Token**——stage 文件中要求加载的 reference 文件，加载后 MUST 输出 `[LOADED] <文件名>`
3. **GATE Token**——阶段完成时 MUST 输出对应 `[GATE-X]` token，使用所属路线定义的结果字段与实际结果值
4. **禁止跳阶段**——没有上一阶段的 GATE Token，禁止执行下一阶段
5. **禁止跳过 LOADED**——没有所有要求的 LOADED Token，禁止继续当前阶段
6. **★ 路线差异 LOADED**——阶段 1 MUST 加载路线文件声明的 API 差异适配指南并输出其 LOADED Token
7. **★ Evidence-based Gate（全局约束）**——任何 Gate 不得仅依据以下因素判定 PASS：
   - Todo completed / checkbox checked
   - Agent 自我声明（"阶段已完成"、"预计没问题"）
   - 静态代码分析（不实际执行编译/测试）
   - "已知限制"标注（不经过根因分析就标注为限制）

   每个 Gate MUST 有 Skill 明确要求的实际证据（编译产物、测试输出、CSV 数据、报告文件等）。证据不足时 MUST 保持 `NOT VERIFIED` / `NOT PASSED` / `BLOCKED` 状态，不得输出 PASS。

8. **★ GATE-4 严格通过条件**——`[GATE-4] PRECISION=N/N_PASS` 仅当满足以下全部条件时允许输出：
   - 所有计划用例均已执行（无 SKIPPED / NOT RUN）
   - N_PASS == N_TESTS（N_FAIL == 0）
   - 未通过修改 threshold / 跳过 case / 修改输入数据等方式规避失败
   - 若存在 FAIL：MUST 输出 `[GATE-4] PRECISION=N/M_PASS BLOCKED`（M < N），并进入所属路线的 `stage_4_precision.md` 根因分析流程

9. **★ GATE-5 证据约束**——`[GATE-5] PERF=COLLECTED` 仅当 10 项证据全部存在时允许输出（见路线包 `routes/<路线>/stages/stage_5_performance.md` Gate 输出条件）。缺少任一证据时 MUST 输出 `[GATE-5] PERF=NOT_COLLECTED BLOCKED`。

## 全局约束

路线包 `route.md` 中的「路线专属约束」与本节同等强制，随 Step R 选定的路线生效。

1. 未完成阶段 0 环境验证前，禁止执行任何编译或测试操作
2. 所有 Shell 命令 MUST 使用阶段 0 记录的变量（`${CANN_SET_ENV}`、`${PYTHON_PATH}`）
3. 禁止自行探测 CANN 路径——检测不到时 MUST 向用户询问
4. 精度测试 MUST 通过 `torch.ops.npu.算子名(...)` 调用（EXEC_NPU_CMD 标准方式）
5. 禁止使用 PyTorch 原生同名接口替代被测迁移实现；CPU 参考可按契约使用 PyTorch
6. 禁止手动写 aclnn C API 调用（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）
7. 编译通过不算完成——MUST 通过精度验证
8. 按所属路线 Stage 3 的模板生成并使用 `build_and_install.sh`，参数须由实际构建帮助与环境确认
9. 精度测试用例数 MUST ≥ 30
10. **★ 精度测试中 NaN/Inf 输出须按原契约核对并归类**——非预期特殊值在根因明确前不得输出 GATE-4 PASS；按所属路线的精度阶段分析，禁止直接用"已知限制"跳过
11. **★ 所有 Gate 均为 evidence-based**——禁止仅依据 Todo completed / Agent 自我声明 / 静态分析 / "预计没问题"判定 PASS。证据不足时保持 NOT VERIFIED / NOT PASSED / BLOCKED（见 Gate 协议第 7 条）
12. **禁止臆造新旧平台架构差异**——MIX 比例（1:1/1:2）、核映射等以官方迁移指导与 API 文档为准，不得凭推断断言；路线未覆盖的架构对不得套用其他路线的差异清单
13. **★ 阶段 6 按所属路线执行**——能力选择、尝试与推迟条件、预算、保留和回退判据均以路线包为准；不得把某条路线的强制项或数值门槛作为全局要求。

## 绝对不要做（每条含替代方案）

路线专属禁忌随路线包 `route.md` 生效；本节为路线无关部分。

| 禁止 | 应该做 |
|------|--------|
| 跳过阶段 0 环境验证 | 执行路线包内 `routes/<路线>/stages/stage_0_env.md` 完整流程，输出 GATE-0 |
| 跳过 Step R 直接开迁，或对表外架构对套用既有路线 | 按 Step R 判定路线并读取路线包 `route.md`；表外组合向用户说明并停止 |
| 跨路线套用 API 差异清单或实现约束 | 只使用所选路线包内的路线专属内容；公共资产按路线声明引用 |
| 自行探测 CANN 路径（`find`/`locate`/扫描 `/home`） | 检测不到时向用户询问，不得猜测 |
| 在 Shell 命令中写死路径 | 使用阶段 0 记录的 `${CANN_SET_ENV}`、`${PYTHON_PATH}` 变量 |
| 用 `torch.gather`/`torch.acosh` 做精度测试 | 用 `torch.ops.npu.算子名(...)`（EXEC_NPU_CMD 绑定） |
| 用 `import ascend_kernel` 或 `torch.ops.aclnn.*` | 用 `torch.ops.npu.算子名(...)`（注册到 npu 命名空间） |
| 手写 `aclnnXxxGetWorkspaceSize` + `aclnnXxx` | 用 `EXEC_NPU_CMD(aclnnXxx, ...)` 宏（参考 `torch_aclnn_helper.h`） |
| 同一算子调用两次 `add_modules_sources` | 合并为一次调用，所有参数放在同一条中 |
| 用 `""` 作为 `TILING_DIR` 元素（被 CMake 吞掉） | 用 `"default"` 代替空字符串 |
| 用 `-soc=` 单横线 | 用 `--soc=<目标 soc>` 双横线，目标值按路线及构建帮助确认 |
| 跨平台照搬 dtype 转换链 | 按所属路线的 API 资料核对目标支持的转换组合与精度约束 |
| `SetCtrlSpr` 后不恢复 | 先 `GetCtrlSpr` 保存，计算完 `SetCtrlSpr` 恢复 |
| 绑定用 `.asc` 文件或 ASC 编译器 | 用 `.cpp` + 纯 C++ 编译（`LANGUAGES CXX`） |
| 用 `TORCH_LIBRARY(custom_ops, m)` 注册 | 用 `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` |
| 假设 aclnn 接口名 = 底层 L0 算子名 | 读 aclnn L2 源码（`op_host/op_api/aclnn_*.cpp`）确认目标平台调度路径 |
| **★ 精度测试出现 NaN/Inf 后直接标注"已知限制"** | **MUST 执行根因分析（除零/溢出/下溢/eps/dtype 转换），归类后才可标注** |
| **★ Gate 仅依据 Todo completed / 自我声明判定 PASS** | **MUST 有实际证据（编译产物/测试输出/CSV/报告文件），证据不足时保持 BLOCKED** |
| 未核查就断言新旧平台的核映射或 MIX 比例差异 | 按所属路线的 Cube 指南、目标官方 API 与具体模板验证 |
| 不按所属路线完成收尾 | 按 `route.md` 的执行条件进入阶段 6，并依据该路线的证据要求报告结果 |
| 无停止条件地反复调试某个新特性 | 按所属路线与任务预算停止、恢复并记录 |
| 为保留新特性而只跑精度子集、放宽阈值、删用例 | 全量复跑 `N_PASS == N_TESTS`，否则回退 |
| 凭芯片型号自行推断架构代号 | 用 `npu-arch` 的映射及来源澄清，再按 Step R 查表；未知字段保留未知 |
