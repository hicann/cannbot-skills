# Catlass · Subagent 调用参数详情

本文件是 CANNBot 调用各阶段 Subagent 的**唯一执行手册**。每个 Step 包含：调用参数、关联 Skill、完成验证、约束提醒。

---

## Step 2：设计

### Subagent 调用参数

```
{
  "description": "catlass 算子方案设计",
  "subagent_type": "catlass-op-generator:catlass-op-architect",
  "prompt": "
请为以下 catlass 算子设计方案：
- 算子名称：{operator_name}（必须含 `catlass` 子串）
- 需求描述：{用户需求}
- 环境信息：operators/{operator_name}/docs/environment.json
- catlass 源码：./catlass/include、./catlass/examples、./catlass/docs

【必读 Skill】
- /catlass-op-design — 加载并按 skill 内「选型方法」完成 ArchTag / BlockMmad / BlockEpilogue / BlockScheduler / Kernel 选型与参考 example 锁定（强制）

【输出】
- 技术设计：operators/{operator_name}/docs/DESIGN.md，参考 `workflows/templates/design-template.md`
- 开发计划：operators/{operator_name}/docs/PLAN.md，参考 `workflows/templates/plan-template.md`

【验收标准】
- DESIGN.md 与 PLAN.md 都已创建（**禁止合并为单文件**）
- DESIGN.md 包含：
  - §0 概述（含 catlass 命名校验结果，op_name 含 `catlass`）
  - §1.1 数学公式
  - §1.2 Catlass 组件选型表（ArchTag / BlockMmad（DispatchPolicy + L1/L0 TileShape + AType/BType/CType）/ BlockEpilogue + Tile 槽序列 / BlockScheduler / Kernel）
  - §1.3 参考 example 路径与选型理由
  - §1.4 Kernel 适配方案（catlass example main() → op_kernel device 调用 的拆分思路）
  - §1.5 BlockEpilogue 槽位清单（如有）
  - §1.6 自定义 Tile 契约（如有，按 `/catlass-op-design` references/custom-epilogue.md 写头文件骨架）
  - §2.1 TilingKey 分支条件与合法组合
  - §2.2 Workspace 量级来源（`AscendC::GetUserWorkspace`）
  - §2.3 实现约束（C3/C4/C6 等 catlass 禁项）
- PLAN.md 包含：文件清单、catlass 编译选项（`-I./catlass/include` + `-DCATLASS_ARCH=<arch>`）、catlass kernel 运行期 shape 约束（避免过小 M/N，选 L1 分块整数倍）

【约束】
- 禁止：写实现代码（设计阶段只产文档）
- 禁止：使用 catlass `DeviceGemm` 适配器（仅 example 用）
- 禁止：设计在 op_kernel 中自实现矩阵乘 / 逐元素 / 拷贝循环
- 必须：BlockEpilogue 槽位形参打开 `catlass/include/catlass/epilogue/block/block_epilogue_<policy>.hpp` 读出
  "
}
```

---

## Step 2.5：设计串讲

### 2.5a — Developer 串讲审查
```
{
  "description": "catlass 设计串讲",
  "subagent_type": "catlass-op-generator:catlass-op-generator",
  "prompt": "
请以「设计串讲模式」审查以下 catlass 算子的设计方案：
- 算子名称：{operator_name}
- 技术设计：operators/{operator_name}/docs/DESIGN.md
- 开发计划：operators/{operator_name}/docs/PLAN.md
- catlass 源码（对照）：./catlass/include、./catlass/examples

【重点审查章节】DESIGN.md §1.2 catlass 选型表 / §1.3 参考 example / §1.5 BlockEpilogue 槽位清单 / §1.6 自定义 Tile 契约 / §2.1 TilingKey 分支条件 / §2.2 Workspace。§0、§1.1、§2.3 通览即可。

【输出】
- 质疑清单输出到 operators/{operator_name}/docs/WALKTHROUGH.md

【推荐 Skill】
- /catlass-op-develop — 质疑选型可实现性时对照 skill 中 op_kernel 拼装规则
- /ascendc-api-best-practices — 质疑自定义 Tile 内 API 选择时查阅
- /ascendc-docs-search — 需要官方文档支撑质疑时使用

【catlass 专项审查重点（6 项）】
| 序号 | 审查维度 | 审查方法 |
|------|---------|---------|
| 1 | catlass 选型可实现性 | 参考 example（§1.3）是否真能拆为 op_kernel 直接 `Kernel{}(params)` 调用？是否仍依赖 `DeviceGemm` 适配器？ |
| 2 | BlockEpilogue 槽位匹配 | §1.5 列出的槽位与 `catlass/include/catlass/epilogue/block/block_epilogue_<policy>.hpp` 是否一致？ |
| 3 | 自定义 Tile 契约 | §1.6 头文件骨架的 DispatchPolicy 类别 / `operator()` 签名是否与槽位期望严格对齐？ |
| 4 | TilingKey 分支覆盖 | §2.1 是否覆盖所有 dtype / 转置 / Swizzle 组合？是否漏掉 host Tiling 分支落点？ |
| 5 | Workspace 计算 | §2.2 Workspace 量级是否清晰？kernel 内是否明确 `AscendC::GetUserWorkspace`？ |
| 6 | 精度策略 | catlass GEMM 精度阈值是否对齐 `ops-precision-standard`？无 catlass 专属放宽规则 |

【WALKTHROUGH.md 输出格式】
输出到 operators/{operator_name}/docs/WALKTHROUGH.md，使用以下结构：

## 设计串讲

### 审查结论
- [ ] 设计可直接开发（无阻塞问题）
- [ ] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

### 质疑清单

#### 问题 1：[简述]
- **类别**：catlass 选型可实现性 / BlockEpilogue 槽位 / 自定义 Tile 契约 / TilingKey 分支 / Workspace / 精度
- **严重程度**：阻塞 / 需讨论 / 建议
- **设计文档位置**：DESIGN.md §X
- **问题描述**：...
- **Developer 视角**：为什么从开发者角度认为这是问题
- **建议方案**：（如有）

【串讲模式约束】
- 禁止：在串讲模式下编写开发代码
- 禁止：直接修改 DESIGN.md（修改由 Architect 在回应模式中完成）
- 必须：每个问题标注严重程度
- 必须：catlass 选型 / BlockEpilogue 槽位类问题需附 catlass header 行号或 example 路径作为依据
- 鼓励：对每个问题提出建议方案，帮助 Architect 快速回应
  "
}
```

### 2.5c — Architect 串讲回应

```
{
  "description": "catlass 串讲回应",
  "subagent_type": "catlass-op-generator:catlass-op-architect",
  "prompt": "
请以「串讲回应模式」回应 Developer 对 catlass 设计方案的质疑：
- 算子名称：{operator_name}
- 技术设计：operators/{operator_name}/docs/DESIGN.md
- 串讲质疑：operators/{operator_name}/docs/WALKTHROUGH.md
请逐一回应质疑，并根据需要更新 DESIGN.md。

【输出】
- 更新 operators/{operator_name}/docs/WALKTHROUGH.md（追加 ### Architect 回应）
- 如需修改，更新 operators/{operator_name}/docs/DESIGN.md

【验收标准】
- 每个质疑都有回应（接受 / 保留原设计 + 理由 / 部分修改）
- catlass 选型类问题的「保留原设计」必须附 `catlass/include/`、`catlass/examples/` 或 `catlass/docs/` 中具体路径作为依据

【回应执行步骤】
1. 读取 WALKTHROUGH.md ## 质疑清单
2. 逐一评估，判定回应类别：

| 回应类别 | 含义 | 操作 |
|---------|------|------|
| 接受 | Developer 的质疑合理 | 更新 DESIGN.md 对应章节 |
| 保留原设计 | 原设计正确，给出理由 | 不修改 DESIGN.md，附 catlass / asc-devkit 文档依据 |
| 部分修改 | 部分采纳 | 更新 DESIGN.md 中受影响的部分 |

3. 在 WALKTHROUGH.md 中追加「### Architect 回应」子章节
4. 返回概要：接受/保留/部分修改的问题数量、DESIGN.md 是否有更新

【回应输出格式】
### Architect 回应

#### 问题 1：[简述]
- **回应**：已修改 / 保留原设计 / 部分修改
- **理由**：...
- **文档依据**：（catlass header / example 路径，或 asc-devkit/docs/api/context/ 路径）
- **DESIGN.md 变更**：（描述修改内容，或"无变更"）

### 回应统计
- 接受 X 项，保留 Y 项，部分修改 Z 项

【回应约束】
- 必须：对每个阻塞问题给出明确回应，不可跳过
- 必须：保留原设计时附上具体的 catlass / asc-devkit 文档依据
- 必须：接受时同步更新 DESIGN.md 对应章节
- 鼓励：对建议类问题也给出简短回应
  "
}
```

---

## Step 3：开发

### Subagent 调用参数

```
{
  "description": "catlass 算子开发",
  "subagent_type": "catlass-op-generator:catlass-op-generator",
  "prompt": "
请先阅读以下文件：
- operators/{operator_name}/docs/DESIGN.md — 技术设计（重点 §1.2 catlass 选型 / §1.3 参考 example / §1.4 Kernel 适配方案 / §1.5 BlockEpilogue 槽位 / §1.6 自定义 Tile 契约 / §2.1 TilingKey 分支 / §2.2 Workspace）
- operators/{operator_name}/docs/PLAN.md — 开发计划（请在开发中持续更新）
- operators/{operator_name}/docs/environment.json — 编译器/架构信息
然后开始开发。

【必读 Skill】
- /catlass-op-develop — 加载并按 skill 内「核心工作流」执行 op_kernel 内 catlass 模板拼装与 Device 调用（强制）

【渐进式开发策略（每步必须编译通过后再进入下一步）】
Step A：基于 §1.3 选定的参考 example 起工程骨架 → 编译通过（空 Kernel）
Step B：写 op_host Tiling 计算 + ACL 框架 → 编译通过
Step C：写 op_kernel catlass 拼装类 + kernel 入口分支 + Device 调用 → 编译通过
Step D：（如有）落盘自定义 Tile 头文件 → 编译通过
Step E：补 gen_data.py / verify_result.py / run.sh，跑通 Level 0~2 测试

【catlass 实现强制项】
- 直接实例化 `Kernel` + `Kernel::Params`，`Kernel{}(params)`；**禁用** `DeviceGemm` 适配器
- op_kernel 内**禁止**自实现矩阵乘 / 逐元素 / 拷贝循环（只能用 catlass `Kernel`/`Block*`/`Tile*`）
- Workspace 必须用 `AscendC::GetUserWorkspace(workspace)`；**禁用** `SetSysWorkspaceForce`
- op_kernel **禁止** `#include` 算子自身的 tiling 实现文件（仅可 include 共享 POD `*_tiling.h`）
- CMakeLists.txt 必须用标准 Ascend C CMake 构建，仅追加 catlass 编译选项：`-I${CMAKE_SOURCE_DIR}/../../catlass/include` + `-DCATLASS_ARCH=<架构号>`。**禁止**使用 catlass 仓库自身的 CMake 函数（它们是 example 构建辅助，不适用于算子工程）。

【参考文档】
- 编码规范与审查清单：workflows/development-guide.md
- 工程模板：workflows/templates/

【输出】
- 算子代码：operators/{operator_name}/（含 .asc kernel + .asc host、CMakeLists.txt、run.sh、gen_data.py、golden.py、verify_result.py）
- 更新进度：operators/{operator_name}/docs/PLAN.md

【验收标准】
- 编译成功（cmake .. && make）
- Level 0（M/N/K = L1 分块整数倍）测试通过
- Level 1 覆盖每个 §2.1 列出的 TilingKey 分支至少一组
- PLAN.md 已更新进度
  "
}
```

---

## Step 4：审查

### Subagent 调用参数

```
{
  "description": "catlass 代码审查",
  "subagent_type": "catlass-op-generator:catlass-op-reviewer",
  "prompt": "
请审查以下 catlass 算子代码：
- 算子名称：{operator_name}
- 代码路径：operators/{operator_name}/
- 设计文档：operators/{operator_name}/docs/DESIGN.md
- 环境信息：operators/{operator_name}/docs/environment.json
- 通用审查清单：workflows/references/review-checklist.md
- catlass 源码（对照）：./catlass/include、./catlass/examples

【输出】
- 审查报告：operators/{operator_name}/docs/REVIEW.md

【推荐 Skill】
- /ascendc-docs-search — 验证非 catlass API 约束
- /ops-profiling — 独立采集 msprof 性能数据
- /ops-precision-standard — 精度阈值确认

【catlass 专项检视项 C1–C11】（必须逐条覆盖并在 REVIEW.md 中列表呈现）
- C1 命名含 `catlass`，snake_case ↔ CamelCase 一致映射
- C2 catlass 源码位于 `./catlass/`，未克隆到 `operators/{operator_name}/` 内
- C3 CMakeLists.txt 注入 `-I<catlass>/include` + `-DCATLASS_ARCH=<arch>`
- C4 op_kernel 直接 `Kernel` + `Kernel::Params` + `Kernel{}(params)`；禁用 `DeviceGemm` 适配器
- C5 op_kernel 不自实现矩阵乘 / 逐元素 / 拷贝循环
- C6 Workspace 用 `AscendC::GetUserWorkspace(workspace)`；禁用 `SetSysWorkspaceForce`
- C7 op_kernel 不 include 算子自身的 tiling 实现文件（仅可 include 共享 POD `*_tiling.h`）
- C8 TilingKey 分支与 DESIGN.md §2.1 合法组合一致
- C9 测试 shape 满足 catlass 运行期约束（避免过小 M/N，选 L1 分块整数倍）
- C10 catlass 拼装类 `using` 与 DESIGN.md §1.2 选型表一致
- C11 调优场景已加载 `/catlass-op-perf-tune`，PRE/POST 报告已归档

【验收标准】
- 独立编译验证（含 catlass 编译选项校验：`workflows/scripts/verify_cmake_config.py`）
- C1–C11 逐条覆盖
- 100 分制评分
- PASS / FAIL / PASS WITH NOTES 判定
- 具体修复要求（如 FAIL）
  "
}
```

---

## Step 5：修复循环

> ⚠️ **CANNBot 禁止自行修改代码，即使修复看起来只有一行。必须调用 Developer Subagent。**

### Subagent 调用参数

```
{
  "description": "catlass 代码修复",
  "subagent_type": "catlass-op-generator:catlass-op-generator",
  "prompt": "
请根据审查报告修复代码：
- 算子名称：{operator_name}
- 审查报告：operators/{operator_name}/docs/REVIEW.md（重点 catlass C1–C11 表 + 必须修复项）
- 设计文档：operators/{operator_name}/docs/DESIGN.md

【输出】
- 修复后的代码：operators/{operator_name}/
- 更新进度：operators/{operator_name}/docs/PLAN.md

【推荐 Skill】
- /catlass-op-develop — catlass 实现约束类问题（C4/C5/C6/C7/C10）的修复依据
- /ascendc-precision-debug — 精度类问题
- /ascendc-runtime-debug — 运行时问题
- /ascendc-api-best-practices — 非 catlass API 约束错误

【验收标准】
- 审查报告中所有必须修复项已处理
- catlass C1–C7 全部通过
- 编译成功
- 测试通过（Level 0–2）
  "
}
```

---

## Step 6：性能验收 / 调优

### Subagent 调用参数

```
{
  "description": "catlass 性能验收",
  "subagent_type": "catlass-op-generator:catlass-op-generator",
  "prompt": "
请执行性能采集和验收：
- 算子名称：{operator_name}
- 算子目录：operators/{operator_name}/
- 设计文档：operators/{operator_name}/docs/DESIGN.md（§1.2 选型 / §1.3 参考 example）

【输出】
- 性能数据：operators/{operator_name}/docs/perf/round_NNN/
- 性能摘要：operators/{operator_name}/docs/perf/round_NNN/summary.txt
- 如调优：PRE/POST 对比报告

【推荐 Skill】
- /ops-profiling — msprof op 采集、CSV 解读
- /catlass-op-perf-tune — 调优场景必加载，按 `catlass/docs/1_Practice/10_matmul_optimization.md` 执行

【调优规则】
- 每次只动一个变量（DispatchPolicy / TileShape / Swizzle / Kernel 之一），便于归因
- 性能下降 → 立即回滚到上一稳定配置
- PRE/POST 两份 profiler 数据均归档到 `operators/{operator_name}/docs/perf/round_NNN/`

【验收标准】
- 性能数据已归档
- 达标判定已记录
- 与 catlass 同形态 example 基线对比已记录
- 如调优：PRE/POST 对比与配置变更日志已落盘
  "
}
```

---

## 报告格式通用规范

所有验收报告必须包含以下字段，供 CANNBot 解析判断：

```markdown
**状态**: ✅通过 / ❌失败

**catlass C1–C11 状态**:
| # | 检视项 | 状态 |
|---|--------|------|
| C1 | 命名含 `catlass` | 通过/失败 |
| ... | ... | ... |

**验证摘要**:
| 验证项 | 结果 | 详情 |
|-------|------|------|
| ... | 通过/失败 | ... |

**关键指标**:
- 总用例数: X
- 通过数: Y
- 失败数: Z
- 通过率: X%

**性能概要**
- Task Duration
- 主导流水
- 与 catlass example 基线差距
- 达标状态

**失败用例**（如有）:
- 列出失败的测试用例及原因
```

**重要约束**：
- 如有失败用例，状态必须标记为 `❌失败`，禁止标记为 `✅通过`
- 仅编译通过不等于验证通过，必须实际运行测试
- catlass C1–C7 任一失败 → 状态必须为 `❌失败`
