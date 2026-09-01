---
name: ops-direct-invoke-workflow
description: 直调算子开发工作流编排技能，承载从算子需求分析到上库的完整流程。触发：用户要求开发新算子、实现某算子接口，或推进算子开发流程时。
---

# 直调算子开发工作流

> 本 skill 承载**整个工作流编排**。加载本 skill 的 Agent 担任 PM 的角色，按下方**统一流程表**逐步调度**算子开发团队角色**完成算子开发。

## 算子开发团队角色总览

以下是所有可供调度的 Agent 角色：

| 角色 | 类型 | 职责 |
|------|------|------|
| **PM**（你自己） | 调度 | 用户交互、流程编排、问题裁定 |
| **architect** | 执行 | 需求分析、开发方案与测试方案设计 |
| **developer-code** | 执行 | 算子代码开发、问题定位 |
| **developer-test** | 执行 | golden、功能/性能用例、白盒测试补全 |
| **developer-doc** | 执行 | 算子文档、开发报告、经验总结 |
| **developer** | 执行 | 跨代码/测试/文档的综合任务，**仅在前三类 Agent 权限不满足任务要求时使用** |
| **QA** | 验收 | 各 CP 点验收，加载对应 `workflow-cp*` Skill 完成判定 |

## 统一流程表

| 编号 | 流程 | 角色 | 输入 | 输出 | 说明 | 备注 |
|------|------|------|------|------|------|------|
| **阶段0：开发准备** | | | | | | |
| 0 | 开发准备 | developer | - | 环境信息文档 | 检查 NPU 设备、CANN 环境、编译环境，并检测 Git 凭据位置候选（只记位置不取内容），**把全部环境信息统计到** `.cannbot/环境信息.md`（共享留根）；该文件为环境信息**唯一来源**——后续子任务的环境信息一律从该文件获取、禁止自行探索环境，缺项须向 PM 申请探索、批准后探索且结果回填文件并追加「环境补充记录」（构建脚本运行期自检属工具链行为，不视为探索） | 交付件可复用；`.cannbot/环境信息.md` 已存在且环境未变时跳过 0 与 CP0，新算子直入 1.1 |
| ⛔ CP0 | 环境确认 | QA | 环境信息文档 | 用户问卷 json | QA 加载 `workflow-cp0` 生成环境确认问卷，直接发送用户并收集结论 | 固定用户确认；有异议即停止当前算子开发；阶段 0 被跳过时一并跳过 |
| **阶段1：需求分析** | | | | | | |
| 1.1 | 需求分析 | architect | 对话上下文、仓库设计约束 | 需求文档 | 数学定义、算子原型、目标芯片、精度/性能要求；代码架构选型出推荐项与依据：候选为 **SIMD 与 SIMT 两种**——**Cube 属 SIMD 的一种实现形态**（矩阵计算单元，可单独或与 RegBase 混合使用，如 AIC Cube + AIV RegBase 的 mix 形态），不单独作为与 SIMD 并列的架构候选；SIMD 实现载体按目标芯片确定：**ascend950 为 RegBase / Cube，其余低版本芯片为 MemBase；RegBase 与 MemBase 互斥（支持 RegBase 的芯片不使用 MemBase）** | 架构选型为用户拍板项，1.1 只出推荐不做决策；缺项与开放取舍统一交 CP1 问卷确认 |
| ⛔ CP1 | 需求确认 | QA | 需求文档 | 验收结论 + 用户确认问卷 json | QA 加载 `workflow-cp1` 核对需求；核对无硬伤后按问卷模板直接发送用户确认并收集结论 | 固定用户确认；硬伤打回 1.1；架构选型未经用户拍板不得进入阶段 2 |
| **阶段2：方案设计**（方案线 / 测试线并行） | | | | | | |
| 2.1 | 黑盒测试设计 | architect | 需求文档 | 测试方案文档 | golden 实现方案、L0/L1/L2 分级用例设计 | 测试线 |
| CP2.1 | 测试检查 | QA | 测试方案文档 | 验收结论 | QA 加载 `workflow-cp2-1` 完成黑盒测试评审 | 不通过打回 2.1 |
| 2.2 | 开发方案设计 | architect | 需求文档 | 开发方案文档 | 承接需求文档已拍板的代码架构做落地细化，设计 Buffer 规划、Tiling、多核切分、Ascend C 接口验证 | 方案线；不改选架构 |
| ⛔ CP2.2 | 方案检查 | QA | 开发方案文档 | 验收结论 + 用户确认问卷 json | QA 加载 `workflow-cp2-2` 完成开发方案评审；评审通过后抽取关键决策点生成确认问卷，直接发送用户并收集结论 | 固定用户确认；QA 评审不通过打回 2.2，用户异议按语义归属回退 2.2 或 1.1 |
| **阶段3：代码开发**（开发线 / 测试线并行） | | | | | | |
| 3.1 | 算子开发 | developer-code | 开发方案文档、修改要求 | 算子代码 | 按方案实现、编译验证通过；被打回时按修改要求调整 | 开发线 |
| 3.2 | 测试工程开发 | developer-test | 测试方案文档 | golden 代码 + 用例表 + 性能采集框架 | 实现 golden、功能用例、性能采集框架 | 测试线 |
| 3.3 | 白盒测试补全 | developer-test | 算子代码、测试代码 | 白盒用例 + 分支覆盖说明 | 按源码枚举分支补全，覆盖尾核/非对齐/tilingkey 等黑盒未涉及分支 | 3.1/3.2 汇合后 |
| 3.4 | 联调 | developer-code | 算子代码 + 测试代码 | 联调报告 + 修复后的算子代码 | 以测试工程的 golden/用例对算子代码做联合调试：全量用例跑通、精度比对通过；代码侧问题直接修复并重新编译验证，测试侧问题（golden/用例/框架缺陷）不自行修改 test 目录，回退 3.2 | 3.1/3.2/3.3 汇合后；不通过按问题归属回退 3.1（代码问题）或 3.2（测试问题） |
| CP3 | 功能验收 | QA | 算子代码 + 测试代码 | 功能验收报告 | QA 加载 `workflow-cp3` 执行全量功能测试（含精度比对） | 不通过按语义归属回退 3.1（算子实现）或 2.1（精度判定口径） |
| **阶段4：性能验收** | | | | | | |
| 4.1 | 性能采集执行 | developer-test | 算子代码（CP3 通过后） | 性能数据 | 用 3.2 搭的性能采集框架跑出性能数据 | 基于 CP3 通过后的最终代码 |
| CP4 | 性能验收 | QA | 算子代码 + 性能数据 | 性能验收报告 | QA 加载 `workflow-cp4` 评估性能是否达标 | 不通过回退 3.1（重走 3.3→3.4→CP3→4.1→CP4）；验收结论为「实现级优化已穷尽，建议按已知限制收口」时不回退，按 error-handling 的需求级硬门槛放宽处理；性能迭代为可插拔步骤，未启用时按 4.1 基线数据验收 |
| **阶段5：代码检视** | | | | | | |
| CP5 | 代码检视 | QA | 全部变更文件 | 检视结论 | QA 加载 `workflow-cp5` 完成多维度代码检视 | 不通过回退 3.1（重走至 CP5） |
| **阶段6：上库准备** | | | | | | |
| 6.1 | 文档补全 | developer-doc | 算子代码 + 设计文档 | 算子文档 | 补全算子使用文档 | |
| **阶段7：开发总结** | | | | | | |
| 7.1 | 开发报告 | developer-doc→PM | 全部交付物 | 开发报告 | developer-doc 整理开发过程与交付物清单，PM 落盘 `.cannbot/<算子名>/` | |
| 7.2 | 经验总结 | developer-doc→PM | 开发过程记录 | 经验总结文档 | developer-doc 沉淀开发经验与踩坑记录，PM 落盘 `.cannbot/<算子名>/` | |

- ⛔ 标记的为**用户确认点**，QA 生成结构化问卷 json，用会话问卷工具（opencode `question` / claude `AskUserQuestion` / dsh `ask_user_question` / trae `AskUserQuestion`）直接发送用户并收集结论；问卷与用户回复成对落盘 `.cannbot/<算子名>/questionnaires/`（回复记为问卷同名 `.reply.json`）
- 可插拔流程步骤（如提交 PR 到上库、性能迭代）由 PM 按 `.cannbot/settings.json` 的 plugins 中启用的插件在其挂载点触发，不在本表列出
- 方案线与测试线在阶段 2、3 并行推进
- CP3/CP4/CP5 任一不通过均回退到算子开发（3.1）后依次重走（重走路径经 3.4 联调）

## 通用约定

- **任务清单**：PM 接到算子开发任务时，先把统一流程表的全部步骤（含各 CP 点）整体加载到自己的 todolist（任务清单工具），随流程推进逐项刷新状态，保证全程进度可见、不漏步、不跳步；可插拔流程插件触发时，其内部步骤表全部步骤并入 todolist。
- **插件注册**：PM 接到算子开发任务、启动工作流时，读取 `.cannbot/settings.json`（init Step 5.5 生成的唯一配置，插件启用判定唯一权威）：
  - **settings 缺失**：不阻塞，按无可插拔流程继续。
  - **`surveyed` 为 `false`**：首次先按 `plugins` 逐项询问用户启用哪些插件（on/off），结果写回各插件的 `enabled` 并置 `surveyed` 为 `true`；非交互场景跳过询问并保持现状。
  - **后续会话**：沿用 `plugins` 中的 `enabled`；用户要求调整时随时可改（也可用 `init.sh --plugin-enable <name> on|off`）。
  - 流程推进到某插件的挂载点时，仅触发 `enabled` 为 `true` 的插件，加载对应 `plugin-*` skill 按其内部步骤表执行；未启用的挂载点自然跳过。
- **状态与恢复**：每阶段（步骤或 CP）完成后**立即**将进度持久化到 `.cannbot/<算子名>/state.json`（结构见 [state-schema.md](references/state-schema.md)）——调度下一步前先落盘，禁止攒到算子开发完成后一次性补写。
- **中间文件**：所有过程产物统一放 `.cannbot/<算子名>/`，环境信息文档作为共享件放 `.cannbot/环境信息.md`。各角色的临时产物（一次性脚本、中间分析、日志等）统一收敛到 `.cannbot/<算子名>/tmp/`，不在 `.cannbot` 根或算子目录下散落。
- **任务下发**：严格按照 [task-prompts.md](references/task-prompts.md) 中的角色、prompt 调度子 Agent。
- **工作流模式**：启动工作流时读取 `.cannbot/settings.json`（init Step 5.5 生成；缺失或字段非法按 `interactive` 处理）。`mode=interactive`（默认）按既有约定执行（问卷直发、汇报进度）；`mode=silent` 为完全无人值守——不输出中间进度、不发问卷（⛔ 确认点由 QA 按静默默认决策执行并落盘 `.reply.json`）、失败自动回退至最大轮次，仅输出：启动时的权限预检警告（见 AGENTS.md）与任务完成总结（含阻断中止性总结）；其余一切不输出（插件内异步等待的告知归插件自身约定，见 [settings.md](references/settings.md)）。结构、默认决策与修改方式见 [settings.md](references/settings.md)。
- **任务回退**：回退修改时，复用原执行子 Agent 的会话追加 prompt，不新建会话。回退循环最大轮次见 [error-handling.md](references/error-handling.md)

## 参考资源

| 资源 | 路径 | 说明 |
|------|------|------|
| 调用契约 | [references/task-prompts.md](references/task-prompts.md) | 各步子 Agent 的调用参数、输入/输出、验收标准 |
| 数据流 | [references/data-flow.md](references/data-flow.md) | 各阶段文件 I/O 与 `.cannbot` 产物清单 |
| 错误处理 | [references/error-handling.md](references/error-handling.md) | 回退策略、最大轮次、恢复规则 |
| 状态结构 | [references/state-schema.md](references/state-schema.md) | `.cannbot/<算子名>/state.json` 字段约定 |
| 状态校验 | `scripts/validate_state.py` | state.json 结构校验（编号合法性、顺序一致性、字段结构），任意时刻可运行 |
| 可插拔流程 | `.cannbot/settings.json` | 已注册插件（`skills/plugin-*`）的挂载点、步骤编号与启用状态；PM 在流程推进到挂载点时触发启用的插件 |
| 工作流配置 | [references/settings.md](references/settings.md) | `.cannbot/settings.json` 结构、模式语义、静默默认决策 |
| 交付件模板 | `workflow-doc-templates` | 需求/方案/验收报告/README/LOG/Issue 等模板 |
| 验收标准 | `workflow-cp*` | 各 CP 点的验收对象、通过指标、判定方式；QA 在对应 CP 点加载 |
