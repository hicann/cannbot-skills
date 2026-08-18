# Task 调用契约

> 各步子 Agent 的调用参数、输入/输出、验收标准。编号与 [SKILL.md 统一流程表](../SKILL.md#统一流程表) 一一对应。
>
> **调用原则**：PM 每阶段首次调度子 Agent 时，必须严格按照本文档指定的角色和 prompt **原样调用**，仅允许替换 prompt 中的 `<算子名>` 项，**严禁干涉实现细节**。
>
> **工具差异（dsh）**：dsh（DeepSeek Harness）无命名 agent 注册（子 Agent 纯 prompt 驱动，角色定义文件位于 `.dsh/agents/<角色>.md`），PM 用 subagent 工具派发：**prompt = 对应角色定义全文（`.dsh/agents/<角色>.md` 正文）+ 本文档 prompt 原样 + 任务输入**，子 Agent 才能获得完整角色上下文；回退/续跑复用原子 Agent 会话（send_message 追加 prompt），不新建会话。**派发时 subagent 工具的 description 必须含角色名**（如「architect：需求分析」）——部署级权限守卫（`hooks/dsh/install.sh`）据此识别子 Agent 角色并按 `.cannbot/permissions/` 规则判权。
>
> **权限行的机制差异**：各 prompt 的【权限】行中「其它写入操作会被 hooks 拦截」在 opencode / claude 由 permission-guard hook 机制保证；在无项目级 hook 环境（dsh / codex）下降级为**提示性约束**——子 Agent 依据 `workflow-agent-permissions` 的规则自律，违规写入不会被拦截，PM 派发时仍须按该 skill 判定角色权限。**dsh 可选升级**：运行 `hooks/dsh/install.sh` 安装部署级守卫后，dsh 恢复机制保证（拦截语义与 opencode/claude 一致）。

# 阶段 0：开发准备

## 0 开发准备

- **角色**：developer

```md
- 【权限】你只可写 `.cannbot/环境信息.md`，其它写入操作会被 hooks 拦截。
- 【输出】填充模板：`workflow-doc-templates/references/环境信息.md`（含「环境补充记录」节），写入 `.cannbot/环境信息.md`。**把全部环境信息统计到该文件**——硬件、软件、凭据及其它后续子任务可能需要的环境项，一次统计完整；该文件是环境信息唯一来源，后续子任务的环境信息一律从该文件获取、不再自行探索环境。
- 【skills】立即加载 `workflow-doc-templates`、`ascendc-env-check`。
- 【验收标准】环境信息文档完整，各检查项有明确结论，环境项一次统计齐全；Git 凭据位置候选已列出且不含任何凭据明文。
- 【探索申请】统计后发现某环境项缺失/信息不全：不得自行探索，先向 PM 提出探索申请，PM 批准后方可探索；探索完成后把结果回填 `.cannbot/环境信息.md` 对应项，并在「环境补充记录」节追加一条记录（时间/补充项/摘要/来源）。构建/测试脚本运行期的自检属工具链行为，不视为探索。
```

## CP0 环境确认

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/questionnaires/`（问卷 json 与同名 `.reply.json`）、`.cannbot/环境信息.md`（按用户结论更新「Git 凭据」节），其它写入操作会被 hooks 拦截。
- 【输入】环境信息文档 `.cannbot/环境信息.md`。
- 【输出】环境确认结果（不通过，立即停止当前工作流流程 / 通过）；问卷与回复落盘 `.cannbot/<算子名>/questionnaires/`。
- 【skills】立即加载 `workflow-cp0`、`workflow-doc-templates`。
```

> **静默模式（`mode=silent`）**：PM 在本 prompt 末尾追加「【静默模式】不发送问卷，按 settings.md 默认决策执行」，QA 落盘 `.reply.json`（`{"mode":"silent","decision":"accepted"}`），不向用户发送问卷。

# 阶段 1：需求分析

## 1.1 需求分析

- **角色**：architect

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】对话上下文、仓库设计约束。
- 【输出】需求文档，写入 `.cannbot/<算子名>/1.1-需求分析.md`；格式模板：`workflow-doc-templates/references/1.1-需求分析.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-knowledge`、`ascendc-regbase-best-practice`、`ascendc-simt-best-practices`。
- 【验收标准】确认项无遗漏，用户原始需求逐条记录；架构选型候选为 SIMD 与 SIMT 两种——**Cube 属 SIMD 的一种实现形态**（矩阵计算单元，可单独或与 RegBase 混合使用，如 AIC Cube + AIV RegBase 的 mix 形态），不单独作为与 SIMD 并列的架构候选；SIMD 实现载体按目标芯片确定：ascend950 为 RegBase / Cube，其余低版本芯片为 MemBase；RegBase 与 MemBase 互斥（支持 RegBase 的芯片不使用 MemBase）；只出推荐，不代替用户决定架构。
```

> **静默模式（`mode=silent`）**：PM 在本 prompt 末尾追加「【静默模式】代码架构选型候选仅 SIMD（实现载体按目标芯片确定：ascend950 为 RegBase / Cube、其余芯片为 MemBase），不评估 SIMT，推荐项直接取 SIMD」。

## CP1 需求确认

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/questionnaires/`（问卷 json 与同名 `.reply.json`），其它写入操作会被 hooks 拦截。
- 【输入】需求文档 `.cannbot/<算子名>/1.1-需求分析.md`。
- 【输出】需求确认结果（不通过，打回 1.1 / 通过）；问卷与回复落盘 `.cannbot/<算子名>/questionnaires/`。
- 【skills】立即加载 `workflow-cp1`、`workflow-doc-templates`。
```

> **静默模式（`mode=silent`）**：PM 在本 prompt 末尾追加「【静默模式】不发送问卷，按 settings.md 默认决策执行」，QA 核对无硬伤即通过（架构选型固定采用 SIMD——含 Cube 等实现载体，静默下 1.1 仅推荐 SIMD、不评估 SIMT），落盘 `.reply.json`（`{"mode":"silent","decision":"accepted"}`）。

# 阶段 2：方案设计（方案线 / 测试线并行）

## 2.1 黑盒测试设计

- **角色**：architect

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】需求文档 `.cannbot/<算子名>/1.1-需求分析.md`。
- 【输出】测试方案文档，写入 `.cannbot/<算子名>/2.1-测试方案设计.md`；格式模板：`workflow-doc-templates/references/2.1-测试方案设计.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-test-develop`、`ascendc-st-design`。
- 【验收标准】golden 方案可行，分级用例覆盖充分（正常/边界/异常/特殊值），覆盖矩阵齐备。
```

## CP2.1 测试检查

- **角色**：QA

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】测试方案文档 `.cannbot/<算子名>/2.1-测试方案设计.md`。
- 【输出】测试检查结论（不通过，打回 2.1 / 通过）
- 【skills】立即加载 `workflow-cp2-1`、`workflow-doc-templates`。
```

## 2.2 开发方案设计

- **角色**：architect

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】需求文档 `.cannbot/<算子名>/1.1-需求分析.md`。
- 【输出】开发方案文档，写入 `.cannbot/<算子名>/2.2-开发方案设计.md`；格式模板：`workflow-doc-templates/references/2.2-开发方案设计.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-knowledge`、`repo-op-templates`、`repo-build-guide`、`ascendc-tiling-design`。
- 【验收标准】代码架构与需求文档拍板结果一致，Tiling/切分策略可行，关键 API 已验证；不改选已拍板的代码架构。
- 【能力边界前置】方案依赖的硬件与编译器能力项（数据类型转换链及其舍入语义、搬运/广播/掩码接口的对齐与地址约束、编译器对手工指令收敛的调度行为、目标架构特有限制）须在本阶段核对出「支持 / 不支持 / 需绕行」结论并注明核对方式，不支持项给绕行方案；舍入语义与 golden 不等价的转换链等同不支持。不留「应该可以 / 待定」。
- 【瓶颈预判可证伪】瓶颈维度预判须给出可核对的量化估算依据与证伪条件，并标注校准状态为「未实测」；不接受无推导的定性断言。实测数据与预判矛盾时以实测为准，回改本节。
- 【优化项落地要求】性能优化项逐条标注落地要求（必落地 / 可选）与是否改变数值计算路径；「必落地」项只面向预判的瓶颈维度。
```

## CP2.2 方案检查

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/questionnaires/`（问卷 json 与同名 `.reply.json`），其它写入操作会被 hooks 拦截。
- 【输入】开发方案文档 `.cannbot/<算子名>/2.2-开发方案设计.md`。
- 【输出】方案检查结果（不通过，按语义归属打回 2.2 或 1.1 / 通过）；问卷与回复落盘 `.cannbot/<算子名>/questionnaires/`。
- 【skills】立即加载 `workflow-cp2-2`、`workflow-doc-templates`。
```

> **静默模式（`mode=silent`）**：PM 在本 prompt 末尾追加「【静默模式】不发送问卷，按 settings.md 默认决策执行」，QA 评审通过即视为用户确认，落盘 `.reply.json`（`{"mode":"silent","decision":"accepted"}`）。

# 阶段 3：代码开发（开发线 / 测试线并行）

## 3.1 算子开发

- **角色**：developer-code

```md
- 【权限】你可写算子代码目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】开发方案文档 `.cannbot/<算子名>/2.2-开发方案设计.md`；被打回时附结构化修改要求。
- 【输出】算子代码：按开发方案实现，编译验证通过。
- 【skills】立即加载 `repo-op-templates`、`repo-coding-rules`、`repo-build-guide`、`repo-knowledge`、`ascendc-direct-invoke-template`、`ascendc-api-best-practices`。
- 【验收标准】编译通过，按方案实现；性能/正确性瓶颈定位后回退给调用方，不自行改 Tiling/切分/接口。
- 【优化项逐项核对】交付前对开发方案「性能优化项」表逐条回填落地状态（已落地 / 未落地及原因 / 放弃及原因）；「必落地」项不得静默不做。回填结果随交付结论回传。
- 【精度回归门】任何**改变数值计算路径**的改动（近似替换、中间精度降级、指令或数据类型转换链变更、归约顺序变更等）落盘后，必须跑测试工程的权威精度断言并附上指标实测值与阈值；不达标即回退该改动，**禁止以「自建容差内」放行**。无法自行执行断言时，在回传结论中明确标注该改动待精度门校验，不得默认通过。
- 【文档同步清单】本轮改动使既有交付件失效时（白盒分支覆盖说明与行号、开发方案中已落地口径的描述、交付说明等），在回传结论中列出**受影响文档清单 + 具体失效点**，交由调用方派发刷新；你不改 test / doc 目录。
```

## 3.2 测试工程开发

- **角色**：developer-test

```md
- 【权限】你可写 test 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】测试方案文档 `.cannbot/<算子名>/2.1-测试方案设计.md`。
- 【输出】golden 代码 + 用例表 + 性能采集框架，写 test 目录。
- 【skills】立即加载 `repo-test-develop`、`ops-precision-standard`、`ops-profiling`。
- 【验收标准】golden 可产出期望输出，用例表按分级覆盖，性能采集框架可用。
- 【权威精度断言】按测试方案的**精度判定口径表**逐输出张量实现可执行硬断言：指标与阈值取口径表取值，不自行放宽、不换指标；每条用例产出「输出张量 / 判定指标 / 实测值 / 阈值 / 结论」并随结果落盘。自建的宽松容差只能作为交叉核对项，不得作为通过依据；两者结论不一致时以口径表为准并标注该分叉。口径表缺项或无法溯源时不自行取值，回退测试方案。
- 【性能数据口径】性能采集框架的输出须含**瓶颈维度分解**（计算 / 搬入 / 搬出 / 标量各自的占比或耗时），并支持对波动用例做多轮采集、记录中位数与离散度。
```

## 3.3 白盒测试补全

- **角色**：developer-test

```md
- 【权限】你可写 test 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码、测试代码。
- 【输出】白盒用例 + 分支覆盖说明，写 test 目录。
- 【skills】立即加载 `repo-test-develop`、`ascendc-whitebox-design`。
- 【验收标准】声明的执行分支 / tilingkey 覆盖达标（阈值由对应验收 skill 给定），产出分支覆盖说明。
```

## 3.4 联调

- **角色**：developer-code

```md
- 【权限】你可写算子代码目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码（3.1 产物）、测试代码（3.2/3.3 产物）。
- 【输出】联调结果（通过 / 不通过——代码问题回退 3.1、测试问题回退 3.2）；联调报告写入 `.cannbot/<算子名>/3.4-联调报告.md`，格式模板：`workflow-doc-templates/references/3.4-联调报告.md`。
- 【skills】立即加载 `repo-build-guide`、`repo-test-develop`、`repo-knowledge`、`ascendc-precision-debug`。
- 【验收标准】全量用例跑通、精度比对通过；代码侧问题已修复并编译验证通过；测试侧问题（golden/用例/框架缺陷）不自行修改 test 目录，在联调报告中定位并指明修改点。
- 【精度口径】「精度比对通过」以测试工程的**权威精度断言**为准，联调报告中逐输出张量记录判定指标实测值与阈值；自建的宽松容差只作交叉核对，两者结论不一致时按权威口径判不通过。
- 【文档同步清单】本轮修复使既有交付件失效时（白盒分支覆盖说明与行号、开发方案中已落地口径的描述、交付说明等），在联调报告中列出**受影响文档清单 + 具体失效点**，交由调用方派发刷新；你不改 test / doc 目录。
```

## CP3 功能验收

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/CP3-功能验收报告.md`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 测试代码。
- 【输出】功能验收结果（通过 / 不通过，按语义归属回退 3.1（算子实现）或 2.1（精度判定口径缺失、不可溯源、与权威源不符））；验收报告写入 `.cannbot/<算子名>/CP3-功能验收报告.md`，格式模板：`workflow-doc-templates/references/CP3-功能验收报告.md`。
- 【skills】立即加载 `workflow-cp3`、`workflow-doc-templates`、`ops-precision-standard`。
- 【精度独立复核】精度项须由你独立跑权威精度探针得出指标实测值（探针脚本落 `.cannbot/<算子名>/tmp/`），不采信开发者产物自述的"精度通过"结论。
```

# 阶段 4：性能验收

## 4.1 性能采集执行

- **角色**：developer-test

```md
- 【权限】你可写 test 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码（功能验收通过后的最终代码）。
- 【输出】性能数据（各 shape/dtype 的耗时、带宽、利用率），落 test 目录采集输出。
- 【skills】立即加载 `repo-test-develop`、`ops-profiling`。
- 【验收标准】性能数据完整覆盖需求关注的 shape/dtype。
- 【瓶颈维度分解】每条用例除总耗时外，须给出计算 / 搬入 / 搬出 / 标量各自的占比或耗时——这组数据用于锁定实测瓶颈维度，缺失则性能数据判为不完整。
- 【方差处理】跨轮波动跨越达标门槛的用例，做多轮稳态采集并记录中位数、min/max 与离散度；采集口径（预热剔除、设备空闲隔离、取值方式）随数据一并记录。以稳态中位数为准，不上报单次最优值。
```

## CP4 性能验收

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/CP4-性能验收报告.md`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 性能数据。
- 【输出】性能验收结果（通过 / 不通过，回退 3.1 / 不通过但实现级优化已穷尽，建议按已知限制收口——该结论不触发回退，转调用方决策）；验收报告写入 `.cannbot/<算子名>/CP4-性能验收报告.md`，格式模板：`workflow-doc-templates/references/CP4-性能验收报告.md`。
- 【skills】立即加载 `workflow-cp4`、`workflow-doc-templates`。
- 【杠杆评估门】存在未达标项时，先列未试杠杆清单（只收录与实测瓶颈维度匹配、且无已知负结果的方向）再定结论；清单为空或仅剩收益趋零的微优化时，出「建议收口」结论并附完整依据，不给回退意见。放宽需求声明的性能门槛超出你的裁定权限，只能建议、不得判为通过。
```

# 阶段 5：代码检视

## CP5 代码检视

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/CP5-代码检视报告.md`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】全部变更文件。
- 【输出】代码检视结果（不通过，回退 3.1 / 通过）；检视报告写入 `.cannbot/<算子名>/CP5-代码检视报告.md`，格式模板：`workflow-doc-templates/references/CP5-代码检视报告.md`。
- 【skills】立即加载 `workflow-cp5`、`workflow-doc-templates`。
```

# 阶段 6：上库准备

## 6.1 文档补全

- **角色**：developer-doc

```md
- 【权限】你可写 doc 目录（仅 md）；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 设计文档（`.cannbot/<算子名>/1.1-需求分析.md`、`.cannbot/<算子名>/2.2-开发方案设计.md`）。
- 【输出】算子文档，写 doc 目录；格式模板：`workflow-doc-templates/references/6.1-算子文档.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-knowledge`、`ascendc-docs-gen`。
- 【验收标准】接口、参数、约束、示例齐全。
```

# 阶段 7：开发总结

## 7.1 开发报告

- **角色**：developer-doc（起草）→ PM（落盘 `.cannbot/<算子名>/`）

```md
- 【权限】你可写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】全部交付物。
- 【输出】开发报告全文（格式模板：`workflow-doc-templates/references/7.1-开发报告.md`），回传调用方落盘，不自行写入。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】开发过程、交付物清单完整。
```

## 7.2 经验总结

- **角色**：developer-doc（起草）→ PM（落盘 `.cannbot/<算子名>/`）

```md
- 【权限】你可写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】开发过程记录。
- 【输出】经验总结文档全文（格式模板：`workflow-doc-templates/references/7.2-经验总结.md`），回传调用方落盘，不自行写入。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】沉淀有效经验与踩坑记录，含对工作流/领域知识的改进建议。
```

## 任务恢复 prompt

中断时按 `.cannbot/<算子名>/state.json` 的 `current_stage` 恢复到对应子 Agent（详见 [state-schema.md](state-schema.md)）。恢复时读取已产出交付件继续，不重跑已通过阶段。

| 中断阶段 | 恢复角色 | 恢复说明 |
|----------|----------|----------|
| 0 / CP0 | developer / QA | 读环境信息文档继续；CP0 问卷未收口（`pending_questionnaire`）则由 QA 重发问卷收集结论 |
| 1.1 | architect | 读需求文档继续 |
| CP1 | QA | 读需求文档继续；问卷未收口（`pending_questionnaire`）则由 QA 重发问卷收集结论 |
| CP2.1 | QA | 读测试方案文档继续 |
| 2.1 / 2.2 | architect | 读方案文档继续 |
| CP2.2 | QA | 按 `pending_questionnaire` 状态续跑：无该字段则重跑评审；`sent` 则由 QA 重发问卷收集结论；`answered` 则无异议进 3.1、有异议按语义归属回退 |
| 3.1 / 3.2 / 3.3 | developer-code / developer-test | 读代码与测试继续 |
| 3.4 | developer-code | 读联调报告继续；测试侧问题回退 3.2 后重走 3.3 → 3.4 |
| 4.1 | developer-test | 读性能数据继续 |
| CP3 / CP4 / CP5 | QA | 读对应报告继续；回退则从 3.1 重走 |
| 6.1 | developer-doc | 读算子文档继续 |
| 7.x | developer-doc | 读交付物继续 |
