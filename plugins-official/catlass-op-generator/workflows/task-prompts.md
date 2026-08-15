# Catlass · Subagent 调用参数详情

本文件是 CANNBot 调用各阶段 Subagent 的**唯一执行手册**。每个 Step 包含：调用参数、关联 Skill、完成验证、约束提醒。

---

## Step 1.5：需求预检

> 本步骤由 CANNBot 自行执行，不调用 Subagent。

### 1. 核数校验（必做）

1. 运行 `npu-smi info` 获取芯片型号
2. 加载 `/npu-arch` skill，在 `references/npu-hardware-params.md` §0 产品映射表中查芯片型号对应的 NpuArch 和产品系列
3. 在 `references/npu-hardware-params.md` §2 各架构参数中找到对应架构的 `cube_core_cnt` / `vector_core_cnt`

获取 `cube_core_cnt` 后：
- 用户核数 > 物理核数 → 告知用户超出硬件能力，等待修正
- 用户核数 == 物理核数 → 提示满载不一定最优（910B 系列实测 20 核优于 24 核）
- 用户核数 < 物理核数 → 通过
- 用户未指定 → 记录物理核数，建议"910B 系列推荐 20 核"

### 2. dtype 合理性校验（必做）

从用户需求提取算子名前缀 + 输入输出 dtype，校验：
- 算子含 `quant` 但输入为 fp16/bf16 → 警告（量化算子输入应为 int8/int4），需确认
- 算子不含 `quant` 但输入为 int8 → 警告（非量化不应用整数输入），需确认
- 输入 int8 但缺 scale 张量 → 提示
- 警告级需用户确认后通过

### 3. FlashAttention 预检（命中 FA/MHA/GQA/fused attention 时必做）

- 输入布局确认：Q/K/V 是否 BNSD `[B,H,S,D]`（若不满足，记录 host 需做的布局转换）
- `D % 16 == 0` 校验（cube 友好），`scale = 1/sqrt(D)` 约定确认
- 目标 SoC 确认：A2 必须走 `PAGED=true` + 恒等 block_table（`PAGED=false` 有 aicore exception），950PR 需单独评估
- baseline 确认：冻结 `aclnnFlashAttentionScore` 作为精度标杆

### 完成输出

写入 `operators/{operator_name}/docs/precheck.md`：激活函数验证结果 + 物理核数与推荐 + dtype 校验结果。

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
- 若需求命中 GDN / KDA / retention / RWKV / linear attention / state recurrence，额外读取 `/catlass-op-design` 的 `references/kernels/attention/linear-attention.md`；由该入口按需渐进读取 open-source reference、shape 覆盖规则、mixed tolerance 精度规则和 A2/A3 stage 经验。先执行参考来源选择：
  1. 只有用户需求中明确把某个本地路径标注为“本地参考实现 / 实现参考 / source-of-truth / 按此实现或 pipeline 对齐”时，才允许读取该本地路径作为 implementation primary reference，并在 OPEN_SOURCE_ALIGNMENT.md 记录 `reference_source=USER_LOCAL`；
  2. 用户没有显式给出本地实现参考路径时，必须按 `references/kernels/attention/linear-attention.md` 的 Reference Source Policy 自动启用远程开源 URL 作为 primary reference，并在 OPEN_SOURCE_ALIGNMENT.md 记录 `reference_source=OPEN_SOURCE`；若当前工作区没有该开源仓源码，应尝试 `git clone` 到工作区可复现目录（如 `tmp/open_source_refs/<repo>`）。如果网络、权限或远程仓不可用导致 clone 失败，不得阻塞 Step 2，改用仓内固化的开源规范摘要、远程 URL、搜索路径和 curated reference 继续设计，并记录 `clone_status=UNAVAILABLE`、失败原因和降级依据；
  3. 用户需求中以“性能 baseline / evaluation baseline / 精度或性能评测 / 对比指标 / 使用 X 评测”形式给出的本地路径，必须记录为 `evaluation_baseline`，只能用于精度/性能指标、shape、报告字段和 baseline_status，禁止作为 implementation reference、source-of-truth 或 pipeline 骨架；
  4. 仓内 GDN/KDA 用例矩阵、mixed tolerance 精度规则、既有 Catlass 经验只能作为 curated reference / 工程经验，不得在未获用户显式实现参考路径时替代开源源码成为 primary reference。
  完成参考来源选择后，再冻结 full-flow vs stage、baseline、dependency graph、workspace/flag 与 shape 覆盖矩阵。
- Linear Attention / GDN / KDA 场景不得因公开 Catlass epilogue 缺少 gate/decay/mask/finalize/scan/state 组件就停在 Architect；必须把 GEMM 节点映射到 Catlass，把非 GEMM 节点设计为可审查的 Catlass-style 自定义 Block/Tile 或 dependency-based stage，并继续输出 DESIGN.md + PLAN.md。用户 prompt 中的数学 contract 优先于任何参考实现。
- Linear Attention / GDN / KDA 场景必须先产出 `operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md`，再写 DESIGN.md。该文件是硬门禁，必须包含：用户数学 contract 逐项冻结、reference_source（`OPEN_SOURCE` 或 `USER_LOCAL`）、clone_status（`CLONED` / `UNAVAILABLE` / `NOT_NEEDED`）、远程 URL + commit/tag/摘要版本或用户本地实现参考路径 + 版本状态、evaluation_baseline（如用户给出 baseline 路径则必填，并标注“仅评测，不作为实现参考”）、仓内 curated reference 列表、参考实现文件级映射表、公式到代码行/函数的对照（clone 不可用时可写“规范摘要章节/远程搜索路径”）、与用户 contract 的语义差异表（scale 位置、mask/clamp、dtype/cast、layout、varlen/partial、workspace/flag、tiling key、baseline 限制）、采用/偏离参考实现的裁决理由。缺失该文件、未按用户输入正确选择 reference_source、把 baseline 误作 implementation reference 或语义差异未列全，Step 2 不通过；仅 clone 失败且已记录降级依据不得作为 Step 2 阻塞项。

【输出】
- 技术设计：operators/{operator_name}/docs/DESIGN.md，参考 `workflows/templates/design-template.md`
- 开发计划：operators/{operator_name}/docs/PLAN.md，参考 `workflows/templates/plan-template.md`
- Linear Attention / GDN / KDA 对齐记录（仅命中 Linear Attention / GDN / KDA / retention / RWKV / state recurrence 时输出）：operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md，参考 `workflows/templates/attention/open-source-alignment-template.md`

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
  - §2.2 Workspace 量级来源（catlass 直调路径指针透传 `GM_ADDR userWs = workspace;`）
  - §2.3 实现约束（C3/C4/C6 等 catlass 禁项）
- PLAN.md 包含：文件清单、catlass 编译选项（`-I./catlass/include` + `-DCATLASS_ARCH=<arch>`）、catlass kernel 运行期 shape 约束（避免过小 M/N，选 L1 分块整数倍）
- Linear Attention / GDN / KDA / retention / RWKV / state recurrence 场景的 PLAN.md 还必须包含 BT/chunk、V/K、HK/HV/GQA、batch/head、TilingKey 与数值边界覆盖矩阵，说明每个 shape 的来源和覆盖类别
- Linear Attention / GDN / KDA / retention / RWKV / state recurrence 场景的 DESIGN.md 还必须包含：用户数学 contract 冻结记录、OPEN_SOURCE_ALIGNMENT.md 摘要与链接、参考实现语义差异记录、非 GEMM 节点的自定义 Block/Tile 契约或 stage 化说明、baseline_status 记录规则；非 Linear Attention 类不要求这些章节
- Linear Attention / GDN / KDA / retention / RWKV / state recurrence 场景必须按 reference_source 选择 pipeline：`USER_LOCAL` 时优先按用户给出的本地实现参考设计；`OPEN_SOURCE` 时优先按开源参考设计，clone 可用则对齐源码，clone 不可用则对齐仓内开源规范摘要与远程搜索路径。未获用户显式本地实现参考路径时，禁止自动扫描或选用开发机本地实现作为 source-of-truth；用户 baseline/评测路径禁止作为实现参考；仓内旧算子/历史实现只能作为 curated engineering reference，偏离 primary reference 必须写入 OPEN_SOURCE_ALIGNMENT.md 裁决表

【约束】
- 禁止：写实现代码（设计阶段只产文档）
- 禁止：使用 catlass `DeviceGemm` 适配器（仅 example 用）
- 禁止：设计在 op_kernel 中自实现矩阵乘 / 逐元素 / 拷贝循环
- 禁止：用户未显式指定本地实现参考路径时，把开发机本地仓库、当前工作区之外的同名实现、历史算子目录或用户 baseline/评测路径当作 primary reference
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
- Linear Attention / GDN / KDA 对齐记录（如适用）：operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md

【重点审查章节】DESIGN.md §1.1 数学公式 / §1.2 catlass 选型表 / §1.3 参考 example / §1.5 BlockEpilogue 槽位清单 / §1.6 自定义 Tile 契约 / §2.1 TilingKey 分支条件 / §2.2 Workspace。§0、§2.3 通览即可。Linear Attention / GDN / KDA 场景额外审查 OPEN_SOURCE_ALIGNMENT.md 与 DESIGN.md 中对应摘要。

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
| 5 | Workspace 计算 | §2.2 Workspace 量级是否清晰？kernel 内是否按起动经路正确取 workspace（catlass hand-launch 直调用指针透传 `GM_ADDR userWs = workspace;`，**禁** `GetUserWorkspace`）？ |
| 6 | 精度策略 | catlass GEMM 精度阈值是否对齐 `ops-precision-standard`？无 catlass 专属放宽规则 |
| 7 | Open-source 语义对齐 | Linear Attention / GDN / KDA 场景是否存在 OPEN_SOURCE_ALIGNMENT.md；用户 contract、参考实现代码、设计方案三者是否逐项对齐；所有 scale/mask/clamp/cast/layout/varlen 差异是否有裁决？ |

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
- **文档依据**：（catlass header / example 路径，或 asc-devkit/docs/api/ 路径）
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
- 若存在：operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md — Linear Attention / GDN / KDA 的开源/仓内参考实现对齐记录
然后开始开发。

【必读 Skill】
- /catlass-op-develop — 加载并按 skill 内「核心工作流」执行 op_kernel 内 catlass 模板拼装与 Device 调用（强制）
- 若 DESIGN 命中 GDN / KDA / retention / RWKV / linear attention / state recurrence，额外读取 `/catlass-op-develop` 的 `references/patterns/linear-attention.md` 与 `/catlass-op-design` 的 `references/kernels/attention/linear-attention.md`，按 GM workspace/flag、远程开源参考规范、仓内 GDN/KDA case reference、L1/L0/UB 复用、shape/precision/perf 规则实现
- Linear Attention / GDN / KDA 场景如果公共 Catlass 组件不能直接表达非 GEMM 逻辑，不得退回 host 计算或空 device kernel；按 DESIGN.md §1.6 实现 Catlass-style 自定义 Block/Tile。自定义组件内部可使用 Ascend C Vector API 和固定 tile 内循环表达逐元素逻辑，但 GEMM 仍必须使用 Catlass。
- Linear Attention / GDN / KDA 场景进入编码前必须完成“source-of-truth check”：读取 OPEN_SOURCE_ALIGNMENT.md 中按 `reference_source` 选定的 primary reference 记录，并把代码实现路线写入 PLAN.md。`reference_source=USER_LOCAL` 时以用户本地实现参考的 kernel pipeline、block scheduler、epilogue/finalize、workspace/flag 和 tiling key 为骨架；`reference_source=OPEN_SOURCE` 且 `clone_status=CLONED` 时以 clone 的开源实现为骨架；`clone_status=UNAVAILABLE` 时以仓内开源规范摘要、远程搜索路径和 curated reference 形成可追溯骨架。禁止在用户未显式给本地实现参考路径时改用开发机本地实现、历史算子目录或用户 baseline/评测路径作为骨架。任何偏离必须先记录到 PLAN.md 和 OPEN_SOURCE_ALIGNMENT.md，并说明用户数学 contract 或 catlass 直调约束依据。

【渐进式开发策略（每步必须编译通过后再进入下一步）】
Step A：基于 §1.3 选定的参考 example 起工程骨架 → 编译通过（空 Kernel）
Step B：写 op_host Tiling 计算 + ACL 框架 → 编译通过
Step C：写 op_kernel catlass 拼装类 + kernel 入口分支 + Device 调用 → 编译通过
Step D：（如有）落盘自定义 Tile 头文件 → 编译通过
Step E：补 gen_data.py / verify_result.py / run.sh，跑通 Level 0~2 测试
  **verify_result.py 强制要求**：必须先读取模板文件 `cannbot-skills/ops/catlass-op-develop/references/verify_result_template.py`，
  完整复制其精度判定逻辑（双标准：MERE/MARE + atol/rtol/error_ratio，通过任一即 PASS），
  只允许修改 OUTPUT_SHAPE 和 OUTPUT_DTYPE。禁止自行编写精度判定函数或自选阈值。

【catlass 实现强制项】
- 直接实例化 `Kernel` + `Kernel::Params`，`Kernel{}(params)`；**禁用** `DeviceGemm` 适配器
- op_kernel 内**禁止**自实现矩阵乘 / 逐元素 / 拷贝循环（只能用 catlass `Kernel`/`Block*`/`Tile*`）
- Linear Attention / GDN / KDA 的例外边界：逐元素/gate/decay/mask/finalize 逻辑可封装在自定义 Catlass-style `Block*`/`Tile*` 组件内部；禁止的是 op_kernel 顶层散乱循环、host 真实计算和自实现 GEMM。
- Workspace 按起动经路取：catlass **hand-launch 直调**（本工作流主流）用指针透传 `GM_ADDR userWs = workspace;`（host `<<<>>>` 传入的 devWorkspace 直接用）；**禁用** `AscendC::GetUserWorkspace`（arch 2201 直调路径下丢入参返回 kfc 地址致 MTE DDR 越界）；**禁用** `SetSysWorkspaceForce`
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
- Linear Attention / GDN / KDA 场景：PLAN.md 记录了 OPEN_SOURCE_ALIGNMENT.md 的 source-of-truth check；代码中的关键 pipeline/epilogue/scheduler/workspace/tiling 选择可追溯到参考实现或显式裁决
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
- Linear Attention / GDN / KDA 对齐记录（如适用）：operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
- 环境信息：operators/{operator_name}/docs/environment.json
- 通用审查清单：workflows/references/review-checklist.md
- catlass 源码（对照）：./catlass/include、./catlass/examples

【输出】
- 审查报告：operators/{operator_name}/docs/REVIEW.md

【推荐 Skill】
- /ascendc-docs-search — 验证非 catlass API 约束
- /ops-profiling — 独立采集 msprof 性能数据
- /ops-precision-standard — 精度阈值确认
- 若需求命中 GDN / KDA / retention / RWKV / linear attention / state recurrence，对照 `/catlass-op-develop` 的 `references/patterns/linear-attention.md`、`/catlass-op-design` 的 `references/kernels/attention/linear-attention.md` 与 `workflows/references/attention-linear-review.md`，检查远程开源 URL、仓内 GDN/KDA 全量 shape、mixed tolerance 精度报告、varlen/partial 和性能归档
- Linear Attention / GDN / KDA 场景额外检查：用户数学 contract 是否冻结到 golden/verify/README/report；非 GEMM 逻辑是否封装为 device 自定义 Block/Tile 或明确 stage 化；host 是否只做 tiling/runner/验证；baseline unsupported/MISSING 是否被记录而不是中断全流程
- Linear Attention / GDN / KDA 场景额外阻塞项：缺失 OPEN_SOURCE_ALIGNMENT.md、`reference_source` 缺失或与用户输入不符、未给用户本地实现参考路径却使用开发机本地实现/历史算子目录/用户 baseline 评测路径作为 primary reference、参考实现路径或规范摘要不可追溯、scale/mask/clamp/cast/layout/varlen/workspace/tiling 差异未裁决、代码 pipeline 明显偏离 primary reference 且无设计裁决，任一项均判 FAIL。开源仓 clone 失败本身不是 FAIL，前提是记录 `clone_status=UNAVAILABLE`、失败原因和摘要降级依据。

【catlass 专项检视项 C1–C11】（必须逐条覆盖并在 REVIEW.md 中列表呈现）
- C1 命名含 `catlass`，snake_case ↔ CamelCase 一致映射
- C2 catlass 源码位于 `./catlass/`，未克隆到 `operators/{operator_name}/` 内
- C3 CMakeLists.txt 注入 `-I<catlass>/include` + `-DCATLASS_ARCH=<arch>`
- C4 op_kernel 直接 `Kernel` + `Kernel::Params` + `Kernel{}(params)`；禁用 `DeviceGemm` 适配器
- C5 op_kernel 不自实现矩阵乘 / 顶层散乱逐元素 / 顶层拷贝循环；Linear Attention 自定义 Block/Tile 内部的固定 tile 逐元素 Vector 逻辑允许，但必须组件化、device 执行、不得手写 GEMM
- C6 Workspace 按起动经路取：catlass hand-launch 直调用指针透传 `GM_ADDR userWs = workspace;`，**禁用** `GetUserWorkspace`（直调路径丢入参返回 kfc 地址）与 `SetSysWorkspaceForce`
- C7 op_kernel 不 include 算子自身的 tiling 实现文件（仅可 include 共享 POD `*_tiling.h`）
- C8 TilingKey 分支与 DESIGN.md §2.1 合法组合一致
- C9 测试 shape 满足 catlass 运行期约束（避免过小 M/N，选 L1 分块整数倍）
- C10 catlass 拼装类 `using` 与 DESIGN.md §1.2 选型表一致
- C11 调优场景已加载 `/catlass-op-perf-tune`，PRE/POST 报告已归档

【Linear Attention / GDN / KDA 专项门禁 LA-OS】（仅命中 Linear Attention / GDN / KDA / retention / RWKV / state recurrence 时启用；其他算子跳过）
- LA-OS open-source 对齐硬门禁：OPEN_SOURCE_ALIGNMENT.md 存在且完整；`reference_source` 为 `OPEN_SOURCE` 或 `USER_LOCAL` 且选择依据正确；`OPEN_SOURCE` 下 clone 成功或降级摘要均有记录；代码关键 pipeline/epilogue/scheduler/workspace/tiling 与 primary reference 或裁决表一致；用户 contract 的 scale/mask/clamp/cast/layout 等高风险点已落到 golden/verify 和 device 实现；REVIEW.md 必须按 `workflows/references/attention-linear-review.md` 逐项列出 LA1-LA18 状态和证据

【验收标准】
- 独立编译验证（含 catlass 编译选项校验：`workflows/scripts/verify_cmake_config.py`）
- C1–C11 逐条覆盖
- Linear Attention / GDN / KDA 场景 LA-OS 逐条覆盖；LA-OS 不通过不得 PASS WITH NOTES
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
- 审查报告：operators/{operator_name}/docs/REVIEW.md（重点 catlass C1–C11 表、Linear Attention 类 LA-OS 状态 + 必须修复项）
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
- 若需求命中 GDN / KDA / retention / RWKV / linear attention / state recurrence，必须按 `/catlass-op-perf-tune` 的 Linear Attention / KDA dAv 排障规则，优先复用已有 inputs/baseline，并区分 baseline unsupported 与 custom 性能失败

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

**Linear Attention / GDN / KDA 专项 LA-OS 状态**（非 Linear Attention 类填写“不适用”）:
| # | 检视项 | 状态 |
|---|--------|------|
| LA1 | full-flow / stage operator 与同语义 baseline | 通过/失败/不适用 |
| LA2 | dependency graph 与 stage 切分 | 通过/失败/不适用 |
| LA3 | 中间量 producer/consumer/pipe/memory/lifetime | 通过/失败/不适用 |
| LA4 | GM workspace slot 与 flag 协议 | 通过/失败/不适用 |
| LA5 | L1 resident / scratch 分离 | 通过/失败/不适用 |
| LA6 | L0/L0C/fixpipe 与 split accumulation | 通过/失败/不适用 |
| LA7 | UB buffer 分组 | 通过/失败/不适用 |
| LA8 | GQA/GVA HK cache scope | 通过/失败/不适用 |
| LA9 | shape-constraints Δ5 覆盖矩阵 | 通过/失败/不适用 |
| LA10 | 精度报告口径 | 通过/失败/不适用 |
| LA11 | 性能报告字段 | 通过/失败/不适用 |
| LA12 | A2/A3 stage checklist | 通过/失败/不适用 |
| LA13 | open-source reference 记录 | 通过/失败/不适用 |
| LA14 | KDA dAv varlen/partial | 通过/失败/不适用 |
| LA15 | mixed tolerance 近零兜底与 matched_ratio/max_abs | 通过/失败/不适用 |
| LA16 | 用户数学 contract 落地 | 通过/失败/不适用 |
| LA17 | 非 GEMM 组件化 / host 不计算 / device 非空 | 通过/失败/不适用 |
| LA18 | reference_source / evaluation_baseline 合法性 | 通过/失败/不适用 |
| LA-OS | LA1-LA18 汇总判定 | 通过/失败/不适用 |

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
