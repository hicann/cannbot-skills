---
name: ops-direct-invoke-flash
description: 当需要从 CPU 函数、数学公式、代码片段或文本描述出发构建新的 Ascend C 或 Ascend950 Reg API 核函数时使用。覆盖从规格说明到经验证的 NPU 核函数的完整路径。
argument-hint: <源文件或描述>
disable-model-invocation: true
effort: high
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Agent, TaskCreate, TaskUpdate, TaskList, TaskGet, SendMessage
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "${CLAUDE_SKILL_DIR}/hooks/pre-build-check.sh"
          if: "Bash(*build*)"
          timeout: 10
          statusMessage: "Checking for host-only helpers in kernel code..."
  PostToolUse:
    - matcher: "Write|Edit"
      hooks:
        - type: command
          command: "${CLAUDE_SKILL_DIR}/hooks/post-edit-reminder.sh"
          if: "Edit(*/src/*.cpp)|Write(*/src/*.cpp)|Edit(*.asc)|Write(*.asc)"
          timeout: 5
          statusMessage: "Verifying kernel code conventions..."
---

# Ascend C 核函数从零构建工作流

你正在根据所提供的算子描述，为华为 NPU 构建一个高性能 Ascend C 核函数。对于 Ascend950 / `dav-3510`，默认生成原生 `AscendC::Reg` 计算代码。

**算子来源：** `$ARGUMENTS`

如果 `$ARGUMENTS` 为空，停下并请用户提供算子来源 —— 它可以是文件路径（C++、PyTorch、Numpy）、数学公式、规格说明文档或文字描述。

### 术语表

- **UB** —— Unified Buffer，Ascend NPU 上的片上 SRAM（不是 "undefined behavior"）

## 何时使用本 Skill

在以下情况使用本 Skill：

- 你手上有一个 CPU 函数、数学公式、代码片段或文字描述，需要产出生产级的 Ascend C 核函数
- 你需要用 `AscendC::Reg` 编写 Ascend950 / `dav-3510` 算子
- 你要从零构建并向算子工程添加一个新算子
- 你需要端到端的算子工作：规格说明、实现以及 NPU 验证

以下情况不要使用：
- 对现有核函数代码做快速修补
- 孤立的测试改动或纯粹的代码讲解
- 修改一个已经实现完成的算子

## 工作流概览

1. 分析算子来源，提取语义，并在算子工程根目录中选择一个已完成的算子作为结构参考。
2. 创建 `docs/{OP}/STATE.md`，然后在编写核函数代码之前先完成定义文档和设计文档。
3. 用子 Agent 评审文档，打磨完善，再以小步增量的方式实现。
4. 运行本地构建/测试，随后在真实 NPU 硬件上进行远端验证。
5. 在 `docs/{OP}/plans/troubleshooting.md` 中记录非平凡问题，并将预防措施反馈回工作流。

## 主路径

### 阶段 0：准备

> 注：本工作流中所有相对路径均相对于**算子工程根目录**（即包含 `build.sh` / `CMakeLists.txt` 的目录）。如果该目录不是当前工作目录，请先进入它。

1. 对输入进行分类，并据此读取/解析 `$ARGUMENTS`：
   - **文件路径**（磁盘上存在）：读取该文件。分类为 PyTorch/Numpy 片段（`torch.*` / `numpy.*`）、CPU 参考实现（其他 `.py`/`.cpp`/`.c`/`.cu`），或规格说明文档（`.md`/`.txt`/`.rst`）。
   - **内联文本**（不是文件路径）：视为数学公式（包含 LaTeX 或数学记法）或文字描述。
2. 提取或推导算子名 `{OP}`：
   - 如果 `$ARGUMENTS` 是文件，从文件名推导。
   - 如果 `$ARGUMENTS` 是公式或描述，向用户询问一个简短的算子名，或从语义中推导一个。
3. 确认 `src/{OP}.cpp` 尚不存在。
4. 创建 `docs/{OP}/plans/`。
5. 阅读算子工程根目录中一个已完成的算子以了解结构，并阅读 `docs/development_guide.md`。如果不存在已完成的算子，则仅依赖 [references/implementation-patterns.md](references/implementation-patterns.md)。
6. 如果目标是 Ascend950 / `dav-3510`，阅读 [references/reg-api-guide.md](references/reg-api-guide.md) 和 [references/reg-api-patterns.yaml](references/reg-api-patterns.yaml)。在设计前启动一个只读的 API 查询子 Agent，检视本地范例并确认允许/禁止的 Reg API 清单。

**退出条件：** 已确定 `{OP}` 名称，`docs/{OP}/plans/` 已存在，已确定参考算子。

### 阶段 1：状态跟踪

依据 [state-template.md](state-template.md) 创建 `docs/{OP}/STATE.md`。这是受 git 跟踪的持久化进度记录。STATE.md 是持久化进度的唯一可信来源。

使用 `TaskCreate` 为每个阶段（阶段 2 到阶段 8）创建会话内任务。开始每个阶段时使用 `TaskUpdate(status="in_progress")`，完成时使用 `TaskUpdate(status="completed")`。在提交节点更新 STATE.md 的勾选框。

**恢复之前的会话：** 如果 `docs/{OP}/STATE.md` 已存在，阅读它以确定哪些阶段已完成。仅为未完成的阶段重新创建 `TaskCreate` 条目。

提交规则：
- 在每个阶段明确的 `Commit` 检查点各提交一次。
- 例外：立即提交初始的 `STATE.md` 引导文件。

**退出条件：** STATE.md 已存在并已提交，已创建会话内任务。

### 阶段 2：桩构建

目标：得到一个可编译的桩，以便开始 TDD。

1. 创建 `include/{OP}.h`。
2. 创建带桩 `Run<T>()` 的 `src/{OP}.cpp`。
3. 更新 `CMakeLists.txt`。
4. 创建带占位测试的 `tests/{OP}_test.cpp`。
5. 在算子工程根目录运行 `bash build.sh && bash run.sh`。
6. 提交。

实现细节与代码模式：见 [references/implementation-patterns.md](references/implementation-patterns.md)。

**退出条件：** `build.sh && run.sh` 以 0 退出，占位测试出现在输出中，已提交。

### 阶段 3：定义文档

编写 `docs/{OP}/{OP}_definition.md`：

- 数学公式
- 输入/输出语义
- 数据类型策略
- 边界场景
- CPU 参考伪代码

在编写 CPU 参考伪代码之前，检查来源（如果提供了代码）是否存在迭代累加模式 —— 见 [references/common-failure-modes.md](references/common-failure-modes.md) § 迭代累加精度。如果输入是公式或描述，验证推导是否严谨，并与用户一起补全缺失的 I/O 细节。

使用 `run_in_background=true` 并行启动 `math-review` 和 `semantics-review` Agent。两者都会阅读定义文档与算子来源（如果存在来源文件）。准确的提示词、验证清单以及结构化判定格式见 [references/review-prompts.md](references/review-prompts.md)。

等待两个 Agent 都完成，然后在提交前吸收其结论。如果任何检查为 FAIL，处理该问题并重新运行相关评审。如果两次评审相互矛盾，阅读双方结论与来源，做出判断，并记录解决方案。

提交。

**退出条件：** 定义文档已存在，两份评审结论均已吸收（没有未解决的 FAIL），已提交。

### 阶段 4：设计文档

编写 `docs/{OP}/{OP}_design.md`：

- 计算策略
- 带 `liveBytesPerElem` 的 UB buffer 清单
- 切分参数
- 向量指令序列
- Cast 链（如果输出 dtype 与计算 dtype 不同）
- 对于 Ascend950 / `dav-3510`：Reg 包装器清单、掩码/尾块策略、`CastTrait` 细节、规约标量槽布局，以及禁止 API 的规避

使用 `run_in_background=true` 并行启动 `ub-review` 和 `instr-review` Agent。准确的提示词与验证清单见 [references/review-prompts.md](references/review-prompts.md)。

等待两个 Agent 都完成，然后吸收其结论。

对于 Ascend950 / `dav-3510`，还需启动 [references/review-prompts.md](references/review-prompts.md) 中的 `reg-api-review`。在仍存在未解决的 Reg API FAIL 结论时，不要提交设计。

在最终确定设计文档之前，对照 [references/common-failure-modes.md](references/common-failure-modes.md) 中的陷阱进行验证 —— 尤其是 UB 拷贝路径、32B DMA 最小值以及 Cast 支持矩阵。

对于 Ascend950 / `dav-3510`，还需对照 [references/reg-api-guide.md](references/reg-api-guide.md) 验证：在规划的向量路径中不得使用 `AscendC::MicroAPI`、不得使用 Membase、除 `asc_vf_call` 外不得使用裸 `asc_*` API，且不得使用经典 AscendC 的 compute/cast/reduce。

提交。

**退出条件：** 设计文档已存在，两份评审结论均已吸收，已提交。

### 阶段 5：CPU 参考实现与测试套件

构建 CPU 参考实现以及完整的 NPU 测试套件：

- 位于 `tests/{OP}_test.cpp` 的 CPU 参考实现
- `tests/{OP}_cases.h` 用例矩阵（最少：每个受支持 dtype 一个用例、一个小 shape、一个非 tile 对齐 shape、一个大 shape，以及按定义文档每个边界场景各一个边界值用例）
- host 侧辅助函数测试
- 参数化 NPU 测试

检查测试命名：如果 `{OP}` 与标准库数学符号冲突（例如 `erf`、`sinh`、`exp`），避免使用顶层 `namespace {OP}` —— 见 [references/common-failure-modes.md](references/common-failure-modes.md) § 测试命名冲突。

构建并运行：
- host 侧辅助函数测试应当通过
- 桩化的 NPU 测试应当失败

提交。

**退出条件：** 测试可编译，host 侧测试通过，针对桩的 NPU 测试失败，已提交。

### 阶段 6：核函数实现

在 `src/{OP}.cpp` 中增量实现：

1. `CalcTiling<T>()`
2. `{OP}_process_tile<T>()`
3. `{OP}_kernel<T>()`
4. `RunOnDevice<T>()`
5. 公开的 `Run<T>()`

对于 Ascend950 / `dav-3510`，按 [references/reg-api-guide.md](references/reg-api-guide.md) 中的 Reg 形态实现向量计算、cast 和规约：

- `__simd_vf__` 函数作用于 `__ubuf__` 指针和 `AscendC::Reg::RegTensor` 值。
- `__aicore__` 包装器将 `LocalTensor.GetPhyAddr()` 转换为 `__ubuf__` 指针并调用 `asc_vf_call`。
- 使用 `AscendC::Reg::UpdateMask` 或 `CreateMask` 生成全量/尾块掩码。
- 将 DMA、入队、UB 分配与同步保留在常规 AscendC 集成代码中。
- 在 Reg 路径中不要使用 `AscendC::MicroAPI`、Membase、除 `asc_vf_call` 外的裸 `asc_*` API，或经典 `AscendC::Mul/Cast/ReduceSum/Sigmoid/Sqrt/Duplicate/Adds/Muls` 风格的计算调用。

每完成一个有意义的步骤后：构建、运行测试、与定义文档和设计文档比对。如果测试失败，先修复再继续。

在最终确定之前，对照 [references/common-failure-modes.md](references/common-failure-modes.md) 验证 —— 尤其是 device 侧辅助函数调用、TBuf 生命周期 和 AscendC Cast 支持矩阵。代码模式见 [references/implementation-patterns.md](references/implementation-patterns.md)。

对于 Ascend950 / `dav-3510`，实现完成后重新运行只读的 API 查询子 Agent 或评审提示词，对照 [references/reg-api-patterns.yaml](references/reg-api-patterns.yaml) 扫描实际的核函数源码。在构建前解决每一项禁止 API 的发现。

在阶段 6 中使用多个 Agent 时，始终使用 `isolation="worktree"`。

提交。

**退出条件：** 所有本地测试通过，已提交。

### 阶段 7：远端验证

使用仓库由提交触发的远端验证流程：

1. 本地提交
2. 检视远端的构建/测试日志
3. 迭代直到远端验证通过

将 `{OP}` 添加到 `run.sh` 时，将其测试二进制放在靠前的位置 —— 排在已知存在偶发失败的算子之前。

在提交前，确认核函数源码包含完整实现（而非阶段 2 的桩）。陈旧输出的故障排查见 [references/common-failure-modes.md](references/common-failure-modes.md) § 远端构建新鲜度。

**退出条件：** 远端构建通过，远端测试通过。

### 阶段 8：收尾文档

1. 在 `docs/index.md` 中加入新算子。
2. 更新 `CLAUDE.md`。
3. 确认 STATE.md 所有勾选框均已勾选。
4. 将所有剩余任务标记为已完成。
5. 做最后一次文档提交。
6. 向用户报告：`{OP}` Ascend C 实现完成。

**退出条件：** 所有文档已更新，STATE.md 全部勾选，已提交。

## 阶段回退

如果在实现阶段 N 时发现某个阶段 M 产物（M < N）存在缺陷，更新该阶段 M 产物，若改动较为实质则重新运行其评审，并在继续阶段 N 之前单独提交该修复。

## Agent 用法

使用 `Agent` 工具进行评审和可并行的分析。模式：

**并行评审 Agent** —— 在单条消息中以 `run_in_background=true` 启动：
- 为每个 Agent 起描述性的名字（例如 `math-review`、`ub-review`）
- 保持提示词具体：`Read X and Y, verify A/B/C, write findings to Z.`
- 要求结构化判定（PASS/FAIL/CONCERN）—— 见 [references/review-prompts.md](references/review-prompts.md)
- 等待完成通知；在继续之前确认结论文件存在且包含判定

**隔离探查** —— 使用 `subagent_type="Explore"` 进行只读研究：
- 在 `src/` 范围内查找参考算子模式
- 搜索 AscendC 文档以获取 API 细节

**worktree 隔离** —— 当 Agent 编辑共享文件时使用 `isolation="worktree"`：
- 防止并行 Agent 互相覆盖各自的核函数代码
- 在多个算子同时移植时必不可少
- 写入不同文件的评审 Agent 不需要 worktree 隔离

**多算子批量移植** —— 为每个算子派生一个 worktree 隔离的 Agent：
- 见 [references/agent-team-patterns.md](references/agent-team-patterns.md)

可复用的提示词模板：[references/review-prompts.md](references/review-prompts.md)

## 故障排查日志

当你遇到非平凡问题时，向以下文件追加一条记录：

`docs/{OP}/plans/troubleshooting.md`

模板与反复出现的失败模式：
- [references/common-failure-modes.md](references/common-failure-modes.md)

如果某条 `Prevention` 项暗示了更好的工作流规则，在可行时把它作为文档改动的一部分，更新本 Skill 或其支撑文件。

## 支撑文件

- [state-template.md](state-template.md)：完整的 `STATE.md` 模板
- [references/implementation-patterns.md](references/implementation-patterns.md)：代码结构、构建、切分、核函数及 API 模式
- [references/reg-api-guide.md](references/reg-api-guide.md)：Ascend950 `AscendC::Reg` 策略、代码形态、模板及评审清单
- [references/reg-api-patterns.yaml](references/reg-api-patterns.yaml)：允许与禁止的 Reg API 扫描模式
- [references/review-prompts.md](references/review-prompts.md)：子 Agent 评审提示词模板
- [references/common-failure-modes.md](references/common-failure-modes.md)：故障排查模板及反复出现的陷阱
- [references/agent-team-patterns.md](references/agent-team-patterns.md)：多算子移植的 Agent Team 协同
- [hooks/pre-build-check.sh](hooks/pre-build-check.sh)：自动化的构建前校验（检测 host-only 辅助函数）
- [hooks/post-edit-reminder.sh](hooks/post-edit-reminder.sh)：编辑后的核函数规范检查
