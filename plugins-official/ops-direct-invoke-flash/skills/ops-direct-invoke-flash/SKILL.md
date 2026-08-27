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
          # 仅在真正触发编译的 make/cmake 命令上运行；避免 `mkdir build`、`rm -rf build` 等误触发。
          if: "Bash(*make*)"
          timeout: 10
          statusMessage: "Checking for host-only helpers in kernel code..."
  PostToolUse:
    - matcher: "Write|Edit"
      hooks:
        - type: command
          command: "${CLAUDE_SKILL_DIR}/hooks/post-edit-reminder.sh"
          if: "Edit(*.asc)|Write(*.asc)"
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
3. 按 `harness.review` 决定是否用子 Agent 评审文档，打磨完善，再以小步增量的方式实现。
4. 运行本地构建/测试，随后在真实 NPU 硬件上进行验证；若 `harness.test_gate` 为 `on`，再执行黑/白盒测试门禁。
5. 在 `docs/{OP}/plans/troubleshooting.md` 中记录非平凡问题，并将预防措施反馈回工作流。

## 算子复杂度分档（贯穿全流程）

在阶段 0 判定算子复杂度档位，记入 STATE.md，用于缩放阶段 3/4 的评审强度与阶段 5 的用例规模。评审强度应随复杂度缩放——对简单算子启动全套多 Agent 评审只会增加时延与 token 而抓不到问题。

| 档位 | 判据 | 阶段 3 评审 | 阶段 4 评审 |
|---|---|---|---|
| **simple** | 纯逐元素、无 Cast 链、无规约、无 UB-to-UB 拷贝（如 add、mul、逐元素多输入融合） | 1 个合并 Agent（数学+语义 checklist） | 1 个合并 Agent（UB+指令+Reg checklist） |
| **complex** | 含规约 / Cast 链 / 迭代累加 / UB-to-UB 拷贝 / 多 pass / Matmul 融合 | 2 个 Agent（math + semantics） | 2~3 个 Agent（ub + instr，dav-3510 加 reg-api） |

判据存疑时按 **complex** 处理。合并评审的提示词模板见 [references/review-prompts.md](references/review-prompts.md)。

## 文档评审开关 `harness.review`

从 Team `AGENTS.md` frontmatter 读取 `harness.review`，取值 `auto` / `on` / `off`，**默认 `auto`**。它与复杂度档位共同决定阶段 3/4 的文档评审、以及阶段 6 的 Reg API 扫描是否起子 Agent：

| 取值 | simple 档 | complex 档 |
|---|---|---|
| `auto`（默认） | **跳过**：阶段 3/4 不起评审 Agent；阶段 6 的禁用 API 扫描由主 Agent 自行 grep | **评审**：阶段 3/4 分项多 Agent；阶段 6 重跑 API 扫描 Agent |
| `on` | **评审**：阶段 3/4 各 1 个合并 Agent；阶段 6 重跑 API 扫描 Agent | 同 `auto` 的 complex 列 |
| `off` | **跳过**：同 `auto` 的 simple 列 | **跳过**：同左 |

跳过时，在 `docs/{OP}/STATE.md` 记录「按配置跳过文档评审（`harness.review=<取值>`，复杂度档位 `<档位>`）」并继续，**不要**生成空的 `review_*.md` 占位文件。

**跳过的是子 Agent 评审，不是校验本身。** 以下检查由主 Agent 自己完成，在任何取值下都必须执行：

- 阶段 4：每个块大小 / 传输计数常量对所有支持 dtype 满足 `count * sizeof(T) >= 32`；UB 预算涵盖所有存活缓冲区且不超限；Cast 链符合支持矩阵。
- 阶段 6 构建前：grep host-only 辅助函数（`ceil_div` / `align_down` / `align_up`）。
- 阶段 6 构建前（dav-3510）：对照 [references/reg-api-patterns.yaml](references/reg-api-patterns.yaml) grep 禁用 API（`AscendC::MicroAPI`、Membase、除 `asc_vf_call` 外的裸 `asc_*`、经典 AscendC 计算/Cast/规约），并确认掩码为元素计数、尾块存储已加掩码。
- 阶段 6/7：构建通过且 `pytest` 全部用例在真机通过——这是最终的正确性关卡，任何取值下都不可跳过。

## 并行起草（缩短评审空等）

本节仅在按 `harness.review` 实际发起了评审时适用；跳过评审时没有 barrier，直接顺序推进即可。

评审 Agent 是 barrier，起草下一份**独立产物**不是。在等待某阶段后台评审时，可并行起草下一阶段中不依赖该评审结论的产物，评审返回后再对照吸收：

- 等阶段 3 定义评审时 → 起草阶段 4 设计文档草稿。
- 等阶段 4 设计评审时 → 起草阶段 5 测试套件。

若评审给出 FAIL 导致上游产物实质变更，则相应回改已起草的下游草稿。

## simple 档快速路径（编译一次 / 真机测一次）

**仅适用于 simple 档，且 CMake/结构照抄自已验证的参考算子（如 `mul`）。** complex 档、或对 `.asc` 结构/构建配置有任何自定义时，不走本快速路径，回退到默认三段式构建。

ASC 是单文件全量编译，改 `.asc` 即触发整个翻译单元重编，「增量」省不了多少。默认流程会编译 **3 次**（阶段 2 骨架、阶段 6 实现、阶段 7 干净重建）、真机全量测试 **2 次**（阶段 6、阶段 7）。本快速路径把它压到**编译 1 次、真机全量测试 1 次**，做法是推迟构建/测试并取消阶段 7 的二次重建：

- **阶段 2（骨架）**：创建 `{OP}.asc` 骨架 + `CMakeLists.txt` + 接口测试文件后，只跑 `cmake` **配置**（`mkdir -p build && cd build && cmake -DCMAKE_ASC_ARCHITECTURES=${NPU_ARCH} ..`）验证工具链/CMake，**不执行 `make`**。骨架的 ASC 编译推迟到阶段 6。
- **阶段 5（测试套件）**：写完整套件后**只做静态检查**（`python3 -m py_compile test_{OP}.py`，目视核对 `SHAPES`/`DTYPES` 与边界用例），**不 build、不 collect、不跑用例**——collection 需加载 `.so`，而此时尚未编译。
- **阶段 6（实现）**：实现完整核函数后，复用阶段 2 的配置做**首次也是唯一一次**构建（`cd build && make -j`；这是全流程唯一的 ASC 编译），随后**一次性**跑接口测试 + `--collect-only` + 完整 NPU 参数化用例。失败则修复后仅重编该文件。
- **阶段 7（验证）**：simple 档**跳过独立的干净重建**——阶段 6 的 `make` 是在阶段 2 全新配置的 build/ 上的首次编译，本就干净。仅确认 `{OP}.asc` 为完整实现（非骨架）、阶段 6 日志显示全部用例通过。**唯有**怀疑陈旧产物时才触发一次 `rm -rf build` 干净重建。

并行化：`harness.review=on` 时，阶段 3/4 的后台评审可与实现草稿重叠——simple 档在评审进行时并行起草阶段 6 实现，评审 FAIL 再回改。按开关跳过评审时无此空等，直接顺序推进。

风险：本路径牺牲了「骨架早编暴露编译错误」与「阶段 7 独立新鲜度重建」两道安全网，换取 ~2/3 的编译墙钟。仅在低风险（照抄参考算子）时使用。

## 主路径

### 阶段 0：准备

> 注：本工作流中所有相对路径均相对于**算子工程根目录**（即包含 `CMakeLists.txt` 的目录，例如 `operators/{OP}/`）。如果该目录不是当前工作目录，请先进入它。

1. 对输入进行分类，并据此读取/解析 `$ARGUMENTS`：
   - **文件路径**（磁盘上存在）：读取该文件。分类为 PyTorch/Numpy 片段（`torch.*` / `numpy.*`）、CPU 参考实现（其他 `.py`/`.cpp`/`.c`/`.cu`），或规格说明文档（`.md`/`.txt`/`.rst`）。
   - **内联文本**（不是文件路径）：视为数学公式（包含 LaTeX 或数学记法）或文字描述。
2. 提取或推导算子名 `{OP}`：
   - 如果 `$ARGUMENTS` 是文件，从文件名推导。
   - 如果 `$ARGUMENTS` 是公式或描述，向用户询问一个简短的算子名，或从语义中推导一个。
3. 确认 `{OP}.asc` 尚不存在。
4. 创建 `docs/{OP}/plans/`。
5. 阅读算子工程根目录中一个已完成的算子以了解结构。如果不存在已完成的算子，则仅依赖 [references/implementation-patterns.md](references/implementation-patterns.md)。
6. 探测目标芯片：运行 `python3 ${CLAUDE_SKILL_DIR}/scripts/detect_soc.py`，从输出中读取 `SocVersion` 与 `NpuArch`（形如 `dav-3510`）。据此判定后续路径——`NpuArch` 为 `dav-3510`（Ascend950）时走 `AscendC::Reg` 路径。若脚本因无 NPU 或环境缺失而失败，向用户询问目标芯片。这两个值会在阶段 1 记入 STATE.md。
7. 如果目标是 Ascend950 / `dav-3510`，阅读 [references/reg-api-guide.md](references/reg-api-guide.md) 和 [references/reg-api-patterns.yaml](references/reg-api-patterns.yaml)，并把 `operators/mul`（dav-3510 Reg 路径的最小参考算子）作为 Reg 计算形态的结构参考。`add`/`sqrt` 用的是经典 AscendC 计算 API，仅供 harness/CMake/test 结构参考。在设计前启动一个只读的 API 查询子 Agent，检视本地范例并确认允许/禁止的 Reg API 清单。
8. 判定算子复杂度档位（simple / complex，见上文「算子复杂度分档」），据此缩放后续评审强度。这将在阶段 1 记入 STATE.md。
9. 从 Team `AGENTS.md` frontmatter 读取 `harness.review`（`auto` / `on` / `off`，默认 `auto`），与步骤 8 的档位一起查「文档评审开关」表，确定阶段 3/4/6 是否发起评审 Agent。该取值也在阶段 1 记入 STATE.md。

**退出条件：** 已确定 `{OP}` 名称，`docs/{OP}/plans/` 已存在，已确定参考算子，已探测到目标芯片（`SocVersion` / `NpuArch`），已判定复杂度档位，已读取 `harness.review`。

### 阶段 1：状态跟踪

依据 [state-template.md](state-template.md) 创建 `docs/{OP}/STATE.md`。这是本地持久化进度记录。STATE.md 是持久化进度的唯一可信来源。

使用 `TaskCreate` 为每个阶段（阶段 2 到阶段 8，`harness.test_gate` 为 `on` 时包含阶段 7.5）创建会话内任务。
开始每个阶段时使用 `TaskUpdate(status="in_progress")`，完成时使用 `TaskUpdate(status="completed")`。完成每个阶段时更新 STATE.md 的勾选框。

**恢复之前的会话：** 如果 `docs/{OP}/STATE.md` 已存在，阅读它以确定哪些阶段已完成。仅为未完成的阶段重新创建 `TaskCreate` 条目。

**退出条件：** STATE.md 已存在，已创建会话内任务。

### 阶段 2：可编译骨架

目标：得到一个可编译的骨架，以便开始 TDD。

> **simple 档：** 改走「simple 档快速路径」——阶段 2 只做 `cmake` 配置、不 `make`，把唯一一次 ASC 编译推迟到阶段 6。以下步骤 4 的 `make` 属默认（complex）路径。

1. 创建单文件 `{OP}.asc`：device kernel + tiling + host 入口 + `TORCH_LIBRARY` 注册的骨架（kernel 可留空，host 入口直接返回空输出张量）。
2. 创建 `CMakeLists.txt`（参照已完成算子：`find_package(ASC)` + torch_npu，产出 `libop_{OP}.so`）。
3. 创建带占位用例的 `test_{OP}.py`（先放一个接口存在性测试）。
4. 在算子工程根目录构建并运行：`mkdir -p build && cd build && cmake -DCMAKE_ASC_ARCHITECTURES=${NPU_ARCH} .. && make -j`，然后回到算子目录运行 `pytest test_{OP}.py -v`。这是**唯一**需要跑 `cmake` 配置的地方；后续阶段（3~6）改源码后只需 `cd build && make -j` 增量编译，不要 `rm -rf build`。干净重建仅留到阶段 7 做新鲜度验证（或结果可疑时）。

实现细节与代码模式：见 [references/implementation-patterns.md](references/implementation-patterns.md)。

**退出条件：** cmake/make 构建成功，`pytest` 可运行且占位测试出现在输出中。

### 阶段 3：定义文档

编写 `docs/{OP}/{OP}_definition.md`：

- 数学公式
- 输入/输出语义
- 数据类型策略
- 边界场景
- CPU 参考伪代码

在编写 CPU 参考伪代码之前，检查来源（如果提供了代码）是否存在迭代累加模式 —— 见 [references/common-failure-modes.md](references/common-failure-modes.md) § 迭代累加精度。如果输入是公式或描述，验证推导是否严谨，并与用户一起补全缺失的 I/O 细节。

按 `harness.review` 与复杂度档位决定本阶段是否评审（见「文档评审开关」，默认 `auto`）：
- **跳过**（`off`，或 `auto` + simple 档）：不发起评审 Agent，在 STATE.md 记录按配置跳过，写完定义文档并进入阶段 4。
- **`on` + simple 档**：`run_in_background=true` 启动 **1 个** `definition-review` 合并 Agent（数学+语义 checklist 合并）。
- **`auto` / `on` + complex 档**：`run_in_background=true` 并行启动 `math-review` 和 `semantics-review` 两个 Agent。

Agent 都会阅读定义文档与算子来源（如果存在来源文件）。准确的提示词（含合并版）、验证清单以及结构化判定格式见 [references/review-prompts.md](references/review-prompts.md)。

发起了评审时：在等待评审时，可并行起草阶段 4 设计文档草稿（见「并行起草」）。等待所有 Agent 完成，然后吸收其结论。如果任何检查为 FAIL，处理该问题并重新运行相关评审。如果两次评审相互矛盾，阅读双方结论与来源，做出判断，并记录解决方案。

**退出条件：** 定义文档已存在；若发起了评审，其结论均已吸收（没有未解决的 FAIL）；若跳过，STATE.md 已记录跳过原因。

### 阶段 4：设计文档

编写 `docs/{OP}/{OP}_design.md`：

- 计算策略
- 带 `liveBytesPerElem` 的 UB buffer 清单
- 切分参数
- 向量指令序列
- Cast 链（如果输出 dtype 与计算 dtype 不同）
- 对于 Ascend950 / `dav-3510`：Reg 包装器清单、掩码/尾块策略、`CastTrait` 细节、规约标量槽布局，以及禁止 API 的规避

按 `harness.review` 与复杂度档位决定本阶段是否评审（见「文档评审开关」，默认 `auto`）：
- **跳过**（`off`，或 `auto` + simple 档）：不发起评审 Agent，在 STATE.md 记录按配置跳过。**下方「设计定稿前的确定性校验」仍须执行。**
- **`on` + simple 档**：`run_in_background=true` 启动 **1 个** `design-review` 合并 Agent（UB 预算 + 指令序列 checklist 合并；dav-3510 时该 checklist 额外含 Reg API 合规项）。
- **`auto` / `on` + complex 档**：`run_in_background=true` 并行启动 `ub-review` 和 `instr-review`；对 Ascend950 / `dav-3510` 再加 `reg-api-review`。

准确的提示词（含合并版）与验证清单见 [references/review-prompts.md](references/review-prompts.md)。发起了评审时：在等待评审时，可并行起草阶段 5 测试套件（见「并行起草」）。等待所有 Agent 完成，然后吸收其结论。在仍存在未解决的 Reg API FAIL 结论时，不要定稿设计。

**设计定稿前的确定性校验（与 `harness.review` 无关，始终执行）：** 逐条核对每个块大小 / 传输计数常量对所有支持 dtype 满足 `count * sizeof(T) >= 32`；UB 预算涵盖所有存活缓冲区且不超限；Cast 链符合支持矩阵。把核对结果写进设计文档——关掉评审后，这三项没有其他兜底。

在最终确定设计文档之前，对照 [references/common-failure-modes.md](references/common-failure-modes.md) 中的陷阱进行验证 —— 尤其是 UB 拷贝路径、32B DMA 最小值以及 Cast 支持矩阵。

对于 Ascend950 / `dav-3510`，还需对照 [references/reg-api-guide.md](references/reg-api-guide.md) 验证：在规划的向量路径中不得使用 `AscendC::MicroAPI`、不得使用 Membase、除 `asc_vf_call` 外不得使用裸 `asc_*` API，且不得使用经典 AscendC 的 compute/cast/reduce。

**退出条件：** 设计文档已存在，确定性校验已完成；若发起了评审，其结论均已吸收；若跳过，STATE.md 已记录跳过原因。

### 阶段 5：测试套件

> **simple 档：** 见「simple 档快速路径」——写完套件只做静态检查（`py_compile`），不 build/collect/跑用例；接口测试与用例收集推迟到阶段 6 唯一一次构建后。

在 `test_{OP}.py` 中构建完整的 pytest 测试套件，以 torch 作为 CPU 参考：

- 模块级 `SHAPES` 用例矩阵（最少：一个小 shape、一个非 tile 对齐 shape、一个大 shape，以及按定义文档每个边界场景各一个边界值用例）与 `DTYPES`（每个受支持 dtype 一项）
- 接口存在性测试：断言算子已注册到 `torch.ops.op_{OP}` 命名空间
- 用 `@pytest.mark.parametrize` 对 `(shape, dtype)` 参数化的 NPU 测试：在 NPU 上执行算子，与 torch CPU 参考用 `torch.allclose`（整型用 `torch.equal`）比对；以 `@pytest.mark.skipif(not torch.npu.is_available())` 守卫

检查命名：算子 host 入口应放在 `namespace op_{OP}` 内（而非顶层 `namespace {OP}`），以避免与标准库数学符号冲突（例如 `erf`、`sinh`、`exp`）—— 见 [references/common-failure-modes.md](references/common-failure-modes.md) § 测试命名冲突。

增量构建（`cd build && make -j`）后验证测试套件就绪：
- 运行接口存在性测试并确认通过：`pytest test_{OP}.py -v -k interface`。
- 确认参数化 NPU 用例能被正常收集：`pytest test_{OP}.py --collect-only -q`（应列出全部 `(shape, dtype)` 组合）。
- **不要**在此对骨架执行整套 NPU 用例——它们注定失败、只是消耗真机时间而不带来新信息；核函数正确性在阶段 6 验证。

**退出条件：** 测试套件就绪，接口测试通过，参数化用例可被收集。

### 阶段 6：核函数实现

> **simple 档：** 见「simple 档快速路径」——本阶段的 `make` 是全流程唯一一次 ASC 编译；构建后一次性跑接口测试 + `--collect-only` + 完整 NPU 用例。

在 `{OP}.asc` 中增量实现：

1. device kernel `{OP}_kernel<T>()`（`__global__ __aicore__`，按 tile 处理 UB 数据）
2. tiling 函数 `calc_{OP}_tiling_params()`（返回 numBlocks / blockLength / tileSize）
3. `namespace op_{OP}` 中的 host 入口：按 dtype 分发，以 `{OP}_kernel<dtype><<<numBlocks, nullptr, aclStream>>>(...)` 启动核函数并返回输出张量
4. `TORCH_LIBRARY(op_{OP}, m)` 的 schema 定义与 `TORCH_LIBRARY_IMPL(op_{OP}, PrivateUse1, m)` 的实现绑定

对于 Ascend950 / `dav-3510`，按 [references/reg-api-guide.md](references/reg-api-guide.md) 中的 Reg 形态实现向量计算、cast 和规约：

- `__simd_vf__` 函数作用于 `__ubuf__` 指针和 `AscendC::Reg::RegTensor` 值。
- `__aicore__` 包装器将 `LocalTensor.GetPhyAddr()` 转换为 `__ubuf__` 指针并调用 `asc_vf_call`。
- 使用 `AscendC::Reg::UpdateMask` 或 `CreateMask` 生成全量/尾块掩码。
- 将 DMA、入队、UB 分配与同步保留在常规 AscendC 集成代码中。
- 在 Reg 路径中不要使用 `AscendC::MicroAPI`、Membase、除 `asc_vf_call` 外的裸 `asc_*` API，或经典 `AscendC::Mul/Cast/ReduceSum/Sigmoid/Sqrt/Duplicate/Adds/Muls` 风格的计算调用。

每完成一个有意义的步骤后：增量构建（`cd build && make -j`，复用阶段 2 的 cmake 配置，勿 `rm -rf build`）、运行测试、与定义文档和设计文档比对。如果测试失败，先修复再继续。

在最终确定之前，对照 [references/common-failure-modes.md](references/common-failure-modes.md) 验证 —— 尤其是 device 侧辅助函数调用、TBuf 生命周期 和 AscendC Cast 支持矩阵。代码模式见 [references/implementation-patterns.md](references/implementation-patterns.md)。

对于 Ascend950 / `dav-3510`，实现完成后对照 [references/reg-api-patterns.yaml](references/reg-api-patterns.yaml) 扫描实际的核函数源码，扫描方式按 `harness.review`（见「文档评审开关」）：
- **跳过**（`off`，或 `auto` + simple 档）：不起子 Agent，由主 Agent 直接对 `{OP}.asc` grep 该 yaml 中的禁止模式，并核对掩码为元素计数、尾块存储已加掩码。
- **其余取值**：重新运行只读的 API 查询子 Agent 或评审提示词做扫描。

无论走哪条路径，都要在构建前解决每一项禁止 API 的发现。

在阶段 6 中使用多个 Agent 时，始终使用 `isolation="worktree"`。

**退出条件：** 所有本地测试通过。

### 阶段 7：验证

> **simple 档：** 见「simple 档快速路径」——跳过独立的干净重建（阶段 6 已是全新配置上的首次编译）；仅在怀疑陈旧产物时才 `rm -rf build` 重建一次。

在目标 NPU 硬件上做最终验证：

1. 在 NPU 上做一次**干净重建**（这是全流程唯一的干净重建，用于排除增量编译的陈旧产物）：`rm -rf build && mkdir build && cd build && cmake -DCMAKE_ASC_ARCHITECTURES=${NPU_ARCH} .. && make -j`，然后 `pytest test_{OP}.py -v`
2. 检视构建/测试日志，迭代直到全部用例通过

确认核函数源码包含完整实现（而非阶段 2 的骨架）。陈旧构建产物的故障排查见 [references/common-failure-modes.md](references/common-failure-modes.md) § 构建新鲜度。

**退出条件：** NPU 上构建通过，测试套件全部通过。

### 阶段 7.5：黑/白盒测试门禁

从 Team `AGENTS.md` frontmatter 读取 `harness.test_gate`，取值为 `on` 或 `off`，默认 `off`。

- 若为 `off`：在 `docs/{OP}/STATE.md` 记录按配置跳过，并继续阶段 8。
- 若为 `on`：按照 `ascendc-st-design` skill 执行黑盒用例生成与执行；
  按照 `ascendc-whitebox-design` skill 执行白盒用例生成与执行；保留真实生成用例、执行结果、日志和 provenance。
- 将黑盒与白盒产物路径写入 `docs/{OP}/test-harness/results/test_gate.json`，路径均相对于 `docs/{OP}`。
- 运行 `python3 ${CLAUDE_SKILL_DIR}/scripts/validate_test_gate.py --operator-doc-dir docs/{OP}`，确认输出 `STATUS: PASSED`。
- 若有精度用例失败，回到阶段 6 修复实现，重新完成阶段 7 和阶段 7.5，并在 `docs/{OP}/plans/troubleshooting.md` 记录非平凡问题。

**退出条件：** `harness.test_gate` 为 `off` 且已记录跳过；或黑/白盒测试门禁通过。

### 阶段 8：收尾文档

1. 确认 STATE.md 所有勾选框均已勾选。
2. 将所有剩余任务标记为已完成。
3. 向用户报告：`{OP}` Ascend C 实现完成。

**退出条件：** STATE.md 全部勾选，所有任务已完成。

## 阶段回退

如果在实现阶段 N 时发现某个阶段 M 产物（M < N）存在缺陷，更新该阶段 M 产物，若改动较为实质则重新运行其评审，再继续阶段 N。

## Agent 用法

使用 `Agent` 工具进行评审和可并行的分析。模式：

**并行评审 Agent** —— 是否发起由 `harness.review` 决定（见「文档评审开关」）；发起时在单条消息中以 `run_in_background=true` 启动：
- 为每个 Agent 起描述性的名字（例如 `math-review`、`ub-review`）
- 保持提示词具体：`Read X and Y, verify A/B/C, write findings to Z.`
- 要求结构化判定（PASS/FAIL/CONCERN）—— 见 [references/review-prompts.md](references/review-prompts.md)
- 等待完成通知；在继续之前确认结论文件存在且包含判定

**隔离探查** —— 使用 `subagent_type="Explore"` 进行只读研究：
- 在 `operators/` 范围内查找参考算子模式
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
- [scripts/detect_soc.py](scripts/detect_soc.py)：探测当前 NPU 的 `SocVersion` 与 `NpuArch`（dav-*），用于阶段 0 判定目标芯片
- [hooks/pre-build-check.sh](hooks/pre-build-check.sh)：自动化的构建前校验（检测 host-only 辅助函数）
- [hooks/post-edit-reminder.sh](hooks/post-edit-reminder.sh)：编辑后的核函数规范检查
