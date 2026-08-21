# SHMEM Agent Skills — SHMEM 算子开发 AI 辅助技能工具集

预配置的 10 技能工具链，覆盖 SHMEM 通信算子与通算融合算子从需求到交付的端到端开发

```
Phase 0 需求确认 → 1 设计 → 2 用例 → 3 代码 → 4 编译+正确性 → 5 代码走读 → 5.5 Torch 接入 → 6 性能基线 → 6.5 性能优化（条件性） → 7 最终交付
```

本仓库提供

- `skill-format-guide.md` — 技能编写规范（frontmatter / 章节顺序 / 约束语言）；**Skill 树仅 `.md`，不附带 `.sh`/`.py` 文件；文档内可引用 `${SHMEM_REPO}/` 仓内脚本与源码**
- `shmem-ops-dev/SKILL.md` — 编排器：Phase 0-7 全流程调度，子 skill 串行编排
- `shmem-ops-*/SKILL.md` — 9 个专职 skill，分别负责算子开发的一个阶段（设计 / 用例 / 代码 / 编译调试 / 正确性 / 走读 / Torch 接入 / 性能评估 / 性能优化）

**仓内文档约定**：Skill 与 SHMEM 仓可分离部署。读 `${SHMEM_REPO}/docs/`、`examples/` 等 **SHMEM 原生**路径前 **MUST** [定位仓库并区分 custom-ops 交付物](shmem-ops-dev/references/shmem-repo-resolution.md)（§1.1）；`docs/` **只读**。

---

## 技能调用方式（用户主动 @ 指定）

本仓库 skill 位于 **`.agents/skills/`**，由用户在对话中 **主动 @ 或引用路径** 指定，例如：

```text
@.agents/skills/shmem-ops-dev/SKILL.md
```

或：

```text
参考 .agents/skills/shmem-ops-dev/SKILL.md，在 custom-ops 下实现 alltoallv…
```

**Agent 被指定 `shmem-ops-dev` 后的固定首步**（Phase 0，先于任何其它 Read/Shell）：

1. Read `shmem-ops-dev/SKILL.md`（用户 @ 的路径；上溯记录 `skills_root`）
2. Read `shmem-ops-dev/references/askquestion-template.md`
3. Read `shmem-ops-dev/references/agent-execution-contract.md`
4. Read `shmem-ops-dev/references/shmem-repo-resolution.md`（Phase 0 内只读；定位规则供 Phase 1+ 使用）
5. **先探测缺项再问**：按 `askquestion-template.md` 对已能从消息/环境/仓库探测出的项直接记录，仅对缺项调用 `AskQuestion`（CANN 仅 A/B + **Other** 填路径）；通算融合算子另读 [fused-compute-phase-gates.md](shmem-ops-dev/references/fused-compute-phase-gates.md)
6. 全部 intake 项确认并写入 `phase0_intake`（含 `shmem_repo`、`skills_root`）后，再调度子 skill / 读仓内源码 / 编译

总索引：[shmem-ops-dev/references/GUIDE.md](shmem-ops-dev/references/GUIDE.md)

子 skill 同理：编排器在对应 Phase **Read 该子 skill 的 `SKILL.md`**，不依赖全局 Skills 自动发现。

---

## 快速开始

1. **@ 指定编排器 skill**，并用自然语言描述算子需求：

> 现在需要你基于 shmem 在 custom-ops（自定义独立算子工程）目录下实现一个 alltoallv 算子，在 alltoall 的基础上，允许不同 PE 传输交换不同的数据量。要求数据类型为 int/fp16，单机 8 卡环境，full-mesh 连接，p2p 带宽 28GB/s。注意采用独立编译模式，不将其作为 shmem 的一个 example。

2. 结合伪代码输入（通信密集型算子，非通算融合）：

> 现在需要你基于这个 moe dispatch 伪代码，在 custom-ops 目录下实现 moe dispatch 通信算子……

3. 编排器按 Phase 0-7 推进，在每个 phase **Read 并执行** 对应 `.agents/skills/shmem-ops-*/SKILL.md`。

**Phase 0 表单**：以 [`askquestion-template.md`](shmem-ops-dev/references/askquestion-template.md) 为准（用户 @ 指定 skill 后 Agent **MUST** Read 该文件再弹窗）。详见 [`intake-checklist.md`](shmem-ops-dev/references/intake-checklist.md)。

---

## 工作流

```text
Phase 0 需求确认
  ↓
Phase 1 设计  ──────────────────────────── design.md
  ↓
Phase 2 用例生成  ───────────────────────── case matrix + scripts
  ↓
Phase 3 代码生成  ───────────────────────── src/ + CMakeLists
  ↓
Phase 4 编译 + 正确性验证
  ↓
Phase 5 代码走读  ───────────────────────── review-report.md
  ↓
Phase 5.5 Torch 接入（条件性，`meta.torch_required: true`）  ── aclshmem_torch.so + torch_test_<op_name>.py
  ↓
Phase 6 性能基线评估（条件性，`meta.performance_required: true`）  ── performance_report.md
  ├─ 达标 ──→ Phase 7 最终交付
  ├─ 未达标 + `performance_auto_optim: false` ──→ Phase 7（仅报告差距）
  └─ 未达标 + `performance_auto_optim: true` ──→ Phase 6.5 性能优化（进入后跑满 `max_opt_rounds` 默认 5 轮，达标不是提前停止理由）──→ Phase 7 最终交付
```

| Phase | 名称 | Skill | 输入 | 输出 |
|-------|------|-------|------|------|
| 0 | 需求确认 | `shmem-ops-dev`（编排器自处理） | 算子需求 + 拓扑 | op_name / dtype / 环境 |
| 1 | 设计 | `shmem-ops-design` | 确认的需求 | `design.md` |
| 2 | 用例生成 | `shmem-ops-testcase-gen` | `design.md` | case matrix + gen_data + checker + `scripts/run.sh` |
| 3 | 代码生成 | `shmem-ops-code-gen` | `design.md` | src/ + CMakeLists.txt + README.md |
| 4 | 编译 + 正确性 | `shmem-ops-compile-debug` + `shmem-ops-correctness-eval` | 源码 + case matrix | 编译产物 + correctness 结果 |
| 5 | 代码走读 | `shmem-ops-code-review` | design.md + 实现代码 | `review-report.md` |
| 5.5 | Torch 接入 | `shmem-ops-torch-bind` | design.md + 算子源码 | `aclshmem_torch.so` + `torch_test_<op_name>.py` |
| 6 | 性能基线 | `shmem-ops-performance-eval` | 通过 Torch 接入（或已跳过）的算子 | `performance_report.md` |
| 6.5 | 性能优化 ⚠条件性 | `shmem-ops-performance-optim` | performance_report.md | 优化代码 + 轮次报告（须 `performance_auto_optim: true` 且 Phase 6 未达标） |
| 7 | 最终交付 | `shmem-ops-dev`（编排器自处理） | 全部产出物 | 交付摘要 |

**阶段门控**：前一阶段检查点全部通过后才进入下一阶段。核心约束：
- 设计驱动编码：代码生成依赖 `design.md` 的 Canonical DSL 和六类模块设计
- 先 correctness 再 performance：功能或正确性未通过时，不做性能优化
- 性能优化轮次：默认「未达标 + `performance_auto_optim: true`」才触发 Phase 6.5；进入后 **MUST** 跑满 `max_opt_rounds`（默认 5）轮，达标 **不是** 提前停止理由；用户明确指定优化轮次时跑满指定轮次。每轮记录假设、改动、正确性复测和性能结果

---

## Skill 概览

| Skill | 类型 | Phase | 职责 |
|-------|------|-------|------|
| `shmem-ops-dev` | 编排器 | 0–7（含 5.5、6.5） | 推进 phase、强制门禁、调度子 skill、Phase 7 交付摘要。本身不直接做领域工作 |
| `shmem-ops-design` | 文档生成 | 1 | 将需求转化为 `design.md`（Canonical DSL + 能力映射 + 缺口分析 + 契约） |
| `shmem-ops-testcase-gen` | 文档/脚本生成 | 2 | 生成 case matrix（≥20 cases, XS/S/M/L 四档）、golden/checker、测试脚本 |
| `shmem-ops-code-gen` | 代码生成 | 3 | 根据 `design.md` 生成算子代码（渐进式：生命周期→内存→传输→同步→计算→调度） |
| `shmem-ops-compile-debug` | 构建与调试 | 4 | 编译、运行、失败分类（编译/运行/正确性/性能四类）、调试闭环 |
| `shmem-ops-correctness-eval` | 验证 | 4 | 执行 case matrix、分类失败、验证 invariants、输出 correctness 报告 |
| `shmem-ops-code-review` | 检查 | 5 | 设计-代码一致性遍历（CMake/block_dim/transport/sync/compute/style），输出 `review-report.md` |
| `shmem-ops-torch-bind` | 集成生成 | 5.5 | 封装为 PyTorch CustomClass，生成 Python 多 PE 测试脚本，编译验证 |
| `shmem-ops-performance-eval` | 评测 | 6 | 性能采集、baseline 对比（HCCL/aclnn）、瓶颈分析、Device Frame Table |
| `shmem-ops-performance-optim` | 迭代优化 | 6.5（条件性） | 5 轮机制优化（Round 1~5：机制改动 + 可选参数扫描），每轮：假设→改动→复测→决策 |

编排器（`shmem-ops-dev`）在每个 phase 调用对应子 skill。子 skill 之间通过文件交换信息：
- `design.md` — 设计契约（design → testcase-gen / code-gen / code-review）
- case matrix + 测试脚本 — 测试输入（testcase-gen → compile-debug / correctness-eval）
- `performance_report.md` — 性能数据（performance-eval ↔ performance-optim），交付判断依据

---

## Skill 详解

### shmem-ops-dev — 编排器

适用场景：收到 SHMEM 算子开发任务，需要端到端完成从需求到交付的全流程。

工作流：Phase 0 需求确认 → Phase 1 设计 → Phase 2 用例 → Phase 3 代码 → Phase 4 编译+正确性 → Phase 5 走读 → Phase 5.5 Torch 接入 → Phase 6 性能基线 → Phase 6.5 性能优化（条件性） → Phase 7 最终交付。

核心原则：Phase 0–7 串行（含子阶段 5.5 / 6.5）、阶段门控、设计驱动编码、先 correctness 再 performance、Phase 6.5 默认未达标才触发且进入后跑满 `max_opt_rounds` 轮、不静默修改核心库。Phase 7 为编排器自处理的交付摘要，无专职 skill。

### shmem-ops-design — 算子设计

适用场景：将用户的自然语言算子需求、伪代码或参考实现转化为结构化的`design.md` 设计文档。

需要提供：算子名称、数据类型、拓扑信息。

交付：`design.md`，包含 Canonical DSL（YAML）、Capability Mapping 表（六类：lifecycle / memory / transport / sync / compute / scheduler）、Gap Analysis、实现边界、编译/测试契约。

### shmem-ops-testcase-gen — 用例生成

适用场景：根据 `design.md` 的 correctness contract 生成测试计划、数据生成脚本和精度验证脚本。

交付：case matrix（≥20 cases，XS/S/M/L 四档规模）、`gen_data.py`（golden 数据生成）、`check_result.py`（精度验证）、`scripts/run.sh`（批量运行脚本）。

### shmem-ops-code-gen — 代码生成

适用场景：根据 `design.md` 生成 SHMEM 算子完整代码结构和实现。

交付：目录结构（`src/` + `scripts/` + `docs/`）、`CMakeLists.txt`、`main.cpp`、`*_kernel.{cpp,h}`、`README.md`。生成过程遵循渐进式顺序：生命周期 → 内存 → 传输 → 同步 → 计算 → 调度。

支持通信算子（`shmem-ops-code-gen/templates/communication/`）与通算融合算子（`shmem-ops-code-gen/templates/fused-compute/`）。模板分文件承载，Agent 按 [communication/GUIDE.md](shmem-ops-code-gen/templates/communication/GUIDE.md) 或 [fused-compute/GUIDE.md](shmem-ops-code-gen/templates/fused-compute/GUIDE.md) 索引分步读取并提取 code block。

### shmem-ops-compile-debug — 编译与调试

适用场景：编译 SHMEM 算子、运行测试、对失败进行分类和调试。

工作流：判断构建模式 → 构建 → 运行 → 失败分类（编译/运行/正确性/性能四类）→ 定位根因。支持独立工程（`independent_project`）和仓内示例（`in_tree_example`）两种构建模式。

**常见环境问题**：`scripts/run.sh` 写死 `IPPORT`/`SHMEM_UID_SESSION_ID` 或与长期运行的 `torch_test_*.py` 冲突 → shmem init 失败 → 输出 bin 全 0 假 FAIL。见 `env-setup.snippet.md` 的 `setup_shmem_dynamic_endpoints` 与 `debug.md` §4.1。

### shmem-ops-correctness-eval — 正确性评估

适用场景：按 case matrix 全量执行正确性验证，输出正确性报告。

工作流：执行 case matrix → 收集结果 → 失败分类（数据生成 / 代码逻辑 /同步时序 / 精度偏差）→ 验证 invariants → 输出报告。

注意：本地 skill 负责结果判定和报告，运行和失败定位由 `shmem-ops-compile-debug`完成。

### shmem-ops-code-review — 代码走读

适用场景：对比 `design.md` 和实现代码，检查设计-代码一致性，生成最终交付的`review-report.md`。

检查维度：CMake 一致性、block_dim 匹配、传输路径（API/engine/chunk）、同步屏障（barrier/signal/wait 正确性）、计算逻辑（语义与精度的匹配）、代码风格。

Phase 5 使用 `interim` 模式生成阶段性 `review-report.md`（性能章节待补齐）；Phase 7 使用 `final` 模式补齐性能摘要。

### shmem-ops-torch-bind — Torch 接入

适用场景：`meta.torch_required: true` 时，将已通过走读的算子封装为 PyTorch CustomClass。

交付：`torch_bind_<op_name>.cpp`、`aclshmem_torch.so`、`scripts/torch_test_<op_name>.py`。参考 `${SHMEM_REPO}/examples/torch_binding/` 与 `${SHMEM_REPO}/examples/python_extension/torch_test/`（先定位 `SHMEM_REPO`）。

当 `meta.torch_required: false` 时，编排器跳过本 Phase。

### shmem-ops-performance-eval — 性能评估

适用场景：`meta.performance_required: true` 且正确性通过后，采集算子性能数据，接入 HCCL/aclnn baseline（C++ 可执行文件 + perf-workflow §1 阶段 C），在聊天与 `performance_report.md` 输出 **SHMEM vs baseline 对比表**，分析性能瓶颈。

交付：`performance_report.md`（§3.5 对比表 + §7 结论），聊天 **MUST 自动输出** S 档 + L 档性能对比表（[perf-chat-output-spec.md](shmem-ops-performance-eval/references/perf-chat-output-spec.md)）。Phase 6.5 每轮 MUST 输出 Δ% 变化表。

### shmem-ops-performance-optim — 性能优化（Phase 6.5，条件性）

适用场景：`meta.performance_required: true` 且 `meta.performance_auto_optim: true`，且 Phase 6 未达标时，需要系统化提升性能。

工作流：瓶颈定位 → 基线锁定 → 5 轮优化迭代（Round 1~5 机制优化 + 可选参数扫描）→ 正确性复测 → 性能验证。

每轮记录：假设 → 代码/参数改动 → 正确性复测结果 → 性能采集 → 与 baseline 对比→ 决策（继续/回溯/停止）。全部 5 轮完成后，将最优参数合并到最终代码。

**注意**：此 Phase 仅在 Phase 6 未达标 **且** `meta.performance_auto_optim: true` 时由编排器自动进入；`performance_auto_optim: false` 时仅报告差距；达标则直接跳至 Phase 7 最终交付。

### Phase 7 — 最终交付（编排器自处理）

适用场景：Phase 6 达标或 Phase 6.5 完成后，输出最终交付摘要。

交付：编排器汇总全部产出物，并调用 `shmem-ops-code-review`（`final` 模式）更新 `review-report.md`。摘要包含算子名称、目录结构校验、修改文件列表、编译命令、测试命令、正确性验证结果表、Torch 结果（若适用）、性能数据和 baseline 对比、优化轮次摘要（如有）、未完成项和风险。

此阶段无专职 skill，由 `shmem-ops-dev` 编排器直接处理。

---

## 开发产物

每个算子在算子目录下生成。Phase 0–7（含条件性 Phase 5.5 / 6.5）完整走通后的文件结构：

```
<op_name>/
├── docs/
│   ├── design.md                  ← Phase 1：Canonical DSL + 能力映射 + 契约
│   ├── review-report.md           ← Phase 5（interim）+ Phase 7（final 补齐）
│   ├── performance_report.md      ← Phase 6-6.5：性能基线 + 优化轮次数据
│   ├── correctness_report.md      ← Phase 4：正确性验证报告
│   └── case_matrix_report.md      ← Phase 4：case matrix 执行结果
├── src/
│   ├── main.cpp                   ← Phase 3：Host 侧入口
│   ├── <op>_kernel.h              ← Phase 3：Device 侧 kernel 声明
│   └── <op>_kernel.cpp            ← Phase 3：Device 侧 kernel 实现
├── torch_binding/                 ← Phase 5.5（torch_required: true）
│   ├── CMakeLists.txt
│   ├── include/shmem_torch_kernel.h
│   └── src/torch_bind_<op>.cpp
├── scripts/
│   ├── gen_data.py                ← Phase 2：golden 数据生成
│   ├── check_result.py            ← Phase 2：精度验证
│   ├── `scripts/run.sh`                     ← Phase 2：运行脚本
│   ├── run_case_matrix.py         ← Phase 2：批量 case 执行
│   ├── torch_test_<op>.py         ← Phase 5.5：PyTorch 多 PE 测试
│   └── perf.sh                      ← Phase 6：性能采集（custom-ops 生成交付物）
├── CMakeLists.txt                 ← Phase 3：构建配置
└── README.md                      ← Phase 3：算子说明文档
```

---

## 仓库布局

```
.agents/skills/
├── README.md                              ← 当前文件
├── skill-format-guide.md                  ← 技能编写规范
├── shmem-ops-dev/
│   ├── SKILL.md                           ← 编排器：Phase 0-7（含 5.5、6.5）全流程调度
│   └── references/
│       ├── askquestion-template.md        ←   Phase 0 Intake：先探测缺项再问（消息/环境/仓库可探测项直接记录，仅缺项 AskQuestion）
│       ├── GUIDE.md                       ←   Phase 0–7 总索引与子 skill 导航
│       ├── intake-checklist.md            ←   启动门禁与 intake 检查清单（探测优先）
│       ├── agent-execution-contract.md    ←   禁止中断/Torch自动/性能双表/重生成检查单
│       ├── fused-compute-phase-gates.md   ←   通算融合算子 Phase 0/6/6.5/7 门控与交付规范
│       ├── idle-npu-selection.md          ←   上板前 npu-smi 选空闲卡写入 device_list
│       ├── shmem-repo-resolution.md       ←   SHMEM_REPO 定位；仓内路径 vs skill 相对路径
│       └── docker-exec-contract.md        ←   用户指定 Docker 时全部命令进容器
├── shmem-ops-design/
│   ├── SKILL.md                           ← 算子设计
│   └── references/
│       ├── op-dsl.md                      ←   Canonical DSL 规范
│       ├── capacity.md                    ←   SHMEM 能力与功能域
│       ├── core-allocation.md             ←   分核 / phase / chunk / lane 策略
│       ├── design-template.md             ←   design.md 输出模板（9 段式）
│       ├── implementation-boundary.md     ←   Device/Host/Python 边界 + unified kernel 路径
│       └── hardware-architecture.md       ←   AI Core / 存储层次 / 传输引擎
├── shmem-ops-testcase-gen/
│   ├── SKILL.md                           ← 测试用例生成
│   └── references/
│       ├── GUIDE.md                       ←   参考索引 + 仓内 examples 路径
│       ├── correctness.md                 ←   golden 构造策略 / invariant→test 映射
│       ├── precision-standard.md          ←   精度标准 / rtol-atol / 双统计判定
│       ├── test-structure-template.md     ←   gen_data / check_result / `scripts/run.sh` 模板
│       └── testcase-scale-standard.md     ←   规模分档（XS/S/M/L）/ case matrix 模板
├── shmem-ops-code-gen/
│   ├── SKILL.md                           ← 代码生成
│   ├── references/
│   │   ├── api.md                         ←   SHMEM API 选择参考
│   │   ├── code-patterns.md               ←   Host/Device 代码组织 / RMA 模式
│   │   ├── atomic-add-pattern.md          ←   SetAtomicAdd 累加模式
│   │   ├── internal-api-boundary.md       ←   aclshmemi_* 内部 API 边界与等价公开 API
│   │   ├── code-style.md                  ←   C/C++ 代码规范
│   │   └── readme-spec.md                 ←   README.md 格式规范
│   └── templates/
│       ├── communication/
│       │   ├── GUIDE.md                    ←   通信算子模板索引与分步阅读表
│       │   ├── templates-cmake-main.md     ←   CMakeLists + main.cpp
│       │   ├── templates-kernel.md           ←   kernel.h + kernel.cpp
│       │   └── templates-scripts.md        ←   gen_data / check_result / `scripts/run.sh`
│       └── fused-compute/
│           └── GUIDE.md                    ←   通算融合算子模板索引与分步阅读表
├── shmem-ops-compile-debug/
│   ├── SKILL.md                           ← 编译与调试
│   └── references/
│       ├── build-test.md                  ←   构建模式 / CMake 模板
│       ├── custom-ops-entrypoints.md      ←   custom-ops 命令代码段（编译/运行/matrix/Torch）
│       ├── cann-env-resolution.md         ←   CANN 环境定位 / set_env.sh 解析
│       ├── env-setup.snippet.md           ←   `scripts/run.sh`运行时环境（内联函数）
│       ├── debug.md                       ←   失败分类 / 调试开关 / 同步问题排查
│       ├── shmem-repo-docs-index.md       ←   仓内 docs/、include/ 只读索引
│       ├── log-debug.md                   ←   SHMEM 日志环境变量与日志格式
│       └── dump-debug.md                  ←   AscendC DumpTensor / printf 调测
├── shmem-ops-correctness-eval/
│   └── SKILL.md                           ← 正确性验证
├── shmem-ops-code-review/
│   ├── SKILL.md                           ← 代码走读
│   └── references/
│       ├── code-review-checklist.md       ←   分类检查项 / 正确-错误示例
│       └── review-report-template.md      ←   走读报告模板（6 段式）
├── shmem-ops-torch-bind/
│   ├── SKILL.md                           ← Torch CustomClass 封装 + Python 多 PE 测试
│   └── references/
│       ├── GUIDE.md                       ←   参考索引 + 仓内 examples 路径
│       └── custom-ops-torch-layout.md     ←   custom-ops 共享 torch_binding 布局
├── shmem-ops-performance-eval/
│   ├── SKILL.md                           ← 性能评估
│   ├── references/
│   │   ├── performance-eval-guide.md      ←   采集流程 / contract 字段 / 结果表格式
│   │   ├── baseline-selection.md          ←   baseline 选择策略 / HCCL-aclnn 清单 / HCCL 展开模式
│   │   ├── baseline-compare-workflow.md   ←   baseline 接入 / 离线对比 / 聊天对比表
│   │   ├── perf-workflow.md               ←   性能三阶段采集命令代码段
│   │   ├── perf-compare-spec.md           ←   同量对比 / 达标口径 / 非饱和规则
│   │   ├── hccl-expansion-mode.md         ←   HCCL_OP_EXPANSION_MODE：优先 AIV，回退 AI_CPU
│   │   ├── perf-chat-output-spec.md       ←   聊天自动输出 / 优化轮次 Δ% 表
│   │   ├── timing-and-metrics-standard.md ←   时间打点 / 双指标方案 / 计算公式
│   │   ├── profiling-tools.md             ←   msprof / SHMEM cycle profiling
│   │   ├── device-profiling-guide.md      ←   SHMEMI_PROF / aclshmemx_get_prof 操作指南
│   │   └── skill-performance-review.md    ←   性能指标历史修正记录（steady_bus 删除等）
│   └── templates/
│       └── performance-report.md          ←   性能报告模板（7 段式）
└── shmem-ops-performance-optim/
    ├── SKILL.md                           ← 性能优化
    └── references/
        ├── optimization-patterns.md       ←   优化模式参考
        └── compute-optimization.md        ←   计算段优化模式（通算融合）
```

---

## 进一步阅读

- [skill-format-guide.md](skill-format-guide.md) — 技能编写规范：frontmatter 定义、章节顺序、约束语言、skill 类型差异、写作清单
- [shmem-ops-dev/SKILL.md](shmem-ops-dev/SKILL.md) — 编排器完整文档：核心原则、子 skill 清单、工作流总览、反模式清单
