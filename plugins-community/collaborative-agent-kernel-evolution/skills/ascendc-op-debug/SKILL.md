---
name: ascendc-op-debug
description: "Unified AscendC operator runtime debug skill. Diagnose precision errors, runtime crashes, hangs, and multicore inconsistencies using a hypothesis-driven protocol with 3-layer evidence (code review → log analysis → tools). Covers hypothesis patterns for Vector operators (cache line conflict, workspace sizing, cross-tile accumulation, einsum semantics, TBuf compiler merge, etc.) plus msSanitizer, msDebug, and msaicerr tool escalation. MUST be invoked before any fix attempt when operator produces wrong output, crashes, hangs, or shows non-deterministic multicore results. Do NOT self-fix without running this skill first. 触发：算子产生错误输出、崩溃、挂死或多核结果不一致时。"
---

# ascendc-op-debug — AscendC 算子 Runtime 调试 Skill

<!-- AUTHORITATIVE: 强制调用条件 — 任何调用方必须在修改代码前完成本 skill 全流程 -->

## 强制调用条件（Mandatory Invocation Gate）

**以下任一症状出现时，调用方（含 cake agent）MUST 在修改任何代码前先执行本 skill 完整诊断流程（Step 1-4）：**

| 触发症状 | 典型表现 | 禁止的跳过行为 |
|---|---|---|
| 运行精度不通过 | 输出全零 / 精度偏差 / mismatch | ❌ 禁止猜测式直接修改代码 |
| 运行时崩溃 | 进程异常退出 / 507034 错误码 | ❌ 禁止重编译而不诊断 |
| 运行挂起/超时 | 多核死锁 / 等待超时 | ❌ 禁止仅加 timeout 掩盖问题 |
| 多核不一致 | 偶发性结果差异 | ❌ 禁止仅重试 |

> 编译器报错（非运行期错误）可直接修复，不需要调用本 skill。

启动本 skill 后立即输出：**`🔍 启动 ascendc-op-debug 诊断，症状：[症状描述]`**

---

## 定位
统一入口：AscendC 算子精度错误 / 运行时崩溃 / 挂起超时的完整诊断-修复-验证协议。
适用于单算子和融合算子（含多 Phase 融合）。

---

## 调试会话协议（CC 执行路径）

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1   症状采集                                                   │
│           ├── 由 cake agent 调用 → 自动从上下文收集                  │
│           └── 直接由用户调用   → 向用户索取症状信息                  │
│           输出：症状确认 | 触发条件 | 文件路径                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Step 2   查 INDEX.md，得候选假设列表（通常 1-3 条）                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Step 3   执行假设验证（按 evidence 层分支）                          │
│           ├── code  层: 读 read_code.md → grep 模式 → 代码匹配       │
│           ├── log   层: 读 parse_log.md → 解析错误码                 │
│           └── tools 层: 运行 msSanitizer / msDebug / msaicerr        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Step 4   输出结果                                                   │
│           根因 + fix_template + verify_cmd                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  After Fix   更新知识库                                              │
│              INDEX.md + TAXONOMY.md + retro/YYYYMMDD_*.md           │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1 — 症状采集

<!-- AUTHORITATIVE: 调用来源决定采集方式 -->

**调用来源判断：**
- **由 cake agent 调用**：从当前会话上下文自动收集（错误日志、命令输出、当前 op_name），**不得询问用户**
- **直接由用户调用**：向用户索取以下信息

采集以下信息（无论来源）：
1. 错误现象（崩溃 / 精度偏 / 挂起 / 多核不一致）
2. 触发条件（单核 or 多核？小 shape or 大 shape？同进程多 shape？）
3. 相关源文件路径（`op_host/` 和 `kernel/` 目录）

采集完成后输出：**`症状确认：[现象] | 触发条件：[条件] | 文件：[路径]`**

### Step 1.5 — api-check 前置（精度类症状强制执行）

**当症状为 `precision_bias` 时，在查 INDEX.md 之前，MUST 先执行：**

```
Read: skills/cake-code-review/api-best-references/api-restrictions.md
```

重点检查：
- **§1 Cast 路径**：是否有 `CAST_ROUND + CAST_ROUND` 两步 cast（H26 双重舍入）
- **§2 Reduce API**：`ReduceMax/ReduceSum` 的 `dst` 是否与 `tmpBuffer` 相同（H19）

如 api-check 已命中根因，直接输出 fix_template，**跳过 Step 2-3**。

> **推荐调试顺序**：api-check（~5min）→ H01-H26 假设验证（~30min）→ 手动 Print/GetValue 调试

### Step 2 — 查 INDEX.md，得候选假设列表

**MUST read INDEX.md before proceeding:**
```
Read: skills/ascendc-op-debug/INDEX.md
```
按 symptom + when 两列定位候选 H 编号（通常 1-3 个）
优先选 evidence=code 的假设（成本最低）

After reading INDEX.md, state: **`候选假设：[H编号列表]，依据：symptom=[xxx] + when=[xxx]`**

### Step 3 — 执行假设验证（按 evidence 层分支）

**evidence=code（Layer 1，优先）**
```
Read: protocols/read_code.md  ← 知道读哪里
按 read_target 中指定的文件和 grep 关键词定位
匹配 code_pattern → 命中则输出 fix_template + verify_cmd
```

**evidence=log（Layer 2，次选）**
```
Read: protocols/parse_log.md
执行 plog 解析或错误码查表
```

**evidence=tool_sanitizer / tool_msaicerr（Layer 3，兜底）**
```
Read: protocols/run_tools.md
告知用户需要专项工具检测，给出命令
```

### Step 4 — 输出结果
命中假设后输出：
- **根因**：一句话（精确到代码行）
- **fix_template**：可直接替换的代码
- **verify_cmd**：验证步骤

所有假设未命中 → 进入 Layer 3 工具检测。

---

## After Fix Protocol（复盘触发器）

**每次成功定位并修复 bug 后，必须执行以下步骤：**

### 复盘步骤
1. 检查 `INDEX.md` — 确认本次 bug 是否已有对应 hypothesis 覆盖
2. **已覆盖**：用本次 `code_pattern` 或 `verify_cmd` 更新现有 H 文件
3. **未覆盖**（新模式）：
   a. 检查 `TAXONOMY.md` — 确认 symptom/when/root_cause 是否有合适的现有值
   b. **有合适值**：直接使用
   c. **无合适值**（真正新类型）：先在 `TAXONOMY.md` 对应维度末尾追加新条目，遵循命名规范
   d. 按 `TEMPLATE.md` 在 `retro/YYYYMMDD_短描述.md` 创建新 hypothesis
   e. 执行 `scripts/validate_hypothesis.sh retro/YYYYMMDD_短描述.md`
   f. 通过后执行 `scripts/build_index.sh` 重建 INDEX.md

> 这一步不是可选项，是协议的组成部分。知识库靠每次复盘自然生长。

---

## 快速参考：症状 → 假设映射

| 症状 | 触发条件 | 首先怀疑 |
|---|---|---|
| 输出全零 | 多核 | H01, H10 |
| 精度偏差 | D=5120 时 | H03, H07 |
| 精度偏差 | 系统性线性偏差 | H05 |
| 崩溃 | 同进程小→大 shape | H02 |
| 崩溃 | 大 shape / 大 D | H02, H08 |
| 多核不一致 | 偶发 | H01, H11 |
| 挂起/超时 | 多核 | H12 |

详细映射见 `INDEX.md`。

---

## 文件结构

```
ascendc-op-debug/
├── SKILL.md          ← 本文件，协议入口
├── TAXONOMY.md       ← 受控词汇表（宪法）
├── TEMPLATE.md       ← 新 hypothesis 模板
├── INDEX.md          ← 自动生成的双索引（symptom + root_cause）
├── hypotheses/       ← H01-H14 假设知识库
├── protocols/        ← 三层证据执行细节
│   ├── read_code.md
│   ├── parse_log.md
│   └── run_tools.md
├── scripts/
│   ├── validate_hypothesis.sh
│   └── build_index.sh
└── retro/            ← 复盘暂存区（待验证的新 hypothesis）
```
