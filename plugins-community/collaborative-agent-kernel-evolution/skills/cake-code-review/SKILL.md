---
name: cake-code-review
disable-model-invocation: true
description: Review AscendC kernel code for structural red-line violations (P0–P3) and algorithm correctness. Use after dsl-lowering, before ascendc-evaluation. Outputs rectification report; guides fix → recompile → unit test → system test.
---

# AscendC 编码红线 Review 技能包

## 概述

这是一套完整的 AscendC 算子编码规范审查、修复、验证工作流技能包。帮助系统性地检查和修复编码红线问题，确保代码质量和安全性。

```
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 1   API 合规预检（本文件，自动执行）                           │
│           基于 api-dispatch.json 扫描黑名单与最佳实践违规            │
│           → 违规? → 输出违规列表  → 无违规? → ✅ PASS               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  阶段 2   Structural Review（ascendc-review.md）                    │
│           扫描 7 大编码红线 + TopN 结构/安全问题                     │
│           输出 JSON + Markdown 报告                                  │
│           └── 输出带优先级的问题列表                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  阶段 3   Algorithm Correctness Check                               │
│           （ascendc-algorithm-check.md）                            │
│           检查归一化链、分母安全、迭代更新、归约消费、端到端闭环     │
│           输出算法问题清单（JSON + Markdown fragment）               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  阶段 4   Fix（ascendc-fix.md）                                     │
│           按 P0 → P1 → P2 → P3 优先级应用标准修复模式               │
│           生成整改报告                                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  阶段 5   Verify（ascendc-verify.md，可选）                         │
│           编译 → UT → ST 三层验证                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**完整执行顺序**：
1. **API 合规预检**（本文件，自动执行）— 基于 `api-dispatch.json` 扫描黑名单与最佳实践违规
2. **Structural Review** (`ascendc-review.md`) — 扫描 7 大编码红线 + TopN 结构/安全问题，输出 JSON + Markdown 报告
3. **Algorithm Correctness Check** (`ascendc-algorithm-check.md`) — 检查归一化链完整性、分母安全与不变量、迭代更新完整性、归约结果消费一致性、端到端闭环
4. **Fix** (`ascendc-fix.md`) — 按 P0→P1→P2→P3 优先级应用标准修复模式，生成整改报告
5. **Verify** (`ascendc-verify.md`) — 仅做编译→UT→ST 三层验证（可选，按需执行）

## 使用方式

### 完整工作流（默认）

```
使用 cake-code-review 对 <算子路径> 算子进行检查
```

### 仅 Review（不修复）

```
使用 ascendc-review 检查 <算子路径> 算子，不要修复
```

### 仅修复特定问题

```
使用 ascendc-fix 修复 <算子名> 的 <问题ID> 问题
```

### 仅验证编译

```
使用 ascendc-verify 验证 <算子路径> 的编译（跳过 UT/ST）
```

---

## API 合规预检（每次 Review 前自动执行）

在进入红线 Review 之前，先执行 API 合规预检。grep 模式从 `api-dispatch.json` 动态派生，新增 API 只需改 dispatch 文件，本流程无需修改。

### 步骤 1：定位 dispatch 文件并生成 grep 模式

```bash
# 动态定位，兼容任意 CWD
DISPATCH=$(find . -maxdepth 6 -name "api-dispatch.json" -path "*/api-best-references/*" 2>/dev/null | head -1)
[ -z "$DISPATCH" ] && echo "ERROR: api-dispatch.json not found" && exit 1

PATTERN=$(python3 -c "
import json, sys
d = json.load(open('$DISPATCH'))
print('|'.join(d['conditional'].keys()))
")

# 动态定位，兼容 output/{op_name}/op_kernel/ 和 output/{op_name}/{op_name}Custom/op_kernel/ 等结构
find output/{op_name} -type f \( -name "*.cpp" -o -name "*.h" \) -path "*/op_kernel/*" 2>/dev/null \
  | xargs -r grep -hEow "($PATTERN)" 2>/dev/null | sort -u
```

### 步骤 2：读取 dispatch 表，确定加载哪些文档

以 `$DISPATCH` 所在目录为基准：
- `always` 字段列出的文件**每次必读**（当前为黑名单）
- `conditional` 字段：将步骤 1 扫描到的 API 名与各 key 正则交集匹配，命中则读取对应文档

### 步骤 3：定向加载，输出违规列表

只读取命中的文档，输出：

```
[API_VIOLATION] {file}:{line} — {违规描述} → {推荐修复方案}
```

无违规则输出：`✅ API 合规检查通过`

> **扩展方式**：新增 API 规则只更新 `api-dispatch.json` + 对应 reference 文件，本流程自动覆盖。

---

## 责任边界

- **API 合规预检** — 负责 API 调度、黑名单和最佳实践 dispatch 检查
- **`ascendc-review.md`** — 负责结构性、红线、安全性和代码模式审查
- **`ascendc-algorithm-check.md`** — 负责算法完整性、数学不变量、归一化链、迭代更新完整性、归约结果消费一致性检查
- **`ascendc-fix.md`** — 负责按优先级应用标准修复模式，生成整改报告；消费 review 和 algorithm-check 的问题清单
- **`ascendc-verify.md`** — 只负责编译、UT、ST 执行和修复后的回归确认
- **`ascendc-evaluation`** — 负责更深层的数值对比、参考实现比较、精度/性能评估；不属于 `ascendc-verify`

---

## 结构/安全检查项（由 ascendc-review.md 执行）

API 合规预检完成后，先进入结构/安全审查阶段；算法完整性检查由后续 `ascendc-algorithm-check.md` 单独负责。

**7 大编码红线**（必查）:
- 1.1 除零保护
- 1.2 数组越界保护
- 1.3 溢出保护（int64 偏移）
- 1.4 变量初始化
- 1.5 空指针保护
- 1.6 资源匹配（InitBuffer/AllocTensor/FreeTensor）
- 1.7 数据竞争

**TopN 问题**:
- 2.1 特殊值处理（nan/inf/±0）
- 2.2 输入校验（shape/attr 范围）
- 2.3 GM 偏移使用 int64_t
- 2.4 stdlib 数学函数禁用
- 2.5 PipeBarrier 命名空间
- 2.6 Tiling 字段名一致性

## 严重性分级

| 级别 | 描述 | 处理时限 |
|------|------|----------|
| P0 - 极严重 | 可能导致崩溃、越界 | 立即修复 |
| P1 - 严重 | 可能导致数据错误 | 24小时内 |
| P2 - 中等 | 影响稳定性 | 1周内 |
| P3 - 低 | 代码可维护性 | 2周内 |

## 文件索引

**子技能**：
- `ascendc-review.md` — 结构/安全红线检查引擎
- `ascendc-algorithm-check.md` — 算法正确性检查
- `ascendc-fix.md` — 标准修复模式应用
- `ascendc-verify.md` — 编译和测试验证

**API 参考**：
- `ascendc-api-check.md` — API 最佳实践参考索引（`api-best-references/` 目录导航入口，含 API 类别索引、场景索引和黑名单入口）
- `api-best-references/api-dispatch.json` — API 合规检查规则的**单一维护点**，新增 API 规则只改此文件

## 输出产物

1. **整改报告** — `review/{算子名}_红线整改.md`（Fix 阶段）
2. **修复后的源代码** — 原地修改（Fix 阶段）
3. **算子文档更新** — `docs/npu_{算子名}.md`（Fix 阶段，如涉及特殊值约束）
4. **验证报告** — `review/{算子名}_验证报告.md`（Verify 阶段）

## 依赖

- 算子源代码（`op_kernel/`, `op_host/`）
- CANN 环境（verify 阶段需要）
- 编译脚本（`build.sh` 或项目等效脚本，verify 阶段需要）

> Continue to the next step in agent workflow
