---
name: code-performance-advisor
description: Diagnose AscendC kernel performance bottlenecks from profiling data (msprof). Matches expert rules first, falls back to LLM analysis. Use when speedup is below target after correctness passes.
---

# Code Performance Advisor

将 profiling 数据和算子代码转化为可执行的优化建议。Expert Rule 匹配优先，低置信度时切换 LLM 深度探索。

---

## 快速开始

### 第一次使用（仅需一次）

```bash
cd skills/code-performance-advisor
bash bootstrap.sh   # 构建规则索引
```

### 每次优化前

```bash
# 从 CAKE2/output/{op}/ 初始化 workspace（自动选最新 profiling CSV）
python3 scripts/analysis_engine/init_workspace.py --op <op_name>

# 重新 profiling 或代码有变化时加 --overwrite
python3 scripts/analysis_engine/init_workspace.py --op <op_name> --overwrite
```

初始化后结构：
```
workspace/inputs/{op}/
├── code/op_host/ + op_kernel/
└── profiling/op_summary.csv
```

### 启动工作流

```bash
python3 scripts/analysis_engine/workflow.py run --op <op_name> --mode interactive

# 中断后恢复
python3 scripts/analysis_engine/workflow.py resume --op <op_name>
```

---

## 工作流状态机

```
┌─────────────────────────────────────────────────────────────────────┐
│  INIT     初始化工作空间，提取 profiling 指标                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  TAG      调用 code_tag 子技能，生成特征标签                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  SCORE    规则打分 + 已应用模式过滤（自动）                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  ROUTE    按置信度分路                                               │
│           ├── Fast (≥ 0.7)        → suggest                         │
│           ├── Moderate (0.3-0.7)  → deep_research                   │
│           ├── Deep (< 0.3)        → deep_research                     │
│           └── Scalar-Locked       → algorithm_redesign              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ (各路汇合)
┌──────────────────────────▼──────────────────────────────────────────┐
│  APPLY    按建议修改代码                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  BUILD    重新编译算子                                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  EVALUATE 重新 profiling（msprof）                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  COMPARE  对比基线 vs 新指标                                         │
│           ├── 有提升? → UPDATE（固化规则）                          │
│           └── 无提升? → 可继续下一轮 APPLY 循环                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  DONE     优化会话结束                                               │
└─────────────────────────────────────────────────────────────────────┘
```

```
INIT → TAG → SCORE → ROUTE → SUGGEST → APPLY → BUILD → EVALUATE → COMPARE → UPDATE → DONE
```

| 阶段 | 谁执行 | 说明 |
|------|--------|------|
| INIT | 引擎 | 调用init_workspace.py初始化，验证输入、提取 profiling 指标 |
| TAG | **Agent** | 调用 `code_tag` 子技能生成特征标签 |
| SCORE | 引擎 | 规则打分 + 已应用模式过滤（自动） |
| ROUTE | 引擎 | Fast(≥0.7) / Moderate(0.3-0.7) / Deep(<0.3) |
| SUGGEST | **Agent** | 根据路径调用对应子技能（见下表） |
| APPLY→COMPARE | 引擎+Agent | 改代码 → 编译 → 评测 → 对比 |
| UPDATE | **Agent** | 调用 `rule_update` 固化有效优化 |

**SUGGEST 子技能选择：**

| 路径 | 子技能 | 触发条件 | 说明 |
|------|--------|----------|------|
| Fast | `suggest` | `max_score ≥ 0.7` | 读 top-1 规则，输出代码逐行修改方案 |
| Moderate | `deep_research` | `0.3 ≤ max_score < 0.7` | 规则匹配弱，切换 5步分析法（Bound→Memory→Pipeline→Tiling→Sync），结合规则排名辅助佐证 |
| Deep | `deep_research` | `max_score < 0.3` | 规则无关，从流水线结构诊断（Wait-Wait / Starvation 等模式） |
| Scalar-Locked | `algorithm_redesign` | `aiv_scalar_ratio > 0.35` 且经历过 ≥1 次 APPLY→EVALUATE 循环后 `max_score < 0.40` | 规则已穷举，瓶颈在算法本身的标量依赖链；提出等价但无分支/无 GetValue 的数学替代方案 |

- Continue to the next step in agent workflow

---

## 其他 CLI

```bash
# 查看当前 session 状态
python3 scripts/analysis_engine/workflow.py status --op <op_name>

# 列出历史 sessions
python3 scripts/analysis_engine/workflow.py sessions --op <op_name>

# 清理旧 sessions
python3 scripts/analysis_engine/session_manager.py cleanup
```

---

## 行为约束

- **有据才说**：每条建议必须引用具体指标或代码行，无证据则报 "Insufficient Data"
- **不重复推荐**：系统已过滤代码中已应用的优化模式（COUNTER_MODE / DOUBLE_BUFFER / BUFFER_FUSION / EXPLICIT_SYNC）
- **达标即停**：性能目标达成后立即停止，不额外生成建议
- **验证才算完成**：改动后必须有 profiling 数据证明提升
- **知识闭环**：有效优化必须通过 `rule_update` 固化为规则

---

## 子技能参考

| 子技能 | 文件 | 触发时机 |
|--------|------|----------|
| init_workspace | `subskills/Init_workspace.md` | 每次优化前 |
| code_tag | `subskills/code_tag.md` | TAG 阶段 |
| suggest | `subskills/suggest.md` | SUGGEST / Fast 路径（max_score ≥ 0.7） |
| deep_research | `subskills/deep_research.md` | SUGGEST / Moderate 与 Deep 路径 |
| **algorithm_redesign** | **`subskills/algorithm_redesign.md`** | **SUGGEST / Scalar-Locked 路径：规则穷举后 aiv_scalar_ratio > 35%** |
| rule_update | `subskills/rule_update.md` | UPDATE 阶段 |
| rules_search | `subskills/rules_search.md` | 手动规则查询 |
