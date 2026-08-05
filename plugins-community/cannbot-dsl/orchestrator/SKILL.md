---
name: cannbotdsl-op-orchestrator
description: "CANNBotDSL 算子开发的 Primary 编排器（Team 入口）。当用户要求用 CANNBotDSL 从零开发一个指定算子、或走完整设计→实现→测试流程时触发。采用「状态机 + 门禁步骤」混合模式，4 阶段（Stage 0 环境预检 → 1 需求理解 → 2 设计 → 3 实现+验证 → 4 测试+审查；+Perf-Tune 独立）。Primary 只做流程管理和 sub-agent 分派，不写代码。3 态路由 PASS/FAIL/DESIGN_ERROR 支持设计回滚，状态持久化到 .orchestrator_state.json 支持跨会话恢复。Triggers: cannbotdsl 开发算子, 用 CANNBotDSL 写算子, cannbotdsl 算子编排, cannbotdsl 工作流。"
---

# cannbotdsl-op-orchestrator

CANNBotDSL 算子开发 Primary 编排器。**状态机 + 门禁步骤混合模式**，4+1 阶段。

**Primary 只做流程管理、门禁判定、sub-agent 分派、状态持久化，绝不直接写 kernel 代码。**

## Sub-agents

| Agent | 阶段 | 定义文件 |
|-------|------|----------|
| `cannbotdsl-kernel-architect` | Stage 2 设计 | `.opencode/agents/cannbotdsl-kernel-architect.md` |
| `cannbotdsl-kernel-developer` | Stage 3 实现+验证 | `.opencode/agents/cannbotdsl-kernel-developer.md` |
| `cannbotdsl-kernel-tester` | Stage 4 测试+审查 | `.opencode/agents/cannbotdsl-kernel-tester.md` |
| `cannbotdsl-perf-tuner` | Perf-Tune（独立触发） | `.opencode/agents/cannbotdsl-perf-tuner.md` |

Primary 直接调用的 skill：`cannbotdsl-env-setup`（Stage 0）、`cannbotdsl-programming-model` + `cannbotdsl-api-reference`（Stage 1 参考）、`cannbotdsl-op-skill-creator`（用户要求新建算子专用 skill 时触发）。

## 状态机

```
Stage 0 环境预检 (Primary)         gate: 环境就绪 / 降级编译验证
  ↓
Stage 1 需求理解 (Primary)         gate: SPEC.md 用户确认
  ↓
Stage 2 设计 (→ architect)         gate: 预算检查 + 设计自洽 + 用户确认
  ↓
Stage 3 实现+验证 (→ developer)    3 态路由 ↓
  ├─ PASS          → Stage 4
  ├─ FAIL          → developer 修复（编译≤5 轮，精度≤8 轮）
  └─ DESIGN_ERROR  → 回滚 Stage 2（附 design_error_reason）
  ↓
Stage 4 测试+审查 (→ tester)       gate: L0+L1+L2 通过 + 审查通过 → COMPLETION.md

独立: Perf-Tune (→ perf-tuner)     baseline → 瓶颈诊断 → 优化 → 精度重验 → 采纳/回滚
```

## 各阶段职责

### Stage 0 — 环境预检（Primary）
调用 `cannbotdsl-env-setup`。检查 cannbotdsl wheel 安装、`import cannbotdsl` 可达性、ASCEND_HOME_PATH/bisheng、pytest markers、NPU 可用性。
**门禁**：`import cannbotdsl` 必须 PASS；`npu_available` 为 false 时降级为编译验证模式（Stage 3 止步编译，Stage 4 只跑 L0）。输出 `env_check.json`。

### Stage 1 — 需求理解（Primary）
明确数学语义（公式、输入输出、dtype 策略）、目标场景（shape 范围、动态 shape、性能目标）。按算子分类决策树归类（纯 Vec / 纯 Cube / Cube+Vec 混合 / Channel-based / 多 Stage 融合）。归到 **Cube+Vec 混合 / 多 Stage 融合** 时，Stage 2 触发 `cannbotdsl-cv-fusion`。
**算子专用 skill 路由**：检查 `op-skills/` 下是否有匹配当前算子的专用 skill（如 flash-attention）。有则在 SPEC.md 中标注，Stage 3 分派 developer 时在 prompt 中指示其加载该 op-skill。
**门禁**：用户确认 `SPEC.md`。

### Stage 2 — 设计（architect）
分派 `cannbotdsl-kernel-architect`。产出 Tiling 推导、Buffer/Channel 预算（硬限制校验）、Channel depth、Cube/Vec 分工、PIPE 同步方案、VF 融合区域规划、验证策略决策。
**门禁**：所有 mem_loc Total bytes ≤ 硬限制；sync 通道配对自洽；用户确认 `TILING.md` + `DESIGN.md`。
**回滚目标**：Stage 3 返回 `DESIGN_ERROR` 时回到此阶段。

### Stage 3 — 实现+验证（developer）
分派 `cannbotdsl-kernel-developer`。渐进式实现（骨架→搬运→计算→同步→dispatch；「骨架」步用 `cannbotdsl-kernel-structure` 把 stage-graph 落成三层类 + 派发循环）→ 编译验证 → NPU 精度验证。
**算子专用 skill**：若 Stage 1 标注了匹配的 op-skill，在分派 prompt 中指示 developer 加载该 skill（如 `cannbotdsl-flash-attention`），以获取算子专属的 blueprint、buffer budget、已知陷阱等知识。
**3 态输出**：
- `PASS` → Stage 4。
- `FAIL` → 开发者修复。编译错误分类 A（语法）/B（IR）/C（设计→DESIGN_ERROR）/D（bisheng）。精度问题按 7 层定位。
- `DESIGN_ERROR` → 回滚 Stage 2，附 `design_error_reason`。

### Stage 4 — 测试+审查+完成报告（tester）
分派 `cannbotdsl-kernel-tester`。分层测试 L0（编译，覆盖 dtype 组合）→ L1（NPU 精度 vs CPU golden）→ L2（tail block、动态 shape、极端 dtype）→ L3（性能，可选）。代码审查（sync 配对、buf_id 冲突、VF 正确性、Buffer 预算）。生成 `COMPLETION.md`。
**门禁**：L0+L1+L2 全通过 + 审查通过。

### Perf-Tune — 性能调优（perf-tuner，独立触发）
主流程完成后用户要求，或对已有 kernel 触发。4 层优化栈（tiling → 核内流水 → 宏级流水 → 系统级）。必须先采集 baseline，每轮优化后精度重验，采纳/回滚基于 msprof 数据。

## 状态持久化

`.orchestrator_state.json`（跨会话恢复用）：

```json
{
  "op_name": "flash_attention",
  "current_stage": 3,
  "stage_history": [0, 1, 2, 3],
  "rollback_count": 0,
  "artifacts": {
    "env_check": "env_check.json", "spec": "SPEC.md",
    "tiling": "TILING.md", "design": "DESIGN.md",
    "kernel": "kernel.py", "precision_result": "precision_result.json"
  },
  "verification_strategy": "complex",
  "stage3_attempts": {"compile": 2, "precision": 0},
  "degraded_mode": false
}
```

## 数据流（文件传递）

```
Stage 0 → 1: env_check.json
Stage 1 → 2: SPEC.md
Stage 2 → 3: SPEC.md + TILING.md + DESIGN.md（含验证策略）
Stage 3 → 4: kernel.py + precision_result.json
Stage 4 →  : test_{op}.py + review_report.md + COMPLETION.md
回滚 3 → 2: precision_result.json { status: "DESIGN_ERROR", design_error_reason: "..." }
```

## 关键原则

1. Primary 不写代码，只编排。
2. 3 态路由处理设计回滚（CANNBotDSL 编译失败常是设计问题：Buffer 超限、sync 死锁、layout 不兼容）。
3. 渐进式验证：Stage 2 预算检查在写代码前拦截超限；Stage 3 编译验证在 NPU 前拦截编译问题；7 层定位在修复前明确问题层次。
4. 降级模式：无 NPU 时开发机仍可完成设计和编译验证。
5. 本仓算子遵循三件套结构（op.py + test/ + bench/）。
6. **算子专用 skill 路由**：`op-skills/` 下的 skill 为特定算子提供专属知识（blueprint、buffer budget、已知陷阱）。Primary 在 Stage 1 检查是否有匹配的 op-skill，有则在 Stage 3 分派 developer 时指示其加载。当前已注册的 op-skills：

| op-skill | 匹配条件 |
|----------|---------|
| `cannbotdsl-flash-attention` | 算子是 Flash Attention 或其变体（causal/GQA/mxfp8/Channel preload；用户说旧 nbuffer-preload 时也路由至此） |
| `cannbotdsl-mla` | 算子是 MLA / Multi-head Latent Attention（q_nope+q_rope / k_nope+k_rope 分段输入，DeepSeek-V2/V3 注意力） |

新增 op-skill 时在此表追加一行即可被工作流识别。

## 参考

- `../../skills/design/cannbotdsl-workflow-design.md`（完整阶段设计、sub-agent、状态 schema）
- `../../skills/design/cannbotdsl-skills-framework-design.md`（skill 清单与优先级）
- `../../skills/README.md`（框架总览）
