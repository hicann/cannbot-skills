---
name: spec-to-design
description: "从算子仓 operators/{operator}/docs/spec.yaml 生成或更新中文 DESIGN.md 和 PLAN.md 的方案设计技能。当用户要求根据 spec.yaml 生成设计文档、方案设计、迭代计划、spec-to-design、更新 DESIGN.md 或执行 ops-registry-invoke 的 1.3 方案设计时触发。"
---

# Spec To Design

## 目标

根据 `operators/{operator_name}/docs/spec.yaml`、`REQUIREMENTS.md` 和本技能模板生成简体中文产物：

- `operators/{operator_name}/docs/DESIGN.md`
- `operators/{operator_name}/docs/PLAN.md`

代码标识、API 名称、dtype、文件名和 YAML key 可保留英文；用户可见说明、章节内容、结论和风险说明必须使用简体中文。

## 路径解析

优先用环境变量定位本技能：

```bash
SPEC_TO_DESIGN_SKILL_DIR="${SPEC_TO_DESIGN_SKILL_DIR:-}"
if [ -z "$SPEC_TO_DESIGN_SKILL_DIR" ]; then
  for p in .opencode/skills/spec-to-design .claude/skills/spec-to-design; do
    [ -d "$p" ] && SPEC_TO_DESIGN_SKILL_DIR="$p" && break
  done
fi
```

后续命令均使用 `$SPEC_TO_DESIGN_SKILL_DIR`。如果变量为空，先定位已安装的 `spec-to-design` skill，不要硬编码 `.claude/skills/...` 或 `.opencode/skills/...`。

## 工作流程

> **执行者分工**（ops-registry-invoke 1.3 流程中）：步骤 1-3、6-8 由主 Agent 直接执行（脚本均为确定性命令）；步骤 4-5 由 5 个 `ascendc-ops-design-*` 分段 Agent 并行完成（每个 bundle 一个 Agent，定义见 `agents/` 目录，必须在同一次响应中同时发起）。独立使用本 skill（不在 1.3 编排内）时，单个 agent 可按本流程串行自做全部步骤。

1. 解析输入：
   - 若用户给出 `spec.yaml` 路径，直接使用。
   - 否则使用 `operators/{operator_name}/docs/spec.yaml`。
   - 同目录读取 `REQUIREMENTS.md`，输出到同目录的 `DESIGN.md` 和 `PLAN.md`。
2. 读取 `references/section-map.md`，确认分段生成边界。
3. 创建设计输入包：
   ```bash
   python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/slice_design_inputs.py" \
     operators/{operator_name}/docs/spec.yaml \
     "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
     operators/{operator_name}/.spec-to-design \
     --requirements operators/{operator_name}/docs/REQUIREMENTS.md \
     --plan-template "$SPEC_TO_DESIGN_SKILL_DIR/templates/PLAN.md.templ" \
     --force
   ```
4. 生成 `operators/{operator_name}/.spec-to-design/bundles/*.md` 对应的 markdown：
   - 1.3 编排内：5 个 bundle 由对应 `ascendc-ops-design-*` Agent 并行生成（分段-Agent 映射见 `references/section-map.md`），各 Agent 额外读取 `DESIGN_PREP.md` 承接路线/API 验证结论。
   - `05-plan.md` 生成完整 `PLAN.md`，不是 `DESIGN.md` 章节。
   - 每个分段只能使用包内 `spec.yaml` 切片、`REQUIREMENTS.md` 摘要和模板摘录；禁止从 sibling spec 或历史设计文档补事实。
5. 将设计章节保存到 `operators/{operator_name}/.spec-to-design/sections/`；将迭代计划保存为 `operators/{operator_name}/.spec-to-design/sections/05-plan.md`。
6. 组装文档：
   ```bash
   python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/assemble_design.py" \
     --spec operators/{operator_name}/docs/spec.yaml \
     --template "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
     --sections operators/{operator_name}/.spec-to-design/sections \
     --output operators/{operator_name}/docs/DESIGN.md \
     --plan-output operators/{operator_name}/docs/PLAN.md \
     --plan-template "$SPEC_TO_DESIGN_SKILL_DIR/templates/PLAN.md.templ"
   ```
7. 校验结构与完整性：
   ```bash
   python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/validate_design.py" \
     --spec operators/{operator_name}/docs/spec.yaml \
     --template "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
     --design operators/{operator_name}/docs/DESIGN.md \
     --plan operators/{operator_name}/docs/PLAN.md

   python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/validate_completeness.py" \
     --spec operators/{operator_name}/docs/spec.yaml \
     --template "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
     --design operators/{operator_name}/docs/DESIGN.md \
     --plan operators/{operator_name}/docs/PLAN.md
   ```
   日志自动写入 `operators/{operator_name}/docs/.validate_completeness.log`（追加模式，含时间戳与 DEBUG 级 gate 加载/校验细节）。校验失败时，**先读该日志**定位触发错误的 paradigm gate（如 `validate_gates_broadcast`），再回到对应分段修订。
   可通过 `--log-file <path>` 覆盖路径，或 `--log-file -` / `VALIDATE_LOG_FILE=-` 禁用文件输出。
8. 若校验失败，修复对应分段后重新组装和校验，直到通过（1.3 编排内：只重发失败分段对应的 `ascendc-ops-design-*` Agent，每个分段最多重炉 2 次）。

## 设计规则

- `spec.yaml` 是 dtype、shape、broadcast、formula/oracle、boundary、extreme、tolerance、determinism、reduction、paradigm_groups 的唯一真值源。
- `REQUIREMENTS.md` 只承接需求背景、运行环境、ACLNN 接口自然语言说明、性能目标和资源约束。
- `DESIGN.md` 必须包含「spec.yaml 一致性映射」，逐项说明 spec 字段在设计中的承接位置。
- `category: Broadcast` 表示算子计算需要对数据进行广播（数据复制/扩展），如 add、mul、where；纯 shape 重组织（expand、broadcast_to、tile）归入 `category: LayoutTransform`。设计时需区分"计算含广播"与"仅做广播"两种场景。
- API 使用必须先查可信来源：CANN 官方文档、`reference/cann/asc-devkit/docs/api/context/`、CANN 安装路径或用户明确提供的资料。未验证 API 只能标为“待验证”，不能写成已支持。
- `DESIGN.md` 必须包含 Tiling 策略、Kernel 模板划分、数据类型支持方案、API 映射、数据流设计、内存管理、API 验证记录、UB 容量验证和风险评估。
- `PLAN.md` 必须包含 YAML frontmatter 和对应的 Markdown 正文。frontmatter schema（字段结构、取值约束）和正文生成规则（`<!-- BEGIN/END -->` 标记）均以 `templates/PLAN.md.templ` 为唯一权威源，禁止在此重复定义。
- 对不确定或 spec 缺失的信息，明确写“待补充/需回到 spec-generation 修订”，不要编造。

## 资源

- `references/section-map.md`：设计分段、并行边界和合并契约。
- `templates/DESIGN.md.templ`：中文算子仓详细设计模板。
- `templates/PLAN.md.templ`：中文迭代执行计划模板。
- `scripts/slice_design_inputs.py`：生成分段输入包。
- `scripts/assemble_design.py`：按模板顺序组装 `DESIGN.md`，并写出 `PLAN.md`。
- `scripts/validate_design.py`：校验结构、章节顺序和关键文档约束。
- `scripts/validate_completeness.py`：校验核心内容是否足够完整。
