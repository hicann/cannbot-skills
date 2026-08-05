---
name: cannbotdsl-op-skill-creator
description: "为 CANNBotDSL 算子创建新的 op-skill 并注册到工作流时使用。当用户说'新建一个 op-skill''给 xxx 算子加个专用 skill'时触发。自动完成：创建 SKILL.md + references/ 骨架、注册到 orchestrator 的 op-skill 路由表、加入 developer agent 绑定列表、在 op-develop 路由表追加路由行、更新 README。Triggers: 新建 op-skill, 创建算子专用 skill, op-skill creator, add op-skill。"
---

# cannbotdsl-op-skill-creator

为特定算子创建 op-skill 并自动注册到工作流。一个 op-skill 为某类算子提供专属知识（blueprint、buffer budget、已知陷阱、代码片段），让 developer sub-agent 在开发该算子时能加载这些知识。

## 触发条件

- 用户要求为某个算子创建专用 skill
- 需要新增一个 op-skill 并确保工作流能自动加载它

## 前置确认

创建前必须向用户确认以下信息：

| 信息 | 说明 | 示例 |
|------|------|------|
| **算子名** | 用于目录名和 skill name，须以 `cannbotdsl-` 开头 | `cannbotdsl-layer-norm` |
| **算子描述** | 一句话说明算子做什么 | `Layer Normalization，沿指定轴做均值方差归一化` |
| **触发条件** | 何时应触发此 skill（关键词/场景） | `layer norm, layernorm, RMS norm, 归一化算子` |
| **匹配条件** | orchestrator 路由表中的匹配判据 | `算子是 LayerNorm / RMSNorm 或其变体` |
| **参考算子** | 仓内已有的同类算子实现（如有） | — |

## 创建步骤

严格按以下顺序执行，每步完成后向用户报告。

### 步骤 1：创建 skill 目录和 SKILL.md

在 `skills/op-skills/<算子名>/` 下创建 `SKILL.md`：

```
skills/op-skills/cannbotdsl-<op-name>/
├── SKILL.md
└── references/          # 可选，按需创建
```

**SKILL.md 模板**（替换 `<...>` 占位符）：

```markdown
---
name: cannbotdsl-<op-name>
description: "<算子描述>。用户要求写新的 <算子名> kernel、在变体之间移植特性、调整 tile 形状或 buffer 预算、修精度问题时触发此 skill。Triggers: <触发关键词>。非 <算子名> 类算子跳过。"
---

# cannbotdsl-<op-name>

<算子名> 专用设计与开发指南。

## 算子语义

<数学公式、输入输出、dtype 策略>

## Tile 与 Buffer 预算

<推荐 tile 形状、各级 buffer 容量核算>

## 已知陷阱

<该算子特有的 API 陷阱、精度风险、同步注意事项>

## 代码骨架

<参考同类算子的代码结构>

## 参考

- `../../core-skills/cannbotdsl-op-design/SKILL.md`（通用设计流程）
- `../../core-skills/cannbotdsl-op-develop/SKILL.md`（通用开发流程）
```

**命名约束**：
- 目录名必须以 `cannbotdsl-` 开头（install.sh 的 glob 是 `cannbotdsl-*/`）
- `name` 字段必须与目录名一致
- `description` 应前置触发关键词，写清"做什么 + 何时触发 + 何时跳过"

如果算子有复杂的专属知识（blueprint、buffer budget 表、mxfp8 规则等），在 `references/` 下创建独立 `.md` 文件，SKILL.md 中按需引用。

### 步骤 2：注册到 orchestrator

编辑 `skills/orchestrator/SKILL.md`，在关键原则 §6 的 op-skill 注册表追加一行：

```markdown
| `cannbotdsl-<op-name>` | <匹配条件> |
```

### 步骤 3：加入 developer agent 绑定列表

编辑 `skills/agents/cannbotdsl-kernel-developer.md`，在"绑定 Skills"的"算子专用 skill（按需加载）"下追加一行：

```markdown
  - `cannbotdsl-<op-name>`（<算子描述>）
```

### 步骤 4：在 op-develop 路由表追加路由行

编辑 `skills/core-skills/cannbotdsl-op-develop/SKILL.md`，在快速路由表中追加一行（插在"写新 DSL 算子或示例"行之后）：

```markdown
| **开发 <算子名> 类算子** | `../../op-skills/cannbotdsl-<op-name>/SKILL.md`（<专属知识摘要>） |
```

### 步骤 5：更新 README

编辑 `skills/README.md`，在 op-skills 表格追加一行：

```markdown
| `cannbotdsl-<op-name>` | <一句话说明> |
```

## 验证

全部步骤完成后，运行以下命令验证：

```bash
# 1. 重装 skills（新 skill 应被识别）
./skills/install.sh

# 2. 确认新 skill 出现在安装列表中
ls .opencode/skills/cannbotdsl-<op-name>/SKILL.md
```

## 门禁

- 5 个步骤全部完成才算创建成功，缺任何一步工作流都无法正确加载新 skill。
- `name` 字段必须与目录名一致，以 `cannbotdsl-` 开头。
- `install.sh`、`opencode.json` 无需手动改动（自动覆盖）。
- 创建完成后提示用户重启 opencode 以加载新 skill。
