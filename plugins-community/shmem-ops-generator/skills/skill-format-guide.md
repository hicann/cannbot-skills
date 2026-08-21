# Skill 格式风格与编写规范

> 本文档总结 SHMEM Agent Skills 的目录结构、frontmatter、章节组织和写作约束，用于后续新增、修改和评审 skills。

---

## 1. 文件与目录结构

### 1.1 命名规则

- **目录名**：一律使用 `kebab-case`（小写字母 + 连字符），如 `shmem-ops-code-gen`、`shmem-ops-compile-debug`
- **主文件**：固定为 `SKILL.md`（全大写），不接受 `skill.md`、`Skill.md` 等变体

### 1.2 目录布局

```
skill-name/
├── SKILL.md                 # 必选 — 技能主定义文件
├── references/              # 可选 — 参考文档（API 规范、硬件架构、模式手册等）
├── scripts/                 # 可选 — 辅助脚本（Python/Shell）
├── templates/               # 可选 — 代码或文档模板
└── examples/                # 可选 — 用例示例、参考实现
```

---

## 2. Frontmatter（YAML 元数据）

每个 SKILL.md **必须**以 YAML frontmatter 开头，包裹在 `---` 之间。

### 2.1 必填字段

```yaml
---
name: skill-name              # kebab-case，与目录名一致
description: "一句话描述技能的功能和触发场景"
---
```

### 2.2 可选字段

```yaml
keywords:                      # 触发关键词列表（支持中英文混合）
  - 关键词1
  - keyword2
metadata:                      # 扩展元信息
  version: "1.0.0"
  author: team-name
  supported-chip: Ascend910B
```

### 2.3 description 的写法

| 要素 | 说明 |
|------|------|
| 能力声明 | 以"做什么"开头：`"XXX 算子端到端开发编排器"` |
| 触发场景 | 说明何时使用：`"当用户需要…时使用"` |
| 关键词枚举 | 尾部列出搜索关键词：`"关键词：算子开发、端到端、全流程"` |

description 可以是单行字符串，也可以使用 YAML `>` 折叠多行：

```yaml
description: >
  多行描述的第一段。
  第二段继续。触发条件和关键词枚举。
```

---

## 3. 文档结构（章节顺序）

SKILL.md 正文按以下顺序组织。并非所有章节都必须出现——根据 skill 复杂度选取适用的章节。

### 3.1 标准章节顺序

```
1. # 标题（H1，全文唯一）
2.    Skill 类型 + 一句话概述
3. ## 核心原则               — 编号列表，3~10 条
4. ## 可用子 Skill 清单       — 表格（仅编排型 skill）
5. ## 工作流总览             — ASCII 流程图
6. ## 反模式清单             — ❌ 开头的禁止项
7. ## Phase N：阶段名称      — 按阶段展开（重复 N 次）
8. ## 阶段间数据流           — 箭头图或表格
9. ## 状态跟踪表             — 总览表格
10. ## 错误恢复               — 中断恢复矩阵
```

### 3.2 各章节模板

#### H1 标题

```markdown
# Ascend Profiling Anomaly Discovery Skill
```

- 全文恰好一个 H1
- 紧跟一行 Skill 类型声明（如有）：`**Skill类型**：流程导向型（Phase 0–7 工作流，含子阶段 5.5 / 6.5，子技能串行编排）`

#### 核心原则

```markdown
## 核心原则

0. **中文写作**：所有交付的 `.md` 文档必须使用中文撰写
1. **阶段串行**：需求确认 → 设计 → 用例生成 → 代码生成 → 编译正确性 → 走读 → Torch 接入（可选）→ 性能采集 → 交付，严格顺序执行
2. **子技能执行**：每个阶段 **MUST** 调用对应子 skill，不得自行实现
```

- 使用**编号列表**
- 每条以**加粗短标题**开头，冒号后跟详细说明
- 用 `MUST`/`NEVER` 标记强制约束

#### 子 Skill 清单（编排型 skill）

```markdown
## 可用子 Skill 清单

| Skill | 路径 | 职责 |
| --- | --- | --- |
| `shmem-ops-design` | `shmem-ops-design/SKILL.md` | 将需求转化为 design.md |
```

#### 工作流总览

```markdown
## 工作流总览

​```
Phase 0       Phase 1       Phase 2         Phase 3        Phase 4            Phase 5       Phase 5.5       Phase 6
需求环境确认 ──▶ 设计文档 ──▶ 用例生成 ──▶ 代码生成 ──▶ 编译+正确性 ──▶ 代码走读 ──▶ Torch 接入 ──▶ 性能采集
—              design     testcase-gen   code-gen    compile-debug    code-review  torch-bind    performance-eval

输入: 算子需求 + 环境            输出: 可交付算子 + Torch 扩展 + 测试 + 性能报告
​```
```

- 使用 ASCII 字符绘制（`──▶`、`→`、`├─`、`└─`）
- 上方标注 Phase 编号，下方标注 skill 名称
- 首尾标注输入/输出

#### 反模式清单

```markdown
## 反模式清单（NEVER DO THESE）

- ❌ 跳过设计阶段直接写代码
- ❌ correctness 失败时做性能优化
- ❌ 编造硬件测试或 profiler 结果
```

- 固定使用 `❌` 前缀
- 每条简洁明确，不加编号

#### Phase 阶段定义

```markdown
## Phase N：阶段名称

**调用 Skill**：`skill-name`（或 `—` 表示无需调用）

### 执行内容

​```
1. 步骤一
2. 步骤二
3. 步骤三
​```

### 检查点

- [ ] 检查项一
- [ ] 检查项二
- [ ] 检查项三

**全部通过 → 进入 Phase N+1**
```

- 每个 Phase 独占一个 H2 章节
- `**调用 Skill**` 明确声明依赖
- 执行内容放在代码块或编号列表中
- **检查点**使用 `- [ ]` 复选框格式
- 末尾用加粗声明跳转条件
- Phase 之间使用 `---` 水平线分隔

#### 阶段间数据流

```markdown
## 阶段间数据流

​```
Phase 0 输出                Phase 1 输入
  op_name、env 确认   ────▶   需求语义、SoC、dtype

Phase 1 输出                Phase 2 输入
  design.md (完整)    ────▶   correctness contract
​```
```

#### 参考文档索引（分析型 skill）

```markdown
## Reference Files — When to Read

| File | When to read | What it contains |
|------|-------------|------------------|
| `references/file.md` | **Always — read first** | 数据格式定义... |
| `references/schema.json` | When producing output | 输出 JSON schema |
```

- 用表格列出每个参考文件
- 标注阅读时机：`Always`（必读）/ 条件触发
- 简述内容用途

---

## 4. 约束语言规范

### 4.1 强度等级

| 标记词 | 强度 | 含义 | 用法示例 |
|--------|------|------|----------|
| **MUST / MANDATORY** | 最强 | 违反即失败 | `**MUST** 调用对应子 skill` |
| **NEVER / FORBIDDEN** | 最强（禁止） | 绝对不允许 | `**NEVER** 跳过设计阶段` |
| **SHOULD / RECOMMENDED** | 推荐 | 除非有充分理由 | `SHOULD 包含中等规模 case` |
| **MAY / OPTIONAL** | 可选 | 读者自行判断 | `MAY 使用高级优化选项` |

### 4.2 格式约定

- 约束词全大写并**加粗**：`**MUST**`、`**NEVER**`
- 与常规描述文字区分，保持视觉醒目
- 在核心原则、执行内容、检查点中均可使用

### 4.3 平台中立（skill 文档 MUST）

skill 与 reference 文档须 **平台/产品中立**，便于在不同 IDE、助手或大模型下复用：

| 禁止写入 | 改用 |
| --- | --- |
| 特定 IDE/助手产品名 | 「对话中 @ 引用本 skill」「编排器」「助手」 |
| 大模型/厂商具体名称 | 「编排器」「子 skill」或不提及 |
| 真实人名、工号、私有容器名 | 占位符：`shmem_test`、`${SHMEM_REPO}`、`/home/<用户名>/` |
| **Skill 目录下的可执行脚本文件**（`.sh`、`.py` 等作为 skill 产物） | 仅用 **Markdown** 代码段；见 [env-setup.snippet.md](shmem-ops-compile-debug/references/env-setup.snippet.md) 等 |
| 引用 **SHMEM 仓原生** 路径 | `` `${SHMEM_REPO}/examples/...` ``、`` `scripts/build.sh` ``、`` `install/set_env.sh` `` |
| 引用 **custom-ops 交付物** 路径 | 允许写 `` `custom-ops/scripts/build.sh` `` 等，但 **MUST** 注明其为 skill 生成、非 upstream；规范以 skill md 为准 |
| 引用不存在的 perf 封装脚本名 | [perf-workflow.md](shmem-ops-performance-eval/references/perf-workflow.md) |
| 产品专有 API 名作为唯一表述 | 并列「结构化 intake 表单（或平台等价提问 UI）」；文件名可保留 `askquestion-template.md` |

### 4.4 仓内 `docs/` 只读（MUST）

- `${SHMEM_REPO}/docs/` 为官方文档，Agent **NEVER** 修改或向其中追加 skill 摘要
- 技能侧补充、索引、排障摘要写在 skill 树内（如 `shmem-repo-docs-index.md`、`log-debug.md`）

### 4.5 Skill 与 SHMEM 仓路径分离（MUST）

Skill **不一定**安装在 SHMEM 仓库内。路径引用分两类：

| 引用目标 | 写法 | 禁止 |
| --- | --- | --- |
| Skill 树内文档 | 相对路径；**仅 `.md` 文件** | 在 `.agents/skills/` 下新增 `.sh`/`.py` 等 skill 附属脚本 |
| **SHMEM 仓原生** | `` `${SHMEM_REPO}/docs/` ``、`` `examples/` ``、`` `scripts/build.sh` ``、`` `install/set_env.sh` `` | 当作 custom-ops 交付物 |
| **custom-ops 交付物**（skill 生成） | 规范见 [custom-ops-entrypoints.md](shmem-ops-compile-debug/references/custom-ops-entrypoints.md) 等 md 代码段；磁盘路径 `` `custom-ops/...` `` 可 Read 核对 | 当作 upstream 官方目录；裸克隆未必存在 |
| 需在 skill 内展开的脚本逻辑 | Markdown 代码段（如 env-setup.snippet.md） | 在 skill 目录放置可 `source` 的 `.sh` 文件 |
| 散文 `examples/...` | 视为 `${SHMEM_REPO}/examples/...` | 跳过 §2 定位 |

规范全文：[shmem-repo-resolution.md](shmem-ops-dev/references/shmem-repo-resolution.md)。仓内文档索引：[shmem-repo-docs-index.md](shmem-ops-compile-debug/references/shmem-repo-docs-index.md)。

---

## 5. 排版与格式规范

### 5.1 标题层级

| 层级 | 用途 | 示例 |
|------|------|------|
| `#` H1 | 全文标题（唯一） | `# SHMEM 算子端到端开发编排` |
| `##` H2 | 主要章节 | `## Phase 1：设计文档` |
| `###` H3 | 子章节 | `### 执行内容` / `### 检查点` |
| `####` H4 | 细分条目 | `#### CANN 环境` |

### 5.2 强调标记

| 格式 | 用途 | 示例 |
|------|------|------|
| `**加粗**` | 关键规则、约束词、重要概念 | `**MUST** 调用子 skill` |
| `` `行内代码` `` | 文件名、命令、变量名、skill 名 | `` `shmem-ops-design` `` |
| `❌` | 反模式条目前缀 | `❌ 跳过设计阶段` |
| `✅` | 校验通过标记 | `✅ 校验点：输入完整` |
| `📝` | 反馈/备注标记 | `📝 反馈：信息收集完成` |
| `- [ ]` | 检查点复选框 | `- [ ] 编译成功` |

### 5.3 表格

表格是结构化信息的主要载体，常见类型：

- **子 Skill 清单表**：`| Skill | 路径 | 职责 |`
- **决策路由表**：`| 条件 | 判定 | 动作 |`
- **参考文件表**：`| 文件 | 阅读时机 | 内容 |`
- **约束表**：`| 约束项 | 值 | 说明 |`
- **快速参考表**：`| 场景 | 方法 | 适用条件 |`

表格要求：
- 表头简洁（2~4 列为宜）
- 单元格内容 <100 字符
- 相关行紧邻排列

### 5.4 代码块

```markdown
​```bash
# Shell 命令
# 见 custom-ops-entrypoints.md / perf-workflow.md
​```

​```python
# Python 脚本示例
import numpy as np
​```

​```
Phase 0 ──▶ Phase 1 ──▶ Phase 2
（无语言标记的代码块用于 ASCII 流程图）
​```
```

- 命令用 `bash` 标记
- 脚本用对应语言标记（`python`/`cpp`）
- ASCII 流程图不加语言标记

### 5.5 分隔线

- Phase 之间使用 `---` 分隔，保持视觉边界清晰
- 大章节跳转前可加 `---`

---

## 6. 语言与术语规范

### 6.1 中英文混合规则

| 内容类别 | 语言 | 示例 |
|----------|------|------|
| 章节标题、描述、分析、结论 | 中文 | `## 核心原则` |
| API 名称、代码片段、变量名 | 英文原文 | `aclshmem_putmem`、`design.md` |
| 数学公式 | 英文/LaTeX | `utilization = actual / peak` |
| MUST/NEVER 等约束词 | 英文大写 | `**MUST** 调用` |
| Skill 名称 | 英文原文 + 行内代码 | `` `shmem-ops-design` `` |

### 6.2 占位符

占位符用于表示需替换的值，常见格式：

- `<op_name>` — 尖括号包裹（最常用）
- `{{OP_NAME}}` — 双花括号
- `[op_name]` — 方括号

同一个 skill 内保持风格统一。

---

## 7. Skill 类型差异

不同类型的 skill 在结构上有差异，但共享上述基础格式。

### 7.1 编排型（Orchestrator）

多阶段工作流编排，串行调用子 skill。

**典型代表**：`ascendc-operator-dev`、`triton-operator-dev`、`shmem-ops-dev`

**必含章节**：核心原则 → 子 Skill 清单表 → 工作流总览图 → 反模式清单 → Phase 0~N → 数据流 → 状态跟踪表 → 错误恢复矩阵

**特征**：
- 每个 Phase 声明 `**调用 Skill**`
- 门控检查点控制阶段流转
- 完整的中断恢复矩阵

### 7.2 执行型（Executor）

单一职责的子 skill，由编排器调用或独立使用。

**典型代表**：`shmem-ops-code-gen`、`shmem-ops-compile-debug`、`shmem-ops-correctness-eval`

**必含章节**：概述 → 参考文件表 → 执行流程/步骤 → 校验点 → 输出物

**特征**：
- 开头列出必读参考文件
- 步骤带校验点和中断条件
- 明确输出产物和格式

### 7.3 分析型（Analyzer）

接收数据，经过多管线分析后输出报告。

**典型代表**：`ascend-profiling-anomaly`、`megatron-change-analyzer`

**必含章节**：Purpose → Reference Files 表 → Pipeline Overview → 数据层级 → 决策规则 → 优雅降级

**特征**：
- 状态机图描述处理管线
- 分层数据模型（如 Step → Structure → Block → Op）
- 优雅降级表（缺失数据时如何继续）

### 7.4 工具型（Utility）

提供特定环境或工具的操作指南。

**典型代表**：`ascend-docker`、`npu-smi`、`hccl-test`

**必含章节**：Quick Start → 参数说明 → 使用模式 → 示例

**特征**：
- 以代码示例为主
- Quick Start 紧跟标题
- 参数用表格列出

### 7.5 审计型（Auditor）

对代码或 skill 进行安全/质量审查。

**典型代表**：`skill-auditor`、`npu-adapter-reviewer`

**必含章节**：When to Use → 审计协议步骤 → 风险矩阵 → 判定结果模板

**特征**：
- 结构化审计步骤（Step 1~N）
- 风险等级分类表
- 最终判定（SAFE / SUSPICIOUS / DANGEROUS / BLOCK）

---

## 8. 编写检查清单

编写或审查 SKILL.md 时，逐项确认：

### 结构完整性

- [ ] YAML frontmatter 包含 `name` 和 `description`
- [ ] `name` 为 kebab-case，与目录名一致
- [ ] 全文恰好一个 H1 标题
- [ ] 章节顺序符合第 3 节规范

### 内容质量

- [ ] 核心原则完整，涵盖该 skill 的关键约束
- [ ] 反模式清单列出常见错误
- [ ] 每个阶段/步骤有明确的检查点
- [ ] 约束使用 MUST/NEVER/SHOULD 分级标记
- [ ] 参考文件表列出阅读时机和内容摘要

### 格式一致性

- [ ] 表格列数 2~4 列，单元格简洁
- [ ] 代码块标注语言
- [ ] Skill 名称用行内代码包裹
- [ ] Phase 之间有 `---` 分隔线
- [ ] 检查点使用 `- [ ]` 格式
- [ ] 反模式使用 `❌` 前缀
- [ ] 占位符格式全文统一
