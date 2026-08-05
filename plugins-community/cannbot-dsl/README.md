# CANNBot DSL Skills

CANNBotDSL 算子开发技能框架，为 opencode 设计。基于 cannbot-dsl 仓（wheel 安装）的实际结构定制。

## 目录结构

```
skills/
├── install.sh               # 安装器：把 skills/agents 挂到 .opencode/
├── README.md                # 本文件
├── core-skills/             # 核心/基础/开发/测试/优化/工具 skill（14 个）
│   ├── cannbotdsl-env-setup/
│   ├── cannbotdsl-api-reference/
│   ├── cannbotdsl-programming-model/
│   ├── cannbotdsl-op-design/
│   ├── cannbotdsl-op-develop/
│   ├── cannbotdsl-probe-debug/
│   ├── cannbotdsl-op-test/
│   ├── cannbotdsl-tiling-design/
│   ├── cannbotdsl-vf-fusion/
│   ├── cannbotdsl-cv-fusion/
│   ├── cannbotdsl-kernel-structure/
│   ├── cannbotdsl-code-review/
│   └── cannbotdsl-perf-optimize/
├── debug-skills/            # 调试/诊断 skill（5 个）
│   ├── cannbotdsl-precision-debug/
│   ├── cannbotdsl-runtime-debug/
│   ├── cannbotdsl-crash-debug/
│   ├── cannbotdsl-msprof-compare/
│   └── cannbotdsl-npu-plog-diagnosis/
├── op-skills/               # 算子专用 skill（可扩展）
│   ├── cannbotdsl-flash-attention/
│   └── cannbotdsl-mla/
├── orchestrator/            # 工作流编排入口
│   └── SKILL.md             # cannbotdsl-op-orchestrator
└── agents/                  # Sub-agent 定义（4 个）
    ├── cannbotdsl-kernel-architect.md
    ├── cannbotdsl-kernel-developer.md
    ├── cannbotdsl-kernel-tester.md
    └── cannbotdsl-perf-tuner.md
```

每个 skill 是一个目录，包含 `SKILL.md`（frontmatter + 指令体）和可选的 `references/` 子目录（详细知识文档，按需加载）。

## 安装

这些 skills 不是独立可移植的：它们假设 opencode 以 **cannbot-dsl 仓库根目录为工作目录** 运行。安装只是把它们挂进 opencode 的发现目录 `.opencode/skills/` 和 `.opencode/agents/`。`.opencode/` 已被 `.gitignore` 忽略，所以每份 checkout 需各自跑一次安装器。

```bash
# 在仓库根目录执行
./skills/install.sh              # 默认 symlink：改 skills/ 源即时生效
./skills/install.sh --copy       # 拷贝而非软链（不支持 symlink 的文件系统）
./skills/install.sh --uninstall  # 卸载（只删安装器创建的项，用户自己的 skill 不动）
```

安装器会：
- 把 `core-skills/cannbotdsl-*/`、`debug-skills/cannbotdsl-*/`、`op-skills/cannbotdsl-*/` 挂为 `.opencode/skills/cannbotdsl-*/`
- 把 `orchestrator/SKILL.md` 挂为 `.opencode/skills/cannbotdsl-op-orchestrator/SKILL.md`
- 把 `agents/cannbotdsl-*.md` 挂为 `.opencode/agents/cannbotdsl-*.md`

幂等可重复执行；卸载安全（仅移除带 `.cannbotdsl-managed` 标记或本安装器创建的软链，保留用户自建 skill）。

安装后**重启 opencode** 让其重新扫描 `.opencode/`。

## 使用

前提：opencode 运行于 cannbot-dsl 仓库内，cannbotdsl wheel 已安装（`pip install cannbotdsl-*-cp312-cp312-manylinux_*_x86_64.whl`），CANN 环境变量已 source（`ASCEND_HOME_PATH` 已设）。无 NPU 时仍可用 translate-only 模式做代码生成验证。

**两种用法：**

1. **单点触发 skill（按需）** — 描述你的任务，opencode 依据每个 SKILL.md 的 `description` 触发词自动加载相应 skill。例：
   - "用 cannbotdsl 写一个 xxx 算子" → `cannbotdsl-op-develop`
   - "这个 kernel 精度对不上，帮我定位" → `cannbotdsl-precision-debug`
   - "跑 msprof 对比两版性能" → `cannbotdsl-msprof-compare`

2. **全流程编排（从零写一个指定算子）** — 触发 `cannbotdsl-op-orchestrator`（说"用 CANNBotDSL 从零开发 xxx 算子，走完整流程"）。它作为 Primary 编排器按 4+1 阶段状态机推进，逐阶段分派 4 个 sub-agent，带 3 态路由（PASS/FAIL/DESIGN_ERROR）和用户门禁确认：

   ```
   Stage 0 环境预检 → 1 需求(SPEC.md) → 2 设计(architect→TILING/DESIGN.md)
   → 3 实现+验证(developer, 渐进式) → 4 测试+审查(tester→COMPLETION.md)
   独立: Perf-Tune(perf-tuner, 可选)
   ```

## Skills 总览

### core-skills（14 个）

| 层级 | Skill | 说明 |
|------|-------|------|
| **基础** | `cannbotdsl-env-setup` | 环境搭建与诊断（wheel 安装、NPU 可用性） |
| **基础** | `cannbotdsl-api-reference` | API 结构化查询入口 |
| **基础** | `cannbotdsl-programming-model` | `@jit`/`@kernel` 编程模型、AST 预处理、Buffer 分配 |
| **设计** | `cannbotdsl-op-design` | 算子设计（分类路由、Buffer 预算、Tiling、流水编排） |
| **设计** | `cannbotdsl-cv-fusion` | Cube+Vec 融合架构（跨核同步、sync 预算） |
| **设计** | `cannbotdsl-tiling-design` | 多级 tiling 策略（tile_view、tail block、layout 代数） |
| **开发** | `cannbotdsl-op-develop` | 算子开发（三件套结构、渐进式实现、质量门禁） |
| **开发** | `cannbotdsl-vf-fusion` | VF 向量折叠（3 种模式、铁律、陷阱） |
| **开发** | `cannbotdsl-kernel-structure` | 三层职责分离编码规范 |
| **开发** | `cannbotdsl-probe-debug` | 静默算错的探针定位法（最小探针、参数扫描、误差形态） |
| **测试** | `cannbotdsl-op-test` | 分层测试（L0-L3）、CPU golden 规范 |
| **测试** | `cannbotdsl-code-review` | 代码审查（Channel sync、VF、Buffer/Channel 预算） |
| **优化** | `cannbotdsl-perf-optimize` | 4 层优化栈（tiling → 核内流水 → 宏级流水 → 系统级） |
| **工具** | `cannbotdsl-op-skill-creator` | 创建新 op-skill 并自动注册到工作流 |

### debug-skills（5 个）

| Skill | 说明 |
|-------|------|
| `cannbotdsl-precision-debug` | NPU 精度定位（7 层定位法） |
| `cannbotdsl-runtime-debug` | 运行时错误分类（A/B/C/D 型） |
| `cannbotdsl-crash-debug` | crash/hang/sync 死锁诊断 |
| `cannbotdsl-msprof-compare` | msprof 性能采集与对比 |
| `cannbotdsl-npu-plog-diagnosis` | 设备 plog 日志解析 |

### op-skills（2 个，可扩展）

| Skill | 说明 |
|-------|------|
| `cannbotdsl-flash-attention` | Flash Attention 专用设计（blueprint、buffer budget、mxfp8） |
| `cannbotdsl-mla` | MLA 专用设计（归约轴 concat 分解、两向 chunk、head folding 47.8x） |

新增算子专用 skill 时，在 `op-skills/<op-name>/` 下创建 `SKILL.md` + `references/`。

### orchestrator（1 个）

| Skill | 说明 |
|-------|------|
| `cannbotdsl-op-orchestrator` | 算子开发 Primary 编排器（4+1 阶段状态机） |

## Agents

| Agent | 职责 | 绑定 Skills |
|-------|------|------------|
| `cannbotdsl-kernel-architect` | 算子设计（Stage 2） | 6 个 |
| `cannbotdsl-kernel-developer` | 算子开发（Stage 3） | 7 个 |
| `cannbotdsl-kernel-tester` | 测试+审查（Stage 4） | 3 个 |
| `cannbotdsl-perf-tuner` | 性能调优（独立） | 2 个 |

所有 agent 均为 `mode: subagent`，由 orchestrator 通过 Task 工具按名字分派。

## 新增 op-skill

`op-skills/` 下的 skill 为特定算子提供专属知识（blueprint、buffer budget、已知陷阱）。新增一个算子专用 skill **推荐使用 `cannbotdsl-op-skill-creator` skill 自动完成**，无需手动改 5 个文件。


### 用法

在 opencode 中说：

> 给 layer-norm 算子新建一个 op-skill

`cannbotdsl-op-skill-creator` 会自动：

1. 在 `skills/op-skills/cannbotdsl-<op>/` 下创建 `SKILL.md` + `references/` 骨架
2. 注册到 `orchestrator/SKILL.md` 的 op-skill 路由表
3. 加入 `agents/cannbotdsl-kernel-developer.md` 的绑定列表
4. 在 `core-skills/cannbotdsl-op-develop/SKILL.md` 路由表追加路由行
5. 更新本 README 的 op-skills 表格

创建完成后运行验证：

```bash
./skills/install.sh                    # 应显示新 skill 被安装
```

### 手动创建（如需）

如不使用 creator skill，需手动完成上述 5 步，详见 `core-skills/cannbotdsl-op-skill-creator/SKILL.md` 中的步骤说明。`install.sh`、`opencode.json` 无需手动改动。

### 命名约束

- 目录名必须以 `cannbotdsl-` 开头（install.sh 的 glob 是 `cannbotdsl-*/`）
- `SKILL.md` frontmatter 的 `name` 字段必须与目录名一致
- `description` 应前置触发关键词，写清"做什么 + 何时触发 + 何时跳过"
