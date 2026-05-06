# NPU 模型推理优化快速入门指南

## 概述

结合 CANN 平台原子化优化特性与 cann-recipes-infer 仓库的模型优化实践，提供 NPU 模型推理端到端优化能力。通过多 Agent 协同（分析→实施→验证）按阶段对模型执行优化，覆盖并行策略、KVCache/FA、融合算子、图模式、多流并行等优化路径。

## 一、环境搭建

### 操作步骤

#### 方式一：项目级安装（推荐）

在项目目录下安装，配置仅对当前项目生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/cannbot-skills.git

# 2. 进入推理优化 team 目录
cd skills/model/teams/infer-model-optimize-team

# 3. 执行初始化脚本（项目级）
bash init.sh project opencode   # OpenCode 用户
bash init.sh project claude     # Claude Code 用户
```

#### 方式二：全局安装

在用户目录下安装，配置全局生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/cannbot-skills.git

# 2. 进入推理优化 team 目录
cd skills/model/teams/infer-model-optimize-team

# 3. 执行初始化脚本（全局）
bash init.sh global opencode    # OpenCode 用户
bash init.sh global claude      # Claude Code 用户
```

### 安装内容

init.sh 脚本会完成以下操作：

| 内容 | OpenCode 项目级 | OpenCode 全局 | Claude 项目级 | Claude 全局 |
|------|----------------|---------------|---------------|-------------|
| Skills 技能模块 | `.opencode/skills/` | `~/.config/opencode/skills/` | `.claude/skills/` | `~/.claude/skills/` |
| Agents 子代理 | `.opencode/agents/` | `~/.config/opencode/agents/` | `.claude/agents/` | `~/.claude/agents/` |
| Hooks 约束脚本 | — | — | `.claude/hooks/` | `~/.claude/hooks/` |
| 配置文件 | `.opencode/AGENTS.md` | `~/.config/opencode/AGENTS.md` | `.claude/CLAUDE.md` + `settings.json` | `~/.claude/CLAUDE.md` + `settings.json` |
| 参考仓库 | 当前目录 `cann-recipes-infer/` | 当前目录 `cann-recipes-infer/` | 当前目录 `cann-recipes-infer/` | 当前目录 `cann-recipes-infer/` |

### 环境校验

执行完上述步骤后，检查目录结构是否符合以下规范：

**项目级安装**：
```
skills/model/teams/infer-model-optimize-team/
├── .opencode/
│   ├── skills/
│   │   ├── model-infer-optimize/
│   │   ├── model-infer-migrator/
│   │   ├── model-infer-parallel-analysis/
│   │   ├── model-infer-parallel-impl/
│   │   ├── model-infer-kvcache/
│   │   ├── model-infer-fusion/
│   │   ├── model-infer-graph-mode/
│   │   ├── model-infer-precision-debug/
│   │   ├── model-infer-runtime-debug/
│   │   ├── model-infer-multi-stream/
│   │   ├── model-infer-prefetch/
│   │   └── model-infer-superkernel/
│   ├── agents/
│   │   ├── model-infer-analyzer.md
│   │   ├── model-infer-implementer.md
│   │   └── model-infer-reviewer.md
│   └── AGENTS.md               # OpenCode；Claude 为 CLAUDE.md + settings.json + hooks/
├── cann-recipes-infer/         # 参考仓库
├── AGENTS.md
├── init.sh
└── quickstart.md
```

## 二、快速上手

### 启动

在初始化完成的目录下执行：

```bash
opencode
```

### 优化模型示例

在交互界面中输入优化需求，会自动匹配对应技能：

```
帮我优化 deepseek-r1 模型的推理性能
```

或直接调用编排入口：

```
/model-infer-optimize
```

### 优化流程

采用分阶段优化，每阶段验证通过后进入下一阶段：

```
阶段 0：模型分析 + 性能基线
    ↓
阶段 1：并行化改造
    ↓
阶段 2：KVCache 静态化 + FA 算子替换
    ↓
阶段 3：融合算子优化
    ↓
阶段 4：图模式适配
    ↓
阶段 5：优化总结
```

每个阶段遵循统一流程：分析 → 方案确认 → 实施 → 验证 → 阶段总结。

### 多 Agent 协同

使用三个专业化 subagent 分工执行：

| Agent | 职责 |
|-------|------|
| model-infer-analyzer | 模型分析、方案设计、策略推荐 |
| model-infer-implementer | 代码改造、调试修复 |
| model-infer-reviewer | 精度验证、性能验证 |

主 Agent 负责编排调度，按阶段派发 subagent 执行。

## 三、可用技能

| Skill | 用途 |
|-------|------|
| `model-infer-optimize` | 端到端优化编排入口 |
| `model-infer-migrator` | 框架适配与基线建立 |
| `model-infer-parallel-analysis` | 并行策略分析 |
| `model-infer-parallel-impl` | 并行切分实施 |
| `model-infer-kvcache` | KVCache + FA 优化 |
| `model-infer-fusion` | 融合算子分析与替换 |
| `model-infer-graph-mode` | 图模式适配 |
| `model-infer-precision-debug` | NPU 推理精度诊断 |
| `model-infer-runtime-debug` | NPU 运行时错误诊断 |
| `model-infer-multi-stream` | 多流并行优化 |
| `model-infer-prefetch` | 权重预取 |
| `model-infer-superkernel` | SuperKernel 适配 |

## 四、开发资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| 模型实现参考 | `cann-recipes-infer/models/` | 各模型的推理实现 |
| 执行框架 | `cann-recipes-infer/executor/` | ModelRunner、模型加载 |
| 共享模块 | `cann-recipes-infer/module/` | Linear、MoE GMM、量化 |
| 模型文档 | `cann-recipes-infer/docs/models/` | 各模型优化指南 |

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合在 team 目录下工作，参考仓库路径自动匹配
- **全局**：技能全局可用，但参考仓库路径需手动定位

### Q: 如何更新技能模块？

重新执行 init.sh 即可，脚本会自动覆盖旧版本。参考仓库会 git pull 更新。

### Q: 可以在 cann-recipes-infer 仓库内直接使用吗？

可以。cann-recipes-infer 仓库内已有 `.claude/skills/` 目录，技能直接可用，无需 init。路径引用直接匹配仓库目录结构。
