# TileLang 算子开发插件

统一入口按本机芯片选择独立工作流：Ascend910 提供基于 tilelang-ascend 的算子开发流程，Ascend950 提供设计、生成、ST、代码检视与性能调优能力。两套技能正文、模板、脚本和角色规则独立维护。

## 安装

在用户项目目录运行插件安装脚本：

```bash
bash /path/to/cannbot-skills/plugins-official/tilelang-op-orchestrator/init.sh project codex /path/to/project
bash /path/to/cannbot-skills/plugins-official/tilelang-op-orchestrator/init.sh project opencode /path/to/project
bash /path/to/cannbot-skills/plugins-official/tilelang-op-orchestrator/init.sh project claude /path/to/project
```

也支持 `trae`、`cursor`、`copilot`、`codearts`；将 `project` 改为 `global` 可使用全局安装模式。完整参数见 `bash init.sh --help`。

安装器只配置技能、角色和入口，不探测设备、不下载或编译 TileLang。SessionStart hook 只注入芯片路由入口。启动会话并选择平台后，环境准备由对应工作流处理。

| 宿主 | 项目技能目录 | 项目角色目录 | 项目入口 |
|---|---|---|---|
| Codex | `.agents/skills/` | `.codex/agents/` | `AGENTS.md` |
| OpenCode | `.opencode/skills/` | `.opencode/agents/` | `AGENTS.md` |
| Claude Code | `.claude/skills/` | `.claude/agents/` | `CLAUDE.md` |
| TRAE | 由现有 IDE / Plugin / CLI 检测选择 | 同配置根目录下 `agents/` | `AGENTS.md` |
| Cursor | `.cursor/skills/` | `.cursor/agents/` | `AGENTS.md` |
| Copilot | `.github/skills/` | `.github/agents/` | `AGENTS.md` |
| CodeArts | `.codeartsdoer/skills/` | `.codeartsdoer/agents/` | `AGENTS.md` |

安装后在用户项目目录启动对应工具，直接描述算子需求。配置通过链接访问本仓库，保留 cannbot-skills 源码目录；移动仓库后重新安装。

## 芯片路由

入口按 [TileLang 芯片路由说明](references/tilelang-routing.md) 按名称加载 `ascendc-env-check` Skill，以该 Skill 的真实目录为基准运行 `scripts/get_npu_arch.py --json`，查询完整 SoC 与 NpuArch 并校验组合：Ascend910B（含 B2C、B4-1）/ Ascend910_93、2201 进入 Ascend910；Ascend950PR / Ascend950DT、3510 进入 Ascend950。未知、冲突或不支持的型号停止路由，不默认使用某一分支；确定型号后再按名称加载 `npu-arch` 查询对应架构与硬件能力。

- [Ascend910 workflow](workflows/Ascend910/workflow.md)：包含设计、开发、分层测试、覆盖门禁、状态恢复和性能调优。开始设计前运行本分支的 `scripts/prepare_framework.sh`，按需克隆并执行 `bash install_ascend.sh`；源码和编译环境有效时跳过对应步骤，再进行环境预检。
- [Ascend950 workflow](workflows/Ascend950/workflow.md)：按设计 → 生成 → 指定用例精度通过 → ST → 询问调优 → 确认后 Flash / Standard → 询问代码检视 → 确认后检视最终交付代码的顺序执行。框架源码和实际 Python 导入版本按本分支技能核实，不使用 Ascend910 的环境安装脚本。

算子产物以用户项目为工作目录：Ascend910 使用 `custom/{op}/`，Ascend950 使用 `operators/`。技能、角色和 workflow 的相对引用以源文件真实目录为基准，不能把工作流源码目录当成算子输出目录。

## 首次准备与进度

`init.sh` 不拉取源码。芯片路由后，各工作流只准备自己的仓库：

| 平台 | 插件内默认源码路径 | 准备步骤 |
|---|---|---|
| Ascend910 | `repositories/Ascend910/tilelang-ascend/` | 克隆及子模块 → 按需编译安装 → 验证编译产物和 Python 环境 |
| Ascend950 | `repositories/Ascend950/ops-tilelang/`、`repositories/Ascend950/tilelang/` | 浅克隆两个主仓库；查阅代码需要时单独获取子模块 |

已有有效源码直接使用。Ascend910 优先使用用户指定路径，会检查实际编译产物及当前 Python 导入路径，验证通过时跳过编译；编译失败或中断时，下次使用保留的源码重试。Ascend950 固定使用插件内路径，不接受路径参数或环境变量覆盖；默认不下载子模块，仅准备参考源码，运行环境需另行核实。`SOURCE_READY` 不要求子模块或编译依赖完整，按需获取方式见其工作流的“源码准备”章节。

准备脚本持续显示 Git 下载和编译日志，每 15 秒输出阶段及耗时；智能体每 15～30 秒同步进展。日志保存在插件 `.preparation/<平台>/` 下，每次执行打印完整路径。失败时显示阶段和退出码，日志用于定位问题。同一源码目录的准备任务互斥，避免重复下载或同时编译。

## 技能与角色

工作流按名称加载已安装的 Skill，并传入已确定的平台 `Ascend910` 或 `Ascend950`；各 Skill 入口负责选择该平台正文，无需每个阶段重新探测芯片。跨 Skill 调用不附正文路径，缺失时停止当前阶段并报告。

| Ascend950 功能 | Skill 名称 |
|---|---|
| 方案设计 | `tilelang-op-design` |
| 算子生成 | `tilelang-op-develop` |
| ST 系统测试 | `tilelang-op-test-design` |
| 性能优化 | `tilelang-perf-optimization` |
| 性能最佳实践 | `tilelang-performance-best-practices` |
| 代码检视 | `tilelang-review` |
| 芯片识别与能力查询 | `ascendc-env-check` / `get_npu_arch.py` → `npu-arch` |
| 性能采集与分析 | `tilelang-op-profiling` |

Ascend910 工作流按阶段加载 `tilelang-env-check`、`tilelang-op-design`、`tilelang-op-test-design`、`tilelang-op-develop`、`tilelang-perf-optimization` 等 Skill；API 与编程模式参考分别加载 `tilelang-api-best-practices`、`tilelang-programming-model-guide`，需要时加载 `tilelang-submodule-pull` 和 `tilelang-review`。子代理分别按角色名 `tilelang-op-analyst`、`tilelang-op-developer`、`tilelang-op-perf-tuner` 调度；角色缺失时停止并报告。

Ascend950 的 ST 能力包括覆盖设计、可信度审查、静态发现、用例规格校验和 pytest 执行证据。生成阶段通过指定用例后直接进入 ST，ST 验收运行已确认的完整目标用例清单。代码检视使用 tilelang-review 的 Ascend950 正文，包含格式检查、适用条例检视与报告生成，只检查明确指定的本地文件或目录。调优完成后经用户确认检视最终交付文件，格式修复另需用户确认。这两项技能也支持独立调用；已有测试按本分支测试执行说明运行。

Ascend950 各阶段的 Python 代码不固定 `target` 或设置后端环境变量；运行命令通过 `TILELANG_DEFAULT_TARGET=pto` 或 `TILELANG_DEFAULT_TARGET=ascend` 选择后端。

Ascend910 使用 `tilelang-op-analyst`、`tilelang-op-developer`、`tilelang-op-perf-tuner`。Ascend950 使用 `tilelang-tuning`、`tilelang-perf-analysis-expert`、`tilelang-perf-impl-expert`；Flash 由当前主 Agent 自主执行。`agents/` 中的发现文件是薄适配，业务正文在各自 `workflows/<platform>/agents/` 中。

## 维护与验证

- 修改 Ascend910 时只改 `Ascend910/`，修改 Ascend950 时只改 `Ascend950/`；不提取公共业务步骤、API 或测试标准。
- 入口共享的只有芯片识别和路径路由。跨 Skill 按名称加载 `ascendc-env-check`，再以其真实目录内的 `scripts/get_npu_arch.py` 取得本机型号；平台确定后按名称加载 `npu-arch` 查询架构与硬件能力。Ascend950 profiling 独立保存在 tilelang-op-profiling，与通用 ops-profiling 相互独立。
- 技能根目录的 `references/`、`scripts/`、`examples/` 等资源链接指向 Ascend910 分支。维护时编辑对应平台目录内的文件；工作流按名称加载 Skill，Skill 正文按自身目录的相对路径引用资源。
- Ascend950 分支的许可证见 [LICENSE](workflows/Ascend950/LICENSE)。
- 更新时逐项核对本分支技能、正文及依赖；保留模板成熟度、版本限制和验证记录。不得直接覆盖另一平台或通用 profiling 内容。
- 新增发现入口时同时更新 `init.sh` 白名单、插件 manifest、marketplace 和对应分支的引用。芯片路由支持范围变更时补充路由测试。

在仓库根目录执行离线检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ops/tilelang-performance-best-practices/Ascend950/scripts/validate_templates.py
```

上述命令只验证模板和 tiling 的静态约束，不访问 NPU。真实算子精度与性能仍由对应平台工作流在目标环境验收。
