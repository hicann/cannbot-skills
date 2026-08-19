# ascendc-port-orchestrator

面向 AscendC 算子的社区编排插件。两个入口共享同一条确定性流水线、安全网和本地知识库：

1. **`ascendc-cross-gen-port` — 跨代际移植**：当前仅支持 **arch22 源算子 → arch35 / A5 目标**（例如 Ascend910C/V220 → Ascend950PR/V300）。来源架构由代码分析确认；目标固定为 A5，不支持把 A3/arch22 作为此模式的目标。
2. **`ascendc-backward-gen` — 正向→反向生成**：由可微 PyTorch 正向规格生成并验证 AscendC 反向（梯度）算子。

## 支持的运行底座

插件已支持 **Claude Code** 和 **OpenCode** 两条独立运行路径。两者共用 `engine/` 中的编排器和安全网；启动器会按当前会话选择后端，用户不需要手工设置 `AOG_HARNESS_BACKEND` 或 `OPENCODE_CONFIG_CONTENT`。

| 底座 | 用户入口 | 运行时要求 |
|---|---|---|
| Claude Code | `/ascendc-cross-gen-port`、`/ascendc-backward-gen` Skills | 可调用的 `claude` CLI 与已配置的认证/模型 |
| OpenCode | 同名 `/ascendc-*` Commands | 可调用的 `opencode` CLI、`node` 或 `bun`、已配置的 provider/model |

OpenCode `1.18.18` 是已验证的建议版本；更低版本只会产生兼容性 warning，不会因版本号本身拒绝执行。可执行文件缺失或安全网检查失败仍会阻断运行。

## 运行前需要准备什么

- 控制端需要 Bash、Python 3.10+、SSH/SCP，以及一个安装后不会被移动或删除的插件副本。
- 在 `.ascendc_env` 中配置实际执行任务的 NPU 主机/容器、CANN 路径、SoC、认证和 Python 环境；其中可能包含凭证，文件已 gitignore，绝不能提交。
- 跨代移植使用实时 A3 参考并在 A5 构建/验证，因此需要可达的 **A3 与独立 A5** 环境；反向生成只需所选目标 NPU，但其正向规格运行环境需具备 PyTorch。
- 远端执行会使用 SSH/SCP 和容器命令；密码认证还需要 `sshpass`，也可使用默认 SSH 配置或 `A3_SSH_KEY` / `A5_SSH_KEY`。

安装、配置、验收标志、离线安装和排障请看：

- [`quickstart.md`](./quickstart.md)：最短的 Claude Code / OpenCode 上手路径。
- [`docs/USAGE.md`](./docs/USAGE.md)：完整前置条件、配置字段和运行说明。
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)：编排器、安全网、双层 KB 和 harness 实现边界。

## 知识库

运行时知识优先级为用户本地 KB（c）> 插件自带 KB（b）> 社区 Skills（a）。用户 KB 默认位于 `~/.ascendc-port/user_kb/`，可用 `ASCENDC_PORT_USER_KB` 覆盖；插件自带 KB 位于 `kb/`，运行时只读。

本插件处于社区维护阶段，源码位于 `plugins-community/ascendc-port-orchestrator/`。
