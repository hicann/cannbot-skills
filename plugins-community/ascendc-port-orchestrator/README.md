# ascendc-port-orchestrator

面向 AscendC 算子的社区编排插件。两个入口共享同一条确定性流水线、安全网和双层知识库：

1. **`ascendc-cross-gen-port` — 跨代际移植**：把 arch22 源算子移植到 **arch35 / A5**（如 Ascend910C/V220 → Ascend950PR/V300）。目标架构用自然语言指定，来源架构由代码分析自动识别。
2. **`ascendc-backward-gen` — 正向→反向生成**：由可微 PyTorch 正向规格生成并验证 AscendC 反向（梯度）算子。

## 输入：待移植实现 + golden（统一格式）

跨代移植的输入统一为两部分，两种来源形态共用同一 golden 约定。**实现目录必选；参考模式必须且只能二选一**
（npubench 或 a3_live，见下表）：

1. **必选：待移植的算子实现目录**（两个 `--mode` 名称按来源格式分类）
   - 常规：arch22 `ops-nn` 源算子目录（`--mode port-a3-ops`，= CANN ops 仓通用格式）；
   - TileLang2AscendC 工程：`model_new_ascendc.py + kernel/` 目录（`--mode port-a3-tilelang2ascendc`，= TileLang2AscendC 插件输出格式）。
2. **KernelBench 风格 golden** —— task `.py` 与同 stem `.json` / `.jsonl` sidecar 文件对
   （`--reference-source npubench --npubench-task <task.py> [--npubench-root <root>]`）。
   两种来源形态的 golden **完全一致**，原样冻结为精度真值，不做任何格式转换。
   常规 ops-nn 来源下为**推荐**输入（可显式改选 `a3_live`）；TileLang2AscendC 工程来源下为**必需**输入（唯一真值）。

golden 的具体格式可直接查看仓内示例 [`examples/npukernelbench-native/`](examples/npukernelbench-native/)：
task `.py` 暴露 `Model` 与 `get_input_groups()`（可选 `get_init_inputs()`），同 stem sidecar 逐 case 描述输入。
输入尚非该格式时，请先由输入提供方准备为该格式并复核语义。

## 两种 golden 模式

| 选择 | 精度/性能对比基准 | 必要 NPU 环境 |
|---|---|---|
| `npubench`（推荐） | 精度对照 golden 输出；**加速比 = 目标实现 vs golden 参考实现**（同一 A5 环境 W3/R5 msprof 实测） | A5 |
| `a3_live`（显式可选，仅常规 ops-nn 来源） | 精度对照当次 A3 实测输出；**加速比 = 目标实现 vs A3 实现实测** | A3 和独立 A5 |

TileLang2AscendC 工程来源只支持 `npubench` golden。
**两种模式的加速比基准不同（golden 参考实现 vs A3 实现），数值不可直接横向比较。**

## 支持的运行底座

插件支持 **Claude Code** 和 **OpenCode** 两条独立运行路径，共用 `engine/` 中的编排器和安全网；启动器会按当前
会话自动选择后端。

| 底座 | 用户入口 | 运行时要求 |
|---|---|---|
| Claude Code | `/ascendc-cross-gen-port`、`/ascendc-backward-gen` Skills | 可调用的 `claude` CLI 与已配置的认证/模型 |
| OpenCode | 同名 `/ascendc-*` Commands | 可调用的 `opencode` CLI、`node` 或 `bun`、已配置的 provider/model |

## 运行前需要准备什么

控制端需要 Bash、Python 3.10+ 和 `bubblewrap`（worker 沙箱，缺失即 fail-closed）、GNU `timeout`、
`cmake`/`g++` 与 Python 开发头文件；目标环境需要 CANN、`torch`/`torch_npu` 与可用的 `npu-smi`，并在启动前
`source` CANN 的 `set_env.sh`。完整依赖清单与检查命令见 [`docs/USAGE.md`](docs/USAGE.md) §1.5。
NPU 主机/容器、CANN 路径、SoC 等配置写入 `engine/workspace/.ascendc_env`（可能含凭证，已 gitignore，绝不能提交）。

## 文档导航

- [`quickstart.md`](quickstart.md)：最短的 Claude Code / OpenCode 上手路径。
- [`docs/USAGE.md`](docs/USAGE.md)：完整前置条件、配置字段、两种 golden 模式与运行说明、FAQ。
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：编排器、安全网、双层 KB 和 harness 实现边界（维护者/评审）。
- [`examples/npukernelbench-native/`](examples/npukernelbench-native/)：golden 输入格式示例。

## 知识库

运行时知识优先级为用户本地 KB（c）> 插件自带 KB（b）> 社区 Skills（a）。用户 KB 默认位于
`~/.ascendc-port/user_kb/`，可用 `ASCENDC_PORT_USER_KB` 覆盖；插件自带 KB 位于 `kb/`，运行时只读。

本插件处于社区维护阶段，源码位于 `plugins-community/ascendc-port-orchestrator/`。
