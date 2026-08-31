# ascendc-port-orchestrator 快速入门

本插件提供两个入口：

- `ascendc-cross-gen-port`：将 **arch22** AscendC 算子移植到 **arch35 / A5**。
- `ascendc-backward-gen`：由可微 PyTorch 正向规格生成反向 AscendC 算子。

可在 **Claude Code** 或 **OpenCode** 中运行；两者的入口名相同，但安装步骤不同。

## 0. 开始前确认

默认由本地 agent 生成算子，并在本地 A5 目标上验证。只有选择显式远程验证时，才需要 SSH/SCP。注意引擎会以
子进程方式拉起子 agent worker，因此即使你已经在 Claude Code / OpenCode 会话里，机器上仍必须有可调用的
CLI 本体：Claude Code 需要可调用的 `claude` CLI 和认证/模型，OpenCode 需要可调用的 `opencode` CLI、
`node` 或 `bun` 以及可用的 provider/model。

实际构建和验证在 `.ascendc_env` 配置的本地 A5 环境中执行。除可用的 CANN、NPU、Python 和相应权限外，还必须安装
`bubblewrap`（worker 沙箱，缺失即 fail-closed 拒绝运行）、`cmake`/`gcc`/`g++` 与 Python 开发头文件，目标环境需有
`torch`/`torch_npu`，并在启动前 `source` CANN 的 `set_env.sh`；部分驱动栈（如 950DT）还需
`export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH` 才能运行 `npu-smi`。完整清单与检查命令见
[`docs/USAGE.md`](./docs/USAGE.md) §1.5「环境前置依赖清单」。

**术语速览**（首次使用建议先扫一眼）：

| 术语 | 含义 |
|---|---|
| arch22 / A3 | 来源代际（如 Ascend910C/V220），待移植算子当前运行的架构 |
| arch35 / A5 | 目标代际（如 Ascend950PR/V300），移植的目的架构 |
| golden | 精度/性能的真值参照：一份被冻结的 KernelBench 风格 task `.py` + 同名 `.json`/`.jsonl` 输入描述文件 |
| lane | 目标机上一块 NPU 设备的编号；用 `npu-smi info` 查看哪些设备空闲 |
| fail-closed | 前置条件不满足时拒绝启动并给出原因，而不是带着错误配置继续跑 |

## 1. 安装到 Claude Code

若尚无仓库 checkout，先克隆：

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills
```

从本地 checkout 添加 marketplace 并安装插件：

```bash
REPO_ROOT=/path/to/cannbot-skills

claude plugin marketplace add "$REPO_ROOT"
claude plugin install ascendc-port-orchestrator@cannbot

# Claude Code 只复制/注册插件；从安装清单取真实安装路径，再执行 init.sh。
PLUGIN_INSTALL_PATH="$(claude plugin list --json | python3 -c '
import json, sys
plugins = json.load(sys.stdin)
print(next(p["installPath"] for p in plugins
           if p.get("id") == "ascendc-port-orchestrator@cannbot"))
')"
test -n "$PLUGIN_INSTALL_PATH"
bash "$PLUGIN_INSTALL_PATH/init.sh" global claude --strict-deps
```

隔离安装时，`marketplace add`、`plugin install`、`plugin list` 和 `init.sh` 必须使用同一个
`CLAUDE_CONFIG_DIR`。安装输出中必须出现 `✓ hooks verified live`；缺少它或命令非零退出时，不要开始跑算子。

## 2. 安装到 OpenCode

OpenCode 必须从完整 checkout 安装，不能使用 Claude marketplace 的缓存目录。若尚无 checkout：

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills
```

全局安装：

```bash
REPO_ROOT=/path/to/cannbot-skills
PLUGIN_DIR="$REPO_ROOT/plugins-community/ascendc-port-orchestrator"
bash "$PLUGIN_DIR/init.sh" global opencode --strict-deps
```

项目级安装时，先进入**你的算子项目目录**，不要在插件目录里运行：

```bash
REPO_ROOT=/path/to/cannbot-skills
PLUGIN_DIR="$REPO_ROOT/plugins-community/ascendc-port-orchestrator"
cd /path/to/your-operator-project
bash "$PLUGIN_DIR/init.sh" project opencode --strict-deps
```

新 OpenCode 配置首次解析插件依赖时需要访问 npm registry，通常需要 1–2 分钟。安装输出中必须同时出现
`opencode resolves injected agents + skills (structural)` 和 `✓ safety net ENFORCES`。`1.18.18` 是建议版本，
不是硬性版本门；缺少 `opencode`、`node`/`bun` 或安全网检查失败才会阻断运行。

## 3. 配置 NPU 环境

安装器会在实际插件目录生成 `engine/workspace/.ascendc_env`（Claude Code 安装时即 §1 的
`$PLUGIN_INSTALL_PATH`，OpenCode 安装时为 checkout 中的 `$PLUGIN_DIR`）。默认使用 `A5_CONTAINER=local` 进行本地生成和验证。
不要提交这个可能含凭证的文件。

```ini
TARGET=a5
A5_HOST=
A5_USER=root
A5_PASSWORD=
A5_CONTAINER=local
A5_CANN_PATH=/usr/local/Ascend/cann-9.0.0
# 目标设备真实报告的 SoC。查询命令：npu-smi info -t board -i <NPU编号>，取输出中的 Chip Name
# （如 Ascend950PR）。留空或填未知值会 fail-closed，禁止 A5 验收。
A5_SOC_VERSION=<目标设备真实 SoC>
A5_DEFAULT_NPU_ID=0
# 留空 = 不启用实时 A3 真值；仅显式选择 a3_live 参考真值时才填 a3_live。
PORT_A3_REFERENCE_SOURCE=
NPU_PYTHON_BIN=<可选：本地 python3 可执行文件路径，或其所在 bin 目录>
BENCHMARK_ROOT=<可选：本地可写的 benchmark 根目录>
```

`A5_SOC_VERSION` 不提供默认值，必须填写**目标设备**实际报告的 SoC（查询命令见上）。注意该字段描述的是
目标机：如果把 `A5_*` 指向一台 Ascend910（arch22）机器，启动时会先给出 warning——可以用它跑 preflight
和代码生成 smoke 来提前验证流程，但最终 A5 npubench 验证会停止，910 上的结果**不能作为移植验收**；
空值或未知值也会 fail-closed。

若要改为远程验证，必须同时填写真实主机和容器名，这不会影响默认本地模式：

```ini
A5_HOST=<目标主机>
A5_CONTAINER=<远端容器名>
# 远端容器内拥有 torch/torch_npu 的 Python；不要填写控制端路径。
A5_NPU_PYTHON_BIN=<远端容器内 python3 或其 bin 目录>
```

只有显式选择实时 A3 真值（`--reference-source a3_live`，仅常规 ops-nn 来源）时才需要额外填写 `A3_*` 块。
需要密钥认证时，可额外设置 `A5_SSH_KEY`（使用 A3 实测时再加 `A3_SSH_KEY`）或通用 `SSH_KEY`。
完整字段与网络/容器前置条件见 [`docs/USAGE.md`](./docs/USAGE.md)。

## 4. 在对应前端运行

**两种等价的使用方式**：日常用对话入口——启动你刚安装的前端（Claude Code 或 OpenCode），使用同名 slash
命令加自然语言描述任务，入口会代你解析参数并调用底层启动器；[`docs/USAGE.md`](./docs/USAGE.md) §3 给出的
`scripts/launch_orchestrator.sh` 命令是同一启动器的手动形态，用于脚本化或排障。不要手工拼
`python -m orchestrator` 或设置 `OPENCODE_CONFIG_CONTENT`。

先准备输入。跨代移植的输入统一为两部分：**待移植的算子实现目录（必选）** + **参考真值（必选且只能二选一：
npubench 或 a3_live）**：

1. **算子实现目录**：arch22 `ops-nn` 源算子目录（命令行形态用 `--source <目录>` 传入），可从 CANN 社区
   [ops-nn 仓](https://gitcode.com/cann/ops-nn) 获取（每个算子一个目录，如 `norm/bn_training_update_grad/`，
   含 `op_host/`、`op_kernel/` 等实现）。
2. **golden**：推荐 npubench 的 KernelBench 风格 golden——task `.py` + 与 task 同名（仅扩展名不同）的
   `.json`/`.jsonl` sidecar 文件对，格式示例见插件内 `examples/npukernelbench-native/`（TileLang2AscendC
   工程来源下 golden 为必需；a3_live 仅常规 ops-nn 来源可选）。

对话入口示例：

```text
/ascendc-cross-gen-port 把 <arch22 ops-nn 源算子目录，算子语义需与 golden 的 Add 一致> 移植到 A5；
使用插件内置 KernelBench 风格 task <插件目录>/examples/npukernelbench-native/level1/example_add.py
作为 golden，根目录为 <插件目录>/examples/npukernelbench-native
```

其中 `<插件目录>`：Claude Code 安装即 §1 的 `$PLUGIN_INSTALL_PATH`，OpenCode 安装为 checkout 中的 `$PLUGIN_DIR`。

预检成功时，本次运行工作区下 `engine/workspace/<算子名>/npubench_evidence/preflight_report.json` 的
`status` 为 `PASS`。

- 两个 `--mode` 名称按来源格式分类：`port-a3-ops`（CANN ops 仓通用格式）/ `port-a3-tilelang2ascendc`
  （TileLang2AscendC 插件输出格式）。来源是 TileLang2AscendC 工程（`model_new_ascendc.py + kernel/`）时，入口会把
  `--mode` 换成 `port-a3-tilelang2ascendc`，golden 参数不变且为**必需**（TileLang2AscendC 来源的唯一真值，缺 task 会在启动时被拒绝）。
- 只有用户明确要求实时 A3 CANN 真值（仅常规 ops-nn 来源）时，才把 golden 换成
  `--reference-source a3_live`，并准备好 A3 与 A5。
- 反向生成（forward spec 的完整格式示例见插件内 `scripts/reference_provider/examples/gelu_spec.py`）：

```text
/ascendc-backward-gen 为 /path/to/forward_spec.py 生成反向算子
```

Claude Code 中它们是 Skills；OpenCode 中它们是安装器写入的 Commands。启动器会自动选择当前 harness。

**运行管理**：一次完整移植通常需要几十分钟到数小时（含模型多轮修复迭代、受控编译与精度/性能评测）。
运行中可看
`engine/workspace/<算子名>/PROGRESS.md` 了解进度；中断续跑与推翻重跑（`--resume` / `--cold-start`）见
[`docs/USAGE.md`](./docs/USAGE.md) §5.3。**成功标准**：`engine/workspace/<算子名>/verification.json`
精度 PASS，且产物与报告归档到 `engine/output/`。

产出（目标算子 + 精度/性能验证报告 + 复现指引）位于**插件目录下**的 `engine/workspace/<op>/`（包括
`verification.json`、日志与进度文件；Claude Code 安装时为 `$PLUGIN_INSTALL_PATH`，OpenCode 为 checkout 中的
`$PLUGIN_DIR`），归档位于 `engine/output/`。完整操作、离线安装、运行前检查和排障见
[`docs/USAGE.md`](docs/USAGE.md)；架构与实现边界见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。
