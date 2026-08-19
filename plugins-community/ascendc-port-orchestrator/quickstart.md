# ascendc-port-orchestrator 快速入门

本插件提供两个入口：

- `ascendc-cross-gen-port`：将 **arch22** AscendC 算子移植到 **arch35 / A5**。
- `ascendc-backward-gen`：由可微 PyTorch 正向规格生成反向 AscendC 算子。

可在 **Claude Code** 或 **OpenCode** 中运行；两者的入口名相同，但安装步骤不同。

## 0. 开始前确认

控制端需要 Bash、`python3`（3.10+）、SSH/SCP，以及已配置完成的目标 harness：Claude Code 需要可调用的 `claude` CLI 和认证/模型；OpenCode 需要可调用的 `opencode` CLI、`node` 或 `bun`、以及可用的 provider/model。插件安装不会验证模型认证是否可用。

实际构建和验证在 `.ascendc_env` 配置的 NPU 主机/容器中执行，而非要求控制端本地安装 CANN。请先确保目标环境有可用的 CANN、NPU、Python 和相应权限；远端密码认证还需要 `sshpass`。

- **跨代移植**：必须同时可访问来源 A3 和独立的目标 A5，源目录必须包含 `op_host/` 与 `op_kernel/`，且会被检测为 arch22。
- **反向生成**：只需所选目标 NPU；输入是定义 `forward(...)` 与 `BACKWARD_SPEC` 的可微 PyTorch `.py` 规格。

## 1. 安装到 Claude Code

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

隔离安装时，`marketplace add`、`plugin install`、`plugin list` 和 `init.sh` 必须使用同一个 `CLAUDE_CONFIG_DIR`。安装输出中必须出现 `✓ hooks verified live`；缺少它或命令非零退出时，不要开始跑算子。

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

新 OpenCode 配置首次解析插件依赖时需要访问 npm registry，通常需要 1–2 分钟。安装输出中必须同时出现 `opencode resolves injected agents + skills (structural)` 和 `✓ safety net ENFORCES`。`1.18.18` 是建议版本，不是硬性版本门；缺少 `opencode`、`node`/`bun` 或安全网检查失败才会阻断运行。

## 3. 配置 NPU 环境

安装器会在实际插件目录生成 `engine/workspace/.ascendc_env`。编辑该文件，填入实际的主机、容器、CANN 路径、SoC、认证和可选 Python 路径；不要提交这个含凭证的文件。

```ini
# --port-a3 固定构建/验证到 A5，且需要实时 A3 参考
TARGET=a5
A5_HOST=<目标 A5 主机>
A5_CONTAINER=<目标 A5 容器>
A5_CANN_PATH=<容器内 CANN 路径>
A5_SOC_VERSION=<完整 SoC 名称>
A3_HOST=<来源 A3 主机>
A3_CONTAINER=<来源 A3 容器>
A3_CANN_PATH=<容器内 CANN 路径>
A3_SOC_VERSION=<完整 SoC 名称>
```

需要密钥认证时，可额外设置 `A3_SSH_KEY` / `A5_SSH_KEY`（或通用 `SSH_KEY`）。反向生成按 `TARGET` 使用对应目标侧配置；不需要 A3 参考环境。完整字段与网络/容器前置条件见 [`docs/USAGE.md`](./docs/USAGE.md)。

## 4. 在对应前端运行

启动你刚安装的前端后，使用同名 slash 入口：

```text
/ascendc-cross-gen-port 把 /path/to/ops-nn/activation/gelu 移植到 A5
/ascendc-backward-gen 为 /path/to/forward_spec.py 生成反向算子
```

Claude Code 中它们是 Skills；OpenCode 中它们是安装器写入的 Commands。启动器会自动选择当前 harness；不要手工拼 `python -m orchestrator` 或设置 `OPENCODE_CONFIG_CONTENT`。

输出会写到 `engine/workspace/<op>/`（包括 `verification.json`、日志与进度文件），归档位于 `engine/output/`。完整操作、离线安装、运行前检查和排障见 [`docs/USAGE.md`](./docs/USAGE.md)。
