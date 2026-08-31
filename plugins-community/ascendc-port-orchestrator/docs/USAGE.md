# ascendc-port-orchestrator — 使用说明

本插件提供两项能力：

1. `ascendc-cross-gen-port`：将 arch22 AscendC 源算子移植到 arch35 / A5。
2. `ascendc-backward-gen`：从可微 PyTorch 正向规格生成反向 AscendC 算子。

两个入口都能在 Claude Code 与 OpenCode 中使用，且都通过同一个启动器进入 `engine/` 中的确定性编排器。

**术语速览**：

| 术语 | 含义 |
|---|---|
| arch22 / A3 | 来源代际（如 Ascend910C/V220），待移植算子当前运行的架构 |
| arch35 / A5 | 目标代际（如 Ascend950PR/V300），移植的目的架构 |
| golden | 精度/性能真值参照：一份被冻结的 KernelBench 风格 task `.py` + 同名 `.json`/`.jsonl` sidecar |
| sidecar | 与 task 同名（仅扩展名不同）的输入描述文件 |
| lane | 目标机上一块 NPU 设备的编号；用 `npu-smi info` 查看哪些设备空闲 |
| fail-closed | 前置条件不满足时拒绝继续并给出原因，而不是带着错误配置继续跑 |
| W3/R5 | 性能评测口径：warmup 3 次、repeats 5 次的 msprof 实测 |
| O0~O6 | 流水线阶段编号（解析→分类→参考采集→移植→构建→精度验证→性能→归档） |

## 0. 输入与参考真值

跨代移植每次运行需要两类输入，分别对应一个独立的选择维度——注意别混淆这两个维度：

1. **待移植的算子实现目录**（必选）：目录本身用 `--source <目录>` 传入，并用 `--mode` 声明它的来源形态
   （来源架构由代码分析自动识别，无需指定）：
   - `--mode port-a3-ops`：CANN ops 仓通用格式，即 arch22 `ops-nn` 源算子目录。源算子可从 CANN 社区
     [ops-nn 仓](https://gitcode.com/cann/ops-nn) 获取，每个算子一个目录（如 `norm/bn_training_update_grad/`，
     含 `op_host/`、`op_kernel/` 等实现）。
   - `--mode port-a3-tilelang2ascendc`：TileLang2AscendC 插件输出格式，即 `model_new_ascendc.py + kernel/` 目录。
2. **参考真值**（必选，且只能二选一）：用 `--reference-source` 声明精度/性能对照谁：
   - `npubench`（推荐）：一份冻结的 KernelBench 风格 golden 文件对，格式见 §0.1；
   - `a3_live`：当次在来源 A3 机器上实测采集，仅 `--mode port-a3-ops` 可用，且需额外 A3 环境（见 §4）。

两类输入缺任何一个都会 fail-closed 拒绝启动；两种参考真值不能混用。差异对比见 §0.2。

### 0.1 golden 格式与冻结粒度（`npubench` 参考）

golden = task `.py` + 与 task 同名（仅扩展名不同）的 `.json` / `.jsonl` sidecar 文件对，传入
`--reference-source npubench --npubench-task <task.py> [--npubench-root <root>]`。冻结的是
**reference program**（task 代码 + 输入描述 sidecar），运行时在 A5 上生成参考输出；不读取 A3 runtime/capture。
两种 `--mode` 来源形态使用**完全相同的 golden**：常规 ops-nn 来源下 golden 为**推荐**输入（也可显式改选
`a3_live`）；TileLang2AscendC 来源下 golden 为**必需**输入（唯一真值，没有 `a3_live` 替代，缺 task 会在启动时被拒绝）。

golden 的具体格式可查看仓内示例 `examples/npukernelbench-native/`：task `.py` 暴露 `Model` 与
`get_input_groups()`（可选 `get_init_inputs()`），同名 sidecar 逐 case 描述输入（JSONL 内容也可放在
`.json` 后缀里）。若输入尚非该格式，请在启动前由输入提供方准备为该格式并复核语义，插件不做自动转换。

注意冻结粒度：`--npubench-root` 声明的目录会被**整体闭包冻结**（task 可能依赖同目录的辅助文件与相对导入，
引擎不做按需裁剪）。因此 root 必须是干净的任务树——不要残留 `build/`、`dist/`、`*.egg-info`、
`__pycache__/` 等构建产物，否则会在 worker spawn 前被安全网 fail-closed 拒绝。最稳妥的做法是把 task
`.py` + 同名 sidecar 单独放进一个只含这两类文件的目录。

### 0.2 两种参考真值的差异与选择

| 选择 | 行为 | 精度/性能对比基准 | 必要 NPU 环境 | 如何选择 |
|---|---|---|---|---|
| `npubench`（推荐 golden） | 逐字节冻结 task `.py` 与同名 `.json` / `.jsonl`，以 task API 为精度真值并走原生评测 | 精度对照 golden 输出；**加速比 = 目标实现 vs golden 参考实现**（同一 A5 环境 W3/R5 msprof 实测） | A5 | `--reference-source npubench --npubench-task <task.py>`，可选 `--npubench-root <root>` |
| `a3_live`（显式可选，仅常规 ops-nn 来源） | 当次在来源 A3 上采集 CANN 实测输出 | 精度对照当次 A3 实测输出；**加速比 = 目标实现 vs A3 实现实测** | A3 和独立 A5 | `--reference-source a3_live` |

选择规则：

- 选 `npubench` 必须带 task；选 `a3_live` 不得带 task，且仅 `--mode port-a3-ops` 来源可用。
  TileLang2AscendC 工程来源（`--mode port-a3-tilelang2ascendc`）**只支持 npubench golden**：引擎会拒绝该来源与
  `a3_live` 的组合，以冻结 task 为唯一真值。
- **注意：两种参考真值的加速比基准不同——npubench 与 golden 参考实现比、a3_live 与 A3 实现比，两者数值不可直接横向比较。**
- 选择 `a3_live` 的两种等价方式：命令行 `--reference-source a3_live`，
  或 `.ascendc_env` 配置 `PORT_A3_REFERENCE_SOURCE=a3_live`（默认留空；未显式选择时裸 `--port-a3-ops` fail-closed）。
  已有旧 workspace 缺少 `reference` 状态块同样 fail closed，普通 resume 不会猜测真值来源。

## 1. 前置条件与安装

### 1.1 本地目标与远程目标

默认部署是**本地目标模式**：由本地 agent 生成候选算子，并在本地 A5 环境完成验证（`A5_CONTAINER=local`）。

远程目标不是默认行为。仅当明确填写 `A5_HOST=<host>` 并把 `A5_CONTAINER` 改成远端实际容器名时，编排器才会走
SSH/SCP 到该容器执行 O2.5/O5；控制端只保留工作区和导入的不可变验证证据。A3 仅在显式选择 `a3_live` 时，
作为本次来源 CANN 真值的执行环境。

- 本地目标需要 Bash、Python 3.10+、CANN、可见 NPU、`torch_npu` 和 task 所需依赖。
- 远程目标额外需要 SSH/SCP；密码认证时还需要 `sshpass`。密钥认证可用 `A3_SSH_KEY` / `A5_SSH_KEY` 或通用 `SSH_KEY`。
- `npubench` task 在目标 A5 环境加载，因此该环境需具备 task 所需的 PyTorch、`torch_npu` 和依赖；它不需要 A3 配置。

| 前端 | 安装来源 | 运行时 |
|---|---|---|
| Claude Code | marketplace 或完整 checkout | 可调用的 `claude` CLI，已配置认证与模型 |
| OpenCode | **完整 `cannbot-skills` checkout** | `opencode` CLI、`node` 或 `bun`、已配置 provider/model；固定子 agent 模型时可设 `AOG_OPENCODE_MODEL*` |

OpenCode `1.18.18` 是已验证的建议版本，不是硬版本门：更低、无法解析或无法查询版本时只给 warning；缺少
`opencode`、`node`/`bun` 或安全网行为探针失败才会阻断运行。

### 1.2 Claude Code 安装

若尚无仓库 checkout，先 `git clone https://gitcode.com/cann/cannbot-skills.git`。

```bash
claude plugin marketplace add /path/to/cannbot-skills
claude plugin install ascendc-port-orchestrator@cannbot

PLUGIN_INSTALL_PATH="$(claude plugin list --json | python3 -c '
import json, sys
plugins = json.load(sys.stdin)
print(next(p["installPath"] for p in plugins
           if p.get("id") == "ascendc-port-orchestrator@cannbot"))
')"
test -n "$PLUGIN_INSTALL_PATH"
bash "$PLUGIN_INSTALL_PATH/init.sh" global claude --strict-deps
```

安装输出必须包含 `✓ hooks verified live`。没有这条或安装退出非零时，不要开始运行任务。
`--strict-deps` 会将缺少 CLI 的默认 warning 升级为安装期失败；隔离安装时，marketplace add/install/list/init
必须使用同一个 `CLAUDE_CONFIG_DIR`。

在容器或远程 Linux 环境中，请为本次运行设置一个容器内的新 `CLAUDE_CONFIG_DIR`，并在该目录中重新执行
marketplace add/install/list 和 `init.sh`。不要把宿主机 checkout 中生成的 `.claude/`、`cannbot-manifest.json`
或其他 Claude 缓存目录直接复制进容器；这些文件可能记录宿主机绝对路径，无法作为容器内 hook 配置使用。

### 1.3 OpenCode 安装

```bash
REPO_ROOT=/path/to/cannbot-skills
PLUGIN_DIR="$REPO_ROOT/plugins-community/ascendc-port-orchestrator"

# 全局安装
bash "$PLUGIN_DIR/init.sh" global opencode --strict-deps

# 或项目级安装到当前项目的 .opencode/
cd /path/to/your-operator-project
bash "$PLUGIN_DIR/init.sh" project opencode --strict-deps
```

OpenCode 不能从 Claude marketplace cache 安装；必须使用完整 checkout。新配置第一次解析依赖可能需要访问
npm registry。安装输出必须包含 `opencode resolves injected agents + skills (structural)` 与
`✓ safety net ENFORCES`：前者证明 OpenCode 接受了私有注入的入口/agent 配置，后者证明守卫实际拒绝
越界 workspace 读取；没有这些标志或安装退出非零时，不要开始运行任务。

流式 watchdog 默认使用 `AOG_STREAM_SILENCE_TIMEOUT_SEC`（1800 秒），可用
`AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC` 单独覆盖。`AOG_OPENCODE_SKIP_RUNTIME_CHECK=1` 仅限测试或短时排障，
不能作为常规配置。

### 1.4 离线或受限网络

OpenCode 首次安装需要一次性拉取 npm 依赖。完全离线时，先在联网机器运行 `init.sh … opencode` 并触发一次
`opencode run`，再将生成的 `$CONFIG_ROOT/{node_modules,package.json,package-lock.json}` 复制到目标机器；或预置
npm 镜像。无论哪种方式，仍须完成上面的 structural 和 safety-net 验证。

`init.sh` 对两种 harness 都会创建用户本地 KB、构建官方 OKF 索引，并从模板 scaffold
`engine/workspace/.ascendc_env`。

如果把 macOS 工作区手工打包复制到 Linux/NPU 容器，建议使用 Git checkout 或在打包前设置
`COPYFILE_DISABLE=1`，避免 AppleDouble `._*` 资源叉文件混入 `kb/okf`。

### 1.5 环境前置依赖清单（Dependencies）

`init.sh --strict-deps` 只校验 harness CLI（`claude`，或 `opencode` + `node`/`bun`）；以下系统依赖**不做安装期
校验**，缺失时会在对应运行阶段才失败。运行前请逐项确认：

| 依赖 | 用途 | 检查命令 | 缺失时行为 |
|---|---|---|---|
| `bubblewrap`（`bwrap`）+ 可创建的 unprivileged user namespace | graybox worker 沙箱，所有 port / backward 运行默认强制 | `bwrap --version`，且 `bwrap --unshare-user --dev-bind / / true` 能成功 | **fail-closed**：首次 spawn worker 即拒绝（`refusing to spawn unsandboxed`）。Debian/Ubuntu 安装：`apt-get install bubblewrap` |
| GNU `timeout` | 启动器看门狗 | `timeout --version` | 启动即 exit 2（fail-early） |
| `python3` ≥ 3.10、`bash`、`git` | 引擎本体、安全网 hooks、归档 | `python3 --version` | hooks 每次工具调用失败 / O0 BLOCKED |
| 控制端 PyYAML | 安全网 hooks 热路径 | `python3 -c 'import yaml; print(yaml.__version__)'` | hook 调用失败，安全网失效 |
| `cmake` ≥ 3.16、`gcc`/`g++`、Python 开发头文件（`Python.h`） | 候选算子受控构建 | `cmake --version`；`python3 -c "import sysconfig,os;print(os.path.isfile(os.path.join(sysconfig.get_path('include'),'Python.h')))"` | 构建阶段失败 |
| CANN toolkit（含 `ccec`、`tikcpp`、`msprof`） | 编译与性能证据 | `ls "$A5_CANN_PATH/compiler/ccec_compiler/bin/ccec"` | 构建 / 性能评测阶段失败 |
| `npu-smi` 位于 `/usr/local/bin` 或 `/usr/bin`（可用 `A5_NPU_SMI_BIN` / `NPU_SMI_BIN` 覆盖） | lane 探测、SoC 核验、lane 健康检查 | `npu-smi info`；查 SoC 用 `npu-smi info -t board -i <NPU编号>`，取输出中的 `Chip Name` | SoC probe fail-closed；lane 探测失败时可用 lane 数静默降级 |
| 目标环境 `torch` + `torch_npu` | task 加载与精度验证 | `python3 -c 'import torch, torch_npu'` | O2.5 / O5 阶段 `ImportError` |
| `ssh`/`scp`（远程目标另需 `sshpass`/`docker`/`rsync`） | 仅远程目标模式 | — | local 模式不需要 |

NPU 运行环境必须在启动前就位：本地 npubench 子进程**整体继承**启动器进程环境。请先执行：

```bash
source <A5_CANN_PATH>/set_env.sh                       # 或 /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH     # 部分驱动栈（如 950DT）的 liburma.so.0 在 /usr/lib64
```

漏掉第二条时，`npu-smi` 和 `import torch_npu` 都会报 `liburma.so.0: cannot open shared object file`。

可选运行期调优：npubench 精度/性能子进程的单任务超时默认 300s（两阶段共用），可用环境变量
`CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC`（正整数，秒）覆盖；非法值（非正整数）在运行前 fail-fast 报错。
大 case 数的算子（如 61-case 注意力算子跑 W3/R5 + msprof profiling）建议设为 1200 或更大。

## 2. 配置 A5 与可选 A3

安装器会在实际插件目录生成 `engine/workspace/.ascendc_env`——Claude Code 安装时即 §1 的
`$PLUGIN_INSTALL_PATH`，OpenCode 安装时为 checkout 中的 `$PLUGIN_DIR`；之后的所有产出也都在该目录的
`engine/workspace/` 下。该文件含连接信息与可能的凭证，已被 gitignore，
绝不能提交。

推荐的 npubench 路径只需填写 A5 块。默认使用本地 A5 target；远程验证是显式 opt-in，只有同时填写
`A5_HOST` 和 `A5_CONTAINER` 才会启用 SSH/SCP。

```ini
TARGET=a5
# 默认：本地 agent 生成、本地 A5 验证。
A5_HOST=
A5_USER=root
A5_PASSWORD=
A5_CONTAINER=local
A5_CANN_PATH=/usr/local/Ascend/cann-9.0.0
# 目标设备真实报告的 SoC。查询命令：npu-smi info -t board -i <NPU编号>，取输出中的 Chip Name
# （如 Ascend950PR）。空值或未知值会 fail-closed。
A5_SOC_VERSION=<目标设备真实 SoC>
A5_DEFAULT_NPU_ID=0
# 留空 = 不启用实时 A3 真值；仅显式选择 a3_live 参考真值时才填 a3_live。
PORT_A3_REFERENCE_SOURCE=
NPU_PYTHON_BIN=<可选：本地 python3 所在目录>
BENCHMARK_ROOT=<可选：本地可写的 benchmark 根目录>
```

`A5_SOC_VERSION` 必须填写**目标设备**实际报告的 SoC（查询命令见上），没有默认值。该字段描述的是目标机：
若把 `A5_*` 指向一台 Ascend910（arch22）机器，启动时会先给出 warning，仍可完成 preflight 和代码生成
smoke，但在最终 A5 npubench 验证开始前会停止——910 上的结果不能作为移植验收。
空值、未知值或格式不合法的 SoC 会直接 fail-closed。

### 2.1 默认：本地生成和验证

在拥有目标 NPU 的环境中运行本地 agent 和插件启动器。模型端点与认证信息按运行环境注入（认证 key 只能
从私有密钥管理注入，不能写入仓库、`.ascendc_env` 或日志）。

以 root 运行时注意（**仅 Claude Code harness**）：`IS_SANDBOX=1` 必须 **export 到启动器进程环境**，不能只写在 settings.json 的 `env`
块里。引擎 spawn worker 时使用 `--permission-mode bypassPermissions`，Claude Code 在 root 下默认拒绝该模式；
进程环境里的 `IS_SANDBOX=1` 声明"已在隔离容器中"，是官方豁免方式。

### 2.2 显式远程验证（opt-in）

只有在控制端无法进入目标容器、并且确实需要 SSH/SCP 时，才把 A5 配置改为远程：

```ini
TARGET=a5
A5_HOST=<目标主机>
A5_USER=root
A5_PASSWORD=<password-or-empty-for-key-auth>
A5_CONTAINER=<远端容器名>
A5_CANN_PATH=/usr/local/Ascend/cann-9.0.0
# 必须填写远端目标设备真实报告的 SoC（查询命令同 §2 本地块）；空值或未知值会 fail-closed。
A5_SOC_VERSION=<目标设备真实 SoC>
# 远端容器内拥有 torch/torch_npu 的 Python；不要填写控制端路径。
A5_NPU_PYTHON_BIN=<远端容器内 python3 或其 bin 目录>
```

远程配置必须同时填写 `A5_HOST` 和远端容器名；缺少 host 的非 `local` 配置会 fail closed，不会回退到本地目标。

只有需要本次来源机的实时 A3 CANN 真值时，保留上述 A5 块并额外填写：

```ini
PORT_A3_REFERENCE_SOURCE=a3_live
A3_HOST=<A3 NPU host>
A3_USER=root
A3_PASSWORD=<password-or-empty-for-key-auth>
A3_CONTAINER=<A3 container>
A3_CANN_PATH=/usr/local/Ascend/cann-9.0.0
A3_SOC_VERSION=<完整 SoC 名称>
A3_CONTAINER_HOME=<A3 容器内 canonical home>
```

`--reference-source` 仅覆盖本次命令，不会改写配置。若选择 `a3_live`，引擎还会执行 A3 容器 mount 检查。

**凭证安全**：上面的 `*_PASSWORD` 只是占位示例。推荐使用密钥认证（留空密码字段，配置 `A5_SSH_KEY` /
`A3_SSH_KEY` 或通用 `SSH_KEY`）或由环境注入凭证，避免明文密码长期留在工作区文件及其备份中；
`.ascendc_env` 含连接信息与凭证，已 gitignore、绝不能提交，并注意文件权限。

## 3. 运行跨代移植

本节命令直接调用底层启动器 `scripts/launch_orchestrator.sh`，适合脚本化与排障；日常在 Claude Code /
OpenCode 会话中使用 `/ascendc-cross-gen-port` 对话入口即可，入口会代你组装同样的启动器调用（见
[`quickstart.md`](../quickstart.md) §4）。`--lane <N>` 指定使用目标机上第 N 块 NPU 设备——用
`npu-smi info` 查看设备列表与占用，选一块空闲设备。

### 3.1 先用仓内示例确认流程

插件内置了一个第一方的 KernelBench 风格 Add task/sidecar 对，位于
`examples/npukernelbench-native/level1/example_add.py` 与同名的 `example_add.json`。它是**原生
`npubench` 输入与评测流程的 smoke/tutorial 示例**：sidecar 故意采用该格式中常见的 JSONL-in-`.json`，task 自己读取
sidecar 并导出 `Model`、`get_input_groups()` 与 `get_init_inputs()`。

```bash
PLUGIN_DIR=/path/to/cannbot-skills/plugins-community/ascendc-port-orchestrator
TASK_ROOT="$PLUGIN_DIR/examples/npukernelbench-native"
TASK="$TASK_ROOT/level1/example_add.py"

bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh"   --skill-base "$PLUGIN_DIR/skills/ascendc-cross-gen-port"   --mode port-a3-ops --source /path/to/arch22/ops-nn/<matching-add-op> --lane 0   --reference-source npubench   --npubench-task "$TASK"   --npubench-root "$TASK_ROOT"
```

把 `<matching-add-op>` 替换为语义与 Add 一致（逐元素加法）的 arch22 `ops-nn` 源目录（获取方式见 §0.1；
ops-nn 仓自带一个最小示例 `examples/add_example`，含 `op_host/`、`op_kernel/`、`op_graph/`，可优先尝试用它
跑通流程）。这个小样例只用于确认**原生文件格式、冻结、
task 加载和评测调用链**；它不是上游 benchmark 语料中的 case，也不能替代代表实际算子的 acceptance golden。

### 3.2 使用自己的原始 KernelBench 风格 task

完成示例验证后，直接把自己的原始 task/sidecar 替换进同一命令，无需任何格式转换：

```bash
PLUGIN_DIR=/path/to/cannbot-skills/plugins-community/ascendc-port-orchestrator
# 你自己的 task 语料根目录（示例结构：<TASK_ROOT>/level1/3_Add.py + 同名 sidecar）
TASK_ROOT=<your-tasks-root>

bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh"   --skill-base "$PLUGIN_DIR/skills/ascendc-cross-gen-port"   --mode port-a3-ops --source /path/to/arch22/ops-nn/<matching-op> --lane 0   --reference-source npubench   --npubench-task "$TASK_ROOT/level1/3_Add.py"   --npubench-root "$TASK_ROOT"
```

**推荐始终显式传入最小 `--npubench-root`**（只含 task、sidecar 与必要 helper 的目录）。`--npubench-root`
可省略，此时根目录为 task 的父目录——若父目录还包含其他任务或构建产物（`build/`、缓存等），整目录闭包冻结会
触发 fail-closed 拒绝（见 §0.1）。task 和 sidecar 必须是同名（仅扩展名不同）的正常文件、位于指定 root 内，不能是 symlink；
缺一个或多个候选 sidecar 都会被拒绝。为保留既有任务的同目录 helper/package import，启动阶段会冻结 root 下所有
普通文件（拒绝 symlink 与特殊文件）并保留原相对路径，后续 resume 只使用冻结副本。

该路径不会转换、重命名或修改 task/sidecar；精度运行其原始 `Model` 与 `get_input_groups()`，并按指定 verifier
优先使用 candidate 的 `get_init_inputs()`（缺失时回退 reference）。性能阶段先由 reference 侧一次性生成输入与
初始化参数并哈希冻结（fixture），reference 与 candidate **共用同一份冻结 fixture**，再由插件内的原生 fixture shim
调用未修改的 `msprof_perf_summary.py --quick --warmup 3 --repeats 5 --keep-prof --device <lane>` 对两侧分别采集，
因此加速比两侧是同一工作负载、可比。每个非空 case 都必须有成对原始 profile；随机输入不会在两侧重新生成。

运行时不会读取 A3 runtime/capture 或 A3 环境变量。`--port-a3-ops` 的源目录仍是待迁移代码范围，而非本路径的功能真值。

### 3.3 TileLang2AscendC 工程来源（port-a3-tilelang2ascendc）

TileLang2AscendC 的完整输出目录是 `model_new_ascendc.py + kernel/`，其中 `kernel/` 至少包含
`CMakeLists.txt`、`register.cpp`、`op_host/` 和 `op_kernel/`。它注册的是 AscendC
`torch.ops.npu.<op>` custom op。此模式与 `port-a3-ops` 只有"待移植实现"的来源形态不同，
**golden 输入完全一致**（KernelBench 风格 task + sidecar）：

```bash
PLUGIN_DIR=/path/to/cannbot-skills/plugins-community/ascendc-port-orchestrator
SOURCE=/path/to/tilelang2ascendc-output/3_Add
TASK_ROOT=/path/to/kernelbench_tasks
TASK="$TASK_ROOT/level1/example_add.py"

timeout --foreground --kill-after=30s 5400s   bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh"   --skill-base "$PLUGIN_DIR/skills/ascendc-cross-gen-port"   --mode port-a3-tilelang2ascendc   --source "$SOURCE"   --lane 0   --cold-start   --reference-source npubench   --npubench-task "$TASK"   --npubench-root "$TASK_ROOT"
```

也可以保留 `--mode port-a3-ops`，并显式增加
`--source-kind port-aclnn-tilelang2ascendc --candidate-kind tilelang2ascendc_custom_op`（`source-kind` 为内部
持久化值，保持原名）；两种写法落入同一个持久化 source kind。启动前必须确认安装健康门输出 `✓ hooks verified live`，认证信息只通过私有环境注入，
不能把 key 写入 `.ascendc_env` 或日志。

关于这条命令的两个细节：

- 外层 `timeout --foreground --kill-after=30s 5400s` 是给长任务加的看门狗（90 分钟），可选；移植是长任务
  （见 §5.3），脚本化运行时建议保留。
- `--cold-start` 表示丢弃已有同名 workspace 从头跑（见 §5.3）；首次运行该算子时可以省略。

## 4. 可选：获取实时 A3 CANN 真值（仅常规 ops-nn 来源）

只有需要本次来源机的 fresh A3 CANN truth 时才使用此路径；它不是 npubench golden 的前置步骤。
TileLang2AscendC 工程来源不支持该路径（引擎拒绝该组合），只使用 npubench golden：

```bash
bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh"   --skill-base "$PLUGIN_DIR/skills/ascendc-cross-gen-port"   --mode port-a3-ops --source /path/to/arch22/ops-nn/<matching-op> --lane 0   --reference-source a3_live
```

`a3_live` 不能与 npubench task 参数混用；这能防止误把冻结 task golden 与 A3 真值混为一谈。
此模式下精度对照当次 A3 实测输出，报告的加速比是目标实现 vs A3 实现实测（见 §0.2）。

## 5. 验证、性能与产物

流水线依次执行 O0→O6：解析、分类、参考采集、移植、A5 构建、精度验证、性能、归档。

- `a3_live`：O2.5 获取本次来源 A3 输出，O5 用既有 A3/A5 精度与性能约束验证候选实现。
- `npubench`：O2.5 只预检冻结的 task bundle；O5 在独立的 npubench 精度/性能通路中运行（远端 A5 的完整性
  核验机制见 §5.1）。
  性能报告必须记录冻结 fixture、每 case profile 覆盖和 W3/R5/`--keep-prof` 合同，任一不完整即不作为 gate PASS。

可在 `engine/workspace/<op>/` 查找 `.opgen_state.json`、冻结 capture、`verification.json`、日志和
`PROGRESS.md`；归档位于 `engine/output/a3_to_a5_port/<op>/`。读取报告时应同时检查 reference source、
precision status、每 case 计数和性能测量状态。

### 5.1 npubench 评测完整性边界

npubench 路径会冻结 task/sidecar、对候选建立内容快照、由 reference 侧一次性生成并哈希 fixture，随后在
独立子进程中进行精度与 W3/R5 性能评测；执行结束后编排器会重新核验 bundle、候选、fixture 和 profile 覆盖。
同时，现有 delegation scanner 会拒绝常见的 Torch/ATen/aclnn 委托及 C++ host fallback。

这些是面向正常 agent 自迭代的完整性保护，不是对同一用户权限下有意恶意 task/candidate 的强隔离或可信 profiler
保证。需要抵御该威胁模型时，应使用独立账号/容器边界和受信采集端。

### 5.2 精度优先模式与性能回补（SKIP_PERF / PERF_BACKFILL）

针对 KernelBench 风格 task 的 npubench 评测支持把 W3/R5 msprof 性能测量从首跑中拆出来：

- **`CANNBOT_NPUBENCH_SKIP_PERF=1`（精度优先）**：跳过性能测量，只银行化精度证据。O5 产出一份
  `status=DEFERRED` 的占位性能报告，报告带 `perf_deferred=true` 标注；真实测量报告则带
  `measurement_completed=true`，并在几何平均加速比低于 1 时标注 `perf_pending_optimization=true`。
  finalize 接受该占位证据并盖 `PARTIAL_PERSIST`（精度已绿、性能待回补）。并行双 lane 同样生效：
  SKIP_PERF 优先于双 lane 并行分支，无论 lease 如何都走 precision-only。
- **`CANNBOT_NPUBENCH_PERF_BACKFILL=1`（事后回补）**：对带 DEFERRED 占位报告的 done 算子，resume 会记录
  一条 done→finalize 的合成转换并重进 finalize，由 O5 在同一候选上补跑 W3/R5 测量并追加进既有证据包。
  **回补前必须 unset `CANNBOT_NPUBENCH_SKIP_PERF`**——否则 O5 只会再产出一份 DEFERRED 占位报告；resume
  在入口处直接拒绝该组合，finalize 契约也会拒收仍处于 DEFERRED（或缺 `measurement_completed=true`）的
  回补结果，不会路由到 done。

```bash
# 首跑：精度优先（快，性能产出 DEFERRED 占位报告）
CANNBOT_NPUBENCH_SKIP_PERF=1 bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh"   --skill-base ... --mode port-a3-ops --source ... --lane <N>   --reference-source npubench --npubench-task ... --npubench-root ...
# 事后回补：resume 重进 finalize 补测性能；必须 unset SKIP_PERF
env -u CANNBOT_NPUBENCH_SKIP_PERF CANNBOT_NPUBENCH_PERF_BACKFILL=1   bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh" <同上参数>
```

### 5.3 中断、续跑与成功验收

- **查看进度**：`engine/workspace/<op>/PROGRESS.md` 与 `.opgen_state.json`（机器可读状态）。
- **中断后续跑**：`bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh" --skill-base <dir> --resume <op>`
  （`<op>` 是 workspace 名），沿用原 workspace 的状态与 source 绑定继续，不会重新猜测真值来源。
- **推翻重跑**：加 `--cold-start`，与 `--resume` 互斥。
- **成功验收**：`engine/workspace/<op>/verification.json` 精度 PASS；性能报告带
  `measurement_completed=true`（除非显式使用了 §5.2 的精度优先模式，此时为 `DEFERRED` 占位、需事后回补）；
  归档位于 `engine/output/a3_to_a5_port/<op>/`。

## 6. 反向生成

反向生成不依赖跨代移植参考。给出定义 `forward(...)` 和 `BACKWARD_SPEC` 的可微 PyTorch 文件即可
（完整格式示例见 `scripts/reference_provider/examples/gelu_spec.py`）：

```text
/ascendc-backward-gen 为 /path/to/forward_spec.py 生成目标 A5 的反向算子
```

对应的启动器手动形态（脚本化/排障用，`--mode backward`）：

```bash
bash "$PLUGIN_DIR/scripts/launch_orchestrator.sh" \
  --skill-base "$PLUGIN_DIR/skills/ascendc-backward-gen" \
  --mode backward --source /path/to/forward_spec.py --lane 0
```

## 7. 常见问题

| 症状 | 处理 |
|---|---|
| 首次运行报 `refusing to spawn unsandboxed` | 缺 `bubblewrap` 或 user namespace 不可用：安装并验证 `bwrap --unshare-user --dev-bind / / true`，见 §1.5 首行。 |
| `npu-smi` 或 `import torch_npu` 报 `liburma.so.0: cannot open shared object file` | 启动前漏了 `export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH`，见 §1.5 末尾。 |
| 启动即因 SoC 校验 fail-closed | `A5_SOC_VERSION` 必须填目标设备 `npu-smi info -t board -i <NPU编号>` 输出中的 `Chip Name`，见 §2。 |
| root 下 Claude Code worker 被拒（`bypassPermissions`） | `IS_SANDBOX=1` 必须 export 到启动器进程环境，见 §2.1（仅 Claude Code）。 |
| 报 `*_SOURCE_STATE_CONFLICT: existing workspace source binding is immutable` | workspace 的 source 绑定不可变。续跑中断的 run 用 `launch_orchestrator.sh --skill-base <dir> --resume <op>`（`<op>` 是 workspace 名）；要推翻重跑才用 `--cold-start`。两者互斥，详见 §5.3。 |
| 推荐 npubench 路径报缺少 task/sidecar | 同时提供 KernelBench 风格 task `.py` 与同名（仅扩展名不同）的 `.json` / `.jsonl`，并显式传 `--reference-source npubench --npubench-task <task.py>`。 |
| A3 mount gate 失败 | 仅在 `a3_live` 路径需要处理；检查 `A3_CONTAINER_HOME` 与实际容器挂载。 |
| NPU 设备错误 | 对齐 task 使用的设备和目标环境。 |
| tilelang2ascendc 工程报 `dynamic module does not define module export function (PyInit__<op>_ext)` | `register.cpp` 缺少 `PYBIND11_MODULE` 入口：仅有 `TORCH_LIBRARY` 注册时，构建出的扩展没有 `PyInit__<op>_ext` 符号。修法：`register.cpp` 增加 `#include <pybind11/pybind11.h>` 和 `PYBIND11_MODULE(_<op>_ext, m) { m.doc() = "..."; }`（模块体可为空，`TORCH_LIBRARY` 静态注册在 load 时完成）；模块名必须与 CMake `OUTPUT_NAME` 一致。受控构建前的 candidate 校验已把此项作为硬性检查，缺 `PYBIND11_MODULE` 会在 build 前直接拒绝。 |
| tilelang2ascendc 工程报找不到 `_<op>_ext` 扩展模块 | `model_new_ascendc.py` 裸 `import _<op>_ext` 在评测隔离舞台里找不到扩展。修法：import 前自带路径引导 `sys.path.insert(0, str(Path(__file__).resolve().parent / "kernel" / "build"))`。 |
| tilelang2ascendc 工程手动验证时 CMake 增量重建报 unknown file type | device obj 二次链接会报 unknown file type。插件的构建路径默认 clean 构建；**手动验证时必须 clean 重建或先删掉 `kernel/build`**，不要在残留的增量 build 目录上排查候选问题。 |
| 大 case 数算子性能阶段报 `isolated task timed out after 300s` | npubench 精度/性能子进程单任务超时默认 300s（`CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC`）。61 个 case 的注意力算子跑 W3/R5 + msprof 超过 300s 属正常：运行前 `export CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC=1200`（或更大）；精度与性能阶段共用该变量，非法值 fail-fast 报错。 |

更多实现边界见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)；最短上手见 [`quickstart.md`](../quickstart.md)。
