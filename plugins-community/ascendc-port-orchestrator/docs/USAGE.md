# ascendc-port-orchestrator — 使用说明

跨代际 AscendC 算子移植插件。两项能力，两个入口 skill，共享一条确定性流水线 + 安全网 + 双层 KB 反馈环：
1. **`ascendc-cross-gen-port`** — 跨代际算子移植（当前 arch22→arch35，如 910C/V220 → 950PR/V300）。
2. **`ascendc-backward-gen`** — 正向→反向（梯度）算子生成。

> 两个入口都是**薄 NL 前端**，底下调用打包在 `engine/` 的编排器（`python -m orchestrator`）。流水线逻辑全在引擎、不在入口。
> 支持两种 agent harness：**Claude Code** 与 **opencode**。harness 是运行在**控制端**上的
> Agent 运行/调度程序，并不是 A3 或 A5 的算子运行环境。两者共用同一个启动器
> （`scripts/launch_orchestrator.sh`）、同一条流水线、同一套 canonical 安全网；差异只在安装面，
> 见 §1。harness 相关的实现说明见 `ARCHITECTURE.md §8`。

---

## 0. 前置条件

### 控制端（插件 Agent 运行环境）与 harness

先区分“谁生成/编排代码”和“代码在哪里构建、验证”这两件事：

```text
用户在控制端发起请求
        │
        ▼
控制端：插件目录 + Python 编排器 + Agent harness（Claude Code 或 OpenCode）
        │  解析请求、调度 Agent、生成/整理候选代码、保存工作区、SSH/SCP 传输
        ├──────────────────────────► A3 参考环境：执行来源算子的实时 CANN 参考
        └──────────────────────────► A5 验证环境：构建候选算子，执行精度与性能验证
```

| 概念 | 是什么 | 负责什么 | 不负责什么 |
|---|---|---|---|
| **控制端** | 你执行安装命令和入口 skill 的笔记本、服务器或容器；也是插件、Python 编排器和 Agent 的运行环境 | 接收任务、让 Agent 生成/修改候选 AscendC 代码、驱动流水线、同步文件并汇总工件 | 默认不在本地编译或运行 NPU 算子；不替代 A3/A5 的 CANN 验证环境 |
| **harness** | 控制端内实际承载/调度 Agent 的 CLI：Claude Code 或 OpenCode | 把编排器派发的 Agent 任务交给模型，返回文本、工具调用和 skill 证据 | 不是一台 NPU 主机，不执行 A3/A5 上的编译、精度或性能测试 |
| **实际执行环境** | `.ascendc_env` 配置的 A3/A5 目标主机（默认在对应容器中执行） | 在 A3 跑来源参考；在 A5 构建候选实现并做精度/性能验证 | 不由 `.ascendc_env` 配置 Agent；即使物理同机，也要把 Agent 会话与验证证据作为不同角色处理 |

> 三种角色是**逻辑角色**。实验环境可以把控制端与某台 NPU 主机放在同一物理机上，但仍要分别满足控制端的
> harness/认证前置条件和 A3/A5 的 NPU/CANN/容器前置条件；不能因为本机能运行 Agent，就假定它已经是可验证算子的环境。

- 控制端需要 **Bash** 与 **Python 3.10+**；引擎和安装器都调用 `python3`。插件安装后会以软链、manifest 和命令文件引用实际插件目录，因此不要移动或删除该目录。
- 控制端需要可用的 SSH/SCP。使用密码认证时还需要 `sshpass`；使用默认 SSH 配置或 `A3_SSH_KEY` / `A5_SSH_KEY`（也可用通用 `SSH_KEY`）时不需要密码认证。
- **Claude Code**：控制端需要可调用的 `claude` CLI（可用 `CLAUDE_BIN` 改名），并已完成认证且有可调用模型。安装器不验证认证状态。
- **OpenCode**：控制端需要可调用的 `opencode` CLI（可用 `AOG_OPENCODE_BIN` 改名）、**`node` 或 `bun`**（node 优先），并已在 OpenCode 中配置可用的 provider/model。安装器不会验证模型认证状态。

| | Claude Code | OpenCode |
|---|---|---|
| 安装来源 | Claude marketplace 或完整 checkout | **仅完整 `cannbot-skills` checkout**；不读取 Claude marketplace cache |
| 运行时 | `claude` CLI | `opencode` CLI + node/bun |
| 子 agent 模型 | Claude Code 自身设置 | OpenCode 自身配置；需要固定时可设 `AOG_OPENCODE_MODEL*` |
| 联网 | marketplace 安装按 Claude Code 自身需求 | 新配置首次解析插件依赖需要 npm registry；离线见 §1.3 |

OpenCode **1.18.18 是已验证的建议线，不是硬版本门**：低版本、无法解析版本或版本查询失败只产生 warning。可执行文件缺失、node/bun 缺失或安全网行为探针失败才会阻断 dispatch。

### 实际执行环境

算子不会在控制端“就地”构建；控制端只生成/编排候选代码并收集结果。引擎通过
`engine/workspace/.ascendc_env` 连接实际的目标主机/容器，并使用 SSH/SCP 与
`docker exec` / `docker cp` 把工作交给相应环境。请在这些环境准备可用的 NPU、CANN、Python
和容器执行权限：

- **A3 参考环境（跨代移植）**：运行来源 arch22 算子的实时 A3-CANN 参考，产出本次任务的真值证据。
- **A5 验证环境（跨代移植）**：接收候选实现，完成 arch35/A5 构建、精度验证和性能测量。
- **跨代移植**当前仅支持 arch22 源算子 → arch35/A5：源目录必须包含 `op_host/` 与 `op_kernel/`，且检测结果必须是 arch22。流程强制使用上述实时 A3-CANN 参考并在独立 A5 上构建/验证，因此 **A3 与 A5 都是硬前置**。
- **反向生成**需要一个定义 `forward(...)` 与 `BACKWARD_SPEC` 的可微 PyTorch 规格，以及所选目标 NPU；不需要 A3 参考环境。运行该规格的环境需要具备 PyTorch。

---

## 1. 安装

安装器同一个（`init.sh`），第二个参数选 harness。**两种 harness 的安装面差别很大**，分别说明。

### 1.1 Claude Code

```bash
claude plugin marketplace add /path/to/cannbot-skills
claude plugin install ascendc-port-orchestrator@cannbot

# Claude Code 只复制插件，不会自动执行 init.sh；从安装清单取真实缓存路径
PLUGIN_INSTALL_PATH="$(claude plugin list --json | python3 -c '
import json, sys
plugins = json.load(sys.stdin)
print(next(p["installPath"] for p in plugins
           if p.get("id") == "ascendc-port-orchestrator@cannbot"))
')"
test -n "$PLUGIN_INSTALL_PATH"
bash "$PLUGIN_INSTALL_PATH/init.sh" global claude --strict-deps
```

装了什么：Skills/Agents 以**逐项软链**进 `$CLAUDE_CONFIG_DIR`（默认 `~/.claude`）；安全网 hook 以
marketplace 的 `hooks/hooks.json` 为唯一注册面（源码 checkout 直跑 `init.sh` 时才写 owner-tagged 配置门）。

> **安装输出中必须出现 `✓ hooks verified live`。** 打勾前安装器会真的执行一次守卫：
> 子 agent 读 `output/` 必须被拒（exit 2）、主 agent 必须放行（exit 0）。没有这一行或退出码非 0，
> **不要开始跑算子** —— 防作弊层没上，产出的"通过"不可信（DEBT-253）。

`--strict-deps` 把缺少 CLI 的默认 warning 升级为安装期失败，避免“装成功、第一次 dispatch 才失败”。隔离安装时，marketplace add/install/list/init 必须使用同一个 `CLAUDE_CONFIG_DIR`。

### 1.2 opencode

```bash
REPO_ROOT=/path/to/cannbot-skills
PLUGIN_DIR="$REPO_ROOT/plugins-community/ascendc-port-orchestrator"
bash "$PLUGIN_DIR/init.sh" global opencode --strict-deps
# 或项目级：装到 $PWD/.opencode
cd /path/to/your-operator-project
bash "$PLUGIN_DIR/init.sh" project opencode --strict-deps
```

以上路径必须来自**完整的 `cannbot-skills` checkout**。OpenCode 安装不会读取 Claude Code 的
marketplace cache（也不会读取 `~/.claude`）；因此不能从 Claude marketplace 的插件缓存目录直接执行
OpenCode 安装，缺少共享 skills 时会明确失败并提示回到 checkout。

安装输出中必须出现 `opencode resolves injected agents + skills (structural)` 与
`✓ safety net ENFORCES`：前者证明 OpenCode 接受了私有注入的入口/agent 配置，后者证明守卫真的会拦
（kernel-worker 读别的 workspace 被拒、读自己的放行）。没有这些标志或退出码非 0，**不要开始跑算子**。

首次在全新 opencode 环境安装时，opencode 会先在 config 目录做一次插件依赖解析（约 1-2 分钟，
需要访问 npm registry；安装器有进度提示与超时兜底）。

运行时会在首次 dispatch 前记录 OpenCode 版本与执行安全网行为探针：`1.18.18` 是已验证的建议线，
低版本、无法解析版本或版本查询失败都会给出 warning，但不会仅因此阻断；
`AOG_OPENCODE_MIN_VERSION` 可显式调整该 warning 建议线。可执行文件缺失和安全网探针失败仍会阻断。
流式 watchdog 默认使用
`AOG_STREAM_SILENCE_TIMEOUT_SEC`（默认 1800 秒），需要时可用
`AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC` 覆盖 OpenCode 单独的值。仅在测试或短时排障中才可设置
`AOG_OPENCODE_SKIP_RUNTIME_CHECK=1`，它会跳过该 fail-closed 门，不能作为常规运行配置。
OpenCode 会设置名为 `CLAUDE_PLUGIN_ROOT` 的兼容性环境变量，使已有 agent prompt 能定位本插件；该变量
不依赖 Claude Code，也不会读取 `~/.claude`。

> 安装面的实现细节（两级证明、agent/适配器经 `OPENCODE_CONFIG_CONTENT` 注入、与 CC 的差异、
> 模型驱动端到端验证）见 `ARCHITECTURE.md §8.2`。

### 1.3 离线 / 受限网络环境

opencode 首次安装需要一次性拉取插件依赖（npm registry）。完全离线时的处理方式：先在一台联网机器上
跑一次 `init.sh … opencode` 并触发一次 `opencode run`，把生成的
`$CONFIG_ROOT/{node_modules,package.json,package-lock.json}` 搬到目标机器；或在目标机器预置 npm 镜像。

### 1.4 两者共用的部分

`init.sh` 还会（与 harness 无关）：建**用户本地 KB(c 层)** 根目录 + 索引、构建官方 OKF 索引、
从模板 scaffold `engine/workspace/.ascendc_env`。

## 2. Preflight（运行前就绪检查）

安装器会 scaffold 配置文件；**文件存在不等于环境已就绪**。先确认你编辑的是实际插件目录中的文件，再按 §3 填完所需字段与认证：

```bash
# 只检查文件存在；字段、连接、NPU 与容器权限仍需按 §3 准备。
test -f <plugin>/engine/workspace/.ascendc_env && echo "config file exists" || echo "先运行 init.sh 并填写 .ascendc_env（见 §3）"
```
- 调用入口后，引擎在生成前先跑 **Phase O0 就绪门**（hook 完整性 + KB + deploy 脚本存在 + 安全网重新证明）+ `.ascendc_env` 解析。跨代移植还会执行 **P129 A3 容器 mount 门**（核对你配置的 `A3_CONTAINER_HOME`）。
- 缺 `.ascendc_env` → 干净报错（不会白跑到 build 才崩）。真正的 NPU 就绪（精度/性能）在实际运行时按阶段判（见 §6）。

---

## 3. 修改配置（`engine/workspace/.ascendc_env`）

引擎的 mode/target/NPU 单一来源。从 `.ascendc_env.template` 拷贝后填：

> 此文件**只配置 A3/A5 算子执行端**；Claude Code/OpenCode、插件目录和 Agent 认证属于控制端，
> 不在这里配置。

```ini
TARGET=a5                          # canonical 目标：a5(950PR/arch35) / a3(910C/arch22)
A5_HOST=<A5 NPU host>              # 目标 NPU（构建、精度验证、性能测量；不运行 Agent）
A5_USER=root
A5_PASSWORD=<...>
A5_CONTAINER=<A5 容器名>
A5_CANN_PATH=/usr/local/Ascend/cann-9.0.0
A5_SOC_VERSION=<npu-smi/GetSocName 返回的完整 SoC 字符串>
A3_HOST=<A3 参考 NPU host>          # 跨代移植：跑 A3-CANN 真值；不运行 Agent
A3_USER=root
A3_PASSWORD=<...>
A3_CONTAINER=<A3 容器名>
A3_CANN_PATH=/usr/local/Ascend/cann-9.0.0
A3_SOC_VERSION=<npu-smi/GetSocName 返回的完整 SoC 字符串>
A3_DEFAULT_NPU_ID=0
A3_CONTAINER_HOME=<A3 容器内 canonical home>   # config-driven；容器内工作路径
NPU_PYTHON_BIN=<包含 python3 的目录；留空则使用 PATH>
```
- **`opgen_mode` 不在这里配**——由 CLI flag 决定（`--port-a3` → `port_a3_to_a5`；`--backward` → `backward`）。
- 跨代移植会使用 `A3_*` 与 `A3_CONTAINER_HOME` 采集当次实时 A3-CANN 真值；A5 字段用于构建和验证目标算子。
- `A3_CONTAINER_HOME` 是**你部署的 A3 容器内 canonical home**（引擎按它拼容器内路径 + 核 P129 mount 门）；不同部署设自己的值。
- `.ascendc_env` 是 **gitignored**（含凭证）——**永不提交**。

---

## 4. 如何做 arch22→arch35 代际移植

**唯一对外接口 = 入口 skill `ascendc-cross-gen-port`**（与 cannbot 使用方式一致）。在 Claude Code 里用自然语言/slash 调用即可（来源架构由代码分析自动识别）：
```
> 把这个算子移植到 arch35（a5）：<来源 AscendC 算子的 ops-nn 目录>
```
例：`把 ~/workspace/cann/ops-nn/activation/gelu 移植到 a5`。skill 内部把请求翻成引擎的 `port_a3_to_a5` mode 并以流式后台启动编排器；下面的 CLI 仅供自动化/排障时确认参数，不需要绕过 skill 手工执行。

### 4.1 实时 A3-CANN 参考

`--port-a3` 提供待移植的 arch22 源算子。跨代移植会在当次任务中于来源 A3 执行 CANN 参考，再在独立
A5 环境构建、验证和测量目标实现；两端都必须按 §3 配置完成。不要绕过入口 skill 手工拼装临时真值。

自动化或排障时可使用统一启动器：

```bash
bash /path/to/ascendc-port-orchestrator/scripts/launch_orchestrator.sh \
  --skill-base /path/to/ascendc-port-orchestrator/skills/ascendc-cross-gen-port \
  --mode port-a3 --source /path/to/arch22/op --lane 0
```

### 4.2 验证、性能与工件

引擎走确定性流水线 **O0→O6**：解析 → 分类 → 实时 A3 参考采集 → 移植 → A5 构建 → 精度验证 → 性能 → 归档。
O2.5 为本次任务采集 A3-CANN 输出，O5 以该输出执行既有 two-tier 精度验证。性能测量保持 A3/A5 的既有
对称性与同 session 约束；若不具备可信的测量条件，不能把结果标为性能通过。

**产出**：`engine/workspace/<op>/verification.json`（customer-view 判据）+ 归档到
`engine/output/a3_to_a5_port/<op>/`（ops-nn 镜像布局）+ 复现指引及 A3 provenance。查看
`precision.status`、逐 case 计数和 A3 参考证据后再判定结果。

---

## 5. 如何做反向（梯度）逻辑生成

**唯一对外接口 = 入口 skill `ascendc-backward-gen`**。在 Claude Code 里用自然语言调用，由一个**可微 PyTorch 正向规格**自动生成反向算子：
```
> 为这个正向算子生成反向（目标 a5）：<forward_spec.py>
```
例：`为 ../scripts/reference_provider/examples/gelu_spec.py 生成反向（目标 a5）`。skill 内部把请求翻成引擎的 `backward` mode（op 名取 `<spec 文件名>_grad`）并流式启动编排器 —— **你不需要、也不应直接跑引擎命令**。

**正向 spec 格式**（示例见 `scripts/reference_provider/examples/gelu_spec.py`）：一个 `.py` 定义可微 `forward(**inputs)` + `BACKWARD_SPEC = {"wrt":[...], "inputs":{name:{"shape":[...]}}, "cases":[...], "dtypes":[...], "seed":N}`。

引擎自动：由正向 spec 用 `torch.autograd.grad`（CPU/fp64）生成**精确梯度真值** → 生成反向 AscendC kernel → 构建 → 精度验证（对真值）→ 报告。**产出**：`engine/workspace/<op>_grad/verification.json`。

---

## 6. 如何维护本地用户知识库（双层 KB）

> ⚠ **实现状态**：双层 KB 与社区 skills 知识源的设计见 `ARCHITECTURE.md §5.4/5.5`；**c 层写入的引擎实现正在做**（当前引擎沉淀写打包 b 层，属已知 gap，见 §5.5）。本节按**设计**写，实现落地后本节更新。

**双层 KB（c > b）+ 社区 skills 知识源（a）**：
- **c = 用户本地 KB**：`~/.ascendc-port/user_kb/`（或 `ASCENDC_PORT_USER_KB`），**运行时可写**、最高优先（覆盖默认）。安装时建根 + `INDEX.md`。
- **b = 插件自带官方 KB**：随插件交付（`references/`），运行时**只读**、发版控制（`OL-N/EC-N/PB-N/P-Pxx` 序号）。
- **a = 社区 skills**：CBA 路由。

**维护操作（设计）**：
- **看**：`cat ~/.ascendc-port/user_kb/INDEX.md`（topic 路由）+ 各条目文件。
- **反馈环（越用越聪明）**：算子生成闭合（精度 PASS）后，「生成后沉淀」把新经验写入 **c 层**（不动 b）；c 条目 ID = `customer:{内容哈希}`（同教训同 ID 幂等），和 b 的序号**不同 namespace → 升级不撞**。
- **手工增删改**：直接编辑 c 层文件 + 更新 `INDEX.md`；错条目可删（c 是你本地的）。
- **回到官方**：`reset-to-official`（读取逃生阀，跳过 c、只用 b>a）。
- **升级安全**：插件发新版只换 b 层、不碰你的 c 层 → 你累积的学习不丢、id 不撞。

---

## 7. FAQ / Debug

| 症状 | 原因 / 排查 |
|---|---|
| `ERROR: failed to load .ascendc_env` (rc=2) | 没填 `.ascendc_env`。跑 `init.sh` scaffold 后填连接信息（§3）。 |
| `P129 mount gate FAILED: /home/... ← ''` | 目标 A3 容器的 home 挂载和 `A3_CONTAINER_HOME` 对不上。要么按提示重建容器挂载，要么改 `.ascendc_env` 的 `A3_CONTAINER_HOME` 匹配实际。 |
| NPU errcode（507035/507057 等） | **不等于硬件坏、别急着 reboot**。按顺序查：容器设备映射 → 权限 → `torch.npu.is_available()`。 |
| 精度 FAIL | 读 `verification.json` 的 per-case 输出定位（哪个 output / dtype / case）。**别急着标 hw-floor**——先做 apples-to-apples probe（对齐输入/dtype/参考）。超越函数按 KB 的 OL-103（Rsqrt/Sigmoid ~fp16、用 Sqrt+scalar / Newton-Raphson）。 |
| 反向 fp16/bf16 精度可疑 | 已知 grader 项（cause_1）：backward 的 golden 见 fp32 输入、kernel 见量化输入 → fp16/bf16 可能 false-FAIL（非真错、待 DEBT 修）。fp32 结果为准；fp16/bf16 标已知项、别误判 regression。 |
| 近零 MARE / 退化 FAIL | 竞品(torch fp32)==golden(fp64) → ratio=inf 的退化 false-FAIL。已由 cannbot 精度 adapter 路由到绝对阈值判（`precision_cannbot_adapter`）。若仍见，报 issue。 |
| run 结束 exit 非 0 但 `verification.json` 是 PASS | 可能崩在 finalize/打包步（非 kernel）。**看 `verification.json` 的 customer-view 判据、别看 exit code**；kernel 已生成+验证。 |
| 产物在哪 | workspace：`engine/workspace/<op>/`（`verification.json`/`PROGRESS.md`/`kernel/`/日志 `.opgen.log`）。归档：`engine/output/`。run log：`orch` 打印的 `/tmp/orch_*.log`。 |
| 长跑中断 | 再次用 skill 对同一算子发起请求即可续跑（引擎内部 `--resume`：读状态文件从中断处继续，不重头跑）。 |

---

## 8. 详细设计参考

本使用说明是操作向导；系统的**详细架构设计**（为什么这么设计、约束、不变量）见：

- **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — 架构设计（cannbot 插件）：总体结构 / 确定性流水线 FSM（§2）/ Agent 角色（§4）/ **双层 KB + 社区 skills 知识源（§5，含 §5.4 统一分层契约 + §5.5 实现状态）** / 安全网 gates（§6）/ 跨代际可扩展（§7）/ Harness 抽象与 opencode 安装面细节（§8）/ 集成装配（§10）。
- **[参考输入设计](./design/reference-inputs/design.md)** — 真值来源与测试来源的边界设计。
- **[`README.md`](../README.md)** — 一页概览 + 底座依赖与适配路线。
- **[`quickstart.md`](../quickstart.md)** — 最短上手。
- **KB 分层统一设计**：KB 分层契约的 canonical 设计文档（`ARCHITECTURE.md §5.4` 引用）。

> 遇到本说明未覆盖的问题，先查 `ARCHITECTURE.md` 对应节，再看 `engine/workspace/<op>/` 的 `PROGRESS.md` + 日志。
