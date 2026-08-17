# ascendc-port-orchestrator — 使用说明

跨代际 AscendC 算子移植插件。两项能力，两个入口 skill，共享一条确定性流水线 + 安全网 + 双层 KB 反馈环：
1. **`ascendc-cross-gen-port`** — 跨代际算子移植（当前 arch22→arch35，如 910C/V220 → 950PR/V300）。
2. **`ascendc-backward-gen`** — 正向→反向（梯度）算子生成。

> 两个入口都是**薄 NL 前端**，底下调用打包在 `engine/` 的编排器（`python -m orchestrator`）。流水线逻辑全在引擎、不在入口。
> 支持两种 agent harness：**Claude Code** 与 **opencode**。两者共用同一个启动器
> （`scripts/launch_orchestrator.sh`）、同一条流水线、同一套 canonical 安全网；差异只在安装面，
> 见 §1。harness 相关的实现说明见 `ARCHITECTURE.md §8`。

---

## 0. 前置条件

**共同**
- **CANN Toolkit ≥ 9.0.0**，已配置 NPU 设备（目标芯片可访问）。
- **Python 3.10+**（引擎）。
- arch22→arch35 移植按参考输入选择环境：默认实时 A3 参考（`a3_live`）需要**来源 A3** 和
  **目标 A5**；显式提供 `model.py + test.py`（`model_reference`）时只需要**目标 A5**。
  这份模型在 A5 验证环境中自行决定使用 CPU 或 NPU，不会被隐式派发到 A3。反向生成只需目标 NPU
  （参考真值 = CPU/fp64 autograd）。

**按 harness**

| | Claude Code | opencode |
|---|---|---|
| 运行时 | `claude` CLI | `opencode` CLI（**已验证版本：1.18.18**） |
| 额外依赖 | — | **`node`**（安全网适配器需要） |
| 联网 | 安装期不需要 | 首次安装需拉一次插件依赖（约 1-2 分钟），完全离线需按 §1.3 预置 |

---

## 1. 安装

安装器同一个（`init.sh`），第二个参数选 harness。**两种 harness 的安装面差别很大**，分别说明。

### 1.1 Claude Code

```bash
claude plugin marketplace add /path/to/cannbot-skills
claude plugin install ascendc-port-orchestrator@cannbot

# Claude Code 只复制插件，不会自动执行 init.sh；从安装清单取真实缓存路径
PLUGIN_INSTALL_PATH="$(claude plugin list --json | jq -r \
  '.[] | select(.id == "ascendc-port-orchestrator@cannbot") | .installPath')"
test -n "$PLUGIN_INSTALL_PATH"
bash "$PLUGIN_INSTALL_PATH/init.sh" global claude
```

装了什么：Skills/Agents 以**逐项软链**进 `$CLAUDE_CONFIG_DIR`（默认 `~/.claude`）；安全网 hook 以
marketplace 的 `hooks/hooks.json` 为唯一注册面（源码 checkout 直跑 `init.sh` 时才写 owner-tagged 配置门）。

> **装完必须看到最后一行 `✓ hooks verified live`。** 打勾前安装器会真的执行一次守卫：
> 子 agent 读 `output/` 必须被拒（exit 2）、主 agent 必须放行（exit 0）。没有这一行或退出码非 0，
> **不要开始跑算子** —— 防作弊层没上，产出的"通过"不可信（DEBT-253）。

### 1.2 opencode

```bash
bash /path/to/ascendc-port-orchestrator/init.sh global opencode
# 或项目级：装到 $PWD/.opencode
bash /path/to/ascendc-port-orchestrator/init.sh project opencode
```

装完同样看**最后一行 `✓ safety net ENFORCES`**：安装器会用真实 opencode 二进制证明配置被接受、
守卫真的会拦（kernel-worker 读别的 workspace 被拒、读自己的放行）。没有这一行或退出码非 0，
**不要开始跑算子**。

首次在全新 opencode 环境安装时，opencode 会先在 config 目录做一次插件依赖解析（约 1-2 分钟，
需要访问 npm registry；安装器有进度提示与超时兜底）。

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

就绪检查由 skill/引擎自动做，你只需确认 NPU 连接配置已填：

```bash
# 确认 .ascendc_env 已填（缺它引擎 fail-fast、rc=2 + 提示）
test -f <plugin>/engine/workspace/.ascendc_env && echo "env OK" || echo "先填 .ascendc_env（见 §3）"
```
- 调用入口后，引擎在生成前先跑 **Phase O0 就绪门**（hook 完整性 + KB + deploy 脚本存在 + 安全网重新证明）+ `.ascendc_env` 解析。只有 `a3_live` 路径还会执行 **P129 A3 容器 mount 门**（核对你配置的 `A3_CONTAINER_HOME`）。
- 缺 `.ascendc_env` → 干净报错（不会白跑到 build 才崩）。真正的 NPU 就绪（精度/性能）在实际运行时按阶段判（见 §6）。

---

## 3. 修改配置（`engine/workspace/.ascendc_env`）

引擎的 mode/target/NPU 单一来源。从 `.ascendc_env.template` 拷贝后填：

```ini
TARGET=a5                          # canonical 目标：a5(950PR/arch35) / a3(910C/arch22)
A5_HOST=<A5 NPU host>              # 目标 NPU（生成+验证）
A5_USER=root
A5_PASSWORD=<...>
A5_CONTAINER=<A5 容器名>
A5_CANN_PATH=/usr/local/Ascend/cann-9.0.0
A5_SOC_VERSION=<npu-smi/GetSocName 返回的完整 SoC 字符串>
A3_HOST=<A3 参考 NPU host>          # 仅 reference.source=a3_live 需要（跑 A3-CANN 真值）
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
- `A3_*` 与 `A3_CONTAINER_HOME` 仅在默认的实时 A3 路径（`reference.source=a3_live`）使用；提供 `--reference-model` 与 `--reference-test` 的外部模型参考只要求 A5 连接配置。
- `A3_CONTAINER_HOME` 是**你部署的 A3 容器内 canonical home**（引擎按它拼容器内路径 + 核 P129 mount 门）；不同部署设自己的值。
- `.ascendc_env` 是 **gitignored**（含凭证）——**永不提交**。

---

## 4. 如何做 arch22→arch35 代际移植

**唯一对外接口 = 入口 skill `ascendc-cross-gen-port`**（与 cannbot 使用方式一致）。在 Claude Code 里用自然语言/slash 调用即可（来源架构由代码分析自动识别）：
```
> 把这个算子移植到 arch35（a5）：<来源 AscendC 算子的 ops-nn 目录>
```
例：`把 ~/workspace/cann/ops-nn/activation/gelu 移植到 a5`。skill 内部把请求翻成引擎的 `port_a3_to_a5` mode 并以流式后台启动编排器；下面的 CLI 仅供自动化/排障时确认参数，不需要绕过 skill 手工执行。

### 4.1 选择功能参考

`--port-a3` 始终提供待移植的 arch22 源算子；一对可选的外部文件只改变**功能参考**，不改变
`opgen_mode`。不要传 `--reference-provider` 或新 mode。

| 传递方式 | 内部 `reference.source` | 真值与所需环境 | 性能报告 |
|---|---|---|---|
| 仅提供源算子 | `a3_live` | 当次在 A3 上执行来源 CANN，再在 A5 构建/验证 | 保留实时 A3/A5 的既有性能契约 |
| 同时提供 `model.py` 和 `test.py` | `model_reference` | 在 A5 验证环境执行这对文件；无需 A3，模型代码自行选择 CPU/NPU | `speedup_vs_model_reference`，不称为 A3/A5 ratio |
| 离线 A3 tensor 包 | 预留 `a3_offline_bundle` | **本版本不支持** | 不能冒充实时 A3 结果 |

外部模型参考的自然语言请求示例：

```text
> 把 <ops-nn 源算子目录> 移植到 A5；使用 /work/model.py 作为功能参考，
> 使用 /work/test.py 作为测试用例。
```

对应的启动器参数为：

```bash
bash /path/to/ascendc-port-orchestrator/scripts/launch_orchestrator.sh \
  --skill-base /path/to/ascendc-port-orchestrator/skills/ascendc-cross-gen-port \
  --mode port-a3 --source /path/to/arch22/op --lane 0 \
  --reference-model /work/model.py --reference-test /work/test.py
```

`--reference-model` 和 `--reference-test` 只能成对用于 `--port-a3`；缺一项会在创建 workspace 前报错：

```text
ERROR: --reference-model requires --reference-test. Provide test.py, or explicitly ask the agent to generate test cases before invoking the run.
ERROR: --reference-test requires --reference-model. Provide model.py together with the test suite.
```

默认测试来源是 `user_supplied`。引擎**不会**因缺少 `test.py` 而让 LLM 静默生成用例；只有用户明确授权
“生成 `test.py` 并执行”时，agent 才可先生成普通测试文件，并在启动时额外传
`--reference-test-origin agent_generated`。最终报告会把这类覆盖标为 agent-generated、未由用户确认。

### 4.2 `model.py` 与 `test.py` 的最小契约

- `model.py` 导出 `create_model()`，返回可调用对象；为兼容已有用例，也接受零参数且可调用的 `Model`。
- `test.py` 必须导出 `get_test_cases()`，返回至少一个 case。每个 case 至少有稳定的 `id`，可选 `args`、
  `kwargs`、`comparison`、`weight` 和 `benchmark.accelerator_devices`。后者列出计时期间可能提交异步工作的设备；CPU-only 用例写空列表。
- 输入由用户测试代码决定 dtype、shape、随机种子和设备。引擎不替模型或输入做设备迁移；`model.py` 在 A5 验证环境中自行使用 CPU 或可访问的 NPU。
- 用例与参考输出在 O2.5 一次性物化、稳定性校验并冻结。resume 复用已绑定的 capture，不重新执行 `get_test_cases()` 或重新计算参考真值。

### 4.3 验证、性能与工件

引擎走确定性流水线 **O0→O6**：解析 → 分类 → 参考采集 → 移植 → A5 构建 → 精度验证 → 性能 → 归档。
其中 `a3_live` 在 O2.5 采集实时 A3-CANN 输出；`model_reference` 则采集已暂存模型与用例的 canonical 输出，
O5 只使用该 capture，绝不回退读取 `a3_outputs`。

外部模型参考的性能基线就是该模型本身：在同一个 A5 session 中以 ABBA 顺序测量参考和候选，默认每侧
10 次 warmup、50 个样本，报告中给出 `speedup_vs_model_reference`。第一期不对该 ratio 套用通用数值阈值；
测量完成是性能门，数值供用户判断。若同步、输出一致性或计时失败，报告为 `MEASUREMENT_FAILED` /
`INCOMPLETE_PERFORMANCE`，不产生 ratio，也不伪装成成功交付。

**产出**：`engine/workspace/<op>/verification.json`（customer-view 判据）+ 归档到
`engine/output/a3_to_a5_port/<op>/`（ops-nn 镜像布局）+ 复现指引。外部模型路径还会保留内容寻址的
`reference_inputs/<bundle-digest>/`、冻结 capture 与 `model_reference_performance.json`；实时 A3 路径则保留
对应的 A3 provenance。查看 `precision.status`、逐 case 计数和与所选参考来源一致的证据，不能把
`model_reference` 的结果解读成 `bit_exact_vs_a3`。

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
| `--reference-model requires --reference-test` | 外部模型参考必须由模型和用户测试文件共同定义。补 `test.py`；只有你明确授权 agent 生成用例时，才使用 `--reference-test-origin agent_generated`。 |
| 外部模型参考却要求/连接 A3 | 检查是否同时传入了 `--reference-model` 与 `--reference-test`。两者齐全时为 `model_reference`，只需要 A5 构建/验证；模型中的 CPU/NPU 选择由模型代码负责。 |
| `MEASUREMENT_FAILED` / `INCOMPLETE_PERFORMANCE` | 外部模型的同会话 ABBA 测量、设备同步或输出一致性未完成。精度证据会保留，但无可信 ratio，不能把它当作 release PASS。 |
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
