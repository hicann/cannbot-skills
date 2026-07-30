# ascendc-port-orchestrator — 使用说明

跨代际 AscendC 算子移植插件。两项能力，两个入口 skill，共享一条确定性流水线 + 安全网 + 双层 KB 反馈环：
1. **`ascendc-cross-gen-port`** — 跨代际算子移植（当前 arch22→arch35，如 910C/V220 → 950PR/V300）。
2. **`ascendc-backward-gen`** — 正向→反向（梯度）算子生成。

> 两个 skill 都是**薄 NL 前端**，底下调用打包在 `engine/` 的编排器（`python -m orchestrator`）。流水线逻辑全在引擎、不在 skill。
> ⚠ 当前依赖 Claude Code 运行时（skill 格式 / hook / sub-agent 经 `claude` 调用）；底座无关适配计划见 `README.md` / `ARCHITECTURE.md §8`。

---

## 0. 前置条件

- **CANN Toolkit ≥ 9.0.0**，已配置 NPU 设备（目标芯片可访问）。
- **Claude Code 运行时**（见上）。
- arch22→arch35 移植还需：来源架构参考 NPU（跑来源 CANN 真值基线）+ 目标架构 NPU（生成 + 精度验证）。反向生成只需目标 NPU（参考真值 = CPU/fp64 autograd）。

---

## 1. 安装

```bash
# 1) 添加 cannbot marketplace 并安装插件及依赖
claude plugin marketplace add /path/to/cannbot-skills
claude plugin install ascendc-port-orchestrator@cannbot

# 2) Claude Code 只复制插件，不会自动执行 init.sh；从安装清单取真实缓存路径
PLUGIN_INSTALL_PATH="$(claude plugin list --json | jq -r \
  '.[] | select(.id == "ascendc-port-orchestrator@cannbot") | .installPath')"
test -n "$PLUGIN_INSTALL_PATH"
bash "$PLUGIN_INSTALL_PATH/init.sh" global claude
```
需要隔离验证或使用非默认 Claude 配置目录时，在上述三条命令中使用同一个
`CLAUDE_CONFIG_DIR`；用户 KB 路径可用 `ASCENDC_PORT_USER_KB` 覆盖。

`init.sh` 会：
1. 将插件内置 Skills/Agents 链接到 Claude 配置目录，并检查 marketplace 提供的 `knowledge-query` 依赖。
2. 验证 marketplace 自动注册的安全网 hooks 及客户 Agent 产物门禁。
3. 建**用户本地 KB(c 层)** 根目录 + 索引，并从模板 scaffold `engine/workspace/.ascendc_env`。

Marketplace 安装只使用插件内的 `hooks/hooks.json` 作为唯一注册面：
`output_read_guard` / `workflow_critic` / `ship_claim_audit` 负责生成期安全网，
`agent-gate-dispatch.py` 按 `agent_type` 路由 worker/optimizer/probe 的产物和进度门禁。
安装器不再把同一组命令复制到用户 settings 和版本化缓存路径，避免每次工具
调用重复执行，也避免升级/卸载后留下失效绝对路径。仅从仓库 checkout 直接执行
`init.sh` 时，才会写入 owner-tagged 的配置门禁。

> **装完请确认最后一行出现 `✓ hooks verified live`。** 安装器在打勾之前会真的执行一次守卫：
> 子 agent 读 `output/` 必须被拒（exit 2）、主 agent 必须放行（exit 0），两条都对才算装好。
> 早期版本只统计装了几个文件就打勾，结果出现过"skills/agents 都在、hook 全程没生效"的**静默
> 缴械安装**（DEBT-253）。**如果这一行没出现或安装以非 0 退出，不要开始跑算子** —— 那说明防作弊
> 层没上，产出的"通过"不可信。

**激活**：插件经 cannbot marketplace 激活（`marketplace.json` 注册插件 + skills）；子 agent（aog-*）经 `plugin.json` 的 `agents` 注册。激活后 Claude 能发现两个入口 skill。

---

## 2. Preflight（运行前就绪检查）

就绪检查由 skill/引擎自动做，你只需确认 NPU 连接配置已填：

```bash
# 确认 .ascendc_env 已填（缺它引擎 fail-fast、rc=2 + 提示）
test -f <plugin>/ascendc-port-orchestrator/engine/workspace/.ascendc_env && echo "env OK" || echo "先填 .ascendc_env（见 §3）"
```
- 调用 skill 后，引擎在生成前先跑 **Phase O0 就绪门**（hook 完整性 + KB + deploy 脚本存在）+ `.ascendc_env` 解析 + **P129 容器 mount 门**（核对目标容器 home 挂载，用你配的 `A3_CONTAINER_HOME`）。
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
A3_HOST=<A3 参考 NPU host>          # 来源 A3 参考（仅 --port-a3 需要，跑 A3-CANN 真值）
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
- `A3_CONTAINER_HOME` 是**你部署的 A3 容器内 canonical home**（引擎按它拼容器内路径 + 核 P129 mount 门）；不同部署设自己的值。
- `.ascendc_env` 是 **gitignored**（含凭证）——**永不提交**。

---

## 4. 如何做 arch22→arch35 代际移植

**唯一对外接口 = 入口 skill `ascendc-cross-gen-port`**（与 cannbot 使用方式一致）。在 Claude Code 里用自然语言/slash 调用即可（来源架构由代码分析自动识别）：
```
> 把这个算子移植到 arch35（a5）：<来源 AscendC 算子的 ops-nn 目录>
```
例：`把 ~/workspace/cann/ops-nn/activation/gelu 移植到 a5`。skill 内部把请求翻成引擎的 `port_a3_to_a5` mode 并以流式后台启动编排器 —— **你不需要、也不应直接跑引擎命令**（`bash orch` 是 skill 内部实现细节，非对外接口）。

引擎走确定性流水线 **O0→O6**：解析 → 分类 → **A3-CANN 参考采集**（在 A3 NPU 跑来源算子取真值基线）→ 移植 → 构建（A5）→ **精度验证**（真值 = A3-CANN 输出、非 CPU-PyTorch）→ [性能优化] → 归档，经安全网校验。

**产出**：`engine/workspace/<op>/verification.json`（customer-view 判据）+ 归档到 `engine/output/a3_to_a5_port/<op>/`（ops-nn 镜像布局）+ 复现指引。看 `verification.json` 的 `precision.status`（PASS / PASS_WITHIN_TOLERANCE）+ 逐 case 计数 + `bit_exact_vs_a3` 证据。

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

- **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — 架构设计（cannbot 插件）：总体结构 / 确定性流水线 FSM（§2）/ Agent 角色（§4）/ **双层 KB + 社区 skills 知识源（§5，含 §5.4 统一分层契约 + §5.5 实现状态）** / 安全网 gates（§6）/ 跨代际可扩展（§7）/ 底座依赖 + 适配计划（§8）/ 集成装配（§10）。
- **[`README.md`](../README.md)** — 一页概览 + 底座依赖与适配路线。
- **[`quickstart.md`](../quickstart.md)** — 最短上手。
- **KB 分层统一设计**（跨 cannbot / a5_ops / npu-autoport 三方共设计的 canonical 契约）：a5_ops `docs/design/KB_TIERING_DESIGN.md`（`ARCHITECTURE.md §5.4` 引它）。

> 遇到本说明未覆盖的问题，先查 `ARCHITECTURE.md` 对应节，再看 `engine/workspace/<op>/` 的 `PROGRESS.md` + 日志。
