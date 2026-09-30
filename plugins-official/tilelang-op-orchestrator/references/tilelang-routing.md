# TileLang 本地芯片路由

仅用于选择独立工作流，不提供 TileLang API、设计或测试规则。

先按名称加载 `ascendc-env-check` Skill（仅使用其探测脚本），将该 Skill 的真实目录设为 `ASCENDC_ENV_CHECK_DIR`，运行：

```bash
python3 "$ASCENDC_ENV_CHECK_DIR/scripts/get_npu_arch.py" --json
```

Codex 中设备查询命令使用 `env -u ASCEND_RT_VISIBLE_DEVICES` 前缀，并由宿主以 `sandbox_permissions=require_escalated` 执行。前缀本身不授予权限。其他宿主按其设备访问权限运行。

脚本输出中，完整 SoC 来自 asys / DSMI，NpuArch 来自 asys 或按完整 SoC 精确匹配的 CANN INI。检测器只查询硬件，不调用 AscendC 或 TileLang 开发技能，不安装框架。

`get_npu_arch.py` 是主编排和独立 TileLang Skill 共用的硬件证据来源。主编排在入口运行一次，将返回的完整 JSON 作为 `evidence`，并把由 `full_soc` 与 `npu_arch` 组合确定的 `platform` 与证据传给后续按名称加载的 Skill，此时下游不重复探测。独立调用的 Skill 没有有效平台证据时，同样按名称加载 `ascendc-env-check` 并从其真实目录运行该脚本。不得从当前 Skill 目录拼接跨 Skill 相对路径。平台确定后再按名称加载 `npu-arch` 查询对应架构与硬件能力，不得用静态映射反推本机型号。

输出 JSON 包含 `full_soc`、`npu_arch` 及其来源等完整证据。只有退出码为 0 且下列完整 SoC / NpuArch 组合一致时才可路由：

| 完整 SoC | NpuArch | platform |
|---|---|---|
| Ascend910B1～B4（含 B2C、B4-1）/ Ascend910_93 系列 | 2201 | Ascend910 |
| Ascend950PR / Ascend950DT 系列 | 3510 | Ascend950 |

型号与架构必须一致。Ascend310B 即使产品名含 A2，也不属于这里的 Ascend910 TileLang 工作流。缺信息、未知平台、矛盾信息和查询超时均返回非零退出码，不选择默认分支。检测器使用本地默认设备；若现场设备异构，先明确本次目标设备并核实，不能把默认设备的结果当作其他设备的结果。
