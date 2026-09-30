---
name: tilelang-op-orchestrator
description: "TileLang 算子开发统一入口。通过本地 NPU 芯片识别路由到完全独立的 Ascend910 或 Ascend950 工作流。"
mode: primary
skills:
  - ascendc-env-check
  - npu-arch
---

# TileLang 算子开发入口

本插件采用统一入口、按芯片分支的架构。`workflows/Ascend910/` 与 `workflows/Ascend950/` 分别定义各平台的工作流、角色和执行顺序；技能正文及其脚本、模板、参考资料按平台隔离，不复用业务规则。

## 识别并路由

1. 先按名称加载 `ascendc-env-check` Skill（仅使用其探测脚本），将该 Skill 的真实目录设为 `ASCENDC_ENV_CHECK_DIR`，按本插件的 [TileLang 芯片识别说明](references/tilelang-routing.md) 运行 `python3 "$ASCENDC_ENV_CHECK_DIR/scripts/get_npu_arch.py" --json`。使用返回的完整 SoC 和 NpuArch 组合校验并路由；选定平台后再按名称加载 `npu-arch` Skill、传入该平台，查询对应架构与硬件能力。Skill 未安装或加载失败时停止并报告；禁止从静态映射、目录名或 `npu-smi` 的 Chip Name 猜测本机型号。
2. 完整读取下表选中工作流，此后只执行该分支规定的结构和顺序，并把平台、完整硬件探测证据、角色名和任务输入传给它调度的子代理。未选中的工作流、技能正文和模板不加载。

| 本地芯片 | 平台 | 工作流 |
|---|---|---|
| Ascend910B 系列 / Ascend910_93 系列，NpuArch=2201 | `Ascend910` | [Ascend910 workflow](workflows/Ascend910/workflow.md) |
| Ascend950PR / Ascend950DT 系列，NpuArch=3510 | `Ascend950` | [Ascend950 workflow](workflows/Ascend950/workflow.md) |

检测失败、信息冲突或型号不支持时，报告检测证据与缺口，不默认选分支。续跑仍需核对本机平台；与已有任务目标冲突时不得直接复用旧结果。

## 路径约定

跨 Skill 调用只使用已安装的 Skill 名称；平台确定后传入 `Ascend910` 或 `Ascend950`，由 Skill 入口选择对应正文。相对文档路径以本文件的真实位置为基准；通过软链接读取时先解析链接。脚本以所属工作流或 Skill 的真实目录定位。安装器生成的入口使用绝对文档路径。算子产物的工作目录是用户启动会话的项目目录，各工作流分别定义 `custom/` 或 `operators/` 产物目录规则，不写入插件的工作流或技能源码目录。

本入口不定义阶段、测试标准、重试或调优策略，也不安装任一 TileLang 框架；这些动作在选中分支后按该分支定义的规则执行。
