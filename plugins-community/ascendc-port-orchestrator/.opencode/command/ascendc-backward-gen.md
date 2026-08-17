---
description: 正向→反向（梯度）算子生成入口。从可微的 PyTorch 正向规格生成 AscendC 反向算子并在 NPU 上验证精度，参数为 forward spec 文件。
subtask: false
---

# ascendc-backward-gen（opencode 入口）

**薄 NL 前端**。生成流水线全部在引擎里（`engine/`，`python -m orchestrator`）。本 command
只负责把请求翻成引擎的 `--backward` mode，再回报结果。

与 Claude Code 侧的 `skills/ascendc-backward-gen/SKILL.md` **共用同一个启动器**
（`scripts/launch_orchestrator.sh`）。

## 参数

`$ARGUMENTS` —— 可微正向规格文件（`.py`），需定义 `forward(...)` 与 `BACKWARD_SPEC` dict。
可另行指定目标芯片（`a3`=910C/V220/arch22，`a5`=950PR/V300/arch35），缺省沿用
`.ascendc_env` 中的 `TARGET`。

## 你要做的（三步）

1. **归一目标**：若用户指定了目标芯片，归一为 canonical target 并写入
   `engine/workspace/.ascendc_env` 的 `TARGET=`。

2. **启动引擎**（`--backward` mode，op 名自动取 `<spec 文件名>_grad`）：

   ```bash
   bash @@PLUGIN_DIR@@/scripts/launch_orchestrator.sh \
     --skill-base @@PLUGIN_DIR@@/skills/ascendc-backward-gen \
     --mode backward \
     --source "$ARGUMENTS" \
     --lane 0
   ```

   启动器负责解析引擎根、校验 `.ascendc_env`、导出 `AOG_HARNESS_BACKEND=opencode`，
   再 exec 编排器。**不要**自己拼命令、不要猜路径、不要重定向或管道化输出。

3. **回报结果**：读 `engine/workspace/<op>/verification.json`，按 `precision.status` +
   逐 case 计数如实回报。**只有精度验证 PASS 才报成功。**

## 与移植模式的关键差异

反向生成的参考真值是**自包含的 CPU/fp64 autograd**（`torch.autograd.grad` over the
forward spec），**不需要来源架构 NPU**；只有构建与精度验证需要目标 NPU。所以
`.ascendc_env` 里只需配好目标侧（`A5_*` 或 `A3_*`，取决于 `TARGET`）。
