---
description: 跨代际 AscendC 算子移植入口（arch22→arch35）。把一个 AscendC 算子从来源架构移植到指定目标架构/产品，参数为来源算子的 ops-nn 目录。
subtask: false
---

# ascendc-cross-gen-port（opencode 入口）

**薄 NL 前端**。移植的全部流水线逻辑在引擎里（`engine/`，`python -m orchestrator`），不在本
command。这里只负责：把用户的话翻成引擎的 `--port-a3` mode 调进去，再回报结果。

与 Claude Code 侧的 `skills/ascendc-cross-gen-port/SKILL.md` **共用同一个启动器**
（`scripts/launch_orchestrator.sh`），两边不各自维护一份启动逻辑。

## 参数

`$ARGUMENTS` —— 来源算子的 ops-nn 目录，例如 `~/workspace/cann/ops-nn/activation/gelu`。
目标架构可用自然语言一并给出（`arch35` / `950PR` / `A5` / SoC 编号皆可）；来源架构由代码
分析自动识别，无需指定。

## 你要做的（三步）

1. **归一目标**：把自然语言目标归一为 canonical target（arch35/950PR/A5/V300 → `a5`；
   arch22/910C/A3/V220 → `a3`）。只归一目标，不猜引擎路径。

2. **启动引擎**（`--port-a3` mode，op 名自动取来源目录 basename）：

   ```bash
   bash @@PLUGIN_DIR@@/scripts/launch_orchestrator.sh \
     --skill-base @@PLUGIN_DIR@@/skills/ascendc-cross-gen-port \
     --mode port-a3 \
     --source "$ARGUMENTS" \
     --lane 0
   ```

   启动器会解析引擎根、校验 `engine/workspace/.ascendc_env`、探测 harness 并导出
   `AOG_HARNESS_BACKEND=opencode`，然后 exec 编排器。

   **不要**自己拼 `python -m orchestrator` 命令、不要从当前目录猜引擎位置、不要 `find /`
   搜索。启动器是唯一入口。

   **不要**把输出重定向到文件、也不要经 `tail`/`head`/`grep` 管道 —— 那会把实时日志吞掉，
   用户 console 变黑箱。让它直接输出。

3. **回报结果**：读 `engine/workspace/<op>/verification.json` 的 customer-view 判据
   （`precision.status` + 逐 case 计数 + `bit_exact_vs_a3` 证据），连同复现指引一并回报。
   **只有精度验证 PASS 才报成功**，否则如实报告失败原因。

## 流水线（引擎负责，此处仅供你理解进度）

O0 就绪门 → O1 解析 → O1.7 分类 → **O2.5 A3-CANN 参考采集**（在来源 NPU 跑真值基线）→
O4 移植 → 构建（A5）→ 精度验证（真值 = A3-CANN 实测输出，非 CPU-PyTorch）→
[性能优化] → 归档。全程经安全网校验，按双层 KB 反馈环（c>b>a）注入/沉淀。

**真值不可豁免**：每次调用都必须当次在来源 NPU 执行并生成与本次调用绑定的 capture
provenance；不能复用缓存、归档或已提交的输出。
