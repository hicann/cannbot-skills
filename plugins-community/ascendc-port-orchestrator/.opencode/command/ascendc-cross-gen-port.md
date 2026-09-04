---
description: 跨代际 AscendC 算子移植入口（arch22→arch35）。参数包括：1） 必选。待port的arch22算子实现目录 2) 推荐。KernelBench风格 (model.py 和test_case.json) 的算子golden与测试集合。
subtask: false
---

# ascendc-cross-gen-port（opencode 入口）

**薄 NL 前端**。移植的全部流水线逻辑在引擎里（`engine/`，`python -m orchestrator`），不在本
command。这里只负责：把用户的话翻成引擎的 `--port-a3-ops` mode 调进去，再回报结果。

与 Claude Code 侧的 `skills/ascendc-cross-gen-port/SKILL.md` **共用同一个启动器**
（`scripts/launch_orchestrator.sh`），两边不各自维护一份启动逻辑。

## 参数

输入统一为两部分：

1. **必选：待移植的算子实现目录**——常规为 arch22 `ops-nn` 源算子目录（如 `~/workspace/cann/ops-nn/activation/gelu`）；
   TileLang2AscendC 工程目录（`model_new_ascendc.py + kernel/`）也可直接作为来源。
2. **KernelBench 风格 golden**——算子实现 `.py`（model.py 风格）+ 测试集合 `test_case.json`
   （同 stem `.json` / `.jsonl` sidecar）文件对；具体格式可查看 `examples/npukernelbench-native/` 示例。
   该输入原样冻结为真值，不做任何格式转换；输入尚非该格式时，请用户先由输入提供方准备为该格式并复核语义。
   常规 ops-nn 来源下为**推荐**（可显式改选 `a3_live`）；TileLang2AscendC 工程来源下为**必需**（唯一真值）。

目标架构可用自然语言一并给出（`arch35` / `950PR` / `A5` / SoC 编号皆可）；来源架构由代码分析自动识别，无需指定。

## 你要做的（三步）

1. **归一目标**：把自然语言目标归一为 canonical target（arch35/950PR/A5/V300 → `a5`；
   arch22/910C/A3/V220 → `a3`）。只归一目标，不猜引擎路径。

2. **启动引擎**（`--port-a3-ops` mode，op 名自动取来源目录 basename）：

   ```bash
   bash @@PLUGIN_DIR@@/scripts/launch_orchestrator.sh \
     --skill-base @@PLUGIN_DIR@@/skills/ascendc-cross-gen-port \
     --mode port-a3-ops \
     --source <ops-nn-op-dir> \
     --lane 0 \
     --reference-source npubench \
     --npubench-task <levelN/task.py> \
     --npubench-root <npu_benchmark-root>
   ```

   来源是 TileLang2AscendC 工程（`model_new_ascendc.py + kernel/`）时，把 `--mode port-a3-ops` 换成
   `--mode port-a3-tilelang2ascendc`，golden 参数不变（两种模式的 golden 相同）。

   用户明确要求实时 A3 CANN 真值时（**仅常规 ops-nn 来源**；TileLang2AscendC 工程来源只支持 npubench
   golden，引擎会拒绝该组合；也可用 `.ascendc_env` 的 `PORT_A3_REFERENCE_SOURCE=a3_live` 等价显式配置），改为：

   ```bash
   bash @@PLUGIN_DIR@@/scripts/launch_orchestrator.sh \
     --skill-base @@PLUGIN_DIR@@/skills/ascendc-cross-gen-port \
     --mode port-a3-ops \
     --source <ops-nn-op-dir> \
     --lane 0 \
     --reference-source a3_live
   ```

   **perf 对比基准差异**：npubench golden 模式下报告的加速比 = 目标实现 vs golden 参考实现（A5 上
   W3/R5 msprof 实测）；a3_live 模式下精度对照当次 A3 实测输出，加速比 = 目标实现 vs A3 实现实测。
   两种模式的加速比基准不同，数值不可直接横向比较。

   启动器会解析引擎根、校验 `engine/workspace/.ascendc_env`、探测 harness 并导出
   `AOG_HARNESS_BACKEND=opencode`，然后 exec 编排器。

   **不要**自己拼 `python -m orchestrator` 命令、不要从当前目录猜引擎位置、不要 `find /`
   搜索。启动器是唯一入口。

   **不要**把输出重定向到文件、也不要经 `tail`/`head`/`grep` 管道 —— 那会把实时日志吞掉，
   用户 console 变黑箱。让它直接输出。

3. **回报结果**：读 `engine/workspace/<op>/verification.json` 的 customer-view 判据
   （`precision.status` + 逐 case 计数 + 与 reference source 一致的证据），连同复现指引一并回报。
   **只有精度验证 PASS 才报成功**，否则如实报告失败原因。

## 流水线（引擎负责，此处仅供你理解进度）

O0 就绪门 → O1 解析 → O1.7 分类 → **O2.5 参考采集**（推荐冻结 KernelBench 风格 task bundle；
仅显式 a3_live 才在来源 NPU 采集 A3-CANN）→ O4 移植 → 构建（A5）→ 精度验证 →
[性能优化] → 归档。
全程经安全网校验，按双层 KB 反馈环（c>b>a）注入/沉淀。

**真值不可豁免**：`npubench` 必须有
内容寻址的原 task/sidecar bundle；a3_live 必须有本次来源 NPU capture provenance。两者均不能以缓存、归档或
已提交输出冒充。
