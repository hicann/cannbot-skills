---
name: model-train-log-visualization
description: 当用户提到训练日志作图、loss/grad_norm 曲线、两份日志对比、误差曲线，或需要从 torchtitan/torchtitan-npu 的日志按 step 提取并可视化指标（含 memory/tps/tflops/mfu/elapsed_time_per_step/indexer_loss）时，优先使用本技能；即使用户只说“画日志曲线”“对比两份日志”也应触发。
---

# model-train-log-visualization 技能

用于从训练 stdout 日志中解析指标并绘制可视化曲线。

## 适用场景

- 用户希望从日志文件绘制 `loss`、`grad_norm` 曲线。
- 用户希望在同一张图中对比两份日志（正常 vs 异常）。
- 用户希望追加 `memory`、`tps`、`tflops`、`mfu`、`elapsed_time_per_step`、`indexer_loss` 曲线。
- 用户希望在双日志对比中查看 `loss` 与 `grad_norm` 的绝对误差与相对误差曲线。

## 所需输入

- 主日志路径（必需）
- 对比日志路径（可选）
- 可选指标列表（可空）
- 输出图片路径（可选，默认输出到仓库根目录的 `outputs/`）

## 执行流程

> 回复用户时先说明能力与做法：本技能用绘图脚本按 `step` 解析日志，并绘制 `loss`、`grad_norm` 曲线（双日志还会绘制绝对/相对误差曲线），再确认输入；不要只反问输入。

### Step 1：与用户交互确认输入

按顺序询问：

1. 主日志路径（必填）
2. 是否需要第二份日志做对比（可选）
3. 是否追加可选指标（通过 `--metrics` 追加；可选：`memory`(等价 `memory_gib`)、`memory_pct`、`tps`、`tflops`、`mfu`(等价 `mfu_pct`)、`elapsed_time_per_step`、`indexer`(等价 `indexer_loss`)）。日志中不存在的可选指标会跳过并输出告警，不影响 `loss`/`grad_norm` 主曲线。
4. 输出路径（可选，不填则在与 `.agents` 同级的 `outputs/` 中自动命名）

### Step 2：调用绘图脚本

脚本路径：

- `.agents/skills/model-train-log-visualization/scripts/plot_training_logs.py`

单日志示例：

```bash
python .agents/skills/model-train-log-visualization/scripts/plot_training_logs.py \
  --log-a /path/to/train.log \
  --metrics memory,tps,indexer \
  --no-show
```

> 说明：`memory` 会映射到 `memory_gib`。未指定 `--output` 时，主图自动保存到仓库根目录的 `outputs/`。
> `outputs/` 不预先创建；脚本首次实际保存图片时自动创建，目录已存在时直接复用。

双日志示例：

```bash
python .agents/skills/model-train-log-visualization/scripts/plot_training_logs.py \
  --log-a /path/to/baseline.log \
  --log-b /path/to/problem.log \
  --metrics memory,tps,indexer \
  --baseline b \
  --no-show
```

> 说明：双日志没有共同 step 时，脚本会报错退出，避免生成误导性对比图。

### Step 3：绘制完成后询问是否生成 PR 贴图小文件

- 首次绘图完成后，**必须先询问用户**是否需要额外生成一张小于 `50 KB`（`50,000` 字节）的 PNG 用于 PR 贴图。
- 不得默认生成 PR 贴图小文件。
- 用户确认需要后，再次调用脚本并附加参数：
  - `--generate-pr-image`
  - 可选：`--pr-image-output /path/to/pr_image.png`
- PR 图保持主图自身的宽高比，不固定为 `1024x768`；脚本通过降低 PNG 渲染分辨率将文件严格控制在 `50 KB` 以下。
- 若未提供 `--pr-image-output`，默认与主图输出在同一目录，文件名为：`<主输出文件名>_pr_under_50kb.png`。

PR 贴图小文件示例（在原命令基础上追加）：

```bash
python .agents/skills/model-train-log-visualization/scripts/plot_training_logs.py \
  --log-a /path/to/train.log \
  --metrics memory,tps,indexer \
  --generate-pr-image \
  --no-show
```

### Step 4：返回结果

输出必须包含：

- 主图路径
- （若用户要求）小于 `50 KB` 的 PR 图路径
- 解析到的 step 范围和关键摘要
- 对齐告警（例如双日志 step 不一致）
- 指标缺失告警（若某些可选指标不存在）

## 输出约束

- 所有曲线横轴统一为 `step`。
- 只有独立的 `loss:` 字段是主 loss；`indexer loss:`等等带前缀字段属于 aux loss，必须先遮蔽再提取主 loss。
- 未显式指定输出路径时，将主图和 PR 图保存到与 `.agents` 同级的仓库根目录 `outputs/`。
- 不要预先创建或提交 `outputs/`。实际保存图片前由脚本自动创建输出目录；目录已存在时直接复用。
- 显式指定的输出路径若包含尚不存在的父目录，也由脚本在保存前自动创建。
- 无论单日志还是双日志，`loss` 与 `grad_norm` 都必须绘制。
- 双日志模式下必须额外绘制：
  - `loss abs error`
  - `loss rel error`
  - `grad_norm abs error`
  - `grad_norm rel error`
- 将绝对误差计算为 `abs(comparison - baseline)`；相对误差保留差值方向，并使用基线绝对值归一化。
- 相对误差无法定义时（基线为 `0` 且差值非 `0`，或任一侧为 `NaN`/`Inf`）记为未定义（NaN），在统计中跳过并输出告警，不得按 `0` 处理。
- 双日志中某个共同 step 的 `loss`/`grad_norm` 任一侧缺失或为非有限值时，从对应误差曲线和统计中跳过该 step 并输出告警，不得用 `0` 填充。
