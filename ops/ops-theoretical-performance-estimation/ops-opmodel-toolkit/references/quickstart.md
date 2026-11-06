# 快速入门：运行自己的矩阵乘模型

先完成[安装](install.md)，激活环境，并在用户项目目录执行以下命令。
示例描述 `C = A @ B`，使用 bf16 输入、fp32 累加和 bf16 输出。

## 1. 创建用户模型目录

```bash
mkdir -p work
cp -R "$OPMODEL_SKILL_DIR/assets/examples/matmul" work/my_matmul
```

目标 `work/my_matmul` 必须尚不存在，避免复制成嵌套目录。
现在 `definition/`、`case.yaml` 和 `optimize/` 都属于你的模型，可以直接编辑。

## 2. 运行完整 case

```bash
python "$OPMODEL_SKILL_DIR/scripts/run_case.py" work/my_matmul \
  --case work/my_matmul/case.yaml --output work/baseline
```

`run_case.py` 是随教程提供的公开 API 调用示例，通过已安装的
`run_native_case` 运行模型。这里计算完整输入形状的全核耗时，输出 trace 和统计文件。

`case.yaml` 中的形状为 `M=256, N=256, K=128`，外层输出分块为 `128 × 128`，
共生成四个输出块。每块完成整个 K 方向的归约。完整参数含义见[模型编写](modeling.md)。

## 3. 读取结果

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('work/baseline/result.json').read_text())
print('总周期:', r['total_cycles'])
print('活跃核数:', r['summary']['active_cores'])
print('各核周期:', r['summary']['core_cycles'])
print('主要瓶颈流水线:', r['evaluation']['bottleneck_pipe'])
PY
```

继续查看 `work/baseline/` 中的 `Pipe_statistic.csv`、`Instruction_statistic.csv`
和 `trace.json`。本例通常只有四个活跃核，硬件配置有更多核并不意味着自动产生更多任务。
这些 CSV 和 trace 主要展示关键核，完整 case 的总体信息以 `result.json` 为准。

## 4. 搜索分核方案

```bash
python -m accel_model.app.cli.split_run work/my_matmul \
  --config work/my_matmul/optimize/split.yaml \
  --out work/split --verify-simpy
```

输出位于 `work/split/split_demo/`。`ranked.yaml` 是候选排名，
`split_plan.rank1.yaml` 是第一名可采纳方案，`PLAN.md` 解释各核分配和耗时。
排名可能并列；可行方案不保证优于当前基线。

应用方案并重新运行同一个 case：

```bash
python "$OPMODEL_SKILL_DIR/scripts/run_case.py" work/my_matmul \
  --case work/my_matmul/case.yaml \
  --plan work/split/split_demo/split_plan.rank1.yaml \
  --output work/with_split
```

下一步按[参数寻优](optimization.md)搜索内层分块，再把选中的参数应用到完整 case。
了解每个模型文件的职责，请阅读[编写自己的模型](modeling.md)。
