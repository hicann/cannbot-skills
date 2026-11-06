# 仿真与模型校验

## 选择运行入口

| 需求 | 入口 | 输出范围 |
| --- | --- | --- |
| 按模型固定配置快速运行 | `opmodel <模型目录> --output <新目录>` | 普通模型的准备配置；省略输出参数时写入模型目录的 `output/` |
| 自定义普通模型的完整 case | `run_native_case` 或教程 `scripts/run_case.py` | 全核总体耗时，并导出关键核或各阶段关键核的 trace/CSV |
| 普通模型的双后端/数值检查 | `model.validation.score_case` | 普通聚合范围；数值需声明执行用例，不代替完整 case 分核结果 |
| 声明了 `physical_execution` 的模型 | `opmodel` 或 `run_native_case` | 按物理模型契约执行，可选择支持的后端和 gates |

主线教程使用普通模型的完整 case 入口。普通 `opmodel` 支持 `--output`，
但不接受物理专属的 `--case/--backend/--gates/--trace`。这两个入口的工作量范围不同。

## 普通模型的完整 case

```bash
python "$OPMODEL_SKILL_DIR/scripts/run_case.py" work/my_matmul \
  --case work/my_matmul/case.yaml --output work/new_run
```

也可使用[Python API](python-api.md)自行编写批量运行程序。传入完整 shape，
设置 `derive=True`，由模型计算逐核派生参数。本路径使用 SimPy，当前不接受
`backend="both"`、物理 execution policies 或物理 gates。

普通完整 case 的 `hardware_path` 覆盖目前也不受支持；它使用模型选定的硬件 profile。
固定配置仿真和 DAG 搜索的硬件覆盖方式见[硬件配置](hardware.md)。

## 输出目录

每次实验设置新目录，例如 `work/baseline`、`work/tile64`。教程调用脚本会拒绝
已经存在的输出目录，避免覆盖上一轮结果。底层各 CLI 的覆盖行为可能不同，
因此搜索实验也应使用新的输出根目录或新的 `name`。

普通完整 case 输出 `result.json`、`params.json`、`assignment.json` 及统计文件。
多阶段模型的 trace/CSV 可能位于子目录，应读取 `result.json` 的 `artifacts` 定位。
输出内容解释见[分析结果](analysis.md)。

## 物理模型

物理模型在 `operator.yaml` 中声明 `physical_execution`，还需要明确引擎映射、
存储区域、访问关系和相关约束。主线矩阵乘示例采用普通模型，不应直接加上物理运行参数。

已有完整物理模型时，可通过 `python -m accel_model.app.case.native_case --help`
查看 `--case`、`--hardware`、`--backend`、`--gates` 等参数。`both` 表示对支持的模型
运行两个后端，不表示任意模型都支持双后端，也不等价于设备实测。

发行资料提供可直接运行的[物理执行示例](../assets/examples/physical/README.md)，
覆盖 Cube/双 AIV 交接、尾块、输出区域和声明式 checks，成本为明确标注的合成值。

## 模型错误与数值正确性

加载和编译会检查字段、表达式、容量及依赖等契约。遇到错误时，先修正输入模型，
不要通过删掉模型自身的约束来取得一个周期数。

模型运行完成说明该次路径完成了执行，不自动证明数学结果正确或性能已校准。
声明了数值语义也不代表每个入口都会自动执行数值对拍。需要数值校验时，应使用
模型支持的对应接口和真实输入；相关状态为 `not_run` 时按未执行理解。

数值输入、报告状态与限制见[数值验收接口](framework/numeric-execution-validation.md)。
使用自己的模型与独立参考验证，不用其他模型的通过记录替代本次验收。
