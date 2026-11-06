# CLI 参考

先激活安装环境。以下 `python -m` 命令从已安装的 OpModel 包加载模块。

| 入口 | 主要参数 | 用途 |
| --- | --- | --- |
| `opmodel-assets list` | 无 | 查看资源 |
| `opmodel-assets export` | `--output DIR` | 向新目录导出模型和配置 |
| `opmodel MODEL` | `--output DIR`、`--hardware NAME_OR_YAML`、`--trace-slots` | 普通模型的固定配置仿真 |
| `python "$OPMODEL_SKILL_DIR/scripts/run_case.py" MODEL` | `--case FILE --output DIR [--plan FILE]` | 本 skill 教程提供的普通完整 case 调用脚本 |
| `python -m accel_model.app.case.native_case MODEL` | `--case FILE --set K=V --output DIR --trace` | 完整 case CLI，额外选项取决于执行模式 |
| `python -m accel_model.app.cli.split_run MODEL` | `--config FILE --out DIR --verify-simpy` | 分核方案评估或搜索 |
| `python -m accel_model.app.cli.dag_run MODEL` | `--config FILE --shape K=V --phases C1 --top-k N --replay-top N --out DIR --hardware NAME_OR_YAML` | DAG 参数和流水阶段搜索 |

`run_case.py` 是资料中的示例文件，不是额外安装的命令。当前普通 `opmodel` 支持
`--output`，省略时写入模型目录的 `output/`；`--case`、`--backend`、`--gates` 和
`--trace` 属于物理分支。普通完整 case 使用教程脚本或[Python API](python-api.md)。

分核和 DAG 搜索将实验 `name` 追加在 `--out` 后，例如 `--out work/dag` 与
`name: dag_demo` 对应 `work/dag/dag_demo/`。每次运行选择新的目录或实验名。

物理模型的 `--backend` 可选择 `simpy/replay/both`，`--gates` 是逗号分隔的检查名。
这些选项不适用于教程普通矩阵乘模型，具体边界见[仿真说明](simulation.md)。

## 进阶命令

```bash
python -m accel_model.app.cli.graph_search --help
python -m accel_model.app.cli.agent_search --help
```

它们分别提供图搜索和搜索会话，参数与模型支持域各有不同。先查看子命令帮助，
不能把一种搜索的配置直接传给另一种。

目标更新使用 `agent_search set-objective/rescore`，规则登记与搜索使用
`register-policy/search-policy`；规格与输出边界见[目标接口](framework/objectives.md)和
[规则接口](framework/policies.md)。这些子命令不适用于 `dag_run`。

公式生成入口为 `python -m accel_model.kernel.dag.gen MODEL`，要求模型提供
`definition/formula.yaml` 与 `definition/cube.yaml`，运行后会写入 `definition/dag.yaml`。
添加 `--check` 只比较生成结果与现有 DAG。教程模型直接提供 DAG，未包含这两份生成输入。
公式生成入口不提供 `--help` 参数。

使用 `accel_model.optimize.dag.reuse` 时，若出现
`AttributeError: 'tuple' object has no attribute 'name'`，保留版本、输入和报错。
分析本例请使用[周期统计和 trace](analysis.md)。
