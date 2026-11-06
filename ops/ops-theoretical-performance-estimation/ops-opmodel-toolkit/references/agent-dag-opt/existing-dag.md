# 生产 DAG 入口

从原始算子目录保存配置与源码签名，提取总 shape、输出与 state 语义、实际硬件和当前 split。六份固定 YAML 位于 `<算子目录>/definition/`，搜索声明位于 `<算子目录>/optimize/search.yaml`，实验配置位于 `<算子目录>/optimize/experiments/`；同时支持根目录 YAML 布局。所有工具均传入**算子根目录**，不能传 `definition/` 或 `optimize/`。同一算子不能混用两种布局。

先在不依赖 `optimize/` 的固定配置入口复现生产 case，保存有效总 shape、分核策略、逐核任务量、最重核 trace 和源码身份。默认 shape 可运行：

```text
python -m accel_model.app.case.native_case <算子目录> --output work/<基线目录>
```

该 CLI 使用算子默认 shape；其他总 shape 使用算子 `run.py` 或 `accel_model.app.case.native_case.run_native_case(..., shape=..., output_dir=...)`。逐核评分需要可用的 workload 推导和分核投影；缺失时保留拒绝原因，不能用单核 workload 冒充总 shape。生产基线是每轮保留的对照，不因候选更快而从报告中消失。

`search.yaml` 声明合法 tile、映射和分核动作。先用 `--actions` 查看单变量动作，再用 `--apply-action` 将所选动作物化为独立候选；每次记录父签名、trace 证据、预期收益和新增代价。候选目录须放在源算子目录之外，且在执行前不存在；物化产物保存七份 YAML、所需算子 Python 运行文件和 `candidate.json`。例如：

```text
python -m accel_model.app.cli.graph_search --actions <算子目录>
python -m accel_model.app.cli.graph_search --apply-action <算子目录> \
  --action-id <动作标识> --evidence <trace证据> \
  --expected-gain <预期收益> --added-cost <新增代价> \
  --output work/<候选目录>
```

需要比较已物化为原生 YAML 的配置时，在 `optimize/experiments/experiment.yaml` 声明 `baseline`、候选目录、总 shape case 与权重、`cache_model: shared_estimated`、任务侧 `objective`；`weighted_geomean_ratio` 与 `max_case_regression` 仅在选择兼容目标时使用，外部 provider 见[目标接口](../framework/objectives.md)。`baseline.path` 和 `candidates[].path` 相对**实验 YAML 所在目录**解析；指向算子根目录或候选根目录。使用 `kind: comparison`、`evaluation.scope: all_nonempty_cores` 和固定基线分核策略；候选分核改变时，在候选项显式声明与 `candidate.json` 一致的 `split_strategy`。`native_workload` 的 `declared_tiers` 为单工作量口径，不用于此全核实验。运行：

```text
python -m accel_model.app.cli.graph_search \
  --experiment <算子目录>/optimize/experiments/experiment.yaml \
  --output work/<实验目录>
```

该实验的范围为完整 case 的全部非空核；结果应包含硬件、成本与语义身份及导出复核状态。缺少逐核投影时不接受该范围。候选拓扑、路由、tile 或 phase 的修改必须可在导出的 YAML 中看到；实验输出和候选均留在 `work/`，不写入生产定义。

`comparison` 原生实验未绑定执行数值验收，报告的 `recommendation=false`。
`parameter_search` 物理 YAML 参数实验 可执行已声明的 `execution_cases`，另列已验收排名，
仍不自动推荐部署；该参数实验要求 `objective: {metric: per_case_min_cycles}`，
外部目标使用支持的会话／比较入口，且说明候选生成范围。四层图比较须核对候选、范围与实际执行绑定的证据，不能仅凭
CPU 样本数量升级资格；见[执行数值验收](../framework/numeric-execution-validation.md)。

多轮实验可使用[会话入口](../framework/graph-search.md)。
`register --kind native --source <候选根目录>` 封存候选及硬件，普通候选引用已登记
假设和可选父候选。实际分核不同于默认时通过 `--options` 显式传入；有动作来源清单
时策略必须一致。评分使用封存快照，下一轮构造继续使用 agent 自己维护的候选目录。
`evaluate` 自动纳入基线，`note` 引用评价记录，`status` 输出不同方向及逐 case 结果。

仅改会话目标使用 `agent_search set-objective/rescore`，不把它们当作 `dag_run` 或
独立实验选项；普通 `evaluate` 仍执行并计预算。详见[复用边界](../framework/objectives.md#复用范围)。

不同执行入口的结果不能直接混排。无法统一比较范围时，分别报告可比较结果及限制。

分核建模、固定 DAG 搜索与全局图搜索 可按瓶颈证据选用。全局图结果仍须检查共享带宽与缓存假设的适用范围。显式 `tier: L2` 是给定路由假设，不等同于自动命中证明。
