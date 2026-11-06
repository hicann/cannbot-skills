# 参数寻优

先保存一个可运行的模型和[完整 case 基线](quickstart.md)。搜索提供候选方案；
最终采用前，应使用同一输入和硬件重新运行完整 case。

需要代理组织方案设计、搜索、完整 case 复评及实现反馈时，使用
[联合寻优工作流](joint-optimization/guide.md)，由本 skill 与 wheel 配合执行。

## 分核搜索

`work/my_matmul/optimize/split.yaml` 的主要结构是：

```yaml
name: split_demo
shape: {M: 256, N: 256, K: 128, Mb: 128, Nb: 128, Kb: 128, M_L0: 64, N_L0: 64, K_L0: 128}
ncores: 32
strategy:
  rule: [greedy, contiguous, round]
  order: [m, n]
  granularity: task
  hit_policy: all_miss
search: {top_k: 3}
```

`shape` 必须与基线对应。候选策略在相同工作量和硬件条件下排名，`top_k` 控制保留
的方案数量。通过 `--verify-simpy` 请求对候选进行 SimPy 复核：

```bash
python -m accel_model.app.cli.split_run work/my_matmul \
  --config work/my_matmul/optimize/split.yaml --out work/split --verify-simpy
```

输出根目录会再附加 `name`。阅读 `ranked.yaml` 中的 `t_total`、`feasible`、
`adoptable`、`verification` 和失败原因；只采用可行且可采纳的方案。
`split_plan.rank1.yaml` 存放策略，`evaluation.rank1.yaml` 和 `PLAN.md` 解释评分。

```bash
python "$OPMODEL_SKILL_DIR/scripts/run_case.py" work/my_matmul \
  --case work/my_matmul/case.yaml \
  --plan work/split/split_demo/split_plan.rank1.yaml --output work/with_split
```

普通完整 case 要求策略核数等于所选硬件的 `usedCoreNum`，本例固定为 32。
分核搜索虽然可以探索其他核数，生成的方案不能不加区分地套入当前完整 case 入口。
当分块完全均衡时，多个策略得到相同周期是正常结果。

## DAG 参数搜索

候选参数来自 `optimize/search.yaml` 的 `parameter_space`。实验
`optimize/dag.yaml` 固定外层块和 M/N 内层块，只留下 `K_L0` 搜索：

```yaml
name: dag_demo
shape: {M: 256, N: 256, K: 128}
phases: C1
tiling: {Mb: 128, Nb: 128, Kb: 128, M_L0: 64, N_L0: 64}
search: {top_k: 3, replay_top: 3}
```

```bash
python -m accel_model.app.cli.dag_run work/my_matmul \
  --config work/my_matmul/optimize/dag.yaml --out work/dag
```

阅读 `work/dag/dag_demo/RANKED.txt`，先看周期、候选分块，再看 `PLAN.md`、trace 和
统计文件。`replay_top` 决定回放的领先候选数。全部参数固定时可能转为单点评估，
不会产生多候选排名；没有搜索空间时也会退化为单点模式。

**DAG 评分与完整 case 总周期不是同一个统计范围。** 不要直接拿 `RANKED.txt`
的周期除以完整 case 周期宣称加速。输出的 `operator.yaml` 和 `dag_optimal.yaml`
用于候选报告，也不应直接覆盖完整模型的六份定义文件。

## 把候选参数应用回完整 case

本例可比较 `K_L0=64` 与 `128`。查看排名后，将选中的值写到新的 case 文件：

```bash
python - <<'PY'
from pathlib import Path
import yaml
case = yaml.safe_load(Path('work/my_matmul/case.yaml').read_text())
case['K_L0'] = 64
Path('work/my_matmul/case_tile64.yaml').write_text(yaml.safe_dump(case, sort_keys=False))
PY
python "$OPMODEL_SKILL_DIR/scripts/run_case.py" work/my_matmul \
  --case work/my_matmul/case_tile64.yaml --output work/tile64
```

按[结果分析](analysis.md)比较 `work/baseline` 与 `work/tile64`。
若候选还改变了 slots、phases 或其他配置，应明确同步这些值，并保留基线副本。
搜索没有找到更优方案时，保留基线；不要把候选排名当作收益承诺。

## 进阶搜索入口

`graph_search` 面向模型支持的图动作、候选和 checkpoint；`agent_search` 提供搜索会话。
它们需要对应模型的语义、改写声明和约束，不能直接复用分核或 DAG 实验 YAML。
先用各自的 `--help` 查看子命令，再检查目标模型是否支持该动作；参见[CLI 参考](cli.md)。

实验输入、产物和使用限制见[图搜索接口](framework/graph-search.md)。
保存命令返回的候选与检查点，不手工修改其内容来改变验证状态。
