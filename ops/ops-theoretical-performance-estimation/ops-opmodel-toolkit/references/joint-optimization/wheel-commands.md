# 使用 wheel 完成模型侧联合寻优

本例使用安装包导出的模型，演示完整 case 运行、联合搜索、候选复评和会话管理。
先按[安装说明](../install.md)准备环境并安装当前兼容的最新版本：
`python -m pip install --upgrade --pre --index-url https://repo.huaweicloud.com/repository/pypi/simple opmodel`。
运行前核对已安装包的实际接口与模型配置；升级后重新验证项目使用的示例。
以下命令在自己的项目根目录执行，`work/joint_*` 应为尚不存在的新目录或文件。
示例验证软件使用流程，不包含设备实现与设备性能结论。

## 准备模型和完整 case

```bash
python -c "from importlib.metadata import version; print(version('opmodel'))"
mkdir -p work
export OPMODEL_WORKSPACE="$PWD/work"
opmodel-assets export --output work/joint_resources
cp -R work/joint_resources/operators/dag/matmul work/joint_model
python - <<'PY'
from pathlib import Path
import yaml
shape = dict(M=256, N=256, K=128, Mb=128, Nb=128, Kb=128,
             M_L0=64, N_L0=64, K_L0=128)
with Path('work/joint_case.yaml').open('x') as f:
    yaml.safe_dump(shape, f, sort_keys=False)
PY
```

工作区不要放在系统 `/tmp` 下。编辑自己的模型目录，不修改安装目录中的资源。
导出的模型与 case 不绑定；换形状时仍需核对 workload、任务空间及容量约束。

```bash
python - <<'PY'
from pathlib import Path
import yaml
from accel_model.app.case.native_case import run_native_case
shape = yaml.safe_load(Path('work/joint_case.yaml').read_text())
out = Path('work/joint_baseline')
if out.exists():
    raise FileExistsError(out)
result = run_native_case('work/joint_model', shape, output_dir=out,
                         derive=True, export_trace=True)
print(result['total_cycles'], result['summary']['active_cores'])
PY
```

读取 `result.json`、`params.json`、`assignment.json` 和 trace/统计文件；CSV 与 trace
主要展示关键核，完整 case 口径看 `result.json`。仿真运行完成不自动证明数学数值已对拍。

## 显式联合搜索

下面调用二进制包内的 `joint_search`，比较四组 K 内层块与 L0A 槽数配置。
矩阵乘示例只声明一个 `C1` stage，因此本次阶段划分和发射策略固定；这不是多阶段搜索。
实际模型支持多种合法阶段方案时，应显式扩充 `phase_candidates`，保持 stage 完整覆盖。

```bash
python - <<'PY'
from pathlib import Path
import json
import yaml
from accel_model.model.case.config import load_config
from accel_model.optimize.dag.hooks import OpHooks
from accel_model.optimize.dag.search import joint_search

model = 'work/joint_model'
shape = yaml.safe_load(Path('work/joint_case.yaml').read_text())
cfg = load_config(model)
out = Path('work/joint_search')
out.mkdir(exist_ok=False)
report = joint_search(
    cfg, OpHooks(model), shape,
    phase_candidates=[cfg['operator']['pipeline']['phases']],
    parameter_candidates=[
        {'K_L0': 64, 'slots_l0a_c1': 1},
        {'K_L0': 64, 'slots_l0a_c1': 2},
        {'K_L0': 128, 'slots_l0a_c1': 1},
        {'K_L0': 128, 'slots_l0a_c1': 2},
    ],
    issue_policies=('default',), top_k=4, max_evaluations=4,
)
(out / 'search.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(report['coverage'])
for row in report['rows']:
    print(row['candidate_id'], row['cycles'], row['requested_params'])
PY
```

此段仅声明实验输入并调用 wheel 接口；候选按下一节单独运行完整 case 复评。
检查 `coverage.complete`、`by_status` 和每条 `audit`，区分已评价、非法契约、容量或依赖
淘汰以及 `budget_skipped`。`rows` 受 `top_k` 截断，不能以它的长度当作实际覆盖总数。
候选被 hook 重新派生后还要检查生效参数；不存在可行候选时保留原因并停止采纳步骤。

## 采纳模型候选并复评完整 case

本例只变化 `K_L0` 和 `slots_l0a_c1`，两者分别属于 case 输入和执行配置。
复制基线模型后应用候选；对其他模型，先按字段契约分类，不要照搬字段名。

```bash
python - <<'PY'
from pathlib import Path
import json
import shutil
import yaml
from accel_model.app.case.native_case import run_native_case

report = json.loads(Path('work/joint_search/search.json').read_text())
if not report['rows']:
    raise RuntimeError('没有可采纳候选，先检查 audit')
chosen = report['rows'][0]
target = Path('work/joint_selected')
out = Path('work/joint_reevaluated')
if target.exists() or out.exists():
    raise FileExistsError('请选择新的候选和输出目录')
shutil.copytree('work/joint_model', target)
op_file = target / 'definition/operator.yaml'
op = yaml.safe_load(op_file.read_text())
op.setdefault('params', {})['slots_l0a_c1'] = chosen['params']['slots_l0a_c1']
op['pipeline']['phases'] = chosen['phases']
op['pipeline']['issue_policy'] = chosen['issue_policy']
op_file.write_text(yaml.safe_dump(op, sort_keys=False, allow_unicode=True))
shape = yaml.safe_load(Path('work/joint_case.yaml').read_text())
shape['K_L0'] = chosen['params']['K_L0']
(target / 'case.yaml').write_text(yaml.safe_dump(shape, sort_keys=False))
result = run_native_case(target, shape, output_dir=out, derive=True, export_trace=True)
baseline = json.loads(Path('work/joint_baseline/result.json').read_text())
print('完整 case 基线/候选周期:', baseline['total_cycles'], result['total_cycles'])
PY
```

候选没有改善时保留基线；局部搜索周期不直接与完整 case 周期比较。
模型候选还需按任务范围交给实现者，完成原正确性和同口径设备计时后才能报告设备收益。

## 分核和固定 DAG 搜索

这两个入口需要各自的实验 YAML，不能使用上面的 `search.json` 代替。
创建与完整 case 同输入的实验：

```bash
python - <<'PY'
from pathlib import Path
import yaml
shape = yaml.safe_load(Path('work/joint_case.yaml').read_text())
split = dict(name='split', shape=shape, ncores=32,
             strategy=dict(rule=['greedy', 'contiguous', 'round'], order=['m', 'n'],
                           granularity='task', hit_policy='all_miss'), search=dict(top_k=3))
dag = dict(name='dag', shape={k: shape[k] for k in ('M', 'N', 'K')}, phases='search',
           tiling={k: shape[k] for k in ('Mb', 'Nb', 'Kb', 'M_L0', 'N_L0')},
           search=dict(top_k=3, replay_top=3))
for name, spec in [('split', split), ('dag', dag)]:
    with Path(f'work/joint_{name}.yaml').open('x') as f:
        yaml.safe_dump(spec, f, sort_keys=False)
PY
python -m accel_model.app.cli.split_run work/joint_model \
  --config work/joint_split.yaml --out work/joint_split --verify-simpy
python -m accel_model.app.cli.dag_run work/joint_model \
  --config work/joint_dag.yaml --out work/joint_dag
```

分核输出 `ranked.yaml`、`split_plan.rank1.yaml` 和评价文件；只有 feasible、adoptable
的方案才可按其 `split` 映射传给 `run_native_case(strategy=...)`。本例使用 profile 的
32 核，不把它作为其他硬件的默认值。DAG 输出的 `RANKED.txt` 和候选报告用于局部比较。
此例只有一个 stage，`phases: search` 仍只有一种划分，应如实记录。

## 图动作与搜索会话

```bash
python -m accel_model.app.cli.graph_search --actions work/joint_model
python -m accel_model.app.cli.graph_search --help
python -m accel_model.app.cli.agent_search --help
```

动作发现不等于实际应用或完成搜索。多轮比较可采用 `agent_search`；下面只演示固定基线
登记和评价，不能把会话创建本身计为结构搜索。

```bash
python - <<'PY'
from pathlib import Path
import yaml
shape = yaml.safe_load(Path('work/joint_case.yaml').read_text())
spec = dict(name='matmul_model_session',
    requirements=dict(semantic='C=A@B，bf16 输入输出，fp32 累加',
                      numerical='保持模型原数值契约；本例不执行设备数值验收'),
    cases=[dict(name='base', params=shape, weight=1)],
    objective=dict(metric='weighted_geomean_ratio'), budget=dict(case_attempts=4))
with Path('work/joint_session.yaml').open('x') as f:
    yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
with Path('work/joint_options.yaml').open('x') as f:
    yaml.safe_dump(dict(derive=True), f)
PY
python -m accel_model.app.cli.agent_search init \
  --spec work/joint_session.yaml --output work/joint_session
python -m accel_model.app.cli.agent_search register \
  --session work/joint_session --id baseline --kind native \
  --source work/joint_model --options work/joint_options.yaml --baseline
python -m accel_model.app.cli.agent_search evaluate \
  --session work/joint_session --candidates baseline
python -m accel_model.app.cli.agent_search status --session work/joint_session
python -m accel_model.app.cli.agent_search close --session work/joint_session \
  --reason planned_scope_complete --evidence '已完成本例固定基线评分'
```

本例显式选择兼容的耗时目标，不表示所有任务都采用它。修改目标与复用已有评价见
[外部目标接口](../framework/objectives.md)；本例的 `joint_search` 不直接接受 provider。

`requirements.numerical` 是契约说明，填写文字不会启动数值检查。
本例最后关闭会话。需要继续候选研究时，在关闭前用
`propose/register/evaluate/note/retain/defer` 维护证据；先查看对应子命令帮助。
`cases.params` 在会话中固定，不能用同一契约覆盖本来需要由候选改变的输入；固定参数
妨碍比较时，应先重新定义公平的比较契约，而不是修改已冻结会话。
候选模型目录与会话目录应分开，原目录编辑不会自动更新已登记快照。
`close` 需要真实结束原因和证据；预算耗尽或用户暂停不表示目标达成。
