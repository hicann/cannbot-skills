# 多输入统一规则接口

用于在多个 case 上评价同一配置或按输入构造的统一分派规则。
每个完整方案保留全部 case 的观察，并按[本轮目标](objectives.md)比较。
逐 case 独立最优值只作诊断，不能拼成已验证规则。

## 用户适配器与规则

用户可导入 `accel_model.optimize.graph.global_search.candidate_policy.PolicyAdapter`，
由可导入的 `module:factory` 无参工厂返回。采用显式 `build` 接入时提供：

| 字段 | 输入约定 |
| --- | --- |
| `inputs` | 构造需要的参数名 tuple，不接收 case 名称、分组或清单序号 |
| `features` | 调用前已知的输入或派生特征；类型为 `integer/category`，明确合法域 |
| `domains` | 参数名到非空有限 tuple 的映射 |
| `build` | 用户函数 `build(params, choices)`，返回合法 `CandidateState` |
| `scope/basis` | 实际建模范围与适用依据 |
| `context/dependencies` | 实际硬件、成本和外部依赖身份 |
| `device_verifier` | 可选外部设备证据校验函数；缺少时仍可探索模型，不据此导入设备交付证据 |

规则 YAML 示例仅说明字段结构，`my_operator.policy:adapter` 须由用户提供：

```yaml
adapter: my_operator.policy:adapter
defaults: {storage: GM}
branches:
  - feature: rows
    op: le
    value: 8
    basis: 用户已验证的容量边界
    choices: {storage: UB}
```

`choices` 必须完整且位于适配器声明域；简单格式最多三个有序分支与默认回退，
支持 `le/divisible/eq`。树格式使用 `tree` 和 `limits`，节点为 `if/then/else`，
叶子给出唯一 `leaf`、完整 `choices` 和可选 `requires`。条件支持整数
`le/lt/ge/gt/divisible`、类别 `eq/in` 和 `all/any/not`；限制
`max_depth/max_leaves/max_predicates`。命中叶子而 `requires` 不满足时拒绝，
不能擅自换实现。输入特征的 `availability: runtime` 不可用于此分派或其派生特征。
边界、容量、版本、任务覆盖和同步仍须独立检查；验证需实际覆盖全部叶子及必要边界。

派生特征可用 `expr` 中的 `input/feature` 引用及二元
`add/sub/mul/floordiv/ceildiv/mod/min/max`；拒绝循环依赖，除数须为正。
整数扩展声明支持 `min/max`，类别支持 `values`；派生边界由调用方提供独立验证输入，
不能把超限或未覆盖边界写成已完整验证。

## 会话调用

先创建会话并登记相关假设。规则文件与搜索文件是不同对象，不能互换。

```text
python -m accel_model.app.cli.agent_search register-policy --session SESSION --id base --spec POLICY.yaml --baseline
python -m accel_model.app.cli.agent_search search-policy --session SESSION --id search-a --spec SEARCH.yaml
python -m accel_model.app.cli.agent_search search-policy --session SESSION --id search-a
```

搜索规格的输入形式如下，参数域与谓词须有任务依据；预算数字仅示意：

```yaml
space:
  adapter: my_operator.policy:adapter
  predicates:
    - {feature: rows, op: le, value: 8, basis: 用户已验证的容量边界}
  conditions: []
  limit: 64
  tree_limits: {max_depth: 2, max_leaves: 3, max_predicates: 4}
hypothesis: storage
options: {}
workers: 1
mode: exhaustive
seed: 0
budget: {generated: 64, construction_attempts: 128, model_calls: 128, wall_seconds: 600}
```

当前支持 `exhaustive/budget` 模式和种子 0；`conditions` 可用 `when/restrict`
收窄有依据的联合参数域，不能代替最终合法性检查。生成、构造尝试、实际模型调用、
复用及未覆盖数分别读取报告；`exhausted` 仅表示声明域已遍历，不表示设备全局最优。
会话的 `case_attempts` 仍限制实际评价，失败与中断也计预算。
复用已有评价仍需时间；没有新增模型调用不保证在墙钟预算内遍历完成。
返回 `budget_exhausted` 时保留未覆盖项，不能写成完整搜索或目标已达成。

同目标可用原运行 ID 恢复；`--max-steps` 可保存阶段进度，异常退出后先确认旧进程
结束再用 `--recover`，不同时启动两个写入者。仅换目标先重评分；需要继续搜索时换
新运行 ID。具体复用范围见[评价复用](objectives.md#复用范围)。

同 shape 的不同数据用 `cases[].scenario`，见[场景接口](runtime-control.md)。
适配器按参数构造，不读取场景答案作分派；用户 Python 不因此获得沙箱隔离。
交付实际规则、输入域、回退、候选及逐 case 验证；模型资格与设备资格分别列出。
