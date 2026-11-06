# 外部目标与评价复用

本页用于用户自行定义优化公式、业务约束和排序，或对已有有效观察重评分。
目标结果、原始周期、数值验证、模型资格和设备资格分别报告；业务 `passed`
不替代后四项证据，目标分数也不等于加速比。

## 仅换目标时的答复与记录

已关闭会话只改目标时，使用 `set-objective`／`rescore` 复用有效原始观察，保持关闭，
不启动搜索。说明复用流程时一并列出需保存的目标修订、原始评价 ID、执行身份、来源、
复用记录及完整目标结果；保留全部并列组，`per_case_winners` 仅作逐 case 诊断。
显式列出缺观察、采样不足或身份失效的项及补评原因，不用笼统的“必要时重验”替代缺口清单。
同时区分普通 `evaluate`：它仍实际执行所选候选及基线并计预算，不是上述重评分的缓存入口。
用户禁止新搜索时不为补缺口擅自重新开放会话；缺口和可继续条件随结果交付。

## 输入与调用

搜索会话和支持的 `comparison` 实验接受下列目标对象。会话 `set-objective --spec`
文件直接存该对象；比较实验放在 `objective:` 下。路径使用用户项目中的实际文件。

```yaml
provider: {path: /workspace/objectives/business.py, factory: create}
config: {}
dependencies: []
```

也可用 `provider: my_objectives.business:create`。工厂无参数，返回实现以下方法的
用户对象；provider 和依赖应可信、可复现，工厂无副作用，随机性由配置固定。
`config` 是 JSON 映射，由用户目标解释。可变文件依赖须列入 `dependencies`，包括
路径形式 provider 使用的外部可变包初始化文件；评价期间不修改目标及依赖。

| 用户方法 | 输入与返回约定 |
| --- | --- |
| `required_observations(config)` | 返回必需的顶层观察字段名列表；嵌套字段及适用域由 provider 检查 |
| `evaluate(matrix, config)` | 返回 JSON 对象：非 `null` 且可序列化的 `score`、布尔 `passed`，可附 `reasons: list[str]` 和 `details`；数值不可为 NaN/无穷 |
| `rank(evaluations, config)` | 返回从优到劣的候选名称组，每个传入候选恰好出现一次，组内并列 |

`matrix` 包含 `candidate`、`target` 和完整 `rows`；每行有 `case` 和 `observations`。
模型观察可包含 `candidate_time/baseline_time/ratio/trace_summary/evaluation/`
`baseline_evaluation`；设备观察另有实际 `samples/rounds/measurement/validation`。
`rank` 的行含 `candidate` 和目标结果。字段存在不代表其范围满足用户目标；声明的
观察缺失、case 不完整或 scope 不符时结果 `complete: false`，不补造数据。

可选 `target: {backend: model, scope: whole_graph, unit: cycles}` 必须对应实际评价。
设备目标使用 `backend: device, scope: full_kernel, unit: us`，采样要求可另声明
`measurement: {rounds: 3, samples: 7}`；这些数字只示意调用方要求，不是通用阈值。
支持标量、向量和并列，不自行将用户目标改为耗时最小、加权均值或固定阈值。
`weighted_geomean_ratio` 等兼容目标仅在本轮明确选用时使用。

```text
python -m accel_model.app.cli.agent_search set-objective --session SESSION --spec OBJECTIVE.yaml
python -m accel_model.app.cli.agent_search rescore --session SESSION
```

Python 会话对象对应 `session.set_objective(spec)` 与 `session.rescore()`。
修改目标源码／依赖后，`rescore` 显式接受新目标身份；`status` 不代替这一操作。
已关闭会话仍可重评分，并保持关闭。只要求重评分时不启动新搜索。

## 复用范围

| 入口／变化 | 使用方式与限制 |
| --- | --- |
| 仅改目标公式、配置、依赖或排序 | `rescore` 使用有效原始观察，不调用模型、不消耗模型调用预算 |
| 新目标缺观察或采样不满足要求 | 保留缺口，按支持入口和预算补评／补采，不从汇总值还原未采集数据 |
| 实现、硬件、输入／场景、成本、后端或执行身份改变 | 重新核验；失效记录保留作历史，不用于当前排名 |
| 同会话 `search-policy` | 可复用执行身份匹配且来源有效的评价，包括匹配的手动评价；是否命中以报告为准 |
| 普通 `evaluate` | 实际运行所选候选及基线并计预算，不自动跳过已有评价，也不保证仅补缺失 case |
| 独立比较实验／跨会话 | 不承诺自动读取其他实验或会话的缓存 |

原始评价 ID、执行身份、来源、复用记录及补评原因与目标修订一并保存。
`reuse_equivalence` 若存在，保留框架返回值，不手填绑定、改签名或改旧数值证据。
`evaluation_identity_mismatch` 不能被目标 `passed` 覆盖；目标改变不能恢复失效证据。
继续规则搜索须使用新的 `search-policy --id`，不能把旧目标下未评价项写成已覆盖。

## 排名与支持限制

完整方案的 `rankings/winners` 依实际目标结果解释；首组并列时保留整组，无唯一
赢家不能手工指定。会话 `per_case_winners` 是逐 case 最小周期的单值诊断，不采用
provider 的完整方案排序，也不表示业务通过；保留其原值，不用它打破业务并列。
其他入口同名字段按自身契约解释。设备结论核对 `device_ranking/deliverable`
及实际资格，不从模型排名推导设备收益。

`dag_run`、`search/joint_search` 不因本页协议而自动接受 provider。
`parameter_search` 实验目前要求 `objective: {metric: per_case_min_cycles}`。
入口不支持用户目标时，说明候选生成范围，再用支持的会话／比较入口评价已取得的
完整方案；这不证明全部搜索按新目标完成。多 case 联合选优使用同一配置或
[统一规则](policies.md)的全部 case 观察，不拼接逐 case 最快配置。
