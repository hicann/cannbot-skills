# testcase.csv 用例契约

唯一持久化测试设计文件是 `TEST_DIR/testcase.csv`。UTF-8 CSV 的列顺序固定：

```text
sheet,case_id,source,branch_ids,unit_id,inputs_json,output_json,seed,expected,assertions_json,obligation_ids,note,status,design_ref,condition,exclude_reason
```

`sheet` 和 `source` 同为 `blackbox`、`whitebox` 或 `tdd`。`status=active` 的行必须有 `inputs_json` 且无排除理由；`status=excluded` 的行必须有 `exclude_reason` 且无可执行输入。固定测试入口只运行 active 行。JSON 单元格使用标准 JSON；逗号分隔 ID 的字段不允许重复。

CSV 使用 `csv.DictWriter` 等标准 CSV 序列化器写入，JSON 单元格先用 `json.dumps` 编码；不要手工拼接含逗号和引号的 CSV。分批保存时保留其他分组及已有 ID，重读文件检查列序、JSON 可解析性和 ID 唯一性。未完成覆盖不代表验收通过。

`synthesize.dtype` 支持两种形式：`dtype: float16` 要求当前用例所有已提供的张量（含 TensorList 成员）使用该 dtype，不作用于标量属性或未提供的可选输入；`dtype: {query: float16, mask: float32}` 只约束列出的输入。混合 dtype 使用映射。缺省不追加 dtype 约束，仍遵守 spec 输入类型校验。字符串形式可直接绑定相应 spec 路径，无需通过 excluded 绕过。

## 黑盒

先运行 `build_blackbox_cases.py --inventory` 读取标准输出的必需 spec 路径及因子目标。每个 active 行用 `BB-...` ID，`design_ref` 以逗号分隔真实 spec 路径，`condition` 写 `L0`、`L1` 或 `L2`，`note` 写用例目的，`inputs_json` 写公开输入配方，`expected` 写 `match_golden`、`match_golden_nan` 或 `raises_error:<spec 错误码>`。L2 只能对应 spec 声明的异常；NaN 比较须回指相应语义。无法执行的必需 spec 路径写单独 excluded 行，`design_ref` 只写该路径并说明理由。校验器验证所有必需路径均被覆盖或排除，并可用 `--coverage` 在标准输出重算取值、成对组合和未命中目标。路径被引用不等于取值覆盖。

## 白盒

每行 ID 为 `WB-<分支 ID>`，`design_ref` 为所提供设计文件中的实际条目，`condition` 描述分支条件。active 行的 `inputs_json` 是公开输入配方，`note` 统一写 `return_value` 或 `dst_args`；后者还需在 `output_json` 提供全部输出配方。无法执行的 DESIGN 分支写 excluded 行及理由。白盒校验器直接核对 DESIGN 引用、输入和输出；不创建分支图副本。

## 实现单元 TDD

每行 ID 为 `TD-<原生 Unit ID 小写>-<用例名>`，`unit_id` 为原生 ID 小写，`obligation_ids` 以逗号分隔该 Unit 的真实验证义务，`branch_ids` 可关联 active 白盒分支。每条原生义务至少被一个 TDD 行覆盖。数值义务使用 `expected=match_golden`；结构义务在 `assertions_json` 中提供非空的 `probe_events` 或 `probe_predicates`。结构专用行使用 `expected=assertions`，联合数值与结构的行可使用 `match_golden`。CSV 中的 `inputs_json`、`output_json` 和 `assertions_json` 是唯一用例内容；不生成适配 Unit、桥接计划或 TDD JSON 副本。

TDD 义务类型与判定必须对应：`semantic` 至少包含 output_shape、output_dtype、output_equals_input、probe_events 或 probe_predicates；`device` 包含 output_device。`performance` 与未知类型拒绝进入 TDD，性能目标须另有实际测量条件与判据。数值判定与输出/探针断言可由同一次调用共同承担；不以普通 Golden 比较覆盖非数值义务。错误码行为使用黑盒的 raises_error 用例。

实际触发条件、确定性值配方、结构谓词及连续调用使用 [TDD 证据契约](evidence.md)。CSV 列保持不变，扩展均写入 assertions_json；新增条件必须由实际输入触发。
