# testcase.csv 用例契约

唯一持久化测试设计文件是 `TEST_DIR/testcase.csv`。UTF-8 CSV 的列顺序固定：

```text
sheet,case_id,source,branch_ids,unit_id,inputs_json,output_json,seed,expected,assertions_json,obligation_ids,note,status,design_ref,condition,exclude_reason
```

`sheet` 和 `source` 同为 `blackbox`、`whitebox`、`tdd` 或 `performance`。`status=active` 的行必须有 `inputs_json` 且无排除理由；`status=excluded` 的行必须有 `exclude_reason` 且无可执行输入。固定测试入口只运行 active 行。JSON 单元格使用标准 JSON；逗号分隔 ID 的字段不允许重复。

CSV 使用 `csv.DictWriter` 等标准 CSV 序列化器写入，JSON 单元格先用 `json.dumps` 编码；不要手工拼接含逗号和引号的 CSV。分批保存时保留其他分组及已有 ID，重读文件检查列序、JSON 可解析性和 ID 唯一性。未完成覆盖不代表验收通过。

`synthesize.dtype` 支持两种形式：`dtype: float16` 要求当前用例所有已提供的张量（含 TensorList 成员）使用该 dtype，不作用于标量属性或未提供的可选输入；`dtype: {query: float16, mask: float32}` 只约束列出的输入。混合 dtype 使用映射。缺省不追加 dtype 约束，仍遵守 spec 输入类型校验。字符串形式可直接绑定相应 spec 路径，无需通过 excluded 绕过。

## 黑盒

先运行 `build_blackbox_cases.py --inventory` 读取标准输出的必需 spec 路径及因子目标。每个 active 行用 `BB-...` ID，`design_ref` 以逗号分隔真实 spec 路径，`condition` 写 `L0`、`L1` 或 `L2`，`note` 写用例目的，`inputs_json` 写公开输入配方，`expected` 写 `match_golden`、`match_golden_nan` 或 `raises_error:<spec 错误码>`。属性输入与 spec 声明的 `list[...]` 类型一致时可写 JSON 数组（元素为标量）。L2 只能对应 spec 声明的异常；NaN 比较须回指相应语义。无法执行的必需 spec 路径写单独 excluded 行，`design_ref` 只写该路径并说明理由。校验器验证所有必需路径均被覆盖或排除，并可用 `--coverage` 在标准输出重算取值、成对组合和未命中目标。路径被引用不等于取值覆盖。

## 白盒

每行 ID 为 `WB-<分支 ID>`，`design_ref` 为所提供设计文件中的实际条目，`condition` 描述分支条件。active 行的 `inputs_json` 是公开输入配方，`note` 统一写 `return_value` 或 `dst_args`；后者在 `output_json` 提供本例实际存在输出的配方：必需输出不得省略，可选输出缺省时省略对应键或写 `null`，TensorList 使用已有 `tensors: [...]` 配方。白盒静态校验核对名称、角色和可静态判定的配方；条件存在性及依赖实际输入的长度在 CPU 缓存准备时核对，不能凭可选标志省略条件要求存在的输出。

每个 active 白盒行必须显式填写 `expected`：数值分支写 `match_golden`；规格要求传播 NaN 的分支写 `match_golden_nan`；仅验证可观察行为时写 `assertions`，并在 `assertions_json` 提供输出形状、类型、设备、与输入相等或输入不变性等断言。`assertions` 的每次调用（含连续调用）都必须有可观察行为断言，仅输入条件或执行控制不构成结果判定。数值比较可同时附带行为断言；不得为绕过数值失败改成 `assertions` 或放宽 NaN 语义。错误码行为使用黑盒的 `raises_error:<spec 错误码>`。

无法执行的 DESIGN 分支写 excluded 行及理由，`expected` 可留空。白盒校验器直接核对 DESIGN 引用、输入、输出及执行判定，拒绝 active 行的空白或不支持的 `expected`；脚本只校验 CSV，不代填判定或创建分支图副本。已有数值白盒行缺少 `expected` 时，由测试设计者核对规格后补写 `match_golden`，重新校验并执行固定入口。

固定 runner 沿用非空 `output_json` 识别目标缓冲区调用；本例全部输出均合法缺省、没有缓冲区时，使用已有 `note=dst_args` 明确调用方式，TDD 行也适用。缺省输出不分配缓冲区、不向目标接口强塞该关键字，返回值按声明顺序保留 `None` 位置。Tensor 与 TensorList 成员的目标缓冲区在每次调用前递归清零，不能依赖配方生成的随机初值。

## 实现单元 TDD

每行 ID 为 `TD-<原生 Unit ID 小写>-<用例名>`，`unit_id` 为原生 ID 小写，`obligation_ids` 以逗号分隔该 Unit 的真实验证义务，`branch_ids` 可关联 active 白盒分支。每条原生义务至少被一个 TDD 行覆盖。数值义务使用 `expected=match_golden`，规格要求传播 NaN 的义务使用 `expected=match_golden_nan`；可观察行为使用 `expected=assertions` 或与 Golden 联合验证。纯结构承诺通过调用方的 review_evidence 验收，不生成结构探针用例。CSV 中的 `inputs_json`、`output_json` 和 `assertions_json` 是唯一用例内容；不生成适配 Unit、桥接计划或 TDD JSON 副本。

`branch_ids` 只用于追溯，不隐式提供调用参数。复用白盒分支的目标缓冲区配方时，由用例生成者明确写入本行既有 `output_json`；校验器不改写 CSV、不在内存里补齐后宣布可执行。目标缓冲区方式缺配方时在准备阶段报出具体用例缺项；全部输出合法缺省的用例仍在本行用已有 `note=dst_args` 明确调用方式。缓存准备和 runner 消费的都是这份最终 CSV。

TDD 义务类型与判定必须对应：`semantic` 至少包含 output_shape、output_dtype、output_equals_input 或 input_unchanged；`device` 包含 output_device。旧 `structural`、`performance` 与未知类型拒绝进入 TDD，性能目标须另有实际测量条件与判据。数值判定与输出/输入不变性断言可由同一次调用共同承担；不以普通 Golden 比较覆盖非数值义务。错误码行为使用黑盒的 raises_error 用例。

输出类断言（output_shape、output_dtype、output_device、output_equals_input）作用于算子全部返回值；算子返回元组/列表时可加 `index`（非负整数）定位单个输出，返回字典时可加 `output`（非空字符串）按名定位；未定位时同一断言须对每个输出元素成立。属性输入与 spec 声明的 `list[...]` 类型一致时可写 JSON 数组（元素为标量）。

实际触发条件、确定性值配方、输入不变性及连续调用使用 [TDD 证据契约](evidence.md)。CSV 列保持不变，扩展均写入 assertions_json；新增条件必须由实际输入触发。

## 用户性能用例

使用 `PERF-...` ID，sheet/source 均为 performance，status 为 active；沿用公开输入/输出 JSON 配方、seed，expected 为 match_golden（规格明确非有限值语义时可用 match_golden_nan）。design_ref 填与黑盒相同规则的真实 spec 路径，condition 描述用户场景，note 记录原始来源；不填写 branch_ids、unit_id、obligation_ids。沿用 assertions_json 的公开断言及调用序列契约，不新增计时字段。

使用 build_performance_cases.py 校验输入与引用，不套用黑盒覆盖率或 L0/L1/L2 条件，不要求 DESIGN 或 Unit。用户未提供时允许零行，不用 excluded 表示未提供或输入不足。完整测试生成仍复用同一 Golden、CPU 检查与缓存；正确性结果不等于性能结果。

并行更新同一 CSV 时，所有写入方使用 update_case_group.py；stdin 接收所属分组完整行的 JSON 数组，脚本加锁重读、保留其他分组并原子保存。锁文件只用于同步，不是第二份用例或进度台账。
