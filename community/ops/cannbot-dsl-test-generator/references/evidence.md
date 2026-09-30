# TDD 输入、触发条件、结构证据与调用生命周期

TDD 需要有效工作量、结构关系或连续调用时阅读本文件。CSV 列不变；扩展只写入 `assertions_json`。脚本校验可执行契约，设计语义和采集来源须另行核对。

## 调用方提供的检验义务

每份 `U*.yaml` 含一个 unit 对象；unit_id 与文件名一致，verification_obligations 为非空列表，每项有全局唯一的 id 和 kind（numerical、structural、semantic、device）。可选 title、depends_on 用于展示边界和阅读顺序；依赖若填写，须引用现有单元且无环。仅这些字段与义务内容参与用例校验，不要求特定产物生成方式。

义务的 behavior、must_distinguish、observable_result、coverage_axes 和设计引用用于确定可区分的输入与判定。可选 input_conditions 描述必须实际满足的触发条件，格式见下文。其他实现规划字段不参与测试生成；本 Skill 不据此判断实现方案是否可行或是否满足源码、编译产物、设备证据要求。

## 确定性数据

张量配方增加两种 fill，仍由 Golden 的 `make_inputs` 在 CPU 构造：

- `values`：`values` 为数值列表或嵌套列表，按行展平，元素数与 shape 的乘积一致；用于小规模索引、前缀和、页映射及非均匀重复构造。
- `arange`：`start` 默认 0、`step` 默认 1；未写 axis 时生成全部元素后 reshape，指定 axis 时沿该轴递增并在其他轴重复。例如 `{shape: [2, 129], dtype: int32, fill: arange, axis: -1}` 每行是 0..128。

已有 normal、常量和特殊值配方保留。固定 seed 与配方共同决定输入；新 Golden 的 make_inputs 可采用 prepare_golden.py 中的 SUPPORT。已有 Golden 复用前核对它确实支持当前配方，不修改数学口径。

## 实际触发条件

Unit 义务可写 `input_conditions`，CSV 对应行必须用 `kind=input_conditions` 的 checks 承接。至少一个绑定用例完整包含该义务的所有条件；校验器物化当前 seed 的真实输入并检查，runner 在调用目标实现前再次检查。失败表示用例未触发目标行为，不是算子数值失败。

```json
[{"kind":"input_conditions","checks":[
  {"input":"indices","metric":"unique_count","axis":-1,
   "where":{"min":0,"max":{"input":"key","dimension":1}},
   "relation":"ge","value":129}
]}]
```

metric 支持 dimension（须有 axis）、numel、count、unique_count。count/unique_count 没有 axis 时检查全张量，有 axis 时逐行检查该轴，每行均须满足。where 的 min 为包含下界，max 为排除上界，eq 为等值；边界为有限数或 `{input, dimension}` 引用另一张量维度。relation 为 eq/ne/ge/gt/le/lt。无法由这些谓词表达的关系返回具体缺口，不编造“已覆盖”。自然语言 coverage_axes 仍须与实际输入和判定逐项核对。

输入条件不能单独承担 structural、semantic 或 device 义务；必须同时有输出或实际探针证据。

## 结构关系

保留 `probe_events` 全等断言，确定性的事件流可继续使用。并行执行宜使用 `probe_predicates`，避免冻结无关任务的完整全序：

```json
[{"kind":"probe_predicates","checks":[
  {"op":"count","where":{"event":"compute","domain":"required"},"relation":"ge","value":1},
  {"op":"before","first":{"event":"ready"},"second":{"event":"consume"},"keys":["slot","generation"]},
  {"op":"paired","first":{"event":"consume"},"second":{"event":"free"},"keys":["slot","generation"]}
]}]
```

事件为字典；where/first/second 按字段等值选择。count 检查匹配数量；paired 检查非空的两侧按 keys 分组数量相同；before 另检查对应事件的列表位置严格先后。复用用 generation 等真实代次区分。事件列表只在采集端定义的顺序确实代表发生顺序时用于 before；跨执行域仅按 Host 收集顺序拼接不能证明时序。

这些谓词不直接证明重叠或事件真实性。设计要求重叠时需真实设备时序或 trace，并明确采集来源与判据；采集代码、字段与运行产物须可追溯。固定 snapshot 和阶段名称不能作为完成证明。

numerical 的 `match_golden` 可以同时携带输入条件、输出和探针断言，使同一次调用验证数值与结构；绑定结构义务必须有 probe_events 或 probe_predicates。semantic/device 仍需对应的输出/探针断言。未使用扩展的现有 CSV 行保持原行为。

## 连续调用与重复

用一个 execution 控制本行的调用序列。先执行 inputs_json，再执行 calls；每次均按当前输入运行独立 Golden/断言，整个序列在同一进程重复。repeat 为 1..100，附加 calls 最多 32 个，不允许嵌套 execution。

```json
[{"kind":"execution","repeat":3,"bitwise":true,"calls":[
  {"inputs":{"x":{"shape":[3,4],"dtype":"float32","fill":"ones"},
             "y":{"shape":[3,4],"dtype":"float32","fill":"zeros"}},"seed":7}
]}]
```

calls 的 inputs 使用同一公开输入配方；seed 缺省沿用本行，assertions 缺省沿用本行非 execution 断言，显式提供时仍须满足绑定义务类型。必要时随附加调用调整输入条件和 shape 断言。bitwise=true 要求 repeat≥2，比较相同序列位置不同轮次的输出存储位，首轮输出立即克隆，避免返回缓冲后续被覆盖。是否要求 bitwise 来自 spec/设计；不把它推广到所有算子。

调用不会重新导入目标模块，不保证公开入口复用同一编译句柄；句柄生命周期仍需对应的实际结构证据。dst_args 序列当前复用本行 output_json，因此选择与该输出布局兼容的输入；需要不同输出分配布局时记录契约缺口，不伪装成已测。
