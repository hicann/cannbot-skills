# TDD 输入、触发条件、设计证据与调用生命周期

TDD 选例及核对有效工作量、结构关系或连续调用时阅读本文件。CSV 列不变；扩展只写入 `assertions_json`。脚本校验可执行契约，设计语义和采集来源须另行核对。

## 调用方提供的检验义务

每份 `U*.yaml` 含一个 unit 对象；unit_id 与文件名一致，verification_obligations 为非空列表，每项有全局唯一的 id 和 kind（numerical、semantic、device）。可选 title、depends_on 用于展示边界和阅读顺序；依赖若填写，须引用现有单元且无环。仅这些字段与义务内容参与用例校验，不要求特定产物生成方式。

义务的 behavior、must_distinguish、observable_result、coverage_axes 和设计引用用于确定可区分的输入与判定。可选 input_conditions 描述必须实际满足的触发条件，格式见下文。其他实现规划字段不参与测试生成；本 Skill 不据此判断实现方案是否可行或是否满足源码、编译产物、设备证据要求。

## 选例与成本

先满足冻结规格、当前 Unit 阶段域、全部适用义务与必要子条件，再优先选择计算规模较小且有判别力的输入。不只按 shape 大小判断：按具体义务核对真实触发、数据分布、公开断言与冻结容差的判别力；需要非退化输入才能暴露错误时采用相应构造，保留原义务要求的零值、合法空张量及特殊值。较小输入不能覆盖时保留必要规模、多块、复用代次和调用顺序，不把本节当作缩减覆盖、性能或结构承诺的依据。

初始选例时，同一 Unit 内触发条件和断言兼容的一行可关联多项义务，不机械地为每项义务另造一例；互斥条件或不同故障面分别覆盖，不强制合并。前序回归复用已有用例，不在每个后继 Unit 重复制一套；当前新增义务仍须有本 Unit 的真实绑定与判别依据，不能借前序用例冒充完成，也不改写义务归属。需要整理尚未冻结的草稿时保留原覆盖与来源，不自动删并已提交用例。

选例取舍沿用已有输入、触发与断言依据，必要说明留在原证据中；不为本节另设成本表、固定用例数、统一规模上限、机械组合穷举或全局最优性证明。验收检查必要覆盖、判别力和取舍依据，不因用例不是理论最少或最小而单独拒收。不得为换取通过删除、排除用例、缩小已有失败输入、放宽断言或修改用户性能 case；不要求新增故障注入或设备实验。

## 确定性数据

张量配方增加两种 fill，仍由 Golden 的 `make_inputs` 在 CPU 构造：

- `values`：`values` 为数值列表或嵌套列表，按行展平，元素数与 shape 的乘积一致；用于小规模索引、前缀和、页映射及非均匀重复构造。
- `arange`：`start` 默认 0、`step` 默认 1；未写 axis 时生成全部元素后 reshape，指定 axis 时沿该轴递增并在其他轴重复。例如 `{shape: [2, 129], dtype: int32, fill: arange, axis: -1}` 每行是 0..128。

已有 normal、常量和特殊值配方保留。固定 seed 与配方共同决定输入；新 Golden 的 make_inputs 可采用 prepare_golden.py 中的 SUPPORT。已有 Golden 复用前核对它确实支持当前配方，不修改数学口径。

## 实际触发条件

规划或草拟前可执行公开只读预检：`scripts/build_tdd_cases.py --check-unit-contracts --units-dir UNITS`，仅检查原生 Unit 测试契约，复用正式入口的同一校验，不需要 spec、DESIGN、CSV、Golden 或设备，不写文件。通过不证明具体用例绑定或真实触发，正式准备和 runner 仍各自校验；发现不支持的条件须当前阻断并交回责任阶段，不因“只是草稿”放行。

Unit 义务可写 `input_conditions`，CSV 对应行必须用 `kind=input_conditions` 的 checks 承接。至少一个绑定用例完整包含该义务的所有条件；校验器物化当前 seed 的真实输入并检查，runner 在调用目标实现前再次检查。失败表示用例未触发目标行为，不是算子数值失败。

```json
[{"kind":"input_conditions","checks":[
  {"input":"indices","metric":"unique_count","axis":-1,
   "where":{"min":0,"max":{"input":"key","dimension":1}},
   "relation":"ge","value":129}
]}]
```

metric 支持 dimension（须有 axis）、numel、count、unique_count。count/unique_count 没有 axis 时检查全张量，有 axis 时逐行检查该轴，每行均须满足。where 的 min 为包含下界，max 为排除上界，eq 为等值；边界为有限数或 `{input, dimension}` 引用另一张量维度。relation 的 eq/ne/ge/gt/le/lt 使用有限数值；multiple_of 的 value 必须是正整数，按整数统计量整除判断（0 是正整数的倍数，是否允许空输入仍由规格决定）；in 的 value 是非空有限数值列表，检查统计值属于该列表。列表不接受布尔、字符串、嵌套值或 NaN/Inf，不支持 not_in；原比较关系不改语义。相同校验与执行逻辑用于准备和正式 runner，输入不满足时在目标调用前失败。不得将 scope.domain_constraints 的键直接搬作此处 relation，也不以输入碰巧满足代替义务条件的实际绑定。无法由这些谓词表达的关系返回具体缺口并阻断当前草稿，不编造“已覆盖”或只记为下游问题。自然语言 coverage_axes 仍须与实际输入和判定逐项核对。

输入条件不能单独承担 semantic 或 device 义务；还须验证实际输出或输入调用后的状态。

## 设计结构检查

算子源码不包含 probe 内容，不生成 `probe/` 或探针契约；固定 runner 不加载探针模块。不要求算子为测试插入设备打印、trace 缓冲、计数、输入副本或同步回读。

纯结构承诺（执行域、生产/消费、搬运链、同步、代次、重叠等）由负责验收的 Agent 通过 Unit 的 `design_constraints.review_evidence` 或 `implementation.feasibility_checks[].review_evidence` 验收，写明 artifact、check、pass_criteria。源码和编译产物可验证静态关系；实际重叠、执行时序等必须有对应版本的外部设备采集证据，不能由静态配置或普通数值通过替代。

### Agent 内部契约兼容

以下是执行 Agent 处理历史输入的规则，不进入用户问卷、使用说明或待办清单。旧 `kind: structural`、`probe_events` 和 `probe_predicates` 显式报错，执行 Agent 将定位和证据交回负责该任务的编排 Agent，由其组织契约修订与复验，不要求使用者了解 probe 或手工修改文件。修订时逐项保留设计引用和原义务标识，将纯结构承诺转入 review_evidence，同时更新 obligation_ids；可从公开输入输出判定的行为改为 numerical、semantic 或 device 义务并保留用例。混合用例保留数值/行为断言，不能仅删除探针断言就宣布原结构承诺已通过。冻结任务由编排 Agent 在启动前或下一轮组织设计及测试契约修订后重新验收，实现节点不得自行改写；只有业务需求、支持范围或验收标准发生变化时才向用户确认。

## 输入不变性

```json
[{"kind":"input_unchanged","inputs":["q","k","v"]}]
```

inputs 为非空、无重复的公开 Tensor 参数名；不存在或非 Tensor 参数拒绝执行。runner 在传给目标入口的实际设备 Tensor 上取调用前快照，调用后检查 shape、dtype、device、stride 和逐位内容，NaN 位模式与 signed zero 也不忽略。检查在测试侧执行，不进入算子、编译工厂或性能计时。可联合 Golden、连续调用和重复验证；每次调用独立检查，不允许把前一调用的快照当作后一调用输入。

numerical 仍以 `match_golden` 为必要依据；semantic 至少有输出形状、类型、与输入关系或 input_unchanged 断言；device 至少检查 output_device。

## 连续调用与重复

用一个 execution 控制本行的调用序列。先执行 inputs_json，再执行 calls；每次均按当前输入运行独立 Golden/断言，整个序列在同一进程重复。repeat 为 1..100，附加 calls 最多 32 个，不允许嵌套 execution。

```json
[{"kind":"execution","repeat":3,"bitwise":true,"calls":[
  {"inputs":{"x":{"shape":[3,4],"dtype":"float32","fill":"ones"},
             "y":{"shape":[3,4],"dtype":"float32","fill":"zeros"}},"seed":7}
]}]
```

calls 的 inputs 使用同一公开输入配方；seed 缺省沿用本行，assertions 缺省沿用本行非 execution 断言，显式提供时仍须满足绑定义务类型。必要时随附加调用调整输入条件和 shape 断言。bitwise=true 要求 repeat≥2，比较相同序列位置不同轮次的输出存储位，首轮输出立即克隆，避免返回缓冲后续被覆盖。是否要求 bitwise 来自 spec/设计；不把它推广到所有算子。

调用不会重新导入目标模块，不保证公开入口复用同一编译句柄；句柄生命周期仍需对应的实际设计证据。dst_args 序列当前复用本行 output_json，因此选择与该输出布局兼容的输入；需要不同输出分配布局时记录契约缺口，不伪装成已测。
