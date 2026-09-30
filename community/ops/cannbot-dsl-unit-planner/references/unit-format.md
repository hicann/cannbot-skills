# Unit YAML 格式

每份 `U*.yaml` 只有一个 `unit` 对象，描述可交付能力、实现边界、前置事实与检验义务。

```yaml
unit:
  unit_id: U01
  title: <可交付的纵向能力>
  kind: minimal_operator # 或 pipeline_framework / feature_increment
  design_sources:
    - file: <设计资料中的实际文件路径>
      section: <对应章节或阶段>
    - file: <相关执行路径的实际文件路径>
      section: <对应分支内容>
  depends_on: []
  dependency_reasons: {} # 非 root：每个依赖 ID 对应真实能力或明确的先后要求
  requires: []
  provides: [<完成后可使用的能力>]
  boundary:
    independent_capability: <可观察的新能力>
    failure_isolation: <失败时可保留的前序结果>
    merge_rejected_because: <为何不与前一单元合并；首单元写 root>
  design_constraints:
    - source: <设计实际文件与章节>
      commitment: <受约束的执行域/数据通路/同步/资源/复杂度等>
      obligation_ids: [VO-U01-CORE] # 可引用自身或依赖闭包义务
      review_evidence: [] # 或 [{artifact: <计划核对的产物>, check: <检查内容>, pass_criteria: <通过判据>}]
  scope:
    supported_domain: <完成时支持的布局、dtype、分支、属性、可选输入及规模>
    implementation_surfaces: [<设计中的模块、阶段或分支>]
    source_files: [] # 只填已确认的实际路径
    out_of_scope: [<后续能力>]
  implementation:
    feasibility_checks: [] # 关键未决项按下方格式填写；没有则为空
    entry: <最小端到端或增量实现>
    steps: [<必要的实现动作>]
  verification_obligations:
    - id: VO-U01-CORE
      kind: numerical # 或 structural / semantic / device
      design_sources: [<Validation.md 的实际路径>, <设计资料中的实际文件路径>]
      behavior: <必须成立的行为>
      must_distinguish: [<至少一种可能混淆的错误实现>]
      observable_result: <可从输出或运行证据观察的结果>
      coverage_axes: [<需要变化的输入或执行维度>]
      input_conditions: [] # 可选；可物化验证的 dimension/numel/count/unique_count 条件
  failure_return: <具体失败证据、返回的设计章节、受影响能力与后继阻断；修订后同步重划>
```

`supported_domain`、`dependency_reasons`、`design_constraints` 与 `feasibility_checks` 为新增规划字段，不改变原有 kind 和检验义务 kind。无相关设计承诺时 `design_constraints` 可为空；约束引用自身或前序义务，增量只记录新增或改变的约束。已有 Unit 迁移时补充实际事实，不虚填已验证。

关键未决项格式（置于 `implementation` 内）：

```yaml
feasibility_checks:
  - design_source: <实际文件与章节/未决项 ID>
    question: <需要确认的组合能力或设备行为>
    known_evidence: <已有证据路径及层级；无证据写未验证>
    required_evidence: <补证适用环境、真实形状/执行路径及成功判据>
    before_step: <本 Unit 首次依赖该能力的动作>
    obligation_ids: [<验证该补证判据的真实义务 ID>]
    review_evidence: [] # 独立检查按 artifact/check/pass_criteria 声明
    on_failure: <返回具体设计章节，阻断本能力及依赖后继>
```

补证是 Unit 内的前置实现动作，不新增 probe Unit kind 或性能义务 kind。`provides` 仅在补证、实现与检验义务均通过后成立；记录前置检查不代表已经通过。

首个 `minimal_operator` 应有数值检验义务；独立的 `pipeline_framework` 应有验证真实生产/消费及设计执行域的结构义务。已有设计 Case 可作为来源线索，但 Unit 不预分配 Case ID。若某项义务无法设计可区分的测试，应调整 Unit 边界或回到设计文档澄清事实。

新增规划字段描述证据要求与边界，不代表自然语言设计承诺或人工证据已经通过。可测试的设计约束与补证判据必须同时写入 `verification_obligations`；不能由公开输入和可观察结果判定的项，明确源码、编译产物或设备证据的检查方法。多块、特殊输入关系、连续调用或并行关系等覆盖要求，不能仅因关联的 Case ID 存在就算完成。

验证义务必须有可执行判定：numerical 检查独立参考结果；structural 检查真实执行事件、编译产物或设备 trace；semantic 检查输出形状、类型、与输入的关系或可观察行为；device 检查实际设备行为。错误码行为须有公开接口异常验证。性能目标写入设计验证方案，明确测量条件与通过判据，不用普通数值验证代替性能证据；无法判定的义务应指出缺口，不能仅关联 Case ID 算覆盖。

结构义务的 `must_distinguish` 应覆盖设计可能被数学等价路线替代的风险；`observable_result` 指明事件的实际采集来源与断言，不以静态常量、阶段名称或 Host 按预期生成的事件自证。多块/复用的 `coverage_axes` 使用实际有效工作量与状态存活条件，具体判别示例见 [可实现性与边界检查](decomposition-checks.md)。

`design_constraints` 与 `feasibility_checks` 每项至少填写 obligation_ids 或 review_evidence，后者每项必须有 artifact、check、pass_criteria。不要只写自由文本 evidence。可测试内容写到义务，源码/编译/设备检查由调用方验收。

`input_conditions` 描述检验义务成立所需的输入事实，例如 `{input: x, metric: dimension, axis: -1, relation: ge, value: 129}`；数值取自当前设计，不使用案例中的固定阈值。metric 为 dimension（必须有 axis）、numel、count 或 unique_count；count/unique_count 可按 axis 逐行检查，并以 where 的 min（含下界）、max（不含上界）、eq 过滤有效值。过滤边界为有限数或 `{input: <公开输入名>, dimension: <轴>}`；relation 为 eq/ne/ge/gt/le/lt，value 为有限数。条件应在实际输入上成立；无法表达的事实明确证据缺口。coverage_axes 仍描述需要覆盖的完整行为，不以输入谓词替代设计符合性检查。
