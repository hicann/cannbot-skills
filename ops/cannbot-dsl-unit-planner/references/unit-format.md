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
      review_evidence: [] # 或 [{artifact: <计划核对的产物>, check: <检查内容>, pass_criteria: <通过判据>, reject_examples: [<看似满足但不算通过的做法>]}]
  scope:
    supported_domain: <完成时支持的布局、dtype、分支、属性、可选输入及规模>
    domain_constraints: # 可机器校验的声明域；必填，无法写出时返回设计修订
      - {input: <公开输入名>, metric: dimension, axis: -1, relation: multiple_of, value: 64}
    implementation_surfaces: [<设计中的模块、阶段或分支>]
    source_files: [] # 只填已确认的实际路径
    out_of_scope: [<后续能力>]
  implementation:
    feasibility_checks: [] # 关键未决项按下方格式填写；没有则为空
    entry: <最小端到端或增量实现>
    steps: [<必要的实现动作>]
  verification_obligations:
    - id: VO-U01-CORE
      kind: numerical # 或 semantic / device
      design_sources: [<Validation.md 的实际路径>, <设计资料中的实际文件路径>]
      behavior: <必须成立的行为>
      must_distinguish: [<至少一种可能混淆的错误实现>]
      observable_result: <可从输出或运行证据观察的结果>
      coverage_axes: [<需要变化的输入或执行维度>]
      input_conditions: [] # 可选；可物化验证的 dimension/numel/count/unique_count 条件
  failure_return: <具体失败证据、返回的设计章节、受影响能力与后继阻断；修订后同步重划>
```

`supported_domain`、`dependency_reasons`、`design_constraints` 与 `feasibility_checks` 描述实现范围与设计承诺；Unit kind 表达能力类型，检验义务使用 numerical、semantic、device。无相关设计承诺时 `design_constraints` 可为空；约束引用自身或前序义务，增量只记录新增或改变的约束。修订已有 Unit 时由规划 Agent 补充实际事实，不虚填已验证。

`domain_constraints` 是 `supported_domain` 的可执行子集，测试生成节点写入 TDD 用例后会逐条核对 active 用例的 `inputs_json`（含 `execution.calls`）；用例输入越界或字段缺失即终止该节点。域校验器对 `raises_error` 的豁免不表示 TDD 接受该 expected；配套 TDD 仍只接受 `match_golden` 或 `assertions`，错误码使用黑盒入口及 Unit 的 review_evidence，见[可实现性与边界检查](decomposition-checks.md)。字段含义：

- `input`：公开输入名；`metric: value` 时为 spec 声明的标量属性名。
- `metric`：`dimension`（单轴长度，须配 `axis`，支持负轴）、`numel`（元素总数）、`dtype`、`value`（标量属性值）。
- `relation`：`eq`、`ne`、`le`、`lt`、`ge`、`gt`、`multiple_of`、`in`；`value` 为数值、字符串、布尔或列表。
- `default`：可选，仅 `metric: value`；用例未提供该标量属性时按此值比较。

只声明公开输入自身的属性，不写由多个输入推导出的量；跨输入关系（例如两维必须整除、某维不得超过另一输入的长度）无法表达的，保留在 `supported_domain` 并列为未机器校验项。本 Unit 相对前序收窄或放宽输入域时，按本 Unit 实际允许的取值声明，不照抄已失效的前序约束（例如前序要求某维等于 1、本 Unit 允许更大值时，不要继续写 `eq: 1`）。有可执行义务却无法写出任何可机器校验的约束时，按缺少依据的未决项记录并返回设计修订，不留空冒充已声明；不得用自然语言描述代替 `domain_constraints`。最小输入按当前 Unit 对应设计路径/能力及前序保留面核对，沿用[可实现性与边界检查](decomposition-checks.md)的唯一口径，不要求全部 Unit 包含同一全局样例。引用已有场景或原规则推导；依据缺失、与规格冲突或覆盖不能闭合时报告设计缺口，不通过改用例形状迁就。

关键未决项格式（置于 `implementation` 内）：

```yaml
feasibility_checks:
  - design_source: <实际文件与章节/未决项 ID>
    question: <需要确认的组合能力或设备行为>
    known_evidence: <已有证据路径及层级；无证据写未验证>
    required_evidence: <补证适用环境、真实形状/执行路径及成功判据>
    before_step: <本 Unit 首次依赖该能力的动作>
    obligation_ids: [<验证该补证判据的真实义务 ID>]
    review_evidence: [] # 独立检查按 artifact/check/pass_criteria/reject_examples 声明
    on_failure: <返回具体设计章节，阻断本能力及依赖后继>
```

补证是 Unit 内的前置实现动作，不新增探针 Unit 或性能义务 kind。`provides` 仅在补证、实现与检验义务均通过后成立；记录前置检查不代表已经通过。

首个 `minimal_operator` 应有数值检验义务；独立的 `pipeline_framework` 应有可执行行为义务，并通过 review_evidence 验证真实生产/消费及设计执行域。已有设计 Case 可作为来源线索，但 Unit 不预分配 Case ID。若某项行为义务无法设计可区分的测试，应调整 Unit 边界或回到设计文档澄清事实。

新增规划字段描述证据要求与边界，不代表自然语言设计承诺或人工证据已经通过。当前 TDD 契约可表达的数值和行为判据写入 `verification_obligations`；错误码拒绝行为由黑盒入口及 `review_evidence` 承接，不强挂异常 TDD；其余不可由公开输入和结果判定的项明确源码、编译或设备证据方法，原要求及到期责任均保留。多块、特殊输入关系、连续调用或并行关系等覆盖要求，不能仅因关联的 Case ID 存在就算完成。

验证义务必须有可执行判定：numerical 检查独立参考结果；semantic 检查输出形状、类型、与输入的关系或可观察行为；device 检查实际设备行为。错误码行为须有现有黑盒公开接口异常验证，证据与 Unit 的责任及时机通过 review_evidence 关联；不能用普通输出断言替代，也不能仅凭关联 Case ID 宣称通过。性能目标写入设计验证方案，明确测量条件与通过判据，不用普通数值验证代替性能证据；无法判定的义务应指出缺口，不能仅关联 Case ID 算覆盖。

结构检查的 `check`、`pass_criteria` 与 `reject_examples` 应覆盖实际违反原设计承诺的风险，包括数值正确但必要执行结构不符合的情况，`artifact` 指明源码、编译产物或外部设备证据；不以静态常量、阶段名称或 Host 按预期生成的事件自证。`reject_examples` 至少写一条实际违反原承诺的具体做法，说明违反条款与判失败依据；跳过必要步骤、错误索引或过早复用资源等适用的局部错误即可，不要求另造完整算法，也不拒绝设计允许的回退或等价表达。必要性能与结构承诺完整保留，只写正向存在性判据不可验收。可执行义务中多块/复用的 `coverage_axes` 使用实际有效工作量与状态存活条件，具体判别示例见 [可实现性与边界检查](decomposition-checks.md)。

`design_constraints` 与 `feasibility_checks` 每项至少填写 obligation_ids 或 review_evidence，后者每项必须有 artifact、check、pass_criteria、reject_examples（至少一条明确判失败的做法）。不要只写自由文本 evidence。当前 TDD 可表达的内容写到义务；错误码及源码/编译/设备证据按上述安排由调用方验收，不删除原承诺。

`input_conditions` 描述检验义务成立所需的输入事实，例如 `{input: x, metric: dimension, axis: -1, relation: ge, value: 129}`；数值取自当前设计，不使用案例中的固定阈值。metric 为 dimension（必须有 axis）、numel、count 或 unique_count；count/unique_count 可按 axis 逐行检查，并以 where 的 min（含下界）、max（不含上界）、eq 过滤有效值。过滤边界为有限数或 `{input: <公开输入名>, dimension: <轴>}`；relation 为 eq/ne/ge/gt/le/lt，value 为有限数。条件应在实际输入上成立；无法表达的事实明确证据缺口。coverage_axes 仍描述需要覆盖的完整行为，不以输入谓词替代设计符合性检查。

## 结构验收

算子文件不包含 probe 实现、调用、参数、开关或相关注释，也不生成 `probe/`。结构承诺保存在 design_constraints 或 feasibility_checks 的 review_evidence 中，明确源码、编译产物、外部设备证据及检查判据。需要真实时序的承诺不能只用源码证明。

pipeline_framework 必须同时具备非空的可执行行为义务和至少一项完整 review_evidence；不为满足字段制造空壳用例。仅有结构检查、没有可交付行为的片段合并进相应能力 Unit。

历史输入的兼容由执行本 Skill 的 Agent 与编排 Agent 协作处理，不作为使用者的操作前提，也不要求使用者理解或修改内部契约。遇到旧 structural 义务时按原设计引用逐项修订，保留原义务标识到新证据的对应关系，并同步 obligation_ids 与用例；校验器拒绝旧类型，不自动将其视为已完成。冻结任务由编排 Agent 在启动前或下一轮组织修订后重新验收，不能让实现节点自行删除原义务；不改变已确认业务要求时不向用户追加迁移问卷。
