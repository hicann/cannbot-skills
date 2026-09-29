# Unit YAML 格式

每份 `U*.yaml` 只有一个 `unit` 对象。Unit planner 给出实现边界与检验义务；测试设计再确定具体输入、Case、Golden、断言和执行命令。

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
  requires: []
  provides: [<完成后可使用的能力>]
  boundary:
    independent_capability: <可观察的新能力>
    failure_isolation: <失败时可保留的前序结果>
    merge_rejected_because: <为何不与前一单元合并；首单元写 root>
  scope:
    implementation_surfaces: [<设计中的模块、阶段或分支>]
    source_files: [] # 只填已确认的实际路径
    out_of_scope: [<后续能力>]
  implementation:
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
  failure_return: <实现问题或设计缺口的返回点>
```

首个 `minimal_operator` 应有数值检验义务；独立的 `pipeline_framework` 应有结构检验义务。已有设计 Case 可作为来源线索，但 Unit 不预分配Case ID。若某项义务无法设计可区分的测试，应调整 Unit 边界或回到设计文档澄清事实。

验证义务必须有可执行判定：numerical 对应 Golden；structural 对应真实 probe_events；semantic 对应输出 shape、dtype、与输入关系或实际事件的断言；device 对应输出设备断言。错误码行为由黑盒异常用例验证。性能目标写入设计验证方案并由性能阶段实测，不用普通数值 TDD 代替性能证据；无法表达的义务应指出缺口，不能仅关联 Case ID 算覆盖。
