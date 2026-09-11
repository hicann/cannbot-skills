# GE 流语义

> 版本提示：语义规则和源码索引应随实际 GE/CANN 版本复核。源码函数名、路径和行号
> 不能替代日志证据，也不能单独证明调用栈或流数量。

## 三层 ID

- `logic_stream_id`：编译期写入算子/TaskDef 的逻辑标识，不是 RT 对象。
- `model_stream_index`：静态模型资源中的流索引，运行时通过 `Logical stream index` 绑定。
- `rt_stream_id`：RT 实际创建的设备执行队列 ID。`Create new stream` 可能包含模型流和辅助流；外部执行入口日志不解析。

同一数字在不同阶段不自动等价，报告必须带层次前缀。

## Shape 与图形态

使用正交属性：

```text
scenario_class: static | dynamic | hybrid | unknown
shape_mode: static_shape | dynamic_shape | unknown
graph_form: single_graph | dynamic_batch_static_branches |
            hybrid_dynamic_static_subgraph | unknown
stream_policy: single_stream | multi_stream | unknown
physical_split: not_applicable | no_split_observed | explicit_split |
                delegated_to_rts | unknown
```

分析器还需要为每个图保存：

```text
graph_class: static_shape | dynamic_shape | unknown
compile_path: known_shape | dynamic_shape | unknown
runtime_path: v1_davinci | v1_hybrid_dynamic | v1_hybrid_known_submodel |
               v2_rt2 | unknown
               stream_scope: graph | submodel | runtime_model
```

`scenario_class` 是模型场景大类：普通 known-shape 模型为 `static`，动态 Shape 或动态
Batch 单独出现时为 `dynamic`；只有同一编译族中另外明确存在动态外层/unknown 图与 known
子图边界时才为 `hybrid`。动态 Batch 生成的 `Batch_N` known 分支本身不构成 Hybrid 证据。
因此动态 Batch 的 `shape_mode` 也应报告为 `dynamic_shape`，表示运行期需要选择档位；这
不否定每个 `Batch_N` 计算分支的 `graph_class=static_shape`。
`root_graph_class` 是更细的根图结构字段，不能用它覆盖每个图的局部路径。动态 Batch
即使根图和 `Batch_N` 分支走 known-shape 编译，也不能把模型场景写成普通静态；其
`graph_form=dynamic_batch_static_branches`，分支的 `compile_path=known_shape`，运行通常
走 V1 `DavinciModel`，而不是 V1 Hybrid 动态执行器。

动态 Batch 是动态控制结构选择多个已知 Shape 的 Batch 子图，模型归属属于动态大类；一般
动态 Shape 静态子图不一定是 Batch。只有日志实际出现的 Batch 设置、label 或分支控制
证据才能确定本次激活的具体档位。

## 编译与运行边界

编译阶段只分配 `stream_id`；实际 RT stream 对象在模型加载/执行阶段创建。静态图可因 Task/SQE 上限发生物理拆流；`RefreshRealStream` 只表示刷新，`SplitNodesToNewStream` 才是明确 GE 拆流证据。动态图多流通常用节点 Event 属性驱动运行期 Record/Wait，不等同于插入 Send/Recv。

编译—运行会话不依赖模型名。V2 优先使用 `Build RT2 executor` 的 graph name 与编译图名精确匹配建立 `strong` 关联（编译和运行 PID 可以不同），并把每条 executor 日志作为运行模型的显式边界；同一 executor 窗口内再按运行 PID、时间顺序和 logical index→RT ID 交集无冲突规则吸收跨 context/分卷文件的资源摘要、`Collect rt2 stream` 和 `SplitRtStreams` 片段。资源摘要重复不是模型边界。没有 graph name 时才使用无图名的 V2 片段，以运行 PID 为硬边界。V1 运行记录仍要求编译/运行 PID 和 GE 内部上下文编号相同、已完成根图早于 `InitRuntimeParams`、时间差不超过 5 秒、编译逻辑流数或最终模型流数等于运行 `model_stream_count`，且中间没有其他根图完成或运行初始化。唯一候选标记 `same_session`，条件缺失、绑定冲突或候选有歧义时标记 `unknown`，并输出对应 `correlation_evidence`。

实现上采用一次有序事件扫描：没有图名的 V2 trace 先暂存为运行片段，扫描结束后按上述边界条件归并并绑定到唯一 workflow；一个 workflow 可吸收多个兼容片段，已吸收片段不再成为独立 runtime-only 会话。每个 workflow 独立保存 phases、evidence、流和算子映射，禁止使用全局首条运行证据回填其它图。Markdown 只呈现流证据和映射，输入文件列表仅作为 JSON 上下文 provenance。

## 运行期角色

- 模型流：由 `InitRuntimeParams.stream_num` 和 `Logical stream index` 绑定的流。
- HCCL/follow 流：只有日志显式给出时才统计。
- 辅助流：特殊 flag、资源维护流或未能绑定到模型流的 GE 创建流，单列。
- `Create new stream` 总数不等于模型流数；未出现在 `Logical stream index` 绑定中的 GE 创建流应显示为未绑定/辅助流。
- V1 日志没有 V2 资源摘要时，V2 申请/reusable/attached 字段为 `not_applicable`（JSON 仍为 `null`），不能把编译期的 `attached stream num` 当成 V2 attached 数。V2 只有同时出现函数/源码受控的 `Collect rt2 stream` GE INFO 日志才记录申请绑定：`OccupyStreamResource` 对应 logical 0，`AcquireStreams` 对应 allocator 新建的其余 logical index；普通函数名或 `AcquireStreams(...)` 参数行不算证据。

### V2 Kernel Trace 运行期流

`KERNEL_TRACE` 宏展开为 GE 的 `[INFO]` 日志。新版本优先使用上面的 `Collect rt2 stream`。
旧版本的 `SplitRtStreams` 或带版本/序号后缀的
`SplitRtStreams_<suffix>` 的
`Get rts stream ... from logical stream <i>, rts stream_id: <id>` 是运行期逻辑流到
RT stream 的直接绑定事实，应输出为 `execution_bindings`，仅在新增绑定缺失时作为
`split_rt_streams_fallback`。指针地址不参与 ID 判断；相同 logical index/RT ID 去重，冲突保留。
没有任一绑定日志时不能将物理流映射填为 0 或宣称未申请流。
Event/Notify 的 Kernel Trace 记录和流绑定分开计数。

### 动态外层图与静态子模型

- 动态编译流数只接受带 `AssignStreamsForDynamicShapeGraph` 函数名的
  `Graph: <graph>, stream num: <n>, event num: <n>` 成功汇总；
  `RefreshInfoOfDynamicShapeGraph` 的 Root/Sub/Total 文本直接忽略。
- `Start assign attached stream for graph <name> with subgraph num` 可能在子图构建之后才出现；有对应真实 known/unknown 图入口时只作为该窗口的范围注记。若没有真实图入口，该 attached-stream 行不得创建虚拟 workflow，也不得进入模型会话。
- V2 的 `Root graph`、`Root model` 和 `Static sub model` 资源摘要按作用域保存在
  `runtime.v2_resource_summaries`。它们不能建立或切分模型边界：完全相同的重复/分卷记录只保留一条规范化摘要，并把其余位置保存在 `v2_resource_duplicate_evidence`；不同 scope/name 可在同一 executor 中共存，同 scope/name 的数量不一致保存在 `v2_resource_conflicts`。
- 动态子图名称符合 `<root>_sub_<N>_unknow(n)`、且没有 `AssignStreamsForDynamicShapeGraph` 和编译逻辑流证据时，不能把每个子图渲染为独立 unknown 图。若匹配到同名 RT2 executor 且运行期只有 logical 0，编译子图标记为 suppressed，运行期保留为 `runtime_graph_identity=dynamic_subgraph_family`；默认映射只显示运行 logical 0 与 RT stream，不显示 unknown 编译 ID 占位。
- 动态外层图、While/Case 等 known/unknown 子图共享 `compile_family_id`。没有显式 `GraphRelation` 时，只能确认同一编译族，不能填写具体父图、父节点或子图索引。
- 运行期 `DavinciModel` 的 `InitRuntimeParams.stream_num`/`Logical stream index` 只描述当前加载的静态模型对象。
- 同一 PID、GE context、model_id 和参数完全相同、间隔不超过 1 ms 且尚未创建/绑定流的连续 `InitRuntimeParams` 视为重复初始化；主 session 只保留一次阶段证据，重复行保存在 `duplicate_init_evidence`。
- V1 Hybrid 的初始化/执行器字符串若实际出现，保存为 `hybrid_context_markers`；这些标记不改变
  `runtime_path`，也不参与流数量和逻辑流到 RT 流映射的计算。

## 图层级

- `At last, root graph` 只表示当前图编译范围的逻辑流汇总，不证明该图是整个模型的全局根图。
- `Start assign attached stream ... with subgraph num` 只提供资源范围和子图数量，不提供具体父图、父节点或子图索引。
- 只有显式 `[GraphRelation] parent_graph ... parent_node ... child_graph ... subgraph_index ...` 记录才建立父子关系；缺少时关系必须为 `unknown`。

## 算子逻辑流证据

- `StreamAllocator::SetLogicStreamIdAttr` 遍历 known-shape/static-subgraph 的所有节点，输出完整的 `Op ... logic stream id is ...` 映射，是编译期主证据。
- `SplitStreams` 的 `node: ..., logical stream id: ...`、`DynamicStreamAllocator::RefreshStreamsForGraphByNodeIds` 的 `Refresh stream by node ids ...` 以及各逻辑流 Pass 的 `Node ... assigned/reassign ...` INFO 日志，是拆流、动态重编号和流变更的补充证据。
- GE 仓 `runtime/v1/.../task_info` 中带节点名的 `logic stream id` 日志可用于运行期 Task 到逻辑流的关联；没有节点名时只作为流级证据。
- 外部 RTS/Runtime 组件的 `ModelExecuteTask`、`ConstructSqeForModelExecuteTask` 等日志不属于 GE 语义范围，解析器必须忽略。
