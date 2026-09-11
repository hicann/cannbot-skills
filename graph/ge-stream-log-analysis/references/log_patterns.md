# GE INFO 流日志模式

> 版本提示：本文件中的源码文件、函数和行号是迁移时用于人工核对的索引，随 GE/CANN
> commit 变化，不能单独作为当前版本 API 或调用关系的证明。分析时应以实际 `[INFO]`
> 日志为事实，并在需要时用用户提供的 `--repo-root` 对应源码版本复核。

本文只维护与 GE 流分配直接相关的日志模式，输入只接受源码确认由 GE 输出的 `[INFO]`
行。函数名、非 INFO/外部 RTS 日志不作为流事实；`[INFO][GEPERFTRACE]` 只作耗时结束证据，
不能当作函数入口、出口或调用栈。
表中标注“无专用日志”的函数只作为源码上下文，不作为日志匹配关键字。

## 1. 流 ID 和作用域

| 名称 | 含义 | 不能替代 |
|---|---|---|
| logic_stream_id | 编译期写入算子/TaskDef 的逻辑流标识 | model_stream_index、rt_stream_id |
| model_stream_index | V1 静态模型资源中的流索引 | 编译期逻辑流 ID |
| rt_stream_id | RT 实际创建的设备流 ID | 逻辑流或模型流索引 |

每个数量都要注明作用域：graph、submodel 或 runtime_model。`RefreshInfoOfDynamicShapeGraph`
打印的 Root/Sub/Total 不是流分析证据，解析器直接忽略；相同数字也不能跨 ID 层直接比较。

## 2. 编译期静态图（known-shape）

本节适用于静态根图、混合模型中的 known-shape 子图，以及动态 Batch 生成的 Batch_N
known-shape 分支。源码调用关系为：

    BuildForKnownShapeGraph
      → BuildModelForGetTask
      → AssignLogicalStreams
      → At last, root graph
      → InsertSyncNodesByLogicStream（内含 SetLogicStreamIdAttr）
      → RefreshRealStream / SplitStreamAndRefreshTaskDef

| 阶段 | 实际 GE INFO 日志消息 | 对应函数 | 证据含义和限制 |
|---|---|---|---|
| 图入口 | `Begin to build known shape graph[<graph>]` | `GraphBuilder::BuildForKnownShapeGraph`（graph_builder.cc:343） | 建立当前 known 图窗口；不证明它是全局根图 |
| Task 构建 | 没有独立成功 GE INFO 日志；可能只有 GEPERFTRACE ... GraphBuilder::BuildModelForGetTask | GraphBuilder::BuildForKnownShapeGraph → ModelBuilder::BuildModelForGetTask（graph_builder.cc:363-366） | 只能确认阶段/耗时，不能从 timing 行推断调用栈 |
| 逻辑流分配执行 | 没有独立入口 GE INFO 日志；GE_TRACE_START(AssignLogicalStreams) 不打印日志 | ModelBuilder::BuildModelForGetTask（model_builder.cc:1159-1162）→ StreamAllocator::AssignLogicalStreams（stream_allocator.cc:327） | 函数会执行，但日志中可能没有分配开始/结束行 |
| 逻辑流最终汇总 | `[Assign][LogicalStream] At last, root graph: <graph>, total stream num: <n>, main stream num: <n>, attached stream num: <n>` | `LogicalStreamAllocator::Assign`（logical_stream_allocator.cc:870） | 当前图窗口的最终逻辑流、主流和附着流数量；root graph 是日志用语，不等于全局根图 |
| 分配过程（补充） | [Assign][NewStreamId]、[Assign][StreamId]、[Update][StreamId]、[Reuse][Stream] | logical_stream_allocator.cc 中各逻辑流 Pass | 说明新分配、重分配或复用原因；中间值不能覆盖 At last |
| 算子映射 | `Op [<name>] OpType [<type>] logic stream id is <id>` | `StreamAllocator::SetLogicStreamIdAttr`（stream_allocator.cc:598-605） | known 图算子到逻辑流的最终映射；只对实际出现的算子下结论 |
| 附着流映射（补充） | logic attached stream id is ... | StreamAllocator::SetLogicStreamIdAttr（stream_allocator.cc:617-622） | 记录算子的 attached stream；不等于主逻辑流数量 |
| 同步完成 | `After InsertSyncNodesByLogicStream, graph:<graph>, stream num:<n>, notify num:<n>, event num:<n>` | `StreamAllocator::InsertSyncNodesByLogicStream`（stream_allocator.cc:628-670） | 同步节点和 Event/Notify 汇总完成；Event/Notify 不计入 RT stream |
| GE 物理流刷新出口 | `After SplitStreamAndRefreshTaskDef, graph:<graph>, stream num:<n>, ...` | `StreamAllocator::SplitStreamAndRefreshTaskDef`（stream_allocator.cc:677-709） | GE 侧物理流/TaskDef 刷新阶段的汇总；是否实际拆出具体节点要看拆流细节 |
| 拆流委托 RTS | Move split stream from ge to rts | StreamAllocator::SplitStreamAndRefreshTaskDef（stream_allocator.cc:685） | GE 不输出拆流结果，标记为 delegated_to_rts |
| 拆流细节（可选） | op name [...] is split to new stream id ...；stream[...]'s task num[...] > max_task_num_one_stream..., split stream to ... | StreamAllocator::SplitStreamForOneNode（stream_allocator.cc:1264）、SplitNodesToNewStream（stream_allocator.cc:1423） | 分别说明节点结果和超限原因；没有这些行不能补写拆流节点 |

ModelBuilder::RefreshRealStream（model_builder.cc:1146）本身没有正常 GE INFO 日志，只调用
SplitStreamAndRefreshTaskDef。因此 RefreshRealStream 的 timing 行只能说明刷新阶段耗时，
必须以 After Split... 或 Move split... 判断物理拆流状态。

静态图数量优先级为：At last 和最终算子映射 > 同步/拆流后的汇总 > Assign/Reuse 中间
记录。没有最终汇总时保持 unknown，不能因为出现默认流 ID 0 就判定为单流。

## 3. 编译期动态图（unknown-shape）

动态图外层会遍历 root 和子图；known 子图仍走第 2 节，不能把整个编译族合并成一条
动态流表。动态路径的关键关系为：

下面流程仅表示源码调用关系，不代表每个函数名都会出现在 GE INFO 日志中。

    BuildForDynamicShapeGraph（外层，可选）
      → BuildForUnknownShapeGraph（root/unknown 图入口）
      → [root 调用范围] DynamicStreamAllocator::AssignStreamsForDynamicShapeGraph
      → BuildModelForGetDynShapeTask

动态流分配日志以 DynamicStreamAllocator::AssignStreamsForDynamicShapeGraph（复数）为准。
该函数成功完成分配和事件插入后，才会打印 Graph: ... 汇总；不要用其他函数名代替这条
日志模式。该函数一次调用会处理 root graph 及其 unknown 子图，子图入口不意味着每个子图
都会各自打印一条 Graph 汇总。未出现该行时只能判定缺少成功出口证据，不能用函数名、
配置行或 timing 行补写流数。

| 阶段 | 实际 GE INFO 日志消息 | 对应函数 | 证据含义和限制 |
|---|---|---|---|
| 动态外层入口 | `Start to build BuildForDynamicShape for dynamic shape.` | `GraphBuilder::BuildForDynamicShapeGraph`（graph_builder.cc:596-599） | 动态根图编排开始；不代表某个子图已分配流 |
| unknown 图入口 | `Begin to build unknown shape graph[<graph>]` | `GraphBuilder::BuildForUnknownShapeGraph`（graph_builder.cc:460-462） | 为每个 unknown 图建立独立窗口 |
| 动态分配入口（无专用成功日志） | 没有“开始分配”GE INFO 日志；ac_parallel_enable: ... 只是配置读取日志 | DynamicStreamAllocator::AssignStreamsForDynamicShapeGraph | 函数可能执行，但配置行不能证明分配成功；以复数函数的 Graph 汇总为出口 |
| 子图/引擎分配 | `Assign stream_id: <id> for engine: <engine>, subgraph: <name>.` | `DynamicStreamAllocator::AssignStreamForSubgraph`（仅对应此条子图日志；dynamic_stream_allocator.cc:171-180） | 动态子图/引擎的逻辑流分配 |
| 子图映射 | `[Assign][StreamId] <id> for Subgraph <name> (engine: <engine>).` | `DynamicStreamAllocator::SetSubgraphStreamToNodes`（dynamic_stream_allocator.cc:409-415） | 子图到逻辑流的映射；逐节点 DEBUG 不纳入本规则 |
| 动态重分配 | `Node: <name>, stream_label: <label>, reassign stream id: <id>.` | `DynamicStreamAllocator::ReassignStreamByStreamLabel`（dynamic_stream_allocator.cc:444-462） | stream label 导致的节点重分配 |
| 动态节点刷新 | `Refresh stream by node ids of graph: <graph>, stream_id: <id>, type: <type>, name: <name>.` | `DynamicStreamAllocator::RefreshStreamsForGraphByNodeIds`（dynamic_stream_allocator.cc:622-640） | 节点级逻辑流刷新；不是物理 RT stream 创建 |
| 动态分配汇总（出口） | `Graph: <graph>, stream num: <n>, event num: <n>.` | `DynamicStreamAllocator::AssignStreamsForDynamicShapeGraph`（dynamic_stream_allocator.cc:63-73） | 当前 root/unknown 图族一次分配调用的流/Event 汇总；缺失时该动态汇总为 unknown |
| 动态跨流同步（可选） | `Insert event <id> between node ... (stream <a>) and ... (stream <b>)` | `DynamicStreamAllocator::InsertEventBetweenTwoNodes`（dynamic_stream_allocator.cc:693-714） | 动态 Event 依赖；不等于静态图物理拆流 |
| 外层 attached 范围（可选） | `Start assign attached stream for graph <graph> with subgraph num: <n>` | `AssignAttachedStreamPass::Run`（assign_attached_stream_pass.cc:33-36） | 有真实图入口时只作为范围注记；没有入口时忽略，不建立虚拟图窗口或模型会话 |

动态图分配器没有静态 SplitStreamAndRefreshTaskDef 等价的物理拆流日志。动态路径中的
Refresh stream ... 是逻辑流重编号；若日志只有外层 Root/Sub/Total 聚合而没有带
`AssignStreamsForDynamicShapeGraph` 函数的 `Graph: ... stream num ... event num ...`，
该 unknown 图的动态流数、动态出口和窗口完成状态都必须保持 `unknown`。
带图名的动态 allocator 汇总在部分 GE INFO 日志版本中可能没有单独的图入口，仍可作为该图的
兼容性恢复依据；但 attached-stream 范围行没有入口时只作未绑定注记，不得反向补建图。

## 4. 运行期实际申请和绑定 stream

运行期按后端分析，不按“静态/动态”字样猜测。执行入口和 Task 下发日志只能证明执行
发生，不能替代 stream 申请证据。

### 4.1 V1 DavinciModel

适用于普通静态模型、动态 Batch 的 known-shape Batch_N 分支，以及 Hybrid 中实际加载的
known 子模型。源码关系为：

    InitRuntimeParams（资源需求）
      → GetOrCreateRtStream
      → ReusableStreamAllocator::CreateNewStream（实际创建）
      → Logical stream index（模型流索引 → RT ID）

| 阶段 | 实际 GE INFO 日志消息 | 对应函数 | 证据含义和限制 |
|---|---|---|---|
| 模型资源需求 | `InitRuntimeParams: model_id=<id>, ... stream_num:<n>, notify_num:<n>, event_num:<n>, label_num:<n>` | `DavinciModel::InitRuntimeParams`（runtime/v1/graph/load/model_manager/davinci_model.cc:691-698） | 当前模型对象声明的资源数量，不等于已创建的 RT stream |
| 实际创建 | `Create new stream: <ptr>, rt stream id: <id>, rt model id: <id>, priority: <p>, stream flag: <f>, task num: <n>.` | `ReusableStreamAllocator::CreateNewStream`（runtime/v1/graph/load/model_manager/reusable_stream_allocator.cc:35-41） | 一个 GE 侧 RT stream 创建事实；GetOrCreateRtStream 本身无独立日志 |
| 模型流绑定 | `Logical stream index: <i>, rtstream: <id>, model: <id>, stream flag: <f>.` | `DavinciModel::InitRuntimeResource`（davinci_model.cc:1436-1484） | model_stream_index → rt_stream_id；这是运行期主绑定证据 |
| 模型/辅助总数 | `model_id=<id>, model total stream num: <total>, model stream num: <model>, hccl follow stream num: <follow>` | `ModelExecutor::GetStreamNum`（runtime/v1/graph/execute/model_executor.cc:904） | 包含模型流、HCCL/follow 流和执行辅助流，不能当作逐条创建数 |
| 绑定上下文（可选） | `model_id=<id>, stream id <i>, first task id <t>`；`aclmdlRIBindStream[...]` | `DavinciModel::BindModelStream`（davinci_model.cc:707-736） | 说明模型绑定顺序；不替代 Logical stream index |

InitModelStream、ModelRunStart、AddHeadStream 等调用路径可能触发 GetOrCreateRtStream，
但没有各自的成功 GE INFO 日志；实际创建仍以 Create new stream 为准。传入用户 stream 时可能
没有创建日志。没有 Create new stream 不能推断“没有申请”。

没有对应 Logical stream index 的 Create new stream，单列为辅助/未绑定流；不要把所有
Create new stream 行直接相加作为模型流数。

### 4.2 V2/RT2

V2 的 GE INFO 日志主要描述模型资源需求：

| 日志 | 对应函数 | 证据 |
|---|---|---|
| `Static sub model <name>, stream_num ..., event_num ..., notify_num ...` | `ModelConverter::GetNonRootModelResourceNum`（runtime/v2/lowering/model_converter.cc:60） | 静态子模型资源需求 |
| `Root graph total stream_num ..., reusable stream num ..., attached stream num ...` | `ModelConverter::GetReusableStreamResourceNum`（model_converter.cc:366） | root aggregate/reusable 资源需求 |
| `Root model <name>, total_stream_num ..., attached_stream_num ...` | `ModelConverter::GetReusableStreamResourceNum`（model_converter.cc:380） | root 模型资源需求 |

V2/RT2 运行期的主绑定日志为以下三类 GE INFO 日志：

| 阶段 | 实际 GE INFO 日志消息 | 对应函数/源码 | 证据含义 |
|---|---|---|---|
| executor 身份 | `Build RT2 executor for root compute graph[<graph>], model[<model>].` | `ModelV2ExecutorBuilder::Build`（runtime/v2/core/builder/model_v2_executor_builder.cc） | 建立一个 V2 runtime workflow；`<graph>` 与编译图名精确相同时作为 `strong` 关联锚点 |
| 外部主流 | `Collect rt2 stream, get rts stream <ptr> from logical stream 0, rts_stream_id: <id>` | `ModelV2Executor::OccupyStreamResource`（runtime/v2/core/model_v2_executor.cc） | 外部 `arg.stream` 绑定到 logical stream 0 |
| allocator 辅助流 | `Collect rt2 stream, get rts stream <ptr> from logical stream <i>, rts_stream_id: <id>` | `StreamAllocator::AcquireStreams`（base/common/allocator/stream_allocator.cc） | allocator 本次新建的 logical stream 到 RTS 流绑定 |

解析必须同时校验 GE INFO 日志中的函数名和源码文件；仅有相同消息正文不能区分主流和辅助流。
`AcquireStreams` 只记录新建流，已有缓存流不会重复打印，不能把函数参数或日志条数直接当作
完整申请数量。

新日志不存在于旧版本时，才允许使用下面的兼容性 fallback：

`[KernelTrace][SplitRtStreams(?:_<任意后缀>)?] Get rts stream <ptr> from logical stream <i>, rts_stream_id: <id>`

该行由 runtime/v2/kernel/common_kernel_impl/stream.cc:55 输出，宏在
runtime/v2/core/debug/kernel_tracing.h:22 展开为 GE INFO 日志。该行只保存为执行确认或旧日志
fallback，不建立 V2 workflow 身份，不覆盖 `Collect rt2 stream` 的绑定结果。相同
`(logical_stream_index, rt_stream_id)` 的重复日志去重；同一 logical index 对应多个 RT ID
时保留冲突。

资源摘要中的 `stream_num`、`reusable stream num` 和 `attached stream num` 可能使用空格、冒号、等号或
`is` 连接数字（例如 `reusable stream num 4`、`attached_stream_num:0`）；解析器统一接受这些实际格式，
但仍只在完整的 V2 资源上下文中记录，不能把静态编译日志的同名字段当成 V2 运行期资源。
资源摘要只是 executor 内部的资源证据，不是模型边界。完全相同的连续/分卷摘要按 scope、name
和各资源数量去重并保留重复位置；静态子模型、root graph 和 root model 等不同 scope 可以共存。
同 scope/name 数量冲突时保留双方并报告冲突，不据此创建新模型。

HybridModel will execute in rt2.0 mode、Load model of graph: ... 和 Start load stage ... 只是
加载/执行上下文，不是 stream 申请。ExecuteSync 中直接创建默认 stream
（runtime/v2/core/model_v2_executor.cc:252-256）也没有独立的 stream 绑定日志。
`AcquireStreams(stream_num=...)` 或普通函数名行不是申请成功证据；只有上述完整
`Collect rt2 stream` GE INFO 日志才能建立 logical stream 到 RT stream 的绑定。

## 5. 结论规则和排除项

### 5.1 统一图级工作流

解析器对输入文件只做一次有序 GE INFO 日志扫描。每个 known/unknown 图入口创建一个
`GraphWorkflow`，随后将该图的编译入口/出口、逻辑流及算子事实、运行资源和运行绑定写入同一
工作流；不会先生成一份全局编译结果、再用“首条运行证据”回填所有图。V2 有 executor graph
name 时先精确匹配编译图（编译/运行 PID 可不同）；没有图名的 V1/V2 运行记录先作为运行片段
保存，完成扫描后再按后端规则寻找唯一编译图：

- PID 和 GE context 相同；
- 编译图完成时间早于运行首条资源/绑定证据，且在同一时间窗口；
- 编译逻辑流数（或静态最终模型流数）与运行模型/trace index 数一致；
- 中间没有另一个模型级根图完成或 `InitRuntimeParams`。

对于 V2 运行片段，每条 `Build RT2 executor` 是显式模型边界；资源摘要不能切分 executor 窗口。
对于没有 model_id/图名的 V2 片段，采用图级连续流规则：运行 PID 是硬边界，GE context 只作辅助
证据；同 PID 下跨 context/轮转文件的绑定只要在相同 logical index 上 RT ID 一致即可取并集，
因此子集、超集和不相交索引集都可归并。动态图只要有自身 `AssignStreamsForDynamicShapeGraph` 成功汇总即可成为候选，
不再按 `outer_graph` 或图名中的 `*_sub_*` 排除。运行 `logical_stream_index` 必须属于编译最终
逻辑流集合；完整集合相等时同时记录 `stream_count_match` 和 `logical_index_set_match`。多个未绑定
候选仍保持歧义；同一 logical index 指向不同 RT ID 时必须拆分运行批次，不能自动合并。

若编译侧只出现严格格式的 `*_sub_<N>_unknow(n)` unknown 图入口，没有动态流汇总和最终逻辑流 ID，
运行侧又通过 executor graph 名称匹配到该 root 且只观察到 logical 0，则隐藏编译 unknown 子图，
保留运行期 root graph 和 `logical 0 -> rts_stream_id`。映射表不填写 `compile_logic_stream_id=unknown`，
而以 `runtime_only` 行展示运行证据；出现 logical 1 或更多绑定时不触发该单流降级。

唯一候选标记 `same_session`（没有 model_id 时附 `inferred`）；V2 executor graph name 与编译图名
精确匹配或明确 model_id 直接匹配时标记 `strong`。多个候选、关键条件缺失或边界冲突保持
`unknown`，候选 ID 仍保留在 JSON 中。
V2 的 `SplitRtStreams` 行本身没有图名或 model_id，因此其 `logical_stream_index` 只在选定
workflow 内按 index 与编译逻辑流配对；一个 workflow 可以吸收多个兼容运行片段，吸收后的片段
不再渲染为 `unknown` runtime-only 行。不能用另一图的第一条绑定行补齐。默认 Markdown 只展示
流证据和映射，不列出输入文件路径清单。

- 图入口只定义分析窗口；逻辑流完成以静态 At last 或动态复数分配器 Graph 汇总为准。
- `stream_policy` 由该图可确认的最终逻辑流数量派生：数量为 1 才是 `single_stream`，大于 1 才是 `multi_stream`；只有聚合数量或中间分配记录时，必须保留作用域并避免伪装成图级最终逻辑流数。
- `Op [name] OpType [type] logic stream id is id` 以及源码核对的动态节点刷新 INFO 组成 `logic_stream_id -> operator name/type` 映射；算子表只列实际观察到的映射，不能从流 ID 反推算子或类型。
- 缺少出口、创建或绑定日志时输出 unknown/partial，不能填 0，也不能默认单流。
- At last、最终算子映射、明确拆流结果优先于 Pass 中间日志；同一数字必须带 ID 层和作用域。
- After SplitStreamAndRefreshTaskDef 表示 GE 侧物理流/TaskDef 刷新阶段完成；具体新增流仍需看拆流
  细节。Move split stream from ge to rts 表示职责委托 RTS；RefreshRealStream 或动态 Refresh
  stream 单独不能证明物理拆流。
- V1 `InitRuntimeParams` 通常是某个 runtime_model/submodel，不能用外层聚合数填充动态图的
  编译流数，也不能据此建立关联。
- aclmdlRIExecuteAsync、ModelExecuteTask、ConstructSqeForModelExecuteTask 等执行记录不能
  替代 stream 申请日志；外部 [RUNTIME] 行完全忽略。
- 动态 Batch 只改变模型场景归属；Batch_N known 分支仍按静态逻辑流和 V1 DavinciModel
  规则分析。
