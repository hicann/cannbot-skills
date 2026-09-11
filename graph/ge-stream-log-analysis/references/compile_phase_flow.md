# 编译期流分配源码边界

> 本文的源码路径和行号是版本相关的核对索引，不是永久接口契约。不同 GE/CANN
> commit 必须结合实际源码和 `[INFO]` 日志重新确认；没有对应日志时不能仅凭函数名补充事实。

## known-shape

源码调用链：

```text
GraphBuilder::BuildForKnownShapeGraph
  → Begin to build known shape graph
  → ModelBuilder::BuildModelForGetTask
  → StreamAllocator::AssignLogicalStreams
  → LogicalStreamAllocator::Assign
  → At last, root graph
  → InsertSyncNodesByLogicStream
  → After InsertSyncNodesByLogicStream
  → RefreshRealStream
  → SplitStreamAndRefreshTaskDef
  → After SplitStreamAndRefreshTaskDef / Move split stream from ge to rts
```

源码位置：

- `compiler/graph/build/graph_builder.cc`
- `compiler/graph/build/model_builder.cc`
- `compiler/graph/build/stream/stream_allocator.cc`
- `compiler/graph/build/stream/logical_stream_allocator.cc`

`Begin to build known shape graph` 是图构建范围入口；`BuildModelForGetTask` 是 Task 构建阶段入口；`AssignLogicalStreams` 才是逻辑流分配入口。`GE_CHK_STATUS_RET(builder.BuildModelForGetTask(...))` 仅在失败时产生日志，不是可靠的开始日志。

## unknown-shape / dynamic-shape

```text
GraphBuilder::BuildForUnknownShapeGraph
  → Begin to build unknown shape graph
  → ModelBuilder::AssignStreamForDynamicShapeGraph
  → DynamicStreamAllocator::AssignStreamsForDynamicShapeGraph
  → BuildModelForGetDynShapeTask
```

动态路径不要求出现 `AssignLogicalStreams`。如果动态模型包含 known-shape 子图，`BuildForUnknownShapeAllGraphs` 可能分别调用 known-shape 和 unknown-shape 构建路径，必须按图名/子图名分开报告。

## 结论规则

- `At last, root graph`：逻辑流分配完成和最终数量。
- `SetLogicStreamIdAttr`/`logic stream id is`：算子到逻辑流映射。
- `After InsertSyncNodesByLogicStream`：跨流同步生成完成。
- `After SplitStreamAndRefreshTaskDef`：GE 物理流刷新/拆分阶段最终汇总。
- `Move split stream from ge to rts`：GE 将拆流职责委托给 RTS。
- `After RefreshRealStream: stream ...`：DEBUG 中间细节，不能单独证明拆流。
