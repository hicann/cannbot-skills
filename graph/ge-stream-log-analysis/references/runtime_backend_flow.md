# 运行期 V1/V2 流资源边界

> 本文的源码路径和行号按迁移时可见版本整理，仅用于人工上下文核对。不同 GE/CANN
> commit 或日志格式可能改变位置和关键字；最终结论必须优先依据实际 GE INFO 日志证据。

## V1 DavinciModel

```text
InitRuntimeParams
  → InitRuntimeResource
  → GetOrCreateRtStream / CreateNewStream
  → Logical stream index
  → BindModelStream
```

主要源码：`runtime/v1/graph/load/model_manager/davinci_model.cc`、`reusable_stream_allocator.cc`。

`Create new stream` 是物理流创建证据，`Logical stream index` 是逻辑/模型流到 RTS stream 的绑定证据。外部 `ModelExecuteTask`/`ConstructSqeForModelExecuteTask` 不属于 GE 仓日志分析范围，解析器忽略。
GE 创建但没有对应 `Logical stream index` 的流必须单列为未绑定/辅助流；它们不计入模型流数。V1 日志中出现的 `attached stream num` 不能填充 V2 的 attached 字段，V2 字段只接受实际出现且经过源码核对的 GE INFO 日志资源摘要；不能用 `AcquireStreams` 源码调用名或普通 INFO 文本代替日志。

## V1 Hybrid

真正的动态 Shape 或混合模型可能使用 `runtime/v1/hybrid/` 的 `HybridDavinciModel`/
`HybridModelRtV1Executor`。源码中的 `Start to init hybrid model`、
`HybridModel will execute in rt1.0 ...` 和 `HybridModel will execute in pipeline mode` 是
不同分支的可选后端/阶段标记，不要求在同一次运行中同时出现，也不是流数量证据；只有日志
实际出现时才能使用。RT2 的 `Succeed init hybrid model ... runtime v2` 和
`HybridModel will execute in rt2.0 mode` 属于另一条执行分支，同样不能与 V1 标记拼成必然链。
不能仅凭普通 DavinciModel 日志假定完整 Hybrid 流映射。动态 Batch 的模型归属仍为 dynamic，
不能用这些可选标记覆盖它。
`DavinciModel` 运行日志通常只对应其中一个 known-shape 子模型；没有 `AssignStreamsForDynamicShapeGraph` 的明确
`Graph: ..., stream num: ..., event num: ...` 时，两侧分别保留，不能用聚合数补齐动态图。

## 动态 Batch

动态 Batch 在模型场景归类上属于 dynamic，但已知 Shape 的 `Batch_N` 分支通常由普通
`DavinciModel` 加载和执行。源码中 `ModelManager::SetDynamicSize`
（`runtime/v1/graph/load/model_manager/model_manager.cc:708-712`）直接取得普通模型并调用
`DavinciModel::SetDynamicSize`，而 `IsDynamicShape`（同文件 `2452-2454`）检查的是
Hybrid 模型表。`aclmdlSetDynamicBatchSize`/`current batch label` 只证明档位设置和选择；
模型流与 RT 流仍使用 V1 `InitRuntimeParams`、`Create new stream` 和
`Logical stream index` 取证。不要因为模型属于动态大类，就把该路径标为 V1 Hybrid 动态。
编译侧的 `Found dynamic batch, shape ...`（`compiler/graph/preprocess/multi_batch_options.cc:527`）
和 `Add batch graph[Batch_N] ...`（`compiler/graph/passes/multi_batch/multi_batch_clone_pass.cc:1515`）
是动态 Batch 场景/分支生成证据；它们不等于 unknown-shape 流分配日志。

## V2/RT2

下面是源码调用链，不是要求日志必须逐行出现的日志序列：

```text
ModelV2Executor::OccupyStreamResource
  → GetReusableStreamNum + GetAttachedStreamNum
  → AcquireStreams(stream_num)
  → StreamExecutor / ExecuteGraph
```

主要源码：`runtime/v2/core/model_v2_executor.cc`、`runtime/v2/lowering/model_converter.cc`、`runtime/v2/core/stream_executor.cc`。

V2 可能没有 V1 的 `Create new stream`/`Logical stream index` INFO。`Build RT2 executor for root compute graph[...]` 建立 executor 身份；`OccupyStreamResource` 和 `AcquireStreams` 实际输出的完整 `Collect rt2 stream` GE INFO 日志分别提供 logical 0 主流和其余 allocator 流的绑定。只有函数/源码位置受控的完整日志正文才算证据，普通函数名或参数行不算。没有 RT stream ID 时输出 `partial` 或 `unknown`，不得套用 V1 映射。`LoweringAndSplitRtStreams` 是创建执行图节点的源码函数，当前没有固定成功 GE INFO 日志，不作为日志锚点。

同一 executor 常把资源摘要、Collect 和 SplitRtStreams 分散到多个线程或日志分卷。解析器只用
`Build RT2 executor` 切分模型窗口；资源摘要重复不切块。同一运行 PID 和 executor 窗口内，
logical index→RT ID 的交集无冲突即可合并并取并集；重复资源摘要语义去重，不同资源 scope 共存。
没有 executor graph name 时 PID 仍是运行片段硬边界，无法唯一关联编译图则保持 `unknown`。

动态 Shape 静态子图可能只打印多个 `Begin to build unknown shape graph[<root>_sub_<N>_unknow(n)]`，
而不打印动态流汇总。若 RT2 executor 使用 `<root>`，且运行期只观察到
`OccupyStreamResource` 的 logical 0 绑定，则将编译子图标记为 suppressed，保留运行期 root graph
和 logical 0 到 RT stream 的证据。此场景不生成 unknown 编译逻辑流占位；若运行期出现多个 logical
index，仍保持普通 partial/unknown 处理。

如果开启了 Kernel Trace，`[KernelTrace][SplitRtStreams(?:_<任意后缀>)?] Get rts stream ... from
logical stream <i>, rts[_ ]stream_id: <id>` 是 V2 运行期的直接逻辑流到 RT stream 绑定，输出到
`runtime_logical_to_rt_bindings`。同一运行会话还可记录 `Get rts notify`、`Sent event` 和
`Waited event`，但 Notify/Event 不计入 RT stream 绑定数量。Kernel Trace 没有 model_id 时，
优先使用 PID、GE context 和时间窗口关联；无法唯一确定会话时保持 model_id/关联状态为
`unknown`，不把 trace 归到前一个 V1 模型。
