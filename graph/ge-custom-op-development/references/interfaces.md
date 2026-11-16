# C++ 能力接口与上下文

签名以目标 CANN 头文件为准，源码基线与定位方法见 [sources.md](sources.md)。
本文件维护 C++ 契约，不复制完整头文件；选型理由见 [scenarios.md](scenarios.md)。

## 目录

- [接口索引](#接口索引)
- [注册与实例](#注册与实例)
- [上下文](#上下文)
- [声明式参数与返回值](#声明式参数与返回值)
- [回调式刷新](#回调式刷新)
- [资源统计](#资源统计)

## 接口索引

| 接口 | 回调 | 用途 |
|---|---|---|
| EagerExecuteOp | Execute(EagerOpExecutionContext*) | 自行加载和 launch |
| HostCpuExecuteOp | Execute(HostCpuOpExecutionContext*) | Host 计算/常量折叠 |
| ShapeInferOp | InferShape / InferDataType | 输出元信息 |
| CompilableOp | Compile(OpCompileContext*) | 图编译期准备产物 |
| PortableOp | Serialize / Deserialize | 自定义状态随 OM 往返 |
| ArgsUpdater | UpdateHostArgs(UpdateArgsContext*) | 回调刷新固定布局地址 |
| AnnotatedArgsOp | DeclareLaunchArgs(AnnotatedArgsContext&) | 静态声明式 launch；since 9.2.0 |
| ResourceUsageReporter | DeclareResourceUsage(ResourceUsageContext&) | 辅流计数；since 9.3.0 |

头文件：`inc/graph_metadef/external/graph/custom_op.h`。
Portable 的确切签名为 `Serialize(std::vector<uint8_t>&)` 与 `Deserialize(const std::vector<uint8_t>&)`，均返回 graphStatus。

C++ 原生实现通常按继承接口暴露能力；Python 适配器另经 capability 门控。
实现 `ShapeInferOp` 还会改变 RT2 的 kernel 类型：`NeedCustomOpInferShape()` 命中 registry 中的 ShapeInferOp 或 op 的 InferenceRule 时，节点 lower 成 `ExecuteCustomOpWithInferShape`，否则为 `ExecuteCustomOp`。
不要为普通 C++ 算子手工编造 capability 注册，也不要用对适配器的裸 dynamic_cast 判断 Python 实际能力。

## 注册与实例

| 宏 | op type 来源 |
|---|---|
| REG_AUTO_MAPPING_OP(cls) | 类名字符串 |
| REG_OP_BACKEND(cls, op_type, backend) | 显式 op_type |
| REG_OP_WITH_PRIORITY(cls, op_type, backend, priority, engine) | 显式 op_type 及优先级/引擎 |

原型 REG_OP 的名称、实现注册的 op type、构图侧 type 必须一致。
例如原型 AddCustom、实现类 AddImplementation，可以使用：

```cpp
REG_OP_BACKEND(AddImplementation, "AddCustom", ge::OpBackend::kDevice);
```

若改用自动映射，类名也必须为 AddCustom。不要把不一致的几个骨架同时编译成一个算子。
Host 用 kHostCPU，device 用 kDevice；默认 custom 路径保留默认优先级。
`kTop`/`kBottom` 是候选优先级，不是覆盖；相同注册键重复会失败。
当前 custom engine 不认领仅 kBottom 的 kCustom 注册。

目标基线注册宏含 GE 版本门控（>= 9.2.0），新头文件与旧运行库混用可能没有实际注册。
同一注册键的实例被复用，不能用单槽成员保存“当前节点”状态；使用完整 key 的自有产物缓存，考虑并发访问与多个模型。

## 上下文

| 上下文 | 常用能力 |
|---|---|
| EagerOpExecutionContext | GetInputTensor、MallocOutputTensor、GetStream、MallocWorkSpace、RequestAttachedStream |
| HostCpuOpExecutionContext | 读取 Host 输入、分配 Host 输出、引用输出 |
| OpCompileContext | 节点输入输出元信息、平台、编译选项 |
| InferShapeContext / InferDataTypeContext | 读取输入元信息并设置输出 |
| AnnotatedArgsContext | I/O 地址、GetStreamId、MallocWorkSpace、RequestAttachedStream、AddLaunch |
| UpdateArgsContext | 继承 Eager 上下文，增加 GetKernelArgs |

- GetShapeSize 是元素数，GetSize 是字节数；负维不能转成合法分配大小。
- ctx/Tensor 只在当前回调内有效；缓存拷贝后的 shape/dtype/bytes，不保留借用指针。
- 输出内存由 ctx 管理。输入地址可能为 const void*，仅在 launch API 要求时转换，不修改输入。
- Eager MallocWorkSpace 返回 void*；Annotated 对应方法返回 WorkspaceAddr，不能混用。
- GetContextValue 可读取 kStaticModel、kAddressRefreshEnabled、kGraphId、kNodeId 和核数信息。核对目标 `kernel_args.h` 与上下文签名；Python 基线未暴露同等方法。

## 声明式参数与返回值

头文件：`exe_graph/runtime/annotated_args_context.h`。
launch info 的字段是 kernel_name、kernel_bin、kernel_bin_size、block_dim、stream_id。
C++ 必须填写非空 bin 与长度；入口使用 device 产物真实符号。
magic 的传递及未验证产物见 [加载兼容性](scenarios.md#加载属性兼容性)。

```cpp
gert::AnnotatedKernelArgs args;
// 顺序与 kernel 参数 ABI 一致；检查每次 AppendArg 的返回状态。
if (args.AppendArg(gert::InputAddr{0U, x->GetAddr()}) != ge::GRAPH_SUCCESS) {
  return ge::GRAPH_FAILED;
}
return ctx.AddLaunch(info, std::move(args));
```

- 追加地址时给 index 和当前地址；InputAddr/OutputAddr 仅有 index 时地址为空。
- 支持 I/O/workspace 地址、uint64 标量及 CustomDataBlock；每个槽位的表达须匹配目标 ABI。
- CustomDataBlock 在追加时深拷贝；核对 size、reserved 和目标版本支持。
- **两参 AddLaunch 返回 graphStatus**，直接检查/透传状态。
- **三参 AddLaunch 返回 AnnotatedLaunchToken**（since 9.3.0），首个 launch 的 predecessors 可为空。后续依赖只引用本上下文已有 token，不重复、不前向引用，并按头文件检查失败哨兵。
- ArgsUpdater 优先于本接口，二者同时存在不会组合刷新能力。
- 仅静态图生成声明式任务；unknown graph 需要 EagerExecuteOp。
- 端侧 OMC 当前限单 launch、主 stream、无依赖及变长 custom data；普通 OM 的能力不能直接推广过去。

## 回调式刷新

从 `ctx->GetKernelArgs(gert::Placement::kPlacementHost, index)` 取回先前持有的 args，检查结果后，仅修改与当前 I/O 地址相关的字段。index 对应 args 分配顺序。
先固定并记录 args ABI，不在 UpdateHostArgs 中重编译 kernel 或修改静态 tiling。
完整对照：`examples/custom_op/args_refresh_add_custom/cpp/ge/custom_op.cpp`。

Python 基线未提供 ArgsUpdater；若必须走回调式刷新则选择 C++。
需要 unknown graph 和静态图兼容时，不要把这个约束误用为禁止 Eager + Annotated 双模。

## 资源统计

ResourceUsageReporter 申报 Eager attached-stream key 集合，用于 CompiledGraphSummary 流数统计。
只申报实际可能使用的 key，不传句柄、内存字节或配额。
回调按节点复用实例；不能留下“上一节点”的统计状态。
声明式节点已按流计划统计，不重复走本接口。空 key 会使校验失败。
