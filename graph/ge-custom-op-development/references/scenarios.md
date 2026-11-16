# 能力选型与执行场景

按编译时机、图形态、持久化需求选接口；kernel 语言只影响产物与加载实现。
API 签名见 [interfaces.md](interfaces.md)，Python 差异见 [python-track.md](python-track.md)。

## 目录

- [编译时机](#编译时机)
- [离线状态携带](#离线状态携带)
- [执行与地址刷新](#执行与地址刷新)
- [输出 shape](#输出-shape)
- [辅流与资源统计](#辅流与资源统计)
- [静态模型重放](#静态模型重放)
- [加载属性兼容性](#加载属性兼容性)

## 编译时机

| 动作发生时机 | 能力 | 注意 |
|---|---|---|
| 图编译前已准备产物 | 无需 CompilableOp | 执行或声明时读取对应产物 |
| 图编译期按节点信息编译/准备 | CompilableOp / compile | 缓存自有数据，key 包含所有特化条件 |
| Execute 内 JIT | EagerExecuteOp / execute | 不因 JIT 需要 CompilableOp；检查重放和部署环境 |
| 宿主预热，回调消费缓存 | 命令式执行 | 如 PyPTO 样例，避免在不支持的回调环境启动编译子进程 |

有限 shape 集合可以提前编好多份按 key 选；“依赖 shape”不自动等于必须图编译期编译。
产物后缀是 `.so` 或 `.npubin` 也不决定编译时机。
离线出 OM 与是否实现 Compile 是独立决策；图编译期也可以编译后立即执行、不保存 OM。

官方对照：`tilelang_add_custom`、`tilelang_add_custom_online`、`tilelang_add_custom_offline`；`compilable_add_custom`、`python_compilable_add_custom` 展示其他编译实现。

## 离线状态携带

先列出执行需要的所有数据，再确认每项在目标机的来源。不要先要求用户选择接口名。

| 内容 | 携带方式 | 必须验证的条件 |
|---|---|---|
| 声明式 launch 的 kernel bytes | GE 根据 launch info 收集到模型 | 名称、长度、加载 metadata 完整；目标版本支持 |
| 自定义常量、tiling 表、shape→bin 索引或命令式独立产物 | PortableOp 的自定义 payload | 保存/恢复闭环，格式版本、长度与内容校验 |
| 与交付件同库的 device 代码 | 交付件 `.so` 随模型收集 | 目标 ABI、依赖库、注册和收集范围 |
| 外部 `.so` 或独立文件 | 显式打包/部署或 Portable | 不能假定间接依赖和临时目录自动进 OM |
| Python 源码及 descriptor | 支持该机制的版本收集 PYTHON_CUSTOM_OPS | 可定位源码、目标 Python/bridge/第三方包 |

`PortableOp` 是任意自定义字节的通道，GE 不解析其用户格式。
对尚无完整携带方案的命令式独立状态，默认建议 PortableOp；声明式已携带的 bin 不必重复序列化，但额外状态可能仍需要它。
同库方案也是可评估选项，无需用户预先点名；说明构建耦合、依赖与验证代价即可。
这里讨论的是离线状态恢复；在线命令式预编译 binary 的默认分发方式见 [产物分发](language-launch.md#缓存和分发)，不因为不需要 `PortableOp` 就推导为字节内嵌 `.so`。

`GeRootModel::CollectCustomOpSo` 在交叉编译分支跳过对应 `.so` 路径收集，不能把“本机同库出 OM 可用”推广为任意交叉部署自动自包含。
运行期 JIT 的结果不会追溯写入已生成 OM；若部署允许 JIT，需显式提供工具链与源码，不能从“离线模型”自行推定部署机一定没有工具链。

## 执行与地址刷新

| 场景 | 默认方案 |
|---|---|
| 仅 unknown graph，或每次执行需要 Host 决策 | EagerExecuteOp / execute |
| 静态图，参数和加载契约已能表达 | AnnotatedArgsOp / declare_launch_args |
| 同一算子需覆盖静态图和 unknown graph | 同类同时提供命令式与声明式，分别验证 |
| 需 C++ 回调刷新固定布局 host args | EagerExecuteOp + ArgsUpdater |

声明式便于 GE 管理地址刷新、模型中的 bin 和跨流依赖；不要用跨样例耗时证明其必然更快。
pyACL 命令式也能使用 `ctx.get_stream()` 对齐 GE 主流。

`GetArgsRefreshStrategy` 先选择 ArgsUpdater，再选择 AnnotatedArgsOp。
同时提供会使声明式刷新不被选中；这与双模的 EagerExecuteOp + AnnotatedArgsOp 不同。

「动态 shape」与 unknown graph 不是同一件事：TorchAir 动态图、档位可枚举的动态输入，都可能仍走已知 shape 的静态图分支。按实际图有无 unknown rank/dim 判断：确为 unknown graph 时声明式任务不生成（见下），必须命令式；仅档位动态时静态声明式仍可用。
双模交付要把两条路径当作两个验收对象，分别给出图形态、执行时机与数值证据。

当前 `GenerateTask` 仅在静态图分支生成声明式任务，unknown graph 走基本 CustomKernelTask，需要 EagerExecuteOp 实现实际执行。双模是机制可行方案，不宣称已有官方单类双模样例或已上板验证。
端侧 OMC 的单 launch/主流限制见 [接口边界](interfaces.md#声明式参数与返回值)。

## 输出 shape

- 可静态推导：新工程默认显式提供 shape/dtype 推导，不依赖“REG_OP 声明过”就能自动推 shape。
- 数据依赖：官方 Where-like 示例编译期给 unknown 维，Execute 按上界分配，kernel 写 shape buffer，同步后 D2H 并更新输出 origin/storage shape。shape buffer 是该示例协议，不是所有 kernel 的统一 ABI。
- 值依赖：用 `IMPL_OP().InputsDataDependency({i})` 声明所需输入值，框架物化值后重推；不应照搬数据依赖例子的 kernel shape-buffer 协议。

分别参考 `data_dependent_shape_custom` 与 `value_dependent_shape_custom`。
`-1` 是 unknown 维，不是可直接分配的元素数上界。

## 辅流与资源统计

| 路线 | 返回值 | 依赖责任 |
|---|---|---|
| Eager RequestAttachedStream(key[, flags]) | 物理 stream，失败 nullptr | 算子建立主流→辅流→主流依赖 |
| Annotated RequestAttachedStream(key) | 逻辑 ID，失败哨兵按头文件检查 | AddLaunch 依赖由 GE 生成同步任务 |

命令式使用回调当前 GE stream，辅流每次在当前上下文申请，不跨模型缓存物理 stream。
如使用可重放 event，需验证 record/wait/reset 的顺序和生命周期；不能只做首轮正确的同步、回调返回立即销毁重放仍需使用的 event。
已有作者实测指出 V1 静态重放缺 reset 可从第二轮读到错误数据；这是指定路径的回归线索，不是未经验证的所有运行时版本通用结论。
测试至少交替输入多轮并检查全部输出。

`ResourceUsageReporter`（目标头文件标注 since 9.3.0）申报 Eager 辅流 key，服务流数统计，不提供内存/workspace/event 配额。声明式流已被 GE 计入，不重复回调该统计接口。
Python 在本次核对基线未提供此能力；目标版本升级后重新探测。

## 静态模型重放

V1 的 `CustomTaskInfo::DistributeEagerOp` 在模型加载/task 分发阶段调用 Execute，之后重放模型任务。不要在这里放必须每轮执行的 Host 判断，并假设它随 RunGraph 重跑。
RT2 动态执行路径需独立验证；不能把 V1 的一次分发扩大为所有场景只执行一次。

V1 与 RT2 的差异直接决定「离线动态 shape 能否实现」，按下表判定，不从 V1 源码推普遍结论：

| | RT1.0（V1） | RT2 |
|---|---|---|
| Execute 调用点 | `custom_task_info.cc` 的 `DistributeEagerOp()`，仅分发阶段一次 | `custom_op_kernel.cc` 的 `ExecuteCustomOpFunc` / `ExecuteCustomOpWithInferShapeFunc`，执行期每轮 |
| 地址刷新 | `UpdateHostArgs()` 对非声明式、非回调式直接返回 SUCCESS | 由每轮执行路径处理 |
| 声明式 block_dim | `kernel_def.block_dim()` 编译期固定 | 由当轮执行决定 |
| 判据 | plog 出现 `CustomTaskInfo Distribute` | plog 出现 `RuntimeV2ModelExecute`（`aclmdlExecuteV2` 按 `IsRuntimeV2Enable` 与 `GetExecutor` 分派） |
| 典型场景 | 静态 OM、多档位 OM | unknown 维 OM |

实现 `ShapeInferOp` 会改变 RT2 的 kernel 类型：`NeedCustomOpInferShape()` 在 registry 中找到 ShapeInferOp、或 op 带 InferenceRule 时返回 true，节点被 lower 成 `ExecuteCustomOpWithInferShape`。
执行侧调用序列与错误码见 [离线 OM 执行契约](build-verify.md#离线-om-执行契约)。

未注册地址刷新不意味着必然算错：GE 内存布局冲突处理可插入 Identity/MemcpyAsync，把外部地址的数据搬入固定内部 buffer。是否插入及搬运代价受具体图和优化影响，通过图 dump、日志和多轮换址测试确认，不给所有图承诺固定一次拷贝或固定加速比。

## 加载属性兼容性

读取真实产物 metadata：入口符号、magic/coreType、参数 ABI、FFTS 和 block_dim。
`AnnotatedKernelLaunchInfo` 无 magic 字段，但 `SetKernelBinAttrs` 会读取节点的 `TVM_ATTR_NAME_MAGIC`，仅在缺失/为空时写默认 AIVEC，因此不能据字段缺失断言 MIX 结构性不可行。

本 Skill 模板未验证 MIX/FFTS 声明式接入。非默认产物需检查目标版本如何传入节点属性、任务生成、Runtime 加载及同步 ABI；未经验证时优先给显式加载的命令式实现，标明验证缺口。
仅设置 magic 也不等于已证明 MIX 全链路可用。具体格式与检查命令见 [产物契约](language-launch.md)。
