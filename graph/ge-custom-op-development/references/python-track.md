# Python ge.custom_op 契约

生成或检视 Python 交付件时读取；签名按目标安装包核对。
通用能力选型见 [scenarios.md](scenarios.md)，打包部署见 [build-verify.md](build-verify.md)。

## 目录

- [原型与回调签名](#原型与回调签名)
- [能力与双模](#能力与双模)
- [生命周期和状态](#生命周期和状态)
- [Portable 契约](#portable-契约)
- [流与执行](#流与执行)
- [OM 部署依赖](#om-部署依赖)
- [插件扫描](#插件扫描)

## 原型与回调签名

`register_op(op_type=...)` 装饰 infer-meta 函数，`register_op_impl(op_type=...)` 装饰实现类。

- 位置参数表示 Input，用 TensorDesc 标注，不带默认值。
- keyword-only 参数表示 Attr，需要标注，可提供默认值。
- 返回标注表示 Output，按目标版本允许的 TensorDesc/容器形式声明。
- compile/declare_launch_args 接收 Input + Output Tensor；execute 接收 Input，输出由 ctx 分配。
- 回调 attrs 与原型的 keyword-only 形参一致；execute 返回 None。
- 属性声明必须与真实计算语义一致；模板未实现 alpha 计算时应拒绝非默认值，不静默忽略。

原型推导只声明受支持的 dtype/shape。不要把同shape当作不需要推导，或把 unknown维用于实际分配。
对固定块长的官方示例 kernel，未经修改不可承诺尾块和空张量支持。

## 能力与双模

注册通过方法反射收集能力，没有 `interfaces=` 参数。

| Python 定义 | 能力 |
|---|---|
| execute | EagerExecuteOp（默认 device） |
| register_kernel(backend=OpBackend.HOST) 装饰 execute | HostCpuExecuteOp |
| compile | CompilableOp |
| declare_launch_args | AnnotatedArgsOp |
| 类自身成对 serialize/deserialize | PortableOp |
| register_op 的 infer-meta | shape/dtype 推导能力 |

execute 与 declare_launch_args 可共存，分别覆盖动态和静态路径；不要写成强制二选一。
声明式通常减少 args/持久化/依赖处理，但不是性能保证；静态与动态都要用时分别验收。
本次基线 Python 不提供 ArgsUpdater、ResourceUsageReporter，升级后按 [来源规则](sources.md)复核。
C++ 适配器继承的基类不等于 Python 已声明的能力；CustomOpCast 还会检查 capability。

## 生命周期和状态

- 实现类可零参调用，框架通过 descriptor.cls() 创建实例；带默认值的可选构造参数也可能满足条件。
- ctx/Tensor 为当前回调的借用对象，不保存，不交给异步 Python 线程继续使用。
- `Tensor.storage_shape.dims` 取实际维度；Tensor.shape 是 StorageShape，不按普通 list 迭代。
- 实例可能被同 op type 的多个节点复用。实例字段并非禁止使用：完整 key 的、自有数据缓存是合法设计。
- key 覆盖参与特化的各输入输出 shape/dtype/format、attrs 和平台；保护并发访问。不能只用单个 `_current_bin` 或把实例状态改成同样共享的类变量。
- 磁盘缓存还要考虑源码及依赖、编译选项和工具链身份；无法可靠确定依赖时不缓存。

当前 Python 模板不跨 Compile 调用复用磁盘产物，每次在独立临时目录编译，仅保存供当前图声明使用的自有 bytes 和按元信息区分的产物；同实例跨目标平台明确拒绝。

## Portable 契约

| 条件 | 约束 |
|---|---|
| 方法归属 | serialize/deserialize 写在类自身，不能只依赖继承方法 |
| 成对 | 缺一个会在注册期 TypeError |
| 构造 | __init__ 零参可调用 |
| payload | 精确 bytes，长度 1..UINT32_MAX |
| 返回 | deserialize 返回 None |

格式由实现定义，建议包含格式版本、长度和完整性校验。
不要用空 bytes 作为有效默认状态；未准备好时明确失败。
声明式 bin 已由 GE 收集时，Portable 只需携带额外状态；详见 [离线状态携带](scenarios.md#离线状态携带)。

## 流与执行

命令式回调通过 get_execute_ctx() 取得 ctx，pyACL launch 可传 ctx.get_stream()。
自行处理 binary/function/args 资源与错误返回，不从全局默认流 launch。
第三方框架是否可注入 stream 需要核对其接口；不能把一个 PyPTO 示例的限制推广到所有框架。

声明式通过 get_declare_launch_args_ctx() 创建 args，依 ABI 追加 input/output/workspace/scalar，用 AnnotatedKernelLaunchInfo 提交。stream_id 使用 ctx.get_stream_id()，它是逻辑 ID，不是物理 stream。
AddLaunch 依赖、端侧限制与返回值按目标 Python 绑定及 C++ 头文件共同核对。

## OM 部署依赖

支持新机制的版本把定义源码和稳定 descriptor 快照保存到 PYTHON_CUSTOM_OPS 分区，加载时恢复。
Portable payload 与声明式 kernel bin 走各自通道；不能只看到 .om 已生成就断定状态完整。

部署机仍需 CPython/libpython、custom_op_runtime、匹配的 GE bridge 和源码 import 的第三方包。
从 REPL/动态 exec/不可定位源码创建的类可能无法收集源码，必须做独立进程部署验证。
不把 Python major/minor 等诊断字段视为已完成兼容性校验。
旧版本可能只在编译阶段用 Python、部署阶段另需 C++ 交付件；检查目标版本，不能混用两代行为。

完整构图链路是生成 ES 包/wheel → GraphBuilder 保存 AIR → ATC 生成 OM → 独立进程加载并对拍。
各阶段可能触发不同回调，按实际日志定位，不断言所有 infer-meta 调用必然仅发生在 ATC。

## 插件扫描

ASCEND_CUSTOM_OPP_PATH 可含 C++ OPP 根目录和 Python 模块目录：

- `.py` 文件可直接加载。
- 目录扫描直接子项：跳过 `_` 开头，加载 `.py` 与含 `__init__.py` 的子包，不递归任意目录。
- 不存在路径会 FileNotFoundError，非 `.py` 文件会 ValueError。

C++ 对失效路径的跳过/告警不代表 Python 也接受；先清理无效条目再诊断 bridge。
