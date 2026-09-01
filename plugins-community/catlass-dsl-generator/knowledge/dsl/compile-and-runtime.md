---
type: CATLASS DSL Programming Concept
title: Kernel、编译与运行时
description: 当前 Host 编译、两级缓存、JitCompiledFunction 重复启动、stream 与 DLPack 工作流。
tags: [catlass-dsl, kernel, compile, runtime, cache, dlpack]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: host-api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/api/host_api_reference.md
    title: Generated TLA DSL Host API reference
  - id: compile-guide
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/zh/kernel_development/core_concepts/compile_and_launch.md
    title: Compile and launch guide
  - id: executor
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/base_dsl/jit_executor.py
    title: JIT compiled function and executor implementation
  - id: execution
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/execution.py
    title: Compile cache and launch ABI implementation
  - id: dlpack-runtime
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/tla/runtime.py
    title: Host tensor and DLPack implementation
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

`@tla.kernel` 返回 `TlaJitFunction`；直接调用 kernel 会失败。公共 Host 路径是
`tla.compile(kernel, *sample_args, options="--npu-arch 3510")`，返回
`JitCompiledFunction`；调用返回对象才会 launch。编译样本可以是 `from_dlpack`
得到的真实绑定，也可以是 `make_fake_tensor` 元数据样本。[^host-api]

首次调用同一个 `JitCompiledFunction` 时延迟创建并加载 `JitExecutor`，后续调用复用
executor。再次显式执行 `tla.compile(...)` 即使命中编译缓存，也会返回新的
`JitCompiledFunction`，不会继承旧对象的 executor 或 stream 状态。[^compile-guide][^executor]

# 用法

```python
import catlass.tla as tla

ta = tla.from_dlpack(a, layout_tag=tla.arch.RowMajor)
tb = tla.from_dlpack(b, layout_tag=tla.arch.RowMajor)
tc = tla.from_dlpack(c, layout_tag=tla.arch.RowMajor)

compiled = tla.compile(kernel, ta, tb, tc, options="--npu-arch 3510")
compiled(ta, tb, tc, block_num=1)
compiled(ta2, tb2, tc2, block_num=1)
```

launch 参数也可通过 `compiled(args=(ta, tb, tc), block_num=1)` 传递，但不能同时
提供位置参数和非空 `args=`。`block_num` 默认 1，必须是整数；`stream=` 是 launch
参数，省略时每次启动查询 executor 设备上的当前 stream。[^host-api][^compile-guide]

# 代码模式

## Metadata-only 与 IR 检查

```python
sample = tla.make_fake_tensor(
    tla.Float32,
    (128,),
    (1,),
    origin_shape=(128,),
    coord=(0,),
    layout_tag=tla.arch.RowMajor,
)
print(kernel.dump_mlir(type_args=[sample]))
compiled = tla.compile(kernel, sample, options="--npu-arch 3510")
```

`make_fake_tensor` 只能作为编译类型样本，launch 时 tensor 必须换成由 `from_dlpack`
绑定的真实 NPU buffer。`dump_mlir` 只证明前端生成了 TLAIR，不证明后端编译或设备正确性。
[^host-api]

## 两级缓存与重复启动

缓存 key 包含 kernel/静态 shape/`Constexpr`、目标架构、编译选项、工具链/ABI 和
kernel 使用的外部 Ascend C 源码；设备 pointer、输入数据、`block_num` 与 stream 不进入
key。查找顺序是进程内缓存、磁盘缓存、重新编译。常用开关为
`CATLASS_DSL_CACHE`、`CATLASS_DSL_CACHE_DIR`、`CATLASS_DSL_FORCE_RECOMPILE`。[^compile-guide][^execution]
可用 `compiled.cache_key` 和 `compiled.kernel_binary_path` 记录实际命中的产物。

## 自包含与显式特化

前端能解析 Python global/closure，但隐藏状态不在 kernel ABI 中。固定 tile 可在 kernel
内定义；运行时 shape 从 tensor metadata 读取；编译期参数使用 `tla.Constexpr[...]`；
运行时参数使用 scalar/tensor/dataclass 字段。不要由 Host 改写模块全局变量来选择 dtype、
layout 或 tile。

# 约束

- 公共目标选择是 `options="--npu-arch 3510"`；`arch_scope="aic.c310"` 不是当前
  `tla.compile` 文档化调用方式。
- launch 关键字是 `block_num`，不是旧接口的 `block_dim`。
- `*launch_args` 与 `args=` 互斥；launch tensor 必须已绑定 NPU buffer。
- 编译样本、kernel ABI、launch 参数的结构与 dtype 必须一致。
- DLPack producer、tensor 与其设备 buffer 必须存活到 launch 和设备同步结束。[^dlpack-runtime]
- 只有 kernel 内有 grid-stride 或等价覆盖逻辑时，`block_num` 小于逻辑任务数才不会漏算。
- 编译缓存命中只复用编译产物；复用已加载 executor 必须复用同一个 `compiled` 对象。
- build、TLAIR dump 或缓存命中均不证明 device correctness。

# 失败表现

- 直接调用 decorated kernel：提示必须先 compile。
- `block_dim` 或 `arch_scope` 报未知参数：沿用了旧 Host API。
- `block_num must be an int`：launch block 数类型错误。
- 同时传位置参数和 `args=`：launch ABI 拒绝歧义参数。
- fake tensor launch 报 buffer 未绑定：metadata-only 样本被误用于执行。
- 相同源码反复后端编译：静态 shape/`Constexpr`/options/外部源码改变，或缓存关闭、损坏。
- 新的 `tla.compile` 对象首次启动仍加载 binary：把编译缓存误解为 executor 复用。
- 只写前 `block_num` 个任务：kernel 没有 grid-stride 覆盖剩余任务。

# 验证方法

先保存 `dump_mlir`，再运行前端 pytest、lowering lit、metadata-only compile，最后做设备
launch 与 oracle。缓存验证要区分“同一 `compiled` 重复启动”和“再次 compile 后启动”，
并记录 cache key、binary path、是否创建新 executor。launch correctness 至少覆盖逻辑任务
数小于、等于和大于 `block_num` 的情况；stream 测试需分别覆盖隐式当前 stream 与显式
`stream=`。本文未执行 NPU 命令。

[^host-api]: 固定提交生成的 Host API 对 kernel、compile、launch、fake tensor 与参数约束的定义。
[^compile-guide]: 固定提交编译启动指南对两级缓存、JitCompiledFunction 生命周期、重复启动与 stream 的说明。
[^executor]: 固定提交 `jit_executor.py` 的延迟 executor 创建与重复启动实现。
[^execution]: 固定提交 `execution.py` 的 cache key、外部源码依赖、artifact 与 ABI 实现。
[^dlpack-runtime]: 固定提交 Host tensor 的真实绑定与 metadata-only 状态实现。
