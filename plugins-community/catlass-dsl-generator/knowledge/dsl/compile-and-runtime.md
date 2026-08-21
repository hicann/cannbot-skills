---
type: CATLASS DSL Programming Concept
title: Kernel、编译与运行时
description: Kernel 自包含约束、装饰、TLAIR dump、compile、launch、DLPack 与 build-only 工作流。
tags: [catlass-dsl, kernel, compile, runtime, dlpack, self-contained]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: readme
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/README.md
    title: TLA DSL build and test guide
  - id: execution
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/execution.py
    title: Kernel execution API
  - id: frontend
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/base_dsl/ast_preprocessor.py
    title: Python frontend global-name resolution
  - id: dlpack
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/tests/test_dlpack_bridge.py
    title: DLPack bridge tests
  - id: dlpack-runtime
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/tla/runtime.py
    title: DLPack tensor implementation
  - id: runtime-hardware
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/runtime.py
    title: Runtime hardware query API
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed.py
    title: Mixed Cube and Vector sub-block example
  - id: vadd
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_vadd/basic_vadd.py
    title: Pure Vector compile and launch example
  - id: mmad-grid-stride
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    title: MMAD grid-stride block loop example
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

函数用 `@tla.kernel` 定义；`dump_mlir(type_args=...)` 生成 TLAIR，
`tla.compile(kernel, *type_args, arch_scope=..., ...)` 构建产物，运行时负责
launch 和缓存。设备 tensor 可通过 DLPack 桥接为 TLA tensor。[^execution][^dlpack]

前端 lower 时会读取 Python 函数的 `__globals__` 来解析名字，compile 则先 lower 出
TLAIR，再把 TLAIR 纳入 cache key。因而模块级常量、可变全局对象和 closure capture
虽然没有出现在 kernel 签名中，仍会成为隐式编译输入。生成的 kernel 必须保持自包含，
不能把这种前端能力当作配置接口。[^frontend][^execution]

# 用法

## Metadata-only 编译

不需要设备 buffer 即可构造 type args 和检查 TLAIR：

```python
from catlass import runtime as runtime_mod

with runtime_mod._eager_capture():
    shape = tla.make_shape(128)
    type_args = (
        tla.Tensor(
            shape,
            tla.Float32,
            origin_shape=shape,
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
            layout_tag=tla.arch.RowMajor,
        ),
    )

print(kernel.dump_mlir(type_args=type_args))
executor = tla.compile(
    kernel,
    *type_args,
    arch_scope="aiv.c310",
    cache=True,
    cache_dir="./artifacts/runtime-cache",
    force_recompile=False,
)
print(executor.kernel_binary_path)
```

# 代码模式

## 自包含 kernel 与显式特化

kernel 内只能从形式参数、函数内局部值、Python 内建符号和受信任的 `tla` API
命名空间取值。固定 tile、dtype 和 params 对象在 kernel 内定义；实际 shape 优先从
tensor metadata 读取：

```python
@tla.kernel
def matmul_kernel(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor):
    l1_m = 256
    l1_n = 256
    l1_k = 128
    input_dtype = gm_a.dtype
    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]
    l1_a_ptr = tla.allocate(l1_m * l1_k, input_dtype, tla.AddressSpace.l1, 512)
    ...
```

同一 solution 必须且只能声明一个 `@tla.kernel`。多种编译期 dtype、shape 或 layout
复用该函数，由 tensor 形式参数的 type/metadata 产生不同编译产物；不得为变体声明
`fp16_kernel`、`bf16_kernel` 等独立 kernel。真正的运行时配置使用 scalar/tensor kernel
参数、tiling data 或 tensor 的 `origin_shape`。Host 侧 executor cache 可以独立存在，
但不得通过选择另一个 kernel、使用 `global` 或修改模块对象改变编译语义。

## DLPack 与 launch

```python
# dev_x/dev_y 是实现 __dlpack__ 的 Ascend/NPU tensor。
tla_x = tla.from_dlpack(dev_x, layout_tag=tla.arch.RowMajor)
tla_y = tla.from_dlpack(dev_y, layout_tag=tla.arch.RowMajor)

device = 0
tla.initialize(device=device)
try:
    executor = tla.compile(
        kernel,
        tla_x,
        tla_y,
        arch_scope="aiv.c310",
        cache=True,
        cache_dir="./artifacts/runtime-cache",
    )
    # task_count 是 kernel 实际切分出的独立任务数。
    task_count = ...
    block_dim = min(task_count, tla.get_vector_core_num(device))
    result = executor(tla_x, tla_y, block_dim=block_dim)
    # 使用当前框架的 device synchronize 后再读取输出。
finally:
    tla.finalize()
```

`from_dlpack(tensor, *, layout_tag, origin_shape=None, assumed_align=None,
stream=-1)` 是 zero-copy 绑定；只接受 Ascend/NPU buffer，CPU/NumPy 不受支持。
row-major dense buffer 应 contiguous；column-major 使用转置后 contiguous 的物理
shape，必要时显式传 `origin_shape`。[^dlpack][^dlpack-runtime]

## Artifact 与 launch 接口

`tla.compile` 返回可调用的 `TlaJitExecutor`。可读取
`kernel_binary_path`、`tlair_mlir`、`lowered_llvm`、`entrypoint`、`kernel_abi`
和 cache 信息；调用形式为 `executor(*args, block_dim=<int>)`。不能同时传位置参数
和 `args=...`，`block_dim` 必须是整数，省略时默认为 1。[^execution]

调用纯 Vector kernel（`arch_scope="aiv.c310"`）时，用
`tla.get_vector_core_num(device)` 查询 VectorCore 数；调用 Cube 或 mixed kernel
（`arch_scope="aic.c310"`）时，用 `tla.get_aicore_num(device)` 查询 AICore 数。
`device` 必须与 `tla.initialize(device=device)` 使用同一设备，可以是设备整数编号或
`"npu:<id>"` 字符串。两个查询不能互换：C310 上一个 AIC block 可包含多个 Vector
sub-block，AICore 数不是纯 AIV launch 的 VectorCore 数。
[^runtime-hardware][^vadd]

只有 kernel 能覆盖 `task_count > block_dim` 的剩余逻辑任务时，才能把 launch 数限制为
`min(task_count, core_count)`。常见覆盖模式是 kernel 内使用
`tla.range(tla.arch.block_idx(), task_count, tla.arch.block_num())` 做 grid-stride
循环；如果 kernel 直接以 `block_idx()` 映射唯一任务且没有循环，截断 block_dim 会漏算
任务。任务数不超过对应 core 数时，`min` 退化为一任务一 block，不改变覆盖关系。
[^runtime-hardware][^mmad-grid-stride]

mixed kernel 的 host `block_dim` 按 AIC block 解释；同一 block 的两个 Vector 执行实例在
`with tla.vector()` 内用 `tla.arch.sub_block_idx()` 区分。`sub_block_idx()` 只表示
当前 block 内的 AIV 子块，不是全局任务索引。若两个 sub-block 写不同 GM 分片，分片
公式、UB 所有权和同步必须成套设计；不能仅把纯 AIV kernel 改成 mixed scope 后让两个
sub-block 独立执行原任务。[^mixed]

开发顺序应为 TLAIR dump、前端 pytest、lowering lit、metadata-only compile，
最后才进入设备 launch。[^readme]

# 约束

- `@tla.kernel` 不得读取模块级 tile、shape、dtype、params 对象、开关或其他用户定义
  全局值，也不得捕获 closure 配置；`tla` API 命名空间和 Python 内建符号除外。
- Host wrapper 不得通过 `global`、`setattr(module, ...)` 或修改容器来改变 kernel 编译
  语义；所有编译期变体必须复用同一 decorated kernel，由形式参数 type/metadata 特化，
  运行时值使用显式 ABI 或 tensor metadata。
- solution 只能声明一个 `@tla.kernel`；不能把 shape、dtype、layout、tile 或 params 变体
  拆成独立 kernel，也不能由 host 做 kernel dispatch。
- `type_args` 必须完整描述 kernel ABI；动态 extent 需显式标量元数据。
- `arch_scope` 必须与 AIV/AIC kernel 类型一致。
- DLPack producer 的 device、dtype、shape 和生命周期必须在 launch 期间有效。
- `initialize` 不能重复调用；`finalize` 必须对应已初始化状态。
- `block_dim` 必须是正整数；AIV scope 使用 `get_vector_core_num`，AIC/mixed scope 使用
  `get_aicore_num`，并显式传入 launch 所用设备。
- 只有存在 grid-stride 或等价覆盖逻辑时才能把 `block_dim` 截断到 core 数；一任务一 block
  的直接映射必须保证 launch 覆盖全部任务。
- mixed Vector 分片必须使用 `sub_block_idx()` 区分 AIV0/AIV1，并验证多 block、奇数
  任务数和每个 GM 输出分片；当前 launcher 不接受用户直接传任意 grid。
- build-only 成功只证明编译链，不证明设备正确性。

# 失败表现

- 同一 kernel 的 TLAIR 随 host 调用顺序变化：kernel 读取了被 wrapper 改写的模块级值。
- 并发 shape/dtype 编译互相污染：多个调用竞争同一可变全局特化状态。
- 函数签名、源码局部值与实际 TLAIR 配置不一致：存在隐藏全局或 closure 编译输入。
- `does not implement __dlpack__()`：输入不是 DLPack producer。
- `CPU / NumPy buffers are not supported`：错误使用 host buffer。
- `` `block_dim` must be an int ``：launch block 数不是整数。
- 输出只完成前 `core_count` 个任务：没有 grid-stride，却把 `block_dim` 截断到 core 数。
- mixed 输出被重复写或部分未写：把 `sub_block_idx()` 当成全局 block 索引，或两个
  Vector sub-block 的 GM 分片重叠。
- `already called` / `requires a prior call`：initialize/finalize 生命周期错误。
- `Compiled artifact is missing runtime options`：artifact 不能直接 launch。
- Python binding、CANN、ABI 或 cache 问题应按 import、compile、launch 分层定位。

# 验证方法

分层保存 `dump_mlir`、pytest、lit、build-only 与 device run 的原始输出，不把
前一层成功升级解释为后一层成功。launch 验证至少覆盖任务数小于、等于和大于对应
core 数的情形；mixed kernel 还要分别覆盖两个 `sub_block_idx()` 和多 host block，检查
无重复写、无遗漏任务。源码审查还要枚举每个 decorated kernel 的自由名字，除 `tla`
和 Python 内建符号外必须为空；分别编译各 dtype/shape 变体，确认 wrapper 没有在编译前
改写模块状态。本文未运行上述环境相关命令。

[^readme]: 固定提交 README 的安装、pytest、lit、build-only 与运行命令。
[^execution]: 固定提交 `execution.py` 的 kernel 编译运行接口。
[^frontend]: 固定提交 `ast_preprocessor.py` 对函数 `__globals__`、closure 和前端转换命名空间的处理。
[^dlpack]: 固定提交 DLPack bridge 测试覆盖的输入约束。
[^dlpack-runtime]: 固定提交 DLPack zero-copy tensor 的签名与 layout 转换。
[^runtime-hardware]: 固定提交 `runtime.py` 的 `get_aicore_num(device)`、`get_vector_core_num(device)` 签名与设备参数规范化实现。
[^mixed]: 固定提交 `basic_mixed.py` 的 mixed Cube/Vector region、`sub_block_idx()` 分片和 host block launch 模式。
[^vadd]: 固定提交 `basic_vadd.py` 的纯 Vector `aiv.c310` 编译与 host launch 模式。
[^mmad-grid-stride]: 固定提交 `basic_matmul.py` 的 `block_idx()`、任务总数和 `block_num()` grid-stride 循环。
