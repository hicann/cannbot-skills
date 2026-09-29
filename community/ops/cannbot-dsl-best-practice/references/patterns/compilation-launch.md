# 编译与启动：复用程序，绑定本次数据

使用 CANNBotDSL 的 `@host` 或显式 `cannbotdsl.compile` 接口，公共结构见 [整体实现](../overall-implementation.md)。编译描述明确静态配置与动态参数，程序调用传入本次数据。

## 统一参数规约

以下是本 skill 的生成约定；接口允许特化某个值，不代表每个调用值都应该成为编译参数。先判断该值是否改变生成代码或静态资源布局，再决定放置位置。

| 类别 | 例子 | 放置与传递方式 |
|---|---|---|
| 编译配置 | tile 大小、Channel depth、静态 dtype/布局变体、容量上限 | 使用具名工厂参数，决定编译变体；在工厂内构造 Kernel、`TensorSpec` 等 |
| 动态张量元数据 | 实际行数、受支持的行跨度 | 用 `Dim` 和 `TensorSpec` 声明范围，调用时绑定当前 Tensor；不为范围内的每个值生成变体 |
| 运行时标量 | `scale`、实际 `block_num`、需要显式传递的有效长度 | 在 `@host` 签名及 compile 参数描述中声明类型，调用程序时传当前值 |
| Host 配置 | 公开选项、校验策略 | 在 Host 处理；若选择不同编译变体，则将选择结果明确传给工厂 |

- 同一个参数只维护一套含义与来源。由 Tensor shape 已表达的维度，除非设备接口需要，不再额外传一个可能不一致的标量。
- `max_rows` 等支持上界是编译配置；本次 `rows` 是动态值。静态容量可以是有依据的具名常量，不能当作本次有效长度或启动核数。
- 运行时标量用 `dtypes.float32`、`dtypes.int64` 等描述参数类型，不能用本次标量值代替类型描述。`Constexpr[T]` 只用于有意特化的参数；Kernel 构造配置也不应捕获本次 Tensor、stream 或启动核数。
- 某个维度确实影响静态布局或代码生成时，可明确设为编译配置，并同步调整支持范围、编译变体；不能仅因它是整数就设为编译参数。

## 显式编译与程序复用

```text
def build_program(column_count: int, max_rows: int, dtype):
    按静态配置构造 host_entry 和 TensorSpec
    return cannbotdsl.compile(host_entry, 对应参数描述)

with build_program(column_count, MAX_ROWS, dtypes.float32) as program:
    对每组满足同一编译契约的输入:
        program(本次输入, 本次输出, 运行时标量)
```

`build_program` 是普通构建函数，不隐含缓存行为。编译返回的程序可重复调用；持有方在最后一次使用完成后释放。编译参数应显式表达所有影响生成代码与静态资源布局的配置，不捕获首次输入的 Tensor、workspace 或 stream。

直接调用 `@host` 时由 DSL 管理编译和执行。显式编译时，底层编译产物缓存与 `ProviderCallable` 句柄复用分别处理；仅缓存输出对象不能代替程序复用。关闭句柄、清理编译缓存和释放 Tensor 各有不同生命周期，不要求算子增加统一清理接口。

**检查：** 重复使用程序时，输入地址和值可以变化，shape、stride 与 dtype 仍满足声明契约；静态配置变化时取得相应程序；最后一次使用完成前保持资源有效。

## 启动核数：Host 确定，Kernel 读取

“运行时启动参数”指每次执行时由 Host 确定实际启动数量，再传给已编译程序的 `@host` 入口。公开 Python 入口在内部完成查询，业务调用方无需指定核数。`core_num` 或 `block_num` 都可以作为变量名，保持含义与传递关系清楚即可。

下面演示按 AIC 逻辑 block 启动的路径。输入已校验为有效 NPU Tensor；`MAX_BLOCK_COUNT` 是当前实现的支持上限，来自调度记录容量、资源布局等约束，使用正整数具名常量定义。

```python
import torch
from cannbotdsl import get_platform_info


def _device_block_num(ref):
    device_index = ref.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    stream = torch.npu.current_stream(device_index)
    info = get_platform_info(stream=stream)
    available_cores = int(info.cube_core_num)
    if not info.available or available_cores <= 0:
        raise RuntimeError("no valid cube core capacity for the input stream")
    return min(available_cores, MAX_BLOCK_COUNT)
```

查询和提交使用同一设备、同一 stream。纯 AIV 路径按其 launch 语义参考 `vector_core_num`；混合核路径确认逻辑 block 与核型的映射。这些有效核数字段已考虑 stream 配额。

```text
# 算子文件顶部
from cannbotdsl.ops.arch import get_block_idx, get_block_num

# 公开 Python 入口内部；实际核数不加入构建函数参数
core_num = _device_block_num(x)
program = build_program(静态配置)
program(x, output, scale, core_num)

# @host 入口：run(x, output, scale, core_num)
op[core_num](x, output, scale)

# Kernel 内按需读取 launch 信息，不再添加一个重复的核数实参
block_idx = get_block_idx()
block_num = get_block_num()
使用 block_idx 与 block_num 划分本次任务
```

compile 参数描述中为 `core_num` 声明 `dtypes.int64`，实际值在 `program(...)` 调用时传入；编译缓存可复用于契约允许的不同核数。`MAX_BLOCK_COUNT` 是支持上限，不能直接代替本次查询结果。

按核分配的 [Workspace](workspace.md) 使用同一个实际核数；存在 [调度记录](aicpu-load-balance-metadata.md) 时，其分核数量与 launch 一致。使用全 launch 屏障还须满足 [同步](synchronization.md) 的驻留与全员参与条件；任务少可以保留空任务核，不能只在启动处随意缩减核数。

**反例：** 将开发设备核数写死在 launch 中；查询失败后用容量上限继续提交 NPU 计算；Host 与 Kernel 分别使用不一致的核数；只改变 launch 数量而复用旧的分核记录。

## 动态维度同时描述 shape 与 stride

下例为二维 ND Tensor，行数与行跨度动态，列连续。共享同一个 `rows` 约束输入输出行数一致；输入行间允许 padding，输出为连续布局。

```python
from cannbotdsl import Dim, TensorSpec, dtypes


def row_specs(column_count: int, max_rows: int, max_row_stride: int):
    rows = Dim("rows", min=1, max=max_rows)
    row_stride = Dim("row_stride", min=column_count, max=max_row_stride)
    input_spec = TensorSpec((rows, column_count), dtypes.float32,
                            stride=(row_stride, 1))
    output_spec = TensorSpec((rows, column_count), dtypes.float32)
    return input_spec, output_spec
```

只有 Kernel 真正支持该 stride 范围时才声明此契约。`TensorSpec(storage_format="nz")` 使用逻辑 shape 与紧凑逻辑 stride，要求受支持的 dtype、rank 2/3 和正维度；不能把物理 NZ 字节布局直接填成逻辑 stride。

列表输入按需使用 `TensorListSpec(element_spec, length)`：列表非空，元素模板仅支持 ND，各位置的动态维度独立绑定，同 dtype/rank 不代表不同元素 shape 必然相等。`Constexpr[T]` 是形参注解，调用时传原始 Python 值，不构造 `Constexpr(...)` 对象。

## 参数多时按连续区间分组

参数描述统一由构建函数维护。以下结构伪码假定入口已有两个相邻的临时张量参数；名字仅表示资源，不规定算法。逐个具名构造描述，再按签名中的连续区间分组：

```text
# 入口签名：run(x, scale, scratch_a, scratch_b, output, block_num)
input_spec = TensorSpec(输入 shape, 输入 dtype, 必要的 stride)
scratch_a_spec = TensorSpec(临时张量 A 的 shape, 对应 dtype)
scratch_b_spec = TensorSpec(临时张量 B 的 shape, 对应 dtype)
output_spec = TensorSpec(输出 shape, 输出 dtype, 必要的 stride)
workspace_specs = (scratch_a_spec, scratch_b_spec)

program = cannbotdsl.compile(
    entry.run,
    input_spec,
    dtypes.float32,  # scale：保留它在输入与 workspace 之间的位置
    *workspace_specs,
    output_spec,
    dtypes.int64,    # block_num
)
program(x, scale, scratch_a, scratch_b, output, block_num)
```

实际名字改成资源用途，如 `partial_sum_spec`；尺寸用语义参数和派生公式表达。可选参数按编译器支持方式明确描述，存在性影响编译结果时计入缓存键。

**反例：** `specs[4] = float` 按数字补写类型；用一张参数编号表替代清楚的签名；将所有 Tensor 描述移到标量前面；把难以辨认的长元组分散到运行与导出入口。优先使用具名描述和直接排列，避免为简单接口增加只转发的包装层。

## 可执行代码前要核实的三件事

- 编译器将哪些 shape、stride、标量作为静态事实，哪些允许启动时传入。
- 编译结果的参数顺序、类型和目标设备；进程内并发构建是否需要锁。
- 启动是否异步、使用哪个 stream，以及输入和临时资源由谁保持存活。

被编译的 Host 与 Kernel 保留可读取的 Python 源码。

**检查：** 相同静态配置但输入值和地址不同；改变一个静态项；改变受支持的动态尺寸；连续调用及要求支持的并发调用。可记录命中键和实际启动尺寸，不能把“缓存命中”当成结果正确的证据。
