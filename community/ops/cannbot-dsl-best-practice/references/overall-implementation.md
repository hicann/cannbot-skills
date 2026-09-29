# 整体实现：统一计算入口与必备导出入口

每个生成算子保留有效的 `@export` 入口，使用 `cannbotdsl.aot.export`。运行与导出共用编译参数构造逻辑。公开 Python 入口负责输入校验和资源准备，`@host` 直接启动 `@kernel`。显式编译使用 `cannbotdsl.compile`，程序的创建、复用与释放由实际调用边界决定。

## 公共结构

```text
公开 Python 入口 → 校验与资源准备 → @host → @kernel
显式编译路径：cannbotdsl.compile(@host 入口, specs...) → ProviderCallable → 绑定本次参数并执行
@export → 同一构建函数 → DSL Native 收集与发布
```

| 位置 | 职责 |
|---|---|
| `operator(...)` | 维护公开参数、错误行为、输出分配与返回结果 |
| `build_program(...)` | 按需集中构造编译参数并返回编译程序 |
| `Program.run(...)` / `@host` | 直接启动 Kernel |
| `Kernel` / `@kernel` | 完成设备实现，按需用 `@jit` 组织计算片段 |
| `export_operator()` / `@export` | 声明导出名称、枚举支持的编译变体 |

这些职责可以放在一个文件中。workspace、metadata、状态、计算角色与阶段按算法需要添加；Torch 接入见 [Torch 注册](patterns/torch-registration.md)。

## 公共结构伪码

以下展示显式编译路径。中文步骤由当前算子填写；输入输出为同 shape 的二维连续 FP32 Tensor，行数动态、列数静态。支持集合与尺寸上界来自公开契约。

```text
import torch
import cannbotdsl
from cannbotdsl import Dim, Tensor, TensorSpec, dtypes, host, kernel
from cannbotdsl.aot import export
from cannbotdsl.ops.arch import get_block_idx, get_block_num

@kernel
class OperatorKernel:
    def __init__(self, column_count):
        self.column_count = column_count

    def __call__(self, x: Tensor, output: Tensor, scale: float):
        block_idx = get_block_idx()
        block_num = get_block_num()
        按需使用 block_idx 与 block_num 划分任务并完成设备实现

class OperatorProgram:
    def __init__(self, column_count):
        self.column_count = column_count

    @host
    def run(self, x: Tensor, output: Tensor, scale: float, core_num: int):
        op = OperatorKernel(self.column_count)
        op[core_num](x, output, scale)

def build_program(column_count: int, max_rows: int):
    entry = OperatorProgram(column_count)
    rows = Dim("rows", min=1, max=max_rows)
    input_spec = TensorSpec((rows, column_count), dtypes.float32)
    output_spec = TensorSpec((rows, column_count), dtypes.float32)
    return cannbotdsl.compile(
        entry.run, input_spec, output_spec,
        dtypes.float32,  # scale
        dtypes.int64,    # core_num
    )

def operator(x, scale):
    校验输入布局、设备、scale 和公开支持范围
    row_count, column_count = x.shape
    core_num = _device_block_num(x)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    with build_program(column_count, MAX_ROWS) as program:
        program(x, output, float(scale), core_num)
    return output

@export("operator_name")
def export_operator():
    for column_count in SUPPORTED_COLUMN_COUNTS:
        with build_program(column_count, MAX_ROWS):
            pass
```

`_device_block_num` 见 [编译与启动](patterns/compilation-launch.md)。Host 将 `core_num` 用于 launch，Kernel 按需通过 `get_block_num()` 读取实际数量。

示例的单次调用拥有程序句柄，并在执行结束后释放。需要连续复用时，将程序的创建和释放放到整组调用之外，每次传入当前 Tensor 与标量。普通 `build_program` 函数本身不缓存句柄；DSL 编译产物缓存与程序句柄复用是两个层次，运行时句柄复用不能使导出回调跳过实际的编译请求。

## 装饰器与调用边界

- 普通 Python 可以直接调用 `@host`；其编译与调用由 DSL 管理。需要显式动态参数规格时，使用 `cannbotdsl.compile`，第一个参数为 `@host` 函数或绑定方法。
- `@host` 不返回计算结果，输出由传入的 Tensor 承接，公开 Python 入口组织返回值。
- Kernel launch 直接写在 `@host` 源码体中；不能通过普通 Python helper 或 `@jit` 间接发起 launch。
- `@jit` 组织 DSL 内部辅助计算，可由 `@host`、`@kernel`、`@jit` 调用并继承执行域。

## 编译契约与可读性

- **参数具名且同序：** 像上例一样列出 `input_spec`、`output_spec`，为标量类型标明形参名；specs、`run` 形参和调用实参逐项对应。
- **配置直接表达：** 构建函数显式接收影响编译的配置；不捕获本次 Tensor、workspace 或 stream。参数分类和连续参数分组见 [编译与启动](patterns/compilation-launch.md)。
- **静态与动态分开：** `column_count` 在本例中有意静态化，`max_rows` 是支持上界；实际行数由 Tensor 绑定，`scale` 与 `core_num` 在调用时传给 Host 入口。不要将本次核数写成固定默认值或塞入编译配置。统一分类与核数选择规则见 [编译与启动](patterns/compilation-launch.md)。
- **尺寸有来源：** 静态尺寸与动态上界来自支持范围，派生尺寸用公式表达。共享逻辑维度复用同一个 `Dim`；需要的 stride 和 storage format 如实声明。
- **避免位置补丁：** 不使用 `specs[4] = float`，也不为此另建参数编号表。参数少时直接排列，参数多时仅按签名中的连续职责区间分组。

## 程序与导出的生命周期

显式编译返回的 `ProviderCallable` 由持有方管理，在最后一次使用结束后通过上下文管理器或 `close()` 释放；关闭后不能继续调用。并发和异步路径按实际接口确认最后使用完成，不能提前释放程序或 Tensor。临时资源见 [Workspace](patterns/workspace.md)。

**检查：** 参数顺序一致；动态尺寸在声明范围内；重复调用绑定本次数据；程序与临时资源在最后使用前有效。没有显式持有程序的调用方无需增加独立清理入口。

`@export("name")` 装饰同步、无参数函数，导入时仅标记入口。`cannbotdsl.aot.build(collectors=..., target=..., output=...)` 发现并执行回调，收集其中实际发生的编译请求；直接调用回调只编译，不发布 Native 目录。输出路径应为新的目标目录。

导出回调使用同一构建函数重新发起编译请求，底层编译产物可命中缓存。DSL 收集器在编译期间保存产物副本；回调通过上下文管理器关闭自己取得的程序句柄，发布临时资源由构建接口管理。运行时已持有的程序继续由其调用方管理，不在导出回调中清理。

**导出检查：** 冷缓存和热缓存后构建均覆盖声明的变体；运行与导出参数契约一致；通过 DSL 的 Native 构建和注册接口验证产物实际可用，不能只检查装饰器存在。

## Kernel 实现参考路由

Kernel 按执行角色分为 CV 融合、单 C 和单 V。各路线共用公开入口、编译与启动结构，设备侧分别组织任务、资源和阶段依赖。

| 路线 | 执行角色 | 结构重点 |
|---|---|---|
| CV 融合 | C 与 V 协同 | 阶段交接、在途任务窗口、槽位复用与排空 |
| 单 C | C 侧执行 | 任务分配、资源准备、C 阶段调用与输出 |
| 单 V | V 侧执行 | 任务分配、资源准备、VF 阶段调用与输出 |

C 指 Cube/AIC，V 指 Vector/AIV；单 C、单 V 表示核型。所需角色由实现需求确定，核数查询见 [编译与启动](patterns/compilation-launch.md)。

以下为结构伪码，计算阶段统一使用占位步骤。`inputs/outputs` 表示实际签名中的具名输入与输出，任务记录表示索引与资源的逻辑关联；落地时按契约展开，不默认引入容器或固定记录协议。资源的形状、类型、布局与构造参数由实现定义。所用导入放在算子源码顶部。

### CV 融合：preload=2 的在途任务窗口

使用两个任务槽组织阶段交接。`preload=2` 表示最多两个已发起、尚未回收的任务，当前消费任务计入窗口。每个槽位持有该任务所需的上下文与交接资源，任务完成后才能复用。

```text
# 算子源码顶部
from cannbotdsl import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num

PRELOAD = 2

@kernel
class CubeVectorKernel:
    def __init__(self, preload=PRELOAD):
        self.preload = preload
        准备 C 与 V 各自需要的资源
        按 preload 准备任务上下文与交接槽位

    def __call__(self, inputs, outputs):
        block_idx = get_block_idx()
        block_num = get_block_num()
        tasks = 分配给当前逻辑 block 的任务
        task_count = 本 block 的任务数

        # 启动：填入存在的任务。
        for sequence in range(min(self.preload, task_count)):
            slot = sequence % self.preload
            将 tasks[sequence] 的上下文绑定到 slot
            提交该任务的生产阶段

        for sequence in range(task_count):
            slot = sequence % self.preload
            在消费前建立该任务的数据就绪依赖
            执行消费阶段并提交该任务的输出
            确认该槽位的最后访问完成，允许复用

            next_sequence = sequence + self.preload
            if next_sequence < task_count:
                将 tasks[next_sequence] 的上下文绑定到 slot
                提交下一任务的生产阶段
```

启动阶段填入至多两个任务；稳定阶段每回收一个槽位就补入一个任务；新任务发完后继续消费剩余任务，完成排空。任务数为 0 或 1 时只处理实际存在的任务。

生产与消费阶段按数据依赖分配给 C/V 角色。本例适用于可组织成一次生产与消费交接的任务；更长的阶段链应按资源的完整存活区间安排推进与回收。任务上下文和交接数据必须使用一致的槽位，地址仍被计算或搬运访问时不得覆盖。

槽位容量、数据就绪和允许覆盖是不同约束。使用 Channel 时，`produce()/consume()` 只选择槽位，不代表同步已经完成。同步依据实际访问建立；主调度条件位于 VF 外。资源与同步写法见 [Channel](patterns/channel-buffer.md)、[同步](patterns/synchronization.md)。

### 单 C：任务与计算阶段

```text
# 算子源码顶部
from cannbotdsl import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num

@kernel
class CubeKernel:
    def __init__(self):
        按静态配置准备 C 侧所需资源

    def __call__(self, inputs, outputs):
        block_idx = get_block_idx()
        block_num = get_block_num()
        task_count = 根据输入元数据确定任务数
        for task_id in range(block_idx, task_count, block_num):
            绑定当前任务的输入、输出与临时资源
            准备输入并建立必要的数据就绪依赖
            执行 C 侧计算阶段
            提交当前任务的输出
            在资源复用前确认最后访问完成
```

示例采用按 block 步进分配任务；有其他调度约束时采用对应的任务映射。计算阶段的内部循环与指令由实现填写，资源准备和同步仅保留实际需要的部分。

### 单 V：任务与 VF 阶段

```text
# 算子源码顶部
from cannbotdsl import kernel, vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num

@kernel
class VectorKernel:
    def __init__(self):
        按静态配置准备 V 侧所需资源

    def __call__(self, inputs, outputs):
        block_idx = get_block_idx()
        block_num = get_block_num()
        task_count = 根据输入元数据确定任务数
        for task_id in range(block_idx, task_count, block_num):
            在 VF 外绑定当前任务的输入、输出与临时资源
            准备输入并建立必要的数据就绪依赖
            with vf(mode="simd"):
                执行 V 侧计算阶段
            建立计算到输出的必要依赖并提交输出
            在资源复用前确认最后访问完成
```

VF 只承接设备计算，资源选择和任务调度位于 VF 外。条件执行与内存屏障规则见 [VF](patterns/vf.md)。

**检查：** 任务完整覆盖且不重复；空任务不访问无效资源；CV 覆盖 0、1、2、3 个任务与槽位回绕；任务上下文与数据一致；所有在途访问结束后才回收资源。设备执行记录按 [生产代码](production-code.md) 检查。
