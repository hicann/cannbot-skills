# 生产代码：清楚的结构与可检查的执行边界

用于算子源码的生成与检查。整体入口与装饰器结构见 [整体实现](overall-implementation.md)；这里规定源码组织、局部写法和执行约束。算子源码只包含公开计算所需的实现与接入代码。

## 源码布局与命名

文件按“模块说明 → 必要的 `__future__` 导入 → 标准库、第三方库、项目导入 → 常量 → 函数与类 → 注册/导出声明”组织。导入组之间留空行，所有 `import`、`from ... import ...` 在顶部完成；函数体、类体和文件中段不再导入，也不用动态导入绕过要求。

Python 命名规则直接按下表使用：

| 对象 | 规则 | 示例 |
|---|---|---|
| 变量、参数、函数、方法 | 小写单词以下划线分隔，即 `snake_case` | `valid_rows`、`workspace_bytes`、`validate_tiling` |
| 类 | 单词首字母大写，即 `CapWords` | `OperatorKernel`、`OperatorProgram` |
| 模块级常量 | 大写单词以下划线分隔 | `ALIGNMENT_BYTES`、`MAX_BLOCK_COUNT` |
| 内部函数或属性 | 在正常名字前加一个下划线 | `_get_compiled_program` |
| 与 Python 关键字冲突的名字 | 改用语义明确的名字，必要时在末尾加下划线 | `class_index`、`class_` |

函数名表达动作，变量名表达数据含义；数量、字节和元素偏移要区分，例如 `row_count`、`workspace_bytes`、`row_offset_elements`。布尔量可用 `is_`、`has_`、`needs_` 表达判断。短循环中的 `row`、`col` 足够，不堆叠无意义前缀。

避免单独使用易混淆的 `l`、`O`、`I`；避免用 `list`、`type`、`sum` 等名字遮蔽仍需使用的内置对象。实例方法首参使用 `self`，类方法首参使用 `cls`；双下划线特殊名称只用于 Python 已定义的协议。公开参数改名需考虑 keyword 调用契约，不能只为统一风格破坏现有接入。

## 常量与数值语义

### 数字具名，系数有来源

Python 使用文件顶部的模块级常量承接“宏”的作用。协议字段、对齐单位、tile 支持集合、资源容量上限和必要的数学常量必须具名，注明必要的规格或接口来源。派生量保留公式，不能把计算结果写成无依据的数字。

```text
# 文件顶部；右侧从规格或实际接口取得
ALIGNMENT_BYTES = 已确认的对齐要求
MAX_BLOCK_COUNT = 当前实现支持的核数上限

# 使用处
padded_bytes = ((raw_bytes + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES) * ALIGNMENT_BYTES
```

- 普通索引起点、计数递增、空长度判断中的 `0/1` 可直接使用；它们不属于隐藏业务含义的数字，无需制造 `ZERO/ONE`。
- `scale` 等可变系数来自公开输入或规格规定的推导式。把任意系数改名为常量仍然属于硬编码；数学近似的系数和适用范围也必须有依据。
- 固定支持范围可以声明为常量，并在入口检查；设备核数等环境事实按 [编译与启动](patterns/compilation-launch.md) 获取，不能固化开发机结果。

### 位运算明确无符号类型与位宽

有符号设备整数不直接参与 `& | ^ ~ << >>`。先确认非负范围、目标位宽和移位量，再使用包内已确认支持的显式无符号类型；掩码和其他操作数采用相容位宽。不能把负数直接转换为无符号值掩盖非法输入。

普通 Python 的 `int` 没有固定无符号位宽；位编码应显式检查范围，按字段定义限制补码取反和移位结果。只是计算整除、取模或对齐时，优先写清楚的算术公式，不为微小优化改写成位技巧。二进制打包确需按位重解释时，明确单独的位编码契约。

## 函数边界与条件校验

同一语义的重复代码超过约 10 行时，考虑提取公共函数；这是检查提示，不是按行数强制拆分。优先复用校验、尺寸推导和重复计算片段，避免分支之间修复不同步。

一个函数完成一个可命名的完整任务，保持参数和返回值明确。短而直观的代码可就地保留；不为一两行增加多层转发，不把参数塞进含义不明的字典，也不为表面相似但语义不同的代码强行抽象。Kernel launch 仍直接位于 `@host`，设备辅助计算按实际支持的 `@jit` 边界提取。

将 tiling 合法性检查集中到 `validate_tiling(...)`，在资源分配、编译和启动前调用。选择策略复杂或需要复用时再提取 `select_tiling(...)`；固定简单配置无需多包一层。基本输入检查见 [Host 校验](patterns/host-validation.md)，同一可信边界内不重复检查。

用提前报错保持正常路径平坦。以下普通 Python 示例检查正整数配置和容量，并指出具体的非法参数：

```python
def validate_tiling(tile_rows, tile_columns, item_bytes, capacity_bytes):
    for name, value in (
        ("tile_rows", tile_rows),
        ("tile_columns", tile_columns),
        ("item_bytes", item_bytes),
        ("capacity_bytes", capacity_bytes),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    required_bytes = tile_rows * tile_columns * item_bytes
    if required_bytes > capacity_bytes:
        raise ValueError(f"tile requires {required_bytes} bytes, capacity is {capacity_bytes}")
```

不要嵌套“检查通过后再检查下一项”的多层 `if/else`。解包、链式比较、集合成员判断、`all/any`、`zip/enumerate` 只在减少理解成本时使用；不同失败原因需要分别定位时保留明确的判断，不为缩短代码合成一个总条件。这些普通 Python 写法不能未经确认搬入 DSL 编译前端。VF 内条件执行见 [VF](patterns/vf.md)。

## 注释与冗余代码

优先用名字和函数结构表达含义。`#` 注释与模块、类、函数的 docstring 使用同一精简标准；把冗余注释搬进 docstring 不算修复。

- 模块说明简述计算职责和支持范围；公开函数说明输入输出契约及非显然限制。内部函数仅补充签名和代码无法表达的约束。
- 同步完成条件、缓冲复用时机、偏移单位、对齐要求、精度及 NaN 边界等必要原因，在最相关的位置用一两句说明。同一事实不在模块说明、函数说明和调用处反复展开；不同使用点确有局部前提时只补充差异。
- 实验过程、临时日志路径、测试通过数量、性能测量记录及修改历史留在调用方提供的设计或验证资料中；源码只保留当前成立的约束。详细推导已有稳定资料时可按需引用，关键安全条件仍须在代码附近说清。
- 删除逐行翻译代码、重复类型或参数列表、长篇背景和已注释掉的旧代码。明显的赋值、分支与函数调用无需补“做什么”的说明。

```text
# 空任务核仍须参与屏障。
if has_work:
    完成本核任务
执行所有参与核的屏障
```

不设机械的注释行数配额；以是否解释了代码本身看不出的原因作为保留标准。

删除恒真 `assert`、同一可信边界内重复且状态未变化的检查、不可达分支、无用变量，以及已确认不改变类型和数值语义的 `+ 0`、`* 1` 等运算。不要用无意义算术制造 Tensor 输出或触发设备执行。

公开 Python 输入校验使用明确的 `TypeError/ValueError` 等异常，不依赖可被优化模式关闭的 `assert`。有实际作用的范围检查、跨边界校验和编译器约束要保留。删除浮点恒等式前确认 signed zero、NaN 和类型转换语义，不能仅按字面形式删代码。

## 设备执行约束

约束覆盖公开入口及其调用的 helper、注册与适配层。以实际是否下发额外设备任务为准；写在 Host 函数中的 Tensor 操作同样受约束。

### Host 职责与设备计算

- Host 负责 `shape/dtype/device/stride` 等元数据校验、Python 标量计算、资源分配、已确认的零拷贝视图和目标 Kernel 启动。
- 设备数据的初始化、计算和必要布局处理由目标 DSL Kernel 完成；包装层不调用其他设备算子补算结果，不使用 `output + 0`、额外乘系数等后处理。
- 普通输入与 tiling 校验只依赖元数据和 Host 参数；不通过 `.item()`、`.cpu()` 或设备 Tensor 转布尔值读取内容，避免引入数据回传与同步。

### 分配、初始化与视图

- 禁止用 `torch.full`、`torch.zeros/zeros_like`、`torch.ones/ones_like`、`Tensor.zero_()/fill_()` 等接口在目标 Kernel 外初始化设备数据；别名、helper 和 `torch.ops` 间接调用遵循同一规则。
- 输出与 workspace 使用 `torch.empty/empty_like` 等未初始化分配接口，并确保实际分配路径不附带设备填充。分配所得内容不可直接读取；目标 Kernel 必须覆盖全部有效输出，在读取临时数据前完成所需写入，并满足跨核、跨阶段的数据依赖。
- 必要的零值或常量填充在目标 Kernel 内完成，不额外启动初始化 Kernel，也不先在 CPU 构造数据再搬到设备绕过约束。
- `view/reshape/contiguous/.to(...)` 等操作仅在已确认不发生数据复制或转换时保留。需要复制的布局调整、`clone`、dtype 转换，以及设备 Tensor 上的算术、比较和归约，不放在包装层执行；不能仅凭接口名称认定操作是零开销的。

### 启动边界

单 Kernel 实现的一次有效计算调用只启动目标 Kernel，不额外启动清零、填充、转换或其他辅助计算。独立 AICPU 计算和再次启动的 Kernel 均属于独立设备执行，不能因名称相同或封装在同一入口内而视为一次启动。

目标 Kernel 内部的 C/V 协作、多核并行、数据搬运与必要同步按计算依赖组织，不属于额外算子启动；同步遵循 [同步](patterns/synchronization.md) 中的约束。

## 轻量检查

- 结合本次改动检查导入位置、命名、常量与系数来源、函数边界、tiling 校验、位运算及冗余代码。
- 阅读完整模块说明、函数 docstring 和本次涉及的注释块：删掉重复叙述与过程记录，并确认同步、复用、单位及数值边界等必要约束仍可就近找到。仅修改注释时保持执行语句不变；修改 docstring 前确认没有读取其内容的调用方或工具。
- 沿公开入口检查直接调用和 helper，明确各操作属于元数据处理、分配、零拷贝视图还是设备执行；注册与适配层遵循相同边界。
- 检查输出覆盖范围、workspace 首次读取前的写入及各执行分支的启动行为。接口语义不明确时先核实实现，不能仅凭名字或搜索结果认定满足约束。
