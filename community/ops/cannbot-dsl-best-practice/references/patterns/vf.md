# VF：先写清寄存器内的计算和有效 lane

适用于向量加载、逐元素操作、归约和类型转换。具体指令、向量宽度、mask 粒度及支持 dtype 以当前 DSL 和设备为准。下面的代码块为结构伪码；明确列出的 API 使用当前包的实际名称，中文步骤由具体实现填写。

导入方式为 `from cannbotdsl import vf`，在设备 DSL 计算中使用上下文管理器，不使用 `@vf`：

```text
with vf(mode="simd"):
    执行寄存器向量计算
```

## 条件执行：VF 内不写 if

本 skill 要求 VF 区域及其展开的辅助逻辑不使用 `if/elif/else`，也不用条件表达式隐藏分支。能在 VF 外决定是否进入整个计算片段时，优先外置判断；逐 lane 的条件使用受支持的 mask/选择指令。

按条件的作用范围选择结构，避免把所有判断都改成循环：

| 条件控制的范围 | 推荐结构 |
|---|---|
| 是否执行整个 VF 片段 | 在 VF 外判断，片段内部保持顺序计算 |
| 同一向量中哪些 lane 有效 | 使用指令支持的 mask 或选择操作；mask 不能保护不支持 masked load 的完整读取 |
| VF 内某个片段是否执行一次 | 前两种结构不合适时，使用零次或一次的 `for` |

**优先结构：在 VF 外决定是否执行整个片段。**

```text
if execute_fragment:
    with vf(mode="simd"):
        执行完整向量计算片段
```

必须在 VF 内按标量条件执行一段代码时，用执行零次或一次的 `for` 表达。循环上界是已规范化的整数 `execute_count`，在其条件可用的位置计算：条件为假取 0，为真取 1。它是整个片段的标量条件，不能直接把 lane mask 当循环次数。

**正例：条件成立执行一次，自然结束。**

```text
# 文件顶部导入 vf、reg；VF 外准备标量循环次数
execute_count = 将标量条件转换为受支持的整数 0 或 1
with vf(mode="simd"):
    for _ in range(0, execute_count):
        完成一次条件向量计算
    执行后续公共计算
```

**反例：直接分支，或用固定循环掩盖分支。**

```text
with vf(mode="simd"):
    if condition:
        执行条件计算

with vf(mode="simd"):
    for _ in range(1):
        if not condition:
            break
        执行条件计算
```

`range(0, execute_count)` 本身保证最多一次，不在循环体内修改上界或依赖 `break` 退出。无论使用外置判断还是条件循环，被跳过的代码都不能是输出或后续输入的唯一初始化位置。实际整数转换和动态循环以 DSL 前端支持为准；这里的规则是生成约定，不宣称接口本身禁止所有分支。

**检查：** 条件假时执行零次，条件真时恰好一次；两条路径都能继续公共计算，且不会读取未初始化结果。检查 VF 展开的 helper，避免间接带入分支。

## 具体接口：计数 mask 与完整向量读取

```text
from cannbotdsl import vf, reg

# tile 已在 VF 外从 Channel 选出，FP32 UB 起点/跨度满足指令对齐
# backing allocation 覆盖所有完整向量读；VL 是已确认的 FP32 lane 数
with vf(mode="simd"):
    counter = reg.mask_counter(valid_elements)
    for offset in range(0, padded_elements, VL):
        mask, counter = reg.update_mask(counter, elem_bits=32)
        values = reg.vload(tile, offset)
        scaled = reg.vmuls(values, scale, mask=mask)
        reg.vstore(tile, offset, scaled, mask)
```

`update_mask` 返回 mask 与剩余计数，循环中显式接回计数。满 mask 使用 `reg.create_mask(pattern="all", elem_bits=32)`。`reg.vload` 不接受 mask 参数，会读取完整向量；只限制运算/写回不能保护越界 load，因此 `padded_elements` 必须在实际分配范围内。

`vload/vstore(post_update=True)` 将 offset 解释为每次访问后推进的元素步长；采用这种模式时不要再手动累加同一游标。`vload_unpack` 不应依赖 post-update 推进，使用显式地址偏移。Channel 的选择操作始终位于 VF 外。

## 核心结构：一行归约，跨向量块累计

前提：二维输入列连续、行跨度明确，输出为每行一个 FP32 和。`VL` 表示当前计算类型的 lane 数。

```text
对每个 row:
    row_sum = FP32(0)                         # 每行重新初始化
    对 col = 0, VL, 2*VL, ... 且 col < cols:
        valid_lanes = min(VL, cols - col)
        mask = lane_id < valid_lanes
        values = 安全读取 input[row * row_stride + col] 的有效 lane
        values = 转为 FP32；无效 lane 填 0
        chunk_sum = 对有效 lane 做求和归约
        row_sum = row_sum + chunk_sum
    output[row * output_stride] = row_sum     # 每行只写最终结果
```

“安全读取”须落到当前指令支持的 masked load，或已分配且可访问的 padding。只对后续计算加 mask，不能保护前面的完整向量读。归约顺序与累计类型按误差要求选择，不保证与串行求和逐位一致。

**反例：** `row_sum` 放在所有行外面；每个向量块覆盖同一个输出；最后一个块沿用满 mask；FP16 输入未经设计就在 FP16 中累计；用逻辑行宽跨越带 padding 的行。

## 类型转换：分清数值转换和存储打包

```text
源地址按源元素类型与 source_stride 计算
读取源元素 → 按指令要求 unpack 到计算 lane
按舍入/溢出契约执行数值 cast
按目标存储要求 pack → 以 target_stride 写入有效元素
```

cast 改变数值表示，pack/unpack 处理存储与 lane 布局，不能互相替代。不同宽度的类型可能对应不同有效 lane 数与 mask 粒度，须随指令确认；不要复制某个实现的固定向量步长。

**检查：** 行宽为 `VL-1`、`VL`、`VL+1`，两行取不同常数以暴露累计状态泄漏；带行 padding；大幅值和抵消输入的误差；窄化转换的舍入、溢出和特殊值契约。诊断记录所用类型、步长和尾部有效数即可。

## 内存屏障：只保护实际访问依赖

这里所说的 `vmembar` 对应包接口 `reg.vmem_bar(mode)`，可用模式为 `"vst_vld"` 和 `"vld_vst"`。它用于 VF 内存访问顺序，不替代 MTE 与向量流水线之间的事件，也不替代跨核同步；作用域选择见 [同步](synchronization.md)。

- 先确定哪些存储地址重叠，以及是写后读还是读后覆盖，再选择对应模式。在必要的依赖边界插入，不在每条寄存器算术、每个循环迭代或 VF 首尾机械添加。
- 后续仍可直接使用已有寄存器值时，保留该值，避免无必要的 store → barrier → load。寄存器之间的值依赖不需要靠内存屏障表达。若地址互不重叠，或已有机制确实覆盖同一访问顺序，不重复添加。
- 不为减少数量而跨越必需的读写依赖合并屏障；不能仅凭源码顺序或一次结果正确就认定屏障冗余。

**正例：同一 VF 中，写回 scratch 后确需重新加载，且该依赖需要显式排序。**

```text
# scratch 的完整向量范围可访问，full_mask 覆盖写入和后续读取
reg.vstore(scratch, offset, values, full_mask)
reg.vmem_bar("vst_vld")
reloaded = reg.vload(scratch, offset)
使用 reloaded 继续计算
```

**反例：纯寄存器计算之间反复插入屏障。**

```text
scaled = reg.vmuls(values, scale, mask=mask)
reg.vmem_bar("vst_vld")
shifted = reg.vadds(scaled, bias, mask=mask)
reg.vmem_bar("vst_vld")
```

`vld_vst` 用于需要保护的读后覆盖边界；不要把两种模式都插入来掩盖未分析的依赖。检查或调整屏障时，核对生成代码中的相关访存与屏障，再用覆盖该依赖的最小用例验证；普通命名、注释修改无需重复设备测试。
