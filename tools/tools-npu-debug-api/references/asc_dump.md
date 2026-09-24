# asc_dump 调试指南

资料基线及版本适用性见 [平台与编程模式支持矩阵](platform_matrix.md#来源基线)。示例为调用片段，使用前需核对目标版本和编程模式。

## 接口功能

- **输入**：类型 `T`、对应地址空间的有效指针（或受支持的寄存器对象）、`uint32_t desc` 和元素数 `dump_size`。
- **行为**：读取指定范围的数据，并通过调试输出链路携带数据及自定义标识。`asc_dump_*` 显式选择地址空间，`asc_dump` 通过重载选择；寄存器变体另见平台限制。
- **输出**：可供观察的数据内容和来源标识，无返回值；`dump_size` 的单位是元素，不是字节。
- **不保证**：不推断内存分配长度，不自动比较参考结果（golden），不证明被观测数据已由生产者写完；调用者仍需满足原有同步和生命周期要求。内部填充（padding）和 FIFO 协议不是公开调用参数。
- **最小示例**：已知 `src` 指向至少 16 个有效 `half` 元素且数据已就绪时，使用 `asc_dump_gm<half>(src, 1, 16);`。未知长度时不能直接套用。

## 头文件和参数

```cpp
#include "utils/debug/asc_dump.h"

// src、ubuf、l1、l0c 须已具有下表所列的指针限定符。
// 前三者至少含 32 个有效 half 元素，l0c 至少含 16 个有效 float 元素。
asc_dump_gm<half>(src, 1, 32);
asc_dump_ubuf<half>(ubuf, 2, 32);
asc_dump_l1buf<half>(l1, 3, 32);
asc_dump_cbuf<float>(l0c, 4, 16);
```

模板参数 `T` 是数据类型；输入地址必须与地址空间匹配，不能通过强制类型转换将其他内存伪装为目标地址空间；`desc` 是用于区分输出来源的 `uint32_t` 自定义标识；`dump_size` 是要打印的元素个数。

## 地址空间选择

| 数据位置 | 优先接口 | 指针限定符 |
| --- | --- | --- |
| Global Memory | `asc_dump_gm` 或 `asc_dump` 重载 | `__gm__` |
| Unified Buffer | `asc_dump_ubuf` 或 `asc_dump` 重载 | `__ubuf__` |
| L1 Buffer | `asc_dump_l1buf` 或 `asc_dump` 重载 | `__cbuf__` |
| L0C Buffer | `asc_dump_cbuf` 或 `asc_dump` 重载 | `__cc__` |
| SIMD VF 寄存器 | `asc_dump_reg` 或寄存器参数的 `asc_dump` | 寄存器类型，平台受限 |

Ascend 950 的公开文档还描述 BiasTable Buffer 和 Fixpipe Buffer 的变体。只有在当前产品文档和编译头文件同时确认支持时才使用这些变体；Fixpipe 保存的可能是硬件参数位域，输出不一定等于上游原始数据。

## 必须检查的限制

1. `dump_size` 超出输入实际元素数会产生未定义行为；先确认地址和长度，再决定打印元素数。
2. 非 32 字节对齐的 dump 需要考虑末尾 padding 对容量的影响；padding 本身不一定会显示。
3. SIMD 场景单核调试输出默认约 30 KB，超限时可能没有输出；配置和上限必须按当前 CANN 文档确认。
4. SIMD VF 中 `asc_dump`、`printf` 和断言共享预留 UB 空间；单条记录过大或关闭 预留 UB 空间 时，接口可能不可用。
5. `ASCENDC_DUMP=0` 可关闭 dump；该接口只用于调试，不应保留在生产性能路径。
6. 多核同时 dump 时，`desc` 应区分阶段，线程/Block 应适当过滤，避免输出不可读或超限。

## 观测有效性

- 长度明确、已知值且不跨边界的 GM/UB 小数组可用于判断输出是否符合预期。
- 模板类型必须与实际存储类型一致，转储元素数不能超过有效长度。
- `desc`、Block 和地址空间标签用于区分数据来源。
- L1/L0C/VF 特殊 Buffer 的支持及结果格式以目标版本文档为准。
