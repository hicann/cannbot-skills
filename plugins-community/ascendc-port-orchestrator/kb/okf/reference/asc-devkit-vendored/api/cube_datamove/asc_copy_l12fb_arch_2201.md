---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "asc_copy_l12fb"
description: "arch 2201（A3/A2）版 asc_copy_l12fb：把量化参数从 L1 Buffer 搬到 Fixpipe Buffer，提供 size 形、n_burst/len_burst 加 gap 的高维切分形与 _sync 同步形；地址须 32 字节对齐，count 形搬运量也须 32 字节对齐。"
tags: [cube_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/c_api/cube_datamove/asc_copy_l12fb/asc_copy_l12fb_arch_2201.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12fb/asc_copy_l12fb_arch_2201.md

# asc_copy_l12fb

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

将数据从L1 Buffer搬运到Fixpipe Buffer中，Fixpipe Buffer用于存放量化参数。

## 函数原型

- 前n个数据搬运
    ```c++
    __aicore__ inline void asc_copy_l12fb(__fbuf__ void* dst, __cbuf__ void* src, uint32_t size)
    ```

- 高维切分搬运
    ```cpp
    __aicore__ inline void asc_copy_l12fb(__fbuf__ void* dst, __cbuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap_size, uint16_t dst_gap_size)
    ```

- 同步搬运
    ```c++
    __aicore__ inline void asc_copy_l12fb_sync(__fbuf__ void* dst, __cbuf__ void* src, uint32_t size)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数起始地址。 |
| src | 输入 | 源操作数起始地址。 |
| size | 输入 | 搬运数据大小（字节）。|
| n_burst | 输入 | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| len_burst | 输入 | 待搬运的每个连续传输数据块的长度，单位为DataBlock（32字节）。取值范围：[1, 65535]。 |
| src_gap_size | 输入 | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |
| dst_gap_size | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考通用地址重叠约束。
- 当采用前n个数据搬运接口时，搬运数据大小要求32字节对齐。

## 调用示例

```cpp
constexpr uint16_t n_burst = 1;
constexpr uint16_t len_burst = 1;
constexpr uint16_t src_gap_size = 0;
constexpr uint16_t dst_gap_size = 1;
__cbuf__ half src[256];
__fbuf__ half dst[256];
asc_copy_l12fb(dst, src, n_burst, len_burst, src_gap_size, dst_gap_size);
```
