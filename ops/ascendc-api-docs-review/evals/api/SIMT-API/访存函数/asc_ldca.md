# asc\_ldca

## 功能说明

首先从Data Cache加载缓存数据，若未命中，则尝试从L2 Cache加载。L2 Cache与Global Memory之间的数据一致性由硬件保证。如果Data Cache和L2 Cache中均未找到所需数据，则从Global Memory中读取数据，然后将其缓存到L2 Cache和Data Cache中。

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品AI Core：不支持
- Atlas 推理系列产品Vector Core：不支持
- Atlas 训练系列产品：不支持

## 函数原型

```
inline long int asc_ldca(long int* address)
```

```
inline unsigned long int asc_ldca(unsigned long int* address)
```

```
inline long long int asc_ldca(long long int* address)
```

```
inline unsigned long long int asc_ldca(unsigned long long int* address)
```

```
inline long2 asc_ldca(long2* address)
```

```
inline ulong2 asc_ldca(ulong2* address)
```

```
inline long4 asc_ldca(long4* address)
```

```
inline ulong4 asc_ldca(ulong4* address)
```

```
inline longlong2 asc_ldca(longlong2* address)
```

```
inline ulonglong2 asc_ldca(ulonglong2* address)
```

```
inline longlong4 asc_ldca(longlong4* address)
```

```
inline ulonglong4 asc_ldca(ulonglong4* address)
```

```
inline signed char asc_ldca(signed char* address)
```

```
inline unsigned char asc_ldca(unsigned char* address)
```

```
inline char2 asc_ldca(char2* address)
```

```
inline char4 asc_ldca(char4* address)
```

```
inline uchar4 asc_ldca(uchar4* address)
```

```
inline short asc_ldca(short* address)
```

```
inline unsigned short asc_ldca(unsigned short* address)
```

```
inline short2 asc_ldca(short2* address)
```

```
inline ushort2 asc_ldca(ushort2* address)
```

```
inline short4 asc_ldca(short4* address)
```

```
inline ushort4 asc_ldca(ushort4* address)
```

```
inline int asc_ldca(int* address)
```

```
inline unsigned int asc_ldca(unsigned int* address)
```

```
inline int2 asc_ldca(int2* address)
```

```
inline uint2 asc_ldca(uint2* address)
```

```
inline int4 asc_ldca(int4* address)
```

```
inline uint4 asc_ldca(uint4* address)
```

```
inline float asc_ldca(float* address)
```

```
inline float2 asc_ldca(float2* address)
```

```
inline float4 asc_ldca(float4* address)
```

```
inline bfloat16_t asc_ldca(bfloat16_t* address)
```

```
inline bfloat16x2_t asc_ldca(bfloat16x2_t* address)
```

```
inline half asc_ldca(half* address)
```

```
inline half2 asc_ldca(half2* address)
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| address | 输入 | Global Memory的地址。 |

## 返回值说明

Data Cache或L2 Cache中的缓存数据，或输入的Global Memory地址上的数据。

## 约束说明

无

## 需要包含的头文件

使用除half、half2、bfloat16\_t、bfloat16x2\_t类型之外的接口需要包含"simt\_api/device\_functions.h"头文件，使用half和half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t和bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/device_functions.h"
```

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例

-   SIMT编程场景：

    ```
    __global__ __launch_bounds__(1024) void kernel_asc_ldca(float* dst, float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = asc_ldca(src + idx);
    }
    ```

-   SIMD与SIMT混合编程场景：

    SIMD与SIMT混合编程场景，需要显式使用地址空间限定符表示地址空间：\_\_gm\_\_表示Global Memory内存空间。

    ```
    __simt_vf__ __launch_bounds__(1024) inline void kernel_asc_ldca(__gm__ float* dst, __gm__ float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = asc_ldca(src + idx);
    }
    ```
