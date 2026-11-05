# asc\_ldcg

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

从L2 Cache加载缓存的数据，如果缓存命中，则直接返回数据。若未命中，则从Global Memory地址预加载数据缓存至L2 Cache，并返回数据。

## 函数原型

```
inline long int asc_ldcg(long int * address)
```

```
inline unsigned long int asc_ldcg(unsigned long int* address)
```

```
inline long long int asc_ldcg(long long int* address)
```

```
inline unsigned long long int asc_ldcg(unsigned long long int* address)
```

```
inline long2 asc_ldcg(long2* address)
```

```
inline ulong2 asc_ldcg(ulong2* address)
```

```
inline long4 asc_ldcg(long4* address)
```

```
inline ulong4 asc_ldcg(ulong4* address)
```

```
inline longlong2 asc_ldcg(longlong2* address)
```

```
inline ulonglong2 asc_ldcg(ulonglong2* address)
```

```
inline longlong4 asc_ldcg(longlong4* address)
```

```
inline ulonglong4 asc_ldcg(ulonglong4* address)
```

```
inline signed char asc_ldcg(signed char* address)
```

```
inline unsigned char asc_ldcg(unsigned char* address)
```

```
inline char2 asc_ldcg(char2* address)
```

```
inline uchar2 asc_ldcg(uchar2* address)
```

```
inline char4 asc_ldcg(char4* address)
```

```
inline uchar4 asc_ldcg(uchar4* address)
```

```
inline short asc_ldcg(short* address)
```

```
inline unsigned short asc_ldcg(unsigned short* address)
```

```
inline short2 asc_ldcg(short2* address)
```

```
inline ushort2 asc_ldcg(ushort2* address)
```

```
inline short4 asc_ldcg(short4* address)
```

```
inline ushort4 asc_ldcg(ushort4* address)
```

```
inline int asc_ldcg(int* address)
```

```
inline unsigned int asc_ldcg(unsigned int* address)
```

```
inline int2 asc_ldcg(int2* address)
```

```
inline uint2 asc_ldcg(uint2* address)
```

```
inline int4 asc_ldcg(int4* address)
```

```
inline uint4 asc_ldcg(uint4* address)
```

```
inline float asc_ldcg(float* address)
```

```
inline float2 asc_ldcg(float2* address)
```

```
inline float4 asc_ldcg(float4* address)
```

```
inline bfloat16_t asc_ldcg(bfloat16_t* address)
```

```
inline bfloat16x2_t asc_ldcg(bfloat16x2_t* address)
```

```
inline half asc_ldcg(half* address)
```

```
inline half2 asc_ldcg(half2* address)
```

## 参数说明

**表 1**  参数说明

| 参数名 | 描述 |
| --- | --- |
| address | Global Memory的地址。 |

## 返回值说明

L2 Cache中的缓存数据或输入的Global Memory地址上的数据。

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
    __global__ __launch_bounds__(1024) void kernel_asc_ldcg(float* dst, float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = asc_ldcg(src + idx);
    }
    ```

-   SIMD与SIMT混合编程场景：

    SIMD与SIMT混合编程场景，需要显式使用地址空间限定符表示地址空间：\_\_gm\_\_表示Global Memory内存空间。

    ```
    __simt_vf__ __launch_bounds__(1024) inline void kernel_asc_ldcg(__gm__ float* dst, __gm__ float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = asc_ldcg(src + idx);
    }
    ```
