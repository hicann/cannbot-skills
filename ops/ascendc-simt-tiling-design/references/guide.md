# SIMT 算子切分设计指南

SIMT 算子的切分设计与 SIMD 有本质差异，不涉及 UB 切分和 Buffer 规划，而是以核数切分 + 线程数设置为核心。

## SIMT vs SIMD 切分差异

| 要素 | SIMD 方式 | SIMT 方式 |
|------|----------|----------|
| 多核切分 | 按 UB 单次处理量切分 | 按元素总量切分（ceil(总量/单核最少元素数)） |
| 单核并行 | UB Buffer + 向量指令 | 线程数（constexpr 编译期常量，1024/2048） |
| 数据搬运 | 需显式 Load/Store | 支持直接读写 GM |
| UB 使用 | 全量使用 | 仅核内共享场景使用，DCache ≥32KB |
| Buffer 规划 | inQueue/outQueue/tmpBuf | TBuf（仅在需要共享内存时使用） |

## 核数切分

```
总核数 = ceil(输出元素总数 / 单核最少处理元素数)
单核最少处理元素数建议 1024
单核最少处理元素需要对 warp (即 32) 进行对齐
```

- 通过 tiling 侧 `SetBlockDim` 设置核数
- 合理设置 `perCoreElements` 避免负载不均
- 避免总数据量少但启动核数多的场景

## 线程数设置

- 默认值: 1024
- 最大值: 2048
- 必须是 `constexpr` 编译期常量
- `LAUNCH_BOUND(N)` 和 `Simt::Dim3(N)` 必须使用同一个常量

### 按算子类型选择

| 算子类型 | 建议线程数 | 原因 |
|---------|-----------|------|
| 搬运类算子 | 2048 / 1024 | 更多线程隐藏内存延迟 |
| 计算类算子 | 512 / 1024 | 寄存器压力大，需平衡 |

### 正确写法

```cpp
constexpr uint32_t THREAD_NUM = 512;

// VF 函数声明
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...);

// VF 调用
Simt::VF_CALL<OpComputeSimt<T>>(Simt::Dim3(THREAD_NUM), args...);
```

### 错误写法（严禁）

```cpp
// 线程数从 tiling 数据获取（运行时变量）
int32_t threadNum = static_cast<int32_t>(tilingData_->threadNum);
Simt::VF_CALL<OpComputeSimt<T>>(Simt::Dim3(threadNum), args...);
```

> 如果数据量较小导致线程空转过多，应通过调整**核数**（`SetBlockDim`）来适配，而不是减少线程数。

## DCache 与 UB 空间设置

SIMT 算子不能使用全部 UB 空间，需为 DCache 预留 >=32KB：

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;
uint64_t ubsize = 256 * 1024;
context->SetLocalMemorySize(ubsize - DCACHE_SIZE);
```

可用 UB = 256KB - 8KB(预留) - 32KB(DCache最低) = **216KB**