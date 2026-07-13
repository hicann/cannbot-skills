# 编码红线与解决方案

> 本仓算子编码中明确禁止的红线。每条红线均为「违反 = 检视不通过」的硬约束，触发即为阻塞项。逐条核对；命中任一条，按「解决方案」列修正后方可通过。

## 1. 硬件参数硬编码（阻塞）

所有硬件参数必须运行时动态获取，禁止写死。命中即自动失败。

| 红线 | 检测 | 解决方案 |
|------|------|---------|
| 写死核数 `blockDim = 8;` | `grep -n "blockDim\s*=\s*[0-9]" *.asc` | `aclrtGetDeviceInfo()` 动态获取核数 |
| 写死核索引 `blockIdx = 0;` | `grep -n "blockIdx\s*=\s*[0-9]" *.asc` | `AscendC::GetBlockIdx()` 获取 |
| 写死分块大小 `constexpr TILE = 4096;` / `#define TILE 1024` | 目视 + grep | 根据 UB 容量动态计算分块大小 |
| 任何硬编码 UB 大小 | 目视 | 运行时动态获取 UB 大小 |
| 纯向量算子用 `ACL_DEV_ATTR_AICORE_CORE_NUM` 取核数 | 目视 | 纯向量用 `ACL_DEV_ATTR_VECTOR_CORE_NUM`，矩阵用 `ACL_DEV_ATTR_CUBE_CORE_NUM` |

## 2. 禁用 API（阻塞）

| 红线 | 原因 | 解决方案 |
|------|------|---------|
| `DataCopy(GM, UB)` / `DataCopy(UB, GM)` | 不支持非对齐数据，易致隐蔽 bug | 统一改用 `DataCopyPad` |
| `GlobalTensor::SetValue/GetValue`（非调试） | 逐元素访问效率极低 | `DataCopyPad` 批量搬运 |
| 高阶封装 API（Softmax / LayerNorm 等） | 黑盒、调试困难 | 基础矢量 API（Add/Mul/Sub/Div/Exp/Log/ReduceSum/ReduceMax/Cast）手动组合 |
| 手动循环逐元素累加做归约 | 性能低 | 使用 `ReduceSum` / `ReduceMax` |

```cpp
// ✅ 正确
AscendC::DataCopyPad(xLocal, xGm, {rowsThisLoop, rLength, rLengthAlign, 0, 0});
// ❌ 红线：数据长度非 32 字节倍数时出错
AscendC::DataCopy(xLocal, xGm, dataLength);
```

## 3. Host 侧计算型操作（阻塞）

Host 侧仅允许 Tiling 参数计算、内存分配/释放、Kernel 启动（`<<<>>>`）、结果验证。禁止对 tensor 做计算型操作——部署时输入/输出在 Device 上，无法搬回 Host。

| 红线 | 解决方案 |
|------|---------|
| Host 侧 Transpose（CPU 三重循环交换维度） | 在 Kernel 侧完成，或调整设计避免 Host 预处理 |
| Host 侧 Reshape / Pad（CPU 循环重排数据） | 在 Kernel 侧完成 |
| Host 侧任何对 tensor 的计算型循环 | 移入 Kernel |

## 4. 流水线与数据流（阻塞）

| 红线 | 解决方案 |
|------|---------|
| DataCopy 后未用 EnQue/DeQue 同步 | DataCopy 后补 EnQue/DeQue 配对同步 |
| 最终结果留在 VECIN buffer（inQueue）未写入 VECOUT | 最后一步计算直接写入 outQueue 的 yLocal，或 `DataCopy(yLocal, xLocal)` 后再 EnQue |
| `AllocTensor` 无对应 `FreeTensor`（内存泄漏） | 每个 Alloc 配对 Free；`grep -c AllocTensor`/`grep -c FreeTensor` 核对数量 |
| EnQue/DeQue 未配对（内存泄漏） | 配对；`grep -c EnQue`/`grep -c DeQue` 核对 |
| 冗余 PipeBarrier（同 pipe 连续操作间加 barrier） | 只在跨 pipe 数据依赖点保留 barrier；同 pipe 连续操作（如连续矢量运算、连续归约）删除 barrier |

> Pipe 归属：PIPE_MTE2（GM→UB 搬运）、PIPE_V（矢量/归约/Cast/Duplicate 等）、PIPE_MTE3（UB→GM 搬运）、Scalar（GetValue/SetValue）。跨 pipe 且存在 RAW/WAW 依赖才需 barrier；同 pipe 硬件保序。

## 5. Kernel 内禁止事项（阻塞）

| 红线 | 解决方案 |
|------|---------|
| 前向声明 Kernel 函数（先声明后定义） | Kernel 函数必须定义在调用之前 |
| Kernel 内使用 `std::` 命名空间函数 | 改用 Ascend C API |
| 动态内存分配（`new` / `malloc`） | 用 `TPipe` + `TQue` 管理内存 |
| 递归调用 | 改写为迭代 |
| 使用未初始化变量 | 使用前初始化 |
| `.cpp` / `.cc` 作为 Kernel 文件扩展名 | Kernel 文件必须为 `.asc`（ASC 编译器只识别 `.asc`，`<<<>>>` 也仅在 `.asc` 中有效） |

## 6. API 用法未经验证（阻塞）

| 红线 | 解决方案 |
|------|---------|
| 凭记忆/猜测写入 API，未查文档 | 用任何 API 前通过 `ascendc-docs-search` 查阅 `{API_NAME}` 官方文档；同名 API 多变体（如 `ReduceMax.md` / `ReduceMax-35.md`）须用 `find "$ASC_DEVKIT_DIR/docs/api/" -name "{APIName}*.md"` 全部查阅后再选用 |
| 检视意见推荐 API 未附文档来源 | 推荐 API 的修复建议须附官方文档来源，禁止凭记忆 |
