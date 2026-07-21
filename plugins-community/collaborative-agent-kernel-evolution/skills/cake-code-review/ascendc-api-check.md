# AscendC API 最佳实践参考索引

> 本文件是 `api-best-references/` 目录的导航入口，供 Review 流程和开发者查阅。
> API 合规检查规则的**单一维护点**是 `api-best-references/api-dispatch.json`。

---

## API 类别索引

| API 类别 | 涵盖 API | 核心文档 | 典型场景 |
|---------|---------|---------|---------|
| **算术运算** | Add, Sub, Mul, Div, Adds, Muls, Exp, Log, Sqrt, Rsqrt | [api-arithmetic.md](api-best-references/api-arithmetic.md) | Softmax, LayerNorm, 广播优化 |
| **归约操作** | ReduceMax, ReduceSum, ReduceMin | [api-reduce.md](api-best-references/api-reduce.md), [api-reduce-pattern.md](api-best-references/api-reduce-pattern.md) | Softmax, LayerNorm, ReduceMean |
| **数据搬运** | DataCopy, DataCopyPad | [api-datacopy.md](api-best-references/api-datacopy.md) | 非对齐处理、多维搬运 |
| **Buffer 管理** | TBuf, TQue, AllocTensor, InitBuffer, FreeTensor | [api-buffer.md](api-best-references/api-buffer.md) | Double Buffer、内存规划 |
| **精度转换** | Cast | [api-precision.md](api-best-references/api-precision.md) | FP16/FP32 混合精度 |
| **流水线同步** | EnQue, DeQue, SetFlag, WaitFlag, CrossCoreSetFlag, CrossCoreWaitFlag | [api-pipeline.md](api-best-references/api-pipeline.md) | 多级流水线、事件同步 |
| **Compare 256B对齐** | Compare | [api-restrictions.md](api-best-references/api-restrictions.md#21-compare-api-256字节对齐约束) | Padding 策略 |
| **repeatTime 限制** | repeatTimes ≤ 255 | [api-repeat-limits.md](api-best-references/api-repeat-limits.md) | 分批处理 |
| **API 限制** | - | [api-restrictions.md](api-best-references/api-restrictions.md) | 禁用 API、编译期限制 |
| **Host Runtime** | aclInit, aclFinalize, aclrtSetDevice, aclrtResetDevice, aclrtGetDeviceInfo, aclrtMalloc, aclrtFree | [api-host-runtime.md](api-best-references/api-host-runtime.md) | 设备初始化、核数获取、内存管理 |

---

## 场景索引

| 使用场景 | 相关文档 | 关键技巧 |
|---------|---------|---------|
| **Softmax/LayerNorm** | [api-reduce.md](api-best-references/api-reduce.md), [api-reduce-pattern.md](api-best-references/api-reduce-pattern.md), [api-arithmetic.md](api-best-references/api-arithmetic.md) | 标量操作、广播优化、Buffer 复用 |
| **逐行处理（AR 模板）** | [api-arithmetic.md](api-best-references/api-arithmetic.md) | Adds/Muls、节省 UB |
| **多行广播（ARA 模板）** | [api-arithmetic.md](api-best-references/api-arithmetic.md) | BinaryRepeatParams.src1RepStride=0、分批处理 |
| **非对齐数据** | [api-datacopy.md](api-best-references/api-datacopy.md) | DataCopyPad、32 字节对齐 |
| **混合精度** | [api-precision.md](api-best-references/api-precision.md) | FP16 输入 FP32 计算 |
| **流水线优化** | [api-pipeline.md](api-best-references/api-pipeline.md), [api-buffer.md](api-best-references/api-buffer.md) | Double Buffer、事件同步 |
| **性能调优** | [api-buffer.md](api-best-references/api-buffer.md), [api-repeat-limits.md](api-best-references/api-repeat-limits.md) | Double Buffer、repeatTimes 优化 |
| **遇到 API 限制** | [api-restrictions.md](api-best-references/api-restrictions.md) | 替代方案、避坑指南 |

---

## 快速参考

完整的 API 参数速查表：[api-quickref.md](api-best-references/api-quickref.md)

---

## ⛔️ API 黑名单

详见 [api-best-references/api-blacklist.md](api-best-references/api-blacklist.md)（唯一维护点，新增禁用 API 只改该文件）。
