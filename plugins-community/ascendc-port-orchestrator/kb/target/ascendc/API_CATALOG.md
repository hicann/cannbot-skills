# AscendC API Catalog — CANN 9.0.0-beta.2

> **版本锁定**: 本文档对应 CANN 社区版 9.0.0-beta.2 (容器 `/usr/local/Ascend/cann-9.0.0`)。
> 升级 CANN 版本后需重新核对 API 可用性。
> **来源**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0003.html
> **采集日期**: 2026-04-16
> **用途**: Worker 在 benchmark/opgen/optimize 模式下查询可用 API。不含实现源码。

---

## 1. 基础数据结构

| API | 说明 |
|-----|------|
| LocalTensor | Local Memory 数据，支持 VECIN/VECOUT/VECCALC/A1/A2/B1/B2/CO1/CO2 |
| GlobalTensor | Global Memory 全局数据 |
| Coordinate | 元组，表示张量多维位置坐标 |
| Layout\<Shape, Stride\> | 多维张量内存布局模板类 |
| TensorTrait | Tensor 元信息（dtype, TPosition, Layout） |

---

## 2. Memory 数据搬运

| API | 说明 |
|-----|------|
| **DataCopy** | GM↔UB 数据搬运（普通/增强/切片/随路格式转换） |
| **Copy** | VECIN/VECCALC/VECOUT 之间搬运，支持 mask + DataBlock 间隔 |

---

## 3. VEC 矢量计算

### 3.1 基础算术

| API | 说明 | 备注 |
|-----|------|------|
| **Exp** | 按元素 e^x | |
| **Ln** | 按元素 ln(x) | |
| **Abs** | 按元素 \|x\| | |
| **Reciprocal** | 按元素 1/x | |
| **Sqrt** | 按元素 √x | |
| **Rsqrt** | 按元素 1/√x | |
| **Relu** | 按元素 max(0, x) | |
| **Add** | 按元素 a + b | |
| **Sub** | 按元素 a - b | |
| **Mul** | 按元素 a * b | |
| **Div** | 按元素 a / b | |
| **Max** | 按元素 max(a, b) | |
| **Min** | 按元素 min(a, b) | |
| **Adds** | 每元素 + 标量 | |
| **Muls** | 每元素 × 标量 | |
| **Subs** | 每元素 - 标量 (或标量 - 每元素) | 支持标量在前/后 |
| **Divs** | 每元素 / 标量 (或标量 / 每元素) | 支持标量在前/后 |
| **Maxs** | 每元素 vs 标量取大 | |
| **Mins** | 每元素 vs 标量取小 | |
| **LeakyRelu** | 按元素 Leaky ReLU | |

### 3.2 逻辑计算

| API | 说明 |
|-----|------|
| Not | 按位取反 |
| And | 按位与 |
| Or | 按位或 |
| ShiftLeft | 左移 |
| ShiftRight | 右移 |
| Ands | 元素 & 标量 |
| Ors | 元素 \| 标量 |

### 3.3 复合计算

| API | 说明 |
|-----|------|
| Axpy | dst = src * scalar + dst. **Single-round hardware FMA** on Ascend950PR fp32 — bit-matches source `fmaf(src, scalar, dst)` across all test distributions incl. cancellation/denormal/large_mag (BabelStream Triad pilot 2026-04-20, 12/12 bit-exact PASS). **Prefer Axpy over `Muls(tmp,x,a); Add(y,b,tmp)`** for any `y = a*x + b` shape when cross-platform bit-exact matters — 2-op path is 2 rounds and WILL drift 1-ULP at rounding boundaries. Preload dst via `Adds(dst, b, 0.0f, count)` (NOT UB→UB DataCopy — see PB-9). fp16/bf16 tie-case semantics unverified as of pilot. |
| CastDequant | 量化 + 精度转换 |
| AddRelu | a + b → max(0, result) |
| AddReluCast | a + b → max(0, result) → Cast |
| AddDeqRelu | a + b → deq → max(0, result) |
| SubRelu | a - b → max(0, result) |
| SubReluCast | a - b → max(0, result) → Cast |
| MulAddDst | dst = src0 * src1 + dst |
| MulCast | a * b → Cast |
| FusedMulAdd | dst = src0 * dst + src1 |
| MulAddRelu | dst = max(0, src0 * dst + src1) |

### 3.4 比较与选择

| API | 说明 |
|-----|------|
| **Compare** | 逐元素比较，结果为 bitmask |
| Compares | 元素 vs 标量比较 |
| **Select** | 根据 selMask 从 src0/src1 选择元素 |
| GatherMask | Compress positions where packed bitmask bits are set (predicate-driven select). User-defined mode: T ∈ {int32, int64, float, half} + U=uint32 (for 4B T) or uint16 (for 2B T). LSB-first bit ordering: bit i of pattern word w selects src element w·32+i. Counter mode (reduceMode=true, mask=N): processes N elements; `rsvdCnt` returned by reference = number of selected elements. Direct `ReinterpretCast<uint32_t>` of `CompareScalar` packed-bitmask UB output works as `src1Pattern`. Verified on Ascend950PR_9579 (probe `output/npukernelbench/src/kernels/22_Nonzero/probes/gathermask_probe/`). NOTE: SIMD binary-scalar ops (Divs/Muls/Adds/Subs) on GatherMask's int32 output are REJECTED per PB-23 — Cast int32→int64 immediately and do all subsequent index arithmetic on int64. Reference pattern: P-P80. |

### 3.5 精度转换

| API | 说明 | **关键约束** |
|-----|------|------------|
| **Cast** | 精度转换 (fp32↔fp16↔bf16↔int8 等) | **bf16→fp32 必须用 SIMD Cast, 不能 static_cast (PB-4)** |

### 3.6 归约计算

| API | 说明 | 备注 |
|-----|------|------|
| ReduceMax | 全局最大值 + 索引 | |
| ReduceMin | 全局最小值 + 索引 | |
| ReduceSum | 全局求和 | |
| **WholeReduceMax** | 每 repeat 内求最大值 + 索引 | |
| **WholeReduceMin** | 每 repeat 内求最小值 + 索引 | |
| **WholeReduceSum** | 每 repeat 内求和 | 常用于 P-P47 tree reduce 的最后一步 |
| BlockReduceMax | 每 repeat 内求最大值 | |
| BlockReduceMin | 每 repeat 内求最小值 | |
| **BlockReduceSum** | 每 repeat 内求和（二叉树方式） | |
| PairReduceSum | 相邻奇偶元素求和 | |

### 3.7 数据转换

| API | 说明 |
|-----|------|
| Transpose | 16×16 矩阵转置 / NCHW↔NHWC |
| TransDataTo5HD | NCHW→NC1HWC0 格式转换 |

### 3.8 数据填充

| API | 说明 |
|-----|------|
| **Duplicate** | 标量填充到向量 |
| Brcb | 8 个数填充到 8 个 datablock |
| CreateVecIndex | 创建递增索引向量 |

### 3.9 数据收集/分散

| API | 说明 |
|-----|------|
| Gather | 按偏移收集元素 |

### 3.10 掩码操作

| API | 说明 |
|-----|------|
| **SetMaskCount** | Counter 模式（自动推断迭代次数） |
| SetMaskNorm | Normal 模式（手动配置迭代次数） |
| SetVectorMask | 设置 VEC 计算 mask |
| ResetMask | 恢复默认 mask |

### 3.11 量化设置

| API | 说明 |
|-----|------|
| SetDeqScale | 设置 DEQSCALE 寄存器 |

---

## 4. 标量计算

| API | 说明 |
|-----|------|
| GetBitCount | uint64 中 0/1 的个数 |
| CountLeadingZero | 前导零个数 |
| Cast (float→half/int32) | 标量类型转换 |
| Cast (float→bfloat16) | **注意 PB-4: bisheng 可能有 bug** |
| Cast (多类型→float) | bf16→float 标量转换 |

---

## 5. 资源管理

| API | 说明 |
|-----|------|
| **TPipe** | 全局内存资源管理框架 |
| GetTPipePtr | 获取 TPipe 指针 |
| TBufPool | 手动管理 UB/L1 物理内存复用 |
| **TQue** | 队列：入队出队 + 任务间同步 |
| **TQueBind** | 绑定 src/dst 逻辑位置 |
| **TBuf** | 临时变量内存管理 |
| InitSpmBuffer | 初始化 SPM Buffer |
| GetUserWorkspace | 获取用户 workspace 指针 |
| SetSysWorkSpace | 设置系统 workspace 指针 |

---

## 6. 同步控制

| API | 说明 | 常用场景 |
|-----|------|---------|
| TQueSync | 同步控制类 | |
| **SetFlag/WaitFlag** | 不同流水线间同步 | SIMD 精细同步 |
| **PipeBarrier** | 阻塞相同流水 | SIMD 粗粒度同步 |
| DataSyncBarrier | 阻塞直到内存访问完成 | |
| IBSet/IBWait | 核间同步（设置/等待标志位） | 多核协作 |
| SyncAll | 全核同步（硬/软） | |
| InitDetermineComputeWorkspace | 初始化 GM 共享内存用于核间同步 | |
| WaitPreBlock/NotifyNextBlock | 核间有序执行 | |

---

## 7. 原子操作

| API | 说明 |
|-----|------|
| **SetAtomicAdd** | 设置 VECOUT→GM 原子累加 |
| SetAtomicType | 设置原子操作数据类型 |
| DisableDmaAtomic | 清空原子操作状态 |
| **AtomicAdd** | GM 地址原子加 |
| AtomicMin | GM 地址原子取小 |
| AtomicMax | GM 地址原子取大 |
| AtomicCas | GM 地址 CAS 操作 |
| AtomicExch | GM 地址原子交换 |

---

## 8. 系统变量

| API | 说明 |
|-----|------|
| **GetBlockNum** | 当前任务 Block 数 |
| **GetBlockIdx** | 当前 core index |
| GetDataBlockSizeInBytes | datablock 大小 (bytes) |
| GetArchVersion | AI 处理器架构版本号 |
| GetUBSizeInBytes | UB 空间大小 |
| GetRuntimeUBSize | 运行时 UB 空间大小 |

---

## 9. 高阶 API (封装常用算法)

### 9.1 数学计算库 (需要临时 buffer)

每个数学函数需配套 `GetXxxMaxMinTmpSize()` 和 `GetXxxTmpBufferFactorSize()` 来分配临时空间。

| API | 说明 | 备注 |
|-----|------|------|
| **Sigmoid** | sigmoid(x) = 1 / (1 + exp(-x)) | **bit-exact match for CANN aclnnSigmoid path** (op#11 evidence: kernel.h:309/328 `Sigmoid(tmp, work, d_al_fp32_)` → 50/50 PASS for `silu(self) = sigmoid(self) * self`). Header: `adv_api/sigmoid/kernel_operator_sigmoid_intf.h` |
| **Silu** | silu(x) = x * sigmoid(x) (Swish-1) | 直接调用避免手写 sigmoid+mul 分解。Header: `adv_api/silu/kernel_operator_silu_intf.h` |
| **Swish** | swish(x) = x * sigmoid(beta * x) | 含 beta 参数变体。Header: `adv_api/swish/kernel_operator_swish_intf.h` |
| **Tanh** | tanh(x) | bit-exact for CANN aclnnTanh path (推论同 Sigmoid 模式) |
| **Sin** / **Cos** / **Tan** | 三角函数 | |
| **Asin** / **Acos** / **Atan** | 反三角函数 | |
| **Sinh** / **Cosh** | 双曲函数 | |
| **Atanh** / **Asinh** / **Acosh** | 反双曲函数 | |
| **Log** | 对数 (可能是 ln 或 log10，查文档确认) | 与基础 Ln 不同 |
| **Power** | x^y 幂运算 | |
| **Erf** / **Erfc** | 误差函数 / 互补误差函数 | GELU 相关。**bit-exact** for CANN aclnnGelu path |
| **Sign** | 符号函数 | |
| **Floor** / **Ceil** | 向下/向上取整 | |
| **Trunc** / **Frac** | 截断整数部分 / 小数部分 | |

> **CRITICAL — catalog 列表不全 ≠ API 不存在** (EC-34 / OL-91): 当 catalog 没记某 API，**必查**
> `cann-{version}/aarch64-linux/asc/include/adv_api/` 真实目录看 advanced API headers 有哪些。
> Catalog miss → OL-80 grep 不到 ≠ 不存在；只代表"我们 KB 还没记"。op#11 经验：手写
> `Exp + Reciprocal` 分解 sigmoid 必定 ULP-drift 反 CANN aclnnSigmoid，但用 advanced
> `Sigmoid()` API bit-exact 通过。这条规则适用于所有 activation：先 `ls` adv_api 目录，
> 找到对应 intf header，用它，避免手写多项式不匹配 CANN polynomial。

**使用模式**:
```cpp
// 1. 获取临时空间大小
uint64_t tmpSize = 0;
GetTanhMaxMinTmpSize(tmpSize, count);
// 2. 分配临时 buffer
TBuf<VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, tmpSize);
auto tmp = tmpBuf.Get<float>();
// 3. 调用
Tanh(dst, src, tmp, count);
```

### 9.2 其他高阶 API

| API | 说明 |
|-----|------|
| **Matmul** | 矩阵乘 (含 Batch Matmul, 稀疏, 量化等变体) |
| **Softmax** | Softmax 计算 |
| **Sort** | 排序 (basic + advanced, P-P42 hardware sort) |
| **FlashAttention** | FlashAttention |

### 9.3 Reg 矢量计算 (寄存器级 SIMT)

SIMT 模式下的寄存器级计算 API，与上面的 SIMD VEC API 平行但面向单线程标量/寄存器操作：

| 类别 | API | 说明 |
|------|-----|------|
| 数据类型 | RegTensor, MaskReg, AddrReg | 寄存器级数据结构 |
| 搬运 | Move, 连续对齐搬入/搬出, 非对齐搬入/搬出 | UB↔寄存器 |
| 算术 | Add, Sub, Mul, Div, Abs, Relu, Exp, Sqrt, Ln, Log, Log2, Log10, Neg, Max, Min | 寄存器级标量 |
| 归约 | Reduce, ReduceDataBlock, PairReduceElem | 寄存器级归约 |
| 直方图 | **Histograms** | 寄存器级直方图计算 |
| 索引 | Arange | 创建递增序列 |

---

## 10. ISASI 接口 (硬件相关，不保证跨版本兼容)

### 关键 ISASI

| API | 说明 | 常用场景 |
|-----|------|---------|
| **DataCopyPad** | 非对齐数据搬运 | **注意 EC-23: UB→GM 在 A5 会崩溃** |
| SetPadValue | 设置 pad 填充值 | |
| Sort32 | 32 个数排序 | |
| MrgSort/MrgSort4 | 多路归并排序 | |
| Scatter | 按偏移分散写入 | |
| Gatherb | 按偏移收集(字节级) | |
| VectorPadding | 按 datablock 填充 | |
| BilinearInterpolation | 双线性插值 | |

---

## Worker 使用指南

1. **写 kernel 前**: 查本文件确认需要的 API 存在
2. **不确定参数**: 在 hiascend.com 查对应 API 的详细页面（或直接试编译看报错）
3. **bf16 相关**: 始终用 SIMD `Cast()` (PB-4)
4. **归约**: 优先用 `WholeReduceSum` + `BlockReduceSum` 组合 (P-P47 pattern)
5. **排序**: 高阶 `Sort` API (P-P42) 或 ISASI `Sort32` + `MrgSort`
6. **原子操作**: `SetAtomicAdd` (SIMD→GM) 或 `AtomicAdd` (SIMT scalar)
7. **注意**: ISASI 接口不保证跨芯片版本兼容，优先使用基础 API / 高阶 API
