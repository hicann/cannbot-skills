# 迁移改写配方

> 阶段 2 的操作手册。**原则：先删后改，改的是语义不是名字**。API 签名以目标 CANN 的 `include/` 头文件为准，本文给出形态与坑位，不替代查证。

## 1. 总流程

```
① 冻结语义基线（公式/dtype/layout/边界/精度口径）
② 删除 A2 不存在的构造（RegBase、SIMT、A5 通路、A5 dtype、A5 同步、A5 SPR）
③ 按 A2 编程模型重建计算主体（MemBase SIMD / AICPU）
④ 重算预算与 tiling（UB 192KB、L0C 128KB、BT 1KB、核数按平台）
⑤ 重接工程（arch22 / soc / config / CMake / def）
⑥ 编译 → 精度 → 性能（阶段 3–5）
```

**不要**在删除完成前接工程编译：A5 残留会让编译器报一堆与真实问题无关的错，浪费构建轮次。

## 2. RegBase / MicroAPI → MemBase SIMD

### 2.1 形态映射（反向对照，来源：`ascendc-cross-gen-migration` 的 `api-mapping.md`）

| RegBase（A5） | MemBase（A2） | 注意事项 |
|---|---|---|
| `RegTensor<T> reg` | `LocalTensor<T> buf`（`TBuf`/`TQue` 申请） | UB 容量成为硬约束，先算预算 |
| `MaskReg p = UpdateMask<T>(count)` | count 参数 + 尾部 `DataCopyPad` 处理 | A2 的 count 语义是元素数；尾块用 `DataCopyPad` 或补零 |
| `CreateMask<T, MaskPattern::ALL>()` | 无对应；整块处理 | — |
| `__VEC_SCOPE__ { ... }` | 直接写在 kernel 主体 | 删除作用域，改由 `SetFlag/WaitFlag` 保序 |
| `DataCopy<T, LoadDist::DIST_UNPACK_B16>(reg, addr)` | `DataCopy(local, gm, count)` | 加载即解包的行为要显式 `Cast` |
| `DataCopy<T, StoreDist::DIST_NORM/PACK>(addr, reg, mask)` | `DataCopy(gm, local, count)` / `DataCopyPad` | 打包存储（fp32→fp16 压缩）需显式 `Cast` 后存 |
| `MicroAPI::Mul/Add/Sub/Div/Max/Min/Abs/Exp/Ln/Sqrt/Rsqrt/Reciprocal(regDst, regSrc..., mask)` | `Mul/Add/Sub/Div/Max/Min/Abs/Exp/Ln/Sqrt/Rsqrt/Reciprocal(dst, src..., count)` | 同名不同签名：**不能只改类型名** |
| `MicroAPI::Muls/Adds(regDst, regSrc, scalar, maskFull)` | `Muls/Adds(dst, src, scalar, count)` | 标量类型按 dtype 显式转换（`(half)1.0f` 之类） |
| `Cast<T1, T0, CAST_FP32_TO_FP16>(dst, src, mask)` | `Cast<half, float>(dst, src, RoundMode::CAST_RINT, count)` | CastTrait 的舍入/饱和语义要逐条映射到 `RoundMode`，见 `ascendc-api-best-practices` |
| `ReduceSum/ReduceMax/ReduceMin(regDst, regSrc, mask)` | `ReduceSum/ReduceMax/ReduceMin(dst, src, work, count)` 或 `WholeReduceSum/BlockReduceSum` | 需要 work buffer；归约轴语义要对齐 |
| `LocalMemBar<MemType::UB>()` | `SetFlag/WaitFlag` 或 `PipeBarrier` | A2 没有隐式屏障，必须显式 |
| `Sort/MrgSort4/Extract/GatherMask(..., mask)` | 同名 API，参数用 count | 语义接近，仍要核对输出布局 |


### 2.3 CastTrait → RoundMode 映射（逐条对照）

RegBase 的 `Cast` 用 `CastTrait`（模板参数，含舍入/饱和语义）；MemBase 用 `AscendC::RoundMode` 枚举。常见映射：

| RegBase CastTrait | A2 `RoundMode` | 说明 |
|---|---|---|
| `CAST_NONE` | `RoundMode::CAST_NONE` | 无舍入（同类型/位宽重解释），跨 dtype 时需确认语义 |
| `CAST_RINT` | `RoundMode::CAST_RINT` | 就近偶数（默认推荐） |
| `CAST_ROUND` | `RoundMode::CAST_ROUND` | 四舍五入 |
| `CAST_FLOOR` | `RoundMode::CAST_FLOOR` | 向下取整 |
| `CAST_CEIL` | `RoundMode::CAST_CEIL` | 向上取整 |
| `CAST_TRUNC` | `RoundMode::CAST_TRUNC` | 截断 |
| `CAST_ODD` | `RoundMode::CAST_ODD` | 奇数舍入 |
| `CAST_FP32_TO_FP16` / `CAST_FP16_TO_INT8` 等组合名 | 对应 `RoundMode::CAST_RINT`（除非原语义明确为其他舍入） | **组合名不是枚举**：拆成 `Cast<dstType, srcType>(..., roundMode, count)`，并逐条核对舍入 |
| `CAST_SATURATE`（饱和） | 无独立枚举 | 溢出语义由目标 dtype + `RoundMode` 决定；INT8 饱和/回绕需用边界值用例验证，不能假设与 A5 一致 |

**硬性要求**：
- 每一处 RegBase `CastTrait` 都要在迁移记录中写明「原 trait → 目标 RoundMode → 验证用例」；
- FP32→FP16/BF16、FP32→INT8、FP16→INT8 至少各 1 个含 subnormal/饱和边界的用例；
- 只改类型名、不映射 RoundMode 会导致精度整体偏移（常见现象：matched_ratio 95%–99%，难定位）。

### 2.4 FP8 非等价处理的影响记录

FP8（E4M3/E5M2）在 A2 上无原生计算通路，只能反量化到 bf16/fp16 再算。必须记录：

| 影响项 | 说明 |
|---|---|
| 精度 | golden 按“反量化后的语义”计算；A5 FP8 原输出只作参考，不作失败判据 |
| 带宽/UB | 反量化后每元素 **1B→2B**，同一 tile 的 UB 占用与 MTE2 搬运量翻倍 |
| MTE2 压力 | 需重算 tile 与 double buffer；`a2_ub_budget.py` 的 buffer 清单要按 2B 元素重算 |
| 性能 | 同一 shape 下 MTE2 时间可能成为主 Bound；性能对比必须用相同反量化路径 |

### 2.2 改写步骤

1. **切边界**：把每个 `__VEC_SCOPE__` 段当作一个"VF 段"，标注它的输入/输出 buffer 与 tile 大小。
2. **算预算**：把段内所有寄存器张量折算成 UB buffer（元素数 × sizeof + 对齐），用 `scripts/a2_ub_budget.py` 复核 192KB。
3. **搬入**：`DataCopy`/`DataCopyPad` 到 `VECIN` 队列；非连续输入先重排或改用带 stride 的搬运。
4. **计算**：按 count 分段循环调用 MemBase API；尾块单独处理（不能用 RegBase 的 mask 自动兜底）。
5. **搬出**：`Cast`（如需要）→ `DataCopyPad` 处理非对齐写。
6. **同步**：`MTE2→V`、`V→MTE3` 各补一对 `SetFlag/WaitFlag`；同 pipe 用 `PipeBarrier`。

### 2.3 高频坑

- **mask → count 的尾块**：RegBase 里 `UpdateMask` 自动处理尾巴，MemBase 里必须显式判断 `tail = n % VL`，否则越界读或脏数据。
- **CastTrait 的饱和/舍入**：`CAST_FP16_TO_INT8` 带 SAT、`CAST_FP32_TO_FLOAT8` 带 SAT 等，映射到 `RoundMode` 时要逐条核对，否则精度测试在小值/极值处爆掉。
- **Pack/Unpack**：RegBase 的 `Pack` 是寄存器位宽压缩，MemBase 没有等价指令，必须用 `Cast` + 布局约定实现。
- **寄存器复用假设**：RegBase 里"两个 reg 不同时活"的假设，在 UB buffer 上不成立，buffer 申请要按峰值。

## 3. SIMT → SIMD / AICPU

### 3.1 先分类

| SIMT 模式 | A2 首选方案 |
|---|---|
| 逐元素 map（`y[i]=f(x[i])`） | 直接 SIMD：`DataCopy` + 向量 API，按 VL 分段 |
| 连续区间读写 | SIMD `DataCopy` + 偏移计算 |
| 规则索引（等差/分块） | `ArithProgression` / `Brcb` / `GatherMask` 生成索引后 `Gather` |
| 不规则索引 gather/scatter | SIMD `Gather`/`GatherMask`（可接受时）；否则 AICPU |
| 排序 / 去重 / 分段归约 | `MrgSort`/`Sort` + `GatherMask`；复杂度高时 AICPU |
| 线程间协作（shuffle、原子序） | AICPU（或改确定性算法） |

### 3.2 改写要点

- SIMT 的 `GetBlockIdx/GetThreadIdx` 线程索引，在 SIMD 里对应"核号 + tile 内偏移"，用 `GetBlockIdx()` + 循环表达。
- SIMT 的全局内存直读，在 A2 上要么走 UB 中转（`DataCopy` 到 UB 再算），要么用 `Gather` 系列直接取；不要在 A2 上假设存在等价"缓存直读"语义。
- AICPU 实现要和主 kernel 的 workspace 契约、同步顺序一起设计，避免"AICPU 还没写完 AIV 就读"。

## 4. dtype 迁移配方

### 4.1 FP8 / HiFloat8 → bf16/fp16（反量化）

```
输入 fp8 + scale(zp) → Cast 到 bf16/fp16（乘 scale） → 计算 → 输出按需量化回 int8/bf16
```

- 转换方式先查证：目标 CANN 若提供 fp8/hif8→fp16 的 `Cast`，直接用；否则按位解码（整数位运算）或 host 侧预处理。**以头文件为准，不凭记忆写转换。**
- 位置：优先在 kernel 内搬入后立即转换（省 UB），tensor 很大且转换可复用时才考虑 host 侧预处理。
- 精度：反量化是"用更高精度重算"，通常优于 A5 的 fp8 计算；但**与 A5 输出不再逐位可比**，验收按 A2 目标 dtype 的标准来。
- 性能：带宽从 1B/元素变 2B/元素，MTE2 压力上升；若原算子受 MTE2 限制，需在阶段 5 重点看。

### 4.2 FP8 → int8（走量化路径）

- 需要 per-tensor / per-channel scale 与 zero-point 语义；不是无损替换。
- 适用：原算子本就是量化 matmul，且 A2 的 int8 Mmad 吞吐足够。
- 必须记录：量化粒度、饱和策略、精度对比。

### 4.3 MXFP8/MXFP4 → 重新设计量化粒度

- 块缩放（32 元素共享指数）在 A2 无硬件支持，按 per-channel/per-token 重做会改变数值语义。
- 必须经用户确认；不接受则判定"该 dtype 不支持"，在 `_def.cpp` 的 A2 配置里去掉该 dtype，并在报告中列出。

### 4.4 共同红线

- 禁止 reinterpret / 指针强转。
- 禁止只改 `_def.cpp` 的 dtype 列表而不改 kernel（会导致 A2 上跑到未实现的模板分支）。
- 每个被迁移的 dtype 都要有独立精度用例（含极值、次正规、全零、饱和边界）。

## 5. Cube 通路改写

### 5.1 L0C→UB 直通 → workspace 中转

```cpp
// A5（示意）：C 直接落 UB
// FixpipeConfig cfg; cfg.isToUB = true; ...
// A2：C 落 GM workspace，再搬回 UB
// 1) tiling 侧把 workspace 大小算进去（GetWorkspaceSizes）
// 2) Fixpipe 写 workspace（按 FixpipeParamsV220 组织参数）
// 3) DataCopy(ubC, workspaceGm, count) 后做后处理
```

注意：多一次 GM 往返 = 带宽 + 延迟，阶段 5 必须评估"缩小 base 块分段回写"与"整块 GM 中转"哪个更快。

### 5.2 Fixpipe 参数

- A5 的 `FixpipeParamsArch3510` / `FixpipeParamsC310` 字段单位（`srcStride` 以 C0_SIZE、`dstStride` 以 element）与 A2 的 `FixpipeParamsV220`（`dstStride` 以 datablock）**不同**，逐字段重算，不要改类型名了事。
- `nSize` 的 16 倍数约束按目标 API 文档核对。

### 5.3 分形与 LoadData

- A5 L0A 分形 NZ → A2 ZZ：`LoadData` 的 `srcStride/dstStride`、`repeatTimes` 全部重算；建议先用最小矩阵（16×16）打通，再放大。
- `LoadData2DParamsV2`（mStep/kStep）在 A2 是否存在取决于 CANN 版本：查头文件；不存在则改 `LoadData2DParams`。

### 5.4 同步

- A5 的 BufferID 隐式同步在 A2 不存在：每个 `LoadData→Mmad→Fixpipe` 链都要显式 `SetFlag/WaitFlag`（`MTE1→M`、`M→FIX`、`FIX→MTE3` 等），flagId 不能与 KFC/框架占用冲突。
- 双主模式删除后，KFC 回到消息驱动：原"双主下才成立"的并行假设要重新审查。

## 6. 同步改写清单

| 场景 | A2 写法 |
|---|---|
| 搬入 → 计算 | `SetFlag<HardEvent::MTE2_V>(id)` / `WaitFlag<HardEvent::MTE2_V>(id)` |
| 计算 → 搬出 | `SetFlag<HardEvent::V_MTE3>(id)` / `WaitFlag<HardEvent::V_MTE3>(id)` |
| 同 pipe 内保序 | `PipeBarrier<PIPE_V>()` / `<PIPE_MTE2>` 等 |
| 跨核（AIC↔AIV） | `CrossCoreSetFlag<2, PIPE_...>` / `CrossCoreWaitFlag<2, PIPE_...>`（模式 2）；必要时 `SyncAll` |
| 多核间 | `SyncAll` / workspace 标志位（按仓约定） |

**flagId 分配**：统一在一处定义常量表，避免与 KFC（框架占用 15/31）和自定义同步冲突；A5 的 `CrossCoreSetFlag<4,...>` 不能照抄。

## 7. Subnormal / FTZ 反向处理

1. **删除** `PRECISION_1ULP_FTZ_FALSE`、`PRECISION_0ULP_FTZ_FALSE`、`XxxSpecificMode`、A5 专用 `XxxConfig` 模板参数。
2. **eps 是否保留**按用户需求二选一，并写入报告：
   - 用户要求"跨平台行为尽量一致"→ 保留 eps，注明它带来的偏差与性能代价；
   - 默认（推荐）→ 删除 eps，让 A2 用它原生的 subnormal 能力，配 subnormal 用例验证。
3. **验证**：构造含 subnormal 输入与 subnormal 中间结果的用例（如极小值 `Ln/Sqrt/Div`），确认 A2 输出不再被清零。
4. **不要**把"A5 上清零、A2 上非零"直接判为精度失败：先按 `references/precision-verification.md` 分桶，确认是 FTZ 差异还是算法错误。

## 8. 预算与 tiling

改写引入的 buffer 变化（寄存器→UB、workspace 中转、fp8→bf16 变宽）都要进预算表。公式与脚本见 `references/resource-budget.md`。
