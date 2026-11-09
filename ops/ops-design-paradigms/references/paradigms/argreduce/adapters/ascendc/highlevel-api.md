# ArgReduce 场景 × API 映射（AscendC）

> **层级**：AscendC 语言适配层。回答"每个场景用什么手法实现选择段"。
>
> **怎么读**：先按场景（AR/ARA）在 §1 总表选一条手法（A~F），再到对应小节看代码骨架。手法之间是"同一件事的不同实现"——都遵守 [selection-semantics.md](../../generic/selection-semantics.md) 的比赛规则（平局/NaN/位置）。谓词真值见 [capability-mapping.md](capability-mapping.md)。

## 1 总表：场景 × 手法

| 场景 | 手法 | 载体 | 适用条件 | 来源 |
|------|------|------|---------|------|
| AR | **A：索引跟踪 API** | `ReduceMax/ReduceMin(..., calIndex=true)` | 简单路径；count 在 API 上界内。**负面实证：NaN 行为不可依赖（argmin 穿刺证伪）——NaN-aware 算子禁用** | Level-2 API |
| AR | **B：寄存器级 ArgMaxV1** | `Compare + Select` 逐行归并（LoadUnAlign 整行载入） | 主路径（工业实现）；支持 int64 变体 | 官方 base.h |
| AR（R 很小） | **C：Gather 转置** | `MicroAPI::Gather` AR→RA 转置 + 一次向量比较 | `R×eb < vRegSize` 且非 32 倍数且 A 够并行 | 官方 ar_gather |
| ARA | **D：Compare+Select 逐行迭代** | `Compare(LE) + Select(TT) + Select(TS)` 沿 R 逐行 | 主路径（Level-2 Pattern API **不返回位置**，不可用） | with-index / 官方 ara |
| ARA | **E：NDDMA 重排** | DataCopy 重排 [A1,R,A0]→[A1,A0,R] 后走 AR 手法 | R×A0 大小合适时 | 官方 ara MODE2 |
| ARA（A1==1） | **F：RA 向量化** | A0 铺 lane（`Duplicate` 等差列），R 逐行比较 | aSize==1 | 官方 ra |
| COPY | 透传 | values `DataCopy`，indices `Duplicate(0)` | R==1 | 官方 copy |
| GROUP_REDUCE 聚合 | 复用 B/D 的 select | 局部 (值,位置) 读写 workspace | — | 官方 group_reduce |

**选型规则**：先按 [mode-mapping.md](mode-mapping.md) 确定模式，再按本表落 API。手法 A 与 B 可达性按平台白名单核实（arch35 工业主路径为 B/C/D/E/F）。

## 2 手法 B：ArgMaxV1（AR 主路径，寄存器级）

工业实现（ops-math `arg_max_with_value_base.h`）的成员函数，结构完整可借鉴：

```cpp
// 签名：值/位置双输出，dimA 行 × dimR 列，startR 块基址（跨块偏移在此施加）
__aicore__ inline void ArgMaxV1(__ubuf__ void* valueDst, __ubuf__ void* indexDst,
                                __ubuf__ void* src, uint16_t dimA, uint16_t dimR, uint32_t startR);
```

核心循环（逐行 LoadUnAlign → 主/尾块归并）：

```cpp
for (uint16_t a = 0; a < dimA; a++) {                    // 逐行
    LoadUnAlign(value0, uSrc, src, offset1);             // 整行载入（非对齐）
    Duplicate(level0, 0, fullMask);                      // 行内位置流起点
    for (r = 0; r < loopTime; r++) {                     // 主块归并：相邻 VL 段两两比较
        LoadUnAlign(value1, uSrc, src, VL);
        Duplicate(level1, r + 1, fullMask);              // 位置流 = 段号×VL（等差生成）
        Compare<CMPMODE::GE>(cmpMask, value0, value1, fullMask);   // GE: first-occurrence
        Compare<CMPMODE::NE>(helpMask, value0, value0, fullMask);  // NaN 自比较
        Or(cmpMask, cmpMask, helpMask, b8Mask);          // NaN 掩码 OR 并入
        Select(value0, value0, value1, cmpMask);         // 值选择
        Select(level0, level0, level1, cmpMask);         // 位置同步选择
    }
    // 尾块处理（tailMask 窄掩码，同结构）→ 值/位置写出
}
```

**int64 变体**：`ArgMaxV1Int64`（值 int64 比较路径）与 `ArgMaxV1Int64ForRLessThanVL64`（R<64 的窄轴优化），按 `ORIG_DTYPE_X == DT_INT64` 编译期选择。

**位置寄存器选型（T5，等宽原则）**：位置通道类型 = 与值通道**等 lane 宽**的整数类型——fp16 → `uint16_t`（128-lane 同宽，掩码共享；工业实现 fp16 路径即此选型）；fp32/int32 → `uint32_t`。**禁止 float 位置寄存器**（lane 不匹配 + `Reduce<SUM>` dtype 表外）。

**跨块偏移**：分块调用时 `startR` 传块基址，块内位置流起点即全局正确——偏移在生成侧一次施加，合并侧不再加。

### 2a `Reduce<SUM>` dtype 白名单（位置拼接的编译期硬约束）

位置拼接（段号×VL + lane → uint32 目的）用 `Reduce<ReduceType::SUM>` 跨宽直写，其 dtype 组合表是**编译期 static_assert 硬约束**（越表 = 编译失败，非运行期错误）。官方表（Reduce.md 表3，CANN 9.0.0 核验）：

| dst ← src | 支持 |
|---|---|
| half ← half / float ← float / int64 ← int64 / uint64 ← uint64 | ✅ |
| int32 ← int16 / **uint32 ← uint16** / int32 ← int32 / uint32 ← uint32 | ✅ |
| **uint32 ← float** | ❌ **表外，编译期拒绝**（实战踩坑：按 float 位置流设计在编译期被拦截） |

uint16 位置的配套展开：`UnPack` 低/高半 → uint32，再 `Adds(startR)` 拼接块基址，reinterpret 直写 int32 输出（无需 Cast）。

## 3 手法 D：ARA 逐行迭代（ARA 主路径）

Level-2 Pattern 归约 API（`Pattern::Reduce::RA` 等）**没有位置输出**，ARA 场景必须手写。骨架（行 = R 步进，lane = A0）：

```cpp
// 布局：UB 内 [R, A0]，沿 R（外层）逐行与累积胜者比较
Duplicate(winValue, /* identity */, mask);
Duplicate(winPos,   0, mask);
for (uint16_t r = 0; r < dimR; r++) {
    Compare<CMPMODE::LE>(cmpMask, winValue, curRow, mask);  // LE: 保留更小=更大值胜（按实例方向）
    Compare<CMPMODE::NE>(nanMask, curRow, curRow, mask);    // NaN 检测
    Or(cmpMask, cmpMask, nanMask, b8Mask);
    Select(winValue, winValue, curRow, cmpMask);
    Duplicate(level1, r, mask);                              // 当前行位置
    Select(winPos, winPos, level1, cmpMask);                 // 位置同步选择
}
// winPos 为 A0 lane 内每列的 R 行号；全局位置换算：行号 + 该输出块的 R 基址
```

**约束**（DAV_2201 实践，arch35 待验证）：

| 约束 | 说明 |
|------|------|
| Compare 对齐 | DAV_2201 要求 count 占 256B 对齐（float 64 元素倍数）；**arch35 部分实证**：Reg 级 Compare 非 32B count 真机通过（穿刺 10002） |
| 位置寄存器 | 行号以**等宽整数**通道存储（fp16 → uint16，块内偏移 ≤ 65536）；DAV_2201 的 float32/2^24 为历史口径 |
| DataCopyPad rightPadding | ≤ 32 字节，大 padding 用 mask 路径 |
| Select 模式 1/2 预留 | DAV_2201 需预留 8K UB（框架管理） |

**跨行位置换算**：胜者行号 r 为块内局部位置；块基址（R 切块偏移）合并时施加（[../../generic/selection-semantics.md](../../generic/selection-semantics.md) §5）。

**搬运效率警示（实测，ARA 路径主要性能风险）**：中间轴归约的跨步搬运——跨步逐元素/逐行搬运的有效带宽可低至官方策略的 30%（实测 505 vs 1661 GB/s，加速比 0.305x）。官方栈疑用 **NDDMA 2D 块搬运/布局转换**（重排 [A1,R,A0]→[A1,A0,R] 后走 AR 手法，即 MODE2；多行大块 NDDMA 实测稳定）。新算子对中间轴大跨步场景（段间隔 ≥ 数百字节）应优先评估 2D 块搬运，避免朴素跨步循环；NDDMA 块粒度选择是待深挖项。

## 4 手法 C/F：Gather 转置与 RA 向量化

```cpp
// 手法 F（RA）：A0 铺 lane —— 构造 0..VL-1 等差 lane 列，R 逐行比较
Duplicate(laneIdx, 0, fullMask);               // lane 列
Add(laneIdx, laneIdx, one, addMask);           // 等差递增（INDEX_ARITH_OP）
for (r = 0; r < dimR; r++) {
    Compare<GT>(cmpMask, curRow, winValue, mask);
    Select(winValue, winValue, curRow, cmpMask);
    Select(winPos, laneIdx /*+ r×A0 基址*/, winPos, cmpMask);
}

// 手法 C（Gather）：R < VL 时把 AR 行转成 RA 列再走 F
Gather(dst, src, idxVec /*= i*dimR+j 索引数列*/, ...);   // AR→RA 转置
// 排除条件：R×eleBytes % 32 == 0（bank 冲突）；源基址须 32B 对齐（不满足先折叠索引通道或放弃）
```

## 5 GROUP_REDUCE 的 workspace 读写

```cpp
// Phase 1 写（局部胜者，含核基址偏移后的全局位置）：
CopyOutWorkSpace(blockIdx * outAAlign);      // [value 段 | indice 段] 连续布局
SyncAll();
// Phase 2 聚合（复用手法 B 的 select 结构，逐核两两归并）：
for (core = 1; core < originCoreNum; core++) {
    Compare<GE>(cmpMask, bestValue, wsValue[core], mask);  // + NaN OR + 平局 GE/LT 按规则
    Select(bestValue, bestValue, wsValue[core], cmpMask);
    Select(bestPos,   bestPos,   wsPos[core],   cmpMask);
}
```

**聚合三要素**（缺一即静默错误）：核基址偏移已施加（Phase 1 侧）、平局规则参与聚合比较、均衡分配无空块核。

**workspace 写读细节（工业口径）**：Phase 1 写出 `blockLen = 有效元素数 × 元素宽度`（仅写 outASize 个有效元素，不写满对齐槽位——槽位尾垃圾由 Phase 2 掩码隔离；写满超集亦可，需核算越界安全性）。

## 5a 编译器约束（bisheng / ccec）

| 约束 | 说明 |
|------|------|
| **双 `__simd_vf__` 共存崩溃** | bisheng ≤ 15.0.5：同一编译单元多个具名 `__simd_vf__` 函数（如 ArgMaxV1VF + UpdateResultVF 共存）在 **-O2 及以上**触发前端 SIGSEGV（exit 139，产物 .o 为 0 字节）；-O0/-O1 与单 VF 提取不触发。**规避**：VF 计算体保持 `__VEC_SCOPE__` 内联形态（工业实现同款）；确需提取时单编译单元仅 1 个具名 VF 并全量编译验证。最小化复现：bisect 单文件单 TilingKey 实例化编译 |
| **多 TilingKey 实例共存的 WithoutCutR 异常（实测坑族）** | 主线包形态（多 TilingKey 模板实例共存 + `if constexpr` TPL 分发）下，AR WithoutCutR 路径大 R 有**精确异常边界**（实测 R=3030 OK / 3031 aicore exception）；**同算法直调单实例复刻全 OK**——算法链无罪，疑 bisheng 多实例共存代码生成问题（与双 `__simd_vf__` 崩溃同族）。**规避**：WithoutCutR 设安全上界守卫（如 `kWithoutCutRSafeR`），超界走 WithCutR 多段搬运（有性能代价，需在设计中权衡标注）；完整定位需按 TilingKey 实例反汇编比对 |
| **多行连续大块搬运对 HBM 碎片化敏感（实测）** | 长进程 HBM 碎片化（交错分配/半释放数 GB）下，多行连续大块（实测 96KB）DataCopyPad 失败；小块（6.7KB）与单行块安全。**规避**：块大小上界守卫 + 全量 ST 长进程回归（单跑稳定全量跑必现，须以全量口径验收） |
| TPL 模板参数机制 | 新式工程**禁止废弃的 `TILING_KEY_IS` 宏**，必须 `ASCENDC_TPL_ARGS_DECL/SEL`（见 [overview.md](overview.md) §3.1） |

## 6 禁止事项

1. 禁止用无位置输出的 Pattern 归约 API 实现选择段（ARA 场景常见误用）
2. 禁止聚合/合并比较退化为纯值比较（丢平局裁决 → 跨核结果不确定）
3. 禁止位置寄存器背离等宽选型（float 位置通道：lane 不匹配 + `Reduce<SUM>` 表外双重不可行）；块内偏移超位置元素容量未分段视为设计缺陷
4. 禁止 NaN 靠主规则自然处理（比较传递性被破坏，必须显式自比较掩码）
5. 禁止 Gather 路径在 32 倍数条件命中或源基址未 32B 对齐时使用（bank 冲突 / 对齐违例）
