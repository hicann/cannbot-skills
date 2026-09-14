# UB 预算估算与溢出排查指南

> **适用范围**: A5 (Ascend 950PR) Vector 算子，UB = 256 KB/AIV；**cube 类算子中 AIV 侧 Vector 路径（UB 由 AIV 核承载）同样适用**——cube 算子的 AIC 侧存储（L0A/L0B/L0C/L1）预算见 `cube-migration-guide.md` 改动 5「片上缓冲容量对照」
> **何时使用**: 编译通过但运行时 507035 / error 340 / "UB out-of-range"；设计 tile 大小时；多 buffer 场景下估算 UB 占用
> **代号说明**: 文中 `P-P*` / `OL-*` 为插件 KB 模式库代号，见 `kb/target/ascendc/patterns/PATTERN_INDEX.md`

---

## 一、UB 预算速查公式

### 1.1 最简公式（Kernel 侧自查）

```cpp
// 每个 buffer 占用的 UB 字节
// TQue:  InitBuffer(que, depth, tileElems * sizeof(T))    → depth × tileElems × sizeof(T)
// TBuf:  InitBuffer(buf, len)                               → len

// 总 UB = Σ(TQue × depth × tileElems × sizeof(T)) + Σ(TBuf × len)

// ⚠️ TBuf 只占 1 份（无 depth 概念），TQue 占 depth 份
```

### 1.2 Host 侧 tileRows 公式

```cpp
uint64_t ubSize;
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

// 每行 UB 占用 = 对齐列数 × 元素大小 × buffer 份数
uint32_t tileRows = ubSize / (alignedCols * sizeof(T) * bufferCount);

// bufferCount 示例：
//   inQueue(VECIN depth=2) = 2 份
//   outQueue(VECOUT depth=2) = 2 份
//   tmpBuf(VECCALC) = 1 份
//   fp32CastBuf(VECCALC) = 1 份
//   bufferCount = 2+2+1+1 = 6
```

### 1.3 SIMT DCache 预留（参考插件 KB `kb/target/ascendc/migration/l1-l2-implementation-guide.md`）

```cpp
namespace Ops { namespace Common {
    constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;  // 40 KB

    inline uint64_t GetAvailableUbSize(platform_ascendc::PlatformAscendC& platform,
                                        bool isRegbase) {
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (isRegbase) ubSize -= SIMT_UB_SIZE_BYTE;
        return ubSize;
    }
}}

// 使用时：
//   SimtMode:    可用 UB = 256 KB - 40 KB = 216 KB
//   pure-SIMD:   可用 UB = 256 KB
```

---

## 二、UB 分配表模板

### 2.1 逐元素算子（双输入 + 双缓冲）

| Buffer 名称 | 单块大小（字节） | 用途 | 数量 | 总大小 |
|------------|----------------|------|:--:|--------|
| inQueueX   | TILE × elemSize | 输入数据缓冲 (VECIN) | 2 | TILE × elemSize × 2 |
| inQueueY   | TILE × elemSize | 输入数据缓冲 (VECIN) | 2 | TILE × elemSize × 2 |
| outQueue   | TILE × elemSize | 输出数据缓冲 (VECOUT) | 2 | TILE × elemSize × 2 |
| **总计**   | — | — | — | **TILE × elemSize × 6** |

约束: `TILE × elemSize × 6 ≤ 可用 UB`

### 2.2 逐元素带 fp32 提升（fp16/bf16）

| Buffer 名称 | 单块大小（字节） | 用途 | 数量 | 总大小 |
|------------|----------------|------|:--:|--------|
| inQueue    | TILE × 2 | 输入 (half) | 4 | TILE × 8 |
| outQueue   | TILE × 2 | 输出 (half) | 2 | TILE × 4 |
| castBuf    | TILE × 4 | fp32 中间结果 | 1 | TILE × 4 |
| **总计**   | — | — | — | **TILE × 16** |

约束: `TILE × 16 ≤ 可用 UB`

### 2.3 归约算子（单轴归约）

| Buffer 名称 | 单块大小（字节） | 用途 | 数量 | 总大小 |
|------------|----------------|------|:--:|--------|
| inQueue    | TILE × elemSize | 输入缓冲 | 2 | TILE × elemSize × 2 |
| tmpBuffer  | reduceLen × elemSize + 32 | 归约临时 | 1 | reduceLen × elemSize + 32 |
| **总计**   | — | — | — | **TILE × elemSize × 2 + reduceLen × elemSize + 32** |

> ⚠️ 归约临时 buffer 逻辑上只需存 1 个值，但硬件要求最小 32B。

### 2.4 LayerNorm 三阶段（多阶段复用）

| 阶段 | Buffer | 单块大小 | 数量 | 阶段内总计 |
|------|--------|---------|:--:|----------|
| Phase 1: mean | inQueue + calcBuf | TILE×4 | 3+1 | TILE×4×4 |
| Phase 2: var | inQueue + calcBuf | TILE×4 | 3+1 | TILE×4×4 |
| Phase 3: norm | inQueue + outQueue + wBuf + bBuf | TILE×4 | 2+2+1+1 | TILE×4×6 |
| **峰值** (Phase 3) | — | — | — | **TILE × 24** |

> 多阶段算子取各阶段的 **峰值** 而非累加，前提是前后阶段通过 PipeBarrier 释放 buffer。

---

## 三、tileLength 反推公式

### 3.1 bufferCoefficient 方法

```cpp
// float32 单输入: 系数 = 4   (1 inQ×2 + 1 outQ×2 = 4)
// float32 双输入: 系数 = 6   (2 inQ×2 + 1 outQ×2 = 6)
// float16 单输入+fp32升精: 系数 = 8
// float16 双输入+fp32升精: 系数 = 12

int64_t bufferCoefficient = 6;  // 从上面的 UB 分配表推导
int64_t maxTileElements = UB_LIMIT / bufferCoefficient;

// 32 字节对齐
int64_t alignElements = 32 / sizeof(T);
int64_t tileLength = (maxTileElements / alignElements) * alignElements;
```

### 3.2 直接反推

从 UB 分配表的总计公式直接反推 TILE：

```
总公式: TILE × elemSize × bufferCount ≤ 可用 UB
反推:   TILE ≤ 可用 UB / (elemSize × bufferCount)

// fp32 双输入 depth=4:
//   TILE ≤ 256×1024 / (4 × 6) = 256×1024 / 24 = 10922
//   32B 对齐: TILE = 10920 (8 的倍数)
```

---

## 四、10 个典型算子 UB 用量参考卡

| 算子 | UB 总容量 | SIMT 预留 | 可用 UB | 主要 UB 用途 | 来源 |
|------|:-------:|:-------:|:------:|------|------|
| MoeInitRouting | 256KB | 40KB | 216KB | 排序缓冲 + Gather 缓冲 | PR4778 |
| DequantSwigluQuant | 256KB | UB_REVERSE | ~216KB | 反量化 + SwiGLU + 量化多缓冲 | PR4778 |
| KvRmsnormRopeCache | 256KB | UB_RESERVED | ~216KB | RMSNorm + RoPE + KVCache 多缓冲 | PR4778 |
| GroupNormSwish | 256KB | 0 | 256KB | GroupNorm 均值/方差 + Swish 缓冲 | PR4778 |
| SwigluQuant | 256KB | 0 | 256KB | SwiGLU + 量化多缓冲 | PR4778 |
| InterleaveRope | 256KB | 0 | 256KB | RoPE 交错乘加缓冲 | PR4778 |
| 逐元素 GELU | 256KB | 0 | 256KB | depth=4 VECIN ×2 + VECOUT ×2 | a5_ops |
| LayerNorm F32 | 256KB | 0 | 256KB | 3-phase 峰值 ~48-72KB | a5_ops |
| AdamW F32 | 256KB | 0 | 256KB | 4×VECIN + 3×VECOUT + 2×VECCALC | a5_ops |
| SparseGather F32 | 256KB | 0 | 256KB | 3×buffer × hdim×4，hdim≤21845 | a5_ops |

> **Critical check**: DequantSwigluQuant 在 H=4992 时，UB 用量 **194KB/192KB** 溢出。解决路径：TQueBind depth=1 (P-P66)、缩小 tile、或 fp16 中间结果。

---

## 五、UB 溢出排查清单

### 5.1 症状→根因速查

| 症状 | 最可能根因 | 排查方向 |
|------|-----------|---------|
| 507035 (illegal instruction) | UB 越界访问 | Step 1 |
| error 340 (UB address not aligned) | 非 32B 对齐的 Duplicate/Add 偏移 | Step 2 |
| MTE write address out of range (263) | TQue 分配总量超过 UB | Step 1 |
| 静默精度污染（无 crash） | DataCopy 32B 对齐溢出写入相邻 tensor | Step 3 |
| 编译期 `static assertion failed: sizeof(UbLayout) <= 248 * 1024: UB buffer too large` | 自研 `struct UbLayout` 打包多块 UB buffer 超 3510 编译期上限 **248KB**（FA-score 实测；struct 打包 + static_assert 守门形态） | Step 1 按 248KB 复核总量 → `GetWithOffset` 多阶段复用压峰值 / 缩 tile；禁止删断言绕过（详见 `cube-migration-guide.md` 低阶直跑编译错误速查） |

### 5.2 排查步骤

**Step 1：手工算总 UB**

```
1. 列出所有 InitBuffer 调用 → 写出每个 buffer 的大小和份数
2. TQue: 大小 × depth（InitBuffer 的第二个参数）
3. TBuf: 大小 × 1
4. 加总
5. 是否 < 可用 UB (256KB 或 216KB SimtMode)？
   ├─ 是 → 不是总量问题，进入 Step 2
   └─ 否 → UB 溢出，需要缩小 tile 或减少 buffer
```

**Step 2：检查 UB 地址对齐**

```
1. 搜索所有 Duplicate(dst[offset], ...) 或 VEC 运算中带下标的写法
2. offset 值能否被 8 (fp32) / 16 (fp16) 整除？
   ├─ 否 → EC-26: 改用 Duplicate(dst, 0, totalCount) 全量归零
   └─ 是 → 不是对齐问题，进入 Step 3
```

**Step 3：检查 DataCopy 32B 越界**

```
1. 搜索所有 DataCopy(gm, ub, count) — 写回 GM 的路径
2. count < 8 (fp32) 或 count < 16 (fp16)？
   ├─ 是 → EC-41: Host 侧 pad buffer 到 ≥ 8/16 元素
   └─ 否 → 进入 Step 4
```

**Step 4：检查多 buffer 是否释放**

```
1. 多阶段算子：阶段间是否有 PipeBarrier 或 alloc tensor 被正确释放？
2. 同一物理 UB 区域是否被两个 TBuf 同时声明？（PB-17 风险）
3. 是否有未配对的 AllocTensor/FreeTensor？（Buffer 泄漏）
```

---

## 六、优化策略

### 6.1 缩小 UB 占用的方法

| 方法 | 效果 | 副作用 | 模式来源 |
|------|------|--------|---------|
| **减小 TILE** | 直接减小 UB | MTE2/VEC 重叠率下降 | 通用 |
| **TQueBind depth=1** | 省 1×TILE 空间 | 流水线串行化 | P-P66 |
| **fp16 中间结果** | elemSize 减半 (float→half) | 精度损失 | 通用 |
| **单 TBuf 多阶段复用** | 不同阶段共用 UB 区域 | 需要 PipeBarrier 隔离 | CAND-NSA-4 |
| **合并小 buffer** | 减少 InitBuffer 调用 | 需要手动偏移管理 | 通用 |
| **减少 queue depth (4→2)** | 省 2×TILE 空间 | MTE2/VEC 重叠下降 | OL-63 反模式 |

### 6.2 多阶段 TBuf 复用模式

```cpp
// 一个 TBuf 覆盖全部 UB，不同阶段用 GetWithOffset 切成不同类型
TBuf<TPosition::VECCALC> allUb;
pipe.InitBuffer(allUb, 192 * 1024);  // 192 KB

// Phase A: softmax
int64_t off = 0;
LocalTensor<float> scores  = allUb.GetWithOffset<float>(rows * cols, off);
off += rows * cols * sizeof(float);
LocalTensor<float> softOut = allUb.GetWithOffset<float>(rows * cols, off);
// ... compute ...
PipeBarrier<PIPE_V>();  // ← 阶段边界（保证 Phase A 写完）

// Phase B: 复用同一块 UB，重新切分
off = 0;
LocalTensor<half> auxIn = allUb.GetWithOffset<half>(rows2 * auxLen, off);
// ... compute ...
```

---

## 七、关键约束汇总

| 约束 | 值 | 来源 |
|------|-----|------|
| UB 总量 (A5) | 256 KB = 262144 B | DavidV100 手册 |
| SIMT DCache 预留 | 40 KB = 40960 B | PR 103 / l1-l2-guide |
| Host 获取 UB 接口 | `platform.GetCoreMemSize(CoreMemType::UB, ubSize)` | AscendC platform API（`platform_ascendc`） |
| InitBuffer 自动对齐 | len 不足 32B 自动补齐 | `cannbot-skills/ops/ascendc-api-best-practices/references/api-buffer.md` |
| Buffer 总数上限 | 64 | 硬件约束（A5） |
| DataCopyPad blockCount 上限 | 4095 | `cannbot-skills/ops/ascendc-api-best-practices/references/api-datacopy.md` |
| DataCopy GM 最小搬运 | 32 B = 8×fp32 | PB-11 / EC-41 |
| 32B 对齐要求 | fp32: %8==0, fp16/bf16: %16==0 | `cannbot-skills/ops/ascendc-api-best-practices/references/api-datacopy.md` |
