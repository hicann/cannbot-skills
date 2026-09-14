# 多阶段算子迁移指南

> **适用场景**: 单个 kernel 需要 2+ 个算法阶段（如 LayerNorm 的 mean→var→norm，或 FusedAttention 的 softmax→scoring→TopK），阶段间需要共享 UB、传递 workspace
> **核心模式来源**: 插件 KB 模式库。文中 `P-P*` / `OL-*` 代号见 `plugins-community/ascendc-port-orchestrator/kb/target/ascendc/patterns/PATTERN_INDEX.md`（按代号路由到 domains/ 详情文件）；`CAND-*` 见同目录 `unverified/candidates.md`。
> **何时加载**: 算子有 ≥2 个顺序阶段，且阶段间 peak UB 需求不同

---

## 一、模式决策树

```
算子有几个阶段？
├─ 2 阶段，阶段内无流水需求 → 2.1 TQueBind depth=1 原地复用
├─ 2 阶段，每阶段内需要流水 → 2.2 独立 TQue 双缓冲 + Ping-Pong
├─ 3+ 阶段，peak UB 接近上限 → 2.3 单 TBuf + GetWithOffset 多阶段复用
├─ 3+ 阶段，跨 core 传递中间结果 → 2.4 GM workspace slot rotation (本文档仅概述)
└─ 需要保持中间结果跨核可见 → 2.4 CAND-FA3 (Cube 类，Vector 类一般不需要)
```

---

## 二、四种核心模式

### 2.1 TQueBind depth=1 原地复用（2 阶段，UB 紧张）

**来源**: P-P66, `memory_access.md`
**适用**: 输入→Compute(原地修改)→输出，UB 预算紧张时省 1×TILE 空间
**副作用**: CopyIn/Compute/CopyOut 串行化，无流水重叠

```cpp
#include "kernel_operator.h"
using namespace AscendC;

class TwoStageInplace {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t N, TPipe* p) {
        pipe_ = p; N_ = N;
        xGm_.SetGlobalBuffer((__gm__ float*)x, N);
        yGm_.SetGlobalBuffer((__gm__ float*)y, N);
        // TQueBind: VECIN 和 VECOUT 共用同一块物理 UB
        pipe_->InitBuffer(bufBind_, 1, TILE * sizeof(float));  // depth=1, 只用 1×TILE
    }

    __aicore__ inline void Process() {
        for (int64_t off = 0; off < N_; off += TILE) {
            int32_t cur = (off + TILE > N_) ? (int32_t)(N_ - off) : TILE;

            // Step 1: CopyIn (GM→UB)
            LocalTensor<float> local = bufBind_.AllocTensor<float>();
            DataCopy(local, xGm_[off], cur);
            bufBind_.EnQue(local);

            // Step 2: Compute (原位修改 — 不分配新 buffer)
            local = bufBind_.DeQue<float>();
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<float> rIn, rOut;
                AscendC::Reg::MaskReg m = AscendC::Reg::UpdateMask<float>(cur);
                AscendC::Reg::LoadAlign(rIn, (__ubuf__ float*)local.GetPhyAddr());
                AscendC::Reg::Abs(rOut, rIn, m);       // Phase A
                AscendC::Reg::Exp(rOut, rOut, m);       // Phase B (原地覆盖)
                AscendC::Reg::StoreAlign(
                    (__ubuf__ float*)local.GetPhyAddr(), rOut, m);
            }
            bufBind_.EnQue(local);

            // Step 3: CopyOut (UB→GM)
            local = bufBind_.DeQue<float>();
            DataCopy(yGm_[off], local, cur);
        }
    }

private:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, yGm_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> bufBind_;
    int64_t N_;
    static constexpr int32_t TILE = 2048;
};
```

**UB 公式**: `1 × TILE × sizeof(T)` — 比独立 TQue 的 `4 × TILE × sizeof(T)` 省 75%。

**适用判断**:
| 条件 | 结论 |
|------|------|
| UB 预算紧张（如 DequantSwigluQuant 194/192KB） | ✅ 用 TQueBind |
| 算子是 memory-bound，带宽已饱和 | ✅ 流水重叠无收益 |
| 算子是 VEC-bound，需要深度流水 | ❌ 用独立 TQue depth=4 |

---

### 2.2 独立 TQue 双缓冲 + Ping-Pong（2 阶段，需流水）

**适用**: 每阶段独立访问 GM，阶段间需要 MTE2/VEC 重叠
**效果**: MTE2(iter N+1) ∥ VEC(iter N) ∥ MTE3(iter N-1)，三级流水

```cpp
class TwoStagePipelined {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t N, TPipe* p) {
        pipe_ = p; N_ = N;
        xGm_.SetGlobalBuffer((__gm__ float*)x, N);
        yGm_.SetGlobalBuffer((__gm__ float*)y, N);
        // depth=4 让 MTE2 最多预取到 3 个 buffer 前面（OL-63）
        pipe_->InitBuffer(inQ_, 4, TILE * sizeof(float));
        pipe_->InitBuffer(outQ_, 2, TILE * sizeof(float));
        // 两阶段独立的临时 buffer
        pipe_->InitBuffer(stage1Buf_, TILE * sizeof(float));
        pipe_->InitBuffer(stage2Buf_, TILE * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loops = (N_ + TILE - 1) / TILE;
        for (int32_t i = 0; i < loops; i++) {
            int32_t offset = i * TILE;
            int32_t cur = (offset + TILE > N_) ? (int32_t)(N_ - offset) : TILE;

            // GM→UB (MTE2 — 可与下一 iter Compute 重叠)
            LocalTensor<float> in = inQ_.DeQue<float>();
            DataCopy(in, xGm_[offset], cur);
            inQ_.EnQue(in);

            // Stage 1: 计算 (VEC)
            LocalTensor<float> stg1 = stage1Buf_.Get<float>();
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<float> rIn, rTmp;
                AscendC::Reg::MaskReg m = AscendC::Reg::UpdateMask<float>(cur);
                AscendC::Reg::LoadAlign(rIn,
                    (__ubuf__ float*)in.GetPhyAddr());
                AscendC::Reg::Exp(rTmp, rIn, m);            // Phase A
                AscendC::Reg::StoreAlign(
                    (__ubuf__ float*)stg1.GetPhyAddr(), rTmp, m);
            }
            // 阶段边界：PipeBarrier 确保 VEC 写完 stg1
            PipeBarrier<PIPE_V>();

            // Stage 2: 计算 (VEC — 复用 stg1 输入)
            LocalTensor<float> stg2 = stage2Buf_.Get<float>();
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<float> rStg1, rOut;
                AscendC::Reg::MaskReg m = AscendC::Reg::UpdateMask<float>(cur);
                AscendC::Reg::LoadAlign(rStg1,
                    (__ubuf__ float*)stg1.GetPhyAddr());
                AscendC::Reg::Muls(rOut, rStg1, 2.0f, m);  // Phase B
                AscendC::Reg::StoreAlign(
                    (__ubuf__ float*)stg2.GetPhyAddr(), rOut, m);
            }
            inQ_.FreeTensor(in);

            // UB→GM (MTE3 — 可与下一 iter Compute 重叠)
            LocalTensor<float> out = outQ_.AllocTensor<float>();
            Adds(out, stg2, 0.0f, cur);  // copy via VEC (规避 PB-9)
            outQ_.EnQue(out);
            LocalTensor<float> res = outQ_.DeQue<float>();
            DataCopy(yGm_[offset], res, cur);
            outQ_.FreeTensor(res);
        }
    }

private:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, yGm_;
    TQue<TPosition::VECIN, 4> inQ_;
    TQue<TPosition::VECOUT, 2> outQ_;
    TBuf<TPosition::VECCALC> stage1Buf_, stage2Buf_;
    int64_t N_;
    static constexpr int32_t TILE = 4096;
};
```

**UB 公式**: `4 × TILE × es + 2 × TILE × es + 2 × TILE × es = 8 × TILE × es`

**阶段边界策略**: `PipeBarrier<PIPE_V>()` 仅在阶段之间使用。阶段内的 MTE2/VEC 重叠由 TQue EnQue/DeQue 自动管理。

**注意**: 如果任一阶段涉及 MTE 写入（如 DataCopy 到 GM 中间结果），阶段边界必须改用 `SetFlag<V_MTE3> + WaitFlag<V_MTE3>`（参考 OL-94 决策表）。

---

### 2.3 单 TBuf + GetWithOffset 多阶段复用（3+ 阶段，UB 紧张）

**来源**: CAND-NSA-4（原始素材为开发期内部笔记，不在本 skill 内）
**适用**: ≥3 个顺序阶段，每个阶段的 peak UB tensor 集合互不重叠，总 peak < 各阶段独立 TBuf 之和
**核心思想**: 一个 TBuf 覆盖全部可用 UB，每个阶段入口用 `GetWithOffset` 重新切分

```cpp
class MultiStageSharedUB {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 int64_t rows, int64_t cols, TPipe* p) {
        pipe_ = p; rows_ = rows; cols_ = (int32_t)cols;
        xGm_.SetGlobalBuffer((__gm__ float*)x, rows * cols);
        yGm_.SetGlobalBuffer((__gm__ float*)y, rows * cols);
        pipe_->InitBuffer(inQ_, 4, TILE * sizeof(float));
        pipe_->InitBuffer(outQ_, 2, TILE * sizeof(float));

        // ★ 一个 TBuf 覆盖全部可用 UB
        pipe_->InitBuffer(allUb_, 192 * 1024);  // 192 KB (预留 SIMT 后)
    }

    __aicore__ inline void Process() {
        for (int32_t r = 0; r < rows_; r++) {
            // ==========================================
            // Phase A: 加载 + 第一次归约 (输出 mean)
            // ==========================================
            int64_t off = 0;
            LocalTensor<float> rowData = allUb_.GetWithOffset<float>(
                cols_, off);
            off += cols_ * sizeof(float);
            LocalTensor<float> meanBuf = allUb_.GetWithOffset<float>(
                8, off);  // 归约结果占 32B (8×fp32)
            off += AlignUp(8 * sizeof(float), 32);
            // off 必须 32B 对齐后才能继续 GetWithOffset (CAND-NSA-4 约束)
            off = AlignUp(off, 32);

            // GM→UB 加载整行
            DataCopy(rowData, xGm_[r * cols_], cols_);
            // RegBase 归约到 meanBuf[0]
            {
                __VEC_SCOPE__ {
                    AscendC::Reg::RegTensor<float> rData, rSum;
                    AscendC::Reg::MaskReg m =
                        AscendC::Reg::UpdateMask<float>(cols_);
                    AscendC::Reg::LoadAlign(rData,
                        (__ubuf__ float*)rowData.GetPhyAddr());
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(
                        rSum, rData, m);
                    AscendC::Reg::StoreAlign(
                        (__ubuf__ float*)meanBuf.GetPhyAddr(), rSum, m);
                }
            }
            float mean = meanBuf.GetValue(0) / cols_;
            PipeBarrier<PIPE_V>();  // ★ 阶段边界

            // ==========================================
            // Phase B: 复用 UB，计算方差
            // ==========================================
            off = 0;  // ★ 重新从 0 开始切分
            LocalTensor<float> diffBuf = allUb_.GetWithOffset<float>(
                cols_, off);
            off += cols_ * sizeof(float);
            LocalTensor<float> sqBuf = allUb_.GetWithOffset<float>(
                cols_, off);
            off += cols_ * sizeof(float);
            LocalTensor<float> varBuf = allUb_.GetWithOffset<float>(
                8, off);

            // x - mean → diffBuf
            {
                __VEC_SCOPE__ {
                    AscendC::Reg::RegTensor<float> rRow, rDiff;
                    AscendC::Reg::MaskReg m =
                        AscendC::Reg::UpdateMask<float>(cols_);
                    AscendC::Reg::LoadAlign(rRow,
                        (__ubuf__ float*)rowData.GetPhyAddr());
                    AscendC::Reg::Subs(rDiff, rRow, mean, m);
                    AscendC::Reg::StoreAlign(
                        (__ubuf__ float*)diffBuf.GetPhyAddr(), rDiff, m);
                }
            }
            // diff² → sqBuf
            {
                __VEC_SCOPE__ {
                    AscendC::Reg::RegTensor<float> rDiff, rSq;
                    AscendC::Reg::MaskReg m =
                        AscendC::Reg::UpdateMask<float>(cols_);
                    AscendC::Reg::LoadAlign(rDiff,
                        (__ubuf__ float*)diffBuf.GetPhyAddr());
                    AscendC::Reg::Mul(rSq, rDiff, rDiff, m);
                    AscendC::Reg::StoreAlign(
                        (__ubuf__ float*)sqBuf.GetPhyAddr(), rSq, m);
                }
            }
            // ReduceSum → var
            {
                __VEC_SCOPE__ {
                    AscendC::Reg::RegTensor<float> rSq, rSum;
                    AscendC::Reg::MaskReg m =
                        AscendC::Reg::UpdateMask<float>(cols_);
                    AscendC::Reg::LoadAlign(rSq,
                        (__ubuf__ float*)sqBuf.GetPhyAddr());
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(
                        rSum, rSq, m);
                    AscendC::Reg::StoreAlign(
                        (__ubuf__ float*)varBuf.GetPhyAddr(), rSum, m);
                }
            }
            float var = varBuf.GetValue(0) / cols_;
            float invStd = 1.0f / sqrtf(var + 1e-5f);
            PipeBarrier<PIPE_V>();  // ★ 阶段边界

            // ==========================================
            // Phase C: 复用 UB，归一化写出
            // ==========================================
            off = 0;
            LocalTensor<float> outBuf = allUb_.GetWithOffset<float>(
                cols_, off);

            // normalize: (x - mean) / std
            {
                __VEC_SCOPE__ {
                    AscendC::Reg::RegTensor<float> rRow, rOut;
                    AscendC::Reg::MaskReg m =
                        AscendC::Reg::UpdateMask<float>(cols_);
                    AscendC::Reg::LoadAlign(rRow,
                        (__ubuf__ float*)rowData.GetPhyAddr());
                    AscendC::Reg::Subs(rOut, rRow, mean, m);
                    AscendC::Reg::Muls(rOut, rOut, invStd, m);
                    AscendC::Reg::StoreAlign(
                        (__ubuf__ float*)outBuf.GetPhyAddr(), rOut, m);
                }
            }
            // 写出到 GM
            DataCopy(yGm_[r * cols_], outBuf, cols_);
            PipeBarrier<PIPE_V>();  // ★ 行结尾同步（下一行可复用 UB）
        }
    }

private:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, yGm_;
    TQue<TPosition::VECIN, 4> inQ_;
    TQue<TPosition::VECOUT, 2> outQ_;
    TBuf<TPosition::VECCALC> allUb_;      // ★ 唯一 TBuf
    int64_t rows_;
    int32_t cols_;
    static constexpr int32_t TILE = 4096;
};
```

**多阶段复用关键规则（CAND-NSA-4）**:

| 规则 | 说明 | 违反后果 |
|------|------|---------|
| 每阶段入口 `off=0` 重新切分 | 通过 `GetWithOffset<NewT>(count, 0)` 获取新 view | 用旧 view 为 use-after-free 类风险 |
| `off = AlignUp(off, 32)` 后才继续 | 32B 对齐是 UB 访问硬约束 | 非对齐访问 → error 340 或静默数据错误 |
| 阶段间 `PipeBarrier<PIPE_V>()` | 保证上一阶段的 VEC 写入完成 | 读后写竞态 → 非确定性精度 |
| **禁止**跨阶段复用旧 view | `diffBuf` 不能在 Phase C 中使用 | 静默错误数据（无编译错误！） |
| **禁止** `ReinterpretCast` 跨阶段 | 必须用 `GetWithOffset<NewT>` 重建 view | view 指向被覆盖的内存 |
| 若阶段间有 MTE 写入 | 改用 `SetFlag<V_MTE3>+WaitFlag<V_MTE3>` | `PipeBarrier<PIPE_V>` 不保证 MTE 完成 |

**适用判断**:
| 条件 | 结论 |
|------|------|
| 阶段 tensor 互不重叠（max(peakA, peakB, peakC) < sum） | ✅ 用此模式 |
| 某阶段需要 TQue 深度流水（流式输入加载重叠计算） | ❌ 不要替换 TQue 为 TBuf re-slice |
| 多个阶段时间上重叠（MTE 尾 + VEC 头并发） | ❌ 用独立 TQue + 独立 TBuf |

---

### 2.4 GM workspace slot rotation（跨 core 管道 — 本文档不展开）

**来源**: CAND-FA3, `candidates.md`（插件 KB `kb/target/ascendc/patterns/unverified/`）
**适用**: 多个 core 之间通过 GM workspace 传递中间结果，需要 module-(MAX_LAG+1) slot 轮转
**使用条件**: MAX_LAG 可预先确定、每 slot 只有单一 producer、consumer 通过 CAND-FA1 跨 core flag 门控
**本文档范围外**，详见 `candidates.md` 的 CAND-FA1（跨核 flag 门控）+ CAND-FA3；跨核 workspace 轮转的落地方案另见 `cube-migration-guide.md` 改动 6（跨核同步）与改动 8（workspace 规划）。

---

## 三、阶段边界同步选择表（OL-94 增强版）

| 场景 | 正确原语 | 错误原语 | 错误后果 |
|------|---------|---------|---------|
| V→V 阶段边界（纯寄存器计算） | `PipeBarrier<PIPE_V>()` | 无 | 竞态 |
| VEC 写 TBuf → 下一阶段 VEC 读 TBuf | `PipeBarrier<PIPE_V>()` | 无 | 脏读 |
| VEC 写 UB → MTE3 搬出到 GM | `SetFlag<V_MTE3>+WaitFlag` | `PipeBarrier<PIPE_V>` | **MTE3 读到旧数据**（OL-94） |
| MTE2 搬入 → VEC 计算 | `SetFlag<MTE2_V>+WaitFlag` 或 TQue EnQue/DeQue | `PipeBarrier<PIPE_ALL>` | 流水效率严重下降 |
| 跨 core GM workspace slot 发布 | CAND-FA1 的 `SetFlag<cross-core>` | `PipeBarrier` | 跨 core 不可见 |

---

## 四、多阶段算子 UB 预算计算

多阶段算子的关键特点是**取各阶段峰值，而非累加**：

```
总 UB = max(阶段A峰值, 阶段B峰值, ...)          # 各阶段 Peak 互不重叠时
使用单 TBuf 复用（2.3）时：
总 UB = max(GetWithOffset 最大偏移 + 最后一阶段峰值)
```

各阶段峰值公式、UB 分配表模板、tileLength 反推与溢出排查见 `ub-budget-guide.md`（UB 预算的单一权威，含「2.4 LayerNorm 三阶段」峰值算例）；不同阶段可用不同 TILE，在阶段入口用 `GetWithOffset` 重新计算布局。

## 五、迁移检查清单（多阶段算子专属）

- [ ] **阶段识别**: 列出所有算法阶段和每阶段的输入/输出 tensor
- [ ] **Peak UB 估算**: 每阶段独立算 UB 需求，取 `max(peakA, peakB, ...)`
- [ ] **模式选择**: 按「一、模式决策树」选定最合适的多阶段模式
- [ ] **边界同步**: 按「三、阶段边界同步选择表」为每对相邻阶段选定正确的同步原语
- [ ] **对齐**: `GetWithOffset` 后 `off = AlignUp(off, 32)`（CAND-NSA-4 约束）
- [ ] **View 隔离**: 每个阶段用自己的 `GetWithOffset<NewT>` 重建 view，不复用旧 view
- [ ] **PipeBarrier 在 V→V 边界**: 纯 VEC 阶段间用 `PipeBarrier<PIPE_V>()`
- [ ] **SetFlag 在 V→MTE 边界**: 如果阶段间有 MTE 写入/读出，用 `SetFlag+WaitFlag`
- [ ] **TQue depth**: 需要流水的阶段设 depth≥2（逐元素推荐 4）
- [ ] **精度验证**: 至少 30 条用例，特别关注跨阶段的数据完整性
