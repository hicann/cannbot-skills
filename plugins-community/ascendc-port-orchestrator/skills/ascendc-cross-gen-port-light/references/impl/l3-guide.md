# L3：SIMT 优化（多线程替代 Scatter/Gather）

> **本指南只讲 L3 的迁移动作**。SIMT 的语法与 API 用法（`__simt_vf__` / `LAUNCH_BOUND` / `Simt::VF_CALL` / 线程索引循环 / 线程数取值 / `AtomicAdd` / `UintDiv` / `__local_mem__`）见 SIMT 最佳实践技能 `cannbot-skills/ops/ascendc-simt-best-practices/`，此处不重复维护。

## 触发条件（全部满足）

1. **包含 Scatter/Gather 操作**：按索引读写 GM，传统模式需 GM→UB→计算→UB→GM 四步
2. **索引逻辑简单**：无需复杂计算，仅需按索引搬移数据
3. **无需 UB 中转**：SIMT 可直接访问 GM
4. **线程并行度高**：数据量足够大，能充分利用 2048 个 SIMT 线程

## 为什么值得改（与 Memory-based 对照）

| 维度 | Memory-based | SIMT |
|------|-------------|------|
| 数据搬运 | GM→UB→计算→UB→GM（4 步） | GM 直读直写（1 步） |
| 流水线 / 同步 | TPipe + TQue；SetFlag/WaitFlag 事件同步 | 无流水线；线程级并行，无需同步 |
| UB 占用 | copyInQueue + copyOutQueue + assistBuffer | **零 UB 占用**（纯 GM 操作时） |
| 适用场景 | 复杂计算、需要 UB 中转 | 简单索引操作（Scatter/Gather） |
| 线程数 | 1（单线程 SIMD） | 最多 2048 |

代表实现对照（MoeInitRouting 的 SrcToDst 段）：Memory-based `MoeSrcToDstOp` ≈160 行 → SIMT `MoeSrcToDstSimtOp` ≈86 行，Scatter/Gather 场景实测提升 50–200%。

## L3 改造动作

### 1. `_apt.cpp`：加 SIMT 分支（编译时宏切换）

```cpp
// _apt.cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_src_to_dst_simt_op.h"  // SIMT 版本
#endif
#include "arch35/moe_src_to_dst_op.h"        // Memory-based 版本

extern "C" __global__ __aicore__ void moe_init_routing(...) {
    // ... 排序部分（两种架构共用）...

    // SrcToDst 段：950 走 SIMT，910b 走 Memory-based
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
#endif

    // ... 后续 Gather 部分（两种架构共用）...
}
```

要点：
- SIMT include 与调用用 `__NPU_ARCH__ == 3510` 保护；Memory-based include 不需要保护
- SIMT 版本**不需要 `TPipe`**，Memory-based 版本需要
- 一个算子内可**混合**：MoeInitRouting 排序段用 Memory-based、SrcToDst 段用 SIMT
- 同文件内多处需要同一分支时，定义统一接口宏避免重复条件编译：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    #define ASCEND_SIMT_AVAILABLE 1
#else
    #define ASCEND_SIMT_AVAILABLE 0
#endif
// 使用处： #if ASCEND_SIMT_AVAILABLE ... #else ... #endif
```

### 2. opDef 侧（L1 已完成，无需额外动作）

L3 不改变 host 侧的入口切换方式——`opFile.value = "<op>_apt"` 与 `config/ascend950/` 属 L1 范围，见 `l1-guide.md`；三种切换机制（编译时宏 / opFile / Tiling 运行时 `IsRegbaseSocVersion()`）的汇总表见 `l1-guide.md`「模式切换机制汇总」。

### 3. Tiling 侧：SIMT DCache 复用 UB，必须预留 40KB

**这是 L3 引入的新约束**：950 的 SIMT DCache 复用 UB 空间，Tiling 中未预留会导致 UB 越界（SIMT DCache 与算子 UB 使用冲突）。

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;   // 40KB

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

> 模板与各算子合规状态见 `ub-budget-guide.md`「SIMT DCache 预留」（UB 预算主题的唯一权威）；L1 升级信号中的相关条目见 `l1-guide.md`。

## L3 特有的坑

| 坑 | 处置 |
|---|---|
| 线程数写死 2048，数据量小时空转 | `threadNum = min(2048, coreRows)`；FPGA 环境用 256 |
| 多线程写同一地址 → 数据竞争 | 用 `Simt::AtomicAdd`（dtype 注意事项见 simt 技能 `intra-core-shared.md`） |
| `__local_mem__` 写 UB 时线程间地址冲突 | 保证每线程写不同地址；需回 UB 供后续 Memory-based 流水消费时才用 `__local_mem__` |
| 只改 kernel 不改 Tiling → UB 越界 | 按上文动作 3 预留 40KB（**L3 必做**） |

## SIMT 迁移检查清单

- [ ] SIMT 核函数已标记 `__simt_vf__` + `LAUNCH_BOUND`（语法见 simt 技能）
- [ ] `Simt::VF_CALL` 启动参数正确（`Simt::Dim3{threadNum,1,1}` + 函数参数）
- [ ] SIMT 函数内只用 `__gm__` 指针与标量（不得用 `LocalTensor` / `RegTensor`）
- [ ] 线程数按数据量动态取值（≤ 2048）
- [ ] SIMT include 与调用已用 `__NPU_ARCH__ == 3510` 保护
- [ ] Memory-based 路径未被破坏（双平台编译验证）
- [ ] 多线程写同一地址处已用 `Simt::AtomicAdd`
- [ ] **Tiling 已预留 SIMT DCache 40KB**
- [ ] `__local_mem__` 写 UB 处无线程间地址冲突

## 代表算子（迁移参考）

| 算子 | SIMT 用途 | 特点 |
|------|----------|------|
| MoeInitRouting | SrcToDst Scatter | 最简示例（全局函数写法） |
| MoeInitRoutingV3 | ExpertTokensCount + RowIdxGather | 用 `__local_mem__` 访问 UB |
| MoeInitRoutingV2 | ExpertTokenOut + SrcToDst | 多个 SIMT 函数组合 |
| MoeInplaceIndexAdd | 原子加 | `Simt::AtomicAdd` + 类静态成员函数写法 |
| GatherV2SimtTwoDim | 二维 Gather | `Simt::UintDiv` + 模板类静态函数 |
| MoeMaskedScatter | Scatter | SIMT 替代传统 Scatter |

## 完整实现示例（SIMT 版类结构）

以 MoeInitRouting 的 SrcToDst 段为例——**迁移时把原 Memory-based Op 类改写成 SIMT 版**，调用侧与 Memory-based 保持一致（Init / Process），差异只在内层：

```cpp
#define THREAD_NUM 2048

class MoeSrcToDstSimtOp {
public:
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow,
        const MoeInitRoutingTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __gm__ int32_t *expandDstToSrcRowGm_;
    __gm__ int32_t *expandedRowIdxGm_;
    int64_t coreRows_;
    int32_t threadNum_;
    int32_t blockIdx_;
    int64_t perCoreRows_;
    const MoeInitRoutingTilingData *srcToDstTilingData_;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(
    int64_t coreRows, int64_t startIndex,
    __gm__ int32_t* expandDstToSrcRowGm,
    __gm__ int32_t* expandedRowIdxGm)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx());
         index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int32_t srcIndex = index + startIndex;
        int32_t dstIndex = expandDstToSrcRowGm[srcIndex];
        expandedRowIdxGm[dstIndex] = srcIndex;   // 按索引直写 GM
    }
}

__aicore__ inline void MoeSrcToDstSimtOp::Process()
{
    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        int32_t startIndex = this->blockIdx_ * this->perCoreRows_;
        Simt::VF_CALL<ComputeSimt>(
            Simt::Dim3{static_cast<uint32_t>(this->threadNum_), 1, 1},
            this->coreRows_, startIndex,
            expandDstToSrcRowGm_, expandedRowIdxGm_);
    }
}
```

要点：外层类与 Memory-based 版同构（多核按 `blockIdx_` 分片、`needCoreNum` 判空核）；内层换成 `__simt_vf__` 函数 + `Simt::VF_CALL` 启动，线程循环用 `GetThreadIdx/GetThreadNum` 步进。

## 相关参考文档路径

- **SIMT 语法 / API / 用法**：`cannbot-skills/ops/ascendc-simt-best-practices/`（overview / vf-declaration / vf-call / thread-stride-pattern / index-calculation / intra-core-shared / data-transfer / multi-core-sync）
- KB 快照 `simt/`：SIMT 编程模型、API 参考（位于 `kb/target/ascendc/migration/`，本方法论 2026-05 旧版存档）
- KB 快照 `migration/`：A2/A3 → A5 迁移方案、API 兼容性、架构差异
