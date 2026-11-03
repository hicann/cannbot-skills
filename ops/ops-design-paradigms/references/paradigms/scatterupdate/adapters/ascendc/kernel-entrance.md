# ScatterUpdate Kernel 入口与三段式编排

> 入口注册 / deterministic 分派 / 双模板 host 结构 / 三段式编排（段尾覆盖）/ 同步纪律。设计文档 §4.1 的真值源。

---

## 1 入口结构（kernel 入口文件）

```cpp
template <uint32_t countMode, uint32_t keyDtype, uint32_t deterministic>
__global__ __aicore__ void xxx_scatter(GM_ADDR var, GM_ADDR index, GM_ADDR src,
                                       GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(XxxTilingData);   // 默认结构取较大者（det ≥ Nondet 覆盖安全）
    if constexpr (deterministic == XXX_MODE_NONDET) {
        // 非确定快速路径：单 VF 直接写覆盖（任意赢即覆盖语义正确——无原子、无 dtype 白名单）
        GET_TILING_DATA_WITH_STRUCT(XxxNondetTilingData, tilingData, tiling);
        NsXxx::NondetProcess<DTYPE_VAR, DTYPE_INDEX>(var, index, src, var_out, &tilingData);
    } else {
        // 确定三段式（assign 条件路径——排序段尾覆盖 last-write-wins）
        GET_TILING_DATA_WITH_STRUCT(XxxTilingData, tilingData, tiling);
        // countMode × keyDtype 的 if constexpr 分派 → Process<..., CountT, KeyT>
    }
}
```

- `DTYPE_VAR / DTYPE_INDEX` 宏由 def.cpp `DynamicCompileStaticFlag(true)` 激活，dtype 组合（由 spec 决定）编译期展开。
- 核函数签名参数顺序与 REG_OP 原型一致（var, index/indices, src/updates, var_out + 框架 workspace/tiling）。
- assign 型入口：张量源 `if constexpr (deterministic)` 二分（nondet 直写任意赢 / det 排序段尾覆盖）；标量源（值同）为 `dedup` 位二分（仅重复度驱动，无确定性门控）。

## 2 Host 侧双模板注册（assign 张量源，六文件结构）

```
op_host/arch35/
├─ xxx_tiling.h          CompileInfo + BaseTiling 声明
├─ xxx_tiling_base.cpp   Base：公共校验/shape 解析（纯只读成员填充，降级安全）
├─ xxx_det_tiling.h/.cpp     确定模板：三段式 tiling 主体 + priority=10 注册（IsCapable ≡ true 兜底）
├─ xxx_nondet_tiling.h/.cpp  非确定模板：IsCapable 门控 + priority=0 注册
└─ xxx_tiling.cpp        入口：Tiling4Xxx → TilingRegistry::DoTilingImpl
```

要点：
- `REGISTER_OPS_TILING_TEMPLATE(OpName, NondetTiling, 0)` / `(OpName, DetTiling, 10)`——priority 越小越优先。
- **Base 的解析必须纯只读**：IsCapable 拒绝后 registry 以下一模板重走完整流程，有副作用会状态污染。
- **门控语义**：assign 张量源 `IsCapable = (GetDeterministic() == 0) && (dupRatio ≤ THRESHOLD)`（**无 dtype 白名单**——对齐单条 store 原子防撕裂全 dtype 安全）——严格 `==0`；异常返回 INT32_MAX 自然落入 det 兜底。
- assign 标量源（值同）单模板 + TilingKey `dedup` 场景位（门控仅 `dupRatio`，无注册分叉——值同天然 bit 确定，确定性开关无意义）。

## 3 Kernel 侧三文件结构（一次到位，勿"先追加后拆分"）

```
op_kernel/arch35/
├─ xxx_common.h        共享：THREAD_NUM / AlignAllocN(UB 32B 对齐) / PackType 字节视角工具
│                      （两路径单一事实源，防 W 计算漂移）
├─ xxx_det_simt.h      确定路径：Phase1 线性化 / Phase2 SortLib / Phase3 段尾覆盖
└─ xxx_nondet_simt.h   非确定路径：单 VF 直写覆盖（无原子，无需 device_atomic_functions.h）
```

## 4 三段式编排（det 路程主体，Process 内）

```cpp
// 0. 空 tensor 早退（先于快除预计算与排序，linearize.md §5）
if (td->totalIndexNum == 0 || td->totalVarElems == 0) { return; }

// 1. UB 参数准备（dims/strides/magic/shift 经 LocalMemAllocator + DataSyncBarrier<UB>）

// 2. Phase1 线性化 VF → 写 workspace 段2
asc_vf_call<Phase1LinearIdxKernel<...>>(dim3(THREAD_NUM), ...);
SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0); SyncAll();

// 3. Phase2 SortLib 稳定排序（段2 → 段3/段4）
pipe.Reset();
SortLib::SortInvoke<KeyT, int32_t, CountT, false>(&pipe, linearIdxBuf, sortedBuf, permBuf, usrWs, params);
pipe.Reset();
SyncAll();

// 4. Phase3 段尾覆盖 VF（段3/段4 + src/var → var_out，last-write-wins）
asc_vf_call<Phase3XxxKernel<...>>(dim3(THREAD_NUM), ...);
SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
```

## 5 同步纪律（实测验证，易错高发区）

| 位置 | 要求 | 原因 |
|------|------|------|
| Phase1 VF 之后 | `SetFlag<V_S>(0); WaitFlag<V_S>(0); SyncAll();` 三连 | V_S join 本核向量队列后，SyncAll 才能保证他核过屏障消费时本核 VF 已写完（缺 V_S 会跨核数据竞争） |
| **kernel 末尾 VF 之后** | **无需任何 join** | 末尾 VF 后无 S 消费；AIV task 结束协议排空向量管线（官方先例 + 实测验证） |
| scalar 写 UB 后 | `DataSyncBarrier<MemDsbT::UB>()` 再 VF_CALL | UB 参数可见性 |
| SortInvoke 前后 | `pipe.Reset()` ×2 + 排序后 `SyncAll()` | 排序 UB 预算完整 + 跨核可见 |
| 循环内 VF_CALL 后 | `SetFlag<V_S>` + `WaitFlag<V_S>` | 下一轮 scalar 写 UB 与在飞 VF 竞争 |
| 跨 VF 写读 workspace | `__gm__ volatile` 写穿或显式 SyncAll | DCache 无跨核失效协议（[common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §3）——VF 写 GM 不清 dcache，后续读须写穿 |

## 6 空 tensor 与函数式拷贝

- **inplace 语义**（var_out == var 同名）：空 tensor 直接 return，非更新区域天然保留。
- **函数式语义**（y 独立输出）：Phase3 前置 Stage A——全核 Grid-Stride `output[pos] = var[pos]` 拷贝（等价 `y = var.clone()`），`asc_syncthreads()` 分隔保证 Stage B 读到已初始化的 output。
