# ScatterUpdate 范式 AscendC 能力映射

> **层级**：AscendC 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 AscendC 真值与 API 实体。

## 1 谓词真值表

| 谓词 | AscendC 真值 | API 实体 / 规则文档 |
|------|------|------|
| `SIMT_SUPPORT` | **true** | `__simt_vf__` + `asc_vf_call`；线程由 `dim3` 指定；见 [kernel-entrance.md](kernel-entrance.md) |
| `VF_SERIAL_EXEC` | **true** | Phase1/2/3 的多次 `asc_vf_call` / 排序区域串行执行，跨区域访存天然有序；见 [kernel-entrance.md](kernel-entrance.md) §5 |
| `VF_START_COST` | 数百 cycle（固定启动成本） | Phase1 线性化合并到单 VF；见 [thread-layout.md](../../generic/thread-layout.md) §1.1 |
| `GM_DIRECT_ACCESS` | **true** | `__gm__` 指针 + SIMT `ldg/stg`，经 DCache/L2；nondet 直写与 Phase3 读 src 均无需 UB 中转 |
| `DCACHE_SIZE` | **32KB**（SIMT 侧） | 无查询接口；见 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §4 |
| `DCACHE_RESERVE_RANGE` | **32KB–128KB** | `DCACHE_MIN` / `DCACHE_MAX` + `SetLocalMemorySize`；见 [tiling-preprocess.md](tiling-preprocess.md) §6 |
| `UB_CAPACITY` | 运行期查询 | `GetCoreMemSize(CoreMemType::UB)`；返回值不含 DCache 预留，须另行扣除；SortLib 预算见 [tiling-preprocess.md](tiling-preprocess.md) §3 |
| `SIMT_REGISTER_MODEL` | 寄存器总量固定；线程数翻倍→每线程寄存器减半；溢出到 GM 栈性能下降 | `__launch_bounds__(THREAD_NUM)`；见 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §5 |
| `SIMT_THREAD_CHOICES` | **256 / 512 / 1024**（示例默认 1024；2048 需按平台/寄存器预算验证） | `__launch_bounds__` + `dim3`；见 [thread-layout.md](../../generic/thread-layout.md) §5 |
| `CORE_NUM` | 运行期查询 | `ascendcPlatform.GetCoreNumAiv()`；det 三阶段核数与 nondet 两步法见 [tiling-preprocess.md](tiling-preprocess.md) §4 |
| `MAX_RANK` | **8**（示例值，按需求调整） | TilingData 的 `dims/strides/srcStrides` 数组长度；dim/nd 归一化上限 |
| `INDEX_DTYPE_SET` | **int32 / int64** | `INDEX_T`；use64Bit 由元素计数决定，不只看 index dtype；见 [tiling-preprocess.md](tiling-preprocess.md) §2 |
| `UINTDIV_MAGIC_SHIFT` | **true**；仅 scalar `__aicore__` | `AscendC::GetUintDivMagicAndShift<T>(magic, shift, divisor)`；host 禁止；出参须为 Local Memory 标量；T ∈ {uint32, uint64} |
| `UINTDIV_VF` | **true**；仅 `__simt_vf__` | `AscendC::Simt::UintDiv<T>(n, magic, shift)`；返回商，余数用 `n - q * divisor`；规则见 [common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §1 |
| `UINT32_MULHI_NATIVE` | **true** | `__umulhi` 单指令、单寄存器；IDX32 可编译期全展开 |
| `UINT64_MUL_HIGH_COST` | `__umul64hi` 多指令模拟，单次约 2.7×；高迭代全展开触发 spill | IDX64 保留运行期循环边界，优先 useInt32Index 降级；见 [linearize.md](linearize.md) §6 与 [common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §3 |
| `BYTE_LEVEL_MOVE` | **true** | `BytePackType<T>` / `PackT<uint8/16/32/64>` + `reinterpret_cast<__gm__ PackT*>`；任意 dtype bit-exact 搬移，含 W==1 标量路径；见 [type-assign.md](type-assign.md) §1 |
| `L2_VOLATILE_RW` | **支持**；`__gm__ volatile` 写穿/读；DCache 无跨核失效协议 | Phase1 写 workspace 段2/段5，Phase3 读段2/段3/段4；见 [kernel-entrance.md](kernel-entrance.md) §5 与 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §3 |
| `CROSS_CORE_SYNC` | **true**；`SyncAll()` + `SetScheduleMode(1)` | Phase1 VF 后 `SetFlag<V_S> + WaitFlag<V_S> + SyncAll`；`SortInvoke` 后 `SyncAll`；超发核靠 `b >= active_cores` 参与屏障；见 [kernel-entrance.md](kernel-entrance.md) §5 与 [thread-layout.md](../../generic/thread-layout.md) §4 |
| `ALIGNED_SINGLE_STORE_ATOMIC` | **true**；对齐 1/2/4/8B 单条 store 原子，全 dtype 无白名单 | nondet 直写依据；不依赖 atomic RMW；见 [type-assign.md](type-assign.md) §1.1 |
| `STABLE_SORT_LIB` | **可用**；升序稳定排序，`activeCores = min(coreCount, tileCount)`，workspace/UB 契约由库定义 | Host `SortLib::SortTilingCompute`；Kernel `SortInvoke`（前后 `pipe.Reset()` ×2，排序后 `SyncAll()`）；见 [deterministic.md](deterministic.md) §2 与 [tiling-preprocess.md](tiling-preprocess.md) §3 |
| `UB_ALIGN_BYTES` | **32B** | `AlignAllocN`；TilingData/UB 参数缓冲按 32B 对齐，workspace 段按 128B 对齐；见 [kernel-entrance.md](kernel-entrance.md) §3 与 [tiling-preprocess.md](tiling-preprocess.md) §5 |
| `WORKSPACE_MODEL` | det 用户 ws 四段（SUBSET 追加段5），128B 对齐；nondet 用户 ws = 0；系统 ws 无条件分配 | `GetWorkspaceSizes(1)` 必须调用；布局见 [tiling-preprocess.md](tiling-preprocess.md) §5 |

## 2 通用术语 → AscendC 实体

| 通用层术语 | AscendC 实体 |
|------|------|
| SIMT 执行域 | `__simt_vf__` 函数 + `asc_vf_call` |
| GM 直访 | `__gm__ T*` 指针；SIMT 线程内 `ldg/stg` 经 DCache/L2 |
| DCache 预留 | `SetLocalMemorySize(ubSize - dcache - STATIC_UB_ESTIMATE)` |
| UB 容量 | `GetCoreMemSize(CoreMemType::UB)` |
| 向量核数 | `GetCoreNumAiv()` |
| 线程数档位 | `__launch_bounds__(THREAD_NUM)` + `dim3(THREAD_NUM)` |
| magic/shift 预计算 | `GetUintDivMagicAndShift<T>`（scalar 端） |
| VF 快除 | `Simt::UintDiv<T>`（VF 端） |
| 字节级搬移 | `PackT<uint8/16/32/64>` / `BytePackType<T>` |
| L2 volatile 写穿 | `__gm__ volatile` 写/读 |
| 跨核同步 | `SyncAll()` + `SetScheduleMode(1)` |
| 对齐单 store 原子 | 对齐 1/2/4/8B 的单条 store |
| 稳定排序库 | `SortLib::SortTilingCompute` / `SortInvoke`，`activeCores` |
| UB 对齐 | `AlignAllocN`（32B） |
| 用户 workspace | `GetWorkspaceSizes(1)` + det 四段/五段布局 |

## 3 关键语言级附加约束

1. `SortInvoke` 前后必须各执行一次 `pipe.Reset()`，排序后 `SyncAll()`，且 host 侧 `SetScheduleMode(1)`；缺少任一环节会破坏 UB 预算或跨核可见性。
2. Phase1 VF 写 workspace 后必须 `SetFlag<HardEvent::V_S>(0) + WaitFlag<HardEvent::V_S>(0) + SyncAll()` 三连；kernel 末尾 Phase3 VF 后无需 join。
3. 跨 VF 写读 GM workspace 必须使用 `__gm__ volatile` 写穿或显式 `SyncAll`；DCache 无跨核失效协议，VF 写 GM 不会自动清理他核 DCache。
4. nondet 路径只依赖对齐单条 store 原子防撕裂，不引入 atomic RMW，也不设 dtype 白名单；重复索引的胜者可以任意。
5. `GetWorkspaceSizes(1)` 必须无条件调用；det 用户 workspace 四段（SUBSET 五段）按 128B 对齐，nondet 用户 workspace 为 0。
6. SortLib UB 预算须扣除静态参数 `STATIC_UB_ESTIMATE`；`UB_ALIGN_BYTES` 为 32B，参数缓冲按此对齐，workspace 段按 128B 对齐。
7. det 路径 `blockDim >= sortActiveCores`；超发核通过 `b >= active_cores` 跳过排序计算并直接到达屏障，避免排序阶段凑不齐屏障。
8. `GetUintDivMagicAndShift` 只能在 kernel scalar 端预计算，禁止在 host/tiling 侧调用；出参禁止直接绑定 `__ubuf__` 左值。
9. `Simt::UintDiv` 只能在 `__simt_vf__` 内调用；余数由 `n - q * divisor` 组合。
10. `DCACHE_SIZE` 不能通过查询接口获得；`UB_CAPACITY` 查询值不包含 DCache 预留，预算时必须自行扣除。
11. 平台相关参数（核数、UB、线程档位、SortLib active_cores）必须通过接口或运行期参数获取，禁止在设计文档中写死为唯一真值。
