# IndexGather 范式 AscendC 能力映射

> **层级**：AscendC 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 AscendC 真值与 API 实体。

## 1 谓词真值表

| 谓词 | AscendC 真值 | API 实体 / 规则文档 |
|------|------|------|
| `SIMT_SUPPORT` | **true** | `__simt_vf__` + `asc_vf_call`；线程由 `dim3` 指定 |
| `VF_SERIAL_EXEC` | **true** | 多次 `asc_vf_call` 区域串行执行，跨区域访存天然有序 |
| `VF_START_COST` | 数百 cycle（固定启动成本） | 纯计算循环合入单个 VF；见 [kernel-entrance.md](kernel-entrance.md) |
| `GM_DIRECT_ACCESS` | **true** | `__gm__` 指针 + SIMT `ldg/stg`，经 DCache/L2；无需 UB 数据缓存 |
| `DCACHE_SIZE` | **32KB**（SIMT 侧） | 无查询接口；见 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §4 |
| `DCACHE_RESERVE_RANGE` | **32KB–128KB** | `DCACHE_MIN` / `DCACHE_MAX` + `SetLocalMemorySize`；见 [tiling-preprocess.md](tiling-preprocess.md) §5 |
| `UB_CAPACITY` | 运行期查询 | `GetCoreMemSize(CoreMemType::UB)`；返回值不含 DCache 预留，须另行扣除 |
| `SIMT_REGISTER_MODEL` | 寄存器总量固定；线程数翻倍→每线程寄存器减半；溢出到 GM 栈性能下降 | `__launch_bounds__(THREAD_NUM)`；见 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §5 |
| `SIMT_THREAD_CHOICES` | **256 / 512 / 1024 / 2048**（示例默认 1024） | `__launch_bounds__` + `dim3`；见 [generic/thread-layout.md](../../generic/thread-layout.md) §4 |
| `CORE_NUM` | 运行期查询 | `ascendcPlatform.GetCoreNumAiv()`；见 [tiling-preprocess.md](tiling-preprocess.md) §4 |
| `MAX_RANK` | **8**（示例值，按需求调整） | `CUSTOM_GATHER_MAX_RANK`；TilingData 的 `outStrides/inStrides` 数组长度 |
| `INDEX_DTYPE_SET` | **int32 / int64** | `DTYPE_INDEX` / `INDEX_T`；IDX32/IDX64 由元素计数决定，不只看 index dtype |
| `UINTDIV_MAGIC_SHIFT` | **true**；仅 scalar `__aicore__` | `AscendC::GetUintDivMagicAndShift<T>(magic, shift, divisor)`；host 禁止；出参须为 Local Memory 标量；T ∈ {uint32, uint64} |
| `UINTDIV_VF` | **true**；仅 `__simt_vf__` | `AscendC::Simt::UintDiv<T>(n, magic, shift)`；返回商，余数用 `n - q * divisor`；规则见 [common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §1 |
| `UINT32_MULHI_NATIVE` | **true** | `__umulhi` 单指令、单寄存器；IDX32 可编译期全展开 |
| `UINT64_MUL_HIGH_COST` | `__umul64hi` 多指令模拟，单次约 2.7×；RANK 5–8 全展开触发 spill | IDX64 保留运行期 rank 截断；见 [kernel-entrance.md](kernel-entrance.md) §3.2 |
| `BYTE_LEVEL_MOVE` | **true** | `BytePackType` / `PackT<uint8/16/32/64>` + `reinterpret_cast<__gm__ PackT*>`；任意 dtype bit-exact 搬移；见 [index.md](index.md) 强制规则 10 |

## 2 通用术语 → AscendC 实体

| 通用层术语 | AscendC 实体 |
|------|------|
| SIMT 执行域 | `__simt_vf__` 函数 + `asc_vf_call` |
| GM 直访 | `__gm__ T*` 指针；SIMT 线程内 `ldg/stg` 经 DCache/L2 |
| DCache 预留 | `SetLocalMemorySize(ubSize - dcache - staticUb)` |
| UB 容量 | `GetCoreMemSize(CoreMemType::UB)` |
| 向量核数 | `GetCoreNumAiv()` |
| 线程数档位 | `__launch_bounds__(THREAD_NUM)` + `dim3(THREAD_NUM)` |
| magic/shift 预计算 | `GetUintDivMagicAndShift<T>`（scalar 端） |
| VF 快除 | `Simt::UintDiv<T>`（VF 端） |
| 字节级搬移 | `PackT<uint8/16/32/64>` / `BytePackType<T>` |

## 3 关键语言级附加约束

1. `GetUintDivMagicAndShift` 只能在 kernel scalar 端预计算，禁止在 host/tiling 侧调用；出参禁止直接绑定 `__ubuf__` 左值。
2. `Simt::UintDiv` 只能在 `__simt_vf__` 内调用，scalar 端调用无效；余数由 `n - q * divisor` 组合。
3. `DCACHE_SIZE` 不能通过查询接口获得；`UB_CAPACITY` 查询值不包含 DCache 预留，预算时必须自行扣除。
4. 平台相关参数（核数、UB、线程档位）必须通过接口或平台配置获取，禁止在设计文档中写死为唯一真值。
