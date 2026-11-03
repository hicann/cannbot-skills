# ScatterUpdate 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法中依赖硬件/语言/库能力的量，统一抽象为**能力谓词**。通用层文档只引用谓词名，不出现具体语言 API。提供专用适配层的语言必须在其 `capability-mapping.md` 中对全部谓词给出真值。
>
> 本表为 ScatterUpdate 范式独立维护的能力真值源，不与其他范式共用能力文件；同名谓词含义保持一致，但真值以本范式适配层为准。

## 1 使用规则

1. **通用层**：算法描述遇到能力依赖点时引用本文档中的谓词名，不写具体数值或 API 名。
2. **适配层**：`adapters/<lang>/capability-mapping.md` 逐项填写谓词真值与对应的 API 实体。
3. **可填性检验**：新语言适配层必须能对每个谓词给出真值；填不出真值说明通用层抽象有缺陷，应修订抽象而非绕过。
4. **冲突处理**：同一能力点若适配层真值与通用层假设冲突，以适配层真值为准，并在设计结论中标注影响（如回退 nondet 直写、减少线程数、禁用排序路径）。

## 2 谓词定义表

| 谓词 | 类型 | 含义 | 影响的通用层知识点 |
|------|------|------|------------------|
| `SIMT_SUPPORT` | bool | 目标芯片是否提供 SIMT 线程执行能力 | `patterns.md` 前置约束；整体 SIMT 路线 |
| `VF_SERIAL_EXEC` | bool | 多个 VF/调度区域是否串行执行，跨区域访存是否天然有序 | `thread-layout` 的三阶段边界；`deterministic` 的 Phase1/2/3 顺序假设 |
| `VF_START_COST` | desc | VF/调度区域固定启动成本的量级 | Phase1 线性化循环合并到单 VF；`thread-layout` |
| `GM_DIRECT_ACCESS` | bool | SIMT 线程是否可直接以 GM 指针访问全局内存（经 DCache/L2），无需先搬 UB | `type-assign` 的直写路径；`linearize` 的读索引/读 src；UB 参数中转范围 |
| `DCACHE_SIZE` | bytes | SIMT 侧 DCache 容量 | `tiling` 的 DCache 预留；SortLib UB 预算 |
| `DCACHE_RESERVE_RANGE` | range[bytes] | host 可配置的 DCache 预留范围 | `tiling-preprocess` 的 DCache 预留与下限兜底 |
| `UB_CAPACITY` | bytes | 可查询的片上 UB 容量（不含 DCache 预留） | SortLib UB 预算；静态参数/UB 缓冲预算；`tiling-preprocess` |
| `SIMT_REGISTER_MODEL` | desc | 线程数与每线程寄存器的关系，以及寄存器溢出行为 | `thread-layout` 的线程数选型；Phase3 perm/slice 寻址链 |
| `SIMT_THREAD_CHOICES` | list[int] | 平台可用的线程数档位 | `thread-layout` 的 THREAD_NUM 选型 |
| `CORE_NUM` | int | host 可查询的向量核数 | det 三阶段核数；nondet 核数两步法；超发核 guard |
| `MAX_RANK` | int | 张量 rank/stride 数组的定长上限 | TilingData 的 dims/strides/srcStrides 数组；dim/nd rank 上限 |
| `INDEX_DTYPE_SET` | list[type] | index 张量支持的 dtype 集合 | `linearize` 的 INDEX_T 分派；IDX32/IDX64 与 use64Bit 判定 |
| `UINTDIV_MAGIC_SHIFT` | desc | scalar 端为固定除数预计算 magic/shift 的能力、输出落点与 host/device 限制 | Phase1/Phase3 坐标解码的快除中转原则 |
| `UINTDIV_VF` | desc | VF/SIMT 计算域用 magic/shift 做固定除数快除的能力及商/余数组合方式 | Phase1 坐标线性化；Phase3 (b,k,s) 一维化 |
| `UINT32_MULHI_NATIVE` | bool | 32 位乘高是否为原生单指令 | key 位宽自适应；IDX32 编译期全展开策略 |
| `UINT64_MUL_HIGH_COST` | desc | 64 位乘高实现代价、寄存器压力与全展开溢出的边界 | IDX64 路径的展开预算；优先 useInt32Index 降级 |
| `BYTE_LEVEL_MOVE` | bool | 是否支持按元素字节宽 bit-exact 搬移，覆盖非常规 dtype | `type-assign` 的全 dtype 字节视角搬运；W==1 标量路径 |
| `L2_VOLATILE_RW` | desc | GM 写入是否需要/支持 volatile 写穿，跨核读取是否可绕过失效的 DCache | det Phase1 写 workspace、Phase3 读 workspace；`kernel-entrance` 的同步纪律 |
| `CROSS_CORE_SYNC` | desc | 跨核屏障、调度模式与超发核参与屏障的能力 | det 三阶段 `SyncAll`；SortLib 多核屏障；`thread-layout` 超发核 guard |
| `ALIGNED_SINGLE_STORE_ATOMIC` | bool | 对齐的 1/2/4/8 字节单条 store 是否原子（无撕裂），是否覆盖全部 dtype | nondet 直写的正确性依据；`type-assign` 的无 dtype 白名单结论 |
| `STABLE_SORT_LIB` | desc | 稳定排序库能力：稳定性、tile/active_cores 运行期参数、workspace/UB 契约 | `deterministic` 的 last-write-wins；`thread-layout` 的排序核数；`tiling-preprocess` 的 SortLib tiling |
| `UB_ALIGN_BYTES` | int | UB 分配与 TilingData/参数缓冲的对齐粒度 | `kernel-entrance` 的 AlignAllocN；`tiling-preprocess` 的 workspace 对齐 |
| `WORKSPACE_MODEL` | desc | 用户 workspace 段数、对齐、det/nondet 差异与系统 workspace 分配义务 | `tiling-preprocess` 的四段/五段布局；`deterministic` 的段表；nondet 零用户 ws |

## 3 谓词使用示例

det 三段式：

> Phase1 写 linearIdx 依赖 `L2_VOLATILE_RW` 与 `CROSS_CORE_SYNC`；Phase2 的 last-write-wins 由 `STABLE_SORT_LIB` 提供，排序核数受 `CORE_NUM` 与 active_cores 约束；Phase3 段尾覆盖复用同一套 workspace 与同步纪律。

nondet 直写：

> 单 VF 直接写 var 依赖 `GM_DIRECT_ACCESS` 与 `ALIGNED_SINGLE_STORE_ATOMIC`；该路径不消费 `STABLE_SORT_LIB`、`CROSS_CORE_SYNC` 与 `WORKSPACE_MODEL` 的 det 分支。

资源预算：

> `UB_CAPACITY` 查询值需先扣除 `DCACHE_RESERVE_RANGE` 与静态参数；workspace 布局由 `WORKSPACE_MODEL` 与 `UB_ALIGN_BYTES` 共同约束。

适配层在 `capability-mapping.md` 中填入具体 API 与限制；通用层公式不因语言不同而改写。
