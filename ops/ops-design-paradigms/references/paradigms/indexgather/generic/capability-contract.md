# IndexGather 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法中依赖硬件/语言能力的量，统一抽象为**能力谓词**。通用层文档只引用谓词名，不出现具体语言 API。提供专用适配层的语言必须在其 `capability-mapping.md` 中对全部谓词给出真值。

## 1 使用规则

1. **通用层**：算法描述遇到能力依赖点时引用本文档中的谓词名，不写具体数值或 API 名。
2. **适配层**：`adapters/<lang>/capability-mapping.md` 逐项填写谓词真值与对应的 API 实体。
3. **可填性检验**：新语言适配层必须能对每个谓词给出真值；填不出真值说明通用层抽象有缺陷，应修订抽象而非绕过。
4. **冲突处理**：同一能力点若适配层真值与通用层假设冲突，以适配层真值为准，并在设计结论中标注影响（如降级为普通除法、减少线程数、回退运行期 rank）。

## 2 谓词定义表

| 谓词 | 类型 | 含义 | 影响的通用层知识点 |
|------|------|------|------------------|
| `SIMT_SUPPORT` | bool | 目标芯片是否提供 SIMT 线程执行能力 | `patterns.md` 前置约束；整体 SIMT 路线 |
| `VF_SERIAL_EXEC` | bool | 多个 VF/调度区域是否串行执行，跨区域访存是否天然有序 | `thread-layout`；`decode-model` 的区域边界假设 |
| `VF_START_COST` | desc | VF/调度区域固定启动成本的量级 | 纯计算循环合并到单 VF；`thread-layout` |
| `GM_DIRECT_ACCESS` | bool | SIMT 线程是否可直接以 GM 指针访问全局内存（经 DCache/L2），无需先搬 UB | `decode-model` 的读收集路径；UB 参数中转范围 |
| `DCACHE_SIZE` | bytes | SIMT 侧 DCache 容量 | UB 预算；`tiling` 的 DCache 预留 |
| `DCACHE_RESERVE_RANGE` | range[bytes] | host 可配置的 DCache 预留范围 | `tiling` 的 DCache 预留与下限兜底 |
| `UB_CAPACITY` | bytes | 可查询的片上 UB 容量（不含 DCache 预留） | 参数缓冲/UB 预算；`tiling` |
| `SIMT_REGISTER_MODEL` | desc | 线程数与每线程寄存器的关系，以及寄存器溢出行为 | `thread-layout` 的线程数选型 |
| `SIMT_THREAD_CHOICES` | list[int] | 平台可用的线程数档位 | `thread-layout` 的 THREAD_NUM 选型 |
| `CORE_NUM` | int | host 可查询的向量核数 | `thread-layout` 的核数两步法 |
| `MAX_RANK` | int | 张量 rank/stride 数组的定长上限 | TilingData 数组定义；合轴后的段数上界 |
| `INDEX_DTYPE_SET` | list[type] | index 张量支持的 dtype 集合 | `decode-bounds`；IDX32/IDX64 路径判定 |
| `UINTDIV_MAGIC_SHIFT` | desc | scalar 端为固定除数预计算 magic/shift 的能力、输出落点与 host/device 限制 | `decode-model` 的快除中转原则 |
| `UINTDIV_VF` | desc | VF/SIMT 计算域用 magic/shift 做固定除数快除的能力及商/余数组合方式 | `decode-model` 的坐标解码；A/B/C 三型解码 |
| `UINT32_MULHI_NATIVE` | bool | 32 位乘高是否为原生单指令 | IDX32 编译期全展开策略 |
| `UINT64_MUL_HIGH_COST` | desc | 64 位乘高实现代价、寄存器压力与全展开溢出的边界 | IDX64 运行期 rank 截断策略 |
| `BYTE_LEVEL_MOVE` | bool | 是否支持按元素字节宽 bit-exact 搬移，覆盖非常规 dtype | 读收集的 dtype 覆盖；字节级搬移路径 |

## 3 谓词使用示例

通用层坐标解码：

> `c = i ÷ outStrides[d]`；`magic/shift` 由 `UINTDIV_MAGIC_SHIFT` 预计算，商由 `UINTDIV_VF` 得到，余数用 `i - c × outStrides[d]` 组合。

通用层线程排布：

> `needCoreNum` 由 `CORE_NUM` 与 `PER_CORE_MIN` 决定；线程数受 `SIMT_REGISTER_MODEL` 与 `SIMT_THREAD_CHOICES` 约束。

适配层在 `capability-mapping.md` 中填入具体 API 与限制；通用层公式不因语言不同而改写。
