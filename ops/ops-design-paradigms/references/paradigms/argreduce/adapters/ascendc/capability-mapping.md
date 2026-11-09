# ArgReduce 能力谓词真值映射（AscendC）

> **层级**：AscendC 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 全部谓词给出真值与 API 实体。
>
> **平台标注**：本表默认面向 arch35（ascend350/950 代）；标注 `DAV_2201` 的约束来自旧代平台实践（with-index 变体知识），在新平台使用前按 API 白名单核实。标注"待验证"的项须作为假设输出。

## 1 谓词真值表

| 谓词 | 真值 / API 实体 | 备注 |
|------|----------------|------|
| `INDEX_TRACKING_REDUCE` | Level-2 API：`AscendC::ReduceMax/ReduceMin(dst, src, tmpBuffer, count, calIndex=true)`；tmpBuffer 尺寸 `AscendC::GetReduceMaxMinTmpSize<T>(count, true)`。输出格式：`dst[0]=值, dst[1]=索引`（索引为位编码 float，需 `reinterpret` 还原） | AR 场景手法 A。**负面实证（2026-10 argmin 穿刺）：`ReduceMin(calIndex)` 的 NaN 行为不可依赖（NaN-aware argreduce 禁用此路径，须寄存器级显式掩码）**；count 上界与 dtype 支持以 API 白名单为准；arch35 工业主路径未用此 API（寄存器级自实现，性能与 int64 支持更好） |
| `VECTOR_COMPARE_SELECT` | 寄存器级：`AscendC::Reg::Compare<T, CMPMODE::GE/GT/LE/LT/NE>(mask, a, b, mask)`、`Select(dst, a, b, mask)`、掩码布尔 `AscendC::Reg::Or/And(mask, m1, m2, b8Mask)`；Level-2：`Compare` + `Select` 掩码模式 | AR 场景手法 B / ARA 主手法。NaN 检测：`Compare NE (v, v)` 自比较 + `Or` 并入选择掩码。DAV_2201 有 Compare 256 字节对齐约束（float 64 元素倍数），arch35 待验证 |
| `INDEX_ARITH_OP` | 等差数列：`Duplicate(reg, start, fullMask)` + 掩码增量；**段内拼接**：`Mul(段号, VL) + Adds(lane)`；**跨宽直写**：`Reduce<SUM>`（dtype 白名单见 [highlevel-api.md](highlevel-api.md) §2a：uint32←uint16 ✓、uint32←float ✗ 编译期拒绝）；**uint16 展开**：`UnPack` 低/高半 → uint32；**块基址**：`Adds(startR)` 一次施加（生成侧，合并侧不再加） | 位置流不在 GM，片上生成；Reduce<SUM> 组合越表 = 编译期 static_assert 失败，非运行期错误 |
| `INDEX_STORAGE_TYPE` | **位置在片上用什么类型存——必须与值通道等宽**（同 lane 数、同掩码粒度，否则掩码对不上）：fp16/bf16（2 字节）→ **uint16**（128 lane/寄存器，块内偏移 ≤ 65536）；fp32/int32（4 字节）→ **uint32**（64 lane，块内偏移 ≤ 2^32）。全局位置 = 块基址 + 块内偏移拼接 → int32 输出上限 2^31。**转换到输出**：uint16 经 UnPack 展开后直写 int32（int64 扩展开启 Cast）。**历史口径**：float32/2^24 是 DAV_2201 的权宜做法（当时 Select 不支持 int32 目的），arch35 不适用 | 块内偏移上界须显式核对（UB 反解 cutRSize ≤ ~29152@216KB，天然满足）；uint16 路径位置写出配套 UnPack 展开 |
| `VECTOR_LANE_WIDTH` | `GetVLSize()` 运行时获取；arch35 UT 配置 VL=256。GATHER/RA 特化判定中的 VL 比较一律引用运行时值 | 不硬编码 |
| `ONCHIP_CAPACITY` | **必须运行时查询**（Host 侧 `PlatformAscendC::GetCoreMemSize(UB)` / 直调路径 `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`），**禁止按文档标称值或仿真配置值硬编码**。实测锚点：Ascend950PR = **216KB（221184B）**；DAV_3510 文档口径 248KB、UT 仿真配置 256KB——三值三来源，一律以运行时查询为准。单 buffer 上限同源查询 | 同代不同 SKU 容量不同；UT 须含实测容量环境下的反解用例 |
| `DATA_ALIGNMENT` | 32 字节（BLOCK_SIZE，DataCopy 搬运对齐） | 尾块 padding、workspace 对齐、切块取整的依据 |
| `MULTICORE_BALANCE` | **余数摊派**（工业口径，A1/R 两方向同款消费）：`blkFactor = size / coreNum`，`blkTailFactor = size % coreNum`（摊派核数）；kernel 侧**前 `blkTailFactor` 核各 +1**、其余核 `blkFactor`，`blockOffset = blockIdx×blkFactor + min(blockIdx, blkTailFactor)`——负载差恒 ≤ 1 | 禁止 CeilDiv 截断式（空块核脏数据风险）；禁止"尾核兜余量"式（`tail = size − factor×(coreNum−1)`，负载差可达余数且 kernel 消费公式不同——2026-10 argmin 穿刺实测证伪） |
| `CROSS_CORE_SYNC` | `AscendC::SyncAll()`（全部核到齐才放行）+ Host 侧 `context->SetScheduleMode(1)`（声明需等待全部核空闲启动） | **仅 GROUP_REDUCE 使用**；核数口径必须物理核数 |
| `WORKSPACE_MODEL` | 总申请 = `workSpaceSize + 16MB`（系统部分）；用户部分仅 GROUP_REDUCE 使用。布局：AR 分支 `[value: core×outAAlign][indice: core×outAAlign]`；RA 分支 `[value: aSize×core×nextAAlign128][indice: aSize×core×indiceAlign128]` | **outAAlign 对齐粒度 = 32B**（工业口径：`ELEMENT_PER_BLOCK = 32/sizeof(T)`，fp16=16 元素；256B 为可选保守超集，代价 workspace ×2——按需声明） |

## 2 谓词组合的落地结论

| 组合点 | 结论 |
|--------|------|
| AR 场景手法 B 的 R 上界 | 单行 R 全载受 UB 三通道预算约束（值 + 等宽位置 + 输出/暂存）；块内偏移受位置元素容量约束（uint16 → 65536，UB 反解天然满足） |
| 位置寄存器选型 | 与值通道等宽（fp16 → uint16 同 128-lane，掩码共享）；`Reduce<SUM>` 直写须在 dtype 白名单内（uint32←uint16 ✓）；全局位置块基址拼接 → 2^31 全域 |
| Gather 特化触发（性能路径，非通用谓词） | `R × eleBytes < vRegSize`（R 一条向量放得下）**且** `R × eleBytes` 非 32 倍数（bank 冲突排除）**且** 源基址 32B 对齐（不满足先折叠索引通道或放弃）**且** A 方向够多核并行。用 `MicroAPI::Gather` 做 UB 内 AR→RA 转置 |
| GROUP_REDUCE 触发 | R 算力型长轴（`R × eleBytes ≥ 物理核 × VL`）且 A 方向无并行度；核数口径必须 `originCoreNum`（物理核） |
| NaN + 平局落地 | 比较掩码 = 主规则掩码 OR NaN 自比较掩码；argmax 用 `GE`（相等保留前者 = first-occurrence），argmin 用 `LE`。**已实证（arch35 真机）**：NaN lane 晚于最大值 lane 构造端到端 bitwise 通过 |
| 寄存器级 Reduce 归约语义 | `Reg::Reduce<MAX/MIN>` 官方承诺产出"值 + 第一个极值索引"，且"输入含 NaN 时写入 NaN 并保存第一个 NaN 的索引"——语义与 argmax 一致，可作 AR 路径候选手法（索引为寄存器内局部位置，跨段拼接仍需位置算术；性能与约束对比待评估） |

## 3 待验证清单

| 项 | 风险 | 验证方式 |
|----|------|---------|
| Compare 对齐约束（arch35） | DAV_2201 的 256B 约束是否延续（**部分已实证**：Reg 级 Compare 非 32B count 真机通过，穿刺 10002） | 白名单 + 穿刺测试 |
| ~~`ReduceMax/ReduceMin(calIndex)` 在 arch35 的可用性~~ | **已结案（2026-10 argmin 穿刺实证）**：NaN 行为不可依赖，NaN-aware argreduce 禁用；非 NaN 语义算子使用前仍须白名单核实 count 上界 | 穿刺报告（issue #ReduceMin NaN） |
| VL 的 dtype 依赖 | VL 随 dtype 宽度变化（fp16 128-lane / fp32 64-lane 已由等宽推导实证） | 运行时 `GetVLSize()` 实测 |
| `Reg::Reduce<MAX/MIN>` 索引输出格式 | lane 局部索引的跨段拼接成本与约束 | API 文档 + 穿刺对比 |
