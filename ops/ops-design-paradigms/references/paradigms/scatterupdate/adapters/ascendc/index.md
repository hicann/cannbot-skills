# ScatterUpdate 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 ScatterUpdate 范式中**强耦合 AscendC 语言**的知识：API 实体、代码模板、参考实现、编译器约束、设计约束。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

**适用前提**：目标芯片需支持 SIMT 编程能力；不支持时本适配层不适用。当前 AscendC 能力真值基于 arch35 / ascend950（DAV_3510）实证，其他芯片需按 [capability-mapping.md](capability-mapping.md) 重新确认。核心依赖 API：`__trap`（utils/debug/asc_assert.h）、`Simt::UintDiv` / `GetUintDivMagicAndShift`、`asc_vf_call`、`DataSyncBarrier` / `SetFlag<HardEvent::V_S>` / `SyncAll`、`SortLib::SortTilingCompute / SortInvoke`（det 排序路径）。

## 2 强制规则（生成前必读）

1. **先定型再设计**：按 patterns.md 场景判定完成维度判定（写语义划界 + index 形态 × 越界 × 确定性），禁止跳过判定直接套用某一型模板；src 表示为型内形态决策
2. **写语义划界**：本适配层仅承载 assign 覆盖写（`out[idx] = src`，纯搬运 bit-exact）；reduce 归约写（`out[idx] op= src`）属归约写范式，不在本适配层范围，禁止混入本范式设计；覆盖顺序（last/first-write-wins，nondet 下为任意赢）是语义决策点
3. **确定性分治**：默认 nondet 直写——多写者任意赢即覆盖语义正确，仅需对齐单条 store 原子防撕裂（全 dtype 无白名单，tensor_scatter_nd_update v2.0 实证）；张量源 `dupRatio > THRESHOLD`（规避同地址并发写慢一个数量级）或确定性请求（last-write-wins 固定顺序）走 det 排序段尾覆盖，标量源（值同）仅高重复走排序（纯性能，值同无确定性问题）
4. **越界形态必须显式决策**（优先级 ①声明→②对标→③默认 error `__trap`）；det/nondet 双模板越界语义必须一致（真值源 [../../generic/bounds.md](../../generic/bounds.md)，写法见 [bounds-impl.md](bounds-impl.md)）；assign 常见对标形态为 skip（哨兵，TF/Ascend 系）
5. **禁止「先截断再比较」**：越界检查在 INDEX_T 原值域进行，窄化截断永远最后
6. **稳定排序是 det 段尾覆盖的确定性来源**：golden 必须复现段尾元素写序（last-write-wins，稳定排序 + 段内原始顺序），纯搬运 binary 比对 (0,0) bit 级
7. **TilingKey 仅编码场景位，禁止枚举 dtype**（dtype 由 DTYPE_* 宏编译期实例化）；张量源 `countMode + keyDtype + deterministic` 三场景位（无 shapeMode；nondet 不消费的维度用占位值）；标量源（值同）为 `countMode + keyDtype + dedup`（dedup 位即路径位，仅 `dupRatio` 驱动，确定性开关不触发）
8. **同步纪律**：Phase1 VF 后 `SetFlag<V_S> + WaitFlag<V_S> + SyncAll` 三连；`SortInvoke` 前后 `pipe.Reset()` ×2；scalar 写 UB 后 `DataSyncBarrier<MemDsbT::UB>()`；kernel 末尾 VF 后无需任何 join（[kernel-entrance.md](kernel-entrance.md) §5 完整实测表）
9. **空输出早退必须先于快除预计算与 SortLib tiling**（N==0 时 strides 可能含 0 除数）
10. **`GetUintDivMagicAndShift` 出参必须为 Local Memory 标量**：禁止直写 `__ubuf__` 左值（[../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §2.2）
11. **字节视角搬运**：全 dtype 按 `BytePackType<T>` 字节视角 bit-exact 搬运（sizeof 1→uint8 / 2→uint16 / 4→uint32 / 8→uint64），含 W==1 标量路径，禁止 `T` 原生赋值（SIMT 原生标量仅到 float/int，`T` 原生赋值是无法纳入 double/fp8 的根因）
12. **能力谓词以 [capability-mapping.md](capability-mapping.md) 为准**：未提供专用映射或标注待验证的项，须在产出中声明假设与验证义务，不得写成已验证事实。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 全局心智模型 | [overview.md](overview.md) | 正交维度模型 / 双模板分派 / TilingKey / TilingData 划分总览 |
| Host 侧 Tiling | [tiling-preprocess.md](tiling-preprocess.md) | dim 归一化 + 校验 / SortLib tiling / 多核切分 / workspace 布局 |
| 能力谓词映射 | [capability-mapping.md](capability-mapping.md) | SIMT/VF、DCache/UB、线程/寄存器、rank/index dtype、快除、mulhi、字节搬移、L2 volatile、跨核同步、单 store 原子、稳定排序、UB 对齐、workspace |
| Kernel 入口与注册 | [kernel-entrance.md](kernel-entrance.md) | 入口分派 / 双模板 host 结构（六文件）/ 三段式编排 / 同步纪律 |
| Phase1 冲突 key | [linearize.md](linearize.md) | 各 index 形态的 Phase1 写法 / srcPos 双输出 / 空 tensor 早退 / uint32 坐标计算域降级 |
| det 段尾覆盖 | [deterministic.md](deterministic.md) | 排序动机与成本 / (b,k,s) 任务分解 / 段尾覆盖骨架 / golden 精确写序 |
| 覆盖写场景 | [type-assign.md](type-assign.md) | 双路径分派（默认 nondet 任意赢）/ 标量源模式 / 退化场景核对表 |
| 越界写法 | [bounds-impl.md](bounds-impl.md) | 三形态 kernel 写法 / nd 逐维判定 / 哨兵实现 |
| 参考实现 | [example/](example/) | assign + nd 双模板基例 |

## 4 公共知识依赖

以下跨范式 AscendC 公共知识位于范式外，按需展开：

| 主题 | 位置 |
|------|------|
| UintDiv 快除通用规则 | [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| 能力谓词映射 | [capability-mapping.md](capability-mapping.md) |
| API 白名单（存在性唯一判据） | [../../../../knowledge/api/regbase_api_whitelist.md](../../../../knowledge/api/regbase_api_whitelist.md) |

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 设计输出按 generic 层语义模型 + 本层实现约束组合产出；通用层冲突时以语义模型为准，实现层冲突时以本层 API 事实为准
4. 编码前必须检视至少一个现有参考实现（见 [example/](example/)，选型参考开源算子表 `references/reference-ops/open_source_operator_table.md`）
