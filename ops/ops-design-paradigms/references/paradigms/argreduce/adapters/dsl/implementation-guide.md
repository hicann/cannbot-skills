# ArgReduce 范式 CANNBotDSL 实现指引

> **层级**：CANNBotDSL 语言适配层叶子文档。通用层算法见 [../../generic/index.md](../../generic/index.md)；谓词真值见 [capability-mapping.md](capability-mapping.md)。
>
> **版本锚**：CANNBotDSL **0.8.0**（Ascend 950PR/950DT，arch35）。API 实体按 0.8 口径；`Channel`/`ops.memcpy`/`ops.sync`/`build_program` 等 0.7.x 名称不再使用，差异见 [capability-mapping.md](capability-mapping.md) §5。
>
> 本指引把 ArgReduce 通用层三层知识（选择语义 / 场景分类 / 切分资源）落到 DSL 的公开入口 / Host launcher / Kernel 三层结构。具体 API 签名与设备能力以调用方提供的 DSL 资料为准，本文不复制 API 签名。

## 1 三层知识 → DSL 三层结构映射

| 通用层知识 | DSL 落地 | 关键约束 |
|------|------|------|
| 轴折叠 A1/R/A0 + 场景判定（EMPTY/COPY/AR/ARA） | **Host launcher 纯元数据操作**：shape 运算 + 判定，结果作 kernel 分支/参数 | 不触发设备计算；判定顺序 EMPTY→COPY→AR/ARA 固定 |
| 选择语义（主规则/平局/NaN） | **Kernel 内 `vf(mode="simd")` 作用域中的 `ops.reg` 比较 + mask 显式构造**：比较方向（argmin 用"小于等于保留前者"类方向）、NaN 掩码（自比较检测 + OR 并入）逐项落地 | 平局规则三处同源（块内/跨块/跨核）；NaN 不依赖 API 原生行为（[capability-mapping.md](capability-mapping.md) §3 规则 3 强制穿刺） |
| 位置流（等差生成/偏移拼接） | Kernel 内 `ops.reg` 算术构造；块基址在生成侧一次施加。**首选 `vdups` 常量广播**（int16 dtype 保持，比 `varange` 逐 lane 增量更高效——段内常量 + 标量基址即可覆盖等差场景）；`varange` 为等价备选（适合非常量位置模式）。**位置→输出转换**：`vload_unpack(..., unpack_mode=UnpackMode.B16_TO_B32)` 展宽（int16 lane 向量展宽为 int32 双 half-lane 写出）+ `vcast` int16→int32（含 RoundingMode）；回写 16 位用 `vstore_pack(UnpackMode.B32_TO_B16)` | 位置元素宽度按 [../../generic/selection-semantics.md](../../generic/selection-semantics.md) §5 等宽原则选型；DSL 不暴露 lane 时声明等价机制并核对块内偏移上界 |
| 切分（FullLoad/Split/多核） | Host 计算切分参数（余数摊派均衡）传入 kernel，或经 tiler 表达 | UB 三通道预算（值+位置+输出/暂存）以运行时查询容量反解；容量假设值代入须声明 |
| GroupReduce（R 借多核） | Phase 1 局部 (值,位置) → GM workspace 普通 `Tensor` → 跨核同步（`global_sync_all(flag_ids=...)`；默认占 flag id 0/1/2）→ Phase 2 聚合 | 聚合三要素（偏移一次施加/平局参与/无空块核）语言中立直接沿用；GroupReduce 属性能特化，默认不纳入，启用前须按 0.8 flag id 规则验证 |

## 2 代码组织（三层职责）

| 层 | 职责 | 边界 |
|------|------|------|
| 公开入口 | dtype/shape/layout/device 参数组合检查；输出分配委托 | 不藏设备计算 |
| Host launcher | 轴折叠、场景判定、切分参数（余数摊派）、输出/workspace 分配、launch | 仅元数据/分配/零拷贝视图；torch 调用触发设备算子即越界 |
| Kernel | 选择段、位置流、尾块掩码、跨块/跨核合并、输出合成 | 按数据所有权拆阶段方法；EMPTY 早退分支在此层 |

## 3 场景 × 搬运策略（DSL 视角）

| 场景 | DSL 搬运策略 | 依据 |
|------|------|------|
| AR（R 连续） | 常规 GM→片上搬运（`mem_copy` 基础搬运） | 数据天然连续 |
| ARA（R 跨步） | 优先评估 **layout transform（布局重排 [A1,R,A0]→[A1,A0,R]）后走 AR 路径**；朴素跨步循环为最后手段 | `mem_copy` + `copy_options`/`Tensor.permute`/`reshape` 可表达布局重排（能力待验证）；AscendC 侧实测：朴素跨步带宽仅为 2D 块搬运 ~30% |
| COPY（R==1） | 纯搬运 + 常量填充（indices 全 0） | 无选择段 |
| EMPTY | 入口短路（N=0 不 launch）或 Kernel 早退，返回空输出 | 0.8 无特殊原语；Host 短路与 Kernel 早退均为文档化形态 |

**模式裁剪（DSL 视角，语言中立原则）**：DSL 任务无需 AscendC 的 8 模式 TilingKey 体系——基础三场景 {COPY, AR, ARA} 即**正确性完备集**（任意 shape 必然路由到其一，AR/ARA 内含切 R 循环覆盖超大 R）；RA/Gather/GroupReduce 等均为性能特化，按 DSL 能力（待验证谓词收敛后）与性能需求裁剪。裁剪必须记录：不纳入清单 + 依据 + 扩展路径；禁止无记录缺省。

## 4 Buffer/RingBuffer 规划

1. buffer 数量按通用层存活节点模型推导：standard 路径 P = 值流 tile + 位置通道（若落 buffer）+ 输出暂存；选择段常驻寄存器时位置通道不占 buffer（以 `ops.reg` 落点语义验证结论为准）
2. 阶段内临时量用 Buffer；仅 GroupReduce 的"局部结果→聚合"交接处评估 RingBuffer（`next_write`/`next_read`，无 `ChannelKind`/`kind`）
3. 单 buffer 容量按 UB 三通道预算反解（[../../generic/splitting.md](../../generic/splitting.md) §1）；Buffer/RingBuffer 的 shape/depth 必须编译期静态（0.8），depth 计入容量；SIMD 路径按 `get_mem_size("ub")`=253952B，若引入 SIMT 路径须扣 32768B 上下文预留；容量与对齐粒度以待验证真值代入并声明假设

## 5 尾块与同步

1. 尾块掩码在 kernel 内处理；不把无效尾行写入输出
2. 跨块合并（Split 模式）与跨核聚合（GroupReduce）的平局规则与块基址偏移一次施加原则语言中立，直接沿用通用层
3. 跨核同步（全核到齐）用 `global_sync_all(flag_ids=...)`（0.8 默认占用 flag id 0/1/2）；GroupReduce 设计默认按保守串行聚合兜底，启用栅栏前须验证 flag id 避让并声明假设

## 6 产出与验证义务

每个 DSL 实现决策按如下格式回答（micro-patterns 纪律）：

```
适用前提（含待验证假设标注）→ 最小代码位置/伪代码 → 生命周期或同步不变量 → 对应 Unit 测试
```

- 验证义务最低集：**语义三声明**（平局/NaN/空 R）各配端到端用例；NaN 用例必须覆盖首/中/尾/全 NaN/与极值混合（强制穿刺规则）；golden 显式 NaN-aware（禁用裸 np.argmin/np.argmax 家族默认 NaN 语义）
- 确定性（同输入 bitwise 可复现）为验收项；引用待验证谓词的结论按 [capability-mapping.md](capability-mapping.md) §3 声明假设

## 7 设计与证据纪律（DSL 专项）

DSL 设计产出没有 AscendC 的 TilingKey 模板兜底，符号与切分参数由设计者自定，故在 [../ascendc/design-spec.md](../ascendc/design-spec.md) §3 语言中立项（场景/平局/偏移/精度）之外，追加以下自查项，每项给出证据位置：

| # | 检查项 | 通过标准 |
|---|--------|---------|
| 1 | 符号与切分参数单一定义源 | 切分参数（如 tile 数、分组数）、轴符号在全部设计文档中只有一个定义；同名符号两处的公式可逐符号复算一致 |
| 2 | 地址/索引公式可复算 | 读址/写址/位置基址公式在主文档与各分支文档间一致；用主 case 的末 group 代入，抽查一条 lane 的实际地址 |
| 3 | 时序与容量自洽 | 槽复用（WAR）时序与 buffer 容量在同一分块口径下复算一致；时序示例必须使用真实执行的组/task，不得引用被跳过组 |
| 4 | 覆盖声明有实据 | 核数、槽覆写、API 使用等覆盖陈述必须由实际触发记录支撑（真实 block_num、真实命中的用例行）；未触发即按未覆盖处理，不得以设计意图代替 |
| 5 | 证据锁定当前版本 | 探针/回归/日志引用当前实现的路径与版本（模块 sha/hash）；禁止硬编码历史轮次路径；新旧日志字节相同视为未重跑 |
