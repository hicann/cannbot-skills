---
type: "workflow"
title: "Kernel Stage 与同步模式"
description: "任务组、双缓冲、ready/free、HardEvent 和 workspace 生命周期。"
tags: ["catlass-cpp", "workflow", "design", "develop", "synchronization", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:architecture-separation-source-check", "at": "2026-09-20T00:00:00Z"}]
sources: [{"id": "pr1069-kernel-stage-sync-patterns", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/kernel-stage-sync-patterns.md", "title": "PR1069 fixed source for Linear Attention Kernel Stage 与同步模式", "kind": "repository"}, {"id": "catlass-arch-separation", "resource": "git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c", "title": "CATLASS fixed source for AtlasA2 and Ascend950 architecture separation", "kind": "repository"}]
consumers: ["catlass-cpp-design", "catlass-cpp-develop"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、阈值、失败条件、状态恢复或验收要求。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

# Kernel 代码与同步模式

本文提炼示例 kernel 的函数划分、数据交接和同步写法，作为 CANNBot 本地候选模式库。开发新算子时，
根据自己的 `docs/api.md`、`docs/design.md` 和目标平台，从候选模式中选择或组合实现，确定函数组织、
任务划分、数据形状、地址和事件数量，在当前算子工程中完成实现。本文不规定固定的 Stage 数量、
固定的 AIV 映射或固定的流水顺序；最终选择必须在设计文档中记录，并通过正确性、资源和性能验证。

所有候选模式都必须满足以下不变约束：数据依赖有明确的生产者和消费者；ready/free、EventID 和
slot 的发布与等待成对；复用物理地址前完成最后一次读写；尾块的无效区域已初始化；空任务不会破坏
通知平衡；不存在依赖未满足、flag 溢出或死锁。候选模式之间只比较实现策略，不降低这些约束。

### 架构适用性

本文件中的混合 kernel、一个 AIC 与两个 AIV 分工、CrossCore ready/free、HardEvent、GM workspace
和 slot 生命周期是 A2/A3 与 A5 的共享模式。数据交接和 Vector 执行模型必须按架构实例化：
A2/A3 使用 `CATLASS_ARCH=2201`、`Arch::AtlasA2` 和 MemBase 流水，Cube→Vector 默认经
`L0C -> GM -> UB`，Vector→Cube 经 GM workspace；A5 使用 `CATLASS_ARCH=3510`、
`Arch::Ascend950`，在目标组件支持时可选择 RegBase/VF、Fixpipe 直达 AIV UB 和 AIV UB 直达
AIC L1。A5 同样可以保留 MemBase 或 GM 中转作为基线和回退路径。[^catlass-arch-separation]

## 1. Kernel 代码组织方式

示例的 kernel 入口负责解析切分参数（tiling）、计算临时工作区（workspace）基址，并按
`ASCEND_IS_AIC`/`AIV` 选择执行对象。矩阵计算单元 Cube 和向量计算单元 Vector 分别使用独立
处理函数，函数不隐式读取尚未完成的结果。按数据依赖和执行单元划分函数；新增或合并函数前，
重新检查仍在使用的数据，以及每次同步通知是否都有对应等待。

示例的混合 kernel 入口显式声明 AIC/AIV 协同类型（例如 `KERNEL_TYPE_MIX_AIC_1_2`），
AIC 和 AIV 各自通过 `Process()` 按核编号和核数跨步领取任务（grid-stride）。host 侧的 tiling
传递规模、地址偏移、任务组和分支开关；公式、数据搬运和同步留在 device 侧。示例中每个处理函数负责
一种执行单元（Cube 或 Vector），并显式接收输入、输出、缓冲槽位（slot）和有效长度；不得通过全局状态
隐式读取另一个处理函数尚未发布的结果。

## 2. 任务组、注意力头（head）和双缓冲

以下以一个 AIC 和两个 AIV 为例，给出逐 head 与同 head 分片两种写法。当前算子按详设选择映射，
并分别计算每个中间张量的范围、地址和容量。任务组和 head 编号可共用：

```text
taskIdx     = chunkTaskIdx * headGroupNum + headGroupIdx
hvBase      = headGroupIdx * taskGroupSize
taskCount   = min(taskGroupSize, HV - hvBase)
head        = hvBase + headOffset  // 0 <= headOffset < taskCount
```

### 逐 head：每个 head 由一个 AIV 完整处理

两个 AIV 交错承担 head。负责当前 head 的 AIV 称为 owner，另一 AIV 称为 non-owner。
后者参与选定协议中的通知和等待；当前 head 的计算、搬运及本地槽位推进放在 owner 分支内：

```text
owner = headOffset % aivCount
if subBlockIdx == owner:
    localOwnerIndex = (headOffset - subBlockIdx) / aivCount
    streamSlot = localOwnerIndex % ubSlotCount
    validRange = [0, D)
    gmAddress(o, i) = workspaceHeadBase + o * physicalRowBytes + i * elementBytes
    ubAddress(o, i) = ownerUbBase[streamSlot] + o * ubRowBytes + i * elementBytes
```

`HV` 是 value head 总数，`subBlockIdx` 是当前 AIV 编号；`D` 是当前张量的连续行长度，`o`
是外层行索引。workspace 的 head 基址按 task、head 和槽位布局取得。UB 容量计入 owner 完整
head 的实际存活数据；若采用双缓冲，`ubSlotCount=2`，每个 AIV 独立轮转，从自己的首个 head 开始。
A5 直连模式下，Cube 输出按当前 head 的 owner 写入对应 AIV UB；A2/A3 以及 A5 的 GM
基线先写对应 GM 区间，再由 owner 搬入本地 UB。non-owner 不读取该 head 的输出。

### 同 head 分片：两个 AIV 处理互不重叠的有效区间

下面将独立计算的连续行维 `D` 分成两片，示例通过对齐切点使两侧 DMA 写区间分离：

```text
alignElems = storeAlignmentBytes / elementBytes  // 取满足所用搬运 API 的整数粒度
split = min(D, AlignUp(CeilDiv(D, 2), alignElems))
begin = (subBlockIdx == 0) ? 0 : split
end   = (subBlockIdx == 0) ? split : D
validCount = end - begin
physicalRowElems = AlignUp(D, alignElems)
streamSlot = headOffset % ubSlotCount            // 两个 AIV 各自管理本地同轮槽位
gmAddress(o, i) = workspaceHeadBase + (o * physicalRowElems + i) * elementBytes
ubAddress(o, j) = localUbBase[streamSlot] + (o * localRowElems + j) * elementBytes
// 有效搬运：i = begin + j，0 <= j < validCount
```

`[0, split)` 与 `[split, D)` 无重叠且覆盖整行有效区。为每行分配完整的 `physicalRowElems`，
`localRowElems` 按本地搬运要求对齐；不足搬运粒度的尾部按目标 API 使用有效长度、padding 和
计算 mask。有效区末尾的物理 padding 由最后一个非空片按后继公式填入中性值，实际 DMA 写范围
与其他片及下一行分离。共享输入按各片公式确定读取范围，容量预算计入重复读取和临时量。其他 layout 先按真实
stride 推导地址，再选择满足相同覆盖与对齐条件的切点。

每个中间张量分别计算分片范围：Vector 写 GM 时，两片落到同一逻辑张量的对应区间；Cube 结果
回到 UB 时，分别写入两片所属 AIV 的本地地址。`validCount=0` 的片跳过计算和数据搬运，仍按
第 3 节的参与方约定发送、消费通知。目标平台若通过 GM 中转实现 Cube→UB，按相同区间写回和搬入。

切分维包含归约、归一化或相邻元素依赖时，先在详设中给出所需完整输入或跨片合并步骤。例如两片
先分别写出 partial，合并方等待两份实际结果到达后计算统计量，再向使用该统计量的片发布 ready；
后续计算等待合并结果。partial 与合并结果分别记录地址、消费者和释放点。

### 槽位与通知轮转

跨任务组复用槽位前，先等待该槽位的最后一次 `MTE3`/`V` 操作完成。双缓冲只
解决同一核内的生产与消费重叠，不能掩盖跨核数据就绪通知（ready）丢失或常驻数据的地址冲突。

跨任务组使用的同步标志（flag）可以按任务轮次轮转，也可为每条独立依赖链分配不同 flag，避免单个
计数 flag 长时间累积超过硬件深度。每次 set 产生一次待消费的通知（token），由对应 wait 消费。
轮转不能改变 ready 的 head 顺序；每次轮转前仍须完成上一轮 token 的 wait。

## 3. 跨 AIC/AIV 的 ready 协议

跨 AIC/AIV 的依赖使用 CATLASS `CrossCoreSetFlag`/`CrossCoreWaitFlag`，并把完成数据传输的管线
（pipe）写进 flag 的模板参数。下表区分“数据可以读取”和“地址可以覆盖”两类条件：

| 通知 | 发布条件 | 消费方 |
|---|---|---|
| `vecToCubeReady` | 当前 head 的有效 Vector 输出全部写入 GM/workspace | AIC 读取完整输入 |
| `workspaceFree` | AIC 完成对该 workspace 区间的最后一次读取 | 下一次写入该区间的 AIV |
| `cubeToVecReady` | Cube 输出写到映射指定的 UB 或中转区 | 当前 head 的 owner 或两个分片 AIV |
| `ubFree` | 所有相关 Vector 消费者完成 UB 区间的最后一次使用 | 下一次向该 UB 区间写入的 AIC |

以下是语义伪代码；`LoadAndRunVector`、`Publish*`、`Wait*`、`CopyCubeResult*` 等辅助动作描述
地址与完成条件，需要根据当前 SoC 的搬运、Fixpipe 和 CrossCore API 实现。`key` 表示详设中的
task、head、slot 和复用轮次，同一 flag 的双方按相同顺序推进。示例选择两个 AIV 都参与的聚合
协议：每个 AIV 各提供一次 ready/free 通知，AIC 等待聚合完成；AIC 的通知由两个 AIV 按协议消费。
模式值、广播或聚合语义及 EventID 分配按目标 API 确认。通知对应的数据粒度与硬件聚合范围
分别推导；其他协议也需写清真实写入方和等待方。workspace 和 UB 独立记录各自槽位的上次使用，
初次使用按设计初始化为空闲。

```cpp
// AIV：vecPart/cubePart 分别按第 2 节映射当前 Vector 输出与 Cube 输出。
// 逐 head 模式只有 owner 的 part 非空；分片模式使用各自 [begin, end)。
if (!vecPart.empty()) {
    WaitLocalBufferReusable(vecPart);
    LoadAndRunVector(vecPart);
}
WaitWorkspaceFree(previousWorkspaceUse);    // 等待该 workspace 槽的上次使用结束
if (!vecPart.empty()) {
    WaitVectorResultReadyForMte3();
    StoreOwnRangeToWorkspace(vecPart);
    WaitOwnMte3Complete();
}
PublishVectorReady(key);                    // 空片或 non-owner 同样贡献协议通知

WaitCubeReady(key);                         // 所有参与 AIV 消费同轮通知
if (!cubePart.empty()) {
    RunVectorStage(cubePart);
    WaitLastLocalUbUseComplete();
}
PublishUbFree(key);                         // 放在该区间的实际最后消费点

// AIC
WaitAllVectorReady(key);                    // 等待两个参与 AIV 的通知聚合
LoadCompleteWorkspaceInputToL1(key);
WaitLastWorkspaceReadComplete();
PublishWorkspaceFree(key);
RunCubeStage(key);
WaitAllVectorUbFree(previousUbUse);         // 等待目标 UB 槽的上次使用结束
WaitCubeResultReadyForFixpipe();
if (cubeMapping == HEAD_OWNER) {
    CopyCubeResultToOwnerUb(key, cubeOwner);
} else {
    CopyCubeResultToAivUb(key, 0, cubePart0);
    CopyCubeResultToAivUb(key, 1, cubePart1); // 空区间跳过搬运
}
WaitAllCubeOutputWritesComplete();
PublishCubeReady(key);
```

同 head 分片时，`WaitAllVectorReady` 对应两个实际写入方各自完成 MTE3；逐 head 时对应 owner
完成写入及 non-owner 参与协议。空片的通知只表示该参与方在本轮没有待完成写入。
将 free 的首次空闲状态、发布次数与下轮等待配对；后续还有消费者时，在它们全部结束后发布 free。
若一块 UB 被多条依赖边共用，按最后使用者共同确定释放条件。

每次 ready 对应当前 head 的实际数据范围，每次 free 对应将复用的物理地址。逐 head、分片
和任务组级归并分别定义参与者及通知次数。跨片归约的 partial 就绪、合并结果就绪和各自释放使用
对应的独立依赖边。通知只连接存在数据依赖或地址复用关系的处理函数，保留其他计算的并行性。

跨核 ready/free 与核内事件共同保护流水；上述完成动作可由目标 API 支持的管线事件和带 pipe
语义的通知落实，并在详设中写明覆盖的操作范围。例如真实 AIV 写入方的 Vector→GM→AIC
交接可对应以下 API 调用片段，事件、聚合模式和搬运参数由当前设计提供：

```cpp
SetFlag<HardEvent::V_MTE3>(vecToStore);
WaitFlag<HardEvent::V_MTE3>(vecToStore);
DataCopy(workspacePart, ubPart, copyParams);
CrossCoreSetFlag<aggregateMode, PIPE_MTE3>(vecToCubeReady);
// AIC：消费本轮协议要求的通知后，读取完整 workspace 输入。
CrossCoreWaitFlag(vecToCubeReady);
```

## 4. 核内事件与流水收尾

同一 AIC/AIV 内使用 `AscendC::HardEvent` 保护 MTE、Vector、Cube、Fixpipe 的局部
读写顺序。事件必须成对分配、使用、回收，并在 kernel 退出前排空最后一轮信号：

```cpp
auto loadToVec = pipe.AllocEventID<HardEvent::MTE2_V>();
auto vecToStore = pipe.AllocEventID<HardEvent::V_MTE3>();
auto storeToLoad = pipe.AllocEventID<HardEvent::MTE3_MTE2>();

DataCopy(ubIn, gmIn, bytes);
SetFlag<HardEvent::MTE2_V>(loadToVec);
WaitFlag<HardEvent::MTE2_V>(loadToVec);
RunVectorStage(ubIn, ubOut);
SetFlag<HardEvent::V_MTE3>(vecToStore);
WaitFlag<HardEvent::V_MTE3>(vecToStore);
DataCopy(gmOut, ubOut, bytes);

// 下一个任务组覆盖 ubIn 前
SetFlag<HardEvent::MTE3_MTE2>(storeToLoad);
WaitFlag<HardEvent::MTE3_MTE2>(storeToLoad);
```

Cube 侧同理使用 `MTE1_MTE2`、`MTE2_MTE1`、`MTE1_M`、`M_MTE1`、`M_FIX`、
`FIX_M` 等事件保护 L1/L0 装载、MMAD 和 Fixpipe。L0A/L0B/L0C 各自维护 ping/pong
事件；切换槽位前等待当前槽位的上一个消费者。kernel 尾部按事件数组逐项 `WaitFlag`
后再 `ReleaseEventID`，避免尚未消费的事件影响下一次调用。

同一执行管线的连续计算按调用顺序组织；跨 pipe、跨核和 workspace producer-consumer 边使用成对
的 `HardEvent` 或 `CrossCore*Flag`，并以对应的 ready/free 粒度证明数据可见性。

按 head 模式给每个地址区间指定 owner，按分片模式给两个写入方分配互不重叠的区间。
跨片归约先写各自 partial，再由设计指定的合并方产生最终结果，并为各段数据配置对应事件。

## 5. 地址生命周期与 workspace 复用

对每个中间量记录产生它的函数、首次和最后使用它的函数，以及何时可以释放或覆盖；同时记录
存储位置、地址偏移、大小、对齐、负责写入的核和缓冲份数：

```text
producer -> first consumer(s) -> last consumer -> release/reuse
location  -> offset/bytes/alignment -> owner -> slot/bank count
```

示例中的数据交接方式：

1. Cube→Vector：A2/A3 按相同有效区间完成 `L0C -> Fixpipe -> GM -> UB`；A5 在目标组件支持时，可通过 Fixpipe 将完整 head 写入 owner AIV UB，或用 dual-destination 将两片分别写入两个 AIV UB。A5 的 GM 中转仍作为基线和回退，并记录额外搬运。
2. Vector→Cube：A2/A3 和通用基线使用 `UB -> MTE3 -> GM workspace -> MTE2 -> L1`，AIC 读到完整数据后才发布消费完成；A5 只有在组件支持且 alias、容量和同步已验证时，才评估 `AIV UB -> AIC L1` 直连。
3. Cube→Cube：结果在 L1 按并发 head 数保留；不能因计算阶段（Stage）编号增加而重新从 GM 搬入。
4. Vector→Vector：结果在 UB 保留到最后一次 Vector 使用；后续计算可原位更新，不能提前覆盖。
5. 同一物理 slot 被新任务使用前，必须等待旧数据的最后一次 MTE3、V 或 Fixpipe 事件；
   仅等待跨核 flag 不足以证明本核写回已经完成。
6. 入口 workspace 可以拆成多个逻辑区域（region）；仅在使用时间不重叠时复用同一地址。
   区域数量和偏移必须依据当前 tiling 和容量计算得出。

workspace 偏移按 `coreIdx`、任务轮次和 head/分片偏移计算；使用两块交替工作区时，确认旧任务
已释放对应区域后再复用。尾块和变长任务使用有效长度和 mask，仍分配完整对齐槽位；完整空任务
可由 host 统一过滤。已进入协同任务的空片保留第 3 节约定的通知、等待和释放步骤。

## 6. 写入设计文档和代码的对应关系

`docs/design.md` 的“逐 Stage 详设”写清当前算子的具体参数和代码对应关系。
采用本示例的写法时，可按下表落实：

| 设计项 | 设计文档内容 | 对应代码示例 |
|---|---|---|
| Stage | 前后依赖、公式、负责执行的核、有效区域 | `ProcessStage*` 或等价明确函数 |
| AIV 映射 | 逐 head 或同 head 分片、有效区间、物理地址、跨片依赖 | owner 分支或分片 begin/end、对应搬运及 mask |
| 跨核依赖 | 通知方向、数据范围、聚合方式、槽位 | 成对 `CrossCoreSetFlag/WaitFlag` |
| 核内依赖 | 写入方和读取方、事件类型、复用条件 | 成对 `SetFlag/WaitFlag` 和事件回收 |
| 缓冲 | UB/L1/L0/workspace 偏移、份数、使用起止点 | `TBuf`、常驻缓冲或 workspace 指针及 ping/pong 索引 |
| 尾块处理 | mask、填充、中性值、空任务 | 有效长度、搬运参数和 Vector（VF 或 MemBase）/Cube mask |

开发阶段按 [`catlass-cpp-develop`](../../skills/catlass-cpp-develop/SKILL.md) 推进，记录事件配对和地址计算的
核对结果。任何新增 Stage、slot、flag 或地址复用都先回写 `docs/design.md`，再修改
kernel，使代码与当前算子已确认的设计保持一致。

## 7. 候选同步与流水模式

同步和流水应先保留一个容易验证的基线，再尝试优化候选。模式编号只表示模式类别，不表示优先级或
最终最优解；同一算子可以在不同 tiling、shape 或 SoC 上选择不同模式。

### 7.1 同步候选

| 模式 | 数据交接 | 适用场景 | 主要代价 |
|---|---|---|---|
| `SYNC_GM_QUEUE` | Vector → GM workspace → Cube，ready/free 双向握手 | 依赖关系复杂、先验证正确性 | GM 往返较多 |
| `SYNC_L1_HANDOFF`（A5） | AIV 直接写 L1，AIC 等待跨核 flag | 仅 A5 且目标组件支持；AIV/Cube owner 明确、L1 容量足够 | flag、alias、版本约束和生命周期更复杂 |
| `SYNC_HEAD_OWNER` | 一个 AIV 负责完整 head，其他 AIV 只参与协议 | head 粒度独立、避免跨片归约 | 负载可能不均衡 |
| `SYNC_HEAD_SPLIT` | 多个 AIV 写同一 head 的互不重叠区间 | 单 head 计算量大且可按连续维切分 | 需要处理分片边界和归约 |

### 7.2 流水候选

| 模式 | 执行关系 | 适用场景 | 主要代价 |
|---|---|---|---|
| `PIPE_SERIAL` | 一个 pack 的各 Stage 依次完成 | 建立精度和同步基线 | 并行度低 |
| `PIPE_PACK_OVERLAP` | pack N 的 Cube 阶段与 pack N+1 的 Vector 阶段重叠 | Vector/Cube 可独立推进 | 需要 slot、flag 和反向 credit |
| `PIPE_RHS_OVERLAP` | RHS 准备提前到叶子完成后，与 Y/A 重叠 | RHS 计算和 Cube 阶段没有真实数据依赖 | L1 常驻数据和生命周期更复杂 |

`PIPE_PACK_OVERLAP` 和 `PIPE_RHS_OVERLAP` 只有在设计文档中画清依赖边、slot 生命周期和 flag 代际后才可
实现。阶段数量本身不是优化目标；应以有效重叠、GM 往返、L1/UB 占用和 kernel 时间为判断依据。

## 8. 候选模式的选择与比较

设计阶段为每个算子记录当前候选，例如：

```text
sync_pattern: SYNC_GM_QUEUE
pipeline_pattern: PIPE_SERIAL
status: baseline
```

尝试优化时复制设计并标记为 `candidate`，例如：

```text
sync_pattern: SYNC_L1_HANDOFF
pipeline_pattern: PIPE_PACK_OVERLAP
status: candidate
```

该候选仅用于已通过组件能力核验的 A5 路径；A2/A3 的对应候选继续使用
`SYNC_GM_QUEUE`，可独立比较 `PIPE_PACK_OVERLAP` 的收益。

候选必须在相同接口、shape、dtype、tiling 约束和测试集下比较：

1. 编译、运行和 sanitizer 检查是否通过；
2. 全量精度和尾块用例是否通过；
3. 是否有未初始化读取、越界、死锁或 flag 不平衡；
4. UB、L1、workspace、EventID 和 CrossCore flag 占用；
5. 相同模型 shape 下的 kernel 延迟和稳定性；
6. 代码复杂度以及失败后的基线回退成本。

只有候选通过上述检查并优于基线时，才将其状态改为 `adopted`，并把选定模式和实测依据写回
`docs/design.md`。算法或接口变化（例如 KKT owner 共享、Cube Score 分解）不在本文件中直接升级为
默认模式，必须先完成独立的语义和精度评审。

[^catlass-arch-separation]: git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c; paths: README.md, include/catlass/arch/arch.hpp, include/catlass/gemm/tile/copy_l0c_to_gm.hpp, include/catlass/gemm/tile/copy_l0c_to_ub.hpp, include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp

[^pr1069-kernel-stage-sync-patterns]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/kernel-stage-sync-patterns.md
