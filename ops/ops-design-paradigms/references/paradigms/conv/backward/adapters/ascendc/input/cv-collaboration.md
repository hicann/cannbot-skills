# C/V 核间协同（AIC/AIV 协作）

> **层级**：input 分支适配层（adapters/ascendc/input/）。本文只维护跨核的**参与者、flagId 与令牌生命周期**：三种数据准备的布局和准入见 [指令映射 §3.4–§3.8](instruction-mapping.md)，子核状态机见 [kernel-split 实现](kernel-split-implementation.md)，核内队列与事件见 [同步实现](sync-implementation.md)。

## 1 参与角色与核型

| 场景 | AIC | AIV |
|---|---|---|
| VecTranspose | 等预处理完成后从 workspace 装载 weight | 全部参与 AIV 分摊转置任务；无任务者仍参与完成通知 |
| C04 | 四通道 A 路窗口；等 L1B 就绪后装载 L0B | 单子块直接准备；双子块按 c04LoadToB1IterIdx_ 奇偶轮换，均走所选握手协议 |
| group enlarge | 按计算组求解，消费块对角 weight | 仅子块 0 负责 L1B 准备 |
| split-HW | 计算各子核，Fixpipe 输出到 UB/workspace | 仅子块 0 执行相位交织与 dedx 写回 |
| split-H | 计算固定 H 相位，按行相位直接写回 | 按配置参与初始化和 VecTranspose；不执行重排 |

- **核型配对**：有 AIV 工作的实例使用目标配置对应的 MIX（AIC_1_1 或 AIC_1_2）。编译常量 CONV3DDX_CV_RATIO 绑定该配置，Host、设备核型及参与者一致。`CheckVectorCoreNum` 在适用目标且 NeedVectorCore=true 时检查 AIC 非零及所需 AIV 数；最低比例按该目标执行契约取 1 或 2。`GetCVRation` 的资源比例用于任务预算，不能据此单独决定编译核型。
- **AIC 数发布**（仅 enableVecTrans）：转置任务总数 `cout*ceilDiv(cin,16)`，所需 AIC 数 `ceilDiv(任务总数, CVRatio)`，与 cube 任务数取大后限制到可用 AIC 数。统一用总通道口径（与 workspace 布局一致；当前浮点 VecTranspose 准入 groups=1，cinG=cin、coutG=cout）。其他分支不套用此公式。

参与者按同一目标配置绑定：

| 执行配置 | C04 搬运者 | C04 通知与核号 |
|---|---|---|
| 分离、每 AIC 两个 AIV | 子块号等于 `c04LoadToB1IterIdx_&1` 的子块 | 两者握手，AIC 等两组通知；AIV 计算核号由配套 GetAicBlockIdx 映射 |
| 分离、每 AIC 一个 AIV | 唯一子块 | 仅一组通知；不等待不存在的第二组 flag |
| C/V 融合 scalar | 按编译配置执行同一 scalar 的对应阶段 | 使用 TQueSync 和融合核号，不能照搬分离式编号/跨核汇合 |

GetAicBlockIdx、GetSubBlockIdx、奇偶轮转及第二组 flag 的编译条件须成套保留；不是把所有硬编码的 2 机械替换为运行时比例。

## 2 同步原语

`CvCrossCoreSet/Wait` 保持源、目的流水两个模板参数，按编译分支二选一：

| 编译分支 | Set | Wait |
|---|---|---|
| 融合（`__CUBE_VECTOR_FUSION_ONLY__`） | `TQueSync<srcPipe,dstPipe>().SetFlag(flagId)` | `TQueSync<srcPipe,dstPipe>().WaitFlag(flagId)` |
| 分离 | `CrossCoreSetFlag<SYNC_MODE, srcPipe>(flagId)` | `CrossCoreWaitFlag<SYNC_MODE, dstPipe>(flagId)` |

同形态的双分支写法另见 [同步实现](sync-implementation.md) §3.2。⚠️ flagId 不是完整的事件身份——模式、参与核、流水与收发方向也要一致；初始化的首个令牌、主循环交接和结束时的剩余令牌回收必须一起实现，不能只抄中间的 set/wait。

## 3 三种握手模式

三种场景的协同都是同一条**"令牌 → 就绪 → 消费 → 归还"回路**，差异只在数据方向与 flagId：

| 模式 | 场景 | 谁通知谁 | flagId | 首末令牌 |
|---|---|---|---|---|
| 一次性全体完成 | VecTranspose | 全部 AIV 完成转置 → 通知全部 AIC | MODE0：`SYNC_AIV_ONLY_ALL_DET_FLAG=2`（AIV 全体完成）；MODE2：`SYNC_AIV_AIC_DET_FLAG=6`（MTE3→MTE2） | 无转置任务的 AIV 也通知；无计算任务的 AIC 由 ProcessForUnUsedCore/CrossCoreWaitVecTrans 消费通知 |
| 段级 L1B 交接 | C04 / group enlarge | AIC 归还 → AIV 准备 → AIC 就绪消费 | `FLAG_MTE1_ID_1`（就绪）/ `FLAG_MTE1_ID_2`（复用）；C04 双子块配置另有 +`CROSS_CORE_FLAG_ID_MAX` | 每个参与通道各有初始复用令牌；B1 最后一次 MTE1 读取完成后归还 |
| 输出缓冲交接 | split-HW | AIC Fixpipe → AIV 交织写回 → 回告 | 就绪 `FLAG_MTE2_VEC_ID`（workspace 路径目的流水 PIPE_MTE2，UB 直达路径 PIPE_V）；回告 `FLAG_FIXP_ID`（MTE3→FIX） | 初始 output free 一次；W 相位双缓冲（W0→buffer0、W1→buffer1）成对回合后归还 |

通用时序（三模式的公共形状）：

```
AIC                                                           AIV
 |                                                            |
 | 初始令牌（可复用/可写） ---------------------------------->|
 |                                                            | 等令牌 -> 准备数据（MTE2->V->MTE3）
 |<------------------------------------------------- 就绪通知 |
 | 等就绪 -> 消费（MTE1 / FIX）                               |
 | 归还令牌（末次读取完成后） ------------------------------->| 下一轮
```

各模式的专属细节：

- **VecTranspose**：转置任务分给全部参与 AIV（不限每对核的子块 0）。融合编译分支用其完成检测和 TQueSync，不照搬 MODE0；分离路径保留 Host 的 `SetScheduleMode(1)` 使 MODE0 参与核一起到位，直调侧按目标运行时落实调度约束（避免多 stream 交错占用参与核）。
- **C04 / group enlarge 的包装函数**：

| 包装函数 | 用途 |
|---|---|
| WaitForVecBeforeLoadToB2 | AIC 等 `FLAG_MTE1_ID_1`；C04 双子块配置再等第二通知 |
| NotifyVecAfterLoadToB2 | AIC 发布 `FLAG_MTE1_ID_2`；C04 双子块配置发布配套第二旗 |
| WaitForCubeBeforeLoadToB1 / NotifyCubeAfterLoadToB1（C04） | 单子块直接准备；双子块都交接令牌，仅奇偶命中的子块搬运 |
| GroupTransdataWeightCore（group enlarge） | 仅子块 0 等复用、清零、整理并发布就绪 |

  C04 的 CrossCoreSetHeadForMix/CrossCoreWaitTailForMix 与 group enlarge 的 CrossCoreSetHead/CrossCoreWaitTail 各自配对；融合模式下 group enlarge 双 B1 buffer 还有额外令牌，收尾不能写成单次固定 wait。B1 可在一次驻留的多个 K 计算间复用；跨任务长期保留的普通 FullLoad 条件排除 C04 与 vector group enlarge。
- **split-HW**：1×1 stride2 的无贡献相位、bias 路径和空间尾部按子核状态机处理，通知次数按实际重排回合核对（不能把每个子核当成一次独立通知）；首令牌和末次回告由 head/tail 协议闭合，时序详见 [kernel-split 实现](kernel-split-implementation.md) 第 5 节。

## 4 workspace 与资源

```
user workspace
+----------------------+---------------------------+
| VecTranspose weights | split-HW 重排槽           |
| 可选，字节范围 [0,V) | 可选，每个 AIC 一个预留槽 |
+----------------------+---------------------------+
                       ^ 重排基址 = workspace + V
```

- V 为转置权重字节数；重排槽按输出 DstT 字节数预算，L0C 按 FP32 累加器预算，两者不能混用。总量与分段见 [指令映射 §3.5](instruction-mapping.md) 与 [kernel-split Tiling §8](../../../generic/input/tiling/templates/kernel-split/tiling.md#8-子核对齐workspace-与-vector-资源)。
- workspace 在 enableVecTrans 或 kSUseWorkSpace 置位时必需；kSCoutFullLoad 单独置位不要求。外层调用保持输入、输出、workspace 到 stream 完成；输出初始化必须在计算和重排写回前完成。

## 5 检查要点

- 按场景核对 AIV 参与者：全部 AIV、仅子块 0、或 C04 单/双子块协议，不能统一提前返回。
- 对照 head→稳态→tail 逐条计算通知与等待次数，包含无任务核、尾块和无贡献相位。
- 区分权重 workspace 的完成检测与 L1B 段级握手，二者不能互相替代。
- workspace 各区偏移、元素/字节单位和实际槽地址一致，重排不得覆盖转置权重。
