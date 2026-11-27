# Winograd 实现：变换、数据搬运与混合核流水

数学模型见 [scene §2.8](../../../generic/filter/scene.md#28-winograd-等价模型f23-变换域求导)，准入和 workspace 预算见 [Winograd Tiling](../../../generic/filter/tiling/templates/winograd/tiling.md)，编译与字段绑定见 [tilingkey-templates](tilingkey-templates.md)。本页沿用 WinoDyFwdTransformer、WinoFmapFwdTransformer、WinoMMAD、WinoInvTransformer 的阶段名。输入和输出为 FP32，A 路始终为 dedy、B 路始终为 x。

## 1 五段数据流

```text
+-----------------------------------------------------------------------------+
| NCDHW: x / dedy                                                             |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+--------------------------------------+--------------------------------------+
| WinoPreTransData: AIC MTE2 -> L1 -> MTE1 -> AIV UB                          |
| AIV 子块 0: MTE3 -> GM 预转换区 (d,n,c1,h,w,c0)                             |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+--------------------------------------+--------------------------------------+
| AIV: WinoFmapFwdTransformer / WinoDyFwdTransformer                          |
| 驻留侧 -> GM nk1c1k0c0 -> AIC MTE2 -> L1A/L1B                               |
| 流式侧 ----------------> AIV MTE3 -> L1A/L1B                                |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+--------------------------------------+--------------------------------------+
| AIC: L1A/L1B -> MTE1 -> L0A/L0B -> 16 点 MMAD -> L0C                        |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+--------------------------------------+--------------------------------------+
| Fixpipe -> AIV UB -> WinoInvTransformer                                     |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+--------------------------------------+--------------------------------------+
| 主轮 -> dedw                                                                |
| 尾轮 -> tailGm -> K 组聚合 -> dedw                                          |
+-----------------------------------------------------------------------------+
```

预转换不是单独的 AIV 转置：AIC 先用 DN2NZ 搬入 L1，再搬 UB，AIV 子块 0 写出 GM；配对 AIV 仍参加通知。两路转完后 `WinoPreTransData::End` 汇合，再进入变换流水。输入保持只读。

### 1.1 tile 与变换坐标

```text
x: 4x4 感受野                             dedy: 2x2 tile
+-----+-----+-----+-----+                 +-----+-----+
| x00 | x01 | x02 | x03 |                 | g00 | g01 |
| x10 | x11 | x12 | x13 |                 | g10 | g11 |
| x20 | x21 | x22 | x23 |                 +-----+-----+
| x30 | x31 | x32 | x33 |                       |
+-----+-----+-----+-----+                       |
            |                                   |
 WinoFmapFwdTransformer               WinoDyFwdTransformer
            |                                   |
            v                                   v
          x[p]                               dedy[p]
            |                                   |
            |                                   |
            +-----------------+-----------------+
                              |
                              v
           p=0..15 各做 co x ci 乘加，沿 tile 归约
                              |
                              v
                     WinoInvTransformer
                              |
                              v
                   dedw[co,ci,kd,0:3,0:3]
```

图中的 g00 等表示 dedy tile 元素。两种正变换的系数和窗口不同，不能共用一个未经区分的 Fwd。p 是变换坐标，不是频率。每个 kd 独立产出 3×3 梯度，od 被折叠进归约坐标；边缘 tile 按真实 x/dedy 边界补零。

### 1.2 排布与地址

```text
+--------------+----------------------------+----------------------------+
| 对象         | 由慢到快的轴序             | 索引说明                   |
+--------------+----------------------------+----------------------------+
| 预转换区     | (d,n,c1,h,w,c0)            | x 用 id，dedy 用 od        |
| GM 驻留区    | (nFold,k1,c1,p,t,c0)       | nFold=od*batch+n           |
| L1 操作数    | (c1,p,t,c0)                | p=0..15，t 为块内 tile     |
| L0C          | (p,co,ci) 的分形视图       | 16 个独立结果片            |
+--------------+----------------------------+----------------------------+
```

`c0=32/sizeof(SrcT)`，当前 FP32 为 8；它与普通 MN_StreamK 数学通道窗口按 16 对齐的粒度分别记录。`NK1C1K0C0::Shape.k0=16*th*tw`（驻留区的打包长度，区别于公共 TilingData 的 k0=8），拆为 p 与块内 t 后，GM 驻留区偏移为：

`off=(((nFold*k1Count+k1Idx)*c1Count+c1Idx)*k0+p*(th*tw)+t)*c0+c0Idx`。

L1 只紧致存储本块有效 tile，采用 `(c1,p,t,c0)`。Load2D 按 p 选择当前矩阵，矩阵 A 为 co×tile，矩阵 B 为 tile×ci；相同 p、od、batch 和 tile 必须配对。workspace 四区依次为 transX、transDy、nk1c1k0c0、tailGm，不能把 Host 求和预算的书写顺序当成设备地址顺序。

## 2 跨核信用队列

CVSyncQue 区分生产 Src、消费 Dst、归还 Pop 三条流水。WaitSlot→写数据→EnQue 与 WaitData→读/用数据→DeQue 配对；初始信用表示生产者在等待归还前能提交几次通知，不是额外分配的物理槽。

| 实例 | Src→Dst→Pop | 信用 | 就绪/归还 flag |
|---|---|---|---|
| FwdTransformGM2L1Queue | MTE3→MTE2→MTE2 | 12 | 1/2，另用 0 做 AIC 汇合 |
| FwdTransformUB2L1Queue | MTE3→MTE1→MTE1 | 2 | 3/4 |
| InvTransformL0C2UBSyncQueue | FIX→V→MTE3 | InvTransformBufCnt | 5/6 |

```text
生产者                              消费者
WaitSlot                            WaitData
    |                                   ^
写数据 -> EnQue -------- ready ---------+
    ^                                   |
    +--------------- free ------------ DeQue
                                  最后消费者之后
```

GM2L1 的 WaitData 先在 PIPE_FIX 等 AIV 通知，再以 flag 0 汇合 AIC 并让 MTE2 等待；该汇合只在本轮需要补驻留时执行。C2V 队列按实际 AIV 数逐个发通知，当前实现用 MODE_4，子块 flag 偏移为 16。

flag 按模式、方向、pipe 和生命周期登记。当前逆变换交织使用 MODE0 flag 7；不要把它改为 SyncAll 内部使用的 11～14。预转换与主变换的事件在切换前结束；目标 SDK 更换时重新核对保留 flag 和 MIX 配比。

## 3 AIC 计算流水与 Buffer

### 3.1 预取顺序

```text
顺序       MTE2 驻留侧装载                    MTE1 / MMAD
  0        LoadL1(0, ping)
  1        LoadL1(1, pong)                    Compute(0, ping)
  2        LoadL1(2, ping)                    Compute(1, pong)
 ...       先提交下一块的装载                  再计算当前块
 last                                        Compute(last)
```

RunMmad 先装当前块，在循环内先提交下一块 LoadL1，再计算当前块，末次仅计算最后一块。无效 D 步不翻转有效装载的配对关系；`firstK` 只在本段首个有效 Compute 后清除。

### 3.2 物理分区

```text
L1:       半区 0                             半区 1
          [L1A: dedy][L1B: x][对齐预留]       [L1A: dedy][L1B: x][对齐预留]

L0A/B:    [slot 0][slot 1][slot 2][slot 3]
          每槽容纳 pointsPerGroup 个变换点；循环覆盖 16 点

L0C:      [p0][p1] ... [p15]
          每片独立累加 co x ci，片跨度为 L0CBytes/16
```

L1 每路的编译预留长度为 `16*th*tw*最大通道档` 元素，ping/pong 以 L1Bytes/2 分开；末分形额外读取和两路末地址也须在半区内。L0 输入采用 L0_BUF_CNT=4；按目标 L0A/L0B 可用空间选 pointsPerGroup∈{1,2,4,8}，保证 16 能整除。当前源码以 L0A 作为两路共同上限，复用时核对 L0B 同容量，或另行收紧候选。一次 MMAD 的 k=tiles.elements，没有另一个 baseK 循环。

L0A/L0B 与 L0C 不能共用预算。编译期 TOTAL_* 容量必须对应本次目标，并与 Host 的平台信息一致。同步为 L1 的 MTE2↔MTE1、L0 槽的 MTE1↔M、L0C 的 M↔FIX。

### 3.3 Fixpipe→UB

Fixpipe 按 Cout 段搬出，ndNum=16；srcNdStride 对应一个 L0C 点片，dstNdStride 对应 UB 一个点片。双 AIV 时 mSize 向偶数对齐、dualDstCtl=1；nSize 按 `C0<float>()` 对齐。AIV 只写真实 co/ci，UB 的 alignedCin 间隔不进入 dedw。

M_FIX 保证结果可读；完成搬出后发布 FIX_M 归还 L0C。L0C2UB 队列还单独保护 UB 槽，不能用 FIX_M 代替 AIV 的 DeQue。

## 4 AIV 正逆变换与 UB 分时

```text
UB 同一地址区，按阶段复用：
+---------------------------------------------------------------+
| 正变换：src/out 槽轮转，MTE2 -> V -> MTE3                     |
+---------------------------------------------------------------+
                                |
                   源/出槽消费完成，停止正变换
                                v
+---------------------------------------------------------------+
| 逆变换：[16 点输入][9 点还原输出] x InvTransformBufCnt        |
+---------------------------------------------------------------+
                                |
                  BlockMTE2ByMTE3：搬出后才恢复装载
                                v
                           下一正变换阶段
```

正变换使用各槽 MTE2↔V、V↔MTE3 事件。逆变换每槽占 25 个 Cout×Cin 点片，16 片接收 Fixpipe，9 片保存还原结果。它独占相应 UB 地址期，CopyOut 完成后通过 MTE3_MTE2 依赖允许复用。

段内在变换域累加，段间在逆变换后的 FP32 结果上合并：首段覆写 dedw 或 tailGm，后续段 SetAtomicAdd，搬出后恢复 SetAtomicNone。尾轮的最终聚合不把各 kGroup 直接原子加到 dedw。

## 5 驻留边界与 D 轴有效性

驻留切分与输出块迭代都消费入口唯一的 rtTiling。residentCBound、水位线及 TaskId 的通道区间必须一致，不能一处按模板 64、一处按运行时 16/32/48 切块；缩窄后的块内任务数按实际剩余通道截断。

```text
固定 kd，遍历同一 K 单位 (od,n,hBlock)，再走全部 wBlock
                             |
                 WinoIsValidD(od,kd,di,padFront)
                             |
           +-----------------+-----------------+
           | true                              | false
           v                                   v
  正变换 / L1 装载 / MMAD              不装载，不做 MMAD
           |                                   |
           +-----------------+-----------------+
                             |
               需要补驻留时，GM2L1 握手次数不变
```

`id=od+kd-padFront` 在 `[0,di)` 内才有效；padFront<kd 长度不意味着所有 od 都有效。AIC 的 NullGm2L1Handshake 与 AIV 的 WaitSlot/EnQue 配对，避免全核 barrier 次数因 pad 跳过而失配。闲置输出核使用 RunMmad<false>，只裁掉真实装载和 MMAD；有效 D 步仍消费 AIV 发出的 UB2L1 空通知，不能连同数据计算一起删掉。L0C2UB 则只为非空输出块生产。

| 本步状态 | GM2L1 | UB2L1 | 段末 L0C2UB |
|---|---|---|---|
| D 有效，输出块非空 | 需要补驻留时配对 | 正常生产/消费 | 正常搬出与逆变换 |
| D 有效，输出块为空 | 需要补驻留时配对 | 空任务仍 WaitSlot/EnQue 与 WaitData/DeQue 配对 | 跳过 |
| D 无效 | 需要补驻留时空握手 | 两端均跳过本步 | 输出块非空时仍按段交接；全无效段在 UB 清零 |

若整个 segment 无有效 K，仍保持 Fixpipe 与 L0C2UB 的正常通知；AIV 在 WaitData 后 Duplicate 清零收到的 16 点输入，再做逆变换。这样首段写零、后续段加零，也不会因单侧跳过搬出导致死锁。

## 6 尾轮切 K 与 kd 子轮隔离

每个 kGroup 处理相同尾块的不同 K 区间，部分和写 `tailGm[(tailBlockId*kGroups+kGroupIdx)*TailBlockSize]`。TailBlockSize 使用编译的 64×64×9 FP32 上限；区域内按真实 cin 紧致存放，UB 的行补齐必须跳过。

```text
K 组 0 -> tailGm[tailBlockId,0]              ----+
K 组 1 -> tailGm[tailBlockId,1]              ----+
K 组 q -> tailGm[tailBlockId,q]              ----+
                                                 |
                                                 v
                      +-----------------------------------------------------+
                      | 屏障 A：部分和写入完成                              |
                      +--------------------------+--------------------------+
                                                 |
                                                 v
                      +--------------------------+--------------------------+
                      | 按 Cout 分配最终写回责任                            |
                      +--------------------------+--------------------------+
                                                 |
                                                 v
                      +--------------------------+--------------------------+
                      | 每位负责者读取全部 K 组并求和                       |
                      +--------------------------+--------------------------+
                                                 |
                                                 v
                      +--------------------------+--------------------------+
                      | CopyOut -> dedw 对应 kd 切片                        |
                      +--------------------------+--------------------------+
                                                 |
                                                 v
                      +--------------------------+--------------------------+
                      | 屏障 B：聚合读和写回完成                            |
                      +--------------------------+--------------------------+
                                                 |
                                                 v
                      +--------------------------+--------------------------+
                      | 下一 kd 子轮复用 tailGm                             |
                      +-----------------------------------------------------+
```

kd>1 时，屏障 A、B 都是 `PipeBarrier<PIPE_ALL>(); SyncAll<false>();`；A 保护聚合读，B 等聚合及写回结束后才允许下一子轮覆写。TailInterleaveWrite 内还有 MODE0 MTE3→MTE2 交接，不能代替这两处跨子轮屏障。

`kLength<kMaxLength` 且本轮需要驻留变换时，短 K 组按 wSteps 补 GM2L1 空通知；额外闲置核并入最后一个 kGroup，维持集群同步。最终写回由各组分担 Cout 区间、仅 AIV 子块 0 执行；每位负责者汇总所有 K 组的数据，而不是只写本组部分和。

## 7 产出检查

- [ ] 预转换两核职责、D 最外层轴序和结束通知一致；四段 workspace 起点正确。
- [ ] 两路正变换按相同 p/tile 配对，L1 紧致块与 GM 驻留块的步幅分别使用。
- [ ] rtTiling 统一、kd 切换清水位；D 轴无效步和闲置核保留正确次数的握手。
- [ ] 段首 firstK、全无效段 UB 清零、首段覆写和后续段原子合并均成立。
- [ ] L1 半区、四份 L0 输入槽、16 份 L0C 及 UB 分时均按目标容量验算。
- [ ] M_FIX/FIX_M、三条信用队列和 MODE0/SyncAll flag 的生命周期完整。
- [ ] 尾轮读取全部 kGroups、Cout 写回互斥，kd 子轮前后屏障都存在。
- [ ] L2 旁路按目标 L2 容量评估；参考实现的容量常数和经验阈值不直接当作通用能力。
