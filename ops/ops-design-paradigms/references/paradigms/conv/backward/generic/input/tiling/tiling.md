# 卷积反向算子 tiling 切分算法

> **层级**：通用知识层（generic/input/tiling/），语言中立；本文件是 tiling 文件夹的入口文档，索引四个求解模板。
>
> **分工**：本文件回答"**切什么、为什么切、四个模板怎么选、共用哪些计算规则、给 Kernel 哪些参数**"。standard、small-shape、l1-resident 共用计算主干；kernel-split 扩展相位循环与写回。完整求解流程在各模板中展开。数据怎么流见 [compute-flow.md](../compute-flow.md)。

---

## 1 总览：tiling 切什么、为什么切

Tiling 决定一次计算的大小（L0 基本块 baseM/baseN/baseK）、L1 驻留范围（stepKa/stepKb 与五路 Buffer 份数）、以及每个输出任务的范围（singleCoreM/Cin/Din）。三个尺度分别使用：`singleCoreM` 决定任务起点，`baseM` 决定核内块容量，`baseUseM_/mCompute` 决定当前有效/对齐跨度；核深度在 K 循环之外完整归约。

五路 pbuffer 份数（al1/bl1/al0/bl0/cl0Pbuffer，取 1 或 2）即**乒乓（双缓冲）开关**：

```
单缓冲：装载 -> 计算等待 -> 计算 -> 装载等待 -> 装载 ...
        串行，搬运期间计算等待。

双缓冲：
+------+-------+--------+--------+-----+
| 阶段 | 起步  | 稳态 0 | 稳态 1 | ... |
+------+-------+--------+--------+-----+
| 搬运 | 装载0 | 装载1  | 装载2  | ... |
| 计算 | 等待  | 计算0  | 计算1  | ... |
+------+-------+--------+--------+-----+
装载与计算重叠，起步阶段等待。
```

- 搬运（MTE2/MTE1）与计算（M/F）使用异步硬件队列；双缓冲允许准备下一块时保留当前块。各层槽份数共同约束重叠，某层单份会限制该层复用，不等同于全部流水串行；
- L1 空间紧张时按「容量回调」（第 7 节）把 L1 份数从 (2,2) 退到 (1,1)：用流水换块大小，取舍由容量不等式裁决，不靠改标志扩大准入。

约束自底向上：L0 容量决定基本块上限，L1 容量决定驻留与任务规模，核数决定任务分发。普通模板的候选超限时按「LegalProtection：容量回调」收缩；专用模板使用自身求解。最终分支与 tilingKey 按所选模板在适配层绑定。

---

## 2 模板选择

先完成规格和能力检查，再按以下优先序进入模板。普通三模板在最终块形状上选择 loadB2Condition；kernel-split 使用自身的固定分支。

Conv3DTranspose 完成自身形状处理后复用本选择器。small-shape、l1-resident、standard 三种模板均调用「可选 bias 的候选预算与实际分配」的公共 L1 检查，并在最终任务确定后复核 bias/BT；bias 不作为独立的 Tiling 策略。kernel-split 改写问题几何（核窗口按 stride 相位拆分），在自身容量模型中计入 bias/scale 预算。

### 2.1 选择顺序

| 优先序 | 模板 | 选择条件与职责 |
|---|---|---|
| 1 | [kernel-split](templates/kernel-split/tiling.md) | stride≥2 且核/步长比满足子核准入，核窗口按 stride 相位拆分为子核族 |
| 2 | [small-shape](templates/small-shape/tiling.md) | K 上限及小任务量条件成立，联合搜索 M/N；可组合完整 K 驻留及收益回退 |
| 3 | [l1-resident](templates/l1-resident/tiling.md) | 前项未命中，完整 K 权重驻留的几何、收益与容量初筛成立 |
| 4 | [standard](templates/standard/tiling.md) | 其余符合本计算路径的规格，经验选块与任务调整 |

条件在各模板的准入章节定义。未命中的模板按登记顺序继续检查；已选模板的容量回调仍使用自身规则，不通过修改标志改变准入结果。

```text
Normalized problem + platform capabilities
    |
    +--> kernel-split: cout-full / regular candidate
    |        |
    |        +--> admitted plan ----------------------+
    |                                                 |
    +--> small-shape -> l1-resident -> standard       |
             |                                        |
             +----------------------------------------+
                                                      |
                                                      v
                                    final blocks / residency / tasks
                                                      |
                                                      v
                                       data preparation + key binding
                                                      |
                                                      v
                                      fields / launch / workspace
```

### 2.2 模板与编译分支

先选择求解模板，再根据最终参数绑定编译分支。同一 key 可以服务不同求解模板，enableFullLoad 等运行时字段继续决定执行行为。具体支持范围与绑定见 [Ascend C](../../../adapters/ascendc/input/tilingkey-templates.md) 或 [DSL](../../../adapters/dsl/input/tiling.md)。

| 数据准备或几何分支 | 对计算与绑定的影响 |
|---|---|
| 普通布局 | FP16/BF16 按格式、核体积与最终 baseN 选择 weight 变换；FP32 固定 B2 仅逆序，单位核直接取块 |
| VecTranspose | 权重来源改为预转置 workspace，B2 仅逆序 |
| C04 | 改写两路 K 排列，AIV 先整理 weight；B2 不转置不逆序 |
| group enlarge | 按计算组通道求解，groupConvMode=1，B2 仅逆序 |
| kernel-split | 改写 M/K 与输出相位，kernelSplitMode=1/2，B2 仅逆序 |

前三种数据准备分支的模型见 [scene](../scene.md#3-数据准备与分组合并)，准入和实现见 [指令映射 §3.4–§3.8](../../../adapters/ascendc/input/instruction-mapping.md)；最终编码见 [Tiling 绑定](../../../adapters/ascendc/input/tilingkey-templates.md)。

### 2.3 扩展方式

仅改变驻留范围、M 任务或 Buffer 份数时，优先使用现有模板和公共 Kernel 的条件分支。N/K 全载按 l1-resident 的参数组合处理，不另设入口。

新增独立求解策略时增加 `templates/<name>/tiling.md`，说明选择条件、起点、求解步骤与参数，再在「模板选择」登记优先序。共用公式维护在本文档；计算/驻留行为变化同步更新 compute-flow 和语言适配层。

---

## 3 输入与输出

输入为前置处理后的几何、dtype/format、存储步幅，以及目标芯片版本、platformInfo 和 SDK 对应的核数、L1/L0/BT 可用容量。容量以 `L1Bytes/L0ABytes/L0BBytes/L0CBytes/BTBytes` 传入，同一份记录用于 Host 求解与 Kernel 分配。输入元素按 dtype（FP16/BF16=2B、FP32=4B），累加恒 4B；分形粒度 `m0=n0=16`，`C0=k0` 随 dtype（16 或 8）。

公共求解接收已完成各自形状检查的 DX 或 Transpose 问题。Transpose 使用其前置处理（[preprocessing.md](../../preprocessing.md) 的转置卷积场景），并传入 hasBiasFlag_、biasBytes 和 BT 能力；DX 的 hasBiasFlag_ 为 false。output_padding 已体现为目标尺寸和边界，真实核长与 K 归约长度保持不变。

| 方向 | 有效长度 | 选块长度 |
|---|---|---|
| M | hi*wi | mValue=alignUp(hi*wi,m0) |
| N 方向 | cinG | nValue=alignUp(cinG,n0) |
| K | coutG*hk*wk | kValue=alignUp(coutG,C0)*hk*wk |
| K（C04） | 每个原始组的有效项仍为 coutG*hk*wk，通道容量按 4 | kValue=alignUp(hk*wk*4,C0)；小 cout 的 K 重定义，见 [C04 规则](../../../adapters/ascendc/input/instruction-mapping.md#36-c04配套的四通道窗口与权重) |
| N/K（group enlarge） | cinOpt=enlarge*CinG、coutOpt=enlarge*CoutG；有效 K 为 coutOpt*hk*wk | 放大组口径，nValue/kValue 按打包组对齐（派生见 [场景分支](../scene.md#31-group-enlarge多个原始组合成一个计算组)） |

输出包括计算块 `base*`、任务 `singleCore*`、两路 `stepK*`、五路 Buffer 份数、coreNum、输出初始化要求。普通模板的 loadB2Condition 按最终 baseN 确定；专用模板按各自契约绑定，key 统一由适配层编码。

三种尺度分别使用：`singleCoreM` 决定任务起点，`baseM` 决定核内块容量，`baseUseM_/mCompute` 决定当前有效/对齐跨度。核深度在 K 循环之外完整归约。

## 4 L0 基本块与 AdjustBaseMNK

第 4～7 节的 AdjustBaseMNK、step 求解与容量回调供普通三模板使用。完整 N/K 驻留作为 l1-resident 的参数分支，沿用这些规则，见其 [N/K 全载分支](templates/l1-resident/tiling.md#41-nk-全载与-m-分核)。

基本块为 `(baseM,baseN,baseK)`。较大 M 复用权重，较大 N 复用 fmap 窗口，较大 K 减少循环；同时分别增加操作数/部分和占用，并影响任务数量。经验优选块是搜索起点，最终效果由相同输入、平台上的性能验证确认。

`al0Pbuffer、bl0Pbuffer、cl0Pbuffer` 分别是 fmap 窗口、反向权重、dedx 部分和的 buffer 份数，取 1 或 2。三块存储分别满足：

| 存储 | 容量约束 |
|---|---|
| L0A | `baseM*baseK*dedyBytes*al0Pbuffer <= L0ABytes` |
| L0B | `baseN*baseK*weightBytes*bl0Pbuffer <= L0BBytes` |
| L0C | `baseM*baseN*accBytes*cl0Pbuffer <= L0CBytes` |

将进入本轮的 M/N 记为 `oldM/oldN`，依次调整：

| 步骤 | 计算与条件 |
|---|---|
| 按形状截断 | `baseM=min(oldM,mValue,L0ABytes//(C0*dedyBytes*al0Pbuffer))`；`baseN=min(oldN,nValue)` |
| N 缩小时扩大 M | 当 `baseN<oldN`，取 `mByL0C=max(L0CBytes//(accBytes*cl0Pbuffer*baseN)//n0,1)*n0`，再取 `baseM=min(maxM,mByL0C,mValue)` |
| M 缩小时扩大 N | 使用上一步的 M；当 `baseM<oldM`，取 `nByL0C=max(L0CBytes//(accBytes*cl0Pbuffer*baseM)//m0,1)*m0`，再取 `baseN=min(1024,nByL0C,nValue)` |

`maxM` 在 `baseN<=n0` 时取 512，否则取 1024。扩大 M 前，按本次反向左、上边界复算右、下覆盖量：

- `bpPadRight=wi-((wo-1)*strideW+1)+(wk-1)*dilationW-backpropPadLeft`。
- `bpPadDown=hi-((ho-1)*strideH+1)+(hk-1)*dilationH-backpropPadUp`。

四项 `backpropPadLeft、backpropPadUp、bpPadRight、bpPadDown` 任一大于搬运补边字段上限时，将 maxM 收紧到 512。本映射的字段为 8 位无符号数，上限为 255；反向边界复算使用有符号宽整数。缩小 M 后仍须完成裁边与指令范围检查。容量推算按分形粒度向下取整，当前 `m0=n0=16`。

选定 K 后，若 `baseM*baseN*accBytes*2 <= L0CBytes`，取 `cl0Pbuffer=2`，否则取 1。最终同时检查三路容量、对齐、L1 窗口范围和任务数。

### 4.1 K 窗口配对

`windowSpan` 对应 Host 的 lenHkWkC0：普通布局取 `Hk*Wk*C0`（C0 随 dtype），C04 取 `alignUp(Hk*Wk*4,C0)`。以下窗口整除与权重预算均使用该分支的 windowSpan。当前两路 L0 的容量、元素字节数和 buffer 份数相同，K 上限为：

`maxK = min(kValue, max(C0, alignDown(L0ABytes//(al0Pbuffer*dedyBytes*max(baseM,baseN)), C0)))`。

从 `maxK` 开始按 `C0` 递减，仅检查大于 `C0` 的候选。候选 `candidateK` 满足以下任一条件，即可作为窗口配对结果：

| 条件 | 配对关系 |
|---|---|
| `Wk>1` | `candidateK` 是 `Wk*C0` 的整数倍 |
| `Hk>1` | `candidateK` 是 `Hk*C0` 的整数倍 |
| 完整窗口 | `candidateK` 与 `windowSpan` 互有整除关系 |

取首个满足条件的候选，也就是满足条件的最大 K。没有命中时，保留进入本轮的 `baseK` 与 `maxK` 中较小者。随后按「L0 基本块与 AdjustBaseMNK」更新 L0C 份数并核对三路容量。

`AdjustBaseMNK` 依次完成本节的 M/N 调整、K 搜索和 L0C 份数更新。

## 5 L1 驻留容量

`stepKa/stepKb` 分别表示一次 dedy、weight 驻留供给的 K 计算块数，预留长度为 `stepKa*baseK`、`stepKb*baseK`；`al1Pbuffer/bl1Pbuffer` 是两路 L1 buffer 份数。

两路分别按 stepKa/stepKb 轮 K 计算重装，始终消费同一归约区间。较大的 step 取较小 step 的整数倍；末次按剩余 K 收缩。

**dedy 预算**先计算 M 块覆盖的 dedx 行数：

| `baseM` 与 `Wi` 的关系 | `coveredRows` |
|---|---|
| 两者互有整除关系 | `ceilDiv(baseM,Wi)` |
| 不整除且 `baseM>Wi` | `baseM//Wi+2` |
| 不整除且 `baseM<Wi` | 2 |

再按有效核高和 dedy 展开范围求驻留尺寸：

| 量 | 计算公式 |
|---|---|
| 有效核高 | `Hd=(Hk-1)*δ_h+1` |
| 驻留行数 | `residentRows=min(coveredRows+Hd-1,(Ho-1)*s_h+1)` |
| 驻留通道预算 | `residentChannels=max((stepKa*baseK)//(Hk*Wk),1)` |
| dedy 字节预算 | `dyBudget=residentRows*(Wo*s_w)*residentChannels*dedyBytes*al1Pbuffer` |

宽度预算使用 `Wo*s_w`，dedy_expand 的逻辑宽度为 `(Wo-1)*s_w+1`，二者分别记录。后文的 `驻留行数(spaceLength)` 使用相同规则，只将 `baseM` 换成 `spaceLength`；覆盖行数须结合本方案的空间任务起点使用。

**权重预算**考虑归约块跨通道窗口的额外覆盖。令 `residentK=stepKb*baseK`：

| `residentK` 与 `windowSpan` 的关系 | 覆盖窗口数 `copyLines` |
|---|---|
| 两者互有整除关系 | `ceilDiv(residentK,windowSpan)` |
| 不整除且 `residentK>windowSpan` | `residentK//windowSpan+2` |
| 不整除且 `residentK<windowSpan` | 2 |

权重字节预算为 `wBudget=baseN*copyLines*windowSpan*weightBytes*bl1Pbuffer`，weightBytes 按输入类型取 2 或 4。

`L1合法(stepKa,stepKb)` 表示以下条件全部成立：step 均为正；每路驻留长度覆盖完整 `kValue` 或为 `windowSpan` 的整数倍；**`dyBudget+wBudget+biasBudget <= L1Bytes`**。两路 step 的整除关系及模板驻留要求在候选选择时一并检查。适配层继续核对 buffer 对齐、bank 间隔和最大访问地址。

### 5.1 可选 bias 的候选预算与实际分配

无 bias 时 `biasBudget=0`；有 bias 时 `biasBudget=alignUp(baseN*biasBytes,64)`。每个候选都使用当前 baseN 重算，包含 CalStepK、l1-resident 驻留搜索和 LegalProtection。等分/阶梯的起点规则保持原样，合法性检查统一计入 bias。

`UpdateIsBiasFullLoad` 按 `baseN*biasBytes` 与 BTBytes 选择 isBiasFullLoad。适配层同时核实分块 BT 通路能力。最终任务参数形成后，还须检查真实分配：

| 检查对象 | 字节数或条件 |
|---|---|
| bias L1 与 BT，各一份 | `biasAlloc=alignUp((isBiasFullLoad ? singleCoreCin : baseN)*biasBytes,64)` |
| BT 容量 | `biasAlloc<=BTBytes` |
| L1 实际范围 | bias 槽加入 ping/pong 放置；同时存活区间不重叠，最大末地址不超过 L1Bytes |

候选预算使用 baseN，实际全量 bias 使用 singleCoreCin；两者不同的任务必须复核后者。改变 baseN、singleCoreCin 或 Buffer 份数后重做检查。全量条件与实际容量不一致时继续调整合法计划或返回资源原因，不发布未通过的字段。standard/small-shape/l1-resident 三种公共模板使用同一规则（kernel-split 在自身容量模型中计入 bias/scale 预算，不调用本节公共检查）；`enableFullLoad` 只控制权重驻留。

## 6 CalStepK：分块驻留的候选

standard 与 small-shape 的非全载分支使用下述分块驻留算法；small-shape 的全载分支复用 l1-resident 的完整 K 搜索。

初始取 `al1Pbuffer=bl1Pbuffer=2`。`coutBlocks=ceilDiv(CoutG,C0)` 表示组内 dedy 通道分形数，候选以 `(stepKa,stepKb)` 记录。

**等分容量候选**先给两路各一半 L1，计算每份 buffer 可容纳的通道分形数：

| 量 | dedy 一路 | 权重一路 |
|---|---|---|
| 一个通道分形的字节数 `blockBytes` | `dedyBytes*residentRows*(Wo*s_w)*C0` | `windowSpan*baseN*weightBytes` |
| buffer 份数 `copies` | `al1Pbuffer` | `bl1Pbuffer` |
| 初始通道分形数 `blocks` | `min(coutBlocks,max(1,(L1Bytes//2//copies)//blockBytes))` | 同左，代入本路参数 |

当 `blocks*copies>=coutBlocks` 时，改取 `blocks=ceilDiv(coutBlocks,copies)`。得到 `dyBlocks/wBlocks` 后，保留较大者，将较小者减为不超过原值且能整除较大者的最大整数。

两路 step 换算为：

| 字段 | 计算公式 |
|---|---|
| `stepKa` | `min(max(1,ceilDiv(dyBlocks*windowSpan,baseK)),65535//baseK)` |
| `stepKb` | `max(1,ceilDiv(wBlocks*windowSpan,baseK))` |

`65535` 限制 dedy 驻留长度对应的搬运字段；`stepKa=0` 时该候选无效。随后将较大 step 向下对齐到较小 step 的整数倍，再检查 L1 合法性。

**阶梯搜索候选**的共同起点为：

- `initialK=min(ceilDiv(coutBlocks,al1Pbuffer)*windowSpan,ceilDiv(L1Bytes//dedyBytes,baseN*al1Pbuffer))`。
- `step=ceilDiv(initialK,baseK)`，两路均取该值。

从起点逐次减 1，遇到首个满足 `L1合法(step,step)` 的值即停止；搜索到 1 时也停止，但仍须校验结果。

| 使用位置 | 选择规则 |
|---|---|
| 首次求解：`CalStepK` | 等分候选合法且 step 之和严格大于阶梯候选时选等分，否则选阶梯；最终结果继续检查 |
| 容量回调：LadderMatchStepMNK/EqualL1MatchStepMNK | 先试阶梯候选；不合法时再试等分候选，返回其合法性 |

候选求解使用当前计算块和 L1 buffer 份数。目标启用驻留均衡策略时，还考虑双份 buffer 下的两路装载比例：

| 条件 | 等分候选的比较规则 |
|---|---|
| 未启用该目标策略，或等分 step 比例不超过 3 | 沿用合法性和 step 之和比较 |
| 启用策略、两路均双份、较大 step 超过较小 step 的 3 倍 | 先判断 B 流量是否主导；非 B 主导时选阶梯 |
| 上项成立且 B 主导 | 按重装成本比较；成本相同再比较 step 之和 |

这里 `bTraffic=mValue*baseN/baseM`、`aTraffic=nValue*baseM/baseN`，B 主导指 bTraffic>3*aTraffic。令 `kTaskCnt=ceilDiv(kValue,baseK)`，候选成本为 `bTraffic*ceilDiv(kTaskCnt,stepKb)+aTraffic*ceilDiv(kTaskCnt,stepKa)`。这些是相对搬运估计，策略开关按目标实现或验证记录绑定，不从容量直接推断。最终仍检查容量、窗口配对和实际 buffer 地址。

## 7 LegalProtection：容量回调

L1 容量不足时，先收缩计算块，再调整 buffer 份数。每次候选变化都重算驻留行数、两路 step 和字节预算；重选驻留使用当前模板绑定的 LadderMatchStepMNK/EqualL1MatchStepMNK。

| 顺序 | 调整规则 |
|---|---|
| 1. 减小 K | 从当前 `baseK-C0` 起，按 `C0` 递减至 `C0`，逐个重选驻留；成功时保留 K 和 step，全部失败时恢复进入搜索前的 K |
| 2. 减小 M/N | 令 `baseK=C0`。当 `baseM>max(Wi,m0)` 且 `baseM>baseN` 时，M 减一个 `m0`；否则 N 减一个 `n0`，各自至少保留一个分形。两者均到搜索下限时停止 |
| 3. 恢复较大 K | M/N 缩小并得到合法驻留后，重算「K 窗口配对」的 `maxK`。若等于当前 K，保留结果；否则从 `maxK-C0` 重新向下搜索。成功时保留，失败时结束本轮块收缩 |
| 4. 调整份数 | 本轮块收缩失败时恢复原 `baseM/baseN/baseK`。L1 份数按 `(2,2)→(2,1)→(1,2)→(1,1)` 推进，每次重新执行容量检查与收缩 |

份数回调按当前字段依次判断，共用同一份参数；内层返回后，仍继续检查后续份数条件。整段结束后以最终参数的合法性为准，记录块长、step 和份数的变化原因。

收缩后按所选模板恢复任务，并使用该模板的 step 求解更新驻留。模板标识保持不变，最终复核 L0/L1 容量、窗口配对和有效范围。

## 8 SetSingleCoreInfo：公共任务恢复

`SetSingleCoreInfo` 先估算空间、通道段数，再调整任务范围：

| 量 | 计算公式 |
|---|---|
| 空间段数 | `spaceParts=ceilDiv(Hi*Wi,baseM)` |
| 通道段数 | `channelParts=ceilDiv(nValue,baseN)` |
| 初始任务数 | `initialTasks=N*Di*G*spaceParts*channelParts` |

令 `nPreferred=256、mPreferred=256`（核体积为 1），其余取 `128、512`。

**通道划分**：当任务数不超过核数、`channelParts` 与 `maxCores` 互有整除关系，或 `baseN>nPreferred` 时，重取 `baseN=alignUp(nValue//channelParts,n0)`。任务字段取 `singleCoreCin=baseN`、`singleCoreCout=CoutG`、`singleCoreDin=1`。

**空间划分**：`singleCoreM` 初始为 `baseM`，按下表调整：

| 条件 | 任务空间长度 |
|---|---|
| `initialTasks<=maxCores` | `singleCoreM=alignUp((Hi*Wi)//spaceParts,Wi)` |
| 任务数更多且 `baseM>=mPreferred` | 候选 `wholeRowM=max(baseM//Wi,1)*Wi`；仅当 `wholeRowM*spaceParts>=mValue` 时采用 |
| 其余情况 | 保持 `singleCoreM=baseM` |

若最终 `singleCoreM<baseM`，将计算块收缩为 `baseM=alignUp(singleCoreM,m0)`。本轮使用调整前的段数，结束后重算任务数与尾块。空间段可以从行中间开始，地址仍按原 `Wi` 恢复。

## 9 Host 传给 Kernel 的参数

### 9.1 几何与寻址

Host 将前置处理的统一问题和本模板最终切分结果交给 Kernel。以下表格给出字段名、已有术语和用途；字段的二进制类型、结构对齐和编码由适配层落实。

| 字段 | 与术语的对应关系 | 单位或用途 |
|---|---|---|
| `batch` | `N` | 批次 |
| `cin、cout` | `Cin、Cout` | 总通道数；保留全局存储步幅 |
| `cinG、coutG` | 原始组时为 CinG/CoutG，合并时为 enlarge*CinG/enlarge*CoutG | 计算组通道数，不能重复放大 |
| `group、oriGroup` | 计算组数 real_g、原始组数 G | 合并后的任务数与原始张量寻址分别使用 |
| `di、hi、wi` | `Di、Hi、Wi` | dedx 空间长度 |
| `dout、ho、wo` | `Dout、Ho、Wo` | dedy 空间长度 |
| `dk、hk、wk` | `Dk、Hk、Wk` | 核采样点数 |
| `strideD/H/W、dilationD/H/W` | `s_d/s_h/s_w、δ_d/δ_h/δ_w` | 卷积步长、膨胀率 |
| `padFront/Back、padUp/Down、padLeft/Right` | 正向各轴两侧 pad | 采样与边界定位 |
| `backpropPadUp/Down/Left/Right、backpropPadTail` | 反向边界的实现参数 | 有符号；由适配层完成边界映射与指令范围处理 |
| `c0` | 数学及布局公式中的 `C0`，dedy 通道分形粒度 | 当前为 16 |
| `cin1、cout1、cin1G、cout1G` | 全通道与组内通道的对齐块数 | 块数，按各自粒度计算 |
| `khDilation、kwDilation` | 有效核高、核宽 | `(核长-1)*dilation+1` |
| `hoExpand、woExpand` | dedy_expand 的高度、宽度 | `(dedy长度-1)*stride+1` |
| `hkWk、dkHkWk` | 核高宽乘积、核体积 | 归约寻址 |

数学上的反向补边 `b^-、b^+` 保留在前置处理结果中；实现字段须与 Kernel 的采样、裁边行为共同核对。深度路径使用 `curDinIdx_+p_d^-−kd*δ_d` 定位 dedy，并检查 stride 整除与边界，保持与 `r_d=Dk−1−kd` 等价。

### 9.2 计算、任务与驻留

| 字段 | Kernel 使用方式 |
|---|---|
| coreNum | 循环任务分发，与启动计算核数一致 |
| singleCoreBatch/Group | 当前均为 1，每任务一个 batch、一个组 |
| singleCoreDin/M/Cin | 任务深度、空间、dedx 通道范围 |
| singleCoreCout | 普通路径取 coutG；C04 发布为 4。K 长度按当前分支的排列计算 |
| singleIterateDk | dk，完成所有有效核深度贡献 |
| baseM/baseN/baseK | 输出块容量、L0 计算与尾块 |
| stepKa/stepKb | 两路父段边界、装载频率与实际尾段 |
| al1/bl1/al0/bl0/cl0Pbuffer | 槽分配、轮转与同步 |
| enableFullLoad | Host 驻留策略；Kernel 当前状态另判定 |
| isBiasFullLoad | 可选 bias 的任务通道全量/当前 N 子块准备方式；无 bias 时不分配对应资源 |
| initOutputFlag | 衔接前置初始化与主计算 |
| iterateOrder | 当前为 1，任务解码仍按计算流程中的轴顺序 |
| kSCoutFullLoad | kernel-split 的 dedy/weight 通道全载判定；影响 L1 装载与执行分支 |
| kSUseWorkSpace | kernel-split 输出重排经 GM workspace 中转的开关；AIV 首指令随之切换 |
| enableVecTrans | 前置权重转置开关：AIV 预先把 B 路源布局整理进 workspace，AIC 改从 workspace 读取；standard 族置位时 loadB2Condition 被强制为仅逆序（见 [Tiling 绑定](../../../adapters/ascendc/input/tilingkey-templates.md)） |
| C04 编译配置 | Host 的 enableC04Flag 决定 K 和四通道窗口，通过 loadB1Condition 绑定实例；不增加同名原 TilingData 字段 |
| enlarge | group 放大倍数：>1 时按 real_g=ceilDiv(groups,enlarge) 个放大组求解；由运行字段表达，不另占 key 实参（见 [Tiling 绑定](../../../adapters/ascendc/input/tilingkey-templates.md)；模型见 [场景分支](../scene.md#31-group-enlarge多个原始组合成一个计算组)） |

`mValue/nValue/kValue`、任务计数、起点和尾长可以由几何与最终字段复算。运行 K 使用 tmpSingleCoreK，C04 与 Host 的对齐 kValue 分别处理，见 [分支记号](../language.md#2-mnk-三元组gemm-映射)。dtype/format 和 loadB2Condition 绑定编译实例；无需把模板名再传给 Kernel。字段类型、共享结构和 key 编码见 [Ascend C Tiling 绑定](../../../adapters/ascendc/input/tilingkey-templates.md)；DSL 的编译配置与运行参数见 [DSL Tiling 绑定](../../../adapters/dsl/input/tiling.md)。

发布前检查：每个有效 dedx 坐标唯一归属，所有有效 K/核深度贡献完整；Buffer 预算、物理末地址、API 范围与初始化依赖同时满足。大乘积先用宽整数检查，再转目标字段。

带 bias 时，归约完成或全无贡献的输出均须写入一次 bias。hasBiasFlag_ 与设备可选指针保持一致，最终 bias 槽、BT 能力和 isBiasFullLoad 一并交付。

---

## 10 检查点

- [ ] 模板选择按优先序执行：kernel-split → small-shape → l1-resident → standard；small-shape 与 l1-resident 同时命中取前者
- [ ] 三路 L0 容量不等式（含 pbuffer 份数）同时满足；K 窗口配对与 weight 核几何一致
- [ ] 驻留预算 `dyBudget+wBudget+biasBudget <= L1Bytes`；候选预算用 baseN、实际分配用 singleCoreCin 分别复核
- [ ] 容量回调按"减 K → 减 M/N → 恢复大 K → 调份数"顺序，不通过改标志扩大准入
- [ ] kernel-split 的子核几何、相位起点与 baseK 整除约束按其模板第 11 节检查；失败回落公共选择器
- [ ] 每个有效 dedx 坐标唯一归属一个任务；大乘积先用宽整数检查
- [ ] 带 bias 时 UpdateIsBiasFullLoad 在最终 baseN/singleCoreCin 形成后重做，BT 容量单独核对
