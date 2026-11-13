# MN_StreamK Tiling 模板（M/N 输出任务与核内分段归约）

本模板按 M/N、核深度和组分配输出，每个任务完成完整归约。共用计算见 [Tiling 入口](../../tiling.md)，布局与段循环见 [compute-flow](../../../compute-flow.md)。当前采用 streamkType=0、原始分组和 FP16/BF16 输入、FP32 输出。

## 1 选择条件

| 条件 | 检查 |
|---|---|
| 语义与规格 | x/dedy/filter_size 的形状、属性和分组一致 |
| 数值与格式 | 两路同为 FP16 或 BF16；x/dedy 格式配对，dedw 选择已实现输出格式 |
| 执行能力 | platformInfo 与 SDK 支持两路搬运、窗口、MMAD、FP32 原子写回及初始化通知 |
| 混合核启动 | 所选 AIC/AIV 配比、TPL 核类型、blockDim 与初始化参与单元一致 |
| 求解结果 | L0/L1、字段范围、物理地址与输出覆盖全部合法 |

当前初始化映射按每个 AIC 配两个 AIV 的启动协议检查可用资源；这是一项执行协议要求。具体架构与 SDK 绑定由适配层能力记录提供，不以写死的 SoC 名称代替查询。

## 2 基本块与宽片起点

InitSplitWOI 在可切宽时，通常以 512 为整宽阈值；左右 pad 超普通字段范围时阈值为 256。wo 不大于阈值时使用原 wo/wi；否则以 128 为输出宽片起点，尾片按剩余宽度派生。

M/N 起点由均分到 256 以内并对齐 16 得到；核高宽积为 9 时优先 144 的倍数。K 从两路 L0 两份槽的容量推出，再按 kValue 和 splitWo 调整。所有初值都可能被容量回调改变。

## 3 任务划分与 L1 驻留

UpdateSingleCoreInfo 取 singleCoreM=baseM，singleCoreN 为至少一个完整 `hk*wk*16` 窗口；拆核时使用一个 ci 分形的任务跨度。baseN 可以将任务分为多个子块。

SetStepK4SplitMN 从 depth=16、两路双缓冲开始，按阶梯缩短驻留；不合法时切换单份并重新搜索，必要时 ShrinkBaseBlock。每次改变基本块或窗口路径后重新计算 Bound 和任务。

```text
                      +---------------+
                      | InitSplitWOI  |
                      +-------+-------+
                              |
                              |
                              v
                    +---------+---------+
                    | MultiCoreSplitMN  |
                    +---------+---------+
                              |
                              |
                              v
+-----------------------------+-----------------------------+
| InitBaseBlock -> SetStepK4SplitMN -> UpdateSingleCoreInfo |
+-----------------------------+-----------------------------+
                              |
                              |
                              v
        +---------------------+---------------------+
        | final L0 / L1 / API / task / init checks  |
        +---------------------+---------------------+
                              |
                              |
                              v
       +----------------------+----------------------+
       | publish fields + key + blockDim + schedule  |
       +---------------------------------------------+
```

## 4 Host 求解顺序

```cpp
InitSplitWOI();
if (!MultiCoreSplitMN()) return false;
// 各 try 内：InitBaseBlock -> SetStepK4SplitMN -> UpdateSingleCoreInfo
if (IsCurBlockL1Invalid()) return false;
AdjustSmallCaseBaseBlock();
// 块变化后重新求 baseK、stepKa/stepKb、Bound 和 singleCore*
DoStreamKTiling();
if (streamkType != 0) return false;
singleCoreBatchDout = uint64_t(batch)*dout;
UpdateSingleCoreInfo();
// 序列化 base*、singleCore*、step*、Bound、Pbuffer、splitWo
if (!CheckFinalPlan()) return false;
PublishTilingData();
```

CheckFinalPlan 与 PublishTilingData 为生成工程的检查/发布函数，分别落实共用 Tiling 第 9～10 节及适配层 Host 发布。没有直接调用外部仓库求解器的依赖。

当前模板不执行跨核 K 归约。MN_StreamK 类名保留实现术语，不能仅凭名称启用 streamkType=1/2、workspace 归约或其他任务分发。

`AdjustSmallCaseBaseBlock` 用剩余可并行的 M/N 分段数改善末轮利用率，不是第三种 Kernel 模板。例如 M=1280、N=128、dk=3、G=1、36 个计算核时，M 块由 256 调整为 112，形成 12×1×3=36 个输出任务；K 和 L1 参数随之重算。`DoStreamKTiling` 判定最终运行时配置，当前生成域接收其结果为 0 的计划，不通过强制清零该字段改写准入结论。

## 5 workspace、核类型与执行协议

| 项目 | 约定 |
|---|---|
| 用户 workspace | 0；本配置不访问用户归约区 |
| 平台/系统 workspace | 按所用工程框架与 SDK 处理，和用户 workspace 分开记录 |
| 初始化 | 整个 FP32 dedw 清零，然后通知 AIC |
| 核类型 | 与 AIV 初始化配对的 MIX 模板实例；AIV 清零并通知后返回 |
| 调度 | 使用模式 0 汇合的入口按 SDK 配置 SetScheduleMode(1) |
| 编译分支 | 模板编号 MN_StreamK，isSplitKernelHW 按实际求解结果 |

## 6 数值示例与产出校验

取 NCDHW：`x=(2,64,9,17,17)`、`dedy=(2,96,4,8,8)`、`dedw=(96,64,3,3,3)`，stride=2、dilation=1、pad=0、G=1。推演能力为 AIC=36、L1=512 KiB、L0A/B=64 KiB、L0C=256 KiB。

| 量 | 结果 |
|---|---|
| 选块长度 | mValue=96、nValue=576、kValue=64 |
| L0 基本块 | baseM=96、baseN=144、baseK=64，L0 三路各两份 |
| L1 驻留 | stepKa=stepKb=1，L1 两路各两份 |
| 单份 Bound | al1Bound=6144、bl1Bound=4624，单位为元素 |
| L1 字节预算 | `(6144*2+4624*2)*2=43072` |
| 任务 | singleCoreM=96、singleCoreN=144、singleCoreCin=16、singleCoreK=64 |
| 分发 | totalCnt=12，usedCoreNum=36；核 0～11 各一任务，其余无主计算任务 |

L0A/B/C 含份数的字节占用分别为 24576、36864、110592。一个任务覆盖 96 个 co、16 个 ci、一个 kd；N 的 144 个位置由 9 个核位置×16 个通道组成。

```text
+-----------+----------------------------------------------------+
| block_idx | task (mIdx,nIdx,kdIdx,gIdx)                        |
+-----------+----------------------------------------------------+
| 0         | (0,0,0,0)                                          |
| 1         | (0,0,1,0)                                          |
| 2         | (0,0,2,0)                                          |
| 3         | (0,1,0,0)                                          |
| ...       | ...                                                |
| 11        | (0,3,2,0)                                          |
| 12..35    | no main task; still follow initialization protocol |
+-----------+----------------------------------------------------+
```

另取 `batch*dout=128、ho=wo=32`，CalcReduceSegments 得到 ND 段长 64、H 段长 32，两个归约段写同一个输出块。初始 dedw 预填非零时，初始化后再运行应与零基值 Golden 一致。

示例是规则复算，不代表设备实测。产出记录保留规格、能力、选中路径、最终字段、资源、任务与归约覆盖；实际 Host 返回值和设备结果分别验收。

## 7 检查要点

- [ ] 分支选择、普通/拆核窗口与切宽路径有实际字段记录。
- [ ] 基本块、驻留、任务和 Bound 都来自同一最终计划。
- [ ] 任务 N 与计算 N 区分，非整窗块存在对应写回实现。
- [ ] 启动 MIX、初始化等待、调度模式和 usedCoreNum 配对。
- [ ] 用户 workspace=0 的设备路径没有访问未分配归约区。
- [ ] 推演示例、Host 求解、设备验证分别记录，不相互替代。
