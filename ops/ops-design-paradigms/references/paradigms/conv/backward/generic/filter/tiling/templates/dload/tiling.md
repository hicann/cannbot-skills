# DLoad Tiling 模板（单批空间驻留）

DLoad 将当前通道块的 x、dedy 空间数据搬入 L1，按输出平面窗口组织 L0 计算；一个任务完成所负责 dedw 块的全部 batch 和输出空间归约。准入由采样几何、驻留容量和指令条件决定，batch、通道数与各轴空间尺寸作为在线求解输入。共用记号见 [Tiling](../../tiling.md)。

模板按 DLoad(8) → Winograd(9) → MN_StreamK(10) 的顺序检查。DLoad 使用独立的驻留与计算流程。

## 1 场景与选择条件

| 维度 | 条件与处理 |
|---|---|
| 格式 | x、dedy、dedw 均为 NCDHW |
| 类型与模式 | x、dedy、dedw 为 FP32；hf32Flag 按 enable_hf32 绑定为 0 或 1，目标支持相应 MMAD 模式 |
| 采样几何 | groups=1，stride/dilation 各轴均为 1；各轴两侧 pad 对称，核尺寸和 pad 满足搬运字段范围 |
| 形状关系 | batch、Cin、Cout 及空间尺寸为正；每轴满足 `out=in+2*pad-kernel+1` |
| 空间驻留 | 每个 batch 的当前 Cin 块 x 体积和当前 Cout 块 dedy 体积，可同时放入一个 L1 半区 |
| 计算与写回 | 基本块满足 L0A/L0B/L0C、搬运参数及输出步幅条件；尾通道和无贡献核深度段有完整写回 |

Di、Hi、Wi 分别参与寻址，Dout、Ho、Wo 分别由形状关系确定。batch 在任务内循环，通道通过切块覆盖，空间规模通过驻留预算判定。

## 2 计算块与任务

令 `hkwk=hk*wk`、`dhwk=dk*hkwk`。本模板中 baseM 对应 Cout，baseN 合并 Cin 块与核高宽，baseK 对应一次消费的输出平面位置数：

| 量 | 定义与选择 |
|---|---|
| Cout 块 | `baseM`，按 16 对齐；优选起点 128 |
| Cin 块 | `cinBlock=baseN/hkwk`，按 16 对齐；优选起点 16 |
| N 长度 | `baseN=cinBlock*hkwk`，排列为 `(tap,Cin1,Cin0)`，`tap=khIdx*wk+kwIdx` |
| 平面窗口 | `baseK=kl0HoWo`，按 16 对齐；优选起点 16 |
| 任务数 | `totalBlocks=ceilDiv(cout,baseM)*ceilDiv(cin,cinBlock)` |
| 计算核数 | `coreNum=min(totalBlocks,availableAicNum)` |

优选起点用于生成候选，最终块形状由第 3 节的容量和字段检查确定。容量不足时按 16 的粒度收缩通道块；每次调整后重算 baseN、任务数和各级槽大小。

任务按 Cout/Cin 二维块分配，可沿 Cin 方向蛇形遍历以复用相邻数据。每个任务拥有该通道对范围内的全部 dk/hk/wk 输出，归约覆盖 `batch*dout*ho*wo`。

```text
一个 Cout 块 × Cin 块
    |
    +--> batch：x/dedy 单批搬入 L1，半区轮换
            |
            +--> dout × 输出平面窗口 × dk
                    |
                    +--> L0A(dedy) × L0B(x 窗口)
                              |
                              v
                      L0C 保留全部核位置部分和
                              |
                              v
                       完整归约后写回 dedw
```

## 3 驻留与资源预算

`inputBytes=accBytes=4`。平台提供 L1Bytes、L0ABytes、L0BBytes、L0CBytes；同一份容量记录用于 Host 预算与 Kernel 槽分配。

### 3.1 L1：一个 batch 的两路空间数据

| 对象 | 单份预算 |
|---|---|
| x 驻留段 | `xL1Bytes=cinBlock*di*hi*wi*inputBytes` |
| dedy 驻留段 | `dyL1Bytes=baseM*dout*alignUp(ho*wo,16)*inputBytes` |
| 半区条件 | `alignUp(xL1Bytes,32)+alignUp(dyL1Bytes,32)<=L1Bytes/2` |

每个半区先放 x，再放 dedy；两个半区交替装载 batch。dedy 搬运的实际通道步幅为 `paddedDhowo=alignUp(dout*ho*wo,16)`，与上述容量上界分别记录。

### 3.2 L0：窗口块与完整核部分和

| 对象 | 条件 |
|---|---|
| L0A 双份 | `2*baseM*baseK*inputBytes<=L0ABytes` |
| L0B 单槽 | `bTileBytes=baseK*cinBlock*hkwk*inputBytes`，至少容纳一槽 |
| L0B 池 | `bBufCnt*bTileBytes<=L0BBytes`，槽数同时满足事件资源；采用槽号 0～3、L0A 事件号 6/7 的配置时，`bBufCnt<=4` |
| L0C 单份 | `cTileBytes=baseM*cinBlock*dhwk*accBytes<=L0CBytes` |
| L0C 双份 | `2*cTileBytes<=L0CBytes` 时可按半区轮换，否则使用单份 |

L0C 保存当前通道块的全部核深度部分和。空间尺寸主要约束 L1 驻留，核体积主要约束 L0B 和 L0C；两者共同决定本次 DLoad 计划。

### 3.3 搬运字段

按最终参数检查以下派生量的字段范围、对齐和最大访问地址：

| 阶段 | 需要检查的量 |
|---|---|
| GM→L1 | `di*hi*wi`、`dout*ho*wo`、paddedDhowo、有效通道数与目标通道步幅 |
| L1→L0A | 平面窗口长度、`mCompute/8`、dedy 父步幅与转置目的步幅；mCompute 按 16 对齐，`mCompute/8` 为偶数 |
| L1→L0B | `di*nCompute`、`di*hkwk`、`nCompute/8`、`hkwk*(nCompute/16)`、核位置及空间窗口起点 |
| MMAD→GM | `nCompute*hkwk`、dhwk、有效 Cout/Cin、全局 Cout 步幅和分形源步幅 |

字段范围由目标 SDK 的接口与实现定义，乘积先用宽整数计算，再填入相应字段。

## 4 Host–Kernel 参数

入口将共享字段绑定到 DLoadConfig：

| 字段 | Kernel 含义 |
|---|---|
| batch、cin/cout、di/hi/wi、dout/ho/wo | 原始张量几何与 GM 步幅 |
| dk/hk/wk、padFront/Up/Left | 核尺寸与对称 pad；其余属性已在准入时确定 |
| baseM | singleShapeAligned16Cout，Cout 计算块上限 |
| baseN | `singleShapeAligned16Cin=baseN/hkwk` |
| baseK | kl0HoWo，输出平面窗口上限 |
| hf32Flag | Init 设置 MMAD 模式，End 恢复关闭状态 |
| coreNum | 实际启动 AIC 数，供二维任务分发 |
| singleCoreM/N/K、singleCoreBatchDout | 共享结构分别填写 baseM/baseN/baseK、1；DLoad 的完整归约由任务内循环完成 |
| splitWo/splitWi、tailWo/tailWi | 分别填写 wo/1、0/0，保留整宽输入 |
| streamkType | 0；本模板按输出块独立完成归约 |

每个任务按剩余通道数得到 `coutUse/cinUse`，对应 cRange.coutLength/cinLength；`mCompute=alignUp(coutUse,16)`、`nCompute=alignUp(cinUse,16)` 分别对应 coutAlign16、cinAlign16。容量按块上限预留，MMAD、dk 段起点与 Fixpipe 源步幅使用同一组实际计算跨度；写回只覆盖 coutUse/cinUse。

对每个 dk，x 的深度坐标为 `id=od+dkIdx-padFront`。首个有效贡献初始化该 dk 段的部分和，后续 batch、od 和平面窗口继续累加；整段无有效深度时写零。各输出块的归属唯一，写回使用总 Cin 和完整 dhwk 的步幅。

## 5 求解与发布

1. 完成格式、精度模式和采样几何检查，取得平台资源与搬运字段范围。
2. 从优选块生成通道候选，逐项核算 L1 半区、L0 各槽和完整核部分和；收缩后重算相关字段。
3. 核对实际父步幅、通道尾块、空深度段和指令访问范围，确定任务分发与槽数。
4. 发布共享字段、启动核数和 `GET_TPL_TILING_KEY(TPL_DLOAD,0,0,TPL_WINOGRAD_DISABLE,0)`。

本模板的用户 workspace 为 0：输入直接驻留 L1，归约在所拥有的 L0C 输出块内完成；平台/系统空间按工程框架处理。候选计划在模板选择阶段完成能力与资源判定，再进入对应 Kernel 的生成。

## 6 产出检查

- [ ] 形状、采样几何、精度模式及平台能力与最终计划一致
- [ ] baseN 可还原 Cin 块，基本块、L1 半区和 L0 池的实际地址满足资源约束
- [ ] 每个 Cout/Cin 输出块完整归约 batch、输出空间和核位置，任务归属唯一
- [ ] 尾块的 mCompute/nCompute、L0C 核深度段起点与 Fixpipe 源步幅一致
- [ ] 空深度段写零，用户 workspace=0 的执行链完整，Init/End 的事件和精度模式配对
