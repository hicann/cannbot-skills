# 卷积反向权重梯度分支：tiling 切分算法

本文是 filter 切分入口，先选择计算模板，再确定任务、驻留与 Host–Kernel 参数。第 3–9 节为 MN_StreamK 共用选块规则；Winograd 和 DLoad 的独立规则在各自模板中展开。任务和布局见 [compute-flow](../compute-flow.md)，平台资源由适配层取得。

## 1 总览：tiling 切什么、为什么切

MN_StreamK 使用以下三个尺度（记号层次见 [language.md](../language.md) §3）；DLoad 和 Winograd 的任务与驻留由各自模板定义：

```
任务（group × kd × M 段 × N 段；一个任务完整归约 batch×dout×ho×wo）
  ──► L0 基本块（baseM × baseN，每次消费 baseK）
        ──► L1 父段（stepKa / stepKb：一份驻留覆盖的 K 块数）
```

M 增大可复用 x 窗口，N 增大可复用 dedy，K 增大可减少循环和换驻；代价分别是 L0/L1 占用、输出尾块和任务数变化。优选块是规则求得的候选，实际性能最优性由同平台、同输入的测量确认。

各路 buffer 份数取 1 或 2 即乒乓开关：**1=串行（装载与计算互等）、2=重叠流水**；本分支 L1 份数随「阶梯驻留候选」（第 6 节）从 (2,2) 退到 (1,1)，L0C 份数由容量不等式直接判定（第 4.1 节）。

## 2 模板选择

### 2.1 选择顺序

模板按注册编号从小到大尝试：**DLoad(8) → Winograd(9) → MN_StreamK(10)**。DLoad 在准入阶段完成通道候选与驻留预算，形成可执行计划；其他模板按各自规则判定。IsCapable 不满足时继续，输入规格错误及已选计划的求解失败单独报告。

```text
统一问题 + 平台/SDK 能力
    |
    +--> DLoad (8) ------> 单批空间驻留 / 通道切块 / 完整归约
    |       不满足准入
    v
    +--> Winograd (9) ---> 双路正变换 / 16 点 MMAD / 逆变换
    |       不满足准入
    v
    +--> MN_StreamK (10) -> 普通/拆核窗口，整宽/切宽，尾轮 K 切分
    |
    v
模板对应字段 + key + blockDim + workspace + 初始化/同步契约
```

[Winograd](templates/winograd/tiling.md)、[DLoad](templates/dload/tiling.md) 和 [MN_StreamK](templates/mn-streamk/tiling.md) 分别维护在线求解规则。DLoad 按直接搬运、单批空间驻留和完整输出块归约组织计算；Winograd 使用变换域流水；MN_StreamK 使用普通窗口与分段归约。

### 2.2 模板与编译分支

模板名表示任务和求解组织；isSplitKernelHW 表示 Kernel 窗口实现；splitWo 表示运行时宽度分段。三者分别发布，不能用其中一项推定其他项。两种 isSplitKernelHW 实例使用相同的任务所有权、分段合并和输出格式。

### 2.3 扩展方式

新增独立策略时增加 `templates/<name>/tiling.md`，说明准入、起点、任务、驻留、求解顺序和验收，并在选择表登记。共用尺寸和资源公式保留在本页；API 与 key 编码维护在 filter 适配层。

### 2.4 在线求解边界

本范式的 tiling 一律**在线求解**：input 与 filter 分支不读取调优数据仓库（knowledge_tiling 等），不引入"命中记录即跳过求解"的回放路径。参考实现中的仓库回放机制属其工程侧调优流程，**不进入生成范式的知识与准入**；生成工程按 §4–§8 的规则求解并全程留痕。
## 3 输入与输出

输入为统一 3D 几何、原张量步幅、dtype/format，以及目标芯片版本、platformInfo 和 SDK 对应的 AIC/AIV 核数、L1/L0/UB 可用容量和指令限制。`L1Bytes/L0ABytes/L0BBytes/L0CBytes/UBBytes` 与 Kernel 分配使用同一份平台能力记录。`inputBytes=sizeof(T)`，FP16/BF16 为 2、FP32 为 4；`accBytes=outputBytes=4`。MN_StreamK 数学选块粒度 C0=16，m0=n0=16；物理 k0/channelSize=32/inputBytes，FP32 为 8。资源计算不得继续套用两字节输入预算。

| 方向 | 数学有效长度 | 内部选块长度 |
|---|---|---|
| M | CoutG | `mValue=alignUp(CoutG,16)` |
| N | CinG*hk*wk | `nValue=alignUp(CinG,16)*hk*wk` |
| K | ho*wo | `kValue=ho*wo` |

输出为基本块、任务、step、buffer 份数、单份 L1 元素预算、splitWo、coreNum、启动 blockDim 和执行分支。尾宽、任务计数和归约分段可以由 Kernel 复算，不作为另一份独立真值。

## 4 L0 基本块与 InitBaseBlock

设 `alignUp/alignDown` 为向上/向下对齐，`ceilDiv` 为向上取整除法。M/N 先按 256 的目标块长均分，再对齐 16；源码的整数除法发生在对齐之前。

### 4.1 M/N 优选起点与窗口关系

```cpp
baseM = mValue > 256 ? alignUp(mValue / ceilDiv(mValue,256),16)
                         : alignUp(mValue,16);
baseN = nValue > 256 ? alignUp(nValue / ceilDiv(nValue,256),16)
                         : alignUp(nValue,16);
uint64_t hw = uint64_t(hk)*wk;
uint64_t nAlign = (hw == 9 ? hw*16 : 16);
if (baseN > nAlign) baseN = alignDown(baseN,nAlign);
uint64_t l1Cin1 = max(baseN/(hw*16),uint64_t(1));
baseN = min(baseN,l1Cin1*hw*16);
cl0Pbuffer = (baseM*baseN*4*2 <= L0CBytes) ? 2 : 1;
```

`hk*wk=9` 时优先 144 的倍数；其他窗口先保持 16 对齐。完整通道窗口的约束落在任务 singleCoreN 上。**L0 baseN 按 16 对齐；普通整窗判定为 `(baseN/16) % (hk*wk) == 0`**，其余由非整窗搬运/写回状态维护核位置连续性。

任务按完整通道窗口恢复，L0 块可继续细分；非整窗搬运与写回由 head_/tail_ 维护核位置和通道边界。

### 4.2 K 的容量与搬运对齐

两路 L0 输入按两份槽预留：

`maxKa=alignDown(L0ABytes/(baseM*inputBytes*2),16)`，
`maxKb=alignDown(L0BBytes/(baseN*inputBytes*2),16)`。

初值 `baseK=min(maxKa,maxKb)`。若 `alignUp(kValue,16)<baseK`，截到该对齐值；否则，当 `splitWo<baseK` 且 splitWo 为 16 倍数时，向下对齐到 splitWo 的倍数。最终 K 为正且 16 对齐。

### 4.3 L0 三路容量

| 存储 | 必须满足 |
|---|---|
| L0A | `baseM*baseK*inputBytes*al0Pbuffer <= L0ABytes` |
| L0B | `baseN*baseK*inputBytes*bl0Pbuffer <= L0BBytes` |
| L0C | `baseM*baseN*4*cl0Pbuffer <= L0CBytes` |

普通输入通路 al0Pbuffer/bl0Pbuffer 为 2；拆核实现可能将 L0B 固定为单槽，仍保留 GetBaseK 中两份容量折算的保守起点。设备槽数、Host 预算和实际指令偏移须一致。

## 5 L1 驻留容量

### 5.1 父段范围与覆盖行数

令 `Ka=stepKa*baseK、Kb=stepKb*baseK`，分别表示两路父段的 K 预留长度。对任一当前宽度 W，覆盖行数 Rows(L,W) 为：

| 条件 | Rows(L,W) |
|---|---|
| L 与 W 互有整除关系 | `ceilDiv(L,W)` |
| 不整除，L>W | `L//W+2` |
| 不整除，L<W | 2 |

切宽时要分别检查完整宽片和非零尾宽片，取预算较大者。尾宽为零表示整除，不作为零长度宽片参与除法。

### 5.2 CalAL1Bound：dedy 单份预算

| 情况 | 单份元素数 |
|---|---|
| splitWo=wo | `al1Bound=Ka*baseM` |
| 切宽片 W | `alignUp(Rows(Ka,W)*W,16)*baseM` |

切宽预算覆盖可能跨行多搬的父段；L1→L0 的实际父跨度仍以本轮真实搬入行数为准。

### 5.3 CalBL1Bound：x 单份预算

先算 `hoCal=Rows(Kb,W)`。普通窗口取 `hiCal=(hoCal-1)*strideH+(hk-1)*dilationH+1`，`Wi=(W-1)*strideW+(wk-1)*dilationW+1`；不切宽时 InitSplitWOI 使用原 wi 作为 splitWi。拆核窗口取 `hiCal=hoCal、Wi=W`。

将 `nBlocks=ceilDiv(baseN,16)`，Host 的 bL1Cin1CopyLen 初值为 `ceilDiv(nBlocks,hk*wk)`。若 `hw>nBlocks` 且不能整除，额外加 1；否则若 `(2*nBlocks)%hw != 0`，额外加 1。这是跨通道窗口时的覆盖预留。

`singleCoreCinBudget=max(baseN/(hw*16),1)*16`，
`bl1Bound=hiCal*Wi*min(singleCoreCinBudget,bL1Cin1CopyLen*16)`。

这里 bL1Cin1CopyLen 为块数。Kernel 的 bL1cin1CopyLen 在有效通道截断后保存通道数，不能再无条件乘 16。

### 5.4 容量与物理槽

L1 容量判断为：

`(al1Bound*al1Pbuffer+bl1Bound*bl1Pbuffer)*inputBytes <= L1Bytes`。

al1Bound/bl1Bound 是**单份元素数**，InitBuffer 时乘 inputBytes 并按 32B 对齐。固定半区分配还要检查每个实际区间：L1A ping 从 0 起，L1B ping 从 aAligned 起，pong 从 halfL1 及 halfL1+aAligned 起。预算之和合法不替代区间交叠与最大末地址检查。

## 6 SetStepK4SplitMN：阶梯驻留候选

初始 depthA1=depthB1=16，两路各两份 L1。对每个 depth，`stepKa=min(max(depthA1/al1Pbuffer,1),ceilDiv(kValue,baseK))`，B 路同理；较大 step 向下对齐为较小 step 的整数倍。

```cpp
depthA1 = depthB1 = 16;
al1Pbuffer = bl1Pbuffer = 2;
bool offDBL1 = false;
while (depthA1 >= 1 && depthB1 >= 1) {
    UpdateStepMNK();
    if (!IsCurBlockL1Invalid()) break;
    depthA1 -= (depthA1 > 2 ? 2 : 1);
    depthB1 -= (depthB1 > 2 ? 2 : 1);
    if (!offDBL1 && (depthA1 <= 2 || depthB1 <= 2)) {
        offDBL1 = true;
        al1Pbuffer = bl1Pbuffer = 1;
        depthA1 = depthB1 = 16;
    }
}
if (IsCurBlockL1Invalid()) {
    ShrinkBaseBlock();
    UpdateStepMNK();
}
UpdateSingleCoreInfo();
```

UpdateStepMNK、两路 Bound 与 ShrinkBaseBlock 分别由第 5、6、7 节规则实现；它们是本工程函数。返回后再次检查最终组合，不能把循环结束等同于合法。

## 7 MultiCoreSplitMN 与容量回调

### 7.1 先按指令范围选择窗口实现

checkLargeSpecs 在 H/W 核长或 dilation 超过 255、stride 超过 63、任一 H/W pad 超过 255，或 `hk*wk*16>65536` 时选择拆核。范围来自当前窗口指令映射，目标 SDK 更换时重新核实。

```cpp
bool large = checkLargeSpecs();
bool invalid = large ? trySplitKernelHW() : tryNormalTiling();
if (!invalid) return true;
if (!large && !trySplitWo()) return true;
return !trySplitKernelAndWo();
```

try* 函数返回 **true 表示仍超容量**；MultiCoreSplitMN 返回 true 表示求解成功。每个 try 都重做基本块、驻留和最终检查：

| 路径 | 动作 |
|---|---|
| tryNormalTiling | 当前普通窗口和 splitWo 下求解 |
| trySplitKernelHW | isSplitKernelHW=true，重算 splitWi/tailWi |
| trySplitWo | 允许切宽时从 128 起按 2 的幂减到 16，逐个求解 |
| trySplitKernelAndWo | 拆核；允许切宽时取 splitWo=16，再求解 |

每次修改 splitWo 或 isSplitKernelHW，都同步更新 tailWo/splitWi/tailWi。各次失败保留可解释结果，最终不合法时返回失败。

### 7.2 ShrinkBaseBlock：先 K、后 M/N、再恢复

| 阶段 | 计算和动作 |
|---|---|
| 缩 K | 根据 L1 缺口与每减 16 个 K 的两路减少量估算步数，baseK 至少 16；必要时逐个分形递减并复查 |
| 缩 M/N | 先减较长方向使其接近另一方向，再按 L1 缺口收缩；始终保持 16 对齐；N 的变化同时重算跨窗口通道预算 |
| 恢复 M | 在剩余 L1、L0C 与 mValue 允许范围内回增 M |
| 恢复 K | 逐个分形回增到最后合法值，再受 L0A/B 和 alignUp(kValue,16) 截断；满足条件时对齐 splitWo |

ShrinkBlockBaseK 的线性估计中，A 路每减一个 K 分形减少 `baseM*16*inputBytes`；B 路按通道覆盖量、splitWi、strideH 和 splitWo 估算减少量，实际仍按 Bound 重新计算。线性估计不替代台阶式窗口容量检查。

回调可能形成非整窗 baseN，Kernel 随之采用对应 Load3D 和 Fixpipe 分段逻辑。不能为了保持起点形状而跳过实际容量检查。

## 8 UpdateSingleCoreInfo：公共任务恢复

| 字段 | 计算 |
|---|---|
| singleCoreM | baseM |
| l1Cin1 | `max(baseN/(hk*wk*16),1)`；拆核路径为 1 |
| singleCoreN | `l1Cin1*hk*wk*16` |
| singleCoreCin | `l1Cin1*16` |
| singleCoreCout | singleCoreM |
| singleCoreK | kValue |
| singleCoreHo | singleCoreK/wo |
| singleCoreBatchDout | batch*dout（DoStreamKTiling 前的基值） |

任务数 `totalCnt=ceilDiv(mValue,singleCoreM)*ceilDiv(nValue,singleCoreN)*dk*G`。共享 coreNum 使用平台计算核数，Kernel 按它轮转。Host 的 usedCoreNum 决定 blockDim：type=0 时取 min(coreNum,totalCnt)，type≠0 时取 min(coreNum,coreStreamK*totalCnt)。尾轮切分随后可能修改 singleCoreBatchDout、singleCoreK；见 [MN_StreamK §4–§5](templates/mn-streamk/tiling.md)。

iterateOrder 取 `mValue>nValue ? 1 : 0`，控制任务内子块顺序。MN_StreamK 当前任务编号解码固定为 group→kd→N→M（快→慢），见 compute-flow 第 6 节。

## 9 Host 传给 Kernel 的参数

### 9.1 几何与寻址

| 字段 | 语义与单位 | 来源 |
|---|---|---|
| batch、cin/cout | 批次数、总通道数 | 统一问题 |
| group/realGroup | 数学组数/执行组数，当前均为 G | 原始分组配置 |
| cin1G/cout1G | **执行组通道跨度，单位为通道，不是块数** | 由求解前的分组数据映射确定 |
| di/hi/wi、dout/ho/wo | x/dedy 空间长度 | 统一问题 |
| dk/hk/wk | 核采样点数 | filter_size |
| strideD/H/W、dilationD/H/W | 正向步长、膨胀率 | 属性归一 |
| padFront/Back、padUp/Down、padLeft/Right | 正向两侧 pad | 属性归一 |
| m0/n0 | 矩阵分形粒度，均为 16 | 能力与类型 |
| k0/channelSize | 32/sizeof(T)，两字节输入为 16、FP32 为 8 | 物理输入通道块 |

数学 CinG/CoutG、Host 的 cin1_g/cout1_g 块数和序列化字段 cin1G/cout1G 分别记录。当前原始分组配置的序列化字段分别取 cin/group、cout/group；单组时等于总通道数。对齐跨度用于 mValue/nValue，不改变 GM 通道跨度。

### 9.2 计算、任务与驻留

| 字段 | Kernel 用途 |
|---|---|
| baseM/baseN/baseK | 一次 L0 计算与尾块 |
| singleCoreM/N/K、singleCoreCin/Cout | 输出任务规模 |
| singleCoreDk/Group/Batch | 当前任务核深度/分组/批次基值 |
| singleCoreBatchDout/singleCoreHo | 归约范围与行数基值 |
| stepKa/stepKb | 父段换驻频率 |
| al1Bound/bl1Bound | 单份 L1 **元素数**，设备乘元素字节数 |
| al1/bl1/al0/bl0/cl0Pbuffer | 各路 buffer 份数 |
| splitWo | 宽片长度；tailWo/splitWi/tailWi 按当前路径派生 |
| iterateOrder | 任务内 M/N 子块遍历 |
| coreNum | Kernel 调度基数；不以 blockDim 代替 |
| streamkType | 尾轮 StreamK 三态：0=无 / 1=BatchDout / 2=HWout（判定链见 [mn-streamk §5](templates/mn-streamk/tiling.md)） |
| hf32Flag | FP32 是否采用 hf32 计算模式；非 FP32 为 0 |

usedCoreNum、coreStreamK、coreBindDirection 和 singleShapeK 是 Host 工作量，不在共享结构中新增同名字段；singleShapeK 的最终范围写入 singleCoreK。

dtype/format、isSplitKernelHW 和模板编号可由编译实例传递；字段类型、共享结构和 key 在 [适配层绑定（Ascend C）](../../../adapters/ascendc/filter/tilingkey-templates.md) / [DSL](../../../adapters/dsl/filter/tiling.md) 定义。归约阈值由 Kernel 的 CalcReduceSegments 消费，不作为另一份可自由修改的 Host 字段。

## 10 检查点

- [ ] M/N 起点、N 的 9 点窗口优选与非整窗回调区别明确。
- [ ] baseK 的截断、splitWo 对齐、三路 L0 预算同时成立。
- [ ] 单份 Bound、buffer 份数、元素字节数和半区地址均已核对。
- [ ] L1 阶梯搜索和 try* 返回值语义一致，失败不作为成功发布。
- [ ] singleCoreN 是整窗任务尺度，baseN 是实际计算尺度。
- [ ] cin1G/cout1G 序列化单位是通道，与 Host 块数区分。
- [ ] coreNum、blockDim、任务轮转和初始化参与核分别核对。
- [ ] streamkType、hf32Flag、原始组映射与 Kernel 实例一致；其他模板按私有页校验。
