# AscendC 多核 Radix Kernel

## 1 执行层次与不变量

每批处理 P_act 行，每行 G 个核协作；slot=blockIdx//G、lane=blockIdx%G，全局行号为 batch×P_act+slot。Init 按 Host 参数建立六段 workspace 视图和单份 tile Queue。D 为值字节数，以下每轮处理一个 digit：

```text
for batch in [0,R):
  用 tmp 的零值视图按 p0..p5 清本批所有 digit 状态
  SyncAll
  按字节轮数奇偶设置值/索引双缓冲方向
  for d in [0,D):
    for tileId=lane; tileId<K; tileId+=G:       # 有效行才访问
      DMA加载当前值 → 输入Queue交接 → Vector构键并提取digit/局部histogram
      DMA保存digit、hist、tile内桶起点到本行workspace
      在UB中累加本核各较小桶计数
    提交本核256桶累计：32位用DMA原子加，64位用SIMT原子加
    SyncAll
    for tileId=lane; tileId<K; tileId+=G:
      重载当前值、已存 digit/hist/桶起点，分别完成 Queue 交接
      Vector形成带aggregate标记的256桶计数，批量DMA发布（末tile可省）
      Sort<uint8> 生成有序 digit 与原块位置 perm
      DMA读取前驱256桶快照 → Vector检查并累加lookback → DMA发布prefix-ready
      后续轮加载当前原轴索引，首轮由 tileStart+perm 生成
      构造 scatter 参数并散射值/原轴索引，完成后释放 tile Queue
    切换 selector
    SyncAll
```

首轮读原输入，后续轮读当前值侧。统计与散射阶段各自加载当前值，局部 Queue/tmp 按 tile 复用。两处 digit 内屏障及批首清零屏障均由全部启动核参与；局部交接见后文事件链。

## 2 统计阶段

首轮读原输入，后续轮读当前 ping/pong 值区。值先转换成符合比较合同的无符号保序键，再提取当前字节；原值仍随索引/重排关系保留。每 tile 产生：

- 当前 digit 字节数组，供排列阶段复用。
- 256 桶真实计数，uint16。
- tile 内各桶独占起点，uint16。
- 本核各桶之前的计数总和，累加到全局独占桶区。

32 位全局累加使用 DMA 原子加，64 位使用 SIMT 原子加。尾 tile 只统计 valid 个元素，不把补齐区算入桶频数。

统计的执行粒度与两类全局数据的用途如下：

```text
coreExclusive[0..255]=0
for 本核负责的tile:
  raw=DMARead当前值侧的valid项；等待输入Queue就绪
  keys=Vector保序变换(raw)                  # 原始GM值不改写
  digit=VectorExtractByte(keys,字节轮次)
  histogram=VectorHistogram256(digit,valid)
  localStart=VectorInclusiveHistogram(digit)-histogram
  coreExclusive += Widen(localStart,CounterT)
  DMAWrite digit、histogram、localStart       # 保存后排列阶段复用
if CounterT为32位:
  SetAtomicAdd<int32_t>
  一次DMA将coreExclusive的256项累加到本行/本digit的globalBucketBase
  SetAtomicNone
else:
  256桶SIMT分工，将各桶coreExclusive原子累加到64位globalBucketBase
等待该核生产完成；所有启动核参加统计末SyncAll
```

Vector 操作按有效元素和256桶分工；局部频数/起点为 uint16，跨tile累计扩为 CounterT。全局桶起点是多核向同一256桶求和，采用原子**累加**；下节状态区是每tile发布自己的编码计数、前驱批量读取，采用DMA**传输**。两者不能合并成一个“全局原子状态”步骤。

## 3 tile 排列与 lookback

每 tile 先发布自己的局部计数状态，再用原生 uint8 Radix Sort 生成稳定排列 `perm[sortedPosition]=originalLocalPosition`。随后回溯本行更早 tile 的同桶计数：

| 状态高两位 | 数值含义 | 消费行为 |
|---|---|---|
| 0 | 尚未发布 | 等待，不能读取未就绪数值 |
| 1 | 本 tile 局部计数 | 累加，继续向更早 tile 回溯 |
| 2 | 包含此前 tile 的完整前缀 | 累加后结束回溯 |
| 3 | 保留 | 不作为有效前缀 |

状态与计数存在同一 CounterT 字中；低位分别保留 30/62 位。计数、状态发布及相应读写完成边界都要保留。参考代码用队列和 Vector/Scalar 事件保护发布、观察及缓冲复用；一个普通 pipe barrier 不能替代这套 GM 协议。

本模板把状态通信具体化为整组256桶的DMA发布与DMA快照，状态检查和前缀累计留在Vector侧。它与上节桶起点的原子求和是两条不同通道：

```text
histogram、localStart、digit和当前原值从各自DMA输入Queue取出
inclusive = VectorWiden(histogram,CounterT)
if tileId+1<K:
  encoded = VectorEncode(inclusive,AGGREGATE_READY)
  交给发布Queue；DMAWrite当前tile的256个encoded字
Sort<uint8>对本tile的valid个digit稳定排序，得到sortedDigit和perm
for prevTile=tileId-1; prevTile>=0; prevTile-=1:
  repeat:
    snapshot=DMARead同slot/同digit/prevTile的256个CounterT字
    等待输入Queue就绪
    aggregateCount,prefixCount=Vector统计snapshot高两位状态
    V_S后Scalar只读取这两个归约计数
    if aggregateCount==256 or prefixCount==256: break
    释放未就绪snapshot，重新DMA读取；inclusive保持不变
  S_V后Vector将snapshot低位计数加到inclusive
  snapshot最后一次Vector读取完成后归还Queue槽
  if prefixCount==256: break                # 前驱已经含更早tile的前缀
if tileId+1<K:
  encoded=VectorEncode(inclusive,PREFIX_READY)
  交给发布Queue；DMAWrite当前tile的256个encoded字
```

一次快照可以包含不同发布进度；只有全部256桶处于aggregate或全部处于prefix时才消费这一快照，其余情况重读。保存的是含本tile的inclusive计数，散射时再扣除本tile计数。发布DMA读完encoded前不可覆盖该UB；snapshot在累计完成前不可复用。上述阶段名不绑定生成代码的类或成员名，DMA字节数、地址单位和事件交接则是算法实现的一部分。

## 4 全局目标地址

对于当前 tile 排列后的第 p 项，令桶为 d、原 tile 内位置为 j：

```text
bucketBase = 全行所有较小桶的元素数
previousTiles = 同桶在更早 tile 中的元素数
localRank = p−localBucketStart[d]
destination = bucketBase + previousTiles + localRank
```

首轮原轴索引为 `tileStart+j`；之后从当前索引 ping/pong 读取。将原值和原轴索引写入另一侧的 destination，保证同桶内按原 tile 次序及块内稳定秩排列。每轮结束后交换 selector；轮数按输入值字节数，不按 CounterT 或输出索引宽度。

一次SIMT散射覆盖完整tile，分成256桶准备和valid项写入两个工作域：

```text
线程t对 bucket=t,t+THREADS,...<256:
  base[bucket]=globalBucketBase[bucket]
               +Count(inclusive[bucket])-histogram[bucket]-localStart[bucket]
全块barrier，所有线程参加
线程t对 p=t,t+THREADS,...<valid:
  jLocal=perm[p]
  originalIndex=首轮 ? tileStart+jLocal : currentIds[jLocal]
  destination=base[sortedDigit[p]]+p
  if 非末轮:
    nextValues[slot*N+destination]=rawValue[jLocal]
    nextIds[slot*N+destination]=originalIndex
  else if first_rank<=destination<first_rank+L:
    out=globalRow*L+destination-first_rank
    若需要values：values[out]=rawValue[jLocal]
    若需要indices：indices[out]=ConvertOutputIndex(originalIndex)
按目标SDK已核对的访存与完成机制发布成对写入，随后参加digit末SyncAll
```

THREADS 是VF线程块参数，与每行核数G不同；当前实现实例取1024，256桶准备仅由有桶任务的线程完成，随后全部线程按stride处理valid项。原值在排列阶段重新加载，构键不覆盖这份用于scatter的值；原轴索引始终随同一个perm移动。对于32位CounterT，base的中间减法按模2^32解释，必须在相同位宽加回p得到合法destination后再扩宽地址；资源和地址上界由Host先验证。

## 5 末轮输出与 workspace 基址

完整 Sort 根据字节轮数奇偶设置初始双缓冲方向，使末轮直接落到最终输出。以下字节公式只描述完整输出 L=N 的参考布局。令 b 为批次、s 为 slot、q 为行内位置，I=sizeof(IndexT)=min(S,J)，globalRow=b×P_act+s：

```text
scratch中间索引 = scratchIndexBase + (s×N+q)×I
输出侧中间索引 = indexOutBase + b×P_act×N×J + (s×N+q)×I
最终索引       = indexOutBase + ((b×P_act+s)×N+q)×J
```

输出侧的批次基址按最终 J 字节推进，中间轮的行距按 I 字节推进。非首轮加载 tile 索引时，在当前侧上述行基址加 tileStart×I，搬 valid×I 字节；uint32 载体下标为对应字节偏移除以4。值两侧都按 D 字节寻址：scratch 值为 `(s×N+q)×D`，输出侧值为 `((b×P_act+s)×N+q)×D`。

4字节值的 Current/Alternate 类型与基址如下；outBatchIndex=indexOutBase+b×P_act×N×J，outBatchValue=valueOutBase+b×P_act×N×D，表内行偏移均从这些基址计算：

| digit | 当前值/索引来源 | 散射值/索引目的 |
|---:|---|---|
| 0 | 原输入 / 无中间索引 | scratchValueBase / scratchIndexBase，IndexT视图 |
| 1 | scratchValueBase / scratchIndexBase，IndexT视图 | outBatchValue / outBatchIndex，IndexT视图 |
| 2 | outBatchValue / outBatchIndex，IndexT视图 | scratchValueBase / scratchIndexBase，IndexT视图 |
| 3 | scratchValueBase / scratchIndexBase，IndexT视图 | outBatchValue / outBatchIndex，最终J字节视图 |

2/8字节值沿同一奇偶规则切换，最后一轮才写最终索引类型；1字节值只有首轮，直接写最终输出。第0轮的 Current 索引无消费者；后续轮以实际读写视图解释 Current/Alternate，而非始终按最终 J 字节行距寻址。

生成 L<N 时，只在末轮 destination 已成为最终 rank 后写目标区间，输出偏移为 `globalRow×L+destination−first_rank`。此前所有轮次仍需完整 N 项；原例将最终输出兼作中间 ping/pong 一侧，派生接口必须提供足够的独立中间区，不能用 M×L 的最终输出承接 N 项中间结果。

## 6 同步、scatter 实参与验证

每个已启动核以相同次数经过 SyncAll，包括尾批的无效行槽位。tile 循环的局部依赖由原语和事件建立，不额外在不等长 tile 循环中插入全核 barrier。

### 中间 scatter 的跨轮可见性

非末轮 scatter 写入完整 N 项值和原轴索引。下一 digit 按新的 tile 区间重载，消费核不一定是上一轮写入核；同一 ping/pong 侧还会在后续轮次或批次重新使用。

参考 `CopyOutGm` 通过 volatile GM 指针分别写值和索引；`ScatterOutput` 随后执行 `PipeBarrier<PIPE_V>()`。`ComputeOnePass` 完成本核 tile 后切换 selector，`ProcessRadix` 执行 digit 末 `SyncAll()`；下一轮统计及排列通过 `DataCopyPad` 和输入 Queue 重载新侧。

核对点如下：

| 环节 | 易错点与实现要求 |
|---|---|
| 值/索引成对发布 | 两者都要对下一轮可见；只保证值或只保证索引写完，仍会破坏原值与原轴位置的配对 |
| scatter 写入 | 不同核的 destination 唯一，不代表位于不同缓存行；不能仅凭无地址冲突就确认缓存访问方式适用于跨核散射 |
| 下一 digit 重载 | 按实际 DMA/SIMT 读取路径，确认生产者写入已到达消费者可见的存储层次；普通赋值、`volatile` 或显式访问接口的语义由目标 SDK/编译结果核对 |
| 局部完成与全核交接 | 写入完成、scatter VF 完成及 digit 末 SyncAll 分别落实；事件或屏障齐全不能代替缓存策略匹配，也不能直接推定私有缓存已刷新/失效 |
| 末轮最终输出 | 只写名次区间，按实际输出消费者建立完成条件；它不再被下一 digit 读取，不能直接沿用它的验证结论证明中间结果安全 |

对应源码调用链为：

```text
CopyOutGm：SIMT经volatile GM指针写values/indices
  → ScatterOutput：PipeBarrier<PIPE_V>()
  → ComputeOnePass：切换selector → ProcessRadix：SyncAll()
  → 下一digit：DataCopyPad → 输入Queue EnQue/DeQue → VF消费
```

这是参考实现事实，不单凭 volatile 或调用顺序推导硬件缓存一致性。目标 SDK 的 GM 访问语义、VF/PIPE_V 完成范围及 SyncAll 的跨核保证仍分别引用对应 API/实现依据；缺少依据的部分保留为验证项。缓存访问或同步的局部替换单独记录原因与适用域，不从这一链默认推导额外原子写或屏障。

下一 digit 读取新侧前，当前侧的完整写入及全核交接必须完成；任一侧的最后读者结束前，不能覆盖该侧。selector 的轮次奇偶与实际读写基址保持一致，不能将更新 selector 视为数据已完成交接。下一批清零状态和复用中间区也须等待前一批最后读写完成；尾批无效行槽仍参加屏障。

验证在已有多 digit、G>1、K>G、尾 tile 和跨批复用用例中覆盖上述交接，包含同地址重跑与更新输入。单 digit 或单核通过不能代表跨轮跨核路径通过。若结果错误，先核对局部排列、桶前缀和 destination；这些正确而 GM 值/索引仍失配时，再定界 scatter 写入及下一轮重载。只对最小失败和受影响场景补证，复用已有有效结果；不以额外屏障或改变时序后偶然通过代替原因定位。

scatter 将 3 个 uint64 偏移放入已结束 Sort 的 tmp 区；Host 至少预留 24 字节。12 个指针和一个 uint32 参数的自然对齐需求按实际 VF ABI 核对，偏移写入、读取及下一次 Sort 覆盖之间保留事件。

验证应覆盖 G=1/G>1、K>G、尾 tile、M>P_act、更新输入重跑，以及 32/64 位计数分支。N=2^30 的自然选路需要相应规模证据，小输入强制实例化不等于真实高索引覆盖。

来源：[多核公共主体](../../example/sort/op_kernel/arch35/common/radix_more_core_base.h) 的 `GetGlobalExclusiveSum/ComputeOnePass/LookbackGlobal/ProcessRadix` 与 [scatter 包装](../../example/sort/op_kernel/arch35/sort_radix_more_core.h)。

## 7 DMA、地址单位与事件链

| 数据 | 每 tile 搬运量 | 逻辑下标 |
|---|---|---|
| 当前原值 | valid×D 字节 | 行基址 + tileId×T |
| 当前中间索引（非首轮） | valid×sizeof(IndexT) 字节 | 行基址 + tileId×T；uint32 载体另乘 sizeof(IndexT)/4 |
| 已提取 digit | valid 字节 | `slot×N+tileId×T` |
| tile hist / 局部桶起点 | 各 256×2 字节 | `(slot×K+tileId)×256` |
| lookback 前驱状态 | 256×S 字节 | `((slot×D+d)×K+prevTile)×256` |
| 全行桶起点 | 256×S 字节 | `(slot×D+d)×256` |

预提取字节区按 P_act×K×T 保守分配，但参考 Kernel 的行 pitch 实际是 N。容量向上补齐与消费时的地址跨度分别维护。表中状态下标按 CounterT 元素计，原始 uint32 workspace 视图需要再乘 S/4。

具体局部交接包括：

1. 输入值、digit 和 histogram 的 EnQue/DeQue 保证各自 DMA 生产完成；状态发布使用独立的计数 Queue 交接。
2. lookback 读取快照后，Vector 统计 256 桶状态；`V_S` 后 Scalar 才读状态计数。需要累加时先 `S_V`，累加完成再 `V_S`，之后才释放该快照给下一次轮询复用。
3. Sort 的 tmp 改作三个 uint64 scatter 参数前先 `V_S`；Scalar 写入后 `S_V`，scatter VF 才能读取。参考包装在 scatter 后执行 `PipeBarrier<PIPE_V>()`，保护 tmp 及 tile 局部输入的复用。
4. digit 末 SyncAll 保护跨核散射结果，批首 SyncAll 保护清零结果。这些全核屏障不替代上述局部 Queue/Vector/Scalar 交接。

## 8 贯穿：M=37、N=8192、T=4096

沿用 [选核算例](tiling.md#6-可复算选核例子)，int32/int64：G=1、K=2、P_act=C_act=37、R=1。核 36 对应 slot=globalRow=36，依次处理 tile0、tile1；二者都为完整 tile。

以 digit d=2、tile1 为例，状态下标为 `((36×4+2)×2+1)×256=75008`，单位为 CounterT 元素；前驱快照取同一行/字节的 tile0。tile hist 下标则为 `(36×2+1)×256`，单位为 uint16 元素。预提取 digit 的行 pitch 为 N，不使用状态区的 tile/bucket 布局。

即使 G=1，两块仍依次执行发布、回溯与散射；全核屏障还保护不同行的阶段推进。末轮若 destination=3，first_rank=3、L=1 时写本行输出元素 36，索引转为 int64；此前各轮仍保留完整 N 项中间状态，参见 [Buffer](dag-buffers.md#4-最终输出与受限区间)。

### 协议演算：跨核、跨批与混合索引宽度

另设 M=5、N=2305、T=1024、G=2、P_act=2、C_act=4、D=4、I=4、J=8，则 K=3、R=3。这里只演算协议，不宣称这是 Host 搜索结果或设备性能最优参数；实际容量仍由 Tiling 查询确定。

- 每 slot 的 lane0 处理 tile0/2，lane1 处理 tile1；tile2 的 valid=257。三批全局行依次为 0/1、2/3、4/无效槽。
- 第 b=1 批先清零全部 digit 的 bins/status，再由4个核参加 SyncAll。考察 s=0、digit=1：两核完成统计和 DMA 原子累加后再次 SyncAll，设桶7的全行较小桶起点为100。
- lane1 的 tile1 假设 H[7]=2、localStart[7]=20：先向256桶状态 DMA 发布 aggregate，再执行本 tile Sort，供后继 tile 与它的排序重叠推进。桶7状态元素下标为 `((0×4+1)×3+1)×256+7=1031`；tile0 前驱对应775。
- tile1 每次 DMA 重读 tile0 的整256桶快照。若混有 aggregate/prefix 状态则重试；假设最终读到全部 prefix-ready，桶7前缀为3，则 inclusive[7]=2+3=5，立即终止回溯。tile1 再批量发布 prefix-ready；tile2 可据此提前结束回溯，末 tile 自身无需发布。
- 假设当前有序位置 p=21 属于桶7，则 `base=100+5-2-20=83`，`destination=83+21=104`。原轴索引从当前 scratch 的 `currentIds[perm[21]]` 读取，与原值成对散射到输出中间侧。
- 本轮该索引地址为 `indexOutBase+1×2×2305×8+104×4=indexOutBase+37296`；最终 digit 对同一名次使用 `indexOutBase+(2×2305+104)×8=indexOutBase+37712`。中间轮和末轮的批内步长分别为4/8字节。
- 本核 scatter 后 PIPE_V，所有核完成本 digit 的 SyncAll，digit2 再按 IndexT 从输出中间侧 DMA 重载；状态使用 digit2 独立切片，hist/桶起点/digit 区按轮复用。
- 第 b=2 批等待前批末轮交接完成，复用同一份 scratch 并重新清零。s=1 的 globalRow=5 无效，跳过值/索引访问，仍参加批首及每 digit 的全部 SyncAll；索引输出批次基址已推进到 `indexOutBase+73760`。

L<N 的派生接口改用独立的两侧完整中间区，按 §5 的区间输出规则计算最终地址，不借用本例的完整输出暂存布局。
