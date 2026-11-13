---
type: CATLASS DSL Operator Example
title: GDN2 Prepare 融合开发指南（标量门控接口）
description: 自包含的 prepare 接口约定与 CATLASS DSL 设计候选：双 task 流水、64-lane 双叶块布局、两行共享前代、Cube 递归逆的成本门禁及关键路径预取；不依赖私有实现或本地源码快照。
tags: [catlass-dsl, operator, linear-attention, gdn2-prepare, mixed-cube-vector, triangular-inverse, simd, task-pairing, prefetch, ascend-950]
status: draft
generated: {by: process:gdn2-prepare-knowledge-summary, at: '2026-10-08T00:00:00Z'}
verified:
  - {by: process:gdn2-prepare-document-source-audit, at: '2026-10-08T00:00:00Z'}
  - {by: process:gdn2-prepare-document-candidate-review, at: '2026-10-12T00:00:00Z'}
sources:
  - id: wu
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/README.md
    title: 公开 GDN W/U 参照：矩阵公式、grouped-head 映射与 chunk 范围；不是本卡融合实现
  - id: solve
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel/arch35/solve_tri_ascend950.h
    title: 公开 AscendC 三角逆参照：独立 tile 调度、片上块合并及跨引擎握手；不是 CATLASS DSL 源码
operator_families: [linear-attention, gdn2-prepare]
arch: [c310]
---

# 接口与概念

本卡中的 `GDN2_prepare` 是一个**标量 gate/beta 的 prepare 开发接口约定**，不是对其他库中
完整 GDN2 接口的声明。它只生成 `g_intra、a_mat、w、u`，不包括跨 chunk 的状态递推
`fwd_h`、最终 attention 输出或 CP 通信。不得把 prepare 时延称为整个 GDN 的端到端时延。

证据分为两层：公开固定提交支持 W/U 计算模式，以及三角逆的片上分块与同步参照；
本文其余内容是明确写出的接口定义、由该定义推导的恒等式和待验证 DSL 设计候选。
参照中的 AscendC 类、flag 编号、布局和 API **不能直接当作 DSL 实现**。[^wu][^solve]

本卡不要求提交或安装 kernel、fixture、私有 Git 对象或实验日志。`draft` 表示尚不提供
可公开复现的完整 DSL 融合实现；`verified` 仅表示文档与所列公开来源的静态核对，不表示上板验收。

## 算子算法

以下完整定义是本卡的讨论前提，接入其他算子时必须逐项核对，不能仅凭名称套用：

- K 为 `[B,HK,T,Kdim]`，V 为 `[B,HV,T,Vdim]`，g/beta 为 `[B,HV,T]`，均连续。
- 首版设计范围取 `C=64、Kdim=Vdim=128、T%C=0、HV%HK=0`；K/V 为 BF16，g/beta 为 FP32。
- 一个 `(b,hv,chunk)` 读取 `hk=hv//(HV/HK)` 的 K；下式 i、j 从当前 chunk 起点计数。

```text
G[i] = log2(e) * sum(-softplus(g[t]), t=0..i)
r[i] = 1 / sqrt(sum(K[i,d]^2, d) + 1e-6)
Khat[i] = r[i] * K[i]
L[i,j] = (i>j) * beta[i] * dot(Khat[i],Khat[j]) * exp2(G[i]-G[j])
A = inverse(I + L)
W = A @ (Khat * (beta * exp2(G))[:,None])
U = A @ (V * beta[:,None])
```

G 在每个 chunk 重置累计，beta 已是乘法系数，不再 sigmoid；L 的 beta 按行 i 缩放。
规定输出 G 为 FP32，A/W/U 为 BF16，输出的 head 维均为 HV；不能用 K 的 HK 维分配 W。
公开 W/U 参照使用自然指数和已给定的 A；代入 `k=Khat、g=ln(2)*G` 可对齐最后两式，
但它不证明本卡额外规定的 softplus、epsilon 或四输出融合已经实现。[^wu]

# 用法

## 分核策略与基本块切分

由上述定义，各 chunk 的四个输出不依赖前一 chunk 的状态，因此优先把 chunk 放入任务轴，
而不是只按 B×H 分核后在核内串行完成所有 chunk。公开三角逆参照也以独立矩阵 tile
调度，并让配对的 AIC/AIV 处理同一 tile；这提供调度参照，不规定 DSL 的 launch grid。[^solve]

建议首版映射：

```text
NT = T / C; Q = B * HV * NT
task = (b * HV + hv) * NT + chunk
chunk = task % NT
hv = (task // NT) % HV
b = task // (NT * HV)
hk = hv // (HV / HK)
```

每个任务独占一个 chunk 的四输出，矩阵内的 solve 依赖保留在该任务中，无需跨任务归约。
Q 是逻辑任务数，不是物理核数；使用一任务/block 或核内循环领取多个任务，应分别测量。
若一次领取两个相邻 task，必须独立解码并保护 `task<Q`；奇数 NT 时相邻 task 可跨 head。

### 双 task 调度与跨任务流水

双 task 是调度候选，不能仅把核内串行循环次数改为 2 后称为流水化：

```text
pair = logical_block_id
task0 = 2 * pair; task1 = task0 + 1
grid = ceil(Q / 2)
每个 task 独立检查 task<Q，再解码 b/hv/chunk/hk
```

例如 B=1、HK=16、HV=32、T=11264、C=64 时，NT=176、Q=5632，候选 grid=2816。
2816 是逻辑 block 数，不是物理核数；28 个常驻 block 循环领取任务也可以让 chunk 并行，
两者仍须比较调度开销、缓冲占用、尾部负载和跨 task 的真实交叠，不从 block 数单独推断性能。
小 Q 可比较一 task/block 的回退，分支阈值须针对目标设备测量，不能固定套用物理核数。

双 task 的目标是让当前任务的 Cube 块修正与下一任务的 Vector 准备交叠，
而不是假定两个任务共享 K。先确定每个任务的缓冲与事件代，再实施后文的提前扫描和预取。

若考虑共享 K 的 head 配对，应配对同一 `(b,hk,chunk)` 下不同 hv，仅复用 raw K/KKT；
不同 hv 的 g/beta 仍使 L、A、W/U 不同。NG=`HV/HK` 为奇数或 1 时必须有完整回退路径。
公开参照中的 grouped-head 读取规则与 W 输出 head 维提供接口依据。[^wu]

# 代码模式

## 数据路径与存储层级

下列是按接口推导的候选，不是对某个未提交 kernel 的源码描述。

### raw KKT 与列系数折叠

先计算 `R=K@K.T`，由 R 的对角取得平方范数，再构造 L，可避免把归一化 K 写到 GM。
也可利用矩阵乘恒等式，把 W/U 的系数折叠到 A 的列：

```text
Ak[i,j] = A[i,j] * beta[j] * exp2(G[j]) * r[j]
Au[i,j] = A[i,j] * beta[j]
W = Ak @ K(raw)
U = Au @ V
```

注意 L 是行缩放，Ak/Au 是列缩放。上述变换在实数代数下等价，但低精度 R 对角的误差、
Ak/Au 提前转换 BF16 的误差，以及乘加顺序均可能改变结果，必须分别做数值验证。
优先评估 K 的 L1 数据能否从 KKT 复用到 W，而不是假定减少一次理论搬运就一定提速。

### 32+32 块逆与两行 SIMD 前代

对 C=64 可选择以下精确分块恒等式：

```text
I+L = [ M00   0  ]       A = [ D0                  0 ]
      [ L10  M11 ]           [ -(D1 @ L10) @ D0    D1 ]
D0=inverse(M00); D1=inverse(M11)
```

它不是截断 Neumann 近似。建议比较 SIMD 求两个 32 阶叶块、Cube 求交叉块的路径，
并尝试让第一乘结果留在 L1 服务第二乘。公开参照采用不同的 16 阶叶块及递归合并，
可参考其 L0C→L1/UB 与握手组织，不能声称它实现了此处的 32+32 SIMD 方案。[^solve]

叶块内部可逐行前代；若历史向量 load 显著，可用两行共享读取的候选：

```text
for i in 0,2,...,30:
    acc0=0; acc1=0
    for j in 0,...,i-1:
        prev=load(D[j])
        acc0 += L[i,j]*prev; acc1 += L[i+1,j]*prev
    row0=unit_row(i)-acc0
    acc1 += L[i+1,i]*row0
    row1=unit_row(i+1)-acc1
    store(D[i],row0); store(D[i+1],row1)
    protect_store_to_later_load()
```

这是数学/依赖伪码，不是可执行 DSL API。每行保持 j 顺序，并直接使用新 row0；
一个 32 阶叶块的历史行读取由 496 次降为 240 次，但活跃值增多，不能据此预测总时延。
两个叶块若打包进同一向量，须另证明 lane、行号和两 AIV 的发布区域不重叠。

### 64-lane 双叶块布局

针对 64-lane FP32 操作数，可把两个 32 阶叶块的同一逻辑行打包为：

```text
lane 0..31  -> D0[row, 0..31]
lane 32..63 -> D1[row, 0..31]
```

一次历史行 load 同时取两个叶块；相应 Ldiag0/Ldiag1 系数分别在两个半向量广播，
不能把一个叶块的标量广播到全部 64 lane。两行联合前代仍按原 j 顺序更新，并让新 row0 留在寄存器中服务 row1。
与每个叶块各占一个 64-lane 向量相比，这个布局可减少对结构零列的运算，
但打包、半向量广播、地址计算和寄存器活跃量也可能抵消收益。

计算布局与两 AIV 的发布所有权分别设计。例如两 AIV 若各在独立 UB 计算完整双叶块，
可各发布每个叶块的 16 行，所有行恰好覆盖一次；这会重复计算，不能宣称优于各算一个叶块。
最终 A/W/U 的输出分片仍需单独映射。必须核对 lane→矩阵坐标、两半系数、UB 地址、L1 发布
和 GM 输出的完整覆盖，不能仅用 sub_block_idx 的地址偏移证明数据布局正确。

联合两行时，240 次是每个叶块的历史行 load 次数；若两叶共享打包 load，仍是 240 条
逻辑向量读取。组间 store→load 保护从 32 次变为 16 次不代表全部屏障数量减半。
扩大到四行或更多行必须作为独立候选，检查编译后临时存储、scalar、向量读取和总时延；
不能根据理论 load/fence 下降认定收益，也不能没有证据就断言寄存器 spill。

### Cube 叶块递归逆与混合路线的选择

公开 SolveTri 的 MCH 在 Cube 求 16 阶叶块逆，MBH 再逐层合并；这是可比较的另一条路线，
不证明把本卡 SIMD 叶块替换为 Cube 一定更快。[^solve]
对严格下三角叶块 Ld，可采用以下数学递推候选：

```text
X = I - Ld; P = Ld
重复 log2(m)-1 层（m=16 或 32）:
    P = P @ P
    X = X + X @ P
```

16 阶三层得到 0..15 次幂，32 阶四层得到 0..31 次幂；Ld^m=0 时，实数代数下
等于 inverse(I+Ld)。这是利用幂零性的有限精确恒等式，不是凭浮点误差可忽略而截断。
FP16/BF16/HF32 的乘法、每层 FIX/cast 和累加位置会改变误差，必须通过调用方精度合同。

比较 Cube 与 SIMD 时，分别记录叶块尺寸、MMAD 数量/格式、L0C 持续累加与初始化方式、
FIX→L1/UB 路数、MTE1、格式转换、同步等待，以及 Cube/Vector 的实际重叠。
两个 32 阶叶块打包到 64×64 Cube 矩阵会计算结构零；紧凑小 tile 又可能增加发射和同步，
两种设计都需单轴测量。减少 Vector 指令但新增 Cube/FIX/MTE 等待时，应以完整 kernel
关键路径判定；SIMD+Cube 修正的历史成绩不能作为全 Cube 叶块路线的性能验收。

### 容量与存活期预算

对本卡设计形状，单个逻辑矩阵的紧凑容量如下；不是编译器实际分配量：

| 数据 | 单矩阵容量 | 候选复用边界 |
| --- | ---: | --- |
| R/L/A，FP32 `[64,64]` | 各 16 KiB | 旧值最后一次消费完成后才可重用 |
| Ak/Au，BF16 `[64,64]` | 各 8 KiB | Cube 读完才可覆盖 |
| K/V，BF16 `[64,128]` | 各 16 KiB | K 若留 L1，须活到 W 完成 |
| W/U，FP32 `[64,128]` | 各 32 KiB | FIX 完成后 pack，搬出后释放 |
| W/U，BF16 `[64,128]` | 各 16 KiB | MTE3 读完后才可给下一任务 |

实际预算还需计入对齐/padding、AIV 分片、双缓冲、标量向量和编译器临时量。
别名复用必须记录“生产完成→最后读取完成→下一写入”事件，不以源代码语句先后替代异步完成。
公开参照展示了片上迭代结果和按层交接；具体能否走 FIX→UB/L1 仍须针对 DSL 版本验证。[^solve]

## 流水排布、同步关系与数值精度

建议的单 kernel、单 launch 分工如下；S0–S5 是本文设计阶段，不是公开参照的函数编号：

| 阶段 | 引擎候选 | 计算与主要依赖 |
| --- | --- | --- |
| S0 | Vector | gate/scan，可与不依赖 gate 的 raw KKT 并行 |
| S1 | Cube | raw KKT；完成后 Vector 才能读 R |
| S2 | Vector SIMD | norm、L、两叶块逆；等待 S0/S1 |
| S3 | Cube | 交叉块两乘；等待叶块逆与 L10 发布 |
| S4 | Vector→Cube | A pack、Ak/Au 列缩放后计算 W/U |
| S5 | Vector/搬运 | W/U 转换及输出，等待 Cube 完成并释放槽位 |

S3 期间尝试处理下一任务的 gate/scan；S5 期间尝试预取下一输入。
先保留正确的 ready/done 双向交接，再调整等待位置；生产者的“可读”与消费者的“可覆盖”是两件事。
公开参照在 AIV 发布与 AIC FIX 完成处交替握手，并使用引擎内事件保护搬运/矩乘。[^solve]

### 提前下一 task 准备的依赖证明

一次只移动下一 task 的一个准备阶段，保留当前 task 的计算顺序：

| 并行候选 | 可提前的前提 | 必须保留的旧数据/完成事件 |
| --- | --- | --- |
| 当前 S3 与下一 gate/scan | 下一 g/beta 读取不依赖当前块修正 | 当前 G/beta 活到当前 Ak/Au pack；下一 scan 使用独立槽或已释放区域 |
| 当前计算与下一 raw KKT 预取 | 下一 R 的生产者已完成，目标 UB 确认空闲 | R 可读事件、MTE2→Vector 可见性，旧槽最后消费者完成 |
| 当前 W/U 与下一 norm/L/leaf | 下一 R/G 就绪，矩阵槽和输出槽不冲突 | 当前 Cube 操作数读取、FIX 输出和 Vector/MTE3 消费释放事件 |

下一 task 的 prepare 与跨 chunk 状态递推无关，但不能因此跳过 ready/done。
提前等待一个必须在当前任务释放后才能产生的下一 ready，可能构成死锁；先画出生产与消费链。
若预计算下一 head，也必须按组内 head 索引处理配对和尾部，不能误用全局 hv 判断奇偶。

每个复用槽至少登记以下五项，不固定旧实现的缓冲名或 flag 编号：

```text
生产者及写入范围 -> 可读事件 -> 最后消费者及读取范围
               -> 可覆盖事件 -> 下一写入范围
```

Vector 发布 UB→L1 后要复用 UB，须等 MTE3 读取完成；仅 store→load 屏障不能保护搬运的读后写。
Cube 的“结果可读”也不等于 Vector 已读完，必须另保护下一 FIX 覆盖。
双缓冲增加的 UB/L1、对齐和编译器临时量应计入容量，握手在主路径、小 Q、奇数 Q 和退出时均闭合。

### 预取随关键路径变化重新评估

叶块求逆或扫描变快后，下一 raw KKT 回读可能进入关键路径，曾无收益的预取可以重新测试。
重新测试前必须提供新流水，指出目标 copy、其后消费者及新增空闲区，不能只重复旧参数。
例如 `[64,64]` FP32 R 需要 16 KiB，可尝试复用尚未用于 W/U 输出的 pack 区，
但必须证明预取前旧值已读完、norm/L 的最后读取先于输出 pack 覆写，且两 AIV 不交叉覆盖。

分别观察 copy 起止、norm/L 消费起点、当前任务是否被 MTE 队列竞争拖慢及完整主用例时延。
保留原 R 生产路径与必要握手；把 FIX→GM→UB 改为 FIX→UB 是另一优化轴，需要单独验证
DSL 版本、目标布局和双 AIV 可见性，不能把“预取前移”当作省去 GM 路径的证明。

建议 gate/scan、norm、L、叶逆保留 FP32；Cube 输入格式、交叉块精度和转换位置必须在设计中显式写出。
前缀树扫描与线性累计、联合多行与编译器 FMA、先折叠后 cast 与先 cast 后乘均需独立比较。
具体 DSL 写法查阅 [Copy 与同步](../../../dsl/copy-and-sync.md) 和
[Tensor 布局与存储](../../../dsl/tensor-layout-memory.md)，不把 AscendC 代码中的 API 或编号直接移植。

# 优化决策

先确认总时延的关键路径，再一次只改一个轴；下表全部是候选，不声明已经提速。

独立重写时，先在 DESIGN 中冻结任务粒度、lane 布局、槽位和阶段依赖，
再比较扫描前移、两行共享和关键路径预取。若主瓶颈是跨 task 等待，先改变流水组织，
而不是仅扩大行展开；若瓶颈是叶块历史读取，先比较两行共享的实际 load/fence。
共享 K 是另一调度轴，节省一次 KKT 不保证胜过相邻 chunk 配对的交叠。

| 观测与前提 | 修改对象及预期变化 | 代价、不变项及回退条件 |
| --- | --- | --- |
| chunk 多但活跃核少 | 把 chunk 加入任务轴，观察有效并行度及尾核时长 | 保持独占输出和 hv→hk；小任务调度开销上升则回退 |
| 叶逆 load/fence 显著 | 两行共享历史加载，观察向量指令及等待 | 活跃值增加；保持逐行累加顺序，临时量/scalar 增长或总时延不降则回退 |
| 32 阶叶块的 64-lane 运算含结构零 | 比较双叶块半向量布局，观察有效 lane、打包和读取 | 两半系数与 AIV 发布单独证明；重复计算、shuffle 或临时量代价抵消收益则回退 |
| Vector 叶逆处于关键路径 | 比较 MCH Cube 叶块和 MBH 合并的完整路径 | 保持单 kernel 与精度；新增 MMAD/FIX/MTE/等待超过被隐藏成本则回退 |
| 中间矩阵 GM 往返显著 | K 驻留 L1、交叉块中间量片上复用，分开实验 | 片上容量和槽位压力增加；精度、所有权或流水停顿恶化则回退 |
| Cube 计算时 Vector 空闲 | 提前下一任务 gate/scan，观察两引擎重叠 | 旧系数必须活到旧 pack 完成；等待转移却无总收益则回退 |
| 输入等待且有空闲 buffer | 提前预取 raw KKT 或输入，观察等待是否被隐藏 | 先证明别名区已释放；MTE 竞争、覆盖或尾任务错误则回退 |
| 同一 K 被多个 hv 重复使用 | 配对共享 K 的 head，观察 K/KKT 搬运与矩乘 | g/beta、L/A 不共享；解码成本可能抵消收益，NG 奇数/1 必须回退 |

# 约束

- 本文的形状、激活、epsilon、dtype 和四输出是设计约定，不是附带实现的支持声明。
- 首版仅讨论整 chunk；token 尾块、varlen、非连续布局、更宽 K/V 必须另做 mask 和列覆盖设计。
- chunk 独立只适用于本文 prepare，不能外推到跨 chunk 状态递推或 CP。
- 禁止用公开 GDN/AscendC 参照替本卡 DSL 候选背书精度或性能，也不从接口名称推导数学等价。[^wu][^solve]

# 失败表现

以下是根据数据依赖提出的首查方向，不是已复现的唯一根因：

- G 正确但 A/W/U 异常：依次比对 R、r、L、叶块逆、交叉块和 pack。
- dump 正确而下一 vec 读取异常：在消费者内部检查值，缩小到“写→第二 vec 读→pack”；核查 view、地址和存活期，不盲加 barrier。
- A 正确但 W/U 异常：检查列缩放、指数底数、cast 边界、hv→hk 和 AIV 输出行归属。
- 仅奇数任务或 head 边界错误：检查独立解码、配对 guard 和预取偏移。
- 局部流水缩短但总时延变长：检查临时量、scalar 开销、MTE 竞争及尾核；没有 spill 证据不应断言 spill。

# 验证方法

先用本文完整公式建立独立 reference，再实现 kernel；不能把候选输出作为 reference。
建议从以下形状矩阵起步，均 C=64、Kdim=Vdim=128；这些是建议测试输入，不是本次提交的测试结果：

```text
(B,HK,HV,T) = (1,1,1,64), (1,2,2,192), (2,2,4,128), (1,16,32,11264)
```

逐输出检查有限值、atol/rtol、匹配比例及最大误差；容差由调用方合同确定。
修改调度另补 Q 奇数、NG=1/2/3/4、跨 batch/head 配对；修改数值路径另补极端 gate、
接近零范数 K 与独立随机种子。匹配比例为 1 不等于 bitwise equal。

验证应包括单 kernel/单 launch 与双引擎结构审计、输出完整覆盖、buffer 存活期及 flag 配对。
先采父版本上板 PipeUtilization 和 simulator，再改一个候选，做精度、anti_hack 与同窗口多次计时。
只比较相同 shape、设备、编译环境及计时口径；主用例与多 case 平均、单 block 仿真与完整 kernel 分开。
流水占用比例可重叠不可相加，缺失的导出事件不能被解释为省去计算。

### 可复现验证门禁

由调用方项目预先登记性能验收目标，不能将本卡的设计候选视为已证明的性能。
冻结以下测量合同后再优化：

- 唯一 kernel/单 launch，范围为四输出 prepare，不能排除某个阶段或混用全 GDN 计时。
- 调用方 reference、dtype、atol/rtol、匹配率和压力用例；旧容差下通过不等于新合同下通过。
- 同一健康空闲设备、编译栈与输入；预热次数和采样次数在整个迭代中保持一致。
- 正式均值、标准差、源码 SHA、真实 launch grid、资源预算、采用/回退决定及双流水范围。

先通过全部精度，再测速并立即登记。采纳需越过预先登记的噪声门槛；最终恢复真实最佳版，
以相同合同重新做独立精度、计时和新鲜双流水。报告是否达到项目预先登记的性能目标，不能用最低单次值、
单 chunk 仿真或不同设备/容差的旧成绩替代最终均值；目标附近的波动另用独立复测披露。
迭代次数只是尝试预算，不能作为性能达标证明。

本卡不包含设备实测结论。通过实验后，应把环境、源码身份、原始证据和回退结论记录到项目 learned，
而不是把私有日志路径或不可访问的源码 commit 加回本卡。

[^wu]: 上列固定提交的公开 RecomputeWUFwd 文档，支持 W/U 公式、grouped-head 规则和 chunk 接口对照，不覆盖本文额外定义的融合步骤。
[^solve]: 上列固定提交的公开 AscendC SolveTri 实现，支持独立 tile、片上块合并和生产消费握手的架构参照，不证明本文 DSL 候选已经实现或提速。
