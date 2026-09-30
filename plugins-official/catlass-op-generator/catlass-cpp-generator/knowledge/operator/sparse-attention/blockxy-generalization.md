---
type: "operator"
title: "block_sparse_attention Arch22 blockX/blockY 泛化约束"
description: "blockX/blockY 泛化为 128 倍数时的接口约束、天然支持点、必须参数化点与容量硬边界。"
tags:
- catlass-cpp
- sparse-attention
- block_sparse_attention
- BlockSparseAttention
- 块稀疏注意力
- arch22
status: "stable"
generated: {"by": "process:bsa-blockxy-generalization-survey"}
verified: [{"by": "process:static-code-survey"}]
sources: [{"id": "sparse-attention-blockxy-generalization-source", "resource": "audit:custody/sparse-attention#blockxy-generalization", "title": "blockxy-generalization 固定来源（提取侧独立保管）", "kind": "audit-record"}]
operator_families: ["sparse-attention"]
architectures: ["DAV_2201"]
---
# 接口与概念

本 concept 是"blockX/blockY 泛化为 128 的倍数"这一 BSA 变体的约束清单，面向算子生成。
证据分级：**文档事实**=V2 接口文档直接可见；**静态调查事实**=经固定版本源码核对的设计约束，
取证记录由提取侧独立保管、不随生成侧分发；**静态推断**=经核对但泛化变体尚未构建实测的结论。
本 concept 不含任何实现仓路径、行号或可读取链接。

## 算子算法

**文档事实**：`blockShapeOptional=[blockShapeX, blockShapeY]` 定义稀疏块大小，缺省 `[128,128]`；
`blockSparseMask` 为 INT8 `[B, N1, ceilDiv(S1,X), ceilDiv(S2,Y)]`，1=该 Q 块与该 KV 块参与计算；
传 nullptr 退化为稠密。q/kv seqlen 不需被 blockShape 整除，向上取整分块，非对齐 tail 天然合法。
blockShapeY 在 Atlas A2/A3 上须为 128 的倍数（950PR/950DT 为 16 的倍数）；文档对 blockShapeX 仅
要求 >0，但本变体的生成目标把两者都收紧为 128 的倍数，即 `(X,Y) ∈ {(128m, 128n)}`。
其余接口约束（D∈{64,128}、FP16/BF16、GQA N1%N2==0、无滑窗/attenMask/paged、确定性计算、
TND 时 actualSeqLengths(Kv) 必传且两者须同时配置）与固定块版本一致。[^sparse-attention-blockxy-generalization-source]

## 分核策略与基本块切分

**静态调查事实**：任务粒度为"1 个头 × ≤qsBaseTile 行 Q"的任务块（qsBaseTile 常用配置 128），
与 mask 块大小解耦。每个 mask X-block 拆成 ⌈X/qsBaseTile⌉ 个任务，同一 X-block 的所有任务
共享同一行稀疏索引；任务索引分解与尾块逻辑对任意 X=128m 成立。KV 方向：选中块经 mask 预处理压缩为逻辑连续序列，QK/PV 的
gather 按"块索引×Y + 块内偏移"寻址、跨选中块边界自动拆分搬运，对任意 Y=128n 成立，末块
`seqlen%Y` 尾长同样正确。mask 预处理只操作块索引空间（Q 块数×KV 块数），与 X/Y 数值无关。
**静态推断**：X/Y 的 128 倍数化不需要新增分派/tilingkey 分支，也不需要改动任务与 GEMM tile
结构（GEMM tile 的 M/N/K 绑定的是任务块与栈长，不是 mask 块）。[^sparse-attention-blockxy-generalization-survey]

### 基础块与 mask 块的换算关系

**静态调查事实**：mask 块只是**索引粒度**；数据通路始终以固定大小的**基础块**推进，两个粒度
按下述换算解耦。术语约定：tiling 给定配置上限 **baseTile**（qsBaseTile/kvSBaseTile），运行时
逐基础块计算实际量 **act**：`act(i) = min(baseTile, 剩余量(i))`——X<qsBaseTile 时 act=X，
整数倍时 act 恒=qsBaseTile，非整数倍时末块 act=X mod qsBaseTile。
**kernel 获取 baseTile 有两种方式，是否由 tiling 传入看整体设计决定**：
(a) **kernel 直接定义**（arch22 现状）：Q 任务行数与 KV stack 窗口为 device 侧常量
    （常用配置 128 / 512 档，窗口注释支持 512/1024 档），host 不计算、不传 baseTile 字段，
    act 的 min 逻辑全在 device 运行时。实现最简、device 自包含、host 不动；代价是换档须
    同步修改双写常量（同源风险）、不能按 shape 自适应。
(b) **tiling 传入**：host 按 shape/精度档计算并下发——Q 侧可用 `min(blockShapeX, qsBaseTile)`
    范式（X 超上限钳制、不足取 X），KV 侧可按 Y 对齐取值（Y>上限时取"≤上限的最大约数"
    以对齐块边界，Y<上限取 Y 本身）。可按 shape/精度档自适应；代价是 host 增加计算与
    字段、host/device 公式须一致性校验。
若变体承诺域固定（如本变体 X/Y 仅为 128 的倍数、档位数少），(a) 通常足够；
若希望窗口随 Y 档位对齐或档位较多，宜选 (b)。

- **Q 方向**：基础块（任务）行数上限由实际 **qsBaseTile** 决定（常用配置 128，host 与 device
  两处写死、须同源），每任务 1 头。blockX 同时决定每个 mask X 块的基础块**数量**
  （⌈X/qsBaseTile⌉ 个）与各基础块的**实际 qs**（act(i) = min(qsBaseTile, X − i×qsBaseTile)，
  X 为 qsBaseTile 整数倍时恒满载，否则最后一个基础块欠载——功能正确、效率损失）；
  同一 mask X 块的所有基础块共享同一行稀疏索引。**X<qsBaseTile 时** ⌈X/qsBaseTile⌉=1，
  每 mask X 块一个 act=X 的欠载基础块（接口合法、效率低）；注意 Q 侧**不做跨 mask X 块拼接**
  （稀疏索引按 mask X 块逐行授权，单任务横跨多 X 块需同时查多行索引，结构不支持），与
  KV 侧 Y<窗口的拼接不对称。本变体以 X%qsBaseTile（常用配置下即 128 的倍数）校验排除此域。
- **KV 方向**：选中 Y 块按稀疏索引顺序拼接成逻辑连续序列（gathered 序列）；基础块是该序列上
  的装载窗口，**窗口长度（kvSBaseTile/kvSTile）由实际 tiling 决定**——常用配置为 512 档
  （实现注释支持 512/1024 档；取 256 等更小值同样合法，上限只受 S/P/OTmp buffer 131072 元素
  ÷ 行数约束），装载函数按参数化窗口运行，**窗口可带 inner offset 起始，不与 Y 块边界对齐**。
  窗口填充的寻址换算：起始偏移 ÷Y → gathered 块号、起始偏移 mod Y → 块内位置、查稀疏索引表
  得原始块号 → 原始序列偏移；按 min(当前 Y 块剩余, 窗口剩余) 逐段搬运直至填满：
  - **Y < 窗口**（如窗口取 512 档时 Y=128/256）：一个窗口内容由多个选中 Y 块**拼接**而成；
  - **Y = 窗口**：起始对齐时一一对应，**但窗口带 inner offset 时仍跨两个 Y 块拼接**；
  - **Y > 窗口**（如窗口取 512 档时 Y=1024）：一个 Y 块被多个窗口**各取一段**；
  - 最后一个原始 Y 块按 `seqlen%Y` 尾长截断；
  - 拼接是**逻辑效果**：物理上每段是独立的 GM→L1 拷贝（按边界拆成多次搬运），并非预先
    合并成单条大拷贝。GEMM 的 L1 chunk（常用配置模板 N=128）同理是编译期模板参数而非定值，
    受 L1/L0 容量约束，本变体不改动。
- **静态推断**：块倍数化只改变上述换算的"块数"参数，不改变基础块本身的大小上限与推进
  结构；任何把基础块大小与 mask 块大小挂钩的改写（如让任务行数随 X 变化）都会连带触碰
  数据路径与容量约束（见# 约束），不属于本变体范围。[^sparse-attention-blockxy-generalization-survey]

## 数据路径与存储层级

**静态调查事实**：GM→L1→L0→UB 的数据通路、S/P/OTmp 三 buffer 流水只取决于任务 Q tile
（qsBaseTile，常用配置 128）与 KV stack 窗口（kvSBaseTile，常用配置 512 档）——两者均为可配置值，
与 X/Y 无关。容量硬边界
（DAV_2201 公开硬件参数与固定 buffer 常量）：S/P/OTmp 每 buffer 元素数上限 131072
（常用配置 128×512=65536，余量 2 倍）；UB 总量 192KB，其中 mask 预处理占用约 177KB、
online softmax 要求列向 padded 长度 ≤8192(float)/16384(half)；L1 总量 512KB，常用配置下
QK+P 双 buffer+V 合计约 352KB；L0A/L0B 64KB、L0C 128KB，L0 tile 上限 [128,128]；
每 AIV 子核行数上限 256。[^sparse-attention-blockxy-generalization-survey]

## 流水排布、同步关系与数值精度

**静态调查事实**：三 buffer 流水（流水距离 2、槽位模 3 轮转）、每 stack 窗口（常用配置 512 档）
一组的 KV 循环、online softmax/rescale 的行循环（按 8192/列宽 自适应分行）均与块大小无关。
低精度（half）softmax 快路径要求"对齐元素数/128 为 2 的幂"，仅列长 512/1024 时进入，
其他列长走通用 tail 路径（功能正确、性能路径不同）。**静态推断**：块倍数化不改变流水结构
与数值精度语义；若顺带引入新 stack 窗口档（如 768），需确认落入通用 tail 路径。

# 用法

生成 blockX/blockY = 128 倍数的 BSA 变体时，先读本 concept 与 V2 接口约束，再按三类清单行动：
天然支持的部分直接复用设计，必须参数化的部分逐点修改，硬约束作为设计红线。
本 concept 是静态调查结论；生成的变体必须经实际构建、精度与容量边界用例验证后才算成立。

# 代码模式

**必须参数化/修改的设计点（静态调查事实，不含实现位置信息）**：

1. host 校验：已有 X>0、Y>0 与 Y%128（A2/A3）校验；本变体承诺 X 也是 128 的倍数时，须追加
   X%128 校验（实现对任意 X 可跑，但接口只应承诺 128 的倍数）。
2. workspace 公式：稀疏索引 workspace 含一个随 ⌈X/128⌉ 放大的富余系数，X>128 时成倍超配
   GM workspace（安全但浪费）；精确化可改为按"Q 块数 × 对齐后 KV 块数 × 4B"计算，读写侧无需改。
3. 常量单一来源化：任务 Q 行数（qsBaseTile，常用配置 128）在 host 与 device 各写死一处，
   两侧的块数换算公式必须同源同步修改。
4. gather chunk=128 token 的隐含等式：QK/PV 的 L1 装载循环隐含 gather chunk 恰为 128；
   本次泛化不动它，但若未来把 chunk 与 blockShapeY 对齐，相关推进逻辑必须同步参数化。
5. 死代码清理：原实现中存在若干只写不读的块数统计变量（其中一处块数商存在量纲错误），
   泛化时易被误读，建议生成实现时规避而非沿用。
6. 测试补齐：既有测试只覆盖 [128,128]，需补 [128,256]/[256,128]/[256,256] 及非对齐
   seqlen 用例。

# 约束

**硬约束（红线，随块/tile 增大而溢出）**：

1. S/P/OTmp workspace：`行数 × stack 长度 ≤ 131072` 元素；任务 Q tile 升 256 或 KV stack
   升 1024 恰好顶满，不能再大。
2. UB：online softmax 列向 padded 长度 ≤ 8192(float)/16384(half)；mask 预处理约 177KB，
   接近 192KB 上限，其行列 tile 不能随意放大。
3. L1 512KB：常用配置约 352KB；head dim 已被限制为 {64,128}；KV stack 升 1024 时合计约
   480KB，勉强可行。
4. L0 tile 上限 [128,128]，不能再放大。
5. 每 AIV 子核行数 ≤256，总行数 ≤512。
6. mask 寻址用 int32：B×N×Q 块数×KV 块数 < 2³¹；Y 增大使 KV 块数减小，反而缓解。
7. 低精度 softmax 快路径的 2 的幂要求（见上），新 stack 长度须确认落入通用 tail 路径。
8. **静态推断红线**：只要不同时放大任务 Q tile 与 KV stack 窗口（常用配置 qsBaseTile=128 /
   kvSBaseTile=512 档配置），块大小本身的倍数化不会触碰上述容量边界。

# 失败表现

- host 校验未收紧 X%128 时，非 128 倍数 X 的调用可能通过校验但产出未承诺的行为。
- workspace 富余系数未处理时，X=256 超配 2 倍 GM workspace，表现为内存浪费而非错误。
- 误改任务 Q tile 或 KV stack 导致 131072 元素、UB 192KB、L1 512KB、L0 64/128KB 溢出时，
  表现为编译期静态断言失败或运行期越界/数据错乱。
- mask 元素总数超 int32 时块索引 gather 寻址溢出，表现为随机错误块被选中。
- 低精度快路径未命中（列长非 512/1024）时仅性能退化，功能仍正确。

# 验证方法

1. 用例矩阵：`[X,Y] ∈ {[128,128],[128,256],[256,128],[256,256]}` × seqlen 对齐/非对齐 ×
   TND/BNSD × FP16/BF16 × LSE 开关；精度按用户 golden 与原判据，O/LSE 都验。
2. 边界用例：单块、长 seq 多块、GQA、空稀疏行行为按 family 既有语义 contract 裁决，
   不自选语义。
3. 容量用例：最大承诺块 [256,256] + 长 KV seq，确认 workspace/UB/L1 不越界。
4. 性能：比较口径与确认纪律按 family `performance-hypotheses.md`（单变量、PRE/POST、
   两次独立正式配对确认）；本 concept 不提供任何性能承诺。
5. 本 concept 的静态推断项在生成的变体完成上述实测前，不得写成已验证结论。

[^sparse-attention-blockxy-generalization-source]: 审计编号 sparse-attention-blockxy-generalization-source；V2 接口文档冻结副本由提取侧独立保管，不随生成侧资料分发。
[^sparse-attention-blockxy-generalization-survey]: 审计编号 sparse-attention-blockxy-generalization-survey；静态调查取证记录（含真实来源、版本与行号映射）由提取侧独立保管，不随生成侧资料分发。
