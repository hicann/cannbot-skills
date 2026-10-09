---
type: "operator"
title: "分段与稀疏 KV 的统一归一化 Attention"
description: "分段与稀疏 KV 的统一归一化 Attention；适用于共享 KV 的窗口、连续压缩与稀疏压缩三模式。"
tags: ["catlass-cpp", "sparse-flash-mla", "golden", "shared-kv", "softmax", "sink", "SWA", "CFA", "SCFA"]
status: "draft"
operator_families: ["sparse-flash-mla"]
architectures: ["atlas-a2-a3", "ascend910b"]
generated: {"by": "swa-general-final"}
verified: []
sources: [{"id": "interface-contract", "resource": "knowledge/operator/sparse-flash-mla/interface.md", "title": "公开输入输出契约", "kind": "contract"}]
---
# 接口与概念

适用计算特征：对一个 query/head，从窗口和/或给定索引选择 KV，所有被选 KV 共享一个 softmax 分母，再以相应 value 加权求和。可有额外 logit 只影响分母（等价零 value 的 sink）。这里索引已由调用者给出，不在算子内做排名。识别依据是数据依赖与归约关系，不依赖入口、类名或变量名。

## 分支适用边界

先按实际标杆确认K/V来源、选择集合、归一化和cast，不按函数名直接套配方。下面的三模式、成对生成、阶段流程和用例矩阵，仅用于生成本页的完整共享KV算子：K=V，支持窗口、窗口+压缩前缀、窗口+压缩稀疏索引三条路径，跨两组共享softmax状态；或用户明确要求生成对应的SparseFlashMla。只借用局部公式开发其他契约的算子时，按需阅读数学机制，不自动引入metadata、三模式或本分支的开发验收要求。标准FA、带独立RoPE的MLA和其他算子沿各自知识分支处理。

## 生成前读取完整接口

确认生成本页完整算子后，读取[完整接口契约](interface.md#api)：其中包含本例标杆的构造/forward 入口、两个完整 PyTorch 接口、四个 ACLNN 符号、host/device 参数对应、输出和版本差异。先确定本次交付层并在 docs/design.md 冻结全部参数与输出，再选组件；不能用简化的 Q/K/V 调用或一条用例的参数子集代替完整接口。这里只在已命中的知识分支内读取，不改变按计算特征选择知识的方式。

## 成对交付要求

输入虽只给标杆或等价公式，但匹配本页完整共享KV三模式契约并要求生成对应算子时，也默认交付主计算与metadata两个公开入口；不要求调用者补报函数名称。只有借用局部计算机制开发不同契约的其他算子时，才按其自身产物范围处理。

当用户要求生成 `sparse_flash_mla` 时，必须在同一次任务中同步生成 `sparse_flash_mla_metadata`，不以用户额外提出“兼容旧接口”为前提。设计、开发和验收都包含两个算子：metadata 具有独立可调用入口并实际生成任务表，主算子消费该输出；仅声明接口、转调用已有 metadata 算子或将调度隐藏在主算子的内部 helper 中，均不算完成成对生成。

此时必读[metadata 调度知识](metadata.md)和[完整接口契约](interface.md#api)，在 docs/design.md 与 docs/validation.md 中列出两个算子的实现、构建和联调验证。该规则限定本算子的交付范围，计算特征识别和通用路由保持原有方式。

# 用法

完成数学识别后进入本仓库 [CATLASS C++ 五阶段流程](workflow.md#routing--五阶段与产物)，并按本家族
[索引](index.md)选择普通 SWA 或特化知识；主核和 metadata 成对设计。
读取接口与 metadata 契约，在对应阶段加载开发和验证知识；不再经过普通 Flash Attention 配方。
三模式 SWA、CFA/HCA、SCFA/CSA 共享 QK/在线 softmax/PV，选择策略分别实现。

## 算子算法

从调用者给出的标杆入口提取以下计算节点。标杆中变量改名、用einsum替换matmul或用显式循环表达，不改变识别方式。接口已有完整定义时逐项映射，不能只按一条输入样例推断。

| 计算节点 | 本实例定位 | 需保留的信息 |
|---|---|---|
| 逻辑张量与有效长度 | `forward`、`trans_shape_to_bnsd` | layout、batch 边界、seqused 优先级；区分输入整理与设备计算 |
| 第一组 KV 的窗口 | `calculate_by_bnsd` 的 `ori_win_start/end` | 因果位置、左右窗口、空行规则 |
| 第二组 KV 的前缀/索引选择 | `calculate_by_bnsd`、`gather_kv` | 可见阈值、索引容量、终止/跳过行为、重复索引 |
| 点积与 V 来源 | `mm1_res`、`v_tile=k_tile.clone()` | 本例同一向量作 K/V，全部 D 参与点积；无独立 RoPE |
| 跨组归一化 | `row_max/row_sum/cur_attn_out` 的递推 | 两组连续共享状态，sink 只计一次 |
| 舍入与输出 | P 的 dtype 转换、末尾除法、LSE/layout helper | FP32 状态、低精度 P、输出 dtype/轴次序 |

`template_idx=0/1/2` 表示本例选择一组窗口 KV、加第二组可见前缀、加第二组稀疏 KV。
`RUN_MODE=0/1/2` 则是稠密/流式/实验数值实现，不能将两组开关混为三个算法模式。
数据生成器中的随机排列、页分配和测试名称也不是待生成算子的排名或压缩计算。

# 代码模式

对 batch 内 query t、head h，令 A 为窗口选中的 KV 列表，B 为第二组选中的 KV 列表；
拼接 J=A⧺B 是保留顺序与重复元素的列表。若一组不存在则取空列表。本实例 K=V：

```text
z_j = scale * dot(q[t,h], kv[j])
Z   = exp(sink[h]) + sum(j in J, exp(z_j))
O   = sum(j in J, exp(z_j) * kv[j]) / Z
LSE = log(Z)
```

sink 不乘 scale，也不进入输出分子。一般 K/V 不同的标杆可复用归一化机制，但必须使用
真实 V，不能强行令 K=V。没有 sink 时删除其项，而不是用 sink=0 代替。

以下表达在实数数学上属于同一模式：显式拼接再 softmax；用选中位置的布尔 mask 做完整
score；`einsum` 表达两次收缩；按 KV 块维护共同 m/l/u。布尔 mask 只有在不丢失重复索引
权重时才等价。两组各做 softmax 再相加、只保留最大 K 个分数、RoPE 分量点积之和均不能
直接判为本实例的同一计算。

实际 RUN_MODE=1 的数值路径（每块后更新同一组状态）：

```text
m = sink[h]; l = 1; u = 0
for tile in selected_A_then_selected_B:       # 原例 tile 长度最多512
    x = fp32_dot(q, Ktile) * scale
    m1 = max(m, max(x)); a = exp(m-m1)
    p = exp(x-m1)
    l = a*l + sum(p_fp32)
    u = a*u + fp32_dot(cast_q_dtype(p), Vtile)
    m = m1
O = cast_q_dtype(u/l)
LSE = m + log(l + 1e-10)                     # 此标杆的实际写法
```

流式块边界影响低精度 P 的舍入。稠密表达能证明数学关系，不能因此要求 BF16/FP16 与
分块路径逐位相同。需要冻结目标数值路径和容差；不得通过改变标杆或删 cast 换取通过。

# 约束

令 Lq/Lo/Lc 为当前 batch 的有效 q/第一组/第二组长度，r 为压缩比，wL/wR 为窗口属性：

```text
p = Lo-Lq+t
mask4: A = ori[max(0,p-wL):p+1+wR]          # wL=-1时从0开始
mask3: A = ori[0:p+1+wR]
mask0: A = ori[0:Lo]
C = min(Lc, floor((Lo-Lq+t+1)/r))           # 第二组mask3
C = min(Lc, floor(Lo/r))                    # 第二组mask0
```

切片末尾遵循张量边界。第二组前缀路径 C<=0 时为空；稀疏路径最多检查前
`min(K,ceil(C/sparse_block_size))` 个索引槽，本例 sparse_block_size=1：遇 -1 终止，
超出可见阈值的索引跳过，合法索引保留重复与顺序。`gather_kv` helper 可额外用
topk_length 限制检查槽数，但当前 `forward` 将传入的两套 topk_length 重置为 None；
不能把 helper 的能力直接当作此入口已启用的能力。负值非 -1 等未明确输入要先界定有效域。

尤其不要把历史 kernel 约束替换进用户的 golden：

| 项目 | 此标杆实际行为 | 既有实现资料的使用边界 |
|---|---|---|
| 第二组位置原点 | 使用 Lo，cmp_residual_kv 未参与该公式 | arch22 实现用 Lc*r+residual；只有两者相等或阈值相同时才兼容 |
| 前缀 -1/不可见 ID | 终止/跳过后按实际选中 KV 归一化 | 旧 gather 的列数处理不同；不能只复用其地址代码 |
| K、D、head、窗口 | 从输入参数提取 | D512、Nkv1、ratio4/128、K<=8192 是特定实现包络，不是此计算的数学限制 |
| 可选参数 | 以实际入口中的使用为准 | Python 注解、helper 签名和旧 ACLNN 的限制不互相替代 |
| 输出 ABI | TND 输出 helper 分配 FP32；LSE helper 最后转置成 [T,Nkv,G] | 与旧设备接口的 dtype/轴次序可能不同，wrapper 差异必须显式记录 |

区分输入：Lo=31、Lc=7、r=4、Lq=4、t=0、residual=0，标杆 C=7，而恢复长度原点
得到6。该输入必须用来检查是否误套已有 kernel 语义，不能为了兼容旧实现而修改用户标杆。
本例在 ori mask3/4 且 t<Lq-Lo 时直接令该行 O/LSE 为0。

# 失败表现

使用该文件的其他运行分支前先做最小执行验证：RUN_MODE=0 的 LSE 写入引用未定义的
`batch/n2Idx/s1Idx`，RUN_MODE=2 是另一条数值路径。标杆自身异常应定位报告，不能固化为
新算子的计算功能。本条目以可执行的 RUN_MODE=1 为依据，不修改用户的 golden。

## 数据路径与存储层级

数据路径：逻辑选择/PA 地址 → 流式 KV tile → QK 的 BlockMmad → mask/exp/sum →
P cast → PV 的 BlockMmad → 输出重缩放/最终除法。两次 GEMM 使用实际 Catlass 组件；
非 GEMM 选择、sink 和状态在自定义 Block/Tile 中实现。不得在 host 预算 gather/attention
后只计剩余 kernel，也不分配完整 Sq×Sk score。golden 的 BNSD 中间形态不要求公开接口改名或改布局。

## 分核策略与基本块切分

- 先按完整 query/head 任务分核，每个任务遍历两组 KV，独占 m/l/u。ragged 情况按可见
  KV 块数估算成本，保证任务完整、无重叠；不能把两组分给两个核各自归一化后直接相加。
- 索引行跨度为容量 K，块内遍历长度是实际选中量。任意 K 分片采用
  `start=i*tile; count=min(tile,L-start)`；K1537 全可见、tile512 时为512+512+512+1。
  原索引槽与过滤后 KV 序号是不同坐标；保留过滤规则，奇数成对读取的第二槽单独判界。
  物理补齐列在 max/exp/sum 前屏蔽为 -inf，PV 的相应 P 置0；补齐列不进入分母，
  也不读取越界索引或沿用上一个 tile 的残留数据。
- TND 的 batch 基址来自前缀和，PA 由 token/page_size 查物理页，再加页内偏移；两组 KV
  页表与页大小独立。分页是存储布局，不能改变可见集合。
- 实际低精度 D512 可从 AtlasA2 的 `MmadAtlasA2Pingpong<true>` 组件探针开始：L1
  `<16,128,128>`、L0 `<16,128,64>`、FP32 C，QK 的 B 为 ColumnMajor、PV 为 RowMajor。
  这是组件探针的选型起点；目标环境的
  header、容量与完整融合性能必须核验，不把它当作所有 shape 的最优配置。
## 流水排布、同步关系与数值精度

- QK/PV 复用 gathered KV 可减少读取，但应延长相应 buffer 的生命周期。按每个 buffer
  标明 producer 完成、consumer 等待与最后消费后归还；预热/稳态/排空分别验证。
  槽数、事件、UB/L1 占用按真实存活资源计算，不照搬 D128 或另一个架构的流水。
- 小 G 可补齐物理 M，逻辑 shape 和输出不变；低占用不直接套普通 GEMM SplitK。
  split-S2 需归并各片 m/l/u，并保证整行 sink 恰好一次。

Attention 数学公式本身不规定独立 metadata 算子或固定1024字协议；要求生成 `sparse_flash_mla` 时，成对交付属于本工程契约，必须同步实现 [metadata 算子与调度协议](metadata.md)。仅借用本页计算机制开发其他 Attention 算子时，可按其自身契约设计调度；不能用这种通用情况省略已要求交付的 `sparse_flash_mla_metadata`。

# 验证方法

验证调用者实际入口，并至少有一份不依赖原函数名/变量名的等价表达。覆盖单路/双路、
稀疏/前缀、非零 sink、重复 ID、-1 终止、跳过未来 ID、空行、ragged、输出 layout/cast；
以 Lo31/Lc7 例分辨位置公式。K 覆盖1/511/512/513/1025/1537/8191/8192，以及实际用户
契约允许的空列表与可见量<K；不要把旧 API 的负例未经核对移到新数学契约。

优化逐项测：gather 占比高时试 KV 复用，观察读取量与 Cube 等待；长短任务混合时试按
完整任务成本分核，观察最慢核；C/V 空泡明显时调 tile/槽数，观察等待与资源占用。每项
都保持选择集合、sink、cast 不变，精度或端到端时延回退则撤回。分别记录输入准备、主
kernel 和需要时的调度生成时间，不把额外 host 计算排除后宣称加速。

[^interface-contract]: [公开输入输出契约](interface.md#api)。
