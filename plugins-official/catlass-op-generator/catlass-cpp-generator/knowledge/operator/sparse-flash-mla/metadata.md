---
type: "operator"
title: "SparseFlashMla 配套 metadata 算子与调度协议"
description: "SparseFlashMla 配套 metadata 算子与调度协议；适用于共享 KV 的窗口、连续压缩与稀疏压缩三模式。"
tags: ["catlass-cpp", "sparse-flash-mla", "metadata", "tiling", "scheduler", "workspace"]
status: "draft"
operator_families: ["sparse-flash-mla"]
architectures: ["atlas-a2-a3", "ascend910b"]
generated: {"by": "swa-general-final"}
verified: []
sources: [{"id": "interface-contract", "resource": "knowledge/operator/sparse-flash-mla/interface.md", "title": "配套入口和参数契约", "kind": "contract"}]
---
# 接口与概念

# 用法

在 interface 阶段确定生产/消费协议，design 阶段推导任务表与资源，implementation 阶段先验证 FA-only，再与主算子联调；validation 阶段分别验收两个公开入口及组合调用。

当用户要求生成 `sparse_flash_mla` 时，本页为必读知识，必须同步生成 `sparse_flash_mla_metadata`，无需用户另行要求 ABI 兼容。配套算子应有独立可调用入口、实际的任务表生成实现、对应构建产物和验证；主算子使用本次生成的 metadata 输出。不得只列出签名、调用环境中已有的 metadata 算子或用常量表代替生成实现。

下面给出910B/arch22的任务表协议，生产端与消费端共同遵守。协议容量、块单位与真实可用核数分开处理。

公开接口与接线先读[完整接口契约](interface.md#api) §3–6：metadata 的 27 个 Python 参数、两段式 C 签名、None 默认归一化及与主接口的对应关系均在那里维护。本页只维护表内协议；声明里的可选输入仍需按模式校验，metadata_input 字典不是公开接口或输出。

# 代码模式

## 数据路径与存储层级

输出 int32[1024]，先清零。FA 容量36×9，FD容量72×8，共900个word，其余124保留。
容量不是实际启用核数；实际核数由环境取得，超容量不能静默丢弃任务。

| 表及起址 | 字段顺序 |
|---|---|
| FA i：9*i | enable；start(BN2,Mblock,S2block)；end(BN2,Mblock,S2block)；firstFDworkspace；maxS2loops |
| FD v：324+8*v | enable；BN2；Mblock；workspaceIndex；parts；Mstart；Mnum；reserved |

BN2=b*Nkv+kv_head，块坐标不是 token 偏移；区间左闭右开、字典序连续，相邻参与核的
end=start，末尾哨兵(B*Nkv,0,0)。未工作核关闭。默认完整M任务、FA-only时起止S2block
为0，FD区全零；CSA Mbase=G，SWA/HCA Mbase=256，S2base=512。生产与消费两端的
单位、循环上界与 workspace 必须一致。

按实际有效长度、窗口、压缩可见量与索引容量估算任务成本，将完整M任务分成连续区间。
均分或加权边界可不同，必须覆盖完整且无重叠。旧实现的压缩位置采用Lc*ratio+residual；
如果调用者标杆用Lo且二者不同，先在[计算契约](computation.md)中解决差异，不能以
“沿用metadata”绕过它。旧CSA主接口K=1..8192，metadata的K=0可表示其他模式；这些
属于旧接口限制，不自动约束新标杆的数学有效域。

## 算子算法

metadata 根据有效长度与模式估算任务成本，产出主 kernel 使用的完整任务区间；不计算 attention 或读取稀疏 ID。

### 新表字段必须可由公开生产接口推导

采用本轮自定义版本化表时，先对每个header/cursor字段写明“公开metadata参数→规范值→编码→main消费方式”。metadata公开接口没有Q/KV tensor、主输入FP16/BF16标记、softmax scale、sink值、PA页表或真实stride。不能因为诊断C ABI共用一个完整descriptor，就误以为正式metadata也能获得这些数据。

| 字段类别 | 允许来源与消费方式 | 典型错误 |
|---|---|---|
| batch/head/D、最大长度、布局、mask/window | 本次metadata公开属性，经校验编码 | 写固定测试shape或上次调用值 |
| 每batch有效长度与存储前缀 | 本次cu/used tensor及缺省规则 | 只读Q而缓存旧KV长度；把effective length当storage base |
| 核数、任务起止、tile策略版本 | 实际设备能力和上述调度输入共同推导，main按同协议恢复 | 表核数与launch blockDim不一致；容量外截断任务 |
| Q/KV FP16/BF16分支 | main从真实Q/KV descriptor判定；若其不影响调度，表不保存dtype或将相应word定义为固定保留值 | metadata生产端写默认FP16，main又要求该word等于BF16，导致配对调用拒绝合法输入 |
| scale/sink/LSE开关、原生stride、页映射 | main本次接口/tiling/control中的真实值，不由metadata臆造 | 为补私有表字段偷改公开metadata签名，或由host先伪造完整表让AICPU只复制 |

若tile策略确实依赖metadata接口看不到的dtype或stride，应重设计为该接口可决定的共同调度，或让main在同一任务覆盖语义内选择物理micro-tile；不能对正式接口增加未授权参数。metadata定义逻辑工作，main可以在逻辑工作内部细分Cube/Vector块，但双方必须对任务边界、单位和写者保持一致。诊断planner和正式AICPU需要通过同一解码契约，不必用不在公开输入域内的字段校验彼此。

独立测试至少包含：相同合法metadata配同shape FP16与BF16主输入；固定Q改变KV有效长度；改变cu存储前缀但保留相同有效长度；同一逻辑长度切换合法PA页表。任何本次输入变化只要影响表字段就重新生成，包含表内编码的cu前缀或由其推导的cursor；不允许仅更新main地址而继续使用过期表。若某项确实不影响表，证明其独立性后仍由main读取本次真实地址/页映射。比较输出、覆盖范围和reserved words，不能只比较INT32[1024]的形状。

## 分核策略与基本块切分

1. 校验长度tensor、batch、layout、ratio、K、设备核数及可选输入组合，规范化每batch有效长度。新metadata入口实际执行这一步，不能硬编码某case的表。
2. 对每个BN2枚举Mblock：CSA按G行一个query，SWA/CFA按Mbase256的逻辑行块；最后有效行数为min(Mbase,Lq*G-Mstart)。空batch不生成有效计算任务，主接口定义的零输出另外覆盖。
3. 按每行可见窗口/前缀或min(K,C)估计S2工作量，任务成本可先用 `Mvalid * ceil(selected_estimate/512)` 加gather成本。metadata接口没有稀疏ID，估算只能用长度和K上界；主kernel仍检查真实ID，不能认为估算量就是有效选中数。
4. 先取 `used=min(available_AIC, task_count, 36)`，用任务成本前缀和找连续分割边界，每个非空核至少一个完整M任务。把任务起止转为(BN2,Mblock,0)，全局末端用(B*Nkv,0,0)。实际启用核数/launch必须一致；设备超表容量时选择可表示核数并记录，不得静默截掉任务。
5. 清零1024word，写FA enable/start/end/maxS2loops，FA-only的firstFDworkspace=0、FD全零。maxS2loops按消费端同一枚举方式取该核最大循环需求，不拿总KV容量代替。
6. 独立decoder展开每核区间。集合应等于全部有效任务、计数恰好一次、起止单调、禁用核和保留区全零。用接口允许的较小核数与多核两种合法表驱动同一主kernel，比较输出；合法调度表不要求逐word与某一均衡算法相同。

### 普通SWA游标中的零storage与零有效长度

采用普通SWA的新版本化表时，先声明任务枚举覆盖有效行还是storage行。若枚举storage以同时负责padding清零，两种“空batch”具有不同含义：

| batch情况 | 任务与写者 | 游标推进 |
|---|---|---|
| `cu_q[b+1]==cu_q[b]`，没有Q存储行 | 本batch没有输出元素，也不产生虚构task或跨核通知 | 在解码下一个task前连续跳过，允许连续多个这样的batch |
| Q storage非空，但`used_q[b]==0`或KV有效长度0 | 已分配O/LSE仍须有确定的零写者；可以保留零计算task，或有独立覆盖完备的清零路径 | 保留的真实task照常计入local_i并参与两AIV握手；只跳过矩阵数学 |
| Q/KV有效，但存在因果无效前缀或storage尾padding | 当前task可混合有效与无效行，按每行有效性计算/清零 | 不能用一个task有工作推出全部行有效 |

每核起点用前缀查找跳过空区间，只解决第一个task。**每次**跨batch后，decoder仍要在`b<B`的前提下循环跳过没有剩余storage任务的batch，再解码下一真实task。只写一次`++b`会在空batch上发出rows=0的假task、消耗metadata计数，最终漏掉后面的非空任务。跳过空batch不递增流水`local_i`、不翻转GM slot、不消费ready；真实零计算task则按上述已选择的写者方案推进。AIC与两AIV必须采用完全相同的枚举规则。

可直接演算的版本化M128例：H64、T=2，`cu_q=[0,2,2,5]`，三batch的Q storage为`[2,0,3]`。若一个逻辑core负责全部storage任务，正确序列为`(b=0,q0=0,rows=128)`、`(b=2,q0=0,rows=128)`、`(b=2,q0=2,rows=64)`，总计3个task，local_i=0/1/2，最后task的第二AIV零行但仍通知。不能插入`(b=1,rows=0)`作为第四项或用它替换第三项。再将中间空batch变成两个连续空batch，或把其storage改为非空但used_q=0，分别检查“跳过”与“明确写零”的区别。

本例保持整个TND的T1>0，不放开原main对全空tensor的契约；独立metadata的B=0合法性单独验证。CPU枚举记录起点/终点和全部task，设备测试检查真实控制区解码、所有O/LSE以及未被写坏的相邻batch。

## FD是单独的优化分支

完整任务不足以占满核心且S2长时，才考虑在任务内部按512列边界划分。FA表端点S2block非零代表部分任务，FD表描述同一Mblock的partial槽、parts和负责归并的M行范围。工作空间必须能容纳所有在途partial，禁止不同split写同一(u,l,m)槽。先验证2/3/多split、边界非末块和空片，再评估收益。

归并使用[开发页S6](development.md#stages)的指数重缩放；普通SplitK求和错误。sink注入一次，partial完成与归并读取建立可靠同步。没有实现/验证FD时显式禁用该调度分支，不能写非零FD表却由主kernel忽略。若因此性能不满足需求，任务仍需继续优化而不是宣称完成。

## 交付与验证

- 当前函数型 golden 并不产生该表。`metadata_input` 字典只是调度输入集合，不能当作
  metadata 输出的独立标杆；要单独实现其协议并验证消费端。
- 直调工程也要交付独立可调用的 `sparse_flash_mla_metadata` 入口、实际任务表生成实现及联调样例，可由 host 计算并上传表；只有主算子内的私有调度 helper 不满足成对交付。正式 ACLNN/AICPU 包必须包含 metadata 的 AICPU 实现、注册、两段式接口、编译和调用，不能用 host helper 或系统已有算子替代本次生成的配套算子。
- 验收先确认两个算子的实现及所选交付形式的构建产物齐全，再从生成的 metadata 公开入口实际产表并传给生成的主算子；缺少配套算子或未完成联调时，不能将整个任务标为完成。
- 检查保留位、关闭核、容量、首尾与相邻区间、完整任务覆盖；覆盖任务少于核数、ragged、
  空batch和M/S2尾部。同一主kernel消费两种合法表后输出一致，比逐word相等更有意义。
- 不因存在FD字段就打开split-S2；归约必须合并各片(m,l,u)，sink全行仅一次，并有完整
  workspace/完成同步。普通GEMM SplitK求和不适用。
- 独立测调度生成时延，再看主kernel最慢核和端到端；改变调度输入后不能复用过期表。

# 约束

公开参数、模式和有效域按冻结契约逐项实现；未支持的组合明确拒绝，不静默删参。

# 失败表现

metadata 先报错只能证明该入口拒绝；主入口须独立验证。错误 shape、槽复用或长度单位会造成漏任务、越界或数值错误。

# 验证方法

按[验证清单](validation.md#coverage)分别调用新 metadata、新主算子及组合调用；核对所有输出、任务覆盖和实际运行日志。

## 流水排布、同步关系与数值精度

metadata 就绪后主算子才能消费；每次长度/模式/调度输入改变重新生成，不能复用旧任务表。device 生命周期与舍入顺序见[开发知识](development.md#stages)。

[^interface-contract]: [配套入口和参数契约](interface.md#api)。
