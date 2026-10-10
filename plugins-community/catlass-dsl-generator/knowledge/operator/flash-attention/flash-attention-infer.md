---
type: CATLASS DSL Operator Example
title: Flash Attention Inference
description: Ascend950 Prefill Flash Attention 的 Online Softmax、GQA、tiling、流水和验证边界；含无源码重写骨架、变体派生图谱与实测硬约束。
tags: [catlass-dsl, operator, flash-attention, attention, prefill, online-softmax]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:fa-dsl-9-operator-delivery, at: '2026-09-11T00:00:00Z'}
sources:
  - id: kernel
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/flash_attention_infer/flash_attention_infer.py
    title: Flash Attention inference kernel and host validation
  - id: tiling
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/flash_attention_infer/fa_tiling.py
    title: Flash Attention tiling data implementation
  - id: guide
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md
    title: Flash Attention inference guide
operator_families: [flash-attention, attention]
arch: [c310]
---

# 接口与概念

## 算子算法

该样例实现 Prefill Flash Attention：以 Q/KV block 做 `QK^T`，用 Online Softmax
维护逐行 max 与 sum，再计算 `P @ V`，避免在 GM 物化完整注意力矩阵。数据流组合
Cube MMAD、Vector softmax/归一化和 L0C→UB、UB→L1 的片上通路。当前实现支持
GQA，并使用 Cube 相对 Vector 的 prelaunch 流水。[^kernel][^guide]

# 用法

## 分核策略与基本块切分

任务空间是 `batch × query_head × query_block`，通过
`tla.range(block_idx, total_tasks, block_num)` 做 grid-stride。每个任务固定一个
Q/head，遍历全部 KV block；GQA 用 `kv_head_idx = head_idx // group_size`。Q/KV
基本块均为 128，两个 AIV 用 `sub_block_idx()` 沿 Q 维各处理最多 64 行。[^kernel][^tiling]

kernel ABI 包含 Q/K/V/O、mask、tiling data 和 Q/KV actual sequence length：

```python
@tla.kernel
def flash_attention_infer_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_o: tla.Tensor,
    mem_mask: tla.Tensor,
    tiling_data: tla.Tensor,
    actual_q_seqlen: tla.Tensor,
    actual_kv_seqlen: tla.Tensor,
):
    ...
```

Host 将 BSND Q/K reshape 为 `[-1, HEAD_DIM]`；K 使用 ColumnMajor 适配转置，
Q/V/O 使用 RowMajor。tiling data 由独立模块打包，actual sequence length 使用以 0
开头的 batch 前缀和。[^tiling][^guide]

# 代码模式

## 数据路径与存储层级

```text
Q/K GM -> L1 -> L0A/L0B -> QK MMAD -> L0C(S fp32)
      -> FIX SPLIT_M -> UB S -> Vector Online Softmax -> UB P(fp16,zNUnAlign)
      -> L1 P + L1 V -> L0A/L0B -> PV MMAD -> L0C(Otmp fp32)
      -> FIX SPLIT_M -> UB Otmp -> Vector rescale/normalize -> GM O
```

Q 的 L1 buffer 为单份，K/V 为双 buffer，P 为三 buffer；QK/PV 的 L0C 和 UB
中间结果分别使用 ping/pong。这样 S、P、PV 均不需要落回 GM。[^kernel][^guide]

Online Softmax 每个 KV block 更新状态：

```text
m_new = max(m, rowmax(S_ij))
P_ij  = exp(S_ij - m_new)
O     = O * exp(m - m_new) + P_ij @ V_j
l     = l * exp(m - m_new) + rowsum(P_ij)
O_out = O / l
```

当前固定参数以 `HEAD_DIM=128`、`Q_BLOCK=128`、`KV_BLOCK=128` 为核心，输入输出
为 FP16，中间分数与累加使用 FP32。生成实现中，固定 block、dtype 和 prelaunch 值应
在 kernel 函数内定义；batch、head、sequence 等实际尺寸通过 tensor metadata、tiling
data 或显式 kernel 参数传入。所有编译期特化复用同一个 kernel，并由形式参数类型或
metadata 生成不同编译产物。[^kernel]

## 流水排布、同步关系与数值精度

`PRE_LAUNCH=2` 让 QK/softmax 比 PV 提前两个 KV block，循环上界额外增加两个迭代
完成流水排空。三组 mode-4 cross flag 分别交接 QK→softmax、softmax/P→PV、
PV→rescale，并为 AIV0/AIV1 单独传 `aiv_id`；同核 L1/L0/CUBE/FIX/MTE3 由普通
flag 管理 buffer 所有权。[^kernel]

Q/K/V/P 为 f16；QK score、row max/sum、PV accumulator 和 rescale 状态为 f32；
P 在写 L1 前由 f32 cast 为 f16，最终 O 在归一化后 cast 为 f16。[^kernel][^guide]

# 无源码重写骨架（蓝图）

> 本节细化到"没有样例源码也能重写整个 kernel"。坐标系：AIC=cube region、
> AIV=vector region（2 个，SPLIT_M 各一半 M 行）。

## kernel 形态铁律

- 全家族**只允许一个 `@tla.kernel`**；禁止按 shape/dtype/layout 变体声明多个
  kernel。固定常量（`HEAD_DIM/Q_BLOCK/KV_BLOCK/PRE_LAUNCH/L0_STAGES/buffer 数`）
  在函数体内定义；kernel 不得读模块级变量或 closure（审查会查自由名字，
  只允许 `tla` 与 Python 内建符号）。
- 运行时 shape 三通道（样例只演示前两个，第三个是突破"仅编译期 shape"的关键）：
  ①tensor metadata：`from_dlpack(t, layout_tag=..., origin_shape=(...))` 绑定后
  kernel 内可读形状；②**显式标量 ABI**：`batch/q_heads/kv_heads/q_seq/kv_seq:
  tla.Int32`、`scale: tla.Float32`，host 传 `tla.Int32(v)`；③**kernel 级 GM 标量读**：
  `mem_lens[cur_batch]` 直接读单元素（kv_lens/cu_seqlens/sink 用）。
- Host 绑定惯例：连续 buffer 以 `(numel//w, w)` 2D RowMajor + `origin_shape`
  绑定（zero-copy）；w=张量行宽（f16 通常 128；rope 类 64）。

## 常量与缓冲清单（fp16 基座，Q_BLOCK=KV_BLOCK=HEAD_DIM=128）

```text
L1(512KB):  l1_q 32KB×1 | l1_k 32KB×2 | l1_v 32KB×2 | l1_p 32KB×3   ≈208KB
L0A(64KB):  ping/pong 32KB×2（Q 与 P 分时复用）
L0B(64KB):  ping/pong 32KB×2（K 与 V 分时复用）
L0C(256KB): l0c_s 64KB×2 | l0c_pv 64KB×2
UB(每 AIV): ub_s 32KB×2(f32,行节距128) | ub_p_f16 (65,128)≈17KB×2(zNUnAlign)
            | ub_pv 32KB×2(f32) | ub_acc 32KB(f32) | max/sum/tmp 若干
stage 计数: mm1L0A/L0B = mm2L0A/L0B = 1（tile 全 128 时）；
            L0A/L0B buffer id = prefixSum(stages) % 2 轮转
```

## flag 协议全表（event id 每引擎对 ≤7，必须配平）

```text
cross_flag（跨核 AIC<->AIV，mode-4）:
  qk_ready_0/1     FIX→VECTOR   S fixpipe 完成 → 可 softmax（按 ubSBufId）
  sm_ready_mm2_0/1/2  MTE3↔MTE1 P 的 L1 buffer 所有权（AIV 写↔AIC 读，按 l1PBufId）
  pv_ready_0/1     FIX↔VECTOR   PV fixpipe 完成 → 可 rescale（按 ubOTmpBufId）
普通 flag（同核引擎对）:
  q_l0a_ready_l1(MTE1→MTE2)/q_l1_ready_l0(MTE2→MTE1)      Q 的 L1 生命周期
  k_l0b_ready_l1_0/1、k_l1_ready_l0_0/1                    K 的 L1 生命周期（MTE1↔MTE2）
  v_l0b_ready_l1_0/1、v_l1_ready_l0_0/1                    V 的 L1 生命周期
  mmad_ready_l0a/l0b_0/1(CUBE→MTE1)、l0a/l0b_ready_mmad_0/1(MTE1→CUBE)  L0 所有权
  fix_ready_mmad_qk/pv_0/1(FIX→CUBE)、mmad_ready_fix_qk/pv_0/1(CUBE→FIX)  L0C 所有权
  v_mte3_0/1(VECTOR→MTE3)、mte3_ready_softmax_0/1(MTE3→VECTOR)  P 的 UB→L1 交接
  mte3_ready_rescale(MTE3→VECTOR)、rescale_v_mte3(VECTOR→MTE3)  O 回写交接
配平规则：序言(cube 区+vector 区)对每个"buffer 空闲"类 flag 先 set 一次；
每轮 set/wait 严格成对；尾声(cube 区)wait 掉所有悬空 set。配平审计法：
数每个 flag 的 set/wait 计数——不等即死锁；缺"反向"flag（缓冲复用无回执）
即竞态（数据被下一轮覆盖）。
```

## 主流水时序（每任务；kv 循环 `for kv_iter in range(0, KV_BLOCK_COUNT+PRE_LAUNCH)`）

```text
任务开始(cube)：Q GM→L1（全任务一次）
kv_iter < KV_BLOCK_COUNT（QK 相位, cube）:
  K GM→L1 → L1→L0B；Q L1→L0A（每块重载）；mmad(l0c_s, l0_q, l0_k, init_c=True)
  → fixpipe L0C→UB S（SPLIT_M，每 AIV 半 M 行）→ cross set qk_ready
vector（softmax 相位）: cross wait qk_ready → 逐行 S×scale(+掩码) →
  init(kv_iter==0: m=rowmax,l=rowsum) 或 update(m_new=max, l=l·e^Δm+Σe, acc·e^Δm)
  → P f32→f16 经 zNUnAlign 瓦片 store → MTE3 copy UB P→L1 P → set sm_ready
kv_pv = kv_iter - PRE_LAUNCH ≥ 0（PV 相位, cube）:
  V GM→L1 → L1→L0B；P L1→L0A；mmad(l0c_pv, l0_p, l0_v, init_c=True)
  → fixpipe → cross set pv_ready
vector（rescale 相位）: acc = acc·e^(m_old−m_new) + pv_new（首块直接赋值）
  → 末块 O = acc/l → cast f16 → GM（按 q_actual 钳制行数）
```

## 数值与 cast 口径

- f32→f16 窄化 cast 结果落在打包位置：`CastParams(reg_slot=ZERO/ONE)` = 偶/奇位；
  P 写出用 even/odd 两寄存器各 cast 后 `tla.bitwise_or` 合并为连续 f16 行
  （S 经 `LoadDist.DIST_DINTLV_B32` 奇偶解交错 load，store 前再交错——
  **奇偶对调是全行级精度错误的典型根因**，tnd 实测踩坑）。
- `tla.deinterleave(x, x)` 是窄行（kv_tile≤64）P store 的等价写法。
- `tla.full(v, dtype)` 只接受字面量；运行时标量转向量须 UB 单元素 store +
  `LoadDist.DIST_BRC_B32` 广播 load。
- **初始状态设计消除分支**：在线 softmax 初值取 m=MINV、l=0，首块
  dm=exp(MINV−m′)=0 使旧状态贡献精确为零——首块无需分支。同一方法的实例：
  sink 锚点并入分母（m₀=sink_h、l₀=1，首块 dm 自动并入）；split-KV 空分段哨兵
  （写 m=MINV、l=0，归并端 w=exp(m−m\*)=0 自然失效）；全掩码行湮灭
  （dm=0 精确清零 acc 与 l，无需特判）。
- **掩码置位用有限大负数（−3e38）而非 −inf**：exp 后恒为 0，softmax 分母与
  split-KV 归并链路全程无 NaN 风险。
- **退化用例位级等价校验**：每个派生场景设计一个位级退化对拍基座——
  sink=−inf ≡ 标准 FA、恒等块表 ≡ 连续存放、单句 TND ≡ causal、
  共享前缀为空 ≡ 无私有段。位级一致是最强的派生正确性证据。

## O 回写与尾块

- 管道恒按整块 Q_BLOCK 处理，GM 回写按 `q_actual = Q_SEQ − q_block_idx·Q_BLOCK`
  钳制（上游样例 q 尾块 ≤112 时双子核 SPLIT_M 分割错位，必须整行写）。
- **偏移高频坑**：若 `o_b_offset` 已把 (b,h[,n]) 全展开进 ptr 偏移，回写视图
  不得再累加 `head_idx·q_plane`——残留会导致跨头覆写竞态（FFA、MLA 各踩一次）。

# 变体派生图谱（FA 家族 8 算子，全部由本骨架派生并交付验证）

| 算子 | 最小改造点 | 关键原语/坑 |
| --- | --- | --- |
| fa_bnsd | 基座本体 | 尾块整行写修复 |
| fa_bnsd_causal | softmax 逐行掩码（`causal_base=row_base+causal_diff−kv_iter·KV_BLOCK+1`，右对齐） | **不跳块**：cross-flag 一致性下逐任务 kv 块数必须一致，掩码-only |
| fa_bnsd_varlen | kernel 级 GM 标量读 `mem_kv_lens[cur_batch]`，行无关前缀掩码 | GM 标量读取代 host 下发 |
| fa_bnsd_sink | kv_iter==0 init 相位播种 `m'=max(m,sink_h)`、`l+=e^{sink−m'}` | sink 经 UB store+DIST_BRC_B32 广播 |
| fa_page_attention | kv 行地址经 `mem_block_table[...]` 查表 + context_lens 前缀 + Sq>1 右对齐因果 | 地址逐 kv 块查表，余同 varlen |
| fa_tnd | [T,N,D] 长流：行 stride N·D；cu 预载 UB；句内因果+句界双侧掩码 | DINTLV 奇偶列序对调坑 |
| ffa（双 KV） | QK/PV 各两次 mmad 累加同一 L0C（init_c=True/False，零搬运加法）；L0B 阶段数×2 | event id≤7：K2/V2 与 K1/V1 **共享** L1 协议 flag（MTE2 双等双载、MTE1 双拷双放）；MTE1 队列位置错→死锁 |
| fa_mla（Dc=512+Dr=64） | Q_BLOCK=32/KV_BLOCK=16 容量选型；L1 拼接 [c_kv\|k_rope]（分形对齐偏移，QK 单 mmad）；V=c_kv 直通；L0A/L0B 专用槽位 | ub_s 行节距必须 128（(1,64) 瓦片跨行覆写，"仅首行正确"）；P UB 宽度恒 128（128 宽合并寄存器 store 防 spill） |

# 关键原语速查（样例未覆盖或文档不全）

- `DIST_US_B8`：b8 上采样 load，每 i8 落入独立 b16 lane——i8→f16 反量化唯一
  正确写法；普通 load 后 cast 会按偶数字节读源（输出每隔一个取到错值）。
- UB zN 存储：`make_tensor_like(ptr, ref, tla.arch.zNUnAlign)` + tile_view 逻辑坐标
  store + `BlockStoreParams(block_stride=缓冲行数)`；f16 落点公式
  `(r,c)→(c//16)·(rows·16)+r·16+(c%16)`（zn_store_probe 实测）。UB(zNUnAlign)→
  L1(zN) 的 `tla.copy` 是几何拷贝，自动翻译不同行数分形几何。
- L0C 累加：同形状两次 `mmad`（第一次 `init_c=True`，其后 `False`）= 免费加法
  （FFA 的 S1+S2、PV1+PV2）。
- i8 mmad 通路：i8 GM→L1(zN)→L0A/L0B→`mmad`→L0C **i32**→fixpipe SPLIT_M 搬 i32
  →vec `i32→f32`（同宽 cast 连续）乘标量。K 转置须 GM 侧 ColumnMajor 视图
  （GM ColumnMajor→UB 非法、UB 跨步 store 转置会 device error）。
- 动态 shape：`origin_shape` + 显式标量 ABI 即可全运行时 shape（打破样例
  "仅编译期 shape"约束）；`tla.compile` 按形参 metadata 的 cache key 复用产物。
- L1 部分瓦片：`tla.copy(l1_tile_view, gm_tile)` 合法（分形对齐偏移），
  可把两段 GM 拼进同一 L1 缓冲（MLA 的 [c_kv|k_rope]）。
- L0A/L0B 槽位可硬编码（Q/P、K/V 专用槽）——`Q_BLOCK//L0_TILE` 整除为 0 时
  原 prefixSum 轮转公式失效，直接写死 buffer id 并保持 flag 按槽配平。

# 硬约束与排障（实测，文档未写）

- **event id ≤7/引擎对**：新增独立 flag 协议前必数现有计数（编译报
  `event id exhausted`）；超限方案=共享协议（见 ffa）。
- **UB 行节距=128**：vec 恒按 `(1, VL_FLOAT_ELE=64)` 瓦片处理 S/P 行；
  窄块（kv_tile<64）若按实际宽度缩行节距，单瓦片 store 跨行覆写邻行
  （症状：仅局部首行正确，其余全错）。分配/视图行节距恒 128。
- **P UB 宽度=128**：P store 恒为 128 宽合并寄存器；缓冲/视图按
  `(Q_BLOCK_SUB+1, 128)` 分配，L1 拷贝只取前 kv_tile 列——否则分形列 spill
  至相邻 UB 缓冲（症状：多块时 P/累加状态被逐步污染）。
- region 变量不逃逸：`with tla.cube()/tla.vector()` 内定义的张量/变量在另一
  region 不可见（编译 NameError）；跨 region 共享的须在 kernel 顶层定义。
- 动态控制流：动态 for 的归纳变量名不能跨循环复用；动态 if 之后使用的变量
  必须在 if 前初始化（FrontendControlFlowLoweringError 直给）。
- `tla.copy` GM(ColumnMajor)→UB 非法；UB 内跨步 store 转置 device error；
  转置只在 GM 侧视图（ColumnMajor+nZ）或 L0 通路完成。
- `from_dlpack` 必须显式 `layout_tag`；host 覆盖 `PYTHONPATH=`（非追加）会丢
  CANN 注入的 acl 模块。

# 调试方法论（8 算子实战沉淀）

1. **微探针法**：对任何不确定原语，写 ~50 行最小 kernel（16×128 单核单块）
   单独验证（v_deq_probe 验证 US_B8 反量化、zn_store_probe 验证 zN 落点、
   i8_s_path_probe 验证 i8 mmad→fixpipe→vec 全链路、mask_probe 验证
   运行时 update_mask）。一次探针 <2 分钟，远快于整 kernel 排障。
2. **分路隔离诊断**：P 提取（V=恒等 → O=P@V 直通）；逐 block 置零隔离；
   逐行/逐列误差分布（"仅首行正确"→行节距；"16 列整数倍缺块"→缓冲复用
   竞态；"头 0 对其余错"→任务解码/头偏移；"奇偶列互换"→DINTLV 对调）。
3. **死锁 vs 竞态判别**：死锁（aicore timeout）→flag 计数不配平或引擎队列
   成环（用配平审计法画 set/wait 链）；竞态（数值错但不挂）→缺反向 flag，
   缓冲被下一轮覆盖。
4. **对照基线**：改动 kernel 时保留可运行的上一版，用同一组形状探针
   （单块/多块/多头/尾块）二分定位引入点。

# 约束

- 样例本身只支持编译期常量 shape；用"动态 shape 三通道"（见骨架节）可突破，
  本家族 8 算子均为全运行时 shape 单 kernel。
- mask 参数目前仅占位并要求全 0，不支持通用 0/1 mask。
- 暂不支持 PagedAttention 或分页 KV cache（fa_page_attention 变体另见派生图谱）。
- 当前 dtype 为 FP16 输入输出、FP32 中间计算，`HEAD_DIM=128`（MLA 变体为
  512+64 维拼接）。int8 W8A8 变体曾实现（S 域反量化 + AIV V 反量化相位，
  `DIST_US_B8` load + i8→f16 单 cast + f16 mul 为验证过的正确写法），
  无同语义参考接口，场景不纳入跟踪。
- `KV_HEAD_NUM <= HEAD_NUM` 且 GQA group 映射必须整除并与 reference 一致。
- kernel 不得从模块级变量或 closure 读取 block、shape、dtype、prelaunch 等配置，也不得
  由 host 在编译前改写这些隐藏状态。
- 不得为 shape、dtype、layout、block 或 prelaunch 变体声明独立 kernel；同一 solution
  只保留一个 `@tla.kernel`。

# 失败表现

- O 保持 sentinel：block/tiling 映射、最终回写或跨流水同步未执行。
- 某些 query 行整行错误：actual sequence 前缀和或 Q block 任务映射错误。
- 仅首行正确、其余行全错：UB 行节距未按 128（(1,64) 瓦片跨行覆写）。
- 输出按 16 列整数倍缺块：UB/L1 缓冲复用缺反向 flag（竞态覆盖）。
- aicore timeout：flag set/wait 不配平或引擎队列成环（新增 flag 最先怀疑）。
- `event id exhausted`：同引擎对 flag 超 7 个，须共享协议。
- 头 0 正确、其余头错：任务解码或头偏移（O 回写累加了重复 head 偏移）。
- 奇偶列成对互换：DINTLV load 的 even/odd 寄存器与 store cast trait 对调。
- 数值溢出/NaN：Online Softmax 的 max rescale、sum 或 FP32 状态维护错误。
- GQA head 串数据：`kv_head_idx = head_idx // group_size` 或 K/V offset 错误。
- 修改 shape 后复用旧 artifact：编译期常量与实际 buffer 不一致。
- 不同 case 的 TLAIR 随执行顺序变化：kernel 的编译期配置来自可变全局状态。

# 验证方法

先检查 TLAIR 中 QK/PV MMAD、Vector softmax 和跨流水同步，再执行 host reference；
要求 O 不再是 sentinel 且误差低于样例阈值。case 至少覆盖 batch>1、GQA、Q/KV tail
和多个 KV block；源码审查要求 decorated kernel 除 `tla` 和 Python 内建符号外没有自由
名字。性能需要在空闲 NPU 上分别 benchmark/profile，源码流水结构本身不构成性能结论。
[^kernel][^guide]

[^kernel]: 固定提交 Flash Attention kernel、host reference、sentinel 与误差校验入口。
[^tiling]: 固定提交 tiling data 和 actual sequence metadata 的打包实现。
[^guide]: 固定提交对算法、支持特性、限制和运行方法的说明。
