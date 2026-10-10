---
type: CATLASS DSL Optimization Guide
title: CANN Performance：Attention 多级流水与负载均衡
description: 从 Flash Attention Lite 与全量化 FIA 递进样例提取片内交换、CV/L0/I-O 多级流水、代价感知分核和 DN 归约候选；另含任务映射 swizzle、掩码窗口化、decode split-KV、HEAD_DIM 参数化、sink 状态播种与单 KV 块双 Q 块融合方案。
tags: [catlass-dsl, optimization, attention, flash-attention, pipeline, mixed, load-balance]
status: draft
generated: {by: process:cann-samples-performance-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: falite
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/flash_attn_lite_story/README.md
    title: Flash Attention Lite v0-v5 optimization story
  - id: fia
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/2_Performance/full_quant_fused_infer_attention_score_story/README.md
    title: Full-quant FIA performance guide
  - id: faexp
    resource: project-evidence:fa_dsl/ops_v2/REPORT.md?kernel-sha256=90d164c9c06c0b4ce93fe992894492d0ff1613b7dec579516fe310bb79384e50
    title: FA 8 类场景开发交付（固定提交 83219db）
operator_families: [flash-attention, attention, mixed]
arch: [c310]
---

# 接口与概念

两个固定专题共同给出一条由外到内的 Attention 优化轴：按实际序列长度估算 tile 代价并均衡分核；
将 C1→V1→C2→V2 从同 tile 串行改为跨 tile 错位；把 S、ΔO 从 L0C 经 Fixpipe 直送 UB；
把 P 从 UB 送 L1；再分别对 L0、K/V L1 和 task 级 Q/O 做双槽所有权。[^falite][^fia]

Flash Attention Lite 的 单槽用 `DONE` 保护整套资源，v3 改为按 slot 的 S/P/O ready 事件，
v4 为 L0A/L0B/L0C 增加双槽并仅保留真实正反依赖，v5 把 V 预取前移至 C1，并让 task t 的
输出 MTE3 与 task t+1 的输入/计算错位。[^falite]

FIA 还提供两个正交候选：对变长 KV tile 使用动态目标负载的连续贪心分配；把 MM1 方向改为
`K × Q^T`，让 Vector 在转置后的 DN 组织上用并行 `vmax/vadd` 代替逐行归约。[^fia]

# 用法

按瓶颈一次只测试一层：核间长尾先测代价感知分核；GM 中间流量高先测 L0C→UB/UB→L1；
AIC/AIV 交替空闲先测 CV 双槽；MTE1/MMAD/Fix 互等再测 L0 双缓冲；task 边界有空洞再测 I/O
双槽；Vector 归约单发占主导时再测 DN 方向。

按场景形态再选一层：长序列 GM 流量主导先测任务映射 swizzle；带掩码形态（因果/句界/变长）先测掩码窗口化；decode（Sq≤16、
分页 KV / MLA）先测 split-KV 双 kernel；多 D 档位先测 HEAD_DIM Constexpr；sink 锚点先测
状态播种；Skv 单块形态先测双 Q 块融合。

# 代码模式

```python
# CV slot 协议：不同 tile 可重叠，同一 slot 必须闭环。
slot = tile_id & 1
cube_c1(tile_id, s_ub[slot]); set(s_ready[slot])
wait(s_ready[slot]); vector_v1(tile_id, s_ub[slot], p_l1[slot]); set(p_ready[slot])
wait(p_ready[slot]); cube_c2(tile_id, p_l1[slot], do_ub[slot]); set(o_ready[slot])
wait(o_ready[slot]); vector_v2(tile_id, do_ub[slot], state)

# 代价感知连续分核。
remaining_cost = sum(tile_cost)
for core in range(core_count):
    target = remaining_cost / (core_count - core)
    assign_next_contiguous_tiles_until_half_tile_rule(target)
```

外层 I/O 双槽必须由最后读取 Q/OAcc 的阶段释放；K/V L1 slot 只有在 C2 读完 V 后才允许下一代
C1 覆盖。DN 候选还必须成套修改 MM1 方向、Vector 坐标和 MM2 输入转置。[^falite][^fia]

## 任务映射 swizzle（头主序）

任务解码由块主序 `task = q_block·H + head` 改为头主序 `task = head·QB + q_block`：
同一 wave 的核集中处理同一头的全部 q 块，K/V 平面在 L2 中被整 wave 复用
（单 wave K/V 足迹从多头降到一头）。适用：任务数远超核数且 KV 流量主导（长序列）；
GQA 同组头共享 K/V 同样受益；Skv 极短形态中性。纯任务顺序变化，数值逐位不变。[^faexp]

因果形态的任务成本随 q 块号单调增长（窗口内块数递增），头主序下同一 wave 内
负载不均；**per-head 蛇形块序（boustrophedon，奇数头反转 q 块序）**把长短任务
交错到同一 wave，负载摊平且不损 L2 复用（同 wave 仍同头）。[^faexp]

## 掩码窗口化（causal/TND/varlen/尾块）

任务内把 KV 循环界收敛到 `[win_lo, win_hi)`（AIC/AIV 同源标量计算同一界值），
满块段零掩码算子，掩码只出现在边界块的独立循环体（每任务 ≤2 块）：

```python
win_hi = min((qb*128 + offset) // 128 + 1, KBC)     # 因果右对齐满块数
for kv in tla.range(c0, win_hi + PRE, c1):           # 动态界；窗口外不载不算
    ...
# 边界块（窗口末一至两块）逐行前缀掩码：
w  = t - 64*g + 1                        # 组 g（列基 64g）可见宽度
wc = w & (-1 ^ (w >> 31))                # 负侧钳 0（位运算）
m  = tla.update_mask(wc)[0]              # wc>64 硬件饱和全通；wc=0 全灭
s  = tla.where(m, s, MINV)               # softmax 遍1 施加
```

TND 句界加每行区间表（任务开头由 cu_seqlens 标量扫描生成 `ub_row_lo[i]`，UB 索引
标量宽度），低切 `where(m_lo, MINV, s)` + 高切 `where(m_hi, s, MINV)`。

## Decode split-KV 双 kernel（page / MLA，Sq≤16）

Sq=1 的 GEMV 形态去掉 AIC/AIV 跨核握手，两个纯向量 kernel 顺序 launch：

```python
# K1：任务 = B×Hq×SPLITS，for t in tla.range(sub_block_idx(), TASKS, block_num=56)
#   每 split 扫 [s*G,(s+1)*G) 个 128-token 块，UB 双缓冲 K/V；
#   S_j = reduce(K_row_j·q_bcast)；在线 softmax（m,l,O_acc f32）
#   收尾写 GM：O_part f32 + m/l f32（空 split 写 m=-inf，归并端自然失效）
# K2：任务 = B×Hq：m*=max_s m_s；w=exp(m_s-m*)；O = ΣO_s·w / Σl_s·w
SPLITS = clamp(ceil(KBC/8), 1, max(1, 112 // (B*Hq)))   # host 计算
```

分页 KV 的块表寻址：逻辑块号经 GM i32 标量读换物理块号；恒等表下与连续存放位级一致。
MLA 叠加掩码窗口化处理 Sq=16 的右对齐因果（每任务非满块 ≤2 个）。

## HEAD_DIM Constexpr 参数化（D∈{32,64,128}）

`HEAD_DIM: tla.Constexpr[int]`，compile 缓存按 (kernel, D) 分键；L0_TILE_K=D；
L0C 累加器描述子显式 `(QROWS, KCOLS)`（不继承 K 视图形状）；AIV 向量侧统一 VPAD=128
等宽（acc/归一/cast 与 D=128 字节同构），O 回写裁 HEAD_DIM；UB/L0C 行 pitch 固定 128。

## sink 锚点状态播种

锚点 logit 等价在线 softmax 初值 `m0 = sink_h, l0 = 1`：kernel 开头 sink[Hq] f32
GM→UB 一次；任务开头 64-lane 播种（BRC 载入 + O(1) 组向量算术，不逐行循环）；
首块 dm = exp(sink−m₁) 自动并入分母；sink=−inf 位级退化标准 FA。

## 单 KV 块双 Q 块融合（KBC==1 形态）

`KBC==1 且 QBC%2==0` 时 host 静态选融合 kernel：任务 = B·Hq·(QBC/2)，每任务 2 相邻
Q 块共享 K/V 的 GM→L1 装载，成员独立 m/l 状态，FUSE_PRE=1 成员流水；原 kernel
逐字节保留（host 按 shape 选路）。[^faexp]

# 约束

- 适用：`c310` mixed Attention；shape/tile、BF16/FP8/HIF8 路径及实际序列分布须单独记录。
  正文中的核数、缓冲字节数、SPLITS 公式常量为当前实现的实测参数，适用平台规格以 arch 字段与 npu-arch 为准，迁移时按目标规格重算。
- 保持：online-softmax 的 `m/l/O` 更新顺序、mask、尾 KV、量化 scale、累加/写回 dtype 不变；
  AIC/AIV 对 slot、SubBlock 和 tile generation 使用同一映射。
- 代价：每层双槽增加 L1/L0/UB 与事件；DN 可能增加转置/布局成本；负载均衡可能拆分 batch，
  引入 Flash-Decoding workspace、归约和同步。
- 可证伪预期：GM 字节、跨核等待、阶段空洞或最长核时长下降，并转化为高于噪声的总延迟改善。
- 不把文档中的单 shape、单次 Task Duration 或理论比例当作 CATLASS 性能结论。
- 蛇形块序仅调整同头内 q 块顺序；跨头反转会破坏 L2 前缀共享。
- 窗口动态循环界须 AIC/AIV 两侧同源同值；结构性跳块（flag 环绕的动态 if）仍死锁，
  跳块只能靠循环界收敛。
- 边界掩码与干净段须写成独立循环体（静态分离）；同循环体内动态 if 掩码分支的
  lowering 非单调（部分路径掩码实测 +34~38%）。
- update_mask 宽度负值必须位运算钳 0（无符号回绕为全通）；>64 硬件饱和全通。
- split-KV 多 kernel 的性能口径须逐迭代求和全部 kernel（含归并 kernel），
  与参考实现双侧对称。
- split-KV 在短 KV（≤6k，KV 全量进 L2）形态 cube 参考路径占优；D=64 在 bnsd 系
  档位水位 0.69~0.75（参考实现形态优势区）。
- 融合后向量管成为首管（占 62%）；normalize+cast 合并方向实测退化。

# 失败表现

- tile 1 起错或 hang：slot generation、ready/free 或跨核方向不闭合，回退单槽。
- 长序列尾部错：online-softmax 状态、tail mask 或拆核归并次序错误。
- 容量/编译失败或并行 block 下降：减少一层双槽，保留更外层高收益流水。
- DN 正确但变慢：转置/重排成本超过归约并行收益，恢复 ND。
- 各阶段 active time 改善但总延迟不变：瓶颈迁移或 overlap 指标被误加，回退该单轴候选。
- 窗口界两侧不一致/排水计数错位 → 挂起（无报错）。
- 负宽度不钳位 → 掩码全通，输出数值错（masked 位泄漏）。
- split-KV 漏计归并 kernel → 比值虚高（口径错误）。
- D=64 用真 64 宽向量路径 → 逐行阶梯宽度错乱（VPAD=128 修复）。

# 验证方法

正确性覆盖单/双/奇数 KV tile、短/长和不等长 batch、mask 尾块、量化边界与多 task；用事件 trace
检查每个 slot 的 producer→consumer→release。随后同配置比较 per-core 最长时长、GM 字节、AIC/AIV
等待、MTE1/MTE2/MTE3、Cube/Vector 与总延迟，多次 benchmark 定义噪声阈值并 fresh 复测 best。

掩码窗口化/split-KV 方案：每场景 workload 精度套件全过（rel ≤ 1e-3）+ 边界自查
（尾块/单块/多句/恒等表/GQA）；性能 msprof Task Duration 逐迭代求和，组均几何均值
对照参考实现做全套件终验（覆盖 8 类场景全量 case）。[^faexp]

[^falite]: 固定提交中 v0-v5 的 GM/片内路径、CV 双槽、L0 双缓冲、KV 预取和 task I/O 双槽。
[^fia]: 固定提交中变长 tile 代价分核、CV/Cube 流水、DN 方向和双缓冲策略。

[^faexp]: 项目实测（Ascend950PR，CANN 9.1.0，2026-09-18）：8 类场景全量套件终验通过；
  掩码配方与饱和语义经单核探针验证。
