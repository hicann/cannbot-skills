---
type: "operator"
title: "block_sparse_attention Arch22 任务边界与槽复用同步模式"
description: "FAI 型三槽距离-2 流水在任务切换与槽环回绕处的同步不变量、易漏竞态点与症状-嫌疑映射。"
tags:
- catlass-cpp
- sparse-attention
- block_sparse_attention
- BlockSparseAttention
- 块稀疏注意力
- arch22
status: "stable"
generated: {"by": "process:task-boundary-sync-distillation"}
verified: [{"by": "process:static-code-survey"}]
sources: [{"id": "sparse-attention-task-boundary-sync-patterns-source", "resource": "audit:custody/sparse-attention#task-boundary-sync-patterns", "title": "task-boundary-sync-patterns 固定来源（提取侧独立保管）", "kind": "audit-record"}]
operator_families: ["sparse-attention"]
architectures: ["DAV_2201"]
---
# 接口与概念

本 concept 面向 FAI 型三槽（距离 2）BSA 流水的实现与调试：任务切换处和槽环回绕处哪些同步
不变量必须成立、哪些点最容易漏、出现非确定性精度错误时按什么顺序排查。
证据为**静态调查事实**（取证记录提取侧独立保管）；候选实现是否满足不变量必须经确定性
复现实测判定。本 concept 不含任何实现仓路径、行号或可读取链接。

## 算子算法

**静态调查事实**：执行结构为"任务（batch×head×Q 块内 qSTile）× KV stack（定长窗口）× 三槽环
（槽数 = 流水距离+1）"。每任务：Q 一次装入 L1 驻留，随后逐 stack 执行 QK(C1) → online
softmax+P 量化（V1) → PV(C2) → rescale/输出（V2)；槽按 stack 计数模 3 轮转，启动时先灌 2 拍、
排空时多走 2 拍纯 V2。任务切换**不引入额外屏障**：上一任务尾部 2 个排空迭代把跨核链自然
延伸到下一任务的第一个 stack。非确定性精度错误（同输入两次运行输出不同）在此结构下的
典型触发条件是：**gathered KV 超过 2 个 stack 窗口（槽环开始回绕）且多任务并发**。

## 分核策略与基本块切分

**静态调查事实**：AIC 跑 QK/PV 与 L1/L0 搬运，AIV 双子核跑 softmax/rescale。三条跨核可见性链：
- `S 槽`：C1 写（FIX pipe Set）→ V1 读（Wait）。
- `P 槽`：V1 写（MTE3 pipe Set）→ C2 读（Wait）；该链同时**传递性保护 S 槽**：C2 的程序序
  晚于 V1 对 S 的全部读取，下一圈/下一任务 C1 覆写同槽因此安全。
- `OTmp 槽`：C2 写（FIX pipe Set）→ V2 读（Wait）。
三条链都是**单 id 计数器**，逐 stack 各 Set/Wait 一次、跨任务计数连续；Set 的 pipe 必须保真
（S/OTmp 用 FIX、P 用 MTE3），且一次 Set 须能唤醒两个 AIV 子核的 Wait（广播语义）。

## 数据路径与存储层级

**静态调查事实**：
- L1 分区：Q（驻留、任务内只读、下一任务直接覆写）、K/P/V 缓冲各占固定偏移互不重叠。
- GM workspace：S/P/OTmp 每核 3 槽，槽内元素数上限固定（行数×窗口长度），S(fp32) 与 P(fp16)
  基址不同不别名。
- UB 状态区（softmax 与 rescale 视图同址映射）：gm/gl（全局行 max/sum）、dm（每 stack 一份的
  修正因子，**3 深环**、槽距=每子核行数上限）、go（O 累加，单行循环时驻 UB）、tv（归约/Brcb/
  LSE 搬运共享 scratch）。
- 跨 stack O 累加：多行循环走 GM 单槽 update（写-读各配一次就地等待事件）；单行循环驻 UB go。
- sparseIdx/count：由独立预处理 kernel 一次性写入 workspace，随后**一次全核屏障**（pipe 计数
  复位 + SyncAll）转为只读；预处理的 flag id 与主循环有重叠，计数复位是隔离两套的关键。

## 流水排布、同步关系与数值精度

**关键不变量（静态调查事实，实现必须逐条对齐）**：

(a) C1(s) 写 S 槽 s%3 的时刻，被"C1 程序序晚于 ≥1 次 C2 调用 + P 就绪标志已置位(s-3)"约束在
    V1(s-3) 读完该槽之后；
(b) C2(s) 写 OTmp 槽 s%3 的时刻，被 P 就绪标志(s)（V1 程序序晚于 V2(s-3)）约束在 V2(s-3)
    读完之后；
(c) 任务切换不新增事件，(a)(b) 两条链经上一任务尾部 2 个排空迭代自然延伸到下一任务
    QK(0)/PV(0)；C1 在任务边界可以先跑（Q 装载、K 装载甚至首个 FIX 都已发出），挡住 S 槽
    覆写的只有这条传递链；
(d) **唯一跨任务专用握手**（仅 LSE 输出构建需要）：上一任务末 stack 的 LSE 搬运（MTE3 读
    tv scratch）→ SetFlag(MTE3→V)；下一任务首个 stack 首个行循环的 softmax 在用 tv 前
    WaitFlag。漏掉它则 tv 被并发撕裂：gLse 输出错或 S 归约错，**表现为 LSE 专属性损伤**；
(e) gm/gl/dm 全部靠"同子核程序序 + 首 stack 覆写分支"管理代际，**无任何事件**：首 stack
    分支必须不读旧 gm/gl；P 就绪标志的置位在 gl 写出之前发出是有意的（gl 不随标志
    发布，只供本核）；dm 槽 3 深环由 V pipe 程序序保护。若把这些状态挪到 GM 或改变核间
    分工，必须自造等价事件。

# 用法

- **设计新流水**：先写下 (a)-(e) 不变量再排事件；任何"优化"（提前 Set flag、换 pipe、改槽数、
  拆主循环为三段）逐条对照不变量。
- **调试非确定性精度错误**：按# 失败表现 的症状-嫌疑映射定位，不要先怀疑数学。
- **审查候选实现**：按# 代码模式 的易漏清单逐条核对，重点核对 flag 的 pipe/mode 与计数平衡。

# 代码模式

**七类易漏竞态点（静态调查事实，按嫌疑度排序）**：

1. **跨核 flag 的 pipe/mode 保真度**：P 链必须 MTE3 pipe、S/OTmp 链必须 FIX pipe、一次 Set
   须广播两个 AIV 子核；Set 提前（未到 S 读完成点）或 pipe 用错，S/P 槽在槽环回绕/任务边界
   被撕裂 → gm/gl/dm 垃圾 → LSE 远劣于 O，非确定。
2. **LSE 专用跨任务握手缺失**（不变量 (d)）：漏掉 → LSE 损伤远重于 O；stack 越多 MTE3 积压
   越深，触发窗口越大。【实测已确认，归入模式 R1，见# 已确认竞态模式（实测）】
3. **槽环几何与排空结构**：槽数 ≠ 距离+1（如 2 槽）→ 同拍内 softmax(s) 写 dm 槽与 rescale(s-2)
   读 dm 槽直接别名；stack 计数必须**每任务清零**、启动 2 拍/排空 2 拍的守卫与 isFirst/isLast
   判定式保持精确。
4. **V 侧自管 flag 的跨任务计数平衡**：每个 flag 的 Set/Wait 必须逐分支配对且预置次数正确；
   某分支多/少一次时，单任务内碰巧不亏不发作，跨任务消耗预置存量后开始漂移 → lo/go/dm/
   update 被在途搬运覆写，间歇出错。
5. **C 侧 QK/PV 共享 flag 池与相位**：QK 与 PV 共用同一组 L0/L1 flag id 池与物理 L0A/L0B/L0C，
   靠一次性预置 + 成员相位跨调用跨任务连续翻转维持平衡；每任务重置相位或给两模块独立 id
   但不重算预置 → 首个任务边界 L0/L1 在途覆写。
6. **预处理→主循环的一次性边界**：漏 pipe 计数复位或 SyncAll、或把主循环预置挪到屏障前 →
   预处理残留计数泄进主循环同 id flag，或标量读到未可见的 sparseIdx/count；多核多任务才暴露。
   【同类机制实测已确认，归入模式 R1，见# 已确认竞态模式（实测）】
7. **首 stack 状态重置与 gl 写序**（逻辑级，确定性错误，嫌疑较低）：首 stack 分支读了旧
   gm/gl → LSE 出现 ln(k) 量级偏移（确定性）；LSE 与 gl 同址覆写（先 Div 用 gl、后 Ln 就地
   写 LSE）的次序不可交换。

**Q 驻留 L1 的跨任务 WAR**：上一任务末 stack 的 Q 读（MTE1）与下一任务 Q 装载（MTE2）之间，
参考实现靠尾部 2 个排空迭代的距离兜底、无事件；给 Q 补事件方向正确但不是全部（见嫌疑 1/2）。

# 约束

- 三条跨核 flag 的 id、pipe、广播语义、逐 stack 配对、跨任务计数连续，一项不可改。
- 槽数 = 流水距离 + 1；dm 环深 = 槽数；stack 计数每任务清零；启动/排空守卫与 isFirst/isLast
  判定式保持精确。
- 首 stack 分支不读旧 gm/gl；P 就绪标志的置位不得晚于 P 可见所需点、不得早于 S 读完成点。
- LSE 输出构建必须带不变量 (d) 的跨任务握手；非 LSE 构建可免。
- 预处理与主循环之间必须有 pipe 计数复位 + 全核同步，且主循环 flag 预置在其后。
- 自造 SetFlag/WaitFlag 链时，Set/Wait 必须逐分支配对；任何分支上多出的 Wait 会在该
  路径死锁。协议改动后先在最简用例上验证活性（不挂死），再验证正确性。

# 失败表现

**症状→嫌疑映射**（静态调查事实）：

- 同输入两次运行输出不同（非确定）+ 仅 gathered KV>2 窗口 + 多任务 → 嫌疑 1/2/4/5/6 之一
  （竞态）；确定性错误才考虑嫌疑 7。
- LSE 误差比 O 高一个量级以上 → 优先嫌疑 1（S 槽/gl-gm-dm 链被撕裂）与嫌疑 2（LSE 跨界
  握手缺失）；O 与 LSE 同量级劣化 → 优先嫌疑 3/5（槽几何/L0-L1 覆写）。
- 单任务不复现、多任务并发才复现 → 嫌疑 2/4/5（跨任务计数/相位/握手）；单任务长 KV 也
  复现 → 嫌疑 1/3（槽回绕处链断）。
- 环境争用加剧时复现率上升 → 队列深度敏感的竞态（嫌疑 1/2/4），不能用空载不复现排除。

# 已确认竞态模式（实测）

**模式 R1：暂存区"异步读出 → 再次复用"握手缺失（实测确认）**

任何经异步 pipe（如 MTE3）从暂存区（UB staging 等）向 GM 读出的位置，若该暂存区随后会被
新内容复写，读出与复写之间必须配 SetFlag/WaitFlag 握手链（MTE3→复写方）；否则暂存区被
下一轮内容撕裂，下游读到混合态。多发于：预处理逐行/逐块 staging、LSE 与状态量的
finalize staging、一切 UB→GM→UB 的中转缓冲。

该模式的判定特征：受害位置随运行漂移（非固定任务/行）；输出大面积漂移而非局部偏差；
争用越强命中率越高。

**验证纪律（实测）**：

- 争用敏感竞态的复现必须放大争用（提高并发度/队列深度）并多次连跑；单次失败与空载
  不复现均不构成判定；复现报告须给出命中率与受害点分布。
- 修复验证以"受害结构连跑多次 0 受害 + 逐比特一致"为关闭标准；回归中出现单发异常时，
  须同例多次连跑复测并复跑全套，不得直接放行。
- 任何"已修复"声明必须列明覆盖范围（覆盖哪些竞态/缓冲、不覆盖哪些）；声明撤回时保留
  历史记录可审计，不删改。

# 验证方法

1. **确定性复现构造**：多任务 × gathered KV 超过两个栈窗口（槽环开始回绕），同输入连续
   多次运行要求逐比特一致；修复有效性的最低证据。
2. **不变量逐条审查**：(a)-(e) 写成检查表，每条给出代码证据或探针证据；未满足项即修复点。
3. **探针**：对嫌疑 flag 做"延迟一侧 Set/Wait"的受控实验，确认症状随 flag 时序移动；
   对状态链（gm/gl/dm/tv）做单元探针验证代际规则。
4. **回归**：修复后全量精度回归 + 鲁棒性（repeat 逐比特一致、双 stream、非法输入拒绝）；
   每任务新增的事件开销记入性能观察点，不与精度结论混淆。
5. 空载环境未复现不得宣称修复成立；争用依赖的失败必须以"加重争用的受控实验"或独立
   评测侧复验为准。

[^sparse-attention-task-boundary-sync-patterns-source]: 审计编号 sparse-attention-task-boundary-sync-patterns-source；静态调查取证记录（含真实来源、版本与机制映射）由提取侧独立保管，不随生成侧资料分发。
