---
name: cannbotdsl-crash-debug
description: "CANNBotDSL kernel 在 NPU 上 crash 或 hang 时做从 Python 层到设备层的全栈定位时使用。当 kernel 出现 SegFault/AIC exception、torch.npu.synchronize() 超时卡死、或间歇性崩溃时触发。Hang 诊断：区分编译慢 vs kernel hang、Channel 生产/消费配对审计、同步原语计数 diff。Crash 诊断：委托 npu-plog-diagnosis 读设备 plog、参数区解码（ABI 错位/坏指针）、Buffer/Channel 地址越界或 Channel 预算超限。间歇性：Channel depth-N 时序、未初始化 storage 读取。Triggers: cannbotdsl crash, hang, SegFault, AIC exception, synchronize 超时, sync 死锁, 参数区, buffer 越界, 间歇性崩溃。Developer 在 Stage 3 调用。"
---

# cannbotdsl-crash-debug

CANNBotDSL kernel 在 NPU 上 crash / hang 的全栈定位，从 Python 层到设备层。Developer 在 Stage 3 调用。前置：本 skill 处理的是**已编译成功、上真机才炸**的问题；编译期错误走 `../cannbotdsl-runtime-debug/SKILL.md`。**需要 NPU** —— 无 NPU 时只能做 §sync 静态审计。

## 触发条件

- crash：SegFault / AIC exception
- hang：`torch.npu.synchronize()` 超时卡死
- 间歇性崩溃（时对时错）

## 第一步：分流 crash vs hang vs 编译慢

`run()` 返回但 `synchronize()` 卡住 → 通常是 kernel 内 sync/调度未结束，**不是编译慢**。先确认：AOT `compile()` 单独跑一遍，若秒回则编译不慢，问题在 execute 阶段。

不要盲等 —— 先做阶段打印（`print_tensor`/`print_scalar`）+ 审计源码里的 sync 结构。

## Hang 诊断：sync 死锁审计（最常见）

hang 的头号原因是 sync 不配对形成 ticket-lock 死锁。审计手段（**无 NPU 也能做**）：

1. 在 kernel 源码里 **diff 同步原语的计数**：`sync_intra_arrive`/`sync_intra_wait` 配对、notify/wait 的 `event_id`：
   - `sync_intra_arrive` 无对应 `sync_intra_wait` → 等待方永久阻塞。
   - `sync_notify(src,dst,eid)` 无对应 `sync_wait` → 等待方永久阻塞。
2. **跨核 arrive/wait 配对计数**：Cube 侧 arrive 数须等于 Vec 侧 wait 数（反之亦然）。per-subblock token 用 `id + subblock*16` 对称。
3. **Channel 生产/消费平衡**：重点核对每条 Channel 的 Write/Read 操作数是否成对出现在数据流上；跨核 channel `Σdepth≤8`。
4. 对比法：和一个已知能跑通的等价版本逐条 diff 同步原语序列，差出来的那条就是死锁点。
5. **负载不均衡致单核 timeout**：sync 计数配平但某核分到的 tile 代价远超平均（causal 下 m-block 轴放 `idx2crd` 最内层且 extent 整除 GRID → 每核工作量恒定不均）。先用 host 侧算术算每核负载（`max(load)/mean(load) > 1.2` → 分发问题，不是 sync 问题），见 `../../core-skills/cannbotdsl-perf-optimize/SKILL.md` 第 0 步。

execution 边界的 allocator rewind **不等于**同步 handoff —— ready token 被 free/init token 顶替是经典 hang 源。

## Crash 诊断：SegFault / AIC exception

1. **委托 `cannbotdsl-npu-plog-diagnosis`** 读设备 plog，拿 AIC/AIV exception 地址与类型。
2. **参数区解码**：ABI 错位 / 坏指针 —— `KernelArgTensor` 结构体 dtype/ndim 与 kernel 签名不符（见 `cannbotdsl-runtime-debug` marshalling 段）。
3. **Buffer 越界**：
   - L0/UB 容量超限：手动别名 + double buffer 时单级已满还 `advance` 第二级。
   - 先算真实字节区间对照硬限制（`../../core-skills/cannbotdsl-op-design/SKILL.md §2`）。

## 间歇性崩溃（时对时错）

- **Channel depth-N 时序错误**：软件流水 lag/prologue/drain 与生产消费顺序不一致，读到还没写完的缓冲区。
- **未初始化 buffer 读取**：表现为随机值或全零；GM/UB 未初始化就读。
- **`const_expr(cond)` 守卫变负失效**：形如 `if const_expr(NPAD > 0):` 的保护，当 `NPAD = VH - BMV` 变负时静默跳过 → 越界写读但无 fault。查所有 `const_expr` 守卫的变量是否可能为负，见 `../../core-skills/cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。
- 复现策略：固定 seed，多次运行看是否稳定错；临时把 `Channel(..., depth=N)` 改为 `depth=1` 排除流水时序问题。

## 定位后路由

| 根因 | 修复归属 |
|------|----------|
| sync 漏配对 / 不对称 | 本层修 kernel sync；对照 `cannbotdsl-code-review` §1 |
| 容量超限 | 回 Stage 2 设计（`cannbotdsl-op-design`），C 型 DESIGN_ERROR |
| ABI / marshalling | `cannbotdsl-runtime-debug` marshalling 段 |
| 疑似框架 lowering bug | 拆最小 case → `cannbotdsl-framework-probe` |

## 门禁

- 报告必须区分 crash / hang / 间歇，并给出**具体 sync 对或 buffer** 的证据（`.asc` 行 / plog 地址），不空泛说"同步有问题"。
- hang 优先用 `.asc` 静态审计（无 NPU 可做），不要只靠反复上真机试。
- 越界/容量类回设计层修，不在实现层加保护绕过。

## 参考

- `../cannbotdsl-npu-plog-diagnosis/SKILL.md`（设备 plog 读取）
- `../cannbotdsl-precision-debug/references/mixed-kernel-debug-lessons.md`（sync/handoff 教训）
- `../cannbotdsl-runtime-debug/SKILL.md`（marshalling、阶段截停）、`../cannbotdsl-code-review/SKILL.md`（Channel 协议审查维度）
