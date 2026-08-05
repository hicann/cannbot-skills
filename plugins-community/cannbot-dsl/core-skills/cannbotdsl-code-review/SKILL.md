---
name: cannbotdsl-code-review
description: "审查 CANNBotDSL kernel 代码时使用。CANNBotDSL kernel 有独特审查维度，通用 C++/Python 审查覆盖不到。当需要检查 Channel 生产/消费配对、硬件 sync 预算、Buffer/Channel 地址与容量、VF 区域正确性、AST 预处理兼容性和 Buffer/Channel 选型时触发。含通用代码质量与审查路由（file/PR/quick review）。Triggers: cannbotdsl 代码审查, Channel 配对, sync 配对, Channel depth, VF 区域审查, Buffer 容量, kernel review。Tester sub-agent 在 Stage 4 调用。"
---

# cannbotdsl-code-review

CANNBotDSL kernel 代码审查。特有维度包括 Channel sync 预算、Buffer/Channel 地址容量、VF 区域正确性和 AST 前端兼容性。

## 触发条件

- 需要审查 CANNBotDSL kernel 代码（新写、重构、PR）
- Tester sub-agent 在 Stage 4 调用

## CANNBotDSL 特有审查维度（逐条 checklist）

### 1. Channel 生产/消费配对完整性 ⭐最高优先级

- 每个 `Channel` 都要有明确的生产者与消费者，**同 channel 同缓冲区**，计数匹配。
- Channel 需要在 `@jit`/`@kernel` 上下文内使用；生产方/消费方 ownership 清晰。
- **验证手段**：在源码层逐条核对每个 Channel 的 Write/Read 操作数成对出现（不需要 NPU）。配对不齐 → runtime hang（转 `cannbotdsl-crash-debug`）。

### 2. Buffer/Channel 资源预算

- Buffer 与 Channel 自动地址共享同一 arena；检查显式 `addr=` alias 的生命周期和高水位是否正确。
- 跨核 channel 全局 `Σ depth ≤ 8`。
- L0A/L0B/L0C double buffer 用 Channel depth 表达，并按每级真实物理字节 × depth 核算；Buffer 不能模拟多级。

### 3. VF 区域正确性

- default `with vf(outputs=[...])`：`outputs` 列表**完整列出**所有区域内写出的 buffer；漏列 → 融合后数据丢失。
- VF 区域内**无分支**、程序序保持（`ascvec-fuse-vf` 依赖此）。
- `mode='register'/'raw'` 不带 `unroll=`/`outputs=`（否则 raise）。
- cast→mem_copy(nd2nz) 边界不可折叠 —— 别指望跨此边界融合。详见 `../cannbotdsl-vf-fusion/SKILL.md`。

### 4. AST 预处理兼容性

- 动态 `for`/`if`/`while` region 内**无** `return`/`raise`/顶层 `break`/`continue`（会 raise，但审查阶段提前发现更好）。
- 编译期常量循环用 `range_constexpr` / `const_expr`，不要误用普通 `range` 导致 IR 爆炸。
- region 需要的闭包变量显式传参，不靠捕获。

### 5. Buffer 预算合理性

- 各 mem_loc 总字节 ≤ 硬限制：UB 256KB、L1 512KB、L0A/B 64KB、L0C 256KB（见 `../cannbotdsl-op-design/SKILL.md §2`）。
- 短生命周期中间量常驻 UB/L1/L0，不无理由落 GM scratch。

### 6. Buffer / Channel 选型

- 单块临时量、驻留累加器 → Buffer；depth-N、double buffer、生产消费同步 → Channel。
- 旧 NBuffer/`.current()`/`.advance()` API 不得出现；Channel 不能表达的旧手动多级方案应明确标“不支持”。
- 执行边界的 allocator rewind ≠ 同步 handoff；ready token 不能被 free/init token 顶替。

### 7. Probe 覆盖检查

- 若 kernel 依赖某个疑似框架限制的行为，确认 `probes/` 有对应结论，或提示补 probe（见 `../cannbotdsl-framework-probe/SKILL.md`）。

### 8. `const_expr(cond)` 守卫变负失效

- 形如 `if const_expr(NPAD > 0):` 的保护，当 `NPAD = VH - BMV` 因上游参数变化而变负时**静默跳过**，而它本该保护的越界写读照常发生 —— 守卫非但没报警，还掩盖了唯一线索。
- **审查动作**：所有 `const_expr(cond)` 守卫，检查 cond 中的变量是否可能为负；若可能，确认有 `assert VH >= BMV` 等下界断言，或参数在 host 侧推导而非硬编码。详见 `../cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。

### 9. Probe 可信度审查

- kernel 依赖探针结论时，审查探针本身的 PASS 是否可信。两种假 PASS：① 一个 `@jit` region 里连写同一 buffer 的多个偏移 → 触发 VF 陷阱 4（桥接陈旧 store）反噬探针；② 用了对称输入（如 `vmadd(a, b, a)`，交换律下两种读法同值）→ 探针永远 PASS。
- **审查动作**：对每个探针问一句「什么情况下这个探针会 FAIL？」答不出来，它测的就不是你以为的东西。详见 `../cannbotdsl-probe-debug/SKILL.md` §2.1。

## 通用代码质量（摘自 kernel-coding-style）

- 硬件角色（Cube/Vector）/数据流角色/边界 helper/顶层调度职责分离；buffer 在最接近使用者的类里申请。
- 领域语义命名，能追踪到公式里的张量/状态/输出；不用 `C/D` 裸缩写、`_emit_xxx` 泛名。
- 共享 handoff 接口要窄：只传共享 buffer/token 的小对象，不把整个角色对象当 buffer 容器传。
- 详见 `../cannbotdsl-kernel-structure/references/kernel-coding-style.md`（完整门禁清单在 `cannbotdsl-op-develop` 质量门禁段）。

## 审查路由

| 场景 | 做法 |
|------|------|
| **quick review**（单函数） | 只跑维度 1-3（sync/资源预算/VF），最易出 runtime bug |
| **file review** | 全部维度 + 通用质量 |
| **PR review** | file review + `git diff` 聚焦改动、+ L0 codegen 断言（`.asc` sync 配对）确认未破坏 |

## 门禁

- sync 配对、Channel sync 预算、Buffer/Channel 容量和 VF outputs 完整性是**必查项**，发现即为 blocking。
- 结论区分"确定的 bug"和"风格建议"；bug 给出 `file:line` 与失败场景。
- 无 NPU 时用 translate + `.asc` 做静态 sync 审计，不要因为不能上真机就跳过 sync 审查。

## 参考

- `../cannbotdsl-kernel-structure/references/kernel-coding-style.md`、`../cannbotdsl-precision-debug/references/mixed-kernel-debug-lessons.md`
- `vf.py`（守卫）
- `../cannbotdsl-vf-fusion/SKILL.md`（VF 正确性）、`../../debug-skills/cannbotdsl-crash-debug/SKILL.md`（同步不齐的运行时后果）
