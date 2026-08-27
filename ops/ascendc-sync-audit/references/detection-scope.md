# 检测范围（detection-scope）

> `sync_audit.py` 与 `ascendc_flow_analyzer.py` 能识别的 API、覆盖的数据流场景与不覆盖场景。

## 能识别的同步 API

| 类别 | API | 说明 |
|------|-----|------|
| 核内 Flag | `SetFlag`/`WaitFlag` | 配对、顺序、个数、方向 |
| 核间 Flag | `CrossCoreSetFlag`/`CrossCoreWaitFlag` | 配对、侧别（AIC/AIV）、flagId 计数 |
| 硬件事件 | `SyncFunc<HardEvent>` | 方向合法性 |
| 流水屏障 | `PipeBarrier<PIPE_*>` | 粒度、连续数量、跨 PIPE 误用 |
| 全核同步 | `SyncAll` | 分支对称性、必要性 |
| 标量屏障 | `SetScalarBar`/`WaitScalarBar` | 识别 |
| 队列同步 | `EnQue`/`DeQue`/`AllocTensor` | 隐式同步点识别 |
| 寄存器屏障 | `Reg::LocalMemBar<VEC_STORE, VEC_LOAD>` | V 内 store→load 屏障（RegBase 模式） |

## 能识别的数据搬运与计算 API

| 类别 | API | 用于 |
|------|-----|------|
| GM↔UB 搬运 | `DataCopy`/`DataCopyPad`（排除 `Te::` 前缀） | SYNC-02 load→compute / compute→load |
| 寄存器级搬运 | `Reg::DataCopy`/`Reg::StoreAlign`/`Reg::LoadAlign` 等 | SYNC-02 V 内 store→load 缺 `Reg::LocalMemBar` |
| Cube 侧搬运 | `Te::Copy(CopyL12L0A/B)`、`Te::Copy(CopyL0C2GM)` | SYNC-02 L1→L0→MMAD→L0C→GM 链路 |
| Cube 计算 | `Te::Mmad` | SYNC-02 MMAD 前后同步 |
| Vector 计算 | `Adds`/`Muls`/`Sub`/`Mul`/`Div`/`Exp`/`Sqrt`/`Log`/`Cast`/`ReduceMax`/`ReduceMin`/`ReduceSum`/`Compare`/`Abs`/`Duplicate` 等 | SYNC-02 计算前后同步 |
| 回收/提前退出 | `return`/`break`/`continue` | SYNC-08 跳过 SetFlag |
| 双缓冲索引 | `(var - N) % M` | SYNC-10 无符号下溢 |

## 覆盖的数据流场景

| 场景 | 检测条例 | 说明 |
|------|---------|------|
| Vector 三段流水（GM→UB→计算→UB→GM） | SYNC-02 | DataCopy 后直接计算、计算后直接搬出 → 缺 EnQue/DeQue 或 Flag |
| Cube 数据流（GM→L1→L0→MMAD→L0C→GM） | SYNC-02 | L1→L0 Copy 后 Mmad 前缺同步、Mmad 后 L0C→GM Copy 前缺同步 |
| 跨 PIPE 依赖 | SYNC-06 | PipeBarrier<PIPE_V> 后跟跨 PIPE 操作（如 DataCopy 出 GM） |
| 循环 buffer 复用 | SYNC-02 | 循环内 Set 多于 Wait → 疑似缺循环顶部 Wait |
| 循环初始/收尾配对 | SYNC-04 | 循环前缺初始 Set、循环后缺收尾 Wait |
| MIX kernel 核间同步 | SYNC-03/04 | `if ASCEND_IS_AIC/AIV` 块内 CrossCore 配对、同侧误配 |
| 同步窗口 buffer 索引一致性 | SYNC-14 | 脚本正则 + analyzer（`ascendc_flow_analyzer.py`）：producer/consumer/sync 索引根变量对比 |

## 检测方式与已知精度限制

| 条例 | 检测方式 | 已知限制 |
|------|---------|---------|
| SYNC-05 方向匹配 | 脚本启发式（`check_sync_direction_reversed`：方向写反检测 + 31 种合法方向表；`check_crosscore_pipe_direction`：CrossCoreWaitFlag 的 PIPE 与被调函数首个 API 所在 PIPE 匹配，被调经「变量类型→using 别名→模板形参 Foo_→类名前缀唯一」链解析，首个 API 在 if/if constexpr 分支内时给出条件双 PIPE wait 修法）+ LLM 补查 | 回溯窗口跨代码块时可能误判前置操作所属 PIPE；跨函数匹配需被调类与调用点同传扫描，`operator` 等高碰撞名不做全局回退（宁漏勿错归属）；首个 API 启发式无法识别「wait 实际保护函数内更深层的 Fixpipe」场景（开发遗留项，本地记录） |
| SYNC-02 跨函数缺同步 | LLM 补查（脚本不做跨函数数据流追踪） | SetFlag 在函数 A、buffer 操作在函数 B 时脚本检不到 |
| SYNC-14 索引一致性 | 脚本正则 + analyzer + case_retriever 历史证据 | — |
| Buffer 别名 | `sync_audit.py` 变量 def-use 链 + analyzer 按行索引别名/buffer 生命周期 | 语义等价（变量名不同但值相同）不做判定 |

## 不覆盖的场景

| 场景 | 原因 | 处理方法 |
|------|------|------------|
| EnQue/DeQue 内存配对 | 属 ascendc-api-best-practices API-6 | 用该 skill 检查 AllocTensor/FreeTensor 配对 |
| HCCL 集合通信时序 | 属 mc2-specific skill | 用该 skill 检查通算融合同步 |
| Tiling 侧 | Host 侧不使用信号同步 | 无需检测 |
| 编译验证 / NPU 真机测试 | 静态分析工具边界 | 修复后用 UT/ST 验证 |
