# Template 评审报告 · <类名>

> 用法：拷贝这份模板填。**结论在最前面**，证据带 `文件:行号`，每条意见都要写清"后果"。

| 项 | 值 |
|---|---|
| 评审对象 | `src/ops/{算子}/algorithm/template/aicpu/ins_temp_<name>.{h,cc}` |
| 算法族 / 变体 | mesh \| nhr \| ring \| hd / one-shot \| two-shot \| … |
| 判据可信度 | mesh/NHR：仓内实现校准；ring/HD：注明哪些是实验判据、哪些是仓内事实 |
| Spec 哈希 / 交付清单 | `<sha256>` / `<清单路径，检查计划交付文件是否被忽略>` |
| 对应 spec | `<path>.md`（无则写「无」，并注明属模式 B 局部改动还是模式 C 只读评审） |
| 检查器 | `review_template.py` → BLOCKER n / MAJOR n / MINOR n / INFO n |
| 基线对比 | 其中 n 条是仓内普遍存在的欠账（不阻塞） |
| 代码版本 / base commit | `<被审 commit 与工作树改动说明>` / `<固定 base commit 或不适用>` |
| 人工检查 / 功能验证 | 分别记录已完成、未完成及证据路径；不得用脚本退出码代替 |

## 结论

<三选一，一句话说清静态评审结果；未完成验证时不要写“可以合入”>
- ✅ 静态评审通过（脚本和必需人工检查均完成）；功能验证状态另列
- ⚠️ 改完 BLOCKER 后重新评审：<列规则号>
- ❓ 需要作者答复 n 个问题：<列规则号>

## BLOCKER

### [W01] 未登记到 src/scatter_aicpu_kernel.cmake

**证据** `src/scatter_aicpu_kernel.cmake` 里找不到 `ops/<算子>/algorithm/template/aicpu/<文件>.cc`
**后果** host 侧编得过，但 device 侧 `libscatter_aicpu_kernel.so` 里没有这个符号，**运行时才崩**。
**修法** 加进 `add_library(scatter_aicpu_kernel SHARED ...)` 那一块。

## MAJOR（需要作者答复）

### [C01] CalcScratchMultiple 里乘了 repeatNum

**证据** `ins_temp_xxx.cc:79`
```cpp
u64 scratchMultiple = templateRankSize_ * tempAlgParams_.repeatNum;
```
**后果** 倍数双重放大（executor 自己会乘 `rankSizeLevelX`）→ 每轮可搬数据量减半 →
loop 次数翻倍，带宽白掉一半。结果仍算得对，**编译和 ST 都不会报错**，只有 review 能抓。
**修法** 去掉 `* repeatNum`，按单次 repeat 报。

### [<规则号>] <一句话>

**证据** / **后果** / **修法或问题**：<同上三段>

## Finding 台账

| ID | 规则/人工 | 有效性 | 本次/存量 | 处置 | 证据与复核 |
|---|---|---|---|---|---|
| F-001 | <规则号或人工> | 确认/误报/待确认 | <归属> | 整改/接受约束/未解决 | <源码身份、位置与复核结果> |

复审沿用 ID，重复告警关联原 ID；真实存量问题不计为误报。总数从台账计算。

## 人工发现（脚本不覆盖）

- Ring 形态及适用规则（Broadcast 的 R08 清单不可被脚本退出码替代）：<…>
- 支持性状态赋值/消费、模式枚举完整性与同步作用域：<…>
- 关键调用链与依赖文件证据索引：<…>
- 切片偏移：<…>
- 错峰顺序：<…>
- notify 下标与 `CalcRes` 申请量的对应：<…>

## MINOR（不阻塞，建议改）

<折叠成一段，只列作者这次改到的；存量欠账另起一句说明「仓内普遍如此」>

## 验证状态与待办

本 skill 只出评审意见。验证由调用工作流执行；静态评审通过不替代功能验收。

- 编译：<状态 / 对应代码版本 / 日志路径>
- hccl-vm：<状态 / 用例覆盖 / 算法命中证据 / 日志路径；新版流程分别记录定向验证与存量回归>
- UT/ST：<调用工作流要求时记录；不能冒充 hccl-vm 结果>
- 覆盖规格：<卡数 / count 边界 / 数据类型（含 INT64）/ PROD>
- 未执行项及原因：<…>
- 修复后验证：<受影响范围及新证据；不能复用修复前的结果代表修复后通过>
