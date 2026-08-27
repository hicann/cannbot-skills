# full-audit — 全量同步检验工作流

> 默认工作流。对 Kernel 源码做同步检验并输出修正建议。

## 阶段

### Stage 0 — 信息收集
1. 确认目标文件/目录
2. 识别算子侧别：单核 / Cube+Vec 分核（MIX）/ 多核
3. **关联文件发现（重要）**：单文件检测时，脚本无法看到被调函数的内部实现（如 `epilogueOp()` 调用的 `block_epilogue_fixpipe.h` 中的 `Relu`/`DataCopyPad`），会导致 CrossCoreWaitFlag 的 PIPE 方向匹配（SYNC-05）漏报。因此：
   - 扫描目标为单文件时，Read 该文件的 `#include` 指令，在同目录或 include 路径中找到关联头文件
   - 将关联头文件一并传给脚本扫描（`python3 scripts/sync_audit.py 目标文件.h 关联头文件1.h 关联头文件2.h`）
   - 若无法找到关联文件，在报告中标注「单文件检测，关联文件未纳入扫描，PIPE 方向匹配（SYNC-05）可能漏报」

### Stage 1 — 脚本扫描（全报不筛）

#### 1a. 运行检测脚本
- 运行：`python3 scripts/sync_audit.py <目标文件+关联文件> --format json`（内部自动调用 case_retriever.py，候选 detail 自带历史 case 证据）
- 运行：`python3 scripts/ascendc_flow_analyzer.py <目标文件+关联文件> --format json`
- analyzer 全程使用正则 frontend，无需安装额外依赖。

#### 1b. 全量编号输出（LLM 不筛选）

> **核心原则：LLM 是转述者，不是判断者。**（本节是 [SKILL.md 执行规则 6「禁止否决脚本候选」](../SKILL.md) 的操作细则，其他 workflow 引用本规则时以 SKILL.md 为准）

LLM 的职责：
1. 把三个脚本输出的**所有**红线和高级别候选**全部编号输出**
2. 原样转述脚本的 message 和 detail（含历史 case 证据）
3. 为每个候选给出修正 diff（基于 case_retriever 的修复提示 + 同族文件风格）
4. 不分析、不判断、不筛选、不否决、不标注"疑似误报"

**LLM 绝对不允许做的事**：
- ❌ 判断某个候选"是否真的是问题"
- ❌ 标注"疑似误报""经分析为合法""无需修改"
- ❌ 用推理排除任何脚本候选（如"同流水保序""不同 buffer 不同 flag""iter0==0 时相等"等）
- ❌ 自行决定不报某个候选

**为什么（否决与补查的不对称性）**：实测 LLM 4 次用推理否决脚本正确候选（同流水保序/bias数据流独立/不同buffer不同flag/iter0==0时相等），均被证明错误。否决造成的漏报用户不可见、不可恢复；而补查（1c）只会**新增**候选，最终仍由用户裁决，错了可被用户排除。因此 LLM 推理只允许用于"加候选"和"给修法"，不允许用于"减候选"。

#### 1c. LLM 补查（只加不减）

LLM 可以**追加**脚本检不到的候选（只加不减，不触碰脚本已有候选）：
- **SYNC-02 跨函数缺同步**：Read 关联文件，找 Relu(V) 后紧跟 DataCopyPad(MTE3) 但中间无同步
- **SYNC-05 CrossCore PIPE 方向**：WaitFlag 后调用的函数内部首个消费操作属于哪个 PIPE
- **SYNC-14 索引一致性**：flag_id 索引变量与 buffer 索引变量是否同源（脚本已自动检测，LLM 可补充跨文件场景）

LLM 补查的候选项标注为「LLM 补充候选」，与脚本候选合并编号。

### Stage 2 — 修正方案展示（不立即修改）

对**所有**红线和高级别候选（脚本的 + LLM 补查的），给出逐行修改方案：

1. **编号**（用 `(1)(2)(3)...`）
2. **文件名 + 精确行号**
3. **修改前/修改后对比**（左侧标注行号）
4. **条例编号与严重级别**
5. **一句话原因**（直接引用脚本 message 或 case_retriever 修复提示）

> **修正原则（按优先级降序，冲突时高优先级胜出）**：
> 1. **case_retriever 修复提示**：历史 case 的 diff 是最可靠的修法参考。机制选择照抄提示（提示"加 PipeBarrier<PIPE_ALL>"就用 PipeBarrier，不得改用 Flag 方案）
> 2. **区分重叠 vs 一次性**：双缓冲/热循环重叠用 SetFlag/WaitFlag 精确方向；一次性初始化路径（如 `isXxxInitialized_` 保护的首次执行）用 PipeBarrier，简洁且不消耗 eventID
> 3. **同族文件风格仅决定表面写法**：命名空间前缀、缩进、helper 复用等表面风格照同族；**不得以"同族有 Flag 辅助函数"为由推翻 1/2 的机制选择**（实测失误：初始化路径本应按 case 提示用 PipeBarrier<PIPE_ALL>，却因同族存在 TPipeSetWaitFlag 而改用 Flag 方案）
>
> **diff 语法自检（出手前必过）**：按 [fix-patterns.md「修正 diff API 签名自检」](../references/fix-patterns.md) 逐行核对——`PipeBarrier` 模板参数是 `PIPE_*` 顶层枚举（**不存在 `HardEvent::PIPE_ALL`**，全限定写法 `AscendC::PipeBarrier<PIPE_ALL>()`）；`SetFlag/WaitFlag` 模板参数是 `HardEvent::方向`。diff 中出现 `HardEvent::PIPE_` 即为必错组合。

> **关键规则：列出所有修改方案后，不立即执行修改。** 向用户提问："以上共 N 个修改方案，你要执行哪几号？" 等用户选择后只执行选中的。

### Stage 3 — 性能建议（可选）

性能级别候选（SYNC-09/11）列出但不编号，供用户参考。

## 输出格式

```
## 同步检验报告

### 需修正（红线 + 高，全部编号，不做筛选）

(1) [SYNC-xx][红线] 文件名:行号
原因：<脚本 message，原样转述>
历史 case：<case_retriever 附带的历史 PR 证据，如有>

修改前（第 NN 行）：
```diff
87:     AscendC::WaitFlag<HardEvent::MTE1_MTE2>(ZERO_FLAG);
```
修改后：
```diff
87:     AscendC::SetFlag<HardEvent::MTE1_MTE2>(ZERO_FLAG);
```

(2) [SYNC-xx][红线] ...

---

以上共 N 个修改方案，你要执行哪几号？（如选 (1)(3)，或输入 all 全部执行）

### LLM 补充候选（如有）
(脚本未检出，LLM 跨函数补查发现)

### 性能建议（可选，不强制）
- [SYNC-09] 文件名:行号  <建议>
```

> **注意**：
> - 脚本候选全报不筛，LLM 只转述不判断
> - 信息级别不输出，性能级别只列出不编号
> - 筛选权完全交给用户——用户看到全部候选后自己决定改哪几个
