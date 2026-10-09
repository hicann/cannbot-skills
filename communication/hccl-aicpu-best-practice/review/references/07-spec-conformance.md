# 07 · 代码 ↔ dataflow spec 对照（S00–S03）

**dataflow spec** 是描述"谁的哪段字节搬到谁的哪段字节、在哪条流上、什么时候同步"的一份 markdown，
新算法 / 重构数据流的改动通常会附一份。有它就用上：**spec 是评审的参照系**，
spec 五分钟能 review 完，代码不能。作者没写 spec 也能评，只是少了 S00–S03 这组契约检查。

本 skill 按下面这套约定解析 spec（章节号 + 表格行），这也是 HCCL 团队 dataflow spec 模板的写法：
第 4 章「Buffer 布局」有一行 `| **scratch 倍数** | ... |`，第 5 章「切片」有 `| \`RPT\` | ... |`，
第 6 章「数据流」用 `sync main -> sub` / `sync sub -> main` 标同步点。

```bash
python3 "$SKILL_DIR/scripts/review_template.py" --repo "$HCCL_REPO" <file>.cc --spec <spec>.md
```

脚本先校验第 4/5/6 章以及 `scratch 倍数`、`RPT` 两行是否存在。格式不完整或仍有 `TBD:`
会报 S00 并停止 S01–S03，避免把“没有解析到”误当成“对照通过”。

## 1. 脚本能对上的四条

| 规则 | 对什么 | 级别 |
|---|---|---|
| **S00** | spec 第 4/5/6 章、关键表格行与数据流围栏是否完整，同步动作是否可解析，是否仍有 `TBD:` | MAJOR |
| **S01** | spec 第 4 章「scratch 倍数」表格行 ↔ `CalcScratchMultiple` 的返回表达式（`N` 会按 `templateRankSize_` 对齐） | MAJOR |
| **S02** | spec 第 5 章 `RPT` ↔ 代码里有没有 repeat 循环。**声明了非 1 的 RPT 却没有循环 = BLOCKER**（只搬第 0 块） | BLOCKER / INFO |
| **S03** | spec 第 6 章双向同步点条数 ↔ `PreSyncInterThreads` / `PostSyncInterThreads` 的调用次数 | MAJOR |

对不上时**不要默认是代码错**：也可能 spec 过期了。结论写成"两者不一致，请作者说明以哪份为准"，
并要求把最终答案回写进 spec。

章节标题只在代码围栏外识别，支持反引号/波浪线围栏。
S03 只统计第 6 章数据流围栏内独占一行的 `sync main -> sub` / `sync sub -> main` 动作，
忽略正文提及和 `#`、`//`、块注释。条件写在外层并缩进动作，行末可加注释。
单线程正文可以明确写“不需要 sync main -> sub”，无需为躲避误报改措辞。
围栏缺失、为空、未闭合或同步动作写法不明确时报告 S00，不能静默当作零同步。
S03 比较的是**静态出现次数**，无法证明循环内外、分支可达性或 notify 配对正确。
即使数量相同，第 6 章也必须人工核对每个同步点所在的 repeat/step/条件分支。
建议在既有伪代码旁注明同步作用域；数量不等时先核对等价控制流，不直接推断代码错误。

## 2. 人工逐章对照（脚本判不了的部分）

| spec 章节 | 对照代码里的 |
|---|---|
| 1 元信息 | 类名 / 文件名 / 父类 / 落盘目录；「所属层级」是分层 → 必须有 repeat 循环 |
| 2 适用条件 | `CalcCostCoeff` 的 `return {}` 条件、`KernelRun` 里的规格校验 |
| 3 资源 | `CalcRes` / `GetRes` 的三个字段、`GetNotifyIdx*` 的下标序列、channel 申请函数 |
| 4 Buffer 布局 | scratch 分段方式与 `CalcSlice`；`hcclBuffBaseOff` 的加法 |
| 5 切片 | `SS` 之类自定义符号的定义、尾片归属、五个 stride 的取法 |
| 6 数据流 | **逐条对**：每条 `->` / `into` 边的 src/dst 句柄、offset、len、挂在哪个 thread、同步位置 |
| 7 边界条件 | 空数据/单 rank、归约特殊路径、支持性状态时序、完整模式枚举与资源分支 |
| 8 Cost model | `CalcCostCoeff` 的各项与 `props.algoType` |
| 9 验证 | 定向与回归双轨、拒绝路径和未覆盖范围；各项绑定当前源码与 Spec 身份 |
| 10 待确认 | **还剩 TBD 就不能合入**——影响语义的 TBD 必须先落定 |

第 6 章是重头：spec 里一个 offset 写错，代码全对也是错的。对照时按**边**逐条读，
每条边确认四件事：**源句柄、目标句柄、长度、挂哪条流**。

## 3. 记法对照表

| spec 记法 | 代码 |
|---|---|
| `IN[off,len]` | `DataSlice(buffInfo.inputPtr, buffInfo.inBuffBaseOff + off, len, cnt)` |
| `OUT[off,len]` | `DataSlice(buffInfo.outputPtr, buffInfo.outBuffBaseOff + off, len, cnt)` |
| `SCR[off,len]` | `DataSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + off, len, cnt)` |
| `SCR@p[off,len]` | `channels.at(p)[ch].remoteCclMem.addr` + 同样的 offset |
| `OUT@p` / `IN@p` | `remoteOutputGraphMode` / `remoteInputGraphMode`（图模式） |
| `->` 覆盖 / `into` 累加 | `LocalCopy` / `LocalReduce` |
| `sync main -> sub` | `PreSyncInterThreads` |
| `sync sub -> main` | `PostSyncInterThreads` |
| `repeat for rpt in 0..RPT-1` | `for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt)` |

**跨 rank 边只写一个方向**：写模式只用 `txSlicesList_`，读模式只用 `rxSlicesList_`。
spec 里同一条边写了两个方向，或代码里两套都填了，都要提出来。

## 4. 没有 spec 的情况

模式 B（局部改动）和模式 C（纯只读评审）都可能没有 spec，这时：

- 不要为了评审现造一份 spec 让作者填——**补写 spec 是另一件事，不是评审的前置**；
- 判据只有一条：改动**动到了**"谁的哪段字节搬到谁的哪段字节、在哪条流上、什么时候同步"
  才要求补 spec；只是改日志、补校验、调常量的，不要求。
