# ScatterUpdate 越界处理：决策方法论（语言中立）

> scatter 写侧越界形态的决策规则 / 竞品证据 / 形态与写语义关系 / 哨兵机制契约 / 双模板一致性约束。设计文档越界部分的语义真值源。
>
> ⚠️ 本文档为 **scatter 自建承载**：决策方法论（优先级、二维独立）与读收集范式同源，但 scatter 是写语义、越界形态集合与读侧不同。具体语言的 kernel 写法（如 AscendC 的 `__trap`/哨兵实现）见对应语言适配层（AscendC：[adapters/ascendc/bounds-impl.md](../adapters/ascendc/bounds-impl.md)）。

---

## 1 决策规则（优先级）

```
① spec / 用户输入显式声明        → 按声明
② 未声明但对标算子语义明确        → 按对标算子（竞品分析）
③ 无法判断                      → 默认 error（fail-fast 最安全）
```

越界处理是**二维独立决策**（负索引怎么办 × 上界怎么办），不是单一三选一。写侧额外约束：**越界行为在 det/nondet 双模板下必须同一语义**——否则同输入一模式报错一成功，路由漂移不可复现。

业界证据：

| 框架/算子 | 负索引 | 上界 | 形态 |
|----------|--------|------|------|
| torch scatter_add / index_add / scatter(reduce) | 报错 | 报错 | **error** |
| TF tensor_scatter_nd_update（GPU）/ Ascend ScatterNdUpdate | 跳过 | 跳过 | **skip** |
| torch index_put_ | 报错 | 报错 | error |
| ONNX ScatterND | 报错 | 报错 | error |
| torch index_add_（axis 负值） | — | — | 属性负值归一化（非索引值） |

## 2 语义定义与防漏判原则

三形态的语义定义（与具体语言无关；AscendC kernel 写法见 [adapters/ascendc/bounds-impl.md](../adapters/ascendc/bounds-impl.md)）：

- **error**：任一越界索引使整次调用硬失败（默认，fail-fast 最安全；对标 torch 系）。
- **skip**：越界 update 丢弃不写入、不报错（对标 TF/Ascend tensor_scatter 系）；排序域通过哨兵 key 过滤（§4）。
- **neg-norm**：负索引按维度归一化（`j += dimSize`），归一化后仍越界按 error 处理（对标 index_put_/ONNX 负索引语义）。

**防漏判原则（全形态统一前提）**：**j 的一切变换（归一化）必须在索引原值域进行，窄化截断永远最后**（`j = 2^32+7` 截断后 =7 会绕过上界检查——≥2^32 超大正值是必测区分性 case）。

**nd 形态逐维判定原则**：nd 形态每 update 有 K 维坐标，**逐维检查、任一维越界即整行失效**（结果累积后统一标记）。neg-norm 在 nd 下**逐维独立归一化**——K 维各自维度尺寸不同，负索引只能每维各自归一化，**不存在「整体归一化」的场景**（与单维写法同构，仅逐维循环展开）。

## 3 形态选择与写语义的关系

| 写语义 | 常见对标形态 | 说明 |
|--------|-------------|------|
| assign（tensor_scatter_nd_update 系） | **skip**（哨兵） | TF/Ascend 语义：越界 update 丢弃，不报错 |
| reduce（scatter_add / index_add 系） | **error** | torch 语义：越界即 IndexError |

> 这是**对标统计规律而非绑定规则**——最终按 §1 优先级决策；spec 未声明且无法判断时默认 error。

## 4 哨兵 key 机制（skip 形态的实现契约）

skip 形态通过排序域过滤实现，契约如下（双模板行为一致约束：det 哨兵沉底 / nondet continue 直跳）：

- 哨兵 = 键值域最大值（KeyT_max）。
- 有效 lin 恒非负；key 位宽按冲突域值域自适应（≤32767→16 位 / ≤2^31-1→32 位 / 64 位）保证 `冲突域值域 ≤ KeyT_max - 1`，**哨兵恒空闲**。
- 哨兵是最大值 → 排序后**自然沉底置尾**，归约/覆盖阶段从尾部跳过即可，不污染有效段。
- 排序域全非负，无符号歧义（radix sort 安全）。

```
Phase1: 越界行 → key = SENTINEL（哨兵）
Phase2: 排序（哨兵沉底）
Phase3: key == SENTINEL → 跳过（不写入）
```

## 5 双模板一致性约束（强制）

1. **det / nondet 同一越界行为（语义一致，非逐字照抄）**：判定条件（`j < 0 || j >= dimSize`）与最终语义（越界 = 不写入 / 报错）在两路径必须一致；实现动作**允许因路径结构而异**——det 有排序用哨兵沉底跳过，nondet 无排序用直接跳过。禁止的是语义自行分叉，不是写法差异。
2. **error 形态不依赖编译开关**：运行时硬失败原语（如 AscendC `__trap`），禁止用会被 release 静默忽略的断言宏。
3. **golden 同步**：error 形态 kernel 硬失败 ⟺ golden raise；skip 形态 golden 跳过该 update（不写入）。
