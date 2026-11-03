# IndexGather 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 IndexGather 范式中**强耦合 AscendC 语言**的知识：API 实体、代码模板、参考实现、编译器约束、设计约束。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

## 2 强制规则（生成前必读）

1. **先分型再设计**：按 patterns.md 场景判定完成四维组合判定（解码结构 × 越界语义 × 规模路径 × tiling 预处理），禁止跳过分型直接套用某一型的模板
2. **越界处理策略必须在设计阶段显式决策**（优先级 ①声明→②对标→③默认 none + err）；越界采用两步模型（Step 1 负索引处理 × Step 2 越界处理），策略与业界证据见 [../../generic/decode-bounds.md](../../generic/decode-bounds.md)
3. **禁止「先截断再比较」**：`INDEX_T=int64` 时先把 j 截断成 uint32 再比较，`j=2^32+7` 截断后=7 绕过上界检查；≥2^32 超大正值是必测区分性 case
4. **IDX64 高迭代禁止编译期全展开**：uint64 乘法高为多指令模拟、占双寄存器，≥5 次快除循环全展开会寄存器溢出（实测 +87%）。IDX32（uint32 原生单指令）无此限制
5. **`indexDtype == int64` 不强制 64 位路径**：`use64Bit = (inputNum > INT32_MAX) || (outSize > INT32_MAX) || (indexNum > INT32_MAX) || (taskNum > INT32_MAX)`，`taskNum = outSize`，indexDtype 不参与
6. **越界上界 cast 按重载区分**：IDX32 特化重载可 `static_cast<INDEX_T>(gatherDimSize)`（cast 无损前提）；IDX64 重载**禁止** cast（溢出会令合法 index 误 trap）
7. **空输出早退必须先于快除预计算**：`outSize == 0` 时 strides 可能含 0 除数
8. **dim 负值归一化 A/C 型必需 + 广播约束仅 C 型**：dim 负值 +R 归一化 + 双端校验；广播约束 `d != dim 时 index.size(d) <= input.size(d)` 仅 C 型
9. **tiling 侧合轴优先**（[tiling-preprocess.md](tiling-preprocess.md)）：把 dim 之外可合并连续维折叠成 prod 轴后再下发，等效解码除法次数 `rank-1 → 段数-1`，无退化场景
10. **字节级搬移（PackT）**：按元素字节宽打包直接搬移，bit-exact——SIMT 原生标量仅到 float（无 fp64），`double`/`hifloat8` 等非常规 dtype 靠 `PackT<uint8/16/32/64>` 字节宽搬移覆盖任意 dtype。TilingKey 仅编码 idxMode 场景位，**禁止枚举 dtype**
11. **`GetUintDivMagicAndShift` 出参必须为 Local Memory 标量**：禁止直接写入 `__ubuf__` 左值（[../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §2.2）
12. **能力谓词以 [capability-mapping.md](capability-mapping.md) 为准**：未提供专用映射或标注待验证的项，须在产出中声明假设与验证义务，不得写成已验证事实。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 全局心智模型 | [overview.md](overview.md) | 三型模板 / TilingKey / TilingData 划分总览 / batch_dims 附加轴 / 命名映射 |
| Host 侧 Tiling | [tiling-preprocess.md](tiling-preprocess.md) | dim 归一化 / 广播校验 / MergeAxis 合轴 C++ 实现 / use64Bit 判定 / 多核切分 / DCache 预留 |
| 能力谓词映射 | [capability-mapping.md](capability-mapping.md) | SIMT/VF、DCache/UB、线程/寄存器、rank/index dtype、快除接口、mulhi、字节级搬移 |
| Kernel 入口与分派 | [kernel-entrance.md](kernel-entrance.md) | 入口双重载分派（idxMode）/ Process 内 rank/K 特化 switch |
| 解码与越界写法 | [decode-bounds-impl.md](decode-bounds-impl.md) | 坐标解码 / magic/shift 预计算 / 两步模型 kernel 写法 / cast 差异与截断安全性 |
| A 型场景 | [type-a.md](type-a.md) | 单坐标整维切片：两段快解 / batch_dims 组合 / 合法值映射 |
| B 型场景 | [type-b.md](type-b.md) | 坐标元组：K 编译期特化 / batch_dims 三连解码 |
| C 型场景 | [type-c.md](type-c.md) | 逐元素对齐：合轴后 rank'-1 次解码 / 广播六类场景 |
| 参考实现 | [example/](example/) | custom_gather 完整包（C 型，覆盖合轴/双重载/越界三个阶段的实现） |

## 4 公共知识依赖

以下跨范式 AscendC 公共知识位于范式外，按需展开：

| 主题 | 位置 |
|------|------|
| UintDiv 快除通用规则 | [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| 能力谓词映射 | [capability-mapping.md](capability-mapping.md) |
| API 白名单（存在性唯一判据） | [../../../../knowledge/api/regbase_api_whitelist.md](../../../../knowledge/api/regbase_api_whitelist.md) |

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 设计输出按 generic 层语义模型 + 本层实现约束组合产出
4. 编码前必须检视至少一个现有参考实现（见 [example/](example/)）
