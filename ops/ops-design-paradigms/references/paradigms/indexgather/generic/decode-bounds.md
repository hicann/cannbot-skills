# IndexGather 越界处理：决策方法论（语言中立）

> 读收集侧越界处理的决策规则 / 竞品证据 / 两步模型 / 防漏判原则 / 易混淆语义注记。设计文档越界部分的语义真值源。具体语言的 kernel 写法（如 AscendC 的 `__trap`/回绕/钳位实现）见对应语言适配层（AscendC：[adapters/ascendc/decode-bounds-impl.md](../adapters/ascendc/decode-bounds-impl.md)）。

---

## 1 决策规则（优先级）

```
① spec / 用户输入显式声明        → 按声明
② 未声明但对标算子语义明确        → 按对标算子（竞品分析）
③ 无法判断                      → 默认 none + err（负索引/上界均报错，fail-fast 最安全）
```

越界处理拆成**顺序两步**：先决定负索引怎么处理，再对 Step 1 后仍越界的值做 Step 2 处理。业界证据：

| 框架/算子 | 负索引处理 | 越界处理 | 预设 |
|----------|-----------|----------|------|
| torch 全系（gather / take_along_dim / index_select / take） | 不处理 | 报错 | **error** |
| **ONNX Gather / GatherElements / GatherND** | **`+dim` 单次回绕** | 报错 | **neg-wrap** |
| NumPy 高级索引 | `+dim` 单次回绕 | 报错 | neg-wrap |
| `np.take(mode='wrap')` | 不处理 | `%dim` 真模 | mod-wrap |
| `np.take(mode='clip')` | 不处理 | 钳位 | clamp |
| JAX `take(mode='fill')` | 不处理 | 填充 | fill(fill_value) |
| JAX `promise_in_bounds` / torch GPU 未定义路径 | 不处理 | 不检查 | undefined |

实现策略指引：Step 1 / Step 2 策略**编译期固定**（spec 声明 → kernel 分支/模板）；仅当算子接口本身暴露 mode 属性（如对标 `np.take(mode=...)` 的算子）才做运行期属性。

## 2 两步越界处理模型

### 2.1 Step 1：负索引处理（`j < 0`）

| 策略 | 行为 |
|------|------|
| `none` | 不处理，交给 Step 2 |
| `+dim` | 单次回绕 `j += dim`；仍 `<0` 时交给 Step 2 |
| `%dim` | 真模回绕，任意负值合法；C++ 负余数须再 `+dim`（见 [decode-bounds-impl.md](../adapters/ascendc/decode-bounds-impl.md) §3） |

Step 1 只负责负索引；上界越界一律留给 Step 2。

### 2.2 Step 2：越界处理（`j < 0 || j >= dim`）

| 策略 | 行为 |
|------|------|
| `err` | 硬失败（默认，fail-fast；对标 torch 系） |
| `fill` | 越界位写填充值（默认 0，`fillValue` 可配置），不发起源读 |
| `skip` | 跳过该索引 / update（写散射用；读收集不选） |
| `%dim` | 真模回绕，任意越界值合法（对标 `np.take(mode='wrap')`） |
| `clamp` | 钳位到 `[0, dim-1]`，任意越界值合法（对标 `np.take(mode='clip')`） |
| `undefined` | 不做逐元素检查（须 spec 显式声明风险；静态校验仍必须有） |

`fill` 的位模式按字节搬运，禁数值转换（标量属性来源：host 位级打包；tensor 来源：运行时按字节读）。Step 2 的 `%dim` / `clamp` 对 Step 1 未处理的负值同样生效，因此不需要在 Step 1 额外增加 `clamp`。

### 2.3 常见预设

| 预设 | Step 1 | Step 2 | 说明 |
|------|--------|--------|------|
| `error` | `none` | `err` | 默认，fail-fast |
| `neg-wrap` | `+dim` | `err` | ONNX / NumPy 高级索引 |
| `mod-wrap` | `none` | `%dim` | `np.take(mode='wrap')` |
| `clamp` | `none` | `clamp` | `np.take(mode='clip')` |
| `fill` | `none` | `fill` | gather 默认填 0，`fillValue` 可配置 |
| `skip` | `none` | `skip` | scatter 越界跳过 |
| `undefined` | `none` | `undefined` | 显式不做逐元素检查 |

预设只是常用组合；Step 1 / Step 2 可按 spec 独立选择（如 `%dim + err` 表示负索引真模、上界报错）。

**防漏判原则（全模型统一前提）**：**j 的一切变换（回绕/钳位/取模/截断）必须在索引原值域进行，窄化截断永远最后**——`j = 2^32 + 7` 截断后 = 7 会绕过上界检查（漏判、按错误偏移读内存）；≥2^32 超大正值是必测区分性 case。

## 3 易混淆语义注记

- **padding_idx / mask_zero**（embedding 族）：特定**合法**索引（∈ [0, dim)）→ 零向量输出（该位梯度置零）——是输出值映射，**不是越界形态**，勿用 bounds 机制实现。
- **bool index**：torch gather 系不允许（必须 long/int），布尔掩码走 masked_select（decompose 路径）。
- **回绕两义**：`+dim`（单次回绕，`[-dim, dim)` 之外交给 Step 2）≠ `%dim`（真模，任意 j 合法）——对标时先确认目标框架是哪种（ONNX/NumPy 高级索引 = 单次；`np.take(mode='wrap')` = 真模）。

## 4 性能注记（策略选型考量）

- `none + err` / `+dim + err` / `none + clamp`：仅比较与加减，零额外成本
- **`%dim`**：整数取模一条（除法级成本，且负值域使快除不直接适用）
- `fill`：越界分支内写标量 + 跳过，合法路径零开销
