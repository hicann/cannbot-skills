# ArgReduce 切分与资源模型

> **层级**：通用知识层（generic/），语言中立。
>
> 覆盖：UB 双通道预算、FullLoad/Split 判定、多核切分、GroupReduce 触发与 workspace 模型、跨块/跨核索引偏移。选择语义见 [selection-semantics.md](selection-semantics.md)；谓词真值由适配层 capability-mapping 提供。

## 1 UB 预算（三通道）

ArgReduce 的片上驻留是**三通道**的——值流、位置流、输出/暂存同时占用片上容量（对比 Reduction 范式的单值通道）：

```
ubNeed(R_chunk) = R_chunk × width(value_dtype)          // ① 值流
                + R_chunk × width(位置元素)              // ② 位置流（等宽选型，见 selection-semantics §5；生成驻留，不从 GM 载入）
                + 输出与暂存项                            // ③ 输出/暂存通道：输出搬运 DB buffer + 值/位置暂存
                + padding（按 DATA_ALIGNMENT）
ARA 场景重排路径另有重排目标 buffer（按适配层映射声明）
```

- **③ 输出/暂存项不可省略**：输出侧 outQueue（双缓冲时 ×2）与块间合并暂存同占片上；漏算将导致预算失真（正确性风险：InitBuffer 超实际容量失败）。按"每输出元素固定字节数"折算计入反解分母
- 预算不等式（单输出块）：`ubNeed + tmpBuffer ≤ UB 可用容量`。tmpBuffer 为 `INDEX_TRACKING_REDUCE` 指令的 scratch 约定（如适用）
- **UB 容量必须运行时查询**（宿主侧平台属性），禁止按文档标称值/仿真配置值硬编码——同代不同 SKU 实测容量不同
- 元素级可用上限建议取**保守口径**（参考实现口径优先于理论值：工业实现 ubElement=57216，理论估算 58304 偏乐观约 2%）

## 2 FullLoad / Split 判定

| 模式 | 判定 | 数据流 |
|------|------|--------|
| **FullLoad** | 单输出块的 R 候选可全部驻留 UB（§1 不等式） | 整块载入 → 块内一次选择 → 直出 |
| **Split**（沿 R 切块） | R 超 UB 预算 | 逐块载入 → 块内局部 (值,位置) → **跨块 select 合并（加 chunkOffset）** |

- 跨块合并的正确性由 select 可结合性保证（[selection-semantics.md](selection-semantics.md) §2），与切块顺序无关
- **位置精度与切块的耦合**：块内偏移必须 ≤ 位置元素整数容量（等宽选型，见 [selection-semantics.md](selection-semantics.md) §5）——切块大小受 UB 预算约束通常天然满足，但必须显式核对；全局位置由块基址 + 块内偏移拼接承载（上限为输出整数类型容量），**不构成切块上界**
- **R 超 UB 单块容量时的声明义务**：当 R 总长超过单块可驻留的 UB 容量（三通道预算反解的上界）时，设计必须**显式声明处理策略**——R 切块循环（跨块合并，见 §2）/ 显式上界限制 + 报错 / 降级路径——**不得隐式默认**。已实战验证的三种策略：AscendC arg_max_custom 用 R 切块循环（cutRSize 反解）；DSL argmin 用显式上界 + RuntimeError（ARA ≤ 512 / AR ≤ 98304，超界报错而非静默错误）

## 3 多核切分

按优先级依次判定（前者放不下才考虑后者）：

| 切核轴 | 适用场景 | 核间关系 | 对应 |
|--------|---------|---------|------|
| **A1 切核** | AR / ARA 通用（输出外层并行） | 核间**无依赖**，每核独立产出自己那批输出 | 首选 |
| **A0 切核** | ARA 且 A1×A0 仍不够开满核、A0 足够大 | 核间无依赖（输出内层并行），可与 A1 切核**复合为双轴切核** | ARA 场景增强 |
| **R 切核（GroupReduce）** | R 很大（算力型）且 A 方向无并行度 | 核间**有依赖**：局部 (值,位置) → workspace → 栅栏 → 聚合 | §4 |

**核数口径**：无依赖切核按数据量估算使用核数；GroupReduce 因栅栏要求全核到齐，**必须按物理核数**发起与同步。

**多核均衡**：按 `MULTICORE_BALANCE` 约定（前 k 核各多担 1 份，负载差 ≤ 1）分配输出块；禁止截断式分配（空块参与 GroupReduce 聚合会读脏数据）。

## 4 GroupReduce（R 借多核）

### 4.1 触发判定

```
触发条件（同时满足）：
  1. R × width ≥ 物理核数 × 算力当量（R 是算力型长轴）
  2. A 方向并行度不足（A1 切核 + A0 切核都开不满核）
  3. 位置可表示（R < INDEX_STORAGE 边界，或已声明分段方案；int32 输出另需 < 2^31）
回退：若 A 方向实际足够（如 A1 接近物理核数），回退普通切核——栅栏开销不值得
```

### 4.2 结构与 workspace

```
Phase 1：每核分到一段 R → 计算局部胜者 (winValue, winPosition)
         → 写 workspace（值段 | 位置段）
全核栅栏（CROSS_CORE_SYNC：全核到齐才放行，含调度模式配置）
Phase 2：聚合——两种形态
  a) 单核聚合：core0 读全部局部结果，逐对 select，写出
  b) 全核并行聚合：每核负责一批输出（A0 方向分块），读该批全部局部结果 select
```

**workspace 布局**（双段连续）：

```
[value: usedCore × aTotalAlign] [position: usedCore × aTotalAlign]
aTotalAlign 按 DATA_ALIGNMENT 对齐；总大小 = 2 × usedCore × aTotalAlign × width
```

**聚合正确性三要素**（缺一即静默错误）：

1. 局部 position 在 Phase 1 写入前**加核基址偏移**（或聚合时统一加，一次施加）
2. 聚合 select **必须施加平局规则**（比较掩码含 position 裁决），不得退化为纯值比较
3. 空块核（分到 0 元素）不得参与（均衡分配保证无空块，见 §3）

## 5 切分与场景的二维组合

场景（scene §3）× 切分（本文档）是**正交**的两个自由度，全组合合法：

| | A1 切核 | A0 切核 | R 切块 | R 切核 |
|---|---|---|---|---|
| **AR** | ✅ 首选 | —（无 A0） | ✅ R 超预算 | ✅ GroupReduce |
| **ARA** | ✅ 首选 | ✅ 增强（可复合双轴） | ✅ | ✅ |
| **COPY** | ✅（纯搬运，按带宽切） | — | — | — |
| **EMPTY** | 全核早退（发起方式由适配层约定） | | | |

适配层按"场景 × shape 相对量（核数/UB/通道宽度）"给出具体模式选择（AscendC 见 [../adapters/ascendc/mode-mapping.md](../adapters/ascendc/mode-mapping.md)，含工业实现的分段枚举映射表）。

## 6 同步（指向独立文档）

切分只决定"数据切到哪里、每个核做什么"，不定义同步原语。同步的**依赖推导规则**（核内流水何时必须等、GroupReduce 何时必须全核对齐）见 [sync-model.md](sync-model.md)；具体语言用事件对还是 Channel 交接，见各适配层。
