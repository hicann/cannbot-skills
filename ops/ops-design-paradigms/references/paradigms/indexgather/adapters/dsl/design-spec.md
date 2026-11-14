# IndexGather 范式 CANNBotDSL 设计约束（design-spec）

> **层级**：DSL 语言适配层。语言中立场景判定见 [../../patterns.md](../../patterns.md)；语言规则与 DSL 专属谓词见 [language-rules.md](language-rules.md)；通用能力映射见 [capability-mapping.md](capability-mapping.md)。

## 输入与输出

**输入**：算子公式（spec 语义）、公开 Python 参数、shape/dtype/layout 范围、越界语义声明（[../../generic/decode-bounds.md](../../generic/decode-bounds.md) 优先级决策结果）、目标平台事实、当前设计问题。

**输出**：分型结论（初版 A/C + 四维组合）、结构键设计、8 维结构化契约表、kernel 骨架代码块、golden 设计、零重编译验证计划；文档文件名与章节组织由调用方指定。

## 按主题读取

| 主题 | 文档 |
|------|------|
| 分型与语义 | [../../generic/semantics.md](../../generic/semantics.md) |
| 越界形态决策 | [../../generic/decode-bounds.md](../../generic/decode-bounds.md) |
| 坐标解码与合轴 | [../../generic/decode-model.md](../../generic/decode-model.md) |
| 语言规则与 DSL 专属谓词 | [language-rules.md](language-rules.md) |
| 通用能力谓词映射 | [capability-mapping.md](capability-mapping.md) |
| 动态 shape 程序结构 | [dynamic-shape.md](dynamic-shape.md) |
| Kernel 写法 | [kernel-impl.md](kernel-impl.md) |
| 快除数学（AscendC 侧） | [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| 参照实现 | [example/](example/README.md) |

## 一致性检查

- 分型结论覆盖 spec 全部维度（解码结构 × 越界语义 × 规模路径 × tiling 预处理）；无分型直接套模板即违规
- **结构键含零 shape 数值**（N/尺寸经 8 维结构 spec + metadata 直读 + @host 派生标量流动）
- **主循环 = 三参 grid-stride** `cannbotdsl.range(tid, bound, stride)`（[language-rules.md](language-rules.md) §3；正向门核验）
- **shape 通道 = 8 维 metadata 直读（VF 外静态索引）+ `@host` 派生标量**（零 params）
- **派发 = compiled-program**（`cannbotdsl.compile` 产物 + 标量 blocks 发射；禁 dim3 JIT 直调于生产）
- 越界检查在 index **原值域**、窄化截断最后；`need_cast` 的宽窄转换在检查之后
- 零填值 = **`dt(0)` 编译期常量**（R1 整改；无无符号 dtype 不触达字面量位转换缺陷）
- **固定除数统一整除**（固定除数统一使用 `//`）
- 合轴保留于 wrapper（MergeAxis 不因动态化放弃）；`gather_dim_size` 在合轴前捕获
- 零重编译断言（同键异 shape 程序同一性 + 二次调用零编译耗时）在验证计划中
- 空输出（total == 0）宿主早退先于一切步长预计算
- 能力依赖以 [generic/capability-contract.md](../../generic/capability-contract.md) 的 17 项通用谓词为清单，以 [capability-mapping.md](capability-mapping.md) 为 DSL 真值；DSL 专属谓词见 [language-rules.md](language-rules.md) §0
