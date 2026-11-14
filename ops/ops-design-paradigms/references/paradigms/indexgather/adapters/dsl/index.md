# IndexGather 范式 CANNBotDSL 语言适配层

> **层级**：DSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)；DSL 语言规则与 DSL 专属谓词表见 [language-rules.md](language-rules.md)（含硬件执行模型 / 存储层级）；通用契约映射见 [capability-mapping.md](capability-mapping.md)。

## 1 定位

本目录承载 IndexGather 范式中**强耦合 CANNBotDSL** 的知识：动态 shape 程序结构、A/C 型 kernel 写法、越界零填、坑与规避。开发语言为 DSL 时与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ DSL 设计与代码产出
```

**参照实现**：custom_index_select（A 型），组织形态见 [example/](example/README.md)。

## 2 强制规则（生成前必读）

1. **先分型再设计**：按 patterns.md 四维判定（解码结构 × 越界语义 × 规模路径 × tiling 预处理），禁止跳过分型套模板
2. **动态 shape 结构键**（[dynamic-shape.md](dynamic-shape.md)）：键 = (data_dtype, idx_dtype, 分型结构, use64, need_cast)——**零 shape 数值**；shape 经 8 维结构 spec + metadata 直读 + `@host` 派生标量运行期流动；派发用 compiled-program（禁 dim3 JIT 直调于生产）
3. **SIMT 主循环恒三参 grid-stride** `cannbotdsl.range(tid, out_size, stride)`（language-rules.md §3）
4. **越界形态显式决策**（优先级 ①声明→②对标→③默认 error）；DSL 的 error 落地 = err flag + wrapper 上报；零填落地 = **`dt(0)` 编译期常量**（R1 整改；无无符号 dtype 不触达字面量位转换缺陷）
5. **禁止「先截断再比较」**：越界检查在 j 原值域进行，need_cast 的宽窄转换在检查之后
6. **去 pack 化**：spec 用原生 dtype（纯拷贝通路对 dtype 不敏感；fp64/complex 等在 torch 层 view 别名）——规避 ABI dtype 守卫与无符号字面量位转换双缺陷
7. **合轴保留于 wrapper**（= generic/decode-model.md MergeAxis）：merge_rank/dim 进结构键；段头广播豁免与两套步长分离照通用层
8. **固定除数统一整除**：固定除数统一使用 `//`（不设快除/整除分支）
9. **空输出早退**先于一切步长预计算
10. **canonical 域锚定**：解码算术链以 `i + dtypes.int64(0)` 锚定（language-rules.md §4）

## 3 阅读路径

| 任务 | 入口 | 内容 |
|------|------|------|
| 设计契约 | [design-spec.md](design-spec.md) | 输入/输出 / 按主题读取 / 一致性检查清单 |
| 能力谓词映射 | [capability-mapping.md](capability-mapping.md) | 通用契约 17 项 DSL 真值；DSL 专属谓词见 language-rules.md §0 |
| 程序结构 | [dynamic-shape.md](dynamic-shape.md) | 结构键 / 8 维 metadata 契约 / 零重编译 / 零元张量（dt(0)） |
| Kernel 写法 | [kernel-impl.md](kernel-impl.md) | A 型三因子 / C 型合轴解码 / 越界零填 / 内联骨架 |
| 参照实现 | [example/](example/README.md) | 算子文件组织与知识映射 |

## 4 范式内语言知识

- [language-rules.md](language-rules.md) — DSL 专属能力谓词表 / import 纪律 / 动态 shape / grid-stride / canonical / volatile / 缺陷规避 / DSL 前端行为模型（必读）
- [capability-mapping.md](capability-mapping.md) — 通用 17 项能力谓词的 DSL 真值映射（必读）
- [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) — 芯片级硬件模型（存储层级/VF/DCache/volatile/超发核，语言中立）
- [../../generic/thread-layout.md](../../generic/thread-layout.md) — SIMT 线程排布模型（一维 grid-stride / 核数两步法 / PER_CORE_MIN，语言中立）
- [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) — 快除数学（AscendC 侧）
- [../../../common/cast-rules.md](../../../common/cast-rules.md) — 位级搬运纪律
