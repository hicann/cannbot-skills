# 归约 / 重排类算子设计模式索引（Design Pattern Index）

生成归约族或重排/搬运类算子前先读本文件，用它决定采用哪套设计结构。模式可以组合，但不要临时发明新模式。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的模式，再只读对应文档的 `设计要点`；只有实现细节不够时，再继续读 translator references 中对应的实现指南。

## 归约族设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| 沿任意维（或全部）`sum / mean / max / min / prod` | `@references/design-patterns/references/reduce_design.md` — (O,R,I) 三分解路由选路径 |
| 均值/方差统计量（`layer_norm` / `rms_norm` / `batch_norm` / `var` / `std`） | 同上（forward 内含归约，落入同一套路由） |
| 需算方差/二阶矩（`var` / `std`，如 norm 类的归一化分母） | reduce_design — 二阶矩算法路线（同趟 sum+sumsq vs 两趟/Welford，按动态范围选） |
| 非末轴归约（内维 I > 1） | reduce_design — Path A 跨行 RA，沿 I 分块 iTile，不物理转置 |
| 末轴归约、单行可装 UB | reduce_design — Path B 多行批归约（AR） |
| 末轴归约、单行超 UB | reduce_design — Path C 分块两级树 |
| 行尾需对齐/补零 | reduce_design — tile 内 pad 语义规划（sum→0、max→-inf、min→+inf、prod→1） |

## 重排/搬运类设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| 奇偶交织 / 规律重排 / stride 切片重组（`chunk`/`split`/`cat`/`stack` 半区拆分） | `@references/design-patterns/references/shuffle_design.md` — 规律 pattern 结构（预留硬件指令映射），不建表 |
| gather / scatter / index 离散取数写 | shuffle_design — 通用结构，避免每-launch 建表 |
| 广播消费（cos/sin S=1、RoPE 交织、RotaryMul） | shuffle_design — 多行共享源行的 tile 数据流（源行去重） |
| 输出需按指定布局 | shuffle_design — 计算即按最终布局摆放，写回退化为连续搬移 |
| 小 shape（固定开销主导） | shuffle_design — 结构选择避开每-launch 固定开销 |

## 决策顺序

多个模式同时出现时，按这个顺序确定设计：

```text
1. 归约：先定 (O,R,I) 三分解与路径（A 跨行 RA / B 多行批归约 / C 分块两级树）。
2. 重排/搬运：再定取数/广播/布局结构（pattern vs 建表、源行共享、最终布局）。
3. 共同：最后定 host 核数分档（按规模带，避免 dispatch ramp）。
```

> 实现层细节（GatherMask 指令、isReuseSource、dstStride、dtype 分路、rightPadding 参数等）
> 在转译阶段处理，见 translator references：`ascendc_reduce_patterns.md` / `ascendc_shuffle_patterns.md`。
