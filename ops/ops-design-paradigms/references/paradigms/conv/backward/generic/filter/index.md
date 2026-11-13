# 卷积反向 filter 分支知识索引

本层建立 x、dedy 到 FP32 dedw 的权重梯度模型。公共轴义与规格处理见 [通用知识索引](../index.md)；本分支的记号见 [language §13](../language.md#13-filter-分支记号)，规格绑定见 [preprocessing §8](../preprocessing.md#8-filter-分支的规格归一)。

## 1 检索表

| 文档 | purpose | read_when | keywords |
|---|---|---|---|
| [scene.md](scene.md) | 场景准入、正向连接求导、M/N/K、能力与数据流 backbone | 建立权重梯度模型 | x、dedy、dedw、filter_size、组隔离 |
| [compute-flow.md](compute-flow.md) | 格式与排布线稿、两路搬运、窗口、地址、任务、归约及 DAG | Kernel 设计与布局推导 | dedy_flat、x_win、L1 父段、Load3D、Fixpipe |
| [tiling/tiling.md](tiling/tiling.md) | 基本块、L1 预算、任务分发、求解和 Host–Kernel 字段 | Tiling 选择与实现 | baseM/N/K、stepKa/b、Bound、singleCore* |
| [tiling/templates/mn-streamk/tiling.md](tiling/templates/mn-streamk/tiling.md) | MN_StreamK 求解顺序、小任务调整、运行时准入和产出示例 | 展开选中的求解模板 | streamkType=0、原始组、普通/拆核、宽片 |
| [verification.md](verification.md) | 输入量化、独立 Golden、布局与贡献覆盖、精度和分支记录 | 验证设计与问题定位 | FP64、autograd、原子基值、尾块、重复执行 |

## 2 阅读路径建议

```text
+------------------------+------------------------+
| Entry: scene / language / preprocessing         |
+------------------------+------------------------+
                         |
                         v
+------------------------+------------------------+
| Design: tiling -> templates -> compute-flow     |
+------------------------+------------------------+
                         |
                         v
+------------------------+------------------------+
| Implementation: filter adapter / API / sync     |
+------------------------+------------------------+
                         |
                         v
+------------------------+------------------------+
| Verification: golden / compare / feedback       |
+------------------------+------------------------+
```

## 3 使用规则

1. 同一个 K 位置在 dedy_flat 与 x_win 中对应相同的 batch/dout/ho/wo；核坐标保持正向顺序。
2. 模板、内部窗口/搬运/写回分支分别记录；最终字段满足准入后再生成 Kernel。
3. 正确性模型、Host 复算、设备结果和性能结论分别留证；函数级构造不代替自然路由覆盖。
4. API 与工程接入见 [filter 语言适配索引](../../adapters/ascendc/filter/index.md)。
