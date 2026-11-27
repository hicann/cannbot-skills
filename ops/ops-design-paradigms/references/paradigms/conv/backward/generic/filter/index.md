# 卷积反向 filter 分支知识索引

> **层级**：filter 分支通用知识（generic/filter/），语言中立。与对应语言的 filter 适配层组合消费：先经 [范式入口](../../patterns.md) 的语言路由确定适配层，再按下表按需展开。

本层建立 x、dedy 到 FP32 dedw 的权重梯度模型。公共轴义与规格处理见 [通用知识索引](../index.md)；本分支记号见 [language.md](language.md)，规格绑定见 [preprocessing §8](../preprocessing.md#8-filter-分支的规格归一)。

## 1 检索表

| 文档 | purpose | read_when | keywords |
|---|---|---|---|
| [language.md](language.md) | 分支专用记号：核心记号速查（M/N/K、A/B 路、任务四元组）、GEMM 映射、切分与执行坐标（base*/singleCore*/step*）、状态与执行判定 | 查阅本分支任一文档前对齐记号（M/N/K 与 input 分支同名不同义） | mValue/nValue/kValue、dedy_flat、x_win、basicBlockIdx、stepKa/b、splitWo、isSplitKernelHW |
| [scene.md](scene.md) | 场景准入、正向连接求导、M/N/K、能力与数据流 backbone | 建立权重梯度模型 | x、dedy、dedw、filter_size、组隔离 |
| [compute-flow.md](compute-flow.md) | 格式与排布线稿、两路搬运、窗口、地址、任务、归约及 DAG | Kernel 设计与布局推导 | dedy_flat、x_win、L1 父段、Load3D、Fixpipe |
| [tiling/tiling.md](tiling/tiling.md) | 基本块、L1 预算、任务分发、求解和 Host–Kernel 字段 | Tiling 选择与实现 | baseM/N/K、stepKa/b、Bound、singleCore* |
| [tiling/templates/mn-streamk/tiling.md](tiling/templates/mn-streamk/tiling.md) | MN_StreamK 求解顺序、小任务调整、运行时准入和产出校验 | 展开选中的求解模板 | streamkType=0/1/2、原始组、普通/拆核、宽片 |
| [tiling/templates/winograd/tiling.md](tiling/templates/winograd/tiling.md) | Winograd 准入、tile 档、运行时 Cout、驻留方向、K 分段与四区 workspace | 命中 winograd 准入时展开 | F(2,3)、16 点、residentTarget、水位线、kGroups、kd 子轮 |
| [tiling/templates/dload/tiling.md](tiling/templates/dload/tiling.md) | DLoad 单批空间驻留、通道选块、资源与字段准入、完整核归约及 workspace=0 | 判定直接搬运和空间驻留方案时展开 | L1 半区、通道尾块、完整归约、FP32/hf32 |
| [verification.md](verification.md) | 输入量化、独立 Golden、布局与贡献覆盖、精度和分支记录 | 验证设计与问题定位 | FP64、autograd、原子基值、尾块、重复执行 |

## 2 阅读路径建议

```
             +-----------------------------------------------+
             | 入门                                          |
             | language.md（分支记号速查，M/N/K 同名不同义） |
             |   v                                           |
             | ../preprocessing.md §8（filter 规格绑定）     |
             |   v                                           |
             | scene.md（计算模型：正向连接求导/GEMM 映射）  |
             +-----------------------+-----------------------+
                                     |
                 +-------------------+-------------------+
                 v                                       v
  +--------------+--------------+       +----------------+----------------+
  | 设计                        |       | 验证                            |
  | tiling/tiling.md            |       | verification.md                 |
  | v（选中模板再展开）         |       | （用例/golden/比对/定位/回写）  |
  | compute-flow.md             |       +---------------------------------+
  | （数据流/迭代/同步）        |
  +--------------+--------------+
                 |
                 v
+----------------+----------------+
| 实现                            |
| ../../adapters/ascendc/filter/  |
| （指令映射/同步/模板/示例）     |
+---------------------------------+
```

## 3 使用规则

1. MN_StreamK 的同一个 K 在 dedy_flat/x_win 中对应相同 batch/dout/ho/wo；Winograd 两路配对相同 p/tile，kd 保持输出核轴。
2. 模板、内部窗口/搬运/写回分支分别记录；最终字段满足准入后再生成 Kernel。
3. 正确性模型、Host 复算、设备结果和性能结论分别留证；函数级构造不代替自然路由覆盖。
4. API 与工程接入按 [语言路由](../../patterns.md#2-语言路由) 进入所选语言的 filter 适配索引。
