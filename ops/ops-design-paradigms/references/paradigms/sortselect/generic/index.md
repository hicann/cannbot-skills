# SortSelect 范式通用知识层索引

> **层级**：通用知识层（generic/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费，先经 [../patterns.md](../patterns.md) 确定语言。

## 检索表

| 文档 | purpose | read_when | keywords |
|---|---|---|---|
| [scene.md](scene.md) | 场景模型、Sort/Argsort/TopK/KthValue 契约与路线分类 | 判断范式与输出语义 | stable、K、largest、NaN、零符号 |
| [capability-contract.md](capability-contract.md) | 排序路线的能力与资源核对项目 | 选择算法、增加适配层 | 排序、归并、同步、对齐、容量 |
| [shape-preprocessing.md](shape-preprocessing.md) | axis 归一化、O/N/I、输入输出寻址及溢出检查 | 设计调用层预处理和布局 | dim、transpose、reshape、pitch |
| [splitting.md](splitting.md) | 序列/排序轴切分、批次与尾块、资源预算 | 推导 tiling 和并行度 | M、N、B、run、bucket、workspace |
| [liveness-fusion.md](liveness-fusion.md) | value/index/key 等存活节点与物理 buffer 复用 | 分配临时空间、裁剪与融合 | L、P、alias、ping/pong、候选 |
| [sync-model.md](sync-model.md) | RAW/WAR、阶段和跨核依赖 | 推导同步点和参与者 | inverse、merge、hist、lookback、barrier |
| [compute-flow.md](compute-flow.md) | 各排序算法与不变量 | 推导排序主体 | insertion、two-stage、merge、radix |
| [output-mapping.md](output-mapping.md) | 统一接口、名次区间与布局流程 | 生成 Sort、Argsort、TopK、KthValue | first_rank、output_length、transpose、keepdim |
| [../source-map.md](../source-map.md) | 算法、Host 参数与 Kernel 的文件和函数导航 | 定位已选语言的具体实现 | 文件、函数、选路、模板 |

## 使用规则

1. 排序能力的实现、适用域及资源限制由命中语言的 capability-mapping 提供。
2. 具体模板、平台常数、编译器约束和源码 schedule 在适配层；不能由语言中立算法推出某 API 必然存在。
