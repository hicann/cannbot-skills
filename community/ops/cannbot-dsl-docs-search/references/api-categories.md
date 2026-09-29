# CANNBotDSL API 分类说明

本参考说明 `inspect_wheel_api.py` 如何把 wheel 中提取的 API 归入职责类别。维护分类规则、评审兜底条目或解释分类产物时读取本文。

## 分类原则

分类使用 API 的真实定义模块，而不是最短公开导入路径。

例如 `cannbotdsl.mem_copy` 的公开路径位于根包，但真实定义是 `cannbotdsl.ops.memcpy.mem_copy`，因此它属于“数据搬运与格式转换”。使用定义模块可以避免根包重导出把不同领域的 API 全部压平到同一类别。

分类只描述 API 的代码职责，不证明硬件支持、算法适用性、数值语义或运行时行为。一个 API 只能进入一个主类别。

## 分类表

规则按表中顺序匹配相对于 package 根的模块前缀；第一个命中的规则生效。

| ID | 中文名称 | 模块前缀 | 收录内容与边界 |
|---|---|---|---|
| `language-and-kernel` | 语言、JIT 与 Kernel | `lang` | constexpr、控制流、JIT、kernel 装饰器、规格声明和语言级 VF 入口。 |
| `tensor-types-and-layout` | Tensor、类型与布局 | `tensor`、`types` | Tensor、dtype、layout、tiler、tensor list、延迟线和复合数据结构。 |
| `memory-and-dataflow` | Buffer、Channel 与数据流 | `buffer`、`channel` | Buffer/Channel 构造、状态、acquire/commit/wait/release 等数据流接口。 |
| `data-movement` | 数据搬运与格式转换 | `ops.memcpy` | DMA、copy engine、layout/format transform、padding、分块搬运和 `mem_copy`。 |
| `vector-register` | Vector/寄存器操作 | `ops.reg` | 寄存器算术、比较、mask、load/store、排列、规约、gather/scatter 和寄存器同步。 |
| `scalar-and-simt` | Scalar 与 SIMT | `ops.scalar`、`ops.simt` | 标量运算、ScalarPtr、标量 cast、位操作、SIMT 线程信息和数学入口。 |
| `cube-matmul-and-convolution` | Cube、Matmul 与 Convolution | `ops.cube`、`ops.matmul`、`ops.conv` | Cube 模式、矩阵乘、卷积规格、卷积 tile 及相关装载/存储接口。 |
| `synchronization` | 同步与缓存控制 | `ops.sync` | pipe、barrier、busy wait、事件、buffer 同步、DCCI/DCI 和跨核同步。 |
| `distributed` | 分布式与通信 | `ops.distributed` | 通信上下文、通信引擎、协议、通信 buffer 和设备指针。 |
| `architecture-and-platform` | 架构与平台信息 | `ops.arch`、`ops.info` | block/core/subblock 索引、系统周期、状态、平台架构和内存容量。 |
| `native-packaging-and-runtime` | Native 打包与运行时 | `package` | Native 注册、收集、构建、发布路径、运行时查找和资源观察。 |
| `other-operations` | 其他算子接口 | `ops` | 在所有专门 `ops.*` 规则之后匹配；承接新出现但尚未建立专类的算子模块。 |
| `internal-implementation` | 内部定义的顶层公开接口 | `core`、`_mlir`、`aicpu` | 真实定义位于内部模块、但由 `cannbotdsl` 顶层明确重导出的 API。 |
| `other` | 其他 API | 无 | 最终兜底，承接不属于上述前缀的定义；任何 API 都不能因无法分类而被丢弃。 |

## 匹配与优先级

- `ops.memcpy`、`ops.reg`、`ops.scalar` 等专门规则必须位于通用 `ops` 规则之前。
- 前缀按模块边界匹配：`ops.reg` 匹配 `ops.reg.arith`，但不匹配 `ops.registered`。
- `internal-implementation` 只描述定义来源；这些条目必须具有 `cannbotdsl.<名称>` 顶层公开路径，内部模块本身不会扩大产出范围。
- `other-operations` 与 `other` 是保全类别，不是错误状态。它们保证新 wheel 模块仍会进入产物。
- 方法继承其定义类所在模块的类别；重导出不会改变分类。

## 公开范围与分类的关系

默认产物只包含可通过 `cannbotdsl.<名称>` 直接访问的顶层公开函数、类、类型别名和常量，以及这些顶层公开类的公开成员。仅能从子模块访问的符号、私有符号和仅内部可访问的定义不进入产物。模块命名空间和内部模块仍参与导出链解析，但不单独生成 API 条目。可见性和类别是两个独立字段：

- `visibility` 回答 API 是否公开；
- `category` 回答 API 属于哪个代码职责领域。

底层定义名以下划线开头但被显式导出为公开别名时，按公开 API 处理；类中的私有或魔术方法默认不进入公开目录。

## 维护规则

当新 wheel 在 `other-operations` 或 `other` 中出现条目时：

1. 查看 API 的定义模块、docstring 和公开导出路径；
2. 判断它是否属于现有类别，还是形成稳定的新职责域；
3. 只有模块职责明确且可复用时才新增或扩展前缀规则；
4. 修改脚本中的 `CATEGORY_RULES`，并同步更新本文分类表；
5. 用真实 wheel 重新生成产物，确认 API 总数不减少、分类计数合理，并且 Markdown 文档与 YAML 索引数量一致。

不要依据单个 API 名称或一次性版本细节创建类别。类别应描述模块长期职责，不能把运行时验证结论、硬件支持矩阵或算子算法选择混入分类定义。
