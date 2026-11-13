# Convolution 范式入口（语言中立）

> 本文件是 Convolution 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的术语、数据流、切分、验证与场景扩展
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的落地知识，提供example示例
>
> 范式路由（`op.paradigms` 单范式标签 → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。
>
> **进入路径（范式标签）**：算子生成工作流（ops-spec-gen 范式标签）对 conv1d/conv2d/conv3d 的打标形态为 `SlidingWindow` + `Contraction` (+ `Padding`)，无单一 `Convolution` 标签；范式路由表 `routes.yaml` 按 `SlidingWindow` 标签命中本文件并注入范式参考，示例路由见 `example-routes.yaml`。经 `SlidingWindow` 标签族到达时按下方「场景判定」甄别：池化（`SlidingWindow` + `Reduction`，无 `Contraction`，max/avg/sum 聚合）不在本范式范围，按范式索引走 `Reduction` 与通用方法论；纯 matmul（仅 `Contraction`）不命中；滑窗注意力按 FA 组合形态另行路由，与本范式同时命中时以公式核对为准（见下方「场景判定」）。
>
> **内容原则**：本范式自含创建卷积算子所需的**原理、方法与流程**，按本范式即可从零构建卷积算子。

---

## 场景判定

满足以下条件的算子属于 Convolution 范式：

1. **滑窗乘加计算语义**——输出元素由输入在多维窗口内的元素与卷积核的乘累加得到：`y(n, co) = bias(co) + Σ_ci filter(co, ci) ⋆ x(n, ci)`，`⋆` 为沿 D/H/W 空间维的滑窗互相关
2. **存在 im2col 意义下的 M/N/K 三元组**——M = 展平后的输出空间维（Ho×Wo 或按行窗口化），N = 输出通道 Co，K = Ci×Kd×Kh×Kw；计算可映射为矩阵乘
3. **计算主体是 Cube 乘加**——数据通路围绕 L1/L0A/L0B/L0C 层级组织。**本范式仅覆盖纯 Cube 与 Cube+Vector 协同的实现方案；纯 Vector 滑窗（UB 内软件滑窗乘加、无 Mmad）与 Winograd 变体不在本范式范围内**

典型场景：Conv1D/Conv2D/Conv3D 前向、空洞/带步长卷积、量化反量化卷积（conv + scale·out + bias）。分组卷积、pad≥kernel 等变体的差异处理见 [generic/scene.md](generic/scene.md)。

**经 `SlidingWindow` 标签族到达时的判定**：池化（`SlidingWindow` + `Reduction`，无 `Contraction`，max/avg/sum 聚合而非核权重乘累加）不满足条件 1/2，不在本范式范围，按范式索引走 `Reduction` 与通用方法论；滑窗注意力（`Contraction` + `Reduction` + `SlidingWindow`，含 softmax 分数归一化与加权值聚合）走 FA 组合范式，不进入本文件。
**标准路径与扩展**：通用知识层以**标准路径**（fp16、NCHW/NCDHW/NCW、groups=1、pad<kernel）为主干展开；超出标准路径的维度（例如dtype/format/groups/大核/小通道等）按 [generic/scene.md](generic/scene.md) 的"与主干差异"方式补充，主干文档不重复。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层| [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 通用方法论 | `ops/ops-design-paradigms/references/general-methodology-dsl.md`（DSL 专用适配层暂缺，按通用方法论兜底，见 [readme.md](readme.md) 支持范围；语言适配层建成后再挂 `adapters/dsl/index.md`） |

**互斥规则**：一次任务只加载一个命中语言的适配层，例如命中了ascendc，就不要去加载dsl。

---

## 通用知识层索引

语言无关的知识，按任务阶段选取（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 | 典型消费时机 |
|------|------|-------------|
| [generic/language.md](generic/language.md) | 统一术语与记号：张量/维度/属性词汇、M/N/K 三元组、切分记号体系（mAL1/nBL1/Tail/fullLoad）、核心记号速查表 | 任何文档阅读前对齐记号 |
| [generic/compute-flow.md](generic/compute-flow.md) | 计算流程：卷积公式与滑窗原理、四侧数据流（fmap/weight/bias/output）、五种数据格式、全部指令（语义/参数限制/单位）、多核分核、单核迭代伪码、四级流水同步 | kernel 实现设计 |
| [generic/tiling.md](generic/tiling.md) | 切分算法：M/HW/basic block 三模式、绑核三步法、L0 先 L1 后决策、fullLoad 四态、A 侧预算分级退化与不可行判定、TilingKey 位打包 | tiling 设计 |
| [generic/verification.md](generic/verification.md) | 验证方法：三层用例设计、输入构造、golden 生成、精度比对与通过判定计算、参考代码指引、失败定位 | 用例设计与精度定位；验证脚本编写 |
| [generic/scene.md](generic/scene.md) | 场景扩展：1D/3D 维度差异、量化卷积、NHWC、groups 对角阵化、pad≥kernel、C04 小通道 | 规格超出标准路径时 |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题。
3. **能力真值**：通用层引用的平台能力（容量、指令限制、累加类型）以命中语言适配层的实际编译环境为准（ascendc 见 [adapters/ascendc/buffer-and-types.md](adapters/ascendc/buffer-and-types.md) 的版本差异警告），存在版本差异时在产出中声明假设并给出验证义务。
4. **实现依据**：Ascend C 编码前检视参考实现 [adapters/ascendc/example/conv2d/](adapters/ascendc/example/conv2d/)（tiling + kernel 主体）；`<<<>>>` 内核直调与 aclnn 注册两段式是拉起该算子的接入形态，由上层工作流技能负责（本仓 `ops/ascendc-direct-invoke-template` / `ops/ascendc-registry-invoke-template`），不在本范式范围；选型参考 `references/reference-ops/open_source_operator_table.md`。参考实现中的具体算法与参数须按当前算子规格重新核定。
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出分支、数据结构、资源预算、同步和验证结论；输出格式由调用方指定。
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准。
