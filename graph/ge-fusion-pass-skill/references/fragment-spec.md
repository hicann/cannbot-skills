# 目标片段描述 + 输入场景（大网络怎么指、可选输入怎么办）

需求分析阶段用，解决两件事：**用户的原始网络很大时，怎么方便地只描述要作用的那一小段**；以及**用户能提供的输入五花八门（可选场景）时怎么退化处理**。产出并入需求分析文档 §3（结构见 `requirements-analysis-template.md`）。

---

## 一、核心观念：融合 pass 按结构匹配，用户描述"目标结构"而非整张网

一个融合 pass 不关心整张网络有多大——它**结构化匹配**目标子图，在所有命中处施作。所以用户**不需要描述整张网络**，只需描述：

```
目标片段 = 目标结构(matches what) + 定位(which instances) + 边界(inputs/outputs) + 守卫(guard) + 可选算子(optional operands)
```

这五项就是"片段规格"，比贴整张大图有效得多，也是 pattern/decompose 类 pass 天然需要的信息。

## 二、片段规格模板（并入需求分析文档 §3）

| 字段 | 内容 | 说明 |
|---|---|---|
| **目标结构** | 要匹配的算子结构，如 `MatMul → Add`、`grouped Conv2D(groups>1)`、`AddCustom(AddCustom(x,0), y)` | 用**导入后真实 GE op type**（先 dump，见 `tips/dump-first-op-type.md`），不是前端框架名 |
| **定位（可选）** | 缩小到哪些实例：节点名/名字前缀（`layer.3/attention/...`）、出现在哪个 scope、第几处 | 大网络里同一目标结构可能命中多处；不填=所有命中处 |
| **边界** | 子图的输入 tensor、输出 tensor（名字或来源算子），替换前后必须兼容 | 决定 SubgraphBoundary / replacement 的接线 |
| **守卫** | 命中后还需满足的条件：dtype、shape、attr（如 `data_format==NCHW`、`groups>1`、通道可被 groups 整除） | 决定 MeetRequirements；不满足保持原图 |
| **可选算子** | 目标结构里可有可无的输入/算子（见第四节） | 决定 replacement 的分支 |
| **目标形态** | 替换/修改后的结构（如 → `GEMM`、→ `Split+多Conv2D+Concat`） | 决定 Replacement |

## 三、大网络里"指出片段"的几种方式（按用户成本从低到高）

用户按手头有什么选一种即可，需求 skill 据此填上表：

1. **只描述目标结构**（最省）：自然语言说"把 MatMul 后接 Add 融成 GEMM"。够 pattern 类 pass 用。用 dump 校真实 op type 后即可写匹配。
2. **目标结构 + 名字前缀/scope 定位**：大网络有层级命名时，给 `blocks.*/mlp/` 之类前缀把作用域收窄。
3. **锚点节点名 + 半径**：给 1~few 个关键节点名（从 dump/模型里拿），"这些节点及其上下游 1~2 跳"。
4. **边界 tensor 名**：给子图的输入/输出 tensor 名，框定区域。
5. **dump 片段**：指向 `PreRunBegin` dump 里的节点名/行段。
6. **最小复现子图**（成本最高、迭代最快）：用户抽出只含该片段的小脚本/小 onnx。**注意**：抽取仅用于快速迭代匹配与验证；pass 仍要能在完整网络上按结构命中，不得写死只认这个小图的节点名。

> **输入只给模型/dump 时**：用 `python3 "$SKILL_ROOT/scripts/adapt_input.py" inventory` 产出 `normalized-graph.json`（schema 见 `normalized-graph-schema.md`），从统一图表达里提取目标片段的节点/边/边界，填入上方片段规格。pbtxt/dump 因可能缺权重属性，提取结果标 `structural`，详见 `input-adaptation.md`。

> 无论哪种方式，写匹配前都要对**真实模型**跑一次 dump 取导入后 op type（`tips/dump-first-op-type.md`）——大网络尤其不能靠前端脚本名硬猜。

## 四、可选输入场景（输入完整度不一时如何退化）

需求 skill 接受一个"输入完整度谱系"，缺什么标什么，需求分析文档照常产出（配合门禁 G1 与文档降级规则）：

| 用户给了什么 | 怎么处理 | 文档标注 |
|---|---|---|
| 完整输入包（`data/`+`CMakeLists.txt`+参考路径+交付物） | 信息最全，正常填全表 | — |
| 仅自然语言目标结构描述 | 产出片段规格，缺的定位/边界按"所有命中处/由匹配推断"填 | 标注哪些是推断 |
| 有模型文件（onnx/pb/air）但无目标结构描述 | 先 dump，协助从 dump 里辨识候选目标结构，再和用户确认 | 标注"目标结构据 dump 推断，待确认" |
| 仅节点名 / dump 片段 | 据此定位片段、反推目标结构与边界 | 标注来源 |
| 什么都没有，只有"把 X 融成 Y" | 只在**真正缺关键信息**（对外接口/目标结构二选一都无法确定）时才发一个最小澄清问题；否则按最可能目标结构给最佳努力需求分析文档并标注假设 | 标注假设 + 待确认 |

### 目标结构内的"可选算子/输入"（optional operands）

典型：`Conv2D` / `MatMul` / `BatchMatMulV2` 的 `bias`、`offset_w`（IR 里声明为 `OPTIONAL_INPUT`）。

> **不要把新建常量当可选输入**：GE 样例 1 的 replacement 里 `alpha`/`beta` 是 `CreateScalar(1)` **新建**的常量输入，不是被匹配节点上可有可无的输入。两者是不同的问题。

#### 关键事实：可选输入在**匹配阶段**就分叉，不是 replacement 里一个 if 能吸收的

融合机制文档 §3.4 规定：**"普通算子节点的输入个数需要和真实图一致"**。而 `GNode::GetInputsSize()` 的定义是"返回节点的**有效输入个数**，即算子的**实际输入个数**"。

两条合起来：**可选输入"存在"与"不存在"，对应两种不同的输入个数，也就是两个不同的 pattern。** 一个 pattern 匹配不了两种形态。只写 replacement 分支而不管 pattern，另一种形态会**静默不命中**——开发者往往误以为"两种都处理了"。

#### 图上有三种形态，不是两种

`inc/graph_metadef/external/graph/named_io_node_builder.h` 对 `OPTIONAL_INPUT` 列了三种表达（以 `INPUT(x), INPUT(w), OPTIONAL_INPUT(bias)` 为例）：

| 形态 | 构图写法 | 图上表现 |
|---|---|---|
| 不传递 | 跳过 `bias`，只加必选输入 | 该输入端口不存在 |
| 传递有效值 | 按 IR 顺序在对应位置 `AddInput("bias")` | 正常输入端口 |
| 传递占位 | `AddInput("bias", TensorDesc(Shape(), FORMAT_RESERVED, DT_UNDEFINED))` | 端口在，但内容无效 |

> IR 里有多个可选输入而只想用靠后那个（要 `offset_w` 不要 `bias`）时，**必须**用占位形式把 `bias` 占住。此时输入个数"看起来有"，但那个输入是无效的——判存在性不能只看个数，还要看 `TensorDesc` 是否为 `DT_UNDEFINED`/`FORMAT_RESERVED`。

#### 三种策略（按成本排序，需求阶段就要选定并写进文档 §7）

| 策略 | 做法 | 何时用 |
|---|---|---|
| **S1 只支持一种形态 + 显式拒绝其余**（推荐起步） | pattern 按必选输入个数建；`MeetRequirements` 里 `GetInputsSize() != N` 就打一条 skip 日志返回 false | 绝大多数场景。范围清晰，"不支持"变成可见日志而非静默不命中 |
| **S2 每种要支持的形态各建一个 pattern** | 与"在线 `MatMul` / 离线 `BatchMatMulV2` 各建一个 pattern"同一手法 | 确需同时支持多形态。代价：N 个可选输入最多 2^N 种形态，每个 pattern 还要配一套 replacement 分支 |
| **S3 退回 `graph_base_pass` 手动改图** | 用 `Graph/GNode` 遍历，自己判输入个数与有效性 | 组合过多，或结构里还混了动态输入个数的算子（那样 `PatternFusionPass` 本就不可用，见 `interface-catalog.md` §一.2） |

**GE 官方唯一专门处理可选输入的样例是 `pattern_base_pass/7_batch_matmul_flatten_pass`（仅 C++ 版），它采用 S1**：README 明写"无 bias/offset_w …… 带 bias、offset_w 的不在本样例优化范围内"，代码里 pattern 用 `es::BatchMatMulV2(input_a, input_b)` 两个输入建，再在 `MeetRequirements` 里 `if (bmm_node.GetInputsSize() != kSupportedInputNum) { skip }`。

#### replacement 侧（选定形态之后才轮到这一步）

- 可选输入**存在时**才连边（`tips/compliant-node-builder-ir-order.md` 骨架里 "bias→2、offset_w→3 仅在非空时连"）；**缺失时**按算子语义补默认或走无 bias 分支。
- 守卫里写清"支持哪些形态、其余形态如何拒绝并打日志"，避免 replacement 在可选输入缺失的实例上断边或建非法图。

> ⚠️ **尚待真实环境验证**：matcher 是否已按输入个数自行拒绝不匹配的形态，文档没写清（样例 7 显式复核了 `GetInputsSize()`，样例 1 的 `es::MatMul(a0,b0)` 却没有）。**工程上照样例 7 做**：那行检查成本极低；将该假设及确认方法写入本 case 的 `requirements-analysis.md` §10。

## 五、输出

把"片段规格"（第二节表）+"输入场景标注"（第四节）并入需求分析文档 §3，传给开发阶段。开发阶段据目标结构/边界/守卫/可选算子写匹配与 replacement，据定位决定是否收窄作用域。
