# 接口清单（interface-catalog）

pass 接口 + 构图接口速查，每条带**签名 / 所属命名空间 / 注意点或指向 tip**。用途：① 需求分析填文档 §6"接口清单"；② 开发做 API 签名核对（核对表门禁见 `tips/api-signature-gate.md`）。

> 本文档给出常用接口形态作为速查参考；实际函数名/类名/模块路径以本地 GE 仓 `$GE_REPO_PATH/examples/fusion_pass/...` 的 README/开发指南与样例实现源码为准。签名落定前仍需过 `api-signature-gate.md`。
>
> **每个接口的 API 文档确切路径见 `ge-repo-map.md` §1 任务路由表 / §3 API 表**（例如 C++ `docs/zh/api/graph_engine_api/cpp/ge/fusion/PatternFusionPass/PatternFusionPass.md`）；GE 仓线上 <https://gitcode.com/cann/ge>。枚举与签名的最终真值以头文件为准：`inc/graph_metadef/external/register/register_custom_pass.h`、`inc/external/ge/fusion/pass/fusion_pass_reg.h`。

---

## 一、接口映射表（判类型用）

| 类型 | 关键信号 | 注册接口 | 阶段 |
|---|---|---|---|
| C++ 函数式 graph pass | `REGISTER_CUSTOM_PASS(name).CustomPassFn(...)` | `REGISTER_CUSTOM_PASS("name").CustomPassFn(fn).Stage(...)` | 常 `kBeforeInferShape` |
| C++ FusionBasePass | 继承 `FusionBasePass`，重写 `Run` | `REG_FUSION_PASS(PassName).Stage(...)` | 按需（见下） |
| C++ PatternFusionPass | 继承 `PatternFusionPass`，重写 `Patterns()`/`Replacement()` | `REG_FUSION_PASS(PassName).Stage(...)` | 按需（见下） |
| C++ DecomposePass | 继承 `DecomposePass`，构造传入 op_types | `REG_DECOMPOSE_PASS(PassName, {"Conv2D"}).Stage(...)` | 常 `kAfterInferShape` |
| Python FusionBasePass | `ge.passes.FusionBasePass`、`@register_fusion_pass`、`run(graph, context)` | `@register_fusion_pass(name=..., stage=PassStage.BEFORE_INFER_SHAPE)` | 按需 |
| Python PatternFusionPass | `ge.passes.PatternFusionPass`、`patterns()`、`replacement()` | `@register_fusion_pass(...)` | 按需 |
| Python DecomposePass | `ge.passes.DecomposePass`、`@register_decompose_pass(op_types=[...])` | `@register_decompose_pass(op_types=[...], stage=PassStage.AFTER_INFER_SHAPE)` | 常 `AFTER_INFER_SHAPE` |

判别要点：
- 对外接口出现 `REGISTER_CUSTOM_PASS(...).CustomPassFn(...)` → 一律 C++ 函数式 graph pass，不因"任务叫融合 pass / 资料在 fusion_pass/ 下 / 文案含 pattern"改去实现 `PatternFusionPass`。
- `graph_base_pass` 不是一个具体 C++ 基类，也不是"裸 Graph API 改边删点"的同义词——继续用对外接口区分函数式 / FusionBasePass。
- 函数式 graph pass 用 `Graph/GNode` 遍历、改边、删点；不实现 `Patterns()`/`Replacement()`/`MeetRequirements()`。
- **三类 C++ 注册宏都支持链式 `.Stage(CustomPassStage::k...)`**（`REGISTER_CUSTOM_PASS` 亦有 `Stage()`，见 `register_custom_pass.h`；GE 样例省略它，走默认值）。`REG_FUSION_PASS` 只传 pass 类名，`REG_DECOMPOSE_PASS` 第二参才是 op 类型列表 `{"Conv2D"}`。
- **阶段不由接口类型固定，由"是否依赖 InferShape 后的真实 shape"决定**：依赖 shape/通道/groups 的（典型 DecomposePass、以及 GE 仓样例 7 `BatchMatmulFlattenPass` 这种 PatternFusionPass）用 `kAfterInferShape`；只按 op type/拓扑匹配的用 `kBeforeInferShape`。

### 一.1 执行阶段枚举（真值：`register_custom_pass.h` 的 `CustomPassStage`）

| C++ 枚举 | 值 | Python `PassStage` | 可注册普通融合 pass？ | 可用起始版本 |
|---|---|---|---|---|
| `kBeforeInferShape` | 0 | `BEFORE_INFER_SHAPE` | ✅ **默认值，首选** | 8.5.0 |
| `kAfterInferShape` | 1 | `AFTER_INFER_SHAPE` | ✅ replacement 需自行保证 shape 连续性 | 8.5.0 |
| `kAfterAssignLogicStream` | 2 | —（Python 无） | ❌ **仅 `CustomAllocateStreamPassFunc`**；误注册普通 pass 会被忽略 / 校验报错 | 8.5.0 |
| `kAfterBuiltinFusionPass` | 3 | `AFTER_BUILTIN_FUSION_PASS` | ✅ 内置融合之后 | 8.5.0 |
| `kAfterOriginGraphOptimize` | 4 | `AFTER_ORIGIN_GRAPH_OPTIMIZE` | ✅ 原图优化之后 | **9.0.0** |
| `kCompatibleInherited` | 5 | —（Python 无） | 兼容继承语义，常规开发不用 | **9.0.0** |
| `kInvalid` | 6 | — | 哨兵，不可用 | — |

- **Python 只暴露 4 个阶段**（无 `AFTER_ASSIGN_LOGIC_STREAM` / `COMPATIBLE_INHERITED`）。
- `kAfterOriginGraphOptimize` 需 CANN ≥ 9.0.0；目标环境版本低于此即不可选，文档 §6 须标注版本门槛。
- 不确定就选默认的 `kBeforeInferShape`——replacement 之后仍会进入 GE 统一 shape 推导。

### 一.2 Pattern 可表达性约束（决定 `PatternFusionPass` 能不能用）

`Pattern` 只表达**数据拓扑**。目标片段含以下任一内容时，`PatternFusionPass` 不可用，须退回 `graph_base_pass`（函数式 / `FusionBasePass`）：

| 不支持的 pattern 内容 | 原因 |
|---|---|
| 控制边 | matcher 不按控制依赖匹配 |
| 子图（嵌套） | 不支持嵌套子图匹配 |
| 动态输入个数 / 动态输出个数的节点 | 匹配时无法确定固定的输入输出边界 |

另有三条必须在需求阶段就确认的边界规则（`fusion_pattern_pass.md` §3）：

- **输入边界**：凡来自子图外部的 Tensor 都要用输入占位符表示。
- **输出边界**：凡替换后仍被子图外部使用的 Tensor 都必须声明为 pattern 输出，否则替换后断图。
- **输入个数要精确**：普通算子节点的输入个数必须与真实图一致，不关心来源的也要用占位符补齐。
- **自包含**：pattern 内部节点若某输出未声明为 pattern 输出，则该输出的消费者必须全在 pattern 内。

**多拓扑 ≠ 多输出**：同时支持 `MatMul+Add` 与 `BatchMatMulV2+Add` 要定义**多个 pattern**；多输出 pattern 指的是"一次匹配暴露多个输出 Tensor"。

### 一.3 V1 / V2 基类：钩子能不能拿到 `CustomPassContext`

`PatternFusionPassV2` / `DecomposePassV2` 是**独立基类**（都继承 `FusionBasePass`），不是同名类的重载。它们把 `CustomPassContext &` 透传进 `MeetRequirements` / `Replacement`，`@since 9.1.0(2026-05)`。**与 V1 共用 `REG_FUSION_PASS` / `REG_DECOMPOSE_PASS` 注册**（工厂返回 `FusionBasePass*`，对版本无感知）。

| 接口 | 用户钩子能拿到 `CustomPassContext`？ | 能否在钩子里读 option / 写 error msg |
|---|---|---|
| C++ 函数式 graph pass（`CustomPassFn`） | ✅ 回调签名即 `(GraphPtr &, CustomPassContext &)` | ✅ |
| C++ `FusionBasePass::Run` | ✅ `Run(GraphPtr &, CustomPassContext &)` | ✅ |
| C++ `PatternFusionPass`（V1） | ❌ `MeetRequirements(match_result)` / `Replacement(match_result)` **无 context** | ❌ |
| C++ `DecomposePass`（V1） | ❌ `MeetRequirements(node)` / `Replacement(node)` **无 context** | ❌ |
| C++ `PatternFusionPassV2` | ✅ 钩子多一个 `CustomPassContext &` 参数 | ✅（**需 CANN ≥ 9.1.0**） |
| C++ `DecomposePassV2` | ✅ 同上 | ✅（**需 CANN ≥ 9.1.0**） |
| Python `FusionBasePass.run(graph, context)` | ✅ | ✅ |
| Python `PatternFusionPass` / `DecomposePass` 钩子 | ❌ 钩子只收 `match_result` / `node` | ❌ |

> **推论（写进文档 §8）**：要在 pattern / decompose 类 pass 的**钩子内**按 option 开关决定是否替换，V1 做不到——要么升 V2（CANN ≥ 9.1.0），要么把开关判断挪到别处（如改用 `FusionBasePass` 自己扫图）。V2 的 `Replacement` 还可以"写入 error msg 后返回 `nullptr` 终止替换"。

### 一.4 匹配严格度：`PatternMatcherConfig` vs `MeetRequirements`

| 判断 | 放哪里 |
|---|---|
| Const 值 / IR 属性**严格相等**，逻辑简单稳定 | `PatternMatcherConfig`（`EnableConstValueMatch` / `EnableIrAttrMatch`） |
| 需要浮点容差、dtype 归一化、多条件组合 | `MeetRequirements`（更清晰，也更可控） |

> Const 值匹配是**严格匹配**：不做浮点容差，不做跨 dtype 归一化。

---

## 二、pass 接口

- **C++ 函数式**：回调类型 `using CustomPassFunc = std::function<Status(ge::GraphPtr &, CustomPassContext &)>`；GE 样例写成 `graphStatus FuseMatMulAndAddPass(GraphPtr &graph, CustomPassContext &ctx)`，成功返回 `GRAPH_SUCCESS`（**无需 `extern "C"`**）。注册：`REGISTER_CUSTOM_PASS("name").CustomPassFn(fn)`；可再链 `.Stage(CustomPassStage::kBeforeInferShape)`，不写则取默认阶段。头文件 `register/register_custom_pass.h`，库 `libregister.so`。
- **C++ FusionBasePass**：`Status Run(GraphPtr &, CustomPassContext &) override`。拓扑替换优先 `Graph/GNode` 扫图 + ES API 构造 replacement graph + `SubgraphBoundary` + `SubgraphRewriter::Replace`；属性/边局部修改或有证据的 fallback 才用 `AddDataEdge`/`RemoveEdge`/`RemoveNode`/`AddNodeByOp`。命名空间见 `tips/ge-fusion-namespace.md`。
- **C++ PatternFusionPass**：继承 `ge::fusion::PatternFusionPass`；实现 `std::vector<PatternUniqPtr> Patterns() override`、`GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &) override`；可选 `bool MeetRequirements(const std::unique_ptr<MatchResult> &) override`（**V1 钩子无 context**，见 §一.3）。`Pattern::CaptureTensor()` 注册捕获项；`PatternMatcherConfigBuilder().EnableConstValueMatch().EnableIrAttrMatch().Build()` 控制 matcher。注册：`REG_FUSION_PASS(PassName).Stage(CustomPassStage::k{Before|After}InferShape)`（阶段按是否依赖 shape 选）。
- **C++ DecomposePass**：继承 `ge::fusion::DecomposePass`；构造 `DecomposePass(const std::vector<AscendString> &op_types)`；用户重写 `Replacement`（必选）、`MeetRequirements`（可选），基类框架入口是 `Run`（见 `docs/zh/api/graph_engine_api/cpp/ge/DecomposePass/DecomposePass.md` → `Run.md`）。**V1 钩子无 context**，要在钩子里读 option 须用 `DecomposePassV2`（见 §一.3）。注册：`REG_DECOMPOSE_PASS(PassName, {"Conv2D"}).Stage(CustomPassStage::kAfterInferShape)`（阶段用 `.Stage()` 显式指定，op 类型是宏第二参的字面量列表）。
- **C++ shape 推导**：只用 `ge::fusion::InferShapeUtil::InferShape`（见 `tips/infershape-util-only.md`）。
- **Python FusionBasePass**：`run(self, graph: Graph, context: PassContext) -> StatusLike`。
  ```python
  from ge.passes import FusionBasePass, register_fusion_pass, PassStage
  @register_fusion_pass(name="MyPass", stage=PassStage.BEFORE_INFER_SHAPE)
  class MyPass(FusionBasePass):
      def run(self, graph, context):
          ...
          return True   # 成功；不改图时同样返回 True（或 None）
  ```
  > ⚠️ **返回值语义与 C++ 相反，不要写反**：Python `run()` 返回 `None` 或**真值**表示**成功**；返回**假值**（`False` 或 **`0`**）表示**失败**。而 C++ 的成功码 `SUCCESS` 整数值是 0（`inc/external/ge/ge_api_error_codes.h`）——把 C++ 习惯照搬成 `return 0` 会让 GE 认为 pass 执行**失败**。
  >
  > `PassContext` 方法：`get_pass_name()` / `set_pass_name(name)` / `get_error_message()` / `set_error_message(msg)` / `get_option_value(key) -> str`（key 非法或底层非 `GRAPH_SUCCESS` 时抛 `RuntimeError`）。仅可在当前 `run` 调用栈内使用，不要存到 `self`。

  若需新建节点/子图替换，先本地 API probe 确认 `ge.passes.SubgraphBoundary`/`SubgraphInput`/`SubgraphOutput`/`SubgraphRewriter` 及所需 ES Python API 存在；齐全则实现真实替换，缺失则生成诊断型 pass（注册成功、扫描目标结构、打印缺失 API 与 skip reason、**返回 `True`**），不虚构 replacement 成功。
- **Python PatternFusionPass**：引擎调 `patterns()`/`meet_requirements()`/`replacement()`，**不调 `run()`**。
  - **不得重写 `run()`**——子类定义了 `run()` 会在**类定义时**抛 `TypeError`。`DecomposePass` 同此约束。
  - 必须实现 `patterns()` **或**至少一个 `@pattern` 方法，且必须实现 `replacement()`；`meet_requirements()` 可选（默认 `True`）。
  - **`@pattern` 与 `patterns()` 不能同时使用**；不支持 `patterns(self, inputs)` 写法。
  - `@pattern`（推荐，表达式写法）：一个方法 = 一个 pattern，多拓扑写多个 `@pattern` 方法。`inputs[i]` 按需创建第 i 个输入，`inputs[:N]` 显式声明连续多输入，**不能直接迭代 `inputs`**（输入个数非预先固定）。返回 `TensorHolder` 为单输出，返回 list/tuple 为**多输出 pattern**（不是多个 pattern）。
  - **`@pattern` 只自动 capture "已访问的外部输入" 和 "`return` 的 pattern 输出"**（顺序：先按输入序号，再按 return 结构顺序）。**未作为输出返回的中间 Tensor 不会被自动 capture**——若 `meet_requirements`/`replacement` 要读中间 Tensor（如 `MatMul` 的输出），必须放弃 `@pattern`，改用显式构图 + `create_pattern(...)` + `pattern.capture_tensor(...)`。
  - `replacement` 三种形态：`replacement(self, match_result) -> Graph`（显式构图）、`replacement(self, inputs) -> TensorHolder`、`replacement(self, inputs, match_result) -> TensorHolder`（需读命中节点属性时用）。
  - `PatternMatcherConfigBuilder().enable_const_value_match().enable_ir_attr_match().build()` 传给 `super().__init__(...)`。
- **Python DecomposePass**：`register_decompose_pass` 是**关键字参数**，`name` / `stage` / `op_types` 三者**均必填**（`op_types` 不可为空，元素须为非空字符串）。
  ```python
  @register_decompose_pass(name="MyDecompose", stage=PassStage.AFTER_INFER_SHAPE, op_types=["Conv2D"])
  class MyDecompose(DecomposePass):
      def meet_requirements(self, node) -> bool: ...   # 可选，默认 True
      def replacement(self, node) -> Graph: ...        # 必选
  ```
  `register_fusion_pass(*, name, stage, kind=None)` 同为关键字参数；`kind` 不填时按基类自动推断（`PatternFusionPass` 子类 → `"pattern_fusion"`，否则 `"fusion_base"`）。**`name` 必须全局唯一。**

---

## 三、构图接口

- **图遍历**：`Graph::GetAllNodes`；`GNode::GetName/GetType/GetAttr/GetInputsSize/GetOutputsSize/GetOutDataNodesAndPortIndexs`。
- **ES 构图**：`EsGraphBuilder::CreateInput/BuildAndReset`；ES 算子 wrapper（如 `ge::es::GEMM`、`es::Conv2D`、`es::SplitD`、`es::Const`）——直接用 es_all 暴露的 wrapper，不预换版本名（`tips/es-all-no-version-rename.md`）；`EsTensorHolder::SetFormat`（`tips/format-sensitive-nchw.md`）。
- **子图替换**：`SubgraphInput`/`SubgraphOutput`/`SubgraphBoundary`；`SubgraphRewriter::Replace`。
- **显式手建已注册节点**：`ge::es::CompliantNodeBuilder`（`OpType`/`IrDefInputsV2`/`IrDefOutputsV2`/`IrDefAttrsV2`）+ `ge::es::AddEdgeAndUpdatePeerDesc`——IR 顺序严格按 op_proto `REG_OP`（`tips/compliant-node-builder-ir-order.md`）；仅在 es_all 未暴露所需 wrapper 时走。
- **shape 推导**：`ge::fusion::InferShapeUtil::InferShape`。
- **Python 对应物**：`ge.passes`（`FusionBasePass`/`PatternFusionPass`/`DecomposePass`/`register_fusion_pass`/`register_decompose_pass`/`PassStage`/`SubgraphRewriter` 等）、`ge.graph`、`ge.es`——以本地 probe 结果为准。
