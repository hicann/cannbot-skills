# 样例地图（example-map）

"目标场景 → GE 仓样例 → 该样例的看点"。用途：开发时（②）先按 case 的**目标片段场景**跳到最贴近的真实样例，照它的写法搭骨架；需求分析（①）用来印证分类。

**这是指针 + 注解，不复制样例源码**（保持单一事实源、防陈旧）。样例实现可全量读，但仍守正确性纪律：签名以真实文档/头文件为准（`tips/api-signature-gate.md`），op type 以 dump 为准（`tips/dump-first-op-type.md`），不照抄样例中的硬编码节点名。

> **路径按规范仓 `cann/ge` 给**（<https://gitcode.com/cann/ge>，布局为 `pattern_base_pass/`、`graph_base_pass/` 分子目录，每样例 `cpp/`+`python/`）。用前先 `find "$GE_REPO_PATH/examples/fusion_pass" -maxdepth 2 -type d` 对齐你的 checkout：个别精简 fork 用扁平布局（`examples/fusion_pass/N_xxx_pass/`）且只保留部分样例，**路径以实测为准**、缺失即按 `knowledge-base.md` 降级、不硬编码。样例完整清单见 `ge-repo-map.md` §6。

## 一、按场景选样例

| 目标场景 | 首选样例（`cann/ge` 路径，相对 `examples/fusion_pass/`） | 接口族 / 阶段 | 看点（重点抄这些写法） | 关联 tip |
|---|---|---|---|---|
| 多算子拓扑**融合为单算子**（A+B→C） | `pattern_base_pass/1_fuse_matmul_add_pass` | PatternFusionPass · `kBeforeInferShape` | ① 同一片段在**在线/离线**下 op type 不同（`MatMul` vs `BatchMatMulV2`）→ 定义**多个 pattern** 分别覆盖；② `CreateInputs<N>()` 建输入、`es::GEMM` 直接用 es_all wrapper 建 replacement；③ `CreateScalar` 造标量常量（alpha/beta） | `dump-first-op-type`、`es-all-no-version-rename` |
| **条件删除/短路**某结构（Add(x,0)→x） | `pattern_base_pass/4_add_zero_pass` | PatternFusionPass · `kBeforeInferShape` | ① pattern 里用 `es::Const(graph_builder)` 显式建常量输入；② **守卫写在 `MeetRequirements`**：读 Const 张量值、按 dtype 分派判是否为 0（`IsTensorValueEqualToZero`），不满足 return false（不在 Replacement 里返 null）；③ replacement 删掉该结构、保留穿越边 | `fragment-spec`（守卫）、`failure-attribution` |
| **拆分/展开**一个算子为子图（分组卷积→Split+Conv×n+Concat） | `pattern_base_pass/6_decompose_grouped_conv_to_splited_pass` | DecomposePass · `kAfterInferShape` | ① DecomposePass 签名是 **GNode 版**：`MeetRequirements(const GNode&)` / `Replacement(const GNode&)`，构造传 op_types；② 守卫依赖 InferShape 后真实属性（`groups!=1 && data_format=="NCHW"`）→ 所以放 `kAfterInferShape`；③ decompose 后**节点数上升是预期**（不是"越少越好"）；④ 用 `InferShape` + `CheckNodeSupportOnAicore` 校验替换图能落地 | `format-sensitive-nchw`、`infershape-util-only`、`compliant-node-builder-ir-order` |
| 需在 MeetRequirements/Replacement 里**取到匹配节点** | `pattern_base_pass/2_fuse_matmul_add_pass_with_capture_tensor` | PatternFusionPass | `Pattern::CaptureTensor()` 注册捕获项、`MatchResult::GetCapturedTensor` 取回（用途按样例 README，写码前读源码核对） | `api-signature-gate` |
| 需**约束匹配**（常量值 / IR 属性参与匹配） | `pattern_base_pass/3_fuse_matmul_add_pass_with_pattern_matcher_config` | PatternFusionPass | `PatternMatcherConfigBuilder().EnableConstValueMatch().EnableIrAttrMatch().Build()`（同上，以源码为准） | `api-signature-gate` |
| **依赖 shape** 的 pattern 融合（如 flatten） | `pattern_base_pass/7_batch_matmul_flatten_pass` | PatternFusionPass · `kAfterInferShape` | 依赖 InferShape 后真实 shape → 注册在 `kAfterInferShape`（印证"阶段由是否依赖 shape 决定"） | `interface-catalog` §一 |
| 自定义算子内的融合 | `pattern_base_pass/5_add_zero_pass_in_custom_op` | PatternFusionPass | 自定义算子 case 的 `es_custom` 构建路径（验证见统一 skill 阶段三“验证顺序与证据”第 3 步） | `es-all-no-version-rename` |
| **函数式 graph pass** / 直接改边删点 | `graph_base_pass/2_move_relu_before_concat_pass`、`graph_base_pass/3_modify_conv_data_format_pass` | 函数式 graph pass · 常 `kBeforeInferShape` | `REGISTER_CUSTOM_PASS(...).CustomPassFn(...)`；用 `Graph/GNode` 遍历改图，**不**实现 `Patterns()/Replacement()`（挪 Relu / 改 Conv data_format） | `interface-catalog` §一判别要点、`dump-log-diff-checklist` |
| **Python** pass（任意接口族） | 样例的 `python/` 子目录（当前 master 快照：14 个高层子样例中 **8 个有** Python；用前以 README/find 复核）+ `examples/fusion_pass/python_fusion_pass_development_guide.md` | 按 case | `@register_fusion_pass`/`@register_decompose_pass`、`patterns()`/`replacement()`；能力缺失时产诊断型 pass | `python-remove-node-lifecycle`、`api-signature-gate` |

> 上表前三行 + 函数式行的看点已对照样例源码；`2/3/5/7` 变体的看点按样例 README 用途给出，**写码前打开对应 `src/` 源码核对**（正确性纪律见开头）。

**子样例内部布局（别想当然，详见 `ge-repo-map.md` §6）**：多数子样例分 `cpp/` + `python/`，源码在 `<样例>/cpp/src/*.cpp` 与 `<样例>/python/src/*.py`。两个例外：`graph_base_pass/1_fuse_matmul_add_pass` 扁平（直接 `src/`，仅 C++）；`pattern_base_pass/7_batch_matmul_flatten_pass` 无 Python 版。找源码用 `find "$GE_REPO_PATH/examples/fusion_pass" -path '*/src/*.cpp'` 最稳。

## 二、共性看点（三个 C++ 样例都体现）

- **日志**：官方样例统一用 `std::cout << "..." << std::endl;` 打关键分支（"Define pattern for X"、"Define replacement for X"、"Define MeetRequirements for X"），直接落 stdout。开发同样用 `std::cout`/`print`（见统一 skill 的阶段二“实现规则”、`tips/cpp-style-naming.md`）。
- **命名/风格**：snake_case 局部变量、PascalCase 类名与重写方法、snake_case 文件名——与参考项目的 camelCase 不同，**以 GE 仓样例为准**（`tips/cpp-style-naming.md`）。
- **头文件/命名空间**：`#include "es_all_ops.h"` + `ge/fusion/pass/<基类>.h`；样例用 `using namespace ge; using namespace fusion;`（产品化 `.so` 的命名空间纪律另见 `tips/ge-fusion-namespace.md`）。
- **注册**：`REG_FUSION_PASS(PassName).Stage(CustomPassStage::k...)` / `REG_DECOMPOSE_PASS(PassName, {"OpType"}).Stage(...)`——阶段由"是否依赖 InferShape 后 shape"决定（`interface-catalog.md`）。

## 三、怎么把样例映射到本 case

1. 用需求分析文档 §3 的**目标片段**（目标结构 + 守卫 + 可选算子 + 目标形态）定位到上表某一行的场景。
2. 打开首选样例的源码（多为 `<样例>/cpp/src/*.cpp`）+ 同级 `README.md`，照它的骨架、日志点、注册阶段写；**匹配对象换成本 case dump 出的真实 op type**，不沿用样例的节点名。
3. 样例缺失/GE 版本不含该样例时，退回 `ge-repo-map.md` 的接口→文档路径，据实标注"无对应样例，按开发指南实现"。
