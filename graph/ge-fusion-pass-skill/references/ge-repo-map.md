# GE 仓目录地图 / 文档索引

GE 资料**已全量开源在 GE 仓**（不再依赖单独的 ge-document 仓）。本册是**唯一的文档索引**：给出确切**相对路径 + 讲什么 + 直接跳哪一节**，让你**一跳到位**，不用逐层翻目录。

线上仓库 <https://gitcode.com/cann/ge>。所有路径均**相对仓根 `$GE_REPO_PATH`**，本册不写死任何绝对路径。

## 0. 先解析仓根（动态确认，禁硬编码）

```bash
if [[ -n "${SKILL_ROOT:-}" && -f "$SKILL_ROOT/scripts/sync_ge_repo.sh" ]]; then
  GE_REPO_PATH="$(bash "$SKILL_ROOT/scripts/sync_ge_repo.sh")" \
    || echo "GE 仓不可达 → 按门禁 G3 如实标注证据缺失，不猜路径"
else
  echo "skill 根未显式提供 → GE 仓同步未运行，按门禁降级"
fi
```

脚本行为：`$GE_REPO_PATH` 已设且有效 → **永远只读复用**（不 fetch/reset，尊重你的本地改动）；否则**默认直接用既有缓存、不刷新**；仅 `--allow-network`（需 skill 先取得用户授权）才刷新 / 克隆缓存到 master 最新。详见 `knowledge-base.md` §二。

> - 中文文档在 `docs/zh/`。**目录名是 `user_guides`（下划线），不是 `user-guides`。**
> - 英文镜像 `docs/en/` 只覆盖 `design/`、`user_guides/`、`contributions/`——**没有 `docs/en/api/`，API 文档只有中文**。`*_en.md` 仅见于少数 README（`docs/README_en.md`、样例目录下的 `README_en.md`）。
> - **实操开发指南不在 `docs/zh/` 下，在 `examples/fusion_pass/`**（见 §6）。`docs/zh/design/architecture.md` 的插件扩展一节也指向那里。

---

## 1. 任务路由表（先查这里，直达文件 + 章节）

| 我要做什么 | 打开 | 直接跳 |
|---|---|---|
| 搞懂融合 pass 到底怎么跑 | `docs/zh/design/features/fusion_pattern_pass.md` | §2 执行链路 |
| **写 pattern 前避坑**（最高频） | 同上 | **§3 边界规则**（3.1 输入边界 / 3.2 输出边界 / 3.3 自包含约束 / 3.4 输入个数要精确 / 3.5 不支持的 pattern 内容 / 3.6 多输出 pattern） |
| 动手前自检 | 同上 | §8 开发前检查清单 |
| 跨 pattern 传张量 / 放宽匹配 | 同上 | §4.1 `CaptureTensor`、§4.2 `PatternMatcherConfig` |
| 选 `kBeforeInferShape` 还是 `kAfterInferShape` | 同上 | §6 Pass 执行阶段 |
| 该用 Python 还是 C++ | 同上 | §7 Python 与 C++ 的关系 |
| 拆算子（DecomposePass）定位 | 同上 | §5 |
| **写 C++ pass（实操）** | `examples/fusion_pass/cpp_fusion_pass_development_guide.md` | 全文 |
| **写 Python pass（实操）** | `examples/fusion_pass/python_fusion_pass_development_guide.md` | 全文 |
| C++ pass 的 `.so` 怎么被 GE 装载 | `docs/zh/design/modules/ge_python/ge_python_pass_design.md` | §1 背景（`opp/vendors/*/custom_fusion_passes/*.so` 被 dlopen） |
| Python pass 插件放哪、怎么被发现 | `docs/zh/user_guides/ge_python/env/ASCEND_GE_PY_PASS_PATH.md` | 全文（82 行） |
| Python pass API 速查 | `docs/zh/user_guides/ge_python/api/Passes.md` | 按类名检索 |
| pass 挂在编译流水线哪一段 | `docs/zh/design/modules/compiler/compiler.md` | §2 Pass 体系、§3 融合优化 |
| 用 ES 构 replacement 图（C++） | `docs/zh/user_guides/es_graph/api/es_cpp.md` | 全文（155 行） |
| 用 ES 构 replacement 图（Python） | `docs/zh/user_guides/es_graph/api/es_python.md` | 全文（308 行） |
| 生成 `es_all` / `es_custom` | `docs/zh/user_guides/es_graph/tools/gen_esb.md` | 命令示例段 |
| 核对某个 API 签名 | `docs/zh/api/graph_engine_api/README.md` 后在分层目录检索 | C++ 优先 `cpp/ge/**`，Python 优先 `python/ge/**`；不要假设扁平 `<API名>.md` |
| 查 GE options / 开关 | `docs/zh/api/graph_engine_api/cpp/ge/options_params/options_parameters_description.md` | 按 key 检索；路径变动时从 API README 的 options 树定位 |
| `MeetRequirements` 里判 format | `docs/zh/user_guides/understanding_ge_format_from_semantics_to_api.md` | Origin vs Storage 两套表示 |
| 理解 pass 里操作的数据结构 | `docs/zh/design/modules/graph_metadef/ascend-ir.md` | 锚点系统、`ComputeGraph`、`OpDesc` |
| 挑一个最像的样例照着写 | `references/example-map.md` → `examples/fusion_pass/` | 见 §6 |

---

## 2. 融合 pass 核心 → ① / ②

### 2.1 机制（先读这篇）

**`docs/zh/design/features/fusion_pattern_pass.md`**（294 行）—— 完整讲清 `PatternFusionPass` 执行链路：**定义 pattern → 在目标图中匹配 → 用 `MeetRequirements` 过滤 → 用 `Replacement` 生成替换图 → 替换和重连**。章节导航见 §1 路由表。**§3 边界规则是写 pass 最容易踩坑的地方，动手前必读。**

### 2.2 实操开发指南（在 `examples/` 下，不在 `docs/` 下）

- **`examples/fusion_pass/README.md`** —— 总入口，给出推荐阅读顺序。
- **`examples/fusion_pass/cpp_fusion_pass_development_guide.md`** —— 编 `.so` 产品化交付。
- **`examples/fusion_pass/python_fusion_pass_development_guide.md`** —— `@pattern` 表达式写法、运行时接入。

### 2.3 Python pass 路线

| 文件 | 行数 | 讲什么 |
|---|---|---|
| `docs/zh/user_guides/ge_python/api/Passes.md` | 883 | **API 手册**：`PassStage` 枚举、`PassContext`、`FusionBasePass`/`PatternFusionPass`/`DecomposePass` 三基类、`PatternMatcherConfig` 与 Builder、`@pattern` 装饰器、`@register_fusion_pass`/`@register_decompose_pass`、`create_pattern`、`Pattern` 与 `MatchResult` |
| `docs/zh/user_guides/ge_python/env/ASCEND_GE_PY_PASS_PATH.md` | 82 | Python pass 插件**路径发现环境变量**：取值格式（单文件 / 目录 / 冒号分隔多路径）、扫描规则、约束 |
| `docs/zh/design/modules/ge_python/ge_python_pass_design.md` | 2134 | **最详尽的设计文档**。§1 背景写明 **C++ 自定义 pass 的装载链路——GE 通过 `opp/vendors/*/custom_fusion_passes/*.so` 发现并 dlopen**；§4 总体架构、§5 运行链路、§6 Python 公共接口、§7 发现机制、**§8 三类 pass 的桥接设计（670 行，最厚）**、§9 Python 图接口补齐、§10 打包与发布、§11 ATC 扩展、§12 文件级开发计划 |

### 2.4 pass 在编译流水线的位置

**`docs/zh/design/modules/compiler/compiler.md`**（607 行）—— AscendIR → OM 的完整变换链路。对写 pass 最有用的两节：

- **§2 图级优化：Pass 体系** —— `GraphPass` 整图粒度、经 `PassManager` 顺序执行；`NodePass` 节点粒度、经 `GEPass` 遍历并**支持重遍历**。
- **§3 融合优化** —— 手写 Pattern 融合走 `compiler/graph/fusion/`；自动融合走 `compiler/graph/optimize/autofuse/`。

---

## 3. API 参考 → ①（填接口清单）/ ②（签名核对）/ ③

**`docs/zh/api/graph_engine_api/`** 是自动生成的分层 API 树，索引页是 **`README.md`**。目录布局会随 GE master 演进，不能把 API 名直接拼成根目录下的文件名；先从 README 找到语言/命名空间，再打开同一目录内的类页、方法页和 constructor/overview 配套页。

融合 pass 的当前路由（相对 `docs/zh/api/graph_engine_api/`）：

| 用途 | C++ / Python 路径族 |
|---|---|
| C++ pass 基类 | `cpp/ge/fusion/FusionBasePass/`、`cpp/ge/fusion/PatternFusionPass/`、`cpp/ge/DecomposePass/` |
| C++ pattern / 匹配 / 子图重写 | `cpp/ge/fusion/Pattern/`、`cpp/ge/fusion/PatternMatcher/`、`cpp/ge/fusion/PatternMatcherConfig*/`、`cpp/ge/fusion/MatchResult/`、`cpp/ge/fusion/SubgraphBoundary/`、`cpp/ge/fusion/SubgraphRewriter/` |
| C++ 上下文与注册阶段 | `cpp/ge/CustomPassContext/`、`cpp/ge/CustomPassStage.md`、`cpp/ge/fusion/FusionPassRegistrationData/` 与 `PassRegistrar/`；注册宏签名仍以头文件为准 |
| C++ ES 构图 | `cpp/ge/es/EsGraphBuilder/`、`cpp/ge/es/EsCGraphBuilder/`、`cpp/ge/es/EsTensorHolder/`、`cpp/ge/es/CompliantNodeBuilder/` |
| Python pass | `python/ge/passes/<类名>/`，例如 `python/ge/passes/PatternFusionPass/` |
| GE options / 开关 | `cpp/ge/options_params/options_parameters_description.md` 及同目录按主题拆分的页面 |
| GNode / Graph | `cpp/ge/GNode/`、`cpp/ge/Graph/` |

定位某个 API 时，优先用 `find "$GE_REPO_PATH/docs/zh/api/graph_engine_api" -type f -iname '<API名>*.md'`，再用 README 所列命名空间缩小结果；签名落定仍走 `tips/api-signature-gate.md`。配套的构造函数、析构函数和 `overview.md` 也属于同一个事实源，不能只打开类主页。

---

## 4. ES 构图 → ②（建 replacement 图）

### 4.1 用户指南（先从 README 进）

| 文件 | 行数 | 讲什么 |
|---|---|---|
| `docs/zh/user_guides/es_graph/README.md` | 102 | **ES 体系入口**。三个核心特点：API 基于算子原型自动生成（自定义算子也能用 ES 构图）、原生支持 C/C++ 并可扩展 Python、通过代码生成 + IR 语义兼容实现全维度 API/ABI 兼容。附快速导航与路线图 |
| `docs/zh/user_guides/es_graph/api/es_cpp.md` | 155 | **C++ 类关系**。对外头文件在 `inc/external/ge/eager_style_graph_builder/`（分 `c/esb_funcs.h` 与 `cpp/`）。逐一讲 `EsGraphBuilder`、`EsTensorHolder`、`EsTensorLike`、`CompliantNodeBuilder` 四个核心类和 C API 函数，带类关系图与示例 |
| `docs/zh/user_guides/es_graph/api/es_python.md` | 308 | **Python 模块**（对应 `api/python/ge/ge/es/`）。`GraphBuilder`、`TensorHolder`（运算符重载）、`InputType` 枚举、作用域管理器（属性作用域 / 控制依赖作用域）、C API 包装层，以及 `es` 与 `graph` 模块的关系 |

### 4.2 工具

| 文件 | 行数 | 讲什么 |
|---|---|---|
| `docs/zh/user_guides/es_graph/tools/gen_esb.md` | 121 | `gen_esb` 生成器。两种生成模式（代码生成 / 历史原型库生成）、`ASCEND_OPP_PATH` 等环境变量要求、全部参数说明、输出文件说明，十来个命令示例 |
| `docs/zh/user_guides/es_graph/tools/generate_es_package_cmake_readme.md` | 538 | `add_es_library` 与 `add_es_library_and_whl` 两个 CMake 函数——前者只生成 C/C++ 动态库，后者额外产出 Python wheel。含前置要求、参数、历史原型库相关 CMake 变量、产物清单、对外 target 与完整示例 |

### 4.3 设计文档（深入原理时看，一般开发不需要）

`docs/zh/design/modules/es_graph/design/`：

| 文件 | 行数 | 讲什么 |
|---|---|---|
| `architecture_design.md` | 2107 | ES 主设计文档。从现阶段构图接口痛点讲起，整体设计、API/ABI 兼容性设计（C / C++ / Python 三语言）、IR 语义兼容、API 风格设计（C API ≈560 行、C++ ≈250 行、Python）、IR 与 API 映射、头文件与模块拆分、构建流程与模块部署 |
| `es_cxx_compatibility_design.md` | 508 | 历史原型库在 ES 场景下的设计。**核心是 §4 重载生成与二义性处理**：`gen_esb` 如何消费历史原型数据生成 C++ 重载接口，二义性怎么解——Gate1/Gate2 的"区间交集"判定、方案 A 的防呆机制。§8 附录给存在性矩阵与二义性检测例子 |
| `history_op_registry_protocol.md` | 184 | 历史原型库**协议**（与上一篇是"协议 vs 实现"关系）。`index.json` / `metadata.json` / `operators.json` 三类文件的字段语义、目录结构与分包、随 Ops run 包发布的数据流、消费者文件系统接口、协议演进的向后兼容规则 |
| `ownership_analysis.md` | 460 | Python 层与 C++ 层**所有权反转**分析。C++ 侧 `EsCGraphBuilder` 持有所有资源，Python 侧却是 `TensorHolder` 持有 `GraphBuilder`——为防 builder 被 GC 后 `t._handle` 悬空。并分析由此带来的循环引用风险与语义不一致，给出弱引用等备选方案 |

`docs/zh/design/modules/es_graph/rfc/`：`rfc-001-feature-cxx_compatibility.md`（93 行，C++ ES API 兼容性提案：目标、详细设计、三阶段实现计划、影响分析，及被否掉的方案 B 版本命名空间与对比）、`README.md`（29 行，RFC 流程与列表）。

---

## 5. Python 构图 API（ge_python）→ ②

| 文件 | 行数 | 讲什么 |
|---|---|---|
| `docs/zh/user_guides/ge_python/README.md` | 127 | **GE-PY 总入口**。明确划分 `graph` 模块（`Graph`/`Node`/`Tensor`/`TensorDesc` 基础图操作）与 `passes` 模块（Python 级自定义 Fusion Pass 能力），含文档导航与模块关系图 |
| `docs/zh/user_guides/ge_python/api/GraphBuilder.md` | 418 | `ge.es.GraphBuilder` 手册。`create_*` 系列创建输入 / 常量 / 标量 / 向量 / 变量等张量，`set_graph_output` 设输出，`build_and_reset` 构建返回 `Graph` |
| `docs/zh/user_guides/ge_python/api/TensorHolder.md` | 295 | `ge.es.TensorHolder` 手册。支持 `+ - * /` 运算符重载，可链式调用 `set_data_type` / `set_format` / `set_shape`，以及 `get_owner_builder` |
| `docs/zh/design/modules/ge_python/ge_python.md` | 703 | GE-PY 类关系设计（对应 `api/python/ge/ge/`）。给出 GE 初始化/释放、离线编译保存模型、session 加图执行三条完整调用链示例 |

---

## 6. 样例（实操，全在 `examples/` 下）→ ②

总入口 `examples/fusion_pass/README.md`；两个样例组各有自己的 `README.md`。

| 样例组入口 | 类型 | 子样例 |
|---|---|---|
| `examples/fusion_pass/pattern_base_pass/README.md`（**推荐优先**） | `PatternFusionPass` / `DecomposePass` | 当前 master 快照为 9 个：`1_fuse_matmul_add_pass` 至 `9_bmm_to_mul_reduce_pass`（具体名称以 README 为准） |
| `examples/fusion_pass/graph_base_pass/README.md` | 直接用 graph 接口改图 | 当前 master 快照为 5 个：`1_fuse_matmul_add_pass`、`2_move_relu_before_concat_pass`、`3_modify_conv_data_format_pass`、`4_mmoe_bmm_split_pass`、`5_mmoe_matmul_pass` |

**子样例目录结构会变，不能把数量或 `cpp/` / `python/` 布局当契约**。用前执行 `find "$GE_REPO_PATH/examples/fusion_pass" -mindepth 2 -maxdepth 2 -type d`，再读取目标样例的 README/CMakeLists；只有实际存在的语言、数据和 `gen_es_api/` 目录可作为开发或验证输入。

ES 构图样例：`examples/es/`、`examples/custom_es_api/`。

> 上表为规范仓 `cann/ge` master 的清单。**用前先 `find "$GE_REPO_PATH/examples/fusion_pass" -maxdepth 2 -type d` 对齐你的 checkout**（精简 fork 可能只保留部分样例），路径以实测为准。按**场景**取用见 `example-map.md`。

> **与开发主流程导航的关系**：本册给「GE 仓里有什么文档/样例、在哪」；`references/pass-development-paradigm.md` 给「开发按什么顺序读它们、每步看哪份」；`references/example-map.md` 给「我的场景对应哪个样例」。三者互补，按需跳。

---

## 7. 背景与机制（写 pass 时会用到）→ ① / ③

| 文件 | 行数 | 讲什么 | 关联 |
|---|---|---|---|
| `docs/zh/design/modules/graph_metadef/ascend-ir.md` | 339 | **AscendIR 中间表示——你在 pass 里操作的就是这套数据结构**。四层对象模型、锚点系统（连接关系的内嵌表达）、`ComputeGraph`、`OpDesc`、算子注册与工厂模式、`GeTensorDesc`、图序列化、图工具类 | 开发 |
| `docs/zh/user_guides/understanding_ge_format_from_semantics_to_api.md` | 260 | GE 中 Format 的建模与接口语义：**Origin 与 Storage 两套表示体系**、Format 优化基本原理 | `MeetRequirements` 判 format 的必要背景；`tips/format-sensitive-nchw.md` |
| `docs/zh/design/architecture.md` | 217 | GE 架构总览。"插件和扩展机制"一节说明 GE 通过插件扩展 AscendC 自定义算子和自定义 Pass，并指向 `examples/`（§6） | 定位 |
| `docs/zh/design/README.md` | 59 | 设计文档集索引 | 导航 |
| `docs/zh/design/constraints/graph_metadef.md` | — | 图编译公共基础结构的软件约束（模块独立性、边界稳定性、接口演化兼容性等六条原则） | 约束 |

### 7.1 其它特性文档 `docs/zh/design/features/`（23 篇）

| 文件 | 讲什么 | 关联 |
|---|---|---|
| `infer_shape.md` | shape 推导机制 | `tips/infershape-util-only.md` |
| `infer_format.md` | format 推导机制 | `tips/format-sensitive-nchw.md` |
| `engine.md` | 引擎选择（select engine / kernel） | `tips/es-all-no-version-rename.md`、`kernel-registration-mismatch.md` |
| `profiling.md` | profiling 机制 | 性能分析 L1 |
| `datadump.md` | dump 机制 | `tips/dump-first-op-type.md`、验证 |
| `constant_folding.md`、`tensormove_delete.md`、`zero_copy.md`、`concat_no_task.md` | 相关图优化 | 图层收益核对（性能 L0） |
| `atc_raw_ge_options.md` | ATC 透传 GE options | 开关策略 |

### 7.2 其它用户指南 `docs/zh/user_guides/`

| 路径 | 讲什么 | 关联 |
|---|---|---|
| `dump/readable_dump.md`、`dump/README.md` | dump 可读化 | 验证、dump 对比 |
| `custom_op/`（`custom_op_v1`、`custom_op_v2`、`README.md`） | 自定义算子 | 自定义算子 case（`es_custom`） |
| `atc_shape_configuration_guide.md`、`offline_compile.md` | ATC / 离线编译 | 验证、触发编译 |

- **约束**：`docs/zh/design/constraints/`（`graph_split.md`、`memory-constraints.md`、`graph_metadef.md`、`stream_allocator.md`…）。
- **模块设计**：`docs/zh/design/modules/`（`compiler/`、`es_graph/`、`ge_python/`、`graph_metadef/`、`runtime/`）。

---

## 8. 源码与头文件（签名回退以此为准）→ ②

- `inc/` —— 对外头文件（命名空间 / 头文件对应见 `tips/ge-fusion-namespace.md`）；ES 对外头文件在 `inc/external/ge/eager_style_graph_builder/`。
- `api/` —— `acl` / `acl_c` / `atc` / `python` / `session` 接口；Python ES 在 `api/python/ge/ge/es/`。
- `compiler/`、`parser/`、`graph_metadef/`、`runtime/`、`base/` —— GE 各模块源码。手写 Pattern 融合在 `compiler/graph/fusion/`，自动融合在 `compiler/graph/optimize/autofuse/`。
- `examples/fusion_pass/**/src/` —— 样例 pass 实现（可读可参考，**不照抄样例中的硬编码节点名**）。

---

## 速查：需求 / 开发 / 验证各读哪几节

| 阶段 | 读本册哪节 |
|---|---|
| ① 需求（接口清单 / 注册阶段 / 开关 / 目标结构） | 1、2、3、7 |
| ② 开发（骨架 / 签名核对 / ES 构图） | 1、2、3、4、5、6、8 |
| ③ 验证诊断（dump / 编译 / format / 性能） | 1、3、7 |
