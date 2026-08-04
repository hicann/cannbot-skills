# 融合 pass 开发主流程导航（pass-development-paradigm）

这是**开发主流程的导航层**：从「输入识别」到「验证」按顺序给出每一步该做什么、权威知识在哪份 reference/tip。**本文不复制规范性事实**——签名、枚举、报错根因、代码骨架都在各自的唯一事实源里，按链接跳过去读。

> 谁读：② 开发（主线）；① 需求分析在做接口选型时回看 §3/§5；③ 验证在定位「写错还是没生效」时回看 §7。
>
> 与其它文档的分工：
> - **接口签名 / 枚举 / 命名空间 / 基类关系** → `interface-catalog.md`（唯一事实源）。
> - **目标片段怎么描述、可选输入怎么办** → `fragment-spec.md`。
> - **场景 → GE 真实样例 → 看点** → `example-map.md`。
> - **GE 文档 / 源码确切路径** → `ge-repo-map.md` §1 任务路由表。
> - **融合未命中 / replacement 失败等诊断** → `fusion-troubleshooting.md`（诊断树）。
> - **逐条硬性做法（命名 / format / InferShape / es_all 决策树 …）** → `references/tips/*.md`（仍是各条事实的权威定义源；迁移映射见 `tips/MIGRATION.md`）。

---

## 1. 输入识别与证据优先级

先弄清「要作用的那一小段」和「用什么对外接口」，再动手。证据优先级（高 → 低）：**GE 仓文档 / 样例 → 用户已给的输入包 → 向用户提问**。能核实就不要问。

- **真实 op type 以 dump 为准**：前端框架算子名（`MatMul`/`Conv2D`/`nn.Add`）经 GE 导入会改名、因模型而异、静态推不出。写任何按 op type 的匹配前，先对真实模型 dump 取导入后名字。→ `tips/dump-first-op-type.md`（最常见的静默失败原因）。
- **大网络怎么指片段**：用户不必描述整张网，只要说清目标结构 + 定位 + 边界 + 守卫 + 可选算子 + 目标形态。退化谱系（从「只描述结构」到「最小复现子图」）见 `fragment-spec.md` §三。
- **输入脚本兼容**：`data/` 下 TF1 graph-mode 脚本 / 缺 `npu_bridge` 的处理纪律见 `tips/tf1-graph-mode-compat.md`（不改 `data/`、不重装 TF、缺 npu_bridge 走离线）。
- **GE 仓核实**：仓根动态解析（默认纯只读、绝不联网），路径与跳节见 `ge-repo-map.md` §1 任务路由表；联网授权协议见统一 skill 阶段一“证据核实、联网与澄清”。

## 2. 目标片段与边界

片段规格六要素（目标结构 / 定位 / 边界 / 守卫 / 可选算子 / 目标形态）的语义、模板、退化矩阵全部在 `fragment-spec.md`。本节只提醒两条**写 pass 前必须先定**的边界：

- **可选输入在匹配阶段就分叉**：`bias`/`offset_w` 等 `OPTIONAL_INPUT`「存在」与「不存在」是两种输入个数 = 两个 pattern，不是 replacement 里一个 if 能吸收的。三种策略 S1/S2/S3 见 `fragment-spec.md` §四。
- **自包含 / 输出边界**：替换后仍被片段外部使用的 Tensor 必须声明为 pattern 输出，否则替换后断图。边界规则全表见 `interface-catalog.md` §一.2。

## 3. graph / pattern / decompose 选型

选型是**硬约束驱动的**，不是按网络形状猜。判定顺序（先命中先定）：

1. **pattern 可表达性否决**：目标片段含控制边 / 嵌套子图 / 动态输入输出个数的节点，或三条边界规则不满足 → 退 `graph_base_pass`（函数式或 `FusionBasePass`）。约束全表见 `interface-catalog.md` §一.2。
2. **目标形态**：一段子图 → 另一结构 → `PatternFusionPass`；单个算子 → 多算子（由算子属性决定）→ `DecomposePass`。
3. **运行期算子数可变**（如合并 N 个同源算子，N 不固定）→ `graph_base_pass`（pattern 是静态拓扑模板）。
4. 对外接口 / 参考路径 / 交付物三者冲突时，以对外接口为准，冲突记进文档 §5.3 被否决方案。

接口映射、注册宏、签名形态见 `interface-catalog.md` §一；场景 → 样例 → 看点见 `example-map.md`。

### 3.1 pass 选型矩阵（权威源）

按目标场景查表定路线，关键理由或退化条件见右列。本矩阵是选型的唯一事实源，统一 skill 的阶段一据此产出需求分析文档 §5。

| 场景 | 首选路线 | 关键理由或退化条件 |
|---|---|---|
| 单个算子拆成多个算子 | `DecomposePass` | 由 op type 找节点，再按属性守卫并生成替换子图 |
| 固定数据拓扑的多→一、多→多、删除或短路 | `PatternFusionPass` | 输入输出边界固定且满足 pattern 约束 |
| 控制边、嵌套子图、动态输入或动态输出 | graph 路线 | pattern 无法表达这些结构 |
| 可选输入组合有限且明确 | 多个 pattern 或只支持一种形态 | 每种实际输入个数对应一种 topology |
| 可选输入组合过多、匹配关系高度动态 | graph 路线 | 避免 pattern 数量组合爆炸 |
| 只改属性、手工改边、删点或需要全图遍历 | graph 路线 | 需要显式图控制 |

"graph 路线" = 函数式 graph pass（`REGISTER_CUSTOM_PASS(...).CustomPassFn(...)`）或 `FusionBasePass`，二者区别与判定见 `interface-catalog.md` §一。可选输入三种策略 S1/S2/S3 见 `fragment-spec.md` §四。

## 4. Python / C++ 选择

由**交付物 + 加载方式**决定，不是由网络形状决定：

- `src/*.cpp` + CMake + `.so` + `REG_*` 宏 → **C++**，装到 `<OPP_ROOT>/vendors/<vendor>/custom_fusion_passes/`，`atc` 触发编译。
- `src/*.py` + `ASCEND_GE_PY_PASS_PATH` + `@register_fusion_pass` / `@register_decompose_pass` → **Python**，**`pyatc`** 触发（同进程加载 Python pass，不能用 `atc`）。

> **这些环境变量怎么解析**：`OPP_ROOT`（由 `ASCEND_OPP_PATH` 或已验证 CANN 根下 `opp/` 解析）、`ASCEND_GE_PY_PASS_PATH` 的取值格式与扫描规则（GE 仓 `docs/zh/user_guides/ge_python/env/ASCEND_GE_PY_PASS_PATH.md`，路径见 `ge-repo-map.md` §1 任务路由表）、`$GE_REPO_PATH`（由 `"$SKILL_ROOT/scripts/sync_ge_repo.sh"` 动态解析，默认纯只读）——CANN 路径一律从 env 展开、禁硬编码默认安装路径，探测与降级规则见 `knowledge-base.md` §四；加载、编译和 dump 的具体执行步骤见统一 skill 阶段三“验证顺序与证据”。

GE 推荐主力路线是 Python（`@pattern` 表达式写法）。Python pass 的特有写法（`@pattern` 与 `patterns()` 互斥、不得重写 `run()`、`capture_tensor` 何时必须、返回值语义与 C++ 相反）见 `interface-catalog.md` §二。

## 5. 注册阶段与开关

**注册阶段**是注册参数（C++ `.Stage(...)`、Python `stage=...`），不是编译 option。完整枚举、版本门槛、Python 只暴露 4 个阶段见 `interface-catalog.md` §一.1。判定规则一句话：只按 op type/拓扑匹配不依赖 shape → `kBeforeInferShape`（默认首选）；依赖 InferShape 后真实 shape/通道/groups/data_format → `kAfterInferShape`（replacement 须自行保证 shape 连续性，见 `tips/infershape-util-only.md`）。

**开关策略**：`CustomPassContext::GetOptionValue` / `PassContext.get_option_value`（C++ 头文件 `@since 9.0.0`，Python 按同版本门槛对待）。但「钩子能不能拿到 context」取决于接口——pattern/decompose 的 **V1 钩子无 context**，要在钩子内读 option 须升 V2（`PatternFusionPassV2`/`DecomposePassV2`，CANN ≥ 9.1.0）。对照表见 `interface-catalog.md` §一.3。

## 6. API 签名核实（写码前硬门禁 G2）

**凡源码里要调用的 GE/ES API，写代码前必须列「API 签名核对表」；没有签名证据不得开始写实现。** 核对表字段、证据优先级、降级条款（文档与头文件都不可达时怎么标假设）全部以 `tips/api-signature-gate.md` 为准。门禁 G2 的判定逻辑见统一 skill 阶段二。

## 7. 实现、加载、验证主流程

实现时的硬性做法分布在各 tip，按场景取用：

| 做什么 | 权威 tip |
|---|---|
| 命名与日志风格（snake_case / `std::cout`/`print`，以 GE 仓样例为准） | `tips/cpp-style-naming.md` |
| 新建算子节点：信任 `es_all` wrapper，不预换版本名（唯一决策树） | `tips/es-all-no-version-rename.md` |
| `es_all` 未暴露所需 wrapper → 显式手建（IR 顺序严格按 `REG_OP`） | `tips/compliant-node-builder-ir-order.md` |
| Conv/Pool 输入 format 必须显式设 NCHW/NHWC，否则 E50002 | `tips/format-sensitive-nchw.md` |
| C++ 推 shape 只用 `InferShapeUtil::InferShape`，禁 `GeUtils::InferShape` | `tips/infershape-util-only.md` |
| pass 框架类都在 `ge::fusion`，禁 `using namespace fusion;` | `tips/ge-fusion-namespace.md` |
| Python `remove_node` 后节点对象即失效，删除前缓存 | `tips/python-remove-node-lifecycle.md` |
| 加载/编译前隔离并追踪本轮 custom pass 产物 | `tips/stale-pass-artifact-cleanup.md` |

加载与验证的具体步骤（CMake / 构建 / ES API / Python 加载 / 触发编译 / dump 对比 / 日志校验 / 性能两档）是统一 skill 阶段三的职责，不在本文重述。**replacement / 加载 / 输出是否正确出了问题**走 `fusion-troubleshooting.md` 的诊断树。

## 8. GE 真实样例指针

**先按目标片段场景查 `example-map.md`**，跳到最贴近的真实 GE 样例，照它的骨架 / 日志点 / 注册阶段写；匹配对象换成本 case dump 出的真实 op type，不沿用样例节点名。样例路径以 `$GE_REPO_PATH` 实测为准（`cann/ge` 布局见 `ge-repo-map.md` §6）。

样例覆盖的场景：多算子融合为单算子（MatMul+Add→GEMM）、条件删除（Add(x,0)→x）、拆分展开（分组卷积→Split+Conv×n+Concat）、capture tensor、PatternMatcherConfig 约束匹配、依赖 shape 的 pattern 融合、函数式 graph pass 改边删点、Python pass 各接口族。
