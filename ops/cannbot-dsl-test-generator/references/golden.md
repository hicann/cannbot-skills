# 直接编写最终 Golden

在 `TEST_DIR/<op>_golden.py` 编写纯 CPU PyTorch 参考实现。文件应包含：

- `DESIGN_REFS`：列出所提供设计文件中真实的算法或阶段条目。
- `def <op>_golden(<spec 输入>, <spec 属性>):`：参数顺序与 spec 相同；可选输入提供默认值。
- `OP_NAME`、`INPUT_NAMES`、`OPTIONAL_INPUT_NAMES`、`OUTPUT_NAMES`、`ATTR_DEFAULTS`、`REQUIRED_ATTR_NAMES`、`TOLERANCE`：与当前 spec 一致的字面量元数据。
- `GOLDEN_FUNCTION = <op>_golden`、`make_inputs`、`simulate`：固定测试入口需要的公共接口。

多输出按 spec.outputs 顺序返回，也可使用以声明输出名为键的映射；单个 Tensor 输出可直接返回 Tensor。固定辅助模块绑定既有 spec 的输出声明，在数值缓存准备、缓存命中及实际比较时核对返回数量、名称、Tensor/TensorList 类型、适用的输入输出 dtype 组合，以及已有明确输出 rank 和输出缓冲区配方的 shape/dtype。可选输出及其存在性条件、TensorList 长度沿用现有声明；不能用空容器代替必需 Tensor。合法零元素 Tensor 仍检查 shape/dtype 并产生比较记录，契约允许的缺省输出或空 TensorList 留下对应检查记录，不以空记录宣布数值通过。符号形状规则及数学等价性仍按原有 CPU 样例、边界及 DESIGN Stage 复核，不把上述结构检查当成完整证明。

Golden 不导入目标算子、CANNBotDSL 或设备库，不复制设备调度。重要数值阶段在注释中回指 spec 字段与 DESIGN ID。运行 `prepare_golden.py --spec SPEC --design DESIGN --test-dir TEST_DIR` 检查函数、元数据、CPU 独立性和容差指标，并安装固定测试入口；脚本不重写 Golden，也不提前执行目标算子。

输出校验只求值会约束输出存在性的 `semantic_cases`，仅约束输入的条件仍由原有输入验证负责。相关条件引用 TensorList 的 dtype 时，非空列表以一致的实际成员 dtype 判定；空列表只有声明唯一 dtype 时才可据此确定，混合成员或多种可能类型不得取第一项猜测。无法确定时明确报告条件求值缺口，不跳过真正影响输出的条件，也不据此宣布目标实现失败。

若 spec 声明了 `op.aicpu_kernel`，算子实现模块须提供 `_aicpu_preprocess(inputs)` 钩子供 NPU 测试时做 AICPU 前置处理；Golden 不实现此钩子，也不导入设备库，保持纯 CPU 参考实现。

make_inputs 必须执行 CSV 的真实配方，包含 values/arange 等当前使用的 fill；可采用 prepare_golden.py 的 SUPPORT。按固定 seed 物化并复核特殊输入关系，不能以 note 或 Case 名称代替真实取值。prepare_golden.py 同时安装 cannbotdsl_case_contract.py，固定 runner 通过它执行输入与结构谓词。

## 精度判定契约

`numerical_tolerance.source` / `pass_criteria` 保留权威来源和判定依据，`per_dtype` 与 Golden 的 `TOLERANCE` 字面量一致。预检不解析自然语言公式，因此不能替代执行者与验收者对来源、公式和阈值的核对。

| metric | 执行语义 |
|---|---|
| `max_relative` | 保持已有行为：逐元素 `abs(actual-golden) <= atol + rtol*abs(golden)`；不要按名称误认成 MARE 公式 |
| `bitwise_equal` | 输出存储位完全相等，包含有符号零 |
| `staged_relative`（浮点默认），`backend: native` | 第一阶段 MERE/MARE；失败后使用独立同精度参考执行小值域、相消与正常域判定，三域均通过才通过 |
| `mean_relative`（显式单阶段），`backend: native` | `e=abs(actual-golden)/(abs(golden)+epsilon)`；**同时**要求 `mean(e) < rtol` 与 `max(e) < max_relative_rtol`，严格小于 |

默认完整两阶段示例（数值仅示范字段，应取当前任务的权威标准；明确单阶段标准才选 mean_relative）：

```yaml
numerical_tolerance:
  source: <权威标准路径、版本和章节>
  per_dtype:
    float16:
      metric: staged_relative
      backend: native
      rtol: 0.0009765625
      max_relative_rtol: 0.009765625
      epsilon: 1.0e-7
      atol: 0.0
      fallback: {reference_mode: native}
```

`mean_relative` 的两个阈值必须显式提供且为有限正数，不自动推导最大阈值；`epsilon` 未写时语义固定为 `1e-7`，写出时须为有限正数。`atol` 可省略或为零，不用于增加绝对误差兜底。原生路径以 fp64 计算有限位置的误差，匹配的同号 Inf 不参与归约，NaN 仅在用例显式声明 `match_golden_nan` 且位置匹配时允许；无有限位置时两项误差为零，空张量同样处理。有限值不匹配 Inf、NaN 位置或 Inf 符号不匹配均失败。

只支持内置 `backend: native`（可省略）；其他 backend 明确失败，不能将旧配置静默改为内置策略。指标、阈值及 NaN/Inf 语义须有已确认依据；该单阶段指标不附加小值域或相消错误点比例；完整分域标准显式选 staged_relative。未支持的判据报告能力缺口，不能近似替代或放宽标准。`prepare_golden.py` 复用同一预检，仅采用已冻结策略或统一内置默认值。

## 完整两阶段比较

规则核对来源：cann-bench 提交 `e90a053df93997e026d19fd4fd42fc03cbf2b2fd` 的 `docs/spec/benchmark_spec.md`、`src/kernel_eval/utils/compare.py` 与 `thresholds.py`，许可证为 CANN Open Software License Agreement Version 2.0。复用本 Skill 内置实现，不依赖外部仓库安装路径。保留有限值溢出的缩放计算保护，不沿用误差溢出后排除比较点的行为。

未指定精度策略时，新生成的浮点用例默认采用内置 `metric: staged_relative`（`backend: native`）；旧 `mean_relative` 仍只执行原有单阶段判定，不能因新能力而静默放宽历史契约。两阶段适用于 float16/bfloat16/float32/float64，整数、布尔或 bitwise_equal 继续使用原策略。策略由确认的要求决定，不根据资料仓名自动选择。只列 MERE/MARE 和阈值的摘要不足以证明要求单阶段；先核对所引用完整标准。新任务默认完整两阶段，明确要求单阶段或其他标准时才覆盖；不自动迁移已冻结契约。默认主参考采用独立高精度计算，按输出契约转换到目标 dtype；同精度参考保留规定的原精度计算、累加及舍入语义。不能只把高精度最终结果 cast 后充作同精度计算。

阶段一沿用 `epsilon`、`rtol`、`max_relative_rtol` 的严格平均/最大相对误差判定，满足即通过。失败后，按同一 Golden 基准检查 small_value/cancel/normal 三域：小值域按绝对误差计错，相消和正常域按 `error > max_relative_rtol` 计错。小值/相消要求 `候选错误数 / max(参考错误数, 1) <= error_ratio_limit`；正常域候选零错误通过，候选非零而参考零错误失败，否则要求同一比值不超过上限。三域均通过才通过。

`fallback` 可覆盖 `small_value_threshold`、`small_value_error`、`cancel_boundary`、`cancel_zero_threshold`、`error_ratio_limit` 和 `reference_mode`，其他键报错。未覆盖时采用以下内置值；小值域为 `abs(g)<small_value_threshold`，相消域为 `abs(actual)<cancel_zero_threshold` 且 `small_value_threshold<=abs(g)<cancel_boundary`，其余有效点为正常域。

| dtype | small_value_threshold | small_value_error | cancel_boundary / cancel_zero_threshold |
|---|---|---|---|
| float16 | 2^-11 | 2^-16 | 2^-5 |
| bfloat16 | 2^-8 | 2^-16 | 2^-3 |
| float32 / float64 | 2^-14 | 2^-30 | 2^-8 |

`error_ratio_limit` 默认 2。`reference_mode` 默认 `native`，准备和缓存加载时必须存在独立 `NATIVE_REFERENCE_FUNCTION`；执行时按需调用，缺失或调用失败明确失败，不自动替换为主参考截断。显式兼容策略 `native_or_truncated` 有参考时调用它，否则使用主参考已转换到输出 dtype 的结果；`native` 强制独立同精度参考存在，`truncated` 明确仅采用主参考截断。截断输出不是同精度计算；只有已确认的显式 `truncated` / `native_or_truncated` 契约允许采用该替代，默认完整两阶段不允许；显式契约要求同精度计算时必须选 native 并提供绑定；参考存在但调用失败时明确失败，不降级。Golden 可增加 `NATIVE_REFERENCE_FUNCTION = <独立的具名CPU函数>`，不能绑定主参考或候选。固定 runner 在实际需要阶段二时，使用本次同一份物化输入与属性的隔离副本调用，按原输出契约校验；同一调用的多个输出共享该次参考计算，不重物化输入，不缓存候选或阶段二通过状态。

两阶段数值比较支持连续和非连续 Tensor，布局/连续性限制仅在输出契约明确要求时单独验收；有限输入的误差中间运算溢出采用缩放计算，不能删除该位置或当作空集合通过，无法可靠判定时明确失败。该策略的 NaN 按位置一致匹配（显式 match_golden_nan 仍须实际触发预期 NaN），双侧同号 Inf 排除比较、异号失败，单侧 Inf 用目标 dtype 最大有限值替换后继续判断；无有效点或合法空张量按原输出契约处理。两阶段函数从已转换到输出 dtype 的主参考计算误差，不能以最终输出 dtype 相同推断参考中间精度。

未指定时的 T 默认值：float16=2^-10、bfloat16=2^-7、float32/float64=2^-13，max_relative_rtol=10*T，epsilon=1e-7；整数/布尔默认 bitwise_equal。输出 dtype 未知或无受支持默认值时报告缺口，不猜测。规格阶段先区分用户/目标规格的明确标准与生成器或示例的初始默认值。无明确要求时，可用 check_tolerance.py --spec SPEC --defaults 只读取得统一默认提案（mode=default_proposal），将 source/per_dtype 写入待冻结 spec 后重新运行普通预检及完整规格校验；提案通过不代表当前 spec 已校验，不覆盖明确标准。规格阶段将实际默认策略及依据写入 spec 的 per_dtype；check_tolerance 在缺省契约时计算并报告同一默认值，仍只读磁盘文件。已有明确策略优先，部分或不受支持的显式契约报错，不能被默认值覆盖。

例（数值须符合本任务确认标准）：

```yaml
numerical_tolerance:
  per_dtype:
    float16:
      metric: staged_relative
      backend: native
      rtol: 0.0009765625
      max_relative_rtol: 0.009765625
      epsilon: 0.0000001
      fallback:
        reference_mode: native
```

所有正式筛选、TDD、全量模式使用同一内置比较器，CSV 的 status 和退出码反映最终判定，详情保留阶段与三域计数。预检不接受额外 `standard_stages` / `evaluation_preconditions` 来补充隐藏规则：完整数值策略写在 per_dtype，输出/断言要求沿其已有公开契约表达；未支持部分必须先报告能力缺口，不能用旁路文件或报告叠加放行。无需外部评测包或自定义 checker backend。

## 独立主参考与计算精度

`GOLDEN_FUNCTION` 绑定唯一主参考，计算精度、累加类型及舍入/转换边界遵守已确认数值契约，并在既有 spec 的依据、设计相关章节及参考证据中说明。主参考精度按已确认契约确定，不按输入 dtype 自动选择；默认完整两阶段采用高精度主参考，并必须提供独立同精度参考，明确采用其他策略时按该契约执行。未指定数值契约时按上述统一默认策略执行；参考路径本身存在语义缺口或与明确要求冲突时先确认，不因失败后更换参考来取得通过。固定缓存直接调用该绑定，输入只物化一次，主参考使用同一份输入及属性的隔离副本；可选输入缺省、属性默认值及显式覆盖沿用规格元数据。`simulate()` 保留为兼容接口，不作为固定缓存或 runner 的期望值来源；主参考缺失或调用失败时明确失败，不回退到它。

同精度参考依据规格、可信参考路径及不违反规格明确要求的冻结 DESIGN 数值链独立编写，使用相同物化输入和属性，承接计算精度、累加类型及影响数值的中间舍入/转换边界。规格明确要求中间保持某精度或禁止某转换时优先，设计不得覆盖；规格未固定这些细节而设计合法采用低精度计算链时，独立 CPU 参考表达相应舍入，不把原始高精度参考全程 fp32 的写法自动当作实现硬约束或该路线的同精度语义。最终 dtype 相同、或高精度计算末尾 cast，不证明路径同精度。参考确定于真实目标验证前，已有歧义先按调用方协议交接，不能仅因候选红灯加入未确认转换或放宽标准；漏表达已确认的合法边界时按来源修正，合法路径变更仍按原版本及缓存失效规则处理。不得调用、别名绑定或复制候选实现，不复制设备切块、流水或同步，不将候选特有补偿写进 Golden。高精度主参考仍沿数学契约，不照搬设计路线额外的低精度舍入；规格明定的量化、舍入仍忠实表达；独立性不禁止两路实现共享已确认的数学语义。沿既有参考说明给出来源，执行、自检与验收核对实际函数，不另设格式或调度模拟。

需要独立交叉核验或高精度误差诊断时，按既有任务范围记录来源、实际比较范围和结果；诊断参考不自动成为生产准入门槛，也不替换主参考。设计舍入链模拟只能说明对应路径与输入，不单独证明完整算子正确。实现与参考不一致时先核对实际输入、参考职责及数值契约，不直接归咎设计或擅自增加补偿。

CPU 缓存 v4 保存同一配方对应的输入、主参考结果及输出模板；Golden 源码、绑定的既有输出契约、PyTorch 版本、配方、seed、调用序列和断言共同决定身份。旧版本缓存不复用，避免沿用通过 `simulate()` 生成的期望值；格式不符、参考或绑定契约变化后重建，命中时仍检查输出契约，不给旧测试结果补签。已有运行保持其冻结 runner、契约和缓存工具，新生成资产使用本版本；不自动迁移旧运行或冻结规格。
