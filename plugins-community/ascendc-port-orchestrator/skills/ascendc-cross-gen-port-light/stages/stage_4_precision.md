# 阶段 4：精度验证（含 Subnormal 测试）

**前置条件**：`[GATE-3] BUILD=PASS INSTALL=PASS` 已输出。

## LOADED 检查点（MUST 全部完成后才能写测试代码）

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/precision-testing/pytorch-binding-build-guide.md` | `[LOADED] pytorch-binding-build-guide` |
| 2 | `references/precision-testing/torch_aclnn_helper.h.template` | `[LOADED] torch_aclnn_helper.h.template` |
| 3 | `references/precision-testing/OPS_PRECISION_STANDARDS.md` | `[LOADED] OPS_PRECISION_STANDARDS` |
| 4 | `references/impl/api-diff-guide.md` | `[LOADED] api-diff-guide` |

**全部 4 个 LOADED Token 输出后，方可继续。任何一个缺失 → STOP。**

## 比对原则：算子与标杆双向互检

本 skill 不依赖外部 golden：迁移只对 A2 源码做局部修改、整体逻辑保持可靠；本阶段的标杆由阶段 1 从 A2 源码逆向合成（CPU 参考实现）。因此 A5 实测结果与标杆的比对是**双向互检**：

- **结果一致** → 迁移与标杆互相印证，双双可信；
- **结果差异大** → 不默认算子错、也不默认标杆错：MUST 先按 Step 4.6.3 重新检验标杆（标杆精度对齐 / denormal 域判别 / 原版二进制对照），确认标杆无误后，再进入算子侧根因分析（Step 4.6.1 / 4.6.2）。

禁止绕过标杆检验直接把差异归因于算子。

**★ 若阶段 1 扫描发现算子使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt，MUST 加载 `api-diff-guide` 并设计 subnormal 测试用例。**

## Step 4.0：全量仓 API 精度参考（按需）

如需查阅具体 API 的精度约束或数据类型支持，按 `references/search-rules.md` 路由到全量仓：

- **API 文档**：`$DEVKIT_PATH/docs/zh/api/` 下查找对应 API 的精度说明和 `<cann-filter>` 标签确认 A5 支持情况
- **头文件**：`$DEVKIT_PATH/include/` 下 grep 函数名确认接口声明

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

## Step 4.1：构建 PyTorch 绑定

### 为什么必须构建

CANN OPP 安装的算子包**不会**自动注册到 `torch.ops.npu`。需要额外构建 PyTorch C++ Extension，通过 `EXEC_NPU_CMD` 将 aclnn 接口注册到 PyTorch。

### 唯一标准方式

使用 `torch_aclnn_helper.h` 中的 `EXEC_NPU_CMD` 宏。

**禁止**：
- 手动 aclnn C API（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）
- `import ascend_kernel` 直接加载 .so
- 不使用 `torch_aclnn_helper.h` 自己写绑定

### 构建流程

按照已加载的 `pytorch-binding-build-guide.md` 执行（aclnn 接口名/调度路径确认与 Python 调用规范见 `references/precision-testing/aclnn-interface-guide.md`）：

1. 创建 `op_name/pytorch/` 目录
2. 复制 `references/precision-testing/torch_aclnn_helper.h.template` 到该目录并重命名为 `torch_aclnn_helper.h`
3. 创建 `.cpp` 绑定文件（使用 `EXEC_NPU_CMD` + `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`）
4. 创建 `CMakeLists.txt`（纯 C++ 编译，`LANGUAGES CXX`，链接 `dl`，**不**链接 `libcust_opapi.so`）
5. 编译：
   ```bash
   source ${CANN_SET_ENV}
   export LD_LIBRARY_PATH=vendors_path/op_api/lib:$LD_LIBRARY_PATH
   cd op_name/pytorch && rm -rf build && mkdir build && cd build
   cmake .. && make -j$(nproc)
   ```

### 关键踩坑点

| 踩坑 | 修复 |
|------|------|
| `NPU arch is not supported!!!` | 使用了 ASC 编译器，改为纯 C++ |
| `aclnnXxx not in libopapi.so` | 运行时找不到 libcust_opapi.so，设置 `LD_LIBRARY_PATH` |
| `undefined symbol: _ZN...` | C++ ABI 不匹配，添加 `-D_GLIBCXX_USE_CXX11_ABI=1` |
| `no member named 'is_npu'` | 改用 `x.device().type() == c10::DeviceType::PrivateUse1` |

### 验证绑定可用

```python
import torch
torch.ops.load_library("path/libcustom_ops.so")
# 确认 torch.ops.npu.op_name 可调用
```

## Step 4.2：生成测试用例

基于阶段 1 的算子源码分析摘要生成。

### 用例设计原则

1. **全 dtype 覆盖**：每种 dtype 都必须测试
2. **shape 不要过大**：单个用例元素数不超过 200K
3. **用例总数 ≥ 30**：`(len(TEST_SHAPES) + len(BOUNDARY_VALUES)) * len(SUPPORTED_DTYPES) >= 30`
4. **覆盖条件触发路径两侧**：迁移方案中含条件触发的机制（如仅特定 dtype/shape 命中的跨核同步路径、L1Carry 类跨核协作），用例须覆盖**命中与不命中两侧**——条件触发型卡死只在特定组合暴露，只测正常组合无法发现（参照阶段 1 Step 1.5 的同步协议处置决策确定需覆盖的组合）
5. **排布全覆盖**：用例必须覆盖迁移前算子支持的所有数据排布（input_layout / format 枚举的全部取值，含变长等特殊排布），逐排布给出精度结论。精度无法通过的排布**仍须保留为正式用例并如实报告失败**，禁止将失败排布从测试集合中静默移除

### 用例组成

**Part A：常规 Shape 测试**

| 维度 | 推荐 shape |
|------|-----------|
| 1D | (128,), (1024,), (4096,), (8192,) |
| 2D | (32, 512), (64, 768), (128, 1024) |
| 3D | (8, 16, 64), (4, 128, 256) |

**Part B：边界值测试**

| 算子类型 | 推荐边界值 |
|----------|-----------|
| 有域限制 | 域边界 + 临近值 + 典型值 + 大值 |
| 无域限制 | 0.0, 1.0, -1.0, 100.0, -100.0 |
| 量化类 | 0.0, 正负1.0, 正负最大值, 正负最小非零值 |

**Part C：★ Subnormal 场景测试（条件性 MUST）**

**触发条件**：阶段 1 扫描发现算子使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal **且**输入可能包含 subnormal。否则跳过本 Part。

按 `api-diff-guide.md` §1 设计 subnormal 测试用例：

| 测试维度 | 推荐输入值 | 说明 |
|---------|----------|------|
| FP32 最小正常数 | `np.float32(np.finfo(np.float32).tiny)` | 正常与 subnormal 的边界 |
| FP32 subnormal 正数 | `np.float32(np.finfo(np.float32).tiny / 2)` | 典型 subnormal 值 |
| FP32 subnormal 极小值 | `np.float32(np.finfo(np.float32).tiny / 1e10)` | 极小 subnormal |
| FP16 最小正常数 | `np.float16(np.finfo(np.float16).tiny)` | FP16 边界 |
| FP16 subnormal 正数 | `np.float16(np.finfo(np.float16).tiny / 2)` | FP16 subnormal |
| 混合输入 | 正常值 + subnormal 值混合 | 验证部分元素 subnormal 的处理 |
| 全 subnormal | 全部元素为 subnormal | 验证纯 subnormal 输入 |

**Subnormal 测试注意事项**：
- CPU 参考实现（如 `torch.log` / `torch.exp`）支持 subnormal，输出非零
- A5 NPU 默认 INTRINSIC 模式下 subnormal → 0，预期精度差异
- 若阶段 2 选择了 eps 规避策略，subnormal 输入应被 eps 抬升，输出应接近 CPU 参考
- 若阶段 2 选择了 `PRECISION_1ULP_FTZ_FALSE`，subnormal 输入应正确处理，输出接近 CPU 参考
- 若阶段 2 选择了默认 INTRINSIC（subnormal→0），subnormal 输入输出为 0，这是预期行为，需在报告中标注

## Step 4.3：生成测试脚本

**MUST** 使用 `references/precision-testing/test_op_precision_aclnn_template.py.template` 模板（复制后去除 `.template` 后缀再填充）。

**前置要求**：编写完整测试脚本前，MUST 先按 `references/precision-testing/precision-test-pre-validation-guide.md` 做小规模逐元素数值前置验证（确认 NPU 与 CPU 参考语义一致）。

### 核心规范

1. **NPU 调用方式**：`torch.ops.npu.op_name(...)`（禁止 PyTorch 原生同名接口）
2. **数据生成**：CPU 生成 → `.to(device)` 搬到 NPU
3. **进度输出**：使用 `print(msg, flush=True)`，每 case 显示当前/总数
4. **NaN 检查**：调用算子前检查输入不含 NaN
5. **量化输出**：round 后 clamp 再转整型，用 MaxAbsErr 而非 MERE/MARE

## Step 4.4：执行精度测试

```bash
source ${CANN_SET_ENV}
export ASCEND_RT_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICE}
${PYTHON_PATH} -m pytest test_op_name_precision.py -v --tb=short
```

## Step 4.5：精度标准

**判定标准**（混合容差 + 双门限，真源 `cannbot-skills/ops/ops-precision-standard/SKILL.md`）：逐元素 `|actual - golden| ≤ atol + rtol × |golden|` **且** `matched_ratio ≥ 0.99` **且** `max_abs_error ≤ max_abs_error_limit`。各 dtype 阈值表见 `references/precision-testing/OPS_PRECISION_STANDARDS.md`。

| 输出类型 | 判定标准 |
|---------|---------|
| 整型输出（INT4/8/16） | 0 误差精确匹配（matched_ratio = 1.0） |
| 浮点输出 | 混合容差 + 双门限：matched_ratio ≥ 0.99 且 max_abs_error ≤ max_abs_error_limit |

MERE/MARE 仅作为**分析指标**输出（误差定位用），不作为通过判定依据。

## Step 4.6：精度问题排查

当精度测试失败时，按五阶段流程在本 skill 内排查。

### ★ Step 4.6.0：失败处理前置约束

1. **禁止规避失败**：不得通过修改 threshold、跳过 case、修改输入数据范围、缩减 dtype 列表等方式让失败用例"通过"
2. **禁止跳过分析**：任何 FAIL / NaN / Inf 都 MUST 进入下方根因分析流程，禁止直接标注"已知限制"后跳过
3. **N_FAIL > 0 时禁止输出 GATE-4 PASS**：必须输出 `[GATE-4] PRECISION=N/M_PASS BLOCKED`（M < N），进入根因分析
4. **★ 已归类的非缺陷小差异：用户接受决策通道**：根因分析完成且差异归类为**非缺陷类**（差异不大、非极端错误/公式错误引起，如 dtype 表示粒度、denormal 域相对误差度量放大、标杆精度不匹配）时，MUST 向用户呈现证据（超标元素数/占比/abs 误差范围/归类依据）并**询问用户是否接受**，而不是停留在 BLOCKED 卡住流程：
   - 用户**接受** → GATE-4 输出条件性通过：`[GATE-4] PRECISION=N/M_PASS CONDITIONAL (USER_ACCEPTED: <归类>, <涉及用例数>)`，豁免明细 MUST 在精度报告中如实呈现
   - 用户**不接受** → 继续按归类处置（修复/升级/停止并报告）
   - **禁止**将极端错误（NaN/Inf 根因未定、公式错误、正常值域真实精度缺陷）纳入此通道——这些仍 MUST 走 BLOCKED 与修复流程；询问时优先使用当前平台的结构化交互工具

### ★ Step 4.6.1：NaN/Inf 强制根因分析

**任何 precision / functional test 出现 NaN 或 Inf 输出时，MUST 执行以下根因分析。在根因确定且归类完成前，禁止输出 GATE-4 PASS。**

**MUST 检查以下根因路径**：

| 序号 | 检查项 | 检查方法 |
|------|--------|---------|
| 1 | Division by zero | 追踪所有 Div/Reciprocal 操作，检查分母是否可能为 0 |
| 2 | Overflow | 检查中间结果是否超出 dtype 最大值（FP16 max=65504, FP32 max=3.4e38） |
| 3 | Underflow | 检查中间结果是否低于 dtype 最小可表示值（含 subnormal） |
| 4 | Sqrt/Rsqrt/Div/Reciprocal 输入 | 检查 Sqrt 输入是否为负数、Div 分母是否为 0 |
| 5 | dtype conversion | 检查 Cast 路径是否丢失精度或产生 0（如 FP32→FP16 时小值下溢） |
| 6 | intermediate values | 追踪计算链中间值，定位首次出现 NaN/Inf 的位置 |
| 7 | eps / epsilon / 常量 | 按 `api-diff-guide.md` §1.6 检查 eps/常量在目标 dtype 下是否下溢为 0、是否导致除零 |
| 8 | reference implementation | 检查 CPU 参考实现是否本身有 bug（如 clamp 逻辑、dtype 转换） |

**MUST 判断问题归类**：

| 归类 | 说明 | 对 GATE-4 的影响 |
|------|------|-----------------|
| A5 架构差异 | A5 subnormal 裁剪/指令行为差异导致 | 必须适配修复后重新测试 |
| 原始算子问题 | A2/A3 上同样存在的问题（非迁移引入） | 需说明并评估是否修复 |
| dtype 表示范围问题 | 常量/eps 值在目标 dtype 下不可表示（如下溢为 0） | 需修改常量值或增加 dtype 守卫 |
| 测试设计问题 | 测试输入数据不合理（如使用超出 dtype 范围的值） | 需修正测试设计后重新测试 |
| reference 问题 | CPU 参考实现本身有 bug | 需修正 reference 后重新测试 |
| 其他 | 以上均不适用 | 需详细说明 |

**禁止直接用"已知限制"跳过分析。** 即使最终结论是"不修复"，也 MUST 在报告中记录：根因、归类、是否 A5 特有、为什么不修复、对迁移验收的影响。

### ★ Step 4.6.2：FP16 精度失败强制调试流程

**当 FP16 FAIL 且 FP32 PASS 时，MUST 执行以下调试流程，禁止直接标注"FP16 已知限制"后结束。**

**Phase 1: 误差分析**——先看数据分布

收集失败用例的 shape、dtype、MaxAbsErr/MeanAbsErr/CosineSim，判断误差特征：

| 失败现象 | 最可能原因 | 下一步 |
|----------|-----------|--------|
| FP16 失败，FP32 通过 | arch35/ 未升精度到 FP32 | Phase 2 查 Cast 路径 |
| 输出全零 | CopyOut 未执行 / GM 偏移错 | Phase 2 查 CopyOut |
| 输出含 NaN/Inf | 除零 / 溢出 | **★ Step 4.6.1 强制根因分析** |
| 全部偏差，CosSim约1 | 系统性精度损失 | Phase 2 查升精度 |
| L2 量化精度异常 | CastTrait SatMode 错误 | Phase 2 查量化路径 |
| 周期性/条纹状错误 | tile 边界 / 搬运偏移 | Phase 3 实验 |
| 仅尾部元素错 | 尾 tile 长度 / 对齐 | Phase 2 查尾 tile |
| **★ subnormal 输入输出全零** | **A5 裁剪 subnormal，默认 INTRINSIC 模式** | **Phase 2 查 Subnormal 适配（`api-diff-guide.md` §1）** |
| **★ 仅极小值输入失败** | **未启用 `PRECISION_1ULP_FTZ_FALSE` 或 eps 规避** | **Phase 2 查 Subnormal 适配策略** |
| **★ matched_ratio 未达标但超标元素极少（十万分之几）、MARE（分析指标）远超 rtol** | **denormal/near-zero 域相对误差放大或标杆精度不匹配（见 Step 4.6.3 判别流程）** | **Step 4.6.3 判别流程** |
| **★ int4 量化 matmul 精度异常** | **Mmad 不支持 int4，未 cast 成 int8** | **Phase 2 查兼容性适配（`api-diff-guide.md` §2）** |
| **卡死/超时（aicore timeout 507014），特定 shape/dtype 组合全部触发（如 fp16/bf16 卡、fp32 不卡）** | **跨核同步死锁：同步点集合粒度错配（flagId 配对假设单对、目标平台模式是全部 AIV）** | 查 Step 1.5 同步协议盘点与同步点，按处置决策禁用不可行路径复测；判别与定位方法见 `cube-debug-lessons.md` Part 1 |

> cube 类算子的异常判别与定位方法（死锁/卡死、mask 与消费粒度、取证打印、路径级验证）见 `references/impl/cube-debug-lessons.md`——本文 Phase 1 速查表给出首因判断，详细排查手段在 cube-debug-lessons 对应 Part。

> **注意**：死锁/卡死类问题（同步点集合粒度错配）插桩输出**无法到达**——卡在同步点前的核永远不会执行到插桩语句，即使执行到的核其 printf 也可能因卡死的核未释放流水而无法回传。此类问题不得依赖插桩定位，直接用"触发条件分析 + 禁用不可行路径复测"验证。

**★ FP16 FAIL + FP32 PASS 时，Phase 2 MUST 检查以下项**：

| 检查项 | 检查方法 | 本次 adam_apply_one 的问题对应 |
|--------|---------|-------------------------------|
| Cast 路径 | 检查 FP16 路径是否升精度到 FP32（BF16 路径通常有 Cast 到 FP32，FP16 路径可能没有） | — |
| Accumulation dtype | 检查累加/乘法中间结果是否在 FP16 还是 FP32 计算 | — |
| Intermediate precision | 追踪计算链每一步的中间精度 | — |
| API precision behavior | 检查 Sqrt/Div 在 FP16 下的精度行为 | — |
| Reference precision | 确认 CPU 参考实现在 FP32 还是 FP16 计算 | — |
| Near-zero MARE | 检查是否因接近零的值导致 MARE 放大（MaxAbs 在阈值内但 MARE 超限）——按 Step 4.6.3 判别流程处理 | — |

**Phase 2 完成后 MUST 给出**：
1. 根因（如"FP16 路径不升精度，近零值 MARE 放大"）
2. 是否 A5 特有（如"否，A2/A3 上同样存在"）
3. 是否修复及理由（如"不修复：原始算子设计决定，非迁移引入"）
4. 对迁移验收的影响（如"FP16 路径精度与 A2/A3 一致，迁移未引入退化"）

**如果不修复，MUST 在精度报告中记录上述 4 项，但不允许将 GATE-4 标记为 PASS——除非 N_FAIL == 0。**

**Phase 2: 代码审查**——按已加载的 `l2-guide.md` 中的检查清单排查

**Phase 3: 实验隔离**——控制变量缩小范围
- 实验 A: block_dim=1（多核隔离）
- 实验 B: 固定/规律输入（地址隔离）
- 实验 D: 缩小 shape（边界隔离）

**Phase 4: 插桩定位**——用 `AscendC::printf` / `DumpTensor` 精确定位

**Phase 5: 修复验证**——修复后重新编译安装测试

最多 3 轮调试循环。3 轮后仍失败则停止并向用户报告。

### ★ Step 4.6.3：MARE 爆炸判别流程（matched_ratio 未达标但超标元素占比极低时 MUST 执行）

**适用信号**：matched_ratio 略低于 0.99、CosSim≈1，但超标元素占比极低（典型 0.001%~0.01%，每十万元素个位数），且 MARE（分析指标）远超 rtol 数十倍。此类"爆炸"绝大多数是**度量在特定数值域的放大**而非真实计算错误，禁止未走本流程就下"精度缺陷"结论，也禁止走完本流程后直接放宽阈值。

**判别三步（顺序执行）**：

**第 1 步：标杆精度对齐检查（最优先——排除"假爆炸"）**

将 golden 的输入按算子声明的 dtype 先量化再计算（`q.to(op_dtype)` 后以 fp32/fp64 累加，中间转换点匹配算子声明行为，即"昇腾小算子拼接"类标杆），重算 MERE/MARE：

- 重算后全部通过 → 根因是**标杆精度高于算子契约**（如 fp16 输入经大 scale 放大后，fp64 标杆与任何实现都差 ~30%）
  - 归类：测试设计问题（参考实现精度不匹配）
  - 处置：修正标杆后重测，GATE-4 以修正后结果计
  - 实测案例：flash_attention_score scale=30 场景——NPU vs fp64 标杆 MERE=3.03e-1（看似爆炸），NPU vs 输入 half 化标杆 MERE=3.37e-4（通过）；纯 CPU half 化输入同样产生 3.06e-1 误差，证明误差来自 fp16 输入量化而非 NPU 计算

**第 2 步：denormal 域分桶统计（识别"度量放大"）**

按 `|golden|` 分桶统计相对误差分布（典型桶：[0,1e-3)/[1e-3,1e-2)/[1e-2,1e-1)/[0.1,1)/[1,+∞)）：

- 超标元素**全部**落在 `|golden| < dtype 最小正常数`（fp16: 6.1e-5，fp32: 1.18e-38）的 denormal 域，且其绝对误差 ≤ 数个 ulp（fp16 denormal ulp=6e-8，实测典型 1-8 ulp）→ 根因是**相对误差度量在 denormal 域无判别力**（denormal 域 ulp 固定，值 7e-7 时 1 ulp 差异即 8% 相对误差）
  - 归类：dtype 表示粒度（原始算子问题，A2/A3 同样存在——可用原版二进制同输入对照证明逐位一致）
  - 处置：不修复（提升中间精度=改变算子数值行为，超出迁移范围）；按 Step 4.6.2 记录 4 项结论；GATE-4 按 Step 4.6.0 第 4 条询问用户是否接受——接受则输出 CONDITIONAL（豁免明细入报告），不接受则按 N_FAIL 实际值输出（`PRECISION=N/M_PASS BLOCKED`）
  - 实测案例：lightning_indexer_grad——3 例 FAIL 共 11 个超标元素，全部 |golden|∈[6.6e-7, 2.4e-5]，abs 误差 1-8 ulp，与原版 950 二进制逐位一致
- 超标元素分布在正常值域 → **真实精度缺陷**，进入 Phase 2 代码审查（Cast 路径/累加精度/中间精度）

**第 3 步：对照实验（可选加固）**

同输入跑原版（未迁移）二进制：结果逐位一致 → "非迁移引入"结论升级为实证；结果不同 → 迁移引入，回到 Phase 2 定位差异。

**禁止事项**：
- 禁止跳过第 1 步直接把 MARE 超标当作 NPU 缺陷（标杆精度不匹配是最常见假因）
- 禁止对 denormal 域元素静默剔除后宣称全过——分桶统计与豁免判定 MUST 在精度报告中如实呈现（超标元素数/占比/abs 误差范围）
- 禁止以此流程为名放宽 MERE 阈值或删除失败用例

## Step 4.7：生成精度报告

MUST 生成并在对话中展示（报告格式用 `references/precision-testing/precision_report_template.md`；批量执行+报告自动生成可用 `references/precision-testing/run_precision_report_aclnn_template.py.template`）：
1. 总览表（总用例/通过/失败/通过率）
2. 常规 Shape 测试结果
3. 边界值测试结果
4. **★ Subnormal 场景测试结果**（如适用）
5. 按 dtype 汇总统计
6. 关键发现（3 条以上结论）
7. **★ Subnormal 适配策略验证结论**（如适用：eps 规避 / 模板参数 / 默认 INTRINSIC 的效果对比）

## Gate 输出条件

- [ ] PyTorch 绑定已构建并通过验证
- [ ] 用例数 >= 30
- [ ] 全部 dtype 已测试
- [ ] **★ Subnormal 场景用例已测试**（当算子使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 时）
- [ ] **★ FP16 subnormal 用例已测试**（`np.float16(np.finfo(np.float16).tiny / 2)`，当算子使用上述 API 时）
- [ ] **★ 无 NaN/Inf 输出，或已按 Step 4.6.1 完成根因分析并归类**
- [ ] **★ 无 FP16 FAIL，或已按 Step 4.6.2 完成五阶段调试并给出根因和归类**
- [ ] **★ N_PASS == N_TESTS（N_FAIL == 0）**——全部用例通过，或存在已归类的非缺陷小差异且已按 Step 4.6.0 第 4 条获得用户接受（CONDITIONAL，豁免明细入报告）
- [ ] 精度报告已生成并在对话中展示
- [ ] **★ Subnormal 适配策略验证结论已展示**（如适用）
- [ ] **★ 未通过修改 threshold / 跳过 case / 修改输入数据等方式规避失败**

**全部通过（N_FAIL == 0）→ 输出 `[GATE-4] PRECISION=N/N_PASS` → 迁移完成**

**存在失败（N_FAIL > 0）→ 输出 `[GATE-4] PRECISION=N/M_PASS BLOCKED`（M < N）→ 进入根因分析流程 → 修复后重新测试 → 重新评估 GATE-4**

**禁止行为**：
- 禁止在 N_FAIL > 0 时输出 GATE-4 PASS
- 禁止通过跳过失败 case 让 N_FAIL 变为 0
- 禁止通过放宽 threshold 让失败 case 变为通过
- 禁止通过修改输入数据范围避免触发失败
- 禁止将 NaN/Inf 直接标注为"已知限制"而不执行 Step 4.6.1 根因分析
