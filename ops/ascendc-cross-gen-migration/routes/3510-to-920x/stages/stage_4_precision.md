# 阶段 4：契约、数值与位宽边界验证

前置：`[GATE-3] BUILD=PASS INSTALL=PASS`。产物为 `${EVIDENCE_DIR}/precision_report.md`、逐例输出及实际调用链证明。

## LOADED 检查点

公共精度资料从 skill 根目录的 `references/precision-testing/` 读取（相对本阶段文件为 `../../../references/precision-testing/`）：

- [pytorch-binding-build-guide.md](../../../references/precision-testing/pytorch-binding-build-guide.md)、[torch_aclnn_helper.h.template](../../../references/precision-testing/torch_aclnn_helper.h.template)：通过 EXEC_NPU_CMD 注册并调用 `torch.ops.npu.<op>`。
- [OPS_PRECISION_STANDARDS.md](../../../references/precision-testing/OPS_PRECISION_STANDARDS.md)：按 dtype 与算子契约使用标准；详细阈值真源为 `ops-precision-standard`。
- [precision-test-pre-validation-guide.md](../../../references/precision-testing/precision-test-pre-validation-guide.md)、[test_op_precision_aclnn_template.py.template](../../../references/precision-testing/test_op_precision_aclnn_template.py.template)：先小规模逐元素核对，再运行测试电池。
- [tiling-and-memory-guide.md](../references/tiling-and-memory-guide.md) 及计划命中的 DMA/Cube/量化专题。

逐文件记录 LOADED、版本和实际适用条件。

## 绑定 dtype 检查

先核对生成绑定的 dtype 映射，不能假设公共 helper 覆盖当前 PyTorch 的全部枚举。涉及 FP8 时，读取本路线 [precision-dtype-map.h.template](../references/precision-dtype-map.h.template)，仅在本次生成的绑定副本中应用：

1. 核对实际 PyTorch `c10/core/ScalarType.h` 和 CANN `acl/acl_base.h` 包含的类型声明，确认 `Float8_e5m2 / Float8_e4m3fn` 分别对应 `ACL_FLOAT8_E5M2 / ACL_FLOAT8_E4M3FN`。
2. 将模板复制为绑定目录的 `precision_dtype_map.h`；保留公共 helper 的映射宏，用 `#include "precision_dtype_map.h"` 替换其 `kATenScalarTypeToAclDataTypeTable` 数组定义，将 tensor、scalar 和 ScalarType 三处查表统一替换为 `MigrationAclDataType(...)`。按命名枚举转换，不按枚举序号排列数组。
3. 用实际头文件编译检查本算子全部契约 dtype；FP8 不得映射为 `ACL_FLOAT`。FNUZ 等未声明对应关系的类型必须明确拒绝，不能按名称近似映射或默认为 FP32。所需类型缺声明或转换未验证时，先解决绑定支持，保持精度阶段 BLOCKED，不缩减 dtype 覆盖。
4. 映射检查通过后，仍须用下面的真实绑定用例验证张量 dtype、存储字节数和调用链；单独的转换检查不能代替算子精度验收。

## Step 4.1：确认测到迁移实现

构建实际 aclnn 的 EXEC_NPU_CMD C++ 绑定，加载 Stage 3 记录的库与自定义环境，用最小合法输入调用。记录实际库路径、符号及 tiling/kernel 路由证据；可结合目标工具链支持的运行日志、trace 或一次 profiler 调用定位，不凭输出数值正确就认定命中自定义实现。无法归因则先修复加载链。

CPU golden 可使用 PyTorch/NumPy；被测 NPU 路径必须调用迁移实现，不能以 PyTorch 同名接口替代。

## Step 4.2：执行完整计划

| 电池 | 覆盖要求 |
|---|---|
| 常规 | ≥30 个 case，覆盖全部契约 dtype/layout、形状分支、空/标量/尾部等适用边界 |
| 数值 | 输入与中间转换、舍入/饱和、极小/极大值；NaN/Inf/零值按原契约设计，期望特殊值与异常新引入的特殊值分开判断 |
| 位宽 | 对实际使用的索引、count、repeat、stride 等字段测临界点前/等于/后；uint16 关注 65535/65536/65537，但最大索引与元素数量分别分析 |
| 资源 | 满 tile/尾 tile、工作预算临界、UB 扩大后触发的较大 tile，覆盖新增 offset/mask/scale 等临时区 |
| 索引/DMA | 有效域边界、非法索引原有处理、重复索引、accumulate、适用/不适用 shape 与回退 |
| packed dtype | 字节数、打包次序、奇数尾部、scale shape/寻址与量化边界；未支持的输入仍按契约拒绝 |
| 确定性 | 仅对原契约要求的路径复跑并验证，保留输入、seed、输出差异 |

普通 case 优先小规模；必须触发的大 shape 专项允许超过 200K 元素。先计算内存预算，分批运行并设置符合环境的超时；字段临界组合无法合法运行时，用 host tiling 测试验证拒绝/分块，再用合法 kernel case 验证对应路径，不强行提交非法硬件参数。

## Step 4.3：失败与标杆复核

有差异先核对参考公式、dtype 量化点、layout 和累加次序，再定位迁移实现；必要时对照原版实现或独立数学参考。异常 NaN/Inf、越界、卡死、计数回绕要形成根因与修复证据。不能删 case、缩 dtype/layout、放宽阈值或仅跑子集换取 PASS。

## Gate

报告列出 `SOC / TARGET_ARCH`、N_TESTS/N_PASS/N_FAIL、完整 case 状态、逐类误差/特殊值/确定性结果、实际调用链和资源/位宽专项证据。9201/9202 双目标分别执行并统计；一侧通过不替代另一侧。

- 全部计划 case 执行且 N_FAIL=0：`[GATE-4] PRECISION=N/N_PASS`，填实际总数。
- 任一 FAIL、SKIPPED、NOT_RUN 或加载链未证实：`[GATE-4] PRECISION=N/M_PASS BLOCKED` 并列明原因。
- 用户明确接受已归因的差异时只能报告 `CONDITIONAL USER_ACCEPTED`，保留真实统计；本路线不将该状态作为全量 PASS 或进入自动特性保留的依据。
