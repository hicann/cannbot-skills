# 机制分诊

仅在用户未明确机制、或已有工程混合注册时读取。按材料和需求判定，不按 kernel 语言判定。

## 目录

- [路径与出口](#路径与出口)
- [已有代码判别](#已有代码判别)
- [新需求判别](#新需求判别)
- [PyTorch 入口](#pytorch-入口)

## 路径与出口

| 路径 | 典型材料 | 处理 |
|---|---|---|
| 直调 | 单测直接 launch kernel | 对应 kernel Skill，不经 GE 图 |
| Tiling 托管（路径 2） | op_host/op_kernel、IMPL_OP + .Tiling/.TilingParse | ops-registry-invoke 或 ascendc-registry-invoke-template |
| 自组织交付件（路径 3） | BaseCustomOp / ge.custom_op、自备产物或 Host 计算 | 本 Skill |
| ONNX 映射 | 将节点解析为已有 GE 算子 | ge-onnx-plugin-development |
| PyTorch 纯 Python 注册 | torch.library + npugraph_ex，不提供 GE 实现 | torch-custom-ops-guide |

这些出口在当前仓库的 `plugins-official/`、`ops/`、`graph/` 中查找。
若目标环境未安装对应 Skill，说明所需能力并给出已有工程入口，不假定可以调用缺失技能。

## 已有代码判别

1. 实际使用 `.Tiling()` / `.TilingParse()` 的标准 Ascend C 工程，默认沿用托管路径。
2. 存在 BaseCustomOp 派生类与 `REG_AUTO_MAPPING_OP` / `REG_OP_BACKEND`，或 ge.custom_op 装饰器，则检查本 Skill 的能力组合。
3. 同时出现两套注册时检查各宏用途，不能只凭 `IMPL_OP` 判为托管：
    - `IMPL_OP().InputsDataDependency()` 可补充值依赖声明。
    - `IMPL_OP().InferShape()/InferDataType()` 可注册推导函数。
    - 新工程默认用 `ShapeInferOp`；已有合法推导不因风格偏好而强制改写。

`REG_OP` 是两条路径共用的原型注册，不是实现注册。
`IMPL_OP` 注册函数实现，BaseCustomOp 注册类工厂；原型与实现分开核对。
官方混合推导示例：`examples/custom_op/ascendc_add_custom/add_custom_kernel/custom_op.cpp`。

## 新需求判别

先看用户是否已选定 GE BaseCustomOp；已选定则直接讨论能力，不反复要求证明选择。
否则结合下列信息给默认方案：

- 现成算子工程、标准 tiling 和一次 launch 足够：推荐沿用托管工程。
- 自备可加载产物、Host 计算、自定义参数、多 kernel/辅流、编译或序列化控制：评估自组织交付件。
- Ascend C 源码或 `.o + .json` 两条路径均可能，文件后缀不足以决策。
- kernel 未就绪时先记录 [契约](kernel-contract.md)，再交给对应 kernel 开发能力。

托管路径也有扩展执行机制，不能概括为“永远只能一次 launch”；本 Skill 只比较标准托管范式，不宣称其他语言只能以某一种机制接入整个 CANN 生态。

## PyTorch 入口

先检查现有 converter、构建脚本和产物，再确定是否还缺交付件：

- 手写 BaseCustomOp C++：构建交付件并配置 OPP，torchair converter 负责 FX→GE 映射。
- 使用生成工具：检查它实际生成的原型、Python callback 与加载方式，避免重复生成 C++ 实现。
- 只做 torch.library + npugraph_ex 注册：转 PyTorch 侧 Skill。

“用了 torchair”既不意味着自动生成交付件，也不意味着交付件必须是 C++。
