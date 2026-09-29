---
name: catlass-cpp-interface
description: CATLASS C++ Linear Attention 五阶段工作流的 01 接口确认。适用于需要初始化专用工程、确认并冻结 operator contract，以及创建和校验 workflow.json 的 Linear Attention 算子；不用于普通 CATLASS 算子。
---

# 01 接口确认

开始前确认 selector 返回 `linear_attention`。新工程运行
`scripts/init_operator_project.sh <operator_name>`；继续工程先运行
`scripts/validate_workflow.py --workflow <operator>/docs/workflow.json`。

使用 `catlass-cpp-knowledge query --family linear-attention --compact` 后读取
`workflow/interface-and-golden-contract.md`。它是核对清单，不替代用户说明。

确认目标设备代际，并把 `docs/workflow.json` 的 `target_architecture` 从 `pending` 设置为
`atlas_a2_a3`（A2/A3）或 `ascend950`（A5）。该字段是后续设计与实现的架构分支权威；
设备代际尚未确认时不得进入 reference。

完成 `docs/api.md`，覆盖入口、输入输出顺序、shape、dtype、layout、state、属性、默认值、
mask、fixed/varlen、chunk、tail、padding、head ratio 和非法输入行为。分别记录
`operator_contract` 与 `golden_contract`；材料冲突按公开接口/用户明确说明、测试和示例、
参考实现、论文或相邻实现的优先级请求用户裁决。

只有用户确认接口后才把 `operator_contract` 设为 `frozen`，并原子更新状态为：
`stage=reference`、`golden_contract=provisional`、`issue_type=null`、
`resume_from=null`、`validation_scope=full`，并保留已确认的非 `pending`
`target_architecture`。更新后必须再次运行 validator。
