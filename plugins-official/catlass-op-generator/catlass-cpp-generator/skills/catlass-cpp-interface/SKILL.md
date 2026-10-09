---
name: catlass-cpp-interface
description: CATLASS C++ 已注册算法族五阶段工作流的 01 接口确认。用于初始化专用工程、确认并冻结 operator contract，以及创建和校验 workflow.json。触发条件：selector 已返回 linear_attention、block_sparse_attention 或 sparse_flash_mla，需要初始化算子工程或继续接口确认。
---

# 01 接口确认

开始前确认 selector 返回 `linear_attention`、`block_sparse_attention` 或 `sparse_flash_mla`。新工程运行
`scripts/init_operator_project.sh <operator_name> --algorithm-family <selector结果>`；继续工程先运行
`scripts/validate_workflow.py --workflow <operator>/docs/workflow.json`。

使用 `catlass-cpp-knowledge query --family <对应家族> --compact` 后读取
`workflow/interface-and-golden-contract.md`。它是核对清单，不替代用户说明。

`sparse_flash_mla` 先完整读取 `operator/sparse-flash-mla/computation.md`、
`interface.md`、`metadata.md`；普通 SWA 同时核对 `interface.md` 中的一般 SWA 合法域。
主算子与独立 metadata 的接口、输出、ValueDepend 和真实输入依赖分别冻结；用户已提供明确
golden/测试脚本时据此完成契约，无需重复征求同一授权。该家族目前只冻结
`target_architecture=atlas_a2_a3`。

确认目标设备代际，并把 `docs/workflow.json` 的 `target_architecture` 从 `pending` 设置为
`atlas_a2_a3`（A2/A3）或 `ascend950`（A5）。该字段是后续设计与实现的架构分支权威；
设备代际尚未确认时不得进入 reference；`block_sparse_attention` 的 Arch22 知识同样仅支持 `atlas_a2_a3`。

完成 `docs/api.md`，覆盖入口、输入输出顺序、shape、dtype、layout、state、属性、默认值、
mask、fixed/varlen、chunk、tail、padding、head ratio 和非法输入行为。分别记录
`operator_contract` 与 `golden_contract`；材料冲突按公开接口/用户明确说明、测试和示例、
参考实现、论文或相邻实现的优先级请求用户裁决。

只有用户确认接口后才把 `operator_contract` 设为 `frozen`，并原子更新状态为：
`stage=reference`、`golden_contract=provisional`、`issue_type=null`、
`resume_from=null`、`validation_scope=full`，并保留已确认的非 `pending`
`target_architecture`。更新后必须再次运行 validator。
