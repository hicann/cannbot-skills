---
name: catlass-cpp-reference
description: CATLASS C++ 已注册算法族五阶段工作流的 02 标杆生成。用于维护唯一 PyTorch CPU 标杆、生成 definition、校准精度策略并冻结 golden contract。触发条件：workflow 处于 reference 阶段且 operator_contract=frozen。
---

# 02 标杆生成

进入条件：workflow 状态为 `reference` 且 `operator_contract=frozen`。
读取 `workflow/interface-and-golden-contract.md` 和
`workflow/precision-policy.md`。

`sparse_flash_mla` 使用家族 `computation.md`、`validation.md` 和本次用户原始
golden/ATK 脚本区分数学参考、低精度 cast 路径与正式验收。原 ATK 的阈值和输入不能因
DUT 误差而放宽；辅助诊断阈值不替代原双标杆。metadata 另验证调度表的任务覆盖及
本次有效长度、页表、layout 变化后的重新生成。

`reference/reference.py` 是唯一可编辑标杆源码。修改后运行：

```bash
python scripts/generate_definition.py --source <reference.py> --template <definition.template.json> --output <definition.json>
python scripts/validate_reference.py --source <reference.py> --definition <definition.json>
```

正式标杆使用纯 PyTorch 在 CPU 上以固定版本、种子、输入和计算顺序执行。
`precision-policy.json` 必须按 dtype、输出值域、shape 和有效区域校准，不能直接把模板阈值当作
当前算子的验收结论。覆盖最小、常用、非对齐、fixed/varlen、chunk、tail、padding、state、
head ratio 和非法输入，并在 `docs/validation.md` 留下至少一种实际运行证据。

用户确认 golden 语义后，设置 `golden_contract=frozen`、`stage=design`，清空问题字段，
保持 `validation_scope=full`，再运行 workflow validator。
