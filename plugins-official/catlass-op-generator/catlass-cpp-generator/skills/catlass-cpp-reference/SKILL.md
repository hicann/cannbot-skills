---
name: catlass-cpp-reference
description: CATLASS C++ Linear Attention 五阶段工作流的 02 标杆生成。适用于需要维护唯一 PyTorch CPU 标杆、生成 definition、校准精度策略并冻结 golden contract 的算子开发场景。
---

# 02 标杆生成

进入条件：workflow 状态为 `reference` 且 `operator_contract=frozen`。
读取 `workflow/interface-and-golden-contract.md` 和
`workflow/precision-policy.md`。

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
