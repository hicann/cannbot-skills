---
skill_name: pypto-op-develop
---

# Case 1: 基于设计方案实现 Add 算子

## Config
- Max Tokens: 150000
- Ascend Platform: A2

## Prompt

我需要实现一个 Add 算子，开发一个完整的 PyPTO 算子。算子规格如下：

算子名称：Add
数学公式：z = x + y（逐元素加法）
输入：x fp16 [1024, 4096], y fp16 [1024, 4096]
输出：z fp16 [1024, 4096]
设计方案摘要：
- API 映射：使用 pypto.add 实现逐元素加法
- 数据类型：fp16
- Tiling 策略：Vector 类型，使用 set_vec_tile_shapes
- 验证方案：使用 add_golden 作为精度基准

请生成完整的算子实现，包含所有必要的文件。

## Expected Output

回复应调用 pypto-op-develop skill，按开发阶段完成算子实现。应覆盖以下要点：
- 阶段一：检查环境（CANN、pto-isa 源码、device_id 设置）
- 阶段二：并行读取参考文件（execution-constraints.md 及各模板文件）
- 输出 4 个文件：
  - add_impl.py：使用 @pypto.frontend.jit 装饰器，包含 add_wrapper() 函数，调用 pypto.add 或等价操作，完成输出写回
  - test_cases.json：包含 op_name、source、test_cases 列表（含输入输出 shape/dtype）
  - test_add.py：遍历读取 test_cases.json，导入 add_golden 和 add_wrapper，使用 assert_allclose 精度对比，含 [PRECISION_PASS]/[PRECISION_FAIL] 标记
  - README.md：面向调用方，含接口说明、使用示例、约束条件
- 阶段三：测试验证，运行 python3 test_add.py
- 输出执行约束适用项列表

## Expectations
- [contains] add_impl.py
- [contains] test_add.py
- [contains] test_cases.json
- [contains] README.md
- [contains] pypto.frontend.jit
- [contains] assert_allclose

---

# Case 2: 前置输入条件与使用边界

## Config
- Max Tokens: 150000
- Ascend Platform: A2

## Prompt

pypto-op-develop 启动前需要准备哪些输入材料？如果缺少需求规格、设计方案或 golden 参考实现会怎样？它和 pypto-golden-generate 以及 pypto-op-design 的职责如何划分？

## Expected Output

回复应说明 pypto-op-develop 的前置条件和职责边界：
- 需要三类输入材料：
  - 需求规格信息：算子名称、数学公式、输入输出规格（shape/dtype）、支持数据类型、精度要求、服务器类型
  - 设计方案信息：API 映射设计、数据规格设计、Tiling 策略、Loop 结构设计、验证方案、性能指标
  - 参考实现信息：golden 函数名 {op}_golden()、参数签名、输出形式
- 信息不足时逐步提问补充，不会在信息不全时开始编码
- 输出 4 个文件：{op}_impl.py、test_cases.json、test_{op}.py、README.md
- golden / impl / test 必须职责分离，禁止混写
- 与 pypto-op-design 的区分：design 负责设计方案（DESIGN.md），develop 负责编码实现
- 与 pypto-golden-generate 的区分：golden-generate 负责纯 torch 参考实现，develop 使用 golden 进行验证但不生成 golden
- 实现前必须逐项核对 references/execution-constraints.md 的约束清单

## Expectations
- [contains] 需求规格
- [contains] 设计方案
- [contains] golden 参考实现
- [contains] 职责分离
- [contains] execution-constraints
