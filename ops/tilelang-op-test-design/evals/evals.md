---
skill_name: tilelang-op-test-design
---

# Case 1: 算子测试方案设计

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

请介绍 tilelang-op-test-design 技能如何为算子设计测试方案。它包含哪些测试层级（L0/L1/L2/Boundary），每个层级测试什么内容？不需要执行任何工具调用。

## Expected Output

回复应从 design.md 提取算子信息生成多层测试配置，包含 L0 门槛测试、L1 功能测试、L2 异常测试、Boundary 边界测试等不同层级。

## Expectations
- [contains] L0
- [contains] L1
- [contains] L2

---

# Case 2: 多场景测试设计能力

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

这个测试设计 skill 支持哪些场景？如果我已经写好了算子代码但还没有测试，应该使用哪个场景？

## Expected Output

回复应说明本技能支持四种场景：场景 A（从 design.md 设计测试）、场景 B（从 custom/{op}/*.py 算子文件补充测试）、场景 C（用户口头描述生成测试模板）、场景 D（测试覆盖率分析）。如果用户已有算子实现代码但无测试，应使用场景 B，通过读取 custom/{op}/ 目录下的算子实现代码来分析计算逻辑，判断算子类别，分析现有测试覆盖情况，然后补充缺失的测试用例。

## Expectations
- [contains] 场景 A
- [contains] 场景 B
- [contains] 场景 C
- [contains] 场景 D
