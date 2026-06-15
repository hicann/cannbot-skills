---
skill_name: tilelang-op-design
---

# Case 1: 算子设计文档生成

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

请介绍 tilelang-op-design 技能生成算子设计文档时需要包含哪些核心设计环节，以及编程模式选型（Developer/Expert）如何选择？不需要执行任何工具调用。

## Expected Output

回复应说明算子设计文档的核心设计环节：编程模式选型、API 映射设计、内存层级规划、Tiling 策略等。应解释 Developer/Expert 模式的区别和适用场景。

## Expectations
- [contains] 编程模式
- [contains] Developer
- [contains] Tiling
- [contains] API 映射

---

# Case 2: 算子设计输入的必填信息

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

设计一个 TileLang-Ascend 算子需要提供哪些信息？如果信息不完整会怎么处理？

## Expected Output

回复应列出必需信息：算子名称、数学公式、输入张量规格（shape/dtype）、输出张量规格（shape/dtype）、编程模式偏好（Developer/Expert/混合）。推荐信息包括典型配置、参考实现、性能目标、动态轴说明。信息不完整时 skill 会通过逐一提问补全，每次只询问一个字段。若被 orchestrator 调度且字段已提供则跳过提问环节。

## Expectations
- [contains] 算子名称
- [contains] 数学公式
- [contains] 输入张量规格
- [contains] 编程模式
