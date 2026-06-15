---
skill_name: pypto-intent-understand
---

# Case 1: 将自然语言需求转化为结构化 SPEC.md

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

我想开发一个 GELU 激活算子，请帮我生成算子需求规格文档。算子参考 PyTorch 的 nn.GELU，使用近似版公式：GELU(x) ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715*x³)))，输入 x fp16 [1024, 4096]，输出与输入 shape 相同。

请生成 SPEC.md。

## Expected Output

回复应调用 pypto-intent-understand skill，按工作流完成需求意图理解并生成 SPEC.md。应覆盖以下要点：
- 对输入进行分类（标准参考类），从已知知识中恢复 GELU 标准定义
- 标记信息置信度（高置信度 ✓）
- 展示 ASCII 数据流图，清晰展示输入输出和计算流程
- 展示规格确认清单（算子名称、公式、输入输出规格、动态轴）
- 标注功能优先级（P0/P1/P2）
- 至少确认一次用户意图
- 最终输出 SPEC.md 到当前目录
- SPEC.md 包含完整的结构化字段：算子名称、公式、输入输出规格、dtype、精度要求等

## Expectations
- [contains] SPEC.md
- [contains] 意图理解
- [contains] GELU
- [contains] 数据流图
- [contains] 确认

---

# Case 2: 使用边界与输入分类

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

pypto-intent-understand 在什么情况下触发？它能处理哪些类型的输入描述？如果不能从用户描述中获取完整信息会怎样？

## Expected Output

回复应说明 pypto-intent-understand 的使用边界和触发条件：
- 当用户描述要开发、实现、创建某个算子时触发
- 支持 4 类输入：标准参考类（常见算子/框架 API 参考）、外部材料类（论文/文档链接）、自定义描述类、直接规格类
- 遵循 6 大核心原则：基于事实不猜测、功能优先级明确、可选参数深度分析、信息完整性、至少确认一次、默认值显式披露
- 核心工作流：快速解析 → 可视化确认 → 可选补充 → 输出文件
- 信息不足时主动提问补充，不凭空猜测
- 输出结构化需求文档 SPEC.md，包含完整字段
- 不生成 API 映射（那是 pypto-api-explore 的职责）
- 不生成实现代码（那是 pypto-op-develop 的职责）

## Expectations
- [contains] 标准参考类
- [contains] 外部材料类
- [contains] 自定义描述类
- [contains] SPEC.md
- [contains] 确认
