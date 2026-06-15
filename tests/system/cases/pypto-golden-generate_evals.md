---
skill_name: pypto-golden-generate
---

# Case 1: 生成 Add 算子的 Golden 参考实现

## Config
- Max Tokens: 150000
- Ascend Platform: A2

## Prompt

我有一个 Add 算子，请帮我生成 golden 参考实现。算子规格如下：

算子名称：Add
数学公式：z = x + y（逐元素加法）
输入：x fp16 [1024, 4096], y fp16 [1024, 4096]
输出：z fp16 [1024, 4096]
典型配置：
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | 无 | x[1024,4096], y[1024,4096] | z[1024,4096] | 标准配置 |

请生成 add_golden.py。

## Expected Output

回复应调用 pypto-golden-generate skill，按工作流完成 golden 参考实现生成。应覆盖以下要点：
- 从算子规格中提取算子名称、公式、输入输出规格
- 检查规格字段完整性，确认必须字段齐全
- 基于模板 templates/golden-template.py 生成 add_golden.py
- 函数名称为 add_golden()，使用纯 PyTorch 实现（torch.add），禁止引入 pypto
- 包含 _validate() 验证函数：典型 case 验证、值域检查、数值稳定性检查
- 包含 if __name__ == "__main__": _validate() 入口
- 通过 python3 add_golden.py 执行验证，exit code 为 0
- 展示验证报告（含 PASS 结果）

## Expectations
- [contains] add_golden
- [contains] torch.add
- [contains] _validate
- [contains] golden

---

# Case 2: 使用边界与触发条件

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

pypto-golden-generate 和 pypto-op-develop 中的 golden 生成有什么不同？什么情况下我应该单独使用 golden 生成 skill，什么情况下可以直接走完整的算子开发流程？

## Expected Output

回复应说明 pypto-golden-generate 的使用边界和触发条件：
- 专注于生成纯 PyTorch golden 参考实现，输出 {op}_golden.py，导出 {op}_golden() 函数
- 基于固定模板 templates/golden-template.py 生成
- 仅用于生成 golden 参考代码，不涉及 PyPTO 算子实现
- 包含完整的验证机制：语法检查、形状一致性、函数签名、值域检查、数值稳定性验证
- 支持自动修复策略（最多 3 次尝试）
- 纯 torch 实现，禁止引入 pypto
- 当只需要验证 golden 逻辑而不需要完整实现时单独使用
- 当已有设计方案需要快速构建验证基准时使用
- 需要算子名称、数学公式、输入输出规格等必须字段才能生成

## Expectations
- [contains] 纯 PyTorch
- [contains] 参考实现
- [contains] golden
- [contains] 验证
