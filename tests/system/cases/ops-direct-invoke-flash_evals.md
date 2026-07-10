---
team_name: ops-direct-invoke-flash
eval_mode: text
---

# Case 1: Flash 版直调开发的核心流程

## Config
- Max Tokens: 500000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: ascendc-direct-invoke-template;ascendc-env-check;gitcode-toolkit

## Prompt

我有一段数学公式，想在昇腾 NPU 上实现为核函数，ops-direct-invoke-flash 团队适合做这个吗？它的开发流程是怎样的？

## Expected Output

回复应覆盖以下要点：
1. ops-direct-invoke-flash 适用于从 CPU 函数、数学公式、代码片段或文本描述出发构建并验证新的 Ascend NPU 核函数
2. 核心流程包含：环境检查 → 设计 → 开发 → 测试 → 验收，覆盖从规格到经验证核函数的完整路径
3. 默认在 operators/ 目录下开发，支持使用 /ops-direct-invoke-flash 技能
4. 输出经验证的 NPU 核函数

## Expectations

- [contains] flash

---

# Case 2: 信息不足时主动确认

## Config
- Max Tokens: 500000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想用 Flash 模式开发一个 NPU 核函数，先告诉我你需要什么信息，不用开始开发。

## Expected Output

回复应主动确认必要信息：待实现的数学公式或算法描述、输入输出数据类型和格式、目标芯片型号等规格信息，而不是在缺少这些关键信息的情况下直接开始开发

## Expectations
