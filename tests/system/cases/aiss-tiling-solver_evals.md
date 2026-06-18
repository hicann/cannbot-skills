---
skill_name: aiss-tiling-solver
eval_mode: text
---
# Case 1: AISS-TilingSolver 工具使用流程

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

如何使用 AISS-TilingSolver 工具为 Ascend C 算子求解最优 tiling 参数？请说明完整的使用步骤。可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 AISS-TilingSolver 的完整使用流程：下载 CLI 工具（tiling_solver 和 platform_info 两个二进制文件）；运行 platform_info 采集硬件平台参数；构造输入 JSON 文件（填入算子类型、形状、数据类型及采集到的硬件参数）；运行 tiling_solver input.json 求解；解读输出 JSON 中的最优 tiling 参数（如 base_m、base_n、base_k 等）。

## Expectations
