---
skill_name: torch-npugraph-ex-template
---

# Case 1: 生成标准 MRE 模板

## Config
- Eval Mode: file_based
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

请为我生成一个 npugraph_ex 的标准 MRE 代码模板，包含完整的 import、模型定义和编译配置。

## Expected Output

生成的模板应包含：import 部分（torch、torchair 等）、模型定义、infer_one_step 函数、if __name__ == "__main__" 入口。应使用 backend="npugraph_ex"、fullgraph=True、dynamic=False 配置。变量命名应遵循固定约定（config、compiled_model、dummy_input、output）。

## Expectations
- [file_list] *.py


---

# Case 2: 生成编译缓存模板

## Config
- Eval Mode: file_based
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

我需要一个带编译缓存的 npugraph_ex MRE 模板，包括 enable_compile_cache 的配置。请帮我生成。

## Expected Output

生成的模板应包含 npugraph_ex compile cache 的配置，包括 enable_compile_cache 的使用。应包含完整的模板三部分结构（imports、infer_one_step、main），变量命名符合约定。

## Expectations
