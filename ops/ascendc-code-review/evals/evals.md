---
skill_name: ascendc-code-review
eval_mode: text
---
# Case 1: 代码检视工作流路由

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ascendc-code-review 技能在接收到代码检视请求时，如何根据用户输入选择合适的工作流？请介绍该技能支持的主要工作流类型以及各自的路由逻辑。不需要执行任何工具调用。

## Expected Output

回复应介绍 ascendc-code-review 的主要工作流类型及其基本的路由逻辑。

---

# Case 2: 代码检视检查项与规范

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在使用 ascendc-code-review 技能进行 Ascend C 代码检视时，主要检查哪些方面的内容？有哪些检视规则和规范文档可以参考？检视完成后如何输出结果？不需要执行任何工具调用。

## Expected Output

回复应介绍代码检视的主要检查维度（安全编码、API 最佳实践、性能约束等）、可参考的规范文档，以及检视结果的输出方式。

---

# Case 3: Ascend C 代码检视请求（正向看护）

## Config
- Max Tokens: 320000
- Max Tokens (deepseek-v4-flash): 576000
- Max Tokens (glm-5): 544000
- Ascend Platform: A2
- Distractor skills: ascendc-task-focus;ascendc-st-design

## Prompt

请帮我快速检视以下 Ascend C 代码是否存在问题：

```cpp
__aicore__ void Add(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    LocalTensor<half> xLocal;
    LocalTensor<half> yLocal;
    LocalTensor<half> zLocal;
    DataCopy(xLocal, xGM);
    DataCopy(yLocal, yGM);
    Add(zLocal, xLocal, yLocal);
    DataCopy(zGM, zLocal);
}
```

请加载 ascendc-code-review 技能完成检视。

## Expected Output

回复应对提交的 Ascend C 代码给出检视意见：涵盖安全编码、API 使用、边界条件等检视维度。

---

# Case 4: 通用 Python 代码风格咨询（负向看护）

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请帮我 review 以下 Python 代码是否符合 PEP 8 规范，有没有潜在的性能问题：

```python
def processData(n):
    result=[]
    for i in range(n):
        temp = i*2
        result.append(temp+1)
    return result
```

我不需要 Ascend C 相关的内容。

## Expected Output

回复应关注 Python 代码规范和性能优化，指出命名风格（如 snake_case）、列表推导式等可改进的点。

