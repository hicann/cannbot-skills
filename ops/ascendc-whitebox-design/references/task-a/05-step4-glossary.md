# 变量含义表

对应 Task A 执行步骤 4：构建路径清单时同步收集变量含义信息，步骤 8 与 `S2P1_path_list.json` 同步写入 `${output_dir}/S2P1_tiling_glossary.md`。

步骤 6-7 可能因 dead 路径回收导致变量增减，词汇表以步骤 8 写入时的最终状态为准。

---

## 格式

Markdown 表格，表头固定：

```markdown
# Tiling 变量含义表

> 算子：{op_name}
> 数据来源：tiling 源码变量提取

| tiling_var | semantic_name | category | source | type | desc |
|------------|--------------|----------|--------|------|------|
| `srcDim_` | x_shape_dim | input_variable | tiling.cpp:L55 | int | 输入tensor x的空间维度 |
```

---

## 规则

1. `tiling_var`：tiling 源码中的原始变量名，与 conditions 和三分类列表中的变量名严格一致
2. `semantic_name`：人类可读的语义名称，格式 `{参数名}_{属性}`（如 `x_shape_dim`）。仅用于文档可读性，不参与任何下游流程
3. `category`：与变量三分类一致（input_variable / caller_option / internal_variable）
4. `source`：tiling 源码中该变量首次赋值或声明的文件:行号。当变量通过框架 API 管理（如 `SetTilingKey()` + `TILING_KEY_IS()`）且不存在对应的源码局部变量名时，标注为 `framework_api`
5. `type`：变量的数据类型（int / float / bool / enum）
6. `desc`：一句话描述变量的含义
7. 三分类列表（`input_variables` / `caller_options` / `internal_variables`）中出现的每个变量都必须在词汇表中有一条记录
8. 按 category 分组排列（input_variable → caller_option → internal_variable），组内按 tiling 源码出现顺序排列
