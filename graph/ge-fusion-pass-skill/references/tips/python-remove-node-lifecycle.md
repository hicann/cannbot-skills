# tip: Python remove_node 后节点对象即失效

> 📎 导航落点：`references/interface-catalog.md` §二（Python pass 接口）。本文件仍是该生命周期纪律的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发。

## 症状

Python pass 里 `graph.remove_node(node)` 之后，又访问 `node.name` / 输入输出 / 属性，导致空引用、崩溃或读到脏数据。

## 根因

Python Graph 调用 `remove_node` 后，被删除的节点对象**立即视为无效引用**——底层句柄已失效，Python 侧引用不再指向有效节点。

## 硬性做法

删除前先缓存好需要的信息，删除后只用缓存值：

```python
# 删除前缓存
name = node.name
op_type = node.type
dbg = describe(node)   # 需要的调试信息

# 再删除
graph.remove_node(node)

# 删除后：只用 name / op_type / dbg，绝不再碰 node.*
logging.info("removed node %s (%s): %s", name, op_type, dbg)
```

删除后**不得**再访问 `node.name`、输入输出或属性。

## 自查

- 每处 `remove_node` 前，要用的 name/类型/调试信息是否都先缓存了？
- 删除之后的代码里，还有没有对已删 `node` 的 `.name`/输入/输出/属性访问？
