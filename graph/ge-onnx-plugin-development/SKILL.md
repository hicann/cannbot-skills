---
name: ge-onnx-plugin-development
description: 使用 ge.onnx_plugin 编写 ONNX Python 插件、把自定义算子接入 GE 时使用。触发场景：PyTorch 自定义算子导出 ONNX、编写 onnx_plugin/parse_node/parse_operator/decompose 回调、注册自定义算子到 GE、atc 编译含自定义算子的 ONNX 模型、设置 ASCEND_CUSTOM_OPP_PATH、算子未注册（E13010/E16002）写法排查。不用于 GE 特性自身代码开发（bridge/registry/parser 改动走常规开发流程）。
---

# ONNX Python 插件开发

辅助用户走通「PyTorch 自定义算子 → ONNX → Python 插件 → atc 编译 → ACL 执行」链路。
插件开发者的职责只有一件事：读取 ONNX 源节点，补充 GE 已创建好的目标算子的属性与端口
（或返回替代子图）——不需要写 kernel，不需要从零创建算子实现。

## 硬规则

1. 生成插件/导出/执行代码必须从本 skill 的 `templates/` 出发填充，禁止凭记忆手写 API；
2. 长尾细节（ES 算子清单、JSON 复合类型全表、`.pyi` 全量签名）必须读 GE 仓权威文档（见文档地图），不凭印象编造；
3. 本 skill 面向插件使用者。用户要求修改 GE 特性自身代码（bridge/loader/registry/parser）时不适用本 skill。

## 入口路由

按用户当前所处阶段选择路径，支持从任意中间步进入：

| 用户带来的 | 路径 |
|-----------|------|
| PyTorch 算子/模型代码（autograd.Function + symbolic） | ① 读 `references/export.md` 写导出脚本 → ② 读 `references/plugin.md` 写插件 → ③ 读 `references/verify.md` 编译执行 |
| 已有含自定义算子的 ONNX 模型 | ① 用 onnx.load 检查 opset_import 与节点 domain（命令见 `references/export.md` §4）→ ② 读 `references/plugin.md` 写插件 → ③ 读 `references/verify.md` |
| 只有算子语义描述 | 先按 `templates/export_onnx.py` 构造 PyTorch demo（forward + symbolic）→ 回到入口 1 |

## 核心约定（动手前自查）

| # | 约定 | 违反后果 |
|---|------|---------|
| 1 | **两侧一致**：symbolic 的域 + custom_opsets 的版本 ↔ 插件的 domain + opsets，取值必须完全一致。`ai.onnx` 允许使用；但不要在 custom_opsets 里给 `ai.onnx` 登记版本（会与 opset_version 互相覆盖，连累全模型官方算子） | E13010/E16002 算子未注册 |
| 2 | **回调选型**：parse_node 优先；节点带 tensor/子图属性或需整体搬运用 parse_operator；GE 无现成算子时加 decompose | 属性读不到或功能缺失 |
| 3 | **端口注册**：动态 IO target（如 PartitionedCall）必须注册端口；静态 IR target（如 Elu）不用。注册顺序 = ONNX 算子输入顺序，位置要占满（可选位置用 register_optional_input） | E19999 Resolve operator IO name failed |
| 4 | **目录隔离**：插件目录只放插件 .py 文件；依赖 torch/numpy/acl 的导出/执行脚本分开放。插件目录内任一 Python 文件语法错误会中断整个编译 | 编译失败且难定位 |
| 5 | **生命周期**：Operator 只在 callback 内有效（返回后失效，不可存全局）；parse_operator/decompose 的 source 只读 | RuntimeError |
| 6 | **opsets 不重叠**：同一 source 只能注册一个插件，不同插件 opsets 不能重叠 | 编译开始即报错退出 |

## 文档地图

| 内容 | 位置 |
|------|------|
| 导出侧写法（symbolic/custom_opsets/属性后缀/两侧一致约定） | `references/export.md` |
| 插件编写（descriptor/三种回调/JSON 结构/端口规则/生命周期） | `references/plugin.md` |
| 编译执行与调试（atc/日志可见性/图dump/ACL） | `references/verify.md` |
| 代码模板（plugin×3/export/执行/run.sh） | `templates/` |
| 用法权威（GE 仓） | `https://gitcode.com/cann/ge/blob/master/examples/onnx_plugin/README.md` |
| API 权威（GE 仓） | `https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/python/ge/onnx_plugin/` |
| ES 构图算子清单（GE 仓） | `https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/es_graph/api/es_python.md` |

长尾细节需反复查证时，可一次性克隆 GE 仓到本地后只读复用：`git clone https://gitcode.com/cann/ge.git`。
注意：`python3 -c "import ge.es.nn ..."` 等 docstring 查证命令依赖已安装 GE Python 包（`import ge` 可用）的环境，与是否克隆仓库无关。
