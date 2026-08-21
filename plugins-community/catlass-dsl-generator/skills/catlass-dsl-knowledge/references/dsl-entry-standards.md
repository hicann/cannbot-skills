# DSL 入库标准

适用于 `knowledge/dsl/`；先执行[公共入库标准](common-entry-standards.md)。

## 取证优先级

固定提交中的实现和测试界定真实接受面，文档解释意图，示例提供可执行模式；三者不一致时记录漂移。

## 强制内容

- **接口**：完整符号、签名、参数/default/enum、返回、异常和导出路径。
- **执行域**：AIC/AIV/host、scope、编译期/运行期值、arch 和版本。
- **Tensor/layout**：logical/storage shape、stride、coordinate、pointer/recast、对齐和 dtype。
- **可执行模式**：必要 import、allocation、scope、copy/compute/sync 与 compile/run 或 IR 检查。
- **组合条件**：与 flag/barrier/allocator/control flow/runtime 的前后置条件。
- **限制与错误**：不支持面、稳定错误或失败阶段。
- **验证层**：区分 Python/AST、TLAIR/lit、build、runtime 和 device correctness。

## 拒绝条件

- 发明关键字、隐式同步/广播、动态 shape 或 layout 行为。
- 只列函数名，或把省略上下文的伪代码标为可运行。
- 从 API 存在推断性能；候选进入 optimization，项目结果进入 learned。

## 验收清单

- [ ] 接口、scope、layout/memory、限制和错误足以离线使用。
- [ ] 代码模式可执行，且说明对应验证层能证明什么。
