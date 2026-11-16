# kernel 接入契约

仅在 kernel 尚未完成或双方接口需要对齐时读取。本文件维护契约字段和交付格式；接口选型的理由与例外统一见 [场景选型](scenarios.md)。

## 契约字段

| 字段 | 需要确认的内容 |
|---|---|
| 执行环境 | Host/Device、SoC、CANN/GE 版本、主机 ABI、交付件语言 |
| 图形态 | 静态图、动态图或两者；是否需要按轮 Host 决策 |
| 编译时机 | 图编译前、Compile 回调、运行期；编译器与依赖如何提供 |
| 参数 ABI | 参数顺序、标量宽度、对齐、地址/值、workspace、tiling 数据、FFTS 等隐含参数 |
| 输出 | dtype/format/shape；数据或值依赖；分配上界与实际 shape 回填方法 |
| 产物 | 源码、device ELF、npubin、共享库、JIT；入口符号、magic/coreType、加载方式 |
| 部署 | 在线或离线；bin、共享库、常量、索引表和 Python 依赖的携带/恢复方式 |
| 缓存 | 所有影响产物的 shape/dtype/format/attrs、平台、源码/依赖与编译选项 |
| 验收 | 输入范围、参考计算、误差、空张量/尾块、多轮换址、多模型或独立进程部署 |

kernel 未编译时允许标记产物字段为“待验证”，不要求先完成 kernel 才能咨询接口。
已有产物时从 `.json` 和 device ELF 核对，不从源码标注推定入口或核类型。

## 可复用产出

```yaml
op_type: AddCustom
backend: device
frontend: ge
implementation: cpp
interfaces: [EagerExecuteOp, ShapeInferOp]
graph_modes: [dynamic]
platform: {soc: 待确认, cann: 待确认, host_arch: 待确认}
compilation: precompiled
artifact: {kind: npubin, entry: 待验证, loading_metadata: 待验证}
args:
  - {name: x, kind: input_address, index: 0}
  - {name: y, kind: input_address, index: 1}
  - {name: z, kind: output_address, index: 0}
output_shape: same_as_x
deployment: online
cache_key: 按实际特化条件定义
verification: [build, registration, numerical, repeated_execution]
open_items: [参数标量及隐含参数, 支持的dtype和shape范围]
```

只输出与任务有关的字段，已有信息直接填入，不将此格式变为逐项询问用户的表单。

## 两个检查示例

- 声明式依次追加 x/y/z，kernel 却期望 x/z/y：修正一方的参数顺序，不能靠输出 shape 推导补救。
- Compile 产物引用临时目录中的 `.so`，离线机不存在该路径：明确分发或序列化策略，用独立进程且移走构建目录的部署验证确认依赖完整。
