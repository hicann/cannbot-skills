# 源码文件索引

需要源码级追溯（如验证某个判定场景的实现逻辑）时使用本索引。源码仓路径必须先向用户确认，行号结论必须在实际源码中验证后才能写入报告。

## GE 核心文件

| 文件 | 关键函数 | 功能 |
|---|---|---|
| `compiler/graph/partition/dynamic_shape_partition.cc` | IsUnknownShapeNode | 算子 unknown 判定主入口 |
| `compiler/graph/partition/dynamic_shape_partition.cc` | IsNodeSupportNoTiling | No tiling 支持判定 |
| `compiler/graph/partition/dynamic_shape_partition.cc` | JudgeUnknownShapeForTilingDependNode | TilingDepend 判定 |
| `compiler/graph/partition/dynamic_shape_partition.cc` | IsNodeSupportAddrRefresh | 地址刷新支持判定 |
| `graph_metadef/graph/utils/node_utils.cc` | GetNodeUnknownShapeStatus | Shape unknown 查询 |

## FE 核心文件

| 文件 | 功能 |
|---|---|
| `engines/cpu_engine/common/aicpu_graph_optimizer/aicpu_graph_optimizer.cc` | DT_STRING 标记 |
| `engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_graph_optimizer.cc` | HCCL 算子标记 |
| `engines/dvpp_engine/common/dvpp_optimizer.cc` | DVPP 算子标记 |
| `engines/nn_engine/optimizer/graph_optimizer/op_setter/op_setter.cc` | ACLNN 算子标记 |
