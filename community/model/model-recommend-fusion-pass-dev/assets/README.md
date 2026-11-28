# <Pattern> Pass

## 功能

识别图内 `<结构模式>`，改写为 `<目标算子/结构>`，减少中间张量往返与下发次数。

## 分类

自定义 Pass（图优化）。

## 实现原理

- 遍历 `graph->GetAllNodes()` 收集目标节点。
- 匹配上下游与常量输入（`GetInDataNodesAndPortIndexs` / `GetOutDataNodesAndPortIndexs`）。
- `AddNodeByOp` 构造新算子，`AddDataEdge`/`RemoveEdge` 重连，`RemoveNode` 删旧节点。
- 注册：`REGISTER_CUSTOM_PASS("<Pattern>Pass")`，阶段 `kAfterInferShape`。

## 工程结构

```
<pattern>_pass/
├── CMakeLists.txt
├── include/pass_utils.h
└── src/<pattern>_pass.cpp
```

## 安装使用

```bash
mkdir build && cd build && cmake .. && make
cp lib<pattern>_pass.so ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/custom_fusion_passes/
# 验证：bash demo/run.sh <id> --dump_graph
```

## 适配版本

- CANN：<version>
- 芯片：<ascend910b / ascend910_93 / ...>
