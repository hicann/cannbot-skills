# <Op> 自定义算子

## 功能

实现 NPU 算子库缺失的 `<功能描述>`，输入 `<...>`，输出 `<...>`。

## 分类

自定义算子（AscendC 内核）。

## 实现原理

- `op_host/<op>_def.cpp`：`OP_ADD` 声明输入输出/属性与 AICore 配置。
- `op_host/<op>_infershape.cpp`：输入→输出形状推导。
- `op_host/arch22/<op>_tiling.cpp`：按行/块分核，`SetBlockDim` + `ASCENDC_TPL_SEL_PARAM` 派发。
- `op_kernel/arch22/<op>.h`：AscendC 内核 `Init`/`Process`。
- `op_api/`：ACLNN 单算子封装（可选）。

## 工程结构

```
<op>/
├── CMakeLists.txt
├── build.sh
├── op_host/{<op>_def.cpp, <op>_infershape.cpp, arch22/<op>_tiling.cpp}
├── op_kernel/{<op>_arch22.cpp, arch22/{<op>.h, <op>_tiling_data.h, <op>_tiling_key.h}}
├── op_api/{aclnn_<op>.h, aclnn_<op>.cpp, <op>.h, <op>.cpp}
├── op_graph/{CMakeLists.txt, <op>_proto.h}
├── framework/                # 可选：配套 onnx 插件
└── tests/ut/                 # gtest UT（按需补充）
```

## 安装使用

```bash
bash build.sh --soc=ascend910b
./build/custom_opp_<os>_<arch>.run --quiet
source $ASCEND_OPP_PATH/vendors/<op>_custom/bin/set_env.bash
# 验证：bash demo/run.sh <id> --verify
```

## 适配版本

- CANN：<version>
- 芯片：<ascend910b / ascend910_93 / ...>
