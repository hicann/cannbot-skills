# <OpType> ONNX 插件

## 功能

把 ONNX `<domain>::<version>::<OpType>` 翻译为 NPU 现有算子组成的等价子图，解决 GE 解析报"算子不支持"。

## 分类

自定义插件（onnx_plugin）。

## 实现原理

- `ParseParamsByOperatorFn`：解析原算子 `"attribute"` JSON，映射属性到目标算子。
- `ParseOpToGraphFn`：用 `Data`/`Const` + NPU 现有算子构建等价子图。
- 注册：`REGISTER_CUSTOM_OP("PartitionedCall")` + `ImplyType::BUILDIN`。
- 翻译依据来源：<GPU 实现源码 / 算子说明文档 / 用户逐项确认>。

## 工程结构

```
<plugin_name>/
├── CMakeLists.txt
├── build.sh
└── framework/onnx_plugin/
    ├── CMakeLists.txt
    ├── sample_common.h
    └── <domain>_<optype>.cc
```

## 安装使用

```bash
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
bash build.sh
source build_out/makepkg/set_env.bash
# 或用 demo 跑通：bash demo/run.sh <id>
```

## 适配版本

- CANN：<version>
- 芯片：<ascend910b / ascend910_93 / ...>
