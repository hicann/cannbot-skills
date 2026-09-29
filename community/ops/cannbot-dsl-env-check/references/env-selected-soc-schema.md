# env_selected_soc.yaml 结构

`env_selected_soc.yaml` 记录当前环境默认选中的第一张运行时可用卡，以及环境探针直接获得的芯片身份信息。

本文件不包含核数、片上存储、L2、全局内存、带宽、数据类型或架构能力等推导信息。

## 成功选择

    selection:
      selection_status: SELECTED
      policy: first_runtime_visible_device
      logical_device_id: 0
      device_name: Ascend950PR_957b
      message: 已选择 environment.yaml 中第一张运行时可用卡
    chip_identification:
      device_name: Ascend950PR_957b
      full_soc: Ascend 950PR_957b V100
      npu_arch: '3510'
      runtime_source: torch_npu
      architecture_source: asys

## 没有可用卡

    selection:
      selection_status: NO_AVAILABLE_DEVICE
      policy: first_runtime_visible_device
      logical_device_id: null
      device_name: null
      message: environment.yaml 未识别到可用且带名称的运行时设备
    chip_identification: null

## 字段说明

| 字段 | 含义 |
|---|---|
| selection_status | `SELECTED` 或 `NO_AVAILABLE_DEVICE` |
| policy | 固定为 `first_runtime_visible_device` |
| logical_device_id | 当前进程中默认选中卡的逻辑编号 |
| device_name | `torch_npu` 返回的设备名称 |
| full_soc | `asys` 返回的完整芯片信息；不可用时为 `null` |
| npu_arch | `asys` 返回的架构号；不可用时为 `null` |
| runtime_source | 设备名称的探测来源 |
| architecture_source | 完整芯片名和架构号的探测来源 |
