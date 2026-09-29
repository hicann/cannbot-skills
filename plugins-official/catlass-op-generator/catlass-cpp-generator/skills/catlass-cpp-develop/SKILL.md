---
name: catlass-cpp-develop
description: CATLASS C++ Linear Attention 五阶段工作流的 04 算子开发。适用于需要按已评审 Stage 实现 device kernel、host tiling 和直调入口，并逐 Stage 构建与定向精度验证的算子开发场景。
---

# 04 算子开发

进入条件：workflow 状态为 `implementation`，两个 contract 均已冻结，设计校验已通过。
完整读取 `workflow/development-and-validation.md`，按实际组件查询
`catlass/`，并在当前固定 CATLASS/CANN 源码中确认 API。构建前核对 `docs/workflow.json`、
`docs/design.md`、编译参数和实际设备四者的架构一致；发现 `2201/AtlasA2` 与
`3510/Ascend950` 混用时返回 design，不得靠编译试错继续。

逐 Stage 执行“实现、编译、设计一致性检查、运行、精度比较、记录”。device 计算必须使用
CATLASS 组件；host tiling、workspace、AIC/AIV 分工、CrossCore/HardEvent、TilingKey 和数据
生命周期必须与 `docs/design.md` 一致。测试程序必须核对全部输出、原始 dtype、shape、有效区域
和关键分区。

每轮构建安装后重启测试进程并确认加载本轮产物。单用例连续运行 60 秒无返回视为 kernel 超时，
清理进程和设备资源后，将现象、Stage、TilingKey 和 `blockDim` 写入 `docs/validation.md`，再按
失败定位结果恢复到最早受影响阶段。

精度实现问题使用 `issue_type=precision_debug`、`resume_from=implementation`、
`validation_scope=precision_targeted`，只跑失败用例、受影响 Stage 和最小边界集合。
设计假设错误使用 `design_issue/design/full` 返回 03；标杆或接口问题分别返回 02/01。
定向精度通过后清空问题字段，设置 `stage=validation`、`validation_scope=full` 并校验。
