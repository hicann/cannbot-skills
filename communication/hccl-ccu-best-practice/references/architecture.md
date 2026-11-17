# 架构与资源生命周期

本文路径中 `HCCL/`、`HCOMM/` 分别指调用方提供的仓根。具体文件哈希和关键符号在 [来源索引](source-map.json)。以目标源码和安装 SDK 为准；参考路径漂移时先定位替代接口。

## 分层

```text
算子入口 → selector → 算法注册表 → executor + TopoMatch
                                           ↓
                              CcuAlgTemplateBase 派生类
                                 CalcRes / KernelRun / FastLaunch
                                           ↓
HCCL op_common：准备 channel、CCU 实例、注册 kernel、持有资源
                                           ↓ dlsym
HCOMM：资源管理、指令生成/注册、kernel launch、CCU 执行
```

- 算法及算子接线在 HCCL 的 `src/ops/<op>/`；HCOMM 提供通信基础能力。
- 公共 template 契约：`src/ops/op_common/algorithm/template/ccu_alg_template_base.h`，继承 `CommonAlgTemplateBase`；不能照搬 AICPU 的 `InsAlgTemplateBase` 接口清单。
- kernel 共用帮助方法：`src/ops/op_common/algorithm/template/ccu/ccu_kernel_alg_base.{h,cc}`；这里的 GroupCopy/ReduceLoopGroup 等是具体实现工具，使用前核对版本、资源和参数契约。
- HCCL 适配入口在 `src/common/hcomm_dlsym/`，C++ CCU 包装头在其 `ccu/` 子目录。HCOMM 对外声明在 `include/ccu/`，底层包间声明在 `pkg_inc/hcomm/ccu/`。阅读底层实现用于确认行为，不等于算法可以直接依赖私有头。

## 资源生命周期

1. **算法选择与分层**：executor/TopoMatch 决定子通信域、template 组合、buffer 类型和 `TemplateDataParams`。这些决定必须追到实际调用方，不能由 template 猜测。
2. **声明资源**：template `CalcRes` 填 `AlgResourceRequest`；`ccuKernelNum` 描述 template 调用的 kernel 数，`ccuKernelInfos` 保存注册信息。多 template 合并后仍须对应。
3. **kernel 注册输入**：`CcuKernelInfo` 含函数名、函数指针、`kernelArg`、channels、`dieId`、`resGroup`。`setKernelArg(shared_ptr)` 维持参数所有权；不要传即将销毁的栈对象。
4. **准备与注册**：HCCL `op_common.cc` 查询需求、准备 channel 和实例，并在 `RegisterCcuKernels` 中校验 kernel 总数；按资源组调用 `HcommCcuKernelRegisterStart/Register/RegisterEnd`。
5. **生成指令图**：kernel 回调接收 `CcuKernelArg`，创建 `Variable/Event/LocalAddr` 等并发出 CCU DSL 指令。资源查询也可能执行生成路径；生成函数不得偷偷进行设备计算或依赖不可重复的 host 副作用。
6. **运行期启动**：template 组装 `uint64_t taskArgs[]`，将 `threadHandle`、注册得到的 kernel handle、参数地址及槽数交给 `HcommCcuKernelLaunch`。HCOMM 在 `ccu_launch.cc` 将参数转换成任务并下发。
7. **快速重放**：`CcuKernelSubmitInfo` 缓存参数，FastLaunch 根据新 buffer 重算会变化的字段；缓存 key、复用条件、失效条件由实际 executor/框架决定，不假设任意 dtype/count/topology 可复用。

`CcuKernelInfo::resGroup` 的源码注释说明不同组会复用资源；因此不能把不同组直接当作可并发 kernel。并发需求必须核对分组、资源别名、调度顺序与跨 kernel 同步。单 kernel 使用的 channel 所属 die 须与注册 die 一致，多 die 设计应明确拆分。

## 两种参数不能混用

| 项目 | 注册期 | 启动期 |
|---|---|---|
| 输入 | `const void **kernelArgs`，样本内为一个算法参数对象 | `const void *taskArgs`，内部按 `uint64_t*` 使用 |
| 消费方 | kernel 回调的 C++ 参数/拓扑/资源创建 | kernel 中 `ccu::LoadArg(var, index)` |
| 计数 | 注册参数个数 | 64 位槽个数，非字节数 |
| 变化 | 改拓扑/资源可能需要重注册 | 地址/长度等按缓存契约刷新 |

调研样本 AllGather Mesh1D Mem2Mem：启动 15 个槽，缓存另追加 2 个基址偏移。**17 个缓存槽不意味着 launch 传 17**。具体映射见 [样本 JSON](../assets/launch-contract.example.json)。该数字只属于这个样本。

## 选路与构建边界

注册名必须同时出现在真实 executor 注册和预期 selector 路径。例如样本 `CcuSchedAllGatherSoleMesh` 绑定 `CcuTempAllGatherMesh1DMem2Mem`，`CcuMSAllGatherSoleMesh` 绑定另一套 template/kernel。不能仅凭名字猜测模式可互换。

属性约束、拓扑可达性、dtype/reduce 支持、in-place 能力与成本候选共同决定实际选中。显式环境配置的语法按当前解析器查证，不把完整注册名直接当成 `HCCL_ALGO` 的值。

已核验样本的 `template/ccu/CMakeLists.txt` 添加 template 源码并进入 `kernel/`，后者将 kernel 源码加入 `hccl` target。executor 注册还带 `AICPU_COMPILE` 与 CANN 版本条件。新增文件沿同目录构建契约登记，不注册到 AICPU scatter kernel 库。
