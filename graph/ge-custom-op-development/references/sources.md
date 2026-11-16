# 版本与来源

本文件统一维护源码依据和复核方法，其他章节只保留必要结论。

## 能力的判定顺序

1. 用户目标环境的头文件、运行库、Python ge 包和 bridge 决定可用能力。
2. 优先读取与安装包对应版本/提交的官方源码解释行为。
3. develop 用于查看新增能力，不能用新分支的实现代替旧安装包行为。

记录 CANN/GE 版本、SoC、主机架构、编译器、Python 和 bridge 状态。
`python3 scripts/check_env.py` 可生成基础环境报告；不自动更新用户仓库、不安装依赖、不代表环境全部可用。
本地副本可能有未提交修改或未合入功能，核对 remote URL、commit、工作区状态后再引用。
不要默认 origin/upstream 指向哪个仓库，也不强制把用户当前分支更新到最新 develop。

## 本次源码核对基线

官方仓库：[cann/ge](https://gitcode.com/cann/ge)。
基线 commit：`915ebdc11357fe12dab1ac7a912c7b1282669f60`。
下列路径均相对于该仓库根，不是某台机器的绝对路径；引用时按符号检索，行号仅辅助定位。

| 事实 | 文件与定位符号 |
|---|---|
| 接口、注册宏、版本门控 | inc/graph_metadef/external/graph/custom_op.h；REG_OP_BACKEND_UNIQ |
| Annotated args/launch/返回值 | inc/graph_metadef/external/exe_graph/runtime/annotated_args_context.h；AddLaunch |
| 地址刷新优先级 | graph_metadef/graph/normal_graph/custom_op_registry.cc；GetArgsRefreshStrategy |
| 声明式静态图分发与 magic | compiler/engines/custom_engine/custom_ops_kernel_builder.cc；GenerateTask、SetKernelBinAttrs |
| 静态默认 shape 行为 | graph_metadef/graph/utils/op_desc_utils_ex.cc；InferCustomOpShapeByDefault |
| V1 task 分发与重放 | runtime/v1/graph/load/model_manager/task_info/ge/custom_task_info.cc；DistributeEagerOp |
| 自定义 so 和 Portable 收集 | base/common/model/ge_root_model.cc；CollectCustomOpSo |
| 在线主动注册与离线注册抑制 | graph_metadef/graph/normal_graph/custom_op_creator_register.cc；IsOfflineCustomOpSoLoading、NeedRegisterLocalCreator |
| pull creator 符号与加载期解析 | inc/graph_metadef/graph/custom_op_pull_registry.h、base/common/helper/custom_op_registry_builder.cc；ResolvePullCreatorSymbols |
| V1 Execute 调用点与地址刷新早退 | runtime/v1/graph/load/model_manager/task_info/ge/custom_task_info.cc；DistributeEagerOp、UpdateHostArgs |
| RT2 每轮执行与 kernel 类型 | runtime/v2/engine/custom/kernel/custom_op_kernel.cc；ExecuteCustomOpFunc、ExecuteCustomOpWithInferShapeFunc |
| ShapeInferOp 影响 RT2 kernel 类型 | runtime/v2/engine/custom/converter/custom_node_converter.cc；NeedCustomOpInferShape |
| OM 执行的 V1/RT2 分派 | api/acl/acl_model/model/model.cpp；aclmdlExecuteV2Impl、RuntimeV2ModelExecute |
| 动态维度接口只走 V1 | runtime/v1/executor/ge_executor.cc；GeExecutor::SetDynamicDims |
| ExecConfig 无 shape 项 | inc/external/acl/acl_mdl.h；aclmdlExecConfigAttr |
| ACL 日志行内函数名出现两次 | api/acl/common/log_inner.h；ACL_LOG_INFO |
| ATC 动态档位参数语义 | api/atc/main_impl.cc、api/atc/atc_flags.cc、inc/framework/om2/model_data/gert_model_data.h；dynamic_dims、gears |
| OPP 加载 | graph_metadef/register/op_lib_register.cc；GetAllCustomOpApiSoPaths |
| Python 注册、签名、Portable | api/python/ge/ge/custom_op/registry.py、_bridge.py、_portable.py |
| Python capability 与不支持项 | runtime/custom_op/python_custom_op_adapter.h/.cc |
| Python 插件扫描 | api/python/ge/ge/_internal/plugin_loader.py |
| Python 模型打包 | base/common/helper/python_custom_op_package_helper.cc |

若路径因版本变化不存在，按文件名与符号查找，不编造缺失接口。

## 官方样例入口

按任务读取 `examples/custom_op/` 的相关子目录，不统计目录数量作为选型依据：

- C++ 直调和原型：ascendc_add_custom、triton_add_custom。
- 编译时机：tilelang_add_custom{,_online,_offline}、compilable_add_custom。
- Python 编译与二进制提取：python_compilable_add_custom。
- 声明式：annotated_args_refresh_add_custom 的 online/offline、cpp/python。
- Host：host_cpu_add_custom；本 Skill 附带其 Python 常量折叠示例。
- shape：data_dependent_shape_custom、value_dependent_shape_custom。
- 刷新与辅流：args_refresh_add_custom、eager_attached_stream_add。

## 取源码与记录证据

有本地仓时只读核对：git remote -v、git rev-parse HEAD、git status --short。
需要确认官方分支时可 git ls-remote https://gitcode.com/cann/ge.git refs/heads/develop，或通过 GitCode API contents 读取指定 ref；不依赖返回 HTML 的网页 raw 地址。
大量检索才考虑独立浅克隆，不覆盖用户工作区。

实测记录应含硬件、软件版本、命令、输入范围、结果和可定位日志/Job；源码推论与实测分开标注。原文中的作者历史实测仅作回归线索，不声称本次重构已重复验证。
发现目标实现与本 Skill 不符时，说明差异与依据，按目标接口工作；不自行向维护者发送消息。
