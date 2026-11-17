# 从设计到实现

## 1. 定向查找

按 [开发入口](development-entry.md) 消费已有接入记录并校验知识卡。MATCH 时复用字段/资源契约，只展开本次改动的方法和新增接口；存在变化时再定向核对实际绑定 executor、基类或对应 template/kernel。新 Ring/Tree 不要求存在相同算法样本；样本用于确认接口及工程接线。

当前标准布局：

```text
src/ops/<op>/
├── selector/
└── algorithm/
    ├── executor/
    └── template/ccu/
        ├── ccu_temp_<name>.h / .cc
        ├── CMakeLists.txt
        └── kernel/
            ├── ccu_kernel_<name>.h / .cc
            └── CMakeLists.txt
```

旧版 `src/ops/<op>/template/ccu` 与 HCOMM `src/legacy/.../alg_ccu_context` 不等于当前接口。定位脚本只提示旧布局，不自动迁移。若任务确实针对旧版本，先重新建立契约；新功能不落 HCOMM legacy。

## 2. Template 与资源声明

AllGather Sole先消费 [框架调用契约](framework-contracts.md) 的方法表；成本入口、默认构造和资源默认行为均按实际调用路径核对。短代码实现按需取用 [代码范式](implementation-patterns.md)，不展开整份样本。

- 类继承 `CcuAlgTemplateBase`，按实际基类和 executor 调用实现 `Describe`、`CalcRes`、`KernelRun`、`FastLaunch`、`GetRes`、`GetThreadNum`、`CalcScratchMultiple` 等所需方法。基类有默认实现不代表目标算法可以依赖它完成工作。
- 从 `subCommRanks` 建立 userRank↔局部 rank↔channel 的映射；不要假设 channel 数等于 rankSize 或按原始 rank 排序。
- `CalcRes` 中准确填写每次 template 调用的 kernel 数与每个 `CcuKernelInfo`。函数名、函数指针、参数类型及所有权须一致。
- `setKernelArg` 保存注册期对象；channels 的序列必须和 kernel 消费序列一致。设置 die/resGroup 时核对 channel 归属及资源复用关系。
- 同步相关 thread/notify 数按 executor 实际使用推导，不能照抄 AICPU 线程通知模型。资源查询和实际注册必须消费同一套有效配置。

## 3. Kernel 指令生成

kernel 回调返回 `CcuResult`，接收 `CcuKernelArg` 并转成对应注册期结构。按设计创建 CCU 对象，加载参数，生成地址交换、搬运/归约和同步。

- host 的 `for/if` 用于固定拓扑、注册参数和指令展开；运行期 Variable 条件使用经过核验的 `CCU_IF/CCU_WHILE` 等。不要用 C++ `break/return` 表达运行期 CCU 分支退出。
- `Variable` 构造、赋值与移动具有指令生成/句柄语义，不能当普通数值变量任意复制；见 [API 说明](api-and-synchronization.md)。
- 对可返回 `CcuResult` 的调用遵循当前 `CCU_CHK_RET`/错误传播方式。控制流宏本身不返回状态，注册成功不能代替分支覆盖。
- `LocalAddr` 与 `RemoteAddr` 的地址和 token 必须来自正确的注册域/peer，按已经确定的数据流计算字节偏移。
- 逐阶段证明 Event/Notify 的生产消费、mask 与复用。流水线深度或 unroll 是算法选择，不是可复制常量。

## 4. 启动参数及 FastLaunch

先填写 [参数契约样本](../assets/launch-contract.example.json) 的同类结构，再逐项映射到代码：

- `launch_args` 为 host 传输的槽序列；`loads` 将每个被消费槽绑定语义字段。
- `launch_count` 是 64 位槽数；检查实际数组长度和 kernel 加载范围。
- `fast_launch.enabled`为true时，声明完整缓存顺序、再次launch的槽数、需要更新字段及物理缓存索引。可选`cache_launch_offset`以uint64槽计数，默认0；检查`cached_args[offset:offset+launch_count]`与`launch_args`一致，前后metadata均不发给kernel。`loads.index`仍从kernel启动参数0开始，`updates.index`按完整缓存物理下标填写，不减offset。
- AllGather样本的基址偏移位于缓存末尾；Reduce带长度头的缓存从槽1启动，见 [Reduce契约样本](../assets/launch-contract.reduce.example.json)（仅8rank布局）。metadata位置取自当前实现；改变排列后同时更新FastLaunch指针起点、槽数和物理索引。
- 核对当前 `CCU_MAX_TASK_ARG_NUM` 等容量限制；JSON 检查器接受目标源码对应的 `cache_capacity`，不会假定所有版本固定 48。

检查器允许多个 `LoadArg` 消费同一槽（广播到不同变量），也允许带明确理由的未消费 padding 槽；不会要求 kernel 必须恰好加载每个槽一次。

## 5. 接线与构建

核对以下实际链路：

1. executor 正确传入 buffer、slice/repeat、kernel 资源和启动上下文。
2. `REGISTER_EXEC_V2` 的算法名、template/TopoMatch 类型正确；`REGISTER_ALG_ATTRS` 如实描述拓扑、rank、dtype/reduce、in-place 范围。
3. selector 显式或默认路径能到达该注册项，范围外请求使用既有回退，不误入新算法。
4. template 和 kernel 的 `.cc` 各自进入实际 CMake target，上级 `add_subdirectory` 可达，版本/引擎守卫正确。
5. 使用的 HCOMM 符号在当前 HCCL dlsym 包装和目标动态库中存在；只发现 HCOMM 新接口不表示 HCCL 当前可用。

先保存改动清单及设计对应关系，再执行 [验证流程](validation.md)。不要通过修改测试期望或默认路径掩盖新算法未被命中。
