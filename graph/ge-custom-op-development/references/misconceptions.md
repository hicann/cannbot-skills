# 常见误解索引

仅在按误解关键词查找时读取。事实与理由维护在链接章节，不重复保存长篇论证。

| 误解 | 正确入口 |
|---|---|
| 看到 IMPL_OP 就属于托管工程 | [机制分诊：已有代码](mechanism-triage.md#已有代码判别) |
| kernel 语言决定 C++/Python 交付件 | [产物与加载](language-launch.md#产物与加载方式) |
| .so 必须只叫 libcust_opapi.so | [OPP 与构建](build-verify.md#opp-与构建) |
| 原型头安装好了就已经注册 | [模板选择](build-verify.md#模板选择) |
| 预编译或出 OM 一定需要 Compile | [编译时机](scenarios.md#编译时机) |
| 出 OM 一定要/一定不要 Portable | [离线状态携带](scenarios.md#离线状态携带) |
| Python 支持 OM 就无需 Python 运行时 | [OM 依赖](python-track.md#om-部署依赖) |
| Python Portable 可以只写一个或继承实现 | [Portable 契约](python-track.md#portable-契约) |
| 共享实例不能有任何实例字段 | [状态与缓存](python-track.md#生命周期和状态) |
| execute 与 declare_launch_args 必须二选一 | [执行与刷新](scenarios.md#执行与地址刷新) |
| ArgsUpdater 和 Annotated 同时实现会组合刷新 | [执行与刷新](scenarios.md#执行与地址刷新) |
| AddLaunch 返回值都可作 token | [参数与返回值](interfaces.md#声明式参数与返回值) |
| 源码 AIV 标注保证产物 AIV | [加载属性与入口](language-launch.md#加载属性与入口) |
| launch info 没有 magic 就证明 MIX 不可能 | [加载属性兼容性](scenarios.md#加载属性兼容性) |
| ceil block_dim 就自动支持尾块和空张量 | [参数与边界](language-launch.md#参数与边界) |
| .o 非空就一定是 device kernel | [二进制提取](language-launch.md#ascend-c-二进制提取) |
| 正确一轮就证明刷新/辅流机制正确 | [数值与重放](troubleshooting.md#数值与重放) |
| kTop/kBottom 是静默覆盖、可随意用 Bottom | [注册与实例](interfaces.md#注册与实例) |
| ResourceUsageReporter 提供配额 | [资源统计](interfaces.md#资源统计) |
| 本地样例数量就是官方能力边界 | [版本与来源](sources.md) |
