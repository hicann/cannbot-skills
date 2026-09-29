# CANNBotDSL 算子通用设计方法

当某个范式没有 DSL 专用适配层时，依据调用方提供的规格、目标设备事实和 DSL API 资料完成设计。本文只给方法，不规定产物文件名或工作流路径。

1. **语义与接口**：写清数学公式、公开 Python 参数、shape/dtype/layout 推导、空输入与错误行为。设备值计算和输出归属必须可追溯到规格。
2. **执行路径**：区分静态配置、编译期特化与运行时参数。只为算法或数据通路有实质差异的情况拆分路径，并说明判定顺序、互斥性和支持域覆盖。
3. **Host 与 Launch**：说明元数据检查、输出和 workspace 分配、零拷贝视图、任务划分、资源预算和 Launch 顺序。资源数值从目标平台证据取得。
4. **DSL Kernel**：从公式推导阶段、任务坐标、搬运、计算、写回、Channel/Buffer 生命周期和同步。按实际路径选择 Cube、Vector 或单一路径；不套样例配置。
5. **验证**：用独立 Golden 覆盖支持的 dtype、shape、layout、分支、尾块、特殊值和错误路径。记录所用 API 的来源与尚待验证的设备行为。

设计中的分支 ID 只用于文档引用。Host 与 Kernel 的参数契约由实际 Python/DSL 调用确定，不需要 AscendC 的 TilingKey、TilingData、OpDef 或 C++ Kernel 结构。
