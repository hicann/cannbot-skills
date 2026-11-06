# AscendCL 接口能力总览

AscendCL（Ascend Computing Language）提供异构计算场景的统一编程入口，覆盖设备管理、内存管理与算子调用三类核心能力。

## 能力分类

| 能力域 | 代表接口 | 说明 |
| --- | --- | --- |
| 设备管理 | aclrtSetDevice / aclrtGetDevice | 指定与查询当前运算设备 |
| 内存管理 | aclrtMalloc / aclrtFree | 申请与释放 Device 侧内存 |
| 算子调用 | aclopExecute | 单算子直调执行 |
| 上下文管理 | aclrtCreateContext |  |
| 事件同步 | aclrtCreateEvent / aclrtRecordEvent | 记录与等待事件 |

## 快速接入

1. 引入头文件并调用 aclInit 完成全局初始化。
2. 调用 aclrtSetDevice 指定运算设备。
3. 调用 aclrtCreateContext 与 aclrtCreateStream 创建上下文与流。
4. 安装完成后点击「完成」按钮退出安装向导。

详细接口参数请参考对应头文件内的注释说明。
