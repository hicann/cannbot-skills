# Host Add 最小完整工程

来源：cann/ge commit `915ebdc11357fe12dab1ac7a912c7b1282669f60`，`examples/custom_op/host_cpu_add_custom/constant_folding/python/`。
CMakeLists.txt、proto/、插件保留官方示例；src/run.py 补充 shape/dtype/finite 校验，run.sh 简化为工程私有构建/安装。

用途：Python 原型与 Host 实现 → 生成 ES wheel/OPP → GraphBuilder → GE Session → 数值断言。
输入为两个 Const，验证的是 Host 常量折叠；不覆盖 Device kernel、动态输入或 OM 恢复。

需要 Linux、匹配基线 API 的 CANN/GE Python 包/bridge、CMake、C++ 编译器和 pip。
先加载目标 CANN 环境，在本目录执行 `bash run.sh`。DEVICE_ID 默认 0，可由环境变量指定。
生成文件与 wheel 安装均位于 build/，不安装到系统 site-packages。

成功标准：run.py 中的输出数量、dtype、shape 和全部 float32 元素断言通过，命令退出 0。
失败时保留日志，不只根据插件中的 print 判断成功。
本次 Skill 重构仅完成离线语法与工程组织检查，未在 CANN/NPU 环境执行该例。
