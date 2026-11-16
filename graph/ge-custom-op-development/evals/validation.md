# 验证与评测

## 离线回归

在仓库根目录执行：

```bash
python3 -m unittest discover -s graph/ge-custom-op-development/evals -p 'test_*.py' -v
```

测试真实模板 helper 与检查脚本，模拟外部编译器/GE 导入，不使用 NPU：源码和头依赖改变会重新编译、独立临时目录、提取 device 段、拒绝 host 对象、固定块 kernel 边界、元信息隔离、跨平台拒绝、非空 Portable 状态及示例数值校验。
另用真实 CMake 配置验证绝对路径实现文件和原型 TU 进入编译列表；该测试使用临时目标目录，不编译或链接 CANN。上述测试不证明真实 bisheng 编译、注册、device launch 或 OM 部署通过。

## Agent 行为评测

`evals.json` 包含 13 条问答和 2 条文件生成用例，按仓库 ST 设计指南维护。
问答的 `expected_output` 按独立语义要点分行，测试方案与实际执行证据分别要求；文件生成用例通过 `file_list` 检查交付物，并由语义评审检查注册、未实现状态与验证边界。
正向选型用例配置了相近的干扰 Skill 和 `skill_activated` 断言；ONNX 映射用例覆盖转出边界。

| 场景 | 用例 ID |
|---|---|
| 在线预编译默认分发与单 `.so` 需求 | 1 |
| OM 状态、Triton/TileLang 加载 | 2、3 |
| 动静态双模与地址刷新 | 4 |
| Python OM / Portable 与流 | 5、6 |
| Tiling 托管与 ONNX 转出 | 7、8 |
| kernel 契约和尾块边界 | 9、10 |
| 版本门控和 MIX 加载属性 | 11、12 |
| Portable 共享实例状态 | 13 |
| C++ / Python 生成工程 | 14、15 |

源码/依赖修改后的编译缓存失效、Host 示例的输出校验，以及固定块边界等已有确定性本地测试，不再重复设置需要真实文件输入的 AI 问答用例。
原型注册及未完成骨架的语义检查保留在 C++ 文件生成用例。

本地 ST 框架的白名单已包含 `ge-custom-op-development`，可按指南执行：

```bash
python3 -m pytest tests/system/scripts/test_skill_basic.py -v -k ge-custom-op-development
python3 -m pytest tests/system/scripts/test_skill_evals.py --skill ge-custom-op-development -v
```

运行前需安装 `tests/system/scripts/requirements.txt` 中的依赖，并配置可用的 opencode 模型。
文件生成用例的存在性断言不代替构建和数值验证；外部 CI 是否执行各个模式以其实际报告为准。

使用现有仓库 ST 框架运行，记录模型/推理强度、Skill commit、运行环境与执行轨迹。
比较重构前后时使用相同任务、模型与资源条件，分别记录：

- 选型/生成/排障任务成功率与错误类型。
- 实际生成物的构建、加载、数值、重放和部署证据。
- 读取的参考文件、无关读取、重复追问、工具失败及上下文用量。
- 多次独立运行中稳定复现的问题，不只看聚合分数波动。

测试环境有 CANN 时，补真实编译和加载；有目标设备时补全数值/重放。
离线部署必须新进程加载、移走原构建/源码目录，再验证数据恢复及数值。
双模需分别证明静态和动态分支；共享实例需验证多节点、多模型和冲突状态。
缺少环境的阶段标为未执行，不伪造全绿，也不把读到某关键词当作成功。
