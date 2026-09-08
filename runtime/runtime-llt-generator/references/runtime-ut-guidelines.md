# Runtime UT Guidelines 路径

本文件只维护目标 CANN Runtime 仓中的规范地址和读取条件。每项同时提供从**目标 Runtime 仓根目录**解析的本地路径，以及 `cann/runtime` 的官方网页地址；不从 RuntimeAgent 仓或 Skill 目录解析这些本地路径，也不在本 Skill 中复制规范正文。

能够定位目标 Runtime 工作树且本地文件存在时，优先读取本地文件，以匹配被测代码版本；无法定位目标 Runtime 工作树或本地文件不存在时，读取对应网页地址。

## 输出方案或生成 LLT 前

依次读取：

| 顺序 | 文档 | Runtime 仓内路径 | 读取重点 |
| --- | --- | --- | --- |
| 1 | [编码规范](https://gitcode.com/cann/runtime/blob/master/docs/zh/guidelines/coding-guidelines.md) | `docs/zh/guidelines/coding-guidelines.md` | 安全、生命周期、错误处理、并发，以及通用和 Runtime 特定代码风格 |
| 2 | [UT代码规范](https://gitcode.com/cann/runtime/blob/master/docs/zh/guidelines/ut-coding-guidelines.md) | `docs/zh/guidelines/ut-coding-guidelines.md` | 有效断言、mock 与全局状态清理、测试隔离和可维护性 |
| 3 | [Runtime DT用例开发总纲](https://gitcode.com/cann/runtime/blob/master/docs/zh/guidelines/dt_guide/README.md) | `docs/zh/guidelines/dt_guide/README.md` | DT 工程布局、构建接入、命名和完整校验 |
| 4 | [UT用例开发指导](https://gitcode.com/cann/runtime/blob/master/docs/zh/guidelines/dt_guide/ut_case_development_guide.md) | `docs/zh/guidelines/dt_guide/ut_case_development_guide.md` | 用例设计检查项、校验方式和 Runtime 常见注意事项 |

## 条件读取

涉及 fixture、gmock/mockcpp、公共 stub/data、模拟器、CMake 挂载、覆盖率或 Sanitizer 时，再读取[测试框架指南](https://gitcode.com/cann/runtime/blob/master/docs/zh/guidelines/dt_guide/test_framework_guide.md)（Runtime 仓内路径：`docs/zh/guidelines/dt_guide/test_framework_guide.md`）。

## 路径漂移

先验证本地地址；本地不可用时验证并读取对应网页地址。文件缺失或改名时，在 `cann/runtime` 的 `docs/zh/guidelines/` 中查找正式替代文件并记录地址漂移；本地和网页均找不到正式来源时停止依赖该规则，不得凭记忆重建规范正文。
