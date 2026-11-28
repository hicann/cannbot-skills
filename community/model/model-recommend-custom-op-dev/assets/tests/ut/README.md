# 单元测试（UT）备注

- **只有自定义算子需要 UT**（插件、Pass 不需要）。
- 本 skill **不提供 UT 样例代码**；请按官方 gtest/faker 规范补 `tiling` 与 `infershape` 用例。
- `run.sh` 应在失败时以非 0 退出，便于 `build.sh -u` 串联；通过时输出 `[PASS]`。
