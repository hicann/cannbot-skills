---
skill_name: tilelang-env-check
---

# Case 1: TileLang-Ascend 环境检查流程

## Config
- Max Tokens: 250000
- Ascend Platform: A2

## Prompt

帮我检查 TileLang-Ascend 开发环境是否配置正确，包括代码仓库、编译状态和环境变量。

## Expected Output

回复应按顺序执行环境检查步骤：检查 Python 包依赖、CANN 环境变量、代码仓库完整性、编译安装状态和环境变量配置。发现问题时自动修复并按依赖顺序处理。

## Expectations
- [contains] ASCEND_HOME_PATH
- [contains] install_ascend.sh
- [contains] set_env.sh
- [contains] quick_verify.py

---

# Case 2: 自动修复策略与依赖顺序

## Config
- Max Tokens: 200000
- Ascend Platform: A2

## Prompt

环境检查发现子模块不完整怎么办？修复子模块后还需要做什么操作？

## Expected Output

回复应说明发现子模块不完整时，先告知用户问题存在，自动调用 tilelang-submodule-pull skill 拉取子模块，完成后告知用户修复结果。由于子模块更新，必须重新执行编译（bash install_ascend.sh），然后设置环境变量（source set_env.sh），最后运行测试验证。强调修复前必须告知用户问题，修复后告知结果，所有提示都要打印给用户。

## Expectations
- [contains] tilelang-submodule-pull
- [contains] install_ascend.sh
- [contains] source set_env.sh
- [contains] 前置
