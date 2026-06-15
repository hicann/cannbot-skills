---
skill_name: tilelang-submodule-pull
---

# Case 1: 子模块和三方库拉取

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

TileLang 的子模块拉取失败了，请帮我重新拉取代码仓库和第三方依赖。

## Expected Output

回复应执行自动拉取流程，包含两个核心操作：git submodule update --init --recursive（拉取子模块）和 git pull --recurse-submodules（拉取主仓最新代码）。应配置 git 镜像加速源（ghfast.top）。支持前台直接运行或后台 nohup 运行，提供实时终端输出和日志记录到 logs/git_pull.log。拉取成功后自动停止脚本。

## Expectations
- [contains] git submodule update --init --recursive
- [contains] git pull --recurse-submodules
- [contains] auto_pull.sh
- [contains] ghfast.top

---

# Case 2: 错误检测与自动重试机制

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

网络不稳定导致子模块拉取失败了，脚本会怎么处理？会不会一直重试到永远？

## Expected Output

回复应说明脚本会自动检测关键词（Could not access、error、fatal、Failed to clone、unable to access），发现错误后等待 1 小时（3600 秒）自动重试。有超时保护机制：后台运行超过 10 小时（36000 秒）自动停止并记录日志，防止忘记关闭。重试过程持续输出到终端和日志文件，方便用户查看进度。

## Expectations
- [contains] Could not access
- [contains] 3600
- [contains] 10 小时
- [contains] 重试
