# pre-commit 配置指导书

## 快速使用

```bash
# 1. 安装 pre-commit
pip3 install pre-commit

# 2. 安装 Git Hooks（此后 git commit 自动触发检查）
cd /path/to/cannbot-skills
pre-commit install        # 取消：pre-commit uninstall

# 3. 配置 git pc 别名（对指定范围提交运行检查，每个环境执行一次）
git config --global alias.pc '!f() { pre-commit run --files $(git diff --name-only "$@"); }; f'
git pc HEAD~x        # 检查最近x笔提交
```

完成上述配置后，`git commit` 时将自动执行检查，无需手动干预。完整说明见下文。

## 一、概述

pre-commit 是一个 Git Hooks 框架，在 `git commit` 时自动运行代码检查和格式化，提前拦截规范问题，避免远程 CI 门禁失败。

| Hook | 功能 |
|------|------|
| **pre-commit-hooks** | 行尾空格、文件末尾换行、YAML/JSON 合法性、大文件、合并冲突标记、私钥检测 |
| **clang-format** | C/C++/asc 代码格式化，遵循 `.clang-format` |
| **ruff-check / ruff-format** | Python 静态检查（自动修复）+ 格式化 |
| **codespell** | 拼写检查（CANN/ascend 等术语已加白名单） |
| **OAT Check** | 开源合规检查（许可证头、二进制/归档文件拦截） |

## 二、环境要求

- **Git**: 2.0+，**Python**: 3.9+，**pre-commit**: 4.0+

> clang-format / ruff / codespell 由 pre-commit 首次运行时自动下载（版本见 `.pre-commit-config.yaml`）；OAT 首次运行时自动 `pip install oat-py`。均无需手动安装。

## 三、安装配置

```bash
# 1. 安装 pre-commit
pip3 install pre-commit

# 2. 安装 Git Hooks
cd /path/to/cannbot-skills
pre-commit install        # 取消：pre-commit uninstall

# 3. 配置 git pc 别名（对指定范围提交运行检查，每个环境执行一次）
git config --global alias.pc '!f() { pre-commit run --files $(git diff --name-only "$@"); }; f'
git pc HEAD~x        # 检查最近x笔提交
```

## 四、日常使用

```bash
# 提交时自动检查
git add . && git commit -m "msg"

# 手动运行
pre-commit run                        # 暂存区
pre-commit run clang-format           # 单个hook

# 检查指定文件
pre-commit run --files src/foo.py tests/bar.cpp

# 检查指定目录（pre-commit 不支持目录参数，需用 find 展开成文件列表）
find ops -type f | xargs pre-commit run --files

# 跳过（紧急情况）
git commit --no-verify -m "msg"
```

> ⚠️ 本仓存量文件（数千 md / 上千 py）从未被检查过，hook 只查**暂存文件**属增量生效；**不建议** `pre-commit run --all-files` 全量跑（会翻出大量存量问题），存量债按目录分批用 `find <目录> | xargs pre-commit run --files` 治理。

## 五、失败排查

搜索输出中的 `Failed` 定位失败项，查看该 hook 下方输出处理，或粘贴给 AI 获取修复建议。

拦截类钩子的报告同时落盘在 `pre-commit_reports/`（不入 git）：`codespell.log`（拼写命中清单）、`oat_result.txt`（合规违规明细），可随时复查。

| 报错                  | 原因               | 处理方式                                          |
|-----------------------|--------------------|---------------------------------------------------|
| clang-format Failed   | 代码存在规范问题 | rebase 最新代码后再做 pre-commit；重新 `git add` 后再次 commit |
| OAT Compliance Failed | 缺版权声明等       | 搜索 `OAT Scan Result Summary` 定位文件；版权声明无法自动订正，需参照仓内文件头手动补齐 |

## 六、补查历史提交（--no-verify 跳过后）

对指定范围的提交变更文件补跑检查：

```bash
git pc HEAD~x        # 最近x笔提交
git pc <commit>^     # 某一笔提交
```

> 部分钩子会自动修复文件，修复后需重新 `git add` 并追加提交。

## 七、检查项说明

1. **pre-commit-hooks** (v4.6.0)：trailing-whitespace（md 文件保留行尾双空格硬换行语法）、end-of-file-fixer、check-yaml/json（`tsconfig*` 与 `.claude/settings.json` 等 JSONC 豁免）、check-added-large-files（`*.jsonl` eval 数据集豁免）、check-merge-conflict、detect-private-key（2 个脱敏测试固件豁免）
2. **clang-format** (v18.1.8)：遵循项目 `.clang-format`（Google 风格，4 空格缩进，120 列限宽，枚举逐行，构造函数初始化列表逐行换行，函数定义大括号换行，指针左对齐 `int* ptr`）；`third_party/`、`kb/`（提取快照）、`.html/.csv/.svg` 排除
3. **ruff** (v0.14.14)：ruff-check（`--fix` 自动修复，豁免 `E501,RUF001-003,E402,E701,E702,N815`——F 系列真 bug 类规则全启用）+ ruff-format
4. **codespell** (v2.4.1)：拼写检查，**覆盖 markdown**（面向用户的主内容）；py/ts/cpp 不查（变量名误报多）；CANN、ascend、`AscendC::Te`、nd 格式等术语已加白名单；`kb/` 提取快照不检
5. **OAT**：基于 oat-py，检查许可证头（YAML/CSV/PNG/mjs 已豁免；**存量无 header 的 sh 脚本触碰时需补齐**，参照仓内文件头）、禁止二进制和归档文件。注：`tools/opsbot` 等自带 `[tool.ruff]` 的子工程按其文件级配置检查

## 八、常见问题

**Q1: 首次提交 OAT 检查很慢？**

首次运行需 `pip install oat-py`，属正常现象，后续很快。

**Q2: 手动全量格式化 C/C++？**

```bash
bash scripts/format_cpp.sh            # 整个仓库
bash scripts/format_cpp.sh ops        # 指定目录
```

自动排除 `build/`、`build_out/`、`third_party/`、`.git/`。

**Q3: 如何全局生效，让新 clone 的仓库自动继承？**

```bash
mkdir -p ~/.git-templates
pre-commit init-templatedir ~/.git-templates
git config --global init.templatedir ~/.git-templates

# 已有仓库需重新拷贝模板
cd /path/to/cannbot-skills && git init
```

> hook 只是启动器，实际检查仍读取当前仓库的 `.pre-commit-config.yaml`；未配置该文件的仓库会自动跳过，互不影响。

取消：`git config --global --unset init.templatedir && rm -rf ~/.git-templates`

**Q4: 何时需要安装本地 clang-format？**

仅手动批量格式化（`bash scripts/format_cpp.sh`）时需要，建议 18.x；日常 `git commit` 触发的 pre-commit 不依赖此工具。

```bash
sudo apt install clang-format
```

## 相关文档

- [pre-commit官方文档](https://pre-commit.com/)
- [clang-format配置](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT工具](https://gitcode.com/openharmony-sig/tools_oat)
