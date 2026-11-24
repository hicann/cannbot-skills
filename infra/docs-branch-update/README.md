# docs-branch-update README

仓库分支固定项自动修改 skill：拉出版本分支后，按用户意图直接完成文档固定修改项，修改完成后输出修改清单。

**前置条件**：在需要修改的版本分支上使用该 skill（先切换到目标分支，如 `git checkout 9.3.0-beta.1`）——链接更新按当前 git 分支确定版本映射，在 master/main 分支上执行会被拒绝，也就无法得到正确的版本映射。

## 功能

| 修改项 | 脚本 | 说明 |
| --- | --- | --- |
| 文档参考链接版本更新 | `scripts/update_links.py` | gitcode.com 与 www.hiascend.com 链接按当前分支映射，`master`/`latest` 替换为对应版本分支或版本号 |
| 删除产品内容 | `scripts/remove_npu_tags.py` | 删除不允许存在的 NPU 类型标签块及内容（缺省范围从文档写作规范在线获取），含混合标签列表改写与接缝处冗余 `<br>`/空行局部清理 |

两类能力相互独立，可只更新链接、只删除标签，或两者都处理。

## 触发方式

在需要修改的版本分支上（见前置条件），直接用自然语言描述需求即可，例如：

> 使用 docs-branch-update，把docs/zh/api目录下文档链接更新到当前分支并删除 x90/9030 相关内容。

> 处理分支固定项修改项。

agent 会根据 SKILL.md 的触发描述匹配场景加载本 skill，并按意图选择模式直接调用对应脚本执行修改（修改后可用 `git diff` 复查、`git checkout` 回退）；说"只更新链接"或"只删除标签"即选择单一模式。

## 使用方法

在仓库根目录执行（Python 3，仅标准库；命令直接修改文件并输出修改清单）：

```bash
# 只更新链接（-d 缺省为 <仓库根>/docs/zh，-b 缺省自动获取当前分支）
python3 infra/docs-branch-update/scripts/update_links.py [-d <目录|文件>] [-b <分支>]

# 只删除产品内容（--npu 缺省从文档写作规范在线获取不允许存在的标签范围，仅查询一次）
python3 infra/docs-branch-update/scripts/remove_npu_tags.py [-d <目录|文件>] [--npu 类型...]

# 两者都处理：依次执行两个脚本
```

| 参数 | 是否可选 | 说明 | 适用脚本 |
| --- | --- | --- | --- |
| `-d, --dir` | 可选 | 目标目录（递归全部 `.md`）或单个 `.md` 文件，默认 `<仓库根>/docs/zh` | 两者 |
| `-b, --branch` | 可选 | 目标版本分支，默认自动获取当前 git 分支 | update_links |
| `--npu` | 可选 | 待删除的 NPU 标签类型，可多个；缺省从文档写作规范"分支文档维护要求-产品标签使用要求"在线获取不允许存在的标签范围（仅查询一次） | remove_npu_tags |
| `--output` | 可选 | 修改清单写入指定文件（markdown） | 两者 |

## 工作流

1. 在需要修改的版本分支上，按需求意图选择模式直接执行（只更新链接 / 只删除产品内容 / 两者都处理）；
2. 查看输出的修改清单（分支映射、逐文件命中统计、链接替换前后对照、NPU 删除/改写统计；update_links 修改完成后会自动对修改后的链接做在线有效性校验，404 无效链接会告警）；
3. 复查后按需提交，必要时 `git checkout` 回退。改动文件较多时用范围级复查代替逐行人审：
   - `git diff --stat` 文件清单、增删行数与修改清单对账（不一致即有异常）；
   - 抽查代表文件的 `git diff` 明细；
   - 重跑一次应零命中（幂等自检，确认无遗漏）。

```bash
# 典型用法 (两个脚本均幂等: 对同一目标执行后重跑一次应零命中, 可作完备性自检; 自检须带相同 -d)
python3 infra/docs-branch-update/scripts/update_links.py -d docs/zh/guide --output 清单.md   # 直接修改+清单落盘
python3 infra/docs-branch-update/scripts/remove_npu_tags.py -d docs/zh/guide --output 清单2.md
python3 infra/docs-branch-update/scripts/update_links.py -d docs/zh/guide                   # 幂等自检: 应零命中
python3 infra/docs-branch-update/scripts/remove_npu_tags.py -d docs/zh/guide                 # 幂等自检: 应零命中
```

## 处理规则摘要

- 链接更新（7 规则）：gitcode cann 仓 `master` -> 当前版本分支（仅匹配完整 `master` 分支段，`master-backup` 等前缀同名的其他分支不受影响；`cann-samples`、`pypto-gym` 除外）；`cann/pypto-gym` 仓（无 beta 分支）`master` -> 上一个正式版本（beta 按分组规则，如 `9.3.0-beta.1` -> `9.2.0`）；gitcode Ascend 仓 `master` -> 映射 Ascend 分支（`9.X.0`->`26.X.0`、`9.X.0-beta.N`->`26.(X-1).0`，`Ascend/docs`、`Ascend/pytorch` 除外）；`Ascend/pytorch` -> CommonTerms 手册名称汇总表在线查询（懒加载，运行过程仅查询一次）；hiascend 三产品 `latest` -> `930beta1`/`2620`/`920`。
- 产品内容删除：单一类型块整删（id 配对、行首前缀整行、CRLF 兼容）、混合标签列表移除目标类型、清空即删、嵌套子块删除保留外层；接缝处 3+ 连续 `<br>` 整删、表格尾 `<br>\|` 收敛、3+ 空行收敛（均局部生效，不误删与标签无关的既有格式）。

完整规则、分支映射表与约束说明见 [SKILL.md](./SKILL.md)。

## 依赖与约束

- Python 3（仅标准库），执行环境需可运行 `git`（识别仓库根目录与当前分支）；
- `update_links.py` 的 `Ascend/pytorch` 仓分支映射需在线访问 gitcode.com（懒加载，仅当目标中存在该类链接时才查询），网络不可用时该类链接保持 `master` 并告警；
- `update_links.py` 修改完成后自动在线校验修改后链接的有效性（gitcode 链接校验分支与路径存在性；hiascend 为 SPA 站点跳过校验），网络不可用时校验结果报告为不可达；并发校验内置 WAF 限流防护（全局限速约 ≤10 req/s，429 按 Retry-After/指数退避重试、耗尽报“限流”，403 延迟重试一次），限流与不可达区分避免误报；
- `remove_npu_tags.py` 的缺省标签范围需在线访问 gitcode.com 获取文档写作规范"分支文档维护要求-产品标签使用要求"（仅在未显式指定 `--npu` 时查询一次），网络不可用时退出并提示显式指定 `--npu`；
- 全程字节级处理，保持文件原有编码（UTF-8/GBK）与行尾（LF/CRLF）不变；
- 当前分支为 `master`/`main`/`HEAD` 时链接更新拒绝执行（更新无意义），可切换版本分支或 `-b` 显式指定。
