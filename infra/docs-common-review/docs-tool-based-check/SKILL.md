---
name: docs-tool-based-check
description: |
  基于 CANN 社区 ci-staticCheck 工具链（docs-ci-v2.js 脚本）对 Ascend 生态仓的 Markdown 文档执行本地静态检查（测试），覆盖五项检查能力：markdownlint（Markdown 语法）、tag-closed-check（HTML 标签闭合）、link-validity-check（链接有效性）、resource-existence-check（资源存在性）、codespell-check（单词拼写），并将脚本结果转换为结构化问题列表（含严重程度、位置、修改建议）。当用户提到以下任一场景时触发：用工具/脚本/静态检查方式检查 Markdown 文档质量、markdownlint 或 MD 规则检查、链接有效性/失效链接/死链检查、图片等资源存在性检查、codespell 单词拼写检查、HTML 标签闭合检查。仅承接用户点名的工具类专项检查（如"只跑 markdownlint"、"只查链接"）；通用质检、文档门禁（CI 门禁项）本地预检、提交前综合自检请使用统一入口 docs-common-review（其门禁模式直接编排本技能的 scripts/run_check.js）。
version: 1.1.0
metadata:
  author: Huawei Technologies Co., Ltd.
  category: utility-doc
---

# 生态资料工具链静态质检技能

本 skill 调用 CANN 社区 [ci-staticCheck 工具链](https://gitcode.com/cann/.gitcode/blob/master/actions/staticcheck/dist/staticcheck/docs-ci/docs-ci-v2.js)的 `docs-ci-v2.js` 脚本，在用户本地对 Markdown 文档执行静态检查。脚本为自包含打包产物，运行时按 pin 版本拉取并缓存（当前 pin 版本见"维护说明"）；其中 codespell-check 所需 cspell 词典未随脚本内联，运行时从脚本缓存目录解析，本次检查项包含 codespell-check 时须按第 2 步先检查词典环境、缺失时安装，否则普通英文单词会被大量误报。

| 检查项 | checkType | 功能说明 |
| --- | --- | --- |
| Markdown 语法检查 | markdownlint | 基于 markdownlint 规则（MDxxx）检查标题、列表、代码块等语法规范性 |
| HTML 标签闭合检查 | tag-closed-check | 检查文档中 HTML 标签是否正确闭合与配对 |
| 链接有效性检查 | link-validity-check | 检查内链、外链、锚点是否可访问 |
| 资源存在性检查 | resource-existence-check | 检查图片等资源引用是否存在 |
| 单词拼写检查 | codespell-check | 基于 cspell 词典与白名单的英文拼写检查（词典未安装时会大量误报，须按第 2 步先检查、缺失时安装） |

**检查项范围仅限上表 5 项**（config.json `ci` 数组的合法值同此，未配置时默认全部执行）。脚本另支持 toc-check、file-naming-check 等，本技能不暴露，视为非法输入。

## 重要原则

1. **只报告脚本真实检出的问题**：最终结果只来自 `scripts/run_check.js` 写出的结果文件，逐条原样转述，禁止增、删、改写；检出 0 项时如实报告"未检出问题"，禁止编造。
2. **每条问题独立报告**：禁止合并同类问题（如同行多个拼写错误）。
3. **检查过程只读取待检查文档**，禁止修改。
4. **执行期间保持静默**：自第 2 步运行检查脚本起至第 3 步汇报前，除 CHECKPOINT 外禁止输出解释性文字、中间进度、总结或表格。
5. **node 运行环境以用户声明为准**：用户明确声明本地未安装 node 时，无条件停止检查并引导安装（见第 1 步「node 运行环境校验」），禁止以当前执行环境（agent 所处环境）中的 node 替代用户本地环境继续执行并产出结果。

## 输入

1. **待检查对象**：一个 Markdown 文档路径、多个路径（英文逗号分隔）、一个文档目录或一个仓库路径。
2. **检查项范围**：用户对话中指定（如"只检查链接和拼写"），仅本次生效；未指定时必须先以可勾选（多选）的选项形式询问用户勾选（见第 1 步），禁止不询问直接使用 config.json 默认范围；仅当用户明确跳过选择时才回退 config.json 默认。如需持久化，提示用户写入 config.json。

### 示例

```
# 全部 5 项检查（未指定范围时会先弹出勾选项）
/docs-tool-based-check "文档路径"

# 对话中指定检查范围：只检查链接和拼写
/docs-tool-based-check "文档路径" 只检查链接有效性和单词拼写
```

## 执行流程

在执行任何步骤前，在对话中先输出 `[CHECKPOINT: 开始]`。

**规则**：如果前一个 CHECKPOINT 未输出，禁止执行后一步骤。

### 第 1 步：解析输入与运行环境校验

识别待检查目标（文件/目录路径），确认项目根目录（通常为当前工作目录）。

**node 运行环境校验（必须先于检查项范围确定）**：node 为 helper 脚本（scripts/run_check.js）与检查脚本的必需运行环境，**以用户声明的本地环境为准**，禁止以当前执行环境（agent 所处环境）中的 node 替代用户本地环境继续执行：

1. **用户明确声明本地未安装 node（或明确表示其本地环境无 node）**：无条件立即停止检查流程——不运行任何脚本、不产出任何检查结果、不进入检查项范围询问。随后可检测执行环境（运行 `node -v`），结果仅用于确定引导方式，**不得据此恢复执行**：
   - `node -v` 不可用（此时执行环境通常即用户本地环境）：执行下方「协助安装」，安装成功且复检通过后可继续本次检查流程。
   - `node -v` 可用（执行环境非用户本地环境，如远端沙箱/CI）：向用户说明"执行环境中检测到 node 不代表用户本地可用，且无法代为安装用户本地环境"，给出各平台安装命令（见下表），请用户在其本地安装完成后重新发起；仅在用户确认其本地 node 可用（安装完成，或澄清此前为误判/PATH 问题）后才继续执行。
2. **用户未声明环境状态**：检测执行环境，运行 `node -v`：可用 → 继续；不可用（命令不存在）→ 执行下方「协助安装」。
3. **协助安装**：向用户说明 node 缺失，并直接帮用户安装——先识别操作系统，再执行对应安装命令，无需用户手动操作：

   | 操作系统 | 安装命令 |
   | --- | --- |
   | Windows | `winget install OpenJS.NodeJS.LTS`（winget 不可用时引导从 https://nodejs.org 下载安装） |
   | Ubuntu/Debian | `sudo apt-get install -y nodejs npm` |
   | EulerOS/openEuler | `sudo yum install -y nodejs npm` |
   | 其他（含 macOS） | 引导从 https://nodejs.org 下载安装（macOS 亦可用 `brew install node`） |

   安装完成后复检 `node -v`：可用 → 继续后续流程；因当前会话 PATH 未刷新而识别不到新装 node → 提示用户重开终端/会话后重试；复检仍不可用（安装失败、无权限、无网络或包管理器不可用）→ 停止执行，给出上表对应的手动安装指引，不得编造或虚构任何检查结果。

用户未指定检查项范围时，必须先以可勾选（多选）的选项形式询问用户，勾选项共 6 个：5 个检查项（markdownlint、tag-closed-check、link-validity-check、resource-existence-check、codespell-check）逐项勾选，外加"全部 5 项（默认）"勾选项。勾选"全部 5 项（默认）"时，本次执行全部 5 项（等价于逐项全勾）；仅勾选部分检查项时，本次只执行所勾选项。按用户勾选结果确定本次范围，仅本次生效；仅当用户明确跳过选择时，才回退使用 config.json 默认范围。

完成后在对话中输出 `[CHECKPOINT: 输入解析完成]`

### 第 2 步：运行检查脚本

**codespell 词典环境检查（仅当本次检查项包含 codespell-check）**：检查脚本运行时从其缓存目录解析 cspell 词典，词典缺失时普通英文单词（如 quick、normal）会被大量误报为"拼写错误"。必须先检查是否已安装，缺失才安装，已安装则跳过：

1. 词典与检查脚本共用缓存目录：`<系统临时目录>/docs-tool-based-check`（可用 `node -e "console.log(require('path').join(require('os').tmpdir(),'docs-tool-based-check'))"` 获取实际路径）。
2. 检查是否已安装：该目录下存在 `node_modules/@cspell/cspell-bundled-dicts` 即视为已安装，直接执行下方检查命令。
3. 未安装时安装：先在缓存目录写入如下 `package.json`（与社区 [docs-ci 依赖清单 package.json](https://gitcode.com/cann/.gitcode/blob/master/actions/staticcheck/dist/staticcheck/docs-ci/package.json) 保持一致，词典即来源于此依赖声明），再在该目录执行 `npm install --no-fund --no-audit`：

   ```json
   {
     "name": "docs-ci",
     "version": "0.0.1",
     "dependencies": {
       "cspell-lib": "9.2.1"
     }
   }
   ```

4. 安装失败（无法访问 npm 源或 npm 不存在）时停止执行，向用户说明原因，建议修复网络或安装 npm 后重试，或本次缩小检查范围（去掉 codespell-check）。

随后通过 Bash 执行 helper 脚本（一条命令完成配置解析、脚本拉取、执行与结果转换）：

```bash
node <技能目录绝对路径>/scripts/run_check.js \
  --targets=<路径1,路径2,...> \
  [--checks=<检查项1,检查项2,...>] \
  [--project-root=<项目根目录>] \
  [--output=<结果文件路径>] \
  [--disable-external] \
  [--proxy=<代理地址>] \
  [--concurrency=<并发数>]
```

- `targets` 必需，多值逗号分隔，含空格时整体加引号；其余参数可选。
- `checks` 不传则用 config.json 默认（全部 5 项）；`project-root` 默认当前工作目录，用于 `.doctools/` 配置解析与结果输出；`output` 默认 `<project-root>/output/tool-based-check-result.json`。
- 内网/无外网环境加 `--disable-external` 跳过外链与外部资源检查，避免误报；`--proxy` 指定外链检查代理；文件多时可调大 `--concurrency`（并发文件数）。

完成后在对话中输出 `[CHECKPOINT: 静态检查完成]`

### 第 3 步：汇报结果

读取结果文件，向用户输出：问题总数与严重程度统计、逐条问题（文件、行号、问题、建议）。误报与有意保留项的处理见"异常与常见问题"。

完成后在对话中输出 `[CHECKPOINT: 结果输出完成]`


## 输出

输出一个质检结果JSON文件

- 默认输出目录：当前项目根目录下创建 `/output/` （如果目录不存在则先创建目录）
- 文件命名规则：`tool-based-check-result.json`

### 质检结果统一 JSON 格式

```json
[
  {
    "rule_group": "xxx",
    "rule_item": "xxx",
    "rule_type": "xxx",
    "rule_name": "xxx",
    "snippet": "xxx",
    "description": "xxx",
    "rule_level": "xxx",
    "file_name": "xxx",
    "line_number": "xxx",
    "location": "xxx",
    "suggestion": "xxx"
  }
]
```

#### 输出字段说明

| 字段名 | 类型 | 描述 | 要求 |
| --- | --- | --- | --- |
| rule_group | string | 问题大类 | 固定为"通用规则" |
| rule_item | string | 规则项名称 | 必须与规则文件中的规则项名称一致 |
| rule_type | string | 规则类型 | 必须与规则文件中的规则类型一致 |
| rule_name | string | 规则名称 | 必须与规则文件中的规则名称一致 |
| snippet | string | 问题内容片段 | 直接引用文档内容 |
| description | string | 问题的具体描述 | 清晰说明问题，不包含行号 |
| rule_level | string | 级别 | 必须与规则文件中的级别一致 |
| file_name | string | 文档名称 | 必须是用户输入的Markdown文档名称 |
| line_number | string | 问题行号 | 必须从带行号的文档内容中直接获取 |
| location | string | 问题所在位置 | 仅"绝对路径:行号"（如 e:/docs/readme.md:13），禁止附加 vscode:// 等协议前缀或任何链接格式；无行号时仅绝对路径 |
| suggestion | string | 问题的具体修改建议 | 直接给出正确形式，不包含行号 |

**类型与严重程度映射**（分级依据：直接影响读者获取内容的定为严重，可辨识的错误定为提示，纯规范性问题按规则影响面分级）：

- link-validity-check、resource-existence-check、tag-closed-check → 正确性 / 严重
- codespell-check → 正确性 / 提示
- markdownlint → 规范性 / 按规则号查 [references/markdownlint-severity.json](references/markdownlint-severity.json)，映射未覆盖的规则默认"一般"

## 配置文件

### 技能目录 config.json（默认检查项范围）

通过 `ci` 数组配置默认检查项，合法值仅上表 5 项。

### 项目级 `<项目根>/.doctools/`（用户自定义）

| 文件 | 作用 | 语义 |
| --- | --- | --- |
| `codespell.whitelist.txt` | 拼写白名单，每行一个单词 | **增量**：叠加在内置社区默认白名单之上 |
| `url.whitelist.txt` | 链接/资源白名单，每行一条正则 | **增量**：叠加在内置社区默认白名单之上 |
| `markdownlint.config.json` | markdownlint 规则开关（关闭某规则设为 `false`，参见 [markdownlint 规则](https://github.com/DavidAnson/markdownlint/blob/main/doc/md001.md)） | **替换**：整体替换脚本内置默认规则配置 |

未创建项目级文件时：白名单使用内置社区默认（技能目录 `whitelist_words.txt`，约 4500 词领域术语；`url.whitelist.json`），markdownlint 使用脚本内置默认规则配置。

## 异常与常见问题

| 情形 | 处理方式 |
| --- | --- |
| node 命令不存在 | 按第 1 步「node 运行环境校验」的「协助安装」处理：直接帮用户安装（按操作系统选择命令），复检仍不可用时才停止执行并给出手动安装指引 |
| 用户声明本地未安装 node，但执行环境中 node 可用 | 以用户声明为准：停止执行，说明执行环境非用户本地环境、无法代为安装，给出各平台安装命令（见第 1 步）；禁止以执行环境的 node 继续执行并产出结果 |
| 非法检查项 | 停止执行，向用户说明非法值并列出全部 5 项合法值，待修正后重试；禁止将非法值写入 config.json |
| 脚本下载失败 | 提示检查网络/代理后重试 |
| 检查脚本执行失败 | 向用户转述脚本错误输出；helper 会保留运行目录（报错信息中有路径）供诊断 |
| codespell 对英文行文报大量"拼写错误"（quick、normal 等常用词也被报） | 词典缺失所致：未按第 2 步安装 cspell 词典时，普通英文单词会被整体误报。先完成词典安装后重跑；个别专有名词或领域词误报，再引导用户加入项目 `.doctools/codespell.whitelist.txt` |
| 安装 cspell 词典失败（npm 不存在或无法访问 npm 源） | 停止执行，向用户说明原因；建议安装/修复 npm 与网络后重试，或本次缩小检查范围（去掉 codespell-check） |
| 外链大量误报 | 网络不通导致：建议重试时加 `--disable-external`，或确认代理设置（`--proxy`） |
| 某失效链接为有意保留 | 引导用户加入项目 `.doctools/url.whitelist.txt`（支持正则） |
| 某 markdownlint 规则不适用 | 引导用户创建项目 `.doctools/markdownlint.config.json` 关闭该规则 |

## 维护说明

- 检查脚本运行时从 CANN 社区拉取并缓存在系统临时目录，**不固化副本在本仓库**，禁止将脚本副本提交到本仓库。当前 pin 版本：`c5ccfec7c5550b2a8a45feb26e7cd5fac4bd607b`（与 `scripts/run_check.js` 的 `SCRIPT_SHA` 常量一致）。
- 更新方法：确认社区脚本无接口变更后，修改 `SCRIPT_SHA` 为新 commit，同步更新本节，并重跑 evals 验证。
- 评测数据集 `evals/dataset/static_check_sample.md` 为 golden 基准：五项检查各埋确定性问题，且无外链与外部资源（结果离线确定、可复现）；修改该文件或更新 `SCRIPT_SHA` 后，必须重跑脚本并以实测结果同步 `evals/evals.json` 中的 golden 描述。
- 内置白名单与社区 [.doctools 配置](https://gitcode.com/cann/.gitcode/blob/master/actions/staticcheck/dist/staticcheck/docs-ci/.doctools)同源，以社区为真源，禁止在仓内修改；修改社区配置请走 CANN 社区 PR 贡献流程。
- cspell 词典依赖与社区 [docs-ci 依赖清单 package.json](https://gitcode.com/cann/.gitcode/blob/master/actions/staticcheck/dist/staticcheck/docs-ci/package.json) 同源（当前 `cspell-lib@9.2.1`）：按第 2 步安装在脚本缓存目录（系统临时目录），随缓存清理，不固化副本在本仓库。社区更新该依赖版本时，须同步修改第 2 步 package.json 示例中的版本号。
- 运行中间产物（ciConfig.json、result.json、output.md）由 helper 自动清理，禁止作为最终结果输出给用户；最终结果以 `output/tool-based-check-result.json` 为准。

## 文件结构

```
docs-tool-based-check/
├── SKILL.md                       # 本文件
├── config.json                    # 默认检查项范围（全量 5 项）
├── whitelist_words.txt            # 内置社区拼写白名单
├── url.whitelist.json             # 内置社区链接白名单
├── scripts/run_check.js           # 运行器（配置解析、脚本拉取、执行、结果转换）
├── references/                # 结果转换属性：check-attributes.json、markdownlint-severity.json（字段语义见 check-attributes.md）
└── evals/                         # 评测用例与数据集
    ├── evals.json                 # 评测用例（关键能力用例 + 范围勾选/非法检查项/环境缺失纪律用例）
    └── dataset/
        └── static_check_sample.md # 综合数据集：五项检查各埋确定性问题，作为 golden 基准
```
