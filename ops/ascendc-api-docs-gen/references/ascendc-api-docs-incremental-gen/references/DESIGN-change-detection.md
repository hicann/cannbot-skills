# API Change Detect — API 变更检测工具

给定变更前后两份代码，自动输出受影响的 API 列表。基于**反向索引**机制，增量检测仅需几秒。

## 快速开始

```bash
# 对比两个代码目录（首次使用，自动建索引）
python pipeline.py --old /path/to/old-repo --new /path/to/new-repo

# 已有索引，做增量更新
python pipeline.py --old /path/to/old-repo --new /path/to/new-repo \
    --index <索引路径>   # 默认 <本模块>/output/api_reverse_index.json，首次运行自动生成（不入库）

# 控制输出位置
python pipeline.py --old /path/to/old --new /path/to/new \
    --index-output output/my_experiment \          # 索引输出目录
    --json --output output/my_experiment/result.json  # JSON 结果文件
```

**输入**：变更前后代码目录（或 git commit 范围）
**输出**：受影响的 API 列表 + 反向索引文件

## 目录结构

```
api_change_detect/
├── config.py                   # 路径配置（PathConfig + REPO_ROOT）
├── pipeline.py                 # 一键入口：编排完整流程
├── api_change_detect.py        # 核心逻辑：索引构建、增量更新、变更检测
├── 项目说明.md                  # 本文件
├── data/                       # 代码数据
│   ├── asc-devkit/             # 新代码（主克隆）
│   └── old_repo/               # 旧代码（git worktree，实验用）
└── output/                     # 输出目录
    └── experiment/             # 实验结果
```

## 执行流程

```
场景 A：首次使用（无索引）

  输入 ①：旧代码目录              输入 ②：新代码目录
       │                               │
       ▼                               ▼
┌──────────────┐              ┌──────────────────┐
│  全量构建索引  │              │  变更检测          │
│  扫描文档目录  │              │  MD5 逐文件对比    │
│  提取 API 名   │              │  找变更文件        │
│  批量找关联文件 │              └────────┬─────────┘
│  构建反向索引  │                       │
│  (69s)        │              变更文件列表
└──────┬───────┘              [file1, file2, ...]
       │                             │
       └──────────┬──────────────────┘
                  ▼
         ┌───────────────┐
         │  受影响 API 查询│
         │  过滤：只看代码 │  ← 只算 include/ impl/ tests/ examples/
         │  变更查反向索引 │     docs/zh/api 变更不计入
         └───────┬───────┘
                 ▼
          受影响的 API 列表
          Add ← impl/basic_api/dav_3510/Add_impl.h


场景 B：日常使用（已有索引，检测变更）

  输入 ③：旧代码    输入 ④：新代码    输入 ⑤：上次的索引
       │                 │                  │
       └────────┬────────┘                  │
                ▼                           ▼
       ┌───────────────┐          ┌──────────────────┐
       │  变更检测       │          │  加载索引          │
       │  MD5 逐文件比对 │          │  校验 commit ID    │
       └───────┬───────┘          │  一致→增量 (5s)     │
               │                  │  不一致→全量 (90s)  │
        变更文件列表               └────────┬─────────┘
               │                           │
               └──────────┬────────────────┘
                          ▼
                 ┌─────────────────┐
                 │  受影响 API 查询 │
                 │  增量：只重算     │
                 │  受影响的 API    │
                 └────────┬────────┘
                          ▼
                   受影响的 API 列表
```

## 索引构建原理

整体分两步：**正查**（API → 文件）→ **翻转**（文件 → API）。

### 第一步：正查（建索引）

从文档目录提取所有 API 名，然后每个 API 去代码里搜关联文件。分 5 类：

| 类别 | 搜索方式 | 规则 |
|------|---------|------|
| 声明文件 | 一次 `grep` 扫全仓 | 在 `include_dirs` 下匹配 `API名(` 或 `API名<`，每个 API 取最优一条 |
| 实现文件 | 按 API 类型分 6 种策略 | 见下方详细规则 |
| 测试文件 | 按类型限目录 walk | 在 `test_dir_map` 对应子目录下搜索 `.cpp/.cc/.py`，二进制匹配 `API名(` |
| 示例文件 | walk `examples/` | 目录名或文件名以 API 名开头（忽略大小写），前缀+分隔符也算匹配 |
| Check 文件 | glob 后批量扫 | `impl/basic_api/utils/kernel_check_*.h`，文件内容包含 API 名 |

**实现文件搜索规则（按 API 类型）：**

| 类型 | 规则 |
|------|------|
| `basic_api` | 声明文件名 `xxx_intf.h` → glob `impl/basic_api/dav_*/xxx_impl.h`。声明在 `reg_compute/reg_base` 下时额外 glob `kernel_operator_*_impl.h`。验证文件内容包含 `ApiNameImpl` |
| `adv_api` | glob `impl/adv_api/detail/**/*impl*.h`，路径含 `/{API名}/` 或 `_{API名}_` 即匹配 |
| `simt_api` | glob `impl/simt_api/**/*impl*.h`，同上按路径+文件名 |
| `tensor_api` | glob `impl/tensor_api/**/*impl*.h`，文件内容包含 API 名 |
| `c_api` | glob `impl/c_api/instr_impl/npu_arch_*/*`，文件内容包含 API 名 |
| `utils` | glob `impl/utils/**/*.h`，文件内容包含 API 名 |

> 注意：以上 impl 搜索的 glob 模式（`dav_*`、`npu_arch_*`、`kernel_operator_*_impl.h`、`kernel_check_*.h` 等）目前写死在 `api_change_detect.py` 中，仅适配 asc-devkit 仓库。如果换仓库结构，需要同步修改这些模式。

### 第二步：翻转

把"API → [文件1, 文件2]"翻成"文件 → [API1, API2]"：

```
正查结果（API → 文件）:
  Add → [add_impl.h, kernel_operator.h, Add.md]
  Sub → [add_impl.h, kernel_operator.h]

翻转得到反向索引（文件 → API）:
  add_impl.h        → [Add, Sub]
  kernel_operator.h → [Add, Sub]
  Add.md            → [Add]
```

### 第三步：反查（变更检测时）

代码变更 → 查反向索引 → 秒级输出受影响 API：

```
改了 add_impl.h → 查表 → Add, Sub 受影响
```

## 模块职责

| 模块 | 涉及代码 | 输入 | 输出 | 职责 |
|------|---------|------|------|------|
| **变更检测** | `pipeline.py` | 旧代码目录 + 新代码目录 | 变更文件列表 | 逐文件 MD5 对比，找新增/修改/删除 |
| **文档扫描** | `api_change_detect.py` | 文档目录（默认 `docs/zh/api`） | API 名称列表 | 递归扫描 md 文件，剥离前缀后缀提取 API 名 |
| **关联文件收集** | `api_change_detect.py` | API 名称列表 + 代码仓库 | 每类关联文件 | 批量 grep/glob 找声明/impl/测试/示例/check 文件 |
| **反向索引构建** | `api_change_detect.py` | 关联文件 + API 映射 | `{文件: [API]}` | 正向翻转为反向，记录 git commit 指纹 |
| **受影响 API 查询** | `api_change_detect.py` | 代码变更文件 + 反向索引 | 受影响 API 列表 | 精确匹配（查索引）+ 模糊匹配（新文件按文件名猜） |

### 增量更新流程

当已有旧索引 + 旧代码目录 + 新代码目录时，走增量更新：

```
diff(旧, 新) → 变更文件 + 删除文件
       │
       ▼
反查旧索引 → 受影响 API（哪些 API 关联了变更文件）
       │
       ▼
在新代码上只重算这些 API 的关联文件
       │
       ▼
从旧索引中移除旧数据 + 写入新数据
       │
       ▼
处理文档变更：新增文档→新 API，删除文档→移除 API
```

## 配置

所有路径配置集中在 `config.py`，通过 `PathConfig` 数据类管理：

```python
@dataclass
class PathConfig:
    doc_dir: str = "docs/zh/api"               # 文档目录
    doc_recursive: bool = True              # 是否递归扫描子目录
    doc_exclude_patterns: list[str] = [...]  # 排除的非 API 文档
    doc_extensions: list[str] = [".md"]     # 文档文件扩展名

    include_dirs: list[str] = [...]         # 声明文件搜索目录
    impl_basic: str = "impl/basic_api"      # basic_api impl 目录
    impl_adv: str = "impl/adv_api"          # adv_api impl 目录
    ...

    test_dir_map: dict = {...}              # 测试目录按类型映射
    examples_dir: str = "examples"

    changed_file_prefixes: list[str] = [...]  # 变更检测关注的前缀
    api_change_prefixes: list[str] = [...]    # API 变更判定前缀
                                              # include/ impl/ tests/ examples/ 变更算 API 变更
                                              # docs/zh/api 变更不算
```

换仓库结构时，只需修改 `PathConfig` 中的路径。默认值适配 `asc-devkit` 仓库结构。

### 命令行参数

| 参数 | 方向 | 说明 |
|------|------|------|
| `--old` | **输入** 路径 | 旧代码目录 |
| `--new` | **输入** 路径 | 新代码目录 |
| `--index` | **输入** 路径 | 已有的索引文件（跳过全量构建） |
| `--git-range` | 字符串 | git diff 范围（如 `HEAD~3`） |
| `--index-output` | **输出** 目录 | 索引保存目录 |
| `--json` | flag | 终端打印 JSON 格式结果 |
| `--output` | **输出** 路径 | JSON 结果写入文件 |
| `--repo` | **输入** 路径 | 仓库根目录（git 模式，默认 config.REPO_ROOT） |

## 索引指纹

索引文件末尾存储 git commit ID 作为指纹：

```json
{
  "impl/basic_api/add_impl.h": ["Add"],
  "__fingerprint__": "3d98a034f21d..."
}
```

加载索引时校验此指纹。旧代码目录的 HEAD commit 与指纹一致 → 增量更新；不一致 → 自动降级为全量构建。

## 输出格式

命令行输出：
```
[3/3] 查询受影响 API
  代码变更: 20 个（用于 API 判定）
  文档/图片变更: 60 个（不计入 API 变更）
  受影响 API: 3 个

  Add
    ← impl/basic_api/dav_3510/add_impl.h
    ← include/basic_api/kernel_operator.h
  Sub
    ← include/basic_api/kernel_operator.h
```

JSON 输出（加 `--json --output xxx.json`）：
```json
{
  "changed_files": ["impl/basic_api/dav_3510/add_impl.h", ...],
  "affected_apis": {
    "Add": ["impl/basic_api/dav_3510/add_impl.h", ...],
    "Sub": ["include/basic_api/kernel_operator.h"]
  }
}
```

## 速度

实测数据（2618 个文档，2121 个 API，4553 个文件）：

| 场景 | 耗时 |
|------|------|
| 首次全量构建 | ~90s |
| 增量更新 | ~8s |
| 查索引找受影响 API | <1s（毫秒级） |
