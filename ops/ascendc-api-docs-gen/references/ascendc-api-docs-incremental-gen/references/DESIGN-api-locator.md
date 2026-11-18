# API Locator — Ascend C API 代码定位与文档重组工具

从 Ascend C 代码库中自动定位指定 API 的所有相关文件（头文件、实现、测试、示例、文档），打包为重组目录，并按**四维表**（API 类型 × 证据维度）生成结构化章节 JSON。

## 快速开始

```bash
# 输入接口文档，输出重组文件夹及所有章节 JSON
python cli.py docs/zh/api/Add.md

# 指定提取特定章节
python cli.py docs/zh/api/Add.md --chapters "函数原型,返回值说明"
```

**输入**：一个接口文档路径（如 `docs/zh/api/SIMD-API/基础API/Reg矢量计算/基础算术/Add-20.md`）
**输出**：`output/{API名}/` 目录，包含：
- `refactor/` ——头文件、impl 文件、test 文件、示例、文档，按原仓路径组织
- `doc_json/{API名}_doc.json` ——文档解析中间 JSON，存储各章节原文
- `anchor/{sub_part}.json` ——7~8 个章节 anchor JSON，每个对应一个文档章节

## 目录结构

```
api_locator/
├── config.py                 # 配置：REPO_ROOT 代码仓库根目录路径（使用前需修改）
├── cli.py                    # [模块①] 命令行入口：解析参数，编排完整流程
├── file_regroup.py           # [模块②] 文件重组导出：export_regroup() 收集文件并打包
├── doc_parser.py             # [模块③] 文档解析：按 ## 标题分割 Markdown，输出 {API名}_doc.json
├── evidence.py               # [模块④] 核心引擎：APILocator._build_sections() 委托 chapters/ 提取证据
├── chapters/                 # [模块④] 8 个章节独立提取器，各按四维表提取 Evidence
│   ├── __init__.py
│   ├── ch1_chip_support.py   #     产品支持情况
│   ├── ch2_function_desc.py  #     功能说明
│   ├── ch3_impl_principle.py #     实现原理（adv_api 专用）
│   ├── ch4_function_proto.py #     函数原型
│   ├── ch5_param_desc.py     #     参数说明
│   ├── ch6_return_type.py    #     返回值说明
│   ├── ch7_constraints.py    #     约束说明
│   └── ch8_examples.py       #     调用示例
├── formatter.py              # [模块⑤] JSON 格式化：source → anchor 转换，写入章节 JSON
├── 项目说明.md               #       本文件（不参与主流程）
```

## 整体流程图

### 模块总览

```
┌─────────────────────────────┐
│  ① cli.py                  │
│  编排完整流程                │
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  ② file_regroup.py          │
│  export_regroup()           │
│  收集文件 + 打包到 output/{API名}/ │
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  ③ doc_parser.py            │
│  解析接口文档 → JSON          │
│  save_chapters_json()       │
│  → {API名}_doc.json          │
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  ④ evidence.py + chapters/  │
│  APILocator._build_sections()│
│  读 doc JSON → 只提取存在的章节│
│  chapters/ch1~ch8 独立提取    │
└─────────────┬───────────────┘
              │ sources (Evidence 列表)
┌─────────────▼───────────────┐
│  ⑤ formatter.py             │
│  读 doc JSON + sources       │
│  → anchors → 章节 JSON       │
└─────────────────────────────┘
```

### 模块职责

| 模块 | 涉及代码 | 输入 | 输出 | 职责 |
|------|---------|------|------|------|
| **cli.py** | `cli.py` | `docs/zh/api/SIMD-API/.../Add-20.md [--chapters "函数原型,参数说明"]` | `output/{API名}/` + `anchor/{sub_part}.json` + `doc_json/{API名}_doc.json` | 命令行入口，编排全流程，支持 `--chapters` 按需提取 |
| **文件重组** | `file_regroup.py` | API 名 + 文档路径 | `output/{API名}/refactor/` 目录 | 收集头文件/impl/test/示例/文档，按原路径拷贝 |
| **文档解析** | `doc_parser.py` | 重组后的文档 | `Add_doc.json` | 按 ## 标题分割 Markdown，输出中间 JSON |
| **证据提取** | `evidence.py` + `chapters/` | doc JSON + impl 文件 + API 名 + 类型 | 各章节 Evidence 列表 | 定位 API 类型 + 查找文件 + 提取四维证据 |
| **JSON 格式化** | `formatter.py` | sources + doc JSON | `anchor/{sub_part}.json` | source → anchor 转换，注入文档原文 |

### 执行流程

```
Step ①: 总入口                  cli.run() → 编排全流程
  ↓
Step ②: 文件重组               export_regroup() → 收集文件 + 打包到 output/{API名}/
  ↓
Step ③: 文档解析 → JSON         doc_parser.save_chapters_json() → {API名}_doc.json
  ↓
Step ④: 证据提取                APILocator._build_sections(api_type, decl_file, doc_chapters_json_path, selected_chapters)
                                chapters/ch1~ch8 读取 doc JSON，只提取文档中存在 / 用户指定的章节
  ↓
Step ⑤: 证据输出               formatter 读 doc JSON + sources → anchors → anchor/{sub_part}.json
```

## 核心概念

### 四维表

每个 API 类型在每个文档章节需要提取四种证据维度：

| 维度 | 说明 | 示例 |
|------|------|------|
| `file` | 文件级证据 | 头文件路径、impl 文件路径 |
| `function` | 函数级证据 | CheckFunc、Impl 函数签名 |
| `code_snippet` | 代码段级证据 | arch 条件、static_assert、声明代码段 |
| `comment_block` | 注释段级证据 | Doxygen 注释 |

### Anchor 类型（4 种，对应四维表）

| Anchor 类型 | 对应四维维度 | 说明 |
|-------------|-------------|------|
| `file` | file | 文件路径级证据 |
| `function` | function | 函数级证据（含 full_function 完整实现） |
| `section` | code_snippet | 代码段级证据 |
| `note` | comment_block | 注释段级证据 |

### 7 种 API 类型

| 类型 | 说明 | 识别特征 |
|------|------|---------|
| `basic_api` | 基础矢量 API | `include/basic_api/` 下的声明文件 |
| `adv_api` | 高级 API | `include/adv_api/` 下的声明文件 |
| `simt_api` | SIMT 标量 API | `include/simt_api/` 下的声明文件 |
| `tensor_api` | Tensor API | `include/tensor_api/` 下的声明文件 |
| `c_api` | C 语言 API | `include/c_api/` 下的声明文件 |
| `aicpu_api` | AI CPU API | `include/aicpu_api/` 下的声明文件 |
| `utils` | 工具函数 | `include/utils/` 下的声明文件 |

### 8 个文档章节

| 章节 | sub_part | 说明 |
|------|----------|------|
| 产品支持情况 | support | 各芯片架构的实现情况 |
| 功能说明 | summary | Doxygen brief 注释 |
| 实现原理 | impl_principle | adv_api 专用 |
| 函数原型 | function_proto | 声明代码段 |
| 参数说明 | para_clarify | Doxygen param 注释 |
| 返回值说明 | return_clarify | 返回值类型 |
| 约束说明 | constraint | 文件 + 函数 + 代码段 + 注释段 |
| 调用示例 | example | 测试文件和示例文件调用 |

## 输出格式

每个章节 JSON 文件结构：

```json
{
  "doc_path": "docs/zh/api/SIMD-API/基础API/Reg矢量计算/基础算术/Add-20.md",
  "doc_name": "Add-20.md",
  "doc_part": "函数原型",
  "sub_part": "function_proto",
  "function_proto": { "content": "..." },
  "anchor_tuple": [
    {
      "anchor_id": "anchor_001",
      "note": "file类型标准结构",
      "anchor_information": {
        "type": "file",
        "level1": { "file_path": "...", "file_name": "..." },
        "level2": { "code_map": "...", "token_hash": "...", "file_功能": "..." },
        "level3": { "tree_id": "..." }
      },
      "evidence": [
        {
          "evidence_id": "ev_001",
          "evidence_information": { "file_path": "..." },
          "gene_rule": { "target": ["function_proto"] }
        }
      ]
    }
  ]
}
```

其中 `gene_rule.target` 字段对应章节的 sub_part，指明该证据属于哪个文档模块。

产品支持情况章节额外注入 `arch` 字段：

```json
{
  "doc_part": "产品支持情况",
  "arch": ["dav_3510", "dav_5102", "dav_c100", ...],
  ...
}
```

## 配置

所有路径配置集中在 `config.py`：

```python
# 代码仓库根目录（asc-devkit 路径）
REPO_ROOT = os.environ.get("ASCDEVKIT_REPO_ROOT", "<asc-devkit 仓库路径>")
```

换环境时只需改这一行，`cli.py` 和 `file_regroup.py` 都会从该配置读取仓库路径。

### 路径拼接规则

所有搜索路径基于 `REPO_ROOT` 拼接：

| 目标 | 拼接方式 | 示例 |
|------|---------|------|
| 声明文件 | `REPO_ROOT` + `include/{类型}/` | `{REPO_ROOT}/include/basic_api/kernel_operator.h` |
| impl 文件 | `REPO_ROOT` + `impl/{类型}/**` | `{REPO_ROOT}/impl/basic_api/**/*AddImpl*` |
| test 文件 | `REPO_ROOT` + `tests/api/{类型}/` | `{REPO_ROOT}/tests/api/basic_api/` |
| 文档文件 | `REPO_ROOT` + `docs/zh/api/` | `{REPO_ROOT}/docs/zh/api/SIMD-API/.../Add-20.md` |
| 示例文件 | `REPO_ROOT` + `examples/` | `{REPO_ROOT}/examples/add/` |

### 代码内部约定

以下路径在代码中硬编码，asc-devkit 仓库目录结构变化时需同步修改：

#### 7 种 API 类型的 impl 搜索路径（`evidence.py` `_build_impl_patterns`）

| 类型 | 硬编码 impl 路径 |
|------|-----------------|
| `basic_api` | `impl/basic_api/dav_*/{base_name}_impl.h`、`kernel_operator_vec_binary_impl.h`、`kernel_operator_vec_unary_impl.h`、`kernel_operator_*.h`、`dav_*/*_impl.h` |
| `adv_api` | `impl/adv_api/detail/{subdir}/{fname}/{fname}_base_impl.h`、`dav_*/*{fname}*impl*.h`、`v*/*{fname}*impl*.h`、`3510/*{fname}*impl*.h`、`{fname}_impl/*{fname}*impl*.h` |
| `simt_api` | `impl/simt_api/cpp/{base}_impl.h`、`impl/simt_api/cpp/dav_3510/kernel_simt_math_impl.h`、`impl/simt_api/{base}_impl.h` |
| `tensor_api` | `impl/{rel.replace('.h', '_impl.h')}`（声明文件同目录名 + `_impl.h`） |
| `c_api` | `impl/c_api/instr_impl/npu_arch_*/{base}` |
| `aicpu_api` | 无 impl，返回空列表 |
| `utils` | `impl/utils/{subdir}/{fname}/*.h` |

#### check 文件搜索路径

| 类型 | 硬编码路径 | 位置 |
|------|-----------|------|
| basic_api | `impl/basic_api/utils/kernel_check_*.h` | `file_regroup.py`、`evidence.py` |
| adv_api | `impl/adv_api/detail/api_check/kernel_api_check.h`、`impl/adv_api/detail/api_check/kernel_check/basic_check/` | `evidence.py` |
| adv_api（旧） | `impl/adv_api/detail/**/*check*.h` | `file_regroup.py` |

#### 其他硬编码

| 约定 | 位置 | 说明 |
|------|------|------|
| `docs/zh/api/` | `file_regroup.py` | "更多样例"链接的拼接路径 |
| `tests/api/{类型}/` | `evidence.py` TEST_DIR_MAP | 测试文件搜索目录，7 种类型全部映射 |
| `include/basic_api/kernel_tensor.h` | `evidence.py` | 运算符重载（operator+）声明文件 |
| `include/basic_api/kernel_operator` | `evidence.py` | 运算符重载最佳匹配优先路径 |

### 典型验证 API

每次验证时跑一遍这 6 个 API，覆盖 4 种 API 类型：

| API | 类型 | 输入路径 |
|-----|------|---------|
| `Add` | basic_api | `docs/zh/api/SIMD-API/基础API/Reg矢量计算/基础算术/Add-20.md` |
| `Fixpipe` | tensor_api | `docs/zh/api/tensor_api/data_move/Fixpipe.md` |
| `LayerNorm` | adv_api | `docs/zh/api/SIMD-API/高阶API/归一化操作/LayerNorm.md` |
| `sqrtf` | simt_api | `docs/zh/api/SIMT-API/数学函数/float类型数学库函数/sqrtf.md` |
| `asc_init` | c_api | （无文档，`locate_by_name`） |
| `printf` | basic_api | `docs/zh/api/SIMD-API/基础API/调试接口/上板打印/printf.md` |

```bash
# 一键验证全部（有文档的 5 个）
python cli.py docs/zh/api/SIMD-API/基础API/Reg矢量计算/基础算术/Add-20.md
python cli.py docs/zh/api/tensor_api/data_move/Fixpipe.md
python cli.py docs/zh/api/SIMD-API/高阶API/归一化操作/LayerNorm.md
python cli.py docs/zh/api/SIMT-API/数学函数/float类型数学库函数/sqrtf.md
python cli.py docs/zh/api/SIMD-API/基础API/调试接口/上板打印/printf.md

# asc_init 无文档，需用 locate_by_name 验证
```


## 附：anchor 数据格式

代码内部以「文档定位 + 锚点数组」的结构描述结果（原样例文件 `src/api_locator/format.json` 已移除，
以本节为准）：

```
{
  "doc": {
    "doc_path": "", "doc_name": "", "doc_part": "",
    "sub_part": { "content": "" }
  },
  "anchor_tuple": [
    {
      "anchor_id": "",
      "note": "file 类型标准结构",
      "anchor_information": {
        "type": "file",                 // file / code / table / figure ...
        "level1": { "file_path": "", "file_name": "" },
        "level2": { "code_map": "", "token": "" },
        "level3": { }
      },
      "evidence": [ { } ]
    }
  ]
}
```

`anchor_information.type` 取 `file` 时按 `level1`（文件路径/文件名）定位；其余类型按对应层级
填 `code_map`、`token` 等字段。`evidence` 是同一条锚点关联的证据列表。
