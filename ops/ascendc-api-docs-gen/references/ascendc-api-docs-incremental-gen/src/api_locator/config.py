# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""api_locator 配置常量：集中管理路径、文件名、章节名、API 类型等静态变量。"""

import os

# 注意：外部消费者通过 sys.modules["config"] 注入，必须用绝对导入

REPO_ROOT = os.environ.get(
    "ASCENDC_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "asc-devkit")),
)

CH_CHIP_SUPPORT = "产品支持情况"
CH_FUNCTION_DESC = "功能说明"
CH_IMPL_PRINCIPLE = "实现原理"
CH_FUNCTION_PROTO = "函数原型"
CH_PARAM_DESC = "参数说明"
CH_RETURN_TYPE = "返回值说明"
CH_CONSTRAINTS = "约束说明"
CH_EXAMPLES = "调用示例"

CHAPTERS = [
    CH_CHIP_SUPPORT,
    CH_FUNCTION_DESC,
    CH_IMPL_PRINCIPLE,
    CH_FUNCTION_PROTO,
    CH_PARAM_DESC,
    CH_RETURN_TYPE,
    CH_CONSTRAINTS,
    CH_EXAMPLES,
]

ADV_API_EXTRA_CHAPTER = "实现原理"

TEST_DIR_MAP = {
    "basic_api": "tests/api/basic_api",
    "adv_api": "tests/api/adv_api",
    "simt_api": "tests/api/simt_api",
    "tensor_api": "tests/api/tensor_api",
    "c_api": "tests/api/c_api",
    "aicpu_api": "tests/api/aicpu_api",
    "utils": "tests/api/utils",
    # #17 修复：补充缺失的测试目录
    "common": "tests/api/common",
    "reg_compute_api": "tests/api/reg_compute_api",
}

# #17 修复：补充搜索的公共测试目录（适用于所有 API 类型）
COMMON_TEST_DIRS = [
    "tests/api/common",
    "tests/api/reg_compute_api",
]

OPERATOR_TO_API = {
    "operator+": "Add",
    "operator-": "Sub",
    "operator*": "Mul",
    "operator/": "Div",
    "operator|": "Or",
    "operator&": "And",
    "operator<": "Less",
    "operator>": "Greater",
    "operator==": "Equal",
    "operator!=": "NotEqual",
    "operator<=": "LessEqual",
    "operator>=": "GreaterEqual",
}

# 运算符类型分类体系
OPERATOR_CATEGORIES = {
    # SymbolOverride 系：运算符是某个功能 API 的重载形式（已有 OPERATOR_TO_API 映射）
    "symbol_override": {
        "operators": [
            "operator+",
            "operator-",
            "operator*",
            "operator/",
            "operator|",
            "operator&",
            "operator<",
            "operator>",
            "operator==",
            "operator!=",
            "operator<=",
            "operator>=",
        ],
        "mapping": OPERATOR_TO_API,
    },
    # subscript 系：operator[] 紧跟参数（Tensor 偏移访问）
    "subscript": {
        "operators": ["operator[]"],
        "parent_classes": {
            "LocalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
            "GlobalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
        },
    },
    # element_access 系：operator() 紧跟参数（Tensor 元素值访问）
    "element_access": {
        "operators": ["operator()"],
        "parent_classes": {
            "LocalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
            "GlobalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
        },
    },
    # assignment 系：operator= 赋值
    "assignment": {
        "operators": ["operator="],
        "parent_classes": {
            "LocalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
            "OpMC2Def": {"decl_file": "include/utils/op_flag.h"},
            "OpAICoreConfig": {"decl_file": "include/utils/op_flag.h"},
        },
    },
}

BASE_API_LIST = [
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Exp",
    "Log",
    "Ln",
    "Muls",
    "Adds",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "Broadcast",
    "Duplicate",
    "Rsqrt",
    "Sqrt",
    "Cast",
    "Select",
    "Relu",
    "Gelu",
    "SoftMax",
    "LayerNorm",
    # #29 修复：追加缺失的基础 API 名
    "Ceil",
    "Floor",
    "Abs",
    "Neg",
    "Max",
    "Min",
    "RealDiv",
    "Not",
    "Reciprocal",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Pow",
    "Clamp",
    "Scatter",
    "Gather",
    "Pad",
    "Resize",
    "Transpose",
    "Concat",
    "Split",
    "Slice",
    "Tile",
    "NotEqual",
    "LessEqual",
    "GreaterEqual",
    "LogicalAnd",
    "LogicalOr",
    "DataCopy",
    "WaitEvent",
    "SetValue",
]

DOC_TO_SUB_PART = {
    "产品支持情况": "support",
    "功能说明": "summary",
    "实现原理": "impl_principle",
    "函数原型": "function_proto",
    "参数说明": "para_clarify",
    "返回值说明": "return_clarify",
    "约束说明": "constraint",
    "调用示例": "example",
}

CPP_EXTENSIONS = {".cpp", ".c", ".cc", ".h", ".hpp"}

# 组A 路径模板常量
IMPL_BASIC_DAV_GLOB = "impl/basic_api/dav_*"
IMPL_CHECK_GLOB = "impl/basic_api/utils/kernel_check_*.h"
IMPL_C_API_NPU_ARCH_GLOB = "impl/c_api/instr_impl/npu_arch_*"
ADV_API_CHECK_FILE = "impl/adv_api/detail/api_check/kernel_api_check.h"
ADV_API_CHECK_DIR = "impl/adv_api/detail/api_check/kernel_check/basic_check/"

# 组B 输出路径/文件名常量
OUTPUT_REGROUP_SUFFIX = ""
OUTPUT_REFACTORED_DIR = "refactor"
OUTPUT_DOC_JSON_DIR = "doc_json"
OUTPUT_ANCHOR_DIR = "anchor"
OUTPUT_EXAMPLES_DIR = "examples"
OUTPUT_EXTRA_DIR = "extra"
OUTPUT_CODE_MAP_DIR = "code_map"
OUTPUT_CODE_MAP_FILES_DIR = "files"
OUTPUT_FUNCTIONS_JSON = "functions.json"
OUTPUT_MAP_JSON_SUFFIX = "_map.json"
OUTPUT_DOC_JSON_SUFFIX = "_doc.json"
DOC_DIR = os.environ.get("ASC_DOC_DIR", "docs/zh/api")
# 兼容旧扁平结构；新结构是 docs/zh/api/<族>/<分类>/<接口>_interface/<接口>.md，
# 找不到时用 resolve_doc_path() 递归搜索。
DOC_CONTEXT_TMPL = DOC_DIR + "/{name}.md"


def resolve_doc_path(repo: str, name: str) -> str | None:
    """按 API 名解析文档路径（仓库相对路径）。

    先试扁平结构 ``<DOC_DIR>/<name>.md``，再在 ``<DOC_DIR>`` 下递归查找
    ``<name>.md``（现网结构为 docs/zh/api/<族>/<分类>/<接口>_interface/<接口>.md）。
    """
    import glob as _glob

    flat = DOC_CONTEXT_TMPL.format(name=name)
    if os.path.exists(os.path.join(repo, flat)):
        return flat
    pattern = os.path.join(repo, DOC_DIR, "**", f"{name}.md")
    hits = sorted(_glob.glob(pattern, recursive=True))
    if not hits:
        return None
    return os.path.relpath(hits[0], repo).replace(os.sep, "/")

# 组D code_map+clang+ext 常量
CODE_MAP_NODE_TYPES = frozenset({"function", "class", "struct", "namespace", "constant", "errorcode"})
CODE_MAP_EDGE_TYPES = frozenset({"calls", "returns"})
CLANG_STD = "-std=c++17"
CLANG_LANG_FLAGS = ["-x", "c++"]
CLANG_ACLNN_DEFINE = "-DACLNN_API="
EXAMPLE_EXTENSIONS = (".asc", ".cpp", ".h", ".py")
