# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

__all__ = [
    "FileChangeDetector",
    "detect_file_changes",
    "DocpartFileExtractor",
    "extract_evidence_mappings",
    "ChangedFileMapper",
    "map_changed_files_to_subparts",
    "run_detection",
]

from .docpart_file_extractor import DocpartFileExtractor, extract_evidence_mappings
from .file_change_detector import FileChangeDetector, detect_file_changes
from .subpart_mapper import ChangedFileMapper, map_changed_files_to_subparts, run_detection
