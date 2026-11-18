# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.logger import debug, error, info, warning


class FileEvidence:
    def __init__(self, file_name: str, line_range: Optional[Tuple[int, int]] = None):
        self.file_name = file_name
        self.line_range = line_range

    def __hash__(self):
        return hash((self.file_name, self.line_range))

    def __eq__(self, other):
        if not isinstance(other, FileEvidence):
            return False
        return self.file_name == other.file_name and self.line_range == other.line_range


class DocpartFileExtractor:
    def __init__(self, evidence_dir: str, fine_grained: bool = True):
        self.evidence_dir = Path(evidence_dir)
        self.fine_grained = fine_grained

    @staticmethod
    def _get_filename_from_path(file_path: str) -> str:
        # file_path 在 api_locator 输出中可能是 list[str]，取第一个元素
        if isinstance(file_path, list):
            file_path = file_path[0] if file_path else ""
        return os.path.basename(file_path).replace("\\", "/").split("/")[-1]

    @staticmethod
    def _get_evidence_type(anchor: Dict) -> str:
        return anchor.get("anchor_information", {}).get("type", "unknown")

    def extract_mappings(self) -> Dict[str, List[str]]:
        mapping = {}

        if not self.evidence_dir.exists():
            return mapping

        for file_path in self.evidence_dir.glob("*.json"):
            if file_path.name.startswith("."):
                continue
            part_mapping = self._parse_evidence_file(file_path)
            for doc_part, evidence_set in part_mapping.items():
                if doc_part not in mapping:
                    mapping[doc_part] = []
                for ev in sorted(evidence_set, key=lambda x: x.file_name):
                    mapping[doc_part].append(ev.file_name)

        return mapping

    def extract_fine_grained_mappings(self) -> Dict[str, List[Dict]]:
        mapping = {}

        if not self.evidence_dir.exists():
            return mapping

        for file_path in self.evidence_dir.glob("*.json"):
            if file_path.name.startswith("."):
                continue
            part_mapping = self._parse_evidence_file(file_path)
            for doc_part, evidence_set in part_mapping.items():
                if doc_part not in mapping:
                    mapping[doc_part] = []
                for ev in sorted(evidence_set, key=lambda x: (x.file_name, x.line_range or (0, 0))):
                    entry = {"file_name": ev.file_name}
                    if ev.line_range is not None:
                        entry["line_range"] = ev.line_range
                    else:
                        entry["line_range"] = None
                    mapping[doc_part].append(entry)

        return mapping

    def get_subpart_file_table(self) -> Dict[str, List[str]]:
        return self.extract_mappings()

    def _extract_evidence_from_anchor(self, anchor: Dict) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
        note_type = self._get_evidence_type(anchor)
        evidence_list = anchor.get("evidence", [])
        results = []

        for evidence_item in evidence_list:
            if "evidence_information" not in evidence_item:
                continue
            info = evidence_item["evidence_information"]
            if "file_path" not in info:
                continue

            raw_path = info["file_path"]
            file_paths = raw_path if isinstance(raw_path, list) else [raw_path]

            for fp in file_paths:
                if not fp:
                    continue
                file_name = self._get_filename_from_path(fp)

                if note_type in ("file", "function"):
                    results.append((file_name, None))
                elif note_type in ("note", "section"):
                    start_line = info.get("start_line")
                    end_line = info.get("end_line")
                    if start_line is not None and end_line is not None:
                        try:
                            results.append((file_name, (int(start_line), int(end_line))))
                        except (ValueError, TypeError):
                            pass

        if not results and evidence_list:
            info = evidence_list[0].get("evidence_information", {})
            if "file_path" in info:
                raw_path = info["file_path"]
                file_paths = raw_path if isinstance(raw_path, list) else [raw_path]
                for fp in file_paths:
                    if fp:
                        file_name = self._get_filename_from_path(fp)
                        results.append((file_name, None))

        return results

    def _parse_evidence_file(self, file_path: Path) -> Dict[str, Set[FileEvidence]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

        # 兼容两种 anchor JSON 格式：
        # snapshot 格式: {"doc": {"doc_part": "support"}, "anchor_tuple": [...]}
        # api_locator 格式: {"sub_part": "support", "doc_part": "产品支持情况", "anchor_tuple": [...]}
        doc_info = data.get("doc", {})
        if doc_info and isinstance(doc_info, dict):
            doc_part = doc_info.get("doc_part", "unknown")
        elif "sub_part" in data:
            doc_part = data.get("sub_part", "unknown")
        else:
            doc_part = "unknown"

        anchor_tuples = data.get("anchor_tuple", [])
        evidence_set = set()

        for anchor in anchor_tuples:
            for file_name, line_range in self._extract_evidence_from_anchor(anchor):
                if file_name:
                    evidence_set.add(FileEvidence(file_name, line_range))

        return {doc_part: evidence_set}


def extract_evidence_mappings(evidence_dir: str, fine_grained: bool = False) -> Dict:
    extractor = DocpartFileExtractor(evidence_dir, fine_grained)
    if fine_grained:
        return extractor.extract_fine_grained_mappings()
    return extractor.extract_mappings()


if __name__ == "__main__":
    from src.generation_validation import config

    extractor = DocpartFileExtractor(
        evidence_dir="./scripts/cann-document-generation/data/outputs/snapshot/abs/base/anchor",
        fine_grained=True,
    )

    if extractor.fine_grained:
        mappings = extractor.extract_fine_grained_mappings()
        print("Subpart 与文件对应关系表 (细粒度):")
        print("-" * 50)
        for subpart, files in sorted(mappings.items()):
            print(f"\n[{subpart}]")
            for file_info in files:
                if file_info["line_range"]:
                    print(
                        f"  - {file_info['file_name']} (行 {file_info['line_range'][0]}-{file_info['line_range'][1]})"
                    )
                else:
                    print(f"  - {file_info['file_name']} (全文)")
        todo_list = [subpart for subpart, files in mappings.items() if files]
        print(todo_list)
    else:
        mappings = extractor.extract_mappings()
        print("Subpart 与文件对应关系表:")
        print("-" * 50)
        for subpart, files in sorted(mappings.items()):
            print(f"\n[{subpart}]")
            for file in files:
                print(f"  - {file}")
    print("-" * 50)
