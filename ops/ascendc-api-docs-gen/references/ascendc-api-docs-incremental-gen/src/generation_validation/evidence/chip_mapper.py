# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Chip mapper - maps chip/arch identifiers to product names"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple

import yaml

from src.logger import info, subsection

from .. import config


class MappingStats:
    """统计映射结果"""

    def __init__(self):
        self.hit_fixed: Set[str] = set()
        self.hit_but_not_in_fixed: Dict[str, List[str]] = {}
        self.no_rule: Set[str] = set()


class ChipMapper:
    """Encapsulates chip_map.yaml loading and matching logic"""

    DEFAULT_YAML_PATH = str(config.CHIP_MAP_YAML)

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = yaml_path or self.DEFAULT_YAML_PATH
        self.templates_yaml_path = str(config.TEMPLATES_YAML)
        self.exact_map: Dict[str, List[str]] = {}
        self.prefix_list: List[Tuple[str, List[str]]] = []
        self.arch_map: Dict[str, List[str]] = {}
        self.product_npu_map: List[dict] = []
        self.fixed_products: List[str] = []
        self._load_templates()
        self._load()

    @staticmethod
    def print_mapping_warnings(stats: MappingStats):
        """打印映射警告"""
        if stats.hit_but_not_in_fixed:
            subsection("类型1: 有映射规则但产品不在标准产品表内（将被过滤）")
            for identifier in sorted(stats.hit_but_not_in_fixed.keys()):
                info(f"\t[FILTER] {identifier:<40} -> 产品不在固定表")

        if stats.no_rule:
            subsection("类型2: chip_map.yaml 中没有映射规则")
            for identifier in sorted(stats.no_rule):
                info(f"\t[NO_RULE] {identifier:<40} -> 无此标识符的规则")

    @staticmethod
    def print_mapping_summary(stats: MappingStats):
        """打印详细的映射成功/失败统计"""
        subsection("映射结果汇总")

        if stats.hit_fixed:
            info(f"成功映射 ({len(stats.hit_fixed)} 个):")
            for ident in sorted(stats.hit_fixed):
                info(f"  - {ident}")
        else:
            info("成功映射: 无")

        if stats.hit_but_not_in_fixed:
            info(f"有映射规则但产品不在标准表 ({len(stats.hit_but_not_in_fixed)} 个):")
            for ident in sorted(stats.hit_but_not_in_fixed.keys()):
                info(f"  - {ident}")

        if stats.no_rule:
            info(f"无映射规则 ({len(stats.no_rule)} 个):")
            for ident in sorted(stats.no_rule):
                info(f"  - {ident}")
        else:
            info("无映射规则: 无")

    @staticmethod
    def _normalize_key(s: str) -> str:
        """Normalize key: lowercase, remove spaces and underscores"""
        return re.sub(r"[\s_\-]", "", s).lower()

    def match_chip(self, chip: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Match chip identifier with exact-match priority, then prefix-match fallback.

        Returns:
            (matched_products, status) where status is 'hit', 'prefix_hit', or None
        """
        key = self._normalize_key(chip)

        if key in self.exact_map:
            return (self.exact_map[key], "hit")

        for prefix, products in self.prefix_list:
            if key.startswith(prefix):
                return (products, "prefix_hit")

        return (None, None)

    def match_arch(self, arch: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Match arch identifier with exact match.

        Returns:
            (matched_products, status) where status is 'hit' or None
        """
        arch_str = str(arch).strip()
        if arch_str in self.arch_map:
            return (self.arch_map[arch_str], "hit")
        return (None, None)

    def collect_supported_products(
        self,
        chips: List[str],
        archs: List[str],
    ) -> Tuple[Set[str], MappingStats]:
        """对每个 chip/arch 做映射，返回支持的产品集合和统计信息"""
        supported: Set[str] = set()
        stats = MappingStats()

        for chip in chips:
            matched, status = self.match_chip(chip)

            if status is None:
                stats.no_rule.add(chip)
            elif status in ("hit", "prefix_hit"):
                if matched:
                    supported.update(matched)
                    stats.hit_fixed.add(chip)
                else:
                    stats.hit_but_not_in_fixed[chip] = []

        for arch in archs:
            matched, status = self.match_arch(arch)

            if status is None:
                stats.no_rule.add(arch)
            elif status == "hit":
                if matched:
                    supported.update(matched)
                    stats.hit_fixed.add(arch)
                else:
                    stats.hit_but_not_in_fixed[arch] = []

        return supported, stats

    def generate_markdown(self, supported: Set[str], use_term_tags: bool = True) -> str:
        """生成产品支持矩阵 markdown。

        Args:
            supported: 支持的产品名集合
            use_term_tags: 是否用 <term> 标签包裹产品名；应跟随原文格式
                （原文无 <term> 时输出纯文本，避免格式漂移）
        """
        lines = []
        for entry in self.product_npu_map:
            npu = entry["npu"]
            pid = entry["id"]
            product = entry["product"]
            mark = "支持" if product in supported else "不支持"
            lines.append(f'<!-- npu="{npu}" {pid} -->')
            if use_term_tags:
                lines.append(f"- <term>{product}</term>：{mark}")
            else:
                lines.append(f"- {product}：{mark}")
            lines.append(f"<!-- end {pid} -->")
        return "\n".join(lines)

    def parse_support_md(self, md_text: str) -> Dict[str, bool]:
        """从原文 markdown 解析'产品支持情况'章节，提取产品与是否支持的关系。
        支持格式：
        - 表格形式：| <term>产品名</term> | √ |  或  | <term>产品名</term> | × |
        - 列表形式：- <term>产品名</term>：支持  或  - <term>产品名</term>：不支持
        - 纯文本：无 <term> 标签的产品名
        返回 {product_name: bool}，仅包含在文本中明确出现的产品。
        """
        result: Dict[str, bool] = {}
        if not md_text:
            return result

        sorted_products = sorted(self.fixed_products, key=len, reverse=True)
        for product in sorted_products:
            escaped = re.escape(product)
            pattern = re.compile(
                r"<term>\s*" + escaped + r"\s*</term>|" + escaped,
                re.DOTALL,
            )
            match = pattern.search(md_text)
            if not match:
                continue

            line_start = md_text.rfind("\n", 0, match.start()) + 1
            line_end = md_text.find("\n", match.end())
            if line_end == -1:
                line_end = len(md_text)
            line = md_text[line_start:line_end]

            if self._line_is_not_supported(line):
                result[product] = False
            elif self._line_is_supported(line):
                result[product] = True

        return result

    @staticmethod
    def _line_is_not_supported(line: str) -> bool:
        if "不支持" in line:
            return True
        if "×" in line:
            return True
        return False

    @staticmethod
    def _line_is_supported(line: str) -> bool:
        if "不支持" in line:
            return False
        if re.search(r"[：:]\s*支持\b", line):
            return True
        if "√" in line:
            return True
        return False

    def _load_templates(self):
        if not os.path.isfile(self.templates_yaml_path):
            raise FileNotFoundError(f"templates.yaml not found: {self.templates_yaml_path}")

        with open(self.templates_yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        support = raw.get("doc_parts", {}).get("support", {})
        self.product_npu_map = support.get("products", [])
        self.fixed_products = [entry["product"] for entry in self.product_npu_map]

    def _load(self):
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"chip_map.yaml not found: {self.yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for entry in raw.get("chip_mapping", []):
            chip_raw = str(entry.get("chip", "")).strip()
            product_raw = str(entry.get("product", "")).strip()
            products = self._parse_product_field(product_raw)
            if not chip_raw:
                continue

            if chip_raw.endswith("*"):
                prefix = self._normalize_key(chip_raw[:-1])
                if products:
                    self.prefix_list.append((prefix, products))
            else:
                key = self._normalize_key(chip_raw)
                self.exact_map.setdefault(key, [])
                for p in products:
                    if p not in self.exact_map[key]:
                        self.exact_map[key].append(p)
                if not products:
                    self.exact_map[key] = []

        self.prefix_list.sort(key=lambda x: len(x[0]), reverse=True)

        for entry in raw.get("npu_arch_mapping", []):
            arch_raw = str(entry.get("arch", "")).strip()
            product_raw = str(entry.get("product", "")).strip()
            products = self._parse_product_field(product_raw)
            if not arch_raw:
                continue
            self.arch_map.setdefault(arch_raw, [])
            for p in products:
                if p not in self.arch_map[arch_raw]:
                    self.arch_map[arch_raw].append(p)
            if not products:
                self.arch_map[arch_raw] = []

    def _normalize_product(self, product: str) -> str:
        """Normalize product name to match fixed_products entries"""
        product = product.strip()
        for fixed in self.fixed_products:
            if product == fixed or fixed.startswith(product):
                return fixed
        return product

    def _parse_product_field(self, product_raw: str) -> List[str]:
        """Parse product field, keep '/' separated products as whole"""
        results: List[str] = []
        parts = re.split(r"[、，]", product_raw)
        for part in parts:
            part = part.strip()
            if part:
                normalized = self._normalize_product(part)
                if normalized and normalized not in results:
                    results.append(normalized)
        return results
