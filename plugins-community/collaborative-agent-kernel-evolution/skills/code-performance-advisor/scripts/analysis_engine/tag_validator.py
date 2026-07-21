#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tag Validator - Ensures all tags comply with tag_taxonony.md

This validator can be used:
1. As a standalone CLI tool: python3 tag_validator.py [rule_pattern]
2. As a library: from tag_validator import validate_tag_file
3. As a hook: Called automatically after tag generation

Features:
- Validates tags against canonical taxonomy
- Suggests corrections for typos (fuzzy matching)
- Supports single-file validation (for hooks)
- Exit code 1 on validation failure (CI-friendly)
"""

import os
import json
import logging
import re
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional
from difflib import get_close_matches

logger = logging.getLogger(__name__)


def extract_valid_tags(taxonomy_path: Path) -> Set[str]:
    """Extract all valid tags from tag_taxonony.md.

    Returns:
        Set of valid tag strings (e.g., {'U.Cube', 'S.MemoryBound', ...})
    """
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract tags from backticks: `U.Cube`, `S.MemoryBound`, etc.
    backtick_tags = set(re.findall(r'`([UOTSC]\.[A-Z][a-zA-Z0-9\.]*)`', content))

    # Also catch tags in list items: * `U.Cube`: ... or * U.Cube: ...
    list_tags = set(re.findall(r'^\s*\*\s+`?([UOTSC]\.[A-Z][a-zA-Z0-9\.]*)`?:', content, re.MULTILINE))

    return backtick_tags.union(list_tags)


def is_likely_new_tag(invalid_tag: str, valid_tags: Set[str]) -> Tuple[bool, str]:
    """Determine if an invalid tag is likely a legitimate new tag vs a typo.

    Args:
        invalid_tag: The tag to analyze
        valid_tags: Set of all valid tags

    Returns:
        Tuple of (is_likely_new, reason)
        - is_likely_new: True if this looks like a new tag, False if likely a typo
        - reason: Explanation of the determination
    """
    # Check 1: Valid prefix
    prefix_match = re.match(r'^([UOTSC])\.', invalid_tag)
    if not prefix_match:
        return False, "Invalid prefix (must be U./O./T./S./C.)"

    prefix = prefix_match.group(1)
    suffix = invalid_tag[len(prefix) + 1:]  # e.g., "HighMemUtil" from "S.HighMemUtil"

    # Check 2: Valid naming convention (CamelCase, starts with uppercase)
    if not re.match(r'^[A-Z][a-zA-Z0-9\.]*$', suffix):
        return False, "Invalid naming convention (must be CamelCase)"

    # Check 3: Get same-prefix tags for analysis
    same_prefix_tags = [t for t in valid_tags if t.startswith(prefix + '.')]

    # Check 4: Very close match exists (likely typo)
    close_matches = get_close_matches(invalid_tag, same_prefix_tags, n=1, cutoff=0.85)
    if close_matches:
        return False, f"Very similar to existing tag: {close_matches[0]} (likely typo)"

    # Check 5: Semantic patterns - check if opposite/complementary tag exists
    # Common patterns: High/Low, Small/Large, Enable/Disable, etc.
    semantic_opposites = {
        'High': 'Low', 'Low': 'High',
        'Small': 'Large', 'Large': 'Small',
        'Enable': 'Disable', 'Disable': 'Enable',
        'Start': 'End', 'End': 'Start',
        'First': 'Last', 'Last': 'First',
    }

    for pattern, opposite in semantic_opposites.items():
        if pattern in suffix:
            # Check if opposite exists
            opposite_tag = prefix + '.' + suffix.replace(pattern, opposite)
            if opposite_tag in valid_tags:
                return True, f"Semantic complement of existing tag: {opposite_tag}"

    # Check 6: Similar structure exists (e.g., S.LowCubeUtil exists, S.HighCubeUtil is reasonable)
    # Extract base pattern (remove High/Low/Small/Large prefixes)
    base_pattern = re.sub(r'^(High|Low|Small|Large|Enable|Disable)', '', suffix)
    if base_pattern != suffix:  # Had a prefix
        for tag in same_prefix_tags:
            tag_suffix = tag[len(prefix) + 1:]
            tag_base = re.sub(r'^(High|Low|Small|Large|Enable|Disable)', '', tag_suffix)
            if tag_base == base_pattern:
                return True, f"Similar pattern to existing tag: {tag}"

    # Check 7: Low similarity to all existing tags (genuinely new concept)
    best_matches = get_close_matches(invalid_tag, same_prefix_tags, n=1, cutoff=0.6)
    if not best_matches:
        return True, "No similar existing tags (new concept)"

    # Default: moderate similarity suggests typo
    return False, f"Moderate similarity to existing tags (likely typo, check: {', '.join(get_close_matches(invalid_tag, same_prefix_tags, n=3, cutoff=0.6))})"


def suggest_corrections(invalid_tag: str, valid_tags: Set[str], threshold: float = 0.8) -> List[str]:
    """Suggest valid tags similar to an invalid tag using fuzzy matching.

    Args:
        invalid_tag: The invalid tag to find corrections for
        valid_tags: Set of all valid tags
        threshold: Similarity threshold (0.0-1.0), default 0.8

    Returns:
        List of suggested valid tags, sorted by similarity
    """
    # Extract prefix (e.g., 'S.' from 'S.InvalidTag')
    prefix_match = re.match(r'^([UOTSC]\.)', invalid_tag)
    if prefix_match:
        prefix = prefix_match.group(1)
        # Only suggest tags with same prefix
        same_prefix_tags = [t for t in valid_tags if t.startswith(prefix)]
        suggestions = get_close_matches(invalid_tag, same_prefix_tags, n=3, cutoff=threshold)
        return suggestions
    else:
        # No valid prefix, suggest any close match
        return get_close_matches(invalid_tag, valid_tags, n=3, cutoff=threshold)


def validate_tag_file(
    tag_file: Path,
    valid_tags: Set[str],
    verbose: bool = True
) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    """Validate a single tag JSON file.

    Args:
        tag_file: Path to *_tags.json file
        valid_tags: Set of valid tags from taxonomy
        verbose: Whether to print validation results

    Returns:
        Tuple of (is_valid, invalid_tags, suggestions_dict)
        - is_valid: True if all tags are valid
        - invalid_tags: List of invalid tag strings
        - suggestions_dict: Map from invalid tag to suggested corrections
    """
    try:
        with open(tag_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract all tags from possible fields
        tag_fields = ["tags", "domain_tags", "symptom_tags", "context_tags", "required_tags"]
        file_tags = set()
        for field in tag_fields:
            if field in data and isinstance(data[field], list):
                file_tags.update(data[field])

        invalid_tags = [t for t in file_tags if t not in valid_tags]

        if invalid_tags:
            suggestions_dict = {tag: suggest_corrections(tag, valid_tags) for tag in invalid_tags}

            if verbose:
                logger.info(f"[INVALID] {tag_file.name}")
                for tag in invalid_tags:
                    # Determine if this is likely a new tag or a typo
                    is_new, reason = is_likely_new_tag(tag, valid_tags)

                    logger.info(f"  ❌ Unknown tag: {tag}")

                    if is_new:
                        logger.info(f"     🆕 This looks like a NEW TAG: {reason}")
                        logger.info(f"     📝 If this is intentional, add to tag_taxonony.md:")
                        logger.info(f"        See: references/standards/TAG_ADDITION_GUIDE.md")
                    else:
                        logger.info(f"     🔍 Likely a TYPO: {reason}")
                        suggestions = suggestions_dict[tag]
                        if suggestions:
                            logger.info(f"     💡 Did you mean: {', '.join(suggestions)}?")
                        else:
                            logger.info(f"     💡 No similar tags found")

            return False, invalid_tags, suggestions_dict
        else:
            if verbose:
                logger.info(f"[OK] {tag_file.name} - All {len(file_tags)} tags valid ✅")
            return True, [], {}

    except Exception as e:
        if verbose:
            logger.info(f"[ERROR] Failed to process {tag_file}: {e}")
        return False, [], {}


def validate_tags(
    rules_dir: Path,
    valid_tags: Set[str],
    target_rule: Optional[str] = None
) -> bool:
    """Validate all rule tag files in a directory (legacy interface).

    Args:
        rules_dir: Root directory containing rule files
        valid_tags: Set of valid tags from taxonomy
        target_rule: Optional pattern to filter specific rule

    Returns:
        True if all validations pass, False otherwise
    """
    json_files = list(rules_dir.rglob("*_tags.json"))

    if target_rule:
        json_files = [f for f in json_files if target_rule in f.name]
        if not json_files:
            logger.info(f"No tag files found for rule pattern: {target_rule}")
            return False

    logger.info(f"Checking {len(json_files)} rule(s)...")

    all_valid = True
    for json_file in json_files:
        is_valid, _, _ = validate_tag_file(json_file, valid_tags, verbose=True)
        if not is_valid:
            all_valid = False

    if all_valid:
        logger.info(f"\n✅ Success: All {len(json_files)} rule(s) have valid tags")
    else:
        logger.info(f"\n❌ Validation failed: Some tags are not in tag_taxonony.md")

    return all_valid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # Auto-detect project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # Go up to skill root

    TAXONOMY_PATH = project_root / "references/standards/tag_taxonony.md"
    RULES_DIR = project_root / "assets/rules"

    # Parse arguments
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if not TAXONOMY_PATH.exists():
        logger.info(f"❌ Taxonomy file not found at {TAXONOMY_PATH}")
        sys.exit(1)

    # Load valid tags
    valid_tags = extract_valid_tags(TAXONOMY_PATH)
    logger.info(f"📚 Loaded {len(valid_tags)} valid tags from taxonomy\n")

    # Validate
    success = validate_tags(RULES_DIR, valid_tags, target)

    sys.exit(0 if success else 1)
