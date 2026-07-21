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

"""Post-tag-generation hook - Auto-validates tags after generation.

This hook is automatically called after code_tag subskill generates tags.
It ensures all tags comply with tag_taxonony.md before proceeding.

Exit codes:
    0: Validation passed
    1: Validation failed (invalid tags found)
    2: Hook execution error
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts" / "analysis_engine"))

from tag_validator import extract_valid_tags, validate_tag_file, is_likely_new_tag


def main():
    """Main hook entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # Detect skill root
    skill_root = SCRIPT_DIR.parent

    # Paths
    taxonomy_path = skill_root / "references/standards/tag_taxonony.md"
    tag_dir = skill_root / "workspace" / "cache" / "tags"

    # Check if taxonomy exists
    if not taxonomy_path.exists():
        logger.info(f"[HOOK ERROR] Taxonomy not found: {taxonomy_path}")
        return 2

    # Load valid tags
    try:
        valid_tags = extract_valid_tags(taxonomy_path)
        logger.info(f"[HOOK] Loaded {len(valid_tags)} valid tags from taxonomy")
    except Exception as e:
        logger.info(f"[HOOK ERROR] Failed to load taxonomy: {e}")
        return 2

    # Find latest generated tag file
    if not tag_dir.exists():
        logger.info(f"[HOOK] No tag directory found at {tag_dir}, skipping validation")
        return 0

    tag_files = sorted(tag_dir.glob("tag_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tag_files:
        logger.info(f"[HOOK] No tag files found in {tag_dir}, skipping validation")
        return 0

    latest_tag_file = tag_files[0]
    logger.info(f"[HOOK] Validating latest tag file: {latest_tag_file.name}")

    # Validate
    try:
        is_valid, invalid_tags, suggestions = validate_tag_file(
            latest_tag_file,
            valid_tags,
            verbose=False  # Custom output below
        )

        if is_valid:
            logger.info(f"[HOOK] ✅ Validation passed - all tags are valid")
            return 0
        else:
            logger.info(f"[HOOK] ❌ Validation failed - {len(invalid_tags)} invalid tag(s) found:")

            # Analyze each invalid tag
            has_likely_new_tags = False
            for tag in invalid_tags:
                is_new, reason = is_likely_new_tag(tag, valid_tags)

                if is_new:
                    logger.info(f"\n  🆕 {tag}")
                    logger.info(f"     {reason}")
                    logger.info(f"     📝 If this is intentional, add to tag_taxonony.md")
                    logger.info(f"        Guide: references/standards/TAG_ADDITION_GUIDE.md")
                    has_likely_new_tags = True
                else:
                    logger.info(f"\n  ❌ {tag}")
                    logger.info(f"     🔍 {reason}")
                    if suggestions[tag]:
                        logger.info(f"     💡 Did you mean: {', '.join(suggestions[tag])}?")

            logger.info(f"\n[HOOK] Fix invalid tags in: {latest_tag_file}")
            logger.info(f"[HOOK] Taxonomy reference: {taxonomy_path}")

            if has_likely_new_tags:
                logger.info(f"\n[HOOK] 💡 Detected potential new tags.")
                logger.info(f"[HOOK]    If adding new tags, follow: references/standards/TAG_ADDITION_GUIDE.md")

            return 1

    except Exception as e:
        logger.info(f"[HOOK ERROR] Validation error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
