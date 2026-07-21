#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# bootstrap.sh - Initialize code-performance-advisor rule index
# This script must be run from the skill root directory

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] Code Performance Advisor - Bootstrap"
echo "[INFO] Skill root: $SCRIPT_DIR"

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if rules directory exists
if [ ! -d "assets/rules/special_rules" ]; then
    echo "[ERROR] Rules directory not found: assets/rules/special_rules"
    echo "[ERROR] Please ensure you are running this script from the skill root"
    exit 1
fi

# Backup existing index if present
if [ -f "assets/manifests/index.json" ]; then
    backup_file="assets/manifests/index.json.backup.$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] Backing up existing index to: $backup_file"
    cp "assets/manifests/index.json" "$backup_file"
fi

# Clear index
rm -f assets/manifests/index.json
echo "[INFO] Building rule index..."

# Count rules
rule_count=0
error_count=0

# Index all special rules
for rule_file in assets/rules/special_rules/R_*/R_*.md; do
    if [ -f "$rule_file" ]; then
        rule_name=$(basename "$(dirname "$rule_file")")

        if python3 scripts/analysis_engine/cli.py update-index --rule "$rule_file" 2>&1 >/dev/null; then
            rule_count=$((rule_count + 1))
            echo -n "."  # Progress indicator
        else
            echo "[WARN] Failed to index: $rule_file"
            error_count=$((error_count + 1))
        fi
    fi
done
echo ""  # New line after progress dots

echo ""
echo "========================================="
echo "Bootstrap Summary"
echo "========================================="
echo "Rules indexed: $rule_count"
echo "Errors: $error_count"

# Verify index
if [ -f "assets/manifests/index.json" ]; then
    indexed_count=$(python3 -c "import json; data=json.load(open('assets/manifests/index.json')); print(len(data['rules']))" 2>/dev/null || echo "0")
    echo "Index file size: $(wc -c < assets/manifests/index.json) bytes"
    echo "Rules in index: $indexed_count"

    if [ "$indexed_count" -gt 0 ]; then
        echo ""
        echo "✅ Bootstrap completed successfully!"
        echo ""
        echo "Next steps:"
        echo "  1. Initialize workspace: python3 scripts/analysis_engine/init_workspace.py --op <operator>"
        echo "  2. Generate tags: Use code_tag subskill"
        echo "  3. Score rules: python3 scripts/analysis_engine/cli.py score"
        exit 0
    else
        echo ""
        echo "❌ Index file created but contains no rules"
        exit 1
    fi
else
    echo ""
    echo "❌ Bootstrap failed: index.json not created"
    exit 1
fi
