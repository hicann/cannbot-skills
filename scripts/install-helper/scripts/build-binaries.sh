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

set -e

# Get version from package.json
VERSION=$(node -e "console.log(require('./package.json').version)")
echo "Building install-helper v${VERSION} binaries..."

# Regenerate embedded plugins data from YAML
echo "Regenerating embedded-plugins.json..."
node scripts/gen-embedded.cjs

# Clean only build artifacts. NEVER `rm -rf bin/` — it would delete the
# git-tracked bin/install-helper.js entry wrapper, causing npm to publish
# a broken main package (bin/ silently missing from the tarball).
mkdir -p bin/
rm -f bin/install-helper-linux-x64 bin/install-helper-linux-arm64 \
      bin/install-helper-darwin-x64 bin/install-helper-darwin-arm64 \
      bin/install-helper-windows-x64 bin/install-helper-windows-x64.exe

# Self-heal: restore the wrapper if a previous run deleted it
if [ ! -f bin/install-helper.js ] || [ ! -f bin/package.json ]; then
  echo "  ! bin/install-helper.js or bin/package.json missing — restoring from git..."
  git checkout -- bin/ 2>/dev/null || {
    echo "  ✗ cannot restore bin/ wrapper files (not a git repo or files untracked)"
    exit 1
  }
  echo "  ✓ restored"
fi

# Target platforms
TARGETS=(
  "bun-linux-x64:linux-x64"
  "bun-linux-arm64:linux-arm64"
  "bun-darwin-x64:darwin-x64"
  "bun-darwin-arm64:darwin-arm64"
  "bun-windows-x64:windows-x64"
)

for entry in "${TARGETS[@]}"; do
  BUN_TARGET="${entry%%:*}"
  OUTPUT_NAME="${entry##*:}"
  
  echo ""
  echo "  Building ${OUTPUT_NAME}..."
  
  if [ "$OUTPUT_NAME" = "windows-x64" ]; then
    OUTPUT_FILE="bin/install-helper-${OUTPUT_NAME}.exe"
  else
    OUTPUT_FILE="bin/install-helper-${OUTPUT_NAME}"
  fi
  
  bun build src/index.ts --compile --target="$BUN_TARGET" --outfile="$OUTPUT_FILE" 2>&1
  
  if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "  ✓ ${OUTPUT_NAME} (${SIZE})"
  else
    echo "  ✗ ${OUTPUT_NAME} FAILED"
    exit 1
  fi
done

echo ""
echo "All binaries built successfully in bin/"
ls -lh bin/
