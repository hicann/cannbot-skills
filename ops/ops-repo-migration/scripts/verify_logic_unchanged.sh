#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# verify_logic_unchanged.sh <src_dir> <op_dir>
# Compare same-path C/C++/Ascend sources after preprocessing comments away.
# Preserve whitespace inside literals and between tokens. Formatting-only changes
# may be reported conservatively. Differences are a review inventory, not a gate.
# Exit: 0 = inventory complete (including differences), 1 = tool/read failure,
#       2 = invalid arguments. New/deleted/moved files need separate review.
set -u
set -o pipefail

SRC="${1:-}"
DST="${2:-}"
if [ -z "$SRC" ] || [ -z "$DST" ] || [ ! -d "$SRC" ] || [ ! -d "$DST" ]; then
    echo "usage: $0 <src_dir> <op_dir>"
    exit 2
fi
SRC="$(cd "$SRC" && pwd)"
DST="$(cd "$DST" && pwd)"
for tool in gcc sha256sum; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[ERROR] required tool not found: $tool" >&2
        exit 1
    fi
done
TMP_DIR=$(mktemp -d) || exit 1
cleanup() {
    cleanup_status=$?
    trap - EXIT
    # Only unlink files created by this script. Unexpected entries must survive.
    for temporary_file in source error src-list dst-list; do
        path="$TMP_DIR/$temporary_file"
        if [ -e "$path" ] || [ -L "$path" ]; then
            rm -- "$path" || cleanup_status=1
        fi
    done
    rmdir -- "$TMP_DIR" || cleanup_status=1
    exit "$cleanup_status"
}
trap cleanup EXIT

strip_and_hash() {
    # Do not erase whitespace: "a b" != "ab", and a + +b != a ++b.
    # -x is explicit so .h/.hpp/.asc are all treated as source, not linker input.
    if ! gcc -x c++ -fpreprocessed -dD -E -P "$1" >"$TMP_DIR/source" 2>"$TMP_DIR/error"; then
        echo "[ERROR] preprocess failed: $1" >&2
        cat "$TMP_DIR/error" >&2
        return 1
    fi
    sha256sum < "$TMP_DIR/source" | cut -d' ' -f1
}

source_files() {
    find "$1" -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
        -o -name '*.h' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.asc' \) \
        -not -path '*/build/*' -not -path '*/build_out/*' -print0 | sort -z
}

same=(); diffs=(); only_src=(); only_dst=(); errors=()
# Materialize discovery first so errors in find/sort cannot disappear in a
# process substitution and leave an apparently successful empty inventory.
source_files "$SRC" > "$TMP_DIR/src-list" || exit 1
source_files "$DST" > "$TMP_DIR/dst-list" || exit 1
while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    if [ ! -f "$DST/$rel" ]; then
        only_src+=("$rel")
        continue
    fi
    if ! src_hash=$(strip_and_hash "$f"); then
        errors+=("$rel")
    elif ! dst_hash=$(strip_and_hash "$DST/$rel"); then
        errors+=("$rel")
    elif [ "$src_hash" = "$dst_hash" ]; then
        same+=("$rel")
    else
        diffs+=("$rel")
    fi
done < "$TMP_DIR/src-list"
while IFS= read -r -d '' f; do
    rel="${f#"$DST"/}"
    [ -f "$SRC/$rel" ] || only_dst+=("$rel")
done < "$TMP_DIR/dst-list"

echo "== source comparison inventory =="
echo "src: $SRC"
echo "dst: $DST"
echo "[IDENTICAL] ${#same[@]} file(s) — same-path preprocessed content matches"
for f in ${same[@]+"${same[@]}"}; do echo "    $f"; done
echo "[DIFFERS] ${#diffs[@]} file(s) — review each source/formatting difference"
for f in ${diffs[@]+"${diffs[@]}"}; do echo "    $f"; done
echo "[ONLY-IN-SRC] ${#only_src[@]} file(s) — review deletion or map to moved file"
for f in ${only_src[@]+"${only_src[@]}"}; do echo "    $f"; done
echo "[ONLY-IN-DST] ${#only_dst[@]} file(s) — review addition or map to moved file"
for f in ${only_dst[@]+"${only_dst[@]}"}; do echo "    $f"; done
echo "[ERROR] ${#errors[@]} file(s) could not be compared"
for f in ${errors[@]+"${errors[@]}"}; do echo "    $f"; done
echo "IDENTICAL=${#same[@]} DIFFERS=${#diffs[@]} ONLY-IN-SRC=${#only_src[@]} ONLY-IN-DST=${#only_dst[@]} ERRORS=${#errors[@]}"
echo "This inventory is not proof of unchanged logic; review differences and path mappings."
[ "${#errors[@]}" -eq 0 ]
