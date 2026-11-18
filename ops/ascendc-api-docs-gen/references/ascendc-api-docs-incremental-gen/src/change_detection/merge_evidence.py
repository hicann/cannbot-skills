# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Merge redirect anchor results into original anchor tuple."""

import json
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.generation_validation.config import EVIDENCE_STAGES

console = Console()


def _get_evidence_content(entry: dict) -> str | None:
    """Extract a dedup content from an anchor entry's evidence.

    Tries ``evidence[0].evidence_information.file_content`` first,
    then ``evidence[0].evidence_information.content``.
    Returns a stripped string, or ``None`` if neither is found.
    """
    if not entry.get("evidence"):
        return None
    ev = entry["evidence"][0].get("evidence_information", {})
    if "file_content" in ev:
        return str(ev["file_content"]).strip()
    if "content" in ev:
        c = ev["content"]
        # content 有时是列表（如 function 类型），取第一个元素的 function_hash 或整个列表
        if isinstance(c, list):
            first = c[0] if c else ""
            if isinstance(first, dict):
                return str(first.get("function_hash", "")).strip()
            return str(c).strip()
        return str(c).strip()
    return None


def _get_anchor_file_path_and_type(entry: dict) -> tuple:
    """Extract (file_path, anchor_type) from anchor_information.

    Returns:
        Tuple of (file_path, type). Either element may be None if absent.
    """
    info = entry.get("anchor_information") or {}
    level1 = info.get("level1") or {}
    fp = level1.get("file_path")
    at = info.get("type")
    fp_str = str(fp).strip() if fp else None
    at_str = str(at).strip() if at else None
    return fp_str, at_str


def merge_redirect_to_anchor(
    output_path: Path, doc_subpart_todo_list: list, anchor_dir: Path | None = None, verbose: bool = False
) -> None:
    """Merge redirect-stage anchor evidence into original anchor files.

    Workflow:
        1. Read each anchor JSON from anchor_dir, filtered by doc_subpart_todo_list.
        2. For each subpart, look up matching redirect results from
           output_path/redirect/{stage}_{subpart}.json for stages file/func/note/section.
        3. Skip redirect entries whose (file_path, anchor_type) key already exists
           in the Modified-side original_tuple, regardless of content. When
           (file_path, type) matches, the Modified-side entry takes precedence
           because it reflects the post-modification code state.
        4. For remaining redirect entries, also de-duplicate against
           original entries by comparing their content fingerprints
           (file_content or evidence_information.content).
        5. For each valid new redirect entry, prefix evidence_ids with "redirect_"
           and append to anchor_tuple.
        6. Write merged results to output_path/evidences/ with the same filenames.

    Redirect entries share the same JSON schema as original anchor entries —
    no format transformation is needed, only content-based deduplication and merging.

    Args:
        output_path: Path to redirect output directory.
        doc_subpart_todo_list: List of subparts to process (e.g. ["summary", "function_proto"]).
        anchor_dir: Path to anchor directory. Defaults to output_path / "anchor".
        verbose: If True, print detailed per-entry debug info to console.
    """
    evidence_output_dir = output_path / "evidences"
    evidence_output_dir.mkdir(parents=True, exist_ok=True)

    if anchor_dir is None:
        anchor_dir = output_path / "anchor"
    redirect_dir = output_path / "redirect"

    if not anchor_dir.is_dir():
        console.print(f"[yellow]No anchor directory found:[/] {anchor_dir}")
        return

    # Match anchor files by exact subpart name
    anchor_files = []
    for sp in doc_subpart_todo_list:
        af_name = f"{sp}.json"
        if (anchor_dir / af_name).is_file():
            anchor_files.append((af_name, sp, sp))

    if not anchor_files:
        console.print("[yellow]No matching anchor files for the given subparts.[/]")
        return

    console.print(f"    [dim]Processing {len(anchor_files)} anchor file(s)...[/dim]")

    total_new = 0
    total_skipped_empty = 0
    total_skipped_duplicate = 0
    total_skip_no_content = 0
    stages = EVIDENCE_STAGES

    for af, subpart, doc_part_key in anchor_files:
        console.print(f"      [cyan]- {af}[/cyan]")
        anchor_path = anchor_dir / af
        with open(anchor_path, "r", encoding="utf-8") as f:
            anchor_data = json.load(f)

        original_tuple = anchor_data.get("anchor_tuple", [])
        # Build content set from original entries
        seen_contents = set()
        for entry in original_tuple:
            fp = _get_evidence_content(entry)
            if fp:
                seen_contents.add(fp)

        # Build (file_path, type) key set from Modified-side entries.
        # When a redirect entry shares the same (file_path, type) as a
        # Modified-side entry, the Modified-side content takes precedence.
        original_anchor_keys = set()
        for entry in original_tuple:
            fp_path, at = _get_anchor_file_path_and_type(entry)
            if fp_path and at:
                original_anchor_keys.add((fp_path, at))

        for stage in stages:
            redirect_filename = f"{stage}_{subpart}.json"
            redirect_path = redirect_dir / redirect_filename
            if not redirect_path.is_file():
                continue

            with open(redirect_path, "r", encoding="utf-8") as f:
                redirect_data = json.load(f)

            redirect_tuple = redirect_data.get("anchor_tuple", [])

            if not redirect_tuple:
                total_skipped_empty += 1
                if verbose:
                    console.print(f"    [dim]  redirect {redirect_path} → empty tuple, skip[/dim]")
                continue

            for entry in redirect_tuple:
                # Skip redirect entries whose (file_path, type) already exists
                # in the Modified-side anchor_tuple. Modified content is preferred
                # because it reflects the post-modification code state; keeping
                # the older Original content can mislead downstream validation
                # into flagging correct generation results as incorrect.
                fp_path, at = _get_anchor_file_path_and_type(entry)
                if fp_path and at and (fp_path, at) in original_anchor_keys:
                    total_skipped_duplicate += 1
                    if verbose:
                        anchor_id = entry.get("anchor_id", "-")
                        console.print(
                            f"    [dim]  {anchor_id}: file_path='{fp_path}' type='{at}' already in Modified, skip[/dim]"
                        )
                    continue

                fp = _get_evidence_content(entry)
                if fp is None:
                    total_skip_no_content += 1
                    if verbose:
                        console.print(f"    [dim]  no content, skip[/dim]")
                    continue

                if fp in seen_contents:
                    total_skipped_duplicate += 1
                    if verbose:
                        anchor_id = entry.get("anchor_id", "-")
                        console.print(f"    [dim]  {anchor_id}: duplicate content, skip[/dim]")
                    continue

                # Prefix evidence_ids with "redirect_" to distinguish from original
                for ev in entry.get("evidence", []):
                    eid = ev.get("evidence_id") or ev.get("envidence_id")
                    if eid and not str(eid).startswith("redirect_"):
                        ev["evidence_id"] = "redirect_" + str(eid)

                original_tuple.append(entry)
                seen_contents.add(fp)
                total_new += 1
                if verbose:
                    anchor_id = entry.get("anchor_id", "-")
                    file_path = entry.get("file_path", "")
                    file_name = Path(file_path).name
                    console.print(f"    [green]  {anchor_id} {stage} → {file_name}[/]")

        out_file = evidence_output_dir / af
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(anchor_data, f, ensure_ascii=False, indent=4)

    console.print(
        f"    [green]Done.[/] written={len(anchor_files)}, "
        + f"[dim]merge={total_new},[/dim] "
        + f"[dim]skip(empty)={total_skipped_empty},[/dim] "
        + f"[dim]skip(duplicate)={total_skipped_duplicate},[/dim] "
        + f"[dim]skip(no_content)={total_skip_no_content}[/dim]"
    )


if __name__ == "__main__":
    from src.generation_validation import config

    re_detection_dir = config.DATA_DIR / "outputs/snapshot/original_b7de76cf/v2/re_detection"
    merge_redirect_to_anchor(
        re_detection_dir,
        ["constraint", "summary"],
        verbose=True,
    )
