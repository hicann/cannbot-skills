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

"""
Suggestion Generator for Code Performance Advisor

This module implements semi-automated optimization suggestion generation
using a hybrid approach:
- Python: Data loading, rule parsing, evidence extraction
- LLM: Pattern matching, code analysis (optional, manual for now)
- Templates: Structured markdown output
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.goal_loader import load_goal, GoalConfig


@dataclass
class RuleInfo:
    """Information about a matched optimization rule."""
    rule_path: Path
    rule_name: str
    score: float
    coverage_ratio: float
    matched_tags: List[str]
    missing_tags: List[str]
    is_general: bool
    conflict: bool


@dataclass
class SuggestionContext:
    """Context data for generating suggestions."""
    op_name: str
    op_dir: Path
    scored_results: Dict[str, Any]
    goal: GoalConfig
    top_rules: List[RuleInfo] = field(default_factory=list)
    profiling_data: Optional[Dict[str, Any]] = None
    operator_code: Optional[str] = None


class SuggestionGenerator:
    """
    Semi-automated suggestion generator.

    Usage:
        gen = SuggestionGenerator(
            scored_results_path="workspace/sessions/<session_id>/scored_results.json",
            op_name="fastgelu"
        )
        suggestions = gen.generate(top_n=3, score_threshold=0.3)
        gen.render_markdown(suggestions, output_path="suggestions.md")
    """

    def __init__(
        self,
        scored_results_path: Path | str,
        op_name: str,
        op_dir: Optional[Path | str] = None,
        root_dir: Optional[Path | str] = None
    ):
        """
        Initialize suggestion generator.

        Args:
            scored_results_path: Path to scored_results.json from Phase 0
            op_name: Operator name (e.g., "fastgelu")
            op_dir: Operator directory (default: workspace/inputs/{op_name})
            root_dir: Project root directory (default: auto-detect from script path)
        """
        self.scored_results_path = Path(scored_results_path)
        self.op_name = op_name

        # Auto-detect root directory
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parents[2]
        else:
            self.root_dir = Path(root_dir)

        # Auto-detect op_dir
        if op_dir is None:
            self.op_dir = self.root_dir / "workspace" / "inputs" / op_name
        else:
            self.op_dir = Path(op_dir)

        # Load context
        self.context = self._load_context()

    def generate(
        self,
        top_n: int = 3,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions.

        Args:
            top_n: Number of top rules to generate suggestions for
            score_threshold: Minimum score to include

        Returns:
            List of suggestion dictionaries
        """
        logger.info(f"\n🔍 Generating suggestions for '{self.op_name}'...")

        # Load top rules
        self.context.top_rules = self._load_top_rules(
            top_n=top_n,
            score_threshold=score_threshold
        )

        if not self.context.top_rules:
            logger.info(f"⚠️  No rules found with score >= {score_threshold}")
            return []

        logger.info(f"✅ Found {len(self.context.top_rules)} applicable rules")

        # Load operator code
        self.context.operator_code = self._load_operator_code()

        # Load profiling data
        self.context.profiling_data = self._load_profiling_data()

        # Generate suggestions for each rule
        suggestions = []
        for idx, rule in enumerate(self.context.top_rules, 1):
            logger.info(f"\n📋 Processing rule {idx}/{len(self.context.top_rules)}: {rule.rule_name}")

            suggestion = self._generate_suggestion_for_rule(rule)
            suggestions.append(suggestion)

        return suggestions

    def render_markdown(
        self,
        suggestions: List[Dict[str, Any]],
        output_path: Path | str
    ) -> None:
        """
        Render suggestions as markdown file using Jinja2 template.

        Args:
            suggestions: List of suggestion dictionaries
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to use Jinja2 template
        try:
            from jinja2 import Environment, FileSystemLoader
            from datetime import datetime, timezone

            # Setup Jinja2 environment
            template_dir = Path(__file__).parent / "templates"
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            template = env.get_template("suggestion.md.jinja2")

            # Render template
            content = template.render(
                op_name=self.op_name,
                op_dir=self.op_dir,
                goal=self.context.goal,
                suggestions=suggestions,
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                analysis_mode="Fast Path" if suggestions and suggestions[0]['score'] > 0.55 else "Deep Analysis",
                primary_bottleneck=self._identify_primary_bottleneck(suggestions)
            )

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"\n✅ Suggestions written to: {output_path}")

        except ImportError:
            logger.info(f"\n⚠️  Jinja2 not available, using simple markdown rendering")
            content = self._render_simple_markdown(suggestions)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"\n✅ Suggestions written to: {output_path}")

    # ========== Internal helpers ==========

    def _load_context(self) -> SuggestionContext:
        """Load all necessary context data."""
        # Load scored results
        if not self.scored_results_path.exists():
            raise FileNotFoundError(
                f"Scored results not found: {self.scored_results_path}\n"
                f"Run: python cli.py score --op {self.op_name}"
            )

        with open(self.scored_results_path, 'r', encoding='utf-8') as f:
            scored_results = json.load(f)

        # Load performance goal (prioritize from scored_results to avoid re-parsing)
        goal_data = scored_results.get('performance_goal')
        if goal_data:
            # Use goal from Phase 0 (already parsed)
            goal = GoalConfig.from_dict(goal_data)
            logger.info(f"✅ Using performance goal from scored results (Phase 0)")
        else:
            # Fallback: load from goal.md
            logger.info(f"⚠️  No performance_goal in scored_results, loading from goal.md")
            goal = load_goal(self.op_dir)

        context = SuggestionContext(
            op_name=self.op_name,
            op_dir=self.op_dir,
            scored_results=scored_results,
            goal=goal
        )

        return context

    def _load_top_rules(
        self,
        top_n: int = 3,
        score_threshold: float = 0.3,
        exclude_conflicts: bool = True
    ) -> List[RuleInfo]:
        """
        Load top-N rules from scored results.

        Args:
            top_n: Number of top rules to consider
            score_threshold: Minimum score to include
            exclude_conflicts: Skip rules with coverage < 1.0

        Returns:
            List of RuleInfo objects
        """
        results = self.context.scored_results.get("results", [])

        top_rules = []
        for rule_data in results:
            # Apply filters
            if rule_data["score"] < score_threshold:
                continue

            if exclude_conflicts and rule_data.get("conflict", False):
                continue

            # Parse rule info
            rule_path = Path(rule_data["rule_path"])
            rule_info = RuleInfo(
                rule_path=rule_path,
                rule_name=rule_path.stem,
                score=rule_data["score"],
                coverage_ratio=rule_data["coverage_ratio"],
                matched_tags=rule_data.get("matched_tags", []),
                missing_tags=rule_data.get("missing_tags", []),
                is_general=rule_data.get("is_general", False),
                conflict=rule_data.get("conflict", False)
            )

            top_rules.append(rule_info)

            if len(top_rules) >= top_n:
                break

        return top_rules

    def _load_operator_code(self) -> Optional[str]:
        """Load operator source code from op_dir/code/op_kernel/."""
        code_dir = self.op_dir / "code" / "op_kernel"

        if not code_dir.exists():
            logger.info(f"⚠️  Warning: Operator code directory not found: {code_dir}")
            return None

        # Find .cpp files
        cpp_files = list(code_dir.glob("*.cpp"))

        if not cpp_files:
            logger.info(f"⚠️  Warning: No .cpp files found in {code_dir}")
            return None

        # Use the first .cpp file (or find one matching op_name)
        target_file = None
        for cpp_file in cpp_files:
            if self.op_name.lower() in cpp_file.stem.lower():
                target_file = cpp_file
                break

        if target_file is None:
            target_file = cpp_files[0]

        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                code = f.read()
            logger.info(f"✅ Loaded operator code: {target_file.name} ({len(code)} chars)")
            return code
        except Exception as e:
            logger.info(f"⚠️  Warning: Failed to read {target_file}: {e}")
            return None

    def _load_profiling_data(self) -> Optional[Dict[str, Any]]:
        """Load profiling CSV data from either legacy or new workspace layout.

        Supported layouts:
        - Legacy: op_dir/profiling_data/profiling_csv/op_summary*.csv
        - New:    op_dir/profiling/op_summary*.csv (or any *.csv)
        """
        legacy_csv_dir = self.op_dir / "profiling_data" / "profiling_csv"
        new_csv_dir = self.op_dir / "profiling"

        csv_dir = legacy_csv_dir if legacy_csv_dir.exists() else new_csv_dir

        if not csv_dir.exists():
            logger.info(f"⚠️  Warning: Profiling CSV directory not found")
            logger.info(f"   Tried legacy: {legacy_csv_dir}")
            logger.info(f"   Tried new:    {new_csv_dir}")
            return None

        # Prefer op_summary*.csv, fallback to any *.csv
        csv_files = list(csv_dir.glob("op_summary*.csv"))
        if not csv_files:
            csv_files = list(csv_dir.glob("*.csv"))

        if not csv_files:
            logger.info(f"⚠️  Warning: No CSV files found in {csv_dir}")
            return None

        # Use the most recent file
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)

        try:
            # For now, just store the file path
            # Full CSV parsing will be in evidence_extractor.py
            logger.info(f"✅ Found profiling CSV: {csv_file.name}")
            return {
                "csv_path": str(csv_file),
                "csv_name": csv_file.name
            }
        except Exception as e:
            logger.info(f"⚠️  Warning: Failed to process {csv_file}: {e}")
            return None

    def _generate_suggestion_for_rule(self, rule: RuleInfo) -> Dict[str, Any]:
        """Generate suggestion for a single rule."""
        from utils.rule_parser import parse_rule
        from utils.code_matcher import CodeMatcher
        from utils.evidence_extractor import extract_evidence

        # Parse rule documentation
        logger.info(f"   📖 Parsing rule documentation...")
        try:
            rule_info = parse_rule(rule.rule_path.parent)
        except Exception as e:
            logger.info(f"   ⚠️  Failed to parse rule: {e}")
            rule_info = None

        # Extract evidence from profiling
        evidence_text = ""
        if self.context.profiling_data:
            logger.info(f"   📊 Extracting profiling evidence...")
            try:
                csv_path = Path(self.context.profiling_data['csv_path'])
                evidence = extract_evidence(csv_path, self.op_name, rule.matched_tags)
                if evidence:
                    evidence_text = evidence.to_markdown()
                    logger.info(f"   ✅ Evidence extracted ({len(evidence_text)} chars)")
                else:
                    logger.info(f"   ⚠️  No evidence found for {self.op_name}")
            except Exception as e:
                logger.info(f"   ⚠️  Failed to extract evidence: {e}")

        # Match code patterns
        code_matches = []
        if self.context.operator_code and rule_info:
            logger.info(f"   🔍 Matching code patterns...")
            try:
                matcher = CodeMatcher(self.context.operator_code)
                # Find all optimization opportunities
                code_matches.extend(matcher.find_explicit_loops())
                code_matches.extend(matcher.find_tail_handling())
            except Exception as e:
                logger.info(f"   ⚠️  Failed to match code: {e}")

        # Build suggestion
        suggestion = {
            "rule_name": rule.rule_name,
            "rule_path": str(rule.rule_path),
            "score": rule.score,
            "coverage_ratio": rule.coverage_ratio,
            "matched_tags": rule.matched_tags,
            "priority": self._calculate_priority(rule.score),
            "op_name": self.op_name,
            "goal": self.context.goal.to_dict(),

            # Filled from parsers
            "problem": rule_info.requirement if rule_info else "Pattern-based optimization opportunity",
            "pattern_description": rule_info.pattern if rule_info else "",
            "triggers": rule_info.triggers if rule_info else "",
            "evidence": evidence_text,

            # Code examples
            "base_code": rule_info.base_code_examples[0].code if rule_info and rule_info.base_code_examples else "",
            "good_code": rule_info.good_code_examples[0].code if rule_info and rule_info.good_code_examples else "",

            # Code matches
            "code_matches": [
                {
                    "line_start": m.line_start,
                    "line_end": m.line_end,
                    "description": m.description,
                    "context": m.context  # Add context for display
                }
                for m in code_matches[:5]  # Top 5 matches
            ],

            # Placeholders
            "code_changes": [],
            "expected_improvement": "30-50% based on similar patterns" if rule.score > 0.6 else "Performance gain expected",
            "verification_method": rule_info.verification if rule_info else "Re-profile after applying changes"
        }

        return suggestion

    @staticmethod
    def _calculate_priority(score: float) -> str:
        """Calculate priority level based on score."""
        if score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def _identify_primary_bottleneck(suggestions: List[Dict[str, Any]]) -> str:
        """Identify primary bottleneck from suggestions with specific metrics."""
        import re

        if not suggestions:
            return "Unknown"

        # Look for evidence in top suggestion
        top_suggestion = suggestions[0]
        evidence = top_suggestion.get('evidence', '')

        if not evidence:
            return "Performance optimization opportunity"

        # Extract specific metrics from evidence
        bottleneck_parts = []

        # Check for bottleneck type
        bottleneck_match = re.search(r'\*\*Bottleneck Type\*\*:\s*(\w+)', evidence)
        if bottleneck_match:
            bottleneck_type = bottleneck_match.group(1)

            # Extract scalar ratio if scalar bottleneck
            if bottleneck_type == 'scalar':
                scalar_match = re.search(r'aiv_scalar_ratio=([0-9.]+)', evidence)
                vec_match = re.search(r'aiv_vec_ratio=([0-9.]+)', evidence)

                if scalar_match:
                    scalar_ratio = float(scalar_match.group(1))
                    bottleneck_parts.append(f"Scalar instruction overhead (aiv_scalar_ratio={scalar_ratio:.3f})")

                    if vec_match:
                        vec_ratio = float(vec_match.group(1))
                        bottleneck_parts.append(f"low vector utilization (aiv_vec_ratio={vec_ratio:.3f})")
                else:
                    bottleneck_parts.append("Scalar instruction overhead")

            elif bottleneck_type == 'vector_underutilized':
                vec_match = re.search(r'aiv_vec_ratio=([0-9.]+)', evidence)
                if vec_match:
                    vec_ratio = float(vec_match.group(1))
                    bottleneck_parts.append(f"Low vector utilization (aiv_vec_ratio={vec_ratio:.3f})")
                else:
                    bottleneck_parts.append("Low vector utilization")
            else:
                bottleneck_parts.append(bottleneck_type.replace('_', ' ').title())

        # Extract scalar time multiplier if available
        time_match = re.search(r'Scalar time.*?is\s+([0-9.]+)x\s+vector time', evidence)
        if time_match:
            multiplier = time_match.group(1)
            bottleneck_parts.append(f"scalar {multiplier}x slower than vector")

        # Combine parts or use generic message
        if bottleneck_parts:
            return ", ".join(bottleneck_parts)
        elif 'scalar' in evidence.lower():
            return "Scalar instruction overhead"
        elif 'vector' in evidence.lower() and 'low' in evidence.lower():
            return "Low vector utilization"
        else:
            return "Performance optimization opportunity"

    def _render_simple_markdown(self, suggestions: List[Dict[str, Any]]) -> str:
        """Simple markdown rendering (before Jinja2 template)."""
        lines = []

        # Header
        lines.append(f"# {self.op_name.title()} Performance Optimization Suggestions")
        lines.append("")
        lines.append(f"**Generated**: Auto-generated by suggest_template.py")
        lines.append(f"**Operator**: {self.op_name}")
        lines.append(f"**Performance Goal**: {self.context.goal.improvement_percentage} improvement")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        if suggestions:
            top_rule = suggestions[0]
            lines.append(f"- **Top Recommendation**: {top_rule['rule_name']} (score: {top_rule['score']:.3f})")
            lines.append(f"- **Priority**: {top_rule['priority']}")
            lines.append(f"- **Matched Tags**: {', '.join(top_rule['matched_tags'])}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Individual suggestions
        for idx, suggestion in enumerate(suggestions, 1):
            lines.append(f"## 🎯 Optimization #{idx}: {suggestion['rule_name']} (Priority: {suggestion['priority']})")
            lines.append("")
            lines.append(f"**Score**: {suggestion['score']:.3f} | **Coverage**: {suggestion['coverage_ratio']:.1%}")
            lines.append("")
            lines.append(f"**Matched Tags**: {', '.join(suggestion['matched_tags'])}")
            lines.append("")
            lines.append("### 📊 Problem Diagnosis")
            lines.append("")
            lines.append("*[To be populated by evidence_extractor and rule_parser]*")
            lines.append("")
            lines.append("### 🔍 Why This Rule Applies")
            lines.append("")
            lines.append("*[To be populated by rule_parser]*")
            lines.append("")
            lines.append("### 🔬 Code Analysis")
            lines.append("")
            lines.append("*[To be populated by code_matcher and rule_parser]*")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)
