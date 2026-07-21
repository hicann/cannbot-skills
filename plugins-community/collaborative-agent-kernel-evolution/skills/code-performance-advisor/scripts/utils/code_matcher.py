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
Code Pattern Matcher for Code Performance Advisor

Identifies code patterns in operator source code that match optimization rules.

Initial version uses keyword-based matching. Future versions can use AST analysis.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeMatch:
    """A match found in operator code."""
    pattern_type: str      # e.g., "explicit_loop", "tail_handling"
    line_start: int
    line_end: int
    matched_text: str
    confidence: float      # 0.0-1.0
    context: str          # Surrounding code context
    description: str


@dataclass
class PatternMatchResult:
    """Result of pattern matching analysis."""
    operator_code: str
    matches: List[CodeMatch] = field(default_factory=list)
    keywords_found: Dict[str, List[int]] = field(default_factory=dict)  # keyword -> line numbers
    statistics: Dict[str, int] = field(default_factory=dict)

    @property
    def has_matches(self) -> bool:
        """Check if any matches were found."""
        return len(self.matches) > 0

    @property
    def match_count(self) -> int:
        """Total number of matches."""
        return len(self.matches)


class CodeMatcher:
    """
    Match code patterns in operator source code.

    Uses keyword-based matching for initial version.
    Can be extended to use AST-based analysis for more sophisticated matching.
    """

    def __init__(self, operator_code: str):
        """
        Initialize matcher with operator code.

        Args:
            operator_code: Source code to analyze
        """
        self.code = operator_code
        self.lines = operator_code.split('\n')

    def find_explicit_loops(self) -> List[CodeMatch]:
        """
        Find explicit for/while loops in code.

        Returns:
            List of matches for loop structures
        """
        matches = []

        for i, line in enumerate(self.lines, start=1):
            # Match C++ for loops
            if re.search(r'\bfor\s*\(', line):
                # Find loop end (simplified - looks for matching braces)
                loop_end = self._find_block_end(i - 1)

                match = CodeMatch(
                    pattern_type="explicit_loop",
                    line_start=i,
                    line_end=loop_end + 1,
                    matched_text='\n'.join(self.lines[i - 1:loop_end + 1]),
                    confidence=0.9,
                    context=self._get_context(i, 3),
                    description=f"Explicit for loop at line {i}"
                )
                matches.append(match)

            # Match while loops
            elif re.search(r'\bwhile\s*\(', line):
                loop_end = self._find_block_end(i - 1)

                match = CodeMatch(
                    pattern_type="explicit_loop",
                    line_start=i,
                    line_end=loop_end + 1,
                    matched_text='\n'.join(self.lines[i - 1:loop_end + 1]),
                    confidence=0.9,
                    context=self._get_context(i, 3),
                    description=f"Explicit while loop at line {i}"
                )
                matches.append(match)

        return matches

    def find_tail_handling(self) -> List[CodeMatch]:
        """
        Find tail data handling patterns (if-else for remainder).

        Returns:
            List of matches for tail handling
        """
        matches = []

        for i, line in enumerate(self.lines, start=1):
            # Look for patterns like: if (tailSize > 0) or if (remainder != 0)
            if re.search(r'\bif\s*\(\s*\w*tail\w*\s*[><!]=?\s*0', line, re.IGNORECASE):
                block_end = self._find_block_end(i - 1)

                match = CodeMatch(
                    pattern_type="tail_handling",
                    line_start=i,
                    line_end=block_end + 1,
                    matched_text='\n'.join(self.lines[i - 1:block_end + 1]),
                    confidence=0.8,
                    context=self._get_context(i, 3),
                    description=f"Tail data handling at line {i}"
                )
                matches.append(match)

            # Also look for modulo operations (often used for tail calculation)
            elif re.search(r'\w+\s*%\s*\w+', line) and 'tail' not in line.lower():
                # Check if this is likely a tail size calculation
                if re.search(r'(tail|remainder|rest)\w*\s*=', line, re.IGNORECASE):
                    match = CodeMatch(
                        pattern_type="tail_calculation",
                        line_start=i,
                        line_end=i,
                        matched_text=line.strip(),
                        confidence=0.7,
                        context=self._get_context(i, 2),
                        description=f"Tail size calculation at line {i}"
                    )
                    matches.append(match)

        return matches

    def find_vector_operations(self) -> List[CodeMatch]:
        """
        Find AscendC vector operations.

        Returns:
            List of matches for vector operations
        """
        matches = []

        # Common AscendC vector operations
        vector_ops = [
            'Add', 'Sub', 'Mul', 'Div', 'Abs', 'Sqrt',
            'Exp', 'Log', 'Sin', 'Cos', 'Tanh',
            'SetVectorMask', 'SetMaskNorm', 'SetMaskCount', 'ResetMask'
        ]

        for i, line in enumerate(self.lines, start=1):
            for op in vector_ops:
                if f'AscendC::{op}' in line or f'::{op}<' in line:
                    match = CodeMatch(
                        pattern_type=f"vector_op_{op.lower()}",
                        line_start=i,
                        line_end=i,
                        matched_text=line.strip(),
                        confidence=1.0,
                        context=self._get_context(i, 1),
                        description=f"Vector operation '{op}' at line {i}"
                    )
                    matches.append(match)

        return matches

    def find_keywords(self, keywords: List[str]) -> Dict[str, List[int]]:
        """
        Find specific keywords in code.

        Args:
            keywords: List of keywords to search for

        Returns:
            Dictionary mapping keywords to line numbers
        """
        keyword_map = {}

        for keyword in keywords:
            lines_found = []
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

            for i, line in enumerate(self.lines, start=1):
                if pattern.search(line):
                    lines_found.append(i)

            if lines_found:
                keyword_map[keyword] = lines_found

        return keyword_map

    def match_pattern(
        self,
        base_code: str,
        good_code: str,
        rule_name: str
    ) -> PatternMatchResult:
        """
        Match operator code against a pattern (base_code vs good_code).

        Args:
            base_code: Anti-pattern code from rule
            good_code: Optimized pattern from rule
            rule_name: Name of the rule

        Returns:
            PatternMatchResult with all matches
        """
        result = PatternMatchResult(operator_code=self.code)

        # Extract keywords from base_code (patterns to look for)
        base_keywords = self._extract_keywords(base_code)

        # Find these keywords in operator code
        result.keywords_found = self.find_keywords(base_keywords)

        # Find common anti-patterns
        result.matches.extend(self.find_explicit_loops())
        result.matches.extend(self.find_tail_handling())
        result.matches.extend(self.find_vector_operations())

        # Calculate statistics
        result.statistics = {
            'total_matches': len(result.matches),
            'explicit_loops': len([m for m in result.matches if m.pattern_type == 'explicit_loop']),
            'tail_handling': len([m for m in result.matches if m.pattern_type == 'tail_handling']),
            'vector_operations': len([m for m in result.matches if m.pattern_type.startswith('vector_op')]),
            'keywords_found': len(result.keywords_found),
            'total_lines': len(self.lines)
        }

        return result

    def _find_block_end(self, start_line: int) -> int:
        """
        Find the end of a code block (closing brace).

        Args:
            start_line: Line number where block starts (0-indexed)

        Returns:
            Line number where block ends (0-indexed)
        """
        brace_count = 0
        found_opening = False

        for i in range(start_line, len(self.lines)):
            line = self.lines[i]

            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1

            # If we've closed all braces, we're done
            if found_opening and brace_count == 0:
                return i

        # If no closing brace found, return start line
        return start_line

    def _get_context(self, line_num: int, context_lines: int = 2) -> str:
        """
        Get surrounding code context for a line.

        Args:
            line_num: Line number (1-indexed)
            context_lines: Number of lines before and after

        Returns:
            Context string
        """
        start = max(0, line_num - context_lines - 1)
        end = min(len(self.lines), line_num + context_lines)

        context_lines_list = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            context_lines_list.append(f"{prefix}{self.lines[i]}")

        return '\n'.join(context_lines_list)

    @staticmethod
    def _extract_keywords(code: str) -> List[str]:
        """
        Extract meaningful keywords from code snippet.

        Args:
            code: Code snippet

        Returns:
            List of keywords
        """
        keywords = set()

        # Common C++ keywords to ignore
        ignore_keywords = {
            'if', 'else', 'for', 'while', 'return', 'const', 'static',
            'void', 'int', 'float', 'double', 'uint32_t', 'bool',
            'true', 'false', 'nullptr', 'auto', 'class', 'struct'
        }

        # Extract identifier-like words
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        words = re.findall(pattern, code)

        for word in words:
            # Filter out common keywords and very short words
            if word.lower() not in ignore_keywords and len(word) > 2:
                keywords.add(word)

        return list(keywords)


# Utility functions

def analyze_code(
    operator_code: str,
    base_code: str,
    good_code: str,
    rule_name: str
) -> PatternMatchResult:
    """
    Convenience function to analyze operator code.

    Args:
        operator_code: Operator source code
        base_code: Anti-pattern from rule
        good_code: Optimized pattern from rule
        rule_name: Rule name

    Returns:
        Pattern match result
    """
    matcher = CodeMatcher(operator_code)
    return matcher.match_pattern(base_code, good_code, rule_name)


def find_optimization_opportunities(
    operator_code: str,
    pattern_types: Optional[List[str]] = None
) -> List[CodeMatch]:
    """
    Find potential optimization opportunities in code.

    Args:
        operator_code: Source code to analyze
        pattern_types: Types of patterns to look for (None = all)

    Returns:
        List of matches
    """
    matcher = CodeMatcher(operator_code)
    matches = []

    if not pattern_types or 'loops' in pattern_types:
        matches.extend(matcher.find_explicit_loops())

    if not pattern_types or 'tail' in pattern_types:
        matches.extend(matcher.find_tail_handling())

    if not pattern_types or 'vector' in pattern_types:
        matches.extend(matcher.find_vector_operations())

    return matches


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if len(sys.argv) < 2:
        logger.info("Usage: python code_matcher.py <operator_code_file>")
        sys.exit(1)

    code_file = Path(sys.argv[1])

    if not code_file.exists():
        logger.info(f"Error: File not found: {code_file}")
        sys.exit(1)

    logger.info(f"Analyzing: {code_file}")
    logger.info("=" * 60)

    with open(code_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # Find all optimization opportunities
    matches = find_optimization_opportunities(code)

    logger.info(f"\n✅ Found {len(matches)} potential optimization opportunities:\n")

    # Group by pattern type
    by_type = {}
    for match in matches:
        if match.pattern_type not in by_type:
            by_type[match.pattern_type] = []
        by_type[match.pattern_type].append(match)

    for pattern_type, type_matches in sorted(by_type.items()):
        logger.info(f"📋 {pattern_type}: {len(type_matches)} occurrences")

        for match in type_matches[:3]:  # Show first 3
            logger.info(f"   Line {match.line_start}: {match.description}")

    # Show example match details
    if matches:
        logger.info(f"\n💡 Example match details (first one):")
        match = matches[0]
        logger.info(f"   Type: {match.pattern_type}")
        logger.info(f"   Lines: {match.line_start}-{match.line_end}")
        logger.info(f"   Confidence: {match.confidence:.0%}")
        logger.info(f"\n   Code:")
        for line in match.matched_text.split('\n')[:5]:
            logger.info(f"      {line}")
