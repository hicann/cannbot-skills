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
Rule Parser for Code Performance Advisor

Extracts structured information from rule markdown files.

Structure of a rule directory:
    R_RULE_NAME/
    ├── R_RULE_NAME.md          # Main rule document
    ├── R_RULE_NAME_tags.json   # Tag definitions
    └── code_snippets/
        └── case0/
            ├── base_code/      # Anti-pattern examples
            │   └── base_code.md
            └── good_code/      # Optimized pattern
                └── good_code.md
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RuleSection:
    """Parsed section from rule document."""
    title: str
    content: str
    level: int  # Header level (1-6)


@dataclass
class CodeExample:
    """Code example from rule."""
    code: str
    language: str  # cpp, python, etc.
    description: str
    file_path: Optional[Path] = None


@dataclass
class RuleInfo:
    """Structured information extracted from a rule."""
    rule_id: str
    rule_path: Path

    # Main sections
    requirement: str
    pattern: str
    inference: str
    triggers: str
    action: str
    constraints: str
    verification: str

    # Code examples
    base_code_examples: List[CodeExample] = field(default_factory=list)
    good_code_examples: List[CodeExample] = field(default_factory=list)

    # Metadata
    tags: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Generate a brief summary of the rule."""
        lines = []
        if self.pattern:
            # Extract first line or paragraph
            first_para = self.pattern.split('\n\n')[0]
            lines.append(first_para.strip())
        return '\n'.join(lines)


class RuleParser:
    """Parse rule markdown files to extract structured information."""

    def __init__(self, rule_path: Path):
        """
        Initialize parser for a rule.

        Args:
            rule_path: Path to rule .md file or rule directory
        """
        if rule_path.is_dir():
            # Find the main .md file in directory
            self.rule_dir = rule_path
            md_files = list(rule_path.glob("*.md"))
            if not md_files:
                raise FileNotFoundError(f"No .md file found in {rule_path}")
            self.rule_file = md_files[0]
        else:
            self.rule_file = rule_path
            self.rule_dir = rule_path.parent

        self.rule_id = self.rule_file.stem

    def parse(self) -> RuleInfo:
        """
        Parse the rule file and extract information.

        Returns:
            RuleInfo object with extracted data
        """
        # Read main document
        with open(self.rule_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse sections
        sections = self._parse_sections(content)

        # Extract main sections
        rule_info = RuleInfo(
            rule_id=self.rule_id,
            rule_path=self.rule_file,
            requirement=self._get_section_content(sections, ['需求场景', 'Requirement']),
            pattern=self._get_section_content(sections, ['模式描述', 'Pattern']),
            inference=self._get_section_content(sections, ['性能损耗因果链', 'Inference', 'Physics']),
            triggers=self._get_section_content(sections, ['触发信号', 'Triggers']),
            action=self._get_section_content(sections, ['动作实现', 'Action']),
            constraints=self._get_section_content(sections, ['约束与副作用', 'Constraints']),
            verification=self._get_section_content(sections, ['验证逻辑', 'Verification'])
        )

        # Extract tags
        rule_info.tags = self._parse_tags(sections)

        # Load code examples
        rule_info.base_code_examples = self._load_code_examples('base_code')
        rule_info.good_code_examples = self._load_code_examples('good_code')

        return rule_info

    def get_pattern_summary(self) -> str:
        """
        Get a concise summary of the optimization pattern.

        Returns:
            Pattern summary string
        """
        rule_info = self.parse()

        # Extract key points from pattern section
        pattern = rule_info.pattern
        if not pattern:
            return ""

        # Look for bullet points or numbered lists
        lines = pattern.split('\n')
        key_points = []

        for line in lines:
            line = line.strip()
            # Match bullet points or numbered items
            if line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line):
                key_points.append(line)

        if key_points:
            return '\n'.join(key_points)
        else:
            # Return first paragraph
            paragraphs = pattern.split('\n\n')
            return paragraphs[0].strip() if paragraphs else ""

    @staticmethod
    def _parse_sections(content: str) -> List[RuleSection]:
        """
        Parse markdown into sections based on headers.

        Args:
            content: Markdown content

        Returns:
            List of RuleSection objects
        """
        sections = []
        lines = content.split('\n')

        current_section = None
        current_content = []

        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#+)\s+(.+)$', line)

            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = RuleSection(title=title, content='', level=level)
                current_content = []
            else:
                # Accumulate content
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        return sections

    @staticmethod
    def _get_section_content(
        sections: List[RuleSection],
        keywords: List[str]
    ) -> str:
        """
        Get content of a section by matching keywords.

        Args:
            sections: List of parsed sections
            keywords: List of possible section titles

        Returns:
            Section content or empty string
        """
        for section in sections:
            for keyword in keywords:
                if keyword.lower() in section.title.lower():
                    return section.content
        return ""

    def _parse_tags(self, sections: List[RuleSection]) -> Dict[str, List[str]]:
        """
        Extract tags from the tags section.

        Args:
            sections: List of parsed sections

        Returns:
            Dictionary of tag categories
        """
        tags = {}

        # Find tags section
        tag_content = self._get_section_content(sections, ['标签', 'Tags', 'Tag'])

        if not tag_content:
            return tags

        # Parse tag lines
        for line in tag_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Match pattern: "- Category: `tag1`, `tag2`"
            match = re.match(r'-\s*(\w+):\s*(.+)', line)
            if match:
                category = match.group(1).strip()
                tag_str = match.group(2).strip()

                # Extract tags (remove backticks and split by comma)
                tag_list = [
                    t.strip().strip('`').strip("'").strip('"')
                    for t in tag_str.split(',')
                ]

                tags[category.lower()] = tag_list

        return tags

    def _load_code_examples(self, example_type: str) -> List[CodeExample]:
        """
        Load code examples from code_snippets directory.

        Args:
            example_type: 'base_code' or 'good_code'

        Returns:
            List of CodeExample objects
        """
        examples = []

        snippets_dir = self.rule_dir / 'code_snippets'
        if not snippets_dir.exists():
            return examples

        # Find all case directories
        case_dirs = [d for d in snippets_dir.iterdir() if d.is_dir()]

        for case_dir in sorted(case_dirs):
            example_dir = case_dir / example_type
            if not example_dir.exists():
                continue

            # Find code files (usually .md files containing code blocks)
            code_files = list(example_dir.glob('*.md'))

            for code_file in code_files:
                with open(code_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract code blocks
                code_blocks = self._extract_code_blocks(content)

                for code, language in code_blocks:
                    example = CodeExample(
                        code=code,
                        language=language,
                        description=f"{example_type} from {case_dir.name}",
                        file_path=code_file
                    )
                    examples.append(example)

        return examples

    @staticmethod
    def _extract_code_blocks(content: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from markdown.

        Args:
            content: Markdown content

        Returns:
            List of (code, language) tuples
        """
        code_blocks = []

        # Pattern: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            code_blocks.append((code, language))

        return code_blocks


# Utility functions

def parse_rule(rule_path: Path) -> RuleInfo:
    """
    Convenience function to parse a rule.

    Args:
        rule_path: Path to rule file or directory

    Returns:
        Parsed RuleInfo object
    """
    parser = RuleParser(rule_path)
    return parser.parse()


def load_code_snippet(rule_path: Path, snippet_type: str, case: str = 'case0') -> Optional[str]:
    """
    Load a specific code snippet from a rule.

    Args:
        rule_path: Path to rule directory
        snippet_type: 'base_code' or 'good_code'
        case: Case name (default: 'case0')

    Returns:
        Code string or None if not found
    """
    snippet_path = rule_path / 'code_snippets' / case / snippet_type

    if not snippet_path.exists():
        return None

    # Find .md file
    md_files = list(snippet_path.glob('*.md'))
    if not md_files:
        return None

    with open(md_files[0], 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract first code block
    pattern = r'```\w*\n(.*?)```'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if len(sys.argv) < 2:
        logger.info("Usage: python rule_parser.py <rule_path>")
        logger.info("Example: python rule_parser.py assets/rules/special_rules/R_API_VECTOR_COUNTER_MODE/")
        sys.exit(1)

    rule_path = Path(sys.argv[1])

    logger.info(f"Parsing rule: {rule_path}")
    logger.info("=" * 60)

    try:
        parser = RuleParser(rule_path)
        rule_info = parser.parse()

        logger.info(f"\n✅ Rule ID: {rule_info.rule_id}")
        logger.info(f"\n📋 Pattern:")
        logger.info(rule_info.pattern[:200] + "..." if len(rule_info.pattern) > 200 else rule_info.pattern)

        logger.info(f"\n🔍 Triggers:")
        logger.info(rule_info.triggers[:200] + "..." if len(rule_info.triggers) > 200 else rule_info.triggers)

        logger.info(f"\n🏷️  Tags:")
        for category, tags in rule_info.tags.items():
            logger.info(f"   {category}: {', '.join(tags)}")

        logger.info(f"\n📝 Code Examples:")
        logger.info(f"   Base code examples: {len(rule_info.base_code_examples)}")
        logger.info(f"   Good code examples: {len(rule_info.good_code_examples)}")

        if rule_info.base_code_examples:
            logger.info(f"\n💡 Base Code Sample (first 10 lines):")
            lines = rule_info.base_code_examples[0].code.split('\n')[:10]
            for line in lines:
                logger.info(f"   {line}")

        if rule_info.good_code_examples:
            logger.info(f"\n✨ Good Code Sample (first 10 lines):")
            lines = rule_info.good_code_examples[0].code.split('\n')[:10]
            for line in lines:
                logger.info(f"   {line}")

    except Exception as e:
        logger.info(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
