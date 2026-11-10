#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * Markdown 代码格式校验工具 (Node.js 版)
 *
 * 从 Markdown 文档中提取 ``` 包裹的代码块，识别语言（Python/C++/Shell），
 * 对每个代码块执行格式规则检查（只查格式不查逻辑），输出 JSON 格式的问题报告。
 *
 * 用法:
 *   node code_lint.js <Markdown文件或文件夹路径> <检查结果总JSON文件绝对路径>
 *
 * 参数:
 *   第一个参数: Markdown 文件路径或文件夹路径
 *   第二个参数: 检查结果总JSON文件的绝对路径（检查结果追加合并到该文件）
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ============================================================
// 辅助函数
// ============================================================

/**
 * 移除字符串和注释，使括号统计更准确
 */
function cleanForBracketCheck(code, lang) {
    let result = code;
    if (lang === 'python') {
        // 移除三引号字符串
        result = result.replace(/"""[\s\S]*?"""/g, '""');
        result = result.replace(/'''[\s\S]*?'''/g, "''");
        // 移除单行注释
        result = result.replace(/#.*/g, '');
    } else if (lang === 'cpp') {
        // 移除块注释
        result = result.replace(/\/\*[\s\S]*?\*\//g, '');
        // 移除单行注释
        result = result.replace(/\/\/.*/g, '');
    } else if (lang === 'shell') {
        // 移除注释（# 后面的内容，但 # 不在行首时不一定是注释）
        result = result.replace(/^#.*$/gm, '');
        result = result.replace(/\s#.*(?=\n?$)/g, '');
    }
    // 移除双引号字符串
    result = result.replace(/"[^"]*"/g, '""');
    // 移除单引号字符串
    result = result.replace(/'[^']*'/g, "''");
    return result;
}

/**
 * 获取行首空白字符
 */
function getLeadingWhitespace(line) {
    return line.slice(0, line.length - line.replace(/^\s+/, '').length);
}

/**
 * 统计字符出现次数（等价于 Python str.count）
 */
function countChar(str, ch) {
    return str.split(ch).length - 1;
}

// ============================================================
// 数据结构
// ============================================================

class Issue {
    constructor(line, column, rule, message, severity, source) {
        this.line = line;
        this.column = column;
        this.rule = rule;
        this.message = message;
        this.severity = severity;
        this.source = source;
    }
}

class CodeBlock {
    constructor(language, rawLang, code, startLine, blockIndex) {
        this.language = language;
        this.raw_lang = rawLang;
        this.code = code;
        this.start_line = startLine;
        this.block_index = blockIndex;
        this.issues = [];
    }
}

// ============================================================
// Markdown 解析器
// ============================================================

class MarkdownParser {
    constructor() {
        this.LANG_MAP = {
            'python': 'python', 'py': 'python', 'python3': 'python',
            'cpp': 'cpp', 'c++': 'cpp', 'c': 'cpp', 'cxx': 'cpp', 'hpp': 'cpp', 'cc': 'cpp',
            'bash': 'shell', 'sh': 'shell', 'shell': 'shell', 'zsh': 'shell',
        };
    }

    parse(content) {
        const blocks = [];
        const lines = content.split('\n');
        const fencePattern = /^```(\S*)\s*$/;
        let blockIndex = 0;
        let i = 0;
        while (i < lines.length) {
            const match = fencePattern.exec(lines[i].replace(/\s+$/, ''));
            if (match) {
                const langTag = match[1].toLowerCase();
                const start = i + 2;  // 1-based 行号，代码内容从下一行开始
                // 找结束的 ```
                const codeLines = [];
                let j = i + 1;
                while (j < lines.length) {
                    if (fencePattern.exec(lines[j].replace(/\s+$/, ''))) {
                        break;
                    }
                    codeLines.push(lines[j]);
                    j++;
                }

                if (j < lines.length) {  // 找到结束 ```
                    const lang = this.LANG_MAP[langTag];
                    if (lang) {
                        blockIndex++;
                        blocks.push(new CodeBlock(
                            lang,
                            langTag,
                            codeLines.join('\n'),
                            start,
                            blockIndex,
                        ));
                    }
                    i = j + 1;
                } else {
                    // 没有找到结束 ```，代码块未闭合
                    i = j;
                }
            } else {
                i++;
            }
        }
        return blocks;
    }
}

// ============================================================
// 规则基类
// ============================================================

class RuleBase {
    constructor() {
        this.name = '';
        this.severity = 'warning';
    }

    check(code, lang = '') {
        throw new Error('NotImplemented');
    }
}

// ============================================================
// 通用规则（所有语言都执行）
// ============================================================

class TrailingWhitespaceRule extends RuleBase {
    constructor() {
        super();
        this.name = 'trailing-whitespace';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            const stripped = line.replace(/[ \t]+$/, '');
            if (line !== stripped) {
                issues.push(new Issue(
                    lineNum, stripped.length + 1,
                    this.name, '行尾有多余空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class TabMixedSpaceRule extends RuleBase {
    constructor() {
        super();
        this.name = 'tab-mixed-space';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            const leading = getLeadingWhitespace(line);
            if (leading.includes('\t') && leading.includes(' ')) {
                issues.push(new Issue(
                    lineNum, 1,
                    this.name, '行首 Tab 和空格混用',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class LineTooLongRule extends RuleBase {
    constructor(maxLength = 100) {
        super();
        this.name = 'line-too-long';
        this.max_length = maxLength;
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            if (line.length > this.max_length) {
                issues.push(new Issue(
                    lineNum, this.max_length + 1,
                    this.name,
                    `行长度 ${line.length} 超过 ${this.max_length} 字符`,
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class BracketMismatchRule extends RuleBase {
    constructor() {
        super();
        this.name = 'bracket-mismatch';
        this.severity = 'error';
    }

    check(code, lang = '') {
        const issues = [];
        const cleaned = cleanForBracketCheck(code, lang);
        const pairs = { '(': ')', '[': ']', '{': '}' };
        const opens = new Set(['(', '[', '{']);
        const closes = new Set([')', ']', '}']);
        const stack = [];  // [char, line, col]

        const lines = cleaned.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            for (let col = 0; col < line.length; col++) {
                const ch = line[col];
                const colNum = col + 1;
                if (opens.has(ch)) {
                    stack.push([ch, lineNum, colNum]);
                } else if (closes.has(ch)) {
                    if (stack.length > 0) {
                        const [topCh, topLine, topCol] = stack[stack.length - 1];
                        if (pairs[topCh] === ch) {
                            stack.pop();
                        } else {
                            issues.push(new Issue(
                                lineNum, colNum,
                                this.name,
                                `括号不匹配: 期望 "${pairs[topCh]}" 但遇到 "${ch}"`,
                                this.severity, line,
                            ));
                            stack.pop();
                        }
                    } else {
                        issues.push(new Issue(
                            lineNum, colNum,
                            this.name,
                            `多余的闭合括号 "${ch}"，无对应开括号`,
                            this.severity, line,
                        ));
                    }
                }
            }
        }

        for (const [ch, lineNum, col] of stack) {
            issues.push(new Issue(
                lineNum, col,
                this.name,
                `未闭合的括号 "${ch}"`,
                this.severity, '',
            ));
        }
        return issues;
    }
}

class ConsecutiveBlankLinesRule extends RuleBase {
    constructor() {
        super();
        this.name = 'consecutive-blank-lines';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        let blankCount = 0;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            if (line.trim() === '') {
                blankCount++;
            } else {
                if (blankCount > 2) {
                    issues.push(new Issue(
                        lineNum - blankCount, 0,
                        this.name,
                        `连续 ${blankCount} 个空行（建议不超过 2 个）`,
                        this.severity, '',
                    ));
                }
                blankCount = 0;
            }
        }
        if (blankCount > 2) {
            issues.push(new Issue(
                lines.length - blankCount + 1, 0,
                this.name,
                `连续 ${blankCount} 个空行（建议不超过 2 个）`,
                this.severity, '',
            ));
        }
        return issues;
    }
}

// ============================================================
// Python 专属规则
// ============================================================

class PyIndentNot4SpaceRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-indent-not-4-space';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            if (line.trim() === '') {
                continue;
            }
            const leading = getLeadingWhitespace(line);
            if (leading.includes('\t')) {
                continue;  // Tab 单独处理
            }
            const spaces = leading.length;
            if (spaces > 0 && spaces % 4 !== 0) {
                issues.push(new Issue(
                    lineNum, 1,
                    this.name,
                    `缩进 ${spaces} 个空格，不是 4 的倍数（PEP 8）`,
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyTabIndentationRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-tab-indentation';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            const leading = getLeadingWhitespace(line);
            if (leading.includes('\t')) {
                issues.push(new Issue(
                    lineNum, 1,
                    this.name,
                    '使用 Tab 缩进（PEP 8 建议使用 4 个空格）',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyMultipleStatementsRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-multiple-statements';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.split('#')[0];
            // 排除 for 循环
            if (/^\s*for\s/.test(cleaned)) {
                continue;
            }
            // 检查分号后是否有非空字符
            for (const m of cleaned.matchAll(/;\s*\S/g)) {
                const col = m.index + 1;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '同行多个语句（使用分号分隔）',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyMissingSpaceAfterCommaRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-missing-space-after-comma';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.split('#')[0];
            // 查找逗号后紧跟非空格字符（排除 ), ], } 和行尾）
            for (const m of cleaned.matchAll(/,(?![\s)\]\}\n])/g)) {
                const col = m.index + 2;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '逗号后缺少空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyMissingSpaceAfterColonRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-missing-space-after-colon';
        this.KEYWORDS = new Set([
            'def', 'if', 'elif', 'for', 'while', 'class',
            'else', 'try', 'except', 'finally', 'with',
        ]);
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 检查是否包含关键字
            const words = [...line.matchAll(/\b(\w+)\b/g)].map(m => m[1]);
            const hasKeyword = words.some(w => this.KEYWORDS.has(w));
            if (!hasKeyword) {
                continue;
            }
            // 找最后一个冒号
            const colonIdx = line.lastIndexOf(':');
            if (colonIdx === -1) {
                continue;
            }
            // 冒号后面有非空格字符
            if (colonIdx < line.length - 1 && !/\s/.test(line[colonIdx + 1])) {
                issues.push(new Issue(
                    lineNum, colonIdx + 2,
                    this.name,
                    '冒号后缺少空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyExtraSpaceBeforeParenRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-extra-space-before-paren';
        this.KEYWORDS = new Set([
            'if', 'for', 'while', 'def', 'class', 'with', 'elif',
            'return', 'print', 'assert',
        ]);
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            for (const m of cleaned.matchAll(/\b(\w+)\s+\(/g)) {
                const word = m[1];
                if (!this.KEYWORDS.has(word)) {
                    const col = m.index + word.length + 1;
                    issues.push(new Issue(
                        lineNum, col,
                        this.name,
                        '函数调用括号前有多余空格',
                        this.severity, line,
                    ));
                }
            }
        }
        return issues;
    }
}

class PyExtraSpaceAfterAssignmentRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-extra-space-after-assignment';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.split('#')[0];
            // 查找 = 后面有多个连续空格（排除 ==, !=, <=, >=, +=, -= 等）
            for (const m of cleaned.matchAll(/(?<![=!<>+\-*/%&|^])=(?![=])/g)) {
                const afterPos = m.index + m[0].length;
                if (afterPos < cleaned.length) {
                    const rest = cleaned.slice(afterPos);
                    if (rest.startsWith('  ')) {
                        const col = afterPos + 1;
                        issues.push(new Issue(
                            lineNum, col,
                            this.name,
                            '赋值运算符 = 后面有多个连续空格',
                            this.severity, line,
                        ));
                    }
                }
            }
        }
        return issues;
    }
}

class PySpaceAfterDotRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-space-after-dot';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.split('#')[0];
            // 查找 . 后面紧跟空格再跟标识符
            for (const m of cleaned.matchAll(/\.\s+\w/g)) {
                const col = m.index + 2;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '属性访问时 . 后面有多余空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyMultipleSpacesAfterCommaRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-multiple-spaces-after-comma';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.split('#')[0];
            // 查找逗号后有2个或更多空格
            for (const m of cleaned.matchAll(/,\s{2,}/g)) {
                const col = m.index + 2;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '逗号后有多个连续空格（应该只有一个空格）',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class PyMissingSpaceBeforeAssignmentRule extends RuleBase {
    constructor() {
        super();
        this.name = 'py-missing-space-before-assignment';
    }

    check(code, lang = '') {
        const issues = [];
        let baseDepth = 0;  // 前面行累积的括号深度
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.split('#')[0];
            // 查找 = 前面没有空格（排除 ==, !=, <=, >=, +=, -= 等）
            for (const m of cleaned.matchAll(/(?<![=!<>+\-*/%&|^])=(?![=])/g)) {
                // 计算到 = 位置为止的括号深度（当前行前缀 + 前面行累积）
                const prefix = cleaned.slice(0, m.index);
                const lineDepth = (countChar(prefix, '(') - countChar(prefix, ')')
                    + countChar(prefix, '[') - countChar(prefix, ']')
                    + countChar(prefix, '{') - countChar(prefix, '}'));
                if (baseDepth + lineDepth > 0) {
                    continue;  // 在括号内，可能是关键字参数，跳过
                }
                const beforePos = m.index - 1;
                if (beforePos >= 0) {
                    const charBefore = cleaned[beforePos];
                    if (charBefore !== ' ') {
                        const col = m.index + 1;
                        issues.push(new Issue(
                            lineNum, col,
                            this.name,
                            '赋值运算符 = 前缺少空格',
                            this.severity, line,
                        ));
                    }
                }
            }
            // 更新基础深度
            baseDepth += (countChar(cleaned, '(') - countChar(cleaned, ')')
                + countChar(cleaned, '[') - countChar(cleaned, ']')
                + countChar(cleaned, '{') - countChar(cleaned, '}'));
            if (baseDepth < 0) {
                baseDepth = 0;
            }
        }
        return issues;
    }
}

// ============================================================
// C++ 专属规则
// ============================================================

class CppTabIndentationRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-tab-indentation';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            const leading = getLeadingWhitespace(line);
            if (leading.includes('\t')) {
                issues.push(new Issue(
                    lineNum, 1,
                    this.name,
                    '使用 Tab 缩进（建议统一使用空格）',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class CppSpaceInParenRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-space-in-paren';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/\/\/.*/g, '');
            // 左括号后紧跟空格再跟非空格
            for (const m of cleaned.matchAll(/\(\s+\S/g)) {
                const col = m.index + 1;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '左括号后有多余空格',
                    this.severity, line,
                ));
            }
            // 非空格后紧跟空格再跟右括号
            for (const m of cleaned.matchAll(/\S\s+\)/g)) {
                const col = m.index + 1;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '右括号前有多余空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class CppExtraSpaceAroundOperatorRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-space-around-operator';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/\/\/.*/g, '');
            // 检查 = 周围缺少空格（排除 ==, !=, <=, >=, +=, -= 等）
            for (const m of cleaned.matchAll(/(?<![=!<>+\-*/%&|^])=(?![=])/g)) {
                const col = m.index + 1;
                // 检查前后是否有空格
                const before = m.index > 0 ? cleaned[m.index - 1] : ' ';
                const after = m.index + 1 < cleaned.length ? cleaned[m.index + 1] : ' ';
                if (before !== ' ' || after !== ' ') {
                    issues.push(new Issue(
                        lineNum, col,
                        this.name,
                        '赋值运算符 = 周围缺少空格',
                        this.severity, line,
                    ));
                }
            }
        }
        return issues;
    }
}

class CppExtraSpaceAfterAssignmentRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-extra-space-after-assignment';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/\/\/.*/g, '');
            // 查找 = 后面有多个连续空格（排除 ==, !=, <=, >=, +=, -= 等）
            for (const m of cleaned.matchAll(/(?<![=!<>+\-*/%&|^])=(?![=])/g)) {
                const afterPos = m.index + m[0].length;
                if (afterPos < cleaned.length) {
                    const rest = cleaned.slice(afterPos);
                    if (rest.startsWith('  ')) {
                        const col = afterPos + 1;
                        issues.push(new Issue(
                            lineNum, col,
                            this.name,
                            '赋值运算符 = 后面有多个连续空格',
                            this.severity, line,
                        ));
                    }
                }
            }
        }
        return issues;
    }
}

class CppSpaceAfterDotRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-space-after-dot';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/\/\/.*/g, '');
            // 查找 . 后面紧跟空格再跟标识符（排除浮点数如 3.14 和省略号 ...）
            for (const m of cleaned.matchAll(/(?<![\d.])\.\s+\w/g)) {
                const col = m.index + 2;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '成员访问 . 后面有多余空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class CppSpaceInArrowOperatorRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-space-in-arrow-operator';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/\/\/.*/g, '');
            // 检测 - 和 > 之间有空格（破坏了 -> 运算符）
            for (const m of cleaned.matchAll(/(?<![-])-\s+>(?![-])/g)) {
                const col = m.index + 1;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '箭头运算符 -> 中间有多余空格',
                    this.severity, line,
                ));
            }
            // 检测 -> 后面有 2+ 个空格
            for (const m of cleaned.matchAll(/->\s{2,}\S/g)) {
                const col = m.index + 3;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '箭头运算符 -> 后面有多个连续空格',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class CppIndentNot4SpaceRule extends RuleBase {
    constructor() {
        super();
        this.name = 'cpp-indent-not-4-space';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            if (line.trim() === '') {
                continue;
            }
            const leading = getLeadingWhitespace(line);
            if (leading.includes('\t')) {
                continue;  // Tab 单独处理
            }
            const spaces = leading.length;
            if (spaces > 0 && spaces % 4 !== 0) {
                issues.push(new Issue(
                    lineNum, 1,
                    this.name,
                    `缩进 ${spaces} 个空格，不是 4 的倍数`,
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

// ============================================================
// Shell 专属规则
// ============================================================

class ShSpaceAroundPipeRule extends RuleBase {
    constructor() {
        super();
        this.name = 'sh-space-around-pipe';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/#.*/g, '');
            // 查找 | 前面或后面缺少空格（排除 ||）
            for (const m of cleaned.matchAll(/(?<!\|)\|(?!\|)/g)) {
                const col = m.index + 1;
                const before = m.index > 0 ? cleaned[m.index - 1] : ' ';
                const after = m.index + 1 < cleaned.length ? cleaned[m.index + 1] : ' ';
                if (before !== ' ' || after !== ' ') {
                    issues.push(new Issue(
                        lineNum, col,
                        this.name,
                        '管道符 | 周围缺少空格',
                        this.severity, line,
                    ));
                }
            }
        }
        return issues;
    }
}

class ShTabIndentationRule extends RuleBase {
    constructor() {
        super();
        this.name = 'sh-tab-indentation';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            const leading = getLeadingWhitespace(line);
            if (leading.includes('\t')) {
                issues.push(new Issue(
                    lineNum, 1,
                    this.name,
                    '使用 Tab 缩进（建议统一使用空格）',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

class ShSpaceAroundRedirectRule extends RuleBase {
    constructor() {
        super();
        this.name = 'sh-space-around-redirect';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串和注释
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            cleaned = cleaned.replace(/#.*/g, '');
            // 检查 > 前面缺少空格（排除 >>, 2>, 1> 等变体）
            for (const m of cleaned.matchAll(/(?<![<>0-9])>(?![>])/g)) {
                const col = m.index + 1;
                const before = m.index > 0 ? cleaned[m.index - 1] : ' ';
                const after = m.index + 1 < cleaned.length ? cleaned[m.index + 1] : ' ';
                if (before !== ' ' || after !== ' ') {
                    issues.push(new Issue(
                        lineNum, col,
                        this.name,
                        '重定向符 > 周围缺少空格',
                        this.severity, line,
                    ));
                }
            }
        }
        return issues;
    }
}

class ShMultipleSpacesRule extends RuleBase {
    constructor() {
        super();
        this.name = 'sh-multiple-spaces';
    }

    check(code, lang = '') {
        const issues = [];
        const lines = code.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            // 移除字符串
            let cleaned = line.replace(/"[^"]*"/g, '""');
            cleaned = cleaned.replace(/'[^']*'/g, "''");
            // 移除注释
            cleaned = cleaned.replace(/#.*/g, '');
            // 去除行首空白（缩进不在此规则检查范围）
            const leadingLen = cleaned.length - cleaned.replace(/^\s+/, '').length;
            const stripped = cleaned.replace(/^\s+/, '');
            // 查找 2+ 个连续空格（在非空白字符之间）
            for (const m of stripped.matchAll(/\S\s{2,}\S/g)) {
                const col = leadingLen + m.index + 2;
                issues.push(new Issue(
                    lineNum, col,
                    this.name,
                    '单词间有多个连续空格（应该只有一个空格）',
                    this.severity, line,
                ));
            }
        }
        return issues;
    }
}

// ============================================================
// Linter 调度器
// ============================================================

class Linter {
    constructor(maxLineLength = 100, langFilter = null) {
        this.max_line_length = maxLineLength;
        this.lang_filter = langFilter ? new Set(langFilter) : null;

        // 通用规则（所有语言都执行）
        this.common_rules = [
            new TrailingWhitespaceRule(),
            new TabMixedSpaceRule(),
            new LineTooLongRule(maxLineLength),
            new BracketMismatchRule(),
            new ConsecutiveBlankLinesRule(),
        ];

        // 语言专属规则
        this.lang_rules = {
            'python': [
                new PyIndentNot4SpaceRule(),
                new PyTabIndentationRule(),
                new PyMultipleStatementsRule(),
                new PyMissingSpaceAfterCommaRule(),
                new PyMissingSpaceAfterColonRule(),
                new PyExtraSpaceBeforeParenRule(),
                new PyExtraSpaceAfterAssignmentRule(),
                new PySpaceAfterDotRule(),
                new PyMultipleSpacesAfterCommaRule(),
                new PyMissingSpaceBeforeAssignmentRule(),
            ],
            'cpp': [
                new CppTabIndentationRule(),
                new CppIndentNot4SpaceRule(),
                new CppSpaceInParenRule(),
                new CppExtraSpaceAroundOperatorRule(),
                new CppExtraSpaceAfterAssignmentRule(),
                new CppSpaceAfterDotRule(),
                new CppSpaceInArrowOperatorRule(),
            ],
            'shell': [
                new ShSpaceAroundPipeRule(),
                new ShTabIndentationRule(),
                new ShSpaceAroundRedirectRule(),
                new ShMultipleSpacesRule(),
            ],
        };
    }

    lint(blocks) {
        for (const block of blocks) {
            if (this.lang_filter && !this.lang_filter.has(block.language)) {
                continue;
            }
            const rules = this.common_rules.concat(this.lang_rules[block.language] || []);
            for (const rule of rules) {
                const issues = rule.check(block.code, block.language);
                // 将代码块内相对行号转换为 Markdown 文档绝对行号
                for (const issue of issues) {
                    issue.line = block.start_line + issue.line - 1;
                }
                block.issues.push(...issues);
            }
            // 按行号排序
            block.issues.sort((a, b) => a.line - b.line || a.column - b.column);
        }
        return blocks;
    }

    toJson(blocks, filepath) {
        let totalIssues = 0;
        for (const b of blocks) {
            totalIssues += b.issues.length;
        }
        const blocksChecked = blocks.filter(
            b => !this.lang_filter || this.lang_filter.has(b.language)
        ).length;
        return {
            file: filepath,
            total_issues: totalIssues,
            blocks_checked: blocksChecked,
            blocks: blocks
                .filter(b => !this.lang_filter || this.lang_filter.has(b.language))
                .map(b => ({
                    language: b.language,
                    block_index: b.block_index,
                    start_line: b.start_line,
                    issue_count: b.issues.length,
                    issues: b.issues.map(iss => ({
                        line: iss.line,
                        column: iss.column,
                        rule: iss.rule,
                        message: iss.message,
                        severity: iss.severity,
                        source: iss.source,
                    })),
                })),
        };
    }
}

// ============================================================
// 辅助函数：递归遍历目录收集所有 .md 文件
// ============================================================

function getMdFiles(dirPath) {
    const mdFiles = [];
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
            mdFiles.push(...getMdFiles(fullPath));
        } else if (entry.isFile() && entry.name.endsWith('.md')) {
            mdFiles.push(path.resolve(fullPath));
        }
    }
    return mdFiles;
}

// ============================================================
// CLI 入口
// ============================================================

function main() {
    const argv = process.argv.slice(2);

    // 第一步：参数校验
    if (argv.length < 2) {
        console.error('错误: 请提供两个参数: <Markdown文件或文件夹路径> <检查结果总JSON文件绝对路径>');
        console.error('用法: node code_lint.js <Markdown文件或文件夹路径> <检查结果总JSON文件绝对路径>');
        process.exit(1);
    }

    const inputPath = argv[0];
    const totalJsonPath = argv[1];

    // 校验输入路径是否存在
    if (!fs.existsSync(inputPath)) {
        console.error(`错误: 路径不存在: ${inputPath}`);
        process.exit(1);
    }

    // 第二步：路径识别与检查
    let mdFiles = [];
    if (fs.statSync(inputPath).isFile()) {
        // 2.1 单文件：直接检查
        mdFiles = [path.resolve(inputPath)];
    } else if (fs.statSync(inputPath).isDirectory()) {
        // 2.2 文件夹：递归提取所有 .md 文件
        mdFiles = getMdFiles(inputPath);
        // 读取 exclude-files.json 排除白名单文件
        const excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
        let excludeList = [];
        try {
            if (fs.existsSync(excludeFilePath)) {
                excludeList = JSON.parse(fs.readFileSync(excludeFilePath, 'utf8'));
            }
        } catch (e) {
            // 无法读取排除列表，跳过排除逻辑
        }
        // 过滤白名单文件
        mdFiles = mdFiles.filter(filePath => {
            const fileName = path.basename(filePath);
            return !excludeList.includes(fileName);
        });
    }

    // 2.3 汇总所有问题到统一集合
    const allResults = [];

    for (const filePath of mdFiles) {
        // 读取文件（统一换行符）
        let content;
        try {
            content = fs.readFileSync(filePath, 'utf-8').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        } catch (e) {
            console.error(`错误: 读取文件失败: ${filePath} - ${e.message}`);
            continue;
        }

        // 解析 Markdown
        const mdParser = new MarkdownParser();
        const blocks = mdParser.parse(content);

        if (blocks.length === 0) {
            continue;
        }

        // 校验
        const linter = new Linter(100, null);
        linter.lint(blocks);

        // 收集问题并转换为新格式
        const absolutePath = path.resolve(filePath);
        const fileName = path.basename(filePath);

        for (const block of blocks) {
            for (const issue of block.issues) {
                allResults.push({
                    rule_group: '通用规则',
                    rule_item: 'punctuation-check',
                    rule_type: '规范性',
                    rule_name: '代码块格式检查',
                    snippet: issue.source,
                    description: issue.message,
                    rule_level: '提示',
                    file_name: fileName,
                    line_number: String(issue.line),
                    location: `vscode://file/${absolutePath}:${issue.line}`,
                    suggestion: issue.message,
                });
            }
        }
    }

    // 第三步：结果合并 — 仅当发现问题时才操作总JSON文件
    if (allResults.length > 0) {
        // 确保总JSON文件的父目录存在
        const parentDir = path.dirname(totalJsonPath);
        if (!fs.existsSync(parentDir)) {
            fs.mkdirSync(parentDir, { recursive: true });
        }

        // 读取总JSON文件：文件存在且非空 → 解析为数组；否则从空数组开始
        let existingResults = [];
        if (fs.existsSync(totalJsonPath)) {
            const fileContent = fs.readFileSync(totalJsonPath, 'utf8').trim();
            if (fileContent) {
                try {
                    existingResults = JSON.parse(fileContent);
                } catch (e) {
                    existingResults = [];
                }
            }
        }

        // 合并：追加本次检查结果
        const merged = [...existingResults, ...allResults];
        fs.writeFileSync(totalJsonPath, JSON.stringify(merged, null, 2), 'utf-8');

        console.error(`校验完成，发现 ${allResults.length} 个格式问题，结果已追加到 ${totalJsonPath}`);
        process.exit(1);
    } else {
        console.error('校验完成，未发现格式问题');
        process.exit(0);
    }
}

main();
