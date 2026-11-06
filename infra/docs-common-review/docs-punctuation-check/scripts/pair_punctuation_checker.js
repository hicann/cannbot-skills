// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * PairPunctuationChecker - 成对标点符号完整性检查器
 *
 * 检查 Markdown 文档内容中成对标点符号是否只写了一半：
 * 1. 检查 10 类成对标点符号：() [] （）《》【】 "" "" '' <> {}
 * 2. 栈式匹配：开始符号入栈，结束符号与栈顶匹配
 * 3. 计数法：直双引号 " 每行出现奇数次则不匹配
 * 4. 例外场景：编号格式、围栏标记行、行内代码、取值范围、HTML标签、引用标记、连续恰好三个尖括号
 *
 * 参考：markdown-punctuation-quality-checker/scripts/miss_punctuation_checker.js (MissPunctuationChecker)
 *       markdown-punctuation-quality-checker/scripts/path_checker.js (PathChecker)
 */
class PairPunctuationChecker {
  /**
   * 构造函数：初始化检查器
   * 对应 delta-design.md §6.1.2 constructor 方法说明
   * @param {string} inputPath - 输入路径（文件或文件夹）
   * @param {string} totalJsonPath - 检查结果总JSON文件绝对路径
   */
  constructor(inputPath, totalJsonPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.errors = [];
    this.excludeList = [];
    this.fileLinesMap = {};
    this.suggestions = {
      mismatch_open: '成对标点符号缺少配对的结束符号，请补充',
      mismatch_close: '成对标点符号缺少配对的开始符号，请补充',
      mismatch_straight_quote: '双引号不成对出现，请补充配对的引号'
    };

    // 成对标点符号映射表（开始符号 → 结束符号），9类栈式匹配标点
    this.pairs = {
      '(': ')',           // 英文小括号 (U+0028 / U+0029)
      '[': ']',           // 英文中括号 (U+005B / U+005D)
      '（': '）',         // 中文小括号 (U+FF08 / U+FF09)
      '《': '》',         // 中文书名号 (U+300A / U+300B)
      '【': '】',         // 中文方括号 (U+3010 / U+3011)
      '\u201C': '\u201D', // 弯双引号 (U+201C / U+201D)
      '\u2018': '\u2019', // 弯单引号 (U+2018 / U+2019)
      '<': '>',           // 英文尖括号 (U+003C / U+003E)
      '「': '」',
      '『': '』',
      '〖': '〗',
      '`': '`',
      '{': '}'            // 大括号 (U+007B / U+007D)
    };

    // 开闭符号相同的成对标点（使用计数法）
    this.sameCharPairs = ['"']; // 仅直双引号 (U+0022)
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md UC004 基本操作流程（步骤1~8）
   * 对应 delta-design.md §6.1.2 checkFile 方法说明
   * @returns {Array} 检查结果数组
   */
  checkFile() {
    // 步骤2-3: 路径类型识别与文件收集
    let mdFiles = [];
    const stat = fs.statSync(this.inputPath);
    if (stat.isFile()) {
      mdFiles = [this.inputPath];
    } else if (stat.isDirectory()) {
      mdFiles = this.getMdFiles(this.inputPath);
      this.loadExcludeList();
      // 按文件名排除白名单文件
      mdFiles = mdFiles.filter(filePath => {
        const fileName = path.basename(filePath);
        return !this.excludeList.includes(fileName);
      });
    }

    // 步骤4: 逐文件检查
    this.errors = [];
    this.fileLinesMap = {};
    mdFiles.forEach(filePath => {
      this.checkPairPunctuation(filePath);
    });

    // 步骤5: 结果汇总输出 — 仅当发现问题时合并追加到总JSON文件
    const results = this.formatResults();
    if (results.length > 0) {
      // 确保总JSON文件的父目录存在
      const parentDir = path.dirname(this.totalJsonPath);
      if (!fs.existsSync(parentDir)) {
        fs.mkdirSync(parentDir, { recursive: true });
      }

      // 读取总JSON文件：文件存在且非空 → 解析为数组；否则从空数组开始
      let existingResults = [];
      if (fs.existsSync(this.totalJsonPath)) {
        const fileContent = fs.readFileSync(this.totalJsonPath, 'utf8').trim();
        if (fileContent) {
          try {
            existingResults = JSON.parse(fileContent);
          } catch (e) {
            existingResults = [];
          }
        }
      }

      // 合并：追加本次检查结果
      const merged = [...existingResults, ...results];
      fs.writeFileSync(this.totalJsonPath, JSON.stringify(merged, null, 2), 'utf8');
    }

    return results;
  }

  /**
   * 递归遍历目录收集所有 .md 文件的绝对路径
   * 对应 delta-spec.md UC004 步骤4"文件夹模式"
   * @param {string} dirPath
   * @returns {string[]}
   */
  getMdFiles(dirPath) {
    const mdFiles = [];
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        mdFiles.push(...this.getMdFiles(fullPath));
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        mdFiles.push(path.resolve(fullPath));
      }
    }
    return mdFiles;
  }

  /**
   * 加载排除文件列表
   * 读取 assets/exclude-files.json，异常时跳过排除逻辑
   * 对应 delta-spec.md UC004 排除文件机制
   */
  loadExcludeList() {
    const excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
    try {
      if (fs.existsSync(excludeFilePath)) {
        this.excludeList = JSON.parse(fs.readFileSync(excludeFilePath, 'utf8'));
      }
    } catch (e) {
      // exclude-files.json 不存在或格式错误，跳过排除逻辑
      this.excludeList = [];
    }
  }

  /**
   * 识别代码段状态
   * 遍历所有行，通过 ``` 标记切换代码段状态
   * 对应 delta-design.md DD-005（``` 围栏标记状态机）
   * @param {string[]} lines
   * @returns {boolean[]} 每行是否在代码段内的布尔数组（与 lines 等长）
   */
  isInCodeBlock(lines) {
    const flags = new Array(lines.length).fill(false);
    let inCodeBlock = false;
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (trimmed.startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        flags[i] = true; // 围栏标记行本身属于代码段
        continue;
      }
      flags[i] = inCodeBlock;
    }
    return flags;
  }

  /**
   * 预处理行内容，标记例外场景
   * 依次调用 removeInlineCode、removeHtmlTags、removeBlockquoteMarker、removeRangePatterns
   * 对应 delta-design.md §6.1.2 preprocessLine 方法说明
   * 返回预处理后的行（保留原始行用于 snippet 提取）
   * @param {string} line
   * @returns {string}
   */
  preprocessLine(line) {
    let processed = line;
    processed = this.removeTripleAngleBrackets(processed);
    processed = this.removeInlineCode(processed);
    processed = this.removeHtmlTags(processed);
    processed = this.removeBlockquoteMarker(processed);
    processed = this.removeRangePatterns(processed);
    return processed;
  }

  /**
   * 移除连续恰好三个尖括号
   * 匹配恰好 3 个 < 或 >（前后不含同类尖括号），替换为等长占位符
   * 超过 3 个（4个及以上）的连续尖括号不匹配，仍需检查
   * 对应 delta-spec.md UC004 例外场景"连续恰好三个尖括号"
   * @param {string} line
   * @returns {string}
   */
  removeTripleAngleBrackets(line) {
    return line.replace(/(?<!<)<{3}(?!<)|(?<!>)>{3}(?!>)/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除行内代码段
   * 匹配反引号包裹的内容（`[^`]+`），替换为等长占位符以保持位置对齐
   * 对应 delta-spec.md UC004 例外场景"行内代码"
   * @param {string} line
   * @returns {string}
   */
  removeInlineCode(line) {
    return line.replace(/`[^`]+`/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除 HTML 标签
   * 匹配 <[^<>]+> 格式的 HTML 标签，替换为等长占位符
   * 使用 [^<>] 而非 [^>] 避免尖括号嵌套场景下贪婪匹配消费多余 <
   * 对应 delta-spec.md UC004 例外场景"HTML标签"
   * @param {string} line
   * @returns {string}
   */
  removeHtmlTags(line) {
    return line.replace(/<[^<>]+>/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除行首引用标记
   * 匹配 ^\s*>+ 格式的引用标记，替换为等长占位符
   * 对应 delta-spec.md UC004 例外场景"引用标记"
   * @param {string} line
   * @returns {string}
   */
  removeBlockquoteMarker(line) {
    return line.replace(/^(\s*>+)/, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除取值范围表示法
   * 匹配 [\[\(]\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*[\]\)] 格式的取值范围（支持负数）
   * 替换为等长占位符以保持位置对齐
   * 对应 delta-spec.md UC004 例外场景"取值范围混合括号"
   * @param {string} line
   * @returns {string}
   */
  removeRangePatterns(line) {
    return line.replace(/[\[\(]\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*[\]\)]/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 栈式扫描成对标点符号
   * 对应 delta-spec.md UC004 匹配算法、delta-design.md DD-003
   * @param {string} line - 预处理后的行
   * @param {number} lineNum - 行号
   * @param {string} filePath - 文件路径
   */
  scanPairs(line, lineNum, filePath) {
    // 维护当前行的栈（按行独立扫描，行末残余报告为不匹配）
    const stack = [];
    const openSymbols = Object.keys(this.pairs);
    const closeToOpen = {};
    for (const open in this.pairs) {
      closeToOpen[this.pairs[open]] = open;
    }
    const closeSymbols = Object.keys(closeToOpen);

    // 统计直引号出现次数（直双引号 U+0022、直单引号 U+0027）
    const straightQuoteCounts = {};
    this.sameCharPairs.forEach(c => straightQuoteCounts[c] = 0);

    for (let i = 0; i < line.length; i++) {
      const char = line[i];

      // 直引号计数法（不参与栈式匹配）
      if (this.sameCharPairs.includes(char)) {
        straightQuoteCounts[char]++;
        continue;
      }

      // 开始符号：入栈
      if (openSymbols.includes(char)) {
        stack.push({ symbol: char, position: i, lineNum: lineNum });
        continue;
      }

      // 结束符号：尝试与栈顶匹配
      if (closeSymbols.includes(char)) {
        const expectedOpen = closeToOpen[char];
        if (stack.length > 0) {
          const top = stack[stack.length - 1];
          if (top.symbol === expectedOpen) {
            // 匹配成功：出栈
            stack.pop();
          } else {
            // 匹配失败：检查编号格式例外
            if (!this.isNumberingFormat(line, i, char)) {
              // 不匹配的结束符号
              this.errors.push({
                type: 'mismatch_close',
                symbol: char,
                message: `结束符号 "${char}" 缺少配对的开始符号`,
                line: lineNum,
                filePath: filePath,
                snippet: this.extractSnippet(line, i, char)
              });
            }
            // 如果是编号格式，跳过不报告
          }
        } else {
          // 栈空：检查编号格式例外
          if (!this.isNumberingFormat(line, i, char)) {
            this.errors.push({
              type: 'mismatch_close',
              symbol: char,
              message: `结束符号 "${char}" 缺少配对的开始符号`,
              line: lineNum,
              filePath: filePath,
              snippet: this.extractSnippet(line, i, char)
            });
          }
        }
      }
    }

    // 行末：栈中剩余的开始符号报告为不匹配
    while (stack.length > 0) {
      const item = stack.pop();
      this.errors.push({
        type: 'mismatch_open',
        symbol: item.symbol,
        message: `开始符号 "${item.symbol}" 缺少配对的结束符号`,
        line: lineNum,
        filePath: filePath,
        snippet: this.extractSnippet(line, item.position, item.symbol)
      });
    }

    // 直双引号奇数次：报告不匹配
    if (straightQuoteCounts['"'] % 2 !== 0) {
      // 找到最后一个直双引号的位置
      let lastQuotePos = line.lastIndexOf('"');
      this.errors.push({
        type: 'mismatch_straight_quote',
        symbol: '"',
        message: '双引号没有成对出现，请修改',
        line: lineNum,
        filePath: filePath,
        snippet: this.extractSnippet(line, lastQuotePos, '"')
      });
    }
  }

  /**
   * 判断结束符号是否为编号格式
   * 检查结束符号前一个字符是否为数字，若是则判定为编号格式（返回 true，跳过该结束符号）
   * 对应 delta-spec.md UC004 Scenario 6 编号格式例外
   * @param {string} line - 预处理后的行
   * @param {number} pos - 结束符号位置
   * @param {string} char - 结束符号字符
   * @returns {boolean}
   */
  isNumberingFormat(line, pos, char) {
    if (pos <= 0) return false;
    const prevChar = line[pos - 1];
    return /\d/.test(prevChar);
  }

  /**
   * 提取不匹配标点符号的上下文片段
   * @param {string} line - 原始行
   * @param {number} pos - 不匹配符号位置
   * @param {string} char - 不匹配符号
   * @returns {string}
   */
  extractSnippet(line, pos, char) {
    const contextLen = 10; // 前后各取10个字符
    const start = Math.max(0, pos - contextLen);
    const end = Math.min(line.length, pos + char.length + contextLen);
    return line.substring(start, end);
  }

  /**
   * 对单个文件执行成对标点符号完整性检查
   * 读取文件内容，识别代码段状态，对非代码段行执行预处理和栈式扫描
   * 对应 delta-design.md §6.1.2 checkPairPunctuation 方法说明
   * @param {string} filePath
   */
  checkPairPunctuation(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      this.fileLinesMap[filePath] = lines;

      lines.forEach((line, index) => {
        const lineNum = index + 1;

        // 仅跳过围栏标记行（```），代码段内容也需要检查
        if (lines[index].trim().startsWith('```')) return;

        // 预处理：标记例外场景
        const processedLine = this.preprocessLine(line);

        // 栈式扫描
        this.scanPairs(processedLine, lineNum, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构（11字段）
   * 对应 delta-spec.md UC004 输出格式、delta-design.md DD-006
   * @returns {Array}
   */
  formatResults() {
    return this.errors.map(error => {
      const filePath = error.filePath || this.inputPath;
      const fileName = path.basename(filePath);
      const absolutePath = path.resolve(filePath);
      const location = `vscode://file/${absolutePath}:${error.line}`;

      return {
        rule_group: '通用规则',
        rule_item: 'StaticCheck_punctuation',
        rule_type: '规范性',
        rule_name: '成对标点符号不完整',
        snippet: error.snippet || '',
        description: error.message,
        rule_level: '一般',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查标点符号是否成对'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md §6.1.7 日志设计
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 成对标点符号检查通过，没有发现错误！');
    } else {
      console.log(`❌ 发现 ${this.errors.length} 个错误：\n`);

      const groupedErrors = {};
      this.errors.forEach(error => {
        if (!groupedErrors[error.type]) {
          groupedErrors[error.type] = [];
        }
        groupedErrors[error.type].push(error);
      });

      Object.keys(groupedErrors).forEach(type => {
        console.log(`【${type}】`);
        groupedErrors[type].forEach(error => {
          const fileName = path.basename(error.filePath || this.inputPath);
          console.log(`  - ${fileName} 第 ${error.line} 行: ${error.message}`);
        });
        console.log('');
      });
    }

    const resultFile = this.totalJsonPath;
    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-design.md §6.1.5 API接口设计
 * 调用方式：node pair_punctuation_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 参数校验：校验参数1和参数2是否存在
  if (args.length < 2) {
    console.log('使用方法: node pair_punctuation_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node pair_punctuation_checker.js ./docs ./output/total_result.json');
    console.log('示例: node pair_punctuation_checker.js ./README.md ./output/total_result.json');
    process.exit(1);
  }

  const inputPath = args[0];
  const totalJsonPath = args[1];

  // G.NOD.02: 规范化路径，防范目录遍历
  const normalizedInput = path.resolve(path.normalize(inputPath));
  const normalizedJsonPath = path.resolve(path.normalize(totalJsonPath));

  // 步骤1: 参数校验 - 输入路径必须存在
  if (!fs.existsSync(normalizedInput)) {
    console.log(`错误: 输入路径 ${inputPath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在
  const parentDir = path.dirname(normalizedJsonPath);
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  const checker = new PairPunctuationChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { PairPunctuationChecker };
