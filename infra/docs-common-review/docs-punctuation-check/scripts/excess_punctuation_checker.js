// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * ExcessPunctuationChecker - 多余标点符号检查器
 *
 * 检查 Markdown 文档内容中连续出现多个标点符号的问题：
 * 1. 检查连续2个及以上标点符号（中英文标点）视为问题
 * 2. 例外场景（双向）：闭括号前后加标点除外
 * 3. 代码段和行内代码中的标点不检查
 *
 * 参考：markdown-punctuation-quality-checker/scripts/pair_punctuation_checker.js (PairPunctuationChecker)
 *       markdown-compliance-quality-checker/scripts/tag_checker.js (TagChecker, location 格式)
 */
class ExcessPunctuationChecker {
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
      excess_punctuation: '存在连续多余的标点符号，请删除多余的标点符号'
    };

    // 标点符号集合（中英文标点符号合集，28个）
    // 对应 delta-design.md §6.1.2 标点符号集合表
    this.punctuationSet = new Set([
      '。', // U+3002 中文句号
      '.',  // U+002E 英文句号
      '，', // U+FF0C 中文逗号
      ',',  // U+002C 英文逗号
      '；', // U+FF1B 中文分号
      ';',  // U+003B 英文分号
      '：', // U+FF1A 中文冒号
      ':',  // U+003A 英文冒号
      '？', // U+FF1F 中文问号
      '?',  // U+003F 英文问号
      '！', // U+FF01 中文感叹号
      '!',  // U+0021 英文感叹号
      '、', // U+3001 中文顿号
      '~',  // U+007E 波浪号
      '-',  // U+002D 连字符
      '_',  // U+005F 下划线
      '*',  // U+002A 星号
      '#',  // U+0023 井号
      '%',  // U+0025 百分号
      // '$',  // U+0024 美元符号
      // '&',  // U+0026 和号
      '+',  // U+002B 加号
      '=',  // U+003D 等号
      '|',  // U+007C 竖线
      '@',  // U+0040 At符号
      '^'   // U+005E 尖号
    ]);

    // 闭括号集合（11个）
    // 对应 delta-design.md §6.1.2 闭括号集合表
    this.closingBrackets = new Set([
      ')',   // U+0029 英文右小括号
      ']',   // U+005D 英文右中括号
      '）',  // U+FF09 中文右小括号
      '》',  // U+300B 中文右书名号
      '】',  // U+3011 中文右方括号
      '\u201D', // U+201D 弯右双引号
      '"',   // U+0022 直双引号
      '\u2019', // U+2019 弯右单引号
      "'",   // U+0027 直单引号
      '>',   // U+003E 英文右尖括号
      '」',   // 直角引号
      '』',   // 直角引号（双）
      '〗',   // 空心方括号
      '*',   // 星号
      '}'    // U+007D 右大括号
    ]);
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md UC005 基本操作流程（步骤1~8）
   * 对应 delta-design.md §6.1.2 checkFile 方法说明
   * @returns {Array} 检查结果数组
   */
  checkFile() {
    // 步骤3-4: 路径类型识别与文件收集
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

    // 步骤5: 逐文件检查
    this.errors = [];
    this.fileLinesMap = {};
    mdFiles.forEach(filePath => {
      this.checkExcessPunctuation(filePath);
    });

    // 步骤6: 结果汇总输出 — 仅当发现问题时合并追加到总JSON文件
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
   * 对应 delta-spec.md UC005 步骤4"文件夹模式"
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
   * 对应 delta-spec.md UC005 排除文件机制
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
   * 调用 removeInlineCode 移除行内代码
   * 对应 delta-design.md §6.1.2 preprocessLine 方法说明
   * 返回预处理后的行（保留原始行用于 snippet 提取）
   * @param {string} line
   * @returns {string}
   */
  preprocessLine(line) {
    let processed = line;
    processed = this.removeInlineCode(processed);
    processed = this.removeHtmlComments(processed);
    processed = this.removeMarkdownHeadings(processed);
    processed = this.removeMarkdownEmphasis(processed);
    processed = this.removeRelativePaths(processed);
    processed = this.removeCliOptions(processed);
    return processed;
  }

  /**
   * 移除行内代码段
   * 匹配反引号包裹的内容（`[^`]+`），替换为等长占位符以保持位置对齐
   * 对应 delta-spec.md UC005 Scenario 6 行内代码排除
   * @param {string} line
   * @returns {string}
   */
  removeInlineCode(line) {
    return line.replace(/`[^`]+`/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除 HTML 注释标签 <!-- ... -->
   * 非贪婪匹配 <!-- 到 -->，替换为等长空格以保持位置对齐
   * 仅处理单行 HTML 注释（同一行内 <!-- 到 -->）
   * 在 removeInlineCode 之后调用，确保行内代码中的 <!-- 不被误处理
   * @param {string} line
   * @returns {string}
   */
  removeHtmlComments(line) {
    return line.replace(/<!--.*?-->/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除 Markdown ATX 标题行首的 # 标记
   * 匹配行首0-3个空格后跟1-6个 #，且 # 后接空格或行尾（符合 ATX 标题规范）
   * 将 # 部分替换为等长空格，前导空白保留不变
   * 不影响正文中的 #（如 `#*` 仍正常检出）
   * @param {string} line
   * @returns {string}
   */
  removeMarkdownHeadings(line) {
    return line.replace(/^(\s{0,3})(#{1,6})(?=\s|$)/, (match, indent, hashes) => {
      return indent + ' '.repeat(hashes.length);
    });
  }

  /**
   * 移除 Markdown 加粗/斜体标记中的 * 字符
   * 匹配2个及以上连续 *，若 * 串前后不全是空白字符则视为强调标记，替换为等长空格
   * 若 * 串前后都是空白字符则不是强调标记，保留不变
   * 不影响单个 *（如列表项 `* item`、正文 `#*`）
   * @param {string} line
   * @returns {string}
   */
  removeMarkdownEmphasis(line) {
    return line.replace(/\*{2,}/g, (match, offset, fullLine) => {
      const before = offset > 0 ? fullLine[offset - 1] : ' ';
      const after = offset + match.length < fullLine.length ? fullLine[offset + match.length] : ' ';
      // 前后都是空白 → 不是强调标记，保留
      if (/\s/.test(before) && /\s/.test(after)) {
        return match;
      }
      // 否则为强调标记，替换为等长空格
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除相对路径中的 ../ 和 ..\
   * 匹配 .. 后跟 / 或 \ 的情况，替换为等长空格以保持位置对齐
   * @param {string} line
   * @returns {string}
   */
  removeRelativePaths(line) {
    return line.replace(/\.\.(?=[\/\\])/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 移除 CLI 选项中的 -- 前缀
   * 匹配 -- 后跟单词字符的情况（如 --soc-version），替换为等长空格
   * @param {string} line
   * @returns {string}
   */
  removeCliOptions(line) {
    return line.replace(/--(?=\w)/g, (match) => {
      return ' '.repeat(match.length);
    });
  }

  /**
   * 检测表格分隔行
   * 表格分隔行仅由空白、连字符、冒号和竖线组成，且必须包含 | 和 -
   * 如：| --- | --- | --- |
   * @param {string} line
   * @returns {boolean}
   */
  isTableSeparator(line) {
    const trimmed = line.trim();
    if (!trimmed.includes('|') || !trimmed.includes('-')) return false;
    return /^[\s\-:|]+$/.test(trimmed);
  }

  /**
   * 检测 Markdown 水平分割线 / YAML front matter 分隔符
   * 整行仅由同一种字符（-、* 或 _）组成且出现3次及以上，允许空白字符
   * 覆盖：---、------、***、___、- - -（带空格）
   * 不匹配：--（不足3个）、---text（含其他字符）
   * @param {string} line
   * @returns {boolean}
   */
  isHorizontalRule(line) {
    const trimmed = line.trim();
    if (trimmed.length < 3) return false;
    const dashCount = (trimmed.match(/-/g) || []).length;
    const starCount = (trimmed.match(/\*/g) || []).length;
    const underCount = (trimmed.match(/_/g) || []).length;
    if (dashCount >= 3 && /^[\s-]+$/.test(trimmed)) return true;
    if (starCount >= 3 && /^[\s*]+$/.test(trimmed)) return true;
    if (underCount >= 3 && /^[\s_]+$/.test(trimmed)) return true;
    return false;
  }

  /**
   * 识别表格区域行（表头行和内容行，不含分隔行）
   * 两遍扫描：第一遍找到所有分隔行，第二遍标记分隔行上方的表头行和下方的内容行
   * @param {string[]} lines
   * @returns {boolean[]} 每行是否属于表格区域（表头或内容，不含分隔行）
   */
  identifyTableRows(lines) {
    const flags = new Array(lines.length).fill(false);

    // 第一遍：找到所有分隔行
    const separatorIndexes = [];
    for (let i = 0; i < lines.length; i++) {
      if (this.isTableSeparator(lines[i])) {
        separatorIndexes.push(i);
      }
    }

    // 第二遍：标记表头行和内容行
    for (const sepIdx of separatorIndexes) {
      // 标记分隔行上一行为表头行（如果含 | 且未被标记）
      if (sepIdx > 0 && lines[sepIdx - 1].includes('|') && !flags[sepIdx - 1]) {
        flags[sepIdx - 1] = true;
      }
      // 标记分隔行下方连续含 | 的内容行
      let j = sepIdx + 1;
      while (j < lines.length && lines[j].includes('|') && !this.isTableSeparator(lines[j])) {
        flags[j] = true;
        j++;
      }
    }

    return flags;
  }

  /**
   * 逐字符扫描连续标点符号
   * 对应 delta-spec.md UC005 规则说明、Scenario 1-5
   * 对应 delta-design.md DD-003 检测算法、DD-004 例外规则算法
   * @param {string} line - 预处理后的行
   * @param {number} lineNum - 行号
   * @param {string} filePath - 文件路径
   */
  scanExcessPunctuation(line, lineNum, filePath) {
    if (!line || line.length === 0) return;

    let i = 0;
    while (i < line.length) {
      // 跳过非标点字符
      if (!this.punctuationSet.has(line[i])) {
        i++;
        continue;
      }

      // 找到标点起始位置，收集连续标点（跳过标点之间的空白字符）
      const start = i;
      const consecutiveChars = [];
      while (i < line.length) {
        // 当前字符是标点，收集
        if (this.punctuationSet.has(line[i])) {
          consecutiveChars.push(line[i]);
          i++;
          continue;
        }
        // 当前字符是空白，向前看——如果空白后仍是标点，则跳过空白继续收集
        if (/\s/.test(line[i])) {
          let j = i;
          while (j < line.length && /\s/.test(line[j])) {
            j++;
          }
          // 空白后仍是标点，跳过空白继续收集
          if (j < line.length && this.punctuationSet.has(line[j])) {
            i = j;
            continue;
          }
          // 空白后不是标点，停止收集
          break;
        }
        // 非标点非空白，停止收集
        break;
      }

      // 连续标点数量 >= 2 时检查是否包含闭括号
      if (consecutiveChars.length >= 2) {
        // DD-004 双向闭括号检测：若连续标点中任一字符是闭括号，则不报告
        const hasClosingBracket = consecutiveChars.some(ch => this.closingBrackets.has(ch));
        if (!hasClosingBracket) {
          // 记录连续多余标点问题
          // length 参数使用实际跨度（含空白），而非 consecutiveChars.length
          const span = i - start;
          const snippet = this.extractSnippet(line, start, span);
          this.errors.push({
            type: 'excess_punctuation',
            message: '存在连续多余的标点符号',
            line: lineNum,
            filePath: filePath,
            snippet: snippet
          });
        }
      }
    }
  }

  /**
   * 提取连续标点符号的上下文片段
   * @param {string} line - 原始行
   * @param {number} pos - 连续标点起始位置
   * @param {number} length - 连续标点长度
   * @returns {string}
   */
  extractSnippet(line, pos, length) {
    const contextLen = 10; // 前后各取10个字符
    const start = Math.max(0, pos - contextLen);
    const end = Math.min(line.length, pos + length + contextLen);
    return line.substring(start, end);
  }

  /**
   * 对单个文件执行多余标点符号检查
   * 读取文件内容，识别代码段状态，对非代码段行执行预处理和逐字符扫描
   * 对应 delta-design.md §6.1.2 checkExcessPunctuation 方法说明
   * @param {string} filePath
   */
  checkExcessPunctuation(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      this.fileLinesMap[filePath] = lines;

      const codeBlockFlags = this.isInCodeBlock(lines);
      const tableRowFlags = this.identifyTableRows(lines);

      lines.forEach((line, index) => {
        const lineNum = index + 1;

        // 跳过代码段内行
        if (codeBlockFlags[index]) return;

        // 跳过表格分隔行
        if (this.isTableSeparator(line)) return;

        // 跳过水平分割线/YAML分隔符
        if (this.isHorizontalRule(line)) return;

        // 预处理：移除行内代码、相对路径、CLI选项
        let processedLine = this.preprocessLine(line);

        // 表格行（表头或内容行）：将 | 替换为空格
        if (tableRowFlags[index]) {
          processedLine = processedLine.replace(/\|/g, ' ');
        }

        // 逐字符扫描连续标点
        this.scanExcessPunctuation(processedLine, lineNum, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构（11字段）
   * 对应 delta-spec.md UC005 输出格式、delta-design.md DD-007
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
        rule_item: 'punctuation-check',
        rule_type: '规范性',
        rule_name: '多余标点符号',
        snippet: error.snippet || '',
        description: error.message,
        rule_level: '严重',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查标点符号是否连续多余'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md §6.1.7 日志设计
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 多余标点符号检查通过，没有发现错误！');
      return;
    }

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

    const resultFile = this.totalJsonPath;
    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-design.md §6.1.5 API接口设计
 * 调用方式：node excess_punctuation_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 步骤1: 参数校验 - 校验参数1和参数2是否存在
  if (args.length < 2) {
    console.log('使用方法: node excess_punctuation_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node excess_punctuation_checker.js ./docs ./output/total_result.json');
    console.log('示例: node excess_punctuation_checker.js ./README.md ./output/total_result.json');
    process.exit(1);
  }

  const inputPath = args[0];
  const totalJsonPath = args[1];

  // G.NOD.02: 规范化路径，防范目录遍历
  const normalizedInput = path.resolve(path.normalize(inputPath));
  const normalizedJsonPath = path.resolve(path.normalize(totalJsonPath));

  // G.NOD.02: 校验规范化后路径不含 .. 防止目录遍历
  if (normalizedInput.includes('..') || normalizedJsonPath.includes('..')) {
    console.log('错误: 路径包含非法的目录遍历字符 ".."');
    process.exit(1);
  }

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

  const checker = new ExcessPunctuationChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { ExcessPunctuationChecker };
