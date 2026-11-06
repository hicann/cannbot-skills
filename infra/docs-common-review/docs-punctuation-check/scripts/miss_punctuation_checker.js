// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * MissPunctuationChecker - 缺失标点符号检查器
 *
 * 检查 Markdown 文档内容中段落或句子结尾是否缺少结束标点符号：
 * 1. 段落结尾只要有任意标点符号（中文或英文标点均可），即视为合规
 * 2. 只检查"末尾完全没有标点符号"的情况
 * 3. 检查范围：正文段落、HTML表格/UL/OL、Markdown表格/有序列表/无序列表
 * 4. 不检查：代码段（```围栏代码块）内内容、标题（# 开头的行）、表格表头（Markdown表头行、HTML <thead>区域、<th>标签）
 *
 * 参考：markdown-punctuation-quality-checker/scripts/path_checker.js (PathChecker)
 */
class MissPunctuationChecker {
  constructor(inputPath, totalJsonPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.errors = [];
    this.excludeList = [];
    this.fileLinesMap = {};
    this.suggestions = {
      missing_chinese_punctuation: '中文段落结尾应有结束标点符号，请补充',
      missing_english_punctuation: '英文段落结尾应有结束标点符号，请补充'
    };
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md 4.3 执行流程步骤1-6
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
      this.checkPunctuation(filePath);
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
   * 参考：path_checker.js getMdFiles
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
   * 对应 delta-spec.md 4.3 排除文件机制
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
   * 对应 delta-design.md DD-004（``` 围栏标记状态机）
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
   * 判断是否为标题行
   * 检查行是否以 # 开头（Markdown 标题语法，如 #、##、###）
   * 对应 delta-spec.md Scenario 4
   * @param {string} line
   * @returns {boolean}
   */
  isHeading(line) {
    return /^#{1,6}\s/.test(line.trim());
  }

  /**
   * 判断是否为中文段落
   * 检测文本中是否包含 CJK 统一表意文字（\u4e00-\u9fa5）
   * 对应 delta-design.md DD-003（CJK 字符检测）
   * @param {string} text
   * @returns {boolean}
   */
  isChineseParagraph(text) {
    return /[\u4e00-\u9fa5]/.test(text);
  }

  /**
   * 检查文本结尾是否有任意标点符号（中文或英文标点均可）
   * 只要末尾有标点符号即视为合规，只报"完全没有标点符号"的情况
   * 对应 delta-spec.md 4.3 段落标点校验
   * @param {string} text
   * @returns {boolean}
   */
  hasEndingPunctuation(text) {
    const trimmed = text.trim();
    if (trimmed.length === 0) return true;
    const lastChar = trimmed[trimmed.length - 1];
    // 中文标点：。，！？、；：""''（）【】《》…—
    // 英文标点：.,!?;:"'()[]…-
    return /[。，！？、；：""''（）【】《》…—.,!?;:"'()\[\]@#$%^&*_~`|/\\]/.test(lastChar);
  }

  /**
   * 判断是否为结构性行（不应检查内容的行）
   * @param {string} line
   * @returns {boolean}
   */
  isStructuralLine(line) {
    const trimmed = line.trim();
    // 空行
    if (trimmed === '') return true;
    // 代码围栏标记
    if (trimmed.startsWith('```')) return true;
    // Markdown 表格分隔行（|---|---|）
    if (/^\|[\s\-:|]+\|?$/.test(trimmed)) return true;
    // 水平分割线（---、***、___）
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) return true;
    return false;
  }

  /**
   * 从行中提取文本内容
   * 去除 HTML 标签、Markdown 列表标记、表格管道符、引用标记等语法
   * @param {string} line
   * @returns {string}
   */
  extractTextContent(line) {
    let text = line.trim();

    // 去除 HTML 标签
    text = text.replace(/<[^>]+>/g, '');

    // 去除 Markdown 无序列表标记（-、*、+ 后跟空格）
    text = text.replace(/^[-*+]\s+/, '');

    // 去除 Markdown 有序列表标记（数字. 后跟空格）
    text = text.replace(/^\d+\.\s+/, '');

    // 去除 Markdown 引用标记（>）
    text = text.replace(/^>\s*/, '');

    return text.trim();
  }

  /**
   * 提取问题片段（含上下文）
   * 从原始行中查找被检查文本的位置，取其前后各10个字符作为上下文
   * 参考：excess_space_checker.js getSnippet
   * @param {string} line 原始行文本
   * @param {string} text 被检查的文本（已提取清洗后的段落/单元格内容）
   * @returns {string} 含上下文的问题片段
   */
  getSnippet(line, text) {
    const contextLen = 10;
    const pos = line.indexOf(text);
    if (pos === -1) {
      return text; // 文本在原始行中未找到（理论上不应发生），回退为完整文本
    }
    const start = Math.max(0, pos - contextLen);
    const end = Math.min(line.length, pos + text.length + contextLen);
    return line.substring(start, end);
  }

  /**
   * 检查段落/句子结尾标点
   * 判断中英文段落，检查结尾是否有任意标点符号，无则记录错误
   * 对应 delta-spec.md 4.3 功能逻辑设计
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   * @param {string} line 原始行文本（用于提取上下文片段）
   */
  checkParagraphPunctuation(text, lineNum, filePath, line) {
    if (!this.hasEndingPunctuation(text)) {
      const type = this.isChineseParagraph(text)
        ? 'missing_chinese_punctuation'
        : 'missing_english_punctuation';
      const message = this.isChineseParagraph(text)
        ? '缺少中文结束标点符号'
        : '缺少英文结束标点符号';
      this.errors.push({
        type: type,
        message: message,
        line: lineNum,
        filePath: filePath,
        snippet: this.getSnippet(line, text)
      });
    }
  }

  /**
   * 从行中提取 HTML 标签内的文本内容
   * 跳过 <th> 标签内容（表头不检查），仅提取 <td>/<li> 等标签内容
   * 注意：<thead> 区域内的 <td> 由调用方（checkPunctuation）通过 inThead 状态跳过
   * @param {string} line
   * @returns {string[]} 提取出的文本片段数组
   */
  extractHtmlCellContent(line) {
    const results = [];
    const trimmed = line.trim();

    // 匹配 <td>...</td> 或 <li>...</li> 标签内的内容（不检查 <th>）
    const cellRe = /<(?:td|li)[^>]*>([\s\S]*?)<\/(?:td|li)>/gi;
    let match;
    while ((match = cellRe.exec(trimmed)) !== null) {
      const text = match[1].trim();
      if (text) {
        results.push(text);
      }
    }

    return results;
  }

  /**
   * 对单个文件执行标点缺失检查
   * 读取文件内容，识别代码段状态，逐行跳过代码段和标题，
   * 对段落/表格/列表内容执行标点检查
   * 表格表头不检查（Markdown表格第一行、HTML <thead>区域内所有单元格、<th>标签）
   * 对应 delta-design.md 6.1.2 方法说明、delta-spec.md 4.3 检查范围
   * @param {string} filePath
   */
  checkPunctuation(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      this.fileLinesMap[filePath] = lines;

      const codeBlockFlags = this.isInCodeBlock(lines);

      // Markdown 表格状态跟踪
      let inMarkdownTable = false;
      let markdownTableHeaderSeen = false;
      // HTML 表格 thead 状态跟踪
      let inThead = false;

      lines.forEach((line, index) => {
        const lineNum = index + 1;

        // 跳过代码段内行
        if (codeBlockFlags[index]) return;

        // 跳过结构性行（空行、代码围栏、水平线）
        if (this.isStructuralLine(line)) {
          // 空行或非表格行结束表格状态
          if (line.trim() === '' || !line.trim().startsWith('|')) {
            inMarkdownTable = false;
            markdownTableHeaderSeen = false;
          } else {
            // 表格分隔行（|---|---|），标记已见表头分隔行
            markdownTableHeaderSeen = true;
          }
          return;
        }

        // 跳过标题行
        if (this.isHeading(line)) return;

        const trimmed = line.trim();

        // Markdown 表格行
        if (trimmed.startsWith('|')) {
          if (!inMarkdownTable) {
            // 第一行是表头，不检查，标记表格开始
            inMarkdownTable = true;
            markdownTableHeaderSeen = false;
            return;
          }
          if (!markdownTableHeaderSeen) {
            // 第二行是分隔行（|---|---|），已被 isStructuralLine 跳过
            // 如果没被跳过（格式不标准），也跳过
            markdownTableHeaderSeen = true;
            return;
          }
          // 数据行：按 | 分割检查每个单元格
          const cells = trimmed.replace(/^\|/, '').replace(/\|$/, '').split('|');
          cells.forEach(cell => {
            const cellText = cell.trim();
            if (cellText) {
              this.checkParagraphPunctuation(cellText, lineNum, filePath, line);
            }
          });
          return;
        }

        // 非 Markdown 表格行，重置表格状态
        inMarkdownTable = false;
        markdownTableHeaderSeen = false;

        // HTML 表格 thead/tbody 状态跟踪
        if (/<thead\b/i.test(trimmed)) {
          inThead = true;
        }
        if (/<\/thead>|<tbody\b/i.test(trimmed)) {
          inThead = false;
        }

        // HTML 表格行：提取 <td>/<li> 内容（跳过 <th> 和 thead 内所有单元格）
        if (/<(?:td|li|th)\b/i.test(trimmed)) {
          if (inThead) {
            // thead 内的所有单元格都不检查（无论 <td> 还是 <th>）
            return;
          }
          const htmlTexts = this.extractHtmlCellContent(line);
          htmlTexts.forEach(text => {
            this.checkParagraphPunctuation(text, lineNum, filePath, line);
          });
          return;
        }

        // 其他内容类型：提取文本内容后检查
        const text = this.extractTextContent(line);
        if (!text) return;
        this.checkParagraphPunctuation(text, lineNum, filePath, line);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构
   * 对应 delta-spec.md 4.3 输出格式、delta-design.md DD-005
   * 参考：path_checker.js formatResults
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
        rule_name: '缺少标点符号',
        snippet: error.snippet || '',
        description: error.message,
        rule_level: '提示',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查标点符号'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md 6.1.7 日志设计
   * 参考：path_checker.js printReport
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 缺失标点符号检查通过，没有发现错误！');
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
 * 对应 delta-design.md 6.1.5 API接口设计、delta-spec.md Scenario 9
 * 调用方式：node miss_punctuation_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('使用方法: node miss_punctuation_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node miss_punctuation_checker.js ./docs ./output/total_result.json');
    console.log('示例: node miss_punctuation_checker.js ./README.md ./output/total_result.json');
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

  const checker = new MissPunctuationChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { MissPunctuationChecker };
