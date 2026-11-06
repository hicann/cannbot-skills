// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * RepeateCharChecker - 重复字/单词检查器
 *
 * 检查 Markdown 文档内容中中文重复字和英文重复词的问题：
 * 1. 中文重复字检测：使用正则 /([^\x01-\x80]+)\1/ 匹配非 ASCII 字符序列连续重复
 * 2. 英文重复词检测：使用正则 /\b([a-zA-Z]+)\s+\1\b/gi 匹配连续重复英文单词（大小写不敏感）
 * 3. 例外场景过滤：排除 "……"、"——" 两个字符串（按 @ 分割，与 Java 代码一致）
 * 4. 代码段（``` 围栏）内容不检查
 *
 * 参考：markdown-punctuation-quality-checker/scripts/excess_punctuation_checker.js (ExcessPunctuationChecker)
 *       markdown-compliance-quality-checker/scripts/tag_checker.js (TagChecker, location 格式)
 */
class RepeateCharChecker {
  /**
   * 构造函数：初始化检查器
   * 对应 delta-design.md DD-001 constructor 方法说明
   * @param {string} inputPath - 输入路径（文件或文件夹）
   * @param {string} totalJsonPath - 检查结果JSON文件绝对路径
   */
  constructor(inputPath, totalJsonPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.errors = [];
    this.excludeList = [];
    this.fileLinesMap = {};
    this.suggestions = {
      repeat_char: '存在重复的字/词，请删除重复的字/词'
    };

    // 中文重复字正则：匹配非 ASCII 字符序列连续重复
    this.chineseRegex = /([^\x01-\x80]+)\1/;

    // 英文重复词正则：大小写不敏感匹配连续重复英文单词
    this.englishRegex = /\b([a-zA-Z]+)\s+\1\b/gi;

    // 例外场景列表：按 @ 分割，与 Java 代码 ignoredWords.split("@") 一致
    this.exceptions = ['……', '——'];
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md UC010 基本操作流程（步骤3~8）
   * 对应 delta-design.md DD-001 checkFile 方法说明
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
      this.checkRepeatChar(filePath);
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
   * 对应 delta-spec.md UC010 步骤4"文件夹模式"
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
   * 对应 delta-spec.md UC010 排除文件机制
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
   * 对应 delta-design.md DD-001（``` 围栏标记状态机）
   * @param {string[]} lines
   * @returns {boolean[]} 每行是否在代码段内的布尔数组（与 lines 等长）
   */
 isInCodeBlock(lines) {
  const flags = new Array(lines.length).fill(false);
  let inCodeBlock = false;
  let fenceChar = ''; // 当前开启围栏的字符：'' 或 '~'     
  let fenceLen = 0;     // 当前开启围栏的长度     
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
     // 围栏识别：行首为 3 个及以上连续的  或 ~
    const fenceMatch = trimmed.match(/^(`{3,}|~{3,})/);
    if (fenceMatch) {
      const fence = fenceMatch[1];
      const char = fence[0];
      const len = fence.length;
        if (inCodeBlock) {
          // 闭合条件：同字符，且长度 >= 开围栏长度（符合 Markdown 规范）
          if (char === fenceChar && len >= fenceLen) {
            inCodeBlock = false;
            fenceChar = '';
            fenceLen = 0;
          }
          // 否则视为代码块内的普通行，不翻转状态
        } else {
          // 开启代码块，记录围栏字符与长度
          inCodeBlock = true;
          fenceChar = char;
          fenceLen = len;
        }
        flags[i] = true; // 围栏标记行本身属于代码段（保留原行为）
        continue;
      }
      flags[i] = inCodeBlock;
    }
    return flags;
  }

  /**
   * 核心检查方法：对非代码段行执行中文重复字和英文重复词正则匹配
   * 对应 delta-spec.md UC010 规则说明、Scenario 1-4
   * 对应 delta-design.md DD-001 scanRepeatChar 方法说明
   * @param {string} line - 行内容
   * @param {number} lineNum - 行号
   * @param {string} filePath - 文件路径
   */
  scanRepeatChar(line, lineNum, filePath) {
    if (!line || line.length === 0) return;

    // 中文重复字检测：创建全局正则用于扫描所有匹配
    const zhRegex = new RegExp(this.chineseRegex.source, 'g');
    let match;
    while ((match = zhRegex.exec(line)) !== null) {
      const matchedText = match[0];
      if (!this.filterExceptions(matchedText)) {
        this.errors.push({
          type: 'repeat_char',
          message: '存在重复的字/词',
          line: lineNum,
          filePath: filePath,
          snippet: matchedText
        });
      }
      // 避免零长度匹配导致死循环
      if (match.index === zhRegex.lastIndex) {
        zhRegex.lastIndex++;
      }
    }

    // 英文重复词检测：重置 lastIndex 后扫描
    this.englishRegex.lastIndex = 0;
    while ((match = this.englishRegex.exec(line)) !== null) {
      const matchedText = match[0];
      if (!this.filterExceptions(matchedText)) {
        this.errors.push({
          type: 'repeat_char',
          message: '存在重复的字/词',
          line: lineNum,
          filePath: filePath,
          snippet: matchedText
        });
      }
      // 避免零长度匹配导致死循环
      if (match.index === this.englishRegex.lastIndex) {
        this.englishRegex.lastIndex++;
      }
    }
  }

  /**
   * 例外场景过滤：检查匹配文本是否在例外列表中
   * 对应 delta-spec.md UC010 规则说明3、Scenario 3-4
   * @param {string} matchedText - 匹配到的文本
   * @returns {boolean} true=例外场景（跳过不报告），false=非例外（报告为问题）
   */
  filterExceptions(matchedText) {
    return this.exceptions.includes(matchedText);
  }

  /**
   * 对单个文件执行重复字/词检查
   * 读取文件内容，识别代码段状态，对非代码段行执行扫描
   * 对应 delta-design.md DD-001 checkRepeatChar 方法说明
   * @param {string} filePath
   */
  checkRepeatChar(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      this.fileLinesMap[filePath] = lines;

      const codeBlockFlags = this.isInCodeBlock(lines);

      lines.forEach((line, index) => {
        const lineNum = index + 1;
        // 跳过代码段内行
        if (codeBlockFlags[index]) return;
        // 对非代码段行执行扫描
        this.scanRepeatChar(line, lineNum, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查其他文件
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构（11字段）
   * 对应 delta-spec.md UC010 输出格式、delta-design.md DD-001 formatResults
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
        rule_item: 'AICheck',
        rule_type: '正确性',
        rule_name: '重复字/单词检查',
        snippet: error.snippet || '',
        description: '存在重复的字/词',
        rule_level: '严重',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '存在重复的字/词，请删除重复的字/词'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md DD-001 日志设计
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 重复字/单词检查通过，没有发现错误！');
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

    console.log('检查结果已保存到: ' + this.totalJsonPath);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-spec.md UC010 执行步骤1-6
 * 对应 delta-design.md DD-001 main 方法说明
 * 调用方式：node repeate_char_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 步骤1: 参数校验 - 校验参数1和参数2是否存在
  if (args.length < 2) {
    console.log('使用方法: node repeate_char_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node repeate_char_checker.js ./docs ./output/total_result.json');
    console.log('示例: node repeate_char_checker.js ./README.md ./output/total_result.json');
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

  // 实例化检查器，执行检查
  const checker = new RepeateCharChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { RepeateCharChecker };
