// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * MissSpaceChecker - 缺少空格检查器
 *
 * 检查 Markdown 文档内容中缺少空格的问题：
 * 1. 英文标点邻接：英文标点符号（.,;:!?)直接与英文字母相邻且无空格（如 Hello,World）
 * 2. 英文与数字间：英文字母与数字相邻且无空格（如 IP112、12types）
 * 3. 数学符号前后：数学符号（= ≠ ≈ ÷ > < >= ≥ <= ≤）前后缺少空格（如 a=b、x≥y）
 *
 * 例外（不检查）：回显内容、代码段（```围栏代码块）、命令行内容、行内代码（反引号包裹）、HTML 标签（<tag> 形式）、Markdown 链接 URL、裸 URL
 * 文件扩展名白名单：当 . 后为常见文件扩展名（如 sh、cpp、md）时跳过不报告
 *
 * 参考：markdown-punctuation-quality-checker/scripts/miss_punctuation_checker.js (MissPunctuationChecker)
 *       markdown-punctuation-quality-checker/scripts/path_checker.js (PathChecker)
 */
class MissSpaceChecker {
  /**
   * @param {string} inputPath  Markdown 文件/文件夹路径
   * @param {string} outputPath 检查结果总JSON文件路径
   */
  constructor(inputPath, outputPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.outputPath = path.resolve(outputPath);
    this.errors = [];
    this.excludeList = [];
    // 排除文件列表路径，可被测试覆盖以验证异常处理
    this.excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
    this.suggestions = {
      punctuation_adjacency: '英文标点符号与英文字母之间应有一个空格，请补充',
      english_number: '英文与数字之间应有一个空格，请补充',
      math_symbol: '数学符号前后应各有一个空格，请补充'
    };
    // 文件扩展名白名单：当 . 后为常见文件扩展名时跳过不报告
    this.fileExtensions = new Set([
      'md', 'js', 'ts', 'cpp', 'c', 'h', 'py', 'json', 'html', 'css',
      'sh', 'log', 'png', 'jpg', 'jpeg', 'gif', 'svg', 'java', 'xml',
      'yaml', 'yml', 'txt', 'csv', 'bat', 'ps1', 'go', 'rs', 'rb',
      'php', 'sql', 'conf', 'cfg', 'ini', 'env', 'vue', 'jsx', 'tsx',
      'htm', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'exe'
    ]);
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
    mdFiles.forEach(filePath => {
      this.checkSpace(filePath);
    });

    // 步骤6: 结果汇总输出
    const results = this.formatResults();

    // 仅在发现问题时才合并写入总JSON文件
    if (results.length > 0) {
      let existingResults = [];
      try {
        const fileContent = fs.readFileSync(this.outputPath, 'utf8');
        const trimmed = fileContent.trim();
        if (trimmed) {
          existingResults = JSON.parse(trimmed);
        }
      } catch (e) {
        // 文件不存在或内容为空或解析失败，视为空数组
        existingResults = [];
      }
      const mergedResults = existingResults.concat(results);
      fs.writeFileSync(this.outputPath, JSON.stringify(mergedResults, null, 2), 'utf8');
    }

    return results;
  }

  /**
   * 递归遍历目录收集所有 .md 文件的绝对路径
   * 参考：miss_punctuation_checker.js getMdFiles
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
   * 对应 delta-spec.md 4.3 排除文件机制、Scenario 12
   */
  loadExcludeList() {
    try {
      if (fs.existsSync(this.excludeFilePath)) {
        this.excludeList = JSON.parse(fs.readFileSync(this.excludeFilePath, 'utf8'));
      }
    } catch (e) {
      // exclude-files.json 不存在或格式错误，跳过排除逻辑
      this.excludeList = [];
    }
  }

  /**
   * 识别代码段状态
   * 遍历所有行，通过 ``` 标记切换代码段状态
   * 对应 delta-design.md DD-003（``` 围栏标记状态机）
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
   * 判断是否为回显/命令行
   * 检测行是否以 `$` 或 `#` 开头的命令行代码（代码段外的行内命令）
   * 对应 delta-spec.md Scenario 5、Scenario 6
   * @param {string} line
   * @returns {boolean}
   */
  isEchoOrCommand(line) {
    const trimmed = line.trim();
    if (trimmed === '') return false;
    return trimmed.startsWith('$') || trimmed.startsWith('#');
  }

  /**
   * 提取行内代码
   * 提取反引号包裹的行内代码内容并跳过，返回剔除行内代码后的文本
   * 对应 delta-design.md DD-004、delta-spec.md Scenario 6
   * @param {string} text
   * @returns {string} 剔除行内代码后的文本
   */
  extractInlineCode(text) {
    // 匹配成对的反引号包裹内容（`xxx`），替换为空字符串
    // 未闭合的反引号不匹配，原样返回
    return text.replace(/`[^`]*`/g, '');
  }

  /**
   * 移除 HTML 标签
   * 正则匹配 <...> 形式的 HTML 标签（如 <table>、<td>、<tr style="...">）并移除
   * 仅移除成对的尖括号包裹内容，不影响数学符号比较表达式（如 a < b、5 > 3）
   * 对应 delta-spec.md Scenario 16、delta-design.md DD-004
   * @param {string} text
   * @returns {string} 移除 HTML 标签后的文本
   */
  extractHtmlTags(text) {
    return text.replace(/<[^>]+>/g, '');
  }

  /**
   * 移除 Markdown 链接 URL 和裸 URL
   * 正则1：Markdown 链接 [text](url) → 保留链接文本，移除 URL 部分
   * 正则2：裸 URL（http:// 或 https:// 开头）→ 移除
   * 对应 delta-spec.md Scenario 17
   * @param {string} text
   * @returns {string} 移除 URL 后的文本
   */
  extractMarkdownLinks(text) {
    // 正则1：Markdown 链接 [text](url) → 替换为 $1（保留链接文本）
    let result = text.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
    // 正则2：裸 URL → 替换为空
    result = result.replace(/https?:\/\/[^\s)"']+/g, '');
    return result;
  }

  /**
   * 规则1：英文标点邻接检测
   * 正则检测英文标点（. , ; : ! ?）直接与英文字母相邻且无空格的情况（如 Hello,World、Hello.World）
   * 方案B：仅检测标点邻接，不检测驼峰命名边界（myVar 不报告）
   * 对应 delta-spec.md Scenario 1、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkEnglishPunctuationAdjacency(text, lineNum, filePath) {
    // 匹配：英文字母+ + 英文标点 + 英文字母+ （三者直接相邻，无空格）
    // 如 Hello,World → 匹配 "Hello,World"；Hello, World → 不匹配（逗号后有空格）
    const re = /[a-zA-Z]+[.,;:!?][a-zA-Z]+/g;
    let match;
    while ((match = re.exec(text)) !== null) {
      const snippet = match[0];

      // 文件扩展名白名单：当匹配的标点为 . 时，检查 . 后的部分是否为常见文件扩展名
      // 如 env.sh → sh 在白名单中 → 跳过；Hello.World → World 不在白名单中 → 报告
      const dotIdx = snippet.indexOf('.');
      if (dotIdx !== -1) {
        const ext = snippet.substring(dotIdx + 1).toLowerCase();
        if (this.fileExtensions.has(ext)) {
          continue;
        }
      }

      this.errors.push({
        type: 'punctuation_adjacency',
        message: '英文标点符号与英文字母相邻缺少空格',
        line: lineNum,
        filePath: filePath,
        snippet: snippet
      });
    }
  }

  /**
   * 规则2：英文与数字间空格检测
   * 正则检测英文字母与数字相邻且无空格的情况（如 IP112、12types）
   * 对应 delta-spec.md Scenario 2、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkEnglishNumberSpacing(text, lineNum, filePath) {
    // 匹配字母与数字混合的 token（同时包含字母和数字即说明缺少空格）
    // 如 IP112 → 匹配 "IP112"；IP 112 → 分别为 "IP" 和 "112"，不匹配
    const tokenRe = /[a-zA-Z0-9]+/g;
    let match;
    while ((match = tokenRe.exec(text)) !== null) {
      const token = match[0];
      const hasLetter = /[a-zA-Z]/.test(token);
      const hasDigit = /\d/.test(token);
      if (hasLetter && hasDigit) {
        this.errors.push({
          type: 'english_number',
          message: '英文与数字之间缺少空格',
          line: lineNum,
          filePath: filePath,
          snippet: token
        });
      }
    }
  }

  /**
   * 规则3：数学符号前后空格检测
   * 正则检测数学符号（= ≠ ≈ ÷ > < >= ≥ <= ≤）前后缺少空格的情况（如 a=b、x≥y、5>3）
   * 对应 delta-spec.md Scenario 3、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkMathSymbolSpacing(text, lineNum, filePath) {
    // 数学符号（多字符优先匹配：>= 在 > 之前，<= 在 < 之前）
    const symbolRe = /(>=|<=|≥|≤|≠|≈|÷|=|>|<)/g;
    let match;
    while ((match = symbolRe.exec(text)) !== null) {
      const symbol = match[0];
      const startIdx = match.index;
      const endIdx = startIdx + symbol.length;

      // 检查符号前是否有空格（行首视为合规）
      const beforeChar = startIdx > 0 ? text[startIdx - 1] : '';
      const beforeOk = startIdx === 0 || beforeChar === ' ' || beforeChar === '\t';

      // 检查符号后是否有空格（行尾视为合规）
      const afterChar = endIdx < text.length ? text[endIdx] : '';
      const afterOk = endIdx === text.length || afterChar === ' ' || afterChar === '\t';

      if (!beforeOk || !afterOk) {
        // 扩展 snippet 到周围非空格字符（完整表达式片段）
        let snippetStart = startIdx;
        while (snippetStart > 0 && text[snippetStart - 1] !== ' ' && text[snippetStart - 1] !== '\t') {
          snippetStart--;
        }
        let snippetEnd = endIdx;
        while (snippetEnd < text.length && text[snippetEnd] !== ' ' && text[snippetEnd] !== '\t') {
          snippetEnd++;
        }
        const snippet = text.substring(snippetStart, snippetEnd);

        this.errors.push({
          type: 'math_symbol',
          message: '数学符号前后缺少空格',
          line: lineNum,
          filePath: filePath,
          snippet: snippet
        });

        // 跳过已报告的片段，避免相邻符号重复报告
        symbolRe.lastIndex = snippetEnd;
      }
    }
  }

  /**
   * 对单个文件执行空格检查
   * 读取文件内容，识别代码段状态，逐行跳过代码段和回显/命令行，
   * 提取行内代码后对剩余内容执行 3 条空格校验规则
   * 对应 delta-design.md 6.1.2 方法说明、delta-spec.md Scenario 13（单文件读取失败跳过）
   * @param {string} filePath
   */
  checkSpace(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      const codeBlockFlags = this.isInCodeBlock(lines);

      lines.forEach((line, index) => {
        const lineNum = index + 1;

        // 跳过代码段内行（含围栏标记行）
        if (codeBlockFlags[index]) return;

        // 跳过回显/命令行
        if (this.isEchoOrCommand(line)) return;

        // 提取并跳过行内代码（反引号包裹内容），移除 HTML 标签，移除 Markdown 链接 URL 和裸 URL
        const textToCheck = this.extractMarkdownLinks(this.extractHtmlTags(this.extractInlineCode(line)));

        // 执行 3 条空格校验规则
        this.checkEnglishPunctuationAdjacency(textToCheck, lineNum, filePath);
        // this.checkEnglishNumberSpacing(textToCheck, lineNum, filePath);  // 暂时注释：英文与数字间空格不检查
        this.checkMathSymbolSpacing(textToCheck, lineNum, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查其他文件
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构，生成 location 字段
   * 对应 delta-spec.md 4.3 输出格式、delta-design.md DD-005
   * 参考：miss_punctuation_checker.js formatResults
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
        rule_name: '缺少空格',
        snippet: error.snippet || '',
        description: error.message,
        rule_level: '提示',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查空格'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md 6.1.7 日志设计
   * 参考：miss_punctuation_checker.js printReport
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 缺少空格检查通过，没有发现错误！');
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
        console.log(`  - ${fileName} 第 ${error.line} 行: ${error.message} (${error.snippet})`);
      });
      console.log('');
    });

    console.log(`检查结果已保存到: ${this.outputPath}`);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-design.md 6.1.5 API接口设计、delta-spec.md Scenario 9
 * 调用方式：node miss_space_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 步骤1: 参数校验
  if (args.length < 2) {
    console.log('使用方法: node miss_space_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件路径>');
    console.log('示例: node miss_space_checker.js ./docs ./output/total_result.json');
    console.log('示例: node miss_space_checker.js ./README.md ./output/total_result.json');
    process.exit(1);
  }

  const inputPath = args[0];
  const outputPath = args[1];

  // G.NOD.02: 规范化路径，防范目录遍历
  const normalizedInput = path.resolve(path.normalize(inputPath));
  const normalizedOutput = path.resolve(path.normalize(outputPath));

  // G.NOD.02: 校验规范化后路径不含 .. 防止目录遍历
  if (normalizedInput.includes('..') || normalizedOutput.includes('..')) {
    console.log('错误: 路径包含非法的目录遍历字符 ".."');
    process.exit(1);
  }

  // 步骤1: 输入路径必须存在
  if (!fs.existsSync(normalizedInput)) {
    console.log(`错误: 输入路径 ${inputPath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在（Scenario 11）
  const outputDir = path.dirname(normalizedOutput);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const checker = new MissSpaceChecker(normalizedInput, normalizedOutput);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { MissSpaceChecker };
