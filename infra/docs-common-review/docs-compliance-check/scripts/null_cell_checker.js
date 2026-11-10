// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * NullCellChecker - 表格空单元格检查器
 *
 * 检查 Markdown 文档中表格单元格的空值问题：
 * 规则1: Markdown 表格按 `|` 分隔的单元格是否为空或仅含空格（跳过表格分隔行）
 * 规则2: HTML 表格 `<td>` 和 `<th>` 元素的文本内容是否为空或仅含空格
 *
 * 不检查代码段（```围栏代码块）内的表格。
 *
 * 参考：markdown-punctuation-quality-checker/scripts/excess_space_checker.js (ExcessSpaceChecker)
 */
class NullCellChecker {
  /**
   * @param {string} inputPath     Markdown 文件/文件夹路径
   * @param {string} totalJsonPath 检查结果总JSON文件绝对路径
   */
  constructor(inputPath, totalJsonPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.errors = [];
    this.excludeList = [];
    // 排除文件列表路径，可被测试覆盖以验证异常处理
    this.excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
    this.suggestions = {
      markdown_empty_cell: '表格单元格不能为空',
      html_empty_cell: '表格单元格不能为空'
    };
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md UC004 基本操作流程 8 步
   * @returns {Array} 检查结果数组
   */
  checkFile() {
    // 步骤3-5: 路径类型识别与文件收集
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

    // 步骤6: 逐文件检查
    this.errors = [];
    mdFiles.forEach(filePath => {
      this.checkNullCell(filePath);
    });

    // 步骤7-8: 结果汇总输出（追加合并到总JSON文件）
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
   * 参考：excess_space_checker.js getMdFiles
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
   * 对应 delta-spec.md Scenario 12
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
   * 对单个文件执行空单元格检查
   * 读取文件内容，识别代码段状态，将代码段内行替换为空行（保持总行数不变），
   * 对非代码段内容执行 Markdown 表格逐行检查和 HTML 表格全文检查
   * 对应 delta-spec.md Scenario 13（单文件读取失败跳过）、Scenario 14（代码段内表格不检查）
   * @param {string} filePath
   */
  checkNullCell(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      let inCodeBlock = false;

      // 构建去除代码段后的完整文本（代码段内行替换为空行，保持行数不变）
      // 用于 HTML 表格全文正则匹配，支持跨行 <td>/<th> 元素
      const processedLines = [];

      lines.forEach((line, index) => {
        const lineNum = index + 1;
        const trimmed = line.trim();

        // 识别代码段状态（``` 围栏标记状态机）
        if (trimmed.startsWith('```')) {
          inCodeBlock = !inCodeBlock;
          processedLines.push(''); // 围栏标记行替换为空行
          return;
        }

        // 跳过代码段内行（Scenario 14）— 替换为空行保持行号一致
        if (inCodeBlock) {
          processedLines.push('');
          return;
        }

        // 对非代码段内容执行 Markdown 表格检查（逐行）
        this.checkMarkdownTable(line, lineNum, filePath);

        // 保留非代码段行用于 HTML 表格全文检查
        processedLines.push(line);
      });

      // 对完整文本执行 HTML 表格检查（支持多行 <td>/<th> 元素）
      const fullText = processedLines.join('\n');
      this.checkHtmlTable(fullText, filePath);
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查其他文件（Scenario 13）
    }
  }

  /**
   * 判断是否为表格分隔行
   * 检测行是否匹配 `|---|---|` 格式的分隔行
   * 对应 delta-spec.md Scenario 3
   * @param {string} line
   * @returns {boolean}
   */
  isTableSeparatorLine(line) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('|')) return false;
    // 分隔行格式：|---|---| 或 |:--|:-:|:--| 等
    // 标准分隔行只含 |、-、:、空格
    const content = trimmed.replace(/[|:\-\s]/g, '');
    return content === '' && trimmed.includes('-') && trimmed.includes('|');
  }

  /**
   * 判断是否为 Markdown 表格行
   * 检测行是否以 `|` 开头且不是分隔行
   * @param {string} line
   * @returns {boolean}
   */
  isMarkdownTableRow(line) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('|')) return false;
    if (this.isTableSeparatorLine(line)) return false;
    return true;
  }

  /**
   * 规则1：Markdown 表格空单元格检查
   * 检测行是否为表格行，跳过分隔行，按 `|` 分隔单元格，
   * 检查每个单元格是否为空或仅含空格
   * 对应 delta-spec.md 规则1、Scenario 1/2/3
   * @param {string} line
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkMarkdownTable(line, lineNum, filePath) {
    if (!this.isMarkdownTableRow(line)) return;

    // 按竖线分隔单元格
    // 行首和行尾的 | 分隔出首尾空字符串，需要正确处理
    const trimmed = line.trim();
    // 去除行首和行尾的 |
    const innerContent = trimmed.replace(/^\|/, '').replace(/\|$/, '');
    const cells = innerContent.split(/(?<!\\)\|/g);

    cells.forEach((cell, index) => {
      const cellTrimmed = cell.trim();
      if (cellTrimmed === '') {
        // 空单元格或仅含空格的单元格
        this.errors.push({
          type: 'markdown_empty_cell',
          message: 'Markdown 表格单元格为空',
          line: lineNum,
          filePath: filePath,
          snippet: line.trim()
        });
      }
    });
  }

  /**
   * 规则2：HTML 表格空 td/th 元素检查
   * 对全文执行正则匹配（支持跨行 <td>/<th> 元素），提取内容后剥离 HTML 标签，
   * 检查文本内容是否为空或仅含空格
   * 对应 delta-spec.md 规则2、Scenario 4/5/6
   * @param {string} text  完整文本（代码段行已替换为空行）
   * @param {string} filePath
   */
  checkHtmlTable(text, filePath) {
    // 匹配 <td>和</td> 之间的内容（[\s\S] 支持跨行匹配）
    const tdRegex = /<td[^>]*>([\s\S]*?)<\/td>/gi;
    const thRegex = /<th[^>]*>([\s\S]*?)<\/th>/gi;

    let match;
    while ((match = tdRegex.exec(text)) !== null) {
      const content = match[1];
      // 剥离嵌套 HTML 标签后检查文本内容是否为空
      const textContent = content.replace(/<[^>]*>/g, '');
      if (textContent.trim() === '') {
        // 通过 match.index 计算行号（统计 match 之前的 \n 数量 + 1）
        const lineNum = text.substring(0, match.index).split('\n').length;
        // snippet 取匹配内容的第一行（避免多行 snippet 过长）
        const snippet = match[0].split('\n')[0];
        this.errors.push({
          type: 'html_empty_cell',
          message: 'HTML 表格 td 元素为空',
          line: lineNum,
          filePath: filePath,
          snippet: snippet
        });
      }
    }

    while ((match = thRegex.exec(text)) !== null) {
      const content = match[1];
      const textContent = content.replace(/<[^>]*>/g, '');
      if (textContent.trim() === '') {
        const lineNum = text.substring(0, match.index).split('\n').length;
        const snippet = match[0].split('\n')[0];
        this.errors.push({
          type: 'html_empty_cell',
          message: 'HTML 表格 th 元素为空',
          line: lineNum,
          filePath: filePath,
          snippet: snippet
        });
      }
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构（11字段），生成 location 字段
   * 对应 delta-spec.md UC004 输出格式、delta-design.md DD-005
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
        rule_item: 'compliance-check',
        rule_type: '规范性',
        rule_name: 'table规范',
        snippet: error.snippet || '',
        description: error.message,
        rule_level: '一般',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: '表格单元格不能为空，请填写内容'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md 6.1.7 日志设计
   */
  printReport() {
    const resultFile = this.totalJsonPath;

    if (this.errors.length === 0) {
      console.log('✅ 表格空单元格检查通过，没有发现错误！');
      console.log(`检查结果已保存到: ${resultFile}`);
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

    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-design.md 6.1.5 API接口设计、delta-spec.md Scenario 9
 * 调用方式：node null_cell_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 步骤1: 参数校验（Scenario 9）
  if (args.length < 2) {
    console.log('使用方法: node null_cell_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node null_cell_checker.js ./docs ./output/total_result.json');
    console.log('示例: node null_cell_checker.js ./README.md ./output/total_result.json');
    process.exit(1);
  }

  const inputPath = args[0];
  const totalJsonPath = args[1];

  // G.NOD.02: 规范化路径，防范目录遍历
  const normalizedInput = path.resolve(path.normalize(inputPath));
  const normalizedTotalJson = path.resolve(path.normalize(totalJsonPath));

  // G.NOD.02: 校验规范化后路径不含 .. 防止目录遍历
  if (normalizedInput.includes('..') || normalizedTotalJson.includes('..')) {
    console.log('错误: 路径包含非法的目录遍历字符 ".."');
    process.exit(1);
  }

  // 步骤1: 输入路径必须存在
  if (!fs.existsSync(normalizedInput)) {
    console.log(`错误: 输入路径 ${inputPath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在
  const parentDir = path.dirname(normalizedTotalJson);
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  const checker = new NullCellChecker(normalizedInput, normalizedTotalJson);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { NullCellChecker };
