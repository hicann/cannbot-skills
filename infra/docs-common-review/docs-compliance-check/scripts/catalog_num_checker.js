// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * CatalogNumChecker - 目录层级检查器
 *
 * 检查 Markdown 文档内容中标题层级是否超过4层：
 * 1. 标题层级不超过4层（即最多 ####）
 * 2. 出现 #####（5级）或 ######（6级）标题为违规
 *
 * 参考：markdown-punctuation-quality-checker/scripts/path_checker.js (PathChecker)
 */
class CatalogNumChecker {
  /**
   * @param {string} inputPath    Markdown 文件/文件夹路径
   * @param {string} totalJsonPath 检查结果总JSON文件绝对路径
   */
  constructor(inputPath, totalJsonPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.errors = [];
    this.excludeList = [];
    this.suggestions = {
      heading_level_exceeded: '文档内目录层级不能超过4层'
    };
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md 4.2.1 执行流程步骤1-8
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
      this.checkCatalogNum(filePath);
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
   * 对单个文件执行标题层级检查
   * 读取文件内容，识别代码段状态（``` 围栏），对非代码段内容逐行执行 checkHeadingLevel
   * @param {string} filePath
   */
  checkCatalogNum(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');

      let inCodeBlock = false;

      lines.forEach((line, lineNum) => {
        // 检测代码段围栏标记（``` 或 ~~~）
        const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);
        if (fenceMatch) {
          inCodeBlock = !inCodeBlock;
          return;
        }

        // 代码段内的 # 内容跳过检查
        if (inCodeBlock) {
          return;
        }

        this.checkHeadingLevel(line, lineNum + 1, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查
    }
  }

  /**
   * 检查标题行层级
   * 检测行是否为 ATX 标题（行首最多3个空格 + # + 空格），统计 # 数量，若超过4则记录问题
   * 对应 delta-spec.md 4.3 标题层级检查规则 + Scenario 1-4, 12-14
   * @param {string} line
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkHeadingLevel(line, lineNum, filePath) {
    // 识别 ATX 标题：行首最多3个空格，然后1-6个#，然后空格或行尾
    // 参考 CommonMark 规范：https://spec.commonmark.org/0.30/#atx-headings
    const headingRe = /^ {0,3}(#{1,6})(\s|$)/;
    const match = line.match(headingRe);
    if (!match) {
      return;
    }

    const hashCount = match[1].length;

    // 层级超过4层（即 ##### 或 ######）为违规
    if (hashCount > 4) {
      this.errors.push({
        type: 'heading_level_exceeded',
        message: `目录层级超过4层，最多允许4级（####）`,
        line: lineNum,
        filePath: filePath,
        headingSnippet: line.trim()
      });
    }
  }

  /**
   * 格式化输出结果
   * 对应 delta-spec.md 4.2.1 输出格式
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
        rule_item: 'StaticCheck_compliance',
        rule_type: '规范性',
        rule_name: '单md层级合法',
        snippet: error.headingSnippet || '',
        description: error.message,
        rule_level: '提示',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查标题层级是否超过4层'
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
      console.log('✅ 目录层级检查通过，没有发现错误！');
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
 * 对应 delta-design.md 6.1.5 API接口设计
 * 调用方式：node catalog_num_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('使用方法: node catalog_num_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node catalog_num_checker.js ./docs ./output/total_result.json');
    console.log('示例: node catalog_num_checker.js ./README.md ./output/total_result.json');
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

  const checker = new CatalogNumChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { CatalogNumChecker };
