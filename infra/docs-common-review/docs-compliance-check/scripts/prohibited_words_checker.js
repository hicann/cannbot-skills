// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

class ProhibitedWordsChecker {
  constructor(mdFilePath, totalJsonPath) {
    this.mdFilePath = path.resolve(mdFilePath);
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.errors = [];
    this.isEmpty = false;

    // 加载 references/prohibited-words.json（相对于脚本所在目录的 ../references/）
    const prohibitedWordsPath = path.join(__dirname, '..', 'references', 'prohibited-words.json');
    if (!fs.existsSync(prohibitedWordsPath)) {
      console.log(`错误: 配置文件 ${prohibitedWordsPath} 不存在`);
      process.exit(1);
    }
    try {
      this.prohibitedWords = JSON.parse(fs.readFileSync(prohibitedWordsPath, 'utf8'));
    } catch (e) {
      console.log(`错误: 无法解析配置文件 ${prohibitedWordsPath}`);
      process.exit(1);
    }

    if (!this.prohibitedWords || this.prohibitedWords.length === 0) {
      this.isEmpty = true;
    }
  }

  /**
   * 递归遍历目录收集所有 .md 文件的绝对路径
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
   * 检查单个文件的敏感词
   * @param {string} filePath
   */
  checkSingleFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    this.matchProhibitedWords(lines, filePath);
  }

  /**
   * 核心匹配逻辑：逐项检查 nonstandardTerm 是否在行中出现
   * @param {string[]} lines
   * @param {string} filePath
   */
  matchProhibitedWords(lines, filePath) {
    for (const item of this.prohibitedWords) {
      const term = item.nonstandardTerm;
      const exceptions = item.exceptions || [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const matchIndex = line.indexOf(term);
        if (matchIndex === -1) {
          continue;
        }

        // 例外判断：行内容包含 exceptions 中任一字符串则跳过
        let isException = false;
        for (const exception of exceptions) {
          if (line.includes(exception)) {
            isException = true;
            break;
          }
        }

        if (isException) {
          continue;
        }

        // 同一 nonstandardTerm 在同一行仅记录一次
        const snippet = this.buildSnippet(line, matchIndex, term);
        const suggestion = this.buildSuggestion(item);

        this.errors.push({
          filePath: filePath,
          line: i + 1,
          snippet: snippet,
          description: item.errorCategory,
          suggestion: suggestion,
          nonstandardTerm: term,
          errorCategory: item.errorCategory
        });
      }
    }
  }

  /**
   * 构建 snippet 字段：nonstandardTerm 前后各取5个字
   * @param {string} line
   * @param {number} matchIndex
   * @param {string} term
   * @returns {string}
   */
  buildSnippet(line, matchIndex, term) {
    const start = Math.max(0, matchIndex - 5);
    const end = Math.min(line.length, matchIndex + term.length + 5);
    return line.substring(start, end);
  }

  /**
   * 构建 suggestion 字段
   * @param {Object} item
   * @returns {string}
   */
  buildSuggestion(item) {
    if (item.errorCategory === '错别字') {
      return item.errorCategory + '：' + item.nonstandardTerm + '。' + item.correctionSuggestion + '应该为：' + item.standardTerm;
    }
    return item.errorCategory + '：' + item.nonstandardTerm + '。' + item.correctionSuggestion;
  }

  /**
   * 主入口方法：识别路径 -> 收集文件 -> 排除白名单 -> 逐一检查 -> 格式化 -> 合并追加到总JSON文件
   * @returns {Array}
   */
  checkFile() {
    // 1. 若 prohibitedWords 为空，直接返回
    if (this.isEmpty) {
      return [];
    }

    // 2. 识别路径类型并收集文件
    let mdFiles = [];
    const stat = fs.statSync(this.mdFilePath);
    if (stat.isFile()) {
      mdFiles = [this.mdFilePath];
    } else if (stat.isDirectory()) {
      mdFiles = this.getMdFiles(this.mdFilePath);
    }

    // 3. 文件夹模式：读取排除列表 + 过滤
    if (stat.isDirectory()) {
      let excludeList = [];
      const excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
      try {
        if (fs.existsSync(excludeFilePath)) {
          excludeList = JSON.parse(fs.readFileSync(excludeFilePath, 'utf8'));
        }
      } catch (e) {
        // 无法读取排除列表，跳过排除逻辑
      }

      mdFiles = mdFiles.filter(filePath => {
        const fileName = path.basename(filePath);
        return !excludeList.includes(fileName);
      });
    }

    // 4. 逐一检查文件
    this.errors = [];
    mdFiles.forEach(filePath => {
      this.checkSingleFile(filePath);
    });

    // 5. 格式化结果
    const results = this.formatResults();

    // 6. 仅当发现问题时，合并追加到总JSON文件
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
   * 将 errors 数组格式化为标准结果格式
   * @returns {Array}
   */
  formatResults() {
    return this.errors.map(error => {
      const absolutePath = path.resolve(error.filePath);
      const fileName = path.basename(error.filePath);
      const location = `vscode://file/${absolutePath}:${error.line}`;
      return {
        rule_group: '通用规则',
        rule_item: 'compliance-check',
        rule_type: '规范性',
        rule_name: '敏感词检查',
        snippet: error.snippet,
        description: error.description,
        rule_level: '提示',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: error.suggestion
      };
    });
  }

  /**
   * 在控制台打印检查报告摘要
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('文档敏感词检查通过，没有发现错误！');
      return;
    }

    console.log(`发现 ${this.errors.length} 个敏感词问题：\n`);

    const groupedErrors = {};
    this.errors.forEach(error => {
      if (!groupedErrors[error.errorCategory]) {
        groupedErrors[error.errorCategory] = [];
      }
      groupedErrors[error.errorCategory].push(error);
    });

    Object.keys(groupedErrors).forEach(category => {
      console.log(`【${category}】`);
      groupedErrors[category].forEach(error => {
        const fileName = path.basename(error.filePath);
        console.log(`  - ${fileName} 第 ${error.line} 行: ${error.nonstandardTerm}`);
      });
      console.log('');
    });

    const resultFile = this.totalJsonPath;
    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

/**
 * main 函数：解析命令行参数并执行检查
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('使用方法: node prohibited_words_checker.js <MD文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node prohibited_words_checker.js ./docs ./output/total_result.json');
    console.log('示例: node prohibited_words_checker.js ./README.md ./output/total_result.json');
    process.exit(1);
  }

  const mdFilePath = args[0];
  const totalJsonPath = args[1];

  if (!fs.existsSync(mdFilePath)) {
    console.log(`错误: 路径 ${mdFilePath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在
  const parentDir = path.dirname(path.resolve(totalJsonPath));
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  const checker = new ProhibitedWordsChecker(mdFilePath, totalJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

module.exports = { ProhibitedWordsChecker, main };

// 当直接运行脚本时执行 main
if (require.main === module) {
  main();
}
