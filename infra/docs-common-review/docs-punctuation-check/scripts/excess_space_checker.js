// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * ExcessSpaceChecker - 多余空格检查器
 *
 * 检查 Markdown 文档内容中多余空格的问题：
 * 1. 中文间空格：中文和中文之间不能有空格（如 `中文 中文`）
 * 2. 中英文间空格：中文和英文之间不能有空格，白名单产品名除外（如 `中文 English`）
 * 3. 中文标点左侧空格：中文标点符号左侧不允许有多余空格（如 `我知道 。`）
 * 4. 中文标点右侧空格：中文标点符号右侧不允许有空格（如 `我知道， 你不知道。`）
 * 5. 英文标点右侧空格：英文标点符号右侧空格不超过1个，末尾不允许有多余空格（如 `Static Check,  punctuation`、`Static Check. `）
 *
 * 例外（不检查）：代码段（```围栏代码块）、回显内容、命令行内容（以 `$` 或 `#` 开头的行）、行内代码（反引号包裹的内容）
 *
 * 参考：markdown-punctuation-quality-checker/scripts/miss_space_checker.js (MissSpaceChecker)
 *       markdown-punctuation-quality-checker/scripts/excess_punctuation_checker.js (ExcessPunctuationChecker)
 */
class ExcessSpaceChecker {
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
    this.whitelist = [];
    // 排除文件列表路径，可被测试覆盖以验证异常处理
    this.excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
    // 产品名白名单路径，可被测试覆盖以验证异常处理
    this.whitelistFilePath = path.join(__dirname, '..', 'references', 'product_names_whitelist.json');
    this.suggestions = {
      chinese_chinese_space: '中文之间不应有空格，请删除多余空格',
      chinese_english_space: '中英文之间不应有空格，请删除多余空格',
      chinese_punct_left_space: '中文标点符号左侧不应有空格，请删除多余空格',
      chinese_punct_right_space: '中文标点符号右侧不应有空格，请删除多余空格',
      english_punct_right_space: '英文标点符号右侧空格不超过1个，请删除多余空格',
      english_punct_left_space: '英文标点符号左侧不应有空格，请删除多余空格'
    };
    // 加载产品名白名单（规则2使用，constructor 中加载便于单元测试直接调用方法）
    this.loadWhitelist();
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
      this.checkExcessSpace(filePath);
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
   * 参考：miss_space_checker.js getMdFiles
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
   * 对应 delta-spec.md 4.3 排除文件机制、Scenario 19
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
   * 加载产品名白名单
   * 读取 references/product_names_whitelist.json（3 类 49 条），异常时白名单为空
   * 对应 delta-spec.md 4.3 白名单匹配逻辑、delta-design.md DD-004
   */
  loadWhitelist() {
    try {
      if (fs.existsSync(this.whitelistFilePath)) {
        const data = JSON.parse(fs.readFileSync(this.whitelistFilePath, 'utf8'));
        this.whitelist = data.product_names || [];
      }
    } catch (e) {
      // product_names_whitelist.json 不存在或格式错误，白名单为空
      this.whitelist = [];
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
   * 对应 delta-spec.md Scenario 12
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
   * 对应 delta-design.md DD-004、delta-spec.md Scenario 13
   * @param {string} text
   * @returns {string} 剔除行内代码后的文本
   */
  extractInlineCode(text) {
    // 匹配成对的反引号包裹内容（`xxx`），替换为空字符串
    // 未闭合的反引号不匹配，原样返回
    return text.replace(/`[^`]*`/g, '');
  }

  /**
   * 提取问题片段（含上下文）
   * 从匹配位置前后各取10个字符作为上下文
   * @param {string} text 原始行文本
   * @param {Array} match 正则匹配结果（match[0]为完整匹配，match.index为起始位置）
   * @returns {string} 含上下文的问题片段
   */
  getSnippet(text, match) {
    const contextLen = 10;
    const start = Math.max(0, match.index - contextLen);
    const end = Math.min(text.length, match.index + match[0].length + contextLen);
    return text.substring(start, end);
  }

  /**
   * 规则1：中文间空格检测
   * 正则检测中文字符与中文字符之间多余空格的情况（如 `中文 中文`）
   * 对应 delta-spec.md 4.3 规则1、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkChineseChineseSpace(text, lineNum, filePath) {
    // 匹配：中文字符 + 空格+ + 中文字符
    const re = /([\u4e00-\u9fff])(\s+)([\u4e00-\u9fff])/g;
    let match;
    while ((match = re.exec(text)) !== null) {
      this.errors.push({
        type: 'chinese_chinese_space',
        message: '中文之间多余空格',
        line: lineNum,
        filePath: filePath,
        snippet: this.getSnippet(text, match)
      });
    }
  }

  /**
   * 规则2：中英文间空格检测
   * 正则检测中文字符与英文字符之间多余空格的情况（如 `中文 English`）
   * 检测到空格后触发 isWhitelisted 白名单匹配，未命中白名单时报告
   * 对应 delta-spec.md 4.3 规则2、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkChineseEnglishSpace(text, lineNum, filePath) {
    // 注意：正则使用 [a-zA-Z] 而非 [a-zA-Z0-9]，有意排除数字（0-9）。
    // 用户需求：数字和单位之间的空格不校验（如 "100 个"、"使用 100"），
    // 因为数字+空格+中文的模式常见于"数字+单位"场景，属于合理用法。
    // 仅对字母与中文之间的空格进行校验（如 "GB 内存"、"使用 GB"）。

    // 模式1：中文 + 空格+ + 英文（字母，不含数字）
    const re1 = /([\u4e00-\u9fff])(\s+)([a-zA-Z])/g;
    let match;
    while ((match = re1.exec(text)) !== null) {
      const spaceStart = match.index + match[1].length;
      const spaceEnd = spaceStart + match[2].length;
      const beforeText = text.substring(0, spaceStart);
      const afterText = text.substring(spaceEnd);

      if (!this.isWhitelisted(text, spaceStart, beforeText, afterText)) {
        this.errors.push({
          type: 'chinese_english_space',
          message: '中英文之间多余空格',
          line: lineNum,
          filePath: filePath,
          snippet: this.getSnippet(text, match)
        });
      }
    }

    // 模式2：英文（字母，不含数字） + 空格+ + 中文
    const re2 = /([a-zA-Z])(\s+)([\u4e00-\u9fff])/g;
    while ((match = re2.exec(text)) !== null) {
      const spaceStart = match.index + match[1].length;
      const spaceEnd = spaceStart + match[2].length;
      const beforeText = text.substring(0, spaceStart);
      const afterText = text.substring(spaceEnd);

      if (!this.isWhitelisted(text, spaceStart, beforeText, afterText)) {
        this.errors.push({
          type: 'chinese_english_space',
          message: '中英文之间多余空格',
          line: lineNum,
          filePath: filePath,
          snippet: this.getSnippet(text, match)
        });
      }
    }
  }

  /**
   * 规则3：中文标点左侧空格检测
   * 正则检测中文标点符号左侧多余空格的情况（如 `我知道 。`）
   * 对应 delta-spec.md 4.3 规则3、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkChinesePunctuationLeftSpace(text, lineNum, filePath) {
    // 中文标点符号集合
    const cnPunct = '，。；：！？、（）《》【】\u201C\u201D\u2018\u2019';
    // 匹配：非空格字符 + 空格+ + 中文标点（避免匹配行首缩进）
    const re = new RegExp('(\\S)(\\s+)([' + cnPunct + '])', 'g');
    let match;
    while ((match = re.exec(text)) !== null) {
      this.errors.push({
        type: 'chinese_punct_left_space',
        message: '中文标点符号左侧多余空格',
        line: lineNum,
        filePath: filePath,
        snippet: this.getSnippet(text, match)
      });
    }
  }

  /**
   * 规则4：中文标点右侧空格检测
   * 正则检测中文标点符号右侧多余空格的情况（如 `我知道， 你不知道。`）
   * 对应 delta-spec.md 4.3 规则4、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkChinesePunctuationRightSpace(text, lineNum, filePath) {
    const cnPunct = '，。；：！？、（）《》【】\u201C\u201D\u2018\u2019';
    // 匹配：中文标点 + 空格+ + 非空格字符（避免匹配行尾空格被其他规则处理）
    const re = new RegExp('([' + cnPunct + '])(\\s+)(\\S)', 'g');
    let match;
    while ((match = re.exec(text)) !== null) {
      const punctEnd = match.index + 1;
      const beforeText = text.substring(0, punctEnd);

      // 白名单边界豁免：当中文标点是白名单产品名的末尾字符时（如 SoC（TsnsC） 的 ）），
      // 标点后的空格属于产品名与右侧中文之间的空格，不报告（对应 delta-spec.md Scenario 5）
      let isExempted = false;
      for (const productName of this.whitelist) {
        if (beforeText.endsWith(productName)) {
          isExempted = true;
          break;
        }
      }

      if (!isExempted) {
        this.errors.push({
          type: 'chinese_punct_right_space',
          message: '中文标点符号右侧多余空格',
          line: lineNum,
          filePath: filePath,
          snippet: this.getSnippet(text, match)
        });
      }
    }
  }

  /**
   * 规则5：英文标点右侧空格检测
   * 正则检测英文标点符号右侧空格超过1个的情况（如 `Static Check,  punctuation`）
   * 以及英文标点符号右侧无字符但有多余空格的情况（如 `Static Check. `）
   * 对应 delta-spec.md 4.3 规则5、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkEnglishPunctuationRightSpace(text, lineNum, filePath) {
    const enPunct = ',.;:!?';
    // 情况1：英文标点 + 2个以上空格 + 非空格字符
    const re1 = new RegExp('([' + enPunct + '])(\\s{2,})(\\S)', 'g');
    let match;
    while ((match = re1.exec(text)) !== null) {
      this.errors.push({
        type: 'english_punct_right_space',
        message: '英文标点符号右侧多余空格',
        line: lineNum,
        filePath: filePath,
        snippet: this.getSnippet(text, match)
      });
    }

    // 情况2：英文标点 + 空格+ + 行尾（末尾多余空格）
    const re2 = new RegExp('([' + enPunct + '])(\\s+)$', 'g');
    while ((match = re2.exec(text)) !== null) {
      this.errors.push({
        type: 'english_punct_right_space',
        message: '英文标点符号末尾多余空格',
        line: lineNum,
        filePath: filePath,
        snippet: this.getSnippet(text, match)
      });
    }
  }

  /**
   * 规则6：英文标点左侧空格检测
   * 正则检测英文标点符号左侧多余空格的情况（如 `tool .`）
   * 对应 delta-spec.md 4.3 规则6、delta-design.md DD-004
   * @param {string} text
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkEnglishPunctuationLeftSpace(text, lineNum, filePath) {
    const enPunct = ',.;:!?';
    // 匹配：非空格字符 + 空格+ + 英文标点（避免匹配行首缩进）
    const re = new RegExp('(\\S)(\\s+)([' + enPunct + '])', 'g');
    let match;
    while ((match = re.exec(text)) !== null) {
      this.errors.push({
        type: 'english_punct_left_space',
        message: '英文标点符号左侧多余空格',
        line: lineNum,
        filePath: filePath,
        snippet: this.getSnippet(text, match)
      });
    }
  }

  /**
   * 白名单匹配（仅影响规则2）
   * 3 类白名单匹配逻辑：
   * - 类别1（中英文混排产品名）：产品名内部空格允许
   * - 类别2（纯英文+数字产品名）和类别3（特殊格式产品名）：右侧出现中文时空格允许，左侧出现中文时空格不允许
   * @param {string} text 行全文
   * @param {number} spaceIndex 空格起始索引
   * @param {string} beforeText 空格前的文本
   * @param {string} afterText 空格后的文本
   * @returns {boolean} true=豁免（不报告），false=不豁免（报告）
   */
  isWhitelisted(text, spaceIndex, beforeText, afterText) {
    // 计算空格文本（用于类别2/3右侧中文的1空格限制）
    const spaceText = text.substring(spaceIndex, text.length - afterText.length);

    for (const productName of this.whitelist) {
      const hasCJK = /[\u4e00-\u9fff]/.test(productName);
      const hasChinesePunct = /[，。；：！？、（）《》【】\u201C\u201D\u2018\u2019]/.test(productName);

      let category;
      if (hasCJK) {
        category = 1; // 中英文混排
      } else if (hasChinesePunct) {
        category = 3; // 特殊格式
      } else {
        category = 2; // 纯英文+数字
      }

      if (category === 1) {
        // 类别1：检查空格是否在产品名内部
        let searchStart = 0;
        let idx;
        while ((idx = text.indexOf(productName, searchStart)) !== -1) {
          const productEnd = idx + productName.length;
          if (spaceIndex > idx && spaceIndex < productEnd) {
            return true; // 空格在产品名内部，豁免
          }
          searchStart = idx + 1;
        }
      } else {
        // 类别2/3：右侧出现中文时，恰好1个空格允许，2个及以上不豁免
        // beforeText 以产品名结尾，且 afterText 以中文开头
        if (beforeText.endsWith(productName) && /^[\u4e00-\u9fff]/.test(afterText)) {
          if (spaceText.length === 1) {
            return true; // 右侧中文，1个空格，豁免
          }
          // 2个及以上空格，不豁免（继续检查其他产品名或最终报告）
        }
        // 左侧出现中文时不豁免（不返回 true，继续检查其他产品名）
      }
    }
    return false; // 未命中白名单，不豁免
  }

  /**
   * 对单个文件执行多余空格检查
   * 读取文件内容，识别代码段状态，逐行跳过代码段和回显/命令行，
   * 提取行内代码后对剩余内容执行 6 条多余空格校验规则
   * 对应 delta-design.md 6.1.2 方法说明、delta-spec.md Scenario 20（单文件读取失败跳过）
   * @param {string} filePath
   */
  checkExcessSpace(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split(/\r?\n/);
      const codeBlockFlags = this.isInCodeBlock(lines);

      lines.forEach((line, index) => {
        const lineNum = index + 1;

        // 跳过代码段内行（含围栏标记行）
        if (codeBlockFlags[index]) return;

        // 跳过回显/命令行
        if (this.isEchoOrCommand(line)) return;

        // 提取并跳过行内代码（反引号包裹内容）
        const textToCheck = this.extractInlineCode(line);

        // 执行 6 条多余空格校验规则
        this.checkChineseChineseSpace(textToCheck, lineNum, filePath);
        this.checkChineseEnglishSpace(textToCheck, lineNum, filePath);
        this.checkChinesePunctuationLeftSpace(textToCheck, lineNum, filePath);
        this.checkChinesePunctuationRightSpace(textToCheck, lineNum, filePath);
        this.checkEnglishPunctuationRightSpace(textToCheck, lineNum, filePath);
        this.checkEnglishPunctuationLeftSpace(textToCheck, lineNum, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查其他文件
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构，生成 location 字段
   * 对应 delta-spec.md 4.3 输出格式、delta-design.md DD-005
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
        rule_name: '多余空格',
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
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 多余空格检查通过，没有发现错误！');
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

    const resultFile = this.totalJsonPath;
    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-design.md 6.1.5 API接口设计、delta-spec.md Scenario 16
 * 调用方式：node excess_space_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 步骤1: 参数校验
  if (args.length < 2) {
    console.log('使用方法: node excess_space_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node excess_space_checker.js ./docs ./output/total_result.json');
    console.log('示例: node excess_space_checker.js ./README.md ./output/total_result.json');
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

  // 步骤1: 输入路径必须存在
  if (!fs.existsSync(normalizedInput)) {
    console.log(`错误: 输入路径 ${inputPath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在
  const parentDir = path.dirname(normalizedJsonPath);
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  const checker = new ExcessSpaceChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { ExcessSpaceChecker };
