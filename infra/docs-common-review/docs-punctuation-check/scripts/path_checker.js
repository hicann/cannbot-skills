// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * PathChecker - 路径符号检查器
 *
 * 检查 Markdown 文档内容中路径符号的使用错误：
 * 1. Windows 路径：必须使用转义的双反斜杠（\\）
 * 2. Linux 路径：必须使用正斜杠（/）
 * 3. 非法字符检查：路径中不得包含 <>:"|?*+ 及空格
 *
 * 参考：markdown-compliance-quality-checker/scripts/tag_checker.js (TagChecker)
 */
class PathChecker {
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
    this.fileLinesMap = {};
    this.suggestions = {
      win_separator_error: 'Windows路径应使用转义的双反斜杠（\\），请检查路径分隔符',
      win_separator_duplicate: 'Windows路径反斜杠数量超过两个，请修正为标准的双反斜杠（\\）',
      win_separator_mixed: 'Windows路径中正斜杠和反斜杠混用，请统一使用双反斜杠（\\）',
      linux_separator_error: 'Linux路径应使用正斜杠（/），请检查路径分隔符',
      linux_separator_mixed: 'Linux路径中正斜杠和反斜杠混用，请统一使用正斜杠（/）',
      linux_separator_duplicate: 'Linux路径中存在多个连续正斜杠，请修正为单个正斜杠（/）',
      illegal_chars: '路径中包含非法字符，请移除 <>:"|?*+ 及空格'
    };
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md 4.2.1 执行流程步骤1-6
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
      this.checkPathSymbols(filePath);
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
   * 参考：tag_checker.js getMdFiles
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
   * 识别代码段状态
   * 遍历所有行，通过 ``` 标记切换代码段状态
   * 参考：excess_space_checker.js isInCodeBlock
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
   * 对单个文件执行路径符号检查
   * 读取文件内容，逐行调用 checkWindowsPath/checkLinuxPath/checkIllegalChars
   * 跳过代码块（``` 围栏代码块）内的行
   * @param {string} filePath
   */
  checkPathSymbols(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      this.fileLinesMap[filePath] = lines;
      const codeBlockFlags = this.isInCodeBlock(lines);

      lines.forEach((line, index) => {
        const lineNum = index + 1;
        // 跳过代码段内行（含围栏标记行）
        if (codeBlockFlags[index]) return;

        this.checkWindowsPath(line, lineNum, filePath);
        this.checkLinuxPath(line, lineNum, filePath);
        this.checkIllegalChars(line, lineNum, filePath);
      });
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查
    }
  }

  /**
   * 校验 Windows 路径
   * 对应 delta-spec.md 4.3 Windows路径校验规则 + Scenario 1-4
   * @param {string} line
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkWindowsPath(line, lineNum, filePath) {
    // 正则匹配 Windows 盘符路径（如 C:\、D:\、C:/）
    // 使用负向回顾断言排除 URL 协议（如 https:// 中的 s:/），确保盘符前不是单词字符
    const winPathRe = /(?<!\w)[A-Za-z]:[\\/]/;
    if (!winPathRe.test(line)) {
      return;
    }

    // 提取路径片段（从盘符开始到行尾或空格或中文等非路径字符）
    const driveMatch = line.match(/(?<!\w)[A-Za-z]:[\\/].*/);
    if (!driveMatch) {
      return;
    }
    let pathStr = driveMatch[0];
    // 截断到第一个中文字符或CJK标点等非路径字符
    const cnIndex = pathStr.search(/[\u3000-\u303f\u4e00-\u9fa5\uff00-\uffef]/);
    if (cnIndex > 0) {
      pathStr = pathStr.substring(0, cnIndex);
    }
    // 截断到空格（路径中不应有空格，空格属于非法字符，由 checkIllegalChars 处理）
    // 但为了路径片段提取，这里截断到空格
    const spaceIndex = pathStr.indexOf(' ');
    if (spaceIndex > 0) {
      pathStr = pathStr.substring(0, spaceIndex);
    }

    const hasBackslash = pathStr.includes('\\');
    const hasForwardSlash = pathStr.includes('/');
    const hasMixed = hasBackslash && hasForwardSlash;

    // 按优先级检查
    if (hasMixed) {
      // S4: 正反斜杠混用 -> 分隔符混用
      this.errors.push({
        type: 'win_separator_mixed',
        message: '分隔符混用',
        line: lineNum,
        filePath: filePath,
        pathSnippet: pathStr
      });
    } else if (hasForwardSlash && !hasBackslash) {
      // S3: 使用正斜杠 -> 分隔符错误
      this.errors.push({
        type: 'win_separator_error',
        message: '分隔符错误',
        line: lineNum,
        filePath: filePath,
        pathSnippet: pathStr
      });
    } else if (hasBackslash) {
      // 检查反斜杠数量
      // 统计连续反斜杠的最大数量
      const backslashSeqs = pathStr.match(/\\+/g) || [];
      const maxBackslashCount = Math.max(...backslashSeqs.map(s => s.length));

      if (maxBackslashCount > 2) {
        // S2: 多于两个反斜杠 -> 分隔符重复
        this.errors.push({
          type: 'win_separator_duplicate',
          message: '分隔符重复',
          line: lineNum,
          filePath: filePath,
          pathSnippet: pathStr
        });
      } else if (maxBackslashCount === 1) {
        // S1: 单反斜杠 -> 分隔符错误
        this.errors.push({
          type: 'win_separator_error',
          message: '分隔符错误',
          line: lineNum,
          filePath: filePath,
          pathSnippet: pathStr
        });
      }
      // maxBackslashCount === 2 是正确的双反斜杠，不报错
    }
  }

  /**
   * 校验 Linux 路径
   * 对应 delta-spec.md 4.3 Linux路径校验规则 + Scenario 5-7
   * @param {string} line
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkLinuxPath(line, lineNum, filePath) {
    // 剥离 URL（如 https://gitcode.com/...）和 HTML 标签（如 </colgroup>）避免误识别为 Linux 路径
    const cleanedLine = line.replace(/\w+:\/\/[^\s]*/g, '').replace(/<[^>]+>/g, '');

    // 如果该行包含 Windows 盘符路径，则跳过 Linux 路径检查（避免误报）
    // 使用负向回顾断言排除 URL 协议（如 https:// 中的 s:/）
    if (/(?<!\w)[A-Za-z]:[\\/]/.test(cleanedLine)) {
      return;
    }

    // 识别 Linux 路径：以 / 开头或包含 /home/、/usr/ 等目录结构
    // 同时识别 \home\ 这种反斜杠形式的错误 Linux 路径
    const linuxPathRe = /(^|[^\w])\/[^\s]*/;
    const linuxBackslashRe = /(^|[^\w])\\(home|usr|etc|var|opt|tmp|root)\\/;

    const isLinuxForward = linuxPathRe.test(cleanedLine);
    const isLinuxBackslash = linuxBackslashRe.test(cleanedLine);

    if (!isLinuxForward && !isLinuxBackslash) {
      return;
    }

    // 提取路径片段
    let pathStr = '';
    if (isLinuxBackslash) {
      // \home\name\file.txt 形式
      const m = cleanedLine.match(/\\(home|usr|etc|var|opt|tmp|root)\\[^\s]*/);
      if (m) {
        pathStr = '\\' + m[0];
      }
    }
    if (!pathStr && isLinuxForward) {
      // /home/name 形式
      const m = cleanedLine.match(/\/[^\s]*/);
      if (m) {
        pathStr = m[0];
        // 截断到中文字符或CJK标点
        const cnIndex = pathStr.search(/[\u3000-\u303f\u4e00-\u9fa5\uff00-\uffef]/);
        if (cnIndex > 0) {
          pathStr = pathStr.substring(0, cnIndex);
        }
      }
    }
    if (!pathStr) {
      return;
    }

    const hasBackslash = pathStr.includes('\\');
    const hasForwardSlash = pathStr.includes('/');
    const hasMixed = hasBackslash && hasForwardSlash;

    // 按优先级检查
    if (hasMixed) {
      // S6: 正反斜杠混用 -> 分隔符混用
      this.errors.push({
        type: 'linux_separator_mixed',
        message: '分隔符混用',
        line: lineNum,
        filePath: filePath,
        pathSnippet: pathStr
      });
    } else if (hasBackslash && !hasForwardSlash) {
      // S5: 使用反斜杠 -> 分隔符错误
      this.errors.push({
        type: 'linux_separator_error',
        message: '分隔符错误',
        line: lineNum,
        filePath: filePath,
        pathSnippet: pathStr
      });
    } else if (hasForwardSlash && !hasBackslash) {
      // 检查多个连续正斜杠
      const slashSeqs = pathStr.match(/\/+/g) || [];
      const maxSlashCount = Math.max(...slashSeqs.map(s => s.length));
      if (maxSlashCount > 1) {
        // S7: 多个连续正斜杠 -> 分隔符重复
        this.errors.push({
          type: 'linux_separator_duplicate',
          message: '分隔符重复',
          line: lineNum,
          filePath: filePath,
          pathSnippet: pathStr
        });
      }
    }
  }

  /**
   * 检查路径中非法字符
   * 对应 delta-spec.md 4.3 非法字符检查 + Scenario 8
   * @param {string} line
   * @param {number} lineNum
   * @param {string} filePath
   */
  checkIllegalChars(line, lineNum, filePath) {
    // 剥离 URL（如 https://gitcode.com/...）和 HTML 标签（如 </colgroup>）避免误识别为路径
    const cleanedLine = line.replace(/\w+:\/\/[^\s]*/g, '').replace(/<[^>]+>/g, '');

    // 识别行中的路径内容（Windows路径或Linux路径）
    // 使用负向回顾断言排除 URL 协议（如 https:// 中的 s:/）
    const winPathRe = /(?<!\w)[A-Za-z]:[\\/][^\s]*/;
    const linuxPathRe = /(^|[^\w])(\/[^\s]*)/;

    const pathsToCheck = [];

    const winMatch = cleanedLine.match(winPathRe);
    if (winMatch) {
      let p = winMatch[0];
      const cnIndex = p.search(/[\u3000-\u303f\u4e00-\u9fa5\uff00-\uffef]/);
      if (cnIndex > 0) {
        p = p.substring(0, cnIndex);
      }
      pathsToCheck.push(p);
    }

    const linuxMatch = cleanedLine.match(linuxPathRe);
    if (linuxMatch && linuxMatch[2]) {
      let p = linuxMatch[2];
      const cnIndex = p.search(/[\u3000-\u303f\u4e00-\u9fa5\uff00-\uffef]/);
      if (cnIndex > 0) {
        p = p.substring(0, cnIndex);
      }
      pathsToCheck.push(p);
    }

    // 同时检查反斜杠形式的 Linux 路径
    const linuxBackslashRe = /\\(home|usr|etc|var|opt|tmp|root)\\[^\s]*/;
    const linuxBsMatch = cleanedLine.match(linuxBackslashRe);
    if (linuxBsMatch) {
      pathsToCheck.push('\\' + linuxBsMatch[0]);
    }

    // 相对路径：不以 / 开头，包含 2+ 个路径分隔符段（3+ 个路径段）
    // 避免误匹配普通文本中的 word/word（如日期 2024/01/01），要求至少 3 个路径段
    const relativePathRe = /(^|[^\w/])([\w.-]+([\\/][\w.*-]+){2,})/;
    const relMatch = cleanedLine.match(relativePathRe);
    if (relMatch && relMatch[2]) {
      let p = relMatch[2];
      const cnIndex = p.search(/[\u3000-\u303f\u4e00-\u9fa5\uff00-\uffef]/);
      if (cnIndex > 0) {
        p = p.substring(0, cnIndex);
      }
      pathsToCheck.push(p);
    }

    // 非法字符正则：<>:"|?*+ 及空格
    // 注意：Windows 盘符冒号（如 C: 中的 :）是合法的，需排除
    const illegalCharRe = /[<>:"|?*+\s]/;

    for (const pathStr of pathsToCheck) {
      // Windows 路径（如 C:\...）跳过盘符冒号，从索引2开始检查
      let checkStr = pathStr;
      const driveMatch = pathStr.match(/^[A-Za-z]:[\\/]/);
      if (driveMatch) {
        checkStr = pathStr.substring(driveMatch[0].length);
      }
      if (illegalCharRe.test(checkStr)) {
        this.errors.push({
          type: 'illegal_chars',
          message: '路径中包含非法字符：<>:"|?*+或空格',
          line: lineNum,
          filePath: filePath,
          pathSnippet: pathStr
        });
        break; // 同一行只记录一次非法字符错误
      }
    }
  }

  /**
   * 格式化输出结果
   * 对应 delta-spec.md 4.2.1 输出格式
   * 参考：tag_checker.js formatResults
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
        rule_name: '路径符号使用错误',
        snippet: error.pathSnippet || '',
        description: error.message,
        rule_level: '提示',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查路径符号是否正确'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md 6.1.7 日志设计
   * 参考：tag_checker.js printReport
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 路径符号检查通过，没有发现错误！');
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
 * 调用方式：node path_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('使用方法: node path_checker.js <Markdown文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node path_checker.js ./docs ./output/total_result.json');
    console.log('示例: node path_checker.js ./README.md ./output/total_result.json');
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

  const checker = new PathChecker(normalizedInput, normalizedJsonPath);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

main();
