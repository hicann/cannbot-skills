// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

/**
 * MixedLanguageChecker - 中英文标点混用检查器
 *
 * 检查 Markdown 文档中中英文标点符号混用问题：
 * - 中文为主的文本中使用中文全角标点
 * - 夹在其中的英文片段，其内部标点保持英文半角习惯
 * - 句末标点归整句主语言
 * - 括号判定看内容语言
 * - 数字内部标点一律半角
 *
 * 跳过区域：代码块、行内代码、YAML front matter、HTML 注释/标签
 *
 * 参考：markdown-punctuation-quality-checker/scripts/miss_space_checker.js (MissSpaceChecker)
 *       markdown-punctuation-quality-checker/scripts/path_checker.js (PathChecker)
 */
class MixedLanguageChecker {
  /**
   * @param {string} inputPath  Markdown 文件/文件夹路径
   * @param {string} outputPath 输出结果文件夹路径
   */
  constructor(inputPath, outputPath) {
    // G.NOD.02 安全规范：对外部输入路径进行规范化处理，防范目录遍历
    this.inputPath = path.resolve(inputPath);
    this.outputPath = path.resolve(outputPath);
    this.errors = [];
    this.excludeList = [];
    this.excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');

    // 标点对照表：英文半角 → 中文全角（11 组）
    this.punctuationMap = {
      ',': '，',   // 逗号
      '.': '。',   // 句号（小数点除外）
      ':': '：',   // 冒号（时间冒号除外）
      ';': '；',   // 分号
      '?': '？',   // 问号
      '!': '！',   // 感叹号
      '(': '（',   // 圆括号
      ')': '）',
      '[': '【',   // 方括号
      ']': '】',
      '{': '｛',   // 花括号
      '}': '｝',
      '"': '\u201c\u201d',    // 直引号 → 弯引号（中文段落中）
      "'": '\u2018\u2019',    // 直单引号 → 弯单引号（中文段落中）
      '...': '……', // 省略号
    };

    // 反向对照表：中文全角 → 英文半角（用于检查英文段落中的全角标点）
    // CR-001 修复：将多字符 value（如弯引号""、省略号……）拆成单字符 key
    this.reversePunctuationMap = {};
    for (const [en, cn] of Object.entries(this.punctuationMap)) {
      for (const cnChar of [...cn]) {
        this.reversePunctuationMap[cnChar] = en;
      }
    }

    // 缩写白名单（3.9 从严校验）
    this.abbreviations = [
      'e.g.', 'i.e.', 'vs.', 'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'etc.'
    ];

    this.suggestions = {
      comma: '中文段落中应使用中文全角逗号"，"，请修正',
      period: '中文段落句末应使用中文全角句号"。"，请修正',
      colon: '中文段落中应使用中文全角冒号"："，请修正',
      semicolon: '中文段落中应使用中文全角分号"；"，请修正',
      question: '中文段落中应使用中文全角问号"？"，请修正',
      exclamation: '中文段落中应使用中文全角感叹号"！"，请修正',
      parenthesis: '中文段落中应使用中文全角括号"（）"，请修正',
      bracket: '中文段落中应使用中文全角方括号"【】"，请修正',
      brace: '中文段落中应使用中文全角花括号"｛｝"，请修正',
      quote: '中文段落中应使用中文弯引号\u201c\u201d\u2018\u2019，请修正',
      ellipsis: '中文段落中应使用中文省略号"……"，请修正',
      en_comma: '英文片段中应使用英文半角逗号","，请修正',
      en_period: '英文片段中应使用英文半角句号"."，请修正',
      en_colon: '英文片段中应使用英文半角冒号":"，请修正',
      en_semicolon: '英文片段中应使用英文半角分号";"，请修正',
      en_question: '英文片段中应使用英文半角问号"?"，请修正',
      en_exclamation: '英文片段中应使用英文半角感叹号"!"，请修正',
      en_parenthesis: '英文片段中应使用英文半角括号"()"，请修正',
      en_bracket: '英文片段中应使用英文半角方括号"[]"，请修正',
      en_brace: '英文片段中应使用英文半角花括号"{}"，请修正',
    };
  }

  /**
   * 主入口：执行完整检查流程
   * 对应 delta-spec.md UC007 基本操作流程、delta-design.md 6.1.1 执行流程图
   * @returns {Array} 检查结果数组
   */
  checkFile() {
    // 步骤2: 清理旧结果
    const resultFile = path.join(this.outputPath, 'mix_language_check_result.json');
    if (fs.existsSync(resultFile)) {
      fs.unlinkSync(resultFile);
    }

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
      this.checkFileContent(filePath);
    });

    // 步骤6: 结果汇总输出
    const results = this.formatResults();
    fs.writeFileSync(resultFile, JSON.stringify(results, null, 2), 'utf8');

    return results;
  }

  /**
   * 递归遍历目录收集所有 .md 文件的绝对路径
   * 参考：miss_space_checker.js getMdFiles
   * @param {string} dirPath
   * @returns {Array} .md 文件绝对路径数组
   */
  getMdFiles(dirPath) {
    const mdFiles = [];
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        mdFiles.push(...this.getMdFiles(fullPath));
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        mdFiles.push(fullPath);
      }
    }
    return mdFiles;
  }

  /**
   * 加载排除文件列表
   * 读取 assets/exclude-files.json，try-catch 处理异常
   * 对应 delta-spec.md UC007 规则说明、delta-design.md 异常场景设计
   */
  loadExcludeList() {
    try {
      if (fs.existsSync(this.excludeFilePath)) {
        const content = fs.readFileSync(this.excludeFilePath, 'utf8');
        this.excludeList = JSON.parse(content);
      }
    } catch (e) {
      // exclude-files.json 不存在或格式错误，跳过排除逻辑
      this.excludeList = [];
    }
  }

  /**
   * Markdown 结构解析
   * 逐行状态机解析，标记每行区域类型，返回可检查文本
   * 对应 delta-spec.md 4.3 Markdown 结构区域处理、delta-design.md DD-003
   * @param {Array} lines 文件行数组
   * @returns {Array} 每行解析结果 { lineNum, type, text }
   */
  parseMarkdownStructure(lines) {
    const result = [];
    let inCodeBlock = false;
    let inFrontMatter = false;
    let frontMatterStarted = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineNum = i + 1;
      let type = 'normal';
      let text = line;

      // 代码块状态机（围栏 ``` 标记）
      if (line.trim().startsWith('```')) {
        if (!inCodeBlock) {
          inCodeBlock = true;
          type = 'codeBlock';
          text = '';
        } else {
          inCodeBlock = false;
          type = 'codeBlock';
          text = '';
        }
        result.push({ lineNum, type, text });
        continue;
      }

      // 代码块内行：整体跳过
      if (inCodeBlock) {
        result.push({ lineNum, type: 'codeBlock', text: '' });
        continue;
      }

      // YAML front matter（文件开头 --- ... ---）
      if (i === 0 && line.trim() === '---') {
        inFrontMatter = true;
        frontMatterStarted = true;
        type = 'frontMatter';
        text = '';
        result.push({ lineNum, type, text });
        continue;
      }
      if (inFrontMatter) {
        if (line.trim() === '---' || line.trim() === '...') {
          inFrontMatter = false;
          type = 'frontMatter';
          text = '';
        } else {
          type = 'frontMatter';
          text = '';
        }
        result.push({ lineNum, type, text });
        continue;
      }

      // HTML 注释（<!-- -->）
      const htmlCommentRegex = /<!--[\s\S]*?-->/g;
      text = text.replace(htmlCommentRegex, '');

      // HTML 标签（<br>, <img>, 等）
      const htmlTagRegex = /<\/?[a-zA-Z][^>]*>/g;
      text = text.replace(htmlTagRegex, '');

      // 行内代码（反引号包裹内容）——跳过反引号内内容
      const inlineCodeRegex = /`[^`]+`/g;
      text = text.replace(inlineCodeRegex, '');

      // 图片 alt 文本：![alt](url) —— 保留 alt 文本（CR-002 修复：必须先于 linkRegex）
      const imgRegex = /!\[([^\]]*)\]\([^)]*\)/g;
      text = text.replace(imgRegex, '$1');

      // 链接 URL：[text](url) —— 跳过 URL 部分，保留 text（CR-002 修复：在 imgRegex 之后）
      const linkRegex = /\[([^\]]*)\]\([^)]*\)/g;
      text = text.replace(linkRegex, '$1');

      // 表格结构符（| --- |）—— 跳过结构符行
      if (/^\|[\s-:|]+\|?\s*$/.test(text.trim())) {
        type = 'tableSeparator';
        text = '';
        result.push({ lineNum, type, text });
        continue;
      }

      // 表格单元格内容：剥离 | 分隔符，保留单元格内容
      if (text.trim().startsWith('|') && text.trim().endsWith('|')) {
        text = text.replace(/^\|/, '').replace(/\|$/, '').replace(/\|/g, ' ');
      }

      // 标题标记（#）—— 跳过 #，检查标题文本
      const headingMatch = text.match(/^(#{1,6})\s+(.*)$/);
      if (headingMatch) {
        text = headingMatch[2];
      }

      // 列表标记（-、*、1.）—— 跳过标记符，检查列表项内容
      const unorderedListMatch = text.match(/^(\s*)([-*+])\s+(.*)$/);
      if (unorderedListMatch) {
        text = unorderedListMatch[3];
      }
      const orderedListMatch = text.match(/^(\s*)(\d+)\.\s+(.*)$/);
      if (orderedListMatch) {
        text = orderedListMatch[3];
      }

      // 引用块（>）—— 剥离 > 语法符后检查内容
      const blockquoteMatch = text.match(/^(\s*)>\s?(.*)$/);
      if (blockquoteMatch) {
        text = blockquoteMatch[2];
      }

      result.push({ lineNum, type, text });
    }

    return result;
  }

  /**
   * 段落主语言判定
   * 统计 CJK 字符数与英文字母数，判定段落主语言
   * 对应 delta-design.md DD-004
   * @param {string} text
   * @returns {string} "chinese" | "english" | "mixed"
   */
  detectParagraphLanguage(text) {
    let cjkCount = 0;
    let enCount = 0;

    for (const char of text) {
      const code = char.charCodeAt(0);
      // CJK 统一表意文字范围：U+4E00–U+9FFF
      if (code >= 0x4E00 && code <= 0x9FFF) {
        cjkCount++;
      }
      // 英文字母：a-z, A-Z
      if ((code >= 0x41 && code <= 0x5A) || (code >= 0x61 && code <= 0x7A)) {
        enCount++;
      }
    }

    if (cjkCount > enCount) return 'chinese';
    if (enCount > 0 && cjkCount === 0) return 'english';
    if (cjkCount > 0 && enCount > 0) return 'mixed';
    return 'mixed';
  }

  /**
   * 识别英文片段
   * 在文本中按字符类型切分，识别由英文字母/数字/半角点连成的连续串
   * 对应 delta-spec.md 3.4 中英文边界判定、delta-design.md 6.1.2
   * @param {string} text
   * @returns {Array} 英文片段起止位置数组 { start, end }
   */
  findEnglishFragments(text) {
    const fragments = [];
    // 匹配连续的英文字母、数字、下划线组成的片段（不含标点）
    const regex = /[a-zA-Z0-9_]+/g;
    let match;
    while ((match = regex.exec(text)) !== null) {
      fragments.push({ start: match.index, end: match.index + match[0].length });
    }
    return fragments;
  }

  /**
   * 获取紧邻字符
   * 从标点位置向左/右扫描，跳过其他标点和空格，返回第一个非标点非空格字符
   * @param {string} text
   * @param {number} pos 标点位置
   * @param {string} direction "left" | "right"
   * @returns {string} 紧邻字符（可能为空字符串）
   */
  getAdjacentChar(text, pos, direction) {
    const punctuations = '.,;:?!()[]{}"\'\u2026\u3002\uff0c\uff1b\uff1a\uff1f\uff01\uff08\uff09\u3010\u3011\uff5b\uff5d\u201c\u201d\u2018\u2019\u2026\u2026';
    if (direction === 'left') {
      for (let i = pos - 1; i >= 0; i--) {
        const char = text[i];
        if (char === ' ' || char === '\t') continue;
        if (punctuations.includes(char)) continue;
        return char;
      }
    } else {
      for (let i = pos + 1; i < text.length; i++) {
        const char = text[i];
        if (char === ' ' || char === '\t') continue;
        if (punctuations.includes(char)) continue;
        return char;
      }
    }
    return '';
  }

  /**
   * 判断是否句末标点
   * 判断字符是否为句号/问号/感叹号且位于行尾或句尾
   * 对应 delta-spec.md 裁决规则 2.1
   * @param {string} char
   * @param {string} text
   * @param {number} pos
   * @returns {boolean}
   */
  isSentenceEndPunctuation(char, text, pos) {
    const sentenceEndChars = '。.?？！!';
    if (!sentenceEndChars.includes(char)) return false;
    // 检查是否在行尾（后面只有空格或到行末）
    for (let i = pos + 1; i < text.length; i++) {
      if (text[i] !== ' ' && text[i] !== '\t') return false;
    }
    return true;
  }

  /**
   * 判断标点是否在英文片段内
   * @param {number} pos 标点位置
   * @param {Array} fragments 英文片段数组
   * @returns {boolean}
   */
  isInEnglishFragment(pos, fragments) {
    for (const frag of fragments) {
      // 严格内部检查：标点位置在英文片段的起止范围内（不含边界）
      if (pos >= frag.start && pos < frag.end) return true;
    }
    return false;
  }

  /**
   * 判断是否数字内部标点
   * 标点两侧均为数字 → 半角（小数点、千分位逗号、版本号点）
   * 对应 delta-spec.md 裁决规则 2.4
   * @param {string} leftChar
   * @param {string} rightChar
   * @returns {boolean}
   */
  isNumberInternalPunctuation(leftChar, rightChar) {
    return /\d/.test(leftChar) && /\d/.test(rightChar);
  }

  /**
   * 判断是否时间冒号
   * 正则 \d{1,2}:\d{2} 识别时间格式
   * 对应 delta-spec.md 裁决规则 2.6
   * @param {string} text
   * @param {number} pos 冒号位置
   * @returns {boolean}
   */
  isTimeColon(text, pos) {
    // 向左查找数字
    let leftStart = pos;
    for (let i = pos - 1; i >= 0; i--) {
      if (/\d/.test(text[i])) {
        leftStart = i;
      } else {
        break;
      }
    }
    // 向右查找数字
    let rightEnd = pos;
    for (let i = pos + 1; i < text.length; i++) {
      if (/\d/.test(text[i])) {
        rightEnd = i + 1;
      } else {
        break;
      }
    }
    const leftDigits = text.substring(leftStart, pos);
    const rightDigits = text.substring(pos + 1, rightEnd);
    // 时间格式：1-2位数字:2位数字
    if (leftDigits.length >= 1 && leftDigits.length <= 2 && rightDigits.length === 2) {
      return true;
    }
    return false;
  }

  /**
   * 判断是否缩写点
   * 检查标点位置是否匹配缩写白名单
   * 对应 delta-spec.md 从严校验 3.9
   * @param {string} text
   * @param {number} pos 标点位置
   * @returns {boolean}
   */
  isAbbreviation(text, pos) {
    // 检查当前位置是否在缩写白名单的任意位置内
    for (const abbr of this.abbreviations) {
      for (let start = Math.max(0, pos - abbr.length + 1); start <= pos; start++) {
        if (start + abbr.length > text.length) continue;
        const substr = text.substring(start, start + abbr.length);
        if (substr === abbr && pos >= start && pos < start + abbr.length) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * 判断是否章节号
   * 正则 \d+\.\d+(\.\d+)* 识别章节号（如 3.2.1）
   * 对应 delta-spec.md 从严校验 3.2
   * @param {string} text
   * @param {number} pos 点号位置
   * @returns {boolean}
   */
  isChapterNumber(text, pos) {
    // 向左查找数字
    let leftStart = pos;
    for (let i = pos - 1; i >= 0; i--) {
      if (/\d/.test(text[i])) {
        leftStart = i;
      } else if (text[i] === '.') {
        // 继续向左查找，可能是多级章节号
        let hasDigitBefore = false;
        for (let j = i - 1; j >= 0; j--) {
          if (/\d/.test(text[j])) {
            leftStart = j;
            hasDigitBefore = true;
          } else {
            break;
          }
        }
        if (hasDigitBefore) {
          i = leftStart - 1;
          continue;
        } else {
          break;
        }
      } else {
        break;
      }
    }
    // 向右查找数字
    let rightEnd = pos;
    for (let i = pos + 1; i < text.length; i++) {
      if (/\d/.test(text[i])) {
        rightEnd = i + 1;
      } else if (text[i] === '.') {
        // 继续向右查找
        let hasDigitAfter = false;
        for (let j = i + 1; j < text.length; j++) {
          if (/\d/.test(text[j])) {
            rightEnd = j + 1;
            hasDigitAfter = true;
          } else {
            break;
          }
        }
        if (hasDigitAfter) {
          i = rightEnd - 1;
          continue;
        } else {
          break;
        }
      } else {
        break;
      }
    }
    const leftDigits = text.substring(leftStart, pos);
    const rightDigits = text.substring(pos + 1, rightEnd);
    // 章节号格式：数字.数字（至少一位数字在两侧）
    return leftDigits.length >= 1 && rightDigits.length >= 1;
  }

  /**
   * 判断是否文件扩展名
   * 正则 \.\w+ 识别文件扩展名（如 .json、.md、.js）
   * 对应 delta-spec.md 从严校验 3.2
   * @param {string} text
   * @param {number} pos 点号位置
   * @returns {boolean}
   */
  isFileExtension(text, pos) {
    // 点号后必须是字母（扩展名）
    const after = text[pos + 1];
    if (!after || !/[a-zA-Z]/.test(after)) return false;

    // 查找扩展名结束位置
    let extEnd = pos + 1;
    for (let i = pos + 1; i < text.length; i++) {
      if (/[a-zA-Z]/.test(text[i])) {
        extEnd = i + 1;
      } else {
        break;
      }
    }
    // 扩展名长度 2-5 个字母
    const ext = text.substring(pos + 1, extEnd);
    if (ext.length < 2 || ext.length > 5) return false;

    // 点号前不能是数字（数字的话是章节号或版本号）
    const before = text[pos - 1];
    if (before && /\d/.test(before)) return false;

    // 点号前可以是字母（如 "file.json"）、空格（如 "读取 .json"）、或行首
    return true;
  }

  /**
   * 判断是否列表编号
   * 正则 ^\d+\. 识别有序列表编号（如 1. 2.）
   * 对应 delta-spec.md 裁决规则 2.8
   * @param {string} text
   * @param {number} pos 点号位置
   * @returns {boolean}
   */
  isListItemNumber(text, pos) {
    // 点号前是数字，且数字在行首（前面只有空格）
    let digitStart = pos;
    for (let i = pos - 1; i >= 0; i--) {
      if (/\d/.test(text[i])) {
        digitStart = i;
      } else if (text[i] === ' ' || text[i] === '\t') {
        continue;
      } else {
        return false;
      }
    }
    // 点号后是空格（列表项格式）
    const after = text[pos + 1];
    return after === ' ' || after === '\t';
  }

  /**
   * 标点应属语言判定
   * 按优先级判定标点应属语言
   * 对应 delta-design.md DD-005、delta-spec.md 裁决规则 2.1-2.2
   * @param {string} text
   * @param {number} pos 标点位置
   * @param {string} paragraphLang 段落主语言
   * @param {Array} englishFragments 英文片段数组
   * @returns {string} "chinese" | "english" | "skip"
   */
  judgePunctuationLanguage(text, pos, paragraphLang, englishFragments) {
    const char = text[pos];

    // 1. 启发式跳过检查（这些场景标点一律半角，不报错）
    const leftChar = this.getAdjacentChar(text, pos, 'left');
    const rightChar = this.getAdjacentChar(text, pos, 'right');

    // 1a. 数字内部标点（小数点、千分位、版本号）—— 裁决 2.4
    if (this.isNumberInternalPunctuation(leftChar, rightChar)) {
      return 'skip';
    }

    // 1b. 时间冒号 —— 裁决 2.6
    if (char === ':' && this.isTimeColon(text, pos)) {
      return 'skip';
    }

    // 1c. 缩写点 —— 从严 3.9
    if (char === '.' && this.isAbbreviation(text, pos)) {
      return 'skip';
    }

    // 1d. 章节号 —— 从严 3.2
    if (char === '.' && this.isChapterNumber(text, pos)) {
      return 'skip';
    }

    // 1e. 文件扩展名 —— 从严 3.2
    if (char === '.' && this.isFileExtension(text, pos)) {
      return 'skip';
    }

    // 1f. 列表编号 —— 裁决 2.8
    if (char === '.' && this.isListItemNumber(text, pos)) {
      return 'skip';
    }

    // 2. 句末标点归整句主语言 —— 裁决 2.1
    if (this.isSentenceEndPunctuation(char, text, pos)) {
      return paragraphLang === 'english' ? 'english' : 'chinese';
    }

    // 3. 英文片段边界法 —— 标点在英文片段内部 → english
    if (this.isInEnglishFragment(pos, englishFragments)) {
      return 'english';
    }

    // 4. 紧邻字符法 —— 标点左右第一个非标点字符的语言
    const leftIsCJK = leftChar && /[\u4E00-\u9FFF]/.test(leftChar);
    const rightIsCJK = rightChar && /[\u4E00-\u9FFF]/.test(rightChar);
    const leftIsEN = leftChar && /[a-zA-Z]/.test(leftChar);
    const rightIsEN = rightChar && /[a-zA-Z]/.test(rightChar);

    // 5. 括号判定看内容语言 —— 裁决 2.2
    if (char === '(' || char === '（') {
      // 查看括号内容语言
      let contentEnd = -1;
      const closeChar = char === '(' ? ')' : '）';
      for (let i = pos + 1; i < text.length; i++) {
        if (text[i] === closeChar) {
          contentEnd = i;
          break;
        }
      }
      if (contentEnd > 0) {
        const content = text.substring(pos + 1, contentEnd);
        const contentLang = this.detectParagraphLanguage(content);
        if (contentLang === 'chinese') return 'chinese';
        if (contentLang === 'english') {
          // 英文内容：检查是否包含内部标点（如逗号）→ 技术标记 → 半角
          if (/[,;:]/.test(content)) return 'english';
          // 单个英文词：看周围上下文
          const before = this.getAdjacentChar(text, pos, 'left');
          if (before && /[\u4E00-\u9FFF]/.test(before)) return 'chinese';
          return 'english';
        }
      }
      // 无法确定内容语言时，归段落主语言
      return paragraphLang === 'english' ? 'english' : 'chinese';
    }

    if (char === ')' || char === '）') {
      // 闭括号与对应开括号一致
      const openChar = char === ')' ? '(' : '（';
      let openPos = -1;
      let depth = 0;
      for (let i = pos - 1; i >= 0; i--) {
        if (text[i] === char) depth++;
        if (text[i] === openChar) {
          if (depth === 0) {
            openPos = i;
            break;
          }
          depth--;
        }
      }
      if (openPos >= 0) {
        return this.judgePunctuationLanguage(text, openPos, paragraphLang, englishFragments);
      }
    }

    // 6. 紧邻字符法判定
    // 两侧都是英文 → 英文标点
    if (leftIsEN && rightIsEN) {
      return 'english';
    }
    // 两侧都是中文 → 中文标点
    if (leftIsCJK && rightIsCJK) {
      return 'chinese';
    }
    // 一侧英文一侧中文 → 边界标点，归段落主语言
    if (leftIsEN && rightIsCJK) {
      return paragraphLang === 'english' ? 'english' : 'chinese';
    }
    if (leftIsCJK && rightIsEN) {
      return paragraphLang === 'english' ? 'english' : 'chinese';
    }
    // 单侧有英文
    if (leftIsEN || rightIsEN) {
      return 'english';
    }
    // 单侧有中文
    if (leftIsCJK || rightIsCJK) {
      return 'chinese';
    }

    // 7. 默认归段落主语言
    return paragraphLang === 'english' ? 'english' : 'chinese';
  }

  /**
   * 标点混用检查核心
   * 对文本中每个标点字符执行检查
   * 对应 delta-design.md 6.1.2 checkPunctuation
   * @param {string} text 可检查文本
   * @param {number} lineNum 行号
   * @param {string} filePath 文件路径
   * @param {string} regionType 区域类型
   */
  checkPunctuation(text, lineNum, filePath, regionType) {
    if (!text || text.trim().length === 0) return;

    // 判定段落主语言
    const paragraphLang = this.detectParagraphLanguage(text);
    // 不跳过英文段落：英文段落可能错误使用中文全角标点（反例2），需检查 reversePunctuationMap

    // 识别英文片段
    const englishFragments = this.findEnglishFragments(text);

    // 遍历每个字符检查标点
    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      // 检查半角标点（英文标点出现在中文上下文中）
      if (this.punctuationMap[char]) {
        const shouldLang = this.judgePunctuationLanguage(text, i, paragraphLang, englishFragments);
        if (shouldLang === 'chinese') {
          // 中文上下文中使用了半角标点 → 报错
          const cnChar = this.punctuationMap[char];
          const suggestionKey = this.getSuggestionKey(char);
          const snippet = this.extractSnippet(text, i);

          this.errors.push({
            type: suggestionKey,
            message: `中文段落中使用了英文半角"${char}"，应使用中文全角"${cnChar}"`,
            line: lineNum,
            filePath: filePath,
            snippet: snippet,
            actualChar: char,
            expectedChar: cnChar,
          });
        }
      }

      // 检查全角标点（中文标点出现在英文片段中）
      if (this.reversePunctuationMap[char]) {
        const shouldLang = this.judgePunctuationLanguage(text, i, paragraphLang, englishFragments);
        if (shouldLang === 'english') {
          // 英文上下文中使用了全角标点 → 报错
          const enChar = this.reversePunctuationMap[char];
          const suggestionKey = 'en_' + this.getSuggestionKey(enChar);
          const snippet = this.extractSnippet(text, i);

          this.errors.push({
            type: suggestionKey,
            message: `英文片段中使用了中文全角"${char}"，应使用英文半角"${enChar}"`,
            line: lineNum,
            filePath: filePath,
            snippet: snippet,
            actualChar: char,
            expectedChar: enChar,
          });
        }
      }

      // 检查省略号
      if (char === '.' && text[i + 1] === '.' && text[i + 2] === '.') {
        // 英文省略号 ... 在中文段落中 → 报错
        const shouldLang = this.judgePunctuationLanguage(text, i, paragraphLang, englishFragments);
        if (shouldLang === 'chinese') {
          const snippet = this.extractSnippet(text, i);
          this.errors.push({
            type: 'ellipsis',
            message: '中文段落中应使用中文省略号"……"，不应使用英文"..."',
            line: lineNum,
            filePath: filePath,
            snippet: snippet,
            actualChar: '...',
            expectedChar: '……',
          });
        }
        i += 2; // 跳过后续两个点
      }
    }
  }

  /**
   * 获取建议 key
   * @param {string} char 标点字符
   * @returns {string}
   */
  getSuggestionKey(char) {
    const keyMap = {
      ',': 'comma',
      '.': 'period',
      ':': 'colon',
      ';': 'semicolon',
      '?': 'question',
      '!': 'exclamation',
      '(': 'parenthesis',
      ')': 'parenthesis',
      '[': 'bracket',
      ']': 'bracket',
      '{': 'brace',
      '}': 'brace',
      '"': 'quote',
      "'": 'quote',
    };
    return keyMap[char] || 'punctuation';
  }

  /**
   * 提取问题片段
   * 从标点位置前后各取若干字符作为 snippet
   * @param {string} text
   * @param {number} pos
   * @returns {string}
   */
  extractSnippet(text, pos) {
    const start = Math.max(0, pos - 10);
    const end = Math.min(text.length, pos + 11);
    return text.substring(start, end).trim();
  }

  /**
   * 对单个文件执行检查
   * 读取文件内容，解析 Markdown 结构，对可检查区域执行标点检查
   * 对应 delta-design.md 6.1.2 checkFileContent
   * @param {string} filePath
   */
  checkFileContent(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      const structure = this.parseMarkdownStructure(lines);

      for (const line of structure) {
        // 跳过代码块、front matter、表格结构符
        if (line.type === 'codeBlock' || line.type === 'frontMatter' || line.type === 'tableSeparator') {
          continue;
        }
        // 对可检查文本执行标点检查
        if (line.text && line.text.trim().length > 0) {
          this.checkPunctuation(line.text, line.lineNum, filePath, line.type);
        }
      }
    } catch (e) {
      // 单文件读取失败，跳过该文件继续检查其他文件
    }
  }

  /**
   * 格式化输出结果
   * 将 errors 数组转换为指定的 JSON 结构，生成 location 字段
   * 对应 delta-spec.md 4.3 输出格式、delta-design.md DD-007
   * 参考：miss_space_checker.js formatResults、tag_checker.js:447
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
        rule_name: '中英文符号混用',
        snippet: error.snippet || '',
        description: error.message,
        rule_level: '一般',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查标点使用'
      };
    });
  }

  /**
   * 输出检查报告
   * 对应 delta-design.md 6.1.7 DFX设计-可观测性
   * 参考：miss_space_checker.js printReport
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 中英文标点混用检查通过，没有发现错误！');
      return;
    }

    console.log(`❌ 发现 ${this.errors.length} 个中英文标点混用错误：\n`);

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

    const resultFile = path.join(this.outputPath, 'mix_language_check_result.json');
    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

/**
 * CLI 入口函数
 * 对应 delta-design.md 6.1.5 API接口设计、delta-spec.md UC007 前置条件
 * 调用方式：node mixed_language_checker.js <Markdown文件/文件夹路径> <输出结果文件夹路径>
 */
function main() {
  const args = process.argv.slice(2);

  // 步骤1: 参数校验
  if (args.length < 2) {
    console.log('使用方法: node mixed_language_checker.js <Markdown文件/文件夹路径> <输出结果文件夹路径>');
    console.log('示例: node mixed_language_checker.js ./docs ./output');
    console.log('示例: node mixed_language_checker.js ./README.md ./output');
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

  // 输出文件夹不存在时自动创建
  if (!fs.existsSync(normalizedOutput)) {
    fs.mkdirSync(normalizedOutput, { recursive: true });
  }

  const checker = new MixedLanguageChecker(normalizedInput, normalizedOutput);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { MixedLanguageChecker };
