// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

class TagChecker {
  constructor(configPath, mdFilePath, totalJsonPath, repoName) {
    this.config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    this.mdFilePath = mdFilePath;
    this.totalJsonPath = path.resolve(totalJsonPath);
    this.repoName = repoName;
    this.errors = [];
    this.tags = [];
    this.positionTags = [];
    this.tagStack = [];
    this.lines = [];
    this.fileLinesMap = {};
    this.suggestions = {
      duplicate_id: '请使用唯一的ID标识符，避免在同一文档中重复使用相同的ID',
      unmatched_end: '请确保每个结束标记都有对应的开始标记，检查标记的嵌套结构',
      unmatched_start: '请确保每个开始标记都有对应的结束标记，检查标记是否正确闭合',
      mismatched_id: '请确保开始标记和结束标记的ID序号保持一致',
      duplicate_end: '请删除重复的结束标记，确保每个标记只闭合一次',
      invalid_npu: '请使用配置文件中定义的有效NPU值，参考npus.json配置文件',
      npu_hierarchy: '请确保子标记的npu必须是父标记npu的子集',
      invalid_ref_format: '请确保位置标签格式为 <!-- @ref: 仓名/res/.../xx_res.md#idxxxxx -->',
      invalid_ref_repo: '请确保位置标签中的仓名与文档所在仓一致',
      invalid_ref_path: '请确保位置标签中的路径以/res/开头，不包含相对路径".."，文件名以_res.md结尾',
      invalid_ref_anchor: '请确保位置标签中锚点存在且符合id+数字格式（如id00001、id123）',
      invalid_spacing: '开始标记的严格格式为 <!-- npu="值" idXXX -->，各部分间距规则：<!--与npu之间1个空格，npu与=之间无空格，=与左引号之间无空格，右引号与id之间1个空格，id与-->之间1个空格',
      invalid_end_spacing: '结束标记的严格格式为 <!-- end idXXX -->，各部分间距规则：<!--与end之间1个空格，end与id之间1个空格，id与-->之间1个空格',
      invalid_id_format: '请确保标记的ID为id+数字格式（如id001、id123）'
    };
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

  checkFile() {
    // 识别路径类型并收集文件
    let mdFiles = [];
    if (fs.statSync(this.mdFilePath).isFile()) {
      mdFiles = [path.resolve(this.mdFilePath)];
    } else if (fs.statSync(this.mdFilePath).isDirectory()) {
      mdFiles = this.getMdFiles(this.mdFilePath);
    }

    // 读取排除列表
    let excludeList = [];
    const excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
    try {
      if (fs.existsSync(excludeFilePath)) {
        excludeList = JSON.parse(fs.readFileSync(excludeFilePath, 'utf8'));
      }
    } catch (e) {
      // 无法读取排除列表，跳过排除逻辑
    }

    // 过滤白名单文件
    mdFiles = mdFiles.filter(filePath => {
      const fileName = path.basename(filePath);
      return !excludeList.includes(fileName);
    });

    // 逐一检查文件
    this.errors = [];
    this.fileLinesMap = {};

    mdFiles.forEach(filePath => {
      this.mdFilePath = filePath;
      const content = fs.readFileSync(filePath, 'utf8');
      this.lines = content.split('\n');
      this.fileLinesMap[filePath] = this.lines;

      this.tags = [];
      this.positionTags = [];
      this.tagStack = [];

      const prevErrorCount = this.errors.length;

      this.parseTags();
      this.validateTags();

      // 仅为本次新增的错误附加文件路径
      for (let i = prevErrorCount; i < this.errors.length; i++) {
        this.errors[i].filePath = filePath;
      }
    });

    const results = this.formatResults();

    // 无论是否有问题都落盘：零问题时写入 []（或保持既有内容），
    // 保证"检查成功"与"结果文件存在"一致，聚合方不至将清洁文档误判为 skipped
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

    return results;
  }

  parseTags() {
    const looseStartRe = /<!--\s*npu\s*=\s*"([^"]+)"\s+(id\d+)\s*-->/;
    const strictStartRe = /<!-- npu="([^"]+)" (id\d+) -->/;
    const looseEndRe = /<!--\s*end\s+(id\d+)\s*-->/;
    const strictEndRe = /<!-- end (id\d+) -->/;
    const broadStartRe = /<!--\s*npu\s*=\s*"([^"]+)"\s+(\S+)\s*-->/;
    const broadEndRe = /<!--\s*end\s+(\S+)\s*-->/;

    this.lines.forEach((line, lineNum) => {
      const startMatch = line.match(looseStartRe);
      const endMatch = line.match(looseEndRe);

      if (startMatch) {
        this.tags.push({
          type: 'start',
          id: startMatch[2],
          npu: startMatch[1],
          line: lineNum + 1,
          content: line.trim()
        });
        if (!strictStartRe.test(line)) {
          this.errors.push({
            type: 'invalid_spacing',
            message: `芯片标签格式错误：开始标记间距不符合规范，标签 "${line.trim()}" 间距不正确，正确格式为 <!-- npu="${startMatch[1]}" ${startMatch[2]} -->`,
            line: lineNum + 1
          });
        }
      } else {
        const broadStartMatch = line.match(broadStartRe);
        if (broadStartMatch) {
          this.errors.push({
            type: 'invalid_id_format',
            message: `芯片标签格式错误：开始标记ID格式不正确，ID应为id+数字格式`,
            line: lineNum + 1
          });
        }
      }

      if (endMatch) {
        this.tags.push({
          type: 'end',
          id: endMatch[1],
          line: lineNum + 1,
          content: line.trim()
        });
        if (!strictEndRe.test(line)) {
          this.errors.push({
            type: 'invalid_end_spacing',
            message: `芯片标签格式错误：结束标记间距不符合规范，标签 "${line.trim()}" 间距不正确，正确格式为 <!-- end ${endMatch[1]} -->`,
            line: lineNum + 1
          });
        }
      } else {
        const broadEndMatch = line.match(broadEndRe);
        if (broadEndMatch) {
          this.errors.push({
            type: 'invalid_id_format',
            message: `芯片标签格式错误：结束标记ID格式不正确，ID应为id+数字格式`,
            line: lineNum + 1
          });
        }
      }

      const refMatch = line.match(/<!--\s*@ref:/);
      if (refMatch) {
        const strictMatch = line.match(/<!--\s*@ref:\s*(\S+?)(\/[^\s#]*)#(\S+?) -->/);
        this.positionTags.push({
          line: lineNum + 1,
          content: line.trim(),
          extracted: !!strictMatch,
          repoName: strictMatch ? strictMatch[1] : null,
          refPath: strictMatch ? strictMatch[2] : null,
          anchor: strictMatch ? strictMatch[3] : null
        });
      }
    });
  }

  validateTags() {
    this.checkDuplicateIds();
    this.checkTagMatching();
    this.checkNpuHierarchy();
    this.checkPositionTags();
  }

  checkDuplicateIds() {
    const startTags = this.tags.filter(t => t.type === 'start');
    const idLines = {};

    startTags.forEach(tag => {
      if (!idLines[tag.id]) {
        idLines[tag.id] = [];
      }
      idLines[tag.id].push(tag.line);
    });

    Object.keys(idLines).forEach(id => {
      if (idLines[id].length > 1) {
        const lines = idLines[id].sort((a, b) => b - a);
        const linesStr = lines.map(l => `第${l}行`).join('、');
        this.errors.push({
          type: 'duplicate_id',
          message: `芯片标签格式错误：ID重复，${id} 在${linesStr}重复出现`,
          line: lines[0]
        });
      }
    });
  }

  checkTagMatching() {
    const processedIds = new Map();
    const positionErrorIds = new Set();
    const duplicateEndLines = {};

    this.tags.forEach(tag => {
      if (tag.type === 'start') {
        this.tagStack.push(tag);
      } else if (tag.type === 'end') {
        if (positionErrorIds.has(tag.id)) {
          return;
        }

        if (this.tagStack.length === 0) {
          this.errors.push({
            type: 'unmatched_end',
            message: `芯片标签格式错误：结束标记 ${tag.id} 没有对应的开始标记或开始标记格式错误`,
            line: tag.line
          });
          return;
        }

        const lastStart = this.tagStack[this.tagStack.length - 1];

        if (lastStart.id !== tag.id) {
          const matchIndex = this.tagStack.findIndex(s => s.id === tag.id);
          if (matchIndex !== -1) {
            for (let i = this.tagStack.length - 1; i >= matchIndex; i--) {
              const startTag = this.tagStack[i];
              this.errors.push({
                type: 'unmatched_start',
                message: `芯片标签格式错误：开始标记 ${startTag.id} 对应的结束标记位置错误`,
                line: startTag.line
              });
              positionErrorIds.add(startTag.id);
              this.tagStack.pop();
            }
          } else {
            this.errors.push({
              type: 'unmatched_end',
              message: `芯片标签格式错误：结束标记 ${tag.id} 没有对应的开始标记或开始标记格式错误`,
              line: tag.line
            });
          }
          return;
        }

        if (processedIds.has(tag.id)) {
          if (!duplicateEndLines[tag.id]) {
            duplicateEndLines[tag.id] = [processedIds.get(tag.id)];
          }
          duplicateEndLines[tag.id].push(tag.line);
          this.tagStack.pop();
          return;
        }

        processedIds.set(tag.id, tag.line);
        this.tagStack.pop();
      }
    });

    Object.keys(duplicateEndLines).forEach(id => {
      const lines = duplicateEndLines[id].sort((a, b) => b - a);
      const linesStr = lines.map(l => `第${l}行`).join('、');
      this.errors.push({
        type: 'duplicate_end',
        message: `芯片标签格式错误：ID重复，${id} 在${linesStr}重复出现`,
        line: lines[0]
      });
    });

    this.tagStack.forEach(tag => {
      if (!positionErrorIds.has(tag.id)) {
        this.errors.push({
          type: 'unmatched_start',
          message: `芯片标签格式错误：开始标记 ${tag.id} 没有对应的结束标记或结束标记格式错误`,
          line: tag.line
        });
      }
    });
  }

  checkNpuHierarchy() {
    const tagTree = this.buildTagTree();
    this.validateNpuInTree(tagTree);
  }

  buildTagTree() {
    const root = { children: [], npu: null, id: null };
    const stack = [root];

    this.tags.forEach(tag => {
      if (tag.type === 'start') {
        const node = {
          id: tag.id,
          npu: tag.npu,
          line: tag.line,
          children: []
        };
        stack[stack.length - 1].children.push(node);
        stack.push(node);
      } else if (tag.type === 'end') {
        if (stack.length > 1) {
          stack.pop();
        }
      }
    });

    return root;
  }

  validateNpuInTree(node) {
    if (node.children.length === 0) return;

    const validNpus = this.config.npus;

    node.children.forEach(child => {
      const childNpus = child.npu.split(',').map(n => n.trim());

      childNpus.forEach(npu => {
        if (!validNpus.includes(npu)) {
          this.errors.push({
            type: 'invalid_npu',
            message: `芯片标签格式错误：无效的NPU值，${npu} 不在配置的有效NPU列表中`,
            line: child.line
          });
        }
      });

      if (node.npu && node.id) {
        const parentNpus = node.npu.split(',').map(n => n.trim());
        const childNpus = child.npu.split(',').map(n => n.trim());

        const missingInParent = childNpus.filter(n => !parentNpus.includes(n));

        if (missingInParent.length > 0) {
          this.errors.push({
            type: 'npu_hierarchy',
            message: `芯片标签格式错误：NPU值不匹配，子标记包含 ${missingInParent.join(',')}，但父标记(第 ${node.line} 行)的npu="${node.npu}"中不包含这些值`,
            line: child.line
          });
        }
      }

      this.validateNpuInTree(child);
    });
  }

  checkPositionTags() {
    this.positionTags.forEach(tag => {
      if (!tag.extracted) {
        this.errors.push({
          type: 'invalid_ref_format',
          message: `位置标签格式错误: 标签 "${tag.content}" 不符合位置标签格式要求，正确格式为 <!-- @ref: 仓名/res/.../xx_res.md#idxxxxx -->`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
        return;
      }

      if (tag.repoName !== this.repoName) {
        this.errors.push({
          type: 'invalid_ref_repo',
          message: `位置标签格式错误: 标签中仓名为"${tag.repoName}"，但文档所在仓为"${this.repoName}"`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
      }

      if (!tag.refPath.startsWith('/res/')) {
        this.errors.push({
          type: 'invalid_ref_path',
          message: `位置标签格式错误: 标签中路径"${tag.refPath}"不以/res/开头`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
      }

      if (tag.refPath.includes('..')) {
        this.errors.push({
          type: 'invalid_ref_path',
          message: `位置标签格式错误: 标签中路径包含相对路径".."`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
      }

      const fileName = tag.refPath.substring(tag.refPath.lastIndexOf('/') + 1);
      if (!fileName.endsWith('_res.md')) {
        this.errors.push({
          type: 'invalid_ref_path',
          message: `位置标签格式错误: 标签中文件名"${fileName}"不以_res.md结尾`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
      }

      if (!tag.anchor) {
        this.errors.push({
          type: 'invalid_ref_anchor',
          message: `位置标签格式错误: 标签缺少id`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
      } else if (!/^id\d+$/.test(tag.anchor)) {
        this.errors.push({
          type: 'invalid_ref_anchor',
          message: `位置标签格式错误: 标签中"${tag.anchor}"不符合id+数字格式`,
          line: tag.line,
          ruleName: '位置标签的合法性检查'
        });
      }
    });
  }

  formatResults() {
    return this.errors.map(error => {
      const filePath = error.filePath || this.mdFilePath;
      const fileName = path.basename(filePath);
      const absolutePath = path.resolve(filePath);
      const lines = this.fileLinesMap[filePath] || this.lines;
      const lineContent = lines[error.line - 1] || '';
      const location = `vscode://file/${absolutePath}:${error.line}`;

      return {
        rule_group: '通用规则',
        rule_item: 'compliance-check',
        rule_type: '规范性',
        rule_name: error.ruleName || '芯片标签的合法性检查',
        snippet: lineContent.trim(),
        description: error.message,
        rule_level: '严重',
        file_name: fileName,
        line_number: error.line.toString(),
        location: location,
        suggestion: this.suggestions[error.type] || '请检查标记语法是否正确'
      };
    });
  }

  printReport() {
    if (this.errors.length === 0) {
      console.log('✅ 文档标记检查通过，没有发现错误！');
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
        const filePath = error.filePath || this.mdFilePath;
        const fileName = path.basename(filePath);
        console.log(`  - ${fileName} 第 ${error.line} 行: ${error.message}`);
      });
      console.log('');
    });

    const resultFile = this.totalJsonPath;
    console.log(`检查结果已保存到: ${resultFile}`);
  }
}

function main() {
  const args = process.argv.slice(2);

  if (args.length < 3) {
    console.log('使用方法: node tag_checker.js <markdown文件/文件夹路径> <检查结果总JSON文件绝对路径> <仓名>');
    console.log('示例: node tag_checker.js ./markdown ./output/total_result.json my-repo');
    process.exit(1);
  }

  const mdPath = args[0];
  const totalJsonPath = args[1];
  const repoName = args[2];
  const configPath = path.join(__dirname, 'npus.json');

  if (!fs.existsSync(mdPath)) {
    console.log(`错误: 路径 ${mdPath} 不存在`);
    process.exit(1);
  }

  if (!fs.existsSync(configPath)) {
    console.log(`错误: 配置文件 ${configPath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在
  const parentDir = path.dirname(path.resolve(totalJsonPath));
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  const checker = new TagChecker(configPath, mdPath, totalJsonPath, repoName);
  checker.checkFile();
  checker.printReport();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

main();
