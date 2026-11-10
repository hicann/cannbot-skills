// SPDX-License-Identifier: CANN-2.0
const fs = require('fs');
const path = require('path');

class EncodingFormatChecker {
  constructor(mdFilePath, totalJsonPath) {
    this.mdFilePath = mdFilePath;
    this.totalJsonPath = totalJsonPath;
    this.errors = [];
    this.suggestions = {
      encoding_error: '请将文件编码转换为纯UTF-8格式，不支持其他格式'
    };
  }

  /**
   * 检查 Buffer 前三个字节是否为 EF BB BF（UTF-8 BOM 标记）
   * @param {Buffer} buffer
   * @returns {boolean}
   */
  isUtf8Bom(buffer) {
    if (buffer.length < 3) {
      return false;
    }
    return buffer[0] === 0xEF && buffer[1] === 0xBB && buffer[2] === 0xBF;
  }

  /**
   * 逐字节验证 UTF-8 编码规则
   * @param {Buffer} buffer
   * @returns {boolean}
   */
  isValidUtf8(buffer) {
    let i = 0;
    while (i < buffer.length) {
      const byte = buffer[i];

      // 0x00-0x7F: 单字节 ASCII
      if (byte <= 0x7F) {
        i++;
        continue;
      }

      // 0xC0-0xC1: 非法（overlong encoding）
      if (byte >= 0xC0 && byte <= 0xC1) {
        return false;
      }

      // 0xF5-0xFF: 非法
      if (byte >= 0xF5) {
        return false;
      }

      // 0x80-0xBF: 孤立后续字节，非法
      if (byte >= 0x80 && byte <= 0xBF) {
        return false;
      }

      // 2 字节序列: 0xC2-0xDF
      if (byte >= 0xC2 && byte <= 0xDF) {
        if (i + 1 >= buffer.length) return false;
        if (buffer[i + 1] < 0x80 || buffer[i + 1] > 0xBF) return false;
        i += 2;
        continue;
      }

      // 3 字节序列: 0xE0-0xEF
      if (byte >= 0xE0 && byte <= 0xEF) {
        if (i + 2 >= buffer.length) return false;
        const b1 = buffer[i + 1];
        const b2 = buffer[i + 2];

        if (b2 < 0x80 || b2 > 0xBF) return false;

        if (byte === 0xE0) {
          // E0 后须为 A0-BF
          if (b1 < 0xA0 || b1 > 0xBF) return false;
        } else if (byte === 0xED) {
          // ED 后须为 80-9F
          if (b1 < 0x80 || b1 > 0x9F) return false;
        } else {
          if (b1 < 0x80 || b1 > 0xBF) return false;
        }

        i += 3;
        continue;
      }

      // 4 字节序列: 0xF0-0xF4
      if (byte >= 0xF0 && byte <= 0xF4) {
        if (i + 3 >= buffer.length) return false;
        const b1 = buffer[i + 1];
        const b2 = buffer[i + 2];
        const b3 = buffer[i + 3];

        if (b2 < 0x80 || b2 > 0xBF) return false;
        if (b3 < 0x80 || b3 > 0xBF) return false;

        if (byte === 0xF0) {
          // F0 后须为 90-BF
          if (b1 < 0x90 || b1 > 0xBF) return false;
        } else if (byte === 0xF4) {
          // F4 后须为 80-8F
          if (b1 < 0x80 || b1 > 0x8F) return false;
        } else {
          if (b1 < 0x80 || b1 > 0xBF) return false;
        }

        i += 4;
        continue;
      }

      // 不应到达此处，但以防万一
      return false;
    }
    return true;
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
   * 读取单个文件字节并检查编码
   * @param {string} filePath
   * @returns {Object|null}
   */
  checkEncoding(filePath) {
    const buffer = fs.readFileSync(filePath);

    // 空文件视为合法
    if (buffer.length === 0) {
      return null;
    }

    // 检查 UTF-8 BOM
    if (this.isUtf8Bom(buffer)) {
      return {
        filePath: filePath,
        snippet: 'UTF-8 BOM编码',
        description: '编码格式错误，格式为：UTF-8 BOM'
      };
    }

    // 验证 UTF-8 字节序列
    if (!this.isValidUtf8(buffer)) {
      return {
        filePath: filePath,
        snippet: '非UTF-8编码',
        description: '编码格式错误，格式为：非UTF-8编码'
      };
    }

    return null;
  }

  /**
   * 主入口方法：收集文件 -> 排除白名单 -> 逐一检查 -> 格式化 -> 合并追加到总JSON
   * @returns {Array}
   */
  checkFile() {
    // 1. 识别路径类型并收集文件
    let mdFiles = [];
    if (fs.statSync(this.mdFilePath).isFile()) {
      mdFiles = [path.resolve(this.mdFilePath)];
    } else if (fs.statSync(this.mdFilePath).isDirectory()) {
      mdFiles = this.getMdFiles(this.mdFilePath);
    }

    // 2. 读取排除列表
    let excludeList = [];
    const excludeFilePath = path.join(__dirname, '..', 'assets', 'exclude-files.json');
    try {
      if (fs.existsSync(excludeFilePath)) {
        excludeList = JSON.parse(fs.readFileSync(excludeFilePath, 'utf8'));
      }
    } catch (e) {
      // 无法读取排除列表，跳过排除逻辑
    }

    // 3. 过滤白名单文件
    mdFiles = mdFiles.filter(filePath => {
      const fileName = path.basename(filePath);
      return !excludeList.includes(fileName);
    });

    // 4. 逐一检查文件编码
    this.errors = [];
    mdFiles.forEach(filePath => {
      const error = this.checkEncoding(filePath);
      if (error) {
        this.errors.push(error);
      }
    });

    // 5. 格式化结果
    const results = this.formatResults();

    // 6. 结果汇总输出 — 无论是否有问题都合并追加到总JSON文件：
    // 零问题时写入 []（或保持既有内容），保证"检查成功"与"结果文件存在"一致，
    // 聚合方不至将清洁文档误判为 skipped
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

    // 7. 打印报告
    this.printReport();

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
      return {
        rule_group: '通用规则',
        rule_item: 'compliance-check',
        rule_type: '规范性',
        rule_name: 'MD编码格式检查',
        snippet: error.snippet,
        description: error.description,
        rule_level: '严重',
        file_name: fileName,
        line_number: '1',
        location: `vscode://file/${absolutePath}:1`,
        suggestion: this.suggestions.encoding_error
      };
    });
  }

  /**
   * 在控制台打印检查报告摘要
   */
  printReport() {
    if (this.errors.length === 0) {
      console.log('编码格式检查通过，没有发现错误！');
      return;
    }

    console.log(`发现 ${this.errors.length} 个编码格式错误：\n`);

    this.errors.forEach(error => {
      console.log(`  - ${path.basename(error.filePath)}: ${error.description}`);
    });

    console.log(`\n检查结果已保存到: ${this.totalJsonPath}`);
  }
}

/**
 * main 函数：解析命令行参数并执行检查
 * 调用方式：node encoding_format_checker.js <MD文件/文件夹路径> <检查结果总JSON文件绝对路径>
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('使用方法: node encoding_format_checker.js <MD文件/文件夹路径> <检查结果总JSON文件绝对路径>');
    console.log('示例: node encoding_format_checker.js ./docs ./output/total_result.json');
    process.exit(1);
  }

  const mdFilePath = args[0];
  const totalJsonPath = args[1];

  if (!fs.existsSync(mdFilePath)) {
    console.log(`错误: 路径 ${mdFilePath} 不存在`);
    process.exit(1);
  }

  // 确保总JSON文件的父目录存在
  const parentDir = path.dirname(totalJsonPath);
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  const checker = new EncodingFormatChecker(mdFilePath, totalJsonPath);
  checker.checkFile();

  if (checker.errors.length > 0) {
    process.exit(1);
  }
}

module.exports = { EncodingFormatChecker, main };

// 当直接运行脚本时执行 main
if (require.main === module) {
  main();
}
