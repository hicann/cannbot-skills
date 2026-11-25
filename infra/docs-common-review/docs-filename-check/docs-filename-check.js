#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * Markdown文件名称质量检查脚本（docs-filename-check）
 * 用于检查Markdown文档名称的规范性和正确性
 *
 * 检查项：中文字符、不允许的特殊字符、名称长度（>70 ASCII）、同目录重名（按去扩展名 Stem 判定）。
 *
 * 用法: node docs-filename-check.js <输入路径> [输出目录]
 *   输入路径: Markdown 文档或包含 Markdown 文档的文件夹（绝对路径）
 *   输出目录: 检查结果存放目录（可选，默认 <脚本目录上三级>/output/）
 *
 * 结果文件: <输出目录>/filename-markdown-check-result.json（运行前删除旧文件后重建；
 *           重名检查按脚本语义仅覆盖目标所在目录第一层的 .md 文件，多目标须各用独立输出目录）。
 *
 * 退出码：0 = 正常执行（含 0 问题）；1 = 参数缺失。
 */
"use strict";

const fs = require("fs");
const path = require("path");

// ---------- 常量 ----------
const RULE_INFO = {
  rule_item_name: "filename-check",
  rule_name: "Filename Check",
  rule_type: "规范性",
  rule_level: "提示",
};

// ---------- 工具函数 ----------
function isMarkdownFile(filePath) {
  return filePath.toLowerCase().endsWith(".md");
}

function getFilenameWithoutExtension(filePath) {
  const filename = path.basename(filePath);
  if (filename.toLowerCase().endsWith(".md")) {
    return filename.slice(0, -3);
  }
  return filename;
}

function makeIssue(basename, locationPath, description, suggestion) {
  return {
    rule_group: "通用规则",
    rule_item: RULE_INFO.rule_item_name,
    rule_type: RULE_INFO.rule_type,
    rule_name: RULE_INFO.rule_name,
    snippet: basename,
    description: description,
    rule_level: RULE_INFO.rule_level,
    file_name: basename,
    line_number: "",
    location: "vscode://file/" + locationPath,
    suggestion: suggestion,
  };
}

// ---------- 单项检查 ----------
// 文件名包含中文字符
function checkChineseCharacters(filenameWithoutExt, fullPath) {
  const chineseChars = filenameWithoutExt.match(/[\u4e00-\u9fff]+/g);
  if (chineseChars) {
    const chineseText = chineseChars.join("");
    const basename = filenameWithoutExt + ".md";
    return makeIssue(
      basename,
      fullPath.replace(/\\/g, "/"),
      '文件名称中包含"' + chineseText + '"中文字符',
      "建议将文件名称修改为纯英文格式"
    );
  }
  return null;
}

// 文件名包含不允许的特殊字符（存在中文问题时跳过）
function checkSpecialCharacters(filenameWithoutExt, fullPath) {
  if (/^[a-zA-Z0-9\-_&]+$/.test(filenameWithoutExt)) {
    return null;
  }
  const invalidChars = [];
  for (const char of filenameWithoutExt) {
    if (!/[a-zA-Z0-9\-_&]/.test(char) && !/[\u4e00-\u9fff]/.test(char)) {
      invalidChars.push(char);
    }
  }
  if (invalidChars.length === 0) {
    return null;
  }
  const invalidText = invalidChars.join("");
  const basename = filenameWithoutExt + ".md";
  return makeIssue(
    basename,
    fullPath.replace(/\\/g, "/"),
    '文件名称中包含不允许的特殊字符"' + invalidText + '"',
    "建议将文件名称中的特殊字符替换为英文字母、数字、英文中横线（-）、英文下划线（_）或&符号"
  );
}

// 文件名长度超过 70 个 ASCII 字符
function checkFilenameLength(filenameWithoutExt, fullPath) {
  let asciiCount = 0;
  for (const char of filenameWithoutExt) {
    if (char.charCodeAt(0) < 128) {
      asciiCount++;
    }
  }
  if (asciiCount <= 70) {
    return null;
  }
  const basename = filenameWithoutExt + ".md";
  return makeIssue(
    basename,
    fullPath.replace(/\\/g, "/"),
    "文件名称超过了70个ASCII字母/数字/符号，当前长度为" + asciiCount,
    "建议缩短文件名称，使其不超过70个ASCII字母/数字/符号"
  );
}

// 同目录第一层重名（按去扩展名 Stem 判定，扩展名大小写不同亦视为重名）
function checkDuplicateFilenames(directory) {
  const issues = [];
  const mdFiles = [];
  for (const file of fs.readdirSync(directory)) {
    const filePath = path.join(directory, file);
    let st;
    try {
      st = fs.statSync(filePath);
    } catch {
      continue;
    }
    if (st.isFile() && isMarkdownFile(filePath)) {
      mdFiles.push({
        full_path: filePath,
        filename: getFilenameWithoutExtension(filePath),
        basename: file,
      });
    }
  }
  const filenameCount = new Map();
  for (const mdFile of mdFiles) {
    if (!filenameCount.has(mdFile.filename)) {
      filenameCount.set(mdFile.filename, []);
    }
    filenameCount.get(mdFile.filename).push(mdFile);
  }
  for (const [filename, files] of filenameCount) {
    if (files.length > 1) {
      for (const fileInfo of files) {
        issues.push(
          makeIssue(
            fileInfo.basename,
            fileInfo.full_path.replace(/\\/g, "/"),
            '目录下存在重名文件"' + filename + '"',
            "建议修改文件名称，确保同一目录下不存在重名的markdown文件"
          )
        );
      }
    }
  }
  return issues;
}

// ---------- 组合检查 ----------
function checkSingleFile(filePath) {
  const issues = [];
  if (!isMarkdownFile(filePath)) {
    return issues;
  }
  const filenameWithoutExt = getFilenameWithoutExtension(filePath);

  const chineseIssue = checkChineseCharacters(filenameWithoutExt, filePath);
  if (chineseIssue) {
    issues.push(chineseIssue);
  } else {
    const specialIssue = checkSpecialCharacters(filenameWithoutExt, filePath);
    if (specialIssue) {
      issues.push(specialIssue);
    }
  }

  const lengthIssue = checkFilenameLength(filenameWithoutExt, filePath);
  if (lengthIssue) {
    issues.push(lengthIssue);
  }

  const duplicateIssues = checkDuplicateFilenames(path.dirname(filePath));
  for (const issue of duplicateIssues) {
    if (issue.file_name === filenameWithoutExt + ".md") {
      issues.push(issue);
    }
  }
  return issues;
}

function checkDirectory(directory) {
  const issues = [];
  const mdFiles = [];
  for (const file of fs.readdirSync(directory)) {
    const filePath = path.join(directory, file);
    let st;
    try {
      st = fs.statSync(filePath);
    } catch {
      continue;
    }
    if (st.isFile() && isMarkdownFile(filePath)) {
      mdFiles.push(filePath);
    }
  }
  for (const mdFile of mdFiles) {
    issues.push(...checkSingleFile(mdFile));
  }

  const duplicateIssues = checkDuplicateFilenames(directory);
  const existingFiles = new Set(issues.map((issue) => issue.file_name));
  for (const duplicateIssue of duplicateIssues) {
    if (!existingFiles.has(duplicateIssue.file_name)) {
      issues.push(duplicateIssue);
    }
  }
  return issues;
}

function run(inputPath) {
  let st = null;
  try {
    st = fs.statSync(inputPath);
  } catch {}
  if (st && st.isFile()) {
    return checkSingleFile(inputPath);
  }
  if (st && st.isDirectory()) {
    return checkDirectory(inputPath);
  }
  console.log("错误：输入路径不存在 - " + inputPath);
  return [];
}

function saveResult(result, outputPath) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  // 与被替换的 python 版文本模式写盘行为保持一致：Windows 下行尾为 CRLF，其余平台 LF；
  // JSON 字符串内的换行已被转义，仅结构换行参与替换；结尾不带换行符。
  const eol = process.platform === "win32" ? "\r\n" : "\n";
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2).replace(/\n/g, eol), "utf8");
  console.log("检查结果已保存到: " + outputPath);
}

// ---------- 主流程 ----------
function main() {
  if (process.argv.length < 3) {
    console.log("使用方法: node docs-filename-check.js <输入路径> [输出路径]");
    console.log("输入路径可以是:");
    console.log("  - Markdown文档的绝对路径");
    console.log("  - 包含Markdown文档的文件夹绝对路径");
    console.log("输出路径（可选）:");
    console.log("  - 检查结果存放的绝对路径（文件夹）");
    console.log("  - 默认为项目根目录下的output文件夹");
    process.exit(1);
  }

  const inputPath = process.argv[2];
  let outputDir;
  if (process.argv.length >= 4) {
    outputDir = process.argv[3];
  } else {
    outputDir = path.join(path.resolve(__dirname, "..", "..", ".."), "output");
  }
  const resultFilePath = path.join(outputDir, "filename-markdown-check-result.json");

  if (fs.existsSync(resultFilePath)) {
    fs.rmSync(resultFilePath, { force: true });
    console.log("已删除旧的检查结果文件: " + resultFilePath);
  }

  const result = run(inputPath);

  if (result.length > 0) {
    console.log("发现 " + result.length + " 个问题:");
    for (const issue of result) {
      console.log("  - " + issue.file_name + ": " + issue.description);
    }
  } else {
    console.log("未发现问题");
  }

  saveResult(result, resultFilePath);
}

main();
