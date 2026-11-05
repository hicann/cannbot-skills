// SPDX-License-Identifier: CANN-2.0
import fs from "node:fs";
import path from "node:path";

// ==================== 路径工具 ====================

function normalize(p) {
  return p.replace(/\\/g, "/");
}

function relpath(fullPath, base) {
  return normalize(path.relative(base, fullPath));
}

async function globFiles(pattern) {
  const results = [];
  for await (const entry of fs.promises.glob(normalize(pattern))) {
    results.push(entry);
  }
  return results;
}

async function globDirs(pattern) {
  const results = [];
  for await (const entry of fs.promises.glob(normalize(pattern), { withFileTypes: true })) {
    if (entry.isDirectory()) {
      results.push(entry.parentPath ? path.join(entry.parentPath, entry.name) : entry.name);
    }
  }
  return results;
}

// ==================== 跨平台文件搜索 ====================

function searchInFiles(root, patterns, dirPrefix = "", extensions = [".h", ".hpp", ".hxx"]) {
  const compiled = patterns.map((p) => new RegExp(p));
  const searchRoot = dirPrefix ? path.join(root, dirPrefix) : root;
  if (!fs.existsSync(searchRoot)) return [];

  const results = [];
  walkDir(searchRoot, (filePath) => {
    if (!extensions.some((ext) => filePath.endsWith(ext))) return;
    const rel = relpath(filePath, root);
    try {
      const lines = fs.readFileSync(filePath, "utf-8").split("\n");
      for (let i = 0; i < lines.length; i++) {
        if (compiled.some((re) => re.test(lines[i]))) {
          results.push(`${rel}:${i + 1}:${lines[i]}`);
          break;
        }
      }
    } catch {}
  });
  return results;
}

// Prefer recognized declarations; preserve the original search as a fallback
// for unsupported C++ syntax and declarations spanning multiple API categories.
function countMatchesInFiles(root, patterns, dirPrefix = "", extensions = [".h", ".hpp", ".hxx"]) {
  const compiled = patterns.map((p) => new RegExp(p));
  const searchRoot = dirPrefix ? path.join(root, dirPrefix) : root;
  if (!fs.existsSync(searchRoot)) return [];

  const results = [];
  walkDir(searchRoot, (filePath) => {
    if (!extensions.some((ext) => filePath.endsWith(ext))) return;
    const rel = relpath(filePath, root);
    try {
      const lines = fs.readFileSync(filePath, "utf-8").split("\n");
      let count = 0;
      for (const line of lines) {
        if (compiled.some((re) => re.test(line))) count++;
      }
      if (count > 0) results.push({ file: rel, count });
    } catch {}
  });
  return results.sort((a, b) => b.count - a.count);
}


function findDeclarationFiles(root, apiName) {
  const searchRoot = path.join(root, "include");
  if (!fs.existsSync(searchRoot)) return [];
  const results = [];
  const namePattern = new RegExp(`\\b${escapeRegex(apiName)}\\s*\\(`, "g");
  walkDir(searchRoot, (filePath) => {
    if (![".h", ".hpp", ".hxx"].includes(path.extname(filePath))) return;
    let source;
    try {
      source = fs.readFileSync(filePath, "utf-8");
    } catch {
      return;
    }
    source = source
      .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, " ")
      .replace(/^\s*#(?:[^\n\\]|\\[\s\S])*$/gm, "");
    let count = 0;
    namePattern.lastIndex = 0;
    for (const match of source.matchAll(namePattern)) {
      const start = Math.max(
        source.lastIndexOf(";", match.index - 1),
        source.lastIndexOf("{", match.index - 1),
        source.lastIndexOf("}", match.index - 1),
      ) + 1;
      const prefix = source.slice(start, match.index)
        .replace(/\b(public|protected|private)\s*:/g, "")
        .replace(/\btemplate\s*<[^;{}]*>/g, "")
        .replace(/\b(?:__in_pipe__|__out_pipe__|__inout_pipe__|__ASC_USE_RESERVED_UBUF__)\s*\([^()]*\)/g, "")
        .replace(/__attribute__\s*\(\([^()]*\)\)/g, "")
        .trim();
      // A declaration needs a return type. Calls, initializers and control-flow
      // expressions must not supply evidence for API classification.
      if (!prefix || /\b(return|co_return|throw|new|delete|if|while|for|switch|case)\b/.test(prefix)) continue;
      if (!/^[\w\s:*&,<>.\[\]]+$/.test(prefix) || !/[A-Za-z_]\w*/.test(prefix)) continue;
      let depth = 1;
      let end = match.index + match[0].length;
      for (; end < source.length && depth; end++) {
        if (source[end] === "(") depth++;
        if (source[end] === ")") depth--;
      }
      if (depth) continue;
      const suffix = source.slice(end);
      if (!/^\s*(?:(?:const|volatile|override|final|__aicore__|__aicpu__)\b\s*|&&?\s*|noexcept(?:\s*\([^()]*\))?\s*)*[;{]/.test(suffix)) continue;
      count++;
    }
    if (count) results.push({ file: relpath(filePath, root), count });
  });
  return results.sort((a, b) => a.file.localeCompare(b.file));
}

function walkDir(dir, callback) {
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walkDir(full, callback);
      else if (entry.isFile()) callback(full);
    }
  } catch {}
}

// ==================== 配置常量 ====================

const TYPE_FROM_PATH = {
  "include/basic_api/": "basic_api",
  "include/adv_api/": "adv_api",
  "include/simt_api/": "simt_api",
  "include/tensor_api/": "tensor_api",
  "include/c_api/": "c_api",
  "include/aicpu_api/": "aicpu_api",
  "include/utils/": "utils",
};

const OPERATOR_TO_API = {
  "operator+": "Add",
  "operator-": "Sub",
  "operator*": "Mul",
  "operator/": "Div",
  "operator|": "Or",
  "operator&": "And",
  "operator<": "Less",
  "operator>": "Greater",
  "operator==": "Equal",
  "operator!=": "NotEqual",
  "operator<=": "LessEqual",
  "operator>=": "GreaterEqual",
};

const OPERATOR_CATEGORIES = {
  subscript: {
    parent_classes: {
      LocalTensor: {
        decl_file: "include/basic_api/kernel_tensor.h",
        impl_file: "impl/basic_api/kernel_tensor_impl.h",
      },
      GlobalTensor: {
        decl_file: "include/basic_api/kernel_tensor.h",
        impl_file: "impl/basic_api/kernel_tensor_impl.h",
      },
    },
  },
  element_access: {
    parent_classes: {
      LocalTensor: {
        decl_file: "include/basic_api/kernel_tensor.h",
        impl_file: "impl/basic_api/kernel_tensor_impl.h",
      },
      GlobalTensor: {
        decl_file: "include/basic_api/kernel_tensor.h",
        impl_file: "impl/basic_api/kernel_tensor_impl.h",
      },
    },
  },
  assignment: {
    parent_classes: {
      LocalTensor: {
        decl_file: "include/basic_api/kernel_tensor.h",
        impl_file: "impl/basic_api/kernel_tensor_impl.h",
      },
      OpMC2Def: { decl_file: "include/utils/op_flag.h" },
      OpAICoreConfig: { decl_file: "include/utils/op_flag.h" },
    },
  },
};

const TEST_DIR_MAP = {
  basic_api: "tests/api/basic_api",
  adv_api: "tests/api/adv_api",
  simt_api: "tests/api/simt_api",
  tensor_api: "tests/api/tensor_api",
  c_api: "tests/api/c_api",
  aicpu_api: "tests/api/aicpu_api",
  utils: "tests/api/utils",
  common: "tests/api/common",
  reg_compute_api: "tests/api/reg_compute_api",
};

const COMMON_TEST_DIRS = ["tests/api/common", "tests/api/reg_compute_api"];

// ==================== 内容匹配 ====================

function fileContains(repo, filePath, pattern) {
  const full = path.join(repo, filePath);
  try {
    const content = fs.readFileSync(full, "utf-8");
    return pattern.test(content);
  } catch {
    return false;
  }
}

function apiInImplFile(repo, filePath, apiName, apiType) {
  const searchName = apiType === "basic_api" ? `${apiName}Impl` : apiName;
  return fileContains(repo, filePath, new RegExp(`\\b${escapeRegex(searchName)}\\b`));
}

function apiNameInFile(repo, filePath, apiName) {
  return fileContains(repo, filePath, new RegExp(`\\b${escapeRegex(apiName)}\\b`));
}

function apiInTestFile(repo, filePath, apiName) {
  return fileContains(repo, filePath, new RegExp(`\\b${escapeRegex(apiName)}\\s*[<(]`));
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ==================== 辅助函数 ====================

function classifyOperator(apiName) {
  if (apiName in OPERATOR_TO_API) return "symbol_override";
  if (apiName.startsWith("operator") && !["operator[]", "operator()", "operator="].includes(apiName))
    return "symbol_override";
  if (apiName === "operator[]") return "subscript";
  if (apiName === "operator()") return "element_access";
  if (apiName === "operator=") return "assignment";
  return null;
}

function extractOperatorFromDoc(docPath) {
  try {
    const lines = fs.readFileSync(docPath, "utf-8").split("\n");
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (line.startsWith("#")) {
        let title = line.replace(/^#+\s+/, "").trim();
        title = title.replace(/<a\s+name="[^"]*"><\/a>/, "").trim();
        if (title) {
          title = title.replace(/\\([\[\]()])/g, "$1");
          title = title.replace(/^(operator)\s+(.+)$/, "$1$2");
          return title;
        }
      } else if (line && !line.startsWith("<")) {
        break;
      }
    }
  } catch {}
  const basename = path.basename(docPath, path.extname(docPath));
  return basename.replace(/-\d+$/, "").trim();
}

function pickBestMatch(lines) {
  const inlineLines = lines.filter(
    (l) => l.includes("inline") || l.includes("__aicore__") || l.includes("__SIMT_DEVICE"),
  );
  if (!inlineLines.length) return lines[0] ?? null;
  for (const l of inlineLines) {
    if (l.includes("include/basic_api/kernel_operator")) return l;
  }
  return inlineLines[0];
}

function typeFromPath(filePath) {
  for (const [prefix, apiType] of Object.entries(TYPE_FROM_PATH)) {
    if (filePath.startsWith(prefix)) return apiType;
  }
  throw new Error(`Unknown API type from path: ${filePath}`);
}

function extractChipDirs(implFiles) {
  const chips = new Set();
  for (const f of implFiles) {
    for (const part of f.split("/")) {
      if (part.startsWith("dav_") || part.startsWith("npu_arch_")) chips.add(part);
    }
  }
  return [...chips].sort();
}

async function globImpl(repo, pattern) {
  const full = path.join(repo, pattern);
  const files = await globFiles(full);
  return files.map((f) => relpath(f, repo)).sort();
}

// ==================== APILocator ====================

export class DeclarationNotFoundError extends Error {}

export class APILocator {
  constructor(repoRoot = ".") {
    this.repo = path.resolve(repoRoot);
  }

  static extractApiName(docPath) {
    const absPath = path.resolve(docPath);
    const basename = path.basename(absPath, path.extname(absPath));
    if (basename.toLowerCase().startsWith("operator") || basename === "运算符重载") {
      return extractOperatorFromDoc(absPath);
    }
    let name = basename;
    name = name.replace(/[（(].*?[）)]/, "");
    name = name.replace(/-.*侧接口$/, "");
    name = name.replace(/接口$/, "");
    return name.replace(/-\d+$/, "").trim();
  }

  determineType(apiName, docPath) {
    const category = classifyOperator(apiName);

    if (category === "symbol_override") {
      const opSymbol = apiName.slice("operator".length);
      if (!opSymbol) throw new Error(`Invalid operator: ${apiName}`);
      const matches = searchInFiles(
        this.repo,
        [`operator\\s*${escapeRegex(opSymbol)}\\s*\\(`],
        "include",
      );
      if (!matches.length) throw new DeclarationNotFoundError(`Operator '${apiName}' not found in include/`);
      const filePath =
        matches.find((m) => m.includes("kernel_tensor.h"))?.split(":")[0] ??
        matches[0].split(":")[0];
      return { api_type: "basic_api", decl_files: [{ file: filePath, count: 1 }] };
    }

    if (category === "subscript" || category === "element_access" || category === "assignment") {
      const parentClasses = OPERATOR_CATEGORIES[category].parent_classes;
      const classNames = Object.keys(parentClasses);
      const resolvedDoc = docPath ? path.resolve(this.repo, docPath) : null;
      const ancestors = resolvedDoc ? normalize(path.dirname(resolvedDoc)).split("/").reverse() : [];
      let owner = ancestors.find((part) => classNames.includes(part));
      if (!owner && resolvedDoc) {
        const content = fs.readFileSync(resolvedDoc, "utf-8");
        // Ignore call examples: they may mention classes other than the owner.
        const introduction = content.split(/^##\s+调用示例/m)[0];
        const candidates = classNames.filter((name) =>
          new RegExp(`\\b${escapeRegex(name)}\\b`).test(introduction),
        );
        if (candidates.length === 1) owner = candidates[0];
      }
      if (!owner) {
        throw new Error(`Cannot determine owner class for '${apiName}'; provide a document identifying one of: ${classNames.join(", ")}`);
      }
      const declFile = parentClasses[owner].decl_file;
      if (!fs.existsSync(path.join(this.repo, declFile))) {
        throw new DeclarationNotFoundError(`Declaration file not found: ${declFile}`);
      }
      return { api_type: typeFromPath(declFile), decl_files: [{ file: declFile, count: 1 }] };
    }

    const declarations = findDeclarationFiles(this.repo, apiName);
    const declarationTypes = new Set(declarations.map((match) => typeFromPath(match.file)));
    if (declarations.length && declarationTypes.size === 1) {
      return { api_type: typeFromPath(declarations[0].file), decl_files: declarations };
    }

    // Keep the previous behavior when the lightweight matcher cannot make a
    // unique classification. This avoids silently losing existing API coverage.
    const matches = countMatchesInFiles(this.repo, [`\\b${escapeRegex(apiName)}\\s*[<(]`], "include");
    if (!matches.length) throw new DeclarationNotFoundError(`API '${apiName}' not found in include/`);
    return { api_type: typeFromPath(matches[0].file), decl_files: matches };
  }

  async findImplFiles(apiType, declFile, apiName) {
    const implApiName = OPERATOR_TO_API[apiName] ?? apiName;

    if (apiType === "aicpu_api") {
      return { impl_files: [], chip_dirs: [], has_chip_specific: false, total_chips: 0 };
    }

    if (apiType === "adv_api") {
      return this.findAdvImpl(implApiName);
    }

    const patterns = this.implPatterns(apiType, declFile, implApiName);
    const found = [];
    const operatorCategory = OPERATOR_CATEGORIES[classifyOperator(apiName)];
    if (operatorCategory) {
      for (const entry of Object.values(operatorCategory.parent_classes)) {
        if (
          entry.decl_file === declFile &&
          entry.impl_file &&
          fs.existsSync(path.join(this.repo, entry.impl_file)) &&
          !found.includes(entry.impl_file)
        ) {
          found.push(entry.impl_file);
        }
      }
    }
    for (const pat of patterns) {
      for (const impl of await globImpl(this.repo, pat)) {
        if (!found.includes(impl) && apiInImplFile(this.repo, impl, implApiName, apiType)) {
          found.push(impl);
        }
      }
    }

    if (apiType === "basic_api" && !found.length) {
      for (const pat of patterns) {
        for (const impl of await globImpl(this.repo, pat)) {
          if (!found.includes(impl) && apiNameInFile(this.repo, impl, implApiName)) {
            found.push(impl);
          }
        }
      }
    }

    if (apiType === "basic_api" && found.length) {
      const davDirs = new Set();
      for (const f of found) {
        for (const p of f.split("/")) {
          if (p.startsWith("dav_")) davDirs.add(p);
        }
      }
      for (const dav of [...davDirs].sort()) {
        for (const impl of await globImpl(this.repo, `impl/basic_api/${dav}/*_impl.h`)) {
          if (!found.includes(impl) && apiNameInFile(this.repo, impl, implApiName)) {
            found.push(impl);
          }
        }
      }
    }

    const chipDirs = extractChipDirs(found);
    const total = (await globDirs(path.join(this.repo, "impl/basic_api/dav_*"))).length;
    return {
      impl_files: found,
      chip_dirs: chipDirs,
      has_chip_specific: chipDirs.length > 0 && chipDirs.length < total,
      total_chips: total,
    };
  }

  async findAdvImpl(implApiName) {
    const patterns = [
      path.join(this.repo, "impl/adv_api/detail/**/*.h"),
      path.join(this.repo, "impl/adv_api/detail/**/*.cpp"),
      path.join(this.repo, "impl/adv_api/tiling/**/*.h"),
      path.join(this.repo, "impl/adv_api/tiling/**/*.cpp"),
    ];
    const files = [];
    for (const pattern of patterns) {
      files.push(...await globFiles(pattern));
    }

    const nameLower = implApiName.toLowerCase();
    const core = [];
    const checks = [];
    const companions = [];
    const tiling = [];
    const companionSuffixes = [
      "_common_impl", "_common", "_check", "_check_common",
      "_utils", "_base_impl", "_base", "_func",
      "_sub_func", "_sub_api", "_config", "_tiling_impl", "_tiling",
    ];

    for (const f of files) {
      const rel = relpath(f, this.repo);
      if (!rel.includes(`/${nameLower}/`)) continue;

      if (apiNameInFile(this.repo, rel, implApiName)) {
        if (!core.includes(rel)) core.push(rel);
      } else if (rel.includes("/api_check/") || rel.includes("/kernel_check/")) {
        checks.push(rel);
      } else if (companionSuffixes.some((s) => path.basename(rel, path.extname(rel)).toLowerCase().endsWith(s))) {
        companions.push(rel);
      } else if (rel.includes("/tiling/")) {
        tiling.push(rel);
      }
    }

    let result = [...core, ...checks, ...companions, ...tiling];
    if (!result.length) {
      for (const f of files) {
        const rel = relpath(f, this.repo);
        if (rel.toLowerCase().includes(nameLower) && apiNameInFile(this.repo, rel, implApiName)) {
          result.push(rel);
        }
      }
    }

    result = [...new Set(result)].sort();
    const chipDirs = extractChipDirs(result);
    const total = (await globDirs(path.join(this.repo, "impl/basic_api/dav_*"))).length;
    return {
      impl_files: result,
      chip_dirs: chipDirs,
      has_chip_specific: chipDirs.length > 0 && chipDirs.length < total,
      total_chips: total,
    };
  }

  implPatterns(apiType, declFile, _apiName) {
    if (apiType === "basic_api") {
      const base = path.basename(declFile).replace("_intf.h", "");
      const pats = [`impl/basic_api/dav_*/${base}_impl.h`];
      if (declFile.includes("/reg_compute/") || declFile.includes("/reg_base/")) {
        pats.push(
          "impl/basic_api/dav_*/kernel_operator_vec_binary_impl.h",
          "impl/basic_api/dav_*/kernel_operator_vec_unary_impl.h",
          "impl/basic_api/dav_*/kernel_operator_*.h",
        );
      }
      pats.push("impl/basic_api/dav_*/*_impl.h");
      return pats;
    }
    if (apiType === "simt_api") return ["impl/simt_api/**/*.h"];
    if (apiType === "tensor_api") return ["impl/tensor_api/**/*.h", "impl/tensor_api/**/*.cpp"];
    if (apiType === "c_api") {
      return [
        "impl/c_api/instr_impl/**/*.h",
        "impl/c_api/instr_impl/**/*.cpp",
        "impl/c_api/instr_impl/**/*.cc",
      ];
    }
    if (apiType === "utils") {
      const rel = relpath(declFile, "include/utils/");
      const subdir = path.dirname(rel);
      const fname = path.basename(rel, path.extname(rel));
      return [`impl/utils/${subdir}/${fname}/*.h`];
    }
    return [];
  }

  async findTestFiles(apiType, apiName) {
    const found = [];

    const search = async (testSubdir) => {
      const full = path.join(this.repo, testSubdir);
      if (!fs.existsSync(full)) return;
      const files = await globFiles(path.join(full, "**/*.cpp"));
      for (const f of files) {
        const cpp = relpath(f, this.repo);
        if (!found.includes(cpp) && apiInTestFile(this.repo, cpp, apiName)) {
          found.push(cpp);
        }
      }
    };

    const testSubdir = TEST_DIR_MAP[apiType];
    if (testSubdir) await search(testSubdir);
    for (const d of COMMON_TEST_DIRS) await search(d);
    if (!testSubdir && !found.length) await search("tests/api");

    return found;
  }
}
