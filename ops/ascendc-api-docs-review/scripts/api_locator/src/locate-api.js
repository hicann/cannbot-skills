// SPDX-License-Identifier: CANN-2.0
import path from "node:path";
import { APILocator, DeclarationNotFoundError } from "./api-locator.js";

async function locate(docPath, repoRoot = ".") {
  repoRoot = path.resolve(repoRoot);
  if (!path.isAbsolute(docPath)) {
    docPath = path.join(repoRoot, docPath);
  }

  const locator = new APILocator(repoRoot);
  const apiName = APILocator.extractApiName(docPath);
  let info;
  try {
    info = locator.determineType(apiName, docPath);
  } catch (e) {
    if (e instanceof DeclarationNotFoundError) return {};
    throw e;
  }

  const seen = new Set();
  const grouped = {};

  const add = (fp) => {
    if (fp && !seen.has(fp)) {
      seen.add(fp);
      const category = fp.split("/")[0];
      (grouped[category] ??= []).push(fp);
    }
  };

  for (const { file } of info.decl_files) add(file);
  const implResult = await locator.findImplFiles(info.api_type, info.decl_files[0].file, apiName);
  for (const fp of implResult.impl_files) add(fp);
  for (const fp of await locator.findTestFiles(info.api_type, apiName)) add(fp);

  return grouped;
}

async function main() {
  const args = process.argv.slice(2);
  let docPath = "";
  let repoRoot = ".";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--repo") {
      repoRoot = args[++i] ?? ".";
    } else if (!args[i].startsWith("--")) {
      docPath = args[i];
    }
  }

  if (!docPath) {
    console.error("用法: node src/locate-api.js <文档路径> [--repo <仓库路径>]");
    process.exit(1);
  }

  try {
    const grouped = await locate(docPath, repoRoot);
    for (const category of Object.keys(grouped).sort()) {
      console.log(`[${category}]`);
      for (const f of grouped[category]) {
        console.log(`  ${f}`);
      }
    }
  } catch (e) {
    console.error(`错误: ${e instanceof Error ? e.message : e}`);
    process.exit(1);
  }
}

main();
