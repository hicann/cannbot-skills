// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// 把会话视图打成单个 JS bundle（去掉 Next 路由依赖），供嵌入式导出 HTML 内联使用
import esbuild from "esbuild"
import path from "node:path"
import fs from "node:fs"
import { fileURLToPath } from "node:url"

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, "..")

const alias = {
  "next/navigation": path.resolve(root, "src/export-view/shims/next-navigation.ts"),
}

// 解析 @/ 别名到 src/ 下实际文件（带扩展名）
const atAliasPlugin = {
  name: "at-alias",
  setup(build) {
    build.onResolve({ filter: /^@\// }, (args) => {
      const rel = args.path.replace(/^@\//, "")
      const base = path.resolve(root, "src", rel)
      for (const ext of ["", ".tsx", ".ts", "/index.tsx", "/index.ts", "/index.js"]) {
        const candidate = base + ext
        if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
          return { path: candidate }
        }
      }
      return { path: base, external: false }
    })
  },
}

await esbuild.build({
  entryPoints: [path.resolve(root, "src/export-view/entry.tsx")],
  bundle: true,
  outfile: path.resolve(root, "public/export-view.js"),
  format: "iife",
  target: "es2020",
  jsx: "automatic",
  loader: { ".ts": "ts", ".tsx": "tsx" },
  alias,
  plugins: [atAliasPlugin],
  resolveExtensions: [".tsx", ".ts", ".jsx", ".js", ".css", ".json"],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
    "process.env.NEXT_PUBLIC_SHOW_ADVANCED_TABS": JSON.stringify("false"),
  },
  legalComments: "none",
  logLevel: "info",
}).catch((e) => {
  console.error(e)
  process.exit(1)
})
