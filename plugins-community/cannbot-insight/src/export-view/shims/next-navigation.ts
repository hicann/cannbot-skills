// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// 嵌入式导出 bundle 的 next/navigation shim：用全局变量替代 Next 路由
export function useSearchParams(): URLSearchParams {
  const fw = (typeof window !== "undefined" && (window as unknown as { __EXPORT_FRAMEWORK?: string }).__EXPORT_FRAMEWORK) || ""
  return new URLSearchParams(fw ? `framework=${fw}` : "")
}
export function useRouter() {
  return { push: () => {}, replace: () => {}, back: () => {}, refresh: () => {} }
}
export function usePathname(): string {
  return "/session/export"
}
export function redirect(): never {
  throw new Error("redirect not supported in export bundle")
}
