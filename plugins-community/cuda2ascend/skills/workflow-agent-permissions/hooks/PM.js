// PM（主 Agent）：只调度不执行。
// 可写：仅 .cannbot（流程中间文件）
// 不可写：代码 / 测试 / 文档（实质产物一律派发给子 Agent）
export default {
  categories: [],
  exts: "*",
}
