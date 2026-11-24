# docs-branch-update 评测样例

本样例用于 docs-branch-update 的 evals 用例, 包含待更新的参考链接与待删除的 NPU 标签。

## 参考链接

- CANN 仓源码: <https://gitcode.com/cann/cann-ops/blob/master/src/sample.py>
- Ascend 仓文档: <https://gitcode.com/Ascend/mindstudio/blob/master/README.md>
- PyPTO 示例: <https://gitcode.com/cann/pypto-gym/blob/master/README.md>
- 社区版文档: <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/ops/sample>
- Pytorch 文档: <https://www.hiascend.com/document/detail/zh/Pytorch/latest/model/sample>
- TF 文档: <https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/latest/ops/sample>

## 产品内容示例

<!-- npu="x90" id1 -->
x90 不允许存在的内容块, 整块删除。
<!-- end id1 -->

<!-- npu="950,x90" id2 -->
混合类型标签块, 移除 x90 后保留本段内容。
<!-- end id2 -->

<!-- npu="x90" id3 -->
外层块开头, 内层嵌套同类型子块需配对删除。
<!-- npu="x90" id4 -->
内层嵌套块。
<!-- end id4 -->
外层块结尾。
<!-- end id3 -->

## 接缝与混合至空示例

接缝示例段落:

<!-- npu="x90" id5 -->
接缝前块内容, 删除后与下方 br 序列形成接缝。
<!-- end id5 -->
<br><br><br>
接缝后文本。

<!-- npu="x90,9030" id6 -->
混合类型标签块, 逐类型移除后列表为空, 整块删除。
<!-- end id6 -->
