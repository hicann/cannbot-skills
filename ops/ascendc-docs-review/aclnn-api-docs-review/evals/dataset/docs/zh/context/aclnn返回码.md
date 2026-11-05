# aclnn返回码

调用aclnn API时，常见的接口返回码如[表1](#zh-cn_topic_0000001563019104_table8155243135018)所示。
对于异常状态码值，可以通过aclGetRecentErrMsg接口（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）获取异常信息，您可以根据报错提示排查问题或者联系技术支持。

**表1**  返回状态码

<a name="zh-cn_topic_0000001563019104_table8155243135018"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001563019104_row111561243135019"><th class="cellrowborder" valign="top" width="30.543054305430545%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001563019104_p6676115185014"><a name="zh-cn_topic_0000001563019104_p6676115185014"></a><a name="zh-cn_topic_0000001563019104_p6676115185014"></a>状态码名称</p>
</th>
<th class="cellrowborder" valign="top" width="15.971597159715973%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001563019104_p16690195185015"><a name="zh-cn_topic_0000001563019104_p16690195185015"></a><a name="zh-cn_topic_0000001563019104_p16690195185015"></a>状态码值</p>
</th>
<th class="cellrowborder" valign="top" width="53.48534853485349%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001563019104_p107021951145010"><a name="zh-cn_topic_0000001563019104_p107021951145010"></a><a name="zh-cn_topic_0000001563019104_p107021951145010"></a>状态码说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001563019104_row2015624345019"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p45716143512"><a name="zh-cn_topic_0000001563019104_p45716143512"></a><a name="zh-cn_topic_0000001563019104_p45716143512"></a>ACLNN_SUCCESS</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p205761419512"><a name="zh-cn_topic_0000001563019104_p205761419512"></a><a name="zh-cn_topic_0000001563019104_p205761419512"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p95741410511"><a name="zh-cn_topic_0000001563019104_p95741410511"></a><a name="zh-cn_topic_0000001563019104_p95741410511"></a>成功。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>ACLNN_ERR_PARAM_NULLPTR</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>161001</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1015624311507"><a name="zh-cn_topic_0000001563019104_p1015624311507"></a><a name="zh-cn_topic_0000001563019104_p1015624311507"></a>参数校验错误，参数中存在非法的nullptr。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row315644318505"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p11156144312509"><a name="zh-cn_topic_0000001563019104_p11156144312509"></a><a name="zh-cn_topic_0000001563019104_p11156144312509"></a>ACLNN_ERR_PARAM_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p915619437501"><a name="zh-cn_topic_0000001563019104_p915619437501"></a><a name="zh-cn_topic_0000001563019104_p915619437501"></a>161002</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p17570123425314"><a name="zh-cn_topic_0000001563019104_p17570123425314"></a><a name="zh-cn_topic_0000001563019104_p17570123425314"></a>参数校验错误，如输入的两个数据类型不满足输入类型推导关系。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row1215674375018"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p11156174305019"><a name="zh-cn_topic_0000001563019104_p11156174305019"></a><a name="zh-cn_topic_0000001563019104_p11156174305019"></a>ACLNN_ERR_RUNTIME_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p9156443185011"><a name="zh-cn_topic_0000001563019104_p9156443185011"></a><a name="zh-cn_topic_0000001563019104_p9156443185011"></a>361001</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p4156543185018"><a name="zh-cn_topic_0000001563019104_p4156543185018"></a><a name="zh-cn_topic_0000001563019104_p4156543185018"></a>API内部调用npu runtime的接口异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACLNN_ERR_INNER_XXX</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>561xxx</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>API内部发生异常。</p>

</td>
</tr>
</tbody>
</table>

更多关于ACLNN_ERR_INNER_XXX类状态码的说明如[表2](#zh-cn_topic_0000001563019104_table3354143205413)所示。

**表2**  异常状态码

<a name="zh-cn_topic_0000001563019104_table3354143205413"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001563019104_row15354124318546"><th class="cellrowborder" valign="top" width="30.183018301830185%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001563019104_p5354164365416"><a name="zh-cn_topic_0000001563019104_p5354164365416"></a><a name="zh-cn_topic_0000001563019104_p5354164365416"></a>状态码名称</p>
</th>
<th class="cellrowborder" valign="top" width="16.521652165216523%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001563019104_p1435494317545"><a name="zh-cn_topic_0000001563019104_p1435494317545"></a><a name="zh-cn_topic_0000001563019104_p1435494317545"></a>状态码值</p>
</th>
<th class="cellrowborder" valign="top" width="53.295329532953296%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001563019104_p5354144325413"><a name="zh-cn_topic_0000001563019104_p5354144325413"></a><a name="zh-cn_topic_0000001563019104_p5354144325413"></a>状态码说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001563019104_row258894155510"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p261245115511"><a name="zh-cn_topic_0000001563019104_p261245115511"></a><a name="zh-cn_topic_0000001563019104_p261245115511"></a>ACLNN_ERR_INNER</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p452316256589"><a name="zh-cn_topic_0000001563019104_p452316256589"></a><a name="zh-cn_topic_0000001563019104_p452316256589"></a>561000</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1358812475513"><a name="zh-cn_topic_0000001563019104_p1358812475513"></a><a name="zh-cn_topic_0000001563019104_p1358812475513"></a>内部异常：API发生内部异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row12354184318545"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p415951713554"><a name="zh-cn_topic_0000001563019104_p415951713554"></a><a name="zh-cn_topic_0000001563019104_p415951713554"></a>ACLNN_ERR_INNER_INFERSHAPE_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p15159617105518"><a name="zh-cn_topic_0000001563019104_p15159617105518"></a><a name="zh-cn_topic_0000001563019104_p15159617105518"></a>561001</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p71581171559"><a name="zh-cn_topic_0000001563019104_p71581171559"></a><a name="zh-cn_topic_0000001563019104_p71581171559"></a>内部异常：API内部进行输出shape推导发生错误。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row163550434545"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p6158517195511"><a name="zh-cn_topic_0000001563019104_p6158517195511"></a><a name="zh-cn_topic_0000001563019104_p6158517195511"></a>ACLNN_ERR_INNER_TILING_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1915881745518"><a name="zh-cn_topic_0000001563019104_p1915881745518"></a><a name="zh-cn_topic_0000001563019104_p1915881745518"></a>561002</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p61572017115518"><a name="zh-cn_topic_0000001563019104_p61572017115518"></a><a name="zh-cn_topic_0000001563019104_p61572017115518"></a>内部异常：API内部做npu kernel的tiling时发生异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row635584311549"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p21875435618"><a name="zh-cn_topic_0000001563019104_p21875435618"></a><a name="zh-cn_topic_0000001563019104_p21875435618"></a>ACLNN_ERR_INNER_FIND_KERNEL_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p71561617195511"><a name="zh-cn_topic_0000001563019104_p71561617195511"></a><a name="zh-cn_topic_0000001563019104_p71561617195511"></a>561003</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10156181715518"><a name="zh-cn_topic_0000001563019104_p10156181715518"></a><a name="zh-cn_topic_0000001563019104_p10156181715518"></a>内部异常：API内部做查找npu kernel异常（可能因为算子二进制包未安装）。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row1235514395412"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p15155111717551"><a name="zh-cn_topic_0000001563019104_p15155111717551"></a><a name="zh-cn_topic_0000001563019104_p15155111717551"></a>ACLNN_ERR_INNER_CREATE_EXECUTOR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1915531735516"><a name="zh-cn_topic_0000001563019104_p1915531735516"></a><a name="zh-cn_topic_0000001563019104_p1915531735516"></a>561101</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1715411174552"><a name="zh-cn_topic_0000001563019104_p1715411174552"></a><a name="zh-cn_topic_0000001563019104_p1715411174552"></a>内部异常：API内部创建aclOpExecutor失败（可能因为操作系统异常）。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row0355104311548"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p101549170557"><a name="zh-cn_topic_0000001563019104_p101549170557"></a><a name="zh-cn_topic_0000001563019104_p101549170557"></a>ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p12153111713557"><a name="zh-cn_topic_0000001563019104_p12153111713557"></a><a name="zh-cn_topic_0000001563019104_p12153111713557"></a>561102</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p913381713554"><a name="zh-cn_topic_0000001563019104_p913381713554"></a><a name="zh-cn_topic_0000001563019104_p913381713554"></a>内部异常：API内部未调用uniqueExecutor ReleaseTo。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row10896192245618"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p178962225564"><a name="zh-cn_topic_0000001563019104_p178962225564"></a><a name="zh-cn_topic_0000001563019104_p178962225564"></a>ACLNN_ERR_INNER_NULLPTR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1589612295620"><a name="zh-cn_topic_0000001563019104_p1589612295620"></a><a name="zh-cn_topic_0000001563019104_p1589612295620"></a>561103</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1589610226566"><a name="zh-cn_topic_0000001563019104_p1589610226566"></a><a name="zh-cn_topic_0000001563019104_p1589610226566"></a>内部异常：aclnn API内部发生异常，出现了nullptr的异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row188961222185610"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p9622952105614"><a name="zh-cn_topic_0000001563019104_p9622952105614"></a><a name="zh-cn_topic_0000001563019104_p9622952105614"></a>ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p873282205914"><a name="zh-cn_topic_0000001563019104_p873282205914"></a><a name="zh-cn_topic_0000001563019104_p873282205914"></a>561104</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p48969225564"><a name="zh-cn_topic_0000001563019104_p48969225564"></a><a name="zh-cn_topic_0000001563019104_p48969225564"></a>内部异常：aclnn API内部发生异常，算子的属性个数异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row989662295618"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14896022205612"><a name="zh-cn_topic_0000001563019104_p14896022205612"></a><a name="zh-cn_topic_0000001563019104_p14896022205612"></a>CONFLICT</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1489632285612"><a name="zh-cn_topic_0000001563019104_p1489632285612"></a><a name="zh-cn_topic_0000001563019104_p1489632285612"></a>561105</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p98961022135619"><a name="zh-cn_topic_0000001563019104_p98961022135619"></a><a name="zh-cn_topic_0000001563019104_p98961022135619"></a>已废弃，请使用最新ACLNN_ERR_INNER_KEY_CONFLICT。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row989662295618"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14896022205612"><a name="zh-cn_topic_0000001563019104_p14896022205612"></a><a name="zh-cn_topic_0000001563019104_p14896022205612"></a>ACLNN_ERR_INNER_KEY_CONFLICT</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1489632285612"><a name="zh-cn_topic_0000001563019104_p1489632285612"></a><a name="zh-cn_topic_0000001563019104_p1489632285612"></a>561105</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p98961022135619"><a name="zh-cn_topic_0000001563019104_p98961022135619"></a><a name="zh-cn_topic_0000001563019104_p98961022135619"></a>内部异常：aclnn API内部发生异常，算子的kernel匹配的hash key发生冲突。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row14896122245614"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p159371233574"><a name="zh-cn_topic_0000001563019104_p159371233574"></a><a name="zh-cn_topic_0000001563019104_p159371233574"></a>ACLNN_ERR_INNER_INVALID_IMPL_MODE</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p39504318575"><a name="zh-cn_topic_0000001563019104_p39504318575"></a><a name="zh-cn_topic_0000001563019104_p39504318575"></a>561106</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1996314313577"><a name="zh-cn_topic_0000001563019104_p1996314313577"></a><a name="zh-cn_topic_0000001563019104_p1996314313577"></a>内部异常：aclnn API内部发生异常，算子的实现模式参数错误。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row1889682255611"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p121306241336"><a name="zh-cn_topic_0000001563019104_p121306241336"></a><a name="zh-cn_topic_0000001563019104_p121306241336"></a>ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14490517115910"><a name="zh-cn_topic_0000001563019104_p14490517115910"></a><a name="zh-cn_topic_0000001563019104_p14490517115910"></a>561107</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1689762212563"><a name="zh-cn_topic_0000001563019104_p1689762212563"></a><a name="zh-cn_topic_0000001563019104_p1689762212563"></a>内部异常：aclnn API内部发生异常，没有检测到需要配置的环境变量ASCEND_OPP_PATH。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row15897112218562"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p188971322185614"><a name="zh-cn_topic_0000001563019104_p188971322185614"></a><a name="zh-cn_topic_0000001563019104_p188971322185614"></a>ACLNN_ERR_INNER_LOAD_JSON_FAILED</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p175481975913"><a name="zh-cn_topic_0000001563019104_p175481975913"></a><a name="zh-cn_topic_0000001563019104_p175481975913"></a>561108</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1389713225560"><a name="zh-cn_topic_0000001563019104_p1389713225560"></a><a name="zh-cn_topic_0000001563019104_p1389713225560"></a>内部异常：aclnn API内部发生异常，加载算子kernel库中算子信息JSON文件失败。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row98304266565"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p88311261563"><a name="zh-cn_topic_0000001563019104_p88311261563"></a><a name="zh-cn_topic_0000001563019104_p88311261563"></a>ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1312692118598"><a name="zh-cn_topic_0000001563019104_p1312692118598"></a><a name="zh-cn_topic_0000001563019104_p1312692118598"></a>561109</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1815317191816"><a name="zh-cn_topic_0000001563019104_p1815317191816"></a><a name="zh-cn_topic_0000001563019104_p1815317191816"></a>内部异常：aclnn API内部发生异常，加载算子kernel库中算子信息JSON文件的某个字段失败。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row68311826135612"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p1783182615617"><a name="zh-cn_topic_0000001563019104_p1783182615617"></a><a name="zh-cn_topic_0000001563019104_p1783182615617"></a>ACLNN_ERR_INNER_JSON_FORMAT_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1457642315912"><a name="zh-cn_topic_0000001563019104_p1457642315912"></a><a name="zh-cn_topic_0000001563019104_p1457642315912"></a>561110</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p183162616566"><a name="zh-cn_topic_0000001563019104_p183162616566"></a><a name="zh-cn_topic_0000001563019104_p183162616566"></a>内部异常：aclnn API内部发生异常，算子kernel库中算子信息JSON文件的format填写为非法值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row7831122615568"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p1621113315313"><a name="zh-cn_topic_0000001563019104_p1621113315313"></a><a name="zh-cn_topic_0000001563019104_p1621113315313"></a>ACLNN_ERR_INNER_JSON_DTYPE_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1649183615914"><a name="zh-cn_topic_0000001563019104_p1649183615914"></a><a name="zh-cn_topic_0000001563019104_p1649183615914"></a>561111</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1831152645612"><a name="zh-cn_topic_0000001563019104_p1831152645612"></a><a name="zh-cn_topic_0000001563019104_p1831152645612"></a>内部异常：aclnn API内部发生异常，算子kernel库中算子信息JSON文件的dtype填写为非法值。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row149375307560"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p993717301567"><a name="zh-cn_topic_0000001563019104_p993717301567"></a><a name="zh-cn_topic_0000001563019104_p993717301567"></a>ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p112821373599"><a name="zh-cn_topic_0000001563019104_p112821373599"></a><a name="zh-cn_topic_0000001563019104_p112821373599"></a>561112</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p29371307569"><a name="zh-cn_topic_0000001563019104_p29371307569"></a><a name="zh-cn_topic_0000001563019104_p29371307569"></a>内部异常：aclnn API内部发生异常，没有加载到算子的二进制kernel库。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row7937183017565"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p1759793716313"><a name="zh-cn_topic_0000001563019104_p1759793716313"></a><a name="zh-cn_topic_0000001563019104_p1759793716313"></a>ACLNN_ERR_INNER_OP_FILE_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p10866237205919"><a name="zh-cn_topic_0000001563019104_p10866237205919"></a><a name="zh-cn_topic_0000001563019104_p10866237205919"></a>561113</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1893763018567"><a name="zh-cn_topic_0000001563019104_p1893763018567"></a><a name="zh-cn_topic_0000001563019104_p1893763018567"></a>内部异常：aclnn API内部发生异常，加载算子JSON文件字段时，发生异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row793723015614"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p159371730135616"><a name="zh-cn_topic_0000001563019104_p159371730135616"></a><a name="zh-cn_topic_0000001563019104_p159371730135616"></a>ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p19606143813591"><a name="zh-cn_topic_0000001563019104_p19606143813591"></a><a name="zh-cn_topic_0000001563019104_p19606143813591"></a>561114</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1693793095614"><a name="zh-cn_topic_0000001563019104_p1693793095614"></a><a name="zh-cn_topic_0000001563019104_p1693793095614"></a>内部异常：aclnn API内部发生异常，算子的属性个数与算子信息JSON中不一致，超过了JSON中指定的attr个数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row16937203010566"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p593714304562"><a name="zh-cn_topic_0000001563019104_p593714304562"></a><a name="zh-cn_topic_0000001563019104_p593714304562"></a>ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1417113925912"><a name="zh-cn_topic_0000001563019104_p1417113925912"></a><a name="zh-cn_topic_0000001563019104_p1417113925912"></a>561115</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p293816300566"><a name="zh-cn_topic_0000001563019104_p293816300566"></a><a name="zh-cn_topic_0000001563019104_p293816300566"></a>内部异常：aclnn API内部发生异常，算子的属性个数与算子信息JSON中不一致，少于JSON中指定的attr个数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row16937203010566"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p593714304562"><a name="zh-cn_topic_0000001563019104_p593714304562"></a><a name="zh-cn_topic_0000001563019104_p593714304562"></a>ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1417113925912"><a name="zh-cn_topic_0000001563019104_p1417113925912"></a><a name="zh-cn_topic_0000001563019104_p1417113925912"></a>561116</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p293816300566"><a name="zh-cn_topic_0000001563019104_p293816300566"></a><a name="zh-cn_topic_0000001563019104_p293816300566"></a>内部异常：aclnn API内部发生异常，算子的输入个数超出32的限制。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row16937203010566"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p593714304562"><a name="zh-cn_topic_0000001563019104_p593714304562"></a><a name="zh-cn_topic_0000001563019104_p593714304562"></a>ACLNN_ERR_INNER_INPUT_JSON_IS_NULL</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1417113925912"><a name="zh-cn_topic_0000001563019104_p1417113925912"></a><a name="zh-cn_topic_0000001563019104_p1417113925912"></a>561117</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p293816300566"><a name="zh-cn_topic_0000001563019104_p293816300566"></a><a name="zh-cn_topic_0000001563019104_p293816300566"></a>内部异常：aclnn API内部发生异常，算子信息JSON文件信息描述有缺失。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row16937203010566"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p593714304562"><a name="zh-cn_topic_0000001563019104_p593714304562"></a><a name="zh-cn_topic_0000001563019104_p593714304562"></a>ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1417113925912"><a name="zh-cn_topic_0000001563019104_p1417113925912"></a><a name="zh-cn_topic_0000001563019104_p1417113925912"></a>561118</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p293816300566"><a name="zh-cn_topic_0000001563019104_p293816300566"></a><a name="zh-cn_topic_0000001563019104_p293816300566"></a>内部异常：aclnn API内部发生异常，解析静态二进制JSON文件中的workspace信息时，发生异常。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row16937203010566"><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p593714304562"><a name="zh-cn_topic_0000001563019104_p593714304562"></a><a name="zh-cn_topic_0000001563019104_p593714304562"></a>ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1417113925912"><a name="zh-cn_topic_0000001563019104_p1417113925912"></a><a name="zh-cn_topic_0000001563019104_p1417113925912"></a>561119</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p293816300566"><a name="zh-cn_topic_0000001563019104_p293816300566"></a><a name="zh-cn_topic_0000001563019104_p293816300566"></a>内部异常：aclnn API内部发生异常，解析静态二进制JSON文件中的核数使用信息时，发生异常。</p>
</td>
</tr>
</tbody>
</table>
