# ThreeInterpolate

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: performs weighted linear feature interpolation based on the 3 nearest neighbors. Given the feature descriptors of known feature points, the indexes of the 3 nearest neighbors of each point to be interpolated, and the corresponding weights, the operator computes the weighted sum of the 3 nearest neighbor features for each feature channel of each point to be interpolated.

- Calculation formula:

$$
y[b, n, c] = \sum_{k=0}^{2} weight[b, n, k] \times features[b, idx[b, n, k], c]
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>features</td>
      <td>Input</td>
      <td>Set of known feature points, with the shape (B, M, C). B = batch, M = number of known points, C = number of channels.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>Input</td>
      <td>Indexes of the 3 nearest neighbors, with the shape (B, N, 3). N is the number of points to be interpolated, each row contains 3 index values, and the value range is [0, M).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Weights of the 3 nearest neighbors, with the shape (B, N, 3). The weights correspond to idx and are usually obtained through reciprocal distance normalization.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Features after interpolation, with the shape (B, N, C). B = batch, N = number of points to be interpolated, C = number of channels.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- features and weight must have the same dtype (both float32 or both float16), and the dtype of y must be the same as that of features.
- idx can independently select int32 or int64.
- All input tensors must be contiguous.
- features, idx, and weight must all be 3D tensors; the B dimension of features must be consistent with those of idx and weight, the N dimension of idx must be consistent with that of weight, and the last dimension of idx and weight must be 3.
- The values of idx must be in the range [0, M), and the caller must ensure that the values do not go out of the range.
- When the output is not empty (B, N, and C are all greater than 0), M must be greater than 0; when M = 0, idx has no valid values, and the operator reports an error.
- Dimension upper limits: B, M, C, and N do not exceed 2^32-1, N×C does not exceed 2^32-1, and B×N×3 and B×M×C do not exceed 2^64-1.
- Empty tensors are supported: when any of B, N, and C is 0, the output is an empty tensor and no calculation is performed; the scenario where the M dimension of features is 0 but the output is not empty is not supported, and the operator reports an error.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph mode</td>
    <td><a href="./examples/arch35/test_geir_three_interpolate.cpp">test_geir_three_interpolate</a></td>
    <td>Call the ThreeInterpolate operator by building a graph through <a href="./op_graph/three_interpolate_proto.h">operator IR</a>, and refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete compilation and verification.</td>
  </tr>
</tbody>
</table>
