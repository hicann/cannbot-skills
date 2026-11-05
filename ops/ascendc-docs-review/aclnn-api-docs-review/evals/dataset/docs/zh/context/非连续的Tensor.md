# 非连续的Tensor

目前大部分算子API的输入aclTensor支持“**非连续的Tensor**”，即一个Tensor可以通过\(shape, strides, offset\)表示。

说明：创建aclTensor可通过[《算子库接口》](https://hiascend.com/document/redirect/CannCommunityOplist)中“公共接口>aclCreateTensor”实现。

## 示例1

例如现有一个shape=\(6, 5\)、strides=\(10, 1\)、offset=22的Tensor，其内存排布如下：
> a<sub>0,0</sub> , a<sub>0,1</sub> , a<sub>0,2</sub> , a<sub>0,3</sub> , a<sub>0,4</sub> , a<sub>0,5</sub> , a<sub>0,6</sub> , a<sub>0,7</sub> , a<sub>0,8</sub> , a<sub>0,9</sub>  
> a<sub>1,0</sub> , a<sub>1,1</sub> , a<sub>1,2</sub> , a<sub>1,3</sub> , a<sub>1,4</sub> , a<sub>1,5</sub> , a<sub>1,6</sub> , a<sub>1,7</sub> , a<sub>1,8</sub> , a<sub>1,9</sub>  
> a<sub>2,0</sub> , a<sub>2,1</sub> , **a<sub>2,2</sub> , a<sub>2,3</sub> , a<sub>2,4</sub> , a<sub>2,5</sub> , a<sub>2,6</sub>** , a<sub>2,7</sub> , a<sub>2,8</sub> , a<sub>2,9</sub>  
> a<sub>3,0</sub> , a<sub>3,1</sub> , **a<sub>3,2</sub> , a<sub>3,3</sub> , a<sub>3,4</sub> , a<sub>3,5</sub> , a<sub>3,6</sub>** , a<sub>3,7</sub> , a<sub>3,8</sub> , a<sub>3,9</sub>  
> a<sub>4,0</sub> , a<sub>4,1</sub> , **a<sub>4,2</sub> , a<sub>4,3</sub> , a<sub>4,4</sub> , a<sub>4,5</sub> , a<sub>4,6</sub>** , a<sub>4,7</sub> , a<sub>4,8</sub> , a<sub>4,9</sub>  
> a<sub>5,0</sub> , a<sub>5,1</sub> , **a<sub>5,2</sub> , a<sub>5,3</sub> , a<sub>5,4</sub> , a<sub>5,5</sub> , a<sub>5,6</sub>** , a<sub>5,7</sub> , a<sub>5,8</sub> , a<sub>5,9</sub>  
> a<sub>6,0</sub> , a<sub>6,1</sub> , **a<sub>6,2</sub> , a<sub>6,3</sub> , a<sub>6,4</sub> , a<sub>6,5</sub> , a<sub>6,6</sub>** , a<sub>6,7</sub> , a<sub>6,8</sub> , a<sub>6,9</sub>  
> a<sub>7,0</sub> , a<sub>7,1</sub> , **a<sub>7,2</sub> , a<sub>7,3</sub> , a<sub>7,4</sub> , a<sub>7,5</sub> , a<sub>7,6</sub>** , a<sub>7,7</sub> , a<sub>7,8</sub> , a<sub>7,9</sub>  
> a<sub>8,0</sub> , a<sub>8,1</sub> , a<sub>8,2</sub> , a<sub>8,3</sub> , a<sub>8,4</sub> , a<sub>8,5</sub> , a<sub>8,6</sub> , a<sub>8,7</sub> , a<sub>8,8</sub> , a<sub>8,9</sub>  
> a<sub>9,0</sub> , a<sub>9,1</sub> , a<sub>9,2</sub> , a<sub>9,3</sub> , a<sub>9,4</sub> , a<sub>9,5</sub> , a<sub>9,6</sub> , a<sub>9,7</sub> , a<sub>9,8</sub> , a<sub>9,9</sub>  

即该Tensor排布如上图的深色位置。这个完整的Tensor在内存排布上是不连续的，strides描述Tensor维度上相邻两个元素的间隔，如果在维度1上的stride为1，该维度是连续的；如果在维度0上的stride为10，那么相邻的元素间隔10个元素，即非连续。offset表示这个Tensor的首元素相对addr的偏移。

## 示例2

例如现有一个shape=\(4, 3\)、strides=\(20, 2\)、offset=22的Tensor，其内存排布如下：

> a<sub>0,0</sub> , a<sub>0,1</sub> , a<sub>0,2</sub> , a<sub>0,3</sub> , a<sub>0,4</sub> , a<sub>0,5</sub> , a<sub>0,6</sub> , a<sub>0,7</sub> , a<sub>0,8</sub> , a<sub>0,9</sub>  
> a<sub>1,0</sub> , a<sub>1,1</sub> , a<sub>1,2</sub> , a<sub>1,3</sub> , a<sub>1,4</sub> , a<sub>1,5</sub> , a<sub>1,6</sub> , a<sub>1,7</sub> , a<sub>1,8</sub> , a<sub>1,9</sub>  
> a<sub>2,0</sub> , a<sub>2,1</sub> , **a<sub>2,2</sub>** , a<sub>2,3</sub> , **a<sub>2,4</sub>** , a<sub>2,5</sub> , **a<sub>2,6</sub>** , a<sub>2,7</sub> , a<sub>2,8</sub> , a<sub>2,9</sub>  
> a<sub>3,0</sub> , a<sub>3,1</sub> , a<sub>3,2</sub> , a<sub>3,3</sub> , a<sub>3,4</sub> , a<sub>3,5</sub> , a<sub>3,6</sub> , a<sub>3,7</sub> , a<sub>3,8</sub> , a<sub>3,9</sub>  
> a<sub>4,0</sub> , a<sub>4,1</sub> , **a<sub>4,2</sub>** , a<sub>4,3</sub> , **a<sub>4,4</sub>** , a<sub>4,5</sub> , **a<sub>4,6</sub>** , a<sub>4,7</sub> , a<sub>4,8</sub> , a<sub>4,9</sub>  
> a<sub>5,0</sub> , a<sub>5,1</sub> , a<sub>5,2</sub> , a<sub>5,3</sub> , a<sub>5,4</sub> , a<sub>5,5</sub> , a<sub>5,6</sub> , a<sub>5,7</sub> , a<sub>5,8</sub> , a<sub>5,9</sub>  
> a<sub>6,0</sub> , a<sub>6,1</sub> , **a<sub>6,2</sub>** , a<sub>6,3</sub> , **a<sub>6,4</sub>** , a<sub>6,5</sub> , **a<sub>6,6</sub>** , a<sub>6,7</sub> , a<sub>6,8</sub> , a<sub>6,9</sub>  
> a<sub>7,0</sub> , a<sub>7,1</sub> , a<sub>7,2</sub> , a<sub>7,3</sub> , a<sub>7,4</sub> , a<sub>7,5</sub> , a<sub>7,6</sub> , a<sub>7,7</sub> , a<sub>7,8</sub> , a<sub>7,9</sub>  
> a<sub>8,0</sub> , a<sub>8,1</sub> , **a<sub>8,2</sub>** , a<sub>8,3</sub> , **a<sub>8,4</sub>** , a<sub>8,5</sub> , **a<sub>8,6</sub>** , a<sub>8,7</sub> , a<sub>8,8</sub> , a<sub>8,9</sub>  
> a<sub>9,0</sub> , a<sub>9,1</sub> , a<sub>9,2</sub> , a<sub>9,3</sub> , a<sub>9,4</sub> , a<sub>9,5</sub> , a<sub>9,6</sub> , a<sub>9,7</sub> , a<sub>9,8</sub> , a<sub>9,9</sub>  

即该Tensor排布如上图的深色位置。这个完整的Tensor在内存排布上是不连续的，strides描述Tensor维度上相邻两个元素的间隔，如果在维度1上的stride为2，该维度上间隔1个元素；如果在维度0上的stride为20，那么相邻的元素间隔20个元素，即非连续。offset表示这个Tensor的首元素相对addr的偏移。
