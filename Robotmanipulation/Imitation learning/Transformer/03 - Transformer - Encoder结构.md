![[Pasted image 20250806200713.png]]

* 上图红色部分是 Transformer 的 Encoder block 结构，可以看到是由 Multi-Head Attention, **Add & Norm, Feed Forward, Add & Norm** 组成的。刚刚已经了解了 Multi-Head Attention 的计算过程，现在了解一下 Add & Norm 和 Feed Forward 部分

#### 4.1 Add & Norm
* Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：
* $$ LayerNorm(X + MultiHeadAttention(X))$$
* $$LayerNorm(X + FeedForward(X))$$
* 其中${X}$ 表示  *Multi-Head Attention 或者 Feed Forward 的输入*    
* $MultiHeadAttention(X)$  和 $FeedForward(X)$ 表示输出 (输出与输入 $X$ 维度是一样的，所以可以相加)

* **Add**指 **X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到
* ![[Pasted image 20250806201555.png]]
* **Norm**指 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛


#### 4.2 Feed Forward
* Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下
	* **X**是输入，Feed Forward 最终得到的输出矩阵的维度与**X**一致。
* $$ max(0, XW_1 + b_1)W_2 + b_2$$


#### 4.3 组成 Encoder
* 通过上面描述的 Multi-Head Attention, Feed Forward, Add & Norm 就可以构造出一个 Encoder block，Encoder block 接收输入矩阵  ，并输出一个矩阵  。通过多个 Encoder block 叠加就可以组成 Encoder。