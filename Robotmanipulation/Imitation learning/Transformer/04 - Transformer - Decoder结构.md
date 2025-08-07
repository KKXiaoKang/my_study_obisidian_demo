![[Pasted image 20250806203107.png]]

> 上图红色部分为 Transformer 的 Decoder block 结构，与 Encoder block 相似，但是存在一些区别：
- 包含两个 Multi-Head Attention 层。
- 第一个 Multi-Head Attention 层采用了 Masked 操作。
- 第二个 Multi-Head Attention 层的**K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 Decoder block 的输出计算。
- 最后有一个 Softmax 层计算下一个翻译单词的概率。


### 5.1 第一个 Multi-Head Attention
* Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。
![[Pasted image 20250806210335.png]]

*第一步*  ：是 Decoder 的输入矩阵和  **Mask**  矩阵，输入矩阵包含 "I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，**Mask** 是一个 5×5 的矩阵。在 **Mask** 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。
![[Pasted image 20250806210700.png]]![[Pasted image 20250806210756.png]]
![[Pasted image 20250806211022.png]]

![[Pasted image 20250806211046.png]]

![[Pasted image 20250806211104.png]]


### 5.2 第二个 Multi-Head Attention
![[Pasted image 20250806211303.png]]


### 5.3 Softmax 预测输出单词
![[Pasted image 20250806211326.png]]

![[Pasted image 20250806211335.png]]

![[Pasted image 20250807094952.png]]