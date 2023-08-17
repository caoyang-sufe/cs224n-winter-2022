

> - **英文标题**：<font face=times new roman size=4>*Scaling Laws for Neural Language Models*</font>
> - **中文标题**：自然语言模型的尺度法则
> - **下载链接**：[<font face=times new roman size=4>*arxiv@2001.08361*</font>](https://arxiv.org/abs/2001.08361)

----

# 序言

本文是[CS224N WINTER 2022 （六）前沿问题探讨（QA、NLG、知识集成与检索、Coreference）](https://caoyang.blog.csdn.net/article/details/125099126)的<font face=times new roman size=4 color=red><i>lecture 17</i></font>内容的补充（只有一篇推荐阅读，就是这一篇）。

本文属于经验性实验分析论文，大部分结论缺少理论推导，不过神经网络这块本身就很少涉及理论推导，笔者觉得很多结论还是很受用的。

----

[toc]

----

## 摘要 <font face=times new roman><i>Abstract</i></font>

- 本文研究的是基于交叉熵损失的语言模型性能的**经验尺度法则**（<font face=times new roman size=4 color=red><i>empirical</i></font> <font face=times new roman size=4 color=red><i>scaling</i></font> <font face=times new roman size=4 color=red><i>laws</i></font>）。

- 本文使用**乘幂法则**（<font face=times new roman size=4 color=red><i>power laws</i></font>）来刻画模型训练的交叉熵与**模型尺寸**（<font face=times new roman size=4 color=red><i>model size</i></font>，即模型参数量）、数据集量、训练计算量的关系。一些其他的模型架构细节（如网络宽度或深度）对交叉熵损失的影响很小。

- 本文利用简单的方程来刻画过拟合程度对模型尺寸或数据集量的依赖性，以及模型训练速度对模型尺寸的依赖性，这些关系可用于计算资源的合理分配。

- 本文发现大模型使用样本的效率显著更高，因此最优的高效训练方式是在**中等**（<font face=times new roman size=4 color=red><i>modest</i></font>）数据集上训练**超大模型**（<font face=times new roman size=4 color=red><i>very large models</i></font>），并在**显著收敛前提前停止**（<font face=times new roman size=4 color=red><i>stop significantly before convergence</i></font>）。

----
## 1 导论 <font face=times new roman><i>Introduction</i></font>

语言模型的性能可能取决于**模型架构**、**模型尺寸**、**训练计算量**、**训练数据量**，本文将对这些因素在<font face=times new roman size=4 color=red><i>Transformer</i></font>模型架构上进行逐一分析。

----

### 1.1 概述 <font face=times new roman><i>Summary</i></font>

本文在<font face=times new roman size=4 color=red><i>Transformer</i></font>语言模型上实验的关键成果概述：

1. **模型性能与尺度**（<font face=times new roman size=4 color=red><i>scale</i></font>）**强相关，与模型形态**（<font face=times new roman size=4 color=red><i>model shape</i></font>，即网络层的宽度和深度等）**弱相关**：<font face=times new roman size=4 color=red><i>Section 3</i></font>

   **尺度**由三部分组成（模型参数量$N$，训练数据量$D$，训练计算量$C$），在合理范围内，模型性能与网络层的宽度或深度关联性很低。

2. **平滑乘幂法则**（<font face=times new roman size=4 color=red><i>smooth power laws</i></font>）：<font face=times new roman size=4 color=red><i>Section 3</i></font>

   在固定其中一个**尺度因素**（<font face=times new roman size=4 color=red><i>scale factor</i></font>），其余两个尺度因素不加限制的条件下，模型性能与三个尺度因素（$N,D,C$）具有乘幂法则关系（<font face=times new roman size=4 color=red><i>Figure 1</i></font>）。

   > <font face=times new roman size=4 color=red><i>Figure 1</i></font>：
   >
   > - 横轴表示三个尺度因素，纵轴表示对测试集上的损失值（用于衡量模型性能）；
   > - 途中可以看出三个尺度因素对模型性能的近似是对数平滑的，具体拟合方程已在图中标注；

   ![f1](https://img-blog.csdnimg.cn/8a007eab06af44a08090c6112ade515f.png#pic_center)

3. **过拟合的普遍性**（<font face=times new roman size=4 color=red><i>Universality of overfitting</i></font>）：<font face=times new roman size=4 color=red><i>Section 4</i></font>

   只要同时扩大$N,D$，模型性能（指损失值）自然会提升，但是如果固定$N,D$中的某一个的数值，然后扩大另一个的数值，模型性能反而会衰减（即发生过拟合）。具体而言，性能衰减程度与$N^{0.74}/D$的比率相关，即如果扩大模型尺寸八倍，就需要将训练数据来扩大五倍。

4. **模型训练的一般性**（<font face=times new roman size=4 color=red><i>Universality of training</i></font>）：<font face=times new roman size=4 color=red><i>Section 5</i></font>

   **训练曲线**（<font face=times new roman size=4 color=red><i>training curves</i></font>，即损失值曲线）是可以通过**乘幂法则**进行预测，且近似可以认为与模型尺寸独立（即只考察与$N,D$的关系），通过训练曲线早期走势可以推断得到最终训练损失值能够收敛到何处。

5. **迁移提升测试性能**（<font face=times new roman size=4 color=red><i>Transfer improves with test performance</i></font>）：<font face=times new roman size=4 color=red><i>Secition 3.2.2</i></font>

   当我们将模型在与训练数据分布不同的测试数据上进行评估时，结果表明尽管存在一定的模型性能损失，但是整体上来看如果验证集上能够得到更好的模型性能，测试集上的模型性能也会有所提升。

6. **样本高效性**（<font face=times new roman size=4 color=red><i>sample efficient</i></font>）：

   大模型比小模型对样本信息的利用更高效，具体而言在更少的优化迭代次数时即可达到同等的模型性能（<font face=times new roman size=4 color=red><i>Figure 2</i></font>）且需要更少的训练数据（<font face=times new roman size=4 color=red><i>Figure 4</i></font>）。

   > <font face=times new roman size=4 color=red><i>Figure 2</i></font>：
   >
   > - 纵轴表示模型性能（即测试损失值，越低越好），左图横轴表示训练数据量，右图横轴表示训练时间；
   > - 图中曲线颜色越深表明模型尺寸（即模型参数量）越少，说明要达到同样的测试损失值需要花费更多的训练数据（左图），右图其实说明的是大模型尽管耗时更长，但是模型性能的上限更高；
   >
   > <font face=times new roman size=4 color=red><i>Figure 4</i></font>：
   >
   > - 纵轴表示模型性能（即测试损失值，越低越好），左图横轴表示训练数据量，右图横轴表示训练时间；
   > - 图中曲线颜色越深表明模型尺寸（即模型参数量）越小，两图表明模型尺寸越大、训练数据量越大、训练时间越长，模型评估越好（符合直觉）；

   ![f2](https://img-blog.csdnimg.cn/63184c8ec495417eac5e8240774911cd.png#pic_center)

   ![f4](https://img-blog.csdnimg.cn/74a38a1b121741a7860ff139310fedc6.png#pic_center)

   ----

7. **收敛是低效的**（<font face=times new roman size=4 color=red><i>convergence is inefficient</i></font>）：<font face=times new roman size=4 color=red><i>Section 6</i></font>

   在固定**算力预算**（<font face=times new roman size=4 color=red><i>compute budget</i></font>）$C$，且不限制模型尺寸$N$和训练数据量$D$的假定下，获得最优模型性能的训练方式是**训练超大模型并在显著收敛前停止**（<font face=times new roman size=4 color=red><i>Figure 3</i></font>），这种方式要比训练小模型直至收敛要样本高效得多。此外，实验表明$D\sim C^{0.27}$，即随着算力提升，需要的训练数据并没有增长太多。

   > <font face=times new roman size=4 color=red><i>Figure 3</i></font>：
   >
   > - 横轴表示算力，纵轴表示对模型性能的贡献提升，整体是一个累积面积图；
   > - 随着算力提升，可以考虑训练**更大的模型**，或使用**更大的批训练量**，或训练**迭代更多的轮次**，实验表明应当将更多的算力应用到训练大模型上；

   ![f3](https://img-blog.csdnimg.cn/839520ce29a8417181ab5c0c8863c1a3.png#pic_center)

   ----

8. **最优批训练量**（<font face=times new roman size=4 color=red><i>optimal batch size</i></font>）：<font face=times new roman size=4 color=red><i>Section 5.1</i></font>

   根据参考文献[<font face=times new roman size=4 color=red><i>MKAT18</i></font>](https://arxiv.org/abs/1812.06162)，理想的批训练量近似为**损失函数值的幂**（<font face=times new roman size=4 color=red><i>power of the loss</i></font>），并且随着**梯度噪声尺度**（<font face=times new roman size=4 color=red><i>gradient noise scale</i></font>）不断调整。本文实验中使用的最大模型，每次可以批训练十亿到二十亿量级的分词。

----

### 1.2 尺度法则概述  <font face=times new roman><i>Summary of Scaling Laws</i></font>

用于构建自回归语言模型的<font face=times new roman size=4 color=red><i>Transformer</i></font>测试损失值可以通过**乘幂法则**进行预测（利用<font face=times new roman size=4 color=red><i>Figure 1</i></font>中关于$N,D,C$的拟合方程）：

1. **测试损失值与非嵌入参数量的关系**：限制模型参数量，在充分大的训练集上训练至损失值收敛

   $$
   L(N)=\left(\frac{N_c}N\right)^{\alpha_N};\quad\alpha_N\sim 0.076;\quad N_c\sim 8.8\times10^{13}\quad\text{(non-embedding parameters)}\tag{1.1}
   $$

2. **测试损失值与训练数据量的关系**：限制训练数据量，在大模型上训练至早停（<font face=times new roman size=4 color=red><i>early stopping</i></font>）

   $$
   L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D};\quad\alpha_D\sim0.095;\quad D_c\sim5.4\times 10^{13}\quad(\text{tokens})\tag{1.2}
   $$

3. **测试损失值与训练时间的关系**：限制训练计算量，在充分大的训练集上训练最优尺寸的模型（以充分小的批训练量）

   $$
   L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}};\quad\alpha_C^{\min}\sim0.050;\quad C_c^{\min}\sim3.1\times10^8\quad(\text{PF-days})\tag{1.3}
   $$

   **笔者注**：注意式$(5.5)$中给出的最优的算力配置要求在**充分小的批训练量**下进行模型训练。

----

其中，尺度系数$\alpha_N,\alpha_D,\alpha_C^{\min}$表明模型性能与对应三个因素的变化尺度关系，比如模型参数量翻倍，测试损失值将会缩小至原先的$2^{-\alpha_N}=0.95$倍。常数$N_c,D_c,C_c^{\min}$的大小取决于**词汇总量**（<font face=times new roman size=4 color=red><i>vocabulary size</i></font>）与**分词**（<font face=times new roman size=4 color=red><i>tokenization</i></font>），无实际含义。

从<font face=times new roman size=4 color=red><i>Figure 1</i></font>中可以发现，式$(1.1)$关于$N$的拟合方程定义域横跨六个数量级，式$(1.2)$关于$D$的拟合方程定义域横跨两个数量级，式$(1.3)$关于$C_{\min}$的拟合方程定义域横跨八个数量级。**模型形态**以及其他<font face=times new roman size=4 color=red><i>Transformer</i></font>的参数（网络层深度与宽度，自注意力头的数量）对测试损失值的影响很小。

本文通过在参考文献[<font face=times new roman size=4 color=red><i>RWC+19</i></font>]（暂未找到链接）的<font face=times new roman size=4 color=red><i>WebText2</i></font>数据集上进行实验得出上述数值结论。

进一步地，本文发现更深入的尺度法则结论：

1. **关键批训练量**（<font face=times new roman size=4 color=red><i>critical batch size</i></font>）关于测试损失值$L$也近似服从乘幂法则：

   $$
   B_{\rm crit}(L)=\frac{B_*}{L^{1/\alpha_B}};\quad B_*\sim 2\times10^8\text{ tokens};\quad a_B\sim 0.21\tag{1.4}
   $$

2. 式$(1.1)$与式$(1.2)$表明，若增加模型尺寸，则应当以**次线性**（<font face=times new roman size=4 color=red><i>sublinearly</i></font>）速度增加训练数据量（$D\propto N^{\alpha_N/\alpha_D}\sim N^{0.74}$），事实上本文发现可以用另一个方程来刻画$N,D$对$L$的联合影响（如<font face=times new roman size=4 color=red><i>Figure 4</i></font>左图所示）：

   $$
   L(N,D)=\left[\left(\frac{N_c}N\right)^{\alpha_N/\alpha_D}+\frac{D_c}D\right]^{\alpha_D}\tag{1.5}
   $$

   本文猜想式$(1.5)$的形式可以推广到其他**生成建模任务**（<font face=times new roman size=4 color=red><i>generative modeling task</i></font>）。

3. 同理在给定有限的训练数据量的条件下，也可以刻画$N,S$对$L$的联合影响（如<font face=times new roman size=4 color=red><i>Figure 4</i></font>右图所示）：

   $$
   L(N,S)=\left(\frac{N_c}N\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}(S)}\right)^{\alpha_S};\quad S_c\sim2.1\times10^3;\quad \alpha_S\sim0.76\tag{1.6}
   $$
   
   其中$S_{\min}(S)$表示式$(5.4)$估计得到的最低优化迭代轮数。

4. 给定**算力预算**$C$且无其他约束的条件下，式$(1.6)$揭示了如下的尺度关系：

   $$
   \begin{aligned}
   N&\propto C^{\alpha_C^{\min}/\alpha_N}\\
   B&\propto C^{\alpha_C^{\min}/\alpha_B}\\
   S&\propto C^{\alpha_C^{\min}/\alpha_S}\\
   D&=B\cdot S
   \end{aligned}\tag{1.7}
   $$
   
   其中：
   
   $$
   \alpha_C^{\min}=\left(\frac1{\alpha_S}+\frac1{\alpha_B}+\frac1{\alpha_N}\right)^{-1}\tag{1.8}
   $$
   
   这与经验上的最优结果（$N\propto C_{\min}^{0.73},B\propto C_{\min}^{0.24},S\propto C_{\min}^{0.03}$）相吻合。

   当**算力预算**$C$增加时，应当尽可能多地将算力分配到增加模型尺寸（如<font face=times new roman size=4 color=red><i>Figure 3</i></font>所示），且此时训练数据量无需增加太多，这间接说明大模型对于训练样本的使用效率要更高。但是实际操作中受硬件限制，往往很难构建超大模型进行训练。

   **笔者注**：这部分结论看起来有点复杂，其实也很好理解，式$(1.7)$对应<font face=times new roman size=4 color=red><i>Figure 3</i></font>中的累积面积图，因为$\alpha_B$与$\alpha_S$显著大于$\alpha_N$，因此增加等量的**算力预算**$C$，根据式$(1.7)$可知$N$同比增加最多，因此尽量将算力资源用在增加模型参数量，而非增加批训练量（这对应式$(1.2)$中使用更多的训练数据）或是迭代轮数（这对应式$(1.3)$中更长的训练时间）。

----

### 1.3 数学标记 <font face=times new roman><i>Notation</i></font>

本文用到的数学标记如下所示：

- $L$：交叉熵损失值；
- $N$：模型参数量，不包括词嵌入与**位置嵌入**（<font face=times new roman size=4 color=red><i>positional embeddings</i></font>）；
- $C$（$\approx6NBS$）：总训练计算量的估计值（不包括嵌入），其中$B$表示批训练量，$S$表示训练优化迭代轮数，单位$\text{PF-days}$表示$10^5\times 24\times 3600=8.64\times 10^{19}$**浮点运算**（<font face=times new roman size=4 color=red><i>floating point operations</i></font>，<font face=times new roman size=4 color=red><i>FLOP</i></font>）；
- $D$：训练数据量（以分词量衡量）；
- $B_{\rm crit}$：参考文献[<font face=times new roman size=4 color=red><i>MKAT18</i></font>](https://arxiv.org/abs/1812.06162)提出的**关键批训练量**（本文<font face=times new roman size=4 color=red><i>Section 5.1</i></font>中有定义），以**关键批训练量**进行批训练能够近似在时间与计算效率之间取得最优的平衡；
- $C_{\min}$：可以达到给定损失值的最小训练计算量（不包括嵌入）估计值，这个数值是在批训练量远小于**关键批训练量**时会使用的训练计算量；
- $S_{\min}$：可以达到给定损失值的最小训练优化迭代轮数估计值，这个数值也是在批训练量远小于**关键批训练量**时会使用的迭代轮数；
- $\alpha_X$：乘幂法则的指数系数值，$X$可以是$N,D,C,S,B,C^{\min}$之一，$L(X)\propto1/X^{\alpha_X}$；
- $n_{\rm ctx}$：输入上下文序列的长度，本文不作特殊说明则默认$n_{\rm ctx}=1024$；
- $n_{\rm vocab}$：词汇表中的分词数量；

----
## 2 研究背景与研究方法 <font face=times new roman><i>Background and Methods</i></font>

- 本文在<font face=times new roman size=4 color=red><i>WebText2</i></font>数据集上训练语言模型，使用的是参考文献[<font face=times new roman size=4 color=red><i>SHB15</i></font>](https://arxiv.org/abs/1508.07909)中的**成对字节编码**（<font face=times new roman size=4 color=red><i>byte-pair encoding</i></font>）进行分词，得到的词汇表总量为$n_{\rm vocab}=50257$。
- 本文在平均$2^{10}=1024$个分词上下文上计算自回归对数似然（即交叉熵损失值），具体在<font face=times new roman size=4 color=red><i>WebText2</i></font>和具有不同文本分布的数据集上计算测试损失值。
- 本文设计的语言模型援用参考文献[<font face=times new roman size=4 color=red><i>VSP+17</i></font>](https://arxiv.org/abs/1706.03762)（<font face=times new roman size=4 color=red><i>Transformer</i></font>提出文<font face=times new roman size=4 color=red><i>Attention is all you need</i></font>）的解码器结构，此外还训练<font face=times new roman size=4 color=red><i>LSTM</i></font>模型和参考文献[<font face=times new roman size=4 color=red><i>DGV+18</i></font>](https://arxiv.org/abs/1807.03819)的通用<font face=times new roman size=4 color=red><i>Transformer</i></font>模型作为对比。

----

### 2.1 <font face=times new roman><i>Transformer</i>的参数与计算尺度 <i>Parameter and Compute Scaling of Transformers</i></font>

本节内容要求熟悉掌握<font face=times new roman size=4 color=red><i>Transformer</i></font>架构，可见笔者对<font face=times new roman size=4 color=red><i>CS224N</i></font>中相关章节摘注的[<font color=red face=times new roman size=4><i>blog</i></font>](https://caoyang.blog.csdn.net/article/details/125072701)。

首先考察模型尺寸$N$。<font face=times new roman size=4 color=red><i>Transformer</i></font>模型架构的参数包括：

- $n_{\rm layer}$：网络层数；
- $d_{\rm model}$：**残差流**（<font face=times new roman size=4 color=red><i>residual stream</i></font>）维度；
- $d_{\rm ff}$：**前馈层**（<font face=times new roman size=4 color=red><i>feed-forward layer</i></font>）维度；
- $d_{\rm attn}$：注意力输出维度；
- $n_{\rm headers}$：每层的注意力头数；

则<font face=times new roman size=4 color=red><i>Tranformer</i></font>模型架构的总参数量为：

$$
N\approx2d_{\rm model}n_{
\rm layer}(2d_{\rm attn}+d_{\rm ff})\overset{\rm default}{=}12n_{\rm layers}d^2_{\rm model}\tag{2.1}
$$

式$(2.1)$中统计的是前馈层和注意力机制的参数量，没有包含截距项和其他**次级项**（<font face=times new roman size=4 color=red><i>sub-leading terms</i></font>，如层正则化、激活函数等）的参数量，因此第一个变换是约等号；在<font face=times new roman size=4 color=red><i>Transformer</i></font>提出文中的默认配置为$d_{\rm attn}=d_{\rm model}=d_{\rm ff}/4$，则第二个等号成立。

此外模型还应当包括参数量为$n_{\rm vocab}d_{\rm model}$的词嵌入矩阵、以及$n_{\rm ctx}d_{\rm model}$的位置嵌入矩阵，本文不讨论这些参数量，从后面的结论来看这样可以生成更加简洁的尺度法则（对照<font face=times new roman size=4 color=red><i>Section 1.3</i></font>中变量$N$的说明）。

接下来考察训练计算量$C$。<font face=times new roman size=4 color=red><i>Transformer</i></font>执行依次正向传播需要的计算量为：

$$
C_{\rm forward}\approx 2N+2n_{\rm layer}n_{\rm ctx}d_{\rm model}\tag{2.2}
$$

式$(2.2)$之所以$N$需要加倍是因为矩阵乘法中涉及乘法与加法（举个最简单的例子，$W_{m\times n}x_{n\times 1}$的参数量为$mn$，矩阵乘法共计执行$mn$次乘法和$m(n-1)$次加法，虽然说加法的耗时远低于乘法，因此通常会被忽略，但是本文这么说还是由着他来），具体的参数量与计算量如下表所示：

|                           运算操作                           |                          参数量                           |                    每个单词的浮点运算量                    |
| :----------------------------------------------------------: | :-------------------------------------------------------: | :--------------------------------------------------------: |
|                            嵌入层                            |        $(n_{\rm vocab}+n_{\rm ctx})d_{\rm model}$         |                      $4d_{\rm model}$                      |
|                      注意力机制：$QKV$                       |         $n_{\rm layer}d_{\rm model}3d_{\rm attn}$         |         $2n_{\rm layer}d_{\rm model}3d_{\rm attn}$         |
| 注意力机制：掩藏（<font face=times new roman size=4 color=red><i>Mask</i></font>） |                            $/$                            |          $2n_{\rm layer}n_{\rm ctx}d_{\rm attn}$           |
| 注意力机制：映射（<font face=times new roman size=4 color=red><i>Project</i></font>） |         $2n_{\rm layer}d_{\rm attn}d_{\rm model}$         |          $2n_{\rm layer}d_{\rm attn}d_{\rm embd}$          |
|                            前馈层                            |          $n_{\rm layer}2d_{\rm model}d_{\rm ff}$          |          $2n_{\rm layer}2d_{\rm model}d_{\rm ff}$          |
| 反嵌入层（<font face=times new roman size=4 color=red><i>de-embed</i></font>） |                            $/$                            |               $2d_{\rm model}n_{\rm vocab}$                |
|                           **合计**                           | $N=2d_{\rm model}n_{\rm layer}(2d_{\rm attn}+d_{\rm ff})$ | $C_{\rm forward}=2N+2n_{\rm layer}n_{\rm ctx}d_{\rm attn}$ |

**笔者注**：原文注意力机制映射的浮点运算量感觉是写错了，应该是$2n_{\rm layer}d_{\rm attn}d_{\rm model}$才对（$d_{\rm embd}$就从来没有再提过），另外关于获取浮点运算量的方法，<font face=times new roman size=4 color=red><i>Python</i></font>中有包括<font face=times new roman size=4 color=red><i>flop</i></font>库在内的多个库可以计算<font face=times new roman size=4 color=red><i>PyTorch</i></font>或<font face=times new roman size=4 color=red><i>Tensorflow</i></font>模型的浮点运算量，这个需要的可以自己查看教程学习。

对于$d_{\rm model}>n_{\rm ctx}/12$的情形，**上下文独立的**（<font face=times new roman size=4 color=red><i>context-dependent</i></font>）计算成本占总计算量的比重非常低，因此本文主要研究$d\gg n_{\rm ctx}/12$的模型，即无需将**上下文独立项**包含再训练计算量的估计中。

最后考察<font face=times new roman size=4 color=red><i>Transformer</i></font>执行依次反向传播的计算量，这大约是正向传播计算量的两倍，因此总计算量$C=C_{\rm forward}+C_{\rm backward}\approx 6N$

----

### 2.2 训练步骤 <font face=times new roman><i>Training Procedures</i></font>

- 不作特殊说明，本文训练模型使用<font face=times new roman size=4 color=red><i>Adam</i></font>优化算法，固定迭代$2.5\times10^5$轮，批训练量为$512$个序列（每个序列包含$1024$个分词）。
- 由于内存限制，本文实验中最大的模型使用参考文献[<font color=red face=times new roman size=4><i>SS18</i></font>](https://arxiv.org/abs/1804.04235)中提出的<font face=times new roman size=4 color=red><i>Adafactor</i></font>训练。
- 本文再模型训练中使用了许多学习率规划的方法（详见附录<font face=times new roman size=4 color=red><i>D.6</i></font>），但是实验结果表明收敛时的结果与学习率规划的方法基本无关。不作特殊说明，本文使用的学习率规划方式是前$3000$轮使用**线性热启动**（<font face=times new roman size=4 color=red><i>linear warmup</i></font>），之后以**余弦衰减**（<font face=times new roman size=4 color=red><i>cosine decay</i></font>）至零。

----

### 2.3 数据集 <font face=times new roman><i>Datasets</i></font>

- 本文使用的数据集是<font face=times new roman size=4 color=red><i>WebText2</i></font>（<font face=times new roman size=4 color=red><i>WebText</i></font>的扩展版本），原始的<font face=times new roman size=4 color=red><i>WebText</i></font>数据集来自从<font face=times new roman size=4 color=red><i>Reddit</i></font>上的爬虫，选取的数据范围是$2017$年$12$月所有至少有$3$个**推荐**（<font face=times new roman size=4 color=red><i>karma</i></font>，这表明网民是否认为该链接有趣）的**导出链接**（<font face=times new roman size=4 color=red><i>outbound links</i></font>），本文添加了$2018$年$1$月到$10$月同样类型的筛选链接构成第二版<font face=times new roman size=4 color=red><i>WebText2</i></font>；
- 本文使用<font face=times new roman size=4 color=red><i>Python</i></font>的<font face=times new roman size=4 color=red><i>Newspaper3k</i></font>库对新扩展的链接进行解析，最终得到的数据量包括$20.3\rm M$的文档，共计$96\rm GB$文本，$1.62\times 10^{10}$个单词；
- 本文使用参考文献[<font face=times new roman size=4 color=red><i>RWC+19</i></font>]提出的**可逆分词器**（<font face=times new roman size=4 color=red><i>reversible tokenizer</i></font>）对语料进行分词，共计生成$2.29\times 10^{10}$个**分词**（<font face=times new roman size=4 color=red><i>token</i></font>），其中$6.6\times 10^8$的分词划分作为测试集；
- 本文还在其他类似的语料库上进行迁移测试，如参考文献[<font face=times new roman size=4 color=red><i>ZKZ+15</i></font>](https://arxiv.org/abs/1506.06724)提出的<font face=times new roman size=4 color=red><i>BooksCorpus</i></font>数据集，参考文献[<font face=times new roman size=4 color=red><i>Fou</i></font>](http://commoncrawl.org)中提出的<font face=times new roman size=4 color=red><i>CommonCrawl</i></font>，英文维基百科，以及一系列可以公开获取的电子书。

----

## 3 经验结果与基本乘幂法则 <font face=times new roman><i>Empirical Results and Basic Power Laws</i></font>

为了刻画语言模型性能随尺度的变化，本文训练大量不同尺度的模型：

- 模型尺寸：从$768$到$1.5\rm B$的非嵌入参数量；
- 训练数据量：从$22\rm M$到$23\rm B$的分词量；
- 不同的模型形态：包括网络层深度、宽度、注意力头数、前馈层的维数；
- 上下文序列长度：大多数都是$1024$，本文也对短文本进行实验；
- 批训练量：大多数都是$2^{19}$，为了探寻**关键批训练量**，本文也进行了一定范围的实验；

本节内容以数据展示与经验性推断为主，后面的章节将详细进行理论分析。

----

### 3.1 <font face=times new roman><i>Transformer</i>近似形态与超参数独立性  <i>Approximate Transformer Shape and Hyperparameter Independence</i></font>

- **结论**：若固定模型参数量$N$，则<font face=times new roman size=4 color=red><i>Transformer</i></font>的性能与**形态参数**（<font face=times new roman size=4 color=red><i>shape parameter</i></font>）$n_{\rm layer},n_{\rm head},d_{\rm ff}$近似独立。

- 为了说明上述结论的正确性，本文固定模型参数量$N$，然后只变动单一参数，对于$n_{\rm head}$来说这是最容易的，但是另外两个发生变动则会影响$N$的数值，因此本文在变动$n_{\rm layer}$或$d_{\rm ff}$的同时，会修正另一个参数值以保持$N\approx 12n_{\rm layer}d_{\rm model}^2$

- 参数$n_{\rm layer}$对模型性能的独立性意味着更深的<font face=times new roman size=4 color=red><i>Transformer</i></font>与更浅的<font face=times new roman size=4 color=red><i>Transformer</i></font>的表现是相近的，这正如参考文献[<font face=times new roman size=4 color=red><i>VWB16</i></font>](https://arxiv.org/abs/1605.06431)中的<font face=times new roman size=4 color=red><i>ResNets</i></font>模型所建议的那样，具体结果如<font face=times new roman size=4 color=red><i>Figure 5</i></font>所示：

  > <font face=times new roman size=4 color=red><i>Figure 5</i></font>：
  >
  > - 横轴表示不同参数的变化，左中右分别是$d_{\rm ff},n_{\rm layer},n_{\rm head}$（与$d_{\rm model}$的比值），纵轴表示测试损失值的增幅；
  > - 整体上来看，参数值的变化范围较大，但测试损失值的增幅并不明显；

  ![f5](https://img-blog.csdnimg.cn/6a5b33d33bfc40b7a285300c0bcefb4a.png#pic_center)

----

### 3.2 无嵌入参数量$N$的性能 <font face=times new roman><i>Performance with Non-Embedding Parameter Count</i></font> $N$

> <font face=times new roman size=4 color=red><i>Figure 6</i></font>：
>
> - 纵轴表示测试损失值，横轴表示参数量，左图是包含嵌入参数的情况，右图则不包含（对应$N$的定义）；
> - 图中刻画不同尺度模型的性能：从小模型（$(n_{\rm layer},d_{\rm model})=(2,128)$）到十亿参数的模型；从形状为$(6,4288)$到$(207,768)$的不同模型形态；
> - 所有的模型训练都是在完整的<font face=times new roman size=4 color=red><i>WebText2</i></font>数据集上进行，并且没有观测到过拟合现象（超大模型除外）；

![f6](https://img-blog.csdnimg.cn/0213c7edc8514e77a88a002fa18b2cd6.png#pic_center)

----

正如<font face=times new roman size=4 color=red><i>Figure 6</i></font>右图所示，$N$与模型性能的关系呈现稳定趋势，因此可以拟合到式$(1.5)$的第一项中：

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N}\tag{3.1}
$$

如果考虑嵌入参数（即<font face=times new roman size=4 color=red><i>Figure 6</i></font>左图），则趋势并不那么明显，这意味着**嵌入矩阵可以尽可能地缩小而不会影响模型性能**（**令人震惊的结论**），在参考文献[<font color=red face=times new roman size=4><i>LCG+19</i></font>](https://arxiv.org/abs/1909.11942)中也有类似的结论。

**笔者注**：这个结论实在是有点恐怖，难道说不同维度的词嵌入本质上对模型性能的贡献是差不太多的？那我们研究词嵌入的方向就应该固定一个通用的维度，只考量词嵌入的算法即可。

本文还在上文提到的其他测试语料集中对上述结论进行验证（如<font face=times new roman size=4 color=red><i>Figure 8</i></font>所示）这表明即便是在模型迁移场景下，关于$N$的乘幂法则依然成立。

> <font face=times new roman size=4 color=red><i>Figure 8</i></font>：
>
> - 纵轴表示测试损失值，横轴表示参数量，左图是包含嵌入参数的情况，右图则不包含（对应$N$的定义）；
> - 图中刻画的是在不同测试语料集上的情况（模型都是在<font color=red face=times new roman size=4><i>WebText2</i></font>上训练得到），大致呈现符合与<font color=red face=times new roman size=4><i>WebText2</i></font>类似的乘幂法则；

![f8](https://img-blog.csdnimg.cn/855b4c8ac86146c58c8c6954d55ba37b.png#pic_center)

****

#### 3.2.1 <font face=times new roman><i>LSTM</i>和通用<i>Transformer</i>的比较 <i>Comparing to LSTMs and Universal Transformers</i></font>

> <font face=times new roman size=4 color=red><i>Figure 7</i></font>：
>
> - 纵轴表示测试损失值，左图横轴表示$N$，右图横轴表示；输入序列中不同位置的分词；
> - 左图刻画的是<font face=times new roman size=4 color=red><i>Transformer</i></font>与不同层数的<font face=times new roman size=4 color=red><i>LSTM</i></font>的模型性能对比（在相同的训练条件下），右图刻画的是不同参数量下的性能对比；
> - 左图说明<font face=times new roman size=4 color=red><i>Transformer</i></font>性能比<font face=times new roman size=4 color=red><i>LSTM</i></font>优越，右图说明<font face=times new roman size=4 color=red><i>LSTM</i></font>虽然在大约前$100$个分词的表现跟<font face=times new roman size=4 color=red><i>Transformer</i></font>相仿，但是之后的分词预测情况就要差很多。

![f7](https://img-blog.csdnimg.cn/8809907e7dc24e588588106a0ef82672.png#pic_center)

----

- 本文附录<font face=times new roman size=4 color=red><i>D.5</i></font>阐述了模型性能与上下文位置的乘幂法则联系，即若**增加大模型的乘幂指数**（<font face=times new roman size=4 color=red><i>increase large powers for larger models</i></font>），可使模型具有**更强的模块识别能力**（<font face=times new roman size=4 color=red><i>improved ability to quickly recognize patterns</i></font>）。
- 本文将标准<font face=times new roman size=4 color=red><i>Transformer</i></font>与参考文献[<font color=red face=times new roman size=4><i>DGV+18</i></font>](https://arxiv.org/abs/1807.03819)中提出的<font face=times new roman size=4 color=red><i>Recurrent Transformer</i></font>进行对比（见附录<font face=times new roman size=4 color=red><i>D.2</i></font>的<font face=times new roman size=4 color=red><i>Figure 17</i></font>），后者的架构中对部分参数进行**重用**（<font face=times new roman size=4 color=red><i>reuse</i></font>），因此关于模型参数量$N$的模型性能要更好一些。

----

#### 3.2.2 推广到不同数据分布 <font face=times new roman><i>Generalization Among Data Distributions</i></font>

本文还在其他分布的文本数据上进行了模型测试，具体已在上文<font face=times new roman size=4 color=red><i>Figure 8</i></font>中详细阐述，迁移训练的结论依然符合平滑乘幂法则。作者认为推广是否成功几乎全部取决于**分布内的验证损失值**（<font face=times new roman size=4 color=red><i>in-distribution validation loss</i></font>），与**模型训练**（<font face=times new roman size=4 color=red><i>during of training</i></font>）或**近似收敛**（<font face=times new roman size=4 color=red><i>proximity to convergence</i></font>）无关，附录<font face=times new roman size=4 color=red><i>D.8</i></font>中的实验结果还表明这种推广与**模型深度**（<font face=times new roman size=4 color=red><i>model depth</i></font>）也无关。

----

### 3.3 数据量与计算力对性能的影响 <font face=times new roman><i>Performance with Dataset Size and Compute</i></font>

- 本文在$(n_{\rm layer},n_{\rm embd})=(36,1280)$的配置下，在<font face=times new roman size=4 color=red><i>WebText2</i></font>的子集上训练模型（一旦测试损失值停止下降则停止训练），由此得到的测试损失值与训练数据量$D$同样很好地拟合乘幂法则（拟合图像见上文<font face=times new roman size=4 color=red><i>Figure 1</i></font>）：

  $$
  L(D)\approx\left(\frac{D_c}D\right)^{\alpha_D}\tag{3.2}
  $$

- 上文中已经论述过$C=6NBS$的合理性，因此给定$C$的条件下，可以遍历不同模型参数量$N$的模型进行训练，直到确定在第$S=C/6NB$轮迭代时模型性能最好的$N$。注意在该实验中批训练量$B$是固定值，因此实验得到的经验结果并非必然最优，在后面的内容中将会引入一个调整过的变量$C_{\min}$来生成更加简洁的规律。这里我们同样得到关于$C$的乘幂法则（拟合图像见上文<font face=times new roman size=4 color=red><i>Figure 1</i></font>）：

  $$
  L(C)\approx\left(\frac{C_c}C\right)^{\alpha_C}\tag{3.3}
  $$

- 后面的内容还将讨论计算量$C$的分配问题，结论在上文已经提过，即尽可能多地分配给模型尺寸，这在附录<font face=times new roman size=4 color=red><i>D.4</i></font>的<font face=times new roman size=4 color=red><i>Figure 19</i></font>中有直接说明。 

----

## 4 无穷数据限制与过拟合绘图 <font face=times new roman><i>Charting the Infinite Data Limit and Overfitting</i></font>

本节着重讨论模型尺寸$N$与训练数据量$D$对测试损失值$L$的联合影响，这将用于指导在模型训练中，增加模型参数量的情况下，需要相应增加多少训练数据。

----

### 4.1 提出的$L(N,D)$方程 <font face=times new roman><i>Proposed</i></font> $L(N,D)$ <font face=times new roman><i>Equation</i></font>

$$
L(N,D)=\left[\left(\frac{N_c}N\right)^{\alpha_N/\alpha_D}+\frac{D_c}D\right]^{\alpha_D}\tag{4.1}
$$

式$(4.1)$照搬自式$(1.5)$，接下来说明这其中蕴含的性质：

1. 改变词汇量或分词方式意味着**重调整**（<font face=times new roman size=4 color=red><i>rescale</i></font>）测试损失值（通过乘以一个整体系数），$L(N,D)$的参数化（包括用其他方式建模$L(N,D)$）必须考虑到这种**重调整**；

2. 固定$D$，令$N\rightarrow\infty$，测试损失值将会收敛到$L(D)$；固定$N$，令$D\rightarrow\infty$，测试损失值将会收敛到$L(N)$；

3. $L(N,D)$应当在$D=\infty$处**可分析**（<font face=times new roman size=4 color=red><i>analytic</i></font>），因而具有$1/D$的整数幂**级数展开**（<font face=times new roman size=4 color=red><i>series expansion</i></font>），该性质的理论支撑要比前两条弱；

   **笔者注**：这条性质不是很能理解（大约是和过拟合相关的），下面虽然接着对这条性质做了说明，但还是没有看明白，笔者猜想所谓$1/D$是说可以将$(4.1)$式中提一个$1/D$的系数出来，然后剩下的部分可以进行级数展开，原文脚注中也提到如果是下面的形式则无法进行$1/D$展开：
   $$
   L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{D_c}{D}\right)^{\alpha_D}\right]^\beta\tag{*}
   $$

   不过作者也说了这条性质未必是正确的。

----

显然本文选择的$L(N,D)$符合**第一条性质**，因为可以通过改变词汇表来**重调整**$N_c,D_c$的取值（这也暗示$N_c,D_c$没有实际含义）。

上文有说过，一旦测试损失值停止降低，则提前停止训练，因此我们期望大模型总是要比小模型表现得好。在固定$D$（有穷值）的条件下，我们也不期望任何模型能够收敛到可能最好的损失值（即文本的熵值）。类似地，模型尺寸固定的情况下，模型性能同样受限。这些结论促成**第二条性质**。

**第三条性质**更多只是**猜想**（<font face=times new roman size=4 color=red><i>speculative</i></font>）。参考文献[<font face=times new roman size=4 color=red><i>AS17</i></font>](https://arxiv.org/abs/1710.03667)中指出**过拟合与训练集的方差或者**<font face=times new roman size=4 color=red><i>signal-to-noise</i></font>**比率相关，且尺度系数为$1/D$**（即训练集越大，越容易过拟合），该结论应当对任意平滑损失函数成立。这条性质解释了式$(4.1)$中$N$和$D$的**非对称性**（<font face=times new roman size=4 color=red><i>asymmetry</i></font>）。

----

### 4.2 实验结果 <font face=times new roman><i>Results</i></font>

- 无论如何，至少式$(4.1)$在实验数据点上拟合结果确实很好，下面是实验得到的具体拟合值：

  |  参数  | $\alpha_N$ | $\alpha_D$ |        $N_c$        |        $D_c$        |
  | :----: | :--------: | :--------: | :-----------------: | :-----------------: |
  | 拟合值 |  $0.076$   |  $0.103$   | $6.4\times 10^{13}$ | $1.8\times 10^{13}$ |

- 本文模型训练中采取$10\%$的<font face=times new roman size=4 color=red><i>dropout</i></font>，且一旦测试损失值不再降低就提前结束训练，具体实验结果如<font face=times new roman size=4 color=red><i>Figure 9</i></font>所示：

  > <font face=times new roman size=4 color=red><i>Figure 9</i></font>：
  >
  > - 左图很好理解，横轴表示模型参数量，纵轴表示测试损失值，可以发现模型参数量越多（$N\uparrow$）、训练数据量（$D\uparrow$）越大，测试损失值越低；
  >
  > - 右图是很有趣的东西，横轴$N^{\alpha_N/\alpha_D}/D$用于衡量模型参数量$N$与训练数据量$D$之间经过标准化的一个比率，纵轴$L/L(D\rightarrow\infty)-1$刻画过拟合程度（$D\rightarrow\infty$时过拟合，测试损失值会非常小，因此这个数值越低说明过拟合程度越高），图中结果说明训练数据量越大（$D\uparrow$）、模型参数量相较训练数据量越少（$N^{\alpha_N/\alpha_D}/D\downarrow$），越容易过拟合。
  >
  >   **笔者注**：右图的横轴是否是对<font face=times new roman size=4 color=red><i>Section 4.1</i></font>中$1/D$级数展开的一种理解？

  ![f9](https://img-blog.csdnimg.cn/98b4a6ad99944e38b812a5eccef22849.png#pic_center)

  ----


- 本文最终得到了最好的模型拟合结果，训练集只使用了$2\times 10^{7}$个分词（为全数据集的$1/1024$），在这个小训练集上，每个<font face=times new roman size=4 color=red><i>epoch</i></font>仅更新$40$个参数。这种小模型训练时过拟合会发生得特别快（具体见下文的<font face=times new roman size=4 color=red><i>Figure 16</i></font>）。

- 除最大的模型之外，本文没有发现训练中发生过拟合（使用完整的$22\rm B$分词训练数据集<font face=times new roman size=4 color=red><i>WebText2</i></font>），因此可以近似认为$D=\infty$，定义：

  $$
  \delta L(N,D)\equiv \frac{L(N,D)}{L(N,\infty)}-1\tag{4.2}
  $$

  用于衡量过拟合程度，代入式$(4.1)$可以推导得出：

  $$
  \delta L\approx\left(1+\left(\frac N{N_c}\right)^{\alpha_N/\alpha_D}\frac{D_c}D\right)^{\alpha_D}-1\tag{4.3}
  $$

  注意即便是在$D$取很大的值时，式$(4.3)$依然具有$1/D$的幂级数展开。

- 仿真实验表明损失值的方差近似为$0.02$，因此为了避免训练时发生过拟合，要求尽量满足：

  $$
  D>5\times 10^3\times N^{0.74}\tag{4.4}
  $$

  在这样的条件下，参数量少于$10^9$的模型可以在$22\rm B$的<font face=times new roman size=4 color=red><i>WebText2</i></font>数据集上进行训练而不发生过拟合（这也就是为什么本文最大的模型会发生过拟合）。

  此外，这还说明当模型尺寸增加时，训练数据量的增速只需要达到次线性即可防止过拟合，但是这个结论可能并不一定可以代表**最大化的高效计算训练**（<font face=times new roman size=4 color=red><i>maximally compute-efficient training</i></font>），因为本文并没有对包括<font face=times new roman size=4 color=red><i>dropout</i></font>概率在内的各种正则项进行优化。

----



----

## 5 模型尺寸与训练时间的尺度法则 <font face=times new roman><i>Scaling Laws with Model Size and Train Time</i></font>

本节将论述$N$和训练时间对测试损失值的联合影响：

- 首先，本文将利用参考文献[<font color=red face=times new roman size=4><i>MKAT18</i></font>](https://arxiv.org/abs/1812.06162)的实验结果来定义一个**通用训练轮数**（<font face=times new roman size=4 color=red><i>universal training step</i></font>），这是对模型训练未使用最优批训练量的考虑；
- 然后，本文将论述可以根据式$(1.6)$来分别拟合测试损失值与模型尺寸和训练时间的关系；
- 最后，本文将使用这些结果来预测最优的训练计算量分配方式（在模型尺寸和训练时间两个因素中进行分配）；

### 5.1 $B_{\rm crit}(L)$处的训练调整 <font face=times new roman><i>Adjustment for Training at</i></font> $B_{\rm crit}(L)$

- 参考文献[<font color=red face=times new roman size=4><i>MKAT18</i></font>](https://arxiv.org/abs/1812.06162)（也可以查阅参考文献[<font color=red face=times new roman size=4><i>SLA+18</i></font>](https://arxiv.org/abs/1811.03600)和参考文献[<font color=red face=times new roman size=4><i>ZLN+19</i></font>](https://arxiv.org/abs/1907.04164)）中提出一个关于批训练量的经验理论，即必然存在一个**关键批训练量**$B_{\rm crit}$，使得从批训练量从$0$增长到$B_{\rm crit}$时，计算效率不断提高，超过$B_{\rm crit}$后计算效率又逐渐降低（即$B_{\rm crit}$是一个极值点），可以使用**梯度噪声尺度**可用于预测$B_{\rm crit}$的数值。

- 具体而言，在许多神经网络任务中，训练轮数$S$与训练中处理过的数据量$E=BS$满足下式：

  $$
  \left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1\tag{5.1}
  $$

  当训练进行到某个固定的训练损失值$L$时，式$(5.1)$中的$S_{\min}$表示最少达到$L$的训练迭代轮数，$E_{\min}$表示最少需要处理的数据量（在附录<font face=times new roman size=4 color=red><i>D.3</i></font>的<font face=times new roman size=4 color=red><i>Figure 18</i></font>中进一步论述式$(5.1)$在<font face=times new roman size=4 color=red><i>Transformer</i></font>中的情况），进而有：

  $$
  B_{\rm crit}(L)\equiv\frac{E_{\min}}{S_{\min}}\tag{5.2}
  $$

  注意这里$B_{\rm crit}$是关于训练损失值的函数（因为$E_{\min}$和$S_{\min}$是关于$L$的函数），当使用**关键批训练量**进行训练时，需要$2S_{\min}$的训练迭代轮数与处理$E=2E_{\min}$训练样本。

- <font face=times new roman size=4 color=red><i>Figure 10</i></font>中拟合得到的**关键批训练量**：

  $$
  B_{\rm crit}(L)\approx\frac{B_*}{L^{1/\alpha_B}};\quad B_*\approx2\times10^{8};\quad \alpha_B\approx0.21\tag{5.3}
  $$

  > <font face=times new roman size=4 color=red><i>Figure 10</i></font>：
  >
  > - 横轴表示训练损失值，纵轴表示**关键批训练量**（以分词量计算）；
  > - 图中绘制的时不同模型尺度下的散点图，虚线是拟合曲线（以噪声尺度衡量），如式$(5.1)$所示；

  ![f10](https://img-blog.csdnimg.cn/df08ac88a43947f0ad060e7189830cd0.png#pic_center)

  ----

  之所以选取式$(5.3)$的参数化方法，是因为损失值最终会不断靠近最小值$L_{\min}$，而**梯度噪声尺度**将会发散（<font face=times new roman size=4 color=red><i>Figure 11</i></font>中的绿色点即为**噪声尺度**），我们希望$B_{\rm crit}$能够尽量拟合**噪声尺度**（<font face=times new roman size=4 color=red><i>noise scale</i></font>）。

  问题在于我们并不知道$L_{\rm min}$的确切值（这只是假想的一个下界），但是这并不关键因为至少我们知道$L_{\rm min}>0$，且$L_{{\min}}$应当比所取得的损失值$L$要小得多，因此在式$(5.3)$中$B_{\rm crit}$在$L\rightarrow0$时将会发散。

- 本文使用$B_{\rm crit}(L)$来估计训练迭代轮数$S$（批训练量$B=2^{19}$）与训练迭代轮数$S_{\rm min}$（$B\gg B_{\rm crit}$）之间的关系：

  $$
  S_{{\min}}(S)\equiv\frac S{1+B_{\rm crit}(L)/B}\quad(\text{minimum steps, at }B\gg B_{\rm crit})\tag{5.4}
  $$

  其中$L$为给定的训练损失值，同理可以定义训练至$L$的**关键计算量**（<font face=times new roman size=4 color=red><i>critical value of compute</i></font>）：

  $$
  C_{\rm min}(C)\equiv\frac{C}{1+B/B_{\rm crit}(L)}\quad(\text{minimum compute, at}B\ll B_{\rm crit})\tag{5.5}
  $$

  其中$C=6NBS$（上文已经推导过）。

### 5.2 $L(N,S_{\min})$的结果以及模型尺寸和计算量对性能的影响 <font face=times new roman><i>Results for</i></font> $L(N,S_{\min})$ <font face=times new roman><i>and Performance with Model Size and Compute</i></font>

- 正如式$(1.6)$提到过的一样，现在我们使用式$(5.4)$中定义的$S_{\min}$来构建损失值关于模型尺寸$N$以及$S_{\min}$的关系式（在有穷数据量限制下）：

  $$
  L(N,S_{\min})=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}\tag{5.6}
  $$
  
  具体的拟合参数值结果如下表所示：

  |  参数  | $\alpha_N$ | $\alpha_S$ |        $N_c$        |      $S_c$       |
  | :----: | :--------: | :--------: | :-----------------: | :--------------: |
  | 拟合值 |  $0.077$   |   $0.76$   | $6.5\times 10^{13}$ | $2.1\times 10^3$ |

  如上文<font face=times new roman size=4 color=red><i>Figure 4</i></font>所示，虽然这些拟合值不是那么完美，但是对于具有简洁形式的式$(5.6)$来说已经很好了。

- > <font face=times new roman size=4 color=red><i>Figure 11</i></font>：
  >
  > - 本图是对模型性能关于模型尺寸的绘图（左图固定计算量预算，右图固定训练迭代轮数）；
  > - 横轴表示模型参数量，纵轴表示测试损失值，折线越深表示计算量宇轩（或训练迭代轮数）越少；
  > - 左图可以观察到模型参数量变大可能会导致过拟合（测试损失值上升），但是提升计算量预算可以缓解过拟合（整体的下界可以看出$L$关于$N$和$C$的关系是符合乘幂法则的）；
  > - 右图可以观察到并没有发生过拟合现象，这可能是因为此时计算量预算足够大；

  ![f11](https://img-blog.csdnimg.cn/a5c779e7e9f2439da7e7eed15591c3bd.png#pic_center)

  ----
  
- 损失值关于$S_{\min}$的乘幂法则独立性揭示了**优化器动力**（<font face=times new roman size=4 color=red><i>optimizer dynamics</i></font>）与**损失值形态**（<font face=times new roman size=4 color=red><i>loss landscape</i></font>）之间的相互作用。由于模型训练后期的拟合效果非常好，当损失值近似为二次，乘幂法则能够提供关于损失值的海森矩阵的**谱信息**（<font face=times new roman size=4 color=red><i>spectrum</i></font>），一般来说这表明**海森特征值密度**（<font face=times new roman size=4 color=red><i>Hessian eigenvalue density</i></font>）近似独立于模型尺寸。

  **笔者注**：非常理论性的描述，完全看不懂。

### 5.3 提前停止步骤的下界 <font face=times new roman><i>Lower Bound on Early Stopping Step</i></font>

- 式$(5.3)$可用于推导出当数据量受限时，模型训练的早停点对应的训练迭代轮数的下界。这是因为给定模型，有穷和无穷的$D$对应的学习曲线直到训练轮数接近$S_{\min}\approx S_{\rm stop}$时都非常类似，这样有穷$D$的过拟合早停点$S_{\rm stop}$就对应无穷$D$的终止点$S_{\min}$，只需要做一个比例放缩即可用$S_{\min}$来估计$S_{\rm stop}$；

- 但是这种方法会低估$S_{\rm stop}$，因为事实上测试损失值在有穷$D$的情况下会下降得越来越慢，因此需要更多的训练迭代轮数才能确实地达到最优测试损失值，即有：

  $$
  S_{\rm stop}(N,D)>\frac{S_c}{[L(N,D)-L(N,\infty)]^{1/\alpha_S}}\tag{5.7}
  $$

  其中$L(N,\infty)$是收敛到的损失值（无穷$D$），附录<font face=times new roman size=4 color=red><i>D.1</i></font>的<font face=times new roman size=4 color=red><i>Figure 16</i></font>对该不等式及其与经验数据的对比有具体说明。

----

## 6 计算量预算的最优分配 <font face=times new roman><i>Optimal Allocation of the Computer Budget</i></font>

<font face=times new roman size=4 color=red><i>Figure 1</i></font>提供了一些经验性的趋势结果，但是这些结都要求一个固定的批训练量$B$，而我们知道事实上我们可以通过$B_{\rm crit}$来实现更高效的训练（如<font face=times new roman size=4 color=red><i>Section 5.1</i></font>论述），本节中我们将基于这一点对上文中的结果进行调整，利用<font face=times new roman size=4 color=red><i>Section 5</i></font>的结论来决定计算量在模型尺寸$N$和训练中处理的总数据量$2B_{\rm crit}S_{{\min}}$之间的最优分配（从经验角度和理论角度）。

----

### 6.1 最优性能与分配  <font face=times new roman><i>Optimal Performance and Allocations</i></font>

- 首先基于式$(5.5)$研究损失值关于最优分配计算量的函数，结果如<font face=times new roman size=4 color=red><i>Figure 13</i></font>所示：

  ![f13](https://img-blog.csdnimg.cn/e631ee3f4a414536a97b794badff3d04.png#pic_center)

  可以发现关于$C_{{\min}}$的拟合比$C$在模型性能上要有所提升。

- 给定$L(C_{{\min}})$，很自然地会想到最优模型尺寸$N(C_{{\min}})$（即给定训练计算量可以得到最小的损失值），这个结果如<font face=times new roman size=4 color=red><i>Figure 14</i></font>所示：

  ![f14](https://img-blog.csdnimg.cn/9ecb223bdde145ceb56b8d8f1554063b.png#pic_center)

  可以发现$N(C_{\min})$同样可以由乘幂法则拟合得很好：

  $$
  N(C_{\min})\propto C_{\min}^{0.73}\tag{6.1}
  $$

- 在<font face=times new roman size=4 color=red><i>Figure 12</i></font>中证明了次优模型尺度对模型训练的影响（附录<font face=times new roman size=4 color=red><i>B.4</i></font>有详细考察）：

  ![f12](https://img-blog.csdnimg.cn/425a0147ef564ac388277f30f06f0cfe.png#pic_center)

  ----

- 定义$C_{\min}=6NB_{\rm crit}S$，可用$N(C_{{\min}})$来挖掘更深层的结论，在上文中已知$B\propto L^{-4.8},L\propto C_{{\min}}^{-0.05}$，可得$B_{\rm crit}\propto C_{{\min}}^{0.24}$，这意味着最优的训练迭代轮数与计算量相关的增长幅度会非常小：

  $$
  S_{\min}\propto C_{\min}^{0.03}\tag{6.2}
  $$

  这与<font face=times new roman size=4 color=red><i>Figure 14</i></font>的经验结果相符合。

- 因此我们应当尽可能多地将计算量分配到模型尺寸上，而非训练迭代轮数，同时要扩大$B\propto B_{\rm crit}$以确保训练迭代轮数保持不变。

----

### 6.2 从$L(N,S_{\min})$得到的预测结果 <font face=times new roman><i>Predictions from</i></font> $L(N,S_{\min})$

- $L(C_{\min})$与分配方式可以通过式$(5.6)$中的$L(N,S_{\min})$进行预测，即将$S_{\min}$替换为$C_{\min}/6NB$，然后拟合最小损失值关于$N$的函数（固定计算量），具体步骤在附录<font face=times new roman size=4 color=red><i>B</i></font>中已有说明。

- 可以进行如下的估计：

  $$
  L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}\tag{6.3}
  $$

  其中：

  $$
  \alpha_C^{\min}=\left(\frac1{\alpha_S}+\frac1{\alpha_B}+\frac1{\alpha_N}\right)^{-1}\approx 0.054\tag{6.4}
  $$

  为了更好的拟合<font face=times new roman size=4 color=red><i>Figure 13</i></font>中的结果，作如下的预测：

  $$
  N(C_{\min})\propto(C_{\min})^{\alpha_C^{\min}/\alpha_N}\approx C_{\min}^{0.71}\tag{6.5}
  $$

  这同样匹配了<font face=times new roman size=4 color=red><i>Figure 14</i></font>中的尺度关系。

----

### 6.3 矛盾与一个猜想 <font face=times new roman><i>Contradictions and a Conjecture</i></font>

- 这小节内容比较难于理解，笔者只能简要说个大概，具体可以自行查阅原文：

  - 注意<font face=times new roman size=4 color=red><i>Figure 13</i></font>中$L(C_{\min})$的性能要比$L(C)$更好，这意味着尺度法则必须在$L(C)$与$L(C_{\min})$这个交点处隔断，但是我们认为这个交点可能提供了更深层次的信息，即<font face=times new roman size=4 color=red><i>Transformer</i></font>语言模型达到最优性能的一个估计点，具体如<font face=times new roman size=4 color=red><i>Figure 15</i></font>所示：

    ![f15](https://img-blog.csdnimg.cn/f4141528ca01486d97d430bc80f8d32f.png#pic_center)

  - 为了确保过拟合，<font face=times new roman size=4 color=red><i>Section 4</i></font>中的结果表明我们需要放缩训练数据量：
    $$
    D\propto N^{0.74}\propto C_{\min}^{0.54}\tag{6.6}
    $$
    其中$N(C_{\min})$在<font face=times new roman size=4 color=red><i>Figure 14</i></font>中已提及；

    此时与**计算有效训练**（<font face=times new roman size=4 color=red><i>compute-efficient training</i></font>，应该是指计算量的有效分配）所需要的数据量进行比较，若以**关键批训练量**进行训练（即$C=2C_{\min}$），且不在训练中重复使用数据，则使用的数据量关于计算量的增长满足下式：
    $$
    D(C_{\min})=\frac{2C_{\min}}{6N(C_{\min})}\approx(4\times 10^{10}\text{ tokens})(C_{\min}/\text{PF-day})^{0.26}\tag{6.7}
    $$
    这是数据集可能增长最快的速率，因此意味着我们只能在单一的<font face=times new roman size=4 color=red><i>epoch</i></font>中训练，但是这个速率比式$(6.6)$中的速率要低得多，这就好像意味着**计算有效训练**最终必然会进入过拟合，即便训练过程中从未重复使用过任何数据。

  - 然后作者做了一大段晦涩难懂的解释（他想重新解释<font face=times new roman size=4 color=red><i>Figure 15</i></font>中的那个交点），有兴趣地自己去看一看，笔者觉得不是很靠谱。个人觉得这种矛盾的出现是因为缺少理论支撑，所有结论都是经验性的，必然存在偏差，许多偏差融合在一起就必然会出现不符合常识的结论。

----

## 7 相关工作 <font face=times new roman><i>Related Work</i></font>

- 乘幂法则最早可以追溯到[<font face=times new roman size=4 color=red><i>THK18</i></font>]；[<font face=times new roman size=4 color=red><i>Was06</i></font>]是关于乘幂法则在模型尺寸和训练数据量的**密度估计**（<font face=times new roman size=4 color=red><i>density estimation</i></font>）；[<font face=times new roman size=4 color=red><i>Bia12</i></font>]是关于在随机森林模型中的乘幂法则研究；
- [<font face=times new roman size=4 color=red><i>BB01, Goo01</i></font>]发现乘幂法则可以用于刻画模型性能与训练数据量的关系；近期的工作[<font face=times new roman size=4 color=red><i>HNA+17, HAD19</i></font>]也研究了模型尺寸与训练数据量之间的尺度关系（这可能是最接近本文研究的相关工作），但是[<font face=times new roman size=4 color=red><i>HNA+17</i></font>]发现的训练数据量与模型尺寸之间的超线性尺度关系（本文则是次线性尺度关系）；[<font face=times new roman size=4 color=red><i>Kom19</i></font>]中关于最优计算量分配的结论与本文有共通之处（包括乘幂法则学习曲线）；[<font face=times new roman size=4 color=red><i>TL19</i></font>]提出的<font face=times new roman size=4 color=red><i>EfficientNet</i></font>中也表现出服从近似乘幂法则的关系（在精确度与模型尺寸之间）；[<font face=times new roman size=4 color=red><i>RRBS19b</i></font>]是非常新的工作，它研究了训练数据量与模型尺寸的尺度关系（在若干不同的数据集上实验）；
- [<font face=times new roman size=4 color=red><i>TL19</i></font>]提出的<font face=times new roman size=4 color=red><i>EfficientNet</i></font>将**图片模型**（<font face=times new roman size=4 color=red><i>image model</i></font>）的最优性能与网络层深度与宽度具有指数级的尺度关系，从而得出宽度与深度之间的乘幂法则尺度关系，但是本文得出的结论表明模型性能与网络层深度与宽度是几乎无关的；[<font face=times new roman size=4 color=red><i>VWB161</i></font>]指出更深的模型与更浅的模型表现相仿，这与本文的结论是一致的；早期工作[<font face=times new roman size=4 color=red><i>ZK16</i></font>]比较宽度与深度，结论是更宽的<font face=times new roman size=4 color=red><i>ResNets</i></font>模型比更深的<font face=times new roman size=4 color=red><i>ResNets</i></font>在图片分类上模型表现得更好；
- [<font face=times new roman size=4 color=red><i>AS17, BHMM18</i></font>]研究**更高地过参数化模型**（<font face=times new roman size=4 color=red><i>highly overparmeterized models</i></font>），发现当模型尺寸接近训练数据量时，会出现一种**堵塞转移**（<font face=times new roman size=4 color=red><i>jamming transition</i></font>），但是本文没有发现类似地转移现象，只是认为训练数据量应当关于模型尺寸是次线性关系；[<font face=times new roman size=4 color=red><i>JGH18, LXS+19</i></font>]提供了一种用于研究尺度关系地框架；[<font face=times new roman size=4 color=red><i>ZLN+19</i></font>]提供了一些精确预测地实际配置方法；[<font face=times new roman size=4 color=red><i>Pap18, GKX19, GARD18</i></font>]是关于海森矩阵谱信息地特征化的内容；

----

## 8 讨论 <font face=times new roman><i>Discussion</i></font>

本节无重要内容，基本是对全文的一个总结，不再赘述。

----

## 附录 <font face=times new roman><i>Appendices</i></font>

### A 尺度法则总结 <font face=times new roman><i>Summary of Power Laws</i></font>

本文所有公式结论一览：

|    参数量     |    数据量     |      计算量      |      批训练量       |                             公式                             |
| :-----------: | :-----------: | :--------------: | :-----------------: | :----------------------------------------------------------: |
|      $N$      |   $\infty$    |     $\infty$     |        固定         |                  $L(N)=(N_c/N)^{\alpha_N}$                   |
|   $\infty$    |      $D$      |       早停       |        固定         |                  $L(D)=(D_c/D)^{\alpha_D}$                   |
|     最优      |   $\infty$    |       $C$        |        固定         |           $L(C)=(C_c/D)^{\alpha_C}\text{(naive)}$            |
| $N_{\rm opt}$ | $D_{\rm opt}$ |    $C_{{\min}}$    | $B\ll B_{\rm crit}$ |        $L(C_{\min})=(C_c^{\min}/C_{\min})^{\alpha_C^{\min}}$         |
|      $N$      |      $D$      |       早停       |        固定         | $L(N,D)=\left[\left(\frac{N_c}N\right)^{\alpha_N/\alpha_D}+\frac{D_c}D\right]^{\alpha_D}$ |
|      $N$      |   $\infty$    | $S\text{ steps}$ |         $B$         | $L(N,S)=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}(S,B)}\right)^{\alpha_S}$ |

经验性的拟合值：

|       乘幂法则        |              量级（与分词方法有关）              |
| :-------------------: | :----------------------------------------------: |
|   $\alpha_N=0.076$    | $N_c=8.8\times 10^{13}\text{ params(non-embed)}$ |
|   $\alpha_D=0.095$    |      $D_c=5.4\times 10^{13}\text{ tokens}$       |
|   $\alpha_C=0.057$    |       $C_c=1.6\times 10^7\text{ PF-days}$        |
| $\alpha_C^{\min}=0.050$ |     $C_c^{\min}=3.1\times 10^8\text{ PF-days}$     |
|    $\alpha_B=0.21$    |        $B_*=2.1\times 8^3\text{ tokens}$         |
|    $\alpha_S=0.76$    |        $S_c=2.1\times 10^3\text{ steps}$         |

高效训练计算量的最优参数：

|                          高效计算值                          |  乘幂法则  |                 尺度                  |
| :----------------------------------------------------------: | :--------: | :-----------------------------------: |
|                $N_{\rm opt}=N_eC^{p_N}_{\min}$                 | $p_N=0.73$ | $N_e=1.3\times 10^9\text{ params}$^{} |
| $B\ll B_{\rm crit}=\frac{B_*}{L^{1/\alpha_B}}=B_eC_{\min}^{p_B}$ | $p_B=0.24$ |  $N_e=2.0\times 10^6\text{ tokens}$   |
|         $S_{\min}=S_eC_{\min}^{p_S}\text{(lower bound)}$         | $p_S=0.03$ |   $S_e=5.4\times 10^3\text{ steps}$   |
|        $D_{\rm opt}=D_eC^{p_D}_{\min}\text{(1 epoch)}$         | $p_D=0.27$ | $D_e=2.0\times 10^{10}\text{ tokens}$ |

----

### B 计算有效边界的经验模型 <font face=times new roman><i>Empirical Model of Compute-Efficient Frontier</i></font>

本节中的$C,S,\alpha_C$都被调整为在**关键批训练量**$B_{\rm crit}$下的数值，使用<font face=times new roman size=4 color=red><i>adj</i></font>标签来避免混淆。

----

#### B.1 定义方程 <font face=times new roman><i>Defining Equations</i></font>

本节将推导最优性能、模型尺寸、训练轮数关于计算量预算的函数，首先从式$(1.6)$开始（照抄过来）：
$$
L(N,S)=\left(\frac{N_c}N\right)^{\alpha_N}+\left(\frac{S_c}{S}\right)^{\alpha_S}\tag{B.1}
$$
其中$S$表示在**关键批训练量**下训练更新的参数量（如式$(5.2)$所示）：
$$
B(L)=\frac{B_*}{L^{1/\alpha_B}}\tag{B.2}
$$
**笔者注**：这里有一些歧义，我们可以香香训练要么在一个常数批训练量$B(L_{\rm target})$，要么在一个变量批训练量$\tilde B(L)$，其中$\tilde B$是**瞬时的**（<font face=times new roman size=4 color=red><i>instantaneous</i></font>）关键批训练量（与$B$相反，这是一个取平均的版本），这两个方案会得到相同的训练迭代轮数，因此我们可以忽视这当中的细微差别。

我们想要确定给定计算量预算下的最优训练参数，因此我们将$S$替换为$C/(6NB(L))$，其中$C$是训练中的浮点运算量（<font face=times new roman size=4 color=red><i>FLOPs</i></font>）：
$$
L(N,C)=\left(\frac{N_c}N\right)^{\alpha_N}+\left(6B_*S_c\frac{B}{L^{1/\alpha_B}C}\right)^{\alpha_S}\tag{B.3}
$$
令$\partial_NL|_C=0$以找到最优点：
$$
0=\frac{\partial L}{\partial N}_C\Rightarrow\frac{\alpha_N}{\alpha_S}\left(\frac{N_c}N\right)^{\alpha_N}=\left(6B_*S_c\frac{B}{L^{1/\alpha_B}C}\right)^{\alpha_S}\tag{B.4}
$$
式$(\rm B.3)(B.4)$共同决定**计算有效边界**（<font face=times new roman size=4 color=red><i>compute-efficient frontier</i></font>）。

----

#### B.2 高效训练 <font face=times new roman><i>Efficient Training</i></font>

首先将$(\rm B.4)$带入$(\rm B.3)$：
$$
L(N_{\rm eff}(C),C)=\left(1+\frac{\alpha_N}{\alpha_S}\right)L(N_{\rm eff},\infty)\tag{B.5}
$$
这意味着对于计算有效训练，我们应当以一个固定比率$\alpha_N/\alpha_S\approx 10\%$进行训练。

接下来决定最优损失与计算量预算的关系，已知（消除$N$得到$C$）：
$$
L(C)=\left(\frac{C_c}{C}\right)^{\alpha_C}\tag{B.6}
$$
其中定义：
$$
\begin{aligned}
\alpha_C&=\left(\frac1{\alpha_S}+\frac1{\alpha_B}+\frac1{\alpha_N}\right)\approx0.052\\
C_c&=6N_cB_*S_c\left(1+\frac{\alpha_N}{\alpha_S}\right)^{\frac1{\alpha_S}+\frac1{\alpha_N}}\left(\frac{\alpha_S}{\alpha_N}\right)^{\frac1{\alpha_S}}
\end{aligned}\tag{B.7&B.8}
$$
类似地消除$L$得到$N(C)$：
$$
\frac{N(C)}{N_c}=\left(\frac C{C_c}\right)^{\frac{\alpha_C}{\alpha_N}}\left(1+\frac{\alpha_N}{\alpha_S}\right)^{\frac1{\alpha_N}}\tag{B.9}
$$
以及：
$$
S(c)=\frac{C_c}{6N_cB_*}\left(\frac C{C_c}\right)^{\frac{\alpha_C}{\alpha_S}}\left(1+\frac{\alpha_N}{\alpha_S}\right)^{-\frac1{\alpha_N}}\tag{B.10}
$$

----

#### B.3 与低效情形的对比 <font face=times new roman><i>Comparison to Inefficient</i></font>

定义收敛因子$f$：
$$
L(N,C)=(1+f)L(N,\infty)\tag{B.11}
$$
对于计算有效训练的情形，我们有$f=\alpha_N/\alpha_S\approx10\%$，但是学者一般使用更小的$f$值，这里我们选取$f'=2\%$作为估计量，对于固定的损失值，可以预测：
$$
\begin{aligned}
\frac{N_f}{N_{f'}}&=\left(\frac{1+f}{1+f'}\right)^{\frac1\alpha_N}\approx 2.7\\
\frac{S_f}{S_f'}&=\left(\frac{1+1/f}{1+1/f'}\right)^{\frac1{\alpha_S}}\approx 0.13\\
\frac{C_f}{C_{f'}}&=\frac{N_f}{N_{f'}}\frac{S_f}{S_f'}\approx0.35
\end{aligned}\tag{B.12-14}
$$
因此计算有效训练使用$7.7$倍更少的参数更新，$2.7$倍更多的参数量，节约$65\%$的计算量（以达到同样的损失值水平）。

----

#### B.4 次优的模型尺寸 <font face=times new roman><i>Suboptimal Model Sizes</i></font>

我们可以解决<font face=times new roman size=4 color=red><i>A.1</i></font>来推导用于达到给定损失值$L$的计算量关于模型尺寸$N$的函数：
$$
C(N,L)=\left(6B_*S_c\frac{N}{L^{1/\alpha_B}}\right)\left(L-\left(\frac{N_c}{N}\right)^{\alpha_N}\right)^{-1/\alpha_S}\tag{B.15}
$$
使用<font face=times new roman size=4 color=red><i>A.6</i></font>到<font face=times new roman size=4 color=red><i>A.9</i></font>，可以消除$L$（以$N_{\rm eff}(L)$来代替），这里我们推导出一个表达式用于使用次优模型尺寸下的计算量比率：
$$
\frac{C(N,N_{\rm eff})}{C(N_{\rm eff},N_{\rm eff})}=\frac N{N_{\rm eff}}\left[1+\frac{\alpha_S}{\alpha_N}\left(1-\left(\frac{N_{\rm eff}}{N}\right)^{\alpha_N}\right)\right]^{-1/\alpha_S}\tag{B.16}
$$
这个结果在<font face=times new roman size=4 color=red><i>Figure X</i></font>（这个图似乎论文中没有给出）中得到了证明，模型尺度在$0.6$到$2.2$倍的最优尺寸区间内只会增加$20\%$的计算量预算，使用小模型会减少这种成本提升的影响。若硬件允许更多的并行与高速训练，则大模型可以在更少的训练迭代轮数达到同等的模型性能（如<font face=times new roman size=4 color=red><i>Figure Y</i></font>所示，这个图也没有给出）：
$$
\frac{S(N,N_{\rm eff})}{S(N_{\rm eff},N_{\rm eff})}=\left[1+\frac{\alpha_S}{\alpha_N}\left(1-\left(\frac{N_{\rm eff}}{N}\right)^{\alpha_N}\right)\right]^{-1/\alpha_S}\tag{B.17}
$$
一个$2.2$倍的大模型需要$45\%$更少的训练迭代轮数（仅提升$20\%$的训练计算量），注意式$(\rm B.17)$对于超大模型可能并不成立。

----

### C 警告 <font face=times new roman><i>Caveats</i></font>

关于本文分析的一些潜在警告：

- 目前对于提出的尺度法则没有坚实的理论基础，大部分只是经验性的结论，比如在超大模型尺寸的条件下，关于$D$的尺度法则就不那么可信；
- 对于$B_{\rm crit}(L)$的预测也并不那么确信；
- 对于很小的训练数据量的情况并没有进行完整的实验分析，如$L(N,D)$对于非常小的$D$的预测可能就不那么准确；
- $C\approx 6NBS$成立的条件请着重关注，许多嵌入参数并没有被考虑进来，以及默认$n_{\rm ctx}>12d_{\rm model}$；
- 本文只微调了学习率以及进行学习率规划，一些其他对尺度可能有影响的超参数（如初始化，动量）都没有被考虑进来；
- 最优的学习率选取相对于目标损失值是敏感的，越接近收敛应当使用越小的学习率，本文没有对大学习率进行实验；

----

### D 附图 <font face=times new roman><i>Supplemental Figures</i></font>

#### D.1 早停与测试训练对比 <font face=times new roman><i>Early Stopping and Test vs Train</i></font>

<font face=times new roman size=4 color=red><i>Section 5.3</i></font>中讨论了<font face=times new roman size=4 color=red><i>Figure 16</i></font>中的结论，它提供了对于早停轮数的一个下界预测，我们还证明了在给定模型尺寸的条件下，训练损失值与测试损失值在不同大小训练集上的情况。

- 左图的横轴刻画的是过拟合程度（如式$(4.2)$），纵轴是早停发生的迭代轮数，红色虚线是早停时间点的下界（根据<font face=times new roman size=4 color=red><i>Section 5.3</i></font>中的理论推导得出）；

- 右图是具体的迭代轮数与损失值的关系，实线表示测试损失值，虚线表示训练损失值，测试损失值一旦升高表明出现过拟合；

- $S_{\rm stop}$与$L(N,D)$是经验数值，$L(N,\infty)$是从$L(N,D)$中计算得到的（令$D\rightarrow \infty$）；

![f16](D:\code\python\project\caoyang\project_012_cs224n\cs224n-winter2022-note\7710354abc904cdcbd5e111d81eb4125.png)

----

#### D.2 通用<font face=times new roman><i>Transformer</i></font>架构 <font face=times new roman><i>Universal Transformers</i></font>

<font face=times new roman size=4 color=red><i>Figure 17</i></font>中对比了标准<font face=times new roman size=4 color=red><i>Transformer</i></font>与循环<font face=times new roman size=4 color=red><i>Transformer</i></font>的性能对比（都重使用参数），测试损失值关于$N$的函数要比关于$C$的函数看起来好一些：

![f17](https://img-blog.csdnimg.cn/dc64e849a3444e25943ec9e751f1c077.png#pic_center)

----

#### D.3 批训练量 <font face=times new roman><i>Batch Size</i></font>

<font face=times new roman size=4 color=red><i>Figure 18</i></font>中衡量了需要处理的样本分词量关于**关键批训练量**的关系，可用于支持<font face=times new roman size=4 color=red><i>Figure 10</i></font>中对$B_{\rm crit}$的预测：

![f18](https://img-blog.csdnimg.cn/571c29b744ec481db12dd3ff1271ffde.png#pic_center)

----

#### D.4 样本效率与模型尺寸 <font face=times new roman><i>Sample Efficiency vs Model Size</i></font>

<font face=times new roman size=4 color=red><i>Figure 2</i></font>中说明大模型训练得更快，因而更具有样本高效性。在<font face=times new roman size=4 color=red><i>Figure 19</i></font>中我们提供另一种方式来审视这个规律，即不同模型达到各种固定损失值所使用的样本量：

![f19](https://img-blog.csdnimg.cn/804673baec15423aa21242a4aaae80de.png#pic_center)

----

#### D.5 上下文独立性 <font face=times new roman><i>Context Dependence</i></font>

<font face=times new roman size=4 color=red><i>Figure 21</i></font>刻画了损失值关于模型尺寸的函数（对于上下文中不同的分词），可以发现以$n_{\rm ctx}=1024$训练的模型表现出最稳定的提升：

![f21](https://img-blog.csdnimg.cn/1f76851fd32847a0bfce5b39eceea3f1.png#pic_center)

----

<font face=times new roman size=4 color=red><i>Figure 20</i></font>表明在固定模型尺寸的条件下，损失值的尺度可以看作是上下文的分词位置$T$的乘幂法则函数，这可能是乘幂法则在语言中的一个潜在联系（或是模型架构和模型优化的更一般的特征）：

- 推论：大模型能够在更少地上下文信息的情况下更有效地发现文本**模式**（<font face=times new roman size=4 color=red><i>pattern</i></font>）；

- <font face=times new roman size=4 color=red><i>Figure 20</i></font>右图中表明每个分词性能在训练迭代轮数中的变化；

![f20](https://img-blog.csdnimg.cn/6216df21df414b4cb32e8e011aec1389.png#pic_center)

本文还对$n_{\rm ctx}=8$的短上下文输入进行实验（为了与长上下文输入进行对比），实验结论是应当更多考虑在大模型上训练长上下文输入。

----

#### D.6 学习率规划与误差分析 <font face=times new roman><i>Learning Rate Schedules and Error Analysis</i></font>

本文实验各种不同的学习率与学习率规划方式，<font face=times new roman size=4 color=red><i>Figure 22</i></font>的左图是一系列学习率规划方式的可视化，右图则是对应的测试性能（在一个小语言模型上测试）：

![f22](https://img-blog.csdnimg.cn/bbc518b88fc348a3a1806d6a2eb89e70.png#pic_center)

结论是不同方式的学习率选取是不关键的，只要总学习率足够大，测试性能的区别就不大。

此外本文发现更大的模型需要更小的学习率以防止训练发散，而小模型往往能够容忍更大的学习率，具体的实验数据拟合关系如下：

$$
LR(N)\approx 0.003239-0.0001395\log(N)\tag{D.1}
$$

我们希望式$(D.1)$可以被进一步优化，因为可能还会与网络宽度、初始化的尺度等因素相关。此外式$(D.1)$在$N>10^{10}$后就会崩坏，但是对于本文考察的模型来说基本已经够用了。

----

#### D.7 拟合细节与乘幂法则质量 <font face=times new roman><i>Fit Details and Power Law Quality</i></font>

本文对$L(N),L(C),L(D)$的函数形式进行了许多不同的估计，如<font face=times new roman size=4 color=red><i>Figure 23</i></font>所示是关于$L(N)$的两种不同估计，显然蓝线（乘幂法则）比红线（对数形式）要精确一些：

![f23](https://img-blog.csdnimg.cn/a570c8763ad04c42b6c9edcdc38f41d3.png#pic_center)

关于拟合数据点的选取，$L(C)$没有测试只有一层的模型，$L(N)$也没有测试只有一层的模型，同时也没有包括未能训练收敛的最大模型。

---

#### D.8 推广与架构 <font face=times new roman><i>Generalization and Architecture</i></font>

<font face=times new roman size=4 color=red><i>Figure 24</i></font>表明推广到其他数据分布仍然不依赖于网络深度（若给定总参数量），似乎只取决于训练分布上的性能：

![f24](https://img-blog.csdnimg.cn/8a801f0bbe0448238987d58fdc404cc8.png#pic_center)

----

## 参考文献 <font face=times new roman><i>References</i></font>

```
[ACDE12] Eduardo G Altmann, Giampaolo Cristadoro, and Mirko Degli Esposti. On the origin of longrange correlations in texts. Proceedings of the National Academy of Sciences, 109(29):11582– 11587, 2012. 25 
[AS17] Madhu S. Advani and Andrew M. Saxe. High-dimensional dynamics of generalization error in neural networks. arXiv, 2017, 1710.03667. 11, 18, 22 
[BB01] Michele Banko and Eric Brill. Scaling to very very large corpora for natural language disambiguation. In Proceedings of the 39th annual meeting on association for computational linguistics, pages 26–33. Association for Computational Linguistics, 2001. 18 
[BHMM18] Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine learning and the bias-variance trade-off. arXiv, 2018, 1812.11118. 18 
[Bia12] GÃŠrard Biau. Analysis of a random forests model. Journal of Machine Learning Research, 13(Apr):1063–1095, 2012. 18 
[CGRS19] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. CoRR, abs/1904.10509, 2019, 1904.10509. URL http://arxiv.org/abs/1904.10509. 19 
[DCLT18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2018, arXiv:1810.04805. 2 
[DGV+18]Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. CoRR, abs/1807.03819, 2018, 1807.03819. URL http://arxiv.org/abs/1807.03819. 6, 9, 23, 24 
[EP94] Werner Ebeling and Thorsten Pöschel. Entropy and long-range correlations in literary english. EPL (Europhysics Letters), 26(4):241, 1994. 25 
[Fou] The Common Crawl Foundation. Common crawl. URL http://commoncrawl.org. 7 
[GARD18] Guy Gur-Ari, Daniel A. Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace. 2018, arXiv:1812.04754. 18 
[GJS+19] Mario Geiger, Arthur Jacot, Stefano Spigler, Franck Gabriel, Levent Sagun, Stéphane d’Ascoli, Giulio Biroli, Clément Hongler, and Matthieu Wyart. Scaling description of generalization with number of parameters in deep learning. arXiv, 2019, 1901.01608. 18 
[GKX19] Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. CoRR, abs/1901.10159, 2019, 1901.10159. URL http://arxiv.org/abs/1901.10159. 18 
[Goo01] Joshua Goodman. A bit of progress in language modeling. CoRR, cs.CL/0108005, 2001. URL http://arxiv.org/abs/cs.CL/0108005. 18 
[GRK17] Scott Gray, Alec Radford, and Diederik P Kingma. Gpu kernels for block-sparse weights. openai.com, 2017. 19 
[HAD19] Joel Hestness, Newsha Ardalani, and Gregory Diamos. Beyond human-level accuracy: Computational challenges in deep learning. In Proceedings of the 24th Symposium on Principles and Practice of Parallel Programming, PPoPP ’19, pages 1–14, New York, NY, USA, 2019. ACM. doi:10.1145/3293883.3295710. 18 
[HCC+18] Yanping Huang, Yonglong Cheng, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. CoRR, abs/1811.06965, 2018, 1811.06965. URL http://arxiv.org/abs/1811.06965. 19 
[HNA+17]Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically, 2017, 1712.00409. 18 
[JGH18] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in neural information processing systems, pages 8571–8580, 2018. 18 
[KB14] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014, 1412.6980. 7 
[Kom19] Aran Komatsuzaki. One epoch is all you need, 2019, arXiv:1906.06669. 18 
[KSH12] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1, NIPS’12, pages 1097–1105, USA, 2012. Curran Associates Inc. URL http://dl.acm.org/citation.cfm?id=2999134.2999257. 19 
[LCG+19] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations, 2019, 1909.11942. 9 
[LOG+19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692, 2019, 1907.11692. URL http://arxiv.org/abs/1907.11692. 2 
[LSP+18] Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. Generating wikipedia by summarizing long sequences. arXiv:1801.10198 [cs], 2018, 1801.10198. URL http://arxiv.org/abs/1801.10198. 2, 6 
[LT16] Henry W Lin and Max Tegmark. Criticality in formal languages and statistical physics. arXiv preprint arXiv:1606.06737, 2016. 25 
[LXS+19] Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha SohlDickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent, 2019, arXiv:1902.06720. 18 
[MKAT18] Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training, 2018, arXiv:1812.06162. 3, 5, 6, 12, 13, 21 
[Pap18] Vardan Papyan. The full spectrum of deep net hessians at scale: Dynamics with sample size. CoRR, abs/1811.07062, 2018, 1811.07062. URL http://arxiv.org/abs/1811.07062. 18 
[RNSS18] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. URL https://s3-us-west-2.amazonaws.com/openaiassets/research-covers/languageunsupervised/language understanding paper. pdf, 2018. 2, 6 
[RRBS19a] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive prediction of the generalization error across scales, 2019, 1909.12673. 18 
[RRBS19b] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive prediction of the generalization error across scales, 2019, arXiv:1909.12673. 18 
[RSR+19] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2019, arXiv:1910.10683. 2 
[RWC+19] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. openai.com, 2019. 2, 5, 6, 7, 8 
[SCP+18] Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, Ryan Sepassi, and Blake Hechtman. Mesh-tensorflow: Deep learning for supercomputers, 2018, 1811.02084. 19 
[SHB15] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. CoRR, 2015, 1508.07909. 6 
[SLA+18] Christopher J. Shallue, Jaehoon Lee, Joe Antognini, Jascha Sohl-Dickstein, Roy Frostig, and George E. Dahl. Measuring the effects of data parallelism on neural network training, 2018, arXiv:1811.03600. 12 
[SS18] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. CoRR, abs/1804.04235, 2018, 1804.04235. URL http://arxiv.org/abs/1804.04235. 7 
[THK18] Stefan Thurner, Rudolf Hanel, and Peter Klimek. Introduction to the theory of complex systems. Oxford University Press, 2018. 18 
[TL19] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. CoRR, abs/1905.11946, 2019, 1905.11946. URL http://arxiv.org/abs/1905.11946. 18 
[VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 5998–6008. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf. 2, 6 
[VWB16] Andreas Veit, Michael Wilber, and Serge Belongie. Residual networks behave like ensembles of relatively shallow networks, 2016, arXiv:1605.06431. 8, 18 
[Was06] Larry Wasserman. All of nonparametric statistics. Springer Science & Business Media, 2006. 18 
[WPN+19] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems, 2019, 1905.00537. 2 
[WRH17] Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Growing a brain: Fine-tuning by increasing model capacity. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jul 2017. doi:10.1109/cvpr.2017.323. 19 
[WYL19] Wei Wen, Feng Yan, and Hai Li. Autogrow: Automatic layer growing in deep convolutional networks, 2019, 1906.02909. 19 
[YDY+19] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. Xlnet: Generalized autoregressive pretraining for language understanding, 2019, arXiv:1906.08237. 2 
[ZK16] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. Procedings of the British Machine Vision Conference 2016, 2016. doi:10.5244/c.30.87. 18 
[ZKZ+15] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. 2015 IEEE International Conference on Computer Vision (ICCV), Dec 2015. doi:10.1109/iccv.2015.11. 7 
[ZLN+19] Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George E. Dahl, Christopher J. Shallue, and Roger B. Grosse. Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. CoRR, abs/1907.04164, 2019, 1907.04164. URL http://arxiv.org/abs/1a907.04164. 12, 18
```


----

![f1](https://img-blog.csdnimg.cn/8a007eab06af44a08090c6112ade515f.png#pic_center)
![f2](https://img-blog.csdnimg.cn/63184c8ec495417eac5e8240774911cd.png#pic_center)
![f3](https://img-blog.csdnimg.cn/839520ce29a8417181ab5c0c8863c1a3.png#pic_center)
![f4](https://img-blog.csdnimg.cn/74a38a1b121741a7860ff139310fedc6.png#pic_center)
![f5](https://img-blog.csdnimg.cn/6a5b33d33bfc40b7a285300c0bcefb4a.png#pic_center)
![f6](https://img-blog.csdnimg.cn/0213c7edc8514e77a88a002fa18b2cd6.png#pic_center)
![f7](https://img-blog.csdnimg.cn/8809907e7dc24e588588106a0ef82672.png#pic_center)
![f8](https://img-blog.csdnimg.cn/855b4c8ac86146c58c8c6954d55ba37b.png#pic_center)
![f9](https://img-blog.csdnimg.cn/98b4a6ad99944e38b812a5eccef22849.png#pic_center)
![f10](https://img-blog.csdnimg.cn/df08ac88a43947f0ad060e7189830cd0.png#pic_center)
![f11](https://img-blog.csdnimg.cn/a5c779e7e9f2439da7e7eed15591c3bd.png#pic_center)
![f12](https://img-blog.csdnimg.cn/425a0147ef564ac388277f30f06f0cfe.png#pic_center)
![f13](https://img-blog.csdnimg.cn/e631ee3f4a414536a97b794badff3d04.png#pic_center)
![f14](https://img-blog.csdnimg.cn/9ecb223bdde145ceb56b8d8f1554063b.png#pic_center)
![f15](https://img-blog.csdnimg.cn/f4141528ca01486d97d430bc80f8d32f.png#pic_center)
![f16](https://img-blog.csdnimg.cn/7710354abc904cdcbd5e111d81eb4125.png#pic_center)
![f17](https://img-blog.csdnimg.cn/dc64e849a3444e25943ec9e751f1c077.png#pic_center)
![f18](https://img-blog.csdnimg.cn/571c29b744ec481db12dd3ff1271ffde.png#pic_center)
![f19](https://img-blog.csdnimg.cn/804673baec15423aa21242a4aaae80de.png#pic_center)
![f20](https://img-blog.csdnimg.cn/6216df21df414b4cb32e8e011aec1389.png#pic_center)
![f21](https://img-blog.csdnimg.cn/1f76851fd32847a0bfce5b39eceea3f1.png#pic_center)
![f22](https://img-blog.csdnimg.cn/bbc518b88fc348a3a1806d6a2eb89e70.png#pic_center)
![f23](https://img-blog.csdnimg.cn/a570c8763ad04c42b6c9edcdc38f41d3.png#pic_center)
![f24](https://img-blog.csdnimg.cn/8a801f0bbe0448238987d58fdc404cc8.png#pic_center)