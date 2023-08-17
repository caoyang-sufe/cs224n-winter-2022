# CS224N WINTER 2022（一）词向量（附Assignment1答案）

[CS224N WINTER 2022（一）词向量（附Assignment1答案）](https://caoyang.blog.csdn.net/article/details/125020572)
[CS224N WINTER 2022（二）反向传播、神经网络、依存分析（附Assignment2答案）](https://blog.csdn.net/CY19980216/article/details/125022559)
[CS224N WINTER 2022（三）RNN、语言模型、梯度消失与梯度爆炸（附Assignment3答案）](https://blog.csdn.net/CY19980216/article/details/125031727)
[CS224N WINTER 2022（四）机器翻译、注意力机制、subword模型（附Assignment4答案）](https://blog.csdn.net/CY19980216/article/details/125055794)
[CS224N WINTER 2022（五）Transformers详解（附Assignment5答案）](https://blog.csdn.net/CY19980216/article/details/125072701)

# 序言

- CS224N WINTER 2022课件可从[https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/)下载，也可从下面网盘中获取：

  ```python
  https://pan.baidu.com/s/1LDD1H3X3RS5wYuhpIeJOkA
  提取码: hpu3
  ```

  本系列博客每个小节的开头也会提供该小结对应课件的下载链接。

- <font color=red>**课件、作业答案、学习笔记**</font>（Updating）：[GitHub@cs224n-winter-2022](https://github.com/umask000/cs224n-winter-2022)

- <font color=red>**关于本系列博客内容的说明**</font>：

  - 笔者根据自己的情况记录较为有用的知识点，并加以少量见解或拓展延申，并非slide内容的完整笔注；

  - CS224N WINTER 2022共计五次作业，笔者提供自己完成的参考答案，不担保其正确性；

  - 由于CSDN限制博客字数，笔者无法将完整内容发表于一篇博客内，只能分篇发布，可从我的[GitHub Repository](https://github.com/umask000/cs224n-winter-2022)中获取完整笔记，<font color=red>**本系列其他分篇博客发布于**</font>（Updating）：

    [CS224N WINTER 2022（一）词向量（附Assignment1答案）](https://caoyang.blog.csdn.net/article/details/125020572)
    
    [CS224N WINTER 2022（二）反向传播、神经网络、依存分析（附Assignment2答案）](https://blog.csdn.net/CY19980216/article/details/125022559)

    [CS224N WINTER 2022（三）RNN、语言模型、梯度消失与梯度爆炸（附Assignment3答案）](https://blog.csdn.net/CY19980216/article/details/125031727)
    
    [CS224N WINTER 2022（四）机器翻译、注意力机制、subword模型（附Assignment4答案）](https://blog.csdn.net/CY19980216/article/details/125055794)
    
    [CS224N WINTER 2022（五）Transformers详解（附Assignment5答案）](https://blog.csdn.net/CY19980216/article/details/125072701)

----

[toc]

----

## lecture 1 词向量

### slides 

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture01-wordvecs1.pdf)]

1. **以WordNet为例的词库存在的缺陷**：slides p.15

   - 难以辨别单词间细微差别：同义词列表缺少适当性语境范围标注。

   - 缺失单词的最新含义；

   - 编纂具有主观性；

   - 需要耗费人力进行更新与应用；

   - 难以应用于精确计算单词相似度；

2. **分布语义学**（Distributional semantics）：slides p.18

   单词含义可由频繁出现在其附近的单词推定，即通过上下文语境来建模单词表示。

3. **Word2Vec（2013年）词向量1模型的思想**：slides p.21

   ![1.](https://img-blog.csdnimg.cn/87412f1394134ba1ae64a764908040bb.png)
   - 已有目标语言的足量语料库与给定的词汇表；

   - 目的是将给定词汇表中的每个单词表示为一个向量；

   - 对于语料库的每一个单词$c$（称为**中心词**），获取其上下文语境$o$（若干**语境词**构成）；

   - <font color=red>使用单词$c$的词向量与语境$o$中各个单词的词向量的相似度来计算在给定$c$的条件下出现$o$的概率（或反过来在给定$o$的条件下出现$c$的概率，即mask的思想）</font>；

   - 不断调整词向量使得④中的条件概率尽可能的大；

4. **Word2Vec模型的目标函数**：slides p.25
   $$
   \text{minimize}_{\{u_{w_i},v_{w_i}\}_{i=1}^{|v|}}\quad J(\theta)=-\frac1T\sum_{t=1}^T\sum_{-m\le j\le m,j\neq0}\log P(w_{t+j}|w_t;\theta)\tag{1.1}
   $$

   其中概率$P$（称为预测函数）的计算方式如下：

   $$
   P(o|c)=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}\tag{1.2}
   $$

   <font color=red>根据assignment2中的说法，这个结果可以理解为是真实的单词概率分布$y$向量与预测的单词概率分布$\hat y$向量之间的交叉熵。</font>

   式中变量说明：

   ① $T$表示语料库规模（即文本长度）；

   ② $V$表示词汇表；

   ③ $m$表示上下文窗口大小；

   ④ $w_i$表示在第$i$个位置上的单词；

   ⑤ $v_w$表示单词$w$作为中心词的词向量；

   ⑥ $u_w$表示单词$w$作为语境词的词向量；

   ⑦ $\theta$表示超参数；

   <font color=red>Word2Vec模型中每个单词都有两个词向量，最终将两个词向量取均值作为模型输出的词向量。</font>

   因此式$(1.1)$中决策变量总数为$2d|V|$，其中$d$为给定的词向量嵌入维度。

   由于变量数量非常多，因此通常选择随机梯度下降法求解Word2Vec模型。

5. **Word2Vec模型预测函数偏导结果的重要意义**：slides p.29-32
   $$
   \begin{aligned}
   \frac{\partial P(o|c)}{\partial v_c}&=\frac{\partial}{\partial v_c}\log\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}\\
   &=\frac{\partial}{\partial v_c}\log\exp(u_o^\top v_c)-\frac{\partial}{\partial v_c}\log\left(\sum_{w\in V}\exp(u_w^\top v_c)\right)\\
   &=\frac{\partial}{\partial v_c}u_o^\top v_c-\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\frac{\partial}{\partial v_c}\sum_{x\in V}\exp(u_x^\top v_c)\\
   &=u_o-\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\sum_{x\in V}\frac{\partial}{\partial v_c}\exp(u_x^\top v_c)\\
   &=u_o-\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\sum_{x\in V}\exp(u_x^\top v_c)\frac{\partial}{\partial v_c}u_x^\top v_c\\
   &=u_o-\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\sum_{x\in V}\exp(u_x^\top v_c)u_x\\
   &=u_o-\sum_{x\in V}\frac{\exp(u_x^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}u_x\\
   &=u_o-\sum_{x\in V}P(x|c)u_x\\
   &=\text{observed}-\text{expected}
   \end{aligned}\tag{1.3}
   $$
   求解式$(1.1)$时通常使用梯度下降法，需要计算预测函数$P$的梯度。从式$(1.3)$可以看出，偏导值的意义是观测值（即观测到的中心词词向量）与期望值（根据上下文推断的词向量）的差值。

6. **Word2Vec模型的变体**：slides p.42

   - Skip-grams（SG）：给定中心词来预测语境词的分布概率；

   - Continuous Bag of Words（CBOW）：通过词袋形式的语境词来预测中心词的分布概率；

   关于这两个模型详看本节notes部分第2点的相关内容。

### notes

[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)]

1. **基于奇异值分解的词向量建模方法**：notes p.3-5

   基本思想是通过构建与词汇表相关的矩阵$X\in\R^{|V|\times n}$（$V$为词汇表），进行奇异值分解$X=UDW^\top$后，将左奇异向量矩阵$U\in\R^{|V|\times k}$作为词向量输出。其中$k$的取值可以通过主奇异值占比进行划定：
   $$
   \frac{\sum_{i=1}^k\sigma_i}{\sum_{i=1}^{|V|}\sigma_i}\tag{1.4}
   $$
   一些常用的矩阵$X$构建：

   - **单词—文档矩阵**，其中$X_{ij}$表示单词$i$在文档$j$中的出现次数；

   - **基于窗口的共现矩阵**，其中$X_{ij}$表示单词$i$与单词$j$在指定窗口大小下同时出现的次数；

   奇异值方法可能存在的问题：

   - 矩阵$X$维度可能会经常变化（新单词新文档经常会出现）；
   - 矩阵$X$极度稀疏；
   - 矩阵$X$维度极大；
   - 奇异值分解本身计算复杂度较高；
   - 为平衡单词出现频数，往往需要对矩阵$X$进行针对性处理（incorporation of some hacks）；

   缓解上述问题的一些常用方法：

   - 去除停用词；
   - 在**基于窗口的共现矩阵**方法中，出现次数可以根据单词距离进行加权（即距离越远，权重越低）；
   - 使用皮尔森相关性并设置负采样；<font color=red>我理解负采样可以缓解稀疏的问题，如两个单词不共现可以赋负值</font>

2. **基于迭代的方法**：notes p.6-14

   - **语言模型**（LM）：预测文本整体出现的概率。

     Unigram模型：
     $$
     P(w_1,w_2,...,w_n)=\prod_{i=2}^nP(w_i)\tag{1.5}
     $$
     Bigram模型：
     $$
     P(w_1,w_2,...,w_n)=\prod_{i=2}^nP(w_i|w_{i-1})\tag{1.6}
     $$

   - **连续词袋模型**（CBOW）：根据上下文预测中心词，类似词袋模型，但是使用连续向量表示，因而得名。

     标记定义：

     - $n$：词向量维度；

     - $v_w$：输入向量，即$w$作为语境词时的词向量；
     - $u_w$：输出向量，即$w$作为中心词时的词向量；
     - $\mathcal{V}\in\R^{n\times|V|}$：输入单词的嵌入矩阵，$v_i$表示第$i$列（单词$w_i$的输入向量）；
     - $\mathcal{U}\in\R^{|V|\times n}$：输出单词的嵌入矩阵，$u_i$表示第$i$行（单词$w_i$的输出向量）；

     模型用法：

     ① 对输入的上下文构建one-hot词向量：$x^{(c-m)},...,x^{(c-1)},x^{(c+1)},...,x^{(c+m)}\in\R^{|V|}$

     ② 将$(1.7)$中的one-hot词向量转换为输入向量：$\mathcal{V}x^{(c-m)},...,\mathcal{V}x^{(c+m)}\in\R^n$

     ③ 将$(1.8)$中的上下文输入向量取均值：
     $$
     \hat v=\frac1{2m}\sum_{-m\le i\le m,i\neq0}\mathcal{V}x^{(c+i)}\in\R^n\tag{1.7}
     $$
     ④ 生成得分向量：$z=\mathcal{U}\hat v\in\R^{|V|}$

     ⑤ 转为softmax概率分布向量（即得到中心词的概率分布）：$\hat y=\text{softmax}(z)\in\R^{|V|}$

     ⑥ 我们期望生成的概率分布$\hat y$能够匹配真实的概率分布$y$（可以统计全语料库得到）；

     现在的问题是需要已知$\mathcal{U}$与$\mathcal{V}$才能进行上述应用，因此需要定义目标函数来学习两个词嵌入矩阵：
     $$
     \begin{aligned}
     \text{minimize}\quad J&=-\log P(w_c|w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m})\\
     &=-\log P(u_c|\hat v)\\
     &=-\log \frac{\exp(u_c^\top\hat v)}{\sum_{j=1}^{|V|}\exp(u_j^\top \hat v)}\\
     &=-u_c^\top\hat v+\log\sum_{j=1}^{|V|}\exp(u_j^\top\hat v)
     \end{aligned}\tag{1.8}
     $$
     <font color=red>这里将$\hat v$视为上下文的一个表示，这种近似处理显然是很不精准的，但是可以得到可以简明求解的显式目标函数。</font>

   - **Skip-Gram模型**（SG）：根据中心词预测上下文。

     标记定义：

     - $n$：词向量维度；

     - $v_w$：输入向量，即$w$作为语境词时的词向量；
     - $u_w$：输出向量，即$w$作为中心词时的词向量；
     - $\mathcal{V}\in\R^{n\times|V|}$：输入单词的嵌入矩阵，$v_i$表示第$i$列（单词$w_i$的输入向量）；
     - $\mathcal{U}\in\R^{|V|\times n}$：输出单词的嵌入矩阵，$u_i$表示第$i$行（单词$w_i$的输出向量）；

     模型用法：

     ① 生成中心词的one-hot向量：$x\in \R^{|V|}$

     ② 转换为中心词的输入向量：$v_c=\mathcal{V}x\in\R^n$

     ③ 生成得分向量：$z=\mathcal{U}v_c\in\R^{|V|}$

     ④ 转换为softmax概率分布向量（即上下文的概率分布）：$\hat y=\text{sofmax}(z)\in\R^{|V|}$

     ⑤ 这里$\hat y_{c-m},...\hat y_{c-1},\hat y_{c+1},...,\hat y_{c+m}$分别对应上下文各个位置单词的概率分布；

     ⑥ 我们同样期望它与真实分布接近；

     <font color=red>这里课件里可能写错，感觉应该要生成$2m$个得分向量才行，不然有些说不通。</font>

     同样需要定义目标函数来计算$\mathcal{U}$与$\mathcal{V}$：
     $$
     \begin{aligned}
     \text{minimize}\quad J&=-\log P(w_{c-m},...,w_{(c-1)},w_{(c+1)},...,w_{(c+m)}|w_c)\\
     &=-\log\prod_{j=0,j\neq m}^{2m}P(w_{c-m+j}|w_c)\\
     &=-\log\prod_{j=0,j\neq m}^{2m}P(u_{c-m+j}|v_c)\\
     &=-\log\prod_{j=0,j\neq m}^{2m}\frac{\exp(u^{\top}_{c-m+j}v_c)}{\sum_{k=1}^{|V|}\exp(u_k^\top v_c)}\\
     &=-\sum_{j=0,j\neq m}^{2m}u_{c-m+j}^\top v_c+2m\log\sum_{k=1}^{|V|}\exp(u_k^\top v_c)
     \end{aligned}\tag{1.9}
     $$
     <font color=red>这里同样做近似处理，其实本质上是交叉熵损失。</font>

   - **负采样**（<font color=red>suggested readings的第2篇paper</font>）：CBOW与SG的目标函数求解计算复杂度太高。

     因此每一步迭代无需遍历整个词汇表，只需采样若干负样本。

     具体而言，以SG模型为例，$(w,c)$是一组中心词$w$与上下文$c$，$D$是正语料库，$\tilde D$是负语料库（<font color=red>可以理解为假数据，即不是人话的语料</font>），定义：

     ① $P(D=1|w,c)$为$(w,c)$出现在语料库中的概率；

     ② $P(D=0|w,c)$为$(w,c)$没有出现在语料库中的概率；

     则可以利用激活函数建模：
     $$
     \begin{aligned}
     P(D=1|w,c,\theta)&=\sigma(v_c^\top v_w)=\frac1{1+e^{-v_c^\top v_w}}\\
     P(D=0|w,c,\theta)&=1-\sigma(v_c^\top v_w)=\frac1{1+e^{v_c^\top v_w}}
     \end{aligned}\tag{1.10}
     $$
     最大似然法求解参数$\theta$：
     $$
     \begin{aligned}
     \theta&=\text{argmax}_\theta\prod_{(w,d)\in D}P(D=1|w,c,\theta)\prod_{(w,d)\in\tilde D}P(D=0|w,c,\theta)\\
     &=\text{argmax}_\theta\sum_{(w,c)\in D}\log\frac{1}{1+\exp(-u_w^\top v_c)}+\sum_{(w,c)\in\tilde D}\log\frac{1}{1+\exp(u_w^\top v_c)}
     \end{aligned}\tag{1.11}
     $$
     对于SG模型，负采样的目标函数（给定中心词$c$，观测到上下文位置$c-m+j$单词）发生变化：
     $$
     -u_{c-m+j}^\top v_c+\log \sum_{k=1}^{|V|}\exp(u_k^\top v_c)\longrightarrow-\log\sigma(u_{c-m+j}^\top v_c)-\sum_{k=1}^K\log \sigma(-\tilde u_{k}v_c)\tag{1.12}
     $$
     对于CBOW模型，负采样的目标函数（给定上下文$\hat v$，观测中心词$u_c$）发生变化：
     $$
     -u_c^\top\hat v+\log\sum_{j=1}^{|V|}\exp(u_j^\top\hat v)\longrightarrow -\log\sigma(u_c^\top \hat v)-\sum_{k=1}^K\log\sigma(-\tilde u_k^\top \hat v)\tag{1.13}
     $$
     其中$\{\tilde u|k=1,...,K\}$采样自噪声分布$P_n(w)$

   - **分层softmax**（<font color=red>suggested readings的第2篇paper</font>）：

     这个感觉不是很重要，我理解是softmax向量可能太长，因此可以分层处理，如每次计算两分类的概率分布，以二叉树的形式不断累进，于是复杂度可以由$O(|V|)$变为$O(\log |V|)$

### suggested readings

两篇推荐阅读都是2013年的paper：

第一篇首次提出**Skip-Gram模型**与**CBOW模型**（[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)）

第二篇首次提出**负采样**与**分层softmax**的概念（[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)）

理论内容大部分已在slides与notes部分详细阐述（在**lecture2**中也有对这两篇paper的内容有所提及），简单记录要点：

1. 关于负采样的统计概念是**噪声对比估计**（Noise Contrastive Estimation）；

2. 高频单词的**欠采样**（Subsample）：定义单词$w_i$的采样概率为
   $$
   P(w_i)=1-\sqrt{\frac t{f(w_i)}}\tag{1.14}
   $$
   其中，$f(w_i)$为单词$w_i$的频数，$t$为给定的阈值，量级通常为$10^{-5}$。

   去除停用词可以视为是一种特殊的**欠采样**

3. SG模型的负采样全局目标函数为（对比式$(1.12)$）：
   $$
   \text{maximize}\quad J(\theta)=\frac1T\sum_{t=1}^TJ_t(\theta)\tag{1.15}
   $$
   其中：
   $$
   J_{t}(\theta)=\log\sigma(u_o^\top v_c)+\sum_{i=1}^k\mathbb{E}_{j\sim P(w)}\log\sigma(-u_k^\top v_c)\tag{1.16}
   $$
   或者更贴近**assignment2**的写法是：
   $$
   J_{\text{neg-sample}}(u_o,v_c,\mathcal{U})=-\log\sigma(u_o^\top v_c)-\sum_{k\in\{K\text{ sampled indices}\}}\log\sigma(-u_k^\top v_c)\tag{1.17}
   $$
   即采样$K$个负样本（利用单词的分布概率），最大化真实的上下文单词出现的概率，最小化中心词附近出现随机单词（负样本）的概率。（注意负样本的词向量在$\sigma$函数中取的是负值）

   <font color=red>这里很tricky的事情是，作者提出采用Unigram分布概率（即单个单词的分布概率）$U(w)$的$3/4$次方，即负采样概率为$P(w)=CU(w)^{3/4}$（$C$为概率正则化系数，确保总概率为一），他们声称这种负采样方式是非常有效的，因为可以使得低频词能够更多的被采样到。</font>

### gensim word vectors example

[[code](http://web.stanford.edu/class/cs224n/materials/Gensim.zip)] [[preview](http://web.stanford.edu/class/cs224n/materials/Gensim word vector visualization.html)]    

关于Gensim库对GloVe词向量可视化的小demo，可作入门学习使用。

其中使用的GloVe词向量项目主页可见[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)。

demo中使用的GloVe.6B可以从[https://nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)处下载得到。

demo中最后使用PCA对给定集合的单词词向量进行降维后取前二的主成分值在二维平面上进行可视化，近似刻画单词之间的距离。

```python
import numpy as np

# Get the interactive Tools for Matplotlib
%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/Users/manning/Corpora/GloVe/glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.most_similar('obama')
model.most_similar('banana')
model.most_similar(negative='banana')

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

analogy('japan', 'japanese', 'australia')
analogy('australia', 'beer', 'france')
analogy('obama', 'clinton', 'reagan')
analogy('tall', 'tallest', 'long')
analogy('good', 'fantastic', 'bad')
print(model.doesnt_match("breakfast cereal dinner lunch".split()))

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
        
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
        
display_pca_scatterplot(model, 
                        ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',
                         'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
                         'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',
                         'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',
                         'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',
                         'homework', 'assignment', 'problem', 'exam', 'test', 'class',
                         'school', 'college', 'university', 'institute'])

display_pca_scatterplot(model, sample=300)
```

### assignment1 参考答案

[[code](http://web.stanford.edu/class/cs224n/assignments/a1.zip)] [[preview](http://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html)]    

**Assignment1参考答案**：囚生CYの[GitHub Repository](https://github.com/umask000/cs224n-winter-2022/tree/main/cs224n-winter2022-solutions/assignment1/a1)

可将nltk_data下载至本地，[GitHub@NLTK](https://github.com/nltk/nltk_data)链接中的package文件夹即为nltk_data（如果GitHub下载太慢可自行搜索网盘资源），下载后将package文件夹名称重命名为nltk_data，并设置NLTK_DATA环境变量为nltk_data文件夹以便于使用（如若不想设置环境变量可以将nltk_data文件夹移动到任意磁盘的根目录下，nltk工具包默认会搜索这些路径，如D:/nltk_data）。

注意package文件夹下所有的语料、分词器、模型等文件都已打包，如若需要使用通常需要对指定的压缩包进行解压，如assignment1中使用到的reuters语料，需要提前将corpora目录下的reuters.zip解压到当前文件夹才能正常使用。

此外建议下载GloVe词向量（https://nlp.stanford.edu/projects/glove/）至本地，JupyterNotebook中的已作相关注释，注意查阅。

笔者使用的gensim版本3.8.1，经验上感觉gensim4.x.x版本与gensim3.x.x版本差别较大（比如gensim4.x.x版本完全找不到BM25算法的接口），建议尽量使用3.x.x版本。


----

## lecture 2 词向量与单词窗口分类

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture02-wordvecs2.pdf)]

1. **直接在原始的共线计数（co-occurence counts）上进行奇异值分解效果会很差！**slide p.18

   比如LSA话题模型（使用**单词—文档**矩阵），会先将**单词—文档**矩阵中统计得到的频数使用TFIDF算法转化为TFIDF的指标值，然后再进行奇异值分解。

   理论解释是频数的分布过于离散，可以使用一些方法先将它们映射到某个固定区间上，如可以取对数，设定最大限值，去除高频停用词等。

2. **两类词向量对比**：slides p.20

   - 基于计数方法的词向量（如LSA模型）：实现简单快速，但仅仅捕获单词相似性，应用效果不好。
   - 基于直接预测的词向量（如CBOW模型）：相对训练较慢，但可以捕获复杂的语义模式，应用效果更好。

3. **对数双线性模型**：slides p.23
   $$
   w_i\cdot w_j=\log P(i|j)\\
   w_x\cdot(w_a-w_b)=\log\frac{P(x|a)}{P(x|b)}\tag{2.1}
   $$
   这个类似知识图谱中基于语义的知识表示方法，旨在使训练得到的词向量能够近似满足单词实际的共现概率。

4. **GloVe词向量模型**（suggested readings第1篇paper）[GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162) (Pennington et al., EMNLP 2014)：slides p.24

   使用的思想即为式$(2.1)$所示，具体而言。目标函数为：
   $$
   J=\sum_{i,j=1}^{|V|}f(X_{ij})(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2\tag{2.2}
   $$
   slides中没有解释该目标函数，大致理解了一下，应该是如下的含义：

   ① $X$是基于窗口的单词共现矩阵，$f(X_{ij})$是对共现矩阵值进行正则化处理（本节第1点所述）；

   ② $w_i$是单词$i$的词向量，应该是作为中心词的词向量；

   ③ $\tilde w_i$根据论文中的说法叫做separate context word vectors，理论上就是作为语境词的词向量；

   ④ $b_i$和$\tilde b_j$是模型中的截距项；

   在本节的notes部分有较为详细的阐述。

5. **一种直观上的词向量评估方法**：slides p.34

   使用[http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)提供的人类标注的单词相似度数据表来辅助评估。

   常规的评估方法可以使用类比法（word vector analogies）：
   $$
   d=\text{argmax}_i\frac{(x_b-x_a+x_c)^\top x_i}{\|x_b-x_a+x_c\|}\tag{2.3}
   $$
   式$(2.3)$可以理解为给定一种相似关系man:woman，试找出king对应的单词是什么？加入能够找到完美契合的单词，那么说明词向量是好的。

   当然实际上都是讲词向量用于下游任务来间接评估其优劣性。

   感觉或许这里的词向量评估相关的内容对assignment1中后半部分的一些书面问题是有一些帮助的。

6. 后面的内容大都关于基础机器学习的回顾，较为浅显不赘述。

### notes

[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)]

1. **GloVe词向量模型**：notes p.1-3

   其思想是利用全局的统计信息来预测单词$j$出现在给定单词$i$所在的语境下的概率（即单词$j$与单词$i$共现的概率），目标函数是加权最小二乘。

   - **单词—单词共现矩阵**$X$：

     标记说明：

     ① $X_{ij}$表示单词$i$与$j$共现的计数；

     ② $X_i=\sum_{k}X_{ik}$表示单词$i$语境下出现的所有单词数量；

     ③ $P_{ij}=P(w_j|w_i)=X_{ij}/X_i$表示单词$j$出现在单词$i$语境下的概率；

   - **最小二乘目标**：

     回顾在SG模型中使用的是softmax来计算单词$j$出现在单词$i$语境下的概率:
     $$
     Q_{ij}=\frac{\exp(u_j^\top v_i)}{\sum_{w=1}^W\exp(u_w^\top v_i)}\tag{2.4}
     $$
     则全局的损失函数可以写作：
     $$
     J=-\sum_{i\in\text{corpus}}\sum_{j\in\text{context}(i)}\log Q_{ij}\tag{2.5}
     $$
     由于单词$i$与单词$j$可能多次共现，因此共现计数加权后的损失函数为：
     $$
     J=-\sum_{i=1}^{|V|}\sum_{j=1}^{|V|}X_{ij}\log Q_{ij}\tag{2.6}
     $$
     将$Q$矩阵正则化后，即可与$P$矩阵进行对照，得到最小二乘的损失函数：
     $$
     \hat J=\sum_{i=1}^{|V|}\sum_{j=1}^{|V|}X_{ij}(\hat P_{ij}-\hat Q_{ij})^2\tag{2.7}
     $$
     其中$\hat P_{ij}=X_{ij},\hat Q_{ij}=\exp(u_j^\top v_i)$，这两个数值都是没有经过正则化的，取值过大会影响优化求解，因此可以考虑取对数：
     $$
     \hat J=\sum_{i=1}^{|V|}\sum_{j=1}^{|V|}X_{i}(\log\hat P_{ij}-\log\hat Q_{ij})^2=\sum_{i=1}^{|V|}\sum_{j=1}^{|V|}X_i(u_j^\top v_i-\log X_{ij})^2\tag{2.8}
     $$
     此外可能权钟$X_i$并不那么可靠，可以引入一个更一般的权重函数，即：
     $$
     \hat J=\sum_{i=1}^{|V|}\sum_{j=1}^{|V|}f(X_{ij})(u_j^\top v_i-\log X_{ij})^2\tag{2.9}
     $$
     这就得到了式$(2.2)$的形式。

2. **窗口分类**（window classification）：notes p.12-13

   这个概念是与词向量评估中的extrinsic类型的方法相联系的（本笔注没有详细注释，因为感觉不是很重要）。

   所谓extrinsic类型的词向量评估方法指的是构建一个新的机器学习任务来对已经生成好的词向量进行分类，这里每个单词都会人工标注多类别的标签值，然后看看是否能够构建一个很好的机器学习模型来成功区分这些词向量，在这个过程中，可以通过重训练（retrain）来提升词向量的效果。notes中指出，<font color=red>重训练是具有风险的</font>。

   > 关于重训练的概念，似乎没有找到很具体的说明，我根据自己的理解，大概的意思是根据extrinsic任务的效果，将那些未能分类准确的单词再拿出来重新学习词向量，但是具体要怎么做这就比较迷，可能就只是调调词向量模型的参数，这个问题暂时未解。

   然后所谓窗口分类就是不只是对一个单词分类，而是对一个窗口内的所有单词构成的上下文进行分类。

### suggested readings

1. GloVe词向量模型首次提出的paper，发表于EMNLP2014，具体已经在slides与notes部分详述原理。记录一下[项目主页](http://nlp.stanford.edu/projects/glove)，词向量数据都可以公开下载。（[GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf)）
2. 一篇关于词向量评估的paper，发表于TACL2015，感觉比较普通，文中实现了若干不同的词向量，并使用了wordsim353数据集进行了词向量生成的评估。（[Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)）
3. 也是一篇关于词向量评估的paper，发表于EMNLP2015，主要讲得就是intrinsic与extrinsic两种类型的评估方法。（[Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)）

### additional readings

1. 关于基于语义的词向量生成方法，发表于TACL2016，标题中的**PMI**（Pointwise Mutual Information）已经多次提到，翻译过来叫作逐点互信息，在这里它用来刻画两个单词的相关度。（[A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)）
2. 关于多义词词向量的一篇paper，发表于TACL2018，它的思想是线性加权，具体而言它认为多义词的词向量可由其各个词义的词向量通过加权求和表示，权重为不同词义出现的频率，感觉如果要做词义消歧方面的工作，可以再参考这篇文章的做法，相对来说还是比较新的研究了。（[Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)）
3. 理论性非常强的一篇paper，发表于NIPS2018，研究对象是词向量维度的选取上的权衡，感觉不太能用得到。（[On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)）

### python review session

[[slides](http://web.stanford.edu/class/cs224n/readings/cs224n-python-review.pdf)] [[notebook](http://web.stanford.edu/class/cs224n/readings/python_tutorial.ipynb)]    

这是一个非常基础的Python入门教程，笔者认为没有必要进行注释，课件中提供了从安装到应用的全过程的文档，以及一个JupyterNotebook的demo，有需求的可以自行查阅。

----
