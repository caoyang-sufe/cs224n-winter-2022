# CS224N WINTER 2022 （六）前沿问题探讨（QA、NLG、知识集成与检索、Coreference）

[CS224N WINTER 2022（一）词向量（附Assignment1答案）](https://caoyang.blog.csdn.net/article/details/125020572)
[CS224N WINTER 2022（二）反向传播、神经网络、依存分析（附Assignment2答案）](https://blog.csdn.net/CY19980216/article/details/125022559)
[CS224N WINTER 2022（三）RNN、语言模型、梯度消失与梯度爆炸（附Assignment3答案）](https://blog.csdn.net/CY19980216/article/details/125031727)
[CS224N WINTER 2022（四）机器翻译、注意力机制、subword模型（附Assignment4答案）](https://blog.csdn.net/CY19980216/article/details/125055794)
[CS224N WINTER 2022（五）Transformers详解（附Assignment5答案）](https://blog.csdn.net/CY19980216/article/details/125072701)

# 序言

第十讲往后属于自然语言处理前沿领域的内容，这部分内容目前暂且过了一遍，做了少量的笔注，因为很多最新的研究和经典的成果仍需细读推荐阅读中提供的paper才能有所得，目前只是做记录性质的，问答（QA）这部分是很有趣的，知识集成与检索又与问答是息息相关的，Coreference就是我们常说的本体识别问题，自然语言生成是一个很宽广的研究领域，大部分的seq2seq任务都可以视为NLG

CS224N告一段落。

----
[toc]

----

## lecture 11 问答系统

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/Danqi-QA-slides-2022.pdf)]     

1. **问答系统的类型**：

   - **答案参考来源**：文本信息、网络文档、知识库、表格、图片。
   - **问题类型**：陈述句（factoid）或非陈述句（non-factoid）、开放领域（open-domain）或限定领域（closed-domain）、简单（simple）问题或复杂（compositional）问题。
   - **答案类型**：一句话、一段话、枚举所有结果、判断题。

2. **[Freebase](http://www.freebase.be/)**：基于非结构化的文本

   ![freebase](https://img-blog.csdnimg.cn/7047ff5ab7944835a606b55b7b1c1522.png)

3. **[SQuAD](https://stanford-qa.com/)**：斯坦福问答数据集，下面推荐阅读部分有较为详细的说明；

   其他的一些问答数据集：TriviaQA，Natural Questions，HotpotQA；

   **如何构建模型解决SQuAD**：

   - 模型输入：$C=\{c_1,...,c_N\},Q=(q_1,...,q_M),c_i\in V,q_i\in V$，其中$N\approx100,M\approx15$

   - 模型输出：$1\le\text{start}\le \text{end}\le N$

   - 2016~2018年大部分使用的是带注意力机制的LSTM模型：

     <img src="https://img-blog.csdnimg.cn/68a92cce0ee3416493a468f00e64c4db.png" alt="请添加图片描述" style="zoom:50%;" />

     <font color=red>下面介绍上图这种BiDAF模型框架的思想（参考论文是[推荐阅读的第二篇](https://arxiv.org/abs/1611.01603)）。</font>

     输入的$C$到一个BiLSTM中，输入Q到另一个BiLSTM中，然后将两者的隐层状态取注意力，继续输出到两层的BiLSTM中，最后由全连接层输出结果。（这个模型架构叫作BiDAF）

     具体而言，BiDAF的嵌入用的GloVe拼接上CNN编码得到的嵌入charEmb：
     $$
     \text{emb}(c_i)=f([\text{GloVe}(c_i);\text{charEmb}(c_i)])\\
     \text{emb}(q_i)=f([\text{GloVe}(q_i);\text{charEmb}(q_i)])
     $$
     然后输入到BiLSTM中：
     $$
     \overset{\rightarrow}{c_i}=\text{LSTM}(\overset{\rightarrow}{c_{i-1}},e(c_i))\in\R^H\\
     \overset{\leftarrow}{c_i}=\text{LSTM}(\overset{\leftarrow}{c_{i-1}},e(c_i))\in\R^H\\
     {\bf c}_i=[\overset{\rightarrow}{c_i};\overset{\leftarrow}{c_i}]\in\R^{2H}\\
     \overset{\rightarrow}{q_i}=\text{LSTM}(\overset{\rightarrow}{q_{i-1}},e(q_i))\in\R^H\\
     \overset{\leftarrow}{q_i}=\text{LSTM}(\overset{\leftarrow}{q_{i-1}},e(q_i))\in\R^H\\
     {\bf q}_i=[\overset{\rightarrow}{q_i};\overset{\leftarrow}{q_i}]\in\R^{2H}
     $$
     接下来的注意力层就是计算${\bf c}_i$和${\bf q}_i$的点积注意力，这里有两个注意力（context-to-query attention与query-to-context attention）

     <img src="https://img-blog.csdnimg.cn/dd91a2c985b043638420a86c61a89d33.png" alt="c2qatt" style="zoom:50%;" />
     <img src="https://img-blog.csdnimg.cn/dd91a2c985b043638420a86c61a89d33.png" alt="c2qatt" style="zoom:50%;" />

     具体而言有如下表达式：
     $$
     S_{i,j}=w_{\rm sim}^\top[{\bf c}_i;{\bf q}_i;{\bf c}_i\odot{\bf q}_i]\in\R\quad w_{\rm sim}\in\R^{6H}
     $$
     context-to-query attention（每个问题单词与$c_i$的关联性）如下计算：
     $$
     \alpha_{i,j}=\text{softmax}_j(S_{i,j})\in\R\quad{\bf a}_i=\sum_{j=1}^M\alpha_{i,j}{\bf q}_j\in\R^{2H}
     $$
     query-to-context attention（每个上下文你单词与$q_i$的关联性）如下计算：
     $$
     \beta_{i}=\text{softmax}_i(\max_{j=1}^M(S_{i,j}))\in\R^N\quad {\bf b}=\sum_{i=1}^N\beta_i{\bf c_i}\in \R^{2H}
     $$
     最终的注意力输出为：
     $$
     {\bf g}_i=[{\bf c}_i;{\bf a}_i;{\bf c}_i\odot{\bf a}_i;{\bf c}_i\odot {\bf b}_i]\in\R^{8H}
     $$
     再往上的Modeling层就比较简单了：
     $$
     m_i=\text{BiLSTM}({\rm g_i})\in\R^{2H}
     $$
     最后的Output层得到预测答案的前后位置标签：
     $$
     p_{\rm start}=\text{softmax}(w_{\rm start}^\top[{\bf g}_i;{\bf m}_i])\\
     p_{\rm end}=\text{softmax}(w_{\rm end}^\top[{\bf g}_i;{\bf m}_i'])\\
     {\bf m}_i'=\text{BiLSTM}({\bf m}_i)\in \R^{2H}\\
     w_{\rm start},w_{\rm end}\in\R^{10H}
     $$
     目标函数是$\text{start}$和$\text{end}$的交叉熵损失：
     $$
     \mathcal{L}=-\log p_{\rm start}(s^*)-\log p_{\rm end}(e^*)
     $$
     这个模型最后的效果是F1得分77.3，消融研究结果为：

     ① 不带context-to-query attention得到F1得分67.7；

     ② 不带query-to-context attention得到F1得分73.7；

     ③ 不带charEmb得到F1得分75.4；

     目前最先进的模型（2017年）可以达到79.7的水平。

   - 2019年之后大多用的是用于阅读理解的BERT或其变体模型的微调：slides p.31

     基本超过了人类的水平：

     |            |  F1  |  EM  |
     | :--------: | :--: | :--: |
     |  人类水平  | 91.2 | 82.3 |
     |   BiDAF    | 77.3 | 67.7 |
     | BERT-base  | 88.5 | 80.8 |
     | BERT-large | 90.9 | 84.1 |
     |   XLNet    | 94.5 | 89.0 |
     |  ReBERTa   | 94.6 | 88.9 |
     |   ALBERT   | 94.8 | 89.3 |

     但是BiDAF的参数量只有2.5M，BERT及其变体至少也是110M~330M的水平，且前者的预训练仅限于GloVe，后者的预训练就很多了。

     虽然先进的模型已经在SQuAD超越了人类，但是这并不意味着阅读理解问题已经被解决，因为在领域外的问题上表现依然很差（TriviaQA，NQ，QuAC，NewsQA）：[What do Models Learn from Question Answering Datasets?](https://arxiv.org/abs/2004.03490)

     ![What do Models Learn from Question Answering Datasets?](https://img-blog.csdnimg.cn/7fb1280a87ca4ef19a6d82cf4997da99.png#pic_center)

4. 很多自然语言处理任务都可以被归约到**阅读理解**：如信息提取、语义角色标注；

5. 是否可以设计更好的预训练目标？

   [SpanBERT](https://arxiv.org/abs/1910.10683)提出了两个新的想法：

   > - masking contiguous spans of words instead of 15% random words
   >
   > - using the two end points of span to predict all the masked words in between = compressing the information of a span into its two endpoints
   >   $$
   >   y_i=f(x_{s-1},x_{e+1},p_{i-s+1})
   >   $$

6. 文档检索类型的QA模型框架（解决开放领域问题）：推荐阅读第三篇[DrQA](https://arxiv.org/pdf/1704.00051.pdf)。

   思想就是有一个文档检索器先检索文档，然后从文档中摘取答案。

   这里的检索器就是TFIDF，阅读器就跟BiDAF做的事情差不多。

   注意我们也可以训练一个检索器（就笔者的经验而言，感觉没这个必要）。

7. 最新的研究还有是生成答案，而非检索答案，相关研究有：

   [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)

8. 一些大语言模型也可以用于开放领域问答，主要就是预测被挖掉的单词，本身填空题也是一种问答：

   [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910)

9. 问答文档中通常包含大量的短语（远远超过词汇表规模，以维基百科为例这个数字是600亿），因此使用稠密的向量表示是不现实的，推荐阅读最后一篇就是讲得这个问题，如何学习短语的稠密表示。另外推荐一篇这个方向的研究[Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index](https://arxiv.org/abs/1906.05807)

### notes

[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes07-QA.pdf)]

本讲的笔记部分详细讲述了一个用于解决基于文本和图像的问答任务的**动态内存网络**（Dynamic Memory Networks，DMN，[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285)）。DMN启发自QA的困难性，即使对于人类来说也很难存储一大段文字到闹钟，因此我们要使用动态内存。

看到有人做了阐述与代码实现：[知乎](https://zhuanlan.zhihu.com/p/43642351)

接下来介绍DMN的各个模块：

![dmn](https://img-blog.csdnimg.cn/9fd6ca57421b4360baf17a5f07b1173c.png)

1. **输入模块**（Input Module）：将长度为$T_1$的的单词序列作为输入，长度为$T_C$的事实表示作为输出。若输出是一系列单词，则有$T_1=T_C$；若输出是一系列句子，则$T_1$等于句子中的单词数量。

   用GRU读取句子，即$h_t=\text{GRU}(x_t,h_{t-1})$，其中$x_t=L[w_t]$，其中$L$是嵌入矩阵，$w_t$是在时间点$t$处的单词。

2. **问题模块**（Question Module）：

   再用一个标准GRU来读取问题：$q_t=\text{GRU}(L[w_t^Q],q_{t-1})$，问题模块的输出是问题的编码表示。

3. **片段内存模块**（Episodic Memory Module）：<font color=red>这是重中之重</font>

   思想：将句子级别的表示（来自输入模块）输入到双向GRU中，并生成片段内存表示。

   记片段内存表示为$m^i$，记**片段表示**（episode representation，注意力的输出）为$e^i$，则初始化片段内存表示为$m^0=q$，然后依次生成$m^i=\text{GRU}(e^i,m^{i-1})$，接着**片段表示**通过输入模块的隐层状态输出不断更新：
   $$
   h_t^i=g_t^i\text{GRU}(c_t,h_{t-1}^i)+(1-g_t^i)h_{t-1}^i\quad e_i=h_{T_C}^i
   $$
   其中注意力向量$g$可以用各种方法生成，但是在[最初的这篇DMN论文](https://arxiv.org/abs/1506.07285)中，提出下面的计算方法是最优的：
   $$
   g_t^i=G(c_t,m^{i-1},q)\\
   G(c,m,q)=\sigma(W^{(2)}\tanh(W^{(1)}z(c,m,q)+b^{(1)})+b^{(2)})\\
   z(c,m,q)=[c,m,q,c\circ q,c\circ m,|c-q|,|c-m|,c^\top W^{(b)}q,c^\top W^{(b)}m]
   $$
   这样的话，若句子与内存中存储的东西或直接与问题相关联，模块中的门控就会被激活，在第$i$步迭代中，若总结得到的知识不足以解答问题，我们可以继续进行第$i+1$步迭代。比如当提问问题Where is the football?，输入序列为John kicked the football与John was in the field，在这个例子中，John和football都可能在第1次迭代中被联系到，使得网络能够基于这两段信息执行一次**传递性推理**（transitive inference）。

4. **回答模块**（Answer Module）：

   仍然是一个GRU解码器，将问题模块的输出、片段内存模块的输出作为输入，输出一个单词（或一个计算结果）：
   $$
   y_t=\text{softmax}(W^{(a)}a_t)\\
   a_t=\text{GRU}([y_{t-1},q],a_{t-1})
   $$

### suggested readings

1. （[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf)）

   包含100K标注好的文章（从维基百科收集，词数在100~150）、问题（来自众筹）、答案（文章中的一小段语句）三元组。SQuAD任务目前**几乎已经解决**，但仍然是最热门的问答数据集之一，持续有相关研究。

   模型评估指标是精确匹配度（即完全找对片段，只有零一两个取值）与F1得分（即计算找到的片段和真实片段的准确率与召回率，然后取调和平均），人类的表现分别是0.823与0.912，具体如下（有的问题会提供多个标准答案）：
   $$
   \text{Q: What did Tesla do in December 1878?}\\
   \text{A: {left Graz, left Graz, left Graz and servered all relations with his familty}}\\
   \text{Prediction: {left Graz and severed}}\\
   \text{Exact match: max}\{0,0,0\}=0\\
   \text{F1-score: max}\left\{0.67,0.67,0.61\right\}
   $$

2. BiDAF模型架构提出文，上面slides部分已经非常详细说明了思想与原理。（[Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/pdf/1611.01603.pdf)）

3. DrQA模型提出文，从文档库中检索文档，再从文档中检索答案解决开放问题。（[Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf)）

4. 联合训练文档检索器和文档阅读器解决开放问题。（[Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/pdf/1906.00300.pdf)）

5. DPR模型，也是用BERT训练文档检索器，但是使用的是问题答案的样本对来训练。（[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)）

6. 这是关于学习短语的（[Learning Dense Representations of Phrases at Scale](https://arxiv.org/pdf/2012.12624.pdf)）

----

## lecture 12 自然语言生成

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture12-generation-final.pdf)]

1. 可以这么理解：自然语言处理（NLP）=自然语言理解（NLU）+自然语言生成（NLG），seq2seq任务通常属于NLG范畴。

2. 解码器中的TopK采样（推荐阅读部分第一篇和第三篇）：slides p.31

   即使我们能够将输出控制在很小的一个词汇范围内，剩下的候选单词构成的概率和可能也是庞大的，统计上称为**厚尾**（heavy tailed），尽管如此，许多候选单词的确大概率都是错的。一般来说K的取值在5/10/20；

   其实这种TopK采样有点类似束搜索，以推荐阅读第三篇为例，它是自动生成故事，因此每次会有10个候选词（K=10），但是每次都生成10个词又太长，因此隔几步就要剪枝。

   然后还提出了一个很重要的概念叫作**softmax temperature**，即对softmax输出进行一个scaling，具体而言：
   $$
   P_t(y_t=w)=\frac{\exp(S_w)}{\sum_{w'\in V}\exp(S_{w'})}\longrightarrow P_t(y_t=w)=\frac{\exp(S_w/\tau)}{\sum_{w'\in V}\exp(S_{w'}/\tau)}
   $$
   这里的$\tau$称为温度参数，用以重新平衡$P_t$：

   - 若$\tau>1$，概率分布趋向于均匀，以便于得到更多的候选输出。
   - 若$\tau<1$，概率分布趋向于突兀（旱的旱死，捞的涝死），以便于得到更少的候选输出。

   <font color=red>永远记住softmax并非是一个解码算法，它只是一个可以用于测试的技术，你可以基于softmax输出设计解码算法（如束搜索或采样）。</font>

   如果解码器的解码出的序列很差，可以考虑多解码一些结果用于候选重排序，这在推荐阅读中都是有提到的。

本节很多内容与**lecture6**重合。

### suggested readings

1. slides p.31提到的NLG解码中的TopK采样。（[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751.pdf)）
2. 文本综述。（[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368.pdf)）
3. slides p.31提到的NLG解码中的TopK采样。（[Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833.pdf)）
4. 对话系统。（[How NOT To Evaluate Your Dialogue System](https://arxiv.org/abs/1603.08023.pdf)）

----

## lecture 13 将知识集成到语言模型

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture-knowledge.pdf)]

1. 语言模型可以用于其他生成文本的任务，如：

   ① 文本综述（Summarization）

   ② 对话系统（Dialogue）

   ③ 自动补全（Autocompletion）

   ④ 机器翻译（Machine learning）

   ⑤ 流畅度评估（Fluency evaluation）

2. **知识感知的语言模型**：

   这里有讲到text-to-SQL，即将自然语言问题转为SQL查询语句。

   这里的意思是将语言模型直接作为知识库，因为语言模型大多已经在海量的非结构化非标注的文本上训练，而知识库需要手动标注。

   但是使用语言模型缺少可解释性、信任度差、且不易调整。

3. **将知识集成到语言模型中**：

   - **将预训练好的实体嵌入添加到单词嵌入中**：ERNIE，QAGNN/GreaseLM

     这里提到比如USA，United States of America，America都是指美国，但是我们实际上会得到三个词向量，因此我们需要引入实体嵌入来告诉模型这仨兄弟是一个人，如果想要将实体嵌入用到文本中，必须先做**实体链接**（entity linking）。

     训练实体嵌入，往往需要知识图谱的方法（[TransE](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)），也可以使用**word-entity co-ocuurence methods**（如[Wikipedia2Vec](https://aclanthology.org/2020.emnlp-demos.4.pdf)），以及使用Transformer对实体描述进行编码（[BLINK](https://arxiv.org/pdf/1911.03814.pdf)）。

     那么得到了实体链接$e_k$后又怎么添加到单词嵌入$w_j$中呢？
     $$
     h_j=F(W_tw_j+W_ee_k+b)
     $$
     其中$e_k$与$w_j$是对应的实体与单词，直觉上$W_te_k$与$W_ew_j$应当在同一向量空间中，因为实体与单词之间存在alignment

     ----

     这种方法的论文ERNIE在推荐阅读部分有详细记录，这里介绍另一个[QAGNN/GreaseLM](https://arxiv.org/abs/2104.06378)，是关于知识图谱与语言模型来进行推理的，用的模型是图神经网络（这篇paper是2021年，非常新的一份研究，值得细看）。

     <img src="https://img-blog.csdnimg.cn/716e56d1fe634393818518cc314330cb.png#pic_center" alt="QAGNN" style="zoom:50%;" />

   - **使用外部存储**：

     上一种方法的问题在于如果知识库改变了，你就需要重新训练实体嵌入，模型也要重新训练。

     因此这里考虑直接使用外部存储，核心思想是在知识图谱的条件下建模语言模型概率。

     当我们在处理语句序列时，顺便构造一个局部知识图谱：

     ![KGLM](https://img-blog.csdnimg.cn/9e8bb3264fdd4690953f908ebbe2c767.png#pic_center)

     注意局部知识图谱应当是完整知识图谱的一个子集，局部知识图谱能够给出一个很强的信号用于预测下一个单词会是什么，具体预测下一个单词方法如图所示：

     ![KGLM2](https://img-blog.csdnimg.cn/b0512e871ff84d0484c571aec2f290f4.png#pic_center)

     我们并不是直接预测下一个单词，而是用语言模型的隐层状态来预测下一个单词的类型，然后分三种情况（对应上图右边框内的结果）：

     1. **在局部KG中的关联实体**：在局部KG中找到得分最高的父节点以及关系（使用语言模型的隐层状态，以及实体和关系的嵌入）：
        $$
        P(p_t)=\text{softmax}(v_p\cdot h_t)
        $$
        其中$p_t$是潜在的父实体，$v_p$是对应的实体嵌入，$h_t$是LM隐层状态，这类似在预测top关系和实体，然后：

        ① 下一个实体将会是KG三元组中的尾实体（top父实体，top关系，尾实体）；

        ② 下一个单词将会是与下一个实体最接近的单词。

     2. **不在局部KG中的关联实体**：找到整体KG得分最高的实体（使用语言模型的隐层状态和实体嵌入），则：

        ① 下一个实体将会是得分最高的预测实体；

        ② 下一个单词将会是与下一个实体最接近的单词。

     3. **非实体**：下一个实体就是None，下一个单词只能用语言模型来预测了。

   - **改动训练数据**：能否间接地将知识集成到非结构化文本中？（推荐阅读第三篇[Pretrained Encyclopedia](https://arxiv.org/pdf/1912.09637.pdf)）

     核心思想是训练模型以区分真假知识，具体方法是将文本中的实体词替换为同类型但不同实体指向的实体词，以构建**负知识陈述句**（negative knowledge statements），即让模型预测是否实体被替换了。比如：J.K. Rowling is the author of Harry Potter是真的，但是把人名换一个就是假的。

     在推荐阅读第三篇paper中定义了一个**实体替换损失**的目标函数以使得模型能够区分真假句子：
     $$
     \mathcal{L}_{\rm entRep}=\textbf{1}_{e\in\mathcal{E}^+}+\log P(e|C)+(1-\textbf{1}_{e\in\mathcal{E}^+})\log(1-P(e|C))
     $$
     其中$e$是一个实体，$C$是上下文，$\mathcal{E}^+$表示真实实体词。

4. 第三部分的内容是评估语言模型中的知识，具体看推荐阅读部分的最后一篇。

5. 一些其他最新的进展：

   - Retrieval-augmented language models：

     [REALM, Guu et al., ICML 2020](https://arxiv.org/abs/2002.08909)
     [RAG, Lewis et al., NeurIPS 2020](https://arxiv.org/abs/2005.11401)
     [Retro, Borgeaud et al., 2022](https://arxiv.org/abs/2112.04426)

   - Modifying knowledge in language models：

     [Fast Model Editing at Scale, Mitchell et al., 2021](https://arxiv.org/abs/2110.11309)

   - More knowledge-aware pretraining for language models：

     [KEPLER, Wang et al., TACL 2020](https://arxiv.org/pdf/1911.06136.pdf)

   - More efficient knowledge systems：

     [NeurIPS Efficient QA challenge](https://efficientqa.github.io/assets/report.pdf)

   - Better knowledge benchmarks：

     [KILT, Petroni et al., NAACL 2021](https://arxiv.org/abs/2009.02252)

### suggested readings

1. 将预训练好的实体嵌入添加到单词嵌入中。（[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf)）

   模型结构（ERNIE：Enhanced Language Representation with Informative Entities）：

   - 文本编码器：多层双向Transformer编码器，用于编码文本语句；

   - 知识编码器：块级结构，每一块由两部分构成：

     ① 两个自注意力层：一个用于是实体嵌入，一个用于单词嵌入；

     ② 一个**融合层**（fusion layer）用于合并两个自注意力层地输出；

     具体而言：
     $$
     \begin{aligned}
     h_j&=\sigma(\tilde W_t^{(i)}\tilde w_{j}^{(i)}+\tilde W_e^{(i)}+\tilde e_k^{(i)}+\tilde b^{(i)})&&\text{fusion representation}\\
     w_j^{(i)}&=\sigma(W_t^{(i)}h_j+b_t^{(i)})&&\text{token embedding output(fed to next block)}\\
     e_k^{(i)}&=\sigma(W_e^{(i)}h_j+b_e^{(i)})&&\text{entity embedding output(fed to next block)}\\
     \end{aligned}
     $$
     ![ERNIE](https://img-blog.csdnimg.cn/bcf608113da144d199116e6915e7e189.png#pic_center)

   训练方式是联合预训练三个任务：**masked language model**，**next sentence prediction**（前两个都是BERT地与训练任务），**Knowledge pretrain task（dEA）**，最后这个任务是随机掩盖掉一些t实体单词的对齐标注，然后预测实体链接，即应当与序列中的哪个单词对应，具体而言：
   $$
   p(e_j|w_i)=\frac{\exp(Ww_i\cdot e_j)}{\sum_{k=1}^m\exp(Ww_i\cdot e_k)}
   $$
   这个预训练的动机是更好地学习单词实体的对齐信息，以免因为给定的实体链接信息而产生过拟合现象。

   最终的目标函数：
   $$
   \mathcal{L}_{\rm ERNIE}=\mathcal{L}_{\rm MLM}+\mathcal{L}_{\rm NSP}+\mathcal{L}_{\rm dEA}
   $$
   这样训练得到的语言模型在下游任务上取得了显著的提升。

   最后谈一下ERNIE的优势与劣势：

   - 成功结合了实体与文本（通过融合层以及知识预训练任务）；
   - 提升了知识增强的下游任务评估结果；
   - 需要提前将输入文本中的实体词与知识库进行链接，这很费时，而且并不容易。

2. 第二种方法，使用外部知识库建模语言模型。（[Barack’s Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling](https://arxiv.org/pdf/1906.07241.pdf)）

3. 第三种方法，改变训练数据，将知识间接融合到非结构化文本中（[Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model](https://arxiv.org/pdf/1912.09637.pdf)）

4. 将语言模型作为知识库。（[Language Models as Knowledge Bases?](https://www.aclweb.org/anthology/D19-1250.pdf)）

   这是关于评估语言模型中的知识的一篇paper，思想是解决到底有多少关系型的知识已经被包含在目前的语言模型中。测试方法就是通过加MASK来让语言模型预测挖掉的MASK处是什么词。

   项目在[https://github.com/facebookresearch/LAMA](https://github.com/facebookresearch/LAMA)，但是LAMA仍然有局限性，它很难解释为什么语言模型会表现得好（比如语言模型只是记住了单词的共现关系，而非真正的知识）。

   于是后来有人做了改进[E-BERT](https://www.aclweb.org/anthology/2020.findings-emnlp.71.pdf)，去掉了那些不需要关系型知识就能预测的MASK。

   还有的[研究](https://www.aclweb.org/anthology/2020.tacl-1.28.pdf)，着重于解决LM可能知道知识，但是无法完成LAMA的检测问题，原因是查询方式不对（句子格式不对），比如已知奥巴马的出生地是夏威夷，但是查询是奥巴马出生于何处，这可能就宕机了。解决方案是生成更多的LAMA提示（prompts），通过从维基百科中挖掘模板，然后使用回译（back-translation）的方式来生成提示。

### project milestone

[[Instructions](http://web.stanford.edu/class/cs224n/project/CS224N_Final_Project_Milestone_Instructions.pdf)]

----

## lecture 14 偏见，毒性与公正

本讲中国人一般不会考虑这些问题。略过，本讲没有slides内容，只有三篇推荐阅读。

### suggested readings

1. （[The Risk of Racial Bias in Hate Speech Detection](https://homes.cs.washington.edu/~msap/pdfs/sap2019risk.pdf)）
2. （[Social Bias Frames](https://homes.cs.washington.edu/~msap/social-bias-frames/)）
3. （[PowerTransformer: Unsupervised Controllable Revision for Biased Language Correction](https://arxiv.org/abs/2010.13816)）

----

## lecture 15 检索增强模型+知识

本讲内容与**lecture13**有重合，都是讲如何讲知识融入模型。

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture15-guu.pdf)]

1. **知识编辑**：slides p.7（推荐阅读第一篇）

   这是一个很新的研究领域，不是很搞得明白，感觉很像是在试图解释神经网络，需要仔细看几篇paper才能彻底搞明白。

   p.8-26论述Transformer的feed-forward层相当于是键值对字典（[Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)）:
   $$
   y=W_2\sigma(W_1x)
   $$
   这里说的问题是说如果某些知识被改变了（例子是埃菲尔铁塔的位置从巴黎改为罗马，研究这个知识的改变，神经网络到底是怎么学会适应这种改变的）。

   三个结论：

   ① Transformer的feedforward网络可以视为键值存储（即知识都存储在前馈网络模块中）；

   ② Transformer倾向于在实体词的最后一个单词上查找实体信息；

   ③ 这是一个新的研究领域，结论可能很快就会变化；

2. **内存增强模型**（momory-augment model）：slides p. 38

   内存可以是：网络文档、数据库条目、训练样本、实体嵌入。

   应用：开放领域对话系统/问答系统，代码生成、图像生成、事实复查；

   关键的三个问题：选取什么样的内存？如何检索内存？如果使用检索到的内存？

   这里举的论文是推荐阅读第二篇（LaMDA，开放领域的对话机器人）：

   - 基本模型仅仅使用了Transformer的解码器，会造成很多错误，因此解决方案是教会LaMDA使用搜索引擎来验证并修正它的回答。

   - 从流程上来看LaMDA是做了多次系统交互进行的实现，下面是一个例子：

     > 模型输入：
     >
     > - User: Base, When wae the Eiffel Tower built?
     > - Base: Research, It was constructed in 1887.
     > - Research: Search engine, Eiffel Tower construction data
     > - Search Engine: Research, 15 essential things to know about the Eiffel Tower. The Eiffel Twoer was built in record time: 2 years, 2 months and 5 days. From late January 1887 to March 31, 1889...
     >
     > 模型输出：
     >
     > - Research: User, Work started on it in January 1887 and it was opened in March 1889.

   结论：

   ① 许多外部检索工具可以接受文本作为输入，并输出检索得到的文本；

   ② 因此使用外部工具可以归结为（boils down to）：1）学习生成文本查询（输入到外部工具）；2）学习理解外部工具的输出

   ③ 这两个任务都可以使用标准Transformer模型来处理；

3. **训练一个神经检索器**：slides p.67

   具体的研究：[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

   即给定一个查询（谁是指环王中最坏的人），然后从维基百科中检索一篇文章，阅读并给出答案。

   **没有训练数据怎么办**：[ORQA](https://arxiv.org/abs/1906.00300)

   本质上检索器就是一个评分函数：
   $$
   f(\text{input},\text{memory})\rightarrow \text{score}
   $$
   可以通过随机采样来近似下面这个分布：
   $$
   p(\text{memory}|\text{input})=\frac{\exp f(\text{input,memory})}{\sum_i\exp f(\text{input})}
   $$
   一旦检索得到一个memory，我们来看看它是否有帮助，方法是计算**Reader's probability of generating right answer**：
   $$
   p(\text{gold answer}|\text{input,memory})
   $$
   若结果较高，则增加这个memory的检索评分，否则降低。

   那么如果我们随机采样一个memory，然后生成一个答案，那么这个答案正确的概率有多少？
   $$
   \sum_{\rm memory}p(\text{memory|input})p(\text{gold_answer|input,memory})
   $$
   上式求和项第一项是检索器的概率，第二项是阅读器成功的概率，乘起来就是每一个memory的尝试。一些memory会成功，另外一些则会失败，QRQA的思想就是使用梯度下降来最大化这个求和式（取对数），最终$p(\text{memory|input})$会自然而然的收敛到很好的memory上。

   **一种用于处理不可数的*查询—回答*样本对的方法**：slides p.83（推荐阅读第三篇）

   结论一览：

   ① 检索器就是一个函数，参数为input和memory，计算得到评分score

   ② 有监督学习：对于每个输入，提供正memory和负memory，然后使得检索器能够正确区分。

   ③ 无法进行有监督学习，则使用end-to-end学习。

   ④ 试错法，若memory有助于提升模型，则给予它更高的评分。

   ⑤ 在试错法中，经常会创建无穷多的数据（使用MASK）

4. **如何使用memory**：slides p.89

   推荐阅读：[Entity-Based Knowledge Conflicts in Question Answering](https://arxiv.org/abs/2109.05052)，检索这块跟问答是分不开的，我最近在做的CAIL2021也是差不多。

   如何合理使用内存是困难的：永远记住不要太过于依赖外部的内存，也不能让模型完全忽视掉外部内存的检索结果。

### suggested readings

这三篇都是近两年非常新的研究，slides部分都有提及。

1. 知识编辑。（[Locating and Editing Factual Knowledge in GPT](https://arxiv.org/abs/2202.05262)）
2. 一种内存增强模型。（[LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239)）
3. 处理无穷多的查询回答样本对（[REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)）

----

## lecture 16 卷积网络，树递归神经网络，成分分析

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture16-CNN-TreeRNN.pdf)]

1. RNN的一个问题是最后的输出会包含太多序列中最后一个单词的信息（因为信息会随着递归而递减）。

2. CNN可以用于文本分类，因为文本的块级特征对于分类来说是比较有意义的。而且CNN是可以并行运算的（kernel窗口的遍历过程是可以并行）。

   CNN的基础知识不记录，老生常谈。

3. 第二部分提到的就是句法树（这个跟依存分析里的依存关系是有区别的，是目前我做的工作，具体关于句法树生成可以回去看**lecture6**的内容），句法树的启发源自于语言是具有递归结构的（主句到从句等等）。

   句法树的生成涉及到成分分析。这部分的研究跟依存分析差不多，都很老，但是如果要用到这些技术，肯定还是要综述这些研究的。

   <font color=red>**然后可以根据句法树的形态构造，类比构建出相适的神经网络结构来对输入文本进行处理！**这个想法很重要（虽然跟原文意思有点背离，但我觉得这似乎可以是一个创新），我觉得的确可以如此，句法树处理可以使用图神经网络的方法处理得到表示，也可以用于指导模型的构建，但是这样的话模型就不是唯一的了，但是我依然可以想办法使用一些门控设置来使得模型具有可变性。这个我要好好思考一下！</font>

   slides p.44-47有一个贪心生成句法树的办法，方法是从底部进行递归，研究哪两个单词可以合并，取概率最高的进行合并后，接着将剩下的输入到下一次递归中，因为句法树的确具有这样的特征，就是前序遍历不会改变单词顺序，所以我可以直接从底部两个两个进行尝试，这样生成的一定是二叉树，但是句法树是可以有多叉的，不过这也不关键，因为树都可以转化为二叉树。解决这一问题的模型是TreeRNN。

   我看到网上也有人做过这一讲的笔记（[http://www.hankcs.com/nlp/cs224n-tree-recursive-neural-networks-and-constituency-parsing.html](http://www.hankcs.com/nlp/cs224n-tree-recursive-neural-networks-and-constituency-parsing.html)），兼听则明。

4. TreeRNN，TreeLSTM，Recursive Neural Tensor Networks

   这里提到了多篇paper，大部分都是Manning写的，建议可以到[arxiv@manning](https://dblp.uni-trier.de/pid/m/ChristopherDManning.html)搜搜看Manning的论文，用关键词Tree检索，可以找到很多有用的东西。

### suggested readings

1. CNN用于序列分类。（[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882.pdf)）
2. （[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)）
3. CNN用于建模语句。（[A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf)）
4. 句法解析树。（[Parsing with Compositional Vector Grammars.](http://www.aclweb.org/anthology/P13-1045)）
5. 句法解析中的成分分析。（[Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/pdf/1805.01052.pdf)）

----

## lecture 17 大模型的正则化法则

### suggested readings

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

这篇值得后面花时间好好看一下。

----

## lecture 18 共指关系

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture18-coref.pdf)]

1. 共指关系（Coreference Resolution）研究就是本体识别研究，即同一实体的不同实例。识别句中的实体是相对简单的，但是要对识别出来的实体词（mention）进行分类是很难的。

2. Mention：指文本中指代某个实体的一块单词

   大致可以分为三类：

   ① 代词：I, your, it, she, him（使用词性tagger发现）

   ② 名称实体：人名，地名（使用命名实体识别系统发现）

   ③ 名词性短语（使用句法解析器发现）

   实体识别（Mention Detection）也并不是那么容易，比如下面的例子中的粗体是Mention吗？

   ① **It** is sunny

   ② **The best donut in the world**

   ③ **100 miles**

   那么如何处理这些bad mentions？可以训练模型，但更多的时候，我们都保留它们作为候选的Mention，总之并非每一个代词都意有所指（或者说确切地指向某个具体的实体）

3. **Anaphora v.s. Coreference**：前者是句法上的复指，后者是词义上的共指。

   ![在这里插入图片描述](https://img-blog.csdnimg.cn/1b112bc3f6654d6ba8f649cdbe3fc49c.png#pic_center)

4. Coreference模型：

   - 基于规则的模型（人称代词消解，pronominal anaphora resolution）

     1976年Hobb提出的简易算法：

     ```
     1. Begin at the NP immediately dominating the pronoun
     2. Go up tree to first NP or S. Call this X, and the path p.
     3. Traverse all branches below X to the left of p, left-to-right, breadth-first. Propose as
     antecedent any NP that has a NP or S between it and X
     4. If X is the highest S in the sentence, traverse the parse trees of the previous sentences
     in the order of recency. Traverse each tree left-to-right, breadth first. When an NP is
     encountered, propose as antecedent. If X not the highest node, go to step 5.
     5. From node X, go up the tree to the first NP or S. Call it X, and the path p.
     6. If X is an NP and the path p to X came from a non-head phrase of X (a specifier or adjunct,
     such as a possessive, PP, apposition, or relative clause), propose X as antecedent
     (The original said “did not pass through the N’ that X immediately dominates”, but
     the Penn Treebank grammar lacks N’ nodes….)
     7. Traverse all branches below X to the left of the path, in a left-to-right, breadth first
     manner. Propose any NP encountered as the antecedent
     8. If X is an S node, traverse all branches of X to the right of the path but do not go
     below any NP or S encountered. Propose any NP as the antecedent.
     9. Go to step 4
     ```

     ![PAS](https://img-blog.csdnimg.cn/d42ae65434d84132a5642a20d8f70729.png#pic_center)

     后来开始有基于知识的人称共指关系消解。

     [www.cs.toronto.edu/~hector/Papers/ijcai-13-paper.pdf](www.cs.toronto.edu/~hector/Papers/ijcai-13-paper.pdf)

     [http://commonsensereasoning.org/winograd.html](http://commonsensereasoning.org/winograd.html)

   - Mention Pari/Mention Ranking

     即配对句中识别出来的Mention，训练一个聚合器来对Mention进行聚合。

     或者更简单一些，直接训练一个二分类器，判断两两Mention是否存在共指：
     $$
     J=-\sum_{i=1}^N\sum_{j=1}^iy_{ij}\log p(m_j,m_i)
     $$
     然后对配对结果进行重排序。

   - End-to-end neural coreference：训练神经网络模型，见推荐阅读第二篇。

5. 提供两个共指关系的demo：

   [http://corenlp.run/](http://corenlp.run/)

   [https://huggingface.co/coref/](https://huggingface.co/coref/)

### suggested readings

1. 教材章节。（[Coreference Resolution Chapter from Jurafsky and Martin](https://web.stanford.edu/~jurafsky/slp3/21.pdf)）
2. （[End-to-end Neural Coreference Resolution](https://arxiv.org/pdf/1707.07045.pdf)）

----

## lecture 19 编辑神经网络

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture-editing.pdf)] 

这一讲是教你怎么调整神经网络（复杂化，简单化，调整网络层的顺序，模型架构，数据发生变化，如何调整模型以适应变化后的数据），分析神经网络模型训练的内存占用，主要是slides太长了，而且内容比较零散，个人觉得实践应该更重要一些，遇到问题再来反查吧。

有点类似**lecture15**的内容。

