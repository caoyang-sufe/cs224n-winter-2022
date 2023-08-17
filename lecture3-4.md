# CS224N WINTER 2022（二）反向传播、神经网络、依存分析（附Assignment2答案）

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

  - 由于CSDN限制博客字数，笔者无法将完整内容发表于一篇博客内，只能分篇发布，可从我的[GitHub Repository](https://github.com/umask000/cs224n-winter-2022)中获取完整笔记，<font color=red>**本系列其他博客发布于**</font>（Updating）：

    [CS224N WINTER 2022（一）词向量（附Assignment1答案）](https://caoyang.blog.csdn.net/article/details/125020572)

    [CS224N WINTER 2022（二）反向传播、神经网络、依存分析（附Assignment2答案）](https://blog.csdn.net/CY19980216/article/details/125022559)
    
    [CS224N WINTER 2022（三）RNN、语言模型、梯度消失与梯度爆炸（附Assignment3答案）](https://blog.csdn.net/CY19980216/article/details/125031727)
    
    [CS224N WINTER 2022（四）机器翻译、注意力机制、subword模型（附Assignment4答案）](https://blog.csdn.net/CY19980216/article/details/125055794)
    
    [CS224N WINTER 2022（五）Transformers详解（附Assignment5答案）](https://blog.csdn.net/CY19980216/article/details/125072701)

----

[toc]

----
## lecture 3 反向传播与神经网络

本节内容属于基础的机器学习与深度学习知识。

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture03-neuralnets.pdf)]

1. **矩阵（向量）链式求导中的计算技巧**：slides p.39

   假定一个神经网络的运算步骤如下所示：
   $$
   x(\text{input})\rightarrow z=Wx+b\rightarrow h=f(z)\rightarrow s=u^\top h\tag{3.1}
   $$
   则计算网络参数$W$与$b$的参数时：
   $$
   \frac{\partial s}{\partial W}=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial W}\quad\frac{\partial s}{\partial b}=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial b}\tag{3.2}
   $$
   可以定义**局部误差信号**（local error signal）：
   $$
   \delta=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}=u^\top\circ f'(z)\tag{3.3}
   $$
   则可以使得计算$(3.2)$式的更加简单，事实上进一步计算可以发现：
   $$
   \begin{aligned}
   \frac{\partial s}{\partial W}&=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial W}=\delta\frac{\partial z}{\partial W}=\delta^\top x^\top\\
   \frac{\partial s}{\partial b}&=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial b}=\delta\frac{\partial z}{\partial b}=\delta
   \end{aligned}\tag{3.4}
   $$
   此时我们称$x$是**局部输入信号**（local input signal）。

   这就是反向传播高效的原因，事实上只需要在神经网络的每条传播路径上存储两端节点变量的偏导值（如神经网络中节点$z$指向节点$h$，则存储$\partial h/\partial z$），即可快速计算任意两个节点变量之间的偏导值。

2. **广义计算图中的反向传播**（General Computation Graph）：slides p.77
   $$
   \frac{\partial z}{\partial x}=\sum_{i=1}^n\frac{\partial z}{\partial y_i}\frac{\partial y_i}{\partial x}\tag{3.5}
   $$
   其中$\{y_1,y_2,...,y_n\}$是$x$指向的所有节点。

### notes

 [[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf)]

1. **神经网络中的常用技巧**：notes p.8-18

   - **梯度检验**：

     自动梯度计算一般使用的是数值近似的方法：
     $$
     f'(x)\approx \frac{f(x+\epsilon)-f(x)}{\epsilon}\tag{3.6}
     $$
     而非解析方法，因此总是存在一定的误差（特殊情况下可能就是错误的）。

     梯度检验是一种用来比较数值梯度值与解析梯度值之间差距的方法。下面是一个简单的demo：

     ```python
     def eval_numerical_gradient(f, x):
     	"""
     	a naive implementation of numerical gradient of f at x
     	- f should be a function that takes a single argument
     	- x is the point (numpy array) to evaluate the gradient
     	at
     	"""
     	fx = f(x) # evaluate function value at original point
     	grad = np.zeros(x.shape)
     	h = 0.00001
     	
     	# iterate over all indexes in x
     	it = np.nditer(x, flags=['multi_index'], op_flags = ['readwrite'])
     	
     	while not it.finished:
     		# evaluate function at x+h
     		ix = it.multi_index
     		old_value = x[ix]
     		x[ix] = old_value + h                       # increment by h
     		fxh_left = f(x)                             # evaluate f(x + h)
     		x[ix] = old_value - h                       # decrement by h
     		fxh_right = f(x)                            # evaluate f(x - h)
     		x[ix] = old_value                           # restore to previous value (very important!)
     		
     		# compute the partial derivative
     		grad[ix] = (fxh_left - fxh_right) / (2 * h) # the slope
     		it.iternext()                               # step to next dimension
     		
     	return grad
     ```

   - **正则化**：常用于处理模型过拟合的问题，一般是在损失函数中引入模型参数矩阵的$F$范数（相当于是所有参数的二范数值）。

   - **Dropout**：常用于处理模型过拟合与降低模型复杂度。在反向传播中以一定概率剪除神经网络的传播路径。

   - **激活函数**：Sigmoid，Tanh，Hard tanh，Soft sign，ReLU，Leaky ReLU

   - **数据预处理**：减均值（中心化），正则化，白化（去除特征之间的相关性，如通过奇异值分解）。

   - **参数初始化**：这个其实还是有讲究的，参数初始值的确对模型优化可能产生显著影响，有兴趣可以扒扒PyTorch中的参数初始化源码，是有很多不同的初始化方式的。如果我记得没错的话，PyTorch中绝大多数层的参数初始化用的都是本节中提到的这种方式：
     $$
     W\sim\text{Uniform}\left(-\sqrt{\frac{6}{n^{(l)}+n^{(l+1)}}},\sqrt{\frac{6}{n^{(l)}+n^{(l+1)}}}\right)\tag{3.7}
     $$
     其中$n^{(l)}$是参数矩阵$W$的**fan-in**值（输入节点数），$n^{(l+1)}$是参数矩阵$W$的**fan-out**值（输出节点数）。

     截距项一般初始化都是零。

   - **学习策略**：学习率可以进行动态调整，有兴趣可以查看PyTorch中与lr_scheduler相关的内容。

   - **动量更新**：梯度下降法一个变体。

     ```python
     # Computes a standard momentum update
     # on parameters x
     v = mu*v - alpha*grad_x
     x += v
     ```

   - **自适应优化方法**（Adaptive Optimization Methods）：如经典的AdaGrad，Adam应该都属于这类方法。RMSProp是AdaGrad的一种变体。

     ```python
     # Assume the gradient dx and parameter vector x
     cache += dx**2
     x += - learning_rate * dx / np.sqrt(cache + 1e-8)
     
     # Update rule for RMS prop
     cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
     x += - learning_rate * dx / (np.sqrt(cache) + eps)
     
     # Update rule for Adam
     m = beta1 * m + (1 - beta1) * dx
     v = beta2 * v + (1 - beta2) * (dx ** 2)
     x += - learning_rate * m / (np.sqrt(v) + eps)
     ```

### suggested readings

1. 矩阵运算相关（主要是求导）在神经网络中的应用。（[matrix calculus notes](http://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf)）

2. 各种多元函数、矩阵函数的求导方法汇总。（[Review of differential calculus](http://web.stanford.edu/class/cs224n/readings/review-differential-calculus.pdf)）

3. 神经网络架构的基础知识，与slides及notes的内容有一定重合。（[CS231n notes on network architectures](http://cs231n.github.io/neural-networks-1/)）

4. 反向传播重点讲解，包含一些代码示例。（[CS231n notes on backprop](http://cs231n.github.io/optimization-2/)）

5. 从求导开始阐述反向传播的数学推导。（[Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf)）

6. 1986年的一篇关于反向传播的老古董paper，<font color=red>反向传播首次提出？</font>。（[Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)）

### additional readings

1. 一篇发布于2016年的反向传播博客。（[Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)）

2. 本节唯一一篇属于paper的阅读内容，感觉更像是一部长篇教材，既讲了神经网络的基础知识，也讲了神经网络在若干自然语言处理任务中的应用，感觉写得很乱。不过也是2011年的内容，可能比较过时。（[Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)）

### assignment2 参考答案

[[code](http://web.stanford.edu/class/cs224n/assignments/a2.zip)] [[handout](http://web.stanford.edu/class/cs224n/assignments/a2.pdf)] [[latex template](http://web.stanford.edu/class/cs224n/assignments/a2_latex_template.zip)]

<font color=red>assignment2参考答案（written+coding）：囚生CYの[GitHub Repository](https://github.com/umask000/cs224n-winter-2022/tree/main/cs224n-winter2022-solutions/assignment2)</font>

#### 1. written

- $(a)$ 根据定义可知：
  $$
  y_w=\left\{\begin{aligned}
  &1&&\text{if }w=o\\
  &0&&\text{otherwise}
  \end{aligned}\right.\Rightarrow
  \text{LHS}=-\sum_{w\in\text{Vocab}}y_w\log(\hat y_w)=-y_o\log(\hat y_o)=-\log(\hat y_o)=\text{RHS}
  \tag{a2.1.1}
  $$

- $(b)$ 本问可参考[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture01-wordvecs1.pdf)]中$\text{29-31}$页的推导过程：
  $$
  \begin{aligned}
  \frac{\partial J_{\text{naive-softmax}}(v_c,o,U)}{\partial v_c}&=-\frac{\partial}{\partial v_c}\log P(O=o|C=c)\\
  &=-\frac{\partial}{\partial v_c}\log\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}\\
  &=-\frac{\partial}{\partial v_c}\log\exp(u_o^\top v_c)+\frac{\partial}{\partial v_c}\log\left(\sum_{w\in V}\exp(u_w^\top v_c)\right)\\
  &=-\frac{\partial}{\partial v_c}u_o^\top v_c+\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\frac{\partial}{\partial v_c}\sum_{x\in V}\exp(u_x^\top v_c)\\
  &=-u_o+\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\sum_{x\in V}\frac{\partial}{\partial v_c}\exp(u_x^\top v_c)\\
  &=-u_o+\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\sum_{x\in V}\exp(u_x^\top v_c)\frac{\partial}{\partial v_c}u_x^\top v_c\\
  &=-u_o+\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\sum_{x\in V}\exp(u_x^\top v_c)u_x\\
  &=-u_o+\sum_{x\in V}\frac{\exp(u_x^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}u_x\\
  &=-u_o+\sum_{x\in V}P(O=x|C=c)u_x\\
  &=-u_o+\sum_{x\in V}\hat y_xu_x\\
  &=-U^\top y+U^\top \hat y\\
  &=U^\top(\hat y-y)
  \end{aligned}\tag{a2.1.2}
  $$
  式$(1.2)$沿用[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)]中的标记，即约定$U\in\R^{|V|\times n},y\in\R^{|V|},\hat y\in\R^{|V|}$，具体如下：
  $$
  \begin{aligned}
  U&=\left[\begin{matrix}u_0&u_1&...&u_{|V|}\end{matrix}\right]\\
  y&=\left[\begin{matrix}0&0&...&1&...&0\end{matrix}\right]^\top\\
  \hat y&=\left[\begin{matrix}\hat y_0&\hat y_1&...&\hat y_o&...&\hat y_{|V|}\end{matrix}\right]^\top
  \end{aligned}\tag{a2.1.3}
  $$

  - $(1)$ 当$\hat y=y$时，梯度为零；
  - $(2)$ 我理解这个问题可能是想说梯度值实际刻画的是观测值与真实值之间误差，因此减去这个误差可以地得到更可靠的$v_c$

- $(c)$ 类似$(b)$中的推导，我们有：
  $$
  \begin{aligned}
  \frac{\partial J_{\text{naive-softmax}}(v_c,o,U)}{\partial u_w}&=-\frac{\partial}{\partial u_w}\log P(O=o|C=c)\\
  &=-\frac{\partial}{\partial u_w}\log\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}\\
  &=-\frac{\partial}{\partial u_w}\log\exp(u_o^\top v_c)+\frac{\partial}{\partial u_w}\log\left(\sum_{w\in V}\exp(u_w^\top v_c)\right)\\
  &=-\frac{\partial}{\partial u_w}u_o^\top v_c+\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\cdot\frac{\partial}{\partial u_w}\sum_{x\in V}\exp(u_x^\top v_c)\\
  &=-\frac{\partial}{\partial u_w}u_o^\top v_c+\frac{1}{\sum_{w\in V}\exp(u_w^\top v_c)}\frac{\partial}{\partial u_w}\exp(u_w^\top v_c)\\
  &=-\frac{\partial}{\partial u_w}u_o^\top v_c+\frac{\exp(u_w^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}v_c\\
  &=-\frac{\partial}{\partial u_w}u_o^\top v_c+P(O=w|C=c)v_c\\
  &=-\frac{\partial}{\partial u_w}u_o^\top v_c+\hat y_wv_c\\
  &=\left\{\begin{aligned}
  &(\hat y_w-1)v_c&&\text{if }w=o\\
  &\hat y_wv_c&&\text{otherwise}
  \end{aligned}\right.\\
  &=(\hat y_w-y_w)v_c
  \end{aligned}\tag{a2.1.4}
  $$

- $(d)$ 根据$(c)$中的结果，可得：
  $$
  \frac{\partial J_{\text{naive-softmax}}(v_c,o,U)}{\partial U}=(\hat y-y)^\top v_c\tag{a2.1.5}
  $$

- $(e)$ 将$\text{ReLU}$激活函数写作分段形式分别求导：
  $$
  f(x)=\left\{\begin{aligned}
  &0&&\text{if }x\lt 0\\
  &x&&\text{if }x\gt 0
  \end{aligned}\right.\Rightarrow 
  f'(x)=\left\{\begin{aligned}
  &0&&\text{if }x\lt 0\\
  &1&&\text{if }x\gt 0
  \end{aligned}\right.\tag{a2.1.6}
  $$

- $(f)$ 有如下推导：
  $$
  \sigma'(x)=\frac{e^x(e^x+1)-e^{2x}}{(e^x+1)^2}=\frac{e^x}{(e^x+1)^2}=\sigma(x)(1-\sigma(x))\tag{a2.1.7}
  $$

- $(g)$ 关于负采样的内容可参考[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)]与[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture02-wordvecs2.pdf)]的相关章节。

  - $(1)$ 关于$v_c$的偏导有如下推导：
    $$
    \begin{aligned}
    \frac{\partial J_{\text{neg-sample}}(v_c,o,U)}{\partial v_c}&=-\frac{\partial}{\partial v_c}\log\sigma(u_o^\top v_c)-\sum_{s=1}^K\frac{\partial}{\partial v_c}\log \sigma(-u_{w_s}^\top v_c)\\
    &=-\frac{\sigma(u_o^\top v_c)(1-\sigma(u_o^\top v_c))}{\sigma(u_o^\top v_c)}\cdot \frac{\partial u_o^\top v_c}{\partial v_c}-\sum_{s=1}^{K}\frac{\sigma(-u_{w_s}^\top v_c)(1-\sigma(-u_{w_s}^\top v_c))}{\sigma(-u_{w_s}^\top v_c)}\frac{\partial-u_{w_s}^\top v_c}{\partial v_c}\\
    &=(\sigma(u_o^\top v_c)-1)u_o+\sum_{s=1}^K(1-\sigma(-u_{w_s}^\top v_c))u_{w_s}
    \end{aligned}
    \tag{a2.1.8}
    $$
    关于$u_o$的偏导有如下推导：
    $$
    \begin{aligned}
    \frac{\partial J_{\text{neg-sample}}(v_c,o,U)}{\partial u_o}&=-\frac{\partial}{\partial u_o}\log\sigma(u_o^\top v_c)-\sum_{s=1}^K\frac{\partial}{\partial u_o}\log \sigma(-u_{w_s}^\top v_c)\\
    &=-\frac{\sigma(u_o^\top v_c)(1-\sigma(u_o^\top v_c))}{\sigma(u_o^\top v_c)}\cdot \frac{\partial u_o^\top v_c}{\partial u_o}\\
    &=(\sigma(u_o^\top v_c)-1)v_c
    \end{aligned}
    \tag{a2.1.9}
    $$
    关于$u_{w_s}$的偏导有如下推导：
    $$
    \begin{aligned}
    \frac{\partial J_{\text{neg-sample}}(v_c,o,U)}{\partial u_{w_s}}&=-\frac{\partial}{\partial u_{w_s}}\log\sigma(u_o^\top v_c)-\sum_{k=1}^K\frac{\partial}{\partial u_{w_s}}\log \sigma(-u_{w_k}^\top v_c)\\
    &=-\sum_{k=1}^{K}\frac{\sigma(-u_{w_k}^\top v_c)(1-\sigma(-u_{w_k}^\top v_c))}{\sigma(-u_{w_k}^\top v_c)}\frac{\partial-u_{w_k}^\top v_c}{\partial u_{w_s}}\\
    &=(1-\sigma(-u_{w_s}^\top v_c))v_c
    \end{aligned}
    \tag{a2.1.10}
    $$

  - $(2)$ 观察$(a2.1.8)(a2.1.9)(a2.1.10)$三个偏导解析式，显然可以重用的部分是：
    $$
    \sigma(u_o^\top v/_c)-1\text{ and }1-\sigma(-u_{w_s}^\top v_c),s=1,2,...,K\tag{a2.1.11}
    $$
    写成要求的矩阵形式即为：
    $$
    \sigma(U_{o,\{w_1,...,w_K\}}^\top v_c)-\textbf{1}\tag{a2.1.12}
    $$

  - $(3)$ 从$(b)(c)$的结果来看，同样有可以重用的部分（$\hat y-y$），但是$(b)$的梯度需要计算矩阵与向量的乘法，耗时较长，而$(g)$中的三个结果使用重用部分后本质上是标量与向量相乘的运算，自然要高效很多。

- $(h)$ 本问与$(g)$的区别在于，负采样可能采样到重复的单词，因此结果与式$(a2.1.10)$稍有区别：
  $$
  \begin{aligned}
  \frac{\partial J_{\text{neg-sample}}(v_c,o,U)}{\partial u_{w_s}}&=-\frac{\partial}{\partial u_{w_s}}\log\sigma(u_o^\top v_c)-\sum_{k=1}^K\frac{\partial}{\partial u_{w_s}}\log \sigma(-u_{w_k}^\top v_c)\\
  &=-\sum_{k=1}^{K}\frac{\sigma(-u_{w_k}^\top v_c)(1-\sigma(-u_{w_k}^\top v_c))}{\sigma(-u_{w_k}^\top v_c)}\frac{\partial-u_{w_k}^\top v_c}{\partial u_{w_s}}\\
  &=\sum_{k=1}^K\textbf{1}_{w_k=w_s}(1-\sigma(-u_{w_s}^\top v_c))v_c
  \end{aligned}
  \tag{a2.1.13}
  $$
  其中$\textbf{1}$为指示函数，若$w_k=w_s$取值为一，否则取值为零。

- $(i)$ 在[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)]中可以查阅$\text{skip-gram}$模型的目标函数：
  $$
  \text{minimize}\quad J=-\sum_{j=0,j\neq m}^{2m}u_{c-m+j}^\top v_c+2m\log\sum_{k=1}^{|V|}\exp(u_k^\top v_c)\tag{a2.1.14}
  $$
  则要求的三个偏导具有如下形式：
  $$
  \begin{aligned}
  &\frac{\partial J_{\text{skip-gram}}(v_c,w_{t-m},...,w_{t+m},U)}{\partial U}&&=\sum_{-m\le j\le m,j\neq0}\frac{\partial J(v_c,w_{t+j},U)}{\partial U}\\
  &\frac{\partial J_{\text{skip-gram}}(v_c,w_{t-m},...,w_{t+m},U)}{\partial v_c}&&=\sum_{-m\le j\le m,j\neq0}\frac{\partial J(v_c,w_{t+j},U)}{\partial v_c}\\
  &\frac{\partial J_{\text{skip-gram}}(v_c,w_{t-m},...,w_{t+m},U)}{\partial v_w}&&=0
  \end{aligned}\tag{a2.1.15}
  $$

#### 2. coding

- $(a)$ 这里第$(4)$小问可能有点问题，始终无法通过测试，但是在往年的代码里是可以通过测试的，感觉或许是测试代码写得有些问题，因为只有$\text{skip-gram}$的负采样损失函数这一项无法通过测试，其他都是正确的。

  - $(1)$ 参考$\text{written}$部分的$(f)$
  - $(2)$ 参考$\text{written}$部分的$(b)(c)(d)$
  - $(3)$ 参考$\text{written}$部分的$(g)$
  - $(4)$ 参考$\text{written}$部分的$(i)$

- $(b)$ 简单的梯度下降法实现。

- $(c)$ 需要事先下载[数据集](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)并解压到$\text{utils}$目录下，运行得到的图片为：

  ![3.1](https://img-blog.csdnimg.cn/992453f7025b4fb08645d673b8246633.png)

----

## lecture 4 依存分析

### slides

[[slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture04-dep-parsing.pdf)] [[slides (annotated)](http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture04-dep-parsing-annotated.pdf)] 两份slides完全相同，后一份带标注（因为需要标注依存关系），本文以后一份为准。

> - 依存分析一定程度上跟句法分析是类似的，旨在挖掘自然语言中词语之间的依赖关系，如词性标注、句法识别、句法树解析等都属于依存分析的范畴，用以实现精准语义识别等任务，推荐一篇[博客](https://www.cnblogs.com/CheeseZH/p/5768389.html)把各种依存关系讲得比较清楚。
> - 笔者近期的在做与中文句法树相关的工作，推荐一个[斯坦福句法解析包](https://nlp.stanford.edu/software/lex-parser.html)，这里面包含了包括中文在内的多种语言的句法解析包（有Python的接口，但是需要安装JDK才能使用），个人感觉要比jieba的功能更齐全一些，至少jieba里目前还没有生成句法树的接口。关于该解析包的用法网上可以搜索到一些教程，笔者列出几个参考过的教程：
>   - [中文词性标注解释及句法分析标注解释_weixin_30642561的博客-CSDN博客](https://blog.csdn.net/weixin_30642561/article/details/97772970?spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-10.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-10.pc_relevant_antiscanv2&utm_relevant_index=14)
>   - [Stanford Parser中文句法分析器的使用_海涛anywn的博客-CSDN博客_句法分析器](https://blog.csdn.net/lihaitao000/article/details/51556923)
>   - [使用Stanford Parser进行句法分析 - Denise_hzf - 博客园](https://www.cnblogs.com/Denise-hzf/p/6612574.html)
>   - [Stanford依存句法关系解释 - 微冷不觉寒 - 博客园](https://www.cnblogs.com/weilen/p/8284411.html)
> - 本节涉及许多关于句法树以及依存关系的内容。

1. **依存分析的概念**：slides p.24

   解析语句，将其中每个单词指向其所依赖的对象（通常句首添加ROOT标签，同样要为ROOT标注一个对象）。

   句法依存标注的基本规则：

   - ROOT有且仅有一个依赖对象；
   - 不要出现环，即$A$指向$B$，$B$又以任何路径指回到$A$，原因是依存分析的目的之一是可以构建句法解析树。
   - 具体的一些句法依存关系与语义依存关系可以查看[博客](https://www.cnblogs.com/CheeseZH/p/5768389.html)

   <font color=red>这里有提到一个projectivity的概念，不是很能搞得明白，有一个结论是CFG树必然是projective的。（在斯坦福句法解析包中，大部分的解析器都是PCFG解析器）</font>

2. **依存分析的方法**：slides p.26

   - **动态规划**：这是一个很老的思想，最早可以追溯到1986年。
   - **图算法**：图算法的思想有点类似最大生成树，分词之间边的权重其实就是分词相互依存的概率，可以通过统计方法决定。
   - **基于规则的硬性约束**
   - **基于转移的解析**（transition-based parser）与**基于确定性的解析**（deterministic dependency parsing），这两个方法在推荐阅读部分有几篇相关的论文可供参考。

   <font color=red>一般来说，依存分析中，动词是句子的核心，会直接连在ROOT下面，然后扩展到全句其他词。</font>

3. **笔者拓展**：

   这一节内容相对晦涩一些，笔者认为可能实践更为重要，因此记录一下近期斯坦福句法解析库的以备后用：

   1. 安装JDK；

   2. 从[stanford-parser-4.2.0-models.jar](https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip)处下载得到stanford-parser-4.2.0.zip（截至本文发布的版本号，之后可能会更新，可自行到[https://nlp.stanford.edu/software/lex-parser.html#Download](https://nlp.stanford.edu/software/lex-parser.html#Download)寻找下载链接）

   3. 接下来我们需要从stanford-parser-4.2.0.zip中找到几个文件：

      ① 根目录下的stanford-parser.jar；

      ② 根目录下的stanford-parser-4.2.0-models.jar；

      ③ 以中文解析为例，从stanford-parser-4.2.0-models.jar中解压得到chinesePCFG.ser.gz；

    4. 下面是生成句法树的demo：

       ```python
       # -*- coding: utf-8 -*-
       # @author: caoyang
       # @email: caoyang@163.sufe.edu.cn
       
       from nltk.parse.stanford import StanfordParser
       
       def generate_parse_tree(tokens):		
       	parser = StanfordParser('stanford-parser.jar',
       							'stanford-parser-4.2.0-models.jar',
       							model_path='chinesePCFG.ser.gz')
       	parse_tree = list(parser.parse(tokens))
       	return parse_tree
       
       parse_tree = generate_parse_tree(tokens=['今天', '是', '一个', '好', '天气'])
       print(parse_tree)	# [Tree('ROOT', [Tree('IP', [Tree('NP', [Tree('NT', ['今天'])]), Tree('VP', [Tree('VC', ['是']), Tree('NP', [Tree('QP', [Tree('CD', ['一个'])]), Tree('ADJP', [Tree('JJ', ['好'])]), Tree('NP', [Tree('NN', ['天气'])])])])])])]
       ```

   5. 注意道中文句法解析的输入必须是已经做好分词的结果，虽然斯坦福句法解析库也可以进行中文分词，但是相对来说jieba要更容易，也做得更好，因此可以考虑先用jieba分好词再输入进行解析得到句法树。

   6. 句法树可以进行可视化，使用stanford-parser-4.2.0.zip根目录下的lexparser-gui.bat即可。

   7. 句法树的叶子节点记录了每一个分词的词性，经过检查，笔者共发现有33种不同的词性：

      ```python
      # Stanford中文词性标注集(对应句法树叶子节点上的标注, 共33个), 具体说明见https://blog.csdn.net/weixin_30642561/article/details/97772970
      STANFORD_POS_TAG = [
      	'AD', 'AS', 'BA', 'CC', 'CD', 'CS', 'DEC', 'DEG', 'DER', 'DEV', 'DT', 
      	'ETC', 'FW', 'IJ', 'JJ', 'LB', 'LC', 'M', 'MSP', 'NN', 'NR', 'NT', 
      	'OD', 'P', 'PN', 'PU', 'SB', 'SP', 'URL', 'VA', 'VC', 'VE', 'VV',
      ]
      ```

      此外非叶子节点标注的是句法成分，经过检查，笔者共发现28种不同的句法成分标记（与词性不重复）：

      ```python
      # Stanford中文句法依存分析标注集(共33+28个, 前33个来自STANFORD_POS_TAG, 后28个对应句法树非叶节点上的标注)
      STANFORD_SYNTACTIC_TAG = STANFORD_POS_TAG + [
      	'ADJP', 'ADVP', 'CLP', 'CP', 'DFL', 'DNP', 'DP', 'DVP', 'FLR', 
      	'FRAG', 'INC', 'INTJ', 'IP', 'LCP', 'LST', 'NP', 'PP', 'PRN', 'QP', 
      	'ROOT', 'UCP', 'VCD', 'VCP', 'VNV', 'VP', 'VPT', 'VRD', 'VSB',
      ]
      ```

   8. 可直接从句法树中提取词性标注结果：

      ```python
      # -*- coding: utf-8 -*-
      # @author: caoyang
      # @email: caoyang@163.sufe.edu.cn
      
      import re
      
      def generate_pos_tags_from_parse_tree(parse_tree, regex_compiler = re.compile('\([^\(\)]+\)', re.I)):
      	if not isinstance(parse_tree, str):
      		# 2022/03/09 19:48:50 只对字符串形式的树进行处理, 原数据结构不便于处理
      		parse_tree = str(parse_tree)
      	
      	results = regex_compiler.findall(parse_tree)
      	pos_tags = list(map(lambda result: tuple(result[1: -1].split(' ', 1)), results))
      	return pos_tags
      ```

   9. 注意分词中的左右小括号会被改写为`-LRB`与`-RRB-`，以与句法树中的括号区分开，此外分词中`\xa0`会被视为空字符串，分词中的`\u3000`与单空格字符将不会被斯坦福句法解析库识别到，这些都是可能造成程序出现问题的小细节：

      ```python
      # 2022/03/29 20:57:45 解析树会把原分词序列中的一些特殊符号分词改写为其他形式
      # 2022/03/29 20:57:54 典型的如-LRB-表示左小括号, -RRB-表示右小括号
      # 2022/03/29 20:57:54 其他的情况暂时没有发现, 但是我估计应该是没有其他了, 这两个是为了避免和句法树形式中的括号混淆
      STANFORD_SPECIAL_SYMBOL_MAPPING = {'(': '-LRB-', ')': '-RRB-', '\xa0': ''}
      STANFORD_IGNORED_SYMBOL = {'\u3000', ' '}
      ```

   10. 可以将解析得到的句法树转为networkx或dgl的图，下面是笔者实现的方法：

       ```python
       # -*- coding: utf-8 -*-
       # @author: caoyang
       # @email: caoyang@163.sufe.edu.cn
       
       import networkx
       import dgl
       
       from networkx import DiGraph, draw
       
       def parse_tree_to_graph(parse_tree, display=False, return_type='networkx', ignore_text=False):
       	assert return_type in ['networkx', 'dgl'], f'Unknown param `return_type`: {return_type}'
       	if not isinstance(parse_tree, str):
       		# 2022/03/09 19:48:50 只对字符串形式的树进行处理, 原数据结构不便于处理
       		warnings.warn(f'句法解析树应为字符串, 而非{type(parse_tree)}')
       		parse_tree = str(parse_tree)
       	
       	graph = DiGraph()		# 2022/03/09 20:48:52 绘制句法树的有向图
       	current_index = 0		# 2022/03/09 20:48:52 记录已经解析到句法树字符串的位置
       	stack = []				# 2022/03/09 20:48:52 用于存储句法树节点的栈(包括句法树标签与文本)	
       	node_id = -1			# 2022/04/23 16:09:19 全局记录节点的id, 考虑将节点信息存储在node.data中
       	
       	# 2022/04/28 23:29:29 合并后的添加节点函数
       	def _add_node(_node_id, _text, _tag, _is_pos, _is_text):
       		_node_data = {
       			'node_id'	: _node_id,										# 2022/05/20 12:48:02 节点编号: 0, 1, 2, ...
       			'text'		: _text,										# 2022/05/20 12:48:02 文本分词内容, 若_is_text为False, 则为None
       			'tag_id'	: STANFORD_SYNTACTIC_TAG_INDEX.get(_tag, -1),	# 2022/05/20 12:48:02 句法树标签内容, 若_is_text为True, 则为None, 否则必然属于STANFORD_SYNTACTIC_TAG集合
       			'is_pos'	: _is_pos,										# 2022/05/20 12:48:02 记录是否是句法树叶子节点上的词性标注
       			'is_text'	: _is_text,										# 2022/05/20 12:48:02 记录是否是句法树叶子节点上的文本分词
       		}
       		graph.add_node(node_for_adding=_node_id, **_node_data)					# 添加新节点
       		if stack:																# 若栈不为空								
       			_stack_top_node_id = stack[-1]['node_id']							# 取栈顶节点的编号				
       			graph.add_edge(u_of_edge=_stack_top_node_id, v_of_edge=_node_id)	# 则栈顶节点(即当前分支节点)指向新节点
       		elif _is_text:
       			raise Exception('叶子节点文本内容找不到父节点标签')
       		if not _is_text:
       			stack.append(_node_data)											# 最后将新节点(非叶/非文本)添加到栈中
       			
       	while current_index < len(parse_tree):
       		# 左括号意味着新分支的开始
       		if parse_tree[current_index] == '(':
       			next_left_parenthese_index = parse_tree.find('(', current_index + 1)	# 寻找下一个左括号的位置
       			next_right_parenthese_index = parse_tree.find(')', current_index + 1)	# 寻找下一个右括号的位置
       			
       			if next_left_parenthese_index == -1 and next_right_parenthese_index == -1:
       				# 左括号后面一定还有括号
       				raise Exception('句法树括号位置不合规')
       			
       			if next_left_parenthese_index < next_right_parenthese_index and next_left_parenthese_index >= 0:
       				# 向右检索最先遇到左括号: 新节点出现
       				new_node = parse_tree[current_index + 1: next_left_parenthese_index].replace(' ', '')	# 向右搜索先遇到左括号: 发现新节点
       				
       				# 2022/05/20 13:00:30 新增断言
       				assert new_node in STANFORD_SYNTACTIC_TAG and new_node not in STANFORD_POS_TAG, f'Unknown syntactic tags: {new_node}'
       				
       				node_id += 1																			# 更新节点编号
       				_add_node(_node_id=node_id, _text=None, _tag=new_node, _is_pos=False, _is_text=False)	# 添加新节点
       				current_index = next_left_parenthese_index												# 将current_index刷新到新的左括号处
       			else:
       				# 向右检索最先遇到右括号: 此时到达叶子节点
       				leave_node = parse_tree[current_index + 1: next_right_parenthese_index]					# 向右搜索先遇到右括号: 此时意味着已经到达叶子节点
       				new_node, text = leave_node.split(' ', 1)												# 叶子节点由词性标注与对应的文本内容两部分构成
       				
       				# 2022/05/20 13:00:30 新增断言
       				assert new_node in STANFORD_POS_TAG, f'Unknown pos tags: {new_node}'
       				
       				node_id += 1																			# 更新节点编号
       				_add_node(_node_id=node_id, _text=None, _tag=new_node, _is_pos=True, _is_text=False)	# 添加叶子节点
       				if not ignore_text:
       					node_id += 1																		# 更新节点编号
       					_add_node(_node_id=node_id, _text=text, _tag=None, _is_pos=False, _is_text=True)	# 添加叶子节点上的文本内容
       				current_index = next_right_parenthese_index + 1											# 将current_index刷新到右括号的下一个位置
       				stack.pop(-1)																			# 弹出栈顶节点, 即叶子节点
       		
       		# 右括号意味着新分支的结束
       		elif parse_tree[current_index] == ')':	
       			stack.pop(-1)
       			current_index += 1
       		
       		# 空格则跳过
       		elif parse_tree[current_index] == ' ':
       			current_index += 1
       		
       		# 理论上不会出现其他情况, 除非字符串根本就不是一棵合法的句法树
       		else:
       			raise Exception(f'不合法的字符: {parse_tree[current_index]}')
       	
       	if display:
       		# 使用matplotlib将图绘制出来
       		draw(graph, with_labels=True)
       		plt.show()
       	
       	if return_type == 'networkx':
       		return graph
       	
       	elif return_type == 'dgl':
       		return dgl.from_networkx(graph, node_attrs=['node_id', 'tag_id',  'is_pos', 'is_text'])
       
       ```

### notes

[[notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes04-dependencyparsing.pdf)]

1. 有两种主要类型的句法树结构：**成分结构**（constituency structures）与**依赖结构**（dependency structure）

2. 突然想到句法树的生成有点类似证明图的生成。

3. **基于Transition的依存分析**：notes p.2-3

   > 基于转换的依赖关系解析依赖于**状态机**，状态机定义所有可能的转移，用以创建从输入句子到依存关系树的映射。学习问题是根据状态机的过渡历史，建立一个能够预测下一个过渡的模型。句法分析问题是在给定先前归纳模型的情况下，为输入句子构造最佳的Transition序列。大多数基于Transition的系统不使用正式的语法。 

   这里举了一个叫作**Greedy Deterministic Transition-Based Parsing**的例子（2003提出），Transition系统是一个状态机：

   - 状态：对于给定的句子$S=w_0w_1...w_n$，状态定义为三元组$c=(\sigma,\beta,A)$

     ① $\sigma$是存储$S$中单词$w_i$的栈（stack）；

     ② $\beta$是存储$A$中单词$w_i$的缓存（buffer）；

     ③ $A$是一系列$(w_i,r,w_j)$（称之为dependency arc）构成的集合，其中$r$是预先定义好的一种依存关系。

   - 初始状态$c_0=([w_0]_\sigma,[w_1,...,w_n]_{\beta},\emptyset)$，其中$w_0$就是ROOT，其他所有单词都在缓存中。

   - 终末状态为$(\sigma,[]_\beta,A)$

   - 然后每次转移的方式有三种：

     ① Shift：将缓存中的第一个单词移除，并置入栈顶；
     $$
     \sigma,w_i|\beta,A\rightarrow \sigma|w_i,\beta,A\tag{4.1}
     $$
     ② Left-Arc：添加一个依存关系组$(w_j,r,w_i)$进入$A$，其中$w_i$是栈顶的下一个单词，$w_j$是栈顶单词，然后从栈中移除$w_i$；（先决条件：栈中至少包含两个单词，且$w_i$不能是ROOT）
     $$
     \sigma|w_i|w_j,\beta,A\rightarrow\sigma|w_j,\beta,A\cup\{r(w_j,w_i)\}\tag{4.2}
     $$
     ③ Right-Arc：添加一个依存关系组$(w_i,r,w_j)$进入$A$，其中$w_i$是栈顶的下一个单词，$w_j$是栈顶单词，然后从栈中移除$w_j$；（先决条件：栈中至少包含两个单词）
     $$
     \sigma|w_i|w_j,\beta,A\rightarrow\sigma|w_j,\beta,A\cup\{r(w_i,w_j)\}\tag{4.3}
     $$

4. **神经依存分析**：notes p.3-5

   这个是非常流行的做法，推荐阅读部分也有两篇较新的paper是关于该主题的。

   以**Greedy Deterministic Transition-Based Parsing**为例，神经网络模型每次预测下一次的转移是哪一个（三分类），这个就有点强化学习的味道了，也就不难理解slides部分提到可以使用动态规划进行依存分析，不过似乎如何定义奖励是比较困难的事情。

   输入神经网络的特征来自单词的嵌入，各种句法标注标签的嵌入（词性，句法成分等）。

### suggested readings

1. 2004年的一篇老古董，讲的是Deterministic dependency parsing（与之对应的基于transition的解析方法）的内容，（[Incrementality in Deterministic Dependency Parsing](https://www.aclweb.org/anthology/W/W04/W04-0308.pdf)）
2. 使用神经网络方法进行依存分析，使用的是基于transition的增量解析方法，2014年发表于EMNLP。（[A Fast and Accurate Dependency Parser using Neural Networks](https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf)）
3. 这是一部著作，但是应该是需要付费下载的，暂时无法获取资源。（[Dependency Parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002)）
4. 这也是一篇用神经网络方法进行依存分析，使用的也是基于transition的增量解析方法，2016年上传于ARXIV（[Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf)）
5. 这篇应该[Stanford Parser](https://nlp.stanford.edu/software/lex-parser.html)创始的paper，讲的是多语言句法解析的方法（[Universal Stanford Dependencies: A cross-linguistic typology](http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf)）
6. 这是上一篇paper对应的网站，这是一个众筹数据的平台，好像也可以下载treebank的数据。（[Universal Dependencies website](http://universaldependencies.org/)）
7. 教材中关于依存分析（[Jurafsky & Martin Chapter 14](https://web.stanford.edu/~jurafsky/slp3/14.pdf)）

此外在[Stanford Parser](https://nlp.stanford.edu/software/lex-parser.html)主页上，每一种语言的解析器都有多篇论文可以参考。

### pytorch tutorial session

[[colab notebook](https://colab.research.google.com/drive/13HGy3-uIIy1KD_WFhG4nVrxJC-3nUUkP?usp=sharing)]

笔者暂时无法下载colab代码，不过应该是一份PyTorch的入门demo，不是那么重要，自己查查API文档基本也够用了。