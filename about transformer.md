# 1 transformer相关

## 1.1 self-attention源码

```python
class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('error')
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # shape of x: batch_size * seq_length * hidden_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # shape of hidden_states: batch_size * seq_length * hidden_size
        
        # shape of mixed_*_layer: batch_size * seq_length * hidden_size
        mixed_query_layer = self.query(hidden_states) #Wq
        mixed_key_layer = self.key(hidden_states) # Wk
        mixed_value_layer = self.value(hidden_states) # Wv

        # shape of query_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # shape of attention_scores: batch_size * num_attention_heads * seq_length * seq_length
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # shape of attention_scores: batch_size * num_attention_heads * seq_length * seq_length
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # shape of value_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        # shape of first context_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        # shape of second context_layer: batch_size * seq_length * num_attention_heads * attention_head_size
        # context_layer 维度恢复到：batch_size * seq_length * hidden_size
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
```



## 1.2 面经题目

### 1.2.1 自注意力计算公式

<img src="E:\code\note\notebook\photos\about transformer\image-20220427174819958.png" alt="image-20220427174819958" style="zoom: 67%;" />

### 1.2.2 多头的作用

原论文中作者的想法是将模型分为多个头，期望其能形成多个相互独立子空间的，随后让模型关注不同方面的信息。

另一方面在计算复杂度中也有体现，苏剑林指出：

<img src="E:\code\note\notebook\photos\about transformer\image-20220427181420899.png" alt="image-20220427181420899" style="zoom: 50%;" />

NIPS2019的论文《Are Sixteen Heads Really Better than One?》指出其实很多的注意力头是冗余的，因此可以剪枝。



### 1.2.3 有几种注意力

加性注意力、双线性注意力、乘性注意力和缩放点积注意力。缩放点积注意力有深度学习框架底层计算方式优化，速度快。

<img src="E:\code\note\notebook\photos\about transformer\image-20220427184730782.png" alt="image-20220427184730782" style="zoom: 80%;" />

### 1.2.4 为什么线性变换、除以根号dk？

- 防止输入softmax的结果过大，梯度消失
- 假设向量q和k的各个分量是相互独立的随机变量，且均值为0，方差为1；那么点积q*k的期望为0，方差为dk，因此进行归一化，使得q*k满足期望为0方差为1的分布

### 1.2.5 BERT 模型参数量计算（Base版本）

- embedding：(30522 + 512 + 2) * 768 --vocabulary, pos_embedding, token_type_embedding 一共1个

  - 总参数量为 23,835,648

- multi_head：768 * 768 * 3 + 768 * 768 --Wq Wk Wv以及线性变化的Wo（分头分开算和一起算一样吗？ 待确定）一共12个

  - 总参数量为28,331,552

- FFN：768 * 3072 + 3072 * 768 一共12个

  - 总参数量为56,623,104

- LayerNorm：768 + 768 -- gamma, beta  分别位于词向量后面， attention之后，最后一个全连接层之后 一共25个

  - 总参数量38,400

  这是不含下游任务分类头参数的数目

### 1.2.6 LayerNorm实现

```python
class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon        # 一个很小的常数，防止除0

    def forward(self, x):
        u = x.mean(-1, keepdim=True)                    # LN是对最后一个维度做Norm，正则化后过一个线性投影，可训练参数体现在这个线性投影上
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
```

<img src="E:\code\note\notebook\photos\about transformer\image-20220428151427213.png" alt="image-20220428151427213" style="zoom: 50%;" />

<img src="E:\code\note\notebook\photos\about transformer\image-20220428151452491.png" alt="image-20220428151452491" style="zoom:50%;" />

### 1.2.7 词表模型BPE、WordPiece、Unigram Language Model

- BPE：先全部拆分成单个字符，然后添加频率最高的相邻子词对，直到达到词表大小
- WordPiece：先全部拆分成单个字符，然后添加具有最大互信息值的两个子词，直到达到词表大小
- ULM：从大到小反过来的



### 1.2.8 提升速度

- 蒸馏、剪枝
- 双塔模型存储静态向量
- int8或fp16
- 提前结束，在前几层就已经大概率知道分类结果的时候提前结束 https://arxiv.org/pdf/2006.04152.pdf
- albert 不同层之间共享参数、矩阵分解

### 1.2.9 不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵会有什么问题？

大部分注意力值集中到自身token，无法有效利用上下文信息。

### 1.2.10 从Alignment 和 Uniformity的角度理解对比表征学习



[《Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere》]: https://arxiv.org/pdf/2005.10242.pdf



文章指出了对比表示学习的两个重要属性：

- alignment：two samples forming a positive pair should be mapped to nearby features, and thus be (mostly) invariant to unneeded noise factors.
- Uniformity: feature vectors should be roughly uniformly distributed on the unit hypersphere, pre-serving as much information of the data as possible.

<img src="E:\code\note\notebook\photos\about transformer\image-20220502222432620.png" alt="image-20220502222432620" style="zoom:67%;" />

# 2 optimizer相关

## 2.1 优化器总结

$J_\theta$: 损失函数

### 2.1.1Gradient Descent 梯度下降

$ \theta_{t+1} = \theta_{t} - \bigtriangledown_{t} = \theta_{t} - \eta \cdot\bigtriangledown_{\theta_{t}}J_{i}(\theta_t,x_i,y_i)$

针对于每一个训练样本都进行一次梯度下降，不分组。

### 2.1.2 Batch Gradient Descent 批量梯度下降

$ \theta_{t+1} = \theta_{t} - \bigtriangledown_{t} = \theta_{t} - \eta \cdot\frac{1}{n}\sum_{i=1}^{n+1}\bigtriangledown_{\theta_{t}}J_{i}(\theta_t,x_i,y_i)$

优点：

- 当损失函数是凸函数（convex）时，BGD能收敛到全局最优；当损失函数非凸时，BGD能收敛到局部最优；

缺点：

- 每次都要根据全部的数据来计算梯度，梯度下降速度慢；
- BGD不能在线训练，也不能根据新数据来实时更新模型；

### 2.1.3 Stochastic Gradient Descent 随机梯度下降

$ \theta_{t+1} = \theta_{t} - \bigtriangledown_{t} = \theta_{t} - \eta \cdot\bigtriangledown_{\theta_{t}}J_{i}(\theta_t,x_i,y_i)$动量 -- i 随机选取

优点：

- 缓慢降低学习率，几乎一定会收敛到局部或非局部最小值；
- 快；
- 可以根据新样本实施地更新模型；

缺点：

- 随机选择引入噪声，更新方向不一定正确，体现在曲线上就是loss的震荡会比较严重；
- 无法克服局部最优的问题；

### 2.1.4 Mini Batch Gradient Descent

$ \theta_{t+1} = \theta_{t} - \bigtriangledown_{t} = \theta_{t} - \eta \cdot\frac{1}{m}\sum_{i=x}^{x+m-1}\bigtriangledown_{\theta_{t}}J_{i}(\theta_t,x_i,y_i)$

m：批量的大小

优点：

- 收敛更加稳定；
- 可以利用高度优化的矩阵库来加速计算过程；

缺点：

- 选择合适的学习率比较困难；
- 容易被困在马鞍面的鞍点；

### 2.1.5 Momentum  动量

参数更新时一定程度上保留更新之前的更新方向

新一轮动量：$m_{t+1} = \beta m_t + (1-\beta)\bigtriangledown_{\theta_{t}}J_{i}(\theta_t,x_i,y_i)$

参数更新：$\theta_{t+1}=\theta_t-\alpha m_{t+1}$

相比于SGD减小了震荡

### 2.1.6 Nesterov Accelerated Gradient 涅斯捷罗夫梯度

相比动量方法，改为了在上一步动量方向更进一点的位置计算梯度并更新参数

新一轮动量：$m_{t+1} = \beta m_t + (1-\beta)\bigtriangledown_{\theta_{t}}J_{i}(\theta_t - \beta \cdot m_t)$

参数更新：$\theta_{t+1}=\theta_t-\alpha m_{t+1}$

### 2.1.7 Adagrad 自适应梯度

在初期学习率一般比较大，因为这时的位置离最优点比较远；当训练快结束时，通常会降低学习率，因为快结束时离最优点比较近，这时使用大的学习率可能会跳过最优点。Adagrad 能使得参数的学习率在训练的过程中越来越小。

令$g _t = \bigtriangledown _{\theta _ t}J(\theta_t)$ 代表t时间时的梯度

$ \theta _ {t+1} = \theta_t - \frac{\eta}{\sqrt{\sum_{i = 1}^{t}{g_ t}^2 + \varepsilon}}g_t$

优点：

- 自动调节参数的学习率

缺点：

- 仍需手动设置学习率$\eta$
- 学习率下降较快，因为分母一直在累计，可能提前结束了学习

### 2.1.8 Adadelta

Adadelta对于Adagrad做出了修改，比Adagrad更加稳定

$ \theta _ {t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}}g_t$

$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_{t}^2$

### 2.1.9 RMSprop Root Mean Squre propogation 均方根反向传播

是Adadelta 中 beta = 0.5 的特例

$ \theta _ {t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}}g_t$

$E[g^2]_t = 0.5 * E[g^2]_{t-1} + 0.5 * g_{t}^2$

### 2.1.10 Adam Adaptive Moment Estimation

Momentum + RMSprop

初始化：

$m_{t+1} = \beta m_t + (1-\beta)g_{t+1}$

$v_{t+1} = \gamma v_t + (1-\gamma)g_{t+1}^2$

偏差修正：

$\hat{m_t} = \frac{m_t}{1-\beta^t}$

$\hat{v_t} = \frac{v_t}{1-\gamma^t}$

最后：

$ \theta _ {t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t} + \varepsilon}}\hat{m_t}$

### 2.1.11 Adamw Adam + wright decate

效果与Adam + L2正则化相同，但是计算效率更高，因为L2正则化需要在loss中加入正则项，之后再计算梯度，但是Adamw直接将正则项的梯度加入反向传播的公式中，省去了在loss中添加正则化的过程。

<img src="E:\code\note\notebook\photos\about transformer\image-20220502145530912.png" alt="image-20220502145530912" style="zoom: 67%;" />

# 3 损失函数相关

## 3.1 损失函数总结

### 3.1.1 0-1损失函数

![image-20220502155925451](E:\code\note\notebook\photos\about transformer\image-20220502155925451.png)
特点：

- 非凸函数
- 感知机使用的是这种损失

### 3.1.2 绝对值损失函数

![image-20220502160111910](E:\code\note\notebook\photos\about transformer\image-20220502160111910.png)

### 3.1.3 log对数损失函数

![image-20220502160126470](E:\code\note\notebook\photos\about transformer\image-20220502160126470.png)

特点：

- 鲁棒性不强，相比于hinge损失对于噪声更加敏感
- 逻辑回归使用

### 3.1.4 平方损失函数

![image-20220502160308582](E:\code\note\notebook\photos\about transformer\image-20220502160308582.png)

特点：

- 经常应用在回归问题中

### 3.1.5 指数损失函数

![image-20220502160434645](E:\code\note\notebook\photos\about transformer\image-20220502160434645.png)

特点：

- 对离群点、噪声非常敏感
- Adaboost中用

### 3.1.6 Hinge损失函数

![image-20220502160527519](E:\code\note\notebook\photos\about transformer\image-20220502160527519.png)

特点：

- SVM使用



### 3.1.7 感知损失函数

![image-20220502162135438](E:\code\note\notebook\photos\about transformer\image-20220502162135438.png)

### 3.1.8 交叉熵损失函数

二分类：

![image-20220502162155545](E:\code\note\notebook\photos\about transformer\image-20220502162155545.png)

多分类：

![image-20220502162215728](E:\code\note\notebook\photos\about transformer\image-20220502162215728.png)

## 3.2 面经题目

### 3.2.1 交叉熵函数与最大似然函数之间的联系和区别

- 区别：交叉熵函数用来描述模型预测值与真实值的差距大小，越大代表越不相近；似然函数的本质是衡量在某个参数下，整体的估计与真实的情况一样的概率，越大代表越相近。
- 联系：交叉熵函数可以由最大似然函数在伯努利分布条件下推导出来，最小化交叉熵函数的本质就是对数似然函数的最大化。

假设一个变量满足伯努利分布：

$P(X=1) = p,P(X=0) = 1- p$

则X的概率密度函数为：

$P(X) = p^X(1-p)^{1-X}$

因为只有一组采样数据D，可以通过统计得到X和1-X的值，但是p的概率是未知的，因此使用极大似然估计来估计p值。

对于采样数据D，对数似然函数为：

$logP(D)=log\prod_{i}^{N}P(D_i)=\sum_{i}^{N}logP(Di)=\sum_{i}^{N}(D_ilogp+(1-D_i)log(1-p))=-loss$

# 4 词向量相关 放弃了 看PDF吧

## 4.1 词向量总结

### 4.1.1 Word2Vec

![image-20220502204806779](E:\code\note\notebook\photos\about transformer\image-20220502204806779.png)

#### 4.1.1.1 Simple CBOW Model 

词表大小 V ，隐藏层维度 W

输入层为维度为 V  的，onehot

经过权重矩阵 $W_{V×W}$ 得到隐藏层向量 h，维度为 W

经过另外一个权重矩阵$W_{W×V}$ 得到输出层向量，维度为 V

目标函数：

![image-20220502211119496](E:\code\note\notebook\photos\about transformer\image-20220502211119496.png)

![image-20220502211126709](E:\code\note\notebook\photos\about transformer\image-20220502211126709.png)

#### 4.1.1.2 CBOW Multi-Word Context Model

和上一个单独CBOW的不同之处在于单个输入变成了多个输入，输入后求平均。

## 4.2 面经题目

### 4.2.1 hierarchical softmax 层次softmax

传统神经网络耗时的地方主要有两个，一个是矩阵运算，一个就是softmax操作。

传统的softmax计算公式如下：

![image-20220502202101299](E:\code\note\notebook\photos\about transformer\image-20220502202101299.png)

求和是一个很耗时的操作，具体实现是一个按照词频排列的哈夫曼树

### 4.1.2 Negative Sampling 负采样

### 4.1.3 onehot向量的问题

- 长度等于词表大小，太大了
- 任意两个词向量之间的余弦相似度为0

