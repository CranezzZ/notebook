# transformer相关

## 一、self-attention源码

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



## 二、面经题目

### 1.自注意力计算公式

<img src="E:\code\note\notebook\photos\about transformer\image-20220427174819958.png" alt="image-20220427174819958" style="zoom: 67%;" />

### 2.多头的作用

原论文中作者的想法是将模型分为多个头，期望其能形成多个相互独立子空间的，随后让模型关注不同方面的信息。

另一方面在计算复杂度中也有体现，苏剑林指出：

<img src="E:\code\note\notebook\photos\about transformer\image-20220427181420899.png" alt="image-20220427181420899" style="zoom: 50%;" />

NIPS2019的论文《Are Sixteen Heads Really Better than One?》指出其实很多的注意力头是冗余的，因此可以剪枝。



### 3.有几种注意力

加性注意力、双线性注意力、乘性注意力和缩放点积注意力。缩放点积注意力有深度学习框架底层计算方式优化，速度快。

<img src="E:\code\note\notebook\photos\about transformer\image-20220427184730782.png" alt="image-20220427184730782" style="zoom: 80%;" />

### 4.为什么线性变换、除以根号dk？

- 防止输入softmax的结果过大，梯度消失
- 假设向量q和k的各个分量是相互独立的随机变量，且均值为0，方差为1；那么点积q*k的期望为0，方差为dk，因此进行归一化，使得q*k满足期望为0方差为1的分布

### 5.BERT 模型参数量计算（Base版本）

- embedding：(30522 + 512 + 2) * 768 --vocabulary, pos_embedding, token_type_embedding 一共1个

  - 总参数量为 23,835,648

- multi_head：768 * 768 * 3 + 768 * 768 --Wq Wk Wv以及线性变化的Wo（分头分开算和一起算一样吗？ 待确定）一共12个

  - 总参数量为28,331,552

- FFN：768 * 3072 + 3072 * 768 一共12个

  - 总参数量为56,623,104

- LayerNorm：768 + 768 -- gamma, beta  分别位于词向量后面， attention之后，最后一个全连接层之后 一共25个

  - 总参数量38,400

  这是不含下游任务分类头参数的数目

### 6.LayerNorm实现

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

### 7.词表模型BPE、WordPiece、Unigram Language Model

- BPE：先全部拆分成单个字符，然后添加频率最高的相邻子词对，直到达到词表大小
- WordPiece：先全部拆分成单个字符，然后添加具有最大互信息值的两个子词，直到达到词表大小
- ULM：从大到小反过来的



### 8.提升速度

- 蒸馏、剪枝
- 双塔模型存储静态向量
- int8或fp16
- 提前结束，在前几层就已经大概率知道分类结果的时候提前结束 https://arxiv.org/pdf/2006.04152.pdf
- albert 不同层之间共享参数、矩阵分解

### 9.不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵会有什么问题？

大部分注意力值集中到自身token，无法有效利用上下文信息。

