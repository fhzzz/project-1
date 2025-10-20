# 代码

## 关于训练细节

- 相似度损失值需要按照batch计算，三元组损失同样，但是找样本对肯定需要全局，此时只需要前向传播，应该可行。

- 调用LLM是每隔几轮进行，所以三元组数据集也是每隔几轮更新，所以可以采用MemoryBank的方式先缓存更新轮的特征向量。这里的疑问在于，尽管三元组数据集每隔几轮更新，但由于使用了相似度损失，所以模型参数也一直在更新，所以这里是否使用MemoryBank还需要仔细考虑

- 顺序采样器和随机采样器的问题，这个可能不是特别重要

- 

## 文件结构及内容
- data
    - clinc
    - banking

- model.py
    - class pretrainBert
    - class clBert

- data_process.py
    - class PrepareData

- pretrain_manager.py
    - CE
    - MLM

- cl_manager.py
    - def 

- config.py
    - 管理各种参数与训练设置

- utils.py
    - 各种有用的函数

- readme
    - 对项目的必要说明

### data_process.py
- 设定Known Class Ratio后**随机选择**：known_label_list, unk_label_list, all_ordered
- 按照标签**分层抽样**得到：train_labeled_samples, train_semi_samples
    - def label2id(): 按照all_ordered的顺序映射，而all_ordered又按照先有标签后无标签的顺序，根据这个前提，可以通过列表长度得到标签样本和无标签样本
    - 需要分层抽样，所以
- train_labeled_samples
- train_semi_samples

- train_labeled_dataloader
    - 是否需要顺序采样？

- train_semi_dataloader
    - 顺序采样？

### main.py
- def eval
    - input: train_semi_dataloader, model
    - return: feats

- def get_sim_score
    - 按公式计算就行
    - input: feats, [samples, emb_dim]
    - return: sim_score, [samples, samples]

- def get_label_R
    - use dataset: train_labeled_samples
    - label_R: [num_labeled_samples, num_labeled_samples]

- def get_unlabel_R
    - use dataset: train_unlabel_samples
    - unlabel_R: [num_unlabel_samples, num_unlabel_samples]
    - 得分区间上下限，按照数据分布(def get_score_info)变化
    - 大于上限：取1
    - 小于下限：取0
    - 区间中部：取值-1
    
- def get_score_info
    - 得到相似度分数的数据分布，以此更新区间上下限[l(\lambda), u(\lambda)]
    - para: sim_score
    - return 均值或者方差？

- def get_uncertainty_Q
    - 根据上下限返回
    - para: 
    - return: uncertainty_pair [pair_num, pair_num]


- def llm_labelling
    - input: uncertainty_pair
    - return: 0/1, 置信度得分
    - 根据样本对索引获取样本文本

- def update_unlabel_R
    - 更新取值为-1的部分
    - 可以计算损失，这里的损失是semi还是只unlabel？
    - 

- def train
    - 完成主训练逻辑
    - 保存最佳模型

- def 

### trip_loss.py
- 

### 动机和出发点
- 使用LLM的强大能力挖掘更多监督信息
- 主动学习的思想，根据模型的判断，找出最有价值的样本对，高效利用LLM
- 同时，一方面LLM只回答是或不是能降低伪标签的负面影响；另一方面输出置信度得分能充分利用LLM的能力，
- 考虑三元组对比损失，再次利用LLM之前的结果，挖掘难样本例子，三元组损失使用相似度度量，并使用置信度加权，避免训练不稳定
- 联合训练，一共有三个损失函数，标签监督，无标签自监督，三元组损失

### 文字思路

1. 得到feats, 计算 sim_score, 同时得到数据分布：均值和方差
2. 根据数据分布得到区间上下界
3. 根据上下界得到矩阵R, 
    1) 标签数据, 根据sim_score和R就可以算出来一个损失
    2) 无标签数据, -1, 0, 1 -> 得到 uncertain_pair
4. 根据LLM得到伪标签
    - 得到样本对索引 [(i, j), ...]
    - 根据索引得到文本对 [(text_i, text_j), ...]
    - 调用LLM，写好提示词，让它输出结果，
    - 需要在这个方法里把前面的变量都真正得到。
    - 
    - 

5. 根据伪标签能天然得到正样本对


### 缓存区
```
# 1. get feats and labels
# train_semi_dataloader是顺序采样器
feats, _ = self.eval(args, dataloader=train_semi_dataloader, get_feats=True)

_, labels = self.eval(args, dataloader=train_labeled_samples, get_feats=True)

# 2. compute sim_score
sim_score = self.get_sim_score(feats)

# 3. get_R
label_R = self.get_label_R(labels=labels)
unlabel_R = self.get_unlabel_R(sim_score=sim_score, l, u)

# 4. get_uncert_pairs


```

### uncert_pairs
```
uncertain_ij = []
for i in range(bsz):
    for j in range()
```

### AI 
- 我改进了附件PDF论文，其余附件包含了我的改进思路(a-pairwise-idea.md)、代码思路(code-idea.ma)、数据处理脚本(data_process.py)、主训练脚本(main.py,未完成)。现在我已经能得到text_pairs，但是我不知道应该如何调用LLM完成我的改进想法，我也没有调用LLM的经验，我希望你能在我代码的基础上实施完善。请你帮我写完llm_labeling方法，你可以给出prompt模板。注意，你需要仔细检查你的代码逻辑。

- 我编写的R矩阵这些我不知道应该按照batch实现，还是按照全部样本实现？如果是batch内的R矩阵，那么需要考虑索引的问题，应该使用全局索引。因为需要从R矩阵得到uncert_pair,需要对其使用LLM

