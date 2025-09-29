# 代码

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
    
def get_score_info
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
