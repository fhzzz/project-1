## 1109周日
- main_cdac.py
    - 这个基本和CDAC原论文第一阶段是一样的，模型直接返回损失，不过增加了早停机制以及保存评估结果最好的模型
    - 和 cdac.py 最大的区别在于模型的设置不同
    - 随机采样器，加载预训练模型，但是首轮效果跟没加载一样，不知道为什么。strict=True/False结果差别不大
    - 应该是没加载成功预训练权重。
    - 另外，评估指标的计算方法有一点小改正

    - 模型保存在 results/clinc/best_cdac_model.pt, cdac_train_1109.log
    - bsz = 32
        - 最初结果：ACC = 22.0  ARI = 8.31  NMI = 56.6  Epoch = 0
        - 最后结果：ACC = 48.0  ARI = 39.69  NMI = 79.38  Epoch = 43
    - bsz = 64
        - 最初结果：ACC = 22.71  ARI = 9.14  NMI = 57.42  Epoch = 0
        - 最后结果：ACC = 52.4  ARI = 43.29  NMI = 82.2  Epoch = 37

- cdac.py
    - 这个是在Bert类里面融合CDAC原论文的计算损失方法，但是这里仅仅作为特征提取器，并没有分类层，得到特征向量之后使用k-means得到聚类结果。
    - 一开始加载了预训练权重，训练结果和之前一样

    - 模型保存在 results/clinc/best_model.pt, cdac_1109.log

    - clinc, KCL=0.25, train_batch_size=32(模型直接返回损失值，需要注意，这里的损失函数还包括有标签样本)
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 58.8  ARI = 45.49  NMI = 80.36  Epoch = 34

    - clinc, KCL=0.25, train_batch_size=32(模型外部实现，配合样本对，损失函数有u-l这一项)
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 57.56  ARI = 44.09  NMI = 79.59  Epoch = 39
    - clinc, KCL=0.25, train_batch_size=32(模型外部实现，配合样本对，损失函数没有u-l这一项)
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 57.56  ARI = 44.09  NMI = 79.59  Epoch = 39
(cdac_1110.log)  
    - clinc, KCL=0.25, train_batch_size=32(模型外部实现，配合样本对，损失函数没有u-l这一项, eta = epoch * 0.01 )
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 57.11  ARI = 43.42  NMI = 79.27  Epoch = 26  

    - clinc, KCL=0.25, train_batch_size=32(模型外部实现，配合样本对，损失函数没有u-l这一项, eta = epoch * 0.005 )
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 58.31  ARI = 44.13  NMI = 79.86  Epoch = 46
        - **思考**：如果eta变化比较缓慢，模型更慢收敛，虽然学得慢，但是学得久很好的弥补了这一点；阈值更新最好有所改变，因为好像后续能选到不确定样本对的样本很少很少；损失函数加上u-l这一项可能会好一点，我要试试
    - clinc, KCL=0.25, train_batch_size=32(模型外部实现，配合样本对，损失函数有u-l这一项, eta = epoch * 0.005 )
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 56.27  ARI = 42.55  NMI = 78.96  Epoch = 36     

ACC = 47.01  ARI = 35.22  NMI = 67.67  Epoch = 1

## 1110周一
- **昨天实验发现的问题**
    - 预训练效果太好，这大概就是为什么之前训练指标一直下降的原因
        - 更换数据集
        - 直接使用训练好的权重

    - 损失函数表达式 u-l 的作用，需要考虑三元组损失的大小决定权重值
        - 从结果上看，有没有这一项结果几乎一样。
    - 区间阈值的更新策略

- **待办**
    - 更换数据集，需要进行预处理来使用transformers, datasets 库
    - 思考解决办法，预训练很好，要怎么利用？如果使用一般的对比损失函数呢？
    - 跑 ALUP / MTP-CLNN / LANID(如果有) / DeepAligned / 

## 1111周二
- 跑了mtp-clnn，但是结果很差，其实有点停滞了，不知道要怎么办了

## 1112周三

## 1116周日
### cdac_2.py
- clinc/eta=0.009/lr=1e-5
    - ACC = 81.64  ARI = 73.34  NMI = 91.78  Epoch = 4
- clinc/eta=0.005/bsz=32/lr=1e-5
    - ACC = 81.6  ARI = 73.74  NMI = 91.95  Epoch = 4
- clinc/eta=0.005/bsz=32/lr=5e-6
    - ACC = 81.24  ARI = 73.31  NMI = 91.91  Epoch = 0

- clinc/eta=0.001/bsz=32/lr=1e-5
    - ACC = 81.91  ARI = 73.4  NMI = 91.91  Epoch = 1

- clinc/eta=0.01/bsz=32/lr=1e-5
    - ACC = 79.02  ARI = 70.48  NMI = 90.95  Epoch = 0
    - ACC = 81.42  ARI = 72.69  NMI = 91.57  Epoch = 1

ACC = 68.18 | ARI = 57.33 | NMI = 86.84

