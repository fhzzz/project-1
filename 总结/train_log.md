## 1109
- main_cdac.py
    - 这个基本和CDAC原论文第一阶段是一样的，模型直接返回损失，不过增加了早停机制以及保存评估结果最好的模型
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

    - clinc, KCL=0.25, train_batch_size=32(模型直接返回损失值)
        - 最初结果：ACC = 26.67  ARI = 11.11  NMI = 59.43  Epoch = 0
        - 最后结果：ACC = 58.8  ARI = 45.49  NMI = 80.36  Epoch = 34

