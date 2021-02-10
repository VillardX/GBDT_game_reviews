# 改进

##  主要内容

1. 重新选取数据，并对数据进行十折交叉验证分割。
2. 对所得数据进行预处理，包括分词、建立词语-序号映射词典、词语-序号转换等。
3. 建立lstm神经网络情感倾向分析模型，以预测评论的情感倾向。
4. 改用lightgbm框架建立GBDT模型，将其与lstm部分代码整合，形成自用tool包。
5. 分别使用GBDT模型与lstm模型在相同的交叉验证集合上进行训练验证，横向对比结果。
6. 以训练集规模为自变量，模型在训练集和测试集上的表现为应变量，研究不同数据量对模型结果的影响。


## 步骤
1. 解压缩[数据包1](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/cv_data.part1.rar)与[数据包2](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/cv_data.part2.rar)。得到本次实验所需要用到的原始数据cv_data.txt
2. 建立自用工具[mytool]()包，包中含有tf_idf向量化，id2word词语编码、lgb模型初始化、lstm模型初始化等功能。
3. 将解压所得数据文件与mytool包置于同一文件夹下，运行
