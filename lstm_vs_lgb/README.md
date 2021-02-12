# 改进

##  主要内容

1. 重新选取数据，并对数据进行十折交叉验证分割。
2. 对所得数据进行预处理，包括分词、建立词语-序号映射词典、词语-序号转换等。
3. 建立lstm神经网络情感倾向分析模型，以预测评论的情感倾向。
4. 改用lightgbm框架建立GBDT模型，将其与lstm部分代码整合，形成自用tool包。
5. 分别使用GBDT模型与lstm模型在相同的交叉验证集合上进行训练验证，横向对比结果。
6. 以训练集规模为自变量，模型在训练集和测试集上的表现为应变量，研究不同数据量对模型结果的影响。


## 步骤
1. 解压缩
[数据包1](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/cv_data.part1.rar)
与
[数据包2](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/cv_data.part2.rar)
。得到本次实验所需要用到的原始数据cv_data.txt
2. 建立自用工具
[mytool](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/mytool.py)
包，包中含有tf_idf向量化，id2word词语编码、lgb模型初始化、lstm模型初始化等功能。
3. 将解压所得数据文件、mytool包以及
[停用词表](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/stopwords.txt)
置于同一文件夹下，运行
[cv_output_model.py](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/cv_output_model.py)。
运行该文件可以分别训练lgb与lstm的十折交叉验证模型，最后在当前文件夹输出'lgbX.txt'与'lstmX.h5'字样的文件，即为每一折训练所得模型。
4. 运行
[cv_load_and_predict.ipynb](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/cv_load_and_predict.ipynb)
并查看lgb模型与lstm模型分别在训练集和验证集上的表现。
5. 运行
[lc_output_model.py](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/lc_output_model.py)。
运行该文件可以分别训练lgb与lstm的学习曲线模型，最后在当前文件夹输出'lc_lgbX.txt'与'lc_lstmX.h5'字样的文件，即为不同规模训练集下的模型。
6. 运行
[lc_load_and_predict.ipynb](https://github.com/VillardX/GBDT_game_reviews/blob/main/lstm_vs_lgb/lc_load_and_predict.ipynb)
并查看lgb模型与lstm模型在不同训练集规模下的效果。
