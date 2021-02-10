import mytool#导入自用工具

import sklearn
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from collections import Counter

path = r'cv_data.txt'
stopword_path = r'stopwords.txt'

data = pd.read_csv(path,engine='python',encoding='utf-8')
data = data.drop(columns=['game_id','user_id','issue_time','user_score','phone_type'])#删除不必要的列

#导入停用词表
stopwords = []

with open(stopword_path,'r',encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip('\n')#readlines会读入\n先去掉
        stopwords.append(line)


train_id_list, test_id_list = mytool.k_s_fold(data,10)#作十折

#lgb
lgb_model_list = []

tf_idf = mytool.tf_idf_method(data,stopwords)
tf_idf.tf_idf_vec()#tf_idf化
X = tf_idf.final_X
y = data['label']
X_name_list = tf_idf.feature_name

#开始训练
for each_train_id_list in train_id_list:
    use_X = X[each_train_id_list,:]
    use_y = y[each_train_id_list]
    
    temp_lgb = mytool.lgbmodel(use_X,use_y,X_name_list)
    temp_lgb.fit()#开始训练
    temp_model = temp_lgb.lgb_model
    
    lgb_model_list.append(temp_model)#将当前模型存入list
    print('ok')

#模型存储
for i,j in enumerate(lgb_model_list):
    filename = 'lgb' + str(i) +'.txt'
    j.save_model(filename)


#lstm
lstm_model_list = []

new_data, dic_id2word, dic_word2id = mytool.idlize(data)  # 构建本数据的词典

# lstm参数
vocab_size = len(dic_id2word.keys())  # 词汇表大小
n_units = 16  # lstm部分输出隐层维度
batch_size = 1024
epochs = 10
max_len = int(new_data['word_num'].quantile(q=0.90))  # 输入维度，评论词数的0.90分位数
output_dim = 32  # 嵌入层输出维度

word_X = mytool.pad_same_dim_X(max_len,new_data)
y = new_data['label']

# 开始训练
for each_train_id_list,each_test_id_list in zip(train_id_list,test_id_list):
    train_X =word_X[each_train_id_list]
    train_y = y[each_train_id_list]

    val_data = (word_X[each_test_id_list],y[each_test_id_list])

    temp_lstm = mytool.keras_lstm(train_X,train_y,vocab_size,output_dim,n_units)
    temp_lstm.fit(epochs, batch_size, val_data=val_data)
    temp_model = temp_lstm.model
    lstm_model_list.append(temp_model)  # 将当前模型存入list
    
    print('ok')

#模型存储
for i,j in enumerate(lstm_model_list):
    filename = 'lstm' + str(i) +'.h5'
    j.save(filename)