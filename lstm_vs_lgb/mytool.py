import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences#序列长度统一化要用到的工具
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb



from collections import Counter
from pprint import pprint

import warnings

warnings.filterwarnings('ignore')



def k_s_fold(data, k, y_name='label'):
    '''
        对原始数据进行分层kfold，即保证每个子集Y取各值的比例和原始数据集相接近
        输入：
            data:原始数据，为dataframe
            k：要将原始数据平分的子集个数
            y_name：标签列列名,为字符串
        输出：
            train_index_list:[train1_id_list,...,traink_id_list]
            test_index_list:[test1_id_list,...,testk_id_list]
    '''
    # 定位X与y
    y = data[y_name]
    X = data[[col for col in data.columns if col != y_name]]

    SKF = StratifiedKFold(n_splits=k,random_state=None, shuffle=False)  # , 
    SKF_spli = SKF.split(X, y)
    train_index_list = []
    test_index_list = []
    for train_index, test_index in SKF_spli:
        train_index_list.append(train_index)
        test_index_list.append(test_index)
    return train_index_list, test_index_list


def idlize(data, seg_name='seg_result'):
    '''
        将分词结果id化
        输入
            seg_name：需要将所分词列表化的列列名，该列下的每一元素形同'只能 说 游戏 做 的 确实 不怎么 如 人意 ， 慢慢 调整 吧',以空格分开
        输出
            data:添加了word_num列的更新data
            dic_id2word:序号到词的映射词典
            dic_word2id:词到序号的映射词典
    '''

    # 构建词典对应表
    seg = data[seg_name]
    seg_spli = seg.str.split(' ')
    seg_list = seg_spli.to_list()

    flatten_seg_list = [each_word for each_sentence_list in seg_list for each_word in each_sentence_list]  # 展开成一个list
    word_list = set(flatten_seg_list)  # 去重
    word_list = list(word_list)#set是无序的,每次运行都会有不一样的顺序
    word_list.sort()#用list的sort功能排序,这样每次就能得到相同的结果了

    # 构建词典，并将出现的词填入词典
    dic_id2word = {i + 1: j for i, j in enumerate(word_list)}

    dic_word2id = dict(zip(dic_id2word.values(), dic_id2word.keys()))  # 数为值单词为键

    # 构建每条评论的分词id映射
    seg_id_list = []
    for each_seg in seg_list:
        temp = []
        for each_word in each_seg:
            temp.append(dic_word2id[each_word])
        seg_id_list.append(temp)

    # 将各条评论id映射加入元数据
    data['seg_id_list'] = seg_id_list
    data['seg_word_list'] = seg_spli

    data['word_num'] = data['seg_id_list'].apply(lambda x: len(x))  # 每条评论的词的个数，对应content_len字数

    return data, dic_id2word, dic_word2id

def pad_same_dim_X(max_len,data,seg_idlist_name='seg_id_list'):
    '''
        将各个样本所分词的idlist化成等长
        输入：
            max_len：所要取的等长长度，每条评论保留词的最大个数
            data：输入原始数据，为dataframe
            seg_idlist_name：分割为idlist的列名
        返回：
            word_id_X：等长化的idlist
    '''
    word_id_X = pad_sequences(maxlen=max_len, sequences=data[seg_idlist_name], padding='post',
                                    value=0) #value用于填充序列，padding补在头pre还是补在尾post
    return word_id_X


class tf_idf_method:
    def __init__(self, data, stp_list, seg_col_name='seg_result',
                 other_x_name=['fun', 'up', 'down', 'play_time', 'phone_code', 'content_len']):
        '''
            初始化
                data:原始数据，为dataframe
                seg_col_name:分词结果所在列列名，为str
                stp_list:停止词词典，为list
                other_x_name：非文字特征，为list
                feature_name:特征名称，初始化为非文字特征other_x_name，后面通过tf_idf_vec操作会加上文字特征，为list
        '''
        self.data = data
        self.seg_col_name = seg_col_name
        self.stop_word = stp_list
        self.other_x_name = other_x_name  # 非文字特征
        self.feature_name = other_x_name
        self.final_X = None  # 最终处理好的X数据


    def tf_idf_vec(self, max_f=5000):
        """
            对分词结果进行tf-idf化
            输入：
                max_f = 最大维数，默认5000维
            输出：
                self.final_data：最终处理好的数据，其他特征在前、文字特征在后的sparse matrix

        """
        corpus = self.data[self.seg_col_name]  # 需要tf-idf向量化的列

        vectorizer = TfidfVectorizer(stop_words=self.stop_word, max_features=max_f)  # 初始化模型,max_features先定5000维，后续可调
        seg_X = vectorizer.fit_transform(corpus)  # 得到的tf-idf向量，是个sparse matrix

        self.feature_name = self.feature_name + vectorizer.get_feature_names()  # 将文字特征加入特征名

        other_x_value = self.data.loc[:, self.other_x_name].values  # 转为np.array
        other_x_sparse = sp.csr_matrix(other_x_value)  # 离散化

        # 非文本特征与文本特征合并，注意先后顺序
        total_X = sp.hstack([other_x_sparse, seg_X]).tocsr()  # 转为csr矩阵，后期交叉验证，多会做行切片

        self.final_X = total_X


class lgbmodel():
    def __init__(self, X, y, X_name_list):
        '''
            初始化
                X：为特征，sparse_matrix
                y：标签，pd.series
                X_name_list:各特征名，为list
        '''
        # 特征名字典
        self.feature_dict = {X_name_list[i]: ('x' + str(i + 1)) for i in
                             range(len(X_name_list))}  # lgb不支持中文特征，所以以xi作为代号，形如{'x1':'fun'...}
        # 数据初始化
        self.train = lgb.Dataset(X, label=y, feature_name=self.feature_dict.values(),
                                 categorical_feature=[self.feature_dict['phone_code']])  # 手机型号是分类属性
        #         self.train_X = lgb.Dataset(X)
        #         self.train_y = lgb.Dataset(y)
        #         self.train = lgb.Dataset(self.train_X, label=self.train_y, feature_name=self.feature_dict.values(), categorical_feature=[self.feature_dict['phone_code']])

        # 模型初始化
        self.params = {'max_depth': 10, 'min_data_in_leaf': 5,
                       'learning_rate': 0.1, 'num_leaves': 1024, 'metric': ['binary_logloss', 'binary_error'],
                       'objective': 'binary', 'nthread': 4, 'verbose': -1, 'feature_fraction': 0.8,
                       'feature_fraction_seed': 1}  # 参数'num_leaves': 35, 'lambda_l1': 0.1, 'lambda_l2': 0.2,
        self.lgb_model = None  # 模型

    def fit(self, num_boost=1000):
        '''
            训练
        '''
        self.lgb_model = lgb.train(self.params, self.train, num_boost, verbose_eval=100)  # verbose_eval迭代多少次打印

class keras_lstm():
    def __init__(self, X, y, vocab_size, embed_out_dim, num_n_units):
        '''
            初始化数据和模型
            X:用于训练的数据，每个样本为一个wordid_list,整体是一个2维array
            y:对应标签
            vocab_size：词汇量个数
            embed_out_dim：嵌入层输出维度
            num_n_units：单一层的输出output size(hidden size)
        '''

#         self.y = to_categorical(data[y_name])  # 改为binary类别
#         self.y = data[y_name]
        self.label_size = 1  # 共有几列
        self.max_len = X.shape[1]#有几列，说明每条评论保留了几个词
        
        self.word_X = X
        self.y = y
        
        # 创建深度学习模型， Embedding + LSTM + Softmax.
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size + 1, output_dim=embed_out_dim,
                                 input_length=self.max_len, mask_zero=True))  # mask=True，说明id为0是没有任何意义
        self.model.add(LSTM(num_n_units, input_shape=(self.word_X.shape[0], self.word_X.shape[1])))
#         self.model.add(Dropout(0.1))
        self.model.add(Dense(self.label_size, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, epochs=10, batch_size=256,val_data=None):
        '''
            开始训练
            参数可选
        '''
        self.model.fit(self.word_X, self.y, epochs=epochs, batch_size=batch_size,validation_data = val_data, verbose=1)
# vocab_size = len(b.keys()) # 词汇表大小
# n_units = 100#？
# batch_size = 1024
# epochs = 20
# max_len = int(a['word_num'].quantile(q=0.80))#输入维度，评论词数的0.95分位数
# output_dim = 64#嵌入层输出维度

