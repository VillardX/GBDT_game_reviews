import pandas as pd
import numpy as np
import random
import copy
from d2l import torch as d2l

import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction,AdamW, get_linear_schedule_with_warmup
from transformers import BertModel
from collections import defaultdict

import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] '
                           '- %(levelname)s: %(message)s',level=logging.WARNING)



##########基于huggingface模型进一步使用自己的数据预训练PLM##########

root_path = r'D:/conda_code/pycode/torch_learning/game_reviews/'
data_path = r'raw_data/cv_data.txt'#数据路径

#PLM模型设置
model_name = 'bert-base-chinese'
MODEL_PATH = 'D:/conda_code/pycode/torch_learning/bert_base_chinese/'
# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# b. 导入配置文件
model_config = BertConfig.from_pretrained(MODEL_PATH)
# 修改配置 - 最后一层的输出是否需要下面两项
model_config.output_hidden_states = False
model_config.output_attentions = False
# 通过配置和路径导入模型
#bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)

################NSP部分预处理##################
data = pd.read_csv(root_path+data_path,engine='python',encoding='utf-8')
data = data.drop(columns=['game_id','user_id','issue_time','user_score','phone_type','seg_result'])#删除不必要的列
data = data[data['content'].notnull()]
data['sentence_seg_list'] = data['content'].str.split('，|。|？|！')
def cal_len(temp_list):
    '''
        计算temp_list长度，temp_list可能还会说float属性，此时报错填-1
    '''
    try:
        return len(temp_list)
    except:
        return -1
data['sentence_seg_num'] = data['sentence_seg_list'].apply(lambda x: cal_len(x))
data = data[data['sentence_seg_num'] > 1]#为作NSP任务，必须掐头去尾，即sentence_seg_list长度大于1
data['temp_A'] = data['sentence_seg_list'].copy()
data['temp_B'] = data['sentence_seg_list'].copy()
data['temp_A'] = data['temp_A'].apply(lambda x:x[:-1])#去尾
data['temp_B'] = data['temp_B'].apply(lambda x:x[1:])#掐头
data['NSP'] = data.apply(lambda each_row: [(a,b) for a,b in zip(each_row['temp_A'],each_row['temp_B'])] , axis=1)

temp_NSP_list = data['NSP'].to_list()#转列表
flat_NSP_list = [each_NSP_pair for each_NSP_list in temp_NSP_list for each_NSP_pair in each_NSP_list]#展开
NSP_list = [each_NSP_pair for each_NSP_pair in flat_NSP_list if (len(each_NSP_pair[0])>0 and len(each_NSP_pair[1])>0)]#筛选长度大于0的短句

sentence_db = sorted(list(set([each_s for each_s_list in NSP_list for each_s in each_s_list])))#所有的短句语料

def _get_next_sentence(sentence, next_sentence, paragraphs):
    '''
        返回用于NSP任务的数据
        输入：
            sentence：当前句子
            next_sentence：sentence的下一句句子
            paragraphs：预料中出现过的所有句子
        输出：
            sentence：当前句子
            next_sentence：经过预处理后的接下来的句子
            is_next：next_sentence是否是sentence的下一句句子
    '''
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(paragraphs)
        is_next = False
    return [sentence, next_sentence, is_next]

NSP_res = []
for each_pair in NSP_list:
    res = _get_next_sentence(sentence=each_pair[0],next_sentence=each_pair[1],paragraphs=sentence_db)
    NSP_res.append(res)
NSP_res = list(zip(*NSP_res))#转置

#构建NSP用的pandas数据结构
NSP_dt = pd.DataFrame()
NSP_dt['sen'] = NSP_res[0]
NSP_dt['next_sen'] = NSP_res[1]
NSP_dt['is_next'] = NSP_res[2]


###########MLM部分预处理##################
def _pad_bert_inputs(pred_positions, mlm_pred_labels, max_len):
    '''
        对经过_get_mlm_data_from_tokens所得每个输入文本的MLM数据进一步作padding
        以使得尺寸进行统一，才能放入一个batch
        输入：
            pred_positions：mlm_input_ids中被mask的位置索引
            mlm_pred_labels：mlm_input_ids中被mask的位置索引的正确词语的id #事实上pred_positions和mlm_pred_labels是等长的
            max_len：在使用huggingfacePLM的tokenizer时设置的最大接受长度
        输出：
            padded_pred_positions： 经过padding后的mlm_input_ids中被mask的位置索引
            padded_mlm_pred_labels：经过padding后的mlm_input_ids中被mask的位置索引的正确词语的id
            mlm_weights：一个列表，与上述两个列表对应，原始位置全是被mask的索引位置为1，被padding的位置为0
    '''
    max_num_mlm_preds = round(max_len * 0.15) + 1 #根据max_len计算每个样本被mask的词数的最大值，作为batch中每个向量的长度
    
    #padding
    #事实上pred_positions和mlm_pred_labels是等长的
    padded_pred_positions = (pred_positions + [0] * (
        max_num_mlm_preds - len(pred_positions)))#剩下的全部填0
    
    padded_mlm_pred_labels = (mlm_pred_labels + [0] * (
        max_num_mlm_preds - len(mlm_pred_labels)))#剩下的全部填0
        
    # Predictions of padded tokens will be filtered out in the loss via
    # multiplication of 0 weights
    # padding词元的预测将通过乘以0权重在损失中过滤掉
    mlm_weights = ([1.0] * len(mlm_pred_labels) + [0.0] * (max_num_mlm_preds - len(pred_positions)))

    return padded_pred_positions, padded_mlm_pred_labels, mlm_weights


def _get_mlm_data_from_tokens(token_ids, special_id2token, special_token2id, normal_id2token):
    '''
        对单个输入文本进行MLM处理
        输入
            token_ids：经过huggingfacePLM的tokenizer处理后得到原始文本每个词的id，且已经根据max_len进行padding
            special_id2token：字典，根据PLM的tokenizer所得的特殊字符的{id:token}，tokenizer有专门的attribute可以直接引用
            special_token2id：字典，根据PLM的tokenizer所得的特殊字符的{token:id}，tokenizer有专门的attribute可以直接引用
            normal_id2token：字典，PLM的tokenizer中所有token筛去special_token所得的剩余token，{id:token}，作为MLM替换字符的集合
        输出：
            mlm_input_ids：完成mask后的token_ids
            pred_positions：mlm_input_ids中被mask的位置索引
            mlm_pred_labels：mlm_input_ids中被mask的位置索引的正确词语的id #事实上pred_positions和mlm_pred_labels是等长的
    '''
    candidate_pred_positions = []#可能会被mask的token的位置
    #token_ids = encoding_res['input_ids'][0]#默认使用pt输出时，时一个二维tensor，所以加[0]
    for i, token_id in enumerate(token_ids):
        if token_id == special_token2id['[PAD]']:
            logging.debug('第{}位开始为padding，故结束'.format(i+1))
            break
        if token_id in special_id2token.keys():
            logging.debug('第{}位开始为特殊字符{}，故略过'.format(i+1,special_id2token[token_id]))
            continue
        candidate_pred_positions.append(i)#加入备选名单
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(candidate_pred_positions) * 0.15))#所有备选名单的词中选15%个

    #使用函数替换
    mlm_input_ids, pred_positions_and_labels = _replace_mlm_tokens(
        token_ids, candidate_pred_positions, num_mlm_preds,  special_token2id, normal_id2token)

    #拆分pred_positions_and_labels
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    
    return mlm_input_ids, pred_positions, mlm_pred_labels
    
    
    
def _replace_mlm_tokens(token_ids, candidate_pred_positions, num_mlm_preds, special_token2id, normal_id2token):
    '''
        对单个输入文本进行15%->(80%,10%,10%)的MLM替换
        输入：
            token_ids：经过huggingfacePLM的tokenizer处理后得到原始文本每个词的id，且已经根据max_len进行padding
            candidate_pred_positions：当前样本可能会被mask掉的所有可能词语所在的位置，即token_ids中所有不等于special_token(如[PAD],[CLS])的id所在的索引
            num_mlm_preds：根据bert原文，应为len(candidate_pred_positions)*15%
            special_token2id：字典，根据PLM的tokenizer所得的特殊字符的{token:id}，tokenizer有专门的attribute可以直接引用
            normal_id2token：字典，PLM的tokenizer中所有token筛去special_token所得的剩余token，{id:token}，作为MLM替换字符的集合
        输出：
            mlm_input_ids：完成mask后的token_ids
            pred_positions_and_labels：一个list，每个元素为二元组，分别为被mask的token所在token_ids中的位置索引、该位置对应的正确token的id，已根据位置索引从小到大排序
    '''
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_ids = [i for i in token_ids]#token_ids.clone()#深拷贝一份
    pred_positions_and_labels = []#每个元素为二元组，分别为原input所在位置、原input所在位置的正确单词
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)#洗牌
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        #获取MASK的内容
        masked_id = None#初始化
        if random.random() < 0.8:
            masked_id = special_token2id['[MASK]']
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_id = token_ids[mlm_pred_position]#不换保持原样
            # 10% of the time: replace the word with a random word
            else:
                masked_id = random.choice(list(normal_id2token.keys()))#随机选一个，有很小的概率还是选到自己
                
        mlm_input_ids[mlm_pred_position] = masked_id
        pred_positions_and_labels.append(
            (mlm_pred_position, token_ids[mlm_pred_position]))#每个元素为二元组，分别为原input所在位置、原input所在位置的正确单词id
    
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])#关于位置进行排序

    return mlm_input_ids, pred_positions_and_labels


############Dataset设置#############
class fur_pre_bert_Dataset(Dataset):
    def __init__(self, pre_train_dt, tokenizer, max_len = 256):
        '''
            输入：
                pre_train_dt:dataframe，三列，sentence，next_sentence，is_next
        '''
        self.dt = pre_train_dt
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.token2id = tokenizer.vocab#token为键，id为值的字典
        self.id2token = dict(zip(tokenizer.vocab.values(),tokenizer.vocab.keys()))
        self.special_token2id = dict(zip(tokenizer.all_special_tokens,tokenizer.all_special_ids))#特殊字符['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.special_id2token = dict(zip(tokenizer.all_special_ids,tokenizer.all_special_tokens))
        
        self.normal_id2token = copy.deepcopy(self.id2token)#深拷贝
        [self.normal_id2token.pop(per_special_id) for per_special_id in self.special_id2token.keys()]#去除了special_id只剩下normal_id
    
    def __len__(self):
        return self.dt.shape[0]
    def __getitem__(self, idx):
        #NSP部分
        is_next_label = self.dt['is_next'][idx].tolist()
        encoding = self.tokenizer(
            self.dt['sen'][idx],self.dt['next_sen'][idx],        # 分词文本
            padding="max_length",              # padding以定义的最大长度
            max_length=self.max_len,        # 分词最大长度
            add_special_tokens=True,        # 添加特殊tokens 【cls】【sep】
            return_token_type_ids=True,    # 返回是前一句还是后一句
            return_attention_mask=True,     # 返回attention_mask
            return_tensors='pt',             # 返回pytorch类型的tensor
            truncation=True # 若大于max则切断
        )  

        #计算有效长度
        valid_lens = [(each_in_ids.shape[0]-each_in_ids.tolist().count(0)) for each_in_ids in encoding['input_ids']]#总id个数-id为0的个数即有效长度
    
        #MLM部分
        mlm_input_ids = []#初始化
        padded_pred_positions, padded_mlm_pred_labels, mlm_weights = [],[],[]#初始化
        for each_input_ids in encoding['input_ids']:
            each_input_ids = each_input_ids.tolist()
            t_mlm_input_ids, t_pred_positions, t_mlm_pred_labels = \
                    _get_mlm_data_from_tokens(token_ids=each_input_ids, 
                                                special_id2token=self.special_id2token, 
                                                special_token2id=self.special_token2id,
                                                normal_id2token=self.normal_id2token)
            #mlm进行padding
            t_padded_pred_positions, t_padded_mlm_pred_labels, t_mlm_weights = \
                            _pad_bert_inputs(t_pred_positions, 
                                            t_mlm_pred_labels, 
                                            self.max_len)
                
            mlm_input_ids.append(t_mlm_input_ids)
            padded_pred_positions.append(t_padded_pred_positions)
            padded_mlm_pred_labels.append(t_padded_mlm_pred_labels)
            mlm_weights.append(t_mlm_weights)

        return {
        'review_text': (self.dt['sen'][idx] + '<seq>'+self.dt['next_sen'][idx]),
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'token_type_ids': encoding['token_type_ids'].flatten(),
        'NSP_target': torch.tensor(is_next_label, dtype=torch.long),
        'mlm_input_ids':torch.tensor(mlm_input_ids,dtype=torch.long).flatten(),
        'padded_pred_postitions':torch.tensor(padded_pred_positions,dtype=torch.long).flatten(),
        'padded_mlm_pred_labels':torch.tensor(padded_mlm_pred_labels,dtype=torch.long).flatten(),
        'mlm_weights':torch.tensor(mlm_weights,dtype=torch.float32).flatten(),#flatten参数默认为1#padded为0.0，其他为1.0
        'valid_lens':torch.tensor(valid_lens,dtype=torch.float32).reshape(-1)
        }

def create_data_loader(dt, tokenizer, max_len, batch_size):
    ds = fur_pre_bert_Dataset(
    pre_train_dt=dt,
    tokenizer=tokenizer,
    max_len=max_len
    )
    return DataLoader(
    ds,
    shuffle=True,
    batch_size=batch_size,
    num_workers=0
    )#num_worker只有在传到gpu时用到，若想遍历，记得设置为0

BATCH_SIZE =8
MAX_LEN = 256
# train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
# val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
# test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data_loader = create_data_loader(NSP_dt.iloc[:8,:], tokenizer, MAX_LEN, BATCH_SIZE)
#data = next(iter(data_loader))#查询组后一组数据


#########模型搭建#########
class MaskLM(nn.Module):
    """
    The masked language model task of BERT.
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        '''
        num_hiddens指单层mlp中间层的单元数，num_inputs是输入的维度
        '''
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                #尺寸从(batch_size,seq_len,num_inputs)变成(batch_size,seq_len,num_hiddens)
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 #对输入的最后一维进行归一化，参数为一个整数，需要等于最后一维的维度
                                 #即对每个embedding进行归一化
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        '''
            此处传入的应该是padded_pred_positions，X为bert输出的last_hidden_state
        '''
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)#拉成一维
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)#重复batch_idx中的每个元素num_pred_positions，并按原序输出
        masked_X = X[batch_idx, pred_positions]#取出每个样本（batch）对应的每个被mask位置的embedding
        #masked_X最后得到的是一个二维矩阵，每一行代表一个mask词的embedding，共有num_masked行
        #有些pred_positions是padding出来的，所以这里会多取许多每个batch的0位置向量，但是后期计算loss的时候会用到mlm_weight的权重为0
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))#由于padding。每个样本的mlm数量是一样的，所以可以直接reshape
        #因为设置了每个batch样本的masked词数量是一样的，所以的可以再将矩阵规整
        #本身取就是按batch顺序取的，所以reshape的时候可以直接排
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """
        The next sentence prediction task of BERT.
    """
    def __init__(self, num_hiddens, num_inputs=768,**kwargs):
        '''
            num_hiddens指单层mlp中间层的单元数，num_inputs是输入的维度
        '''
        super(NextSentencePred, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                    nn.Tanh(),
                                   nn.Linear(num_hiddens, 2))

    def forward(self, X):
        '''
            此处传入的X为bert输出的pooler_output
        '''
        nsp_Y_hat = self.mlp(X)
        return nsp_Y_hat

class torch_bert(nn.Module):
    def __init__(self):#, n_classes
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_PATH, config = model_config) # bert预训练模型
        self.mlm = MaskLM(vocab_size=21128, num_hiddens=768, num_inputs=768)#mlm输入的维度
        self.nsp = NextSentencePred(num_inputs=768,num_hiddens=768)

    def forward(self,input_ids,attention_mask,token_type_ids,padded_pred_postitions):
        output = self.bert(
          input_ids,
          attention_mask,
          token_type_ids
        ) 
        mlm_Y_hat = self.mlm(X=output['last_hidden_state'],pred_positions=padded_pred_postitions)
        nsp_Y_hat = self.nsp(X=output['pooler_output'])
   
        return output,mlm_Y_hat,nsp_Y_hat

#随机初始化
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch_bert().to(device)#网络初始化
loss = nn.CrossEntropyLoss(reduction='none').to(device)#损失函数初始化
#参数reduction默认为"mean"，表示对所有样本的loss取均值，最终返回只有一个值
#参数reduction取"none"，表示保留每一个样本的loss
#参考https://blog.csdn.net/u012633319/article/details/111093144

def _get_batch_loss_bert(net, loss,
                         input_ids,attention_mask,token_type_ids,padded_pred_postitions,
                         mlm_Y,mlm_weights_X,nsp_y,
                         vocab_size=21128):
    '''
        获取MLM和NSP任务的损失
        输入：
            net：为torch_bert
            loss:nn.crossentropy()函数
            vocab_size:词典中词的个数
            input_ids,attention_mask,token_type_ids,padded_pred_postitions:即为batch_data中的各个键
            mlm_Y：对应batch_data['padded_mlm_pred_labels']
            mlm_weights_X：对应batch_data['mlm_weights']
            nsp_y:对应batch_data['NSP_target']
    '''
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(input_ids,attention_mask,token_type_ids,padded_pred_postitions)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1)#乘以各自权重决定有不有效
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)#求平均
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    nsp_l = nsp_l.mean() 
    #计算总的loss
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train_bert(train_iter, net, loss, vocab_size, device, num_steps):
    '''
        正式开始训练
        输入：
            train_iter：对应data_loader
            net：对应网络
            loss：对应loss
            vocab_size：tokenizer词典词数
            device：设备情况
            num_steps：代替epoch，直接迭代次数
    '''
    net = net.to(device)
    loss = loss.to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=2e-5)#可以设置warm_up但是没必要，这里是further_pretrain
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语⾔模型损失的和，下⼀句预测任务损失的和，句⼦对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch_dt in train_iter:
            #数据传送
            input_ids = batch_dt['input_ids'].to(device)
            attention_mask = batch_dt['attention_mask'].to(device)
            token_type_ids = batch_dt['token_type_ids'].to(device)
            padded_pred_postitions = batch_dt['padded_pred_postitions'].to(device)
            mlm_Y = batch_dt['padded_mlm_pred_labels'].to(device)
            mlm_weights_X = batch_dt['mlm_weights'].to(device)
            nsp_y = batch_dt['NSP_target'].to(device)
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net,loss,input_ids,
                                                    attention_mask,token_type_ids,padded_pred_postitions,
                                                     mlm_Y,mlm_weights_X,nsp_y,
                                                  vocab_size=vocab_size)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, input_ids.shape[0], 1)#tokens_X.shape[0]样本数，即batch_size
            timer.stop()
            animator.add(step + 1,(metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
    print(f'MLM loss {metric[0] / metric[3]:.3f}, 'f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on 'f'{str(device)}')

train_bert(data_loader, net, loss, 21128, device, 50)
