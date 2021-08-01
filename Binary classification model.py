import spacy
import numpy as np
import random
import math
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ast import literal_eval
import pandas as pd
from rouge import Rouge
import matplotlib.pyplot as plt
import sklearn
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from collections import Counter
from langdetect import detect

# data cleaning
import re
from zhon.hanzi import punctuation as ch_pun
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from tqdm import tqdm
from sklearn.model_selection import train_test_split


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(1).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Clas_Encoder(nn.Module):
    def __init__(self):
        super(Clas_Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.projection = nn.Linear(d_model, 50, bias=False).cuda()
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_logits = nn.LayerNorm(32*50).cuda()(self.projection(enc_outputs).view(enc_outputs.shape[0], -1))
        return enc_logits


class Clas_Model(nn.Module):
    def __init__(self):
        super(Clas_Model, self).__init__()
        self.encoder = Clas_Encoder().cuda()
        self.tran = nn.Linear(32*50*2, 300, bias=False).cuda()
        self.projection = nn.Linear(300, 1, bias=False).cuda()

    def forward(self, enc_inputs1, enc_inputs2):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs1 = self.encoder(enc_inputs1)
        enc_outputs2 = self.encoder(enc_inputs2)
        enc_outputs = torch.cat([enc_outputs1.detach(), enc_outputs2.detach()],1)
        enc_outputs = nn.LayerNorm(300).cuda()(self.tran(enc_outputs))
        logits = nn.Sigmoid().cuda()(self.projection(enc_outputs))  # logits: [batch_size, 2]
        return logits





def blank_filling(idx_obj, tgt=False, inputs=False):
    '''
    fill the sentences with <Blank> to ensure sentence length consistency
    idx_obj: list of tokens that are transformed into indices

    '''
    obj = idx_obj.copy()
    max_len = max([len(i) for i in obj])
    print('Maximum length is {}'.format(max_len))
    for idx in tqdm(range(len(obj))):
        if inputs == True:
            obj[idx] = obj[idx][:-1]
        current_len = len(obj[idx])
        obj[idx] = obj[idx] + [1] * (32 - current_len)
        if tgt == True:
            obj[idx] = obj[idx][1:] + [1]
    print('Current length is {}'.format(len(obj[idx])))
    return obj


def np_to_tensor(obj):
    '''
    transfer the Series in dataframe to tensor
    obj: list of input

    '''
    ten = torch.LongTensor(obj)
    return ten


# convert to dataset that can be accepted by torch
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs

  def __len__(self):
    return self.enc_inputs.shape[0]

  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]



def class_clean(tgt1,tgt2, labels):
    '''
    tokenise the input clause and transform the tokens into indices
    :param content: a list of clauses to be summarised
    :param src_vocab: dictionary of source document
    :param tgt: a list of target summaries
    :param tgt_vocab: dictionary of target summary document
    :param idx: True if the input is the cleaned version in indices
    :return: iterator of content and summary token lists
    '''
    tgt1 = blank_filling(tgt1, tgt = True)
    tgt2 = blank_filling(tgt2, tgt=True)
    loader = Data.DataLoader(MyDataSet(np_to_tensor(tgt1), np_to_tensor(tgt2), np_to_tensor(labels)),
                             20, False)
    return loader



def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    # for i, batch in tqdm(enumerate(iterator)):
    #     src = batch[0].T.cuda()
    #     trg = batch[1].T.cuda()  # trg = [trg_len, batch_size]
    for input1, input2, label in tqdm(iterator):
        input1 = input1.cuda()
        input2 = input2.cuda()
        label = torch.FloatTensor(label.numpy()).cuda()

        # pred = [trg_len, batch_size, pred_dim]
        outputs = model(input1, input2)
        loss = criterion(outputs.view(-1), label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def val_evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for input1, input2, label in iterator:
            input1 = input1.cuda()
            input2 = input2.cuda()
            label = torch.FloatTensor(label.numpy()).cuda()

            # pred = [trg_len, batch_size, pred_dim]
            outputs = model(input1, input2)
            loss = criterion(outputs.view(-1), label.view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def predict(model, iterator, criterion):
    model.eval()
    same = 0
    total = 0
    with torch.no_grad():
        for input1, input2, label in iterator:
            input1 = input1.cuda()
            input2 = input2.cuda()
            label = label.view(-1).tolist()

            # pred = [trg_len, batch_size, pred_dim]
            outputs = model(input1, input2)
            outputs = outputs.view(-1).cpu().numpy()
            outputs = [0 if i <= 0.5 else 1 for i in outputs]
            for i,j in zip(label,outputs):
                total += 1
                if i == j:
                    same += 1

    return same / total





def main():
    best_loss = float('-inf')
    # model.load_state_dict(torch.load('Att_Seq_last.pt'))
    train_list = []
    val_list = []
    n_epochs =200
    count = 0

    for epoch in range(n_epochs):
        print('epoch {} starts'.format(epoch+1))
        start_time = time.time()
        count += 1

        train_loss = train(model, loader, optimizer, criterion)
        train_list.append(train_loss)
        valid_loss = predict(model, test_loader, criterion)
        val_list.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        torch.save(model.state_dict(), 'clas_last.pt')

        if valid_loss > best_loss:
            count = 0
            print('Best, saved!')
            best_loss = valid_loss
            torch.save(model.state_dict(), 'clas_best.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. ROUGE_2: {valid_loss:.3f}')

        if count == 5:
            print('early stop')
            break



if __name__ == "__main__":
    root_dir = './'
    file_name = 'cont_w2i_all.json'
    str_file = root_dir + file_name
    with open(str_file, 'r') as f:
        print("Load str file from {}".format(str_file))
        src_vocab = json.load(f)
    idx2cont = {i: w for i, w in enumerate(src_vocab)}
    src_vocab_size = len(src_vocab)

    file_name = 'sum_w2i_all.json'
    str_file = root_dir + file_name
    with open(str_file, 'r') as f:
        print("Load str file from {}".format(str_file))
        tgt_vocab = json.load(f)
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    src_vocab_size = len(tgt_vocab)

    src_len = 188  # enc_input max sequence length
    tgt_len = 26  # dec_input(=dec_output) max sequence length

    # Transformer Parameters
    d_model = 50  # Embedding Size
    d_ff = 50  # FeedForward dimension
    d_k = d_v =50  # dimension of K(=Q), V
    n_layers = 2  # number of Encoder of Decoder Layer
    n_heads = 5  # number of heads in Multi-Head Attention

    idx_data = pd.read_csv('classification_task_cleaned.csv')
    X = idx_data['trans']  # [:200]
    y = idx_data['seq']  # [:200]
    tgt = idx_data['labels']  # [:200]
    X = X.apply(lambda x: literal_eval(x))
    y = y.apply(lambda x: literal_eval(x))
    X_train, X_val, y_train, y_val, tgt_train, tgt_val = train_test_split(X, y, tgt, test_size=0.2,
                                                                          random_state=12306)  # split the data with random state 12306

    # convert to dataset that can be accepted by torch
    enc_inputs1 = X_train.tolist()
    enc_inputs2 = y_train.tolist()
    tgt = tgt_train.tolist()
    model = Clas_Model().cuda()
    criterion = torch.nn.BCELoss()
#nn.CrossEntropyLoss(ignore_index=1)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    loader = class_clean(enc_inputs1,enc_inputs2,tgt)
    enc_inputs1_ = X_val.tolist()
    enc_inputs2_ = y_val.tolist()
    tgt_ = tgt_val.tolist()
    test_loader = class_clean(enc_inputs1_,enc_inputs2_,tgt_)
    # model.load_state_dict(torch.load('transformer_best.pt'))
    main()
    model.eval()
    with torch.no_grad():
        for input1, input2, label in test_loader:
            input1 = input1.cuda()
            input2 = input2.cuda()
            label = label.cuda()

            # pred = [trg_len, batch_size, pred_dim]
            outputs = model(input1, input2)
            print(outputs)
            print(label.view(-1))












