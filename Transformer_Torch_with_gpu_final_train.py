# -*- coding: utf-8 -*-
'''
  Some of the codes are learnt from
  Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612, modify by wmathor
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ast import literal_eval
import pandas as pd
from torch.autograd import Variable
from rouge import Rouge
import time
import matplotlib.pyplot as plt


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
# sentences = [
#         # enc_input           dec_input         dec_output
#         ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
#         ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
# ]

# Padding Should be Zero
# src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
# src_vocab_size = len(src_vocab)
#
# tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
# idx2word = {i: w for i, w in enumerate(tgt_vocab)}
# tgt_vocab_size = len(tgt_vocab)

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

# These parameters are not useful
src_len = 200 # enc_input max sequence length
tgt_len = 30 # dec_input(=dec_output) max sequence length
########################################################

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention



# def make_data(sentences):
#     enc_inputs, dec_inputs, dec_outputs = [], [], []
#     for i in range(len(sentences)):
#       enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
#       dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
#       dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
#
#       enc_inputs.extend(enc_input)
#       dec_inputs.extend(dec_input)
#       dec_outputs.extend(dec_output)
#
#     return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
#
# enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
#
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
#
# loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

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
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

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
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
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
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
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
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

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
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda() # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda() # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

# model = Transformer().cuda()
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
#
# for epoch in range(1000):
#     for enc_inputs, dec_inputs, dec_outputs in loader:
#       '''
#       enc_inputs: [batch_size, src_len]
#       dec_inputs: [batch_size, tgt_len]
#       dec_outputs: [batch_size, tgt_len]
#       '''
#       enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
#       # outputs: [batch_size * tgt_len, tgt_vocab_size]
#       outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#       loss = criterion(outputs, dec_outputs.view(-1))
#       print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
#
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()

def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data).cuda()
    terminal = False
    next_symbol = start_symbol
    while not terminal:         
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["<EoS>"]: # or dec_input.shape(-1)==30:
            terminal = True
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], -1)
        # print(next_word)
    return dec_input

# Test
# enc_inputs, _, _ = next(iter(loader))
# for i in range(len(enc_inputs)):
#     greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
#     predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
#     predict = predict.data.max(1, keepdim=True)[1]
#     print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
#


# ---------------------------------- training & testing ----------------------------------
def np_to_tensor(obj):
    '''
    transfer the Series in dataframe to tensor
    obj: series or dataframe

    '''
    lis = obj.values.tolist()
    ten = torch.LongTensor(lis)
    return ten

idx_data = pd.read_csv('idx_data_all.csv')
X = idx_data['Content']
y = idx_data['Summary']
tgt = idx_data['Tgt_Summary']
X = X.apply(lambda x: literal_eval(x))
y = y.apply(lambda x: literal_eval(x))
tgt = tgt.apply(lambda x: literal_eval(x))
X_train,X_val,y_train,y_val,tgt_train,tgt_val = train_test_split(X,y,tgt,test_size=0.1,random_state=12306)#split the data with random state 12306
X_val,X_test,y_val,y_test,tgt_val,tgt_test = train_test_split(X_val,y_val,tgt_val,test_size=0.5,random_state=12306)#split the data with random state 12306

# convert to dataset that can be accepted by torch
enc_inputs, dec_inputs, dec_outputs = np_to_tensor(X_train), np_to_tensor(y_train), np_to_tensor(tgt_train)
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 20, True)

dev_enc_inputs, dev_dec_inputs, dev_dec_outputs = np_to_tensor(X_val), np_to_tensor(y_val), np_to_tensor(tgt_val)
dev_loader = Data.DataLoader(MyDataSet(dev_enc_inputs, dev_dec_inputs, dev_dec_outputs),20, False)

test_enc_inputs, test_dec_inputs, test_dec_outputs = np_to_tensor(X_test), np_to_tensor(y_test), np_to_tensor(tgt_test)
test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs),20, False)


model = Transformer().cuda()
criterion = nn.CrossEntropyLoss(ignore_index = 1)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


# def init_weights(m):
#     for name, param in m.named_parameters():
#         nn.init.normal_(param.data, mean=0, std=0.01)
#
#
# model.apply(init_weights)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    # for i, batch in tqdm(enumerate(iterator)):
    #     src = batch[0].T.cuda()
    #     trg = batch[1].T.cuda()  # trg = [trg_len, batch_size]
    for src, trg, out in tqdm(iterator):
        src = src.cuda()
        trg = trg.cuda()
        out = out.cuda()

        # pred = [trg_len, batch_size, pred_dim]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(src, trg)
        loss = criterion(outputs, out.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


"""...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing."""




def evaluate(model, iterator):
    model.eval()
    epoch_ROUGE = 0
    length = 0
    with torch.no_grad():
        for src_,_,out_ in iterator:
        # output = [trg_len, batch_size, output_dim]
            for idx in range(len(src_)):
                src = src_[idx].cuda()
                out = out_[idx].cuda()
                length += 1
                predict = greedy_decoder(model, src.view(1, -1), start_symbol=tgt_vocab["<SoS>"])  # turn off teacher forcing
                predict = ' '.join([idx2word[n.item()] for n in predict.squeeze()])
                truth = ' '.join(['<SoS>'] + [idx2word[n.item()] for n in out.squeeze() if n != 1])
                ROUGE = rouge(predict, truth)[1]['r']
                epoch_ROUGE += ROUGE

    return epoch_ROUGE / length

def val_evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].cuda()
            trg = batch[1].cuda()  # trg = [trg_len, batch_size]
            out = batch[2].cuda()


            # output = [trg_len, batch_size, output_dim]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(src, trg)
            loss = criterion(outputs, out.view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


"""Finally, define a timing function."""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def rouge(a, b):
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b, avg=True)  # a和b里面包含多个句子的时候用
    rouge_score1 = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl = rouge_score["rouge-l"]

    return r1, r2, rl



best_valid_ROUGE = float('-inf')
# model.load_state_dict(torch.load('Att_Seq_last.pt'))
enc_inputs, _, target = next(iter(dev_loader))
enc_inputs = enc_inputs.cuda()
ROU_list = []
train_list = []
val_list = []
n_epochs =10
count = 0

for epoch in range(n_epochs):
    print('epoch {} starts'.format(epoch+1))
    start_time = time.time()
    count += 1

    train_loss = train(model, loader, optimizer, criterion)
    train_list.append(train_loss)
    valid_loss = val_evaluate(model, dev_loader, criterion)
    val_list.append(valid_loss)
    # pred = []
    # truth = []
    # for i in range(len(enc_inputs)):
    #     predict = greedy_decoder(model, enc_inputs[i].view(-1, 1), start_symbol=tgt_vocab["<SoS>"]).cuda()
    #     pred.append(' '.join([idx2word[n.item()] for n in predict.squeeze()]))
    #     truth.append(' '.join(['<SoS>'] + [idx2word[n.item()] for n in target[i].squeeze() if n != 1]))
    #
    # valid_ROUGE, _, _ = rouge(pred, truth)
    # valid_ROUGE = valid_ROUGE['r']
    valid_ROUGE = evaluate(model, dev_loader)
    ROU_list.append(valid_ROUGE)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    torch.save(model.state_dict(), 'Transformer_last.pt')

    if valid_ROUGE > best_valid_ROUGE:
        count = 0
        print('Best, saved!')
        best_valid_ROUGE = valid_ROUGE
        torch.save(model.state_dict(), 'Transformer_best.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. ROUGE_2: {valid_ROUGE:.3f}')

    if count == 5:
        print('early stop')
        break

# """Finally, we test the model on the test set using these "best" parameters."""

def plot_train_vs_val_precision(label_list,hyp_list,train_list,val_list):
    '''
    plots the train vs. validation precision curve
    --- arguments ---
    label_list: a list with the format [title, X_label, y_label]
    hyp_list: list of values used for hyparameter optimisation (num-of-epochs in this case)
    train_precision: the list of precision score for training set
    val_precision: the list of precision score for testing set
    '''
    plt.figure(figsize=(10, 5))
    plt.title(label_list[0],fontsize=15,y=1.04)
    plt.plot(hyp_list,train_list, "b-", linewidth=1,label='train_set_loss')
    plt.plot(hyp_list,val_list, "g-", linewidth=1,label='validation_set_loss')
    plt.xlabel(label_list[1], fontsize=12)
    plt.ylabel(label_list[2], fontsize=12)
    plt.legend(loc="lower right", fontsize=8)
    plt.show()


label_list = ['train_loss vs val_loss', 'n_epochs', 'loss']
hyp_list = [i+1 for i in range(len(train_list))]
plot_train_vs_val_precision(label_list,hyp_list,train_list,val_list)

plt.figure(figsize=(10, 5))
plt.title('ROUGE-2 Trend',fontsize=15,y=1.04)
plt.plot(hyp_list, ROU_list, "r-", linewidth=1, label='validation_ROUGE-2')
plt.xlabel(label_list[1], fontsize=12)
plt.ylabel('ROUGE recall rate', fontsize=12)
plt.legend(loc="lower right", fontsize=8)
plt.show()

# test set
model.load_state_dict(torch.load('transformer_best.pt'))
#model.load_state_dict(torch.load('train_100.pt'))
test_ROUGE = evaluate(model, test_loader)

print(f'| Test ROUGE: {test_ROUGE:.3f}')


def final_evaluate(model, iterator):
    model.eval()
    epoch_ROUGE_1p = 0
    epoch_ROUGE_2p = 0
    epoch_ROUGE_1r = 0
    epoch_ROUGE_2r = 0
    length = 0
    with torch.no_grad():
        for src_,_,out_ in iterator:
        # output = [trg_len, batch_size, output_dim]
            for idx in range(len(src_)):
                src = src_[idx].cuda()
                out = out_[idx].cuda()
                length += 1
                predict = greedy_decoder(model, src.view(1, -1), start_symbol=tgt_vocab["<SoS>"])  # turn off teacher forcing
                predict = ' '.join([idx2word[n.item()] for n in predict.squeeze()])
                truth = ' '.join(['<SoS>'] + [idx2word[n.item()] for n in out.squeeze() if n != 1])
                epoch_ROUGE_2r += rouge(predict, truth)[1]['r']
                epoch_ROUGE_2p += rouge(predict, truth)[1]['p']
                epoch_ROUGE_1r += rouge(predict, truth)[0]['r']
                epoch_ROUGE_1p += rouge(predict, truth)[0]['p']

    return epoch_ROUGE_1r / length,epoch_ROUGE_1p / length,epoch_ROUGE_2r / length,epoch_ROUGE_2p / length

model.load_state_dict(torch.load('Transformer_best.pt'))

ROUGE_1r, ROUGE_1p, ROUGE_2r, ROUGE_2p = final_evaluate(model, dev_loader)

print(f'ROUGE_1 precision: {ROUGE_1p:.3f}  ', f'ROUGE_1 recall: {ROUGE_1r:.3f}  ',
      f'ROUGE_3 precision: {ROUGE_2p:.3f}  ',f'ROUGE_1 recall: {ROUGE_2r:.3f}')





# test some of the summaries
# for i in range(len(enc_inputs)):
#     predict = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["<SoS>"]).cuda()
#     # predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
#     # predict = predict.cuda()
#     # predict = predict.data.max(1, keepdim=True)[1]
#     print('Sentence {}'.format(i+1))
#     print('Raw Document: ',' '.join([idx2cont[j] for j in enc_inputs[i].cpu().numpy() if j!=1]),
#           '\n\n Generated Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),
#           '\n\n target: ', ' '.join([idx2word[n.item()] for n in target[i].squeeze() if n!=1]), '\n')
#     # print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])




